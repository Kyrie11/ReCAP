from __future__ import annotations

from collections import Counter
import hashlib
import json
from time import perf_counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ocrap.data.serialization import write_json
from ocrap.evaluation.metrics import (
    best_shared_option_index,
    deployable_recovery_success,
    false_recoverability_admission,
    nominal_utility_preservation,
    post_contact_deployability_score,
    summarize_selection_metrics,
)
from ocrap.external_baselines.data import (
    group_sample_paths, load_external_sample, _branch_arrays, _topology_arrays,
    _history_arrays, _history_scene_arrays, _history_frame, _prefix_traj_array, _source_scene_arrays, _topology_scene_context,
    _actor_topology_arrays, _map_topology_arrays, use_teacher_branch_context,
)
from ocrap.external_baselines.models import build_model_from_cfg
from ocrap.external_baselines.runtime import configure_cuda_runtime, resolve_amp_dtype
from ocrap.external_baselines.observed_risk import build_observed_risk_context, observed_risk_profile, observed_risk_profiles, observed_risk_profiles_and_context
from ocrap.external_baselines.policies import ExternalSelection, select_external_policy
from ocrap.models.data import samples_to_feature_matrix


def _scalar(d: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(np.asarray(d.get(key, default)).item())
    except Exception:
        return float(default)


def _load_checkpoint(checkpoint: str | Path | None, cfg: dict[str, Any]) -> tuple[torch.nn.Module | None, dict[str, Any], torch.device]:
    if not checkpoint:
        return None, cfg, torch.device("cpu")
    path = Path(checkpoint)
    if not path.exists():
        return None, cfg, torch.device("cpu")
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    ckpt_cfg = ckpt.get("cfg", cfg) or cfg
    merged = dict(ckpt_cfg)
    # Runtime evaluation knobs may override policy thresholds and method lists,
    # but the checkpoint owns model geometry (d_model/layers/heads/max tokens).
    if isinstance(cfg.get("external_baselines", {}), dict):
        eb = dict(merged.get("external_baselines", {}) or {})
        rt = cfg.get("external_baselines", {}) or {}
        for key in ("methods", "policy", "baseline"):
            if key in rt:
                if key == "policy" and isinstance(rt.get(key), dict) and isinstance(eb.get(key), dict):
                    tmp = dict(eb.get(key) or {})
                    tmp.update(rt.get(key) or {})
                    eb[key] = tmp
                else:
                    eb[key] = rt[key]
        # max_candidates is safe to increase/decrease for padding at inference,
        # but not for reconstructing the learned positional parameter.  Keep the
        # checkpoint value when model_state is loaded.
        merged["external_baselines"] = eb
    eb = merged.setdefault("external_baselines", {})
    mcfg = eb.setdefault("model", {})
    if "max_candidates" in ckpt:
        eb["max_candidates"] = int(ckpt.get("max_candidates"))
        mcfg["max_candidates"] = int(ckpt.get("max_candidates"))
    for ck, mk in [
        ("num_roots", "num_roots"), ("num_options", "num_options"), ("root_feature_dim", "root_feature_dim"),
        ("num_topology_agents", "num_topology_agents"), ("topology_feature_dim", "topology_feature_dim"),
        ("actor_topology_feature_dim", "actor_topology_feature_dim"), ("num_topology_map", "num_topology_map"),
        ("map_topology_feature_dim", "map_topology_feature_dim"), ("history_len", "history_len"),
        ("neighbors_to_predict", "neighbors_to_predict"), ("future_len", "future_len"),
    ]:
        if ck in ckpt:
            mcfg[mk] = int(ckpt[ck])
    device_req = str(((merged.get("external_baselines", {}) or {}).get("training", {}) or {}).get("device", (merged.get("training", {}) or {}).get("device", "auto")))
    device = torch.device("cuda" if device_req == "auto" and torch.cuda.is_available() else ("cpu" if device_req == "auto" else device_req))
    training_cfg = ((merged.get("external_baselines", {}) or {}).get("training", {}) or {})
    configure_cuda_runtime(training_cfg, device, log=False)
    model = build_model_from_cfg(int(ckpt["input_dim"]), merged).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, merged, device


def _predict_group(model: torch.nn.Module | None, samples: list[dict[str, Any]], cfg: dict[str, Any], device: torch.device) -> dict[str, np.ndarray] | None:
    if model is None or not samples:
        return None
    bcfg = (cfg.get("external_baselines", {}) or {})
    mcfg = (bcfg.get("model", {}) or {})
    arch = str(mcfg.get("arch", bcfg.get("baseline", ""))).lower()
    need_history = arch in {"gameformer", "gameformer_lite", "gameformer_levelk", "plantf", "plan_tf", "plantf_adapter", "pluto", "pluto_adapter"}
    need_prefix_traj = need_history or arch in {"route_bc_wayformer", "wayformer_bc", "wayformer_scene_bc"}
    implementation = str(mcfg.get("implementation", bcfg.get("implementation", ""))).lower()
    need_source_scene = (arch in {"route_bc_wayformer", "wayformer_bc", "wayformer_scene_bc"}) or (implementation in {"source_port", "source_port_v54", "sourceported_v54"} and arch in {"gameformer", "gameformer_lite", "gameformer_levelk", "plantf", "plan_tf", "plantf_adapter", "pluto", "pluto_adapter"})
    # Source ports consume candidate-independent actor/map tensors, not the old
    # handcrafted candidate-specific PlanTF/PLUTO topology proxy.
    need_topology = arch in {"betop", "betop_lite", "betopnet", "betopnet_lite"} or (not need_source_scene and arch in {"plantf", "plan_tf", "plantf_adapter", "pluto", "pluto_adapter"})
    use_branch_context = use_teacher_branch_context(cfg)

    max_candidates = int(bcfg.get("max_candidates", len(samples)))
    n = min(len(samples), max_candidates)
    feats = samples_to_feature_matrix(samples[:n], cfg, shared_scene=True)
    if feats.size == 0:
        return None
    D = int(feats.shape[1])
    x = np.zeros((1, max_candidates, D), dtype=np.float32)
    mask = np.zeros((1, max_candidates), dtype=bool)
    x[0, :n] = feats[:n]
    mask[0, :n] = True

    kwargs: dict[str, torch.Tensor | None] = {}
    if use_branch_context:
        bm0, rf0, _, _, _ = _branch_arrays(samples[0], cfg)
        K, L, Fdim = int(bm0.shape[0]), int(bm0.shape[1]), int(rf0.shape[-1])
        branch_margins = np.zeros((1, max_candidates, K, L), dtype=np.float32)
        root_features = np.zeros((1, max_candidates, K, Fdim), dtype=np.float32)
        root_probs = np.zeros((1, max_candidates, K), dtype=np.float32)
        root_valid = np.zeros((1, max_candidates, K), dtype=bool)
        option_valid = np.zeros((1, max_candidates, L), dtype=bool)
        for i, d in enumerate(samples[:n]):
            bm, rf, rp, rv, ov = _branch_arrays(d, cfg)
            branch_margins[0, i], root_features[0, i] = bm, rf
            root_probs[0, i], root_valid[0, i], option_valid[0, i] = rp, rv, ov
        kwargs.update({
            "branch_margins": torch.from_numpy(branch_margins).to(device, non_blocking=True),
            "root_features": torch.from_numpy(root_features).to(device, non_blocking=True),
            "root_probs": torch.from_numpy(root_probs).to(device, non_blocking=True),
            "root_valid": torch.from_numpy(root_valid).to(device, non_blocking=True),
            "option_valid": torch.from_numpy(option_valid).to(device, non_blocking=True),
        })

    if need_history:
        ego0, neigh0, nv0, origin0, _yaw0, rot0 = _history_scene_arrays(samples[0], cfg)
        pref0, _ = _prefix_traj_array(samples[0], cfg, origin0, rot0)
        H, A_hist, T = int(ego0.shape[0]), int(neigh0.shape[0]), int(pref0.shape[0])
        ego_history = np.zeros((1, max_candidates, H, 9), dtype=np.float32)
        neighbor_history = np.zeros((1, max_candidates, A_hist, H, 9), dtype=np.float32)
        neighbor_valid = np.zeros((1, max_candidates, A_hist, H), dtype=bool)
        prefix_traj = np.zeros((1, max_candidates, T, 2), dtype=np.float32)
        prefix_valid = np.zeros((1, max_candidates, T), dtype=bool)
        if n:
            ego_history[0, :n] = ego0[None, ...]
            neighbor_history[0, :n] = neigh0[None, ...]
            neighbor_valid[0, :n] = nv0[None, ...]
        for i, d in enumerate(samples[:n]):
            pt, pv = _prefix_traj_array(d, cfg, origin0, rot0)
            prefix_traj[0, i], prefix_valid[0, i] = pt, pv
        kwargs.update({
            "ego_history": torch.from_numpy(ego_history).to(device, non_blocking=True),
            "neighbor_history": torch.from_numpy(neighbor_history).to(device, non_blocking=True),
            "neighbor_valid": torch.from_numpy(neighbor_valid).to(device, non_blocking=True),
            "prefix_traj": torch.from_numpy(prefix_traj).to(device, non_blocking=True),
            "prefix_valid": torch.from_numpy(prefix_valid).to(device, non_blocking=True),
        })
    elif need_prefix_traj:
        _hist0, _valid0, _ego0, origin0, _yaw0, rot0 = _history_frame(samples[0])
        pref0, _ = _prefix_traj_array(samples[0], cfg, origin0, rot0)
        T = int(pref0.shape[0])
        prefix_traj = np.zeros((1, max_candidates, T, 2), dtype=np.float32)
        prefix_valid = np.zeros((1, max_candidates, T), dtype=bool)
        for i, d in enumerate(samples[:n]):
            pt, pv = _prefix_traj_array(d, cfg, origin0, rot0)
            prefix_traj[0, i], prefix_valid[0, i] = pt, pv
        kwargs.update({
            "prefix_traj": torch.from_numpy(prefix_traj).to(device, non_blocking=True),
            "prefix_valid": torch.from_numpy(prefix_valid).to(device, non_blocking=True),
        })

    if need_source_scene:
        sa, sav, sc, smp, smpv, smeta, smc, smask, scl = _source_scene_arrays(samples[0], cfg)
        kwargs.update({
            "source_agent_history": torch.from_numpy(sa[None]).to(device, non_blocking=True),
            "source_agent_valid": torch.from_numpy(sav[None]).to(device, non_blocking=True),
            "source_current_state": torch.from_numpy(sc[None]).to(device, non_blocking=True),
            "source_map_points": torch.from_numpy(smp[None]).to(device, non_blocking=True),
            "source_map_point_valid": torch.from_numpy(smpv[None]).to(device, non_blocking=True),
            "source_map_meta": torch.from_numpy(smeta[None]).to(device, non_blocking=True),
            "source_map_center": torch.from_numpy(smc[None]).to(device, non_blocking=True),
            "source_map_valid": torch.from_numpy(smask[None]).to(device, non_blocking=True),
            "source_centerline": torch.from_numpy(scl[None]).to(device, non_blocking=True),
        })

    if need_topology:
        topology_scene = _topology_scene_context(samples[0], cfg)
        actor0, _, _ = _actor_topology_arrays(samples[0], cfg, topology_scene)
        map0, _, _ = _map_topology_arrays(samples[0], cfg, topology_scene)
        A_top, AF = int(actor0.shape[0]), int(actor0.shape[-1])
        M_top, MF = int(map0.shape[0]), int(map0.shape[-1])
        actor_topology_features = np.zeros((1, max_candidates, A_top, AF), dtype=np.float32)
        actor_topology_mask = np.zeros((1, max_candidates, A_top), dtype=bool)
        map_topology_features = np.zeros((1, max_candidates, M_top, MF), dtype=np.float32)
        map_topology_mask = np.zeros((1, max_candidates, M_top), dtype=bool)
        for i, d in enumerate(samples[:n]):
            af, _, am = _actor_topology_arrays(d, cfg, topology_scene)
            mf, _, mm = _map_topology_arrays(d, cfg, topology_scene)
            actor_topology_features[0, i], actor_topology_mask[0, i] = af, am
            map_topology_features[0, i], map_topology_mask[0, i] = mf, mm
        kwargs.update({
            "actor_topology_features": torch.from_numpy(actor_topology_features).to(device, non_blocking=True),
            "actor_topology_mask": torch.from_numpy(actor_topology_mask).to(device, non_blocking=True),
            "map_topology_features": torch.from_numpy(map_topology_features).to(device, non_blocking=True),
            "map_topology_mask": torch.from_numpy(map_topology_mask).to(device, non_blocking=True),
        })

    training_cfg = ((cfg.get("external_baselines", {}) or {}).get("training", {}) or {})
    amp_enabled = bool(training_cfg.get("amp", True)) and device.type == "cuda"
    amp_dtype = resolve_amp_dtype(training_cfg, device)
    with torch.inference_mode():
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            out = model(
                torch.from_numpy(x).to(device, non_blocking=True),
                torch.from_numpy(mask).to(device, non_blocking=True),
                **kwargs,
            )
    result: dict[str, np.ndarray] = {}
    for k, v in out.items():
        if isinstance(v, list):
            continue
        result[k] = v.squeeze(0).detach().float().cpu().numpy()[:n]
    return result

def _yaw_rate_violation_proxy(d: dict[str, Any], yaw_rate_max: float = 0.6) -> float:
    """Return whether the candidate exceeds the configured yaw-rate limit.

    OC-RAP's ego prefix schema is [x,y,vx,vy,heading,yaw_rate,speed,length,width].
    Prefer the stored yaw-rate channel; fall back to a heading derivative only
    for legacy samples without that channel.
    """
    states = np.asarray(d.get("prefix_states", np.zeros((0, 0))), dtype=float)
    if states.ndim != 2 or states.shape[0] == 0:
        return 0.0
    if states.shape[1] >= 6:
        rate = np.abs(np.nan_to_num(states[:, 5], nan=0.0))
    elif states.shape[1] >= 5 and states.shape[0] >= 2:
        heading = np.unwrap(states[:, 4])
        dt = float(((d.get("dt", 0.1) if not isinstance(d.get("dt", 0.1), np.ndarray) else np.asarray(d.get("dt", 0.1)).item())))
        rate = np.abs(np.diff(heading)) / max(dt, 1e-3)
    else:
        return 0.0
    return float(np.max(rate) > float(yaw_rate_max)) if rate.size else 0.0


def _contact_extra_metrics(records: list[dict[str, Any]]) -> dict[str, float | None]:
    if not records:
        return {}
    out: dict[str, float | None] = {}
    out["secondary_collision_rate"] = float(np.mean([r.get("selected_hard_violation", 0.0) > 0.0 or r.get("selected_harm_proxy", 0.0) > 0.05 for r in records]))
    out["stable_stop_success"] = float(np.mean([r.get("drs", 0.0) >= 0.5 and r.get("selected_hard_violation", 0.0) <= 0.0 for r in records]))
    out["max_yaw_rate_violation"] = float(np.mean([r.get("yaw_rate_violation_proxy", 0.0) > 0.0 for r in records]))
    out["route_rejoin_success"] = float(np.mean([r.get("nup", 0.0) >= 0.5 and r.get("selected_harm_proxy", 0.0) <= 0.5 for r in records]))
    out["mean_harm_proxy"] = float(np.mean([r.get("selected_harm_proxy", 0.0) for r in records]))
    out["post_contact_deployability_score"] = float(np.mean([r.get("post_contact_deployability", 0.0) for r in records]))
    return out


def _record_for_selection(method: str, samples: list[dict[str, Any]], sel: ExternalSelection, cfg: dict[str, Any], *, observed_profiles: list[Any] | None = None) -> dict[str, Any]:
    idx = int(np.clip(sel.selected_index, 0, max(len(samples) - 1, 0)))
    chosen = samples[idx]
    utility = np.asarray([_scalar(d, "utility", 0.0) for d in samples], dtype=float)
    r_dep = np.asarray([_scalar(d, "r_dep_star", 0.0) for d in samples], dtype=float)
    r_orc = np.asarray([_scalar(d, "r_orc_star", 0.0) for d in samples], dtype=float)
    odg = float(np.asarray(chosen.get("oracle_gap_star", r_orc[idx] - r_dep[idx])).item()) if samples else 0.0
    selected_option = sel.selected_option
    if selected_option is None:
        selected_option = best_shared_option_index(chosen.get("m_star", np.zeros((0, 0))), chosen.get("root_probs", np.zeros((0,))), gamma=0.0, root_valid=chosen.get("root_valid", None), option_valid=chosen.get("option_valid", None))
    drs = deployable_recovery_success(chosen.get("m_star", np.zeros((0, 0))), chosen.get("root_probs", np.zeros((0,))), int(selected_option), chosen.get("root_valid", None))
    nup = nominal_utility_preservation(utility[0] if utility.size else 0.0, utility[idx] if utility.size else 0.0, sigma_u=float((cfg.get("metrics", {}) or {}).get("sigma_u", 1.0)))
    observed = observed_profiles[idx] if observed_profiles is not None and len(observed_profiles) > idx else observed_risk_profile(chosen, cfg)
    return {
        "method": method,
        "fra_cand": false_recoverability_admission(sel.admitted, r_dep),
        "fra_exec": float(r_dep[idx] < 0.0) if r_dep.size else 0.0,
        "drs": float(drs),
        "odg": float(odg),
        "post_contact_deployability": float(post_contact_deployability_score(float(drs), float(r_dep[idx]) if r_dep.size else 0.0, float(odg))),
        "nup": float(nup["bounded_NUP"]),
        "artifact": bool(_scalar(chosen, "i_art_star", 0.0) > 0.5),
        "selected_artifact": bool(_scalar(chosen, "i_art_star", 0.0) > 0.5),
        "selection_reason": sel.reason,
        "selected_index": idx,
        "selected_option": int(selected_option),
        "selected_utility": float(utility[idx]) if utility.size else 0.0,
        "selected_teacher_r_dep": float(r_dep[idx]) if r_dep.size else 0.0,
        "selected_teacher_r_orc": float(r_orc[idx]) if r_orc.size else 0.0,
        "selected_admitted": bool(sel.admitted[idx]) if len(sel.admitted) > idx else False,
        "num_admitted": int(np.asarray(sel.admitted, dtype=bool).sum()),
        "num_admitted_interventions": int(np.asarray(sel.admitted, dtype=bool)[1:].sum()) if len(sel.admitted) > 1 else 0,
        "selected_hard_violation": _scalar(chosen, "hard_violation", 0.0),
        "selected_harm_proxy": _scalar(chosen, "harm_proxy", 0.0),
        "selected_observed_expected_risk": float(observed.expected_loss),
        "selected_observed_cvar_risk": float(observed.cvar_loss),
        "selected_observed_worst_risk": float(observed.worst_loss),
        "selected_observed_collision_probability": float(observed.collision_probability),
        "selected_observed_min_clearance_m": float(observed.min_clearance),
        "selected_observed_min_ttc_s": float(observed.min_ttc),
        "selected_observed_severity_proxy": float(observed.severity_proxy),
        "selected_observed_backup_margin_m": float(observed.backup_margin),
        "observed_calibration_target_teacher_hard_violation": float(_scalar(chosen, "hard_violation", 0.0) > 0.0),
        "yaw_rate_violation_proxy": _yaw_rate_violation_proxy(chosen, yaw_rate_max=float(cfg.get("yaw_rate_max_rps", 0.6))),
    }



def _binary_calibration(records: list[dict[str, Any]], *, bins: int = 10) -> dict[str, float]:
    if not records:
        return {"observed_collision_brier_vs_teacher_hard_violation": float("nan"), "observed_collision_ece_vs_teacher_hard_violation": float("nan")}
    p = np.clip(np.asarray([r.get("selected_observed_collision_probability", np.nan) for r in records], dtype=float), 0.0, 1.0)
    y = np.asarray([r.get("observed_calibration_target_teacher_hard_violation", np.nan) for r in records], dtype=float)
    ok = np.isfinite(p) & np.isfinite(y)
    if not ok.any():
        return {"observed_collision_brier_vs_teacher_hard_violation": float("nan"), "observed_collision_ece_vs_teacher_hard_violation": float("nan")}
    p, y = p[ok], y[ok]
    brier = float(np.mean((p - y) ** 2))
    ece = 0.0
    edges = np.linspace(0.0, 1.0, max(int(bins), 1) + 1)
    for i in range(len(edges) - 1):
        m = (p >= edges[i]) & (p < edges[i + 1] if i + 1 < len(edges) - 1 else p <= edges[i + 1])
        if m.any():
            ece += float(m.mean()) * abs(float(p[m].mean()) - float(y[m].mean()))
    return {"observed_collision_brier_vs_teacher_hard_violation": brier, "observed_collision_ece_vs_teacher_hard_violation": float(ece)}


def _finite_summary(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=float)
    finite = a[np.isfinite(a)]
    return {
        "mean": float(np.mean(finite)) if finite.size else float("nan"),
        "p05": float(np.percentile(finite, 5.0)) if finite.size else float("nan"),
        "finite_rate": float(finite.size / max(a.size, 1)),
    }


def _summarize(records: list[dict[str, Any]], method: str, num_groups: int, source: str) -> dict[str, Any]:
    result = summarize_selection_metrics(records)
    if records:
        result.update({
            "intervention_rate": float(np.mean([int(r.get("selected_index", 0)) != 0 for r in records])),
            "selected_admitted_rate": float(np.mean([bool(r.get("selected_admitted", False)) for r in records])),
            "mean_num_admitted": float(np.mean([float(r.get("num_admitted", 0.0)) for r in records])),
            "mean_num_admitted_interventions": float(np.mean([float(r.get("num_admitted_interventions", 0.0)) for r in records])),
            "mean_selected_teacher_R_dep": float(np.mean([r.get("selected_teacher_r_dep", 0.0) for r in records])),
            "mean_selected_teacher_R_orc": float(np.mean([r.get("selected_teacher_r_orc", 0.0) for r in records])),
            "mean_selected_utility": float(np.mean([r.get("selected_utility", 0.0) for r in records])),
            "mean_selected_observed_expected_risk": float(np.mean([r.get("selected_observed_expected_risk", 0.0) for r in records])),
            "mean_selected_observed_cvar_risk": float(np.mean([r.get("selected_observed_cvar_risk", 0.0) for r in records])),
            "mean_selected_observed_worst_risk": float(np.mean([r.get("selected_observed_worst_risk", 0.0) for r in records])),
            "mean_selected_observed_collision_probability": float(np.mean([r.get("selected_observed_collision_probability", 0.0) for r in records])),
            "mean_selected_observed_min_clearance_m": float(np.mean([r.get("selected_observed_min_clearance_m", 0.0) for r in records])),
            "mean_selected_observed_backup_margin_m": float(np.mean([r.get("selected_observed_backup_margin_m", 0.0) for r in records])),
            "mean_selected_observed_severity_proxy": float(np.mean([r.get("selected_observed_severity_proxy", 0.0) for r in records])),
            "mean_selection_time_ms": float(np.mean([r.get("selection_time_ms", 0.0) for r in records])),
            "p95_selection_time_ms": float(np.percentile([r.get("selection_time_ms", 0.0) for r in records], 95.0)),
            "selection_reason_counts": dict(Counter(str(r.get("selection_reason", "")) for r in records)),
        })
        ttc = _finite_summary([r.get("selected_observed_min_ttc_s", float("inf")) for r in records])
        result.update({"mean_selected_observed_min_ttc_s": ttc["mean"], "p05_selected_observed_min_ttc_s": ttc["p05"], "finite_selected_observed_ttc_rate": ttc["finite_rate"]})
        result.update(_binary_calibration(records, bins=10))
        result.update(_contact_extra_metrics(records))
    result.update({"method": method, "num_scene_time_groups": int(num_groups), "num_records": int(len(records)), "source": source})
    return result


def evaluate_external_baselines(
    dataset: str,
    output: str,
    cfg: dict[str, Any],
    *,
    split: str = "test",
    checkpoint: str | None = None,
    baselines: str | list[str] | None = None,
) -> dict[str, Any]:
    bcfg = cfg.setdefault("external_baselines", {})
    if baselines is None:
        baselines = bcfg.get("methods", [bcfg.get("baseline", "route_bc_lite")])
    if isinstance(baselines, str):
        methods = [m.strip() for m in baselines.split(",") if m.strip()]
    else:
        methods = [str(m).strip() for m in baselines if str(m).strip()]
    if not methods:
        raise ValueError("No external baselines requested")
    model, model_cfg, device = _load_checkpoint(checkpoint, cfg)
    groups = group_sample_paths(dataset, split=split)
    records_by_method: dict[str, list[dict[str, Any]]] = {m: [] for m in methods}
    timing = {"load_s": 0.0, "model_inference_s": 0.0, "observed_risk_s": 0.0, "selection_s_by_method": {m: 0.0 for m in methods}}
    eval_start = perf_counter()
    for gi, paths in enumerate(groups, 1):
        tick = perf_counter()
        samples = [load_external_sample(p) for p in paths]
        samples = sorted(samples, key=lambda d: int(np.asarray(d.get("candidate_index", 0)).item()))
        timing["load_s"] += perf_counter() - tick
        tick = perf_counter()
        model_outputs = _predict_group(model, samples, model_cfg, device)
        timing["model_inference_s"] += perf_counter() - tick
        # All deployable hand-designed methods share exactly the same observation-
        # conditioned risk profiles. Compute them once per candidate group rather
        # than once per method and once again for the selected record.
        oracle_names = {"oracle_filter", "oracle_recovery_filter", "branchwise_oracle_filter", "oracle_branchwise_recovery"}
        learned_names = {"route_bc", "route_bc_lite", "waymax_bc", "waymax_bc_lite", "wayformer_bc", "wayformer_style_bc", "route_bc_wayformer", "gameformer", "gameformer_lite", "gameformer_levelk", "betop", "betop_lite", "betopnet", "betopnet_lite", "plantf", "plan_tf", "plantf_adapter", "pluto", "pluto_adapter"}
        pure_learned = learned_names
        context_only_names = {
            "dr_cvar_safety_filter", "distributionally_robust_cvar_filter", "safaoui_dr_cvar_filter",
            "conformal_predictive_safety_filter", "conformal_safety_filter", "cpsf",
        }
        predictor_free_paper_names = {
            "postimpact_mpc", "postimpact_mpc_lite", "post_impact_mpc_lite", "postimpact_mpc_paper", "integrated_postimpact_mpc",
            "postimpact_motion_tvlqr", "postimpact_motion_planning", "wang2022_postimpact", "postimpact_tvlqr",
        }
        simple_names = oracle_names | pure_learned | predictor_free_paper_names
        need_profiles = any(m.lower() not in simple_names | context_only_names for m in methods)
        need_context = need_profiles or any(m.lower() in context_only_names for m in methods)
        tick = perf_counter()
        if need_profiles:
            profiles, risk_context = observed_risk_profiles_and_context(samples, model_cfg)
        elif need_context:
            profiles, risk_context = None, build_observed_risk_context(samples[0], model_cfg)
        else:
            profiles, risk_context = None, None
        timing["observed_risk_s"] += perf_counter() - tick
        for method in methods:
            ml = method.lower()
            use_profiles = profiles if ml not in simple_names | context_only_names else None
            use_context = risk_context if ml in context_only_names or (profiles is not None and ml not in simple_names) else None
            tick = perf_counter()
            sel = select_external_policy(
                method, samples, model_cfg, model_outputs=model_outputs,
                precomputed_profiles=use_profiles, precomputed_context=use_context,
            )
            selection_s = perf_counter() - tick
            timing["selection_s_by_method"][method] += selection_s
            record = _record_for_selection(method, samples, sel, model_cfg, observed_profiles=profiles)
            record["selection_time_ms"] = 1000.0 * selection_s
            records_by_method[method].append(record)
        if gi == 1 or gi % 500 == 0:
            print({"event": "external_eval_progress", "groups_done": gi, "num_groups": len(groups)}, flush=True)
    learned_methods = {"route_bc", "route_bc_lite", "waymax_bc", "waymax_bc_lite", "wayformer_bc", "wayformer_style_bc", "route_bc_wayformer", "gameformer", "gameformer_lite", "gameformer_levelk", "betop", "betop_lite", "betopnet", "betopnet_lite", "plantf", "plan_tf", "plantf_adapter", "pluto", "pluto_adapter"}
    oracle_methods = {"oracle_filter", "oracle_recovery_filter", "branchwise_oracle_filter", "oracle_branchwise_recovery"}
    summaries = {}
    for m in methods:
        ml = m.lower()
        if model is not None and ml in learned_methods:
            source = "learned_checkpoint_observation_only_inputs"
        elif ml in oracle_methods:
            source = "teacher_only_oracle_upper_bound"
        else:
            source = "observation_only_rule_or_optimizer"
        summaries[m] = _summarize(records_by_method[m], m, len(groups), source)
    timing["total_s"] = perf_counter() - eval_start
    timing["groups_per_second"] = float(len(groups) / max(timing["total_s"], 1e-9))
    timing["candidates_per_second"] = float(sum(len(x) for x in groups) / max(timing["total_s"], 1e-9))
    requested_cfg_raw = json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    result = {
        "artifact_status": "complete",
        "dataset": str(dataset),
        "split": split,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "requested_config_fingerprint": hashlib.sha256(requested_cfg_raw.encode("utf-8")).hexdigest(),
        "method_order": methods,
        "methods": summaries,
        "timing": timing,
        "metric_protocol": {"observed_collision_calibration_target": "teacher hard_violation used only after selection; diagnostic, not a deployable policy input", "min_ttc_definition": "first time mode-wise geometric clearance reaches risk_ttc_clearance_threshold_m"},
    }
    if output:
        write_json(result, output)
    return result
