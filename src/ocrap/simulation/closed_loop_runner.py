from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib
import json
import os
import re
from time import perf_counter

import numpy as np

from ocrap.data.build.builder import build_feature_only_samples_for_history, build_labeled_samples_for_candidate_indices, build_samples_for_history
from ocrap.data.build.history import construct_history
from ocrap.data.schema import pad_recovery_params
from ocrap.data.serialization import write_json
from ocrap.data.waymax_loader import iter_waymax_womd_scenarios, raw_scenario_from_waymax_state
from ocrap.evaluation.baselines import select_baseline
from ocrap.evaluation.metrics import best_option_indices, deployable_recovery_success, false_recoverability_admission, nominal_utility_preservation, option_execution_semantics, post_contact_deployability_score, predicted_option_success
from ocrap.models.data import iter_sample_paths_many, scalar_metadata_for_path
from ocrap.models.inference import ModelBundle, load_model_bundle, predict_sample, predict_samples, teacher_prediction_from_sample
from ocrap.external_baselines.policies import select_external_policy
from ocrap.external_baselines.evaluate import _load_checkpoint as _load_external_checkpoint, _predict_group as _predict_external_group
from ocrap.simulation.waymax_rollout import _as_np, _bicycle_action, _make_env, _metric_summary, _sdc_index
from ocrap.planning.route_lattice import project_to_route
from ocrap.utils.geometry import compute_ttc, min_box_clearance, rotation_matrix
from ocrap.utils.regimes import bucket_aliases as canonical_bucket_aliases, canonical_regime_name, is_post_contact_bucket


EXTERNAL_CLOSED_LOOP_METHODS = {
    "marc", "marc_lite", "marc_contingency",
    "robust_scenario_mpc", "scenario_mpc", "batkovic_scenario_mpc",
    "dr_cvar_safety_filter", "distributionally_robust_cvar_filter", "safaoui_dr_cvar_filter",
    "conformal_predictive_safety_filter", "conformal_safety_filter", "cpsf",
    "pdm_closed", "pdm_closed_adapter", "idm", "idm_planner",
    "racp", "racp_lite", "risk_aware_contingency",
    "expected_risk", "expected_risk_filter", "expected_risk_planner",
    "cvar_risk", "cvar_risk_filter", "cvar_planner",
    "dro_cvar", "dro_cvar_filter", "dro_cvar_safety_filter", "dr_cvar_filter",
    "predictive_safety_filter", "psf", "cbf_backup_filter", "predictive_cbf_backup", "backup_cbf_filter",
    "oracle_filter", "oracle_recovery_filter", "branchwise_oracle_filter", "oracle_branchwise_recovery",
    "postimpact_mpc", "postimpact_mpc_lite", "post_impact_mpc_lite", "postimpact_mpc_paper", "integrated_postimpact_mpc",
    "post_crash_braking", "post_crash_braking_rule", "stable_stop", "stable_stop_rule", "postcrash_stable_stop",
    "postimpact_motion_tvlqr", "postimpact_motion_planning", "wang2022_postimpact", "postimpact_tvlqr",
    "post_collision_restoration", "trajectory_restoration", "post_collision_trajectory_restoration", "post_collision_restoration_heuristic", "ackermann_restoration",
    "severity_minimization", "severity_minimization_planner", "unavoidable_collision_planner", "crash_mitigation_planner", "uc_severity_planner",
    "compensatory_postimpact_mpc", "cao_postimpact_mpc",
    "robust_postimpact_control", "postimpact_sliding_mode", "ao_postimpact_control",
    "gameformer", "gameformer_lite", "gameformer_levelk",
    "route_bc", "route_bc_lite", "waymax_bc", "waymax_bc_lite", "wayformer_bc", "wayformer_style_bc", "route_bc_wayformer",
    "betop", "betop_lite", "betopnet", "betopnet_lite",
    "plantf", "plan_tf", "plantf_adapter",
    "pluto", "pluto_adapter", "pdm_hybrid", "pdm_hybrid_adapter",
}
# Only the deliberately non-deployable oracle upper bound consumes OC-RAP
# counterfactual teacher tensors during action selection.
EXTERNAL_TEACHER_REQUIRED_METHODS = {
    "oracle_filter", "oracle_recovery_filter", "branchwise_oracle_filter", "oracle_branchwise_recovery",
}
EXTERNAL_LEARNED_METHODS = {
    "gameformer", "gameformer_lite", "gameformer_levelk",
    "route_bc", "route_bc_lite", "waymax_bc", "waymax_bc_lite", "wayformer_bc", "wayformer_style_bc", "route_bc_wayformer",
    "betop", "betop_lite", "betopnet", "betopnet_lite",
    "plantf", "plan_tf", "plantf_adapter",
    "pluto", "pluto_adapter",
}



@dataclass
class ClosedLoopDecision:
    scene_id: str
    step_index: int
    time_index: int
    method: str
    selected_index: int
    selected_macro: str
    selected_candidate_index: int
    selection_reason: str
    selected_utility: float
    selected_teacher_r_dep: float | None
    selected_teacher_r_orc: float | None
    selected_pred_r_dep: float | None
    selected_pred_r_orc: float | None
    selected_pred_gap: float | None
    selected_pred_drs: float | None
    selected_direct_recovery_value: float | None
    selected_direct_recovery_std: float | None
    selected_direct_recovery_opportunity: float | None
    nominal_direct_recovery_value: float | None
    direct_recovery_advantage: float | None
    selected_nominal_deviation: float | None
    selected_odg: float | None
    selected_post_contact_deployability: float | None
    selected_artifact: bool | None
    audit_candidate_count: int | None
    audit_best_candidate_index: int | None
    audit_best_macro: str | None
    audit_best_teacher_r_dep: float | None
    audit_best_drs: float | None
    audit_best_pred_r_dep: float | None
    audit_best_pred_gap: float | None
    audit_best_pred_drs: float | None
    audit_best_hard: float | None
    audit_best_harm: float | None
    audit_selected_r_dep_regret: float | None
    audit_has_recoverable_candidate: bool | None
    audit_selector_miss: bool | None
    audit_best_pcd_candidate_index: int | None
    audit_best_pcd_macro: str | None
    audit_best_pcd: float | None
    audit_best_pcd_drs: float | None
    audit_best_pcd_teacher_r_dep: float | None
    audit_best_pcd_pred_r_dep: float | None
    audit_best_pcd_pred_gap: float | None
    audit_best_pcd_pred_drs: float | None
    audit_selected_pcd_regret: float | None
    audit_pcd_selector_miss: bool | None
    audit_paper_best_pcd_macro: str | None
    audit_paper_best_pcd: float | None
    audit_paper_best_pcd_drs: float | None
    audit_paper_best_pcd_teacher_r_dep: float | None
    audit_paper_selected_pcd_regret: float | None
    audit_paper_pcd_selector_miss: bool | None
    fra_exec: float | None
    fra_cand: float | None
    drs: float | None
    nup: float
    metrics_after_step: dict[str, float]


def _canonical_womd_scene_id(value: Any) -> str:
    """Normalize official or legacy OC-RAP scene ids.

    The ``__wx########`` suffix is a source-order provenance field, not scene
    identity.  Official WOMD ``scenario/id`` is authoritative; pre-v48.28
    ``waymax_<hash>`` ids remain migration aliases only.
    """
    text = str(value or "").strip()
    return re.sub(r"__wx\d{8}$", "", text)


def _legacy_waymax_source_index(value: Any) -> int | None:
    match = re.search(r"__wx(\d{8})$", str(value or "").strip())
    return int(match.group(1)) if match else None


def _source_role_from_pattern(value: Any) -> str:
    text = str(value or "").lower()
    if "validation_interactive" in text:
        return "validation_interactive"
    if re.search(r"(^|[/_])validation([/_]|$)", text):
        return "validation"
    if "training" in text:
        return "training"
    if "testing" in text or "/test" in text:
        return "test"
    return "unknown"


def _raw_scene_identity_keys(raw: Any) -> set[str]:
    metadata = getattr(raw, "metadata", {}) or {}
    values = [
        getattr(raw, "scenario_id", ""),
        metadata.get("original_scenario_id"),
        metadata.get("official_scenario_id"),
        metadata.get("legacy_scenario_id"),
    ]
    return {key for key in (_canonical_womd_scene_id(v) for v in values) if key}


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(np.asarray(x).reshape(-1)[0])
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _safe_optional_float(x: Any) -> float | None:
    try:
        v = float(np.asarray(x).reshape(-1)[0])
        return v if np.isfinite(v) else None
    except Exception:
        return None


def _prefix_nominal_deviation(samples: list) -> np.ndarray:
    if not samples:
        return np.zeros((0,), dtype=np.float32)
    try:
        ref = np.asarray(samples[0].prefix.prefix_states, dtype=float)[:, :2]
    except Exception:
        return np.zeros((len(samples),), dtype=np.float32)
    vals: list[float] = []
    for sample in samples:
        try:
            xy = np.asarray(sample.prefix.prefix_states, dtype=float)[:, :2]
            T = min(len(ref), len(xy))
            vals.append(0.0 if T <= 0 else float(np.sqrt(np.mean(np.sum((xy[:T] - ref[:T]) ** 2, axis=-1))) / 5.0))
        except Exception:
            vals.append(0.0)
    return np.asarray(vals, dtype=np.float32)




def _external_label_budget_for_method(method: str, cfg: dict, n: int) -> int:
    """Budget for exact teacher-label materialization in external closed-loop runs.

    External rule/optimization baselines still use their paper-defined
    branch/root/contact scores; the speedup is to evaluate those scores on a
    small macro-diverse candidate lattice instead of rebuilding full OC-RAP
    labels for every generated prefix at every replan.  Set
    closed_loop.exhaustive_teacher_labels=true or external_label_max_candidates<=0
    to recover the previous exhaustive path.
    """
    if n <= 0:
        return 0
    cl_cfg = cfg.get("closed_loop", {}) if isinstance(cfg.get("closed_loop", {}), dict) else {}
    raw = cl_cfg.get("external_label_max_candidates", None)
    if raw is None:
        # Near-contact filters mainly need nominal plus one representative per
        # semantic macro. Contact planners benefit from a slightly wider lattice
        # because brake/stabilize/restore/severity candidates can trade off.
        ml = str(method).lower()
        if ml in {
            "postimpact_mpc", "postimpact_mpc_lite", "post_impact_mpc_lite",
            "postimpact_mpc_paper", "integrated_postimpact_mpc",
            "post_crash_braking", "post_crash_braking_rule", "stable_stop",
            "stable_stop_rule", "postcrash_stable_stop",
            "post_collision_restoration", "trajectory_restoration",
            "post_collision_trajectory_restoration", "post_collision_restoration_heuristic",
            "ackermann_restoration", "severity_minimization",
            "severity_minimization_planner", "unavoidable_collision_planner",
            "crash_mitigation_planner", "uc_severity_planner",
        }:
            raw = 10
        else:
            raw = 8
    try:
        budget = int(raw)
    except Exception:
        budget = 8
    if budget <= 0:
        return n
    return int(max(1, min(n, budget)))


def _sample_macro_name(sample: Any) -> str:
    try:
        return str(getattr(sample.prefix, "macro_name", "") or "").strip().lower()
    except Exception:
        return ""


def _prefix_deviation_from_nominal(samples: list) -> np.ndarray:
    if not samples:
        return np.zeros((0,), dtype=np.float32)
    try:
        ref = np.asarray(samples[0].prefix.prefix_states, dtype=float)[:, :2]
    except Exception:
        return np.zeros((len(samples),), dtype=np.float32)
    vals: list[float] = []
    for s in samples:
        try:
            xy = np.asarray(s.prefix.prefix_states, dtype=float)[:, :2]
            T = min(ref.shape[0], xy.shape[0])
            vals.append(0.0 if T <= 0 else float(np.sqrt(np.mean(np.sum((xy[:T] - ref[:T]) ** 2, axis=-1))) / 5.0))
        except Exception:
            vals.append(0.0)
    return np.asarray(vals, dtype=np.float32)


def _contact_macro_priority(method: str) -> list[str]:
    ml = str(method).lower()
    if ml in {"post_crash_braking", "post_crash_braking_rule", "stable_stop", "stable_stop_rule", "postcrash_stable_stop"}:
        return ["nominal", "brake", "stabilize", "yield", "pull_over"]
    if ml in {"post_collision_restoration", "trajectory_restoration", "post_collision_trajectory_restoration", "post_collision_restoration_heuristic", "ackermann_restoration"}:
        return ["nominal", "stabilize", "lane_shift", "merge", "yield", "pull_over", "keep"]
    if ml in {"severity_minimization", "severity_minimization_planner", "unavoidable_collision_planner", "crash_mitigation_planner", "uc_severity_planner"}:
        return ["nominal", "brake", "stabilize", "yield", "lane_shift", "merge", "pull_over"]
    if ml in {"postimpact_mpc", "postimpact_mpc_lite", "post_impact_mpc_lite", "postimpact_mpc_paper", "integrated_postimpact_mpc"}:
        return ["nominal", "brake", "stabilize", "yield", "lane_shift", "merge", "pull_over"]
    return ["nominal", "brake", "yield", "stabilize", "merge", "lane_shift", "pull_over", "keep"]


def _preselect_external_label_candidate_indices(samples: list, method: str, cfg: dict) -> list[int]:
    """Macro-diverse candidate subset for exact teacher labels.

    The expensive part of near/contact closed-loop external baselines is not the
    MARC/RACP/CVaR/MPC logic itself; it is materializing full Waymax/OC-MERO
    labels for every prefix.  This helper keeps the exact same downstream policy
    functions but chooses a small candidate lattice to label: nominal anchor,
    paper-relevant macro families, then high-utility/low-deviation backups.
    """
    n = len(samples)
    if n <= 0:
        return []
    cl_cfg = cfg.get("closed_loop", {}) if isinstance(cfg.get("closed_loop", {}), dict) else {}
    budget = _external_label_budget_for_method(method, cfg, n)
    if budget >= n or bool(cl_cfg.get("exhaustive_teacher_labels", False)):
        return [int(getattr(s, "candidate_index", i)) for i, s in enumerate(samples)]

    selected: list[int] = []
    selected_pos: set[int] = set()

    def add_pos(pos: int | None) -> None:
        if pos is None or pos < 0 or pos >= n or pos in selected_pos:
            return
        try:
            cid = int(getattr(samples[pos], "candidate_index", pos))
        except Exception:
            cid = int(pos)
        selected.append(cid)
        selected_pos.add(pos)

    # Always include nominal/log replay anchor.
    nominal_pos = next((i for i, s in enumerate(samples) if int(getattr(s, "candidate_index", i)) == 0 or _sample_macro_name(s) == "nominal"), 0)
    add_pos(int(nominal_pos))

    macros = [_sample_macro_name(s) for s in samples]
    utility = np.asarray([_safe_float(getattr(s.prefix, "utility", 0.0)) for s in samples], dtype=float)
    hard = np.asarray([_safe_float(getattr(s.prefix, "hard_violation", 0.0)) for s in samples], dtype=float)
    harm = np.asarray([_safe_float(getattr(s.prefix, "harm_proxy", 0.0)) for s in samples], dtype=float)
    dev = _prefix_deviation_from_nominal(samples).astype(float)
    pcfg = ((cfg.get("external_baselines", {}) or {}).get("policy", {}) or {})
    geom_score = utility - float(pcfg.get("label_preselect_hard_weight", 4.0)) * np.maximum(0.0, hard) - float(pcfg.get("label_preselect_harm_weight", 0.75)) * np.maximum(0.0, harm) - float(pcfg.get("label_preselect_deviation_weight", 0.20)) * dev

    # Add one best candidate from each paper-relevant macro family.
    for macro in _contact_macro_priority(method):
        if len(selected) >= budget:
            break
        idxs = [i for i, m in enumerate(macros) if m == macro]
        if not idxs:
            continue
        best = max(idxs, key=lambda i: float(geom_score[i]))
        add_pos(int(best))

    # Preserve diversity for any remaining generated macro not listed above.
    if bool(cl_cfg.get("external_label_macro_diversity", True)):
        for macro in sorted(set(macros)):
            if len(selected) >= budget:
                break
            if not macro or any(macros[i] == macro for i in selected_pos):
                continue
            idxs = [i for i, m in enumerate(macros) if m == macro]
            if idxs:
                add_pos(int(max(idxs, key=lambda i: float(geom_score[i]))))

    # Fill remaining budget by score.
    order = np.argsort(np.where(np.isfinite(geom_score), geom_score, -np.inf))[::-1]
    for pos in order.tolist():
        if len(selected) >= budget:
            break
        add_pos(int(pos))
    return selected[:budget]



def _candidate_lookup(samples: list) -> dict[int, int]:
    out: dict[int, int] = {}
    for i, sample in enumerate(samples):
        try:
            out[int(sample.candidate_index)] = int(i)
        except Exception:
            continue
    return out


def _select_audit_candidate_indices(samples: list, info: dict[str, Any], selected_sample, cfg: dict) -> list[int]:
    """Choose a small, diagnostic candidate set for online label audit.

    The selected-only audit answers whether the executed action was deployably
    recoverable, but not whether OC-RAP missed a better candidate.  This helper
    adds a few high-value alternatives without falling back to expensive
    all-candidate teacher labeling.
    """
    if not samples:
        return []
    cl_cfg = cfg.get("closed_loop", {}) if isinstance(cfg.get("closed_loop", {}), dict) else {}
    k = max(1, int(cl_cfg.get("audit_top_k", 4) or 4))
    max_extra = max(0, int(cl_cfg.get("audit_max_extra_candidates", k) or k))
    selected_cid = int(getattr(selected_sample, "candidate_index", 0))
    out: list[int] = []

    def add(cid: int | None) -> None:
        if cid is None:
            return
        try:
            c = int(cid)
        except Exception:
            return
        if c not in out:
            out.append(c)

    add(selected_cid)
    add(0)  # nominal anchor
    lookup = _candidate_lookup(samples)
    utility = np.asarray(info.get("utility", []), dtype=float).reshape(-1)
    pred_r = np.asarray(info.get("pred_r_dep", []), dtype=float).reshape(-1)
    pred_gap = np.maximum(0.0, np.asarray(info.get("pred_gap", np.zeros_like(pred_r)), dtype=float).reshape(-1))
    dev = np.maximum(0.0, np.asarray(info.get("nominal_deviation", np.zeros_like(pred_r)), dtype=float).reshape(-1))
    sel_cfg = cfg.get("selection", {}) if isinstance(cfg.get("selection", {}), dict) else {}
    beta = float(sel_cfg.get("lcb_beta", 0.10))
    rec_lcb = pred_r - beta * pred_gap

    def add_ranked(score: np.ndarray, largest: bool = True, count: int = 1) -> None:
        if score.size == 0:
            return
        vals = np.asarray(score, dtype=float)
        vals = np.where(np.isfinite(vals), vals, -np.inf if largest else np.inf)
        order = np.argsort(vals)
        if largest:
            order = order[::-1]
        added = 0
        for idx in order.tolist():
            if idx < 0 or idx >= len(samples):
                continue
            add(int(getattr(samples[idx], "candidate_index", idx)))
            added += 1
            if added >= count:
                break

    add_ranked(rec_lcb, largest=True, count=k)
    add_ranked(pred_r, largest=True, count=max(1, k // 2))
    add_ranked(utility - float(sel_cfg.get("deviation_penalty", 0.15)) * dev, largest=True, count=max(1, k // 2))
    add_ranked(pred_gap, largest=False, count=1)

    # Keep macro diversity among audited alternatives, which is helpful for
    # diagnosing whether failures come from candidate coverage or selection.
    seen_macro = {str(getattr(samples[lookup[c]], "prefix", None).macro_name) for c in out if c in lookup and getattr(samples[lookup[c]], "prefix", None) is not None}
    for sample in samples:
        if len(out) >= 1 + max_extra:
            break
        macro = str(getattr(sample.prefix, "macro_name", ""))
        if macro and macro not in seen_macro:
            add(int(sample.candidate_index))
            seen_macro.add(macro)
    return out[: 1 + max_extra]


def _prediction_q_for_candidate(samples: list, info: dict[str, Any], candidate_index: int, selected_fallback_idx: int) -> np.ndarray | int:
    lookup = _candidate_lookup(samples)
    idx = lookup.get(int(candidate_index), int(selected_fallback_idx))
    try:
        return info["items"][idx]["pred"].q
    except Exception:
        return 0

def _mean_finite(values: list[Any], default: float | None = None) -> float | None:
    vals: list[float] = []
    for x in values:
        if x is None:
            continue
        try:
            v = float(x)
        except Exception:
            continue
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return default
    return float(np.mean(vals))



def _load_json_mapping(path: str | Path) -> dict[str, float]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text())
    except Exception:
        return {}
    if isinstance(raw, dict) and isinstance(raw.get("gamma_rec_by_bucket"), dict):
        raw = raw["gamma_rec_by_bucket"]
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in raw.items():
        try:
            fv = float(v)
            if np.isfinite(fv):
                out[str(k)] = fv
        except Exception:
            continue
    return out


def _apply_gamma_rec_by_bucket_file(cfg: dict) -> dict:
    sel = cfg.get("selection", {}) if isinstance(cfg.get("selection", {}), dict) else {}
    local = dict(cfg)
    new_sel = dict(sel)
    changed = False

    # Backward-compatible scalar recovery threshold map.
    path = sel.get("gamma_rec_by_bucket_file", sel.get("gamma_rec_by_bucket_path", None))
    if path:
        mapping = _load_json_mapping(path)
        if mapping:
            existing = new_sel.get("gamma_rec_by_bucket", {})
            merged = dict(existing) if isinstance(existing, dict) else {}
            merged.update(mapping)
            new_sel["gamma_rec_by_bucket"] = merged
            changed = True

    # v15: allow calibrated auxiliary selector maps to be supplied as JSON files
    # using the same bucket-name format as gamma_rec_by_bucket_file.
    for file_key, target_key in [
        ("option_drs_certificate_threshold_by_bucket_file", "option_drs_certificate_threshold_by_bucket"),
        ("option_drs_certificate_max_gap_by_bucket_file", "option_drs_certificate_max_gap_by_bucket"),
        ("option_drs_certificate_rec_slack_by_bucket_file", "option_drs_certificate_rec_slack_by_bucket"),
    ]:
        path = sel.get(file_key, None)
        if not path:
            continue
        mapping = _load_json_mapping(path)
        if not mapping:
            continue
        existing = new_sel.get(target_key, {})
        merged = dict(existing) if isinstance(existing, dict) else {}
        merged.update(mapping)
        new_sel[target_key] = merged
        changed = True

    if changed:
        local["selection"] = new_sel
        return local
    return cfg
def _current_timestep(state: Any) -> int:
    try:
        return int(_as_np(state.timestep).reshape(()).item())
    except Exception:
        return 0


def _state_geometry_metrics(state: Any, sdc: int) -> dict[str, float]:
    """Current-step physical margins used to interpret near/contact behavior.

    The paper defines near-contact using clearance/TTC, but earlier closed-loop
    reports exposed only generic Waymax metrics.  These fields make the regime
    claim observable at execution time without changing the planner inputs.
    """
    try:
        tr = state.sim_trajectory
        t = _current_timestep(state)
        x = _as_np(tr.x)[:, t]
        y = _as_np(tr.y)[:, t]
        vx = _as_np(tr.vel_x)[:, t]
        vy = _as_np(tr.vel_y)[:, t]
        yaw = _as_np(tr.yaw)[:, t]
        length = _as_np(tr.length)[:, t]
        width = _as_np(tr.width)[:, t]
        height = _as_np(tr.height)[:, t]
        valid = _as_np(tr.valid)[:, t].astype(bool)
        if sdc < 0 or sdc >= len(x) or not bool(valid[sdc]):
            return {}
        ego_box = np.asarray([x[sdc], y[sdc], vx[sdc], vy[sdc], yaw[sdc], length[sdc], width[sdc], height[sdc], 1.0], dtype=np.float32)
        keep = np.arange(len(x)) != int(sdc)
        boxes = np.stack([x, y, vx, vy, yaw, length, width, height, np.ones_like(x)], axis=-1).astype(np.float32)[keep]
        other_valid = valid[keep]
        ego_state = np.zeros((16,), dtype=np.float32)
        ego_state[0], ego_state[1] = float(x[sdc]), float(y[sdc])
        ego_state[3], ego_state[4] = float(vx[sdc]), float(vy[sdc])
        ego_state[7] = float(yaw[sdc])
        ego_state[10], ego_state[11], ego_state[12] = float(length[sdc]), float(width[sdc]), float(height[sdc])
        return {
            "min_clearance_m": float(min_box_clearance(ego_box, boxes, other_valid)),
            "ttc_s": float(compute_ttc(ego_state, boxes, other_valid)),
            "ego_speed_mps": float(np.hypot(vx[sdc], vy[sdc])),
            "ego_yaw_rad": float(yaw[sdc]),
        }
    except Exception:
        return {}


def _capture_render_context(
    state: Any,
    sdc: int,
    *,
    radius_m: float = 80.0,
    max_polylines: int = 192,
    max_points_per_polyline: int = 64,
) -> dict[str, Any]:
    """Capture static nearby roadgraph geometry once per rendered scene.

    Full benchmark runs keep rendering disabled.  For the selected ten video
    targets, storing compact roadgraph polylines makes escape space and
    off-road behavior interpretable without serializing the same map at every
    simulator step.  Missing/older Waymax roadgraph fields are tolerated.
    """
    try:
        rg = getattr(state, "roadgraph_points", None)
        if rg is None:
            return {}
        x = _as_np(rg.x).reshape(-1)
        y = _as_np(rg.y).reshape(-1)
        valid = _as_np(rg.valid).reshape(-1).astype(bool)
        ids_raw = getattr(rg, "ids", None)
        types_raw = getattr(rg, "types", None)
        ids = _as_np(ids_raw).reshape(-1) if ids_raw is not None else np.arange(len(x), dtype=np.int64)
        types = _as_np(types_raw).reshape(-1) if types_raw is not None else np.zeros(len(x), dtype=np.int64)
        tr = state.sim_trajectory
        t = _current_timestep(state)
        cx = float(_as_np(tr.x)[sdc, t]); cy = float(_as_np(tr.y)[sdc, t])
        keep = np.flatnonzero(valid & (((x - cx) ** 2 + (y - cy) ** 2) <= float(radius_m) ** 2))
        if keep.size == 0:
            return {}
        groups: list[dict[str, Any]] = []
        # Preserve original point order within each feature id; this is the
        # ordering carried by the WOMD roadgraph arrays.
        for gid in np.unique(ids[keep]):
            idx = keep[ids[keep] == gid]
            if idx.size < 2:
                continue
            if idx.size > max_points_per_polyline:
                sample = np.linspace(0, idx.size - 1, max_points_per_polyline).round().astype(int)
                idx = idx[sample]
            type_values = types[idx]
            type_id = int(type_values[0]) if type_values.size else 0
            groups.append({
                "id": int(gid) if np.issubdtype(np.asarray(gid).dtype, np.number) else str(gid),
                "type": type_id,
                "xy": [[float(x[i]), float(y[i])] for i in idx],
            })
            if len(groups) >= max(1, int(max_polylines)):
                break
        return {
            "roadgraph_polylines": groups,
            "center_xy": [cx, cy],
            "radius_m": float(radius_m),
        } if groups else {}
    except Exception:
        return {}


def _capture_render_frame(
    state: Any,
    sdc: int,
    *,
    max_agents: int = 64,
    selected_macro: str = "nominal",
    selected_candidate_index: int = 0,
    selection_reason: str = "initial",
    metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Capture a compact, renderer-independent Waymax frame.

    The trace is enabled only for explicitly requested diagnostic/toy-example
    runs.  It records physical state, not rasterized pixels, so the same scene
    can be rendered reproducibly as MP4/GIF without depending on a Waymax GUI.
    """
    tr = state.sim_trajectory
    t = _current_timestep(state)
    x = _as_np(tr.x)[:, t]
    y = _as_np(tr.y)[:, t]
    yaw = _as_np(tr.yaw)[:, t]
    length = _as_np(tr.length)[:, t]
    width = _as_np(tr.width)[:, t]
    valid = _as_np(tr.valid)[:, t].astype(bool)
    if sdc < 0 or sdc >= len(x):
        raise IndexError(f"invalid SDC index {sdc} for {len(x)} objects")
    indices = np.flatnonzero(valid)
    if indices.size:
        dist2 = (x[indices] - x[sdc]) ** 2 + (y[indices] - y[sdc]) ** 2
        indices = indices[np.argsort(dist2)[: max(1, int(max_agents))]]
    agents = [
        {
            "object_index": int(i),
            "is_sdc": bool(int(i) == int(sdc)),
            "x": float(x[i]), "y": float(y[i]), "yaw": float(yaw[i]),
            "length": float(length[i]), "width": float(width[i]),
        }
        for i in indices
    ]
    return {
        "time_index": int(t),
        "selected_macro": str(selected_macro),
        "selected_candidate_index": int(selected_candidate_index),
        "selection_reason": str(selection_reason),
        "metrics": {str(k): float(v) for k, v in (metrics or {}).items() if np.isfinite(v)},
        "agents": agents,
    }


def _observable_regime_name(state: Any, sdc: int, cfg: dict, fallback: str = "") -> str:
    """Infer the runtime policy regime from current observable geometry only.

    This prevents the closed-loop planner from receiving the evaluation bucket
    as an oracle neural input. Contact takes priority over near-contact; all
    remaining states use the Safe policy. Thresholds match the paper's regime
    definitions and can be overridden through ``regime_thresholds``.
    """
    sel = cfg.get("selection", {}) if isinstance(cfg.get("selection", {}), dict) else {}
    if not bool(sel.get("auto_regime_from_observation", False)):
        return str(fallback or "")
    thresholds = cfg.get("regime_thresholds", {}) if isinstance(cfg.get("regime_thresholds", {}), dict) else {}
    tau_d = float(thresholds.get("tau_d", 2.0))
    tau_ttc = float(thresholds.get("tau_ttc", 3.0))
    tau_contact = float(thresholds.get("tau_contact", 0.8))
    metrics = _state_geometry_metrics(state, sdc)
    if not metrics:
        return str(fallback or "safe")
    clearance = float(metrics.get("min_clearance_m", float("inf")))
    ttc = float(metrics.get("ttc_s", float("inf")))
    if clearance < tau_contact:
        return "contact"
    if clearance < tau_d or ttc < tau_ttc:
        return "near_contact"
    return "safe"


def _state_geometry_snapshot(state: Any, sdc: int, *, timestep: int | None = None) -> tuple[dict[str, float], list[float]]:
    """Return current geometry metrics and ego XY in one trajectory read."""
    try:
        tr = state.sim_trajectory
        t = _current_timestep(state) if timestep is None else int(timestep)
        # _state_geometry_metrics intentionally uses state's current timestep.
        # Test/runtime calls pass the current value, so this preserves semantics.
        metrics = _state_geometry_metrics(state, sdc)
        xy = [float(_as_np(tr.x)[sdc, t]), float(_as_np(tr.y)[sdc, t])]
        return metrics, xy
    except Exception:
        return _state_geometry_metrics(state, sdc), []


def _step_metrics_geometry_snapshot(waymax_env: Any, state: Any, sdc: int) -> tuple[dict[str, float], list[float], int, bool]:
    """Collect Waymax metrics, physical margins, trace point and state flags."""
    metrics = _metric_summary(waymax_env, state, sdc)
    geom, xy = _state_geometry_snapshot(state, sdc)
    metrics.update(geom)
    return metrics, xy, _current_timestep(state), _scene_done(state)


def _global_route_from_history(history: Any) -> tuple[np.ndarray | None, str | None]:
    """Recover a fixed global route reference from the first rollout history.

    Validation TFExample shards can omit ``path_samples/*``. In that case the
    history builder creates a logged-future route proxy in the ego frame. The
    fixed first-step route is transformed back to global coordinates so paired
    policies use the same route-progress reference without reading test labels.
    """
    route = np.asarray(getattr(history, "route", np.zeros((0, 2))), dtype=np.float64)
    if route.ndim != 2 or route.shape[0] < 2 or route.shape[1] < 2:
        return None, None
    xy_local = route[:, :2]
    xy_local = xy_local[np.all(np.isfinite(xy_local), axis=1)]
    if xy_local.shape[0] < 2:
        return None, None
    metadata = getattr(history, "metadata", {}) or {}
    try:
        ego_xy = np.asarray(metadata.get("ego_global_xy"), dtype=np.float64).reshape(2)
        heading = float(metadata.get("ego_global_heading", 0.0))
    except (TypeError, ValueError):
        return None, None
    # Inverse of transform_points_to_ego for row-vector points.
    xy_global = xy_local @ rotation_matrix(-heading).astype(np.float64) + ego_xy
    length = float(np.sum(np.linalg.norm(np.diff(xy_global, axis=0), axis=1)))
    if not np.isfinite(length) or length < 1.0:
        return None, None
    source = (
        "history_future_route_proxy"
        if bool(metadata.get("route_sanitized", False))
        else "waymax_sdc_route"
    )
    return xy_global.astype(np.float32), source


def _route_progression_from_trace(
    state_xy_trace: list[list[float]], route_xy_global: np.ndarray | None
) -> float | None:
    """Return signed arc-length progress along a fixed scene route."""
    if route_xy_global is None or len(state_xy_trace) < 2:
        return None
    trace = np.asarray(state_xy_trace, dtype=np.float32)
    if trace.ndim != 2 or trace.shape[1] < 2:
        return None
    trace = trace[np.all(np.isfinite(trace[:, :2]), axis=1), :2]
    if trace.shape[0] < 2:
        return None
    route = np.zeros((len(route_xy_global), 6), dtype=np.float32)
    route[:, :2] = np.asarray(route_xy_global, dtype=np.float32)[:, :2]
    try:
        start = project_to_route(trace[0], route)
        end = project_to_route(trace[-1], route)
    except Exception:
        return None
    return float(end.s - start.s)


def _scene_done(state: Any) -> bool:
    try:
        return bool(_as_np(state.is_done).reshape(()).item())
    except Exception:
        try:
            return _current_timestep(state) >= int(_as_np(state.log_trajectory.valid).shape[-1]) - 2
        except Exception:
            return False


def _sample_to_dict(sample) -> dict[str, Any]:
    return sample.to_npz_dict()


def _sample_to_inference_dict(sample) -> dict[str, Any]:
    """Return only the fields needed by online model inference/selection.

    ``DatasetSample.to_npz_dict()`` is intentionally complete because it is used
    for persisted training/evaluation samples.  In closed-loop fast mode, however,
    calling it for every candidate at every replan repeatedly copies large shared
    history/map/BEV arrays and serializes diagnostics/future metadata that the
    model never reads.  This lightweight view preserves the exact model features
    and selector inputs while avoiding those extra CPU allocations.
    """
    h = sample.h_t
    p = sample.prefix
    return {
        "scene_id": sample.scene_id,
        "original_scenario_id": sample.original_scenario_id,
        "time_index": np.int64(sample.time_index),
        "candidate_index": np.int64(sample.candidate_index),
        "split_id": sample.split_id,
        "is_nominal": np.int64(sample.is_nominal),
        "agent_history": np.asarray(h.agent_history, dtype=np.float32),
        "agent_valid": np.asarray(h.agent_valid, dtype=np.float32),
        "map_polylines": np.asarray(h.map_polylines, dtype=np.float32),
        "map_valid": np.asarray(h.map_valid, dtype=np.float32),
        "dynamic_map": np.asarray(h.dynamic_map, dtype=np.float32),
        "route": np.asarray(h.route, dtype=np.float32),
        "bev_occ": np.asarray(h.occ_mask, dtype=np.float32),
        "ego_state": np.asarray(h.ego_state, dtype=np.float32),
        "prefix_states": np.asarray(p.prefix_states, dtype=np.float32),
        "prefix_controls": np.asarray(p.prefix_controls, dtype=np.float32),
        "prefix_macro_id": np.int64(p.macro_id),
        "prefix_macro_type_id": np.int64((p.diagnostics or {}).get("macro_type_id", p.macro_id)),
        "prefix_macro_name": p.macro_name,
        "prefix_param": np.asarray(p.params, dtype=np.float32),
        "utility": np.float32(p.utility),
        "hard_violation": np.float32(p.hard_violation),
        "harm_proxy": np.float32(p.harm_proxy),
        "feasible": np.int64(p.feasible),
        "root_probs": np.asarray(sample.root_probs, dtype=np.float32),
        "root_signature": np.asarray(sample.root_signature, dtype=np.float32),
        "root_future_signature": np.asarray(sample.root_future_signature, dtype=np.float32),
        "root_valid": np.asarray(sample.root_valid, dtype=np.float32),
        "y_obs": np.asarray(sample.y_obs, dtype=np.float32),
        "c_star": np.asarray(sample.c_star, dtype=np.float32),
        "m_star": np.asarray(sample.m_star, dtype=np.float32),
        "option_valid": np.asarray(sample.option_valid, dtype=np.float32),
        "recovery_modes": np.asarray([g.mode for g in sample.recovery_options]),
        "recovery_params": pad_recovery_params(sample.recovery_options).astype(np.float32, copy=False),
        "r_orc_star": np.float32(sample.r_orc_star),
        "r_dep_star": np.float32(sample.r_dep_star),
        "oracle_gap_star": np.float32(sample.oracle_gap_star),
        "i_art_star": np.int64(sample.i_art_star),
    }


def _sample_to_audit_dict(sample) -> dict[str, Any]:
    """Small teacher-label view used by closed-loop audit metrics.

    Audit metrics only consume root probabilities, margins, option validity and
    the precomputed OC-MERO scalars.  Avoid copying history/map/BEV/future
    diagnostics through ``to_npz_dict`` for every audited candidate.
    """
    return {
        "candidate_index": np.int64(sample.candidate_index),
        "root_probs": np.asarray(sample.root_probs, dtype=np.float32),
        "root_valid": np.asarray(sample.root_valid, dtype=np.float32),
        "m_star": np.asarray(sample.m_star, dtype=np.float32),
        "option_valid": np.asarray(sample.option_valid, dtype=np.float32),
        "r_dep_star": np.float32(sample.r_dep_star),
        "r_orc_star": np.float32(sample.r_orc_star),
        "oracle_gap_star": np.float32(sample.oracle_gap_star),
    }


def _select_prefix(
    samples: list,
    bundle: ModelBundle | None,
    cfg: dict,
    method: str,
    gamma: float,
    *,
    compute_teacher_labels: bool = True,
    external_model: Any | None = None,
    external_model_cfg: dict | None = None,
    external_device: Any | None = None,
) -> tuple[int, dict[str, Any]]:
    if compute_teacher_labels:
        dicts = [_sample_to_dict(s) for s in samples]
    else:
        dicts = [_sample_to_inference_dict(s) for s in samples]
    preds = predict_samples(dicts, bundle, cfg, shared_scene_features=True) if bundle is not None else [predict_sample(d, None, cfg) for d in dicts]
    items = []
    for s, d, pred in zip(samples, dicts, preds):
        teacher = teacher_prediction_from_sample(d, cfg) if compute_teacher_labels else None
        items.append({"sample": s, "data": d, "pred": pred, "teacher": teacher})

    method_l = str(method).lower()
    if method_l in EXTERNAL_CLOSED_LOOP_METHODS:
        # Post-impact source controllers use absolute time since the first
        # post-contact decision (e.g. Ghosh 2026 open-loop steering/force
        # pulses and Lu/Ford braking termination).  Keep this runtime-only
        # observable outside stored dataset tensors to avoid any label leak.
        sel_cfg = cfg.get("selection", {}) if isinstance(cfg.get("selection", {}), dict) else {}
        elapsed = sel_cfg.get("runtime_contact_elapsed_s", None)
        if elapsed is not None:
            for d in dicts:
                d["runtime_contact_elapsed_s"] = np.asarray(float(elapsed), dtype=np.float32)
        ext_cfg = external_model_cfg if isinstance(external_model_cfg, dict) else cfg
        ext_outputs = _predict_external_group(external_model, dicts, ext_cfg, external_device) if external_model is not None else {}
        selected_ext = select_external_policy(method_l, dicts, ext_cfg, model_outputs=ext_outputs)
        # Continue through the common metric/audit path by exposing the external
        # selection through the same fields used by built-in baselines.
        external_selected = selected_ext
    else:
        external_selected = None

    utility = np.asarray([_safe_float(x["data"].get("utility", 0.0)) for x in items], dtype=np.float32)
    pred_r_dep = np.asarray([float(x["pred"].r_dep) for x in items], dtype=np.float32)
    pred_r_orc = np.asarray([float(x["pred"].r_orc) for x in items], dtype=np.float32)
    pred_gap = np.asarray([float(x["pred"].gap) for x in items], dtype=np.float32)
    sel_tmp = cfg.get("selection", {}) if isinstance(cfg.get("selection", {}), dict) else {}
    active_bucket = str(sel_tmp.get("active_bucket_name", sel_tmp.get("regime_name", "")) or "")
    drs_gamma = _drs_success_gamma_for_bucket(gamma, cfg, active_bucket)
    option_semantics = option_execution_semantics(cfg)
    pred_drs = np.asarray([predicted_option_success(x["pred"].q, x["pred"].root_probs, gamma=drs_gamma, root_valid=x["data"].get("root_valid", None), option_valid=x["data"].get("option_valid", None), semantics=option_semantics) for x in items], dtype=np.float32)
    pred_direct_value = np.asarray([np.nan if x["pred"].direct_recovery_value is None else float(x["pred"].direct_recovery_value) for x in items], dtype=np.float32)
    pred_direct_rank = np.asarray([np.nan if x["pred"].direct_recovery_rank is None else float(x["pred"].direct_recovery_rank) for x in items], dtype=np.float32)
    pred_direct_rank = np.where(np.isfinite(pred_direct_rank), pred_direct_rank, pred_direct_value).astype(np.float32)
    pred_direct_std = np.asarray([np.nan if x["pred"].direct_recovery_std is None else float(x["pred"].direct_recovery_std) for x in items], dtype=np.float32)
    pred_direct_delta = np.asarray([np.nan if x["pred"].direct_recovery_delta is None else float(x["pred"].direct_recovery_delta) for x in items], dtype=np.float32)
    pred_direct_delta_std = np.asarray([np.nan if x["pred"].direct_recovery_delta_std is None else float(x["pred"].direct_recovery_delta_std) for x in items], dtype=np.float32)
    pred_direct_opportunity = np.asarray([np.nan if x["pred"].direct_recovery_opportunity is None else float(x["pred"].direct_recovery_opportunity) for x in items], dtype=np.float32)
    pred_direct_harm = np.asarray([np.nan if x["pred"].direct_recovery_harm is None else float(x["pred"].direct_recovery_harm) for x in items], dtype=np.float32)
    opp_logits = np.asarray([np.nan if x["pred"].direct_recovery_opportunity_logit is None else float(x["pred"].direct_recovery_opportunity_logit) for x in items], dtype=np.float32)
    harm_logits = np.asarray([np.nan if x["pred"].direct_recovery_harm_logit is None else float(x["pred"].direct_recovery_harm_logit) for x in items], dtype=np.float32)
    nominal_ids = [i for i, x in enumerate(items) if _safe_float(x["data"].get("is_nominal", 0.0)) > 0.5]
    risk_source = str(sel_tmp.get("direct_value_risk_source", "heads") or "heads").strip().lower()
    if nominal_ids:
        ni = nominal_ids[0]
        if risk_source == "conformal_delta" and np.isfinite(pred_direct_delta).any():
            pred_direct_value = np.where(np.isfinite(pred_direct_delta), pred_direct_delta, -np.inf).astype(np.float32)
            pred_direct_value[ni] = 0.0
            q = float(sel_tmp.get("direct_value_conformal_overprediction_quantile", sel_tmp.get("direct_value_additive_q", 0.0)) or 0.0)
            temp = max(1.0e-4, float(sel_tmp.get("direct_value_conformal_temperature", 0.02) or 0.02))
            pos_gain = float(sel_tmp.get("direct_value_positive_gain", 0.015))
            neg_gain = float(sel_tmp.get("direct_value_negative_gain", 0.010))
            gain_lcb = pred_direct_value - q
            pred_direct_std = np.zeros_like(pred_direct_value, dtype=np.float32)
            pred_direct_opportunity = (1.0 / (1.0 + np.exp(-np.clip((gain_lcb - pos_gain) / temp, -30.0, 30.0)))).astype(np.float32)
            pred_direct_harm = (1.0 / (1.0 + np.exp(-np.clip((-neg_gain - gain_lcb) / temp, -30.0, 30.0)))).astype(np.float32)
        elif risk_source == "direct_delta" and np.isfinite(pred_direct_delta).any():
            from math import erf, sqrt
            pred_direct_value = np.where(np.isfinite(pred_direct_delta), pred_direct_delta, -np.inf).astype(np.float32)
            pred_direct_value[ni] = 0.0
            pred_direct_std = np.where(np.isfinite(pred_direct_delta_std), pred_direct_delta_std, np.inf).astype(np.float32)
            pred_direct_std[ni] = 0.0
            delta_mean = pred_direct_value
            delta_std = np.maximum(1.0e-6, pred_direct_std)
            pos_gain = float(sel_tmp.get("direct_value_positive_gain", 0.015))
            neg_gain = float(sel_tmp.get("direct_value_negative_gain", 0.010))
            z_opp = np.clip((delta_mean - pos_gain) / delta_std, -12.0, 12.0)
            z_harm = np.clip((-neg_gain - delta_mean) / delta_std, -12.0, 12.0)
            normal_cdf = np.vectorize(lambda z: 0.5 * (1.0 + erf(float(z) / sqrt(2.0))))
            pred_direct_opportunity = normal_cdf(z_opp).astype(np.float32)
            pred_direct_harm = normal_cdf(z_harm).astype(np.float32)
        elif risk_source == "ordinal_evidence" and np.isfinite(opp_logits[ni]) and np.isfinite(harm_logits[ni]):
            opp_delta = np.clip(opp_logits - opp_logits[ni], -30.0, 30.0)
            harm_delta = np.clip(harm_logits - harm_logits[ni], -30.0, 30.0)
            pred_direct_opportunity = (1.0 / (1.0 + np.exp(-opp_delta))).astype(np.float32)
            pred_direct_harm = (1.0 / (1.0 + np.exp(-harm_delta))).astype(np.float32)
            if np.isfinite(pred_direct_delta).any():
                pred_direct_value = np.where(np.isfinite(pred_direct_delta), pred_direct_delta, -np.inf).astype(np.float32)
                pred_direct_value = pred_direct_value - float(pred_direct_value[ni])
            else:
                pred_direct_value = (pred_direct_opportunity - pred_direct_harm).astype(np.float32)
                pred_direct_value[ni] = 0.0
            pred_direct_std = np.zeros_like(pred_direct_value, dtype=np.float32)
        elif risk_source == "delta_distribution" and np.isfinite(pred_direct_value[ni]):
            from math import erf, sqrt
            delta_mean = pred_direct_value - pred_direct_value[ni]
            delta_std = np.sqrt(np.maximum(1.0e-6, pred_direct_std ** 2 + pred_direct_std[ni] ** 2))
            pos_gain = float(sel_tmp.get("direct_value_positive_gain", 0.015))
            neg_gain = float(sel_tmp.get("direct_value_negative_gain", 0.010))
            z_opp = np.clip((delta_mean - pos_gain) / delta_std, -12.0, 12.0)
            z_harm = np.clip((-neg_gain - delta_mean) / delta_std, -12.0, 12.0)
            normal_cdf = np.vectorize(lambda z: 0.5 * (1.0 + erf(float(z) / sqrt(2.0))))
            pred_direct_opportunity = normal_cdf(z_opp).astype(np.float32)
            pred_direct_harm = normal_cdf(z_harm).astype(np.float32)
        elif np.isfinite(opp_logits[ni]):
            delta = np.clip(opp_logits - opp_logits[ni], -30.0, 30.0)
            pred_direct_opportunity = (1.0 / (1.0 + np.exp(-delta))).astype(np.float32)
            if np.isfinite(harm_logits[ni]):
                hdelta = np.clip(harm_logits - harm_logits[ni], -30.0, 30.0)
                pred_direct_harm = (1.0 / (1.0 + np.exp(-hdelta))).astype(np.float32)
    macro_names = [str(x["data"].get("prefix_macro_name", "")) for x in items]
    nominal_deviation = _prefix_nominal_deviation(samples)
    if compute_teacher_labels:
        teacher_r_dep = np.asarray([_safe_float(x["data"].get("r_dep_star", 0.0)) for x in items], dtype=np.float32)
        teacher_r_orc = np.asarray([_safe_float(x["data"].get("r_orc_star", 0.0)) for x in items], dtype=np.float32)
    else:
        teacher_r_dep = np.full((len(items),), np.nan, dtype=np.float32)
        teacher_r_orc = np.full((len(items),), np.nan, dtype=np.float32)
    hard = np.asarray([_safe_float(x["data"].get("hard_violation", 0.0)) for x in items], dtype=np.float32)
    harm = np.asarray([_safe_float(x["data"].get("harm_proxy", 0.0)) for x in items], dtype=np.float32)
    feasible = np.asarray([bool(int(_safe_float(x["data"].get("feasible", 1.0), 1.0))) for x in items], dtype=bool)
    sel_cfg = cfg.get("selection", {}) if isinstance(cfg.get("selection", {}), dict) else {}
    if external_selected is not None:
        selected = external_selected
    else:
        selected = select_baseline(
            method,
            utility,
            pred_r_dep,
            teacher_r_dep,
            teacher_r_orc,
            hard,
            harm,
            feasible,
            gamma,
            float(sel_cfg.get("gamma_H", 0.0)),
            float(sel_cfg.get("gamma_D", 5.0)),
            cfg,
            pred_r_orc=pred_r_orc,
            pred_gap=pred_gap,
            nominal_deviation=nominal_deviation,
            pred_drs=pred_drs,
            pred_direct_value=pred_direct_value,
            pred_direct_rank=pred_direct_rank,
            pred_direct_std=pred_direct_std,
            pred_direct_opportunity=pred_direct_opportunity,
            pred_direct_harm=pred_direct_harm,
            candidate_macro_names=macro_names,
        )
    idx = int(selected.selected_index)
    chosen = items[idx]
    nup = nominal_utility_preservation(utility[0] if len(utility) else 0.0, utility[idx], sigma_u=float((cfg.get("metrics", {}) or {}).get("sigma_u", 1.0)))

    if compute_teacher_labels:
        d = chosen["data"]
        # Evaluate one globally shared option.  If OC-RAP did not intervene
        # (nominal index 0), use the teacher option for a nominal-prefix
        # diagnostic; if it intervened, evaluate the model-selected shared
        # recovery option against the teacher margins.
        use_model_option = bool(method == "ocrap" and idx != 0)
        q_eval = chosen["pred"].q if use_model_option else chosen["teacher"].q
        opt_gamma = drs_gamma if use_model_option else 0.0
        selected_option = best_option_indices(
            q_eval, d["root_probs"], gamma=opt_gamma,
            root_valid=d.get("root_valid", None), option_valid=d.get("option_valid", None),
            semantics=option_semantics,
        )
        drs = deployable_recovery_success(d["m_star"], d["root_probs"], selected_option, d.get("root_valid", None))
        fra_cand = false_recoverability_admission(selected.admitted, teacher_r_dep)
    else:
        drs = None
        fra_cand = None

    info = {
        "items": items,
        "utility": utility,
        "teacher_r_dep": teacher_r_dep,
        "teacher_r_orc": teacher_r_orc,
        "selection": selected,
        "pred_r_dep": pred_r_dep,
        "pred_r_orc": pred_r_orc,
        "pred_gap": pred_gap,
        "pred_drs": pred_drs,
        "pred_direct_value": pred_direct_value,
        "pred_direct_std": pred_direct_std,
        "pred_direct_opportunity": pred_direct_opportunity,
        "pred_direct_rank": pred_direct_rank,
        "nominal_deviation": nominal_deviation,
        "labels_available": bool(compute_teacher_labels),
        "drs": None if drs is None else float(drs),
        "nup": float(nup["bounded_NUP"]),
        "fra_cand": None if fra_cand is None else float(fra_cand),
    }
    return idx, info


def _strip_version_suffix(name: str) -> str:
    base, sep, version = str(name).rpartition("_v")
    return base if sep and version.isdigit() and base else str(name)


def _bucket_gamma_aliases(name: str | None) -> list[str]:
    return canonical_bucket_aliases(name)


def _is_post_contact_bucket_name(bucket_name: str | None) -> bool:
    """Recognise semantic contact targets under dataset provenance prefixes.

    ``near_contact`` remains non-contact by construction in
    :func:`canonical_regime_name`.
    """
    return is_post_contact_bucket(bucket_name)



def _gamma_for_bucket(base_gamma: float, cfg: dict, bucket_name: str | None) -> float:
    """Return a regime/bucket-specific calibrated recovery threshold when set."""
    sel_cfg = cfg.get("selection", {}) if isinstance(cfg.get("selection", {}), dict) else {}
    mapping = sel_cfg.get("gamma_rec_by_bucket", {})
    if not isinstance(mapping, dict) or not bucket_name:
        return float(base_gamma)
    for key in _bucket_gamma_aliases(bucket_name):
        if key in mapping and mapping[key] not in {None, ""}:
            try:
                val = float(mapping[key])
                if np.isfinite(val):
                    return val
            except Exception:
                continue
    return float(base_gamma)


def _drs_success_gamma_for_bucket(base_gamma: float, cfg: dict, bucket_name: str | None) -> float:
    sel_cfg = cfg.get("selection", {}) if isinstance(cfg.get("selection", {}), dict) else {}
    default = sel_cfg.get("drs_success_gamma", 0.0)
    for map_key in ("drs_success_gamma_by_bucket", "drs_success_gamma_by_regime"):
        mapping = sel_cfg.get(map_key, None)
        if isinstance(mapping, dict):
            for key in _bucket_gamma_aliases(bucket_name):
                if key in mapping and mapping[key] not in {None, ""}:
                    try:
                        return float(mapping[key])
                    except Exception:
                        pass
    try:
        return float(default)
    except Exception:
        return 0.0


def _validate_closed_loop_selector_config(cfg: dict, method: str) -> None:
    if str(method).lower() != "ocrap":
        return
    cl_cfg = cfg.get("closed_loop", {}) if isinstance(cfg.get("closed_loop", {}), dict) else {}
    sel = cfg.get("selection", {}) if isinstance(cfg.get("selection", {}), dict) else {}
    selector = str(sel.get("ocrap_selector", sel.get("selector", "lcb_constrained"))).lower()
    if bool(cl_cfg.get("require_calibrated_selector", False)) and selector not in {"calibrated", "calibrated_constrained", "soft_constrained", "budgeted_calibrated"}:
        raise ValueError(f"closed-loop requires calibrated OC-RAP selector, but selection.ocrap_selector={selector!r}")
    if bool(cl_cfg.get("require_gamma_by_bucket", False)) and not sel.get("gamma_rec_by_bucket", {}):
        raise ValueError("closed-loop requires non-empty selection.gamma_rec_by_bucket; check gamma_rec_by_bucket_file path")

def _rollout_one_scene(
    raw,
    scenario_rank: int,
    bundle: ModelBundle | None,
    cfg: dict,
    method: str,
    gamma: float,
    *,
    start_time_index_override: int | None = None,
    bucket_name: str | None = None,
    target_key: str | None = None,
    external_model: Any | None = None,
    external_model_cfg: dict | None = None,
    external_device: Any | None = None,
) -> dict[str, Any]:
    import jax  # type: ignore

    cl_cfg = cfg.get("closed_loop", {}) if isinstance(cfg.get("closed_loop", {}), dict) else {}
    max_steps = int(cl_cfg.get("max_steps", 40))
    replan_interval = max(1, int(cl_cfg.get("replan_interval_steps", 1)))
    start_t = start_time_index_override if start_time_index_override is not None else cl_cfg.get("start_time_index", None)
    if start_t is None:
        start_t = int((cfg.get("waymax", {}) or {}).get("init_history_steps", 11)) - 1
    start_t = int(start_t)
    requested_label_mode = str(cl_cfg.get("label_mode", "fast")).lower()
    label_mode = requested_label_mode
    teacher_required_methods = {"ocrap_teacher"} | set(EXTERNAL_TEACHER_REQUIRED_METHODS)
    branchwise_methods = {"backup_filter", "oracle_filter", "contingency"}
    force_teacher_baselines = bool(cl_cfg.get("force_teacher_baselines", False))
    if method in teacher_required_methods or (method == "ocrap" and bundle is None) or (force_teacher_baselines and method in branchwise_methods):
        # Full teacher labels are very expensive online. Branch-wise/oracle
        # baselines therefore run as predicted-oracle proxies in fast mode by
        # default, and use true teacher labels only when explicitly requested.
        label_mode = "all"
    selected_label_audit = label_mode in {"selected", "selected_only", "audit_selected", "executed"}
    coverage_label_audit = label_mode in {"selected_topk", "topk", "coverage", "audit_topk", "selected_coverage"}
    compute_teacher_labels = label_mode in {"all", "full", "teacher", "labels"}
    external_sparse_labels = (
        compute_teacher_labels
        and method in EXTERNAL_TEACHER_REQUIRED_METHODS
        and bool(cl_cfg.get("external_sparse_labels", True))
        and not bool(cl_cfg.get("exhaustive_teacher_labels", False))
    )
    sparse_label_decisions = 0
    sparse_label_candidates_total = 0
    sparse_label_full_candidates_total = 0
    audit_every_n_steps = max(1, int(cl_cfg.get("audit_every_n_steps", 1) or 1))
    audit_max_labels = int(cl_cfg.get("audit_max_labels", 0) or 0)
    audit_auto_capped = False
    if coverage_label_audit and audit_max_labels <= 0:
        auto_cap = int(cl_cfg.get("audit_auto_max_labels", 256) or 0)
        if auto_cap > 0:
            audit_max_labels = auto_cap
            audit_auto_capped = True
    audit_labels_done = 0
    progress = bool(cl_cfg.get("progress", True))
    progress_every = max(1, int(cl_cfg.get("progress_every_steps", 5)))
    profile_timing = bool(cl_cfg.get("profile_timing", True))
    timing_totals = {
        "state_history": 0.0,
        "candidate_features": 0.0,
        "teacher_labels": 0.0,
        "policy_selection": 0.0,
        "audit_labels": 0.0,
        "waymax_step_metrics": 0.0,
    }
    scene_wall_t0 = perf_counter()

    state0 = raw.metadata.get("_waymax_state")
    if state0 is None:
        raise ValueError("Closed-loop runner requires Waymax RawScenario metadata['_waymax_state'].")
    local_cfg = dict(cfg)
    local_cfg["_waymax_init_steps_override"] = start_t + 1
    wx_env, _dyn_name = _make_env(state0, local_cfg, allow_new=bool((cfg.get("waymax", {}) or {}).get("allow_new_objects_after_warmup", True)))
    state = wx_env.reset(state0, rng=jax.random.PRNGKey(int((cfg.get("seed", 7) + scenario_rank) & 0x7FFFFFFF)))
    sdc = _sdc_index(state)
    decisions: list[ClosedLoopDecision] = []
    active_regime_trace: list[str] = []
    metric_trace: list[dict[str, float]] = []
    state_xy_trace: list[list[float]] = []
    render_trace: list[dict[str, Any]] = []
    render_context: dict[str, Any] = {}
    render_trace_enabled = bool(cl_cfg.get("render_trace", cl_cfg.get("render", False)))
    render_max_agents = max(1, int(cl_cfg.get("render_max_agents", 64)))
    render_max_roadgraph_polylines = max(1, int(cl_cfg.get("render_max_roadgraph_polylines", 192)))
    render_max_points_per_polyline = max(2, int(cl_cfg.get("render_max_points_per_polyline", 64)))
    render_roadgraph_radius_m = max(10.0, float(cl_cfg.get("render_roadgraph_radius_m", 80.0)))
    route_reference_global: np.ndarray | None = None
    route_reference_source: str | None = None
    try:
        initial_tr = state.sim_trajectory
        initial_tt = _current_timestep(state)
        state_xy_trace.append([
            float(_as_np(initial_tr.x)[sdc, initial_tt]),
            float(_as_np(initial_tr.y)[sdc, initial_tt]),
        ])
    except Exception:
        pass
    if render_trace_enabled:
        render_context = _capture_render_context(
            state, sdc, radius_m=render_roadgraph_radius_m,
            max_polylines=render_max_roadgraph_polylines,
            max_points_per_polyline=render_max_points_per_polyline,
        )
        try:
            render_trace.append(
                _capture_render_frame(
                    state, sdc, max_agents=render_max_agents,
                    selected_macro="nominal", selected_candidate_index=0,
                    selection_reason="initial",
                    metrics=_state_geometry_metrics(state, sdc),
                )
            )
        except Exception as exc:
            if progress:
                print({"event": "closed_loop_render_trace_initial_failed", "scene_rank": scenario_rank, "error": str(exc)}, flush=True)
    interventions_used = 0
    last_intervention_step = -10**9
    previous_selected_macro = "nominal"
    same_macro_run_length = 0
    audit_bucket_name = bucket_name or str((cfg.get("selection", {}) or {}).get("active_bucket_name", "") or "")
    drs_gamma = _drs_success_gamma_for_bucket(gamma, cfg, audit_bucket_name)
    # ``_select_prefix`` computes the same execution semantics locally for the
    # main selector path.  The selected/top-k teacher-label audit below lives in
    # this outer scope and must use the identical semantics explicitly; without
    # this binding the audit raised NameError while the actual rollout continued.
    option_semantics = option_execution_semantics(cfg)

    # These settings are invariant across replans.  Constructing/deep-copying
    # them at every simulator step was pure overhead.
    eval_cfg = dict(cfg)
    if cl_cfg.get("num_candidate_prefixes", None) is not None:
        eval_cfg["num_candidate_prefixes"] = int(cl_cfg["num_candidate_prefixes"])
    quality = dict(eval_cfg.get("dataset_quality", {}) or {})
    quality.update({
        "balanced_two_pass": False,
        "max_accepted_prefixes_per_scene_time": 0,
        "min_artifact_prefixes_per_scene_time": 0,
        "min_nonartifact_prefixes_per_scene_time": 0,
        "min_obs_negative_fraction_per_sample": 0.0,
        "require_negative_deployable_sample": False,
        "require_artifact_pairs": False,
        "artifact_pair_mode": "tag",
    })
    eval_cfg["dataset_quality"] = quality
    cl_num_options = cl_cfg.get("num_recovery_options", None)
    feature_num_options = int(cl_num_options) if cl_num_options is not None else int(
        eval_cfg.get("num_recovery_options", getattr(bundle.model, "num_options", 24) if bundle is not None else 24)
    )
    feature_num_roots = int(getattr(bundle.model, "num_roots", int(cfg.get("num_roots", 8)))) if bundle is not None else int(cfg.get("num_roots", 8))

    for step_idx in range(max_steps):
        if _scene_done(state):
            break
        t = _current_timestep(state)
        if progress and (step_idx == 0 or step_idx % progress_every == 0):
            print({"event": "closed_loop_step", "scene_rank": scenario_rank, "scene_id": str(raw.scenario_id), "step": step_idx, "time_index": int(t), "label_mode": label_mode}, flush=True)
        timing_t0 = perf_counter()
        spliced_raw = raw_scenario_from_waymax_state(
            state,
            f"{raw.scenario_id}__cl{scenario_rank:04d}",
            scenario_rank,
            cfg,
            trajectory_mode="closed_loop_splice",
            splice_until=t,
            static_template=raw,
        )
        hist = construct_history(spliced_raw, t, cfg)
        if route_reference_global is None:
            route_reference_global, route_reference_source = _global_route_from_history(hist)
        hist.metadata["_waymax_state"] = state
        hist.metadata["_waymax_branch_from_current"] = True
        hist.metadata["waymax_planning_timestep"] = int(t)
        if profile_timing:
            timing_totals["state_history"] += perf_counter() - timing_t0
        if compute_teacher_labels:
            if external_sparse_labels:
                timing_t0 = perf_counter()
                feature_samples = build_feature_only_samples_for_history(
                    hist,
                    "closed_loop",
                    eval_cfg,
                    num_roots=feature_num_roots,
                    num_options=feature_num_options,
                )
                if profile_timing:
                    timing_totals["candidate_features"] += perf_counter() - timing_t0
                audit_indices = _preselect_external_label_candidate_indices(feature_samples, method, cfg)
                sparse_label_decisions += 1
                sparse_label_full_candidates_total += int(len(feature_samples))
                sparse_label_candidates_total += int(len(audit_indices))
                timing_t0 = perf_counter()
                samples = build_labeled_samples_for_candidate_indices(
                    hist,
                    "closed_loop",
                    eval_cfg,
                    audit_indices,
                    num_roots=feature_num_roots,
                    num_options=feature_num_options,
                    prefixes=[s.prefix for s in feature_samples],
                    recovery_options=feature_samples[0].recovery_options if feature_samples else None,
                    recovery_option_valid=feature_samples[0].option_valid if feature_samples else None,
                    assign_regime_labels=False,
                )
                if profile_timing:
                    timing_totals["teacher_labels"] += perf_counter() - timing_t0
            else:
                timing_t0 = perf_counter()
                samples = build_samples_for_history(hist, "closed_loop", eval_cfg)
                if profile_timing:
                    timing_totals["teacher_labels"] += perf_counter() - timing_t0
        else:
            timing_t0 = perf_counter()
            samples = build_feature_only_samples_for_history(
                hist,
                "closed_loop",
                eval_cfg,
                num_roots=feature_num_roots,
                num_options=feature_num_options,
            )
            if profile_timing:
                timing_totals["candidate_features"] += perf_counter() - timing_t0
        if not samples:
            break
        select_cfg = cfg
        if method != "ocrap":
            # Runtime-only post-contact clock for source controllers whose
            # published law is parameterised in seconds after impact.  The
            # dataset itself is unchanged and the same command line remains
            # valid for all external baselines.
            select_cfg = dict(cfg)
            sel_local = dict(select_cfg.get("selection", {}) or {}) if isinstance(select_cfg.get("selection", {}), dict) else {}
            contact_policy = (cfg.get("external_baselines", {}) or {}).get("policy", {}) if isinstance(cfg.get("external_baselines", {}), dict) else {}
            contact_dt = float(contact_policy.get("contact_dt", 0.1)) if isinstance(contact_policy, dict) else 0.1
            sel_local["runtime_contact_elapsed_s"] = float(step_idx) * contact_dt
            select_cfg["selection"] = sel_local
        if method == "ocrap":
            # Give the selector the active regime/bucket and the running
            # intervention count.  This is intentionally not written back into
            # the top-level config so each scene rollout remains independent.
            select_cfg = dict(cfg)
            sel_local = dict(select_cfg.get("selection", {}) or {}) if isinstance(select_cfg.get("selection", {}), dict) else {}
            active_bucket_name = _observable_regime_name(state, sdc, cfg, fallback=bucket_name or "")
            active_regime_trace.append(str(active_bucket_name))
            sel_local["active_bucket_name"] = active_bucket_name
            sel_local["intervention_budget_used"] = int(interventions_used)
            # Number of decisions considered before the current selection.
            # Use step_idx + 1 so the early rollout does not look artificially
            # over-budget after a single intervention.
            sel_local["intervention_budget_steps"] = max(1, int(step_idx) + 1)
            sel_local["steps_since_last_intervention"] = int(step_idx) - int(last_intervention_step)
            sel_local["previous_selected_macro"] = str(previous_selected_macro)
            sel_local["same_macro_run_length"] = int(same_macro_run_length)
            select_cfg["selection"] = sel_local
        if method != "ocrap":
            active_regime_trace.append(_observable_regime_name(state, sdc, cfg, fallback=bucket_name or ""))
        timing_t0 = perf_counter()
        sel_idx, info = _select_prefix(
            samples,
            bundle,
            select_cfg,
            method,
            gamma,
            compute_teacher_labels=compute_teacher_labels,
            external_model=external_model,
            external_model_cfg=external_model_cfg,
            external_device=external_device,
        )
        if profile_timing:
            timing_totals["policy_selection"] += perf_counter() - timing_t0
        selected_sample = samples[sel_idx]
        try:
            if int(getattr(selected_sample, "candidate_index", sel_idx)) != 0:
                interventions_used += 1
                last_intervention_step = int(step_idx)
        except Exception:
            if int(sel_idx) != 0:
                interventions_used += 1
                last_intervention_step = int(step_idx)
        prefix = selected_sample.prefix
        current_macro = str(prefix.macro_name)
        if current_macro == previous_selected_macro:
            same_macro_run_length += 1
        else:
            previous_selected_macro = current_macro
            same_macro_run_length = 1
        selected_audit_sample = None
        selected_audit_data = None
        selected_audit_drs = None
        selected_audit_fra_exec = None
        selected_audit_pcds = None
        audit_candidate_count = None
        audit_best_candidate_index = None
        audit_best_macro = None
        audit_best_teacher_r_dep = None
        audit_best_drs = None
        audit_best_pred_r_dep = None
        audit_best_pred_gap = None
        audit_best_pred_drs = None
        audit_best_hard = None
        audit_best_harm = None
        audit_selected_r_dep_regret = None
        audit_has_recoverable_candidate = None
        audit_selector_miss = None
        audit_best_pcd_candidate_index = None
        audit_best_pcd_macro = None
        audit_best_pcd = None
        audit_best_pcd_drs = None
        audit_best_pcd_teacher_r_dep = None
        audit_best_pcd_pred_r_dep = None
        audit_best_pcd_pred_gap = None
        audit_best_pcd_pred_drs = None
        audit_selected_pcd_regret = None
        audit_pcd_selector_miss = None
        audit_paper_best_pcd_macro = None
        audit_paper_best_pcd = None
        audit_paper_best_pcd_drs = None
        audit_paper_best_pcd_teacher_r_dep = None
        audit_paper_selected_pcd_regret = None
        audit_paper_pcd_selector_miss = None
        if (selected_label_audit or coverage_label_audit) and (step_idx % audit_every_n_steps == 0) and (audit_max_labels <= 0 or audit_labels_done < audit_max_labels):
            timing_t0 = perf_counter()
            try:
                audit_indices = ([int(selected_sample.candidate_index)] if selected_label_audit else _select_audit_candidate_indices(samples, info, selected_sample, cfg))
                labeled = build_labeled_samples_for_candidate_indices(
                    hist,
                    "closed_loop",
                    eval_cfg,
                    audit_indices,
                    num_roots=feature_num_roots,
                    num_options=feature_num_options,
                    prefixes=[s.prefix for s in samples],
                    recovery_options=samples[0].recovery_options if samples else None,
                    recovery_option_valid=samples[0].option_valid if samples else None,
                    assign_regime_labels=False,
                )
                if labeled:
                    by_cid = {int(s.candidate_index): s for s in labeled}
                    audit_data_by_cid = {int(s.candidate_index): _sample_to_audit_dict(s) for s in labeled}
                    selected_audit_sample = by_cid.get(int(selected_sample.candidate_index), labeled[0])
                    selected_audit_data = audit_data_by_cid[int(selected_audit_sample.candidate_index)]
                    pred_q = info["items"][sel_idx]["pred"].q
                    use_model_option = bool(method == "ocrap" and int(selected_sample.candidate_index) != 0)
                    q_eval = pred_q if use_model_option else selected_audit_data["m_star"]
                    opt_gamma = drs_gamma if use_model_option else 0.0
                    selected_option = best_option_indices(
                        q_eval,
                        selected_audit_data["root_probs"],
                        gamma=opt_gamma,
                        root_valid=selected_audit_data.get("root_valid", None),
                        option_valid=selected_audit_data.get("option_valid", None),
                        semantics=option_semantics,
                    )
                    selected_audit_drs = deployable_recovery_success(
                        selected_audit_data["m_star"],
                        selected_audit_data["root_probs"],
                        selected_option,
                        selected_audit_data.get("root_valid", None),
                    )
                    selected_r_dep_star = _safe_float(selected_audit_data.get("r_dep_star", 0.0))
                    selected_odg_star = _safe_float(selected_audit_data.get("oracle_gap_star", 0.0))
                    selected_audit_pcds = post_contact_deployability_score(selected_audit_drs, selected_r_dep_star, selected_odg_star)
                    selected_audit_fra_exec = float(selected_r_dep_star < 0.0)
                    audit_candidate_count = int(len(labeled))
                    # Coverage audit: among a small top-k subset, determine
                    # whether any alternative had better deployable headroom.
                    best_r = -float("inf")
                    best_drs = None
                    best_cid = None
                    best_pcd = -float("inf")
                    best_pcd_drs = None
                    best_pcd_r = None
                    best_pcd_cid = None
                    best_paper_pcd = -float("inf")
                    best_paper_pcd_drs = None
                    best_paper_pcd_r = None
                    best_paper_pcd_macro = None
                    item_by_cid = {int(getattr(item["sample"], "candidate_index", i)): (i, item) for i, item in enumerate(info.get("items", []))}
                    pred_q_by_cid = {cid: item["pred"].q for cid, (_, item) in item_by_cid.items()}
                    selected_pred_q = info["items"][sel_idx]["pred"].q
                    for lab in labeled:
                        cid = int(lab.candidate_index)
                        ld = audit_data_by_cid[cid]
                        macro_i = None
                        try:
                            macro_i = str(getattr(getattr(item_by_cid.get(cid, (None, {}))[1].get("sample", None), "prefix", None), "macro_name", "")) or None
                        except Exception:
                            try:
                                macro_i = str(getattr(getattr(lab, "prefix", None), "macro_name", "")) or None
                            except Exception:
                                macro_i = None
                        r_star = _safe_float(ld.get("r_dep_star", -float("inf")), -float("inf"))
                        pred_q_i = pred_q_by_cid.get(cid, selected_pred_q)
                        use_model_option_i = bool(method == "ocrap" and cid != 0)
                        q_eval_i = pred_q_i if use_model_option_i else ld["m_star"]
                        opt_gamma_i = drs_gamma if use_model_option_i else 0.0
                        opt_i = best_option_indices(
                            q_eval_i,
                            ld["root_probs"],
                            gamma=opt_gamma_i,
                            root_valid=ld.get("root_valid", None),
                            option_valid=ld.get("option_valid", None),
                            semantics=option_semantics,
                        )
                        drs_i = deployable_recovery_success(ld["m_star"], ld["root_probs"], opt_i, ld.get("root_valid", None))
                        odg_i = _safe_float(ld.get("oracle_gap_star", 0.0), 0.0)
                        pcd_i = post_contact_deployability_score(float(drs_i), float(r_star), float(odg_i))
                        # v23 paper-eligible audit upper bound.  The global top-k
                        # audit can choose exploratory families such as lane_shift.
                        # This upper bound keeps nominal plus the paper's
                        # recovery-eligible macro families only, so the reported
                        # miss rate matches the executable action set.
                        mi = str(macro_i or "").strip().lower()
                        paper_eligible = mi in {"nominal", "brake", "stabilize", "yield", "merge"}
                        if paper_eligible and pcd_i > best_paper_pcd:
                            best_paper_pcd = float(pcd_i)
                            best_paper_pcd_drs = float(drs_i)
                            best_paper_pcd_r = float(r_star)
                            best_paper_pcd_macro = macro_i
                        if r_star > best_r:
                            best_r = float(r_star)
                            best_drs = float(drs_i)
                            best_cid = int(cid)
                        if pcd_i > best_pcd:
                            best_pcd = float(pcd_i)
                            best_pcd_drs = float(drs_i)
                            best_pcd_r = float(r_star)
                            best_pcd_cid = int(cid)
                    if best_cid is not None and np.isfinite(best_r):
                        audit_best_candidate_index = int(best_cid)
                        audit_best_teacher_r_dep = float(best_r)
                        audit_best_drs = None if best_drs is None else float(best_drs)
                        bi_item = item_by_cid.get(int(best_cid), None)
                        if bi_item is not None:
                            bi, bit = bi_item
                            audit_best_macro = str(getattr(getattr(bit.get("sample"), "prefix", None), "macro_name", "")) or None
                            audit_best_pred_r_dep = _safe_optional_float(info.get("pred_r_dep", [])[bi])
                            audit_best_pred_gap = _safe_optional_float(info.get("pred_gap", [])[bi])
                            audit_best_pred_drs = _safe_optional_float(info.get("pred_drs", [])[bi])
                            audit_best_hard = _safe_optional_float(bit.get("data", {}).get("hard_violation", None))
                            audit_best_harm = _safe_optional_float(bit.get("data", {}).get("harm_proxy", None))
                        audit_selected_r_dep_regret = float(best_r - selected_r_dep_star)
                        audit_has_recoverable_candidate = bool(best_r >= 0.0)
                        audit_selector_miss = bool(selected_r_dep_star < 0.0 and best_r >= 0.0)
                    if best_pcd_cid is not None and np.isfinite(best_pcd):
                        audit_best_pcd_candidate_index = int(best_pcd_cid)
                        audit_best_pcd = float(best_pcd)
                        audit_best_pcd_drs = None if best_pcd_drs is None else float(best_pcd_drs)
                        audit_best_pcd_teacher_r_dep = None if best_pcd_r is None else float(best_pcd_r)
                        pcd_item = item_by_cid.get(int(best_pcd_cid), None)
                        if pcd_item is not None:
                            pi, pit = pcd_item
                            audit_best_pcd_macro = str(getattr(getattr(pit.get("sample"), "prefix", None), "macro_name", "")) or None
                            audit_best_pcd_pred_r_dep = _safe_optional_float(info.get("pred_r_dep", [])[pi])
                            audit_best_pcd_pred_gap = _safe_optional_float(info.get("pred_gap", [])[pi])
                            audit_best_pcd_pred_drs = _safe_optional_float(info.get("pred_drs", [])[pi])
                        audit_selected_pcd_regret = float(best_pcd - float(selected_audit_pcds))
                        pcd_eps = float(cl_cfg.get("audit_pcd_miss_epsilon", 0.02) or 0.02)
                        audit_pcd_selector_miss = bool(float(selected_audit_pcds) + pcd_eps < best_pcd)
                    if np.isfinite(best_paper_pcd):
                        audit_paper_best_pcd_macro = None if best_paper_pcd_macro is None else str(best_paper_pcd_macro)
                        audit_paper_best_pcd = float(best_paper_pcd)
                        audit_paper_best_pcd_drs = None if best_paper_pcd_drs is None else float(best_paper_pcd_drs)
                        audit_paper_best_pcd_teacher_r_dep = None if best_paper_pcd_r is None else float(best_paper_pcd_r)
                        audit_paper_selected_pcd_regret = float(best_paper_pcd - float(selected_audit_pcds))
                        pcd_eps = float(cl_cfg.get("audit_pcd_miss_epsilon", 0.02) or 0.02)
                        audit_paper_pcd_selector_miss = bool(float(selected_audit_pcds) + pcd_eps < best_paper_pcd)
                    audit_labels_done += int(len(labeled))
            except Exception as exc:
                if progress:
                    print({"event": "closed_loop_selected_label_audit_failed", "scene_rank": scenario_rank, "step": step_idx, "error": str(exc)}, flush=True)
            if profile_timing:
                timing_totals["audit_labels"] += perf_counter() - timing_t0
        controls = prefix.prefix_controls if prefix.prefix_controls.size else np.zeros((1, 4), dtype=np.float32)
        metrics_after: dict[str, float] = {}
        timing_t0 = perf_counter()
        for k in range(min(replan_interval, max(1, controls.shape[0]))):
            ctrl = controls[min(k, controls.shape[0] - 1)]
            action = _bicycle_action(int(state.num_objects), sdc, float(ctrl[0]), float(ctrl[1]), float(cfg.get("wheelbase_m", 2.8)))
            state = wx_env.step(state, action)
            metrics_after = _metric_summary(wx_env, state, sdc)
            metrics_after.update(_state_geometry_metrics(state, sdc))
            metric_trace.append(metrics_after)
            try:
                tr = state.sim_trajectory
                tt = _current_timestep(state)
                state_xy_trace.append([float(_as_np(tr.x)[sdc, tt]), float(_as_np(tr.y)[sdc, tt])])
            except Exception:
                pass
            if render_trace_enabled:
                try:
                    render_trace.append(
                        _capture_render_frame(
                            state, sdc, max_agents=render_max_agents,
                            selected_macro=str(prefix.macro_name),
                            selected_candidate_index=int(selected_sample.candidate_index),
                            selection_reason=str(info["selection"].reason),
                            metrics=metrics_after,
                        )
                    )
                except Exception as exc:
                    if progress:
                        print({"event": "closed_loop_render_trace_frame_failed", "scene_rank": scenario_rank, "step": step_idx, "error": str(exc)}, flush=True)
            if _scene_done(state):
                break
        if profile_timing:
            timing_totals["waymax_step_metrics"] += perf_counter() - timing_t0
        teacher_r_dep = info["teacher_r_dep"]
        teacher_r_orc = info["teacher_r_orc"]
        utility = info["utility"]
        selected_teacher_r_dep = _safe_optional_float(teacher_r_dep[sel_idx]) if compute_teacher_labels else (_safe_optional_float(selected_audit_data.get("r_dep_star")) if selected_audit_data is not None else None)
        selected_teacher_r_orc = _safe_optional_float(teacher_r_orc[sel_idx]) if compute_teacher_labels else (_safe_optional_float(selected_audit_data.get("r_orc_star")) if selected_audit_data is not None else None)
        selected_pred_r_dep = _safe_optional_float(info["pred_r_dep"][sel_idx])
        selected_pred_r_orc = _safe_optional_float(info["pred_r_orc"][sel_idx])
        selected_pred_gap = _safe_optional_float(info["pred_gap"][sel_idx])
        selected_pred_drs = _safe_optional_float(info["pred_drs"][sel_idx])
        direct_values = info.get("pred_direct_value", None)
        direct_stds = info.get("pred_direct_std", None)
        direct_opportunities = info.get("pred_direct_opportunity", None)
        selected_direct_recovery_value = _safe_optional_float(direct_values[sel_idx]) if direct_values is not None else None
        selected_direct_recovery_std = _safe_optional_float(direct_stds[sel_idx]) if direct_stds is not None else None
        selected_direct_recovery_opportunity = _safe_optional_float(direct_opportunities[sel_idx]) if direct_opportunities is not None else None
        nominal_direct_recovery_value = _safe_optional_float(direct_values[0]) if direct_values is not None and len(direct_values) else None
        direct_recovery_advantage = (selected_direct_recovery_value - nominal_direct_recovery_value) if selected_direct_recovery_value is not None and nominal_direct_recovery_value is not None else None
        selected_nominal_deviation = _safe_optional_float(info["nominal_deviation"][sel_idx])
        selected_odg = _safe_optional_float(selected_sample.oracle_gap_star) if compute_teacher_labels else (_safe_optional_float(selected_audit_data.get("oracle_gap_star")) if selected_audit_data is not None else None)
        selected_post_contact_deployability = selected_audit_pcds
        if selected_post_contact_deployability is None and compute_teacher_labels and info["drs"] is not None and selected_teacher_r_dep is not None and selected_odg is not None:
            selected_post_contact_deployability = post_contact_deployability_score(float(info["drs"]), float(selected_teacher_r_dep), float(selected_odg))
        selected_artifact = bool(selected_sample.i_art_star) if compute_teacher_labels else (bool(int(_safe_float(selected_audit_data.get("i_art_star", 0.0)))) if selected_audit_data is not None else None)
        decisions.append(
            ClosedLoopDecision(
                scene_id=str(raw.scenario_id),
                step_index=int(step_idx),
                time_index=int(t),
                method=str(method),
                selected_index=int(sel_idx),
                selected_macro=str(prefix.macro_name),
                selected_candidate_index=int(selected_sample.candidate_index),
                selection_reason=str(info["selection"].reason),
                selected_utility=float(utility[sel_idx]),
                selected_teacher_r_dep=selected_teacher_r_dep,
                selected_teacher_r_orc=selected_teacher_r_orc,
                selected_pred_r_dep=selected_pred_r_dep,
                selected_pred_r_orc=selected_pred_r_orc,
                selected_pred_gap=selected_pred_gap,
                selected_pred_drs=selected_pred_drs,
                selected_direct_recovery_value=selected_direct_recovery_value,
                selected_direct_recovery_std=selected_direct_recovery_std,
                selected_direct_recovery_opportunity=selected_direct_recovery_opportunity,
                nominal_direct_recovery_value=nominal_direct_recovery_value,
                direct_recovery_advantage=direct_recovery_advantage,
                selected_nominal_deviation=selected_nominal_deviation,
                selected_odg=selected_odg,
                selected_post_contact_deployability=selected_post_contact_deployability,
                selected_artifact=selected_artifact,
                audit_candidate_count=audit_candidate_count,
                audit_best_candidate_index=audit_best_candidate_index,
                audit_best_macro=audit_best_macro,
                audit_best_teacher_r_dep=audit_best_teacher_r_dep,
                audit_best_drs=audit_best_drs,
                audit_best_pred_r_dep=audit_best_pred_r_dep,
                audit_best_pred_gap=audit_best_pred_gap,
                audit_best_pred_drs=audit_best_pred_drs,
                audit_best_hard=audit_best_hard,
                audit_best_harm=audit_best_harm,
                audit_selected_r_dep_regret=audit_selected_r_dep_regret,
                audit_has_recoverable_candidate=audit_has_recoverable_candidate,
                audit_selector_miss=audit_selector_miss,
                audit_best_pcd_candidate_index=audit_best_pcd_candidate_index,
                audit_best_pcd_macro=audit_best_pcd_macro,
                audit_best_pcd=audit_best_pcd,
                audit_best_pcd_drs=audit_best_pcd_drs,
                audit_best_pcd_teacher_r_dep=audit_best_pcd_teacher_r_dep,
                audit_best_pcd_pred_r_dep=audit_best_pcd_pred_r_dep,
                audit_best_pcd_pred_gap=audit_best_pcd_pred_gap,
                audit_best_pcd_pred_drs=audit_best_pcd_pred_drs,
                audit_selected_pcd_regret=audit_selected_pcd_regret,
                audit_pcd_selector_miss=audit_pcd_selector_miss,
                audit_paper_best_pcd_macro=audit_paper_best_pcd_macro,
                audit_paper_best_pcd=audit_paper_best_pcd,
                audit_paper_best_pcd_drs=audit_paper_best_pcd_drs,
                audit_paper_best_pcd_teacher_r_dep=audit_paper_best_pcd_teacher_r_dep,
                audit_paper_selected_pcd_regret=audit_paper_selected_pcd_regret,
                audit_paper_pcd_selector_miss=audit_paper_pcd_selector_miss,
                fra_exec=(selected_audit_fra_exec if selected_audit_fra_exec is not None else (None if selected_teacher_r_dep is None else float(selected_teacher_r_dep < 0.0))),
                fra_cand=info["fra_cand"],
                drs=(None if selected_audit_drs is None else float(selected_audit_drs)) if (selected_label_audit or coverage_label_audit) else info["drs"],
                nup=float(info["nup"]),
                metrics_after_step=metrics_after,
            )
        )

    decision_dicts = [d.__dict__ for d in decisions]
    metric_names = sorted({k for m in metric_trace for k in m.keys()})
    metric_summary: dict[str, float] = {}
    for name in metric_names:
        vals = [float(m.get(name, 0.0)) for m in metric_trace if np.isfinite(float(m.get(name, 0.0)))]
        if vals:
            metric_summary[f"{name}_mean"] = float(np.mean(vals))
            metric_summary[f"{name}_max"] = float(np.max(vals))
            metric_summary[f"{name}_any"] = float(np.max(vals) > 0.0)
            if name in {"min_clearance_m", "ttc_s", "ego_speed_mps"}:
                metric_summary[f"{name}_min"] = float(np.min(vals))
                metric_summary[f"{name}_p05"] = float(np.quantile(vals, 0.05))
    clearance_vals = [float(m["min_clearance_m"]) for m in metric_trace if "min_clearance_m" in m and np.isfinite(float(m["min_clearance_m"]))]
    ttc_vals = [float(m["ttc_s"]) for m in metric_trace if "ttc_s" in m and np.isfinite(float(m["ttc_s"]))]
    speed_vals = [float(m["ego_speed_mps"]) for m in metric_trace if "ego_speed_mps" in m and np.isfinite(float(m["ego_speed_mps"]))]
    yaw_vals = [float(m["ego_yaw_rad"]) for m in metric_trace if "ego_yaw_rad" in m and np.isfinite(float(m["ego_yaw_rad"]))]
    # WOMD/Waymax motion states are sampled at 10 Hz by default.  Make the
    # interval configurable so the Safe paired report can publish comfort and
    # heading-rate statistics instead of silently omitting them.
    metric_dt_s = float(cl_cfg.get("metric_dt_s", 0.1) or 0.1)
    metric_dt_s = max(metric_dt_s, 1.0e-3)
    acceleration = np.asarray([], dtype=np.float64)
    jerk = np.asarray([], dtype=np.float64)
    if len(speed_vals) >= 2:
        acceleration = np.diff(np.asarray(speed_vals, dtype=np.float64)) / metric_dt_s
        metric_summary["acceleration_abs_p95_mps2"] = float(np.quantile(np.abs(acceleration), 0.95)) if acceleration.size else 0.0
        metric_summary["acceleration_max_mps2"] = float(np.max(acceleration)) if acceleration.size else 0.0
        metric_summary["deceleration_max_mps2"] = float(max(0.0, -np.min(acceleration))) if acceleration.size else 0.0
    if acceleration.size >= 2:
        jerk = np.diff(acceleration) / metric_dt_s
        metric_summary["jerk_p95"] = float(np.quantile(np.abs(jerk), 0.95)) if jerk.size else 0.0
        metric_summary["jerk_max_abs"] = float(np.max(np.abs(jerk))) if jerk.size else 0.0
    yaw_rate = np.asarray([], dtype=np.float64)
    if len(yaw_vals) >= 2:
        yaw_unwrapped = np.unwrap(np.asarray(yaw_vals, dtype=np.float64))
        yaw_rate = np.diff(yaw_unwrapped) / metric_dt_s
        metric_summary["yaw_rate_p95"] = float(np.quantile(np.abs(yaw_rate), 0.95)) if yaw_rate.size else 0.0
        metric_summary["yaw_rate_max_abs"] = float(np.max(np.abs(yaw_rate))) if yaw_rate.size else 0.0
    overlap_flags = [bool(float(m.get("overlap", 0.0)) > 0.0) for m in metric_trace]
    offroad_flags = [bool(float(m.get("offroad", 0.0)) > 0.0) for m in metric_trace]

    def _binary_run_stats(flags: list[bool]) -> tuple[int, int]:
        episodes = 0
        longest = 0
        run = 0
        for i, flag in enumerate(flags + [False]):
            if flag:
                if i == 0 or not flags[i - 1]:
                    episodes += 1
                run += 1
            else:
                longest = max(longest, run)
                run = 0
        return episodes, longest
    overlap_episode_count = int(sum(flag and (i == 0 or not overlap_flags[i - 1]) for i, flag in enumerate(overlap_flags)))
    metric_steps = int(len(metric_trace))
    near_count = int(sum(c <= 2.0 for c in clearance_vals))
    critical_ttc_count = int(sum(t <= 3.0 for t in ttc_vals))
    near_zero_clearance_count = int(sum(c <= 0.05 for c in clearance_vals))
    metric_summary["num_metric_steps"] = metric_steps
    metric_summary["near_contact_exposure_count"] = near_count
    metric_summary["critical_ttc_exposure_count"] = critical_ttc_count
    metric_summary["near_zero_clearance_exposure_count"] = near_zero_clearance_count
    metric_summary["near_contact_exposure_rate"] = float(near_count / max(len(clearance_vals), 1)) if clearance_vals else 0.0
    metric_summary["critical_ttc_exposure_rate"] = float(critical_ttc_count / max(len(ttc_vals), 1)) if ttc_vals else 0.0
    metric_summary["near_contact_exposure_duration_s"] = float(near_count * metric_dt_s)
    metric_summary["critical_ttc_exposure_duration_s"] = float(critical_ttc_count * metric_dt_s)
    near_flags = [bool(c <= 2.0) for c in clearance_vals]
    critical_ttc_flags = [bool(t <= 3.0) for t in ttc_vals]
    near_episodes, near_longest = _binary_run_stats(near_flags)
    ttc_episodes, ttc_longest = _binary_run_stats(critical_ttc_flags)
    metric_summary["near_contact_exposure_episode_count"] = float(near_episodes)
    metric_summary["near_contact_longest_exposure_run_steps"] = float(near_longest)
    metric_summary["near_contact_longest_exposure_run_s"] = float(near_longest * metric_dt_s)
    metric_summary["critical_ttc_exposure_episode_count"] = float(ttc_episodes)
    metric_summary["critical_ttc_longest_exposure_run_steps"] = float(ttc_longest)
    metric_summary["critical_ttc_longest_exposure_run_s"] = float(ttc_longest * metric_dt_s)
    if clearance_vals:
        min_clearance_idx = int(np.argmin(np.asarray(clearance_vals)))
        metric_summary["time_to_min_clearance_steps"] = float(min_clearance_idx)
        metric_summary["time_to_min_clearance_s"] = float(min_clearance_idx * metric_dt_s)
        metric_summary["terminal_clearance_m"] = float(clearance_vals[-1])
        metric_summary["clearance_recovery_gain_m"] = float(clearance_vals[-1] - clearance_vals[min_clearance_idx])
    else:
        metric_summary["time_to_min_clearance_steps"] = float("nan")
        metric_summary["time_to_min_clearance_s"] = float("nan")
        metric_summary["terminal_clearance_m"] = float("nan")
        metric_summary["clearance_recovery_gain_m"] = float("nan")
    if ttc_vals:
        min_ttc_idx = int(np.argmin(np.asarray(ttc_vals)))
        metric_summary["time_to_min_ttc_steps"] = float(min_ttc_idx)
        metric_summary["time_to_min_ttc_s"] = float(min_ttc_idx * metric_dt_s)
        metric_summary["terminal_ttc_s"] = float(ttc_vals[-1])
        metric_summary["ttc_recovery_gain_s"] = float(ttc_vals[-1] - ttc_vals[min_ttc_idx])
    else:
        metric_summary["time_to_min_ttc_steps"] = float("nan")
        metric_summary["time_to_min_ttc_s"] = float("nan")
        metric_summary["terminal_ttc_s"] = float("nan")
        metric_summary["ttc_recovery_gain_s"] = float("nan")
    # Continuous margin-deficit integrals are less brittle than a single minimum
    # and distinguish a brief close pass from sustained unsafe proximity.
    metric_summary["clearance_deficit_auc_m_s"] = float(
        sum(max(0.0, 2.0 - c) for c in clearance_vals) * metric_dt_s
    )
    metric_summary["ttc_deficit_auc_s2"] = float(
        sum(max(0.0, 3.0 - t) for t in ttc_vals) * metric_dt_s
    )
    # A radius-based clearance <= 5 cm is not equivalent to simulator contact.
    # Keep the legacy key for compatibility, but publish the unambiguous name.
    metric_summary["near_zero_clearance_exposure_rate"] = float(near_zero_clearance_count / max(len(clearance_vals), 1)) if clearance_vals else 0.0
    metric_summary["contact_exposure_rate"] = metric_summary["near_zero_clearance_exposure_rate"]
    metric_summary["overlap_episode_count"] = overlap_episode_count
    metric_summary["secondary_overlap_event"] = float(overlap_episode_count >= 2)
    overlap_duration_steps = int(sum(overlap_flags))
    longest_overlap_run_steps = 0
    overlap_run = 0
    for flag in overlap_flags + [False]:
        if flag:
            overlap_run += 1
        else:
            longest_overlap_run_steps = max(longest_overlap_run_steps, overlap_run)
            overlap_run = 0
    metric_summary["overlap_duration_steps"] = float(overlap_duration_steps)
    metric_summary["overlap_duration_s"] = float(overlap_duration_steps * metric_dt_s)
    metric_summary["longest_overlap_run_steps"] = float(longest_overlap_run_steps)
    metric_summary["longest_overlap_run_s"] = float(longest_overlap_run_steps * metric_dt_s)

    # Contact-only escape-space diagnostics.  These are computed from observable
    # closed-loop geometry, not from the offline PCD teacher.  A sustained escape
    # requires three consecutive non-overlap steps above a configurable clearance.
    aligned_clearance = [
        float(m.get("min_clearance_m", float("nan"))) for m in metric_trace
    ]
    observed_first_contact_idx = next((i for i, flag in enumerate(overlap_flags) if flag), None)
    is_post_contact_target = _is_post_contact_bucket_name(bucket_name)
    # Targeted post-contact rollouts may start after the initiating collision. In
    # that case step 0 is the causal contact anchor even when the first simulated
    # state is already separated. Any later overlap episode is a re-contact.
    contact_anchor_idx = (
        0 if is_post_contact_target else observed_first_contact_idx
    )
    metric_summary["first_contact_step"] = (
        float(observed_first_contact_idx) if observed_first_contact_idx is not None else float("nan")
    )
    metric_summary["contact_anchor_step"] = (
        float(contact_anchor_idx) if contact_anchor_idx is not None else float("nan")
    )
    overlap_starts = [i for i, flag in enumerate(overlap_flags) if flag and (i == 0 or not overlap_flags[i - 1])]
    if is_post_contact_target:
        recontact_starts = [i for i in overlap_starts if i > 0 or not overlap_flags[0]]
    elif observed_first_contact_idx is not None:
        recontact_starts = overlap_starts[1:]
    else:
        recontact_starts = []
    metric_summary["recontact_episode_count"] = float(len(recontact_starts))
    metric_summary["recontact_event"] = float(bool(recontact_starts))
    metric_summary["secondary_overlap_event"] = metric_summary["recontact_event"]
    if contact_anchor_idx is not None:
        post_indices = [
            i for i in range(contact_anchor_idx, len(metric_trace))
            if np.isfinite(aligned_clearance[i])
        ]
        post_clearance = [aligned_clearance[i] for i in post_indices]
        post_overlap_flags = overlap_flags[contact_anchor_idx:]
        metric_summary["post_contact_overlap_duration_s"] = float(
            sum(bool(x) for x in post_overlap_flags) * metric_dt_s
        )
        metric_summary["post_contact_overlap_rate"] = float(
            sum(bool(x) for x in post_overlap_flags) / max(len(post_overlap_flags), 1)
        )
        metric_summary["post_contact_clearance_m_max"] = (
            float(max(post_clearance)) if post_clearance else float("nan")
        )
        metric_summary["post_contact_clearance_m_mean"] = (
            float(np.mean(post_clearance)) if post_clearance else float("nan")
        )
        metric_summary["post_contact_free_space_auc_m_s"] = float(
            sum(max(0.0, c) for c in post_clearance) * metric_dt_s
        )
        post_duration_s = max(metric_dt_s, len(post_clearance) * metric_dt_s)
        metric_summary["post_contact_free_space_auc_normalized_m"] = float(
            metric_summary["post_contact_free_space_auc_m_s"] / post_duration_s
        )
        contact_clearance_target = float(cl_cfg.get("post_contact_clearance_target_m", 0.5) or 0.5)
        metric_summary["post_contact_clearance_deficit_auc_m_s"] = float(
            sum(max(0.0, contact_clearance_target - c) for c in post_clearance) * metric_dt_s
        )
        metric_summary["post_contact_terminal_clearance_m"] = (
            float(post_clearance[-1]) if post_clearance else float("nan")
        )
        if post_clearance:
            peak_local = int(np.argmax(np.asarray(post_clearance)))
            metric_summary["time_to_peak_post_contact_clearance_s"] = float(peak_local * metric_dt_s)
            metric_summary["post_contact_clearance_gain_m"] = float(post_clearance[-1] - post_clearance[0])
        else:
            metric_summary["time_to_peak_post_contact_clearance_s"] = float("nan")
            metric_summary["post_contact_clearance_gain_m"] = float("nan")
        escape_clearance = float(cl_cfg.get("post_contact_escape_clearance_m", 0.5) or 0.5)
        escape_sustain_steps = max(1, int(cl_cfg.get("post_contact_escape_sustain_steps", 3) or 3))
        escape_idx = None
        for end in range(contact_anchor_idx + escape_sustain_steps - 1, len(metric_trace)):
            begin = end - escape_sustain_steps + 1
            window_clearance = aligned_clearance[begin : end + 1]
            if (
                all(np.isfinite(c) and c >= escape_clearance for c in window_clearance)
                and not any(overlap_flags[begin : end + 1])
            ):
                escape_idx = begin
                break
        metric_summary["post_contact_escape_event"] = float(escape_idx is not None)
        metric_summary["time_to_post_contact_escape_steps"] = (
            float(escape_idx - contact_anchor_idx) if escape_idx is not None else float("nan")
        )
        metric_summary["time_to_post_contact_escape_s"] = (
            float((escape_idx - contact_anchor_idx) * metric_dt_s)
            if escape_idx is not None else float("nan")
        )
    else:
        metric_summary["post_contact_overlap_duration_s"] = float("nan")
        metric_summary["post_contact_overlap_rate"] = float("nan")
        metric_summary["post_contact_clearance_m_max"] = float("nan")
        metric_summary["post_contact_clearance_m_mean"] = float("nan")
        metric_summary["post_contact_free_space_auc_m_s"] = float("nan")
        metric_summary["post_contact_free_space_auc_normalized_m"] = float("nan")
        metric_summary["post_contact_clearance_deficit_auc_m_s"] = float("nan")
        metric_summary["post_contact_terminal_clearance_m"] = float("nan")
        metric_summary["time_to_peak_post_contact_clearance_s"] = float("nan")
        metric_summary["post_contact_clearance_gain_m"] = float("nan")
        metric_summary["post_contact_escape_event"] = 0.0
        metric_summary["time_to_post_contact_escape_steps"] = float("nan")
        metric_summary["time_to_post_contact_escape_s"] = float("nan")
    stable_sustain_steps = max(3, int(cl_cfg.get("stable_stop_sustain_steps", 5) or 5))
    stable_speed_mps = float(cl_cfg.get("stable_stop_speed_mps", 0.5) or 0.5)
    stable_yaw_rate = float(cl_cfg.get("stable_stop_yaw_rate_radps", 0.25) or 0.25)
    yaw_rate_aligned = [0.0] + np.abs(yaw_rate).tolist() if yaw_rate.size else [0.0] * len(speed_vals)
    tail_speeds = speed_vals[-stable_sustain_steps:] if speed_vals else []
    tail_overlaps = overlap_flags[-stable_sustain_steps:] if overlap_flags else []
    tail_offroad = offroad_flags[-stable_sustain_steps:] if offroad_flags else []
    tail_yaw_rate = yaw_rate_aligned[-stable_sustain_steps:] if yaw_rate_aligned else []
    stable_tail = (
        len(tail_speeds) >= stable_sustain_steps
        and max(tail_speeds) <= stable_speed_mps
        and not any(tail_overlaps)
    )
    stable_tail_quality = bool(
        stable_tail and not any(tail_offroad)
        and (not tail_yaw_rate or max(tail_yaw_rate) <= stable_yaw_rate)
    )
    initial_moving = bool(speed_vals) and max(speed_vals[: min(stable_sustain_steps, len(speed_vals))]) > stable_speed_mps
    metric_summary["stable_stop_event"] = float(stable_tail)
    metric_summary["stable_stop_quality_event"] = float(stable_tail_quality)
    metric_summary["stable_stop_eligible"] = float(initial_moving)
    metric_summary["new_stable_stop_event"] = float(initial_moving and stable_tail)
    metric_summary["new_stable_stop_quality_event"] = float(initial_moving and stable_tail_quality)
    stable_idx = None
    stable_quality_idx = None
    if initial_moving and len(speed_vals) >= stable_sustain_steps:
        for j in range(stable_sustain_steps - 1, len(speed_vals)):
            begin = j - stable_sustain_steps + 1
            speed_ok = max(speed_vals[begin : j + 1]) <= stable_speed_mps
            overlap_ok = not any(overlap_flags[begin : j + 1])
            if stable_idx is None and speed_ok and overlap_ok:
                stable_idx = begin
            offroad_ok = not any(offroad_flags[begin : j + 1])
            yaw_ok = max(yaw_rate_aligned[begin : j + 1]) <= stable_yaw_rate
            if speed_ok and overlap_ok and offroad_ok and yaw_ok:
                stable_quality_idx = begin
                break
    metric_summary["time_to_stable_stop_steps"] = float(stable_idx) if stable_idx is not None else float("nan")
    metric_summary["time_to_stable_stop_s"] = float(stable_idx * metric_dt_s) if stable_idx is not None else float("nan")
    metric_summary["time_to_stable_stop_quality_s"] = (
        float(stable_quality_idx * metric_dt_s) if stable_quality_idx is not None else float("nan")
    )
    intervention_flags = [bool(d.selected_candidate_index != 0) for d in decisions]
    intervention_episode_count = int(sum(flag and (i == 0 or not intervention_flags[i - 1]) for i, flag in enumerate(intervention_flags)))
    intervention_run_lengths: list[int] = []
    run_len = 0
    for flag in intervention_flags + [False]:
        if flag:
            run_len += 1
        elif run_len > 0:
            intervention_run_lengths.append(run_len)
            run_len = 0
    macro_switch_count = int(sum(decisions[i].selected_macro != decisions[i - 1].selected_macro for i in range(1, len(decisions))))
    wall_s = perf_counter() - scene_wall_t0
    measured_s = float(sum(timing_totals.values()))
    route_progression = _route_progression_from_trace(
        state_xy_trace, route_reference_global
    )
    if route_progression is not None:
        metric_summary["route_progression_m"] = float(route_progression)

    timing_summary = {
        "enabled": bool(profile_timing),
        "wall_s": float(wall_s),
        "measured_s": measured_s,
        "other_overhead_s": float(max(0.0, wall_s - measured_s)),
        "totals_s": {k: float(v) for k, v in timing_totals.items()},
        "per_decision_s": {k: float(v / max(len(decisions), 1)) for k, v in timing_totals.items()},
    }
    out = {
        "scene_id": str(raw.scenario_id),
        "bucket_name": bucket_name,
        "canonical_regime": canonical_regime_name(bucket_name),
        "bucket_aliases": _bucket_gamma_aliases(bucket_name),
        "post_contact_target": bool(is_post_contact_target),
        "target_key": target_key,
        "target_time_index": int(start_time_index_override) if start_time_index_override is not None else None,
        "route_progression": route_progression,
        "route_progression_source": route_reference_source,
        "num_decisions": int(len(decisions)),
        "num_metric_steps": int(len(metric_trace)),
        "method": method,
        "gamma_rec": float(gamma),
        "label_mode": label_mode,
        "labels_available": bool(compute_teacher_labels or selected_label_audit or coverage_label_audit),
        "external_sparse_labels": bool(external_sparse_labels),
        "external_sparse_label_decisions": int(sparse_label_decisions),
        "external_sparse_label_candidates_mean": (float(sparse_label_candidates_total) / max(float(sparse_label_decisions), 1.0)) if sparse_label_decisions else None,
        "external_sparse_full_candidates_mean": (float(sparse_label_full_candidates_total) / max(float(sparse_label_decisions), 1.0)) if sparse_label_decisions else None,
        "selected_label_audit": bool(selected_label_audit or coverage_label_audit),
        "coverage_label_audit": bool(coverage_label_audit),
        "audit_every_n_steps": int(audit_every_n_steps),
        "audit_labels_done": int(audit_labels_done),
        "audit_max_labels": int(audit_max_labels),
        "audit_auto_capped": bool(audit_auto_capped),
        "closed_loop_FRA_exec": _mean_finite([d.fra_exec for d in decisions]),
        "closed_loop_FRA_cand": _mean_finite([d.fra_cand for d in decisions]),
        "closed_loop_DRS": _mean_finite([d.drs for d in decisions]),
        "closed_loop_ODG": _mean_finite([d.selected_odg for d in decisions]),
        "closed_loop_post_contact_deployability": _mean_finite([d.selected_post_contact_deployability for d in decisions]),
        "closed_loop_artifact_selection_rate": _mean_finite([float(d.selected_artifact) for d in decisions if d.selected_artifact is not None]),
        "closed_loop_audit_candidate_count": _mean_finite([d.audit_candidate_count for d in decisions]),
        "closed_loop_audit_best_R_dep": _mean_finite([d.audit_best_teacher_r_dep for d in decisions]),
        "closed_loop_audit_best_DRS": _mean_finite([d.audit_best_drs for d in decisions]),
        "closed_loop_audit_selected_R_dep_regret": _mean_finite([d.audit_selected_r_dep_regret for d in decisions]),
        "closed_loop_audit_best_PCD": _mean_finite([d.audit_best_pcd for d in decisions]),
        "closed_loop_audit_selected_PCD_regret": _mean_finite([d.audit_selected_pcd_regret for d in decisions]),
        "closed_loop_audit_recoverable_candidate_rate": _mean_finite([float(d.audit_has_recoverable_candidate) for d in decisions if d.audit_has_recoverable_candidate is not None]),
        "closed_loop_audit_selector_miss_rate": _mean_finite([float(d.audit_selector_miss) for d in decisions if d.audit_selector_miss is not None]),
        "closed_loop_audit_pcd_selector_miss_rate": _mean_finite([float(d.audit_pcd_selector_miss) for d in decisions if d.audit_pcd_selector_miss is not None]),
        "closed_loop_audit_paper_best_PCD": _mean_finite([d.audit_paper_best_pcd for d in decisions]),
        "closed_loop_audit_paper_selected_PCD_regret": _mean_finite([d.audit_paper_selected_pcd_regret for d in decisions]),
        "closed_loop_audit_paper_pcd_selector_miss_rate": _mean_finite([float(d.audit_paper_pcd_selector_miss) for d in decisions if d.audit_paper_pcd_selector_miss is not None]),
        "closed_loop_bounded_NUP": _mean_finite([d.nup for d in decisions], default=0.0),
        "closed_loop_pred_r_dep": _mean_finite([d.selected_pred_r_dep for d in decisions]),
        "closed_loop_pred_gap": _mean_finite([d.selected_pred_gap for d in decisions]),
        "closed_loop_pred_DRS_proxy": _mean_finite([d.selected_pred_drs for d in decisions]),
        "closed_loop_direct_recovery_value": _mean_finite([d.selected_direct_recovery_value for d in decisions]),
        "closed_loop_direct_recovery_std": _mean_finite([d.selected_direct_recovery_std for d in decisions]),
        "closed_loop_direct_recovery_opportunity": _mean_finite([d.selected_direct_recovery_opportunity for d in decisions]),
        "closed_loop_direct_recovery_advantage": _mean_finite([d.direct_recovery_advantage for d in decisions]),
        "closed_loop_nominal_deviation": _mean_finite([d.selected_nominal_deviation for d in decisions]),
        "active_regime_counts": {name: int(active_regime_trace.count(name)) for name in sorted(set(active_regime_trace))},
        "intervention_rate": _mean_finite([float(d.selected_candidate_index != 0) for d in decisions], default=0.0),
        "intervention_episode_count": intervention_episode_count,
        "intervention_episode_rate": float(intervention_episode_count / max(len(decisions), 1)),
        "mean_intervention_run_length": float(np.mean(intervention_run_lengths)) if intervention_run_lengths else 0.0,
        "max_intervention_run_length": int(max(intervention_run_lengths)) if intervention_run_lengths else 0,
        "macro_switch_rate": float(macro_switch_count / max(len(decisions) - 1, 1)),
        "metric_summary": metric_summary,
        "macro_counts": {m: int(sum(d.selected_macro == m for d in decisions)) for m in sorted({d.selected_macro for d in decisions})},
        "audit_best_macro_counts": {m: int(sum(d.audit_best_macro == m for d in decisions)) for m in sorted({d.audit_best_macro for d in decisions if d.audit_best_macro is not None})},
        "audit_miss_best_macro_counts": {m: int(sum((d.audit_selector_miss is True) and d.audit_best_macro == m for d in decisions)) for m in sorted({d.audit_best_macro for d in decisions if d.audit_best_macro is not None})},
        "audit_miss_selected_macro_counts": {m: int(sum((d.audit_selector_miss is True) and d.selected_macro == m for d in decisions)) for m in sorted({d.selected_macro for d in decisions})},
        "audit_pcd_best_macro_counts": {m: int(sum(d.audit_best_pcd_macro == m for d in decisions)) for m in sorted({d.audit_best_pcd_macro for d in decisions if d.audit_best_pcd_macro is not None})},
        "audit_pcd_miss_best_macro_counts": {m: int(sum((d.audit_pcd_selector_miss is True) and d.audit_best_pcd_macro == m for d in decisions)) for m in sorted({d.audit_best_pcd_macro for d in decisions if d.audit_best_pcd_macro is not None})},
        "audit_paper_pcd_best_macro_counts": {m: int(sum(d.audit_paper_best_pcd_macro == m for d in decisions)) for m in sorted({d.audit_paper_best_pcd_macro for d in decisions if d.audit_paper_best_pcd_macro is not None})},
        "audit_paper_pcd_miss_best_macro_counts": {m: int(sum((d.audit_paper_pcd_selector_miss is True) and d.audit_paper_best_pcd_macro == m for d in decisions)) for m in sorted({d.audit_paper_best_pcd_macro for d in decisions if d.audit_paper_best_pcd_macro is not None})},
        "selection_reason_counts": {r: int(sum(d.selection_reason == r for d in decisions)) for r in sorted({d.selection_reason for d in decisions})},
        "timing": timing_summary,
        "decisions": decision_dicts,
    }
    if bool(cl_cfg.get("save_trace_npz", False)):
        out["state_xy_trace"] = state_xy_trace
    if render_trace_enabled:
        out["render_trace"] = render_trace
        if render_context:
            out["render_context"] = render_context
        out["render_trace_schema"] = "ocrap.v50.agent_boxes_roadgraph.v2"
    return out

_HEAVY_SCENE_STORAGE_KEYS = {
    "decisions",
    "render_trace",
    "render_context",
    "render_trace_schema",
    "state_xy_trace",
}


def _normalize_scene_storage_detail(value: Any, *, default: str = "full") -> str:
    detail = str(value or default).strip().lower().replace("-", "_")
    aliases = {"compact": "metrics", "summary": "metrics", "metric": "metrics", "trace": "full"}
    detail = aliases.get(detail, detail)
    if detail not in {"full", "metrics"}:
        raise ValueError(f"closed_loop scene storage detail must be 'full' or 'metrics', got {value!r}")
    return detail


def _scene_storage_view(scene: dict[str, Any], detail: str) -> dict[str, Any]:
    """Return a storage-safe scene record without changing evaluated metrics.

    ``metrics`` removes step-by-step decisions and render payloads. All scalar
    scene metrics, target identity, intervention summaries, macro counts and
    timing information remain available for aggregation, paired tables and
    qualitative-scene selection. Selected video reruns use ``full`` journals.
    """
    detail = _normalize_scene_storage_detail(detail)
    if detail == "full":
        return scene
    compact = {
        key: value
        for key, value in scene.items()
        if key not in _HEAVY_SCENE_STORAGE_KEYS and not str(key).endswith("_trace")
    }
    compact["scene_storage_detail"] = "metrics"
    return compact


def _aggregate_scene_results(scene_results: list[dict[str, Any]], method: str, source: str) -> dict[str, Any]:
    keys = ["closed_loop_FRA_exec", "closed_loop_FRA_cand", "closed_loop_DRS", "closed_loop_ODG", "closed_loop_post_contact_deployability", "closed_loop_artifact_selection_rate", "closed_loop_audit_candidate_count", "closed_loop_audit_best_R_dep", "closed_loop_audit_best_DRS", "closed_loop_audit_selected_R_dep_regret", "closed_loop_audit_best_PCD", "closed_loop_audit_selected_PCD_regret", "closed_loop_audit_recoverable_candidate_rate", "closed_loop_audit_selector_miss_rate", "closed_loop_audit_pcd_selector_miss_rate", "closed_loop_audit_paper_best_PCD", "closed_loop_audit_paper_selected_PCD_regret", "closed_loop_audit_paper_pcd_selector_miss_rate", "closed_loop_bounded_NUP", "closed_loop_pred_r_dep", "closed_loop_pred_gap", "closed_loop_pred_DRS_proxy", "closed_loop_direct_recovery_value", "closed_loop_direct_recovery_std", "closed_loop_direct_recovery_advantage", "closed_loop_nominal_deviation", "intervention_rate", "intervention_episode_rate", "mean_intervention_run_length", "max_intervention_run_length", "macro_switch_rate", "external_sparse_label_candidates_mean", "external_sparse_full_candidates_mean"]
    agg: dict[str, Any] = {
        "source": source,
        "method": method,
        "num_scenes": int(len(scene_results)),
        "num_decisions": int(sum(int(s.get("num_decisions", 0)) for s in scene_results)),
        "num_metric_steps": int(sum(int(s.get("num_metric_steps", 0)) for s in scene_results)),
        "label_modes": sorted({str(s.get("label_mode", "unknown")) for s in scene_results}),
        "metrics_valid": bool(scene_results),
        "empty_reason": None if scene_results else "no_closed_loop_scenes",
        "runtime_contract": {
            "bucket_names": sorted({str(s.get("bucket_name", "")) for s in scene_results}),
            "canonical_regimes": sorted({str(s.get("canonical_regime", "")) for s in scene_results}),
            "gamma_rec_values": sorted({float(s.get("gamma_rec", 0.0)) for s in scene_results}),
            "post_contact_target_count": int(sum(bool(s.get("post_contact_target", False)) for s in scene_results)),
        },
    }
    for k in keys:
        vals = [s.get(k, None) for s in scene_results if int(s.get("num_decisions", 0)) > 0]
        if k == "max_intervention_run_length":
            finite = [float(v) for v in vals if v is not None and np.isfinite(float(v))]
            agg[k] = int(max(finite, default=0.0))
        elif k == "mean_intervention_run_length":
            total_interventions = sum(int(round(float(s.get("intervention_rate", 0.0)) * int(s.get("num_decisions", 0)))) for s in scene_results)
            total_episodes = sum(int(s.get("intervention_episode_count", 0)) for s in scene_results)
            agg[k] = float(total_interventions / max(total_episodes, 1)) if total_episodes else 0.0
        else:
            agg[k] = _mean_finite(vals)
    agg["intervention_episode_count"] = int(sum(int(s.get("intervention_episode_count", 0)) for s in scene_results))
    agg["intervention_scene_rate"] = float(np.mean([int(s.get("intervention_episode_count", 0)) > 0 for s in scene_results])) if scene_results else 0.0
    metric_names = sorted({mk for s in scene_results for mk in (s.get("metric_summary", {}) or {}).keys()})
    agg["waymax_metrics"] = {}
    for mk in metric_names:
        pairs = [((s.get("metric_summary", {}) or {}).get(mk, None), int(s.get("num_metric_steps", 0))) for s in scene_results]
        finite = [(float(v), w) for v, w in pairs if v is not None and np.isfinite(float(v))]
        if not finite:
            agg["waymax_metrics"][mk] = None
        elif mk.endswith("_count") or mk in {"overlap_episode_count", "num_metric_steps"}:
            agg["waymax_metrics"][mk] = float(sum(v for v, _ in finite))
        elif mk.endswith("_any"):
            agg["waymax_metrics"][mk] = float(max(v for v, _ in finite))
        elif mk.endswith("_max") or "_max_" in mk:
            agg["waymax_metrics"][mk] = float(max(v for v, _ in finite))
        elif mk.endswith("_min") or "_min_" in mk:
            agg["waymax_metrics"][mk] = float(min(v for v, _ in finite))
        elif mk.endswith("_rate"):
            denom = sum(max(w, 0) for _, w in finite)
            agg["waymax_metrics"][mk] = float(sum(v * max(w, 0) for v, w in finite) / max(denom, 1))
        else:
            agg["waymax_metrics"][mk] = float(np.mean([v for v, _ in finite]))
    wm = agg["waymax_metrics"]
    def _scene_mean(name: str) -> float:
        vals = [float((sc.get("metric_summary", {}) or {}).get(name)) for sc in scene_results
                if (sc.get("metric_summary", {}) or {}).get(name) is not None
                and np.isfinite(float((sc.get("metric_summary", {}) or {}).get(name)))]
        return float(np.mean(vals)) if vals else None
    def _step_weighted(name: str) -> float:
        vals = [(float((sc.get("metric_summary", {}) or {}).get(name)), int(sc.get("num_metric_steps", 0)))
                for sc in scene_results if (sc.get("metric_summary", {}) or {}).get(name) is not None
                and np.isfinite(float((sc.get("metric_summary", {}) or {}).get(name)))]
        den = sum(max(w, 0) for _, w in vals)
        return float(sum(v * max(w, 0) for v, w in vals) / max(den, 1)) if vals and den > 0 else None
    agg["collision_scene_rate"] = _scene_mean("overlap_any")
    agg["collision_step_rate"] = _step_weighted("overlap_mean")
    agg["offroad_scene_rate"] = _scene_mean("offroad_any")
    agg["offroad_step_rate"] = _step_weighted("offroad_mean")
    # Event-valued contact metrics are Bernoulli scene outcomes.  Aggregate
    # them as scene rates rather than ``any scene triggered`` maxima.
    agg["secondary_overlap_scene_rate"] = _scene_mean("secondary_overlap_event")
    agg["recontact_scene_rate"] = _scene_mean("recontact_event")
    agg["new_stable_stop_scene_rate"] = _scene_mean("new_stable_stop_event")
    agg["new_stable_stop_quality_scene_rate"] = _scene_mean("new_stable_stop_quality_event")
    agg["post_contact_escape_scene_rate"] = _scene_mean("post_contact_escape_event")
    agg["minimum_clearance_m"] = wm.get("min_clearance_m_min")
    agg["minimum_ttc_s"] = wm.get("ttc_s_min")
    # Distribution across scenes is the publication-level unit; do not average
    # per-scene p05 values and call it a global p05.
    for base in ("min_clearance_m", "ttc_s"):
        vals = [float((s.get("metric_summary", {}) or {}).get(f"{base}_min")) for s in scene_results if (s.get("metric_summary", {}) or {}).get(f"{base}_min") is not None and np.isfinite(float((s.get("metric_summary", {}) or {}).get(f"{base}_min")))]
        if vals:
            agg["waymax_metrics"][f"scene_{base}_median"] = float(np.median(vals))
            agg["waymax_metrics"][f"scene_{base}_p05"] = float(np.quantile(vals, 0.05))
    macro_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    audit_best_macro_counts: dict[str, int] = {}
    audit_miss_best_macro_counts: dict[str, int] = {}
    audit_miss_selected_macro_counts: dict[str, int] = {}
    audit_pcd_best_macro_counts: dict[str, int] = {}
    audit_pcd_miss_best_macro_counts: dict[str, int] = {}
    audit_paper_pcd_best_macro_counts: dict[str, int] = {}
    audit_paper_pcd_miss_best_macro_counts: dict[str, int] = {}
    for s in scene_results:
        for m, c in (s.get("macro_counts", {}) or {}).items():
            macro_counts[m] = macro_counts.get(m, 0) + int(c)
        for m, c in (s.get("audit_best_macro_counts", {}) or {}).items():
            audit_best_macro_counts[m] = audit_best_macro_counts.get(m, 0) + int(c)
        for m, c in (s.get("audit_miss_best_macro_counts", {}) or {}).items():
            audit_miss_best_macro_counts[m] = audit_miss_best_macro_counts.get(m, 0) + int(c)
        for m, c in (s.get("audit_miss_selected_macro_counts", {}) or {}).items():
            audit_miss_selected_macro_counts[m] = audit_miss_selected_macro_counts.get(m, 0) + int(c)
        for m, c in (s.get("audit_pcd_best_macro_counts", {}) or {}).items():
            audit_pcd_best_macro_counts[m] = audit_pcd_best_macro_counts.get(m, 0) + int(c)
        for m, c in (s.get("audit_pcd_miss_best_macro_counts", {}) or {}).items():
            audit_pcd_miss_best_macro_counts[m] = audit_pcd_miss_best_macro_counts.get(m, 0) + int(c)
        for m, c in (s.get("audit_paper_pcd_best_macro_counts", {}) or {}).items():
            audit_paper_pcd_best_macro_counts[m] = audit_paper_pcd_best_macro_counts.get(m, 0) + int(c)
        for m, c in (s.get("audit_paper_pcd_miss_best_macro_counts", {}) or {}).items():
            audit_paper_pcd_miss_best_macro_counts[m] = audit_paper_pcd_miss_best_macro_counts.get(m, 0) + int(c)
        for r, c in (s.get("selection_reason_counts", {}) or {}).items():
            reason_counts[r] = reason_counts.get(r, 0) + int(c)
    agg["macro_counts"] = macro_counts
    agg["audit_best_macro_counts"] = audit_best_macro_counts
    agg["audit_miss_best_macro_counts"] = audit_miss_best_macro_counts
    agg["audit_miss_selected_macro_counts"] = audit_miss_selected_macro_counts
    agg["audit_pcd_best_macro_counts"] = audit_pcd_best_macro_counts
    agg["audit_pcd_miss_best_macro_counts"] = audit_pcd_miss_best_macro_counts
    agg["audit_paper_pcd_best_macro_counts"] = audit_paper_pcd_best_macro_counts
    agg["audit_paper_pcd_miss_best_macro_counts"] = audit_paper_pcd_miss_best_macro_counts
    agg["selection_reason_counts"] = reason_counts
    timing_names = sorted({name for s in scene_results for name in ((s.get("timing", {}) or {}).get("totals_s", {}) or {}).keys()})
    timing_totals = {
        name: float(sum(float((((s.get("timing", {}) or {}).get("totals_s", {}) or {}).get(name, 0.0))) for s in scene_results))
        for name in timing_names
    }
    timing_wall = float(sum(float((s.get("timing", {}) or {}).get("wall_s", 0.0)) for s in scene_results))
    agg["timing"] = {
        "scene_wall_sum_s": timing_wall,
        "totals_s": timing_totals,
        "per_decision_s": {name: float(value / max(int(agg["num_decisions"]), 1)) for name, value in timing_totals.items()},
        "measured_fraction": float(sum(timing_totals.values()) / max(timing_wall, 1.0e-9)),
    }
    return agg




def _dataset_label_for_sample_path(path: Path) -> str:
    try:
        return path.parent.parent.name if path.parent.name == "samples" else path.parent.name
    except Exception:
        return "dataset"


def _load_target_key_filter(path_value: str | Path | None) -> set[str]:
    """Load target keys from JSON, JSONL or one-key-per-line text.

    Selection files written by ``select_critical_scenes_v48_34.py`` are
    accepted directly, as are ``{"target_keys": [...]}`` documents.
    """
    raw = str(path_value or "").strip()
    if not raw:
        return set()
    path = Path(raw)
    if not path.is_file():
        raise FileNotFoundError(f"closed_loop.target_keys_file does not exist: {path}")

    def collect(value: Any, out: set[str]) -> None:
        if isinstance(value, str):
            text = value.strip()
            if text:
                out.add(text)
        elif isinstance(value, list):
            for item in value:
                collect(item, out)
        elif isinstance(value, dict):
            for key in ("target_key", "resume_key"):
                if value.get(key):
                    text = str(value[key]).strip()
                    if text.startswith("target:"):
                        text = text[len("target:"):]
                    if text:
                        out.add(text)
                    return
            for key in ("target_keys", "selected", "items", "scenes"):
                if key in value:
                    collect(value[key], out)

    keys: set[str] = set()
    text = path.read_text(encoding="utf-8")
    try:
        collect(json.loads(text), keys)
    except json.JSONDecodeError:
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                collect(json.loads(line), keys)
            except json.JSONDecodeError:
                collect(line, keys)
    return keys


def _load_closed_loop_targets(dataset_spec: str | None, cfg: dict) -> list[dict[str, Any]]:
    """Load unique (bucket, scene_id, time_index) targets from OC-RAP offline roots."""
    if not dataset_spec:
        return []
    cl_cfg = cfg.get("closed_loop", {}) if isinstance(cfg.get("closed_loop", {}), dict) else {}
    split_filter = str(cl_cfg.get("bucket_split", "") or "").strip()
    max_targets = int(cl_cfg.get("max_bucket_targets", 0) or 0)
    max_per_scene = int(cl_cfg.get("max_targets_per_scene", 1) or 1)
    requested_keys = _load_target_key_filter(cl_cfg.get("target_keys_file"))
    require_requested_keys = bool(cl_cfg.get("require_target_keys", bool(requested_keys)))
    # Include the resolved content, not only its path, in the run fingerprint.
    if requested_keys:
        cl_cfg["_resolved_target_keys"] = sorted(requested_keys)
    paths = iter_sample_paths_many(dataset_spec)
    seen: set[tuple[str, str, int]] = set()
    per_scene: dict[str, int] = {}
    targets: list[dict[str, Any]] = []
    for p in paths:
        split = str(scalar_metadata_for_path(p, "split_id", ""))
        if split_filter and split != split_filter:
            continue
        scene_id = str(scalar_metadata_for_path(p, "scene_id", ""))
        original_scene_id = str(scalar_metadata_for_path(p, "original_scenario_id", ""))
        official_scene_id = str(scalar_metadata_for_path(p, "official_scenario_id", ""))
        legacy_scene_id = str(scalar_metadata_for_path(p, "legacy_scenario_id", ""))
        if not scene_id and not original_scene_id and not official_scene_id:
            continue
        canonical_scene_id = _canonical_womd_scene_id(official_scene_id or original_scene_id or scene_id)
        if not canonical_scene_id:
            continue
        source_index_raw = scalar_metadata_for_path(p, "source_scenario_index", -1)
        try:
            source_scenario_index = int(float(source_index_raw))
        except Exception:
            source_scenario_index = -1
        if source_scenario_index < 0:
            legacy_idx = _legacy_waymax_source_index(original_scene_id)
            if legacy_idx is None:
                legacy_idx = _legacy_waymax_source_index(scene_id)
            source_scenario_index = int(legacy_idx) if legacy_idx is not None else -1
        source_role = str(scalar_metadata_for_path(p, "womd_source_role", "") or "unknown")
        max_num_objects_raw = scalar_metadata_for_path(p, "waymax_max_num_objects", -1)
        try:
            waymax_max_num_objects = int(float(max_num_objects_raw))
        except Exception:
            waymax_max_num_objects = -1
        try:
            time_index = int(float(scalar_metadata_for_path(p, "time_index", 0)))
        except Exception:
            continue
        bucket = _dataset_label_for_sample_path(Path(p))
        target_key = f"{bucket}:{canonical_scene_id}:t{int(time_index)}"
        if requested_keys and target_key not in requested_keys:
            continue
        key = (bucket, canonical_scene_id, time_index)
        if key in seen:
            continue
        if per_scene.get(canonical_scene_id, 0) >= max_per_scene:
            continue
        seen.add(key)
        per_scene[canonical_scene_id] = per_scene.get(canonical_scene_id, 0) + 1
        aliases = {
            _canonical_womd_scene_id(v)
            for v in (scene_id, original_scene_id, official_scene_id, legacy_scene_id)
            if str(v or "").strip()
        }
        aliases.discard("")
        targets.append({
            "bucket_name": bucket,
            "scene_id": canonical_scene_id,
            "scene_aliases": sorted(aliases),
            "saved_scene_id": scene_id,
            "original_scenario_id": original_scene_id or None,
            "official_scenario_id": official_scene_id or None,
            "legacy_scenario_id": legacy_scene_id or None,
            "source_scenario_index": int(source_scenario_index),
            "womd_source_role": source_role,
            "waymax_max_num_objects": int(waymax_max_num_objects),
            "time_index": int(time_index),
            "target_key": target_key,
        })
        if max_targets > 0 and len(targets) >= max_targets:
            break
    if requested_keys and require_requested_keys:
        loaded = {str(t.get("target_key")) for t in targets}
        missing = sorted(requested_keys - loaded)
        if missing:
            raise ValueError(
                "closed_loop.target_keys_file contains keys absent from the bucket dataset: "
                + ", ".join(missing[:20])
            )
    return targets


def _aggregate_with_buckets(scene_results: list[dict[str, Any]], method: str, source: str) -> dict[str, Any]:
    result = _aggregate_scene_results(scene_results, method, source)
    buckets = sorted({str(s.get("bucket_name")) for s in scene_results if s.get("bucket_name")})
    if buckets:
        result["per_bucket"] = {}
        for b in buckets:
            sub = [s for s in scene_results if str(s.get("bucket_name")) == b]
            result["per_bucket"][b] = _aggregate_scene_results(sub, method, source)
    return result

_RESUME_OPERATIONAL_KEYS = {
    "resume",
    "resume_force",
    "resume_allow_legacy_partial",
    "resume_fsync",
    "save_partial",
    "partial_write_every_scenes",
    "progress",
    "progress_every_steps",
    "keep_resume_files_after_success",
    # Serialization controls do not affect actions or metrics and therefore
    # must not invalidate a compatible interrupted run.
    "result_scene_detail",
    "scene_journal_detail",
    "memory_scene_detail",
    "include_scenes_in_result",
    "include_scenes_in_partial",
}


def _closed_loop_fingerprint(
    dataset_patterns: str,
    checkpoint: str | Path | None,
    method: str,
    target_spec: str,
    cfg: dict,
) -> str:
    """Fingerprint all result-affecting inputs, excluding persistence controls."""
    local = dict(cfg)
    cl = dict(local.get("closed_loop", {}) or {})
    for key in _RESUME_OPERATIONAL_KEYS:
        cl.pop(key, None)
    local["closed_loop"] = cl
    ckpt_info: dict[str, Any] | None = None
    if checkpoint:
        cp = Path(checkpoint)
        ckpt_info = {"path": str(cp.resolve())}
        try:
            st = cp.stat()
            ckpt_info.update({"size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)})
        except OSError:
            ckpt_info["missing"] = True
    payload = {
        "version": 1,
        "dataset_patterns": str(dataset_patterns),
        "checkpoint": ckpt_info,
        "method": str(method),
        "target_spec": str(target_spec or ""),
        "config": local,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _scene_resume_key(scene: dict[str, Any]) -> str:
    target_key = str(scene.get("target_key", "") or "").strip()
    if target_key:
        return f"target:{target_key}"
    scene_id = str(scene.get("scene_id", "") or "").strip()
    bucket = str(scene.get("bucket_name", "") or "").strip()
    time_index = scene.get("target_time_index", None)
    if bucket or time_index is not None:
        return f"bucket:{bucket}|scene:{scene_id}|t:{time_index}"
    return f"scene:{scene_id}"


def _expected_resume_key(scene_id: str, target: dict[str, Any]) -> str:
    target_key = str(target.get("target_key", "") or "").strip()
    if target_key:
        return f"target:{target_key}"
    bucket = str(target.get("bucket_name", "") or "").strip()
    time_index = target.get("time_index", None)
    if bucket or time_index is not None:
        return f"bucket:{bucket}|scene:{scene_id}|t:{time_index}"
    return f"scene:{scene_id}"


def _read_json_if_valid(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def _resume_metadata_compatible(
    data: dict[str, Any],
    *,
    fingerprint: str,
    method: str,
    target_spec: str,
    force: bool,
    allow_legacy: bool,
    source_name: str,
) -> tuple[bool, bool]:
    """Return (compatible, legacy_without_fingerprint)."""
    saved_fp = str(data.get("run_fingerprint", "") or "")
    if saved_fp:
        if saved_fp == fingerprint or force:
            return True, False
        raise ValueError(
            f"Refusing to resume {source_name}: run_fingerprint differs. "
            "Use a new --output path, or set closed_loop.resume_force=true only after verifying the configs are evaluation-equivalent."
        )
    if not allow_legacy and not force:
        return False, True
    saved_method = str(data.get("method", method) or method).lower()
    saved_target = str(data.get("bucket_dataset", target_spec) or "")
    compatible = (saved_method == str(method).lower()) and (saved_target == str(target_spec or ""))
    if not compatible and not force:
        raise ValueError(
            f"Refusing to resume legacy {source_name}: method or bucket_dataset differs. "
            "Use a new --output path or set closed_loop.resume_force=true after manual verification."
        )
    return bool(compatible or force), True


def _load_resume_scene_results(
    *,
    output_path: Path,
    partial_path: Path,
    journal_path: Path,
    fingerprint: str,
    method: str,
    target_spec: str,
    force: bool,
    allow_legacy: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load completed scenes from final/partial snapshots and append-only journal."""
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    meta: dict[str, Any] = {"sources": [], "legacy_sources": [], "prior_raw_scenarios_seen": 0}

    def add_scenes(data: dict[str, Any], source_name: str) -> None:
        compatible, legacy = _resume_metadata_compatible(
            data,
            fingerprint=fingerprint,
            method=method,
            target_spec=target_spec,
            force=force,
            allow_legacy=allow_legacy,
            source_name=source_name,
        )
        if not compatible:
            return
        scenes = data.get("scenes", [])
        if not isinstance(scenes, list):
            return
        meta["sources"].append(source_name)
        if legacy:
            meta["legacy_sources"].append(source_name)
        try:
            meta["prior_raw_scenarios_seen"] = max(
                int(meta["prior_raw_scenarios_seen"]), int(data.get("raw_scenarios_seen", 0) or 0)
            )
        except Exception:
            pass
        for scene in scenes:
            if not isinstance(scene, dict):
                continue
            key = _scene_resume_key(scene)
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(scene)

    # A legacy interrupted run may coexist with an older completed final file at
    # the same output path. Prefer the partial snapshot in that case; new files
    # carry fingerprints and can be safely merged in either order.
    partial_data = _read_json_if_valid(partial_path)
    final_data = _read_json_if_valid(output_path)
    if partial_data is not None:
        add_scenes(partial_data, "partial")
    if final_data is not None:
        final_has_fp = bool(str(final_data.get("run_fingerprint", "") or ""))
        if final_has_fp or partial_data is None:
            add_scenes(final_data, "final")

    if journal_path.exists():
        loaded_any = False
        legacy_journal = False
        try:
            with journal_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        # A kill during append can leave only the final line torn.
                        continue
                    if not isinstance(record, dict) or not isinstance(record.get("scene"), dict):
                        continue
                    saved_fp = str(record.get("run_fingerprint", "") or "")
                    if saved_fp and saved_fp != fingerprint and not force:
                        raise ValueError(
                            "Refusing to resume scene journal: run_fingerprint differs. "
                            "Use a new --output path or closed_loop.resume_force=true after verification."
                        )
                    if not saved_fp:
                        if not allow_legacy and not force:
                            continue
                        legacy_journal = True
                    scene = record["scene"]
                    key = _scene_resume_key(scene)
                    if key and key not in seen:
                        seen.add(key)
                        merged.append(scene)
                        loaded_any = True
        except OSError:
            pass
        if loaded_any:
            meta["sources"].append("journal")
        if legacy_journal:
            meta["legacy_sources"].append("journal")
    return merged, meta


def _append_scene_journal(path: Path, fingerprint: str, scene: dict[str, Any], *, fsync: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"version": 1, "run_fingerprint": fingerprint, "resume_key": _scene_resume_key(scene), "scene": scene}
    encoded = json.dumps(record, sort_keys=True, ensure_ascii=False, default=str) + "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(encoded)
        f.flush()
        if fsync:
            os.fsync(f.fileno())


def _write_closed_loop_progress(
    path: Path,
    *,
    fingerprint: str,
    status: str,
    completed: int,
    total: int,
    current: dict[str, Any] | None = None,
    resumed: int = 0,
) -> None:
    write_json(
        {
            "version": 1,
            "run_fingerprint": fingerprint,
            "status": status,
            "completed_rollouts": int(completed),
            "requested_rollouts": int(total),
            "resumed_rollouts": int(resumed),
            "current": current,
        },
        path,
    )


def _closed_loop_rollout_limit(*, target_count: int, max_rollouts: int, max_scenes: int) -> int:
    """Resolve the rollout cap; zero means all for an explicit target set."""
    if int(target_count) > 0:
        return int(target_count) if int(max_rollouts) <= 0 else min(int(target_count), int(max_rollouts))
    return max(int(max_scenes), 0)


def closed_loop_evaluate(dataset_patterns: str, checkpoint: str | Path | None, output: str | Path, cfg: dict) -> dict[str, Any]:
    if not str(dataset_patterns).strip() or str(dataset_patterns).strip().startswith("@"):
        raise ValueError(
            "closed-loop --dataset is empty or only contains an @limit suffix. "
            "Pass an explicit WOMD TFRecord path, e.g. ${WOMD_VAL}@150 after exporting WOMD_VAL."
        )
    cl_cfg = cfg.get("closed_loop", {}) if isinstance(cfg.get("closed_loop", {}), dict) else {}
    max_scenes = int(cl_cfg.get("max_scenarios", cfg.get("max_scenarios") or 8))
    # For bucket-targeted evaluation, 0 means all loaded targets.  This is
    # required for full-regime runs; the historical ``or max_scenes`` behavior
    # silently turned 0 back into a small exploratory cap.
    raw_max_rollouts = cl_cfg.get("max_rollouts", max_scenes)
    max_rollouts = int(raw_max_rollouts) if raw_max_rollouts not in {None, ""} else max_scenes
    raw_max_scenarios = cl_cfg.get("raw_max_scenarios", cfg.get("max_scenarios", None))
    raw_max_scenarios = None if raw_max_scenarios in {None, "", 0, "0"} else int(raw_max_scenarios)
    method = str(cl_cfg.get("method", "ocrap")).lower()
    gamma = float((cfg.get("selection", {}) or {}).get("gamma_rec", 0.0))
    if not np.isfinite(gamma) and not bool(cl_cfg.get("allow_infinite_gamma", False)):
        raise ValueError(
            "closed-loop selection.gamma_rec is not finite. Re-read calibration.json with the requested delta "
            "or pass --set closed_loop.allow_infinite_gamma=true only for debugging."
        )
    local = _apply_gamma_rec_by_bucket_file(dict(cfg))
    _validate_closed_loop_selector_config(local, method)
    local["data_source"] = "womd"
    local["simulation_backend"] = "waymax_closed_loop"
    local["womd_patterns"] = dataset_patterns
    local["max_scenarios"] = max_scenes

    # Closed-loop evaluation is frequently run on WOMD TFExample shards whose
    # records do not contain `path_samples/*` route features.  The offline
    # OC-RAP dataset-build commands already disable SDC-path parsing for the
    # held-out test buckets; keep the online evaluator consistent by default.
    # Set `--set closed_loop.use_sdc_paths=true` only when the raw WOMD shards
    # are known to contain path_samples/{xyz,valid,id,arc_length,on_route}.
    wx = dict(local.get("waymax", {}) or {})
    use_sdc_paths = bool(cl_cfg.get("use_sdc_paths", False))
    wx["dataloader_include_sdc_paths"] = use_sdc_paths
    if not use_sdc_paths:
        route_metrics = {"sdc_wrongway", "sdc_off_route", "sdc_progression"}
        metrics = wx.get("metrics_to_run", [])
        if metrics:
            wx["metrics_to_run"] = [str(m) for m in metrics if str(m) not in route_metrics]

    # The external near-contact/contact baselines only need branch/root teacher
    # labels for the candidate lattice they score.  Online future-metric rows are
    # useful for dataset diagnostics, but they are redundant in true Waymax
    # receding-horizon evaluation because executed-step metrics are already
    # collected from the simulator after each env.step().  Keep this disabled for
    # external closed-loop speed even if old shell commands still pass
    # --set waymax.compute_future_metrics=true.  Re-enable explicitly with
    # --set closed_loop.external_online_future_metrics=true when doing a small
    # exhaustive diagnostic run.
    if method in EXTERNAL_CLOSED_LOOP_METHODS and not bool(cl_cfg.get("external_online_future_metrics", False)):
        wx["compute_future_metrics"] = False
        if int(wx.get("teacher_rollout_top_k_options", 0) or 0) <= 0:
            wx["teacher_rollout_top_k_options"] = int(cl_cfg.get("external_teacher_rollout_top_k_options", 4) or 4)
        wx["teacher_metrics_stride"] = int(wx.get("teacher_metrics_stride", 0) or 0)
    local["waymax"] = wx

    art = dict(local.get("artifact", {}) or {})
    mine_p = float(cl_cfg.get("artifact_mine_probability", 0.0) or 0.0)
    art["force_mine"] = mine_p > 0.0
    art["mine_probability"] = max(0.0, min(1.0, mine_p))
    local["artifact"] = art
    external_model = None
    external_model_cfg = None
    external_device = None
    if method in EXTERNAL_CLOSED_LOOP_METHODS:
        bundle = None
        external_ckpt = checkpoint or cl_cfg.get("external_checkpoint", None)
        if external_ckpt:
            external_model, external_model_cfg, external_device = _load_external_checkpoint(external_ckpt, local)
            if external_model is None and method in EXTERNAL_LEARNED_METHODS:
                raise FileNotFoundError(f"Could not load external baseline checkpoint for closed-loop evaluation: {external_ckpt}")
        if external_model is not None:
            source = "external_checkpoint_observation_only_policy"
        elif method in EXTERNAL_TEACHER_REQUIRED_METHODS:
            source = "teacher_only_oracle_upper_bound"
        else:
            source = "observation_only_external_policy"
    else:
        bundle = load_model_bundle(checkpoint, local)
        if checkpoint and bundle is None:
            raise FileNotFoundError(f"Could not load model checkpoint for closed-loop evaluation: {checkpoint}")
        source = "model" if bundle is not None else "teacher_fallback"
    output_path = Path(output)
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    journal_path = output_path.with_suffix(output_path.suffix + ".scenes.jsonl")
    progress_path = output_path.with_suffix(output_path.suffix + ".progress.json")
    save_partial = bool(cl_cfg.get("save_partial", True))
    progress = bool(cl_cfg.get("progress", True))
    resume = bool(cl_cfg.get("resume", True))
    resume_force = bool(cl_cfg.get("resume_force", False))
    resume_allow_legacy = bool(cl_cfg.get("resume_allow_legacy_partial", True))
    resume_fsync = bool(cl_cfg.get("resume_fsync", False))
    partial_every = max(1, int(cl_cfg.get("partial_write_every_scenes", 4) or 4))
    result_scene_detail = _normalize_scene_storage_detail(cl_cfg.get("result_scene_detail", "full"))
    journal_scene_detail = _normalize_scene_storage_detail(
        cl_cfg.get("scene_journal_detail", result_scene_detail)
    )
    memory_scene_detail = _normalize_scene_storage_detail(
        cl_cfg.get("memory_scene_detail", journal_scene_detail)
    )
    include_scenes_in_result = bool(cl_cfg.get("include_scenes_in_result", True))
    include_scenes_in_partial = bool(cl_cfg.get("include_scenes_in_partial", True))
    target_spec = str(cl_cfg.get("bucket_dataset", cl_cfg.get("target_dataset", "")) or "").strip()
    targets = _load_closed_loop_targets(target_spec, local)
    if target_spec and bool(cl_cfg.get("require_bucket_targets", False)) and not targets:
        raise ValueError(
            "closed_loop.bucket_dataset was provided but no targets matched; "
            "check closed_loop.bucket_split/manifest metadata instead of silently "
            "falling back to arbitrary WOMD scenes"
        )
    target_map: dict[str, list[dict[str, Any]]] = {}
    target_index_map: dict[int, list[dict[str, Any]]] = {}
    for t in targets:
        aliases = set(t.get("scene_aliases") or [])
        aliases.add(_canonical_womd_scene_id(t["scene_id"]))
        for alias in aliases:
            if alias:
                target_map.setdefault(alias, []).append(t)
        raw_source_index = t.get("source_scenario_index", -1)
        source_index = -1 if raw_source_index is None else int(raw_source_index)
        if source_index >= 0:
            target_index_map.setdefault(source_index, []).append(t)
    raw_scan_bound_source = "explicit" if raw_max_scenarios is not None else "unbounded"
    # Legacy datasets encode original Waymax record indices in ``__wxNNN``.
    # When every requested target has such an index, stop after the largest one
    # instead of decoding unrelated trailing records.  This is a record bound,
    # independent of the @N TFRecord shard declaration.
    if targets and raw_max_scenarios is None:
        target_source_indices = [
            -1 if t.get("source_scenario_index", -1) is None else int(t.get("source_scenario_index", -1))
            for t in targets
        ]
        scenario_start_index = int(local.get("scenario_start_index", 0) or 0)
        scenario_stride = int(local.get("scenario_stride", 1) or 1)
        if target_source_indices and all(idx >= 0 for idx in target_source_indices) and scenario_start_index == 0 and scenario_stride == 1:
            raw_max_scenarios = max(target_source_indices) + 1
            raw_scan_bound_source = "derived_from_max_target_source_index"
    raw_source_role = _source_role_from_pattern(dataset_patterns)
    declared_target_roles = {str(t.get("womd_source_role", "unknown")) for t in targets}
    known_target_roles = {r for r in declared_target_roles if r not in {"", "unknown"}}
    if targets and known_target_roles and raw_source_role not in known_target_roles:
        raise RuntimeError(
            "Closed-loop target/source split mismatch: targets were built from "
            f"{sorted(known_target_roles)}, but raw WOMD source is {raw_source_role}. "
            "Use the same standard validation split and shard set used by the "
            "adaptation-dev builder."
        )
    allow_legacy_index = bool(cl_cfg.get("allow_legacy_source_index_targets", True))
    run_fingerprint = _closed_loop_fingerprint(dataset_patterns, checkpoint, method, target_spec, local)
    total_rollouts = _closed_loop_rollout_limit(
        target_count=len(targets), max_rollouts=max_rollouts, max_scenes=max_scenes
    )
    resume_meta: dict[str, Any] = {"sources": [], "legacy_sources": [], "prior_raw_scenarios_seen": 0}
    if resume:
        scene_results, resume_meta = _load_resume_scene_results(
            output_path=output_path,
            partial_path=partial_path,
            journal_path=journal_path,
            fingerprint=run_fingerprint,
            method=method,
            target_spec=target_spec,
            force=resume_force,
            allow_legacy=resume_allow_legacy,
        )
        # Old v50 journals may contain full decision traces. Compact them in
        # memory immediately so a resumed metric-only run does not recreate the
        # original RAM spike.
        scene_results = [_scene_storage_view(scene, memory_scene_detail) for scene in scene_results]
    else:
        scene_results = []
        for stale in (partial_path, journal_path, progress_path):
            try:
                stale.unlink(missing_ok=True)
            except OSError:
                pass
    completed_keys = {_scene_resume_key(scene) for scene in scene_results}
    resumed_rollouts = len(scene_results)
    raw_seen_this_run = 0
    raw_seen = int(resume_meta.get("prior_raw_scenarios_seen", 0) or 0)
    matched_targets = sum(1 for scene in scene_results if scene.get("target_key") or scene.get("bucket_name")) if targets else 0
    _write_closed_loop_progress(
        progress_path,
        fingerprint=run_fingerprint,
        status="resuming" if resumed_rollouts else "running",
        completed=len(scene_results),
        total=total_rollouts,
        resumed=resumed_rollouts,
    )
    if progress and resumed_rollouts:
        print({
            "event": "closed_loop_resume_loaded",
            "completed_rollouts": resumed_rollouts,
            "sources": resume_meta.get("sources", []),
            "legacy_sources": resume_meta.get("legacy_sources", []),
            "journal": str(journal_path),
        }, flush=True)
    if progress and targets:
        print({"event": "closed_loop_bucket_targets_loaded", "num_targets": len(targets), "num_target_scenes": len({t["scene_id"] for t in targets}), "num_legacy_source_indices": len(target_index_map), "target_source_roles": sorted(declared_target_roles), "raw_source_role": raw_source_role, "target_scene_examples": sorted(target_map)[:5], "max_rollouts": max_rollouts, "raw_max_scenarios": raw_max_scenarios, "raw_scan_bound_source": raw_scan_bound_source, "target_keys_file": cl_cfg.get("target_keys_file")}, flush=True)
    new_scenes_since_partial = 0
    for i, raw in enumerate(iter_waymax_womd_scenarios(dataset_patterns, max_scenarios=raw_max_scenarios if targets else max_scenes, parser_cfg=local)):
        raw_seen_this_run += 1
        raw_seen = max(raw_seen, raw_seen_this_run)
        raw_identity_keys = _raw_scene_identity_keys(raw)
        raw_targets_by_key: dict[str, dict[str, Any]] = {}
        for identity_key in raw_identity_keys:
            for target in target_map.get(identity_key, []):
                raw_targets_by_key[str(target.get("target_key"))] = target
        raw_index = int((getattr(raw, "metadata", {}) or {}).get("_waymax_scenario_index", i))
        if targets and not raw_targets_by_key and allow_legacy_index:
            for target in target_index_map.get(raw_index, []):
                # Source-order matching is permitted only on the same declared
                # split. It is a migration path for pre-v48.28 datasets, never
                # the primary scene identity.
                target_role = str(target.get("womd_source_role", "unknown"))
                if target_role in {"", "unknown", raw_source_role}:
                    raw_targets_by_key[str(target.get("target_key"))] = target
        raw_targets = list(raw_targets_by_key.values()) if targets else [{"bucket_name": None, "time_index": None, "target_key": None}]
        if targets and not raw_targets:
            continue
        for target in raw_targets:
            if len(scene_results) >= total_rollouts:
                break
            resume_key = _expected_resume_key(str(raw.scenario_id), target)
            if resume_key in completed_keys:
                if progress:
                    print({
                        "event": "closed_loop_scene_resume_skip",
                        "scene_id": str(raw.scenario_id),
                        "bucket": target.get("bucket_name"),
                        "start_time_index": target.get("time_index"),
                        "resume_key": resume_key,
                    }, flush=True)
                continue
            rank = len(scene_results)
            current_progress = {
                "scene_rank": rank,
                "raw_rank": i,
                "scene_id": str(raw.scenario_id),
                "bucket": target.get("bucket_name"),
                "start_time_index": target.get("time_index"),
                "resume_key": resume_key,
            }
            _write_closed_loop_progress(
                progress_path,
                fingerprint=run_fingerprint,
                status="running_scene",
                completed=len(scene_results),
                total=total_rollouts,
                current=current_progress,
                resumed=resumed_rollouts,
            )
            if progress:
                print({"event": "closed_loop_scene_start", **current_progress, "max_rollouts": total_rollouts}, flush=True)
            gamma_i = _gamma_for_bucket(gamma, local, target.get("bucket_name"))
            scene_result = _rollout_one_scene(
                raw,
                rank,
                bundle,
                local,
                method,
                gamma_i,
                start_time_index_override=target.get("time_index"),
                bucket_name=target.get("bucket_name"),
                target_key=target.get("target_key"),
                external_model=external_model,
                external_model_cfg=external_model_cfg,
                external_device=external_device,
            )
            memory_scene = _scene_storage_view(scene_result, memory_scene_detail)
            journal_scene = _scene_storage_view(scene_result, journal_scene_detail)
            scene_results.append(memory_scene)
            completed_keys.add(_scene_resume_key(memory_scene))
            matched_targets += int(bool(targets))
            new_scenes_since_partial += 1

            # The JSONL journal is the authoritative scene-granular resume and
            # pairing artifact. Metric-only runs keep it compact; selective
            # video reruns request full render traces explicitly.
            if resume or save_partial:
                _append_scene_journal(journal_path, run_fingerprint, journal_scene, fsync=resume_fsync)
            if save_partial and new_scenes_since_partial >= partial_every:
                partial = _aggregate_with_buckets(scene_results, method, source)
                partial.update({
                    "partial": True,
                    "run_fingerprint": run_fingerprint,
                    "resume_supported": True,
                    "bucket_dataset": target_spec or None,
                    "bucket_target_count": len(targets),
                    "bucket_matched_rollouts": matched_targets,
                    "raw_scenarios_seen": raw_seen,
                    "raw_scenarios_seen_this_run": raw_seen_this_run,
                    "target_id_matching": "official_or_legacy_alias_with_provenance_checked_source_index_fallback",
                    "scene_storage_detail": memory_scene_detail,
                    "scene_journal_detail": journal_scene_detail,
                })
                if include_scenes_in_partial:
                    partial["scenes"] = scene_results
                write_json(partial, partial_path, fsync=resume_fsync)
                new_scenes_since_partial = 0
            _write_closed_loop_progress(
                progress_path,
                fingerprint=run_fingerprint,
                status="running",
                completed=len(scene_results),
                total=total_rollouts,
                resumed=resumed_rollouts,
            )
            if progress:
                print({"event": "closed_loop_scene_done", "scene_rank": rank, "num_decisions": scene_result.get("num_decisions", 0), "completed_rollouts": len(scene_results)}, flush=True)
        if len(scene_results) >= total_rollouts:
            break

    if targets and matched_targets == 0 and bool(cl_cfg.get("require_bucket_targets", False)):
        raise RuntimeError(
            "No requested bucket target scene matched the raw WOMD source after "
            f"scanning {raw_seen_this_run} scenarios. Verify that adaptation-dev "
            "and closed loop use the same WOMD split/shard set. Pre-v48.28 target "
            "ids are local state hashes; use the standard validation source and "
            "enable closed_loop.allow_legacy_source_index_targets for migration."
        )

    # Always leave a final valid partial snapshot, even when the last group has
    # fewer than partial_write_every_scenes rollouts.
    if save_partial:
        partial = _aggregate_with_buckets(scene_results, method, source)
        partial.update({
            "partial": True,
            "run_fingerprint": run_fingerprint,
            "resume_supported": True,
            "bucket_dataset": target_spec or None,
            "bucket_target_count": len(targets),
            "bucket_matched_rollouts": matched_targets,
            "raw_scenarios_seen": raw_seen,
            "raw_scenarios_seen_this_run": raw_seen_this_run,
            "target_id_matching": "official_or_legacy_alias_with_provenance_checked_source_index_fallback",
            "scene_storage_detail": memory_scene_detail,
            "scene_journal_detail": journal_scene_detail,
        })
        if include_scenes_in_partial:
            partial["scenes"] = scene_results
        write_json(partial, partial_path, fsync=resume_fsync)
    result = _aggregate_with_buckets(scene_results, method, source)
    result["bucket_dataset"] = target_spec or None
    result["bucket_target_count"] = len(targets)
    result["bucket_matched_rollouts"] = matched_targets
    result["target_keys_file"] = str(cl_cfg.get("target_keys_file", "") or "") or None
    result["raw_scan_bound_source"] = raw_scan_bound_source
    result["raw_scenarios_seen"] = raw_seen
    result["raw_scenarios_seen_this_run"] = raw_seen_this_run
    result["run_fingerprint"] = run_fingerprint
    result["resume_supported"] = True
    result["resume"] = {
        "enabled": resume,
        "resumed_rollouts": int(resumed_rollouts),
        "sources": list(resume_meta.get("sources", [])),
        "legacy_sources": list(resume_meta.get("legacy_sources", [])),
        "partial_path": str(partial_path),
        "journal_path": str(journal_path),
        "progress_path": str(progress_path),
        "granularity": "completed_scene_or_bucket_target",
    }
    if resume_meta.get("legacy_sources"):
        result.setdefault("warnings", []).append(
            "Resumed from a legacy snapshot without run_fingerprint. Method and bucket_dataset were checked, but use a new output path when checkpoint/result-affecting config changed."
        )
    if targets and matched_targets == 0:
        result.setdefault("warnings", []).append("No offline bucket scene_id matched the supplied WOMD raw dataset/pattern. Check WOMD_VAL vs WOMD_VAL_INTERACTIVE and scenario_start_index/raw_max_scenarios.")
    if include_scenes_in_result:
        result["scenes"] = [_scene_storage_view(scene, result_scene_detail) for scene in scene_results]
    result["scene_storage_detail"] = memory_scene_detail
    result["scene_journal_detail"] = journal_scene_detail
    result["scenes_embedded"] = bool(include_scenes_in_result)
    result["gamma_rec"] = gamma
    eff_wx = local.get("waymax", {}) if isinstance(local.get("waymax", {}), dict) else {}
    eff_cl = local.get("closed_loop", {}) if isinstance(local.get("closed_loop", {}), dict) else {}
    result["closed_loop_speed_config"] = {
        "external_sparse_labels": bool(eff_cl.get("external_sparse_labels", False)),
        "external_label_max_candidates": eff_cl.get("external_label_max_candidates", None),
        "num_recovery_options": eff_cl.get("num_recovery_options", None),
        "compute_future_metrics": bool(eff_wx.get("compute_future_metrics", False)),
        "teacher_metrics_stride": int(eff_wx.get("teacher_metrics_stride", 0) or 0),
        "teacher_rollout_top_k_options": int(eff_wx.get("teacher_rollout_top_k_options", 0) or 0),
        "dataloader_include_sdc_paths": bool(eff_wx.get("dataloader_include_sdc_paths", False)),
        "shared_scene_feature_extraction": True,
        "audit_lightweight_serialization": True,
        "partial_write_every_scenes": int(partial_every),
        "scene_journal": True,
        "result_scene_detail": result_scene_detail,
        "scene_journal_detail": journal_scene_detail,
        "memory_scene_detail": memory_scene_detail,
        "include_scenes_in_result": bool(include_scenes_in_result),
        "include_scenes_in_partial": bool(include_scenes_in_partial),
    }
    result["gamma_rec_by_bucket"] = (local.get("selection", {}) or {}).get("gamma_rec_by_bucket", {}) if isinstance(local.get("selection", {}), dict) else {}
    result["selector_config"] = {
        "ocrap_selector": (local.get("selection", {}) or {}).get("ocrap_selector", None) if isinstance(local.get("selection", {}), dict) else None,
        "drs_success_gamma": (local.get("selection", {}) or {}).get("drs_success_gamma", None) if isinstance(local.get("selection", {}), dict) else None,
        "safe_force_nominal_when_feasible": (local.get("selection", {}) or {}).get("safe_force_nominal_when_feasible", None) if isinstance(local.get("selection", {}), dict) else None,
        "safe_force_nominal_mode": (local.get("selection", {}) or {}).get("safe_force_nominal_mode", None) if isinstance(local.get("selection", {}), dict) else None,
        "safe_force_nominal_when_feasible_by_bucket": (local.get("selection", {}) or {}).get("safe_force_nominal_when_feasible_by_bucket", {}) if isinstance(local.get("selection", {}), dict) else {},
        "safe_force_nominal_mode_by_bucket": (local.get("selection", {}) or {}).get("safe_force_nominal_mode_by_bucket", {}) if isinstance(local.get("selection", {}), dict) else {},
        "stress_preserve_nominal_min_drs_drop_by_bucket": (local.get("selection", {}) or {}).get("stress_preserve_nominal_min_drs_drop_by_bucket", {}) if isinstance(local.get("selection", {}), dict) else {},
        "require_admitted_intervention": (local.get("selection", {}) or {}).get("require_admitted_intervention", None) if isinstance(local.get("selection", {}), dict) else None,
        "require_admitted_intervention_by_bucket": (local.get("selection", {}) or {}).get("require_admitted_intervention_by_bucket", {}) if isinstance(local.get("selection", {}), dict) else {},
        "require_intervention_evidence": (local.get("selection", {}) or {}).get("require_intervention_evidence", None) if isinstance(local.get("selection", {}), dict) else None,
        "require_intervention_evidence_by_bucket": (local.get("selection", {}) or {}).get("require_intervention_evidence_by_bucket", {}) if isinstance(local.get("selection", {}), dict) else {},
        "intervention_min_rec_lcb_gain_by_bucket": (local.get("selection", {}) or {}).get("intervention_min_rec_lcb_gain_by_bucket", {}) if isinstance(local.get("selection", {}), dict) else {},
        "intervention_min_drs_gain_by_bucket": (local.get("selection", {}) or {}).get("intervention_min_drs_gain_by_bucket", {}) if isinstance(local.get("selection", {}), dict) else {},
        "intervention_min_gap_reduction_by_bucket": (local.get("selection", {}) or {}).get("intervention_min_gap_reduction_by_bucket", {}) if isinstance(local.get("selection", {}), dict) else {},
        "option_drs_certificate": (local.get("selection", {}) or {}).get("option_drs_certificate", None) if isinstance(local.get("selection", {}), dict) else None,
        "option_drs_certificate_by_bucket": (local.get("selection", {}) or {}).get("option_drs_certificate_by_bucket", {}) if isinstance(local.get("selection", {}), dict) else {},
        "option_drs_certificate_threshold_by_bucket": (local.get("selection", {}) or {}).get("option_drs_certificate_threshold_by_bucket", {}) if isinstance(local.get("selection", {}), dict) else {},
        "option_drs_certificate_max_gap_by_bucket": (local.get("selection", {}) or {}).get("option_drs_certificate_max_gap_by_bucket", {}) if isinstance(local.get("selection", {}), dict) else {},
        "option_drs_certificate_rec_slack_by_bucket": (local.get("selection", {}) or {}).get("option_drs_certificate_rec_slack_by_bucket", {}) if isinstance(local.get("selection", {}), dict) else {},
        "relative_recovery_certificate": (local.get("selection", {}) or {}).get("relative_recovery_certificate", None) if isinstance(local.get("selection", {}), dict) else None,
        "relative_recovery_certificate_by_bucket": (local.get("selection", {}) or {}).get("relative_recovery_certificate_by_bucket", {}) if isinstance(local.get("selection", {}), dict) else {},
        "relative_recovery_min_rec_gain_by_bucket": (local.get("selection", {}) or {}).get("relative_recovery_min_rec_gain_by_bucket", {}) if isinstance(local.get("selection", {}), dict) else {},
        "relative_recovery_min_drs_by_bucket": (local.get("selection", {}) or {}).get("relative_recovery_min_drs_by_bucket", {}) if isinstance(local.get("selection", {}), dict) else {},
        "relative_recovery_max_gap_by_bucket": (local.get("selection", {}) or {}).get("relative_recovery_max_gap_by_bucket", {}) if isinstance(local.get("selection", {}), dict) else {},
        "relative_recovery_min_gap_reduction_by_bucket": (local.get("selection", {}) or {}).get("relative_recovery_min_gap_reduction_by_bucket", {}) if isinstance(local.get("selection", {}), dict) else {},
        "relative_recovery_gate_by_bucket": (local.get("selection", {}) or {}).get("relative_recovery_gate_by_bucket", {}) if isinstance(local.get("selection", {}), dict) else {},
        "relative_recovery_use_recovery_pool_by_bucket": (local.get("selection", {}) or {}).get("relative_recovery_use_recovery_pool_by_bucket", {}) if isinstance(local.get("selection", {}), dict) else {},
        "recovery_cert_max_hard_by_bucket": (local.get("selection", {}) or {}).get("recovery_cert_max_hard_by_bucket", {}) if isinstance(local.get("selection", {}), dict) else {},
        "recovery_cert_max_harm_by_bucket": (local.get("selection", {}) or {}).get("recovery_cert_max_harm_by_bucket", {}) if isinstance(local.get("selection", {}), dict) else {},
    }
    result["notes"] = [
        "This is a true Waymax receding-horizon loop: reset once, select an action from current SimulatorState, step the environment, then replan from the updated SimulatorState.",
        "closed_loop.label_mode=fast skips expensive per-candidate OC-MERO teacher labels inside the online loop. Use closed_loop.label_mode=selected to label only the executed candidate; use label_mode=selected_topk/coverage to label selected plus a small diagnostic top-k subset; use label_mode=all only for tiny exhaustive audits.",
        "Non-SDC actors use Waymax/default log-playback dynamics unless controlled by the environment configuration.",
    ]
    write_json(result, output_path, fsync=resume_fsync)
    _write_closed_loop_progress(
        progress_path,
        fingerprint=run_fingerprint,
        status="complete",
        completed=len(scene_results),
        total=total_rollouts,
        resumed=resumed_rollouts,
    )
    return result
