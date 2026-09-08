#!/usr/bin/env python3
from __future__ import annotations

"""Calibrate the Strawn/Ayanian/Lindemann CPSF on held-out raw WOMD futures.

Unlike the v55 scalar classifier calibration, this implements Algorithm 1 of
"Conformal Predictive Safety Filter for Autonomous Systems": for each prediction
step h we compute the trajectory-prediction nonconformity

    R_{t+h} = ||tau_{t+h} - Y(tau_{0:t})||_2,

set delta_bar = delta / T, append +infinity to the calibration scores, and use
p = ceil((N+1)(1-delta_bar)).  The resulting per-horizon C_h values are frozen
for test/closed-loop use.  No OC-RAP hard_violation/harm/teacher label is read.

OC-RAP's common observation-only multimodal predictor is used as Y and collapsed
to its probability-weighted point prediction.  The cited CPSF explicitly allows
an arbitrary trajectory predictor; this keeps the predictor interface common
across Near-Contact baselines while faithfully reproducing the CP/safety-filter
mechanism.  Runtime Eq. (7) is solved over the executable candidate lattice.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

# CPSF calibration only reconstructs raw WOMD futures for conformal residuals;
# all JAX computation is forced onto CPU before Waymax/JAX is imported.
#
# Do NOT blank CUDA_VISIBLE_DEVICES here.  GPU-enabled JAX wheels may still
# initialize their CUDA PJRT plugin during discovery even with JAX_PLATFORMS=cpu;
# with CUDA_VISIBLE_DEVICES="" that discovery can emit CUDA_ERROR_NO_DEVICE.
# The launcher therefore leaves one allocated GPU visible for plugin discovery
# while JAX_PLATFORMS=cpu keeps actual calibration computation on CPU.
# Explicit caller settings still win because setdefault is used.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from ocrap.config import load_config
from ocrap.data.serialization import write_json
from ocrap.data.waymax_loader import iter_waymax_womd_scenarios, iter_waymax_womd_scenarios_selected
from ocrap.data.womd.sharded_path import resolve_womd_spec
from ocrap.external_baselines.data import group_sample_paths, load_external_sample
from ocrap.external_baselines.observed_risk import build_observed_risk_context
from ocrap.utils.geometry import transform_states_to_ego


def split_conformal_qhat(scores: np.ndarray, alpha: float) -> float:
    """Legacy helper kept for API compatibility with old unit tests."""
    scores = np.sort(np.asarray(scores, dtype=float).reshape(-1))
    if scores.size == 0:
        raise ValueError("No calibration scores")
    alpha = float(np.clip(alpha, 1e-12, 1.0))
    k = int(math.ceil((scores.size + 1) * (1.0 - alpha)))
    if k >= scores.size + 1:
        return float("inf")
    k = max(k, 1)
    return float(scores[k - 1])


def cpsf_conformal_radius(scores: np.ndarray, delta_bar: float) -> tuple[float, int]:
    """Algorithm-1 pth score with the explicit (N+1)th +infinity sentinel."""
    x = np.sort(np.asarray(scores, dtype=float).reshape(-1))
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("inf"), 1
    delta_bar = float(delta_bar)
    if not (0.0 < delta_bar < 1.0):
        raise ValueError(f"delta_bar must be in (0,1), got {delta_bar}")
    p = int(math.ceil((x.size + 1) * (1.0 - delta_bar)))
    p = max(p, 1)
    if p == x.size + 1:
        return float("inf"), p
    if p > x.size + 1:
        return float("inf"), p
    return float(x[p - 1]), p


def _scalar(d: dict[str, Any], key: str, default: Any = None) -> Any:
    try:
        v = np.asarray(d.get(key, default)).item()
        if isinstance(v, bytes):
            return v.decode("utf-8", errors="ignore")
        return v
    except Exception:
        return default


def _canonical_scene_id(value: Any) -> str:
    text = str(value or "").strip()
    return re.sub(r"__wx\d{8}$", "", text)


def _source_index_from_scene_id(value: Any) -> int | None:
    m = re.search(r"__wx(\d{8})$", str(value or "").strip())
    return int(m.group(1)) if m else None


def _raw_match_ids(raw: Any) -> set[str]:
    """Return every stable identity alias exposed by the current WOMD loader.

    Canonical OC-RAP datasets built before official ``scenario/id`` retention
    use ``waymax_<hash>`` as their scene identity.  Current loaders preserve the
    official WOMD id but also reproduce that legacy hash in metadata.  Treat
    both as aliases for the *same* raw scene; ``__wx########`` is only a source
    order hint and is deliberately stripped.
    """
    meta = getattr(raw, "metadata", {}) or {}
    return {
        x
        for x in (
            _canonical_scene_id(getattr(raw, "scenario_id", "")),
            _canonical_scene_id(meta.get("original_scenario_id")),
            _canonical_scene_id(meta.get("official_scenario_id")),
            _canonical_scene_id(meta.get("legacy_scenario_id")),
        )
        if x
    }


def _target_groups(dataset: str, split: str) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    groups = group_sample_paths(dataset, split=split)
    rows: list[dict[str, Any]] = []
    by_scene: dict[str, list[int]] = defaultdict(list)
    for gi, paths in enumerate(groups):
        if not paths:
            continue
        sample = load_external_sample(paths[0])
        scene_id = str(_scalar(sample, "scene_id", ""))
        official_id = str(_scalar(sample, "official_scenario_id", "") or "")
        original_id = str(_scalar(sample, "original_scenario_id", "") or "")
        time_index = int(_scalar(sample, "time_index", 0))
        source_index_value = _scalar(sample, "source_scenario_index", -1)
        source_index = -1 if source_index_value is None else int(source_index_value)
        # Older NPZ serialization accidentally mapped source index 0 to -1 via
        # ``value or -1``.  The persisted __wx######## suffix is an exact
        # provenance fallback, so recover it without consulting any future data.
        if source_index < 0:
            suffix_index = _source_index_from_scene_id(scene_id)
            if suffix_index is not None:
                source_index = suffix_index
        match_ids = {
            x for x in (
                _canonical_scene_id(scene_id),
                _canonical_scene_id(official_id),
                _canonical_scene_id(original_id),
            ) if x
        }
        if not match_ids:
            continue
        scene_key = _canonical_scene_id(official_id or original_id or scene_id)
        rows.append({
            "group_index": gi,
            "scene_id": scene_id,
            "scene_key": scene_key,
            "match_ids": match_ids,
            "source_scenario_index": source_index,
            "time_index": time_index,
            "sample": sample,
        })
        by_scene[scene_key].append(len(rows) - 1)
    return rows, dict(by_scene)


def _visible_actor_indices(sample: dict[str, Any]) -> list[int]:
    hist = np.asarray(sample.get("agent_history", np.zeros((0, 0, 0))), dtype=float)
    valid = np.asarray(sample.get("agent_valid", np.zeros((0, 0))), dtype=bool)
    if hist.ndim != 3 or valid.ndim != 2 or hist.shape[:2] != valid.shape:
        return []
    return [a for a in range(1, hist.shape[1]) if bool(np.any(valid[:, a]))]


def _point_prediction(sample: dict[str, Any], cfg: dict[str, Any], horizon: int) -> tuple[np.ndarray, list[int]]:
    # actor_xy [modes, visible_actors, time, 2], with time[0] = current observation.
    context = build_observed_risk_context(sample, cfg, horizon=horizon + 1)
    if context.actor_xy.ndim != 4 or context.actor_xy.shape[1] == 0:
        return np.zeros((0, horizon + 1, 2), dtype=float), []
    point = np.tensordot(np.asarray(context.weights, dtype=float), context.actor_xy, axes=(0, 0))
    return np.asarray(point, dtype=float), _visible_actor_indices(sample)


def _group_nonconformity(
    raw,
    row: dict[str, Any],
    cfg: dict[str, Any],
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    sample = row["sample"]
    t = int(row["time_index"])
    if t < 0 or t >= raw.agent_states.shape[0]:
        return np.full(horizon, np.nan), np.zeros(horizon, dtype=int), float("inf")
    point, actor_ids = _point_prediction(sample, cfg, horizon)
    if point.shape[0] == 0 or not actor_ids:
        return np.full(horizon, np.nan), np.zeros(horizon, dtype=int), 0.0

    # OC-RAP's data builder deterministically reorders tracks as
    #   [sdc_track_index] + [all remaining raw WOMD track indices]
    # before truncating to max_agents (see data/build/history.py).  The serialized
    # NPZ intentionally does not store this private order, so reconstruct it from
    # the raw scene and the sample's agent dimension.  Never use sample actor
    # indices directly against raw.agent_states: that silently pairs the conformal
    # residual with the wrong actor whenever the raw SDC track is not index 0.
    hist = np.asarray(sample.get("agent_history", np.zeros((0, 0, 0))), dtype=float)
    valid = np.asarray(sample.get("agent_valid", np.zeros((0, 0))), dtype=bool)
    sdc = int(raw.sdc_track_index)
    raw_order = [sdc] + [i for i in range(raw.agent_states.shape[1]) if i != sdc]
    raw_order = raw_order[: hist.shape[1] if hist.ndim == 3 else 0]
    if not raw_order or raw_order[0] != sdc:
        return np.full(horizon, np.nan), np.zeros(horizon, dtype=int), float("inf")

    # Verify that raw scene/order and the OC-RAP sample refer to the same current
    # observation before using ground-truth future. This catches wrong WOMD source
    # families (e.g. validation_interactive vs standard validation) fail-closed.
    ego_raw = raw.agent_states[t, int(raw.sdc_track_index)]
    end = min(raw.agent_states.shape[0], t + horizon + 1)
    future = raw.agent_states[t:end]
    future_valid = raw.agent_valid[t:end]
    future_e = transform_states_to_ego(future, ego_raw)
    alignment = 0.0
    if hist.ndim == 3 and valid.ndim == 2 and hist.shape[:2] == valid.shape and hist.shape[0]:
        cur = hist[-1]
        errs = []
        for sample_i in [0, *actor_ids]:
            if sample_i >= cur.shape[0] or sample_i >= len(raw_order):
                continue
            raw_i = int(raw_order[sample_i])
            if raw_i < future_e.shape[1] and bool(valid[-1, sample_i]) and bool(future_valid[0, raw_i]):
                errs.append(float(np.linalg.norm(cur[sample_i, :2] - future_e[0, raw_i, :2])))
        alignment = max(errs) if errs else 0.0

    scores = np.full(horizon, np.nan, dtype=float)
    counts = np.zeros(horizon, dtype=int)
    # The paper's tau is the joint 2m-dimensional agent state.  Use the L2 norm
    # across every visible actor that has a valid ground-truth state at this h.
    # The runtime Eq. (7) can then safely apply the same C_h to each individual
    # actor because every component error is bounded by the joint norm.
    for h in range(1, horizon + 1):
        if h >= future_e.shape[0] or h >= future_valid.shape[0]:
            continue
        pairs = []
        for pred_i, sample_i in enumerate(actor_ids):
            if pred_i >= point.shape[0] or sample_i >= len(raw_order):
                continue
            raw_i = int(raw_order[sample_i])
            if raw_i >= future_e.shape[1] or not bool(future_valid[h, raw_i]):
                continue
            pairs.append(future_e[h, raw_i, :2] - point[pred_i, h, :2])
        if pairs:
            e = np.asarray(pairs, dtype=float).reshape(-1)
            scores[h - 1] = float(np.linalg.norm(e, ord=2))
            counts[h - 1] = len(pairs)
    return scores, counts, alignment


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="calibration")
    ap.add_argument("--womd-pattern", required=True, help="Raw WOMD source used to build the calibration split; normally standard validation@150")
    ap.add_argument("--delta", type=float, default=None, help="Total CPSF failure probability delta")
    ap.add_argument("--alpha", type=float, default=None, help="Backward-compatible alias for --delta")
    ap.add_argument("--prediction-horizon", type=int, default=None)
    ap.add_argument("--mission-horizon", type=int, default=None)
    ap.add_argument("--calibration-unit", choices=("group", "scene_max"), default="group")
    ap.add_argument("--alignment-tolerance-m", type=float, default=0.25)
    ap.add_argument(
        "--source-index-policy",
        choices=("hint", "strict"),
        default="hint",
        help=(
            "Treat source_scenario_index as a verified acceleration hint (default) or "
            "fail immediately on an index/identity mismatch. Stable scenario aliases "
            "remain authoritative in both modes."
        ),
    )
    ap.add_argument("--allow-infinite", action="store_true")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    pcfg = ((cfg.get("external_baselines", {}) or {}).get("policy", {}) or {})
    delta = args.delta if args.delta is not None else (args.alpha if args.alpha is not None else 0.10)
    delta = float(delta)
    if not (0.0 < delta < 1.0):
        raise SystemExit(f"delta must be in (0,1), got {delta}")
    H = int(args.prediction_horizon or pcfg.get("cpsf_prediction_horizon_steps", 7))
    Tmission = int(args.mission_horizon or pcfg.get("cpsf_mission_horizon_steps", 40))
    if H <= 0 or Tmission <= 0:
        raise SystemExit("prediction/mission horizons must be positive")
    delta_bar = delta / float(Tmission)

    rows, by_scene = _target_groups(args.dataset, args.split)
    if not rows:
        raise SystemExit(f"No grouped samples in {args.dataset!r} for split={args.split!r}")

    resolution = resolve_womd_spec(args.womd_pattern)
    if not resolution.valid:
        raise SystemExit(f"Invalid WOMD spec for CPSF calibration: {resolution.as_dict()}")

    scores_by_group: dict[int, np.ndarray] = {}
    actors_by_group: dict[int, np.ndarray] = {}
    alignments: list[float] = []
    matched_scene_ids: set[str] = set()
    target_ids = set(by_scene)
    unresolved_rows: set[int] = set(range(len(rows)))
    source_index_verified_groups = 0
    identity_fallback_groups = 0
    source_index_mismatches: list[dict[str, Any]] = []
    source_index_mismatch_count = 0

    # calibration_near_contact is built from WOMD TFExample shards.  Those
    # records are tensorflow.Example messages, *not* Scenario protos.  Reuse the
    # exact Waymax TFExample preprocessing path used by dataset construction so
    # object truncation/order, SDC index and state layout are identical.
    #
    # ``source_scenario_index`` is intentionally only an acceleration hint.  It
    # is not a stable scene identity across loader/version/path-order changes.
    # A selected replay is accepted only after an official/legacy-id alias
    # check.  Legacy pre-v48.28 datasets contain ``waymax_<hash>`` identities;
    # current raw loaders expose the same value as ``legacy_scenario_id`` even
    # when ``scenario_id`` is now the official WOMD id.  If an index hint truly
    # points at a different scene, fall back to identity matching rather than
    # either using the wrong future or aborting the whole calibration.
    by_source: dict[int, list[int]] = defaultdict(list)
    for row_i, row in enumerate(rows):
        idx = int(row.get("source_scenario_index", -1))
        if idx >= 0:
            by_source[idx].append(row_i)

    def consume(
        raw: Any,
        row_indices: list[int],
        *,
        match_mode: str,
        strict_source_index: bool = False,
    ) -> int:
        nonlocal source_index_verified_groups, identity_fallback_groups, source_index_mismatch_count
        raw_ids = _raw_match_ids(raw)
        consumed = 0
        for row_i in row_indices:
            if row_i not in unresolved_rows:
                continue
            row = rows[row_i]
            if raw_ids and row["match_ids"] and not (raw_ids & row["match_ids"]):
                mismatch = {
                    "source_index": int(row.get("source_scenario_index", -1)),
                    "sample_ids": sorted(row["match_ids"]),
                    "raw_ids": sorted(raw_ids),
                }
                if strict_source_index:
                    raise RuntimeError(
                        "CPSF source-index provenance mismatch: "
                        f"source_index={mismatch['source_index']} "
                        f"sample_ids={mismatch['sample_ids']} raw_ids={mismatch['raw_ids']}. "
                        "Stable scenario identity disagrees with the source-index hint."
                    )
                source_index_mismatch_count += 1
                if len(source_index_mismatches) < 50:
                    source_index_mismatches.append(mismatch)
                continue
            sid = str(row["scene_key"])
            score, counts, alignment = _group_nonconformity(raw, row, cfg, H)
            if alignment > float(args.alignment_tolerance_m):
                raise RuntimeError(
                    f"CPSF raw/sample alignment failed for scene={sid} t={row['time_index']}: "
                    f"max_current_position_error={alignment:.3f}m > {args.alignment_tolerance_m:.3f}m. "
                    "Check that --womd-pattern matches the WOMD source used to build calibration_near_contact."
                )
            scores_by_group[row_i] = score
            actors_by_group[row_i] = counts
            alignments.append(alignment)
            matched_scene_ids.add(sid)
            unresolved_rows.discard(row_i)
            consumed += 1
            if match_mode == "source_index_verified":
                source_index_verified_groups += 1
            elif match_mode == "identity_fallback":
                identity_fallback_groups += 1
        return consumed

    # Fast path: replay historical indices, but verify every replay against all
    # current raw identity aliases (official + legacy).  This is exact for the
    # canonical dataset while remaining migration-safe after v48.28 id retention.
    if by_source and all(int(r.get("source_scenario_index", -1)) >= 0 for r in rows):
        requested = sorted(by_source)
        for raw in iter_waymax_womd_scenarios_selected(args.womd_pattern, requested, parser_cfg=cfg):
            raw_idx = int((getattr(raw, "metadata", {}) or {}).get("_waymax_scenario_index", -1))
            if raw_idx in by_source:
                consume(
                    raw,
                    by_source[raw_idx],
                    match_mode="source_index_verified",
                    strict_source_index=(args.source_index_policy == "strict"),
                )

    # Identity fallback is authoritative.  It serves two cases: legacy datasets
    # whose record-order hint no longer agrees with the current loader, and
    # datasets without a source index.  Match on official/legacy aliases and
    # materialize only rows that are still unresolved.
    if unresolved_rows:
        id_to_rows: dict[str, list[int]] = defaultdict(list)
        for row_i in sorted(unresolved_rows):
            for sid in rows[row_i]["match_ids"]:
                id_to_rows[sid].append(row_i)
        for raw in iter_waymax_womd_scenarios(args.womd_pattern, max_scenarios=None, parser_cfg=cfg):
            raw_ids = _raw_match_ids(raw)
            row_indices = sorted(
                {i for sid in raw_ids for i in id_to_rows.get(sid, []) if i in unresolved_rows}
            )
            if row_indices:
                consume(raw, row_indices, match_mode="identity_fallback")
            if not unresolved_rows:
                break

    if unresolved_rows:
        unresolved_examples = [
            {
                "scene": rows[i]["scene_key"],
                "time_index": int(rows[i]["time_index"]),
                "source_scenario_index": int(rows[i].get("source_scenario_index", -1)),
                "match_ids": sorted(rows[i]["match_ids"]),
            }
            for i in sorted(unresolved_rows)[:10]
        ]
        raise RuntimeError(
            f"CPSF calibration matched {len(scores_by_group)}/{len(rows)} planning-decision groups; "
            f"unresolved examples={unresolved_examples}. The raw WOMD source does not contain the "
            "stable official/legacy identities required by calibration_near_contact."
        )

    missing_scenes = sorted(target_ids - matched_scene_ids)
    if missing_scenes:
        raise RuntimeError(
            f"CPSF calibration matched {len(matched_scene_ids)}/{len(target_ids)} scenes; "
            f"missing examples={missing_scenes[:10]}. Raw WOMD source does not match the calibration dataset."
        )

    if args.calibration_unit == "scene_max":
        # Strict clustered variant: one exchangeable trajectory score per WOMD
        # scene, taking the worst calibration decision in that scene per horizon.
        per_h: list[list[float]] = [[] for _ in range(H)]
        for sid, inds in by_scene.items():
            for h in range(H):
                vals = [float(scores_by_group[i][h]) for i in inds if i in scores_by_group and np.isfinite(scores_by_group[i][h])]
                if vals:
                    per_h[h].append(max(vals))
    else:
        # The OC-RAP experimental unit is a scene/time planning decision group;
        # importantly, candidate duplicates are never counted as extra CP data.
        per_h = [[] for _ in range(H)]
        for i in range(len(rows)):
            score = scores_by_group.get(i)
            if score is None:
                continue
            for h in range(H):
                if np.isfinite(score[h]):
                    per_h[h].append(float(score[h]))

    intervals: list[float] = []
    quantile_indices: list[int] = []
    num_scores: list[int] = []
    for h in range(H):
        vals = np.asarray(per_h[h], dtype=float)
        radius, pidx = cpsf_conformal_radius(vals, delta_bar)
        intervals.append(radius)
        quantile_indices.append(int(pidx))
        num_scores.append(int(np.isfinite(vals).sum()))

    infinite = [i + 1 for i, x in enumerate(intervals) if not np.isfinite(x)]
    if infinite and not args.allow_infinite:
        min_n = min(num_scores[i - 1] for i in infinite)
        required_delta = Tmission / float(min_n + 1) if min_n >= 0 else float("inf")
        raise RuntimeError(
            "Exact CPSF finite-sample quantile is +inf at horizons " + str(infinite) + ". "
            f"delta={delta}, T={Tmission}, delta_bar={delta_bar:.6g}, min_N={min_n}. "
            f"With this calibration size, choose delta > {required_delta:.6g}, use a shorter mission horizon, "
            "or collect more independent calibration trajectories. The implementation will not silently clip the paper's +inf sentinel."
        )

    config_fingerprint = hashlib.sha256(
        json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()
    actor_counts = np.concatenate([x for x in actors_by_group.values()]) if actors_by_group else np.zeros(0)
    result = {
        "protocol": "strawn_ayanian_lindemann_cpsf_algorithm1_per_horizon_raw_womd_nonconformity",
        "requested_config_fingerprint": config_fingerprint,
        "dataset": str(Path(args.dataset)),
        "split": args.split,
        "womd_pattern": str(args.womd_pattern),
        "womd_resolved_files": len(resolution.files),
        "delta": delta,
        "alpha": delta,  # backward-readable alias; semantically this is CPSF delta.
        "mission_horizon": Tmission,
        "prediction_horizon": H,
        "mission_horizon_T": Tmission,
        "prediction_horizon_H": H,
        "delta_bar": delta_bar,
        "calibration_unit": args.calibration_unit,
        "exchangeability_scope": (
            "independent_womd_scene_with_within_scene_horizonwise_max" if args.calibration_unit == "scene_max"
            else "ocrap_planning_decision_group; multiple groups may originate from one WOMD scene"
        ),
        "formal_exchangeability_note": (
            "scene_max is the stricter clustered adaptation to the paper's independently sampled trajectory assumption"
            if args.calibration_unit == "scene_max" else
            "group mode preserves the OC-RAP planning-decision evaluation unit but does not make different times from the same scene independent; do not claim the paper's iid trajectory guarantee without this qualification"
        ),
        "num_target_groups": len(rows),
        "num_target_scenes": len(target_ids),
        "num_matched_groups": len(scores_by_group),
        "num_matched_scenes": len(matched_scene_ids),
        "source_index_policy": args.source_index_policy,
        "source_index_verified_groups": int(source_index_verified_groups),
        "identity_fallback_groups": int(identity_fallback_groups),
        "source_index_mismatch_count": int(source_index_mismatch_count),
        "source_index_mismatch_examples": source_index_mismatches[:10],
        "scene_identity_policy": "official_or_legacy_alias_authoritative; source_scenario_index_hint_only",
        "num_scores_by_horizon": num_scores,
        "quantile_index_p_by_horizon": quantile_indices,
        "conformal_prediction_intervals_m": intervals,
        "infinite_horizons": infinite,
        "mean_visible_future_actors": float(np.mean(actor_counts[actor_counts > 0])) if np.any(actor_counts > 0) else 0.0,
        "max_current_alignment_error_m": float(max(alignments) if alignments else 0.0),
        "teacher_labels_used": False,
        "test_labels_used": False,
        "prediction_model": "shared_observation_only_multimodal_kinematic_predictor_probability_weighted_point_forecast",
        "note": (
            "C_h uses the paper's joint-agent L2 nonconformity and explicit (N+1)th infinity convention. "
            "Only calibration_near_contact raw WOMD futures are used; hard_violation/harm/root teacher tensors are not read."
        ),
    }
    write_json(result, args.output)
    print(result)


if __name__ == "__main__":
    main()
