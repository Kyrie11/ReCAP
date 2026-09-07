#!/usr/bin/env python3
"""Create regime-specific publication summaries from external closed-loop runs.

The raw closed-loop evaluator intentionally emits a superset of metrics.  This
script applies the paper reporting contract: Safe publishes nominal closed-loop
quality/comfort, Near publishes closed-loop plus low-headroom recovery metrics,
and Contact publishes post-contact recovery/stability metrics only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ocrap.external_baselines.provenance import MAIN_TABLE_BY_REGIME


def _load(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def _metric(d: dict[str, Any], key: str) -> Any:
    if key in d:
        return d.get(key)
    wm = d.get("waymax_metrics") or {}
    return wm.get(key) if isinstance(wm, dict) else None


COMMON = ("method", "source", "label_modes", "num_scenes", "num_decisions")
SAFE = (
    "collision_scene_rate",
    "offroad_scene_rate",
    "minimum_clearance_m",
    "scene_min_clearance_m_median",
    "scene_min_clearance_m_p05",
    "minimum_ttc_s",
    "scene_ttc_s_median",
    "scene_ttc_s_p05",
    "acceleration_abs_p95_mps2",
    "jerk_p95",
    "yaw_rate_p95",
    "acceleration_max_mps2",
    "deceleration_max_mps2",
    "jerk_max_abs",
    "yaw_rate_max_abs",
    "route_progression_m",
    "closed_loop_bounded_NUP",
    "closed_loop_nominal_deviation",
    "intervention_rate",
    "intervention_scene_rate",
)
NEAR = (
    "collision_scene_rate",
    "offroad_scene_rate",
    "minimum_clearance_m",
    "scene_min_clearance_m_p05",
    "minimum_ttc_s",
    "scene_ttc_s_p05",
    "near_contact_exposure_rate",
    "near_contact_exposure_duration_s",
    "near_contact_exposure_episode_count",
    "near_contact_longest_exposure_run_s",
    "critical_ttc_exposure_rate",
    "critical_ttc_exposure_duration_s",
    "critical_ttc_exposure_episode_count",
    "critical_ttc_longest_exposure_run_s",
    "near_zero_clearance_exposure_rate",
    "time_to_min_clearance_s",
    "terminal_clearance_m",
    "clearance_recovery_gain_m",
    "time_to_min_ttc_s",
    "terminal_ttc_s",
    "ttc_recovery_gain_s",
    "clearance_deficit_auc_m_s",
    "ttc_deficit_auc_s2",
    "closed_loop_FRA_exec",
    "closed_loop_FRA_cand",
    "closed_loop_DRS",
    "closed_loop_ODG",
    "closed_loop_bounded_NUP",
    "acceleration_abs_p95_mps2",
    "jerk_p95",
    "yaw_rate_p95",
    "intervention_rate",
    "intervention_scene_rate",
)
CONTACT = (
    "overlap_episode_count",
    "overlap_duration_s",
    "longest_overlap_run_s",
    "post_contact_terminal_clearance_m",
    "post_contact_free_space_auc_m_s",
    "post_contact_free_space_auc_normalized_m",
    "post_contact_clearance_gain_m",
    "time_to_peak_post_contact_clearance_s",
    "post_contact_escape_scene_rate",
    "time_to_post_contact_escape_s",
    "recontact_scene_rate",
    "recontact_episode_count",
    "secondary_overlap_scene_rate",
    "new_stable_stop_scene_rate",
    "new_stable_stop_quality_scene_rate",
    "time_to_stable_stop_s",
    "time_to_stable_stop_quality_s",
    "post_contact_overlap_duration_s",
    "post_contact_overlap_rate",
    "post_contact_clearance_deficit_auc_m_s",
    "post_contact_clearance_m_mean",
    "post_contact_clearance_m_max",
    "acceleration_abs_p95_mps2",
    "jerk_p95",
    "yaw_rate_p95",
)
BY_REGIME = {"safe": SAFE, "near": NEAR, "contact": CONTACT}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--regime", choices=sorted(BY_REGIME), required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--methods", default="", help="Comma-separated override; defaults to the six main-table methods.")
    ap.add_argument("--womd-spec", default="")
    args = ap.parse_args()

    methods = [x.strip() for x in args.methods.split(",") if x.strip()] or list(MAIN_TABLE_BY_REGIME[args.regime])
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    keys = BY_REGIME[args.regime]
    for method in methods:
        path = args.run / f"closed_loop_{method}.json"
        d = _load(path)
        if d is None:
            missing.append(method)
            continue
        row: dict[str, Any] = {k: d.get(k) for k in COMMON}
        # Older artifacts used singular label_mode while current aggregates expose label_modes.
        if row.get("label_modes") is None:
            row["label_modes"] = [d.get("label_mode")] if d.get("label_mode") is not None else None
        for key in keys:
            row[key] = _metric(d, key)
        row["artifact"] = str(path)
        rows.append(row)

    contract_note = {
        "safe": "Nominal closed-loop safety, comfort, preservation and unintended intervention only; no post-contact recovery metrics.",
        "near": "Closed-loop safety plus low-headroom/extreme exposure, recovery and selected-candidate OC-RAP teacher diagnostics; no post-contact metrics.",
        "contact": "Post-contact escape, re-contact, stable-stop and free-space recovery only; ordinary nominal/near closed-loop metrics are intentionally excluded from the publication summary.",
    }[args.regime]
    doc = {
        "schema_version": 1,
        "regime": args.regime,
        "metric_contract": contract_note,
        "main_table_methods": methods,
        "womd_spec": args.womd_spec or None,
        "num_methods_found": len(rows),
        "missing_methods": missing,
        "methods": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "external_closed_loop_summary", "regime": args.regime, "output": str(args.output), "num_methods": len(rows), "missing": missing}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
