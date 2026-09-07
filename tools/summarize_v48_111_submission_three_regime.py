#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
from typing import Any

REGIME_KEYS = {
    "safe": [
        "num_scenes", "num_decisions", "collision_scene_rate", "offroad_scene_rate",
        "closed_loop_bounded_NUP", "intervention_rate",
    ],
    "near": [
        "num_scenes", "num_decisions", "collision_scene_rate", "offroad_scene_rate",
        "scene_min_clearance_m_p05", "scene_ttc_s_p05", "critical_ttc_exposure_duration_s",
        "closed_loop_bounded_NUP", "intervention_rate",
    ],
    "contact": [
        "num_scenes", "num_decisions", "post_contact_terminal_clearance_m",
        "post_contact_free_space_auc_normalized_m", "post_contact_clearance_gain_m",
        "post_contact_escape_scene_rate", "recontact_scene_rate", "secondary_overlap_scene_rate",
        "new_stable_stop_quality_scene_rate", "post_contact_overlap_duration_s", "offroad_scene_rate",
    ],
}


def read_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object: {path}")
    return obj


def scalar(v: Any) -> Any:
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--variants", required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    ap.add_argument("--output-md", type=Path, required=True)
    args = ap.parse_args()

    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    docs: dict[str, Any] = {}
    for variant in variants:
        vroot = args.root / variant
        idx_path = vroot / "OCRAP_THREE_REGIME_RUN_INDEX.json"
        if not idx_path.is_file():
            errors.append(f"missing run index for {variant}: {idx_path}")
            continue
        idx = read_json(idx_path)
        if not idx.get("complete"):
            errors.append(f"three-regime run incomplete for {variant}: {idx.get('failed_or_incomplete_regimes')}")
        vdoc: dict[str, Any] = {"complete": bool(idx.get("complete")), "regimes": {}}
        for regime, keys in REGIME_KEYS.items():
            p = vroot / regime / "closed_loop_ocrap.json"
            if not p.is_file():
                errors.append(f"missing {variant}/{regime} result: {p}")
                continue
            d = read_json(p)
            metrics = {k: scalar(d.get(k)) for k in keys}
            vdoc["regimes"][regime] = metrics
            row = {"variant": variant, "regime": regime}
            row.update(metrics)
            rows.append(row)
        docs[variant] = vdoc

    out = {
        "schema": "ocrap-v48.111-submission-three-regime-summary-v1",
        "engineering_version": "v48.111-submission-eval-1",
        "valid": not errors,
        "errors": errors,
        "scientific_contract": {
            "cnro_remains_audit_only": True,
            "frozen_checkpoint_evaluation": True,
            "finetuning_performed": False,
            "recalibration_performed": False,
            "dataset_reconstruction": False,
            "safe_near_contact_share_one_policy": True,
        },
        "variants": docs,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fieldnames = ["variant", "regime"] + sorted({k for r in rows for k in r if k not in {"variant", "regime"}})
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    md = ["# V48.111 submission three-regime metrics", ""]
    for variant in variants:
        md += [f"## {variant}", ""]
        vdoc = docs.get(variant, {})
        for regime in ("safe", "near", "contact"):
            metrics = (vdoc.get("regimes") or {}).get(regime)
            if not metrics:
                md += [f"### {regime}", "Missing/incomplete.", ""]
                continue
            md += [f"### {regime}", "", "| Metric | Value |", "|---|---:|"]
            for k, v in metrics.items():
                if isinstance(v, float): s = f"{v:.6g}"
                else: s = str(v)
                md.append(f"| `{k}` | {s} |")
            md.append("")
    if errors:
        md += ["## Errors", ""] + [f"- {e}" for e in errors] + [""]
    args.output_md.write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"valid": not errors, "rows": len(rows), "json": str(args.output_json)}))
    return 0 if not errors else 30


if __name__ == "__main__":
    raise SystemExit(main())
