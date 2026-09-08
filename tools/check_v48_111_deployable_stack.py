#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _parse_env(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _b(doc, key: str) -> bool:
    return bool(doc.get(key, False))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Fail-closed V48.111 submission deployment contract checker."
    )
    ap.add_argument("--model-run", type=Path, required=True)
    ap.add_argument("--variant", choices=("balanced", "precision"), required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    candidate_root = args.model_run / "candidates" / args.variant
    ckpt = candidate_root / "model_v48_trac_sr" / "best.pt"
    policy = candidate_root / "POLICY_CONTRACT.env"
    gamma = candidate_root / "calibration" / "gamma_rec_by_bucket_v48.json"

    errors: list[str] = []
    for p, label in ((ckpt, "checkpoint"), (policy, "policy contract"), (gamma, "bucket calibration")):
        if not p.is_file():
            errors.append(f"missing {label}: {p}")

    out = {
        "schema": "ocrap-v48.111-deployable-stack-contract-v1",
        "engineering_version": "v48.111-deployed-evidence-frozen-1",
        "variant": args.variant,
        "model_run": str(args.model_run),
        "candidate_root": str(candidate_root),
        "scientific_status": "submission_diagnostic_not_historical_source_GO",
        "finetuning_allowed": False,
        "recalibration_allowed": False,
        "cnro_runtime_integration": False,
        "regime_router": False,
        "boundary_transport": False,
        "errors": errors,
    }

    if not errors:
        doc = _load_checkpoint(ckpt)
        env = _parse_env(policy)
        train_cfg = ((doc.get("cfg") or {}).get("training") or {})

        flags = {
            "semantic_correction": _b(doc, "direct_recovery_absolute_semantic_witness_correction"),
            "active_set_alignment": _b(doc, "direct_recovery_semantic_witness_active_set_alignment"),
            "path_stop_alignment": _b(doc, "direct_recovery_semantic_witness_path_stop_alignment"),
            "classlocal_transport": _b(doc, "direct_recovery_semantic_witness_classlocal_transport"),
            "route_alignment": _b(doc, "direct_recovery_semantic_witness_route_alignment"),
            "reentry_alignment": _b(doc, "direct_recovery_semantic_witness_reentry_alignment"),
            "control_projection": _b(doc, "direct_recovery_semantic_witness_control_projection"),
            "boundary_transport": _b(doc, "direct_recovery_semantic_witness_boundary_transport"),
            "projection_fidelity": _b(doc, "direct_recovery_semantic_witness_projection_fidelity_weighting"),
            "active_constraint_typed_source": _b(doc, "direct_recovery_semantic_witness_active_constraint_typed_source"),
            "root_tail_source": _b(doc, "direct_recovery_semantic_witness_root_tail_source"),
            "structured_tail_field": _b(doc, "direct_recovery_semantic_witness_structured_tail_field"),
            "signed_tail_channels": _b(doc, "direct_recovery_semantic_witness_signed_tail_channels"),
            "counterfactual_tail_response": _b(doc, "direct_recovery_semantic_witness_counterfactual_tail_response"),
            "action_response_adapter": _b(doc, "direct_recovery_semantic_witness_action_response_adapter"),
            "action_response_state_conditioning": _b(doc, "direct_recovery_semantic_witness_action_response_state_conditioning"),
            "action_root_bilinear_interaction": _b(doc, "direct_recovery_semantic_witness_action_root_bilinear_interaction"),
            "quotient_tail_response": _b(doc, "direct_recovery_semantic_witness_quotient_tail_response"),
            "tail_localization": _b(doc, "direct_recovery_semantic_witness_tail_localization"),
            "demand_normalized_fidelity": _b(doc, "direct_recovery_semantic_witness_demand_normalized_fidelity"),
            "robust_occupancy": _b(doc, "direct_recovery_semantic_witness_robust_occupancy"),
            "soft_occupancy_disagreement": _b(doc, "direct_recovery_semantic_witness_soft_occupancy_disagreement"),
            "boundary_localized_occupancy_trust": _b(doc, "direct_recovery_semantic_witness_boundary_localized_occupancy_trust"),
            "history_occupancy_reachability": _b(doc, "direct_recovery_semantic_witness_history_occupancy_reachability"),
            "interaction_box_support": _b(doc, "direct_recovery_semantic_witness_interaction_box_support"),
            "interaction_hull_support": _b(doc, "direct_recovery_semantic_witness_interaction_hull_support"),
            "interaction_anchor_support": _b(doc, "direct_recovery_semantic_witness_interaction_anchor_support"),
            "interaction_response_support": _b(doc, "direct_recovery_semantic_witness_interaction_response_support"),
        }
        expected_flags = {
            "semantic_correction": True,
            "active_set_alignment": True,
            "path_stop_alignment": False,
            "classlocal_transport": False,
            "route_alignment": True,
            "reentry_alignment": True,
            "control_projection": True,
            "boundary_transport": False,
            "projection_fidelity": False,
            "active_constraint_typed_source": False,
            "root_tail_source": True,
            "structured_tail_field": False,
            "signed_tail_channels": False,
            "counterfactual_tail_response": False,
            "action_response_adapter": False,
            "action_response_state_conditioning": False,
            "action_root_bilinear_interaction": False,
            "quotient_tail_response": False,
            "tail_localization": True,
            "demand_normalized_fidelity": False,
            "robust_occupancy": False,
            "soft_occupancy_disagreement": False,
            "boundary_localized_occupancy_trust": False,
            "history_occupancy_reachability": False,
            "interaction_box_support": False,
            "interaction_hull_support": False,
            "interaction_anchor_support": False,
            "interaction_response_support": False,
        }
        if flags != expected_flags:
            errors.append(f"checkpoint mechanism flags differ from frozen V48.80 owner: {flags}")

        feature_schema = int(doc.get("direct_recovery_absolute_semantic_witness_feature_schema", 0) or 0)
        feature_source = str(doc.get("direct_recovery_absolute_semantic_witness_feature_source", "") or "")
        if feature_schema != 3 or feature_source != "projected_boundary_common_executable_recovery_witness":
            errors.append(
                f"unexpected semantic witness feature contract: schema={feature_schema}, source={feature_source!r}"
            )

        expected_policy = {
            "PROPOSAL_TOP_K": "5",
            "SELECTION_SEMANTICS": "rank_topk_then_absolute_feasibility_then_relative_filter_then_evidence_rerank",
            "ABSOLUTE_FEASIBILITY_MODE": "learned",
            "ABSOLUTE_FEASIBILITY_THRESHOLD": "0.5",
            "ABSOLUTE_FEASIBILITY_TRUTH_CONTRACT": "structural_interval_bounds",
            "ABSOLUTE_FEASIBILITY_SUPERVISION_OBJECTIVE": "signed_margin_interval_huber",
        }
        policy_mismatch = {k: (env.get(k), v) for k, v in expected_policy.items() if env.get(k) != v}
        if policy_mismatch:
            errors.append(f"policy contract mismatch: {policy_mismatch}")

        truth = str(train_cfg.get("direct_value_absolute_feasibility_truth_contract", "legacy_full"))
        objective = str(train_cfg.get("direct_value_absolute_feasibility_supervision_objective", "binary_sign"))
        if truth != "structural_interval_bounds" or objective != "signed_margin_interval_huber":
            errors.append(f"checkpoint truth/objective mismatch: truth={truth}, objective={objective}")

        gamma_doc = json.loads(gamma.read_text(encoding="utf-8"))
        gammas = gamma_doc.get("gamma_rec_by_bucket") or {}
        required_buckets = ("test_safe", "test_near_contact", "test_contact")
        for bucket in required_buckets:
            val = gammas.get(bucket)
            if not isinstance(val, (int, float)) or not math.isfinite(float(val)):
                errors.append(f"invalid/missing frozen calibration for {bucket}: {val!r}")

        trainable_prefixes = [str(x) for x in (doc.get("trainable_param_prefixes") or [])]
        if trainable_prefixes and trainable_prefixes != ["direct_absolute_root_tail_source_scale"]:
            errors.append(f"unexpected checkpoint trainable-prefix provenance: {trainable_prefixes}")

        out.update(
            {
                "checkpoint": str(ckpt),
                "checkpoint_sha256": _sha256(ckpt),
                "policy_contract": str(policy),
                "policy_contract_sha256": _sha256(policy),
                "calibration": str(gamma),
                "calibration_sha256": _sha256(gamma),
                "calibration_values": {k: gammas.get(k) for k in required_buckets},
                "checkpoint_mechanism_flags": flags,
                "checkpoint_mechanism_flags_expected": expected_flags,
                "semantic_witness_feature_schema": feature_schema,
                "semantic_witness_feature_source": feature_source,
                "policy_contract_expected": expected_policy,
                "absolute_feasibility_truth_contract": truth,
                "absolute_feasibility_supervision_objective": objective,
                "trainable_param_prefixes_provenance": trainable_prefixes,
                "deployment_stack": [
                    "OC-MERO observation-consistent deployability",
                    "actuator-realizable executable recovery",
                    "active-set alignment",
                    "route semantics",
                    "persistent re-entry semantics",
                    "historical V48.80 nested-tail absolute source scaffold",
                    "RIFA role isolation",
                ],
                "explicitly_not_deployed": [
                    "V48.111 CNRO audit probe",
                    "V48.110 raw active-agent topology",
                    "class-local transport",
                    "path-stop Main",
                    "projection-fidelity weighting",
                    "robust/soft occupancy branches",
                    "interaction ball/box/hull/anchor/jerk branches",
                    "active-typed or structured-tail field families",
                    "boundary transport",
                    "regime router",
                ],
            }
        )

    out["errors"] = errors
    out["valid"] = not errors
    out["attribution_ready"] = not errors
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "v48_111_deployable_stack_contract", "variant": args.variant, "valid": not errors, "output": str(args.output)}))
    return 0 if not errors else 30


if __name__ == "__main__":
    raise SystemExit(main())
