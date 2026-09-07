#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from ocrap.v48_111_constraint_native_geometry_orientation import ENGINEERING_VERSION, contract_checks

ACTIVE = [
    "scripts/run_v48_111_dcp_drfc_bcde_rifa_cngo_two_gpu.sh",
    "src/ocrap/v48_111_constraint_native_geometry_orientation.py",
    "tools/run_v48_111_constraint_native_geometry_orientation_audit.py",
    "tools/compare_v48_111_cngo.py",
    "tools/check_v48_111_runtime_code_contract.py",
    "tools/check_v48_111_pipeline_complete.py",
]

def sha(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()

def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--repo", type=Path, required=True); ap.add_argument("--output", type=Path, required=True); a = ap.parse_args()
    repo = a.repo.resolve(); errors = []; files = {}
    for rel in ACTIVE:
        p = (repo / rel).resolve(); inside = str(p).startswith(str(repo)); ok = p.is_file() and inside
        files[rel] = {"exists": p.is_file(), "inside_repo": inside, "path": str(p), "sha256": sha(p) if p.is_file() else None}
        if not ok: errors.append(f"runtime_file:{rel}")
    checks = contract_checks()
    for k, v in checks.items():
        if not v: errors.append(f"synthetic:{k}")
    out = {
        "schema": "ocrap-v48.111-cngo-runtime-code-contract-v1", "engineering_version": ENGINEERING_VERSION,
        "valid": not errors, "attribution_ready": not errors, "errors": errors, "runtime_files": files,
        "scientific_contract": {
            "audit_only": True, "constraint_native_candidate_agent_geometry": True,
            "score_family": "linear_on_fixed_constraint_native_candidate_response_features",
            "raw_candidate_dim": 156, "gap_feature_dim": 160, "flow_feature_dim": 164,
            "prefix_complete_states": 8,
            "agent_set_definition": "historical_raw_agent_tokens_current_observation",
            "active_selector": "minimum_cv_circle_clearance_over_first_8_complete_prefix_states",
            "gap_geometry": "candidate_minus_nominal_delta_h_at_candidate_and_nominal_active_agent_time_pairs",
            "flow_geometry": "candidate_minus_nominal_delta_hdot_at_same_active_agent_time_pairs",
            "global_translation_rotation_invariant_geometry": True,
            "physical_zero_preserving_rms_scaler": True,
            "candidate_identity_shuffle": True,
            "convex_closed_form_ridge": True, "strictly_convex_unique_solution": True, "iterative_optimizer_used": False,
            "ridge_lambda_rule": "1_over_axis_train_rows", "same_solver_for_base_gap_flow": True,
            "posthoc_feature_selection": False, "threshold_sweep": False, "capacity_sweep": False, "lr_or_epoch_sweep": False,
            "stage_i_parameters_trained": 0, "root_decoder_parameters_trained": 0, "source_parameters_trained": 0, "planner_parameters_trained": 0,
            "boundary_transport": False, "broad_encoder_training": False, "regime_conditioning": False, "teacher_metadata_input_to_model": False,
            "dataset_reconstruction": False,
        },
        "synthetic_checks": checks, "test_roots_read": False,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"valid": out["valid"], "errors": errors}))
    return 0 if out["valid"] else 30

if __name__ == "__main__": raise SystemExit(main())
