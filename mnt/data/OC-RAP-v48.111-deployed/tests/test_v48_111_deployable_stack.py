from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch


def _make_owner(tmp_path: Path, bad_boundary: bool = False) -> Path:
    run = tmp_path / "run"
    root = run / "candidates" / "balanced"
    (root / "model_v48_trac_sr").mkdir(parents=True)
    (root / "calibration").mkdir(parents=True)
    doc = {
        "model_state": {"direct_absolute_root_tail_source_scale": torch.zeros(1)},
        "trainable_param_prefixes": ["direct_absolute_root_tail_source_scale"],
        "direct_recovery_absolute_semantic_witness_correction": True,
        "direct_recovery_semantic_witness_active_set_alignment": True,
        "direct_recovery_semantic_witness_path_stop_alignment": False,
        "direct_recovery_semantic_witness_classlocal_transport": False,
        "direct_recovery_semantic_witness_route_alignment": True,
        "direct_recovery_semantic_witness_reentry_alignment": True,
        "direct_recovery_semantic_witness_control_projection": True,
        "direct_recovery_semantic_witness_boundary_transport": bad_boundary,
        "direct_recovery_semantic_witness_projection_fidelity_weighting": False,
        "direct_recovery_semantic_witness_active_constraint_typed_source": False,
        "direct_recovery_semantic_witness_root_tail_source": True,
        "direct_recovery_semantic_witness_structured_tail_field": False,
        "direct_recovery_semantic_witness_signed_tail_channels": False,
        "direct_recovery_semantic_witness_counterfactual_tail_response": False,
        "direct_recovery_semantic_witness_action_response_adapter": False,
        "direct_recovery_semantic_witness_action_response_state_conditioning": False,
        "direct_recovery_semantic_witness_action_root_bilinear_interaction": False,
        "direct_recovery_semantic_witness_quotient_tail_response": False,
        "direct_recovery_semantic_witness_tail_localization": True,
        "direct_recovery_semantic_witness_demand_normalized_fidelity": False,
        "direct_recovery_semantic_witness_robust_occupancy": False,
        "direct_recovery_semantic_witness_soft_occupancy_disagreement": False,
        "direct_recovery_semantic_witness_boundary_localized_occupancy_trust": False,
        "direct_recovery_semantic_witness_history_occupancy_reachability": False,
        "direct_recovery_semantic_witness_interaction_box_support": False,
        "direct_recovery_semantic_witness_interaction_hull_support": False,
        "direct_recovery_semantic_witness_interaction_anchor_support": False,
        "direct_recovery_semantic_witness_interaction_response_support": False,
        "direct_recovery_absolute_semantic_witness_feature_schema": 3,
        "direct_recovery_absolute_semantic_witness_feature_source": "projected_boundary_common_executable_recovery_witness",
        "cfg": {"training": {
            "direct_value_absolute_feasibility_truth_contract": "structural_interval_bounds",
            "direct_value_absolute_feasibility_supervision_objective": "signed_margin_interval_huber",
        }},
    }
    torch.save(doc, root / "model_v48_trac_sr" / "best.pt")
    (root / "POLICY_CONTRACT.env").write_text(
        "PROPOSAL_TOP_K=5\n"
        "SELECTION_SEMANTICS=rank_topk_then_absolute_feasibility_then_relative_filter_then_evidence_rerank\n"
        "ABSOLUTE_FEASIBILITY_MODE=learned\n"
        "ABSOLUTE_FEASIBILITY_THRESHOLD=0.5\n"
        "ABSOLUTE_FEASIBILITY_TRUTH_CONTRACT=structural_interval_bounds\n"
        "ABSOLUTE_FEASIBILITY_SUPERVISION_OBJECTIVE=signed_margin_interval_huber\n"
    )
    (root / "calibration" / "gamma_rec_by_bucket_v48.json").write_text(json.dumps({
        "gamma_rec_by_bucket": {"test_safe": 0.5, "test_near_contact": 0.4, "test_contact": 0.3}
    }))
    return run


def test_deployable_stack_accepts_v48_80_owner(tmp_path: Path):
    repo = Path(__file__).resolve().parents[1]
    run = _make_owner(tmp_path)
    out = tmp_path / "out.json"
    cp = subprocess.run([sys.executable, str(repo / "tools/check_v48_111_deployable_stack.py"),
                         "--model-run", str(run), "--variant", "balanced", "--output", str(out)])
    assert cp.returncode == 0
    doc = json.loads(out.read_text())
    assert doc["valid"] is True
    assert doc["cnro_runtime_integration"] is False
    assert doc["finetuning_allowed"] is False


def test_deployable_stack_rejects_boundary_transport(tmp_path: Path):
    repo = Path(__file__).resolve().parents[1]
    run = _make_owner(tmp_path, bad_boundary=True)
    out = tmp_path / "out.json"
    cp = subprocess.run([sys.executable, str(repo / "tools/check_v48_111_deployable_stack.py"),
                         "--model-run", str(run), "--variant", "balanced", "--output", str(out)])
    assert cp.returncode == 30
    doc = json.loads(out.read_text())
    assert doc["valid"] is False
