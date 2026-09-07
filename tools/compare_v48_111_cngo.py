#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any

ENGINEERING_VERSION = "v48.111.0-OC-CNGO"
V110_ENGINEERING_VERSION = "v48.110.0-OC-CATO"
AUTHORITATIVE_V110_COMPARISON_SHA256 = "5bb9bbac2b5a88cb9419308804afdfce22643cd986df284224e1c9f3617e1c9d"
ROLES = ("dev_near", "dev_contact", "certificate_near", "certificate_contact")
VARIANTS = ("balanced", "precision")


def _sha(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()
def _ok(v: Any, t: float) -> bool: return v is not None and float(v) >= t

def _cell_pass(doc, space, role, metric):
    m = doc[f"{space}_cells"][role][f"{metric}_true"]
    return _ok(m.get("auc"), .65) and _ok(m.get("auc_vs_shuffled"), .05)

def _top_pass(doc, space, role, metric):
    m = doc[f"{space}_cells"][role][f"{metric}_true"]
    return _ok(m.get("top1_vs_shuffled"), .10)

def _action_gate(docs, space, metric):
    cell_pass = [[v, r] for v in VARIANTS for r in ROLES if _cell_pass(docs[v], space, r, metric)]
    top_cells = [[v, r] for v in VARIANTS for r in ROLES if _top_pass(docs[v], space, r, metric)]
    # Balanced/Precision are provenance channels for this raw-feature audit, not
    # independent statistical replicates. A unique role passes only if both
    # variants pass, preventing duplicated counts from manufacturing GO.
    roles = [r for r in ROLES if all(_cell_pass(docs[v], space, r, metric) for v in VARIANTS)]
    top_roles = [r for r in ROLES if all(_top_pass(docs[v], space, r, metric) for v in VARIANTS)]
    cross = lambda rs: any("near" in r for r in rs) and any("contact" in r for r in rs)
    go = len(roles) >= 3 and cross(roles) and len(top_roles) >= 2 and cross(top_roles)
    local = len(top_roles) >= 2 and cross(top_roles)
    return {
        "go": bool(go), "local_order": bool(local),
        "positive_cells": cell_pass, "top1_material_cells": top_cells,
        "unique_roles": roles, "unique_top1_roles": top_roles,
        "balanced_precision_not_counted_as_independent_replicates": True,
    }


def _result_errors(o, v):
    e = []
    checks = [
        (o.get("valid") is True, "valid"), (o.get("engineering_version") == ENGINEERING_VERSION, "version"), (o.get("variant") == v, "variant"),
        (o.get("audit_only") is True, "audit"), (o.get("convex_closed_form_ridge") is True, "convex"), (o.get("strictly_convex_unique_solution") is True, "unique"),
        (o.get("iterative_optimizer_used") is False, "no_iterative_optimizer"),
        (o.get("score_family") == "linear_on_fixed_constraint_native_candidate_response_features", "score_family"),
        (o.get("nominal_zero_score_by_construction") is True, "nominal_zero"), (o.get("candidate_identity_shuffle") is True, "shuffle"),
        (int(o.get("raw_candidate_dim", -1)) == 156, "candidate_dim"), (int(o.get("gap_feature_dim", -1)) == 160, "gap_dim"), (int(o.get("flow_feature_dim", -1)) == 164, "flow_dim"),
        (float(o.get("max_normal_equation_residual", 1.0)) <= 1e-7, "normal_residual"),
        (int(o.get("stage_i_parameters_trained", -1)) == 0, "stage_i"), (int(o.get("root_decoder_parameters_trained", -1)) == 0, "root"),
        (int(o.get("source_parameters_trained", -1)) == 0, "source"), (int(o.get("planner_parameters_trained", -1)) == 0, "planner"),
        (o.get("regime_conditioning") is False, "regime"), (o.get("boundary_transport") is False, "boundary"),
        (o.get("teacher_metadata_input_to_model") is False, "teacher"), (o.get("test_roots_read") is False, "test_roots"),
        (o.get("posthoc_feature_selection") is False, "posthoc"),
    ]
    for ok, n in checks:
        if not ok: e.append(f"{v}:{n}")
    for space in ("base", "gap", "flow"):
        for role in ROLES:
            if role not in o.get(f"{space}_cells", {}): e.append(f"{v}:{space}:{role}")
    for ev, evo in o.get("events", {}).items():
        if float(evo.get("agent_set_candidate_delta_max_abs", 1.0)) > 1e-6: e.append(f"{v}:{ev}:agent_delta")
        if int(evo.get("agent_mask_candidate_delta_count", 1)) != 0: e.append(f"{v}:{ev}:agent_mask")
        if int(evo.get("prefix_complete_states", -1)) != 8: e.append(f"{v}:{ev}:prefix_steps")
    return e


def _deltas(docs, a, b, metric):
    out = []
    for v in VARIANTS:
        for role in ROLES:
            x = docs[v][f"{a}_cells"][role][f"{metric}_true"].get("auc")
            y = docs[v][f"{b}_cells"][role][f"{metric}_true"].get("auc")
            out.append({"variant": v, "role": role, f"{a}_auc": x, f"{b}_auc": y, f"{b}_minus_{a}": None if x is None or y is None else float(y)-float(x)})
    return out


def _historical_deltas(docs, v110_docs, space, metric):
    out = []
    for v in VARIANTS:
        for role in ROLES:
            old = v110_docs[v]["topology_cells"][role][f"{metric}_true"].get("auc")
            new = docs[v][f"{space}_cells"][role][f"{metric}_true"].get("auc")
            out.append({"variant": v, "role": role, "v110_topology_auc": old, f"v111_{space}_auc": new, "delta": None if old is None or new is None else float(new)-float(old)})
    return out


def _material_flow_roles(docs, metric, threshold=.003):
    roles = []
    for r in ROLES:
        vals = []
        for v in VARIANTS:
            g = docs[v]["gap_cells"][r][f"{metric}_true"].get("auc")
            f = docs[v]["flow_cells"][r][f"{metric}_true"].get("auc")
            vals.append(None if g is None or f is None else float(f)-float(g))
        if all(x is not None and x >= threshold for x in vals): roles.append(r)
    return roles


def main() -> int:
    ap = argparse.ArgumentParser()
    for k in ("balanced", "precision", "v110_pipeline", "v110_comparison", "v110_balanced", "v110_precision"):
        ap.add_argument("--" + k.replace("_", "-"), dest=k, type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True); a = ap.parse_args(); errors = []
    docs = {v: json.loads(getattr(a, v).read_text()) for v in VARIANTS}
    old = {v: json.loads(getattr(a, f"v110_{v}").read_text()) for v in VARIANTS}
    for v, o in docs.items(): errors += _result_errors(o, v)
    p110 = json.loads(a.v110_pipeline.read_text()); c110 = json.loads(a.v110_comparison.read_text()); d110 = c110.get("preregistered_decision") or {}
    if _sha(a.v110_comparison) != AUTHORITATIVE_V110_COMPARISON_SHA256: errors.append("v110_comparison_sha")
    if not (p110.get("valid") and p110.get("attribution_ready") and p110.get("engineering_version") == V110_ENGINEERING_VERSION and p110.get("preregistered_status") == "CANDIDATE_AGENT_TOPOLOGY_STOP"):
        errors.append("v110_pipeline")
    if not (c110.get("valid") and c110.get("attribution_ready") and d110.get("status") == "CANDIDATE_AGENT_TOPOLOGY_STOP" and d110.get("next_branch") == "close_coordinatewise_agent_topology_then_preregister_constraint_native_candidate_agent_geometry_audit_no_training_or_source_sweep"):
        errors.append("v110_branch")
    for v in VARIANTS:
        if old[v].get("engineering_version") != V110_ENGINEERING_VERSION or old[v].get("valid") is not True: errors.append(f"v110_{v}")
        # Exact base identity with V48.110 control proves labels/rows/solver stayed aligned.
        for metric in ("support", "reserve"):
            for role in ROLES:
                got = docs[v]["base_cells"][role][f"{metric}_true"].get("auc")
                exp = old[v]["base_cells"][role][f"{metric}_true"].get("auc")
                if got is None or exp is None or abs(float(got)-float(exp)) > 1e-12:
                    errors.append(f"{v}:{role}:{metric}:v110_base_identity")

    gs = _action_gate(docs, "gap", "support") if not errors else {"go": False, "local_order": False, "unique_roles": [], "unique_top1_roles": []}
    gr = _action_gate(docs, "gap", "reserve") if not errors else {"go": False, "local_order": False, "unique_roles": [], "unique_top1_roles": []}
    fs = _action_gate(docs, "flow", "support") if not errors else {"go": False, "local_order": False, "unique_roles": [], "unique_top1_roles": []}
    fr = _action_gate(docs, "flow", "reserve") if not errors else {"go": False, "local_order": False, "unique_roles": [], "unique_top1_roles": []}
    flow_inc_s = _material_flow_roles(docs, "support") if not errors else []
    flow_inc_r = _material_flow_roles(docs, "reserve") if not errors else []

    if errors:
        status = "V48_111_ENGINEERING_STOP"; next_branch = "fix_v48_111_engineering_and_rerun_same_constraint_native_geometry_audit"
    elif fs["go"] and fr["go"]:
        status = "CONSTRAINT_NATIVE_FLOW_BOTH_AXES_GO"; next_branch = "promote_candidate_conditioned_constraint_native_recovery_orientation_then_preregister_one_nominal_invariant_main_carrier_no_capacity_sweep"
    elif gs["go"] and gr["go"]:
        status = "CONSTRAINT_NATIVE_GAP_BOTH_AXES_GO"; next_branch = "promote_signed_gap_active_constraint_response_then_keep_normal_flow_diagnostic_no_capacity_sweep"
    elif (fs["go"] or gs["go"]) and not (fr["go"] or gr["go"]):
        status = "CONSTRAINT_NATIVE_PARTIAL_SUPPORT"; next_branch = "support_geometry_go_reserve_stop_then_preregister_constraint_type_debt_owner_audit_no_geometry_or_source_sweep"
    elif (fr["go"] or gr["go"]) and not (fs["go"] or gs["go"]):
        status = "CONSTRAINT_NATIVE_PARTIAL_RESERVE"; next_branch = "reserve_geometry_go_support_stop_then_preregister_support_establishment_constraint_switch_audit_no_geometry_or_source_sweep"
    elif fs["local_order"] and fr["local_order"]:
        status = "CONSTRAINT_NATIVE_LOCAL_ORDER_ONLY"; next_branch = "same_constraint_native_feature_local_order_both_axes_then_preregister_convex_pairwise_ranking_audit_no_feature_change"
    else:
        status = "CONSTRAINT_NATIVE_GEOMETRY_STOP"; next_branch = "close_fixed_cv_circle_agent_geometry_then_preregister_constraint_type_owner_geometry_audit_no_training_or_parameter_sweep"

    train_counts = {v: docs[v].get("train_counts", {}) for v in VARIANTS}
    capacity = {}
    for v in VARIANTS:
        capacity[v] = {}
        for axis in ("support", "reserve"):
            n = int(train_counts[v].get(axis, 0) or 0)
            capacity[v][axis] = {"rows": n, "base_dim": 156, "gap_dim": 160, "flow_dim": 164,
                                 "base_dim_over_rows": None if n == 0 else 156/n,
                                 "gap_dim_over_rows": None if n == 0 else 160/n,
                                 "flow_dim_over_rows": None if n == 0 else 164/n,
                                 "v110_topology_dim_over_rows": None if n == 0 else 3588/n}
    dec = {
        "status": status, "next_branch": next_branch,
        "gap_support": gs, "gap_reserve": gr, "flow_support": fs, "flow_reserve": fr,
        "flow_minus_gap_material_unique_roles": {"support": flow_inc_s, "reserve": flow_inc_r},
        "diagnostic_auc_deltas_gap_minus_base": {"support": _deltas(docs, "base", "gap", "support"), "reserve": _deltas(docs, "base", "gap", "reserve")},
        "diagnostic_auc_deltas_flow_minus_gap": {"support": _deltas(docs, "gap", "flow", "support"), "reserve": _deltas(docs, "gap", "flow", "reserve")},
        "diagnostic_auc_deltas_v111_flow_minus_v110_topology": {"support": _historical_deltas(docs, old, "flow", "support"), "reserve": _historical_deltas(docs, old, "flow", "reserve")},
        "capacity_diagnostics": capacity,
        "unique_role_gate_primary": True, "balanced_precision_are_provenance_channels_not_independent_replicates": True,
        "candidate_conditioned_active_constraint_principle_closed": False,
        "fixed_cv_circle_coordinatewise_agent_topology_family_closed": True,
        "source_training_authorized": False, "broad_encoder_training_authorized": False,
        "boundary_transport_authorized": False, "dataset_reconstruction_authorized": False, "regime_conditioned_policy_authorized": False,
    }
    out = {
        "schema": "ocrap-v48.111-cngo-comparison-v1", "engineering_version": ENGINEERING_VERSION,
        "valid": not errors, "attribution_ready": not errors, "errors": errors,
        "experiment_type": "audit_only_constraint_native_candidate_agent_geometry_orientation",
        "preregistered_decision": dec,
        "planner_parameters_trained": 0, "stage_i_parameters_trained": 0, "root_decoder_parameters_trained": 0, "source_parameters_trained": 0,
        "relative_ranker_modified": False, "boundary_transport": False, "dataset_reconstruction": False, "regime_conditioning": False,
        "test_roots_read": False, "v110_comparison_sha256": _sha(a.v110_comparison), "authoritative_v110_comparison_sha256": AUTHORITATIVE_V110_COMPARISON_SHA256,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"valid": out["valid"], "status": status, "errors": errors}))
    return 0 if out["valid"] else 30


if __name__ == "__main__":
    raise SystemExit(main())
