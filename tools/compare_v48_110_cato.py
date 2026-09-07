#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from typing import Any

ENGINEERING_VERSION="v48.110.0-OC-CATO"; V109_ENGINEERING_VERSION="v48.109.0-OC-RCSO"
AUTHORITATIVE_V109_COMPARISON_SHA256="4aa8d8846a39de6fa3797464ce3e9587c148872a7a46d5a8f118fcba9c983627"
ROLES=("dev_near","dev_contact","certificate_near","certificate_contact")

def _sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def _ok(v:Any,t:float)->bool:return v is not None and float(v)>=t
def _cross(rs:set[str],n:int)->bool:return len(rs)>=n and any("near" in x for x in rs) and any("contact" in x for x in rs)

def _action_gate(docs,space,metric):
    positive=[];top=[];roles=set();top_roles=set()
    for v,d in docs.items():
        for role in ROLES:
            m=d[f"{space}_cells"][role][f"{metric}_true"]
            if _ok(m.get("auc"),.65) and _ok(m.get("auc_vs_shuffled"),.05): positive.append([v,role]);roles.add(role)
            if _ok(m.get("top1_vs_shuffled"),.10): top.append([v,role]);top_roles.add(role)
    go=len(positive)>=6 and _cross(roles,3) and len(top)>=4 and _cross(top_roles,2)
    local_order=len(top)>=4 and _cross(top_roles,2)
    return {"go":bool(go),"local_order":bool(local_order),"positive_cells":positive,"top1_material_cells":top,"roles":sorted(roles),"top1_roles":sorted(top_roles)}

def _result_errors(o,v):
    e=[]
    checks=[
      (o.get("valid") is True,"valid"),(o.get("engineering_version")==ENGINEERING_VERSION,"version"),(o.get("variant")==v,"variant"),
      (o.get("audit_only") is True,"audit"),(o.get("convex_closed_form_ridge") is True,"convex"),(o.get("strictly_convex_unique_solution") is True,"unique"),
      (o.get("iterative_optimizer_used") is False,"no_iterative_optimizer"),(o.get("score_family")=="linear_on_fixed_candidate_agent_topology_features","score_family"),
      (o.get("nominal_zero_score_by_construction") is True,"nominal_zero"),(o.get("agent_set_candidate_invariant") is True,"agent_invariant"),
      (o.get("agent_topology_permutation_invariant") is True,"agent_perm"),(o.get("posthoc_feature_selection") is False,"posthoc"),
      (int(o.get("raw_candidate_dim",-1))==156,"candidate_dim"),(int(o.get("agent_dim",-1))==10,"agent_dim"),
      (int(o.get("nearest_dim",-1))==1716,"nearest_dim"),(int(o.get("topology_dim",-1))==3588,"topology_dim"),
      (float(o.get("max_normal_equation_residual",1.0))<=1e-7,"normal_residual"),
      (int(o.get("stage_i_parameters_trained",-1))==0,"stage_i"),(int(o.get("root_decoder_parameters_trained",-1))==0,"root"),
      (int(o.get("source_parameters_trained",-1))==0,"source"),(int(o.get("planner_parameters_trained",-1))==0,"planner"),
      (o.get("regime_conditioning") is False,"regime"),(o.get("boundary_transport") is False,"boundary"),
      (o.get("teacher_metadata_input_to_model") is False,"teacher"),(o.get("test_roots_read") is False,"test_roots"),
    ]
    for ok,n in checks:
        if not ok:e.append(f"{v}:{n}")
    for space in ("base","nearest","topology"):
        for role in ROLES:
            if role not in o.get(f"{space}_cells",{}):e.append(f"{v}:{space}:{role}")
    for ev,evo in o.get("events",{}).items():
        if float(evo.get("agent_set_candidate_delta_max_abs",1.0))>1e-6:e.append(f"{v}:{ev}:agent_delta")
        if int(evo.get("agent_mask_candidate_delta_count",1))!=0:e.append(f"{v}:{ev}:agent_mask")
    return e

def _v109_base_auc_map(c109,metric):
    out={}
    rows=((c109.get("preregistered_decision") or {}).get("diagnostic_auc_deltas_relational_minus_base") or {}).get(metric,[])
    for r in rows: out[(r.get("variant"),r.get("role"))]=r.get("base_auc")
    return out

def _deltas(docs,space_a,space_b,metric):
    out=[]
    for v in ("balanced","precision"):
        for role in ROLES:
            a=docs[v][f"{space_a}_cells"][role][f"{metric}_true"].get("auc")
            b=docs[v][f"{space_b}_cells"][role][f"{metric}_true"].get("auc")
            out.append({"variant":v,"role":role,f"{space_a}_auc":a,f"{space_b}_auc":b,f"{space_b}_minus_{space_a}":None if a is None or b is None else float(b)-float(a)})
    return out

def main()->int:
    ap=argparse.ArgumentParser()
    for k in ("balanced","precision","v109_pipeline","v109_comparison"):ap.add_argument("--"+k.replace("_","-"),dest=k,type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True);a=ap.parse_args();errors=[]
    docs={v:json.loads(getattr(a,v).read_text()) for v in ("balanced","precision")}
    for v,o in docs.items():errors+=_result_errors(o,v)
    p109=json.loads(a.v109_pipeline.read_text());c109=json.loads(a.v109_comparison.read_text());d109=c109.get("preregistered_decision") or {}
    if _sha(a.v109_comparison)!=AUTHORITATIVE_V109_COMPARISON_SHA256:errors.append("v109_comparison_sha")
    if not(p109.get("valid") and p109.get("attribution_ready") and p109.get("engineering_version")==V109_ENGINEERING_VERSION and p109.get("preregistered_status")=="RAW_SCENE_RELATIONAL_STOP"):errors.append("v109_pipeline")
    if not(c109.get("valid") and c109.get("attribution_ready") and d109.get("status")=="RAW_SCENE_RELATIONAL_STOP" and d109.get("next_branch")=="close_bilinear_relational_orientation_then_preregister_candidate_to_agent_constraint_topology_audit_no_training_or_source_sweep"):errors.append("v109_branch")
    # Base is the exact V48.109 closed-form candidate-only control; require AUC identity.
    for metric in ("support","reserve"):
        ref=_v109_base_auc_map(c109,metric)
        for v in ("balanced","precision"):
            for role in ROLES:
                got=docs[v]["base_cells"][role][f"{metric}_true"].get("auc")
                exp=ref.get((v,role))
                if got is None or exp is None or abs(float(got)-float(exp))>1e-12: errors.append(f"{v}:{role}:{metric}:v109_base_identity")
    mg_s=_action_gate(docs,"nearest","support") if not errors else {"go":False,"local_order":False,"positive_cells":[],"top1_material_cells":[],"roles":[],"top1_roles":[]}
    mg_r=_action_gate(docs,"nearest","reserve") if not errors else {"go":False,"local_order":False,"positive_cells":[],"top1_material_cells":[],"roles":[],"top1_roles":[]}
    tp_s=_action_gate(docs,"topology","support") if not errors else {"go":False,"local_order":False,"positive_cells":[],"top1_material_cells":[],"roles":[],"top1_roles":[]}
    tp_r=_action_gate(docs,"topology","reserve") if not errors else {"go":False,"local_order":False,"positive_cells":[],"top1_material_cells":[],"roles":[],"top1_roles":[]}
    if errors:
        status="V48_110_ENGINEERING_STOP";next_branch="fix_v48_110_engineering_and_rerun_same_candidate_agent_topology_audit"
    elif tp_s["go"] and tp_r["go"]:
        status="CANDIDATE_AGENT_TOPOLOGY_BOTH_AXES_GO";next_branch="active_constraint_topology_sufficient_then_preregister_one_nominal_invariant_topology_response_carrier_no_source_or_transformer_sweep"
    elif tp_s["go"] and not tp_r["go"]:
        status="CANDIDATE_AGENT_TOPOLOGY_PARTIAL_SUPPORT";next_branch="topology_support_go_reserve_stop_then_preregister_signed_debt_active_constraint_flow_audit_no_training_or_source_sweep"
    elif tp_r["go"] and not tp_s["go"]:
        status="CANDIDATE_AGENT_TOPOLOGY_PARTIAL_RESERVE";next_branch="topology_reserve_go_support_stop_then_preregister_support_establishment_active_constraint_switch_audit_no_training_or_source_sweep"
    elif tp_s["local_order"] and tp_r["local_order"]:
        status="CANDIDATE_AGENT_TOPOLOGY_LOCAL_ORDER_ONLY";next_branch="same_topology_local_order_both_axes_then_preregister_convex_pairwise_ranking_audit_no_feature_or_source_change"
    else:
        status="CANDIDATE_AGENT_TOPOLOGY_STOP";next_branch="close_coordinatewise_agent_topology_then_preregister_constraint_native_candidate_agent_geometry_audit_no_training_or_source_sweep"
    dec={
      "status":status,"next_branch":next_branch,
      "nearest_support_go":mg_s["go"],"nearest_reserve_go":mg_r["go"],"topology_support_go":tp_s["go"],"topology_reserve_go":tp_r["go"],
      "topology_support_local_order":tp_s["local_order"],"topology_reserve_local_order":tp_r["local_order"],
      "nearest_support_positive_cells":mg_s["positive_cells"],"nearest_support_top1_material_cells":mg_s["top1_material_cells"],"nearest_support_roles":mg_s["roles"],
      "nearest_reserve_positive_cells":mg_r["positive_cells"],"nearest_reserve_top1_material_cells":mg_r["top1_material_cells"],"nearest_reserve_roles":mg_r["roles"],
      "topology_support_positive_cells":tp_s["positive_cells"],"topology_support_top1_material_cells":tp_s["top1_material_cells"],"topology_support_roles":tp_s["roles"],
      "topology_reserve_positive_cells":tp_r["positive_cells"],"topology_reserve_top1_material_cells":tp_r["top1_material_cells"],"topology_reserve_roles":tp_r["roles"],
      "diagnostic_auc_deltas_nearest_minus_base":{"support":_deltas(docs,"base","nearest","support"),"reserve":_deltas(docs,"base","nearest","reserve")},
      "diagnostic_auc_deltas_topology_minus_nearest":{"support":_deltas(docs,"nearest","topology","support"),"reserve":_deltas(docs,"nearest","topology","reserve")},
      "convex_solution_owner":True,"active_set_topology_branch_owner":True,"source_training_authorized":False,"broad_encoder_training_authorized":False,
      "boundary_transport_authorized":False,"dataset_reconstruction_authorized":False,"regime_conditioned_policy_authorized":False,
    }
    out={
      "schema":"ocrap-v48.110-cato-comparison-v1","engineering_version":ENGINEERING_VERSION,"valid":not errors,"attribution_ready":not errors,"errors":errors,
      "experiment_type":"audit_only_candidate_agent_active_constraint_topology_orientation","preregistered_decision":dec,
      "planner_parameters_trained":0,"stage_i_parameters_trained":0,"root_decoder_parameters_trained":0,"source_parameters_trained":0,
      "relative_ranker_modified":False,"regime_conditioning":False,"boundary_transport":False,"teacher_metadata_input_to_model":False,"test_roots_read":False,
      "v48_109_pipeline_sha256":_sha(a.v109_pipeline),"v48_109_comparison_sha256":_sha(a.v109_comparison),
      "authoritative_v48_109_comparison_sha256":AUTHORITATIVE_V109_COMPARISON_SHA256,
    }
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"valid":out["valid"],"status":status,"errors":errors}));return 0 if out["valid"] else 30
if __name__=="__main__":raise SystemExit(main())
