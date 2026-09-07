#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from typing import Any

ENGINEERING_VERSION="v48.109.0-OC-RCSO"; V108_ENGINEERING_VERSION="v48.108.1-OC-RPAP"
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
    return {"go":bool(go),"positive_cells":positive,"top1_material_cells":top,"roles":sorted(roles),"top1_roles":sorted(top_roles)}

def _result_errors(o,v):
    e=[]
    checks=[
      (o.get("valid") is True,"valid"),(o.get("engineering_version")==ENGINEERING_VERSION,"version"),(o.get("variant")==v,"variant"),
      (o.get("audit_only") is True,"audit"),(o.get("convex_closed_form_ridge") is True,"convex"),(o.get("strictly_convex_unique_solution") is True,"unique"),
      (o.get("iterative_optimizer_used") is False,"no_iterative_optimizer"),(o.get("score_family")=="u_T_w_plus_u_T_W_c","score_family"),
      (o.get("nominal_zero_score_by_construction") is True,"nominal_zero"),(o.get("scene_context_candidate_invariant") is True,"scene_invariant"),
      (o.get("agent_set_summary_permutation_invariant") is True,"agent_perm"),(o.get("posthoc_feature_selection") is False,"posthoc"),
      (int(o.get("raw_candidate_dim",-1))==156,"candidate_dim"),(int(o.get("raw_scene_context_dim",-1))==240,"context_dim"),(int(o.get("relational_dim",-1))==37596,"rel_dim"),
      (float(o.get("max_normal_equation_residual",1.0))<=1e-7,"normal_residual"),
      (int(o.get("stage_i_parameters_trained",-1))==0,"stage_i"),(int(o.get("root_decoder_parameters_trained",-1))==0,"root"),
      (int(o.get("source_parameters_trained",-1))==0,"source"),(int(o.get("planner_parameters_trained",-1))==0,"planner"),
      (o.get("regime_conditioning") is False,"regime"),(o.get("boundary_transport") is False,"boundary"),
      (o.get("teacher_metadata_input_to_model") is False,"teacher"),(o.get("test_roots_read") is False,"test_roots"),
    ]
    for ok,n in checks:
        if not ok:e.append(f"{v}:{n}")
    for space in ("base","relational"):
        for role in ROLES:
            if role not in o.get(f"{space}_cells",{}):e.append(f"{v}:{space}:{role}")
    for ev,evo in o.get("events",{}).items():
        if float(evo.get("scene_context_candidate_delta_max_abs",1.0))>1e-6:e.append(f"{v}:{ev}:scene_delta")
    return e

def _deltas(docs,metric):
    out=[]
    for v in ("balanced","precision"):
        for role in ROLES:
            b=docs[v]["base_cells"][role][f"{metric}_true"].get("auc")
            r=docs[v]["relational_cells"][role][f"{metric}_true"].get("auc")
            out.append({"variant":v,"role":role,"base_auc":b,"relational_auc":r,"relational_minus_base":None if b is None or r is None else float(r)-float(b)})
    return out

def main()->int:
    ap=argparse.ArgumentParser()
    for k in ("balanced","precision","v108_pipeline","v108_comparison"):ap.add_argument("--"+k.replace("_","-"),dest=k,type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True);a=ap.parse_args();errors=[]
    docs={v:json.loads(getattr(a,v).read_text()) for v in ("balanced","precision")}
    for v,o in docs.items():errors+=_result_errors(o,v)
    p108=json.loads(a.v108_pipeline.read_text());c108=json.loads(a.v108_comparison.read_text());d108=c108.get("preregistered_decision") or {}
    if not(p108.get("valid") and p108.get("attribution_ready") and p108.get("engineering_version")==V108_ENGINEERING_VERSION and p108.get("preregistered_status")=="RAW_ACTION_PATHWAY_STOP"):errors.append("v108_pipeline")
    if not(c108.get("valid") and c108.get("attribution_ready") and d108.get("status")=="RAW_ACTION_PATHWAY_STOP" and d108.get("projection_structural_injectivity") is True and d108.get("next_branch")=="raw_candidate_pathway_insufficient_then_preregister_raw_candidate_scene_relational_interaction_audit_no_training_or_source_sweep"):errors.append("v108_branch")
    bs=_action_gate(docs,"base","support") if not errors else {"go":False,"positive_cells":[],"top1_material_cells":[],"roles":[],"top1_roles":[]}
    br=_action_gate(docs,"base","reserve") if not errors else {"go":False,"positive_cells":[],"top1_material_cells":[],"roles":[],"top1_roles":[]}
    rs=_action_gate(docs,"relational","support") if not errors else {"go":False,"positive_cells":[],"top1_material_cells":[],"roles":[],"top1_roles":[]}
    rr=_action_gate(docs,"relational","reserve") if not errors else {"go":False,"positive_cells":[],"top1_material_cells":[],"roles":[],"top1_roles":[]}
    if errors:
        status="V48_109_ENGINEERING_STOP";next_branch="fix_v48_109_engineering_and_rerun_same_convex_relational_audit"
    elif bs["go"] and br["go"]:
        status="RAW_CONVEX_BASE_BOTH_AXES_GO";next_branch="raw_convex_base_sufficient_then_preregister_one_nominal_invariant_candidate_only_convex_response_carrier_no_transformer_or_source_sweep"
    elif rs["go"] and rr["go"]:
        status="RAW_SCENE_RELATIONAL_BOTH_AXES_GO";next_branch="relational_orientation_sufficient_then_preregister_one_nominal_invariant_context_conditioned_response_carrier_no_transformer_or_source_sweep"
    elif rs["go"] and not rr["go"]:
        status="RAW_SCENE_RELATIONAL_PARTIAL_SUPPORT";next_branch="relational_support_go_reserve_stop_then_preregister_signed_debt_flow_relational_structure_audit_no_training_or_source_sweep"
    elif rr["go"] and not rs["go"]:
        status="RAW_SCENE_RELATIONAL_PARTIAL_RESERVE";next_branch="relational_reserve_go_support_stop_then_preregister_support_establishment_constraint_topology_audit_no_training_or_source_sweep"
    else:
        status="RAW_SCENE_RELATIONAL_STOP";next_branch="close_bilinear_relational_orientation_then_preregister_candidate_to_agent_constraint_topology_audit_no_training_or_source_sweep"
    dec={
      "status":status,"next_branch":next_branch,
      "base_support_go":bs["go"],"base_reserve_go":br["go"],"relational_support_go":rs["go"],"relational_reserve_go":rr["go"],
      "base_support_positive_cells":bs["positive_cells"],"base_support_top1_material_cells":bs["top1_material_cells"],"base_support_roles":bs["roles"],
      "base_reserve_positive_cells":br["positive_cells"],"base_reserve_top1_material_cells":br["top1_material_cells"],"base_reserve_roles":br["roles"],
      "relational_support_positive_cells":rs["positive_cells"],"relational_support_top1_material_cells":rs["top1_material_cells"],"relational_support_roles":rs["roles"],
      "relational_reserve_positive_cells":rr["positive_cells"],"relational_reserve_top1_material_cells":rr["top1_material_cells"],"relational_reserve_roles":rr["roles"],
      "diagnostic_auc_deltas_relational_minus_base":{"support":_deltas(docs,"support"),"reserve":_deltas(docs,"reserve")},
      "convex_solution_owner":True,"source_training_authorized":False,"broad_encoder_training_authorized":False,"boundary_transport_authorized":False,
      "dataset_reconstruction_authorized":False,"regime_conditioned_policy_authorized":False,
    }
    out={
      "schema":"ocrap-v48.109-rcso-comparison-v1","engineering_version":ENGINEERING_VERSION,"valid":not errors,"attribution_ready":not errors,"errors":errors,
      "experiment_type":"audit_only_raw_candidate_scene_convex_relational_orientation","preregistered_decision":dec,
      "planner_parameters_trained":0,"stage_i_parameters_trained":0,"root_decoder_parameters_trained":0,"source_parameters_trained":0,
      "relative_ranker_modified":False,"regime_conditioning":False,"boundary_transport":False,"teacher_metadata_input_to_model":False,"test_roots_read":False,
      "v48_108_pipeline_sha256":_sha(a.v108_pipeline),"v48_108_comparison_sha256":_sha(a.v108_comparison),
    }
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"valid":out["valid"],"status":status,"errors":errors}));return 0 if out["valid"] else 30
if __name__=="__main__":raise SystemExit(main())
