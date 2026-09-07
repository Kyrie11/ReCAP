#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from typing import Any
from ocrap.v48_108_raw_to_projected_action_pathway_audit import projection_structural_injectivity_event

ENGINEERING_VERSION="v48.108.1-OC-RPAP"; V107_ENGINEERING_VERSION="v48.107.0-OC-FNAO"
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
    checks=[(o.get("valid") is True,"valid"),(o.get("engineering_version")==ENGINEERING_VERSION,"version"),(o.get("variant")==v,"variant"),
      (o.get("audit_only") is True,"audit"),(o.get("same_v48_102_target_specific_probe_recipe") is True,"probe_recipe"),(o.get("raw_pathway_fixed_by_encoder_layout") is True,"fixed_path"),(o.get("posthoc_token_selection") is False,"posthoc"),(o.get("projected_delta_block_map_absolute_error_diagnostic_only") is True,"fp32_abs_diagnostic_only"),
      (int(o.get("stage_i_parameters_trained",-1))==0,"stage_i"),(int(o.get("root_decoder_parameters_trained",-1))==0,"root"),(int(o.get("source_parameters_trained",-1))==0,"source"),(int(o.get("planner_parameters_trained",-1))==0,"planner"),
      (o.get("regime_conditioning") is False,"regime"),(o.get("boundary_transport") is False,"boundary"),(o.get("teacher_metadata_input_to_model") is False,"teacher"),(o.get("test_roots_read") is False,"test_roots")]
    for ok,n in checks:
        if not ok:e.append(f"{v}:{n}")
    for space in ("raw","projected"):
        for role in ROLES:
            if role not in o.get(f"{space}_cells",{}):e.append(f"{v}:{space}:{role}")
    return e

def _deltas(docs,metric):
    out=[]
    for v in ("balanced","precision"):
        for role in ROLES:
            r=docs[v]["raw_cells"][role][f"{metric}_true"].get("auc")
            p=docs[v]["projected_cells"][role][f"{metric}_true"].get("auc")
            out.append({"variant":v,"role":role,"raw_auc":r,"projected_auc":p,"projected_minus_raw":None if r is None or p is None else float(p)-float(r)})
    return out

def main()->int:
    ap=argparse.ArgumentParser()
    for k in ("balanced","precision","v107_pipeline","v107_comparison"):ap.add_argument("--"+k.replace("_","-"),dest=k,type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True);a=ap.parse_args();errors=[]
    docs={v:json.loads(getattr(a,v).read_text()) for v in ("balanced","precision")}
    for v,o in docs.items():errors+=_result_errors(o,v)
    p107=json.loads(a.v107_pipeline.read_text());c107=json.loads(a.v107_comparison.read_text());d107=c107.get("preregistered_decision") or {}
    if not(p107.get("valid") and p107.get("attribution_ready") and p107.get("engineering_version")==V107_ENGINEERING_VERSION and p107.get("preregistered_status")=="FIRST_BLOCK_NOMINAL_INVARIANT_ACTION_ORIENTATION_STOP"):errors.append("v107_pipeline")
    if not(c107.get("valid") and c107.get("attribution_ready") and d107.get("status")=="FIRST_BLOCK_NOMINAL_INVARIANT_ACTION_ORIENTATION_STOP" and d107.get("next_branch")=="close_first_block_orientation_then_preregister_raw_to_projected_action_pathway_audit_no_broad_encoder_or_source_sweep"):errors.append("v107_branch")
    event_structural={v:bool(all(projection_structural_injectivity_event(e) for e in o.get("events",{}).values())) for v,o in docs.items()}
    for v,o in docs.items():
        if bool(o.get("projection_structural_injectivity")) != event_structural[v]: errors.append(f"{v}:structural_flag_mismatch")
    structural=bool(not errors and all(event_structural.values()))
    rs=_action_gate(docs,"raw","support") if not errors else {"go":False,"positive_cells":[],"top1_material_cells":[],"roles":[],"top1_roles":[]}
    rr=_action_gate(docs,"raw","reserve") if not errors else {"go":False,"positive_cells":[],"top1_material_cells":[],"roles":[],"top1_roles":[]}
    ps=_action_gate(docs,"projected","support") if not errors else {"go":False,"positive_cells":[],"top1_material_cells":[],"roles":[],"top1_roles":[]}
    pr=_action_gate(docs,"projected","reserve") if not errors else {"go":False,"positive_cells":[],"top1_material_cells":[],"roles":[],"top1_roles":[]}
    if errors:
        status="V48_108_ENGINEERING_STOP";next_branch="fix_v48_108_engineering_and_rerun_same_raw_to_projected_audit"
    elif not structural:
        status="RAW_TO_PROJECTED_ACTION_PATHWAY_STRUCTURAL_STOP";next_branch="projection_not_injective_then_preregister_minimal_projection_preservation_repair_no_transformer_or_source_sweep"
    elif rs["go"] and rr["go"]:
        status="RAW_ACTION_PATHWAY_BOTH_AXES_GO";next_branch="raw_both_axes_sufficient_then_preregister_one_direct_nominal_invariant_projected_path_response_carrier_no_transformer_or_source_sweep"
    elif rs["go"] and not rr["go"]:
        status="RAW_ACTION_PATHWAY_PARTIAL_SUPPORT";next_branch="raw_support_sufficient_reserve_stop_then_preregister_raw_candidate_scene_reserve_interaction_audit_no_training_or_source_sweep"
    elif rr["go"] and not rs["go"]:
        status="RAW_ACTION_PATHWAY_PARTIAL_RESERVE";next_branch="raw_reserve_sufficient_support_stop_then_preregister_raw_candidate_scene_support_interaction_audit_no_training_or_source_sweep"
    else:
        status="RAW_ACTION_PATHWAY_STOP";next_branch="raw_candidate_pathway_insufficient_then_preregister_raw_candidate_scene_relational_interaction_audit_no_training_or_source_sweep"
    dec={"status":status,"next_branch":next_branch,"projection_structural_injectivity":structural,"projected_delta_block_map_absolute_error_diagnostic_only":True,
      "raw_support_go":rs["go"],"raw_reserve_go":rr["go"],"projected_support_go":ps["go"],"projected_reserve_go":pr["go"],
      "raw_support_positive_cells":rs["positive_cells"],"raw_support_top1_material_cells":rs["top1_material_cells"],"raw_support_roles":rs["roles"],
      "raw_reserve_positive_cells":rr["positive_cells"],"raw_reserve_top1_material_cells":rr["top1_material_cells"],"raw_reserve_roles":rr["roles"],
      "projected_support_positive_cells":ps["positive_cells"],"projected_reserve_positive_cells":pr["positive_cells"],
      "diagnostic_auc_deltas_projected_minus_raw":{"support":_deltas(docs,"support"),"reserve":_deltas(docs,"reserve")},
      "source_training_authorized":False,"broad_encoder_training_authorized":False,"boundary_transport_authorized":False,"dataset_reconstruction_authorized":False,"regime_conditioned_policy_authorized":False}
    out={"schema":"ocrap-v48.108-rpap-comparison-v1","engineering_version":ENGINEERING_VERSION,"valid":not errors,"attribution_ready":not errors,"errors":errors,"experiment_type":"audit_only_raw_to_projected_candidate_action_pathway","preregistered_decision":dec,
      "planner_parameters_trained":0,"stage_i_parameters_trained":0,"root_decoder_parameters_trained":0,"source_parameters_trained":0,"relative_ranker_modified":False,"regime_conditioning":False,"boundary_transport":False,"teacher_metadata_input_to_model":False,"test_roots_read":False,
      "v48_107_pipeline_sha256":_sha(a.v107_pipeline),"v48_107_comparison_sha256":_sha(a.v107_comparison)}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");print(json.dumps({"valid":out["valid"],"status":status,"errors":errors}));return 0 if out["valid"] else 30
if __name__=="__main__":raise SystemExit(main())
