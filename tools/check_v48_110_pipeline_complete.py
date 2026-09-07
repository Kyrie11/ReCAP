#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
AUTHORITATIVE_V109_COMPARISON_SHA256="4aa8d8846a39de6fa3797464ce3e9587c148872a7a46d5a8f118fcba9c983627"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser()
 for k in ("runtime","balanced","precision","balanced_state","precision_state","comparison","v48_109_pipeline","v48_109_comparison"):ap.add_argument("--"+k.replace("_","-"),dest=k,type=Path,required=True)
 ap.add_argument("--output",type=Path,required=True);a=ap.parse_args();errors=[]
 docs={k:json.loads(getattr(a,k).read_text()) for k in ("runtime","balanced","precision","comparison","v48_109_pipeline","v48_109_comparison")}
 if not(docs["runtime"].get("valid") and docs["runtime"].get("attribution_ready")):errors.append("runtime")
 for v in ("balanced","precision"):
  d=docs[v]
  if not(d.get("valid") and d.get("engineering_version")=="v48.110.0-OC-CATO" and d.get("variant")==v and d.get("audit_only") and d.get("convex_closed_form_ridge")):errors.append(v)
 if not(docs["comparison"].get("valid") and docs["comparison"].get("attribution_ready")):errors.append("comparison")
 if sha(a.v48_109_comparison)!=AUTHORITATIVE_V109_COMPARISON_SHA256:errors.append("v109_comparison_sha")
 if not(docs["v48_109_pipeline"].get("valid") and docs["v48_109_pipeline"].get("preregistered_status")=="RAW_SCENE_RELATIONAL_STOP" and docs["v48_109_pipeline"].get("engineering_version")=="v48.109.0-OC-RCSO"):errors.append("v109_pipeline")
 d109=docs["v48_109_comparison"].get("preregistered_decision") or {}
 if not(d109.get("status")=="RAW_SCENE_RELATIONAL_STOP" and d109.get("next_branch")=="close_bilinear_relational_orientation_then_preregister_candidate_to_agent_constraint_topology_audit_no_training_or_source_sweep"):errors.append("v109_branch")
 artifacts={}
 for k in ("balanced","precision","balanced_state","precision_state","comparison","runtime"):
  p=getattr(a,k);artifacts[k]={"path":str(p.resolve()),"sha256":sha(p)}
 status=(docs["comparison"].get("preregistered_decision") or {}).get("status")
 out={
  "schema":"ocrap-v48.110-cato-pipeline-complete-v1","engineering_version":"v48.110.0-OC-CATO","valid":not errors,"attribution_ready":not errors,"errors":errors,
  "experiment_type":"audit_only_candidate_agent_active_constraint_topology_orientation","artifacts":artifacts,"preregistered_status":status,
  "planner_parameters_trained":0,"stage_i_parameters_trained":0,"root_decoder_parameters_trained":0,"source_parameters_trained":0,
  "boundary_transport":False,"dataset_reconstruction":False,"regime_conditioning":False,"teacher_metadata_input_to_model":False,"test_roots_read":False,
  "v48_109_pipeline_sha256":sha(a.v48_109_pipeline),"v48_109_comparison_sha256":sha(a.v48_109_comparison),"authoritative_v48_109_comparison_sha256":AUTHORITATIVE_V109_COMPARISON_SHA256,
 }
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");print(json.dumps({"valid":out["valid"],"status":status,"errors":errors}));return 0 if out["valid"] else 30
if __name__=="__main__":raise SystemExit(main())
