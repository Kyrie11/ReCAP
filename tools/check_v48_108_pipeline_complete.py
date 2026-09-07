#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser()
 for k in ("runtime","balanced","precision","balanced_state","precision_state","comparison","v48_107_pipeline","v48_107_comparison"):ap.add_argument("--"+k.replace("_","-"),dest=k,type=Path,required=True)
 ap.add_argument("--output",type=Path,required=True);a=ap.parse_args();errors=[]
 docs={k:json.loads(getattr(a,k).read_text()) for k in ("runtime","balanced","precision","comparison","v48_107_pipeline","v48_107_comparison")}
 if not(docs["runtime"].get("valid") and docs["runtime"].get("attribution_ready")):errors.append("runtime")
 for v in ("balanced","precision"):
  d=docs[v]
  if not(d.get("valid") and d.get("engineering_version")=="v48.108.0-OC-RPAP" and d.get("variant")==v and d.get("audit_only")):errors.append(v)
 if not(docs["comparison"].get("valid") and docs["comparison"].get("attribution_ready")):errors.append("comparison")
 if not(docs["v48_107_pipeline"].get("valid") and docs["v48_107_pipeline"].get("preregistered_status")=="FIRST_BLOCK_NOMINAL_INVARIANT_ACTION_ORIENTATION_STOP"):errors.append("v107_pipeline")
 d107=docs["v48_107_comparison"].get("preregistered_decision") or {}
 if not(d107.get("status")=="FIRST_BLOCK_NOMINAL_INVARIANT_ACTION_ORIENTATION_STOP" and d107.get("next_branch")=="close_first_block_orientation_then_preregister_raw_to_projected_action_pathway_audit_no_broad_encoder_or_source_sweep"):errors.append("v107_branch")
 artifacts={}
 for k in ("balanced","precision","balanced_state","precision_state","comparison","runtime"):
  p=getattr(a,k);artifacts[k]={"path":str(p.resolve()),"sha256":sha(p)}
 status=(docs["comparison"].get("preregistered_decision") or {}).get("status")
 out={"schema":"ocrap-v48.108-rpap-pipeline-complete-v1","engineering_version":"v48.108.0-OC-RPAP","valid":not errors,"attribution_ready":not errors,"errors":errors,"experiment_type":"audit_only_raw_to_projected_candidate_action_pathway","artifacts":artifacts,"preregistered_status":status,
      "planner_parameters_trained":0,"stage_i_parameters_trained":0,"root_decoder_parameters_trained":0,"source_parameters_trained":0,"boundary_transport":False,"dataset_reconstruction":False,"regime_conditioning":False,"teacher_metadata_input_to_model":False,"test_roots_read":False,
      "v48_107_pipeline_sha256":sha(a.v48_107_pipeline),"v48_107_comparison_sha256":sha(a.v48_107_comparison)}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");print(json.dumps({"valid":out["valid"],"status":status,"errors":errors}));return 0 if out["valid"] else 30
if __name__=="__main__":raise SystemExit(main())
