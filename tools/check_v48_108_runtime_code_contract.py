#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from ocrap.v48_108_raw_to_projected_action_pathway_audit import ENGINEERING_VERSION,contract_checks
ACTIVE=["scripts/run_v48_108_dcp_drfc_bcde_rifa_rpap_two_gpu.sh","src/ocrap/v48_108_raw_to_projected_action_pathway_audit.py","tools/run_v48_108_raw_to_projected_action_pathway_audit.py","tools/compare_v48_108_rpap.py","tools/check_v48_108_runtime_code_contract.py","tools/check_v48_108_pipeline_complete.py"]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--repo",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);a=ap.parse_args();repo=a.repo.resolve();errors=[];files={}
 for rel in ACTIVE:
  p=(repo/rel).resolve();ok=p.is_file() and str(p).startswith(str(repo));files[rel]={"exists":p.is_file(),"inside_repo":str(p).startswith(str(repo)),"path":str(p),"sha256":sha(p) if p.is_file() else None}
  if not ok:errors.append(f"runtime_file:{rel}")
 checks=contract_checks(192)
 for k,v in checks.items():
  if not v:errors.append(f"synthetic:{k}")
 out={"schema":"ocrap-v48.108-rpap-runtime-code-contract-v1","engineering_version":ENGINEERING_VERSION,"valid":not errors,"attribution_ready":not errors,"errors":errors,"runtime_files":files,
      "scientific_contract":{"audit_only":True,"raw_to_projected_candidate_action_pathway":True,"raw_pathway_fixed_by_structured_encoder_layout":True,"raw_pathway_groups":["ego","prefix_param","macro_scalar","prefix_state","control"],"projected_counterpart_fixed_token_indices":[1,2,3,4,5],"projection_rank_and_pseudoinverse_reconstruction_audited":True,"same_v48_102_target_specific_probe_recipe":True,"within_group_action_permutation_control":True,"posthoc_token_selection":False,"stage_i_parameters_trained":0,"root_decoder_parameters_trained":0,"source_parameters_trained":0,"planner_parameters_trained":0,"boundary_transport":False,"broad_encoder_training":False,"regime_conditioning":False,"teacher_metadata_input_to_model":False,"threshold_sweep":False,"capacity_sweep":False},
      "synthetic_checks":checks,"test_roots_read":False}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");print(json.dumps({"valid":out["valid"],"errors":errors}));return 0 if out["valid"] else 30
if __name__=="__main__":raise SystemExit(main())
