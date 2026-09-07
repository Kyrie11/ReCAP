#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from ocrap.v48_109_raw_candidate_scene_orientation import ENGINEERING_VERSION,contract_checks
ACTIVE=[
"scripts/run_v48_109_dcp_drfc_bcde_rifa_rcso_two_gpu.sh",
"src/ocrap/v48_109_raw_candidate_scene_orientation.py",
"tools/run_v48_109_raw_candidate_scene_orientation_audit.py",
"tools/compare_v48_109_rcso.py",
"tools/check_v48_109_runtime_code_contract.py",
"tools/check_v48_109_pipeline_complete.py",
]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--repo",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);a=ap.parse_args();repo=a.repo.resolve();errors=[];files={}
 for rel in ACTIVE:
  p=(repo/rel).resolve();ok=p.is_file() and str(p).startswith(str(repo));files[rel]={"exists":p.is_file(),"inside_repo":str(p).startswith(str(repo)),"path":str(p),"sha256":sha(p) if p.is_file() else None}
  if not ok:errors.append(f"runtime_file:{rel}")
 checks=contract_checks()
 for k,v in checks.items():
  if not v:errors.append(f"synthetic:{k}")
 out={
   "schema":"ocrap-v48.109-rcso-runtime-code-contract-v1","engineering_version":ENGINEERING_VERSION,"valid":not errors,"attribution_ready":not errors,"errors":errors,"runtime_files":files,
   "scientific_contract":{
     "audit_only":True,"raw_candidate_scene_relational_orientation":True,"score_family":"u_T_w_plus_u_T_W_c",
     "raw_candidate_dim":156,"raw_scene_context_dim":240,"relational_dim":37596,
     "scene_context_definition":"agent_summary_plus_agent_mean_std_max_min_plus_bev_route_map_dyn",
     "agent_set_summary_permutation_invariant":True,"scene_context_candidate_invariant":True,
     "support_response_coordinate":"candidate_minus_nominal_raw_delta",
     "reserve_response_coordinate":"delta_times_one_plus_tanh_nominal_raw_state",
     "convex_closed_form_ridge":True,"strictly_convex_unique_solution":True,"iterative_optimizer_used":False,
     "ridge_lambda_rule":"1_over_axis_train_rows","nominal_zero_score_by_construction":True,
     "same_convex_solver_for_base_and_relational_control":True,"within_group_action_permutation_control":True,
     "posthoc_feature_selection":False,"stage_i_parameters_trained":0,"root_decoder_parameters_trained":0,"source_parameters_trained":0,"planner_parameters_trained":0,
     "boundary_transport":False,"broad_encoder_training":False,"regime_conditioning":False,"teacher_metadata_input_to_model":False,
     "threshold_sweep":False,"capacity_sweep":False,"lr_or_epoch_sweep":False,
   },
   "synthetic_checks":checks,"test_roots_read":False,
 }
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");print(json.dumps({"valid":out["valid"],"errors":errors}));return 0 if out["valid"] else 30
if __name__=="__main__":raise SystemExit(main())
