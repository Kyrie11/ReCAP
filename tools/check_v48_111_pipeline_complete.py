#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
AUTHORITATIVE_V110_COMPARISON_SHA256='5bb9bbac2b5a88cb9419308804afdfce22643cd986df284224e1c9f3617e1c9d'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser()
 for k in ('runtime','balanced','precision','balanced_state','precision_state','comparison','v48_110_pipeline','v48_110_comparison'):ap.add_argument('--'+k.replace('_','-'),dest=k,type=Path,required=True)
 ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();errors=[]
 docs={k:json.loads(getattr(a,k).read_text()) for k in ('runtime','balanced','precision','comparison','v48_110_pipeline','v48_110_comparison')}
 if not(docs['runtime'].get('valid') and docs['runtime'].get('attribution_ready')):errors.append('runtime')
 for v in ('balanced','precision'):
  d=docs[v]
  if not(d.get('valid') and d.get('engineering_version')=='v48.111.0-OC-CNRO' and d.get('variant')==v and d.get('audit_only') and d.get('convex_closed_form_ridge') and d.get('capacity_matched_nearest_vs_active')):errors.append(v)
 if not(docs['comparison'].get('valid') and docs['comparison'].get('attribution_ready')):errors.append('comparison')
 if sha(a.v48_110_comparison)!=AUTHORITATIVE_V110_COMPARISON_SHA256:errors.append('v110_comparison_sha')
 if not(docs['v48_110_pipeline'].get('valid') and docs['v48_110_pipeline'].get('attribution_ready') and docs['v48_110_pipeline'].get('engineering_version')=='v48.110.0-OC-CATO' and docs['v48_110_pipeline'].get('preregistered_status')=='CANDIDATE_AGENT_TOPOLOGY_STOP'):errors.append('v110_pipeline')
 d110=docs['v48_110_comparison'].get('preregistered_decision') or {}
 if not(d110.get('status')=='CANDIDATE_AGENT_TOPOLOGY_STOP' and d110.get('next_branch')=='close_coordinatewise_agent_topology_then_preregister_constraint_native_candidate_agent_geometry_audit_no_training_or_source_sweep'):errors.append('v110_branch')
 artifacts={}
 for k in ('balanced','precision','balanced_state','precision_state','comparison','runtime'):
  p=getattr(a,k);artifacts[k]={'path':str(p.resolve()),'sha256':sha(p)}
 status=(docs['comparison'].get('preregistered_decision') or {}).get('status')
 out={'schema':'ocrap-v48.111-cnro-pipeline-complete-v1','engineering_version':'v48.111.0-OC-CNRO','valid':not errors,'attribution_ready':not errors,'errors':errors,'experiment_type':'audit_only_constraint_native_candidate_agent_recovery_orientation','artifacts':artifacts,'preregistered_status':status,
      'planner_parameters_trained':0,'stage_i_parameters_trained':0,'root_decoder_parameters_trained':0,'source_parameters_trained':0,'boundary_transport':False,'dataset_reconstruction':False,'regime_conditioning':False,'teacher_metadata_input_to_model':False,'test_roots_read':False,
      'v48_110_pipeline_sha256':sha(a.v48_110_pipeline),'v48_110_comparison_sha256':sha(a.v48_110_comparison),'authoritative_v48_110_comparison_sha256':AUTHORITATIVE_V110_COMPARISON_SHA256}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps({'valid':out['valid'],'status':status,'errors':errors}));return 0 if out['valid'] else 30
if __name__=='__main__':raise SystemExit(main())
