#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, json, time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ocrap.models.data import OCRAPSampleDataset
from ocrap.models.inference import load_model_bundle
from ocrap.models.encoders import StructuredTokenEncoder
from ocrap.v48_96_support_reserve_root_observability import feature_only_dataset_cfg
from ocrap.v48_108_raw_to_projected_action_pathway_audit import raw_candidate_pathway, action_features
from ocrap.v48_109_raw_candidate_scene_orientation import fit_closed_form_ridge, ridge_scores
from ocrap.v48_110_candidate_agent_topology_orientation import (
    ALGORITHM_NAME, ENGINEERING_VERSION, RAW_CANDIDATE_DIM, AGENT_DIM, NEAREST_DIM, TOPOLOGY_DIM,
    raw_agent_set, candidate_agent_clearance_topology, fit_active_topology_scaler,
    base_features, nearest_features, topology_features,
)
from tools.run_v48_102_stage_i_action_information_transport_audit import (
    action_subset, build_v93_map, label_groups, split_role, auc,
)
from tools.run_v48_97_executable_recovery_state import ROLES, sha256

V109_ENGINEERING_VERSION = "v48.109.0-OC-RCSO"
AUTHORITATIVE_V109_COMPARISON_SHA256 = "4aa8d8846a39de6fa3797464ce3e9587c148872a7a46d5a8f118fcba9c983627"


def _cache_key(checkpoint: Path, index_path: Path, role_filter: str | None, v93_path: Path | None) -> str:
    payload = {"version":ENGINEERING_VERSION,"checkpoint":sha256(checkpoint),"index":sha256(index_path),"role":role_filter,"v93":sha256(v93_path) if v93_path and v93_path.is_file() else None}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

def _stack(items,key):return torch.stack([it[key] for it in items])

def extract_records(*,checkpoint:Path,index_path:Path,role_filter:str|None,v93_path:Path|None,cache_dir:Path,device:str):
    key=_cache_key(checkpoint,index_path,role_filter,v93_path);cache_dir.mkdir(parents=True,exist_ok=True);cp=cache_dir/f"{key}.pt"
    if cp.is_file():
        obj=torch.load(cp,map_location='cpu',weights_only=False)
        if obj.get('cache_key')==key:return obj['records'],obj['event']
    v93=build_v93_map(v93_path);groups=label_groups(index_path,role_filter=role_filter,v93_map=v93)
    needed=[]
    for g in groups:needed.append(Path(g['nominal_path']));needed.extend(Path(c['path']) for c in g['candidates'])
    seen=set();paths=[]
    for p in needed:
        q=str(p.resolve())
        if q not in seen:seen.add(q);paths.append(p)
    bundle=load_model_bundle(checkpoint,{"training":{"device":device}})
    if bundle is None:raise RuntimeError(f"cannot load checkpoint {checkpoint}")
    model=bundle.model.eval();[p.requires_grad_(False) for p in model.parameters()]
    if not isinstance(model.encoder,StructuredTokenEncoder):raise RuntimeError('V48.110 requires StructuredTokenEncoder')
    enc=model.encoder.eval();dev=bundle.device
    if len(enc.encoder.layers)!=2:raise RuntimeError('V48.110 requires historical two-layer Stage-I')
    sample_rate=float(bundle.cfg.get('sample_rate_hz',10.0) or 10.0)
    cfg,feature_event=feature_only_dataset_cfg(bundle.cfg,cache_dir=str(cache_dir/'tensor'),workers=8);ds=OCRAPSampleDataset(paths,cfg)
    if ds.absolute_truth_contract_event.get('enabled') or ds.action_response_truth_event.get('enabled'):raise RuntimeError('V48.110 feature-only dataset unexpectedly attached truth sidecars')
    if [str(p.resolve()) for p in paths] != [str(p.resolve()) for p in ds.paths]:raise RuntimeError('V48.110 dataset path order differs from index')
    idx={str(p.resolve()):i for i,p in enumerate(ds.paths)}
    records=[];agent_delta_max=0.0;mask_delta_max=0
    for g in groups:
        ordered=[g['nominal_path']]+[c['path'] for c in g['candidates']]
        if any(str(Path(p).resolve()) not in idx for p in ordered):continue
        items=[ds[idx[str(Path(p).resolve())]] for p in ordered];x=_stack(items,'x').to(dev)
        with torch.no_grad():
            raw=raw_candidate_pathway(x,enc.layout);agents,mask=raw_agent_set(x,enc.layout)
            agent_delta_max=max(agent_delta_max,float((agents[1:]-agents[0:1]).abs().max().item()));mask_delta_max=max(mask_delta_max,int(torch.count_nonzero(mask[1:]!=mask[0:1]).item()))
            state,delta,reserve_context=action_features(raw)
            cand=raw[1:].cpu().numpy();an=agents[0:1].expand(delta.shape[0],-1,-1).cpu().numpy();mn=mask[0:1].expand(delta.shape[0],-1).cpu().numpy()
            stn=state.cpu().numpy();dn=delta.cpu().numpy();qn=reserve_context.cpu().numpy()
        a1,a2,nr,topo=candidate_agent_clearance_topology(cand,an,mn,sample_rate)
        for j,c in enumerate(g['candidates']):
            records.append({
                'group':tuple(g['key']),'candidate':int(c['candidate']),'group_mode':g['group_mode'],'safe_positive':bool(c['safe_positive']),'teacher_harmful':bool(c['teacher_harmful']),'mediation_mode':c['mediation_mode'],
                'raw_state':stn[j],'candidate_raw':cand[j],'support_u':dn[j],'reserve_u':qn[j],'agents':an[j],'agent_mask':mn[j],
                'active1':int(a1[j]),'active2':int(a2[j]),'nearest':int(nr[j]),'topology_scalar':topo[j],
            })
    if agent_delta_max>1e-6 or mask_delta_max!=0:raise RuntimeError(f'V48.110 agent set changed across candidate actions values={agent_delta_max} mask={mask_delta_max}')
    event={
        'records':len(records),'groups':len(groups),'raw_candidate_dim':RAW_CANDIDATE_DIM,'agent_dim':AGENT_DIM,'nearest_dim':NEAREST_DIM,'topology_dim':TOPOLOGY_DIM,
        'agent_set_candidate_delta_max_abs':agent_delta_max,'agent_mask_candidate_delta_count':mask_delta_max,'sample_rate_hz':sample_rate,
        'agent_set_definition':'historical_raw_agent_tokens_current_observation','active_selector':'minimum_cv_circle_clearance_over_first_8_complete_prefix_states','topology_features':'u_plus_active1_active2_agent_cross_plus_clearance_gap_cross',
        'feature_only_dataset_contract':feature_event,'tensor_cache_event':ds.tensor_cache_event,'encoder_layer_count':2,
    }
    torch.save({'cache_key':key,'records':records,'event':event},cp);return records,event

def _perm_indices(records):
    groups=defaultdict(list)
    for i,r in enumerate(records):groups[tuple(r['group'])].append(i)
    idx=np.arange(len(records))
    for ids in groups.values():
        ids=sorted(ids,key=lambda i:int(records[i]['candidate']));idx[ids]=np.roll(np.asarray(ids,dtype=np.int64),1)
    return idx

def _arrays(records,key):
    U=np.stack([r[key] for r in records]).astype(np.float64);A=np.stack([r['agents'] for r in records]).astype(np.float64);M=np.stack([r['agent_mask'] for r in records]).astype(bool)
    a1=np.asarray([r['active1'] for r in records],dtype=np.int64);a2=np.asarray([r['active2'] for r in records],dtype=np.int64);nr=np.asarray([r['nearest'] for r in records],dtype=np.int64);topo=np.stack([r['topology_scalar'] for r in records]).astype(np.float64);y=np.asarray([r['label'] for r in records],dtype=np.int64)
    return U,A,M,a1,a2,nr,topo,y

def _fit_axis(records,key):
    U,A,M,a1,a2,nr,topo,y=_arrays(records,key);sc=fit_active_topology_scaler(U,A,M,topo);pi=_perm_indices(records)
    Up=U[pi];a1p=a1[pi];a2p=a2[pi];nrp=nr[pi];topop=topo[pi]
    models={
      'base_true':fit_closed_form_ridge(base_features(U,sc),y),'base_shuffle':fit_closed_form_ridge(base_features(Up,sc),y),
      'nearest_true':fit_closed_form_ridge(nearest_features(U,A,M,nr,sc),y),'nearest_shuffle':fit_closed_form_ridge(nearest_features(Up,A,M,nrp,sc),y),
      'topology_true':fit_closed_form_ridge(topology_features(U,A,M,a1,a2,topo,sc),y),'topology_shuffle':fit_closed_form_ridge(topology_features(Up,A,M,a1p,a2p,topop,sc),y),
    }
    return {'scaler':sc,'models':models,'count':len(records)}

def _fit_family(records):
    su=action_subset(records,'drs_activation');re=action_subset(records,'deployability_gain');return {'support':_fit_axis(su,'support_u'),'reserve':_fit_axis(re,'reserve_u'),'counts':{'support':len(su),'reserve':len(re)}}

def _metric(records,sc):
    if not records:return {'rows':0,'positive_rows':0,'negative_rows':0,'auc':None,'top1':None,'powered_groups':0}
    y=np.asarray([r['label'] for r in records],dtype=np.int64);sc=np.asarray(sc,dtype=np.float64);groups=defaultdict(list)
    for i,r in enumerate(records):groups[tuple(r['group'])].append(i)
    powered=[ids for ids in groups.values() if any(y[i]==1 for i in ids) and any(y[i]==0 for i in ids)]
    top1=float(np.mean([y[max(ids,key=lambda i:float(sc[i]))]==1 for ids in powered])) if powered else None
    return {'rows':len(records),'positive_rows':int(y.sum()),'negative_rows':int(len(y)-y.sum()),'auc':auc(y,sc),'top1':top1,'powered_groups':len(powered)}

def _eval_axis(records,key,fit):
    if records:U,A,M,a1,a2,nr,topo,_=_arrays(records,key);pi=_perm_indices(records);Up=U[pi];a1p=a1[pi];a2p=a2[pi];nrp=nr[pi];topop=topo[pi]
    else:
        U=np.empty((0,156));A=np.empty((0,32,10));M=np.empty((0,32),bool);a1=a2=nr=np.empty((0,),int);topo=np.empty((0,2));Up=U;a1p=a2p=nrp=a1;topop=topo
    sc=fit['scaler'];m=fit['models'];feats={
      'base':(base_features(U,sc),base_features(Up,sc)),
      'nearest':(nearest_features(U,A,M,nr,sc),nearest_features(Up,A,M,nrp,sc)),
      'topology':(topology_features(U,A,M,a1,a2,topo,sc),topology_features(Up,A,M,a1p,a2p,topop,sc)),
    };out={}
    for space,(f,fp) in feats.items():
        t=_metric(records,ridge_scores(m[f'{space}_true'],f));s=_metric(records,ridge_scores(m[f'{space}_shuffle'],fp));t['auc_vs_shuffled']=None if t['auc'] is None or s['auc'] is None else float(t['auc']-s['auc']);t['top1_vs_shuffled']=None if t['top1'] is None or s['top1'] is None else float(t['top1']-s['top1']);out[space]=(t,s)
    return out

def _eval_family(dev_records,cert_records,family):
    cells={k:{} for k in ('base','nearest','topology')}
    for role in ROLES:
        src=dev_records if role.startswith('dev_') else cert_records;rr=split_role(src,role);su=action_subset(rr,'drs_activation');re=action_subset(rr,'deployability_gain');sm=_eval_axis(su,'support_u',family['support']);rm=_eval_axis(re,'reserve_u',family['reserve'])
        for space in cells:cells[space][role]={'support_true':sm[space][0],'support_shuffled':sm[space][1],'reserve_true':rm[space][0],'reserve_shuffled':rm[space][1]}
    return cells

def _pack_axis(fit):
    sc=fit['scaler'];return {'scaler':{'u_scale':sc.u_scale,'a_mu':sc.a_mu,'a_sd':sc.a_sd,'t_mu':sc.t_mu,'t_sd':sc.t_sd},'models':{k:{'coef':v.coef,'ridge_lambda':v.ridge_lambda,'objective':v.objective,'normal_equation_residual':v.normal_equation_residual} for k,v in fit['models'].items()},'count':fit['count']}

def main()->int:
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--train-index',type=Path,required=True);ap.add_argument('--dev-index',type=Path,required=True);ap.add_argument('--certificate-index',type=Path,required=True);ap.add_argument('--v93-audit',type=Path,required=True);ap.add_argument('--cache-dir',type=Path,required=True);ap.add_argument('--device',default='cuda');ap.add_argument('--variant',required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--state-output',type=Path,required=True);a=ap.parse_args();t0=time.perf_counter()
    tr,etr=extract_records(checkpoint=a.checkpoint,index_path=a.train_index,role_filter=None,v93_path=None,cache_dir=a.cache_dir/'train',device=a.device);dv=[];ce=[];events={'train':etr}
    for role in ('dev_near','dev_contact'):
        r,e=extract_records(checkpoint=a.checkpoint,index_path=a.dev_index,role_filter=role,v93_path=a.v93_audit,cache_dir=a.cache_dir/role,device=a.device);dv+=r;events[role]=e
    for role in ('certificate_near','certificate_contact'):
        r,e=extract_records(checkpoint=a.checkpoint,index_path=a.certificate_index,role_filter=role,v93_path=a.v93_audit,cache_dir=a.cache_dir/role,device=a.device);ce+=r;events[role]=e
    if not tr or not dv or not ce:raise RuntimeError('V48.110 empty audit records')
    fam=_fit_family(tr);cells=_eval_family(dv,ce,fam);max_resid=max(v.normal_equation_residual for axis in ('support','reserve') for v in fam[axis]['models'].values())
    result={
      'schema':'ocrap-v48.110-candidate-agent-topology-orientation-audit-v1','engineering_version':ENGINEERING_VERSION,'algorithm_name':ALGORITHM_NAME,'valid':True,'variant':a.variant,'audit_only':True,'checkpoint':str(a.checkpoint.resolve()),'checkpoint_sha256':sha256(a.checkpoint),
      'base_cells':cells['base'],'nearest_cells':cells['nearest'],'topology_cells':cells['topology'],'events':events,'train_counts':fam['counts'],'convex_closed_form_ridge':True,'strictly_convex_unique_solution':True,'iterative_optimizer_used':False,'ridge_lambda_rule':'1_over_axis_train_rows','max_normal_equation_residual':max_resid,
      'score_family':'linear_on_fixed_candidate_agent_topology_features','nominal_zero_score_by_construction':True,'agent_set_candidate_invariant':True,'agent_topology_permutation_invariant':True,'active_selector':'minimum_cv_circle_clearance_over_first_8_complete_prefix_states','topology_features':'u_plus_active1_active2_agent_cross_plus_clearance_gap_cross',
      'raw_candidate_dim':RAW_CANDIDATE_DIM,'agent_dim':AGENT_DIM,'nearest_dim':NEAREST_DIM,'topology_dim':TOPOLOGY_DIM,'planner_parameters_trained':0,'stage_i_parameters_trained':0,'root_decoder_parameters_trained':0,'source_parameters_trained':0,'relative_ranker_modified':False,'regime_conditioning':False,'boundary_transport':False,'teacher_metadata_input_to_model':False,'test_roots_read':False,'posthoc_feature_selection':False,'elapsed_seconds':float(time.perf_counter()-t0),
    }
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');torch.save({'schema':'ocrap-v48.110-candidate-agent-topology-state-v1','engineering_version':ENGINEERING_VERSION,'algorithm_name':ALGORITHM_NAME,'variant':a.variant,'support':_pack_axis(fam['support']),'reserve':_pack_axis(fam['reserve']),'convex_closed_form_ridge':True,'strictly_convex_unique_solution':True,'iterative_optimizer_used':False,'checkpoint_sha256':sha256(a.checkpoint)},a.state_output)
    print(json.dumps({'valid':True,'variant':a.variant,'max_normal_equation_residual':max_resid,'elapsed_seconds':result['elapsed_seconds']}));return 0
if __name__=='__main__':raise SystemExit(main())
