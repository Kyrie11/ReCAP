#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any

ENGINEERING_VERSION="v48.111.0-OC-CNRO"
V110_ENGINEERING_VERSION="v48.110.0-OC-CATO"
AUTHORITATIVE_V110_COMPARISON_SHA256="5bb9bbac2b5a88cb9419308804afdfce22643cd986df284224e1c9f3617e1c9d"
ROLES=("dev_near","dev_contact","certificate_near","certificate_contact")

def _sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def _ok(v:Any,t:float)->bool:return v is not None and float(v)>=t
def _cross(rs:set[str],n:int)->bool:return len(rs)>=n and any('near' in x for x in rs) and any('contact' in x for x in rs)

def _action_gate(docs,space,metric):
    positive=[];top=[];roles=set();top_roles=set()
    for v,d in docs.items():
        for role in ROLES:
            m=d[f'{space}_cells'][role][f'{metric}_true']
            if _ok(m.get('auc'),.65) and _ok(m.get('auc_vs_shuffled'),.05):positive.append([v,role]);roles.add(role)
            if _ok(m.get('top1_vs_shuffled'),.10):top.append([v,role]);top_roles.add(role)
    go=len(positive)>=6 and _cross(roles,3) and len(top)>=4 and _cross(top_roles,2)
    return {'go':bool(go),'local_order':bool(len(top)>=4 and _cross(top_roles,2)),'positive_cells':positive,'top1_material_cells':top,'roles':sorted(roles),'top1_roles':sorted(top_roles)}

def _switch_gate(docs,metric):
    rows=[];positive=[];material=[];roles=set()
    for v in ('balanced','precision'):
        for role in ROLES:
            n=docs[v]['nearest_cells'][role][f'{metric}_true'].get('auc');a=docs[v]['active_cells'][role][f'{metric}_true'].get('auc')
            d=None if n is None or a is None else float(a)-float(n)
            row={'variant':v,'role':role,'nearest_auc':n,'active_auc':a,'active_minus_nearest':d};rows.append(row)
            if d is not None and d>0:positive.append([v,role]);roles.add(role)
            if d is not None and d>=.01:material.append([v,role])
    # 6/8 == 3/4 roles when balanced/precision raw-audit metrics are duplicate.
    go=len(positive)>=6 and _cross(roles,3) and len(material)>=4
    return {'go':bool(go),'positive_cells':positive,'material_cells':material,'roles':sorted(roles),'rows':rows}

def _result_errors(o,v):
    e=[];checks=[
      (o.get('valid') is True,'valid'),(o.get('engineering_version')==ENGINEERING_VERSION,'version'),(o.get('variant')==v,'variant'),(o.get('audit_only') is True,'audit'),
      (o.get('convex_closed_form_ridge') is True,'convex'),(o.get('strictly_convex_unique_solution') is True,'unique'),(o.get('iterative_optimizer_used') is False,'no_iterative_optimizer'),
      (o.get('score_family')=='linear_on_fixed_constraint_native_candidate_agent_response_features','score_family'),(o.get('nominal_zero_score_by_construction') is True,'nominal_zero'),
      (o.get('agent_set_candidate_invariant') is True,'agent_invariant'),(o.get('agent_pair_selector_permutation_invariant') is True,'agent_perm'),(o.get('capacity_matched_nearest_vs_active') is True,'capacity_match'),
      (int(o.get('matched_family_dimension',-1))==188,'matched_dim'),(int(o.get('geometry_dim',-1))==32,'geometry_dim'),(float(o.get('max_normal_equation_residual',1.0))<=1e-7,'normal_residual'),
      (int(o.get('stage_i_parameters_trained',-1))==0,'stage_i'),(int(o.get('root_decoder_parameters_trained',-1))==0,'root'),(int(o.get('source_parameters_trained',-1))==0,'source'),(int(o.get('planner_parameters_trained',-1))==0,'planner'),
      (o.get('regime_conditioning') is False,'regime'),(o.get('boundary_transport') is False,'boundary'),(o.get('teacher_metadata_input_to_model') is False,'teacher'),(o.get('test_roots_read') is False,'test_roots'),
    ]
    for ok,n in checks:
        if not ok:e.append(f'{v}:{n}')
    for space in ('base','nearest','active'):
        for role in ROLES:
            if role not in o.get(f'{space}_cells',{}):e.append(f'{v}:{space}:{role}')
    for ev,evo in o.get('events',{}).items():
        if float(evo.get('agent_set_candidate_delta_max_abs',1.0))>1e-6:e.append(f'{v}:{ev}:agent_delta')
        if int(evo.get('agent_mask_candidate_delta_count',1))!=0:e.append(f'{v}:{ev}:agent_mask')
    return e

def _v110_base_auc(c110,variant,role,metric):
    # V48.110 comparison records base values inside nearest-minus-base diagnostic rows.
    rows=((c110.get('preregistered_decision') or {}).get('diagnostic_auc_deltas_nearest_minus_base') or {}).get(metric,[])
    for r in rows:
        if r.get('variant')==variant and r.get('role')==role:return r.get('base_auc')
    return None

def _variant_identity(docs):
    diffs=[]
    for space in ('base','nearest','active'):
        for role in ROLES:
            for metric in ('support','reserve'):
                for suffix in ('auc','auc_vs_shuffled','top1','top1_vs_shuffled'):
                    a=docs['balanced'][f'{space}_cells'][role][f'{metric}_true'].get(suffix);b=docs['precision'][f'{space}_cells'][role][f'{metric}_true'].get(suffix)
                    if a is None and b is None:continue
                    if a is None or b is None or abs(float(a)-float(b))>1e-12:diffs.append([space,role,metric,suffix,a,b])
    return {'exact':not diffs,'differences':diffs,'effective_unique_roles_if_exact':4 if not diffs else 8}

def _power(docs,space):
    out=[]
    for v in ('balanced','precision'):
        for role in ROLES:
            for metric in ('support','reserve'):
                m=docs[v][f'{space}_cells'][role][f'{metric}_true']
                out.append({'variant':v,'role':role,'axis':metric,'rows':m.get('rows'),'positive_rows':m.get('positive_rows'),'negative_rows':m.get('negative_rows'),'powered_groups':m.get('powered_groups'),'underpowered':bool((m.get('powered_groups') or 0)<3 or (m.get('positive_rows') or 0)<5 or (m.get('negative_rows') or 0)<5)})
    return out

def main()->int:
    ap=argparse.ArgumentParser()
    for k in ('balanced','precision','v110_pipeline','v110_comparison'):ap.add_argument('--'+k.replace('_','-'),dest=k,type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();errors=[]
    docs={v:json.loads(getattr(a,v).read_text()) for v in ('balanced','precision')}
    for v,o in docs.items():errors+=_result_errors(o,v)
    p110=json.loads(a.v110_pipeline.read_text());c110=json.loads(a.v110_comparison.read_text());d110=c110.get('preregistered_decision') or {}
    if _sha(a.v110_comparison)!=AUTHORITATIVE_V110_COMPARISON_SHA256:errors.append('v110_comparison_sha')
    if not(p110.get('valid') and p110.get('attribution_ready') and p110.get('engineering_version')==V110_ENGINEERING_VERSION and p110.get('preregistered_status')=='CANDIDATE_AGENT_TOPOLOGY_STOP'):errors.append('v110_pipeline')
    if not(c110.get('valid') and c110.get('attribution_ready') and d110.get('status')=='CANDIDATE_AGENT_TOPOLOGY_STOP' and d110.get('next_branch')=='close_coordinatewise_agent_topology_then_preregister_constraint_native_candidate_agent_geometry_audit_no_training_or_source_sweep'):errors.append('v110_branch')
    # Exact V48.110/V48.109 base identity: unchanged response coordinate + solver.
    for metric in ('support','reserve'):
        for v in ('balanced','precision'):
            for role in ROLES:
                got=docs[v]['base_cells'][role][f'{metric}_true'].get('auc');exp=_v110_base_auc(c110,v,role,metric)
                if got is None or exp is None or abs(float(got)-float(exp))>1e-12:errors.append(f'{v}:{role}:{metric}:v110_base_identity')
    ngs=_action_gate(docs,'nearest','support') if not errors else {'go':False,'local_order':False,'positive_cells':[],'top1_material_cells':[],'roles':[],'top1_roles':[]}
    ngr=_action_gate(docs,'nearest','reserve') if not errors else {'go':False,'local_order':False,'positive_cells':[],'top1_material_cells':[],'roles':[],'top1_roles':[]}
    ags=_action_gate(docs,'active','support') if not errors else {'go':False,'local_order':False,'positive_cells':[],'top1_material_cells':[],'roles':[],'top1_roles':[]}
    agr=_action_gate(docs,'active','reserve') if not errors else {'go':False,'local_order':False,'positive_cells':[],'top1_material_cells':[],'roles':[],'top1_roles':[]}
    sis=_switch_gate(docs,'support') if not errors else {'go':False,'rows':[],'positive_cells':[],'material_cells':[],'roles':[]}
    sir=_switch_gate(docs,'reserve') if not errors else {'go':False,'rows':[],'positive_cells':[],'material_cells':[],'roles':[]}
    if errors:
        status='V48_111_ENGINEERING_STOP';branch='fix_v48_111_engineering_and_rerun_same_constraint_native_audit'
    elif ags['go'] and agr['go'] and sis['go'] and sir['go']:
        status='CONSTRAINT_NATIVE_ACTIVE_GEOMETRY_BOTH_AXES_GO';branch='promote_constraint_native_candidate_conditioned_recovery_orientation_then_preregister_one_nominal_invariant_response_carrier_no_source_or_transformer_sweep'
    elif ags['go'] and sis['go'] and not (agr['go'] and sir['go']):
        status='CONSTRAINT_NATIVE_ACTIVE_GEOMETRY_SUPPORT_ONLY';branch='support_constraint_geometry_go_reserve_stop_then_audit_signed_debt_normal_response_only'
    elif agr['go'] and sir['go'] and not (ags['go'] and sis['go']):
        status='CONSTRAINT_NATIVE_ACTIVE_GEOMETRY_RESERVE_ONLY';branch='reserve_constraint_geometry_go_support_stop_then_audit_support_establishment_normal_response_only'
    elif ags['local_order'] and agr['local_order']:
        status='CONSTRAINT_NATIVE_ACTIVE_GEOMETRY_LOCAL_ORDER_ONLY';branch='same_constraint_native_features_then_one_convex_pairwise_audit_no_feature_or_source_change'
    else:
        status='CONSTRAINT_NATIVE_ACTIVE_GEOMETRY_STOP';branch='close_fixed_cv_circle_agent_geometry_then_preregister_heterogeneous_active_constraint_normal_cone_audit_no_training_or_source_sweep'
    ident=_variant_identity(docs)
    dec={'status':status,'next_branch':branch,'nearest_support_gate':ngs,'nearest_reserve_gate':ngr,'active_support_gate':ags,'active_reserve_gate':agr,'active_minus_nearest_support_gate':sis,'active_minus_nearest_reserve_gate':sir,
         'capacity_matched_nearest_vs_active':True,'matched_dimension':188,'geometry_dimension':32,'balanced_precision_metric_identity':ident,'power_diagnostics':_power(docs,'active'),
         'scientific_note':'same strict convex solver removes optimizer ambiguity; 188-vs-188 active/nearest contrast additionally removes the V48.110 family-width confound',
         'source_training_authorized':False,'broad_encoder_training_authorized':False,'boundary_transport_authorized':False,'dataset_reconstruction_authorized':False,'regime_conditioned_policy_authorized':False}
    out={'schema':'ocrap-v48.111-cnro-comparison-v1','engineering_version':ENGINEERING_VERSION,'valid':not errors,'attribution_ready':not errors,'errors':errors,'experiment_type':'audit_only_constraint_native_candidate_agent_recovery_orientation','preregistered_decision':dec,
         'planner_parameters_trained':0,'stage_i_parameters_trained':0,'root_decoder_parameters_trained':0,'source_parameters_trained':0,'relative_ranker_modified':False,'regime_conditioning':False,'boundary_transport':False,'teacher_metadata_input_to_model':False,'test_roots_read':False,
         'v48_110_pipeline_sha256':_sha(a.v110_pipeline),'v48_110_comparison_sha256':_sha(a.v110_comparison),'authoritative_v48_110_comparison_sha256':AUTHORITATIVE_V110_COMPARISON_SHA256}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps({'valid':out['valid'],'status':status,'errors':errors}));return 0 if out['valid'] else 30
if __name__=='__main__':raise SystemExit(main())
