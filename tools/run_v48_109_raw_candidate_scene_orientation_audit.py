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
from ocrap.v48_109_raw_candidate_scene_orientation import (
    ALGORITHM_NAME, ENGINEERING_VERSION, RAW_CANDIDATE_DIM, RAW_SCENE_CONTEXT_DIM, RELATIONAL_DIM,
    raw_scene_context_summary, fit_relational_scaler, base_features, relational_features,
    fit_closed_form_ridge, ridge_scores,
)
from tools.run_v48_102_stage_i_action_information_transport_audit import (
    action_subset, build_v93_map, label_groups, permute_within_group, split_role, auc,
)
from tools.run_v48_97_executable_recovery_state import ROLES, sha256

V108_ENGINEERING_VERSION = "v48.108.1-OC-RPAP"


def _cache_key(checkpoint: Path, index_path: Path, role_filter: str | None, v93_path: Path | None) -> str:
    payload = {
        "version": ENGINEERING_VERSION,
        "checkpoint": sha256(checkpoint),
        "index": sha256(index_path),
        "role": role_filter,
        "v93": sha256(v93_path) if v93_path and v93_path.is_file() else None,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _stack(items, key):
    return torch.stack([it[key] for it in items])


def extract_records(*, checkpoint: Path, index_path: Path, role_filter: str | None,
                    v93_path: Path | None, cache_dir: Path, device: str):
    key = _cache_key(checkpoint, index_path, role_filter, v93_path)
    cache_dir.mkdir(parents=True, exist_ok=True); cp = cache_dir / f"{key}.pt"
    if cp.is_file():
        obj = torch.load(cp, map_location="cpu", weights_only=False)
        if obj.get("cache_key") == key:
            return obj["records"], obj["event"]

    v93 = build_v93_map(v93_path)
    groups = label_groups(index_path, role_filter=role_filter, v93_map=v93)
    needed=[]
    for g in groups:
        needed.append(Path(g["nominal_path"])); needed.extend(Path(c["path"]) for c in g["candidates"])
    seen=set(); paths=[]
    for p in needed:
        q=str(p.resolve())
        if q not in seen: seen.add(q); paths.append(p)

    bundle=load_model_bundle(checkpoint,{"training":{"device":device}})
    if bundle is None: raise RuntimeError(f"cannot load checkpoint {checkpoint}")
    model=bundle.model.eval(); [p.requires_grad_(False) for p in model.parameters()]
    if not isinstance(model.encoder, StructuredTokenEncoder): raise RuntimeError("V48.109 requires StructuredTokenEncoder")
    enc=model.encoder.eval(); dev=bundle.device
    if len(enc.encoder.layers)!=2: raise RuntimeError("V48.109 requires historical two-layer Stage-I")
    cfg,feature_event=feature_only_dataset_cfg(bundle.cfg,cache_dir=str(cache_dir/"tensor"),workers=8)
    ds=OCRAPSampleDataset(paths,cfg)
    if ds.absolute_truth_contract_event.get("enabled") or ds.action_response_truth_event.get("enabled"):
        raise RuntimeError("V48.109 feature-only dataset unexpectedly attached truth sidecars")
    if [str(p.resolve()) for p in paths] != [str(p.resolve()) for p in ds.paths]:
        raise RuntimeError("V48.109 dataset path order differs from index")
    idx={str(p.resolve()):i for i,p in enumerate(ds.paths)}

    records=[]; scene_delta_max=0.0
    for g in groups:
        ordered=[g["nominal_path"]]+[c["path"] for c in g["candidates"]]
        if any(str(Path(p).resolve()) not in idx for p in ordered): continue
        items=[ds[idx[str(Path(p).resolve())]] for p in ordered]
        x=_stack(items,"x").to(dev)
        with torch.no_grad():
            raw=raw_candidate_pathway(x,enc.layout)
            scene=raw_scene_context_summary(x,enc.layout)
            scene_delta_max=max(scene_delta_max,float((scene[1:]-scene[0:1]).abs().max().item()))
            state,delta,reserve_context=action_features(raw)
            sn=scene[0:1].expand(delta.shape[0],-1)
            stn=state.cpu().numpy(); dn=delta.cpu().numpy(); qn=reserve_context.cpu().numpy(); cn=sn.cpu().numpy()
        for j,c in enumerate(g["candidates"]):
            records.append({
                "group":tuple(g["key"]),"candidate":int(c["candidate"]),"group_mode":g["group_mode"],
                "safe_positive":bool(c["safe_positive"]),"teacher_harmful":bool(c["teacher_harmful"]),"mediation_mode":c["mediation_mode"],
                "raw_state":stn[j],"support_u":dn[j],"reserve_u":qn[j],"scene_context":cn[j],
            })
    if scene_delta_max > 1.0e-6:
        raise RuntimeError(f"V48.109 scene context changed across candidate actions: {scene_delta_max}")
    event={
        "records":len(records),"groups":len(groups),"raw_candidate_dim":RAW_CANDIDATE_DIM,
        "raw_scene_context_dim":RAW_SCENE_CONTEXT_DIM,"relational_dim":RELATIONAL_DIM,
        "scene_context_candidate_delta_max_abs":scene_delta_max,
        "scene_context_definition":"agent_summary_plus_agent_mean_std_max_min_plus_bev_route_map_dyn",
        "agent_set_summary_permutation_invariant":True,
        "feature_only_dataset_contract":feature_event,"tensor_cache_event":ds.tensor_cache_event,
        "encoder_layer_count":2,
    }
    torch.save({"cache_key":key,"records":records,"event":event},cp)
    return records,event


def _fit_axis(records: list[dict[str,Any]], key: str):
    U=np.stack([r[key] for r in records]).astype(np.float64)
    C=np.stack([r["scene_context"] for r in records]).astype(np.float64)
    y=np.asarray([r["label"] for r in records],dtype=np.int64)
    scaler=fit_relational_scaler(U,C)
    Uperm=permute_within_group(records,key).astype(np.float64)
    models={
        "base_true":fit_closed_form_ridge(base_features(U,scaler),y),
        "base_shuffle":fit_closed_form_ridge(base_features(Uperm,scaler),y),
        "relational_true":fit_closed_form_ridge(relational_features(U,C,scaler),y),
        "relational_shuffle":fit_closed_form_ridge(relational_features(Uperm,C,scaler),y),
    }
    return {"scaler":scaler,"models":models,"count":len(records)}


def _fit_family(records):
    su=action_subset(records,"drs_activation"); re=action_subset(records,"deployability_gain")
    return {"support":_fit_axis(su,"support_u"),"reserve":_fit_axis(re,"reserve_u"),"counts":{"support":len(su),"reserve":len(re)}}


def _metric_from_scores(records, sc):
    if not records:
        return {"rows":0,"positive_rows":0,"negative_rows":0,"auc":None,"top1":None,"powered_groups":0}
    y=np.asarray([r["label"] for r in records],dtype=np.int64); sc=np.asarray(sc,dtype=np.float64)
    groups=defaultdict(list)
    for i,r in enumerate(records): groups[tuple(r["group"])].append(i)
    powered=[ids for ids in groups.values() if any(y[i]==1 for i in ids) and any(y[i]==0 for i in ids)]
    top1=float(np.mean([y[max(ids,key=lambda i:float(sc[i]))]==1 for ids in powered])) if powered else None
    return {"rows":len(records),"positive_rows":int(y.sum()),"negative_rows":int(len(y)-y.sum()),"auc":auc(y,sc),"top1":top1,"powered_groups":len(powered)}


def _eval_axis(records,key,fit):
    U=np.stack([r[key] for r in records]).astype(np.float64) if records else np.empty((0,RAW_CANDIDATE_DIM))
    C=np.stack([r["scene_context"] for r in records]).astype(np.float64) if records else np.empty((0,RAW_SCENE_CONTEXT_DIM))
    Up=permute_within_group(records,key).astype(np.float64) if records else U
    sca=fit["scaler"]; m=fit["models"]
    out={}
    for space,feat,featp in (
        ("base",base_features(U,sca),base_features(Up,sca)),
        ("relational",relational_features(U,C,sca),relational_features(Up,C,sca)),
    ):
        t=_metric_from_scores(records,ridge_scores(m[f"{space}_true"],feat))
        s=_metric_from_scores(records,ridge_scores(m[f"{space}_shuffle"],featp))
        t["auc_vs_shuffled"]=None if t["auc"] is None or s["auc"] is None else float(t["auc"]-s["auc"])
        t["top1_vs_shuffled"]=None if t["top1"] is None or s["top1"] is None else float(t["top1"]-s["top1"])
        out[space]=(t,s)
    return out


def _eval_family(dev_records,cert_records,family):
    base_cells={}; rel_cells={}
    for role in ROLES:
        src=dev_records if role.startswith("dev_") else cert_records
        rr=split_role(src,role); su=action_subset(rr,"drs_activation"); re=action_subset(rr,"deployability_gain")
        sm=_eval_axis(su,"support_u",family["support"]); rm=_eval_axis(re,"reserve_u",family["reserve"])
        for cells,space in ((base_cells,"base"),(rel_cells,"relational")):
            cells[role]={
                "support_true":sm[space][0],"support_shuffled":sm[space][1],
                "reserve_true":rm[space][0],"reserve_shuffled":rm[space][1],
            }
    return base_cells,rel_cells


def _pack_axis(fit):
    sc=fit["scaler"]
    return {
        "scaler":{"u_scale":sc.u_scale,"c_mu":sc.c_mu,"c_sd":sc.c_sd},
        "models":{k:{"coef":v.coef,"ridge_lambda":v.ridge_lambda,"objective":v.objective,"normal_equation_residual":v.normal_equation_residual} for k,v in fit["models"].items()},
        "count":fit["count"],
    }


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",type=Path,required=True); ap.add_argument("--train-index",type=Path,required=True)
    ap.add_argument("--dev-index",type=Path,required=True); ap.add_argument("--certificate-index",type=Path,required=True)
    ap.add_argument("--v93-audit",type=Path,required=True); ap.add_argument("--cache-dir",type=Path,required=True)
    ap.add_argument("--device",default="cuda"); ap.add_argument("--variant",required=True)
    ap.add_argument("--output",type=Path,required=True); ap.add_argument("--state-output",type=Path,required=True)
    a=ap.parse_args(); t0=time.perf_counter()
    tr,etr=extract_records(checkpoint=a.checkpoint,index_path=a.train_index,role_filter=None,v93_path=None,cache_dir=a.cache_dir/"train",device=a.device)
    dv=[];ce=[];events={"train":etr}
    for role in ("dev_near","dev_contact"):
        r,e=extract_records(checkpoint=a.checkpoint,index_path=a.dev_index,role_filter=role,v93_path=a.v93_audit,cache_dir=a.cache_dir/role,device=a.device);dv+=r;events[role]=e
    for role in ("certificate_near","certificate_contact"):
        r,e=extract_records(checkpoint=a.checkpoint,index_path=a.certificate_index,role_filter=role,v93_path=a.v93_audit,cache_dir=a.cache_dir/role,device=a.device);ce+=r;events[role]=e
    if not tr or not dv or not ce: raise RuntimeError("V48.109 empty audit records")
    fam=_fit_family(tr); base_cells,rel_cells=_eval_family(dv,ce,fam)
    max_resid=max(v.normal_equation_residual for axis in ("support","reserve") for v in fam[axis]["models"].values())
    result={
        "schema":"ocrap-v48.109-raw-candidate-scene-convex-relational-orientation-audit-v1","engineering_version":ENGINEERING_VERSION,"algorithm_name":ALGORITHM_NAME,
        "valid":True,"variant":a.variant,"audit_only":True,"checkpoint":str(a.checkpoint.resolve()),"checkpoint_sha256":sha256(a.checkpoint),
        "base_cells":base_cells,"relational_cells":rel_cells,"events":events,"train_counts":fam["counts"],
        "convex_closed_form_ridge":True,"strictly_convex_unique_solution":True,"iterative_optimizer_used":False,
        "ridge_lambda_rule":"1_over_axis_train_rows","max_normal_equation_residual":max_resid,
        "score_family":"u_T_w_plus_u_T_W_c","nominal_zero_score_by_construction":True,
        "scene_context_candidate_invariant":True,"agent_set_summary_permutation_invariant":True,
        "raw_candidate_dim":RAW_CANDIDATE_DIM,"raw_scene_context_dim":RAW_SCENE_CONTEXT_DIM,"relational_dim":RELATIONAL_DIM,
        "planner_parameters_trained":0,"stage_i_parameters_trained":0,"root_decoder_parameters_trained":0,"source_parameters_trained":0,
        "relative_ranker_modified":False,"regime_conditioning":False,"boundary_transport":False,"teacher_metadata_input_to_model":False,"test_roots_read":False,
        "posthoc_feature_selection":False,"elapsed_seconds":float(time.perf_counter()-t0),
    }
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    torch.save({
        "schema":"ocrap-v48.109-convex-relational-orientation-state-v1","engineering_version":ENGINEERING_VERSION,"algorithm_name":ALGORITHM_NAME,"variant":a.variant,
        "support":_pack_axis(fam["support"]),"reserve":_pack_axis(fam["reserve"]),
        "convex_closed_form_ridge":True,"strictly_convex_unique_solution":True,"iterative_optimizer_used":False,
        "checkpoint_sha256":sha256(a.checkpoint),
    },a.state_output)
    print(json.dumps({"valid":True,"variant":a.variant,"max_normal_equation_residual":max_resid,"elapsed_seconds":result["elapsed_seconds"]}))
    return 0

if __name__=="__main__": raise SystemExit(main())
