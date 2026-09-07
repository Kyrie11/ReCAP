#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, json, time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ocrap.models.data import OCRAPSampleDataset
from ocrap.models.inference import load_model_bundle
from ocrap.models.encoders import StructuredTokenEncoder
from ocrap.v48_96_support_reserve_root_observability import feature_only_dataset_cfg
from ocrap.v48_108_raw_to_projected_action_pathway_audit import (
    ALGORITHM_NAME, ENGINEERING_VERSION, action_features, projection_structure,projection_structural_injectivity_event,
    projected_candidate_pathway, raw_candidate_pathway, raw_pathway_dim,
    raw_static_context, reconstruct_raw_delta_from_projected,
)
from tools.run_v48_102_stage_i_action_information_transport_audit import (
    action_metrics, action_subset, build_v93_map, fit_binary, label_groups,
    permute_within_group, scores, split_role, state_metrics, state_records,
)
from tools.run_v48_97_executable_recovery_state import ROLES, sha256

V107_ENGINEERING_VERSION = "v48.107.0-OC-FNAO"


def _cache_key(checkpoint: Path, index_path: Path, role_filter: str | None, v93_path: Path | None, *, version: str = ENGINEERING_VERSION) -> str:
    payload = {
        "version": version, "checkpoint": sha256(checkpoint),
        "index": sha256(index_path), "role": role_filter,
        "v93": sha256(v93_path) if v93_path and v93_path.is_file() else None,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _stack(items, key):
    return torch.stack([it[key] for it in items])


def extract_raw_projected_records(*, checkpoint: Path, index_path: Path, role_filter: str | None,
                                  v93_path: Path | None, cache_dir: Path, device: str):
    key = _cache_key(checkpoint, index_path, role_filter, v93_path)
    cache_dir.mkdir(parents=True, exist_ok=True); cp = cache_dir / f"{key}.pt"
    # V48.108.1 changes only the structural branch owner, not feature extraction.
    # Reuse a byte-compatible V48.108.0 feature cache when available.
    legacy_key = _cache_key(checkpoint, index_path, role_filter, v93_path, version="v48.108.0-OC-RPAP")
    legacy_cp = cache_dir / f"{legacy_key}.pt"
    if not cp.is_file() and legacy_cp.is_file():
        cp = legacy_cp; key = legacy_key
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
    if not isinstance(model.encoder,StructuredTokenEncoder): raise RuntimeError("V48.108 requires StructuredTokenEncoder")
    enc=model.encoder.eval(); dev=bundle.device
    if len(enc.encoder.layers)!=2: raise RuntimeError("V48.108 requires historical two-layer Stage-I")
    cfg,feature_event=feature_only_dataset_cfg(bundle.cfg,cache_dir=str(cache_dir/"tensor"),workers=8)
    ds=OCRAPSampleDataset(paths,cfg)
    if ds.absolute_truth_contract_event.get("enabled") or ds.action_response_truth_event.get("enabled"):
        raise RuntimeError("V48.108 feature-only dataset unexpectedly attached truth sidecars")
    if [str(p.resolve()) for p in paths] != [str(p.resolve()) for p in ds.paths]:
        raise RuntimeError("V48.108 dataset path order differs from index")
    idx={str(p.resolve()):i for i,p in enumerate(ds.paths)}

    records=[]; static_max=0.0; rec_max=0.0; rec_rel=0.0; proj_delta_max=0.0
    pstruct=projection_structure(enc)
    for g in groups:
        ordered=[g["nominal_path"]]+[c["path"] for c in g["candidates"]]
        if any(str(Path(p).resolve()) not in idx for p in ordered): continue
        items=[ds[idx[str(Path(p).resolve())]] for p in ordered]
        x=_stack(items,"x").to(dev)
        with torch.no_grad():
            raw=raw_candidate_pathway(x,enc.layout)
            proj=projected_candidate_pathway(enc,x)
            stat=raw_static_context(x,enc.layout)
            rs,rd,rc=action_features(raw); ps,pd,pc=action_features(proj)
            static_max=max(static_max,float((stat[1:]-stat[0:1]).abs().max().item()))
            ydelta,xrec=reconstruct_raw_delta_from_projected(enc,rd)
            # projected path delta must equal the direct block-diagonal projection.
            proj_delta_err=float((pd-ydelta.to(pd.device)).abs().max().item())
            proj_delta_max=max(proj_delta_max,proj_delta_err)
            err=(rd-xrec.to(rd.device)).abs(); rec_max=max(rec_max,float(err.max().item()))
            den=float(rd.norm().item()); rel=float(err.norm().item()/max(den,1e-12)); rec_rel=max(rec_rel,rel)
            rn,rdn,rcn=rs.cpu().numpy(),rd.cpu().numpy(),rc.cpu().numpy()
            pn,pdn,pcn=ps.cpu().numpy(),pd.cpu().numpy(),pc.cpu().numpy()
        for j,c in enumerate(g["candidates"]):
            records.append({
                "group":tuple(g["key"]),"candidate":int(c["candidate"]),"group_mode":g["group_mode"],
                "safe_positive":bool(c["safe_positive"]),"teacher_harmful":bool(c["teacher_harmful"]),"mediation_mode":c["mediation_mode"],
                "raw_state":rn[j],"raw_delta":rdn[j],"raw_context":rcn[j],
                "projected_state":pn[j],"projected_delta":pdn[j],"projected_context":pcn[j],
            })
    if static_max>1e-6:
        raise RuntimeError(f"V48.108 raw static context changed across candidate actions: {static_max}")
    event={
        "records":len(records),"groups":len(groups),"raw_candidate_pathway_dim":raw_pathway_dim(enc.layout),
        "projected_candidate_pathway_dim":5*int(enc.d_model),"projection_structure":pstruct,
        "projection_all_full_column_rank":all(bool(v["full_column_rank"]) for v in pstruct.values()),
        "raw_static_context_candidate_delta_max_abs":static_max,
        "raw_delta_reconstruction_max_abs":rec_max,"raw_delta_reconstruction_max_rel_l2":rec_rel,
        "projected_delta_block_map_max_abs":proj_delta_max,"feature_only_dataset_contract":feature_event,
        "tensor_cache_event":ds.tensor_cache_event,"encoder_layer_count":2,
        "raw_pathway_groups":["ego","prefix_param","macro_scalar","prefix_state","control"],
        "projected_token_indices":[1,2,3,4,5],
    }
    torch.save({"cache_key":key,"records":records,"event":event},cp)
    return records,event


def _fit_family(records, prefix: str, device: str, seed: int):
    st=state_records([{**r,"state":r[f"{prefix}_state"]} for r in records])
    su=action_subset(records,"drs_activation"); re=action_subset(records,"deployability_gain")
    sm,smu,ssd=fit_binary(np.stack([r["state"] for r in st]),np.asarray([r["label"] for r in st]),device,seed=seed)
    um,umu,usd=fit_binary(np.stack([r[f"{prefix}_delta"] for r in su]),np.asarray([r["label"] for r in su]),device,seed=seed+1)
    rm,rmu,rsd=fit_binary(np.stack([r[f"{prefix}_context"] for r in re]),np.asarray([r["label"] for r in re]),device,seed=seed+2)
    sup_perm=permute_within_group(su,f"{prefix}_delta"); up,upmu,upsd=fit_binary(sup_perm,np.asarray([r["label"] for r in su]),device,seed=seed+3)
    res_perm=permute_within_group(re,f"{prefix}_context"); rp,rpmu,rpsd=fit_binary(res_perm,np.asarray([r["label"] for r in re]),device,seed=seed+4)
    return {
        "state":(sm,smu,ssd),"support":(um,umu,usd),"reserve":(rm,rmu,rsd),
        "support_shuffle":(up,upmu,upsd),"reserve_shuffle":(rp,rpmu,rpsd),
        "counts":{"state":len(st),"support":len(su),"reserve":len(re)},
    }


def _eval_family(dev_records, cert_records, family, prefix, device):
    cells={}
    for role in ROLES:
        src = dev_records if role.startswith("dev_") else cert_records
        rr=split_role(src,role)
        sr=state_records([{**r,"state":r[f"{prefix}_state"]} for r in rr])
        ur=action_subset(rr,"drs_activation"); qr=action_subset(rr,"deployability_gain")
        sm,smu,ssd=family["state"]; um,umu,usd=family["support"]; rm,rmu,rsd=family["reserve"]
        up,upmu,upsd=family["support_shuffle"]; rp,rpmu,rpsd=family["reserve_shuffle"]
        s=state_metrics(sr,sm,smu,ssd,device)
        ut=action_metrics(ur,f"{prefix}_delta",um,umu,usd,device)
        us=action_metrics(ur,f"{prefix}_delta",up,upmu,upsd,device,X_override=permute_within_group(ur,f"{prefix}_delta") if ur else None)
        rt=action_metrics(qr,f"{prefix}_context",rm,rmu,rsd,device)
        rs=action_metrics(qr,f"{prefix}_context",rp,rpmu,rpsd,device,X_override=permute_within_group(qr,f"{prefix}_context") if qr else None)
        for t,z in ((ut,us),(rt,rs)):
            t["auc_vs_shuffled"]=None if t["auc"] is None or z["auc"] is None else float(t["auc"]-z["auc"])
            t["top1_vs_shuffled"]=None if t["top1"] is None or z["top1"] is None else float(t["top1"]-z["top1"])
        cells[role]={"state":s,"support_true":ut,"support_shuffled":us,"reserve_true":rt,"reserve_shuffled":rs}
    return cells


def _state_dict_pack(family):
    out={}
    for name in ("state","support","reserve","support_shuffle","reserve_shuffle"):
        m,mu,sd=family[name]
        out[name]={"state_dict":{k:v.detach().cpu() for k,v in m.state_dict().items()},"mu":np.asarray(mu),"sd":np.asarray(sd)}
    return out


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",type=Path,required=True); ap.add_argument("--train-index",type=Path,required=True)
    ap.add_argument("--dev-index",type=Path,required=True); ap.add_argument("--certificate-index",type=Path,required=True)
    ap.add_argument("--v93-audit",type=Path,required=True); ap.add_argument("--cache-dir",type=Path,required=True)
    ap.add_argument("--device",default="cuda"); ap.add_argument("--variant",required=True)
    ap.add_argument("--output",type=Path,required=True); ap.add_argument("--state-output",type=Path,required=True)
    a=ap.parse_args(); t0=time.perf_counter()
    tr,etr=extract_raw_projected_records(checkpoint=a.checkpoint,index_path=a.train_index,role_filter=None,v93_path=None,cache_dir=a.cache_dir/"train",device=a.device)
    dv=[]; ce=[]; events={"train":etr}
    for role in ("dev_near","dev_contact"):
        r,e=extract_raw_projected_records(checkpoint=a.checkpoint,index_path=a.dev_index,role_filter=role,v93_path=a.v93_audit,cache_dir=a.cache_dir/role,device=a.device); dv+=r; events[role]=e
    for role in ("certificate_near","certificate_contact"):
        r,e=extract_raw_projected_records(checkpoint=a.checkpoint,index_path=a.certificate_index,role_filter=role,v93_path=a.v93_audit,cache_dir=a.cache_dir/role,device=a.device); ce+=r; events[role]=e
    if not tr or not dv or not ce: raise RuntimeError("V48.108 empty audit records")
    raw=_fit_family(tr,"raw",a.device,108); proj=_fit_family(tr,"projected",a.device,118)
    raw_cells=_eval_family(dv,ce,raw,"raw",a.device); proj_cells=_eval_family(dv,ce,proj,"projected",a.device)
    structural=all(projection_structural_injectivity_event(e) for e in events.values())
    result={
        "schema":"ocrap-v48.108-raw-to-projected-action-pathway-audit-v1","engineering_version":ENGINEERING_VERSION,"algorithm_name":ALGORITHM_NAME,
        "valid":True,"variant":a.variant,"audit_only":True,"checkpoint":str(a.checkpoint.resolve()),"checkpoint_sha256":sha256(a.checkpoint),
        "raw_cells":raw_cells,"projected_cells":proj_cells,"events":events,"projection_structural_injectivity":structural,"projected_delta_block_map_absolute_error_diagnostic_only":True,
        "train_counts":{"raw":raw["counts"],"projected":proj["counts"]},
        "planner_parameters_trained":0,"stage_i_parameters_trained":0,"root_decoder_parameters_trained":0,"source_parameters_trained":0,
        "relative_ranker_modified":False,"regime_conditioning":False,"boundary_transport":False,"teacher_metadata_input_to_model":False,"test_roots_read":False,
        "same_v48_102_target_specific_probe_recipe":True,"raw_pathway_fixed_by_encoder_layout":True,"posthoc_token_selection":False,
        "elapsed_seconds":float(time.perf_counter()-t0),
    }
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    torch.save({
        "schema":"ocrap-v48.108-raw-projected-probe-state-v1","engineering_version":ENGINEERING_VERSION,"algorithm_name":ALGORITHM_NAME,"variant":a.variant,
        "raw_probes":_state_dict_pack(raw),"projected_probes":_state_dict_pack(proj),"projection_structural_injectivity":structural,"projected_delta_block_map_absolute_error_diagnostic_only":True,
        "checkpoint_sha256":sha256(a.checkpoint),
    },a.state_output)
    print(json.dumps({"valid":True,"variant":a.variant,"projection_structural_injectivity":structural,"elapsed_seconds":result["elapsed_seconds"]}))
    return 0

if __name__=="__main__": raise SystemExit(main())
