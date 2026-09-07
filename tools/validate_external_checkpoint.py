#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import torch


def _load(path: Path) -> dict[str, Any]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"checkpoint root must be a dict, got {type(obj).__name__}")
    return obj


def _validate(path: Path, *, require_contract: bool, require_implementation_version: str | None = None) -> dict[str, Any]:
    ckpt = _load(path)
    missing_keys = [key for key in ("cfg", "input_dim", "max_candidates", "model_state") if key not in ckpt]
    if missing_keys:
        raise ValueError(f"required checkpoint keys are missing: {missing_keys}")
    state = ckpt.get("model_state")
    if not isinstance(state, dict) or not state:
        raise ValueError("model_state is missing or empty")
    bad_tensors: list[str] = []
    for name, value in state.items():
        if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex()):
            if not bool(torch.isfinite(value).all()):
                bad_tensors.append(str(name))
                if len(bad_tensors) >= 20:
                    break
    if bad_tensors:
        raise ValueError(f"non-finite model tensors: {bad_tensors}")
    val_loss = float(ckpt.get("val_loss", float("nan")))
    if not math.isfinite(val_loss):
        raise ValueError(f"val_loss is not finite: {val_loss!r}")
    contract = ckpt.get("input_contract") or {}
    if require_contract:
        if int(contract.get("version", 0)) < 2:
            raise ValueError(f"input_contract.version must be >=2, got {contract.get('version')!r}")
        if contract.get("deployable_feature_only") is not True:
            raise ValueError("checkpoint is teacher-conditioned; deployable_feature_only is not true")
    implementation_version = str(ckpt.get("implementation_version", (((ckpt.get("cfg") or {}).get("external_baselines", {}) or {}).get("model", {}) or {}).get("implementation", "legacy_adapter")))
    if require_implementation_version is not None and implementation_version != str(require_implementation_version):
        raise ValueError(
            f"implementation_version mismatch: expected {require_implementation_version!r}, got {implementation_version!r}"
        )
    return {
        "checkpoint": str(path),
        "baseline": ckpt.get("baseline"),
        "epoch": int(ckpt.get("epoch", 0)),
        "val_loss": val_loss,
        "world_size": int(ckpt.get("world_size", 1)),
        "global_batch_size": int(ckpt.get("global_batch_size", 0)),
        "input_contract_version": int(contract.get("version", 0)),
        "deployable_feature_only": contract.get("deployable_feature_only"),
        "implementation_version": implementation_version,
        "num_model_tensors": len(state),
    }


def _atomic_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.tmp-{os.getpid()}")
    try:
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate an external-baseline PyTorch checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--promote-from", default=None, help="Validate and atomically copy this checkpoint when --checkpoint is missing.")
    parser.add_argument("--require-deployable-contract", action="store_true")
    parser.add_argument("--require-implementation-version", default=None)
    parser.add_argument("--allow-promotion", action="store_true", help="Explicitly permit --promote-from to become the requested checkpoint.")
    args = parser.parse_args()

    dst = Path(args.checkpoint)
    promoted = False
    if not dst.is_file():
        src = Path(args.promote_from) if args.promote_from else None
        if src is None or not src.is_file():
            raise SystemExit(f"missing checkpoint: {dst}")
        if not args.allow_promotion:
            raise SystemExit(
                f"missing checkpoint: {dst}; fallback exists at {src}, but promotion was not explicitly enabled"
            )
        try:
            source_info = _validate(src, require_contract=bool(args.require_deployable_contract), require_implementation_version=args.require_implementation_version)
        except Exception as exc:
            raise SystemExit(f"invalid fallback checkpoint {src}: {exc}") from None
        _atomic_copy(src, dst)
        promoted = True
        print(json.dumps({"event": "checkpoint_promoted", "source": str(src), "target": str(dst), **source_info}, sort_keys=True))

    try:
        info = _validate(dst, require_contract=bool(args.require_deployable_contract), require_implementation_version=args.require_implementation_version)
    except Exception as exc:
        raise SystemExit(f"invalid checkpoint {dst}: {exc}") from None
    print(json.dumps({"event": "checkpoint_valid", "promoted": promoted, **info}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
