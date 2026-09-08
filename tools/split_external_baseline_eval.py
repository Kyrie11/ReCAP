#!/usr/bin/env python3
"""Split a multi-method external-baseline evaluation artifact.

The evaluator is deliberately able to score several non-learning baselines in a
single dataset pass.  This helper restores the historical one-JSON-per-method
file contract used by the launch scripts and downstream table tooling.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict) or not isinstance(obj.get("methods"), dict):
        raise ValueError(f"Not an external-baseline evaluation artifact: {path}")
    return obj


def _singleton(parent: dict[str, Any], method: str, parent_path: Path) -> dict[str, Any]:
    methods = parent["methods"]
    if method not in methods:
        raise KeyError(f"Method {method!r} not found; available={sorted(methods)}")
    out = copy.deepcopy(parent)
    out["method_order"] = [method]
    out["methods"] = {method: copy.deepcopy(methods[method])}
    timing = out.get("timing")
    if isinstance(timing, dict):
        by_method = timing.get("selection_s_by_method")
        if isinstance(by_method, dict):
            timing["selection_s_by_method"] = {method: by_method.get(method, 0.0)}
        # The shared load/risk time cannot be uniquely attributed to one method.
        # Keep it for transparency and mark that it came from a batched parent.
        timing["batched_parent"] = str(parent_path)
    out["batched_evaluation_parent"] = str(parent_path)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--prefix", required=True, help="Filename prefix, e.g. eval_contact_")
    ap.add_argument("--methods", default="", help="Comma-separated subset; default=parent method_order")
    args = ap.parse_args()

    src = Path(args.input)
    parent = _load(src)
    requested = [x.strip() for x in args.methods.split(",") if x.strip()]
    if not requested:
        requested = [str(x) for x in parent.get("method_order", parent["methods"].keys())]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for method in requested:
        dst = out_dir / f"{args.prefix}{method}.json"
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        tmp.write_text(json.dumps(_singleton(parent, method, src), indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(dst)
        print({"event": "split_external_eval", "method": method, "output": str(dst)}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
