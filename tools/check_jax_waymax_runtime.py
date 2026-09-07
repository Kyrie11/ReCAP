#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from importlib import metadata
from pathlib import Path
from typing import Any


PACKAGES = (
    "jax",
    "jaxlib",
    "jax-cuda12-plugin",
    "jax-cuda12-pjrt",
    "jax-cuda13-plugin",
    "jax-cuda13-pjrt",
    "waymo-waymax",
    "waymax",
    "tensorflow",
    "tensorflow-cpu",
)


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def collect_versions() -> dict[str, str]:
    return {name: v for name in PACKAGES if (v := _version(name)) is not None}


def _base(v: str | None) -> str | None:
    return None if v is None else v.split("+", 1)[0]


def obvious_stack_mismatches(versions: dict[str, str]) -> list[str]:
    out: list[str] = []
    jaxlib = _base(versions.get("jaxlib"))
    for name in ("jax-cuda12-plugin", "jax-cuda12-pjrt", "jax-cuda13-plugin", "jax-cuda13-pjrt"):
        v = _base(versions.get(name))
        if v and jaxlib and v != jaxlib:
            out.append(f"{name}=={v} does not match jaxlib=={jaxlib}")
    return out


def repair_hint(versions: dict[str, str]) -> str:
    cuda = "cuda13" if any(k.startswith("jax-cuda13") for k in versions) else "cuda12"
    return (
        "Recommended clean repair (JAX-managed CUDA wheels):\n"
        "  python -m pip uninstall -y jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt "
        "jax-cuda13-plugin jax-cuda13-pjrt\n"
        f"  python -m pip install --upgrade \"jax[{cuda}]\"\n"
        "  python -m pip install --upgrade "
        "git+https://github.com/waymo-research/waymax.git@main#egg=waymo-waymax\n"
        "If you intentionally use a system CUDA toolkit instead, use the matching "
        f"\"jax[{cuda}-local]\" extra rather than mixing plugin/PJRT versions."
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--require-gpu", action="store_true")
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    versions = collect_versions()
    result: dict[str, Any] = {
        "versions": versions,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
        "obvious_version_mismatches": obvious_stack_mismatches(versions),
        "ok": False,
    }
    try:
        import jax  # type: ignore

        devices = jax.devices()
        platforms = sorted({str(getattr(d, "platform", "unknown")) for d in devices})
        result.update({
            "jax_import_version": getattr(jax, "__version__", versions.get("jax")),
            "devices": [str(d) for d in devices],
            "platforms": platforms,
        })
        if args.require_gpu and "gpu" not in platforms and "cuda" not in platforms:
            raise RuntimeError(f"JAX initialized but no GPU backend is visible; platforms={platforms}")
        result["ok"] = True
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result["repair_hint"] = repair_hint(versions)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["ok"]:
        print("\n[JAX/WAYMAX PREFLIGHT FAILED]", file=sys.stderr)
        for line in result.get("obvious_version_mismatches", []):
            print(f"- {line}", file=sys.stderr)
        print(result.get("repair_hint", repair_hint(versions)), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
