#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
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
    "torch",
    "numpy",
    "scipy",
    "ml-dtypes",
    "flax",
    "chex",
    "optax",
    "orbax-checkpoint",
    "nvidia-cublas-cu12",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cudnn-cu12",
    "nvidia-nccl-cu12",
    "nvidia-nvjitlink-cu12",
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
    jax = _base(versions.get("jax"))
    if jax and jaxlib and jax != jaxlib:
        out.append(f"jax=={jax} does not match jaxlib=={jaxlib}")
    for name in ("jax-cuda12-plugin", "jax-cuda12-pjrt", "jax-cuda13-plugin", "jax-cuda13-pjrt"):
        v = _base(versions.get(name))
        if v and jaxlib and v != jaxlib:
            out.append(f"{name}=={v} does not match jaxlib=={jaxlib}")
    if any(k.startswith("jax-cuda12") for k in versions) and any(k.startswith("jax-cuda13") for k in versions):
        out.append("both JAX CUDA12 and CUDA13 plugins are installed")
    if "tensorflow" in versions and "tensorflow-cpu" in versions:
        out.append(
            "tensorflow and tensorflow-cpu are both installed; they provide the same import namespace and must not coexist"
        )
    if versions.get("jax") == "0.6.2":
        cudnn = versions.get("nvidia-cudnn-cu12")
        if cudnn:
            try:
                parts = tuple(int(x) for x in cudnn.split(".")[:3])
                if parts < (9, 10, 1):
                    out.append(f"nvidia-cudnn-cu12=={cudnn} is too old for the A30/JAX 0.6.2 repair target")
            except ValueError:
                pass
    if os.environ.get("LD_LIBRARY_PATH"):
        out.append(
            "LD_LIBRARY_PATH is set; it can override pip-managed CUDA/cuDNN libraries and make JAX load an older system cuDNN"
        )
    return out


def repair_hint() -> str:
    return (
        "Recommended repair for this repository / Python 3.10 / A30:\n"
        "  conda create -n ocrap-a30 python=3.10.16 pip -y\n"
        "  conda activate ocrap-a30\n"
        "  cd <OC-RAP-repo>\n"
        "  bash scripts/install_a30_py310_runtime.sh\n"
        "Do not set JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1. The original cuDNN warning can indicate incorrect multi-GPU results.\n"
        "This lock intentionally uses JAX 0.6.2 CUDA12 and PyTorch 2.8.0 cu128; driver 570.x is below the CUDA13 JAX driver floor."
    )


def _nvidia_smi() -> dict[str, Any]:
    try:
        cp = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,name,memory.total", "--format=csv,noheader,nounits"],
            text=True, capture_output=True, check=True, timeout=10,
        )
    except Exception as exc:
        return {"error": str(exc)}
    rows = []
    for line in cp.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 3:
            rows.append({"driver_version": parts[0], "name": parts[1], "memory_total_mib": parts[2]})
    return {"gpus": rows}


def _check_tensorflow_cpu_only(result: dict[str, Any]) -> None:
    # Import after JAX GPU discovery. TensorFlow is only the WOMD input pipeline
    # in OC-RAP, so hide GPUs through TF's own API without touching CUDA_VISIBLE_DEVICES.
    if os.environ.get("OCRAP_TENSORFLOW_CPU_ONLY", "1").strip().lower() in {"0", "false", "no", "off"}:
        result["tensorflow_cpu_only_requested"] = False
        return
    import tensorflow as tf  # type: ignore

    tf.config.set_visible_devices([], "GPU")
    result["tensorflow_cpu_only_requested"] = True
    result["tensorflow_version"] = getattr(tf, "__version__", None)
    result["tensorflow_visible_gpus"] = [str(d) for d in tf.config.get_visible_devices("GPU")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--require-gpu", action="store_true")
    ap.add_argument(
        "--check-waymax", action="store_true",
        help="Explicit compatibility alias; Waymax/TensorFlow are checked by default unless --jax-only is used.",
    )
    ap.add_argument("--jax-only", action="store_true", help="Skip TensorFlow and Waymax imports.")
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    versions = collect_versions()
    mismatches = obvious_stack_mismatches(versions)
    result: dict[str, Any] = {
        "python": sys.version.split()[0],
        "versions": versions,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
        "nvidia_smi": _nvidia_smi(),
        "obvious_version_mismatches": mismatches,
        "ok": False,
    }
    try:
        if mismatches:
            raise RuntimeError("; ".join(mismatches))

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

        # A tiny actual device computation catches plugin discovery that succeeds
        # but fails when XLA first touches CUDA.
        import jax.numpy as jnp  # type: ignore

        checksum = float(jax.device_get(jnp.sum(jnp.arange(1024, dtype=jnp.float32))))
        result["jax_smoke_checksum"] = checksum

        if not args.jax_only:
            _check_tensorflow_cpu_only(result)
            import waymax  # type: ignore
            from waymax import config as _wx_config  # noqa: F401
            from waymax import dataloader as _wx_dataloader  # noqa: F401

            result["waymax_import_ok"] = True
            result["waymax_module"] = getattr(waymax, "__file__", None)

        # Torch is optional for a pure Waymax stack but required by this repo's
        # learned Safe baselines. Report it and verify CUDA if installed.
        try:
            import torch  # type: ignore

            result["torch_import_version"] = getattr(torch, "__version__", versions.get("torch"))
            result["torch_cuda_build"] = torch.version.cuda
            result["torch_cuda_available"] = bool(torch.cuda.is_available())
            if args.require_gpu and not torch.cuda.is_available():
                raise RuntimeError("JAX sees a GPU but PyTorch CUDA is unavailable")
        except ImportError:
            result["torch_import_version"] = None

        result["ok"] = True
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result["repair_hint"] = repair_hint()

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["ok"]:
        print("\n[JAX/WAYMAX PREFLIGHT FAILED]", file=sys.stderr)
        for line in result.get("obvious_version_mismatches", []):
            print(f"- {line}", file=sys.stderr)
        print(result.get("repair_hint", repair_hint()), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
