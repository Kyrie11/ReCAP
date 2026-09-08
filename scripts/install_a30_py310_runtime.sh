#!/usr/bin/env bash
set -euo pipefail

REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
CONSTRAINTS="$REPO/constraints/a30_py310_runtime.txt"

python - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit(
        f"This lock is for Python 3.10, got {sys.version.split()[0]}. "
        "Create a fresh env: conda create -n ocrap-a30 python=3.10.16 pip -y"
    )
PY

python -m pip install --upgrade pip setuptools wheel

# Never repair this particular failure by skipping JAX CUDA checks.  Clean out
# mixed JAX/TF installations first; tensorflow and tensorflow-cpu share the same
# import namespace and must not coexist.
python -m pip uninstall -y \
  jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt jax-cuda13-plugin jax-cuda13-pjrt \
  tensorflow tensorflow-cpu tensorflow-intel \
  torch torchvision torchaudio || true

# JAX recommends pip-managed CUDA and warns that LD_LIBRARY_PATH can override
# those wheels.  Do not permanently edit the user's shell; sanitize only this
# installation/verification process.
unset LD_LIBRARY_PATH || true

# 1) Install the exact CUDA-12.8 PyTorch wheel first.  Its shared NVIDIA wheels
# include cuDNN 9.10.2.21, which also satisfies JAX 0.6.2's CUDA-12 runtime.
python -m pip install -c "$CONSTRAINTS" \
  torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# 2) Install JAX 0.6.2 CUDA12 through the pip-managed CUDA extra.  Constraints
# keep common CUDA libraries on the same CUDA-12.8 versions used by Torch.
python -m pip install -c "$CONSTRAINTS" "jax[cuda12]==0.6.2"

# 3) TensorFlow is used by the WOMD/Waymax input pipeline, not as the accelerator
# backend in OC-RAP.  Install the normal Linux CPU dependency path (NO and-cuda
# extra); OC-RAP additionally hides GPUs from TensorFlow before Waymax imports.
python -m pip install -c "$CONSTRAINTS" \
  tensorflow==2.18.1 numpy==1.26.4 scipy==1.13.1 ml-dtypes==0.5.1 protobuf==4.25.8

# 4) Freeze the JAX ecosystem before installing Waymax so Flax/Orbax cannot
# opportunistically upgrade JAX beyond the Python-3.10 validated stack.
python -m pip install -c "$CONSTRAINTS" \
  flax==0.10.6 chex==0.1.89 optax==0.2.4 orbax-checkpoint==0.11.12

# Waymax's official repository currently installs from main and declares
# tensorflow>=2.11, jax>=0.4.6, chex>=0.1.6 and flax>=0.6.7.  The pinned stack
# above satisfies those constraints, so pip should not replace it.
python -m pip install -c "$CONSTRAINTS" \
  "git+https://github.com/waymo-research/waymax.git@main#egg=waymo-waymax"

# Install OC-RAP itself without letting editable-install dependency resolution
# replace the pinned Torch/JAX stack.
python -m pip install -e . --no-deps
python -m pip install "PyYAML>=6.0" "tqdm>=4.66" "crc32c>=2.3"

python -m pip check

echo
echo "[VERIFY] package/runtime stack"
OCRAP_TENSORFLOW_CPU_ONLY=1 python tools/check_jax_waymax_runtime.py --require-gpu --check-waymax

echo
echo "[VERIFY] PyTorch CUDA"
python - <<'PY'
import torch
print({
    "torch": torch.__version__,
    "torch_cuda_build": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
    "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
})
if not torch.cuda.is_available():
    raise SystemExit("PyTorch CUDA is not available")
PY
