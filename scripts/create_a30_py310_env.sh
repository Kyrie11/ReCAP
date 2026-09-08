#!/usr/bin/env bash
set -euo pipefail
ENV_NAME="${1:-ocrap-a30}"
conda create -n "$ENV_NAME" python=3.10.16 pip -y
echo "Created $ENV_NAME. Next: conda activate $ENV_NAME && bash scripts/install_a30_py310_runtime.sh"
