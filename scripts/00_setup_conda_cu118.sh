#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-text2sql}"
PYPI_MIRROR="${PYPI_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"

conda create -n "${ENV_NAME}" python=3.10 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

python -m pip install -U pip
python -m pip install torch==2.7.0+cu118 torchvision==0.22.0+cu118 torchaudio==2.7.0+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-train-cu118.txt -i "${PYPI_MIRROR}"

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu0:", torch.cuda.get_device_name(0))
PY

echo "Done. Activate with: conda activate ${ENV_NAME}"
