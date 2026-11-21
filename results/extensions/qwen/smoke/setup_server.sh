#!/bin/bash
# Run on the course GPU box (121.12.5.46) as qzheng610.
# Builds an isolated venv with CUDA torch + Qwen deps.
set -e
BASE=/data/disk1/home/qzheng610/qwen_env
mkdir -p "$BASE"
python3 -m venv "$BASE"
PIP="$BASE/bin/pip"
PYTHON="$BASE/bin/python"

$PIP install -q -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip
# CUDA 12.4 wheels (driver 570 supports cu124)
$PIP install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124
# Qwen stack
$PIP install -q -i https://pypi.tuna.tsinghua.edu.cn/simple \
    transformers accelerate qwen-vl-utils sentencepiece pillow numpy safetensors huggingface_hub

echo "=== VERIFY ==="
$PYTHON -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
$PYTHON -c "import transformers, qwen_vl_utils; print('transformers', transformers.__version__, 'qwen_vl_utils ok')"
echo "SETUP_DONE"
