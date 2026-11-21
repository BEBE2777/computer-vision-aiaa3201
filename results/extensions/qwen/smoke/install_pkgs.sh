#!/bin/bash
# Install Qwen deps into an isolated target dir, reusing the venv's existing torch.
V=/tmp/slv176/venv/bin/python
TARGET=/data/disk2/home/qzheng610/qwen_work/pylibs
LOG=/data/disk2/home/qzheng610/qwen_work/pkgs.log
exec > "$LOG" 2>&1
echo "START $(date)"
mkdir -p "$TARGET"

# main packages without pulling their own torch/numpy duplicates
$V -m pip install --no-deps --target "$TARGET" -i https://pypi.tuna.tsinghua.edu.cn/simple \
  transformers accelerate qwen-vl-utils sentencepiece safetensors huggingface_hub pillow numpy scikit-learn

# lightweight pure-python deps that transformers/qwen-vl-utils actually import at runtime
$V -m pip install --no-deps --target "$TARGET" -i https://pypi.tuna.tsinghua.edu.cn/simple \
  packaging pyyaml regex tqdm requests filelock fsspec networkx jinja2 markupsafe \
  typing-extensions sympy mpmath threadpoolctl scipy

echo "EXIT=$? $(date)"
echo "=== verify ==="
PYTHONPATH="$TARGET" $V -c "import transformers;print('transformers',transformers.__version__)"
PYTHONPATH="$TARGET" $V -c "import qwen_vl_utils;print('qwen_vl_utils OK')"
PYTHONPATH="$TARGET" $V -c "import accelerate;print('accelerate',accelerate.__version__)"
PYTHONPATH="$TARGET" $V -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
PYTHONPATH="$TARGET" $V -c "import av" 2>&1 | head -2
echo "DONE"
