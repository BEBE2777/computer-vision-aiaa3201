#!/usr/bin/env bash
# AIAA3201 Project 2 -- remote environment bootstrap (A6000 server)
# Layout: code in ~/aiaa3201_p2 , heavy stuff in /tmp/slv176 (root fs has ~80G free)
set -u
WORK=/tmp/slv176
LOG=$WORK/logs
mkdir -p "$WORK" "$LOG"
export PATH="$HOME/.local/bin:$PATH"
export UV_INDEX_URL=https://mirrors.cernet.edu.cn/pypi/web/simple
export UV_HTTP_TIMEOUT=180

echo "=== [1/5] uv ==="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh -o "$WORK/uv_install.sh" || { echo "UV_DOWNLOAD_FAIL"; exit 11; }
  sh "$WORK/uv_install.sh" || { echo "UV_INSTALL_FAIL"; exit 12; }
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version || exit 13

echo "=== [2/5] venv ==="
uv venv "$WORK/venv" --python "$(command -v python3)" || exit 21

echo "=== [3/5] torch ==="
uv pip install --python "$WORK/venv/bin/python" torch torchvision 2>&1 | tail -5
"$WORK/venv/bin/python" -c "import torch;print('TORCH_OK',torch.__version__,torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" || exit 31

echo "=== [4/5] rest ==="
uv pip install --python "$WORK/venv/bin/python" \
  "numpy==1.26.4" opencv-python-headless matplotlib Pillow scipy scikit-learn faiss-cpu \
  tqdm tifffile PyYAML pandas timm 2>&1 | tail -5
"$WORK/venv/bin/python" -c "
import cv2, numpy, faiss, sklearn, tifffile, yaml, pandas
print('DEPS_OK', numpy.__version__, faiss.__version__, cv2.__version__)
" || exit 41

echo "=== [5/5] dinov2 backbone smoke test (downloads weights) ==="
"$WORK/venv/bin/python" - <<'PY' || exit 51
import torch, time
t0 = time.time()
m = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
m.eval().cuda()
print('DINOV2_OK', sum(p.numel() for p in m.parameters())/1e6, 'M params', '%.1fs' % (time.time()-t0))
x = torch.randn(1, 3, 448, 448).cuda()
with torch.inference_mode():
    out = m.get_intermediate_layers(x, n=[6, 9, 12])
print('FEAT_SHAPES', [tuple(o.shape) for o in out])
PY

echo "ALL_DONE"
