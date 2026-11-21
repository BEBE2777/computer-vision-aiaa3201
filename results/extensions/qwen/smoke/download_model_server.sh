#!/bin/bash
# Download Qwen2-VL-2B directly on the GPU box via hf-mirror (fast internal network).
V=/tmp/slv176/venv/bin/python
T=/data/disk2/home/qzheng610/qwen_work/pylibs
DST=/data/disk2/home/qzheng610/qwen_work/qwen2vl2b_local
LOG=/data/disk2/home/qzheng610/qwen_work/model_dl.log
exec > "$LOG" 2>&1
echo "START $(date)"
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=$T
mkdir -p "$DST"
$V - <<PY
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen2-VL-2B-Instruct", local_dir="$DST", max_workers=8)
print("DOWNLOADED", p)
PY
echo "EXIT=$? $(date)"
du -sh "$DST"
echo DONE
