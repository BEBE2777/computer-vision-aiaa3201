#!/bin/bash
# Fetch the large model shard with curl (resumable), replacing the stalled hf_hub download.
DST=/data/disk2/home/qzheng610/qwen_work/qwen2vl2b_local
LOG=/data/disk2/home/qzheng610/qwen_work/shard1.log
exec > "$LOG" 2>&1
echo "START $(date)"
BASE=https://hf-mirror.com/Qwen/Qwen2-VL-2B-Instruct/resolve/main
# remove stalled incomplete file, start clean
find "$DST/.cache/huggingface/download" -name "*.incomplete" -delete 2>/dev/null
curl -L --retry 5 --retry-delay 2 -C - -o "$DST/model-00001-of-00002.safetensors" \
  "$BASE/model-00001-of-00002.safetensors"
echo "EXIT=$? $(date)"
stat -c '%n %s' "$DST/model-00001-of-00002.safetensors"
echo DONE
