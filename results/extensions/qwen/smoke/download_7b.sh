#!/bin/bash
# Download Qwen2-VL-7B-Instruct to server, parallel shards via curl -C - (resumable).
BASE=/data/disk2/home/qzheng610/qwen_work
DST=$BASE/qwen2vl7b_local
LOG=$BASE/dl7b.log
mkdir -p "$DST"
exec > "$LOG" 2>&1
echo "START $(date)"

B="https://hf-mirror.com/Qwen/Qwen2-VL-7B-Instruct/resolve/main"
# small config/tokenizer files first
for f in config.json generation_config.json model.safetensors.index.json preprocessor_config.json tokenizer.json tokenizer_config.json vocab.json; do
  curl -sL --retry 3 --max-time 300 -o "$DST/$f" "$B/$f"
  echo "small $f $(stat -c%s "$DST/$f" 2>/dev/null) $(date)"
done

# 5 shards in parallel, resumable
SHARDS="model-00001-of-00005.safetensors model-00002-of-00005.safetensors model-00003-of-00005.safetensors model-00004-of-00005.safetensors model-00005-of-00005.safetensors"
for s in $SHARDS; do
  (
    curl -sL --retry 5 --retry-delay 3 -C - -o "$DST/$s" "$B/$s"
    echo "DONE $s $(stat -c%s "$DST/$s" 2>/dev/null) $(date)"
  ) &
done
wait
echo "ALL_SHARDS_DONE $(date)"
# verify tokenizer json valid
/tmp/slv176/venv/bin/python -c "import json;json.load(open('$DST/tokenizer.json'));json.load(open('$DST/vocab.json'));json.load(open('$DST/model.safetensors.index.json'));print('JSON_OK')"
echo "EXIT=0 $(date)"
