#!/bin/bash
# Wait for model download, verify shards, then run the GPU smoke test (1-shot, 2 per class).
BASE=/data/disk2/home/qzheng610/qwen_work
MODEL=$BASE/qwen2vl2b_local
LOG=$BASE/auto_smoke.log
exec > "$LOG" 2>&1
echo "START $(date)"

# wait until both shards are present and full size (wait at most 60 min)
S1=$MODEL/model-00001-of-00002.safetensors
S2=$MODEL/model-00002-of-00002.safetensors
for i in $(seq 1 360); do
  sz1=$(stat -c %s "$S1" 2>/dev/null || echo 0)
  sz2=$(stat -c %s "$S2" 2>/dev/null || echo 0)
  if [ "$sz1" -eq 3988609112 ] && [ "$sz2" -eq 429441656 ]; then
    echo "model complete at $(date)"; break
  fi
  if [ $((i % 30)) -eq 0 ]; then echo "waiting... s1=$sz1 s2=$sz2 $(date)"; fi
  sleep 10
done

sz1=$(stat -c %s "$S1" 2>/dev/null || echo 0)
if [ "$sz1" -ne 3988609112 ]; then echo "MODEL INCOMPLETE s1=$sz1"; exit 1; fi

# sanity check tokenizer + config
V=/tmp/slv176/venv/bin/python
export PYTHONPATH=$BASE/pylibs
$V -c "import json; json.load(open('$MODEL/tokenizer.json')); json.load(open('$MODEL/vocab.json')); print('tokenizer OK')" || exit 2

echo "=== RUN GPU SMOKE 1-shot ==="
cd $BASE
$V gpu_smoke.py --shots 1 --seed 0 --per-class 2 --max-new-tokens 256 2>&1
echo "SMOKE_EXIT=$? $(date)"
echo "DONE"
