#!/bin/bash
# Wait for 7B shards to finish, verify sizes, then run 7B GPU smoke with the
# SAME frozen prompt/schema/eval scope as the 2B smoke (1-shot, seed 0, 2/class).
BASE=/data/disk2/home/qzheng610/qwen_work
DST=$BASE/qwen2vl7b_local
LOG=$BASE/auto_smoke_7b.log
exec > "$LOG" 2>&1
echo "START $(date)"

# expected sizes (bytes) from HF mirror
declare -A EXP
EXP[model-00001-of-00005.safetensors]=3898600000
EXP[model-00002-of-00005.safetensors]=3864700000
EXP[model-00003-of-00005.safetensors]=3864700000
EXP[model-00004-of-00005.safetensors]=3864700000
EXP[model-00005-of-00005.safetensors]=1090000000

# wait until all 5 shards reach expected size (tolerance 5MB), max 120 min
for i in $(seq 1 720); do
  all_ok=1
  for s in "${!EXP[@]}"; do
    sz=$(stat -c %s "$DST/$s" 2>/dev/null || echo 0)
    if [ "$sz" -lt "$(( ${EXP[$s]} - 5000000 ))" ]; then all_ok=0; break; fi
  done
  if [ "$all_ok" -eq 1 ]; then echo "shards complete at $(date)"; break; fi
  if [ $((i % 60)) -eq 0 ]; then
    echo "waiting... $(date)"
    for s in "${!EXP[@]}"; do echo "  $s $(stat -c %s "$DST/$s" 2>/dev/null)"; done
  fi
  sleep 10
done

# verify final sizes
for s in "${!EXP[@]}"; do
  sz=$(stat -c %s "$DST/$s" 2>/dev/null || echo 0)
  if [ "$sz" -lt "$(( ${EXP[$s]} - 5000000 ))" ]; then echo "SHARD_INCOMPLETE $s $sz"; exit 1; fi
done

V=/tmp/slv176/venv/bin/python
export PYTHONPATH=$BASE/pylibs
$V -c "import json;json.load(open('$DST/tokenizer.json'));json.load(open('$DST/model.safetensors.index.json'));print('JSON_OK')" || exit 2

echo "=== RUN GPU SMOKE 7B 1-shot ==="
cd $BASE
$V gpu_smoke.py --model 7b --shots 1 --seed 0 --per-class 2 --max-new-tokens 256 2>&1
echo "SMOKE_EXIT=$? $(date)"
echo "DONE_7B"
