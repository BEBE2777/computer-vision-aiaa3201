#!/bin/bash
# Part 3 step 2 smoke test, CLIP side (fp32 weights after dtype fix).
set -u
cd /data/disk1/home/slv176/aiaa3201_p2 || exit 1
PY=/tmp/slv176/venv/bin/python
export OMP_NUM_THREADS=1 DINOV2_HUB_DIR=/tmp/slv176/third_party/dinov2 MPLBACKEND=Agg PYTHONUNBUFFERED=1
CACHE=/data/disk1/home/slv176/featcache32
OUT=/tmp/slv176/grid
LOG=/tmp/slv176/grid/smoke_clip.log
: > "$LOG"

run() {
  echo "########## $1 ##########" >> "$LOG"
  $PY run_grid.py --name "$1" --cache "$CACHE" --out "$OUT" --force \
      --shots 1 --seeds 0 --categories candle capsules "${@:2}" >> "$LOG" 2>&1
  echo "exit=$? for $1" >> "$LOG"
}

run smoke_clip448_nomask --backbone clip --resolution 448 --no_mask
run smoke_clip224_nomask --backbone clip --resolution 224 --no_mask

echo "ALL_CLIP_DONE" >> "$LOG"
touch /tmp/slv176/grid/SMOKE_CLIP_DONE
