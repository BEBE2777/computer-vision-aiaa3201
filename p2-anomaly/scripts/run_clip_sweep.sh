#!/bin/bash
# CLIP layer sweep at 448 (pilot: which block carries usable dense features?)
set -u
cd /data/disk1/home/slv176/aiaa3201_p2 || exit 1
PY=/tmp/slv176/venv/bin/python
export OMP_NUM_THREADS=1 DINOV2_HUB_DIR=/tmp/slv176/third_party/dinov2 MPLBACKEND=Agg PYTHONUNBUFFERED=1
CACHE=/data/disk1/home/slv176/featcache32
OUT=/tmp/slv176/grid
LOG=/tmp/slv176/grid/clip_layer_sweep.log
: > "$LOG"

for L in 6 9; do
  echo "########## clip448_L$L ##########" >> "$LOG"
  $PY run_grid.py --name "smoke_clip448_L$L" --cache "$CACHE" --out "$OUT" --force \
      --backbone clip --resolution 448 --no_mask --layers $L \
      --shots 1 --seeds 0 --categories candle capsules >> "$LOG" 2>&1
  echo "exit=$? for L$L" >> "$LOG"
done

echo "ALL_SWEEP_DONE" >> "$LOG"
touch /tmp/slv176/grid/SWEEP_DONE
