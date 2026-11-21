#!/bin/bash
# Pilot: finish backbone/preprocessing/layer choice for the frozen E1 config.
cd /data/disk1/home/slv176/aiaa3201_p2 || exit 1
PY=/tmp/slv176/venv/bin/python
export OMP_NUM_THREADS=1 DINOV2_HUB_DIR=/tmp/slv176/third_party/dinov2 MPLBACKEND=Agg PYTHONUNBUFFERED=1
CACHE=/data/disk1/home/slv176/featcache32
OUT=/tmp/slv176/grid
LOG=/tmp/slv176/grid/pilot2.log
: > "$LOG"

one () {
  name="$1"; shift
  echo "########## $name ##########" >> "$LOG"
  $PY run_grid.py --name "$name" --cache "$CACHE" --out "$OUT" --force \
      --shots 1 --seeds 0 --categories candle capsules "$@" >> "$LOG" 2>&1
  echo "exit=$?" >> "$LOG"
}

one pilot_clip448_squash_L6   --backbone clip --resolution 448 --no_mask --layers 6  --clip_resize_mode squash
one pilot_clip448_squash_L12  --backbone clip --resolution 448 --no_mask --layers 12 --clip_resize_mode squash
one pilot_clip448_shorde_L9   --backbone clip --resolution 448 --no_mask --layers 9  --clip_resize_mode short_edge

echo "ALL_PILOT2_DONE" >> "$LOG"
touch /tmp/slv176/grid/PILOT2_DONE
