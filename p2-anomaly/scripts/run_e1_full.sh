#!/bin/bash
# E1 full grids under the frozen protocol.
#   CLIP: ViT-B/16 @448, short_edge preprocessing (DINOv2-identical framing),
#         layer 9, no PCA mask
#   DINO: ViT-S/14 @448, layer 12, no PCA mask (controlled counterpart)
cd /data/disk1/home/slv176/aiaa3201_p2 || exit 1
PY=/tmp/slv176/venv/bin/python
export OMP_NUM_THREADS=1 DINOV2_HUB_DIR=/tmp/slv176/third_party/dinov2 MPLBACKEND=Agg PYTHONUNBUFFERED=1
CACHE=/data/disk1/home/slv176/featcache32
OUT=/tmp/slv176/grid
LOG=/tmp/slv176/grid/e1_full.log
: > "$LOG"

echo "########## clip448_shorde_L9 full ##########" >> "$LOG"
$PY run_grid.py --name e1_clip448_L9_nomask --cache "$CACHE" --out "$OUT" --force \
    --backbone clip --resolution 448 --no_mask --layers 9 --clip_resize_mode short_edge \
    --shots 1 2 4 8 --seeds 0 1 2 >> "$LOG" 2>&1
echo "exit=$? clip" >> "$LOG"

echo "########## dinov2 448 nomask full ##########" >> "$LOG"
$PY run_grid.py --name e1_dino448_nomask --cache "$CACHE" --out "$OUT" --force \
    --backbone dinov2 --resolution 448 --no_mask --layers 12 \
    --shots 1 2 4 8 --seeds 0 1 2 >> "$LOG" 2>&1
echo "exit=$? dino" >> "$LOG"

echo "ALL_E1_DONE" >> "$LOG"
touch /tmp/slv176/grid/E1_DONE
