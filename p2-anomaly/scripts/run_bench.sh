#!/bin/bash
# Measure per-image latency + peak GPU memory for both visual arms (brief 6.2).
# Run AFTER the E1 grids are done; nothing here touches the feature cache or results.
#
#   scp scripts/bench_inference.py scripts/run_bench.sh slv176@121.12.5.46:10084:aiaa3201_p2/scripts/
#   ssh ... 'cd aiaa3201_p2 && bash scripts/run_bench.sh'
#
# Output: /tmp/slv176/bench/bench_report.md  (table for the paper) + per-run JSON.
# The directory is under /tmp, so copy it back before the node restarts.
cd /data/disk1/home/slv176/aiaa3201_p2 || exit 1
PY=/tmp/slv176/venv/bin/python
OUT=/tmp/slv176/bench
LOG=$OUT/run_bench.log
mkdir -p "$OUT"
: > "$LOG"

# VisA lives in /tmp during a session; fall back to the disk copy if it was moved.
DATA=/tmp/slv176/data/VisA_pytorch/1cls
[ -d "$DATA" ] || DATA=/data/disk1/home/slv176/data/VisA_pytorch/1cls
if [ ! -d "$DATA" ]; then
    echo "VisA not found in either location; edit DATA in this script" | tee -a "$LOG" >&2
    exit 1
fi

export DINOV2_HUB_DIR=/tmp/slv176/third_party/dinov2 PYTHONUNBUFFERED=1

echo "########## DINOv2 ViT-S/14 @448 L12 (no mask) ##########" >> "$LOG"
$PY scripts/bench_inference.py --backbone dinov2 --model dinov2_vits14 \
    --resolution 448 --layers 12 --data_root "$DATA" --n_per_cat 30 --passes 2 \
    --out "$OUT" >> "$LOG" 2>&1
echo "exit=$? dinov2" >> "$LOG"

echo "########## CLIP ViT-B/16 @448 L9 short_edge (E1 arm) ##########" >> "$LOG"
$PY scripts/bench_inference.py --backbone clip --model ViT-B/16 \
    --resolution 448 --layers 9 --clip_resize_mode short_edge \
    --clip_download_root /data/disk1/home/slv176/models/clip \
    --data_root "$DATA" --n_per_cat 30 --passes 2 --out "$OUT" >> "$LOG" 2>&1
echo "exit=$? clip" >> "$LOG"

$PY scripts/bench_inference.py --report "$OUT" >> "$LOG" 2>&1
echo "BENCH_DONE" >> "$LOG"
touch "$OUT/BENCH_DONE"
tail -40 "$LOG"
