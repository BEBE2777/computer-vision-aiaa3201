#!/usr/bin/env bash
# Part 2 -- reproduction of the official AnomalyDINO (DINOv2 ViT-S/14, 448 px, agnostic
# preprocessing). Six VisA categories, shots {1,2,4,8}, seeds {0,1,2}, pixel-level metrics on.
#
# Usage: bash scripts/run_part2.sh [TAG] [EXTRA_ARGS...]
#   TAG defaults to "calib20"
set -u
TAG="${1:-calib20}"
shift || true

REPO="$HOME/aiaa3201_p2"
VENV=/tmp/slv176/venv
export DINOV2_HUB_DIR=/tmp/slv176/third_party/dinov2
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

cd "$REPO/code/anomalydino"
ln -sfn /tmp/slv176/results_VisA results_VisA

mkdir -p /tmp/slv176/logs
LOG=/tmp/slv176/logs/part2_${TAG}.log
echo "logging to $LOG"

nohup "$VENV/bin/python" run_anomalydino.py \
  --dataset VisA \
  --model_name dinov2_vits14 \
  --data_root /tmp/slv176/data/VisA_pytorch/1cls \
  --preprocess agnostic \
  --resolution 448 \
  --shots 1 2 4 8 \
  --num_seeds 3 \
  --calib_exclude 20 \
  --eval_clf --no-eval_segm \
  --faiss_on_cpu \
  --warmup_iters 5 \
  --tag "$TAG" "$@" > "$LOG" 2>&1 < /dev/null &

echo "pid=$!"
