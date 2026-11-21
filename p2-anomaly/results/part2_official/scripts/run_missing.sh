#!/bin/bash
# AIAA3201 P2 - fill in the 9 missing official AnomalyDINO cells.
# Runs as qzheng610, writes into a self-owned tree; reads slv176 assets read-only.
set -u
WORK=/tmp/qzheng610_p2
cd "$WORK" || exit 1
export DINOV2_HUB_DIR=/tmp/slv176/third_party/dinov2
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export HOME=/data/disk1/home/qzheng610

PY=/tmp/slv176/venv/bin/python
SC=/data/disk1/home/slv176/aiaa3201_p2/code/anomalydino/run_anomalydino.py
TAG=calib20r2
LOGD="$WORK/logs"
mkdir -p "$LOGD"

COMMON="--dataset VisA --model_name dinov2_vits14 --data_root /tmp/slv176/data/VisA_pytorch/1cls --preprocess agnostic --resolution 448 --calib_exclude 20 --eval_clf --no-eval_segm --faiss_on_cpu --warmup_iters 3 --refs_dump --tag $TAG"

stamp() { date '+%F %T'; }

runone() {
  name=$1; shift
  echo "[$(stamp)] START $name :: $*" >> "$LOGD/driver.log"
  $PY $SC $COMMON "$@" > "$LOGD/$name.log" 2>&1
  rc=$?
  echo "[$(stamp)] END   $name rc=$rc" >> "$LOGD/driver.log"
  df -h "$WORK" | tail -1 >> "$LOGD/driver.log"
}

echo "[$(stamp)] driver start user=$(id -un) tag=$TAG work=$WORK" >> "$LOGD/driver.log"

# 1) missing seed-0 cell, keep example plots as requested
runone part2_4shot_seed0 --shots 4 --num_seeds 1 --just_seed 0 --save_examples

# 2) all missing seed-1 cells, no example plots (save space)
runone part2_seed1 --shots 1 2 4 8 --num_seeds 1 --just_seed 1 --no-save_examples

# 3) all missing seed-2 cells
runone part2_seed2 --shots 1 2 4 8 --num_seeds 1 --just_seed 2 --no-save_examples

echo "[$(stamp)] ALL DONE" >> "$LOGD/driver.log"
touch "$LOGD/ALL_DONE"
