#!/bin/bash
# Stage 2: the --no-save_examples path crashes in create_sample_plots (missing per-object dir),
# so use the fork's default (save_examples=True), identical to the 4-shot seed-0 run.
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
rm -f "$LOGD/ALL_DONE"

COMMON="--dataset VisA --model_name dinov2_vits14 --data_root /tmp/slv176/data/VisA_pytorch/1cls --preprocess agnostic --resolution 448 --calib_exclude 20 --eval_clf --no-eval_segm --faiss_on_cpu --warmup_iters 3 --refs_dump --tag $TAG --save_examples"

stamp() { date '+%F %T'; }

runone() {
  name=$1; shift
  echo "[$(stamp)] START $name :: $*" >> "$LOGD/driver.log"
  $PY $SC $COMMON "$@" > "$LOGD/$name.log" 2>&1
  rc=$?
  echo "[$(stamp)] END   $name rc=$rc" >> "$LOGD/driver.log"
  df -h "$WORK" | tail -1 >> "$LOGD/driver.log"
}

echo "[$(stamp)] stage2 start (save_examples=True, 1-shot seeds already done)" >> "$LOGD/driver.log"
runone part2_seed1 --shots 1 2 4 8 --num_seeds 1 --just_seed 1
runone part2_seed2 --shots 1 2 4 8 --num_seeds 1 --just_seed 2
echo "[$(stamp)] ALL DONE" >> "$LOGD/driver.log"
touch "$LOGD/ALL_DONE"
