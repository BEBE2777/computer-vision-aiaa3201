#!/bin/bash
# Run ONE v5-ext cell (single shots/seed), strictly gated:
#   flock mutex -> smoke manifest (1 good+1 bad per cat, refs verified vs
#   archive) -> smoke run -> smoke verify -> full test run -> verify ->
#   full calib run -> verify. Any failure aborts before the next step.
# Usage: bash run_one_v5ext.sh SHOTS SEED
set -u
SHOTS=$1
SEED=$2
BASE=/data/disk2/home/qzheng610/qwen_work
V=/tmp/slv176/venv/bin/python
export PYTHONPATH=$BASE/pylibs
cd $BASE/v2 || exit 1
mkdir -p logs_v5ext
TAG="s${SEED}_${SHOTS}sh"
LOCK=/tmp/v5ext.lock

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "ABORT $TAG: another v5-ext cell holds the lock ($(date))"; exit 2
fi
echo "CELL START $TAG $(date)"

TM=manifest_v5_${TAG}_test.json
CM=manifest_v5_${TAG}_calib.json
[ -f "$TM" ] || { echo "MISSING $TM"; exit 1; }
[ -f "$CM" ] || { echo "MISSING $CM"; exit 1; }

# 1. smoke manifest (1 good + 1 bad per category, refs verified vs archive)
$V build_smoke_v5ext.py --shots "$SHOTS" --seed "$SEED" || exit 1

# 2. smoke run (fresh, separate output)
$V run_v5.py --manifest manifest_v5_${TAG}_smoke.json \
     --out smoke_v5_${TAG}.json --fresh > logs_v5ext/smoke_${TAG}.log 2>&1 \
  || { echo "SMOKE RUN EXIT $TAG"; tail -5 logs_v5ext/smoke_${TAG}.log; exit 1; }
$V verify_v5ext.py --manifest manifest_v5_${TAG}_smoke.json --out smoke_v5_${TAG}.json \
  || { echo "SMOKE VERIFY FAIL $TAG"; exit 1; }

# 3. full test run (resume-capable)
$V run_v5.py --manifest $TM --out predictions_v5_${TAG}_test.json \
     > logs_v5ext/test_${TAG}.log 2>&1 \
  || { echo "TEST RUN EXIT nonzero $TAG"; tail -5 logs_v5ext/test_${TAG}.log; exit 1; }
$V verify_v5ext.py --manifest $TM --out predictions_v5_${TAG}_test.json \
  || { echo "TEST VERIFY FAIL $TAG"; exit 1; }

# 4. full calib run
$V run_v5.py --manifest $CM --out predictions_v5_${TAG}_calib.json \
     > logs_v5ext/calib_${TAG}.log 2>&1 \
  || { echo "CALIB RUN EXIT nonzero $TAG"; tail -5 logs_v5ext/calib_${TAG}.log; exit 1; }
$V verify_v5ext.py --manifest $CM --out predictions_v5_${TAG}_calib.json \
  || { echo "CALIB VERIFY FAIL $TAG"; exit 1; }

echo "CELL OK $TAG $(date)"
