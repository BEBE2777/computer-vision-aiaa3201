#!/bin/bash
# Run ONE v5-ext cell (single shots/seed), strictly gated:
#   flock mutex -> smoke manifest (1 good+1 bad per cat, refs verified vs
#   archive) -> smoke run -> smoke verify -> full test run -> verify ->
#   full calib run -> verify. Any failure aborts before the next step.
# Outputs go to /dev/shm (RAM disk): /data/disk2 was 100% full and killed a
# run at step 230/696 by failing to write the .partial (2026-11-25). Results
# are archived to $BASE/v2/ and downloaded after CELL OK.
# Usage: bash run_one_v5ext.sh SHOTS SEED
set -u
SHOTS=$1
SEED=$2
BASE=/data/disk2/home/qzheng610/qwen_work
OUTDIR=/dev/shm/v5ext_out
V=/tmp/slv176/venv/bin/python
export PYTHONPATH=$BASE/pylibs
cd $BASE/v2 || exit 1
mkdir -p $OUTDIR/logs
TAG="s${SEED}_${SHOTS}sh"
LOCK=/tmp/v5ext.lock

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "ABORT $TAG: another v5-ext cell holds the lock ($(date))"; exit 2
fi
echo "CELL START $TAG $(date) outdir=$OUTDIR shm_avail=$(df -h $OUTDIR | awk 'NR==2{print $4}')"

TM=manifest_v5_${TAG}_test.json
CM=manifest_v5_${TAG}_calib.json
[ -f "$TM" ] || { echo "MISSING $TM"; exit 1; }
[ -f "$CM" ] || { echo "MISSING $CM"; exit 1; }

# 0. carry over any earlier partial from $BASE/v2 so resume is real
for split in test calib; do
  if [ -f "$BASE/v2/predictions_v5_${TAG}_${split}.json.partial" ] && \
     [ ! -f "$OUTDIR/predictions_v5_${TAG}_${split}.json.partial" ]; then
    cp "$BASE/v2/predictions_v5_${TAG}_${split}.json.partial" \
       "$OUTDIR/predictions_v5_${TAG}_${split}.json.partial"
    echo "RESUME seed: carried $split partial ($(wc -c < "$OUTDIR/predictions_v5_${TAG}_${split}.json.partial") bytes)"
  fi
done

# 1. smoke manifest (1 good + 1 bad per category, refs verified vs archive)
$V build_smoke_v5ext.py --shots "$SHOTS" --seed "$SEED" || exit 1

# 2. smoke run (fresh, separate output)
$V run_v5.py --manifest manifest_v5_${TAG}_smoke.json \
     --out $OUTDIR/smoke_v5_${TAG}.json --fresh \
     > $OUTDIR/logs/smoke_${TAG}.log 2>&1 \
  || { echo "SMOKE RUN EXIT $TAG"; tail -5 $OUTDIR/logs/smoke_${TAG}.log; exit 1; }
$V verify_v5ext.py --manifest manifest_v5_${TAG}_smoke.json \
     --out $OUTDIR/smoke_v5_${TAG}.json \
  || { echo "SMOKE VERIFY FAIL $TAG"; exit 1; }

# 3. full test run (resume-capable)
$V run_v5.py --manifest $TM --out $OUTDIR/predictions_v5_${TAG}_test.json \
     > $OUTDIR/logs/test_${TAG}.log 2>&1 \
  || { echo "TEST RUN EXIT nonzero $TAG"; tail -5 $OUTDIR/logs/test_${TAG}.log; exit 1; }
$V verify_v5ext.py --manifest $TM --out $OUTDIR/predictions_v5_${TAG}_test.json \
  || { echo "TEST VERIFY FAIL $TAG"; exit 1; }

# 4. full calib run
$V run_v5.py --manifest $CM --out $OUTDIR/predictions_v5_${TAG}_calib.json \
     > $OUTDIR/logs/calib_${TAG}.log 2>&1 \
  || { echo "CALIB RUN EXIT nonzero $TAG"; tail -5 $OUTDIR/logs/calib_${TAG}.log; exit 1; }
$V verify_v5ext.py --manifest $CM --out $OUTDIR/predictions_v5_${TAG}_calib.json \
  || { echo "CALIB VERIFY FAIL $TAG"; exit 1; }

# 5. archive (best effort; disk2 may be full, results stay in /dev/shm meanwhile)
cp $OUTDIR/smoke_v5_${TAG}.json $OUTDIR/predictions_v5_${TAG}_test.json \
   $OUTDIR/predictions_v5_${TAG}_calib.json $BASE/v2/ 2>/dev/null \
   && echo "ARCHIVED $TAG to $BASE/v2/" || echo "WARN $TAG archive copy failed (disk2 full?)"

echo "CELL OK $TAG $(date)"
