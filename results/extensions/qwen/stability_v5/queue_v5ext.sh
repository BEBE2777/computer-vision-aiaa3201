#!/bin/bash
# v5 extension queue: 2/4/8-shot x seed 0/1/2 (9 configs), strictly sequential.
# Per qwen_stability_v5.md section 3.3 (pre-declared, gate passed: 1-shot
# seeds 0/1/2 complete and reviewed, commit 6c1e86f).
# Smoke (12 test imgs, --fresh) before each full run; abort queue on smoke fail.
# Full runs are resume-capable (no --fresh): .partial merges by category/query_id.
BASE=/data/disk2/home/qzheng610/qwen_work
V=/tmp/slv176/venv/bin/python
export PYTHONPATH=$BASE/pylibs
cd $BASE/v2 || exit 1
mkdir -p logs_v5ext

# safety: refuse to start if any run_v5 process is alive (OOM lesson from v5)
if ps aux | grep '[r]un_v5.py' > /dev/null 2>&1; then
  echo "ABORT: run_v5.py already running"; exit 1
fi

echo "QUEUE START $(date)"
for SHOTS in 2 4 8; do
  for SEED in 0 1 2; do
    TAG="s${SEED}_${SHOTS}sh"
    TM=manifest_v5_${TAG}_test.json
    CM=manifest_v5_${TAG}_calib.json
    [ -f "$TM" ] || { echo "MISSING $TM"; exit 1; }
    [ -f "$CM" ] || { echo "MISSING $CM"; exit 1; }
    if [ -f predictions_v5_${TAG}_test.json ] && [ -f predictions_v5_${TAG}_calib.json ]; then
      echo "SKIP $TAG (finals exist)"; continue
    fi
    # smoke: first 12 test entries, fresh, separate output file
    $V run_v5.py --manifest $TM --out smoke_v5_${TAG}.json --limit 12 --fresh \
        > logs_v5ext/smoke_${TAG}.log 2>&1
    n_ok=$($V - <<PYEOF
import json
try:
    d = json.load(open('smoke_v5_${TAG}.json'))['records']
    print(sum(1 for r in d if r.get('status') == 'ok'))
except Exception:
    print(0)
PYEOF
)
    if [ "$n_ok" != "12" ]; then
      echo "SMOKE FAIL $TAG ($n_ok/12) $(date)"
      tail -5 logs_v5ext/smoke_${TAG}.log
      exit 1
    fi
    echo "SMOKE OK $TAG $(date)"
    # full runs
    $V run_v5.py --manifest $TM --out predictions_v5_${TAG}_test.json \
        > logs_v5ext/test_${TAG}.log 2>&1
    echo "TEST DONE $TAG $(date)"
    $V run_v5.py --manifest $CM --out predictions_v5_${TAG}_calib.json \
        > logs_v5ext/calib_${TAG}.log 2>&1
    echo "CALIB DONE $TAG $(date)"
  done
done
echo "QUEUE DONE $(date)"
