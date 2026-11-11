#!/usr/bin/env bash
# Evaluate the six VisA categories of one grid configuration, split across four processes.
#
# Why: the scoring bottleneck is a 1-NN search of ~1k query patches against a 5k-45k patch
# bank. That search is *faster* on one thread than on twelve (thread sync dominates), so the
# cores are better spent on four independent categories. Measured per-category wall time drops
# by roughly 4-5x compared with a single multi-threaded process.
#
# Usage: bash scripts/run_grid_parallel.sh <name> <layers> <fusion> [extra args...]
set -u
NAME=${1:?name}; shift
LAYERS=${1:?layers}; shift
FUSION=${1:?fusion}; shift

PROJ=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-/tmp/slv176/venv/bin/python}
OUT=${OUT:-/tmp/slv176/grid}
LOGS=${LOGS:-/tmp/slv176/logs}
export DINOV2_HUB_DIR=${DINOV2_HUB_DIR:-/tmp/slv176/third_party/dinov2}
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$OUT" "$LOGS"

# category groups -> one process each. NB: do not call this GROUPS -- bash keeps a
# special read-only variable of that name (the user group IDs) and silently overrides it.
CAT_GROUPS=("candle" "capsules cashew" "chewinggum fryum" "pipe_fryum")
i=0
for g in "${CAT_GROUPS[@]}"; do
  i=$((i + 1))
  # shellcheck disable=SC2086
  ( cd "$PROJ" && "$PY" run_grid.py --name "${NAME}_p${i}" --layers $LAYERS --fusion "$FUSION" \
        --categories $g --out "$OUT" "$@" > "$LOGS/${NAME}_p${i}.log" 2>&1 ) &
done
wait

"$PY" "$PROJ/merge_grid.py" --out "$OUT/${NAME}_final.json" \
    "$OUT/${NAME}_p1.json" "$OUT/${NAME}_p2.json" "$OUT/${NAME}_p3.json" "$OUT/${NAME}_p4.json"
