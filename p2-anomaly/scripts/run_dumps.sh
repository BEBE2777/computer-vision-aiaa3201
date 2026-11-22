#!/bin/bash
# Dump per-image scores for every (shot, seed) cell of the two E1 arms.
#
# Why: score_fusion.py needs per-image scores, and only 1-shot/seed-0 was dumped for the
# error taxonomy. With all 12 cells dumped the alpha sweep becomes a real 12-cell
# sensitivity table instead of a single pilot, and the fused-vs-single comparison can be
# averaged over shots and seeds like every other number in the paper.
#
# Everything here reads the fp32 feature cache, so there is no model forward pass:
# it is pure k-NN, a few minutes per cell, N_JOBS cells at a time (CPU only, the GPU is
# left to whoever else is on the node).
#
#   bash scripts/run_dumps.sh            # all 12 cells x 2 backbones, 4 at a time
#   N_JOBS=1 bash scripts/run_dumps.sh   # serial (if the node is busy)
#
# Output: /tmp/slv176/grid/scores/e1_{dinov2,clip}*_s{shot}_seed{seed}.json  (COPY BACK)

cd /data/disk1/home/slv176/aiaa3201_p2 || exit 1
PY=/tmp/slv176/venv/bin/python
OUT=/tmp/slv176/grid/scores
LOG=/tmp/slv176/logs/dumps
CACHE=/data/disk1/home/slv176/featcache32
DATA=/tmp/slv176/data/VisA_pytorch/1cls
[ -d "$DATA" ] || DATA=/data/disk1/home/slv176/data/VisA_pytorch/1cls
N_JOBS=${N_JOBS:-4}
ARM=${ARM:-both}          # both | dinov2 | clip
SHOTS=${SHOTS:-"1 2 4 8"} # subset for a quicker pass over fewer cells

mkdir -p "$OUT" "$LOG"
export DINOV2_HUB_DIR=/tmp/slv176/third_party/dinov2 PYTHONUNBUFFERED=1

dino() {  # $1 shot, $2 seed
    local name="e1_dino448_nomask_s$1_seed$2"
    $PY scripts/dump_scores.py --name "$name" --backbone dinov2 --layers 12 --no_mask \
        --data_root "$DATA" --cache "$CACHE" --out "$OUT" --shot "$1" --seed "$2" \
        > "$LOG/$name.log" 2>&1
    echo "done $name exit=$?"
}

clip() {  # $1 shot, $2 seed
    local name="e1_clip448_L9_nomask_s$1_seed$2"
    $PY scripts/dump_scores.py --name "$name" --backbone clip --layers 9 \
        --clip_resize_mode short_edge --clip_download_root /data/disk1/home/slv176/models/clip \
        --no_mask --data_root "$DATA" --cache "$CACHE" --out "$OUT" --shot "$1" --seed "$2" \
        > "$LOG/$name.log" 2>&1
    echo "done $name exit=$?"
}

n=0
if [ "$ARM" != "clip" ]; then
    for shot in $SHOTS; do
        for seed in 0 1 2; do
            # 1-shot/seed-0 was dumped earlier under a different name; skip the duplicate.
            if [ "$shot" != "1" ] || [ "$seed" != "0" ]; then
                dino "$shot" "$seed" &
                n=$((n + 1)); [ $((n % N_JOBS)) -eq 0 ] && wait
            fi
        done
    done
    wait
fi
if [ "$ARM" != "dinov2" ]; then
    for shot in $SHOTS; do
        for seed in 0 1 2; do
            if [ "$shot" != "1" ] || [ "$seed" != "0" ]; then
                clip "$shot" "$seed" &
                n=$((n + 1)); [ $((n % N_JOBS)) -eq 0 ] && wait
            fi
        done
    done
    wait
fi

ls -l "$OUT"/*.json | wc -l
echo DUMPS_DONE
