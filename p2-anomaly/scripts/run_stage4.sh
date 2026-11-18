#!/bin/bash
# Stage 4 artifacts that do not depend on the VLM branch.
# Waits for the E1 grid (run_e1_full.sh) to finish, then:
#   1. per-image scores for DINOv2@448 (no mask) and CLIP@448 block 9 (no mask)
#   2. one shared case list per category (so both backbones dump the SAME images)
#   3. score maps for those cases -> heatmap panel
#   4. two-backbone error taxonomy (visual half; the VLM column is filled in later)
#   5. main tables (Part 1 / Part 2 official / DINO no-mask / CLIP) + AUROC-vs-shots curves
cd /data/disk1/home/slv176/aiaa3201_p2 || exit 1
PY=/tmp/slv176/venv/bin/python
export OMP_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1 DINOV2_HUB_DIR=/tmp/slv176/third_party/dinov2
W=/tmp/slv176/grid
S=$W/scores
LOG=$W/stage4.log
mkdir -p "$S" "$W/figs"
: > "$LOG"

echo "waiting for E1_DONE at $(date)" >> "$LOG"
while [ ! -f "$W/E1_DONE" ]; do sleep 60; done
echo "E1 finished at $(date)" >> "$LOG"

$PY scripts/dump_scores.py --name e1_dino448_nomask --backbone dinov2 --resolution 448 \
    --layers 12 --no_mask --out "$S" >> "$LOG" 2>&1
echo "exit=$? dino_scores" >> "$LOG"
$PY scripts/dump_scores.py --name e1_clip448_L9_nomask --backbone clip --resolution 448 \
    --layers 9 --clip_resize_mode short_edge --no_mask --out "$S" >> "$LOG" 2>&1
echo "exit=$? clip_scores" >> "$LOG"

$PY scripts/pick_cases.py --a "$S/e1_dino448_nomask.json" --b "$S/e1_clip448_L9_nomask.json" \
    --out "$S/cases.json" --max-rows 6 >> "$LOG" 2>&1
echo "exit=$? pick_cases" >> "$LOG"

for spec in "dino:e1_dino448_nomask:dinov2:12:" "clip:e1_clip448_L9_nomask:clip:9:short_edge"; do
    tag=$(echo "$spec" | cut -d: -f1)
    name=$(echo "$spec" | cut -d: -f2)
    bb=$(echo "$spec" | cut -d: -f3)
    ly=$(echo "$spec" | cut -d: -f4)
    rm=$(echo "$spec" | cut -d: -f5)
    extra=""
    [ -n "$rm" ] && extra="--clip_resize_mode $rm"
    $PY scripts/dump_scores.py --name "$name" --backbone "$bb" --resolution 448 \
        --layers $ly --no_mask --out "$S" --cases "$S/cases.json" $extra >> "$LOG" 2>&1
    echo "exit=$? ${tag}_maps" >> "$LOG"
done

$PY scripts/make_heatmap_panel.py \
    --a "$S/e1_dino448_nomask.json" --a-name "DINOv2 ViT-S/14 @448" \
    --b "$S/e1_clip448_L9_nomask.json" --b-name "CLIP ViT-B/16 @448 (block 9)" \
    --out "$W/figs/heatmap_panel.png" --per-cat 1 >> "$LOG" 2>&1
echo "exit=$? heatmap" >> "$LOG"

$PY scripts/error_taxonomy.py \
    --a "$S/e1_dino448_nomask.json" --b "$S/e1_clip448_L9_nomask.json" \
    --a-name "DINOv2@448" --b-name "CLIP@448" \
    --out "$W/error_taxonomy.md" --csv "$W/error_cases.csv" >> "$LOG" 2>&1
echo "exit=$? taxonomy" >> "$LOG"

echo "########## tables ##########" >> "$LOG"
$PY scripts/build_part2_grid_json.py --src results/part2_official/agg/summary_448.json \
    --out "$W/part2_official_agg_final.json" >> "$LOG" 2>&1
echo "exit=$? build_part2" >> "$LOG"
$PY make_tables.py --part1 results/part1_template \
    --grid "AnomalyDINO (Part 2, official code, 448)=$W/part2_official_agg_final.json" \
           "DINOv2 ViT-S/14 448, no mask (E1 control)=$W/e1_dino448_nomask_final.json" \
           "CLIP ViT-B/16 448 block 9, no mask (Part 3)=$W/e1_clip448_L9_nomask_final.json" \
    --out "$W/tables" >> "$LOG" 2>&1
echo "exit=$? tables" >> "$LOG"
$PY make_figures.py --tables "$W/tables/metrics.json" --out "$W/figs" >> "$LOG" 2>&1
echo "exit=$? curves" >> "$LOG"

tar czf /tmp/slv176/stage4_artifacts.tgz -C "$W" tables figs error_taxonomy.md error_cases.csv \
    scores >> "$LOG" 2>&1
echo "exit=$? tar" >> "$LOG"

echo "ALL_STAGE4_DONE" >> "$LOG"
touch "$W/STAGE4_DONE"
