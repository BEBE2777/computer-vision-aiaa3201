# E1 full grid — CLIP vs DINOv2 under the frozen protocol

Stage 3 / step 4 (E1 full grid) and stage 4 (tables, taxonomy, heatmaps) of
`docs/part3/实验规划与两人分工.md`. The frozen configuration is the one recorded in
`e1_pilot.md` §3: CLIP ViT-B/16 @448, block 9, `short_edge` framing, no foreground mask;
control arm DINOv2 ViT-S/14 @448, block 12, same no-mask protocol. 6 categories × 4 shots ×
3 seeds = 72 cells per arm; every number below is mean ± std over the three seeds of the
six-category mean, computed by the single aggregation path in `p2common.summarize`.

Raw grid JSONs: `scores/e1_{clip448_L9,dino448}_nomask_final.json`; per-image score dumps of the
1-shot seed-0 cell: `scores/e1_*_nomask.json`.

## 1. Headline

| Method | 1-shot | 2-shot | 4-shot | 8-shot |
|---|---|---|---|---|
| Template difference (Part 1) | 52.00 ± 1.13 | 54.08 ± 3.38 | 55.36 ± 3.22 | 57.96 ± 0.47 |
| AnomalyDINO (Part 2, official, masked) | **94.61 ± 0.84** | **95.33 ± 0.46** | **96.15 ± 0.40** | **96.62 ± 0.57** |
| DINOv2 @448, no mask (E1 control) | 93.34 ± 0.72 | 94.16 ± 0.27 | 94.92 ± 0.49 | 95.37 ± 0.72 |
| CLIP @448 block 9, no mask (Part 3) | 87.70 ± 1.24 | 89.22 ± 0.29 | 89.27 ± 0.38 | 89.48 ± 0.23 |

Image-level AUROC (%). Pixel-level AUROC: 98.08 / 98.22 / 98.36 / 98.47 (official),
97.06 / 97.57 / 97.91 / 98.16 (DINOv2 no-mask control), 94.44 / 94.97 / 95.22 / 95.46 (CLIP);
Part 1 is 65.06 → 66.14. Full per-category tables: `tables.md`, curves: `figs/auroc_vs_shots.png`.

Read-outs:

* Under the *controlled* protocol (identical scoring, framing, resolution, no mask), CLIP is
  **5.6 image-AUROC points behind DINOv2 at 1-shot** (87.70 vs 93.34) and 2.6 pixel points behind
  (94.44 vs 97.06). The official masked baseline adds another ~1.3 image points on top of the
  control, so the PCA foreground mask is worth ~1.3 points for DINOv2 and was removed from both
  arms only to keep the backbone the single variable.
* **The gap does not close with more references.** CLIP gains +1.8 image points from 1-shot to
  8-shot; DINOv2 gains +2.0 over the same range and stays ahead at every shot count. Few-shot
  deficit is a property of the features, not of sample size.
* Per category the deficit is uneven (image AUROC, 1-shot, CLIP − control): capsules −15.1,
  fryum −10.3, candle −4.7, cashew −4.2, chewinggum −2.6, **pipe_fryum +3.2**. CLIP beats
  DINOv2 on pipe_fryum at 1- and 2-shot and is within the seed spread at 4/8-shot, and it ties
  DINOv2 on pixel AUROC for fryum (+0.1) despite being 10 points worse on fryum image AUROC —
  its failures there are score-localisation failures, not ranking failures. The one category
  where CLIP is ahead hints at complementary information, which is what the fusion experiments
  (E3/E4) and the VLM branch are for.

## 2. Error taxonomy (1-shot, seed 0, thresholds from the 20 calibration images)

Verdict per test image from each arm's own calibration threshold; the two verdicts are crossed:

| category | n | both right | only DINOv2 right | only CLIP right | both wrong |
|---|---|---|---|---|---|
| candle | 200 | 141 | 29 | 18 | 12 |
| capsules | 160 | 70 | 57 | 6 | 27 |
| cashew | 150 | 83 | 27 | 7 | 33 |
| chewinggum | 150 | 127 | 14 | 4 | 5 |
| fryum | 150 | 71 | 56 | 6 | 17 |
| pipe_fryum | 150 | 115 | 10 | 17 | 8 |
| **all six** | 960 | 607 (63.2%) | 193 (20.1%) | 58 (6.0%) | 102 (10.6%) |

* Defects missed by both arms: 90 of 600; normal images flagged by both: 12 of 360.
* CLIP is right on **58 images where DINOv2 is wrong**; 35 of those 58 are on candle and
  pipe_fryum, the two categories where CLIP's per-category deficit is smallest or reversed. That
  is the quantitative version of the complementarity argument: a fusion of the two score sets
  starts from a 63.2% agreement base and can only gain, never lose, the 6.0% "only CLIP" cell.
* The per-image CSV (`error_cases.csv`) carries an empty `vlm_verdict` column; once the Qwen
  branch produces per-image scores, the fourth cell ("only the VLM is right") completes the
  taxonomy without recomputing anything.

## 3. Qualitative: where the two encoders disagree

`figs/heatmap_panel.png` scores the **same** test images (one shared both-miss case per
category, chosen by `scripts/pick_cases.py`) with both arms, next to the ground-truth mask; each
map is labelled with its top-1 % score and that arm's calibration threshold.

The shared failure mode is consistent: the two arms miss the same small defects (all six panel
rows are defects both arms score below their threshold), and the maps show why. DINOv2's
distance field is near-flat over the background with a few isolated hotspots, while CLIP's
block-9 map is high-variance everywhere — large areas of the normal background sit close to the
CLIP score threshold. Two consequences follow, both visible in the numbers: CLIP needs a higher
threshold to keep its false-alarm rate at the same operating point (e.g. candle 0.355 vs DINOv2
0.269), and tiny defects drown in the background variance of the CLIP map, which is exactly the
capsules (−15.1 image) and fryum (−10.3 image) regime.

## 4. Reproduce

```bash
bash scripts/run_e1_full.sh     # CLIP grid, then the DINOv2 control; writes E1_DONE
bash scripts/run_stage4.sh      # waits for E1_DONE: per-image scores, cases, heatmaps,
                                # taxonomy, tables, curves
python scripts/build_part2_grid_json.py   # official Part 2 aggregate -> grid layout
python make_tables.py --part1 results/part1_template --grid ...
python make_figures.py --tables results/part3/metrics.json --out results/part3/figs
```

Runtime on the shared A6000 (GPU mostly idle, kNN is CPU-bound): CLIP grid 265.6 min, DINOv2
control ~4 h, stage 4 ~27 min.
