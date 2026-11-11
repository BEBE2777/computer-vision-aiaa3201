# AIAA3201 Project 2 - result snapshot (2026-11-11)

AUROC in %. VisA six categories, 1cls split, 960 test images, 448px input,
agnostic preprocessing. Per category, 20 normal train images are reserved for the
95th-percentile OK/NG threshold only and never enter the reference pool.

Part 1 and Part 2 share the same reference sets (archived per shot/seed in each
run's refs/ folder).

## Part 1 - template difference baseline (3 seeds, mean of 6 categories)

| shots | baseline Image | baseline Pixel | +fg-crop Image | +fg-crop Pixel |
|---|---|---|---|---|
| 1-shot | 52.0 | 65.1 | 55.3 | 64.8 |
| 2-shot | 54.1 | 64.9 | 55.0 | 65.7 |
| 4-shot | 55.4 | 65.5 | 55.6 | 65.8 |
| 8-shot | 58.0 | 66.1 | 57.0 | 66.1 |

## Part 2 - AnomalyDINO official path (dinov2_vits14, 448, agnostic, seed 0)

| shots | Image AUROC | Pixel AUROC |
|---|---|---|
| 1 | 94.62 | 98.02 |
| 2 | 94.70 | 98.23 |
| 8 | 96.22 | 98.47 |

## Part 2 - cached grid runner (3 seeds; scoring differs from official, see HANDOFF)

| shots | Image | Pixel |
|---|---|---|
| 1 | 89.52 | 96.28 |
| 2 | 90.25 | 96.41 |
| 4 | 91.51 | 96.57 |
| 8 | 92.11 | 96.70 |

## Part 2 official, per-category image AUROC (seed 0)

| shots | candle | capsules | cashew | chewinggum | fryum | pipe_fryum | MEAN |
|---|---|---|---|---|---|---|---|
| 1 | 90.4 | 98.8 | 92.9 | 98.3 | 94.5 | 92.9 | 94.62 |
| 2 | 90.6 | 98.6 | 93.5 | 98.7 | 96.1 | 90.7 | 94.70 |
| 8 | 93.1 | 97.4 | 95.0 | 98.9 | 96.9 | 96.1 | 96.22 |

Files: part1_template/ part1_crop/ (summary.json = per-seed values),
part2_official/eval448_*shot.json (image+pixel, per category), part2_cached/.
