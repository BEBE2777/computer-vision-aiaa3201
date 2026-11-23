# Score fusion — visual_pair_1shot_seed0

* branch 1: `dinov2`
* branch 2: `clip`
* images: 960 test images over 6 categories
* shot = 1, seed = 0 (single cell)
* main alpha (pre-registered) = 0.5

## alpha sweep (image AUROC, %)

| alpha | macro over categories | pooled |  (α=1 should equal dinov2, α=0 clip) |
|---:|---:|---:|---|
| 0.00 | 85.97 | 80.72 | single-branch check |
| 0.25 | 91.61 | 87.05 |  |
| 0.50 | 93.75 | 90.07 |  |
| 0.75 | 94.05 | 91.09 |  |
| 1.00 | 93.11 | 90.78 | single-branch check |

## per-category image AUROC (%) at alpha = 0.5

| category | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 |
|---|---:|---:|---:|---:|---:|
| candle | 87.90 | 90.79 | 91.45 | 91.27 | 90.25 |
| capsules | 80.02 | 92.40 | 95.67 | 96.02 | 95.93 |
| cashew | 83.98 | 89.24 | 91.92 | 92.28 | 90.62 |
| chewinggum | 94.28 | 96.38 | 97.58 | 97.90 | 97.96 |
| fryum | 74.42 | 84.70 | 89.84 | 91.96 | 92.32 |
| pipe_fryum | 95.24 | 96.16 | 96.02 | 94.86 | 91.60 |
| **macro** | 85.97 | 91.61 | 93.75 | 94.05 | 93.11 |

## decision at the fused calibration threshold (alpha = 0.5, 95th percentile of calibration scores)

| TP | FP | TN | FN | recall | FPR |
|---:|---:|---:|---:|---:|---:|
| 490 | 63 | 297 | 110 | 81.7% | 17.5% |

Operating point of every alpha (threshold = 95th percentile of that alpha's fused *calibration* score, per category; test labels only count the outcome):

| alpha | TP | FP | TN | FN | recall | FPR |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 356 | 51 | 309 | 244 | 59.3% | 14.2% |
| 0.25 | 441 | 61 | 299 | 159 | 73.5% | 16.9% |
| 0.50 | 490 | 63 | 297 | 110 | 81.7% | 17.5% |
| 0.75 | 497 | 50 | 310 | 103 | 82.8% | 13.9% |
| 1.00 | 474 | 34 | 326 | 126 | 79.0% | 9.4% |

At the nominal 95th percentile the *target* false-alarm rate is 5% of the normal test images; the realised rates above are higher (single branches included), i.e. 20 training-normal calibration images do not cover the variation of the normal test images. Report these as achieved operating points, not as a calibrated 5% guarantee.

Per category: candle TP/FP/FN = 93/35/7, capsules TP/FP/FN = 67/3/33, cashew TP/FP/FN = 60/1/40, chewinggum TP/FP/FN = 95/6/5, fryum TP/FP/FN = 79/7/21, pipe_fryum TP/FP/FN = 96/11/4

Calibration scale per category (rule fired / median / sigma): candle b1=mad(0.212, 0.0240) b2=mad(0.331, 0.0176); capsules b1=mad(0.169, 0.0144) b2=mad(0.378, 0.0436); cashew b1=mad(0.212, 0.0320) b2=mad(0.295, 0.0222); chewinggum b1=mad(0.178, 0.0199) b2=mad(0.324, 0.0148); fryum b1=mad(0.174, 0.0254) b2=mad(0.309, 0.0375); pipe_fryum b1=mad(0.200, 0.0481) b2=mad(0.289, 0.0247)

Note: alpha is *reported*, not selected — the main column is the pre-registered 0.5, and the sweep is a sensitivity analysis (sections 3.2 and E4 of the plan).
