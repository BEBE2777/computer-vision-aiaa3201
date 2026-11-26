# Error taxonomy, 3-way: DINOv2 vs CLIP vs Qwen (696-image subset)

Subset: the 696 VisA test images that carry Qwen v4 predictions (the 264 images used by the
earlier Qwen development pilots v1-v3 are excluded from E2); all three arms are scored on
identical images. Verdicts from each arm's own calibration threshold: 95th percentile of the
20 normal calibration scores, `numpy.percentile` linear interpolation -- the same rule as the
E1 visual taxonomy. Qwen anomaly score = `-fc_score` (sanity check: macro AUROC 0.9575
reproduces REPORT_V4 exactly).

## 3-way tally (696 images)

| cell | n | share |
|---|---:|---:|
| all three right | 554 | 79.6% |
| visual arms right, Qwen wrong | 30 | 4.3% |
| visual arms disagree | 39 | 5.6% |
| **only Qwen right** (both visual arms wrong) | **48** | **6.9%** |
| none right | 25 | 3.6% |

Qwen is right on 602/696 (86.5%). Where the two visual arms are both wrong (73 images),
Qwen recovers 48 (65.8%). Conversely, of the 623 images
at least one visual arm solves, Qwen misses only 30 outright. This is the countable
complementarity behind the exploratory equal-weight Qwen+DINOv2 fusion (97.68 macro AUROC on
this subset; post-hoc, unconfirmed -- see
`results/extensions/qwen/confirm_full_v4/fusion_qwen_dino_exploratory_v4.md`).

## Per category

| category | all right | visual only | visual split | qwen only | none |
|---|---:|---:|---:|---:|---:|
| candle | 127 | 7 | 14 | 3 | 5 |
| capsules | 76 | 9 | 7 | 19 | 5 |
| cashew | 71 | 3 | 12 | 13 | 7 |
| chewinggum | 97 | 2 | 3 | 2 | 2 |
| fryum | 92 | 2 | 1 | 6 | 5 |
| pipe_fryum | 91 | 7 | 2 | 5 | 1 |

Qwen thresholds (anomaly scale): candle 14.41, capsules 17.68, cashew 17.78, chewinggum 15.95, fryum 15.51, pipe_fryum 13.42.
Note: REPORT_V4 thresholds use a different finite-sample quantile estimator (index-19 max);
this table uses the E1 rule for cross-arm comparability, so per-class operating points differ
slightly (e.g. cashew 17.78 vs 22.09).

Per-image verdicts: `error_cases.csv` column `vlm_verdict` (right/wrong; 696 of 960 rows
filled, the rest are outside E2's evaluation scope). Raw scores:
`results/extensions/qwen/confirm_full_v4/predictions_v4_test.json`.
