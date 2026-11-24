# v4 statistical review addendum

This post-hoc addendum preserves all frozen predictions, directions, thresholds, and point estimates. AUROC uncertainty is recomputed with label-stratified paired bootstrap; FPR/recall use exact two-sided Clopper–Pearson intervals.

Bootstrap: 20,000 resamples, seed 20261125; within each category, good and bad images are resampled separately, and the same resampled images are paired across methods.

## Macro AUROC and paired differences

| Comparison | Point estimate | 95% interval |
|---|---:|---:|
| QWEN macro AUROC | 0.9575 | [0.9423, 0.9711] |
| DINO macro AUROC | 0.9313 | [0.9109, 0.9505] |
| CLIP macro AUROC | 0.8530 | [0.8225, 0.8826] |
| QWEN MINUS DINO macro AUROC difference | +0.0262 | [+0.0035, +0.0493] |
| QWEN MINUS CLIP macro AUROC difference | +0.1045 | [+0.0731, +0.1370] |

The paired intervals estimate differences on this fixed 696-image subset and 1-shot/seed0 configuration. They do not measure variation across shots, reference seeds, or datasets.

## Threshold operating-point intervals

| Category | FPR (FP/n, exact 95% CI) | Recall (TP/n, exact 95% CI) |
|---|---|---|
| candle | 0.141 (11/78; [0.073, 0.238]) | 0.923 (72/78; [0.840, 0.971]) |
| capsules | 0.053 (2/38; [0.006, 0.177]) | 0.846 (66/78; [0.747, 0.918]) |
| cashew | 0.000 (0/28; [0.000, 0.123]) | 0.256 (20/78; [0.164, 0.368]) |
| chewinggum | 0.000 (0/28; [0.000, 0.123]) | 0.859 (67/78; [0.762, 0.927]) |
| fryum | 0.000 (0/28; [0.000, 0.123]) | 0.859 (67/78; [0.762, 0.927]) |
| pipe_fryum | 0.036 (1/28; [0.001, 0.183]) | 0.949 (74/78; [0.874, 0.986]) |

For zero observed false positives, the empirical bootstrap interval is degenerate at [0,0]. The exact binomial interval is used here to retain finite-sample uncertainty.

## Threshold convention

Qwen v4 implementation selects sorted calibration score at index int(20*0.95)=19 (the maximum); E1 DINO/CLIP uses numpy.percentile(...,95) linear interpolation. Calibration images match, percentile estimators differ.
Because the finite-sample estimators differ, describe the observed operating points separately and do not call them estimator-matched FPR comparisons.
