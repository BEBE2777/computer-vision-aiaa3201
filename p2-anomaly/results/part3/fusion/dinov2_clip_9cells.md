# Two-arm score fusion — alpha sweep over 9 cells

Both arms are the E1 configuration (DINOv2 ViT-S/14 @448 block 12 and CLIP ViT-B/16 @448 block 9, no foreground mask), fused after per-category median/MAD normalisation on the 20 held-out normal calibration images. alpha = 0.5 was fixed before any result was seen.

Cells: shots [1, 2, 4] x seeds [0, 1, 2] (9 of 9).

## Six-category image AUROC (%) across cells

| alpha | mean | std | min | max | note |
|---:|---:|---:|---:|---:|---|
| 0.00 | 88.73 | 1.12 | 85.97 | 89.55 | = CLIP alone |
| 0.25 | 93.40 | 0.78 | 91.61 | 94.23 | -- |
| 0.50 | 94.82 | 0.59 | 93.75 | 95.46 | pre-registered |
| 0.75 | 94.88 | 0.62 | 93.89 | 95.79 | -- |
| 1.00 | 94.14 | 0.88 | 92.59 | 95.48 | = DINOv2 alone |

Fused (alpha=0.5) minus the better single arm: **+0.68** points (per-cell gains: +0.63, +1.59, +0.45, +1.24, +0.53, +0.19, +1.17, +0.39, -0.04).

## Achieved operating point (threshold = 95th percentile of that alpha's fused calibration scores, per category)

| alpha | recall | FPR |
|---:|---:|---:|
| 0.00 | 68.2% | 14.3% |
| 0.25 | 79.1% | 15.6% |
| 0.50 | 82.9% | 15.7% |
| 0.75 | 82.9% | 15.2% |
| 1.00 | 80.4% | 12.8% |

The nominal target at the 95th percentile is a 5% false-alarm rate; the realised rates are higher for every arm and every alpha, i.e. 20 normal calibration images per category do not cover the variation of the normal test images. Quote these as achieved operating points, not as a calibrated 5% guarantee.

## Per cell (six-category macro AUROC, %)

| shot | seed | a=0.00 | a=0.25 | a=0.50 | a=0.75 | a=1.00 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 85.97 | 91.61 | 93.75 | 94.05 | 93.11 |
| 1 | 1 | 88.30 | 93.04 | 94.18 | 93.89 | 92.59 |
| 1 | 2 | 88.81 | 93.36 | 94.77 | 94.99 | 94.31 |
| 2 | 0 | 88.81 | 93.55 | 95.02 | 94.72 | 93.78 |
| 2 | 1 | 89.46 | 93.93 | 94.91 | 94.98 | 94.38 |
| 2 | 2 | 89.39 | 93.15 | 94.52 | 94.82 | 94.33 |
| 4 | 0 | 89.52 | 94.23 | 95.46 | 95.17 | 94.30 |
| 4 | 1 | 89.55 | 93.92 | 95.37 | 95.55 | 94.98 |
| 4 | 2 | 88.74 | 93.82 | 95.44 | 95.79 | 95.48 |
