# Qwen + DINO score fusion — v4 post-hoc exploration

> This analysis was initiated after inspecting the v4 single-branch results. It is exploratory, not a preregistered confirmatory result. No weights were selected from test labels; the full alpha sweep is shown.

- Test images: 696, exactly aligned by category, label, and image ID.
- Qwen score: frozen v4 `fc_conf = -fc_score`; DINO score: E1 DINOv2 @448, no mask, 1-shot/seed0.
- Each branch is normalized independently per category using the 20 normal calibration scores: median and 1.4826×MAD, with the existing `score_fusion.py` fallback rule.
- `alpha` is the Qwen share; `alpha=0` is DINO alone and `alpha=1` is Qwen alone.
- Operating threshold for each fused score is the linear-interpolated 95th percentile of its fused calibration scores. These operating points are exploratory.

## Results

| Qwen weight α | Macro image AUROC | Pooled image AUROC | Recall | FPR |
|---:|---:|---:|---:|---:|
| 0.00 | 0.9313 | 0.9103 | 0.795 | 0.101 |
| 0.25 | 0.9727 | 0.9436 | 0.870 | 0.083 |
| 0.50 | 0.9768 | 0.9440 | 0.912 | 0.175 |
| 0.75 | 0.9705 | 0.9343 | 0.906 | 0.189 |
| 1.00 | 0.9575 | 0.9214 | 0.891 | 0.189 |

## Per-category macro AUROC

| Category | α=0.00 | α=0.25 | α=0.50 | α=0.75 | α=1.00 |
|---|---:|---:|---:|---:|---:|
| candle | 0.9014 | 0.9464 | 0.9479 | 0.9451 | 0.9426 |
| capsules | 0.9497 | 0.9713 | 0.9663 | 0.9521 | 0.9200 |
| cashew | 0.9231 | 0.9835 | 0.9895 | 0.9684 | 0.9318 |
| chewinggum | 0.9799 | 0.9886 | 0.9918 | 0.9899 | 0.9867 |
| fryum | 0.9203 | 0.9556 | 0.9707 | 0.9730 | 0.9716 |
| pipe_fryum | 0.9135 | 0.9908 | 0.9945 | 0.9945 | 0.9922 |

Endpoint identity check: max AUROC drift from the corresponding single branch = 0.00e+00.
At alpha=1, the AUROC matches the v4 Qwen endpoint, but its displayed operating point uses the fusion script's linear-interpolated calibration percentile; it therefore differs from the v4 frozen max-order-statistic threshold.

Interpretation is limited to this fixed 696-image subset and 1-shot/seed0. Do not present the best alpha as a selected or independently validated model; the sweep is a sensitivity analysis after the single-branch results were already observed.
