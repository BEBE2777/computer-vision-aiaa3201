# Part 2 — official AnomalyDINO results (PRELIMINARY, seed 0)

Source: `p2-anomaly/results/tables.md` (committed). Official AnomalyDINO
pipeline, `dinov2_vits14`, 448 px, `agnostic` preprocessing, VisA six
categories (960 test images, 1cls split). Per category, 20 normal training
images are reserved for the 95th-percentile OK/NG threshold only and never
enter the reference pool.

## Main table — Image / Pixel AUROC (%), seed 0 only

| shots | Image AUROC | Pixel AUROC | Status |
|---|---|---|---|
| 1 | 94.62 | 98.02 | done (seed 0) |
| 2 | 94.70 | 98.23 | done (seed 0) |
| 8 | 96.22 | 98.47 | done (seed 0) |

> **These are preliminary single-seed numbers.** The complete multi-seed
> stability evaluation (seeds 0/1/2) will be completed later.

## Per-category Image AUROC (seed 0)

| shots | candle | capsules | cashew | chewinggum | fryum | pipe_fryum | MEAN |
|---|---|---|---|---|---|---|---|
| 1 | 90.4 | 98.8 | 92.9 | 98.3 | 94.5 | 92.9 | 94.62 |
| 2 | 90.6 | 98.6 | 93.5 | 98.7 | 96.1 | 90.7 | 94.70 |
| 8 | 93.1 | 97.4 | 95.0 | 98.9 | 96.9 | 96.1 | 96.22 |

## Configurations not yet completed

| configuration | status |
|---|---|
| 4-shot (any seed) | **not run yet** |
| 2-shot / 8-shot, seeds 1 and 2 | **not run yet** |
| 1-shot, seeds 1 and 2 | **not run yet** |

One-line explanation: the full multi-seed stability evaluation (all shots ×
seeds 0/1/2) will be completed in follow-up runs before the final report.

## Related but separate

The cached grid runner (`run_grid.py`) reports 89.52 → 92.11 image AUROC for
1 → 8 shots (3 seeds), but its scoring step still differs from the official
pipeline, so those numbers are **not** cross-validated yet and must not be
presented as Part 2 results. Reconciliation is tracked in `HANDOFF.md`.
