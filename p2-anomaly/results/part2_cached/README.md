# Cached-runner DINOv2 baseline (post-parity-fix)

`base_L12_v2_final.json` — six VisA categories × shots {1,2,4,8} × seeds {0,1,2},
DINOv2 ViT-S/14 @448, block 12, single-layer kNN, official agnostic protocol
(PCA foreground mask on queries, full reference bank).

## Why v2

The previous file (`base_L12_final.json`, now deleted) was produced by a cached
runner with three protocol defects. Full analysis in
`results/part2_official/parity_report.md`; the dominant one:

**cache-key collision** — VisA reuses image basenames across `test/good`,
`test/bad` and `train/good`. With basename-only cache keys, 8–10 bad test
images per category silently reused the features of the same-named good image,
and ~90 calibration images per category reused test-bad features. Capsules
1-shot image AUROC was depressed by ~5 points as a result.

The old numbers were therefore not comparable with the official baseline and
are not used anywhere.

## Verification

After the fixes (`run_grid.py` from commit `faad4ff`), the full grid was re-run
on a fresh float32 cache (`/data/disk1/home/slv176/featcache32`, server).

- 72/72 category-cells (12 shot/seed combos × 6 categories) match the archived
  official per-seed metrics (`eval448_*shot_seed*.json`) with **max abs
  difference 1.7e-09**
- Macro means reproduce the frozen official table exactly:

| shot | Image AUROC | Pixel AUROC |
|---:|---:|---:|
| 1 | 94.61 ± 0.84 | 98.08 ± 0.05 |
| 2 | 95.33 ± 0.46 | 98.22 ± 0.07 |
| 4 | 96.15 ± 0.40 | 98.36 ± 0.03 |
| 8 | 96.62 ± 0.57 | 98.47 ± 0.01 |

- runtime 8247 s on a shared A6000; `base_L12_v2.json` holds the per-cell
  intermediate file written during the run.

## Reproduction

```bash
python run_grid.py --name base_L12_v2 --layers 12 --fusion single \
  --cache /data/disk1/home/slv176/featcache32 --out /tmp/slv176/grid
```

Cache directories from before the fix must not be reused: keys changed
(`good_*`, `bad_*`, `train_*` prefixes) and the old float16 cache is lossy.
