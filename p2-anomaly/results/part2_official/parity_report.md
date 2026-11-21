# Parity report: official vs cached scoring path (Part 2, A line, phase 2)

Date: 2026-11-18 · Author: SibeiLYU (with agent assistance)
Scope: explain and close the gap between the official AnomalyDINO path
(`code/anomalydino/run_anomalydino.py`) and the cached re-implementation
(`run_grid.py` + `src/ad_core.py`), originally observed as
**cached 89.52 vs official 94.62 mean image AUROC** (1-shot, seed 0).

## Verdict

After the fixes below, the cached path reproduces the official path **exactly**
(all six VisA categories, image and pixel AUROC, 1-shot seed 0, agreement
< 1e-3, in practice bit-equal):

| category    | official img | cached img | official px | cached px |
|-------------|-------------|------------|-------------|-----------|
| candle      | 0.9036      | 0.9036     | 0.9857      | 0.9857    |
| capsules    | 0.9878      | 0.9878     | 0.9548      | 0.9548    |
| cashew      | 0.9290      | 0.9290     | 0.9887      | 0.9887    |
| chewinggum  | 0.9826      | 0.9826     | 0.9957      | 0.9957    |
| fryum       | 0.9452      | 0.9452     | 0.9658      | 0.9658    |
| pipe_fryum  | 0.9288      | 0.9288     | 0.9907      | 0.9907    |
| **MEAN**    | **0.9462**  | 0.9462     | **0.9802**  | 0.9802    |

Run: `parity32_all6` on the GPU server (fresh fp32 cache, fixed keys).
Official numbers from `results/part2/metrics/eval448_1shot_seed0.json`.

## Root causes found (ranked by impact)

### 1. Cache-key collision on duplicate basenames (the dominant bug)

VisA reuses file names across splits:

- test/good vs test/bad share **8–10 basenames per category** (capsules 8,
  cashew 10, chewinggum 9, fryum 10, pipe_fryum 10; candle 0 — which is why
  candle always matched);
- train/good vs test/bad share **~90 basenames per category**.

The cached runner keyed the feature cache by image basename only. Consequences:

- a bad test image silently reused the features of the same-named good test
  image → its anomaly score collapsed to a normal-like value (e.g. capsules
  `046.JPG` scored 0.387 with its own features, 0.143 with the collided ones);
  with 8/15 capsules anomalies affected, image AUROC fell by ~5 points;
- calibration images (train) reused test-bad features → wrong threshold /
  calibration statistics.

**Fix**: `run_grid.cache_key()` namespaces every cache entry by its parent
directory (`good_000.JPG`, `bad_011.JPG`, `train_...`). Old caches are not
compatible and must be re-extracted (delete the cache dir once).

### 2. Reference bank masking (systematic protocol deviation)

Official agnostic preprocessing keeps **all** reference patches in the memory
bank (`mask_ref_images=False`), while the cached runner used to PCA-mask
reference backgrounds too. Effect measured on capsules 1-shot seed 0:
-1.0 point image AUROC (0.9878 → 0.9775).

**Fix**: `extract_image(..., mask_query=...)` — references extracted with
`mask_query=False`, queries with `mask_query=True`, matching the official split.

### 3. float16 feature cache (lossy for scoring)

Storing features as fp16 shifts patch features by up to ~8e-3. On knife-edge
images this flips the nearest-neighbour members of the top-1% aggregate.
Measured in isolation with a fixed full bank and fixed masks, fp16 queries gave
capsules 0.9385 vs 0.9878 with fp32 — i.e. before fix #1 was understood this
effect was confounded with it; in reality **fp16 alone was not the 5-point
driver, the collision was**, but fp16 is still a real (smaller) perturbation.

**Fix**: cache stores float32.

### 4. Unseeded randomized PCA mask (process instability, minor)

`compute_background_mask` used `PCA(svd_solver='randomized')` without a
`random_state`; masks of borderline images could flip between processes (up to
~460/1536 patches on capsules). Under the *full* bank this turned out to be
harmless (AUROC invariant over 5 random draws), but it is a reproducibility
hazard.

**Fix**: `ad_core.deterministic_background_mask()` pins `random_state=0`.

## Hypotheses ruled out along the way

- Reference-image selection differs between paths — ruled out (identical).
- `mean_top1p`, distance normalization, top-1% aggregation differ — ruled out
  (bit-equal implementations).
- Fresh forward features unstable across processes — ruled out (bit-stable in
  two independent processes).
- Cached npz masks/feats corrupt — ruled out; the collision was keying, not
  corruption.

## Reusable probe

`scripts/parity_official_vs_cached.py` (GPU server) scores the same images
under the official protocol and the cached protocol and, with `--full`,
reproduces the archived per-category AUROC — rerun it after any change to
either path.

## Implications

- The frozen official Part 2 numbers (94.61 / 98.08 etc.) were always correct
  and remain the authority.
- The old cached grid results (`results/part2_cached/base_L12*.json`) are
  **invalidated** by the collision bug; any Part 3 variant must be re-run with
  the fixed cached path (fresh cache directory).
