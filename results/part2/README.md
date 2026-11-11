# Part 2 — official AnomalyDINO results (all shot/seed combinations)

Method: **AnomalyDINO** with a frozen **DINOv2 ViT-S/14** backbone (k-NN patch scoring, k=1,
L2-normalised), reproducing the official code path. Dataset: **VisA** (1cls split), six
categories: `candle, capsules, cashew, chewinggum, fryum, pipe_fryum` — 960 test images per
(shots, seed) cell.

Fixed settings for every cell:

- `model_name=dinov2_vits14`, `resolution=448`, `preprocess=agnostic`
- `calib_exclude=20` (the 20 normal train images per category are held out for the
  95th-percentile OK/NG threshold only; they never enter the reference pool)
- `--eval_clf --no-eval_segm --faiss_on_cpu --warmup_iters 3`
- `--no-eval_segm` everywhere: the official segmentation/AUPRO evaluation needs ~11 GB RAM
  and was skipped on the shared GPU box.
- Pixel AUROC is computed **post hoc** from the stored patch-distance maps with the project's
  `eval_maps.py` protocol (map → bilinear resize to 448 → Gaussian σ=4; image score = mean of
  the top 1% of the raw map).
- Reference selection reuses the fixed per-seed slices (deterministic
  `pool[seed*n:(seed+1)*n]` after excluding the first 20 sorted normal train files); no
  re-sampling. The dumped `refs/*.json` here were verified to match that rule exactly.

## Results (Image AUROC / Pixel AUROC, %)

| shots | seed 0 | seed 1 | seed 2 | mean ± std (image) | mean ± std (pixel) |
|---:|---|---|---|---|---|
| 1 | 94.62 / 98.02 ¹ | 93.58 / 98.07 ² | 95.63 / 98.15 ² | 94.61 ± 0.84 | 98.08 ± 0.05 |
| 2 | 94.70 / 98.23 ¹ | 95.82 / 98.30 ² | 95.46 / 98.13 ² | 95.33 ± 0.46 | 98.22 ± 0.07 |
| 4 | 95.84 / 98.40 ² | 95.90 / 98.32 ² | 96.72 / 98.37 ² | 96.15 ± 0.40 | 98.36 ± 0.03 |
| 8 | 96.22 / 98.47 ¹ | 97.42 / 98.48 ² | 96.21 / 98.46 ² | 96.62 ± 0.57 | 98.47 ± 0.01 |

¹ **Pre-existing seed-0 results** (`1/2/8-shot`): already on the server under
`/tmp/slv176/results_VisA/dinov2_vits14_448/` (tags `calib20` / `xcheck`). Only re-scored
with `eval_maps.py` for the pixel column — numbers match the previously reported values
exactly; the original run outputs were not modified.

² **Newly run in this session** (2026-11-11, 20:23–22:34, serial on a shared A6000): the
missing `4-shot seed 0` plus all `seeds 1, 2` for `1/2/4/8-shot`, written with tag
`calib20r2` under `/tmp/qzheng610_p2/results_VisA/dinov2_vits14_448/` (the shared
`/tmp/slv176` is not writable by the student account used for the run). Each new cell ran
with `rc=0`, all 960 test images present, and `refs/` archived per run.

Cross-checks done: re-scoring the pre-existing seed-0 cells with the same evaluation scripts
reproduced the published numbers to two decimals; every new `refs/*.json` matches the
deterministic reference-slice rule; no run was killed or OOMed (peak GPU memory < 1 GB).

## What is in this directory

```
results/part2/
├── summary_all_shots_seeds.csv / .json   # 12 cells: per-shot/seed Image+Pixel AUROC,
│                                         # status (existing-seed0 / new), run_dir, per-category
├── eval448_<shot>shot_seed<k>.json      # per-cell pixel+image metrics from eval_maps.py
├── refs/<shot>shot_seed<k>/<obj>_shot=<n>_seed=<k>.json   # reference-image filename lists
├── args/<run_name>/args.yaml + preprocess.yaml            # exact run configurations
└── selected_heatmaps/*.png            # 7 representative anomaly-map montages
```

Deliberately **not** committed: VisA images or ground-truth masks, raw `*.npy` anomaly maps,
DINOv2 weights, the server venv, or run logs. Nothing in this directory contains credentials.

## Reproduction

On the GPU server (code at `/data/disk1/home/slv176/aiaa3201_p2`, venv
`/tmp/slv176/venv/bin/python`, data `/tmp/slv176/data/VisA_pytorch/1cls`):

```bash
export DINOV2_HUB_DIR=/tmp/slv176/third_party/dinov2
export MPLBACKEND=Agg OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1

# detection runs (one process at a time, shared GPU)
python /data/disk1/home/slv176/aiaa3201_p2/code/anomalydino/run_anomalydino.py \
  --dataset VisA --model_name dinov2_vits14 \
  --data_root /tmp/slv176/data/VisA_pytorch/1cls \
  --preprocess agnostic --resolution 448 \
  --shots 4 --num_seeds 1 --just_seed 0 \
  --calib_exclude 20 --eval_clf --no-eval_segm --faiss_on_cpu \
  --warmup_iters 3 --tag calib20 --save_examples

# pixel AUROC afterwards, from the stored patch-distance maps
python /data/disk1/home/slv176/aiaa3201_p2/eval_maps.py --run <run_dir> --out <out.json>
```

For the remaining cells swap in `--shots 1 2 4 8 --just_seed {1,2}` (same flags otherwise).
On a different host, `DINOV2_HUB_DIR` points at a local clone of the DINOv2 repo with the
`dinov2_vits14_pretrain.pth` checkpoint pre-fetched (see `p2-anomaly/scripts/fetch_dinov2.sh`).

## Part 3 status

**Part 3 has not produced experimental conclusions yet.** The cached-grid runner numbers are
not comparable to the official path (known scoring-protocol gap, see `HANDOFF.md`) and are
**not** included here. Nothing in this directory should be read as a Part 3 improvement.
