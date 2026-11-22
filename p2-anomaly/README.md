# Few-Shot Industrial Visual Anomaly Detection on VisA

Course project for **AIAA 3201 — Introduction to Computer Vision**.

Given a handful of defect-free reference images of a product, we must decide whether a test
image is anomalous (**image-level**) and where the defect is (**pixel-level**). The project has
three parts:

| Part | Content | Status |
|---|---|---|
| 1 | Classical normality modelling: template difference baseline | done |
| 2 | Reproduction of AnomalyDINO (DINOv2 ViT-S/14, 448 px, agnostic preprocessing) | done |
| 3 | Justified improvement: second visual arm (CLIP) + pre-registered score fusion + cost/reliability accounting | done |

Evaluation uses **six VisA categories** (`candle, capsules, cashew, chewinggum, fryum,
pipe_fryum`), the official 1-class split, shots `{1,2,4,8}` and seeds `{0,1,2}`.

## Repository layout

```
.
├── prepare_visa.py              # build the 1cls layout from the official split CSV
├── run_part1_template.py        # Part 1: template-difference baseline
├── run_grid.py                  # cached, layer-aware runner (Part 2 parity + Part 3 variants)
├── eval_maps.py                 # pixel/image metrics from saved anomaly maps (448 protocol)
├── make_tables.py               # collect runs into report tables
├── src/
│   ├── p2common.py              # shared data / metric utilities (one protocol for all methods)
│   ├── ad_core.py               # cached DINOv2 patch-feature extraction + multi-layer fusion
│   └── clip_core.py             # CLIP ViT-B/16 patch-token extraction (ln_post+proj, raster order)
├── code/anomalydino/            # fork of the official AnomalyDINO repo (changes tagged [P2])
├── scripts/
│   ├── parity_official_vs_cached.py   # bit-level parity: official AnomalyDINO vs cached runner
│   ├── verify_clip_interface.py       # CLIP interface probe (grid, order, position-emb interp)
│   ├── dump_scores.py                 # per-image score dumps for one (backbone, shot, seed) cell
│   ├── score_fusion.py                # two-arm score fusion: median/MAD calibration + alpha sweep
│   ├── fuse_sweep.py                  # aggregate per-image dumps across cells -> alpha sensitivity
│   ├── bench_inference.py             # per-image latency (stage breakdown) + peak GPU memory
│   ├── plot_per_category.py / plot_pipeline.py / make_heatmap_panel.py
│   └── run_*.sh                       # server launchers (one GPU job at a time)
├── docs/                        # writing drafts, references, CVPR-style report
└── results/                     # metrics (json), tables, figures, per-image score dumps
```

## Data preparation

VisA is read from the course mirror; only the six required categories are materialised.

```bash
python prepare_visa.py \
  --csv  <VisA>/split_csv/1cls.csv \
  --src  <VisA> \
  --dst  data/VisA_pytorch/1cls
```

This mirrors `spot-diff/utils/prepare_data.py --split-type 1cls`: images are symlinked (the
shared disk is nearly full), masks are copied and binarised to 0/255.

## Setup

```bash
uv venv .venv --python 3.10
uv pip install -r requirements.txt      # torch 2.7.1+cu126, faiss-cpu, ...
export DINOV2_HUB_DIR=/path/to/dinov2   # local clone of facebookresearch/dinov2
```

### Model weights

| Model | Source | How it is loaded |
|---|---|---|
| DINOv2 ViT-S/14 | official `dinov2_vits14_pretrain.pth` from `dl.fbaipublicfiles.com` | `torch.hub` with `source='local'` against a clone of `facebookresearch/dinov2` (`DINOV2_HUB_DIR`) |
| CLIP ViT-B/16 | OpenAI CLIP weights (`ViT-B-16.pt`) | downloaded once into `--clip_download_root` (default on the server: `/data/disk1/home/slv176/models/clip`); no HuggingFace path is used |

Two environment notes from the course server (kept here because they change the commands):

1. `torch.hub.load('facebookresearch/dinov2', ...)` probes github.com HTML, which the server
   drops. We clone the official repo once and load with `source='local'`; the weights are still
   the official `dinov2_vits14_pretrain.pth` from `dl.fbaipublicfiles.com`.
2. The server cannot reach huggingface.co, so no timm/HF download path is used anywhere
   (CLIP is loaded from the OpenAI checkpoint via `clip.load` with a local download root).

## Reproducing the results

```bash
# Part 1 -- template difference (no alignment)
python run_part1_template.py --data_root data/VisA_pytorch/1cls \
       --out results/part1_template --shots 1 2 4 8 --seeds 0 1 2 --align none

# Part 2 -- official AnomalyDINO
cd code/anomalydino
python run_anomalydino.py --dataset VisA --model_name dinov2_vits14 \
       --data_root ../../data/VisA_pytorch/1cls --preprocess agnostic --resolution 448 \
       --shots 1 2 4 8 --num_seeds 3 --calib_exclude 20 \
       --eval_clf --no-eval_segm --faiss_on_cpu --tag calib20
```

`--calib_exclude 20` is the project's calibration split: the first 20 normal training files of
each category (sorted) are reserved for the demo OK/NG threshold and are removed from the
few-shot reference pool before sampling. The reference filenames chosen for every
`(category, shot, seed)` are dumped to `refs/` so that Parts 1 and 3 reuse exactly the same
images.

### Part 2 parity (cached runner == official repo)

All Part 3 variants run on the cached, layer-aware runner `run_grid.py`. Its equivalence with
the official AnomalyDINO code path is verified bit-level first:

```bash
python scripts/parity_official_vs_cached.py   # max |AUROC diff| must be ~1e-7
```

The authoritative frozen baseline is `results/part2_cached/base_L12_v2_final.json` (72 cells,
max deviation 1.7e-09 vs the official run); the parity report is
`results/part2_official/parity_report.md`. Part 3 variants must run against the repaired
runner and a fresh fp32 feature cache (`--cache`).

### Part 3 -- visual arms, fusion, cost

```bash
# E1 arms: DINOv2 @448 (no mask, control) and CLIP ViT-B/16 @448 block 9 (no mask)
python run_grid.py --name e1_dino448_nomask --backbone dinov2 --layers 12 --no_mask \
       --data_root <data> --cache <fp32-cache> --out <out>
python run_grid.py --name e1_clip448_L9_nomask --backbone clip --layers 9 \
       --clip_resize_mode short_edge --no_mask \
       --data_root <data> --cache <fp32-cache> --out <out>

# Per-image score dumps for the fusion cells (cached features -> kNN only, no forward)
python scripts/dump_scores.py --name e1_dino448_nomask_s1_seed0 --backbone dinov2 \
       --layers 12 --no_mask --shot 1 --seed 0 --data_root <data> --cache <cache> --out <out>

# Two-arm fusion (pre-registered): per-category median/MAD standardisation on the 20
# calibration scores, alpha sweep; alpha=1 / alpha=0 reproduce the single arms exactly
python scripts/fuse_sweep.py --scores results/part3/scores --shots 1 2 4 --plot

# Cost: per-image latency (stage breakdown) + peak GPU memory, batch=1
python scripts/bench_inference.py --backbone dinov2 --layers 12 --threads 6 \
       --data_root <data> --out results/part3/bench_t6
python scripts/bench_inference.py --backbone clip --layers 9 --clip_resize_mode short_edge \
       --threads 6 --data_root <data> --out results/part3/bench_t6
```

Feature-cache note: `run_grid.py` reuses one fp32 patch-feature cache per backbone; the cache
key includes the split, so no cross-split contamination is possible (an earlier key collision
was found and fixed; the parity report documents it).

### Evaluation protocol

All methods are scored by the same code (`src/p2common.py`):

* image score = mean of the top 1 % patch distances (the AnomalyDINO aggregation);
* image AUROC over all test images of a category, then unweighted mean over the six categories;
* pixel AUROC over all test images, **including normal images with all-zero masks**;
* anomaly maps are resampled to 448 × 448 before pixel AUROC.

The last point is a deviation from the official evaluation, which pools full-resolution maps.
Reason: pooling 960 full-resolution maps needs ~11 GB and was OOM-killed on the shared server.
The deviation is applied identically to every method, so comparisons stay fair; the official
full-resolution image-level numbers are reported alongside as a sanity check.

Scores are **never** rescaled per test image before computing AUROC. Fusion thresholds are
calibrated on the 20 reserved normal images only; realised false-alarm rates are reported as
achieved operating points, not as calibrated guarantees (see the fusion report).

## Results

Headline numbers (image AUROC, mean ± std over seeds 0/1/2, six VisA categories):

| Method | 1-shot | 2-shot | 4-shot | 8-shot |
|---|---|---|---|---|
| Template difference (Part 1) | 52.0–58.0 | 54.1 | 55.4 | 58.0 |
| AnomalyDINO official (Part 2, masked) | **94.61 ± 0.84** | **95.33** | **96.15** | **96.62** |
| DINOv2 @448, no mask (control) | 93.34 ± 0.72 | 94.16 | 94.92 | 95.37 |
| CLIP @448 block 9, no mask (E1) | 87.70 ± 1.24 | 89.22 | 89.27 | 89.48 |
| **DINOv2+CLIP fusion, α=0.5** (shots 1/2/4) | 94.82 ± 0.59 | — | — | — |

Pixel AUROC for Part 2: 98.08 / 98.22 / 98.36 / 98.47.

Detailed artefacts:

* `results/part3/tables.md` — full grid tables (all parts, all cells);
* `results/part3/e1_full.md` — E1 comparison narrative (per-category deltas, error taxonomy);
* `results/part3/fusion/dinov2_clip_9cells.md` — fusion alpha sweep across 9 cells;
* `results/part3/bench_t6/bench_report.md` — measured latency/memory
  (DINOv2 148.5 ms / 120 MiB, CLIP 123.4 ms / 655 MiB per image, batch=1, 6 threads);
* `results/part3/error_cases.csv` — per-image verdicts of both arms (multimodal column pending);
* `results/part3/figs/` — `auroc_vs_shots.png`, `per_category_1shot.png`, `heatmap_panel.png`,
  `pipeline.png`;
* `docs/part3/report/main.pdf` — the CVPR-style report (sources in `docs/part3/report/`).

## Acknowledgements

This repository forks [AnomalyDINO](https://github.com/dammsi/AnomalyDINO) (WACV 2025) and uses
the VisA dataset collected by Zou et al. (ECCV 2022). Changes to the forked code are marked
with `[P2]` comments.
