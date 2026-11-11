# AIAA3201 Computer Vision — Project 2

Few-shot industrial visual anomaly detection on the VisA dataset.

Coursework repository: assignment briefs, the Project 2 implementation, and the scripts used
to run it on a remote GPU server.

## Repository layout

```
.
├── Project_2__Few_Shot_Industrial_Visual_Anomaly_Detection.pdf   # assignment brief
├── project_AIAA3201_reminder.pdf                                 # submission notes
└── p2-anomaly/
    ├── prepare_visa.py            # build the VisA 1cls layout from the official split CSV
    ├── run_part1_template.py      # Part 1 — classical template-difference baseline
    ├── run_grid.py                # cached grid runner for Part 2 / Part 3 configurations
    ├── make_tables.py             # collect per-run metrics into report tables
    ├── code/
    │   ├── run_anomalydino.py     # official AnomalyDINO entry point (unmodified copy)
    │   └── anomalydino/           # AnomalyDINO source (DINOv2 ViT-S/14, k-NN patch scoring)
    ├── src/
    │   ├── p2common.py            # shared dataset / metric code — one evaluator for all methods
    │   └── ad_core.py             # cached, layer-aware re-implementation of AnomalyDINO scoring
    ├── scripts/                   # remote bootstrap, DINOv2 prefetch, Part 2 sweep
    └── results/                   # run outputs (git-ignored)
```

## What each part does

| Part | Method | Entry point |
|---|---|---|
| 1 | Template difference: `0.5·|dI| + 0.5·(1−SSIM)` against the closest reference, ECC-aligned, Gaussian-smoothed, top-1 % mean pooling | `run_part1_template.py` |
| 2 | Reproduce AnomalyDINO: DINOv2 ViT-S/14, 448 px, agnostic preprocessing, 6 VisA categories, shots {1,2,4,8}, seeds {0,1,2} | `scripts/run_part2.sh` |
| 3 | Improvement: multi-block feature fusion (late calibration-based fusion and early concatenation) over the same protocol | `run_grid.py` |

All three parts are scored by the **same** code in `src/p2common.py` so that the comparison is
apples-to-apples. Pixel-level AUROC is computed on maps resampled to 448×448 for every method.

## Setup

Python 3.10+ with the packages in `code/anomalydino/requirements.txt` (torch, torchvision,
opencv-python, scikit-image, scipy, numpy, matplotlib, tqdm, pandas).

```bash
pip install -r p2-anomaly/code/anomalydino/requirements.txt
```

DINOv2 weights are pulled through `torch.hub`. On hosts that cannot reach `github.com`
reliably, run `bash p2-anomaly/scripts/fetch_dinov2.sh` once to clone the repo and pre-fetch
the ViT-S/14 checkpoint, then point `DINOV2_HUB_DIR` at the clone.

## Data

`third_party/` is not tracked in this repository. The evaluation needs the VisA split CSVs and
the `1cls` preparation utility from Amazon Science's SPot-the-Difference release:

```bash
git clone --depth 1 https://github.com/amazon-science/spot-diff.git p2-anomaly/third_party/spot-diff
```

Then download `VisA_20220922.tar` (see that repository's README) and build the layout:

```bash
python p2-anomaly/prepare_visa.py --src <VisA_pytorch_dir> --dst <out_dir>
```

`prepare_visa.py` symlinks images by default (masks are copied, since they get binarised) to
avoid duplicating ~0.7 GB.

## Running

```bash
# Part 1 — template baseline
python run_part1_template.py --variant align=affine --out results/part1_template

# Part 2 — official AnomalyDINO sweep (remote, see scripts/run_part2.sh for the full command)
bash scripts/run_part2.sh calib20

# Part 3 — multi-block fusion grid
python run_grid.py --name fuse_L6_9_12 --layers 6 9 12 --fusion late_z

# Collect metrics into report tables
python make_tables.py --root . --part2 calib20 --part1 align=affine --out results/tables
```

## Notes

- `scripts/remote_setup.sh` and `scripts/run_part2.sh` were written for a shared A6000 server
  and contain that machine's paths (`/tmp/slv176`, `~/aiaa3201_p2`). Adjust them for another host.
- Results are written under `results/`, which is git-ignored apart from `.gitkeep`.
