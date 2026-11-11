# HANDOFF - what is done and what to continue

Updated: 2026-11-11 17:55 GMT+8 by Sibei (agent session). Results snapshot in
[`p2-anomaly/results/tables.md`](p2-anomaly/results/tables.md).

## Server (GPU box)

- Host `121.12.5.46`, port `10084`, user `slv176`. SSH key auth
  (`~/.ssh/aiaa3201_a6000` on Sibei's laptop - teammate needs own key: send
  pubkey, then `echo '<pubkey>' >> ~/.ssh/authorized_keys` on the server).
- GPU: RTX A6000 49 GB, **shared** - check `nvidia-smi` before launching; run
  **one** GPU job at a time.
- Disk: `/data/disk1` ~99% full - **do not write anything there except code in
  `$HOME/aiaa3201_p2`**. All big files live under `/tmp/slv176/` (72 GB free,
  but `/tmp` does not survive a reboot).
- Layout:
  - code: `/data/disk1/home/slv176/aiaa3201_p2`
  - venv: `/tmp/slv176/venv` (torch 2.7.1+cu126, faiss, cv2, scikit-image)
  - data: `/tmp/slv176/data/VisA_pytorch/1cls` (six categories, 960 test imgs)
  - DINOv2: repo at `/tmp/slv176/third_party/dinov2`, weights at
    `~/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth`
  - results: `/tmp/slv176/{part1,part1_crop,grid,results_VisA}`, logs in
    `/tmp/slv176/logs/`
  - raw VisA + split csv: `/data/disk1/shared/aiaa3201/project2_anomaly/VisA`

## Done (experiments completed, numbers in tables.md)

1. **Part 1 baseline** - full grid 6 cats x {1,2,4,8}-shot x seeds 0/1/2:
   Image AUROC 52.0 -> 58.0, Pixel 65.1 -> 66.1 (3-seed mean). Weak, as expected;
   failure mode diagnosed (object misalignment rings dominate the difference map).
2. **Part 1 + foreground-crop ablation** - full grid done: small consistent gain
   at 1-4 shot image level (52.0 -> 55.3 at 1-shot).
3. **Part 2 official AnomalyDINO** - seed 0: 1-shot 94.62/98.02,
   2-shot 94.70/98.23, 8-shot 96.22/98.47 (image/pixel). Reference filename
   lists archived in each run's `refs/` dir.
4. **Part 2 cached grid runner** (`run_grid.py`) - full 72-cell grid done.

## Top-priority open items (in order)

### 1. Reconcile official vs cached scoring (BLOCKS Part 3)

Official 1-shot image AUROC = 94.62, cached runner = 89.52 on the same protocol.
The suspected difference: the official path computes image score as top-1%
mean **after** smoothing/resizing to 448, while `src/ad_core.py` scores on raw
patch distances. Fix in this order:

```bash
cd ~/aiaa3201_p2
# 1) score the official 1-shot saved anomaly maps with the cached scorer
#    (anomaly_maps/*.npy already on disk) and compare per-image scores;
# 2) whichever step diverges, port the official post-processing into
#    src/ad_core.py `image_score`, rerun the cached grid (it only re-scores,
#    features are cached in /tmp/slv176/featcache, ~20 min for all 72 cells);
# 3) re-generate results tables.
```

### 2. Part 2 official: seeds 1,2 for 2/4/8-shot

Currently only seed 0 exists for 2/8-shot; 4-shot missing entirely.
One GPU process, serial, ~2.5 h for {2,4,8} x seeds {1,2}:

```bash
cd /tmp/slv176 && OMP_NUM_THREADS=1 \
DINOV2_HUB_DIR=/tmp/slv176/third_party/dinov2 MPLBACKEND=Agg PYTHONUNBUFFERED=1 \
nohup /tmp/slv176/venv/bin/python /data/disk1/home/slv176/aiaa3201_p2/code/anomalydino/run_anomalydino.py \
  --dataset VisA --model_name dinov2_vits14 --data_root /tmp/slv176/data/VisA_pytorch/1cls \
  --preprocess agnostic --resolution 448 --shots 2 4 8 --num_seeds 2 --just_seed 1 \
  --calib_exclude 20 --eval_clf --no-eval_segm --faiss_on_cpu --warmup_iters 3 --tag xcheck \
  > /tmp/slv176/logs/part2_seeds12.log 2>&1 &
# then pixel metrics per run dir:
python /data/disk1/home/slv176/aiaa3201_p2/eval_maps.py --run <run_dir>
```

### 3. Persist light results back home

`/tmp` is wiped on reboot. Copy any new `metrics_seed*.json` / `eval448_*.json`
/ `refs/` to `~/aiaa3201_p2/results_backup/` right after a run finishes, and
commit to git (results/ in repo is fine - only light JSON/MD/PNG).

### 4. Later (only after 1+2)

- Part 3 improvement runs (scaffold ready: `run_grid.py --layers 6 9 12
  --fusion late_z`), but **only after** the scoring protocol is frozen.
- Hard-case figure / video material: `debug_template.py` and `make_figures.py`
  are ready to use.

## Repo conventions

- main branch, results under `p2-anomaly/results/` (light files only),
  commit message style: short imperative subject + bullet body.
- Do not commit: datasets, weights, anomaly-map npy/tiff files, logs.
