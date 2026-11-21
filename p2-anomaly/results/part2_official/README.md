# Part 2 official AnomalyDINO — server backup (archived 2026-11-18)

Backup of the light-weight raw outputs of the **official** AnomalyDINO Part 2 runs,
retrieved from the GPU server (`121.12.5.46:10084`, user `slv176`) before the
`/tmp` copies could be lost. The aggregated per-seed metrics, refs and args derived
from these runs live in the top-level frozen archive `results/part2/` (verified
byte-identical against the server bundle on 2026-11-18); this directory holds the
additional raw artifacts that were **not** previously committed.

## Contents

```
p2-anomaly/results/part2_official/
├── eval448_{1,2,8}shot.json            # seed-0 cells re-scored with eval_maps.py
│                                       # (1/2/8-shot; archived earlier, unchanged)
├── metrics_{2,8}shot_seed0.json        # official metrics_seed=0.json (xcheck runs, slv176)
├── agg/                                # aggregation outputs from the 2026-11-11 rerun
│   ├── cell_<shot>shot_seed<k>.json    #   one summary json per cell (12 cells)
│   ├── eval448_4shot_seed0.json        #   4-shot seed-0 post-hoc scoring
│   ├── perseed_448.csv / table_448_wide.csv
│   └── summary_448.csv / summary_448.json
├── raw_runs/
│   ├── slv176_seed0/                   # pre-existing seed-0 runs (tags calib20 / xcheck)
│   │   └── <run_name>/{args.yaml, preprocess.yaml, measurements_seed=0.csv,
│   │                    metrics_seed=0.json (xcheck only), refs/*.json}
│   └── qzheng610_calib20r2/            # rerun 2026-11-11 under /tmp/qzheng610_p2
│       └── <shot>-shot_preprocess=agnostic_calib20r2/{args.yaml, preprocess.yaml,
│                        measurements_seed={0,1,2}.csv, metrics_seed={0,1,2}.json, refs/*.json}
├── logs/                               # driver + per-seed run logs from the rerun
└── scripts/                            # rerun orchestration scripts (run_missing*.sh,
                                        # stage.py, finalize.py, eval_perseed.py, watch.sh)
```

## Provenance and verification

- Source paths on the server:
  - `/tmp/slv176/results_VisA/dinov2_vits14_448/` (pre-existing seed-0 cells)
  - `/tmp/qzheng610_p2/results_VisA/dinov2_vits14_448/` (new cells, seeds 1/2 + 4-shot seed 0)
  - `/tmp/qzheng610_p2/{agg,scripts,logs}` and `/tmp/qzheng610_p2/upload.tgz`
- Consistency checks performed 2026-11-18:
  - `refs/` in the server bundle are byte-identical to `results/part2/refs/` (diff clean
    for 1/2/4/8-shot seed 0 and 4-shot seeds 1/2).
  - `args/`, per-seed `eval448_*_seed*.json` identical to `results/part2/`;
    `summary_all_shots_seeds.csv` differs only in line endings (CRLF vs LF).
  - Sample files re-checked by md5 against the server after download (summary_448.json,
    4-shot metrics_seed=0.json, candle_shot=4_seed=2.json) — all match.
- The old empty `metrics_1shot_seed0.json` (0 bytes) was **removed**: no such file ever
  existed on the server — the 1-shot seed-0 cell has `measurements_seed=0.csv` only, and
  its Image/Pixel AUROC lives in `eval448_1shot_seed0.json` / `agg/cell_1shot_seed0.json`.

Deliberately **not** committed: raw `anomaly_maps/` (npy/tiff), per-category example PNGs
(`anomaly_maps_examples_*.png`, `reference_samples.png`, `examples/`), VisA images or GT
masks, model weights, or the server venv. The large raw tarball `p2_part2_calib20r2.tgz`
(~168 MB) also remains server-side only.
