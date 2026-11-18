# Part 3 results (visual branch, E1 + stage-4 tables/figures)

Artifacts in this directory are produced by the scripts listed below; every number comes from the
frozen protocol recorded in `e1_pilot.md` §3 (CLIP ViT-B/16 @448, block 9, `short_edge` framing,
no foreground mask; DINOv2 ViT-S/14 @448 block 12 with the same no-mask protocol as the control).

| file | what it holds | produced by |
|---|---|---|
| `e1_pilot.md` | pilot rows and the frozen configuration (stage 3, steps 2–3) | manual |
| `e1_full.md` | full 6×4×3 grid: three-line comparison, per-category tables, sensitivity rows | `make_tables.py` + narrative |
| `tables.md` | machine-generated tables (image and pixel AUROC, mean ± std over seeds) | `make_tables.py` |
| `metrics.json` | the same numbers in JSON, consumed by `make_figures.py` | `make_tables.py` |
| `error_taxonomy.md` / `error_cases.csv` | per-image verdicts of the two controlled arms, crossed 2×2 | `scripts/error_taxonomy.py` |
| `figs/auroc_vs_shots.png` | AUROC vs shots for all four lines | `make_figures.py` |
| `figs/heatmap_panel.png` | same test images scored by both backbones, with ground truth | `scripts/make_heatmap_panel.py` |
| `part2_official_agg_final.json` | Part 2 official aggregates reshaped into the grid layout so a single aggregation path covers all lines | `scripts/build_part2_grid_json.py` |

Per-image score dumps (the raw input of the taxonomy and the heatmaps) live outside the repo
because they are large: `<server>/grid/scores/*.json`.

Reproduce, in order:

```bash
# 1. grids (CLIP line, then the DINOv2 control at the same resolution/protocol)
bash scripts/run_e1_full.sh
# 2. per-image scores, shared case list, heatmaps, taxonomy, tables, curves
bash scripts/run_stage4.sh
```

`run_stage4.sh` waits for the `E1_DONE` marker written by `run_e1_full.sh`, so the two can be
launched back to back.
