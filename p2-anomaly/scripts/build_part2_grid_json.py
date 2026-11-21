"""Convert the Part 2 official-code aggregate into the same JSON shape run_grid.py emits.

Why: `make_tables.py` has exactly one aggregation path (`p2common.summarize`) so that no two
tables can disagree about how a class mean or a seed spread is formed. The official AnomalyDINO
runs are stored as `results/part2_official/agg/summary_448.json` (cells keyed "<shot>-<seed>"),
which is a different shape. This adapter rewrites those cells into the grid layout

    {"seed_values": {metric: {"<shot>-shot": {category: {"<seed>": value}}}}}

so the official baseline can be passed to `make_tables.py --grid` like any local run, instead of
adding a second summarisation path for it.

Usage
-----
python scripts/build_part2_grid_json.py \
    --src results/part2_official/agg/summary_448.json \
    --out results/part3/part2_official_agg_final.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import p2common as C

METRICS = ("image_auroc", "pixel_auroc")


def convert(src):
    raw = json.load(open(src))
    cells = raw["cells"]
    seed_values = {m: {} for m in METRICS}
    for key, cell in cells.items():
        shot, seed = int(cell["shot"]), int(cell["seed"])
        shot_key = f"{shot}-shot"
        for m in METRICS:
            for cat, rec in cell.get("per_category", {}).items():
                val = rec.get(m)
                if val is None or val != val:
                    continue
                seed_values[m].setdefault(shot_key, {}).setdefault(cat, {})[str(seed)] = float(val)
    final = C.summarize(seed_values, C.VISA_6)
    final["seed_values"] = seed_values
    final["source"] = os.path.abspath(src)
    final["config"] = {"backbone": "dinov2_vits14", "resolution": 448, "layers": [12],
                       "preprocess": "agnostic (PCA foreground mask, calib20)",
                       "impl": "official AnomalyDINO detection.py"}
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="results/part2_official/agg/summary_448.json")
    ap.add_argument("--out", default="results/part3/part2_official_agg_final.json")
    args = ap.parse_args()
    final = convert(args.src)
    C.jdump(final, args.out)
    print(f"wrote {args.out}")
    for m in METRICS:
        for k in sorted(final[m], key=lambda s: int(s.split("-")[0])):
            r = final[m][k]
            print(f"  {m:12s} {k:8s} MEAN={r['MEAN'] * 100:.2f} ({r['MEAN_std'] * 100:.2f})")


if __name__ == "__main__":
    main()
