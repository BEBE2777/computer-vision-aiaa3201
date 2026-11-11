"""Collect per-run metrics into report tables (image AUROC / pixel AUROC, mean +- std over seeds).

Sources
-------
* Part 2 / Part 3 : official-style run dirs
      results_VisA/dinov2_vits14_448/<shot>-shot_preprocess=agnostic_<TAG>/metrics_seed=<s>.json
    which contain per-category keys `classification_AUROC` and `seg_AUROC`.
* Part 1          : results/part1_template/<variant>/results.json
    which stores per-category mean and std over seeds already.

Usage
-----
python make_tables.py --root . --part2 calib20 --part1 align=none --out results/tables
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from p2common import VISA_6, jdump


def load_official(base_dir, tag, cats=VISA_6, shots=(1, 2, 4, 8)):
    """{shot: {cat: {"image": (mean, std), "pixel": (mean, std)}}}"""
    out = {}
    for shot in shots:
        d = os.path.join(base_dir, f"{shot}-shot_preprocess=agnostic_{tag}")
        if not os.path.isdir(d):
            continue
        files = sorted(glob.glob(os.path.join(d, "metrics_seed=*.json")))
        if not files:
            continue
        per_cat = {c: {"image": [], "pixel": []} for c in cats}
        for f in files:
            m = json.load(open(f))
            for c in cats:
                if c not in m:
                    continue
                per_cat[c]["image"].append(m[c].get("classification_AUROC"))
                per_cat[c]["pixel"].append(m[c].get("seg_AUROC"))
        out[shot] = {}
        for c in cats:
            vals = {}
            for k in ("image", "pixel"):
                v = [x for x in per_cat[c][k] if x is not None]
                vals[k] = (float(np.mean(v)), float(np.std(v))) if v else (float("nan"),) * 2
            out[shot][c] = vals
    return out


def load_part1(variant_dir, cats=VISA_6):
    """{shot: {cat: {"image": (mean, std), "pixel": (mean, std)}}}"""
    path = os.path.join(variant_dir, "results.json")
    if not os.path.isfile(path):
        return {}
    raw = json.load(open(path))
    out = {}
    for shot_key, row in raw.get("image_auroc", {}).items():
        shot = int(re.match(r"(\d+)-shot", shot_key).group(1))
        out[shot] = {}
        for c in cats:
            out[shot][c] = {
                "image": (row.get(c, float("nan")), row.get(c + "_std", 0.0)),
                "pixel": (raw["pixel_auroc"][shot_key].get(c, float("nan")),
                          raw["pixel_auroc"][shot_key].get(c + "_std", 0.0)),
            }
    return out


def load_grid(path, cats=VISA_6):
    """{shot: {cat: {"image": (m,s), "pixel": (m,s)}}} from run_grid.py's *_final.json"""
    if not os.path.isfile(path):
        return {}
    raw = json.load(open(path))
    out = {}
    for metric, key in (("image_auroc", "image"), ("pixel_auroc", "pixel")):
        for sk, row in raw.get(metric, {}).items():
            shot = int(re.match(r"(\d+)-shot", sk).group(1))
            out.setdefault(shot, {})
            for c in cats:
                out[shot].setdefault(c, {})[key] = (row.get(c, float("nan")),
                                                    row.get(c + "_std", 0.0))
    # make sure both keys exist for every (shot, cat)
    for shot in out:
        for c in out[shot]:
            out[shot][c].setdefault("image", (float("nan"),) * 2)
            out[shot][c].setdefault("pixel", (float("nan"),) * 2)
    return out


def category_mean(per_cat, cats, key):
    vals = [per_cat[c][key][0] for c in cats if per_cat[c][key][0] == per_cat[c][key][0]]
    return float(np.mean(vals)) if vals else float("nan")


def markdown_table(runs, cats, metric_key, shots):
    lines = ["| Method | " + " | ".join(f"{s}-shot" for s in shots) + " |",
             "|---|" + "---|" * len(shots)]
    for name, data in runs.items():
        cells = []
        for s in shots:
            if s not in data:
                cells.append("-")
                continue
            m = category_mean(data[s], cats, metric_key)
            # pooled std across categories is not meaningful; report the seed std of the mean
            stds = [data[s][c][metric_key][1] for c in cats]
            stds = [x for x in stds if x == x]
            cells.append(f"{m*100:.2f} ± {np.mean(stds)*100:.2f}" if stds else f"{m*100:.2f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def per_category_table(name, data, cats, shots, metric_key):
    lines = [f"### {name}",
             "| Category | " + " | ".join(f"{s}-shot" for s in shots) + " |",
             "|---|" + "---|" * len(shots)]
    for c in cats:
        cells = []
        for s in shots:
            if s in data and c in data[s]:
                m, sd = data[s][c][metric_key]
                cells.append(f"{m*100:.2f} ± {sd*100:.2f}")
            else:
                cells.append("-")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    cells = []
    for s in shots:
        cells.append(f"{category_mean(data[s], cats, metric_key)*100:.2f}" if s in data else "-")
    lines.append("| **Mean** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--part2", default="calib20", help="tag of the Part 2 run")
    ap.add_argument("--part1", default="align=none", help="variant dir under results/part1_template")
    ap.add_argument("--grid", nargs="*", default=[],
                    help="extra runs as name=path/to/grid_final.json")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="extra runs as name=tag (official-style) e.g. ours_layers=layers")
    ap.add_argument("--out", default="results/tables")
    args = ap.parse_args()

    cats = VISA_6
    shots = [1, 2, 4, 8]
    base = os.path.join(args.root, "code/anomalydino/results_VisA/dinov2_vits14_448")
    if not os.path.isdir(base):
        base = os.path.join(args.root, "results_VisA/dinov2_vits14_448")

    runs = {}
    p1 = load_part1(os.path.join(args.root, "results/part1_template", args.part1), cats)
    if p1:
        runs["Template difference (ours)"] = p1
    p2 = load_official(base, args.part2, cats, shots)
    if p2:
        runs["AnomalyDINO (official code)"] = p2
    for spec in args.extra:
        name, tag = spec.split("=", 1)
        d = load_official(base, tag, cats, shots)
        if d:
            runs[name] = d

    for spec in args.grid:
        name, path = spec.split("=", 1)
        d = load_grid(path, cats)
        if d:
            runs[name] = d

    os.makedirs(args.out, exist_ok=True)
    md = ["# Results", ""]
    for metric, key in (("Image-level AUROC (%)", "image"), ("Pixel-level AUROC (%)", "pixel")):
        md.append(f"## {metric}")
        md.append(markdown_table(runs, cats, key, shots))
        md.append("")
        for name, data in runs.items():
            md.append(per_category_table(name, data, cats, shots, key))
            md.append("")
    text = "\n".join(md)
    open(os.path.join(args.out, "tables.md"), "w").write(text)
    print(text)

    jdump({name: {str(s): {c: {k: data[s][c][k] for k in ("image", "pixel")}
                           for c in data[s]} for s in data}
           for name, data in runs.items()}, os.path.join(args.out, "metrics.json"))


if __name__ == "__main__":
    main()
