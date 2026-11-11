"""Collect per-run metrics into report tables (image AUROC / pixel AUROC, mean +- std over seeds).

Everything is read at the (shot, category, seed) level and aggregated by one function
(`p2common.summarize`), so the tables cannot disagree about how a class mean or a seed spread
is formed.

Sources
-------
* Part 1 : `<dir>/summary.json`  -- {metric: {"<shot>-shot": {cat: {seed: value}}}}
* Part 2 (official code) : `<results_VisA>/<shot>-shot_preprocess=agnostic_<TAG>/metrics_seed=<s>.json`
* Part 2 / Part 3 (cached grid) : `<name>_final.json` from run_grid.py or merge_grid.py, which
  also keeps the per-seed values under `seed_values`.

Usage
-----
python make_tables.py --part1 results/part1_template --part2-stage results_VisA/dinov2_vits14_448 \
    --part2 calib20 --grid "Multi-block fusion=/tmp/slv176/grid/fuse_final.json" --out results/tables
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from p2common import VISA_6, jdump, summarize


def _shot_key(k):
    return int(re.match(r"(\d+)-shot", k).group(1))


def load_part1(path, cats=VISA_6):
    """Per-seed values from a run_part1_template.py output directory."""
    for name in ("summary.json", "results.json"):
        p = os.path.join(path, name)
        if os.path.isfile(p):
            raw = json.load(open(p))
            if all(isinstance(v, dict) and any(isinstance(x, dict) for x in v.values())
                   for v in raw.get("image_auroc", {}).values()):
                # already {shot: {cat: {seed: val}}}
                return summarize(raw, cats)
    return {}


def load_official(base_dir, tag, cats=VISA_6, shots=(1, 2, 4, 8)):
    """Official-style run dirs: one metrics json per seed, keyed by category."""
    summary = {"image_auroc": {}, "pixel_auroc": {}}
    for shot in shots:
        d = os.path.join(base_dir, f"{shot}-shot_preprocess=agnostic_{tag}")
        files = sorted(glob.glob(os.path.join(d, "metrics_seed=*.json")))
        for f in files:
            seed = re.search(r"metrics_seed=(\d+)", f).group(1)
            m = json.load(open(f))
            for cat in cats:
                if cat not in m:
                    continue
                summary["image_auroc"].setdefault(f"{shot}-shot", {}).setdefault(cat, {})[seed] = \
                    m[cat].get("classification_AUROC")
                summary["pixel_auroc"].setdefault(f"{shot}-shot", {}).setdefault(cat, {})[seed] = \
                    m[cat].get("seg_AUROC")
    if not summary["image_auroc"]:
        return {}
    return summarize(summary, cats)


def load_grid(path, cats=VISA_6):
    """Read a run_grid.py / merge_grid.py output; per-seed values preferred, else the row means."""
    if not os.path.isfile(path):
        return {}
    raw = json.load(open(path))
    if "seed_values" in raw:
        return summarize(raw["seed_values"], cats)
    final = {}
    for metric in ("image_auroc", "pixel_auroc"):
        final[metric] = {}
        for sk, row in raw.get(metric, {}).items():
            r = {}
            for c in cats:
                r[c] = row.get(c, float("nan"))
                r[c + "_std"] = row.get(c + "_std", float("nan"))
            r["MEAN"] = row.get("MEAN", float("nan"))
            r["MEAN_std"] = row.get("MEAN_std", float("nan"))
            r["n_seeds"] = row.get("n_seeds", 0)
            final[metric][sk] = r
    return final


def _fmt(v, sd, nd=2):
    if v != v:
        return "-"
    return f"{v*100:.{nd}f} ± {sd*100:.2f}" if sd == sd else f"{v*100:.{nd}f}"


def summary_table(runs, shots, key):
    lines = ["| Method | " + " | ".join(f"{s}-shot" for s in shots) + " |",
             "|---|" + "---|" * len(shots)]
    for name, data in runs.items():
        cells = []
        for s in shots:
            sk = f"{s}-shot"
            row = data.get(key, {}).get(sk)
            cells.append(_fmt(row["MEAN"], row["MEAN_std"]) if row else "-")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def per_category_table(name, data, cats, shots, key):
    lines = [f"### {name}",
             "| Category | " + " | ".join(f"{s}-shot" for s in shots) + " |",
             "|---|" + "---|" * len(shots)]
    for c in cats:
        cells = []
        for s in shots:
            row = data.get(key, {}).get(f"{s}-shot")
            cells.append(_fmt(row[c], row[c + "_std"]) if row else "-")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    cells = [_fmt(data.get(key, {}).get(f"{s}-shot", {}).get("MEAN", float("nan")),
                  data.get(key, {}).get(f"{s}-shot", {}).get("MEAN_std", float("nan")))
             for s in shots]
    lines.append("| **Mean** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--part1", default="results/part1_template", help="Part 1 output directory")
    ap.add_argument("--part2-stage", default=None,
                    help="dir holding <shot>-shot_preprocess=agnostic_<TAG>/ (default: auto-detect)")
    ap.add_argument("--part2", default="calib20", help="tag of the official Part 2 run")
    ap.add_argument("--grid", nargs="*", default=[],
                    help='extra runs as "name=path/to/grid_final.json"')
    ap.add_argument("--out", default="results/tables")
    args = ap.parse_args()

    cats, shots = VISA_6, [1, 2, 4, 8]
    runs = {}

    p1 = load_part1(os.path.join(args.root, args.part1), cats)
    if p1:
        runs["Template difference (Part 1)"] = p1

    stage = args.part2_stage
    if stage is None:
        for cand in ("results_VisA/dinov2_vits14_448",
                     "code/anomalydino/results_VisA/dinov2_vits14_448"):
            if os.path.isdir(os.path.join(args.root, cand)):
                stage = cand
                break
    if stage:
        p2 = load_official(os.path.join(args.root, stage), args.part2, cats, shots)
        if p2:
            runs["AnomalyDINO (Part 2, official code)"] = p2

    for spec in args.grid:
        name, path = spec.split("=", 1)
        d = load_grid(path, cats)
        if d:
            runs[name] = d

    if not runs:
        raise SystemExit("no results found -- check --part1 / --part2-stage / --grid")

    os.makedirs(args.out, exist_ok=True)
    md = ["# Results", "",
          "Mean ± std over seeds (0/1/2); the std is taken over seeds of the six-category mean.",
          ""]
    for metric, key in (("Image-level AUROC (%)", "image_auroc"),
                        ("Pixel-level AUROC (%)", "pixel_auroc")):
        md.append(f"## {metric}")
        md.append(summary_table(runs, shots, key))
        md.append("")
        for name, data in runs.items():
            md.append(per_category_table(name, data, cats, shots, key))
            md.append("")
    text = "\n".join(md)
    open(os.path.join(args.out, "tables.md"), "w").write(text)
    print(text)

    jdump(runs, os.path.join(args.out, "metrics.json"))


if __name__ == "__main__":
    main()
