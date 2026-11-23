"""Alpha sweep of the two-arm score fusion over every dumped (shot, seed) cell.

`score_fusion.py` fuses one cell and writes one report. This script runs the same
pre-registered protocol on all 12 cells (shots {1,2,4,8} x seeds {0,1,2}) of the two E1
arms and aggregates, so the fusion claim is a mean over cells instead of a single pilot.

What it does
------------
For each cell: load the two dump JSONs, align them image by image (the same check
`score_fusion.py` runs), normalise with the 20 held-out normal calibration images of
each category, and evaluate alpha in {0, 0.25, 0.5, 0.75, 1}. Then it reports, per alpha,
the mean and standard deviation of the six-category image AUROC *across cells*, and the
achieved operating point (recall / false-positive rate at the 95th percentile of the
fused calibration scores).

Nothing here selects alpha on test data: 0.5 is fixed in advance, and the alpha=1 / alpha=0
rows are the single-branch identity check (they must reproduce the AUROC of each arm alone,
which is also what the E1 tables report).

Usage
-----
python scripts/fuse_sweep.py --scores results/part3/scores --out results/part3/fusion \
       --tag dinov2_clip_12cells --plot

Dump naming convention (written by scripts/run_dumps.sh):
    e1_dino448_nomask.json                  shot 1, seed 0 (first dump, legacy name)
    e1_dino448_nomask_s{shot}_seed{seed}.json
    e1_clip448_L9_nomask.json / ..._s{shot}_seed{seed}.json
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_fusion as F  # noqa: E402


ARMS = {
    "dinov2": ("e1_dino448_nomask{cell}.json", "e1_dino448_nomask.json"),
    "clip": ("e1_clip448_L9_nomask{cell}.json", "e1_clip448_L9_nomask.json"),
}


def find_dump(scores_dir, arm, shot, seed):
    """Locate one cell's dump; shot 1 / seed 0 was written under the legacy name."""
    with_cell, legacy = ARMS[arm]
    cell = f"_s{shot}_seed{seed}"
    for pat in (with_cell.format(cell=cell), legacy):
        hits = sorted(glob.glob(os.path.join(scores_dir, pat)))
        if hits:
            return hits[0]
    return None


def collect(scores_dir, shots, seeds):
    """Paired cells: every (shot, seed) for which both arms have a dump."""
    have = {arm: {(shot, seed): find_dump(scores_dir, arm, shot, seed)
                  for shot in shots for seed in seeds} for arm in ARMS}
    cells = []
    for shot in shots:
        for seed in seeds:
            d, c = have["dinov2"][(shot, seed)], have["clip"][(shot, seed)]
            if d is None or c is None:
                print(f"[skip] shot={shot} seed={seed}: "
                      f"dinov2={'ok' if d else 'missing'}, clip={'ok' if c else 'missing'}")
                continue
            cells.append((shot, seed, d, c))
    return cells


def run_cell(dino_path, clip_path, alphas):
    a, b = F.load_dump(dino_path), F.load_dump(clip_path)
    common, _ = F.align(a, b, require_exact=True)
    rows, _, _, _, decisions = F.fuse_cells(common, alphas, 0.5)
    return rows, decisions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results/part3/scores")
    ap.add_argument("--out", default="results/part3/fusion")
    ap.add_argument("--tag", default="dinov2_clip_12cells")
    ap.add_argument("--shots", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--alphas", nargs="+", type=float, default=F.DEFAULT_ALPHAS)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    cells = collect(args.scores, args.shots, args.seeds)
    if not cells:
        raise SystemExit(f"no paired dumps under {args.scores}")

    print(f"[info] {len(cells)} cells paired")
    os.makedirs(args.out, exist_ok=True)

    records = []
    for shot, seed, dino_path, clip_path in cells:
        rows, decisions = run_cell(dino_path, clip_path, args.alphas)
        rec = {"shot": shot, "seed": seed,
               "dinov2": os.path.basename(dino_path), "clip": os.path.basename(clip_path),
               "alphas": [{"alpha": r["alpha"], "macro": r["macro"],
                           "recall": d["recall"], "fpr": d["fpr"]}
                          for r, d in zip(rows, decisions)]}
        records.append(rec)
        line = "  ".join(f"a={r['alpha']:.2f}:{r['macro'] * 100:5.2f}" for r in rows)
        print(f"[cell] shot={shot} seed={seed}  {line}")

    def agg(key):
        out = []
        for i, alpha in enumerate(args.alphas):
            vals = np.array([r["alphas"][i][key] for r in records], dtype=float)
            out.append({"alpha": alpha, "mean": float(vals.mean()),
                        "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                        "min": float(vals.min()), "max": float(vals.max()),
                        "values": [float(v) for v in vals]})
        return out

    summary = {"tag": args.tag, "n_cells": len(records),
               "cells": [{"shot": r["shot"], "seed": r["seed"]} for r in records],
               "macro": agg("macro"), "recall": agg("recall"), "fpr": agg("fpr"),
               "per_cell": records}
    with open(os.path.join(args.out, f"{args.tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    b1 = next(x for x in summary["macro"] if abs(x["alpha"] - 1.0) < 1e-12)
    b2 = next(x for x in summary["macro"] if abs(x["alpha"] - 0.0) < 1e-12)
    main = next(x for x in summary["macro"] if abs(x["alpha"] - 0.5) < 1e-12)

    lines = [f"# Two-arm score fusion — alpha sweep over {len(records)} cells", "",
             "Both arms are the E1 configuration (DINOv2 ViT-S/14 @448 block 12 and CLIP "
             "ViT-B/16 @448 block 9, no foreground mask), fused after per-category "
             "median/MAD normalisation on the 20 held-out normal calibration images. "
             "alpha = 0.5 was fixed before any result was seen.", "",
             f"Cells: shots {args.shots} x seeds {args.seeds} "
             f"({len(records)} of {len(args.shots) * len(args.seeds)}).", "",
             "## Six-category image AUROC (%) across cells", "",
             "| alpha | mean | std | min | max | note |", "|---:|---:|---:|---:|---:|---|"]
    for m, r, f_ in zip(summary["macro"], summary["recall"], summary["fpr"]):
        note = "--"
        if abs(m["alpha"] - 1.0) < 1e-12:
            note = "= DINOv2 alone"
        elif abs(m["alpha"] - 0.0) < 1e-12:
            note = "= CLIP alone"
        elif abs(m["alpha"] - 0.5) < 1e-12:
            note = "pre-registered"
        lines.append(f"| {m['alpha']:.2f} | {m['mean'] * 100:.2f} | {m['std'] * 100:.2f} "
                     f"| {m['min'] * 100:.2f} | {m['max'] * 100:.2f} | {note} |")

    gain = (main["mean"] - max(b1["mean"], b2["mean"])) * 100
    lines += ["", f"Fused (alpha=0.5) minus the better single arm: **{gain:+.2f}** points "
                  f"(per-cell gains: "
                  + ", ".join(f"{g:+.2f}" for g in
                              (np.array(main["values"]) - np.maximum(b1["values"],
                                                                      b2["values"])) * 100)
                  + ").", "",
              "## Achieved operating point (threshold = 95th percentile of that alpha's "
              "fused calibration scores, per category)", "",
              "| alpha | recall | FPR |", "|---:|---:|---:|"]
    for r, f_ in zip(summary["recall"], summary["fpr"]):
        lines.append(f"| {r['alpha']:.2f} | {r['mean'] * 100:.1f}% | {f_['mean'] * 100:.1f}% |")
    lines += ["", "The nominal target at the 95th percentile is a 5% false-alarm rate; the "
                  "realised rates are higher for every arm and every alpha, i.e. 20 normal "
                  "calibration images per category do not cover the variation of the normal "
                  "test images. Quote these as achieved operating points, not as a calibrated "
                  "5% guarantee.", "",
              "## Per cell (six-category macro AUROC, %)", "",
              "| shot | seed | " + " | ".join(f"a={a:.2f}" for a in args.alphas) + " |",
              "|---:|---:|" + "---:|" * len(args.alphas)]
    for r in records:
        lines.append(f"| {r['shot']} | {r['seed']} | "
                     + " | ".join(f"{a['macro'] * 100:.2f}" for a in r["alphas"]) + " |")

    path = os.path.join(args.out, f"{args.tag}.md")
    open(path, "w").write("\n".join(lines) + "\n")
    print(f"wrote {path}")

    if args.plot:
        plot(summary, os.path.join(args.out, f"{args.tag}_alpha.png"))


def plot(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    al = [m["alpha"] for m in summary["macro"]]
    mean = np.array([m["mean"] for m in summary["macro"]]) * 100
    std = np.array([m["std"] for m in summary["macro"]]) * 100
    b1, b2 = mean[-1], mean[0]

    fig, ax = plt.subplots(figsize=(3.4, 2.5), dpi=300)
    ax.axhline(b1, ls=":", lw=1.0, color="#1f4e79", label="DINOv2 only")
    ax.axhline(b2, ls=":", lw=1.0, color="#c0504d", label="CLIP only")
    ax.errorbar(al, mean, yerr=std, marker="o", ms=4, lw=1.4, capsize=2.5,
                color="#333333", label="fused (mean $\\pm$ sd over cells)")
    ax.set_xlabel("fusion weight $\\alpha$ (DINOv2 share)")
    ax.set_ylabel("image AUROC (%)")
    ax.set_xticks(al)
    ax.legend(fontsize=5.8, frameon=False, loc="lower right")
    ax.tick_params(labelsize=6.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
