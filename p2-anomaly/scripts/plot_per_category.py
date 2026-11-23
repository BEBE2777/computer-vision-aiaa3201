"""Per-category figure for the E1 controlled comparison (report Figure 2).

Panel (a): 1-shot image AUROC per category, DINOv2 control vs CLIP, error bars = std over the
three reference seeds. Panel (b): the same comparison as the per-category gap CLIP - DINOv2
across all four shot counts, so the "gap does not close with more references" claim and the
one category where CLIP is ahead (pipe_fryum) are both visible at a glance.

Data source is `results/part3/metrics.json` (written by `make_tables.py`), so the figure can
never drift from the tables.

    python scripts/plot_per_category.py --metrics results/part3/metrics.json \
        --out results/part3/figs
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CATS = ["candle", "capsules", "cashew", "chewinggum", "fryum", "pipe_fryum"]
SHOTS = [1, 2, 4, 8]
DINO_KEY = "DINOv2 ViT-S/14 448, no mask (E1 control)"
CLIP_KEY = "CLIP ViT-B/16 448 block 9, no mask (Part 3)"
C_DINO, C_CLIP = "#2ca02c", "#d62728"  # same colours as figs/auroc_vs_shots.png


def series(metrics, key, metric="image_auroc", shot=1):
    row = metrics[key][metric][f"{shot}-shot"]
    y = np.array([row[c] for c in CATS]) * 100
    e = np.array([row[c + "_std"] for c in CATS]) * 100
    return y, e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="results/part3/metrics.json")
    ap.add_argument("--out", default="results/part3/figs")
    args = ap.parse_args()
    m = json.load(open(args.metrics))
    os.makedirs(args.out, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5))

    # ---- panel (a): 1-shot bars with seed std -------------------------------------
    ax = axes[0]
    y_d, e_d = series(m, DINO_KEY, shot=1)
    y_c, e_c = series(m, CLIP_KEY, shot=1)
    x = np.arange(len(CATS))
    w = 0.38
    ax.bar(x - w / 2, y_d, w, yerr=e_d, capsize=2.5, color=C_DINO, label="DINOv2 @448 (control)")
    ax.bar(x + w / 2, y_c, w, yerr=e_c, capsize=2.5, color=C_CLIP, label="CLIP @448 block 9")
    for xi, (d, c) in enumerate(zip(y_d, y_c)):
        gap = c - d
        ax.annotate(f"{gap:+.1f}", (xi, max(d, c) + 3.4), ha="center", fontsize=7,
                    color=C_CLIP if gap < 0 else "#1a7f37", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(CATS, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(70, 104)
    ax.set_ylabel("image AUROC (%)")
    ax.set_title("Per-category, 1-shot (mean ± std over seeds)", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7, loc="lower left")

    # ---- panel (b): the gap across shot counts ------------------------------------
    ax = axes[1]
    for i, (cat, color) in enumerate(zip(CATS, plt.cm.tab10.colors)):
        gaps = [series(m, CLIP_KEY, shot=s)[0][i] - series(m, DINO_KEY, shot=s)[0][i]
                for s in SHOTS]
        ax.plot(SHOTS, gaps, marker="o", ms=3.5, lw=1.2, color=color, label=cat)
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_xticks(SHOTS)
    ax.set_xticklabels([str(s) for s in SHOTS])
    ax.set_xlabel("references per category (shots)")
    ax.set_ylabel("CLIP − DINOv2 (image AUROC points)")
    ax.set_title("The gap per category, all shots", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6.5, ncol=2, loc="lower right")

    fig.tight_layout()
    p = os.path.join(args.out, "per_category_1shot.png")
    fig.savefig(p, dpi=170)
    print("wrote", p)

    # console read-out for the report text
    print("\nΔ (CLIP − DINOv2), image AUROC points, mean over seeds:")
    hdr = "category   " + "".join(f"{s}-shot   " for s in SHOTS)
    print(hdr)
    for i, cat in enumerate(CATS):
        row = [series(m, CLIP_KEY, shot=s)[0][i] - series(m, DINO_KEY, shot=s)[0][i]
               for s in SHOTS]
        print(f"{cat:<11}" + "".join(f"{g:+7.1f}   " for g in row))


if __name__ == "__main__":
    main()
