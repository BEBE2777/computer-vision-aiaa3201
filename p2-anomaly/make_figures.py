"""Report figures: AUROC-vs-shots curves and a hard-case panel (Part 1 vs Part 2).

    python make_figures.py --tables results/tables/metrics.json \
        --part2-run results_VisA/dinov2_vits14_448/1-shot_preprocess=agnostic_calib20 \
        --out results/figs

The hard-case panel picks test images that Part 2 still gets wrong (missed defects and false
alarms) and shows, for the same image, the ground-truth mask and both score maps. Part 1 maps
are recomputed here with the exact 1-shot reference set recorded for that (category, seed).
"""

import argparse
import json
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import p2common as C
import run_part1_template as p1


def plot_curves(metrics_path, out_dir):
    m = json.load(open(metrics_path))
    shots = [1, 2, 4, 8]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for ax, key, title in zip(axes, ("image_auroc", "pixel_auroc"),
                              ("Image-level AUROC", "Pixel-level AUROC")):
        for name, data in m.items():
            ys, es = [], []
            for s in shots:
                row = data.get(key, {}).get(f"{s}-shot")
                ys.append(row["MEAN"] * 100 if row else np.nan)
                es.append(row.get("MEAN_std", 0.0) * 100 if row else 0)
            ax.errorbar(shots, ys, yerr=es, marker="o", capsize=3, label=name)
        ax.set_xscale("log", base=2)
        ax.set_xticks(shots)
        ax.set_xticklabels([str(s) for s in shots])
        ax.set_xlabel("references per category (shots)")
        ax.set_ylabel("AUROC (%)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.axhline(50, color="k", lw=0.8, ls=":")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    p = os.path.join(out_dir, "auroc_vs_shots.png")
    fig.savefig(p, dpi=160)
    print("wrote", p)


def pick_hard_cases(run_dir, root, cats, seed=0, n=5):
    """Bad images that score below the good-image 95th percentile, spread over categories."""
    cases = []
    for cat in cats:
        scores, labels, items = [], [], []
        for label, img_path, mask_path in C.test_items(root, cat):
            stem = os.path.splitext(os.path.basename(img_path))[0]
            sub = "good" if label == 0 else "bad"
            npy = os.path.join(run_dir, "anomaly_maps", f"seed={seed}", cat, "test", sub, stem + ".npy")
            if not os.path.isfile(npy):
                continue
            scores.append(C.top1p_mean(np.load(npy).astype(np.float32).ravel()))
            labels.append(label)
            items.append((cat, label, img_path, mask_path, npy))
        scores, labels = np.array(scores), np.array(labels)
        if labels.max() == 0:
            continue
        thr = np.percentile(scores[labels == 0], 95)
        bad = np.where(labels == 1)[0]
        missed = bad[np.argsort(scores[bad])[:1]]          # lowest-scoring defects
        for i in missed:
            cases.append((items[i], float(scores[i]), thr, "missed defect"))
    return cases[:n]


def plot_qualitative(run_dir, root, out_dir, cats, seed=0):
    cases = pick_hard_cases(run_dir, root, cats, seed, n=5)
    if not cases:
        print("no hard cases found")
        return
    fig, axes = plt.subplots(len(cases), 4, figsize=(7.4, 1.85 * len(cases)))
    for r, ((cat, label, img_path, mask_path, npy), score, thr, kind) in enumerate(cases):
        _, pool = C.ref_pool(root, cat, 20)
        refs = [C.prep_square(C.load_rgb(os.path.join(C.train_dir(root, cat), n)))
                for n in C.shot_refs(pool, 1, seed)]
        img = C.prep_square(C.load_rgb(img_path))
        d1 = p1.score_one(img.copy(), refs, p1.gray01(refs[0]), False, False)
        d2 = cv2.resize(np.load(npy).astype(np.float32), (C.EVAL_SIZE,) * 2,
                        interpolation=cv2.INTER_LINEAR)
        gt = C.load_mask(mask_path, C.EVAL_SIZE)

        def norm(x):
            x = x - x.min()
            return x / max(float(x.max()), 1e-6)

        panels = [(img, "image", None), (np.stack([gt * 255] * 3, -1), "ground truth", None),
                  (cv2.applyColorMap((norm(d1) * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1],
                   f"Part 1 (score {p1.top1p_mean(d1):.3f})", None),
                  (cv2.applyColorMap((norm(d2) * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1],
                   f"Part 2 (score {score:.3f}, thr {thr:.3f})", None)]
        for c, (im, title, _) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(np.asarray(im).astype(np.uint8))
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=7)
            else:
                ax.set_xlabel(title if c >= 2 else "", fontsize=6)
            if c == 0:
                ax.set_ylabel(f"{cat}\n{kind}", fontsize=6)
    fig.tight_layout()
    p = os.path.join(out_dir, "hard_cases.png")
    fig.savefig(p, dpi=150)
    print("wrote", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="results/tables/metrics.json")
    ap.add_argument("--part2-run", default=None)
    ap.add_argument("--root", default="/tmp/slv176/data/VisA_pytorch/1cls")
    ap.add_argument("--out", default="results/figs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if os.path.isfile(args.tables):
        plot_curves(args.tables, args.out)
    else:
        print("no metrics.json, skipping curves")
    if args.part2_run and os.path.isdir(args.part2_run):
        plot_qualitative(args.part2_run, args.root, args.out, C.VISA_6, args.seed)


if __name__ == "__main__":
    main()
