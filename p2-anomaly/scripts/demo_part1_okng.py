"""Part 1 OK/NG decision demo -- single lightweight export, CPU only.

Reproduces the EXISTING Part 1 template-difference baseline configuration
(align=none, no fg-crop, no gradient term) for one (category, shot, seed):
  - same reference filenames (first 20 normal training files excluded, seed slice);
  - same preprocessing (shorter edge -> 448, centre-crop 448x448);
  - same score rule (Gaussian-smoothed diff map, image score = mean of top 1%);
  - same threshold rule (95th percentile of the 20 calibration scores).

Self-checks before exporting:
  * recomputed threshold must match the value stored in the committed
    results/part1_template/summary.json for this config (1e-9 tolerance);
  * recomputed image AUROC over the full test split must match the stored
    image_auroc value (1e-6 tolerance).

Then it exports one normal and one defective test image chosen by a DETERMINISTIC
rule that does not look at anomaly scores at all: the first normal and the first
defective item in test_items order (filename-sorted within each ground-truth
group). Any misjudgment of these fixed samples is kept and reported honestly.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import run_part1_template as P  # noqa: E402  (score_one, prep helpers)
from p2common import (EVAL_SIZE, image_auroc, load_rgb, percentile_threshold,  # noqa: E402
                      prep_square, ref_pool, shot_refs, test_items, top1p_mean)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/tmp/slv176/data/VisA_pytorch/1cls")
    ap.add_argument("--category", default="chewinggum")
    ap.add_argument("--shot", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calib_exclude", type=int, default=20)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--expected_threshold", type=float, default=None,
                    help="value committed in results/part1_template/summary.json")
    ap.add_argument("--expected_auroc", type=float, default=None,
                    help="image_auroc committed in results/part1_template/summary.json")
    ap.add_argument("--summary_json", default=None,
                    help="path to summary.json for the self-check values")
    args = ap.parse_args()

    if args.summary_json and (args.expected_threshold is None or args.expected_auroc is None):
        s = json.load(open(args.summary_json))
        args.expected_threshold = s["threshold"][f"{args.shot}-shot"][args.category][str(args.seed)]
        args.expected_auroc = s["image_auroc"][f"{args.shot}-shot"][args.category][str(args.seed)]

    root, cat, shot, seed = args.data_root, args.category, args.shot, args.seed
    os.makedirs(args.outdir, exist_ok=True)

    calib_names, pool = ref_pool(root, cat, args.calib_exclude)
    ref_names = shot_refs(pool, shot, seed)
    print("reference_files:", ref_names)
    print("calibration files:", len(calib_names))

    # optional cross-check against the Part 2 refs dump (same sampling rule)
    refs_json = os.path.join(os.path.dirname(args.outdir), "refs",
                             f"{cat}_shot={shot}_seed={seed}.json")
    if os.path.exists(refs_json):
        official = json.load(open(refs_json)).get("reference_files")
        if official and list(official) != list(ref_names):
            raise SystemExit(f"reference mismatch vs {refs_json}: {official} != {ref_names}")
        print("reference filenames match official refs dump:", refs_json)

    ref_dir = os.path.join(root, cat, "train", "good")
    refs = [prep_square(load_rgb(os.path.join(ref_dir, n))) for n in ref_names]
    anchor_gray = P.gray01(refs[0])

    calib_scores = [top1p_mean(P.score_one(prep_square(load_rgb(os.path.join(ref_dir, n))),
                                          refs, anchor_gray, False, False))
                    for n in calib_names]
    thr = percentile_threshold(calib_scores, 95.0)
    print("threshold = %.9f" % thr)
    if args.expected_threshold is not None:
        assert abs(thr - args.expected_threshold) < 1e-9, \
            f"threshold drift: {thr} vs {args.expected_threshold}"

    # full test split -- needed anyway for the AUROC self-check
    items = test_items(root, cat)
    scores, labels, names_all = [], [], []
    for label, img_path, _ in items:
        im = prep_square(load_rgb(img_path))
        s = top1p_mean(P.score_one(im, refs, anchor_gray, False, False))
        scores.append(s)
        labels.append(label)
        names_all.append(os.path.relpath(img_path, root).replace("\\", "/"))
    au = image_auroc(np.asarray(scores), np.asarray(labels))
    print("image AUROC = %.6f" % au)
    if args.expected_auroc is not None:
        assert abs(au - args.expected_auroc) < 1e-6, \
            f"AUROC drift: {au} vs {args.expected_auroc}"

    # deterministic, score-independent selection: first normal and first defective
    # item in test_items order (filename-sorted within each ground-truth group).
    # Labels are used ONLY to identify the two groups, never to compare scores.
    ni = next(i for i, l in enumerate(labels) if l == 0)
    di = next(i for i, l in enumerate(labels) if l == 1)
    print("chosen normal:", names_all[ni], scores[ni])
    print("chosen defective:", names_all[di], scores[di])

    # export the two maps/images (recompute to keep memory small)
    samples = []
    vmax = 0.0
    maps = {}
    for idx, gt in ((ni, "normal"), (di, "defective")):
        im = prep_square(load_rgb(os.path.join(root, names_all[idx])))
        m = P.score_one(im, refs, anchor_gray, False, False)
        maps[gt] = m
        vmax = max(vmax, float(m.max()))
        s = top1p_mean(m)
        samples.append({
            "filename": names_all[idx],
            "ground_truth": gt,
            "score": round(s, 6),
            "prediction": "NG" if s > thr else "OK",
            "input_image": f"results/figs/part1_okng_demo_{cat}_{shot}shot_seed{seed}/{gt}_input.jpg",
            "heatmap": f"results/figs/part1_okng_demo_{cat}_{shot}shot_seed{seed}/{gt}_heatmap.png",
        })
        print(f"[{gt}] {names_all[idx]} score={s:.4f} -> {samples[-1]['prediction']}")

    figdir = os.path.join(args.outdir, f"figs_{cat}_{shot}shot_seed{seed}")
    os.makedirs(figdir, exist_ok=True)
    for gt in ("normal", "defective"):
        s = samples[0] if gt == "normal" else samples[1]
        im = prep_square(load_rgb(os.path.join(root, s["filename"])))
        cv2.imwrite(os.path.join(figdir, gt + "_input.jpg"),
                    cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        m = maps[gt]
        fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=170)
        imh = ax.imshow(m, cmap="inferno", vmin=0.0, vmax=vmax, interpolation="nearest")
        ax.set_title(f"{gt} | score={s['score']:.3f}\nthr={thr:.3f} -> {s['prediction']}",
                     fontsize=10, color="#101010")
        ax.axis("off")
        fig.colorbar(imh, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, gt + "_heatmap.png"), facecolor="white")
        plt.close(fig)

    # 4-panel PPT-ready figure, shared colour scale (no per-image normalisation)
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 4.3), dpi=170)
    fig.suptitle(f"Part 1 template-difference baseline - {cat}, {shot}-shot, seed {seed}"
                 f"   |   threshold = {thr:.3f} (95th pct of 20 calibration normals)"
                 f"   |   score > threshold -> NG",
                 fontsize=12, fontweight="bold", color="#101010", y=1.02)
    for k, gt in enumerate(("normal", "defective")):
        s = samples[0] if gt == "normal" else samples[1]
        im = prep_square(load_rgb(os.path.join(root, s["filename"])))
        axes[2 * k].imshow(im)
        axes[2 * k].set_title(f"{gt} input\n{s['filename']}", fontsize=9.5, color="#101010")
        axes[2 * k].axis("off")
        axes[2 * k + 1].imshow(maps[gt], cmap="inferno", vmin=0.0, vmax=vmax,
                               interpolation="nearest")
        ok_ng = "OK" if s["prediction"] == "OK" else "NG"
        col = "#1a7f37" if ok_ng == "OK" else "#c62828"
        axes[2 * k + 1].set_title(f"heatmap  |  score = {s['score']:.3f}\n"
                                  f"{'<' if ok_ng == 'OK' else '>'} thr {thr:.3f}  ->  ",
                                  fontsize=9.5, color="#101010")
        axes[2 * k + 1].text(0.96, 0.05, ok_ng, transform=axes[2 * k + 1].transAxes,
                             ha="right", va="bottom", fontsize=15, fontweight="bold", color=col,
                             bbox=dict(facecolor="white", alpha=0.85, edgecolor="none",
                                       boxstyle="round,pad=0.22"))
        axes[2 * k + 1].axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    grid_png = os.path.join(args.outdir, f"part1_okng_demo_{cat}_{shot}shot_seed{seed}.png")
    fig.savefig(grid_png, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote", grid_png)

    out = {
        "category": cat,
        "shot": shot,
        "seed": seed,
        "method": "Part 1 template-difference baseline (align=none, no fg-crop, no gradient term)",
        "score_rule": "mean of top 1% values in the Gaussian-smoothed (sigma=4) "
                      "0.5*|dI| + 0.5*(1-SSIM) difference map vs the closest reference",
        "threshold_rule": "95th percentile over 20 held-out normal calibration images",
        "sample_selection_rule": "first normal and first defective test item in "
                                 "deterministic filename order; selected independently "
                                 "of anomaly score",
        "threshold": round(thr, 6),
        "reference_files": ref_names,
        "calibration_files": calib_names,
        "preprocessing": "shorter edge resized to 448 px, centre-cropped to 448x448",
        "self_check": {
            "recomputed_image_auroc_full_test_split": round(au, 6),
            "committed_image_auroc": args.expected_auroc,
            "committed_threshold": args.expected_threshold,
        },
        "samples": samples,
    }
    jpath = os.path.join(args.outdir, f"demo_{cat}_{shot}shot_seed{seed}.json")
    json.dump(out, open(jpath, "w"), indent=2)
    print("wrote", jpath)


if __name__ == "__main__":
    main()
