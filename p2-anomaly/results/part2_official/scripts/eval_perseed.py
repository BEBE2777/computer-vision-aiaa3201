"""Per-(shot, seed) image/pixel AUROC at 448 for the official AnomalyDINO runs.

Same map pipeline as the project's eval_maps.py (npy -> 448 bilinear -> gaussian sigma 4;
image score = mean of top 1% of the raw patch-distance map), but evaluated one seed at a
time so that every seed of the 4x3 grid can be reported separately.

Writes one json per cell plus a flat CSV and a shot-level summary CSV.
"""

import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

SRC = "/data/disk1/home/slv176/aiaa3201_p2/src"
sys.path.insert(0, SRC)
import p2common as C  # noqa: E402

TREES = ["/tmp/qzheng610_p2/results_VisA/dinov2_vits14_448",
         "/tmp/slv176/results_VisA/dinov2_vits14_448"]
SHOTS = [1, 2, 4, 8]
SEEDS = [0, 1, 2]


def map448(npy, size=C.EVAL_SIZE):
    a = np.load(npy).astype(np.float32)
    a = cv2.resize(a, (size, size), interpolation=cv2.INTER_LINEAR)
    return gaussian_filter(a, sigma=4)


def find_run(shot, seed):
    """All run dirs that hold anomaly_maps/seed=<seed> for this shot count."""
    hits = []
    for t in TREES:
        for d in sorted(glob.glob(os.path.join(t, "%d-shot_preprocess=agnostic_*" % shot))):
            if not os.path.isdir(d):
                continue
            if os.path.isdir(os.path.join(d, "anomaly_maps", "seed=%d" % seed)):
                hits.append(d)
    # prefer the runs made in this session, then the shortest path
    hits.sort(key=lambda p: (0 if "calib20r2" in os.path.basename(p) else 1, len(p), p))
    return hits


def eval_cell(run_dir, shot, seed, root, cats):
    map_root = os.path.join(run_dir, "anomaly_maps", "seed=%d" % seed)
    out = {}
    for cat in cats:
        scores, labels, maps, gts = [], [], [], []
        for label, img, mask in C.test_items(root, cat):
            stem = os.path.splitext(os.path.basename(img))[0]
            sub = "good" if label == 0 else "bad"
            npy = os.path.join(map_root, cat, "test", sub, stem + ".npy")
            if not os.path.isfile(npy):
                continue
            scores.append(C.top1p_mean(np.load(npy).astype(np.float32).ravel()))
            labels.append(label)
            maps.append(map448(npy))
            gts.append(C.load_mask(mask, C.EVAL_SIZE))
        if not maps:
            continue
        out[cat] = {"n": len(labels),
                    "image_auroc": C.image_auroc(scores, labels),
                    "pixel_auroc": C.pooled_pixel_auroc(maps, gts)}
    return out


def check_refs(run_dir, shot, seed, root, cats):
    """Compare the dumped reference list with the deterministic slice rule."""
    bad = []
    for cat in cats:
        p = os.path.join(run_dir, "refs", "%s_shot=%d_seed=%d.json" % (cat, shot, seed))
        if not os.path.isfile(p):
            bad.append((cat, "missing refs dump"))
            continue
        dumped = json.load(open(p))
        _, pool = C.ref_pool(root, cat, calib_exclude=dumped.get("calib_exclude", 20))
        if dumped["reference_files"] != C.shot_refs(pool, shot, seed):
            bad.append((cat, "refs mismatch"))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/tmp/slv176/data/VisA_pytorch/1cls")
    ap.add_argument("--out", default="/tmp/qzheng610_p2/agg")
    ap.add_argument("--shots", type=int, nargs="+", default=SHOTS)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows, cells, ref_problems = [], {}, {}
    for shot in args.shots:
        for seed in args.seeds:
            hits = find_run(shot, seed)
            if not hits:
                print("MISSING shot=%d seed=%d" % (shot, seed))
                continue
            run_dir = hits[0]
            per_cat = eval_cell(run_dir, shot, seed, args.root, C.VISA_6)
            if len(per_cat) != len(C.VISA_6):
                print("INCOMPLETE shot=%d seed=%d (%d/%d cats) in %s"
                      % (shot, seed, len(per_cat), len(C.VISA_6), run_dir))
            mean_img = C.mean_over_categories({c: v["image_auroc"] for c, v in per_cat.items()})
            mean_px = C.mean_over_categories({c: v["pixel_auroc"] for c, v in per_cat.items()})
            cell = {"shot": shot, "seed": seed, "run_dir": run_dir,
                    "mean_image_auroc": mean_img, "mean_pixel_auroc": mean_px,
                    "per_category": per_cat}
            cells["%d-%d" % (shot, seed)] = cell
            C.jdump(cell, os.path.join(args.out, "cell_%dshot_seed%d.json" % (shot, seed)))
            for cat in C.VISA_6:
                if cat in per_cat:
                    rows.append([shot, seed, cat, "%.6f" % per_cat[cat]["image_auroc"],
                                 "%.6f" % per_cat[cat]["pixel_auroc"]])
            rows.append([shot, seed, "MEAN", "%.6f" % mean_img, "%.6f" % mean_px])
            bad = check_refs(run_dir, shot, seed, args.root, C.VISA_6)
            if bad:
                ref_problems["%d-%d" % (shot, seed)] = bad
            print("shot=%d seed=%d image=%.4f pixel=%.4f  (%s)"
                  % (shot, seed, mean_img, mean_px, os.path.basename(run_dir)))

    with open(os.path.join(args.out, "perseed_448.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["shot", "seed", "category", "image_auroc", "pixel_auroc"])
        w.writerows(rows)

    summ = []
    for shot in args.shots:
        for metric in ("image_auroc", "pixel_auroc"):
            vals = [cells["%d-%d" % (shot, s)]["mean_%s" % metric]
                    for s in args.seeds if "%d-%d" % (shot, s) in cells]
            if vals:
                summ.append([shot, metric, len(vals), "%.4f" % (100 * float(np.mean(vals))),
                             "%.4f" % (100 * float(np.std(vals)))])
    with open(os.path.join(args.out, "summary_448.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["shot", "metric", "n_seeds", "mean_pct", "std_pct"])
        w.writerows(summ)

    C.jdump({"cells": cells, "ref_problems": ref_problems,
             "summary": summ}, os.path.join(args.out, "summary_448.json"))
    print("\n== summary (mean +- std over seeds, %) ==")
    for r in summ:
        print("%2d-shot  %-12s n=%d  %.2f +- %.2f" % (r[0], r[1], r[2], float(r[3]), float(r[4])))
    if ref_problems:
        print("\n!! refs problems:", ref_problems)


if __name__ == "__main__":
    main()
