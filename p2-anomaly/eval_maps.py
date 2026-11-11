"""Pixel-level (and image-level) metrics for one official AnomalyDINO shot directory.

The official `src/post_eval.py` pools full-resolution maps, which needs ~11 GB of RAM for the
960 VisA test images and got OOM-killed on the shared server. This script instead reads the
saved patch-distance maps (`anomaly_maps/seed=<s>/<cat>/test/<type>/*.npy`), upsamples them to
448x448, and scores them with exactly the same code used for the template baseline and the
Part 3 variants (`src/p2common.py`). One protocol for every method.

Usage:
  python eval_maps.py --run results_VisA/dinov2_vits14_448/1-shot_preprocess=agnostic_calib20
"""

import argparse
import os
import re
import sys

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import p2common as C


def map448(npy, size=C.EVAL_SIZE):
    a = np.load(npy).astype(np.float32)
    a = cv2.resize(a, (size, size), interpolation=cv2.INTER_LINEAR)
    return gaussian_filter(a, sigma=4)


def eval_shot_dir(run_dir, root, cats=None):
    cats = cats or C.VISA_6
    seeds = sorted(int(re.search(r"seed=(\d+)", d).group(1))
                   for d in os.listdir(os.path.join(run_dir, "anomaly_maps"))
                   if d.startswith("seed="))
    shot = int(re.match(r"(\d+)-shot_", os.path.basename(run_dir.rstrip("/"))).group(1))
    per_seed = {}
    for seed in seeds:
        map_root = os.path.join(run_dir, "anomaly_maps", f"seed={seed}")
        per_cat = {}
        for cat in cats:
            scores, labels, maps, gts = [], [], [], []
            for label, img_path, mask_path in C.test_items(root, cat):
                stem = os.path.splitext(os.path.basename(img_path))[0]
                sub = "good" if label == 0 else "bad"
                npy = os.path.join(map_root, cat, "test", sub, stem + ".npy")
                if not os.path.isfile(npy):
                    continue
                scores.append(C.top1p_mean(np.load(npy).astype(np.float32).ravel()))
                labels.append(label)
                maps.append(map448(npy))
                gts.append(C.load_mask(mask_path, C.EVAL_SIZE))
            if maps:
                per_cat[cat] = {"n": len(labels),
                                "image_auroc": C.image_auroc(scores, labels),
                                "pixel_auroc": C.pooled_pixel_auroc(maps, gts)}
        per_seed[str(seed)] = per_cat
    return shot, per_seed


def summarise(shot, per_seed, cats=None):
    cats = cats or C.VISA_6
    final = {"image_auroc": {}, "pixel_auroc": {}}
    for metric in final:
        row = {}
        for cat in cats:
            vals = [per_seed[s][cat][metric] for s in per_seed if cat in per_seed[s]]
            row[cat] = float(np.mean(vals)) if vals else float("nan")
            row[cat + "_std"] = float(np.std(vals)) if vals else float("nan")
        row["MEAN"] = C.mean_over_categories({c: row[c] for c in cats})
        row["MEAN_std"] = float(np.mean([row[c + "_std"] for c in cats]))
        final[metric][f"{shot}-shot"] = row
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data_root", default="/tmp/slv176/data/VisA_pytorch/1cls")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_dir = args.run.rstrip("/")
    shot, per_seed = eval_shot_dir(run_dir, args.data_root)
    final = summarise(shot, per_seed)
    out = args.out or os.path.join(os.path.dirname(run_dir), "eval448_%dshot.json" % shot)
    C.jdump({"n": {c: per_seed[str(list(per_seed)[0])][c]["n"] for c in C.VISA_6}, **final}, out)
    print("wrote", out)
    for metric, d in final.items():
        for sk, row in d.items():
            print(f"{metric} {sk}: " + " ".join(f"{c}={row[c]:.4f}" for c in C.VISA_6)
                  + f"  MEAN={row['MEAN']:.4f}")


if __name__ == "__main__":
    main()
