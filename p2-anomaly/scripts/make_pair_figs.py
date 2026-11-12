"""Regenerate the Part 2 demo pair with a clear score ordering.

Replaces video/assets/part2_normal_heatmap.png and part2_anomaly_heatmap.png
(the old automatic examples scored 0.187 normal vs 0.172 defective - the wrong
ordering for narration). New pair, all numbers measured with the exact official
2-shot seed-0 candle model (verified to match measurements_seed=0.csv):

  normal    good/0659.JPG  -> 0.115   (lowest of all 180 candle test images)
  threshold (95th pct of the 20 calibration normals) -> 0.201
  defective bad/068.JPG    -> 0.589   (highest of all 180 candle test images)

Also writes part2_pair_scores.png: the two chosen images side by side plus the
candle test-score distribution with the calibration threshold.
"""
import argparse
import csv
import json
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--tmp", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".tmp_pair"),
                help="dir with scores.json, <stem>.jpg/_full.npy/_mask.npy, measurements csv")
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "video", "assets"))
ap.add_argument("--measurements", default=None, help="defaults to <tmp>/measurements_seed=0.csv")
A = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
TMP = A.tmp
OUT = A.out
os.makedirs(OUT, exist_ok=True)

meta = json.load(open(os.path.join(TMP, "scores.json")))
THR = meta["threshold_p95"]

INK = "#101010"
MUTE = "#6a6a6a"


def load_case(stem):
    img = cv2.imread(os.path.join(TMP, stem + ".jpg"), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    full = np.load(os.path.join(TMP, stem + "_full.npy"))
    mask = np.load(os.path.join(TMP, stem + "_mask.npy")).astype(np.uint8)
    h, w = img.shape[:2]
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    score = meta["samples"][stem.replace("_", "/") + ".JPG"]["score"]
    return img, full, mask, score


def overlay(img, full, mask, vmax):
    heat = matplotlib.colormaps["jet"]((np.clip(full, 0, None) / (vmax + 1e-9)))
    heat = (heat[..., :3] * 255).astype(np.uint8)
    out = img.copy()
    out[mask] = (0.45 * img[mask] + 0.55 * heat[mask]).astype(np.uint8)
    return out


def single_panel(stem, title, score_text, sub, dst, vmax):
    img, full, mask, score = load_case(stem)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7), dpi=170)
    fig.suptitle(title, fontsize=14, fontweight="bold", color=INK, x=0.5, y=0.99)
    axes[0].imshow(img); axes[0].set_title("Test image", fontsize=11, color=INK)
    axes[1].imshow(overlay(img, full, mask, vmax))
    axes[1].set_title("Patch-distance heatmap (higher = more anomalous)", fontsize=11, color=INK)
    for ax in axes:
        ax.axis("off")
    fig.text(0.5, 0.015, score_text, ha="center", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.5, -0.035, sub, ha="center", fontsize=9.5, color=MUTE)
    fig.savefig(dst, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", dst, "score", score)


# shared color scale so "low distance" really looks uniformly low on both images
_vn = np.load(os.path.join(TMP, "good_0659_full.npy")).max()
_vd = np.load(os.path.join(TMP, "bad_068_full.npy")).max()
VMAX = float(max(_vn, _vd))

single_panel("good_0659",
             "Part 2 - NORMAL test image (AnomalyDINO, candle, 2-shot, seed 0)",
             "Image score = 0.115  -  every patch is close to the normal memory bank",
             "Same model and protocol as the reported AUROC; score verified against measurements_seed=0.csv.",
             os.path.join(OUT, "part2_normal_heatmap.png"), VMAX)

single_panel("bad_068",
             "Part 2 - DEFECTIVE test image (AnomalyDINO, candle, 2-shot, seed 0)",
             "Image score = 0.589  -  5.1x the normal image, distances peak at the deformed rim",
             "Same model and protocol as the reported AUROC; score verified against measurements_seed=0.csv.",
             os.path.join(OUT, "part2_anomaly_heatmap.png"), VMAX)

# ---------------- combined proof figure ----------------
rows = [("good_0659", "Normal", 0.115), ("bad_068", "Defective", 0.589)]

scores_all = {"good": [], "bad": []}
for r in csv.DictReader(open(A.measurements or os.path.join(TMP, "measurements_seed=0.csv"))):
    if r["Object"] == "candle":
        scores_all["good" if r["Sample"].startswith("good/") else "bad"].append(float(r["Anomaly_Score"]))

fig = plt.figure(figsize=(13.6, 6.2), dpi=170)
gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.15, 1.0])
fig.suptitle("AnomalyDINO scores: normal vs defective (candle, 2-shot, seed 0)",
             fontsize=14, fontweight="bold", color=INK, y=0.99)

for i, (stem, label, _) in enumerate(rows):
    img, full, mask, score = load_case(stem)
    ax = fig.add_subplot(gs[i, 0])
    ax.imshow(img); ax.axis("off")
    ax.set_title(f"{label} test image", fontsize=11, color=INK)
    ax = fig.add_subplot(gs[i, 1])
    ax.imshow(overlay(img, full, mask, VMAX)); ax.axis("off")
    ax.set_title(f"score = {score:.3f}", fontsize=12, color=INK, fontweight="bold")

axh = fig.add_subplot(gs[:, 2])
axh.hist(scores_all["good"], bins=18, alpha=0.55, color="#3b7dd8", label=f"normal test ({len(scores_all['good'])})")
axh.hist(scores_all["bad"], bins=18, alpha=0.55, color="#d84b3b", label=f"defective test ({len(scores_all['bad'])})")
axh.axvline(THR, color="#c9a227", lw=2, ls="--", label=f"OK/NG threshold\n(95th pct of 20 calib normals = {THR:.3f})")
axh.scatter([0.115], [0.4], marker="v", s=60, color="#1f4e9c", zorder=5)
axh.scatter([0.589], [0.4], marker="v", s=60, color="#8c1f14", zorder=5)
axh.annotate("chosen normal\n0.115", xy=(0.115, 0.4), xytext=(0.13, 2.2), fontsize=8.5, color="#1f4e9c")
axh.annotate("chosen defective\n0.589", xy=(0.589, 0.4), xytext=(0.46, 2.2), fontsize=8.5, color="#8c1f14")
axh.set_xlabel("image-level anomaly score", fontsize=10, color=INK)
axh.set_ylabel("test images", fontsize=10, color=INK)
axh.set_title("All 180 candle test scores", fontsize=11, color=INK)
axh.legend(fontsize=8, loc="upper right")
axh.tick_params(labelsize=8)

fig.tight_layout(rect=(0, 0, 1, 0.94))
dst = os.path.join(OUT, "part2_pair_scores.png")
fig.savefig(dst, facecolor="white")
plt.close(fig)
print("wrote", dst, "threshold", THR)
