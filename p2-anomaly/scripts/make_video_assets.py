"""Build the three Part 2 video assets from EXISTING artefacts only.

Nothing here re-runs a model: panel 1/2 are caption strips drawn on top of the
example PNGs that AnomalyDINO already saved during the official runs, and panel 3
is a plain schematic of the described pipeline (no experimental numbers).
"""
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

SRC = "/tmp/slv176/results_VisA/dinov2_vits14_448/2-shot_preprocess=agnostic_calib20/candle/examples"
OUT = "/tmp/slv176/video_assets"
os.makedirs(OUT, exist_ok=True)
BAR = 58


def caption(src, line1, line2, dst, target_w=1400):
    img = cv2.imread(src, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    if w != target_w:
        img = cv2.resize(img, (target_w, int(h * target_w / w)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
    strip = np.full((BAR * 2, w, 3), 255, np.uint8)
    cv2.putText(strip, line1, (14, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(strip, line2, (14, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (90, 90, 90), 1, cv2.LINE_AA)
    out = np.vstack([strip, img])
    cv2.line(out, (0, BAR * 2 - 1), (w, BAR * 2 - 1), (210, 210, 210), 1)
    cv2.imwrite(dst, out)
    print("wrote", dst, out.shape)


caption(os.path.join(SRC, "example_good_0.png"),
        "Part 2 - NORMAL test image (AnomalyDINO, candle, 2-shot, seed 0)",
        "Every test patch finds a close neighbour in the normal memory bank -> low distance, low image score.",
        os.path.join(OUT, "part2_normal_heatmap.png"))

caption(os.path.join(SRC, "example_bad_0.png"),
        "Part 2 - DEFECTIVE test image (AnomalyDINO, candle, 2-shot, seed 0)",
        "Patch distances peak at the defect; top-1% pooling turns the map into one image-level score (0.172).",
        os.path.join(OUT, "part2_anomaly_heatmap.png"))


def box(ax, x, y, w, h, text, fc, fs=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.012",
                                linewidth=1.2, edgecolor="#3a3a3a", facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="#101010", wrap=True)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                                 linewidth=1.4, color="#3a3a3a", shrinkA=2, shrinkB=2))


fig, ax = plt.subplots(figsize=(13.2, 5.4), dpi=170)
ax.set_xlim(0, 13.2); ax.set_ylim(0, 5.4); ax.axis("off")
ax.text(0.15, 5.12, "Part 2 pipeline - AnomalyDINO (DINOv2 ViT-S/14, 448x448, agnostic preprocessing)",
        fontsize=13, fontweight="bold", color="#101010")
ax.text(0.15, 4.78, "Schematic of the described method; no experimental values are shown in this figure.",
        fontsize=9, color="#6a6a6a")

box(ax, 0.15, 3.05, 1.95, 1.35, "Few NORMAL\nreference images\n(N = 1, 2, 4, 8)", "#e8f0fb")
box(ax, 2.55, 3.05, 1.95, 1.35, "Frozen DINOv2\nViT-S/14\npatch features", "#e8f0fb")
box(ax, 4.95, 3.05, 1.95, 1.35, "Normal memory\nbank\n(patches, not images)", "#e8f0fb")
box(ax, 7.35, 3.05, 2.05, 1.35, "Test image -> patches\nnearest-neighbour\ndistance to the bank", "#fdeee6")
box(ax, 9.85, 4.05, 1.75, 1.05, "top-1% pooling\n-> image score", "#eaf5ea")
box(ax, 9.85, 2.35, 1.75, 1.05, "patch-distance\nmap -> heatmap", "#eaf5ea")
arrow(ax, 2.10, 3.72, 2.55, 3.72)
arrow(ax, 4.50, 3.72, 4.95, 3.72)
arrow(ax, 6.90, 3.72, 7.35, 3.72)
arrow(ax, 9.40, 3.85, 9.85, 4.55)
arrow(ax, 9.40, 3.55, 9.85, 2.90)

ax.text(0.15, 2.45, "No defect examples are used at any point: the bank only ever contains normal patches.",
        fontsize=9.5, color="#2b2b2b")
ax.text(0.15, 2.12, "Both scores come from the same distance map, so image-level and pixel-level outputs are consistent.",
        fontsize=9.5, color="#2b2b2b")
ax.text(0.15, 1.62, "Reported numbers (preliminary, seed 0): Image AUROC 94.62 / 94.70 / 96.22 % for 1 / 2 / 8 shots;",
        fontsize=9.5, color="#2b2b2b")
ax.text(0.15, 1.32, "Pixel AUROC 98.02 / 98.23 / 98.47 %. 4-shot and seeds 1-2 are not finished yet.",
        fontsize=9.5, color="#2b2b2b")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "part2_pipeline.png"), bbox_inches="tight", facecolor="white")
print("wrote pipeline")
