"""Draw the method pipeline as a vector figure for the CVPR report.

The figure has to carry four facts at once, because they are what the method section
claims: (1) the support path builds a memory bank from K references and their
rotations; (2) the query path goes through the *same* frozen encoder and is scored by
1-NN; (3) the two encoders (DINOv2, and CLIP for E1) are interchangeable arms sharing
everything else; (4) fusion with the multimodal branch is a separate, optional module.

Outputs (both, so the paper can use the PDF and the review can use the PNG):
    docs/part3/report/figures/pipeline.pdf
    p2-anomaly/results/part3/figs/pipeline.png

Everything is drawn as text + patches: no dataset image is embedded, so no number or
appearance in the figure can drift away from the results tables.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(ROOT)

plt.rcParams.update({
    "pdf.fonttype": 42,          # vector text, editable and searchable in the PDF
    "ps.fonttype": 42,
    "font.family": "DejaVu Sans",
    "font.size": 6.6,
    "axes.linewidth": 0,
})

W, H = 100.0, 46.0
EDGE = "#3C3C3C"
DARK = "#1A1A1A"
GREY = "#6E6E6E"
BLUE_FC, BLUE_EC = "#E8EEF6", "#2F4F73"
WARM_FC, WARM_EC = "#FBF1E4", "#8A6A3A"
NEUTRAL_FC = "#F4F4F4"


def box(ax, x, y, w, h, lines, fc="white", ec=EDGE, ls="-", fs=6.6, lw=0.8,
        color=DARK, weight="normal", zorder=3, rounding=0.9, va="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={rounding}",
                                linewidth=lw, facecolor=fc, edgecolor=ec,
                                linestyle=ls, zorder=zorder, mutation_aspect=1.0))
    yy = y + h / 2 if va == "center" else y + h - 1.15
    ax.text(x + w / 2, yy, lines, ha="center",
            va="center" if va == "center" else "top", fontsize=fs, color=color,
            linespacing=1.45, zorder=zorder + 1, fontweight=weight)


def arrow(ax, p0, p1, ls="-", color=EDGE, lw=0.8, style="-|>", ms=6.5, zorder=2,
          connectionstyle="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms,
                                 linewidth=lw, color=color, linestyle=ls,
                                 shrinkA=0, shrinkB=0, zorder=zorder,
                                 connectionstyle=connectionstyle))


def path(ax, pts, ls="-", color=EDGE, lw=0.8, head=True):
    """Polyline with an arrow head on the last segment (orthogonal routing)."""
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, ls=ls,
                solid_capstyle="butt", zorder=2)
    arrow(ax, pts[-2], pts[-1], ls=ls, color=color, lw=lw,
          style="-|>" if head else "-")


def label(ax, x, y, text, fs=5.9, color=GREY, ha="center", va="center", bg=True,
          rot=0):
    ax.text(x, y, text, fontsize=fs, color=color, ha=ha, va=va, rotation=rot,
            zorder=6, linespacing=1.3,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.95)
            if bg else None)


def main():
    fig = plt.figure(figsize=(7.0, 3.42))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---------------- geometry (columns / bands)
    c1x, c1w = 1.0, 13.2
    c2x, c2w = 16.6, 24.4
    c3x, c3w = 43.4, 17.4
    c4x, c4w = 62.8, 16.4
    c5x, c5w = 82.0, 17.0
    top_h, mid_h, low_h = 9.0, 9.0, 9.0
    y_top, y_mid, y_low = 33.4, 20.6, 4.4

    # ---------------- support path
    box(ax, c1x, y_top, c1w, top_h,
        "K normal references\n$K\\in\\{1,2,4,8\\}$\n$\\times$8 rotations each", fs=6.2)
    box(ax, c3x, y_top, c3w, top_h,
        "memory bank\nreference patch tokens\nL2-normalised", fc=NEUTRAL_FC)
    arrow(ax, (c1x + c1w, y_top + top_h / 2), (c2x, y_top + top_h / 2))
    arrow(ax, (c2x + c2w, y_top + top_h / 2), (c3x, y_top + top_h / 2))

    # ---------------- encoder: one container, two interchangeable arms
    enc_y, enc_h = y_mid, y_top + top_h - y_mid
    ax.add_patch(FancyBboxPatch((c2x, enc_y), c2w, enc_h,
                                boxstyle="round,pad=0,rounding_size=0.9",
                                linewidth=0.9, facecolor=BLUE_FC, edgecolor=BLUE_EC,
                                zorder=2))
    ax.text(c2x + c2w / 2, enc_y + enc_h - 1.25,
            "frozen visual encoder", ha="center", va="center",
            fontsize=6.5, color=BLUE_EC, fontweight="bold", zorder=4)
    ax.text(c2x + c2w / 2, enc_y + enc_h - 2.85,
            "no training $\\cdot$ shared weights",
            ha="center", va="center", fontsize=5.7, color=BLUE_EC, zorder=4)
    box(ax, c2x + 1.0, 31.6, c2w - 2.0, 6.2,
        "DINOv2 ViT-S/14 @ 448\n32$\\times$32 patch tokens",
        fc="white", ec=BLUE_EC, lw=0.7, fs=6.3)
    box(ax, c2x + 1.0, 22.0, c2w - 2.0, 6.2,
        "CLIP ViT-B/16 @ 448, block 9\n28$\\times$28 patch tokens  (E1 arm)",
        fc="white", ec=BLUE_EC, lw=0.7, fs=6.3)
    ax.text(c2x + c2w / 2, 30.2, "or", ha="center", va="center", fontsize=5.8,
            color=GREY, style="italic", zorder=4)

    # ---------------- query path
    box(ax, c1x, y_mid, c1w, mid_h, "1 test image", fs=6.5)
    box(ax, c3x, y_mid, c3w, mid_h,
        "1-NN cosine distance\nquery patch $\\rightarrow$\nnearest bank entry")
    arrow(ax, (c1x + c1w, y_mid + mid_h / 2), (c2x, y_mid + mid_h / 2))
    arrow(ax, (c2x + c2w, y_mid + mid_h / 2), (c3x, y_mid + mid_h / 2))
    arrow(ax, (c3x + c3w / 2, y_top), (c3x + c3w / 2, y_mid + mid_h))

    # ---------------- distance map -> score
    box(ax, c3x, y_low, c3w, low_h,
        "patch distance map\n$d\\in\\mathbb{R}^{32\\times32}$")
    arrow(ax, (c3x + c3w / 2, y_mid), (c3x + c3w / 2, y_low + low_h))
    box(ax, c4x, y_low, c4w, low_h,
        "top-1% mean pooling\n$\\rightarrow$ image score $s$")
    arrow(ax, (c3x + c3w, y_low + low_h / 2), (c4x, y_low + low_h / 2))

    # ---------------- decision
    box(ax, c5x, y_low, c5w, low_h,
        "OK / NG at $\\tau$\nimage AUROC", fc=WARM_FC, ec=WARM_EC)
    arrow(ax, (c4x + c4w, y_low + low_h / 2), (c5x, y_low + low_h / 2))

    # threshold source: 20 calibration images, routed below everything
    box(ax, c1x, y_low, c1w, low_h,
        "20 held-out\nnormal images\n(calibration only)", fs=6.3, fc=NEUTRAL_FC)
    path(ax, [(c1x + c1w / 2, y_low), (c1x + c1w / 2, 1.7),
              (c5x + c5w - 2.6, 1.7), (c5x + c5w - 2.6, y_low)],
         ls=(0, (3.2, 1.6)), color=GREY, lw=0.7)
    label(ax, (c3x + c3w / 2), 2.65, "$\\tau$ = 95th percentile of calibration scores",
          fs=5.7, color=GREY)

    # ---------------- multimodal branch + fusion (optional, dashed)
    box(ax, c5x, y_mid, c5w, mid_h,
        "VLM branch (E2)\nQwen-VL: refs + test\n$\\rightarrow$ verdict, $s_{\\mathrm{VLM}}$",
        fc="white", ec=GREY, ls=(0, (3.2, 1.6)), fs=6.1)
    box(ax, c4x, y_mid, c4w, mid_h,
        "score fusion (E3/E5)\n$s=\\alpha z_{\\mathrm{vis}}+(1-\\alpha)z_{\\mathrm{VLM}}$\n"
        "$\\alpha=0.5$, pre-registered",
        fc="white", ec=GREY, ls=(0, (3.2, 1.6)), fs=6.1)
    arrow(ax, (c5x, y_mid + mid_h / 2), (c4x + c4w, y_mid + mid_h / 2),
          ls=(0, (3.2, 1.6)), color=GREY, lw=0.7)
    arrow(ax, (c4x + c4w / 2 - 3.0, y_low + low_h), (c4x + c4w / 2 - 3.0, y_mid),
          ls=(0, (3.2, 1.6)), color=GREY, lw=0.7)
    label(ax, c4x + c4w / 2 - 3.0 - 2.5, (y_low + low_h + y_mid) / 2, "$s$", fs=6.0)
    path(ax, [(c4x + c4w / 2 + 3.4, y_mid), (c4x + c4w / 2 + 3.4, 16.7),
              (c5x + c5w - 6.6, 16.7), (c5x + c5w - 6.6, y_low + low_h)],
         ls=(0, (3.2, 1.6)), color=GREY, lw=0.7)

    # ---------------- notes
    ax.text(0.8, H - 0.7, "solid: evaluated in this report        "
                          "dashed: multimodal branch / optional module",
            fontsize=5.6, color=GREY, ha="left", va="top")

    out_pdf = os.path.join(REPO, "docs", "part3", "report", "figures", "pipeline.pdf")
    out_png = os.path.join(ROOT, "results", "part3", "figs", "pipeline.png")
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_pdf, format="pdf")
    fig.savefig(out_png, dpi=300)
    print("wrote", out_pdf)
    print("wrote", out_png)


if __name__ == "__main__":
    main()
