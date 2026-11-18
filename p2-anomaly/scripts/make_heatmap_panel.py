"""Heatmap panel: same test image scored by DINOv2 and CLIP under the frozen protocol.

Reads two `dump_scores.py` outputs (each carrying `map_files`), keeps the images that at least
one backbone misjudges, and draws

    [ image | ground truth | DINOv2 score map | CLIP score map ]

with each map's top-1%-mean score and that backbone's own calibration threshold in the label, so
"score > thr" can be read off directly. All maps are resampled to 448x448 exactly as in scoring,
and the RGB panel is squashed to the same box, so the overlay geometry matches even for
non-square inputs.

Usage
-----
python scripts/make_heatmap_panel.py \
    --a scores/e1_dino448_nomask.json --a-name "DINOv2 ViT-S/14 @448" \
    --b scores/e1_clip448_L9_nomask.json --b-name "CLIP ViT-B/16 @448 (block 9)" \
    --root /tmp/slv176/data/VisA_pytorch/1cls \
    --out /tmp/slv176/grid/figs/heatmap_panel.png --per-cat 1
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

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
import p2common as C


def load(path):
    d = json.load(open(path))
    maps = {}
    mdir = os.path.join(os.path.dirname(path), "maps")
    for cat, rec in d["categories"].items():
        for key, info in rec.get("map_files", {}).items():
            p = os.path.join(mdir, os.path.basename(path).replace(".json", ""),
                             f"{cat}__{key.split('/')[0]}_{key.split('/')[1]}.npy")
            if os.path.isfile(p):
                maps[(cat, key)] = (np.load(p).astype(np.float32), info)
    return d, maps


def norm(x):
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / max(hi - lo, 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--a-name", default="DINOv2")
    ap.add_argument("--b-name", default="CLIP")
    ap.add_argument("--root", default="/tmp/slv176/data/VisA_pytorch/1cls")
    ap.add_argument("--out", default="heatmap_panel.png")
    ap.add_argument("--per-cat", type=int, default=1)
    ap.add_argument("--cats", nargs="*", default=None)
    args = ap.parse_args()

    da, ma = load(args.a)
    db, mb = load(args.b)
    cats = args.cats or [c for c in C.VISA_6 if c in da["categories"] and c in db["categories"]]

    rows = []
    for cat in cats:
        keys = [k for k in ma if k[0] == cat and k in mb]
        # prefer images both backbones agree are hard, then the rest
        both_wrong = [k for k in keys
                      if (ma[k][1]["score"] > da["threshold"][cat]) != bool(ma[k][1]["label"])
                      or (mb[k][1]["score"] > db["threshold"][cat]) != bool(mb[k][1]["label"])]
        keys = sorted(both_wrong, key=lambda k: -abs(ma[k][1]["score"] - da["threshold"][cat]))
        for k in keys[:args.per_cat]:
            rows.append((cat, k))
    if not rows:
        print("no shared case maps -- run dump_scores.py with --maps on both backbones")
        return

    fig, axes = plt.subplots(len(rows), 4, figsize=(8.2, 2.05 * len(rows)), squeeze=False)
    for r, (cat, key) in enumerate(rows):
        split, stem = key.split("/")
        img_path = os.path.join(args.root, cat, "test", split, stem + ".jpg")
        if not os.path.isfile(img_path):
            for ext in (".png", ".bmp", ".jpeg"):
                if os.path.isfile(os.path.join(args.root, cat, "test", split, stem + ext)):
                    img_path = os.path.join(args.root, cat, "test", split, stem + ext)
                    break
        img = cv2.resize(C.load_rgb(img_path), (C.EVAL_SIZE, C.EVAL_SIZE),
                         interpolation=cv2.INTER_AREA)
        gt = C.load_mask(os.path.join(args.root, cat, "ground_truth", "bad", stem + ".png"),
                         C.EVAL_SIZE) if split == "bad" else np.zeros((C.EVAL_SIZE,) * 2, np.uint8)
        panels = [(img, f"{cat} / {split} / {stem[:18]}"),
                  (np.stack([gt * 255] * 3, -1).astype(np.uint8), "ground truth")]
        for d, m, name, thr in ((da, ma, args.a_name, da["threshold"][cat]),
                                (db, mb, args.b_name, db["threshold"][cat])):
            s = m[key][1]["score"]
            verdict = "TP" if s > thr else "FP" if split == "good" else "FN"
            cm = cv2.applyColorMap((norm(d) * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
            panels.append((cm, f"{name}\nscore {s:.3f} (thr {thr:.3f}) {verdict}"))
        for c, (im, title) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(np.asarray(im).astype(np.uint8))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, fontsize=6.5)
    fig.suptitle("Same images, two frozen backbones (448, no foreground mask, 1-shot seed 0)",
                 fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=170)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
