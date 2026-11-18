"""Spatial alignment probe: does the token grid point at the patch it came from?

For a given backbone we occlude a 3x3 patch block at a known grid position and ask
which token changed most. If token order / positional-embedding interpolation is
correct, the argmax token sits at the occluded block for both CLIP and DINOv2.
A mis-ordered or mis-interpolated position embedding shows up as a spatially
incoherent (or shifted) response.
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import p2common as C  # noqa: E402

DATA = "/tmp/slv176/data/VisA_pytorch/1cls"


def build(backbone, res, layers):
    torch.set_num_threads(1)
    if backbone == "clip":
        from clip_core import LayerCLIP
        return LayerCLIP("ViT-B/16", "cuda", res, layers,
                         pos_embedding_mode="interp",
                         download_root="/data/disk1/home/slv176/models/clip")
    from ad_core import LayerDINOv2
    return LayerDINOv2("dinov2_vits14", "cuda", res, layers)


def probe(model, img, backbone, patch_frac=3):
    t, grid = model.prepare_image(img)
    gh, gw = grid
    ps = model.patch_size
    base = model.extract_layers(t)[-1]  # (gh*gw, D)

    # centre-ish blocks, away from borders
    targets = [(gh // 4, gw // 4), (gh // 2, gw // 2), (3 * gh // 4, 3 * gw // 4)]
    rows = []
    for (ci, cj) in targets:
        occ = t.clone()
        half = patch_frac // 2
        r0, r1 = (ci - half) * ps, (ci - half + patch_frac) * ps
        c0, c1 = (cj - half) * ps, (cj - half + patch_frac) * ps
        occ[:, r0:r1, c0:c1] = 0.0
        f = model.extract_layers(occ)[-1]
        # cosine distance per token
        a = base / (np.linalg.norm(base, axis=1, keepdims=True) + 1e-8)
        b = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
        sim = (a * b).sum(1)
        k = int(np.argmax(1.0 - sim))
        pi, pj = k // gw, k % gw
        rows.append((ci, cj, pi, pj, abs(pi - ci) + abs(pj - cj), float(sim[k])))
    return grid, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="clip")
    ap.add_argument("--resolution", type=int, default=448)
    ap.add_argument("--layers", nargs="+", type=int, default=[9])
    ap.add_argument("--category", default="candle")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    model = build(args.backbone, args.resolution, tuple(args.layers))
    _, patch_size = None, model.patch_size
    items = C.test_items(DATA, args.category)
    print(f"{args.backbone}@{args.resolution} layers={args.layers} "
          f"patch={patch_size} grid={model.grid}")
    for label, path, _ in items[: args.n]:
        img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        grid, rows = probe(model, img, args.backbone)
        print(f"  {os.path.basename(path)} grid={grid}")
        for ci, cj, pi, pj, l1, s in rows:
            print(f"    occlude=({ci},{cj}) -> argmax=({pi},{pj}) L1={l1} cos={s:.4f}")


if __name__ == "__main__":
    main()
