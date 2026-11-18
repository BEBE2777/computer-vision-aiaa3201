"""Feature anisotropy probe: how similar are patch features inside one image?

CLIP's dense features are known to be anisotropic (all patches close in cosine
space after global attention mixing), which flattens top-1% kNN distances. This
measures the mean pairwise cosine similarity of patch features within an image,
for the backbone/layer we freeze, against the DINOv2 reference.
"""
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


def build(backbone, res, layers, prep):
    torch.set_num_threads(1)
    if backbone == "clip":
        from clip_core import LayerCLIP
        return LayerCLIP("ViT-B/16", "cuda", res, layers, pos_embedding_mode="interp",
                         download_root="/data/disk1/home/slv176/models/clip",
                         resize_mode=prep)
    from ad_core import LayerDINOv2
    return LayerDINOv2("dinov2_vits14", "cuda", res, layers)


def stats(model, imgs, rng, n_sample=200):
    sims, eff_rank = [], []
    for img in imgs:
        t, grid = model.prepare_image(img)
        f = model.extract_layers(t)[-1]
        f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
        idx = rng.choice(f.shape[0], size=min(n_sample, f.shape[0]), replace=False)
        s = f[idx] @ f[idx].T
        iu = np.triu_indices(len(idx), k=1)
        sims.append(float(s[iu].mean()))
        sv = np.linalg.svd(f[idx] - f[idx].mean(0), compute_uv=False)
        p = sv ** 2 / (sv ** 2).sum()
        eff_rank.append(float(np.exp(-(p * np.log(p + 1e-12)).sum())))
    return float(np.mean(sims)), float(np.mean(eff_rank))


def main():
    rng = np.random.default_rng(0)
    for cat in ("candle", "capsules"):
        items = C.test_items(DATA, cat)
        imgs = [cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                for _, p, _ in items[:3]]
        print(f"== {cat} (3 test images) ==")
        for name, bb, res, layers, prep in (
            ("DINOv2@448 L12", "dinov2", 448, (12,), "std"),
            ("CLIP@448 L12", "clip", 448, (12,), "short_edge"),
            ("CLIP@448 L9", "clip", 448, (9,), "short_edge"),
            ("CLIP@448 L6", "clip", 448, (6,), "short_edge"),
        ):
            m = build(bb, res, layers, prep)
            ms, er = stats(m, imgs, rng)
            print(f"  {name:16s} mean pairwise cos={ms:.4f}  effective rank={er:.1f}")
            del m
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
