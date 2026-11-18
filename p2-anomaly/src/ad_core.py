"""Cached, layer-aware re-implementation of the AnomalyDINO scoring pipeline.

Why this exists
---------------
The official `run_anomalydino.py` re-extracts DINOv2 features for every test image once per
(shot, seed) pair: 12 configurations x 960 images on the shared A6000. That is wasteful for
Part 3, where a dozen variants must be compared under an identical protocol.

This module extracts every needed patch-feature tensor exactly once (references with their 8
rotations, and test images), caches it, and then only re-builds the memory bank / k-NN for a
given (shot, seed). The scoring is a line-by-line port of `src/detection.py` so that
`--layers 12 --fusion single` reproduces the official numbers; `validate_grid.py` checks this.

Extra capability (used by Part 3): patch features can be taken from several DINOv2 blocks and
fused, either by late fusion of per-layer distance maps (calibrated on the 20 calibration
images) or by early fusion of concatenated features.
"""

import os
import sys

import cv2
import faiss
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_AD = os.path.join(os.path.dirname(_HERE), "code", "anomalydino")
sys.path.insert(0, _AD)

from src.backbones import DINOv2Wrapper  # noqa: E402
from src.utils import augment_image  # noqa: E402


def mean_top1p(distances):
    distances = np.asarray(distances).ravel()
    k = int(len(distances) * 0.01)
    if k == 0:
        return float(np.max(distances))
    return float(np.mean(np.sort(distances)[-k:]))


class LayerDINOv2(DINOv2Wrapper):
    """DINOv2 ViT with identical preprocessing to the official wrapper, multi-layer output."""

    def __init__(self, model_name="dinov2_vits14", device="cuda", smaller_edge_size=448,
                 layers=(12,)):
        assert all(1 <= l <= 12 for l in layers), "ViT-S/14 exposes 12 blocks"
        self.layers = tuple(layers)
        super().__init__(model_name, device, smaller_edge_size)

    def extract_layers(self, image_tensor):
        with torch.inference_mode():
            outs = self.model.get_intermediate_layers(
                image_tensor.unsqueeze(0).to(self.device), n=[l - 1 for l in self.layers])
        return [o.squeeze(0).float().cpu().numpy() for o in outs]

    # official behaviour == last block only
    def extract_features(self, image_tensor):
        return self.extract_layers(image_tensor)[-1]


def deterministic_background_mask(feats, grid, threshold=10, kernel_size=3, border=0.2):
    """PCA foreground mask with a pinned random_state.

    Mirrors DINOv2Wrapper.compute_background_mask(masking_type=True) but fixes the
    randomized-PCA seed: sklearn draws a new random projection on every call when
    random_state is unset, which flips borderline masks between processes (up to
    ~460/1536 patches on VisA capsules) and moves image AUROC by several points
    when combined with a foreground-only memory bank. See
    results/part2_official/parity_report.md.
    """
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1, svd_solver="randomized", random_state=0)
    first_pc = pca.fit_transform(feats.astype(np.float32))
    mask = first_pc > threshold
    m = mask.reshape(grid)[int(grid[0] * border):int(grid[0] * (1 - border)),
                           int(grid[1] * border):int(grid[1] * (1 - border))]
    if m.sum() <= m.size * 0.35:
        mask = -first_pc > threshold
    mask = cv2.dilate(mask.astype(np.uint8),
                      np.ones((kernel_size, kernel_size), np.uint8)).astype(bool)
    mask = cv2.morphologyEx(mask.astype(np.uint8),
                            cv2.MORPH_CLOSE,
                            np.ones((kernel_size, kernel_size), np.uint8)).astype(bool)
    return mask.squeeze()


def extract_image(model, path_or_img, rotations=1, mask_query=True):
    """Return (list_of_feature_arrays_per_layer_per_rotation, grid_size, mask_per_rotation).

    mask_query reproduces the official preprocessing split: reference images enter the
    memory bank UNMASKED (official mask_ref_images=False -> masking_type=False), while
    test/calibration queries are PCA-masked (masking_type=True). The cached runner
    historically masked references too, which is a systematic protocol deviation from
    the official path.
    """
    if isinstance(path_or_img, str):
        img = cv2.cvtColor(cv2.imread(path_or_img, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    else:
        img = path_or_img
    views = augment_image(img) if rotations > 1 else [img]
    feats, masks, grid = [], [], None
    for v in views:
        t, grid = model.prepare_image(v)
        per_layer = model.extract_layers(t)
        f = per_layer[-1]  # background mask is estimated on the last block, as in the paper
        mask = model.compute_background_mask(f, grid, threshold=10, masking_type=mask_query)
        feats.append(per_layer)
        masks.append(mask)
    return feats, grid, masks


def build_bank(per_layer_feats, masks, normalize=True):
    """Concatenate masked patches of several reference views into one bank per layer."""
    n_layers = len(per_layer_feats[0])
    banks = [[] for _ in range(n_layers)]
    for feats, m in zip(per_layer_feats, masks):
        for i in range(n_layers):
            banks[i].append(feats[i][m])
    banks = [np.concatenate(b, axis=0).astype("float32") for b in banks]
    if normalize:
        for b in banks:
            faiss.normalize_L2(b)
    return banks


def make_index(bank):
    idx = faiss.IndexFlatL2(bank.shape[1])
    idx.add(bank)
    return idx


def patch_distances(index, feats, mask, normalize=True):
    """Per-patch 1-NN cosine distance over the full grid; masked patches stay at 0."""
    f = feats[mask].astype("float32")
    if normalize:
        faiss.normalize_L2(f)
    d, _ = index.search(f, k=1)
    out = np.zeros(mask.shape[0], dtype=np.float64)
    out[mask] = d.squeeze() / 2.0
    return out


# ---------------------------------------------------------------- fusion
def fuse_distances(list_of_maps, mode="late_mean", stats=None):
    """Combine per-layer distance maps.

    stats: list of (mu, sigma) per layer, estimated on the calibration images only.
    """
    if len(list_of_maps) == 1:
        return list_of_maps[0]
    if mode == "late_mean":
        return np.mean(np.stack(list_of_maps, 0), axis=0)
    if mode == "late_z":
        zs = []
        for d, (mu, sg) in zip(list_of_maps, stats):
            zs.append((d - mu) / max(sg, 1e-6))
        return np.mean(np.stack(zs, 0), axis=0)
    if mode == "late_min":
        return np.min(np.stack(list_of_maps, 0), axis=0)
    raise ValueError(mode)


def models_equal(a, b):
    return np.allclose(a, b, atol=1e-6)
