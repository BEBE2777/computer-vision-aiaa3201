"""Shared utilities for AIAA3201 Project 2 (few-shot industrial anomaly detection).

Design notes
------------
* All methods (template baseline, AnomalyDINO, improved AnomalyDINO) are evaluated by
  the SAME code in this module so that the comparison is apples-to-apples.
* Pixel-level AUROC is computed on maps resampled to `EVAL_SIZE` (448x448) instead of the
  native image resolution. Reason: pooling ~1.5 M pixels x 200 images per category does not
  fit in the shared server memory. Every method goes through the same resampling.
"""

import json
import os

import cv2
import numpy as np

VISA_6 = ["candle", "capsules", "cashew", "chewinggum", "fryum", "pipe_fryum"]
EVAL_SIZE = 448


# ---------------------------------------------------------------- data layout
def train_dir(root, cat):
    return os.path.join(root, cat, "train", "good")


def test_items(root, cat):
    """Return [(label, image_path, mask_path_or_None), ...] for one category.

    label is 0 (normal) / 1 (anomalous).
    """
    items = []
    for sub, label in (("good", 0), ("bad", 1)):
        d = os.path.join(root, cat, "test", sub)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if not name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            img = os.path.join(d, name)
            mask = None
            if label == 1:
                mask = os.path.join(root, cat, "ground_truth", "bad",
                                    os.path.splitext(name)[0] + ".png")
            items.append((label, img, mask))
    return items


def ref_pool(root, cat, calib_exclude=20):
    """Sorted normal training files, split into (calibration, reference pool).

    Mirrors the sampling rule used inside the patched AnomalyDINO `detection.py`, so
    every method sees exactly the same reference images for a given (shot, seed).
    """
    names = sorted(os.listdir(train_dir(root, cat)))
    names = [n for n in names if n.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    return names[:calib_exclude], names[calib_exclude:]


def shot_refs(pool, shot, seed):
    return pool[seed * shot:(seed + 1) * shot]


# ---------------------------------------------------------------- image io
def load_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def crop_to_multiple(img, m=14):
    h, w = img.shape[:2]
    return img[:h - h % m, :w - w % m]


def prep_square(img, size=EVAL_SIZE):
    """Resize the shorter edge to `size`, centre-crop a `size x size` patch.

    Used by the template baseline so that a pixel-wise comparison is well defined.
    """
    h, w = img.shape[:2]
    scale = size / min(h, w)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    y0 = (nh - size) // 2
    x0 = (nw - size) // 2
    return img[y0:y0 + size, x0:x0 + size]


def load_mask(path, size=EVAL_SIZE):
    if path is None:
        return np.zeros((size, size), dtype=np.uint8)
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8)


# ---------------------------------------------------------------- scoring
def top1p_mean(x):
    """Mean of the top 1 % values -- the aggregation used by AnomalyDINO."""
    x = np.asarray(x).ravel()
    k = int(len(x) * 0.01)
    if k == 0:
        return float(np.max(x))
    return float(np.mean(np.sort(x)[-k:]))


def percentile_threshold(scores, q=95.0):
    return float(np.percentile(np.asarray(scores), q))


# ---------------------------------------------------------------- metrics
def image_auroc(anomaly_scores, labels):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(np.asarray(labels), np.asarray(anomaly_scores)))


def pooled_pixel_auroc(maps, masks):
    """Exact pooled pixel AUROC without sklearn.

    sklearn's roc_auc_score needs ~59 s for 100 images of 448x448 (20 M pooled samples) because
    it sorts the full score vector with a big constant factor. This computes the same quantity
    from the rank sum of the positives against the sorted negative scores, which is exact and
    ~30x faster (sort only the negatives, then two searchsorted calls).
    """
    scores = np.concatenate([m.ravel() for m in maps]).astype(np.float32)
    labels = np.concatenate([m.ravel() for m in masks]).astype(bool)
    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    neg = np.sort(scores[~labels])
    pos = scores[labels]
    lt = np.searchsorted(neg, pos, side="left")
    eq = np.searchsorted(neg, pos, side="right") - lt
    return float(np.mean((lt + 0.5 * eq) / n_neg))


def mean_over_categories(d):
    vals = [v for v in d.values() if v == v]
    return float(np.mean(vals)) if vals else float("nan")


# ---------------------------------------------------------------- misc
def jdump(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def jload(path):
    with open(path) as f:
        return json.load(f)
