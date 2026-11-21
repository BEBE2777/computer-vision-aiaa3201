"""Parity probe: official AnomalyDINO scoring path vs cached path (ad_core.py), step by step.

Runs both paths on the SAME reference image and the SAME 10 test images (5 good + 5 bad)
of one category, with every intermediate recorded, and scores four variants so the first
divergence can be localised:

  O            : official protocol -- bank keeps ALL patches of the 8 rotated ref views
                 (mask_ref_images=False), float32 features computed fresh, test patches
                 PCA-masked (agnostic, masking=True), score = mean_top1p over the full
                 flattened distance grid (masked patches contribute 0).
  O_fp16       : same as O but features go through the float16 feature cache rounding.
  C_masked     : cached-as-coded -- bank only keeps foreground patches of each rotation
                 (ad_core.extract_image always applies masking_type=True), float32.
  C_asrun      : cached-as-run  -- masked bank + float16-rounded features (what
                 run_grid.py actually scored through the npz cache).

Divergence attribution: max|O - O_fp16| isolates the float16 cache; max|O - C_masked|
isolates the reference-bank masking; C_asrun reproduces the published cached numbers.
"""

import argparse
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import cv2
import faiss
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))          # repo root -> src/
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "src"))

import p2common as C
from ad_core import LayerDINOv2, mean_top1p  # noqa: E402  (ad_core puts AnomalyDINO src/ on path)
from src.utils import augment_image  # noqa: E402  (this is code/anomalydino/src/utils.py)


def sha(a):
    """Cheap deterministic fingerprint of an array."""
    a = np.asarray(a, dtype=np.float64)
    return {"shape": list(a.shape), "sum": float(a.sum()),
            "mean": float(a.mean()), "absmax": float(np.abs(a).max())}


def bank_and_index(feats_per_view, mask_per_view, dtype="float32"):
    arrs, masks = [], []
    for f, m in zip(feats_per_view, mask_per_view):
        f32 = f.astype(np.float32)
        if dtype == "float16":
            f32 = f32.astype(np.float16).astype(np.float32)
        arrs.append(f32)
        masks.append(m)
    bank = np.concatenate([f[m] for f, m in zip(arrs, masks)], axis=0).astype("float32")
    faiss.normalize_L2(bank)
    idx = faiss.IndexFlatL2(bank.shape[1])
    idx.add(bank)
    return idx, bank.shape[0]


def score(idx, feats, mask, dtype="float32"):
    f = feats.astype(np.float32)
    if dtype == "float16":
        f = f.astype(np.float16).astype(np.float32)
    f = f[mask].astype("float32")
    faiss.normalize_L2(f)
    d, _ = idx.search(f, k=1)
    out = np.zeros(mask.shape[0], dtype=np.float64)
    out[mask] = d.squeeze() / 2.0
    grid = GRID  # set globally after first prepare_image
    return out, out.reshape(grid), float(mean_top1p(out.flatten()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/tmp/slv176/data/VisA_pytorch/1cls")
    ap.add_argument("--model", default="dinov2_vits14")
    ap.add_argument("--resolution", type=int, default=448)
    ap.add_argument("--category", default="candle")
    ap.add_argument("--shot", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calib_exclude", type=int, default=20)
    ap.add_argument("--n_test", type=int, default=10, help="5 good + 5 bad (or n//2 each)")
    ap.add_argument("--full", action="store_true",
                    help="score ALL test images under O and C_asrun and report Image AUROC")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(1)
    model = LayerDINOv2(args.model, device, args.resolution, (12,))

    cat = args.category
    calib_names, pool = C.ref_pool(args.data_root, cat, args.calib_exclude)
    sel = pool[args.seed * args.shot:(args.seed + 1) * args.shot]
    ref_dir = C.train_dir(args.data_root, cat)

    items = C.test_items(args.data_root, cat)
    good = [(lab, p, mk) for lab, p, mk in items if lab == 0][:args.n_test // 2]
    bad = [(lab, p, mk) for lab, p, mk in items if lab == 1][:args.n_test - len(good)]
    probe = good + bad

    rep = {"config": vars(args), "calib_first3": calib_names[:3], "ref_selected": sel,
           "probe_images": [{"label": int(l), "path": os.path.basename(p)} for l, p, _ in probe],
           "steps": {}}

    # ---- step 1: reference features, 8 rotations, fresh float32
    ref_img = cv2.cvtColor(cv2.imread(os.path.join(ref_dir, sel[0]), cv2.IMREAD_COLOR),
                           cv2.COLOR_BGR2RGB)
    views = augment_image(ref_img)
    feats_v, maskTrue_v, ones_v, grids = [], [], [], set()
    with torch.inference_mode():
        for v in views:
            t, grid = model.prepare_image(v)
            grids.add(grid)
            f = model.extract_layers(t)[-1]
            mt = model.compute_background_mask(f, grid, threshold=10, masking_type=True)
            feats_v.append(f)
            maskTrue_v.append(mt)
            ones_v.append(np.ones(f.shape[0], dtype=bool))
    GRID = grids.pop()
    rep["steps"]["ref"] = {"n_views": len(views), "grid": list(GRID),
                           "patches_per_view": int(feats_v[0].shape[0]),
                           "fg_patches_per_view": [int(m.sum()) for m in maskTrue_v]}
    globals()["GRID"] = GRID

    # ---- step 2: banks
    idx_O, n_O = bank_and_index(feats_v, ones_v, "float32")
    idx_O16, _ = bank_and_index(feats_v, ones_v, "float16")
    idx_C, n_C = bank_and_index(feats_v, maskTrue_v, "float32")
    idx_C16, _ = bank_and_index(feats_v, maskTrue_v, "float16")
    rep["steps"]["bank"] = {"official_all_patches": int(n_O), "cached_masked_patches": int(n_C)}

    # ---- step 3: probe images
    rows = []
    variants = [("O", idx_O, "float32"), ("O_fp16", idx_O16, "float16"),
                ("C_masked", idx_C, "float32"), ("C_asrun", idx_C16, "float16")]
    with torch.inference_mode():
        for lab, path, _mk in probe:
            img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            t, grid = model.prepare_image(img)
            f = model.extract_layers(t)[-1]
            mask2 = model.compute_background_mask(f, grid, threshold=10, masking_type=True)
            row = {"image": os.path.basename(path), "label": int(lab),
                   "test_fg_patches": int(mask2.sum()), "scores": {}, "map_absdiff_vs_O": {}}
            base_map = None
            for name, idx, dt in variants:
                out, m, s = score(idx, f, mask2, dt)
                row["scores"][name] = s
                if name == "O":
                    base_map = out
                else:
                    row["map_absdiff_vs_O"][name] = float(np.abs(out - base_map).max())
            rows.append(row)
            print(json.dumps(row), flush=True)
    rep["probe"] = rows

    # ---- step 3b: optional full-protocol AUROC under both banks
    if args.full:
        rows = []  # skip the 10-image probe in full mode
        results_full = {}
        for name, idx, dt in (("O", idx_O, "float32"), ("C_asrun", idx_C16, "float16")):
            scores, labels = [], []
            with torch.inference_mode():
                for i, (lab, path, _mk) in enumerate(items):
                    img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                    t, grid = model.prepare_image(img)
                    f = model.extract_layers(t)[-1]
                    mask2 = model.compute_background_mask(f, grid, threshold=10,
                                                          masking_type=True)
                    _, _, s = score(idx, f, mask2, dt)
                    scores.append(s)
                    labels.append(int(lab))
                    if i % 200 == 0:
                        print(f"[{name}] {i}/{len(items)}", flush=True)
            au = C.image_auroc(scores, labels)
            results_full[name] = au
            print(f"[{name}] FULL image AUROC = {au:.4f}", flush=True)
        rep["full_image_auroc"] = results_full

    # ---- step 4: aggregate
    agg = {}
    for name, _, _ in variants[1:]:
        ds = [r["scores"]["O"] - r["scores"][name] for r in rows]
        agg[name] = {"max_abs_score_diff": float(np.max(np.abs(ds))),
                     "mean_abs_score_diff": float(np.mean(np.abs(ds)))}
    rep["summary"] = agg
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=2)
    print("=== SUMMARY ===")
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
