"""Cached grid runner: evaluate one or several feature/fusion configurations under an
identical protocol (VisA six categories, agnostic preprocessing, shots 1/2/4/8, seeds 0/1/2).

Examples
--------
# reproduce the official single-layer scoring with our cached code
python run_grid.py --name base_L12 --layers 12 --fusion single

# Part 3 improvement: late fusion of blocks 6, 9, 12 with calibration-based normalisation
python run_grid.py --name fuse_L6_9_12 --layers 6 9 12 --fusion late_z
"""

import argparse
import os
import sys
import time

# The k-NN searches (1k queries against a 5k-45k patch bank) are CPU-bound; on this node a
# bare-metal 1-NN over an 8,960-patch bank costs 163 ms at 1 thread, 132 ms at 2 and 62 ms at
# 6 (measured 2026-11-22, scripts/bench_inference.py --threads). The grid pins 1 thread not
# because it is fastest but because it makes the wall-clock reproducible and leaves cores for
# parallel category runs (scripts/run_grid_parallel.sh). Must be set before faiss is imported.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import cv2
import faiss
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import p2common as C
from ad_core import (LayerDINOv2, build_bank, deterministic_background_mask,
                     extract_image, fuse_distances, make_index, mean_top1p,
                     patch_distances)
from clip_core import LayerCLIP


# ---------------------------------------------------------------- feature cache
def cache_key(path):
    """Cache key for an image file.

    IMPORTANT: basenames alone are NOT unique in VisA -- e.g. capsules has 8
    basenames shared between test/good and test/bad, and ~90 train names shared
    with test/bad. A basename-only key made bad test images silently reuse the
    features of the same-named good image (score collapse -> up to 5 image-AUROC
    points on capsules) and let calibration images reuse test-bad features.
    Namespace by the parent directory instead. See
    results/part2_official/parity_report.md.
    """
    sub = os.path.basename(os.path.dirname(os.path.abspath(path)))
    return f"{sub}_{os.path.basename(path)}"


def cache_path(cache_root, cat, name, n_views, layers):
    tag = "L" + "-".join(str(l) for l in layers)
    return os.path.join(cache_root, tag, cat, f"{name}__v{n_views}.npz")


def cache_path_backbone(cache_root, backbone, cat, name, n_views, layers, resolution,
                        prep="std"):
    """Backbone-scoped cache path: DINO and CLIP features must never share keys.

    Resolution MUST be part of the key: the same image at 224 and 448 produces
    different patch grids, so a resolution-agnostic key silently reuses 448
    features for a 224 run (the E1 resolution control would be meaningless).
    """
    tag = f"{backbone}{resolution}_{prep}_L" + "-".join(str(l) for l in layers)
    return os.path.join(cache_root, tag, cat, f"{name}__v{n_views}.npz")


def get_features(model, cache_root, cat, path, name, n_views, layers, mask_query=True,
                 backbone="dinov2", use_mask=True, resolution=448, prep="std"):
    """Feature tensors from the npz cache; masks are recomputed on reload.

    The npz stores full-grid features per (view, layer); the mask protocol is NOT part
    of the cache key, so on reload the masks are always rebuilt with the current
    protocol instead of trusting the stale ones stored next to the features:

    - mask_query=False (reference images): all patches enter the bank, matching the
      official `mask_ref_images=False` behaviour.
    - mask_query=True (test / calibration queries): deterministic PCA foreground mask
      (fixed random_state) -- the official agnostic preprocessing, made process-stable.
    - use_mask=False (--no_mask): the controlled backbone comparison (E1) runs
      without feature-dependent foreground masks for BOTH refs and queries;
      overrides mask_query.

    See results/part2_official/parity_report.md for the parity analysis behind this.
    """
    effective_mask = mask_query and use_mask
    cp = cache_path_backbone(cache_root, backbone, cat, name, n_views, layers, resolution,
                             prep)
    if os.path.isfile(cp):
        z = np.load(cp, allow_pickle=True)
        feats = [f.astype(np.float32) for f in z["feats"]]
        grid = tuple(int(x) for x in z["grid"])
        if not effective_mask:
            masks = [np.ones(f[-1].shape[0], dtype=bool) for f in feats]
        else:
            # mask is estimated on the last block, as in extract_image
            masks = [deterministic_background_mask(f[-1], grid) for f in feats]
        return feats, grid, masks
    feats, grid, masks = extract_image(model, path, rotations=n_views,
                                       mask_query=effective_mask)
    os.makedirs(os.path.dirname(cp), exist_ok=True)
    # NOTE: features are stored in float32. A float16 cache is lossy for SCORING:
    # query-side rounding shifts patch features by up to ~8e-3, which flips the
    # nearest-neighbour members of the top-1% on knife-edge images and moved
    # capsules 1-shot image AUROC by ~5 points (0.9878 -> 0.9385) in the parity
    # probe. See results/part2_official/parity_report.md.
    np.savez_compressed(cp, feats=np.stack(feats).astype(np.float32),
                        masks=np.stack(masks), grid=np.array(grid))
    return feats, grid, masks


def score_image(banks_or_index, feats, mask, fusion, stats, normalize=True, concat=False):
    if concat:
        f = np.concatenate([feats[i] for i in range(len(feats))], axis=1)[mask].astype("float32")
        faiss.normalize_L2(f)
        d, _ = banks_or_index.search(f, k=1)
        out = np.zeros(mask.shape[0], dtype=np.float64)
        out[mask] = d.squeeze() / 2.0
        return out
    maps = [patch_distances(idx, feats[i], mask, normalize)
            for i, idx in enumerate(banks_or_index)]
    return fuse_distances(maps, fusion, stats)


def cv2_resize(arr, size):
    a = cv2.resize(arr.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)
    return gaussian_filter(a, sigma=4)


def run(args):
    cats = args.categories or C.VISA_6
    layers = tuple(args.layers)
    root = args.data_root
    out_path = os.path.join(args.out, f"{args.name}.json")
    if os.path.isfile(out_path) and not args.force:
        print("exists, skipping:", out_path)
        return

    torch.set_num_threads(1)
    device = "cuda" if args.gpu else "cpu"
    if args.backbone == "clip":
        model = LayerCLIP(args.model, device, args.resolution, layers,
                          pos_embedding_mode=args.pos_embedding_mode,
                          download_root=args.clip_download_root,
                          resize_mode=args.clip_resize_mode)
        prep = args.clip_resize_mode
    else:
        model = LayerDINOv2(args.model, device, args.resolution, layers)
        prep = "std"

    summary = {"image_auroc": {}, "pixel_auroc": {}, "threshold": {}, "meta": vars(args)}
    t_start = time.time()

    for cat in cats:
        calib_names, pool = C.ref_pool(root, cat, args.calib_exclude)
        ref_dir = C.train_dir(root, cat)
        items = C.test_items(root, cat)

        max_ref = max(args.shots) * max(s + 1 for s in args.seeds)
        ref_names = pool[:max_ref]

        # --- extract once
        ref_cache = {}
        for n in ref_names:
            ref_cache[n] = get_features(model, args.cache, cat, os.path.join(ref_dir, n),
                                        cache_key(os.path.join(ref_dir, n)), 8, layers,
                                        mask_query=False, backbone=args.backbone,
                                        use_mask=args.use_mask,
                                        resolution=args.resolution, prep=prep)
        test_cache = [get_features(model, args.cache, cat, p,
                                   cache_key(p), 1, layers, backbone=args.backbone,
                                   use_mask=args.use_mask,
                                   resolution=args.resolution, prep=prep)
                      for _, p, _ in items]
        calib_cache = [get_features(model, args.cache, cat, os.path.join(ref_dir, n),
                                    cache_key(os.path.join(ref_dir, n)), 1, layers,
                                    backbone=args.backbone, use_mask=args.use_mask,
                                    resolution=args.resolution, prep=prep)
                       for n in calib_names]

        for shot in args.shots:
            for seed in args.seeds:
                t0 = time.time()
                sel = pool[seed * shot:(seed + 1) * shot]
                feats_all, masks_all = [], []
                for n in sel:
                    f, _, m = ref_cache[n]
                    feats_all += f
                    masks_all += m
                banks = build_bank(feats_all, masks_all)

                if args.fusion == "concat":
                    bank = np.concatenate(banks, axis=1)
                    index = make_index(bank)
                    stats, concat = None, True
                else:
                    index = [make_index(b) for b in banks]
                    concat = False
                    stats = None
                    if args.fusion == "late_z":
                        # per-layer mean/std of patch distances, estimated on the 20 calibration
                        # images only (no test label ever touches this)
                        calib_maps = [[] for _ in banks]
                        for f, _, m in calib_cache:
                            for i, idx in enumerate(index):
                                calib_maps[i].append(patch_distances(idx, f[i], m[0]))
                        stats = [(float(np.concatenate(v).mean()),
                                  float(np.concatenate(v).std()) + 1e-6) for v in calib_maps]

                scores, labels, maps, gts = [], [], [], []
                for (label, _, mask_path), (f, grid, m) in zip(items, test_cache):
                    d = score_image(index, f[0], m[0], args.fusion, stats, concat=concat)
                    scores.append(mean_top1p(d))
                    labels.append(label)
                    maps.append(cv2_resize(d.reshape(grid), C.EVAL_SIZE))
                    gts.append(C.load_mask(mask_path, C.EVAL_SIZE))

                calib_scores = []
                for f, grid, m in calib_cache:
                    d = score_image(index, f[0], m[0], args.fusion, stats, concat=concat)
                    calib_scores.append(mean_top1p(d))
                thr = C.percentile_threshold(calib_scores, 95.0)

                au_img = C.image_auroc(scores, labels)
                au_px = C.pooled_pixel_auroc(maps, gts)
                for metric, val in (("image_auroc", au_img), ("pixel_auroc", au_px)):
                    summary[metric].setdefault(f"{shot}-shot", {}).setdefault(cat, {})[str(seed)] = val
                summary["threshold"].setdefault(f"{shot}-shot", {}).setdefault(cat, {})[str(seed)] = thr
                print(f"[{cat}] shot={shot} seed={seed} img={au_img:.4f} px={au_px:.4f} "
                      f"({time.time()-t0:.1f}s)", flush=True)

        C.jdump(summary, out_path)

    final = C.summarize(summary, cats)
    final["seed_values"] = {m: summary[m] for m in ("image_auroc", "pixel_auroc")}
    final["threshold"] = summary["threshold"]
    final["config"] = vars(args)
    final["layers"] = list(layers)
    final["runtime_s"] = time.time() - t_start
    C.jdump(final, os.path.join(args.out, f"{args.name}_final.json"))

    print(f"\n=== {args.name} ===")
    for metric in ("image_auroc", "pixel_auroc"):
        for shot_key in sorted(final[metric], key=lambda k: int(k.split("-")[0])):
            r = final[metric][shot_key]
            print(f"  {metric:12s} {shot_key:8s} MEAN={r['MEAN']*100:.2f} "
                  f"({r['MEAN_std']*100:.2f})")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--data_root", default="/tmp/slv176/data/VisA_pytorch/1cls")
    p.add_argument("--cache", default="/tmp/slv176/featcache")
    p.add_argument("--out", default="/tmp/slv176/grid")
    p.add_argument("--backbone", default="dinov2", choices=["dinov2", "clip"],
                   help="visual backbone for E1 (default: official-compatible DINOv2)")
    p.add_argument("--model", default=None,
                   help="backbone checkpoint id (default: dinov2_vits14 / ViT-B/16)")
    p.add_argument("--resolution", type=int, default=448)
    p.add_argument("--layers", nargs="+", type=int, default=[12])
    p.add_argument("--pos_embedding_mode", default="interp", choices=["native", "interp"],
                   help="CLIP positional embedding for non-native resolution")
    p.add_argument("--clip_download_root", default="/data/disk1/home/slv176/models/clip")
    p.add_argument("--clip_resize_mode", default="short_edge_crop",
                   choices=["short_edge_crop", "squash", "short_edge"],
                   help="CLIP preprocessing: official short-edge+centre-crop, squash the "
                        "whole image to a square, or short_edge (DINOv2-wrapper style: "
                        "full frame, rectangular grid) -- the controlled E1 setting")
    p.add_argument("--no_mask", dest="use_mask", action="store_false",
                   help="disable PCA foreground masks (controlled E1 comparison; "
                        "forced semantics: refs and queries all-patch)")
    p.add_argument("--fusion", default="single",
                   choices=["single", "late_mean", "late_z", "late_min", "concat"])
    p.add_argument("--shots", nargs="+", type=int, default=[1, 2, 4, 8])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--calib_exclude", type=int, default=20)
    p.add_argument("--categories", nargs="*", default=None)
    p.add_argument("--gpu", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    if args.model is None:
        args.model = "ViT-B/16" if args.backbone == "clip" else "dinov2_vits14"
    if args.backbone == "clip" and args.resolution == 448:
        # CLIP ViT-B/16 native is 224; 448 needs interpolation (verify first)
        print("NOTE: CLIP at 448 uses bicubic positional-embedding interpolation")
    return args


if __name__ == "__main__":
    run(parse_args())
