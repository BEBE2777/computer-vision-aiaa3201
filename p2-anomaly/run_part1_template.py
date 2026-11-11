"""Part 1 -- Classical normality modelling: template difference baseline.

Algorithm (per category, shot, seed):
  1. reference images = the SAME filenames AnomalyDINO uses (same calibration exclusion
     of the first 20 normal training files and the same seed-based slice);
  2. every image (test, calibration and reference) is resized so that its shorter edge is
     448 px and centre-cropped to 448x448 -- pixel-wise comparison needs a common frame;
  3. optional ECC (MOTION_EUCLIDEAN) alignment of everything onto the first reference;
  4. difference map between a test image and each reference = 0.5 * |dI| + 0.5 * (1 - SSIM);
     the "closest template" (lowest mean difference) is kept;
  5. Gaussian smoothing (sigma = 4), image score = mean of the top 1 % map values
     (same aggregation as AnomalyDINO, so image AUROC is comparable);
  6. OK/NG threshold = 95th percentile of the 20 calibration images' scores.

No per-image rescaling of scores happens anywhere: AUROC is computed on raw scores.
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from p2common import (EVAL_SIZE, VISA_6, image_auroc, jdump, load_mask, load_rgb,
                      percentile_threshold, pooled_pixel_auroc,
                      prep_square, ref_pool, shot_refs, summarize, test_items, top1p_mean)


def ecc_warp_to_anchor(anchor_gray, img_gray):
    """Return the affine warp that maps `img_gray` onto `anchor_gray` (ECC, euclidean)."""
    warp = np.eye(2, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
    try:
        _, warp = cv2.findTransformECC(anchor_gray, img_gray, warp, cv2.MOTION_EUCLIDEAN,
                                       crit, None, 5)
    except cv2.error:
        pass
    return warp


def apply_warp(img, warp):
    h, w = img.shape[:2]
    return cv2.warpAffine(img, warp, (w, h),
                          flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)


def gray01(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0


def fg_box(img, margin=0.03):
    """Bounding box of the object, estimated from its distance to the border intensity.

    The naive template baseline fails mostly because the object sits at a different position
    and scale in every shot (VisA is photographed, not fixtured). Cropping the object and
    rescaling it to a fixed size removes both, leaving the difference operator to do its job.
    Returns None when the estimate is implausible, so the caller falls back to the full frame.
    """
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (5, 5), 0).astype(np.float32)
    border = np.concatenate([g[:8, :].ravel(), g[-8:, :].ravel(),
                             g[:, :8].ravel(), g[:, -8:].ravel()])
    dev = np.abs(g - float(np.median(border)))
    thr = max(12.0, float(dev.mean() + dev.std()))
    m = (dev > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n < 2:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = (stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP],
                  stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT])
    if w < 24 or h < 24 or w * h > 0.95 * img.shape[0] * img.shape[1]:
        return None
    mx, my = int(margin * w), int(margin * h)
    return (max(0, x - mx), max(0, y - my),
            min(img.shape[1], x + w + mx), min(img.shape[0], y + h + my))


def apply_box(arr, box, size=EVAL_SIZE, nearest=False):
    """Crop `arr` to `box` and resize to `size` x `size` (bilinear for RGB, nearest for masks)."""
    x0, y0, x1, y1 = box
    c = arr[y0:y1, x0:x1]
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.resize(c, (size, size), interpolation=interp)


def diff_map(ref_rgb, img_rgb, use_grad=False):
    """Fixed-scale (no per-image normalisation) difference map in [0, 1]."""
    gr, gi = gray01(ref_rgb), gray01(img_rgb)
    d_abs = np.abs(gi - gr)
    _, ssim_map = structural_similarity(gr, gi, data_range=1.0, full=True, win_size=11)
    d = 0.5 * d_abs + 0.5 * np.clip(1.0 - ssim_map, 0.0, 1.0)
    if use_grad:
        kr = np.stack(np.gradient(gr), 0)
        ki = np.stack(np.gradient(gi), 0)
        d_grad = np.abs(ki - kr).mean(0) * 2.0
        d = d + 0.5 * np.clip(d_grad, 0.0, 1.0)
    # NOTE: deliberately NO per-image normalisation here -- the absolute magnitude of the
    # difference is what an image-level score must reflect, and rescaling each test image
    # separately is exactly what the project PDF forbids when computing AUROC.
    return d


def score_one(img, refs, anchor_gray, align, use_grad):
    """Score a single image against a set of prepared references.

    `align` must be a bool -- passing the CLI string ("none"/"ecc") used to be truthy for
    both values, which silently enabled alignment in the "none" setting.
    """
    if align:
        warp = ecc_warp_to_anchor(anchor_gray, gray01(img))
        img = apply_warp(img, warp)
    best, best_mean = None, np.inf
    for r in refs:
        d = diff_map(r, img, use_grad=use_grad)
        m = float(d.mean())
        if m < best_mean:
            best, best_mean = d, m
    return gaussian_filter(best, sigma=4)


def run(args):
    root = args.data_root
    out_root = os.path.join(args.out, f"align={args.align}" + ("_crop" if args.crop else "") + ("_grad" if args.use_grad else ""))
    os.makedirs(out_root, exist_ok=True)

    cats = args.categories or VISA_6
    summary = {"image_auroc": {}, "pixel_auroc": {}, "threshold": {}}

    for cat in cats:
        # references get prepared once per (cat, shot, seed)
        _, pool = ref_pool(root, cat, args.calib_exclude)
        calib_names, _ = ref_pool(root, cat, args.calib_exclude)
        calib_paths = [os.path.join(root, cat, "train", "good", n) for n in calib_names]

        for shot in args.shots:
            for seed in args.seeds:
                do_align = (args.align == "ecc")
                t0 = time.time()
                ref_names = shot_refs(pool, shot, seed)
                ref_dir = os.path.join(root, cat, "train", "good")
                refs = []
                for n in ref_names:
                    im = prep_square(load_rgb(os.path.join(ref_dir, n)))
                    if args.crop:
                        b = fg_box(im)
                        im = apply_box(im, b) if b else im
                    refs.append(im)

                anchor_gray = gray01(refs[0])
                if do_align:
                    warps = [ecc_warp_to_anchor(anchor_gray, gray01(r)) for r in refs[1:]]
                    refs = [refs[0]] + [apply_warp(r, w) for r, w in zip(refs[1:], warps)]

                # calibration scores -> OK/NG threshold (demo only, never used for AUROC)
                calib_prepped = []
                for p in calib_paths:
                    im = prep_square(load_rgb(p))
                    if args.crop:
                        b = fg_box(im)
                        im = apply_box(im, b) if b else im
                    calib_prepped.append(im)
                calib_scores = [top1p_mean(score_one(im, refs, anchor_gray, do_align, args.use_grad))
                                for im in calib_prepped]
                thr = percentile_threshold(calib_scores, 95.0)

                items = test_items(root, cat)
                scores, labels, maps, masks = [], [], [], []
                for label, img_path, mask_path in items:
                    im = prep_square(load_rgb(img_path))
                    mk = load_mask(mask_path, EVAL_SIZE)
                    if args.crop:
                        b = fg_box(im)
                        if b:
                            im = apply_box(im, b)
                            mk = apply_box(mk, b, nearest=True)
                    s_map = score_one(im, refs, anchor_gray, do_align, args.use_grad)
                    scores.append(top1p_mean(s_map))
                    labels.append(label)
                    maps.append(s_map.astype(np.float32))
                    masks.append(mk)

                au_img = image_auroc(scores, labels)
                au_px = pooled_pixel_auroc(maps, masks)

                for metric, val in (("image_auroc", au_img), ("pixel_auroc", au_px)):
                    summary[metric].setdefault(f"{shot}-shot", {}).setdefault(cat, {})[str(seed)] = val
                summary["threshold"].setdefault(f"{shot}-shot", {}).setdefault(cat, {})[str(seed)] = thr

                # keep maps for one representative configuration only (qualitative figures)
                if shot == args.save_maps_shot and seed == args.save_maps_seed:
                    d = os.path.join(out_root, cat, f"shot={shot}_seed={seed}")
                    os.makedirs(d, exist_ok=True)
                    np.savez_compressed(os.path.join(d, "maps.npz"),
                                        maps=np.stack(maps).astype(np.float16),
                                        masks=np.stack(masks),
                                        labels=np.array(labels),
                                        scores=np.array(scores, dtype=np.float32),
                                        threshold=np.array([thr]))
                    jdump({"reference_files": ref_names, "calibration_files": calib_names,
                           "test_files": [os.path.basename(i[1]) for i in items]},
                          os.path.join(d, "files.json"))

                print(f"[{cat}] {shot}-shot seed={seed} imgAUROC={au_img:.4f} "
                      f"pxAUROC={au_px:.4f} thr={thr:.4f} ({time.time()-t0:.1f}s)", flush=True)

                jdump(summary, os.path.join(out_root, "summary.json"))

    final = summarize(summary, cats)
    jdump(final, os.path.join(out_root, "results.json"))

    print("\n=== Part 1 template baseline ===")
    for metric in ("image_auroc", "pixel_auroc"):
        print(metric)
        for shot_key in sorted(final[metric], key=lambda k: int(k.split("-")[0])):
            row = final[metric][shot_key]
            print("  %-8s %s  MEAN=%.4f" % (shot_key,
                  " ".join(f"{c[:5]}={row[c]:.3f}" for c in cats), row["MEAN"]))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--out", default="results/part1_template")
    p.add_argument("--categories", nargs="*", default=None)
    p.add_argument("--shots", nargs="+", type=int, default=[1, 2, 4, 8])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--calib_exclude", type=int, default=20)
    p.add_argument("--align", choices=["none", "ecc"], default="none")
    p.add_argument("--use_grad", action="store_true")
    p.add_argument("--crop", action="store_true",
                   help="normalise object scale/position by cropping to a foreground box")
    p.add_argument("--save_maps_shot", type=int, default=4)
    p.add_argument("--save_maps_seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
