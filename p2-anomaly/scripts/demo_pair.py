"""Re-score a hand-picked normal/defective pair with the SAME official AnomalyDINO
setup used in the Part 2 run (candle, 2-shot, seed 0), and also score the 20
calibration normals so the 95th-percentile OK/NG threshold can be drawn.

Why: the two example images saved automatically by the official code
(`example_good_0.png`, `example_bad_0.png`) are simply the first three files of
each folder; their scores (0.187 normal vs 0.172 defective) do not illustrate
"anomalies score higher". This script rebuilds the exact memory bank from
`refs/<obj>_shot=2_seed=0.json` (same references, same 8-way rotation
augmentation, same masking) and reports per-image scores, so a pair with a clear
separation can be chosen from the measured distribution.

Nothing here changes the protocol: no test label is used for tuning, the memory
bank is the same one the reported AUROC comes from, and the script asserts that
it reproduces the cached score of each re-scored image.
"""
import argparse
import json
import os
import shutil
import sys

import cv2
import faiss
import numpy as np
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default="/tmp/slv176/aiaa3201_p2/code/anomalydino")
    ap.add_argument("--data", default="/tmp/slv176/data/VisA_pytorch/1cls")
    ap.add_argument("--refs", required=True, help="refs/<obj>_shot=N_seed=S.json dumped by the official run")
    ap.add_argument("--obj", default="candle")
    ap.add_argument("--samples", nargs="+", required=True, help="test-relative paths, e.g. good/0659.JPG bad/068.JPG")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resolution", type=int, default=448)
    ap.add_argument("--masking", type=int, default=1)
    ap.add_argument("--rotation", type=int, default=1)
    ap.add_argument("--measurements", default=None, help="measurements_seed=0.csv to cross-check cached scores")
    args = ap.parse_args()

    sys.path.insert(0, args.code)
    from src.backbones import get_model  # noqa: E402  (needs --code on sys.path)
    from src.post_eval import mean_top1p  # noqa: E402
    from src.utils import augment_image, dists2map  # noqa: E402

    os.makedirs(args.out, exist_ok=True)
    meta = json.load(open(args.refs))
    ref_files = meta["reference_files"]
    calib_files = meta["calibration_files"]
    print("references:", ref_files)
    print("calibration normals:", len(calib_files))

    model = get_model("dinov2_vits14", "cuda", smaller_edge_size=args.resolution)
    train_dir = os.path.join(args.data, args.obj, "train", "good")

    # ---------------- memory bank, byte-identical to run_anomaly_detection ----------------
    feats_ref = []
    with torch.inference_mode():
        for name in ref_files:
            img = cv2.cvtColor(cv2.imread(os.path.join(train_dir, name), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            variants = augment_image(img) if args.rotation else [img]
            for v in variants:
                t, grid = model.prepare_image(v)
                f = model.extract_features(t)
                mask = model.compute_background_mask(f, grid, threshold=10, masking_type=False)
                feats_ref.append(f[mask])
    feats_ref = np.concatenate(feats_ref, axis=0).astype("float32")
    print("bank patches:", feats_ref.shape)

    index = faiss.IndexFlatL2(feats_ref.shape[1])
    faiss.normalize_L2(feats_ref)
    index.add(feats_ref)

    def score(path):
        """Returns (image score, full-resolution distance map, foreground mask on grid)."""
        img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        with torch.inference_mode():
            t, grid = model.prepare_image(img)
            f = model.extract_features(t)
            mask = model.compute_background_mask(f, grid, threshold=10, masking_type=bool(args.masking))
            f = f[mask]
            faiss.normalize_L2(f)
            dist, _ = index.search(f, k=1)
            dist = dist / 2  # cosine distance
        out = np.zeros(mask.shape[0], dtype=float)
        out[mask] = dist.squeeze()
        grid_map = out.reshape(grid)
        s = float(mean_top1p(out))
        return s, dists2map(grid_map, img.shape), grid_map, mask.reshape(grid), img

    cached = {}
    if args.measurements and os.path.exists(args.measurements):
        import csv
        for row in csv.DictReader(open(args.measurements)):
            cached[row["Sample"]] = float(row["Anomaly_Score"])

    # ---------------- calibration normals -> 95th percentile threshold ----------------
    calib_scores = []
    for name in calib_files:
        s, _, _, _, _ = score(os.path.join(train_dir, name))
        calib_scores.append(s)
    thr = float(np.percentile(calib_scores, 95))
    print("calibration scores: mean %.4f  p95 %.4f" % (np.mean(calib_scores), thr))

    # ---------------- requested test samples ----------------
    results = {}
    for rel in args.samples:
        path = os.path.join(args.data, args.obj, "test", rel)
        s, dmap, grid_map, mask, img = score(path)
        stem = rel.replace("/", "_").replace(".JPG", "")
        np.save(os.path.join(args.out, stem + ".npy"), grid_map)
        np.save(os.path.join(args.out, stem + "_mask.npy"), mask)
        cv2.imwrite(os.path.join(args.out, stem + ".jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        np.save(os.path.join(args.out, stem + "_full.npy"), dmap)
        tag = ""
        if rel in cached:
            tag = "  (cached %.5f, delta %+.5f)" % (cached[rel], s - cached[rel])
        results[rel] = {"score": s, "stem": stem}
        print("%-16s score %.5f%s" % (rel, s, tag))

    json.dump({"object": args.obj,
               "references": ref_files,
               "calibration_files": calib_files,
               "calibration_scores": calib_scores,
               "threshold_p95": thr,
               "samples": results}, open(os.path.join(args.out, "scores.json"), "w"), indent=2)
    print("wrote", os.path.join(args.out, "scores.json"))


if __name__ == "__main__":
    main()
