"""Dump per-image anomaly scores (and chosen score maps) for one frozen grid cell.

`run_grid.py` only keeps the aggregate AUROC per (category, shot, seed), which is enough for
the tables but not for the error analysis in the report: deciding "both right / only DINO right
/ only CLIP right / both wrong" needs the score of every single test image, and the heatmap panel
needs the score map of the specific images that were misjudged.

Everything here reuses `run_grid.py`'s own helpers (`cache_key`, `get_features`, `build_bank`,
`score_image`), so a dumped score is by construction the same number that went into the AUROC.
Features come from the on-disk cache built by the grid run, so this pass costs one k-NN pass per
image -- seconds per category, not minutes.

The images whose maps are kept are chosen *before* this runs (see `pick_cases.py`) and passed in
with `--cases`, so both backbones dump maps for exactly the same images and the panel is a true
like-for-like comparison.

Output
------
<out>/<name>.json
    {"config": {...}, "shot": 1, "seed": 0, "threshold": {cat: thr},
     "categories": {cat: {"paths": [...], "labels": [...], "scores": [...],
                          "calib_scores": [...], "thr": thr, "auroc": auroc,
                          "map_files": {"good/stem": {"score": s, "label": 0}}}}}
<out>/maps/<name>/<cat>__<split>_<stem>.npy

Usage
-----
python scripts/dump_scores.py --name e1_clip448_L9_nomask --backbone clip --layers 9 \
    --clip_resize_mode short_edge --no_mask --cases cases.json
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "src"))

import p2common as C
import run_grid as G  # noqa: E402  (sets thread env vars on import, by design)
from ad_core import LayerDINOv2, build_bank, make_index, mean_top1p  # noqa: E402
from clip_core import LayerCLIP  # noqa: E402


def build_model(args):
    if args.backbone == "clip":
        model = LayerCLIP(args.model, args.device, args.resolution, tuple(args.layers),
                          pos_embedding_mode=args.pos_embedding_mode,
                          download_root=args.clip_download_root,
                          resize_mode=args.clip_resize_mode)
        return model, args.clip_resize_mode
    model = LayerDINOv2(args.model, args.device, args.resolution, tuple(args.layers))
    return model, "std"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--backbone", default="dinov2", choices=["dinov2", "clip"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--resolution", type=int, default=448)
    ap.add_argument("--layers", nargs="+", type=int, default=[12])
    ap.add_argument("--pos_embedding_mode", default="interp", choices=["native", "interp"])
    ap.add_argument("--clip_resize_mode", default="short_edge",
                    choices=["short_edge_crop", "squash", "short_edge"])
    ap.add_argument("--clip_download_root", default="/data/disk1/home/slv176/models/clip")
    ap.add_argument("--no_mask", dest="use_mask", action="store_false")
    ap.add_argument("--data_root", default="/tmp/slv176/data/VisA_pytorch/1cls")
    ap.add_argument("--cache", default="/data/disk1/home/slv176/featcache32")
    ap.add_argument("--out", default="/tmp/slv176/grid/scores")
    ap.add_argument("--shot", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calib_exclude", type=int, default=20)
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--cases", default=None,
                    help='JSON {"cat": ["good/stem", ...]} whose score maps are kept')
    ap.add_argument("--gpu", default=True, action=argparse.BooleanOptionalAction)
    args = ap.parse_args()

    if args.model is None:  # same normalisation as run_grid.parse_args()
        args.model = "ViT-B/16" if args.backbone == "clip" else "dinov2_vits14"
    args.device = "cuda" if args.gpu else "cpu"
    layers = tuple(args.layers)
    cats = args.categories or C.VISA_6
    root = args.data_root
    os.makedirs(args.out, exist_ok=True)
    map_root = os.path.join(args.out, "maps", args.name)
    want = json.load(open(args.cases)) if args.cases else {}
    if want:
        os.makedirs(map_root, exist_ok=True)

    model, prep = build_model(args)
    result = {"config": {k: v for k, v in vars(args).items() if k != "device"},
              "backbone": args.backbone, "layers": list(layers),
              "resolution": args.resolution, "use_mask": args.use_mask,
              "shot": args.shot, "seed": args.seed, "threshold": {}, "categories": {}}

    for cat in cats:
        calib_names, pool = C.ref_pool(root, cat, args.calib_exclude)
        ref_dir = C.train_dir(root, cat)
        items = C.test_items(root, cat)

        feats_all, masks_all = [], []
        for n in C.shot_refs(pool, args.shot, args.seed):
            f, _, m = G.get_features(model, args.cache, cat, os.path.join(ref_dir, n),
                                     G.cache_key(os.path.join(ref_dir, n)), 8, layers,
                                     mask_query=False, backbone=args.backbone,
                                     use_mask=args.use_mask, resolution=args.resolution,
                                     prep=prep)
            feats_all += f
            masks_all += m
        index = [make_index(b) for b in build_bank(feats_all, masks_all)]

        paths, labels, scores, maps = [], [], [], []
        for label, img_path, mask_path in items:
            f, grid, m = G.get_features(model, args.cache, cat, img_path, G.cache_key(img_path),
                                        1, layers, backbone=args.backbone,
                                        use_mask=args.use_mask, resolution=args.resolution,
                                        prep=prep)
            d = G.score_image(index, f[0], m[0], "single", None)
            paths.append(os.path.relpath(img_path, root))
            labels.append(int(label))
            scores.append(float(mean_top1p(d)))
            maps.append((d, grid, mask_path))

        calib_scores = []
        for n in calib_names:
            f, grid, m = G.get_features(model, args.cache, cat, os.path.join(ref_dir, n),
                                        G.cache_key(os.path.join(ref_dir, n)), 1, layers,
                                        backbone=args.backbone, use_mask=args.use_mask,
                                        resolution=args.resolution, prep=prep)
            calib_scores.append(float(mean_top1p(G.score_image(index, f[0], m[0], "single", None))))
        thr = C.percentile_threshold(calib_scores, 95.0)

        rec = {"paths": paths, "labels": labels, "scores": scores,
               "calib_scores": calib_scores, "thr": thr,
               "auroc": C.image_auroc(scores, labels)}

        keep = set(want.get(cat, []))
        for i, rel in enumerate(paths):
            split, fn = rel.replace("\\", "/").split("/")[-2:]
            stem = os.path.splitext(fn)[0]
            if f"{split}/{stem}" not in keep:
                continue
            d, grid, mask_path = maps[i]
            np.save(os.path.join(map_root, f"{cat}__{split}_{stem}.npy"),
                    G.cv2_resize(d.reshape(grid), C.EVAL_SIZE).astype(np.float32))
            rec.setdefault("map_files", {})[f"{split}/{stem}"] = {
                "score": float(scores[i]), "label": int(labels[i]),
                "has_gt": bool(mask_path is not None)}

        result["threshold"][cat] = thr
        result["categories"][cat] = rec
        print(f"[{args.name}] {cat}: auroc={rec['auroc']:.4f} thr={thr:.4f} n={len(paths)} "
              f"maps={len(rec.get('map_files', {}))}", flush=True)

    out_path = os.path.join(args.out, f"{args.name}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    mean = float(np.mean([r["auroc"] for r in result["categories"].values()]))
    print(f"wrote {out_path}  six-category image AUROC = {mean * 100:.2f}")


if __name__ == "__main__":
    main()
