"""Dump a qualitative montage for one category: image / GT mask / template-difference map.

Rows are test images; columns are [RGB input, ground-truth mask, Part 1 score map, normalised for
display only]. Used to check that the score map lines up with the object and its mask, and to
show why the Part 1 image-level score saturates.

    python debug_template.py chewinggum --root <VisA_pytorch>/1cls --shot 1 --seed 0
"""

import argparse
import importlib.util
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import p2common as C

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("p1", os.path.join(_HERE, "run_part1_template.py"))
p1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("category")
    ap.add_argument("--root", default="/tmp/slv176/data/VisA_pytorch/1cls")
    ap.add_argument("--shot", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_good", type=int, default=2)
    ap.add_argument("--n_bad", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--align", default="none")
    args = ap.parse_args()

    cat = args.category
    _, pool = C.ref_pool(args.root, cat, 20)
    ref_names = C.shot_refs(pool, args.shot, args.seed)
    refs = [C.prep_square(C.load_rgb(os.path.join(C.train_dir(args.root, cat), n)))
            for n in ref_names]
    anchor = p1.gray01(refs[0])
    do_align = args.align == "ecc"

    items = C.test_items(args.root, cat)
    good = [it for it in items if it[0] == 0][:args.n_good]
    bad = [it for it in items if it[0] == 1][:args.n_bad]

    rows = []
    first = ("ref", os.path.join(C.train_dir(args.root, cat), ref_names[0]), None)
    for label, path, mask_path in [first] + good + bad:
        img = C.prep_square(C.load_rgb(path))
        d = p1.score_one(img.copy(), refs, anchor, do_align, False)
        v = d / max(float(d.max()), 1e-6)
        heat = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        m = (C.load_mask(mask_path, C.EVAL_SIZE) if mask_path
             else np.zeros((C.EVAL_SIZE,) * 2, np.uint8))
        rows.append(np.concatenate([img, np.stack([m * 255] * 3, -1), heat], axis=1))
    grid = np.concatenate(rows, axis=0)
    out = args.out or f"/tmp/slv176/debug/{cat}_montage.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print("wrote", out, "rows:", len(rows))


if __name__ == "__main__":
    main()
