"""Build the VisA 1cls layout used by AnomalyDINO from the official split CSV.

Layout produced (identical to spot-diff `utils/prepare_data.py --split-type 1cls`):
    <dst>/<cat>/train/good/*.JPG
    <dst>/<cat>/test/good/*.JPG
    <dst>/<cat>/test/bad/*.JPG
    <dst>/<cat>/ground_truth/bad/*.png      (binarised to 0/255)

Images are symlinked by default to avoid duplicating ~0.7 GB on a nearly full shared disk;
masks are copied because they need binarisation. Use --copy to materialise images.
"""

import argparse
import csv
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from p2common import VISA_6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="official split_csv/1cls.csv")
    ap.add_argument("--src", required=True, help="VisA root holding <cat>/Data/...")
    ap.add_argument("--dst", required=True, help="output VisA_pytorch/1cls root")
    ap.add_argument("--categories", nargs="*", default=VISA_6)
    ap.add_argument("--copy", action="store_true")
    args = ap.parse_args()

    cats = set(args.categories)
    for c in args.categories:
        for sub in ("train/good", "test/good", "test/bad", "ground_truth/bad"):
            os.makedirs(os.path.join(args.dst, c, sub), exist_ok=True)

    n_img = n_mask = 0
    with open(args.csv, newline="") as f:
        for row in csv.DictReader(f):
            obj, split, label = row["object"], row["split"], row["label"]
            if obj not in cats:
                continue
            lbl = "good" if label == "normal" else "bad"
            image_src = os.path.join(args.src, row["image"])
            image_dst = os.path.join(args.dst, obj, split, lbl, os.path.basename(image_src))
            if os.path.lexists(image_dst):
                os.remove(image_dst)
            if args.copy:
                import shutil
                shutil.copyfile(image_src, image_dst)
            else:
                os.symlink(os.path.abspath(image_src), image_dst)
            n_img += 1

            if split == "test" and lbl == "bad":
                mask_src = os.path.join(args.src, row["mask"])
                m = np.array(Image.open(mask_src))
                m = np.where(m != 0, 255, 0).astype(np.uint8)
                out = os.path.join(args.dst, obj, "ground_truth", "bad",
                                   os.path.basename(mask_src))
                Image.fromarray(m).save(out)
                n_mask += 1

    print(f"done: {n_img} images, {n_mask} masks -> {args.dst}")
    for c in args.categories:
        tr = len(os.listdir(os.path.join(args.dst, c, "train", "good")))
        te = len(os.listdir(os.path.join(args.dst, c, "test", "good")))
        ba = len(os.listdir(os.path.join(args.dst, c, "test", "bad")))
        gt = len(os.listdir(os.path.join(args.dst, c, "ground_truth", "bad")))
        print(f"  {c:12s} train={tr:4d} test_good={te:4d} test_bad={ba:4d} masks={gt:4d}")


if __name__ == "__main__":
    main()
