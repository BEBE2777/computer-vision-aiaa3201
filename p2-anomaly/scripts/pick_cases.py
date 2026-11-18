"""Choose the handful of test images whose score maps go into the heatmap panel.

The panel must score the *same* images with both backbones, otherwise "CLIP looks blurrier" is
confounded by "CLIP was shown an easier image". This script reads the two per-image score dumps
produced by `dump_scores.py` (no maps needed), and for every category picks one case by
priority:

  1. a defect both backbones miss   (the shared failure mode -- the interesting one)
  2. a defect that only one backbone misses (the disagreement)
  3. a false alarm on a normal image
  4. a defect both catch cleanly (contrast row)

Output is the `cases.json` consumed by `dump_scores.py --cases`.

Usage
-----
python scripts/pick_cases.py --a scores/e1_dino448_nomask.json \
    --b scores/e1_clip448_L9_nomask.json --out scores/cases.json --max-rows 6
"""

import argparse
import json
import os

PRIORITY = ("both_miss", "one_miss", "false_alarm", "both_catch")


def verdict(score, thr, label):
    return (score > thr) == bool(label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="first backbone scores (DINO)")
    ap.add_argument("--b", required=True, help="second backbone scores (CLIP)")
    ap.add_argument("--out", default="cases.json")
    ap.add_argument("--max-rows", type=int, default=6)
    ap.add_argument("--cats", nargs="*", default=None)
    args = ap.parse_args()

    da, db = json.load(open(args.a)), json.load(open(args.b))
    cats = args.cats or [c for c in da["categories"] if c in db["categories"]]

    picked, rows = {}, []
    for cat in cats:
        ra, rb = da["categories"][cat], db["categories"][cat]
        # both dumps iterate C.test_items() in the same order, so index i is the same image;
        # paths are compared as a guard against a category-order mismatch
        buckets = {k: [] for k in PRIORITY}
        for i, rel in enumerate(ra["paths"]):
            if i >= len(rb["paths"]) or rb["paths"][i] != rel:
                continue
            label = ra["labels"][i]
            ok_a = verdict(ra["scores"][i], ra["thr"], label)
            ok_b = verdict(rb["scores"][i], rb["thr"], label)
            split, fn = rel.replace("\\", "/").split("/")[-2:]
            key = f"{split}/{os.path.splitext(fn)[0]}"
            margin = abs(ra["scores"][i] - ra["thr"]) + abs(rb["scores"][i] - rb["thr"])
            if not ok_a and not ok_b:
                buckets["both_miss" if label == 1 else "false_alarm"].append((margin, key))
            elif ok_a != ok_b:
                buckets["one_miss"].append((-margin, key))
            elif label == 1:
                buckets["both_catch"].append((-margin, key))
        for kind in PRIORITY:
            if buckets[kind]:
                key = sorted(buckets[kind])[0][1]
                picked.setdefault(cat, []).append(key)
                rows.append((cat, kind, key))
                break

    rows = rows[:args.max_rows]
    out = {}
    for cat, _, key in rows:
        out.setdefault(cat, []).append(key)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")
    for cat, kind, key in rows:
        print(f"  {cat:11s} {kind:12s} {key}")


if __name__ == "__main__":
    main()
