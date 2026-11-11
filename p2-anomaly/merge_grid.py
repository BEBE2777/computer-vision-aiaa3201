"""Merge several `run_grid.py` result files into one table over all six categories.

`run_grid.py` writes, per run, a raw summary (`<name>.json`: per (shot, category, seed)) and a
`<name>_final.json` whose MEAN only covers the categories in that run. When the six categories
are split across several processes (scripts/run_grid_parallel.sh), each process produces its
own pair; this stitches the raw files back into one table.

    python merge_grid.py --out /tmp/slv176/grid/base_L12_final.json \
        /tmp/slv176/grid/base_L12_p1.json /tmp/slv176/grid/base_L12_p2.json ...
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import p2common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", nargs="+", help="raw `<name>.json` files written by run_grid.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cats", nargs="*", default=None)
    args = ap.parse_args()

    cats = args.cats or C.VISA_6
    merged = {"image_auroc": {}, "pixel_auroc": {}, "threshold": {}}
    config, runtime, layers, missing = None, 0.0, None, []
    for path in args.raw:
        if not os.path.isfile(path):
            missing.append(path)
            continue
        d = json.load(open(path))
        for metric in merged:
            for shot_key, per_cat in d.get(metric, {}).items():
                tgt = merged[metric].setdefault(shot_key, {})
                for cat, per_seed in per_cat.items():
                    tgt.setdefault(cat, {}).update(per_seed)
        config = config or d.get("meta")
        layers = layers or d.get("layers")
        runtime += d.get("runtime_s", 0.0) or 0.0
    if missing:
        print("WARNING: missing inputs:", ", ".join(missing))

    final = C.summarize(merged, cats)
    final["seed_values"] = {m: merged[m] for m in ("image_auroc", "pixel_auroc")}
    final["threshold"] = merged["threshold"]
    final["config"] = config
    final["layers"] = layers
    final["runtime_s"] = runtime
    final["merged_from"] = args.raw

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(final, f, indent=1)

    for metric in ("image_auroc", "pixel_auroc"):
        for shot_key in sorted(final[metric], key=lambda k: int(k.split("-")[0])):
            r = final[metric][shot_key]
            print(f"{metric:12s} {shot_key:8s} MEAN={r['MEAN']*100:6.2f} "
                  f"({r['MEAN_std']*100:.2f})  "
                  + " ".join(f"{c[:6]}={r[c]*100:.1f}" for c in cats))


if __name__ == "__main__":
    main()
