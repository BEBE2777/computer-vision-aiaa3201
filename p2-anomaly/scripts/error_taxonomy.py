"""Two-backbone error taxonomy: both right / only one right / both wrong.

AUROC is threshold-free, so it cannot say *which* images a method gets wrong. For the report's
qualitative section each test image is given a verdict from its own backbone's calibration
threshold (95th percentile of the 20 normal calibration images, never a test label), and the two
verdicts are crossed into a 2x2 table.

The fourth cell of the final taxonomy ("only the VLM is right") needs the Qwen branch's per-image
scores, which are produced by the other half of the project. This script therefore writes the
visual-only 2x2 now and leaves a `vlm` column empty in the per-image CSV, so the fourth cell can
be filled in later without recomputing anything.

Usage
-----
python scripts/error_taxonomy.py --a scores/e1_dino448_nomask.json \
    --b scores/e1_clip448_L9_nomask.json --a-name "DINOv2@448" --b-name "CLIP@448" \
    --out results/part3/error_taxonomy.md --csv results/part3/error_cases.csv
"""

import argparse
import csv
import json
import os

CELLS = ("both_right", "only_a_right", "only_b_right", "both_wrong")
LABEL = {
    "both_right": "both right",
    "only_a_right": "only A right",
    "only_b_right": "only B right",
    "both_wrong": "both wrong",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--a-name", default="DINOv2@448")
    ap.add_argument("--b-name", default="CLIP@448")
    ap.add_argument("--out", default="error_taxonomy.md")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    da, db = json.load(open(args.a)), json.load(open(args.b))
    cats = [c for c in da["categories"] if c in db["categories"]]

    rows, tally = [], {}
    for cat in cats:
        ra, rb = da["categories"][cat], db["categories"][cat]
        t = {k: 0 for k in CELLS}
        t.update({"n_defect": 0, "n_normal": 0})
        for i, rel in enumerate(ra["paths"]):
            if i >= len(rb["paths"]) or rb["paths"][i] != rel:
                continue
            label = ra["labels"][i]
            sa, sb = ra["scores"][i], rb["scores"][i]
            ok_a = (sa > ra["thr"]) == bool(label)
            ok_b = (sb > rb["thr"]) == bool(label)
            key = ("both_right" if ok_a and ok_b else
                   "only_a_right" if ok_a else
                   "only_b_right" if ok_b else "both_wrong")
            t[key] += 1
            t["n_defect" if label == 1 else "n_normal"] += 1
            rows.append({"category": cat, "image": rel, "label": "defect" if label else "normal",
                         f"score_{args.a_name}": round(sa, 5),
                         f"score_{args.b_name}": round(sb, 5),
                         "verdict": LABEL[key], "vlm_verdict": ""})
        t["category"] = cat
        tally[cat] = t

    tot = {k: sum(t[k] for t in tally.values()) for k in CELLS}
    tot.update({k: sum(t[k] for t in tally.values()) for k in ("n_defect", "n_normal")})
    n = tot["n_defect"] + tot["n_normal"]

    md = [f"# Error taxonomy -- {args.a_name} vs {args.b_name}", "",
          "Verdict per image from each backbone's own calibration threshold "
          "(95th percentile of 20 normal calibration images); no test label is used to set it.",
          "",
          f"| category | n | both right | only {args.a_name} right | only {args.b_name} right "
          "| both wrong |", "|---|---|---|---|---|---|"]
    for cat in cats:
        t = tally[cat]
        md.append(f"| {cat} | {t['n_defect'] + t['n_normal']} | {t['both_right']} "
                  f"| {t['only_a_right']} | {t['only_b_right']} | {t['both_wrong']} |")
    md.append(f"| **all six** | {n} | {tot['both_right']} | {tot['only_a_right']} "
              f"| {tot['only_b_right']} | {tot['both_wrong']} |")
    md += ["", f"Share of all test images: both right {tot['both_right'] / n:.1%}, "
               f"only {args.a_name} {tot['only_a_right'] / n:.1%}, "
               f"only {args.b_name} {tot['only_b_right'] / n:.1%}, "
               f"both wrong {tot['both_wrong'] / n:.1%}.", "",
           f"Defects missed by both: {sum(1 for r in rows if r['verdict'] == 'both wrong' and r['label'] == 'defect')}"
           f" of {tot['n_defect']} defects; normal images flagged by both: "
           f"{sum(1 for r in rows if r['verdict'] == 'both wrong' and r['label'] == 'normal')}"
           f" of {tot['n_normal']} normals.", "",
           "The `vlm_verdict` column in the companion CSV is intentionally empty: the fourth "
           "taxonomy cell (only the VLM is right) needs the Qwen branch's per-image scores."]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    open(args.out, "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print("\nwrote", args.csv)


if __name__ == "__main__":
    main()
