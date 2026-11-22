"""Offline score-level fusion for the Part 3 branches (E3/E4, reusable for E5).

Fusion only needs *saved* per-image predictions -- (category, image, label, score) for every
branch plus the 20 normal calibration scores per category -- so the whole alpha sweep runs in
milliseconds on a laptop and is reproducible from the repository. No model, no GPU, no test
label in any decision (the test labels are read only to compute AUROC at the end).

Pre-registered protocol (docs/part3/实验规划与两人分工.md, sections 3.2 and E3-E5)
------------------------------------------------------------------------------
For every (category, shot, seed) and every branch b, with the 20 held-out normal calibration
images of that category:

    z_b = (s_b - median(calib_b)) / scale_b
    scale_b = max(1.4826 * MAD(calib_b), epsilon),  MAD = median(|calib_b - median(calib_b)|)
    fused   = alpha * z_1 + (1 - alpha) * z_2

Degenerate scale falls back to the calibration standard deviation and then to a fixed epsilon
(1e-6); which fallback fired is recorded per (category, branch) in the output JSON. Larger
score means more anomalous for every branch, so `fused` keeps that direction.

Main result is alpha = 0.5, fixed *before* looking at any result. alpha in {0, 0.25, 0.5, 0.75, 1}
is reported as a sensitivity analysis only; alpha = 1 and alpha = 0 must reproduce the
single-branch AUROC of branch 1 and branch 2 exactly (data-alignment and direction check).

Inputs
------
dump JSON (the format written by ``scripts/dump_scores.py``)::

    {"shot": 1, "seed": 0,
     "categories": {"candle": {"paths": [...], "labels": [...], "scores": [...],
                               "calib_scores": [...]}, ...}}

predictions JSONL (the schema the Qwen branch is expected to write; one row per query)::

    {"category": "candle", "image": "candle/test/bad/0001.JPG", "label": 1,
     "anomaly_score": 0.83, "is_calibration": false}

    For a calibration row, ``is_calibration`` is true (or ``split`` is "calibration").

Usage
-----
# machinery check on the two visual arms, 1-shot seed 0
python scripts/score_fusion.py \
    --b1 dinov2=results/part3/scores/e1_dino448_nomask.json \
    --b2 clip=results/part3/scores/e1_clip448_L9_nomask.json \
    --tag visual_pair_1shot_seed0 --plot

# once the Qwen branch has dumped per-image scores
python scripts/score_fusion.py \
    --b1 dinov2=results/part3/scores/e1_dino448_nomask.json \
    --b2 qwen=results/part3/scores/qwen_1shot_seed0.jsonl --b2-format jsonl \
    --tag e3_qwen_1shot_seed0 --plot

# unit checks that do not need any file
python scripts/score_fusion.py --selftest
"""

import argparse
import json
import os
import sys

import numpy as np

VISA_6 = ["candle", "capsules", "cashew", "chewinggum", "fryum", "pipe_fryum"]  # p2common.VISA_6
EPS = 1e-6
MAD_TO_SIGMA = 1.4826
DEFAULT_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


# ---------------------------------------------------------------- metrics
def auroc(scores, labels):
    """Exact AUROC via average ranks (identical to sklearn.roc_auc_score, ties handled)."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    sorted_s = s[order]
    ranks = np.empty(s.size, dtype=np.float64)
    i = 0
    while i < s.size:                      # average ranks inside each tie block
        j = i
        while j + 1 < s.size and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auroc_bruteforce(scores, labels):
    """O(n^2) reference implementation, only for --selftest."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    pos, neg = s[y], s[~y]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (pos.size * neg.size))


# ---------------------------------------------------------------- normalisation
def robust_scale(calib):
    """median / MAD scale with the pre-registered fallback chain. Returns (median, scale, how)."""
    c = np.asarray(calib, dtype=np.float64)
    med = float(np.median(c))
    mad = float(np.median(np.abs(c - med)))
    scale = MAD_TO_SIGMA * mad
    how = "mad"
    if not np.isfinite(scale) or scale <= EPS:
        scale = float(np.std(c))
        how = "std"
    if not np.isfinite(scale) or scale <= EPS:
        scale = EPS
        how = "eps"
    return med, float(scale), how


def zscore(scores, med, scale):
    return (np.asarray(scores, dtype=np.float64) - med) / scale


# ---------------------------------------------------------------- inputs
def _norm_key(path):
    return str(path).replace("\\", "/").lstrip("./")


def load_dump(path):
    """dump_scores.py JSON -> {category: {"test": {...}, "calib": np.array}}."""
    with open(path) as f:
        d = json.load(f)
    out = {"meta": {"shot": d.get("shot"), "seed": d.get("seed"),
                    "backbone": d.get("backbone"), "path": path},
           "categories": {}}
    for cat, rec in d["categories"].items():
        out["categories"][cat] = {
            "paths": [_norm_key(p) for p in rec["paths"]],
            "labels": np.asarray(rec["labels"], dtype=int),
            "scores": np.asarray(rec["scores"], dtype=np.float64),
            "calib": np.asarray(rec["calib_scores"], dtype=np.float64),
        }
    return out


def load_jsonl(path):
    """Predictions JSONL -> the same structure as load_dump()."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    cats = {}
    for r in rows:
        cat = r["category"]
        is_calib = bool(r.get("is_calibration")) or str(r.get("split", "")).lower() in (
            "calibration", "calib")
        if is_calib:
            cats.setdefault(cat, {"calib_rows": []})["calib_rows"].append(float(r["anomaly_score"]))
        else:
            rec = cats.setdefault(cat, {"calib_rows": [], "paths": [], "labels": [], "scores": []})
            rec["paths"].append(_norm_key(r["image"]))
            rec["labels"].append(int(r["label"]))
            rec["scores"].append(float(r["anomaly_score"]))
    out = {"meta": {"shot": None, "seed": None, "path": path}, "categories": {}}
    for cat, rec in cats.items():
        out["categories"][cat] = {
            "paths": rec["paths"],
            "labels": np.asarray(rec["labels"], dtype=int),
            "scores": np.asarray(rec["scores"], dtype=np.float64),
            "calib": np.asarray(sorted(rec["calib_rows"]), dtype=np.float64),
        }
    return out


def parse_branch(spec):
    """'name=path' -> (name, path)."""
    if "=" not in spec:
        raise SystemExit(f"--b1/--b2 expect name=path, got {spec!r}")
    name, path = spec.split("=", 1)
    return name.strip(), path


# ---------------------------------------------------------------- alignment
def align(a, b, require_exact=True):
    """Intersect the two branches on (category, image path); fail loudly on mismatch."""
    common, report = {}, []
    for cat in VISA_6:
        if cat not in a["categories"] or cat not in b["categories"]:
            report.append(f"{cat}: missing in one branch, skipped")
            continue
        ra, rb = a["categories"][cat], b["categories"][cat]
        ia = {p: i for i, p in enumerate(ra["paths"])}
        ib = {p: i for i, p in enumerate(rb["paths"])}
        keys = [p for p in ra["paths"] if p in ib]
        report.append(f"{cat}: {len(keys)}/{len(ra['paths'])} aligned")
        if require_exact and (len(keys) != len(ra["paths"]) or len(keys) != len(rb["paths"])):
            missing_a = sorted(set(ib) - set(ia))[:3]
            missing_b = sorted(set(ia) - set(ib))[:3]
            raise SystemExit(
                f"alignment error in {cat}: {len(ra['paths'])} vs {len(rb['paths'])} images; "
                f"only-in-b {missing_a}, only-in-a {missing_b}. "
                f"Pass --allow_partial to score the intersection instead.")
        ja = np.array([ia[p] for p in keys])
        jb = np.array([ib[p] for p in keys])
        la, lb = ra["labels"][ja], rb["labels"][jb]
        if not np.array_equal(la, lb):
            raise SystemExit(f"label mismatch in {cat}: the two branches disagree on ground truth")
        common[cat] = {"paths": keys, "labels": la,
                       "s1": ra["scores"][ja], "s2": rb["scores"][jb],
                       "calib1": ra["calib"], "calib2": rb["calib"]}
    return common, report


# ---------------------------------------------------------------- fusion
def fuse_cells(cells, alphas, main_alpha):
    """Normalise both branches per category and evaluate every alpha."""
    per_cat = {}
    for cat, c in cells.items():
        med1, sc1, how1 = robust_scale(c["calib1"])
        med2, sc2, how2 = robust_scale(c["calib2"])
        z1, z2 = zscore(c["s1"], med1, sc1), zscore(c["s2"], med2, sc2)
        cz1, cz2 = zscore(c["calib1"], med1, sc1), zscore(c["calib2"], med2, sc2)
        per_cat[cat] = {"labels": c["labels"], "z1": z1, "z2": z2, "calib_z1": cz1,
                        "calib_z2": cz2, "scale": {1: how1, 2: how2},
                        "median": {1: med1, 2: med2}, "sigma": {1: sc1, 2: sc2},
                        "n": int(len(c["labels"]))}

    rows = []
    for alpha in alphas:
        per_cat_auc, all_s, all_y = {}, [], []
        for cat, c in per_cat.items():
            fused = alpha * c["z1"] + (1 - alpha) * c["z2"]
            per_cat_auc[cat] = auroc(fused, c["labels"])
            all_s.append(fused)
            all_y.append(c["labels"])
        macro = float(np.mean([v for v in per_cat_auc.values() if np.isfinite(v)]))
        pooled = auroc(np.concatenate(all_s), np.concatenate(all_y))
        rows.append({"alpha": alpha, "macro": macro, "pooled": pooled, "per_category": per_cat_auc})

    main = next(r for r in rows if abs(r["alpha"] - main_alpha) < 1e-12)
    decisions = [decision_table(per_cat, r["alpha"]) for r in rows]
    decision = next(d for d, r in zip(decisions, rows) if abs(r["alpha"] - main_alpha) < 1e-12)
    return rows, main, decision, per_cat, decisions


def decision_table(cells, alpha, q=95.0):
    """Fused OK/NG at the 95th percentile of the fused *calibration* scores (no test label)."""
    tot = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    out = {}
    for cat, c in cells.items():
        fused_calib = alpha * c["calib_z1"] + (1 - alpha) * c["calib_z2"]
        thr = float(np.percentile(fused_calib, q))
        fused = alpha * c["z1"] + (1 - alpha) * c["z2"]
        pred = fused >= thr
        y = c["labels"].astype(bool)
        conf = {"tp": int((pred & y).sum()), "fp": int((pred & ~y).sum()),
                "tn": int((~pred & ~y).sum()), "fn": int((~pred & y).sum()), "thr": thr}
        out[cat] = conf
        for k in tot:
            tot[k] += conf[k]
    rec = tot["tp"] / max(tot["tp"] + tot["fn"], 1)
    fpr = tot["fp"] / max(tot["fp"] + tot["tn"], 1)
    return {"per_category": out, "total": tot,
            "recall": rec, "fpr": fpr, "threshold_quantile": q}


# ---------------------------------------------------------------- selftest
def selftest(dumps=()):
    rng = np.random.default_rng(0)
    m = 0
    for n in (2, 7, 50, 361, 960):
        for _ in range(20):
            s = rng.normal(size=n)
            y = rng.integers(0, 2, size=n)
            if y.min() == y.max():
                continue
            d = abs(auroc(s, y) - auroc_bruteforce(s, y))
            m = max(m, d)
    # heavy ties: coarse rounding
    for n in (200, 960):
        s = np.round(rng.normal(size=n), 1)
        y = rng.integers(0, 2, size=n)
        m = max(m, abs(auroc(s, y) - auroc_bruteforce(s, y)))
    print(f"[selftest] AUROC vs brute force: max |diff| = {m:.2e}")
    assert m < 1e-12, "AUROC implementation disagrees with the brute-force reference"

    for path in dumps:
        with open(path) as f:
            raw = json.load(f)
        worst = 0.0
        for cat, rec in raw["categories"].items():
            if "auroc" not in rec:
                continue
            mine = auroc(rec["scores"], rec["labels"])
            worst = max(worst, abs(mine - float(rec["auroc"])))
        print(f"[selftest] {os.path.basename(path)}: max |AUROC - dump value| = {worst:.2e}")
        assert worst < 1e-12, f"AUROC does not reproduce the dump's sklearn value in {path}"
    print("[selftest] ok")


# ---------------------------------------------------------------- reporting
def write_outputs(rows, main, decision, per_cat, meta, out_dir, tag, plot, decisions=None):
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"fusion_{tag}")
    payload = {"meta": meta, "alphas": rows, "main": main, "decision": decision,
               "decisions": decisions,
               "normalisation": {cat: {"scale_rule": {str(k): v for k, v in c["scale"].items()},
                                       "median": {str(k): c["median"][k] for k in c["median"]},
                                       "sigma": {str(k): c["sigma"][k] for k in c["sigma"]},
                                       "n_test": c["n"]}
                                 for cat, c in per_cat.items()}}
    with open(stem + ".json", "w") as f:
        json.dump(payload, f, indent=2)

    b1, b2 = meta["branch1"], meta["branch2"]
    lines = [f"# Score fusion — {tag}", ""]
    lines += [f"* branch 1: `{b1}`", f"* branch 2: `{b2}`",
              f"* images: {meta['n_total']} test images over {len(main['per_category'])} categories",
              f"* shot = {meta['shot']}, seed = {meta['seed']} (single cell)"
              if meta["shot"] is not None else "* shot/seed: not recorded in the inputs",
              f"* main alpha (pre-registered) = {meta['main_alpha']}", ""]
    lines += ["## alpha sweep (image AUROC, %)", "",
              f"| alpha | macro over categories | pooled |  (α=1 should equal {b1}, α=0 {b2}) |",
              "|---:|---:|---:|---|"]
    for r in rows:
        note = ""
        if r["alpha"] == 1.0:
            note = "single-branch check"
        elif r["alpha"] == 0.0:
            note = "single-branch check"
        lines.append(f"| {r['alpha']:.2f} | {r['macro'] * 100:.2f} | {r['pooled'] * 100:.2f} | {note} |")
    lines += ["", f"## per-category image AUROC (%) at alpha = {meta['main_alpha']}", "",
              "| category | " + " | ".join(f"{r['alpha']:.2f}" for r in rows) + " |",
              "|---|" + "---:|" * len(rows)]
    for cat in main["per_category"]:
        vals = " | ".join(f"{r['per_category'][cat] * 100:.2f}" for r in rows)
        lines.append(f"| {cat} | {vals} |")
    lines.append("| **macro** | " + " | ".join(f"{r['macro'] * 100:.2f}" for r in rows) + " |")

    t = decision["total"]
    lines += ["", f"## decision at the fused calibration threshold (alpha = {meta['main_alpha']}, "
                  f"{decision['threshold_quantile']:.0f}th percentile of calibration scores)", "",
              "| TP | FP | TN | FN | recall | FPR |", "|---:|---:|---:|---:|---:|---:|",
              f"| {t['tp']} | {t['fp']} | {t['tn']} | {t['fn']} | "
              f"{decision['recall'] * 100:.1f}% | {decision['fpr'] * 100:.1f}% |"]
    if decisions:
        lines += ["", "Operating point of every alpha (threshold = 95th percentile of that "
                      "alpha's fused *calibration* score, per category; test labels only count "
                      "the outcome):", "",
                  "| alpha | TP | FP | TN | FN | recall | FPR |", "|---:|---:|---:|---:|---:|---:|---:|"]
        for r, d in zip(rows, decisions):
            t = d["total"]
            lines.append(f"| {r['alpha']:.2f} | {t['tp']} | {t['fp']} | {t['tn']} | {t['fn']} | "
                         f"{d['recall'] * 100:.1f}% | {d['fpr'] * 100:.1f}% |")
        lines += ["", f"At the nominal 95th percentile the *target* false-alarm rate is 5% of the "
                      f"normal test images; the realised rates above are higher (single branches "
                      f"included), i.e. 20 training-normal calibration images do not cover the "
                      f"variation of the normal test images. Report these as achieved operating "
                      f"points, not as a calibrated 5% guarantee."]
    lines += ["", "Per category: " + ", ".join(
        f"{cat} TP/FP/FN = {v['tp']}/{v['fp']}/{v['fn']}"
        for cat, v in decision["per_category"].items())]
    lines += ["", "Calibration scale per category (rule fired / median / sigma): " + "; ".join(
        f"{cat} b1={c['scale'][1]}({c['median'][1]:.3f}, {c['sigma'][1]:.4f}) "
        f"b2={c['scale'][2]}({c['median'][2]:.3f}, {c['sigma'][2]:.4f})"
        for cat, c in per_cat.items())]
    lines += ["", "Note: alpha is *reported*, not selected — the main column is the pre-registered "
                  "0.5, and the sweep is a sensitivity analysis (sections 3.2 and E4 of the plan)."]
    with open(stem + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {stem}.json / .md")

    if plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed, skipping alpha curve")
            return stem
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        xs = [r["alpha"] for r in rows]
        ax.plot(xs, [r["macro"] * 100 for r in rows], marker="o", label="macro over 6 categories")
        ax.plot(xs, [r["pooled"] * 100 for r in rows], marker="s", ls="--", label="pooled")
        ax.axvline(meta["main_alpha"], color="0.5", lw=0.8, ls=":")
        ax.set_xlabel(r"$\alpha$  (weight on " + b1 + ")")
        ax.set_ylabel("image AUROC (%)")
        ax.set_title(f"{b1} + {b2}, shot={meta['shot']} seed={meta['seed']}", fontsize=8)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        p = stem + "_alpha.png"
        fig.savefig(p, dpi=170)
        print("wrote", p)
    return stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b1", help="branch 1 as name=path (dump JSON by default)")
    ap.add_argument("--b2", help="branch 2 as name=path")
    ap.add_argument("--b2-format", default="dump", choices=["dump", "jsonl"])
    ap.add_argument("--b1-format", default="dump", choices=["dump", "jsonl"])
    ap.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS)
    ap.add_argument("--main-alpha", type=float, default=0.5)
    ap.add_argument("--out", default="results/part3/fusion")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--allow_partial", action="store_true",
                    help="score the intersection instead of failing on an incomplete branch")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dump", nargs="*", default=[],
                    help="with --selftest: dump JSONs whose stored AUROC must be reproduced")
    args = ap.parse_args()

    if args.selftest:
        selftest(args.dump)
        return
    if not (args.b1 and args.b2):
        raise SystemExit("need --b1 and --b2 (or --selftest)")

    n1, p1 = parse_branch(args.b1)
    n2, p2 = parse_branch(args.b2)
    loaders = {"dump": load_dump, "jsonl": load_jsonl}
    A, B = loaders[args.b1_format](p1), loaders[args.b2_format](p2)

    cells, report = align(A, B, require_exact=not args.allow_partial)
    if not cells:
        raise SystemExit("no common category between the two branches")
    for line in report:
        print("[align]", line)

    rows, main_row, decision, per_cat, decisions = fuse_cells(cells, args.alphas, args.main_alpha)

    # single-branch consistency: alpha=1 -> branch 1, alpha=0 -> branch 2, on the aligned set
    single = {1: {cat: auroc(c["s1"], c["labels"]) for cat, c in cells.items()},
              2: {cat: auroc(c["s2"], c["labels"]) for cat, c in cells.items()}}
    r1 = next((r for r in rows if r["alpha"] == 1.0), None)
    r0 = next((r for r in rows if r["alpha"] == 0.0), None)
    worst = 0.0
    for branch, row in ((1, r1), (2, r0)):
        if row is None:
            continue
        for cat in row["per_category"]:
            worst = max(worst, abs(row["per_category"][cat] - single[branch][cat]))
    print(f"[check] alpha=1/0 reproduce the single branches: max |diff| = {worst:.2e} AUROC")
    if worst > 1e-9:
        print("[check] WARNING: check failed — alignment or normalisation is off")

    meta = {"branch1": n1, "branch2": n2, "path1": p1, "path2": p2,
            "shot": A["meta"]["shot"], "seed": A["meta"]["seed"],
            "main_alpha": args.main_alpha, "n_total": int(sum(c["n"] for c in per_cat.values())),
            "single_branch_macro": {"branch1": float(np.mean(list(single[1].values()))),
                                    "branch2": float(np.mean(list(single[2].values())))},
            "consistency_max_abs_diff": worst,
            "alphas_requested": args.alphas}
    write_outputs(rows, main_row, decision, per_cat, meta, args.out, args.tag, args.plot,
                  decisions)

    for r in rows:
        print(f"alpha={r['alpha']:.2f}  macro={r['macro'] * 100:.2f}  pooled={r['pooled'] * 100:.2f}")
    t = decision["total"]
    print(f"decision @alpha={args.main_alpha}: recall={decision['recall'] * 100:.1f}% "
          f"FPR={decision['fpr'] * 100:.1f}%  (TP {t['tp']} / FP {t['fp']} / FN {t['fn']})")


if __name__ == "__main__":
    main()
