"""Post-hoc statistical audit of the frozen v4 point estimates.

This script does not alter v4 predictions, score directions, or thresholds. It
adds label-stratified bootstrap intervals for macro-AUROC and paired method
differences, plus exact binomial intervals for thresholded FPR/recall. The
analysis is conditional on the v4 1-shot/seed0 image subset.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

BASE = Path(__file__).resolve().parent
TEST = BASE / "predictions_v4_test.json"
SUMMARY = BASE / "summary_v4.json"
MATCHED = BASE / "matched_baselines_v4.json"
OUT_JSON = BASE / "reviewed_statistics_v4.json"
OUT_MD = BASE / "reviewed_statistics_v4.md"
BOOT = 20000
SEED = 20261125
CATS = ["candle", "capsules", "cashew", "chewinggum", "fryum", "pipe_fryum"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def auc_from_pairs(pos, neg):
    return float((pos[:, None] > neg[None, :]).mean()
                 + 0.5 * (pos[:, None] == neg[None, :]).mean())


def exact_binomial_ci(k, n, alpha=0.05):
    """Two-sided Clopper-Pearson interval."""
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return [lo, hi]


def main():
    predictions = read(TEST)["records"]
    summary = read(SUMMARY)
    rows = read(MATCHED)["rows"]
    rng = np.random.default_rng(SEED)

    scores = {
        "qwen": {k: np.array([r["qwen_fc_score_signflip"] for r in rows if r["category"] == k], dtype=float) for k in CATS},
        "dino": {k: np.array([r["dino_score"] for r in rows if r["category"] == k], dtype=float) for k in CATS},
        "clip": {k: np.array([r["clip_score"] for r in rows if r["category"] == k], dtype=float) for k in CATS},
    }
    labels = {k: np.array([1 if r["label"] == "bad" else 0 for r in rows if r["category"] == k], dtype=int) for k in CATS}
    point = {}
    for method in scores:
        point[method] = float(np.mean([
            auc_from_pairs(scores[method][c][labels[c] == 1], scores[method][c][labels[c] == 0])
            for c in CATS
        ]))

    boot = {k: [] for k in ("qwen", "dino", "clip", "qwen_minus_dino", "qwen_minus_clip")}
    for _ in range(BOOT):
        aucs = {m: [] for m in ("qwen", "dino", "clip")}
        for c in CATS:
            pos_idx = np.flatnonzero(labels[c] == 1)
            neg_idx = np.flatnonzero(labels[c] == 0)
            pidx = rng.choice(pos_idx, size=len(pos_idx), replace=True)
            nidx = rng.choice(neg_idx, size=len(neg_idx), replace=True)
            for method in aucs:
                p = scores[method][c][pidx]
                n = scores[method][c][nidx]
                aucs[method].append(float((p[:, None] > n[None, :]).mean()
                                          + 0.5 * (p[:, None] == n[None, :]).mean()))
        macro = {m: float(np.mean(v)) for m, v in aucs.items()}
        for method in ("qwen", "dino", "clip"):
            boot[method].append(macro[method])
        boot["qwen_minus_dino"].append(macro["qwen"] - macro["dino"])
        boot["qwen_minus_clip"].append(macro["qwen"] - macro["clip"])

    ci = {key: [float(v) for v in np.quantile(values, [0.025, 0.975])]
          for key, values in boot.items()}
    ci["bootstrap_replicates"] = BOOT
    ci["seed"] = SEED

    thresholds = summary["qwen_fc_conf"]["thresholds"]
    op = {}
    for c in CATS:
        rs = [r for r in predictions if r["category"] == c]
        thr = thresholds[c]
        normal = [r for r in rs if r["true_label_for_eval_ONLY"] == "good"]
        anomalous = [r for r in rs if r["true_label_for_eval_ONLY"] == "bad"]
        fp = sum(-r["fc_score"] >= thr for r in normal)
        tp = sum(-r["fc_score"] >= thr for r in anomalous)
        op[c] = {
            "fp": fp, "n_normal": len(normal), "fpr": fp / len(normal),
            "fpr_ci95_clopper_pearson": exact_binomial_ci(fp, len(normal)),
            "tp": tp, "n_anomalous": len(anomalous), "recall": tp / len(anomalous),
            "recall_ci95_clopper_pearson": exact_binomial_ci(tp, len(anomalous)),
        }

    result = {
        "scope": "v4 fixed 1-shot/seed0 696-image subset; intervals condition on this configuration",
        "macro_auroc_point": point,
        "macro_auroc_ci95_label_stratified_bootstrap": {k: ci[k] for k in ("qwen", "dino", "clip")},
        "paired_macro_auroc_difference_point": {
            "qwen_minus_dino": point["qwen"] - point["dino"],
            "qwen_minus_clip": point["qwen"] - point["clip"],
        },
        "paired_macro_auroc_difference_ci95_label_stratified_bootstrap": {
            "qwen_minus_dino": ci["qwen_minus_dino"],
            "qwen_minus_clip": ci["qwen_minus_clip"],
        },
        "bootstrap": {"replicates": BOOT, "seed": SEED, "resampling": "within each category and true-label stratum; same sampled images paired across methods"},
        "operating_points_exact_binomial_ci95": op,
        "threshold_estimator_note": "Qwen v4 implementation selects sorted calibration score at index int(20*0.95)=19 (the maximum); E1 DINO/CLIP uses numpy.percentile(...,95) linear interpolation. Calibration images match, percentile estimators differ.",
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    # Keep the primary summary artifact consistent with the corrected intervals.
    summary["qwen_fc_conf"]["macro_ci95"] = ci["qwen"]
    summary["qwen_fc_conf"]["fpr_ci95"] = {
        c: op[c]["fpr_ci95_clopper_pearson"] for c in CATS
    }
    summary["qwen_fc_conf"]["recall_ci95"] = {
        c: op[c]["recall_ci95_clopper_pearson"] for c in CATS
    }
    summary["qwen_fc_conf"]["operating_point_ci_method"] = "two-sided Clopper-Pearson exact binomial"
    summary["dino"]["macro_ci95"] = ci["dino"]
    summary["clip"]["macro_ci95"] = ci["clip"]
    summary["paired_macro_auroc_comparisons"] = {
        "bootstrap_replicates": BOOT,
        "seed": SEED,
        "resampling": "within category and true-label strata; paired images across methods",
        "qwen_minus_dino": {
            "point": result["paired_macro_auroc_difference_point"]["qwen_minus_dino"],
            "ci95": ci["qwen_minus_dino"],
        },
        "qwen_minus_clip": {
            "point": result["paired_macro_auroc_difference_point"]["qwen_minus_clip"],
            "ci95": ci["qwen_minus_clip"],
        },
    }
    SUMMARY.write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# v4 statistical review addendum", "",
        "This post-hoc addendum preserves all frozen predictions, directions, thresholds, and point estimates. AUROC uncertainty is recomputed with label-stratified paired bootstrap; FPR/recall use exact two-sided Clopper–Pearson intervals.", "",
        f"Bootstrap: {BOOT:,} resamples, seed {SEED}; within each category, good and bad images are resampled separately, and the same resampled images are paired across methods.", "",
        "## Macro AUROC and paired differences", "",
        "| Comparison | Point estimate | 95% interval |", "|---|---:|---:|",
    ]
    for m in ("qwen", "dino", "clip"):
        lo, hi = ci[m]
        lines.append(f"| {m.upper()} macro AUROC | {point[m]:.4f} | [{lo:.4f}, {hi:.4f}] |")
    for m in ("qwen_minus_dino", "qwen_minus_clip"):
        lo, hi = ci[m]
        delta = point["qwen"] - point["dino" if m.endswith("dino") else "clip"]
        lines.append(f"| {m.replace('_', ' ').upper()} macro AUROC difference | {delta:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    lines += ["", "The paired intervals estimate differences on this fixed 696-image subset and 1-shot/seed0 configuration. They do not measure variation across shots, reference seeds, or datasets.", "", "## Threshold operating-point intervals", "", "| Category | FPR (FP/n, exact 95% CI) | Recall (TP/n, exact 95% CI) |", "|---|---|---|"]
    for c in CATS:
        x = op[c]
        fl, fu = x["fpr_ci95_clopper_pearson"]
        rl, ru = x["recall_ci95_clopper_pearson"]
        lines.append(f"| {c} | {x['fpr']:.3f} ({x['fp']}/{x['n_normal']}; [{fl:.3f}, {fu:.3f}]) | {x['recall']:.3f} ({x['tp']}/{x['n_anomalous']}; [{rl:.3f}, {ru:.3f}]) |")
    lines += ["", "For zero observed false positives, the empirical bootstrap interval is degenerate at [0,0]. The exact binomial interval is used here to retain finite-sample uncertainty.", "", "## Threshold convention", "", result["threshold_estimator_note"], "Because the finite-sample estimators differ, describe the observed operating points separately and do not call them estimator-matched FPR comparisons.", ""]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"updated {SUMMARY}; wrote {OUT_JSON} and {OUT_MD}")
    print("paired deltas:", result["paired_macro_auroc_difference_point"])
    print("paired CIs:", result["paired_macro_auroc_difference_ci95_label_stratified_bootstrap"])
    print("zero-error FPR CIs:", {c: op[c]["fpr_ci95_clopper_pearson"] for c in CATS if op[c]["fp"] == 0})


if __name__ == "__main__":
    main()
