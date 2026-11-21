"""Compute Image AUROC from saved Qwen predictions (offline, no model calls).

Aggregation follows the project protocol: per-category AUROC -> 6-class macro mean
per seed -> mean/std over seeds. Also reports AUPRC and failure coverage.
"""
import os, sys, json, argparse
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

CATS = ["candle", "capsules", "cashew", "chewinggum", "fryum", "pipe_fryum"]


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    if len(np.unique(y)) < 2 or not np.all(np.isfinite(s)):
        return None
    return float(roc_auc_score(y, s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expected-per-cell", type=int, default=960)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.predictions, encoding="utf-8") if l.strip()]
    # group by (shot, seed, category)
    groups = {}
    for r in recs:
        if r.get("status") != "ok" or r.get("anomaly_score") is None:
            continue
        key = (r["shot"], r["reference_seed"], r["category"])
        groups.setdefault(key, []).append(r)

    per_cell = {}
    for (shot, seed, cat), rs in groups.items():
        y = [1 if r["true_label_for_eval_ONLY"] == "bad" else 0 for r in rs]
        s = [r["anomaly_score"] for r in rs]
        per_cell[f"{shot}shot_seed{seed}_{cat}"] = {
            "n": len(rs), "image_auroc": auroc(y, s),
            "image_auprc": float(average_precision_score(y, s)) if len(np.unique(y)) > 1 else None,
            "score_range": [float(min(s)), float(max(s))],
            "unique_scores": len(set(s)),
        }

    # macro over categories per (shot,seed), then mean/std over seeds
    shots = sorted({k[0] for k in groups})
    seeds = sorted({k[1] for k in groups})
    by_shot_seed = {}
    for shot in shots:
        for seed in seeds:
            vals = [per_cell.get(f"{shot}shot_seed{seed}_{c}", {}).get("image_auroc")
                    for c in CATS]
            vals = [v for v in vals if v is not None]
            by_shot_seed[f"{shot}shot_seed{seed}"] = {
                "macro_image_auroc": float(np.mean(vals)) if vals else None,
                "n_categories": len(vals),
            }
    summary = {}
    for shot in shots:
        vals = [by_shot_seed[f"{shot}shot_seed{s}"]["macro_image_auroc"]
                for s in seeds if by_shot_seed[f"{shot}shot_seed{s}"]["macro_image_auroc"] is not None]
        summary[f"{shot}shot"] = {
            "mean": float(np.mean(vals)) if vals else None,
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "n_seeds": len(vals),
        }

    out = {
        "source": args.predictions, "n_predictions": len(recs),
        "n_scored": sum(1 for r in recs if r.get("status") == "ok" and r.get("anomaly_score") is not None),
        "n_failed": sum(1 for r in recs if r.get("status") != "ok" or r.get("anomaly_score") is None),
        "per_cell": per_cell, "per_shot_seed": by_shot_seed, "per_shot_summary": summary,
    }
    json.dump(out, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(json.dumps({"n_predictions": out["n_predictions"], "n_scored": out["n_scored"],
                      "per_shot_summary": summary}, indent=2))


if __name__ == "__main__":
    main()
