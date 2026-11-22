"""Offline evaluation for the v2 exploratory pilot. Reads predictions_v2_7b.json
(records carry true_label_for_eval_ONLY), computes Image AUROC for BOTH scores:
  - anomaly_score (legacy JSON readout)
  - fc_score / fc_score_mean (forced-choice sequence logprob readout)
If a score has zero variance, AUROC is reported as undefined (NOT 0.5).
No fusion, no method selection — both readouts are reported as-is.
"""
import json, sys

def auroc(scores, labels):
    """Rank-based AUROC (handles ties). labels: 1 = anomalous."""
    pairs = sorted(zip(scores, labels))
    n = len(pairs)
    pos = sum(labels)
    neg = n - pos
    if pos == 0 or neg == 0:
        return None, "no positives or negatives"
    vals = [s for s, _ in pairs]
    if len(set(vals)) == 1:
        return None, "no variance"
    # average rank with ties
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    rp = sum(r for r, (_, l) in zip(ranks, pairs) if l == 1)
    return (rp - pos * (pos + 1) / 2) / (pos * neg), ""


def evaluate(recs, key):
    scores = [r[key] for r in recs if r.get(key) is not None]
    labels = [1 if r["true_label_for_eval_ONLY"] == "bad" else 0
              for r in recs if r.get(key) is not None]
    auc, note = auroc(scores, labels)
    cats = sorted({r["category"] for r in recs})
    per_cat = {}
    for c in cats:
        rs = [r for r in recs if r["category"] == c and r.get(key) is not None]
        a, n2 = auroc([r[key] for r in rs],
                      [1 if r["true_label_for_eval_ONLY"] == "bad" else 0 for r in rs])
        per_cat[c] = None if a is None else round(a, 4)
        if a is None:
            per_cat[c] = f"undefined ({n2})"
    valid = [v for v in per_cat.values() if isinstance(v, float)]
    macro = round(sum(valid) / len(valid), 4) if valid and len(valid) == len(cats) else None
    return {"key": key, "auroc_macro6": macro, "per_category": per_cat,
            "note": note, "n_scored": len(scores),
            "unique_values": len(set(scores))}


def main(path):
    r = json.load(open(path))
    recs = r["records"]
    meta = r["meta"]
    n = len(recs)
    n_json_ok = sum(1 for x in recs if x["status"] == "ok")
    summary = {
        "protocol": meta["protocol"], "model_id": meta["model_id"],
        "manifest_n": n,
        "json_valid_rate": round(n_json_ok / n, 4),
        "mean_latency_gen_ms": round(sum(x["latency_gen_ms"] for x in recs) / n, 1),
        "mean_latency_fc_ms": round(sum(x["latency_fc_ms"] for x in recs) / n, 1),
        "p95_latency_gen_ms": round(sorted(x["latency_gen_ms"] for x in recs)[int(n * 0.95) - 1], 1),
        "mean_peak_gpu_gb": round(sum(x["peak_gpu_gb"] for x in recs) / n, 2),
        "readouts": [
            evaluate(recs, "anomaly_score"),
            evaluate(recs, "fc_score"),
            evaluate(recs, "fc_score_mean"),
        ],
        "decisions": {},
    }
    from collections import Counter
    summary["decisions"] = dict(Counter(str(x["decision"]) for x in recs))
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    out = path.replace("predictions", "summary").replace(".json", "_summary.json")
    json.dump(summary, open(out, "w"), indent=1, ensure_ascii=False)
    print("WROTE", out)
    return summary


if __name__ == "__main__":
    main(sys.argv[1])
