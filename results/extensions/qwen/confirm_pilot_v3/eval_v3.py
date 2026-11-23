"""Evaluation for the v3 confirmation pilot. Applies the PRESPECIFIED primary
direction fc_conf = -fc_score = sum_logP(C_OK) - sum_logP(C_NG) and reports the
unflipped readouts alongside. Success criterion (frozen in the protocol):
macro AUROC >= 0.70 AND >= 5/6 categories > 0.55.
"""
import json, sys
from collections import Counter


def auroc(scores, labels):
    pairs = sorted(zip(scores, labels))
    n = len(pairs)
    pos = sum(labels)
    neg = n - pos
    if pos == 0 or neg == 0:
        return None
    if len({s for s, _ in pairs}) == 1:
        return None
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    rp = sum(r for r, (_, l) in zip(ranks, pairs) if l == 1)
    return (rp - pos * (pos + 1) / 2) / (pos * neg)


def readout(recs, fn):
    cats = sorted({r["category"] for r in recs})
    per_cat = {}
    for c in cats:
        rs = [r for r in recs if r["category"] == c]
        per_cat[c] = auroc([fn(r) for r in rs],
                           [1 if r["true_label_for_eval_ONLY"] == "bad" else 0 for r in rs])
    vals = [per_cat[c] for c in cats]
    macro = None if any(v is None for v in vals) else sum(vals) / len(vals)
    return {"macro": None if macro is None else round(macro, 4),
            "per_category": {c: (None if v is None else round(v, 4)) for c, v in per_cat.items()},
            "n_above_055": sum(1 for v in vals if v is not None and v > 0.55),
            "n_defined": sum(1 for v in vals if v is not None)}


def main(path):
    r = json.load(open(path))
    recs = r["records"]
    n = len(recs)
    out = {
        "protocol": "qwen_confirmation_pilot_v3",
        "model_id": r["meta"]["model_id"],
        "n": n,
        "json_valid_rate": round(sum(x["status"] == "ok" for x in recs) / n, 4),
        "mean_latency_gen_ms": round(sum(x["latency_gen_ms"] for x in recs) / n, 1),
        "p95_latency_gen_ms": round(sorted(x["latency_gen_ms"] for x in recs)[int(n * 0.95) - 1], 1),
        "mean_latency_fc_ms": round(sum(x["latency_fc_ms"] for x in recs) / n, 1),
        "mean_peak_gpu_gb": round(sum(x["peak_gpu_gb"] for x in recs) / n, 2),
        "decisions": dict(Counter(str(x["decision"]) for x in recs)),
        "readouts": {
            # PRIMARY, prespecified
            "fc_conf": readout(recs, lambda x: -x["fc_score"]),
            # secondary, reported as measured
            "fc_score_unflipped": readout(recs, lambda x: x["fc_score"]),
            "fc_score_mean_unflipped": readout(recs, lambda x: x["fc_score_mean"]),
            "json_anomaly_score": readout(recs, lambda x: x["anomaly_score"] or 0.0),
        },
    }
    prim = out["readouts"]["fc_conf"]
    out["criterion"] = {
        "rule": "macro>=0.70 and >=5/6 categories>0.55",
        "macro": prim["macro"], "n_above_055": prim["n_above_055"],
        "passed": bool(prim["macro"] is not None and prim["macro"] >= 0.70
                       and prim["n_above_055"] >= 5),
    }
    print(json.dumps(out, indent=1, ensure_ascii=False))
    outp = path.replace("predictions", "summary").replace(".json", "_summary.json")
    json.dump(out, open(outp, "w"), indent=1, ensure_ascii=False)
    print("WROTE", outp)


if __name__ == "__main__":
    main(sys.argv[1])
