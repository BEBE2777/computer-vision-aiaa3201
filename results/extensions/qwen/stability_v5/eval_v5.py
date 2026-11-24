"""Evaluation for the v5 reference-set stability pilot (qwen_stability_v5.md).

Per seed:
- fc_conf = -fc_score (frozen direction)
- per-class threshold = calib p95 of fc_conf (frozen rule, secondary)
- per-class AUROC + macro; stratified bootstrap CIs
- PAIRED comparison vs DINO/CLIP on the identical images (same shot/seed
  dumps): bootstrap resamples image indices per class ONCE and applies them
  to both methods -> CI of paired macro-AUROC difference
- failures / missing IDs reported, never dropped silently
"""
import argparse, json, os, random, statistics

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))  # .../aiaa3201_repo
QV = os.path.join(REPO, "results", "extensions", "qwen")
SCORES = os.path.join(REPO, "p2-anomaly", "results", "part3", "scores")
CATS = ["candle", "capsules", "cashew", "chewinggum", "fryum", "pipe_fryum"]
BOOT = 10000


def auroc(scores, labels):
    pairs = sorted(zip(scores, labels))
    n = len(pairs)
    pos = sum(labels)
    neg = n - pos
    if pos == 0 or neg == 0 or len({s for s, _ in pairs}) == 1:
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


def load_dump(name):
    d = json.load(open(os.path.join(SCORES, name)))
    out = {}
    for cat, v in d["categories"].items():
        for p, l, s in zip(v["paths"], v["labels"], v["scores"]):
            out[(cat, "bad" if l == 1 else "good", os.path.basename(p))] = float(s)
        out.setdefault(("__thr__", cat), v["thr"])
    return out


def macro_auc(groups):
    aucs = [auroc(sc, lb) for sc, lb in groups]
    if any(a is None for a in aucs):
        return None
    return statistics.mean(aucs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rng = random.Random(20261125)
    pt = json.load(open(a.test))["records"]
    pc = json.load(open(a.calib))["records"]

    fails = [f"{x['category']}/{x['query_id']}" for x in pt if x.get("status") != "ok"]
    thr = {}
    for c in CATS:
        vals = sorted(-x["fc_score"] for x in pc if x["category"] == c)
        thr[c] = vals[int(len(vals) * 0.95)]

    # build aligned per-class lists for qwen and both baselines
    dino = load_dump(f"e1_dino448_nomask_s1_seed{a.seed}.json")
    clip = load_dump(f"e1_clip448_L9_nomask_s1_seed{a.seed}.json")

    q_groups, d_groups, c_groups = [], [], []
    per, missing_ids = {}, []
    for c in CATS:
        rs = [x for x in pt if x["category"] == c and x.get("status") == "ok"]
        sc_q, sc_d, sc_c, lb = [], [], [], []
        for x in rs:
            key = (c, x["true_label_for_eval_ONLY"], x["query_id"])
            if key not in dino or key not in clip:
                missing_ids.append(f"{c}/{x['query_id']}")
                continue
            sc_q.append(-x["fc_score"])
            sc_d.append(dino[key])
            sc_c.append(clip[key])
            lb.append(1 if x["true_label_for_eval_ONLY"] == "bad" else 0)
        q_groups.append((sc_q, lb)); d_groups.append((sc_d, lb)); c_groups.append((sc_c, lb))
        nrm = [s for s, l in zip(sc_q, lb) if l == 0]
        ano = [s for s, l in zip(sc_q, lb) if l == 1]
        per[c] = {
            "auroc": auroc(sc_q, lb),
            "thr_p95_calib": round(thr[c], 4),
            "fpr": round(sum(1 for s in nrm if s >= thr[c]) / max(len(nrm), 1), 4),
            "recall": round(sum(1 for s in ano if s >= thr[c]) / max(len(ano), 1), 4),
            "n_normal": len(nrm), "n_anom": len(ano),
        }

    macro_q = macro_auc(q_groups)

    # stratified bootstrap CI for qwen macro
    idx_lo = []
    for _ in range(BOOT):
        res = []
        for sc, lb in q_groups:
            idx = [rng.randrange(len(sc)) for _ in range(len(sc))]
            res.append(([sc[i] for i in idx], [lb[i] for i in idx]))
        idx_lo.append(macro_auc(res))
    ci_q = (round(sorted(idx_lo)[int(0.025 * BOOT)], 4),
            round(sorted(idx_lo)[int(0.975 * BOOT)], 4))

    # PAIRED bootstrap: same resampled indices applied to qwen and baseline
    def paired_ci(b_groups):
        deltas = []
        for _ in range(BOOT):
            mq, mb = [], []
            for (sq, lb), (sb, _) in zip(q_groups, b_groups):
                idx = [rng.randrange(len(sq)) for _ in range(len(sq))]
                sqb = [sq[i] for i in idx]; sbb = [sb[i] for i in idx]
                lbb = [lb[i] for i in idx]
                mq.append(auroc(sqb, lbb)); mb.append(auroc(sbb, lbb))
            if all(v is not None for v in mq + mb):
                deltas.append(statistics.mean(mq) - statistics.mean(mb))
        deltas.sort()
        return {"delta_mean": round(statistics.mean(deltas), 4),
                "delta_ci95": (round(deltas[int(0.025 * BOOT)], 4),
                               round(deltas[int(0.975 * BOOT)], 4))}

    paired = {"qwen_minus_dino": paired_ci(d_groups),
              "qwen_minus_clip": paired_ci(c_groups)}

    # baselines standalone
    def base_stats(g):
        aucmap = {c: auroc(sc, lb) for c, (sc, lb) in zip(CATS, g)}
        return {"per_category_auroc": {c: round(v, 4) for c, v in aucmap.items()},
                "macro_auroc": round(statistics.mean(aucmap.values()), 4)}
    d_stats, c_stats = base_stats(d_groups), base_stats(c_groups)

    lat = [x["latency_fc_ms"] for x in pt if x.get("latency_fc_ms")]
    out = {
        "protocol": "qwen_fc_conf_stability_v5",
        "refs_seed": a.seed, "n_test": len(pt),
        "n_fail": len(fails), "fail_ids": fails,
        "missing_in_baseline_dumps": missing_ids,
        "latency_fc_ms": {"mean": round(statistics.mean(lat), 1),
                          "p95": round(sorted(lat)[int(len(lat) * 0.95) - 1], 1)},
        "mean_peak_gpu_gb": round(statistics.mean(x["peak_gpu_gb"] for x in pt if x.get("peak_gpu_gb")), 2),
        "qwen_fc_conf": {
            "per_category": per,
            "macro_auroc": round(macro_q, 4), "macro_ci95": ci_q,
        },
        "dino": d_stats, "clip": c_stats,
        "paired": paired,
    }
    print(json.dumps(out, indent=1, ensure_ascii=False))
    outp = a.out or a.test.replace("predictions", "summary").replace(".json", "_summary.json")
    json.dump(out, open(outp, "w"), indent=1, ensure_ascii=False)
    print("WROTE", outp)


if __name__ == "__main__":
    main()
