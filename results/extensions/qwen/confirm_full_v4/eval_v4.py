"""Evaluation for the v4 held-out confirmation (protocol qwen_fc_conf_heldout_v4.md).

Frozen rules applied here:
- primary score fc_conf = -fc_score = sum_logP(C_OK) - sum_logP(C_NG)
- per-class threshold thr_c = 95th percentile of fc_conf on the class's 20
  held-out calibration normal images (predictions_v4_calib.json)
- DINO/CLIP use their OWN stored calib-derived thresholds from the E1 dumps
- stratified bootstrap (10k) 95% CIs for macro AUROC, FPR, recall
"""
import json, os, random, statistics, sys

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


def boot_ci(groups, stat_fn, rng):
    """groups: list of (scores, labels) tuples per class. stat_fn -> dict of stats."""
    stats = {k: [] for k in stat_fn(*groups)}
    for _ in range(BOOT):
        res = []
        for sc, lb in groups:
            idx = [rng.randrange(len(sc)) for _ in range(len(sc))]
            res.append(([sc[i] for i in idx], [lb[i] for i in idx]))
        for k, v in stat_fn(*res).items():
            stats[k].append(v)
    return {k: (round(sorted(v)[int(0.025 * BOOT)], 4),
                round(sorted(v)[int(0.975 * BOOT)], 4)) for k, v in stats.items()}


def main(pred_test_path, pred_calib_path):
    rng = random.Random(20261124)
    pt = json.load(open(pred_test_path))["records"]
    pc = json.load(open(pred_calib_path))["records"]

    # frozen thresholds on fc_conf
    thr = {}
    for c in CATS:
        vals = sorted(-x["fc_score"] for x in pc if x["category"] == c)
        thr[c] = vals[int(len(vals) * 0.95)]

    groups = []  # (fc_conf scores, labels) per class
    per = {}
    for c in CATS:
        rs = [x for x in pt if x["category"] == c]
        sc = [-x["fc_score"] for x in rs]
        lb = [1 if x["true_label_for_eval_ONLY"] == "bad" else 0 for x in rs]
        groups.append((sc, lb))
        nrm = [s for s, l in zip(sc, lb) if l == 0]
        ano = [s for s, l in zip(sc, lb) if l == 1]
        per[c] = {
            "auroc": auroc(sc, lb),
            "thr_p95_calib": round(thr[c], 4),
            "fpr": round(sum(1 for s in nrm if s >= thr[c]) / max(len(nrm), 1), 4),
            "recall": round(sum(1 for s in ano if s >= thr[c]) / max(len(ano), 1), 4),
            "n_normal": len(nrm), "n_anom": len(ano),
        }

    def stats(*gs):
        aucs = [auroc(sc, lb) for sc, lb in gs]
        macro = statistics.mean(aucs)
        out = {"macro_auroc": macro}
        for c, (sc, lb) in zip(CATS, gs):
            nrm = [s for s, l in zip(sc, lb) if l == 0]
            ano = [s for s, l in zip(sc, lb) if l == 1]
            t = thr[c]
            out[f"fpr_{c}"] = sum(1 for s in nrm if s >= t) / max(len(nrm), 1)
            out[f"rec_{c}"] = sum(1 for s in ano if s >= t) / max(len(ano), 1)
        return out

    ci = boot_ci(groups, stats, rng)

    dino = load_dump("e1_dino448_nomask.json")
    clip = load_dump("e1_clip448_L9_nomask.json")

    def baseline_stats(dump, name):
        g, perb, macro_parts = [], {}, []
        for c in CATS:
            rs = [x for x in pt if x["category"] == c]
            sc = [dump[(c, x["true_label_for_eval_ONLY"], x["query_id"])] for x in rs]
            lb = [1 if x["true_label_for_eval_ONLY"] == "bad" else 0 for x in rs]
            g.append((sc, lb))
            t = dump[("__thr__", c)]
            nrm = [s for s, l in zip(sc, lb) if l == 0]
            ano = [s for s, l in zip(sc, lb) if l == 1]
            perb[c] = {"auroc": auroc(sc, lb),
                       "thr_stored": round(t, 4),
                       "fpr": round(sum(1 for s in nrm if s >= t) / max(len(nrm), 1), 4),
                       "recall": round(sum(1 for s in ano if s >= t) / max(len(ano), 1), 4)}
            macro_parts.append(perb[c]["auroc"])
        aucs = [auroc(sc, lb) for sc, lb in g]
        ci_b = boot_ci(g, lambda *gs: {"macro_auroc": statistics.mean(
            [auroc(s, l) for s, l in gs])}, rng)
        return {"name": name, "macro_auroc": round(statistics.mean(aucs), 4),
                "macro_ci95": ci_b["macro_auroc"], "per_category": perb}

    lat_gen = [x["latency_gen_ms"] for x in pt if x.get("latency_gen_ms")]
    lat_fc = [x["latency_fc_ms"] for x in pt]
    n = len(pt)
    out = {
        "protocol": "qwen_fc_conf_heldout_v4",
        "n_test": n,
        "json_valid_rate": round(sum(x["status"] == "ok" for x in pt) / n, 4),
        "mean_latency_gen_ms": round(statistics.mean(lat_gen), 1) if lat_gen else None,
        "p95_latency_gen_ms": round(sorted(lat_gen)[int(n * 0.95) - 1], 1) if lat_gen else None,
        "mean_latency_fc_ms": round(statistics.mean(lat_fc), 1),
        "mean_peak_gpu_gb": round(statistics.mean(x["peak_gpu_gb"] for x in pt), 2),
        "qwen_fc_conf": {
            "thresholds": {c: round(thr[c], 4) for c in CATS},
            "per_category": per,
            "macro_auroc": round(statistics.mean(
                [per[c]["auroc"] for c in CATS]), 4),
            "macro_ci95": ci["macro_auroc"],
            "fpr_ci95": {c: ci[f"fpr_{c}"] for c in CATS},
            "recall_ci95": {c: ci[f"rec_{c}"] for c in CATS},
        },
        "dino": baseline_stats(dino, "DINOv2@448 (1-shot seed0, own calib thr)"),
        "clip": baseline_stats(clip, "CLIP L9@448 (1-shot seed0, own calib thr)"),
    }
    print(json.dumps(out, indent=1, ensure_ascii=False))
    outp = pred_test_path.replace("predictions", "summary").replace(".json", "_summary.json")
    json.dump(out, open(outp, "w"), indent=1, ensure_ascii=False)
    print("WROTE", outp)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
