"""Matched DINO / CLIP comparison on exactly the v2 pilot's 120 images.

Source: committed per-image dumps from the E1 toolchain (1-shot, seed 0, 448,
nomask) whose reference sampling is the same pool[seed*n:(seed+1)*n] rule that
was verified byte-identical to the archived refs used by the Qwen manifest.
No model forward pass is needed; scores are read offline.

Outputs matched_baselines_v2.json with per-image scores for all three readouts
plus per-category / macro AUROC on the same 120 images.
"""
import json, os, sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))  # .../aiaa3201_repo
BASE = os.path.join(REPO, "p2-anomaly", "results", "part3", "scores")
QWEN = os.path.join("results", "extensions", "qwen", "v2_pilot")


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


def load_dump(name):
    d = json.load(open(os.path.join(BASE, name)))
    out = {}
    for cat, v in d["categories"].items():
        for p, l, s in zip(v["paths"], v["labels"], v["scores"]):
            key = (cat, "bad" if l == 1 else "good", os.path.basename(p))
            out[key] = float(s)
    return out, d


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(QWEN, "manifest_v2.json"))
    ap.add_argument("--predictions", default=os.path.join(QWEN, "predictions_v2_7b.json"))
    ap.add_argument("--out", default=os.path.join(QWEN, "matched_baselines_v2.json"))
    a = ap.parse_args()
    man = json.load(open(a.manifest))
    need = [(e["category"], e["true_label_for_eval_ONLY"], e["query_id"]) for e in man["entries"]]
    dino, dino_meta = load_dump("e1_dino448_nomask.json")
    clip, clip_meta = load_dump("e1_clip448_L9_nomask.json")
    qwen = json.load(open(a.predictions))["records"]
    qmap = {(r["category"], r["true_label_for_eval_ONLY"], r["query_id"]): r for r in qwen}

    rows = []
    for k in need:
        q = qmap[k]
        rows.append({
            "category": k[0], "label": k[1], "query_id": k[2],
            "dino_score": dino[k], "clip_score": clip[k],
            "qwen_json_score": q["anomaly_score"],
            "qwen_fc_score": q["fc_score"],
            "qwen_fc_score_signflip": -q["fc_score"],
        })

    cats = sorted({r["category"] for r in rows})
    res = {"n": len(rows), "cats": cats, "per_category": {}, "macro": {}}
    keys = ["dino_score", "clip_score", "qwen_json_score", "qwen_fc_score", "qwen_fc_score_signflip"]
    for cat in cats:
        rs = [r for r in rows if r["category"] == cat]
        labels = [1 if r["label"] == "bad" else 0 for r in rs]
        res["per_category"][cat] = {k: auroc([r[k] for r in rs], labels) for k in keys}
    for k in keys:
        vals = [res["per_category"][c][k] for c in cats]
        res["macro"][k] = (None if any(v is None for v in vals)
                           else round(sum(vals) / len(vals), 4))
    labels_all = [1 if r["label"] == "bad" else 0 for r in rows]
    res["pooled_120"] = {k: auroc([r[k] for r in rows], labels_all) for k in keys}
    res["sources"] = {"dino": "p2-anomaly/results/part3/scores/e1_dino448_nomask.json",
                      "clip": "p2-anomaly/results/part3/scores/e1_clip448_L9_nomask.json",
                      "qwen": "results/extensions/qwen/v2_pilot/predictions_v2_7b.json",
                      "matched": "1-shot seed0, 448, nomask, archived refs; same 120 images"}
    out = a.out
    json.dump({"summary": res, "rows": rows}, open(out, "w"), indent=1, ensure_ascii=False)
    for k in keys:
        pc = " ".join(f"{c}:{('undef' if res['per_category'][c][k] is None else format(res['per_category'][c][k], '.3f'))}" for c in cats)
        print(f"{k:26s} macro={res['macro'][k]} | {pc}")
    print("WROTE", out)


if __name__ == "__main__":
    main()
