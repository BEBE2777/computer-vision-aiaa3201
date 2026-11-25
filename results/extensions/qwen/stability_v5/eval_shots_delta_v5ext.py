"""Paired shot-count comparison for v5-ext (e.g. 2-shot vs 1-shot seed S).

Same 696 test images in both runs -> per-image pairing. Bootstrap is
stratified by category x label (good/bad resampled separately within each
category), and the SAME resampled indices are applied to BOTH shot settings,
so the macro-AUROC delta and its 95% CI are paired at the image level.

Also reports per-class AUROC deltas and per-image fc latency change.
Nothing selected post hoc: all six categories reported.

Usage:
  eval_shots_delta_v5ext.py --base <1shot predictions> --treat <Nshot predictions> \
      --seed S --treat-shots 2 [--out file]
"""
import argparse, json, random, statistics, sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_v5 import auroc  # noqa: E402

CATS = ["candle", "capsules", "cashew", "chewinggum", "fryum", "pipe_fryum"]
BOOT = 10000


def load_scores(path):
    """-> {(cat, label, query_id): (fc_conf, latency_ms)}"""
    out = {}
    for r in json.load(open(path))["records"]:
        if r.get("status") != "ok":
            continue
        k = (r["category"], r["true_label_for_eval_ONLY"], r["query_id"])
        out[k] = (-r["fc_score"], r.get("latency_fc_ms"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="1-shot predictions json")
    ap.add_argument("--treat", required=True, help="N-shot predictions json")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--treat-shots", type=int, required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rng = random.Random(20261125 + a.treat_shots * 100 + a.seed)
    base = load_scores(a.base)
    treat = load_scores(a.treat)

    common = sorted(set(base) & set(treat),
                    key=lambda k: (CATS.index(k[0]), k[1], k[2]))
    only_base = sorted(set(base) - set(treat))
    only_treat = sorted(set(treat) - set(base))

    # per-category, per-label aligned lists (image-level pairing)
    cat_data = {c: {"good": ([], []), "bad": ([], [])} for c in CATS}
    lat_b, lat_t = [], []
    for k in common:
        c, lab, _ = k
        cat_data[c][lab][0].append(base[k][0])
        cat_data[c][lab][1].append(treat[k][0])
        if base[k][1] and treat[k][1]:
            lat_b.append(base[k][1]); lat_t.append(treat[k][1])

    base_groups, treat_groups = [], []
    base_pc, treat_pc, delta_pc = {}, {}, {}
    for c in CATS:
        gb, tb = cat_data[c]["good"]
        lb_, lt = cat_data[c]["bad"]
        sb = gb + lb_
        labs = [0] * len(gb) + [1] * len(lb_)
        st_ = tb + lt
        ab, at = auroc(sb, labs), auroc(st_, labs)
        base_groups.append((sb, labs)); treat_groups.append((st_, labs))
        base_pc[c] = round(ab, 4)
        treat_pc[c] = round(at, 4)
        delta_pc[c] = None if (ab is None or at is None) else round(at - ab, 4)

    m_base = statistics.mean(base_pc.values())
    m_treat = statistics.mean(treat_pc.values())

    # stratified bootstrap (category x label), shared indices for both settings
    deltas = []
    for _ in range(BOOT):
        mb, mt = [], []
        for c in CATS:
            gb, tb = cat_data[c]["good"]
            lb_, lt = cat_data[c]["bad"]
            ng, nb = len(gb), len(lb_)
            ig = [rng.randrange(ng) for _ in range(ng)]
            ib = [rng.randrange(nb) for _ in range(nb)]
            sb = [gb[i] for i in ig] + [lb_[i] for i in ib]
            st_ = [tb[i] for i in ig] + [lt[i] for i in ib]
            labs = [0] * ng + [1] * nb
            mb.append(auroc(sb, labs)); mt.append(auroc(st_, labs))
        if all(v is not None for v in mb + mt):
            deltas.append(statistics.mean(mt) - statistics.mean(mb))
    deltas.sort()
    ci = (round(deltas[int(0.025 * BOOT)], 4), round(deltas[int(0.975 * BOOT)], 4))

    lat_b.sort(); lat_t.sort()
    out = {
        "comparison": f"{a.treat_shots}shot_vs_1shot_seed{a.seed}",
        "n_common_images": len(common),
        "excluded_only_base": len(only_base),
        "excluded_only_treat": len(only_treat),
        "base_macro_auroc": round(m_base, 4),
        "treat_macro_auroc": round(m_treat, 4),
        "delta_macro_auroc": round(m_treat - m_base, 4),
        "delta_ci95": ci,
        "per_category_delta": delta_pc,
        "base_per_category": base_pc,
        "treat_per_category": treat_pc,
        "latency_fc_ms": {"base_mean": round(statistics.mean(lat_b), 1),
                          "treat_mean": round(statistics.mean(lat_t), 1),
                          "base_p95": round(lat_b[int(len(lat_b) * 0.95) - 1], 1),
                          "treat_p95": round(lat_t[int(len(lat_t) * 0.95) - 1], 1)},
    }
    print(json.dumps(out, indent=1, ensure_ascii=False))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
        print("WROTE", a.out)


if __name__ == "__main__":
    main()
