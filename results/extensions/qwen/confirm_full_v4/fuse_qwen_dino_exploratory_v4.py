"""Post-hoc exploratory Qwen + DINO score fusion on the v4 held-out records.

This analysis was initiated after inspecting the single-branch v4 metrics. It is
therefore exploratory, not a preregistered confirmatory result. It uses only
saved per-image predictions and normal calibration scores; no GPU inference.
All alpha values are reported, with no test-set weight selection.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "p2-anomaly" / "scripts"))
import score_fusion as SF  # noqa: E402

BASE = Path(__file__).resolve().parent
DINO_PATH = ROOT / "p2-anomaly/results/part3/scores/e1_dino448_nomask.json"
TEST_PATH = BASE / "predictions_v4_test.json"
CALIB_PATH = BASE / "predictions_v4_calib.json"
OUT_JSON = BASE / "fusion_qwen_dino_exploratory_v4.json"
OUT_MD = BASE / "fusion_qwen_dino_exploratory_v4.md"
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]  # alpha is Qwen's share


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    qtest = load_json(TEST_PATH)["records"]
    qcal = load_json(CALIB_PATH)["records"]
    dino = SF.load_dump(str(DINO_PATH))

    cells = {}
    for cat in SF.VISA_6:
        qrs = [r for r in qtest if r["category"] == cat]
        qcrs = [r for r in qcal if r["category"] == cat]
        assert qrs and len(qcrs) == 20, f"unexpected Qwen data count for {cat}"
        q_paths = [f"{cat}/test/{r['true_label_for_eval_ONLY']}/{r['query_id']}" for r in qrs]
        q_labels = [1 if r["true_label_for_eval_ONLY"] == "bad" else 0 for r in qrs]
        q_scores = [-float(r["fc_score"]) for r in qrs]  # frozen v4 direction
        q_calib = [-float(r["fc_score"]) for r in qcrs]

        d = dino["categories"][cat]
        d_index = {p: i for i, p in enumerate(d["paths"])}
        missing = sorted(set(q_paths) - set(d_index))
        assert not missing, f"DINO dump missing {cat} paths: {missing[:3]}"
        ix = [d_index[p] for p in q_paths]
        d_labels = d["labels"][ix].tolist()
        assert d_labels == q_labels, f"label mismatch between Qwen and DINO in {cat}"

        cells[cat] = {
            "paths": q_paths,
            "labels": SF.np.asarray(q_labels, dtype=int),
            "s1": q_scores,
            "s2": d["scores"][ix],
            "calib1": q_calib,
            "calib2": d["calib"],
        }

    rows, main_row, decision, per_cat, decisions = SF.fuse_cells(cells, ALPHAS, 0.5)
    n_total = sum(c["n"] for c in per_cat.values())
    max_endpoint_drift = 0.0
    for row in rows:
        if row["alpha"] in (0.0, 1.0):
            expected_arm = 2 if row["alpha"] == 0.0 else 1
            for cat, c in per_cat.items():
                s = c["z1"] if expected_arm == 1 else c["z2"]
                endpoint_auc = SF.auroc(s, c["labels"])
                max_endpoint_drift = max(max_endpoint_drift,
                    abs(endpoint_auc - row["per_category"][cat]))
    assert max_endpoint_drift < 1e-10, f"single-arm endpoint mismatch: {max_endpoint_drift}"

    normalisation = {}
    for cat, c in per_cat.items():
        normalisation[cat] = {
            "qwen_center": c["median"][1], "qwen_scale": c["sigma"][1],
            "qwen_scale_fallback": c["scale"][1],
            "dino_center": c["median"][2], "dino_scale": c["sigma"][2],
            "dino_scale_fallback": c["scale"][2],
            "n_test": c["n"], "n_calib": len(c["calib_z1"]),
        }

    payload = {
        "status": "post-hoc exploratory analysis; not preregistered",
        "protocol": "Qwen fc_conf + DINOv2 E1, 1-shot seed0, v4 696-image subset",
        "alpha_definition": "alpha * z_Qwen + (1-alpha) * z_DINO",
        "normalisation": "per-category median / (1.4826*MAD) on the 20 normal calibration images; score_fusion.py fallback chain",
        "alphas": rows,
        "alpha_0_5": main_row,
        "operating_points_by_alpha": decisions,
        "operating_point_alpha_0_5": decision,
        "normalisation_details": normalisation,
        "alignment": {"n_total": n_total, "max_endpoint_auc_drift": max_endpoint_drift},
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    lines = [
        "# Qwen + DINO score fusion — v4 post-hoc exploration", "",
        "> This analysis was initiated after inspecting the v4 single-branch results. It is exploratory, not a preregistered confirmatory result. No weights were selected from test labels; the full alpha sweep is shown.", "",
        "- Test images: 696, exactly aligned by category, label, and image ID.",
        "- Qwen score: frozen v4 `fc_conf = -fc_score`; DINO score: E1 DINOv2 @448, no mask, 1-shot/seed0.",
        "- Each branch is normalized independently per category using the 20 normal calibration scores: median and 1.4826×MAD, with the existing `score_fusion.py` fallback rule.",
        "- `alpha` is the Qwen share; `alpha=0` is DINO alone and `alpha=1` is Qwen alone.",
        "- Operating threshold for each fused score is the linear-interpolated 95th percentile of its fused calibration scores. These operating points are exploratory.", "",
        "## Results", "",
        "| Qwen weight α | Macro image AUROC | Pooled image AUROC | Recall | FPR |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row, dec in zip(rows, decisions):
        lines.append(f"| {row['alpha']:.2f} | {row['macro']:.4f} | {row['pooled']:.4f} | {dec['recall']:.3f} | {dec['fpr']:.3f} |")
    lines += ["", "## Per-category macro AUROC", "", "| Category | " + " | ".join(f"α={a:.2f}" for a in ALPHAS) + " |", "|---|" + "---:|" * len(ALPHAS)]
    for cat in SF.VISA_6:
        vals = " | ".join(f"{r['per_category'][cat]:.4f}" for r in rows)
        lines.append(f"| {cat} | {vals} |")
    lines += ["", f"Endpoint identity check: max AUROC drift from the corresponding single branch = {max_endpoint_drift:.2e}.",
              "At alpha=1, the AUROC matches the v4 Qwen endpoint, but its displayed operating point uses the fusion script's linear-interpolated calibration percentile; it therefore differs from the v4 frozen max-order-statistic threshold.",
              "", "Interpretation is limited to this fixed 696-image subset and 1-shot/seed0. Do not present the best alpha as a selected or independently validated model; the sweep is a sensitivity analysis after the single-branch results were already observed.", ""]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"aligned images: {n_total}; endpoint drift: {max_endpoint_drift:.3e}")
    for row, dec in zip(rows, decisions):
        print(f"alpha={row['alpha']:.2f} macro={row['macro']:.4f} pooled={row['pooled']:.4f} recall={dec['recall']:.4f} FPR={dec['fpr']:.4f}")
    print(f"wrote {OUT_JSON} and {OUT_MD}")


if __name__ == "__main__":
    main()
