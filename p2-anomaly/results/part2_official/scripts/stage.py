"""Stage the deliverables for upload (server side)."""
import csv, json, os, shutil, subprocess

AGG = "/tmp/qzheng610_p2/agg"
BASE = "/tmp/qzheng610_p2/results_VisA/dinov2_vits14_448"
UP = "/tmp/qzheng610_p2/upload/p2_anomaly_part2_results"
LOGS = "/tmp/qzheng610_p2/logs"
SHOTS, SEEDS = [1, 2, 4, 8], [0, 1, 2]

if os.path.isdir("/tmp/qzheng610_p2/upload"):
    shutil.rmtree("/tmp/qzheng610_p2/upload")
for d in ("metrics", "refs", "logs", "args", "selected_heatmaps"):
    os.makedirs(os.path.join(UP, d), exist_ok=True)

cells = json.load(open(os.path.join(AGG, "summary_448.json")))["cells"]

# ---------------- metrics
rows = []
json_cells = {}
for sh in SHOTS:
    for sd in SEEDS:
        k = "%d-%d" % (sh, sd)
        if k not in cells:
            continue
        c = cells[k]
        run_dir = c["run_dir"]
        is_new = "calib20r2" in run_dir
        rows.append([sh, sd, "new" if is_new else "existing-seed0",
                     "%.2f" % (100 * c["mean_image_auroc"]),
                     "%.2f" % (100 * c["mean_pixel_auroc"]),
                     run_dir])
        json_cells[k] = {"shot": sh, "seed": sd,
                         "status": "new run (this session)" if is_new else "pre-existing seed-0 result",
                         "run_dir": run_dir,
                         "image_auroc_pct": round(100 * c["mean_image_auroc"], 4),
                         "pixel_auroc_pct": round(100 * c["mean_pixel_auroc"], 4),
                         "per_category": {cat: {"n": v["n"],
                                                "image_auroc_pct": round(100 * v["image_auroc"], 4),
                                                "pixel_auroc_pct": round(100 * v["pixel_auroc"], 4)}
                                          for cat, v in c["per_category"].items()}}
        shutil.copy(os.path.join(AGG, "cell_%dshot_seed%d.json" % (sh, sd)),
                    os.path.join(UP, "metrics", "eval448_%dshot_seed%d.json" % (sh, sd)))

with open(os.path.join(UP, "metrics", "summary_all_shots_seeds.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["shots", "seed", "status", "image_auroc", "pixel_auroc", "run_dir"])
    w.writerows(rows)

per_shot = {}
for sh in SHOTS:
    entry = {}
    for metric, key in (("image_auroc", "image_auroc_pct"), ("pixel_auroc", "pixel_auroc_pct")):
        vals = [json_cells["%d-%d" % (sh, s)][key] for s in SEEDS if "%d-%d" % (sh, s) in json_cells]
        m = sum(vals) / len(vals)
        sd_ = (sum((x - m) ** 2 for x in vals) / len(vals)) ** 0.5
        entry[metric] = {"per_seed": vals, "mean": round(m, 4), "std": round(sd_, 4), "n_seeds": len(vals)}
    per_shot["%d-shot" % sh] = entry

meta = {
    "dataset": "VisA (1cls split), six categories: candle, capsules, cashew, chewinggum, fryum, pipe_fryum",
    "method": "official AnomalyDINO fork, dinov2_vits14 (frozen), resolution 448, preprocess agnostic",
    "settings": {"calib_exclude": 20, "faiss_on_cpu": True, "warmup_iters": 3,
                 "eval_clf": True, "eval_segm": False, "k_neighbors": 1, "knn_metric": "L2_normalized"},
    "test_images_per_cell": 960,
    "reference_selection": "deterministic slice pool[seed*n:(seed+1)*n] after removing the first 20 sorted "
                           "normal train files; verified identical to the dumped refs/*.json",
    "part3": "NOT run. No cached-grid / fusion numbers are included here.",
    "eval_segm": "not used anywhere (official segmentation eval needs ~11 GB RAM and was avoided)",
    "pixel_auroc_source": "eval_maps.py / eval_perseed.py protocol: patch-distance map -> 448 bilinear -> "
                          "gaussian sigma 4; image score = mean of top 1% of the raw map",
    "note_on_location": "/tmp/slv176 is owned by slv176 and not writable by qzheng610, so the new runs were "
                        "written under /tmp/qzheng610_p2 with tag calib20r2. Existing seed-0 results were only read.",
    "cells": json_cells,
    "per_shot_mean_std": per_shot,
}
json.dump(meta, open(os.path.join(UP, "metrics", "summary_all_shots_seeds.json"), "w"), indent=2)

# ---------------- refs (nested by shot/seed)
n_refs = 0
for sh in SHOTS:
    for sd in SEEDS:
        k = "%d-%d" % (sh, sd)
        if k not in cells:
            continue
        src = os.path.join(cells[k]["run_dir"], "refs")
        dst = os.path.join(UP, "refs", "%dshot_seed%d" % (sh, sd))
        os.makedirs(dst, exist_ok=True)
        for fn in sorted(os.listdir(src)):
            if fn.endswith(".json") and "shot=%d_seed=%d" % (sh, sd) in fn:
                shutil.copy(os.path.join(src, fn), dst)
                n_refs += 1

# ---------------- args (one dir per run)
seen = {}
for sh in SHOTS:
    for sd in SEEDS:
        k = "%d-%d" % (sh, sd)
        if k not in cells:
            continue
        run_dir = cells[k]["run_dir"]
        tag = os.path.basename(run_dir).split("agnostic_")[-1]
        dst = os.path.join(UP, "args", os.path.basename(run_dir))
        if dst in seen:
            continue
        seen[dst] = 1
        os.makedirs(dst, exist_ok=True)
        for fn in ("args.yaml", "preprocess.yaml"):
            p = os.path.join(run_dir, fn)
            if os.path.isfile(p):
                shutil.copy(p, dst)

# ---------------- logs
for fn in ("part2_4shot_seed0.log", "part2_seed1.log", "part2_seed2.log", "driver.log"):
    p = os.path.join(LOGS, fn)
    if os.path.isfile(p):
        shutil.copy(p, os.path.join(UP, "logs"))

# ---------------- representative heatmaps
picks = [
    ("4-shot_preprocess=agnostic_calib20r2/chewinggum/anomaly_maps_examples_seed=0.png", "map_4shot_seed0_chewinggum.png"),
    ("4-shot_preprocess=agnostic_calib20r2/chewinggum/reference_samples.png", "refs_4shot_chewinggum.png"),
    ("8-shot_preprocess=agnostic_calib20r2/chewinggum/anomaly_maps_examples_seed=1.png", "map_8shot_seed1_chewinggum.png"),
    ("8-shot_preprocess=agnostic_calib20r2/chewinggum/anomaly_maps_examples_seed=2.png", "map_8shot_seed2_chewinggum.png"),
    ("2-shot_preprocess=agnostic_calib20r2/candle/anomaly_maps_examples_seed=1.png", "map_2shot_seed1_candle.png"),
    ("2-shot_preprocess=agnostic_calib20r2/capsules/anomaly_maps_examples_seed=2.png", "map_2shot_seed2_capsules.png"),
    ("4-shot_preprocess=agnostic_calib20r2/candle/examples/example_bad_0.png", "example_4shot_seed0_candle_bad0.png"),
]
for rel, name in picks:
    p = os.path.join(BASE, rel)
    if os.path.isfile(p):
        shutil.copy(p, os.path.join(UP, "selected_heatmaps", name))

print("refs copied:", n_refs)
for d in ("metrics", "refs", "logs", "args", "selected_heatmaps"):
    p = os.path.join(UP, d)
    n = sum(len(f) for _, _, f in os.walk(p))
    sz = subprocess.run(["du", "-sh", p], capture_output=True, text=True).stdout.split()[0]
    print("%-18s %3d files  %s" % (d, n, sz))
subprocess.run(["tar", "czf", "/tmp/qzheng610_p2/upload.tgz", "-C", "/tmp/qzheng610_p2/upload",
                "p2_anomaly_part2_results"], check=True)
print("tarball MB:", os.path.getsize("/tmp/qzheng610_p2/upload.tgz") // 1024 // 1024)
print(open(os.path.join(UP, "metrics", "summary_all_shots_seeds.csv")).read())
