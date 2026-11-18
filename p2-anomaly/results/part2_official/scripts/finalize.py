import csv, json, os, subprocess, collections

AGG = "/tmp/qzheng610_p2/agg"
SHOTS = [1, 2, 4, 8]
SEEDS = [0, 1, 2]

rows = list(csv.DictReader(open(os.path.join(AGG, "perseed_448.csv"))))
mean = {}
for r in rows:
    if r["category"] == "MEAN":
        mean[(int(r["shot"]), int(r["seed"]))] = (float(r["image_auroc"]), float(r["pixel_auroc"]))

# wide table: one row per shot
wide = os.path.join(AGG, "table_448_wide.csv")
with open(wide, "w", newline="") as f:
    w = csv.writer(f)
    hdr = ["shots"]
    for s in SEEDS:
        hdr += ["img_seed%d" % s, "px_seed%d" % s]
    hdr += ["img_mean", "img_std", "px_mean", "px_std"]
    w.writerow(hdr)
    for sh in SHOTS:
        imgs = [100 * mean[(sh, s)][0] for s in SEEDS]
        pxs = [100 * mean[(sh, s)][1] for s in SEEDS]
        m = lambda v: sum(v) / len(v)
        sd = lambda v: (sum((x - m(v)) ** 2 for x in v) / len(v)) ** 0.5
        inter = []
        for i, p in zip(imgs, pxs):
            inter += ["%.2f" % i, "%.2f" % p]
        w.writerow([sh] + inter
                   + ["%.2f" % m(imgs), "%.2f" % sd(imgs), "%.2f" % m(pxs), "%.2f" % sd(pxs)])
print(open(wide).read())

# sanity: images per category in each cell
missing = []
for k, v in json.load(open(os.path.join(AGG, "summary_448.json")))["cells"].items():
    n = {c: d["n"] for c, d in v["per_category"].items()}
    tot = sum(n.values())
    if tot != 960:
        missing.append((k, tot, n))
print("cells with !=960 test images:", missing if missing else "none")

# inventory of produced artefacts
base = "/tmp/qzheng610_p2/results_VisA/dinov2_vits14_448"
inv = collections.Counter()
new_files = []
for d in sorted(os.listdir(base)):
    if "calib20r2" not in d:
        continue
    p = os.path.join(base, d)
    for root, _, files in os.walk(p):
        for fn in files:
            rel = os.path.relpath(os.path.join(root, fn), base)
            if "/anomaly_maps/" in "/" + rel:
                inv["anomaly_maps/*.npy"] += 1
            elif rel.endswith(".png"):
                inv["png"] += 1
            else:
                new_files.append(rel)
print("\nfile counts:", dict(inv))
print("\nnon-map, non-png files (%d):" % len(new_files))
for f in sorted(new_files):
    print("  ", f)

subprocess.run(["tar", "czf", "/tmp/qzheng610_p2/p2_part2_calib20r2.tgz",
                "-C", "/tmp/qzheng610_p2", "results_VisA", "agg", "logs", "scripts"],
               check=True)
print("\ntarball:", os.path.getsize("/tmp/qzheng610_p2/p2_part2_calib20r2.tgz") // 1024 // 1024, "MB")
