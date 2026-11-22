#!/bin/bash
# v2 pilot: rebuild a LEAN VisA 1cls layout (symlinks only, NO masks — this pilot
# needs no pixel-level GT). Same CSV logic as the v1 build_data.sh.
set -e
LOG=/data/disk2/home/qzheng610/qwen_work/build_data_v2.log
exec > "$LOG" 2>&1
echo "START $(date)"
SRC=/data/disk1/shared/aiaa3201/project2_anomaly/VisA
DST=/data/disk2/home/qzheng610/qwen_work/data/VisA_pytorch/1cls
CSV=$SRC/split_csv/1cls.csv

mkdir -p "$DST"
/tmp/slv176/venv/bin/python - "$CSV" "$SRC" "$DST" <<'PY'
import csv, os, sys
csv_path, src, dst = sys.argv[1], sys.argv[2], sys.argv[3]
cats = ["candle","capsules","cashew","chewinggum","fryum","pipe_fryum"]
for c in cats:
    for sub in ("train/good","test/good","test/bad"):
        os.makedirs(os.path.join(dst,c,sub), exist_ok=True)
n_img=0
with open(csv_path, newline="") as f:
    for row in csv.DictReader(f):
        obj, split, label = row["object"], row["split"], row["label"]
        if obj not in cats: continue
        lbl = "good" if label=="normal" else "bad"
        image_src = os.path.join(src, row["image"])
        image_dst = os.path.join(dst, obj, split, lbl, os.path.basename(image_src))
        if not os.path.lexists(image_dst):
            os.symlink(os.path.abspath(image_src), image_dst)
        n_img += 1
print(f"images={n_img} (symlinks only, masks skipped)")
for c in cats:
    tr=len(os.listdir(os.path.join(dst,c,"train","good")))
    tg=len(os.listdir(os.path.join(dst,c,"test","good")))
    tb=len(os.listdir(os.path.join(dst,c,"test","bad")))
    print(f"{c:12s} train={tr} test_good={tg} test_bad={tb}")
PY
echo "BUILD_DONE $(date)"
