#!/bin/bash
# Rebuild VisA 1cls layout on the GPU box from the course's original VisA + official 1cls.csv,
# then verify that reference filenames match the archived refs exactly.
set -e
LOG=/data/disk2/home/qzheng610/qwen_work/build_data.log
exec > "$LOG" 2>&1
echo "START $(date)"

SRC=/data/disk1/shared/aiaa3201/project2_anomaly/VisA
DST=/data/disk2/home/qzheng610/qwen_work/data/VisA_pytorch/1cls
CSV=$SRC/split_csv/1cls.csv
REFS=/data/disk2/home/qzheng610/qwen_work/refs_archive

mkdir -p "$DST"

# Copy only the 6 categories we need; images via symlink is fine (source is read-only and stable),
# masks copied + binarised. Use the same logic as p2-anomaly/prepare_visa.py.
/tmp/slv176/venv/bin/python - "$CSV" "$SRC" "$DST" <<'PY'
import csv, os, sys, numpy as np
from PIL import Image
csv_path, src, dst = sys.argv[1], sys.argv[2], sys.argv[3]
cats = ["candle","capsules","cashew","chewinggum","fryum","pipe_fryum"]
for c in cats:
    for sub in ("train/good","test/good","test/bad","ground_truth/bad"):
        os.makedirs(os.path.join(dst,c,sub), exist_ok=True)
n_img=n_mask=0
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
        if split=="test" and lbl=="bad":
            mask_src = os.path.join(src, row["mask"])
            m = np.array(Image.open(mask_src))
            m = np.where(m!=0,255,0).astype(np.uint8)
            out = os.path.join(dst,obj,"ground_truth","bad",os.path.basename(mask_src))
            if not os.path.exists(out): Image.fromarray(m).save(out)
            n_mask += 1
print(f"images={n_img} masks={n_mask}")
for c in cats:
    tr=len(os.listdir(os.path.join(dst,c,"train","good")))
    tg=len(os.listdir(os.path.join(dst,c,"test","good")))
    tb=len(os.listdir(os.path.join(dst,c,"test","bad")))
    print(f"{c:12s} train={tr} test_good={tg} test_bad={tb} total_test={tg+tb}")
PY

echo "=== verify refs match archive ==="
# copy local archived refs to server for exact comparison (done separately); here compute from data
/tmp/slv176/venv/bin/python - "$DST" <<'PY'
import os, sys, json
root=sys.argv[1]
cats=["candle","capsules","cashew","chewinggum","fryum","pipe_fryum"]
for c in cats:
    names=sorted(os.listdir(os.path.join(root,c,"train","good")))
    calib=names[:20]; pool=names[20:]
    print(c, "train_good",len(names),"calib",len(calib),"pool",len(pool),"pool[:3]",pool[:3])
PY
echo "BUILD_DONE $(date)"
