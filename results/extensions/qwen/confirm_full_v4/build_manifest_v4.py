"""Build frozen v4 manifests on the server BEFORE inference.

test  : ALL remaining 1cls test images after excluding v1+v2+v3 picks.
calib : per class, sorted(train/good)[:20] (same rule as p2-anomaly calib).
"""
import os, json
from collections import Counter

DATA = "/data/disk2/home/qzheng610/qwen_work/data/VisA_pytorch/1cls"
REFS = "/data/disk2/home/qzheng610/qwen_work/refs_archive/1shot_seed0"
V2 = "/data/disk2/home/qzheng610/qwen_work/v2"
CATS = ["candle", "capsules", "cashew", "chewinggum", "fryum", "pipe_fryum"]

old = set(json.load(open(f"{V2}/old_pilot_images_all.json")))
excluded = set(old)
for mf in ("manifest_v2.json", "manifest_v3.json"):
    m = json.load(open(f"{V2}/{mf}"))
    excluded |= {f"{e['category']}/{e['true_label_for_eval_ONLY']}/{e['query_id']}"
                 for e in m["entries"]}
print("excluded union:", len(excluded))


def refs_for(cat):
    r = json.load(open(f"{REFS}/{cat}_shot=1_seed=0.json"))["reference_files"]
    return r, [f"{DATA}/{cat}/train/good/{x}" for x in r]


# ---- test manifest ----
entries = []
for cat in CATS:
    refs, rpaths = refs_for(cat)
    for lbl in ("good", "bad"):
        files = sorted(os.listdir(f"{DATA}/{cat}/test/{lbl}"))
        cand = [f for f in files if f"{cat}/{lbl}/{f}" not in excluded]
        for name in cand:
            entries.append({
                "category": cat, "true_label_for_eval_ONLY": lbl, "query_id": name,
                "query_path": f"{DATA}/{cat}/test/{lbl}/{name}",
                "reference_files": refs, "reference_paths": rpaths,
                "subset_rule": "all remaining 1cls test after v1+v2+v3 exclusion",
            })
man = {"protocol": "qwen_fc_conf_heldout_v4", "frozen_before_inference": True,
       "shots": 1, "refs_seed": 0, "n_images": len(entries),
       "excluded_union_n": len(excluded), "entries": entries}
json.dump(man, open(f"{V2}/manifest_v4_test.json", "w"), indent=1, ensure_ascii=False)
c = Counter((e["category"], e["true_label_for_eval_ONLY"]) for e in entries)
print("test manifest:", len(entries), dict(sorted(c.items())))

# ---- calibration manifest ----
calib = []
for cat in CATS:
    refs, rpaths = refs_for(cat)
    names = sorted(os.listdir(f"{DATA}/{cat}/train/good"))[:20]
    assert len(names) == 20, (cat, len(names))
    for name in names:
        calib.append({
            "category": cat, "true_label_for_eval_ONLY": "good", "query_id": name,
            "query_path": f"{DATA}/{cat}/train/good/{name}",
            "reference_files": refs, "reference_paths": rpaths,
            "subset_rule": "sorted(train/good)[:20] calibration, same as p2-anomaly",
        })
man_c = {"protocol": "qwen_fc_conf_heldout_v4", "frozen_before_inference": True,
         "shots": 1, "refs_seed": 0, "n_images": len(calib), "entries": calib}
json.dump(man_c, open(f"{V2}/manifest_v4_calib.json", "w"), indent=1, ensure_ascii=False)
print("calib manifest:", len(calib))
overlap = {f"{e['category']}/{e['true_label_for_eval_ONLY']}/{e['query_id']}" for e in entries} & excluded
print("test/calib overlap with excluded:", len(overlap))
