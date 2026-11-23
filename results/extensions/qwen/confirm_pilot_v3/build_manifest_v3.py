"""Build the frozen v3 confirmation manifest on the server BEFORE inference.

Fresh images only: excludes the v1 pilot 24 and the v2 pilot 120.
Rule (qwen_confirmation_pilot_v3.md section 2): rng=random.Random(20261123),
per category in fixed order, 10 good + 10 bad from the remaining pool.
"""
import os, json, random
from collections import Counter

DATA = "/data/disk2/home/qzheng610/qwen_work/data/VisA_pytorch/1cls"
REFS = "/data/disk2/home/qzheng610/qwen_work/refs_archive/1shot_seed0"
V2 = "/data/disk2/home/qzheng610/qwen_work/v2"
OUT = V2 + "/manifest_v3.json"

CATS = ["candle", "capsules", "cashew", "chewinggum", "fryum", "pipe_fryum"]
N = 10
SEED = 20261123

old = set(json.load(open(f"{V2}/old_pilot_images_all.json")))
m2 = json.load(open(f"{V2}/manifest_v2.json"))
v2_ids = {f"{e['category']}/{e['true_label_for_eval_ONLY']}/{e['query_id']}" for e in m2["entries"]}
excluded = old | v2_ids
print(f"excluded: v1={len(old)} v2={len(v2_ids)} union={len(excluded)}")

rng = random.Random(SEED)
entries = []
for cat in CATS:
    refs = json.load(open(f"{REFS}/{cat}_shot=1_seed=0.json"))["reference_files"]
    for lbl in ("good", "bad"):
        files = sorted(os.listdir(f"{DATA}/{cat}/test/{lbl}"))
        cand = [f for f in files if f"{cat}/{lbl}/{f}" not in excluded]
        assert len(cand) >= N, f"{cat}/{lbl}: only {len(cand)} left"
        for name in rng.sample(cand, N):
            entries.append({
                "category": cat,
                "true_label_for_eval_ONLY": lbl,
                "query_id": name,
                "query_path": f"{DATA}/{cat}/test/{lbl}/{name}",
                "reference_files": refs,
                "reference_paths": [f"{DATA}/{cat}/train/good/{r}" for r in refs],
                "subset_rule": f"Random({SEED}) sample 10 per label, v1+v2 excluded",
            })

man = {"protocol": "qwen_confirmation_pilot_v3", "frozen_before_inference": True,
       "sampling_seed": SEED, "shots": 1, "refs_seed": 0, "n_images": len(entries),
       "excluded_v1": sorted(old), "excluded_v2_n": len(v2_ids), "entries": entries}
json.dump(man, open(OUT, "w"), indent=1, ensure_ascii=False)
c = Counter((e["category"], e["true_label_for_eval_ONLY"]) for e in entries)
assert all(v == N for v in c.values()), c
ids = {f"{e['category']}/{e['true_label_for_eval_ONLY']}/{e['query_id']}" for e in entries}
print("manifest:", OUT, "n =", len(entries))
print("overlap v1:", len(ids & old), "overlap v2:", len(ids & v2_ids))
print("per-class balance ok")
