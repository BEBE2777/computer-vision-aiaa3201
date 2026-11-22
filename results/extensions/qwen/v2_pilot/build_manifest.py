"""Build the frozen v2 manifest on the server BEFORE any inference.

Deterministic rule (see qwen_exploratory_pilot_v2.md section 4):
- exclude the 24 images of the v1 pilot (old_pilot_images_all.json, "cat/lbl/name")
- per category in fixed order, rng=random.Random(20261122):
    goods = rng.sample(sorted(test/good) minus old, 10)
    bads  = rng.sample(sorted(test/bad) minus old, 10)
- references: refs_archive/1shot_seed0/<cat>_shot=1_seed=0.json -> reference_files
Labels are stored ONLY in true_label_for_eval_ONLY (never sent to the model).
"""
import os, sys, json, random

DATA = "/data/disk2/home/qzheng610/qwen_work/data/VisA_pytorch/1cls"
REFS = "/data/disk2/home/qzheng610/qwen_work/refs_archive/1shot_seed0"
OLD = "/data/disk2/home/qzheng610/qwen_work/v2/old_pilot_images_all.json"
OUT = "/data/disk2/home/qzheng610/qwen_work/v2/manifest_v2.json"

CATS = ["candle", "capsules", "cashew", "chewinggum", "fryum", "pipe_fryum"]
N_PER_LABEL = 10
SEED = 20261122

old = set(json.load(open(OLD)))
rng = random.Random(SEED)
entries = []
for cat in CATS:
    refs = json.load(open(f"{REFS}/{cat}_shot=1_seed=0.json"))["reference_files"]
    for lbl in ("good", "bad"):
        files = sorted(os.listdir(f"{DATA}/{cat}/test/{lbl}"))
        cand = [f for f in files if f"{cat}/{lbl}/{f}" not in old]
        assert len(cand) >= N_PER_LABEL, f"{cat}/{lbl}: only {len(cand)} candidates"
        picks = rng.sample(cand, N_PER_LABEL)
        for name in picks:
            entries.append({
                "category": cat,
                "true_label_for_eval_ONLY": lbl,   # never sent to the model
                "query_id": name,
                "query_path": f"{DATA}/{cat}/test/{lbl}/{name}",
                "reference_files": refs,
                "reference_paths": [f"{DATA}/{cat}/train/good/{r}" for r in refs],
                "subset_rule": f"Random({SEED}) sample 10 per label, v1-pilot excluded",
            })

manifest = {
    "protocol": "qwen_exploratory_pilot_v2",
    "frozen_before_inference": True,
    "sampling_seed": SEED,
    "shots": 1, "refs_seed": 0,
    "n_images": len(entries),
    "old_pilot_excluded": sorted(old),
    "entries": entries,
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(manifest, open(OUT, "w"), indent=1, ensure_ascii=False)
# sanity: label balance
from collections import Counter
c = Counter((e["category"], e["true_label_for_eval_ONLY"]) for e in entries)
print("manifest written:", OUT, "n =", len(entries))
print("balance ok:", all(v == 10 for v in c.values()), dict(sorted(c.items())) if not all(v==10 for v in c.values()) else "6 cats x 10 good + 10 bad")
overlap = {e["category"]+"/"+e["true_label_for_eval_ONLY"]+"/"+e["query_id"] for e in entries} & old
print("overlap with old pilot:", len(overlap))
