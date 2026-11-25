"""Build frozen v5 manifests on the server BEFORE any v5 inference.

Reuses the exact query IDs of the v4 manifests (stability check, no
resampling) and swaps in archived reference images for the requested
refs seed and shot count: refs_archive/{shots}shot_seed{S}/<cat>_shot={shots}_seed={S}.json.

Usage: build_manifest_v5.py --seed 1|2 --split test|calib [--shots 1|2|4|8]
--shots 1 (default) keeps the original filenames manifest_v5_s{S}_{split}.json;
shots>1 writes manifest_v5_s{S}_{shots}sh_{split}.json.
"""
import argparse, json, os

BASE = "/data/disk2/home/qzheng610/qwen_work"
DATA = BASE + "/data/VisA_pytorch/1cls"

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--split", choices=["test", "calib"], required=True)
ap.add_argument("--shots", type=int, choices=[1, 2, 4, 8], default=1)
a = ap.parse_args()

src = json.load(open(f"{BASE}/v2/manifest_v4_{a.split}.json"))
assert src["n_images"] == len(src["entries"])
src_ids = {(e["category"], e["true_label_for_eval_ONLY"], e["query_id"]) for e in src["entries"]}

entries = []
ref_summary = {}
for e in src["entries"]:
    cat = e["category"]
    refj = json.load(open(f"{BASE}/refs_archive/{a.shots}shot_seed{a.seed}/{cat}_shot={a.shots}_seed={a.seed}.json"))
    refs = refj["reference_files"]
    assert refs, f"empty refs for {cat}"
    assert len(refs) == a.shots, f"{cat}: got {len(refs)} refs, expected {a.shots}"
    entries.append({
        "category": cat,
        "true_label_for_eval_ONLY": e["true_label_for_eval_ONLY"],
        "query_id": e["query_id"],
        "query_path": e["query_path"],
        "reference_files": refs,
        "reference_paths": [f"{DATA}/{cat}/train/good/{r}" for r in refs],
        "subset_rule": f"query IDs identical to v4 {a.split} manifest; refs = archived {a.shots}shot_seed{a.seed}",
    })
    ref_summary[cat] = refs

new_ids = {(e["category"], e["true_label_for_eval_ONLY"], e["query_id"]) for e in entries}
assert new_ids == src_ids, "query ID set changed!"

# verify paths exist
missing = [p for e in entries for p in e["reference_paths"] + [e["query_path"]] if not os.path.exists(p)]
assert not missing, f"missing files: {missing[:3]}"

# compare refs vs seed0 (informational)
diff_cats = []
for cat, refs in ref_summary.items():
    s0 = json.load(open(f"{BASE}/refs_archive/1shot_seed0/{cat}_shot=1_seed=0.json"))["reference_files"]
    if refs != s0:
        diff_cats.append(cat)
print(f"refs differ from seed0 in {len(diff_cats)}/6 cats: {diff_cats}")

manifest = {
    "protocol": "qwen_fc_conf_stability_v5",
    "frozen_before_inference": True,
    "shots": a.shots, "refs_seed": a.seed,
    "split": a.split,
    "source_manifest": f"manifest_v4_{a.split}.json (query IDs identical)",
    "n_images": len(entries),
    "entries": entries,
}
sh = "" if a.shots == 1 else f"{a.shots}sh_"
out = f"{BASE}/v2/manifest_v5_s{a.seed}_{sh}{a.split}.json"
json.dump(manifest, open(out, "w"), indent=1, ensure_ascii=False)
import hashlib
print("WROTE", out, "n =", len(entries),
      "sha256", hashlib.sha256(open(out, "rb").read()).hexdigest()[:16])
