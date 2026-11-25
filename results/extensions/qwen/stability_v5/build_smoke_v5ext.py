"""Build a 12-entry smoke manifest: 1 good + 1 bad per category, taken from
the frozen test manifest (deterministic: first such entry in manifest order).
Also verifies every entry's reference_files against the archived refs json
AND enforces reference count == shots before any smoke inference.

Usage: build_smoke_v5ext.py --shots S --seed N   (server)
Writes manifest_v5_s{N}_{S}sh_smoke.json next to the test manifest.
"""
import argparse, json, os, sys

BASE = "/data/disk2/home/qzheng610/qwen_work"
DATA = BASE + "/data/VisA_pytorch/1cls"
CATS = ["candle", "capsules", "cashew", "chewinggum", "fryum", "pipe_fryum"]

ap = argparse.ArgumentParser()
ap.add_argument("--shots", type=int, choices=[2, 4, 8], required=True)
ap.add_argument("--seed", type=int, required=True)
a = ap.parse_args()

tag = f"s{a.seed}_{a.shots}sh"
test = json.load(open(f"{BASE}/v2/manifest_v5_{tag}_test.json"))

entries, seen = [], set()
for cat in CATS:
    for want in ("good", "bad"):
        e = next((x for x in test["entries"]
                  if x["category"] == cat and x["true_label_for_eval_ONLY"] == want), None)
        assert e, f"no {want} entry for {cat}"
        assert (cat, e["query_id"]) not in seen, f"duplicate {cat}/{e['query_id']}"
        seen.add((cat, e["query_id"]))
        # verify refs against the archived json + count == shots
        refj = json.load(open(f"{BASE}/refs_archive/{a.shots}shot_seed{a.seed}/"
                              f"{cat}_shot={a.shots}_seed={a.seed}.json"))
        assert e["reference_files"] == refj["reference_files"], \
            f"refs mismatch vs archive for {cat}"
        assert len(e["reference_files"]) == a.shots, f"ref count != {a.shots} for {cat}"
        for p in e["reference_paths"] + [e["query_path"]]:
            assert os.path.exists(p), f"missing file {p}"
        entries.append(e)

assert len(entries) == 12
out = f"{BASE}/v2/manifest_v5_{tag}_smoke.json"
json.dump({"protocol": "qwen_fc_conf_stability_v5", "frozen_before_inference": True,
           "shots": a.shots, "refs_seed": a.seed, "split": "smoke",
           "n_images": 12, "entries": entries},
          open(out, "w"), indent=1, ensure_ascii=False)
print("WROTE", out, "entries:", [(e["category"], e["true_label_for_eval_ONLY"]) for e in entries])
