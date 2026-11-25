"""Acceptance check for one v5-ext run cell. Exits 0 only if:
- every manifest entry has a matching record (category/query_id set equal)
- records are UNIQUE (duplicate category/query_id records fail: dict-merge
  would otherwise hide them)
- per-record true_label matches the manifest entry
- reference_ids match the manifest entry (archived refs respected)
- metadata prompt/candidate sha256 equal the FROZEN values (v4/v5 lineage)
- 0 records with status != ok; all ok scores finite

Usage: verify_v5ext.py --manifest M --out P
"""
import argparse, json, math, sys

# frozen since v4 (prompt_v1.txt; C_OK / C_NG literals in run_v5.py)
FROZEN_PROMPT_SHA = "596118ac764bc609c0e2885fdc09964898e26a171a20f6d4e4ac49eeb64a87af"
FROZEN_C_OK_SHA = "54eeda5fb6b99bda920db0b89f794d40182c60a96ccb30ca4d9697f9a34b6993"
FROZEN_C_NG_SHA = "7b29b30e232e129abd39ab711845c378835e43d069c01bcbaa6e0315b2a8e533"

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

man = json.load(open(a.manifest))["entries"]
d = json.load(open(a.out))
recs = d.get("records", [])
errs = []

def rec_key(r):
    return f"{r['category']}/{r['true_label_for_eval_ONLY']}/{r['query_id']}"

by_key = {}
dups = set()
for r in recs:
    k = rec_key(r)
    if k in by_key:
        dups.add(k)
    else:
        by_key[k] = r
if dups:
    errs.append(f"DUPLICATE records: {sorted(dups)[:5]}")

want = {f"{e['category']}/{e['true_label_for_eval_ONLY']}/{e['query_id']}": e for e in man}
missing = [k for k in want if k not in by_key]
extra = [k for k in by_key if k not in want]
if missing:
    errs.append(f"missing {len(missing)}: {missing[:5]}")
if extra:
    errs.append(f"extra {len(extra)}: {extra[:5]}")
if len(recs) != len(man):
    errs.append(f"record count {len(recs)} != manifest {len(man)}")

label_bad = []
ref_bad = []
score_bad = []
for e in man:
    k = f"{e['category']}/{e['true_label_for_eval_ONLY']}/{e['query_id']}"
    r = by_key.get(k)
    if not r:
        continue
    if r.get("true_label_for_eval_ONLY") != e["true_label_for_eval_ONLY"]:
        label_bad.append(k)
    if r.get("reference_ids") != e["reference_files"]:
        ref_bad.append(k)
    if r.get("status") != "ok":
        continue
    for f in ("fc_score", "sum_logP_OK", "sum_logP_NG"):
        v = r.get(f)
        if v is None or not isinstance(v, (int, float)) or not math.isfinite(v):
            score_bad.append(f"{k}:{f}")
            break
if label_bad:
    errs.append(f"label mismatch: {label_bad[:5]}")
if ref_bad:
    errs.append(f"reference_ids mismatch: {ref_bad[:5]}")
if score_bad:
    errs.append(f"non-finite/missing scores: {score_bad[:5]}")

n_fail = sum(1 for r in recs if r.get("status") != "ok")
if n_fail:
    errs.append(f"{n_fail} failed records: " +
                str([rec_key(r) for r in recs
                     if r.get("status") != "ok"][:5]))

meta = d.get("meta", {})
sha = meta.get("candidates_sha256", {})
if meta.get("prompt_sha256") != FROZEN_PROMPT_SHA:
    errs.append(f"prompt sha mismatch: {meta.get('prompt_sha256')}")
if sha.get("C_OK") != FROZEN_C_OK_SHA or sha.get("C_NG") != FROZEN_C_NG_SHA:
    errs.append(f"candidate sha mismatch: {sha}")

if errs:
    print("VERIFY_FAIL", a.out)
    for e in errs:
        print("  -", e)
    sys.exit(1)
print(f"VERIFY_OK {a.out}: n={len(recs)} unique labels_ok refs_ok "
      f"sha_ok 0_fail all_finite")
