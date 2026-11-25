"""Acceptance check for one v5-ext run cell. Exits 0 only if:
- every manifest entry has a matching record (category/query_id set equal)
- 0 records with status != ok
- every ok record has finite fc_score / sum_logP_OK / sum_logP_NG
- reference_ids match the manifest entry (archived refs respected)

Usage: verify_v5ext.py --manifest M --out P
"""
import argparse, json, math, sys

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

man = json.load(open(a.manifest))["entries"]
d = json.load(open(a.out))
recs = d.get("records", [])
errs = []

by_key = {}
for r in recs:
    by_key[f"{r['category']}/{r['query_id']}"] = r

want = {f"{e['category']}/{e['query_id']}": e for e in man}
missing = [k for k in want if k not in by_key]
extra = [k for k in by_key if k not in want]
if missing:
    errs.append(f"missing {len(missing)}: {missing[:5]}")
if extra:
    errs.append(f"extra {len(extra)}: {extra[:5]}")

n_fail = sum(1 for r in recs if r.get("status") != "ok")
if n_fail:
    errs.append(f"{n_fail} failed records: " +
                str([f"{r['category']}/{r['query_id']}" for r in recs
                     if r.get("status") != "ok"][:5]))

bad_score = []
ref_bad = []
for e in man:
    r = by_key.get(f"{e['category']}/{e['query_id']}")
    if not r or r.get("status") != "ok":
        continue
    for f in ("fc_score", "sum_logP_OK", "sum_logP_NG"):
        v = r.get(f)
        if v is None or not isinstance(v, (int, float)) or not math.isfinite(v):
            bad_score.append(f"{e['category']}/{e['query_id']}:{f}")
            break
    if r.get("reference_ids") != e["reference_files"]:
        ref_bad.append(f"{e['category']}/{e['query_id']}")
if bad_score:
    errs.append(f"non-finite/missing scores: {bad_score[:5]}")
if ref_bad:
    errs.append(f"reference_ids mismatch: {ref_bad[:5]}")

meta = d.get("meta", {})
if meta.get("prompt_sha256") is None:
    errs.append("metadata missing prompt sha")

if errs:
    print("VERIFY_FAIL", a.out)
    for e in errs:
        print("  -", e)
    sys.exit(1)
print(f"VERIFY_OK {a.out}: n={len(recs)} 0_fail all_finite refs_ok")
