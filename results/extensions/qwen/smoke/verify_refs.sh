#!/bin/bash
# Verify archived refs match rebuilt 1cls reference pool exactly.
LOG=/data/disk2/home/qzheng610/qwen_work/verify_refs.log
exec > "$LOG" 2>&1
echo "START $(date)"
REFS=/data/disk2/home/qzheng610/qwen_work/refs_archive
DATA=/data/disk2/home/qzheng610/qwen_work/data/VisA_pytorch/1cls
/tmp/slv176/venv/bin/python - "$REFS" "$DATA" <<'PY'
import os, sys, json
refs_root, data_root = sys.argv[1], sys.argv[2]
cats=["candle","capsules","cashew","chewinggum","fryum","pipe_fryum"]
ok=bad=0
for shot in (1,2,4,8):
    for seed in (0,1,2):
        d=os.path.join(refs_root,f"{shot}shot_seed{seed}")
        if not os.path.isdir(d):
            print("MISSING DIR",d); bad+=1; continue
        for c in cats:
            rp=os.path.join(d,f"{c}_shot={shot}_seed={seed}.json")
            if not os.path.exists(rp):
                print("MISSING FILE",rp); bad+=1; continue
            rec=json.load(open(rp))
            archived=rec["reference_files"]
            names=sorted(os.listdir(os.path.join(data_root,c,"train","good")))
            pool=names[20:]
            expected=pool[seed*shot:(seed+1)*shot]
            if archived==expected:
                ok+=1
            else:
                bad+=1
                print("MISMATCH",shot,seed,c,"archived",archived,"expected",expected)
print(f"OK={ok} BAD={bad} TOTAL={ok+bad}")
PY
echo "VERIFY_DONE $(date)"
