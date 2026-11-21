"""B0 GPU smoke test on the course A6000.

Uses REAL VisA images rebuilt from the official 1cls.csv, with reference images
taken from the ARCHIVED refs (same as the DINO baseline). For each query we pass
only: category name, n normal reference images, the query image, and the frozen
prompt. NO labels, masks, good/bad paths or filename truth are shown to the model.

Purpose: verify multi-image input (1/4/8 refs), JSON validity, continuous score
behaviour, latency and peak GPU memory on the actual deployment device.
"""
import os, sys, json, time, hashlib, argparse, random
import torch

BASE = "/data/disk2/home/qzheng610/qwen_work"
DATA = BASE + "/data/VisA_pytorch/1cls"
REFS = BASE + "/refs_archive"
PROMPT = open("/data/disk2/home/qzheng610/qwen_work/prompt_v1.txt", encoding="utf-8").read()
OUT = BASE + "/smoke_out"

CATS = ["candle", "capsules", "cashew", "chewinggum", "fryum", "pipe_fryum"]


def load_refs(shot, seed, cat):
    p = f"{REFS}/{shot}shot_seed{seed}/{cat}_shot={shot}_seed={seed}.json"
    return json.load(open(p))["reference_files"]


def list_queries(cat):
    good = sorted(os.listdir(f"{DATA}/{cat}/test/good"))
    bad = sorted(os.listdir(f"{DATA}/{cat}/test/bad"))
    return [("good", g) for g in good], [("bad", b) for b in bad]


def build_messages(cat, ref_files, query_path, n):
    content = [{"type": "text", "text": PROMPT.replace("{category}", cat).replace("{n}", str(n))}]
    for i, r in enumerate(ref_files, 1):
        content.append({"type": "text", "text": f"REFERENCE_{i}:"})
        content.append({"type": "image", "image": f"{DATA}/{cat}/train/good/{r}"})
    content.append({"type": "text", "text": "QUERY:"})
    content.append({"type": "image", "image": query_path})
    return [{"role": "user", "content": content}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, nargs="+", default=[1])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-class", type=int, default=3, help="N good + N bad queries per class")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--model", choices=["2b", "7b"], default="2b",
                    help="checkpoint to smoke-test; 2b is the frozen default")
    args = ap.parse_args()

    model_dir = f"{BASE}/qwen2vl{'7b' if args.model=='7b' else '2b'}_local"
    model_id = "Qwen/Qwen2-VL-7B-Instruct" if args.model == "7b" else "Qwen/Qwen2-VL-2B-Instruct"
    os.makedirs(OUT, exist_ok=True)
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    print(f"load model {model_dir} ...", flush=True)
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_dir, local_files_only=True, torch_dtype=torch.bfloat16).cuda().eval()
    print(f"model on {torch.cuda.get_device_name(0)} bf16", flush=True)

    rng = random.Random(0)
    records = []
    for shot in args.shots:
        for cat in CATS:
            refs = load_refs(shot, args.seed, cat)
            good, bad = list_queries(cat)
            picks = [(lbl, p) for lst in (good, bad) for lbl, p in rng.sample(lst, min(args.per_class, len(lst)))]
            for true_label, qname in picks:
                qpath = f"{DATA}/{cat}/test/{true_label}/{qname}"
                messages = build_messages(cat, refs, qpath, shot)
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                images, videos = process_vision_info(messages)
                inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt", padding=True).to("cuda")
                torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                         pad_token_id=processor.tokenizer.eos_token_id)
                lat = (time.time()-t0)*1000
                peak = torch.cuda.max_memory_allocated()/1e9
                raw = processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
                parsed, err = None, None
                try:
                    parsed = json.loads(raw.strip())
                except Exception as e1:
                    try:
                        parsed = json.loads(raw[raw.index("{"):raw.rindex("}")+1])
                    except Exception as e2:
                        err = f"{e1} | {e2}"
                rec = {
                    "shot": shot, "seed": args.seed, "category": cat,
                    "true_label_for_eval_ONLY": true_label,  # never sent to model
                    "query_id": qname, "reference_ids": refs,
                    "latency_ms": round(lat, 1), "peak_gpu_gb": round(peak, 2),
                    "input_tokens": int(inputs.input_ids.shape[1]),
                    "status": "ok" if parsed else "json_fail", "parse_error": err,
                    "anomaly_score": parsed.get("anomaly_score") if parsed else None,
                    "decision": parsed.get("decision") if parsed else None,
                    "confidence": parsed.get("confidence") if parsed else None,
                    "raw_response": raw,
                }
                records.append(rec)
                print(f"shot={shot} {cat:12s} {true_label:4s} lat={lat:.0f}ms peak={peak:.1f}GB "
                      f"score={rec['anomaly_score']} dec={rec['decision']} {rec['status']}", flush=True)

    report = {
        "model_id": model_id, "device": torch.cuda.get_device_name(0),
        "dtype": "bfloat16", "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "n_records": len(records),
        "json_valid_rate": sum(r["status"]=="ok" for r in records)/len(records),
        "mean_latency_ms": sum(r["latency_ms"] for r in records)/len(records),
        "mean_peak_gpu_gb": sum(r["peak_gpu_gb"] for r in records)/len(records),
        "records": records,
    }
    op = f"{OUT}/gpu_smoke_{args.model}_shots_{'_'.join(map(str,args.shots))}.json"
    json.dump(report, open(op, "w"), indent=2, ensure_ascii=False)
    print("WROTE", op, flush=True)


if __name__ == "__main__":
    main()
