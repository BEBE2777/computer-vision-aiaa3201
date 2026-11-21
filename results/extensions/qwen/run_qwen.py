"""Qwen-VL few-shot anomaly inspection over VisA (B line, independent of CLIP).

For each (shot, seed): every category's 960 test images are queried with n archived
normal references + the frozen prompt. Predictions, raw responses, failures, latency
and GPU memory are saved per image. NO label/mask/filename-truth enters the prompt.

Usage on GPU box:
  PYTHONPATH=<pylibs> /tmp/slv176/venv/bin/python run_qwen.py \
    --shots 1 2 4 8 --seeds 0 1 2 --out <run_dir>
"""
import os, sys, json, time, hashlib, argparse, traceback
import torch

BASE = "/data/disk2/home/qzheng610/qwen_work"
DATA = BASE + "/data/VisA_pytorch/1cls"
REFS = BASE + "/refs_archive"
PROMPT_PATH = BASE + "/prompt_v1.txt"
PROMPT = open(PROMPT_PATH, encoding="utf-8").read()
PROMPT_SHA = hashlib.sha256(PROMPT.encode()).hexdigest()
CATS = ["candle", "capsules", "cashew", "chewinggum", "fryum", "pipe_fryum"]


def load_refs(shot, seed, cat):
    p = f"{REFS}/{shot}shot_seed{seed}/{cat}_shot={shot}_seed={seed}.json"
    return json.load(open(p))["reference_files"]


def list_queries(cat):
    """Return (label, filename) pairs; label is for offline eval only."""
    out = []
    for lbl, sub in (("good", "good"), ("bad", "bad")):
        for f in sorted(os.listdir(f"{DATA}/{cat}/test/{sub}")):
            out.append((lbl, f, f"{DATA}/{cat}/test/{sub}/{f}"))
    return out


def build_messages(cat, refs, query_path, n):
    content = [{"type": "text", "text": PROMPT.replace("{category}", cat).replace("{n}", str(n))}]
    for i, r in enumerate(refs, 1):
        content += [{"type": "text", "text": f"REFERENCE_{i}:"},
                    {"type": "image", "image": f"{DATA}/{cat}/train/good/{r}"}]
    content += [{"type": "text", "text": "QUERY:"},
                {"type": "image", "image": query_path}]
    return [{"role": "user", "content": content}]


def parse_json(raw):
    try:
        return json.loads(raw.strip()), None
    except Exception as e1:
        try:
            return json.loads(raw[raw.index("{"):raw.rindex("}")+1]), None
        except Exception as e2:
            return None, f"{type(e1).__name__}:{e1} | {type(e2).__name__}:{e2}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--categories", nargs="*", default=CATS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit-per-class", type=int, default=0, help="0=all; >0 for subset")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--model", choices=["2b", "7b"], default="2b")
    args = ap.parse_args()

    model_dir = f"{BASE}/qwen2vl{'7b' if args.model=='7b' else '2b'}_local"
    model_id = "Qwen/Qwen2-VL-7B-Instruct" if args.model == "7b" else "Qwen/Qwen2-VL-2B-Instruct"
    os.makedirs(args.out, exist_ok=True)
    pred_path = os.path.join(args.out, "predictions.jsonl")
    fail_path = os.path.join(args.out, "failures.jsonl")
    raw_dir = os.path.join(args.out, "raw_responses")
    os.makedirs(raw_dir, exist_ok=True)

    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    print(f"loading model {model_dir} ...", flush=True)
    t_load = time.time()
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_dir, local_files_only=True, torch_dtype=torch.bfloat16).cuda().eval()
    load_s = time.time() - t_load
    config = {
        "experiment_id": "qwen_inspect_v1", "model_id": model_id,
        "model_revision": open(os.path.join(model_dir, "config.json")).read()[:0] or "hf-snapshot-2026-11-18",
        "precision": "bfloat16", "device": torch.cuda.get_device_name(0),
        "prompt_file": "prompts/qwen_inspect_v1.txt", "prompt_sha256": PROMPT_SHA,
        "schema_file": "prompts/schema_v1.json",
        "decode": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        "shots": args.shots, "seeds": args.seeds, "categories": args.categories,
        "limit_per_class": args.limit_per_class, "model_load_seconds": round(load_s, 1),
        "transformers": __import__("transformers").__version__, "torch": torch.__version__,
    }
    json.dump(config, open(os.path.join(args.out, "config.json"), "w"), indent=2)

    f_pred = open(pred_path, "a", encoding="utf-8")
    f_fail = open(fail_path, "a", encoding="utf-8")
    n_ok = n_fail = 0
    latencies = []

    for shot in args.shots:
        for seed in args.seeds:
            for cat in args.categories:
                refs = load_refs(shot, seed, cat)
                queries = list_queries(cat)
                if args.limit_per_class:
                    # deterministic balanced subset: 1:1 good/bad
                    good = [q for q in queries if q[0] == "good"]
                    bad = [q for q in queries if q[0] == "bad"]
                    k = min(args.limit_per_class, len(good), len(bad))
                    queries = good[:k] + bad[:k]
                for true_label, qname, qpath in queries:
                    rec = {
                        "experiment_id": config["experiment_id"], "category": cat,
                        "image_id": os.path.splitext(qname)[0], "shot": shot,
                        "reference_seed": seed, "repeat_id": 0,
                        "reference_ids": refs, "model_id": model_id,
                        "prompt_sha256": PROMPT_SHA,
                        "true_label_for_eval_ONLY": true_label,
                        "status": "ok", "latency_ms": None, "input_tokens": None,
                        "output_tokens": None, "peak_gpu_gb": None,
                        "anomaly_score": None, "decision": None, "confidence": None,
                        "defect_type": None, "evidence": None, "regions": None,
                        "raw_response_file": None, "error": None,
                    }
                    try:
                        messages = build_messages(cat, refs, qpath, shot)
                        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        images, videos = process_vision_info(messages)
                        inputs = processor(text=[text], images=images, videos=videos,
                                           return_tensors="pt", padding=True).to("cuda")
                        torch.cuda.reset_peak_memory_stats()
                        t0 = time.time()
                        with torch.no_grad():
                            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                                 do_sample=False,
                                                 pad_token_id=processor.tokenizer.eos_token_id)
                        lat = (time.time() - t0) * 1000
                        peak = torch.cuda.max_memory_allocated() / 1e9
                        raw = processor.batch_decode(out[:, inputs.input_ids.shape[1]:],
                                                     skip_special_tokens=True)[0]
                        parsed, err = parse_json(raw)
                        rid = f"s{shot}_sd{seed}_{cat}_{os.path.splitext(qname)[0]}"
                        rf = os.path.join(raw_dir, rid + ".txt")
                        open(rf, "w", encoding="utf-8").write(raw)
                        rec.update({
                            "latency_ms": round(lat, 1), "peak_gpu_gb": round(peak, 2),
                            "input_tokens": int(inputs.input_ids.shape[1]),
                            "output_tokens": int(out.shape[1] - inputs.input_ids.shape[1]),
                            "raw_response_file": os.path.relpath(rf, args.out),
                        })
                        if parsed is not None:
                            rec.update({
                                "anomaly_score": parsed.get("anomaly_score"),
                                "decision": parsed.get("decision"),
                                "confidence": parsed.get("confidence"),
                                "defect_type": parsed.get("defect_type"),
                                "evidence": parsed.get("evidence"),
                                "regions": parsed.get("regions"),
                            })
                            n_ok += 1
                            latencies.append(lat)
                        else:
                            rec.update({"status": "json_fail", "error": err})
                            n_fail += 1
                            f_fail.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    except Exception as e:
                        rec.update({"status": "error", "error": f"{type(e).__name__}:{e}"})
                        n_fail += 1
                        f_fail.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        traceback.print_exc()
                    f_pred.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f_pred.flush(); f_fail.flush()
                    if (n_ok + n_fail) % 20 == 0:
                        print(f"progress ok={n_ok} fail={n_fail} shot={shot} seed={seed} {cat}", flush=True)

    f_pred.close(); f_fail.close()
    cost = {
        "n_success": n_ok, "n_fail": n_fail,
        "json_valid_rate": n_ok / (n_ok + n_fail) if (n_ok + n_fail) else 0,
        "mean_latency_ms": sum(latencies)/len(latencies) if latencies else None,
        "p95_latency_ms": sorted(latencies)[int(0.95*len(latencies))] if latencies else None,
        "total_inference_seconds": sum(latencies)/1000,
        "device": torch.cuda.get_device_name(0),
        "cost_note": "Local GPU; monetary cost N/A. GPU-hours = total_inference_seconds/3600.",
    }
    json.dump(cost, open(os.path.join(args.out, "cost_summary.json"), "w"), indent=2)
    print("DONE", json.dumps(cost), flush=True)


if __name__ == "__main__":
    main()
