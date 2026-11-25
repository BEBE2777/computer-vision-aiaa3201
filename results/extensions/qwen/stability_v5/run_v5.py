"""v5 stability runner (server, A6000). Forced-choice scoring ONLY.

Differences from run_v2.py (per qwen_stability_v5.md):
- fc-only (no free-generation JSON readout; known-degenerate since v1)
- protocol metadata: v5, prompt sha256, candidate sha256, manifest path
- resume: merges by manifest query_id; failures recorded, never dropped
Score convention: fc_score = sum_logP(C_NG) - sum_logP(C_OK);
fc_conf = -fc_score (higher = more anomalous). Direction frozen.
"""
import os, json, time, hashlib, argparse
import torch

BASE = "/data/disk2/home/qzheng610/qwen_work"
PROMPT = open(BASE + "/prompt_v1.txt", encoding="utf-8").read()

C_OK = '{"decision":"OK","anomaly_score":0.0,"confidence":0.5,"defect_type":"unknown","evidence":"","regions":[]}'
C_NG = '{"decision":"NG","anomaly_score":1.0,"confidence":0.5,"defect_type":"unknown","evidence":"","regions":[]}'


def build_messages(entry):
    cat, n = entry["category"], len(entry["reference_files"])
    content = [{"type": "text", "text": PROMPT.replace("{category}", cat).replace("{n}", str(n))}]
    for i, p in enumerate(entry["reference_paths"], 1):
        content.append({"type": "text", "text": f"REFERENCE_{i}:"})
        content.append({"type": "image", "image": p})
    content.append({"type": "text", "text": "QUERY:"})
    content.append({"type": "image", "image": entry["query_path"]})
    return [{"role": "user", "content": content}]


def score_candidates(model, processor, prompt_inputs):
    tok = processor.tokenizer
    sums = {}
    n_cand = None
    for name, cand in (("OK", C_OK), ("NG", C_NG)):
        cand_ids = tok(cand, add_special_tokens=False).input_ids
        n_cand = len(cand_ids)
        ids = torch.cat([prompt_inputs["input_ids"],
                         torch.tensor([cand_ids], dtype=torch.long,
                                      device=prompt_inputs["input_ids"].device)], dim=1)
        attn = torch.cat([prompt_inputs["attention_mask"],
                          torch.ones(1, n_cand, dtype=torch.long,
                                     device=ids.device)], dim=1)
        kwargs = {"input_ids": ids, "attention_mask": attn,
                  "pixel_values": prompt_inputs.get("pixel_values"),
                  "image_grid_thw": prompt_inputs.get("image_grid_thw")}
        if "mm_token_type_ids" in prompt_inputs:
            kwargs["mm_token_type_ids"] = torch.cat(
                [prompt_inputs["mm_token_type_ids"],
                 torch.zeros(1, n_cand, dtype=torch.long, device=ids.device)], dim=1)
        with torch.no_grad():
            out = model(**kwargs)
        logprobs = torch.log_softmax(out.logits[0].float(), dim=-1)
        L = ids.shape[1]
        s = 0.0
        for i in range(L - n_cand, L):
            s += logprobs[i - 1, ids[0, i].item()].item()
        sums[name] = s
        del out, logprobs
    return sums["OK"], sums["NG"], n_cand


def meta(args, model_id, n):
    return {
        "protocol": "qwen_fc_conf_stability_v5",
        "model_id": model_id, "model_dir": args.model, "dtype": "bfloat16",
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "candidates_sha256": {
            "C_OK": hashlib.sha256(C_OK.encode()).hexdigest(),
            "C_NG": hashlib.sha256(C_NG.encode()).hexdigest()},
        "candidates": {"C_OK": C_OK, "C_NG": C_NG},
        "fc_score_def": "sum_logP(C_NG) - sum_logP(C_OK); fc_conf = -fc_score",
        "manifest": args.manifest, "n_records": n,
        "device": "NVIDIA RTX A6000",
    }


def load_done(out_path):
    """Resume: collect records already scored (from partial or final).
    Key = category/label/query_id: bare query_id is NOT unique across VisA
    categories (0000.JPG reused), and test/good vs test/bad also share
    filenames (capsules has 24 such pairs) -> the label must be in the key,
    otherwise a resume silently drops one of the two images."""
    done = {}
    for p in (out_path + ".partial", out_path):
        if os.path.exists(p):
            try:
                for r in json.load(open(p))["records"]:
                    done[f"{r['category']}/{r['true_label_for_eval_ONLY']}/{r['query_id']}"] = r
            except Exception as e:
                print(f"WARN: cannot read {p}: {e}", flush=True)
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["2b", "7b"], default="7b")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N entries")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--fresh", action="store_true", help="ignore previous records")
    args = ap.parse_args()

    model_dir = f"{BASE}/qwen2vl{args.model}_local"
    model_id = "Qwen/Qwen2-VL-7B-Instruct" if args.model == "7b" else "Qwen/Qwen2-VL-2B-Instruct"

    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    print(f"load {model_dir} ...", flush=True)
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_dir, local_files_only=True, torch_dtype=torch.bfloat16).cuda().eval()
    print(f"model on {torch.cuda.get_device_name(0)} bf16", flush=True)

    manifest = json.load(open(args.manifest))
    entries = manifest["entries"][args.offset: args.offset + args.limit] if args.limit else manifest["entries"][args.offset:]
    done = {} if args.fresh else load_done(args.out)
    todo = [e for e in entries
            if f"{e['category']}/{e['true_label_for_eval_ONLY']}/{e['query_id']}" not in done]
    n_fail_before = sum(1 for r in done.values() if r.get("status") == "fail")
    print(f"entries: {len(todo)} to run (+{len(done)} resumed, "
          f"{n_fail_before} prior failures preserved)", flush=True)

    records = list(done.values())
    t_start = time.time()
    for idx, e in enumerate(todo):
        rec = {"category": e["category"],
               "true_label_for_eval_ONLY": e["true_label_for_eval_ONLY"],
               "query_id": e["query_id"],
               "reference_ids": e["reference_files"],
               "status": "fail", "error": None}
        try:
            messages = build_messages(e)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos = process_vision_info(messages)
            inputs = processor(text=[text], images=images, videos=videos,
                               return_tensors="pt", padding=True).to("cuda")
            torch.cuda.reset_peak_memory_stats()
            t1 = time.time()
            s_ok, s_ng, n_cand = score_candidates(model, processor, inputs)
            fc_ms = (time.time() - t1) * 1000
            rec.update({
                "input_tokens": int(inputs.input_ids.shape[1]),
                "latency_fc_ms": round(fc_ms, 1),
                "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                "status": "ok",
                "sum_logP_OK": round(s_ok, 4),
                "sum_logP_NG": round(s_ng, 4),
                "n_cand_tokens": n_cand,
                "fc_score": round(s_ng - s_ok, 4),
                "fc_score_mean": round((s_ng - s_ok) / max(n_cand, 1), 4),
            })
            print(f"[{args.offset+idx+1}/{len(entries)}] {e['category']:11s} "
                  f"{e['true_label_for_eval_ONLY']:4s} fc={fc_ms:.0f}ms "
                  f"fc_conf={-(s_ng - s_ok):.2f}", flush=True)
        except Exception as ex:
            rec["error"] = f"{type(ex).__name__}: {ex}"[:300]
            print(f"[{args.offset+idx+1}] FAIL {e['category']}/{e['query_id']}: {rec['error']}", flush=True)
        records.append(rec)
        if (idx + 1) % 10 == 0 or idx == len(todo) - 1:
            json.dump({"meta": meta(args, model_id, len(records)), "records": records},
                      open(args.out + ".partial", "w"), indent=1, ensure_ascii=False)

    # final: order records by manifest order, keep failures visible
    order = {f"{e['category']}/{e['true_label_for_eval_ONLY']}/{e['query_id']}": i
             for i, e in enumerate(entries)}
    records.sort(key=lambda r: order.get(
        f"{r['category']}/{r['true_label_for_eval_ONLY']}/{r['query_id']}", 10**9))
    json.dump({"meta": meta(args, model_id, len(records)), "records": records},
              open(args.out, "w"), indent=1, ensure_ascii=False)
    n_fail = sum(1 for r in records if r.get("status") == "fail")
    print(f"WROTE {args.out} n={len(records)} fail={n_fail} "
          f"total {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
