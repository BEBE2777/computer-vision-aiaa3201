"""v2 exploratory pilot runner (server, A6000). Reads the FROZEN manifest and
per query produces BOTH scores:
  (1) legacy: greedy JSON generation -> anomaly_score (same as v1 protocol)
  (2) forced-choice: teacher-forced sequence log-prob of two frozen candidate
      completions; fc_score = sum_logP(C_NG) - sum_logP(C_OK)

No labels, masks, or filenames with truth cues are sent to the model; the
manifest's true_label_for_eval_ONLY is never touched here (offline eval only).
"""
import os, sys, json, time, hashlib, argparse
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


def parse_json(raw):
    try:
        return json.loads(raw.strip()), None
    except Exception as e1:
        try:
            return json.loads(raw[raw.index("{"):raw.rindex("}") + 1]), str(e1)
        except Exception as e2:
            return None, f"{e1} | {e2}"


def score_candidates(model, processor, prompt_inputs):
    """Two teacher-forced forwards; candidate ids appended after the generation
    prompt. Returns (sum_logP_OK, sum_logP_NG, n_cand_tokens)."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["2b", "7b"], default="7b")
    ap.add_argument("--manifest", default=BASE + "/v2/manifest_v2.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N entries")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--skip-gen", action="store_true",
                    help="calibration mode: fc scoring only, no JSON generation")
    args = ap.parse_args()

    out_path = args.out or f"{BASE}/v2/predictions_v2_{args.model}.json"
    model_dir = f"{BASE}/qwen2vl{args.model}_local"
    model_id = f"Qwen/Qwen2-VL-{args.model.upper().replace('B','-B')}-Instruct".replace("2-B", "2-B")
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
    print(f"entries: {len(entries)} (offset {args.offset})", flush=True)

    records = []
    t_start = time.time()
    for idx, e in enumerate(entries):
        messages = build_messages(e)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        inputs = processor(text=[text], images=images, videos=videos,
                           return_tensors="pt", padding=True).to("cuda")
        torch.cuda.reset_peak_memory_stats()

        # pass A: greedy JSON (legacy readout)
        gen_ms, raw, parsed, perr = None, "", None, None
        if not args.skip_gen:
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                     do_sample=False,
                                     pad_token_id=processor.tokenizer.eos_token_id)
            gen_ms = (time.time() - t0) * 1000
            raw = processor.batch_decode(out[:, inputs.input_ids.shape[1]:],
                                         skip_special_tokens=True)[0]
            parsed, perr = parse_json(raw)

        # pass B/C: forced-choice scoring
        t1 = time.time()
        s_ok, s_ng, n_cand = score_candidates(model, processor, inputs)
        fc_ms = (time.time() - t1) * 1000

        rec = {
            "idx": args.offset + idx,
            "category": e["category"],
            "true_label_for_eval_ONLY": e["true_label_for_eval_ONLY"],
            "query_id": e["query_id"],
            "reference_ids": e["reference_files"],
            "input_tokens": int(inputs.input_ids.shape[1]),
            "latency_gen_ms": round(gen_ms, 1) if gen_ms is not None else None,
            "latency_fc_ms": round(fc_ms, 1),
            "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "status": ("fc_only" if args.skip_gen else
                       ("ok" if parsed else "json_fail")),
            "parse_error": perr,
            "anomaly_score": parsed.get("anomaly_score") if parsed else None,
            "decision": parsed.get("decision") if parsed else None,
            "confidence": parsed.get("confidence") if parsed else None,
            "raw_response": raw,
            "sum_logP_OK": round(s_ok, 4),
            "sum_logP_NG": round(s_ng, 4),
            "n_cand_tokens": n_cand,
            "fc_score": round(s_ng - s_ok, 4),
            "fc_score_mean": round((s_ng - s_ok) / max(n_cand, 1), 4),
        }
        records.append(rec)
        gen_str = "skip" if gen_ms is None else f"{gen_ms:.0f}ms"
        print(f"[{args.offset+idx+1}/{len(entries)+args.offset}] {e['category']:11s} "
              f"{e['true_label_for_eval_ONLY']:4s} gen={gen_str} fc={fc_ms:.0f}ms "
              f"json={rec['anomaly_score']} fc={rec['fc_score']:.2f}", flush=True)
        if (idx + 1) % 10 == 0 or idx == len(entries) - 1:
            partial = {"meta": meta(args, model_id, len(records)), "records": records}
            tmp = out_path + ".partial"
            json.dump(partial, open(tmp, "w"), indent=1, ensure_ascii=False)

    report = {"meta": meta(args, model_id, len(records)), "records": records}
    json.dump(report, open(out_path, "w"), indent=1, ensure_ascii=False)
    print("WROTE", out_path, f"total {time.time()-t_start:.0f}s", flush=True)


def meta(args, model_id, n):
    return {
        "protocol": "qwen_exploratory_pilot_v2",
        "model_id": model_id, "model_dir": args.model, "dtype": "bfloat16",
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "candidates": {"C_OK": C_OK, "C_NG": C_NG},
        "fc_score_def": "sum_logP(C_NG) - sum_logP(C_OK)",
        "manifest": args.manifest, "n_records": n,
        "device": "NVIDIA RTX A6000",
    }


if __name__ == "__main__":
    main()
