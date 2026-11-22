"""Feasibility check (B line, exploratory pilot v2): can we compute sequence
log-probabilities of candidate answers "OK" vs "NG" completions with the
installed transformers + local Qwen2-VL-2B-Instruct checkpoint?

Method: teacher-forced scoring. For each candidate completion string C:
  input_ids = prompt_ids (with image placeholders expanded) + candidate_ids
  forward pass -> log_softmax over vocab -> sum log P(token_i) for i in candidate positions
forced_choice_score = logP(C_NG) - logP(C_OK)   (higher = more anomalous)

No labels or ground truth are involved. Runs on CPU with two synthetic images.
This script only checks CODE feasibility; it is NOT an evaluation result.
"""
import torch, json, sys
from PIL import Image, ImageDraw

MODEL_DIR = "C:/Users/qzheng/.cache/huggingface/hub/qwen2vl2b_local"

# Candidate completions (mirror the model's natural JSON output format; neutral
# confidence 0.5 for both; only decision + anomaly_score differ).
C_OK = '{"decision":"OK","anomaly_score":0.0,"confidence":0.5,"defect_type":"unknown","evidence":"","regions":[]}'
C_NG = '{"decision":"NG","anomaly_score":1.0,"confidence":0.5,"defect_type":"unknown","evidence":"","regions":[]}'

PROMPT = ("You are inspecting a {category} product. Compare the QUERY image against "
          "{n} REFERENCE images of normal products. Output a single JSON object with "
          'keys: decision (OK or NG), anomaly_score (0.0-1.0), confidence, defect_type, '
          'evidence, regions. No extra text.')


def synth(color, mark):
    img = Image.new("RGB", (224, 224), color)
    d = ImageDraw.Draw(img)
    d.ellipse([60, 60, 164, 164], outline=(0, 0, 0), width=6)
    if mark:
        d.line([60, 60, 164, 164], fill=(200, 30, 30), width=8)
    return img


def score_candidate(model, tok, prompt_inputs, cand_text):
    """Teacher-forced scoring of candidate continuation appended after the
    chat-template prompt (images already expanded into prompt input_ids)."""
    cand_ids = tok(cand_text, add_special_tokens=False).input_ids
    k = len(cand_ids)
    import torch
    input_ids = torch.cat([prompt_inputs["input_ids"],
                           torch.tensor([cand_ids], dtype=torch.long)], dim=1)
    attn = torch.cat([prompt_inputs["attention_mask"],
                      torch.ones(1, k, dtype=torch.long)], dim=1)
    kwargs = {"input_ids": input_ids, "attention_mask": attn,
              "pixel_values": prompt_inputs.get("pixel_values"),
              "image_grid_thw": prompt_inputs.get("image_grid_thw")}
    if "mm_token_type_ids" in prompt_inputs:
        import torch as _t
        kwargs["mm_token_type_ids"] = _t.cat(
            [prompt_inputs["mm_token_type_ids"], _t.zeros(1, k, dtype=torch.long)], dim=1)
    L = input_ids.shape[1]
    with torch.no_grad():
        out = model(**kwargs)
        logprobs = torch.log_softmax(out.logits[0].float(), dim=-1)
        lp = 0.0
        per_tok = []
        for i in range(L - k, L):
            t = input_ids[0, i].item()
            v = logprobs[i - 1, t].item()
            lp += v
            per_tok.append(round(v, 4))
    return lp, per_tok, int(L)


def main():
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    print("loading processor/model (CPU) ...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_DIR, local_files_only=True, torch_dtype=torch.float32)
    model.eval()
    print("loaded OK", flush=True)

    images = [synth((210, 210, 210), False), synth((200, 205, 215), True)]
    messages = [{"role": "user", "content": [
        {"type": "text", "text": PROMPT.replace("{category}", "candle").replace("{n}", "2")},
        {"type": "text", "text": "REFERENCE_1:"},
        {"type": "image", "image": images[0]},
        {"type": "text", "text": "REFERENCE_2:"},
        {"type": "image", "image": images[1]},
        {"type": "text", "text": "QUERY:"},
        {"type": "image", "image": images[1]},
    ]}]
    prompt_inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt")
    tok = processor.tokenizer
    print("prompt input_ids shape:", tuple(prompt_inputs["input_ids"].shape), flush=True)

    res = {}
    for name, cand in [("OK", C_OK), ("NG", C_NG)]:
        lp, per_tok, L = score_candidate(model, tok, prompt_inputs, cand)
        res[name] = {"sum_logprob": round(lp, 4), "total_len": L, "per_token_head": per_tok[:6]}
        print(f"{name}: sum_logP={lp:.4f} total_len={L}", flush=True)

    fcs = res["NG"]["sum_logprob"] - res["OK"]["sum_logprob"]
    res["forced_choice_score"] = round(fcs, 4)
    res["feasible"] = True
    print("forced_choice_score (logP_NG - logP_OK) =", round(fcs, 4), flush=True)
    json.dump(res, open(__file__.replace(".py", "_result.json"), "w"), indent=2)
    print("FEASIBILITY: PASS", flush=True)


if __name__ == "__main__":
    main()
