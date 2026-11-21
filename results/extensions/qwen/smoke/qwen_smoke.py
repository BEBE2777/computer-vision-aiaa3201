"""B0 smoke test for the Qwen-VL inspection branch.

Uses SYNTHETIC images only (plain geometric shapes). No VisA data, no labels,
no masks, no defect ground truth. Purpose: verify that a Qwen2-VL checkpoint
can accept 1/4/8 normal references + 1 query, return parseable JSON with a
continuous anomaly score, and record latency / memory / JSON validity.
"""
import argparse, json, os, time, hashlib, torch
from pathlib import Path
from PIL import Image, ImageDraw

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
LOCAL_DIR = r"C:/Users/qzheng/.cache/huggingface/hub/qwen2vl2b_local"
PROMPT_FILE = Path(__file__).resolve().parent.parent / "prompts" / "qwen_inspect_v1.txt"
SCHEMA_FILE = Path(__file__).resolve().parent.parent / "prompts" / "schema_v1.json"
OUT = Path(__file__).resolve().parent


def make_synth(path, shape, color, defect=False):
    img = Image.new("RGB", (448, 448), (235, 235, 230))
    d = ImageDraw.Draw(img)
    if shape == "circle":
        d.ellipse([120, 120, 328, 328], fill=color, outline=(60, 60, 60), width=4)
    elif shape == "square":
        d.rectangle([120, 120, 328, 328], fill=color, outline=(60, 60, 60), width=4)
    else:
        d.polygon([(224, 100), (340, 340), (108, 340)], fill=color, outline=(60, 60, 60))
    if defect:
        # a small dark mark, clearly visible, synthetic "defect"
        d.ellipse([210, 200, 245, 235], fill=(20, 20, 20))
    img.save(path)
    return path


def build_inputs(processor, n_ref, shape, defect_query):
    prompt = PROMPT_FILE.read_text(encoding="utf-8")
    tmpdir = OUT / "synth"
    tmpdir.mkdir(exist_ok=True)
    refs = []
    for i in range(n_ref):
        p = tmpdir / f"ref_{n_ref}_{i}.png"
        make_synth(p, shape, (180, 210, 190), defect=False)
        refs.append(p)
    qp = tmpdir / f"query_{'bad' if defect_query else 'good'}_{n_ref}.png"
    make_synth(qp, shape, (180, 210, 190), defect=defect_query)

    content = []
    content.append({"type": "text", "text": prompt.replace("{category}", "synthetic part").replace("{n}", str(n_ref))})
    for i, r in enumerate(refs, 1):
        content.append({"type": "text", "text": f"REFERENCE_{i}:"})
        content.append({"type": "image", "image": str(r)})
    content.append({"type": "text", "text": "QUERY:"})
    content.append({"type": "image", "image": str(qp)})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    from qwen_vl_utils import process_vision_info
    images, videos = process_vision_info(messages)
    inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt", padding=True)
    return inputs, [str(r) for r in refs], str(qp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-refs", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--sha-low", action="store_true", help="use float32 on CPU (no CUDA here)")
    args = ap.parse_args()

    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    torch.manual_seed(0)
    print("loading processor/model ...", flush=True)
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(LOCAL_DIR, local_files_only=True)
    kwargs = dict(local_files_only=True, torch_dtype=torch.float32)
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(LOCAL_DIR, **kwargs)
    except TypeError:
        model = Qwen2VLForConditionalGeneration.from_pretrained(LOCAL_DIR, local_files_only=True)
    model.eval()
    print("model loaded in %.1fs, device=cpu, dtype=float32" % (time.time() - t0), flush=True)

    results = []
    for n_ref in args.n_refs:
        for defect in (False, True):
            inputs, refs, qpath = build_inputs(processor, n_ref, "circle", defect)
            with torch.no_grad():
                t1 = time.time()
                out = model.generate(**inputs, max_new_tokens=256, do_sample=False, pad_token_id=processor.tokenizer.eos_token_id)
                latency = (time.time() - t1) * 1000
            raw = processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
            # parse: try strict json, then fenced block, then first {...}
            parsed, err = None, None
            try:
                parsed = json.loads(raw.strip())
            except Exception as e1:
                try:
                    seg = raw[raw.index("{"): raw.rindex("}") + 1]
                    parsed = json.loads(seg)
                except Exception as e2:
                    err = f"{type(e1).__name__}:{e1} | {type(e2).__name__}:{e2}"
            rec = {
                "n_ref": n_ref, "query_kind": "synthetic_defect" if defect else "synthetic_normal",
                "latency_ms": round(latency, 1), "status": "ok" if parsed else "json_fail",
                "parse_error": err, "raw_response": raw, "parsed": parsed,
                "anomaly_score": parsed.get("anomaly_score") if parsed else None,
                "decision": parsed.get("decision") if parsed else None,
                "reference_paths": refs, "query_path": qpath,
                "query_sha256": hashlib.sha256(open(qpath, "rb").read()).hexdigest(),
                "input_token_count": int(inputs.input_ids.shape[1]),
            }
            results.append(rec)
            print(f"n_ref={n_ref} defect={defect} latency={latency:.0f}ms status={rec['status']} score={rec['anomaly_score']} decision={rec['decision']}", flush=True)
            print("  raw:", raw[:300].replace("\n", " "), flush=True)

    report = {
        "model_id": MODEL_ID, "model_dir": LOCAL_DIR, "device": "cpu", "dtype": "float32",
        "prompt_sha256": hashlib.sha256(PROMPT_FILE.read_bytes()).hexdigest(),
        "schema_sha256": hashlib.sha256(SCHEMA_FILE.read_bytes()).hexdigest(),
        "transformers": __import__("transformers").__version__, "torch": torch.__version__,
        "json_valid_rate": sum(r["status"] == "ok" for r in results) / len(results),
        "mean_latency_ms": sum(r["latency_ms"] for r in results) / len(results),
        "records": results,
    }
    (OUT / "smoke_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("WROTE", OUT / "smoke_report.json", flush=True)


if __name__ == "__main__":
    main()
