"""Per-image inference latency and peak GPU memory for the frozen kNN pipeline.

Why this exists (Part 3, brief section 6.2)
-------------------------------------------
The paper must report, for each system, the *average inference time per image* and
the *peak GPU memory on one device*, and must state whether feature extraction is
included. Until now the only timing on record was grid wall-clock (which mixes
reference-feature extraction, the 12 (shot, seed) cells and evaluator overhead), so
it cannot be quoted as a per-image number.

What is measured
----------------
The deployed path for one test image, at batch size 1, with everything the grid
runner does per query image:

    decode (cv2) -> prepare_image (resize + normalise + H2D) -> backbone forward
    (patch tokens, fp32, returned to CPU) -> faiss 1-NN over the reference bank
    -> top-1% mean pooling -> (optional) distance-map resize for pixel AUROC

Feature extraction IS included; the reference bank is built once outside the timed
loop and is therefore amortised (it is a per-category setup cost, not a per-image
cost). Both facts are recorded in the output so the paper can state the boundary.

Peak memory comes from the CUDA allocator: `max_memory_allocated` (tensors actually
in use) and `max_memory_reserved` (allocator footprint). The delta over the resident
model weights is what a second concurrent model would have to fit into.

Reproducibility notes
---------------------
- OMP/MKL threads are pinned to 1 *before* faiss is imported, exactly as in
  run_grid.py: the small k-NN searches there are synchronisation-bound (462 ms with
  12 threads vs 87 ms with 1), so the reported latency matches the configuration
  that produced the reported AUROC numbers.
- The feature cache is bypassed on purpose: a cache hit would measure disk I/O.
- Warm-up iterations run first (CUDA context, cuBLAS heuristics, allocator growth)
  and are reported separately, never folded into the mean.

Usage on the shared A6000
-------------------------
    PY=/tmp/slv176/venv/bin/python
    $PY scripts/bench_inference.py --backbone dinov2 --layers 12 --out /tmp/slv176/bench
    $PY scripts/bench_inference.py --backbone clip   --layers 9 --clip_resize_mode short_edge \
        --out /tmp/slv176/bench
    $PY scripts/bench_inference.py --report /tmp/slv176/bench

The last call merges every bench_*.json in that directory into bench_report.md,
which holds the table to paste into the paper. It needs no GPU and no torch.
"""

import argparse
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO = os.path.dirname(_ROOT)


# ------------------------------------------------------------------ pure helpers
def stats(values_ms):
    """Mean/median/p95 with the sample count kept, so the paper can quote n."""
    v = sorted(float(x) for x in values_ms)
    n = len(v)
    return {"n": n,
            "mean": statistics.fmean(v),
            "median": statistics.median(v),
            "std": statistics.pstdev(v) if n > 1 else 0.0,
            "p05": v[int(round(0.05 * (n - 1)))],
            "p95": v[int(round(0.95 * (n - 1)))],
            "min": v[0], "max": v[-1]}


def _cpu_name():
    if platform.system() == "Linux":
        try:
            txt = subprocess.run(["lscpu"], capture_output=True, text=True,
                                 timeout=5).stdout
            return txt.split("Model name:")[1].split("\n")[0].strip()
        except Exception:
            return platform.processor()
    return platform.processor()


def env_info(torch, device):
    info = {"host": socket.gethostname(), "platform": platform.platform(),
            "python": sys.version.split()[0], "torch": torch.__version__,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "batch_size": 1, "cpu": _cpu_name(),
            "threads": {k: os.environ.get(k) for k in
                        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
            "torch_num_threads": torch.get_num_threads()}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(device)
        info["gpu"] = {"name": p.name, "total_mib": p.total_memory / 2 ** 20,
                       "sm": f"{p.major}.{p.minor}",
                       "count": torch.cuda.device_count()}
    return info


# ----------------------------------------------------------------------- runner
def run(args):
    # Pinned before faiss is imported, exactly as in run_grid.py.
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")

    import cv2
    import faiss
    import numpy as np
    import torch
    from scipy.ndimage import gaussian_filter

    sys.path.insert(0, os.path.join(_ROOT, "src"))
    import p2common as C
    from ad_core import (LayerDINOv2, extract_image, make_index, mean_top1p,
                         patch_distances)
    from clip_core import LayerCLIP

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available on this host")

    torch.set_num_threads(1)
    os.makedirs(args.out, exist_ok=True)

    def sync():
        if args.device == "cuda":
            torch.cuda.synchronize()

    def mib():
        return {"allocated": torch.cuda.memory_allocated() / 2 ** 20,
                "reserved": torch.cuda.memory_reserved() / 2 ** 20,
                "peak_allocated": torch.cuda.max_memory_allocated() / 2 ** 20,
                "peak_reserved": torch.cuda.max_memory_reserved() / 2 ** 20}

    def resize_map(arr, size):
        """Same resize + smoothing the grid runner applies before pixel AUROC."""
        a = cv2.resize(arr.astype(np.float32), (size, size),
                       interpolation=cv2.INTER_LINEAR)
        return gaussian_filter(a, sigma=4)

    cats = args.categories or list(C.VISA_6)
    images = []
    for cat in cats:
        items = C.test_items(args.data_root, cat)
        step = max(1, len(items) // args.n_per_cat)
        images += [(cat, items[i][1]) for i in range(0, len(items), step)][:args.n_per_cat]
    if not images:
        raise SystemExit(f"no test images under {args.data_root}")

    # ---- model (load time reported separately; not part of per-image cost)
    t0 = time.perf_counter()
    if args.backbone == "clip":
        model = LayerCLIP(args.model, args.device, args.resolution, tuple(args.layers),
                          pos_embedding_mode=args.pos_embedding_mode,
                          download_root=args.clip_download_root,
                          resize_mode=args.clip_resize_mode)
        n_params = sum(p.numel() for p in model.visual.parameters())
    else:
        model = LayerDINOv2(args.model, args.device, args.resolution,
                            tuple(args.layers))
        n_params = sum(p.numel() for p in model.model.parameters())
    sync()
    load_s = time.perf_counter() - t0
    resident = mib() if args.device == "cuda" else None

    # ---- reference bank: 1-shot, `ref_views` rotations, unmasked (amortised cost)
    pool = C.ref_pool(args.data_root, cats[0], 20)[1]
    ref_path = os.path.join(C.train_dir(args.data_root, cats[0]), pool[0])
    ref_feats, ref_grid, ref_masks = extract_image(model, ref_path,
                                                   rotations=args.ref_views,
                                                   mask_query=False)
    bank = np.concatenate([f[-1][m] for f, m in zip(ref_feats, ref_masks)],
                          axis=0).astype("float32")
    faiss.normalize_L2(bank)
    index = make_index(bank)
    bank_patches = int(bank.shape[0])
    del bank, ref_feats, ref_masks

    stages = {k: [] for k in ("decode", "preprocess", "forward", "knn", "pooling",
                              "pixel_map", "end_to_end")}
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # ---- warm-up (untimed: CUDA context, kernels, allocator growth)
    warm_ms = []
    for _, path in images[:args.warmup]:
        t0 = time.perf_counter()
        img = C.load_rgb(path)
        tensor, _ = model.prepare_image(img)
        tensor = tensor.to(args.device, non_blocking=False)
        feats = model.extract_layers(tensor)
        patch_distances(index, feats[-1], np.ones(feats[-1].shape[0], dtype=bool))
        sync()
        warm_ms.append((time.perf_counter() - t0) * 1e3)

    # ---- timed passes
    for _ in range(args.passes):
        for _, path in images:
            t0 = time.perf_counter()
            img = C.load_rgb(path)
            t1 = time.perf_counter()
            tensor, grid = model.prepare_image(img)
            # DINOv2Wrapper defers .to(device) to the forward call while LayerCLIP does it
            # inside prepare_image; upload here so both backbones share one boundary
            # (the second .to() is a no-op for CLIP).
            tensor = tensor.to(args.device, non_blocking=False)
            sync()
            t2 = time.perf_counter()
            feats = model.extract_layers(tensor)
            sync()
            t3 = time.perf_counter()
            mask = np.ones(feats[-1].shape[0], dtype=bool)
            d = patch_distances(index, feats[-1], mask)
            t4 = time.perf_counter()
            mean_top1p(d)  # the scalar the detector reports, kept inside the timed path
            t5 = time.perf_counter()
            if args.with_pixel:
                resize_map(d.reshape(grid), C.EVAL_SIZE)
            sync()
            t6 = time.perf_counter()
            stages["decode"].append((t1 - t0) * 1e3)
            stages["preprocess"].append((t2 - t1) * 1e3)
            stages["forward"].append((t3 - t2) * 1e3)
            stages["knn"].append((t4 - t3) * 1e3)
            stages["pooling"].append((t5 - t4) * 1e3)
            stages["pixel_map"].append((t6 - t5) * 1e3)
            stages["end_to_end"].append((t6 - t0) * 1e3)

    peak = mib() if args.device == "cuda" else None
    patch_grid = [int(x) for x in grid]

    result = {
        "env": env_info(torch, args.device),
        "config": {
            "backbone": args.backbone, "model": args.model,
            "resolution": args.resolution, "layers": list(args.layers),
            "clip_resize_mode": args.clip_resize_mode if args.backbone == "clip" else None,
            "pos_embedding_mode": args.pos_embedding_mode if args.backbone == "clip" else None,
            "ref_views": args.ref_views, "categories": cats,
            "n_per_cat": args.n_per_cat, "passes": args.passes, "warmup": args.warmup,
            "with_pixel_map": args.with_pixel,
            "includes_feature_extraction": True,
            "bank_built_outside_timing": True,
            "reference_grid": [int(x) for x in ref_grid],
        },
        "model": {"params_visual_m": n_params / 1e6, "patch_grid": patch_grid,
                  "patch_tokens": patch_grid[0] * patch_grid[1], "load_s": load_s},
        "bank": {"patches": bank_patches, "ref_views": args.ref_views,
                 "reference_images": 1},
        "warmup_ms": warm_ms,
        "stages_ms": {k: stats(v) for k, v in stages.items()},
        "memory_mib": None if peak is None else {
            "weights_resident": resident["allocated"],
            "weights_resident_reserved": resident["reserved"],
            "peak_allocated": peak["peak_allocated"],
            "peak_reserved": peak["peak_reserved"],
            "activation_overhead": peak["peak_allocated"] - resident["allocated"],
        },
        "throughput_img_s": 1000.0 / statistics.fmean(stages["end_to_end"]),
    }

    tag = (f"{args.backbone}{args.resolution}_L{'-'.join(str(l) for l in args.layers)}"
           + (f"_{args.clip_resize_mode}" if args.backbone == "clip" else ""))
    path = os.path.join(args.out, f"bench_{tag}.json")
    C.jdump(result, path)

    e = result["stages_ms"]["end_to_end"]
    print(f"\n[{tag}] {e['n']} timed inferences, bank={bank_patches} patches, "
          f"grid={patch_grid[0]}x{patch_grid[1]}")
    for k in ("decode", "preprocess", "forward", "knn", "pooling", "pixel_map",
              "end_to_end"):
        s = result["stages_ms"][k]
        print(f"  {k:11s} mean={s['mean']:8.2f} ms  median={s['median']:8.2f}  "
              f"p95={s['p95']:8.2f}  sd={s['std']:6.2f}")
    if result["memory_mib"]:
        m = result["memory_mib"]
        print(f"  weights {m['weights_resident']:.0f} MiB | peak allocated "
              f"{m['peak_allocated']:.0f} MiB | peak reserved {m['peak_reserved']:.0f} MiB")
    print(f"  throughput {result['throughput_img_s']:.2f} img/s -> {path}")
    return result


# ----------------------------------------------------------------------- report
def report(out_dir):
    files = sorted(f for f in os.listdir(out_dir)
                   if f.startswith("bench_") and f.endswith(".json"))
    if not files:
        raise SystemExit(f"no bench_*.json in {out_dir}")
    rows = [json.load(open(os.path.join(out_dir, f), encoding="utf-8")) for f in files]

    pretty = {"dinov2_vits14": "ViT-S/14", "dinov2_vitb14": "ViT-B/14"}

    def name(r):
        c = r["config"]
        model = pretty.get(c["model"], c["model"])
        if c["backbone"] == "clip":
            return f"CLIP {model} @{c['resolution']}, block {c['layers'][-1]}"
        return f"DINOv2 {model} @{c['resolution']}, block {c['layers'][-1]}"

    hosts = {(r["env"].get("gpu", {}).get("name"), r["env"]["torch"]) for r in rows}
    if len(hosts) > 1:
        print("WARNING: runs came from different GPUs / torch builds:", hosts)

    lines = ["# Inference cost per image (measured, not estimated)", "",
             "Generated by `scripts/bench_inference.py --report`. Batch size 1; feature "
             "extraction included; the reference bank is built once and amortised, so its "
             "patch count is listed per row.", "",
             "## Paper table", "",
             "| Encoder | Patch grid | Bank patches | Latency / image (ms) | Throughput "
             "(img/s) | Peak GPU mem (MiB) | Vision params (M) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        s = r["stages_ms"]["end_to_end"]
        m = r["memory_mib"] or {}
        pa = m.get("peak_allocated")
        lines.append(
            f"| {name(r)} | {r['model']['patch_grid'][0]}x{r['model']['patch_grid'][1]} "
            f"| {r['bank']['patches']} | {s['mean']:.1f} +/- {s['std']:.1f} "
            f"(median {s['median']:.1f}) | {r['throughput_img_s']:.2f} "
            f"| {pa:.0f} | {r['model']['params_visual_m']:.1f} |"
            if pa is not None else
            f"| {name(r)} | {r['model']['patch_grid'][0]}x{r['model']['patch_grid'][1]} "
            f"| {r['bank']['patches']} | {s['mean']:.1f} +/- {s['std']:.1f} | "
            f"{r['throughput_img_s']:.2f} | n/a (CPU) | "
            f"{r['model']['params_visual_m']:.1f} |")

    lines += ["", "## Stage breakdown (ms; share is of the end-to-end mean)", ""]
    for r in rows:
        e2e = r["stages_ms"]["end_to_end"]["mean"]
        lines += [f"**{name(r)}**", "",
                  "| stage | mean | median | p95 | share |", "|---|---:|---:|---:|---:|"]
        for k in ("decode", "preprocess", "forward", "knn", "pooling", "pixel_map",
                  "end_to_end"):
            s = r["stages_ms"][k]
            lines.append(f"| {k} | {s['mean']:.2f} | {s['median']:.2f} | {s['p95']:.2f} "
                         f"| {s['mean'] / e2e * 100:.1f}% |")
        lines.append("")

    r0 = rows[0]
    m0 = r0["memory_mib"] or {}
    lines += ["## Conditions", "",
              f"- Host `{r0['env']['host']}`, {r0['env'].get('gpu', {}).get('name', 'CPU')}, "
              f"torch {r0['env']['torch']} / CUDA {r0['env']['cuda']}",
              f"- {r0['config']['passes']} timed passes over "
              f"{r0['stages_ms']['end_to_end']['n'] // max(r0['config']['passes'], 1)} images, "
              f"{r0['config']['warmup']} warm-up iterations discarded",
              "- faiss/MKL threads pinned to 1, torch.set_num_threads(1), batch size 1",
              f"- CUDA allocator: weights {m0.get('weights_resident', float('nan')):.0f} MiB, "
              f"peak allocated {m0.get('peak_allocated', float('nan')):.0f} MiB, "
              f"peak reserved {m0.get('peak_reserved', float('nan')):.0f} MiB",
              "", "**Scope.** One test image end to end: decode, resize/normalise, upload, "
              "backbone forward, 1-NN over the reference bank, top-1% pooling, and the "
              "distance-map resize used for pixel AUROC. Excludes the one-off reference-bank "
              "construction (K references x 8 rotations) and excludes the VLM branch, which "
              "its owner times separately.", ""]
    path = os.path.join(out_dir, "bench_report.md")
    open(path, "w", encoding="utf-8").write("\n".join(lines))
    print(f"wrote {path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="dinov2", choices=["dinov2", "clip"])
    p.add_argument("--model", default=None)
    p.add_argument("--resolution", type=int, default=448)
    p.add_argument("--layers", nargs="+", type=int, default=[12])
    p.add_argument("--pos_embedding_mode", default="interp", choices=["native", "interp"])
    p.add_argument("--clip_download_root", default="/data/disk1/home/slv176/models/clip")
    p.add_argument("--clip_resize_mode", default="short_edge",
                   choices=["short_edge_crop", "squash", "short_edge"])
    p.add_argument("--data_root", default="/tmp/slv176/data/VisA_pytorch/1cls")
    p.add_argument("--categories", nargs="*", default=None)
    p.add_argument("--n_per_cat", type=int, default=30)
    p.add_argument("--ref_views", type=int, default=8,
                   help="rotations per reference image in the 1-shot bank")
    p.add_argument("--passes", type=int, default=2)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--with_pixel", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="/tmp/slv176/bench")
    p.add_argument("--report", default=None,
                   help="merge every bench_*.json in this directory into bench_report.md")
    args = p.parse_args()
    if args.model is None:
        args.model = "ViT-B/16" if args.backbone == "clip" else "dinov2_vits14"
    return args


if __name__ == "__main__":
    a = parse_args()
    if a.report:
        report(a.report)
    else:
        run(a)
