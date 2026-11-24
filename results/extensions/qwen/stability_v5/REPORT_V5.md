# Qwen2-VL fc_conf — Reference-Set Stability (v5) Report

Protocol: `qwen_stability_v5.md` (frozen before inference, commit `6a6a2b6`).
All numbers below refer to the **696 held-out VisA test images of v4**
re-scored under different reference sets. This is a stability check, **not**
an independent blind test (the images were already used in v4).

## 1. Setup

| Item | Value |
|---|---|
| Model | Qwen2-VL-7B-Instruct, bf16, A6000 |
| Score | `fc_conf = ΣlogP(C_OK) − ΣlogP(C_NG)` (frozen; higher = more anomalous) |
| Readout | forced-choice only (JSON readout known-degenerate since v1, not re-run) |
| Test images | 696, query IDs identical to v4 (`manifest_v4_test.json`) |
| Calibration | 120 normal images (20/class, same v4 set); per-class p95 thresholds, secondary metric only |
| References | archived `results/part2/refs/1shot_seed{S}/` — no resampling; refs differ from seed 0 in 6/6 classes for both new seeds |
| Manifest hashes (sha256, frozen) | s1 test `3b8e4538…`, s1 calib `174c38c2…`, s2 test `3bebc77a…`, s2 calib `3d90cf8c…` |
| Failures | s1: 0/696, s2: 0/696 (all 1392 records valid) |

Seed 0 row = v4 run, re-evaluated with the identical script for paired comparisons.

## 2. Main result — ranking stability across reference seeds

| Refs seed | Qwen fc_conf (95% CI) | DINOv2 @448 | CLIP L9 @448 | ΔQwen−DINO (paired 95% CI) | ΔQwen−CLIP (paired 95% CI) |
|---|---|---|---|---|---|
| 0 | 0.9575 [0.9426, 0.9710] | 0.9313 | 0.8530 | +0.0262 [+0.0028, +0.0501] | +0.1045 [+0.0732, +0.1377] |
| 1 | 0.9477 [0.9310, 0.9628] | 0.9259 | 0.8771 | +0.0218 [−0.0016, +0.0457] | +0.0705 [+0.0434, +0.0980] |
| 2 | 0.9543 [0.9387, 0.9679] | 0.9417 | 0.8798 | +0.0127 [−0.0074, +0.0329] | +0.0745 [+0.0478, +0.1019] |
| **across seeds** | **0.9532 ± 0.0050** (range 0.0098) | 0.9330 ± 0.0080 | 0.8700 ± 0.0148 | | |

- Qwen macro AUROC varies by **< 0.01** across the three reference sets;
  its variation is smaller than DINOv2's (0.016) and CLIP's (0.027).
- Paired Δ vs DINOv2: positive at all three seeds but the CI includes 0 at
  seeds 1 and 2 → **do not claim Qwen beats DINOv2**; the honest statement is
  "comparable, with a small positive point estimate".
- Paired Δ vs CLIP: CI excludes 0 at all three seeds → Qwen ranks
  consistently above CLIP on these images.

## 3. Per-class stability (Qwen fc_conf, mean ± std over seeds)

| Class | AUROC (mean ± std) | range | p95-threshold FPR / recall (seeds 0/1/2) |
|---|---|---|---|
| candle | 0.9444 ± 0.0054 | 0.0104 | FPR 0.141/0.167/0.064, recall 0.923/0.910/0.885 |
| capsules | 0.9164 ± 0.0032 | 0.0061 | FPR 0.053/0.105/0.158, recall 0.846/0.808/0.808 |
| cashew | 0.9307 ± 0.0012 | 0.0023 | FPR 0.000/0.000/0.000, recall **0.256/0.333/0.269** |
| chewinggum | 0.9803 ± 0.0058 | 0.0114 | FPR 0.000/0.000/0.000, recall 0.859/0.846/0.872 |
| fryum | 0.9628 ± 0.0077 | 0.0142 | FPR 0.000/0.000/0.000, recall 0.859/0.808/0.859 |
| pipe_fryum | 0.9843 ± 0.0137 | 0.0238 | FPR 0.036/0.071/0.286, recall 0.949/0.885/0.987 |

- Class ranking is stable: chewinggum / pipe_fryum / fryum strong,
  capsules weakest (0.914–0.920) — the same ordering in all three seeds.
- **Threshold behaviour is not stable even though AUROC is**: cashew recall
  stays low (≈0.26–0.33) at every seed because its calibration distribution
  is heavy-tailed (the p95 threshold sits ~4–8 above the other classes).
  FPR at pipe_fryum jumps to 0.286 at seed 2. These are properties of the
  frozen p95 rule, not of the ranking score; they reproduce v4's warning
  that AUROC and threshold metrics must be reported separately.

## 4. Cost

| Seed | fc latency mean / p95 | 696-image wall time | peak GPU |
|---|---|---|---|
| 0 (v4) | 2636 / — ms | 3851 s (incl. JSON pass) | 23.5 GB |
| 1 | 4292 / 6588 ms | 3328 s | 23.5 GB |
| 2 | 5850 / 6963 ms | 4317 s | 23.5 GB |

Latency differences track GPU contention from other users on the shared
A6000, not the reference set (identical computation).

## 5. What this does and does not establish

Establishes:
- fc_conf's ranking performance on these 696 images is **not an artifact of
  the seed-0 reference images**: macro AUROC stays within 0.948–0.958 across
  three independent 1-shot reference sets.
- The model-vs-baseline ordering (Qwen ≈ DINOv2 > CLIP) is stable across
  reference seeds.

Does not establish:
- Independent confirmation: the same 696 images selected nothing here, but
  they were already evaluated in v4, and the fc_conf direction itself was
  chosen after seeing v2. Only a new dataset (or a held-out split never
  touched during method selection) can provide a blind number.
- Any shot/seed beyond 1-shot × {0,1,2}.
- Anything about thresholded deployment performance beyond the frozen p95
  rule's behaviour.

## 6. Next step (gated, per protocol §3.3)

Stability holds → the pre-declared 2/4/8-shot × seed 0/1/2 matrix (11 configs,
≈ 12 h GPU at these latencies) is now justified. Recommended order:
2-shot seed 0 first (answers "does one more reference help or hurt?"), then
decide on the rest. No fusion work until that matrix exists.

---

## 中文核对要点

- 本轮只换了参考图（seed 1/2，归档 refs，6/6 类与 seed0 不同），测试图与 v4 完全相同（696 张），所以结论是"稳定性"而非"独立验证"。
- 三个 seed 宏 AUROC：0.9575 / 0.9477 / 0.9543，波动 <0.01，比 DINO（0.016）、CLIP（0.027）都小。
- 与 DINO 的配对差三次都为正但 seed1/2 的 CI 含 0 → 不能写"超过 DINO"；与 CLIP 三次 CI 都排除 0 → 可写"稳定高于 CLIP"。
- cashew 召回低（0.26–0.33）在三 seed 全部复现，属 p95 阈值规则对重尾校准分布敏感；pipe_fryum 在 seed2 FPR 0.286 也属阈值规则问题，不是排序分数退化。
- 失败 0/1392；期间踩过并发进程互踩的坑（多次 SIGTERM 的启动实际都活了），已清干净重跑，最终数据来自单进程运行。
- 下一步按协议门槛：2/4/8-shot × seed 0/1/2 共 11 个配置，约 12 小时 GPU，建议先跑 2-shot seed 0。
