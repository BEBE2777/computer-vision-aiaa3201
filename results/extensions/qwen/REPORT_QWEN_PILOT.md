# Qwen-VL Multimodal Pilot — Report-Ready Text (B line)

Purpose: drop-in paragraphs for the CVPR-format report. Scope is deliberately
limited to the 24-image pilot. Every claim below is supported by the saved
raw responses in `smoke/gpu_smoke_report.json` (2B) and
`smoke/gpu_smoke_7b_report.json` (7B).

> Usage note (not part of the report): treat this as a **pilot / negative
> result**, NOT a fullVisA evaluation. Do not rank it against full-test-set
> DINO/CLIP numbers. If a comparison table is needed, add a matched 24-image
> DINO/CLIP row, or keep Qwen in a separate "pilot" row/footnote. AUROC is
> **undefined** (zero score variance), never report 0.5.

---

## Paragraph 1 — Method and frozen input protocol

We additionally probed a vision-language alternative to embedding-based
few-shot detectors: a multimodal large language model asked to inspect a query
image against a small set of normal references. To prevent label leakage, each
query was presented in an **independent context** containing only (i) the text
category name, (ii) the *n* normal reference images for that cell (taken
verbatim from the shared reference pool used by the DINO baseline), (iii) the
single query image, and (iv) a fixed instruction prompt. No good/bad paths,
 filenames, ground-truth labels, anomaly masks, heatmaps, or DINO scores were
ever shown to the model; the true label was stored separately and used only
for offline scoring. The prompt, output JSON schema (fields `decision`,
`anomaly_score`, `confidence`, `defect_type`, `evidence`, `regions`), greedy
decoding, and the reference images were frozen before evaluation and held
constant across models.

## Paragraph 2 — Pilot setup

Because a full 1/2/4/8-shot × 3-seed × 960-image grid (11,520 queries) is
costly, we first ran a **24-image pilot**: one shot (`n = 1`), seed 0, two
normal and two anomalous queries drawn from each of the six VisA categories
(2 × 2 × 6 = 24). Both `Qwen2-VL-2B-Instruct` and `Qwen2-VL-7B-Instruct` were
evaluated on exactly this pilot, in bfloat16 on a single RTX A6000 (48 GB).
The 7B checkpoint was a one-time, explicitly documented revision to the
frozen protocol — the only changed item; prompt, schema, decoding, references
 and evaluation scope were unchanged.

## Paragraph 3 — Observations

On this 24-image pilot, both checkpoints produced well-formed JSON (100%
parse rate) but failed to separate normal from anomalous images. For the 2B
model, `anomaly_score` was identically 0.0 on all 24 queries (23 `OK`, 1 `NG`),
and several anomalous images were described as "identical to the normal
reference images, with no visible defects." For the 7B model, `anomaly_score`
was again identically 0.0 on all 24 queries, all decisions were `OK`, and the
24 responses collapsed to only two distinct strings (differing solely in
`confidence` 0 vs 1, with empty `evidence`). Because the predicted scores have
zero variance, an image-level AUROC is **undefined** on this pilot; we
therefore do not report one. Table 1 summarises the pilot.

**Table 1.** Qwen-VL pilot results (24 images, 1 shot, seed 0). AUROC is
undefined because scores are constant.

| Model | JSON valid | anomaly_score range | OK / NG | Mean latency | Peak GPU mem |
|---|---|---|---|---|---|
| Qwen2-VL-2B | 100% | {0.0} | 23 / 1 | 4.13 s/img | 5.2 GB |
| Qwen2-VL-7B | 100% | {0.0} | 24 / 0 | 2.69 s/img | 17.7 GB |

## Paragraph 4 — Sanity checks and limitations

We verified this is a model behaviour and not a pipeline defect: the two
checkpoints consume identical input token counts (e.g., 4257 tokens on the
same first image), share the same image-preprocessing configuration, and the
chat template contains the expected vision tokens; an initial template-loading
error was traced to one missing config file and fixed before the reported run.
**These observations are limited to a 24-image, single-shot pilot and do not
establish that Qwen-VL lacks anomaly-detection ability across VisA as a
whole.** Given that the pilot showed no discriminative signal and that a full
grid would add little information relative to its cost, we did not run the
11,520-query grid and do not tune the prompt to chase scores. We therefore
report Qwen-VL as a multimodal pilot negative result and base the main
comparison on the controlled DINO/CLIP embedding-based experiments.

---

对应中文要点（写报告时核对用）：
- 范围：24 张 pilot，1-shot，seed 0，六类各 2 正常 + 2 异常；**不是全 VisA**。
- 2B：score 全 0，23 OK / 1 NG，bad 被描述成"与正常图一致"。
- 7B：score 全 0，24/24 OK，响应仅 2 种（evidence 空）。
- AUROC **无定义**（分数无方差），**不要写 0.5**。
- 已排除管线 bug（token 数一致、预处理一致、chat template 正确）。
- 不外推："pilot 上未能区分" ≠ "整个 VisA 上没有能力"。
- 不与完整 CLIP/DINO 数字直接排名；如需进主表，须补同 24 张的对照行。
