# Qwen fc_conf Reference-Set Stability Pilot (v5)

**Status: FROZEN before any v5 inference.** Commit at freeze: `8c31bce`.
Date frozen: 2026-11-24.

## 1. Purpose and scope (non-blind, stated up front)

v4 measured fc_conf on 696 held-out VisA test images with **1-shot, seed 0**
references (macro AUROC 0.958). The purpose of v5 is to test whether that
result depends on the specific seed-0 reference images.

**Boundary condition (explicit):** the 696 test images have already been
looked at (v4). v5 is therefore a **reference-set stability check**, NOT a
new independent/blind confirmation. Claims from v5 are limited to:
"fc_conf ranking is stable/unstable under reference-image changes on the
same 696 images." A truly independent confirmation would require data never
used for method or weight selection.

## 2. Frozen configuration

| Item | Value |
|---|---|
| Model | Qwen2-VL-7B-Instruct, local weights, bf16 (same checkpoint as v4) |
| Prompt | `prompt_v1.txt` (sha256 `596118ac…`, unchanged since v1) |
| Candidates | C_OK / C_NG identical to v4 (sha256 recorded in runner metadata) |
| Image processing | qwen_vl_utils + processor, identical to v4 |
| Score | `fc_score = ΣlogP(C_NG) − ΣlogP(C_OK)`; reported as `fc_conf = −fc_score` (higher = more anomalous). Direction frozen; never revisited. |
| Readout | forced-choice ONLY. The free-generation JSON readout is known-degenerate (v1/v2/v3/v4) and is NOT re-run. |
| Test set | the exact 696 query IDs of `manifest_v4_test.json` (no resampling) |
| Calibration set | the exact 120 query IDs of `manifest_v4_calib.json`; used only for per-class p95 threshold reporting (pre-fixed rule, same as v4); never enters model context as labels |
| Reference images | archived refs ONLY: `results/part2/refs/1shot_seed{S}/<cat>_shot=1_seed={S}.json` → `reference_files`; no re-sampling, no re-selection |
| Label isolation | labels/masks/filename truth cues/DINO/CLIP scores never sent to the model; `true_label_for_eval_ONLY` untouched by the runner |

## 3. Run plan

1. Configs: **1-shot seed 1** and **1-shot seed 2**, both fully executed
   regardless of the seed-1 outcome. No prompt/score/direction changes
   between runs.
2. Per-seed smoke (before each full run): ≥1 normal + ≥1 anomalous query per
   category; checks: all six categories load, refs match the archived
   seed's `reference_files`, fc_conf finite and non-constant, record
   query_ids match the manifest, no labels in model input path.
3. 2/4/8-shot × seed 0/1/2 (11 further configs) are **pre-declared but
   gated**: run only after the two 1-shot seeds are complete and reviewed.
   v4 covers 1-shot seed 0.
4. Resume rule: interrupted runs merge by manifest `query_id` only; failed
   or missing samples are recorded as failures, never silently dropped.

## 4. Metrics (all reported together, nothing selected post hoc)

- Per-class Image AUROC + six-class macro, per seed.
- Seed-wise mean / std / range of macro AUROC (seeds 0/1/2, seed 0 = v4).
- Per-class n_good / n_bad, failure and missing IDs.
- Per-image fc latency, total wall time, peak GPU memory.
- Matched DINO / CLIP (same images, same shot, same seed, archived refs)
  from `p2-anomaly/results/part3/scores/e1_{dino448,clip448_L9}_nomask_s1_seed{S}.json`;
  paired per-image comparison on identical images. If a config has no
  matching dump, it is reported as "no matched comparison" — never ranked
  against a different subset.
- Secondary: per-class FPR / recall at the frozen calib-p95 threshold,
  per seed. Threshold behaviour is reported separately from AUROC; the
  v4 cashew case shows these must not be conflated.

## 5. Known caveat carried from v4

The two candidates change BOTH `decision` and `anomaly_score` fields, so
fc_conf reflects preference over the whole JSON string, not only the OK/NG
verdict. A "decision-only" ablation is a separate future protocol; its
results must not be merged into v4/v5 numbers.

## 6. Out of scope for v5

No fusion, no fusion-weight selection, no prompt changes, no direction
changes, no other shots/seeds beyond the plan above, no absolute claims
about full-VisA performance beyond the tested images.
