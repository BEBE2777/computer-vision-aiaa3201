# Experiments — visual branch (A's draft)

> Status (2026-11-22): draft of the report's Experiments section, A's half (setup recap,
> main tables, per-category analysis, error taxonomy, visual-pair fusion pilot, cost).
> Every number is copied from the machine-generated authority files — `p2-anomaly/results/
> part3/tables.md` (from `make_tables.py`) and `e1_full.md` — and cross-checked against
> `metrics.json`. Fusion numbers come from `results/part3/fusion/fusion_visual_pair_1shot_seed0.md`.
> B's Qwen subsection slots in after §5.4; TODO markers show where.

## 5.1 Setup

We evaluate on six VisA categories (candle, capsules, cashew, chewinggum, fryum, pipe_fryum)
with 1/2/4/8 normal references and seeds 0/1/2. The same 960 test images are scored by every
method; references for a given (shot, seed) are `pool[s·shot:(s+1)·shot]` of the sorted
normal training pool, and the first 20 normal training images per category are held out as a
calibration set that never enters a memory bank or a prompt. Image AUROC is the primary
metric, reported per category and as the unweighted six-category mean; mean ± std is taken
over the three per-seed class means. Pixel AUROC uses maps resampled to 448×448 for all
methods (the shared-GPU memory limit forbids pooling native-resolution pixels; the resampling
is identical everywhere, so comparisons stay well defined). OK/NG thresholds for the
qualitative analysis are the 95th percentile of each method's own 20 calibration scores.
Full protocol details are in Method §2–4; this section reports what the protocol produces.

Four systems are compared. **Part 1** is the template-difference baseline (ECC-registered
RGB/SSIM difference, closest template). **Part 2** is the official AnomalyDINO
[damm2025anomalydino] reproduction (DINOv2 ViT-S/14 @448, agnostic preprocessing, PCA
foreground mask), verified cell-by-cell against the published pipeline
(`results/part2_official/parity_report.md`; all 72 cells agree to the last printed digit).
**E1 control** is DINOv2 @448 with the mask disabled — the identical-scoring counterpart of
our CLIP arm. **E1 CLIP** is CLIP ViT-B/16 @448, block 9, `short_edge` framing, no mask,
with patch tokens extracted, position-embedding interpolated and probe-verified as described
in Method §4. The controlled pair (control vs CLIP) differs *only* in the encoder; Part 2
differs additionally in preprocessing (its mask is worth ≈1.3 image points, measured as
official minus control) and is the complete-system reference, not a controlled arm.

## 5.2 Main results

**Table 1 — image-level AUROC (%), mean ± std over seeds.**

| Method | 1-shot | 2-shot | 4-shot | 8-shot |
|---|---|---|---|---|
| Template difference (Part 1) | 52.00 ± 1.13 | 54.08 ± 3.38 | 55.36 ± 3.22 | 57.96 ± 0.47 |
| AnomalyDINO (Part 2, official, masked) | **94.61 ± 0.84** | **95.33 ± 0.46** | **96.15 ± 0.40** | **96.62 ± 0.57** |
| DINOv2 @448, no mask (E1 control) | 93.34 ± 0.72 | 94.16 ± 0.27 | 94.92 ± 0.49 | 95.37 ± 0.72 |
| CLIP @448 block 9, no mask (E1) | 87.70 ± 1.24 | 89.22 ± 0.29 | 89.27 ± 0.38 | 89.48 ± 0.23 |

**Table 2 — pixel-level AUROC (%), same aggregation.**

| Method | 1-shot | 2-shot | 4-shot | 8-shot |
|---|---|---|---|---|
| Template difference (Part 1) | 65.06 ± 0.41 | 64.85 ± 0.39 | 65.47 ± 0.29 | 66.14 ± 0.22 |
| AnomalyDINO (Part 2, official, masked) | **98.08 ± 0.05** | **98.22 ± 0.07** | **98.36 ± 0.03** | **98.47 ± 0.01** |
| DINOv2 @448, no mask (E1 control) | 97.06 ± 0.32 | 97.57 ± 0.11 | 97.91 ± 0.06 | 98.16 ± 0.04 |
| CLIP @448 block 9, no mask (E1) | 94.44 ± 0.10 | 94.97 ± 0.09 | 95.22 ± 0.06 | 95.46 ± 0.05 |

Read-out. Under the controlled protocol, replacing DINOv2 with CLIP costs **5.6 image points
at 1-shot** (87.70 vs 93.34) and 2.6 pixel points, and the deficit **does not close with more
references**: CLIP gains +1.8 image points from 1-shot to 8-shot, the DINOv2 control gains
+2.0, and the ordering never flips. The few-shot deficit is therefore a property of the
features, not of sample size. The official masked baseline sits another ≈1.3 image points
above the unmasked control at every shot count, which prices the PCA foreground mask and
explains why it was removed from both controlled arms rather than granted to one.

**Figure references:** Fig. 2 (`figs/auroc_vs_shots.png`) shows all four lines against shots;
Fig. 3 (`figs/per_category_1shot.png`, produced by `scripts/plot_per_category.py`) shows the
per-category bars and the per-category gap across shots.

## 5.3 Per-category analysis

**Table 3 — Δ = CLIP − DINOv2, image AUROC points (mean over seeds).**

| category | 1-shot | 2-shot | 4-shot | 8-shot |
|---|---:|---:|---:|---:|
| candle | −4.7 | −2.3 | −3.4 | −4.0 |
| capsules | −15.1 | −16.2 | −16.4 | −17.3 |
| cashew | −4.2 | −5.4 | −8.7 | −11.0 |
| chewinggum | −2.6 | −1.7 | −2.0 | −2.2 |
| fryum | −10.3 | −4.7 | −3.2 | −0.7 |
| pipe_fryum | **+3.2** | +0.6 | −0.3 | −0.2 |

Three regimes are visible (Fig. 3). *Persistent, large deficits* — capsules starts 15.1
behind and *widens* to 17.3; its defects are sub-patch-scale colour/end-cap anomalies where
CLIP's 16-px, language-aligned tokens keep firing on normal background variation (see §5.5).
*Diverging trends* — fryum starts 10.3 behind but converges to −0.7 at 8-shot, the only
category where more references nearly close the gap. *A reversal* — pipe_fryum is the one
category where CLIP is ahead at 1-shot (+3.2) and statistically tied at 4/8-shot, i.e. on
highly regular, near-periodic objects the globally aligned features are not strictly worse.
An aggregate-only table would hide all three; the reversal and the fryum pixel-tie (CLIP
matches the control's pixel AUROC there, +0.1, while trailing 10.3 on image AUROC) are the
quantitative footing for the complementarity claim that the fusion experiments test.

## 5.4 Why CLIP trails: anisotropy and map variance

The mechanism was measured, not assumed (probes in `scripts/probe_feature_anisotropy.py`,
frozen in the pilot, `e1_pilot.md` §3). Patch features inside one image are far more
anisotropic in CLIP's last block — mean pairwise cosine similarity 0.57–0.67 against 0.36
for DINOv2 L12 — so nearest-neighbour *distances* compress unevenly across the image. The
layer sweep found block 9 the least anisotropic of {6, 9, 12} and the best under the frozen
framing (pilot, candle 1-shot: block 12 scored 69.4 image AUROC vs 89.9 for block 9); blocks
6/12 are reported as sensitivity, not candidates. The direct visual consequence is in
Fig. 4 (`figs/heatmap_panel.png`): DINOv2's distance field is near-flat over the background
with isolated hotspots, while CLIP's block-9 map is high-variance everywhere, so large areas
of *normal* background sit close to the decision threshold (candle: CLIP's calibration
threshold 0.355 vs DINOv2's 0.269 at the same nominal operating point). Small defects then
drown in background variance — exactly the capsules (−15.1) and fryum (−10.3) regime.

## 5.5 Error taxonomy (1-shot, seed 0)

Each test image receives a verdict from its own arm's calibration threshold; the two verdicts
are crossed image-by-image (`results/part3/error_cases.csv`, 960 rows).

| category | n | both right | only DINOv2 | only CLIP | both wrong |
|---|---:|---:|---:|---:|---:|
| candle | 200 | 141 | 29 | 18 | 12 |
| capsules | 160 | 70 | 57 | 6 | 27 |
| cashew | 150 | 83 | 27 | 7 | 33 |
| chewinggum | 150 | 127 | 14 | 4 | 5 |
| fryum | 150 | 71 | 56 | 6 | 17 |
| pipe_fryum | 150 | 115 | 10 | 17 | 8 |
| **all six** | 960 | **607 (63.2%)** | **193 (20.1%)** | **58 (6.0%)** | **102 (10.6%)** |

The arms agree on only 63.2% of images. Defects missed by both: 90 of 600; normals flagged by
both: 12 of 360. CLIP is right on **58 images where DINOv2 is wrong** (35 of them on candle
and pipe_fryum, the two categories where its deficit is smallest or reversed) — a concrete,
countable complementarity that bounds any score-fusion gain from below and motivates §5.6.
The fourth cell ("only the VLM is right") will complete the taxonomy once the Qwen branch's
per-image scores fill the `vlm_verdict` column (TODO-B). Fig. 4 shows the six shared
both-miss cases (one per category, chosen by `scripts/pick_cases.py` before either map was
inspected) with ground truth and both score maps.

## 5.6 Score-level fusion of the visual pair — machinery pilot

> Label in the report: *pilot, 1-shot seed 0, not comparable with Table 1's three-seed rows*.
> This subsection exists to validate the fusion protocol before the Qwen branch lands; it
> must not be presented as a main result.

With the pre-registered rule — per-category median/MAD standardization on the 20 calibration
scores, fused = α·z_DINOv2 + (1−α)·z_CLIP, main α = 0.5 (`scripts/score_fusion.py`; the
α = 1/α = 0 rows reproduce the single branches exactly, max deviation 0.0) — the fused scores
on the 960 images of the 1-shot seed-0 cell give:

| α (weight on DINOv2 z) | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 |
|---|---:|---:|---:|---:|---:|
| image AUROC, macro (%) | 85.97 | 91.61 | **93.75** | 94.05 | 93.11 |
| realized FPR at nominal 5% | 14.2% | 16.9% | 17.5% | 13.9% | 9.4% |

Two honest readings. *Ranking*: at the pre-registered α = 0.5 the fused score beats the
better single arm by +0.64 points (94.05 at α = 0.75 is the sweep's maximum and is reported
as sensitivity only); the gain is bounded by the 58 "only CLIP right" images and is consistent
with the taxonomy. *Reliability*: the realized false-alarm rate at the nominal 5% threshold is
9.4–17.5% across α, *including the single branches* — 20 training-normal calibration images
do not cover the variation of the normal test images, so thresholds must be reported as
achieved operating points, not calibrated guarantees [guo2017calibration,
geifman2017selective]. Full per-category numbers and the normalization audit trail are in
`results/part3/fusion/fusion_visual_pair_1shot_seed0.md`.

TODO-A(server): rerun the same command once Qwen scores arrive (E3/E4), and re-run per
(shot, seed) for the confirmatory grid — the script consumes saved per-image dumps only.

## 5.7 Cost

Grid wall-clock on the shared RTX A6000 (kNN is CPU-bound; GPU mostly idle during search):
CLIP grid 265.6 min, DINOv2 control ≈4 h, stage-4 (per-image dumps, heatmaps, taxonomy,
tables, curves) ≈27 min. Features are extracted exactly once per backbone and cached in
float32; the α sweep after that is offline and costs milliseconds.

> TODO-A(server, mandatory per brief §6.2): measure **average inference time per image** and
> **peak GPU memory** on one device for both backbones (and the Qwen branch), stating whether
> feature extraction is included. Not yet measured — currently only grid wall-clock exists.
> This is a brief requirement, do not submit without it.

## 5.8 Threats to validity

The controlled arms differ in patch size (14 vs 16 px) and parameter count (22 M vs 86 M
vision parameters), so E1 measures *this* CLIP encoder against *this* DINOv2 encoder, not
"CLIP vs DINOv2" in general; no patch-size-matched pair is publicly released. CLIP runs at a
resolution and normalization different from its pretraining setup; the bicubic position-table
interpolation addresses this and is probe-verified (Method §4.1) but cannot be perfect. The
fusion pilot is one cell (1-shot, seed 0) of the two *visual* arms; nothing in it may be
generalized until the per-(shot, seed) grid and the Qwen branch are in. Finally, all six
categories are object-centric product photographs; none of these conclusions should be quoted
for texture categories or scene-level anomalies, and MVTec AD 2's harder scenarios
[hecklerkram2026mvtec2] indicate the easier part of the problem space is what we measure.

---

## 合并前核对清单（报告定稿前）

- [ ] B 的 Qwen 小节插入 §5.4 之后；`error_cases.csv` 的 `vlm_verdict`（现 0/960）填齐后
      补第四格并重述 §5.5 的互补性论证
- [ ] E3/E4（DINO+Qwen）与 E5（CLIP+Qwen）跑完后，§5.6 升级为正式小节，删"pilot"限定
- [ ] §5.7 的延迟/显存测量（服务器，brief 硬性要求）
- [ ] 所有表格数字重新与 `tables.md` 机器核对一遍（防止改稿过程中手滑）
