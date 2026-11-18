# Method — visual branch (draft, Part 3 / E1)

> Status: prose is final; the two result tables marked `TODO` are filled from
> `results/part3/e1_full.md` once the E1 grid finishes. Report language is English, so this
> draft is English.

## 1. Setup and evaluation protocol

We evaluate on six categories of VisA (candle, capsules, cashew, chewinggum, fryum,
pipe_fryum) in the few-shot setting, with 1, 2, 4 and 8 normal reference images per category and
three sampling seeds (0/1/2). For every category the first 20 training images are held out as a
calibration set and never used as references; the remaining normal images form the pool from
which the k-th shot of the s-th seed is drawn (`pool[s*shot:(s+1)*shot]`), so every method sees
exactly the same references for a given (shot, seed).

Two metrics are reported. Image-level AUROC uses the mean of the top 1 % of the pixel distance
map as the anomaly score, as in AnomalyDINO. Pixel-level AUROC is computed on maps resampled to
448×448 for all methods, because pooling the native-resolution pixels of 900 images does not fit
in the shared GPU memory; the same resampling is applied everywhere, so the comparison between
methods stays well defined. Mean ± std is always taken over seeds: the six-category mean is
formed per seed first, and the standard deviation is taken over the three per-seed means. The
decision threshold used only for the qualitative error analysis is the 95th percentile of the
scores of the 20 calibration images, i.e. it never sees a test label.

## 2. Part 1 — template difference baseline

The baseline treats the reference set as a template. Images are resized so that the shorter edge
is 448 and centre-cropped to 448×448, because a pixel-wise comparison needs a common frame; the
ECC (Euclidean) registration maps every image onto the first reference; the difference between a
test image and a reference is `0.5·|ΔI| + 0.5·(1 − SSIM)`; the closest template (lowest mean
difference) becomes the anomaly map, which is Gaussian-smoothed and aggregated with the same
top-1 % mean as AnomalyDINO so that image AUROC stays comparable. Scores are never rescaled per
image, and the OK/NG threshold is the 95th percentile of the 20 calibration scores. The baseline
needs no training and no pretrained weights, and it fixes the lower bound of the comparison: a
pretrained-feature method that does not beat it is not earning its complexity.

## 3. Part 2 — AnomalyDINO reproduction

AnomalyDINO extracts dense patch features from DINOv2 ViT-S/14 at 448 (32×32 patch grid),
builds a memory bank from the reference patches (each reference contributes eight rotations,
which makes the bank rotation-tolerant), and scores a test patch by its L2 distance to the
nearest bank entry; the pixel map is the upsampled distance field, smoothed with a Gaussian. We
first reproduced this pipeline with the official code (Part 2), then re-implemented the scoring
in a cached runner so that Part 3 variants can be compared without re-extracting features.

The re-implementation is verified rather than assumed: `scripts/parity_official_vs_cached.py`
compares the two pipelines cell by cell, and after the fixes described in
`results/part2_official/parity_report.md` all 72 cells (6 categories × 4 shots × 3 seeds) agree
to the last printed digit. Two defects were found and fixed on the way, and both matter for
Part 3:

1. **Cache-key collision.** VisA reuses file names across splits (capsules shares eight basenames
   between `test/good` and `test/bad`, and ~90 between `train/good` and `test/bad`). A
   basename-keyed feature cache therefore served a defective image the features of a normal
   image with the same name, which cost up to 5 image-AUROC points on capsules. Cache keys are
   now namespaced by the parent directory.
2. **Lossy cache dtype.** Storing cached features as float16 perturbed query features by up to
   ~8·10⁻³, enough to change the top-1 % nearest neighbours of borderline images; the cache is
   now float32.

With those fixes the cached pipeline reproduces the official numbers on all six categories, so
Part 3 differences can be attributed to the backbone rather than to the harness.

## 4. Part 3 / E1 — replacing the backbone with CLIP

The question E1 asks is narrow: with the scoring rule, the reference sampling, the calibration
and the resolution all held fixed, what changes when the dense features come from a
language-aligned image encoder (CLIP ViT-B/16) instead of a self-supervised one (DINOv2
ViT-S/14)? DINOv2 features are trained to be locally discriminative, which is what a
nearest-neighbour patch matcher wants; CLIP features are trained to place a whole image next to
a text description, and are known to be strongly anisotropic at the patch level. Whether that
costs accuracy in few-shot anomaly detection is an empirical question, and it is the one the
rest of this section answers.

### 4.1 Extracting patch tokens from CLIP

`clip.encode_image` returns only the pooled global feature, so the patch tokens have to be taken
from the vision tower directly: the input is patch-embedded and passed through
`visual.transformer.resblocks`, and for block *l* we take the output of the residual stream,
drop the class token, apply `ln_post` and the projection matrix (`proj`) so that the vectors live
in the same space as the global embedding, and keep the remaining `grid_h × grid_w` tokens in
raster order. Three properties were verified explicitly, because a silent mistake in any of them
produces plausible-looking but wrong AUROC:

* **Grid order.** A synthetic probe replaces one patch of the input with a white square; the
  argmax of the extracted tokens must be `row*14 + col`. It is (5·14+8 = 78), confirming raster
  order rather than, e.g., column-major or CLS-first offset.
* **Positional embedding at 448.** CLIP is trained at 224 with a 14×14 position grid; a 448
  input produces a 28×28 grid, so the learned position table is interpolated bicubically to the
  actual grid shape (which is rectangular under the `short_edge` preprocessing, hence shape-aware
  rather than fixed 28×28). An occlusion probe checks that the score map responds at the occluded
  location: the argmax lands within 3 patches of the true occlusion centre, which rules out a
  transposed or mis-scaled position table.
* **Forward equivalence.** For the native 224 input, our manual forward and the official
  `encode_image` agree bit-for-bit on the class token (difference 0.0), so the token extraction
  path is not introducing an off-by-one block or a missing normalisation.

The resulting class `clip_core.LayerCLIP` mirrors the interface of `ad_core.LayerDINOv2`
(`prepare_image` → `extract_layers` → patch features + grid), which lets the same cached
`run_grid.py` scoring pipeline evaluate both backbones; no scoring code is duplicated.

### 4.2 Choices frozen before the grid

The pilot (candle and capsules, 1-shot, seed 0) fixed four settings; none of them were tuned on
the test split afterwards.

* **Resolution 448, not 224.** The interpolated position table is stable at 448, so we keep the
  same resolution as Part 2 instead of falling back to CLIP's native 224. This also removes the
  need for a resolution-matched DINOv2 control at 224: the control is DINOv2 at 448.
* **Layer 9 of 12.** Patch features were swept over blocks 6, 9 and 12. The last block is the
  most anisotropic in CLIP (mean pairwise cosine similarity of patch features inside one image
  0.57–0.67, against 0.36 for DINOv2 L12), and its dense matching suffers for it: at 1-shot on
  the pilot, block 12 scored 69.4 image AUROC on candle against 89.9 for block 9 under the same
  centre-crop framing. Block 9 has the lowest anisotropy of the three (0.29–0.31) and the best
  pixel AUROC under the frozen framing (candle 96.99, capsules 86.42); blocks 6 and 12 are
  reported as a sensitivity check, not as candidates.
* **No foreground mask, for both backbones.** AnomalyDINO's agnostic preprocessing estimates a
  foreground mask by PCA on DINOv2 features; the threshold is tied to that feature geometry and
  has no counterpart for CLIP. Rather than give one backbone a preprocessing advantage, E1
  disables the mask for both (`--no_mask`), so the only difference between the two lines is the
  encoder.
* **`short_edge` framing.** The official CLIP preprocessing shortens the shorter edge and then
  centre-crops to a square, which cuts objects out of frame on categories whose objects are
  off-centre (capsules pixel AUROC dropped to 75.5 under the crop). `short_edge` keeps the full
  frame and lets the patch grid be rectangular, matching what the DINOv2 wrapper does; the same
  framing is therefore used for both backbones.

### 4.3 Controlled comparison

The CLIP line and the DINOv2 control line share: the reference images (same pool, same
shot/seed slicing), the calibration set, the 448 resolution, the no-mask protocol, the
`short_edge` framing, the k-NN scoring rule (faiss `IndexFlatL2`, top-1 % mean, Gaussian-smoothed
map), and the metric code. The Part 2 official line uses the same scoring rule but the official
implementation, its PCA foreground mask and its own preprocessing; it is the reference point, not
a controlled arm. Part 1 is reported for completeness.

TODO: table — image/pixel AUROC (mean ± std over 3 seeds) per shot for Part 1, Part 2 official,
DINOv2@448 no-mask, CLIP@448 block 9.

TODO: table — per-category image AUROC at 1-shot for the two controlled arms.

## 5. Qualitative analysis

Tables cannot say which images a method gets wrong, so the two controlled arms are also compared
image by image. Every test image receives a verdict from its own backbone's calibration
threshold, and the two verdicts are crossed into a 2×2 table (both right / only DINOv2 right /
only CLIP right / both wrong; `results/part3/error_taxonomy.csv`). The heatmap panel in
`results/part3/figs/heatmap_panel.png` shows, for the same test images, the ground-truth mask and
both score maps, so a disagreement can be read as a difference in where the two encoders place
their nearest-neighbour distances rather than as an aggregate number.

The fourth cell of the taxonomy used in the full report — "only the VLM is right" — needs the
Qwen branch's per-image scores; the CSV carries an empty `vlm_verdict` column so that it can be
filled in without recomputing the visual side.

## 6. Threats to validity

The two arms differ in patch size (14 px for ViT-S/14, 16 px for ViT-B/16) and in parameter count
(22 M vs 86 M vision parameters), so E1 measures "this CLIP encoder vs this DINOv2 encoder" and
not "CLIP vs DINOv2" in general; a patch-size-matched pair (ViT-B/14 is not released) is not
available. CLIP's input normalisation and resolution also differ from its training setup, which
the position-embedding interpolation addresses but does not eliminate. Finally, all six
categories are object-centric product photos; conclusions need not transfer to textures or to
scene-level anomalies.
