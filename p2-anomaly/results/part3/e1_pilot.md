# E1 pilot — CLIP backbone under the frozen Part-2 protocol

Stage 3 / step 2 of `docs/part3/实验规划与两人分工.md` (E1: CLIP visual features + few-shot kNN
vs the official DINOv2 baseline).

Scope: pilot on **candle + capsules**, **1-shot, seed 0**, kNN protocol frozen from Part 2
(L2-normalised patch features, 8-rotation reference bank, top-1% mean distance, pooled pixel AUROC,
20 normal calibration images for the threshold). All rows below are **no PCA mask** except the
official baseline row, which is the archived masked run.

## 1. What the pilot changed in the frozen configuration

Three issues showed up before the full grid, all of them resolvable without touching the
scoring protocol:

**(a) The final CLIP block is unusable for dense matching.**
CLIP's last block mixes global context into every patch token, so intra-image patch features
collapse towards a common direction. Measured mean pairwise cosine similarity inside one image
(200 sampled patches, 3 test images):

| features | candle | capsules | effective rank (candle / capsules) |
|---|---|---|---|
| DINOv2 @448 L12 | 0.361 | 0.381 | 18.4 / 16.1 |
| CLIP @448 L12 | 0.569 | 0.666 | 53.6 / 48.0 |
| CLIP @448 L9 | 0.306 | 0.289 | 31.4 / 29.5 |
| CLIP @448 L6 | 0.261 | 0.204 | 43.9 / 55.6 |

The final block loses 17–22 image-AUROC points against mid blocks for exactly this reason.

**(b) Official CLIP preprocessing discards object pixels.**
`openai/CLIP` resizes the shorter edge and then **centre-crops** to a square. On VisA the object
is not always centred, and the crop removes parts of it — the anomaly map then has nothing to
compare against. Switching to the DINOv2-wrapper framing (shorter edge → 448, trim the remainder
to a patch multiple, rectangular grid, no content removed) recovers a large amount of pixel AUROC:

| CLIP @448 L9, 1-shot seed 0 | candle img / px | capsules img / px |
|---|---|---|
| centre-crop (official CLIP) | 89.89 / 96.41 | 72.67 / 75.45 |
| squash whole image to square | 85.89 / 97.50 | 80.83 / 86.69 |
| **short edge, no crop (DINOv2 framing)** | **87.90 / 96.99** | **80.02 / 86.42** |

**(c) `clip.load` returns fp16 weights on CUDA.** Feeding fp32 tensors raises
`Input type (torch.cuda.FloatTensor) and weight type (torch.cuda.HalfTensor)`. The wrapper now
calls `model.float()`: fp16 patch features would reintroduce exactly the quantisation noise that
was removed from the feature cache in Part 2 (`parity_report.md`).

## 2. Full pilot table (1-shot, seed 0, no mask)

| configuration | candle img | capsules img | candle px | capsules px |
|---|---|---|---|---|
| DINOv2 @448 L12, PCA mask (**official archive**) | 90.36 | 98.78 | 98.57 | 95.48 |
| DINOv2 @448 L12, no mask | 90.25 | 95.93 | 98.76 | 95.66 |
| DINOv2 @224 L12, no mask | 79.71 | 85.23 | 97.06 | 95.79 |
| CLIP @448 L12, centre-crop | 69.39 | 70.77 | 90.62 | 71.08 |
| CLIP @448 L9, centre-crop | 89.89 | 72.67 | 96.41 | 75.45 |
| CLIP @448 L6, centre-crop | 85.46 | 83.87 | 94.38 | 73.31 |
| CLIP @224 L12, centre-crop | 79.70 | 67.13 | 87.74 | 75.71 |
| CLIP @448 L6, squash | 83.52 | 87.73 | 95.82 | 85.57 |
| CLIP @448 L12, squash | 73.95 | 67.95 | 93.70 | 88.15 |
| CLIP @448 L9, squash | 85.89 | 80.83 | 97.50 | 86.69 |
| CLIP @448 L6, short edge | 86.01 | 88.25 | 95.38 | 85.32 |
| **CLIP @448 L9, short edge** | **87.90** | **80.02** | **96.99** | **86.42** |
| CLIP @224 L9, short edge | 83.22 | 78.53 | 95.92 | 90.60 |

## 3. Frozen E1 configuration

| item | frozen value | reason |
|---|---|---|
| backbone | CLIP ViT-B/16 (512-d patch tokens, `ln_post` + `proj` on the final block only) | standard checkpoint; patch width matches a text-alignment variant later |
| resolution | 448, bicubic positional-embedding interpolation | same evaluation grid as the DINOv2 baseline (28×28); 224 native is within 1–2 points, kept as a sensitivity row |
| preprocessing | `short_edge` (shorter edge → 448, no content cropped, rectangular grid) | identical framing to the DINOv2 wrapper, removes the centre-crop confound |
| feature block | 9 (of 12) | best pixel AUROC of the pilot, first block with low anisotropy; L6 and L12 reported as sensitivity |
| mask | none (PCA masks off for both backbones) | controlled E1 comparison; keeps the backbone the only variable |
| everything else | kNN, rotations, top-1%, calibration threshold | frozen from Part 2, unchanged |

Pilot means over the two categories: CLIP@448 L9 short_edge 84.0 (image) / 91.7 (pixel);
the same-protocol DINOv2 reference is 93.1 / 97.2. CLIP is behind but not degenerate, and the
gap is concentrated in pixel localisation on capsules — which is the honest E1 result and the
motivation for the multimodal branch rather than a reason to tune the protocol further.

## 4. Reproduce

```bash
# pilot rows
python run_grid.py --name pilot --backbone clip --resolution 448 --no_mask --layers 9 \
    --clip_resize_mode short_edge --shots 1 --seeds 0 --categories candle capsules
# full grid
python run_grid.py --name e1_clip448_L9_nomask --backbone clip --resolution 448 --no_mask \
    --layers 9 --clip_resize_mode short_edge --shots 1 2 4 8 --seeds 0 1 2
```

Interface verification (`scripts/verify_clip_interface.py`): token raster order proven before
attention, manual forward bit-identical to `encode_image` (max diff 0.0), 448 interpolation
cos 0.945 against the native global vector, and the occlusion probe
(`scripts/probe_clip_alignment.py`) puts the max-response token within 3 patches of the occluded
block — so the grid geometry is right and the results above are feature quality, not a wiring bug.
