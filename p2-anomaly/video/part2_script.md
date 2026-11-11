# Part 2 video script — AnomalyDINO (~55 s, 150 words)

**Status note for the narrator:** all numbers below are *preliminary, seed 0*
results from the official AnomalyDINO pipeline. Do not present them as final;
4-shot and seeds 1/2 are still pending.

---

For Part Two, we reproduce AnomalyDINO, a few-shot anomaly detection method
built on DINOv2. It needs only a handful of *normal* reference images — no
defect examples at all. Each reference is resized to 448 by 448 and passed
through a frozen DINOv2 Vision Transformer, which outputs patch-level features.
These features form a memory bank of normal appearance. For a test image, every
patch is matched to its nearest neighbour in this memory bank, and that distance
becomes the patch-level anomaly score. We pool the top one percent of patch
distances into one image-level score, and the full patch distance map is
upsampled into a pixel-level heatmap. As preliminary evidence — a single
seed-zero run on the six VisA categories — the official pipeline reaches 94.6
percent image AUROC with one reference image, and 96.2 with eight, while pixel
AUROC stays around 98 percent. The full multi-seed evaluation is in progress.

## Suggested visuals

| Time | Asset |
|---|---|
| 0–12 s | `assets/part2_pipeline.png` (method schematic) |
| 12–25 s | `assets/part2_normal_heatmap.png` (normal case) |
| 25–40 s | `assets/part2_anomaly_heatmap.png` (defect case, score 0.172) |
| 40–55 s | `part2_results_table.md` rendered as a slide |
