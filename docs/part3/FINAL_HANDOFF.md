# Final delivery handoff

`main` contains Part 1/2 results and A's CLIP grid, nine-cell DINOv2--CLIP fusion,
error taxonomy, figures, GPU benchmark and CVPR draft. Qwen code, frozen manifests
and per-image predictions are on `codex/part2-missing-results`.

### Report-ready E2 text

Qwen2-VL-7B-Instruct receives a category, normal references and a query image.
We score two fixed JSON completions by teacher forcing and subtract the NG sequence
log probability from the OK sequence log probability. Both the numeric score and
decision differ across candidates. The sign was selected on a development pilot, then
fixed. This branch provides an image score only.

On the same 696 VisA test images at 1-shot seed 0, macro image AUROC was 95.75 for
Qwen, 93.13 for DINOv2 and 85.30 for CLIP. Across three reference seeds on these
images, Qwen scored 95.32 +/- 0.50 at 1-shot and 95.66 +/- 0.26 at 2-shot (mean and
sample SD). All three paired intervals for 2-shot minus 1-shot included zero.
Only seed 0 retained the original reference; seeds 1 and 2 compare distinct archived
sets. We recommend 1-shot and did not run 4/8-shot for Qwen. Cashew recall under
the normal-calibrated threshold was 25.6% at 1-shot seed 0. Forced-choice scoring
averaged 2.64 s/image on the shared A6000; the v4 run recorded 23.47 GB mean per-image
peak allocation. These timings are separate from the visual end-to-end benchmark.
The equal-weight Qwen--DINOv2 fusion scored 97.68 on the same seed-0 subset as a
post-hoc exploration, without independent confirmation.

Sources: [v4 audit](../../results/extensions/qwen/confirm_full_v4/REPORT_V4.md),
[1-shot](../../results/extensions/qwen/stability_v5/REPORT_V5.md),
[2-shot](../../results/extensions/qwen/stability_v5/shots2_seed_summary_v5ext.json),
[Related Work](../../results/extensions/qwen/RELATED_WORK_TABLE.md).

### Submission checklist

1. Integrate the B branch; keep the 696-image comparison separate from visual full-grid numbers.
2. Incorporate E2 in the CVPR source, resolve author names and remaining placeholders,
   check the brief's references, then rebuild the final PDF and check the body-page limit.
3. Assemble `anomaly_demo.zip` with runnable demo or recording, representative outputs,
   and README. Existing Part 1 assets are inputs for this package. Label any heatmap as
   coming from the visual branch and keep the Qwen score separate.
4. Open the final report and ZIP and check the public GitHub link before Canvas submission.
