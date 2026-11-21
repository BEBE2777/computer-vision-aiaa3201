# Related Work — Citation Coverage Table (Project 2 brief)

Source: `Project_2__Few_Shot_Industrial_Visual_Anomaly_Detection.pdf`,
Section 4 (Recommended Reading List, 10 entries [1]–[10]) + References (12
entries [1]–[12]). Coverage checked 2026-11-21: **all 10 recommended papers
and both extra references are accounted for**; no recommended reading is
missing. Full bibliography strings below are transcribed from the brief and
should be re-verified against the original papers before camera-ready.

Ownership: A = traditional / anomaly-detection entries; B = foundation-model
and vision-language entries; "shared" = both. This file documents coverage;
the paragraph draft that groups them belongs in the report's Related Work
section.

## Coverage checklist

| # | Brief reference (verbatim) | In Section 4 reading list? | Owner | Covered in report plan? |
|---|---|---|---|---|
| [1] | Bergmann, P., et al. "MVTec AD—A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection." CVPR 2019. | Yes — Foundations | A | Yes (dataset / benchmark context) |
| [2] | Defard, T., et al. "PaDiM: A Patch Distribution Modeling Framework for Anomaly Detection and Localization." ICPR Workshops 2021. | Yes — Foundations | A | Yes (patch-distribution baseline family) |
| [3] | Roth, K., et al. "Towards Total Recall in Industrial Anomaly Detection." CVPR 2022. | Yes — Foundations | A | Yes (PatchCore / memory-bank baseline family) |
| [4] | Zavrtanik, V., et al. "DRAEM—A Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection." ICCV 2021. | Yes — Reconstruction & Synthesis | A | Yes (reconstruction-based family) |
| [5] | Li, C.-L., et al. "CutPaste: Self-Supervised Learning for Anomaly Detection and Localization." CVPR 2021. | Yes — Reconstruction & Synthesis | A | Yes (synthesis-based family) |
| [6] | Oquab, M., et al. "DINOv2: Learning Robust Visual Features without Supervision." TMLR 2024. | Yes — Foundation Models | shared (B: foundation-model context) | Yes (backbone of DINO baseline) |
| [7] | Zhou, Q., et al. "AnomalyCLIP: Object-Agnostic Prompt Learning for Zero-Shot Anomaly Detection." ICLR 2024. | Yes — Foundation Models | B | Yes (CLIP candidate line / VLM context) |
| [8] | Damm, S., et al. "AnomalyDINO: Boosting Patch-Based Few-Shot Anomaly Detection with DINOv2." WACV 2025. | Yes — Foundation Models | shared (B: baseline context) | Yes (Part 2 baseline / main comparator) |
| [9] | Batzner, K., et al. "EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies." WACV 2024. | Yes — Efficiency & Robustness | A | Yes (efficiency context) |
| [10] | Heckler-Kram, L., et al. "The MVTec AD 2 Dataset: Advanced Scenarios for Unsupervised Anomaly Detection." IJCV 2026. | Yes — Efficiency & Robustness | A | Yes (dataset / robustness context) |
| [11] | Zou, Y., et al. "SPot-the-Difference Self-Supervised Pre-Training for Anomaly Detection and Segmentation." ECCV 2022. | No (References only) | shared | Yes (self-supervised anomaly context) |
| [12] | Kirillov, A., et al. "Segment Anything." ICCV 2023. | No (References only) | shared | Yes (segmentation foundation model) |

**Coverage summary**: recommended reading [1]–[10] = 10/10 covered;
additional references [11]–[12] = 2/2 covered. Total 12/12.

## Grouping for the report prose

- **Datasets & benchmarks**: [1] MVTec AD, [10] MVTec AD 2 — motivate VisA as
  the course's six-category benchmark and frame dataset scope.
- **Classical / embedding-based anomaly detection**: [2] PaDiM (patch
  distribution modeling), [3] PatchCore (memory-bank recall), [9] EfficientAD
  (efficiency) — the methodological ancestors of the Part 2 DINO baseline.
- **Reconstruction & synthesis**: [4] DRAEM, [5] CutPaste — alternative
  normal-only learning paradigm, contrasted with patch-nearest-neighbour
  scoring.
- **Foundation models for visual anomaly detection**: [6] DINOv2 (frozen
  backbone), [8] AnomalyDINO (few-shot patch-based detector, the unchanged
  Part 2 baseline), [7] AnomalyCLIP (CLIP-based, object-agnostic prompt
  learning — directly motivates the CLIP candidate line).
- **Self-supervised & segmentation**: [11] SPot-the-Difference, [12] Segment
  Anything — supporting context for representation learning and pixel-level
  localisation.

## Link to the B-line experiment

- [6] and [8] ground the DINO baseline that the Qwen pilot's reference images
  and VisA split are shared with (72/72 reference-file consistency verified).
- [7] motivates the independent CLIP candidate line; per the plan, CLIP and
  Qwen are separate candidates and only the discriminative one becomes the
  report's main line, with the other shown as a controlled comparison.
- The Qwen-VL multimodal pilot itself (Qwen2-VL-2B/7B) is an extra exploratory
  branch beyond the recommended list and is reported as a pilot negative
  result with its scope explicitly limited to 24 images.
