# Error taxonomy -- DINOv2@448 vs CLIP@448

Verdict per image from each backbone's own calibration threshold (95th percentile of 20 normal calibration images); no test label is used to set it.

| category | n | both right | only DINOv2@448 right | only CLIP@448 right | both wrong |
|---|---|---|---|---|---|
| candle | 200 | 141 | 29 | 18 | 12 |
| capsules | 160 | 70 | 57 | 6 | 27 |
| cashew | 150 | 83 | 27 | 7 | 33 |
| chewinggum | 150 | 127 | 14 | 4 | 5 |
| fryum | 150 | 71 | 56 | 6 | 17 |
| pipe_fryum | 150 | 115 | 10 | 17 | 8 |
| **all six** | 960 | 607 | 193 | 58 | 102 |

Share of all test images: both right 63.2%, only DINOv2@448 20.1%, only CLIP@448 6.0%, both wrong 10.6%.

Defects missed by both: 90 of 600 defects; normal images flagged by both: 12 of 360 normals.

The `vlm_verdict` column in the companion CSV is intentionally empty: the fourth taxonomy cell (only the VLM is right) needs the Qwen branch's per-image scores.
