# Qwen fc_conf 留出全量确认 v4 — 冻结协议

冻结时间：2026-11-24，在任何 v4 推理运行前。
v1/v2/v3 的协议、产物、结论原文保留，本文件不修改它们。

## 1. 目的与范围
在**排除 v1（24）+ v2（120）+ v3（120）后剩余的全部 VisA 1cls 测试图**上，
用已确认方向的 fc_conf 做 1-shot/seed 0 确认实验，并给出带校准阈值的
FPR/召回率与不确定性区间。预期 696 张（以 manifest 实际数为准）。

## 2. 冻结项
- 模型：仅 Qwen2-VL-7B-Instruct（bf16，A6000）。不跑 2B、不跑其它 shot/seed。
- 方向（与 v3 一致，不再改动）：`fc_conf = ΣlogP(C_OK) − ΣlogP(C_NG)`，越大越异常。
- 候选补全、prompt v1、teacher-forced 打分：与 v2/v3 完全一致，不改 prompt。
- 输入隔离同前：标签/mask/其他模型分数绝不进模型。
- 不做任何融合。

## 3. 抽样与 manifest（先冻结后运行）
- 测试集：每类 `test/good`、`test/bad` 中排除 v1+v2+v3 已用图后的**全部**剩余图。
- 校准集：每类 `sorted(train/good)[:20]`（与 p2-anomaly calib_exclude=20 同规则，
  与 DINO/CLIP dump 的 calib_scores 同一批图），共 120 张；这些图不进入测试集。
- 图片 ID 与路径先写入 `manifest_v4_test.json` / `manifest_v4_calib.json`
  并冻结，运行只读 manifest。

## 4. 阈值规则（预先规定，运行前冻结）
- 每类 c：`thr_c = fc_conf 在该类 20 张校准图上的 95% 分位数`。
- 判异常：`fc_conf ≥ thr_c`（逐类阈值，逐类计算 FPR/召回率）。
- 不做任何阈值搜索或事后调整。

## 5. 指标
- AUROC：逐类 + 六类宏平均；95% CI 用分层 bootstrap（每类内有放回重采样，
  10,000 次，宏平均取 2.5/97.5 分位）。
- FPR：该类正常测试图中 fc_conf ≥ thr_c 的比例；召回率：异常测试图中的比例；
  各自给 bootstrap 95% CI。
- 耗时：逐图生成/打分延迟 mean/p95、总运行时间、JSON 有效率、峰值显存。
- matched 对照：同一批测试图的 DINOv2/CLIP 逐图分数（E1 dump，1-shot/seed0/448/
  nomask，refs 同源），阈值用各自 dump 中由同一批校准图得到的 thr（每方法自校准），
  报告其 AUROC/FPR/召回率供对照；不做融合。
- 分数无方差 → 写"AUROC 无定义"。无像素输出 → 不报 Pixel AUROC。

## 6. 结论限定
所有结论只针对本 manifest 实际评测的图片子集；不外推全 VisA 官方口径，
不与完整网格数字直接混排（对照的 DINO/CLIP 数字同样是子集估计）。

## 7. 产物
`confirm_full_v4/`：本协议、`build_manifest_v4.py`、`manifest_v4_*.json`、
`predictions_v4_calib.json`、`predictions_v4_test.json`、`summary_v4*.json`、
`matched_baselines_v4.json`、`REPORT_V4.md`。
