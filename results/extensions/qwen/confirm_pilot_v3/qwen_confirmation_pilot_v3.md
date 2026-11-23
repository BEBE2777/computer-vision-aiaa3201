# Qwen 独立确认 pilot v3 — 冻结协议（方向预先规定）

冻结时间：2026-11-23，在任何 v3 推理运行前。
本文件是 v2（`../v2_pilot/qwen_exploratory_pilot_v2.md`）的**独立确认实验**。
v1/v2 的全部产物与结论原文保留，本文件不修改它们。

## 1. 假设来源（明确非盲）
v2 pilot（120 张）观察到：`fc_score = ΣlogP(C_NG) − ΣlogP(C_OK)` 的 Image AUROC
六类一致低于 0.5（宏 0.0417）；翻转符号后宏 0.9583。该方向是**看到 v2 结果后**
提出的假设，因此必须在**全新、未参与 v2 的图片**上、以**预先规定的方向**确认，
否则不能用。v2 的 120 张与 v1 的 24 张一律不进入 v3。

## 2. 评测范围与抽样（冻结，先于推理）
- 六类各 10 张正常 + 10 张异常，共 120 张，1-shot、refs seed 0（沿用归档 refs）。
- 排除：v1 pilot 24 张（`../v2_pilot/old_pilot_images_all.json`）+ v2 pilot 120 张
  （`../v2_pilot/manifest_v2.json`）。
- 抽样：每类对剩余 `sorted(test/good)`、`sorted(test/bad)` 用
  `random.Random(20261123)` 依次 `sample(10)`，类别顺序固定。
- 图片 ID 先写入 `manifest_v3.json` 并冻结，再运行。

## 3. 模型与评分（与 v2 完全一致，唯一变量是图片集）
- 主模型 Qwen2-VL-7B-Instruct（bf16，A6000），2B 为规模对照。
- 输入、prompt v1、两个候选补全字符串、teacher-forced 打分方式全部沿用 v2，
  不新增任何模型/prompt/候选变量。

## 4. 预先规定的分数方向（关键）
- **主分数（confirmatory）**：`fc_conf = ΣlogP(C_OK) − ΣlogP(C_NG)`（即 v2 fc_score 取负）。
  依据：v2 六类一致反向。
- 同时报告未翻转的 `fc_score`、`fc_score_mean` 与旧 JSON `anomaly_score`，
  不因 v3 结果更换"最佳方法"。

## 5. 预先规定的成功判据
- 确认成功：`fc_conf` 的六类宏平均 Image AUROC **≥ 0.70**，且**至少 5/6 类 > 0.55**。
- 若未达到：记为"v2 的反向现象**未获确认**"，报告实际数值，不做进一步调参
  （不改 prompt、不换候选、不换模型、不调阈值）。
- 任何情况下不在这批图上做融合、不挑符号、不重采样。

## 6. 指标与产物
- Image AUROC（每类 + 六类宏平均）、JSON 有效率、mean/p95 延迟、峰值显存。
- 分数无方差 → 写"AUROC 无定义"。无像素输出 → 不报 Pixel AUROC。
- 产物：`manifest_v3.json`、`predictions_v3_{7b,2b}.json`、`summary_v3*.json`、
  `REPORT_V3.md`。提交前检查无权重/数据/密钥/密码。

## 7. 与 DINO/CLIP 的关系
`../v2_pilot/matched_baselines_v2.json` 给出同一 120 张（v2 图）上的
DINO 0.9333 / CLIP 0.8667 / Qwen(fc_conf) 0.9583。v3 只检验 Qwen 的方向是否
在留出图上成立；即使成立，与 DINO/CLIP 的普适比较仍需完整网格，v3 不外推。
