# Qwen 探索性 pilot v2 — forced-choice 序列对数概率（冻结协议）

冻结时间：2026-11-22（本文件写入后、任何 v2 推理运行前）。
本协议是**新增的、范围受控的一轮探索 pilot**。它不修改、不覆盖 v1 冻结协议
（`../PROTOCOL_FROZEN.md`）的任何内容，不恢复 11,520 张全量网格。
v1 的全部产物（2B/7B JSON 打分 pilot、负结果记录）原样保留。

## 1. 动机与定位
v1 pilot（24 图，1-shot，seed 0）中 Qwen2-VL-2B/7B 的自由生成 JSON 输出
`anomaly_score` 恒为 0.0（AUROC 无定义）。一个假设是：**自由生成的数值字段
不是模型表达图像证据的可靠通道**——模型可能"知道"但不会写出连续分数。
v2 因此增加一种完全不同的读出方式：不要求模型生成 JSON，而是计算两个候选
回答的**序列对数概率**（forced-choice），用相对偏好作为异常分数。
两种读出在 v2 中**同等报告**，不根据这批图挑"最佳方法"。

结论范围严格限定于本 pilot 的 120 张图，不得外推到整个 VisA。

## 2. 可行性检查（已完成，推理前）
- 脚本：`feasibility_logprob_cpu.py`（本地 CPU，Qwen2-VL-2B，合成图）。
- 结果：teacher-forced 前向 + 候选 token 段 log-softmax 求和可正常计算；
  合成图上 logP(NG) − logP(OK) = −1.559（偏向 OK，方向合理）。
- 模型自报 confidence **不作为校准概率**，仅存档。

## 3. 模型（冻结）
- 主模型：`Qwen/Qwen2-VL-7B-Instruct`，bfloat16，单张 A6000。
  依据：协议 v1.1 已解冻并实测的最终 checkpoint；v2 不再引入新模型变量。
- 2B 为可选复核（同一 manifest、同一评分流程），仅作规模对照，不改变结论主对象。

## 4. 评测范围与抽样（冻结，先于任何推理）
- 六类各抽 10 张正常 + 10 张异常 test 图，共 120 张；固定 1-shot、seed 0。
- 参考图：沿用仓库归档 `results/part2/refs/1shot_seed0/`（服务器 `refs_archive`），
  与 DINO 基线、v1 pilot 同源。每类前 20 张正常训练图仍为留出校准图，不入池。
- 抽样规则（确定性、可复现、在看到任何模型结果之前冻结）：
  1. 旧 v1 pilot 的 24 张图（2B/7B 选图一致，清单存档 `old_pilot_images_all.json`）**排除**；
  2. 每类分别对 `sorted(test/good)` 与 `sorted(test/bad)` 的剩余文件，
     用 `random.Random(20261122)` 依次 `sample(10)`（类别顺序固定：
     candle, capsules, cashew, chewinggum, fryum, pipe_fryum）；
  3. 图片 ID 在推理前写入 `manifest_v2.json`（含每图路径、标签字段仅存
     `true_label_for_eval_ONLY`）并回传仓库冻结；运行只读 manifest。

## 5. 输入隔离（与 v1 相同，绝不放松）
每次 query 的输入只有：类别名、1 张归档正常参考图、query 图、冻结 prompt v1
（sha256 `596118ac…7af`，占位符替换不变）。**绝不输入**：标签、文件名中的真值
线索、GT mask、DINO/CLIP 分数。标签仅存于 `true_label_for_eval_ONLY` 供离线评测。

## 6. 两种评分方式（预先规定，同时报告）
### 6.1 旧读出（与 v1 相同）
贪心生成（`do_sample=False`，max_new_tokens=256）→ JSON 解析 → `anomaly_score`。
### 6.2 新读出：forced-choice 序列对数概率
两个候选补全在 v2 内冻结（镜像模型自然 JSON 风格、字段顺序一致、confidence 取
中性 0.5，仅 decision 与 anomaly_score 不同）：
```
C_OK = {"decision":"OK","anomaly_score":0.0,"confidence":0.5,"defect_type":"unknown","evidence":"","regions":[]}
C_NG = {"decision":"NG","anomaly_score":1.0,"confidence":0.5,"defect_type":"unknown","evidence":"","regions":[]}
```
每个候选做一次 teacher-forced 前向：候选 token 段的 log-softmax 逐 token 求和。
- 主分数：`fc_score = Σ logP(C_NG) − Σ logP(C_OK)`（越大越异常）。
- 辅助分数：`fc_score_mean`（按候选 token 数归一），以及两个候选各自的
  ΣlogP 数值本身。
- 不做温度缩放、不做长度外的任何归一化变体挑选。

## 7. 指标
- 每种分数分别报告 Image AUROC（每类 + 六类宏平均）。分数无方差时明确写
  **"AUROC 无定义"**（不写 0.5）。
- 附加：JSON 有效率、mean/p95 延迟、峰值显存、input_tokens。
- 无像素级输出 → 不报 Pixel AUROC。
- **matched DINO 对照**：仓库现有 `results/part2/eval448_*.json` 只有每类聚合
  分数，没有逐图分数落盘 → 本轮不做 DINO 逐图对照，标记为后续依赖（需
  DINO runner 重跑 120 张逐图 dump 或找到已有逐图缓存）。

## 8. 后续动作门槛（预先承诺）
- 仅当 Qwen 在这 120 张上产生**有方差**的分数时，才建议下一步做 DINO top-k
  区域 crop 复核。
- 仅当 CLIP 线与 Qwen 线都有有效分数时，才讨论融合；本轮不做任何融合，
  不用这批图挑融合权重。

## 9. 产物清单
`v2_pilot/` 下：本协议、可行性脚本与结果、`old_pilot_images_all.json`、
`build_manifest.py`、`run_v2.py`（服务器端）、`manifest_v2.json`（冻结）、
`predictions_v2.json`（逐图原始记录）、`summary_v2.json`、`REPORT_V2.md`。
旧产物一律不动。提交前检查：无模型权重、数据集、密钥、密码进 Git。
