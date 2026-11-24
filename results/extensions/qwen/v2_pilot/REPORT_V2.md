# Qwen 探索性 pilot v2 — 结果报告（forced-choice 序列对数概率）

日期：2026-11-22。协议：`qwen_exploratory_pilot_v2.md`（推理前冻结）。
Manifest：`manifest_v2.json`，sha256 `8b04d21705d2ec3acd131da58b367a00e1dff146f86a2ad450269109d94fa3a4`
（120 张 = 六类各 10 正常 + 10 异常，1-shot/seed 0 归档参考图，排除 v1 pilot 24 张，抽样种子 20261122）。
**结论只适用于这 120 张 pilot，不外推到整个 VisA。**

## 运行
- 服务器：课程 A6000（48GB），bf16，贪心解码 + teacher-forced 打分，无付费 API。
- 命令：`run_v2.py --model {7b,2b}`（读取冻结 manifest；标签仅存
  `true_label_for_eval_ONLY`，从未进入模型输入）。
- 耗时：7B 649s、2B 663s（含 3 次前向/query：贪心生成 + 两个候选打分）。

## 两种读出的结果（预先规定，同时报告，不挑"最佳"）

### 旧读出：贪心 JSON anomaly_score
| 模型 | JSON 有效率 | AUROC |
|---|---|---|
| 7B | 100% | cashew 0.55 / chewinggum 0.60 / fryum 0.55；candle、capsules、pipe_fryum 分数无方差 → **AUROC 无定义**（4 个唯一值，118 OK / 2 NG） |
| 2B | 100% | 六类全部无方差 → **AUROC 无定义**（116 OK / 4 NG） |

与 v1 pilot 结论一致：自由生成的数值字段仍近退化。

### 新读出：forced-choice 序列对数概率
`fc_score = Σ logP(C_NG) − Σ logP(C_OK)`（越大越异常）。

| 模型 | 宏平均 AUROC | 每类 |
|---|---|---|
| 7B | **0.0417** | candle 0.00, capsules 0.08, cashew 0.16, chewinggum 0.00, fryum 0.01, pipe_fryum 0.00 |
| 2B | **0.5567** | candle 0.37, capsules 0.21, cashew 0.53, chewinggum 0.78, fryum 0.74, pipe_fryum 0.71 |

关键观察：
1. **fc_score 有完整方差**（7B 120 个唯一值；分数逐图不同），不是 v1 那种退化。
2. **7B 上一致性反向**：六类 AUROC 全部 < 0.5（0.00–0.16）。若事后翻转符号，
   等效 AUROC 为每类 0.84–1.00、宏平均 0.958。**该翻转是看到结果后才知道的，
   只能作为假设记录**——同一批图既产生观察又定义翻转，不能当作有效结果；
   确认它需要在**留出图上预先规定分数方向**再测。
3. 2B 无一致方向（宏平均 0.557，类别间 0.21–0.78 摆动），近似随机偏弱信号。
   "反向可能与模型规模有关"是一个可写进讨论的假设；两轮均为小样本，只能说
   2B 未能复现，不能断言规模因果，需留出集确认。

## 工程指标
| | 7B | 2B |
|---|---|---|
| JSON 有效率 | 100% | 100% |
| 生成延迟 mean / p95 | 2508 / 2940 ms | 3617 / — |
| 打分延迟 mean | 2566 ms | 1609 ms |
| 峰值显存 mean | 23.46 GB | 11.12 GB |

## Matched DINO / CLIP 对照（同一批 120 张）
来源：`p2-anomaly/results/part3/scores/e1_dino448_nomask.json` 与
`e1_clip448_L9_nomask.json` 的逐图 `paths/labels/scores`（1-shot、seed 0、448、
nomask）。该 dump 的参考图采样与归档 refs 同为 `pool[seed*n:(seed+1)*n]`
（calib_exclude 20），120 张全部命中，故为严格 matched。脚本：
`extract_matched_baselines.py`，明细 `matched_baselines_v2.json`。

| 读出（同一 120 张） | 六类宏平均 AUROC |
|---|---|
| DINOv2 @448（1-shot seed0） | **0.9333**（0.89–0.98） |
| CLIP L9 @448（1-shot seed0） | **0.8667**（0.67–0.99） |
| Qwen 7B JSON anomaly_score | 无定义（3 类无方差），其余 0.55–0.60 |
| Qwen 7B fc_score（按 v2 定义，未翻转） | 0.0417 |
| Qwen 7B fc_score（翻转，v2 事后观察值） | 0.9583 |

结论：在 v2 这批图上，DINO/CLIP 明显有效；Qwen 只有翻转符号后才看起来有效，
因此必须用独立图片确认（见 `../confirm_pilot_v3/`）。未做融合。

## 后续动作门槛（协议 §8 预先承诺，未触发）
- DINO top-k 区域 crop 复核：仅在 Qwen 分数方向经留出集确认有效后。
- 与 CLIP 融合：仅在两条分支都有有效分数后。本轮**未做**任何融合或权重挑选。

## 结论（严格限定）
在 120 张 v2 pilot 上：
- 自由生成 JSON 分数依旧退化（沿用 v1 负结果，限定范围不变）；
- forced-choice 分数有方差，但 7B 呈一致反向、2B 近随机；
  一致反向是事后观察到的假设，不是已确认的方法。
- 不据此主张 Qwen 在 VisA 上有效或无效；所有表述限于本 pilot。
