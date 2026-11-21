# Qwen 多模态异常检查 — 冻结协议 v1

冻结时间：2026-11-18。本文件记录 B0/B1 之后、正式评测开始前锁定的全部配置。
**之后不得根据测试结果改动这里的任何内容**（prompt、模型、阈值、解码、评测范围）。

## 分支与代码
- 分支：`codex/part2-missing-results`（B 线独立实验）。
- 所有产物在 `results/extensions/qwen/`。

## 数据与参考图（与 DINO 基线同源）
- VisA 官方 1cls split，六类：candle(200), capsules(160), cashew(150), chewinggum(150), fryum(150), pipe_fryum(150)，共 960 测试图/评测单元。
- 数据来源：课程 GPU box `/data/disk1/shared/aiaa3201/project2_anomaly/VisA` + 官方 `split_csv/1cls.csv`，在 GPU box 上重建 1cls 布局（symlink 图像 + 二值化 mask）。
- 参考图：直接使用仓库归档 `results/part2/refs/`，**不重新采样**。
- **一致性已验证**：重建数据的 `pool[seed*shot:(seed+1)*shot]` 与归档 `reference_files` 在 72/72 个 category×shot×seed 组合上完全一致（见 `smoke/verify_refs.log`）。
- 每类前 20 张正常训练图为留出的校准图，不进入参考池，也不进入 VLM 上下文。

## 输入约定（严格防止标签泄漏）
每次 query 的上下文只包含：
1. 固定的类别名称（如 "candle"）；
2. 该 cell 的 n 张正常参考图（来自归档 refs）；
3. 一张待检 query 图；
4. 冻结的 prompt v1。

**绝不输入**：good/bad 路径、文件名中的真值线索、label、GT mask、异常热图、DINO 分数、类别流行度、任何文字真值。
每个 query 独立上下文，不混入上一张 query。真实标签仅以 `true_label_for_eval_ONLY` 字段写在预测记录里，供离线计算 AUROC，从不进模型。

## 模型（冻结）
- `Qwen/Qwen2-VL-2B-Instruct`，bfloat16，单张 A6000 48GB。
- 数据集 checkpoint 通过 hf-mirror 下载到 GPU box `/data/disk2/home/qzheng610/qwen_work/qwen2vl2b_local`。
- 解码：`do_sample=False`（贪心），`max_new_tokens=256`，无 temperature/top-p 调整。
- 视觉输入：Qwen2-VL 原生动态分辨率，processor 默认 `min_pixels/max_pixels`。

## Prompt（冻结）
- 文件：`prompts/qwen_inspect_v1.txt`
- sha256：`596118ac764bc609c0e2885fdc09964898e26a171a20f6d4e4ac49eeb64a87af`
- 只做占位符替换 `{category}` 和 `{n}`，不增删字句。

## 输出 schema（冻结）
- 文件：`prompts/schema_v1.json`
- sha256：`fedf3ad7bdab097bf590e8f8c9b9a175ad06467af62e61f48f42e73e0c3d8d54`
- 字段：decision(OK/NG/UNCERTAIN), anomaly_score[0,1], confidence[0,1], defect_type, evidence, regions[]。
- `anomaly_score` 为主排序分数；`confidence` 单存，不当作校准概率。
- JSON 解析：先严格 json.loads，失败则截取首个 `{...}` 再解析，仍失败记为 `json_fail` 并保留原始响应，不静默丢弃。

## 评测范围（确认性网格）
- 主张：1/2/4/8 shot × seed 0/1/2 × 六类 960 测试图 = 11,520 个 query。
- 若算力/时间不足，**缩小范围必须与 CLIP 分支使用同一子集**，并在报告中标注为 subset，不与完整网格直接排名。
- 主指标：Image AUROC（每类先算，再六类宏平均，再对 seed 宏平均算 mean/std）。
- Qwen 不产生可信像素图，不报告 Pixel AUROC，不把图像级分数涂成热力图。
- 附加：Image AUPRC、固定正常校准阈值下 FPR/FNR、mean/p95 延迟、JSON 失败率、显存峰值、token 用量。

## 已知风险与负结果（预先声明，不靠改 prompt 规避）
- 本地 CPU smoke 显示：Qwen2-VL-2B 在合成图上对正常/异常均输出 anomaly_score=0.0（pilot 性质，样本很少）。若 A6000 上真实数据的 pilot 仍如此，将作为 pilot 负结果如实报告，不反复调 prompt 追分。任何结论都限定在实际测试的试点图数量上，不外推到整个 VisA 数据集。
- 无可用付费 API 凭据（DASHSCOPE/OPENAI），不消耗付费额度。
- 完整网格在 A6000 上的耗时待 GPU smoke 实测后外推；如超出预算按上条缩小范围。

## 缓存键
包含：model id/revision、精度、prompt hash、所有图像内容 hash 及顺序、解码配置。不只用文件名。

---

## 协议修订 v1.1 — 模型解冻（2026-11-19，用户批准）

**唯一改动项：模型 checkpoint 从 2B 换成 7B。** 其余一切保持冻结：prompt v1（sha256 不变）、schema v1、解码配置、评测范围、输入约定、主指标全部不动。

- 理由：A6000 上真实 VisA 数据的 2B GPU **pilot smoke**（24 图，1-shot）显示 anomaly_score 恒为 0.0、23/24 判 OK，原始响应确认模型把 bad 图描述为"identical to normal"——**在这 24 张试点图上未能区分正常与异常**（解析无错误，图像输入已验证生效）。这不构成对模型在整个 VisA 上能力的断言。基于该 pilot 信号，继续用 2B 跑 11,520 张全量网格的边际价值很低，故不启动全量。
- 决策：用户确认「换 7B 重跑 smoke」。此为对冻结协议中「模型」一项的**一次性、显式解冻**，仅此一项。
- 新模型：`Qwen/Qwen2-VL-7B-Instruct`，bfloat16，同单张 A6000 48GB（7B bf16 权重 ~16GB，可容纳）。下载到 `qwen_work/qwen2vl7b_local`。
- 2B 结果保留：本地 CPU smoke + A6000 GPU smoke（`smoke/gpu_smoke_report.json`）作为负结果证据留存，报告中如实呈现。
- 判定规则不变：7B smoke 若仍无区分度，同样记为负结果，**不通过反复改 prompt 追分**；换模型只此一次，不迭代换更大 checkpoint。
- 脚本改动：`gpu_smoke.py` / `run_qwen.py` 增加 `--model {2b,7b}` 参数，默认 `2b`（保持 2B 可复现），载入路径与 model_id 随之切换。

### 7B smoke 结果（2026-11-19，负结果确认）
- 报告：`smoke/gpu_smoke_7b_report.json`（A6000 bf16，1-shot/seed0/2 每类，24 图）。
- 工程先验已排除：7B 与 2B 的 `input_tokens` 完全一致（如首图均 4257）、`preprocessor_config` 一致、chat_template 为官方正确版本（含 `<|image_pad|>`）——图像确实进入模型，`apply_chat_template` 正常（首次失败仅因下载漏了 `chat_template.json`，补齐后通过）。故以下为模型真实行为，非管线 bug。
- 指标：JSON 有效率 100%，mean 延迟 2692ms/图，峰值显存 17.66GB。
- **anomaly_score 唯一值 = {0.0}；decision 24/24 全 OK；24 条响应仅 2 种（`confidence` 0 或 1，`evidence` 全空）**。**在这 24 张试点图上未能区分正常与异常**；因预测分数无方差，**AUROC 无定义（不可计算，不是 0.5）**。
- 结论：按协议判定规则，7B 在此 pilot 上同为负结果。**不启动 Qwen 全量网格**（pilot 已显示无区分度，全量 11,520 query 的边际价值很低，非"已完成"），不反复改 prompt 追分，不再换更大 checkpoint（换模型仅此一次）。**报告措辞严格限定为"24 张 smoke/pilot 上未能区分"，不得据此断言 Qwen-VL 在整个 VisA 上没有异常检测能力**。Qwen 分支以 pilot 负结果如实写入报告；主线转向 CLIP 候选 + DINO 基线对比。
- 用户确认（2026-11-19）：「如果仍然几乎都输出 0 或无法区分正常与异常，就记录负结果，停止 Qwen 全量运行，转向 CLIP 主线」。

## 脚本与产物状态说明（供队友阅读，避免误判）
Qwen 线已结束于 pilot 负结果，**没有待启动的全量实验**。以下脚本按用途分类：

- **报告直接可用的产物**：`smoke/gpu_smoke_report.json`（2B pilot）、`smoke/gpu_smoke_7b_report.json`（7B pilot，含逐图原始响应）、`smoke/smoke_report.json`（本地 CPU 合成图）、`prompts/qwen_inspect_v1.txt` + `prompts/schema_v1.json`（冻结）、本报告段落见 `REPORT_QWEN_PILOT.md`。
- **评测代码（可复现，非待运行）**：`run_qwen.py`（全网格 runner，支持 `--model 2b/7b`）、`evaluate_predictions.py`（AUROC 汇总）、`smoke/gpu_smoke.py`（pilot smoke）。这些是已写好的工具，因 pilot 无区分度而**未按全量参数执行**；保留仅为可复现与方法描述。
- **一次性环境/数据搭建脚本（已完成，非实验）**：`smoke/` 下 `build_data.sh`（重建 1cls）、`verify_refs.sh`（72/72 一致性校验）、`install_pkgs.sh`（装依赖）、`download_model_server.sh`/`download_7b.sh`/`fetch_shard1.sh`（下模型）、`ssh_run.py`/`ssh_exec.py`/`ssh_put.py`/`upload_model.py`（远程执行/传输）、`auto_smoke.sh`/`auto_smoke_7b.sh`/`wait_pull_7b.py`（等待下载+自动跑 smoke 的编排）。这些是工程脚手架，全部已执行完毕。
- **合成图**：`smoke/synth/*.png` 仅用于最初 CPU 可行性 smoke，不含真实数据。
- 运行成本：本地 GPU，无货币成本；2B pilot 2.7s/图、7B pilot 2.7s/图，均远快于 CPU（37–72s/图）。不含 11,520 全网格（未运行）。
