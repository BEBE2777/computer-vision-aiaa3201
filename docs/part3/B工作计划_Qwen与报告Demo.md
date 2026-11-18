# B 工作计划：Qwen 独立实验、Related Work 与 Demo

## 目标和边界

B 独立完成一条多模态异常检测实验线：给 Qwen-VL 一组正常参考图和一张测试图，让模型判断测试图是否异常，并输出可评价的连续异常分数。B 负责该分支的代码、运行、预测、分析、成本记录；撰写报告中的多模态 Related Work 与 Qwen Experiments；制作 Project 2 brief 要求的 Demo 录屏和 `anomaly_demo.zip`。

实验不依赖 A 的 CLIP 特征或 ROI 输出，因此可独立开工。CLIP 与 Qwen 完成后，两人再比较结果，决定报告重点讲哪一个候选改进。选中者可称为“本次探索中表现最好的候选”；如果用同一测试集挑选，必须披露选择过程，不声称该测试集上的赢家经过独立验证。另一条分支及 Part 2 DINO 基线仍要在表中报告。

课程 brief 的硬要求：报告使用 CVPR LaTeX 模板，正文 6–8 页（参考文献不计）；Related Work 覆盖推荐阅读清单；Abstract 末尾放公开 GitHub 链接。Project 2 专项 brief 要求 runnable demo 或录制演示，并把视频和代表输出装入 `anomaly_demo.zip`。Canvas 截止 12 月 11 日 11:59 PM，年份及时区以课程页面为准。

## 开始前的代码与状态审计

先检查当前分支和工作区，不覆盖他人改动。阅读：

1. `docs/part3/实验规划与两人分工.md`
2. `docs/part3/后续执行完整Prompt.md`
3. `p2-anomaly/README.md`
4. `results/part2/README.md`、汇总 JSON/CSV 和已归档 refs
5. `p2-anomaly/src/p2common.py`、`p2-anomaly/eval_maps.py`
6. `Project_2__Few_Shot_Industrial_Visual_Anomaly_Detection.pdf`

先给用户一份短状态报告：Qwen 权重/推理方式是否可用、图片和 refs 在哪里、每类图数能否完整覆盖、是否有 GPU/API 预算、当前分支是否干净。历史服务器路径与 `/tmp` 状态都需重新核实，不能当作当前有效路径。

## 实验协议

- 数据：VisA 官方 1cls split 六类 `candle, capsules, cashew, chewinggum, fryum, pipe_fryum`。
- shots `{1,2,4,8}`、reference seeds `{0,1,2}`。直接使用仓库归档 refs，不自行重新抽样。
- query 是所有六类测试图；同一图片在所有方法中使用同一 ID。类别标签、good/bad 路径、mask、GT、文件名中的真值暗示不得输入模型。
- 每次 query 独立上下文；只给类别名、该 cell 指定的 n 张正常 refs、query 和冻结 prompt。0-shot 可作独立额外行，不能冒称 few-shot。
- 主要指标 Image AUROC：每类分别计算，再六类宏平均；seed 汇总按项目现有口径。Qwen 不产生可信像素图时不报告 Pixel AUROC，也不把图像级分数涂成热力图。
- 如用每类留出的 20 张正常校准图校准分数或阈值，报告明确写 `n-shot references + 20 normal calibration images/category`。这些图不能进入模型参考上下文或 memory bank。
- 不用测试标签改 prompt、阈值、模型或解码参数。若试点结果用于选 prompt/模型，试点图片标记为开发数据；正式比较对所有方法使用同一未参与选择的评测子集。若无法隔离，结果标注为探索性。

## 推荐实现流程

### B0：模型可行性和 1-cell smoke test

选择一个可实际运行的 Qwen-VL Instruct checkpoint；固定并记录模型 ID、revision、processor、精度/量化、解码参数、输入图像处理、视觉 token 限额及运行设备。先试 1-shot/seed0、六类每类少量 good/bad query，之后试 4-shot 和 8-shot 输入，检查多图顺序、上下文长度、OOM、JSON 有效率、异常分数是否出现正常变化、每图延迟和显存。

如果本地模型太慢或无法部署，评估已有付费 API，但先估算少量调用费用，不得未经用户授权购买服务或消耗付费额度。若预算不够，缩小覆盖范围并同步告诉 A，让 CLIP 采用相同的确认性范围。

**B0 通过条件：**query/reference 顺序稳定；JSON 可解析且失败可记录；模型确实能返回连续分数；输入没有标签泄漏；8-shot 方案可完成或明确决定对所有方法缩小评测范围。分数质量不佳也可继续作为负结果，不能靠在测试集上反复改提示追分。

### B1：冻结 Prompt 与输出 schema

使用一个基础版本，版本名例如 `qwen_inspect_v1`。prompt 固定说明前 n 张是正常参考图、最后一张为待检 query；忽略姿态/光照/背景的无害变化；仅判断 query；证据不足时允许 uncertain；不得臆造缺陷。使用结构化 JSON：

```json
{
  "decision": "OK|NG|UNCERTAIN",
  "anomaly_score": 0.0,
  "confidence": 0.0,
  "defect_type": "unknown",
  "evidence": "short visible evidence",
  "regions": []
}
```

`anomaly_score` 是排序分数，范围 [0,1]，越高越异常；`confidence` 单独保存，不当作校准概率。`regions` 可空。若坐标不可稳定输出，不要求区域字段参与主结果。JSON 解析失败按固定重试策略处理并计入失败率，不能默默丢弃。

Prompt 初稿冻结后，最多做预先声明的少量开发集比较，例如“直接分类”与“对照正常参考图”，之后固定一个版本跑确认性评测。所有版本都保存 prompt 文件和 hash。

### B2：全量或共同范围推理

启动前估算调用总数。完整主网格约 11,520 个测试 query（4 shots × 3 seeds × 960 images），如果还将每个校准正常图作为 query，会增加调用；本分支可以仅在确有归一化需要时使用校准 query，先与 A 统一协议。先产出一个 shot/seed 完整 cell，核对覆盖、耗时与失败后再扩展。

预测必须按 `experiment_id, category, image_id, shot, reference_seed` 保存，文件至少记录：reference_ids、模型 ID/revision、prompt hash、连续分数、decision、regions、status、latency_ms、输入/输出 token（如适用）、峰值 GPU 内存（能测时）、失败原因。另保存原始响应，权限允许时避免记录敏感凭据。缓存键包含图像 hash、参考图及顺序、prompt、模型版本和解码配置。

### B3：统一评价与分析

使用项目评价代码或与 A 确认共同评测入口。检查每类预期测试图数量和每个 shot/seed 覆盖，无缺失时才算完整结果。输出逐类 Image AUROC、六类宏平均、shot/seed 均值与标准差；可补 Image AUPRC、校准阈值下 FPR/FNR、平均/p95 延迟、调用失败率和实际成本。若预测缺失，报告 missing IDs；完整网格不得静默忽略失败样本。

整理至少五个具有代表性的困难案例，按真实错误类型解释：细小纹理、结构/缺件、姿态或照明变化、背景误报等。类别或缺陷分析只用于事后解释，不能回流修改已冻结方法。记录 Qwen 正确、误报、漏报、拒答/格式失败案例；避免只展示成功图。

### B4：Related Work

在 CVPR 报告同一份 Related Work 中负责视觉语言/多模态部分；与 A 共用一篇文献表，不另做重复 essay。核对 Project 2 brief 的 12 条推荐引用全部被报告引用，B 重点撰写 AnomalyCLIP、CLIP/WinCLIP 及相关多模态检查；A 撰写 MVTec、PaDiM、PatchCore、DRAEM、CutPaste、DINOv2、AnomalyDINO、EfficientAD、MVTec AD 2 等传统/视觉异常检测路线，双方共同处理 VisA/SPot-the-Difference 与 Segment Anything 等其余条目。逐项核对作者、年份、标题、出版信息与链接，优先用原论文和官方代码。清楚区分文献既有结论与本项目实验发现。

### B5：必交 Demo

录制 1–2 分钟 Demo：输入一张产品图，展示 Qwen 分支可解释的异常分数/OK-NG，以及实际可用的区域输出（若 Qwen 不稳定输出区域，则展示视觉分支热图并标明热图来源）。加入正常和异常示例，并至少呈现一个真实失败/误判案例。若使用录屏而非 runnable app，展示完整推理流程和输出即可，不声称实时速度，除非已实测。

打包为 `anomaly_demo.zip`，包括 demo 视频、代表性预测结果/图片、简短 README（如何观看、模型来源、该 Demo 对应的方法与限制）。检查压缩包能解开、视频能播放、图片和图中文字清楚。该 Demo 与已完成的 4 分钟中期展示用途不同，不能用中期视频代替。

### B6：报告与最终提交支持

- 写 CVPR LaTeX 报告中的多模态 Related Work、Qwen 方法与实验、失败案例和局限；建议约占正文 2–3 页，由全稿排版决定。
- 报告全文 6–8 页（不含 References），包括 Abstract & Introduction、Related Work、Method、Experiments、Conclusion；Abstract 末尾放公开 GitHub 链接。
- 向 A 交付可直接引用的表格 CSV/LaTeX、定性图、图注、逐类数字与成本说明。
- 完成 `anomaly_demo.zip` 后由 A 做内容复核；最后报告 ZIP 由 B 组装、A 解压检查，项目组共同确认 Canvas 提交回执。

## B 的里程碑与交付目录

| 阶段 | 交付 |
|---|---|
| B0 | `qwen_smoke_report.md`：可运行性、输入限制、耗时、显存/费用、JSON 率、失败情况 |
| B1 | `prompts/qwen_inspect_v1.txt`、模型配置、schema、冻结协议说明 |
| B2 | `results/extensions/qwen/<run_id>/predictions.jsonl`、`raw_responses/` 或安全的响应归档、`failures.jsonl`、`cost_summary.json` |
| B3 | `metrics.json`、逐类/逐 seed 表、五个困难案例图及说明、运行成本表 |
| B4 | Related Work 段落、BibTeX 条目、来源核对表 |
| B5 | `anomaly_demo.zip` 与演示视频 |
| B6 | 可并入主报告的 LaTeX 片段、图表、最终 ZIP 所需 Demo 与材料 |

建议先导报告而后正式扩展，避免过早投入大量 API/GPU 调用。任何时候如果需要缩小 Qwen 网格，应先与 A 对齐共同评测子集，保证 CLIP 对照公平；不能把少量 Qwen 子集和完整 CLIP 网格直接宣称为公平排名。
