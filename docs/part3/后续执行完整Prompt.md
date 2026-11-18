# 可复制的后续执行 Prompt

以下正文可交给后续 Coding Agent 或研究助手。角色 A/B 是两名项目成员的责任分工，不代表要求自动创建两个 AI agent。

---

你是我们的计算机视觉项目研究与实现助手。请基于项目真实状态推进后续实验，不要只给通用建议，也不要把计划写成完成。请先审计本地证据，再开展小规模探索，冻结协议后扩大实验。当前工作目录为 `C:/Users/qzheng/Desktop/computer_vision`，主代码位于 `final_projects/aiaa3201_repo/p2-anomaly`。

一、老师的原始反馈（必须完整回应）

老师说：

“可以 not only dinov2，多尝试其他的模型，像是 qwen，还有其他的 multimodal model；也可以 not traditional feature map，拿去问 GPT AI 等等；fused output two branches，然后去做对比。反正就是我们可以多做点对比，多做点 experiment。”

我们希望你充分 brainstorm，不局限于传统 feature-map 方法。具体研究问题包括：替代视觉表示是否有效？直接问多模态模型能否检查缺陷？局部与语义分支能否互补？局部放大、文本规范和按需复查是否值得探索？我们不预设融合必然优于单分支。

二、项目结构：Part 2 与 Part 3\r\n\r\nPart 2 已完成：官方 AnomalyDINO（DINOv2）基线及其 6 类、4 种 shot、3 个 seed 结果。Part 3 是本次所有后续工作的正式归属，包括替代视觉表示、Qwen/GPT 多模态直接检查、双分支融合、ROI 复查、相关工作 essay 与探索性消融。报告、目录、实验编号、分工和答辩表述都必须按此区分：Part 2 = 已完成基线；Part 3 = 计划并逐步完成的扩展。除非课程正式说明 Part 3 定义不同，不要把这些新实验称为 Part 2 扩展。\r\n\r\n三、现有状态与必须先读的证据

请读以下文件，遇到进度冲突时优先依据结果文件和较新的状态说明，不机械照抄旧 HANDOFF：

1. `final_projects/aiaa3201_repo/results/part2/README.md`
2. `final_projects/aiaa3201_repo/results/part2/summary_all_shots_seeds.csv`
3. `final_projects/aiaa3201_repo/results/part2/refs/`
4. `final_projects/aiaa3201_repo/p2-anomaly/eval_maps.py`
5. `final_projects/aiaa3201_repo/p2-anomaly/src/p2common.py`
6. `final_projects/aiaa3201_repo/p2-anomaly/code/anomalydino/src/detection.py`
7. `final_projects/aiaa3201_repo/p2-anomaly/src/ad_core.py`
8. `final_projects/aiaa3201_repo/HANDOFF.md`
9. `final_projects/ppt_materials_latest/README_FOR_QWEN.md`
10. `final_projects/followup_plan_20261118/实验规划与两人分工.md`

已知状态，仍请核实：

- Part 1 模板差分及部分裁剪探索已有结果，不能称裁剪稳定有效。
- Part 2 官方 AnomalyDINO 已完成六类、1/2/4/8-shot、seed 0/1/2。
- 官方 mean±std，单位 %：1-shot 94.61±0.84 / 98.08±0.05；2-shot 95.33±0.46 / 98.22±0.07；4-shot 96.15±0.40 / 98.36±0.03；8-shot 96.62±0.57 / 98.47±0.01（Image / Pixel AUROC）。
- 配置 DINOv2 ViT-S/14、448、agnostic、FAISS CPU，每类排除 20 张正常校准图。
- cached path 是可选的工程加速方案；历史上与官方路径存在结果差异。可完全绕开它，以官方 DINO 结果作基线；只有复用 cached path 时才需要 parity 检查。
- 旧文件认为官方图像分数可能在平滑/上采样后计算，但本地代码实际是在原始距离图上 top 1% 平均。请定位首个真实分歧，不把旧推测当修复结论。
- CLIP、Qwen/GPT、语义双分支尚无已验证完成的结果。
- 历史服务器是共享 A6000，曾限制一个 GPU 作业；磁盘紧张、部分数据在 `/tmp`、HF 下载不可达。以上均是历史记录，需重新检查实时状态。

四、cached method 的选择说明

cached method 是可选的工程加速实现，不是实验协议的必需部分，也不要求本项目采用。默认建议：保留官方 AnomalyDINO 作为 DINO 基线，直接使用其已完成的结果、配置、refs 及可取得的逐图预测；独立实现并运行 CLIP 与 Qwen 分支，再在相同 image_id 上做融合。若决定复用 cached method，再单独做官方/缓存逐图 parity 并记录差异；不复用就跳过 parity，不要因此阻塞 CLIP/Qwen。若官方逐图预测暂不可得，先导出/重跑官方路径的预测，或明确设计一个公平的新评测，不要假设缓存实现等价。

'五、相关工作检索与 Part 3 essay

老师还建议多查资料并撰写 essay。请把文献调研作为 Part 3 的正式工作包，而非实验做完后的装饰。先检索并阅读原始论文/官方仓库，重点覆盖：few-shot anomaly detection、DINOv2/AnomalyDINO、CLIP/WinCLIP/WinCLIP+、AnomalyCLIP 或其他视觉语言异常检测、多模态大模型用于工业检查、视觉定位/ROI 复查、多分支分数融合与校准。

交付一份约 1500–2500 词英文 essay（若课程有字数要求，以课程要求为准），附中文提纲，结构建议：问题背景；传统局部特征路线及局限；视觉语言/多模态路线；两类信息的互补性与融合；我们 Part 2 基线和 Part 3 研究问题；实验协议、公平性和资源约束；预期贡献与风险。明确哪些是文献已有结果、哪些是我们的实验计划、哪些是我们实测结果。不得把计划写成结论，也不得把不同数据集/协议的论文指标直接与本项目数字排名。

建立文献表，至少包含论文题目、作者/年份、方法、是否使用异常训练数据或像素标注、是否支持定位、与本项目的关系、可靠链接。优先引用同行评审论文、arXiv 原文和官方代码；核对引用元数据，不引用搜索摘要或无法验证的二手说法。essay 初稿在全量实验启动前完成，方便据文献确定比较对象；实验结束后再把真实结果与限制补入讨论。建议成员 A 主写局部视觉/异常检测路线，成员 B 主写多模态/视觉语言路线，两人共同写融合、研究问题与最终讨论。

六、固定协议'

1. 六类：candle、capsules、cashew、chewinggum、fryum、pipe_fryum；shots={1,2,4,8}；reference seeds={0,1,2}。
2. 每个 shot/seed 共 960 张测试图；完整模型网格为 11,520 个 query 配置实例。直接使用归档 refs，不重新采样，也不把不同 shot 静默改成嵌套参考集。
3. 输入模型只包含对应参考图、query、类别名称和事先固定的提示。不得暴露 label、mask、good/bad 路径或缺陷真值。每个 query 独立上下文。
4. 主指标 Image AUROC；空间方法再报 Pixel AUROC。Pixel 使用项目 448×448 协议、正常图全零 mask。输出缺失必须报错/标记，不静默跳过。
5. 先计算每类 AUROC，再每 seed 六类宏平均，再对三个 seed 宏平均算 mean/std；固定 std 口径。VLM 解码重复另用 repeat_id。
6. 每类额外 20 张正常校准图若用于尺度估计/阈值，需披露“n-shot reference + 20 normal calibration images per class”；它们不进入 memory bank 或 VLM 参考上下文。
7. 测试标签不能用来调 prompt、α、模型、阈值或 ROI。默认主融合 α=0.5，α={0,0.25,0.5,0.75,1} 全报告为敏感性分析，不能挑测试最优 α 冒充部署选择。
8. 若查看测试先导子集后调过设计，标记为开发数据，并对确认性比较排除该子集、所有方法重算相同覆盖。或只用训练正常图及其合成诊断缺陷开发，保留完整测试用于冻结后的评测。
9. 分清受控骨干比较与完整系统比较。DINO 的 PCA mask 阈值不是通用预处理；受控实验采用一致、与骨干无关的空间处理，并补对应 DINO 对照。若 CLIP 只能用 224，需加 DINO@224；448 评价网格不等于输入分辨率一致。

四、实验路线与优先级

P0 必做：

- E0：整理并冻结官方 DINO 基线与逐图输出。cached method 可选。只有团队明确选择复用它时，才检查参考图、预处理、旋转、PCA 掩码、特征、L2 归一化、kNN 距离、patch 排布和图像分数的 parity；不复用则跳过此检查，直接从官方路径继续。
- E1：至少一个非 DINO 视觉表示，优先 CLIP 视觉 patch + 相同 kNN。官方 encode_image 不等于 patch map，需要验证 token 提取与尺寸。如果实现阻塞，可用已有 ViTWrapper 先做烟雾测试，明确它还不是完成的公平对比。
- E2：一个 Qwen-VL Instruct 独立语义分支。准确 checkpoint 以可获得性和实测资源决定，固定 model/revision/processor/precision。输入相同 n 张正常图 + query，不提供 DINO 分数或热图。
- E3：DINO 与 Qwen 图像分数融合。
- E4：五个 α 离线敏感性分析；不重复调用模型。

P1 推荐：

- E5：CLIP + 同一个 Qwen 分支 + 同一归一化/α；两个视觉骨干都比较单独与融合，才能检验融合机制是否可迁移。
- E6：Qwen 全图 vs 全图+固定网格 crop vs 全图+DINO top-k ROI。ROI 不得来自 GT。重点看微小缺陷与额外成本。
- E7：有预算时加 GPT 视觉模型，在相同固定子集/参考图/prompt/JSON 下对照。提前估算 20 次调用的耗时和费用，不能无预算展开完整网格。

P2 先 brainstorm，选至多一项额外实现：

- 从当次正常参考图生成文字正常规范，再检查 query；不读取额外参考资源。
- CLIP normal/defective 文本提示作为低成本语义基线。
- 视觉分数靠近正常校准阈值时才调用 VLM，评估性能-调用率-成本曲线，并加随机调用对照。
- 分支分歧后再复查，注意它已经支付首轮两分支成本。
- 正常图上合成可控小缺陷做诊断，不能替代真实测试。
- 参考图顺序/拼图稳定性。
- VLM box + 分割器，明确额外模型、框定位误差与资源。
- 原有 DINO 多层融合只作为 E0 后的辅助消融，不取代多模态路线。

借鉴 WinCLIP/WinCLIP+ 时准确说明是启发还是忠实复现，不把自定义 CLIP-kNN 命名为复现；专用方法若使用额外异常标注/辅助数据，单列资源条件。探索想法未经全面文献核查，不能称原创首创。

七、输出协议与多模态检查提示

每个 query 的 JSON 至少包括：

```json
{
  "decision": "UNCERTAIN",
  "anomaly_score": 0.5,
  "confidence": 0.3,
  "defect_type": "unknown",
  "evidence": "Brief visible evidence only.",
  "regions": []
}
```

以上数字仅展示 schema，不作为默认预测。decision 允许 OK/NG/UNCERTAIN；anomaly_score 与 confidence 分开，二者不被称为经过校准的概率。regions 可为空；有框时采用约定坐标，如 `{bbox_xyxy:[x1,y1,x2,y2],score:0.8}`。归一化 xyxy 的原点为左上、范围 [0,1]、要求 x1<x2、y1<y2；若模型原生坐标不同，由适配器统一换算并验证。

基础英文 inspection prompt：

```text
You are inspecting an industrial product for visible defects.
Product category: {category}.
The first {n} labeled images are normal reference images. The final image is
the QUERY image to inspect. Evaluate the QUERY only.

Compare its surface, shape, structure, and visible components with the normal
references. Account for harmless changes in pose, position, lighting, and
background. Do not require pixel-perfect alignment. Do not invent invisible
defects. Report uncertainty when visible evidence is insufficient.

Return only valid JSON with:
- decision: OK, NG, or UNCERTAIN;
- anomaly_score: a continuous score in [0,1], higher for stronger visible
  evidence of a defect; this is a ranking score, not a calibrated probability;
- confidence: confidence in your assessment, separately in [0,1];
- defect_type: a short descriptive label, or unknown;
- evidence: one short statement of the visible evidence;
- regions: zero or more suspected QUERY regions with bbox_xyxy in normalized
  [0,1] coordinates and a region score. Coordinates refer to the original
  QUERY image, not a reference image. Return [] if no region is supported.

Do not infer the answer from file names, metadata, or category prevalence.
```

runner 按 reference_1…reference_n、QUERY 逐图标注并确认顺序；不要让占位符留在真实 prompt。解释文本不进入 AUROC。独立语义实验不输入视觉异常图。ROI 版本单独版本化，明确 crop 在 query 中的位置、所有框映射回原 query，并保持原 n-shot 信息预算。

八、融合、失败和成本

- 使用正常校准 query 的 median/MAD 估计每分支尺度，预设退化回退策略；方向统一为越大越异常。仅正常数据不能选择异常 AUROC 最优 α。
- `fused = α*z_visual + (1-α)*z_vlm`；主 α=0.5，端点应恢复各分支的类内 AUROC。按 image_id、category、shot、seed 连接，不按行号合并。
- 只融合图像分数，不声称 Pixel AUROC 提升；若仍显示 DINO 热图，明确定位来自 DINO。VLM 框 rasterization 单列为粗定位，不冒充精细分割。
- 不把全图分数涂成热图；正标量调整整图可能影响 pooled pixel AUROC，但不是图内定位改善。
- 保存原始响应、解析结果、配置、prompt hash、model/revision、参考图及顺序、图像 hash、延迟、tokens、状态和失败重试。失败不当作正常，不删难例；固定回退并报告失败率。
- 模型失败、超时、分数高度离散都要保留。API 不可用时不得编造 GPT 结果，以本地 Qwen 继续，并标记覆盖范围。
- 完整主 VLM 测试 11,520 次，加正常校准 query 1,440 次，合计约 12,960 次，另计重试/ROI/repeats。根据先导实测外推耗时和预算。
- 记录参考构建时间、在线时间、端到端时间、mean/p95、GPU 峰值、调用率、tokens 和费用；API 服务端显存不可见写 N/A，不拿客户端显存冒充。缓存后离线融合时间不能冒充全部推理时间。

九、Part 3 两人责任和时间安排

成员 A：官方/缓存 parity、数据与 refs、视觉 adapter/kNN/maps、统一 evaluator、逐图 join/归一化/α 曲线、视觉错例、定量表与复现。

成员 B：Qwen/GPT 模型获取与多图推理、固定 prompt/JSON/失败恢复/缓存、语义全网格、ROI prompt 与按需调用、语义错例和成本表。

共同负责：冻结协议、先导范围与开发隔离、跨骨干融合、每日 15 分钟交接、相互复核、结论与报告。A 供 ROI 坐标，B 做复查；B 供语义预测，A 做融合。不要把一个人仅安排成 PPT 制作。

按约 10 个工作日：D1–2 接口与 parity/VLM smoke；D3–4 先导并冻结方案；D5–7 全网格及离线融合；D8–9 一项关键探索与错误分析；D10 图表、复现和报告。当前没给实际截止日期，这是工作量建议。共享 GPU 先核查再排串行作业，代码和分析可同时进行。运行后立即备份轻量结果，防止 `/tmp` 丢失。

十、Part 3 实施要求与交付

首先输出：证据审计表、已完成/未完成边界、当前可执行命令和依赖、预计调用量、首轮优先级；随后在已获授权的环境中落实最小 smoke test 与模块化实现，不停留在空泛计划。当前任务若仅授权规划，则先交付可审阅规划，不自行购买服务、消耗付费 API 或启动长时 GPU 作业。

实现时保留官方结果，不覆盖历史目录；所有新结果带 experiment_id/config hash。提交以下内容：

1. 冻结协议、refs/query/calibration manifest 与 prediction schema。
2. parity 诊断报告，或使用官方分数的清晰回退方案。
3. 视觉和 VLM runner、融合及统一评价入口。
4. 完整逐图预测、校准预测、原始响应、失败与成本记录。
5. 主表、按类表、shot 曲线、α 曲线、跨骨干融合对照、ROI 对照和成本曲线（未完成项目标记 pending）。
6. 四类分歧案例，既展示被修复的错误，也展示融合新增的错误。
7. 两人进度表、复现方式、中英文答辩解释，严格区分实验事实、解释和计划。

不要承诺必然提升。若语义分支/融合失败，解释互补性不足、尺度不稳或细节丢失等可由证据支持的原因；不能靠反复测试集调参制造提升。我们最终需要可复现、可解释的受控比较与有边界的探索结论。




