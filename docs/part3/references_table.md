# Part 3 文献表与引用覆盖清单

维护人：A（2026-11-22）。元数据来源分两类：标 `[brief]` 的条目直接采用课程 brief（Project 2 指令 PDF）自带 References 列表的写法；其余条目在 2026-11-22 逐条核对过 arXiv / 出版方页面。BibTeX 见同目录 `references.bib`；引用键以它为准。

## 1. 文献表

| # | 题目 | 作者 / 年份 / 出处 | 是否用异常训练数据或像素标注 | 支持定位 | 与本项目的关系 | 链接 |
|---|---|---|---|---|---|---|
| 1 | MVTec AD — A Comprehensive Real-World Dataset… | Bergmann et al., CVPR 2019 `[brief]` | 否（无监督，仅正常训练） | 是（像素 GT 评测） | 评测协议来源：image + pixel AUROC、正常-only 设定 | [官网](https://www.mvtec.com/company/research/datasets/mvtec-ad) |
| 2 | PaDiM | Defard et al., ICPR-W 2021 `[brief]` | 否 | 是 | patch 特征统计建模路线；与本项目同属"冻结特征 + 正常建模" | [arXiv:2011.08785](https://arxiv.org/abs/2011.08785) |
| 3 | Towards Total Recall (PatchCore) | Roth et al., CVPR 2022 `[brief]` | 否 | 是 | coreset 记忆库 + 最近邻，本项目的 kNN 评分规则的成熟形态 | [arXiv:2106.08265](https://arxiv.org/abs/2106.08265) |
| 4 | DRAEM | Zavrtanik et al., ICCV 2021 `[brief]` | 是（合成缺陷训练） | 是 | 重建/合成路线代表；解释为何 few-shot 场景不与它数值比较 | [arXiv:2108.07610](https://arxiv.org/abs/2108.07610) |
| 5 | CutPaste | Li et al., CVPR 2021 `[brief]` | 是（自生成伪缺陷） | 是 | 合成缺陷思路 → 我们只用于潜在校准/开发集，不作评分规则 | [arXiv:2104.04015](https://arxiv.org/abs/2104.04015) |
| 6 | DINOv2 | Oquab et al., TMLR 2024 `[brief]` | 否（自监督预训练） | —（特征提取器） | Part 2/3 的 DINO 分支骨干；"局部判别性特征"论点的出处 | [arXiv:2304.07193](https://arxiv.org/abs/2304.07193) |
| 7 | AnomalyCLIP | Zhou et al., ICLR 2024 `[brief]` | 否（辅助数据学 prompt，非目标域异常） | 是 | 语言侧路线代表（B 段主写）；E1 取其视觉塔作纯视觉对照的动机 | [arXiv:2310.18961](https://arxiv.org/abs/2310.18961) |
| 8 | AnomalyDINO | Damm et al., WACV 2025, pp.1319–1329 `[brief]` | 否（training-free） | 是 | Part 2 复现对象；E1 的协议来源与被对照系统 | [arXiv:2405.14529](https://arxiv.org/abs/2405.14529) |
| 9 | EfficientAD | Batzner et al., WACV 2024 `[brief]` | 是（正常图像训练） | 是 | 效率上限参照；我们的成本账本（延迟/显存）报告义务的对照 | [arXiv:2301.13367](https://arxiv.org/abs/2301.13367) |
| 10 | MVTec AD 2 | Heckler-Kram et al., IJCV 2026 `[brief]` | 否 | 是 | 说明现有基准饱和、激励更难的协议与可靠性报告 | [arXiv:2503.21622](https://arxiv.org/abs/2503.21622) |
| 11 | SPot-the-Difference (VisA) | Zou et al., ECCV 2022 `[brief]` | 否（自监督预训练 + 正常参考） | 是 | 本项目数据集出处（六类 VisA 1cls split） | [arXiv:2207.14315](https://arxiv.org/abs/2207.14315) |
| 12 | Segment Anything (SAM) | Kirillov et al., ICCV 2023 `[brief]` | 否（大规模分割监督，非异常） | 是 | 背景/前景抑制的可选工具；我们列为 future work，不进对照 | [arXiv:2304.02643](https://arxiv.org/abs/2304.02643) |
| 13 | CLIP | Radford et al., ICML 2021 | 否 | —（特征/相似度） | E1 第二骨干（仅用视觉塔）；语言对齐特征的出处 | [arXiv:2103.00020](https://arxiv.org/abs/2103.00020) |
| 14 | WinCLIP | Jeong et al., CVPR 2023, pp.19606–19616 | 否（零/少样本，正常图给 WinCLIP+） | 是 | patch/窗口 + 文本少样本路线；我们自定义 CLIP baseline 的命名边界 | [arXiv:2303.14814](https://arxiv.org/abs/2303.14814) |
| 15 | SPADE | Cohen & Hoshen, arXiv 2020 | 否 | 是 | kNN 图像级检索异常检测的早期形态 | [arXiv:2005.11653](https://arxiv.org/abs/2005.11653) |
| 16 | RegAD | Huang et al., ECCV 2022, pp.303–319 | 否（跨类别配准代理任务） | 是 | 有训练的 few-shot 路线对照：说明"训练型 few-shot"与我们 training-free 设定的差异 | [arXiv:2207.07361](https://arxiv.org/abs/2207.07361) |
| 17 | On Calibration of Modern Neural Networks | Guo et al., NeurIPS 2017 | — | — | 校准概念出处；支撑"阈值只用校准图 + 报告实际 FPR"的做法 | [arXiv:1706.04599](https://arxiv.org/abs/1706.04599) |
| 18 | Selective Classification for Deep Neural Networks | Geifman & El-Yaniv, NeurIPS 2017 | — | — | risk–coverage 视角；支撑 uncertain/拒判方向（brief Direction F） | [arXiv:1705.08500](https://arxiv.org/abs/1705.08500) |

## 2. 阅读清单覆盖核对（brief 第 4 节 → 引用位置）

| brief 条目 | 引用键 | Related Work 位置 | 报告其他位置 |
|---|---|---|---|
| [1] MVTec AD | `bergmann2019mvtec` | 第 1、2 段 | Experiments 协议段 |
| [2] PaDiM | `defard2021padim` | 第 2 段 | — |
| [3] PatchCore | `roth2022patchcore` | 第 2 段 | Method（kNN 规则） |
| [4] DRAEM | `zavrtanik2021draem` | 第 3 段 | — |
| [5] CutPaste | `li2021cutpaste` | 第 3 段 | 讨论合成缺陷 |
| [6] DINOv2 | `oquab2024dinov2` | 第 4 段 | Method 骨干 |
| [7] AnomalyCLIP | `zhou2024anomalyclip` | 第 5 段（B 段加深） | Part 3 讨论 |
| [8] AnomalyDINO | `damm2025anomalydino` | 第 4 段 | Method / Experiments 全程 |
| [9] EfficientAD | `batzner2024efficientad` | 第 4、6 段 | 成本段 |
| [10] MVTec AD 2 | `hecklerkram2026mvtec2` | 第 1 段 | Conclusion（future work） |
| [11] VisA / SPot | `zou2022spot` | 第 1 段 | Dataset 段 |
| [12] SAM | `kirillov2023sam` | 第 6 段 | future work |

## 3. 命名边界（写作时必须遵守，来自分工文档 §3.3 / §10）

- 我们没有复现 WinCLIP：任何"窗口/文本"的对比只能写成 *自己的 CLIP baseline*，引用 WinCLIP 作方法渊源。
- 论文数字不得与本项目六类 VisA 表直接排名（协议不同）；引用它们只说方向性结论。
- RegAD/DRAEM/CutPaste 属"有训练"路线，只作路线对比，不进数值表。
- AnomalyCLIP 若最终未跑，引用时写"optional extension 未纳入本报告的数值比较"，不冒充结果。
