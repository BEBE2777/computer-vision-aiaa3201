# Part 3 essay 初稿（A 段成文，B 段留骨架）

> 状态（2026-11-22）：全文目标 1500–2500 词。A 负责的章节已写成英文正文（约 1700 词，
> 见各节字数标注）；B 负责的多模态一节（§4）只给了必须覆盖的要点骨架，待 B 填写后合并、
> 统一口径。引用键对应 `references.bib`，文献元数据核对记录见 `references_table.md`。
> 规则：文献已证结论、我们的实测、计划中的工作三类话术严格分开；论文数字不与本项目六类
> VisA 表直接排名。

## 中文提纲

1. 引言：少样本工业检测的问题定义与约束（共同，A 起草）
2. 传统路线：无缺陷样本下建模"正常"（A）
3. 基础模型特征与免训练配方（A）
4. 视觉语言 / 多模态路线（**B**）
5. 两类证据是否互补：融合的机制与陷阱（共同，A 起草，含我们的先导实测）
6. 我们的 Part 2 基线与 Part 3 研究问题（A）
7. 实验协议、公平性与资源约束（A）
8. 预期贡献与风险（共同，A 起草）

---

## 1. Introduction *(common, drafted by A, ~260 words)*

Industrial visual inspection asks a narrow question with unforgiving constraints: given a
handful of defect-free images of one product, decide whether a newly captured image contains
a defect, and if possible, where. The constraint that makes the problem research-worthy is
the scarcity of the evidence. Collecting defective samples of a rare failure mode is expensive
by definition — defects are, precisely, the events that almost never happen — and a
production line re-tooled for a new product cannot wait for a labelled corpus. Few-shot
anomaly detection therefore inverts the usual deep-learning recipe: instead of fitting a
model to many labelled examples of a fixed task, it asks how far a *frozen* representation,
learned elsewhere and once, can be pressed into service for a product it has never seen.

This essay reviews the two families of answers the community has given. The first keeps the
problem purely visual: model "normal appearance" from the few reference images and flag
deviations from it, a line that runs from handcrafted template comparison through
statistical patch models to modern self-supervised features [bergmann2019mvtec,
defard2021padim, roth2022patchcore, oquab2024dinov2, damm2025anomalydino]. The second brings
language in: vision-language models let an inspector describe what "defect-free" means in
words and transfer that description across products [radford2021clip, jeong2023winclip,
zhou2024anomalyclip]. Between them lies an open empirical question — which kind of frozen
evidence, visual or linguistic, actually carries the information that separates a subtle
scratch from a shift in pose or lighting — and a systems question of whether the two can be
combined into something better than either alone.

Our project sits exactly at this junction. Part 2 reproduces the strongest training-free
visual recipe, AnomalyDINO [damm2025anomalydino]; Part 3 changes one variable at a time —
the visual encoder (our E1), the addition of a direct multimodal inspector (the Qwen branch),
and finally a calibrated score-level fusion — on the six object-centric categories of VisA
[zou2022spot], under a protocol frozen before any variant is scored.

## 2. The classical route: modelling normality without defects *(A, ~430 words)*

The founding observation of the field is that "normal" can be learned from normal images
alone [bergmann2019mvtec]. MVTec AD fixed the evaluation protocol — image-level detection
plus pixel-level localization against masks — and every dataset that followed, VisA
included [zou2022spot], sharpened it: VisA's ten product categories contain defects that are
subtle at image level (a mis-aligned capsule end, a thin crack), which is why image AUROC on
VisA is much harder than on MVTec AD and why localization quality does not imply detection
quality.

The first family of methods compares a test image against an explicit model of normal
appearance. SPADE [cohen2021spade] retrieves the nearest normal images and aligns local
features; PaDiM [defard2021padim] fits a Gaussian to each patch position's features and
scores by Mahalanobis distance; PatchCore [roth2022patchcore] perfects the nearest-neighbour
variant with a coreset-subsampled memory bank, reporting near-saturated full-shot numbers on
MVTec AD. These methods share one assumption worth stating plainly: the *representation* is
frozen and good, and only the normality model is learned. Their differences — distance
metric, memory budget, per-position statistics — matter, but they are second-order compared
with the choice of representation itself.

The second family synthesizes the defects it cannot collect. CutPaste [li2021cutpaste]
trains a discriminator against self-inflicted, deliberately unrealistic perturbations; DRAEM
[zavrtanik2021draem] reconstructs normal appearance and lets a discriminative head learn on
simulated flaws. Both need a large corpus of the target product, which puts them outside the
few-shot regime by construction; their lesson for us is not their numbers but their use of
synthesis as a *supervision substitute* — the same idea reappears in our protocol only as a
permitted calibration aid, never as a scoring rule fitted on the test distribution.

Efficiency closes the classical line: EfficientAD [batzner2024efficientad] shows that
high-accuracy inspection is compatible with millisecond-level inference, and MVTec AD 2
[hecklerkram2026mvtec2] shows that on transparent, overlapping, badly lit objects current
methods — fast or accurate — still leave enormous headroom. Benchmarks, in other words, are
no longer the bottleneck; representations and decisions are.

What the classical route does *not* tell you is which frozen representation to trust when
only one to eight normal images exist. Full-shot comparisons let a weak representation be
compensated by a strong normality model; at 1-shot there is nothing to compensate with. That
is the gap the foundation-feature line addresses.

## 3. Foundation features and the training-free recipe *(A, ~330 words)*

DINOv2 [oquab2024dinov2] changed the economics of the few-shot regime by making dense,
locally discriminative patch features available without any task-specific training. Its patch
tokens are trained to be comparable across images and positions, which is exactly the
property a nearest-neighbour matcher needs: a defect patch should find no close normal
neighbour, while every legitimate variation of the product should.

AnomalyDINO [damm2025anomalydino] turns this into a deliberately minimal system: extract
DINOv2 patch features from 1–8 normal references, build a memory bank (with rotation
augmentation of the references), score each test patch by its distance to the nearest bank
entry, and aggregate the distance map into an image score by averaging its top 1%. Nothing is
trained; the entire method is the representation plus the matching rule. The authors show
this training-free recipe rivals vision-language few-shot methods that carry far more
machinery — a result we reproduce as our Part 2 and confirm on VisA's six object categories
(94.61 image AUROC at 1-shot under the official configuration).

RegAD [huang2022regad] represents the trained alternative: learn a category-agnostic
registration model once, across many products, and compare registered features at test time.
It earns cross-category generalization through a training phase that our setting forbids, and
we keep it as the reference point for what "trained few-shot" buys and costs.

Two consequences follow for experimental design. First, if the representation dominates, then
*changing the representation while holding the matching rule fixed* is the cleanest
measurement the regime allows — this is our experiment E1, and the reason we compare CLIP's
vision tower against DINOv2 under an identical scoring pipeline rather than comparing
published numbers across datasets. Second, the recipe's simplicity makes its failure modes
interpretable: with no training in the loop, every error traces back to either the features
or the scoring rule, which is what our error taxonomy exploits.

## 4. The vision-language route *(B — skeleton, to be written)*

B 的正文要点（约 400–500 词）：
- CLIP 的对齐机制一句话带过 [radford2021clip]；重点讲异常检测怎么用文本：WinCLIP 的
  normal/abnormal 状态词组合与窗口/patch 聚合、少样本扩展 WinCLIP+ [jeong2023winclip]；
  AnomalyCLIP 的 object-agnostic prompt 学习与跨类别迁移 [zhou2024anomalyclip]。
- 必须交代的边界：这两类方法的报告数字来自 MVTec/VisA 全类别协议，不得与我们六类表排名；
  我们没有复现 WinCLIP，只能称"自定义 CLIP baseline"。
- Qwen 分支（E2）的动机：直接用指令跟随的多模态模型看"n 张正常参考 + 1 张测试图"，
  输出结构化判断与连续分数——把"比较"交给模型而非特征距离；引用一般多模态大模型做
  工业检查的可行性讨论（此段 B 需补 1–2 篇有据可查的文献，不得凭空引）。
- 结尾一句接 §5：两条路线给的是不同类型的证据，问题变成怎么合并。

## 5. Are the two kinds of evidence complementary? *(common, drafted by A, ~330 words)*

Fusion is where the two routes meet, and where the most honest negative results live. We
pre-registered a deliberately boring mechanism before looking at any outcome: each branch's
score is standardized per category with the median and MAD of its scores on 20 held-out
normal calibration images (falling back to the standard deviation, then to a constant, when
the MAD degenerates), and the fused score is α·z₁ + (1−α)·z₂ with the main result at α = 0.5;
the five-point α sweep is reported as a sensitivity analysis, never as a tuned optimum, and
no test label participates in any choice [guo2017calibration].

Three pitfalls deserve as much space as any gain. *Scale*: raw scores from different branches
live on different scales, so un-normalized averaging silently weights one branch; the MAD
normalization makes the weight α mean what it says. *Selection*: picking α on test AUROC
turns the sensitivity curve into an oracle; we report the whole curve and quote the
pre-registered point. *Calibration of the decision*: a threshold that is the nominal 95th
percentile of 20 calibration scores is not a 5% false-alarm guarantee — in our own pilot the
realised false-alarm rate on normal test images was 9–18% depending on the branches, a
concrete instance of why risk–coverage rather than fixed-threshold reporting is the honest
frame [geifman2017selective].

As a machinery check before the multimodal branch is ready, we fused the *two visual* arms of
E1 (DINOv2 and CLIP, 1-shot, seed 0). The pre-registered α = 0.5 fusion reaches 93.75 image
AUROC against 93.11 for the better single arm (+0.64); the sweep peaks at α = 0.75 (94.05).
One cell of one seed proves nothing about deployment — but it does prove the plumbing:
α = 1 and α = 0 reproduce the single branches exactly, the alignment is image-exact, and the
gain lands exactly where the error taxonomy predicted (58 test images that only CLIP judges
correctly).

## 6. Our baseline and Part 3 research questions *(A, ~230 words)*

Our Part 2 is the official AnomalyDINO recipe on VisA's six object-centric categories —
DINOv2 ViT-S/14 at 448 with agnostic preprocessing, 1/2/4/8 references, seeds 0/1/2 —
reproduced to the reported digits and archived with per-image scores, reference manifests and
configuration (94.61 / 95.33 / 96.15 / 96.62 image AUROC across shot counts). Part 1, a
template-difference baseline with ECC registration and SSIM, anchors the low end (52–58) and
demonstrates in the cleanest possible way why representation, not alignment, is the binding
constraint.

Part 3 asks four questions, one per experiment: **E1** — with the scoring rule, references,
resolution and calibration all held fixed, what changes when the frozen encoder is CLIP's
vision tower instead of DINOv2? **E2 (B)** — can a multimodal model inspect a test image
against the same references directly, and does its judgement correlate with either encoder's?
**E3/E4** — do the branches contain complementary errors such that calibrated fusion helps,
and how sensitive is the answer to α? **E5** — does the same fusion rule transfer across the
two visual backbones, or is the gain an accident of one pair? The design principle throughout
is that one rung of the ladder moves at a time.

## 7. Protocol, fairness and resource constraints *(A, ~260 words)*

Every method in this project sees the same data: the six VisA categories' official 1cls
split, the same 960 test images, the same reference images per (shot, seed) drawn as
`pool[seed·shot : (seed+1)·shot]` from the sorted normal training pool, and the same 20
held-out normal calibration images per category used *only* for thresholds and score-scale
estimation — never as references, never as model context. Metrics follow the brief: image
AUROC as the primary metric, per-category and macro-averaged; pixel AUROC only for methods
with a genuine spatial output; mean ± std over the three seeds of the per-seed six-category
mean.

Fairness is tracked at two levels. In the *controlled* comparison (E1), both arms share the
scoring rule, framing, resolution and calibration, so the backbone is the only difference;
the official masked baseline is reported alongside as a complete system, and the ~1.3-point
value of its PCA foreground mask is measured, not assumed. In the *system* comparison
(Part 2 official vs the Qwen branch), heterogeneous preprocessing is allowed but every
deviation — input size, extra views, prompts, decoding — is disclosed, and we report
latency, token/call counts and peak memory alongside accuracy.

Two constraints shaped the protocol. Compute is a shared A6000 with a one-job-at-a-time
policy, which forced serial scheduling and the decision to extract each backbone's features
exactly once; and the calibration budget is fixed at 20 normal images per category, which is
why thresholds are reported as achieved operating points rather than calibrated guarantees.
All settings were frozen from pilot categories before any full-grid run, and no prompt,
weight or threshold was revised after seeing test results.

## 8. Expected contributions and risks *(common, drafted by A, ~200 words)*

The essay's claim, stated in advance: we expect (i) a controlled measurement of the cost of
language alignment in dense anomaly matching — our current evidence says 5.6 image-AUROC
points at 1-shot, not recovered by more references, with one category reversed; (ii) an
independent multimodal branch whose agreement and disagreement with the visual branches are
quantified image-by-image, not anecdotally; (iii) a fusion result with a pre-registered
weight, full sensitivity curve, and an honest account of the realised false-alarm rate;
and (iv) an error taxonomy in which every conclusion is grounded in named, reproducible test
images.

The risks are symmetric and stated up front. Fusion may not beat the stronger branch once
the Qwen branch's scores are calibrated — the taxonomy predicts a small gain bounded by the
"only CLIP right" 6.0% of images, and a null or negative result on E5 would read as "the
gain is an accident of this backbone pair". The multimodal branch may prove unreliable as a
scorer (self-reported scores are not calibrated probabilities); in that case its contribution
is the disagreement signal, not the number. And the whole study runs on one dataset's six
object-centric categories: conclusions about encoders or fusion must be scoped to that
regime, which MVTec AD 2's harder scenarios [hecklerkram2026mvtec2] suggest is the easier
part of the problem space.

---

## 合并前的待办

- [ ] B 填 §4（含 1–2 篇可查证的多模态检查文献）
- [ ] E2/E3/E5 实测数字到位后，§5/§8 更新为最终口径（目前 §5 的 93.75/94.05 是 1-shot
      seed0 的机制验证值，标注为 pilot，不得进主表）
- [ ] 全文词数控制在 1500–2500（当前 A 段约 1700，B 段约 450 预算 → 约 2150）
- [ ] 交叉审阅：A 审 B 的引用与失效处理，B 审 A 的协议表述
