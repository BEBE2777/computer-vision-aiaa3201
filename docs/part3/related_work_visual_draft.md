# Related Work — visual branch (A's part)

> Status: draft for the report's Related Work section, A's half (traditional vision and
> anomaly detection). English, CVPR style, `\cite` keys resolve against
> `docs/part3/references.bib`. The multimodal / vision-language half is B's; the last
> paragraph hands over to it. Every paper in the brief's Recommended Reading List is cited
> here (coverage checklist in `references_table.md`).

## Related Work

**Industrial anomaly detection and its benchmarks.** Industrial inspection is usually cast as
normal-only learning: the training set contains defect-free images, and a defect is any
deviation from the modelled normal appearance [bergmann2019mvtec]. MVTec AD
[bergmann2019mvtec] established the evaluation protocol of image-level detection plus
pixel-level localization, and later work extended it to larger and more varied data: VisA
[zou2022spot] offers ten product categories with subtle, low-contrast defects (we use its six
object-centric categories), and MVTec AD 2 [hecklerkram2026mvtec2] adds transparent objects
and severe lighting variation to expose the remaining headroom of current methods. Our
protocol follows this line, including the separation of image AUROC (the main metric) from
pixel AUROC, and the use of a held-out set of normal images for score calibration rather than
for model fitting.

**Nearest-neighbour and statistical modelling of patch features.** A first family of methods
compares each test location against a model of normal appearance built from pretrained
features. SPADE [cohen2021spade] scores a test image by the distance to its nearest neighbours
in the training set, with a pixel-level variant that aligns intermediate activations; PaDiM
[defard2021padim] models the per-patch feature distribution as a multivariate Gaussian and
uses Mahalanobis distance; PatchCore [roth2022patchcore] refines the nearest-neighbour recipe
with a coreset-subsampled memory bank and reports near-perfect full-shot results on MVTec AD.
All three inherit the same inductive bias that our Part 2 baseline uses — a *frozen* general
feature extractor plus a normality model that never sees a defect — and differ mainly in what
"normal" is modelled as and how much memory it needs.

**Reconstruction and synthesis.** A second family learns to reconstruct or synthesize normal
appearance and flags what reconstruction fails to explain. CutPaste [li2021cutpaste] trains a
classifier on self-generated, unnatural perturbations; DRAEM [zavrtanik2021draem] pairs a
reconstruction network with a discriminative head on simulated defects. These methods are
effective but need a large training corpus of the very product under inspection, which is the
regime few-shot inspection explicitly excludes; we therefore do not compare against them
numerically, but they motivate the synthetic-defect direction we keep as a possible
calibration aid rather than a scoring rule.

**Foundation-model features and few-shot inspection.** The decisive change for the few-shot
regime is representation quality rather than the normality model. DINOv2 [oquab2024dinov2]
produces dense, locally discriminative patch features from self-supervised pretraining, and
AnomalyDINO [damm2025anomalydino] shows that simply running patch-level nearest neighbours on
those features — training-free, no fine-tuning, no auxiliary data — rivals and often exceeds
vision-language few-shot methods. RegAD [huang2022regad] instead *trains* a
category-agnostic registration model across many categories so that new categories need no
re-training; it demonstrates that few-shot methods can share structure across products, at the
cost of a training phase our setting does not allow. On the systems side, EfficientAD
[batzner2024efficientad] shows that millisecond-level inference is compatible with
high accuracy, which frames the cost accounting we report for our own runs. Our Part 2 is the
official AnomalyDINO recipe; our Part 3 asks whether its conclusion — *good frozen visual
features are enough* — survives a change of the frozen encoder, which AnomalyDINO itself does
not test.

**Language-aligned encoders, used vision-only.** CLIP [radford2021clip] aligns images with
text and dominates zero-/few-shot transfer through prompts: WinCLIP [jeong2023winclip]
aggregates window- and patch-level features against handcrafted normal/abnormal state words,
and AnomalyCLIP [zhou2024anomalyclip] learns object-agnostic prompts so that one text prompt
transfers across product categories. Both operate on the *text* side of the alignment; the
dense quality of CLIP's *vision tower* — what its patch tokens say before any prompt is
applied — is rarely examined. Our E1 fills exactly that gap: we take patch tokens from the
CLIP vision encoder, keep the k-NN scoring rule, resolution, reference sampling and
calibration identical to the DINOv2 control, and measure how much language alignment costs
dense anomaly matching. The result (a 5.6-point image-AUROC deficit at 1-shot that does not
close with more references, with one reversed category) quantifies the trade-off the
prompt-based line implicitly accepts.

**Reliability of the decision, not just the score.** Anomaly scores are only useful through a
threshold, and thresholds inherit the failure modes of probability calibration
[guo2017calibration]. Following the risk–coverage view of selective prediction
[geifman2017selective], we derive every OK/NG threshold from 20 held-out normal calibration
images and report the achieved false-alarm rate rather than assuming the nominal 5%. The gap
we measure between the nominal and realised rates is a concrete instance of why
EfficientAD-style speed and PatchCore-style accuracy are not the whole story: an inspection
system also needs a threshold one can trust. Segmentation prompts such as SAM
[kirillov2023sam] offer a complementary route to background suppression that we discuss as
future work rather than use, because a mask from a general segmenter would add a second
moving part to a comparison designed to keep the backbone the only variable.

**Positioning and hand-over.** In summary, existing work either trains on the target product
[li2021cutpaste, zavrtanik2021draem, huang2022regad], freezes one vision backbone
[damm2025anomalydino, roth2022patchcore], or moves the comparison to the text side of a
vision-language model [jeong2023winclip, zhou2024anomalyclip]. What is missing is a controlled
measurement of *which frozen encoder* dense nearest-neighbour matching should run on, and an
error analysis that says where the encoders disagree rather than only how far apart their
averages are. That is the contribution of our visual branch. The complementary question —
whether a multimodal model asked directly, in language, about reference and test images adds
information that neither encoder's score contains — is reviewed in the next subsection.
