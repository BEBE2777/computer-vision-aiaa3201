# Part 3 plan script — multi-layer DINOv2 feature fusion (~35 s, 95 words)

**Status note for the narrator:** this section describes a *plan only*. No
Part 3 experiment has been run yet; the cached grid runner's numbers must not
be quoted as verified Part 3 conclusions (its scoring still differs from the
official pipeline — see `HANDOFF.md`).

---

Our Part One template baseline is highly sensitive to object position,
background variation, and alignment. And AnomalyDINO scores with only the
final-layer features of DINOv2, which may miss small local textures. Our
planned improvement is multi-layer feature fusion: intermediate transformer
layers capture fine local texture, while the final layer contributes semantic
stability; we combine them with per-layer normalisation. We will compare
against unmodified Part Two on the same VisA data, the same reference
filenames, and the same shots and seeds, reporting image AUROC, pixel AUROC,
runtime, and GPU memory. To be clear: this is a plan, not a conclusion.

## Suggested visuals

- A simple two-track diagram: "intermediate layers → local texture" and
  "final layer → semantic stability" merging into one fused memory bank.
- Explicit caption: **"Plan only — no experimental numbers yet."**
