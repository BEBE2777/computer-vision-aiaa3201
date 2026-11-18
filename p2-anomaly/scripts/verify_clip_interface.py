"""Verification probes for LayerCLIP (E1 interface, run BEFORE freezing protocol).

Checks (CPU-only, safe to run while the DINO grid occupies the GPU):
  V1 shapes: 224 -> 14x14=196 tokens, 448 interp -> 28x28=784 tokens
  V2 raster grid order: shifting the image right by one patch must shift the
      best-matching patch features one column right (row-major order)
  V3 native 224: patch-token mean + ln_post/proj pipeline reproduces the
      official global embedding space (cosine of mean-patch vs encode_image)
  V4 pos-emb interpolation: 448 global embedding stays close to 224's
  V5 determinism: repeated extraction is bit-identical
"""
import os
import sys

import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from clip_core import LayerCLIP  # noqa: E402

ROOT = "/tmp/slv176/data/VisA_pytorch/1cls"
DL = "/data/disk1/home/slv176/models/clip"

m224 = LayerCLIP("ViT-B/16", device="cpu", resolution=224, layers=(12,),
                 pos_embedding_mode="native", download_root=DL)
m448 = LayerCLIP("ViT-B/16", device="cpu", resolution=448, layers=(12,),
                 pos_embedding_mode="interp", download_root=DL)

img = cv2.cvtColor(cv2.imread(os.path.join(ROOT, "capsules", "test", "bad", "000.JPG"),
                              cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

# ---- V1 shapes
t224, g224 = m224.prepare_image(img)
f224 = m224.extract_layers(t224)[-1]
t448, g448 = m448.prepare_image(img)
f448 = m448.extract_layers(t448)[-1]
print(f"V1 shapes: 224 -> feats {f224.shape} grid {g224}; "
      f"448 -> feats {f448.shape} grid {g448}")
assert f224.shape == (196, 512) and f448.shape == (784, 512)
print("V1 PASS")

# ---- V2 raster order: shift image right by one patch (16 px), compare
shifted = np.roll(img, 16, axis=1)
ts, _ = m224.prepare_image(shifted)
fs = m224.extract_layers(ts)[-1]
g = 14
ok, n = 0, 0
for i in range(g):
    for j in range(1, g):  # shifted-out column j=0 wraps; start at 1
        a = fs[i * g + j]
        d_same = np.linalg.norm(a - f224[i * g + j])
        d_left = np.linalg.norm(a - f224[i * g + (j - 1)])
        n += 1
        ok += d_left < d_same
print(f"V2 raster order: {ok}/{n} shifted patches match the LEFT neighbour "
      f"better than the same position")
print("V2 PASS" if ok / n > 0.8 else "V2 FAIL (grid order not row-major?)")

# ---- V3 native-224 pipeline vs official encode_image
with torch.inference_mode():
    glob = m224.model.encode_image(t224.unsqueeze(0)).float().numpy()[0]
mean_patch = f224.mean(axis=0)
cos = float(np.dot(glob, mean_patch) /
            (np.linalg.norm(glob) * np.linalg.norm(mean_patch)))
print(f"V3 cos(mean patch feats, official encode_image) = {cos:.4f}")
print("V3 PASS" if cos > 0.9 else "V3 CHECK (post-processing mismatch?)")

# ---- V4 interpolation sanity
with torch.inference_mode():
    glob448 = m448.model.encode_image(t448.unsqueeze(0)).float().numpy()[0]
cos448 = float(np.dot(glob, glob448) /
               (np.linalg.norm(glob) * np.linalg.norm(glob448)))
print(f"V4 cos(global@224, global@448-interp) = {cos448:.4f}")
print("V4 PASS" if cos448 > 0.85 else "V4 CHECK (interp may distort)")

# ---- V5 determinism
f224b = m224.extract_layers(t224)[-1]
print(f"V5 repeat extraction bit-equal: {np.array_equal(f224, f224b)}")
print("V5 PASS" if np.array_equal(f224, f224b) else "V5 FAIL")

# ---- V6 CPU forward speed (rough, informs feasibility if GPU is busy)
import time
t0 = time.time()
_ = m448.extract_layers(t448)
print(f"V6 CPU 448 forward: {time.time() - t0:.2f}s per image")
