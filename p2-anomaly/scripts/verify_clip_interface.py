"""Verification probes for LayerCLIP (E1 interface) — v2, decisive tests.

  V1 shapes: 224 -> 14x14=196 tokens, 448 interp -> 28x28=784 tokens
  V2 raster order (pre-attention, decisive): a white 16x16 block at patch
     (row 5, col 8) must peak at conv1 activation index 5*14+8
  V3 forward-math parity: manual CLS path (pos-emb, ln_pre, 12 blocks,
     ln_post, @proj) must equal official encode_image bit-closely at 224
  V4 interpolation sanity: global embedding via OUR path at 448 vs 224
  V5 determinism: repeated extraction bit-identical
  V6 CPU forward speed
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

# ---- V2 raster order via conv1 (ordering is decided here; attention never reorders)
blk = np.zeros((224, 224, 3), dtype=np.uint8)
blk[5 * 16:(5 + 1) * 16, 8 * 16:(8 + 1) * 16] = 255
tb, _ = m224.prepare_image(blk)
with torch.inference_mode():
    act = m224.visual.conv1(tb.unsqueeze(0))          # (1, width, 14, 14)
act = act.reshape(act.shape[1], -1).numpy()            # (width, 196) raster
patch_act = act.mean(axis=0)
idx = int(np.argmax(patch_act))
expect = 5 * 14 + 8
print(f"V2 white block at patch (5,8): argmax activation index {idx} "
      f"(expect {expect}, row-major)")
top5 = np.argsort(-patch_act)[:5]
print(f"V2 top-5 indices: {sorted(top5.tolist())}")
print("V2 PASS" if idx == expect else "V2 FAIL")

# ---- V3 forward-math parity: manual CLS path vs official encode_image
with torch.inference_mode():
    v = m224.visual
    x = v.conv1(t224.unsqueeze(0))
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    cls = v.class_embedding.to(x.dtype) + torch.zeros(1, 1, x.shape[-1], dtype=x.dtype)
    x = torch.cat([cls, x], dim=1) + v.positional_embedding.to(x.dtype)
    x = v.ln_pre(x).permute(1, 0, 2)
    for blk_ in v.transformer.resblocks:
        x = blk_(x)
    x = x.permute(1, 0, 2)
    cls_out = v.ln_post(x[:, 0, :]) @ v.proj
    official = m224.model.encode_image(t224.unsqueeze(0))
    diff = float((cls_out - official).abs().max())
print(f"V3 manual CLS vs official encode_image: max abs diff = {diff:.3e}")
print("V3 PASS" if diff < 1e-3 else "V3 FAIL (forward math differs)")

# ---- V4 interpolation sanity: global via OUR path at 448 vs 224
with torch.inference_mode():
    v = m448.visual
    x = v.conv1(t448.unsqueeze(0))
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    cls = v.class_embedding.to(x.dtype) + torch.zeros(1, 1, x.shape[-1], dtype=x.dtype)
    x = torch.cat([cls, x], dim=1) + m448._pos_emb_interp.to(x.dtype)
    x = v.ln_pre(x).permute(1, 0, 2)
    for blk_ in v.transformer.resblocks:
        x = blk_(x)
    x = x.permute(1, 0, 2)
    g448_out = (v.ln_post(x[:, 0, :]) @ v.proj).float().numpy()[0]
g224_out = cls_out.float().numpy()[0]
cos448 = float(np.dot(g224_out, g448_out) /
               (np.linalg.norm(g224_out) * np.linalg.norm(g448_out)))
print(f"V4 cos(global@224, global@448-interp, our path) = {cos448:.4f}")
print("V4 PASS" if cos448 > 0.85 else "V4 CHECK (interp may distort)")

# ---- V5 determinism
f224b = m224.extract_layers(t224)[-1]
print(f"V5 repeat extraction bit-equal: {np.array_equal(f224, f224b)}")
print("V5 PASS" if np.array_equal(f224, f224b) else "V5 FAIL")

# ---- V6 CPU forward speed
import time
t0 = time.time()
_ = m448.extract_layers(t448)
print(f"V6 CPU 448 forward: {time.time() - t0:.2f}s per image")
