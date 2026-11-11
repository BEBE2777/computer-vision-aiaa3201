#!/usr/bin/env bash
# [P2] torch.hub.load('facebookresearch/dinov2', ...) probes github.com HTML, which this
# server drops. Clone the official repo once and pre-fetch the ViT-S/14 weights from
# dl.fbaipublicfiles.com (reachable), then load with source='local'.
set -eu
WORK=/tmp/slv176
TP=$WORK/third_party
mkdir -p "$TP" "$HOME/.cache/torch/hub/checkpoints"
cd "$TP"

if [ ! -d dinov2 ]; then
  git clone --depth 1 https://github.com/facebookresearch/dinov2.git
fi
ls dinov2/hubconf.py

CK=$HOME/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth
if [ ! -s "$CK" ]; then
  curl -L --retry 5 --retry-delay 3 -C - \
    -o "$CK" https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth
fi
ls -lh "$CK"

"$WORK/venv/bin/python" - <<'PY'
import torch
m = torch.hub.load('/tmp/slv176/third_party/dinov2', 'dinov2_vits14', source='local', trust_repo=True)
m.eval().cuda()
print('DINOV2_OK', sum(p.numel() for p in m.parameters())/1e6, 'M params')
x = torch.randn(1, 3, 448, 448).cuda()
with torch.inference_mode():
    out = m.get_intermediate_layers(x, n=[6, 9, 12])
print('FEAT_SHAPES', [tuple(o.shape) for o in out])
print('patch_size', m.patch_size, 'embed_dim', m.embed_dim)
PY
echo "DINO_READY"
