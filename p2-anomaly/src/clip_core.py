"""CLIP ViT patch-feature extractor aligned with the ad_core scoring pipeline.

Why this exists (Part 3, E1)
----------------------------
E1 swaps the visual backbone in the few-shot kNN pipeline while keeping every
other protocol element frozen (nearest neighbour, L2-normalised features,
distance, rotation augmentation, scoring rule). Official CLIP's `encode_image`
returns only the global image embedding; dense patch features require calling
the ViT directly. This module extracts final-block patch tokens with the same
post-processing CLIP applies to the global feature (ln_post + projection), so
the feature space stays compatible with text prompts later (E-text).

Interface mirrors `ad_core.LayerDINOv2`:
    model.prepare_image(img)  -> (tensor, grid (h, w))
    model.extract_layers(t)   -> list of patch-feature arrays (one per layer)
    model.compute_background_mask(f, grid, threshold, masking_type)

Differences vs DINOv2 that are handled here:
- CLIP ViTs are trained at fixed resolution (224 for ViT-B/16); larger inputs
  require bicubic interpolation of the positional-embedding grid (CLS row
  kept unmodified). `pos_embedding_mode="native"|"interp"` makes the choice
  explicit so it can be frozen in the protocol.
- Patch tokens are returned WITHOUT the CLS token, in row-major grid order
  (verified in scripts/verify_clip_interface.py).
- No PCA background mask is applied in the controlled comparison; the official
  DINO agnostic preprocessing stays a separate system-level baseline. A
  compute_background_mask stub is provided for parity with the interface.
"""

import numpy as np
import torch
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode


class LayerCLIP:
    """CLIP ViT wrapper exposing patch tokens from selected transformer blocks."""

    # CLIP normalisation (OpenAI constants)
    MEAN = (0.48145466, 0.4578275, 0.40821073)
    STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, model_name="ViT-B/16", device="cuda", resolution=224,
                 layers=(12,), pos_embedding_mode="interp", download_root=None):
        import clip

        assert pos_embedding_mode in ("native", "interp")
        self.clip_model_name = model_name
        self.device = device
        self.resolution = resolution
        self.layers = tuple(layers)
        self.pos_embedding_mode = pos_embedding_mode

        model, _ = clip.load(model_name, device=device, download_root=download_root)
        model.eval()
        self.model = model
        self.visual = model.visual

        # ViT geometry
        self.patch_size = self.visual.conv1.kernel_size[0]  # 16 or 14/32
        self.width = self.visual.conv1.out_channels
        native_res = self.visual.input_resolution
        self.native_grid = native_res // self.patch_size
        self.grid = (resolution // self.patch_size, resolution // self.patch_size)

        # final block index (n_layers of the transformer)
        self.n_blocks = len(self.visual.transformer.resblocks)
        assert all(1 <= l <= self.n_blocks for l in layers), \
            f"CLIP {model_name} exposes {self.n_blocks} blocks"

        if resolution != native_res:
            if pos_embedding_mode != "interp":
                raise ValueError(
                    f"resolution {resolution} != native {native_res} requires "
                    "pos_embedding_mode='interp' (or a native-resolution control run)")
            self._pos_emb_interp = self._build_interp_pos_emb()
        else:
            self._pos_emb_interp = None

        self.normalize = T.Normalize(self.MEAN, self.STD)

    # ------------------------------------------------------------- pos-emb
    def _build_interp_pos_emb(self):
        """Bicubic-interpolated positional embedding for the larger grid.

        Layout: [CLS (1 row) | grid_h*grid_w patches] x width. The CLS row is
        copied unchanged; the patch rows are resized native_grid -> grid.
        """
        pe = self.visual.positional_embedding.detach()  # (1+n_native**2, width)
        cls_pe = pe[:1]
        patch_pe = pe[1:].reshape(self.native_grid, self.native_grid, self.width)
        # (1, width, g, g) for interpolate
        t = patch_pe.permute(2, 0, 1).unsqueeze(0).float()
        gh, gw = self.grid
        t = torch.nn.functional.interpolate(
            t, size=(gh, gw), mode="bicubic", align_corners=False)
        patch_pe_i = t.squeeze(0).permute(1, 2, 0).to(pe.dtype)
        out = torch.cat([cls_pe, patch_pe_i.reshape(gh * gw, self.width)], dim=0)
        return out.to(self.visual.positional_embedding.device)

    # ------------------------------------------------------------- interface
    def prepare_image(self, img_rgb):
        """img_rgb: uint8 HxWx3 RGB numpy -> (float tensor 3xSxS, grid (h, w)).

        Follows the official CLIP preprocessing: resize the shorter edge to
        `resolution` with bicubic interpolation (antialiased), then centre
        crop to a square. The ViT therefore always sees a fixed grid.
        """
        t = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
        _, h, w = t.shape
        scale = self.resolution / min(h, w)
        nh, nw = max(self.resolution, round(h * scale)), max(self.resolution, round(w * scale))
        t = torch.nn.functional.interpolate(
            t.unsqueeze(0), size=(nh, nw), mode="bicubic",
            align_corners=False, antialias=True).squeeze(0)
        top = (nh - self.resolution) // 2
        left = (nw - self.resolution) // 2
        t = t[:, top:top + self.resolution, left:left + self.resolution]
        return self.normalize(t).to(self.device), self.grid

    def extract_layers(self, image_tensor):
        """Return list of (grid_h*grid_w, width) patch-token arrays, one per layer.

        Patch tokens keep row-major grid order; the CLS token is dropped.
        Final-layer tokens are post-processed with ln_post + @proj (the same
        transform CLIP applies to its global embedding), earlier layers get
        raw block outputs.
        """
        with torch.inference_mode():
            x = image_tensor.unsqueeze(0).to(self.device)
            v = self.visual

            x = v.conv1(x)                                   # (1, width, gh, gw)
            gh, gw = x.shape[-2:]
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # raster order
            cls = v.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
            x = torch.cat([cls, x], dim=1)                   # (1, 1+gh*gw, width)
            pe = self._pos_emb_interp if self._pos_emb_interp is not None \
                else v.positional_embedding
            x = x + pe.to(x.dtype)
            x = v.ln_pre(x)

            x = x.permute(1, 0, 2)                           # NLD -> LND
            outs = []
            for li, blk in enumerate(v.transformer.resblocks, start=1):
                x = blk(x)
                if li in self.layers:
                    y = x.permute(1, 0, 2)                   # LND -> NLD
                    patches = y[:, 1:, :]                    # drop CLS
                    if li == self.n_blocks:
                        patches = v.ln_post(patches) @ v.proj
                    outs.append(patches.squeeze(0).float().cpu().numpy())
        return outs

    def encode_image_global(self, image_tensor):
        """Global CLIP embedding (for text-alignment work later)."""
        with torch.inference_mode():
            return self.model.encode_image(image_tensor.unsqueeze(0)).float().cpu().numpy()[0]

    # ------------------------------------------------------------- mask stub
    def compute_background_mask(self, feats, grid, threshold=10, masking_type=True):
        """Kept for interface parity with LayerDINOv2.

        The controlled E1 comparison runs WITHOUT feature-dependent foreground
        masks (protocol 3.3): returning all-True keeps every patch in play.
        If the frozen protocol later adopts a mask, implement it here behind a
        flag and re-verify parity.
        """
        n = grid[0] * grid[1]
        return np.ones(n, dtype=bool)
