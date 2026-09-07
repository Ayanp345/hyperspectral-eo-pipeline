"""
src/dl/spectral_transformer.py
----------------------------------
A Spectral-Spatial Transformer for hyperspectral patch classification,
architecturally inspired by SpectralFormer (Hong, D. et al., 2021,
"SpectralFormer: Rethinking Hyperspectral Image Classification with
Transformers," IEEE TGRS) -- the first major transformer architecture
purpose-built for hyperspectral data, motivated by the same reasoning
DETR/ViT brought to natural images: self-attention can model long-range
dependencies between spectral bands (or spatial positions) that local CNN
kernels only reach after many stacked layers.

Design here:
    1. The spectral dimension (PCA-reduced bands) is split into contiguous
       "spectral groups" of `spectral_group_size` neighboring bands each --
       analogous to how SpectralFormer groups neighboring bands into tokens
       rather than treating every single band as its own token (which would
       make the sequence length equal to the (often large) band count and
       waste attention on near-duplicate adjacent bands).
    2. Each group's small spatial patch across those bands is flattened and
       linearly projected into an embedding -- this is the "tokenization"
       step, directly analogous to ViT's patch embedding, but grouping along
       the *spectral* axis instead of splitting a large image into spatial
       patches.
    3. A learnable [CLS] token + learnable positional embeddings (since
       *spectral order* is physically meaningful, unlike a bag of words) are
       prepended, then a standard pre-norm Transformer encoder
       (multi-head self-attention + MLP blocks) lets every spectral group
       attend to every other one -- e.g. letting a red-edge-region token
       directly attend to a SWIR clay-absorption token, something a small
       CNN kernel could not do without many layers of receptive-field growth.
    4. The [CLS] token's final representation is classified by a linear head.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SpectralGroupEmbed(nn.Module):
    """Flattens each (group_size, patch, patch) spectral-spatial group into one token."""

    def __init__(self, group_size: int, patch_size: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(group_size * patch_size * patch_size, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, num_groups, group_size, H, W)
        b, g, gs, h, w = x.shape
        x = x.reshape(b, g, gs * h * w)
        return self.proj(x)  # (B, num_groups, embed_dim)


class SpectralSpatialTransformer(nn.Module):
    def __init__(self, num_classes: int, pca_components: int = 30, patch_size: int = 9,
                 embed_dim: int = 64, depth: int = 4, num_heads: int = 4,
                 mlp_ratio: float = 2.0, dropout: float = 0.1, spectral_group_size: int = 10):
        super().__init__()
        self.group_size = spectral_group_size
        self.num_groups = math.ceil(pca_components / spectral_group_size)
        self.pad_bands = self.num_groups * spectral_group_size - pca_components

        self.embed = SpectralGroupEmbed(spectral_group_size, patch_size, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_groups + 1, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 1, pca_components, patch, patch) -- the same input format
        `HybridSN` consumes (`src/dl/dataset.py` produces this shape once),
        so both models are drop-in swappable in `src/dl/train.py`.
        """
        if x.dim() == 5:
            x = x.squeeze(1)  # (B, C, H, W)
        b, c, h, w = x.shape

        if self.pad_bands > 0:
            pad = torch.zeros(b, self.pad_bands, h, w, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)

        x = x.reshape(b, self.num_groups, self.group_size, h, w)
        tokens = self.embed(x)                                    # (B, G, D)

        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)                  # (B, G+1, D)
        tokens = tokens + self.pos_embed[:, : tokens.shape[1], :]

        encoded = self.encoder(tokens)
        encoded = self.norm(encoded)
        cls_out = encoded[:, 0]                                   # pooled [CLS] representation
        return self.head(cls_out)
