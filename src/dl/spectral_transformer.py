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
