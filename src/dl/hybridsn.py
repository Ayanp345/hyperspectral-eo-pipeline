from __future__ import annotations

import torch
import torch.nn as nn


class HybridSN(nn.Module):
    def __init__(self, num_classes: int, pca_components: int = 30, patch_size: int = 9,
                 dropout: float = 0.4):
        super().__init__()

        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(7, 3, 3)), nn.BatchNorm3d(8), nn.ReLU(inplace=True),
        )
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=(5, 3, 3)), nn.BatchNorm3d(16), nn.ReLU(inplace=True),
        )
        self.conv3d_3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3)), nn.BatchNorm3d(32), nn.ReLU(inplace=True),
        )

        # -- Work out how many spectral "slabs" survive the three 3D convs so
        #    we know the input channel count for the 2D conv (channels =
        #    32 output feature maps * remaining spectral depth). Run this
        #    probe in eval() mode so BatchNorm uses its (safe, deterministic)
        #    running statistics instead of trying to compute batch statistics
        #    from a single dummy sample.
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 1, pca_components, patch_size, patch_size)
            feat = self.conv3d_3(self.conv3d_2(self.conv3d_1(dummy)))
            _, c3d, depth, h3d, w3d = feat.shape
            conv2d_in_channels = c3d * depth
            flat_input_shape = feat.reshape(1, conv2d_in_channels, h3d, w3d)

        self.conv2d = nn.Sequential(
            nn.Conv2d(conv2d_in_channels, 64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )

        self.eval()  # re-assert eval mode now that conv2d (a fresh, train-mode submodule) exists
        with torch.no_grad():
            feat2d = self.conv2d(flat_input_shape)
            flatten_dim = feat2d.numel()
        self.train()  # restore default training mode for normal use

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        self._conv2d_in_channels = conv2d_in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 1, pca_components, patch, patch) -> logits (batch, num_classes)."""
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)
        x = self.conv3d_3(x)                       # (B, 32, depth', H', W')
        b, c, d, h, w = x.shape
        x = x.reshape(b, c * d, h, w)               # fold spectral depth into channels
        x = self.conv2d(x)
        return self.classifier(x)
