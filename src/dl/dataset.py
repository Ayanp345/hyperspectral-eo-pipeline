"""
src/dl/dataset.py
--------------------
Spectral-spatial patch dataset for hyperspectral classification -- the
standard input representation used by essentially every deep HSI
classification paper (HybridSN, SpectralFormer, SSRN, etc.): rather than
classifying one pixel from its spectrum alone, a small spatial neighborhood
(`patch_size` x `patch_size`) around each labeled pixel is extracted, giving
the network local spatial context (texture, edges) alongside the full
spectral signature.

Pipeline:
    1. PCA-reduce the corrected reflectance cube to `pca_components` bands
       (deep spectral-spatial nets are normally run on a PCA-reduced cube --
       this is exactly what the original HybridSN paper does -- both to
       control the 3D-conv parameter count and to denoise).
    2. Pad the cube by patch_size // 2 on each spatial side (reflect padding)
       so every pixel, including scene edges, can form a full patch.
    3. For every labeled pixel, extract a (patch_size, patch_size,
       pca_components) cube -> reshaped to (1, pca_components, patch_size,
       patch_size) for a 3D-CNN, ready to feed HybridSN directly.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - handled by caller via try/except import torch
    torch = None
    Dataset = object


class HSIPatchDataset(Dataset):
    """A torch Dataset yielding (patch, label) pairs for HSI classification."""

    def __init__(self, pca_cube: np.ndarray, label_map: np.ndarray,
                 pixel_indices: np.ndarray, patch_size: int = 9):
        """
        Parameters
        ----------
        pca_cube      : (H, W, C) PCA-reduced reflectance cube.
        label_map     : (H, W) int class labels.
        pixel_indices : (N, 2) array of (row, col) pixel coordinates this
                        dataset instance should serve (i.e. the train/val/test
                        split membership).
        patch_size    : spatial patch side length (odd number recommended).
        """
        if torch is None:
            raise ImportError("PyTorch is required for src.dl.dataset.HSIPatchDataset")

        self.patch_size = patch_size
        self.half = patch_size // 2
        self.pca_cube = np.pad(
            pca_cube, ((self.half, self.half), (self.half, self.half), (0, 0)), mode="reflect"
        )
        self.label_map = label_map
        self.pixel_indices = pixel_indices

    def __len__(self) -> int:
        return len(self.pixel_indices)

    def __getitem__(self, idx: int):
        r, c = self.pixel_indices[idx]
        r_pad, c_pad = r + self.half, c + self.half
        patch = self.pca_cube[
            r_pad - self.half: r_pad + self.half + 1,
            c_pad - self.half: c_pad + self.half + 1, :,
        ]  # (patch, patch, C)
        # -> (1, C, patch, patch) : channel-first, single "grayscale" volume
        # dimension for the 3D convolutions in HybridSN.
        patch = np.transpose(patch, (2, 0, 1))[None, ...].astype(np.float32)
        label = int(self.label_map[r, c])
        return torch.from_numpy(patch), torch.tensor(label, dtype=torch.long)


def stratified_pixel_split(
    label_map: np.ndarray, train_ratio: float = 0.6, val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-class stratified split of pixel coordinates into train/val/test sets,
    matching the standard protocol used in HSI classification benchmarks
    (e.g. Indian Pines papers hold out a fixed % of labeled pixels per class).
    """
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []

    for cls in np.unique(label_map):
        coords = np.argwhere(label_map == cls)
        rng.shuffle(coords)
        n = len(coords)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        train_idx.append(coords[:n_train])
        val_idx.append(coords[n_train:n_train + n_val])
        test_idx.append(coords[n_train + n_val:])

    return (
        np.concatenate(train_idx, axis=0),
        np.concatenate(val_idx, axis=0),
        np.concatenate(test_idx, axis=0),
    )
