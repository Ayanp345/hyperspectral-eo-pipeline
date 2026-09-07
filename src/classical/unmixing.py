"""
src/classical/unmixing.py
---------------------------
Linear spectral unmixing under the Linear Mixing Model (LMM):

    pixel_spectrum ≈ sum_k( abundance_k * endmember_k )   s.t.  sum_k abundance_k = 1,  abundance_k >= 0

This is the same generative model `src/data/synthetic_hsi.py` used to build
the scene, so unmixing recovers the ground-truth per-pixel class abundances
(sub-pixel composition) -- the hyperspectral capability that is impossible
with ordinary RGB/multispectral imagery and is one of the main reasons
Pixxel's customers (agriculture, mining) want *hyperspectral*, not just
higher-resolution, imagery: two adjacent 5 m pixels of "90% healthy crop /
10% soil" and "60% healthy crop / 40% stressed crop" look identical in RGB
but are spectrally distinguishable.

We solve the Fully Constrained Least Squares (FCLS) problem approximately via
non-negative least squares (NNLS) followed by renormalization -- a standard,
fast approximation to full FCLS used widely in practice.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls


def unmix_pixel(spectrum: np.ndarray, endmembers: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    spectrum   : (B,)
    endmembers : (K, B)

    Returns
    -------
    abundances : (K,) non-negative, sums to 1.
    """
    sol, _ = nnls(endmembers.T, spectrum)
    total = sol.sum()
    if total <= 1e-9:
        return np.ones_like(sol) / len(sol)
    return sol / total


def unmix_cube(cube: np.ndarray, endmembers: np.ndarray, stride: int = 1) -> np.ndarray:
    """
    Per-pixel FCLS-style unmixing (NNLS solved independently per pixel, which
    is the standard, embarrassingly-parallel way FCLS is implemented -- there
    is no closed form for the inequality-constrained problem).

    Parameters
    ----------
    cube       : (H, W, B)
    endmembers : (K, B)
    stride     : spatial stride for a fast demo pass (e.g. stride=2 unmixes
                 every other pixel in each direction, then nearest-neighbour
                 upsamples the result 4x faster) -- set to 1 for a full,
                 research-grade per-pixel solve.

    Returns
    -------
    abundance_maps : (H, W, K)
    """
    h, w, b = cube.shape
    k = endmembers.shape[0]

    sub = cube[::stride, ::stride, :]
    sh, sw, _ = sub.shape
    flat = sub.reshape(-1, b)
    out = np.empty((flat.shape[0], k), dtype=np.float32)
    for i in range(flat.shape[0]):
        out[i] = unmix_pixel(flat[i], endmembers)
    small_maps = out.reshape(sh, sw, k)

    if stride == 1:
        return small_maps

    # nearest-neighbour upsample back to (H, W, K)
    row_idx = np.clip(np.arange(h) // stride, 0, sh - 1)
    col_idx = np.clip(np.arange(w) // stride, 0, sw - 1)
    return small_maps[row_idx][:, col_idx]


def unmixing_rmse(estimated_abundance: np.ndarray, true_abundance: np.ndarray) -> float:
    """Root-mean-squared error between estimated and ground-truth abundance maps."""
    return float(np.sqrt(np.mean((estimated_abundance - true_abundance) ** 2)))
