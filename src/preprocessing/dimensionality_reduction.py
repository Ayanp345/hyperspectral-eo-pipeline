"""
src/preprocessing/dimensionality_reduction.py
------------------------------------------------
Stage 3 of the processing chain: hundreds of narrow, highly-correlated
spectral bands -> a compact, information-dense feature space.

Implements, from scratch (numpy/scipy only, no scikit-learn dependency so the
math is fully transparent):

1. PCA (Principal Component Analysis) via SVD -- orders components by
   *variance*.
2. MNF (Minimum Noise Fraction, Green et al., 1988) -- orders components by
   *signal-to-noise ratio* instead of raw variance, which is the transform
   real hyperspectral / imaging-spectroscopy pipelines generally prefer,
   because a high-variance band is not necessarily a high-information band if
   most of that variance is sensor noise. MNF is the standard technique used
   ahead of spectral unmixing and target/anomaly detection in operational
   imaging-spectrometer processing chains.

MNF algorithm:
    a. Estimate the *noise* covariance Σ_n via the shift-difference method:
       noise is approximated as the pixel-to-pixel spatial difference of the
       image (real ground signal is spatially correlated over a few pixels;
       independent sensor noise is not), so Cov(X - shift(X)) ≈ 2·Σ_n.
    b. Noise-whiten the data: W = U_n Λ_n^(-1/2), where Σ_n = U_n Λ_n U_n^T.
    c. PCA the noise-whitened data -> components are now ranked by
       decreasing signal-to-noise ratio.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def _flatten(cube: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    h, w, b = cube.shape
    return cube.reshape(-1, b), (h, w, b)


def pca_transform(cube: np.ndarray, variance_target: float = 0.995,
                   max_components: int = 40) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Principal Component Analysis via SVD.

    Returns
    -------
    components_img       : (H, W, k) projected image, k chosen to reach
                            `variance_target` cumulative explained variance
                            (capped at `max_components`).
    explained_var_ratio   : (B,) explained variance ratio for *all* components.
    principal_axes        : (B, B) rows are principal axes (eigenvectors),
                             sorted by descending eigenvalue.
    k                      : number of components kept.
    """
    x, (h, w, b) = _flatten(cube)
    mu = x.mean(axis=0, keepdims=True)
    xc = x - mu

    # Economy SVD: xc = U S Vt : columns of V are the principal axes.
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    eigvals = (s ** 2) / (x.shape[0] - 1)
    explained_var_ratio = eigvals / eigvals.sum()

    cumulative = np.cumsum(explained_var_ratio)
    k = int(np.searchsorted(cumulative, variance_target) + 1)
    k = int(np.clip(k, 1, min(max_components, b)))

    principal_axes = vt  # (B, B), row i = i-th principal axis
    projected = xc @ principal_axes[:k].T  # (N, k)
    components_img = projected.reshape(h, w, k).astype(np.float32)

    return components_img, explained_var_ratio.astype(np.float32), principal_axes.astype(np.float32), k


def pca_reduce_fixed(cube: np.ndarray, n_components: int) -> np.ndarray:
    """Same SVD-based PCA as `pca_transform`, but keeping a fixed component
    count rather than a variance target -- used by the deep-learning stage,
    which needs a fixed, known input channel count for its conv layers."""
    x, (h, w, b) = _flatten(cube)
    mu = x.mean(axis=0, keepdims=True)
    xc = x - mu
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    k = int(np.clip(n_components, 1, b))
    projected = xc @ vt[:k].T
    return projected.reshape(h, w, k).astype(np.float32)


def _estimate_noise_covariance(cube: np.ndarray) -> np.ndarray:
    """Shift-difference noise covariance estimate (see module docstring)."""
    horiz_diff = cube[:, 1:, :] - cube[:, :-1, :]
    vert_diff = cube[1:, :, :] - cube[:-1, :, :]
    diffs = np.concatenate(
        [horiz_diff.reshape(-1, cube.shape[-1]), vert_diff.reshape(-1, cube.shape[-1])],
        axis=0,
    )
    cov_diff = np.cov(diffs, rowvar=False)
    return cov_diff / 2.0


def mnf_transform(cube: np.ndarray, n_components: int = 20,
                   eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Minimum Noise Fraction transform.

    Returns
    -------
    mnf_img       : (H, W, n_components) MNF components, ordered by
                     decreasing signal-to-noise ratio (component 1 has the
                     highest SNR / is the "cleanest").
    snr_eigenvalues : (B,) the MNF eigenvalues (~ SNR of each component; a
                     value near 1 or below indicates a component dominated by
                     noise and safe to discard).
    """
    x, (h, w, b) = _flatten(cube)
    mu = x.mean(axis=0, keepdims=True)
    xc = x - mu

    noise_cov = _estimate_noise_covariance(cube)
    noise_cov += eps * np.eye(b)  # numerical regularization
    n_eigval, n_eigvec = np.linalg.eigh(noise_cov)
    n_eigval = np.clip(n_eigval, eps, None)
    whitening = n_eigvec @ np.diag(1.0 / np.sqrt(n_eigval))  # (B, B)

    xw = xc @ whitening  # noise-whitened data
    signal_cov_w = np.cov(xw, rowvar=False)
    s_eigval, s_eigvec = np.linalg.eigh(signal_cov_w)

    # eigh returns ascending order; flip to descending SNR.
    order = np.argsort(s_eigval)[::-1]
    s_eigval = s_eigval[order]
    s_eigvec = s_eigvec[:, order]

    mnf_axes = whitening @ s_eigvec  # (B, B)
    n_components = int(np.clip(n_components, 1, b))
    mnf_proj = xc @ mnf_axes[:, :n_components]
    mnf_img = mnf_proj.reshape(h, w, n_components).astype(np.float32)

    return mnf_img, s_eigval.astype(np.float32)
