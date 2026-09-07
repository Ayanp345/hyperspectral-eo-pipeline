from __future__ import annotations

from typing import Tuple

import numpy as np


def rx_anomaly_score(cube: np.ndarray, reg_eps: float = 1e-6) -> np.ndarray:
    """
    Parameters
    ----------
    cube : (H, W, B)

    Returns
    -------
    score_map : (H, W) squared Mahalanobis distance (higher = more anomalous).
    """
    h, w, b = cube.shape
    flat = cube.reshape(-1, b)
    mu = flat.mean(axis=0)
    cov = np.cov(flat, rowvar=False) + reg_eps * np.eye(b)
    cov_inv = np.linalg.inv(cov)

    centered = flat - mu[None, :]
    # (x-mu)^T Sigma^-1 (x-mu), computed row-wise
    scores = np.einsum("ij,jk,ik->i", centered, cov_inv, centered)
    return scores.reshape(h, w).astype(np.float32)


def threshold_anomalies(score_map: np.ndarray, percentile: float = 99.0) -> Tuple[np.ndarray, float]:
    """Binarize an RX score map at a given percentile threshold."""
    thresh = np.percentile(score_map, percentile)
    return (score_map >= thresh), float(thresh)
