from __future__ import annotations

from typing import Tuple

import numpy as np


def matched_filter_score(cube: np.ndarray, target_signature: np.ndarray,
                          reg_eps: float = 1e-6) -> np.ndarray:
    """
    Parameters
    ----------
    cube             : (H, W, B) radiance or reflectance cube.
    target_signature : (B,) unit target spectral signature (e.g. a trace-gas
                       absorption feature).

    Returns
    -------
    score_map : (H, W) matched-filter score, approximately proportional to
                target concentration / column enhancement.
    """
    h, w, b = cube.shape
    flat = cube.reshape(-1, b)
    mu = flat.mean(axis=0)
    cov = np.cov(flat, rowvar=False) + reg_eps * np.eye(b)
    cov_inv = np.linalg.inv(cov)

    centered = flat - mu[None, :]
    denom = float(target_signature @ cov_inv @ target_signature) + 1e-12
    scores = (centered @ cov_inv @ target_signature) / denom
    return scores.reshape(h, w).astype(np.float32)


def detect_plume(score_map: np.ndarray, percentile: float = 97.0) -> Tuple[np.ndarray, float]:
    """Binarize a matched-filter score map at a percentile threshold."""
    thresh = np.percentile(score_map, percentile)
    return (score_map >= thresh), float(thresh)


def evaluate_detection(pred_mask: np.ndarray, true_mask: np.ndarray) -> dict:
    """Precision / recall / IoU of the predicted plume mask vs. ground truth."""
    pred = pred_mask.astype(bool)
    true = true_mask.astype(bool)
    tp = np.logical_and(pred, true).sum()
    fp = np.logical_and(pred, ~true).sum()
    fn = np.logical_and(~pred, true).sum()
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    iou = tp / (np.logical_or(pred, true).sum() + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return {"precision": float(precision), "recall": float(recall),
            "iou": float(iou), "f1": float(f1)}
