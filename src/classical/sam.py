"""
src/classical/sam.py
----------------------
Spectral Angle Mapper (SAM) -- Kruse et al. (1993).

SAM treats each pixel spectrum and each reference (endmember) spectrum as a
vector in B-dimensional space and classifies by the smallest angle between
them, which makes it invariant to multiplicative illumination/albedo scaling
-- a useful property for a physically-motivated first-pass classifier before
resorting to a trained model.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def spectral_angle(pixel_spectra: np.ndarray, reference_spectra: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    pixel_spectra     : (N, B)
    reference_spectra : (K, B)

    Returns
    -------
    angles_deg : (N, K) spectral angle in degrees between every pixel and
                 every reference spectrum.
    """
    p = pixel_spectra / (np.linalg.norm(pixel_spectra, axis=1, keepdims=True) + 1e-12)
    r = reference_spectra / (np.linalg.norm(reference_spectra, axis=1, keepdims=True) + 1e-12)
    cos_theta = np.clip(p @ r.T, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def sam_classify(cube: np.ndarray, reference_spectra: np.ndarray,
                  threshold_deg: float = 8.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classify every pixel in `cube` by nearest spectral angle to one of
    `reference_spectra` (e.g. the endmember library). Pixels whose minimum
    angle exceeds `threshold_deg` are labeled -1 ("unclassified / novel
    material"), which is the standard SAM convention for flagging materials
    not present in the reference library.

    Returns
    -------
    labels       : (H, W) int, class index or -1.
    min_angle_deg : (H, W) float, the winning (smallest) angle, useful as a
                    confidence / purity map.
    """
    h, w, b = cube.shape
    flat = cube.reshape(-1, b)
    angles = spectral_angle(flat, reference_spectra)
    labels = angles.argmin(axis=1)
    min_angle = angles.min(axis=1)
    labels = np.where(min_angle > threshold_deg, -1, labels)
    return labels.reshape(h, w), min_angle.reshape(h, w)
