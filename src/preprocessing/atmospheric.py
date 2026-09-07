from __future__ import annotations

from typing import Tuple

import numpy as np

from src.physics import radiance_to_reflectance


def estimate_path_radiance(radiance: np.ndarray, dark_percentile: float = 0.5) -> np.ndarray:
    """
    Per-band path radiance estimate = the `dark_percentile`-th percentile of
    radiance values across the whole scene, following the Dark Object
    Subtraction (DOS) method (Chavez, 1988).
    """
    h, w, b = radiance.shape
    flat = radiance.reshape(-1, b)
    return np.percentile(flat, dark_percentile, axis=0)


def atmospheric_correct(
    radiance: np.ndarray,
    wavelengths_nm: np.ndarray,
    dark_percentile: float = 0.5,
    sun_zenith_deg: float = 25.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    DOS-estimated path radiance + known-solar-irradiance physical inversion.

    Parameters
    ----------
    radiance        : (H, W, B) at-sensor radiance.
    wavelengths_nm  : (B,) band centers.
    dark_percentile : percentile used to estimate the additive path-radiance
                      floor per band (Dark Object Subtraction).
    sun_zenith_deg  : solar zenith angle at acquisition time -- real EO
                      products carry this in their metadata (from orbit/time
                      geometry), so treating it as known here mirrors
                      operational practice.

    Returns
    -------
    reflectance_est : (H, W, B) float32 estimated surface reflectance.
    path_radiance   : (B,) the estimated per-band path radiance that was
                       subtracted (useful for diagnostics/plots).
    """
    path_radiance = estimate_path_radiance(radiance, dark_percentile)
    reflectance_est = radiance_to_reflectance(radiance, wavelengths_nm, sun_zenith_deg, path_radiance)
    reflectance_est = np.clip(reflectance_est, 0.0, 1.5).astype(np.float32)
    return reflectance_est, path_radiance.astype(np.float32)
