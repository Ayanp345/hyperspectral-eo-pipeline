from __future__ import annotations

from typing import Tuple

import numpy as np


def _nearest_band(wavelengths_nm: np.ndarray, target_nm: float) -> int:
    return int(np.argmin(np.abs(wavelengths_nm - target_nm)))


def compute_ndvi(cube: np.ndarray, wavelengths_nm: np.ndarray,
                  red_nm: float = 670, nir_nm: float = 860) -> np.ndarray:
    ri = _nearest_band(wavelengths_nm, red_nm)
    ni = _nearest_band(wavelengths_nm, nir_nm)
    red = cube[..., ri].astype(np.float64)
    nir = cube[..., ni].astype(np.float64)
    return (nir - red) / (nir + red + 1e-9)


def compute_red_edge_position(cube: np.ndarray, wavelengths_nm: np.ndarray,
                               vegetation_mask: np.ndarray = None) -> np.ndarray:
    """
    Guyot & Baret (1988) four-point linear REP estimator. Only physically
    meaningful over vegetated pixels (a near-zero R740-R700 denominator over
    non-vegetated, spectrally flat surfaces makes the formula blow up); pass
    `vegetation_mask` to set non-vegetated pixels to NaN rather than an
    arbitrary huge/undefined number.
    """
    i670 = _nearest_band(wavelengths_nm, 670)
    i700 = _nearest_band(wavelengths_nm, 700)
    i740 = _nearest_band(wavelengths_nm, 740)
    i780 = _nearest_band(wavelengths_nm, 780)

    r670 = cube[..., i670].astype(np.float64)
    r700 = cube[..., i700].astype(np.float64)
    r740 = cube[..., i740].astype(np.float64)
    r780 = cube[..., i780].astype(np.float64)

    numerator = (r670 + r780) / 2.0 - r700
    denominator = r740 - r700
    safe_denominator = np.where(np.abs(denominator) < 1e-3, np.nan, denominator)
    rep = 700.0 + 40.0 * (numerator / safe_denominator)
    rep = np.clip(rep, 650.0, 800.0)

    if vegetation_mask is not None:
        rep = np.where(vegetation_mask, rep, np.nan)
    return rep


def classify_vegetation_stress(
    cube: np.ndarray, wavelengths_nm: np.ndarray,
    ndvi_veg_threshold: float = 0.25, rep_stress_threshold_nm: float = 715.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    ndvi        : (H, W)
    rep         : (H, W) red-edge position in nm
    stress_mask : (H, W) bool, True where pixel is vegetated (NDVI above
                  threshold) *and* shows a stress-indicative blue-shifted
                  red-edge position.
    """
    ndvi = compute_ndvi(cube, wavelengths_nm)
    is_vegetation = ndvi > ndvi_veg_threshold
    rep = compute_red_edge_position(cube, wavelengths_nm, vegetation_mask=is_vegetation)
    is_stressed = rep < rep_stress_threshold_nm  # NaN comparisons are False, so non-veg pixels drop out
    stress_mask = np.logical_and(is_vegetation, is_stressed)
    return ndvi, rep, stress_mask
