"""
src/applications/water_quality.py
-------------------------------------
Simple, standard band-ratio water-quality proxy indices -- another of
Pixxel's stated application areas. Both indices below are widely published,
simplified forms used throughout the ocean/inland-water remote sensing
literature; they are illustrative proxies, not calibrated concentration
retrievals (real turbidity/chlorophyll-a retrievals require in-situ
calibration against water samples).

- Normalized Difference Turbidity-proxy Index (NDTI-like):
      NDTI = (R_red - R_green) / (R_red + R_green)
  Suspended sediment raises red-band reflectance relative to green, so higher
  NDTI ~ higher turbidity.

- Blue/Green ratio chlorophyll proxy (OC-type):
      chl_proxy = R_blue / R_green
  Clear, low-chlorophyll water reflects relatively more blue light; as algal
  chlorophyll increases, blue absorption increases and green reflectance
  rises relative to blue, lowering this ratio.
"""
from __future__ import annotations

import numpy as np


def _nearest_band(wavelengths_nm: np.ndarray, target_nm: float) -> int:
    return int(np.argmin(np.abs(wavelengths_nm - target_nm)))


def compute_turbidity_index(cube: np.ndarray, wavelengths_nm: np.ndarray,
                             red_nm: float = 660, green_nm: float = 560) -> np.ndarray:
    ri = _nearest_band(wavelengths_nm, red_nm)
    gi = _nearest_band(wavelengths_nm, green_nm)
    red = cube[..., ri].astype(np.float64)
    green = cube[..., gi].astype(np.float64)
    return (red - green) / (red + green + 1e-9)


def compute_chlorophyll_proxy(cube: np.ndarray, wavelengths_nm: np.ndarray,
                               blue_nm: float = 480, green_nm: float = 560) -> np.ndarray:
    bi = _nearest_band(wavelengths_nm, blue_nm)
    gi = _nearest_band(wavelengths_nm, green_nm)
    blue = cube[..., bi].astype(np.float64)
    green = cube[..., gi].astype(np.float64)
    return blue / (green + 1e-9)
