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
