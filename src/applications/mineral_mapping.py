"""
src/applications/mineral_mapping.py
--------------------------------------
Mineral / lithology mapping via continuum removal (Clark & Roush, 1984) --
the standard technique in imaging-spectroscopy mineral exploration (the same
family of methods used by AVIRIS/PRISMA/EnMAP mineral-mapping products, and
directly relevant to Pixxel's stated mining-exploration use case).

Continuum removal isolates a diagnostic absorption feature by:
  1. fitting a "continuum" -- the convex hull / straight line connecting the
     local reflectance maxima on either side of the absorption feature;
  2. dividing the observed spectrum by that continuum, so the featureless
     background reflectance becomes ~1.0 everywhere and only the absorption
     dip remains visible;
  3. the *depth* of that normalized dip (1 - min) is a mineral-abundance-
     correlated index that is far more diagnostic than a raw two-band ratio,
     because it is normalized against the local (possibly sloped, mineral-
     dependent) background reflectance.

Two features are mapped, matching the endmembers in
`src/data/spectral_library.py`:
    - Iron-oxide charge-transfer absorption near 900 nm (VNIR -- visible on
      *both* Firefly and Honeybee).
    - Al-OH clay-mineral doublet absorption near 2200 nm (SWIR-only --
      visible *only* in Honeybee mode; deliberately returns all-zero on
      Firefly, which is itself an instructive result about why SWIR
      coverage matters for exploration geology).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def _band_window(wavelengths_nm: np.ndarray, center_nm: float, half_width_nm: float) -> Tuple[int, int]:
    lo = np.searchsorted(wavelengths_nm, center_nm - half_width_nm)
    hi = np.searchsorted(wavelengths_nm, center_nm + half_width_nm)
    return int(max(lo, 0)), int(min(hi, len(wavelengths_nm) - 1))


def continuum_removed_depth(cube: np.ndarray, wavelengths_nm: np.ndarray,
                             center_nm: float, half_width_nm: float) -> np.ndarray:
    """
    For every pixel, fit a straight-line continuum between the two edge
    wavelengths of the window and return the maximum normalized absorption
    depth (1 - R/continuum) within that window.
    """
    lo, hi = _band_window(wavelengths_nm, center_nm, half_width_nm)
    if hi - lo < 3:
        return np.zeros(cube.shape[:2], dtype=np.float32)

    wl_window = wavelengths_nm[lo:hi + 1]
    spec_window = cube[..., lo:hi + 1].astype(np.float64)  # (H, W, w)

    r_start = spec_window[..., 0:1]
    r_end = spec_window[..., -1:]
    frac = ((wl_window - wl_window[0]) / (wl_window[-1] - wl_window[0] + 1e-9))[None, None, :]
    continuum = r_start + frac * (r_end - r_start)

    ratio = spec_window / np.clip(continuum, 1e-6, None)
    depth = 1.0 - ratio.min(axis=-1)
    return np.clip(depth, 0, None).astype(np.float32)


def map_iron_oxide(cube: np.ndarray, wavelengths_nm: np.ndarray, center_nm: float = 900.0) -> np.ndarray:
    return continuum_removed_depth(cube, wavelengths_nm, center_nm, half_width_nm=80.0)


def map_clay_mineral(cube: np.ndarray, wavelengths_nm: np.ndarray, center_nm: float = 2200.0) -> np.ndarray:
    if wavelengths_nm.max() < center_nm + 50:
        # SWIR not available in this sensor mode (e.g. Firefly VNIR-only) --
        # the clay Al-OH doublet is simply not observable, so we report that
        # explicitly rather than fabricating a number.
        return np.zeros(cube.shape[:2], dtype=np.float32)
    return continuum_removed_depth(cube, wavelengths_nm, center_nm, half_width_nm=90.0)
