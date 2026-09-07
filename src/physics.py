"""
src/physics.py
-----------------
Small shared radiative-transfer building blocks used by BOTH:
  - the forward simulator (`src/data/synthetic_hsi.py`, reflectance -> radiance)
  - the inverse atmospheric correction (`src/preprocessing/atmospheric.py`,
    radiance -> reflectance)

This mirrors how real atmospheric correction actually works: the top-of-
atmosphere solar irradiance spectrum is a well-measured, *known* physical
quantity (published reference spectra like Thuillier 2003 / ChKur are simply
looked up, not solved for). What a real atmospheric correction algorithm
(e.g. Pixxel's piSOFIT/ISOFIT) actually estimates is the *unknown* atmospheric
state -- water vapor, aerosol optical depth, and the resulting path radiance
and gas transmittance -- via an optimal-estimation inversion against a
radiative-transfer lookup table (MODTRAN/6S).

Keeping the solar irradiance model here as a single shared function is the
honest way to reflect that division of labor in this simplified educational
pipeline: the simulator and the corrector agree on the "known" solar physics,
while the correction module (`atmospheric.py`) still has to *estimate* the
unknown additive path-radiance term itself (via Dark Object Subtraction) --
it is not simply handed the answer.
"""
from __future__ import annotations

import numpy as np


def solar_irradiance_curve(wavelengths_nm: np.ndarray) -> np.ndarray:
    """
    A smooth, illustrative top-of-atmosphere solar spectral irradiance curve
    (peaks in the visible, decays through NIR/SWIR). Not a metrology-grade
    solar spectrum -- just physically-shaped enough that reflectance <->
    radiance conversion behaves sensibly across the VNIR-SWIR range.
    """
    wl = wavelengths_nm
    e0 = np.exp(-0.5 * ((wl - 500.0) / 400.0) ** 2)
    decay = np.exp(-np.clip(wl - 500, 0, None) / 1800.0)
    return 1500.0 * e0 * decay + 50.0


def rayleigh_path_radiance(wavelengths_nm: np.ndarray) -> np.ndarray:
    """Toy Rayleigh-like path radiance ~ 1/wavelength^4, larger at short wavelengths."""
    wl_um = wavelengths_nm / 1000.0
    return 6.0 / (wl_um ** 4 + 0.05)


def reflectance_to_radiance(reflectance: np.ndarray, wavelengths_nm: np.ndarray,
                             sun_zenith_deg: float, include_path_radiance: bool = True) -> np.ndarray:
    """Forward model: surface reflectance -> at-sensor radiance (no noise)."""
    irr = solar_irradiance_curve(wavelengths_nm)
    cos_sz = np.cos(np.deg2rad(sun_zenith_deg))
    radiance = reflectance * irr[None, None, :] * cos_sz / np.pi
    if include_path_radiance:
        radiance = radiance + rayleigh_path_radiance(wavelengths_nm)[None, None, :]
    return radiance


def radiance_to_reflectance(radiance: np.ndarray, wavelengths_nm: np.ndarray,
                             sun_zenith_deg: float, path_radiance: np.ndarray) -> np.ndarray:
    """
    Inverse model: at-sensor radiance -> surface reflectance, GIVEN a known
    solar irradiance curve/geometry and an *estimated* path radiance (the
    only unknown a real algorithm has to solve for; here it comes from Dark
    Object Subtraction in `atmospheric.py`).
    """
    irr = solar_irradiance_curve(wavelengths_nm)
    cos_sz = np.cos(np.deg2rad(sun_zenith_deg))
    numerator = radiance - path_radiance[None, None, :]
    denominator = (irr[None, None, :] * cos_sz / np.pi)
    reflectance = numerator / np.clip(denominator, 1e-6, None)
    return reflectance
