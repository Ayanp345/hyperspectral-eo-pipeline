"""
src/preprocessing/atmospheric.py
----------------------------------
Stage 2 of the processing chain: at-sensor radiance -> surface reflectance
("atmospheric correction").

What Pixxel actually does (documented publicly):
    Pixxel runs "piSOFIT", their customized implementation of NASA JPL's
    open-source ISOFIT (Imaging Spectrometer Optimal FITting) model. ISOFIT
    is an *optimal-estimation* atmospheric correction: it jointly inverts a
    radiative-transfer model (e.g. MODTRAN/6S-derived lookup tables) together
    with a surface-reflectance prior, solving for the most likely atmospheric
    state (water vapor, aerosol optical depth, etc.) and surface reflectance
    simultaneously via non-linear least squares / Bayesian optimal estimation.

What this module does instead, and why:
    A full radiative-transfer optimal-estimation inversion needs external
    atmospheric radiative-transfer lookup tables (MODTRAN / 6SV) that are not
    available in this environment. Reproducing the *real* piSOFIT/ISOFIT
    algorithm is out of scope for a self-contained educational project, so
    this module implements the classical, well-documented **empirical**
    approach used for decades before optimal-estimation methods became
    practical, split into exactly the two unknowns a real correction has to
    solve for:

      1. Dark-Object Subtraction (DOS) -- Chavez (1988): the darkest pixels in
         a scene should be near-zero reflectance for opaque surfaces (deep
         water, shadow); any residual signal there is attributed to additive
         atmospheric path radiance/scattering, and is estimated (NOT assumed
         known) per band as a low percentile of the scene's radiance.
      2. Given that path-radiance estimate, invert the same physical
         radiance = reflectance * solar_irradiance * cos(sun_zenith) / pi
                    + path_radiance
         forward equation used by the simulator (`src/physics.py`) for
         reflectance, treating the solar irradiance spectrum and illumination
         geometry as *known* quantities -- exactly as real algorithms do,
         since the solar spectrum is a well-measured external reference, not
         something retrieved per-scene.

    This is explicitly a *simplified stand-in* for piSOFIT/ISOFIT (it omits
    gaseous absorption/transmittance retrieval and aerosol scattering
    entirely) -- but unlike a naive per-band contrast stretch, it correctly
    preserves cross-band spectral *shape*, which every downstream classical
    algorithm (SAM, matched filter, unmixing) depends on. The interface
    (`atmospheric_correct(radiance, wavelengths) -> reflectance`) is written
    so a real ISOFIT call could be dropped in later without touching any
    other module.
"""
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
