"""
src/preprocessing/radiometric.py
----------------------------------
Stage 1 of the processing chain: raw sensor digital numbers (DN) -> at-sensor
spectral radiance, using per-band linear calibration coefficients (gain +
offset). This mirrors the first step of any real satellite imaging-spectrometer
pipeline (Level-0 -> Level-1 radiometric calibration), where onboard/pre-flight
calibration coefficients convert raw counts into physical radiance units
(W / m^2 / sr / um).
"""
from __future__ import annotations

import numpy as np


def radiometric_calibrate(dn: np.ndarray, gain: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """
    Convert raw digital numbers to at-sensor radiance.

    Parameters
    ----------
    dn      : (H, W, B) raw digital numbers.
    gain    : (B,) per-band radiometric gain.
    offset  : (B,) per-band radiometric offset.

    Returns
    -------
    radiance : (H, W, B) float32 at-sensor radiance.
    """
    dn = dn.astype(np.float32)
    radiance = dn * gain[None, None, :] + offset[None, None, :]
    return radiance.astype(np.float32)
