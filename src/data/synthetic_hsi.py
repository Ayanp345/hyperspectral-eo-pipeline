"""
src/data/synthetic_hsi.py
--------------------------
Generates a synthetic, physically-motivated hyperspectral scene that mimics
the *structure* of Pixxel's real data products (band count / wavelength
coverage of the Firefly and Honeybee constellations, raw-DN quantization,
sensor noise, spectral "smile", and a SWIR trace-gas plume) so the rest of
the pipeline (preprocessing -> classical spectral analysis -> ML/DL
classification -> application products) has something concrete and
reproducible to run on, end to end, with zero external downloads.

Why synthetic data, and not a real Pixxel scene?
    Pixxel's raw/calibrated imagery is proprietary tasked data delivered to
    paying customers through their API/Aurora platform; it is not a public
    dataset. Using a physics-based simulator that reproduces Firefly/Honeybee's
    real sensor characteristics (band count, spectral range, GSD, noise, SWIR
    gas-absorption physics) lets every stage of this project be demonstrated,
    tested and graded end-to-end without needing access to that data. Swap in
    real tasked imagery (or a public HSI benchmark such as Indian Pines /
    Pavia University / Houston 2018) by writing a loader that returns the same
    `HyperspectralScene` structure -- see `src/data/real_data_loader.py`.

Ground truth returned alongside the cube:
    - `label_map`      : hard per-pixel class id (argmax abundance) — used for
                          classification (ML/DL) benchmarking.
    - `abundance_maps` : per-pixel fractional class membership summing to 1 —
                          used as ground truth for the spectral-unmixing demo.
    - `plume_mask` / `plume_concentration_ppm_m` : synthetic methane plume
                          ground truth — used for the matched-filter /
                          gas-detection demo (only physically detectable in
                          "honeybee" SWIR mode; correctly *invisible* in
                          "firefly" VNIR-only mode, which is itself a useful,
                          realistic teaching point about *why* Pixxel built a
                          second, SWIR-capable constellation).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.config import AttrDict
from src.data.spectral_library import SpectralLibrary
from src.physics import rayleigh_path_radiance, solar_irradiance_curve


@dataclass
class HyperspectralScene:
    wavelengths_nm: np.ndarray            # (B,)
    reflectance: np.ndarray               # (H, W, B) ground-truth surface reflectance in [0,1]
    radiance: np.ndarray                  # (H, W, B) simulated at-sensor radiance
    dn: np.ndarray                        # (H, W, B) simulated raw quantized digital numbers
    calibration_gain: np.ndarray          # (B,) radiometric gain used to build `dn`
    calibration_offset: np.ndarray        # (B,) radiometric offset used to build `dn`
    label_map: np.ndarray                 # (H, W) int class id, argmax of abundance
    abundance_maps: np.ndarray            # (H, W, n_classes) fractional abundance, sums to 1
    class_names: list
    plume_mask: Optional[np.ndarray]      # (H, W) bool, True where synthetic CH4 plume present
    plume_concentration_ppm_m: Optional[np.ndarray]  # (H, W) float column enhancement
    mode: str                             # "firefly" | "honeybee"
    solar_irradiance: np.ndarray          # (B,) illustrative solar irradiance curve


def generate_wavelengths(sensor_cfg: AttrDict) -> np.ndarray:
    """Build the wavelength grid matching the configured Pixxel sensor mode."""
    vnir = np.linspace(sensor_cfg.vnir_range_nm[0], sensor_cfg.vnir_range_nm[1],
                        sensor_cfg.vnir_bands)
    if sensor_cfg.mode == "firefly":
        return vnir
    swir = np.linspace(sensor_cfg.swir_range_nm[0] + 1, sensor_cfg.swir_range_nm[1],
                        sensor_cfg.swir_bands)
    return np.concatenate([vnir, swir])


def _voronoi_abundance_maps(h: int, w: int, n_classes: int, rng: np.random.Generator,
                             n_seeds_per_class: int = 3, softness: float = 3.0) -> np.ndarray:
    """
    Build smooth, sub-pixel-mixed class-abundance maps using an
    inverse-distance-weighted Voronoi-style scheme: several random seed
    points per class are scattered across the scene; each pixel's abundance
    for a class is the softmax-like inverse distance to that class's nearest
    seeds. This produces contiguous, geographically plausible regions with
    smoothly *blended* boundaries -- i.e. realistic mixed pixels -- rather
    than hard rectangular blocks.
    """
    yy, xx = np.mgrid[0:h, 0:w]
    coords = np.stack([yy, xx], axis=-1).astype(np.float64)  # (H, W, 2)

    class_dist = np.zeros((h, w, n_classes), dtype=np.float64)
    for c in range(n_classes):
        seeds = rng.uniform(low=[0, 0], high=[h, w], size=(n_seeds_per_class, 2))
        d = np.stack(
            [np.sqrt(((coords - s) ** 2).sum(axis=-1)) for s in seeds], axis=-1
        ).min(axis=-1)
        class_dist[..., c] = d

    # Convert distances to abundances via a softmin (closer seed -> higher share).
    logits = -class_dist / softness
    logits -= logits.max(axis=-1, keepdims=True)
    weights = np.exp(logits)
    abundance = weights / weights.sum(axis=-1, keepdims=True)
    return abundance


def _apply_spectral_smile(cube: np.ndarray, wavelengths: np.ndarray, max_shift_nm: float = 1.5) -> np.ndarray:
    """
    Simulate the classic hyperspectral-sensor "smile" artifact: the effective
    center wavelength of each band drifts slightly as a function of the
    across-track (column) pixel position, because of off-axis aberration in
    the spectrometer optics. We approximate it by, for each column, resampling
    the spectrum onto a slightly shifted wavelength grid (parabolic shift
    profile peaking at the swath edges) via linear interpolation.
    """
    h, w, b = cube.shape
    col_norm = np.linspace(-1, 1, w)
    shift_profile = max_shift_nm * (col_norm ** 2)  # parabolic, 0 at center, max at edges
    out = np.empty_like(cube)
    for x in range(w):
        shifted_wl = wavelengths + shift_profile[x]
        # interpolate every row (vectorized across H) at once
        out[:, x, :] = np.apply_along_axis(
            lambda row: np.interp(wavelengths, shifted_wl, row), axis=1, arr=cube[:, x, :]
        )
    return out


def generate_scene(cfg: AttrDict) -> HyperspectralScene:
    """
    Build one full synthetic Pixxel-like hyperspectral scene according to the
    project configuration (`configs/config.yaml`).
    """
    sensor_cfg = cfg.sensor
    scene_cfg = cfg.scene
    rng = np.random.default_rng(scene_cfg.seed)

    wl = generate_wavelengths(sensor_cfg)
    lib = SpectralLibrary(wl)
    endmembers = lib.matrix()                       # (n_classes, B)
    n_classes = endmembers.shape[0]

    h, w = scene_cfg.height, scene_cfg.width
    abundance = _voronoi_abundance_maps(h, w, n_classes, rng)
    label_map = abundance.argmax(axis=-1).astype(np.int32)

    # Linear mixing model: each pixel's reflectance is the abundance-weighted
    # sum of endmember reflectances -- the standard model used throughout
    # hyperspectral unmixing literature.
    reflectance = np.tensordot(abundance, endmembers, axes=([2], [0]))  # (H, W, B)
    reflectance = np.clip(reflectance, 1e-4, 1.0)

    # --- Inject a synthetic SWIR methane plume (Honeybee-mode only) --------
    plume_mask = None
    plume_conc = None
    if scene_cfg.inject_methane_plume and sensor_cfg.mode == "honeybee":
        cy, cx = scene_cfg.plume_center
        r = scene_cfg.plume_radius
        yy, xx = np.mgrid[0:h, 0:w]
        dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
        plume_conc = scene_cfg.plume_peak_enhancement_ppm_m * np.exp(-dist2 / (2 * (r ** 2)))
        plume_mask = plume_conc > (0.05 * scene_cfg.plume_peak_enhancement_ppm_m)

        absorb_coeff = lib.methane_absorption_coefficient()          # (B,)
        # small physical scaling constant tuned so peak absorption is a
        # realistic few-percent radiance dip, matching real point-source
        # methane retrievals (which detect *subtle* fractional absorptions).
        k = 3.0e-4
        tau = k * plume_conc[..., None] * absorb_coeff[None, None, :]  # (H, W, B)
        transmittance = np.exp(-tau)
        reflectance = reflectance * transmittance

    # --- Radiative-transfer-ish forward model: reflectance -> radiance -----
    irr = solar_irradiance_curve(wl)
    sun_zenith_deg = float(getattr(scene_cfg, "sun_zenith_deg", 25.0))
    cos_sz = np.cos(np.deg2rad(sun_zenith_deg))
    path_radiance = rayleigh_path_radiance(wl)
    radiance = reflectance * irr[None, None, :] * cos_sz / np.pi + path_radiance[None, None, :]

    if scene_cfg.add_smile_effect:
        radiance = _apply_spectral_smile(radiance, wl)

    # --- Sensor noise: Poisson shot noise + Gaussian read noise -------------
    if scene_cfg.add_shot_noise:
        shot_scale = np.sqrt(np.clip(radiance, 1e-6, None))
        radiance = radiance + rng.normal(0, 1, size=radiance.shape) * shot_scale * 0.02

    snr_linear = 10 ** (scene_cfg.snr_db / 20.0)
    noise_std = radiance.std(axis=(0, 1), keepdims=True) / snr_linear
    radiance_noisy = radiance + rng.normal(0, 1, size=radiance.shape) * noise_std
    radiance_noisy = np.clip(radiance_noisy, 0, None)

    # --- Quantize to raw sensor digital numbers (DN) ------------------------
    max_dn = 2 ** sensor_cfg.bit_depth - 1
    per_band_max = radiance_noisy.max(axis=(0, 1)) + 1e-9
    gain = per_band_max / max_dn                      # DN -> radiance: radiance = dn*gain + offset
    offset = np.zeros_like(gain)
    dn = np.clip(radiance_noisy / gain[None, None, :], 0, max_dn).astype(np.uint16)

    return HyperspectralScene(
        wavelengths_nm=wl,
        reflectance=reflectance.astype(np.float32),
        radiance=radiance_noisy.astype(np.float32),
        dn=dn,
        calibration_gain=gain.astype(np.float32),
        calibration_offset=offset.astype(np.float32),
        label_map=label_map,
        abundance_maps=abundance.astype(np.float32),
        class_names=list(lib.CLASS_NAMES),
        plume_mask=plume_mask,
        plume_concentration_ppm_m=plume_conc.astype(np.float32) if plume_conc is not None else None,
        mode=sensor_cfg.mode,
        solar_irradiance=irr.astype(np.float32),
    )
