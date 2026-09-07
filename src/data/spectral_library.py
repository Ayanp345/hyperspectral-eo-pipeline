"""
src/data/spectral_library.py
-----------------------------
A small, physically-motivated spectral reflectance library used to drive the
synthetic hyperspectral scene generator.

Every material is built as a smooth "continuum" reflectance curve with one or
more Gaussian *absorption features* subtracted from it. This is the same
structural model real imaging-spectroscopy scientists use to *describe*
mineral/vegetation spectra (continuum + absorption features), even though the
exact curves here are illustrative, not laboratory-measured (e.g. USGS
splib07 / ECOSTRESS) reflectance spectra.

Wavelength-diagnostic features encoded here (all standard, widely published
remote-sensing facts):

- Vegetation:  chlorophyll absorption ~670 nm, a steep "red edge" rise between
  ~700-750 nm, and a high NIR plateau ~760-900 nm (leaf cell structure
  scattering). Canopy/leaf stress flattens the red edge and lowers the NIR
  plateau -- this is the physiological basis for crop-stress remote sensing.
- Iron oxides (e.g. hematite/goethite-bearing soils): a broad ferric-iron
  charge-transfer absorption centered near 900 nm.
- Clay minerals (e.g. kaolinite/illite-like phyllosilicates): an Al-OH
  absorption doublet near 2160-2220 nm (only resolvable with SWIR bands,
  i.e. Pixxel's Honeybee constellation, not VNIR-only Firefly).
- Water: reflectance increases blue->green then drops steeply through
  NIR/SWIR due to strong liquid-water absorption.
- Methane (CH4): a narrow absorption band near 2298-2312 nm in the SWIR,
  which is the actual spectral region real hyperspectral/imaging-spectrometer
  methane point-source detection algorithms (matched filters) target.

None of this is dual-use or sensitive: it mirrors publicly published remote
sensing science used for agriculture, mining exploration, and greenhouse-gas
monitoring.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


def _gaussian_dip(wl: np.ndarray, center: float, depth: float, width: float) -> np.ndarray:
    """A Gaussian absorption dip to subtract from a continuum reflectance curve."""
    return depth * np.exp(-0.5 * ((wl - center) / width) ** 2)


def _sigmoid(wl: np.ndarray, center: float, steepness: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-steepness * (wl - center)))


@dataclass
class Material:
    name: str
    class_id: int
    color_rgb: Tuple[int, int, int]           # for map visualization only
    reflectance: np.ndarray = field(repr=False)


class SpectralLibrary:
    """Builds per-material reflectance curves over a supplied wavelength grid."""

    #: canonical class ordering used everywhere in the project
    CLASS_NAMES: List[str] = [
        "water",
        "healthy_vegetation",
        "stressed_vegetation",
        "bare_soil",
        "iron_oxide_mineral",
        "clay_mineral",
        "urban_built_up",
    ]

    def __init__(self, wavelengths_nm: np.ndarray):
        self.wl = wavelengths_nm.astype(np.float64)
        self.materials: Dict[str, Material] = {}
        self._build_all()

    # ------------------------------------------------------------------ #
    # individual material spectra
    # ------------------------------------------------------------------ #
    def _water(self) -> np.ndarray:
        wl = self.wl
        # Peak reflectance in blue-green, then a smooth, physically realistic
        # decline through NIR/SWIR from strong liquid-water absorption
        # (continuous everywhere -- no hard cutoff at any single wavelength).
        base = 0.06 * np.exp(-0.5 * ((wl - 490) / 60) ** 2)
        decline_envelope = 1.0 / (1.0 + np.exp((wl - 880) / 25.0))  # smooth sigmoid roll-off
        decline = 0.05 * np.exp(-(wl - 470) / 550.0) * decline_envelope
        r = base + decline
        r = r * np.exp(-np.clip(wl - 900, 0, None) / 250.0)
        return np.clip(r, 0.001, 0.15)

    def _vegetation(self, stressed: bool = False) -> np.ndarray:
        wl = self.wl
        # Green reflectance bump (chlorophyll), red absorption trough,
        # steep red-edge rise, high NIR plateau.
        green_bump = 0.10 * np.exp(-0.5 * ((wl - 550) / 25) ** 2)
        red_absorb = _gaussian_dip(wl, 670, 0.09, 20)
        red_edge = _sigmoid(wl, 722, 0.09)
        nir_plateau = 0.55 * red_edge
        base = 0.05 + green_bump - red_absorb + nir_plateau
        r = np.clip(base, 0.02, 0.9)
        if stressed:
            # Physiological stress: shrinks chlorophyll absorption depth,
            # blue-shifts / flattens the red edge, lowers the NIR plateau --
            # exactly the mechanism hyperspectral crop-stress indices exploit.
            red_absorb_s = _gaussian_dip(wl, 670, 0.045, 22)
            red_edge_s = _sigmoid(wl, 705, 0.07)
            nir_plateau_s = 0.32 * red_edge_s
            base_s = 0.06 + 0.06 * np.exp(-0.5 * ((wl - 580) / 30) ** 2) - red_absorb_s + nir_plateau_s
            r = np.clip(base_s, 0.02, 0.9)
        return r

    def _bare_soil(self) -> np.ndarray:
        wl = self.wl
        # Smoothly increasing continuum typical of dry mineral soils.
        r = 0.08 + 0.22 * _sigmoid(wl, 650, 0.01)
        r += 0.03 * np.exp(-0.5 * ((wl - 1450) / 200) ** 2) * 0  # placeholder, kept flat if VNIR-only
        return np.clip(r, 0.05, 0.5)

    def _iron_oxide(self) -> np.ndarray:
        wl = self.wl
        r = self._bare_soil() * 1.1
        r = r - _gaussian_dip(wl, 900, 0.09, 60)   # ferric charge-transfer absorption
        r = r - _gaussian_dip(wl, 550, 0.03, 30)   # weaker crystal-field feature
        return np.clip(r, 0.03, 0.5)

    def _clay_mineral(self) -> np.ndarray:
        wl = self.wl
        r = self._bare_soil() * 0.95
        # Al-OH doublet only visible where SWIR coverage exists (Honeybee).
        r = r - _gaussian_dip(wl, 2160, 0.10, 15)
        r = r - _gaussian_dip(wl, 2205, 0.14, 12)
        return np.clip(r, 0.03, 0.5)

    def _urban(self) -> np.ndarray:
        wl = self.wl
        # Flat, moderate-high, slightly noisy-looking (concrete/asphalt mix)
        r = 0.18 + 0.05 * np.sin(wl / 120.0) * 0.3
        r = r + 0.02 * np.exp(-0.5 * ((wl - 500) / 100) ** 2)
        return np.clip(r, 0.08, 0.35)

    # ------------------------------------------------------------------ #
    def _build_all(self) -> None:
        palette = {
            "water": (30, 80, 200),
            "healthy_vegetation": (30, 160, 60),
            "stressed_vegetation": (190, 170, 40),
            "bare_soil": (150, 110, 70),
            "iron_oxide_mineral": (190, 90, 40),
            "clay_mineral": (170, 140, 190),
            "urban_built_up": (120, 120, 120),
        }
        curves = {
            "water": self._water(),
            "healthy_vegetation": self._vegetation(stressed=False),
            "stressed_vegetation": self._vegetation(stressed=True),
            "bare_soil": self._bare_soil(),
            "iron_oxide_mineral": self._iron_oxide(),
            "clay_mineral": self._clay_mineral(),
            "urban_built_up": self._urban(),
        }
        for i, name in enumerate(self.CLASS_NAMES):
            self.materials[name] = Material(
                name=name,
                class_id=i,
                color_rgb=palette[name],
                reflectance=curves[name],
            )

    def matrix(self) -> np.ndarray:
        """Return an (n_classes, n_bands) endmember matrix, class-id ordered."""
        return np.stack([self.materials[n].reflectance for n in self.CLASS_NAMES], axis=0)

    def methane_absorption_coefficient(self) -> np.ndarray:
        """
        A narrow-band unit absorption spectrum shaped like the real CH4
        SWIR absorption feature (~2298-2312 nm) used by matched-filter gas
        detection algorithms. Values are unitless "optical depth per unit
        path concentration" for simulation purposes only. This is always
        POSITIVE (it represents an absorption coefficient); increasing gas
        concentration REDUCES reflectance/radiance proportional to this
        spectrum (see `methane_target_signature` for the correctly-signed
        matched-filter target).
        """
        wl = self.wl
        feature = _gaussian_dip(wl, 2305, 1.0, 6.0)
        return feature / (feature.max() + 1e-12)

    def methane_target_signature(self) -> np.ndarray:
        """
        The matched-filter TARGET vector for CH4 detection: since higher gas
        concentration *reduces* reflectance proportional to
        `methane_absorption_coefficient()`, the direction a gas-affected
        pixel actually moves in spectral space is the *negative* of that
        absorption spectrum. Matched filters (see
        `src/classical/matched_filter.py`) are defined to score pixels
        highest when they deviate from the background mean *along* the
        target direction -- so the target passed in must already carry this
        sign, matching how real absorption-feature matched-filter gas
        retrievals are formulated (Manolakis & Shaw, 2002).
        """
        return -self.methane_absorption_coefficient()
