# pixxel-hsi-cv

** Hyperspectral Earth-observation computer-vision pipeline, methodologically modeled on how [Pixxel](https://www.pixxel.space/)'s Firefly and Honeybee hyperspectral satellite constellations are actually built and used.**

This project simulates a Pixxel-like hyperspectral scene from first physical principles, then runs it through a full processing chain — radiometric calibration → atmospheric correction → dimensionality reduction → classical spectral analysis → machine-learning and deep-learning classification → application products (crop stress, mineral mapping, methane detection, water quality) — with every stage grounded in a real, citable technique from the remote-sensing / hyperspectral-CV literature.

---

## 1. Why this project is built the way it is

Pixxel's actual imagery is proprietary, customer-tasked data delivered through their API/Aurora platform — it is not a public dataset, so no project can legitimately "run on Pixxel's real data" without a paid tasking contract. What *is* public, and what this project is built from, is Pixxel's own published sensor specification and processing methodology:

| Constellation | Spectral range | Bands | GSD | Swath | Status |
|---|---|---|---|---|---|
| **Firefly** (operational) | 470–900 nm (VNIR) | 150+ | 5.4 m | 40 km | 6 satellites in orbit as of early 2026 |
| **Honeybee** (next-gen) | 470–2500 nm (VNIR+SWIR) | ~160 VNIR + ~100 SWIR | 5 m | 30 km VNIR / 10 km SWIR | First launch targeted 2026 |

Pixxel's stated atmospheric correction, **piSOFIT**, is their customized implementation of NASA JPL's open-source **ISOFIT** (Imaging Spectrometer Optimal FITting) — an optimal-estimation inversion against a radiative-transfer lookup table. Their stated application areas are agriculture/crop stress, mineral exploration, methane/GHG leak detection, water quality, wildfire risk, and defense.

This project's synthetic-data simulator (`src/data/synthetic_hsi.py`) reproduces the *structural* properties of that real sensor — band count, wavelength coverage, GSD-consistent noise, raw-DN quantization, spectral "smile" — via a physically-motivated forward radiative-transfer model, so every downstream stage has something concrete, reproducible, and ground-truthed to run against, with **zero external downloads required**. Swapping in real tasked imagery (or a public HSI benchmark such as Indian Pines / Pavia University / Houston 2018) only requires writing a loader that returns the same `HyperspectralScene` structure — see the docstring in `src/data/synthetic_hsi.py`.

---

## 2. Pipeline architecture

```
┌─────────────────────┐
│  Physics-based       │   src/data/synthetic_hsi.py
│  scene simulator     │   (Voronoi material regions -> linear mixing ->
│  (reflectance)        │    gas-plume injection -> radiance -> noise -> DN)
└──────────┬───────────┘
           │ raw DN (uint16, 12-bit)
           ▼
┌─────────────────────┐
│ Radiometric          │   src/preprocessing/radiometric.py
│ calibration           │   DN -> at-sensor radiance (gain + offset)
└──────────┬───────────┘
           ▼
┌─────────────────────┐
│ Atmospheric           │   src/preprocessing/atmospheric.py
│ correction             │   Dark-Object Subtraction + known-solar-irradiance
│ (piSOFIT surrogate)    │   inversion -> surface reflectance
└──────────┬───────────┘
           ▼
┌─────────────────────┐
│ Dimensionality        │   src/preprocessing/dimensionality_reduction.py
│ reduction              │   PCA (variance-ranked) + MNF (SNR-ranked)
└──────────┬───────────┘
           │
     ┌─────┴──────────────────────────────┬───────────────────────┐
     ▼                                     ▼                       ▼
┌───────────────┐                 ┌────────────────┐     ┌──────────────────┐
│ Classical      │                 │ ML baseline     │     │ Deep learning     │
│ spectral algos │                 │ (Random Forest) │     │ (HybridSN,        │
│ SAM, unmixing, │                 │ src/ml/         │     │ Spectral-Spatial  │
│ RX, matched    │                 │ baseline.py     │     │ Transformer)      │
│ filter         │                 │                 │     │ src/dl/           │
│ src/classical/ │                 │                 │     │                   │
└───────┬────────┘                 └────────┬────────┘     └─────────┬─────────┘
        │                                    │                        │
        └──────────────────┬─────────────────┴────────────────────────┘
                            ▼
                 ┌──────────────────────┐
                 │ Application products  │   src/applications/
                 │ crop stress · mineral │
                 │ mapping · methane ·   │
                 │ water quality          │
                 └──────────────────────┘
```

Every stage is independently runnable and independently testable (see `tests/test_pipeline.py`); `main.py` chains them together.

---

## 3. Methodology, stage by stage

### 3.1 Synthetic scene simulation (`src/data/`)

- **`spectral_library.py`** builds physically-structured reflectance curves for 7 materials (water, healthy/stressed vegetation, bare soil, iron-oxide mineral, clay mineral, urban) as a smooth continuum with Gaussian absorption dips at real diagnostic wavelengths: chlorophyll absorption (~670 nm), the vegetation red edge (~700–750 nm), the ferric-iron charge-transfer absorption (~900 nm), the clay Al-OH doublet (~2160/2205 nm), and the CH₄ SWIR absorption feature (~2305 nm) — all real, published remote-sensing diagnostic bands.
- **`synthetic_hsi.py`** places these materials across the scene using a smooth Voronoi-style abundance field (soft-min of distance to random per-class seed points), giving **sub-pixel mixed boundaries** — the actual generative model spectral unmixing assumes. It then:
  1. Optionally injects a Gaussian-profile methane plume via the Beer-Lambert law (`transmittance = exp(-k·concentration·absorption_coefficient(λ))`) — **only when the sensor mode is `honeybee`**, since the 2305 nm CH₄ feature is physically outside Firefly's 470–900 nm VNIR range. (This is a deliberate, instructive property of the simulation, not a limitation — see §5.)
  2. Converts reflectance to at-sensor radiance via `radiance = reflectance · solar_irradiance(λ) · cos(θ_sun) / π + path_radiance(λ)` (`src/physics.py`).
  3. Applies a spectral "smile" (per-column wavelength micro-shift), Poisson shot noise, and Gaussian read noise at a configurable SNR.
  4. Quantizes to raw 12-bit digital numbers, exactly like a real sensor's ADC.

### 3.2 Radiometric calibration (`src/preprocessing/radiometric.py`)
Linear per-band gain/offset inversion of DN → radiance — the universal first step (Level-0 → Level-1) of any real satellite imaging-spectrometer pipeline.

### 3.3 Atmospheric correction (`src/preprocessing/atmospheric.py`)
**What Pixxel really does:** piSOFIT/ISOFIT jointly inverts a MODTRAN/6S radiative-transfer lookup table together with a surface-reflectance prior via non-linear optimal estimation, solving for atmospheric water vapor and aerosol state simultaneously with reflectance.

**What this project does, and why:** a full RT lookup-table inversion needs external MODTRAN/6SV tables not available in a self-contained environment. Instead, this module implements the classical **empirical** two-step method real operational pipelines used for decades before optimal estimation became practical:
1. **Dark-Object Subtraction** (Chavez, 1988) *estimates* (does not assume) the additive path-radiance term per band from the scene's darkest percentile of pixels.
2. Given that estimate, inverts the *same* physical forward equation the simulator used, treating the solar irradiance spectrum and illumination geometry as known quantities — exactly as real algorithms do, since the solar spectrum is a measured external reference, not something retrieved per-scene.

This correctly preserves cross-band spectral **shape** (critical: a naive independent per-band contrast stretch was tried and rejected during development — see §6 "Debugging log" — because it destroyed the shape every classical algorithm below depends on). Measured mean absolute error vs. ground-truth reflectance: **~0.024** (see `tests/test_pipeline.py::test_radiometric_and_atmospheric_correction_recovers_reflectance`, bound at 0.08).

### 3.4 Dimensionality reduction (`src/preprocessing/dimensionality_reduction.py`)
- **PCA** via SVD, ranked by variance.
- **MNF** (Minimum Noise Fraction, Green et al., 1988), ranked by **signal-to-noise ratio** instead: noise covariance is estimated via the shift-difference method (spatially-adjacent-pixel differencing, since real ground signal is spatially correlated but sensor noise is not), the data is noise-whitened, then PCA'd. This is the transform real operational hyperspectral pipelines prefer ahead of unmixing/anomaly detection, because high variance ≠ high information if that variance is mostly noise.
- On the default (Honeybee, 260-band) synthetic scene, **only ~4 PCA components are needed for 99.5% variance** — because the scene is generated by *linearly mixing 7 materials*, its true spectral rank is at most 7. This mirrors a well-documented real phenomenon in hyperspectral imaging called **virtual dimensionality**: real scenes also have effective spectral dimensionality far below their band count, which is precisely *why* PCA/MNF compression works at all.

### 3.5 Classical spectral algorithms (`src/classical/`)
All four are real, published, still-current-baseline algorithms:
- **Spectral Angle Mapper** (Kruse et al., 1993) — illumination-invariant nearest-reference classification by vector angle.
- **Linear spectral unmixing** (FCLS-style, via per-pixel NNLS + renormalization) — recovers sub-pixel material abundances under the same linear-mixing model used to generate the scene.
- **RX anomaly detector** (Reed & Yu, 1990) — the standard unsupervised "flag anything spectrally unusual" baseline, still compared against in modern deep anomaly-detection papers.
- **Matched-filter gas detection** (Manolakis & Shaw, 2002) — the same algorithm class real operational methane point-source retrievals (AVIRIS-NG/EMIT/GHGSat-style) use, with a physically-derived target signature (`SpectralLibrary.methane_target_signature()`).

### 3.6 Machine-learning baseline (`src/ml/baseline.py`)
PCA features + local-window mean/std spatial texture → Random Forest. This is the standard "hand-engineered spectral-spatial + ensemble" baseline any HSI deep-learning paper is expected to beat.

### 3.7 Deep learning (`src/dl/`) — requires `pip install torch`
- **HybridSN** (Roy et al., 2020, IEEE GRSL) — 3 stacked 3D convolutions (jointly modeling local spectral **and** spatial structure) followed by a 2D convolution (pure spatial pattern over the now-compact joint feature maps) and an MLP head. The 3D→2D channel/spatial reshaping is computed **dynamically via a dummy forward pass** at construction time, so it is correct for any `(patch_size, pca_components)` configuration, not hardcoded to the original paper's values.
- **Spectral-Spatial Transformer** (architecturally inspired by SpectralFormer, Hong et al., 2021, IEEE TGRS) — spectral bands are grouped into contiguous "spectral tokens", linearly embedded, prepended with a learnable `[CLS]` token + positional embeddings, and passed through a standard pre-norm Transformer encoder — letting e.g. a red-edge token attend directly to a SWIR clay-absorption token, which a small CNN kernel could only do after many layers of receptive-field growth.

### 3.8 Application products (`src/applications/`)
- **Crop/vegetation health**: pseudo-NDVI + Red-Edge Inflection Point via the Guyot & Baret (1988) four-point linear method — stress causes a measurable **blue-shift** in the red-edge position before visible symptoms appear, which is exactly why fine hyperspectral (not broadband multispectral) red-edge sampling is valuable.
- **Mineral mapping**: continuum removal (Clark & Roush, 1984) isolates the iron-oxide (~900 nm) and clay Al-OH (~2200 nm, SWIR-only) absorption features — the clay index deliberately returns all-zero in Firefly (VNIR-only) mode, since that feature is physically unobservable there.
- **Methane detection**: wraps the matched filter into a full detection + illustrative ppm·m-proxy quantification workflow.
- **Water quality**: standard band-ratio turbidity and blue/green chlorophyll proxy indices.

---

## 4. Results (this repository's own test run, `configs/config.yaml` defaults, Honeybee mode, 128×128 scene)

| Stage | Metric | Value |
|---|---|---|
| Atmospheric correction | Mean abs. error vs. true reflectance | 0.024 |
| PCA | Components for 99.5% variance | 4 / 260 |
| Spectral Angle Mapper | Pixel agreement with ground truth | 52.5% |
| Linear unmixing | RMSE vs. true abundances | 0.20 |
| Matched-filter CH₄ detection | Precision / Recall / IoU | 0.82 / 0.71 / 0.62 |
| Random Forest baseline | OA / AA / Kappa | 0.992 / 0.992 / 0.991 |

Regenerate all figures + these numbers with `python main.py all` (deep-learning numbers require `pip install torch` first — see §7).

---

## 5. A worked, physically-real teaching point: why Pixxel built a second, SWIR constellation

Run the pipeline in `firefly` mode (`sensor.mode: firefly` in `configs/config.yaml`) and the methane-detection and clay-mineral-mapping stages both cleanly report "not observable in this sensor mode" rather than a fabricated number — because the CH₄ (2305 nm) and clay Al-OH (2200 nm) diagnostic features are physically outside Firefly's 470–900 nm VNIR range. This is real physics: it is the actual, published reason Pixxel is building **Honeybee**, a second SWIR-capable constellation, on top of the already-operational VNIR-only Firefly.

---

## 6. Known simplifications and honest limitations

- **Synthetic, not real, imagery.** Pixxel's real data is proprietary tasked imagery; this project's data is a physics-based simulation designed to have the right *structural* properties (band count, noise, mixing, absorption features), not a real scene. Swap in a real public benchmark (Indian Pines / Pavia / Houston) or licensed Pixxel data by writing a loader matching the `HyperspectralScene` interface.
- **Atmospheric correction is a simplified DOS + known-irradiance surrogate**, not a real optimal-estimation MODTRAN/6S inversion like Pixxel's actual piSOFIT/ISOFIT — this project is explicit about that trade-off everywhere it's relevant (see §3.3).
- **Vegetation endmember spectra omit SWIR water-absorption troughs** (~1450/1950 nm) for a cleaner minimal endmember set; a research-grade spectral library (e.g. USGS splib07) would include them.
- **The RX anomaly detector**, as expected from the published algorithm, is sensitive to noise and mixed-pixel boundaries as well as "true" anomalies — this is a documented, real limitation of global RX, not a bug in this implementation.
- **The illustrative methane ppm·m quantification** in `src/applications/methane_detection.py` is a simple linear calibration fit for demonstration; a radiometrically validated retrieval needs a real unit-absorption spectrum and careful surface-albedo correction, as in the published matched-filter gas-retrieval literature.
- **The deep-learning stage (HybridSN / Spectral-Spatial Transformer) requires `pip install torch`.** The architectures' layer-shape arithmetic was verified by hand and their construction logic was fixed for a PyTorch BatchNorm/eval-mode pitfall found during development (see below), but — because the development sandbox this project was built in has no internet access to install PyTorch — the two networks could not be executed end-to-end during development. Every other module in this pipeline (everything under `src/data/`, `src/preprocessing/`, `src/classical/`, `src/ml/`, `src/applications/`, `src/evaluation/`, `src/visualization/`) *was* actually executed, and the numbers in §4 are real output, not projections.

### Debugging log (kept intentionally, as part of the methodology)
Two real, fixed-during-development bugs are worth understanding, because they're instructive failures in hyperspectral pipeline design, not just footnotes:
1. **Atmospheric correction destroying spectral shape.** An earlier version of `atmospheric_correct()` did an independent per-band percentile contrast-stretch. This looked reasonable band-by-band, but it silently destroyed the *relative* magnitude across bands — and every classical algorithm downstream (SAM, unmixing, matched filter) depends on that cross-band shape, not per-band contrast. Symptom: SAM classification agreement was 1.3% (near-random). Fix: replaced with the known-solar-irradiance physical inversion in §3.3. Result: SAM agreement rose to 52.5%.
2. **Matched-filter sign convention.** A gas absorption feature *reduces* reflectance proportional to its absorption coefficient — so the correct matched-filter target is the *negative* of the absorption spectrum, not the raw (positive) absorption spectrum. Getting this backwards doesn't crash anything; it just silently detects the plume as a *low*-score region instead of a high-score one. Symptom: precision/recall near zero even though the underlying signal was clearly present (verified directly by inspecting reflectance depth at plume vs. background pixels). Fix: `SpectralLibrary.methane_target_signature()` now returns the correctly-signed target explicitly, with the sign convention documented in its docstring so it can't be gotten wrong again by a future caller.

---

## 7. Installation & usage

```bash
pip install -r requirements.txt        # torch is optional; everything else is required

python main.py all                     # run the entire pipeline end-to-end
# or run stages individually:
python main.py generate                # simulate + visualize the scene only
python main.py preprocess              # + radiometric/atmospheric/PCA/MNF
python main.py classical               # + SAM, unmixing, RX, matched filter
python main.py baseline                # + Random Forest classifier
python main.py apps                    # + crop/mineral/methane/water products
python main.py deep                    # + HybridSN + Spectral-Spatial Transformer (needs torch)

python tests/test_pipeline.py          # run the test suite (or: pytest tests/ -v)
```

All parameters (sensor mode, scene size, noise, model hyperparameters) are in `configs/config.yaml` — no code changes needed to, e.g., switch to `firefly` mode, enlarge the scene, or change the Random Forest tree count.

Figures are written to `outputs/figures/*.png`, metrics to `outputs/metrics/*.json`.

### Project layout
```
configs/config.yaml            central configuration for every stage
main.py                        CLI entry point
src/
  physics.py                   shared forward/inverse radiative-transfer model
  data/
    spectral_library.py        physically-motivated endmember reflectance curves
    synthetic_hsi.py           Pixxel-like scene simulator
  preprocessing/
    radiometric.py             DN -> radiance
    atmospheric.py             radiance -> reflectance (DOS + physical inversion)
    dimensionality_reduction.py PCA + MNF (from scratch)
  classical/
    sam.py, unmixing.py, anomaly.py, matched_filter.py
  ml/baseline.py                Random Forest spectral-spatial classifier
  dl/
    dataset.py, hybridsn.py, spectral_transformer.py, train.py
  applications/
    vegetation_health.py, mineral_mapping.py, methane_detection.py, water_quality.py
  evaluation/metrics.py         OA / AA / Kappa / confusion matrix
  visualization/viz.py          all figure generation
tests/test_pipeline.py
```

---

## 8. References

- Pixxel — [Firefly & Honeybee constellation specifications](https://support.pixxel.space/hc/en-us/categories/18372518312604-Satellites-Imagery), [piSOFIT atmospheric correction](https://support.pixxel.space/hc/en-us/categories/18372518312604-Satellites-Imagery)
- Chavez, P. S. (1988). "An improved dark-object subtraction technique for atmospheric scattering correction." *Remote Sensing of Environment*.
- Green, A. A., et al. (1988). "A transformation for ordering multispectral data in terms of image quality with implications for noise removal" (MNF). *IEEE TGRS*.
- Kruse, F. A., et al. (1993). "The spectral image processing system (SIPS) — interactive visualization and analysis of imaging spectrometer data" (SAM). *Remote Sensing of Environment*.
- Reed, I. S., & Yu, X. (1990). "Adaptive multiple-band CFAR detection of an optical pattern with unknown spectral distribution" (RX detector). *IEEE Trans. ASSP*.
- Manolakis, D., & Shaw, G. (2002). "Detection algorithms for hyperspectral imaging applications" (matched filter). *IEEE Signal Processing Magazine*.
- Clark, R. N., & Roush, T. L. (1984). "Reflectance spectroscopy: Quantitative analysis techniques for remote sensing applications" (continuum removal). *J. Geophysical Research*.
- Guyot, G., & Baret, F. (1988). "Utilisation de la haute resolution spectrale pour suivre l'etat des couverts vegetaux" (red-edge position). *4th Int. Colloquium on Spectral Signatures*.
- Roy, S. K., Krishna, G., Dubey, S. R., & Chaudhuri, B. B. (2020). "HybridSN: Exploring 3-D–2-D CNN Feature Hierarchy for Hyperspectral Image Classification." *IEEE GRSL*.
- Hong, D., et al. (2021). "SpectralFormer: Rethinking Hyperspectral Image Classification with Transformers." *IEEE TGRS*.
- Bioucas-Dias, J. M., et al. (2012). "Hyperspectral Unmixing Overview: Geometrical, Statistical, and Sparse Regression-Based Approaches." *IEEE JSTARS*.

## License
MIT — see `LICENSE`. Educational/research project, not affiliated with or endorsed by Pixxel.
