from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.data.spectral_library import SpectralLibrary
from src.data.synthetic_hsi import generate_scene, generate_wavelengths
from src.preprocessing.atmospheric import atmospheric_correct
from src.preprocessing.dimensionality_reduction import mnf_transform, pca_transform
from src.preprocessing.radiometric import radiometric_calibrate


def _small_cfg():
    """A tiny, fast configuration for unit tests (small scene, few epochs)."""
    cfg = load_config("configs/config.yaml")
    cfg.scene.height = 32
    cfg.scene.width = 32
    cfg.scene.plume_center = [20, 10]
    cfg.scene.plume_radius = 5
    return cfg


def test_wavelength_generation():
    cfg = _small_cfg()
    wl_firefly_cfg = cfg.sensor
    wl_firefly_cfg.mode = "firefly"
    wl = generate_wavelengths(wl_firefly_cfg)
    assert len(wl) == cfg.sensor.vnir_bands
    assert wl.min() >= cfg.sensor.vnir_range_nm[0]
    assert wl.max() <= cfg.sensor.vnir_range_nm[1]

    wl_firefly_cfg.mode = "honeybee"
    wl2 = generate_wavelengths(wl_firefly_cfg)
    assert len(wl2) == cfg.sensor.vnir_bands + cfg.sensor.swir_bands
    assert wl2.max() > cfg.sensor.vnir_range_nm[1]


def test_scene_generation_shapes_and_ranges():
    cfg = _small_cfg()
    scene = generate_scene(cfg)
    h, w = cfg.scene.height, cfg.scene.width
    n_bands = len(scene.wavelengths_nm)

    assert scene.reflectance.shape == (h, w, n_bands)
    assert scene.radiance.shape == (h, w, n_bands)
    assert scene.dn.shape == (h, w, n_bands)
    assert scene.label_map.shape == (h, w)
    assert scene.abundance_maps.shape == (h, w, len(scene.class_names))

    assert scene.reflectance.min() >= 0.0
    assert np.allclose(scene.abundance_maps.sum(axis=-1), 1.0, atol=1e-4)
    assert scene.dn.max() <= 2 ** cfg.sensor.bit_depth - 1

    # every class should appear at least once (Voronoi seeding guarantees this)
    assert set(np.unique(scene.label_map).tolist()) == set(range(len(scene.class_names)))


def test_methane_plume_only_in_honeybee_mode():
    cfg = _small_cfg()
    cfg.sensor.mode = "firefly"
    scene = generate_scene(cfg)
    assert scene.plume_mask is None, "Firefly (VNIR-only) scenes should not carry a SWIR gas plume"

    cfg2 = _small_cfg()
    cfg2.sensor.mode = "honeybee"
    scene2 = generate_scene(cfg2)
    assert scene2.plume_mask is not None
    assert scene2.plume_mask.sum() > 0


def test_radiometric_and_atmospheric_correction_recovers_reflectance():
    cfg = _small_cfg()
    scene = generate_scene(cfg)
    radiance_cal = radiometric_calibrate(scene.dn, scene.calibration_gain, scene.calibration_offset)
    reflectance_est, path_radiance = atmospheric_correct(
        radiance_cal, scene.wavelengths_nm, cfg.preprocessing.dark_object_percentile,
        sun_zenith_deg=cfg.scene.sun_zenith_deg,
    )
    assert reflectance_est.shape == scene.reflectance.shape
    mae = float(np.mean(np.abs(reflectance_est - scene.reflectance)))
    # The simplified DOS-based correction should recover the true surface
    # reflectance to within a small absolute error (loose bound: this is an
    # empirical surrogate for real optimal-estimation atmospheric correction,
    # not an exact inversion).
    assert mae < 0.08, f"atmospheric correction MAE too high: {mae}"


def test_pca_and_mnf_shapes():
    cfg = _small_cfg()
    scene = generate_scene(cfg)
    pca_img, evr, axes, k = pca_transform(scene.reflectance, variance_target=0.99)
    assert pca_img.shape[:2] == scene.reflectance.shape[:2]
    assert pca_img.shape[2] == k
    assert np.isclose(evr.sum(), 1.0, atol=1e-3)

    mnf_img, snr_eig = mnf_transform(scene.reflectance, n_components=10)
    assert mnf_img.shape == (cfg.scene.height, cfg.scene.width, 10)
    # SNR eigenvalues should be sorted descending
    assert np.all(np.diff(snr_eig[:10]) <= 1e-6)


def test_sam_classifier_beats_random_chance():
    from src.classical.sam import sam_classify

    cfg = _small_cfg()
    scene = generate_scene(cfg)
    lib = SpectralLibrary(scene.wavelengths_nm)
    labels, angles = sam_classify(scene.reflectance, lib.matrix(), threshold_deg=15.0)
    valid = labels >= 0
    acc = np.mean(labels[valid] == scene.label_map[valid])
    n_classes = len(scene.class_names)
    assert acc > 1.0 / n_classes, "SAM should clearly beat random-chance accuracy"


def test_rx_anomaly_detector_runs():
    from src.classical.anomaly import rx_anomaly_score, threshold_anomalies

    cfg = _small_cfg()
    scene = generate_scene(cfg)
    score = rx_anomaly_score(scene.reflectance)
    assert score.shape == (cfg.scene.height, cfg.scene.width)
    mask, thresh = threshold_anomalies(score, percentile=95.0)
    assert 0 < mask.mean() < 0.10  # ~5% flagged at the 95th percentile


def test_matched_filter_detects_methane_plume():
    from src.classical.matched_filter import detect_plume, evaluate_detection, matched_filter_score

    cfg = _small_cfg()
    cfg.sensor.mode = "honeybee"
    scene = generate_scene(cfg)
    lib = SpectralLibrary(scene.wavelengths_nm)
    target = lib.methane_target_signature()

    score = matched_filter_score(scene.reflectance, target)
    mask, _ = detect_plume(score, percentile=90.0)
    metrics = evaluate_detection(mask, scene.plume_mask)
    assert metrics["iou"] > 0.15, f"matched filter IoU too low: {metrics}"


def test_unmixing_recovers_abundances_reasonably():
    from src.classical.unmixing import unmix_cube, unmixing_rmse

    cfg = _small_cfg()
    scene = generate_scene(cfg)
    lib = SpectralLibrary(scene.wavelengths_nm)
    est = unmix_cube(scene.reflectance, lib.matrix(), stride=2)
    rmse = unmixing_rmse(est, scene.abundance_maps)
    assert rmse < 0.30


def test_random_forest_baseline_beats_majority_class():
    from src.ml.baseline import train_random_forest
    from src.preprocessing.dimensionality_reduction import pca_transform

    cfg = _small_cfg()
    scene = generate_scene(cfg)
    pca_img, _, _, _ = pca_transform(scene.reflectance, variance_target=0.995)
    lib = SpectralLibrary(scene.wavelengths_nm)

    _, metrics, full_pred = train_random_forest(
        pca_img, scene.label_map, lib.CLASS_NAMES, n_estimators=50, test_size=0.3,
    )
    majority_frac = np.bincount(scene.label_map.ravel()).max() / scene.label_map.size
    assert metrics["overall_accuracy"] > majority_frac


def test_vegetation_mineral_water_applications_run():
    from src.applications.mineral_mapping import map_clay_mineral, map_iron_oxide
    from src.applications.vegetation_health import classify_vegetation_stress
    from src.applications.water_quality import compute_chlorophyll_proxy, compute_turbidity_index

    cfg = _small_cfg()
    scene = generate_scene(cfg)
    wl = scene.wavelengths_nm

    ndvi, rep, stress_mask = classify_vegetation_stress(scene.reflectance, wl)
    assert ndvi.shape == (cfg.scene.height, cfg.scene.width)
    assert np.nanmax(rep) <= 800.0 and np.nanmin(rep) >= 650.0

    iron_map = map_iron_oxide(scene.reflectance, wl)
    clay_map = map_clay_mineral(scene.reflectance, wl)
    assert iron_map.shape == ndvi.shape
    assert clay_map.shape == ndvi.shape

    turbidity = compute_turbidity_index(scene.reflectance, wl)
    chlorophyll = compute_chlorophyll_proxy(scene.reflectance, wl)
    assert turbidity.shape == ndvi.shape
    assert chlorophyll.shape == ndvi.shape


def test_deep_learning_models_build_and_forward():
    """Skips cleanly (does not fail) if PyTorch is not installed."""
    try:
        import torch
    except ImportError:
        print("SKIPPED: PyTorch not installed in this environment.")
        return

    from src.dl.hybridsn import HybridSN
    from src.dl.spectral_transformer import SpectralSpatialTransformer

    num_classes, pca_components, patch_size, batch = 7, 30, 9, 4
    x = torch.randn(batch, 1, pca_components, patch_size, patch_size)

    hybridsn = HybridSN(num_classes=num_classes, pca_components=pca_components, patch_size=patch_size)
    out1 = hybridsn(x)
    assert out1.shape == (batch, num_classes)

    transformer = SpectralSpatialTransformer(
        num_classes=num_classes, pca_components=pca_components, patch_size=patch_size,
        embed_dim=32, depth=2, num_heads=4, spectral_group_size=10,
    )
    out2 = transformer(x)
    assert out2.shape == (batch, num_classes)


if __name__ == "__main__":
    # Allow running without pytest: `python tests/test_pipeline.py`
    test_fns = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    passed, failed = 0, 0
    for fn in test_fns:
        try:
            fn()
            print(f"PASS: {fn.__name__}")
            passed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL: {fn.__name__} -- {exc}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
