#!/usr/bin/env python3
"""
main.py
========
Single CLI entry point for the whole pixxel-hsi-cv pipeline.

Usage
-----
    python main.py all                 # run everything end-to-end, save all figures + metrics
    python main.py generate            # just simulate + save the scene
    python main.py preprocess          # radiometric -> atmospheric -> PCA/MNF
    python main.py classical           # SAM, unmixing, RX anomaly, matched filter
    python main.py baseline            # Random Forest spectral-spatial classifier
    python main.py apps                # vegetation / mineral / methane / water products
    python main.py deep                # HybridSN + Spectral-Spatial Transformer (needs torch)

Every subcommand accepts `--config configs/config.yaml` (default shown) and
`--outdir outputs` to override where figures/metrics are written.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import load_config
from src.data.spectral_library import SpectralLibrary
from src.data.synthetic_hsi import generate_scene
from src.preprocessing.atmospheric import atmospheric_correct
from src.preprocessing.dimensionality_reduction import mnf_transform, pca_transform
from src.preprocessing.radiometric import radiometric_calibrate
from src.visualization import viz


PALETTE = [
    (30, 80, 200), (30, 160, 60), (190, 170, 40), (150, 110, 70),
    (190, 90, 40), (170, 140, 190), (120, 120, 120),
]


def _log(msg: str) -> None:
    print(f"[pixxel-hsi-cv] {msg}", flush=True)


def _save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else str(o))


# --------------------------------------------------------------------------- #
def cmd_generate(cfg, outdir):
    _log(f"Simulating a synthetic Pixxel-{cfg.sensor.mode}-mode scene "
         f"({len(generate_scene(cfg).wavelengths_nm)} bands) ...")
    scene = generate_scene(cfg)
    _log(f"  reflectance range: [{scene.reflectance.min():.3f}, {scene.reflectance.max():.3f}]")
    _log(f"  raw DN range:      [{scene.dn.min()}, {scene.dn.max()}] "
         f"({cfg.sensor.bit_depth}-bit)")
    viz.save_false_color(scene.radiance, scene.wavelengths_nm,
                          f"{outdir}/figures/01_false_color_radiance.png",
                          title=f"Simulated Firefly/Honeybee scene ({cfg.sensor.mode} mode) -- radiance")
    lib = SpectralLibrary(scene.wavelengths_nm)
    spectra = {n: lib.materials[n].reflectance for n in lib.CLASS_NAMES}
    viz.save_spectral_signatures(
        scene.wavelengths_nm, spectra, f"{outdir}/figures/02_reference_spectra.png",
        vlines=[(670, "red abs."), (722, "red edge"), (900, "Fe3+"), (2200, "Al-OH clay"), (2305, "CH4")],
    )
    viz.save_classification_map(scene.label_map, lib.CLASS_NAMES, PALETTE,
                                 f"{outdir}/figures/03_ground_truth_labels.png",
                                 title="Ground-truth material class map")
    return scene


def cmd_preprocess(cfg, outdir, scene=None):
    scene = scene or cmd_generate(cfg, outdir)
    _log("Radiometric calibration (DN -> radiance) ...")
    radiance_cal = radiometric_calibrate(scene.dn, scene.calibration_gain, scene.calibration_offset)

    _log("Atmospheric correction (simplified DOS + flat-field surrogate for piSOFIT/ISOFIT) ...")
    reflectance_est, path_radiance = atmospheric_correct(
        radiance_cal, scene.wavelengths_nm, cfg.preprocessing.dark_object_percentile,
        sun_zenith_deg=float(getattr(cfg.scene, "sun_zenith_deg", 25.0)),
    )
    mae = float(np.mean(np.abs(reflectance_est - scene.reflectance)))
    _log(f"  mean abs. error vs. ground-truth reflectance: {mae:.4f}")

    _log("PCA dimensionality reduction ...")
    pca_img, evr, axes, k = pca_transform(reflectance_est, cfg.preprocessing.pca_variance_target)
    _log(f"  kept {k} components for {cfg.preprocessing.pca_variance_target:.1%} variance "
         f"(from {reflectance_est.shape[-1]} bands)")
    viz.save_variance_plot(evr, f"{outdir}/figures/04_pca_variance.png", k_used=k)

    _log("MNF transform (Minimum Noise Fraction) ...")
    mnf_img, snr_eig = mnf_transform(reflectance_est, n_components=min(20, k))
    _log(f"  top-5 MNF SNR eigenvalues: {np.round(snr_eig[:5], 2)}")

    viz.save_false_color(reflectance_est, scene.wavelengths_nm,
                          f"{outdir}/figures/05_corrected_reflectance.png",
                          title="Atmospherically-corrected reflectance estimate")

    return {
        "scene": scene, "reflectance_est": reflectance_est, "path_radiance": path_radiance,
        "pca_img": pca_img, "pca_axes": axes, "pca_k": k, "mnf_img": mnf_img,
    }


def cmd_classical(cfg, outdir, pre=None):
    from src.classical.anomaly import rx_anomaly_score, threshold_anomalies
    from src.classical.matched_filter import matched_filter_score, detect_plume, evaluate_detection
    from src.classical.sam import sam_classify
    from src.classical.unmixing import unmix_cube, unmixing_rmse

    pre = pre or cmd_preprocess(cfg, outdir)
    scene = pre["scene"]
    refl = pre["reflectance_est"]
    lib = SpectralLibrary(scene.wavelengths_nm)
    endmembers = lib.matrix()

    _log("Spectral Angle Mapper classification ...")
    sam_labels, sam_angle = sam_classify(refl, endmembers, cfg.classical.sam_threshold_deg)
    sam_acc = float(np.mean(sam_labels == scene.label_map))
    _log(f"  SAM pixel agreement with ground truth: {sam_acc:.3f}")

    _log("Linear spectral unmixing (FCLS-style NNLS, stride=2 for speed) ...")
    t0 = time.time()
    abundance_est = unmix_cube(refl, endmembers, stride=2)
    rmse = unmixing_rmse(abundance_est, scene.abundance_maps)
    _log(f"  unmixing RMSE vs. ground-truth abundances: {rmse:.4f}  ({time.time()-t0:.1f}s)")
    viz.save_abundance_maps(abundance_est, lib.CLASS_NAMES, f"{outdir}/figures/06_abundance_maps.png")

    _log("RX global anomaly detection ...")
    rx_score = rx_anomaly_score(refl)
    rx_mask, rx_thresh = threshold_anomalies(rx_score, cfg.classical.rx_anomaly_percentile)
    rgb = viz.false_color_composite(refl, scene.wavelengths_nm)
    viz.save_heatmap_overlay(rgb, rx_score, f"{outdir}/figures/07_rx_anomaly.png",
                              title=f"RX anomaly score (>{cfg.classical.rx_anomaly_percentile}th pct outlined)",
                              mask=rx_mask)

    results = {
        "sam_pixel_agreement_with_ground_truth": sam_acc,
        "unmixing_rmse": rmse,
        "rx_anomaly_threshold": rx_thresh,
        "rx_anomaly_fraction": float(rx_mask.mean()),
    }

    if scene.mode == "honeybee" and scene.plume_mask is not None:
        _log("Matched-filter SWIR methane plume detection ...")
        target = lib.methane_target_signature()
        mf_score = matched_filter_score(refl, target)
        mf_mask, mf_thresh = detect_plume(mf_score, percentile=cfg.classical.methane_detection_percentile)
        det_metrics = evaluate_detection(mf_mask, scene.plume_mask)
        _log(f"  detection precision={det_metrics['precision']:.3f} "
             f"recall={det_metrics['recall']:.3f} IoU={det_metrics['iou']:.3f}")
        viz.save_heatmap_overlay(rgb, mf_score, f"{outdir}/figures/08_methane_matched_filter.png",
                                  title="CH4 matched-filter score (plume boundary outlined)",
                                  cmap="magma", mask=scene.plume_mask)
        results["methane_detection"] = det_metrics
    else:
        _log("Skipping methane detection: sensor mode is 'firefly' (VNIR-only) -- "
             "the 2305 nm CH4 SWIR feature is physically outside its spectral range.")

    _save_json(results, f"{outdir}/metrics/classical_results.json")
    return results


def cmd_baseline(cfg, outdir, pre=None):
    from src.ml.baseline import train_random_forest

    pre = pre or cmd_preprocess(cfg, outdir)
    scene = pre["scene"]
    lib = SpectralLibrary(scene.wavelengths_nm)

    _log("Training Random Forest spectral-spatial baseline classifier ...")
    model, metrics, full_pred = train_random_forest(
        pre["pca_img"], scene.label_map, lib.CLASS_NAMES,
        n_estimators=cfg.ml_baseline.n_estimators, max_depth=cfg.ml_baseline.max_depth,
        test_size=cfg.ml_baseline.test_size, use_spatial=cfg.ml_baseline.use_spatial_features,
        window=cfg.ml_baseline.spatial_window,
    )
    _log(f"  OA={metrics['overall_accuracy']:.4f}  AA={metrics['average_accuracy']:.4f}  "
         f"Kappa={metrics['cohen_kappa']:.4f}")

    viz.save_classification_map(full_pred, lib.CLASS_NAMES, PALETTE,
                                 f"{outdir}/figures/09_rf_classification_map.png",
                                 title="Random Forest predicted class map (full scene)")
    viz.save_confusion_matrix(metrics["confusion_matrix"], lib.CLASS_NAMES,
                               f"{outdir}/figures/10_rf_confusion_matrix.png")
    _save_json(metrics, f"{outdir}/metrics/random_forest_metrics.json")
    return metrics


def cmd_apps(cfg, outdir, pre=None):
    from src.applications.mineral_mapping import map_clay_mineral, map_iron_oxide
    from src.applications.methane_detection import run_methane_detection
    from src.applications.vegetation_health import classify_vegetation_stress
    from src.applications.water_quality import compute_chlorophyll_proxy, compute_turbidity_index

    pre = pre or cmd_preprocess(cfg, outdir)
    scene = pre["scene"]
    refl = pre["reflectance_est"]
    wl = scene.wavelengths_nm
    lib = SpectralLibrary(wl)

    _log("Application: crop / vegetation health (NDVI + red-edge position) ...")
    ndvi, rep, stress_mask = classify_vegetation_stress(refl, wl)
    viz.save_generic_map(ndvi, f"{outdir}/figures/11_ndvi.png", "Pseudo-NDVI", cmap="RdYlGn")
    viz.save_generic_map(rep, f"{outdir}/figures/12_red_edge_position.png",
                          "Red-Edge Inflection Point (nm)", cmap="viridis",
                          colorbar_label="nm")

    _log("Application: mineral mapping (continuum-removal absorption depth) ...")
    iron_map = map_iron_oxide(refl, wl)
    clay_map = map_clay_mineral(refl, wl)
    viz.save_generic_map(iron_map, f"{outdir}/figures/13_iron_oxide_depth.png",
                          "Iron-oxide absorption depth (~900nm)", cmap="Oranges")
    if scene.mode == "honeybee":
        viz.save_generic_map(clay_map, f"{outdir}/figures/14_clay_mineral_depth.png",
                              "Clay Al-OH absorption depth (~2200nm)", cmap="Purples")

    _log("Application: water quality proxies ...")
    turbidity = compute_turbidity_index(refl, wl)
    chlorophyll = compute_chlorophyll_proxy(refl, wl)
    viz.save_generic_map(turbidity, f"{outdir}/figures/15_turbidity_index.png",
                          "Turbidity proxy index", cmap="YlOrBr")
    viz.save_generic_map(chlorophyll, f"{outdir}/figures/16_chlorophyll_proxy.png",
                          "Chlorophyll (blue/green) proxy", cmap="BuGn")

    app_results = {
        "vegetation_stress_fraction": float(stress_mask.mean()),
        "mean_ndvi": float(ndvi.mean()),
        "mean_iron_oxide_depth": float(iron_map.mean()),
        "mean_clay_depth": float(clay_map.mean()) if scene.mode == "honeybee" else None,
    }

    if scene.mode == "honeybee" and scene.plume_mask is not None:
        target = lib.methane_target_signature()
        score_map, det_mask, diag = run_methane_detection(
            refl, target, true_mask=scene.plume_mask,
            true_concentration=scene.plume_concentration_ppm_m,
        )
        app_results["methane_detection_diagnostics"] = diag

    _save_json(app_results, f"{outdir}/metrics/application_results.json")
    return app_results


def cmd_deep(cfg, outdir, pre=None):
    try:
        import torch  # noqa: F401
    except ImportError:
        _log("PyTorch is not installed in this environment -- skipping the deep-learning stage.")
        _log("Install with:  pip install torch  (see requirements.txt), then re-run `python main.py deep`.")
        return None

    from src.dl.train import run_training

    pre = pre or cmd_preprocess(cfg, outdir)
    scene = pre["scene"]
    lib = SpectralLibrary(scene.wavelengths_nm)

    _log("Training HybridSN (3D-2D CNN) on PCA-reduced spectral-spatial patches ...")
    hybridsn_metrics = run_training(cfg, pre["reflectance_est"], scene.label_map,
                                     lib.CLASS_NAMES, outdir, model_name="hybridsn")

    _log("Training Spectral-Spatial Transformer ...")
    transformer_metrics = run_training(cfg, pre["reflectance_est"], scene.label_map,
                                        lib.CLASS_NAMES, outdir, model_name="spectral_transformer")

    results = {"hybridsn": hybridsn_metrics, "spectral_transformer": transformer_metrics}
    _save_json(results, f"{outdir}/metrics/deep_learning_metrics.json")
    return results


def cmd_all(cfg, outdir):
    scene = cmd_generate(cfg, outdir)
    pre = cmd_preprocess(cfg, outdir, scene=scene)
    cmd_classical(cfg, outdir, pre=pre)
    cmd_baseline(cfg, outdir, pre=pre)
    cmd_apps(cfg, outdir, pre=pre)
    cmd_deep(cfg, outdir, pre=pre)
    _log("Pipeline complete. See outputs/figures/*.png and outputs/metrics/*.json")


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="pixxel-hsi-cv pipeline")
    parser.add_argument("stage", choices=[
        "generate", "preprocess", "classical", "baseline", "apps", "deep", "all",
    ])
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--outdir", default="outputs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dispatch = {
        "generate": cmd_generate, "preprocess": cmd_preprocess, "classical": cmd_classical,
        "baseline": cmd_baseline, "apps": cmd_apps, "deep": cmd_deep, "all": cmd_all,
    }
    dispatch[args.stage](cfg, args.outdir)


if __name__ == "__main__":
    main()
