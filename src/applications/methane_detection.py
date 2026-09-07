from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from src.classical.matched_filter import detect_plume, evaluate_detection, matched_filter_score


def run_methane_detection(
    cube: np.ndarray,
    target_signature: np.ndarray,
    true_mask: Optional[np.ndarray] = None,
    true_concentration: Optional[np.ndarray] = None,
    detection_percentile: float = 90.0,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Returns
    -------
    score_map       : (H, W) raw matched-filter score.
    detected_mask   : (H, W) bool thresholded detection.
    diagnostics     : dict with threshold, and precision/recall/IoU/F1 against
                       ground truth if `true_mask` was supplied.
    """
    score_map = matched_filter_score(cube, target_signature)
    detected_mask, thresh = detect_plume(score_map, percentile=detection_percentile)

    diagnostics: Dict = {"threshold": thresh}
    if true_mask is not None:
        diagnostics.update(evaluate_detection(detected_mask, true_mask))

    if true_concentration is not None and true_mask is not None and true_mask.any():
        # Simple least-squares calibration: concentration ≈ a * score, fit
        # only where a plume truly exists, for illustration.
        s = score_map[true_mask]
        c = true_concentration[true_mask]
        denom = float((s ** 2).sum()) + 1e-9
        a = float((s * c).sum() / denom)
        diagnostics["illustrative_calibration_slope"] = a
        diagnostics["estimated_peak_ppm_m"] = float(a * score_map.max())

    return score_map, detected_mask, diagnostics
