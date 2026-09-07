from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.ndimage import uniform_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.evaluation.metrics import classification_metrics


def add_spatial_texture(pca_cube: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Append, for every PCA band, a local mean and local std-dev computed over a
    `window` x `window` sliding neighborhood -- a cheap, classical texture
    descriptor that gives the classifier some notion of spatial context
    beyond a single pixel's spectrum (helps distinguish e.g. contiguous urban
    blocks from isolated bright pixels of similar reflectance).
    """
    h, w, k = pca_cube.shape
    local_mean = np.stack([uniform_filter(pca_cube[..., i], size=window) for i in range(k)], axis=-1)
    sq_mean = np.stack([uniform_filter(pca_cube[..., i] ** 2, size=window) for i in range(k)], axis=-1)
    local_var = np.clip(sq_mean - local_mean ** 2, 0, None)
    local_std = np.sqrt(local_var)
    return np.concatenate([pca_cube, local_mean, local_std], axis=-1)


def build_feature_matrix(pca_cube: np.ndarray, use_spatial: bool = True, window: int = 5) -> np.ndarray:
    feats = add_spatial_texture(pca_cube, window) if use_spatial else pca_cube
    h, w, f = feats.shape
    return feats.reshape(-1, f)


def train_random_forest(
    pca_cube: np.ndarray,
    label_map: np.ndarray,
    class_names,
    n_estimators: int = 300,
    max_depth: int = 20,
    test_size: float = 0.3,
    use_spatial: bool = True,
    window: int = 5,
    random_state: int = 42,
) -> Tuple[RandomForestClassifier, Dict, np.ndarray]:
    """
    Train/evaluate a Random Forest spectral-spatial classifier.

    Returns
    -------
    model         : fitted RandomForestClassifier
    metrics       : dict from `classification_metrics`
    full_pred_map : (H, W) predicted label for *every* pixel in the scene
                    (train + test), useful for the classification-map figure.
    """
    h, w = label_map.shape
    x = build_feature_matrix(pca_cube, use_spatial, window)
    y = label_map.reshape(-1)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1,
        random_state=random_state, class_weight="balanced_subsample",
    )
    model.fit(x_train, y_train)

    y_pred_test = model.predict(x_test)
    metrics = classification_metrics(y_test, y_pred_test, class_names)

    full_pred = model.predict(x).reshape(h, w)
    return model, metrics, full_pred
