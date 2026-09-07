from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names) -> Dict:
    oa = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    per_class_acc = np.diag(cm) / np.clip(cm.sum(axis=1), 1, None)
    aa = float(np.mean(per_class_acc))
    kappa = cohen_kappa_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=class_names, labels=list(range(len(class_names))),
        zero_division=0, output_dict=True,
    )
    return {
        "overall_accuracy": float(oa),
        "average_accuracy": float(aa),
        "cohen_kappa": float(kappa),
        "per_class_accuracy": {n: float(a) for n, a in zip(class_names, per_class_acc)},
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }
