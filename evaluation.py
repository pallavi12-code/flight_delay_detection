"""Evaluation utilities kept independent from model training."""

from typing import Any, Dict

import numpy as np
from sklearn.metrics import classification_report, roc_auc_score


def evaluate_predictions(y_true, probabilities, threshold: float) -> Dict[str, Any]:
    """Return held-out metrics for probabilities and a chosen threshold."""
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    if y_true.shape[0] != probabilities.shape[0]:
        raise ValueError("Labels and probabilities must have equal length.")
    predictions = (probabilities >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities))
        if len(np.unique(y_true)) > 1 else None,
        "threshold": float(threshold),
        "classification_report": classification_report(
            y_true, predictions, output_dict=True, zero_division=0
        ),
    }
