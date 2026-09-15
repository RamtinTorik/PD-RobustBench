"""Evaluation metrics for binary PD detection (positive class = parkinson)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

THRESHOLD = 0.5


def compute_metrics(y_true, y_prob, threshold: float = THRESHOLD) -> dict:
    """Compute the full metric set from labels and positive-class probabilities.

    Returns dict with accuracy, balanced_accuracy, precision, sensitivity
    (= recall of the PD class), specificity, f1, auc and confusion counts.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "n": len(y_true),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        out["auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        out["auc"] = float("nan")
    return out


def best_threshold(y_true, y_prob) -> float:
    """Select the decision threshold maximising balanced accuracy on a
    validation set (used because HandPD is imbalanced and heads are small)."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    best_t, best_ba = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        ba = 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))
        if ba > best_ba:
            best_ba, best_t = ba, float(t)
    return best_t


def format_metrics(m: dict) -> str:
    return (f"acc={m['accuracy']:.3f} bacc={m['balanced_accuracy']:.3f} "
            f"sens={m['sensitivity']:.3f} spec={m['specificity']:.3f} "
            f"f1={m['f1']:.3f} auc={m['auc']:.3f}")
