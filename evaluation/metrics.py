from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def _safe_metric(fn, *args, default: float | None = None, **kwargs):
    try:
        return float(fn(*args, **kwargs))
    except Exception:
        return default


def classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    family_true: np.ndarray | None = None,
    family_pred: np.ndarray | None = None,
    arch_true: np.ndarray | None = None,
    arch_pred: np.ndarray | None = None,
) -> dict[str, Any]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = y_prob.argmax(axis=1)
    metrics: dict[str, Any] = {
        "accuracy": _safe_metric(accuracy_score, y_true, y_pred),
        "precision": _safe_metric(precision_score, y_true, y_pred, average="weighted", zero_division=0),
        "recall": _safe_metric(recall_score, y_true, y_pred, average="weighted", zero_division=0),
        "f1": _safe_metric(f1_score, y_true, y_pred, average="weighted", zero_division=0),
        "roc_auc": None,
        "family_accuracy": None,
        "cross_architecture_accuracy": None,
    }
    if y_prob.ndim == 2 and y_prob.shape[1] == 2:
        metrics["roc_auc"] = _safe_metric(roc_auc_score, y_true, y_prob[:, 1])
    if family_true is not None and family_pred is not None and len(family_true):
        metrics["family_accuracy"] = _safe_metric(accuracy_score, np.asarray(family_true), np.asarray(family_pred))
    if arch_true is not None and arch_pred is not None and len(arch_true):
        metrics["cross_architecture_accuracy"] = _safe_metric(accuracy_score, np.asarray(arch_true), np.asarray(arch_pred))
    return metrics

