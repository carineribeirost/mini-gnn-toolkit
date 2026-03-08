"""Evaluation metrics — numpy/sklearn, runs on CPU after inference."""
from __future__ import annotations

import numpy as np

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
    _SKLEARN = True
except ImportError:
    _SKLEARN = False


def regression_metrics(preds: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    mae  = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    ss_res = float(np.sum((targets - preds) ** 2))
    ss_tot = float(np.sum((targets - targets.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def classification_metrics(preds: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """preds are logits; targets are binary labels."""
    probs  = 1.0 / (1.0 + np.exp(-preds))
    binary = (probs >= 0.5).astype(np.float32)
    acc    = float(np.mean(binary == targets))
    metrics: dict[str, float] = {"accuracy": acc}
    if _SKLEARN:
        try:
            metrics["roc_auc"] = float(roc_auc_score(targets, probs, average="macro"))
            metrics["pr_auc"]  = float(average_precision_score(targets, probs, average="macro"))
        except Exception:
            pass
    return metrics


def compute_metrics(preds: np.ndarray, targets: np.ndarray, task: str) -> dict[str, float]:
    if task == "regression":
        return regression_metrics(preds, targets)
    else:
        return classification_metrics(preds, targets)
