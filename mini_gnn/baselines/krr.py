"""Kernel Ridge Regression baseline on RDKit 2D physicochemical descriptors."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mini_gnn.data.descriptors import DescriptorScaler
from mini_gnn.training.metrics import compute_metrics


@dataclass
class KRRConfig:
    alpha:  float = 1e-6
    gamma:  float = 1e-3   # RBF bandwidth; None → 1/n_features
    kernel: str   = "rbf"


def load_krr_config(path: str) -> KRRConfig:
    import tomllib
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return KRRConfig(**raw.get("model", {}))


def _make_krr(cfg: KRRConfig):
    gamma = cfg.gamma if cfg.gamma > 0 else None
    try:
        from cuml.kernel_ridge import KernelRidge
        return KernelRidge(alpha=cfg.alpha, gamma=gamma, kernel=cfg.kernel)
    except ImportError:
        from sklearn.kernel_ridge import KernelRidge
        return KernelRidge(alpha=cfg.alpha, gamma=gamma, kernel=cfg.kernel)


def _get_targets(dataset: dict, idx: np.ndarray) -> np.ndarray:
    return np.array(
        [dataset["graphs"][i].globals[0, :-1] for i in idx], dtype=np.float32
    )


def train_krr(
    dataset: dict,
    cfg: KRRConfig,
    task: str,
) -> tuple[object, dict[str, dict]]:
    """
    Train a Kernel Ridge Regressor on scaled RDKit 2D descriptors.

    For classification tasks KRR regresses directly on 0/1 labels; the raw
    predictions are treated as logits (no sigmoid applied internally).

    Returns:
        model    — fitted sklearn/cuML KRR object
        metrics  — {'train': {...}, 'val': {...}, 'test': {...}}
    """
    tr = dataset["train_idx"]
    va = dataset["val_idx"]
    te = dataset["test_idx"]

    scaler = DescriptorScaler.from_dict({
        "desc_mean": dataset["desc_mean"].tolist(),
        "desc_std":  dataset["desc_std"].tolist(),
    })
    desc = dataset["descriptors"]
    X_tr = scaler.transform(desc[tr])
    X_va = scaler.transform(desc[va])
    X_te = scaler.transform(desc[te])

    y_tr = _get_targets(dataset, tr)
    y_va = _get_targets(dataset, va)
    y_te = _get_targets(dataset, te)

    if y_tr.shape[1] == 1:
        y_tr = y_tr[:, 0]
        y_va = y_va[:, 0]
        y_te = y_te[:, 0]

    krr = _make_krr(cfg)
    krr.fit(X_tr, y_tr)

    def _eval(X, y):
        preds = krr.predict(X).reshape(len(X), -1).astype(np.float32)
        tgts  = y.reshape(-1, 1).astype(np.float32) if y.ndim == 1 else y.astype(np.float32)
        # clamp KRR outputs used as logits to prevent metric overflow
        if task != "regression":
            preds = np.clip(preds, -10.0, 10.0)
        return compute_metrics(preds, tgts, task)

    return krr, {
        "train": _eval(X_tr, y_tr),
        "val":   _eval(X_va, y_va),
        "test":  _eval(X_te, y_te),
    }
