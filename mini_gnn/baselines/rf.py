"""Random Forest baseline on RDKit 2D physicochemical descriptors."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mini_gnn.data.descriptors import DescriptorScaler
from mini_gnn.training.metrics import compute_metrics


@dataclass
class RFConfig:
    n_estimators: int   = 500
    max_depth:    int   = 0     # 0 = unlimited
    n_jobs:       int   = -1


def load_rf_config(path: str) -> RFConfig:
    import tomllib
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return RFConfig(**raw.get("model", {}))


def _make_rf(task: str, cfg: RFConfig):
    max_depth = cfg.max_depth if cfg.max_depth > 0 else None
    try:
        if task == "regression":
            from cuml.ensemble import RandomForestRegressor
            return RandomForestRegressor(n_estimators=cfg.n_estimators, max_depth=max_depth)
        else:
            from cuml.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=cfg.n_estimators, max_depth=max_depth)
    except ImportError:
        if task == "regression":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(
                n_estimators=cfg.n_estimators, max_depth=max_depth, n_jobs=cfg.n_jobs
            )
        else:
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=cfg.n_estimators, max_depth=max_depth, n_jobs=cfg.n_jobs
            )


def _get_targets(dataset: dict, idx: np.ndarray) -> np.ndarray:
    """Extract targets from graph globals (excludes the trailing mask dimension)."""
    return np.array(
        [dataset["graphs"][i].globals[0, :-1] for i in idx], dtype=np.float32
    )


def _predict_as_logits(rf, X: np.ndarray, task: str) -> np.ndarray:
    """
    Return predictions in logit space so compute_metrics receives the same
    format as GNN outputs (logits for classification, raw values for regression).
    """
    if task == "regression":
        return rf.predict(X).reshape(len(X), -1).astype(np.float32)

    # classification: get probabilities, convert to logits
    proba = rf.predict_proba(X)
    if isinstance(proba, list):
        # multi-label: list of (N, 2) arrays — take positive-class column
        proba = np.column_stack([p[:, 1] for p in proba])
    elif proba.ndim == 2 and proba.shape[1] == 2:
        proba = proba[:, 1:]
    proba = np.clip(proba, 1e-7, 1 - 1e-7)
    return np.log(proba / (1 - proba)).astype(np.float32)


def train_rf(
    dataset: dict,
    cfg: RFConfig,
    task: str,
) -> tuple[object, dict[str, dict]]:
    """
    Train a Random Forest on scaled RDKit 2D descriptors.

    Returns:
        model    — fitted sklearn/cuML RF object
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

    # single-target: sklearn prefers 1-D y
    if y_tr.shape[1] == 1:
        y_tr = y_tr[:, 0]
        y_va = y_va[:, 0]
        y_te = y_te[:, 0]

    rf = _make_rf(task, cfg)
    rf.fit(X_tr, y_tr)

    def _eval(X, y):
        preds = _predict_as_logits(rf, X, task)
        tgts  = y.reshape(-1, 1).astype(np.float32) if y.ndim == 1 else y.astype(np.float32)
        return compute_metrics(preds, tgts, task)

    return rf, {
        "train": _eval(X_tr, y_tr),
        "val":   _eval(X_va, y_va),
        "test":  _eval(X_te, y_te),
    }
