"""
Baseline runner — trains RF and KRR, returns results in a unified dict.

Also provides a helper to merge baseline results with GNN results for
side-by-side comparison tables (used by the evaluation module).
"""
from __future__ import annotations

from pathlib import Path

from mini_gnn.baselines.krr import KRRConfig, train_krr
from mini_gnn.baselines.rf import RFConfig, train_rf


def run_baselines(
    dataset: dict,
    task: str,
    rf_cfg:  RFConfig  | None = None,
    krr_cfg: KRRConfig | None = None,
) -> dict[str, dict[str, dict]]:
    """
    Train RF and KRR baselines and return their metrics.

    Args:
        dataset:  dict from load_dataset()
        task:     "regression" | "classification" | "multilabel_classification"
        rf_cfg:   RFConfig (defaults if None)
        krr_cfg:  KRRConfig (defaults if None)

    Returns:
        {
            "rf":  {"train": {...}, "val": {...}, "test": {...}},
            "krr": {"train": {...}, "val": {...}, "test": {...}},
        }
    """
    rf_cfg  = rf_cfg  or RFConfig()
    krr_cfg = krr_cfg or KRRConfig()

    print("Training RF baseline …")
    _, rf_metrics  = train_rf(dataset, rf_cfg, task)

    print("Training KRR baseline …")
    _, krr_metrics = train_krr(dataset, krr_cfg, task)

    return {"rf": rf_metrics, "krr": krr_metrics}


def run_baselines_from_configs(
    dataset:     dict,
    task:        str,
    rf_cfg_path:  str | Path | None = None,
    krr_cfg_path: str | Path | None = None,
) -> dict[str, dict[str, dict]]:
    """Load configs from .toml files, then run baselines."""
    from mini_gnn.baselines.krr import load_krr_config
    from mini_gnn.baselines.rf import load_rf_config

    rf_cfg  = load_rf_config(rf_cfg_path)   if rf_cfg_path  else RFConfig()
    krr_cfg = load_krr_config(krr_cfg_path) if krr_cfg_path else KRRConfig()
    return run_baselines(dataset, task, rf_cfg=rf_cfg, krr_cfg=krr_cfg)
