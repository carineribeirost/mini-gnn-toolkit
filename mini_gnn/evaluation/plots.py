"""
Comparison plots: bar charts for test metrics, learning curves for GNN training.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _require_mpl():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError("matplotlib is required for plots: uv add matplotlib")


def plot_comparison(
    rows:    list[dict],
    metric:  str,
    title:   str = "",
    out:     str | Path | None = None,
    lower_is_better: bool = True,
) -> None:
    """
    Bar chart comparing all models on a single metric (test split).

    Args:
        rows:            list of row dicts from evaluation.results.to_rows()
        metric:          metric key to plot (e.g. "rmse", "roc_auc")
        title:           plot title
        out:             save path (.png / .pdf); shows interactively if None
        lower_is_better: controls which bar is highlighted as best
    """
    plt = _require_mpl()

    models = [r["model"] for r in rows if metric in r]
    values = [r[metric]  for r in rows if metric in r]

    if not models:
        print(f"No data for metric {metric!r}")
        return

    best_fn = min if lower_is_better else max
    best_v  = best_fn(v for v in values if isinstance(v, float) and not np.isnan(v))
    colors  = ["#e07b54" if abs(v - best_v) < 1e-9 else "#5b8db8" for v in values]

    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.2), 4))
    bars = ax.bar(models, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.set_ylabel(metric)
    ax.set_title(title or f"Test {metric} comparison")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()

    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        print(f"Saved plot → {out}")
    else:
        plt.show()
    plt.close(fig)


def plot_learning_curves(
    histories:  dict[str, dict],   # arch -> history dict from train()
    metric:     str = "val_loss",
    title:      str = "",
    out:        str | Path | None = None,
) -> None:
    """
    Line plot of training/validation loss (or any history key) across epochs
    for multiple GNN architectures.
    """
    plt = _require_mpl()

    fig, ax = plt.subplots(figsize=(8, 4))
    for arch, hist in histories.items():
        if metric not in hist:
            continue
        values = hist[metric]
        ax.plot(range(1, len(values) + 1), values, label=arch)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title or metric.replace("_", " ").capitalize())
    ax.legend(fontsize=8)
    plt.tight_layout()

    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        print(f"Saved plot → {out}")
    else:
        plt.show()
    plt.close(fig)


def plot_pred_vs_true(
    preds:   np.ndarray,
    targets: np.ndarray,
    model:   str = "",
    out:     str | Path | None = None,
) -> None:
    """Scatter plot of predicted vs true values (regression)."""
    plt = _require_mpl()

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(targets, preds, alpha=0.5, s=15, color="#5b8db8")
    lo = min(targets.min(), preds.min())
    hi = max(targets.max(), preds.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Pred vs True — {model}" if model else "Pred vs True")
    plt.tight_layout()

    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        print(f"Saved plot → {out}")
    else:
        plt.show()
    plt.close(fig)
