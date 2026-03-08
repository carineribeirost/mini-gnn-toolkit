"""
Collect and merge GNN + baseline metrics into a unified results table.

All metrics dicts have the shape:
    {"train": {metric: float, ...}, "val": {...}, "test": {...}}

The results table is a list of dicts, one row per model, with columns:
    model, split, metric_1, metric_2, ...
"""
from __future__ import annotations

import json
from pathlib import Path


def merge_results(
    gnn_results:      dict[str, dict[str, dict]],   # arch -> split -> metric -> float
    baseline_results: dict[str, dict[str, dict]],   # name -> split -> metric -> float
) -> dict[str, dict[str, dict]]:
    """Merge GNN and baseline results into a single dict keyed by model name."""
    return {**gnn_results, **baseline_results}


def to_rows(
    results: dict[str, dict[str, dict]],
    split:   str = "test",
) -> list[dict]:
    """
    Flatten results into a list of dicts for tabular display.

    Each row: {"model": name, metric_1: value, ...}
    """
    rows = []
    for model_name, splits in results.items():
        if split not in splits:
            continue
        row = {"model": model_name}
        row.update(splits[split])
        rows.append(row)
    return rows


def save_results(results: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def load_results(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)
