"""
Pretty-print comparison tables to stdout and optionally to a CSV file.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.4f}"


def print_table(
    rows:      list[dict],
    title:     str = "",
    highlight: str = "",   # metric to bold-mark the best value
) -> None:
    """Print a comparison table to stdout using plain text."""
    if not rows:
        return

    metrics = [k for k in rows[0] if k != "model"]
    col_w   = max(len(m) for m in metrics + ["model"]) + 2
    model_w = max(len(r["model"]) for r in rows) + 2

    sep = "-" * (model_w + col_w * len(metrics))

    if title:
        print(f"\n{title}")
    print(sep)
    header = f"{'model':<{model_w}}" + "".join(f"{m:>{col_w}}" for m in metrics)
    print(header)
    print(sep)

    # find best value per metric for highlighting
    best: dict[str, float] = {}
    if highlight and highlight in metrics:
        vals = [r[highlight] for r in rows if isinstance(r.get(highlight), float)
                and not np.isnan(r[highlight])]
        if vals:
            # lower is better for error metrics, higher for auc/r2
            lower_better = any(k in highlight for k in ("mae", "rmse", "loss"))
            best[highlight] = min(vals) if lower_better else max(vals)

    for row in rows:
        model = row["model"]
        parts = []
        for m in metrics:
            v   = row.get(m)
            raw = _fmt(v)
            if m in best and isinstance(v, float) and abs(v - best[m]) < 1e-9:
                raw = f"*{raw}*"   # mark best with asterisks (plain text)
            parts.append(f"{raw:>{col_w}}")
        print(f"{model:<{model_w}}" + "".join(parts))

    print(sep)


def save_csv(
    rows:     list[dict],
    path:     str | Path,
) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(v) if isinstance(v, float) else v
                             for k, v in row.items()})
    print(f"Saved CSV → {path}")
