import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int   = 128
    num_layers: int   = 4
    dropout:    float = 0.0
    readout:    str   = "mean"    # "sum" | "mean" | "max" | "attention"
    num_heads:  int   = 4         # GAT only
    epsilon:    float = 0.0       # GIN only (initial value; learnable)
    delta:      float = 5.0       # PNA only — average training-set node degree


@dataclass(frozen=True)
class TrainConfig:
    lr:           float = 1e-3
    weight_decay: float = 0.0
    epochs:       int   = 200
    batch_size:   int   = 32
    patience:     int   = 30      # early-stopping patience in epochs
    grad_clip:    float = 1.0     # max gradient norm (0 = disabled)
    scheduler:    str   = "cosine"  # "cosine" | "constant" | "warmup_cosine"
    warmup_steps: int   = 0


@dataclass(frozen=True)
class DatasetConfig:
    name:        str   = "custom"
    path:        str   = ""
    target_col:  str   = "target"
    max_nodes:   int   = 64
    max_edges:   int   = 256
    task:        str   = "regression"   # "regression" | "classification" | "multilabel_classification"
    n_targets:   int   = 1
    split:       str   = "scaffold"     # "scaffold" | "random"
    # normalisation stats (filled by preprocess.py, regression only)
    target_mean: float = 0.0
    target_std:  float = 1.0


@dataclass(frozen=True)
class Config:
    model:   ModelConfig   = field(default_factory=ModelConfig)
    train:   TrainConfig   = field(default_factory=TrainConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    seed:    int           = 42
    arch:    str           = "gcn"  # "gcn"|"gin"|"mpnn"|"gat"|"pna"


# ---- loading helpers ----

def _deep_replace(dc: Any, updates: dict) -> Any:
    """Recursively apply a flat or nested dict of overrides to a frozen dataclass."""
    changes = {}
    for k, v in updates.items():
        if hasattr(dc, k):
            current = getattr(dc, k)
            if hasattr(current, "__dataclass_fields__") and isinstance(v, dict):
                changes[k] = _deep_replace(current, v)
            else:
                changes[k] = v
    return replace(dc, **changes)


def load_config(path: str | Path, overrides: dict | None = None) -> Config:
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    cfg = Config(
        model=ModelConfig(**raw.get("model", {})),
        train=TrainConfig(**raw.get("train", {})),
        dataset=DatasetConfig(**raw.get("dataset", {})),
        seed=raw.get("seed", 42),
        arch=raw.get("arch", "gcn"),
    )

    if overrides:
        cfg = _deep_replace(cfg, overrides)

    return cfg


def merge_with_default(path: str | Path, default_path: str | Path | None = None) -> Config:
    """Load default.toml first, then overlay the given config."""
    base = Config()
    if default_path is not None:
        base = load_config(default_path)
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return _deep_replace(base, raw)
