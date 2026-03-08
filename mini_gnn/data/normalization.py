"""
Target normalization. Stats are computed on the training split only.
"""
import numpy as np


class TargetNormalizer:
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = float(mean)
        self.std  = float(std) if std != 0.0 else 1.0

    @classmethod
    def fit(cls, targets: np.ndarray) -> "TargetNormalizer":
        return cls(mean=float(targets.mean()), std=float(targets.std()))

    def transform(self, targets: np.ndarray) -> np.ndarray:
        return ((targets - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, targets: np.ndarray) -> np.ndarray:
        return (targets * self.std + self.mean).astype(np.float32)

    def to_dict(self) -> dict:
        return {"target_mean": self.mean, "target_std": self.std}

    @classmethod
    def from_dict(cls, d: dict) -> "TargetNormalizer":
        return cls(mean=d["target_mean"], std=d["target_std"])
