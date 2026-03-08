"""
RDKit 2D physicochemical descriptors for classical ML baselines.

Returns a fixed (N, N_DESC) float32 matrix computed purely from 2D SMILES —
no conformer generation, no 3D geometry.

NaN / inf values (can occur for some descriptor/molecule combos) are
replaced with 0. The fixed feature width is enforced so downstream code
always receives the same matrix shape regardless of which descriptors
are finite for a given molecule.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors


# All 2D descriptor names available in this RDKit version, in a stable order.
# Computed once at import time so the list is consistent across calls.
_DESC_NAMES: list[str] = [name for name, _ in Descriptors.descList]
N_DESCRIPTORS: int = len(_DESC_NAMES)   # typically 208–210 depending on RDKit version

_DESC_FNS = {name: fn for name, fn in Descriptors.descList}


def rdkit_2d_descriptors(smiles_list: list[str]) -> np.ndarray:
    """
    Compute RDKit 2D physicochemical descriptors for a list of SMILES.

    Args:
        smiles_list: list of SMILES strings

    Returns:
        (N, N_DESCRIPTORS) float32 array. Rows for invalid SMILES are all 0.
    """
    out = np.zeros((len(smiles_list), N_DESCRIPTORS), dtype=np.float32)

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        row = np.array(
            [_DESC_FNS[name](mol) for name in _DESC_NAMES],
            dtype=np.float64,
        )
        # replace NaN / inf with 0
        row = np.where(np.isfinite(row), row, 0.0)
        out[i] = row.astype(np.float32)

    return out


class DescriptorScaler:
    """
    Standardises descriptor matrices using training-split statistics.
    Must be fit on training data only.
    """
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_:  np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "DescriptorScaler":
        self.mean_ = X.mean(axis=0).astype(np.float32)
        std = X.std(axis=0).astype(np.float32)
        self.std_ = np.where(std > 0, std, 1.0)   # avoid division by zero
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def to_dict(self) -> dict:
        return {
            "desc_mean": self.mean_.tolist(),
            "desc_std":  self.std_.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DescriptorScaler":
        s = cls()
        s.mean_ = np.array(d["desc_mean"], dtype=np.float32)
        s.std_  = np.array(d["desc_std"],  dtype=np.float32)
        return s
