"""
ASE Atoms / CIF / POSCAR → jraph.GraphsTuple.

Builds a graph using a radial cutoff neighbor list that respects periodic
boundary conditions. Edge features are the interatomic distance and the
unit displacement vector (4 values total), padded to BOND_FEAT_DIM.
"""
import numpy as np
from pathlib import Path

from mini_gnn.data.graph_tuple import make_graph
from mini_gnn.utils.types import ATOM_FEAT_DIM, BOND_FEAT_DIM

_PERIOD_ENDS = [2, 10, 18, 36, 54, 86, 118]


def _atom_features_ase(atomic_number: int) -> np.ndarray:
    """Compact structural atom features: period and group one-hot + atomic number normalised."""
    from mini_gnn.data.featurizers import _PERIODS, _GROUPS, _one_hot_unk
    period = next(i + 1 for i, e in enumerate(_PERIOD_ENDS) if atomic_number <= e)
    # crude group from valence electrons — good enough for structural tasks
    group = max(1, atomic_number - ([0] + _PERIOD_ENDS)[period - 1])
    feats = (
        _one_hot_unk(period, _PERIODS)         # 8
        + _one_hot_unk(group, _GROUPS)          # 19
        + [atomic_number / 118.0]              # 1  normalised Z
    )
    # pad to ATOM_FEAT_DIM with zeros
    feats += [0.0] * (ATOM_FEAT_DIM - len(feats))
    return np.array(feats[:ATOM_FEAT_DIM], dtype=np.float32)


def _edge_features_from_displacement(displacement: np.ndarray) -> np.ndarray:
    """4-dim edge feature: [distance, dx/d, dy/d, dz/d], padded to BOND_FEAT_DIM."""
    d = float(np.linalg.norm(displacement))
    unit = displacement / (d + 1e-8)
    feats = np.array([d, *unit], dtype=np.float32)            # (4,)
    feats = np.pad(feats, (0, BOND_FEAT_DIM - 4))             # (BOND_FEAT_DIM,)
    return feats


def structure_to_graph(
    structure,                  # ase.Atoms or path to CIF/POSCAR/XYZ
    target: np.ndarray,
    cutoff: float = 5.0,
) -> "jraph.GraphsTuple":
    import jraph
    from ase.io import read as ase_read
    from ase.neighborlist import neighbor_list

    if isinstance(structure, (str, Path)):
        atoms = ase_read(str(structure))
    else:
        atoms = structure

    node_feats = np.stack([
        _atom_features_ase(z) for z in atoms.get_atomic_numbers()
    ])

    i_idx, j_idx, distances, displacements = neighbor_list(
        "ijdD", atoms, cutoff=cutoff
    )

    if len(i_idx) == 0:
        # isolated atoms — add self-loops
        n = len(atoms)
        i_idx = np.arange(n, dtype=np.int32)
        j_idx = np.arange(n, dtype=np.int32)
        displacements = np.zeros((n, 3), dtype=np.float32)

    edge_feats = np.stack([
        _edge_features_from_displacement(d) for d in displacements
    ])
    senders   = i_idx.astype(np.int32)
    receivers = j_idx.astype(np.int32)
    target    = np.atleast_1d(np.array(target, dtype=np.float32))

    return make_graph(node_feats, edge_feats, senders, receivers, target)
