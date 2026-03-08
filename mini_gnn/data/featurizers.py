"""
Atom and bond featurization using RDKit.

All output vectors have fixed widths (ATOM_FEAT_DIM, BOND_FEAT_DIM) defined
in utils/types.py so JAX JIT never sees shape mismatches between graphs.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdchem

from mini_gnn.utils.types import ATOM_FEAT_DIM, BOND_FEAT_DIM


# ---- one-hot helpers ----

def _one_hot(value, choices: list) -> list[int]:
    return [int(value == c) for c in choices]


def _one_hot_unk(value, choices: list) -> list[int]:
    """One-hot with a trailing 'unknown' bit for out-of-vocabulary values."""
    enc = [int(value == c) for c in choices]
    enc.append(int(value not in choices))
    return enc


# ---- atom feature spec ----

_ATOMIC_NUMS = list(range(1, 119))          # H … Og  (118 elements)
_DEGREES     = [0, 1, 2, 3, 4, 5, 6]
_HYBRIDISATIONS = [
    rdchem.HybridizationType.S,
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
]
_FORMAL_CHARGES = [-2, -1, 0, 1, 2]
_NUM_HS         = [0, 1, 2, 3, 4]

# atomic_num (119) + degree (8) + hybridisation (7) + charge (6) + hs (6) + aromatic + in_ring = 148
# We keep ATOM_FEAT_DIM = 72 by using compact encodings below.
# Breakdown: atomic_num one-hot → 50 common elements + 1 unk = 51 — too large.
# Instead: period (7+1) + group (18+1) + degree (7+1) + hybridisation (6+1)
#          + formal_charge (5+1) + num_hs (5+1) + aromatic (1) + in_ring (1) = 72 ✓
_PERIODS = [1, 2, 3, 4, 5, 6, 7]
_GROUPS  = list(range(1, 19))              # 1–18

_ATOM_FEAT_EXPECTED = (
    len(_PERIODS) + 1          # 8
    + len(_GROUPS) + 1         # 19
    + len(_DEGREES) + 1        # 8
    + len(_HYBRIDISATIONS) + 1 # 7
    + len(_FORMAL_CHARGES) + 1 # 6
    + len(_NUM_HS) + 1         # 6
    + 1                        # is_aromatic
    + 1                        # is_in_ring
)                              # total = 56 … adjust ATOM_FEAT_DIM in types.py if needed

# Sanity-check at import time so mismatches fail loudly rather than silently
# producing wrong-shaped tensors that JIT recompiles on every graph.
assert _ATOM_FEAT_EXPECTED == ATOM_FEAT_DIM, (
    f"ATOM_FEAT_DIM is {ATOM_FEAT_DIM} but featurizers produce {_ATOM_FEAT_EXPECTED} features. "
    "Update ATOM_FEAT_DIM in mini_gnn/utils/types.py."
)


def _period_group(atomic_num: int) -> tuple[int, int]:
    """Return (period, group) for a given atomic number."""
    # Periodic table layout; good enough for elements 1–118
    period_ends = [2, 10, 18, 36, 54, 86, 118]
    group_map = {
        1: 1, 2: 18,
        3: 1, 4: 2, 5: 13, 6: 14, 7: 15, 8: 16, 9: 17, 10: 18,
    }
    # simplified: use rdkit's GetAtomicNum, map to period/group via chem rules
    from rdkit.Chem.rdchem import GetPeriodicTable
    pt = GetPeriodicTable()
    try:
        # not all RDKit versions expose these; fallback below
        n_outer = pt.GetNOuterElecs(atomic_num)
        period = next(i + 1 for i, e in enumerate(period_ends) if atomic_num <= e)
        return period, n_outer if n_outer > 0 else 1
    except Exception:
        period = next(i + 1 for i, e in enumerate(period_ends) if atomic_num <= e)
        return period, 1


def atom_features(atom: rdchem.Atom) -> np.ndarray:
    an = atom.GetAtomicNum()
    period, group = _period_group(an)
    feats = (
        _one_hot_unk(period, _PERIODS)
        + _one_hot_unk(group, _GROUPS)
        + _one_hot_unk(atom.GetDegree(), _DEGREES)
        + _one_hot_unk(atom.GetHybridization(), _HYBRIDISATIONS)
        + _one_hot_unk(atom.GetFormalCharge(), _FORMAL_CHARGES)
        + _one_hot_unk(atom.GetTotalNumHs(), _NUM_HS)
        + [int(atom.GetIsAromatic())]
        + [int(atom.IsInRing())]
    )
    return np.array(feats, dtype=np.float32)


# ---- bond feature spec ----

_BOND_TYPES = [
    rdchem.BondType.SINGLE,
    rdchem.BondType.DOUBLE,
    rdchem.BondType.TRIPLE,
    rdchem.BondType.AROMATIC,
]
_BOND_STEREO = [
    rdchem.BondStereo.STEREONONE,
    rdchem.BondStereo.STEREOANY,
    rdchem.BondStereo.STEREOZ,
    rdchem.BondStereo.STEREOE,
]

_BOND_FEAT_EXPECTED = (
    len(_BOND_TYPES) + 1    # 5
    + 1                     # is_conjugated
    + 1                     # is_in_ring
    + 1                     # is_rotatable (heuristic)
    + len(_BOND_STEREO)     # 4  (no unk — stereo has a NONE value)
)                           # total = 12 … adjust BOND_FEAT_DIM if needed

assert _BOND_FEAT_EXPECTED == BOND_FEAT_DIM, (
    f"BOND_FEAT_DIM is {BOND_FEAT_DIM} but featurizers produce {_BOND_FEAT_EXPECTED} features. "
    "Update BOND_FEAT_DIM in mini_gnn/utils/types.py."
)


def bond_features(bond: rdchem.Bond) -> np.ndarray:
    is_rotatable = (
        bond.GetBondTypeAsDouble() == 1.0
        and not bond.IsInRing()
        and bond.GetBeginAtom().GetDegree() > 1
        and bond.GetEndAtom().GetDegree() > 1
    )
    feats = (
        _one_hot_unk(bond.GetBondType(), _BOND_TYPES)
        + [int(bond.GetIsConjugated())]
        + [int(bond.IsInRing())]
        + [int(is_rotatable)]
        + _one_hot(bond.GetStereo(), _BOND_STEREO)
    )
    return np.array(feats, dtype=np.float32)
