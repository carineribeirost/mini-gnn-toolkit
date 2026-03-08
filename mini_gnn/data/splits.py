"""
Deterministic train/val/test splitting.

Scaffold split uses RDKit's MurckoScaffold to group molecules by core structure,
ensuring test scaffolds are unseen during training (harder, more realistic).
"""
import numpy as np


def random_split(
    n: int,
    train: float = 0.8,
    val: float   = 0.1,
    seed: int    = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng     = np.random.default_rng(seed)
    idx     = rng.permutation(n)
    n_train = int(n * train)
    n_val   = int(n * val)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


def scaffold_split(
    smiles_list: list[str],
    train: float = 0.8,
    val: float   = 0.1,
    seed: int    = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group by Murcko scaffold, then assign groups to splits."""
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit import Chem
    from collections import defaultdict

    scaffolds: dict[str, list[int]] = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            scaffolds[""].append(i)
            continue
        core = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        scaffolds[core].append(i)

    rng    = np.random.default_rng(seed)
    groups = list(scaffolds.values())
    rng.shuffle(groups)

    n = sum(len(g) for g in groups)
    n_train, n_val = int(n * train), int(n * val)

    train_idx, val_idx, test_idx = [], [], []
    for g in groups:
        if len(train_idx) < n_train:
            train_idx.extend(g)
        elif len(val_idx) < n_val:
            val_idx.extend(g)
        else:
            test_idx.extend(g)

    return (
        np.array(train_idx, dtype=np.int64),
        np.array(val_idx,   dtype=np.int64),
        np.array(test_idx,  dtype=np.int64),
    )
