from __future__ import annotations

"""
SMILES string → jraph.GraphsTuple.

Each undirected bond becomes two directed edges (i→j and j→i).
Hydrogen handling is controlled by the `add_hs` flag.
"""
import numpy as np
from rdkit import Chem

from mini_gnn.data.featurizers import atom_features, bond_features
from mini_gnn.data.graph_tuple import make_graph


def smiles_to_graph(
    smiles: str,
    target: np.ndarray,
    add_hs: bool = False,
) -> jraph.GraphsTuple | None:
    """
    Convert a SMILES string to a GraphsTuple.
    Returns None if RDKit cannot parse the molecule.
    """
    import jraph

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if add_hs:
        mol = Chem.AddHs(mol)

    node_feats = np.stack([atom_features(a) for a in mol.GetAtoms()])

    senders, receivers, edge_feats = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        senders    += [i, j]
        receivers  += [j, i]
        edge_feats += [bf, bf]

    if not edge_feats:
        from mini_gnn.utils.types import BOND_FEAT_DIM
        senders, receivers = [0], [0]
        edge_feats = [np.zeros(BOND_FEAT_DIM, dtype=np.float32)]

    edge_feats = np.stack(edge_feats)
    senders    = np.array(senders,   dtype=np.int32)
    receivers  = np.array(receivers, dtype=np.int32)
    target     = np.atleast_1d(np.array(target, dtype=np.float32))

    return make_graph(node_feats, edge_feats, senders, receivers, target)
