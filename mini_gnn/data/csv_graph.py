"""
Generic edge-list CSV → jraph.GraphsTuple.

Expected CSV columns:
    graph_id  sender  receiver  [edge_feat_0 … edge_feat_k]  target

Node features are optional. If absent, nodes are initialised with a one-hot
degree encoding padded to ATOM_FEAT_DIM.
"""
import numpy as np
import pandas as pd

from mini_gnn.data.graph_tuple import make_graph
from mini_gnn.utils.types import ATOM_FEAT_DIM, BOND_FEAT_DIM


def _degree_node_features(senders: np.ndarray, receivers: np.ndarray, n_nodes: int) -> np.ndarray:
    degrees = np.bincount(receivers, minlength=n_nodes).astype(np.float32)
    feats = np.zeros((n_nodes, ATOM_FEAT_DIM), dtype=np.float32)
    feats[:, 0] = degrees / (degrees.max() + 1e-8)  # normalised in-degree, dim 0
    return feats


def csv_to_graphs(
    path: str,
    target_col: str = "target",
    node_feat_cols: list[str] | None = None,
    edge_feat_cols: list[str] | None = None,
    graph_id_col: str = "graph_id",
    sender_col: str = "sender",
    receiver_col: str = "receiver",
) -> list["jraph.GraphsTuple"]:
    df = pd.read_csv(path)
    graphs = []

    for gid, group in df.groupby(graph_id_col):
        senders   = group[sender_col].values.astype(np.int32)
        receivers = group[receiver_col].values.astype(np.int32)
        n_nodes   = int(max(senders.max(), receivers.max())) + 1

        if edge_feat_cols:
            raw_edge = group[edge_feat_cols].values.astype(np.float32)
            if raw_edge.shape[1] < BOND_FEAT_DIM:
                raw_edge = np.pad(raw_edge, ((0, 0), (0, BOND_FEAT_DIM - raw_edge.shape[1])))
            edge_feats = raw_edge[:, :BOND_FEAT_DIM]
        else:
            edge_feats = np.zeros((len(senders), BOND_FEAT_DIM), dtype=np.float32)

        if node_feat_cols:
            # node features may appear on edge rows — take unique per node index
            node_df = group.drop_duplicates(subset=[sender_col])
            raw_node = node_df[node_feat_cols].values.astype(np.float32)
            if raw_node.shape[1] < ATOM_FEAT_DIM:
                raw_node = np.pad(raw_node, ((0, 0), (0, ATOM_FEAT_DIM - raw_node.shape[1])))
            node_feats = raw_node[:n_nodes, :ATOM_FEAT_DIM]
        else:
            node_feats = _degree_node_features(senders, receivers, n_nodes)

        target = np.atleast_1d(np.array(group[target_col].iloc[0], dtype=np.float32))
        graphs.append(make_graph(node_feats, edge_feats, senders, receivers, target))

    return graphs
