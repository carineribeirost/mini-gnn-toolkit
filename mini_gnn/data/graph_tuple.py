"""
Helpers for building and padding jraph.GraphsTuple objects.

Convention:
  - globals shape: (1, n_targets + 1)  — last column is a binary mask
    (1 = real graph, 0 = dummy padding graph)
  - All edge lists are directed (each undirected bond → two directed edges)
"""
import numpy as np
import jraph

from mini_gnn.utils.types import ATOM_FEAT_DIM, BOND_FEAT_DIM


def make_graph(
    node_feats: np.ndarray,      # (N, ATOM_FEAT_DIM)
    edge_feats: np.ndarray,      # (E, BOND_FEAT_DIM)  — directed
    senders: np.ndarray,         # (E,) int32
    receivers: np.ndarray,       # (E,) int32
    target: np.ndarray,          # (n_targets,) float32
) -> jraph.GraphsTuple:
    """Build a single-graph GraphsTuple. target is stored in globals[:, :-1];
    the last column of globals is the mask flag set to 1 (real graph)."""
    n_targets = target.shape[0]
    globals_ = np.concatenate([target, np.ones(1, dtype=np.float32)])[None]  # (1, n_targets+1)

    return jraph.GraphsTuple(
        nodes=node_feats.astype(np.float32),
        edges=edge_feats.astype(np.float32),
        senders=senders.astype(np.int32),
        receivers=receivers.astype(np.int32),
        globals=globals_.astype(np.float32),
        n_node=np.array([node_feats.shape[0]], dtype=np.int32),
        n_edge=np.array([edge_feats.shape[0]], dtype=np.int32),
    )


def make_dummy_graph(n_targets: int) -> jraph.GraphsTuple:
    """A single-node self-loop graph with target=0 and mask=0 (ignored in loss)."""
    return jraph.GraphsTuple(
        nodes=np.zeros((1, ATOM_FEAT_DIM), dtype=np.float32),
        edges=np.zeros((1, BOND_FEAT_DIM), dtype=np.float32),
        senders=np.array([0], dtype=np.int32),
        receivers=np.array([0], dtype=np.int32),
        globals=np.zeros((1, n_targets + 1), dtype=np.float32),  # mask = 0
        n_node=np.array([1], dtype=np.int32),
        n_edge=np.array([1], dtype=np.int32),
    )


def pad_batch(
    graphs: list[jraph.GraphsTuple],
    max_nodes: int,
    max_edges: int,
    n_targets: int,
) -> jraph.GraphsTuple:
    """
    Batch a list of graphs and pad to (max_nodes, max_edges) by appending
    dummy graphs. Returns a single GraphsTuple with static shapes for JIT.
    """
    batched = jraph.batch(graphs)
    total_nodes = int(batched.n_node.sum())
    total_edges = int(batched.n_edge.sum())

    pad_nodes = max_nodes - total_nodes
    pad_edges = max_edges - total_edges

    if pad_nodes < 0 or pad_edges < 0:
        raise ValueError(
            f"Batch exceeds padding limits: nodes {total_nodes}/{max_nodes}, "
            f"edges {total_edges}/{max_edges}. Increase max_nodes/max_edges in dataset config."
        )

    return jraph.pad_with_graphs(batched, n_node=max_nodes, n_edge=max_edges)


def targets_and_mask(batch: jraph.GraphsTuple):
    """Split globals into (targets, mask). Both are JAX arrays."""
    return batch.globals[:, :-1], batch.globals[:, -1]
