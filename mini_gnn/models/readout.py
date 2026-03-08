"""
Global graph readout (pooling) functions.

Each function takes node features and graph structure information and returns
a (n_graphs, hidden_dim) array. All are pure JAX — no Flax module needed
for sum/mean/max. Attention readout is a lightweight Flax module.

The calling convention matches what GNN models expect:
    readout_fn(node_feats, graph) -> (n_graphs, hidden_dim)
"""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp
import jraph

from mini_gnn.utils.types import Array


def _n_graphs(graph: jraph.GraphsTuple) -> int:
    return graph.n_node.shape[0]


def sum_readout(node_feats: Array, graph: jraph.GraphsTuple) -> Array:
    """Sum node features within each graph."""
    return jraph.segment_sum(
        node_feats,
        jnp.repeat(jnp.arange(_n_graphs(graph)), graph.n_node),
        num_segments=_n_graphs(graph),
    )


def mean_readout(node_feats: Array, graph: jraph.GraphsTuple) -> Array:
    """Mean of node features within each graph."""
    graph_ids = jnp.repeat(jnp.arange(_n_graphs(graph)), graph.n_node)
    total   = jraph.segment_sum(node_feats, graph_ids, num_segments=_n_graphs(graph))
    counts  = jraph.segment_sum(
        jnp.ones(node_feats.shape[0]), graph_ids, num_segments=_n_graphs(graph)
    )
    return total / jnp.maximum(counts[:, None], 1.0)


def max_readout(node_feats: Array, graph: jraph.GraphsTuple) -> Array:
    """Max of node features within each graph (element-wise)."""
    graph_ids = jnp.repeat(jnp.arange(_n_graphs(graph)), graph.n_node)
    # segment_max is not in jraph; use jax.ops.segment_max
    return jax.ops.segment_max(
        node_feats,
        graph_ids,
        num_segments=_n_graphs(graph),
    )


READOUT_FNS = {
    "sum":  sum_readout,
    "mean": mean_readout,
    "max":  max_readout,
}


class AttentionReadout(nn.Module):
    """
    Soft-attention global readout.

    Learns a scalar attention score per node, applies softmax within each
    graph (via segment_softmax), then computes a weighted sum.
    """
    hidden_dim: int

    @nn.compact
    def __call__(self, node_feats: Array, graph: jraph.GraphsTuple) -> Array:
        n_graphs  = _n_graphs(graph)
        graph_ids = jnp.repeat(jnp.arange(n_graphs), graph.n_node)

        scores = nn.Dense(1)(node_feats).squeeze(-1)             # (N,)
        weights = jraph.segment_softmax(scores, graph_ids, num_segments=n_graphs)  # (N,)
        weighted = node_feats * weights[:, None]                  # (N, D)
        return jraph.segment_sum(weighted, graph_ids, num_segments=n_graphs)  # (G, D)


def get_readout(name: str, hidden_dim: int):
    """
    Return (readout_module_or_None, readout_fn_or_None).

    For sum/mean/max, the module is None and the fn is a pure function.
    For attention, the module is an AttentionReadout Flax module whose
    params must be included in the model's param tree.
    """
    if name in READOUT_FNS:
        return None, READOUT_FNS[name]
    if name == "attention":
        return AttentionReadout(hidden_dim=hidden_dim), None
    raise ValueError(f"Unknown readout '{name}'. Choose from: {list(READOUT_FNS) + ['attention']}")
