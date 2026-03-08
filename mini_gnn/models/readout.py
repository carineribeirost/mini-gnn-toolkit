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
    return graph.n_node.shape[0]   # static — shape is known at trace time


def _graph_ids(node_feats: Array, graph: jraph.GraphsTuple) -> Array:
    """
    Build a (N,) int array mapping each node to its graph index.
    total_repeat_length=node_feats.shape[0] is static (padding size),
    which is required for jnp.repeat inside jax.jit.
    """
    return jnp.repeat(
        jnp.arange(_n_graphs(graph)),
        graph.n_node,
        total_repeat_length=node_feats.shape[0],
    )


def sum_readout(node_feats: Array, graph: jraph.GraphsTuple) -> Array:
    return jraph.segment_sum(
        node_feats,
        _graph_ids(node_feats, graph),
        num_segments=_n_graphs(graph),
    )


def mean_readout(node_feats: Array, graph: jraph.GraphsTuple) -> Array:
    graph_ids = _graph_ids(node_feats, graph)
    total  = jraph.segment_sum(node_feats, graph_ids, num_segments=_n_graphs(graph))
    counts = jraph.segment_sum(
        jnp.ones(node_feats.shape[0]), graph_ids, num_segments=_n_graphs(graph)
    )
    return total / jnp.maximum(counts[:, None], 1.0)


def max_readout(node_feats: Array, graph: jraph.GraphsTuple) -> Array:
    return jax.ops.segment_max(
        node_feats,
        _graph_ids(node_feats, graph),
        num_segments=_n_graphs(graph),
    )


READOUT_FNS = {
    "sum":  sum_readout,
    "mean": mean_readout,
    "max":  max_readout,
}


class AttentionReadout(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, node_feats: Array, graph: jraph.GraphsTuple) -> Array:
        n_graphs  = _n_graphs(graph)
        graph_ids = _graph_ids(node_feats, graph)

        scores  = nn.Dense(1)(node_feats).squeeze(-1)
        weights = jraph.segment_softmax(scores, graph_ids, num_segments=n_graphs)
        weighted = node_feats * weights[:, None]
        return jraph.segment_sum(weighted, graph_ids, num_segments=n_graphs)


def get_readout(name: str, hidden_dim: int):
    if name in READOUT_FNS:
        return None, READOUT_FNS[name]
    if name == "attention":
        return AttentionReadout(hidden_dim=hidden_dim), None
    raise ValueError(f"Unknown readout '{name}'. Choose from: {list(READOUT_FNS) + ['attention']}")
