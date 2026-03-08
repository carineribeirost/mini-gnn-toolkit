"""
Graph Convolutional Network (Kipf & Welling, 2017).

Each layer aggregates neighbour features via degree-normalised mean,
concatenates with the node's own features, then applies a linear + LayerNorm + ReLU.
"""
from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp
import jraph

from mini_gnn.models.base import register
from mini_gnn.models.mlp import MLP
from mini_gnn.models.readout import get_readout
from mini_gnn.utils.types import Array


@register("gcn")
class GCN(nn.Module):
    hidden_dim: int   = 128
    num_layers: int   = 4
    dropout:    float = 0.0
    readout:    str   = "mean"
    n_targets:  int   = 1

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple, train: bool = False) -> Array:
        h = nn.Dense(self.hidden_dim)(graph.nodes)

        for _ in range(self.num_layers):
            agg = jraph.segment_mean(
                h[graph.senders],
                graph.receivers,
                num_segments=h.shape[0],
            )
            h = nn.Dense(self.hidden_dim)(jnp.concatenate([h, agg], axis=-1))
            h = nn.LayerNorm()(h)
            h = nn.relu(h)
            if self.dropout > 0.0:
                h = nn.Dropout(rate=self.dropout, deterministic=not train)(h)

        readout_mod, readout_fn = get_readout(self.readout, self.hidden_dim)
        if readout_mod is not None:
            graph_feats = readout_mod(h, graph)
        else:
            graph_feats = readout_fn(h, graph)

        return nn.Dense(self.n_targets)(graph_feats)
