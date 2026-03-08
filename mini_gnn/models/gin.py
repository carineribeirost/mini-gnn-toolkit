"""
Graph Isomorphism Network (Xu et al., 2019).

h_v' = MLP( (1 + eps) * h_v + sum_{u in N(v)} h_u )

epsilon is a learnable scalar parameter initialised to 0.
GIN uses sum aggregation (not mean) — critical for its expressiveness proof.
Readout defaults to sum across nodes (also matches the paper).
"""
from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp
import jraph

from mini_gnn.models.base import register
from mini_gnn.models.mlp import MLP
from mini_gnn.models.readout import get_readout
from mini_gnn.utils.types import Array


class GINLayer(nn.Module):
    hidden_dim: int
    dropout:    float = 0.0

    @nn.compact
    def __call__(self, h: Array, senders: Array, receivers: Array, train: bool) -> Array:
        eps = self.param("eps", nn.initializers.zeros, ())
        agg = jraph.segment_sum(
            h[senders],
            receivers,
            num_segments=h.shape[0],
        )
        out = MLP(
            hidden_dim=self.hidden_dim,
            out_dim=self.hidden_dim,
            num_layers=2,
            dropout=self.dropout,
            layer_norm=True,
        )((1.0 + eps) * h + agg, train=train)
        return out


@register("gin")
class GIN(nn.Module):
    hidden_dim: int   = 128
    num_layers: int   = 4
    dropout:    float = 0.5
    readout:    str   = "sum"
    n_targets:  int   = 1

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple, train: bool = False) -> Array:
        h = nn.Dense(self.hidden_dim)(graph.nodes)

        for _ in range(self.num_layers):
            h = GINLayer(hidden_dim=self.hidden_dim, dropout=self.dropout)(
                h, graph.senders, graph.receivers, train
            )

        readout_mod, readout_fn = get_readout(self.readout, self.hidden_dim)
        if readout_mod is not None:
            graph_feats = readout_mod(h, graph)
        else:
            graph_feats = readout_fn(h, graph)

        return MLP(hidden_dim=self.hidden_dim, out_dim=self.n_targets, num_layers=2)(
            graph_feats, train=train
        )
