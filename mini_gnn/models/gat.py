"""
Graph Attention Network v1 (Veličković et al., 2018).

Attention coefficients are computed per directed edge, then normalised
with jraph.segment_softmax (per-node neighbourhood softmax — not global).

Multi-head attention:
  - Intermediate layers: concatenate heads → output dim = hidden_dim * num_heads
    then project back to hidden_dim with a linear layer.
  - Final layer: average heads → output dim = hidden_dim.

This keeps the output dim constant (hidden_dim) regardless of num_heads,
which is required for the readout to always see the same shape.
"""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp
import jraph

from mini_gnn.models.base import register
from mini_gnn.models.mlp import MLP
from mini_gnn.models.readout import get_readout
from mini_gnn.utils.types import Array


class GATLayer(nn.Module):
    hidden_dim: int
    num_heads:  int   = 4
    dropout:    float = 0.6
    final:      bool  = False   # True → average heads, False → concat + project

    @nn.compact
    def __call__(
        self,
        h:         Array,   # (N, hidden_dim)
        senders:   Array,
        receivers: Array,
        train:     bool,
    ) -> Array:
        head_dim = self.hidden_dim // self.num_heads

        # linear projection per head
        # W: (N, num_heads, head_dim)
        Wh = nn.Dense(self.num_heads * head_dim, use_bias=False)(h)
        Wh = Wh.reshape(h.shape[0], self.num_heads, head_dim)

        # attention logit vectors a_src and a_dst (one scalar per head per node)
        a_src = nn.Dense(self.num_heads, use_bias=False)(h)  # (N, num_heads)
        a_dst = nn.Dense(self.num_heads, use_bias=False)(h)  # (N, num_heads)

        # compute edge attention logits: e_{ij} = LeakyReLU(a_src_i + a_dst_j)
        e = nn.leaky_relu(a_src[senders] + a_dst[receivers], negative_slope=0.2)  # (E, num_heads)

        # per-node neighbourhood softmax using jraph (NOT jax.nn.softmax)
        # segment_softmax expects 1-D, so flatten over heads, then restore
        E = e.shape[0]
        e_flat   = e.reshape(E * self.num_heads)
        rcv_flat = jnp.repeat(receivers, self.num_heads) * self.num_heads + jnp.tile(
            jnp.arange(self.num_heads), E
        )
        alpha_flat = jraph.segment_softmax(
            e_flat,
            rcv_flat,
            num_segments=h.shape[0] * self.num_heads,
        )
        alpha = alpha_flat.reshape(E, self.num_heads)   # (E, num_heads)

        if self.dropout > 0.0 and train:
            alpha = nn.Dropout(rate=self.dropout, deterministic=False)(alpha)

        # weighted aggregation: sum over senders per receiver, per head
        # Wh[senders]: (E, num_heads, head_dim)
        # alpha:        (E, num_heads, 1)
        weighted = Wh[senders] * alpha[:, :, None]          # (E, num_heads, head_dim)
        weighted = weighted.reshape(E, self.num_heads * head_dim)

        agg = jraph.segment_sum(weighted, receivers, num_segments=h.shape[0])
        # agg: (N, num_heads * head_dim)
        agg = agg.reshape(h.shape[0], self.num_heads, head_dim)

        if self.final:
            # average heads → (N, hidden_dim)
            out = agg.mean(axis=1)
        else:
            # concatenate heads → project back to hidden_dim
            out = nn.relu(agg.reshape(h.shape[0], self.num_heads * head_dim))
            out = nn.Dense(self.hidden_dim)(out)

        return nn.LayerNorm()(out)


@register("gat")
class GAT(nn.Module):
    hidden_dim: int   = 128
    num_layers: int   = 4
    dropout:    float = 0.6
    readout:    str   = "mean"
    num_heads:  int   = 4
    n_targets:  int   = 1

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple, train: bool = False) -> Array:
        h = nn.Dense(self.hidden_dim)(graph.nodes)

        for i in range(self.num_layers):
            is_final = (i == self.num_layers - 1)
            h = GATLayer(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                final=is_final,
            )(h, graph.senders, graph.receivers, train)

        readout_mod, readout_fn = get_readout(self.readout, self.hidden_dim)
        if readout_mod is not None:
            graph_feats = readout_mod(h, graph)
        else:
            graph_feats = readout_fn(h, graph)

        return nn.Dense(self.n_targets)(graph_feats)
