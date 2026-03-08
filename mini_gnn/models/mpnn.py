"""
Message Passing Neural Network (Gilmer et al., 2017).

Edge features are used directly in the message:
    m_{uv} = MLP_msg( h_u || h_v || e_{uv} )
    h_v'   = GRU( h_v, sum_{u} m_{uv} )

The GRU update gives MPNN stronger expressiveness than simple
sum/mean aggregation, at the cost of a recurrent dependency per layer.
"""
from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp
import jraph

from mini_gnn.models.base import register
from mini_gnn.models.mlp import MLP
from mini_gnn.models.readout import get_readout
from mini_gnn.utils.types import Array, BOND_FEAT_DIM


class MPNNLayer(nn.Module):
    hidden_dim: int
    dropout:    float = 0.0

    @nn.compact
    def __call__(
        self,
        h:         Array,   # (N, hidden_dim)
        edges:     Array,   # (E, BOND_FEAT_DIM)
        senders:   Array,
        receivers: Array,
        train:     bool,
    ) -> Array:
        # build messages: sender features || receiver features || edge features
        msg_input = jnp.concatenate(
            [h[senders], h[receivers], edges], axis=-1
        )
        msgs = MLP(
            hidden_dim=self.hidden_dim,
            out_dim=self.hidden_dim,
            num_layers=2,
            dropout=self.dropout,
        )(msg_input, train=train)

        # aggregate messages at each receiver node
        agg = jraph.segment_sum(msgs, receivers, num_segments=h.shape[0])

        # GRU-style update
        h_new = nn.GRUCell(features=self.hidden_dim)(h, agg)[0]
        return h_new


@register("mpnn")
class MPNN(nn.Module):
    hidden_dim: int   = 128
    num_layers: int   = 4
    dropout:    float = 0.0
    readout:    str   = "mean"
    n_targets:  int   = 1

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple, train: bool = False) -> Array:
        # project node features; also embed edge features to hidden_dim
        h = nn.Dense(self.hidden_dim)(graph.nodes)
        e = nn.Dense(self.hidden_dim)(graph.edges)

        for _ in range(self.num_layers):
            h = MPNNLayer(hidden_dim=self.hidden_dim, dropout=self.dropout)(
                h, e, graph.senders, graph.receivers, train
            )

        readout_mod, readout_fn = get_readout(self.readout, self.hidden_dim)
        if readout_mod is not None:
            graph_feats = readout_mod(h, graph)
        else:
            graph_feats = readout_fn(h, graph)

        return MLP(hidden_dim=self.hidden_dim, out_dim=self.n_targets, num_layers=2)(
            graph_feats, train=train
        )
