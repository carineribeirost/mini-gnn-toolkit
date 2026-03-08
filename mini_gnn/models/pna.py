"""
Principal Neighbourhood Aggregation (Corso et al., 2020).

Uses multiple aggregators (sum, mean, max, std) and degree-based scalers
(identity, amplification, attenuation) for a richer neighbourhood encoding.

The average degree (delta) is a dataset statistic — not learnable.
It must be passed at model construction time from the dataset config.
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


def _pna_aggregate(
    h:         Array,   # (N, D)
    senders:   Array,
    receivers: Array,
    degrees:   Array,   # (N,) in-degree of each node
    delta:     float,   # average degree in training set
) -> Array:
    """
    Compute PNA aggregation: 4 aggregators × 3 scalers = 12 × D features per node.
    """
    N = h.shape[0]

    # --- aggregators ---
    agg_sum  = jraph.segment_sum(h[senders], receivers, num_segments=N)
    agg_mean = jraph.segment_mean(h[senders], receivers, num_segments=N)
    agg_max  = jax.ops.segment_max(h[senders], receivers, num_segments=N)

    # std: sqrt(E[x^2] - E[x]^2), clamped to avoid sqrt of negatives
    agg_sq   = jraph.segment_mean(h[senders] ** 2, receivers, num_segments=N)
    agg_std  = jnp.sqrt(jnp.maximum(agg_sq - agg_mean ** 2, 1e-8))

    agg = jnp.concatenate([agg_sum, agg_mean, agg_max, agg_std], axis=-1)  # (N, 4D)

    # --- scalers (degree-based) ---
    d      = degrees[:, None].astype(jnp.float32)                           # (N, 1)
    log_d  = jnp.log(d + 1.0)
    log_delta = jnp.log(jnp.array(delta + 1.0))

    scale_identity = jnp.ones_like(d)
    scale_amplify  = log_d / jnp.maximum(log_delta, 1e-8)
    scale_attenuate = log_delta / jnp.maximum(log_d, 1e-8)

    # apply each scaler to the full aggregator vector
    scaled = jnp.concatenate([
        agg * scale_identity,
        agg * scale_amplify,
        agg * scale_attenuate,
    ], axis=-1)                                                              # (N, 12D)

    return scaled


class PNALayer(nn.Module):
    hidden_dim: int
    delta:      float   # average training-set degree
    dropout:    float = 0.0

    @nn.compact
    def __call__(
        self,
        h:         Array,
        senders:   Array,
        receivers: Array,
        train:     bool,
    ) -> Array:
        N = h.shape[0]
        degrees = jraph.segment_sum(
            jnp.ones(senders.shape[0]), receivers, num_segments=N
        )

        agg = _pna_aggregate(h, senders, receivers, degrees, self.delta)

        # input to MLP: self features || PNA aggregation
        inp = jnp.concatenate([h, agg], axis=-1)
        out = MLP(
            hidden_dim=self.hidden_dim,
            out_dim=self.hidden_dim,
            num_layers=2,
            dropout=self.dropout,
            layer_norm=True,
        )(inp, train=train)
        return out


@register("pna")
class PNA(nn.Module):
    hidden_dim: int   = 64     # keep small — PNA is wide internally (12x aggregators)
    num_layers: int   = 4
    dropout:    float = 0.0
    readout:    str   = "mean"
    n_targets:  int   = 1
    delta:      float = 5.0    # set from dataset at preprocessing; default is a rough guess

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple, train: bool = False) -> Array:
        h = nn.Dense(self.hidden_dim)(graph.nodes)

        for _ in range(self.num_layers):
            h = PNALayer(hidden_dim=self.hidden_dim, delta=self.delta, dropout=self.dropout)(
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
