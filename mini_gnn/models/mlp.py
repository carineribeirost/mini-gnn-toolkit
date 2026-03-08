"""
Shared MLP block used by all GNN architectures.

Flax Linen module — stateless, params managed externally via init/apply.
"""
from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from mini_gnn.utils.types import Array


class MLP(nn.Module):
    """
    Multi-layer perceptron with configurable depth, width, activation,
    optional layer norm, and dropout.

    Args:
        hidden_dim:   width of every hidden layer
        out_dim:      width of the output layer
        num_layers:   total number of linear layers (including output)
        activation:   JAX activation function (default: relu)
        dropout:      dropout rate applied after each hidden layer (0 = off)
        layer_norm:   whether to apply LayerNorm after each hidden layer
    """
    hidden_dim: int
    out_dim:    int
    num_layers: int   = 2
    dropout:    float = 0.0
    layer_norm: bool  = False

    @nn.compact
    def __call__(self, x: Array, train: bool = False) -> Array:
        for _ in range(self.num_layers - 1):
            x = nn.Dense(self.hidden_dim)(x)
            if self.layer_norm:
                x = nn.LayerNorm()(x)
            x = nn.relu(x)
            if self.dropout > 0.0:
                x = nn.Dropout(rate=self.dropout, deterministic=not train)(x)
        x = nn.Dense(self.out_dim)(x)
        return x
