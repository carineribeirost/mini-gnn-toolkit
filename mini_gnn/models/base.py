"""
GNNModel protocol — the shared calling convention all architectures implement.

Every GNN in this toolkit is a Flax nn.Module whose __call__ matches:

    def __call__(self, graph: jraph.GraphsTuple, train: bool) -> jax.Array:
        # returns (n_graphs, n_targets) predictions

This Protocol enables the evaluation runner to swap models without
any isinstance checks or conditional logic.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import jax
import jraph


@runtime_checkable
class GNNModel(Protocol):
    def __call__(self, graph: jraph.GraphsTuple, train: bool) -> jax.Array:
        """
        Forward pass.

        Args:
            graph: a padded jraph.GraphsTuple (fixed shapes for JIT)
            train: True during training (enables dropout), False at eval

        Returns:
            Array of shape (n_graphs, n_targets)
        """
        ...


# Registry — maps arch name to its module class.
# Populated lazily by each model file to avoid circular imports.
_REGISTRY: dict[str, type] = {}


def register(name: str):
    """Decorator to register a Flax module as a named architecture."""
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_model_class(name: str) -> type:
    if name not in _REGISTRY:
        # trigger registration by importing the module
        import mini_gnn.models.gcn   # noqa: F401
        import mini_gnn.models.gin   # noqa: F401
        import mini_gnn.models.mpnn  # noqa: F401
        import mini_gnn.models.gat   # noqa: F401
        try:
            import mini_gnn.models.pna  # noqa: F401
        except ImportError:
            pass
    if name not in _REGISTRY:
        raise ValueError(f"Unknown architecture '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]


def build_model(arch: str, hidden_dim: int, num_layers: int,
                dropout: float, readout: str, **kwargs):
    """Instantiate a model by arch name, filtering kwargs to accepted fields."""
    import dataclasses
    cls = get_model_class(arch)
    accepted = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    return cls(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        readout=readout,
        **filtered,
    )
