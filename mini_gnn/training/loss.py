"""Loss functions for regression and (binary/multi-label) classification."""
from __future__ import annotations

import jax.numpy as jnp

from mini_gnn.data.graph_tuple import targets_and_mask
from mini_gnn.utils.types import Array


def masked_mse(preds: Array, batch) -> Array:
    targets, mask = targets_and_mask(batch)   # (G, T), (G,)
    mask = mask[:, None]                      # (G, 1) broadcast over targets
    sq_err = ((preds - targets) ** 2) * mask
    return sq_err.sum() / jnp.maximum(mask.sum(), 1.0)


def masked_bce(preds: Array, batch) -> Array:
    """Numerically stable sigmoid BCE — preds are logits."""
    targets, mask = targets_and_mask(batch)
    mask = mask[:, None]
    # log(1 + exp(-|x|)) + max(x, 0) - x*y  (stable form)
    bce = jnp.maximum(preds, 0) - preds * targets + jnp.log1p(jnp.exp(-jnp.abs(preds)))
    return (bce * mask).sum() / jnp.maximum(mask.sum(), 1.0)


def compute_loss(preds: Array, batch, task: str) -> Array:
    if task == "regression":
        return masked_mse(preds, batch)
    else:
        return masked_bce(preds, batch)
