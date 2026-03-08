"""
Batching with padding for JIT-compatible fixed shapes.

Every batch is padded to (max_nodes, max_edges) by appending dummy graphs
via jraph.pad_with_graphs. The dummy graph has mask=0 so it is ignored in loss.
"""
from __future__ import annotations

import random as _random

import jraph
import numpy as np

from mini_gnn.data.graph_tuple import targets_and_mask


def make_batches(
    graphs: list[jraph.GraphsTuple],
    batch_size: int,
    max_nodes: int,
    max_edges: int,
    shuffle: bool = True,
    seed: int | None = None,
) -> list[jraph.GraphsTuple]:
    """
    Split graphs into padded batches.

    Returns a list of GraphsTuples, each with static shapes
    (max_nodes nodes, max_edges edges) ready for jax.jit.
    """
    if shuffle:
        rng = _random.Random(seed)
        graphs = list(graphs)
        rng.shuffle(graphs)

    # n_graph = batch_size + 1 (one slot reserved for the dummy padding graph).
    # This is fixed across all batches so JIT never retraces on graph count.
    n_graph = batch_size + 1

    batches = []
    for start in range(0, len(graphs), batch_size):
        chunk = graphs[start : start + batch_size]
        padded = jraph.pad_with_graphs(
            jraph.batch(chunk),
            n_node=max_nodes,
            n_edge=max_edges,
            n_graph=n_graph,
        )
        batches.append(padded)

    return batches


def batch_size_from_graph(batch: jraph.GraphsTuple) -> int:
    """Number of real (non-padding) graphs in a padded batch."""
    _, mask = targets_and_mask(batch)
    return int(np.asarray(mask).sum())
