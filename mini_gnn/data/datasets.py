"""
Dataset registry, loading, and .npz caching.

A cached dataset .npz contains:
    nodes_<i>, edges_<i>, senders_<i>, receivers_<i>, globals_<i>  — per graph i
    n_node_<i>, n_edge_<i>
    train_idx, val_idx, test_idx
    fingerprints           (N, n_bits)  float32
    coulomb                (N, max_atoms) float32
    target_mean, target_std  (scalars)
    max_nodes, max_edges, n_targets  (scalars)
"""
from __future__ import annotations

from pathlib import Path

import jraph
import numpy as np

from mini_gnn.data.graph_tuple import make_dummy_graph


# ---- serialisation ----

def save_dataset(
    path: str | Path,
    graphs: list[jraph.GraphsTuple],
    train_idx: np.ndarray,
    val_idx:   np.ndarray,
    test_idx:  np.ndarray,
    fingerprints: np.ndarray,
    coulomb:      np.ndarray,
    target_mean: float,
    target_std:  float,
    max_nodes: int,
    max_edges: int,
    n_targets: int,
) -> None:
    arrays: dict[str, np.ndarray] = {}
    for i, g in enumerate(graphs):
        arrays[f"nodes_{i}"]     = g.nodes
        arrays[f"edges_{i}"]     = g.edges
        arrays[f"senders_{i}"]   = g.senders
        arrays[f"receivers_{i}"] = g.receivers
        arrays[f"globals_{i}"]   = g.globals
        arrays[f"n_node_{i}"]    = g.n_node
        arrays[f"n_edge_{i}"]    = g.n_edge

    arrays["train_idx"]    = train_idx
    arrays["val_idx"]      = val_idx
    arrays["test_idx"]     = test_idx
    arrays["fingerprints"] = fingerprints
    arrays["coulomb"]      = coulomb
    arrays["target_mean"]  = np.array(target_mean, dtype=np.float32)
    arrays["target_std"]   = np.array(target_std,  dtype=np.float32)
    arrays["max_nodes"]    = np.array(max_nodes,   dtype=np.int32)
    arrays["max_edges"]    = np.array(max_edges,   dtype=np.int32)
    arrays["n_targets"]    = np.array(n_targets,   dtype=np.int32)
    arrays["n_graphs"]     = np.array(len(graphs), dtype=np.int32)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_dataset(path: str | Path) -> dict:
    data = np.load(str(path), allow_pickle=False)
    n    = int(data["n_graphs"])

    graphs = []
    for i in range(n):
        graphs.append(jraph.GraphsTuple(
            nodes=data[f"nodes_{i}"],
            edges=data[f"edges_{i}"],
            senders=data[f"senders_{i}"],
            receivers=data[f"receivers_{i}"],
            globals=data[f"globals_{i}"],
            n_node=data[f"n_node_{i}"],
            n_edge=data[f"n_edge_{i}"],
        ))

    return {
        "graphs":       graphs,
        "train_idx":    data["train_idx"],
        "val_idx":      data["val_idx"],
        "test_idx":     data["test_idx"],
        "fingerprints": data["fingerprints"],
        "coulomb":      data["coulomb"],
        "target_mean":  float(data["target_mean"]),
        "target_std":   float(data["target_std"]),
        "max_nodes":    int(data["max_nodes"]),
        "max_edges":    int(data["max_edges"]),
        "n_targets":    int(data["n_targets"]),
    }


def split_dataset(dataset: dict, split: str) -> dict:
    """Return a view of the dataset restricted to train/val/test indices."""
    idx = dataset[f"{split}_idx"]
    return {
        **dataset,
        "graphs":       [dataset["graphs"][i] for i in idx],
        "fingerprints": dataset["fingerprints"][idx],
        "coulomb":      dataset["coulomb"][idx],
    }
