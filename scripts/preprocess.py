"""
Preprocess a MoleculeNet dataset into a cached .npz file.

Usage:
    uv run python scripts/preprocess.py --dataset lipophilicity
    uv run python scripts/preprocess.py --dataset esol
    uv run python scripts/preprocess.py --dataset bbbp
    uv run python scripts/preprocess.py --dataset bace
    uv run python scripts/preprocess.py --dataset tox21

DeepChem is used to download and load raw CSV data.
The script builds 2D molecular graphs from SMILES, computes RDKit 2D
descriptors for baseline models, applies scaffold splitting, and writes
the cache to data/cache/<dataset>.npz.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from mini_gnn.data.datasets import save_dataset
from mini_gnn.data.descriptors import DescriptorScaler, rdkit_2d_descriptors
from mini_gnn.data.normalization import TargetNormalizer
from mini_gnn.data.smiles import smiles_to_graph
from mini_gnn.data.splits import random_split, scaffold_split


# ---- dataset loaders ----

def _load_deepchem(name: str):
    """
    Returns (smiles_list, targets, task) using DeepChem loaders.
    targets: (N,) for single-task, (N, T) for multi-task.
    """
    import deepchem as dc

    loaders = {
        "esol":          dc.molnet.load_delaney,
        "lipophilicity": dc.molnet.load_lipo,
        "bbbp":          dc.molnet.load_bbbp,
        "bace":          dc.molnet.load_bace_regression,
        "tox21":         dc.molnet.load_tox21,
    }
    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name!r}. Available: {list(loaders)}")

    print(f"Downloading / loading {name} via DeepChem …")
    tasks, (train, valid, test), _ = loaders[name](featurizer="Raw", splitter=None)

    # DeepChem Raw featurizer stores SMILES in X
    all_splits = [train, valid, test]
    smiles = []
    targets_list = []
    for split in all_splits:
        for i in range(len(split)):
            smi = split.X[i]
            y   = split.y[i]
            smiles.append(smi)
            targets_list.append(y)

    targets = np.array(targets_list, dtype=np.float32)  # (N,) or (N, T)
    if targets.ndim == 1:
        targets = targets[:, None]
    return smiles, targets, tasks


# ---- main ----

def preprocess(
    dataset_name: str,
    split_type:   str = "scaffold",
    train_frac:   float = 0.8,
    val_frac:     float = 0.1,
    seed:         int   = 42,
) -> None:
    smiles_all, targets_all, tasks = _load_deepchem(dataset_name)
    N = len(smiles_all)
    print(f"Loaded {N} molecules, {len(tasks)} task(s): {tasks}")

    # ---- build graphs ----
    print("Building 2D molecular graphs …")
    graphs = []
    valid_idx = []
    for i, (smi, tgt) in enumerate(zip(smiles_all, targets_all)):
        g = smiles_to_graph(smi, tgt)
        if g is not None:
            graphs.append(g)
            valid_idx.append(i)

    smiles_valid  = [smiles_all[i]  for i in valid_idx]
    targets_valid = targets_all[valid_idx]
    print(f"  {len(graphs)} valid graphs ({N - len(graphs)} SMILES failed)")

    # ---- split ----
    print(f"Splitting ({split_type}) …")
    if split_type == "scaffold":
        tr_idx, va_idx, te_idx = scaffold_split(smiles_valid, train=train_frac, val=val_frac, seed=seed)
    else:
        tr_idx, va_idx, te_idx = random_split(len(graphs), train=train_frac, val=val_frac, seed=seed)
    print(f"  train={len(tr_idx)}  val={len(va_idx)}  test={len(te_idx)}")

    # ---- RDKit 2D descriptors ----
    print("Computing RDKit 2D descriptors …")
    descriptors_raw = rdkit_2d_descriptors(smiles_valid)  # (N, D)

    scaler = DescriptorScaler().fit(descriptors_raw[tr_idx])
    # store raw (unscaled) — scaling is applied at training time
    # desc_mean/desc_std stored in .npz so baselines can reproduce the transform

    # ---- target normalisation (regression only) ----
    task_name = dataset_name
    is_regression = dataset_name in {"esol", "lipophilicity", "bace"}

    if is_regression:
        norm = TargetNormalizer.fit(targets_valid[tr_idx, 0])
        target_mean, target_std = norm.mean, norm.std
        print(f"  target mean={target_mean:.4f}  std={target_std:.4f}")
    else:
        target_mean, target_std = 0.0, 1.0

    # ---- compute max_nodes / max_edges ----
    max_nodes = max(int(g.n_node[0]) for g in graphs) + 2   # +2 slack for padding dummy
    max_edges = max(int(g.n_edge[0]) for g in graphs) + 4
    print(f"  max_nodes={max_nodes}  max_edges={max_edges}")

    # ---- save ----
    out_path = ROOT / "data" / "cache" / f"{dataset_name}.npz"
    save_dataset(
        path=out_path,
        graphs=graphs,
        train_idx=np.array(tr_idx, dtype=np.int32),
        val_idx=np.array(va_idx,   dtype=np.int32),
        test_idx=np.array(te_idx,  dtype=np.int32),
        descriptors=descriptors_raw,
        desc_mean=scaler.mean_,
        desc_std=scaler.std_,
        target_mean=target_mean,
        target_std=target_std,
        max_nodes=max_nodes,
        max_edges=max_edges,
        n_targets=targets_valid.shape[1],
    )
    print(f"Saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=["esol", "lipophilicity", "bbbp", "bace", "tox21"])
    parser.add_argument("--split",  default="scaffold", choices=["scaffold", "random"])
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()
    preprocess(args.dataset, split_type=args.split, seed=args.seed)


if __name__ == "__main__":
    main()
