"""
Preprocess a MoleculeNet dataset into a cached .npz file.

Two modes:

  1. Known dataset — downloads CSV from DeepChem S3 on first run, caches in data/raw/:
        uv run python scripts/preprocess.py --dataset lipophilicity

  2. Local CSV — point at any CSV you already have:
        uv run python scripts/preprocess.py \\
            --csv path/to/my.csv \\
            --smiles_col smiles \\
            --target_cols exp \\
            --task regression \\
            --name my_dataset

No DeepChem, PyTorch, or TensorFlow required.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from mini_gnn.data.datasets import save_dataset
from mini_gnn.data.descriptors import DescriptorScaler, rdkit_2d_descriptors
from mini_gnn.data.normalization import TargetNormalizer
from mini_gnn.data.smiles import smiles_to_graph
from mini_gnn.data.splits import random_split, scaffold_split


# ---- known dataset registry ----

_S3 = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets"

KNOWN: dict[str, dict] = {
    "esol": {
        "url":         f"{_S3}/delaney-processed.csv",
        "smiles_col":  "smiles",
        "target_cols": ["measured log solubility in mols per litre"],
        "task":        "regression",
    },
    "lipophilicity": {
        "url":         f"{_S3}/Lipophilicity.csv",
        "smiles_col":  "smiles",
        "target_cols": ["exp"],
        "task":        "regression",
    },
    "bbbp": {
        "url":         f"{_S3}/BBBP.csv",
        "smiles_col":  "smiles",
        "target_cols": ["p_np"],
        "task":        "classification",
    },
    "bace": {
        "url":         f"{_S3}/bace.csv",
        "smiles_col":  "mol",
        "target_cols": ["pIC50"],
        "task":        "regression",
    },
    "tox21": {
        "url":         f"{_S3}/tox21.csv",
        "smiles_col":  "smiles",
        "target_cols": [
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
            "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
            "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
        ],
        "task":        "multilabel_classification",
    },
}


# ---- loaders ----

def _load_known(name: str) -> tuple[list[str], np.ndarray, str]:
    info    = KNOWN[name]
    raw_dir = ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / f"{name}.csv"

    if not csv_path.exists():
        print(f"Downloading {name} from {info['url']} …")
        urllib.request.urlretrieve(info["url"], csv_path)
        print(f"  Cached at {csv_path}")
    else:
        print(f"Using cached CSV: {csv_path}")

    df      = pd.read_csv(csv_path)
    smiles  = df[info["smiles_col"]].tolist()
    targets = np.nan_to_num(
        df[info["target_cols"]].values.astype(np.float32), nan=0.0
    )
    return smiles, targets, info["task"]


def _load_local(
    csv_path:    str,
    smiles_col:  str,
    target_cols: list[str],
    task:        str,
) -> tuple[list[str], np.ndarray, str]:
    print(f"Loading local CSV: {csv_path}")
    df      = pd.read_csv(csv_path)
    smiles  = df[smiles_col].tolist()
    targets = np.nan_to_num(
        df[target_cols].values.astype(np.float32), nan=0.0
    )
    return smiles, targets, task


# ---- core processing (shared) ----

def preprocess(
    name:         str,
    smiles_all:   list[str],
    targets_all:  np.ndarray,
    task:         str,
    split_type:   str   = "scaffold",
    train_frac:   float = 0.8,
    val_frac:     float = 0.1,
    seed:         int   = 42,
) -> None:
    N         = len(smiles_all)
    n_targets = targets_all.shape[1]
    print(f"Loaded {N} molecules, {n_targets} target(s), task={task}")

    # ---- build graphs ----
    print("Building 2D molecular graphs …")
    graphs, valid_idx = [], []
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
        tr_idx, va_idx, te_idx = scaffold_split(
            smiles_valid, train=train_frac, val=val_frac, seed=seed
        )
    else:
        tr_idx, va_idx, te_idx = random_split(
            len(graphs), train=train_frac, val=val_frac, seed=seed
        )
    print(f"  train={len(tr_idx)}  val={len(va_idx)}  test={len(te_idx)}")

    # ---- RDKit 2D descriptors ----
    print("Computing RDKit 2D descriptors …")
    descriptors_raw = rdkit_2d_descriptors(smiles_valid)
    scaler = DescriptorScaler().fit(descriptors_raw[tr_idx])

    # ---- target normalisation (regression only) ----
    if task == "regression":
        norm = TargetNormalizer.fit(targets_valid[tr_idx, 0])
        target_mean, target_std = norm.mean, norm.std
        print(f"  target mean={target_mean:.4f}  std={target_std:.4f}")
    else:
        target_mean, target_std = 0.0, 1.0

    # ---- padding sizes ----
    max_nodes = max(int(g.n_node[0]) for g in graphs) + 2
    max_edges = max(int(g.n_edge[0]) for g in graphs) + 4
    print(f"  max_nodes={max_nodes}  max_edges={max_edges}")

    # ---- save ----
    out_path = ROOT / "data" / "cache" / f"{name}.npz"
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
        n_targets=n_targets,
    )
    print(f"Saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess a dataset into a .npz cache for mini-gnn-toolkit"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dataset", choices=list(KNOWN),
        help="Known MoleculeNet dataset — downloaded automatically",
    )
    mode.add_argument(
        "--csv", metavar="PATH",
        help="Path to a local CSV file",
    )

    # local CSV options (only used with --csv)
    parser.add_argument("--smiles_col",  default="smiles")
    parser.add_argument("--target_cols", nargs="+", default=["target"])
    parser.add_argument("--task",
                        choices=["regression", "classification", "multilabel_classification"],
                        default="regression")
    parser.add_argument("--name", default="custom",
                        help="Output name for the .npz file (used with --csv)")

    parser.add_argument("--split", default="scaffold",
                        choices=["scaffold", "random"])
    parser.add_argument("--seed",  type=int, default=42)

    args = parser.parse_args()

    if args.dataset:
        smiles, targets, task = _load_known(args.dataset)
        name = args.dataset
    else:
        smiles, targets, task = _load_local(
            args.csv, args.smiles_col, args.target_cols, args.task
        )
        name = args.name

    preprocess(name, smiles, targets, task, split_type=args.split, seed=args.seed)


if __name__ == "__main__":
    main()
