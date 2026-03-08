# mini-gnn-toolkit

A lightweight toolkit for training and benchmarking graph neural network architectures on drug discovery datasets. Molecules are represented as 2D molecular graphs built from SMILES. All GNNs are implemented in JAX + jraph + Flax and compared side-by-side against classical baselines.

## Overview

**GNN architectures:** GCN, GIN, MPNN, GAT, PNA

**Classical baselines:** Random Forest and Kernel Ridge Regression on RDKit 2D physicochemical descriptors (208 features)

**Datasets (MoleculeNet):**
| Dataset | Task | Molecules |
|---|---|---|
| ESOL | Regression (solubility) | 1128 |
| Lipophilicity | Regression (logD) | 4200 |
| BACE | Regression (pIC50) | 1513 |
| BBBP | Classification (BBB penetration) | 2039 |
| Tox21 | Multi-label classification (12 tasks) | 7831 |

**Splitting:** Murcko scaffold split by default (same as MoleculeNet benchmarks)

**Metrics:** MAE, RMSE, R² for regression — ROC-AUC, PR-AUC, accuracy for classification

---

## Installation

Requires Python 3.11+. Uses [uv](https://github.com/astral-sh/uv) for environment management.

```bash
git clone git@github.com:carineribeirost/mini-gnn-toolkit.git
cd mini-gnn-toolkit
uv sync
```

> **GPU note:** `jax[cuda12]` is included but requires a GPU with compute capability sm_70+. For older GPUs (e.g. GTX 1050 Ti), set `JAX_PLATFORMS=cpu` before running anything.

For GPU baseline acceleration (optional, NVIDIA only):
```bash
uv sync --extra rapids
```

---

## Quickstart

### 1. Preprocess a dataset

Downloads the CSV from the DeepChem S3 bucket on first run, caches it in `data/raw/`, and writes a `.npz` to `data/cache/`.

```bash
uv run python scripts/preprocess.py --dataset lipophilicity
```

Or point at a local CSV you already have:

```bash
uv run python scripts/preprocess.py \
    --csv path/to/my.csv \
    --smiles_col smiles \
    --target_cols activity \
    --task regression \
    --name my_dataset
```

### 2. Run the full benchmark

Trains all 5 GNN architectures and both baselines, prints a comparison table, and saves results and plots to `--out`.

```bash
uv run python main.py evaluate \
    --dataset_path data/cache/lipophilicity.npz \
    --out results/lipophilicity
```

Example output:
```
Test-split results (regression)
------------------------------------
model     mae    rmse      r2
------------------------------------
gcn    0.6821  0.8934  0.4123
gin    0.6102  0.8241  0.4891
mpnn   0.5934  0.7998  0.5102
gat    0.6234  0.8112  0.4987
pna    0.5801  0.7834  0.5234
rf     0.6501  0.8712  0.4312
krr    0.7012  0.9123  0.3901
------------------------------------
```

### 3. Train a single architecture

```bash
uv run python main.py train \
    --arch gin \
    --config configs/gin.toml \
    --dataset_path data/cache/lipophilicity.npz \
    --out results/gin_lipo
```

---

## Project structure

```
mini-gnn-toolkit/
├── mini_gnn/
│   ├── data/
│   │   ├── featurizers.py      # atom (56-dim) and bond (12-dim) features from SMILES
│   │   ├── smiles.py           # SMILES -> jraph.GraphsTuple
│   │   ├── structure.py        # ASE structure -> GraphsTuple (3D, optional)
│   │   ├── descriptors.py      # RDKit 2D physicochemical descriptors
│   │   ├── datasets.py         # .npz save/load
│   │   ├── batching.py         # padded batching for JIT-compatible static shapes
│   │   ├── splits.py           # scaffold and random splits
│   │   └── normalization.py    # target standardisation
│   ├── models/
│   │   ├── gcn.py              # Graph Convolutional Network
│   │   ├── gin.py              # Graph Isomorphism Network
│   │   ├── mpnn.py             # Message Passing Neural Network (edge features + GRU)
│   │   ├── gat.py              # Graph Attention Network (multi-head)
│   │   ├── pna.py              # Principal Neighbourhood Aggregation
│   │   ├── readout.py          # sum / mean / max / attention graph readout
│   │   └── mlp.py              # shared MLP block
│   ├── training/
│   │   ├── config.py           # frozen dataclasses + toml loader
│   │   ├── loss.py             # masked MSE and BCE (dummy-graph-aware)
│   │   ├── metrics.py          # MAE, RMSE, R², ROC-AUC, PR-AUC
│   │   └── trainer.py          # JIT train step, early stopping, checkpointing
│   ├── baselines/
│   │   ├── rf.py               # Random Forest (sklearn / cuML)
│   │   ├── krr.py              # Kernel Ridge Regression (sklearn / cuML)
│   │   └── runner.py           # trains both baselines, returns unified metrics
│   └── evaluation/
│       ├── results.py          # merge and serialise results dicts
│       ├── tables.py           # pretty-print comparison tables, save CSV
│       ├── plots.py            # bar charts, learning curves, pred-vs-true
│       └── runner.py           # full experiment: train all archs + baselines + report
├── configs/
│   ├── datasets/               # per-dataset config (max_nodes, task, split, ...)
│   ├── gcn.toml / gin.toml / mpnn.toml / gat.toml / pna.toml
│   ├── baseline_rf.toml
│   └── baseline_krr.toml
├── scripts/
│   └── preprocess.py           # CSV -> .npz pipeline
├── notebooks/
│   ├── 01_data_pipeline_tests.ipynb
│   ├── 02_model_infrastructure_tests.ipynb
│   ├── 03_gnn_models_tests.ipynb
│   └── 04_benchmark.ipynb
└── main.py                     # CLI entry point
```

---

## Configuration

All hyperparameters are controlled via TOML files. Example:

```toml
# configs/gin.toml
[model]
hidden_dim = 128
num_layers = 4
dropout    = 0.1
readout    = "mean"

[train]
lr           = 1e-3
weight_decay = 1e-5
epochs       = 200
batch_size   = 64
patience     = 30
grad_clip    = 1.0
scheduler    = "cosine"

[dataset]
name      = "lipophilicity"
task      = "regression"
n_targets = 1
max_nodes = 117
max_edges = 240
```

Pass a config to the CLI with `--config configs/gin.toml`.

---

## Graph representation

**Atom features (56-dim):** period, group, degree, hybridisation, formal charge, H count, aromaticity, ring membership

**Bond features (12-dim):** bond type, conjugation, ring membership, rotatable bond, stereo

All graphs use directed edges (two per bond). Single-atom molecules get a self-loop.

**Baselines** use RDKit 2D physicochemical descriptors instead of graph features: 208 descriptors including MW, logP, TPSA, HBD, HBA, rotatable bonds, ring counts, etc. Standardised using training-split statistics.

---

## Dependencies

| Package | Role |
|---|---|
| `jax` | JIT compilation, autodiff |
| `jraph` | JAX graph library (GraphsTuple, segment ops) |
| `flax` | Neural network modules |
| `optax` | Optimisers and LR schedules |
| `orbax-checkpoint` | Parameter checkpointing |
| `rdkit` | Molecule parsing, featurisation, descriptors |
| `scikit-learn` | RF, KRR, ROC-AUC metrics |
| `numpy`, `pandas` | Data handling |
| `matplotlib` | Plots |
| `ase` | 3D structure support (optional) |
