"""
Full evaluation runner:
  - Trains each GNN architecture
  - Trains RF and KRR baselines
  - Collects test-split metrics
  - Prints comparison table
  - Saves CSV + plots
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from mini_gnn.baselines.runner import run_baselines
from mini_gnn.data.datasets import load_dataset
from mini_gnn.evaluation.plots import plot_comparison, plot_learning_curves
from mini_gnn.evaluation.results import merge_results, save_results, to_rows
from mini_gnn.evaluation.tables import print_table, save_csv
from mini_gnn.training.config import Config, ModelConfig, TrainConfig, DatasetConfig
from mini_gnn.training.trainer import eval_epoch, train
from mini_gnn.models.base import build_model


def _test_metrics(
    cfg:     Config,
    dataset: dict,
    params:  Any,
) -> dict[str, float]:
    from mini_gnn.data.batching import make_batches
    from mini_gnn.data.datasets import split_dataset
    from mini_gnn.training.metrics import compute_metrics

    bs = cfg.train.batch_size
    batch_max_nodes = cfg.dataset.max_nodes * bs + 1
    batch_max_edges = cfg.dataset.max_edges * bs + 2

    test_data    = split_dataset(dataset, "test")
    test_batches = make_batches(
        test_data["graphs"],
        batch_size=bs,
        max_nodes=batch_max_nodes,
        max_edges=batch_max_edges,
        shuffle=False,
    )
    model = build_model(
        cfg.arch,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        readout=cfg.model.readout,
        n_targets=cfg.dataset.n_targets,
        num_heads=cfg.model.num_heads,
        epsilon=cfg.model.epsilon,
        delta=cfg.model.delta,
    )
    _, preds, targets = eval_epoch(model.apply, params, test_batches, cfg.dataset.task)
    return compute_metrics(preds, targets, cfg.dataset.task)


def run_experiment(
    dataset_path:   str | Path,
    archs:          list[str],
    model_cfg:      ModelConfig   | None = None,
    train_cfg:      TrainConfig   | None = None,
    dataset_cfg:    DatasetConfig | None = None,
    seed:           int  = 42,
    out_dir:        str | Path | None = None,
    run_baselines_: bool = True,
    verbose:        bool = True,
) -> dict[str, dict[str, dict]]:
    """
    Train all archs on a dataset and (optionally) train baselines.
    Prints a comparison table and saves results.

    Args:
        dataset_path:    path to preprocessed .npz
        archs:           list of GNN arch names, e.g. ["gcn","gin","mpnn","gat","pna"]
        model_cfg:       shared ModelConfig for all GNNs
        train_cfg:       TrainConfig
        dataset_cfg:     DatasetConfig (max_nodes, max_edges, task, n_targets, …)
        seed:            global seed
        out_dir:         if given, saves results.json, comparison.csv, plots/
        run_baselines_:  whether to train RF + KRR
        verbose:         print progress

    Returns:
        all_results — model_name -> {"train": {...}, "val": {...}, "test": {...}}
    """
    dataset    = load_dataset(dataset_path)
    model_cfg  = model_cfg  or ModelConfig()
    train_cfg  = train_cfg  or TrainConfig()
    dataset_cfg = dataset_cfg or DatasetConfig(
        max_nodes=dataset["max_nodes"],
        max_edges=dataset["max_edges"],
        n_targets=dataset["n_targets"],
    )
    task = dataset_cfg.task

    gnn_results: dict[str, dict[str, dict]] = {}
    histories:   dict[str, dict]            = {}

    for arch in archs:
        if verbose:
            print(f"\n{'='*50}\nTraining {arch.upper()} …\n{'='*50}")
        cfg = Config(
            arch=arch,
            model=model_cfg,
            train=train_cfg,
            dataset=dataset_cfg,
            seed=seed,
        )
        best_params, history = train(cfg, dataset, verbose=verbose)
        histories[arch] = history

        val_metrics  = history["val_metrics"][-1] if history["val_metrics"] else {}
        test_metrics = _test_metrics(cfg, dataset, best_params)

        # reconstruct train metrics from last epoch
        from mini_gnn.data.batching import make_batches
        from mini_gnn.data.datasets import split_dataset
        from mini_gnn.training.metrics import compute_metrics
        from mini_gnn.models.base import build_model as _bm

        model = _bm(arch, hidden_dim=model_cfg.hidden_dim, num_layers=model_cfg.num_layers,
                    dropout=model_cfg.dropout, readout=model_cfg.readout,
                    n_targets=dataset_cfg.n_targets, num_heads=model_cfg.num_heads,
                    epsilon=model_cfg.epsilon, delta=model_cfg.delta)
        tr_data     = split_dataset(dataset, "train")
        tr_batches  = make_batches(tr_data["graphs"], batch_size=train_cfg.batch_size,
                                   max_nodes=dataset_cfg.max_nodes * train_cfg.batch_size + 1,
                                   max_edges=dataset_cfg.max_edges * train_cfg.batch_size + 2,
                                   shuffle=False)
        _, tr_preds, tr_tgts = eval_epoch(model.apply, best_params, tr_batches, task)
        train_metrics = compute_metrics(tr_preds, tr_tgts, task)

        gnn_results[arch] = {
            "train": train_metrics,
            "val":   val_metrics,
            "test":  test_metrics,
        }

    # baselines
    baseline_results: dict[str, dict[str, dict]] = {}
    if run_baselines_:
        if verbose:
            print(f"\n{'='*50}\nBaselines\n{'='*50}")
        baseline_results = run_baselines(dataset, task)

    all_results = merge_results(gnn_results, baseline_results)

    # determine primary metric for highlighting
    primary = "rmse" if task == "regression" else "roc_auc"

    # print table
    test_rows = to_rows(all_results, split="test")
    print_table(test_rows, title=f"\nTest-split results ({task})", highlight=primary)

    # save outputs
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        save_results(all_results, out_dir / "results.json")
        save_csv(test_rows, out_dir / "comparison.csv")

        if primary in (test_rows[0] if test_rows else {}):
            lower = primary in ("mae", "rmse")
            plot_comparison(
                test_rows, metric=primary,
                title=f"Test {primary}",
                out=out_dir / "plots" / f"comparison_{primary}.png",
                lower_is_better=lower,
            )

        if histories:
            plot_learning_curves(
                histories, metric="val_loss",
                title="Validation loss",
                out=out_dir / "plots" / "val_loss_curves.png",
            )

    return all_results
