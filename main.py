"""
mini-gnn-toolkit CLI

Sub-commands:
    preprocess   Build .npz cache from a MoleculeNet dataset
    train        Train a single GNN architecture
    evaluate     Train all archs + baselines, print comparison table
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_preprocess(args):
    from scripts.preprocess import preprocess
    preprocess(args.dataset, split_type=args.split, seed=args.seed)


def _cmd_train(args):
    import tomllib
    import jax

    from mini_gnn.data.datasets import load_dataset
    from mini_gnn.training.config import (
        Config, DatasetConfig, ModelConfig, TrainConfig, load_config
    )
    from mini_gnn.training.trainer import train
    from mini_gnn.evaluation.results import save_results

    cfg = load_config(args.config) if args.config else Config(arch=args.arch)

    # dataset path can be overridden via --dataset_path
    if args.dataset_path:
        cfg = Config(
            arch=cfg.arch, model=cfg.model, train=cfg.train, seed=cfg.seed,
            dataset=DatasetConfig(
                path=args.dataset_path,
                max_nodes=cfg.dataset.max_nodes,
                max_edges=cfg.dataset.max_edges,
                task=cfg.dataset.task,
                n_targets=cfg.dataset.n_targets,
            ),
        )

    dataset = load_dataset(cfg.dataset.path)
    print(f"Loaded dataset from {cfg.dataset.path}")

    best_params, history = train(cfg, dataset, verbose=True)

    if args.out:
        save_results(
            {"arch": cfg.arch, "history": history},
            Path(args.out) / "train_history.json",
        )
        print(f"History saved to {args.out}/train_history.json")


def _cmd_evaluate(args):
    import tomllib
    from mini_gnn.training.config import ModelConfig, TrainConfig, DatasetConfig
    from mini_gnn.evaluation.runner import run_experiment

    archs = args.archs or ["gcn", "gin", "mpnn", "gat", "pna"]

    model_cfg   = ModelConfig()
    train_cfg   = TrainConfig()
    dataset_cfg = None   # None → run_experiment reads max_nodes/max_edges from .npz

    if args.config:
        from mini_gnn.training.config import load_config
        cfg = load_config(args.config)
        model_cfg, train_cfg, dataset_cfg = cfg.model, cfg.train, cfg.dataset

    run_experiment(
        dataset_path=args.dataset_path,
        archs=archs,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        dataset_cfg=dataset_cfg,
        seed=args.seed,
        out_dir=args.out,
        run_baselines_=not args.no_baselines,
        verbose=True,
    )


def main():
    parser = argparse.ArgumentParser(
        prog="mini-gnn",
        description="Mini GNN Toolkit for drug discovery benchmarks",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- preprocess ----
    p_pre = sub.add_parser("preprocess", help="Build .npz cache from MoleculeNet")
    p_pre.add_argument("--dataset", required=True,
                       choices=["esol", "lipophilicity", "bbbp", "bace", "tox21"])
    p_pre.add_argument("--split",  default="scaffold", choices=["scaffold", "random"])
    p_pre.add_argument("--seed",   type=int, default=42)
    p_pre.set_defaults(func=_cmd_preprocess)

    # ---- train ----
    p_tr = sub.add_parser("train", help="Train a single GNN")
    p_tr.add_argument("--arch",         default="gcn",
                      choices=["gcn", "gin", "mpnn", "gat", "pna"])
    p_tr.add_argument("--config",       default=None, help="Path to .toml config")
    p_tr.add_argument("--dataset_path", default=None)
    p_tr.add_argument("--out",          default=None, help="Output directory")
    p_tr.set_defaults(func=_cmd_train)

    # ---- evaluate ----
    p_ev = sub.add_parser("evaluate", help="Full benchmark: all archs + baselines")
    p_ev.add_argument("--dataset_path", required=True)
    p_ev.add_argument("--archs",    nargs="+", default=None,
                      choices=["gcn", "gin", "mpnn", "gat", "pna"])
    p_ev.add_argument("--config",   default=None, help="Path to .toml config")
    p_ev.add_argument("--out",      default=None, help="Output directory for results/plots")
    p_ev.add_argument("--seed",     type=int, default=42)
    p_ev.add_argument("--no_baselines", action="store_true")
    p_ev.set_defaults(func=_cmd_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
