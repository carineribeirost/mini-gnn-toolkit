"""Training loop with early stopping and optional checkpointing."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

from mini_gnn.data.batching import make_batches
from mini_gnn.data.datasets import split_dataset
from mini_gnn.data.graph_tuple import targets_and_mask
from mini_gnn.models.base import build_model
from mini_gnn.training.config import Config
from mini_gnn.training.loss import compute_loss
from mini_gnn.training.metrics import compute_metrics


# ---- optimizer helpers ----

def _make_schedule(cfg):
    steps = cfg.train.epochs * 200   # rough upper bound; decays to near-zero
    if cfg.train.scheduler == "constant":
        return optax.constant_schedule(cfg.train.lr)
    if cfg.train.scheduler == "cosine":
        return optax.cosine_decay_schedule(cfg.train.lr, decay_steps=steps)
    if cfg.train.scheduler == "warmup_cosine":
        return optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=cfg.train.lr,
            warmup_steps=cfg.train.warmup_steps,
            decay_steps=steps,
            end_value=1e-6,
        )
    raise ValueError(f"Unknown scheduler: {cfg.train.scheduler!r}")


def _make_optimizer(cfg) -> optax.GradientTransformation:
    schedule = _make_schedule(cfg)
    chain = []
    if cfg.train.grad_clip > 0:
        chain.append(optax.clip_by_global_norm(cfg.train.grad_clip))
    chain.append(optax.adamw(schedule, weight_decay=cfg.train.weight_decay))
    return optax.chain(*chain)


# ---- evaluation helper ----

def eval_epoch(
    apply_fn,
    params: Any,
    batches: list,
    task: str,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Run inference over batches.

    Returns:
        mean_loss  — average loss over batches
        preds      — (N_real, n_targets) float32
        targets    — (N_real, n_targets) float32
    """
    jit_apply = jax.jit(apply_fn)
    total_loss = 0.0
    all_preds, all_targets = [], []

    for batch in batches:
        preds_jax = jit_apply(params, batch, train=False)
        total_loss += float(compute_loss(preds_jax, batch, task))

        preds_np   = np.asarray(preds_jax)
        tgts_np, mask_np = targets_and_mask(batch)
        tgts_np  = np.asarray(tgts_np)
        mask_np  = np.asarray(mask_np)

        real = mask_np > 0.5
        all_preds.append(preds_np[real])
        all_targets.append(tgts_np[real])

    mean_loss = total_loss / max(len(batches), 1)
    preds_cat   = np.concatenate(all_preds,   axis=0) if all_preds   else np.empty((0, 1))
    targets_cat = np.concatenate(all_targets, axis=0) if all_targets else np.empty((0, 1))
    return mean_loss, preds_cat, targets_cat


# ---- checkpoint ----

def _save_checkpoint(checkpoint_dir: str | Path, params: Any, epoch: int) -> None:
    try:
        import orbax.checkpoint as ocp
        path = Path(checkpoint_dir) / f"epoch_{epoch:04d}"
        ocp.StandardCheckpointer().save(str(path), params)
    except Exception as exc:
        print(f"[checkpoint] save failed: {exc}")


# ---- main training function ----

def train(
    cfg: Config,
    dataset: dict,
    *,
    checkpoint_dir: str | Path | None = None,
    verbose: bool = True,
) -> tuple[Any, dict]:
    """
    Train a GNN described by cfg on a preprocessed dataset dict.

    Args:
        cfg:            full Config (arch, model, train, dataset sections)
        dataset:        dict returned by load_dataset()
        checkpoint_dir: if given, best params are saved here via orbax
        verbose:        print progress every 10 epochs

    Returns:
        best_params — Flax variable dict with best validation score
        history     — dict of lists: train_loss, val_loss, val_metrics
    """
    task = cfg.dataset.task
    rng  = jax.random.PRNGKey(cfg.seed)

    # ---- data splits ----
    train_data = split_dataset(dataset, "train")
    val_data   = split_dataset(dataset, "val")

    def _make_train_batches(seed_int: int):
        return make_batches(
            train_data["graphs"],
            batch_size=cfg.train.batch_size,
            max_nodes=cfg.dataset.max_nodes,
            max_edges=cfg.dataset.max_edges,
            shuffle=True,
            seed=seed_int,
        )

    val_batches = make_batches(
        val_data["graphs"],
        batch_size=cfg.train.batch_size,
        max_nodes=cfg.dataset.max_nodes,
        max_edges=cfg.dataset.max_edges,
        shuffle=False,
    )

    # ---- model + init ----
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

    rng, init_rng = jax.random.split(rng)
    init_batch = _make_train_batches(cfg.seed)[0]
    params = model.init(init_rng, init_batch)

    # ---- optimizer + train state ----
    optimizer = _make_optimizer(cfg)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )

    # ---- JIT train step ----
    @jax.jit
    def train_step(state, batch, dropout_rng):
        def loss_fn(params):
            preds = state.apply_fn(
                params, batch, train=True,
                rngs={"dropout": dropout_rng},
            )
            return compute_loss(preds, batch, task)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads), loss

    # ---- early stopping setup ----
    is_regression  = task == "regression"
    primary_metric = "rmse" if is_regression else "roc_auc"
    best_val       = float("inf") if is_regression else float("-inf")
    best_params    = state.params
    patience_count = 0

    history: dict[str, list] = {"train_loss": [], "val_loss": [], "val_metrics": []}

    for epoch in range(cfg.train.epochs):
        # shuffle train batches each epoch
        rng, data_rng, dropout_rng = jax.random.split(rng, 3)
        train_batches = _make_train_batches(int(jax.random.randint(data_rng, (), 0, 2**30)))

        epoch_loss = 0.0
        for batch in train_batches:
            rng, step_rng = jax.random.split(rng)
            state, loss = train_step(state, batch, step_rng)
            epoch_loss += float(loss)
        epoch_loss /= len(train_batches)

        val_loss, val_preds, val_targets = eval_epoch(
            model.apply, state.params, val_batches, task
        )
        val_metrics = compute_metrics(val_preds, val_targets, task)

        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)
        history["val_metrics"].append(val_metrics)

        # check improvement
        current = val_metrics.get(primary_metric, val_loss)
        improved = (current < best_val) if is_regression else (current > best_val)

        if improved:
            best_val       = current
            best_params    = state.params
            patience_count = 0
            if checkpoint_dir is not None:
                _save_checkpoint(checkpoint_dir, best_params, epoch)
        else:
            patience_count += 1
            if patience_count >= cfg.train.patience:
                if verbose:
                    print(f"[epoch {epoch + 1}] early stopping (patience={cfg.train.patience})")
                break

        if verbose and (epoch + 1) % 10 == 0:
            mstr = "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
            print(f"[{epoch+1:4d}] train={epoch_loss:.4f}  val={val_loss:.4f}  {mstr}")

    return best_params, history
