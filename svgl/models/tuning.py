"""
Hyperparameter tuning utilities for base GNN training (SGC/GCN).
"""

from __future__ import annotations

from itertools import product
from typing import Any, Dict, Iterable, List, Tuple

import torch

from .gnn import create_model
from ..utils import fix_seed


def iter_param_grid(search_space: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    """Yield cartesian products of a simple param grid."""
    keys = list(search_space.keys())
    values = [search_space[k] for k in keys]
    for combo in product(*values):
        yield dict(zip(keys, combo))


def tune_base_gnn_hparams(
    *,
    model_name: str,
    num_features: int,
    num_classes: int,
    train_data,
    val_data,
    test_data=None,
    seed: int = 0,
    device: str = "cpu",
    hidden_dims: List[int],
    num_epochs_list: List[int],
    lr_list: List[float],
    weight_decay_list: List[float],
    verbose: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    """
    Grid-search base GNN hyperparameters and select by validation accuracy.

    Returns:
        (best_hparams, best_metrics, history)
    """
    grid = {
        "hidden_dim": hidden_dims,
        "num_epochs": num_epochs_list,
        "lr": lr_list,
        "weight_decay": weight_decay_list,
    }

    best_hparams: Dict[str, Any] = {}
    best_metrics: Dict[str, Any] = {"val_acc": float("-inf")}
    history: List[Dict[str, Any]] = []

    for params in iter_param_grid(grid):
        fix_seed(seed)

        model = create_model(
            model_name,
            num_features,
            num_classes,
            config={"hidden_dim": int(params["hidden_dim"]), "num_layers": 2, "dropout": 0.5},
            seed=seed,
        ).to(device)

        model.fit(
            train_data,
            num_epochs=int(params["num_epochs"]),
            lr=float(params["lr"]),
            weight_decay=float(params["weight_decay"]),
            device=device,
        )

        _, val_acc = model.predict_valid(val_data, device=device)
        test_acc = None
        if test_data is not None:
            _, test_acc = model.predict(test_data, device=device)

        trial = {
            **params,
            "val_acc": float(val_acc),
            "test_acc": None if test_acc is None else float(test_acc),
        }
        history.append(trial)

        if verbose:
            msg = (
                f"trial hidden_dim={params['hidden_dim']} epochs={params['num_epochs']} "
                f"lr={params['lr']} wd={params['weight_decay']} -> val_acc={val_acc:.4f}"
            )
            if test_acc is not None:
                msg += f", test_acc={test_acc:.4f}"
            print(msg)

        if float(val_acc) > float(best_metrics["val_acc"]):
            best_hparams = {
                "hidden_dim": int(params["hidden_dim"]),
                "num_epochs": int(params["num_epochs"]),
                "lr": float(params["lr"]),
                "weight_decay": float(params["weight_decay"]),
            }
            best_metrics = {
                "val_acc": float(val_acc),
                "test_acc": None if test_acc is None else float(test_acc),
            }

        del model
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not best_hparams:
        raise RuntimeError("No hyperparameter trials were run (empty search space).")

    return best_hparams, best_metrics, history

