#!/usr/bin/env python3
"""
Tune base GNN (SGC/GCN) hyperparameters and write to configs/best_hparams.yaml.

This replicates the "train_gnn.py" grid-search behavior from the original repo,
but integrates with the reorganized codebase and preprocessing.
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
import yaml

# Add parent directory to path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from svgl.data.datasets import load_dataset  # noqa: E402
from svgl.data.preprocess import preprocess_data, load_preprocessed_data  # noqa: E402
from svgl.models.tuning import tune_base_gnn_hparams  # noqa: E402
from svgl.utils import fix_seed  # noqa: E402


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(BASE_DIR, "data")


def _load_yaml(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data or {}


def _save_yaml(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=True)


def main():
    parser = argparse.ArgumentParser(description="Tune base GNN hyperparameters (grid search).")
    parser.add_argument("--datasets", nargs="+", default=["Cora"])
    parser.add_argument("--seed", type=int, default=0, help="Seed for split + init (default: 0)")
    parser.add_argument("--setting", type=str, default="inductive", choices=["inductive", "transductive"])
    parser.add_argument("--use_pmlp", action="store_true", help="Use PMLP (no edges in training)")
    parser.add_argument("--use_gnn", dest="use_pmlp", action="store_false", help="Use GNN training edges")
    parser.set_defaults(use_pmlp=True)
    parser.add_argument("--model", type=str, default="sgc", choices=["sgc", "gcn"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--search_space",
        type=str,
        default=os.path.join(BASE_DIR, "configs", "hparam_search.yaml"),
        help="YAML file with grid search space",
    )
    parser.add_argument(
        "--best_hparams_file",
        type=str,
        default=os.path.join(BASE_DIR, "configs", "best_hparams.yaml"),
        help="YAML file to update with best hyperparameters",
    )
    parser.add_argument(
        "--history_dir",
        type=str,
        default=None,
        help="Optional directory to write per-dataset tuning history JSON",
    )
    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU.")
        device = "cpu"

    search_cfg = _load_yaml(args.search_space)
    grid = search_cfg.get("defaults", search_cfg)
    required = ["hidden_dims", "num_epochs_list", "lr_list", "weight_decay_list"]
    missing = [k for k in required if k not in grid]
    if missing:
        raise ValueError(f"search_space missing keys: {missing}")

    best_map = _load_yaml(args.best_hparams_file)

    for dataset_name in args.datasets:
        print(f"\n=== Tuning {dataset_name} ({args.setting}, {'pmlp' if args.use_pmlp else 'gnn'}, {args.model}) ===")
        fix_seed(args.seed)

        dataset = load_dataset(dataset_name, root=DATA_ROOT)
        data = dataset[0].to(device)

        cache_dir = os.path.join(BASE_DIR, "outputs", "cache", "tuning", dataset_name, f"seed_{args.seed}")
        os.makedirs(cache_dir, exist_ok=True)

        try:
            split_data = load_preprocessed_data(
                dataset_name,
                cache_dir=cache_dir,
                setting=args.setting,
                use_pmlp=args.use_pmlp,
                data_seed=args.seed,
            )
            print("Loaded cached split.")
        except FileNotFoundError:
            split_data = preprocess_data(
                dataset_name=dataset_name,
                root=DATA_ROOT,
                cache_dir=cache_dir,
                setting=args.setting,
                use_pmlp=args.use_pmlp,
                data_seed=args.seed,
                device=device,
            )

        # Masks
        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
        train_mask[split_data["train_indices"]] = True
        data.train_mask = train_mask

        val_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
        val_mask[split_data["val_indices"]] = True
        data.val_mask = val_mask

        test_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
        test_mask[split_data["test_indices"]] = True
        data.test_mask = test_mask

        train_data = data.clone()
        train_data.edge_index = torch.tensor(split_data["train_edge_index"], dtype=torch.long, device=device)

        val_data = data.clone()
        val_data.edge_index = torch.tensor(split_data["val_edge_index"], dtype=torch.long, device=device)

        test_data = data.clone()
        test_data.edge_index = torch.tensor(split_data["test_edge_index"], dtype=torch.long, device=device)

        best_hparams, best_metrics, history = tune_base_gnn_hparams(
            model_name=args.model,
            num_features=data.num_features,
            num_classes=dataset.num_classes,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            seed=args.seed,
            device=device,
            hidden_dims=list(map(int, grid["hidden_dims"])),
            num_epochs_list=list(map(int, grid["num_epochs_list"])),
            lr_list=list(map(float, grid["lr_list"])),
            weight_decay_list=list(map(float, grid["weight_decay_list"])),
            verbose=True,
        )

        key = f"{dataset_name}_{args.setting}_{'pmlp' if args.use_pmlp else 'gnn'}_{args.model}"
        best_map[key] = best_hparams
        _save_yaml(args.best_hparams_file, best_map)

        print(f"\nBest key: {key}")
        print(f"Best hparams: {best_hparams}")
        print(f"Best metrics: {best_metrics}")
        print(f"Updated: {args.best_hparams_file}")

        if args.history_dir:
            os.makedirs(args.history_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(args.history_dir, f"tuning_{key}_seed{args.seed}_{ts}.json")
            with open(out_path, "w") as f:
                json.dump(
                    {
                        "key": key,
                        "seed": args.seed,
                        "best_hparams": best_hparams,
                        "best_metrics": best_metrics,
                        "history": history,
                    },
                    f,
                    indent=2,
                )
            print(f"Wrote tuning history: {out_path}")


if __name__ == "__main__":
    main()

