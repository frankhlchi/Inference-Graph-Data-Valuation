#!/usr/bin/env python3
"""
Quick demonstration of SVGL on a small dataset.

This script runs the complete SVGL pipeline on Cora with minimal samples
to quickly verify the installation and demonstrate the framework.

Usage:
    python scripts/run_demo.py --dataset Cora --num_samples 5
"""

import argparse
import os
import sys
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from svgl.data import load_dataset, preprocess_data, load_preprocessed_data
from svgl.models import create_model
from svgl.valuation import sample_permutations, shapley_regression
from svgl.utils import fix_seed, get_device, print_config


def train_model(model, data, split_data, config, device):
    """Train the GNN model."""
    # Prepare training data
    data = data.to(device)
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    data.train_mask[split_data['train_indices']] = True
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    data.val_mask[split_data['val_indices']] = True
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    data.test_mask[split_data['test_indices']] = True

    # Use appropriate edge index for training
    train_data = data.clone()
    train_data.edge_index = torch.tensor(split_data['train_edge_index'], device=device)

    # Train
    model.to(device)
    model.fit(train_data,
              num_epochs=config['num_epochs'],
              lr=config['lr'],
              weight_decay=config['weight_decay'],
              device=device)

    # Evaluate
    val_data = data.clone()
    val_data.edge_index = torch.tensor(split_data['val_edge_index'], device=device)

    _, val_acc = model.predict_valid(val_data, device)
    return val_acc


def main():
    parser = argparse.ArgumentParser(description='SVGL Quick Demo')
    parser.add_argument('--dataset', type=str, default='Cora',
                        choices=['Cora', 'Citeseer', 'Pubmed'],
                        help='Dataset to use (default: Cora)')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of permutation samples (default: 5)')
    parser.add_argument('--setting', type=str, default='inductive',
                        choices=['inductive', 'transductive'],
                        help='Experiment setting (default: inductive)')
    parser.add_argument('--model', type=str, default='sgc',
                        choices=['sgc', 'gcn'],
                        help='GNN model (default: sgc)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cpu, cuda, cuda:0, etc.')
    args = parser.parse_args()

    # Configuration
    config = {
        'dataset': args.dataset,
        'model': args.model,
        'setting': args.setting,
        'use_pmlp': True,
        'num_samples': args.num_samples,
        'seed': args.seed,
        'hidden_dim': 128,
        'num_epochs': 400,
        'lr': 0.01,
        'weight_decay': 5e-4,
    }

    print("\n" + "="*60)
    print(" SVGL: Shapley-Guided Utility Learning Demo")
    print("="*60)
    print_config(config, "Configuration")

    # Setup
    fix_seed(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}\n")

    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_root = os.path.join(base_dir, 'data')
    cache_dir = os.path.join(base_dir, 'outputs', 'cache')
    results_dir = os.path.join(base_dir, 'outputs', 'results', f'{args.dataset}_demo')

    os.makedirs(results_dir, exist_ok=True)

    # Step 1: Load and preprocess data
    print("Step 1: Loading and preprocessing data...")
    start = time.time()

    dataset = load_dataset(args.dataset, root=data_root)
    data = dataset[0].to(device)

    try:
        split_data = load_preprocessed_data(
            args.dataset, cache_dir=cache_dir,
            setting=args.setting, use_pmlp=config['use_pmlp'],
            data_seed=args.seed
        )
        print(f"  Loaded cached split data")
    except FileNotFoundError:
        split_data = preprocess_data(
            args.dataset, root=data_root, cache_dir=cache_dir,
            setting=args.setting, use_pmlp=config['use_pmlp'],
            data_seed=args.seed, device=str(device)
        )
        print(f"  Created new split data")

    print(f"  Dataset: {args.dataset}")
    print(f"  Nodes: {data.num_nodes}, Features: {data.num_features}, Classes: {dataset.num_classes}")
    print(f"  Train: {len(split_data['train_indices'])}, Val: {len(split_data['val_indices'])}, Test: {len(split_data['test_indices'])}")
    print(f"  Time: {time.time() - start:.2f}s\n")

    # Step 2: Train GNN model
    print("Step 2: Training GNN model...")
    start = time.time()

    model_config = {'hidden_dim': config['hidden_dim']}
    model = create_model(args.model, data.num_features, dataset.num_classes, config=model_config, seed=args.seed)

    val_acc = train_model(model, data, split_data, config, device)
    print(f"  Model: {args.model.upper()}")
    print(f"  Validation accuracy: {val_acc:.4f}")
    print(f"  Time: {time.time() - start:.2f}s\n")

    # Step 3: Permutation sampling
    print("Step 3: Sampling permutations for Shapley estimation...")
    start = time.time()

    # Prepare data for sampling
    sample_data = data.clone()
    sample_data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    sample_data.train_mask[split_data['train_indices']] = True
    sample_data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    sample_data.val_mask[split_data['val_indices']] = True
    sample_data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    sample_data.test_mask[split_data['test_indices']] = True
    sample_data.edge_index = torch.tensor(split_data['val_edge_index'], device=device)

    val_samples_dir = os.path.join(results_dir, 'val_samples')
    sample_permutations(
        model, sample_data, split_data['val_indices'], split_data,
        num_samples=args.num_samples,
        output_dir=val_samples_dir,
        random_seed=args.seed,
        device=str(device),
        mask_type='val',
        verbose=True
    )
    print(f"  Samples saved to: {val_samples_dir}")
    print(f"  Time: {time.time() - start:.2f}s\n")

    # Step 4: SVGL regression
    print("Step 4: Training SVGL model...")
    start = time.time()

    results = shapley_regression(
        val_samples_dir,
        device=str(device),
        verbose=True
    )

    print(f"\n  Selected features: {len(results['selected_features'])}")
    for name, weight in results['selected_features'][:5]:
        print(f"    - {name}: {weight:.4f}")
    if len(results['selected_features']) > 5:
        print(f"    ... and {len(results['selected_features']) - 5} more")

    print(f"\n  Validation margin correlation: {results['val_correlation']:.4f}")
    print(f"  Time: {time.time() - start:.2f}s\n")

    # Summary
    print("="*60)
    print(" Demo Complete!")
    print("="*60)
    print(f"\nResults saved to: {results_dir}")
    print("\nKey findings:")
    print(f"  - GNN validation accuracy: {val_acc:.4f}")
    print(f"  - SVGL correlation: {results['val_correlation']:.4f}")
    print(f"  - Selected {len(results['selected_features'])} features for utility prediction")
    print("\nTo run full experiments with more samples, use:")
    print(f"  python scripts/run_demo.py --dataset {args.dataset} --num_samples 30")
    print()


if __name__ == '__main__':
    main()
