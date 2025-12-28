#!/usr/bin/env python3
"""
Large-scale parallel reproduction experiments for SVGL.

Runs experiments across multiple datasets with GPU acceleration.
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm
import yaml

from svgl.data.datasets import load_dataset, get_dataset_info
from svgl.data.preprocess import preprocess_data, load_preprocessed_data
from svgl.models.gnn import create_model
from svgl.valuation.sampling import sample_permutations
from svgl.valuation.shapley import estimate_shapley_values, shapley_regression
from svgl.utils import fix_seed


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(BASE_DIR, 'data')


# Datasets to reproduce
DATASETS = [
    'Cora', 'Citeseer', 'Pubmed',          # Planetoid
    'CS', 'Physics',                       # Coauthor
    'Computers', 'Photo',                  # Amazon
    'WikiCS',                              # Wikipedia
    'chameleon', 'squirrel',               # WikipediaNetwork
    'Roman-empire', 'Amazon-ratings',      # HeterophilousGraphDataset (paper)
]

# Default experiment configuration
DEFAULT_CONFIG = {
    'num_samples': 30,
    'model': 'sgc',
    'setting': 'inductive',
    'use_pmlp': True,
    'num_epochs': 400,
    'lr': 0.01,
    'weight_decay': 5e-4,
    'hidden_dim': 128,
}


def _load_best_hparams(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return data or {}


def _apply_best_hparams(dataset_name: str, base_config: dict, best_map: dict):
    """
    Apply dataset-specific overrides from a best-hparams map.

    Key format:
        <Dataset>_<setting>_<pmlp|gnn>_<model>
    """
    config = dict(base_config)
    if not best_map:
        return config, None

    key = f"{dataset_name}_{config['setting']}_{'pmlp' if config['use_pmlp'] else 'gnn'}_{config['model']}"
    overrides = best_map.get(key) or best_map.get(dataset_name) or {}

    if not isinstance(overrides, dict) or not overrides:
        return config, None

    for field in ('hidden_dim', 'num_epochs', 'lr', 'weight_decay'):
        if field in overrides and overrides[field] is not None:
            config[field] = overrides[field]

    return config, key


def run_single_experiment(dataset_name: str, seed: int, config: dict,
                          device: str, base_output_dir: str,
                          best_hparams: dict = None) -> dict:
    """Run a single SVGL experiment on one dataset with one seed."""

    fix_seed(seed)
    exp_config, hparams_key = _apply_best_hparams(dataset_name, config, best_hparams or {})
    if hparams_key:
        print(
            f"  [{dataset_name}] Using best_hparams[{hparams_key}]: "
            f"hidden_dim={exp_config['hidden_dim']}, epochs={exp_config['num_epochs']}, "
            f"lr={exp_config['lr']}, wd={exp_config['weight_decay']}"
        )
    result = {
        'dataset': dataset_name,
        'seed': seed,
        'config': exp_config,
        'hparams_key': hparams_key,
        'status': 'failed',
    }

    # Create output directory for this experiment
    exp_dir = os.path.join(base_output_dir, dataset_name, f'seed_{seed}')
    val_dir = os.path.join(exp_dir, 'val_samples')
    test_dir = os.path.join(exp_dir, 'test_samples')
    cache_dir = os.path.join(exp_dir, 'cache')
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    try:
        start_time = time.time()

        # 1. Load and preprocess data
        print(f"  [{dataset_name}] Loading dataset...")
        dataset = load_dataset(dataset_name, root=DATA_ROOT)
        data = dataset[0].to(device)

        # Preprocess
        print(f"  [{dataset_name}] Preprocessing...")
        try:
            split_data = load_preprocessed_data(
                dataset_name,
                cache_dir=cache_dir,
                setting=config['setting'],
                use_pmlp=config['use_pmlp'],
                data_seed=seed,
            )
            print(f"  [{dataset_name}] Loaded cached split data")
        except FileNotFoundError:
            split_data = preprocess_data(
                dataset_name=dataset_name,
                root=DATA_ROOT,
                cache_dir=cache_dir,
                setting=config['setting'],
                use_pmlp=config['use_pmlp'],
                data_seed=seed,
                device=device
            )

        # 2. Create and train model
        print(f"  [{dataset_name}] Training {exp_config['model'].upper()} model...")
        model_config = {
            'hidden_dim': exp_config['hidden_dim'],
            'num_layers': 2,
            'dropout': 0.5
        }
        model = create_model(
            exp_config['model'],
            data.num_features,
            dataset.num_classes,
            model_config,
            seed=seed
        ).to(device)

        # Setup masks
        train_indices = split_data['train_indices']
        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
        train_mask[train_indices] = True
        data.train_mask = train_mask

        val_indices = split_data['val_indices']
        val_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
        val_mask[val_indices] = True
        data.val_mask = val_mask

        test_indices = split_data['test_indices']
        test_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
        test_mask[test_indices] = True
        data.test_mask = test_mask

        # Set up edge_index for training (PMLP vs GNN is encoded in split_data)
        train_data = data.clone()
        train_data.edge_index = torch.tensor(
            split_data['train_edge_index'], dtype=torch.long, device=device
        )

        # Train
        model.fit(train_data, num_epochs=exp_config['num_epochs'], lr=exp_config['lr'],
                  weight_decay=exp_config['weight_decay'], device=device)

        # Evaluate on the corresponding inductive/transductive subgraphs
        val_data = data.clone()
        val_data.edge_index = torch.tensor(split_data['val_edge_index'], dtype=torch.long, device=device)
        _, val_acc = model.predict_valid(val_data, device=device)

        test_data = data.clone()
        test_data.edge_index = torch.tensor(split_data['test_edge_index'], dtype=torch.long, device=device)
        _, test_acc = model.predict(test_data, device=device)
        print(f"  [{dataset_name}] Val acc: {val_acc:.4f}, Test acc: {test_acc:.4f}")

        # 3. Sample permutations for validation
        print(f"  [{dataset_name}] Sampling {exp_config['num_samples']} validation permutations...")
        sample_permutations(
            model=model,
            data=val_data,
            target_nodes=val_indices,
            split_data=split_data,
            num_samples=exp_config['num_samples'],
            output_dir=val_dir,
            random_seed=seed,
            device=device,
            mask_type='val',
            verbose=False,
            neighbors_cache_path=os.path.join(cache_dir, 'neighbors_val_2hop.pkl'),
            skip_existing=True,
            lightweight=exp_config.get('lightweight_samples', False),
        )

        # 4. Sample permutations for test
        print(f"  [{dataset_name}] Sampling {exp_config['num_samples']} test permutations...")
        sample_permutations(
            model=model,
            data=test_data,
            target_nodes=test_indices,
            split_data=split_data,
            num_samples=exp_config['num_samples'],
            output_dir=test_dir,
            random_seed=seed + 1000,  # Different seed for test
            device=device,
            mask_type='test',
            verbose=False,
            neighbors_cache_path=os.path.join(cache_dir, 'neighbors_test_2hop.pkl'),
            skip_existing=True,
            lightweight=exp_config.get('lightweight_samples', False),
        )

        # 5. Run SVGL regression
        print(f"  [{dataset_name}] Training SVGL model...")
        svgl_results = shapley_regression(
            val_data_dir=val_dir,
            test_data_dir=test_dir,
            device=device,
            verbose=False
        )

        elapsed = time.time() - start_time

        result.update({
            'status': 'success',
            'val_acc': val_acc,
            'test_acc': test_acc,
            'val_correlation': svgl_results.get('val_correlation', np.nan),
            'test_correlation': svgl_results.get('test_correlation', np.nan),
            'selected_features': svgl_results.get('selected_features', []),
            'best_alpha': svgl_results.get('best_alpha', None),
            'elapsed_time': elapsed,
        })

        print(f"✓ {dataset_name} seed={seed}: val_acc={val_acc:.4f}, "
              f"val_corr={result['val_correlation']:.4f}, "
              f"test_corr={result['test_correlation']:.4f}, time={elapsed:.1f}s")

    except Exception as e:
        import traceback
        result['error'] = str(e)
        result['traceback'] = traceback.format_exc()
        print(f"✗ {dataset_name} seed={seed}: {e}")

    # Persist per-run result for easy resume/debugging.
    result_path = os.path.join(exp_dir, 'result.json')
    try:
        with open(result_path, 'w') as f:
            json.dump(_make_serializable([result])[0], f, indent=2)
    except Exception:
        pass

    return result


def aggregate_results(all_results: list) -> dict:
    """Aggregate results across seeds for each dataset."""
    from collections import defaultdict

    dataset_results = defaultdict(list)
    for r in all_results:
        if r['status'] == 'success':
            dataset_results[r['dataset']].append(r)

    summary = {}
    for dataset, results in dataset_results.items():
        if results:
            val_accs = [r['val_acc'] for r in results]
            test_accs = [r['test_acc'] for r in results]
            val_corrs = [r['val_correlation'] for r in results if not np.isnan(r['val_correlation'])]
            test_corrs = [r['test_correlation'] for r in results if not np.isnan(r['test_correlation'])]

            summary[dataset] = {
                'val_acc_mean': np.mean(val_accs),
                'val_acc_std': np.std(val_accs),
                'test_acc_mean': np.mean(test_accs),
                'test_acc_std': np.std(test_accs),
                'val_corr_mean': np.mean(val_corrs) if val_corrs else np.nan,
                'val_corr_std': np.std(val_corrs) if val_corrs else np.nan,
                'test_corr_mean': np.mean(test_corrs) if test_corrs else np.nan,
                'test_corr_std': np.std(test_corrs) if test_corrs else np.nan,
                'num_runs': len(results),
            }

    return summary


def print_summary_table(summary: dict, datasets: list):
    """Print a formatted summary table."""
    print("\n" + "="*95)
    print("SVGL Reproduction Results Summary")
    print("="*95)
    print(f"{'Dataset':<14} {'Val Acc':<16} {'Test Acc':<16} {'Val Corr':<16} {'Test Corr':<16} {'N':<3}")
    print("-"*95)

    for dataset in datasets:
        if dataset in summary:
            s = summary[dataset]
            val_acc = f"{s['val_acc_mean']:.4f}±{s['val_acc_std']:.4f}"
            test_acc = f"{s['test_acc_mean']:.4f}±{s['test_acc_std']:.4f}"
            val_corr = f"{s['val_corr_mean']:.4f}±{s['val_corr_std']:.4f}" if not np.isnan(s['val_corr_mean']) else "N/A"
            test_corr = f"{s['test_corr_mean']:.4f}±{s['test_corr_std']:.4f}" if not np.isnan(s['test_corr_mean']) else "N/A"
            print(f"{dataset:<14} {val_acc:<16} {test_acc:<16} {val_corr:<16} {test_corr:<16} {s['num_runs']:<3}")
        else:
            print(f"{dataset:<14} {'FAILED':<16} {'-':<16} {'-':<16} {'-':<16} {'0':<3}")

    print("="*95)


def main():
    parser = argparse.ArgumentParser(description='Run SVGL parallel experiments')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Specific datasets to run (default: all)')
    parser.add_argument('--num_samples', type=int, default=30,
                        help='Number of permutation samples')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2],
                        help='Random seeds to use')
    parser.add_argument('--model', type=str, default='sgc', choices=['sgc', 'gcn'],
                        help='GNN model type')
    parser.add_argument('--setting', type=str, default=DEFAULT_CONFIG['setting'],
                        choices=['inductive', 'transductive'],
                        help='Inductive or transductive setting')
    parser.add_argument('--pmlp', dest='use_pmlp', action='store_true',
                        help='Use PMLP setting (no edges during training)')
    parser.add_argument('--gnn', dest='use_pmlp', action='store_false',
                        help='Use GNN setting (use edges during training)')
    parser.set_defaults(use_pmlp=DEFAULT_CONFIG['use_pmlp'])
    parser.add_argument('--num_epochs', type=int, default=DEFAULT_CONFIG['num_epochs'],
                        help='Training epochs')
    parser.add_argument('--lr', type=float, default=DEFAULT_CONFIG['lr'],
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=DEFAULT_CONFIG['weight_decay'],
                        help='Weight decay')
    parser.add_argument('--hidden_dim', type=int, default=DEFAULT_CONFIG['hidden_dim'],
                        help='Hidden dimension')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (cuda:0 or cpu)')
    parser.add_argument('--jobs', type=int, default=None,
                        help='Max concurrent experiments (default: auto)')
    parser.add_argument('--mp_start', type=str, default='spawn',
                        choices=['fork', 'spawn', 'forkserver'],
                        help='Multiprocessing start method')
    parser.add_argument('--allow_cuda_parallel', action='store_true',
                        help='Allow >1 worker with CUDA (unsafe)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print planned runs and exit')
    parser.add_argument('--output_dir', type=str, default='./outputs/reproduction',
                        help='Output directory for results')
    parser.add_argument('--run_dir', type=str, default=None,
                        help='Use an existing run directory (disables timestamp subdir)')
    parser.add_argument(
        '--best_hparams_file',
        type=str,
        default=os.path.join(BASE_DIR, 'configs', 'best_hparams.yaml'),
        help='YAML file with dataset-specific best hyperparameters (optional)',
    )
    parser.add_argument(
        '--no_best_hparams',
        action='store_true',
        help='Disable dataset-specific hyperparameter overrides',
    )
    parser.add_argument('--resume', action='store_true',
                        help='Skip experiments with existing successful `result.json`')
    parser.add_argument('--lightweight_samples', dest='lightweight_samples', action='store_true',
                        help='Store only scalar per-step features (recommended for scale)')
    parser.add_argument('--full_samples', dest='lightweight_samples', action='store_false',
                        help='Store full arrays per step (large outputs)')
    parser.set_defaults(lightweight_samples=True)
    args = parser.parse_args()

    # Setup
    datasets = args.datasets if args.datasets else DATASETS
    device = args.device if torch.cuda.is_available() else 'cpu'
    if device.startswith('cuda') and args.jobs and args.jobs > 1 and not args.allow_cuda_parallel:
        print("Warning: CUDA device selected; forcing --jobs=1 for safety. "
              "Use --allow_cuda_parallel to override.")
        args.jobs = 1

    if args.jobs is None:
        if device.startswith('cuda'):
            args.jobs = 1
        else:
            args.jobs = min(8, (os.cpu_count() or 1))

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.run_dir:
        output_dir = os.path.abspath(args.run_dir)
    else:
        output_dir = os.path.abspath(os.path.join(args.output_dir, timestamp))
    os.makedirs(output_dir, exist_ok=True)

    config = DEFAULT_CONFIG.copy()
    config['num_samples'] = args.num_samples
    config['model'] = args.model
    config['setting'] = args.setting
    config['use_pmlp'] = args.use_pmlp
    config['num_epochs'] = args.num_epochs
    config['lr'] = args.lr
    config['weight_decay'] = args.weight_decay
    config['hidden_dim'] = args.hidden_dim
    config['lightweight_samples'] = args.lightweight_samples
    best_hparams = {} if args.no_best_hparams else _load_best_hparams(args.best_hparams_file)

    print(f"\nSVGL Large-Scale Reproduction")
    print(f"="*50)
    print(f"Datasets: {datasets}")
    print(f"Seeds: {args.seeds}")
    print(f"Samples: {config['num_samples']}")
    print(f"Model: {config['model']}")
    print(f"Setting: {config['setting']} ({'pmlp' if config['use_pmlp'] else 'gnn'})")
    print(f"Device: {device}")
    print(f"Jobs: {args.jobs} ({args.mp_start})")
    print(f"Output: {output_dir}")
    if best_hparams:
        print(f"Best hparams: {os.path.abspath(args.best_hparams_file)} ({len(best_hparams)} keys)")
    print(f"="*50)

    # Save config
    config_file = os.path.join(output_dir, 'config.json')
    with open(config_file, 'w') as f:
        json.dump({
            'datasets': datasets,
            'seeds': args.seeds,
            'config': config,
            'device': device,
            'best_hparams_enabled': not args.no_best_hparams,
            'best_hparams_file': os.path.abspath(args.best_hparams_file),
        }, f, indent=2)

    tasks = [(dataset, seed) for dataset in datasets for seed in args.seeds]
    all_results = []
    if args.resume:
        remaining = []
        for dataset, seed in tasks:
            exp_dir = os.path.join(output_dir, dataset, f'seed_{seed}')
            existing = _load_existing_result(exp_dir)
            if existing and existing.get('status') == 'success':
                all_results.append(existing)
            else:
                remaining.append((dataset, seed))
        tasks = remaining

    if args.dry_run:
        print("\nPlanned runs:")
        for dataset, seed in tasks:
            print(f"  - {dataset} seed={seed}")
        if args.resume:
            print(f"Skipped (already done): {len(all_results)} experiments")
        print(f"Total to run: {len(tasks)} experiments")
        return [], {}

    # Avoid thread oversubscription when running many processes.
    if args.jobs and args.jobs > 1:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

    results_file = os.path.join(output_dir, 'all_results.json')

    if args.jobs == 1:
        for idx, (dataset, seed) in enumerate(tasks, start=1):
            print(f"\n[{idx}/{len(tasks)}] {dataset} seed={seed}")
            result = run_single_experiment(dataset, seed, config, device, output_dir, best_hparams)
            all_results.append(result)
            with open(results_file, 'w') as f:
                json.dump(_make_serializable(all_results), f, indent=2)
    else:
        mp_ctx = mp.get_context(args.mp_start)
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=mp_ctx) as executor:
            futures = {
                executor.submit(run_single_experiment, dataset, seed, config, device, output_dir, best_hparams): (dataset, seed)
                for dataset, seed in tasks
            }
            pbar = tqdm(total=len(futures), desc="Experiments", unit="exp")
            try:
                for future in as_completed(futures):
                    result = future.result()
                    all_results.append(result)
                    with open(results_file, 'w') as f:
                        json.dump(_make_serializable(all_results), f, indent=2)
                    pbar.update(1)
            finally:
                pbar.close()

    # Aggregate and print summary
    summary = aggregate_results(all_results)
    print_summary_table(summary, datasets)

    # Save final summary
    summary_file = os.path.join(output_dir, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump(_make_serializable_dict(summary), f, indent=2)

    print(f"\nResults saved to: {output_dir}")

    return all_results, summary


def _make_serializable(results: list) -> list:
    """Convert numpy types for JSON serialization."""
    serializable = []
    for r in results:
        sr = {}
        for k, v in r.items():
            if isinstance(v, np.ndarray):
                sr[k] = v.tolist()
            elif isinstance(v, (np.float32, np.float64)):
                sr[k] = float(v)
            elif isinstance(v, (np.int32, np.int64)):
                sr[k] = int(v)
            elif isinstance(v, float) and np.isnan(v):
                sr[k] = None
            else:
                sr[k] = v
        serializable.append(sr)
    return serializable


def _make_serializable_dict(d: dict) -> dict:
    """Convert numpy types in nested dict for JSON serialization."""
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _make_serializable_dict(v)
        elif isinstance(v, np.ndarray):
            result[k] = v.tolist()
        elif isinstance(v, (np.float32, np.float64)):
            result[k] = float(v)
        elif isinstance(v, (np.int32, np.int64)):
            result[k] = int(v)
        elif isinstance(v, float) and np.isnan(v):
            result[k] = None
        else:
            result[k] = v
    return result


def _load_existing_result(exp_dir: str):
    """Load an existing per-experiment result if present."""
    result_path = os.path.join(exp_dir, 'result.json')
    if not os.path.exists(result_path):
        return None
    try:
        with open(result_path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


if __name__ == '__main__':
    main()
