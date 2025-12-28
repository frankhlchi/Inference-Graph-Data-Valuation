"""
Utility functions for SVGL.
"""

import os
import json
import pickle
import random
import numpy as np
import torch


def fix_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_str: str = 'auto') -> torch.device:
    """
    Get PyTorch device.

    Args:
        device_str: 'auto', 'cpu', 'cuda', or 'cuda:N'

    Returns:
        torch.device object
    """
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device_str.startswith('cuda') and not torch.cuda.is_available():
        print(f"Warning: CUDA not available, using CPU")
        return torch.device('cpu')
    else:
        return torch.device(device_str)


def save_results(results: dict, output_path: str, format: str = 'auto'):
    """
    Save results to file.

    Args:
        results: Dictionary of results
        output_path: Path to save file
        format: 'json', 'pickle', or 'auto' (inferred from extension)
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    if format == 'auto':
        if output_path.endswith('.json'):
            format = 'json'
        else:
            format = 'pickle'

    if format == 'json':
        # Convert non-serializable objects
        serializable = _make_serializable(results)
        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2)
    else:
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)

    print(f"Results saved to: {output_path}")


def load_results(input_path: str) -> dict:
    """
    Load results from file.

    Args:
        input_path: Path to load file

    Returns:
        Dictionary of results
    """
    if input_path.endswith('.json'):
        with open(input_path, 'r') as f:
            return json.load(f)
    else:
        with open(input_path, 'rb') as f:
            return pickle.load(f)


def _make_serializable(obj):
    """Convert objects to JSON-serializable format."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif hasattr(obj, '__dict__'):
        return str(obj)
    else:
        return obj


def print_config(config: dict, title: str = "Configuration"):
    """Pretty print configuration."""
    print(f"\n{'='*50}")
    print(f" {title}")
    print('='*50)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print('='*50 + '\n')
