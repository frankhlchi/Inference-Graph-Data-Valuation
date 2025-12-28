"""
Shapley value estimation and regression for SVGL.

Implements the core Shapley-guided utility learning algorithm.
"""

import os
import pickle
import numpy as np
from scipy import stats
from tqdm import tqdm

from ..models.lasso import train_lasso, LassoRegression

# Feature methods used for Shapley value estimation
DEFAULT_METHODS = [
    'cosine_similarity',
    'avg_edge_cosine_similarity',
    'avg_max_class_cosine_similarity',
    'argmax_confidence',
    'confidence',
    'negative_entropy',
    'propagated_max_probs',
    'propagated_target_probs',
    'max_conf_gap',
    'true_acc'  # Ground truth for training
]


def estimate_shapley_values(data_dir: str, methods: list = None,
                            verbose: bool = True) -> tuple:
    """
    Estimate Shapley values from permutation samples.

    Args:
        data_dir: Directory containing sample pickle files
        methods: List of feature methods to use
        verbose: Whether to print progress

    Returns:
        Tuple of (shapley_values_dict, interaction_methods)
    """
    if methods is None:
        methods = DEFAULT_METHODS

    shapley_values = {method: {} for method in methods}

    # Create interaction features
    interaction_methods = []
    for i, m1 in enumerate(methods[:-1]):
        for m2 in methods[i:-1]:
            interaction = f"interaction({m1},{m2})"
            interaction_methods.append(interaction)
            shapley_values[interaction] = {}

    all_methods = methods + interaction_methods

    # Count samples
    sample_files = [f for f in os.listdir(data_dir)
                    if f.startswith('sample_') and f.endswith('.pkl')]

    if verbose:
        print(f"Processing {len(sample_files)} samples...")

    iterator = sample_files
    if verbose:
        iterator = tqdm(sample_files, desc="Estimating Shapley values")

    for sample_file in iterator:
        sample_path = os.path.join(data_dir, sample_file)
        with open(sample_path, 'rb') as f:
            sample_data = pickle.load(f)

        prev_scores = {method: None for method in all_methods}

        for step, step_data in sorted(sample_data.items(), key=lambda x: int(x[0])):
            # Extract current scores for each method
            for method in methods:
                current_score = _extract_method_score(method, step_data)

                if prev_scores[method] is not None:
                    marginal = current_score - prev_scores[method]
                    new_node = step_data.get('node')
                    if new_node is not None:
                        if new_node not in shapley_values[method]:
                            shapley_values[method][new_node] = 0
                        shapley_values[method][new_node] += marginal

                prev_scores[method] = current_score

            # Handle interaction features
            for interaction in interaction_methods:
                m1, m2 = interaction[12:-1].split(',')
                if prev_scores[m1] is not None and prev_scores[m2] is not None:
                    current_score = prev_scores[m1] * prev_scores[m2]

                    if prev_scores[interaction] is not None:
                        marginal = current_score - prev_scores[interaction]
                        new_node = step_data.get('node')
                        if new_node is not None:
                            if new_node not in shapley_values[interaction]:
                                shapley_values[interaction][new_node] = 0
                            shapley_values[interaction][new_node] += marginal

                    prev_scores[interaction] = current_score

    return shapley_values, interaction_methods


def _extract_method_score(method: str, step_data: dict) -> float:
    """Extract score for a method from step data."""
    if method == 'true_acc':
        return step_data.get('val_acc', step_data.get('test_acc', 0))
    elif method == 'negative_entropy':
        return step_data['negative_entropy']
    elif method == 'propagated_max_probs':
        return np.mean(step_data['propagated_max_probs'])
    elif method == 'propagated_target_probs':
        return np.mean(step_data['propagated_target_probs'])
    elif method == 'cosine_similarity':
        return np.mean(step_data['cosine_similarities'])
    elif method == 'confidence':
        return np.mean(step_data['predicted_class_confidences'])
    elif method == 'argmax_confidence':
        return np.mean(step_data['max_probs'])
    elif method in ['step', 'added_node_count', 'current_edge_count',
                    'edge_ratio', 'avg_edge_cosine_similarity',
                    'avg_max_class_cosine_similarity', 'max_conf_gap']:
        return step_data[method]
    else:
        return step_data.get(method, 0)


def prepare_regression_data(shapley_values: dict, methods: list) -> tuple:
    """
    Prepare feature matrix and target for regression.

    Args:
        shapley_values: Dictionary of Shapley values per method
        methods: List of methods (last one should be 'true_acc')

    Returns:
        Tuple of (X, y) numpy arrays
    """
    feature_methods = methods[:-1]  # Exclude true_acc
    nodes = list(shapley_values[methods[0]].keys())

    X = np.array([[shapley_values[m].get(node, 0) for m in feature_methods]
                  for node in nodes])
    y = np.array([shapley_values['true_acc'].get(node, 0) for node in nodes])

    return X, y


def shapley_regression(val_data_dir: str, test_data_dir: str = None,
                       device: str = 'cpu', verbose: bool = True) -> dict:
    """
    Train SVGL model and optionally evaluate on test data.

    Args:
        val_data_dir: Directory with validation permutation samples
        test_data_dir: Optional directory with test permutation samples
        device: Computation device
        verbose: Whether to print progress

    Returns:
        Dictionary with results
    """
    # Estimate Shapley values from validation data
    if verbose:
        print("Estimating Shapley values from validation data...")

    shapley_values, interaction_methods = estimate_shapley_values(
        val_data_dir, verbose=verbose
    )

    # Prepare regression data
    all_methods = DEFAULT_METHODS[:-1] + interaction_methods
    X, y = prepare_regression_data(shapley_values, all_methods + ['true_acc'])

    if verbose:
        print(f"Training data shape: X={X.shape}, y={y.shape}")

    # Train LASSO model
    if verbose:
        print("\nTraining SVGL model...")

    model, selected_features, best_alpha = train_lasso(
        X, y, all_methods, device=device, verbose=verbose
    )

    results = {
        'model': model,
        'selected_features': selected_features,
        'best_alpha': best_alpha,
        'feature_names': all_methods,
    }

    # Evaluate on validation data
    if verbose:
        print("\nEvaluating on validation data...")

    val_corr, val_true, val_pred = apply_model(
        model, val_data_dir, all_methods, interaction_methods,
        is_validation=True, device=device
    )

    results['val_correlation'] = val_corr
    if verbose:
        print(f"Validation margin correlation: {val_corr:.4f}")

    # Evaluate on test data if provided
    if test_data_dir is not None and os.path.exists(test_data_dir):
        if verbose:
            print("\nEvaluating on test data...")

        test_corr, test_true, test_pred = apply_model(
            model, test_data_dir, all_methods, interaction_methods,
            is_validation=False, device=device
        )

        results['test_correlation'] = test_corr
        if verbose:
            print(f"Test margin correlation: {test_corr:.4f}")

    return results


def apply_model(model: LassoRegression, data_dir: str, all_methods: list,
                interaction_methods: list, is_validation: bool = True,
                device: str = 'cpu') -> tuple:
    """
    Apply trained model to data and compute correlation.

    Args:
        model: Trained LASSO model
        data_dir: Directory with permutation samples
        all_methods: All feature method names
        interaction_methods: Interaction feature names
        is_validation: Whether processing validation data
        device: Computation device

    Returns:
        Tuple of (correlation, true_margins, pred_margins)
    """
    import torch

    true_margins = []
    pred_margins = []

    for sample_file in os.listdir(data_dir):
        if not (sample_file.startswith('sample_') and sample_file.endswith('.pkl')):
            continue

        sample_path = os.path.join(data_dir, sample_file)
        with open(sample_path, 'rb') as f:
            sample_data = pickle.load(f)

        sample_true = []
        sample_pred = []

        for step, step_data in sorted(sample_data.items(), key=lambda x: int(x[0])):
            # Extract features
            features = []
            for method in all_methods:
                if method == 'true_acc':
                    continue
                elif method in interaction_methods:
                    m1, m2 = method[12:-1].split(',')
                    idx1 = all_methods.index(m1)
                    idx2 = all_methods.index(m2)
                    features.append(features[idx1] * features[idx2])
                else:
                    features.append(_extract_method_score(method, step_data))

            features_tensor = torch.FloatTensor(features).reshape(1, -1).to(device)
            with torch.no_grad():
                pred = model(features_tensor).item()

            acc_key = 'val_acc' if is_validation else 'test_acc'
            sample_true.append(step_data.get(acc_key, 0))
            sample_pred.append(pred)

        true_margins.extend(np.diff(sample_true))
        pred_margins.extend(np.diff(sample_pred))

    true_margins = np.array(true_margins)
    pred_margins = np.array(pred_margins)

    # Filter invalid values
    mask = np.isfinite(true_margins) & np.isfinite(pred_margins)
    true_margins = true_margins[mask]
    pred_margins = pred_margins[mask]

    if len(true_margins) > 0:
        correlation, _ = stats.pearsonr(true_margins, pred_margins)
    else:
        correlation = np.nan

    return correlation, true_margins, pred_margins
