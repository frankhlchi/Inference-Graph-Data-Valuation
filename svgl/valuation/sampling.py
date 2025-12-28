"""
Permutation sampling for Shapley value estimation.

Implements structure-aware sampling following connectivity constraints.
"""

import os
import pickle
import random
import numpy as np
import torch
from torch_geometric.utils import k_hop_subgraph
from tqdm import tqdm
from typing import Optional

from .features import evaluate_subgraph, compute_train_features


def get_candidate_nodes(data, active_nodes: set, hop: int, exclude_nodes: set, device: str) -> set:
    """Get candidate nodes within hop distance, excluding already visited nodes."""
    candidates = set()
    for node in active_nodes:
        new_nodes, _, _, _ = k_hop_subgraph(
            node, hop, data.edge_index,
            relabel_nodes=False, num_nodes=data.num_nodes
        )
        candidates.update(new_nodes.cpu().numpy().tolist())
    return candidates - exclude_nodes


def get_2hop_neighbors(data, node: int) -> set:
    """Get all nodes within 2-hop distance of a given node."""
    subset, _, _, _ = k_hop_subgraph(
        node, num_hops=2, edge_index=data.edge_index,
        relabel_nodes=False, num_nodes=data.num_nodes
    )
    return set(subset.tolist())


def sample_permutations(model, data, target_nodes: list, split_data: dict,
                        num_samples: int = 30, output_dir: str = './outputs/results/',
                        random_seed: int = 0, device: str = 'cpu',
                        mask_type: str = 'val', verbose: bool = True,
                        neighbors_cache_path: Optional[str] = None,
                        skip_existing: bool = False,
                        lightweight: bool = False) -> None:
    """
    Sample permutations and evaluate subgraphs for Shapley value estimation.

    Args:
        model: Trained GNN model
        data: PyG Data object
        target_nodes: List of target node indices
        split_data: Preprocessed split data
        num_samples: Number of permutation samples
        output_dir: Directory to save results
        random_seed: Base random seed
        device: Computation device
        mask_type: 'val' or 'test'
        verbose: Whether to print progress
    """
    os.makedirs(output_dir, exist_ok=True)

    # Determine which samples still need to be generated.
    sample_indices = list(range(num_samples))
    if skip_existing:
        sample_indices = [
            i for i in sample_indices
            if not os.path.exists(os.path.join(output_dir, f'sample_{i}.pkl'))
        ]
        if not sample_indices:
            if verbose:
                print(f"All {num_samples} {mask_type} samples already exist in: {output_dir}")
            return

    # Compute training features
    train_prototype, train_class_features, full_edge_count = compute_train_features(
        model, data, split_data, device, mask_type=mask_type
    )

    # Compute (or load) 2-hop neighbors for the target nodes only.
    # This matches the original algorithm usage (only target nodes are queried),
    # and avoids building an enormous dict for large graphs.
    target_nodes = [int(n) for n in target_nodes]
    all_2hop_neighbors = {}
    if neighbors_cache_path and os.path.exists(neighbors_cache_path):
        with open(neighbors_cache_path, 'rb') as f:
            cached = pickle.load(f)
            if isinstance(cached, dict):
                all_2hop_neighbors = cached

    missing = [n for n in target_nodes if n not in all_2hop_neighbors]
    for node in missing:
        all_2hop_neighbors[node] = get_2hop_neighbors(data, node)

    if neighbors_cache_path and missing:
        os.makedirs(os.path.dirname(neighbors_cache_path) or '.', exist_ok=True)
        with open(neighbors_cache_path, 'wb') as f:
            pickle.dump(all_2hop_neighbors, f)

    # Get target predictions using full graph
    target_mask = data.val_mask if mask_type == 'val' else data.test_mask
    with torch.no_grad():
        logits, _ = model(data)
        target_predictions = logits[target_mask].argmax(dim=1)

        # Get original predictions without edges
        data_no_edges = data.clone()
        data_no_edges.edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        original_out, _ = model(data_no_edges)
        original_pred = original_out[target_mask].argmax(dim=1)

    iterator = sample_indices
    if verbose:
        iterator = tqdm(iterator, desc=f"Sampling {mask_type} permutations")

    for i in iterator:
        np.random.seed(random_seed + i)
        random.seed(random_seed + i)

        sample_path = os.path.join(output_dir, f'sample_{i}.pkl')

        # Sample one permutation
        sample_results = _sample_single_permutation(
            model, data, target_nodes, all_2hop_neighbors,
            train_prototype, train_class_features, target_predictions,
            original_pred, full_edge_count, device, mask_type,
            lightweight=lightweight
        )

        # Save results
        with open(sample_path, 'wb') as f:
            pickle.dump(sample_results, f)

    if verbose:
        print(f"Results saved to: {output_dir}")


def _sample_single_permutation(model, data, target_nodes: list,
                               all_2hop_neighbors: dict,
                               train_prototype: torch.Tensor,
                               train_class_features: torch.Tensor,
                               target_predictions: torch.Tensor,
                               original_pred: torch.Tensor,
                               full_edge_count: int,
                               device: str, mask_type: str,
                               lightweight: bool = False) -> dict:
    """Sample a single permutation and evaluate subgraphs at each step."""
    visited_nodes = set(target_nodes)
    all_neighbors = set.union(*[all_2hop_neighbors[node] for node in target_nodes])
    active_nodes = get_candidate_nodes(data, set(target_nodes), 1, visited_nodes, device)

    neighbor_counts = {node: 0 for node in target_nodes}
    sample_results = {}

    # Step 0: Initial evaluation (target nodes only)
    data_clone = data.clone()
    subgraph_nodes = list(visited_nodes)
    node_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    node_mask[subgraph_nodes] = True
    edge_mask = node_mask[data_clone.edge_index[0]] & node_mask[data_clone.edge_index[1]]
    data_clone.edge_index = data_clone.edge_index[:, edge_mask]

    results = evaluate_subgraph(
        model, data_clone, subgraph_nodes, device,
        train_prototype, target_predictions, neighbor_counts,
        0, full_edge_count, train_class_features, original_pred, mask_type
    )
    if lightweight:
        _lighten_step_results(results, mask_type)
    results['node'] = None
    sample_results[0] = results

    step = 1
    while active_nodes and len(visited_nodes) < len(all_neighbors):
        # Sample next node
        new_node = random.sample(list(active_nodes), 1)[0]
        active_nodes.remove(new_node)
        visited_nodes.add(new_node)

        # Update neighbor counts
        for target in target_nodes:
            if new_node in all_2hop_neighbors[target]:
                neighbor_counts[target] += 1

        # Update active nodes
        new_candidates = get_candidate_nodes(data, {new_node}, 1, visited_nodes, device)
        active_nodes.update(new_candidates & all_neighbors)

        # Update subgraph
        data_clone = data.clone()
        subgraph_nodes = list(visited_nodes)
        node_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
        node_mask[subgraph_nodes] = True
        edge_mask = node_mask[data_clone.edge_index[0]] & node_mask[data_clone.edge_index[1]]
        data_clone.edge_index = data_clone.edge_index[:, edge_mask]

        # Evaluate
        results = evaluate_subgraph(
            model, data_clone, subgraph_nodes, device,
            train_prototype, target_predictions, neighbor_counts,
            step, full_edge_count, train_class_features, original_pred, mask_type
        )
        if lightweight:
            _lighten_step_results(results, mask_type)
        results['node'] = new_node
        sample_results[step] = results

        step += 1

    if visited_nodes != all_neighbors:
        missing = all_neighbors - visited_nodes
        raise RuntimeError(
            f"Not all 2-hop neighbors were visited (missing={len(missing)}). "
            f"mask_type={mask_type}, target_nodes={len(target_nodes)}"
        )

    return sample_results


def _lighten_step_results(step_results: dict, mask_type: str) -> None:
    """Reduce per-step payload size (for large-scale sampling)."""
    array_keys_to_mean = [
        'cosine_similarities',
        'max_probs',
        'predicted_class_confidences',
        'propagated_max_probs',
        'propagated_target_probs',
    ]
    for key in array_keys_to_mean:
        if key in step_results:
            step_results[key] = float(np.mean(step_results[key]))

    for key in (
        f'{mask_type}_probs',
        f'{mask_type}_pred',
        f'{mask_type}_labels',
        'neighbor_counts',
    ):
        step_results.pop(key, None)
