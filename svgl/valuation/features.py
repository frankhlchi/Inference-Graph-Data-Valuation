"""
Feature extraction for Shapley-guided utility learning.

Extracts transferable features that capture both graph structure and model behavior.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph

from ..models.gnn import SGConvNoWeight


def _mean_edge_cosine_similarity(x: torch.Tensor, edge_index: torch.Tensor,
                                 chunk_size: int = 20000) -> float:
    """Compute mean cosine similarity over edges with bounded memory."""
    num_edges = int(edge_index.shape[1])
    if num_edges == 0:
        return 0.0

    src = edge_index[0]
    dst = edge_index[1]

    total = 0.0
    count = 0
    for start in range(0, num_edges, chunk_size):
        end = min(num_edges, start + chunk_size)
        sims = F.cosine_similarity(x[src[start:end]], x[dst[start:end]], dim=1)
        total += sims.sum().item()
        count += sims.numel()
    return total / max(count, 1)


def extract_features(model, data, subgraph_nodes: list, device: str,
                     train_prototype: torch.Tensor, target_predictions: torch.Tensor,
                     train_class_features: torch.Tensor, full_edge_count: int,
                     step: int, mask_type: str = 'val') -> dict:
    """
    Extract transferable features from a subgraph for utility prediction.

    Args:
        model: Trained GNN model
        data: PyG Data object
        subgraph_nodes: List of nodes in current subgraph
        device: Computation device
        train_prototype: Mean training embedding
        target_predictions: Predicted labels for target nodes
        train_class_features: Per-class mean embeddings from training
        full_edge_count: Total edge count for ratio calculation
        step: Current permutation step
        mask_type: 'val' or 'test'

    Returns:
        Dictionary of extracted features
    """
    # Create subgraph mask
    node_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    node_mask[subgraph_nodes] = True
    edge_mask = node_mask[data.edge_index[0]] & node_mask[data.edge_index[1]]
    subgraph_edge_index = data.edge_index[:, edge_mask]

    # Get target mask
    target_mask = data.val_mask if mask_type == 'val' else data.test_mask

    model.eval()
    conv = SGConvNoWeight(K=2).to(device)

    with torch.no_grad():
        # Create subgraph data
        subgraph_data = data.clone()
        subgraph_data.edge_index = subgraph_edge_index

        # Get model predictions
        out, _ = model(subgraph_data)
        probs = out.exp()
        _, pred = out.max(dim=1)

        target_probs = probs[target_mask]
        target_pred = pred[target_mask]
        target_labels = data.y[target_mask]

        # Accuracy (ground truth utility)
        accuracy = (target_pred == target_labels).float().mean().item()

        # Negative entropy (model uncertainty)
        negative_entropy = -torch.sum(target_probs * torch.log(target_probs + 1e-8), dim=1).mean().item()

        # Embedding-based features
        target_embeddings = conv(data.x, subgraph_edge_index)[target_mask]
        target_embeddings = F.normalize(target_embeddings, p=2, dim=1)

        # Cosine similarity to training prototype
        cosine_similarities = F.cosine_similarity(target_embeddings, train_prototype.unsqueeze(0))

        # Confidence features
        max_probs, _ = target_probs.max(dim=1)
        predicted_confidences = target_probs[torch.arange(len(target_predictions)), target_predictions]

        # Structural features
        current_edge_count = subgraph_edge_index.shape[1]
        edge_ratio = current_edge_count / full_edge_count if full_edge_count > 0 else 0

        # Edge cosine similarity (chunked to avoid large GPU allocations)
        avg_edge_cosine = _mean_edge_cosine_similarity(data.x, subgraph_edge_index)

        # Class-wise cosine similarity
        class_cosine_sims = F.cosine_similarity(
            target_embeddings.unsqueeze(1),
            train_class_features.unsqueeze(0)
        )
        avg_max_class_cosine = class_cosine_sims.max(dim=1)[0].mean().item()

        # Propagated predictions (feature-only baseline)
        feature_data = data.clone()
        feature_data.edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        feature_out, _ = model(feature_data)
        feature_probs = feature_out.exp()

        propagated_probs = conv(feature_probs, subgraph_edge_index)
        propagated_probs = F.normalize(propagated_probs, p=1, dim=1)

        propagated_max_probs = propagated_probs.max(dim=1)[0][target_mask]
        propagated_target_probs = propagated_probs[target_mask, target_predictions]

        # Confidence gap (top-2 probability difference)
        sorted_probs, _ = target_probs.sort(descending=True)
        max_conf_gap = (sorted_probs[:, 0] - sorted_probs[:, 1]).abs().mean().item()

    return {
        'val_acc' if mask_type == 'val' else 'test_acc': accuracy,
        'negative_entropy': negative_entropy,
        'cosine_similarities': cosine_similarities.cpu().numpy(),
        'max_probs': max_probs.cpu().numpy(),
        'predicted_class_confidences': predicted_confidences.cpu().numpy(),
        'step': step,
        'added_node_count': len(subgraph_nodes),
        'current_edge_count': current_edge_count,
        'edge_ratio': edge_ratio,
        'avg_edge_cosine_similarity': avg_edge_cosine,
        'avg_max_class_cosine_similarity': avg_max_class_cosine,
        'propagated_max_probs': propagated_max_probs.cpu().numpy(),
        'propagated_target_probs': propagated_target_probs.cpu().numpy(),
        'max_conf_gap': max_conf_gap,
        'val_probs' if mask_type == 'val' else 'test_probs': target_probs.cpu().numpy(),
        'val_pred' if mask_type == 'val' else 'test_pred': target_pred.cpu().numpy(),
        'val_labels' if mask_type == 'val' else 'test_labels': target_labels.cpu().numpy(),
    }


def compute_train_features(model, data, split_data: dict, device: str, mask_type: str = 'val') -> tuple:
    """
    Compute training prototype and class-wise features.

    Args:
        model: Trained GNN model
        data: PyG Data object
        split_data: Preprocessed split data
        device: Computation device

    Returns:
        Tuple of (train_prototype, train_class_features, full_edge_count)
    """
    conv = SGConvNoWeight(K=2).to(device)

    train_edge_index = torch.tensor(split_data['train_edge_index'], dtype=torch.long, device=device)

    with torch.no_grad():
        train_embeddings = conv(data.x, train_edge_index)[data.train_mask]
        train_prototype = F.normalize(train_embeddings.mean(dim=0), p=2, dim=0)

        # Compute per-class features
        train_labels = data.y[data.train_mask]
        num_classes = data.y.max().item() + 1

        class_features = []
        for c in range(num_classes):
            class_mask = train_labels == c
            if class_mask.sum() > 0:
                class_emb = train_embeddings[class_mask].mean(dim=0)
                class_features.append(class_emb)
            else:
                # Use zeros for missing classes
                class_features.append(torch.zeros_like(train_prototype))

        train_class_features = torch.stack(class_features)

    edge_key = 'val_edge_index' if mask_type == 'val' else 'test_edge_index'
    if edge_key in split_data and hasattr(split_data[edge_key], "shape"):
        full_edge_count = int(split_data[edge_key].shape[1])
    else:
        full_edge_count = int(data.edge_index.shape[1])

    return train_prototype, train_class_features, full_edge_count


def evaluate_subgraph(model, data, subgraph_nodes: list, device: str,
                      train_prototype: torch.Tensor, target_predictions: torch.Tensor,
                      neighbor_counts: dict, step: int, full_edge_count: int,
                      train_class_features: torch.Tensor, original_pred: torch.Tensor,
                      mask_type: str = 'val') -> dict:
    """
    Evaluate a subgraph and extract all features.

    This is the main function called during permutation sampling.
    """
    features = extract_features(
        model, data, subgraph_nodes, device,
        train_prototype, target_predictions, train_class_features,
        full_edge_count, step, mask_type
    )

    features['neighbor_counts'] = neighbor_counts

    return features
