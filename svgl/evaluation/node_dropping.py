"""
Node-dropping evaluation and AUC computation.

Implements Algorithm "Node Dropping Evaluation" from the SVGL paper:
rank nodes by estimated Structure-aware Shapley values, then iteratively remove
top-k nodes from the test graph and measure the target-node accuracy curve.
"""

from __future__ import annotations

import math
import os
import pickle
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from svgl.valuation.shapley import DEFAULT_METHODS


BASE_FEATURES: List[str] = DEFAULT_METHODS[:-1]  # exclude `true_acc`


def build_interaction_methods(base_methods: Optional[List[str]] = None) -> List[str]:
    base_methods = base_methods or BASE_FEATURES
    interactions: List[str] = []
    for i, m1 in enumerate(base_methods):
        for m2 in base_methods[i:]:
            interactions.append(f"interaction({m1},{m2})")
    return interactions


def parse_selected_features(selected_features) -> Dict[str, float]:
    """
    Parse `selected_features` from `result.json`.

    Expected formats:
      - [["feature_name", 0.12], ...]  (JSON-serialized tuples)
      - [("feature_name", 0.12), ...]
    """
    weights: Dict[str, float] = {}
    if not selected_features:
        return weights

    for item in selected_features:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            name, weight = item
        elif isinstance(item, dict) and "name" in item:
            name = item["name"]
            weight = item.get("weight")
        else:
            continue
        if name is None or weight is None:
            continue
        try:
            weights[str(name)] = float(weight)
        except (TypeError, ValueError):
            continue

    return weights


def _mean_float(value) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (float, int, np.floating, np.integer)):
        out = float(value)
        return out if math.isfinite(out) else 0.0
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0.0
        out = float(value.float().mean().item())
        return out if math.isfinite(out) else 0.0
    try:
        arr = np.asarray(value)
        if arr.size == 0:
            return 0.0
        out = float(np.nanmean(arr))
        return out if math.isfinite(out) else 0.0
    except Exception:
        return 0.0


def extract_base_features(step_data: dict, base_methods: Optional[List[str]] = None) -> Dict[str, float]:
    base_methods = base_methods or BASE_FEATURES
    feats: Dict[str, float] = {}

    for method in base_methods:
        if method == "negative_entropy":
            feats[method] = _mean_float(step_data.get("negative_entropy", 0.0))
        elif method == "propagated_max_probs":
            feats[method] = _mean_float(step_data.get("propagated_max_probs", 0.0))
        elif method == "propagated_target_probs":
            feats[method] = _mean_float(step_data.get("propagated_target_probs", 0.0))
        elif method == "cosine_similarity":
            feats[method] = _mean_float(step_data.get("cosine_similarities", 0.0))
        elif method == "confidence":
            feats[method] = _mean_float(step_data.get("predicted_class_confidences", 0.0))
        elif method == "argmax_confidence":
            feats[method] = _mean_float(step_data.get("max_probs", 0.0))
        else:
            feats[method] = _mean_float(step_data.get(method, 0.0))

    return feats


def feature_value(name: str, base_feats: Dict[str, float]) -> float:
    if name in base_feats:
        return base_feats[name]
    if name.startswith("interaction(") and name.endswith(")"):
        inner = name[len("interaction("):-1]
        m1, m2 = inner.split(",", 1)
        return base_feats.get(m1, 0.0) * base_feats.get(m2, 0.0)
    raise KeyError(name)


def predict_utility(step_data: dict, weights: Dict[str, float],
                    base_methods: Optional[List[str]] = None) -> float:
    """
    Predict utility via a sparse linear model (intercept cancels in margins).

    The weights are taken from `result.json`'s `selected_features`.
    """
    if not weights:
        return 0.0

    base_feats = extract_base_features(step_data, base_methods=base_methods)
    total = 0.0
    for name, w in weights.items():
        try:
            x = feature_value(name, base_feats)
        except KeyError:
            x = _mean_float(step_data.get(name, 0.0))
        total += float(w) * float(x)
    return total


def _iter_sample_files(samples_dir: str, max_samples: Optional[int] = None) -> List[str]:
    files = [
        f for f in os.listdir(samples_dir)
        if f.startswith("sample_") and f.endswith(".pkl")
    ]
    files.sort(key=lambda s: int(s[len("sample_"):-len(".pkl")]))
    if max_samples is not None:
        files = files[:max_samples]
    return files


def estimate_shapley_from_samples(
    samples_dir: str,
    weights: Dict[str, float],
    utility_key: str,
    max_samples: Optional[int] = None,
) -> Tuple[Dict[int, float], Dict[int, float], int]:
    """
    Estimate per-node Structure-aware Shapley values from permutation samples.

    Returns:
      - shapley_pred: estimated Shapley values using predicted utility (w^T x(S))
      - shapley_true: Shapley values using the ground-truth utility in `utility_key`
      - num_samples: number of permutations processed
    """
    shapley_pred: Dict[int, float] = {}
    shapley_true: Dict[int, float] = {}

    sample_files = _iter_sample_files(samples_dir, max_samples=max_samples)
    for sample_file in sample_files:
        sample_path = os.path.join(samples_dir, sample_file)
        with open(sample_path, "rb") as f:
            sample_data = pickle.load(f)

        prev_pred = None
        prev_true = None
        for step, step_data in sorted(sample_data.items(), key=lambda x: int(x[0])):
            pred_u = predict_utility(step_data, weights)
            true_u = _mean_float(step_data.get(utility_key, 0.0))

            if prev_pred is not None:
                node = step_data.get("node")
                if node is not None:
                    node = int(node)
                    shapley_pred[node] = shapley_pred.get(node, 0.0) + (pred_u - prev_pred)
                    shapley_true[node] = shapley_true.get(node, 0.0) + (true_u - prev_true)

            prev_pred = pred_u
            prev_true = true_u

    m = len(sample_files)
    if m:
        for d in (shapley_pred, shapley_true):
            for k in list(d.keys()):
                d[k] /= m
    return shapley_pred, shapley_true, m


@torch.no_grad()
def _test_accuracy(model, data) -> float:
    model.eval()
    out, _ = model(data)
    pred = out.argmax(dim=1)
    correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
    denom = int(data.test_mask.sum().item())
    return float(correct) / float(denom) if denom > 0 else float("nan")


def node_dropping_curve(
    model,
    data,
    base_edge_index: torch.Tensor,
    ranked_nodes: List[int],
    device: str,
    stride: int = 1,
) -> Tuple[List[int], List[float]]:
    """
    Compute accuracy curve for removing top-k ranked nodes.

    Returns:
      - ks: list of removed-node counts (includes 0)
      - accs: accuracy at each k (aligned with ks)
    """
    if stride < 1:
        raise ValueError("--stride must be >= 1")

    K = len(ranked_nodes)
    if K == 0:
        return [0], [_test_accuracy(model, data)]

    data = data.to(device)
    base_edge_index = base_edge_index.to(device)

    # Precompute edge-removal order: each edge disappears at min(rank[u], rank[v]).
    rank = torch.full((data.num_nodes,), K + 1, dtype=torch.long, device=device)
    ranked_t = torch.tensor(ranked_nodes, dtype=torch.long, device=device)
    rank[ranked_t] = torch.arange(1, K + 1, dtype=torch.long, device=device)

    removal_step = torch.minimum(rank[base_edge_index[0]], rank[base_edge_index[1]])
    sort_idx = torch.argsort(removal_step)
    edge_sorted = base_edge_index[:, sort_idx]
    removal_sorted = removal_step[sort_idx]

    ks = list(range(0, K + 1, stride))
    if ks[-1] != K:
        ks.append(K)

    thresholds = torch.tensor([k + 1 for k in ks], dtype=removal_sorted.dtype, device=device)
    cutoffs = torch.searchsorted(removal_sorted, thresholds)

    accs: List[float] = []
    for pos in cutoffs.tolist():
        data.edge_index = edge_sorted[:, pos:]
        accs.append(_test_accuracy(model, data))

    return ks, accs

