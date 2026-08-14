"""Graph split generation and cache helpers for SVGL.

This module restores the preprocessing implementation that was present before
the repository was reorganized into the :mod:`svgl` package.  It preserves the
paper's inductive/transductive split protocol while exposing the keyword names
used by the current scripts.
"""

from __future__ import annotations

import os
import pickle
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch_geometric.utils import k_hop_subgraph, subgraph

from .datasets import canonicalize_dataset_name, get_dataset_info, load_dataset


PLANETOID_DATASETS = frozenset({"Cora", "Citeseer", "Pubmed"})

_INDUCTIVE_RATIOS = {
    "Cora": (None, 0.10, 0.10),
    "Citeseer": (None, 0.10, 0.10),
    "Pubmed": (None, 0.05, 0.05),
    "Physics": (0.01, 0.01, 0.01),
    "CoraFull": (0.10, 0.01, 0.01),
    "chameleon": (0.05, 0.05, 0.05),
    "squirrel": (0.05, 0.05, 0.05),
    "crocodile": (0.05, 0.05, 0.05),
    "actor": (0.05, 0.05, 0.05),
    "Roman-empire": (0.05, 0.05, 0.05),
    "Amazon-ratings": (0.05, 0.05, 0.05),
    "Minesweeper": (0.05, 0.05, 0.05),
    "Tolokers": (0.05, 0.05, 0.05),
    "Questions": (0.05, 0.05, 0.05),
}

_DEFAULT_INDUCTIVE_RATIOS = (0.01, 0.03, 0.03)
_DEFAULT_TRANSDUCTIVE_RATIOS = (0.01, 0.01, 0.01)


def _resolve_use_pmlp(use_pmlp: bool, pmlp: Optional[bool]) -> bool:
    """Resolve the current ``use_pmlp`` name and historical ``pmlp`` alias."""

    return bool(use_pmlp if pmlp is None else pmlp)


def _seed_everything(seed: Optional[int]) -> np.random.RandomState:
    if seed is None:
        return np.random.RandomState()

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return np.random.RandomState(seed)


def _as_index_list(indices) -> List[int]:
    if isinstance(indices, torch.Tensor):
        return [int(value) for value in indices.detach().cpu().reshape(-1).tolist()]
    if isinstance(indices, np.ndarray):
        return [int(value) for value in indices.reshape(-1).tolist()]
    return [int(value) for value in indices]


def _indices_from_mask(mask: torch.Tensor) -> List[int]:
    if mask is None:
        raise ValueError("Dataset does not provide the required split mask")
    if mask.ndim != 1:
        # Some PyG datasets expose multiple predefined splits as columns.  The
        # paper's preprocessing uses a single split, so select the first one.
        mask = mask[:, 0]
    return _as_index_list(mask.nonzero(as_tuple=True)[0])


def _empty_edge_index() -> np.ndarray:
    return np.empty((2, 0), dtype=np.int64)


def _edge_index_to_numpy(edge_index: torch.Tensor) -> np.ndarray:
    return edge_index.detach().cpu().numpy().astype(np.int64, copy=False)


def _induced_edges(data, nodes: Sequence[int]) -> np.ndarray:
    if not nodes:
        return _empty_edge_index()
    node_tensor = torch.as_tensor(
        nodes, dtype=torch.long, device=data.edge_index.device
    )
    edge_index, _ = subgraph(
        node_tensor,
        data.edge_index,
        relabel_nodes=False,
        num_nodes=data.num_nodes,
    )
    return _edge_index_to_numpy(edge_index)


def _context_edges(
    data,
    target_nodes: Sequence[int],
    allowed_nodes: Sequence[int],
    num_hops: int = 2,
) -> np.ndarray:
    """Return target-local edges whose endpoints stay inside one context."""

    if not target_nodes or not allowed_nodes:
        return _empty_edge_index()

    device = data.edge_index.device
    target_tensor = torch.as_tensor(target_nodes, dtype=torch.long, device=device)
    _, edge_index, _, _ = k_hop_subgraph(
        node_idx=target_tensor,
        num_hops=num_hops,
        edge_index=data.edge_index,
        relabel_nodes=False,
        num_nodes=data.num_nodes,
    )

    allowed_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    allowed_mask[
        torch.as_tensor(allowed_nodes, dtype=torch.long, device=device)
    ] = True
    edge_mask = allowed_mask[edge_index[0]] & allowed_mask[edge_index[1]]
    return _edge_index_to_numpy(edge_index[:, edge_mask])


def get_candidate_nodes(
    data,
    active_nodes: Iterable[int],
    hop: int,
    exclude_nodes: Iterable[int],
    device=None,
) -> set:
    """Return nodes within ``hop`` steps, excluding the supplied node set.

    ``device`` is retained for compatibility with the historical helper.
    """

    del device
    candidates = set()
    for node in active_nodes:
        nodes, _, _, _ = k_hop_subgraph(
            int(node),
            hop,
            data.edge_index,
            relabel_nodes=False,
            num_nodes=data.num_nodes,
        )
        candidates.update(_as_index_list(nodes))
    return candidates.difference(int(node) for node in exclude_nodes)


def sample_nodes(
    data,
    train_ratio: float = 0.01,
    val_ratio: float = 0.03,
    test_ratio: float = 0.03,
    data_seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample disjoint target-node splits with the historical ratio protocol."""

    ratios = (train_ratio, val_ratio, test_ratio)
    if any(ratio < 0 or ratio > 1 for ratio in ratios):
        raise ValueError("split ratios must each be between 0 and 1")
    if sum(ratios) > 1:
        raise ValueError("train_ratio + val_ratio + test_ratio must be <= 1")

    rng = np.random.RandomState(data_seed)
    permutation = rng.permutation(int(data.num_nodes))
    train_size = int(data.num_nodes * train_ratio)
    val_size = int(data.num_nodes * val_ratio)
    test_size = int(data.num_nodes * test_ratio)
    if min(train_size, val_size, test_size) < 1:
        raise ValueError(
            "Dataset is too small for the requested split ratios: "
            f"nodes={data.num_nodes}, ratios="
            f"({train_ratio}, {val_ratio}, {test_ratio})"
        )

    return (
        permutation[:train_size],
        permutation[train_size : train_size + val_size],
        permutation[
            train_size + val_size : train_size + val_size + test_size
        ],
    )


def _partition_context_nodes(
    num_nodes: int,
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    test_indices: Sequence[int],
    val_ratio: float,
    test_ratio: float,
    rng: np.random.RandomState,
) -> Tuple[List[int], List[int]]:
    used = set(train_indices) | set(val_indices) | set(test_indices)
    remaining = np.asarray(sorted(set(range(num_nodes)) - used), dtype=np.int64)
    rng.shuffle(remaining)

    denominator = val_ratio + test_ratio
    if denominator <= 0:
        raise ValueError("val_ratio and test_ratio cannot both be zero")
    val_extra_count = int(len(remaining) * val_ratio / denominator)

    val_all_nodes = list(val_indices) + _as_index_list(remaining[:val_extra_count])
    test_all_nodes = list(test_indices) + _as_index_list(remaining[val_extra_count:])
    return val_all_nodes, test_all_nodes


def _process_inductive_planetoid(
    data,
    dataset_name: str,
    data_seed: int,
    use_pmlp: bool,
) -> Dict[str, object]:
    rng = np.random.RandomState(data_seed)
    num_nodes = int(data.num_nodes)
    _, val_ratio, test_ratio = _INDUCTIVE_RATIOS[dataset_name]

    train_indices = _indices_from_mask(data.train_mask)
    available = np.asarray(
        sorted(set(range(num_nodes)) - set(train_indices)), dtype=np.int64
    )
    rng.shuffle(available)

    num_val = int(num_nodes * val_ratio)
    num_test = int(num_nodes * test_ratio)
    if num_val < 1 or num_test < 1 or num_val + num_test > len(available):
        raise ValueError(
            f"Cannot create validation/test targets for {dataset_name}: "
            f"nodes={num_nodes}, available={len(available)}"
        )

    val_indices = _as_index_list(available[:num_val])
    test_indices = _as_index_list(available[num_val : num_val + num_test])
    val_all_nodes, test_all_nodes = _partition_context_nodes(
        num_nodes,
        train_indices,
        val_indices,
        test_indices,
        val_ratio,
        test_ratio,
        rng,
    )

    return {
        "train_indices": train_indices,
        "train_edge_index": (
            _empty_edge_index()
            if use_pmlp
            else _induced_edges(data, train_indices)
        ),
        "val_indices": val_indices,
        "val_edge_index": _context_edges(data, val_indices, val_all_nodes),
        "test_indices": test_indices,
        "test_edge_index": _context_edges(data, test_indices, test_all_nodes),
        "val_all_nodes": val_all_nodes,
        "test_all_nodes": test_all_nodes,
    }


def _process_inductive_non_planetoid(
    data,
    dataset_name: str,
    data_seed: int,
    use_pmlp: bool,
) -> Dict[str, object]:
    rng = np.random.RandomState(data_seed)
    train_ratio, val_ratio, test_ratio = _INDUCTIVE_RATIOS.get(
        dataset_name, _DEFAULT_INDUCTIVE_RATIOS
    )
    train_raw, val_raw, test_raw = sample_nodes(
        data,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        data_seed=data_seed,
    )
    train_indices = _as_index_list(train_raw)
    val_indices = _as_index_list(val_raw)
    test_indices = _as_index_list(test_raw)
    val_all_nodes, test_all_nodes = _partition_context_nodes(
        int(data.num_nodes),
        train_indices,
        val_indices,
        test_indices,
        val_ratio,
        test_ratio,
        rng,
    )

    return {
        "train_indices": train_indices,
        "train_edge_index": (
            _empty_edge_index()
            if use_pmlp
            else _induced_edges(data, train_indices)
        ),
        "val_indices": val_indices,
        "val_edge_index": _context_edges(data, val_indices, val_all_nodes),
        "test_indices": test_indices,
        "test_edge_index": _context_edges(data, test_indices, test_all_nodes),
        "val_all_nodes": val_all_nodes,
        "test_all_nodes": test_all_nodes,
    }


def _process_transductive_planetoid(data, use_pmlp: bool) -> Dict[str, object]:
    all_edges = _edge_index_to_numpy(data.edge_index)
    return {
        "train_indices": _indices_from_mask(data.train_mask),
        "train_edge_index": _empty_edge_index() if use_pmlp else all_edges,
        "val_indices": _indices_from_mask(data.val_mask),
        "val_edge_index": all_edges,
        "test_indices": _indices_from_mask(data.test_mask),
        "test_edge_index": all_edges,
    }


def _process_transductive_non_planetoid(
    data,
    dataset_name: str,
    data_seed: int,
    use_pmlp: bool,
) -> Dict[str, object]:
    ratios = _DEFAULT_TRANSDUCTIVE_RATIOS
    if dataset_name == "Physics":
        ratios = (0.001, 0.001, 0.001)
    train_raw, val_raw, test_raw = sample_nodes(
        data,
        train_ratio=ratios[0],
        val_ratio=ratios[1],
        test_ratio=ratios[2],
        data_seed=data_seed,
    )
    all_edges = _edge_index_to_numpy(data.edge_index)
    return {
        "train_indices": _as_index_list(train_raw),
        "train_edge_index": _empty_edge_index() if use_pmlp else all_edges,
        "val_indices": _as_index_list(val_raw),
        "val_edge_index": all_edges,
        "test_indices": _as_index_list(test_raw),
        "test_edge_index": all_edges,
    }


def _cache_file_name(
    dataset_name: str,
    sampling_method: str,
    data_seed: Optional[int],
    setting: str,
    use_pmlp: bool,
    include_seed: bool = True,
) -> str:
    seed_part = (
        f"_seed{int(data_seed)}"
        if include_seed and data_seed is not None
        else ""
    )
    mode = "pmlp" if use_pmlp else "gnn"
    return f"{dataset_name}_{sampling_method}{seed_part}_{setting}_{mode}_split.pkl"


def _cache_path(
    dataset_name: str,
    cache_dir: os.PathLike,
    sampling_method: str,
    data_seed: Optional[int],
    setting: str,
    use_pmlp: bool,
    include_seed: bool = True,
) -> Path:
    return Path(cache_dir) / _cache_file_name(
        dataset_name,
        sampling_method,
        data_seed,
        setting,
        use_pmlp,
        include_seed=include_seed,
    )


def save_preprocessed_data(
    split_data: Dict[str, object],
    dataset_name: str,
    cache_dir: os.PathLike = "./graph_split",
    sampling_method: str = "default",
    seed: Optional[int] = None,
    data_seed: Optional[int] = 0,
    setting: str = "inductive",
    use_pmlp: bool = True,
    *,
    pmlp: Optional[bool] = None,
) -> Path:
    """Persist a generated split and return the cache path."""

    del seed  # Model initialization seed does not affect graph splits.
    name = canonicalize_dataset_name(dataset_name)
    resolved_pmlp = _resolve_use_pmlp(use_pmlp, pmlp)
    path = _cache_path(
        name,
        cache_dir,
        sampling_method,
        data_seed,
        setting,
        resolved_pmlp,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(split_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_preprocessed_data(
    dataset_name: str,
    cache_dir: os.PathLike = "./graph_split",
    sampling_method: str = "default",
    seed: Optional[int] = None,
    data_seed: Optional[int] = 0,
    setting: str = "inductive",
    use_pmlp: bool = True,
    *,
    pmlp: Optional[bool] = None,
) -> Dict[str, object]:
    """Load a cached split.

    Seeded cache names prevent runs with different data seeds from silently
    sharing a split.  The unseeded name used by the original script remains a
    read-only fallback for existing experiment directories.
    """

    del seed
    name = canonicalize_dataset_name(dataset_name)
    resolved_pmlp = _resolve_use_pmlp(use_pmlp, pmlp)
    current_path = _cache_path(
        name,
        cache_dir,
        sampling_method,
        data_seed,
        setting,
        resolved_pmlp,
    )
    legacy_path = _cache_path(
        name,
        cache_dir,
        sampling_method,
        data_seed,
        setting,
        resolved_pmlp,
        include_seed=False,
    )

    legacy_seed_rejected = False
    legacy_cached_seed = None
    for path in dict.fromkeys((current_path, legacy_path)):
        if path.exists():
            with path.open("rb") as handle:
                split_data = pickle.load(handle)
            if path == legacy_path and data_seed is not None:
                metadata = split_data.get("metadata", {})
                cached_seed = metadata.get("data_seed")
                if cached_seed is None or int(cached_seed) != int(data_seed):
                    legacy_seed_rejected = True
                    legacy_cached_seed = cached_seed
                    continue
            return split_data

    message = (
        "Preprocessed data file not found. Checked: "
        + ", ".join(str(path) for path in (current_path, legacy_path))
    )
    if legacy_seed_rejected:
        message += (
            f". Legacy cache seed {legacy_cached_seed} does not match "
            f"requested seed {data_seed}"
        )
    raise FileNotFoundError(message)


def preprocess_data(
    dataset_name: str,
    root: str = "./data",
    device: str = "cpu",
    cache_dir: os.PathLike = "./graph_split",
    sampling_method: str = "default",
    seed: int = 0,
    data_seed: int = 0,
    setting: str = "inductive",
    use_pmlp: bool = True,
    *,
    pmlp: Optional[bool] = None,
) -> Dict[str, object]:
    """Load a dataset, create paper-compatible splits, and cache them.

    Args:
        dataset_name: Name accepted by :func:`svgl.data.load_dataset`.
        root: PyG dataset download/cache directory.
        device: Device used for graph filtering.
        cache_dir: Directory in which to store the split pickle.
        sampling_method: Retained in cache metadata; ``default`` and ``random``
            both use seeded random sampling for datasets without fixed masks.
        seed: Model seed retained in metadata for historical compatibility.
        data_seed: Random seed controlling target and context splits.
        setting: ``inductive`` or ``transductive``.
        use_pmlp: If true, the training graph has no edges.
        pmlp: Historical keyword alias for ``use_pmlp``.
    """

    if setting not in {"inductive", "transductive"}:
        raise ValueError("setting must be 'inductive' or 'transductive'")
    if sampling_method not in {"default", "random"}:
        raise ValueError("sampling_method must be 'default' or 'random'")

    name = canonicalize_dataset_name(dataset_name)
    resolved_pmlp = _resolve_use_pmlp(use_pmlp, pmlp)
    _seed_everything(data_seed)

    dataset = load_dataset(name, root=root)
    data = dataset[0].to(device)
    category = get_dataset_info(name)["category"]

    if setting == "inductive":
        if category == "planetoid":
            split_data = _process_inductive_planetoid(
                data, name, data_seed, resolved_pmlp
            )
        else:
            split_data = _process_inductive_non_planetoid(
                data, name, data_seed, resolved_pmlp
            )
    elif category == "planetoid":
        split_data = _process_transductive_planetoid(data, resolved_pmlp)
    else:
        split_data = _process_transductive_non_planetoid(
            data, name, data_seed, resolved_pmlp
        )

    split_data["metadata"] = {
        "dataset": name,
        "setting": setting,
        "use_pmlp": resolved_pmlp,
        "pmlp": resolved_pmlp,
        "sampling_method": sampling_method,
        "seed": seed,
        "data_seed": data_seed,
    }
    save_preprocessed_data(
        split_data,
        name,
        cache_dir=cache_dir,
        sampling_method=sampling_method,
        seed=seed,
        data_seed=data_seed,
        setting=setting,
        use_pmlp=resolved_pmlp,
    )
    return split_data


__all__ = [
    "PLANETOID_DATASETS",
    "get_candidate_nodes",
    "load_preprocessed_data",
    "preprocess_data",
    "sample_nodes",
    "save_preprocessed_data",
]
