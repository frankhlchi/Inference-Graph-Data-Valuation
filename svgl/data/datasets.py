"""Dataset loading helpers for SVGL experiments.

The project originally kept these helpers in a top-level ``preprocess.py``
script.  Keeping dataset construction separate from split generation makes the
package API usable by both the command-line scripts and external callers.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import torch_geometric.transforms as T
from torch_geometric.datasets import (
    Actor,
    Amazon,
    Coauthor,
    CoraFull,
    DBLP,
    HeterophilousGraphDataset,
    Planetoid,
    WikiCS,
    WikipediaNetwork,
)


_DATASET_INFO: Dict[str, Dict[str, Any]] = {
    "Cora": {"family": "Planetoid", "category": "planetoid"},
    "Citeseer": {"family": "Planetoid", "category": "planetoid"},
    "Pubmed": {"family": "Planetoid", "category": "planetoid"},
    "CoraFull": {"family": "CoraFull", "category": "citation"},
    "CS": {"family": "Coauthor", "category": "coauthor"},
    "Physics": {"family": "Coauthor", "category": "coauthor"},
    "Computers": {"family": "Amazon", "category": "amazon"},
    "Photo": {"family": "Amazon", "category": "amazon"},
    "DBLP": {"family": "DBLP", "category": "citation"},
    "WikiCS": {"family": "WikiCS", "category": "wikipedia"},
    "chameleon": {
        "family": "WikipediaNetwork",
        "category": "heterophilous",
    },
    "squirrel": {
        "family": "WikipediaNetwork",
        "category": "heterophilous",
    },
    "crocodile": {
        "family": "WikipediaNetwork",
        "category": "heterophilous",
    },
    "actor": {"family": "Actor", "category": "heterophilous"},
    "Roman-empire": {
        "family": "HeterophilousGraphDataset",
        "category": "heterophilous",
    },
    "Amazon-ratings": {
        "family": "HeterophilousGraphDataset",
        "category": "heterophilous",
    },
    "Minesweeper": {
        "family": "HeterophilousGraphDataset",
        "category": "heterophilous",
    },
    "Tolokers": {
        "family": "HeterophilousGraphDataset",
        "category": "heterophilous",
    },
    "Questions": {
        "family": "HeterophilousGraphDataset",
        "category": "heterophilous",
    },
    "ogbn-arxiv": {"family": "OGB", "category": "ogb"},
}

_ALIASES = {
    name.lower().replace("_", "-"): name for name in _DATASET_INFO
}


def _normalize_ogb_data(data):
    """Normalize OGB features and expose class labels as a one-dimensional tensor."""

    data = T.NormalizeFeatures()(data)
    if getattr(data, "y", None) is not None:
        data.y = data.y.reshape(-1)
    return data


def canonicalize_dataset_name(dataset_name: str) -> str:
    """Return the spelling expected by the underlying dataset provider."""

    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise ValueError("dataset_name must be a non-empty string")

    key = dataset_name.strip().lower().replace("_", "-")
    try:
        return _ALIASES[key]
    except KeyError as exc:
        supported = ", ".join(_DATASET_INFO)
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. Supported datasets: {supported}"
        ) from exc


def get_dataset_info(dataset_name: Optional[str] = None) -> Dict[str, Any]:
    """Return metadata for one dataset, or all supported datasets.

    A copy is returned so callers can safely add run-specific metadata without
    mutating the package registry.
    """

    if dataset_name is None:
        return deepcopy(_DATASET_INFO)

    canonical_name = canonicalize_dataset_name(dataset_name)
    info = deepcopy(_DATASET_INFO[canonical_name])
    info["name"] = canonical_name
    return info


def load_dataset(dataset_name: str, root: str = "./data"):
    """Load and normalize a supported graph dataset.

    Dataset files are downloaded by PyTorch Geometric on first use.  OGB is an
    optional dependency and is imported only when ``ogbn-arxiv`` is requested.
    """

    name = canonicalize_dataset_name(dataset_name)
    family = _DATASET_INFO[name]["family"]
    transform = T.NormalizeFeatures()
    # DBLP, WikiCS, and Actor use generic ``raw``/``processed`` directory names;
    # isolate only those loaders.  Other PyG loaders already append their own
    # dataset name, so passing ``root / name`` would create paths such as
    # ``data/Cora/Cora`` and bypass existing caches.
    if family in {"DBLP", "WikiCS", "Actor"}:
        dataset_root = str(Path(root) / name)
    else:
        dataset_root = str(root)

    if family == "Planetoid":
        return Planetoid(root=dataset_root, name=name, transform=transform)
    if family == "CoraFull":
        return CoraFull(root=dataset_root, transform=transform)
    if family == "Coauthor":
        return Coauthor(root=dataset_root, name=name, transform=transform)
    if family == "Amazon":
        return Amazon(root=dataset_root, name=name, transform=transform)
    if family == "DBLP":
        return DBLP(root=dataset_root, transform=transform)
    if family == "WikiCS":
        return WikiCS(root=dataset_root, transform=transform)
    if family == "WikipediaNetwork":
        kwargs: Dict[str, Any] = {"root": dataset_root, "name": name}
        if name == "crocodile":
            kwargs["geom_gcn_preprocess"] = False
        return WikipediaNetwork(**kwargs)
    if family == "Actor":
        return Actor(root=dataset_root)
    if family == "HeterophilousGraphDataset":
        return HeterophilousGraphDataset(root=dataset_root, name=name)
    if family == "OGB":
        try:
            from ogb.nodeproppred import PygNodePropPredDataset
        except ImportError as exc:
            raise ImportError(
                "Loading ogbn-arxiv requires the optional `ogb` package. "
                "Install it with `pip install ogb`."
            ) from exc
        return PygNodePropPredDataset(
            name=name, root=dataset_root, transform=_normalize_ogb_data
        )

    # ``family`` comes from the private registry above, so reaching this branch
    # would indicate a package bug rather than unsupported user input.
    raise RuntimeError(f"No loader is registered for dataset family: {family}")


__all__ = [
    "canonicalize_dataset_name",
    "get_dataset_info",
    "load_dataset",
]
