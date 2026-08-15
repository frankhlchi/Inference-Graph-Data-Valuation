from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

import svgl
from svgl.data import datasets as datasets_module
from svgl.data import preprocess as preprocess_module
from svgl.data.datasets import get_dataset_info
from svgl.data.preprocess import load_preprocessed_data, preprocess_data


class _Dataset:
    def __init__(self, data):
        self.data = data
        self.num_classes = int(data.y.max().item()) + 1

    def __getitem__(self, index):
        assert index == 0
        return self.data


def _ring_graph(num_nodes=30, planetoid=True):
    source = torch.arange(num_nodes, dtype=torch.long)
    target = (source + 1) % num_nodes
    edge_index = torch.cat(
        (
            torch.stack((source, target)),
            torch.stack((target, source)),
        ),
        dim=1,
    )
    data = Data(
        x=torch.eye(num_nodes),
        y=torch.arange(num_nodes, dtype=torch.long) % 3,
        edge_index=edge_index,
    )
    if planetoid:
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.train_mask[:3] = True
        data.val_mask[3:6] = True
        data.test_mask[6:9] = True
    return data


def _patch_dataset(monkeypatch, data):
    monkeypatch.setattr(
        preprocess_module, "load_dataset", lambda dataset_name, root: _Dataset(data)
    )


def _edge_pairs(edge_index):
    return set(map(tuple, np.asarray(edge_index).T.tolist()))


def test_public_import_surface():
    assert callable(svgl.load_dataset)
    assert callable(svgl.preprocess_data)
    assert callable(svgl.load_preprocessed_data)
    assert callable(svgl.create_model)


def test_dataset_registry_and_loader_routing(monkeypatch, tmp_path):
    calls = {}

    def fake_planetoid(**kwargs):
        calls.update(kwargs)
        return object()

    monkeypatch.setattr(datasets_module, "Planetoid", fake_planetoid)
    result = datasets_module.load_dataset("cora", root=str(tmp_path))

    assert result is not None
    assert calls["name"] == "Cora"
    assert Path(calls["root"]) == tmp_path
    assert get_dataset_info("roman_empire")["name"] == "Roman-empire"
    with pytest.raises(ValueError, match="Unsupported dataset"):
        datasets_module.load_dataset("not-a-dataset", root=str(tmp_path))


@pytest.mark.parametrize("use_pmlp", [True, False])
def test_inductive_planetoid_split(monkeypatch, tmp_path, use_pmlp):
    data = _ring_graph()
    _patch_dataset(monkeypatch, data)

    split = preprocess_data(
        "Cora",
        cache_dir=tmp_path,
        data_seed=7,
        setting="inductive",
        use_pmlp=use_pmlp,
    )

    required = {
        "train_indices",
        "train_edge_index",
        "val_indices",
        "val_edge_index",
        "test_indices",
        "test_edge_index",
    }
    assert required.issubset(split)
    for key in ("train_edge_index", "val_edge_index", "test_edge_index"):
        assert split[key].shape[0] == 2

    train = set(split["train_indices"])
    val = set(split["val_indices"])
    test = set(split["test_indices"])
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert set(split["val_all_nodes"]).isdisjoint(split["test_all_nodes"])

    if use_pmlp:
        assert split["train_edge_index"].shape == (2, 0)
    else:
        assert _edge_pairs(split["train_edge_index"])
        assert all(u in train and v in train for u, v in _edge_pairs(split["train_edge_index"]))


def test_transductive_planetoid_and_legacy_keyword(monkeypatch, tmp_path):
    data = _ring_graph()
    _patch_dataset(monkeypatch, data)

    split = preprocess_data(
        "Cora",
        cache_dir=tmp_path,
        setting="transductive",
        pmlp=False,
    )

    expected_edges = _edge_pairs(data.edge_index.numpy())
    assert split["train_indices"] == [0, 1, 2]
    assert split["val_indices"] == [3, 4, 5]
    assert split["test_indices"] == [6, 7, 8]
    assert _edge_pairs(split["train_edge_index"]) == expected_edges
    assert _edge_pairs(split["val_edge_index"]) == expected_edges
    assert _edge_pairs(split["test_edge_index"]) == expected_edges


def test_non_planetoid_paths_are_implemented(monkeypatch, tmp_path):
    data = _ring_graph(num_nodes=200, planetoid=False)
    _patch_dataset(monkeypatch, data)

    inductive = preprocess_data(
        "CS",
        cache_dir=tmp_path / "inductive",
        data_seed=11,
        setting="inductive",
        use_pmlp=True,
    )
    transductive = preprocess_data(
        "CS",
        cache_dir=tmp_path / "transductive",
        data_seed=11,
        setting="transductive",
        use_pmlp=True,
    )

    assert len(inductive["train_indices"]) == 2
    assert len(inductive["val_indices"]) == 6
    assert len(inductive["test_indices"]) == 6
    assert inductive["train_edge_index"].shape == (2, 0)
    assert transductive["train_edge_index"].shape == (2, 0)
    assert _edge_pairs(transductive["val_edge_index"]) == _edge_pairs(
        data.edge_index.numpy()
    )


def test_seeded_cache_round_trip_and_determinism(monkeypatch, tmp_path):
    data = _ring_graph()
    _patch_dataset(monkeypatch, data)

    split_1 = preprocess_data("Cora", cache_dir=tmp_path, data_seed=1)
    split_1_again = preprocess_data("Cora", cache_dir=tmp_path, data_seed=1)
    split_2 = preprocess_data("Cora", cache_dir=tmp_path, data_seed=2)

    assert split_1["val_indices"] == split_1_again["val_indices"]
    assert split_1["test_indices"] == split_1_again["test_indices"]
    assert (
        split_1["val_indices"] != split_2["val_indices"]
        or split_1["test_indices"] != split_2["test_indices"]
    )

    cache_names = {path.name for path in tmp_path.glob("*.pkl")}
    assert "Cora_default_seed1_inductive_pmlp_split.pkl" in cache_names
    assert "Cora_default_seed2_inductive_pmlp_split.pkl" in cache_names

    loaded = load_preprocessed_data("Cora", cache_dir=tmp_path, data_seed=1)
    assert loaded["val_indices"] == split_1["val_indices"]


def test_legacy_cache_seed_must_match(tmp_path):
    legacy_path = tmp_path / "Cora_default_inductive_pmlp_split.pkl"
    legacy_split = {"metadata": {"data_seed": 1}, "val_indices": [1]}
    with legacy_path.open("wb") as handle:
        import pickle

        pickle.dump(legacy_split, handle)

    assert load_preprocessed_data(
        "Cora", cache_dir=tmp_path, data_seed=1
    )["val_indices"] == [1]
    with pytest.raises(FileNotFoundError, match="does not match"):
        load_preprocessed_data("Cora", cache_dir=tmp_path, data_seed=999)


def test_sample_nodes_validates_ratios():
    data = _ring_graph(num_nodes=100, planetoid=False)
    with pytest.raises(ValueError, match="between 0 and 1"):
        preprocess_module.sample_nodes(data, train_ratio=-0.1)
    with pytest.raises(ValueError, match="must be <= 1"):
        preprocess_module.sample_nodes(
            data, train_ratio=0.6, val_ratio=0.3, test_ratio=0.3
        )


def test_ogb_transform_flattens_labels():
    data = Data(x=torch.ones((3, 2)), y=torch.tensor([[0], [1], [2]]))
    transformed = datasets_module._normalize_ogb_data(data)
    assert transformed.y.shape == (3,)


def test_invalid_preprocessing_options_are_rejected(monkeypatch, tmp_path):
    _patch_dataset(monkeypatch, _ring_graph())
    with pytest.raises(ValueError, match="setting"):
        preprocess_data("Cora", cache_dir=tmp_path, setting="other")
    with pytest.raises(ValueError, match="sampling_method"):
        preprocess_data("Cora", cache_dir=tmp_path, sampling_method="other")
