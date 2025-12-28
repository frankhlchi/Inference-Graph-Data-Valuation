"""
Graph Neural Network models for SVGL.

Includes SGC (Simplified Graph Convolution) and GCN (Graph Convolutional Network).
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv, SGConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.utils import spmm
from typing import Optional


class SGCNet(nn.Module):
    """Simplified Graph Convolution Network."""

    def __init__(self, num_features: int, num_classes: int, hidden_dim: int = 128, seed: int = 0):
        super(SGCNet, self).__init__()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.conv = SGConv(num_features, hidden_dim, K=2)
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_data):
        x, edge_index = input_data.x, input_data.edge_index
        hidden = self.conv(x, edge_index)
        x = self.linear(hidden)
        return F.log_softmax(x, dim=1), hidden

    def fit(self, dataset, num_epochs: int = 400, lr: float = 0.01,
            weight_decay: float = 5e-4, device: str = 'cpu'):
        model = self.to(device)
        input_data = dataset.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        model.train()

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            out, _ = model(input_data)
            loss = F.nll_loss(out[input_data.train_mask], input_data.y[input_data.train_mask])
            loss.backward()
            optimizer.step()

    def get_embedding(self, input_data, device: str = 'cpu'):
        self.eval()
        input_data = input_data.to(device)
        with torch.no_grad():
            x, edge_index = input_data.x, input_data.edge_index
            embedding = self.conv(x, edge_index)
        return embedding

    def predict(self, dataset, device: str = 'cpu'):
        model = self.to(device)
        input_data = dataset.to(device)
        model.eval()
        _, pred = model(input_data)[0].max(dim=1)
        correct = float(pred[input_data.test_mask].eq(input_data.y[input_data.test_mask]).sum().item())
        acc = correct / input_data.test_mask.sum().item()
        return pred, acc

    def predict_valid(self, dataset, device: str = 'cpu'):
        model = self.to(device)
        input_data = dataset.to(device)
        model.eval()
        _, pred = model(input_data)[0].max(dim=1)
        correct = float(pred[input_data.val_mask].eq(input_data.y[input_data.val_mask]).sum().item())
        acc = correct / input_data.val_mask.sum().item()
        return pred, acc


class GCNNet(nn.Module):
    """Graph Convolutional Network."""

    def __init__(self, num_features: int, num_classes: int, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.5, seed: Optional[int] = None):
        super(GCNNet, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(num_features, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, num_classes))
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1), x

    def fit(self, data, num_epochs: int, lr: float, weight_decay: float, device):
        self.to(device)
        data = data.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        self.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            out, _ = self(data)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

    def get_embedding(self, data, device: str = 'cpu'):
        self.eval()
        data = data.to(device)
        with torch.no_grad():
            x, edge_index = data.x, data.edge_index
            for i, conv in enumerate(self.convs[:-1]):
                x = conv(x, edge_index)
                x = F.relu(x)
                if i < len(self.convs) - 2:
                    x = F.dropout(x, p=self.dropout, training=False)
        return x

    def predict(self, data, device):
        self.eval()
        data = data.to(device)
        with torch.no_grad():
            out, _ = self(data)
            pred = out.argmax(dim=1)
            correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
            acc = correct / data.test_mask.sum().item()
        return pred.cpu().numpy(), acc

    def predict_valid(self, data, device):
        self.eval()
        data = data.to(device)
        with torch.no_grad():
            out, _ = self(data)
            pred = out.argmax(dim=1)
            correct = pred[data.val_mask].eq(data.y[data.val_mask]).sum().item()
            acc = correct / data.val_mask.sum().item()
        return pred.cpu().numpy(), acc


class SGConvNoWeight(MessagePassing):
    """SGConv without learnable weights for feature propagation."""

    def __init__(self, K: int = 2, cached: bool = False, add_self_loops: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.K = K
        self.cached = cached
        self.add_self_loops = add_self_loops

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        if isinstance(edge_index, Tensor):
            edge_index, edge_weight = gcn_norm(
                edge_index, edge_weight, x.size(self.node_dim), False,
                self.add_self_loops, self.flow, dtype=x.dtype)
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, x.size(self.node_dim), False,
                self.add_self_loops, self.flow, dtype=x.dtype)

        for k in range(self.K):
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
        return x

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_model(model_name: str, num_features: int, num_classes: int,
                 config: Optional[dict] = None, seed: int = 0):
    """
    Create a GNN model with specified configuration.

    Args:
        model_name: 'sgc' or 'gcn'
        num_features: Number of input features
        num_classes: Number of output classes
        config: Optional configuration dict with 'hidden_dim', 'num_layers', 'dropout'
        seed: Random seed for reproducibility

    Returns:
        Initialized model
    """
    if config is None:
        config = {'hidden_dim': 128, 'num_layers': 2, 'dropout': 0.5}

    if model_name.lower() == 'sgc':
        return SGCNet(num_features, num_classes,
                      hidden_dim=config.get('hidden_dim', 128), seed=seed)
    elif model_name.lower() == 'gcn':
        return GCNNet(num_features, num_classes,
                      hidden_dim=config.get('hidden_dim', 128),
                      num_layers=config.get('num_layers', 2),
                      dropout=config.get('dropout', 0.5),
                      seed=seed)
    else:
        raise ValueError(f"Unsupported model: {model_name}. Use 'sgc' or 'gcn'.")
