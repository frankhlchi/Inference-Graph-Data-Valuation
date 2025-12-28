"""
SVGL: Shapley-Guided Utility Learning for Graph Inference Data Valuation

A framework for quantifying the importance of test-time neighbors in Graph Neural Networks
using Shapley value estimation with transferable utility learning.
"""

__version__ = "0.1.0"
__author__ = "SVGL Authors"

from .models.gnn import SGCNet, GCNNet, SGConvNoWeight
from .models.lasso import LassoRegression
from .data.datasets import load_dataset
from .data.preprocess import preprocess_data, load_preprocessed_data

__all__ = [
    "SGCNet",
    "GCNNet",
    "SGConvNoWeight",
    "LassoRegression",
    "load_dataset",
    "preprocess_data",
    "load_preprocessed_data",
]
