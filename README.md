# Graph Inference Data Valuation Framework (SVGL)

![Inference_Data_Valuation_Poster_1](https://github.com/user-attachments/assets/4e598383-17fc-4754-846b-7e0a77039379)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/)

Official implementation of **"Shapley-Guided Utility Learning for Effective Graph Inference Data Valuation"**.

## Overview

SVGL is a novel framework for quantifying the importance of test-time neighbors in Graph Neural Networks (GNNs). It addresses the challenge of evaluating data importance without test labels by:

1. **Transferable Feature Extraction**: Combining data-specific and model-specific features to approximate test accuracy
2. **Shapley-guided Optimization**: Directly optimizing Shapley value prediction through feature Shapley decomposition
3. **Structure-Aware Valuation**: Respecting graph connectivity constraints in value estimation

## Installation

```bash
# Clone the repository
git clone https://github.com/frankhlchi/Inference-Graph-Data-Valuation.git
cd Inference-Graph-Data-Valuation

# Create conda environment (recommended)
conda activate pydvl  # or create new: conda create -n svgl python=3.9

# Install dependencies
pip install -r requirements.txt

# Install SVGL in development mode
pip install -e .
```

## Quick Start

### Run Demo (Recommended)

```bash
# Quick demo on Cora dataset (takes ~2 minutes)
python scripts/run_demo.py --dataset Cora --num_samples 5

# Full experiment with more samples
python scripts/run_demo.py --dataset Cora --num_samples 30
```

### Large-Scale Reproduction

```bash
# Run multiple datasets / seeds (writes to outputs/reproduction/<timestamp>/)
python scripts/run_parallel_experiments.py \
  --datasets Cora Citeseer Pubmed CS Physics Computers Photo WikiCS \
  --seeds 0 1 2 --num_samples 30 --device cuda:0 --jobs 1
```

### Optional: Base GNN Hyperparameter Tuning

The original codebase includes a grid-search step for the base GNN. This repo supports the
same workflow via `scripts/tune_gnn_hparams.py`, and `scripts/run_parallel_experiments.py`
will automatically apply matching overrides from `configs/best_hparams.yaml` (empty by default).

```bash
# Grid-search base GNN hyperparameters and write to configs/best_hparams.yaml (time-consuming)
python scripts/tune_gnn_hparams.py --datasets Cora Citeseer Pubmed --seed 0 --device cuda:0
```

To disable overrides and force the CLI flags/defaults, pass `--no_best_hparams`.

### Python API

```python
from svgl import load_dataset, preprocess_data, create_model
from svgl.valuation import sample_permutations, shapley_regression
from svgl.utils import fix_seed, get_device

# Setup
fix_seed(42)
device = get_device('auto')

# Load and preprocess data
dataset = load_dataset('Cora', root='./data/')
data = dataset[0].to(device)
split_data = preprocess_data('Cora', setting='inductive', use_pmlp=True)

# Train GNN model
model = create_model('sgc', data.num_features, dataset.num_classes)
# ... training and valuation
```

## Project Structure

```
Inference-Graph-Data-Valuation/
├── svgl/                       # Main package
│   ├── data/                   # Data loading and preprocessing
│   ├── models/                 # SGC, GCN, LASSO + tuning
│   ├── valuation/              # Sampling, features, Shapley + SVGL regression
│   └── utils/                  # Helper functions
├── scripts/                    # Runnable entrypoints
│   ├── run_demo.py
│   ├── run_parallel_experiments.py
│   ├── tune_gnn_hparams.py
│   └── check_progress.sh
├── configs/                    # Config files
│   ├── default.yaml
│   ├── hparam_search.yaml
│   └── best_hparams.yaml
└── outputs/                    # Results directory
```

## Supported Datasets

| Category | Datasets |
|----------|----------|
| Planetoid | Cora, Citeseer, Pubmed |
| Coauthor | CS, Physics |
| Heterophilous | Roman-empire, Amazon-ratings |
| OGB (appendix) | ogbn-arxiv |

## Citation

```bibtex
@article{chi2025shapley,
  title={Shapley-Guided Utility Learning for Effective Graph Inference Data Valuation},
  author={Chi, Hongliang and Wu, Qiong and Zhou, Zhengyi and Ma, Yao},
  journal={arXiv preprint arXiv:2503.18195},
  year={2025}
}
```
