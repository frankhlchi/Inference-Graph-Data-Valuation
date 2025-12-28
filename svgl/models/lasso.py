"""
Lasso regression model for Shapley value prediction.

Implements non-negative constrained LASSO for interpretable feature selection.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from tqdm import tqdm


class LassoRegression(nn.Module):
    """
    Non-negative constrained LASSO regression model.

    The model learns weights that are constrained to be non-negative,
    ensuring interpretability of feature importance.
    """

    def __init__(self, input_dim: int):
        super(LassoRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.intercept = nn.Parameter(torch.zeros(1))
        self.apply_constraint()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze() + self.intercept

    def apply_constraint(self):
        """Apply non-negativity constraint on weights."""
        with torch.no_grad():
            self.linear.weight.data = torch.clamp(self.linear.weight.data, min=0)


def train_lasso(X: np.ndarray, y: np.ndarray, feature_names: list,
                device: str = 'cpu', k_folds: int = 5,
                verbose: bool = True) -> tuple:
    """
    Train LASSO regression with cross-validation for alpha selection.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target values (n_samples,)
        feature_names: List of feature names
        device: Device to use for training
        k_folds: Number of cross-validation folds
        verbose: Whether to print progress

    Returns:
        Tuple of (trained_model, selected_features, best_alpha)
    """
    # Handle NaN values
    X = np.nan_to_num(X, nan=0.0)
    input_dim = X.shape[1]

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    alphas = torch.logspace(-8, 2, 100, device=device)
    best_alpha = None
    best_model_state = None
    best_overall_score = float('-inf')

    iterator = tqdm(alphas, desc="Searching for best alpha") if verbose else alphas

    for alpha in iterator:
        fold_scores = []
        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            X_train_t = torch.FloatTensor(X_train).to(device)
            y_train_t = torch.FloatTensor(y_train).to(device)
            X_val_t = torch.FloatTensor(X_val).to(device)
            y_val_t = torch.FloatTensor(y_val).to(device)

            model = LassoRegression(input_dim).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=alpha)

            best_val_score = float('-inf')
            patience = 500
            no_improve = 0

            for epoch in range(10000):
                model.train()
                optimizer.zero_grad()
                y_pred = model(X_train_t)
                loss = nn.MSELoss()(y_pred, y_train_t) + alpha * model.linear.weight.abs().sum()
                loss.backward()
                optimizer.step()
                model.apply_constraint()

                model.eval()
                with torch.no_grad():
                    y_val_pred = model(X_val_t)
                    val_score = -nn.MSELoss()(y_val_pred, y_val_t).item()

                if val_score > best_val_score:
                    best_val_score = val_score
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve >= patience:
                    break

            fold_scores.append(best_val_score)

        avg_score = np.mean(fold_scores)
        if avg_score > best_overall_score:
            best_overall_score = avg_score
            best_alpha = alpha.item()

    # Train final model on all data
    final_model = LassoRegression(input_dim).to(device)
    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.FloatTensor(y).to(device)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=0.001, weight_decay=best_alpha)

    best_score = float('-inf')
    patience = 1000
    no_improve = 0

    for epoch in range(10000):
        final_model.train()
        optimizer.zero_grad()
        y_pred = final_model(X_tensor)
        loss = nn.MSELoss()(y_pred, y_tensor) + best_alpha * final_model.linear.weight.abs().sum()
        loss.backward()
        optimizer.step()
        final_model.apply_constraint()

        final_model.eval()
        with torch.no_grad():
            y_pred = final_model(X_tensor)
            score = -nn.MSELoss()(y_pred, y_tensor).item()

        if score > best_score:
            best_score = score
            best_model_state = final_model.state_dict()
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    final_model.load_state_dict(best_model_state)

    # Get selected features
    weights = final_model.linear.weight.squeeze()
    selected_indices = torch.where(weights > 1e-5)[0].cpu().numpy()
    selected_features = [(feature_names[i], weights[i].item()) for i in selected_indices]

    if verbose:
        print(f"\nBest alpha: {best_alpha:.10f}")
        print(f"Number of selected features: {len(selected_features)}")
        print("Selected features:")
        for name, weight in selected_features:
            print(f"  - {name}: {weight:.6f}")

    return final_model, selected_features, best_alpha
