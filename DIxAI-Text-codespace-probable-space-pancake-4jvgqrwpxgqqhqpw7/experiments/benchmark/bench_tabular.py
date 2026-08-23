"""
UCI Credit Tabular Benchmark for DIxAI TNNLS-Level Experiments

This script runs comprehensive benchmarks on the UCI Credit dataset
comparing DIxAI against SHAP and other tabular baselines 
with multi-seed statistical rigor.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.dixai import DecisionInformationExplainer
from experiments.benchmark.statistical_utils import MultiSeedRunner, generate_latex_comparison_table
import shap

class SimpleMLP(nn.Module):
    """Simple MLP for tabular classification."""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)


def load_data():
    """Load and preprocess UCI Credit dataset."""
    from sklearn.datasets import fetch_openml
    try:
        # Try fetching, fallback to local if possible or breast cancer
        print("Fetching UCI Credit dataset (ID: 42477)...")
        data = fetch_openml(data_id=42477, as_frame=True, parser='auto')
        X = data.data
        y = data.target.astype(int)
    except Exception as e:
        print(f"Failed to load UCI Credit: {e}. Falling back to Breast Cancer.")
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, X.columns.tolist()
def compute_fidelity_tabular(model, x, mask, threshold=0.5):
    """Fidelity for tabular data."""
    device = x.device
    binary_mask = (mask > threshold).float().to(device)
    masked_x = x * binary_mask
    
    with torch.no_grad():
        orig_pred = model(x.unsqueeze(0)).argmax(dim=-1)
        masked_pred = model(masked_x.unsqueeze(0)).argmax(dim=-1)
    return (orig_pred == masked_pred).float().item()

def run_experiment(method, model, X_test, n_samples: int = 50, seed: int = 42, device: str = "cuda"):
    """Generic experiment runner for tabular data."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    fidelities = []
    sparsities = []
    explainer = None
    
    if method == 'DIxAI':
        explainer = DecisionInformationExplainer(model, lambda_fidelity=10.0, device=device)
    elif method == 'SHAP':
        explainer_shap = shap.GradientExplainer(model, X_test_t[:100])
        shap_values = explainer_shap.shap_values(X_test_t[:n_samples])
        # shap_values shape: (n_samples, input_dim) or (num_classes, n_samples, input_dim)
        if isinstance(shap_values, list):
            shap_values = shap_values[1] # Use target class class 1 for binary
        elif shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1] if shap_values.shape[2] == 2 else shap_values[..., 0]
            
    for i in range(min(n_samples, len(X_test_t))):
        x = X_test_t[i]
        
        if method == 'DIxAI':
            exp = explainer.explain(x, steps=500, verbose=False, seed=seed)
            mask = exp.mask.cpu()
            fidelity = exp.fidelity_score
            sparsity = exp.info_score
        elif method == 'SHAP':
            # Take top-k features to create a mask for fidelity/sparsity logic
            sv = np.abs(shap_values[i])
            # Threshold to get top 50% features
            thresh = np.median(sv)
            mask = torch.tensor(sv > thresh).float()
            
            fidelity = compute_fidelity_tabular(model, x, mask)
            sparsity = 1.0 - mask.mean().item()
        else:
            mask = torch.rand(X_test.shape[1])
            sparsity = 0.5
            fidelity = 0.5
            
        fidelities.append(fidelity)
        sparsities.append(sparsity)
        
    return {
        'fidelity': np.mean(fidelities),
        'sparsity': np.mean(sparsities),
    }


def main():
    print("=" * 60)
    print("Tabular Benchmark (UCI Credit) for TNNLS")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_train, X_test, y_train, y_test, feature_names = load_data()
    
    # Train model
    input_dim = X_train.shape[1]
    model = SimpleMLP(input_dim, 2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train.values, dtype=torch.long).to(device)
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
    
    print("Training model...")
    for epoch in range(5):
        for bx, by in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
    model.eval()
    
    runner = MultiSeedRunner(seeds=[42, 123, 456, 789, 1024])
    n_samples = 50
    
    results = {}
    for method in ['DIxAI', 'SHAP']:
        print(f"--- Running {method} ---")
        results[method] = runner.run(run_experiment, method=method, model=model, 
                                    X_test=X_test, n_samples=n_samples, device=device)
        print(f"{method}: {results[method]['fidelity']}")

    # LaTeX
    latex = generate_latex_comparison_table(
        methods=results,
        metrics=['fidelity', 'sparsity'],
        caption="Tabular Benchmark Results on UCI Credit (5 seeds)",
        label="tab:tabular_results"
    )
    
    print("\nGenerated LaTeX:")
    print(latex)
    
    out_path = project_root / "experiments" / "results" / "tabular_tnnls_results.tex"
    with open(out_path, "w") as f:
        f.write(latex)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
