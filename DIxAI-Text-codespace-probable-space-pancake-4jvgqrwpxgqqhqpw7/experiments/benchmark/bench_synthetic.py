
"""
Benchmarking Feature Selection on Synthetic Data with Known Ground Truth
(PAMI Safety Hardening - Item 2)

Tests if DIxAI can correctly identify the 'informative' features
in a controlled synthetic dataset where ground truth is known.

Metrics:
- Precision@k: Fraction of top-k features that are truly informative.
- Recall@k: Fraction of informative features found in top-k.
- AUCPR: Area Under Precision-Recall Curve.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, auc
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
from dixai import DecisionInformationExplainer

# --- 1. Data Generation ---
def get_synthetic_data(n_samples=2000, n_features=25, n_informative=5, random_state=42):
    """
    Generate synthetic classification data.
    Returns: X, y, informative_indices
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        class_sep=1.5, # Moderate separation
        shuffle=False, # Important: Features 0..n_informative-1 are informative
        random_state=random_state
    )
    
    # Shuffle features to make it harder (and keep track of indices)
    # Actually, let's keep it simple: First 5 are informative.
    # If we shuffle, we need to return the permutation.
    # For this benchmark, we can rely on make_classification's default behavior 
    # (if shuffle=False, informative are at the beginning).
    # But standard practice is to shuffle.
    
    rng = np.random.RandomState(random_state)
    perm = rng.permutation(n_features)
    X = X[:, perm]
    
    # Identify informative indices
    # Default make_classification (shuffle=False): indices 0..n_informative-1 are informative.
    # After permutation: original indices [0..n_inf-1] mapped to perm[0..n_inf-1]?
    # No, X[:, perm] moves column i to column perm[i]? No.
    # X[:, perm] sets new column i to be old column perm[i].
    # So if old column j was informative, it is now at index i where perm[i] == j.
    # We want indices i such that perm[i] < n_informative.
    
    informative_idx = np.where(perm < n_informative)[0]
    
    return X, y, informative_idx

# --- 2. Simple Model ---
class SimpleMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x): return self.net(x)

# --- 3. Metrics ---
def compute_metrics(attribution, informative_idx, k=5):
    """
    Compute Precision/Recall against ground truth.
    attribution: (n_features,) tensor/array of importance scores
    informative_idx: list/array of true informative feature indices
    """
    # Normalize/Rank
    if isinstance(attribution, torch.Tensor):
        attr = attribution.abs().detach().cpu().numpy()
    else:
        attr = np.abs(attribution)
        
    # Get top-k indices
    top_k_idx = np.argsort(attr)[::-1][:k]
    
    # Intersection
    intersection = len(set(top_k_idx) & set(informative_idx))
    
    precision = intersection / k
    recall = intersection / len(informative_idx)
    
    # AUCPR (Treat as binary classification of features)
    y_true = np.zeros(len(attr))
    y_true[informative_idx] = 1
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, attr)
    auc_score = auc(recall_curve, precision_curve)
    
    return precision, recall, auc_score

def run_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Synthetic Ground Truth Benchmark on {device}")
    
    # Params
    N_FEATURES = 50
    N_INFORMATIVE = 10
    N_SAMPLES = 2000
    
    # 1. Data
    print("Generating synthetic data...")
    X, y, true_idx = get_synthetic_data(N_SAMPLES, N_FEATURES, N_INFORMATIVE)
    X = StandardScaler().fit_transform(X) # Scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)
    
    # 2. Train Model
    print("Training MLP...")
    model = SimpleMLP(N_FEATURES).to(device)
    opt = optim.Adam(model.parameters(), lr=0.01)
    crit = nn.CrossEntropyLoss()
    
    for epoch in range(50):
        opt.zero_grad()
        loss = crit(model(X_train_t), y_train_t)
        loss.backward()
        opt.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.4f}")
            
    # Test Acc
    model.eval()
    with torch.no_grad():
        acc = (model(X_test_t).argmax(1) == y_test_t).float().mean()
    print(f"Test Accuracy: {acc:.4f}")
    
    # 3. Explain and Evaluate
    print("Evaluating Explanations...")
    dixai = DecisionInformationExplainer(model, lambda_fidelity=5.0, device=device)
    ig = IntegratedGradients(model)
    
    results = []
    
    N_EVAL = 100
    for i in range(N_EVAL):
        x = X_test_t[i]
        target = y_test_t[i].item()
        
        # A. Random Baseline
        rand_attr = np.random.rand(N_FEATURES)
        p, r, a = compute_metrics(rand_attr, true_idx, k=N_INFORMATIVE)
        results.append({'Method': 'Random', 'Precision': p, 'Recall': r, 'AUCPR': a})
        
        # B. Integrated Gradients
        ig_attr = ig.attribute(x.unsqueeze(0), target=target).squeeze().abs().detach().cpu().numpy()
        p, r, a = compute_metrics(ig_attr, true_idx, k=N_INFORMATIVE)
        results.append({'Method': 'IntegratedGrad', 'Precision': p, 'Recall': r, 'AUCPR': a})
        
        # C. DIxAI (Tune lambda)
        for lam in [0.5, 2.0]:
            try:
                # Re-initialize to update lambda_fidelity
                dixai_tuned = DecisionInformationExplainer(model, lambda_fidelity=lam, device=device)
                exp = dixai_tuned.explain(x, steps=500, anneal=True, verbose=False)
                dixai_attr = exp.mask.cpu().numpy().flatten()
                p, r, a = compute_metrics(dixai_attr, true_idx, k=N_INFORMATIVE)
                results.append({'Method': f'DIxAI (lam={lam})', 'Precision': p, 'Recall': r, 'AUCPR': a})
            except Exception as e:
                print(f"DIxAI Error: {e}")
            
    # Summary
    df = pd.DataFrame(results)
    summary = df.groupby('Method').mean().reset_index()
    print("\n--- Synthetic Benchmark Results (N=100) ---")
    print(summary)
    
    # Save
    os.makedirs('experiments/results', exist_ok=True)
    summary.to_csv('experiments/results/synthetic_benchmark.csv', index=False)
    print("Saved to experiments/results/synthetic_benchmark.csv")

if __name__ == "__main__":
    run_benchmark()
