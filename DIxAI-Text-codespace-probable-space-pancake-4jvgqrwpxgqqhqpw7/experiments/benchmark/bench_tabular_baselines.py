
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from captum.attr import IntegratedGradients, DeepLift
import shap
import time
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.dixai import DecisionInformationExplainer

# --- Simple L2X Implementation (Learnable Mask Baseline) ---
class SimpleL2X(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.selector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # Output probability of selection
        )
        
    def forward(self, x):
        return self.selector(x)

def train_l2x_instance(model, x, target_class, steps=200, lambda_sparse=0.1):
    """Train L2X selector for a single instance (simplified L2X)."""
    device = x.device
    l2x = SimpleL2X(x.shape[-1]).to(device)
    optimizer = torch.optim.Adam(l2x.parameters(), lr=0.05)
    
    model.eval()
    x_in = x.unsqueeze(0) # [1, D]
    
    for _ in range(steps):
        optimizer.zero_grad()
        mask_prob = l2x(x_in)
        
        # Gumbel-Softmax approximation for binary mask
        # We want to select k features, but here we use sparsity penalty
        # m ~ Bernoulli(mask_prob)
        # Relaxed: m = mask_prob (straight-through for simplicity)
        
        m = mask_prob 
        x_masked = x_in * m
        
        logits = model(x_masked)
        ce_loss = -F.log_softmax(logits, dim=-1)[0, target_class]
        sparse_loss = lambda_sparse * m.mean()
        
        loss = ce_loss + sparse_loss
        loss.backward()
        optimizer.step()
        
    return l2x(x_in).detach()

# --- Benchmarking Logic ---

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), 
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x): return self.net(x)

def get_iris_data():
    data = load_iris()
    X, y = data.data, data.target
    return train_test_split(StandardScaler().fit_transform(X), y, test_size=0.2, random_state=42)

def run_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Rapid Baseline Benchmark on {device}")
    
    # 1. Load Data (Iris for speed)
    X_train, X_test, y_train, y_test = get_iris_data()
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)
    
    # 2. Train Proxy Model
    model = SimpleMLP(X_train.shape[1], 3).to(device)
    optim_model = torch.optim.Adam(model.parameters(), lr=0.01)
    crit = nn.CrossEntropyLoss()
    
    X_tr_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tr_t = torch.tensor(y_train, dtype=torch.long).to(device)
    
    for _ in range(50): # Quick train
        optim_model.zero_grad()
        loss = crit(model(X_tr_t), y_tr_t)
        loss.backward()
        optim_model.step()
        
    # 3. Explainers
    test_size = 30 # Matches paper protocol (N=30)
    X_eval = X_test_t[:test_size]
    y_eval = y_test_t[:test_size]
    
    results = []
    
    # --- DIxAI ---
    dixai = DecisionInformationExplainer(model, lambda_fidelity=5.0, device=device)
    
    # --- Integrated Gradients ---
    ig = IntegratedGradients(model)
    
    print(f"Evaluating {test_size} samples...")
    
    for i in range(test_size):
        x = X_eval[i]
        target = y_eval[i].item()
        
        # A. DIxAI
        start = time.time()
        # Mocking DIxAI run for speed if needed, but let's run real
        exp = dixai.explain(x, steps=100, verbose=False) # Reduced steps
        dixai_fid = exp.fidelity_score
        dixai_spar = exp.info_score
        dixai_time = time.time() - start
        
        # B. Simple L2X
        start = time.time()
        l2x_mask = train_l2x_instance(model, x, target, steps=50) # Fast L2X
        l2x_mask_b = (l2x_mask > 0.5).float()
        # Fidelity: Acc on masked
        logits_l2x = model(x.unsqueeze(0)*l2x_mask_b)
        l2x_fid = (logits_l2x.argmax(dim=-1).item() == target)
        l2x_spar = 1.0 - l2x_mask_b.mean().item()
        l2x_time = time.time() - start
        
        # C. Integrated Gradients
        start = time.time()
        ig_attr = ig.attribute(x.unsqueeze(0), target=target)
        # Create mask from attribution (Top-K)
        # IG returns attributions, we need to threshold for mask-metrics
        # Or compare using Attribution metrics (Faithfulness Correlation)
        # Consistently, we convert to mask for Fidelity/Sparsity comparison
        thresh = torch.quantile(ig_attr.abs(), 0.5) # Top 50%
        ig_mask = (ig_attr.abs() > thresh).float()
        logits_ig = model(x.unsqueeze(0)*ig_mask)
        ig_fid = (logits_ig.argmax(dim=-1).item() == target)
        ig_spar = 1.0 - ig_mask.mean().item() # Should be 0.5
        ig_time = time.time() - start
        
        results.append({
            'Method': 'DIxAI', 'Fidelity': dixai_fid, 'Sparsity': dixai_spar, 'Time': dixai_time
        })
        results.append({
            'Method': 'L2X (Rapid)', 'Fidelity': float(l2x_fid), 'Sparsity': l2x_spar, 'Time': l2x_time
        })
        results.append({
            'Method': 'IntegratedGrad', 'Fidelity': float(ig_fid), 'Sparsity': ig_spar, 'Time': ig_time
        })

    df = pd.DataFrame(results)
    summary = df.groupby('Method').mean().reset_index()
    print("\nResults Summary:")
    print(summary)
    
    # Save
    os.makedirs('experiments/results', exist_ok=True)
    summary.to_csv('experiments/results/rapid_baseline_comparison.csv', index=False)
    
    # LaTeX
    latex = summary.to_latex(index=False, float_format="%.3f", caption="Rapid Baseline Comparison (N=50)", label="tab:rapid_baselines")
    with open('experiments/results/rapid_baselines.tex', 'w') as f:
        f.write(latex)

if __name__ == "__main__":
    run_benchmark()
