
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from captum.attr import IntegratedGradients
import os
import sys
from scipy import stats

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.dixai import DecisionInformationExplainer

# --- Models & Helpers ---

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x): return self.net(x)

class SimpleL2X(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.selector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim), nn.Sigmoid()
        )
    def forward(self, x): return self.selector(x)

def train_l2x_instance(model, x, target_class, steps=100, lambda_sparse=0.1):
    device = x.device
    l2x = SimpleL2X(x.shape[-1]).to(device)
    optimizer = optim.Adam(l2x.parameters(), lr=0.05)
    model.eval()
    x_in = x.unsqueeze(0)
    for _ in range(steps):
        optimizer.zero_grad()
        m = l2x(x_in)
        loss = -F.log_softmax(model(x_in * m), dim=-1)[0, target_class] + lambda_sparse * m.mean()
        loss.backward()
        optimizer.step()
    return l2x(x_in).detach()

def get_data():
    data = load_iris()
    X, y = data.data, data.target
    return train_test_split(StandardScaler().fit_transform(X), y, test_size=0.2, random_state=42)

def bootstrap_ci(data, n_boot=1000, ci=0.95):
    """Compute Bootstrap Confidence Interval."""
    means = []
    data = np.array(data)
    for _ in range(n_boot):
        sample = np.random.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    return lower, upper

def cohens_d(x, y):
    """Compute Cohen's d effect size."""
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)

# --- Main Benchmark ---

def run_multiseed():
    seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Rigorous Multi-Seed Benchmark on {device} (Seeds={len(seeds)})")
    
    X_train, X_test, y_train, y_test = get_data()
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    
    results = {'DIxAI': [], 'L2X': [], 'IG': []}

    for seed in seeds:
        print(f"Processing Seed {seed}...")
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Train Model
        model = SimpleMLP(4, 3).to(device)
        opt = optim.Adam(model.parameters(), lr=0.01)
        crit = nn.CrossEntropyLoss()
        for _ in range(50):
            loss = crit(model(X_train_t), y_train_t)
            opt.zero_grad(); loss.backward(); opt.step()
            
        # Explainers
        dixai = DecisionInformationExplainer(model, lambda_fidelity=5.0, device=device)
        ig = IntegratedGradients(model)
        
        # Method Accumulators for this Seed
        seed_stats = {'DIxAI': {'fid': [], 'spar': []}, 
                      'L2X': {'fid': [], 'spar': []}, 
                      'IG': {'fid': [], 'spar': []}}
        
        N = 30 # Sample size matches paper
        for i in range(N):
            x = X_test_t[i]
            target = y_test_t[i].item()
            
            # DIxAI
            exp = dixai.explain(x, steps=100, verbose=False)
            seed_stats['DIxAI']['fid'].append(exp.fidelity_score)
            seed_stats['DIxAI']['spar'].append(exp.info_score)
            
            # L2X
            mask = train_l2x_instance(model, x, target, steps=50)
            bind = (mask > 0.5).float()
            fid = (model(x.unsqueeze(0)*bind).argmax(dim=-1).item() == target)
            spar = 1.0 - bind.mean().item()
            seed_stats['L2X']['fid'].append(float(fid))
            seed_stats['L2X']['spar'].append(spar)
            
            # IG
            attr = ig.attribute(x.unsqueeze(0), target=target)
            mask = (attr.abs() > torch.quantile(attr.abs(), 0.5)).float()
            fid = (model(x.unsqueeze(0)*mask).argmax(dim=-1).item() == target)
            spar = 0.5
            seed_stats['IG']['fid'].append(float(fid))
            seed_stats['IG']['spar'].append(spar)

        # Store mean of this seed
        for m in ['DIxAI', 'L2X', 'IG']:
            results[m].append({
                'Fidelity': np.mean(seed_stats[m]['fid']),
                'Sparsity': np.mean(seed_stats[m]['spar'])
            })
            
    # --- Statistical Analysis ---
    report = "\n--- Statistical Report (N=10 Seeds) ---\n"
    dataframes = {}
    
    for m in ['DIxAI', 'L2X', 'IG']:
        df = pd.DataFrame(results[m])
        dataframes[m] = df
        
        fid_mean, fid_std = df['Fidelity'].mean(), df['Fidelity'].std()
        spar_mean, spar_std = df['Sparsity'].mean(), df['Sparsity'].std()
        fid_ci = bootstrap_ci(df['Fidelity'])
        spar_ci = bootstrap_ci(df['Sparsity'])
        
        report += f"\nMethod: {m}\n"
        report += f"Fidelity: {fid_mean:.3f} ± {fid_std:.3f} (95% CI: [{fid_ci[0]:.3f}, {fid_ci[1]:.3f}])\n"
        report += f"Sparsity: {spar_mean:.3f} ± {spar_std:.3f} (95% CI: [{spar_ci[0]:.3f}, {spar_ci[1]:.3f}])\n"
        
    # Effect Sizes
    d_dixai_l2x = cohens_d(dataframes['DIxAI']['Sparsity'], dataframes['L2X']['Sparsity'])
    report += f"\nEffect Size (DIxAI vs L2X Sparsity): Cohen's d = {d_dixai_l2x:.3f}\n"
    
    print(report)
    with open('experiments/results/robustness_report.txt', 'w') as f:
        f.write(report)
        
    # Save CSVs
    for m, df in dataframes.items():
        df['Method'] = m
    full_df = pd.concat(dataframes.values())
    full_df.to_csv('experiments/results/robustness_stats.csv', index=False)

if __name__ == "__main__":
    run_multiseed()
