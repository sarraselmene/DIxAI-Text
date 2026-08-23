"""
Ablation Study for DIxAI
Tests the contribution of each component to overall performance.

Configurations:
1. Full DIxAI (baseline)
2. No IB Term (lambda_info = 0)
3. No Fidelity Term (lambda = 0)
4. Deterministic Mask (no Gumbel noise)
5. No TV Regularization
6. No Annealing
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============== Model Definition ==============
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# ============== Ablation Configurations ==============
class AblationConfig:
    def __init__(self, name, use_gumbel=True, use_tv=True, use_annealing=True, 
                 lambda_info=1.0, lambda_fidelity=10.0):
        self.name = name
        self.use_gumbel = use_gumbel
        self.use_tv = use_tv
        self.use_annealing = use_annealing
        self.lambda_info = lambda_info
        self.lambda_fidelity = lambda_fidelity

CONFIGS = [
    AblationConfig("Full DIxAI", use_gumbel=True, use_tv=True, use_annealing=True, lambda_info=1.0, lambda_fidelity=10.0),
    AblationConfig("No IB Term", use_gumbel=True, use_tv=True, use_annealing=True, lambda_info=0.0, lambda_fidelity=10.0),
    AblationConfig("No Fidelity", use_gumbel=True, use_tv=True, use_annealing=True, lambda_info=1.0, lambda_fidelity=0.0),
    AblationConfig("Deterministic", use_gumbel=False, use_tv=True, use_annealing=True, lambda_info=1.0, lambda_fidelity=10.0),
    AblationConfig("No TV", use_gumbel=True, use_tv=False, use_annealing=True, lambda_info=1.0, lambda_fidelity=10.0),
    AblationConfig("No Annealing", use_gumbel=True, use_tv=True, use_annealing=False, lambda_info=1.0, lambda_fidelity=10.0),
]

# ============== Simplified Explainer for Ablation ==============
def run_ablation_explain(model, x, baseline, config, steps=300, lr=0.1):
    """Run a single explanation with specified ablation configuration."""
    device = x.device
    d = x.shape[-1]
    
    # Initialize mask logits
    logits = nn.Parameter(torch.zeros(d, device=device))
    optimizer = torch.optim.Adam([logits], lr=lr)
    
    # Get original prediction
    with torch.no_grad():
        orig_pred = model(x)
        orig_class = orig_pred.argmax(dim=-1)
    
    history = {"loss": [], "sparsity": [], "fidelity": []}
    
    for step in range(steps):
        optimizer.zero_grad()
        
        # Temperature annealing
        if config.use_annealing:
            tau = max(0.5, 1.0 - step / steps * 0.5)
        else:
            tau = 1.0
        
        # Gumbel-Softmax or deterministic
        if config.use_gumbel:
            u = torch.rand_like(logits)
            g = -torch.log(-torch.log(u + 1e-10) + 1e-10)
            mask = torch.sigmoid((logits + g) / tau)
        else:
            mask = torch.sigmoid(logits)
        
        # Apply mask
        z = x * mask + baseline * (1 - mask)
        masked_pred = model(z)
        
        # Information term (sparsity)
        loss_info = mask.mean() * config.lambda_info
        
        # Fidelity term (KL divergence)
        log_p = torch.log_softmax(masked_pred, dim=-1)
        q = torch.softmax(orig_pred, dim=-1)
        loss_fidelity = (q * (torch.log(q + 1e-10) - log_p)).sum() * config.lambda_fidelity
        
        # TV regularization (for tabular: L1 diff between adjacent features)
        if config.use_tv and d > 1:
            tv_loss = torch.abs(mask[1:] - mask[:-1]).mean() * 0.1
        else:
            tv_loss = 0.0
        
        # Total loss
        loss = loss_info + loss_fidelity + tv_loss
        
        loss.backward()
        optimizer.step()
        
        # Track metrics
        with torch.no_grad():
            sparsity = (mask < 0.5).float().mean().item()
            fidelity = (masked_pred.argmax(dim=-1) == orig_class).float().item()
            history["loss"].append(loss.item())
            history["sparsity"].append(sparsity)
            history["fidelity"].append(fidelity)
    
    # Final mask
    with torch.no_grad():
        final_mask = torch.sigmoid(logits)
        binary_mask = (final_mask > 0.5).float()
        z_final = x * binary_mask + baseline * (1 - binary_mask)
        final_pred = model(z_final)
        final_fidelity = (final_pred.argmax(dim=-1) == orig_class).float().item()
        final_sparsity = (binary_mask < 0.5).float().mean().item()
    
    return {
        "fidelity": final_fidelity,
        "sparsity": final_sparsity,
        "convergence_steps": len(history["loss"]),
        "mask": binary_mask.cpu().numpy()
    }

# ============== Main Ablation Study ==============
def run_ablation_study():
    print("=" * 60)
    print("DIxAI Ablation Study")
    print("=" * 60)
    
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train model
    model = SimpleMLP(input_dim=4, hidden_dim=64, output_dim=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    
    print("\nTraining classifier...")
    for epoch in range(200):
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
    
    # Calculate test accuracy
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    with torch.no_grad():
        test_acc = (model(X_test_t).argmax(dim=-1) == y_test_t).float().mean().item()
    print(f"Test Accuracy: {test_acc:.2%}")
    
    # Baseline (dataset mean)
    baseline = torch.tensor(X_train.mean(axis=0), dtype=torch.float32)
    
    # Run ablation for each configuration
    results = []
    n_samples = min(30, len(X_test))
    
    for config in CONFIGS:
        print(f"\n--- Running: {config.name} ---")
        fidelities, sparsities = [], []
        
        for i in range(n_samples):
            x = X_test_t[i:i+1]
            result = run_ablation_explain(model, x, baseline, config, steps=300)
            fidelities.append(result["fidelity"])
            sparsities.append(result["sparsity"])
        
        avg_fidelity = np.mean(fidelities)
        avg_sparsity = np.mean(sparsities)
        std_fidelity = np.std(fidelities)
        std_sparsity = np.std(sparsities)
        
        results.append({
            "Configuration": config.name,
            "Fidelity": f"{avg_fidelity:.2f} ± {std_fidelity:.2f}",
            "Sparsity": f"{avg_sparsity:.2f} ± {std_sparsity:.2f}",
            "Fidelity_mean": avg_fidelity,
            "Sparsity_mean": avg_sparsity,
        })
        print(f"  Fidelity: {avg_fidelity:.2f} ± {std_fidelity:.2f}")
        print(f"  Sparsity: {avg_sparsity:.2f} ± {std_sparsity:.2f}")
    
    # Save results
    df = pd.DataFrame(results)
    output_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'ablation_study_results.csv')
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # Print LaTeX table
    print("\n" + "=" * 60)
    print("LaTeX Table for Manuscript:")
    print("=" * 60)
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Ablation Study: Contribution of DIxAI Components}")
    print(r"\label{tab:ablation}")
    print(r"\begin{tabular}{@{}lcc@{}}")
    print(r"\toprule")
    print(r"\textbf{Configuration} & \textbf{Fidelity} & \textbf{Sparsity} \\ \midrule")
    for r in results:
        print(f"{r['Configuration']:20} & {r['Fidelity']:15} & {r['Sparsity']:15} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    
    return df

if __name__ == "__main__":
    run_ablation_study()
