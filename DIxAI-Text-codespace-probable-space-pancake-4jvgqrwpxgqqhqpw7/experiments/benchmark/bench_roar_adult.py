import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.dixai import DecisionInformationExplainer
from experiments.benchmark.datasets import get_tabular_dataset
from experiments.benchmark.models import SimpleMLP, train_model, evaluate_model

def run_roar_adult():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running Adult ROAR Benchmark on {device}...")
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. Load Data
    X_train, X_test, y_train, y_test = get_tabular_dataset('adult')
    X_train, X_test = X_train.to(device), X_test.to(device)
    y_train, y_test = y_train.to(device), y_test.to(device)
    input_dim = X_train.shape[1]
    
    # 2. Train Gold Model
    print("Training Gold Model...")
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=128, shuffle=False)
    
    gold_model = SimpleMLP(input_dim=input_dim).to(device)
    train_model(gold_model, train_loader, epochs=10, lr=0.001)
    base_acc = evaluate_model(gold_model, test_loader)
    print(f"Base Accuracy: {base_acc:.4f}")
    
    # 3. Calculate Feature Importance via DIxAI
    # We explain a representative subset of training data to get global rankings
    print("Calculating DIxAI feature rankings...")
    explainer = DecisionInformationExplainer(gold_model, lambda_fidelity=20.0, device=device)
    
    # Explain 50 random training samples
    indices = np.random.choice(len(X_train), 50, replace=False)
    masks = []
    for idx in indices:
        x_sample = X_train[idx].to(device)
        exp = explainer.explain(x_sample, steps=500, anneal=True, verbose=False)
        masks.append(exp.mask.flatten().detach().cpu().numpy())
    
    # Aggregate importance
    avg_mask = np.mean(masks, axis=0)
    # Get indices of features sorted by importance (descending)
    important_indices = np.argsort(avg_mask)[::-1].copy()
    
    # 4. ROAR Protocol: Remove and Retrain
    fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    roar_results = {"DIxAI": [], "Random": []}
    
    for frac in fractions:
        n_remove = int(frac * input_dim)
        print(f"\nEvaluating fraction {frac:.1f} (removing {n_remove} features)...")
        
        # --- DIxAI Removal ---
        print(f"Retraining with top {n_remove} DIxAI features removed...")
        X_train_dixai = X_train.detach().clone()
        X_test_dixai = X_test.detach().clone()
        if n_remove > 0:
            removed_cols = torch.tensor(important_indices[:n_remove], dtype=torch.long).to(device)
            X_train_dixai.index_fill_(1, removed_cols, 0)
            X_test_dixai.index_fill_(1, removed_cols, 0)
            
        train_loader_dixai = DataLoader(TensorDataset(X_train_dixai, y_train), batch_size=256, shuffle=True)
        test_loader_dixai = DataLoader(TensorDataset(X_test_dixai, y_test), batch_size=256, shuffle=False)
        
        model_dixai = SimpleMLP(input_dim=input_dim).to(device)
        train_model(model_dixai, train_loader_dixai, epochs=5, lr=0.001)
        acc_dixai = evaluate_model(model_dixai, test_loader_dixai)
        roar_results["DIxAI"].append(acc_dixai)
        
        # --- Random Removal ---
        print(f"Retraining with {n_remove} random features removed...")
        X_train_rand = X_train.detach().clone()
        X_test_rand = X_test.detach().clone()
        if n_remove > 0:
            rand_indices = torch.tensor(np.random.choice(range(input_dim), n_remove, replace=False), dtype=torch.long).to(device)
            X_train_rand.index_fill_(1, rand_indices, 0)
            X_test_rand.index_fill_(1, rand_indices, 0)
            
        train_loader_rand = DataLoader(TensorDataset(X_train_rand, y_train), batch_size=256, shuffle=True)
        test_loader_rand = DataLoader(TensorDataset(X_test_rand, y_test), batch_size=256, shuffle=False)
        
        model_rand = SimpleMLP(input_dim=input_dim).to(device)
        train_model(model_rand, train_loader_rand, epochs=5, lr=0.001)
        acc_rand = evaluate_model(model_rand, test_loader_rand)
        roar_results["Random"].append(acc_rand)
        
        print(f"Frac: {frac:.1f} | DIxAI Acc: {acc_dixai:.4f} | Random Acc: {acc_rand:.4f}")

    # 5. Plot and Save
    plt.figure(figsize=(10, 6))
    plt.plot(fractions, roar_results["DIxAI"], 'o-', color='#e74c3c', linewidth=3, markersize=10, label='DIxAI (Important Removed)')
    plt.plot(fractions, roar_results["Random"], 's--', color='#34495e', linewidth=2, markersize=8, label='Random Removal')
    
    plt.xlabel("Fraction of Features Removed", fontsize=14, fontweight='bold')
    plt.ylabel("Retrained Model Accuracy", fontsize=14, fontweight='bold')
    plt.title("ROAR Faithfulness Benchmark (Adult Income)", fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.ylim(0.5, 1.0) # Accuracy range for Adult
    
    # Aesthetics
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    
    os.makedirs('experiments/results', exist_ok=True)
    save_path = 'experiments/results/roar_curve.png'
    plt.savefig(save_path, dpi=300)
    print(f"\nFinal ROAR plot saved to {save_path}")
    
    # Copy to latex figures
    os.makedirs('latex/figures', exist_ok=True)
    latex_path = 'latex/figures/roar_curve.png'
    import shutil
    shutil.copy(save_path, latex_path)
    print(f"Copy saved to {latex_path}")

if __name__ == "__main__":
    run_roar_adult()
