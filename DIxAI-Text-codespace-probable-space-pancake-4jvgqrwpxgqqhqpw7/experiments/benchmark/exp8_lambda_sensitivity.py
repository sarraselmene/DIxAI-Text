import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from dixai.explainer import DecisionInformationExplainer
from experiments.benchmark.metrics import decision_fidelity, sparsity
from experiments.benchmark.datasets import get_tabular_dataset
from experiments.benchmark.models import SimpleMLP, train_model

def run_lambda_sensitivity_experiment(dataset_name='adult'):
    print(f"--- Running Lambda Sensitivity Experiment on {dataset_name} ---")
    os.makedirs('experiments/results', exist_ok=True)
    
    # 1. Load Data
    X_train, X_test, y_train, y_test = get_tabular_dataset(dataset_name)
    input_dim = X_train.shape[1]
    num_classes = 2 # Adult is binary
    
    # 2. Train Model
    model = SimpleMLP(input_dim=input_dim, output_dim=num_classes)
    train_model(model, DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True), epochs=10)
    
    # 3. Parameter Sweep
    lambdas = np.logspace(-1, 2, 12).tolist() # [0.1, ..., 100] with 12 points
    X_explain = X_test[:50] # Increased for Adult stability
    
    results = []
    
    for lam in lambdas:
        print(f"Testing Lambda={lam:.2f}...")
        explainer = DecisionInformationExplainer(model, lambda_fidelity=lam)
        
        f_vals = []
        s_vals = []
        
        for x in X_explain:
            exp = explainer.explain(x, steps=300) # Slightly fewer steps for speed on larger N
            f_vals.append(decision_fidelity(model, x, exp.mask))
            s_vals.append(sparsity(exp.mask))
            
        results.append({
            'lambda': lam,
            'fidelity': np.mean(f_vals),
            'sparsity': np.mean(s_vals)
        })

    # 4. Plotting
    lams = [r['lambda'] for r in results]
    fids = [r['fidelity'] for r in results]
    spars = [r['sparsity'] for r in results]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Lambda (Fidelity Weight)')
    ax1.set_ylabel('Decision Fidelity', color=color)
    ax1.semilogx(lams, fids, 'o-', color=color, label='Fidelity')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which='both', linestyle='--')
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Sparsity', color=color)
    ax2.semilogx(lams, spars, 's--', color=color, label='Sparsity')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title(f'DIxAI Sensitivity to $\lambda$ ({dataset_name.capitalize()} Dataset)')
    fig.tight_layout()
    
    save_path = f'experiments/results/exp8_lambda_sensitivity_{dataset_name}.png'
    plt.savefig(save_path)
    print(f"Result saved to {save_path}")
    plt.close()

from torch.utils.data import TensorDataset, DataLoader
if __name__ == "__main__":
    run_lambda_sensitivity_experiment('adult')
