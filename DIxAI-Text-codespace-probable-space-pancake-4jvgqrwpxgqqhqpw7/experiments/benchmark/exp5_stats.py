import torch
import numpy as np
import scipy.stats as stats
import os
from dixai.explainer import DecisionInformationExplainer
from experiments.benchmark.metrics import decision_fidelity, sparsity
from experiments.benchmark.datasets import get_tabular_dataset
from experiments.benchmark.models import SimpleMLP, train_model
from experiments.benchmark.baselines import SHAPBaseline
from torch.utils.data import DataLoader, TensorDataset

def compute_ci(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def run_statistical_experiment(dataset_name='adult'):
    print(f"--- Running Advanced Statistical Significance Experiment on {dataset_name} ---")
    
    # 1. Load Data
    X_train, X_test, y_train, y_test = get_tabular_dataset(dataset_name)
    input_dim = X_train.shape[1]
    num_classes = len(torch.unique(y_train))
    
    # 2. Train Model
    model = SimpleMLP(input_dim=input_dim, output_dim=num_classes)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    train_model(model, train_loader, epochs=8)
    
    # 3. Setup
    n_samples = 100 # Increased for robustness
    X_explain = X_test[:n_samples]
    
    dix_f, dix_s = [], []
    sha_f, sha_s = [], []
    
    explainer = DecisionInformationExplainer(model, lambda_fidelity=20.0) # Using standard lambda
    shap_bl = SHAPBaseline(model)
    
    print(f"Collecting data for {n_samples} samples...")
    for i in range(n_samples):
        x = X_explain[i]
        with torch.no_grad():
            target = torch.argmax(model(x.unsqueeze(0)), dim=-1).item()
        
        # DIxAI
        exp = explainer.explain(x, steps=500)
        dix_f.append(decision_fidelity(model, x, exp.mask))
        dix_s.append(sparsity(exp.mask))
        
        # SHAP (matched sparsity)
        attr = shap_bl.explain(x, background=X_train[:100], target=target)
        k = int(max(1, round((1 - dix_s[-1]) * len(attr))))
        if k >= len(attr):
            mask_s = torch.ones_like(attr)
        else:
            # Better thresholding: use indices to avoid tie-breaking issues
            top_indices = torch.topk(attr, k).indices
            mask_s = torch.zeros_like(attr)
            mask_s[top_indices] = 1.0
            
        sha_f.append(decision_fidelity(model, x, mask_s))
        sha_s.append(sparsity(mask_s))
        if (i+1) % 20 == 0:
            print(f"  Processed {i+1}/{n_samples}...")
        
    # 4. Robust Statistical Analysis
    # Wilcoxon signed-rank test (paired samples, non-parametric)
    # This is more robust than t-test for fidelity (which is often binary)
    stat_f, p_f = stats.wilcoxon(dix_f, sha_f, alternative='greater') if np.sum(np.abs(np.array(dix_f)-np.array(sha_f))) > 0 else (0, 1.0)
    stat_s, p_s = stats.wilcoxon(dix_s, sha_s) if np.sum(np.abs(np.array(dix_s)-np.array(sha_s))) > 0 else (0, 1.0)
    
    # Confidence Intervals
    dix_f_m, dix_f_low, dix_f_high = compute_ci(dix_f)
    sha_f_m, sha_f_low, sha_f_high = compute_ci(sha_f)
    
    print("\n--- Statistical Results ---")
    print(f"DIxAI Fidelity: {dix_f_m:.4f} [95% CI: {dix_f_low:.4f}, {dix_f_high:.4f}]")
    print(f"SHAP Fidelity:  {sha_f_m:.4f} [95% CI: {sha_f_low:.4f}, {sha_f_high:.4f}]")
    print(f"Wilcoxon Significance (Fidelity): p={p_f:.5f}")
    
    # 5. Save Results
    results_path = 'experiments/results/exp5_stats_robust.txt'
    with open(results_path, 'w') as f:
        f.write("Robust Statistical Significance Report (N=100)\n")
        f.write("=============================================\n\n")
        f.write(f"DIxAI Fidelity: {dix_f_m:.4f} (95% CI: {dix_f_low:.4f}-{dix_f_high:.4f})\n")
        f.write(f"SHAP Fidelity:  {sha_f_m:.4f} (95% CI: {sha_f_low:.4f}-{sha_f_high:.4f})\n")
        f.write(f"Wilcoxon p-value (Fidelity > SHAP): {p_f:.5e}\n")
        f.write(f"DIxAI Sparsity Mean: {np.mean(dix_s):.4f}\n")
        f.write(f"SHAP Sparsity Mean:  {np.mean(sha_s):.4f}\n")

    print(f"\nRobust results saved to {results_path}")

if __name__ == "__main__":
    run_statistical_experiment('adult')
