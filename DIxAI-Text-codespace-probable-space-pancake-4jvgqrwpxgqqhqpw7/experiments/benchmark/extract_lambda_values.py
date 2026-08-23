import torch
import numpy as np
import os
from dixai.explainer import DecisionInformationExplainer
from experiments.benchmark.metrics import decision_fidelity, sparsity
from experiments.benchmark.datasets import get_tabular_dataset
from experiments.benchmark.models import SimpleMLP, train_model
from torch.utils.data import TensorDataset, DataLoader

def get_exact_values():
    dataset_name = 'adult'
    X_train, X_test, y_train, y_test = get_tabular_dataset(dataset_name)
    input_dim = X_train.shape[1]
    model = SimpleMLP(input_dim=input_dim, output_dim=2)
    train_model(model, DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True), epochs=5)
    
    lambdas = [0.1, 0.5, 1.0, 5.0, 10.0, 100.0]
    X_explain = X_test[:20]
    
    print(f"{'Lambda':>10} | {'Fidelity':>10} | {'Sparsity':>10}")
    print("-" * 36)
    
    for lam in lambdas:
        explainer = DecisionInformationExplainer(model, lambda_fidelity=lam)
        f_vals, s_vals = [], []
        for x in X_explain:
            exp = explainer.explain(x, steps=300, verbose=False)
            f_vals.append(decision_fidelity(model, x, exp.mask))
            s_vals.append(sparsity(exp.mask))
        print(f"{lam:10.2f} | {np.mean(f_vals):10.4f} | {np.mean(s_vals):10.4f}")

if __name__ == "__main__":
    get_exact_values()
