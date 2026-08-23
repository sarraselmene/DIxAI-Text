import torch
import time
import numpy as np
import os
from dixai.explainer import DecisionInformationExplainer
from experiments.benchmark.datasets import get_tabular_dataset
from experiments.benchmark.models import SimpleMLP, train_model
from experiments.benchmark.baselines import SHAPBaseline, GradientSaliency
from torch.utils.data import DataLoader, TensorDataset

def run_timing_experiment(dataset_name='adult'):
    print(f"--- Running Timing Experiment on {dataset_name} ---")
    
    # 1. Load Data
    X_train, X_test, y_train, y_test = get_tabular_dataset(dataset_name)
    input_dim = X_train.shape[1]
    num_classes = len(torch.unique(y_train))
    
    # 2. Train Model
    model = SimpleMLP(input_dim=input_dim, output_dim=num_classes)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    train_model(model, train_loader, epochs=5)
    
    # 3. Timing Setup
    baselines = {
        'DIXAI': None,
        'SHAP': SHAPBaseline(model),
        'Saliency': GradientSaliency(model)
    }
    
    X_explain = X_test[:50]
    results = {}
    
    for name in baselines.keys():
        print(f"Timing {name}...")
        start_time = time.time()
        
        for i in range(len(X_explain)):
            x = X_explain[i]
            target = torch.argmax(model(x.unsqueeze(0)), dim=-1).item()
            
            if name == 'DIXAI':
                explainer = DecisionInformationExplainer(model, lambda_fidelity=5.0)
                _ = explainer.explain(x, steps=500)
            elif name == 'SHAP':
                _ = baselines[name].explain(x, background=X_train[:50], target=target)
            else:
                _ = baselines[name].explain(x, target=target)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / len(X_explain)
        results[name] = avg_time
        print(f"  Avg Time for {name}: {avg_time:.4f}s")
        
    # 4. Save results to a simple text file
    with open('experiments/results/exp4_timing.txt', 'w') as f:
        f.write("Method,Avg Time (s)\n")
        for name, t in results.items():
            f.write(f"{name},{t:.4f}\n")
    print("Results saved to experiments/results/exp4_timing.txt")

if __name__ == "__main__":
    run_timing_experiment('adult')
