import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from dixai.explainer import DecisionInformationExplainer
from experiments.benchmark.metrics import decision_fidelity, sparsity
from experiments.benchmark.datasets import get_tabular_dataset
from experiments.benchmark.models import SimpleMLP, train_model, evaluate_model
from experiments.benchmark.baselines import SHAPBaseline, LIMEBaseline, GradientSaliency, RandomBaseline
from torch.utils.data import DataLoader, TensorDataset

def run_fidelity_sparsity_experiment(dataset_name='iris'):
    print(f"--- Running Fidelity-Sparsity Experiment on {dataset_name} ---")
    os.makedirs('experiments/results', exist_ok=True)
    
    # 1. Load Data
    X_train, X_test, y_train, y_test = get_tabular_dataset(dataset_name)
    input_dim = X_train.shape[1]
    num_classes = len(torch.unique(y_train))
    
    # 2. Train Model
    model = SimpleMLP(input_dim=input_dim, output_dim=num_classes)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=16, shuffle=False)
    
    train_model(model, train_loader, epochs=10)
    acc = evaluate_model(model, test_loader)
    print(f"Model Accuracy: {acc:.4f}")
    
    # 3. Setup Baselines
    baselines = {
        'SHAP': SHAPBaseline(model),
        'Saliency': GradientSaliency(model),
        'Random': RandomBaseline(model)
    }
    
    # 4. Run DIXAI for different lambdas
    lambdas = [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]
    dixai_results = []
    
    # Use a subset of test data for explanation (e.g., 20 samples)
    X_explain = X_test[:20]
    
    print("Evaluating DIXAI...")
    for lam in lambdas:
        explainer = DecisionInformationExplainer(model, lambda_fidelity=lam)
        fidelities = []
        sparsities = []
        
        for i in range(len(X_explain)):
            x = X_explain[i]
            explanation = explainer.explain(x, steps=500)
            mask = explanation.mask
            
            # Detailed logging
            # print(f"DIXAI EVAL: i={i}, x={x.shape}, mask={mask.shape}")
            f = decision_fidelity(model, x, mask)
            s = sparsity(mask)
            
            fidelities.append(f)
            sparsities.append(s)
            
        dixai_results.append({
            'lambda': lam,
            'fidelity': np.mean(fidelities),
            'sparsity': np.mean(sparsities)
        })
        print(f"  Lambda={lam}: Fidelity={np.mean(fidelities):.4f}, Sparsity={np.mean(sparsities):.4f}")
        
    # 5. Run Baselines (thresholding to get curve)
    baseline_results = {}
    
    for name, b_model in baselines.items():
        print(f"Evaluating {name}...")
        points = []
        
        # Get attributions for all samples
        all_attrs = []
        for i in range(len(X_explain)):
            kwargs = {'target': torch.argmax(model(X_explain[i].unsqueeze(0)), dim=-1).item()}
            if name == 'SHAP':
                kwargs['background'] = X_train[:50]
            
            attr = b_model.explain(X_explain[i], **kwargs)
            all_attrs.append(attr)
            
        # Sweep thresholds to get sparsity/fidelity curve
        for t in np.linspace(0, 1, 11):
            f_list = []
            s_list = []
            for i in range(len(X_explain)):
                attr = all_attrs[i]
                k = int((1-t) * len(attr)) 
                if k == 0:
                    mask = torch.zeros_like(attr)
                elif k == len(attr):
                    mask = torch.ones_like(attr)
                else:
                    threshold = torch.topk(attr, k).values[-1]
                    mask = (attr >= threshold).float()
                
                f_list.append(decision_fidelity(model, X_explain[i], mask))
                s_list.append(sparsity(mask))
            
            points.append({
                'fidelity': np.mean(f_list),
                'sparsity': np.mean(s_list)
            })
            
        baseline_results[name] = points

    # 6. Plotting
    plt.figure(figsize=(8, 6))
    
    f_d = [r['fidelity'] for r in dixai_results]
    s_d = [r['sparsity'] for r in dixai_results]
    plt.plot(s_d, f_d, 'ro-', label='DIXAI (Ours)')
    
    for name, points in baseline_results.items():
        f_b = [p['fidelity'] for p in points]
        s_b = [p['sparsity'] for p in points]
        sorted_idx = np.argsort(s_b)
        plt.plot(np.array(s_b)[sorted_idx], np.array(f_b)[sorted_idx], '--', label=name)
        
    plt.xlabel('Sparsity (Higher=More Compressed)')
    plt.ylabel('Decision Fidelity')
    plt.title(f'Fidelity-Sparsity Pareto Frontier ({dataset_name})')
    plt.legend()
    plt.grid(True)
    
    save_path = f'experiments/results/exp1_pareto_{dataset_name}.png'
    plt.savefig(save_path)
    print(f"Result saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    run_fidelity_sparsity_experiment('iris')
