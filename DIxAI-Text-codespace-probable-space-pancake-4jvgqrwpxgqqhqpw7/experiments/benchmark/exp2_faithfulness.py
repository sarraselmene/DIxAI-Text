import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from dixai.explainer import DecisionInformationExplainer
from experiments.benchmark.metrics import compute_insertion_deletion_curves, auc_metric
from experiments.benchmark.datasets import get_tabular_dataset
from experiments.benchmark.models import SimpleMLP, train_model
from experiments.benchmark.baselines import SHAPBaseline, GradientSaliency
from torch.utils.data import DataLoader, TensorDataset

def run_faithfulness_experiment(dataset_name='iris'):
    print(f"--- Running Faithfulness Experiment (Insertion/Deletion) on {dataset_name} ---")
    os.makedirs('experiments/results', exist_ok=True)
    
    # 1. Load Data
    X_train, X_test, y_train, y_test = get_tabular_dataset(dataset_name)
    input_dim = X_train.shape[1]
    num_classes = len(torch.unique(y_train))
    
    # 2. Train Model
    model = SimpleMLP(input_dim=input_dim, output_dim=num_classes)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
    train_model(model, train_loader, epochs=10)
    
    # 3. Setup Baselines
    baselines = {
        'DIXAI': None, # To be created per lambda
        'SHAP': SHAPBaseline(model),
        'Saliency': GradientSaliency(model)
    }
    
    X_explain = X_test[:10]
    steps = 10
    
    results = {}
    
    for name in baselines.keys():
        print(f"Evaluating {name}...")
        ins_curves = []
        del_curves = []
        
        for i in range(len(X_explain)):
            x = X_explain[i]
            target_class = torch.argmax(model(x.unsqueeze(0)), dim=-1).item()
            
            if name == 'DIXAI':
                # Use a mid-range lambda
                explainer = DecisionInformationExplainer(model, lambda_fidelity=5.0)
                explanation = explainer.explain(x, steps=500)
                attr = explanation.mask_probs
            elif name == 'SHAP':
                attr = baselines[name].explain(x, background=X_train[:50], target=target_class)
            else:
                attr = baselines[name].explain(x, target=target_class)
            
            # Ensure attr has no singleton batch dim if returned as (1, Dim)
            attr = attr.squeeze()
            
            # Compute curves
            r_ins, v_ins = compute_insertion_deletion_curves(model, x, attr, steps=steps, mode='insertion')
            r_del, v_del = compute_insertion_deletion_curves(model, x, attr, steps=steps, mode='deletion')
            
            ins_curves.append(v_ins)
            del_curves.append(v_del)
            
        results[name] = {
            'ins': np.mean(ins_curves, axis=0),
            'del': np.mean(del_curves, axis=0),
            'auc_ins': auc_metric(np.linspace(0, 1, steps), np.mean(ins_curves, axis=0)),
            'auc_del': auc_metric(np.linspace(0, 1, steps), np.mean(del_curves, axis=0))
        }
    
    print("\n--- Faithfulness Results Summary ---")
    for name, data in results.items():
        print(f"{name}: Ins AUC={data['auc_ins']:.3f}, Del AUC={data['auc_del']:.3f}")
    print("------------------------------------\n")

    # 4. Plotting
    ratios = np.linspace(0, 1, steps)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Insertion
    for name, data in results.items():
        axes[0].plot(ratios, data['ins'], label=f"{name} (AUC={data['auc_ins']:.3f})")
    axes[0].set_title('Insertion Curve (Higher AUC is Better)')
    axes[0].set_xlabel('Fraction of Features Added')
    axes[0].set_ylabel('Target Probability')
    axes[0].legend()
    axes[0].grid(True)
    
    # Deletion
    for name, data in results.items():
        axes[1].plot(ratios, data['del'], label=f"{name} (AUC={data['auc_del']:.3f})")
    axes[1].set_title('Deletion Curve (Lower AUC is Better)')
    axes[1].set_xlabel('Fraction of Features Removed')
    axes[1].set_ylabel('Target Probability')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    save_path = f'experiments/results/exp2_faithfulness_{dataset_name}.png'
    plt.savefig(save_path)
    print(f"Result saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    run_faithfulness_experiment('adult')
