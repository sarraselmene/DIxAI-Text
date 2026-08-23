import torch
import numpy as np
from experiments.benchmark.metrics import compute_insertion_deletion_curves, auc_metric
from experiments.benchmark.datasets import get_tabular_dataset
from experiments.benchmark.models import SimpleMLP, train_model
from experiments.benchmark.baselines import SHAPBaseline, GradientSaliency
from torch.utils.data import DataLoader, TensorDataset

def get_auc_summary():
    dataset_name = 'adult'
    X_train, X_test, y_train, y_test = get_tabular_dataset(dataset_name)
    model = SimpleMLP(input_dim=X_train.shape[1], output_dim=2)
    train_model(model, DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True), epochs=5)
    
    x = X_test[0]
    target = torch.argmax(model(x.unsqueeze(0))).item()
    
    baselines = {
        'SHAP': SHAPBaseline(model),
        'Saliency': GradientSaliency(model)
    }
    
    for name, bl in baselines.items():
        if name == 'SHAP':
            attr = bl.explain(x, background=X_train[:50], target=target)
        else:
            attr = bl.explain(x, target=target)
        
        attr = attr.squeeze()
        r_ins, v_ins = compute_insertion_deletion_curves(model, x, attr, steps=10, mode='insertion')
        r_del, v_del = compute_insertion_deletion_curves(model, x, attr, steps=10, mode='deletion')
        print(f"{name}: Ins={auc_metric(np.linspace(0,1,10), v_ins):.3f}, Del={auc_metric(np.linspace(0,1,10), v_del):.3f}")

if __name__ == "__main__":
    get_auc_summary()
