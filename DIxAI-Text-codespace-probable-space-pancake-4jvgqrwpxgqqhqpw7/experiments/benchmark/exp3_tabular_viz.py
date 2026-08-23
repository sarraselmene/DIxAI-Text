
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.datasets import load_iris
from dixai.explainer import DecisionInformationExplainer
from dixai.visualization import plot_feature_importance, plot_global_importance
from experiments.benchmark.datasets import get_tabular_dataset
from experiments.benchmark.models import SimpleMLP, train_model
from torch.utils.data import DataLoader, TensorDataset

def run_tabular_visualizations():
    print("--- Running Tabular Visualizations (Iris) ---")
    os.makedirs('experiments/results/visualizations', exist_ok=True)
    
    # 1. Load Data
    iris = load_iris()
    feature_names = iris.feature_names
    X_train, X_test, y_train, y_test = get_tabular_dataset('iris')
    input_dim = X_train.shape[1]
    num_classes = len(torch.unique(y_train))
    
    # 2. Train Model
    model = SimpleMLP(input_dim=input_dim, output_dim=num_classes)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
    train_model(model, train_loader, epochs=20)
    model.eval()
    
    explainer = DecisionInformationExplainer(model, lambda_fidelity=10.0, task='classification')
    
    # 3. Local Explanation
    sample_idx = 0
    x_sample = X_test[sample_idx]
    explanation = explainer.explain(x_sample, steps=500, lr=0.1)
    
    local_path = "experiments/results/visualizations/iris_local.png"
    plot_feature_importance(explanation, feature_names=feature_names, 
                            title=f"DIXAI Local Explanation (Iris Sample {sample_idx})", 
                            save_path=local_path, show=False)
    
    # 4. Global Importance (Aggregated)
    print("Generating global importance...")
    explanations = []
    for i in range(len(X_test)):
        x_s = X_test[i]
        exp = explainer.explain(x_s, steps=300, lr=0.1)
        explanations.append(exp)
        
    global_path = "experiments/results/visualizations/iris_global.png"
    plot_global_importance(explanations, feature_names=feature_names, 
                           title="DIXAI Global Feature Importance (Iris Dataset)", 
                           save_path=global_path, show=False)

if __name__ == "__main__":
    run_tabular_visualizations()
