
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.datasets import load_iris
from dixai.explainer import DecisionInformationExplainer
from dixai.visualization import plot_beeswarm, plot_lambda_trajectory
from experiments.benchmark.datasets import get_tabular_dataset
from experiments.benchmark.models import SimpleMLP, train_model
from torch.utils.data import DataLoader, TensorDataset

def run_advanced_visualizations():
    print("--- Running Advanced DIXAI Visualizations (Iris) ---")
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
    train_model(model, train_loader, epochs=10)
    model.eval()
    
    explainer = DecisionInformationExplainer(model, lambda_fidelity=10.0, task='classification')
    
    # 3. Generate Beeswarm Plot
    print("Generating Master Beeswarm plot...")
    explanations = []
    for i in range(len(X_test)):
        x_s = X_test[i]
        exp = explainer.explain(x_s, steps=300, lr=0.1)
        explanations.append(exp)
        
    beeswarm_path = "experiments/results/visualizations/iris_beeswarm.png"
    # Highlight the first sample in the test set
    plot_beeswarm(explanations, X_test.numpy(), feature_names=feature_names, 
                  highlight_idx=0,
                  title="Master DIXAI Diagnostic Beeswarm (Iris)", 
                  save_path=beeswarm_path, show=False)
    
    # 4. Generate Lambda Trajectory
    print("Generating Lambda trajectory plot...")
    sample_idx = 0
    x_sample = X_test[sample_idx]
    
    lambdas = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    sweep_results = []
    for l in lambdas:
        explainer_l = DecisionInformationExplainer(model, lambda_fidelity=l)
        exp_l = explainer_l.explain(x_sample, steps=500, lr=0.1)
        sweep_results.append({'lambda': l, 'explanation': exp_l})
        
    trajectory_path = "experiments/results/visualizations/iris_lambda_trajectory.png"
    plot_lambda_trajectory(sweep_results, feature_names=feature_names,
                           title=f"DIXAI Feature Retention Trajectory (Iris Sample {sample_idx})",
                           save_path=trajectory_path, show=False)

if __name__ == "__main__":
    run_advanced_visualizations()
