import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import pandas as pd
import os

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class MINE(nn.Module):
    def __init__(self, input_dim, hidden_dim=100):
        super().__init__()
        self.T = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, y):
        # Joint distribution
        joint = torch.cat((x, y), dim=1)
        
        # Marginal distribution (shuffle y within batch)
        # To avoid "leakage", we ensure the marginal is constructed from the same set
        # but completely decorrelated.
        y_shuffle = y[torch.randperm(y.size(0))]
        marginal = torch.cat((x, y_shuffle), dim=1)
        
        t_joint = self.T(joint)
        t_marginal = self.T(marginal)
        
        # Donsker-Varadhan lower bound
        # E[T(x,y)] - log(E[exp(T(x,y_sub))])
        mi = t_joint.mean() - torch.log(torch.exp(t_marginal).mean() + 1e-10)
        return mi

def generate_correlated_gaussian(n_samples, dim, rho=0.8):
    """
    Generate X and Y such that each component X_i is correlated with Y_i with coeff rho.
    True MI = -0.5 * dim * log(1 - rho^2)
    """
    mean = np.zeros(2 * dim)
    cov = np.eye(2 * dim)
    for i in range(dim):
        cov[i, i + dim] = rho
        cov[i + dim, i] = rho
    
    data = np.random.multivariate_normal(mean, cov, n_samples)
    x = data[:, :dim]
    y = data[:, dim:]
    
    true_mi = -0.5 * dim * np.log(1 - rho**2)
    return torch.FloatTensor(x), torch.FloatTensor(y), true_mi

def train_mine(x_train, y_train, x_val, y_val, n_epochs=500):
    """
    Train MINE on one set and evaluate on another to avoid leakage.
    """
    dim = x_train.shape[1]
    mine = MINE(dim)
    optimizer = optim.Adam(mine.parameters(), lr=0.005)
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        # Train on joint samples
        mi_train = mine(x_train, y_train)
        loss = -mi_train
        loss.backward()
        optimizer.step()
        
    # Evaluate on val set to get "no-leakage" estimate
    with torch.no_grad():
        mi_est = mine(x_val, y_val).item()
    return mi_est

def run_sensitivity_sweep():
    print("Starting MI Estimator Sensitivity Sweep...")
    # Parameters
    dimensions = [2, 5, 10, 20, 50, 100]
    sample_sizes = [500, 1000, 2000, 5000]
    rho = 0.8
    results = []

    for d in dimensions:
        for n in sample_sizes:
            # Generate Train and Val separately to prevent leakage
            x_train, y_train, true_mi = generate_correlated_gaussian(n, d, rho)
            x_val, y_val, _ = generate_correlated_gaussian(n, d, rho)
            
            mi_est = train_mine(x_train, y_train, x_val, y_val)
            recovery = (mi_est / true_mi) * 100
            
            print(f"Dim: {d:3d} | N: {n:5d} | True MI: {true_mi:6.2f} | Est MI: {mi_est:6.2f} | Recovery: {recovery:5.1f}%")
            results.append({
                'Dimension': d,
                'Sample_Size': n,
                'True_MI': true_mi,
                'Est_MI': mi_est,
                'Recovery': recovery
            })

    df = pd.DataFrame(results)
    df.to_csv("experiments/results/mi_sensitivity_results.csv", index=False)
    
    # Plotting with Interpolation
    plt.figure(figsize=(10, 6))
    for n in sample_sizes:
        sub_df = df[df['Sample_Size'] == n]
        x = sub_df['Dimension'].values
        y = sub_df['Recovery'].values
        
        # Interpolate for smooth curves
        x_new = np.linspace(x.min(), x.max(), 300)
        spl = make_interp_spline(x, y, k=3)
        y_smooth = spl(x_new)
        
        plt.plot(x_new, y_smooth, label=f'N={n}')
        plt.scatter(x, y, alpha=0.5)

    plt.xlabel('Embedding Dimension (d)')
    plt.ylabel('MI Recovery (%)')
    plt.title('MINE Estimator Efficiency vs. Dimensionality (No Leakage)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.axhline(100, color='red', linestyle=':', label='Ideal')
    
    save_path = "experiments/results/mi_bias_propagation.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved diagnostic plot to {save_path}")

if __name__ == "__main__":
    if not os.path.exists("experiments/results"):
        os.makedirs("experiments/results")
    run_sensitivity_sweep()
