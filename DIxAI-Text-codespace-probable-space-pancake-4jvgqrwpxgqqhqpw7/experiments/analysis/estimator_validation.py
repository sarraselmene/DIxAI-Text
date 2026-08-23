import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def true_mi_gaussian(dim, rho):
    """
    True Mutual Information for two dim-dimensional Gaussian vectors X and Y
    with component-wise correlation rho.
    I(X;Y) = sum I(X_i; Y_i) = -0.5 * dim * ln(1 - rho^2)
    """
    return -0.5 * dim * np.log(1 - rho**2 + 1e-10)

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
        # x: [B, D], y: [B, D]
        batch_size = x.size(0)
        
        # Joint distribution
        joint = torch.cat((x, y), dim=1)
        
        # Marginal distribution (shuffle y)
        y_shuffle = y[torch.randperm(batch_size)]
        marginal = torch.cat((x, y_shuffle), dim=1)
        
        t_joint = self.T(joint)
        t_marginal = self.T(marginal)
        
        # Donsker-Varadhan lower bound
        # E[T(x,y)] - log(E[exp(T(x,y_sub))])
        mi = t_joint.mean() - torch.log(torch.exp(t_marginal).mean() + 1e-10)
        return mi

def train_mine(dim, rho, steps=2000, batch_size=256, lr=1e-3, device='cpu'):
    model = MINE(dim, hidden_dim=100).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    mi_estimates = []
    
    # Theoretical MI
    true_mi = true_mi_gaussian(dim, rho)
    
    for i in range(steps):
        # Generate data: y = rho*x + sqrt(1-rho^2)*eps
        x = torch.randn(batch_size, dim).to(device)
        eps = torch.randn(batch_size, dim).to(device)
        y = rho * x + np.sqrt(1 - rho**2) * eps
        
        optimizer.zero_grad()
        mi = model(x, y)
        loss = -mi
        loss.backward()
        optimizer.step()
        
        # Moving average for stability
        with torch.no_grad():
            curr_mi = mi.item()
            if i > 100:
                # bias correction in DV is implicit, but let's just track value
                pass
            mi_estimates.append(curr_mi)
            
    # Return average of last 100 steps
    final_est = np.mean(mi_estimates[-100:])
    return final_est, true_mi, mi_estimates

def run_validation():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running MI Validation on {device}...")
    
    dims = [10, 50, 100] # Dimensions to test
    rhos = [0.5, 0.9]    # Correlations to test '
    
    results = {}
    
    plt.figure(figsize=(10, 6))
    
    for d in dims:
        est_curve = []
        true_curve = []
        for r in rhos:
            print(f"Testing Dim={d}, Rho={r}...")
            est, true_val, curve = train_mine(d, r, steps=1000, device=device)
            print(f"  -> True MI: {true_val:.4f}, Est MI: {est:.4f}")
            est_curve.append(est)
            true_curve.append(true_val)
            
        plt.scatter(true_curve, est_curve, label=f'Dim={d}')
        
    # Diagonal
    max_val = max(max(true_curve), max(est_curve)) * 1.1
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Ideal')
    
    plt.xlabel('True Mutual Information (Analytical)')
    plt.ylabel('Estimated MI (MINE)')
    plt.title('MINE Estimator Validation on Gaussian Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs('experiments/results', exist_ok=True)
    plt.savefig('experiments/results/mi_validation_plot.png')
    print("Validation plot saved to experiments/results/mi_validation_plot.png")

if __name__ == "__main__":
    run_validation()
