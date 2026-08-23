
import torch
import torch.nn as nn
import torch.optim as optim

class MINE(nn.Module):
    """
    Mutual Information Neural Estimator (MINE).
    Implementation based on 'MINE: Mutual Information Neural Estimation' (Belghazi et al., 2018).
    Estimates I(X; Z) = E[T(x, z)] - log(E[exp(T(x, z'))]) where z' is from the marginal P(Z).
    """
    def __init__(self, x_dim: int, z_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.T = nn.Sequential(
            nn.Linear(x_dim + z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x, z):
        return self.T(torch.cat([x, z], dim=-1))

    def compute_mi(self, x, z):
        """
        Computes the MINE lower bound for MI.
        Args:
            x: Input batch [B, x_dim]
            z: Explanation batch [B, z_dim]
        """
        # Joint samples
        t_joint = self(x, z)
        
        # Marginal samples (shuffle z)
        z_marg = z[torch.randperm(z.size(0))]
        t_marg = self(x, z_marg)
        
        # Lower bound using Donsker-Varadhan representation
        # log-sum-exp trick for numerical stability
        mi_bound = torch.mean(t_joint) - torch.log(torch.mean(torch.exp(t_marg)) + 1e-8)
        return mi_bound

class MINETrainer:
    def __init__(self, x_dim, z_dim, lr=1e-4, device='cpu'):
        self.device = device
        self.model = MINE(x_dim, z_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
    def step(self, x, z):
        self.optimizer.zero_grad()
        mi = self.model.compute_mi(x, z)
        loss = -mi # Maximize MI
        loss.backward()
        self.optimizer.step()
        return mi.item()
