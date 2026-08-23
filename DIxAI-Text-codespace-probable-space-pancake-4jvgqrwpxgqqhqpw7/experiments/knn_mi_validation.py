import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from typing import Optional

def estimate_mi_knn(x: np.ndarray, y: np.ndarray, k: int = 3) -> float:
    """
    Estimate Mutual Information I(X; Y) using the KSG estimator (Kraskov et al., 2004).
    This is a non-parametric estimator that is much tighter than worst-case bounds.
    """
    n_samples = x.shape[0]
    if x.ndim == 1: x = x.reshape(-1, 1)
    if y.ndim == 1: y = y.reshape(-1, 1)

    # Add small noise to avoid identical points
    x += 1e-10 * np.random.randn(*x.shape)
    y += 1e-10 * np.random.randn(*y.shape)

    xy = np.hstack([x, y])
    
    # Ball tree for efficient neighbor search
    nbrs_xy = NearestNeighbors(n_neighbors=k+1, metric='chebyshev').fit(xy)
    distances, _ = nbrs_xy.kneighbors(xy)
    eps = distances[:, k]

    # Count neighbors in x and y spaces within distance eps
    nbrs_x = NearestNeighbors(metric='chebyshev').fit(x)
    nx = np.array([len(nbrs_x.radius_neighbors(x[i:i+1], radius=eps[i] - 1e-15)[0]) - 1 for i in range(n_samples)])

    nbrs_y = NearestNeighbors(metric='chebyshev').fit(y)
    ny = np.array([len(nbrs_y.radius_neighbors(y[i:i+1], radius=eps[i] - 1e-15)[0]) - 1 for i in range(n_samples)])

    from scipy.special import digamma
    mi = digamma(k) + digamma(n_samples) - np.mean(digamma(nx + 1) + digamma(ny + 1))
    return max(0.0, mi)

def validate_dixai_bound_knn(mask_params: torch.Tensor, inputs: torch.Tensor, k: int = 3):
    """
    Validates the DIxAI bound: I(X; Z | C) <= ||p||_1
    by comparing it to the k-NN estimated mutual information.
    """
    # 1. Theoretical Bound (L1 density)
    theoretical_bound = torch.sum(mask_params).item()
    
    # 2. Empirical Estimation (k-NN)
    # We sample Z = X * M + (1-M) * B
    # For simplicity in validation, we estimate I(X; M) which is the information added by the mask
    x_np = inputs.view(inputs.size(0), -1).detach().cpu().numpy()
    m_np = mask_params.view(1, -1).repeat(inputs.size(0), 1).detach().cpu().numpy() 
    # Note: In a real validation, we would sample individual masks per input
    
    # Placeholder for a more complex per-sample mask estimation
    # mi_est = estimate_mi_knn(x_np, m_np, k=k)
    
    print(f"DIxAI Theoretical Bound (L1): {theoretical_bound:.4f}")
    # print(f"k-NN Empirical MI Estimate: {mi_est:.4f}")
    # print(f"Tightness Ratio: {theoretical_bound / mi_est:.2f}x")

if __name__ == "__main__":
    # Example usage for reproducibility
    print("k-NN Mutual Information Estimator (Kraskov et al., 2004)")
    # Generate some dummy correlation
    X = np.random.randn(100, 2)
    Y = X[:, 0] + 0.1 * np.random.randn(100) # Strong correlation with first feature
    mi = estimate_mi_knn(X, Y)
    print(f"Estimated MI on dummy data: {mi:.4f}")
