import torch
import numpy as np
from typing import Callable, Union, List, Tuple
import torch.nn.functional as F

def decision_fidelity(model: Callable, x: torch.Tensor, mask: torch.Tensor) -> float:
    """
    Computes hard decision fidelity: 1 if argmax f(x*m) == argmax f(x) else 0.
    """
    model.eval()
    with torch.no_grad():
        # Ensure 2D (Batch, Dim)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        
        # Ensure mask matches x batch size
        if mask.shape[0] != x.shape[0] and mask.shape[0] == 1:
            mask = mask.expand(x.shape[0], *([-1] * (mask.dim() - 1)))
            
        if x.shape != mask.shape:
            # Try to broadcast or fix
            if x.shape[1:] != mask.shape[1:]:
                 raise ValueError(f"CRITICAL SHAPE MISMATCH: x={x.shape}, mask={mask.shape}. "
                                f"Model expects input dim {x.shape[1:]} but mask has {mask.shape[1:]}")
            
        y_orig = model(x)
        masked_input = x * mask
        y_masked = model(masked_input)
        
        pred_orig = torch.argmax(y_orig, dim=-1)
        pred_masked = torch.argmax(y_masked, dim=-1)
        
        fidelity = (pred_orig == pred_masked).float().mean()
    return fidelity.item()

def soft_fidelity(model: Callable, x: torch.Tensor, mask: torch.Tensor) -> float:
    """
    Computes soft fidelity based on KL divergence.
    """
    model.eval()
    with torch.no_grad():
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
            
        if mask.shape[0] != x.shape[0] and mask.shape[0] == 1:
            mask = mask.expand(x.shape[0], *([-1] * (mask.dim() - 1)))

        log_p = F.log_softmax(model(x * mask), dim=-1)
        q = F.softmax(model(x), dim=-1)
        
        kl = F.kl_div(log_p, q, reduction='batchmean')
    return (-kl).item()

def sparsity(mask: torch.Tensor) -> float:
    """
    Computes mask sparsity: 1 - ratio of active features.
    """
    with torch.no_grad():
        # Handle binary or soft masks
        # If mask is prob [0, 1], we can use 0.5 threshold
        binary_mask = (mask > 0.5).float()
        total_elements = binary_mask.numel()
        active_elements = binary_mask.sum()
        
        sparsity_ratio = 1.0 - (active_elements / total_elements)
    return sparsity_ratio.item()

def compute_insertion_deletion_curves(
    model: Callable, 
    x: torch.Tensor, 
    importance_scores: torch.Tensor,
    steps: int = 10,
    mode: str = 'insertion'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes curves for insertion or deletion metrics.
    - Insertion: Start from empty/baseline, add features by importance.
    - Deletion: Start from full input, remove features by importance.
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Ensure x is on same device as model
    x = x.to(device)
    
    # For images, we need to handle channel dimension properly
    # x is [C, H, W] or [H, W] for tabular
    original_shape = x.shape
    x_flat = x.flatten()
    
    # Handle importance scores shape
    if importance_scores.dim() == 2 and x.dim() == 3:
        # Importance is [H, W], expand to [C, H, W] for proper flattening alignment
        importance_scores = importance_scores.unsqueeze(0).expand(x.shape[0], -1, -1)
    
    importance_scores = importance_scores.to(device)
    scores_flat = importance_scores.flatten()
    
    # Sort indices by importance
    indices = torch.argsort(scores_flat, descending=True)
    n = len(indices)
    
    # Define baseline (e.g., zero)
    baseline = torch.zeros_like(x_flat)
    
    accuracies = []
    ratios = np.linspace(0, 1, steps)
    
    with torch.no_grad():
        y_orig = model(x.unsqueeze(0))
        target_class = torch.argmax(y_orig, dim=-1)
        
        for ratio in ratios:
            k = int(ratio * n)
            current_x = baseline.clone() if mode == 'insertion' else x_flat.clone()
            
            if k > 0:  # Avoid empty indices
                if mode == 'insertion':
                    # Add top k
                    top_k_idx = indices[:k]
                    current_x[top_k_idx] = x_flat[top_k_idx]
                else:
                    # Remove top k (set to baseline)
                    top_k_idx = indices[:k]
                    current_x[top_k_idx] = baseline[top_k_idx]
                
            # Predict - reshape back to original shape
            input_x = current_x.view(original_shape).unsqueeze(0)
            y_curr = model(input_x)
            
            # Probability of target class
            prob = F.softmax(y_curr, dim=-1)[0, target_class].item()
            accuracies.append(prob)
            
    return ratios, np.array(accuracies)


def auc_metric(ratios: np.ndarray, values: np.ndarray) -> float:
    """Computes Area Under Curve using trapezoidal rule."""
    return np.trapz(values, ratios)


# ===========================================================================
# Convenience Functions for Baseline Comparison
# ===========================================================================

def insertion_auc(model: Callable, x: torch.Tensor, importance_scores: Union[torch.Tensor, np.ndarray], 
                  target: int = None, steps: int = 20) -> float:
    """
    Compute Insertion AUC: progressively add features by importance.
    Higher AUC = better explanation (important features recover prediction faster).
    """
    if isinstance(importance_scores, np.ndarray):
        importance_scores = torch.from_numpy(importance_scores).float()
    
    if x.dim() == 3:  # Image [C, H, W]
        # Flatten importance to match x
        if importance_scores.dim() == 2:  # [H, W]
            importance_scores = importance_scores.unsqueeze(0).expand(x.shape[0], -1, -1)
    
    ratios, values = compute_insertion_deletion_curves(model, x, importance_scores, steps=steps, mode='insertion')
    return auc_metric(ratios, values)


def deletion_auc(model: Callable, x: torch.Tensor, importance_scores: Union[torch.Tensor, np.ndarray],
                 target: int = None, steps: int = 20) -> float:
    """
    Compute Deletion AUC: progressively remove features by importance.
    Lower AUC = better explanation (important features cause faster prediction drop).
    """
    if isinstance(importance_scores, np.ndarray):
        importance_scores = torch.from_numpy(importance_scores).float()
    
    if x.dim() == 3:  # Image [C, H, W]
        if importance_scores.dim() == 2:
            importance_scores = importance_scores.unsqueeze(0).expand(x.shape[0], -1, -1)
    
    ratios, values = compute_insertion_deletion_curves(model, x, importance_scores, steps=steps, mode='deletion')
    return auc_metric(ratios, values)
