
import torch
import numpy as np

def calculate_fidelity(y_pred_original: torch.Tensor, y_pred_masked: torch.Tensor, task='classification'):
    """
    Calculates the fidelity of the explanation.
    For classification: Fraction of decisions preserved (argmax match).
    For regression: Negative MSE or R2-like score (here we use simple error metric).
    """
    with torch.no_grad():
        if task == 'classification':
            labels_orig = torch.argmax(y_pred_original, dim=-1)
            labels_masked = torch.argmax(y_pred_masked, dim=-1)
            decision_match = (labels_orig == labels_masked).float().mean().item()
            return decision_match
        else:
            # For regression, we might return -MSE or similar.
            mse = torch.mean((y_pred_original - y_pred_masked) ** 2).item()
            return -mse

def calculate_sparsity(mask: torch.Tensor):
    """
    Calculates the sparsity of the explanation (fraction of features removed).
    """
    with torch.no_grad():
        # Mask values are in [0, 1].
        # Sparsity = 1 - density
        density = mask.mean().item()
        return 1.0 - density

def calculate_continuity(mask: torch.Tensor):
    """
    Calculates the continuity (smoothness) of the explanation mask.
    Measured as 1 - normalized Total Variation (TV).
    Only applicable to spatial data (Image/Time-series).
    """
    if mask.dim() < 2:
        return 1.0 # Trivial continuity for scalar/unstructured data
        
    with torch.no_grad():
        # TV calculation
        tv_h = torch.abs(mask[..., 1:, :] - mask[..., :-1, :]).sum()
        tv_w = torch.abs(mask[..., :, 1:] - mask[..., :, :-1]).sum()
        tv = (tv_h + tv_w).item()
        
        # Normalize by max possible TV (roughly number of elements)
        max_tv = mask.numel() * 2
        return 1.0 - (tv / max_tv)

def calculate_roar_curve(model_class, train_dataset, test_dataset, expert_masks, steps=10, device='cpu'):
    """
    Calculate Remove-And-Retrain (ROAR) curve.
    Hooker et al. (2019).
    
    Args:
        model_class: Class of the model to retrain.
        train_dataset: (x, y) training data.
        test_dataset: (x, y) testing data.
        expert_masks: Masks for the training data (ordered by importance).
        steps: Number of removal levels (e.g., remove 10%, 20%... 90% features).
    """
    x_train, y_train = train_dataset
    x_test, y_test = test_dataset
    results = []
    
    # Baseline accuracy (no removal)
    results.append(1.0) # Normalized accuracy
    
    # TODO: Implement full retraining loop logic
    # This usually happens in an experiment script. 
    # Here we define the interface or helper logic.
    return np.array(results)

def calculate_insertion_deletion_curves(model, x, importance_scores, n_steps=20, baseline=None):
    """
    Calculates both Insertion and Deletion curves.
    
    Args:
        model: Black-box model wrapper.
        x: Input instance (C, H, W) or (Dim).
        importance_scores: Saliency map or mask probs.
        n_steps: Number of points on the curve.
        baseline: Reference value for masked-out features.
        
    Returns:
        insertion_scores, deletion_scores (np.arrays)
    """
    model.eval()
    if baseline is None:
        baseline = torch.zeros_like(x)
        
    # Flatten everything for easier indexing
    x_flat = x.flatten()
    baseline_flat = baseline.flatten()
    scores_flat = importance_scores.flatten()
    
    # Sort indices by importance (descending)
    indices = torch.argsort(scores_flat, descending=True)
    n_features = x_flat.numel()
    
    insertion_preds = []
    deletion_preds = []
    
    # Steps
    step_size = max(1, n_features // n_steps)
    
    with torch.no_grad():
        # Deletion: Start with full x, gradually remove important features
        for i in range(0, n_features + 1, step_size):
            # Deletion: Top i features replaced by baseline
            x_del = x_flat.clone()
            x_del[indices[:i]] = baseline_flat[indices[:i]]
            y_del = model(x_del.view_as(x).unsqueeze(0))
            deletion_preds.append(torch.softmax(y_del, dim=-1).max().item())
            
            # Insertion: Start with baseline, gradually add important features
            x_ins = baseline_flat.clone()
            x_ins[indices[:i]] = x_flat[indices[:i]]
            y_ins = model(x_ins.view_as(x).unsqueeze(0))
            insertion_preds.append(torch.softmax(y_ins, dim=-1).max().item())
            
    return np.array(insertion_preds), np.array(deletion_preds)

def calculate_auc(curve):
    """Calculates Area Under Curve using trapezoidal rule."""
    return np.trapz(curve) / len(curve)
