import torch
import torch.nn as nn
from typing import Optional


class DecisionInformationLoss(nn.Module):
    """
    Lagrangian Objective for Decision-Information Bottleneck:
    L = I(X; Z) + lambda * E[dist(f(X), f(Z))]
    """
    def __init__(self, lambda_fidelity: float = 1.0, task: str = 'classification', lambda_tv: float = 0.0):
        """
        Args:
            lambda_fidelity: Weight for the decision preservation term. 
            task: 'classification' or 'regression'
            lambda_tv: Weight for Total Variation (spatial smoothness) loss.
        """
        super().__init__()
        self.lambda_fidelity = lambda_fidelity
        self.lambda_tv = lambda_tv
        self.task = task
        
        if task == 'classification':
            self.fidelity_criterion = nn.KLDivLoss(reduction='batchmean', log_target=True)
        elif task == 'regression':
            self.fidelity_criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unknown task: {task}")

    def _total_variation(self, mask: torch.Tensor, x: Optional[torch.Tensor] = None):
        """
        Computes Total Variation loss. 
        If x is provided, uses Edge-Aware TV (EA-TV).
        """
        if mask.dim() < 2:
            return torch.tensor(0.0).to(mask.device)
        
        # Assume mask is (C, H, W) or (Batch, C, H, W)
        if mask.dim() == 4: # (B, C, H, W)
            diff_h = mask[:, :, 1:, :] - mask[:, :, :-1, :]
            diff_w = mask[:, :, :, 1:] - mask[:, :, :, :-1]
            
            if x is not None:
                # Calculate image gradients for filtering
                # x is typically (Batch, 3, H, W)
                x_grad_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean(dim=1, keepdim=True)
                x_grad_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean(dim=1, keepdim=True)
                
                # Weight TV loss inversely by image gradients
                # High image gradient -> lower weight for mask gradient
                weight_h = torch.exp(-10.0 * x_grad_h) 
                weight_w = torch.exp(-10.0 * x_grad_w)
                
                # Align dimensions
                tv = (torch.sqrt((diff_h * weight_h)[:, :, :, 1:]**2 + (diff_w * weight_w)[:, :, 1:, :]**2 + 1e-8)).mean()
            else:
                tv = torch.sqrt(diff_h[:, :, :, 1:]**2 + diff_w[:, :, 1:, :]**2 + 1e-8).mean()
                
        elif mask.dim() == 3: # (C, H, W)
            diff_h = mask[:, 1:, :] - mask[:, :-1, :]
            diff_w = mask[:, :, 1:] - mask[:, :, :-1]
            tv = torch.sqrt(diff_h[:, :, 1:]**2 + diff_w[:, 1:, :]**2 + 1e-8).mean()
        else: # (H, W)
            diff_h = mask[1:, :] - mask[:-1, :]
            diff_w = mask[:, 1:] - mask[:, :-1]
            tv = torch.sqrt(diff_h[:, 1:]**2 + diff_w[1:, :]**2 + 1e-8).mean()

        return tv


    def forward(self, mask: torch.Tensor, y_pred_original: torch.Tensor, y_pred_masked: torch.Tensor, x: Optional[torch.Tensor] = None, mine_estimator: Optional[nn.Module] = None):
        """
        Args:
            mask: The generated mask [Batch, Dim] or [Dim].
            y_pred_original: Predictions of f(X) (logits). 
            y_pred_masked: Predictions of f(Z) (logits).
            x: Optional original input for Edge-Aware TV / MINE.
            mine_estimator: Optional MINE model to compute MI instead of mask mean.
        """
        # 1. Information Term: I(X; Z)
        if mine_estimator is not None and x is not None:
            # Reshape inputs for MINE (Batch, Dim)
            x_flat = x.view(x.size(0), -1)
            # Z = X * M for MINE input (assuming B=0 for MI complexity)
            z_flat = (x * mask).view(x.size(0), -1)
            info_loss = mine_estimator.compute_mi(x_flat, z_flat)
        else:
            info_loss = mask.mean()
        
        # 2. Fidelity Term: E[dist(f(X), f(Z))]
        if self.task == 'classification':
             # ModelWrapper already returns log-probabilities
             fidelity_loss = self.fidelity_criterion(y_pred_masked, y_pred_original)
        else:
            fidelity_loss = self.fidelity_criterion(y_pred_masked, y_pred_original)
            
        # 3. Spatial Regularization (TV)
        tv_loss = self._total_variation(mask, x)
            
        # Total Lagrangian
        total_loss = info_loss + self.lambda_fidelity * fidelity_loss + self.lambda_tv * tv_loss
        
        return total_loss, info_loss, fidelity_loss

