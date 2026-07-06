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

    def _contiguity_penalty(self, token_mask: torch.Tensor) -> torch.Tensor:
        """
        Pénalité de contiguïté séquentielle pour le texte (Verrou 2).

        Remplace la TV loss 2D (images) par une pénalité 1D (séquences).

        Principe :
            La TV loss 2D pénalise |M[i,j] - M[i+1,j]| + |M[i,j] - M[i,j+1]|
            pour encourager des régions spatiales continues.

            Pour le texte, on veut des SPANS contigus :
                "vraiment excellent" ← bien
                "vraiment ... excellent" ← pénalisé

            Formule : Σ_t |m_t - m_{t-1}|
            Plus cette somme est grande → plus le masque est "épars/fragmenté"
            On veut la minimiser.

        Args:
            token_mask : (B, T) — importance par token

        Returns:
            pénalité scalaire
        """
        if token_mask.dim() == 1:
            token_mask = token_mask.unsqueeze(0)

        # Différence entre tokens consécutifs
        # (B, T-1)
        diff = token_mask[:, 1:] - token_mask[:, :-1]

        # Valeur absolue et moyenne
        penalty = diff.abs().mean()

        return penalty
    def forward(
        self,
        mask: torch.Tensor,
        y_pred_original: torch.Tensor,
        y_pred_masked: torch.Tensor,
        x: Optional[torch.Tensor] = None,
        mine_estimator: Optional[nn.Module] = None,
        modality: str = "image"   # ← NOUVEAU PARAMÈTRE
    ):
        # 1. Information Term: I(X; Z)
        if mine_estimator is not None and x is not None:
            x_flat = x.view(x.size(0), -1)
            z_flat = (x * mask).view(x.size(0), -1)
            info_loss = mine_estimator.compute_mi(x_flat, z_flat)
        else:
            info_loss = mask.mean()

        # 2. Fidelity Term
        if self.task == 'classification':
            fidelity_loss = self.fidelity_criterion(y_pred_masked, y_pred_original)
        else:
            fidelity_loss = self.fidelity_criterion(y_pred_masked, y_pred_original)

        # 3. Régularisation spatiale — TV pour images, contiguïté pour texte
        if modality == "text":
            tv_loss = self._contiguity_penalty(mask)
        else:
            tv_loss = self._total_variation(mask, x)

        total_loss = info_loss + self.lambda_fidelity * fidelity_loss + self.lambda_tv * tv_loss

        return total_loss, info_loss, fidelity_loss

