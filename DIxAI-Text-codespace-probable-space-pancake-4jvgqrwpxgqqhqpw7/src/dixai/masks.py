
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class GumbelSoftmaxMask(nn.Module):
    """
    Learnable mask optimization using Gumbel-Softmax (Concrete) relaxation.
    Supports multi-scale masking by optimizing a lower-resolution mask and 
    upsampling it to the input resolution.
    """
    def __init__(self, input_shape, temperature: float = 2.0/3.0, init_logits: float = -2.0, downsample_factor: int = 1, use_spatial_prior: bool = False):
        super().__init__()
        
        # Handle both int (tabular) and tuple (image) shapes
        if isinstance(input_shape, int):
            self.full_shape = (input_shape,)
        else:
            self.full_shape = tuple(input_shape)
            
        self.downsample_factor = downsample_factor
        
        # Calculate optimization shape
        if downsample_factor > 1 and len(self.full_shape) >= 2:
            # Only for images (C, H, W) or (H, W)
            if len(self.full_shape) == 3:
                c, h, w = self.full_shape
                self.opt_shape = (1, h // downsample_factor, w // downsample_factor)
            else:
                h, w = self.full_shape
                self.opt_shape = (h // downsample_factor, w // downsample_factor)
        else:
            self.opt_shape = self.full_shape

        # Initial mask parameter
        self.mask_logits = nn.Parameter(torch.full(self.opt_shape, init_logits))
        self.temperature = temperature
        
        # Spatial Prior: Center-weighted bias for images
        self.spatial_prior = None
        if use_spatial_prior and len(self.opt_shape) >= 2:
            h_opt, w_opt = self.opt_shape[-2:]
            y = torch.linspace(-1, 1, h_opt)
            x = torch.linspace(-1, 1, w_opt)
            mesh_y, mesh_x = torch.meshgrid(y, x, indexing='ij')
            # Gaussian blob in the center
            dist = torch.sqrt(mesh_x**2 + mesh_y**2)
            prior = torch.exp(-dist**2 / 0.5) # Sigma=0.5
            self.spatial_prior = nn.Parameter(prior.unsqueeze(0) if len(self.opt_shape)==3 else prior, requires_grad=False)

    def _guided_refinement(self, mask: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Refines the upsampled mask using the input image as a guide.
        Uses a simplified Guided Filter approach to snap mask edges to image edges.
        """
        # Ensure x is 4D (B, C, H, W) and mask is 4D (B, 1, H, W)
        if x.dim() == 3: x = x.unsqueeze(0)
        if mask.dim() == 2: mask = mask.unsqueeze(0).unsqueeze(0)
        if mask.dim() == 3: mask = mask.unsqueeze(1)
        
        # 1. Grayscale guide
        guide = x.mean(dim=1, keepdim=True)
        
        # 2. Local stats (3x3 box filter)
        def box_filter(img, r=1):
            return F.avg_pool2d(img, kernel_size=2*r+1, stride=1, padding=r)
        
        r = 1
        eps = 1e-4
        
        mean_I = box_filter(guide, r)
        mean_p = box_filter(mask, r)
        mean_Ip = box_filter(guide * mask, r)
        cov_Ip = mean_Ip - mean_I * mean_p
        
        mean_II = box_filter(guide * guide, r)
        var_I = mean_II - mean_I * mean_I
        
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
        
        mean_a = box_filter(a, r)
        mean_b = box_filter(b, r)
        
        q = mean_a * guide + mean_b
        return q.squeeze(1) if q.shape[1] == 1 else q

    def forward(self, x: torch.Tensor, training: bool = True, baseline: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [Batch, C, H, W]
            training: If True, uses stochastic sampling.
            baseline: Optional reference value. 
        """
        logits = self.mask_logits
        if self.spatial_prior is not None:
            logits = logits + self.spatial_prior

        if training:
            uniform = torch.rand_like(logits)
            gumbel_noise = -torch.log(-torch.log(uniform + 1e-20) + 1e-20)
            y_soft = torch.sigmoid((logits + gumbel_noise) / self.temperature)
            mask = y_soft
        else:
            mask = (logits > 0).float()
            
        # Handle Multi-scale Upsampling
        if self.downsample_factor > 1 and len(self.full_shape) >= 2:
            # Baseline bilinear upsampling
            upsampled_mask = F.interpolate(
                mask.unsqueeze(0) if mask.dim() == 2 else mask.unsqueeze(0), 
                size=self.full_shape[-2:], 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
            
            # --- GUIDED REFINEMENT ---
            if x is not None:
                upsampled_mask = self._guided_refinement(upsampled_mask, x)
                # Keep in [0, 1]
                upsampled_mask = torch.clamp(upsampled_mask, 0, 1)
            
            mask = upsampled_mask

        # Match batch dimension
        if x.dim() == mask.dim() + 1:
            mask = mask.unsqueeze(0)
            
        if baseline is None:
            return x * mask, mask
        else:
            return x * mask + (1 - mask) * baseline, mask

    def get_mask_probs(self, x: Optional[torch.Tensor] = None):
        logits = self.mask_logits
        if self.spatial_prior is not None:
            logits = logits + self.spatial_prior
        mask_probs = torch.sigmoid(logits)
        
        if self.downsample_factor > 1 and len(self.full_shape) >= 2:
            upsampled_probs = F.interpolate(
                mask_probs.unsqueeze(0) if mask_probs.dim() == 2 else mask_probs.unsqueeze(0), 
                size=self.full_shape[-2:], 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
            
            if x is not None:
                upsampled_probs = self._guided_refinement(upsampled_probs, x)
                upsampled_probs = torch.clamp(upsampled_probs, 0, 1)
                
            return upsampled_probs
            
        return mask_probs

