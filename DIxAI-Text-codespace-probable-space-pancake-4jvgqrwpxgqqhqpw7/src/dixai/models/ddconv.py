import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DDConv2d(nn.Module):
    """
    Directional Derivative Convolution (DD-Conv)
    Applies finite difference filters at multiple angles (0, 45, 90, 135...)
    to capture anisotropic structural information.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, directions=4, trainable_kernels=True):
        super(DDConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.directions = directions
        
        # Base derivative kernels (e.g., Sobel-like for 0, 45, 90, 135 degrees)
        # We initialize them as fixed but allow them to be trainable if requested
        self.kernels = nn.Parameter(self._get_directional_kernels(), requires_grad=trainable_kernels)
        
        # Learnable combination weights for directional features
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels * directions, 1, 1))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)

    def _get_directional_kernels(self):
        """Generates finite difference kernels for various directions."""
        k = self.kernel_size
        center = k // 2
        kernels = []
        
        # 0 degree (Horizontal derivative)
        h_kernel = torch.zeros(1, 1, k, k)
        h_kernel[0, 0, center, center-1] = -1
        h_kernel[0, 0, center, center+1] = 1
        kernels.append(h_kernel)
        
        # 90 degree (Vertical derivative)
        v_kernel = torch.zeros(1, 1, k, k)
        v_kernel[0, 0, center-1, center] = -1
        v_kernel[0, 0, center+1, center] = 1
        kernels.append(v_kernel)
        
        # 45 degree (Diagonal derivative)
        d1_kernel = torch.zeros(1, 1, k, k)
        d1_kernel[0, 0, center-1, center-1] = -1
        d1_kernel[0, 0, center+1, center+1] = 1
        kernels.append(d1_kernel)
        
        # 135 degree (Anti-diagonal derivative)
        d2_kernel = torch.zeros(1, 1, k, k)
        d2_kernel[0, 0, center-1, center+1] = -1
        d2_kernel[0, 0, center+1, center-1] = 1
        kernels.append(d2_kernel)
        
        # If more directions requested, we can rotate these (stub)
        if self.directions > 4:
            # Placeholder for interpolation-based rotation
            pass
            
        return torch.cat(kernels, 0) # (directions, 1, k, k)

    def forward(self, x):
        # x: (N, C, H, W)
        batch_size, channels, h, w = x.size()
        
        # Expand kernels for all input channels
        # kernels shape: (directions, 1, k, k) -> (C*directions, 1, k, k)
        # We use groups=channels to apply directional kernels per channel
        expanded_kernels = self.kernels.repeat(channels, 1, 1, 1).to(x.device)
        
        # Grouped convolution: each channel gets its directional derivatives
        directional_features = F.conv2d(x, expanded_kernels, padding=self.kernel_size//2, groups=channels)
        # directional_features: (N, C*directions, H, W)
        
        # Combine directional features using a 1x1 conv
        out = F.conv2d(directional_features, self.weight, self.bias)
        return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)

class HierarchicalBlock(nn.Module):
    def __init__(self, channels, nhead=4, dropout=0.1):
        super(HierarchicalBlock, self).__init__()
        self.ddconv = DDConv2d(channels, channels)
        self.norm1 = nn.BatchNorm2d(channels)
        self.attention = SpatialAttention()
        
        self.mhsa = nn.MultiheadAttention(channels, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        
        self.linear = nn.Linear(channels, channels)
        self.norm3 = nn.LayerNorm(channels)
        
    def forward(self, x):
        # 1. DD-Conv + Spatial Attention
        res = x
        x = self.ddconv(x)
        x = self.norm1(x)
        x = self.attention(x)
        x = x + res
        
        # 2. Flatten for MHSA
        N, C, H, W = x.size()
        x_flat = x.view(N, C, H*W).permute(0, 2, 1) # (N, L, C)
        
        # 3. MHSA
        res_flat = x_flat
        x_flat, _ = self.mhsa(x_flat, x_flat, x_flat)
        x_flat = self.norm2(x_flat + res_flat)
        
        # 4. Linear Transform
        res_flat = x_flat
        x_flat = self.linear(x_flat)
        x_flat = self.norm3(x_flat + res_flat)
        
        # 5. Restore Shape
        x = x_flat.permute(0, 2, 1).view(N, C, H, W)
        return x
