import torch
import torch.nn as nn
from .ddconv import HierarchicalBlock

class AmortizedExplainer(nn.Module):
    """
    Amortized Explainer Network (ExplainerNet).
    Learns to map (Input, FeatureMap) -> Decision Mask in a single forward pass.
    """
    def __init__(self, in_channels=3, feature_channels=512, hidden_dim=128):
        super().__init__()
        # Encoder for raw input
        self.input_enc = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 112x112
            nn.ReLU()
        )
        
        # Feature map adapter (assuming global average pooling or similar has been bypassed)
        self.feat_adapter = nn.Sequential(
            nn.Conv2d(feature_channels, 128, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False) # Back to 112x112 to match input_enc
        )
        
        # Hierarchical Bottleneck (DD-Conv + Attention)
        self.bottleneck = nn.Sequential(
            HierarchicalBlock(64 + 128),
            HierarchicalBlock(192)
        )
        
        # Decoder to mask
        self.decoder = nn.Sequential(
            nn.Conv2d(192, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # Back to 224x224
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, feat):
        """
        x: Raw input (N, 3, H, W)
        feat: Feature map from black-box penultimate layer (N, C, H', W')
        """
        x_enc = self.input_enc(x)
        feat_enc = self.feat_adapter(feat)
        
        combined = torch.cat([x_enc, feat_enc], dim=1)
        b_out = self.bottleneck(combined)
        
        mask = self.decoder(b_out)
        return mask
