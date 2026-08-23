
import torch
from dixai.masks import GumbelSoftmaxMask

def test_broadcast():
    # Simulate Explainer Logic
    x_sample = torch.randn(1, 28, 28)
    batch_x = x_sample.unsqueeze(0) # (1, 1, 28, 28)
    print(f"batch_x shape: {batch_x.shape}")
    
    input_shape = x_sample.shape # (1, 28, 28)
    mask_module = GumbelSoftmaxMask(input_shape)
    
    print(f"logits shape: {mask_module.mask_logits.shape}") # Should be (1, 28, 28)
    
    masked_x, mask = mask_module(batch_x, training=True)
    
    print(f"Output masked_x shape: {masked_x.shape}")
    print(f"Output mask shape: {mask.shape}")
    
    if masked_x.dim() == 5:
        print("FAILURE: masked_x is 5D!")
    elif masked_x.dim() == 4:
        print("SUCCESS: masked_x is 4D.")
    else:
        print(f"Unknown dim: {masked_x.dim()}")

if __name__ == "__main__":
    test_broadcast()
