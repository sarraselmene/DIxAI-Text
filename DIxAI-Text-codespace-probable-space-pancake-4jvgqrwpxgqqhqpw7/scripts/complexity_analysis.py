import torch
import torch.nn as nn
from dixai.models.ddconv import DDConv2d

def calculate_complexity(in_c, out_c, h, w):
    # Standard Conv
    std_params = in_c * out_c * 3 * 3
    std_flops = std_params * h * w
    
    # DD-Conv
    # 4 directional kernels (fixed basis) + 1x1 conv layer to combine
    # weight: (out_c, in_c * 4, 1, 1)
    dd_params = (out_c * (in_c * 4)) + out_c
    dd_flops = (in_c * 4 * h * w * 9) + (dd_params * h * w) 
    
    print(f"Layer Configuration: {in_c}->{out_c} at {h}x{w}")
    print(f"Standard Conv: Params={std_params:,}, FLOPs={std_flops:,}")
    print(f"DD-Conv:       Params={dd_params:,}, FLOPs={dd_flops:,}")
    print(f"Reduction in Params: {100 * (1 - dd_params/std_params):.2f}%")

if __name__ == "__main__":
    calculate_complexity(64, 64, 32, 32)
    print("-" * 20)
    calculate_complexity(128, 128, 16, 16)
