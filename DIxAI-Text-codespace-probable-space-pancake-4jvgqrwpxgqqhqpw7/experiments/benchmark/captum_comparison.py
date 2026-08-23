import sys
import os
# Ensure project root is in path
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
from dixai import DecisionInformationExplainer
from experiments.benchmark.baselines import IntegratedGradientsBaseline, DeepLIFTBaseline, RISEBaseline
import time

def compare_baselines_imagenet(limit=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Baseline Comparison (DIxAI vs Captum vs RISE) on {device}...")
    
    # 1. Load Model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device)
    
    # Disable inplace ReLUs for Captum (IG/DeepLIFT) compatibility
    def fix_relu_aggressive(model):
        for name, child in model.named_children():
            if isinstance(child, nn.ReLU):
                setattr(model, name, nn.ReLU(inplace=False))
            else:
                fix_relu_aggressive(child)
    fix_relu_aggressive(model)
    
    model.eval()
    
    # 2. Setup Data
    sample_dir = "data/imagenet_samples"
    if not os.path.exists(sample_dir):
        print("Error: ImageNet samples not found.")
        return
        
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Initialize Explainers
    dixai_explainer = DecisionInformationExplainer(model=model, lambda_fidelity=50.0, device=device)
    ig_baseline = IntegratedGradientsBaseline(model)
    dl_baseline = DeepLIFTBaseline(model)
    rise_baseline = RISEBaseline(model, n_masks=100) # Fast version
    
    results = []
    sample_files = sorted([f for f in os.listdir(sample_dir) if f.endswith(".jpg")])[:limit]
    
    for filename in sample_files:
        print(f"  Comparing on {filename}...")
        path = os.path.join(sample_dir, filename)
        img = Image.open(path).convert('RGB')
        img_tensor = transform(img).to(device)
        print(f"    Input shape: {img_tensor.shape}")
        
        # 1. DIxAI
        try:
            start = time.time()
            d_exp = dixai_explainer.explain(img_tensor, steps=500, verbose=False)
            d_time = time.time() - start
            print(f"    DIxAI Success: {d_time:.2f}s")
        except Exception as e:
            print(f"    DIxAI Error: {e}")
            d_time, d_exp = 0, None
        
        # 2. IG
        try:
            start = time.time()
            # IG requires gradients
            ig_attr = ig_baseline.explain(img_tensor)
            ig_time = time.time() - start
            print(f"    IG Success: {ig_time:.2f}s")
        except Exception as e:
            print(f"    IG Error: {type(e).__name__}: {e}")
            ig_time = 0
        
        # 3. DeepLIFT
        try:
            start = time.time()
            # DeepLIFT requires gradients
            dl_attr = dl_baseline.explain(img_tensor)
            dl_time = time.time() - start
            print(f"    DeepLIFT Success: {dl_time:.2f}s")
        except Exception as e:
            print(f"    DeepLIFT Error: {type(e).__name__}: {e}")
            dl_time = 0
        
        # 4. RISE
        try:
            start = time.time()
            rise_attr = rise_baseline.explain(img_tensor)
            rise_time = time.time() - start
            print(f"    RISE Success: {rise_time:.2f}s")
        except Exception as e:
            print(f"    RISE Error: {e}")
            rise_time = 0
        
        results.append({
            "filename": filename,
            "dixai_time": d_time,
            "ig_time": ig_time,
            "dl_time": dl_time,
            "rise_time": rise_time,
            "dixai_fidelity": d_exp.fidelity_score,
            "dixai_sparsity": d_exp.info_score
        })

    df = pd.DataFrame(results)
    output_path = "experiments/results/baseline_comparison_stats.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Baseline comparison results saved to {output_path}")
    print(df.mean(numeric_only=True))

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run DIxAI vs Baselines on ImageNet')
    parser.add_argument('--limit', type=int, default=10, help='Number of samples to evaluate')
    args = parser.parse_args()
    
    compare_baselines_imagenet(limit=args.limit)
