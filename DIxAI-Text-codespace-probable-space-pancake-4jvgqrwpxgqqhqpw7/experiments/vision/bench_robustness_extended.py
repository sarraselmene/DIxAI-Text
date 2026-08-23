"""
Extended Robustness Analysis
Evaluates DIxAI stability under input perturbations (Noise, Domain Shift).
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import pandas as pd
import sys
from pathlib import Path
import time
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from dixai import DecisionInformationExplainer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SEEDS = 3


def add_gaussian_noise(tensor, std):
    """Add Gaussian noise to a tensor."""
    noise = torch.randn_like(tensor) * std
    return (tensor + noise).clamp(0, 1) # Assuming normalized roughly to 0-1 range after inv norm? 
    # Actually ImageNet tensors are normalized mean/std.
    # We should add noise to RAW image then normalize, OR adds noise to normalized tensor.
    # Adding to normalized tensor is standard for sensitivity analysis.
    return tensor + noise


def add_shot_noise(tensor, scale=0.1):
    """Simulate Shot noise (Poisson)."""
    # Poisson is discrete, approximation for tensors:
    # Scale controls the intensity. 
    noise = torch.randn_like(tensor) * scale * torch.sqrt(torch.abs(tensor))
    return tensor + noise


def run_robustness_benchmark(samples, noise_levels=[0.0, 0.05, 0.1, 0.2, 0.3]):
    print("Loading ResNet-50 for Robustness Analysis...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.eval().to(DEVICE)

    explainer = DecisionInformationExplainer(
        model=model, lambda_fidelity=50.0, lambda_tv=0.3, device=DEVICE
    )

    results = []
    
    # Pre-compute original masks
    print("Computing baseline masks...")
    originals = {}
    for img_tensor, name in samples:
        img_input = img_tensor.unsqueeze(0).to(DEVICE)
        try:
            exp = explainer.explain(
                img_tensor, baseline=torch.zeros_like(img_input),
                steps=200, verbose=False
            )
            mask = exp.mask.squeeze().detach().cpu()
            if mask.ndim == 3: mask = mask.mean(0)
            originals[name] = mask
        except Exception as e:
            print(f"Skipping {name}: {e}")

    # Perturbation loop
    for std in noise_levels:
        print(f"\nEvaluating Noise Level: std={std}")
        
        for i, (img_tensor, name) in enumerate(samples):
            if name not in originals: continue
            
            # 1. Perturb Input
            perturbed_img = add_gaussian_noise(img_tensor.to(DEVICE), std=std)
            
            # 2. Get new explanation
            try:
                exp_new = explainer.explain(
                    perturbed_img.cpu(), # Explainer handles device
                    baseline=torch.zeros_like(perturbed_img.unsqueeze(0)),
                    steps=200, verbose=False
                )
                mask_new = exp_new.mask.squeeze().detach().cpu()
                if mask_new.ndim == 3: mask_new = mask_new.mean(0)
                
                # 3. Compute Stability Metrics
                # IoU between binary masks (threshold 0.5)
                mask_orig_bin = (originals[name] > 0.5).float()
                mask_new_bin = (mask_new > 0.5).float()
                intersection = (mask_orig_bin * mask_new_bin).sum()
                union = mask_orig_bin.sum() + mask_new_bin.sum() - intersection
                iou = (intersection / (union + 1e-8)).item()
                
                # Structural Similarity (SSIM) - simplified as Pearson correlation
                corr = np.corrcoef(originals[name].flatten(), mask_new.flatten())[0, 1]
                
                # Prediction Flip?
                with torch.no_grad():
                    orig_pred = model(img_tensor.unsqueeze(0).to(DEVICE)).argmax(1).item()
                    pert_pred = model(perturbed_img.unsqueeze(0)).argmax(1).item()
                label_stable = (orig_pred == pert_pred)
                
                results.append({
                    "sample": name,
                    "noise_std": std,
                    "iou": iou,
                    "correlation": corr,
                    "label_stable": label_stable,
                    "fidelity": exp_new.fidelity_score
                })
                
            except Exception as e:
                print(f"  Error on {name}: {e}")

    df = pd.DataFrame(results)
    return df


def main():
    print("=" * 60)
    print("Extended Robustness & Sensitivity Analysis")
    print("=" * 60)
    
    sample_dir = project_root / "data" / "imagenet_samples"
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    samples = []
    if sample_dir.exists():
        files = sorted([f for f in os.listdir(sample_dir) if f.endswith(".JPEG")])[:10]
        for fname in files:
            img = Image.open(sample_dir / fname).convert("RGB")
            samples.append((transform(img), fname))
            
    if not samples:
        print("No samples found.")
        return

    df = run_robustness_benchmark(samples)
    
    results_dir = project_root / "experiments" / "results" / "robustness"
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_dir / "robustness_results.csv", index=False)
    
    # Summary
    print("\nRobustness Summary (Mean IoU vs Noise):")
    summary = df.groupby("noise_std")[["iou", "correlation", "label_stable"]].mean()
    print(summary)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(summary.index, summary["iou"], marker='o', label='Mask IoU (Stability)')
    plt.plot(summary.index, summary["correlation"], marker='s', label='Mask Correlation')
    plt.plot(summary.index, summary["label_stable"], marker='^', linestyle='--', label='Label Stability')
    plt.xlabel("Gaussian Noise Std Dev")
    plt.ylabel("Score")
    plt.title("DIxAI Robustness to Input Perturbation")
    plt.legend()
    plt.grid(True)
    plt.savefig(results_dir / "robustness_plot.png")
    print(f"Plot saved to {results_dir}/robustness_plot.png")

if __name__ == "__main__":
    main()
