"""
Cross-Architecture Validation (ViT)
Verifies DIxAI on Vision Transformer (ViT-B/16) with baseline comparisons.
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

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "experiments/benchmark"))

from dixai import DecisionInformationExplainer
from dixai.baselines import RISEExplainer, IntegratedGradientsExplainer

# DEVICE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_vit_model():
    print("  Loading pre-trained ViT-B/16...")
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    model.eval()
    return model.to(DEVICE)

def get_samples(limit=10):
    sample_dir = "data/imagenet_samples"
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    samples = []
    if os.path.exists(sample_dir):
        files = [f for f in os.listdir(sample_dir) if f.endswith((".jpg", ".JPEG", ".png"))]
        for filename in sorted(files)[:limit]:
            path = os.path.join(sample_dir, filename)
            img = Image.open(path).convert('RGB')
            samples.append((transform(img), filename))
    return samples

def run_dixai_on_vit(model, samples, explainer, results_dir):
    """Run DIxAI explainer on ViT samples."""
    results = []
    
    for i, (img_tensor, name) in enumerate(samples):
        img_input = img_tensor.unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = model(img_input)
            target = output.argmax(1).item()
            
        print(f"  DIxAI: Explaining {name} (Target: {target})...")
        
        start_time = time.time()
        explanation = explainer.explain(
            img_tensor,
            baseline=torch.zeros_like(img_input),
            steps=300,
            downsample_factor=7,
            use_spatial_prior=True,
            anneal=True,
            verbose=False
        )
        latency = time.time() - start_time
        
        # Visualization for first sample
        if i == 0:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            img_np = img_tensor.cpu().permute(1, 2, 0).numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            plt.imshow(img_np)
            plt.title("Input")
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            mask = explanation.mask_probs.squeeze().cpu().numpy()
            if mask.ndim == 3: mask = mask.mean(axis=0)
            plt.imshow(mask, cmap='RdBu_r')
            plt.title("DIxAI Mask (ViT)")
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            masked_img = img_tensor.to(DEVICE) * explanation.mask.to(DEVICE)
            masked_np = masked_img.squeeze().cpu().permute(1, 2, 0).numpy()
            masked_np = (masked_np - masked_np.min()) / (masked_np.max() - masked_np.min())
            plt.imshow(masked_np)
            plt.title(f"Sufficient Bits (F={explanation.fidelity_score:.2f})")
            plt.axis('off')
            
            plt.savefig(f"{results_dir}/vit_dixai_explanation.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"    Saved to {results_dir}/vit_dixai_explanation.png")

        results.append({
            "sample": name,
            "method": "DIxAI",
            "fidelity": explanation.fidelity_score,
            "sparsity": explanation.info_score,
            "latency": latency
        })
    
    return results

def run_rise_on_vit(model, samples, results_dir):
    """Run RISE explainer on ViT samples."""
    results = []
    explainer = RISEExplainer(model=model, n_masks=500, device=DEVICE)
    
    for i, (img_tensor, name) in enumerate(samples):
        img_input = img_tensor.unsqueeze(0).to(DEVICE)
        
        print(f"  RISE: Explaining {name}...")
        start_time = time.time()
        
        try:
            mask = explainer.explain(img_input)
            latency = time.time() - start_time
            
            mask_binary = (mask > 0.5).float()
            sparsity = 1.0 - mask_binary.mean().item()
            
            # Visualization for first sample
            if i == 0:
                plt.figure(figsize=(6, 4))
                plt.imshow(mask.cpu().numpy(), cmap='hot')
                plt.colorbar()
                plt.title("RISE Saliency (ViT)")
                plt.axis('off')
                plt.savefig(f"{results_dir}/vit_rise_explanation.png", dpi=150, bbox_inches='tight')
                plt.close()
            
            results.append({
                "sample": name,
                "method": "RISE",
                "fidelity": None,  # RISE doesn't compute fidelity directly
                "sparsity": sparsity,
                "latency": latency
            })
        except Exception as e:
            print(f"    Error: {e}")
    
    return results

def run_ig_on_vit(model, samples, results_dir):
    """Run Integrated Gradients on ViT samples."""
    results = []
    
    try:
        explainer = IntegratedGradientsExplainer(model=model, device=DEVICE, n_steps=50)
    except ImportError as e:
        print(f"  Skipping IG: {e}")
        return results
    
    for i, (img_tensor, name) in enumerate(samples):
        img_input = img_tensor.unsqueeze(0).to(DEVICE)
        
        print(f"  IG: Explaining {name}...")
        start_time = time.time()
        
        try:
            attr_map = explainer.explain(img_input)
            latency = time.time() - start_time
            
            attr_binary = (attr_map > 0.5).float()
            sparsity = 1.0 - attr_binary.mean().item()
            
            # Visualization for first sample
            if i == 0:
                plt.figure(figsize=(6, 4))
                plt.imshow(attr_map.cpu().numpy(), cmap='hot')
                plt.colorbar()
                plt.title("Integrated Gradients (ViT)")
                plt.axis('off')
                plt.savefig(f"{results_dir}/vit_ig_explanation.png", dpi=150, bbox_inches='tight')
                plt.close()
            
            results.append({
                "sample": name,
                "method": "IntegratedGradients",
                "fidelity": None,
                "sparsity": sparsity,
                "latency": latency
            })
        except Exception as e:
            print(f"    Error: {e}")
    
    return results

def run_vit_benchmark():
    """Run comprehensive ViT benchmark with DIxAI and baselines."""
    print("Starting ViT Cross-Architecture Validation...")
    print("=" * 60)
    
    samples = get_samples(limit=10)
    if not samples:
        print("Error: No samples found in data/imagenet_samples.")
        return

    model = get_vit_model()
    
    explainer = DecisionInformationExplainer(
        model=model,
        lambda_fidelity=50.0,
        lambda_tv=0.3,
        device=DEVICE
    )
    
    results_dir = "experiments/results/vit_validation"
    os.makedirs(results_dir, exist_ok=True)
    
    all_results = []
    
    # Run DIxAI
    print("\n[1/3] Running DIxAI on ViT...")
    dixai_results = run_dixai_on_vit(model, samples, explainer, results_dir)
    all_results.extend(dixai_results)
    
    # Run RISE
    print("\n[2/3] Running RISE on ViT...")
    rise_results = run_rise_on_vit(model, samples, results_dir)
    all_results.extend(rise_results)
    
    # Run Integrated Gradients
    print("\n[3/3] Running Integrated Gradients on ViT...")
    ig_results = run_ig_on_vit(model, samples, results_dir)
    all_results.extend(ig_results)
    
    # Save and summarize
    df = pd.DataFrame(all_results)
    df.to_csv(f"{results_dir}/vit_benchmark_results.csv", index=False)
    
    print("\n" + "=" * 60)
    print("ViT CROSS-ARCHITECTURE VALIDATION SUMMARY")
    print("=" * 60)
    summary = df.groupby('method').agg({
        'fidelity': ['mean', 'std'],
        'sparsity': ['mean', 'std'],
        'latency': 'mean'
    }).round(4)
    print(summary)
    print(f"\nResults saved to {results_dir}/vit_benchmark_results.csv")

if __name__ == "__main__":
    run_vit_benchmark()

