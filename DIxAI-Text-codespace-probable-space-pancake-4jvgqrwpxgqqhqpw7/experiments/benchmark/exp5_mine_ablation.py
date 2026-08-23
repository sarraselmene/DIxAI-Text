"""
MINE Ablation Study
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
from tqdm import tqdm

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from dixai import DecisionInformationExplainer
from dixai.mine import MINETrainer
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# DEVICE
DEVICE = torch.device("cpu")

def get_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    return model.to(DEVICE)

def get_sample():
    sample_dir = "data/imagenet_samples"
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # No normalization here for MINE estimation consistency across lambda
    ])
    
    if os.path.exists(sample_dir):
        files = [f for f in os.listdir(sample_dir) if f.endswith(".jpg")]
        if files:
            path = os.path.join(sample_dir, files[0])
            img = Image.open(path).convert('RGB')
            return transform(img).unsqueeze(0).to(DEVICE)
    return None

def run_mine_ablation():
    print("Starting MINE Ablation Study...")
    x = get_sample()
    if x is None:
        print("Error: No ImageNet samples found.")
        return
    
    model = get_model()
    
    # Lambdas to sweep
    lambdas = [0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    
    mi_estimates = []
    mask_densities = []
    
    results_dir = "experiments/results"
    os.makedirs(results_dir, exist_ok=True)

    # For each lambda, generate a mask and estimate MI
    for lam in tqdm(lambdas, desc="Sweeping Lambda"):
        explainer = DecisionInformationExplainer(
            model=model,
            lambda_fidelity=lam,
            device=DEVICE
        )
        
        # Generate explanation
        explanation = explainer.explain(
            x.squeeze(0),
            steps=300,
            downsample_factor=7,
            verbose=False
        )
        
        mask = explanation.mask_probs # [1, 1, 32, 32] or similar
        density = mask.mean().item()
        mask_densities.append(density)
        
        # Estimate MI using MINE
        # X is [1, 3, 224, 224], Z is [1, 3, 224, 224]
        # We flatten them for MINE
        x_flat = x.view(1, -1)
        z_flat = (x * explanation.mask).view(1, -1)
        
        # Initialize MINE for these dimensions
        x_dim = x_flat.shape[1]
        z_dim = z_flat.shape[1]
        trainer = MINETrainer(x_dim, z_dim, lr=1e-3, device=DEVICE)
        
        # Train MINE to estimate MI(X; Z)
        x_flat = x_flat.to(DEVICE)
        z_flat = z_flat.to(DEVICE)
        # (Rest of logic)
        # Actually, IB in DIxAI is per-instance.
        # To make MINE work on a single instance, we could use multiple noise realizations 
        # but DIxAI masks are deterministic given the image (after optimization).
        # A better way is to justify it theoretically, but let's try to show 
        # that higher lambda -> higher fidelity -> higher density -> higher info.
        
        # For a more robust empirical validation, we would use a small batch of images.
        # Let's use 10 images to get a batch for MINE.
        
    # Redo with a small batch
    run_batch_mine_ablation(model, lambdas, results_dir)

def run_batch_mine_ablation(model, lambdas, results_dir):
    sample_dir = "data/imagenet_samples"
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(112), # Smaller crop
        transforms.ToTensor(),
    ])
    
    files = [f for f in os.listdir(sample_dir) if f.endswith(".jpg")][:3] # 3 samples
    x_batch = []
    for f in files:
        img = Image.open(os.path.join(sample_dir, f)).convert('RGB')
        x_batch.append(transform(img))
    x_batch = torch.stack(x_batch).to(DEVICE)
    
    mi_estimates = []
    mask_densities = []
    
    for lam in tqdm(lambdas, desc="MINE Fast Sweep"):
        explainer = DecisionInformationExplainer(
            model=model,
            lambda_fidelity=lam,
            device=DEVICE
        )
        
        masks = []
        for i in range(x_batch.shape[0]):
            exp = explainer.explain(x_batch[i], steps=100, verbose=False) # 100 steps
            masks.append(exp.mask)
        
        masks = torch.stack(masks).to(DEVICE)
        z_batch = x_batch * masks
        
        density = masks.mean().item()
        mask_densities.append(density)
        
        x_flat = x_batch.view(x_batch.shape[0], -1)
        z_flat = z_batch.view(z_batch.shape[0], -1)
        
        trainer = MINETrainer(x_flat.shape[1], z_flat.shape[1], lr=1e-3, device=DEVICE)
        
        x_flat = x_flat.to(DEVICE)
        z_flat = z_flat.to(DEVICE)
        for _ in range(50): # 50 MINE steps
            mi = trainer.step(x_flat, z_flat)
            
        mi_estimates.append(mi)
        print(f"  Lambda: {lam:4.1f} | Density: {density:.4f} | Est. MI: {mi:.4f}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(mask_densities, mi_estimates, color='blue', s=100)
    plt.xlabel("Mask Density ($||M||_1$)")
    plt.ylabel("Est. MI (MINE)")
    plt.title("Empirical Justification of IB Proxy")
    plt.savefig(f"{results_dir}/mine_correlation.png")
    plt.close()

if __name__ == "__main__":
    print(f"Running Fast MINE Ablation on {DEVICE}...")
    lambdas = [1.0, 10.0, 50.0, 100.0] # Fewer lambdas
    results_dir = "experiments/results"
    os.makedirs(results_dir, exist_ok=True)
    model = get_model()
    run_batch_mine_ablation(model, lambdas, results_dir)
