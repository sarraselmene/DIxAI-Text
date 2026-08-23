
import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import DTD
from torch.utils.data import DataLoader
from dixai import DecisionInformationExplainer
import matplotlib.pyplot as plt
import numpy as np
import sys

def benchmark_dtd(limit=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting DTD Texture Benchmark on {device}...", flush=True)
    
    # 1. Load Data
    data_dir = "data/dtd"
    os.makedirs(data_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    try:
        print("Downloading/Loading DTD dataset...", flush=True)
        dataset = DTD(root=data_dir, split='test', download=True, transform=transform)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    except Exception as e:
        print(f"Error loading DTD: {e}", flush=True)
        return

    # 2. Load Model (Texture-sensitive)
    # ResNet50 is standard, though VGG is often used for texture. Sticking to ResNet for consistency.
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device)
    model.eval()
    
    explainer = DecisionInformationExplainer(model=model, lambda_fidelity=50.0, device=device)
    
    results = []
    
    print(f"Benchmarking on {limit} samples...", flush=True)
    for i, (img, label) in enumerate(dataloader):
        if i >= limit: break
        
        img = img.to(device)
        
        # Explain
        # For textures, spatial structure is key. 
        # We might expect DIxAI to highlight repeating patterns.
        exp = explainer.explain(img[0], steps=300, verbose=False, seed=42)
        
        results.append({
            "index": i,
            "fidelity": exp.fidelity_score,
            "sparsity": exp.info_score,
            "prediction": label.item()
        })
        
        # Save visualization for first 5
        if i < 5:
            # Visualize
            mask = exp.mask.cpu().squeeze()
            if mask.dim() == 3 and mask.shape[0] == 3:
                mask = mask.mean(0)
            print(f"DEBUG: i={i}, mask shape: {mask.shape}", flush=True)
            print(f"DEBUG: i={i}, img shape: {img.shape}", flush=True)
            
            # img is (1, 3, 224, 224), so img[0] is (3, 224, 224)
            # We need (224, 224, 3) for matplotlib
            
            # Defensive coding
            img_tensor = img[0].cpu()
            if img_tensor.shape[0] == 3: # (3, H, W)
                 img_disp = img_tensor.permute(1, 2, 0)
            else:
                 img_disp = img_tensor
                 
            print(f"DEBUG: i={i}, img_disp shape before unnorm: {img_disp.shape}", flush=True)
            
            # Unnormalize
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])
            img_disp = img_disp * std + mean
            
            print(f"DEBUG: i={i}, img_disp shape after unnorm: {img_disp.shape}", flush=True)
            
            img_disp = torch.clamp(img_disp, 0, 1)
            img_disp_np = img_disp.numpy()
            
            fig, ax = plt.subplots(1, 2, figsize=(8, 4))
            ax[0].imshow(img_disp_np)
            ax[0].set_title(f"Texture Input (Class {label.item()})")
            ax[0].axis('off')
            
            ax[1].imshow(img_disp_np)
            ax[1].imshow(mask, cmap='jet', alpha=0.5)
            ax[1].set_title(f"DIxAI (F={exp.fidelity_score:.2f})")
            ax[1].axis('off')
            
            plt.savefig(f"experiments/results/dtd_sample_{i}.png")
            plt.close()
            print(f"Saved dtd_sample_{i}.png", flush=True)
            
    # Stats
    fidelities = [r['fidelity'] for r in results]
    sparsities = [r['sparsity'] for r in results]
    
    print("-" * 30, flush=True)
    print(f"DTD Benchmark Results (N={limit})", flush=True)
    print(f"Avg Fidelity: {np.mean(fidelities):.4f}", flush=True)
    print(f"Avg Sparsity: {np.mean(sparsities):.4f}", flush=True)
    print("-" * 30, flush=True)

if __name__ == "__main__":
    benchmark_dtd()
