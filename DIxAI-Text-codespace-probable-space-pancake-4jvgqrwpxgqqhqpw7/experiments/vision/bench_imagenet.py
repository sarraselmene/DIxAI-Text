import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import requests
from io import BytesIO
from dixai import DecisionInformationExplainer

# --- 1. Load ImageNet Model & Data ---
def get_imagenet_model():
    print("  Loading pre-trained ResNet-50 (ImageNet-1K)...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.eval()
    return model

import torchvision

def get_sample_images():
    """
    Load real ImageNet samples from local directory.
    """
    sample_dir = "data/imagenet_samples"
    if not os.path.exists(sample_dir):
        print(f"  Warning: {sample_dir} not found. Running download_imagenet_samples.py...")
        import subprocess
        subprocess.run(["python", "download_imagenet_samples.py"])
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    samples = []
    if os.path.exists(sample_dir):
        # Only load 3 representative samples for qualitative plot
        target_samples = ["sample_1.jpg", "sample_2.jpg", "sample_3.jpg"]
        for filename in target_samples:
            path = os.path.join(sample_dir, filename)
            if os.path.exists(path):
                img = Image.open(path).convert('RGB')
                samples.append((transform(img), filename))
            else:
                print(f"  Warning: {filename} not found in {sample_dir}")
                
    if not samples:
        print("  Using CIFAR-10 Upsampled Fallback (224x224)...")
        # Fallback logic remains as a safety net
        import torchvision
        cifar_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        samples.append((cifar_dataset[10][0], "airplane_up"))
        samples.append((cifar_dataset[6][0], "car_up"))
            
    return samples


# --- 2. Explanation Pipeline ---
def run_benchmark():
    print("Initializing ImageNet (224x224) Scale Benchmark...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    samples = get_sample_images()
    if not samples:
        print("Error: No samples available. Check internet connection.")
        return
        
    model = get_imagenet_model().to(device)
    
    # Standard ImageNet normalization for denormalizing
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    
    def denormalize(t):
        return (t * std + mean).clamp(0, 1)

    explainer = DecisionInformationExplainer(
        model=model,
        lambda_fidelity=50.0, 
        lambda_tv=0.3,
        task='classification',
        device=device
    )
    
    results_dir = "experiments/results/visualizations"
    os.makedirs(results_dir, exist_ok=True)
    
    fig, axes = plt.subplots(len(samples), 3, figsize=(18, 6 * len(samples)))
    # plt.suptitle("DIxAI: ImageNet Scale Multi-Modal Interpretability (224x224)", fontsize=32, fontweight='black', y=1.02)
    
    for i, (img_tensor, name) in enumerate(samples):
        # ... (rest of the loop remains same)
        img_input = img_tensor.unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            output = model(img_input)
            pred_idx = output.argmax(1).item()
        
        print(f"Explaining {name} (Predicted Index: {pred_idx})...")
        
        # Explain with Multi-scale: 224 / 7 = 32x32 mask grid
        baseline = torch.zeros_like(img_input)
        explanation = explainer.explain(
            img_tensor,
            baseline=baseline,
            steps=1000,
            downsample_factor=7,
            use_spatial_prior=True,
            anneal=True,
            verbose=False
        )
        
        # Viz
        orig_img = denormalize(img_input).detach().squeeze().cpu().permute(1, 2, 0).numpy()
        mask = explanation.mask_probs.squeeze().cpu().numpy()
        if mask.ndim == 3: mask = mask.mean(axis=0)
        
        masked_tensor = img_tensor.to(device) * explanation.mask.to(device) + (1 - explanation.mask.to(device)) * baseline.squeeze().to(device)
        masked_img = denormalize(masked_tensor).squeeze().cpu().permute(1, 2, 0).numpy()

        # Col 0: Input
        axes[i, 0].imshow(orig_img)
        axes[i, 0].set_title(f"INPUT", fontsize=30, fontweight='black')
        axes[i, 0].axis('off')
        
        # Col 1: DIxAI Mask
        axes[i, 1].imshow(orig_img, alpha=0.3)
        im = axes[i, 1].imshow(mask, cmap='RdBu_r', alpha=0.8, vmin=0, vmax=1)
        axes[i, 1].set_title("DIxAI BOTTLE-NECK", fontsize=30, fontweight='black')
        axes[i, 1].axis('off')
        
        # Col 2: Sufficient Bits
        axes[i, 2].imshow(masked_img)
        axes[i, 2].set_title(f"SUFFICIENT BITS", fontsize=30, fontweight='black')
        axes[i, 2].axis('off')
        
        print(f"  Fidelity: {explanation.fidelity_score:.4f}, Sparsity: {explanation.info_score:.4f}")

    plt.tight_layout()
    save_path = os.path.join(results_dir, "imagenet_qualitative.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
    print(f"ImageNet qualitative results saved to {save_path}")

if __name__ == "__main__":
    run_benchmark()
