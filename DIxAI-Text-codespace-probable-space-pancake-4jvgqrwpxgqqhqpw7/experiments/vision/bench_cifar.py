import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from dixai import DecisionInformationExplainer

# --- 1. Load CIFAR-10 and Model ---
def get_cifar_model():
    """
    Load a pre-trained ResNet-20 for CIFAR-10 using torch.hub.
    This ensures we get real semantic interpretations.
    """
    print("  Loading pre-trained ResNet-20 for CIFAR-10...")
    try:
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True, verbose=False)
    except:
        print("  Warning: Could not load hub model, falling back to random ResNet (Fidelity may be low).")
        model = torchvision.models.resnet18(num_classes=10)
    model.eval()
    return model

def get_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    return dataset

# --- 2. Explanation Pipeline ---
def run_benchmark():
    print("Initializing CIFAR-10 Generalization Benchmark...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = get_data()
    model = get_cifar_model().to(device)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Pick a few diverse samples
    indices = [10, 25, 42, 100] # Random diverse indices
    
    explainer = DecisionInformationExplainer(
        model=model,
        lambda_fidelity=100.0, # Increased for stricter decision preservation
        lambda_tv=0.5, # Increased for smoother blobs
        task='classification',
        device=device
    )
    
    results_dir = "experiments/results/visualizations"
    os.makedirs(results_dir, exist_ok=True)
    
    # Define a high-contrast zebra banding
    zebra_color = 'gray'
    zebra_alpha = 0.05

    # 4 Columns: Input, Saliency, DIxAI Masterpiece, Sufficient Bits
    fig, axes = plt.subplots(len(indices), 4, figsize=(24, 6 * len(indices)))
    # suptitle removed for manuscript integration
    
    for i, idx in enumerate(indices):
        img_tensor, label = dataset[idx]
        img_input = img_tensor.unsqueeze(0).to(device)
        img_input.requires_grad = True # For Saliency
        
        # Get prediction
        output = model(img_input)
        pred_idx = output.argmax(1).item()
        
        # 1. Saliency Baseline (Vanilla Gradients)
        score = output[0, pred_idx]
        score.backward()
        saliency = torch.abs(img_input.grad.data).squeeze().cpu().permute(1, 2, 0).numpy()
        saliency = saliency.mean(axis=2)
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        
        print(f"Explaining Sample {idx} (True: {classes[label]}, Pred: {classes[pred_idx]})...")
        
        # 2. DIxAI Explanation (Spatial Prior + Annealing)
        baseline = torch.zeros_like(img_input) 
        explanation = explainer.explain(
            img_tensor, 
            baseline=baseline, 
            steps=1000, 
            init_logits=-1.0, 
            downsample_factor=2,
            use_spatial_prior=True, # New Phase 2 feature
            anneal=True,            # New Phase 2 feature
            verbose=False
        )
        
        # Process for visualization
        def denormalize(t):
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).to(device)
            return (t * std + mean).clamp(0, 1)

        orig_img = denormalize(img_input).detach().squeeze().cpu().permute(1, 2, 0).numpy()
        mask = explanation.mask_probs.squeeze().cpu().numpy()
        if mask.ndim == 3:
            mask = mask.mean(axis=0)
        
        # Col 0: Original
        axes[i, 0].imshow(orig_img)
        axes[i, 0].set_title(f"INPUT: {classes[pred_idx].upper()}", fontsize=28, fontweight='black', pad=15)
        axes[i, 0].axis('off')
        
        # Col 1: Saliency (Comparison)
        axes[i, 1].imshow(saliency, cmap='hot')
        axes[i, 1].set_title("SALIENCY (GRADIENT)", fontsize=28, fontweight='black', pad=15)
        axes[i, 1].axis('off')
        
        # Col 2: DIxAI Masterpiece
        axes[i, 2].imshow(orig_img, alpha=0.3)
        im = axes[i, 2].imshow(mask, cmap='RdBu_r', alpha=0.8, vmin=0, vmax=1)
        axes[i, 2].set_title("DIxAI BOTTLE-NECK", fontsize=28, fontweight='black', pad=15)
        axes[i, 2].axis('off')
        
        # Col 3: Sufficient Representation
        with torch.no_grad():
            masked_tensor = img_tensor.to(device) * explanation.mask.to(device) + (1 - explanation.mask.to(device)) * baseline.squeeze().to(device)
            masked_img = denormalize(masked_tensor).squeeze().cpu().permute(1, 2, 0).numpy()
        
        axes[i, 3].imshow(masked_img)
        axes[i, 3].set_title("SUFFICIENT BITS", fontsize=28, fontweight='black', pad=15)
        axes[i, 3].axis('off')
        
        # Add a light zebra banding for rows
        if i % 2 == 0:
            rect = plt.Rectangle((0, i/len(indices)), 1, 1/len(indices), transform=fig.transFigure, color=zebra_color, alpha=zebra_alpha, zorder=-1)
            fig.patches.append(rect)

        print(f"  Fidelity: {explanation.fidelity_score:.4f}, Sparsity: {explanation.info_score:.4f}")

    # Premium colorbar
    cbar_ax = fig.add_axes([0.35, 0.04, 0.3, 0.015])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Information Density $P(Z|X)$', fontsize=20, fontweight='bold', labelpad=15)
    cbar.ax.tick_params(labelsize=18)

    plt.tight_layout(pad=0.2)
    save_path = os.path.join(results_dir, "cifar_qualitative.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
    print(f"CIFAR-10 qualitative results saved to {save_path}")

if __name__ == "__main__":
    run_benchmark()
