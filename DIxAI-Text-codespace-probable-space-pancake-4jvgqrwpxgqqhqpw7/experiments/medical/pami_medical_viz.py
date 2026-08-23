# pami_medical_viz.py
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path
import torchxrayvision as xrv
import cv2

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from dixai import DecisionInformationExplainer

def create_clinical_audit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Creating PAMI Clinical Audit on {device}...")
    
    # 1. Load Model
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval().to(device)
    
    # 2. Setup Data
    sample_dir = r"d:\Haythem\Temp\AAAA-New Research\decision_information_xai\data\medical_samples"
    img_path = os.path.join(sample_dir, "medical_000.jpg")
    if not os.path.exists(img_path):
        print(f"Sample {img_path} not found.")
        return
        
    img = Image.open(img_path).convert('RGB')
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 2048) - 1024)
    ])
    
    img_tensor = transform(img).to(device).float()
    
    # Explainer
    explainer = DecisionInformationExplainer(
        model=model,
        lambda_fidelity=120.0, # High fidelity for medical
        lambda_tv=1.5,         # High TV for smooth anatomical regions
        device=device
    )
    
    # Explain target: Effusion (index 7 in torchxrayvision model.pathologies)
    target_pathology = "Effusion"
    target_idx = model.pathologies.index(target_pathology)
    
    print(f"Generating clinical explanation for {target_pathology}...")
    exp = explainer.explain(
        img_tensor,
        steps=800,
        anneal=True,
        use_spatial_prior=True,
        verbose=False
    )
    
    # --- Visualization Aesthetics ---
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor='#111111')
    
    orig_np = img_tensor.cpu().permute(1, 2, 0).numpy()
    orig_np = (orig_np - orig_np.min()) / (orig_np.max() - orig_np.min())
    orig_np = orig_np.squeeze()
    
    mask = exp.mask.squeeze().cpu().numpy()
    
    # 1. Original
    axes[0].imshow(orig_np, cmap='bone')
    axes[0].set_title("Clinical Input (X-Ray)", color='white', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # 2. Decision Bottleneck (Overlay)
    axes[1].imshow(orig_np, cmap='bone')
    # Use cv2 to create a beautiful heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    axes[1].imshow(heatmap, alpha=0.4)
    axes[1].contour(mask, levels=[0.5], colors='cyan', linewidths=1.5)
    axes[1].set_title(f"Aunomy Bottleneck ($Z$)", color='white', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # 3. Sufficient Anatomy
    sufficient = orig_np * mask
    axes[2].imshow(sufficient, cmap='bone')
    axes[2].set_title("Sufficient Clinical Data", color='white', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    # 4. Binary Evidence
    binary = (mask > 0.5).astype(float)
    axes[3].imshow(binary, cmap='gray')
    axes[3].set_title(f"Evidence Mask ($F={exp.fidelity_score:.2f}$)", color='white', fontsize=14, fontweight='bold')
    axes[3].axis('off')
    
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/medical_clinical_audit.png', dpi=300, bbox_inches='tight')
    print("Clinical Audit Figure Saved to figures/medical_clinical_audit.png")

if __name__ == "__main__":
    create_clinical_audit()
