"""
Human Study Stimuli Generator
Generates side-by-side comparisons (Original | DIxAI | SHAP/GradCAM) for the survey.
"""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from dixai import DecisionInformationExplainer
from dixai.baselines import RISEExplainer, GradCAMPlusPlusExplainer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_imagenet_samples(limit=10):
    sample_dir = project_root / "data" / "imagenet_samples"
    if not sample_dir.exists():
        return []
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    samples = []
    files = sorted([f for f in os.listdir(sample_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    for fname in files[:limit]:
        img = Image.open(sample_dir / fname).convert("RGB")
        samples.append((transform(img), fname))
    return samples

def generate_survey_sheets():
    print("Generating Human Study Survey Sheets...")
    
    # Model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.eval().to(DEVICE)
    
    # Explainers
    dixai = DecisionInformationExplainer(model, lambda_fidelity=50.0, device=DEVICE)
    baseline = RISEExplainer(model, n_masks=500, device=DEVICE) # Proxy for SHAP
    
    # Data
    samples = get_imagenet_samples(limit=5)
    if not samples:
        print("No samples found.")
        return

    output_dir = project_root / "experiments" / "human_study" / "stimuli"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, (img_tensor, name) in enumerate(samples):
        print(f"Processing {name}...")
        img_input = img_tensor.unsqueeze(0).to(DEVICE)
        
        # 1. Prediction
        with torch.no_grad():
            output = model(img_input)
            pred_idx = output.argmax(1).item()
            # Get class name if available (skipping for compactness)
            pred_label = f"Class {pred_idx}"
            
        # 2. DIxAI
        exp_dixai = dixai.explain(
            img_tensor, baseline=torch.zeros_like(img_input),
            steps=200, verbose=False
        )
        mask_dixai = exp_dixai.mask.squeeze().detach().cpu().numpy()
        if mask_dixai.ndim == 3: mask_dixai = mask_dixai.mean(0)
        
        # 3. Baseline (RISE as SHAP proxy)
        mask_base = baseline.explain(img_input).cpu().numpy()
        
        # 4. Visualization
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        
        # Create Panel
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original
        axes[0].imshow(img_np)
        axes[0].set_title(f"Input\nPred: {pred_label}")
        axes[0].axis('off')
        
        # DIxAI
        axes[1].imshow(img_np)
        axes[1].imshow(mask_dixai, cmap='jet', alpha=0.5)
        axes[1].set_title("Ours (DIxAI)")
        axes[1].axis('off')
        
        # Baseline
        axes[2].imshow(img_np)
        axes[2].imshow(mask_base, cmap='jet', alpha=0.5)
        axes[2].set_title("Baseline (SHAP/RISE)")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"survey_q{i+1}_{name.split('.')[0]}.png")
        plt.close()
        
    print(f"Generated {len(samples)} survey sheets in {output_dir}")

if __name__ == "__main__":
    generate_survey_sheets()
