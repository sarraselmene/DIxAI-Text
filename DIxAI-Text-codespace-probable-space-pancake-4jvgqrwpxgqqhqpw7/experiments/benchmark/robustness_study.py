import sys
import os
# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), 'src'))
sys.path.append(os.getcwd())

import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import argparse
import pandas as pd
from dixai import DecisionInformationExplainer
from dixai.metrics import calculate_fidelity

def robustness_study(limit=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Robustness Study on {device}...")
    
    # Enforce Determinism
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Load Model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device)
    model.eval()
    
    explainer = DecisionInformationExplainer(model=model, lambda_fidelity=50.0, device=device)
    
    # Transformations
    noise_levels = [0.0, 0.1, 0.2]
    rotations = [0, 15, 30]
    
    results = []
    sample_dir = "data/imagenet_samples"
    sample_files = sorted([f for f in os.listdir(sample_dir) if f.endswith(".jpg")])[:limit]
    
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    for filename in sample_files:
        path = os.path.join(sample_dir, filename)
        img = Image.open(path).convert('RGB')
        
        # Base Explanation
        x = transform(img).to(device)
        base_exp = explainer.explain(x, steps=300, verbose=False, seed=42)
        base_mask = base_exp.mask
        
        # 1. Noise Robustness
        for sigma in noise_levels:
            if sigma == 0: continue
            
            noisy_img = torch.clamp(x + torch.randn_like(x) * sigma, 0, 1) # Simple noise injection
            # Note: Normalization usually happens *after* noise for raw images, 
            # but since x is already normalized, we add noise in normalized space for simplicity
            # or we should add noise to raw image. Let's add to tensor before normalization if possible.
            # For simplicity here, we add to normalized tensor.
            
            noisy_exp = explainer.explain(noisy_img, steps=300, verbose=False, seed=42)
            
            # Intersection over Union of masks
            intersection = (base_mask * noisy_exp.mask).sum()
            union = (base_mask + noisy_exp.mask).sum() - intersection
            iou = (intersection / (union + 1e-8)).item()
            
            results.append({
                "filename": filename,
                "perturbation": "noise",
                "level": sigma,
                "iou": iou,
                "fidelity": noisy_exp.fidelity_score
            })
            print(f"File {filename} Noise {sigma}: IoU={iou:.2f}")

        # 2. Rotation Robustness
        for angle in rotations:
            if angle == 0: continue
            
            # Rotate original image
            rot_img = img.rotate(angle)
            rot_x = transform(rot_img).to(device)
            
            rot_exp = explainer.explain(rot_x, steps=300, verbose=False, seed=42)
            
            # Rotate base mask to match
            # We need to rotate the *mask* to compare.
            # This is tricky because mask is tensor.
            # We can use transforms.functional.rotate on the mask tensor.
            
            base_mask_pil = transforms.ToPILImage()(base_mask.float().squeeze().cpu())
            base_mask_rot = transforms.functional.rotate(base_mask_pil, angle)
            base_mask_rot_tensor = transforms.ToTensor()(base_mask_rot).round()
            
            intersection = (base_mask_rot_tensor * rot_exp.mask).sum()
            union = (base_mask_rot_tensor + rot_exp.mask).sum() - intersection
            iou = (intersection / (union + 1e-8)).item()
            
            results.append({
                "filename": filename,
                "perturbation": "rotation",
                "level": angle,
                "iou": iou,
                "fidelity": rot_exp.fidelity_score
            })
            print(f"File {filename} Rotate {angle}: IoU={iou:.2f}")
            
    df = pd.DataFrame(results)
    df.to_csv("experiments/results/robustness_stats.csv", index=False)
    print("Robustness study completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=10)
    args = parser.parse_args()
    robustness_study(args.limit)
