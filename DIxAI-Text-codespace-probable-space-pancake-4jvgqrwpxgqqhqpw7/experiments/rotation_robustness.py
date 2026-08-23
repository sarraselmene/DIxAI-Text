import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dixai import DecisionInformationExplainer

def calculate_iou(mask1, mask2):
    intersection = (mask1 * mask2).sum()
    union = (mask1 + mask2).sum() - intersection
    if union == 0: return 1.0
    return (intersection / union).item()

def evaluate_rotation_robustness():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Hardened Rotation Robustness Experiment (Mean-Fill + Annealing) on {device}...")
    
    # 1. Load Pre-trained Model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device)
    model.eval()
    
    # 2. Load and Preprocess Sample Image
    img_path = "data/imagenet_samples/sample_0000.jpg"
    if not os.path.exists(img_path):
        img_path = "data/imagenet_samples/sample_1.jpg"
    
    img = Image.open(img_path).convert('RGB')
    
    # Crop first to ensure consistent pixel content
    crop_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ])
    original_crop = crop_transform(img)
    
    # 3. Define Analysis Angles
    angles = [0, 15, 30, 45]
    results = []
    
    # Base transform (Normalize)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()
    
    # Explainer - use high lambda and annealing
    explainer = DecisionInformationExplainer(model, lambda_fidelity=200.0, device=device)
    
    # Get Baseline Mask (0 degrees)
    print("Generating baseline explanation (0 deg)...")
    base_tensor = normalize(to_tensor(original_crop)).unsqueeze(0).to(device)
    base_exp = explainer.explain(base_tensor, steps=1000, verbose=False, init_logits=0.0, anneal=True)
    base_mask = base_exp.mask 
    
    print(f"Baseline (0 deg): Fidelity={base_exp.fidelity_score:.2f}, Sparsity={base_exp.info_score:.2f}")
    
    print("\nEvaluating Rotations...")
    print(f"{'Angle':<10} | {'Fidelity':<10} | {'Sparsity':<10} | {'IoU_w_Rotated_Base':<20}")
    print("-" * 60)
    
    # Mean color for filling rotation corners (ImageNet mean)
    mean_color = (124, 116, 104) 
    
    for angle in angles:
        if angle == 0:
            print(f"{angle:<10} | {base_exp.fidelity_score:<10.2f} | {base_exp.info_score:<10.2f} | {1.0:<20.2f}")
            results.append({'Angle': 0, 'Fidelity': base_exp.fidelity_score, 'Sparsity': base_exp.info_score, 'IoU': 1.0})
            continue
            
        # 1. Rotate the CROPPED image with MEAN FILL
        rot_pil = original_crop.rotate(angle, fillcolor=mean_color)
        rot_tensor = normalize(to_tensor(rot_pil)).unsqueeze(0).to(device)
        
        # 2. Explain Rotated Image with Annealing
        exp = explainer.explain(rot_tensor, steps=1000, verbose=False, init_logits=0.0, anneal=True)
        
        # 3. Rotate Baseline Mask to compare
        base_mask_pil = transforms.ToPILImage()(base_mask.squeeze().cpu())
        base_mask_rot_pil = base_mask_pil.rotate(angle) # Masks are 1-ch, background 0 is fine
        base_mask_rot = transforms.ToTensor()(base_mask_rot_pil)
        
        # Threshold for IoU
        m1 = (exp.mask > 0.5).float()
        m2 = (base_mask_rot > 0.5).float()
        iou = calculate_iou(m1, m2)
        
        results.append({
            'Angle': angle,
            'Fidelity': exp.fidelity_score,
            'Sparsity': exp.info_score,
            'IoU': iou
        })
        print(f"{angle:<10} | {exp.fidelity_score:<10.2f} | {exp.info_score:<10.2f} | {iou:<20.2f}")

    # Save Results
    df = pd.DataFrame(results)
    os.makedirs('experiments/results', exist_ok=True)
    df.to_csv('experiments/results/rotation_robustness.csv', index=False)
    print("\nResults saved to experiments/results/rotation_robustness.csv")

if __name__ == "__main__":
    evaluate_rotation_robustness()
