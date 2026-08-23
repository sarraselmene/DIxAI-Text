import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import os
import sys

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.dixai import DecisionInformationExplainer

def evaluate_baseline_sensitivity():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Baseline Sensitivity Analysis on {device}...")
    
    # 1. Model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device)
    model.eval()
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 2. Samples
    samples = ['sample_0000.jpg', 'sample_0001.jpg', 'sample_0002.jpg']
    
    # Normalization constants for baselines
    norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)
    norm_std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)
    
    explainer = DecisionInformationExplainer(model, lambda_fidelity=100.0, device=device)
    results = []
    
    print(f"{'Sample':<15} | {'Baseline':<20} | {'Fidelity':<10} | {'Density':<10}")
    print("-" * 65)
    
    for s_name in samples:
        img_path = f"data/imagenet_samples/{s_name}"
        if not os.path.exists(img_path):
            print(f"Skipping {s_name} (not found)")
            continue
            
        try:
            img = Image.open(img_path).convert('RGB')
            x = transform(img).unsqueeze(0).to(device)
            
            # Check Pred
            with torch.no_grad():
                orig_logits = model(x)
                orig_class = orig_logits.argmax(dim=-1).item()
                orig_conf = torch.softmax(orig_logits, dim=-1).max().item()
                # print(f"  {s_name}: Class={orig_class}, Conf={orig_conf:.2f}")

            baselines = {
                'Black (Zero)': (torch.zeros_like(x) - norm_mean) / norm_std,
                'Dataset Mean': torch.zeros_like(x),
                'Gaussian Noise': torch.randn_like(x)
            }

            for name, base_tensor in baselines.items():
                exp = explainer.explain(x, steps=500, verbose=False, baseline=base_tensor, init_logits=0.0)
                
                results.append({
                    'Sample': s_name,
                    'Baseline': name,
                    'Fidelity': exp.fidelity_score,
                    'Density': exp.info_score
                })
                print(f"{s_name:<15} | {name:<20} | {exp.fidelity_score:<10.2f} | {exp.info_score:<10.2f}")
                
        except Exception as e:
            print(f"Error processing {s_name}: {e}")

    # Save
    if results:
        df = pd.DataFrame(results)
        df.to_csv('experiments/results/baseline_sensitivity.csv', index=False)
        print("\nResults saved to experiments/results/baseline_sensitivity.csv")
    else:
        print("\nNo results generated.")

if __name__ == "__main__":
    evaluate_baseline_sensitivity()
