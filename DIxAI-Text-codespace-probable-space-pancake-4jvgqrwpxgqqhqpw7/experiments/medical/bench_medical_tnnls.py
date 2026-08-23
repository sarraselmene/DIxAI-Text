"""
Standardized Medical Imaging Benchmark for TNNLS (CheXpert Subset)
Comparing DIxAI against RISE and GradCAM++ with multi-seed rigor.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.dixai import DecisionInformationExplainer
from experiments.benchmark.statistical_utils import MultiSeedRunner, generate_latex_comparison_table
from experiments.benchmark.tnnls_baselines import run_baseline
from src.dixai.metrics import calculate_fidelity, calculate_sparsity

class MedicalDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.image_paths = list(self.data_dir.glob("*.jpg")) + list(self.data_dir.glob("*.png"))
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0 # Dummy label

def run_experiment(method, model, dataloader, n_samples=50, seed=42, device="cuda"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    explainer = None
    if method == "DIxAI":
        # Increased lambda_fidelity to 50.0 for better decision preservation
        explainer = DecisionInformationExplainer(model, lambda_fidelity=50.0, device=device)
        
    # Define normalized black baseline (Medical images have black backgrounds)
    # Using ImageNet normalization stats as placeholders if specific ones aren't available, but here we match the script's transform
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device).view(1, 3, 1, 1)
    norm_black = (torch.zeros((1, 3, 1, 1)).to(device) - mean) / std
    fidelities = []
    sparsities = []
    
    count = 0
    for images, _ in dataloader:
        if count >= n_samples: break
        
        for i in range(images.size(0)):
            if count >= n_samples: break
            x = images[i:i+1].to(device)
            
            try:
                if method == "DIxAI":
                    # Medical images often benefit from higher spatial priors and proper baseline
                    exp = explainer.explain(x, steps=300, downsample_factor=4, baseline=norm_black, verbose=False)
                    mask = exp.mask
                    fidelity = exp.fidelity_score
                    sparsity = exp.info_score
                else:
                    mask = run_baseline(method, model, x)
                    # Ensure mask has (1, 1, H, W) for broadcasting
                    if mask.dim() == 2:
                        mask = mask.unsqueeze(0).unsqueeze(0)
                    elif mask.dim() == 3:
                        mask = mask.unsqueeze(0)
                        
                    # Apply mask with normalized black baseline
                    with torch.no_grad():
                        y_orig = model(x)
                        y_masked = model(x * mask + (1 - mask) * norm_black)
                    fidelity = calculate_fidelity(y_orig, y_masked)
                    sparsity = calculate_sparsity(mask)
                    
                fidelities.append(fidelity)
                sparsities.append(sparsity)
                count += 1
            except Exception as e:
                print(f"Error on sample {count} with {method}: {e}")
                continue
                
    return {
        "fidelity": np.mean(fidelities) if fidelities else 0.0,
        "sparsity": np.mean(sparsities) if sparsities else 0.0
    }

def main():
    print("="*60)
    print("Medical Imaging Benchmark for TNNLS (CheXpert Subset)")
    print("DEBUG: Running FIXED version with correct baseline fidelity calc.")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load densenet121 pre-trained (standard for CheXpert)
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT).to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    data_dir = project_root / "data" / "medical_samples"
    if not data_dir.exists():
        print(f"Data directory {data_dir} not found. Creating placeholder.")
        data_dir.mkdir(parents=True, exist_ok=True)
        
    dataset = MedicalDataset(data_dir, transform=transform)
    if len(dataset) == 0:
        print("No medical samples found! Please add CheXpert samples to data/medical_samples.")
        return
        
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    runner = MultiSeedRunner(seeds=[42, 123, 456, 789, 1024])
    methods = ["DIxAI", "RISE", "GradCAM++"]
    
    results = {}
    for method in methods:
        print(f"--- Testing {method} ---")
        results[method] = runner.run(run_experiment, method=method, model=model, 
                               dataloader=dataloader, n_samples=min(50, len(dataset)))
        print(f"{method}: Fid={results[method]['fidelity']:.4f}, Spa={results[method]['sparsity']:.4f}")
        
    latex = generate_latex_comparison_table(
        results, 
        metrics=["fidelity", "sparsity"],
        caption="Medical Imaging Benchmark Results (CheXpert Subset)",
        label="tab:medical_results"
    )
    
    print("\nGenerated LaTeX Table:")
    print(latex)
    
    out_path = project_root / "experiments" / "results" / "medical_tnnls_results.tex"
    with open(out_path, "w") as f:
        f.write(latex)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
