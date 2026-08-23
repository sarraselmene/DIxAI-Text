"""
ImageNet-1K Benchmark for DIxAI TNNLS-Level Experiments

This script runs benchmarks on ImageNet-1K (subset) 
comparing DIxAI against modern explainability baselines.
"""

import os
import sys
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Subset
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.dixai import DecisionInformationExplainer
from experiments.benchmark.statistical_utils import MultiSeedRunner, generate_latex_comparison_table
from experiments.benchmark.tnnls_baselines import RISEExplainer, GradCAMPlusPlusExplainer


class ImageListDataset(torch.utils.data.Dataset):
    """Custom dataset for a flat list of images."""
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = sorted(list(self.root_dir.glob("*.jpg")))
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        from PIL import Image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0  # Dummy label

def load_imagenet_subset(data_dir: str = "./data/imagenet", n_samples: int = 5000):
    """Load a subset of ImageNet validation set."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    if not os.path.exists(data_dir):
        print(f"Warning: Data directory {data_dir} not found. Returning empty dataset.")
        return []
        
    try:
        dataset = ImageListDataset(data_dir, transform=transform)
        if len(dataset) == 0:
            return []
        indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
        subset = Subset(dataset, indices)
        return subset
    except Exception as e:
        print(f"Error loading ImageNet: {e}")
        return []


def compute_fidelity(model, x, mask, baseline=None, threshold=0.5):
    """Compute explanation fidelity."""
    device = next(model.parameters()).device
    x = x.to(device)
    
    if baseline is None:
        baseline = torch.zeros_like(x)
    
    # Binarize mask and handle dimensions
    binary_mask = (mask > threshold).float().to(device)
    
    # Squeeze redundant leading dimensions if any
    while binary_mask.dim() > x.dim():
        if binary_mask.shape[0] == 1:
            binary_mask = binary_mask.squeeze(0)
        else:
            break
            
    # Unsqueeze to match x's dimensions if needed
    while binary_mask.dim() < x.dim():
        binary_mask = binary_mask.unsqueeze(0)
        
    binary_mask = binary_mask.expand_as(x)
    
    masked_x = x * binary_mask + baseline * (1 - binary_mask)
    
    with torch.no_grad():
        orig_logits = model(x.unsqueeze(0))
        orig_pred = orig_logits.argmax(dim=-1)
        masked_logits = model(masked_x.unsqueeze(0))
        masked_pred = masked_logits.argmax(dim=-1)
    
    return (orig_pred == masked_pred).float().item()


def compute_sparsity(mask, threshold=0.5):
    """Fraction of masked pixels."""
    binary_mask = (mask > threshold).float()
    return 1 - binary_mask.mean().item()


def run_experiment(method, model, dataset, n_samples: int = 50, seed: int = 42, device: str = "cuda"):
    """Generic experiment runner for ImageNet."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    fidelities = []
    sparsities = []
    
    if len(dataset) == 0:
        return {'fidelity': 0.0, 'sparsity': 0.0}
        
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Setup explainer based on method
    if method == 'DIxAI':
        explainer = DecisionInformationExplainer(model, lambda_fidelity=50.0, device=device)
    elif method == 'RISE':
        explainer = RISEExplainer(model, n_masks=2000, device=device)
    elif method == 'GradCAM++':
        # Find last conv layer in ResNet
        target_layer = None
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                target_layer = m
        explainer = GradCAMPlusPlusExplainer(model, target_layer, device=device)
    else:
        raise ValueError(f"Unknown method {method}")

    for i, (x, y) in enumerate(loader):
        if i >= n_samples:
            break
        
        x = x.to(device)
        try:
            if method == 'DIxAI':
                explanation = explainer.explain(x, steps=500, init_logits=-2.0, seed=seed)
                mask = explanation.mask.cpu()
            else:
                mask = explainer.explain(x)
                if method == 'RISE':
                    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
                mask = mask.cpu()
            
            fidelities.append(compute_fidelity(model, x.squeeze(0), mask))
            sparsities.append(compute_sparsity(mask))
        except Exception as e:
            continue
            
    return {
        'fidelity': np.mean(fidelities) if fidelities else 0.0,
        'sparsity': np.mean(sparsities) if sparsities else 0.0,
    }


def main():
    print("=" * 60)
    print("ImageNet-1K (Large Scale) Benchmark")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load ResNet-50
    model = models.resnet50(pretrained=True).to(device).eval()
    
    # Load data
    data_dir = project_root / "data" / "imagenet_samples"
    dataset = load_imagenet_subset(str(data_dir), n_samples=500)
    
    if len(dataset) == 0:
        print("Error: No ImageNet samples found. Please run download_imagenet_samples.py first.")
        return

    runner = MultiSeedRunner(n_seeds=10)
    n_test = 5000  # Full 5K validation set
    
    print("\nRunning ImageNet Benchmarks...")
    
    results = {}
    for method in ['DIxAI', 'RISE', 'GradCAM++']:
        print(f"--- Testing {method} ---")
        results[method] = runner.run(run_experiment, method=method, model=model, 
                                    dataset=dataset, n_samples=n_test, device=device)
        print(f"{method}: {results[method]['fidelity']}")

    # LaTeX Table
    latex = generate_latex_comparison_table(
        methods=results,
        metrics=['fidelity', 'sparsity'],
        caption="ImageNet Large-Scale Evaluation ($N=5,000$ samples)",
        label="tab:imagenet_large"
    )
    
    print("\nGenerated LaTeX Table:")
    print(latex)
    
    # Save
    out_path = project_root / "experiments" / "results" / "imagenet_large_results.tex"
    with open(out_path, "w") as f:
        f.write(latex)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
