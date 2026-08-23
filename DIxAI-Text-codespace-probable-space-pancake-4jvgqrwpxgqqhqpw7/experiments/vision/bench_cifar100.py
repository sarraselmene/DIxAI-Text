"""
CIFAR-100 Benchmark for DIxAI TNNLS-Level Experiments

This script runs comprehensive benchmarks on CIFAR-100 dataset
comparing DIxAI against modern explainability baselines.
"""

import os
import sys
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.dixai import DecisionInformationExplainer
from experiments.benchmark.statistical_utils import MultiSeedRunner, compute_statistics, generate_latex_comparison_table
from experiments.benchmark.tnnls_baselines import RISEExplainer, GradCAMPlusPlusExplainer


class SimpleCIFAR100Net(nn.Module):
    """Simple CNN for CIFAR-100 classification."""
    
def get_model(device: str = "cuda"):
    """Load a pretrained ResNet-18 model for CIFAR-100."""
    print("Initializing model...")
    # Use pretrained ImageNet weights as a strong feature extractor
    model = torchvision.models.resnet18(weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, 100)  # Adjust for 100 classes
    model = model.to(device)
    model.eval()
    return model


def load_cifar100(data_dir: str = "./data", n_samples: int = 500):
    """Load CIFAR-100 test set with upsampling to 224x224 for ResNet-18."""
    transform = transforms.Compose([
        transforms.Resize(224),  # Upsample to ImageNet resolution for pretrained weights
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=transform
    )
    
    # Sample subset
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    subset = Subset(dataset, indices)
    
    return subset


def compute_fidelity(model, x, mask, baseline=None, threshold=0.5):
    """Compute explanation fidelity (prediction preservation)."""
    device = next(model.parameters()).device
    x = x.to(device)
    
    if baseline is None:
        baseline = torch.zeros_like(x)
    else:
        baseline = baseline.to(device)
    
    # Binarize mask and handle dimensions
    binary_mask = (mask > threshold).float().to(device)
    
    # Squeeze redundant leading dimensions if any
    while binary_mask.dim() > x.dim():
        if binary_mask.shape[0] == 1:
            binary_mask = binary_mask.squeeze(0)
        else:
            break
            
    # Unsqueeze to match x's dimensions if needed (e.g., if mask is H,W and x is C,H,W)
    while binary_mask.dim() < x.dim():
        binary_mask = binary_mask.unsqueeze(0)
        
    binary_mask = binary_mask.expand_as(x)
    
    # Apply mask
    masked_x = x * binary_mask + baseline * (1 - binary_mask)
    
    with torch.no_grad():
        orig_pred = model(x.unsqueeze(0) if x.dim() == 3 else x).argmax(dim=-1)
        masked_pred = model(masked_x.unsqueeze(0) if masked_x.dim() == 3 else masked_x).argmax(dim=-1)
    
    return (orig_pred == masked_pred).float().item()


def compute_sparsity(mask, threshold=0.5):
    """Compute explanation sparsity (fraction of masked-out pixels)."""
    binary_mask = (mask > threshold).float()
    return 1 - binary_mask.mean().item()


def run_dixai_experiment(model, dataset, n_samples: int = 100, seed: int = 42, device: str = "cuda"):
    """Run DIxAI benchmark on CIFAR-100."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = model.to(device).eval()
    # Increased lambda for better decision preservation
    explainer = DecisionInformationExplainer(model, lambda_fidelity=50.0, device=device)
    
    # CIFAR-100 Normalized neutral baseline (approx mean)
    mean = torch.tensor([0.5071, 0.4867, 0.4408]).to(device).view(1, 3, 1, 1)
    std = torch.tensor([0.2675, 0.2565, 0.2761]).to(device).view(1, 3, 1, 1)
    norm_baseline = (torch.full((1, 3, 1, 1), 0.5).to(device) - mean) / std # Gray baseline
    
    fidelities = []
    sparsities = []
    
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    for i, (x, y) in enumerate(loader):
        if i >= n_samples:
            break
        
        x = x.to(device)
        
        try:
            # Redefine baseline inside loop to ensure scope consistency during long runs
            mean = torch.tensor([0.5071, 0.4867, 0.4408]).to(device).view(1, 3, 1, 1)
            std = torch.tensor([0.2675, 0.2565, 0.2761]).to(device).view(1, 3, 1, 1)
            norm_baseline = (torch.full((1, 3, 1, 1), 0.5).to(device) - mean) / std # Gray baseline
            
            explanation = explainer.explain(x, steps=300, lr=0.1, baseline=norm_baseline, seed=seed)
            mask = explanation.mask.cpu()
            
            fidelity = compute_fidelity(model, x.squeeze(0).cpu(), mask, baseline=norm_baseline.cpu())
            sparsity = compute_sparsity(mask)
            
            fidelities.append(fidelity)
            sparsities.append(sparsity)
        except Exception as e:
            print(f"Error on sample {i}: {e}")
            continue
    
    return {
        'fidelity': np.mean(fidelities),
        'sparsity': np.mean(sparsities),
    }


def run_rise_experiment(model, dataset, n_samples: int = 100, seed: int = 42, device: str = "cuda"):
    """Run RISE benchmark on CIFAR-100."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # CIFAR-100 Normalized neutral baseline (approx mean)
    mean = torch.tensor([0.5071, 0.4867, 0.4408]).to(device).view(1, 3, 1, 1)
    std = torch.tensor([0.2675, 0.2565, 0.2761]).to(device).view(1, 3, 1, 1)
    norm_baseline = (torch.full((1, 3, 1, 1), 0.5).to(device) - mean) / std # Gray baseline
    model = model.to(device).eval()
    explainer = RISEExplainer(model, n_masks=1000, device=device)
    
    fidelities = []
    sparsities = []
    
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    for i, (x, y) in enumerate(loader):
        if i >= n_samples:
            break
        
        x = x.to(device)
        
        try:
            mask = explainer.explain(x).cpu()
            # Handle dimensions
            if mask.dim() == 2: mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3: mask = mask.unsqueeze(0)

            # Normalize and threshold
            mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
            
            fidelity = compute_fidelity(model, x.squeeze(0), mask, baseline=norm_baseline.cpu())
            sparsity = compute_sparsity(mask.cpu())
            
            fidelities.append(fidelity)
            sparsities.append(sparsity)
        except Exception as e:
            print(f"Error on sample {i}: {e}")
            continue
    
    return {
        'fidelity': np.mean(fidelities),
        'sparsity': np.mean(sparsities),
    }


def run_gradcampp_experiment(model, dataset, n_samples: int = 100, seed: int = 42, device: str = "cuda"):
    """Run GradCAM++ benchmark on CIFAR-100."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # CIFAR-100 Normalized neutral baseline (approx mean)
    mean = torch.tensor([0.5071, 0.4867, 0.4408]).to(device).view(1, 3, 1, 1)
    std = torch.tensor([0.2675, 0.2565, 0.2761]).to(device).view(1, 3, 1, 1)
    norm_baseline = (torch.full((1, 3, 1, 1), 0.5).to(device) - mean) / std # Gray baseline
    model = model.to(device).eval()
    # For ResNet-18, the last conv layer is model.layer4
    target_layer = model.layer4
    explainer = GradCAMPlusPlusExplainer(model, target_layer, device=device)
    
    fidelities = []
    sparsities = []
    
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    for i, (x, y) in enumerate(loader):
        if i >= n_samples:
            break
        
        x = x.to(device)
        
        try:
            mask = explainer.explain(x).cpu()
            if mask.dim() == 2: mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3: mask = mask.unsqueeze(0)
            
            fidelity = compute_fidelity(model, x.squeeze(0), mask, baseline=norm_baseline.cpu())
            sparsity = compute_sparsity(mask.cpu())
            
            fidelities.append(fidelity)
            sparsities.append(sparsity)
        except Exception as e:
            print(f"Error on sample {i}: {e}")
            continue
    
    return {
        'fidelity': np.mean(fidelities),
        'sparsity': np.mean(sparsities),
    }


def main():
    """Run full CIFAR-100 benchmark suite."""
    print("=" * 60)
    print("CIFAR-100 Benchmark for DIxAI TNNLS Submission")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load dataset
    print("\nLoading CIFAR-100 dataset...")
    dataset = load_cifar100(n_samples=500)
    print(f"Loaded {len(dataset)} samples")
    
    # Initialize model using the new pretrained ResNet-18 loader
    model = get_model(device=device)
    
    # Multi-seed runner for statistical rigor
    runner = MultiSeedRunner(seeds=[42, 123, 456, 789, 1024])
    
    # Run experiments
    n_samples = 50  # Reduced for speed
    
    print("\n--- Running DIxAI ---")
    dixai_results = runner.run(run_dixai_experiment, model=model, dataset=dataset, 
                               n_samples=n_samples, device=device)
    print(f"DIxAI - Fidelity: {dixai_results['fidelity']}, Sparsity: {dixai_results['sparsity']}")
    
    print("\n--- Running RISE ---")
    rise_results = runner.run(run_rise_experiment, model=model, dataset=dataset,
                              n_samples=n_samples, device=device)
    print(f"RISE - Fidelity: {rise_results['fidelity']}, Sparsity: {rise_results['sparsity']}")
    
    print("\n--- Running GradCAM++ ---")
    gradcam_results = runner.run(run_gradcampp_experiment, model=model, dataset=dataset,
                                 n_samples=n_samples, device=device)
    print(f"GradCAM++ - Fidelity: {gradcam_results['fidelity']}, Sparsity: {gradcam_results['sparsity']}")
    
    # Generate LaTeX table
    methods = {
        'DIxAI': dixai_results,
        'RISE': rise_results,
        'GradCAM++': gradcam_results,
    }
    
    latex_table = generate_latex_comparison_table(
        methods=methods,
        metrics=['fidelity', 'sparsity'],
        caption="CIFAR-100 Benchmark Results (mean ± std over 5 seeds)",
        label="tab:cifar100"
    )
    
    print("\n" + "=" * 60)
    print("LaTeX Table:")
    print("=" * 60)
    print(latex_table)
    
    # Save results
    results_dir = project_root / "experiments" / "results"
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / "cifar100_benchmark.tex", "w") as f:
        f.write(latex_table)
    
    print(f"\nResults saved to {results_dir / 'cifar100_benchmark.tex'}")


if __name__ == "__main__":
    main()
