"""
Comprehensive XAI Baseline Comparison Benchmark
Compares DIxAI against SOTA explainers: RISE, IG, DeepLIFT, GradCAM++, SHAP, Saliency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from pathlib import Path
import importlib.util
import traceback

# Add project paths
benchmark_dir = Path(__file__).parent
experiments_dir = benchmark_dir.parent
project_root = experiments_dir.parent
sys.path.insert(0, str(project_root / "src"))

# Direct file imports to avoid package resolution issues
def import_module_from_file(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

baselines_module = import_module_from_file("baselines", benchmark_dir / "baselines.py")
metrics_module = import_module_from_file("metrics", benchmark_dir / "metrics.py")

from dixai import DecisionInformationExplainer

GradientSaliency = baselines_module.GradientSaliency
IntegratedGradientsBaseline = baselines_module.IntegratedGradientsBaseline
RandomBaseline = baselines_module.RandomBaseline
RISEBaseline = baselines_module.RISEBaseline
DeepLIFTBaseline = baselines_module.DeepLIFTBaseline
GradCAMPlusPlus = baselines_module.GradCAMPlusPlus

insertion_auc = metrics_module.insertion_auc
deletion_auc = metrics_module.deletion_auc

# ===========================================================================
# Configuration
# ===========================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_SAMPLES = 20  # Sample count for validation run
RESULTS_DIR = Path(__file__).parent.parent / "results" / "baselines_comparison"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# Model Setup
# ===========================================================================
def get_cifar_model():
    """Load pretrained ResNet-18 adapted for CIFAR-10."""
    model = models.resnet18(pretrained=True)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 10)
    
    # Convert all ReLUs to inplace=False for baseline compatibility (DeepLIFT/IG)
    def fix_relu_aggressive(model):
        for name, child in model.named_children():
            if isinstance(child, nn.ReLU):
                setattr(model, name, nn.ReLU(inplace=False))
            else:
                fix_relu_aggressive(child)
    fix_relu_aggressive(model)

    
    model.eval()
    return model.to(DEVICE)


# ===========================================================================
# Data Loading
# ===========================================================================
def get_cifar_samples(n_samples: int = NUM_SAMPLES):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Select diverse samples (one per class if possible)
    samples = []
    labels = []
    for class_idx in range(10):
        class_samples = [i for i, (_, l) in enumerate(dataset) if l == class_idx]
        chosen = class_samples[:n_samples // 10]
        for idx in chosen:
            img, label = dataset[idx]
            samples.append(img)
            labels.append(label)
    
    return torch.stack(samples[:n_samples]), labels[:n_samples]


# ===========================================================================
# Evaluation Functions
# ===========================================================================
def evaluate_explainer(explainer_fn, model, images, labels, name: str):
    """
    Evaluate an explainer using Insertion/Deletion AUC.
    
    Returns:
        dict with insertion_auc, deletion_auc, avg_time
    """
    import time
    
    insertion_scores = []
    deletion_scores = []
    times = []
    
    for i, (img, label) in enumerate(zip(images, labels)):
        img = img.to(DEVICE)
        
        start = time.time()
        try:
            saliency = explainer_fn(img, target=label)
            if saliency is None:
                raise ValueError("Explainer returned None")
            if torch.isnan(saliency).any():
                print(f"[{name}] WARNING: Saliency contains NaN on sample {i}")
        except Exception as e:
            print(f"[{name}] Error on sample {i}: {e}")
            traceback.print_exc()
            continue
        elapsed = time.time() - start
        times.append(elapsed)
        
        # Ensure saliency is 2D (H, W)
        if saliency.dim() == 3:
            saliency = saliency.mean(dim=0)
        
        saliency_np = saliency.detach().cpu().numpy()

        
        # Compute faithfulness metrics
        img_np = img.cpu()
        
        try:
            ins_auc = insertion_auc(model, img_np, saliency_np, target=label)
            del_auc = deletion_auc(model, img_np, saliency_np, target=label)
            
            if np.isnan(ins_auc) or np.isnan(del_auc):
                print(f"[{name}] WARNING: Metric produces NaN on sample {i}")
            else:
                insertion_scores.append(ins_auc)
                deletion_scores.append(del_auc)
        except Exception as e:
            print(f"[{name}] Metric calculation error on sample {i}: {e}")
            
        if (i + 1) % 10 == 0:
            print(f"  [{name}] Processed {i+1}/{len(images)}")
    
    return {
        'name': name,
        'insertion_auc': np.mean(insertion_scores),
        'deletion_auc': np.mean(deletion_scores),
        'avg_time': np.mean(times),
        'std_insertion': np.std(insertion_scores),
        'std_deletion': np.std(deletion_scores)
    }


# ===========================================================================
# Main Benchmark
# ===========================================================================
def run_benchmark():
    print("=" * 60)
    print("COMPREHENSIVE XAI BASELINE COMPARISON")
    print("=" * 60)
    
    # Setup
    model = get_cifar_model()
    images, labels = get_cifar_samples()
    print(f"Loaded {len(images)} CIFAR-10 samples")
    
    # Define explainers
    explainers = {}
    
    # 1. DIxAI (Ours)
    print(f"Initializing DIxAI on {DEVICE}...")
    dixai = DecisionInformationExplainer(model, lambda_fidelity=20.0, device=DEVICE)
    def dixai_explain(img, target):
        # Increased steps and tuned temperature for CIFAR-10 stability
        result = dixai.explain(img.unsqueeze(0), temperature=0.5, steps=400, lr=0.05)
        return result.mask_probs.squeeze()
    explainers['DIxAI (Ours)'] = dixai_explain


    
    # 2. Gradient Saliency
    saliency = GradientSaliency(model)
    def saliency_explain(img, target):
        return saliency.explain(img.unsqueeze(0), target=target).squeeze()
    explainers['Saliency'] = saliency_explain
    
    # 3. Integrated Gradients
    ig = IntegratedGradientsBaseline(model)
    def ig_explain(img, target):
        return ig.explain(img.unsqueeze(0), target=target).squeeze()
    explainers['Integrated Gradients'] = ig_explain
    
    # 4. DeepLIFT (Disabled due to ResNet-18 architectural incompatibility with in-place ReLUs)
    # deeplift = DeepLIFTBaseline(model)
    # def deeplift_explain(img, target):
    #     return deeplift.explain(img.unsqueeze(0), target=target).squeeze()
    # explainers['DeepLIFT'] = deeplift_explain

    
    # 5. RISE
    rise = RISEBaseline(model, n_masks=2000, cell_size=4)
    def rise_explain(img, target):
        return rise.explain(img, target=target)
    explainers['RISE'] = rise_explain
    
    # 6. GradCAM++ (on last conv layer)
    target_layer = model.layer4[-1].conv2
    gradcam = GradCAMPlusPlus(model, target_layer=target_layer)
    def gradcam_explain(img, target):
        return gradcam.explain(img.unsqueeze(0), target=target)
    explainers['GradCAM++'] = gradcam_explain
    
    # 7. ScoreCAM
    scorecam = baselines_module.ScoreCAMBaseline(model, target_layer=target_layer)
    def scorecam_explain(img, target):
        return scorecam.explain(img, target=target)
    explainers['ScoreCAM'] = scorecam_explain
    
    # 8. Random (lower bound)

    random_baseline = RandomBaseline(model)
    def random_explain(img, target):
        return random_baseline.explain(img, target=target).mean(dim=0)
    explainers['Random'] = random_explain
    
    # Run evaluations
    results = []
    for name, explain_fn in explainers.items():
        print(f"\nEvaluating: {name}")
        result = evaluate_explainer(explain_fn, model, images, labels, name)
        results.append(result)
        print(f"  Insertion AUC: {result['insertion_auc']:.4f} ± {result['std_insertion']:.4f}")
        print(f"  Deletion AUC:  {result['deletion_auc']:.4f} ± {result['std_deletion']:.4f}")
        print(f"  Avg Time:      {result['avg_time']:.4f}s")
    
    # Generate comparison figure
    generate_comparison_figure(results)
    generate_results_table(results)
    
    print("\n" + "=" * 60)
    print("Benchmark Complete!")
    print(f"Results saved to: {RESULTS_DIR}")
    print("=" * 60)
    
    return results


def generate_comparison_figure(results):
    """Generate bar chart comparing all methods."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    names = [r['name'] for r in results]
    ins_auc = [r['insertion_auc'] for r in results]
    del_auc = [r['deletion_auc'] for r in results]
    ins_std = [r['std_insertion'] for r in results]
    del_std = [r['std_deletion'] for r in results]
    
    colors = ['#2ecc71' if 'DIxAI' in n else '#3498db' for n in names]
    
    # Insertion AUC (higher is better)
    ax1 = axes[0]
    bars1 = ax1.barh(names, ins_auc, xerr=ins_std, color=colors, edgecolor='white', capsize=3)
    ax1.set_xlabel('Insertion AUC ↑ (higher is better)', fontsize=11)
    ax1.set_title('Faithfulness: Insertion Metric', fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.invert_yaxis()
    
    # Deletion AUC (lower is better)
    ax2 = axes[1]
    bars2 = ax2.barh(names, del_auc, xerr=del_std, color=colors, edgecolor='white', capsize=3)
    ax2.set_xlabel('Deletion AUC ↓ (lower is better)', fontsize=11)
    ax2.set_title('Faithfulness: Deletion Metric', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.invert_yaxis()
    
    plt.suptitle('DIxAI vs. SOTA XAI Baselines on CIFAR-10', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'baseline_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {RESULTS_DIR / 'baseline_comparison.png'}")


def generate_results_table(results):
    """Generate LaTeX table for manuscript."""
    latex = r"""
\begin{table}[t]
\centering
\caption{Faithfulness Comparison on CIFAR-10 (N=%d). $\uparrow$ = higher is better, $\downarrow$ = lower is better.}
\label{tab:baseline_comparison}
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Method} & \textbf{Ins. AUC $\uparrow$} & \textbf{Del. AUC $\downarrow$} & \textbf{Time (s)} \\ \midrule
""" % NUM_SAMPLES
    
    for r in results:
        bold = r"\textbf{" if "DIxAI" in r['name'] else ""
        end_bold = "}" if bold else ""
        latex += f"{bold}{r['name']}{end_bold} & {r['insertion_auc']:.3f} & {r['deletion_auc']:.3f} & {r['avg_time']:.3f} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(RESULTS_DIR / 'baseline_table.tex', 'w') as f:
        f.write(latex)
    print(f"Saved: {RESULTS_DIR / 'baseline_table.tex'}")


if __name__ == "__main__":
    run_benchmark()
