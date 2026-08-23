"""
Quantitative ImageNet-1K Benchmark
Computes Faithfulness (Ins/Del AUC), Fidelity, and Sparsity on high-res ImageNet samples.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import pandas as pd
import sys
from pathlib import Path
import time

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "experiments/benchmark"))

from dixai import DecisionInformationExplainer
from metrics import insertion_auc, deletion_auc

# DEVICE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_imagenet_model():
    print("  Loading pre-trained ResNet-50...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Convert all ReLUs to inplace=False for compatibility
    def fix_relu_aggressive(model):
        for name, child in model.named_children():
            if isinstance(child, nn.ReLU):
                setattr(model, name, nn.ReLU(inplace=False))
            else:
                fix_relu_aggressive(child)
    fix_relu_aggressive(model)
    
    model.eval()
    return model.to(DEVICE)

def get_samples(limit=50):
    sample_dir = "data/imagenet_samples"
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    samples = []
    if os.path.exists(sample_dir):
        files = [f for f in os.listdir(sample_dir) if f.endswith(".jpg")]
        for filename in sorted(files)[:limit]:
            path = os.path.join(sample_dir, filename)
            img = Image.open(path).convert('RGB')
            samples.append((transform(img), filename))
    return samples

def run_quant_benchmark():
    print("Starting Quantitative ImageNet Benchmark...")
    samples = get_samples(limit=50)
    if not samples:
        print("Error: No samples found in data/imagenet_samples.")
        return

    model = get_imagenet_model()
    
    explainer = DecisionInformationExplainer(
        model=model,
        lambda_fidelity=50.0,
        lambda_tv=0.3,
        device=DEVICE
    )
    
    results = []
    
    print(f"Beginning evaluation on {len(samples)} samples...")
    
    for i, (img_tensor, name) in enumerate(samples):
        img_input = img_tensor.unsqueeze(0).to(DEVICE)
        
        # Get target label from prediction
        with torch.no_grad():
            output = model(img_input)
            target = output.argmax(1).item()
            
        start_time = time.time()
        
        # Explain with Multi-scale: 224 / 7 = 32x32 mask grid
        # Reduced steps for efficiency while maintaining robustness
        explanation = explainer.explain(
            img_tensor,
            baseline=torch.zeros_like(img_input),
            steps=300, 
            downsample_factor=7,
            use_spatial_prior=True,
            anneal=True,
            verbose=False
        )
        
        elapsed = time.time() - start_time
        
        # Metrics
        saliency_np = explanation.mask_probs.squeeze().detach().cpu().numpy()
        if saliency_np.ndim == 3:
            saliency_np = saliency_np.mean(axis=0)
            
        img_np = img_tensor.cpu()
        
        ins_auc = insertion_auc(model, img_np, saliency_np, target=target, steps=10)
        del_auc = deletion_auc(model, img_np, saliency_np, target=target, steps=10)
        
        results.append({
            "sample": name,
            "target": target,
            "fidelity": explanation.fidelity_score,
            "sparsity": explanation.info_score,
            "ins_auc": ins_auc,
            "del_auc": del_auc,
            "time": elapsed
        })
        
        if (i + 1) % 5 == 0:
            print(f"  Processed {i+1}/{len(samples)}: Avg Ins AUC: {np.mean([r['ins_auc'] for r in results]):.4f}")

    df = pd.DataFrame(results)
    
    # Bootstrap CIs for Metrics
    def bootstrap_ci(data, n_iterations=1000):
        stats = []
        for _ in range(n_iterations):
            sample = np.random.choice(data, size=len(data), replace=True)
            stats.append(np.mean(sample))
        return np.percentile(stats, [2.5, 97.5])

    summary_stats = {}
    for col in ["fidelity", "sparsity", "ins_auc", "del_auc"]:
        mean = df[col].mean()
        ci = bootstrap_ci(df[col].values)
        summary_stats[col] = {"mean": mean, "ci_low": ci[0], "ci_high": ci[1]}

    print("\n" + "="*50)
    print(f"IMAGENET QUANTITATIVE RESULTS (N={len(samples)}, Bootstrap CI 95%)")
    print("="*50)
    for k, v in summary_stats.items():
        print(f"{k:10s} : {v['mean']:.4f} [{v['ci_low']:.4f}, {v['ci_high']:.4f}]")
    print("="*50)

    output_dir = "experiments/results/imagenet_quant"
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(f"{output_dir}/imagenet_metrics_bootstrapped.csv", index=False)
    
    # Save summary to LaTeX
    with open(f"{output_dir}/imagenet_summary.tex", "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Quantitative Performance on ImageNet-1K (N=50).}\n")
        f.write("\\begin{tabular}{@{}lccc@{}}\n\\toprule\n")
        f.write("\\textbf{Metric} & \\textbf{Value} \\\\ \\midrule\n")
        f.write(f"Decision Fidelity & {summary['fidelity']:.4f} \\\\\n")
        f.write(f"Mask Sparsity & {summary['sparsity']:.4f} \\\\\n")
        f.write(f"Insertion AUC & {summary['ins_auc']:.4f} \\\\\n")
        f.write(f"Deletion AUC & {summary['del_auc']:.4f} \\\\\n")
        f.write(f"Avg Time per Image (s) & {summary['time']:.2f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

if __name__ == "__main__":
    run_quant_benchmark()
