"""
Cross-Architecture Generalization Benchmark
Validates DIxAI across ResNet-50, ViT-B/16, ConvNeXt-Tiny, and Swin-Tiny
on ImageNet validation samples with full metric reporting.
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

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from dixai import DecisionInformationExplainer
from dixai.baselines import RISEExplainer, IntegratedGradientsExplainer, GradCAMPlusPlusExplainer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SEEDS = 5

# ── Model Zoo ──────────────────────────────────────────────────────────

def load_model(arch_name):
    """Load a pretrained model by architecture name."""
    loaders = {
        "ResNet-50": lambda: models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2),
        "ViT-B/16": lambda: models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1),
        "ConvNeXt-T": lambda: models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
        "Swin-T": lambda: models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1),
    }
    print(f"  Loading {arch_name}...")
    model = loaders[arch_name]()
    model.eval()
    return model.to(DEVICE)


def get_target_layer(model, arch_name):
    """Return the last convolutional / feature layer for Grad-CAM++."""
    if arch_name == "ResNet-50":
        return model.layer4[-1].conv3
    elif arch_name == "ViT-B/16":
        return model.encoder.layers[-1].ln_1
    elif arch_name == "ConvNeXt-T":
        return model.features[-1][-1].block[-1]
    elif arch_name == "Swin-T":
        return model.features[-1][-1].norm1
    return None

# ── Data ──────────────────────────────────────────────────────────────

def get_imagenet_samples(limit=20):
    sample_dir = project_root / "data" / "imagenet_samples"
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    samples = []
    if sample_dir.exists():
        files = sorted([f for f in os.listdir(sample_dir)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        for fname in files[:limit]:
            img = Image.open(sample_dir / fname).convert("RGB")
            samples.append((transform(img), fname))
    return samples

# ── Metrics ───────────────────────────────────────────────────────────

def compute_insertion_auc(model, img_tensor, mask, n_steps=10):
    """Insertion AUC: progressively reveal important pixels."""
    img = img_tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        orig_prob = torch.softmax(model(img), 1).max().item()
        target = model(img).argmax(1).item()

    flat_mask = mask.flatten()
    order = flat_mask.argsort(descending=True)
    step_size = max(1, len(order) // n_steps)

    scores = [0.0]
    for k in range(1, n_steps + 1):
        idx = order[:k * step_size]
        canvas = torch.zeros_like(img)
        canvas.view(-1)[idx.repeat(3)] = img.view(-1)[idx.repeat(3)]  # simplified
        with torch.no_grad():
            prob = torch.softmax(model(canvas), 1)[0, target].item()
        scores.append(prob)
    return np.trapz(scores, dx=1.0 / n_steps)


def compute_deletion_auc(model, img_tensor, mask, n_steps=10):
    """Deletion AUC: progressively remove important pixels."""
    img = img_tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        target = model(img).argmax(1).item()
        orig_prob = torch.softmax(model(img), 1)[0, target].item()

    flat_mask = mask.flatten()
    order = flat_mask.argsort(descending=True)
    step_size = max(1, len(order) // n_steps)

    scores = [orig_prob]
    for k in range(1, n_steps + 1):
        idx = order[:k * step_size]
        canvas = img.clone()
        canvas.view(-1)[idx.repeat(3)] = 0.0
        with torch.no_grad():
            prob = torch.softmax(model(canvas), 1)[0, target].item()
        scores.append(prob)
    return np.trapz(scores, dx=1.0 / n_steps)

# ── Benchmark Core ────────────────────────────────────────────────────

def run_architecture_benchmark(arch_name, samples, seed=42):
    """Run DIxAI on a single architecture, return per-sample metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = load_model(arch_name)
    explainer = DecisionInformationExplainer(
        model=model, lambda_fidelity=50.0, lambda_tv=0.3, device=DEVICE
    )

    results = []
    for img_tensor, name in samples:
        img_input = img_tensor.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            target = model(img_input).argmax(1).item()

        start = time.time()
        try:
            explanation = explainer.explain(
                img_tensor,
                baseline=torch.zeros_like(img_input),
                steps=300,
                downsample_factor=7 if "ViT" in arch_name or "Swin" in arch_name else 4,
                use_spatial_prior=True,
                anneal=True,
                verbose=False,
            )
            latency = time.time() - start

            mask = explanation.mask.squeeze().cpu()
            if mask.ndim == 3:
                mask = mask.mean(0)

            results.append({
                "architecture": arch_name,
                "sample": name,
                "seed": seed,
                "fidelity": explanation.fidelity_score,
                "sparsity": explanation.info_score,
                "latency": latency,
                "target": target,
            })
        except Exception as e:
            print(f"    Error on {name}: {e}")
            results.append({
                "architecture": arch_name,
                "sample": name,
                "seed": seed,
                "fidelity": None,
                "sparsity": None,
                "latency": None,
                "target": target,
            })

    del model
    torch.cuda.empty_cache()
    return results

# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Cross-Architecture Generalization Benchmark")
    print("=" * 70)

    architectures = ["ResNet-50", "ViT-B/16", "ConvNeXt-T", "Swin-T"]
    samples = get_imagenet_samples(limit=20)
    if not samples:
        print("ERROR: No samples found in data/imagenet_samples/")
        print("  → Place ImageNet validation images there first.")
        return

    print(f"Loaded {len(samples)} ImageNet samples.")
    print(f"Architectures: {architectures}")
    print(f"Seeds: {N_SEEDS}\n")

    results_dir = project_root / "experiments" / "results" / "cross_arch"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for arch in architectures:
        print(f"\n{'─' * 60}")
        print(f"  Architecture: {arch}")
        print(f"{'─' * 60}")
        for seed in range(N_SEEDS):
            print(f"    Seed {seed + 1}/{N_SEEDS}...")
            arch_results = run_architecture_benchmark(arch, samples, seed=seed)
            all_results.extend(arch_results)

    df = pd.DataFrame(all_results)
    df.to_csv(results_dir / "cross_arch_results.csv", index=False)

    # Summary table
    print("\n" + "=" * 70)
    print("CROSS-ARCHITECTURE SUMMARY (mean ± std over seeds)")
    print("=" * 70)

    summary = df.groupby("architecture").agg(
        fidelity_mean=("fidelity", "mean"),
        fidelity_std=("fidelity", "std"),
        sparsity_mean=("sparsity", "mean"),
        sparsity_std=("sparsity", "std"),
        latency_mean=("latency", "mean"),
    ).round(4)

    for arch in architectures:
        if arch in summary.index:
            row = summary.loc[arch]
            print(f"  {arch:15s} | Fidelity: {row.fidelity_mean:.3f}±{row.fidelity_std:.3f} | "
                  f"Sparsity: {row.sparsity_mean:.3f}±{row.sparsity_std:.3f} | "
                  f"Latency: {row.latency_mean:.2f}s")

    print(f"\nResults saved to {results_dir / 'cross_arch_results.csv'}")


if __name__ == "__main__":
    main()
