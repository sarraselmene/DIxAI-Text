import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from dixai.explainer import DecisionInformationExplainer
from experiments.benchmark.datasets import get_vision_dataloader
from experiments.benchmark.models import SimpleCNN, train_model, evaluate_model

def run_mnist_benchmark():
    print("--- Running MNIST Vision Benchmark ---")
    os.makedirs('experiments/results/visualizations', exist_ok=True)
    
    # 1. Load Data
    train_loader, test_loader = get_vision_dataloader('mnist', batch_size=64, subset_size=1000)
    
    # 2. Train Model
    model = SimpleCNN(in_channels=1, num_classes=10)
    train_model(model, train_loader, epochs=5)
    acc = evaluate_model(model, test_loader)
    print(f"CNN Accuracy: {acc:.4f}")
    
    # 3. Explain Samples
    # Get 5 samples from test set
    it = iter(test_loader)
    images, labels = next(it)
    
    # Aggressive sparsity settings for MNIST
    explainer = DecisionInformationExplainer(model, lambda_fidelity=0.1)
    
    num_samples = 3
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    # suptitle removed for manuscript integration
    
    for i in range(num_samples):
        img = images[i]
        label = labels[i]
        
        # Original Prediction
        with torch.no_grad():
            pred = torch.argmax(model(img.unsqueeze(0)), dim=-1).item()
        
        print(f"Explaining Sample {i}: True={label}, Pred={pred}")
        
        # Explain with high steps and deep negative initialization
        # For normalized MNIST, the "Black" background is -1.0
        baseline = torch.full_like(img, -1.0)
        explanation = explainer.explain(img, steps=2000, lr=0.1, init_logits=-4.0, baseline=baseline, verbose=False)
        mask = explanation.mask.squeeze().numpy() # (28, 28)
        mask_probs = explanation.mask_probs.squeeze().numpy()
        
        # Plotting
        axes[i, 0].imshow(img.squeeze().numpy(), cmap='gray')
        axes[i, 0].set_title(f"Original (Digit {label})")
        axes[i, 0].axis('off')
        
        # Heatmap Overlay (Only show significant probabilities)
        axes[i, 1].imshow(img.squeeze().numpy(), cmap='gray')
        masked_probs_viz = np.where(mask_probs > 0.05, mask_probs, np.nan)
        axes[i, 1].imshow(masked_probs_viz, cmap='jet', alpha=0.8)
        axes[i, 1].set_title(f"Mask Overlay (Density={explanation.info_score:.2f})")
        axes[i, 1].axis('off')
        
        # Masked result (Binarized at 0.5)
        # We use the -1.0 baseline for the visualization background
        binary_mask = (mask > 0.5)
        masked_img = np.where(binary_mask, img.squeeze().numpy(), -1.0)
        axes[i, 2].imshow(masked_img, vmin=-1, vmax=1, cmap='gray')
        axes[i, 2].set_title(f"Sufficient Features (Fid={explanation.fidelity_score:.1f})")
        axes[i, 2].axis('off')
        
    plt.tight_layout(pad=0.2)
    save_path = 'experiments/results/visualizations/mnist_qualitative.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
    print(f"Visualizations saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    run_mnist_benchmark()
