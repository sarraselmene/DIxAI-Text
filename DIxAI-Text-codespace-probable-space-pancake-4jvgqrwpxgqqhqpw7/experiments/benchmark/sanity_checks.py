
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import copy
import numpy as np
import matplotlib.pyplot as plt

# Try to import from installed package, otherwise fall back to src
try:
    from dixai import DecisionInformationExplainer
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
    from dixai import DecisionInformationExplainer

from models import SimpleCNN, train_model

def run_sanity_check_cifar():
    """
    Implements the 'Sanity Checks for Saliency Maps' (Adebayo et al., 2018) for CIFAR-10.
    Specifically: Model Parameter Randomization.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running CIFAR-10 sanity check on {device}...")
    
    # 0. Set seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. Setup Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True)
    
    # Get a sample image
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    img = images[0].unsqueeze(0).to(device)
    label = labels[0].item()
    
    # 2. Setup Models
    # A. Trained Model
    model = SimpleCNN(in_channels=3, num_classes=10).to(device)
    # Just use a pretrained-style or briefly trained model
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    
    print("Briefly training model to establish baseline...")
    # Train for 1 epoch for non-randomness
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for i, (inputs, targets) in enumerate(trainloader):
        optimizer.zero_grad()
        outputs = model(inputs.to(device))
        loss = criterion(outputs, targets.to(device))
        loss.backward()
        optimizer.step()
        if i >= 50: break # Sufficient for non-random
    model.eval()
    
    # 3. Explain Trained Model
    print("Explaining trained model...")
    explainer = DecisionInformationExplainer(model, lambda_fidelity=5.0, device=device)
    exp_normal = explainer.explain(img, steps=300, anneal=True, verbose=False)
    mask_normal = exp_normal.mask.squeeze().detach().cpu().numpy()
    
    # 4. Randomize Model Parameters (Cascading)
    print("Randomizing model parameters (Cascading)...")
    random_model = copy.deepcopy(model)
    # Randomize the classifier layers (cascading from top down)
    for layer in random_model.classifier:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            
    # Also randomize the features block
    for layer in random_model.features:
        if isinstance(layer, nn.Conv2d):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
    
    random_model.to(device)
    random_model.eval()
    
    # 5. Explain Random Model
    print("Explaining random model...")
    explainer_rand = DecisionInformationExplainer(random_model, lambda_fidelity=5.0, device=device)
    exp_rand = explainer_rand.explain(img, steps=300, anneal=True, verbose=False)
    mask_rand = exp_rand.mask.squeeze().detach().cpu().numpy()
    
    # 6. Compare
    # SSIM or Correlation on the masks
    corr = np.corrcoef(mask_normal.flatten(), mask_rand.flatten())[0, 1]
    print(f"\n--- Sanity Check: Model Parameter Randomization ---")
    print(f"Correlation (Trained vs Random): {corr:.4f}")
    
    # 7. Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    # Ensure (H, W, C)
    orig = img.detach().cpu().squeeze().numpy()
    if orig.ndim == 3 and orig.shape[0] == 3:
        orig = orig.transpose(1, 2, 0)
    
    orig = orig * 0.5 + 0.5 # unnormalize
    axes[0].imshow(orig.clip(0, 1))
    axes[0].set_title(f"Original (Class: {testset.classes[label]})")
    axes[0].axis('off')
    
    # Trained Explanation
    # mask_normal has same shape as input: (1, 3, 32, 32)
    # We mean across channels for visualization
    m_norm = mask_normal.squeeze()
    if m_norm.ndim == 3:
        m_norm = m_norm.mean(0)
    
    axes[1].imshow(m_norm, cmap='hot')
    axes[1].set_title(f"Trained Explanation")
    axes[1].axis('off')
    
    # Random Explanation
    m_rand = mask_rand.squeeze()
    if m_rand.ndim == 3:
        m_rand = m_rand.mean(0)
        
    axes[2].imshow(m_rand, cmap='hot')
    axes[2].set_title(f"Random Explanation\nScore: {corr:.4f}")
    axes[2].axis('off')
    
    plot_path = os.path.join(os.path.dirname(__file__), '../results/sanity_cifar.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Sanity check plot saved to {plot_path}")
    
    # Save quantitative result
    with open('experiments/results/sanity_check_results.txt', 'w') as f:
        f.write(f"Sanity Check (Parameter Randomization) Correlation: {corr:.6f}\n")

if __name__ == "__main__":
    run_sanity_check_cifar()
