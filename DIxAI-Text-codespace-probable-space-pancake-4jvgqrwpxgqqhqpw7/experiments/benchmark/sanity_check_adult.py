import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.dixai import DecisionInformationExplainer
from experiments.benchmark.datasets import get_tabular_dataset
from experiments.benchmark.models import SimpleMLP

def train_model_on_device(model, train_loader, device, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    model.eval()

def run_sanity_check_adult():
    """
    Implements Parameter Randomization check on Adult dataset.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running Adult Sanity Check on {device}...")
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. Load Data
    X_train, X_test, y_train, y_test = get_tabular_dataset('adult')
    input_dim = X_train.shape[1]
    num_classes = 2 # Adult is binary
    
    # 2. Setup Model
    model = SimpleMLP(input_dim=input_dim, output_dim=num_classes).to(device)
    
    # 3. Train Model
    print("Training base model on Adult...")
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    train_model_on_device(model, train_loader, device, epochs=5)
    
    # 4. Get Explanation for Trained Model
    print("Explaining trained model...")
    # Pick a sample
    idx = 0
    x_val = X_test[idx].to(device)
    
    explainer = DecisionInformationExplainer(model, lambda_fidelity=20.0, device=device)
    exp_normal = explainer.explain(x_val, steps=500, anneal=True, verbose=False)
    mask_normal = exp_normal.mask.flatten().detach().cpu().numpy()
    
    # 5. Randomize Model Parameters
    print("Randomizing model...")
    random_model = copy.deepcopy(model)
    def weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    random_model.apply(weights_init)
    random_model.to(device)
    random_model.eval()
    
    # 6. Get Explanation for Random Model
    print("Explaining random model...")
    explainer_rand = DecisionInformationExplainer(random_model, lambda_fidelity=20.0, device=device)
    exp_rand = explainer_rand.explain(x_val, steps=500, anneal=True, verbose=False)
    mask_rand = exp_rand.mask.flatten().detach().cpu().numpy()
    
    # 7. Correlation
    if np.std(mask_normal) == 0 or np.std(mask_rand) == 0:
        corr = 0.0
    else:
        corr = np.corrcoef(mask_normal, mask_rand)[0, 1]
    print(f"Randomization Correlation: {corr:.4f}")
    
    # 8. Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    feat_indices = np.arange(len(mask_normal))
    
    ax1.bar(feat_indices, mask_normal, color='teal', alpha=0.8)
    ax1.set_title(f"Trained Model (Adult)\nFid: {exp_normal.fidelity_score:.2f}", fontsize=12)
    ax1.set_ylim(0, 1.1)
    ax1.set_xlabel("Feature Index")
    ax1.set_ylabel("Selection Probability")
    
    ax2.bar(feat_indices, mask_rand, color='crimson', alpha=0.8)
    ax2.set_title(f"Random Model (Adult)\nCorrelation: {corr:.4f}", fontsize=12)
    ax2.set_ylim(0, 1.1)
    ax2.set_xlabel("Feature Index")
    
    plt.suptitle("Sanity Check: Model Parameter Randomization", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs('figures', exist_ok=True)
    save_path = 'figures/sanity_check_randomization_adult.png'
    plt.savefig(save_path, dpi=300)
    print(f"Saved sanity check plot to {save_path}")

if __name__ == "__main__":
    run_sanity_check_adult()
