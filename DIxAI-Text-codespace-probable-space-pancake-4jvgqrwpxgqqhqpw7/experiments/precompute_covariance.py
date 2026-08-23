
import torch
import numpy as np
import os
import argparse
from tqdm import tqdm
from torchvision import models, transforms
from torchvision.datasets import CIFAR10, ImageDataFolder
from torch.utils.data import DataLoader

def precompute_covariance(data_path, output_path="covariance_matrix.pt", batch_size=128, device="cuda"):
    print(f"--- Precomputing Class-Conditional Covariance ---")
    
    if not torch.cuda.is_available() and device == "cuda":
        print("Warning: CUDA not available, using CPU.")
        device = "cpu"
        
    # Feature Extractor (ResNet50)
    # We use the penultimate layer features
    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Identity() # Remove classification head
    model.to(device)
    model.eval()
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset (Mocking CIFAR for simplicity in this script, easily swappable)
    if not os.path.exists(data_path):
        print(f"Dataset path {data_path} not found. Utilizing fake data for demonstration.")
        # Create fake features (N samples, D dimension)
        N = 1000
        D = 2048
        features = torch.randn(N, D)
        labels = torch.randint(0, 10, (N,))
    else:
        # Real dataloading logic would go here
        pass

    # Compute Covariance per Class
    unique_classes = torch.unique(labels)
    covariances = {}
    
    for c in tqdm(unique_classes, desc="Processing Classes"):
        class_mask = (labels == c)
        class_features = features[class_mask]
        
        if len(class_features) < 2:
            print(f"Skipping class {c.item()} (insufficient samples)")
            continue
            
        # Centering
        mean = torch.mean(class_features, dim=0)
        centered = class_features - mean
        
        # Covariance calculation: (X^T X) / (N-1)
        cov = torch.mm(centered.t(), centered) / (centered.shape[0] - 1)
        
        # We only save the diagonal (variance) and a low-rank approximation in practice to save space
        # For this artifact, we save the full diagonal for efficient mask regularization
        covariances[c.item()] = cov.diag().cpu()
        
    torch.save(covariances, output_path)
    print(f"✔ Covariances saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./data", help="Data directory")
    parser.add_argument("--out", type=str, default="checkpoints/covariance.pt", help="Output path")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    precompute_covariance(args.data, args.out)
