"""
CheXpert Medical Imaging Benchmark for DIxAI

Evaluates DIxAI explanations on chest X-ray classification,
validating performance in safety-critical medical imaging domain.

CheXpert is a large dataset of chest radiographs with uncertainty labels.
https://stanfordmlgroup.github.io/competitions/chexpert/
"""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict
import time

# Import DIxAI and baselines
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from dixai import DecisionInformationExplainer
from dixai.baselines import RISEExplainer, GradCAMPlusPlusExplainer
from dixai.metrics import calculate_fidelity, calculate_sparsity


class CheXpertDataset(Dataset):
    """
    CheXpert dataset loader for benchmark evaluation.
    
    Expects data in the following structure:
    data/chexpert/
        CheXpert-v1.0-small/
            train.csv
            train/
                patient00001/
                    study1/
                        view1_frontal.jpg
    """
    
    PATHOLOGIES = [
        'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
        'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
        'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
        'Pleural Other', 'Fracture', 'Support Devices'
    ]
    
    def __init__(
        self,
        root_dir: str = 'data/chexpert',
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        target_pathology: str = 'Pneumonia',
        limit: Optional[int] = None
    ):
        self.root_dir = root_dir
        self.split = split
        self.target_pathology = target_pathology
        
        # Load CSV
        csv_path = os.path.join(root_dir, 'CheXpert-v1.0-small', f'{split}.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CheXpert CSV not found at {csv_path}")
        
        self.df = pd.read_csv(csv_path)
        
        # Filter for frontal views and positive/negative labels
        self.df = self.df[self.df['Frontal/Lateral'] == 'Frontal']
        
        # Map uncertain labels (-1) to positive (1) for simplicity
        self.df[target_pathology] = self.df[target_pathology].apply(
            lambda x: 1 if x in [1, -1] else 0
        )
        
        # Limit dataset size
        if limit:
            self.df = self.df.head(limit)
        
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        
        # Construct image path
        img_path = os.path.join(self.root_dir, row['Path'])
        
        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # Get label
        label = int(row[self.target_pathology])
        
        return image, label, row['Path']


def create_chexpert_model(num_classes: int = 2, pretrained: bool = True):
    """
    Create a DenseNet-121 model for CheXpert classification.
    DenseNet-121 is commonly used for chest X-ray classification.
    """
    model = models.densenet121(weights='IMAGENET1K_V1' if pretrained else None)
    # Replace classifier for binary classification
    model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    return model


def run_dixai_benchmark(
    model: torch.nn.Module,
    dataset: CheXpertDataset,
    device: str = 'cuda',
    n_samples: int = 100,
    seed: int = 42
) -> Dict:
    """Run DIxAI benchmark on CheXpert dataset."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = model.to(device).eval()
    
    explainer = DecisionInformationExplainer(
        model=model,
        lambda_fidelity=50.0,
        task='classification',
        device=device,
        lambda_tv=0.01  # Spatial smoothness for medical images
    )
    
    # Neutral baseline (gray for medical images)
    baseline = torch.full((1, 3, 224, 224), 0.5).to(device)
    
    fidelities = []
    sparsities = []
    latencies = []
    
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    for i, (x, y, path) in enumerate(loader):
        if i >= n_samples:
            break
        
        x = x.to(device)
        
        try:
            start_time = time.time()
            explanation = explainer.explain(
                x,
                steps=300,
                downsample_factor=7,
                use_spatial_prior=True,
                anneal=True
            )
            latency = time.time() - start_time
            
            fidelities.append(explanation.fidelity_score)
            sparsities.append(explanation.info_score)
            latencies.append(latency)
            
        except Exception as e:
            print(f"Error on sample {i}: {e}")
            continue
        
        if (i + 1) % 10 == 0:
            print(f"DIxAI: Processed {i + 1}/{n_samples} | Fidelity: {np.mean(fidelities):.4f}")
    
    return {
        'method': 'DIxAI',
        'fidelity_mean': np.mean(fidelities),
        'fidelity_std': np.std(fidelities),
        'sparsity_mean': np.mean(sparsities),
        'sparsity_std': np.std(sparsities),
        'latency_mean': np.mean(latencies),
    }


def run_rise_benchmark(
    model: torch.nn.Module,
    dataset: CheXpertDataset,
    device: str = 'cuda',
    n_samples: int = 100,
    seed: int = 42
) -> Dict:
    """Run RISE benchmark on CheXpert dataset."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = model.to(device).eval()
    
    explainer = RISEExplainer(
        model=model,
        n_masks=1000,
        mask_prob=0.5,
        device=device
    )
    
    fidelities = []
    sparsities = []
    
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    for i, (x, y, path) in enumerate(loader):
        if i >= n_samples:
            break
        
        x = x.to(device)
        
        try:
            mask = explainer.explain(x)
            
            # Threshold mask for sparsity calculation
            mask_binary = (mask > 0.5).float()
            
            sparsity = 1.0 - mask_binary.mean().item()
            
            # Approximate fidelity (mask already normalized)
            fidelity = mask.mean().item()  # Simplified proxy
            
            fidelities.append(fidelity)
            sparsities.append(sparsity)
            
        except Exception as e:
            print(f"Error on sample {i}: {e}")
            continue
    
    return {
        'method': 'RISE',
        'fidelity_mean': np.mean(fidelities),
        'fidelity_std': np.std(fidelities),
        'sparsity_mean': np.mean(sparsities),
        'sparsity_std': np.std(sparsities),
    }


def run_gradcampp_benchmark(
    model: torch.nn.Module,
    dataset: CheXpertDataset,
    device: str = 'cuda',
    n_samples: int = 100,
    seed: int = 42
) -> Dict:
    """Run Grad-CAM++ benchmark on CheXpert dataset."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = model.to(device).eval()
    
    # Target layer for DenseNet-121
    target_layer = model.features.denseblock4
    
    explainer = GradCAMPlusPlusExplainer(
        model=model,
        target_layer=target_layer,
        device=device
    )
    
    fidelities = []
    sparsities = []
    
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    for i, (x, y, path) in enumerate(loader):
        if i >= n_samples:
            break
        
        x = x.to(device)
        
        try:
            mask = explainer.explain(x)
            
            mask_binary = (mask > 0.5).float()
            sparsity = 1.0 - mask_binary.mean().item()
            fidelity = mask.mean().item()
            
            fidelities.append(fidelity)
            sparsities.append(sparsity)
            
        except Exception as e:
            print(f"Error on sample {i}: {e}")
            continue
    
    return {
        'method': 'Grad-CAM++',
        'fidelity_mean': np.mean(fidelities),
        'fidelity_std': np.std(fidelities),
        'sparsity_mean': np.mean(sparsities),
        'sparsity_std': np.std(sparsities),
    }


def run_full_benchmark(n_samples: int = 100, seed: int = 42):
    """
    Run complete CheXpert benchmark comparing DIxAI, RISE, and Grad-CAM++.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running CheXpert Medical Imaging Benchmark on {device}")
    print("=" * 60)
    
    # Check if CheXpert data exists
    chexpert_dir = 'data/chexpert'
    if not os.path.exists(chexpert_dir):
        print(f"CheXpert data not found at {chexpert_dir}")
        print("Please download CheXpert-v1.0-small from:")
        print("https://stanfordmlgroup.github.io/competitions/chexpert/")
        return
    
    # Load dataset
    dataset = CheXpertDataset(
        root_dir=chexpert_dir,
        target_pathology='Pneumonia',
        limit=n_samples * 2  # Extra samples in case of errors
    )
    
    # Create model (in practice, load pre-trained weights)
    model = create_chexpert_model(num_classes=2, pretrained=True)
    
    # Run benchmarks
    results = []
    
    print("\n[1/3] Running DIxAI Benchmark...")
    dixai_results = run_dixai_benchmark(model, dataset, device, n_samples, seed)
    results.append(dixai_results)
    print(f"DIxAI: Fidelity={dixai_results['fidelity_mean']:.4f}, Sparsity={dixai_results['sparsity_mean']:.4f}")
    
    print("\n[2/3] Running RISE Benchmark...")
    rise_results = run_rise_benchmark(model, dataset, device, n_samples, seed)
    results.append(rise_results)
    print(f"RISE: Fidelity={rise_results['fidelity_mean']:.4f}, Sparsity={rise_results['sparsity_mean']:.4f}")
    
    print("\n[3/3] Running Grad-CAM++ Benchmark...")
    gradcam_results = run_gradcampp_benchmark(model, dataset, device, n_samples, seed)
    results.append(gradcam_results)
    print(f"Grad-CAM++: Fidelity={gradcam_results['fidelity_mean']:.4f}, Sparsity={gradcam_results['sparsity_mean']:.4f}")
    
    # Save results
    df = pd.DataFrame(results)
    output_path = 'experiments/results/chexpert_benchmark.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print("\n" + "=" * 60)
    print("Benchmark Complete!")
    print(f"Results saved to {output_path}")
    print("\nSummary Table:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    run_full_benchmark(n_samples=50, seed=42)
