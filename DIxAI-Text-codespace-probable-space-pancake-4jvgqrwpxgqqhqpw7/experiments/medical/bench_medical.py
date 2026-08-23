import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import torchxrayvision as xrv
from dixai import DecisionInformationExplainer

def run_medical_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Medical Imaging (torchxrayvision DenseNet-121) Benchmark on {device}...")
    
    # 1. Load Pre-trained Medical DenseNet-121 (Trained on CheXpert/NIH/etc)
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval().to(device)
    
    # 2. Setup Data
    sample_dir = r"d:\Haythem\Temp\AAAA-New Research\decision_information_xai\data\medical_samples"
    if not os.path.exists(sample_dir):
        print(f"Error: {sample_dir} not found. Run downloader first.")
        return
        
    samples = sorted([f for f in os.listdir(sample_dir) if f.endswith(".png") or f.endswith(".jpg")])
    if not samples:
        print("No samples found in data/medical_samples.")
        return
        
    # Standard transforms for medical DenseNet (Single channel, 224x224, [-1024, 1024] range)
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        # torchxrayvision models (except some) expect [-1024, 1024]
        transforms.Lambda(lambda x: (x * 2048) - 1024)
    ])
    
    explainer = DecisionInformationExplainer(
        model=model,
        lambda_fidelity=100.0, 
        lambda_tv=1.2,        # Increased for anatomical contiguity
        task='classification',
        device=device
    )
    
    fig, axes = plt.subplots(len(samples), 3, figsize=(15, 6 * len(samples)))
    if len(samples) == 1: axes = axes.reshape(1, -1)
    
    plt.suptitle("DIxAI Clinical Diagnostic: Pathologically Sufficient Anatomy", fontsize=28, fontweight='black', y=0.98)
    
    # We want to explain a specific pathology (e.g., Pneumonia or Effusion)
    # torchxrayvision model.pathologies: ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia', 'Lung Opacity', 'Enlarged Cardiomediastinum', 'Lung Lesion', 'Fracture', 'Support Devices']
    target_pathology = "Pneumonia"
    target_idx = model.pathologies.index(target_pathology)
    
    for i, filename in enumerate(samples):
        path = os.path.join(sample_dir, filename)
        img = Image.open(path).convert('RGB')
        # Convert to grayscale for torchxrayvision which usually expects single channel, 
        # but our explainer/model wrapper might handle RGB. 
        # torchxrayvision models usually take [1, 1, 224, 224] but here we repeat to 3 for consistency with DIxAI RGB expectations
        img_tensor = transform(img).to(device).float()
        print(f"    img_tensor shape: {img_tensor.shape}")
        
        # Get prediction
        with torch.no_grad():
            output = model(img_tensor.unsqueeze(0))
            print(f"    output shape: {output.shape}")
            pred_val = output[0, target_idx].item()
            conf = torch.sigmoid(output[0, target_idx]).item()
        
        print(f"    Explaining {filename} ({target_pathology} Score: {pred_val:.2f}, Prob: {conf:.2f})...")
        
        # Explain
        explanation = explainer.explain(
            img_tensor,
            steps=1000, 
            downsample_factor=4, 
            use_spatial_prior=True, # Critical for medical anatomy
            anneal=True,
            verbose=False
        )
        
        # Visualize
        # torchxrayvision normalization is different, basically -1024 to 1024 or similar, 
        # but our transform used normalize(..., 255) which is -1024 to 1024.
        # Let's map back to 0-1 for display
        orig_img = img_tensor.cpu().permute(1, 2, 0).numpy()
        orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min())
        
        mask = explanation.mask_probs.squeeze().cpu().numpy()
        
        # Col 0: Original
        axes[i, 0].imshow(orig_img[:,:,0], cmap='gray')
        axes[i, 0].set_title(f"Clinical X-Ray ({filename})", fontsize=12, fontweight='bold')
        axes[i, 0].axis('off')
        
        # Col 1: Pathological Decision Bottleneck
        axes[i, 1].imshow(orig_img[:,:,0], cmap='gray')
        overlay = axes[i, 1].imshow(mask, cmap='magma', alpha=0.5)
        axes[i, 1].contour(mask, levels=[0.5], colors='yellow', linewidths=1.5, alpha=0.8)
        axes[i, 1].set_title(f"{target_pathology} Bottleneck", fontsize=12, fontweight='bold')
        axes[i, 1].axis('off')
        
        # Col 2: Sufficient Clinical Representation
        masked_img = orig_img * mask[..., None]
        axes[i, 2].imshow(masked_img[:,:,0], cmap='gray')
        axes[i, 2].set_title(f"Sufficient Anatomy (F={explanation.fidelity_score:.2f})", fontsize=12, fontweight='bold')
        axes[i, 2].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = "experiments/results/visualizations/medical_interpretability.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Medical results saved to {save_path}")

if __name__ == "__main__":
    run_medical_benchmark()
