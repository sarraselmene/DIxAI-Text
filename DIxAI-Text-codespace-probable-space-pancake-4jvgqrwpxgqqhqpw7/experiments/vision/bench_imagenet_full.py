import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd
import numpy as np
from dixai import DecisionInformationExplainer
import time

def run_quantitative_benchmark(limit=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Quantitative ImageNet Benchmark (N={limit}) on {device}...")
    
    # 1. Load Model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device)
    model.eval()
    
    # 2. Setup Data
    sample_dir = "data/imagenet_samples"
    if not os.path.exists(sample_dir):
        print("Error: Samples not found. Run download_imagenet_samples.py first.")
        return
        
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    explainer = DecisionInformationExplainer(
        model=model,
        lambda_fidelity=50.0,
        task='classification',
        device=device
    )
    
    results = []
    
    sample_files = sorted([f for f in os.listdir(sample_dir) if f.endswith(".jpg")])[:limit]
    
    for i, filename in enumerate(sample_files):
        path = os.path.join(sample_dir, filename)
        img = Image.open(path).convert('RGB')
        img_tensor = transform(img).to(device)
        
        start_time = time.time()
        # Generate DIxAI Explanation
        explanation = explainer.explain(
            img_tensor,
            steps=500,
            downsample_factor=7, # 32x32 grid for speed
            use_spatial_prior=True,
            anneal=True,
            verbose=False
        )
        latency = time.time() - start_time
        
        # Log metrics
        results.append({
            "filename": filename,
            "fidelity": explanation.fidelity_score,
            "sparsity": explanation.info_score,
            "latency": latency
        })
        
        if (i+1) % 10 == 0:
            avg_fid = np.mean([r['fidelity'] for r in results])
            print(f"Processed {i+1}/{len(sample_files)} | Avg Fidelity: {avg_fid:.4f}")

    # 3. Save Results
    df = pd.DataFrame(results)
    output_path = "experiments/results/imagenet_quantitative_dixai.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print("\nBenchmark Complete!")
    print(f"Mean Fidelity: {df['fidelity'].mean():.4f}")
    print(f"Mean Sparsity: {df['sparsity'].mean():.4f}")
    print(f"Mean Latency:  {df['latency'].mean():.4f}s")
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    run_quantitative_benchmark(limit=100) # Start with 100 for verification, can be extended to 1000
