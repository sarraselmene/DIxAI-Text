
import sys
import os
import torch
import torch.nn as nn
import time
import pandas as pd
import numpy as np

# Try to import from installed package, otherwise fall back to src
try:
    from dixai import DecisionInformationExplainer
    from dixai.baselines import GradCAMPlusPlusExplainer, RISEExplainer
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
    from dixai import DecisionInformationExplainer
    from dixai.baselines import GradCAMPlusPlusExplainer, RISEExplainer

def measure_runtime(n_samples=5):
    # Use GPU if possible
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Dummy data for timing
    x = torch.randn(1, 3, 224, 224).to(device)
    model = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.Flatten(), nn.Linear(16*224*224, 2)).to(device)
    model.eval()
    
    results = {"Method": [], "Avg_Time_Sec": []}
    
    # Warmup
    _ = model(x)
    
    # 1. DIxAI (500 steps)
    explainer = DecisionInformationExplainer(model, device=device)
    start = time.time()
    for _ in range(n_samples):
        explainer.explain(x, steps=500)
    results["Method"].append("DIxAI (500 steps)")
    results["Avg_Time_Sec"].append((time.time() - start) / n_samples)
    
    # 2. RISE (1000 masks)
    rise = RISEExplainer(model, n_masks=1000, device=device)
    start = time.time()
    for _ in range(n_samples):
        rise.explain(x)
    results["Method"].append("RISE (1000 masks)")
    results["Avg_Time_Sec"].append((time.time() - start) / n_samples)
    
    # 3. Grad-CAM++
    target_layer = model[0]
    gcpp = GradCAMPlusPlusExplainer(model, target_layer, device=device)
    start = time.time()
    for _ in range(n_samples):
        gcpp.explain(x)
    results["Method"].append("Grad-CAM++")
    results["Avg_Time_Sec"].append((time.time() - start) / n_samples)

    df = pd.DataFrame(results)
    print("\n--- Runtime Comparison ---")
    print(df)
    
    output_path = os.path.join(os.path.dirname(__file__), '../results/runtime_comparison.csv')
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    measure_runtime()
