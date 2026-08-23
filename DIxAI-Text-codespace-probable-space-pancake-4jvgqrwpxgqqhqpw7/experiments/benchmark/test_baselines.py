"""
Minimal test to validate baseline integrations.
"""
import torch
import torch.nn as nn
import torchvision.models as models
import sys
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from dixai import DecisionInformationExplainer

print("="*50)
print("Testing DIxAI + Baselines Integration")
print("="*50)

# Simple model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(512, 10)
model.eval()

# Test input
x = torch.randn(1, 3, 32, 32)
print(f"Input shape: {x.shape}")

# Test DIxAI
print("\n1. Testing DIxAI...")
try:
    explainer = DecisionInformationExplainer(model, lambda_fidelity=10.0)
    result = explainer.explain(x, steps=50, temperature=0.67)
    print(f"   Mask probs shape: {result['mask_probs'].shape}")
    print("   ✓ DIxAI works!")
except Exception as e:
    print(f"   ✗ DIxAI error: {e}")

# Test Saliency
print("\n2. Testing Gradient Saliency...")
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("baselines", Path(__file__).parent / "baselines.py")
    baselines = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baselines)
    
    saliency = baselines.GradientSaliency(model)
    attr = saliency.explain(x, target=0)
    print(f"   Attribution shape: {attr.shape}")
    print("   ✓ Saliency works!")
except Exception as e:
    print(f"   ✗ Saliency error: {e}")

# Test RISE
print("\n3. Testing RISE...")
try:
    rise = baselines.RISEBaseline(model, n_masks=100, cell_size=4)
    saliency_map = rise.explain(x.squeeze(0), target=0)
    print(f"   Saliency shape: {saliency_map.shape}")
    print("   ✓ RISE works!")
except Exception as e:
    print(f"   ✗ RISE error: {e}")

# Test DeepLIFT
print("\n4. Testing DeepLIFT...")
try:
    deeplift = baselines.DeepLIFTBaseline(model)
    attr = deeplift.explain(x, target=0)
    print(f"   Attribution shape: {attr.shape}")
    print("   ✓ DeepLIFT works!")
except Exception as e:
    print(f"   ✗ DeepLIFT error: {e}")

# Test Insertion/Deletion AUC
print("\n5. Testing Insertion/Deletion AUC...")
try:
    spec2 = importlib.util.spec_from_file_location("metrics", Path(__file__).parent / "metrics.py")
    metrics = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(metrics)
    
    fake_importance = torch.rand(3, 32, 32)
    ins_auc = metrics.insertion_auc(model, x.squeeze(0), fake_importance, target=0, steps=5)
    del_auc = metrics.deletion_auc(model, x.squeeze(0), fake_importance, target=0, steps=5)
    print(f"   Insertion AUC: {ins_auc:.4f}")
    print(f"   Deletion AUC: {del_auc:.4f}")
    print("   ✓ Metrics work!")
except Exception as e:
    print(f"   ✗ Metrics error: {e}")

print("\n" + "="*50)
print("Test Complete!")
print("="*50)
