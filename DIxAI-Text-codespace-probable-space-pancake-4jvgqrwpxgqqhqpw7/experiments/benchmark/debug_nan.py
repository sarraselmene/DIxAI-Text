
import torch
import torch.nn as nn
import torchvision.models as models
import sys
from pathlib import Path
import importlib.util

# Setup paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from dixai import DecisionInformationExplainer

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model():
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 10)
    def fix_relu(module):
        if isinstance(module, nn.ReLU):
            module.inplace = False
    model.apply(fix_relu)
    model.eval()
    return model.to(DEVICE)

model = get_model()
x = torch.randn(1, 3, 32, 32).to(DEVICE)
target = 0

print("--- Debugging DeepLIFT ---")
try:
    from captum.attr import DeepLift
    dl = DeepLift(model)
    attr = dl.attribute(x, target=target)
    print(f"DeepLIFT attr shape: {attr.shape}")
    print(f"DeepLIFT attr sum: {attr.sum().item()}")
    print(f"DeepLIFT attr contains nan: {torch.isnan(attr).any().item()}")
except Exception as e:
    import traceback
    print(f"DeepLIFT Error: {e}")
    traceback.print_exc()

print("\n--- Debugging DIxAI ---")
try:
    dixai = DecisionInformationExplainer(model, lambda_fidelity=20.0, device=DEVICE)
    # Using small steps for quick debug
    result = dixai.explain(x, steps=50, temperature=0.5, lr=0.05, verbose=True)
    print(f"DIxAI result keys: {result.__dict__.keys()}")
    print(f"DIxAI mask_probs shape: {result.mask_probs.shape}")
    print(f"DIxAI mask_probs contains nan: {torch.isnan(result.mask_probs).any().item()}")
    print(f"DIxAI mask_probs range: {result.mask_probs.min().item():.4f} to {result.mask_probs.max().item():.4f}")
    
    # Test insertion AUC
    spec = importlib.util.spec_from_file_location("metrics", Path(__file__).parent / "metrics.py")
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    
    saliency = result.mask_probs.squeeze().detach().cpu().numpy()
    auc = metrics.insertion_auc(model, x.squeeze(0).cpu(), saliency, target=target, steps=5)
    print(f"DIxAI Insertion AUC: {auc}")
    
except Exception as e:
    import traceback
    print(f"DIxAI Error: {e}")
    traceback.print_exc()
