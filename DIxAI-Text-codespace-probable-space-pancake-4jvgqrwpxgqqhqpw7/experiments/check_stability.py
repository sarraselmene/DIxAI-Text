
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))
sys.path.append(os.getcwd())

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from dixai import DecisionInformationExplainer

def check_stability():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Checking Stability on {device}...")
    
    # Enforce Determinism
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device)
    model.eval()
    
    explainer = DecisionInformationExplainer(model=model, lambda_fidelity=50.0, device=device)
    
    # Load one sample
    sample_path = "data/imagenet_samples/sample_0000.jpg"
    img = Image.open(sample_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    x = transform(img).to(device)
    
    masks = []
    print("Running 3 passes on same input with SEED=42...")
    for i in range(3):
        # We also need to check if loss is identical.
        # But explain() doesn't return loss history in easy way unless we modify it.
        # But explain() returns explanation object which has history.
        exp = explainer.explain(x, steps=300, verbose=False, seed=42)
        masks.append(exp.mask)
        print(f"Pass {i}: Fidelity={exp.fidelity_score:.2f}, Sparsity={exp.info_score:.2f}, FinalLoss={exp.history['loss'][-1]:.4f}")
        
    # Calculate pairwise IoU
    iou_12 = (masks[0] * masks[1]).sum() / ((masks[0] + masks[1]).sum() - (masks[0] * masks[1]).sum())
    iou_23 = (masks[1] * masks[2]).sum() / ((masks[1] + masks[2]).sum() - (masks[1] * masks[2]).sum())
    
    print(f"IoU (Pass 1 vs 2): {iou_12:.4f}")
    print(f"IoU (Pass 2 vs 3): {iou_23:.4f}")

if __name__ == "__main__":
    check_stability()
