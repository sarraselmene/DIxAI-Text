import torch
from dixai.explainer import DecisionInformationExplainer
from experiments.benchmark.metrics import decision_fidelity, sparsity
from experiments.benchmark.models import SimpleMLP

def debug_shapes():
    input_dim = 4
    num_classes = 3
    model = SimpleMLP(input_dim=input_dim, output_dim=num_classes)
    
    x = torch.randn(input_dim)
    explainer = DecisionInformationExplainer(model, lambda_fidelity=1.0)
    
    explanation = explainer.explain(x, steps=5)
    mask = explanation.mask
    
    print(f"DEBUG: x.shape={x.shape}")
    print(f"DEBUG: mask.shape={mask.shape}")
    
    # Try manual forward
    try:
        y_orig = model(x.unsqueeze(0))
        print(f"DEBUG: y_orig.shape={y_orig.shape}")
        
        masked_x = x.unsqueeze(0) * mask
        print(f"DEBUG: masked_x.shape={masked_x.shape}")
        
        y_masked = model(masked_x)
        print(f"DEBUG: y_masked.shape={y_masked.shape}")
    except Exception as e:
        print(f"DEBUG: FAILED manual forward: {e}")
        
    try:
        fid = decision_fidelity(model, x, mask)
        print(f"DEBUG: fidelity={fid}")
    except Exception as e:
        print(f"DEBUG: FAILED decision_fidelity: {e}")

if __name__ == "__main__":
    debug_shapes()
