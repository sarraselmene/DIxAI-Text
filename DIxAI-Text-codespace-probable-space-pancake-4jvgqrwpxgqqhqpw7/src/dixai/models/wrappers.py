
import torch
import torch.nn as nn
import numpy as np

class ModelWrapper(nn.Module):
    """
    Wraps a black-box model to ensure compatibility with DIXAI.
    Ensures output is a torch tensor.
    """
    def __init__(self, model, task='classification'):
        super().__init__()
        self.model = model
        self.task = task
        
        # Determine if model is a torch module or sklearn-like
        self.is_torch = isinstance(model, nn.Module)

    def forward(self, x: torch.Tensor):
        if self.is_torch:
            out = self.model(x)
            if self.task == 'classification':
                # Ensure output is log-probabilities for KL-divergence stability
                # Most torch models return raw logits
                out = torch.log_softmax(out, dim=-1)
        else:
            # Assume numpy/sklearn model
            x_np = x.detach().cpu().numpy()
            if self.task == 'classification':
                # Expecting predict_proba for classification
                if hasattr(self.model, 'predict_proba'):
                    out_np = self.model.predict_proba(x_np)
                    # Convert probs to log_probs for numerical stability in KL loss
                    out = torch.tensor(out_np, dtype=torch.float32, device=x.device)
                    out = torch.log(out + 1e-10) 
                else:
                    raise ValueError("Model must implement predict_proba for classification")
            else:
                out_np = self.model.predict(x_np)
                out = torch.tensor(out_np, dtype=torch.float32, device=x.device)
        
        return out
