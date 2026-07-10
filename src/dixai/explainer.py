import torch
import torch.optim as optim
import torch.nn.functional as F
from typing import Optional, Dict

from .masks import GumbelSoftmaxMask
from .objective import DecisionInformationLoss
from .mine import MINETrainer


class DecisionInformationExplanation:
    """
    Container for the explanation results.
    """
    def __init__(self, mask: torch.Tensor, fidelity: float, sparsity: float, continuity: float, history: Optional[Dict] = None):
        self.mask = mask
        self.fidelity_score = fidelity
        self.sparsity_score = sparsity
        self.continuity_score = continuity
        self.history = history or {}

    def __repr__(self):
        return (f"DecisionInformationExplanation(Fidelity={self.fidelity_score:.3f}, "
                f"Sparsity={self.sparsity_score:.3f}, Continuity={self.continuity_score:.3f})")


class DecisionInformationExplainer:
    """
    Main explainer class implementing the Decision-Information eXplanation (DIxAI) framework.
    Optimizes a mask per-instance to find the minimal sufficient explanation.
    """

    def __init__(
        self,
        model,
        lambda_fidelity: float = 1.0,
        lambda_tv: float = 0.0,
        task: str = "classification",
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.lambda_fidelity = lambda_fidelity
        self.lambda_tv = lambda_tv
        self.task = task
        self.loss_fn = DecisionInformationLoss(
            lambda_fidelity=lambda_fidelity, task=task, lambda_tv=lambda_tv
        )

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        logits = out.logits if hasattr(out, "logits") else out
        return torch.log_softmax(logits, dim=-1)

    def explain(
        self,
        x: torch.Tensor,
        baseline: torch.Tensor,
        steps: int = 300,
        lr: float = 0.1,
        temperature: float = 2.0 / 3.0,
        init_logits: float = -2.0,
        anneal: bool = True,
        verbose: bool = False,
        seed: Optional[int] = None,
    ) -> DecisionInformationExplanation:
        if seed is not None:
            torch.manual_seed(seed)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        x = x.to(self.device)
        baseline = baseline.to(self.device)
        if x.dim() == baseline.dim():
            x_batched = x.unsqueeze(0) if x.dim() == 3 else x
        else:
            x_batched = x

        with torch.no_grad():
            y_orig = self._forward(x_batched)

        mask_module = GumbelSoftmaxMask(
            shape=x.shape, temperature=temperature, init_logits=init_logits
        ).to(self.device)

        optimizer = optim.Adam(mask_module.parameters(), lr=lr)

        temp_start, temp_end = temperature, 0.1
        final_lambda_fid = self.lambda_fidelity
        final_lambda_tv = self.lambda_tv

        history: Dict[str, list] = {"loss": [], "fidelity": [], "tv": []}

        for step in range(steps):
            optimizer.zero_grad()

            if anneal and steps > 1:
                tau = temp_start * (temp_end / temp_start) ** (step / (steps - 1))
                mask_module.temperature = tau
                self.loss_fn.lambda_fidelity = final_lambda_fid * (step / (steps - 1))
                self.loss_fn.lambda_tv = final_lambda_tv * (step / (steps - 1))

            z, mask = mask_module(x, baseline, training=True)
            z_batched = z.unsqueeze(0) if z.dim() == 3 else z
            y_masked = self._forward(z_batched)

            loss, fidelity_loss, tv_loss = self.loss_fn(mask, y_orig, y_masked)

            loss.backward()
            optimizer.step()

            history["loss"].append(loss.item())
            history["fidelity"].append(fidelity_loss.item())
            history["tv"].append(tv_loss.item())

            if verbose and step % 50 == 0:
                print(f"Step {step}: Loss={loss.item():.4f} Fid={fidelity_loss.item():.3f} "
                      f"TV={tv_loss.item():.3f} Temp={mask_module.temperature:.2f}")

        with torch.no_grad():
            z_final, mask_final = mask_module(x, baseline, training=False)
            z_final_batched = z_final.unsqueeze(0) if z_final.dim() == 3 else z_final
            y_final = self._forward(z_final_batched)
            fidelity = F.kl_div(y_final, y_orig, reduction="batchmean", log_target=True).item()
            sparsity = mask_final.mean().item()
            continuity = 1.0

        return DecisionInformationExplanation(
            mask=mask_final.cpu(),
            fidelity=fidelity,
            sparsity=sparsity,
            continuity=continuity,
            history=history,
        )