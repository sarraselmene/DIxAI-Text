"""
RISE: Randomized Input Sampling for Explanation of Black-box Models

Implementation based on:
Petsiuk, V., Das, A., & Saenko, K. (2018). RISE: Randomized Input Sampling for Explanation of Black-box Models.
British Machine Vision Conference (BMVC).

This module provides a baseline explainer for comparison with DIxAI.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class RISEExplainer:
    """
    RISE (Randomized Input Sampling for Explanation) explainer.
    
    Generates importance maps by probing the model with randomly masked versions
    of the input and weighting these masks by the model's response.
    
    Args:
        model: The black-box model to explain.
        n_masks: Number of random masks to generate.
        mask_prob: Probability of each cell being masked (1 = keep).
        cell_size: Size of each cell in the low-resolution mask grid.
        device: 'cpu' or 'cuda'.
    """
    
    def __init__(
        self,
        model,
        n_masks: int = 1000,
        mask_prob: float = 0.5,
        cell_size: int = 7,
        device: str = 'cpu'
    ):
        self.model = model
        self.n_masks = n_masks
        self.mask_prob = mask_prob
        self.cell_size = cell_size
        self.device = device
        self.model.eval()
    
    def _generate_masks(self, input_size: Tuple[int, int]) -> torch.Tensor:
        """
        Generate random binary masks at low resolution and upsample.
        
        Args:
            input_size: (H, W) of the input image.
            
        Returns:
            Tensor of shape (n_masks, 1, H, W) with values in [0, 1].
        """
        H, W = input_size
        # Low-resolution grid
        h = H // self.cell_size + 1
        w = W // self.cell_size + 1
        
        # Generate random binary masks
        masks = torch.bernoulli(
            torch.full((self.n_masks, 1, h, w), self.mask_prob)
        )
        
        # Upsample to input size with bilinear interpolation
        # Add random offset for better coverage
        masks = F.interpolate(masks, size=(H, W), mode='bilinear', align_corners=False)
        
        return masks.to(self.device)
    
    def explain(
        self,
        x: torch.Tensor,
        target_class: Optional[int] = None,
        batch_size: int = 100
    ) -> torch.Tensor:
        """
        Generate RISE saliency map for a single input.
        
        Args:
            x: Input tensor of shape (1, C, H, W) or (C, H, W).
            target_class: Class index to explain. If None, uses predicted class.
            batch_size: Batch size for processing masks.
            
        Returns:
            Saliency map of shape (H, W) with importance scores.
        """
        # Ensure batch dimension
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        x = x.to(self.device)
        _, C, H, W = x.shape
        
        # Get target class if not specified
        if target_class is None:
            with torch.no_grad():
                out = self.model(x)
                if out.dim() > 1:
                    target_class = out.argmax(dim=1).item()
                else:
                    target_class = 0
        
        # Generate masks
        masks = self._generate_masks((H, W))
        
        # Initialize saliency accumulator
        saliency = torch.zeros(H, W, device=self.device)
        weight_sum = torch.zeros(H, W, device=self.device)
        
        # Process in batches
        n_batches = (self.n_masks + batch_size - 1) // batch_size
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, self.n_masks)
            batch_masks = masks[start_idx:end_idx]
            
            # Apply masks to input (element-wise)
            # Expand x to match batch of masks
            x_masked = x * batch_masks  # (batch, C, H, W)
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(x_masked)
                
                # Handle different output formats
                if outputs.dim() > 1:
                    # Classification: get probability for target class
                    probs = F.softmax(outputs, dim=1)
                    scores = probs[:, target_class]
                else:
                    # Regression or single output
                    scores = outputs.squeeze()
            
            # Accumulate weighted saliency
            for j, score in enumerate(scores):
                saliency += score * batch_masks[j, 0]
                weight_sum += batch_masks[j, 0]
        
        # Normalize by the weight sum (avoids division by zero)
        saliency = saliency / (weight_sum + 1e-8)
        
        # Normalize to [0, 1]
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        
        return saliency
    
    def explain_batch(
        self,
        x_batch: torch.Tensor,
        target_classes: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate RISE saliency maps for a batch of inputs.
        
        Args:
            x_batch: Input tensor of shape (B, C, H, W).
            target_classes: Optional tensor of target classes for each input.
            
        Returns:
            Saliency maps of shape (B, H, W).
        """
        saliency_maps = []
        
        for i in range(x_batch.shape[0]):
            target = target_classes[i].item() if target_classes is not None else None
            saliency = self.explain(x_batch[i:i+1], target_class=target)
            saliency_maps.append(saliency)
        
        return torch.stack(saliency_maps)


class GradCAMPlusPlusExplainer:
    """
    Grad-CAM++ explainer for CNN models.
    
    An improved version of Grad-CAM that provides better localization
    for multiple instances of the same class.
    
    Args:
        model: The CNN model to explain.
        target_layer: The convolutional layer to use for CAM computation.
        device: 'cpu' or 'cuda'.
    """
    
    def __init__(self, model, target_layer, device: str = 'cpu'):
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def explain(
        self,
        x: torch.Tensor,
        target_class: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate Grad-CAM++ saliency map for a single input.
        
        Args:
            x: Input tensor of shape (1, C, H, W) or (C, H, W).
            target_class: Class index to explain. If None, uses predicted class.
            
        Returns:
            Saliency map of shape (H, W) with importance scores.
        """
        # Ensure batch dimension
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        x = x.to(self.device)
        x.requires_grad = True
        
        # Forward pass
        output = self.model(x)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass for target class
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Grad-CAM++ weighting
        gradients = self.gradients
        activations = self.activations
        
        # Compute alpha (weights for Grad-CAM++)
        grad_2 = gradients ** 2
        grad_3 = gradients ** 3
        
        alpha_num = grad_2
        alpha_denom = 2 * grad_2 + activations * grad_3 + 1e-8
        alpha = alpha_num / alpha_denom
        
        # Positive gradient weighting
        weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)
        
        # Weighted combination of activations
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Upsample to input size
        cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze()
        
        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam.detach()


# ==============================================================================
# Captum-based Explainers (requires: pip install captum)
# ==============================================================================

class IntegratedGradientsExplainer:
    """
    Integrated Gradients explainer using Captum library.
    
    Computes attributions by integrating gradients along the path
    from a baseline to the input.
    
    Args:
        model: The model to explain.
        device: 'cpu' or 'cuda'.
        n_steps: Number of steps for the integral approximation.
    """
    
    def __init__(self, model, device: str = 'cpu', n_steps: int = 50):
        self.model = model
        self.device = device
        self.n_steps = n_steps
        self._ig = None
        
        try:
            from captum.attr import IntegratedGradients
            self._ig = IntegratedGradients(model)
        except ImportError:
            raise ImportError("Captum is required for IntegratedGradients. Install with: pip install captum")
    
    def explain(
        self,
        x: torch.Tensor,
        target_class: Optional[int] = None,
        baseline: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate Integrated Gradients attribution map.
        
        Args:
            x: Input tensor of shape (1, C, H, W) or (C, H, W).
            target_class: Class index to explain. If None, uses predicted class.
            baseline: Baseline input. If None, uses zeros.
            
        Returns:
            Attribution map of shape (H, W).
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        x = x.to(self.device)
        
        if target_class is None:
            with torch.no_grad():
                out = self.model(x)
                target_class = out.argmax(dim=1).item()
        
        if baseline is None:
            baseline = torch.zeros_like(x)
        
        attributions = self._ig.attribute(
            x,
            baselines=baseline,
            target=target_class,
            n_steps=self.n_steps
        )
        
        # Sum across channels and take absolute value
        attr_map = attributions.squeeze().abs().sum(dim=0)
        
        # Normalize
        attr_map = (attr_map - attr_map.min()) / (attr_map.max() - attr_map.min() + 1e-8)
        
        return attr_map.detach()


class DeepLIFTExplainer:
    """
    DeepLIFT explainer using Captum library.
    
    Computes attributions by decomposing the output prediction
    in terms of contributions from each input feature.
    
    Args:
        model: The model to explain.
        device: 'cpu' or 'cuda'.
    """
    
    def __init__(self, model, device: str = 'cpu'):
        self.model = model
        self.device = device
        self._deeplift = None
        
        try:
            from captum.attr import DeepLift
            self._deeplift = DeepLift(model)
        except ImportError:
            raise ImportError("Captum is required for DeepLIFT. Install with: pip install captum")
    
    def explain(
        self,
        x: torch.Tensor,
        target_class: Optional[int] = None,
        baseline: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate DeepLIFT attribution map.
        
        Args:
            x: Input tensor of shape (1, C, H, W) or (C, H, W).
            target_class: Class index to explain. If None, uses predicted class.
            baseline: Baseline input. If None, uses zeros.
            
        Returns:
            Attribution map of shape (H, W).
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        x = x.to(self.device)
        
        if target_class is None:
            with torch.no_grad():
                out = self.model(x)
                target_class = out.argmax(dim=1).item()
        
        if baseline is None:
            baseline = torch.zeros_like(x)
        
        attributions = self._deeplift.attribute(
            x,
            baselines=baseline,
            target=target_class
        )
        
        # Sum across channels and take absolute value
        attr_map = attributions.squeeze().abs().sum(dim=0)
        
        # Normalize
        attr_map = (attr_map - attr_map.min()) / (attr_map.max() - attr_map.min() + 1e-8)
        
        return attr_map.detach()


# ==============================================================================
# Extended Baselines for Comprehensive Comparison
# ==============================================================================

class ExtremalPerturbationExplainer:
    """
    Extremal Perturbation explainer (Fong et al., ICCV 2019).
    
    Finds the smallest smooth mask that maximally preserves or destroys
    model predictions. Uses blur-based perturbation for smoother masks.
    
    Args:
        model: The model to explain.
        device: 'cpu' or 'cuda'.
        area_fraction: Target fraction of the image to preserve.
        n_steps: Number of optimization steps.
        lr: Learning rate.
    """
    
    def __init__(self, model, device='cpu', area_fraction=0.1, n_steps=200, lr=0.05):
        self.model = model
        self.device = device
        self.area_fraction = area_fraction
        self.n_steps = n_steps
        self.lr = lr
        self.model.eval()
    
    def explain(self, x, target_class=None):
        """Generate extremal perturbation mask."""
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.to(self.device)
        
        if target_class is None:
            with torch.no_grad():
                target_class = self.model(x).argmax(1).item()
        
        _, C, H, W = x.shape
        # Low-res learnable mask (smooth by construction)
        mask_logits = torch.zeros(1, 1, H // 4, W // 4,
                                  device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([mask_logits], lr=self.lr)
        
        for step in range(self.n_steps):
            mask = torch.sigmoid(mask_logits)
            mask_up = F.interpolate(mask, size=(H, W), mode='bilinear',
                                    align_corners=False)
            
            # Blurred perturbation
            blurred = self._gaussian_blur(x, kernel_size=11)
            perturbed = x * mask_up + blurred * (1 - mask_up)
            
            output = self.model(perturbed)
            prob = F.softmax(output, dim=1)[0, target_class]
            
            # Maximize preservation + area constraint
            area = mask.mean()
            loss = -prob + 10.0 * F.relu(area - self.area_fraction)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        final_mask = torch.sigmoid(mask_logits)
        final_mask = F.interpolate(final_mask, size=(H, W), mode='bilinear',
                                   align_corners=False)
        return final_mask.squeeze().detach()
    
    @staticmethod
    def _gaussian_blur(x, kernel_size=11, sigma=5.0):
        """Apply Gaussian blur as a perturbation function."""
        channels = x.shape[1]
        ax = torch.arange(kernel_size, dtype=torch.float32, device=x.device) - kernel_size // 2
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
        padding = kernel_size // 2
        return F.conv2d(x, kernel, padding=padding, groups=channels)


class IBAExplainer:
    """
    Information Bottleneck Attribution (Schulz et al., ICLR 2020).
    
    Injects a learnable information bottleneck at an intermediate layer,
    then optimizes the bottleneck to find the minimal information
    sufficient for the prediction.
    
    Args:
        model: The model to explain.
        target_layer: The layer to inject the bottleneck.
        device: 'cpu' or 'cuda'.
        beta: KL penalty weight.
        n_steps: Optimization steps.
        lr: Learning rate.
    """
    
    def __init__(self, model, target_layer, device='cpu',
                 beta=10.0, n_steps=100, lr=0.01):
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.beta = beta
        self.n_steps = n_steps
        self.lr = lr
        self.model.eval()
        self._stored_activation = None
        self._hook = None
    
    def explain(self, x, target_class=None):
        """Generate IBA attribution map."""
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.to(self.device)
        
        if target_class is None:
            with torch.no_grad():
                target_class = self.model(x).argmax(1).item()
        
        # Forward pass to get activation shape
        stored = {}
        def hook_fn(module, input, output):
            stored['act'] = output
        handle = self.target_layer.register_forward_hook(hook_fn)
        
        with torch.no_grad():
            self.model(x)
        act_shape = stored['act'].shape
        handle.remove()
        
        # Learnable noise mask (alpha controls information flow)
        alpha = torch.ones(act_shape, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([alpha], lr=self.lr)
        
        for step in range(self.n_steps):
            # Inject noise proportional to (1 - alpha)
            def inject_noise(module, input, output):
                noise = torch.randn_like(output)
                mask_val = torch.sigmoid(alpha)
                return output * mask_val + noise * (1 - mask_val) * output.std()
            
            handle = self.target_layer.register_forward_hook(inject_noise)
            
            out = self.model(x)
            prob = F.softmax(out, dim=1)[0, target_class]
            
            # KL penalty: encourage sparsity (most alpha → 0)
            mask_val = torch.sigmoid(alpha)
            kl = mask_val.mean()
            
            loss = -prob + self.beta * kl
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            handle.remove()
        
        # Attribution map from final alpha
        attr = torch.sigmoid(alpha).squeeze()
        if attr.dim() >= 2:
            # Pool spatial dimensions if needed, then upsample
            if attr.dim() == 3:
                attr = attr.mean(0)
            attr = F.interpolate(attr.unsqueeze(0).unsqueeze(0),
                                 size=x.shape[2:], mode='bilinear',
                                 align_corners=False).squeeze()
        
        # Normalize
        attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)
        return attr.detach()


class INVASEExplainer:
    """
    INVASE-style Instance-Wise Variable Selection (Yoon et al., ICLR 2019).
    
    Simplified vision adaptation: learns a per-instance binary selector
    that identifies the minimal feature subset sufficient for prediction.
    
    For vision inputs, operates on a spatial grid of patches.
    
    Args:
        model: The model to explain.
        device: 'cpu' or 'cuda'.
        patch_size: Size of spatial patches for selection.
        n_steps: Optimization steps.
        lr: Learning rate.
        sparsity_weight: L1 penalty for selector sparsity.
    """
    
    def __init__(self, model, device='cpu', patch_size=16,
                 n_steps=200, lr=0.01, sparsity_weight=0.5):
        self.model = model
        self.device = device
        self.patch_size = patch_size
        self.n_steps = n_steps
        self.lr = lr
        self.sparsity_weight = sparsity_weight
        self.model.eval()
    
    def explain(self, x, target_class=None):
        """Generate INVASE-style attribution map."""
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.to(self.device)
        
        if target_class is None:
            with torch.no_grad():
                target_class = self.model(x).argmax(1).item()
        
        _, C, H, W = x.shape
        gh = H // self.patch_size
        gw = W // self.patch_size
        
        # Learnable selection logits (per patch)
        selector_logits = torch.zeros(1, 1, gh, gw,
                                       device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([selector_logits], lr=self.lr)
        
        baseline = torch.zeros_like(x)
        
        with torch.no_grad():
            orig_out = F.softmax(self.model(x), dim=1)
        
        for step in range(self.n_steps):
            # Gumbel-Softmax selection
            temp = max(0.5, 2.0 * (1 - step / self.n_steps))
            u = torch.rand_like(selector_logits).clamp(1e-6, 1 - 1e-6)
            gumbel = -torch.log(-torch.log(u))
            mask_soft = torch.sigmoid((selector_logits + gumbel) / temp)
            
            # Upsample to full resolution
            mask_up = F.interpolate(mask_soft, size=(H, W), mode='bilinear',
                                    align_corners=False)
            
            # Apply mask
            z = x * mask_up + baseline * (1 - mask_up)
            out = F.softmax(self.model(z), dim=1)
            
            # Prediction loss + sparsity
            pred_loss = F.kl_div(out.log(), orig_out, reduction='batchmean')
            sparsity_loss = mask_soft.mean()
            
            loss = pred_loss + self.sparsity_weight * sparsity_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Final mask
        final_mask = torch.sigmoid(selector_logits)
        final_mask = F.interpolate(final_mask, size=(H, W), mode='bilinear',
                                   align_corners=False)
        attr = final_mask.squeeze()
        attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)
        return attr.detach()

