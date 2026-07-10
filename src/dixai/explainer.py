
import torch
import torch.optim as optim
from typing import Optional, Dict

from .masks import GumbelSoftmaxMask
from .objective import DecisionInformationLoss
from .models.wrappers import ModelWrapper
from .metrics import calculate_fidelity, calculate_sparsity

class DecisionInformationExplanation:
    def __init__(self, mask: torch.Tensor, mask_probs: torch.Tensor, fidelity: float, info_score: float):
        self.mask = mask
        self.mask_probs = mask_probs # Continuous probabilities for ranking
        self.fidelity_score = fidelity
        self.info_score = info_score 
    
    def __repr__(self):
        return f"Explanation(Fidelity={self.fidelity_score:.3f}, InfoRetained={self.info_score:.3f})"

class DecisionInformationExplainer:
    """
    Explainer implementing Decision-Information Conservation.
    """
    def __init__(self, model, lambda_fidelity: float = 10.0, task: str = 'classification', device: str = 'cpu', lambda_tv: float = 0.0):
        """
        Args:
            model: The black-box model to explain.
            lambda_fidelity: Weight for the decision preservation objective.
            task: 'classification' or 'regression'.
            device: 'cpu' or 'cuda'.
            lambda_tv: Weight for spatial smoothness (TV loss).
        """
        self.device = torch.device(device)
        self.model = ModelWrapper(model, task)
        self.lambda_fidelity = lambda_fidelity
        self.lambda_tv = lambda_tv
        self.task = task
        self.loss_fn = DecisionInformationLoss(lambda_fidelity, task, lambda_tv)

    def explain(self, x: torch.Tensor, steps: int = 500, lr: float = 0.1, temperature: float = 2.0/3.0, init_logits: float = -2.0, baseline: Optional[torch.Tensor] = None, downsample_factor: int = 1, use_spatial_prior: bool = False, anneal: bool = False, verbose: bool = False, use_mine: bool = False, seed: Optional[int] = None) -> DecisionInformationExplanation:
        """
        Generates an explanation for a single instance x.
        
        Args:
            x: Input tensor for single instance. 
            steps: Optimization steps.
            lr: Learning rate.
            init_logits: Initial value for mask logits.
            baseline: Reference value for masked-out pixels.
            downsample_factor: Factor for multi-scale masking.
            use_spatial_prior: Use center-weighted bias.
            anneal: Gradually decrease temperature and increase lambda.
            verbose: Print progress.
            use_mine: Use MINE estimator instead of L1 sparsity.
            seed: Random seed for reproducibility.
        """
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                
        # Support both (C, H, W) and (B, C, H, W)
        if x.dim() == 4:
            batch_x = x.to(self.device)
            input_shape = x.shape[1:]
        elif x.dim() == 3:
            batch_x = x.unsqueeze(0).to(self.device)
            input_shape = x.shape
        elif x.dim() == 2: # (B, Dim)
            batch_x = x.to(self.device)
            input_shape = x.shape[1]
        elif x.dim() == 1: # (Dim)
            batch_x = x.unsqueeze(0).to(self.device)
            input_shape = x.shape[0]
        else:
            batch_x = x.to(self.device)
            input_shape = x.shape
            
        # Freeze model
        if isinstance(self.model.model, torch.nn.Module):
            self.model.model.eval()
            for param in self.model.model.parameters():
                param.requires_grad = False
                
        # Get original prediction (Target)
        with torch.no_grad():
            y_orig = self.model(batch_x)
            
        # Initialize Mask with multi-scale support
        mask_module = GumbelSoftmaxMask(
            input_shape, 
            temperature=temperature,
            init_logits=init_logits, 
            downsample_factor=downsample_factor,
            use_spatial_prior=use_spatial_prior
        ).to(self.device)

        optimizer = optim.Adam(mask_module.parameters(), lr=lr)
        
        # MINE Setup
        mine_estimator = None
        if use_mine:
            from .mine import MINETrainer
            x_dim = batch_x.view(batch_x.size(0), -1).size(1)
            mine_trainer = MINETrainer(x_dim, x_dim)
            mine_estimator = mine_trainer.model.to(self.device)
        
        # Annealing setup
        temp_start, temp_end = 2.0/3.0, 0.1
        final_lambda = self.lambda_fidelity
        
        history = {"loss": [], "info": [], "fidelity": []}
        
        # Optimization Loop
        for step in range(steps):
            optimizer.zero_grad()
            
            # Annealing
            if anneal:
                tau = temp_start * (temp_end / temp_start) ** (step / steps)
                mask_module.temperature = tau
                self.loss_fn.lambda_fidelity = final_lambda * (step / steps)
            
            masked_x, mask = mask_module(batch_x, training=True, baseline=baseline)
            
            if isinstance(input_shape, tuple) and len(input_shape) >= 3 and masked_x.dim() == len(input_shape) + 2:
                masked_x = masked_x.reshape(1, *input_shape) 

            y_masked = self.model(masked_x)
            
            # Train MINE if used
            if use_mine:
                x_flat = batch_x.view(batch_x.size(0), -1).detach()
                z_flat = (batch_x * mask).view(batch_x.size(0), -1).detach()
                # Run multiple MINE steps per explainer step for stability
                for _ in range(5):
                    mine_trainer.step(x_flat, z_flat)

            loss, info_loss, fidelity_loss = self.loss_fn(mask, y_orig, y_masked, batch_x, mine_estimator=mine_estimator)

            loss.backward()
            optimizer.step()
            
            history["loss"].append(loss.item())
            history["info"].append(info_loss.item())
            history["fidelity"].append(fidelity_loss.item())
            
            if verbose and step % 100 == 0:
                print(f"Step {step}: Loss={loss.item():.4f}, Info={info_loss.item():.2f}, Fid={fidelity_loss.item():.2f}, Temp={mask_module.temperature:.2f}")

            # Early Stopping Check
            if step > 200 and len(history["loss"]) > 20:
                recent_loss = sum(history["loss"][-20:]) / 20
                prev_loss = sum(history["loss"][-40:-20]) / 20
                if abs(recent_loss - prev_loss) < 1e-4:
                    if verbose:
                        print(f"Early stopping at step {step} due to convergence.")
                    break
        
        # Restore lambda_fidelity
        self.loss_fn.lambda_fidelity = final_lambda
                
        with torch.no_grad():
            masked_x_final, final_mask = mask_module(batch_x, training=False, baseline=baseline)
            mask_probs = mask_module.get_mask_probs(batch_x).cpu()
            
            if isinstance(input_shape, tuple) and len(input_shape) >= 3 and masked_x_final.dim() == len(input_shape) + 2:
                masked_x_final = masked_x_final.reshape(1, *input_shape)
                
            y_masked_final = self.model(masked_x_final)
            fidelity = calculate_fidelity(y_orig, y_masked_final, self.task)
            density = final_mask.mean().item()
            
        explanation = DecisionInformationExplanation(
            mask=final_mask.cpu(),
            mask_probs=mask_probs,
            fidelity=fidelity,
            info_score=density
        )
        explanation.history = history
        return explanation

    def explain_batch(self, x_batch: torch.Tensor, steps: int = 500, lr: float = 0.1, verbose: bool = False):
        """
        Generates explanations for a batch of instances.
        Currently implements a loop, can be parallelized in future.
        """
        explanations = []
        for i in range(len(x_batch)):
            if verbose:
                print(f"Explaining {i+1}/{len(x_batch)}...")
            explanations.append(self.explain(x_batch[i], steps, lr, verbose=False))
        return explanations

    def sweep_lambda(self, x: torch.Tensor, lambdas: list, steps: int = 500, verbose: bool = False):
        """
        Sweeps the lambda_fidelity parameter to show the trade-off.
        """
        original_lambda = self.lambda_fidelity
        results = []
        
        for l in lambdas:
            self.lambda_fidelity = l
            self.loss_fn.lambda_fidelity = l # Update loss fn too
            self.loss_fn.lambda_tv = self.lambda_tv # Ensure TV is preserved
            
            exp = self.explain(x, steps, verbose=verbose)
            results.append({"lambda": l, "explanation": exp})
            
        self.lambda_fidelity = original_lambda
        self.loss_fn.lambda_fidelity = original_lambda
        return results
    
    def explain_text(
        self,
        texts,
        bert_wrapper,
        sentiment_model,
        steps: int = 300,
        lr: float = 0.01,
        temperature: float = 2.0,
        init_logits: float = 2.0,
        baseline_mode: str = "mask_token",
        anneal: bool = True,
        verbose: bool = False,
        seed: int = None
    ) -> "DecisionInformationExplanation":
        from .nlp.token_mask import GumbelSoftmaxTokenMask

        if seed is not None:
            torch.manual_seed(seed)

        # ── 1. Embeddings AVANT les 12 couches (pré-encodeur) ────────────────
        
        embeddings, attention_mask, input_ids, token_type_ids = bert_wrapper.get_input_embeddings(texts)
        embeddings = embeddings.to(self.device)
        attention_mask = attention_mask.to(self.device)
        B, T, D = embeddings.shape

        # ── 2. Baseline, MÊME étage pré-encodeur ─────────────────────────────
        if baseline_mode == "mask_token":
            mask_id = bert_wrapper.tokenizer.mask_token_id
            mask_ids = torch.full((B, T), mask_id, device=self.device)
            with torch.no_grad():
                baseline = bert_wrapper.model.embeddings(input_ids=mask_ids)
        else:
            baseline = torch.zeros_like(embeddings)

        # ── 3. Prédiction originale — modèle complet (12 couches + tête) ─────
        with torch.no_grad():
            y_orig = sentiment_model.forward_from_embeddings(embeddings, attention_mask)

        # ── 4. TokenMask ──────────────────────────────────────────────────────
        token_mask_module = GumbelSoftmaxTokenMask(
            seq_len=T, init_logits=init_logits, temperature=temperature
        ).to(self.device)
        optimizer = torch.optim.Adam(token_mask_module.parameters(), lr=lr)
        history = {"loss": [], "info": [], "fidelity": []}

        # ── 5. Boucle d'optimisation ──────────────────────────────────────────
        for step in range(steps):
            optimizer.zero_grad()

            if anneal:
                tau = max(2.0 / 3.0, temperature * (0.995 ** step))
                token_mask_module.temperature = tau

            masked_emb, t_mask = token_mask_module(
                embeddings, baseline, training=True, attention_mask=attention_mask
            )

            y_masked = sentiment_model.forward_from_embeddings(masked_emb, attention_mask)

            loss, info_loss, fidelity_loss = self.loss_fn(
                t_mask, y_orig, y_masked, modality="text"
            )
            loss.backward()
            optimizer.step()

            history["loss"].append(loss.item())
            history["info"].append(info_loss.item())
            history["fidelity"].append(fidelity_loss.item())

            if verbose and step % 50 == 0:
                print(f"[Text {step:3d}] loss={loss.item():.4f} "
                    f"info={info_loss.item():.3f} fidelity={fidelity_loss.item():.3f} "
                    f"τ={token_mask_module.temperature:.3f}")

            if step > 100 and len(history["loss"]) > 20:
                recent = sum(history["loss"][-20:]) / 20
                prev = sum(history["loss"][-40:-20]) / 20
                if abs(recent - prev) < 1e-5:
                    if verbose:
                        print(f"Early stopping at step {step}")
                    break

        # ── 6. Masque final (une seule évaluation, réutilisée) ───────────────
        with torch.no_grad():
            masked_emb_final, final_token_mask = token_mask_module(
                embeddings, baseline, training=False, attention_mask=attention_mask
            )
            mask_probs = token_mask_module.get_mask_probs()

            y_final = sentiment_model.forward_from_embeddings(masked_emb_final, attention_mask)
            fidelity = calculate_fidelity(y_orig, y_final, self.task)
            density = final_token_mask.mean().item()

        explanation = DecisionInformationExplanation(
            mask=final_token_mask.cpu(),
            mask_probs=mask_probs.cpu(),
            fidelity=fidelity,
            info_score=density
        )
        explanation.history = history
        return explanation