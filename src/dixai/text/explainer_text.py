"""
explainer_text.py — Explainer DIxAI-Text principal.

Assemble Verrou 1 + Verrou 2 + Verrou 3 dans une boucle d'optimisation.

Modèle attendu : HuggingFace *ForSequenceClassification (BERT, RoBERTa...).

FIXES appliqués :
  - Bug lambda : reset des lambdas au début de chaque explain()
    (l'annealing mutait self.loss_fn en place, causant des valeurs
    incorrectes sur les appels suivants avec anneal=False).
  - Clarification : content_mask vs attention_mask dans la loss.
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
from typing import Optional, Dict, List

from .masks_text import TokenGumbelSoftmaxMask
from .baselines_text import TextBaselineProvider, BaselineType
from .objective_text import DecisionInformationLossText


class TextExplanation:
    """
    Conteneur pour les résultats d'une explication DIxAI-Text.

    Attributs principaux :
        mask_probs     : (T,) probabilités continues dans [0,1] par token.
        mask           : alias de mask_probs (compatibilité scripts).
        tokens         : liste de strings (tokens décodés), longueur T.
        valid_mask     : (T,) content_mask binaire (1=contenu, 0=spécial/padding).
        predicted_class: int, classe prédite par le modèle sur le texte complet.
        fidelity_score : KL(f(x) || f(z)) final (plus bas = mieux).
        sparsity_score : taux moyen de tokens de contenu gardés (soft, dans [0,1]).
        contiguity_score: TV_seq final (plus bas = spans plus compacts).
        history        : dict avec les loss par step (pour debugging).
    """

    def __init__(
        self,
        mask_probs: torch.Tensor,
        tokens: List[str],
        fidelity: float,
        sparsity: float,
        contiguity: float,
        valid_mask: Optional[torch.Tensor] = None,
        predicted_class: Optional[int] = None,
        history: Optional[Dict[str, List[float]]] = None,
    ):
        self.mask_probs = mask_probs          # (T,) continu [0,1]
        self.mask = mask_probs                # alias pour compat scripts anciens
        self.tokens = tokens                  # liste de strings, longueur T
        self.fidelity_score = fidelity
        self.sparsity_score = sparsity
        self.contiguity_score = contiguity
        self.valid_mask = valid_mask          # (T,) binaire content tokens
        self.predicted_class = predicted_class  # int
        self.history = history or {}

    def top_tokens(self, threshold: float = 0.5) -> List[str]:
        """Retourne les tokens dont mask_probs > threshold."""
        return [
            tok for tok, p in zip(self.tokens, self.mask_probs.tolist())
            if p > threshold
        ]

    def selected_indices(self, threshold: float = 0.5) -> List[int]:
        """Retourne les indices des tokens sélectionnés (prob > threshold)."""
        return [
            i for i, p in enumerate(self.mask_probs.tolist())
            if p > threshold
        ]

    def top_k_tokens(self, k: int = 5) -> List[str]:
        """Retourne les k tokens avec les plus hautes probabilités."""
        probs_list = self.mask_probs.tolist()
        top_idx = sorted(range(len(probs_list)), key=lambda i: probs_list[i], reverse=True)[:k]
        return [self.tokens[i] for i in top_idx]

    def __repr__(self):
        return (
            f"TextExplanation("
            f"Fidelity={self.fidelity_score:.3f}, "
            f"Sparsity={self.sparsity_score:.3f}, "
            f"Contiguity={self.contiguity_score:.3f}, "
            f"PredClass={self.predicted_class})"
        )


class TextDecisionInformationExplainer:
    """
    Explainer DIxAI-Text pour modèles HuggingFace de classification de séquences.

    Usage:
        explainer = TextDecisionInformationExplainer(
            model, tokenizer,
            lambda_fidelity=10.0,
            lambda_contiguity=2.0,
            baseline_type=BaselineType.MASK_TOKEN,
            device="cpu",
        )
        explanation = explainer.explain("this movie was not very good", steps=300)
        print(explanation.top_tokens(threshold=0.5))
    """

    def __init__(
        self,
        model,
        tokenizer,
        lambda_fidelity: float = 10.0,
        lambda_contiguity: float = 1.0,
        baseline_type: str = BaselineType.MASK_TOKEN,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.lambda_fidelity = lambda_fidelity
        self.lambda_contiguity = lambda_contiguity
        self.baseline_type = baseline_type

        self.loss_fn = DecisionInformationLossText(
            lambda_fidelity=lambda_fidelity,
            task="classification",
            lambda_contiguity=lambda_contiguity,
        )
        self.baseline_provider = TextBaselineProvider(model, tokenizer, baseline_type)
        self.embedding_layer = model.get_input_embeddings()

    def _forward_from_embeds(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Passe avant depuis embeddings -> log_softmax(logits)."""
        out = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = out.logits if hasattr(out, "logits") else out
        return torch.log_softmax(logits, dim=-1)

    def explain(
        self,
        text: str,
        steps: int = 300,
        lr: float = 0.1,
        temperature: float = 2.0 / 3.0,
        init_logits: float = -2.0,
        anneal: bool = True,
        verbose: bool = False,
        seed: Optional[int] = None,
        fidelity_ramp_frac: float = 0.5,
        temp_end: float = 0.1,
    ) -> TextExplanation:
        """
        Optimise un masque par instance pour trouver le sous-ensemble minimal
        de tokens suffisant pour reproduire la décision du modèle.

        Args:
            text            : phrase à expliquer.
            steps           : nombre de pas d'optimisation Adam.
            lr              : learning rate Adam.
            temperature     : température initiale Gumbel-Sigmoid.
            init_logits     : logits initiaux du masque (négatif = sparse-first).
            anneal          : si True, annealing température + lambda sur l'entraînement.
            verbose         : si True, affiche la loss tous les 50 steps.
            seed            : graine pour reproductibilité.
            fidelity_ramp_frac: fraction de l'entraînement pour le ramp lambda (0.5 = 50%).
            temp_end        : température finale pour l'annealing.

        Returns:
            TextExplanation avec mask_probs, tokens, scores, valid_mask, predicted_class.
        """
        if seed is not None:
            torch.manual_seed(seed)

        # ==============================================================
        # FIX CRITICAL : Reset des lambdas AVANT chaque explication
        # L'annealing mutait self.loss_fn en place. Sans ce reset,
        # un appel explain(anneal=True) suivi de explain(anneal=False)
        # utiliserait les lambdas mutés au lieu des originaux.
        # ==============================================================
        self.loss_fn.lambda_fidelity = self.lambda_fidelity
        self.loss_fn.lambda_contiguity = self.lambda_contiguity

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # --- Tokenisation ---
        enc = self.tokenizer(text, return_tensors="pt", truncation=True)
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        T = input_ids.shape[1]

        # --- Masques structurels ---
        special_ids_list = self.tokenizer.get_special_tokens_mask(
            input_ids[0].tolist(), already_has_special_tokens=True
        )
        special_tokens_mask = torch.tensor(
            special_ids_list, device=self.device, dtype=attention_mask.dtype
        ).unsqueeze(0)  # (1, T)

        # content_mask : 1 sur tokens réels non-spéciaux, 0 ailleurs
        # C'est ce masque qu'on passe à la loss comme "attention_mask"
        # pour exclure padding + tokens spéciaux du calcul I(X;Z) et TV_seq
        content_mask = attention_mask * (1 - special_tokens_mask)  # (1, T)

        # --- Embeddings + prédiction originale ---
        with torch.no_grad():
            x_embeds = self.embedding_layer(input_ids)  # (1, T, D)
            y_orig = self._forward_from_embeds(x_embeds, attention_mask)
            # Classe prédite sur le texte complet
            predicted_class = int(y_orig.argmax(dim=-1).item())

        # --- Baseline ---
        baseline = self.baseline_provider.get_baseline(
            seq_len=T, device=self.device
        )  # (T, D)

        # --- Module de masque ---
        mask_module = TokenGumbelSoftmaxMask(
            seq_len=T,
            temperature=temperature,
            init_logits=init_logits,
        ).to(self.device)

        optimizer = optim.Adam(mask_module.parameters(), lr=lr)

        temp_start = temperature
        final_lambda_fid = self.lambda_fidelity
        final_lambda_con = self.lambda_contiguity

        history: Dict[str, List[float]] = {
            "loss": [], "info": [], "fidelity": [], "contiguity": [],
            "temperature": [], "lambda_fid": [], "lambda_con": [],
        }

        # --- Boucle d'optimisation ---
        for step in range(steps):
            optimizer.zero_grad()

            if anneal and steps > 1:
                progress = step / (steps - 1)

                # Annealing température (geometric decay)
                tau = temp_start * (temp_end / temp_start) ** progress
                mask_module.temperature = tau

                # Ramp lambda : atteint la valeur finale à fidelity_ramp_frac
                fid_progress = min(1.0, progress / fidelity_ramp_frac)
                current_lambda_fid = final_lambda_fid * fid_progress
                current_lambda_con = final_lambda_con * fid_progress

                # Mise à jour des lambdas de la loss
                self.loss_fn.lambda_fidelity = current_lambda_fid
                self.loss_fn.lambda_contiguity = current_lambda_con

            z_embeds, mask = mask_module(
                x_embeds, baseline,
                training=True,
                attention_mask=attention_mask,
                special_tokens_mask=special_tokens_mask,
            )
            y_masked = self._forward_from_embeds(z_embeds, attention_mask)

            # content_mask passé comme attention_mask à la loss
            # pour exclure les tokens non-contenu du calcul
            loss, info_loss, fidelity_loss, contiguity_loss = self.loss_fn(
                mask, y_orig, y_masked, attention_mask=content_mask
            )

            loss.backward()
            optimizer.step()

            # Historique
            history["loss"].append(loss.item())
            history["info"].append(info_loss.item())
            history["fidelity"].append(fidelity_loss.item())
            history["contiguity"].append(contiguity_loss.item())
            history["temperature"].append(mask_module.temperature)
            history["lambda_fid"].append(self.loss_fn.lambda_fidelity)
            history["lambda_con"].append(self.loss_fn.lambda_contiguity)

            if verbose and step % 50 == 0:
                print(
                    f"Step {step:3d}: Loss={loss.item():.4f} "
                    f"Info={info_loss.item():.3f} "
                    f"Fid={fidelity_loss.item():.3f} "
                    f"Contig={contiguity_loss.item():.3f} "
                    f"Temp={mask_module.temperature:.3f} "
                    f"λ_fid={self.loss_fn.lambda_fidelity:.1f} "
                    f"λ_con={self.loss_fn.lambda_contiguity:.1f}"
                )

        # --- Évaluation finale (déterministe, sans bruit Gumbel) ---
        with torch.no_grad():
            z_final, mask_final = mask_module(
                x_embeds, baseline,
                training=False,
                attention_mask=attention_mask,
                special_tokens_mask=special_tokens_mask,
            )
            final_probs = mask_final[0]  # (T,)

            # Vérification de cohérence
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
            if len(tokens) != T:
                raise RuntimeError(
                    f"Tokenization mismatch: len(tokens)={len(tokens)} vs T={T}"
                )

            y_final = self._forward_from_embeds(z_final, attention_mask)

            # Fidelity : KL(f(z) || f(x)) — plus bas = mieux
            fidelity = F.kl_div(
                y_final, y_orig,
                reduction="batchmean",
                log_target=True,
            ).item()

            # Soft sparsity : proportion moyenne de tokens de contenu gardés
            n_content = content_mask.sum().clamp(min=1).item()
            sparsity = (final_probs * content_mask[0]).sum().item() / n_content

            # Contiguity : TV_seq sur tokens de contenu (plus bas = spans compacts)
            contiguity = self.loss_fn._sequential_contiguity(
                mask_final, content_mask
            ).item()

            # valid_mask : content_mask (T,) pour scripts externes
            valid_mask_1d = content_mask[0].cpu()  # (T,)

        # ==============================================================
        # FIX : Reset des lambdas APRÈS l'explication aussi
        # pour que l'explainer soit dans un état cohérent
        # ==============================================================
        self.loss_fn.lambda_fidelity = self.lambda_fidelity
        self.loss_fn.lambda_contiguity = self.lambda_contiguity

        return TextExplanation(
            mask_probs=final_probs.cpu(),
            tokens=tokens,
            fidelity=fidelity,
            sparsity=sparsity,
            contiguity=contiguity,
            valid_mask=valid_mask_1d,
            predicted_class=predicted_class,
            history=history,
        )

    def explain_batch(
        self,
        texts: List[str],
        steps: int = 300,
        lr: float = 0.1,
        temperature: float = 2.0 / 3.0,
        init_logits: float = -2.0,
        anneal: bool = True,
        verbose: bool = False,
        seed: Optional[int] = None,
    ) -> List[TextExplanation]:
        """
        Explique un batch de textes séquentiellement.

        Args:
            texts : liste de phrases à expliquer.
            (autres args identiques à explain())

        Returns:
            Liste de TextExplanation.
        """
        explanations = []
        for i, text in enumerate(texts):
            if verbose:
                print(f"\n[{i+1}/{len(texts)}] Explaining: {text[:60]}...")
            exp = self.explain(
                text, steps=steps, lr=lr,
                temperature=temperature, init_logits=init_logits,
                anneal=anneal, verbose=verbose,
                seed=seed,
            )
            explanations.append(exp)
        return explanations