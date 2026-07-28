"""
Assemble Verrou 1 + Verrou 2 + Verrou 3 dans une boucle d'optimisation.

Modèle attendu : HuggingFace *ForSequenceClassification (BERT, RoBERTa...).
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
from typing import Optional, Dict

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
    """

    def __init__(
        self,
        mask_probs: torch.Tensor,
        tokens,
        fidelity: float,
        sparsity: float,
        contiguity: float,
        valid_mask: Optional[torch.Tensor] = None,
        predicted_class: Optional[int] = None,
    ):
        self.mask_probs = mask_probs          # (T,) continu [0,1]
        self.mask = mask_probs                # alias pour compat scripts anciens
        self.tokens = tokens                  # liste de strings, longueur T
        self.fidelity_score = fidelity
        self.sparsity_score = sparsity
        self.contiguity_score = contiguity
        self.valid_mask = valid_mask          # (T,) binaire content tokens
        self.predicted_class = predicted_class  # int

    def top_tokens(self, threshold: float = 0.5):
        """Retourne les tokens dont mask_probs > threshold."""
        return [
            tok for tok, p in zip(self.tokens, self.mask_probs.tolist())
            if p > threshold
        ]

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
    ) -> TextExplanation:
        """
        Optimise un masque par instance pour trouver le sous-ensemble minimal
        de tokens suffisant pour reproduire la décision du modèle.

        Args:
            text       : phrase à expliquer.
            steps      : nombre de pas d'optimisation Adam.
            lr         : learning rate Adam.
            temperature: température initiale Gumbel-Sigmoid.
            init_logits: logits initiaux du masque (négatif = sparse-first).
            anneal     : si True, annealing température + lambda sur l'entraînement.
            verbose    : si True, affiche la loss tous les 50 steps.
            seed       : graine pour reproductibilité.

        Returns:
            TextExplanation avec mask_probs, tokens, scores, valid_mask, predicted_class.
        """
        if seed is not None:
            torch.manual_seed(seed)

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

        temp_start, temp_end = temperature, 0.1
        final_lambda_fid = self.lambda_fidelity
        final_lambda_con = self.lambda_contiguity
        # Ramp rapide : lambda_fidelity atteint sa valeur finale à 50% de l'entraînement
        fidelity_ramp_frac = 0.5

        history: Dict[str, list] = {
            "loss": [], "info": [], "fidelity": [], "contiguity": []
        }

        # --- Boucle d'optimisation ---
        for step in range(steps):
            optimizer.zero_grad()

            if anneal and steps > 1:
                progress = step / (steps - 1)
                # Annealing température
                tau = temp_start * (temp_end / temp_start) ** progress
                mask_module.temperature = tau
                # Ramp lambda fidelity/contiguity
                fid_progress = min(1.0, progress / fidelity_ramp_frac)
                self.loss_fn.lambda_fidelity = final_lambda_fid * fid_progress
                self.loss_fn.lambda_contiguity = final_lambda_con * fid_progress

            z_embeds, mask = mask_module(
                x_embeds, baseline,
                training=True,
                attention_mask=attention_mask,
                special_tokens_mask=special_tokens_mask,
            )
            y_masked = self._forward_from_embeds(z_embeds, attention_mask)

            loss, info_loss, fidelity_loss, contiguity_loss = self.loss_fn(
                mask, y_orig, y_masked, attention_mask=content_mask
            )

            loss.backward()
            optimizer.step()

            history["loss"].append(loss.item())
            history["info"].append(info_loss.item())
            history["fidelity"].append(fidelity_loss.item())
            history["contiguity"].append(contiguity_loss.item())

            if verbose and step % 50 == 0:
                print(
                    f"Step {step:3d}: Loss={loss.item():.4f} "
                    f"Info={info_loss.item():.3f} "
                    f"Fid={fidelity_loss.item():.3f} "
                    f"Contig={contiguity_loss.item():.3f} "
                    f"Temp={mask_module.temperature:.3f}"
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

        return TextExplanation(
            mask_probs=final_probs.cpu(),
            tokens=tokens,
            fidelity=fidelity,
            sparsity=sparsity,
            contiguity=contiguity,
            valid_mask=valid_mask_1d,
            predicted_class=predicted_class,
        )
