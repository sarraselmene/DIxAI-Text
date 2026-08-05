"""
Assemble Verrou 1 + Verrou 2 + Verrou 3 dans une boucle d'optimisation, sur le même
principe que `dixai.explainer.DecisionInformationExplainer` (optimisation par instance,
Adam sur les logits du mask, un mask_module ré-initialisé à chaque appel de `explain`).

Le modèle attendu est un modèle HuggingFace `*ForSequenceClassification` (BERT, RoBERTa...),
appelé via `model(inputs_embeds=..., attention_mask=...)`.
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
from typing import Optional, Dict

from .masks_text import TokenGumbelSoftmaxMask
from .baselines_text import TextBaselineProvider, BaselineType
from .objective_text import DecisionInformationLossText


class TextExplanation:
    def __init__(self, mask_probs: torch.Tensor, tokens, fidelity: float, sparsity: float, contiguity: float):
        self.mask_probs = mask_probs  # (T,)
        self.tokens = tokens          # liste de strings (décodage tokenizer), longueur T
        self.fidelity_score = fidelity
        self.sparsity_score = sparsity
        self.contiguity_score = contiguity

    def top_tokens(self, threshold: float = 0.5):
        return [tok for tok, p in zip(self.tokens, self.mask_probs.tolist()) if p > threshold]

    def __repr__(self):
        return (
            f"TextExplanation(Fidelity={self.fidelity_score:.3f}, "
            f"Sparsity={self.sparsity_score:.3f}, Contiguity={self.contiguity_score:.3f})"
        )


class TextDecisionInformationExplainer:
    """
    Explainer DIxAI-Text pour modèles HuggingFace de classification de séquences.

    Usage:
        explainer = TextDecisionInformationExplainer(
            model, tokenizer, lambda_fidelity=10.0, lambda_contiguity=1.0,
            baseline_type="mask_token", device="cpu",
        )
        explanation = explainer.explain("this movie was not very good", steps=300)
        print(explanation.top_tokens())
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
        self.loss_fn = DecisionInformationLossText(
            lambda_fidelity=lambda_fidelity,
            task="classification",
            lambda_contiguity=lambda_contiguity,
        )
        self.baseline_provider = TextBaselineProvider(model, tokenizer, baseline_type)
        self.embedding_layer = model.get_input_embeddings()

    def _forward_from_embeds(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
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
        debug: bool = False,
    ) -> TextExplanation:
        if seed is not None:
            torch.manual_seed(seed)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        enc = self.tokenizer(text, return_tensors="pt", truncation=True)
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        T = input_ids.shape[1]

        # Tokens structurels ([CLS], [SEP], ...) : toujours gardés (mask=1), et exclus
        # du calcul d'information/sparsité/contiguïté (ce ne sont pas des candidats
        # à la sélection, donc ils ne doivent ni compter comme "coût" ni pouvoir
        # servir de raccourci pour préserver artificiellement la fidélité).
        special_ids_list = self.tokenizer.get_special_tokens_mask(
            input_ids[0].tolist(), already_has_special_tokens=True
        )
        special_tokens_mask = torch.tensor(
            special_ids_list, device=self.device, dtype=attention_mask.dtype
        ).unsqueeze(0)
        content_mask = attention_mask * (1 - special_tokens_mask)

        with torch.no_grad():
            x_embeds = self.embedding_layer(input_ids)  # (1, T, D)
            y_orig = self._forward_from_embeds(x_embeds, attention_mask)

        baseline = self.baseline_provider.get_baseline(seq_len=T, device=self.device)  # (T, D)

        # --- diagnostic : que donne le modèle si on remplace TOUT le contenu par la
        # baseline (en gardant [CLS]/[SEP] réels) ? Sert à vérifier si le mask final
        # apporte un vrai gain de fidélité par rapport à la baseline seule, ou si le
        # collapse vers un mask quasi-nul est simplement dû au fait que la baseline
        # suffit déjà à reproduire y_orig (auquel cas fidelity_loss n'a jamais de
        # vrai signal à exploiter).
        if debug:
            with torch.no_grad():
                z_null = baseline.unsqueeze(0).clone()  # (1, T, D)
                stm_bool = special_tokens_mask[0].bool()
                z_null[:, stm_bool] = x_embeds[:, stm_bool]
                y_null = self._forward_from_embeds(z_null, attention_mask)
                kl_null = F.kl_div(y_null, y_orig, reduction="batchmean", log_target=True).item()
                print(f"[diag] KL(y_orig, y_null_baseline) = {kl_null:.4f}")
                print(f"[diag] y_orig probs : {y_orig.exp().tolist()}")
                print(f"[diag] y_null probs : {y_null.exp().tolist()}")
        # --- fin diagnostic ---

        mask_module = TokenGumbelSoftmaxMask(
            seq_len=T, temperature=temperature, init_logits=init_logits
        ).to(self.device)

        optimizer = optim.Adam(mask_module.parameters(), lr=lr)

        temp_start, temp_end = temperature, 0.1
        final_lambda_fid = self.lambda_fidelity
        final_lambda_con = self.lambda_contiguity

        # La pression de fidélité doit atteindre sa pleine intensité BIEN AVANT que le
        # mask ne devienne "dur" (temperature basse). Sinon le mask se fige vers 0
        # (minimisation de info_loss) avant que fidelity_loss n'ait pu le corriger,
        # avec des gradients qui s'évanouissent une fois la température basse.
        fidelity_ramp_frac = 0.5  # lambda_fidelity atteint sa valeur finale à 50% de l'entraînement

        history: Dict[str, list] = {"loss": [], "info": [], "fidelity": [], "contiguity": []}

        for step in range(steps):
            optimizer.zero_grad()

            if anneal and steps > 1:
                progress = step / (steps - 1)

                # Température : anneal sur toute la durée (inchangé)
                tau = temp_start * (temp_end / temp_start) ** progress
                mask_module.temperature = tau

                # Lambda fidelity/contiguity : ramp rapide, plafonne à fidelity_ramp_frac
                fid_progress = min(1.0, progress / fidelity_ramp_frac)
                self.loss_fn.lambda_fidelity = final_lambda_fid * fid_progress
                self.loss_fn.lambda_contiguity = final_lambda_con * fid_progress

            z_embeds, mask = mask_module(
                x_embeds, baseline, training=True,
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
                    f"Step {step}: Loss={loss.item():.4f} Info={info_loss.item():.3f} "
                    f"Fid={fidelity_loss.item():.3f} Contig={contiguity_loss.item():.3f} "
                    f"Temp={mask_module.temperature:.2f}"
                )

        with torch.no_grad():
            z_final, mask_final = mask_module(
                x_embeds, baseline, training=False,
                attention_mask=attention_mask,
                special_tokens_mask=special_tokens_mask,
            )
            final_probs = mask_final[0]
            y_final = self._forward_from_embeds(z_final, attention_mask)
            fidelity = F.kl_div(y_final, y_orig, reduction="batchmean", log_target=True).item()
            sparsity = (final_probs * content_mask[0]).sum().item() / content_mask.sum().clamp(min=1).item()
            contiguity = self.loss_fn._sequential_contiguity(mask_final, content_mask).item()

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

        return TextExplanation(
            mask_probs=final_probs.cpu(),
            tokens=tokens,
            fidelity=fidelity,
            sparsity=sparsity,
            contiguity=contiguity,
        )