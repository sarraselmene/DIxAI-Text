"""
Verrou 1 — Masquage différentiable sur token embeddings.

Adapte dixai.masks.GumbelSoftmaxMask au cas du texte : une séquence de T tokens,
chacun représenté par un embedding contextuel de dimension D.

Différences clés par rapport à la version image :
  - Une seule valeur de mask (scalaire) par TOKEN : mask_logits shape (T,).
  - Pas d'upsampling multi-échelle.
  - Mélange au niveau du vecteur d'embedding entier :
        z_t = m_t * x_t + (1 - m_t) * b_t
  - Padding toujours à 0 (jamais sélectionné).
  - Tokens spéciaux ([CLS],[SEP]) toujours à 1 (structurels, pas candidats).

Utilitaires debug (récupérés de text_mask.py, désormais intégré ici) :
  - debug_tokens() : pretty-print avec emojis 🟢🟡🔴
  - report()       : tableau complet mask_logits / probabilités
  - print_logits() : affichage rapide des logits courants
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List


class TokenGumbelSoftmaxMask(nn.Module):
    """
    Masque Gumbel-Softmax (relaxation Concrete) appliqué à une séquence de tokens.

    Args:
        seq_len: longueur de séquence T (padding inclus).
        temperature: température tau de la relaxation Concrete/Gumbel-Sigmoid.
        init_logits: valeur initiale des logits (négative = sparse-first).
        hard: si True, straight-through estimator (mask binaire forward,
              gradient continu backward).
    """

    def __init__(
        self,
        seq_len: int,
        temperature: float = 2.0 / 3.0,
        init_logits: float = -2.0,
        hard: bool = False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.temperature = temperature
        self.hard = hard
        # Un logit scalaire par position de token -> shape (T,)
        self.mask_logits = nn.Parameter(torch.full((seq_len,), init_logits))

    def _sample_mask(self, training: bool) -> torch.Tensor:
        """Échantillonne le masque (T,) avec ou sans bruit Gumbel."""
        logits = self.mask_logits  # (T,)

        if training:
            uniform = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
            gumbel_noise = -torch.log(-torch.log(uniform))
            y_soft = torch.sigmoid((logits + gumbel_noise) / self.temperature)
        else:
            y_soft = torch.sigmoid(logits)

        if self.hard:
            y_hard = (y_soft > 0.5).float()
            # Straight-through : forward=hard, backward=soft
            mask = y_hard.detach() - y_soft.detach() + y_soft
        else:
            mask = y_soft

        return mask  # (T,)

    def forward(
        self,
        token_embeddings: torch.Tensor,
        baseline: torch.Tensor,
        training: bool = True,
        attention_mask: Optional[torch.Tensor] = None,
        special_tokens_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            token_embeddings: (B, T, D) embeddings BERT/RoBERTa.
            baseline: (B, T, D) ou (T, D) embeddings de référence B.
            training: si True, bruit Gumbel ; sinon déterministe.
            attention_mask: (B, T) 1=réel, 0=padding. Force mask=0 sur padding.
            special_tokens_mask: (B, T) 1=[CLS]/[SEP], 0=contenu.
                Force mask=1 sur tokens structurels.

        Returns:
            z: (B, T, D) embeddings mélangés z = m*x + (1-m)*b
            mask: (B, T) valeurs dans [0,1]
        """
        assert token_embeddings.dim() == 3, "Attendu (B, T, D)"
        B, T, D = token_embeddings.shape
        assert T == self.seq_len, (
            f"seq_len configuré ({self.seq_len}) != T observé ({T}). "
            f"Recréer le mask_module avec seq_len={T}."
        )

        mask = self._sample_mask(training)       # (T,)
        mask = mask.unsqueeze(0).expand(B, T)    # (B, T)

        # Padding : jamais sélectionné
        if attention_mask is not None:
            mask = mask * attention_mask.to(mask.dtype)

        # Tokens spéciaux : toujours gardés (mask=1 forcé)
        if special_tokens_mask is not None:
            stm = special_tokens_mask.to(mask.dtype)
            mask = mask * (1 - stm) + stm

        # Clamp défensif : garantit [0,1] même après opérations
        mask = mask.clamp(0.0, 1.0)

        # Baseline : expand si nécessaire
        if baseline.dim() == 2:
            baseline = baseline.unsqueeze(0).expand(B, T, D)

        mask_expanded = mask.unsqueeze(-1)  # (B, T, 1) -> broadcast sur D
        z = mask_expanded * token_embeddings + (1 - mask_expanded) * baseline

        return z, mask

    # ==================================================================
    # Utilitaires (récupérés de text_mask.py, désormais intégré ici)
    # ==================================================================

    def get_mask_probs(self) -> torch.Tensor:
        """Probabilités déterministes par token, shape (T,). Sans bruit Gumbel."""
        return torch.sigmoid(self.mask_logits)

    def selected_token_indices(self, threshold: float = 0.5) -> torch.Tensor:
        """Indices des tokens sélectionnés (prob > threshold). Utile pour ERASER."""
        probs = self.get_mask_probs()
        return torch.nonzero(probs > threshold, as_tuple=False).squeeze(-1)

    def debug_tokens(
        self,
        tokenizer,
        input_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        """
        Pretty-print token importance avec emojis.

        Args:
            tokenizer : HuggingFace tokenizer.
            input_ids : (B, T) token IDs.
            mask      : (B, T) mask probabilités.
        """
        print("\n" + "=" * 80)
        print("TOKEN IMPORTANCE (debug)")
        print("=" * 80)

        tokens = tokenizer.convert_ids_to_tokens(
            input_ids[0].detach().cpu().tolist()
        )
        values = mask[0].detach().cpu().tolist()

        for token, score in zip(tokens, values):
            if score > 0.8:
                symbol = "🟢"
            elif score > 0.5:
                symbol = "🟡"
            elif score > 0.2:
                symbol = "🟠"
            else:
                symbol = "🔴"
            print(f"  {symbol} {token:<20} {score:.4f}")

        print("=" * 80)

    def print_logits(self) -> None:
        """Affiche les logits et probabilités courants du masque."""
        print("\nCurrent logits:")
        print(self.mask_logits.detach().cpu())
        print("\nCurrent probabilities:")
        print(self.get_mask_probs().detach().cpu())

    def report(
        self,
        tokenizer,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Rapport complet : token, index, probabilité.

        Args:
            tokenizer      : HuggingFace tokenizer.
            input_ids      : (B, T) token IDs.
            attention_mask : (B, T) optionnel, skip les tokens padding.
        """
        probs = self.get_mask_probs().detach().cpu()
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())

        print("\n" + "=" * 80)
        print("MASK REPORT")
        print("=" * 80)

        for i, (tok, p) in enumerate(zip(tokens, probs)):
            if attention_mask is not None:
                if attention_mask[0, i] == 0:
                    continue
            print(f"  {i:02d} | {tok:<20} | {p:.4f}")

        print("=" * 80)