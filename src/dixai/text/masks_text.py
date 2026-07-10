"""
Verrou 1 — Masquage différentiable sur token embeddings.

Adapte `dixai.masks.GumbelSoftmaxMask` (conçu pour des pixels / features tabulaires,
grille (C,H,W) ou (Dim,)) au cas du texte : une séquence de T tokens, chacun représenté
par un embedding contextuel de dimension D (sortie de la couche d'embedding BERT/RoBERTa,
ou d'une couche intermédiaire si on veut expliquer un niveau plus profond).

Différences clés par rapport à la version image :
  - Une seule valeur de mask (scalaire) par TOKEN, pas par dimension d'embedding :
    on veut sélectionner des tokens, pas des coordonnées individuelles du vecteur.
    -> mask_logits a la forme (T,), pas (T, D).
  - Pas d'upsampling multi-échelle (pas de notion de résolution spatiale en NLP).
  - Le mélange mask/baseline se fait au niveau du vecteur d'embedding entier :
        z_t = m_t * x_t + (1 - m_t) * b_t
    où m_t in [0,1] est répété (broadcast) sur les D dimensions de l'embedding.
  - Un masque de padding (attention_mask) doit être respecté : les tokens de padding
    sont toujours à 0 (jamais sélectionnés, ne comptent pas dans la pénalité de sparsité).
  - Les tokens spéciaux ([CLS], [SEP], ...) sont structurels, pas du contenu : ils sont
    toujours forcés à mask=1 via special_tokens_mask (sinon l'optimiseur peut exploiter
    leur embedding comme raccourci pour préserver la fidélité sans rien dire du texte).
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class TokenGumbelSoftmaxMask(nn.Module):
    """
    Masque Gumbel-Softmax (relaxation Concrete) appliqué à une séquence de tokens.

    Args:
        seq_len: longueur de séquence T (nombre de tokens, padding inclus).
        temperature: température tau de la relaxation Concrete/Gumbel-Sigmoid.
        init_logits: valeur initiale des logits (négative = mask proche de 0 au début,
            comportement "sparse-first" cohérent avec le reste du framework DIxAI).
        hard: si True, utilise le straight-through estimator (mask binaire en forward,
            gradient continu en backward). Utile pour les métriques ERASER qui attendent
            des sélections discrètes de tokens.
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
        logits = self.mask_logits  # (T,)

        if training:
            uniform = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
            gumbel_noise = -torch.log(-torch.log(uniform))
            y_soft = torch.sigmoid((logits + gumbel_noise) / self.temperature)
        else:
            y_soft = torch.sigmoid(logits)

        if self.hard:
            y_hard = (y_soft > 0.5).float()
            # straight-through: forward = hard, backward = soft
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
            token_embeddings: (B, T, D) embeddings BERT/RoBERTa de l'entrée x.
            baseline: (B, T, D) ou (T, D) embeddings de référence B (voir baselines_text.py).
            training: si True, échantillonnage stochastique (Gumbel noise) ; sinon seuillage dur.
            attention_mask: (B, T) 1 pour tokens réels, 0 pour padding. Si fourni, force
                mask=0 sur le padding (le padding n'est jamais "sélectionné").
            special_tokens_mask: (B, T) 1 pour [CLS]/[SEP]/etc, 0 pour tokens de contenu.
                Si fourni, force mask=1 sur ces positions : ce ne sont pas des tokens de
                "contenu" candidats à la sélection, ils sont structurellement nécessaires
                au modèle et ne doivent jamais apparaître comme "importants" dans
                l'explication (sinon l'optimiseur peut exploiter leur embedding comme
                raccourci pour préserver la fidélité sans rien dire du texte).

        Returns:
            z: (B, T, D) embeddings mélangés z = m*x + (1-m)*b
            mask: (B, T) probabilités/valeurs de mask par token, dans [0,1]
        """
        assert token_embeddings.dim() == 3, "attendu (B, T, D)"
        B, T, D = token_embeddings.shape
        assert T == self.seq_len, f"seq_len configuré ({self.seq_len}) != T observé ({T})"

        mask = self._sample_mask(training)          # (T,)
        mask = mask.unsqueeze(0).expand(B, T)        # (B, T)

        if attention_mask is not None:
            # Le padding n'est jamais "conservé" : on force mask=0 sur le padding.
            # (gradient nul sur ces positions, ce qui est le comportement voulu)
            mask = mask * attention_mask.to(mask.dtype)

        if special_tokens_mask is not None:
            stm = special_tokens_mask.to(mask.dtype)
            # mask = 1 forcé là où stm == 1, sinon on garde la valeur calculée
            mask = mask * (1 - stm) + stm

        if baseline.dim() == 2:
            baseline = baseline.unsqueeze(0).expand(B, T, D)

        mask_expanded = mask.unsqueeze(-1)  # (B, T, 1) -> broadcast sur D
        z = mask_expanded * token_embeddings + (1 - mask_expanded) * baseline

        return z, mask

    def get_mask_probs(self) -> torch.Tensor:
        """Probabilités de sélection par token (déterministe, pas de bruit Gumbel), shape (T,)."""
        return torch.sigmoid(self.mask_logits)

    def selected_token_indices(self, threshold: float = 0.5) -> torch.Tensor:
        """Indices des tokens sélectionnés (mask_prob > threshold). Utile pour ERASER."""
        probs = self.get_mask_probs()
        return torch.nonzero(probs > threshold, as_tuple=False).squeeze(-1)