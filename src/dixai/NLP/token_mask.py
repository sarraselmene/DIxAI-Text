"""
GumbelSoftmaxTokenMask : masquage différentiable au niveau des tokens.

Verrou 1 du stage DIxAI-Text.

Pourquoi un logit PAR TOKEN (et non par dimension) ?
    - DIxAI cherche le sous-ensemble MINIMAL de tokens suffisants.
    - On veut décider : "garde ce token" ou "remplace-le par la baseline".
    - Cette décision est BINAIRE par token → 1 logit par token.
    - Si on mettait 1 logit par dimension, on masquerait des dimensions
      individuelles d'un embedding, ce qui n'a pas de sens sémantique.

Formule Gumbel-Softmax :
    g_i ~ Gumbel(0,1)   (bruit)
    m_i = sigmoid( (θ_i + g_i) / τ )

    τ grand (ex: 1.0) → m_i continu ≈ 0.5  (exploration)
    τ petit (ex: 0.1) → m_i binaire {0,1}   (décision nette)

Application du masque :
    Z_t = m_t * e_t + (1 - m_t) * B_t

    e_t : embedding original du token t
    B_t : baseline (embedding [MASK] ou zéro)
    m_t : score du masque pour le token t ∈ (0,1)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class GumbelSoftmaxTokenMask(nn.Module):
    """
    Masque différentiable pour les token embeddings BERT.

    Compatible avec DIxAI : même interface que GumbelSoftmaxMask,
    adaptée à la modalité texte.

    Args:
        seq_len    : nombre de tokens dans la séquence (T)
        init_logits: valeur initiale des logits (négatif = biais vers sparsité)
                     -2.0 → P(garder token) ≈ 12% au départ
        temperature: τ du Gumbel-Softmax
    """

    def __init__(
        self,
        seq_len: int,
        init_logits: float = -2.0,
        temperature: float = 1.0
    ):
        super().__init__()

        # UN logit par token — c'est la seule bonne conception
        # shape : (1, T, 1)  → broadcaste sur (B, T, D)
        self.mask_logits = nn.Parameter(
            torch.full((1, seq_len, 1), init_logits)
        )
        self.temperature = temperature
        self.seq_len = seq_len

    def forward(
        self,
        embeddings: torch.Tensor,
        baseline: torch.Tensor,
        training: bool = True,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applique le masque Gumbel-Softmax sur les embeddings.

        Args:
            embeddings     : (B, T, D) — embeddings BERT originaux
            baseline       : (B, T, D) — baseline (embedding [MASK] ou zéro)
            training       : True → masque soft + bruit Gumbel
                             False → masque binaire (inférence)
            attention_mask : (B, T) optionnel — force m=0 sur les tokens
                             de padding (on ne veut jamais les sélectionner)

        Returns:
            masked_embeddings : (B, T, D)  — Z = M*e + (1-M)*B
            token_mask        : (B, T)     — importance par token ∈ (0,1)
        """
        B, T, D = embeddings.shape

        if training:
            # Bruit de Gumbel : g = -log(-log(U)),  U ~ Uniform(0,1)
            uniform = torch.rand_like(self.mask_logits).clamp(1e-8, 1 - 1e-8)
            gumbel_noise = -torch.log(-torch.log(uniform))

            # Score avec température
            scores = (self.mask_logits + gumbel_noise) / self.temperature

            # Masque soft ∈ (0,1) — différentiable
            mask = torch.sigmoid(scores)  # (1, T, 1)
        else:
            # Masque binaire {0,1} — inférence
            # seuil à 0.5 → logit > 0 → token conservé
            mask = (self.mask_logits > 0).float()  # (1, T, 1)

        # Appliquer le masque sur les tokens de padding
        # Un token de padding ne doit JAMAIS être sélectionné
        if attention_mask is not None:
            # attention_mask : (B, T) → (B, T, 1)
            pad_mask = attention_mask.unsqueeze(-1).float()
            mask = mask * pad_mask  # force m=0 là où padding

        # Broadcast : (1, T, 1) → (B, T, D)
        mask_expanded = mask.expand(B, T, D)

        # Z = M ⊙ e + (1-M) ⊙ B
        masked_embeddings = embeddings * mask_expanded + baseline * (1 - mask_expanded)

        # token_mask : (B, T) — 1 valeur par token pour les métriques
        token_mask = mask.squeeze(-1).expand(B, T)  # (B, T)

        return masked_embeddings, token_mask

    def get_mask_probs(self) -> torch.Tensor:
        """
        Retourne les probabilités de sélection de chaque token.
        Utilisé pour le ranking après optimisation.

        Returns:
            probs : (T,)  — P(token_t sélectionné)
        """
        return torch.sigmoid(self.mask_logits).squeeze()  # (T,)

    def get_selected_tokens(self, threshold: float = 0.5) -> torch.Tensor:
        """
        Retourne le masque binaire final.

        Returns:
            binary_mask : (T,)  — 1 si token sélectionné, 0 sinon
        """
        return (self.mask_logits.squeeze() > 0).float()