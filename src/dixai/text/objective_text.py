"""
Verrou 3 — Régularisation de cohérence séquentielle.

`dixai.objective.DecisionInformationLoss` pénalise les variations spatiales du mask
en 2D (TV loss sur H,W) pour éviter des masques "poivre et sel" en vision. En texte,
l'équivalent est une pénalité de variation totale 1D le long de l'axe des tokens :

    TV_seq(m) = (1/(T-1)) * sum_t | m_{t+1} - m_t |

Minimiser ce terme encourage le mask à être constant par morceaux, donc à sélectionner
des SPANS de tokens contigus (ex: "not very good") plutôt que des tokens épars et peu
interprétables (ex: "not ... good" avec 3 tokens sautés entre les deux).

On garde exactement la même structure d'objectif Lagrangien que la version originale :

    L = I(X;Z) + lambda_fidelity * E[dist(f(X), f(Z))] + lambda_contiguity * TV_seq(m)

pour rester compatible avec le reste du pipeline DIxAI (mine.py, metrics.py, etc.).
"""

import torch
import torch.nn as nn
from typing import Optional


class DecisionInformationLossText(nn.Module):
    """
    Objectif DIB pour le texte : terme d'information + fidélité + contiguïté séquentielle.

    Args:
        lambda_fidelity   : poids du terme de préservation de la décision.
        task              : 'classification' ou 'regression'.
        lambda_contiguity : poids de la pénalité de contiguïté séquentielle
                            (remplace lambda_tv).
                            Si 0.0, contiguity_loss est retourné comme 0.0
                            sans calculer TV_seq.
        attention_mask_aware: si True, ignore les paires de tokens adjacents
                            dont l'un des deux est du padding lors du calcul
                            de TV_seq (évite de pénaliser la frontière
                            texte/padding).
    """

    def __init__(
        self,
        lambda_fidelity: float = 1.0,
        task: str = "classification",
        lambda_contiguity: float = 0.0,
        attention_mask_aware: bool = True,
    ):
        super().__init__()
        self.lambda_fidelity = lambda_fidelity
        self.lambda_contiguity = lambda_contiguity
        self.task = task
        self.attention_mask_aware = attention_mask_aware

        if task == "classification":
            self.fidelity_criterion = nn.KLDivLoss(
                reduction="batchmean",
                log_target=True,
            )
        elif task == "regression":
            self.fidelity_criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unknown task: {task}")

    def _sequential_contiguity(
        self,
        mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calcule la Total Variation séquentielle sur les tokens de contenu.

        TV_seq(m) = (1/|V|) * sum_{(t,t+1) in V} |m_{t+1} - m_t|

        Args:
            mask          : (B, T) probabilités de sélection par token.
            attention_mask: (B, T) optionnel, 1 pour tokens réels (contenu).

        Returns:
            tv : scalaire >= 0.
        """
        if mask.size(-1) < 2:
            return torch.tensor(0.0, device=mask.device)

        diff = mask[:, 1:] - mask[:, :-1]  # (B, T-1)

        if self.attention_mask_aware and attention_mask is not None:
            # Une paire (t, t+1) est valide seulement si les deux
            # tokens sont réels (non-padding, non-spéciaux)
            pair_valid = (
                attention_mask[:, 1:] * attention_mask[:, :-1]
            ).to(diff.dtype)
            num_valid = pair_valid.sum().clamp(min=1.0)
            tv = (diff.abs() * pair_valid).sum() / num_valid
        else:
            tv = diff.abs().mean()

        return tv

    def forward(
        self,
        mask: torch.Tensor,
        y_pred_original: torch.Tensor,
        y_pred_masked: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Calcule la loss DIxAI-Text complète.

        Args:
            mask            : (B, T) mask par token (valeurs dans [0,1]).
            y_pred_original : log-probs de f(X)  — (B, C).
            y_pred_masked   : log-probs de f(Z)  — (B, C).
            attention_mask  : (B, T) optionnel, pour ignorer le padding
                              dans info_loss et TV_seq.

        Returns:
            total_loss      : loss totale scalaire.
            info_loss       : terme I(X;Z) — proportion de tokens gardés.
            fidelity_loss   : terme KL(f(X) || f(Z)).
            contiguity_loss : terme TV_seq — 0.0 si lambda_contiguity == 0.
        """
        # ------------------------------------------------------------------
        # 1. Terme d'information I(X;Z)
        #    ~ taux moyen de sélection sur les tokens de contenu
        # ------------------------------------------------------------------
        if attention_mask is not None:
            am = attention_mask.to(mask.dtype)
            info_loss = (mask * am).sum() / am.sum().clamp(min=1.0)
        else:
            info_loss = mask.mean()

        # ------------------------------------------------------------------
        # 2. Terme de fidélité E[dist(f(X), f(Z))]
        # ------------------------------------------------------------------
        fidelity_loss = self.fidelity_criterion(y_pred_masked, y_pred_original)

        # ------------------------------------------------------------------
        # 3. Cohérence séquentielle (remplace la TV spatiale)
        #    CORRECTION : si lambda_contiguity == 0.0, on retourne 0.0
        #    directement sans calculer TV_seq pour que le test 10 passe.
        # ------------------------------------------------------------------
        if self.lambda_contiguity == 0.0:
            contiguity_loss = torch.tensor(
                0.0,
                device=mask.device,
                dtype=mask.dtype,
            )
        else:
            contiguity_loss = self._sequential_contiguity(
                mask, attention_mask
            )

        # ------------------------------------------------------------------
        # 4. Loss totale (Lagrangien DIxAI)
        # ------------------------------------------------------------------
        total_loss = (
            info_loss
            + self.lambda_fidelity * fidelity_loss
            + self.lambda_contiguity * contiguity_loss
        )

        return total_loss, info_loss, fidelity_loss, contiguity_loss
