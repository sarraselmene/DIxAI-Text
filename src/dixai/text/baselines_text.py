"""
Verrou 2 — Définition de la baseline B pour le texte.

En vision, B est typiquement une image noire, floutée, ou du bruit. En NLP, "supprimer"
un token n'a pas d'équivalent aussi naturel : on ne peut pas mettre un embedding à zéro
sans sortir de la distribution que le modèle a vue à l'entraînement (out-of-distribution
baseline -> le modèle réagit de façon non informative, biaisant l'explication).

On propose 3 baselines, correspondant chacune à un contrat différent avec le modèle :

  1. "mask_token"   : remplace le token par l'embedding du token spécial [MASK] du
                      tokenizer (ex: BERT). C'est la baseline la plus fidèle au
                      pré-entraînement MLM de BERT -> comportement du modèle bien défini.
  2. "pad_token"     : remplace le token par l'embedding du token [PAD]. Moins coûteux,
                      mais BERT n'a pas été entraîné à "raisonner" sur du padding au
                      milieu d'une séquence -> risque léger d'OOD.
  3. "mean_embedding": remplace le token par la moyenne empirique des embeddings du
                      vocabulaire (ou de la séquence courante). Baseline "neutre" au sens
                      de Shapley/Integrated Gradients (proche de l'esprit du zero-baseline
                      en vision, mais dans l'espace des embeddings BERT).

Le choix de la baseline est un hyperparamètre à ablationner (Semaine 7-8, cf. plan de
travail : "études d'ablation ... baseline B").
"""

from enum import Enum
import torch
import torch.nn as nn


class BaselineType(str, Enum):
    MASK_TOKEN = "mask_token"
    PAD_TOKEN = "pad_token"
    MEAN_EMBEDDING = "mean_embedding"


class TextBaselineProvider:
    """
    Calcule le tenseur de baseline B, shape (T, D) ou (D,), à partir d'un modèle
    HuggingFace (n'importe quel modèle avec `.get_input_embeddings()`, ex. BERT/RoBERTa)
    et de son tokenizer.

    Usage:
        provider = TextBaselineProvider(model, tokenizer, baseline_type="mask_token")
        baseline = provider.get_baseline(seq_len=input_ids.shape[1], device=device)
        # baseline: (T, D) -> à passer à TokenGumbelSoftmaxMask.forward(..., baseline=baseline)
    """

    def __init__(self, model: nn.Module, tokenizer, baseline_type: str = BaselineType.MASK_TOKEN):
        self.model = model
        self.tokenizer = tokenizer
        self.baseline_type = BaselineType(baseline_type)
        self.embedding_layer = model.get_input_embeddings()  # nn.Embedding

        self._mean_embedding_cache = None

    def _mask_token_embedding(self) -> torch.Tensor:
        mask_id = self.tokenizer.mask_token_id
        if mask_id is None:
            raise ValueError(
                "Ce tokenizer n'a pas de token [MASK] (ex: certains modèles causaux). "
                "Utilisez baseline_type='pad_token' ou 'mean_embedding' à la place."
            )
        with torch.no_grad():
            return self.embedding_layer.weight[mask_id].clone()  # (D,)

    def _pad_token_embedding(self) -> torch.Tensor:
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("Ce tokenizer n'a pas de token [PAD].")
        with torch.no_grad():
            return self.embedding_layer.weight[pad_id].clone()  # (D,)

    def _mean_embedding(self) -> torch.Tensor:
        # Moyenne sur tout le vocabulaire, en excluant les tokens spéciaux si possible.
        if self._mean_embedding_cache is not None:
            return self._mean_embedding_cache
        with torch.no_grad():
            weight = self.embedding_layer.weight  # (V, D)
            special_ids = set(self.tokenizer.all_special_ids or [])
            if special_ids:
                keep = torch.ones(weight.size(0), dtype=torch.bool)
                for sid in special_ids:
                    if 0 <= sid < weight.size(0):
                        keep[sid] = False
                mean_emb = weight[keep].mean(dim=0)
            else:
                mean_emb = weight.mean(dim=0)
        self._mean_embedding_cache = mean_emb.clone()
        return self._mean_embedding_cache

    def get_baseline_vector(self) -> torch.Tensor:
        """Retourne le vecteur de baseline (D,), indépendant de la position."""
        if self.baseline_type == BaselineType.MASK_TOKEN:
            return self._mask_token_embedding()
        elif self.baseline_type == BaselineType.PAD_TOKEN:
            return self._pad_token_embedding()
        elif self.baseline_type == BaselineType.MEAN_EMBEDDING:
            return self._mean_embedding()
        else:
            raise ValueError(f"baseline_type inconnu: {self.baseline_type}")

    def get_baseline(self, seq_len: int, device=None) -> torch.Tensor:
        """Retourne la baseline répétée sur toute la séquence: (T, D)."""
        vec = self.get_baseline_vector()
        if device is not None:
            vec = vec.to(device)
        return vec.unsqueeze(0).expand(seq_len, -1).clone()
