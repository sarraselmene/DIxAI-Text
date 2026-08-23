"""
Verrou 1 — Définition de la baseline B pour le texte.

3 baselines disponibles :
  1. mask_token    : embedding du token [MASK] (BERT MLM, in-distribution).
  2. pad_token     : embedding du token [PAD].
  3. mean_embedding: moyenne des embeddings du vocabulaire (hors spéciaux).

Le choix de la baseline est un hyperparamètre à ablationner.
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
    Calcule le tenseur de baseline B, shape (T, D) ou (D,).

    Usage:
        provider = TextBaselineProvider(model, tokenizer, BaselineType.MASK_TOKEN)
        baseline = provider.get_baseline(seq_len=T, device=device)  # (T, D)
        vec      = provider.get_baseline_vector()                    # (D,)
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        baseline_type: str = BaselineType.MASK_TOKEN,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.baseline_type = BaselineType(baseline_type)
        self.embedding_layer = model.get_input_embeddings()
        self._mean_embedding_cache = None

    def _mask_token_embedding(self) -> torch.Tensor:
        mask_id = self.tokenizer.mask_token_id
        if mask_id is None:
            raise ValueError(
                "Ce tokenizer n'a pas de token [MASK]. "
                "Utilisez baseline_type='pad_token' ou 'mean_embedding'."
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

    def get_baseline_vector(self, device=None) -> torch.Tensor:
        """
        Retourne le vecteur de baseline (D,).
        Si device est fourni, déplace le vecteur sur ce device.
        """
        if self.baseline_type == BaselineType.MASK_TOKEN:
            vec = self._mask_token_embedding()
        elif self.baseline_type == BaselineType.PAD_TOKEN:
            vec = self._pad_token_embedding()
        elif self.baseline_type == BaselineType.MEAN_EMBEDDING:
            vec = self._mean_embedding()
        else:
            raise ValueError(f"baseline_type inconnu: {self.baseline_type}")

        if device is not None:
            vec = vec.to(device)
        return vec

    def get_baseline(self, seq_len: int, device=None) -> torch.Tensor:
        """Retourne la baseline répétée sur toute la séquence : (T, D)."""
        vec = self.get_baseline_vector(device=device)
        return vec.unsqueeze(0).expand(seq_len, -1).clone()
