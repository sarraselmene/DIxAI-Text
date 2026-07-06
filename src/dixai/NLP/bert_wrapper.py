"""
BertWrapper : convertit du texte en embeddings contextuels BERT.
Point d'entrée du pipeline DIxAI-Text.

VERSION CORRIGÉE : ajoute l'accès à la couche d'embedding PRE-encodeur
(avant les 12 couches Transformer), qui est le tenseur qu'il faut
réellement masquer pour que le masquage ait un effet sur l'attention.
"""

import torch
from transformers import AutoTokenizer, AutoModel
from typing import Union, List, Tuple


class BertWrapper:
    """
    Encapsule BERT pour produire des embeddings de tokens.

    Le modèle est gelé — on n'entraîne jamais BERT ici.
    Seul le TokenMask sera entraîné par DIxAI.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.to(self.device)

    def encode(
        self,
        texts: Union[str, List[str]],
        max_length: int = 128
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ATTENTION : cette méthode retourne last_hidden_state, c'est-à-dire
        la sortie APRÈS les 12 couches Transformer.
        Utile uniquement pour inspecter/visualiser les embeddings finaux,
        PAS pour construire le pipeline de masquage DIxAI.
        Pour le masquage, utiliser get_input_embeddings() ci-dessous.
        """
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = outputs.last_hidden_state
        return embeddings, inputs["attention_mask"]

    def get_input_embeddings(
        self,
        texts: Union[str, List[str]],
        max_length: int = 128
    ):
        """
        Retourne les embeddings AVANT les 12 couches Transformer
        (sortie de la couche d'embedding : word + position + segment).

        C'est CE tenseur qu'il faut masquer dans le pipeline DIxAI-Text,
        pour que le masquage ait un effet réel une fois rejoué dans
        les 12 couches d'attention.

        Returns:
            input_embeds   : (B, T, D)
            attention_mask : (B, T)
            input_ids      : (B, T)
            token_type_ids : (B, T) ou None
        """
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt"
        ).to(self.device)

        token_type_ids = inputs.get("token_type_ids", None)

        with torch.no_grad():
            input_embeds = self.model.embeddings(
                input_ids=inputs["input_ids"],
                token_type_ids=token_type_ids
            )

        return input_embeds, inputs["attention_mask"], inputs["input_ids"], token_type_ids

    def get_mask_baseline_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Baseline correcte : embedding [MASK] passé par la MÊME couche
        d'embedding (pas last_hidden_state), avec la même shape que
        input_embeds (B, T, D) pour un remplacement token-par-token cohérent.

        Args:
            input_ids : (B, T) — utilisé uniquement pour connaître B et T

        Returns:
            baseline : (B, T, D)
        """
        B, T = input_ids.shape
        mask_ids = torch.full((B, T), self.tokenizer.mask_token_id, device=self.device)

        with torch.no_grad():
            baseline = self.model.embeddings(input_ids=mask_ids)

        return baseline

    def forward_from_embeddings(self, input_embeds: torch.Tensor, attention_mask: torch.Tensor):
        """
        Rejoue les 12 couches Transformer à partir d'embeddings donnés
        (masqués ou non). Utile pour du debug direct sur BertWrapper
        (le vrai pipeline d'explication passe plutôt par un modèle de
        classification fine-tuné, voir sentiment_wrapper.py).

        Returns:
            last_hidden_state : (B, T, D)
        """
        outputs = self.model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask
        )
        return outputs.last_hidden_state

    def decode_tokens(self, texts: Union[str, List[str]], max_length: int = 128):
        if isinstance(texts, str):
            texts = [texts]

        tokens = []
        for text in texts:
            ids = self.tokenizer.encode(
                text, truncation=True, max_length=max_length
            )
            tokens.append(self.tokenizer.convert_ids_to_tokens(ids))

        return tokens