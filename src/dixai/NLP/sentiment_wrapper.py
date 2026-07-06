"""
BertSentimentWrapper : le modèle noir f() que DIxAI doit expliquer.

C'est un BERT fine-tuné pour la classification (ex: SST-2), pas un
classifieur jouet. Il accepte inputs_embeds pour permettre au masquage
de traverser réellement les 12 couches Transformer.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification


class BertSentimentWrapper:
    """
    Encapsule un BERT fine-tuné pour la classification de sentiment.
    Gelé : DIxAI n'entraîne jamais ce modèle, seulement le masque.
    """

    def __init__(
        self,
        model_name: str = "textattack/bert-base-uncased-SST-2",
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.to(self.device)

    def forward_from_embeddings(
        self,
        input_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Fait tourner les 12 couches Transformer + tête de classification
        à partir d'embeddings (masqués ou non).

        Returns:
            log_probs : (B, num_classes)
        """
        outputs = self.model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        return torch.log_softmax(outputs.logits, dim=-1)