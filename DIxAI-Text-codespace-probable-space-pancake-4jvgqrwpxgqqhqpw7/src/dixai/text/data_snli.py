"""
data_snli.py — Chargement SNLI pour DIxAI-Text.

SNLI = Stanford Natural Language Inference
Tâche : étant donné (premise, hypothesis), prédire
        entailment / neutral / contradiction.

Pour BERT NLI : on concatène premise + [SEP] + hypothesis
comme un seul texte (format standard HuggingFace NLI).

Usage:
    from dixai.text.data_snli import load_snli
    examples = load_snli(limit=50)
    for ex in examples:
        print(ex.text)   # premise [SEP] hypothesis
        print(ex.label)  # 0=entailment, 1=neutral, 2=contradiction
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SNLIExample:
    """
    Un exemple SNLI.

    Attributs:
        premise   : phrase de départ.
        hypothesis: phrase à vérifier.
        label     : 0=entailment, 1=neutral, 2=contradiction.
        text      : concaténation pour le modèle BERT NLI.
        idx       : index dans le dataset.
    """
    premise: str
    hypothesis: str
    label: int
    text: str = ""
    idx: Optional[int] = None

    def __post_init__(self):
        # Format standard BERT NLI : premise [SEP] hypothesis
        if not self.text:
            self.text = f"{self.premise} [SEP] {self.hypothesis}"


# Fallback manuel si HuggingFace non disponible
_SNLI_FALLBACK = [
    SNLIExample(
        premise="A man is playing guitar on stage.",
        hypothesis="A person is performing music.",
        label=0, idx=0,
    ),
    SNLIExample(
        premise="A woman is reading a book in the park.",
        hypothesis="A woman is sleeping.",
        label=2, idx=1,
    ),
    SNLIExample(
        premise="Two children are playing football outside.",
        hypothesis="Some kids are doing an outdoor activity.",
        label=0, idx=2,
    ),
    SNLIExample(
        premise="A dog is running on the beach.",
        hypothesis="The animal is indoors.",
        label=2, idx=3,
    ),
    SNLIExample(
        premise="A chef is preparing food in a restaurant kitchen.",
        hypothesis="Someone is cooking.",
        label=0, idx=4,
    ),
    SNLIExample(
        premise="Three men are sitting at a table drinking coffee.",
        hypothesis="The men are having breakfast.",
        label=1, idx=5,
    ),
    SNLIExample(
        premise="A young girl is dancing in the street.",
        hypothesis="A child is performing outside.",
        label=0, idx=6,
    ),
    SNLIExample(
        premise="A cat is sleeping on a sofa.",
        hypothesis="The cat is running in the garden.",
        label=2, idx=7,
    ),
]

LABEL_NAMES_SNLI = {
    0: "ENTAILMENT",
    1: "NEUTRAL",
    2: "CONTRADICTION",
}


def load_snli(
    limit: int = 50,
    split: str = "validation",
    seed: int = 42,
) -> List[SNLIExample]:
    """
    Charge des exemples SNLI depuis HuggingFace datasets.

    Args:
        limit: nombre maximum d'exemples à retourner.
        split: "train", "validation" ou "test".
        seed : graine pour reproductibilité.

    Returns:
        Liste de SNLIExample.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("snli", split=split)

        # Filtrer les exemples sans label (-1 = pas d'accord entre annotateurs)
        ds = ds.filter(lambda x: x["label"] != -1)

        rng = np.random.default_rng(seed)
        n = min(limit, len(ds))
        indices = rng.choice(len(ds), size=n, replace=False).tolist()

        examples = [
            SNLIExample(
                premise=ds[int(i)]["premise"].strip(),
                hypothesis=ds[int(i)]["hypothesis"].strip(),
                label=ds[int(i)]["label"],
                idx=int(i),
            )
            for i in indices
        ]
        print(f"[SNLI] Chargé {len(examples)} exemples ({split}).")
        return examples

    except Exception as e:
        print(f"[SNLI] Fallback liste manuelle ({e}).")
        return _SNLI_FALLBACK[:limit]
