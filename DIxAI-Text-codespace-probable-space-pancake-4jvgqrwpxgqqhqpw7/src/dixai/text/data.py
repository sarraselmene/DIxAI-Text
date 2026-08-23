"""
Chargement des données pour les expériences DIxAI-Text.

Supporte :
  - SST-2 (HuggingFace datasets glue/sst2)
  - Fallback manuel si datasets non disponible

Usage:
    from dixai.text.data import load_sst2
    examples = load_sst2(limit=50)
    for ex in examples:
        print(ex.text, ex.label)
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class TextExample:
    """Un exemple de classification de texte."""
    text: str
    label: Optional[int] = None   # 0=négatif, 1=positif pour SST-2
    idx: Optional[int] = None


# Fallback SST-2 manuel (si HuggingFace datasets non disponible)
_SST2_FALLBACK = [
    TextExample("This movie was not very good, honestly quite disappointing.", 0, 0),
    TextExample("An absolute masterpiece with brilliant performances.", 1, 1),
    TextExample("The plot was boring and the acting felt wooden.", 0, 2),
    TextExample("I really enjoyed every minute of this film.", 1, 3),
    TextExample("Waste of time, poorly written and badly directed.", 0, 4),
    TextExample("A charming, heartfelt story that stayed with me.", 1, 5),
    TextExample("Predictable and dull from start to finish.", 0, 6),
    TextExample("One of the best films I have seen this year.", 1, 7),
    TextExample("Terrible acting and a nonsensical plot.", 0, 8),
    TextExample("A beautiful and moving piece of cinema.", 1, 9),
    TextExample("I fell asleep halfway through, so boring.", 0, 10),
    TextExample("Funny, clever, and surprisingly touching.", 1, 11),
    TextExample("The worst film I have seen in years.", 0, 12),
    TextExample("Stunning visuals and a gripping story.", 1, 13),
    TextExample("Completely forgettable and uninspired.", 0, 14),
    TextExample("A triumph of storytelling and direction.", 1, 15),
]


def load_sst2(
    limit: int = 50,
    split: str = "validation",
    seed: int = 42,
) -> List[TextExample]:
    """
    Charge des exemples SST-2.

    Args:
        limit: nombre maximum d'exemples à retourner.
        split: "train", "validation" ou "test".
        seed : graine pour le sous-échantillonnage aléatoire.

    Returns:
        Liste de TextExample.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("glue", "sst2", split=split)
        rng = np.random.default_rng(seed)
        n = min(limit, len(ds))
        indices = rng.choice(len(ds), size=n, replace=False).tolist()
        examples = [
            TextExample(
                text=ds[i]["sentence"].strip(),
                label=ds[i]["label"],
                idx=i,
            )
            for i in indices
        ]
        print(f"[data] Chargé {len(examples)} exemples depuis glue/sst2 ({split}).")
        return examples

    except Exception as e:
        print(f"[data] Fallback liste manuelle ({e}).")
        examples = _SST2_FALLBACK[:limit]
        return examples
