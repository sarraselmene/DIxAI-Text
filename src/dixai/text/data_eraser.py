"""
data_eraser.py — Chargement du benchmark ERASER officiel.

Supporte :
  - SST  (sentiment, annotations token-level)
  - e-SNLI (NLI avec explications humaines)

Format ERASER : fichiers JSONL avec champs
  annotation_id, classification, evidences, document

Usage:
    from dixai.text.data_eraser import load_eraser_sst, load_eraser_esnli
    examples = load_eraser_sst(data_dir="data/eraser/sst", split="test")
    for ex in examples:
        print(ex.text)
        print(ex.gold_token_mask)  # annotations humaines
"""

import os
import json
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Structures de données
# ---------------------------------------------------------------------------

@dataclass
class ERASERExample:
    """
    Un exemple du benchmark ERASER.

    Attributs:
        annotation_id  : identifiant unique de l'exemple.
        text           : texte complet (document).
        label          : label de classification (str ou int).
        tokens         : liste de tokens (si pré-tokenisé).
        gold_spans     : liste de (start_token, end_token) annotés par humains.
        gold_token_mask: (T,) masque binaire des tokens annotés.
    """
    annotation_id: str
    text: str
    label: str
    tokens: List[str] = field(default_factory=list)
    gold_spans: List[Tuple[int, int]] = field(default_factory=list)
    gold_token_mask: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def _parse_label_sst(label_str: str) -> int:
    """Convertit label SST string -> int."""
    mapping = {
        "POS": 1, "NEG": 0,
        "POSITIVE": 1, "NEGATIVE": 0,
        "1": 1, "0": 0,
    }
    return mapping.get(label_str.upper(), -1)


def _parse_label_nli(label_str: str) -> int:
    """Convertit label NLI string -> int."""
    mapping = {
        "ENTAILMENT": 0,
        "NEUTRAL": 1,
        "CONTRADICTION": 2,
        "entailment": 0,
        "neutral": 1,
        "contradiction": 2,
    }
    return mapping.get(label_str, -1)


def _build_gold_token_mask(
    tokens: List[str],
    gold_spans: List[Tuple[int, int]],
) -> torch.Tensor:
    """
    Construit un masque binaire (T,) à partir des spans annotés.

    Args:
        tokens    : liste de tokens du document.
        gold_spans: liste de (start_token, end_token) [end exclusif].

    Returns:
        gold_mask: (T,) binaire.
    """
    T = len(tokens)
    mask = torch.zeros(T, dtype=torch.float32)
    for (start, end) in gold_spans:
        start = max(0, start)
        end = min(T, end)
        mask[start:end] = 1.0
    return mask


# ---------------------------------------------------------------------------
# Chargement SST ERASER
# ---------------------------------------------------------------------------

def load_eraser_sst(
    data_dir: str,
    split: str = "test",
    limit: Optional[int] = None,
    seed: int = 42,
) -> List[ERASERExample]:
    """
    Charge le dataset SST du benchmark ERASER.

    Args:
        data_dir: chemin vers le dossier ERASER/sst/
        split   : "train", "val" ou "test".
        limit   : nombre max d'exemples (None = tous).
        seed    : graine pour sous-échantillonnage.

    Returns:
        Liste de ERASERExample.
    """
    split_file_map = {
        "train": "train.jsonl",
        "val":   "val.jsonl",
        "validation": "val.jsonl",
        "test":  "test.jsonl",
    }
    filename = split_file_map.get(split, f"{split}.jsonl")
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Fichier ERASER SST introuvable : {filepath}\n"
            f"Télécharge depuis : https://www.eraserbenchmark.com/zipped/sst.tar.gz"
        )

    examples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)

            text      = data.get("document", "")
            tokens    = text.split()
            label_str = data.get("classification", "")

            gold_spans = []
            for evidence_group in data.get("evidences", []):
                for ev in evidence_group:
                    start = ev.get("start_token", 0)
                    end   = ev.get("end_token", 0)
                    gold_spans.append((start, end))

            gold_mask = _build_gold_token_mask(tokens, gold_spans)

            examples.append(ERASERExample(
                annotation_id=data.get("annotation_id", ""),
                text=text,
                label=label_str,
                tokens=tokens,
                gold_spans=gold_spans,
                gold_token_mask=gold_mask,
            ))

    if limit is not None and limit < len(examples):
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(examples), size=limit, replace=False).tolist()
        examples = [examples[i] for i in indices]

    print(f"[ERASER SST] Chargé {len(examples)} exemples ({split}).")
    return examples


# ---------------------------------------------------------------------------
# Chargement e-SNLI ERASER
# ---------------------------------------------------------------------------

def load_eraser_esnli(
    data_dir: str,
    split: str = "test",
    limit: Optional[int] = None,
    seed: int = 42,
) -> List[ERASERExample]:
    """
    Charge le dataset e-SNLI du benchmark ERASER.

    Args:
        data_dir: chemin vers le dossier ERASER/esnli/
        split   : "train", "val" ou "test".
        limit   : nombre max d'exemples.
        seed    : graine.

    Returns:
        Liste de ERASERExample.
    """
    split_file_map = {
        "train": "train.jsonl",
        "val":   "val.jsonl",
        "validation": "val.jsonl",
        "test":  "test.jsonl",
    }
    filename = split_file_map.get(split, f"{split}.jsonl")
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Fichier ERASER e-SNLI introuvable : {filepath}\n"
            f"Télécharge depuis : https://www.eraserbenchmark.com/zipped/esnli.tar.gz"
        )

    # Charge docs.jsonl pour récupérer les textes complets
    docs_path = os.path.join(data_dir, "docs.jsonl")
    docs = {}
    if os.path.exists(docs_path):
        with open(docs_path, "r", encoding="utf-8") as df:
            for dline in df:
                dline = dline.strip()
                if not dline:
                    continue
                doc = json.loads(dline)
                docs[doc["docid"]] = doc["document"]

    examples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)

            ann_id     = data.get("annotation_id", "")
            premise    = docs.get(f"{ann_id}_premise", "")
            hypothesis = docs.get(f"{ann_id}_hypothesis", "")
            text       = premise + " " + hypothesis
            tokens     = text.split()
            label_str  = data.get("classification", "")

            # Offset : les spans hypothesis sont relatifs à leur propre doc,
            # il faut les décaler de len(premise_tokens) + 1 (espace)
            premise_len = len(premise.split())

            gold_spans = []
            for evidence_group in data.get("evidences", []):
                for ev in evidence_group:
                    start = ev.get("start_token", 0)
                    end   = ev.get("end_token", 0)
                    docid = ev.get("docid", "")
                    # Décalage si span vient de l'hypothesis
                    if docid.endswith("_hypothesis"):
                        start += premise_len + 1
                        end   += premise_len + 1
                    gold_spans.append((start, end))

            gold_mask = _build_gold_token_mask(tokens, gold_spans)

            examples.append(ERASERExample(
                annotation_id=ann_id,
                text=text,
                label=label_str,
                tokens=tokens,
                gold_spans=gold_spans,
                gold_token_mask=gold_mask,
            ))

    if limit is not None and limit < len(examples):
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(examples), size=limit, replace=False).tolist()
        examples = [examples[i] for i in indices]

    print(f"[ERASER e-SNLI] Chargé {len(examples)} exemples ({split}).")
    return examples


# ---------------------------------------------------------------------------
# Fallback HuggingFace (si pas de fichiers locaux)
# ---------------------------------------------------------------------------

def _extract_marked_tokens(
    marked_text: str,
    offset: int = 0,
) -> List[Tuple[int, int]]:
    """Extrait les spans des tokens marqués avec * dans e-SNLI HF."""
    spans = []
    words = marked_text.split()
    pos = offset
    for word in words:
        if word.startswith("*") and word.endswith("*"):
            spans.append((pos, pos + 1))
        pos += 1
    return spans


def load_esnli_hf(
    limit: int = 50,
    split: str = "validation",
    seed: int = 42,
) -> List[ERASERExample]:
    """
    Charge e-SNLI depuis HuggingFace datasets.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("esnli", split=split, trust_remote_code=False)
    except Exception as e:
        raise RuntimeError(
            f"Impossible de charger e-SNLI depuis HuggingFace : {e}\n"
            f"Installe datasets : pip install datasets"
        )

    rng = np.random.default_rng(seed)
    n = min(limit, len(ds))
    indices = rng.choice(len(ds), size=n, replace=False).tolist()

    label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}

    examples = []
    for i in indices:
        row        = ds[int(i)]
        premise    = row["premise"].strip()
        hypothesis = row["hypothesis"].strip()
        text       = premise + " " + hypothesis
        tokens     = text.split()
        label_str  = label_map.get(row["label"], "unknown")

        gold_spans = []
        marked1 = row.get("sentence1_marked_1", "")
        marked2 = row.get("sentence2_marked_1", "")
        if marked1:
            gold_spans += _extract_marked_tokens(marked1, offset=0)
        if marked2:
            gold_spans += _extract_marked_tokens(
                marked2, offset=len(premise.split())
            )

        gold_mask = _build_gold_token_mask(tokens, gold_spans)

        examples.append(ERASERExample(
            annotation_id=str(i),
            text=text,
            label=label_str,
            tokens=tokens,
            gold_spans=gold_spans,
            gold_token_mask=gold_mask,
        ))

    print(f"[e-SNLI HF] Chargé {len(examples)} exemples ({split}).")
    return examples
