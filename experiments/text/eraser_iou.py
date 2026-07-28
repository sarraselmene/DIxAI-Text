"""
eraser_iou.py — Évaluation IoU DIxAI-Text vs annotations humaines ERASER.

Calcule l'IoU token-level entre :
  - les tokens sélectionnés par DIxAI-Text (threshold=0.5)
  - les annotations humaines du benchmark ERASER

Lancement:
    # Avec fichiers ERASER locaux :
    python experiments/text/eraser_iou.py --data_dir data/eraser/sst --dataset sst

    # Avec HuggingFace (e-SNLI) :
    python experiments/text/eraser_iou.py --dataset esnli --use_hf
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from dixai.text import (
    TextDecisionInformationExplainer,
    BaselineType,
)
from dixai.text.data_eraser import (
    load_eraser_sst,
    load_eraser_esnli,
    load_esnli_hf,
    ERASERExample,
)


# ---------------------------------------------------------------------------
# IoU token-level
# ---------------------------------------------------------------------------

def token_iou(pred: torch.Tensor, gold: torch.Tensor) -> float:
    """
    IoU token-level entre deux masques binaires (T,).

    Returns:
        float dans [0,1]. 1.0 = accord parfait.
    """
    p = pred.bool()
    g = gold.bool()
    intersection = (p & g).float().sum().item()
    union        = (p | g).float().sum().item()
    if union == 0:
        return 1.0
    return intersection / union


def align_eraser_mask_to_bert_tokens(
    tokenizer,
    text: str,
    eraser_tokens: list,
    eraser_gold_mask: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Aligne le masque gold ERASER (basé sur tokens whitespace)
    avec les tokens BERT (sous-mots).

    Stratégie : un token BERT est gold si le mot whitespace
    correspondant est gold.

    Args:
        tokenizer       : tokenizer HuggingFace.
        text            : texte original.
        eraser_tokens   : tokens whitespace ERASER.
        eraser_gold_mask: (T_eraser,) masque gold ERASER.
        device          : device.

    Returns:
        bert_gold_mask: (T_bert,) masque gold aligné sur tokens BERT.
    """
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        return_offsets_mapping=True,
    )
    offsets = enc["offset_mapping"][0].tolist()  # [(start_char, end_char)]
    T_bert  = len(offsets)

    # Construire mapping char -> word_idx (whitespace)
    char_to_word = {}
    char_pos = 0
    for word_idx, word in enumerate(eraser_tokens):
        for _ in word:
            char_to_word[char_pos] = word_idx
            char_pos += 1
        char_to_word[char_pos] = -1  # espace
        char_pos += 1

    bert_gold_mask = torch.zeros(T_bert, dtype=torch.float32)
    for t_idx, (start_char, end_char) in enumerate(offsets):
        if start_char == end_char:
            continue  # token spécial ([CLS], [SEP], [PAD])
        word_idx = char_to_word.get(start_char, -1)
        if 0 <= word_idx < len(eraser_gold_mask):
            bert_gold_mask[t_idx] = eraser_gold_mask[word_idx].item()

    return bert_gold_mask.to(device)


def get_special_token_mask(
    tokenizer,
    text: str,
    T: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Retourne un masque (T,) où 0 = token spécial ([CLS],[SEP],[PAD])
    et 1 = token normal.

    Args:
        tokenizer: tokenizer HuggingFace.
        text     : texte original.
        T        : longueur cible du masque.
        device   : device.

    Returns:
        special_mask: (T,) float, 0 sur tokens spéciaux.
    """
    enc = tokenizer(text, return_tensors="pt", truncation=True)
    input_ids = enc["input_ids"][0][:T]

    special_ids = {
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.pad_token_id,
    }
    special_mask = torch.tensor(
        [0.0 if tid.item() in special_ids else 1.0 for tid in input_ids],
        dtype=torch.float32,
    )
    # Padding si input_ids plus court que T
    if len(special_mask) < T:
        special_mask = torch.cat([
            special_mask,
            torch.zeros(T - len(special_mask)),
        ])
    return special_mask.to(device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="textattack/bert-base-uncased-SST-2")
    parser.add_argument("--dataset", default="sst", choices=["sst", "esnli"])
    parser.add_argument("--data_dir", default="data/eraser/sst",
                        help="Dossier ERASER local (si --use_hf non activé).")
    parser.add_argument("--use_hf", action="store_true",
                        help="Utiliser HuggingFace datasets (e-SNLI uniquement).")
    parser.add_argument("--split",  default="test")
    parser.add_argument("--limit",  type=int, default=50)
    parser.add_argument("--steps",  type=int, default=50)   # réduit pour la vitesse
    parser.add_argument("--seed",   type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cpu"

    # --- Modèle ---
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
    ).to(device)
    model.eval()

    # --- Explainer ---
    explainer = TextDecisionInformationExplainer(
        model, tokenizer,
        lambda_fidelity=10.0,
        lambda_contiguity=2.0,
        baseline_type=BaselineType.MASK_TOKEN,
        device=device,
    )

    # --- Chargement données ERASER ---
    if args.use_hf and args.dataset == "esnli":
        examples = load_esnli_hf(
            limit=args.limit, split=args.split, seed=args.seed,
        )
    elif args.dataset == "sst":
        examples = load_eraser_sst(
            data_dir=args.data_dir,
            split=args.split,
            limit=args.limit,
            seed=args.seed,
        )
    elif args.dataset == "esnli":
        examples = load_eraser_esnli(
            data_dir=args.data_dir,
            split=args.split,
            limit=args.limit,
            seed=args.seed,
        )
    else:
        raise ValueError(f"Dataset inconnu : {args.dataset}")

    # --- Calcul IoU ---
    iou_list       = []
    precision_list = []
    recall_list    = []

    print("\n" + "=" * 70)
    print(f"DIxAI-Text — IoU vs annotations ERASER ({args.dataset}, {args.split})")
    print("=" * 70)

    for i, ex in enumerate(examples):
        print(f"\n[{i+1}/{len(examples)}] {ex.text[:60]}...")

        # Explication DIxAI
        explanation = explainer.explain(
            ex.text, steps=args.steps, seed=args.seed,
        )

        # Masque prédit (threshold=0.5)
        pred_mask = (explanation.mask_probs > 0.5).float()  # (T_bert,)

        # Aligner masque gold ERASER -> tokens BERT
        if ex.gold_token_mask is not None and len(ex.tokens) > 0:
            gold_mask = align_eraser_mask_to_bert_tokens(
                tokenizer, ex.text,
                ex.tokens, ex.gold_token_mask,
                device=device,
            )
        else:
            print("  -> Pas d'annotations gold, skipped.")
            continue

        # Ajuster taille
        T         = min(pred_mask.shape[0], gold_mask.shape[0])
        pred_mask = pred_mask[:T]
        gold_mask = gold_mask[:T]

        # Exclure [CLS], [SEP], [PAD] du calcul
        special_mask = get_special_token_mask(tokenizer, ex.text, T, device)
        pred_mask    = pred_mask * special_mask
        gold_mask    = gold_mask * special_mask

        # IoU
        iou = token_iou(pred_mask, gold_mask)
        iou_list.append(iou)

        # Precision / Recall
        tp        = (pred_mask.bool() & gold_mask.bool()).float().sum().item()
        fp        = (pred_mask.bool() & ~gold_mask.bool()).float().sum().item()
        fn        = (~pred_mask.bool() & gold_mask.bool()).float().sum().item()
        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)
        precision_list.append(precision)
        recall_list.append(recall)

        print(f"  Tokens prédits : {explanation.top_tokens(threshold=0.5)}")
        print(f"  Gold spans     : {ex.gold_spans}")
        print(f"  IoU={iou:.4f} | Precision={precision:.4f} | Recall={recall:.4f}")

    # --- Résultats finaux ---
    print("\n" + "=" * 70)
    print("=== Résultats finaux ===")
    print(f"IoU moyen       : {np.mean(iou_list):.4f} ± {np.std(iou_list):.4f}")
    print(f"Precision moyen : {np.mean(precision_list):.4f} ± {np.std(precision_list):.4f}")
    print(f"Recall moyen    : {np.mean(recall_list):.4f} ± {np.std(recall_list):.4f}")
    print(f"N exemples      : {len(iou_list)}")


if __name__ == "__main__":
    main()
