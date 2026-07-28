"""
demo_snli.py — Démo DIxAI-Text sur SNLI (inférence textuelle).

Modèle NLI utilisé : cross-encoder/nli-deberta-v3-small
(ou tout modèle HF ForSequenceClassification entraîné sur NLI).

Lancement:
    python experiments/text/demo_snli.py
"""

import sys
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from dixai.text import (
    TextDecisionInformationExplainer,
    BaselineType,
    TextBaselineProvider,
)
from dixai.text.metrics_text import compute_eraser_scores_both_strategies
from dixai.text.data_snli import load_snli, LABEL_NAMES_SNLI


def main():
    device = torch.device("cpu")

    # Modèle NLI (entraîné sur SNLI/MNLI)
    # Alternative : "typeform/distilbert-base-uncased-mnli"
    model_name = "cross-encoder/nli-deberta-v3-small"

    print(f"Chargement modèle NLI : {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
    ).to(device)
    model.eval()

    # Baseline cohérente avec ERASER
    baseline_type = BaselineType.MASK_TOKEN
    baseline_provider = TextBaselineProvider(model, tokenizer, baseline_type)
    baseline_vec = baseline_provider.get_baseline_vector(device=device)

    explainer = TextDecisionInformationExplainer(
        model, tokenizer,
        lambda_fidelity=10.0,
        lambda_contiguity=2.0,
        baseline_type=baseline_type,
        device="cpu",
    )

    # Charger exemples SNLI
    examples = load_snli(limit=4, split="validation", seed=42)

    print("\n" + "=" * 70)
    print("DIxAI-Text — Démo qualitative SNLI (NLI)")
    print("=" * 70)

    for ex in examples:
        print(f"\n>>> Premise    : {ex.premise}")
        print(f"    Hypothesis : {ex.hypothesis}")
        print(f"    Label vrai : {LABEL_NAMES_SNLI.get(ex.label, ex.label)}")

        # Explication sur le texte concaténé
        explanation = explainer.explain(
            ex.text,
            steps=200,
            lr=0.2,
            verbose=False,
            seed=0,
        )

        pred_label = LABEL_NAMES_SNLI.get(
            explanation.predicted_class,
            explanation.predicted_class
        )
        print(f"    Classe prédite : {pred_label}")
        print(f"    {explanation}")

        # Tokens sélectionnés
        selected = explanation.top_tokens(threshold=0.5)
        print(f"    Tokens sélectionnés (threshold=0.5) : {selected}")

        # Probabilités par token
        print("    Probabilités par token :")
        for tok, p in zip(explanation.tokens, explanation.mask_probs.tolist()):
            bar = "█" * int(p * 20)
            marker = " <--" if p > 0.5 else ""
            print(f"      {tok:>15s} | {p:.3f} {bar}{marker}")

        # Métriques ERASER
        scores = compute_eraser_scores_both_strategies(
            model, tokenizer, ex.text,
            mask_probs=explanation.mask_probs,
            baseline_vec=baseline_vec,
            device="cpu",
            threshold=0.5,
            k_fraction=0.2,
        )

        print("\n    Métriques ERASER :")
        for strategy_name, s in scores.items():
            print(f"      [{strategy_name}]")
            print(f"        Sufficiency       (bas=mieux)  : {s.sufficiency:.4f}")
            print(f"        Comprehensiveness (haut=mieux) : {s.comprehensiveness:.4f}")
            print(f"        Tokens gardés : {s.n_tokens_kept}/{s.n_tokens_total}")

    print("\n" + "=" * 70)
    print("Fin de la démo SNLI.")


if __name__ == "__main__":
    main()
