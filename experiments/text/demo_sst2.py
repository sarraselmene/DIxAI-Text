"""
Démo qualitative DIxAI-Text sur SST-2.

Affiche les tokens sélectionnés et les probabilités par token
pour quelques exemples SST-2.

Lancement:
    python experiments/text/demo_sst2.py
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
)
from dixai.text.data import load_sst2


def main():
    device = torch.device("cpu")
    model_name = "textattack/bert-base-uncased-SST-2"

    print("Chargement du modèle...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        attn_implementation="eager",
        use_safetensors=True,
    ).to(device)
    model.eval()

    # Baseline : mask_token (cohérent avec ERASER)
    explainer = TextDecisionInformationExplainer(
        model, tokenizer,
        lambda_fidelity=10.0,
        lambda_contiguity=2.0,
        baseline_type=BaselineType.MASK_TOKEN,
        device="cpu",
    )

    # Charger quelques exemples SST-2
    examples = load_sst2(limit=6, split="validation", seed=42)

    label_names = {0: "NEGATIVE", 1: "POSITIVE"}

    print("\n" + "=" * 70)
    print("DIxAI-Text — Démo qualitative SST-2")
    print("=" * 70)

    for ex in examples:
        print(f'\n>>> Texte : "{ex.text}"')
        if ex.label is not None:
            print(f"    Label vrai : {label_names.get(ex.label, ex.label)}")

        explanation = explainer.explain(
            ex.text,
            steps=200,
            lr=0.2,
            verbose=False,
            seed=0,
        )

        pred_label = label_names.get(explanation.predicted_class, explanation.predicted_class)
        print(f"    Classe prédite : {pred_label}")
        print(f"    {explanation}")

        # Tokens sélectionnés (threshold=0.5)
        selected = explanation.top_tokens(threshold=0.5)
        print(f"    Tokens sélectionnés (threshold=0.5) : {selected}")

        # Barre de probabilités
        print("    Probabilités par token :")
        for tok, p in zip(explanation.tokens, explanation.mask_probs.tolist()):
            bar = "█" * int(p * 20)
            marker = " <--" if p > 0.5 else ""
            print(f"      {tok:>15s} | {p:.3f} {bar}{marker}")

    print("\n" + "=" * 70)
    print("Fin de la démo.")


if __name__ == "__main__":
    main()
