"""
Démo : calcul des métriques ERASER (sufficiency, comprehensiveness)
sur des explications produites par TextDecisionInformationExplainer.

Lancement:
    python experiments/text/demo_eraser_metrics.py
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


def main():
    device = torch.device("cpu")
    model_name = "textattack/bert-base-uncased-SST-2"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        attn_implementation="eager",
        use_safetensors=True,
    ).to(device)
    model.eval()

    texts = [
        "This movie was not very good, honestly quite disappointing.",
        "An absolute masterpiece with brilliant performances.",
        "The plot was boring and the acting felt wooden.",
        "I really enjoyed every minute of this film.",
    ]

    # IMPORTANT : même baseline_type dans explainer ET dans ERASER
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

    print("=" * 70)
    print("DIxAI-Text — Métriques ERASER (threshold=0.5 et top_k=0.2)")
    print("=" * 70)

    for text in texts:
        print(f'\n>>> "{text}"')

        explanation = explainer.explain(
            text, steps=200, lr=0.2, verbose=False, seed=0
        )

        print(f"  {explanation}")
        print(f"  Classe prédite : {explanation.predicted_class}")
        print(f"  Tokens sélectionnés (threshold=0.5) : "
              f"{explanation.top_tokens(threshold=0.5)}")

        # Affichage barre de probabilités
        print("  Probabilités par token :")
        for tok, p in zip(explanation.tokens, explanation.mask_probs.tolist()):
            bar = "█" * int(p * 20)
            print(f"    {tok:>15s} | {p:.2f} {bar}")

        # Métriques ERASER (threshold ET top_k)
        scores = compute_eraser_scores_both_strategies(
            model, tokenizer, text,
            mask_probs=explanation.mask_probs,
            baseline_vec=baseline_vec,
            device="cpu",
            threshold=0.5,
            k_fraction=0.2,
        )

        print("\n  Métriques ERASER :")
        for strategy_name, s in scores.items():
            print(f"    [{strategy_name}]")
            print(f"      Sufficiency       (bas=mieux)  : {s.sufficiency:.4f}")
            print(f"      Comprehensiveness (haut=mieux) : {s.comprehensiveness:.4f}")
            print(f"      Tokens gardés : {s.n_tokens_kept}/{s.n_tokens_total}")

    print("\n" + "=" * 70)
    print("Fin de la démo ERASER.")


if __name__ == "__main__":
    main()
