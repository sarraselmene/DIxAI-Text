"""
Démo : calcul des métriques ERASER (sufficiency, comprehensiveness) sur des
explications produites par TextDecisionInformationExplainer.

Lancement:
    python experiments/text/demo_eraser_metrics.py
"""
import sys, os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dixai.text import TextDecisionInformationExplainer, BaselineType, TextBaselineProvider
from dixai.text.metrics_text import compute_eraser_scores_both_strategies


def main():
    device = torch.device("cpu")
    model_name = "textattack/bert-base-uncased-SST-2"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, attn_implementation="eager", use_safetensors=True
    ).to(device)
    model.eval()

    texts = [
        "This movie was not very good, honestly quite disappointing.",
        "An absolute masterpiece with brilliant performances.",
    ]

    baseline_type = BaselineType.MASK_TOKEN
    baseline_provider = TextBaselineProvider(model, tokenizer, baseline_type)
    baseline_vec = baseline_provider.get_baseline_vector()

    explainer = TextDecisionInformationExplainer(
        model, tokenizer,
        lambda_fidelity=10.0, lambda_contiguity=2.0,
        baseline_type=baseline_type, device="cpu",
    )

    for text in texts:
        print(f"\n=== \"{text}\" ===")
        explanation = explainer.explain(text, steps=200, lr=0.2, verbose=False, seed=0)
        print(explanation)
        print("Tokens sélectionnés (seuil 0.5) :", explanation.top_tokens(threshold=0.5))

        scores = compute_eraser_scores_both_strategies(
            model, tokenizer, text,
            mask_probs=explanation.mask_probs,
            baseline_vec=baseline_vec,
            device="cpu",
            threshold=0.5,
            k_fraction=0.2,
        )

        for strategy_name, s in scores.items():
            print(f"\n  Stratégie: {strategy_name}")
            print(f"    Sufficiency      (bas=mieux) : {s.sufficiency:.4f}")
            print(f"    Comprehensiveness (haut=mieux): {s.comprehensiveness:.4f}")
            print(f"    Tokens gardés: {s.n_tokens_kept}/{s.n_tokens_total}")


if __name__ == "__main__":
    main()