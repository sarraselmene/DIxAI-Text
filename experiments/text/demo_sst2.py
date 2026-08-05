"""
Démo Semaine 3-4 : DIxAI-Text sur un modèle BERT fine-tuné pour la classification de
sentiments (SST-2 style). À exécuter en LOCAL (nécessite un accès réseau à HuggingFace,
absent du présent environnement sandbox).

Installation:
    pip install transformers

Lancement:
    python experiments/text/demo_sst2.py
"""
import sys, os
# 1. Résout l'erreur OpenMP (Error #15)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 2. Désactive les optimisations de parallélisation conflictuelles sur Mac M1/M2/M3
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dixai.text import TextDecisionInformationExplainer, BaselineType


def main():
    # Force l'appareil cible sur le CPU pour éviter le Bus Error lié au GPU/MPS de macOS
    device = torch.device("cpu")

    model_name = "textattack/bert-base-uncased-SST-2"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 3. On charge explicitement le modèle directement sur le CPU
    model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    attn_implementation="eager",
    use_safetensors=True,
    ).to(device)
   
    model.eval()

    text = "This movie was not very good, honestly quite disappointing."

    for baseline_type in [BaselineType.MASK_TOKEN, BaselineType.PAD_TOKEN, BaselineType.MEAN_EMBEDDING]:
        print(f"\n=== Baseline: {baseline_type.value} ===")
        explainer = TextDecisionInformationExplainer(
            model,
            tokenizer,
            lambda_fidelity=2.0,
            lambda_contiguity=2.0,
            baseline_type=baseline_type,
            device="cpu",
        )

        #explanation = explainer.explain(text, steps=200, lr=0.2, verbose=False, seed=0)
        explanation = explainer.explain(text, steps=200, lr=0.2, verbose=False, seed=0, debug=True)
        print(explanation)
        print("Tokens sélectionnés :", explanation.top_tokens(threshold=0.5))

        for tok, p in zip(explanation.tokens, explanation.mask_probs.tolist()):
            print(f"  {tok:>15s} : {p:.2f}")


if __name__ == "__main__":
    main()
