#python tests/run_gumbel_demo.py
""" 

le fameux bug est fixe, Le changement est dans ton script de démo, tests/run_gumbel_demo.py —
 précisément la ligne qui définit model_name et les arguments de chargement du modèle :
 model_name = "textattack/bert-base-uncased-SST-2"   # au lieu de distilbert-...
 
"""
import torch
torch.set_num_threads(1)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dixai.text import TextDecisionInformationExplainer, BaselineType

model_name = "textattack/bert-base-uncased-SST-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    attn_implementation="eager",
    use_safetensors=True,
)
model.eval()

text = "This movie was not very good, honestly quite disappointing."

explainer = TextDecisionInformationExplainer(
    model, tokenizer,
    lambda_fidelity=10.0,
    lambda_contiguity=0.5,
    baseline_type=BaselineType.MASK_TOKEN,
    device="cpu",
)
explanation_a = explainer.explain(text, steps=300, lr=0.2, verbose=True, seed=0)
explanation_b = explainer.explain(text, steps=200, lr=0.05, verbose=True, seed=0)

def show(name, explanation):
    print(f"\n--- Résultat {name} ---")
    print(explanation)
    print("Tokens sélectionnés :", explanation.top_tokens(threshold=0.5))
    for tok, p in zip(explanation.tokens, explanation.mask_probs.tolist()):
        bar = "█" * int(p * 20)
        print(f"{tok:>15s} | {p:.2f} {bar}")

show("A (300 steps, lr=0.2)", explanation_a)
show("B (200 steps, lr=0.05)", explanation_b)
