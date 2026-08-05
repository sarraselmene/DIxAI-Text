import sys, os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dixai.text import TextDecisionInformationExplainer, BaselineType

device = torch.device("cpu")
MODEL_NAME = "textattack/bert-base-uncased-SST-2"

sentences = [
    "this movie was not very good, honestly quite disappointing.",
    "an absolute masterpiece, I loved every second of it.",
    "the plot was boring and the acting was terrible.",
    "surprisingly delightful, a hidden gem.",
    "waste of time, do not watch this.",
    "brilliant performances and a gripping story.",
    "I was not impressed at all.",
    "one of the best films I have seen this year.",
    "the pacing was slow but the ending redeemed it.",
    "utterly forgettable and poorly written.",
]

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    attn_implementation="eager",
    use_safetensors=True,
).to(device)
model.eval()

explainer = TextDecisionInformationExplainer(
    model, tok,
    lambda_fidelity=5.0,
    lambda_contiguity=0.5,
    baseline_type=BaselineType.MASK_TOKEN,
    device="cpu",
)

results = []
for s in sentences:
    exp = explainer.explain(s, steps=500, lr=0.15, verbose=False, seed=42)
    results.append(exp)
    print(f"\n>>> {s}")
    print(exp.top_tokens(threshold=0.5))
    print(f"Fidelity={exp.fidelity_score:.4f}  Sparsity={exp.sparsity_score:.4f}  Contiguity={exp.contiguity_score:.4f}")

fids = [r.fidelity_score for r in results]
sps = [r.sparsity_score for r in results]
cons = [r.contiguity_score for r in results]

print("\n=== MOYENNES SUR", len(sentences), "PHRASES ===")
print(f"Fidelity moyenne   : {sum(fids)/len(fids):.4f}")
print(f"Sparsity moyenne   : {sum(sps)/len(sps):.4f}")
print(f"Contiguity moyenne : {sum(cons)/len(cons):.4f}")
