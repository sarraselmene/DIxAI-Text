import sys, os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import torch.nn.functional as F
import numpy as np

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dixai.text import TextDecisionInformationExplainer, BaselineType

from lime.lime_text import LimeTextExplainer
from captum.attr import IntegratedGradients
import shap

device = torch.device("cpu")
MODEL_NAME = "textattack/bert-base-uncased-SST-2"
N_EXAMPLES = 5

dataset = load_dataset("glue", "sst2", split="validation").select(range(N_EXAMPLES))

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, attn_implementation="eager", use_safetensors=True,
).to(device)
model.eval()

# --- DIxAI ---
explainer = TextDecisionInformationExplainer(
    model, tok, lambda_fidelity=2.0, lambda_contiguity=2.0,
    baseline_type=BaselineType.MASK_TOKEN, device="cpu",
)

# --- LIME ---
lime_explainer = LimeTextExplainer(class_names=["negative", "positive"])
def lime_predict_proba(texts):
    enc = tok(list(texts), return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        probs = F.softmax(model(**enc).logits, dim=-1)
    return probs.numpy()

# --- SHAP ---
shap_masker = shap.maskers.Text(tok)
def shap_predict(texts):
    enc = tok(list(texts), return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        probs = F.softmax(model(**enc).logits, dim=-1)
    return probs.numpy()
shap_explainer = shap.Explainer(shap_predict, shap_masker)

# --- Integrated Gradients (captum) ---
embedding_layer = model.get_input_embeddings()
def forward_from_embeds(embeds, attention_mask):
    out = model(inputs_embeds=embeds, attention_mask=attention_mask)
    return F.softmax(out.logits, dim=-1)
ig = IntegratedGradients(forward_from_embeds)

for i, ex in enumerate(dataset):
    s = ex["sentence"]
    print(f"\n{'='*80}\n[{i+1}/{N_EXAMPLES}] {s.strip()}\n{'='*80}")

    enc = tok(s, return_tensors="pt").to(device)
    tokens = tok.convert_ids_to_tokens(enc["input_ids"][0])
    pred_class = model(**enc).logits.argmax(-1).item()

    # DIxAI
    exp = explainer.explain(s, steps=200, lr=0.2, verbose=False, seed=0)
    print("DIxAI          :", exp.top_tokens(threshold=0.5))

    # LIME
    try:
        lime_exp = lime_explainer.explain_instance(s, lime_predict_proba, num_features=6, num_samples=200)
        lime_top = [w for w, wt in lime_exp.as_list() if wt > 0]
        print("LIME           :", lime_top)
    except Exception as e:
        print("LIME           : ERREUR -", e)

    # SHAP
    try:
        shap_values = shap_explainer([s])
        vals = shap_values[0].values[:, pred_class] if shap_values[0].values.ndim > 1 else shap_values[0].values
        shap_toks = shap_values[0].data
        top_idx = np.argsort(np.abs(vals))[-6:][::-1]
        print("SHAP           :", [shap_toks[j] for j in top_idx])
    except Exception as e:
        print("SHAP           : ERREUR -", e)

    # Integrated Gradients
    try:
        embeds = embedding_layer(enc["input_ids"])
        baseline_embeds = torch.zeros_like(embeds)
        attributions = ig.attribute(embeds, baselines=baseline_embeds,
                                     additional_forward_args=(enc["attention_mask"],),
                                     target=pred_class, n_steps=50)
        attr_scores = attributions.sum(dim=-1).squeeze(0).detach().numpy()
        top_idx = np.argsort(np.abs(attr_scores))[-6:][::-1]
        print("IntegratedGrad :", [tokens[j] for j in top_idx])
    except Exception as e:
        print("IntegratedGrad : ERREUR -", e)

print("\n\nComparaison terminée.")
