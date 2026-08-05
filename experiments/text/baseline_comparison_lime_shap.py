"""
experiments/text/baseline_comparison_lime_shap.py

Calcule les explications LIME et SHAP pour les memes 50 phrases SST-2
utilisees pour DIxAI-Text, Random, Attention et Integrated Gradients,
puis evalue leur Sufficiency/Comprehensiveness avec le meme protocole
top-k (sparsite appariee) et le meme masquage par embedding baseline
(mean_embedding) que le reste du pipeline DIxAI-Text.

Installation:
    pip install lime shap

Lancement:
    python experiments/text/baseline_comparison_lime_shap.py
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from lime.lime_text import LimeTextExplainer
import shap

from dixai.text.baselines_text import TextBaselineProvider, BaselineType

MODEL_NAME = "textattack/bert-base-uncased-SST-2"
N_SAMPLES = 50
SEED = 42                 # A CONFIRMER : meme seed que Random/Attention/IG ?
TOP_K_FRACTION = 0.15     # A CONFIRMER : sparsite appariee a DIxAI-Text (~0.12-0.17 en ablation)

torch.set_num_threads(1)
device = torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, attn_implementation="eager", use_safetensors=True,
).to(device)
model.eval()
for p in model.parameters():
    p.requires_grad = False

baseline_provider = TextBaselineProvider(model, tokenizer, BaselineType.MEAN_EMBEDDING)
embedding_layer = model.get_input_embeddings()


def predict_proba(texts):
    """predict_fn requis par LIME et SHAP : renvoie les probabilites softmax."""
    enc = tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        logits = model(**enc).logits
    return F.softmax(logits, dim=-1).numpy()


def sufficiency_comprehensiveness(text, word_scores, top_k_fraction=TOP_K_FRACTION):
    """
    Suff/Comp calcules au niveau EMBEDDING (coherent avec le protocole
    DIxAI-Text), pas au niveau texte brut, pour une comparaison equitable.

    word_scores : liste de (mot, score d'importance) issue de LIME/SHAP.
    """
    enc = tokenizer(text, return_tensors="pt", truncation=True)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    T = input_ids.shape[1]

    special_ids_list = tokenizer.get_special_tokens_mask(
        input_ids[0].tolist(), already_has_special_tokens=True
    )
    special_tokens_mask = torch.tensor(special_ids_list).unsqueeze(0)

    # Alignement mot -> sous-tokens WordPiece (regle simple par prefixe "##")
    token_scores = np.zeros(T)
    word_score_map = {w.lower(): s for w, s in word_scores}
    for i, tok in enumerate(tokens):
        clean = tok.replace("##", "").lower()
        if clean in word_score_map:
            token_scores[i] = word_score_map[clean]

    n_content = int((attention_mask[0] * (1 - special_tokens_mask[0])).sum().item())
    k = max(1, int(round(top_k_fraction * n_content)))
    content_idx = [
        i for i in range(T)
        if attention_mask[0, i] == 1 and special_tokens_mask[0, i] == 0
    ]
    top_idx = sorted(content_idx, key=lambda i: -token_scores[i])[:k]

    with torch.no_grad():
        x_embeds = embedding_layer(input_ids)
        baseline = baseline_provider.get_baseline(seq_len=T).unsqueeze(0)
        y_orig = F.softmax(model(inputs_embeds=x_embeds, attention_mask=attention_mask).logits, dim=-1)

        mask_suff = torch.zeros(1, T)
        mask_suff[0, top_idx] = 1
        mask_suff[0, special_tokens_mask[0].bool()] = 1
        z_suff = mask_suff.unsqueeze(-1) * x_embeds + (1 - mask_suff.unsqueeze(-1)) * baseline
        y_suff = F.softmax(model(inputs_embeds=z_suff, attention_mask=attention_mask).logits, dim=-1)

        mask_comp = torch.ones(1, T)
        mask_comp[0, top_idx] = 0
        z_comp = mask_comp.unsqueeze(-1) * x_embeds + (1 - mask_comp.unsqueeze(-1)) * baseline
        y_comp = F.softmax(model(inputs_embeds=z_comp, attention_mask=attention_mask).logits, dim=-1)

    pred_class = y_orig.argmax(dim=-1).item()
    suff = (y_orig[0, pred_class] - y_suff[0, pred_class]).item()
    comp = (y_orig[0, pred_class] - y_comp[0, pred_class]).item()
    return suff, comp


def run_lime(text, num_features=10, num_samples=200):
    pred_class = int(np.argmax(predict_proba([text])[0]))
    explainer = LimeTextExplainer(class_names=["negative", "positive"])
    exp = explainer.explain_instance(
        text, predict_proba, num_features=num_features,
        num_samples=num_samples, labels=(pred_class,)
    )
    return exp.as_list(label=pred_class)


def run_shap(text):
    masker = shap.maskers.Text(tokenizer=r"\W+")
    explainer = shap.Explainer(predict_proba, masker)
    sv = explainer([text])
    pred_class = int(np.argmax(predict_proba([text])[0]))
    words = sv.data[0]
    scores = sv.values[0][:, pred_class]
    return list(zip(words, scores))


def main():
    dataset = load_dataset("glue", "sst2", split="validation")
    rng = np.random.default_rng(SEED)
    indices = rng.choice(len(dataset), size=N_SAMPLES, replace=False)
    sentences = [dataset[int(i)]["sentence"] for i in indices]

    lime_suffs, lime_comps = [], []
    shap_suffs, shap_comps = [], []

    for i, text in enumerate(sentences):
        print(f"[{i+1}/{N_SAMPLES}] {text[:60]}...")

        word_scores_lime = run_lime(text)
        s, c = sufficiency_comprehensiveness(text, word_scores_lime)
        lime_suffs.append(s)
        lime_comps.append(c)

        word_scores_shap = run_shap(text)
        s, c = sufficiency_comprehensiveness(text, word_scores_shap)
        shap_suffs.append(s)
        shap_comps.append(c)

    print(f"\nLIME  -> Comp={np.mean(lime_comps):.4f} +- {np.std(lime_comps):.4f}"
          f"  Suff={np.mean(lime_suffs):.4f} +- {np.std(lime_suffs):.4f}")
    print(f"SHAP  -> Comp={np.mean(shap_comps):.4f} +- {np.std(shap_comps):.4f}"
          f"  Suff={np.mean(shap_suffs):.4f} +- {np.std(shap_suffs):.4f}")


if __name__ == "__main__":
    main()