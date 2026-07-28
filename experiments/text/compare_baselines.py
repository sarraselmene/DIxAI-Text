"""
compare_baseline.py — Comparaison complète des méthodes XAI sur SST-2.

Méthodes comparées :
  1. DIxAI-Text       (notre méthode)
  2. Integrated Grads (IG)
  3. Attention        ([CLS] -> token, dernière couche)
  4. LIME-texte       (perturbation locale)
  5. SHAP-texte       (valeurs de Shapley)
  6. Random           (borne inférieure)

Protocole ERASER : threshold=0.5, même baseline_type partout.
Matched sparsity : k = nb tokens sélectionnés par DIxAI.

Lancement:
    python experiments/text/compare_baselines.py --limit 50

Dépendances optionnelles :
    pip install lime shap
"""

import argparse
import os
import sys
import torch
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from dixai.text import (
    TextDecisionInformationExplainer,
    BaselineType,
    TextBaselineProvider,
)
from dixai.text.metrics_text import compute_eraser_scores
from dixai.text.data import load_sst2


# ---------------------------------------------------------------------------
# Utilitaires communs
# ---------------------------------------------------------------------------

def get_content_mask(tokenizer, text, device):
    """
    Retourne (input_ids, attention_mask, content_mask, special_tokens_mask).
    content_mask : (T,) 1=token de contenu, 0=spécial/padding.
    """
    enc = tokenizer(text, return_tensors="pt", truncation=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    special_ids_list = tokenizer.get_special_tokens_mask(
        input_ids[0].tolist(), already_has_special_tokens=True
    )
    special_tokens_mask = torch.tensor(
        special_ids_list, device=device, dtype=attention_mask.dtype
    )
    content_mask = (
        attention_mask[0] * (1 - special_tokens_mask)
    ).float()  # (T,)

    return input_ids, attention_mask, content_mask, special_tokens_mask


def topk_rationale(scores, content_mask, k):
    """
    Masque binaire (T,) : top-k tokens de contenu selon scores.
    """
    s = scores.clone().float()
    s[content_mask == 0] = -1e9
    idx = torch.argsort(s, descending=True)[:k]
    m = torch.zeros_like(content_mask, dtype=torch.float32)
    m[idx] = 1.0
    return m


# ---------------------------------------------------------------------------
# Méthode 1 : Integrated Gradients
# ---------------------------------------------------------------------------

def integrated_grads_scores(
    model,
    embedding_layer,
    input_ids,
    attention_mask,
    target_class,
    steps=32,
    device="cpu",
):
    """
    IG sur embeddings -> score par token (T,).
    score[t] = sum_d |IG[t,d]|
    """
    dev = torch.device(device)
    embeds = embedding_layer(input_ids.to(dev))   # (1, T, D)
    baseline = torch.zeros_like(embeds)
    total_grad = torch.zeros_like(embeds)

    for alpha in torch.linspace(0, 1, steps):
        x = (
            baseline + alpha * (embeds - baseline)
        ).clone().detach().requires_grad_(True)
        out = model(
            inputs_embeds=x,
            attention_mask=attention_mask.to(dev),
        )
        logits = out.logits if hasattr(out, "logits") else out
        score = logits[0, target_class]
        grad = torch.autograd.grad(score, x)[0]
        total_grad += grad.detach()

    ig = (embeds.detach() - baseline) * total_grad / steps  # (1, T, D)
    token_scores = ig.abs().sum(dim=-1).squeeze(0)           # (T,)
    return token_scores


# ---------------------------------------------------------------------------
# Méthode 2 : Attention
# ---------------------------------------------------------------------------

def attention_scores(model, input_ids, attention_mask, device="cpu"):
    """
    Score par token = moyenne des têtes d'attention (dernière couche),
    ligne [CLS] -> token t. Retourne (T,) ou None.
    """
    dev = torch.device(device)
    with torch.no_grad():
        out = model(
            input_ids=input_ids.to(dev),
            attention_mask=attention_mask.to(dev),
            output_attentions=True,
        )
    attentions = getattr(out, "attentions", None)
    if not attentions:
        return None
    last = attentions[-1][0]        # (heads, T, T)
    cls_row = last.mean(dim=0)[0]   # (T,) : ligne CLS moyennée sur têtes
    return cls_row.detach()


# ---------------------------------------------------------------------------
# Méthode 3 : LIME
# ---------------------------------------------------------------------------

def lime_scores(
    model,
    tokenizer,
    text,
    target_class,
    device="cpu",
    num_samples=300,
):
    """
    Score LIME par token (T,).
    Nécessite : pip install lime
    """
    try:
        from lime.lime_text import LimeTextExplainer
    except ImportError:
        print("  [LIME] Non disponible. Installe : pip install lime")
        return None

    dev = torch.device(device)

    def predict_fn(texts):
        enc = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            probs = torch.softmax(out.logits, dim=-1)
        return probs.cpu().numpy()

    # Nombre de classes
    n_classes = model.config.num_labels
    class_names = [str(i) for i in range(n_classes)]

    explainer_lime = LimeTextExplainer(class_names=class_names)
    try:
        exp = explainer_lime.explain_instance(
            text,
            predict_fn,
            num_features=50,
            num_samples=num_samples,
            labels=[target_class],
        )
    except Exception as e:
        print(f"  [LIME] Erreur : {e}")
        return None

    # Scores par mot LIME
    word_scores = dict(exp.as_list(label=target_class))

    # Aligner avec tokens BERT
    enc = tokenizer(text, return_tensors="pt", truncation=True)
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].tolist())
    T = len(tokens)
    scores = torch.zeros(T)

    for t_idx, tok in enumerate(tokens):
        clean_tok = tok.replace("##", "").lower().strip()
        best_score = 0.0
        for word, score in word_scores.items():
            if clean_tok and clean_tok in word.lower():
                best_score = max(best_score, abs(score))
        scores[t_idx] = best_score

    return scores


# ---------------------------------------------------------------------------
# Méthode 4 : SHAP
# ---------------------------------------------------------------------------

def shap_scores(
    model,
    tokenizer,
    text,
    target_class,
    device="cpu",
    max_evals=100,
):
    """
    Score SHAP par token (T,).
    Nécessite : pip install shap transformers
    """
    try:
        import shap
    except ImportError:
        print("  [SHAP] Non disponible. Installe : pip install shap")
        return None

    dev = torch.device(device)

    def predict_fn(texts):
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        enc = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            probs = torch.softmax(out.logits, dim=-1)
        return probs.cpu().numpy()

    try:
        # SHAP Partition explainer (recommandé pour texte)
        masker = shap.maskers.Text(tokenizer)
        explainer_shap = shap.Explainer(
            predict_fn,
            masker,
            output_names=[str(i) for i in range(model.config.num_labels)],
        )
        shap_values = explainer_shap(
            [text],
            max_evals=max_evals,
            batch_size=8,
        )
        # shap_values.values : (1, T_shap, n_classes)
        vals = shap_values.values[0, :, target_class]  # (T_shap,)
        scores_shap = torch.tensor(np.abs(vals), dtype=torch.float32)

    except Exception as e:
        print(f"  [SHAP] Erreur : {e}")
        return None

    # Aligner avec tokens BERT (SHAP peut avoir une tokenisation différente)
    enc = tokenizer(text, return_tensors="pt", truncation=True)
    T_bert = enc["input_ids"].shape[1]

    # Padding ou troncature pour aligner
    T_shap = scores_shap.shape[0]
    if T_shap >= T_bert:
        scores_aligned = scores_shap[:T_bert]
    else:
        pad = torch.zeros(T_bert - T_shap)
        scores_aligned = torch.cat([scores_shap, pad])

    return scores_aligned


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="textattack/bert-base-uncased-SST-2"
    )
    parser.add_argument(
        "--limit", type=int, default=50,
        help="Nombre d'exemples SST-2 à évaluer."
    )
    parser.add_argument(
        "--steps", type=int, default=200,
        help="Nombre de steps d'optimisation DIxAI."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ig_steps", type=int, default=32,
        help="Nombre de steps pour Integrated Gradients."
    )
    parser.add_argument(
        "--lime_samples", type=int, default=300,
        help="Nombre de samples LIME."
    )
    parser.add_argument(
        "--shap_evals", type=int, default=100,
        help="Nombre d'évaluations SHAP."
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["dixai", "integrated_grads", "attention", "lime", "shap", "random"],
        help="Méthodes à comparer."
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cpu"

    # --- Modèle & tokenizer ---
    print(f"Chargement modèle : {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        attn_implementation="eager",
        use_safetensors=True,
    ).to(device)
    mdl.eval()

    embedding_layer = mdl.get_input_embeddings()

    # --- Baseline UNIQUE pour tout le pipeline ---
    baseline_type = BaselineType.MASK_TOKEN
    baseline_provider = TextBaselineProvider(mdl, tok, baseline_type)
    baseline_vec = baseline_provider.get_baseline_vector(device=device)

    # --- Explainer DIxAI ---
    explainer = TextDecisionInformationExplainer(
        mdl, tok,
        lambda_fidelity=10.0,
        lambda_contiguity=2.0,
        baseline_type=baseline_type,
        device=device,
    )

    # --- Données SST-2 ---
    examples = load_sst2(limit=args.limit)

    # --- Résultats ---
    results = {m: {"suff": [], "comp": []} for m in args.methods}

    for i, ex in enumerate(examples):
        text = ex.text
        print(f"\n[{i+1}/{args.limit}] {text[:60]}...")

        # Tokenisation commune
        input_ids, attention_mask, content_mask, _ = get_content_mask(
            tok, text, device
        )
        T = input_ids.shape[1]
        n_content = int(content_mask.sum().item())

        if n_content == 0:
            print("  -> Skipped (no content tokens)")
            continue

        # --- DIxAI ---
        explanation = explainer.explain(
            text, steps=args.steps, seed=args.seed
        )
        dixai_mask_probs = explanation.mask_probs  # (T,) continu
        target_class = explanation.predicted_class

        # k matched : nb tokens sélectionnés par DIxAI (threshold=0.5)
        matched_k = max(1, int(
            (dixai_mask_probs > 0.5).float()[content_mask.bool()].sum().item()
        ))

        # Scores ERASER pour chaque méthode
        method_rationales = {}

        # 1) DIxAI : mask_probs continu -> threshold dans compute_eraser_scores
        if "dixai" in args.methods:
            method_rationales["dixai"] = dixai_mask_probs

        # 2) Integrated Gradients
        if "integrated_grads" in args.methods:
            ig = integrated_grads_scores(
                mdl, embedding_layer,
                input_ids, attention_mask,
                target_class=target_class,
                steps=args.ig_steps,
                device=device,
            )
            method_rationales["integrated_grads"] = topk_rationale(
                ig, content_mask, matched_k
            )

        # 3) Attention
        if "attention" in args.methods:
            att = attention_scores(mdl, input_ids, attention_mask, device=device)
            if att is not None:
                method_rationales["attention"] = topk_rationale(
                    att, content_mask, matched_k
                )

        # 4) LIME
        if "lime" in args.methods:
            lime = lime_scores(
                mdl, tok, text,
                target_class=target_class,
                device=device,
                num_samples=args.lime_samples,
            )
            if lime is not None:
                method_rationales["lime"] = topk_rationale(
                    lime, content_mask, matched_k
                )

        # 5) SHAP
        if "shap" in args.methods:
            shap = shap_scores(
                mdl, tok, text,
                target_class=target_class,
                device=device,
                max_evals=args.shap_evals,
            )
            if shap is not None:
                method_rationales["shap"] = topk_rationale(
                    shap, content_mask, matched_k
                )

        # 6) Random
        if "random" in args.methods:
            rand_scores = torch.rand(T)
            method_rationales["random"] = topk_rationale(
                rand_scores, content_mask, matched_k
            )

        # --- Calcul ERASER pour chaque méthode ---
        for method, mask_probs in method_rationales.items():
            # DIxAI : threshold=0.5 sur mask_probs continu
            # Autres : threshold=0.5 sur masque binaire {0,1}
            s = compute_eraser_scores(
                mdl, tok, text,
                mask_probs=mask_probs,
                baseline_vec=baseline_vec,
                device=device,
                strategy="threshold",
                threshold=0.5,
            )
            results[method]["suff"].append(s.sufficiency)
            results[method]["comp"].append(s.comprehensiveness)
            print(
                f"  {method:>18} | "
                f"Suff={s.sufficiency:.4f} | "
                f"Comp={s.comprehensiveness:.4f} | "
                f"Tokens={s.n_tokens_kept}/{s.n_tokens_total}"
            )

    # --- Tableau final ---
    print("\n" + "=" * 70)
    print("=== Résultats finaux (SST-2, ERASER threshold=0.5) ===")
    print("=" * 70)
    print(
        f"{'Méthode':>20} | "
        f"{'Comprehensiveness':>20} | "
        f"{'Sufficiency':>13}"
    )
    print("-" * 70)

    for method in args.methods:
        comp = results[method]["comp"]
        suff = results[method]["suff"]
        if not comp:
            print(f"{method:>20} | {'N/A':>20} | {'N/A':>13}")
            continue
        print(
            f"{method:>20} | "
            f"{np.mean(comp):>8.4f} ± {np.std(comp):.4f}     | "
            f"{np.mean(suff):>6.4f} ± {np.std(suff):.4f}"
        )

    print("\nNote: Comprehensiveness↑ = mieux. Sufficiency↓ = mieux.")
    print("      DIxAI-Text doit surpasser toutes les baselines.")


if __name__ == "__main__":
    main()
