"""
Experiment 3 — Baseline comparison on SST-2 (Verrou 3).

Compares DIxAI-Text against standard NLP attribution methods under the same
ERASER faithfulness protocol (comprehensiveness / sufficiency). Each method
produces a token saliency ranking; we threshold it at the DIxAI sparsity level
so the comparison is at matched rationale length.

Methods:
  - dixai            : the optimized Decision-Information mask (this work)
  - integrated_grads : Integrated Gradients on the embedding layer (captum)
  - attention        : last-layer [CLS] attention to each token
  - random           : random ranking (sanity lower bound)

LIME / SHAP / HEDGE hooks are left as clearly marked stubs (heavier deps).

    python experiments/text/compare_baselines.py --limit 50

Requires:  pip install -e ".[text]"   (+ captum for integrated_grads)
"""

import argparse

import torch
import torch.nn.functional as F


def integrated_gradients_ranking(wrapper, embeds, attention_mask, target_class, steps=32):
    """Integrated Gradients attribution per token (sum over embedding dim)."""
    baseline = torch.zeros_like(embeds)
    total_grad = torch.zeros_like(embeds)
    for alpha in torch.linspace(0, 1, steps):
        x = (baseline + alpha * (embeds - baseline)).clone().requires_grad_(True)
        logprobs = wrapper(x, attention_mask)
        score = logprobs[0, target_class]
        grad = torch.autograd.grad(score, x)[0]
        total_grad += grad
    ig = (embeds - baseline) * total_grad / steps
    return ig.sum(dim=-1).squeeze(0)  # (L,)


def attention_ranking(model, input_ids, attention_mask):
    """Mean last-layer attention from [CLS] to every token, if exposed."""
    out = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
    attentions = getattr(out, "attentions", None)
    if not attentions:
        return None
    last = attentions[-1][0]  # (heads, L, L)
    return last.mean(0)[0]     # CLS row, (L,)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="distilbert-base-uncased-finetuned-sst-2-english")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--methods", nargs="+",
                   default=["dixai", "integrated_grads", "attention", "random"])
    args = p.parse_args()

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from dixai.text import TextDecisionInformationExplainer
    from dixai.text.data import load_sst2
    from dixai.text.metrics import (
        make_eraser_scorer, comprehensiveness, sufficiency,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModelForSequenceClassification.from_pretrained(args.model)
    explainer = TextDecisionInformationExplainer(mdl, tok, device=device)

    examples = load_sst2(limit=args.limit)
    results = {m: {"comp": [], "suff": []} for m in args.methods}

    for ex in examples:
        exp = explainer.explain(ex.text, steps=args.steps, seed=0)
        enc = explainer._tokenize(ex.text, None)
        embeds = explainer.wrapper.embed(enc["input_ids"]).detach()
        baseline = explainer._baseline_embeddings(embeds, enc["attention_mask"]).detach()
        keep_mask = (
            enc["special_tokens_mask"].bool() | (enc["attention_mask"] == 0)
        ).float()
        predict = make_eraser_scorer(
            explainer.wrapper, embeds, baseline, enc["attention_mask"], keep_mask
        )
        content = exp.valid_mask
        c = exp.predicted_class
        k = max(1, int(exp.mask[content.bool()].sum().item()))  # matched length

        def top_k_mask(scores):
            s = scores.clone()
            s[content == 0] = float("-inf")
            idx = torch.argsort(s, descending=True)[:k]
            m = torch.zeros_like(content)
            m[idx] = 1.0
            return m

        for method in args.methods:
            if method == "dixai":
                rationale = exp.mask * content
            elif method == "integrated_grads":
                ig = integrated_gradients_ranking(
                    explainer.wrapper, embeds, enc["attention_mask"], c
                )
                rationale = top_k_mask(ig.abs())
            elif method == "attention":
                att = attention_ranking(mdl, enc["input_ids"], enc["attention_mask"])
                if att is None:
                    continue
                rationale = top_k_mask(att.detach())
            elif method == "random":
                rationale = top_k_mask(torch.rand_like(content))
            else:
                continue

            results[method]["comp"].append(
                comprehensiveness(predict, rationale, content, c)
            )
            results[method]["suff"].append(
                sufficiency(predict, rationale, content, c)
            )

    print("\n=== Baseline comparison (SST-2, matched sparsity) ===")
    print(f"{'method':>18} | {'comprehensiveness':>17} | {'sufficiency':>11}")
    print("-" * 54)
    for m in args.methods:
        comp = results[m]["comp"]
        suff = results[m]["suff"]
        if not comp:
            continue
        print(f"{m:>18} | {sum(comp)/len(comp):>17.4f} | {sum(suff)/len(suff):>11.4f}")

    print("\nNote: LIME / SHAP-text / HEDGE require extra packages; add them as")
    print("methods following the same top-k ERASER protocol above.")


if __name__ == "__main__":
    main()
