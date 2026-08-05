"""
Experiment 3 — Baseline comparison on SST-2 (Verrou 3).

Compares DIxAI-Text against standard NLP attribution methods under ERASER
faithfulness protocol.

Methods:
  - dixai            : Decision-Information mask
  - integrated_grads : Integrated Gradients
  - attention        : CLS attention
  - random           : Random baseline

Run:
    python experiments/text/compare_baseline.py --limit 50
"""


import argparse
import torch


# ============================================================
# Integrated Gradients
# ============================================================

def integrated_gradients_ranking(
        wrapper,
        embeds,
        attention_mask,
        target_class,
        steps=32
):
    """
    Integrated Gradients attribution per token.
    """

    baseline = torch.zeros_like(embeds)

    total_grad = torch.zeros_like(embeds)


    for alpha in torch.linspace(0, 1, steps):

        x = (
            baseline +
            alpha * (embeds - baseline)
        ).clone().requires_grad_(True)


        logits = wrapper(
            x,
            attention_mask
        )


        score = logits[0, target_class]


        grad = torch.autograd.grad(
            score,
            x,
            retain_graph=False
        )[0]


        total_grad += grad


    ig = (
        (embeds - baseline)
        *
        total_grad
        /
        steps
    )


    return ig.sum(dim=-1).squeeze(0)



# ============================================================
# Attention attribution
# ============================================================

def attention_ranking(
        model,
        input_ids,
        attention_mask
):
    """
    CLS attention from last transformer layer.
    """

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True
    )


    attentions = getattr(
        output,
        "attentions",
        None
    )


    if attentions is None:
        return None


    last = attentions[-1][0]


    return last.mean(0)[0]



# ============================================================
# Main experiment
# ============================================================

def main():


    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--model",
        default="distilbert-base-uncased-finetuned-sst-2-english"
    )


    parser.add_argument(
        "--limit",
        type=int,
        default=50
    )


    parser.add_argument(
        "--steps",
        type=int,
        default=300
    )


    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "dixai",
            "integrated_grads",
            "attention",
            "random"
        ]
    )


    args = parser.parse_args()



    # imports locaux

    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer
    )


    from dixai.text import (
        TextDecisionInformationExplainer
    )


    from dixai.text.data import (
        load_sst2
    )


    from dixai.text.metrics_text import (
        compute_eraser_scores
    )



    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )



    print("Loading model...")


    tokenizer = AutoTokenizer.from_pretrained(
        args.model
    )


    model = AutoModelForSequenceClassification.from_pretrained(
        args.model
    ).to(device)



    explainer = TextDecisionInformationExplainer(
        model,
        tokenizer,
        device=device
    )



    examples = load_sst2(
        limit=args.limit
    )



    results = {

        method:
        {
            "comp": [],
            "suff": []
        }

        for method in args.methods
    }



    # =====================================================
    # Evaluation loop
    # =====================================================


    for idx, ex in enumerate(examples):


        print(
            f"\nExample {idx+1}/{len(examples)}"
        )


        explanation = explainer.explain(
            ex.text,
            steps=args.steps,
            seed=0
        )



        enc = explainer._tokenize(
            ex.text,
            None
        )



        embeds = explainer.wrapper.embed(
            enc["input_ids"]
        ).detach()



        baseline = (
            explainer
            ._baseline_embeddings(
                embeds,
                enc["attention_mask"]
            )
            .detach()
        )



        content = explanation.valid_mask



        target_class = (
            explanation.predicted_class
        )



        # same rationale length for every method

        k = max(
            1,
            int(
                explanation
                .mask[content.bool()]
                .sum()
                .item()
            )
        )



        def top_k_mask(scores):

            scores = scores.clone()


            scores[
                content == 0
            ] = float("-inf")


            indices = torch.argsort(
                scores,
                descending=True
            )[:k]


            mask = torch.zeros_like(
                content
            )


            mask[indices] = 1.0


            return mask



        # =================================================
        # Compare methods
        # =================================================


        for method in args.methods:



            if method == "dixai":

                rationale = (
                    explanation.mask
                    *
                    content
                )



            elif method == "integrated_grads":


                ig = integrated_gradients_ranking(
                    explainer.wrapper,
                    embeds,
                    enc["attention_mask"],
                    target_class
                )


                rationale = top_k_mask(
                    ig.abs()
                )



            elif method == "attention":


                att = attention_ranking(
                    model,
                    enc["input_ids"],
                    enc["attention_mask"]
                )


                if att is None:
                    continue


                rationale = top_k_mask(
                    att.detach()
                )



            elif method == "random":


                rationale = top_k_mask(
                    torch.rand_like(content)
                )


            else:

                continue



            scores = compute_eraser_scores(
                wrapper=explainer.wrapper,
                embeddings=embeds,
                baseline_embeddings=baseline,
                attention_mask=enc["attention_mask"],
                rationale_mask=rationale,
                target_class=target_class
            )



            results[method]["comp"].append(
                scores["comprehensiveness"]
            )


            results[method]["suff"].append(
                scores["sufficiency"]
            )



    # =====================================================
    # Results
    # =====================================================


    print(
        "\n\n=== Baseline comparison (SST-2) ==="
    )


    print(
        f"{'Method':>20} | {'Comprehensiveness':>18} | {'Sufficiency':>12}"
    )


    print("-"*60)



    for method in args.methods:


        comp = results[method]["comp"]

        suff = results[method]["suff"]



        if len(comp) == 0:
            continue



        print(
            f"{method:>20} | "
            f"{sum(comp)/len(comp):18.4f} | "
            f"{sum(suff)/len(suff):12.4f}"
        )



if __name__ == "__main__":
    main()