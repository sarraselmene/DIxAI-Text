# test_verrou1_pipeline.py
"""
Test du Verrou 1 : masquage AVANT les 12 couches (via bert_wrapper),
prédiction via le vrai modèle fine-tuné SST-2 (via sentiment_model).
"""

import torch
from dixai.nlp.bert_wrapper import BertWrapper
from dixai.nlp.sentiment_wrapper import BertSentimentWrapper
from dixai.explainer import DecisionInformationExplainer

TEXT = "The movie was truly excellent and touching."


def test_masking_changes_prediction_through_bert():
    print("\n--- Test : le masquage a un effet réel via les 12 couches ---")

    bert_wrapper = BertWrapper("bert-base-uncased")
    sentiment_model = BertSentimentWrapper("textattack/bert-base-uncased-SST-2")

    embeddings, attention_mask, input_ids, *_ = bert_wrapper.get_input_embeddings(TEXT)

    mask_id = bert_wrapper.tokenizer.mask_token_id
    mask_ids = torch.full_like(input_ids, mask_id)
    with torch.no_grad():
        baseline = bert_wrapper.model.embeddings(input_ids=mask_ids)

        y_orig = sentiment_model.forward_from_embeddings(embeddings, attention_mask)

        # On ne garde qu'un seul token (test extrême)
        hard_mask = torch.zeros(1, embeddings.shape[1], 1)
        hard_mask[:, 1, :] = 1.0
        z = embeddings * hard_mask + baseline * (1 - hard_mask)
        y_masked = sentiment_model.forward_from_embeddings(z, attention_mask)

    diff = (y_orig - y_masked).abs().max().item()
    print(f"y_orig   = {y_orig}")
    print(f"y_masked = {y_masked}")
    print(f"écart    = {diff:.6f}")

    assert diff > 1e-3, "Le masquage extrême ne change presque rien à la prédiction."
    print("✓ OK : masquer des tokens change bien la prédiction.")


def test_explain_text_pipeline():
    print("\n--- Test : pipeline complet explain_text() ---")

    bert_wrapper = BertWrapper("bert-base-uncased")
    sentiment_model = BertSentimentWrapper("textattack/bert-base-uncased-SST-2")

    explainer = DecisionInformationExplainer(
        model=sentiment_model,     # non utilisé pour le texte, mais requis par __init__
        lambda_fidelity=10.0,
        task="classification",
        lambda_tv=0.5,
    )

    explanation = explainer.explain_text(
        TEXT,
        bert_wrapper,
        sentiment_model,           # passé explicitement — voir correction ci-dessus
        steps=60,
        lr=0.1,
        temperature=1.0,
        anneal=True,
        verbose=True,
        seed=42,
    )

    losses = explanation.history["loss"]
    print(f"\nLoss step 0   : {losses[0]:.4f}")
    print(f"Loss dernière : {losses[-1]:.4f}")
    assert losses[-1] < losses[0], "La loss n'a pas diminué."

    input_ids = bert_wrapper.tokenizer(TEXT)["input_ids"]
    tokens = bert_wrapper.tokenizer.convert_ids_to_tokens(input_ids)
    scores = explanation.mask_probs.squeeze().tolist()

    print(f"\nFidélité finale : {explanation.fidelity_score:.3f}")
    print(f"Densité (info)  : {explanation.info_score:.3f}")
    print("\nToken\t\tScore\tGardé ?")
    for tok, score in zip(tokens, scores):
        garde = "OUI" if score > 0.5 else "non"
        print(f"{tok:15s}\t{score:.3f}\t{garde}")

    print("✓ OK : pipeline complet exécuté, loss décroissante.")


if __name__ == "__main__":
    test_masking_changes_prediction_through_bert()
    test_explain_text_pipeline()
    print("\n=== Tests Verrou 1 passés ===")