"""
Tests unitaires pour Verrou 1 (masks_text) et Verrou 3 (objective_text) avec des
tenseurs synthétiques : ne nécessite PAS de télécharger BERT, donc exécutable hors-ligne.

Lancer avec: pytest tests/test_text_masking.py -v
"""
import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dixai.text.masks_text import TokenGumbelSoftmaxMask
from dixai.text.objective_text import DecisionInformationLossText


def test_mask_output_shapes():
    B, T, D = 2, 6, 8
    x = torch.randn(B, T, D)
    baseline = torch.randn(T, D)
    mask_module = TokenGumbelSoftmaxMask(seq_len=T)

    z, mask = mask_module(x, baseline, training=True)

    assert z.shape == (B, T, D)
    assert mask.shape == (B, T)
    assert torch.all(mask >= 0) and torch.all(mask <= 1)


def test_mask_respects_padding():
    B, T, D = 1, 5, 4
    x = torch.randn(B, T, D)
    baseline = torch.zeros(T, D)
    attention_mask = torch.tensor([[1, 1, 1, 0, 0]])  # 2 tokens de padding

    mask_module = TokenGumbelSoftmaxMask(seq_len=T, init_logits=5.0)  # forcer mask ~ 1
    _, mask = mask_module(x, baseline, training=False, attention_mask=attention_mask)

    # Le padding doit toujours avoir mask == 0
    assert torch.allclose(mask[0, 3:], torch.zeros(2))
    # Les tokens réels avec init_logits=5.0 doivent être proches de 1
    assert torch.all(mask[0, :3] > 0.9)


def test_baseline_recovers_original_when_mask_is_one():
    # Si mask == 1 partout, z doit être exactement x (pas de mélange avec la baseline)
    B, T, D = 1, 4, 3
    x = torch.randn(B, T, D)
    baseline = torch.randn(T, D)

    mask_module = TokenGumbelSoftmaxMask(seq_len=T, init_logits=10.0)
    z, mask = mask_module(x, baseline, training=False)

    assert torch.allclose(z, x, atol=1e-3)


def test_baseline_recovers_when_mask_is_zero():
    B, T, D = 1, 4, 3
    x = torch.randn(B, T, D)
    baseline = torch.randn(T, D)

    mask_module = TokenGumbelSoftmaxMask(seq_len=T, init_logits=-10.0)
    z, mask = mask_module(x, baseline, training=False)

    # sigmoid(-10) ~ 4.5e-5, donc pas exactement 0 -> tolérance légèrement relâchée
    assert torch.allclose(z, baseline.unsqueeze(0), atol=1e-3)


def test_contiguity_penalty_prefers_contiguous_spans():
    """
    Deux masks avec exactement le même nombre de tokens sélectionnés (même sparsité),
    mais l'un contigu ("span") et l'autre épars ("scattered") : la pénalité de
    contiguïté doit être strictement plus faible pour le mask contigu.
    """
    loss_fn = DecisionInformationLossText(lambda_contiguity=1.0)

    contiguous = torch.tensor([[0.0, 1.0, 1.0, 1.0, 0.0, 0.0]])
    scattered = torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]])

    tv_contiguous = loss_fn._sequential_contiguity(contiguous)
    tv_scattered = loss_fn._sequential_contiguity(scattered)

    assert tv_contiguous < tv_scattered


def test_objective_forward_returns_four_terms():
    B, T = 2, 5
    n_classes = 3
    mask = torch.rand(B, T)
    y_orig = torch.log_softmax(torch.randn(B, n_classes), dim=-1)
    y_masked = torch.log_softmax(torch.randn(B, n_classes), dim=-1)
    attention_mask = torch.ones(B, T)

    loss_fn = DecisionInformationLossText(lambda_fidelity=2.0, lambda_contiguity=0.5)
    total, info, fidelity, contiguity = loss_fn(mask, y_orig, y_masked, attention_mask)

    for t in (total, info, fidelity, contiguity):
        assert isinstance(t, torch.Tensor)
        assert t.dim() == 0  # scalaire

    # Vérifie que le total correspond bien à la combinaison Lagrangienne attendue
    expected = info + 2.0 * fidelity + 0.5 * contiguity
    assert torch.allclose(total, expected, atol=1e-5)


def test_mask_gradients_flow_to_logits():
    """Sanity check indispensable pour l'optimisation Adam : le gradient doit atteindre mask_logits."""
    B, T, D = 1, 5, 4
    x = torch.randn(B, T, D)
    baseline = torch.randn(T, D)
    mask_module = TokenGumbelSoftmaxMask(seq_len=T)

    z, mask = mask_module(x, baseline, training=True)
    loss = z.sum() + mask.sum()
    loss.backward()

    assert mask_module.mask_logits.grad is not None
    assert not torch.all(mask_module.mask_logits.grad == 0)


def test_special_tokens_are_always_kept():
    """
    [CLS]/[SEP] (special_tokens_mask=1) doivent toujours avoir mask=1, quels que
    soient les logits appris -- ce sont des tokens structurels, pas du contenu
    candidat à la sélection.
    """
    B, T, D = 1, 5, 4
    x = torch.randn(B, T, D)
    baseline = torch.randn(T, D)
    # positions 0 et 4 = tokens spéciaux (ex: [CLS] ... [SEP])
    special_tokens_mask = torch.tensor([[1, 0, 0, 0, 1]])

    # init_logits très négatif -> sans le fix, mask serait ~0 partout
    mask_module = TokenGumbelSoftmaxMask(seq_len=T, init_logits=-10.0)
    _, mask = mask_module(
        x, baseline, training=False, special_tokens_mask=special_tokens_mask
    )

    assert torch.allclose(mask[0, [0, 4]], torch.ones(2), atol=1e-3)
    # les tokens de contenu restent, eux, proches de 0 comme attendu
    assert torch.all(mask[0, 1:4] < 0.1)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))