"""
Tests unitaires pour DIxAI-Text — Verrou 1 (masquage) + Verrou 2 (contiguité).

Exécutables HORS-LIGNE (sans téléchargement de BERT).
Utilise un modèle linéaire minimal comme proxy.

Lancement:
    python tests/test_verrou1.py
    ou
    pytest tests/test_verrou1.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
import unittest

from dixai.text.masks_text import TokenGumbelSoftmaxMask
from dixai.text.objective_text import DecisionInformationLossText


class TestTokenGumbelSoftmaxMask(unittest.TestCase):

    def setUp(self):
        """Setup commun : séquence de longueur T=8, embedding D=16."""
        self.T = 8
        self.D = 16
        self.B = 1
        self.mask_module = TokenGumbelSoftmaxMask(seq_len=self.T, temperature=0.5)
        self.x = torch.randn(self.B, self.T, self.D)
        self.baseline = torch.zeros(self.B, self.T, self.D)

    def test_1_output_shapes(self):
        """Test 1 : les sorties z et mask ont les bonnes formes."""
        z, mask = self.mask_module(self.x, self.baseline, training=False)
        self.assertEqual(z.shape, (self.B, self.T, self.D),
                         "z doit avoir shape (B, T, D)")
        self.assertEqual(mask.shape, (self.B, self.T),
                         "mask doit avoir shape (B, T)")

    def test_2_mask_bounds(self):
        """Test 2 : mask dans [0,1] après clamp défensif."""
        z, mask = self.mask_module(self.x, self.baseline, training=False)
        self.assertTrue(mask.min().item() >= 0.0 - 1e-6,
                        "mask doit être >= 0")
        self.assertTrue(mask.max().item() <= 1.0 + 1e-6,
                        "mask doit être <= 1")

    def test_3_padding_zeroed(self):
        """Test 3 : mask=0 sur les positions padding."""
        attention_mask = torch.ones(self.B, self.T)
        attention_mask[0, 5:] = 0  # positions 5,6,7 = padding

        z, mask = self.mask_module(
            self.x, self.baseline,
            training=False,
            attention_mask=attention_mask,
        )
        pad_mask_values = mask[0, 5:].tolist()
        for v in pad_mask_values:
            self.assertAlmostEqual(v, 0.0, places=5,
                                   msg=f"Padding doit avoir mask=0, got {v}")

    def test_4_special_tokens_forced_to_one(self):
        """Test 4 : tokens spéciaux ([CLS]/[SEP]) forcés à mask=1."""
        attention_mask = torch.ones(self.B, self.T)
        special_tokens_mask = torch.zeros(self.B, self.T)
        special_tokens_mask[0, 0] = 1  # [CLS]
        special_tokens_mask[0, 7] = 1  # [SEP]

        # Forcer logits très négatifs pour que sans forçage, mask -> 0
        with torch.no_grad():
            self.mask_module.mask_logits.fill_(-10.0)

        z, mask = self.mask_module(
            self.x, self.baseline,
            training=False,
            attention_mask=attention_mask,
            special_tokens_mask=special_tokens_mask,
        )
        self.assertAlmostEqual(mask[0, 0].item(), 1.0, places=4,
                               msg="[CLS] doit avoir mask=1")
        self.assertAlmostEqual(mask[0, 7].item(), 1.0, places=4,
                               msg="[SEP] doit avoir mask=1")

    def test_5_reconstruction_mask_one(self):
        """Test 5 : si mask=1 partout, z doit être proche de x."""
        with torch.no_grad():
            self.mask_module.mask_logits.fill_(10.0)  # sigmoid(10) ~ 1

        z, mask = self.mask_module(self.x, self.baseline, training=False)
        self.assertTrue(
            torch.allclose(z, self.x, atol=1e-3),
            "Si mask=1, z doit être proche de x"
        )

    def test_6_reconstruction_mask_zero(self):
        """Test 6 : si mask=0 partout, z doit être proche de baseline."""
        with torch.no_grad():
            self.mask_module.mask_logits.fill_(-10.0)  # sigmoid(-10) ~ 0

        z, mask = self.mask_module(self.x, self.baseline, training=False)
        self.assertTrue(
            torch.allclose(z, self.baseline, atol=1e-3),
            "Si mask=0, z doit être proche de baseline"
        )

    def test_7_gradients_flow(self):
        """Test 7 : les gradients atteignent mask_logits (Adam peut optimiser)."""
        # Modèle proxy minimal
        linear = nn.Linear(self.D, 2)
        optimizer = torch.optim.Adam(self.mask_module.parameters(), lr=0.01)

        optimizer.zero_grad()
        z, mask = self.mask_module(self.x, self.baseline, training=True)
        # Loss simple : moyenne des logits du modèle proxy
        out = linear(z.mean(dim=1))
        loss = out.sum()
        loss.backward()

        grad = self.mask_module.mask_logits.grad
        self.assertIsNotNone(grad, "mask_logits doit avoir un gradient")
        self.assertFalse(
            torch.all(grad == 0),
            "Le gradient de mask_logits ne doit pas être nul partout"
        )


class TestDecisionInformationLossText(unittest.TestCase):

    def setUp(self):
        self.T = 8
        self.B = 1
        self.C = 2  # 2 classes (SST-2)
        self.loss_fn = DecisionInformationLossText(
            lambda_fidelity=10.0,
            lambda_contiguity=2.0,
        )

    def test_8_loss_components(self):
        """Test 8 : les 4 composantes de la loss sont bien retournées."""
        mask = torch.rand(self.B, self.T)
        attention_mask = torch.ones(self.B, self.T)
        y_orig = torch.log_softmax(torch.randn(self.B, self.C), dim=-1)
        y_masked = torch.log_softmax(torch.randn(self.B, self.C), dim=-1)

        loss, info, fid, contig = self.loss_fn(
            mask, y_orig, y_masked, attention_mask=attention_mask
        )

        self.assertIsInstance(loss.item(), float, "loss doit être un scalaire")
        self.assertIsInstance(info.item(), float, "info_loss doit être un scalaire")
        self.assertIsInstance(fid.item(), float, "fidelity_loss doit être un scalaire")
        self.assertIsInstance(contig.item(), float, "contiguity_loss doit être un scalaire")

    def test_9_contiguity_span_vs_sparse(self):
        """
        Test 9 : TV_seq d'un span contigu < TV_seq d'une sélection éparse
        à sparsité égale.
        """
        T = 10
        attention_mask = torch.ones(1, T)

        # Span contigu : tokens 3,4,5 sélectionnés
        mask_span = torch.zeros(1, T)
        mask_span[0, 3:6] = 1.0

        # Sélection éparse : tokens 1,5,9 sélectionnés (même nb)
        mask_sparse = torch.zeros(1, T)
        mask_sparse[0, 1] = 1.0
        mask_sparse[0, 5] = 1.0
        mask_sparse[0, 9] = 1.0

        tv_span = self.loss_fn._sequential_contiguity(mask_span, attention_mask)
        tv_sparse = self.loss_fn._sequential_contiguity(mask_sparse, attention_mask)

        self.assertLess(
            tv_span.item(), tv_sparse.item(),
            f"TV_seq(span)={tv_span.item():.4f} doit être < "
            f"TV_seq(sparse)={tv_sparse.item():.4f}"
        )

    def test_10_no_contiguity_when_lambda_zero(self):
        """Test 10 : si lambda_contiguity=0, contiguity_loss=0."""
        loss_fn_no_contig = DecisionInformationLossText(
            lambda_fidelity=10.0,
            lambda_contiguity=0.0,
        )
        mask = torch.rand(self.B, self.T)
        attention_mask = torch.ones(self.B, self.T)
        y = torch.log_softmax(torch.randn(self.B, self.C), dim=-1)

        _, _, _, contig = loss_fn_no_contig(mask, y, y, attention_mask=attention_mask)
        self.assertAlmostEqual(contig.item(), 0.0, places=6,
                               msg="contiguity_loss doit être 0 si lambda=0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
