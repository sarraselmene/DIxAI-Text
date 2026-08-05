"""
debug_gumbel.py

Diagnostic du TokenGumbelSoftmaxMask.

Ce script ne teste PAS BERT.
Il vérifie uniquement le comportement du Gumbel-Softmax.

À lancer avec :

python tests/debug_gumbel.py
"""

import torch
from dixai.text.masks_text import TokenGumbelSoftmaxMask


torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

B = 2
T = 12
D = 768

mask_module = TokenGumbelSoftmaxMask(
    seq_len=T,
    temperature=0.67,
    init_logits=-2.0,
    hard=True,
).to(device)

optimizer = torch.optim.Adam(mask_module.parameters(), lr=0.2)

token_embeddings = torch.randn(B, T, D, device=device)
baseline = torch.zeros_like(token_embeddings)

attention_mask = torch.ones(B, T, device=device)

special_tokens_mask = torch.zeros(B, T, device=device)
special_tokens_mask[:, 0] = 1
special_tokens_mask[:, -1] = 1


print("=" * 80)
print("Initial probabilities")
print(mask_module.get_mask_probs())
print("=" * 80)

for step in range(201):

    optimizer.zero_grad()

    z, mask = mask_module(
        token_embeddings,
        baseline,
        training=True,
        attention_mask=attention_mask,
        special_tokens_mask=special_tokens_mask,
    )

    ##############################################################
    # Dummy objective
    #
    # On pousse artificiellement quelques tokens vers 1
    # et les autres vers 0.
    ##############################################################

    target = torch.zeros_like(mask)
    target[:, 3] = 1
    target[:, 7] = 1

    loss = ((mask - target) ** 2).mean()

    loss.backward()

    grad_norm = mask_module.mask_logits.grad.norm().item()

    optimizer.step()

    if step % 20 == 0:

        probs = mask_module.get_mask_probs().detach()

        near_zero = (probs < 0.05).sum().item()
        near_one = (probs > 0.95).sum().item()
        uncertain = ((probs >= 0.05) & (probs <= 0.95)).sum().item()

        print(f"\nStep {step}")
        print("-" * 60)

        print(f"Loss           : {loss.item():.5f}")
        print(f"Gradient norm  : {grad_norm:.6f}")

        print(f"Temperature    : {mask_module.temperature:.3f}")

        print(f"Mean prob      : {probs.mean():.4f}")
        print(f"Std            : {probs.std():.4f}")

        print(f"<0.05          : {near_zero}")
        print(f">0.95          : {near_one}")
        print(f"Intermediate   : {uncertain}")

        print("\nProbabilities")

        for i, p in enumerate(probs):

            print(f"Token {i:2d} : {p.item():.3f}")

print("\n")
print("=" * 80)
print("Final probabilities")
print(mask_module.get_mask_probs())
print("=" * 80)