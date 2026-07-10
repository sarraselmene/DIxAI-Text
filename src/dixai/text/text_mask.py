
import torch
import torch.nn as nn
from typing import Optional, Tuple


class TokenGumbelSoftmaxMask(nn.Module):
    """
    Gumbel-Softmax mask for NLP token selection.

    z_t = m_t * x_t + (1 - m_t) * b_t

    where:
        x_t : token embedding
        b_t : baseline embedding
        m_t : learned mask value in [0,1]
    """

    def __init__(
        self,
        seq_len: int,
        temperature: float = 2.0 / 3.0,
        init_logits: float = -2.0,
        hard: bool = False,
        verbose: bool = False,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.temperature = temperature
        self.hard = hard
        self.verbose = verbose

        # One learnable logit per token position
        self.mask_logits = nn.Parameter(
            torch.full((seq_len,), init_logits)
        )

    ###########################################################################
    # GUMBEL SAMPLING
    ###########################################################################

    def _sample_mask(self, training: bool) -> torch.Tensor:

        logits = self.mask_logits

        if training:

            uniform = torch.rand_like(logits).clamp(
                1e-6,
                1 - 1e-6
            )

            gumbel_noise = -torch.log(
                -torch.log(uniform)
            )

            y_soft = torch.sigmoid(
                (logits + gumbel_noise)
                / self.temperature
            )

        else:

            y_soft = torch.sigmoid(logits)

        if self.hard:

            y_hard = (y_soft > 0.5).float()

            mask = (
                y_hard.detach()
                - y_soft.detach()
                + y_soft
            )

        else:

            mask = y_soft

        return mask

    ###########################################################################
    # FORWARD
    ###########################################################################

    def forward(
        self,
        token_embeddings: torch.Tensor,
        baseline: torch.Tensor,
        training: bool = True,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        assert token_embeddings.dim() == 3, \
            "Expected shape (B,T,D)"

        B, T, D = token_embeddings.shape

        assert T == self.seq_len, \
            f"Configured seq_len={self.seq_len} but got T={T}"

        if self.verbose:
            print("\n" + "=" * 80)
            print("FORWARD PASS")
            print("=" * 80)
            print(f"Batch Size     : {B}")
            print(f"Sequence Length: {T}")
            print(f"Embedding Dim  : {D}")

        ############################################################
        # Sample mask
        ############################################################

        mask = self._sample_mask(training)

        if self.verbose:
            print("\nRaw sampled mask:")
            print(mask.detach().cpu())

        ############################################################
        # Expand to batch
        ############################################################

        mask = mask.unsqueeze(0).expand(B, T)

        ############################################################
        # Apply padding mask
        ############################################################

        if attention_mask is not None:

            mask = mask * attention_mask.to(mask.dtype)

            if self.verbose:
                print("\nPadding mask applied.")

        ############################################################
        # Expand baseline if needed
        ############################################################

        if baseline.dim() == 2:

            baseline = baseline.unsqueeze(0).expand(
                B,
                T,
                D
            )

        ############################################################
        # Blend embeddings
        ############################################################

        mask_expanded = mask.unsqueeze(-1)

        z = (
            mask_expanded * token_embeddings
            + (1 - mask_expanded) * baseline
        )

        ############################################################
        # Statistics
        ############################################################

        if self.verbose:

            probs = self.get_mask_probs().detach()

            print("\nMask statistics")

            print(
                f"Mean Prob : {probs.mean():.4f}"
            )

            print(
                f"Max Prob  : {probs.max():.4f}"
            )

            print(
                f"Min Prob  : {probs.min():.4f}"
            )

            if attention_mask is not None:

                active_tokens = (
                    attention_mask.sum()
                    .item()
                )

                selected_tokens = (
                    mask.sum()
                    .item()
                )

                print(
                    f"Active Tokens : {active_tokens}"
                )

                print(
                    f"Selected Tokens : {selected_tokens:.2f}"
                )

                print(
                    f"Sparsity Ratio : "
                    f"{selected_tokens / active_tokens:.4f}"
                )

            print("=" * 80)

        return z, mask

    ###########################################################################
    # UTILITIES
    ###########################################################################

    def get_mask_probs(self):

        return torch.sigmoid(
            self.mask_logits
        )

    def selected_token_indices(
        self,
        threshold: float = 0.5
    ):

        probs = self.get_mask_probs()

        return torch.nonzero(
            probs > threshold,
            as_tuple=False
        ).squeeze(-1)

    ###########################################################################
    # DEBUGGING
    ###########################################################################

    def debug_tokens(
        self,
        tokenizer,
        input_ids,
        mask,
    ):
        """
        Pretty print token importance.

        Parameters
        ----------
        tokenizer : HuggingFace tokenizer
        input_ids : (B,T)
        mask : (B,T)
        """

        print("\n" + "=" * 80)
        print("TOKEN IMPORTANCE")
        print("=" * 80)

        tokens = tokenizer.convert_ids_to_tokens(
            input_ids[0].detach()
            .cpu()
            .tolist()
        )

        values = (
            mask[0]
            .detach()
            .cpu()
            .tolist()
        )

        for token, score in zip(
            tokens,
            values
        ):

            if score > 0.8:
                symbol = "🟢"

            elif score > 0.5:
                symbol = "🟡"

            else:
                symbol = "🔴"

            print(
                f"{symbol} "
                f"{token:<20} "
                f"{score:.4f}"
            )

        print("=" * 80)

    ###########################################################################
    # TRAINING MONITOR
    ###########################################################################

    def print_logits(self):

        print("\nCurrent logits:")
        print(
            self.mask_logits
            .detach()
            .cpu()
        )

        print("\nCurrent probabilities:")
        print(
            self.get_mask_probs()
            .detach()
            .cpu()
        )

    ###########################################################################
    # FULL REPORT
    ###########################################################################

    def report(
        self,
        tokenizer,
        input_ids,
        attention_mask=None,
    ):

        probs = (
            self.get_mask_probs()
            .detach()
            .cpu()
        )

        tokens = tokenizer.convert_ids_to_tokens(
            input_ids[0].cpu().tolist()
        )

        print("\n" + "=" * 80)
        print("MASK REPORT")
        print("=" * 80)

        for i, (tok, p) in enumerate(
            zip(tokens, probs)
        ):

            if attention_mask is not None:

                if attention_mask[0, i] == 0:
                    continue

            print(
                f"{i:02d} | "
                f"{tok:<20} | "
                f"{p:.4f}"
            )

        print("=" * 80)

