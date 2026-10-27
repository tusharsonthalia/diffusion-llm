import torch
from jaxtyping import Float, Int
from torch import Tensor

from model.modules import log_softmax

# large finite stand-in for -inf
NEG_INF = -1e6


def subs_parameterization(
    logits: Float[Tensor, "batch seq vocab"],
    xt: Int[Tensor, "batch seq"],
    mask_index: int,
) -> Float[Tensor, "batch seq vocab"]:
    """SUBS parameterization: raw logits -> log x_theta, the denoiser's per-position
    distribution over the clean token. Shared by MDLM, BD3-LM and the samplers.

    1. zero masking probability -- clean data never contains [MASK], so its logit is
        pushed to NEG_INF before normalizing;
    2. carry-over unmasking -- an unmasked x_t token is exactly x0, so those rows are
        replaced by a one-hot (0 at x_t, NEG_INF elsewhere)

    Args:
        logits: Shape (batch, seq, vocab), raw backbone output.
        xt: Shape (batch, seq), noisy token ids.
        mask_index (int): Token id of [MASK].

    Returns:
        Log-probabilities of shape (batch, seq, vocab).
    """
    # rule 1: zero mass on [MASK], then normalize
    bias = torch.zeros(logits.shape[-1], device=logits.device, dtype=logits.dtype)
    bias[mask_index] = NEG_INF
    log_probs = log_softmax(logits + bias, dim=-1)

    # rule 2: one-hot carry-over rows
    carry = torch.full_like(log_probs, NEG_INF).scatter(-1, xt.unsqueeze(-1), 0.0)

    # (batch, seq, 1): broadcasts the per-position choice across the vocab axis
    unmasked = (xt != mask_index).unsqueeze(-1)

    return torch.where(unmasked, carry, log_probs)
