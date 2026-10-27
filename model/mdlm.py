import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from model.modules import DiT, log_softmax


class MDLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        d_cond: int,
        time_conditioning: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.backbone = DiT(
            vocab_size, max_seq_len, d_model, n_heads, 
            n_layers, d_ff, d_cond, device, dtype,
        )
        self.mask_index = vocab_size - 1
        self.time_conditioning = time_conditioning

    def forward(
        self, 
        xt: Int[Tensor, "batch seq"],
        sigma: Float[Tensor, " batch"],
    ):
        if not self.time_conditioning:
            sigma = torch.zeros_like(sigma)

        logits = self.backbone(xt, sigma)

        return subs_parameterization(logits, xt, self.mask_index)


def subs_parameterization(
    logits: Float[Tensor, "batch seq vocab"],
    xt: Int[Tensor, "batch seq"], 
    mask_index: int,
) -> Float[Tensor, "batch seq vocab"]:
    # large finite stand-in for -inf
    penalty = -1e6

    # rule 1: zero mass on [MASK], then normalize.
    bias = torch.zeros(logits.shape[-1], device=logits.device, dtype=logits.dtype)
    bias[mask_index] = penalty
    log_probs = log_softmax(logits + bias, dim=-1)

    # rule 2: one-hot rows (0 at x_t's token, penalty elsewhere).
    carry = torch.full_like(log_probs, penalty).scatter(-1, xt.unsqueeze(-1), 0.0)

    # (batch, seq, 1): broadcasts the per-position choice across the vocab axis
    unmasked = (xt != mask_index).unsqueeze(-1)

    return torch.where(unmasked, carry, log_probs)