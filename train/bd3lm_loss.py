import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from model.bd3lm import BD3LM


def sample_t_blocks(
    batch_size: int,
    seq_len: int,
    block_size: int,
    sampling_eps_min: float = 1e-3,
    sampling_eps_max: float = 1.0,
    antithetic: bool = True,
    device: torch.device | None = None,
) -> Float[Tensor, "batch seq"]:
    """Sample one diffusion time per (sequence, block), repeated over the block's tokens.

    Args:
        batch_size (int): Number of sequences B.
        seq_len (int): n, tokens per sequence. Must be a multiple of block_size.
        block_size (int): Tokens per block.
        sampling_eps_min (float): beta, the smallest t.
        sampling_eps_max (float): omega, the largest t.
        antithetic (bool): Stratify over (batch, block); False = i.i.d. uniform.
        device (torch.device | None): Device to create t on.

    Returns:
        Tensor of shape (batch_size, seq_len), constant within each block.
    """
    assert seq_len % block_size == 0, f"seq_len {seq_len} must be a multiple of block_size {block_size}"

    num_blocks = seq_len // block_size
    num_draws = batch_size * num_blocks

    u = torch.rand(batch_size, num_blocks, device=device)
    if antithetic:
        offsets = torch.arange(num_draws, device=device).reshape(batch_size, num_blocks) / num_draws
        u = u / num_draws + offsets

    t = (sampling_eps_max - sampling_eps_min) * u + sampling_eps_min

    # (batch, num_blocks) -> (batch, seq): every token of a block shares its t
    return t.repeat_interleave(block_size, dim=-1)


def bd3lm_loss(
    model: BD3LM,
    x0: Int[Tensor, "batch seq"],
    sampling_eps_min: float = 1e-3,
    sampling_eps_max: float = 1.0,
    attention_mask: Bool[Tensor, "batch seq"] | None = None,
) -> Float[Tensor, ""]:
    """BD3-LM loss: the MDLM objective applied per block, with per-token times:

        L = (1 / #tokens) * sum_{b,l} (1 / t_{b,l}) * -log x_theta^l([x_t; x_0])[x0^l]

    where t_{b,l} is the time of the block containing position l. BD3-LM's log-linear
    schedule uses move_chance = t and weight 1/t directly (no (1 - eps) factor, no
    sigma). Only masked positions contribute: SUBS makes log x_theta[x0] = 0 elsewhere.

    Args:
        model (BD3LM): Returns SUBS log-probs for the x_t half from forward(xt, x0).
        x0: Shape (batch, seq), clean token ids (no [MASK]).
        sampling_eps_min (float): beta, lower end of the clipped schedule.
        sampling_eps_max (float): omega, upper end of the clipped schedule.
        attention_mask: Shape (batch, seq), True = count this position. None =
            every position counts.

    Returns:
        Scalar: summed weighted NLL divided by the number of counted tokens.
    """
    batch_size = x0.shape[0]
    seq_len = x0.shape[1]
    block_size = model.block_size

    t = sample_t_blocks(
        batch_size, seq_len, block_size, sampling_eps_min, 
        sampling_eps_max, device=x0.device
    )

    # forward process: mask each token with its own block's probability t
    masked = torch.rand(x0.shape, device=x0.device) < t
    xt = torch.where(masked, model.mask_index, x0)

    # SUBS log-probs (batch, seq, vocab) -> log-prob of the clean token (batch, seq)
    log_probs = model(xt, x0)
    log_p_x0 = log_probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)

    # t is already per token, (batch, seq)
    inv_t = 1 / t
    nll = -log_p_x0 * inv_t

    if attention_mask is None:
        return nll.mean()

    # excluded positions drop out of both the numerator and the token count
    token_mask = attention_mask.to(nll.dtype)
    return (nll * token_mask).sum() / token_mask.sum()
