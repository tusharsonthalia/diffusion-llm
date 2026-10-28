import torch
from jaxtyping import Float, Int
from torch import Tensor

from model.bd3lm import BD3LM
from model.kv_cache import KVCache

EPS = 1e-10


def sample_categorical(
    probs: Float[Tensor, "... vocab"],
) -> Int[Tensor, "..."]:
    """Draw one sample per row from categorical probabilities.

    Gumbel-max in disguise: with E = -log(u) ~ Exp(1),

        argmax(p / E) = argmax(log p - log E), and -log E ~ Gumbel(0, 1),

    so this is argmax(log p + Gumbel noise), an exact sample from p / sum(p).

    The noise is drawn in float64: in float32, low-probability tokens are undersampled
    because both p and the noise lose resolution near zero.

    Args:
        probs: Shape (..., vocab), non-negative, any scale.

    Returns:
        Sampled indices of shape probs.shape[:-1].
    """
    u = torch.rand_like(probs, dtype=torch.float64)
    gumbel_norm = EPS - torch.log(u + EPS)

    return torch.argmax(probs / gumbel_norm, dim=-1)


@torch.no_grad()
def sample_bd3lm(
    model: BD3LM,
    batch_size: int,
    n_blocks: int,
    steps: int,
    kv_cache: bool = False,
) -> Int[Tensor, "batch seq"]:
    """Generate n_blocks blocks left to right with BD3-LM's block-by-block sampler.

    Each block starts all-[MASK] and is denoised for `steps` ancestral steps from
    t = 1 to t = 0, conditioned on the finished blocks before it. With kv_cache=True,
    finished blocks are committed to a KV cache and each step runs only the current
    block; otherwise the whole accumulated sequence is re-encoded every step.

    RNG: the only random draw is sample_categorical's, once per denoising step.

    Args:
        model (BD3LM): The block diffusion model (sigma is always 0).
        batch_size (int): Number of sequences to generate.
        n_blocks (int): Number of blocks; output length is n_blocks * block_size.
        steps (int): Denoising steps T per block.
        kv_cache (bool): Use the KV cache (fast) instead of re-encoding (reference).

    Returns:
        Token ids of shape (batch_size, n_blocks * block_size), no [MASK] tokens.
    """
    device = model.attn_mask.device
    bs, mask_index = model.block_size, model.mask_index

    # x holds only FINISHED blocks; the cache (if used) holds their K/V
    x = torch.empty(batch_size, 0, device=device, dtype=torch.long)
    cache = KVCache.empty(model.n_layers) if kv_cache else None

    for _ in range(n_blocks):
        block = torch.full((batch_size, bs), mask_index, device=device, dtype=torch.long)

        for i in range(steps):
            # current time t and next time s; s = 0 at the last step
            t, s = 1 - i / steps, 1 - (i + 1) / steps

            # p_theta(x0 | x_t) for the block, conditioned on all finished blocks
            if cache is not None:
                log_probs = model.cached_block_logprobs(block, cache)
            else:
                log_probs = model.block_logprobs(torch.cat([x, block], dim=-1))

            # ancestral update (move_chance = t): a masked token is unmasked by time s; 
            # otherwise it stays masked (weight s / t).Float64 keeps
            # low-probability tokens from being undersampled.
            q = torch.exp(log_probs).double() * (1 - s / t)
            q[..., mask_index] = s / t

            new_tokens = sample_categorical(q)

            block = torch.where(block != mask_index, block, new_tokens)

        # s = 0 at the last step, so the block is fully unmasked here
        if cache is not None:
            model.commit_block(block, cache)
        x = torch.cat([x, block], dim=-1)

    return x
