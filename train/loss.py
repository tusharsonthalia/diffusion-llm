import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from model.mdlm import MDLM
from train.noise import LogLinearNoise


def sample_t(
    batch_size: int,
    sampling_eps: float = 1e-3,
    antithetic: bool = True,
    device: torch.device | None = None,
) -> Float[Tensor, " batch"]:
    """Sample one diffusion time per sequence, t in [sampling_eps, 1).

    Antithetic (stratified) sampling splits [0, 1) into batch_size equal strata
    and draws one u inside each:

        u_i = u~_i / B + i / B,   u~_i ~ U[0, 1)    -> u_i in [i/B, (i+1)/B)

    so every batch spans light to heavy masking. This cuts the variance of the
    batch loss (whose weight ~1/t blows up near 0) while staying unbiased. The
    reference's "% 1" is a no-op, since u~/B + (B-1)/B < 1 always.

    t is then mapped onto [sampling_eps, 1): near t = 0 almost nothing is masked
    and the 1/t loss weight explodes.

    Args:
        batch_size (int): Number of sequences B.
        sampling_eps (float): Smallest t that can be sampled.
        antithetic (bool): Stratify across the batch; False = i.i.d. uniform.
        device (torch.device | None): Device to create t on.

    Returns:
        Tensor of shape (batch_size,).
    """
    u = torch.rand(batch_size, device=device)
    if antithetic:
        offsets = torch.arange(batch_size, device=device) / batch_size
        u = u / batch_size + offsets

    return (1 - sampling_eps) * u + sampling_eps


def q_xt(
    x0: Int[Tensor, "batch seq"],
    move_chance: Float[Tensor, " batch"],
    mask_index: int,
) -> Int[Tensor, "batch seq"]:
    """Forward (masking) process: replace each token of x0 with [MASK]
    independently, with its sequence's probability move_chance.

    Args:
        x0: Shape (batch, seq), clean token ids.
        move_chance: Shape (batch,), per-sequence masking probability
            1 - exp(-sigma(t)) -- already a probability, computed by the caller.
        mask_index (int): Token id of [MASK].

    Returns:
        x_t of shape (batch, seq): x0 with the masked positions set to mask_index.
    """
    token_probs = torch.rand(x0.shape, device=x0.device)
    mask_positions = token_probs < move_chance.reshape(-1, 1)

    return torch.where(mask_positions, mask_index, x0)


def mdlm_loss(
    model: MDLM,
    noise: LogLinearNoise,
    x0: Int[Tensor, "batch seq"],
    attention_mask: Bool[Tensor, "batch seq"] | None = None,
    sampling_eps: float = 1e-3,
) -> Float[Tensor, ""]:
    """Continuous-time MDLM loss (negative ELBO with the SUBS parameterization):

        L = E_t [ dsigma(t) / (exp(sigma(t)) - 1) * sum_l -log x_theta^l(x_t)[x0^l] ]

    Args:
        model (MDLM): Returns SUBS log-probs from forward(xt, sigma).
        noise (LogLinearNoise): The noise schedule.
        x0: Shape (batch, seq), clean token ids (no [MASK]).
        attention_mask: Shape (batch, seq), True = count this position. None =
            every position counts.
        sampling_eps (float): Smallest t that can be sampled.

    Returns:
        Scalar: summed weighted NLL divided by the number of counted tokens.
    """
    t = sample_t(x0.shape[0], sampling_eps, device=x0.device)
    sigma, dsigma = noise(t)

    move_chance = -torch.expm1(-sigma)
    xt = q_xt(x0, move_chance, model.mask_index)

    # SUBS log-probs (batch, seq, vocab) -> log-prob of the clean token (batch, seq)
    log_probs = model(xt, sigma)
    log_p_x0 = log_probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)

    # per-sequence weight, (batch,) -> (batch, 1) to broadcast over positions
    weight = dsigma / torch.expm1(sigma)
    nll = -log_p_x0 * weight.unsqueeze(-1)

    if attention_mask is None:
        return nll.mean()

    # excluded positions drop out of both the numerator and the token count
    token_mask = attention_mask.to(nll.dtype)
    return (nll * token_mask).sum() / token_mask.sum()
