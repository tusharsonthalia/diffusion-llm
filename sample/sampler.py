import torch
from jaxtyping import Float, Int
from torch import Tensor

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
