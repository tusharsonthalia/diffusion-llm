import torch
from jaxtyping import Float
from torch import Tensor


class LogLinearNoise:
    """Log-linear noise schedule for absorbing-state (masked) diffusion:

        sigma(t)   = -log(1 - (1 - eps) * t)
        dsigma/dt  = (1 - eps) / (1 - (1 - eps) * t)

    sigma(t) is the total masking rate accumulated up to time t, so a token is
    still unmasked with probability alpha_t = exp(-sigma(t)) = 1 - (1 - eps) * t.
    The masking probability 1 - alpha_t = (1 - eps) * t is therefore LINEAR in t.

    Used by the forward process (masking probability), as the model's noise-level
    input, by the loss weight (needs both sigma and dsigma/dt), and by the sampler.

    Args:
        eps (float): Keeps everything finite at t = 1: without it sigma(1) =
            -log(0) = inf. With eps = 1e-3, sigma(1) ~ 6.91 and at most 99.9% of
            tokens are masked.
    """
    def __init__(
        self,
        eps: float = 1e-3,
    ):
        self.eps = eps

    def __call__(
        self,
        t: Float[Tensor, " batch"],
    ) -> tuple[Float[Tensor, " batch"], Float[Tensor, " batch"]]:
        """Evaluate the schedule at one time per sequence.

        Args:
            t: Shape (batch,), diffusion times in [0, 1].

        Returns:
            (sigma, dsigma_dt), each of shape (batch,). sigma(0) = 0 and both are
            increasing in t.
        """
        t_coeff = (1 - self.eps) * t

        # -log(1 - x) via log1p: accurate for small x, where 1 - x rounds to 1
        sigma_t = -torch.log1p(-t_coeff)

        # chain rule on -log(u), u = 1 - (1 - eps) t: (-1/u) * (-(1 - eps))
        dsigma_dt = (1 - self.eps) / (1 - t_coeff)

        return sigma_t, dsigma_dt
