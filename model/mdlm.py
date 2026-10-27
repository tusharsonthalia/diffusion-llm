import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from model.modules import DiT
from model.subs import subs_parameterization


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
    ) -> Float[Tensor, "batch seq vocab"]:
        if not self.time_conditioning:
            sigma = torch.zeros_like(sigma)

        logits = self.backbone(xt, sigma)

        return subs_parameterization(logits, xt, self.mask_index)

