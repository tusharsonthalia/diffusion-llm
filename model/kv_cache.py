from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor


@dataclass
class KVCache:
    """Per-layer key/value cache for committed (clean) blocks in BD3-LM sampling.

    Each layer's slot holds the post-RoPE keys and values of every committed token,
    laid out as attention produces them: (batch, n_heads, L_cache, head_dim).

    Start a new sequence with KVCache.empty(n_layers). Slots are None until the first
    block is committed.

    Attributes:
        k, v (list[Tensor | None]): One entry per layer, (batch, n_heads, seq, head_dim).
    """
    k: list[Tensor | None]
    v: list[Tensor | None]

    @classmethod
    def empty(cls, n_layers: int) -> "KVCache":
        """A cache with no committed tokens, one slot per layer."""
        return cls(k=[None] * n_layers, v=[None] * n_layers)

    @property
    def length(self) -> int:
        """Number of committed tokens."""
        k0 = self.k[0]
        return 0 if k0 is None else k0.shape[-2]

    def get(
        self,
        layer_idx: int,
    ) -> tuple[Float[Tensor, "batch heads cache d_k"] | None, Float[Tensor, "batch heads cache d_k"] | None]:
        """The cached (k, v) for one layer, or (None, None) if nothing is committed."""
        return self.k[layer_idx], self.v[layer_idx]

    def append(
        self,
        layer_idx: int,
        k: Float[Tensor, "batch heads new d_k"],
        v: Float[Tensor, "batch heads new d_k"],
    ) -> None:
        """Append one layer's post-RoPE keys/values for newly committed tokens."""
        stored_k, stored_v = self.get(layer_idx)

        if stored_k is None or stored_v is None:
            self.k[layer_idx], self.v[layer_idx] = k, v
        else:
            self.k[layer_idx] = torch.cat([stored_k, k], dim=-2)
            self.v[layer_idx] = torch.cat([stored_v, v], dim=-2)
