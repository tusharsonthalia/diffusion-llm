import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn

from model.modules import DiT
from model.subs import subs_parameterization


def block_diff_mask(
    seq_len: int,
    block_size: int,
    device: torch.device | None = None,
) -> Bool[Tensor, "two_seq_len two_seq_len"]:
    """BD3-LM training attention mask over x_input = cat(x_t, x_0), length 2n.

    Indices [0, n) are the noisy copy x_t, [n, 2n) the clean copy x_0:

        noisy q, block b -> noisy keys in block b          (block diagonal)
        noisy q, block b -> clean keys in blocks < b       (offset block-causal)
        clean q, block b -> clean keys in blocks <= b      (block-causal)

    block_size = seq_len reduces the noisy half to MDLM's full attention;
    block_size = 1 makes the clean half token-causal (an AR model).

    Args:
        seq_len (int): n, the length of one copy of the sequence.
        block_size (int): Tokens per block.
        device (torch.device | None): Device to build the mask on.

    Returns:
        Boolean tensor of shape (2n, 2n).
    """
    # column of query indices vs row of key indices
    queries = torch.arange(2 * seq_len, device=device).reshape(-1, 1)
    keys = torch.arange(2 * seq_len, device=device).reshape(1, -1)

    # segmenting the QK^T mask into four segments, 
    #           [noisy, noisy] | [noisy, clean]
    #           [clean, noisy] | [clean, clean]
    q_clean, k_clean = queries >= seq_len, keys >= seq_len

    # block id WITHIN each half, so noisy and clean ids are directly comparable
    q_block = (queries % seq_len) // block_size
    k_block = (keys % seq_len) // block_size

    # getting the mask on 3 of the 4 segnments
    block_diagonal = ~q_clean & ~k_clean & (q_block == k_block)
    offset_block_causal = ~q_clean & k_clean & (k_block < q_block)
    block_causal = q_clean & k_clean & (k_block <= q_block)

    return block_diagonal | offset_block_causal | block_causal

class BD3LM(nn.Module):
    """Block diffusion LM: autoregressive over blocks, masked diffusion within each.

    Training runs the DiT once on the 2L input cat(x_t, x_0), with positions shared
    between the two copies and the block diffusion mask routing each noisy block to
    itself plus the clean blocks before it. Only the x_t half is scored (SUBS).

    sigma is always 0 (time_conditioning off, as in the reference): the model infers
    each block's noise level from its [MASK] tokens.

    Args:
        vocab_size (int): Vocabulary size, including [MASK] (the last index).
        max_seq_len (int): n, the training sequence length. Must be a multiple of
            block_size.
        d_model, n_heads, n_layers, d_ff, d_cond: DiT configuration.
        block_size (int): Tokens per block. block_size = max_seq_len is MDLM.
        device (torch.device | None): Device to store the parameters on.
        dtype (torch.dtype | None): Data type of the parameters.

    Attributes:
        backbone (DiT): The shared diffusion transformer.
        attn_mask (Tensor): (2n, 2n) block diffusion mask.
        positions (Tensor): (2n,) RoPE positions arange(n) repeated twice.
    """
    attn_mask: Bool[Tensor, "two_seq_len two_seq_len"]
    positions: Int[Tensor, " two_seq_len"]

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        d_cond: int,
        block_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        assert max_seq_len % block_size == 0, (
            f"max_seq_len {max_seq_len} must be a multiple of block_size {block_size}"
        )

        self.backbone = DiT(
            vocab_size, max_seq_len, d_model, n_heads,
            n_layers, d_ff, d_cond, device, dtype,
        )
        self.mask_index = vocab_size - 1
        self.max_seq_len = max_seq_len
        self.block_size = block_size

        # both are fully determined by (n, block_size): build once, never save
        self.register_buffer(
            "attn_mask", block_diff_mask(max_seq_len, block_size, device), persistent=False,
        )
        self.register_buffer(
            "positions", torch.arange(max_seq_len, device=device).repeat(2), persistent=False,
        )

    def forward(
        self,
        xt: Int[Tensor, "batch seq"],
        x0: Int[Tensor, "batch seq"],
    ) -> Float[Tensor, "batch seq vocab"]:
        """Training-time forward on the 2L input.

        Args:
            xt: Shape (batch, n), noisy tokens (masked per block).
            x0: Shape (batch, n), clean tokens.

        Returns:
            SUBS log-probs for the x_t half, shape (batch, n, vocab).
        """
        n = self.max_seq_len
        assert xt.shape[1] == n, f"expected sequences of length {n} (the precomputed mask), got {xt.shape[1]}"

        x_input = torch.cat([xt, x0], dim=-1)
        sigma = torch.zeros(xt.shape[0], device=xt.device)

        logits = self.backbone(x_input, sigma, mask=self.attn_mask, positions=self.positions)[:, :n]

        return subs_parameterization(logits, xt, self.mask_index)
