import math

import torch
from einops import einsum, rearrange
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn


class Linear(nn.Module):
    """A bias-free linear transformation, y = x @ W.T.

    Follows the interface of torch.nn.Linear, without the bias term. Every layer
    in the DiT is built on this (the project uses no biases anywhere).

    Args:
        in_features (int): Size of the last dimension of the input.
        out_features (int): Size of the last dimension of the output.
        device (torch.device | None): Device to store the parameter on.
        dtype (torch.dtype | None): Data type of the parameter.

    Attributes:
        weight (nn.Parameter): Shape (out_features, in_features) -- i.e. W, not
            W-transpose. Each row holds the weights of a single output unit, so
            a row is contiguous in row-major memory.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), dtype=dtype, device=device)
        )

        # truncated normal, sigma^2 = 2 / (d_in + d_out), cut at +/- 3 sigma
        std = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(
        self,
        x: Float[Tensor, "... d_in"],
    ) -> Float[Tensor, "... d_out"]:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    """Token embedding table; maps integer token IDs to dense vectors.

    Follows the interface of torch.nn.Embedding. The forward pass is a row
    gather.

    Args:
        vocab_size (int): Vocabulary size, including [MASK].
        d_model (int): Dimension of each embedding vector.
        device (torch.device | None): Device to store the parameter on.
        dtype (torch.dtype | None): Data type of the parameter.

    Attributes:
        weight (nn.Parameter): Shape (vocab_size, d_model). Vocab first so that
            each embedding vector is a contiguous row.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        # (vocab, d_model): rows are gathered, so d_model must be the last axis
        self.weight = nn.Parameter(
            torch.empty((vocab_size, d_model), dtype=dtype, device=device)
        )

        # unit-variance truncated normal, cut at +/- 3
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(
        self,
        token_ids: Int[Tensor, "..."],
    ) -> Float[Tensor, "... d_model"]:
        """Look up embeddings for a batch of token IDs.

        Args:
            token_ids: Integer tensor of any shape, values in [0, vocab_size).

        Returns:
            Tensor of shape token_ids.shape + (d_model,).
        """
        return self.weight[token_ids]


class TimestepEmbedder(nn.Module):
    """Embeds the scalar noise level sigma of each sequence into a conditioning
    vector, via sinusoidal (Fourier) features followed by a 2-layer MLP:

        f_i      = exp(-ln(P) * i / (D/2))                 i = 0 .. D/2 - 1
        emb(s)   = [cos(s * f_0), ..., cos(s * f_{D/2-1}),
                    sin(s * f_0), ..., sin(s * f_{D/2-1})]  (cos first, then sin)
        out(s)   = fc2(SiLU(fc1(emb(s))))

    Args:
        d_cond (int): Size of the conditioning vector (independent of d_model).
        frequency_dim (int): D, the number of sinusoidal features (D/2
            frequencies x {cos, sin}). Independent of d_model; fc1 maps it to
            d_cond.
        period (float): P, the max period. Sets the lowest frequency f_min ~ 1/P.
            Not a sequence length.
        device (torch.device | None): Device to store the parameters on.
        dtype (torch.dtype | None): Data type of the MLP parameters.

    Attributes:
        frequencies (Tensor): Shape (frequency_dim/2,), always float32, registered
            with persistent=False -- fully derived from the constructor args.
        fc1 (Linear): (d_cond, frequency_dim) projection of the sinusoidal features.
        fc2 (Linear): (d_cond, d_cond) output projection, after the SiLU.
    """
    frequencies: Float[Tensor, " half_frequency_dim"]

    def __init__(
        self,
        d_cond: int,
        frequency_dim: int = 256,
        period: float = 10000.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        half_frequency_dim = frequency_dim // 2
        exponents = torch.arange(half_frequency_dim, dtype=torch.float32, device=device) / half_frequency_dim
        frequencies = torch.exp(-math.log(period) * exponents)
        self.register_buffer("frequencies", frequencies, persistent=False)

        self.fc1 = Linear(frequency_dim, d_cond, device, dtype)
        self.fc2 = Linear(d_cond, d_cond, device, dtype)

    def forward(
        self,
        sigma: Float[Tensor, " batch"],
    ) -> Float[Tensor, "batch d_cond"]:
        """Embed one noise level per sequence.

        Args:
            sigma: Shape (batch,). One noise level per sequence, shared by all
                of its tokens.

        Returns:
            Tensor of shape (batch, d_cond), before the backbone's extra SiLU.
        """
        args = einsum(sigma, self.frequencies, "batch, freq -> batch freq")
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        return self.fc2(silu(self.fc1(embedding)))


class RoPE(nn.Module):
    """Rotary Position Embedding, non-interleaved ("rotate-half") layout.

    Rotates coordinate pairs of q and k as 2D vectors by an angle proportional
    to the token's position:

        theta(p, k) = p / Theta^(2k / d_k)      k = 0 .. d_k/2 - 1

    Pairing is rotate-half -- coordinate k is paired with k + d_k/2, i.e.
    (x0, x_{d/2}), (x1, x_{d/2+1}). Has no learnable parameters.

    Args:
        head_dim (int): PER-HEAD dimension (d_model // n_heads), not d_model.
            Must be even.
        max_seq_len (int): Number of position rows to precompute.
        theta (float): Base Theta of the frequency ladder.
        device (torch.device | None): Device to build the tables on.

    Attributes:
        sin_table, cos_table (Tensor): Shape (max_seq_len, head_dim/2), float32,
            registered with persistent=False -- fully derived from the
            constructor args.
    """
    sin_table: Float[Tensor, "max_seq_len half_head_dim"]
    cos_table: Float[Tensor, "max_seq_len half_head_dim"]

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        theta: float = 10000.0,
        device: torch.device | None = None,
    ):
        super().__init__()
        assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"

        seq_positions = torch.arange(max_seq_len, dtype=torch.float64, device=device)

        # pair_idx = 0, 2, 4, ... -> one inverse frequency per pair
        pair_idx = torch.arange(0, head_dim, 2, dtype=torch.float64, device=device)
        inv_freq = theta ** (-pair_idx / head_dim)

        # outer product -> (max_seq_len, head_dim/2)
        angles = einsum(seq_positions, inv_freq, "seq, half -> seq half")

        self.register_buffer("sin_table", torch.sin(angles).float(), persistent=False)
        self.register_buffer("cos_table", torch.cos(angles).float(), persistent=False)

    def forward(
        self,
        x: Float[Tensor, "... seq d_k"],
        positions: Int[Tensor, " seq"] | None = None,
    ) -> Float[Tensor, "... seq d_k"]:
        """Rotate the pairs of x according to each token's position.

        Args:
            x: Shape (..., seq, d_k). Any number of leading batch/head dims.
            positions: Shape (seq,), integer ABSOLUTE positions. Defaults to
                0 .. seq-1.

        Returns:
            Tensor of the same shape and dtype as x.
        """
        if positions is None:
            positions = torch.arange(x.shape[-2], device=x.device)

        sin = self.sin_table[positions].to(x.dtype)
        cos = self.cos_table[positions].to(x.dtype)

        # (..., seq, d_k/2) each: the two halves hold the two coordinates of every pair
        x_1, x_2 = x.chunk(2, dim=-1)

        out_1 = x_1 * cos - x_2 * sin
        out_2 = x_1 * sin + x_2 * cos

        # put the halves back end to end: (..., seq, d_k)
        return torch.cat([out_1, out_2], dim=-1)


class LayerNorm(nn.Module):
    """Weight-only layer normalization:

        LN(x) = (x - mean(x)) / sqrt(Var(x) + eps) * gamma

    Normalizes over the LAST axis only; every leading dimension is treated as a
    batch dimension.

    Args:
        d_model (int): Size of the normalized (last) dimension.
        eps (float): Added to the variance for numerical stability.
        device (torch.device | None): Device to store the parameter on.
        dtype (torch.dtype | None): Data type of the parameter.

    Attributes:
        weight (nn.Parameter): Learned per-feature gain gamma of shape (d_model,),
            initialized to ones so the layer starts as a pure normalizer.
    """
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.ones((d_model,), dtype=dtype, device=device)
        )
        self.eps = eps

    def forward(
        self,
        x: Float[Tensor, "... d_model"],
    ) -> Float[Tensor, "... d_model"]:
        """Normalize x over its last dimension, returning the original dtype."""
        # mean / variance in bf16 lose precision, so normalize in fp32 and cast back
        original_dtype = x.dtype
        upcasted_x = x.to(dtype=torch.float32)

        numerator = upcasted_x - upcasted_x.mean(dim=-1, keepdim=True)
        inv_std = torch.rsqrt(upcasted_x.var(dim=-1, keepdim=True, correction=0) + self.eps)

        return (self.weight * numerator * inv_std).to(dtype=original_dtype)


class MLP(nn.Module):
    """Position-wise feed-forward network with a tanh-approximate GELU:

        MLP(x) = fc2(GELU(fc1(x)))

    Args:
        d_model (int): Input and output dimension.
        d_ff (int): Inner dimension, conventionally 4 * d_model.
        device (torch.device | None): Device to store the parameters on.
        dtype (torch.dtype | None): Data type of the parameters.

    Attributes:
        fc1 (Linear): (d_ff, d_model) up-projection.
        fc2 (Linear): (d_model, d_ff) down-projection.
        gelu (nn.GELU): tanh approximation.
    """
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.fc1 = Linear(d_model, d_ff, device, dtype)
        self.fc2 = Linear(d_ff, d_model, device, dtype)
        self.gelu = nn.GELU(approximate="tanh")

    def forward(
        self,
        x: Float[Tensor, "... d_model"],
    ) -> Float[Tensor, "... d_model"]:
        """Apply the feed-forward transform. Shape is preserved."""
        return self.fc2(self.gelu(self.fc1(x)))


class BidirectionalMultiHeadAttention(nn.Module):
    """Multi-head self-attention with RoPE and a fused QKV projection.

    Bidirectional: with mask=None every position attends to every position (no
    causal mask). An optional boolean mask (True = may attend) is supported.

    Args:
        d_model (int): Model dimension. Must be divisible by n_heads.
        n_heads (int): Number of attention heads.
        rope (RoPE): Rotary embedding applied to q and k. Passed in so that a
            single instance (and its sin/cos tables) is shared by every layer.
        device (torch.device | None): Device to store the parameters on.
        dtype (torch.dtype | None): Data type of the parameters.

    Attributes:
        qkv (Linear): (3 * d_model, d_model) fused projection.
        out (Linear): (d_model, d_model) output projection.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        rope: RoPE,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        assert d_model % n_heads == 0, "d_model should be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rope = rope

        # one (3d x d) matmul instead of three (d x d): x is read once, one launch
        self.qkv = Linear(d_model, 3 * d_model, device, dtype)
        self.out = Linear(d_model, d_model, device, dtype)

    def forward(
        self,
        x: Float[Tensor, "... seq d_model"],
        mask: Bool[Tensor, "... seq_q seq_k"] | None = None,
        positions: Int[Tensor, " seq"] | None = None,
    ) -> Float[Tensor, "... seq d_model"]:
        """Attend over the sequence.

        Args:
            x: Shape (..., seq, d_model).
            mask: Boolean, True = may attend, broadcastable to
                (..., n_heads, seq_q, seq_k). None = full bidirectional attention.
            positions: Shape (seq,), absolute positions for RoPE. Defaults to
                0 .. seq-1.

        Returns:
            Tensor of shape (..., seq, d_model).
        """
        qkv_proj = self.qkv(x)

        # split the fused axis (three, heads, d_k)
        # unpack into q, k, v, each (..., n_heads, seq, d_k).
        q, k, v = rearrange(
            qkv_proj,
            "... seq (three n_heads d_k) -> three ... n_heads seq d_k",
            three=3, n_heads=self.n_heads,
        )

        # positions are encoded by rotating q and k only
        q = self.rope(q, positions)
        k = self.rope(k, positions)

        # merge heads back: (..., n_heads, seq, d_k) -> (..., seq, d_model)
        o = scaled_dot_product_attention(q, k, v, mask)
        o = rearrange(o, "... n_heads seq d_k -> ... seq (n_heads d_k)")

        return self.out(o)


class AdaLN(nn.Module):
    """Adaptive LayerNorm modulation: maps the conditioning vector c to n_chunks
    per-channel vectors (shift / scale / gate) that modulate a block's sublayers.

        [chunk_0, ..., chunk_{n-1}] = split_n(W c)

    adaLN-Zero: W is zero-initialized, so at init every chunk is 0 -- shifts and
    scales are identity (modulate(x, 0, 0) = x) and gates switch every sublayer
    off, so each DiT block starts as the identity function.

    Args:
        d_cond (int): Size of the conditioning vector c.
        d_model (int): Size of each output chunk (the model dimension).
        n_chunks (int): Number of chunks: 6 in a DiT block (shift/scale/gate for
            attention and MLP), 2 in the final layer (shift/scale).
        device (torch.device | None): Device to store the parameters on.
        dtype (torch.dtype | None): Data type of the parameters.

    Attributes:
        proj (Linear): (n_chunks * d_model, d_cond) projection, zero-initialized.
            Output features are laid out (n_chunks, d_model): chunk j is
            features [j * d_model, (j + 1) * d_model).
    """
    def __init__(
        self,
        d_cond: int,
        d_model: int,
        n_chunks: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.proj = Linear(d_cond, d_model * n_chunks, device, dtype)
        self.d_model = d_model

        nn.init.zeros_(self.proj.weight)

    def forward(
        self,
        c: Float[Tensor, "... d_cond"],
    ) -> Float[Tensor, "n_chunks ... 1 d_model"]:
        """Project c and split it into n_chunks modulation vectors.

        Args:
            c: Shape (batch, d_cond), one conditioning vector per sequence.

        Returns:
            Tensor of shape (n_chunks, batch, 1, d_model). Unpacks along dim 0
            into n_chunks tensors; the size-1 axis broadcasts each one over the
            sequence, since conditioning is per sequence, not per token.
        """
        result = rearrange(
            self.proj(c),
            "... (n_chunks d_model) -> n_chunks ... 1 d_model",
            d_model=self.d_model
        )

        return result


class DiTBlock(nn.Module):
    """Diffusion Transformer block: pre-norm attention + MLP, each modulated and
    gated by the conditioning vector through adaLN:

        h   = x + gate_msa * Attn(modulate(LN1(x), shift_msa, scale_msa))
        out = h + gate_mlp * MLP(modulate(LN2(h), shift_mlp, scale_mlp))

    The gate scales each sublayer's output before the residual add; with the
    zero-initialized adaLN every gate starts at 0, so the block starts as the
    identity.

    Args:
        d_model (int): Model dimension.
        n_heads (int): Number of attention heads.
        d_ff (int): MLP inner dimension.
        d_cond (int): Size of the conditioning vector.
        rope (RoPE): Rotary embedding shared with every other block.
        device (torch.device | None): Device to store the parameters on.
        dtype (torch.dtype | None): Data type of the parameters.

    Attributes:
        ln1, ln2 (LayerNorm): Weight-only norms before attention / MLP.
        attn (BidirectionalMultiHeadAttention): Self-attention with RoPE.
        mlp (MLP): GELU feed-forward network.
        ada_ln (AdaLN): 6 chunks, in order: shift_msa, scale_msa, gate_msa,
            shift_mlp, scale_mlp, gate_mlp.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        d_cond: int,
        rope: RoPE,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.rope = rope

        self.ln1 = LayerNorm(d_model, device=device, dtype=dtype)
        self.ln2 = LayerNorm(d_model, device=device, dtype=dtype)

        self.mlp = MLP(d_model, d_ff, device, dtype)
        self.attn = BidirectionalMultiHeadAttention(d_model, n_heads, rope, device, dtype)

        self.ada_ln = AdaLN(d_cond, d_model, 6, device, dtype)

    def forward(
        self,
        x: Float[Tensor, "... seq d_model"],
        c: Float[Tensor, "... d_cond"],
        mask: Bool[Tensor, "... seq seq"] | None = None,
        positions: Int[Tensor, " seq"] | None = None,
    ) -> Float[Tensor, "... seq d_model"]:
        """Apply the block.

        Args:
            x: Shape (batch, seq, d_model), the residual stream.
            c: Shape (batch, d_cond), the conditioning vector.
            mask: Boolean, True = may attend. None = full bidirectional attention.
            positions: Shape (seq,), absolute positions for RoPE.

        Returns:
            Tensor of shape (batch, seq, d_model).
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.ada_ln(c)

        h = x + gate_msa * self.attn(modulate(self.ln1(x), shift_msa, scale_msa), mask, positions)
        out = h + gate_mlp * self.mlp(modulate(self.ln2(h), shift_mlp, scale_mlp))

        return out


class FinalLayer(nn.Module):
    """Output head: adaLN-modulated LayerNorm, then a projection to the vocabulary.

        logits = lm_head(modulate(LN(x), shift, scale))

    Like a DiT block but with only shift and scale (no gate). lm_head is
    zero-initialized, so the model outputs all-zero logits at init, and it is not
    tied to the input embedding.

    Args:
        d_model (int): Model dimension.
        vocab_size (int): Vocabulary size, including [MASK].
        d_cond (int): Size of the conditioning vector.
        device (torch.device | None): Device to store the parameters on.
        dtype (torch.dtype | None): Data type of the parameters.

    Attributes:
        ln (LayerNorm): Weight-only final norm.
        lm_head (Linear): (vocab_size, d_model) output projection, zero-initialized.
        ada_ln (AdaLN): 2 chunks, in order: shift, scale.
    """
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        d_cond: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.ln = LayerNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device, dtype)

        self.ada_ln = AdaLN(d_cond, d_model, 2, device, dtype)

        nn.init.zeros_(self.lm_head.weight)

    def forward(
        self,
        x: Float[Tensor, "... seq d_model"],
        c: Float[Tensor, "... d_cond"],
    ) -> Float[Tensor, "... seq vocab"]:
        """Map the final hidden states to raw logits.

        Args:
            x: Shape (batch, seq, d_model).
            c: Shape (batch, d_cond), the conditioning vector.

        Returns:
            Tensor of shape (batch, seq, vocab_size).
        """
        shift, scale = self.ada_ln(c)

        logits = self.lm_head(modulate(self.ln(x), shift, scale))

        return logits


class DiT(nn.Module):
    """Diffusion Transformer backbone: (tokens x_t, noise level sigma) -> logits.

        x      = Embedding(tokens)
        c      = SiLU(TimestepEmbedder(sigma))
        x      = DiTBlock(x, c)  x n_layers
        logits = FinalLayer(x, c)

    Bidirectional throughout. A single RoPE instance is built here and shared by
    every block. Returns raw logits.

    Args:
        vocab_size (int): Vocabulary size, including [MASK] (the last index).
        max_seq_len (int): Longest sequence RoPE precomputes positions for.
        d_model (int): Model dimension.
        n_heads (int): Number of attention heads.
        n_layers (int): Number of DiT blocks.
        d_ff (int): MLP inner dimension.
        d_cond (int): Size of the conditioning vector (independent of d_model).
        device (torch.device | None): Device to store the parameters on.
        dtype (torch.dtype | None): Data type of the parameters.

    Attributes:
        token_embeddings (Embedding): (vocab_size, d_model) input embedding.
        timestep_embedder (TimestepEmbedder): sigma -> (batch, d_cond).
        rope (RoPE): Shared rotary embedding, head_dim = d_model // n_heads.
        layers (nn.ModuleList[DiTBlock]): The n_layers blocks.
        final_layer (FinalLayer): Output head.
    """
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        d_cond: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)
        self.timestep_embedder = TimestepEmbedder(d_cond, device=device, dtype=dtype)
        self.rope = RoPE(d_model // n_heads, max_seq_len, device=device)

        self.layers = nn.ModuleList(
            [
                DiTBlock(
                    d_model, n_heads, d_ff,
                    d_cond, self.rope, device, dtype,
                )
                for _ in range(n_layers)
            ]
        )

        self.final_layer = FinalLayer(d_model, vocab_size, d_cond, device, dtype)

    def forward(
        self,
        tokens: Int[Tensor, "... seq"],
        sigma: Float[Tensor, " batch"],
        mask: Bool[Tensor, "... seq seq"] | None = None,
        positions: Int[Tensor, " seq"] | None = None,
    ) -> Float[Tensor, "... seq vocab"]:
        """Run the backbone.

        Args:
            tokens: Shape (batch, seq), token ids of x_t (may contain [MASK]).
            sigma: Shape (batch,), one noise level per sequence.
            mask: Boolean attention mask, True = may attend. None = full
                bidirectional attention.
            positions: Shape (seq,), absolute positions for RoPE (BD3-LM).

        Returns:
            Raw logits of shape (batch, seq, vocab_size).
        """
        # the embedder omits this final SiLU; the backbone applies it before adaLN
        c = silu(self.timestep_embedder(sigma))

        x = self.token_embeddings(tokens)

        for layer in self.layers:
            x = layer(x, c, mask, positions)

        x = self.final_layer(x, c)

        return x


def softmax(x: Float[Tensor, "..."], dim: int = -1) -> Float[Tensor, "..."]:
    """Numerically stable softmax along `dim`."""
    z = x - x.amax(dim=dim, keepdim=True)

    exp_z = torch.exp(z)
    sum_exp = torch.sum(exp_z, dim=dim, keepdim=True)

    return exp_z / sum_exp


def scaled_dot_product_attention(
    q: Float[Tensor, "... seq_q d_k"],
    k: Float[Tensor, "... seq_k d_k"],
    v: Float[Tensor, "... seq_k d_v"],
    mask: Bool[Tensor, "... seq_q seq_k"] | None = None,
) -> Float[Tensor, "... seq_q d_v"]:
    """Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k) + M) V.

    Args:
        q: Shape (..., seq_q, d_k).
        k: Shape (..., seq_k, d_k).
        v: Shape (..., seq_k, d_v).
        mask: Boolean, True = may attend, broadcastable to (..., seq_q, seq_k).
            Masked scores become -inf (M above). A row with every key masked
            yields NaN.

    Returns:
        Tensor of shape (..., seq_q, d_v), in v's dtype.
    """
    d_k = q.shape[-1]
    scale = 1 / (d_k ** 0.5)

    # upcast before the matmul, so the logits themselves are fp32
    scores = einsum(
        q.to(dtype=torch.float32), k.to(dtype=torch.float32),
        "... seq_q d_k, ... seq_k d_k -> ... seq_q seq_k",
    ) * scale

    if mask is not None:
        scores = torch.where(mask, scores, -torch.inf)

    probabilities = softmax(scores, dim=-1).to(dtype=v.dtype)

    return einsum(probabilities, v, "... seq_q seq_k, ... seq_k d_v -> ... seq_q d_v")


def silu(x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    """SiLU / Swish activation: x * sigmoid(x)."""
    return x * torch.sigmoid(x)

def modulate(
    h: Float[Tensor, "batch seq d_model"],
    shift: Float[Tensor, "batch 1 d_model"],
    scale: Float[Tensor, "batch 1 d_model"],
) -> Float[Tensor, "batch seq d_model"]:
    """adaLN modulation: h * (1 + scale) + shift.

    The (1 + scale) form makes zero-initialized adaLN an identity: with
    shift = scale = 0 the input passes through unchanged.
    """
    return h * (1 + scale) + shift


def log_softmax(x: Float[Tensor, "..."], dim: int = -1) -> Float[Tensor, "..."]:
    """Numerically stable log-softmax along `dim`:

        log_softmax(x)_i = (x_i - m) - log(sum_j exp(x_j - m)),   m = max_j x_j
    """
    shifted = x - x.amax(dim=dim, keepdim=True)
    shifted_exp = torch.exp(shifted)

    return shifted - torch.log(shifted_exp.sum(dim=dim, keepdim=True))