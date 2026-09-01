"""
Modern encoder-block primitives for the Rung-1 architecture bakeoff (J10).

Nothing here is wired into a model yet. J10a is deliberately primitives only:
each piece is small enough to check against its formula in isolation, before any
of it can influence a training run. Integration is J10c, and the legacy
configuration must keep passing the entire existing suite when it lands.

Why these three, and what the evidence does and does not say. AMPLIFY and ESM-C
motivate pre-norm, rotary positions, and gated feed-forwards as a modern default
for protein language models. That is a default, not a proof for this data regime
or this model size -- which is exactly why they arrive as opt-in config fields
with legacy defaults and get compared, rather than as an upgrade.

The parameter-accounting helper exists because the most likely way this bakeoff
goes wrong is not a bug: it is comparing a SwiGLU model that quietly has more
parameters than its GeLU counterpart, and reading the extra capacity as an
architecture win.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


class RMSNorm(nn.Module):
    """
    Root-mean-square layer normalization.

    ``x * rsqrt(mean(x^2) + eps) * weight``

    Differs from ``nn.LayerNorm`` in two ways that both matter here: there is no
    mean subtraction, and there is no bias. So it is not a drop-in replacement at
    the checkpoint level -- an RMSNorm block has strictly fewer parameters than a
    LayerNorm one, which is why the norm type is part of the architecture
    fingerprint rather than a free choice.

    The reduction runs in float32 regardless of the input dtype. The variance of a
    d_model-wide activation is a sum of squares, and accumulating it in float16
    loses precision exactly where the normalization is most sensitive; the cast
    costs nothing measurable and removes a class of AMP-only discrepancy that
    would otherwise look like an architecture effect.

    The OUTPUT dtype follows the input, matching ``nn.LayerNorm`` outside
    autocast. One documented difference: under autocast ``nn.LayerNorm`` is on
    torch's fp32 force-list and returns float32, while this module is a custom
    op that autocast does not intercept and so returns the input dtype. Norm type
    is a bakeoff axis, so that difference is worth knowing rather than
    discovering -- it is a property of the arm, not a bug.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.normalized_shape = int(normalized_shape)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        dtype = hidden.dtype
        promoted = hidden.float()
        variance = promoted.pow(2).mean(dim=-1, keepdim=True)
        normalized = promoted * torch.rsqrt(variance + self.eps)
        # Scale in float32 and cast ONCE at the end. Casting before the scale
        # would let a float32 `weight` promote the result back up, so a float16
        # input would silently come out float32 -- which is what an earlier draft
        # of this did, caught by the dtype test rather than by reading.
        return (normalized * self.weight.float()).to(dtype)

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}"


class SwiGLU(nn.Module):
    """
    Gated feed-forward network: ``down(silu(gate(x)) * up(x))``.

    Three projections rather than two, which is the whole parameter-accounting
    problem: at a given hidden width a SwiGLU FFN has 1.5x the parameters of a
    GeLU FFN. ``hidden_dim`` is therefore REQUIRED and never inferred from the
    legacy ``d_ff``. Passing ``d_ff`` here would silently build a 50% larger
    feed-forward and invite the bakeoff to credit the architecture for the extra
    capacity.

    Use :func:`swiglu_width_matching_gelu` to pick a width that matches the GeLU
    parameter count, and report the residual difference rather than assuming it
    is zero.
    """

    def __init__(self, d_model: int, hidden_dim: int, bias: bool = True) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("SwiGLU hidden_dim must be > 0")
        self.d_model = int(d_model)
        self.hidden_dim = int(hidden_dim)
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(hidden)) * self.up_proj(hidden)
        )

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, hidden_dim={self.hidden_dim}"


def gelu_ffn_parameter_count(d_model: int, d_ff: int, bias: bool = True) -> int:
    """Parameters in the legacy two-projection GeLU feed-forward."""
    weights = d_model * d_ff + d_ff * d_model
    biases = (d_ff + d_model) if bias else 0
    return weights + biases


def swiglu_parameter_count(d_model: int, hidden_dim: int, bias: bool = True) -> int:
    """Parameters in a three-projection SwiGLU feed-forward."""
    weights = 2 * (d_model * hidden_dim) + hidden_dim * d_model
    biases = (2 * hidden_dim + d_model) if bias else 0
    return weights + biases


@dataclass(frozen=True)
class WidthProposal:
    """
    A proposed SwiGLU width and how closely it matches the GeLU baseline.

    ``residual_parameters`` is signed and reported rather than hidden: a rounded
    width will not match exactly, and a bakeoff that claims "parameter-matched"
    without saying by how much is claiming more than it measured.
    """

    d_model: int
    gelu_d_ff: int
    proposed_hidden_dim: int
    gelu_parameters: int
    swiglu_parameters: int
    multiple_of: int

    @property
    def residual_parameters(self) -> int:
        """SwiGLU minus GeLU. Positive means the SwiGLU arm is larger."""
        return self.swiglu_parameters - self.gelu_parameters

    @property
    def residual_fraction(self) -> float:
        return self.residual_parameters / self.gelu_parameters


def swiglu_width_matching_gelu(
    d_model: int,
    d_ff: int,
    bias: bool = True,
    multiple_of: int = 8,
) -> WidthProposal:
    """
    Propose the nearest hardware-friendly SwiGLU width matching a GeLU FFN's
    parameter count.

    Exact parity would need ``hidden = 2/3 * d_ff``, which is rarely an integer
    and rarely a good tensor shape. This rounds to a multiple of ``multiple_of``
    and reports the residual, so the bakeoff can state the mismatch instead of
    asserting there is none.

    Rounding to the NEAREST multiple, not down: rounding down would make every
    SwiGLU arm systematically smaller, which is the same bias as the naive
    ``hidden = d_ff`` mistake with the sign flipped.
    """
    if multiple_of <= 0:
        raise ValueError("multiple_of must be > 0")
    target = (2.0 / 3.0) * d_ff
    candidate = max(multiple_of, int(round(target / multiple_of)) * multiple_of)
    return WidthProposal(
        d_model=d_model,
        gelu_d_ff=d_ff,
        proposed_hidden_dim=candidate,
        gelu_parameters=gelu_ffn_parameter_count(d_model, d_ff, bias),
        swiglu_parameters=swiglu_parameter_count(d_model, candidate, bias),
        multiple_of=multiple_of,
    )


def resolve_head_count(explicit: int | None, legacy: int) -> int:
    """
    Resolve a head count, with ``None`` inheriting the legacy ``n_heads``.

    Self-attention and cross-attention resolve SEPARATELY so the bakeoff can vary
    the encoder's head shape (``head_dim=64`` is its own candidate axis) without
    touching the antigen fusion, which J10 explicitly does not modernize. Making
    each explicit is what stops a change to ``n_heads`` from silently altering
    both.
    """
    return legacy if explicit is None else int(explicit)


def validate_head_count(d_model: int, n_heads: int, label: str) -> None:
    """Reject a head count that does not divide ``d_model``, naming which one."""
    if n_heads <= 0:
        raise ValueError(f"{label} must be > 0")
    if d_model % n_heads != 0:
        raise ValueError(
            f"{label}={n_heads} does not divide d_model={d_model} "
            f"(head_dim would be {d_model / n_heads})"
        )


# --------------------------------------------------------------------------- #
# J10b: rotary positions and non-causal self-attention
#
# Utilities only -- nothing here is wired into a model. Integration is J10c.
# --------------------------------------------------------------------------- #
def position_ids_from_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Position ids for real tokens, counting from 1, with padding at 0.

    Deliberately the SAME convention as ``LearnedPositionalEmbedding``: a cumsum
    over the attention mask, so padding consumes no position. Switching
    ``position_encoding`` from learned to rope must change how position is
    REPRESENTED, not which position a token gets -- otherwise the bakeoff would
    be comparing two things at once, and the difference would be invisible
    because both arms would look internally consistent.
    """
    if attention_mask.dim() != 2:
        raise ValueError("attention_mask must have shape [batch_size, seq_len]")
    position_ids = attention_mask.long().cumsum(dim=1)
    return position_ids.masked_fill(attention_mask == 0, 0)


class RotaryEmbedding(nn.Module):
    """
    Rotary position embedding (RoFormer).

    Encodes absolute position by rotating each 2-dimensional slice of a head's
    query and key vectors, with the useful consequence that the resulting
    attention logit depends only on the RELATIVE offset between two positions.

    Has **no learned parameters** -- ``inv_freq`` is a buffer, not a Parameter.
    That is the point of listing it as an architecture candidate: it removes the
    learned position table entirely, so a rope model has strictly fewer
    parameters than a learned-position one and cannot run out of table at a
    length it never saw. It does NOT by itself make the model good at unseen
    lengths; nothing trains those positions.

    ``head_dim`` must be even: the rotation acts on pairs of channels.
    """

    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(
                f"RoPE head_dim must be even (rotation acts on channel pairs); got {head_dim}"
            )
        self.head_dim = int(head_dim)
        self.base = float(base)
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        # A buffer, not a Parameter: it must move with the model and be saved,
        # but never receive a gradient.
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            position_ids: ``[batch, seq]`` integer positions.

        Returns:
            ``(cos, sin)``, each ``[batch, 1, seq, head_dim]`` -- the head axis is
            size 1 so they broadcast across heads.
        """
        positions = position_ids.to(self.inv_freq.dtype)
        # [batch, seq, head_dim/2]
        freqs = positions.unsqueeze(-1) * self.inv_freq
        # Duplicate rather than interleave, matching `rotate_half` below. The two
        # conventions are both valid and NOT interchangeable; mixing them is a
        # silent correctness bug, so they live next to each other.
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().unsqueeze(1), emb.sin().unsqueeze(1)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the halves of the last dimension: ``[a, b] -> [-b, a]``."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply the rotation to queries and keys.

    Values are deliberately NOT rotated: RoPE encodes position in the
    query/key inner product, and rotating values would mix position into the
    content being aggregated.
    """
    cos = cos.to(query.dtype)
    sin = sin.to(query.dtype)
    rotated_query = (query * cos) + (rotate_half(query) * sin)
    rotated_key = (key * cos) + (rotate_half(key) * sin)
    return rotated_query, rotated_key


class RotarySelfAttention(nn.Module):
    """
    Non-causal multi-head self-attention with rotary positions.

    Non-causal by construction: this is a masked language model, so a token must
    see the whole sequence in both directions. There is no causal mask anywhere
    in this class and a test asserts its absence -- a causal mask here would
    quietly halve the context and still train.

    Padding is handled by an explicit key-padding mask. A row with NO valid keys
    would make softmax divide by zero and produce NaN, which then propagates
    through the residual stream and poisons the whole batch; such rows are
    detected and their output zeroed instead.

    ``use_rope=False`` turns the rotation OFF entirely, leaving plain non-causal
    attention with no positional term of its own. This is what a learned-position
    block needs, and it has to be an explicit flag: ``position_ids=None`` means
    "derive the positions from the mask", NOT "skip the rotation", so a caller
    that tried to disable RoPE by withholding positions would get RoPE anyway --
    on top of the learned table -- and the arm would still train.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        bias: bool = True,
        dropout: float = 0.0,
        rope_base: float = 10000.0,
        use_rope: bool = True,
    ) -> None:
        super().__init__()
        validate_head_count(d_model, n_heads, "n_heads")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.dropout = float(dropout)
        self.use_rope = bool(use_rope)

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        # Not built at all when unused: the even-``head_dim`` requirement is a
        # property of the rotation, so it should not constrain an arm that does
        # not rotate. `inv_freq` is a non-persistent buffer either way, so this
        # changes no state dict.
        self.rotary = (
            RotaryEmbedding(self.head_dim, base=rope_base) if self.use_rope else None
        )

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        return x.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden: ``[batch, seq, d_model]``.
            attention_mask: ``[batch, seq]``, 1 for real tokens and 0 for padding.
            position_ids: optional explicit positions; derived from the mask when
                omitted, using the same convention as the learned table. Ignored
                when ``use_rope`` is False -- there is nothing to position.
        """
        if hidden.dim() != 3:
            raise ValueError("hidden must have shape [batch, seq, d_model]")
        if attention_mask.shape != hidden.shape[:2]:
            raise ValueError("attention_mask must have shape [batch, seq]")

        query = self._split_heads(self.q_proj(hidden))
        key = self._split_heads(self.k_proj(hidden))
        value = self._split_heads(self.v_proj(hidden))

        if self.use_rope:
            if position_ids is None:
                position_ids = position_ids_from_mask(attention_mask)
            cos, sin = self.rotary(position_ids)
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # [batch, 1, 1, seq]; True means "may be attended to". No causal term.
        keep = attention_mask.bool()[:, None, None, :]

        # A row with no valid key would softmax over an all -inf row -> NaN, which
        # then spreads through the residual stream into every OTHER example in the
        # batch. Such rows get a permissive mask here and are zeroed after the
        # output projection, so a fully padded example cannot poison its batch.
        row_is_empty = ~attention_mask.bool().any(dim=1)          # [batch]
        safe_keep = keep | row_is_empty[:, None, None, None]

        context = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=safe_keep,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )

        batch, _, seq, _ = context.shape
        merged = context.transpose(1, 2).reshape(batch, seq, self.d_model)
        output = self.out_proj(merged)
        # AFTER the projection, not before: out_proj has a bias, so zeroing the
        # context alone would still leave a constant on a fully padded row.
        return output.masked_fill(row_is_empty[:, None, None], 0.0)


# --------------------------------------------------------------------------- #
# J10c: the configurable encoder block and stack
# --------------------------------------------------------------------------- #
LEGACY_BLOCK = {
    "position_encoding": "learned",
    "norm_type": "layernorm",
    "ffn_type": "gelu",
    "attention_bias": True,
    "ffn_bias": True,
    "encoder_n_heads": None,
}


def is_legacy_block(config) -> bool:
    """
    Is this exactly the pre-J10 encoder?

    Used to route: a legacy configuration keeps running through
    ``nn.TransformerEncoder`` untouched, so "the legacy config passes the entire
    existing suite" is true by construction rather than by a parity argument. Any
    departure -- even one that could in principle be expressed by the custom
    block -- takes the new path, because a block that is *nearly* the old one is
    the worst outcome: it would silently differ.
    """
    return all(getattr(config, key) == value for key, value in LEGACY_BLOCK.items())


def build_norm(config, d_model: int) -> nn.Module:
    """LayerNorm or RMSNorm, per ``config.norm_type``."""
    if config.norm_type == "rmsnorm":
        return RMSNorm(d_model)
    return nn.LayerNorm(d_model)


def build_ffn(config, d_model: int) -> nn.Module:
    """
    The feed-forward branch, per ``config.ffn_type``.

    The SwiGLU width comes from ``swiglu_hidden_dim`` and is never derived from
    ``d_ff`` -- see `SwiGLU`. `MLMConfig.validate` already refuses the
    unset case; this asserts it again because building a silently mis-sized FFN
    is worse than a crash.
    """
    if config.ffn_type == "swiglu":
        if config.swiglu_hidden_dim is None:
            raise ValueError("ffn_type='swiglu' requires swiglu_hidden_dim")
        return SwiGLU(d_model, config.swiglu_hidden_dim, bias=config.ffn_bias)
    activation = nn.GELU() if config.activation == "gelu" else nn.ReLU()
    return nn.Sequential(
        nn.Linear(d_model, config.d_ff, bias=config.ffn_bias),
        activation,
        nn.Linear(config.d_ff, d_model, bias=config.ffn_bias),
    )


class ModernEncoderLayer(nn.Module):
    """
    One configurable encoder layer.

    Pre-norm and post-norm are both supported because ``norm_first`` and
    ``norm_type`` are ORTHOGONAL: where the norm sits is a different decision
    from which norm it is, and collapsing them would make one unreachable. The
    canonical configuration sets both explicitly (``norm_first: true``,
    ``norm_type: rmsnorm``).

    Attention is always :class:`RotarySelfAttention`; when
    ``position_encoding == "learned"`` the rotation is simply not applied, so the
    two position schemes share one attention implementation rather than two that
    could drift apart. "Not applied" is carried by the explicit ``use_rope`` flag
    on the attention module, not by withholding ``position_ids``: withholding
    them means "derive them from the mask", so the learned arm used to add the
    learned table AND rotate on top of it -- two position encodings in an
    experiment whose whole purpose is to compare one against the other.
    """

    def __init__(self, config, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.norm_first = bool(config.norm_first)
        self.use_rope = config.position_encoding == "rope"
        self.norm1 = build_norm(config, d_model)
        self.norm2 = build_norm(config, d_model)
        self.attn = RotarySelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            bias=config.attention_bias,
            dropout=config.dropout,
            use_rope=self.use_rope,
        )
        self.ffn = build_ffn(config, d_model)
        self.dropout = nn.Dropout(config.dropout)

    def _attend(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        # With a learned table the positions are already IN the embeddings, so
        # applying RoPE too would encode position twice. The attention module
        # owns that decision through `use_rope`; passing `None` here would only
        # have made it DERIVE the positions and rotate anyway.
        return self.attn(hidden, attention_mask, position_ids)

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.norm_first:
            hidden = hidden + self.dropout(
                self._attend(self.norm1(hidden), attention_mask, position_ids)
            )
            hidden = hidden + self.dropout(self.ffn(self.norm2(hidden)))
            return hidden
        hidden = self.norm1(
            hidden + self.dropout(self._attend(hidden, attention_mask, position_ids))
        )
        return self.norm2(hidden + self.dropout(self.ffn(hidden)))


class ModernEncoderStack(nn.Module):
    """``n_layers`` configurable layers. The final norm lives on the encoder."""

    def __init__(self, config, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            ModernEncoderLayer(config, d_model, n_heads) for _ in range(config.n_layers)
        )

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden = layer(hidden, attention_mask, position_ids)
        return hidden


def apply_modern_residual_depth_scaling(stack: ModernEncoderStack, config) -> None:
    """
    The modern-stack counterpart of ``apply_residual_depth_scaling``.

    Same rule, same two kinds of projection: the ones that WRITE into the
    residual stream. For attention that is ``out_proj``; for the FFN it is
    ``down_proj`` under SwiGLU and the second ``Linear`` under GeLU. Scaling the
    wrong tensor here would look identical at init and diverge over training.
    """
    if not config.scale_residual_init:
        return
    scale = 1.0 / math.sqrt(2.0 * max(1, config.n_layers))
    with torch.no_grad():
        for layer in stack.layers:
            layer.attn.out_proj.weight.mul_(scale)
            ffn = layer.ffn
            if isinstance(ffn, SwiGLU):
                ffn.down_proj.weight.mul_(scale)
            else:
                # nn.Sequential(Linear, activation, Linear): the LAST Linear.
                writers = [m for m in ffn if isinstance(m, nn.Linear)]
                writers[-1].weight.mul_(scale)
