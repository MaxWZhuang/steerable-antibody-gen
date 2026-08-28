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
