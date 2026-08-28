"""
J10a: config fields and modern-block primitives, in isolation.

Nothing here is wired into a model. J10a exists as its own step so each primitive
can be checked against its FORMULA before it can influence a training run -- an
RMSNorm that is subtly wrong still trains, still converges, and still reports a
plausible loss, and it would be discovered as an unexplained architecture result
rather than as a bug.

The other half of these tests is the compatibility surface: every new field is
architecture identity, so a checkpoint trained under one value must not load
under another. `position_encoding` is the dangerous one -- it does not change the
parameter set, so without a guard a rope checkpoint would load cleanly into a
learned-position model and silently compute something else.
"""
from __future__ import annotations

import pytest
import torch

from smallAntibodyGen.models.mlm import MLMConfig
from smallAntibodyGen.models.transformer import (
    RMSNorm,
    SwiGLU,
    gelu_ffn_parameter_count,
    resolve_head_count,
    swiglu_parameter_count,
    swiglu_width_matching_gelu,
    validate_head_count,
)


def _config(**overrides) -> MLMConfig:
    base = dict(
        vocab_size=35, pad_token_id=0, max_length=64,
        d_model=32, n_heads=4, n_layers=2, d_ff=64,
    )
    base.update(overrides)
    return MLMConfig(**base)


# --------------------------------------------------------------------------- #
# RMSNorm
# --------------------------------------------------------------------------- #
def test_rmsnorm_matches_its_formula():
    """Checked against the arithmetic, not against another implementation."""
    norm = RMSNorm(4, eps=0.0)
    x = torch.tensor([[3.0, 4.0, 0.0, 0.0]])
    # rms = sqrt((9 + 16) / 4) = 2.5
    expected = x / 2.5
    assert torch.allclose(norm(x), expected, atol=1e-6)


def test_rmsnorm_does_not_subtract_the_mean():
    """
    The defining difference from LayerNorm. A constant-shifted input keeps its
    shift under RMSNorm; under LayerNorm it would be centred away. Getting this
    wrong yields something that trains fine and is not RMSNorm.
    """
    norm = RMSNorm(4, eps=0.0)
    shifted = torch.tensor([[10.0, 10.0, 10.0, 10.0]])
    out = norm(shifted)
    assert torch.allclose(out, torch.ones_like(out), atol=1e-6)
    assert out.abs().sum() > 0  # LayerNorm would have produced zeros


def test_rmsnorm_has_a_weight_and_no_bias():
    """Strictly fewer parameters than LayerNorm, which is why norm_type is part
    of the architecture identity rather than a free swap."""
    norm = RMSNorm(8)
    names = dict(norm.named_parameters())
    assert set(names) == {"weight"}
    assert torch.equal(names["weight"], torch.ones(8))


def test_rmsnorm_reduces_in_float32_but_returns_the_input_dtype():
    """
    Two properties at once, and an earlier draft got the second one wrong.

    The reduction is a sum of squares; accumulating it in half precision loses
    accuracy exactly where normalization is most sensitive. But the OUTPUT dtype
    must follow the input, matching `nn.LayerNorm` outside autocast -- scaling by
    a float32 `weight` before the cast silently promoted a float16 input to
    float32, which would have changed the dtype of the whole residual stream just
    by swapping the norm type.
    """
    norm = RMSNorm(4, eps=0.0)
    x = torch.tensor([[3.0, 4.0, 0.0, 0.0]], dtype=torch.float16)
    out = norm(x)
    assert out.dtype == torch.float16
    assert torch.allclose(out.float(), (x.float() / 2.5), atol=1e-3)


def test_rmsnorm_is_finite_on_an_all_zero_row():
    """eps is what stops a padded or dead row becoming NaN and poisoning the
    whole batch through the residual stream."""
    norm = RMSNorm(4)
    out = norm(torch.zeros(2, 4))
    assert torch.isfinite(out).all()


# --------------------------------------------------------------------------- #
# SwiGLU
# --------------------------------------------------------------------------- #
def test_swiglu_matches_its_formula():
    ffn = SwiGLU(d_model=3, hidden_dim=5)
    with torch.no_grad():
        x = torch.randn(2, 3)
        expected = ffn.down_proj(
            torch.nn.functional.silu(ffn.gate_proj(x)) * ffn.up_proj(x)
        )
        assert torch.allclose(ffn(x), expected, atol=1e-6)


def test_swiglu_has_three_projections():
    """Three, not two -- which is the entire parameter-accounting problem."""
    ffn = SwiGLU(d_model=8, hidden_dim=16)
    linears = [m for m in ffn.modules() if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 3


def test_swiglu_rejects_a_nonpositive_width():
    with pytest.raises(ValueError, match="hidden_dim must be > 0"):
        SwiGLU(d_model=8, hidden_dim=0)


def test_reusing_d_ff_as_the_swiglu_width_would_be_a_50_percent_larger_ffn():
    """
    The mistake the config refuses to allow, quantified. At d_model=256 and
    d_ff=1024 -- this repository's actual shape -- naively passing d_ff builds a
    feed-forward with half again as many parameters, and a bakeoff would read the
    extra capacity as an architecture win.
    """
    gelu = gelu_ffn_parameter_count(256, 1024)
    naive = swiglu_parameter_count(256, 1024)
    assert naive > gelu
    assert 0.49 < (naive - gelu) / gelu < 0.51


def test_width_proposal_lands_near_parameter_parity():
    """The helper's job: propose a width whose parameter count matches GeLU's,
    and say by how much it misses."""
    proposal = swiglu_width_matching_gelu(256, 1024)
    assert proposal.proposed_hidden_dim % 8 == 0
    assert abs(proposal.residual_fraction) < 0.01
    assert proposal.gelu_parameters == gelu_ffn_parameter_count(256, 1024)
    assert proposal.swiglu_parameters == swiglu_parameter_count(
        256, proposal.proposed_hidden_dim
    )


def test_width_proposal_reports_a_signed_residual():
    """
    Rounding to a tensor-friendly multiple cannot hit parity exactly, and a
    bakeoff claiming "parameter-matched" without saying by how much is claiming
    more than it measured. The sign matters too: it says which arm is larger.
    """
    proposal = swiglu_width_matching_gelu(256, 1024)
    assert isinstance(proposal.residual_parameters, int)
    assert proposal.residual_parameters == (
        proposal.swiglu_parameters - proposal.gelu_parameters
    )


def test_width_proposal_rounds_to_nearest_not_down():
    """Rounding down would make every SwiGLU arm systematically smaller -- the
    same bias as the naive mistake with the sign flipped."""
    # 2/3 * 100 = 66.67 -> nearest multiple of 8 is 64; nearest of 4 is 68.
    assert swiglu_width_matching_gelu(16, 100, multiple_of=4).proposed_hidden_dim == 68


# --------------------------------------------------------------------------- #
# Head-count resolution
# --------------------------------------------------------------------------- #
def test_head_counts_default_to_the_legacy_value():
    assert resolve_head_count(None, 8) == 8
    assert resolve_head_count(4, 8) == 4


def test_encoder_and_cross_attention_heads_resolve_independently():
    """
    J10 explicitly does not modernize the antigen cross-attention. Varying the
    encoder's head shape (head_dim=64 is its own bakeoff axis) must not drag the
    fusion along with it.
    """
    config = _config(d_model=64, n_heads=8, encoder_n_heads=4)
    assert config.resolved_encoder_n_heads == 4
    assert config.resolved_cross_attention_n_heads == 8


def test_a_head_count_that_does_not_divide_d_model_is_rejected_by_name():
    with pytest.raises(ValueError, match="encoder_n_heads=5 does not divide"):
        validate_head_count(32, 5, "encoder_n_heads")
    with pytest.raises(ValueError, match="encoder_n_heads"):
        _config(d_model=32, encoder_n_heads=5).validate()
    with pytest.raises(ValueError, match="cross_attention_n_heads"):
        _config(d_model=32, cross_attention_n_heads=7).validate()


# --------------------------------------------------------------------------- #
# Config: defaults are the legacy block
# --------------------------------------------------------------------------- #
def test_defaults_describe_the_current_model():
    """
    An unmodified config must build exactly today's architecture. This is the
    guarantee that lets J10a land before the bakeoff has run.
    """
    config = _config()
    assert config.position_encoding == "learned"
    assert config.norm_type == "layernorm"
    assert config.ffn_type == "gelu"
    assert config.swiglu_hidden_dim is None
    assert config.attention_bias is True
    assert config.ffn_bias is True
    assert config.encoder_n_heads is None
    assert config.cross_attention_n_heads is None
    assert config.resolved_encoder_n_heads == config.n_heads
    assert config.resolved_cross_attention_n_heads == config.n_heads
    config.validate()


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("position_encoding", "sinusoidal", "position_encoding must be"),
        ("norm_type", "batchnorm", "norm_type must be"),
        ("ffn_type", "relu", "ffn_type must be"),
    ],
)
def test_unknown_values_are_rejected(field, value, message):
    with pytest.raises(ValueError, match=message):
        _config(**{field: value}).validate()


def test_swiglu_without_an_explicit_width_is_rejected():
    """
    The width is never inferred from d_ff. The error says why, because the
    obvious "fix" -- passing d_ff -- is the actual mistake.
    """
    with pytest.raises(ValueError, match="requires an explicit swiglu_hidden_dim"):
        _config(ffn_type="swiglu").validate()
    with pytest.raises(ValueError, match="1.5x larger"):
        _config(ffn_type="swiglu").validate()


def test_a_width_set_under_gelu_is_rejected_rather_than_ignored():
    """Silently ignoring it would let a config claim a SwiGLU width while
    building a GeLU feed-forward."""
    with pytest.raises(ValueError, match="silently ignored"):
        _config(ffn_type="gelu", swiglu_hidden_dim=680).validate()


def test_a_valid_swiglu_configuration_passes():
    proposal = swiglu_width_matching_gelu(32, 64)
    _config(ffn_type="swiglu", swiglu_hidden_dim=proposal.proposed_hidden_dim).validate()


# --------------------------------------------------------------------------- #
# Compatibility surface
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "field",
    [
        "position_encoding",
        "norm_type",
        "ffn_type",
        "swiglu_hidden_dim",
        "attention_bias",
        "ffn_bias",
        "encoder_n_heads",
        "cross_attention_n_heads",
    ],
)
def test_every_new_field_is_warm_start_checked(field):
    """
    Each is architecture identity. `position_encoding` is the one that most
    needs this: it does not change the parameter SET, so a strict load would
    accept a rope checkpoint into a learned-position model and compute something
    different without complaint.
    """
    from smallAntibodyGen import experiment

    assert field in experiment.WARM_START_ARCHITECTURE_KEYS


def test_the_new_fields_are_not_treated_as_antigen_only():
    """
    They describe the ANTIBODY encoder, so they must not inherit the
    stream-introducing exemption that lets antigen fields change on a
    stage-2 -> stage-3 transition.
    """
    from smallAntibodyGen import experiment

    for field in ("position_encoding", "norm_type", "ffn_type", "encoder_n_heads"):
        assert field not in experiment.ANTIGEN_ONLY_ARCHITECTURE_KEYS


def test_the_fields_reach_the_architecture_fingerprint(tmp_path):
    """A field absent from the fingerprint would let two different architectures
    share a run hash."""
    from smallAntibodyGen import experiment
    from smallAntibodyGen.tokenizer import AminoAcidTokenizer

    tok = AminoAcidTokenizer()
    legacy = experiment.architecture_manifest(_config(), tok, "AntibodyMLM")
    modern = experiment.architecture_manifest(
        _config(norm_type="rmsnorm"), tok, "AntibodyMLM"
    )
    assert legacy["model_config"]["norm_type"] == "layernorm"
    assert modern["model_config"]["norm_type"] == "rmsnorm"
    assert experiment.hash_payload(legacy) != experiment.hash_payload(modern)
