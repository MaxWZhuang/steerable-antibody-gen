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
    RotaryEmbedding,
    RotarySelfAttention,
    SwiGLU,
    apply_rotary_pos_emb,
    position_ids_from_mask,
    rotate_half,
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


# --------------------------------------------------------------------------- #
# J10b: rotary positions and non-causal self-attention
#
# Every property here is one that would leave a model that still trains, still
# converges, and reports a plausible loss while being quietly wrong -- a causal
# mask that halves the context, padding that leaks, or positions that shift when
# the encoding changes.
# --------------------------------------------------------------------------- #
def test_position_ids_match_the_learned_table_convention():
    """
    Switching `position_encoding` must change how position is REPRESENTED, not
    which position a token gets. Both paths derive ids by cumsum over the mask,
    counting real tokens from 1 and leaving padding at 0, so padding consumes no
    position.
    """
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    ids = position_ids_from_mask(mask)
    assert ids.tolist() == [[1, 2, 3, 0, 0], [1, 2, 3, 4, 5]]


def test_rotary_has_no_learned_parameters():
    """
    A rope model has strictly fewer parameters than a learned-position one --
    there is no position table at all. `inv_freq` is a buffer so it moves with
    the model without ever receiving a gradient.
    """
    rotary = RotaryEmbedding(8)
    assert list(rotary.parameters()) == []
    assert "inv_freq" in dict(rotary.named_buffers())


def test_rotary_rejects_an_odd_head_dim():
    """The rotation acts on channel pairs; an odd head_dim would silently drop
    or misalign a channel."""
    with pytest.raises(ValueError, match="head_dim must be even"):
        RotaryEmbedding(7)


def test_rotate_half_matches_its_definition():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    assert torch.equal(rotate_half(x), torch.tensor([[-3.0, -4.0, 1.0, 2.0]]))


def test_rotation_preserves_vector_norm():
    """A rotation changes direction, not length. A norm change here would mean
    RoPE is rescaling activations, which is not what it is for."""
    rotary = RotaryEmbedding(8)
    q = torch.randn(1, 2, 5, 8)
    cos, sin = rotary(torch.arange(5).unsqueeze(0))
    rq, _ = apply_rotary_pos_emb(q, q, cos, sin)
    assert torch.allclose(rq.norm(dim=-1), q.norm(dim=-1), atol=1e-5)


def test_attention_logits_depend_only_on_relative_position():
    """
    THE defining property of RoPE, and the reason it is a candidate at all.

    Two vectors at absolute positions (m, n) and at (m+k, n+k) must produce the
    same query-key inner product. If this fails, the implementation encodes
    absolute position wearing RoPE's name -- and it would still train.
    """
    rotary = RotaryEmbedding(16)
    q = torch.randn(1, 1, 1, 16)
    k = torch.randn(1, 1, 1, 16)

    def logit(pos_q: int, pos_k: int) -> float:
        cq, sq = rotary(torch.tensor([[pos_q]]))
        ck, sk = rotary(torch.tensor([[pos_k]]))
        rq, _ = apply_rotary_pos_emb(q, q, cq, sq)
        _, rk = apply_rotary_pos_emb(k, k, ck, sk)
        return float((rq * rk).sum())

    assert logit(3, 7) == pytest.approx(logit(10, 14), abs=1e-4)
    assert logit(3, 7) == pytest.approx(logit(103, 107), abs=1e-3)
    # And it is genuinely position-dependent, not a constant.
    assert logit(3, 7) != pytest.approx(logit(3, 20), abs=1e-3)


def test_self_attention_output_shape():
    attn = RotarySelfAttention(d_model=32, n_heads=4).eval()
    hidden = torch.randn(2, 6, 32)
    mask = torch.ones(2, 6, dtype=torch.long)
    assert attn(hidden, mask).shape == (2, 6, 32)


def test_attention_is_not_causal():
    """
    This is a masked language model: a token must see the whole sequence in both
    directions. A causal mask would halve the effective context and the model
    would still train, so the absence is asserted rather than assumed -- the
    first position's output must react to the LAST token.
    """
    attn = RotarySelfAttention(d_model=32, n_heads=4).eval()
    mask = torch.ones(1, 5, dtype=torch.long)
    hidden = torch.randn(1, 5, 32)

    with torch.no_grad():
        before = attn(hidden, mask)[0, 0].clone()
        changed = hidden.clone()
        changed[0, 4] += 5.0          # perturb the LAST token only
        after = attn(changed, mask)[0, 0]

    assert not torch.allclose(before, after, atol=1e-6), (
        "position 0 did not react to the final token: attention is causal"
    )


def test_padded_tokens_cannot_affect_unpadded_outputs():
    """
    Padding leakage is silent: the model trains, and validation quietly depends
    on whatever garbage sits in the pad slots. Changing padded content must leave
    every real position bit-identical.
    """
    attn = RotarySelfAttention(d_model=32, n_heads=4).eval()
    mask = torch.tensor([[1, 1, 1, 0, 0]])
    hidden = torch.randn(1, 5, 32)

    with torch.no_grad():
        before = attn(hidden, mask)[:, :3].clone()
        polluted = hidden.clone()
        polluted[0, 3:] = 99.0
        after = attn(polluted, mask)[:, :3]

    assert torch.allclose(before, after, atol=1e-6)


def test_a_fully_padded_row_is_finite_and_does_not_poison_the_batch():
    """
    Softmax over an all-masked row divides by zero and yields NaN, which then
    spreads through the residual stream into every other example in the batch.
    The row is zeroed instead -- and the point is the OTHER row stays correct.
    """
    attn = RotarySelfAttention(d_model=32, n_heads=4).eval()
    mask = torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1]])
    hidden = torch.randn(2, 4, 32)

    with torch.no_grad():
        out = attn(hidden, mask)
        alone = attn(hidden[1:], mask[1:])

    assert torch.isfinite(out).all()
    assert torch.equal(out[0], torch.zeros_like(out[0]))
    assert torch.allclose(out[1], alone[0], atol=1e-6)


def test_backward_is_finite_with_heavy_padding():
    """Gradients, not just activations: a NaN that only appears in the backward
    pass is just as fatal and far less visible."""
    attn = RotarySelfAttention(d_model=32, n_heads=4)
    mask = torch.tensor([[1, 0, 0, 0], [0, 0, 0, 0]])
    hidden = torch.randn(2, 4, 32, requires_grad=True)
    attn(hidden, mask).sum().backward()
    assert torch.isfinite(hidden.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in attn.parameters() if p.grad is not None)


def test_eval_mode_is_deterministic():
    """Dropout must be off in eval, or two identical evaluations disagree and
    every comparison inherits the noise."""
    attn = RotarySelfAttention(d_model=32, n_heads=4, dropout=0.5).eval()
    hidden = torch.randn(1, 5, 32)
    mask = torch.ones(1, 5, dtype=torch.long)
    with torch.no_grad():
        assert torch.equal(attn(hidden, mask), attn(hidden, mask))


def test_bias_free_attention_really_has_no_biases():
    """`attention_bias: false` is one of the adopted canonical settings, so the
    parameter set has to actually reflect it."""
    attn = RotarySelfAttention(d_model=32, n_heads=4, bias=False)
    assert not any(name.endswith(".bias") for name, _ in attn.named_parameters())
    with_bias = RotarySelfAttention(d_model=32, n_heads=4, bias=True)
    assert any(name.endswith(".bias") for name, _ in with_bias.named_parameters())


def test_head_count_must_divide_d_model():
    with pytest.raises(ValueError, match="does not divide"):
        RotarySelfAttention(d_model=32, n_heads=5)


def test_explicit_position_ids_override_the_mask_derivation():
    """
    J10c will need to pass positions explicitly in places. The override has to
    actually take effect, or a caller would silently get mask-derived positions.
    """
    attn = RotarySelfAttention(d_model=32, n_heads=4).eval()
    hidden = torch.randn(1, 4, 32)
    mask = torch.ones(1, 4, dtype=torch.long)
    with torch.no_grad():
        derived = attn(hidden, mask)
        shifted = attn(hidden, mask, position_ids=torch.tensor([[5, 6, 7, 8]]))
    # Same relative offsets, so the ATTENTION pattern matches even though the
    # absolute positions differ -- which is the relative-position property again,
    # now observed end to end through the whole attention op.
    assert torch.allclose(derived, shifted, atol=1e-4)


# --------------------------------------------------------------------------- #
# J10c: integration
#
# The legacy configuration is not re-implemented -- it still runs through
# nn.TransformerEncoder -- so "legacy passes the existing suite" is true by
# construction. What needs testing is the modern path, and that the routing
# between them cannot be entered by accident.
# --------------------------------------------------------------------------- #
CANONICAL_MODERN = dict(
    position_encoding="rope",
    norm_type="rmsnorm",
    ffn_type="swiglu",
    attention_bias=False,
    ffn_bias=False,
    norm_first=True,
)


def _modern(**overrides):
    params = dict(CANONICAL_MODERN)
    params["swiglu_hidden_dim"] = 40
    params.update(overrides)
    return _config(**params)


def test_a_legacy_config_takes_the_legacy_path():
    """Routing, asserted. If a legacy config quietly took the new block, every
    existing checkpoint and every recorded metric would silently change meaning."""
    from smallAntibodyGen.models.mlm import TransformerSequenceEncoder

    encoder = TransformerSequenceEncoder(_config())
    assert encoder.is_legacy_block is True
    assert isinstance(encoder.encoder, torch.nn.TransformerEncoder)
    assert encoder.position_embedding is not None


@pytest.mark.parametrize(
    "override",
    [
        {"position_encoding": "rope"},
        {"norm_type": "rmsnorm"},
        {"ffn_type": "swiglu", "swiglu_hidden_dim": 40},
        {"attention_bias": False},
        {"ffn_bias": False},
        {"encoder_n_heads": 2},
    ],
)
def test_any_single_departure_takes_the_modern_path(override):
    """
    A block that is NEARLY the legacy one is the worst outcome -- it would differ
    silently. One changed field is enough to leave the legacy path entirely.
    """
    from smallAntibodyGen.models.mlm import TransformerSequenceEncoder

    encoder = TransformerSequenceEncoder(_config(**override))
    assert encoder.is_legacy_block is False
    assert not isinstance(encoder.encoder, torch.nn.TransformerEncoder)


def test_a_rope_encoder_has_no_learned_positional_parameter():
    """
    The adopted canonical architecture removes the learned table entirely. A
    non-parameter rotary cache is fine -- `inv_freq` is a buffer -- but no
    PARAMETER may be a position table.
    """
    from smallAntibodyGen.models.mlm import TransformerSequenceEncoder

    encoder = TransformerSequenceEncoder(_modern())
    assert encoder.position_embedding is None
    assert encoder.uses_learned_positions is False
    names = [name for name, _ in encoder.named_parameters()]
    assert not any("position_embedding" in name for name in names)
    # The rotary tables exist, as buffers.
    buffers = [name for name, _ in encoder.named_buffers()]
    assert any("inv_freq" in name for name in buffers)


def test_a_learned_position_modern_block_keeps_its_table():
    """`position_encoding` and the other fields are independent: choosing RMSNorm
    must not silently remove the learned positions as well."""
    from smallAntibodyGen.models.mlm import TransformerSequenceEncoder

    encoder = TransformerSequenceEncoder(_config(norm_type="rmsnorm"))
    assert encoder.position_embedding is not None
    assert encoder.uses_learned_positions is True


def test_norm_placement_and_norm_operator_stay_independent():
    """
    Where the norm sits and which norm it is are orthogonal architecture
    properties. Collapsing them would make post-norm RMSNorm unreachable, and the
    canonical config sets both explicitly.
    """
    from smallAntibodyGen.models.transformer import RMSNorm as _RMSNorm
    from smallAntibodyGen.models.mlm import TransformerSequenceEncoder

    pre = TransformerSequenceEncoder(_modern(norm_first=True))
    post = TransformerSequenceEncoder(_modern(norm_first=False))
    assert pre.encoder.layers[0].norm_first is True
    assert post.encoder.layers[0].norm_first is False
    for encoder in (pre, post):
        assert isinstance(encoder.encoder.layers[0].norm1, _RMSNorm)
        assert isinstance(encoder.final_norm, _RMSNorm)


def test_the_final_norm_follows_the_configured_norm_type():
    """One normalization operator through the whole stack, including the final
    norm -- a LayerNorm tail on an RMSNorm stack is a silent hybrid."""
    from smallAntibodyGen.models.transformer import RMSNorm as _RMSNorm
    from smallAntibodyGen.models.mlm import TransformerSequenceEncoder

    assert isinstance(TransformerSequenceEncoder(_modern()).final_norm, _RMSNorm)
    assert isinstance(
        TransformerSequenceEncoder(_config()).final_norm, torch.nn.LayerNorm
    )


# --- behaviour of the integrated modern encoder ----------------------------- #
def _mlm(**overrides):
    from smallAntibodyGen.models.mlm import AntibodyMLM

    return AntibodyMLM(_modern(**overrides))


def test_modern_model_forward_and_backward_are_finite():
    model = _mlm()
    ids = torch.randint(3, 30, (2, 12))
    mask = torch.ones_like(ids)
    logits = model(ids, mask)
    logits = logits[0] if isinstance(logits, tuple) else logits
    assert logits.shape == (2, 12, 35)
    logits.sum().backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


def test_modern_model_is_finite_under_heavy_padding():
    """Full masking must not produce NaN -- and one dead row must not take the
    batch with it."""
    model = _mlm()
    ids = torch.randint(3, 30, (2, 8))
    mask = torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]])
    out = model(ids, mask)
    out = out[0] if isinstance(out, tuple) else out
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


def test_modern_model_padding_does_not_leak_into_real_positions():
    model = _mlm().eval()
    ids = torch.randint(3, 30, (1, 8))
    mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]])
    with torch.no_grad():
        before = model(ids, mask)
        before = (before[0] if isinstance(before, tuple) else before)[:, :4].clone()
        polluted = ids.clone()
        polluted[0, 4:] = 29
        after = model(polluted, mask)
        after = (after[0] if isinstance(after, tuple) else after)[:, :4]
    assert torch.allclose(before, after, atol=1e-5)


def test_modern_model_is_not_causal():
    model = _mlm().eval()
    ids = torch.randint(3, 30, (1, 6))
    mask = torch.ones_like(ids)
    with torch.no_grad():
        base = model(ids, mask)
        base = (base[0] if isinstance(base, tuple) else base)[0, 0].clone()
        changed = ids.clone()
        changed[0, 5] = (changed[0, 5] + 7) % 30 + 3
        after = model(changed, mask)
        after = (after[0] if isinstance(after, tuple) else after)[0, 0]
    assert not torch.allclose(base, after, atol=1e-6)


def test_modern_model_eval_is_deterministic():
    model = _mlm(dropout=0.5).eval()
    ids = torch.randint(3, 30, (1, 6))
    mask = torch.ones_like(ids)
    with torch.no_grad():
        a = model(ids, mask)
        b = model(ids, mask)
    a = a[0] if isinstance(a, tuple) else a
    b = b[0] if isinstance(b, tuple) else b
    assert torch.equal(a, b)


def test_modern_model_saves_and_reloads_strictly():
    """
    Strict reload, because this repo's rule is that a parameter mismatch means
    retrain and never a silent coercion. A rope model's state dict must round
    trip without the missing/unexpected keys a half-wired branch would produce.
    """
    model = _mlm()
    state = model.state_dict()
    fresh = _mlm()
    fresh.load_state_dict(state, strict=True)

    model.eval()
    fresh.eval()
    ids = torch.randint(3, 30, (1, 10))
    mask = torch.ones_like(ids)
    with torch.no_grad():
        a = model(ids, mask)
        b = fresh(ids, mask)
    a = a[0] if isinstance(a, tuple) else a
    b = b[0] if isinstance(b, tuple) else b
    assert torch.equal(a, b)


def test_a_legacy_checkpoint_does_not_load_into_a_modern_model():
    """
    The two are different architectures. Loading across them must fail loudly --
    a silent partial load would train from a half-initialized model behind an
    ordinary-looking loss curve.
    """
    from smallAntibodyGen.models.mlm import AntibodyMLM

    legacy_state = AntibodyMLM(_config()).state_dict()
    with pytest.raises(RuntimeError):
        _mlm().load_state_dict(legacy_state, strict=True)


def test_tied_output_weights_still_tie_under_the_modern_block():
    from smallAntibodyGen.models.mlm import AntibodyMLM

    model = AntibodyMLM(_modern(tie_weights=True))
    assert model.lm_head.weight is model.sequence_encoder.token_embedding.weight


def test_residual_depth_scaling_reaches_the_swiglu_writer():
    """
    The scaled tensor must be the one that WRITES into the residual stream --
    `down_proj` for SwiGLU. Scaling the wrong projection looks identical at init
    and diverges over training, which is the hardest kind of thing to attribute.
    """
    from smallAntibodyGen.models.mlm import TransformerSequenceEncoder

    scaled = TransformerSequenceEncoder(_modern(scale_residual_init=True))
    unscaled = TransformerSequenceEncoder(_modern(scale_residual_init=False))
    scaled_std = scaled.encoder.layers[0].ffn.down_proj.weight.std().item()
    unscaled_std = unscaled.encoder.layers[0].ffn.down_proj.weight.std().item()
    assert scaled_std < unscaled_std * 0.8
    # And the INPUT-side projections are left alone.
    assert scaled.encoder.layers[0].ffn.gate_proj.weight.std().item() == pytest.approx(
        unscaled.encoder.layers[0].ffn.gate_proj.weight.std().item(), rel=0.25
    )


def test_canonical_parameter_counts_are_pinned():
    """
    A snapshot of the three architectures at the canonical shape, confirmed BY
    CONSTRUCTION rather than by formula.

    The two modern arms are what J11 compares. The 1,585,152 gap between them is
    entirely the SwiGLU width: 3 x 256 x (1024 - 680) x 6 layers. Losing the
    learned position table (289 x 256 = 73,984 rows, the +1 being the reserved
    pad row) is why the 680 arm lands slightly BELOW legacy despite gating.
    """
    from smallAntibodyGen.models.mlm import AntibodyMLM

    def count(config):
        return sum(p.numel() for p in AntibodyMLM(config).parameters())

    canonical = dict(
        vocab_size=35, pad_token_id=0, max_length=288,
        d_model=256, n_heads=8, n_layers=6, d_ff=1024, dropout=0.1,
    )
    legacy = count(MLMConfig(**canonical))
    modern_680 = count(MLMConfig(**canonical, **CANONICAL_MODERN, swiglu_hidden_dim=680))
    modern_1024 = count(MLMConfig(**canonical, **CANONICAL_MODERN, swiglu_hidden_dim=1024))

    assert legacy == 4_822_530
    assert modern_680 == 4_719_106
    assert modern_1024 == 6_304_258
    assert modern_1024 - modern_680 == 1_585_152 == 3 * 256 * (1024 - 680) * 6
    assert (289 * 256) == 73_984


# --------------------------------------------------------------------------- #
# Dormant architecture controls
#
# Both bugs below are the same shape: a knob that VALIDATES and then does
# nothing. Neither changes the parameter set, so neither could be caught by a
# shape error, a strict load, or a crash -- an arm labelled "learned positions"
# or "4 cross-attention heads" would train, converge, and be written up as an
# architecture result for a block it never ran.
#
# The probes are permutation-based rather than value-based: self-attention with
# no positional term of its own is permutation-EQUIVARIANT, so `f(Px) == P f(x)`
# holds exactly when (and only when) no rotation is applied. Each probe is
# paired with a power check on the opposite arm, because a probe that cannot
# distinguish the two arms would pass against the broken code.
# --------------------------------------------------------------------------- #
_PERMUTATION = torch.tensor([3, 0, 5, 1, 4, 2])


def _equivariance_gap(config) -> float:
    """``max |f(Px) - P f(x)|`` for one `ModernEncoderLayer` on a full mask."""
    from smallAntibodyGen.models.transformer import ModernEncoderLayer

    torch.manual_seed(0)
    layer = ModernEncoderLayer(config, config.d_model, config.n_heads).eval()
    hidden = torch.randn(1, 6, config.d_model)
    mask = torch.ones(1, 6, dtype=torch.long)
    positions = torch.arange(1, 7).unsqueeze(0)
    with torch.no_grad():
        base = layer(hidden, mask, positions)
        permuted = layer(hidden[:, _PERMUTATION, :], mask, positions)
    return float((permuted - base[:, _PERMUTATION, :]).abs().max())


def test_a_learned_position_block_does_not_also_rotate():
    """
    THE finding. `position_ids=None` means "derive the positions from the mask",
    NOT "skip the rotation", so a learned-position block used to add the learned
    table and then rotate on top of it -- two position encodings in the one
    experiment whose purpose is to compare them.

    With no positional term inside attention the block is permutation-
    equivariant, so this gap is float noise. Under RoPE it is not (next test).
    """
    assert _equivariance_gap(_config(norm_type="rmsnorm")) < 1e-5


def test_the_rope_arm_is_position_sensitive_so_the_probe_has_power():
    """
    Power check for the probe above, on the SAME code path with one field
    changed. A probe that reported "no positional term" for both arms would pass
    against the broken implementation and prove nothing.
    """
    gap = _equivariance_gap(_config(norm_type="rmsnorm", position_encoding="rope"))
    assert gap > 1e-3


def test_position_encoding_changes_what_the_modern_block_computes():
    """
    Stated as the property the bakeoff needs: two arms differing ONLY in
    `position_encoding` must not be the same function. Before the fix these were
    bit-identical -- the field selected a learned table on top and changed
    nothing inside attention.
    """
    from smallAntibodyGen.models.transformer import ModernEncoderLayer

    def run(config):
        torch.manual_seed(1234)
        layer = ModernEncoderLayer(config, config.d_model, config.n_heads).eval()
        torch.manual_seed(7)
        hidden = torch.randn(2, 5, config.d_model)
        mask = torch.ones(2, 5, dtype=torch.long)
        with torch.no_grad():
            return layer(hidden, mask, torch.arange(1, 6).unsqueeze(0).expand(2, 5))

    learned = run(_config(norm_type="rmsnorm"))
    rope = run(_config(norm_type="rmsnorm", position_encoding="rope"))
    # Same seed, so the weights are identical: any difference is the rotation.
    assert not torch.allclose(learned, rope, atol=1e-6)


def test_a_learned_position_block_builds_no_rotary_table():
    """The even-`head_dim` requirement belongs to the rotation, so an arm that
    does not rotate should not carry it."""
    from smallAntibodyGen.models.transformer import ModernEncoderLayer

    learned = ModernEncoderLayer(_config(norm_type="rmsnorm"), 32, 4)
    rope = ModernEncoderLayer(_config(norm_type="rmsnorm", position_encoding="rope"), 32, 4)
    assert learned.attn.use_rope is False and learned.attn.rotary is None
    assert rope.attn.use_rope is True and rope.attn.rotary is not None


def test_use_rope_false_ignores_position_ids_entirely():
    """
    Unit-level statement of the same contract, with its own power check.

    The positions here are NOT a uniform shift, so they change the RELATIVE
    offsets: RoPE genuinely responds to them, which is what makes "no response"
    evidence of no rotation rather than evidence of RoPE's shift-invariance.
    """
    hidden = torch.randn(1, 4, 32)
    mask = torch.ones(1, 4, dtype=torch.long)
    scrambled = torch.tensor([[1, 5, 2, 9]])

    off = RotarySelfAttention(d_model=32, n_heads=4, use_rope=False).eval()
    with torch.no_grad():
        assert torch.equal(off(hidden, mask), off(hidden, mask, position_ids=scrambled))

    on = RotarySelfAttention(d_model=32, n_heads=4, use_rope=True).eval()
    with torch.no_grad():
        assert not torch.allclose(
            on(hidden, mask), on(hidden, mask, position_ids=scrambled), atol=1e-5
        )


def test_the_learned_table_is_the_only_positional_signal_in_a_learned_encoder():
    """
    End to end through `TransformerSequenceEncoder`, so the fix is wired all the
    way up and not only inside one layer. Zero the position table and a learned
    encoder must become order-blind; anything left is a second position
    encoding.
    """
    from smallAntibodyGen.models.mlm import TransformerSequenceEncoder

    torch.manual_seed(0)
    encoder = TransformerSequenceEncoder(_config(norm_type="rmsnorm")).eval()
    assert encoder.position_embedding is not None, "fixture must exercise the learned path"
    assert encoder.is_legacy_block is False, "fixture must exercise the modern block"
    with torch.no_grad():
        encoder.position_embedding.embedding.weight.zero_()
        ids = torch.tensor([[4, 9, 12, 7, 5, 20]])
        base, _ = encoder(ids)
        permuted, _ = encoder(ids[:, _PERMUTATION])
    assert torch.allclose(permuted, base[:, _PERMUTATION, :], atol=1e-5)


def test_cross_attention_head_count_follows_its_own_knob():
    """
    `cross_attention_n_heads` validated and was then discarded for
    `config.n_heads`. It exists precisely so the encoder's head shape can move
    without dragging the antigen fusion along, so an arm that set it ran the
    default fusion under a changed label.
    """
    from smallAntibodyGen.models.mlm import AntibodyAntigenCrossAttention

    config = _config(d_model=32, n_heads=4, cross_attention_n_heads=2)
    # The fixture must actually differ from the fallback, or it proves nothing.
    assert config.cross_attention_n_heads != config.n_heads
    model = AntibodyAntigenCrossAttention(config)
    assert model.antibody_to_antigen.num_heads == 2
    assert model.antigen_to_antibody.num_heads == 2
    # The ENCODER is untouched: the two head counts resolve independently.
    assert model.antibody_encoder.encoder.layers[0].self_attn.num_heads == 4


def test_cross_attention_head_count_defaults_to_n_heads():
    """`None` inherits `n_heads`, which is what every canonical config sets, so
    the v5 chain builds exactly what it built before the knob was honored."""
    from smallAntibodyGen.models.mlm import AntibodyAntigenCrossAttention

    config = _modern(d_model=32, n_heads=4)
    assert config.cross_attention_n_heads is None
    model = AntibodyAntigenCrossAttention(config)
    assert model.antibody_to_antigen.num_heads == 4
    assert model.antigen_to_antibody.num_heads == 4


def test_the_cross_attention_head_count_leaves_no_trace_in_the_state_dict():
    """
    Why the dormant knob was silent, pinned as a fact rather than a footnote.

    `nn.MultiheadAttention` packs all heads into one `in_proj_weight`, so the
    head count changes no parameter name and no shape: a checkpoint trained at
    eight heads loads cleanly into a two-head model and computes something else.
    Only the architecture fingerprint stands between that and a silent result --
    which is why `cross_attention_n_heads` is a fingerprint key.
    """
    from smallAntibodyGen.models.mlm import AntibodyAntigenCrossAttention

    def signature(**overrides):
        model = AntibodyAntigenCrossAttention(_config(d_model=32, n_heads=4, **overrides))
        return sorted((k, tuple(v.shape)) for k, v in model.state_dict().items())

    assert signature() == signature(cross_attention_n_heads=2)
