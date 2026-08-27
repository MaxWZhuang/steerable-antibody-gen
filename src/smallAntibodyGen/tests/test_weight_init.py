"""Tests for the model's weight-initialization scheme.

Why this file exists
--------------------
Before `initializer_range` / `scale_residual_init`, the models relied entirely on
PyTorch's per-module defaults. `nn.Embedding` defaults to `N(0, 1)`, and with
`tie_weights=True` that unit-variance table IS the output projection -- so a
freshly built d_model=256 model produced logits with std ~29 and a masked-LM loss
of ~136 nats against an ideal `ln(vocab) ~ 3.6`.

That is not a cosmetic issue. It means the opening phase of every from-scratch run
is spent shrinking the embedding norm rather than learning residue statistics, and
because `grad_clip_norm` is 1.0 in every config, the gradients that would do the
shrinking are clipped. `test_initial_masked_loss_is_near_uniform` is the test that
would have caught it.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from smallAntibodyGen.models.mlm import (
    AntibodyAntigenCrossAttention,
    AntibodyMLM,
    MLMConfig,
)


def _config(tokenizer, **overrides) -> MLMConfig:
    base = dict(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_id,
        max_length=192,
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        dropout=0.0,
    )
    base.update(overrides)
    return MLMConfig(**base)


# --------------------------------------------------------------------------- #
# The property that actually matters: a fresh model starts near-uniform.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("norm_first", [True, False])
def test_initial_masked_loss_is_near_uniform(tokenizer, norm_first: bool):
    """A freshly initialized MLM must start close to `ln(vocab_size)`.

    The regression guarded here produced ~136 nats (pre-LN) / ~96 (post-LN). The
    bound is deliberately loose -- this is catching an order-of-magnitude defect,
    not pinning an exact number across PyTorch versions.
    """
    torch.manual_seed(0)
    model = AntibodyMLM(_config(tokenizer, norm_first=norm_first)).eval()

    ids = torch.randint(6, tokenizer.vocab_size, (8, 48))
    labels = ids.clone()
    corrupt = torch.rand(ids.shape) < 0.15
    corrupt[:, 0] = True  # guarantee at least one target per batch
    masked = ids.clone()
    masked[corrupt] = tokenizer.mask_id
    labels[~corrupt] = -100

    with torch.no_grad():
        logits = model(masked, torch.ones_like(ids))
    loss = F.cross_entropy(
        logits.reshape(-1, tokenizer.vocab_size), labels.reshape(-1), ignore_index=-100
    )

    uniform = math.log(tokenizer.vocab_size)
    assert loss.item() < 3.0 * uniform, (
        f"initial masked loss {loss.item():.2f} is far above ln(vocab)={uniform:.2f}; "
        "the embedding/head init scale has regressed"
    )


def test_embedding_scale_matches_initializer_range(tokenizer):
    torch.manual_seed(0)
    config = _config(tokenizer, initializer_range=0.02)
    model = AntibodyMLM(config)
    for table in (
        model.sequence_encoder.token_embedding.weight,
        model.sequence_encoder.position_embedding.embedding.weight,
    ):
        assert table.std().item() == pytest.approx(0.02, rel=0.15)


def test_padding_rows_are_zeroed_after_init(tokenizer):
    """`normal_` overwrites the zero pad row PyTorch installs at construction.

    Pad positions are masked out of attention, but their embedding still enters
    the residual stream at that position and the LM head reads every position, so
    a non-zero pad row is real signal leaking into an ignored slot.
    """
    torch.manual_seed(0)
    model = AntibodyMLM(_config(tokenizer))
    encoder = model.sequence_encoder
    assert torch.count_nonzero(encoder.token_embedding.weight[tokenizer.pad_id]) == 0
    assert torch.count_nonzero(encoder.position_embedding.embedding.weight[0]) == 0


def test_residual_output_projections_are_depth_scaled(tokenizer):
    """Residual WRITES (attn.out_proj, ffn.linear2) scale by 1/sqrt(2 * n_layers).

    In a pre-LN stack every layer adds into an un-normalized residual stream, so
    without this the stream's variance grows with depth and each later layer's
    contribution becomes a smaller fraction of what the head reads.
    """
    torch.manual_seed(0)
    config = _config(tokenizer, n_layers=6, initializer_range=0.02)
    model = AntibodyMLM(config)
    expected = 0.02 / math.sqrt(2.0 * 6)

    for layer in model.sequence_encoder.encoder.layers:
        assert layer.self_attn.out_proj.weight.std().item() == pytest.approx(
            expected, rel=0.2
        )
        assert layer.linear2.weight.std().item() == pytest.approx(expected, rel=0.2)
        # The branch INPUTS are not scaled -- only the writes into the stream.
        assert layer.linear1.weight.std().item() == pytest.approx(0.02, rel=0.15)


def test_depth_scaling_can_be_disabled(tokenizer):
    torch.manual_seed(0)
    model = AntibodyMLM(_config(tokenizer, scale_residual_init=False))
    layer = model.sequence_encoder.encoder.layers[0]
    assert layer.linear2.weight.std().item() == pytest.approx(0.02, rel=0.15)


def test_dual_stream_fusion_and_heads_are_initialized(tokenizer):
    """The fusion block and heads are owned by the dual-stream class, not by the
    encoders, so they need their own init pass or they keep PyTorch's defaults."""
    torch.manual_seed(0)
    model = AntibodyAntigenCrossAttention(_config(tokenizer))
    for weight in (
        model.antibody_to_antigen.in_proj_weight,
        model.antigen_to_antibody.in_proj_weight,
        model.fusion_mlp[0].weight,
        model.compatibility_head.weight,
    ):
        assert weight.std().item() == pytest.approx(0.02, rel=0.3)
    assert torch.count_nonzero(model.compatibility_head.bias) == 0


def test_invalid_init_config_fails_loud(tokenizer):
    with pytest.raises(ValueError, match="initializer_range"):
        _config(tokenizer, initializer_range=0.0).validate()
    with pytest.raises(ValueError, match="scale_residual_init"):
        _config(tokenizer, scale_residual_init="yes").validate()
