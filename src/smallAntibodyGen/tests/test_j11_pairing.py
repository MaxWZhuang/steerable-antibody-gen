"""
J11 pairing: the two SwiGLU-width arms must start from identical weights.

J11 asks whether ~33.6% more capacity buys a practical HCDR3 benefit, and the
promotion rule turns on 1.0 absolute percentage point with all three seeds
agreeing. That bar is only meaningful if the seeds are PAIRED.

They are not, by default. The arms differ only in the SwiGLU projections, but the
wider arm's FFN consumes more initialization RNG, so every parameter constructed
after it lands on a different value -- attention, norms, embeddings, the head.
Three seeds of a comparison whose arms start from different weights is not three
paired observations; it is six unrelated runs, and a 1-point difference is well
inside what that produces.
"""
from __future__ import annotations

import pytest
import torch

from smallAntibodyGen.experiments import init_parity as ip
from smallAntibodyGen.models.mlm import AntibodyMLM, MLMConfig

MODERN = dict(
    position_encoding="rope",
    norm_type="rmsnorm",
    ffn_type="swiglu",
    attention_bias=False,
    ffn_bias=False,
    norm_first=True,
)


def _arm(width: int, **overrides):
    params = dict(
        vocab_size=35, pad_token_id=0, max_length=32,
        d_model=32, n_heads=4, n_layers=2, d_ff=64, dropout=0.1,
    )
    params.update(MODERN)
    params["swiglu_hidden_dim"] = width
    params.update(overrides)
    config = MLMConfig(**params)
    config.validate()
    return config, AntibodyMLM(config)


def test_naive_seeding_leaves_the_arms_unpaired():
    """
    The premise. Same seed, and most same-shape parameters still differ, because
    the wider FFN shifts every draw after it.
    """
    torch.manual_seed(7)
    _, narrow = _arm(24)
    torch.manual_seed(7)
    _, wide = _arm(48)

    report = ip.compare_parameter_sets(narrow, wide)
    assert report["differing"], "expected unpaired arms before the fix"
    with pytest.raises(AssertionError, match="not paired"):
        ip.assert_arms_are_paired(narrow, wide)


def test_pairing_makes_every_same_shape_parameter_identical():
    """After the per-module pass the arms differ ONLY where their shapes differ."""
    narrow_cfg, narrow = _arm(24)
    wide_cfg, wide = _arm(48)
    ip.reinitialize_by_module_name(narrow, narrow_cfg, seed=42)
    ip.reinitialize_by_module_name(wide, wide_cfg, seed=42)

    report = ip.assert_arms_are_paired(narrow, wide)
    assert report["differing"] == []
    assert report["identical"], "nothing was held fixed"
    assert report["shape_mismatch"], "the SwiGLU projections should differ in shape"


def test_only_the_swiglu_projections_differ_in_shape():
    """
    The axis under test, named. If anything OTHER than the feed-forward differed
    in shape, the comparison would be measuring more than width.
    """
    narrow_cfg, narrow = _arm(24)
    wide_cfg, wide = _arm(48)
    ip.reinitialize_by_module_name(narrow, narrow_cfg, seed=42)
    ip.reinitialize_by_module_name(wide, wide_cfg, seed=42)

    report = ip.compare_parameter_sets(narrow, wide)
    assert all(".ffn." in name for name in report["shape_mismatch"])
    assert all(
        any(proj in name for proj in ("gate_proj", "up_proj", "down_proj"))
        for name in report["shape_mismatch"]
    )


def test_pairing_is_independent_of_construction_order():
    """
    Seeds are derived per module NAME, so an arm built under a completely
    different RNG position still lands on the same shared weights. That is what
    makes the pairing robust rather than incidental.
    """
    narrow_cfg, narrow = _arm(24)
    torch.manual_seed(999_999)  # deliberately different construction stream
    wide_cfg, wide = _arm(48)
    ip.reinitialize_by_module_name(narrow, narrow_cfg, seed=42)
    ip.reinitialize_by_module_name(wide, wide_cfg, seed=42)
    ip.assert_arms_are_paired(narrow, wide)


def test_different_seeds_give_different_pairs():
    """
    The three seeds must actually be three different starting points, or the
    "all three seeds favor 1024" clause is one observation counted thrice.
    """
    cfg_a, a = _arm(24)
    cfg_b, b = _arm(24)
    ip.reinitialize_by_module_name(a, cfg_a, seed=42)
    ip.reinitialize_by_module_name(b, cfg_b, seed=31415)
    report = ip.compare_parameter_sets(a, b)
    assert report["differing"], "two seeds produced identical weights"


def test_residual_depth_scaling_survives_reinitialization():
    """
    Scaling is a post-init multiply, so re-initializing erases it. Forgetting to
    re-apply would silently un-damp the residual writes and change training
    dynamics for BOTH arms -- a change nobody asked for, invisible in the diff.
    """
    scaled_cfg, scaled = _arm(24, scale_residual_init=True)
    unscaled_cfg, unscaled = _arm(24, scale_residual_init=False)
    ip.reinitialize_by_module_name(scaled, scaled_cfg, seed=42)
    ip.reinitialize_by_module_name(unscaled, unscaled_cfg, seed=42)

    scaled_std = scaled.sequence_encoder.encoder.layers[0].ffn.down_proj.weight.std()
    unscaled_std = unscaled.sequence_encoder.encoder.layers[0].ffn.down_proj.weight.std()
    assert float(scaled_std) < float(unscaled_std) * 0.8


def test_reinitialization_restores_the_global_rng():
    """The pass must not shift the data-order stream; that is what
    `reset_training_rng` is for, deliberately and separately."""
    cfg, model = _arm(24)
    torch.manual_seed(123)
    before = torch.get_rng_state()
    ip.reinitialize_by_module_name(model, cfg, seed=42)
    assert torch.equal(before, torch.get_rng_state())


def test_reset_training_rng_puts_both_arms_on_one_stream():
    """
    Construction consumes different amounts of RNG per arm, so without an
    explicit reset the arms would see different data order and dropout masks --
    reintroducing in the training stream exactly the confound the init pass just
    removed from the parameters.
    """
    narrow_cfg, narrow = _arm(24)
    ip.reset_training_rng(42)
    first = torch.randn(4)

    wide_cfg, wide = _arm(48)  # consumes a different amount of RNG
    ip.reset_training_rng(42)
    second = torch.randn(4)

    assert torch.equal(first, second)


def test_the_canonical_arms_pair_at_full_scale():
    """
    The real thing: 680 against 1024 at the canonical shape. 40 same-shape
    parameters held identical, 18 differing only in the feed-forward width.
    """
    def build(width):
        params = dict(
            vocab_size=35, pad_token_id=0, max_length=288,
            d_model=256, n_heads=8, n_layers=6, d_ff=1024, dropout=0.1,
        )
        params.update(MODERN)
        params["swiglu_hidden_dim"] = width
        config = MLMConfig(**params)
        config.validate()
        return config, AntibodyMLM(config)

    cfg_680, arm_680 = build(680)
    cfg_1024, arm_1024 = build(1024)
    ip.reinitialize_by_module_name(arm_680, cfg_680, seed=42)
    ip.reinitialize_by_module_name(arm_1024, cfg_1024, seed=42)

    report = ip.assert_arms_are_paired(arm_680, arm_1024)
    assert len(report["identical"]) == 40
    assert len(report["shape_mismatch"]) == 18
    assert report["differing"] == []
