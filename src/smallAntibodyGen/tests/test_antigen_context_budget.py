"""
Contract tests for the antigen stream's token budget (AB-07).

Before this change `antigen_max_length` was read nowhere in `models/mlm.py`
except its own validator: the scratch antigen encoder was built from the same
`MLMConfig` as the antibody encoder and inherited `max_length`, and all three
tokenization call sites clamped the antigen to `max_length` on the scratch path.
The knob looked live and was inert, while 87.31% of training rows had their
antigen truncated.

Two obligations are tested here and they pull in opposite directions:

1. the knob must now actually work, at BOTH the model and the tokenization
   layer, with the positional table sized to match; and
2. the DEFAULT must reproduce the old behavior exactly, so every checked-in
   config, every existing checkpoint, and every recorded metric stays valid.
"""
from __future__ import annotations

import torch

from smallAntibodyGen.antigen_tokenization import resolve_antigen_encode_max_length
from smallAntibodyGen.models.mlm import AntibodyAntigenCrossAttention, MLMConfig


def _config(**overrides) -> MLMConfig:
    base = dict(
        vocab_size=35,
        pad_token_id=0,
        max_length=64,
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=64,
    )
    base.update(overrides)
    return MLMConfig(**base)


# --------------------------------------------------------------------------- #
# The default must be the old behavior, exactly
# --------------------------------------------------------------------------- #
def test_default_antigen_budget_inherits_the_antibody_max_length():
    """
    The `None` sentinel reproduces pre-AB-07 behavior. This is the compatibility
    guarantee that lets the fix ship without invalidating existing checkpoints:
    an unmodified config builds the identical parameter set.
    """
    config = _config()
    assert config.antigen_max_length is None
    assert config.effective_antigen_max_length == config.max_length

    model = AntibodyAntigenCrossAttention(config)
    assert model.antigen_encoder.max_length == config.max_length
    assert model.antibody_encoder.max_length == config.max_length


def test_default_build_has_the_same_parameter_shapes_as_a_symmetric_build():
    """
    An explicit `antigen_max_length == max_length` and the `None` default must
    produce byte-identical parameter SHAPES. If they diverged, the sentinel would
    be a second code path rather than a spelling of the same one, and a
    checkpoint saved under one could fail to load under the other.
    """
    implicit = AntibodyAntigenCrossAttention(_config())
    explicit = AntibodyAntigenCrossAttention(_config(antigen_max_length=64))

    left = {k: tuple(v.shape) for k, v in implicit.state_dict().items()}
    right = {k: tuple(v.shape) for k, v in explicit.state_dict().items()}
    assert left == right


def test_a_default_checkpoint_still_loads_strictly():
    """
    CLAUDE.md's rule is that checkpoint loading is strict and a mismatch means
    retrain, never silent coercion. So the default path must keep loading a
    checkpoint saved before this change -- which, given the shapes above, means
    a round trip through `state_dict` with `strict=True`.
    """
    saved = AntibodyAntigenCrossAttention(_config()).state_dict()
    fresh = AntibodyAntigenCrossAttention(_config())
    fresh.load_state_dict(saved, strict=True)


# --------------------------------------------------------------------------- #
# The knob must now be real
# --------------------------------------------------------------------------- #
def test_antigen_encoder_gets_its_own_budget_when_configured():
    """The whole point of AB-07: the two streams size independently."""
    model = AntibodyAntigenCrossAttention(_config(max_length=64, antigen_max_length=256))
    assert model.antibody_encoder.max_length == 64
    assert model.antigen_encoder.max_length == 256


def test_positional_capacity_matches_the_token_budget():
    """
    The positional table and the length guard must be sized from ONE number.
    Splitting them is how a model accepts a sequence it has no position for --
    it would either index out of range or silently reuse a position.

    `LearnedPositionalEmbedding` allocates `n + 1` rows because index 0 is
    reserved for padding, so capacity for `n` real tokens means `n + 1` rows.
    """
    model = AntibodyAntigenCrossAttention(_config(max_length=64, antigen_max_length=256))
    table = model.antigen_encoder.position_embedding.embedding
    assert table.num_embeddings == 256 + 1
    assert model.antigen_encoder.position_embedding.max_length == 256


def test_a_long_antigen_runs_end_to_end_and_one_token_over_is_refused():
    """
    Exactly at the budget must work; one token past it must fail loudly rather
    than truncate silently. The boundary is the assertion -- an off-by-one here
    is the difference between a guard and a decoration.
    """
    config = _config(max_length=32, antigen_max_length=128)
    model = AntibodyAntigenCrossAttention(config)
    model.eval()

    antibody = torch.randint(3, 30, (2, 32))
    at_budget = torch.randint(3, 30, (2, 128))
    with torch.no_grad():
        logits, compat = model(antibody, torch.ones_like(antibody), at_budget,
                               torch.ones_like(at_budget))
    assert logits.shape == (2, 32, config.vocab_size)
    assert compat.shape[0] == 2

    over_budget = torch.randint(3, 30, (2, 129))
    try:
        model(antibody, torch.ones_like(antibody), over_budget,
              torch.ones_like(over_budget))
    except ValueError as exc:
        assert "129" in str(exc) and "128" in str(exc)
    else:
        raise AssertionError("an antigen past the budget must be refused")


def test_the_validator_ceiling_covers_the_measured_corpus():
    """
    The census measured antigens up to 2042 tokens. A ceiling below that cannot
    express a setting that covers the data, which is what the old 1024 cap did.
    """
    _config(antigen_max_length=2048).validate()
    _config(antigen_max_length=8192).validate()
    for bad in (0, -1, 8193):
        try:
            _config(antigen_max_length=bad).validate()
        except ValueError as exc:
            assert "antigen_max_length must be None or in" in str(exc)
        else:
            raise AssertionError(f"antigen_max_length={bad} must be rejected")


# --------------------------------------------------------------------------- #
# Train/inference parity -- the expensive bug class
# --------------------------------------------------------------------------- #
def test_every_call_site_resolves_the_budget_through_one_function():
    """
    The rule lived in three hand-copied expressions: the training collator, the
    infiller, and the compatibility scorer. Three copies is three chances for the
    training encoder and the generation encoder to disagree about how many
    antigen tokens the model sees -- silent, and only visible as degraded
    generation.

    This asserts the resolver is the single definition and that its contract is
    the one the model's own accessor uses, so the tokenized input and the
    positional table cannot be sized by different rules.
    """
    for antigen_budget, antibody_budget in ((None, 192), (512, 192), (2048, 288)):
        expected = resolve_antigen_encode_max_length(antigen_budget, antibody_budget)
        config = _config(max_length=antibody_budget, antigen_max_length=antigen_budget)
        assert config.effective_antigen_max_length == expected

        model = AntibodyAntigenCrossAttention(config)
        # The model's antigen encoder is sized by the SAME number the collator
        # will use to truncate, which is the parity that matters.
        assert model.antigen_encoder.max_length == expected


def test_collator_and_infiller_agree_on_the_budget(tokenizer):
    """
    Parity proved at the two real call sites rather than at the helper, because
    the helper agreeing with itself proves nothing. Both objects are constructed
    the way production constructs them and asked what budget they resolved.
    """
    from smallAntibodyGen.data.MLMCollator import AntibodyAntigenCollator
    from smallAntibodyGen.infill.hcdr3 import FixedLengthHCDR3Infiller

    for antigen_budget in (None, 256):
        collator = AntibodyAntigenCollator(
            tokenizer=tokenizer,
            max_length=64,
            antigen_max_length=antigen_budget,
        )
        expected = resolve_antigen_encode_max_length(antigen_budget, 64)
        assert collator._antigen_encode_max_length == expected

        model = AntibodyAntigenCrossAttention(
            _config(max_length=64, antigen_max_length=antigen_budget)
        )
        infiller = FixedLengthHCDR3Infiller(
            model=model,
            tokenizer=tokenizer,
            max_length=64,
            antigen_max_length=antigen_budget,
        )
        assert infiller._antigen_encode_max_length == expected
        # And the model the infiller drives is sized to the same number.
        assert model.antigen_encoder.max_length == expected


def test_scratch_path_no_longer_silently_clamps_the_antigen(tokenizer):
    """
    The specific defect, stated as a test. On the scratch path the collator used
    to discard `antigen_max_length` and use the antibody `max_length`, so an
    antigen budget larger than the antibody's had no effect at all.
    """
    from smallAntibodyGen.data.MLMCollator import AntibodyAntigenCollator

    collator = AntibodyAntigenCollator(
        tokenizer=tokenizer,
        max_length=64,
        antigen_max_length=512,
        antigen_encoder_type="scratch",
    )
    assert collator._antigen_encode_max_length == 512, (
        "the scratch path is clamping the antigen to the antibody max_length again"
    )


# --------------------------------------------------------------------------- #
# Warm-start across a changed antigen budget
# --------------------------------------------------------------------------- #
def test_warm_start_drops_only_the_shape_mismatched_tensor_and_says_so(
    project_root, tmp_path, capsys
):
    """
    `initialize_antigen_refine_from_checkpoint` clones the antibody encoder into
    BOTH branches. Once the antigen stream has its own budget, its positional
    table is a different size than the antibody table it would be seeded from, and
    `load_state_dict(strict=False)` raises on a shape mismatch rather than
    ignoring it -- so the stage-3 warm start died outright at 288/1024.

    The contract: drop exactly the mismatched tensor, warm-start everything else,
    and NAME what was dropped. A partially warm-started encoder nobody was told
    about is indistinguishable from a fully warm-started one in the loss curve.
    """
    import sys as _sys
    import importlib.util
    from dataclasses import asdict

    script = project_root.parents[1] / "scripts" / "mlm_train.py"
    spec = importlib.util.spec_from_file_location("mlm_train_ws", script)
    mt = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    _sys.modules[spec.name] = mt
    spec.loader.exec_module(mt)

    # A stage-2-shaped checkpoint: antibody-only, antigen table sized like the
    # antibody one (64), against a dual-stream model whose antigen budget is 256.
    donor = AntibodyAntigenCrossAttention(_config(max_length=64))
    ckpt = tmp_path / "stage2.pt"
    torch.save({"model_state_dict": donor.state_dict(), "epoch": 0}, ckpt)

    target = AntibodyAntigenCrossAttention(_config(max_length=64, antigen_max_length=256))
    mt.initialize_antigen_refine_from_checkpoint(ckpt, target)

    out = capsys.readouterr().out
    assert "NOT warm-started" in out
    assert "antigen_encoder.position_embedding.embedding.weight" in out
    assert "(65, 32)" in out and "(257, 32)" in out

    # The mismatched table stayed at fresh init...
    assert target.antigen_encoder.position_embedding.embedding.weight.shape == (257, 32)
    # ...while shape-compatible antigen weights WERE warm-started, which is the
    # half of the contract that a blanket "skip the antigen encoder" would lose.
    assert torch.equal(
        target.antigen_encoder.token_embedding.weight,
        donor.antigen_encoder.token_embedding.weight,
    )
    assert torch.equal(
        target.antibody_encoder.token_embedding.weight,
        donor.antibody_encoder.token_embedding.weight,
    )


def test_warm_start_with_matching_budgets_drops_nothing(project_root, tmp_path, capsys):
    """The filter must be inert when the budgets agree -- otherwise it would be a
    behavior change for every existing config rather than a new capability."""
    import sys as _sys
    import importlib.util

    script = project_root.parents[1] / "scripts" / "mlm_train.py"
    spec = importlib.util.spec_from_file_location("mlm_train_ws2", script)
    mt = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    _sys.modules[spec.name] = mt
    spec.loader.exec_module(mt)

    donor = AntibodyAntigenCrossAttention(_config(max_length=64))
    ckpt = tmp_path / "stage2_match.pt"
    torch.save({"model_state_dict": donor.state_dict(), "epoch": 0}, ckpt)

    target = AntibodyAntigenCrossAttention(_config(max_length=64))
    mt.initialize_antigen_refine_from_checkpoint(ckpt, target)

    assert "NOT warm-started" not in capsys.readouterr().out
    assert torch.equal(
        target.antigen_encoder.position_embedding.embedding.weight,
        donor.antigen_encoder.position_embedding.embedding.weight,
    )
