"""Tests for the `fusion_gate` knob — zero-init gating of the fusion sublayer.

WHY THIS EXISTS
---------------
Stage 3 warm-starts the antibody encoder and LM head from stage 2 and then
inserts `AntibodyAntigenCrossAttention.fuse` between them. That insertion is not
a no-op at step zero, so the stage-2 representation is perturbed before a single
gradient arrives.

Gating only the cross-attention branch does NOT fix it. In pre-LN the fused
stream is built as::

    antibody_hidden = fusion_out_norm(antibody_hidden + antibody_ctx)

so zeroing `antibody_ctx` still leaves `fusion_out_norm(antibody_hidden)` — a
norm that was never in stage 2's computation. The gate must therefore wrap the
WHOLE inserted transformation::

    candidate = fusion_out_norm(x + cross_attention)
    x = x + tanh(alpha) * (candidate - x)      # alpha = 0 -> exact identity

The load-bearing tests here are
`test_gate_makes_fusion_an_exact_identity_at_init` and
`test_stage3_init_reproduces_stage2_logits_exactly`. The first proves the
sublayer is an identity; the second proves that identity is what stage 3 needs.
`test_ungated_fusion_is_not_an_identity` is the fault injection: it fails if the
gate is removed, so the suite cannot stay green with the mechanism deleted.

The gate CHANGES THE PARAMETER SET, so a cross-mode load fails `strict=True` on
its own. The init-compat check still names it, because a named error beats a
list of unexpected keys.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from smallAntibodyGen.models.mlm import (
    AntibodyAntigenCrossAttention,
    AntibodyMLM,
    MLMConfig,
)


def load_mlm_train_module(project_root: Path):
    script_path = project_root.parents[1] / "scripts" / "mlm_train.py"
    spec = importlib.util.spec_from_file_location("mlm_train", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _model_config(**overrides) -> MLMConfig:
    base = dict(
        vocab_size=30,
        pad_token_id=0,
        max_length=32,
        d_model=16,
        n_heads=2,
        n_layers=2,
        d_ff=32,
        dropout=0.0,
    )
    base.update(overrides)
    return MLMConfig(**base)


def _streams(batch: int = 2, ab_len: int = 12, ag_len: int = 10):
    """Deterministic antibody/antigen hidden states and masks."""
    torch.manual_seed(0)
    antibody_hidden = torch.randn(batch, ab_len, 16)
    antigen_hidden = torch.randn(batch, ag_len, 16)
    antibody_mask = torch.ones(batch, ab_len, dtype=torch.long)
    antigen_mask = torch.ones(batch, ag_len, dtype=torch.long)
    # A real pad tail on each stream, so the key_padding_mask path is exercised.
    antibody_mask[:, -3:] = 0
    antigen_mask[:, -2:] = 0
    return antibody_hidden, antibody_mask, antigen_hidden, antigen_mask


# --------------------------------------------------------------------------- #
# Default is OFF on both config surfaces: existing v4 antigen checkpoints keep
# loading, and no shipped run changes behaviour by upgrading the code.
# --------------------------------------------------------------------------- #
def test_fusion_gate_defaults_off_on_model_config():
    assert _model_config().fusion_gate is False


def test_fusion_gate_defaults_off_on_train_config(tmp_path: Path, project_root: Path):
    mlm_train = load_mlm_train_module(project_root)
    data_path = tmp_path / "tiny.jsonl.gz"
    data_path.write_text("", encoding="utf-8")

    cfg = mlm_train.parse_args(["--data-path", str(data_path)])

    assert cfg.fusion_gate is False


def test_fusion_gate_is_rejected_on_an_antibody_only_stage(
    tmp_path: Path, project_root: Path
):
    """There is no fusion sublayer to gate outside the antigen stages."""
    mlm_train = load_mlm_train_module(project_root)
    data_path = tmp_path / "tiny.jsonl.gz"
    data_path.write_text("", encoding="utf-8")

    cfg = mlm_train.parse_args(["--data-path", str(data_path)])
    with pytest.raises(ValueError, match="fusion_gate is only meaningful"):
        dataclasses.replace(
            cfg, fusion_gate=True, training_stage="paired_refine"
        ).validate()


def test_fusion_gate_rejects_a_non_bool():
    with pytest.raises(ValueError, match="fusion_gate must be a bool"):
        _model_config(fusion_gate="yes").validate()


# --------------------------------------------------------------------------- #
# Parameter set
# --------------------------------------------------------------------------- #
def test_gate_off_registers_no_gate_parameters():
    model = AntibodyAntigenCrossAttention(_model_config(fusion_gate=False))
    assert not [k for k in model.state_dict() if "fusion_gate" in k]


def test_gate_on_registers_one_scalar_per_stream():
    model = AntibodyAntigenCrossAttention(_model_config(fusion_gate=True))
    keys = sorted(k for k in model.state_dict() if "fusion_gate" in k)
    assert keys == ["fusion_gate_antibody", "fusion_gate_antigen"]
    for key in keys:
        param = model.state_dict()[key]
        assert param.shape == torch.Size([])
        assert float(param) == 0.0


def test_gate_parameters_are_trainable():
    model = AntibodyAntigenCrossAttention(_model_config(fusion_gate=True))
    assert model.fusion_gate_antibody.requires_grad
    assert model.fusion_gate_antigen.requires_grad


def test_gate_draws_no_extra_init_rng():
    """A scalar of zeros consumes no RNG, so every shared weight is untouched."""
    torch.manual_seed(1234)
    off = AntibodyAntigenCrossAttention(_model_config(fusion_gate=False))
    torch.manual_seed(1234)
    on = AntibodyAntigenCrossAttention(_model_config(fusion_gate=True))

    off_state = off.state_dict()
    for key, value in on.state_dict().items():
        if "fusion_gate" in key:
            continue
        assert torch.equal(value, off_state[key]), f"{key} moved when the gate was enabled"


# --------------------------------------------------------------------------- #
# The identity contract, in BOTH norm modes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("norm_first", [True, False])
def test_gate_makes_fusion_an_exact_identity_at_init(norm_first: bool):
    model = AntibodyAntigenCrossAttention(
        _model_config(fusion_gate=True, norm_first=norm_first)
    )
    model.eval()
    ab, ab_mask, ag, ag_mask = _streams()

    with torch.no_grad():
        fused_ab, fused_ag = model.fuse(ab, ab_mask, ag, ag_mask)

    # Byte-exact, not allclose: "approximately the inherited computation" is the
    # bug this knob exists to remove.
    assert torch.equal(fused_ab, ab)
    assert torch.equal(fused_ag, ag)


@pytest.mark.parametrize("norm_first", [True, False])
def test_ungated_fusion_is_not_an_identity(norm_first: bool):
    """Fault injection: delete the gate and this suite must go red."""
    model = AntibodyAntigenCrossAttention(
        _model_config(fusion_gate=False, norm_first=norm_first)
    )
    model.eval()
    ab, ab_mask, ag, ag_mask = _streams()

    with torch.no_grad():
        fused_ab, fused_ag = model.fuse(ab, ab_mask, ag, ag_mask)

    assert not torch.equal(fused_ab, ab)
    assert not torch.equal(fused_ag, ag)


@pytest.mark.parametrize("norm_first", [True, False])
def test_zeroing_only_cross_attention_is_still_not_an_identity(norm_first: bool):
    """The precise reason branch-gating is insufficient: the norm survives it.

    This is the claim from the research brief's C1 section, pinned as a test so
    nobody re-proposes the cheaper fix.
    """
    model = AntibodyAntigenCrossAttention(
        _model_config(fusion_gate=False, norm_first=norm_first)
    )
    model.eval()
    # Force both cross-attention branches to emit exactly zero.
    for attn in (model.antibody_to_antigen, model.antigen_to_antibody):
        torch.nn.init.zeros_(attn.out_proj.weight)
        torch.nn.init.zeros_(attn.out_proj.bias)

    ab, ab_mask, ag, ag_mask = _streams()
    with torch.no_grad():
        fused_ab, fused_ag = model.fuse(ab, ab_mask, ag, ag_mask)

    # Zero branch, and the stream STILL moved -- because a norm remains on it.
    assert not torch.equal(fused_ab, ab)
    assert not torch.equal(fused_ag, ag)


def test_open_gate_changes_the_output():
    """The gate must be a real knob, not a permanent off switch."""
    model = AntibodyAntigenCrossAttention(_model_config(fusion_gate=True))
    model.eval()
    ab, ab_mask, ag, ag_mask = _streams()

    with torch.no_grad():
        closed_ab, closed_ag = model.fuse(ab, ab_mask, ag, ag_mask)
        model.fusion_gate_antibody.fill_(1.0)
        model.fusion_gate_antigen.fill_(1.0)
        open_ab, open_ag = model.fuse(ab, ab_mask, ag, ag_mask)

    assert torch.equal(closed_ab, ab)
    assert not torch.equal(open_ab, closed_ab)
    assert not torch.equal(open_ag, closed_ag)


def test_gate_receives_gradient_so_it_can_open():
    model = AntibodyAntigenCrossAttention(_model_config(fusion_gate=True))
    ab, ab_mask, ag, ag_mask = _streams()

    fused_ab, _ = model.fuse(ab, ab_mask, ag, ag_mask)
    fused_ab.sum().backward()

    grad = model.fusion_gate_antibody.grad
    assert grad is not None
    # A differentiable zero would leave the gate permanently shut.
    assert float(grad.abs()) > 0.0


# --------------------------------------------------------------------------- #
# The reason the identity matters: the stage 2 -> stage 3 handoff
# --------------------------------------------------------------------------- #
def test_stage3_init_reproduces_stage2_logits_exactly(project_root: Path):
    """Warm-start parity, end to end, on the real translation helper."""
    mlm_train = load_mlm_train_module(project_root)
    config = _model_config(fusion_gate=True)

    torch.manual_seed(7)
    stage2 = AntibodyMLM(config)
    stage2.eval()

    stage3 = AntibodyAntigenCrossAttention(config)
    translated = mlm_train.build_antigen_refine_init_state_dict(stage2.state_dict())
    missing, unexpected = stage3.load_state_dict(translated, strict=False)
    assert not unexpected
    stage3.eval()

    torch.manual_seed(11)
    antibody_ids = torch.randint(1, 30, (2, 12))
    antibody_mask = torch.ones(2, 12, dtype=torch.long)
    antigen_ids = torch.randint(1, 30, (2, 10))
    antigen_mask = torch.ones(2, 10, dtype=torch.long)

    with torch.no_grad():
        stage2_logits = stage2(antibody_ids, antibody_mask)
        stage3_logits, _ = stage3(
            antibody_ids, antibody_mask, antigen_ids, antigen_mask
        )[:2]

    assert torch.equal(stage3_logits, stage2_logits)


def test_stage3_without_the_gate_does_not_reproduce_stage2_logits(project_root: Path):
    """The same check with the gate off -- the regression this knob removes."""
    mlm_train = load_mlm_train_module(project_root)
    config = _model_config(fusion_gate=False)

    torch.manual_seed(7)
    stage2 = AntibodyMLM(config)
    stage2.eval()

    stage3 = AntibodyAntigenCrossAttention(config)
    stage3.load_state_dict(
        mlm_train.build_antigen_refine_init_state_dict(stage2.state_dict()),
        strict=False,
    )
    stage3.eval()

    torch.manual_seed(11)
    antibody_ids = torch.randint(1, 30, (2, 12))
    antibody_mask = torch.ones(2, 12, dtype=torch.long)
    antigen_ids = torch.randint(1, 30, (2, 10))
    antigen_mask = torch.ones(2, 10, dtype=torch.long)

    with torch.no_grad():
        stage2_logits = stage2(antibody_ids, antibody_mask)
        stage3_logits, _ = stage3(
            antibody_ids, antibody_mask, antigen_ids, antigen_mask
        )[:2]

    assert not torch.equal(stage3_logits, stage2_logits)


# --------------------------------------------------------------------------- #
# Checkpoint safety
# --------------------------------------------------------------------------- #
def test_gate_mismatch_breaks_a_strict_load():
    """Bucket A: the parameter set differs, so strict=True catches it loudly."""
    off = AntibodyAntigenCrossAttention(_model_config(fusion_gate=False))
    on = AntibodyAntigenCrossAttention(_model_config(fusion_gate=True))
    with pytest.raises(RuntimeError, match="fusion_gate"):
        on.load_state_dict(off.state_dict(), strict=True)


def test_fusion_gate_mismatch_is_named_by_init_compat_check(
    tmp_path: Path, project_root: Path
):
    mlm_train = load_mlm_train_module(project_root)
    checkpoint_path = tmp_path / "parent.pt"
    torch.save({"train_config": {"fusion_gate": False}}, checkpoint_path)

    cfg = dataclasses.replace(
        mlm_train.parse_args(["--data-path", str(tmp_path / "tiny.jsonl.gz")]),
        fusion_gate=True,
        training_stage="antigen_real_label_refine",
    )
    with pytest.raises(ValueError, match="fusion_gate"):
        mlm_train.validate_init_checkpoint_compatibility(cfg, checkpoint_path)


def test_antibody_only_parent_may_turn_the_gate_on(tmp_path: Path, project_root: Path):
    """The stage 2 -> stage 3 launch itself.

    REGRESSION: the first version of this check enforced gate equality against
    EVERY parent. Stage 2 is antibody-only and therefore records
    `fusion_gate=False`, so enabling the gate in the stage-3 config made the
    launch fail before model initialization -- exactly the transition the gate
    exists for. The parity tests never caught it because they call the weight
    translation helper directly and skip this validator entirely.
    """
    mlm_train = load_mlm_train_module(project_root)
    checkpoint_path = tmp_path / "paired_parent.pt"
    torch.save(
        {
            "train_config": {"fusion_gate": False, "norm_first": True},
            # A real antibody-only parent: `sequence_encoder.*`, no fusion keys.
            # Detection reads the WEIGHTS, so this works with no fingerprint --
            # which is also the legacy-checkpoint case.
            "model_state_dict": {
                "sequence_encoder.final_norm.weight": torch.ones(4),
                "lm_head.weight": torch.ones(4, 4),
            },
        },
        checkpoint_path,
    )

    cfg = dataclasses.replace(
        mlm_train.parse_args(["--data-path", str(tmp_path / "tiny.jsonl.gz")]),
        fusion_gate=True,
        training_stage="antigen_real_label_refine",
    )
    # Must not raise: an antibody-only parent has no fusion sublayer to disagree.
    mlm_train.validate_init_checkpoint_compatibility(cfg, checkpoint_path)


def test_dual_stream_parent_must_still_match(tmp_path: Path, project_root: Path):
    """The exemption is scoped, not a hole: a trained fusion sublayer still binds."""
    mlm_train = load_mlm_train_module(project_root)
    checkpoint_path = tmp_path / "antigen_parent.pt"
    torch.save(
        {
            "train_config": {"fusion_gate": False, "norm_first": True},
            # A dual-stream parent: it already HAS an ungated fusion sublayer.
            "model_state_dict": {
                "antibody_encoder.final_norm.weight": torch.ones(4),
                "antibody_to_antigen.in_proj_weight": torch.ones(4, 4),
                "fusion_norm_antibody.weight": torch.ones(4),
            },
        },
        checkpoint_path,
    )

    cfg = dataclasses.replace(
        mlm_train.parse_args(["--data-path", str(tmp_path / "tiny.jsonl.gz")]),
        fusion_gate=True,
        training_stage="antigen_hcdr3_infill_refine",
    )
    with pytest.raises(ValueError, match="fusion_gate"):
        mlm_train.validate_init_checkpoint_compatibility(cfg, checkpoint_path)


def test_gates_are_in_the_new_module_lr_group(project_root: Path):
    """REGRESSION: the gates were warm-start-new but got the BASE lr.

    Every other parameter the translation leaves uninitialized is routed through
    `new_module_lr_multiplier`; the two gate scalars were not, so at a multiplier
    of 5 the sublayer trained at 5x while the gate deciding how much of it the
    model reads trained at 1x.
    """
    mlm_train = load_mlm_train_module(project_root)
    assert "fusion_gate_antibody" in mlm_train.NEW_MODULE_LR_PREFIXES
    assert "fusion_gate_antigen" in mlm_train.NEW_MODULE_LR_PREFIXES
    # The names are leaf parameters, so a trailing dot would match nothing.
    model = AntibodyAntigenCrossAttention(_model_config(fusion_gate=True))
    for name in ("fusion_gate_antibody", "fusion_gate_antigen"):
        assert name in dict(model.named_parameters())
        assert name.startswith(mlm_train.NEW_MODULE_LR_PREFIXES)


def test_legacy_checkpoint_without_the_key_is_read_as_gate_off(
    tmp_path: Path, project_root: Path
):
    """Absent means the historical value, never 'unknown'."""
    mlm_train = load_mlm_train_module(project_root)
    checkpoint_path = tmp_path / "legacy.pt"
    torch.save({"train_config": {"d_model": 16}}, checkpoint_path)

    cfg = dataclasses.replace(
        mlm_train.parse_args(["--data-path", str(tmp_path / "tiny.jsonl.gz")]),
        fusion_gate=True,
        training_stage="antigen_real_label_refine",
    )
    with pytest.raises(ValueError, match="fusion_gate"):
        mlm_train.validate_init_checkpoint_compatibility(cfg, checkpoint_path)
