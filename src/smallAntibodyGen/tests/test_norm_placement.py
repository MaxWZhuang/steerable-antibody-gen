"""Tests for the `norm_first` (pre-LN vs post-LN) knob.

The knob is default-off: `norm_first=False` reproduces the historical post-LN
stack exactly, so every pre-existing config and checkpoint is unaffected.

The load-bearing test here is
`test_norm_first_mismatch_is_rejected_by_init_compat_check`. Post-LN and pre-LN
`nn.TransformerEncoderLayer`s have *identical* parameter names and shapes — only
the forward order differs — so `load_state_dict(..., strict=True)` reports
"All keys matched successfully" while the model computes something completely
different. The strict load cannot catch this; the init-compat check must.
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


# --------------------------------------------------------------------------- #
# Default-off discipline.
# --------------------------------------------------------------------------- #
def test_norm_first_defaults_to_post_norm_on_model_config():
    assert MLMConfig(vocab_size=30, pad_token_id=0, max_length=32).norm_first is False


def test_norm_first_defaults_to_post_norm_on_train_config(tmp_path: Path, project_root: Path):
    mlm_train = load_mlm_train_module(project_root)
    data_path = tmp_path / "tiny.jsonl.gz"
    data_path.write_text("", encoding="utf-8")

    cfg = mlm_train.parse_args(["--data-path", str(data_path)])

    assert cfg.norm_first is False


# --------------------------------------------------------------------------- #
# The knob is actually plumbed through to the encoder layers.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("norm_first", [False, True])
def test_norm_first_reaches_every_encoder_layer(norm_first: bool):
    model = AntibodyMLM(_model_config(norm_first=norm_first))

    layers = model.sequence_encoder.encoder.layers
    assert len(layers) == 2
    assert all(layer.norm_first is norm_first for layer in layers)


@pytest.mark.parametrize("norm_first", [False, True])
def test_norm_first_reaches_both_streams_of_the_dual_stream_model(norm_first: bool):
    model = AntibodyAntigenCrossAttention(_model_config(norm_first=norm_first))

    for encoder in (model.antibody_encoder, model.antigen_encoder):
        assert all(layer.norm_first is norm_first for layer in encoder.encoder.layers)


def test_norm_first_changes_the_forward_computation():
    """Pre-LN and post-LN must differ numerically from identical weights.

    Guards against the knob being accepted, stored, and then ignored.
    """
    input_ids = torch.tensor([[1, 5, 6, 7, 2]])

    torch.manual_seed(0)
    post = AntibodyMLM(_model_config(norm_first=False)).eval()
    torch.manual_seed(0)
    pre = AntibodyMLM(_model_config(norm_first=True)).eval()

    # Identical parameters: only the forward order differs.
    missing = pre.load_state_dict(post.state_dict(), strict=True)
    assert not missing.missing_keys and not missing.unexpected_keys

    with torch.no_grad():
        post_logits = post(input_ids)
        pre_logits = pre(input_ids)

    assert not torch.allclose(post_logits, pre_logits, atol=1e-4)


def test_prenorm_registers_terminal_fusion_norms_and_postnorm_does_not():
    """Pre-LN fusion needs a terminal norm; post-LN must gain no new parameters."""
    pre = AntibodyAntigenCrossAttention(_model_config(norm_first=True))
    post = AntibodyAntigenCrossAttention(_model_config(norm_first=False))

    assert hasattr(pre, "fusion_out_norm_antibody")
    assert hasattr(pre, "fusion_out_norm_antigen")
    assert not hasattr(post, "fusion_out_norm_antibody")
    assert not hasattr(post, "fusion_out_norm_antigen")


def test_prenorm_fusion_moves_the_norm_off_the_residual_path():
    """Post-LN fusion is `LayerNorm(x + ctx)`; pre-LN is `norm(x + attn(norm(x)))`.

    A pre-LN encoder feeding a post-LN fusion block would be architecturally
    inconsistent, so `fuse` must honor the same flag as the encoder stacks.
    """
    antibody = torch.randn(1, 4, 16)
    antigen = torch.randn(1, 5, 16)
    ab_mask = torch.ones(1, 4, dtype=torch.long)
    ag_mask = torch.ones(1, 5, dtype=torch.long)

    torch.manual_seed(0)
    pre = AntibodyAntigenCrossAttention(_model_config(norm_first=True)).eval()
    torch.manual_seed(0)
    post = AntibodyAntigenCrossAttention(_model_config(norm_first=False)).eval()

    with torch.no_grad():
        pre_ab, _ = pre.fuse(antibody, ab_mask, antigen, ag_mask)
        post_ab, _ = post.fuse(antibody, ab_mask, antigen, ag_mask)

    assert not torch.allclose(pre_ab, post_ab, atol=1e-4)


def test_dual_stream_checkpoints_do_not_load_across_norm_modes():
    """Defense in depth: the extra terminal norms make the strict load fail too.

    The single-stream model has no such structural difference (see the
    companion test below), which is why the init-compat check is the real gate.
    """
    pre = AntibodyAntigenCrossAttention(_model_config(norm_first=True))
    post = AntibodyAntigenCrossAttention(_model_config(norm_first=False))

    with pytest.raises(RuntimeError):
        pre.load_state_dict(post.state_dict(), strict=True)


# --------------------------------------------------------------------------- #
# The gate: a norm-placement mismatch must be fatal, not silent.
# --------------------------------------------------------------------------- #
def test_prenorm_and_postnorm_checkpoints_are_silently_load_compatible():
    """Pins WHY the init-compat check must cover `norm_first`.

    If this ever starts failing, `strict=True` has begun catching the mismatch
    on its own and the guard below could be revisited.
    """
    torch.manual_seed(0)
    post = AntibodyMLM(_model_config(norm_first=False))
    torch.manual_seed(0)
    pre = AntibodyMLM(_model_config(norm_first=True))

    result = pre.load_state_dict(post.state_dict(), strict=True)

    assert not result.missing_keys
    assert not result.unexpected_keys


def test_norm_first_mismatch_is_rejected_by_init_compat_check(
    tmp_path: Path, project_root: Path
):
    mlm_train = load_mlm_train_module(project_root)
    data_path = tmp_path / "tiny.jsonl.gz"
    data_path.write_text("", encoding="utf-8")

    ckpt_path = tmp_path / "post_norm_best.pt"
    torch.save(
        {
            "train_config": {
                "d_model": 256,
                "n_heads": 8,
                "n_layers": 6,
                "d_ff": 1024,
                "dropout": 0.1,
                "max_length": 192,
                "norm_first": False,
            }
        },
        ckpt_path,
    )

    cfg = mlm_train.parse_args(
        ["--data-path", str(data_path), "--init-checkpoint", str(ckpt_path), "--norm-first"]
    )

    with pytest.raises(ValueError, match="norm_first"):
        mlm_train.validate_init_checkpoint_compatibility(cfg, ckpt_path)


def test_matching_norm_first_passes_init_compat_check(tmp_path: Path, project_root: Path):
    mlm_train = load_mlm_train_module(project_root)
    data_path = tmp_path / "tiny.jsonl.gz"
    data_path.write_text("", encoding="utf-8")

    ckpt_path = tmp_path / "pre_norm_best.pt"
    torch.save(
        {
            "train_config": {
                "d_model": 256,
                "n_heads": 8,
                "n_layers": 6,
                "d_ff": 1024,
                "dropout": 0.1,
                "max_length": 192,
                "norm_first": True,
            }
        },
        ckpt_path,
    )

    cfg = mlm_train.parse_args(
        ["--data-path", str(data_path), "--init-checkpoint", str(ckpt_path), "--norm-first"]
    )

    mlm_train.validate_init_checkpoint_compatibility(cfg, ckpt_path)  # must not raise


def test_legacy_checkpoint_without_norm_first_is_treated_as_post_norm(
    tmp_path: Path, project_root: Path
):
    """Pre-knob checkpoints have no `norm_first` key; they were all post-LN.

    The generic `ckpt_value is None -> skip` path would wave a pre-LN run
    through against a post-LN checkpoint, so `norm_first` needs the explicit
    False default instead.
    """
    mlm_train = load_mlm_train_module(project_root)
    data_path = tmp_path / "tiny.jsonl.gz"
    data_path.write_text("", encoding="utf-8")

    ckpt_path = tmp_path / "legacy_best.pt"
    torch.save(
        {
            "train_config": {
                "d_model": 256,
                "n_heads": 8,
                "n_layers": 6,
                "d_ff": 1024,
                "dropout": 0.1,
                "max_length": 192,
            }
        },
        ckpt_path,
    )

    cfg = mlm_train.parse_args(
        ["--data-path", str(data_path), "--init-checkpoint", str(ckpt_path), "--norm-first"]
    )

    with pytest.raises(ValueError, match="norm_first"):
        mlm_train.validate_init_checkpoint_compatibility(cfg, ckpt_path)


# --------------------------------------------------------------------------- #
# Config plumbing: flat key, nested `model:` key, and CLI precedence.
# --------------------------------------------------------------------------- #
def test_norm_first_can_be_set_from_the_nested_model_section(
    tmp_path: Path, project_root: Path
):
    mlm_train = load_mlm_train_module(project_root)
    if mlm_train.yaml is None:
        pytest.skip("PyYAML not installed in test environment")
    data_path = tmp_path / "tiny.jsonl.gz"
    data_path.write_text("", encoding="utf-8")

    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        "\n".join([f"data_path: {data_path}", "model:", "  norm_first: true"]),
        encoding="utf-8",
    )

    cfg = mlm_train.parse_args(["--config", str(config_path)])

    assert cfg.norm_first is True


def test_generation_path_reconstructs_norm_first_from_the_checkpoint(
    tmp_path: Path, project_root: Path
):
    """Train/inference parity: the infiller must rebuild the SAME norm placement.

    `hcdr3_infill.config_from_checkpoint` starts from dataclass defaults and
    overlays the checkpoint's saved `train_config`. If `norm_first` were not a
    TrainConfig field, the overlay would drop it and generation would build a
    post-LN model to hold pre-LN weights -- and for the antibody stream that
    swap is invisible to `strict=True`.
    """
    mlm_train = load_mlm_train_module(project_root)
    data_path = tmp_path / "tiny.jsonl.gz"
    data_path.write_text("", encoding="utf-8")

    checkpoint = {
        "train_config": {
            "data_path": str(data_path),
            "training_stage": "antigen_hcdr3_infill_refine",
            "norm_first": True,
        }
    }

    valid_fields = {f.name for f in dataclasses.fields(mlm_train.TrainConfig)}
    assert "norm_first" in valid_fields, "norm_first must survive the overlay filter"

    merged = mlm_train._train_config_defaults()
    merged.update(
        {k: v for k, v in checkpoint["train_config"].items() if k in valid_fields}
    )
    cfg = mlm_train.TrainConfig(**merged)

    assert cfg.norm_first is True

    model = mlm_train.build_model(
        mlm_train.build_tokenizer(), cfg, torch.device("cpu")
    )
    assert isinstance(model, AntibodyAntigenCrossAttention)
    assert all(
        layer.norm_first is True for layer in model.antibody_encoder.encoder.layers
    )
    assert hasattr(model, "fusion_out_norm_antibody")


def test_cli_no_norm_first_overrides_a_config_that_enables_it(
    tmp_path: Path, project_root: Path
):
    mlm_train = load_mlm_train_module(project_root)
    if mlm_train.yaml is None:
        pytest.skip("PyYAML not installed in test environment")
    data_path = tmp_path / "tiny.jsonl.gz"
    data_path.write_text("", encoding="utf-8")

    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        "\n".join([f"data_path: {data_path}", "norm_first: true"]),
        encoding="utf-8",
    )

    cfg = mlm_train.parse_args(["--config", str(config_path), "--no-norm-first"])

    assert cfg.norm_first is False
