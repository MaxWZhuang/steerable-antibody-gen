"""Tests for opt-in training infrastructure added to scripts/mlm_train.py:
cosine LR decay, early stopping, and optional TensorBoard logging.

Every feature is default-off; the first tests here pin that the historical
warmup-then-constant behavior is unchanged when the new knobs are left at their
defaults.
"""
from __future__ import annotations

import importlib.util
import math
import sys
import warnings
from pathlib import Path

import pytest
import torch


def load_mlm_train_module(project_root: Path):
    script_path = project_root.parents[1] / "scripts" / "mlm_train.py"
    spec = importlib.util.spec_from_file_location("mlm_train", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _lr_trace(sched, optimizer, n_steps):
    seen = []
    with warnings.catch_warnings():
        # The scheduler is stepped without a paired optimizer.step() here; that
        # ordering warning is irrelevant to the LR values under test.
        warnings.simplefilter("ignore", UserWarning)
        for _ in range(n_steps):
            seen.append(optimizer.param_groups[0]["lr"])
            sched.step()
    return seen


# --------------------------------------------------------------------------- #
# Config: new fields default to the historical behavior and validate.
# --------------------------------------------------------------------------- #
def test_new_schedule_fields_default_to_historical_behavior(tmp_path: Path, project_root: Path):
    mlm_train = load_mlm_train_module(project_root)
    data_path = tmp_path / "tiny.jsonl.gz"
    data_path.write_text("", encoding="utf-8")

    cfg = mlm_train.parse_args(["--data-path", str(data_path)])

    assert cfg.lr_schedule == "constant"
    assert cfg.min_lr_ratio == 0.0
    assert cfg.early_stopping_patience == 0
    assert cfg.early_stopping_min_delta == 0.0
    assert cfg.tensorboard is False


def test_parse_args_accepts_new_schedule_flags(tmp_path: Path, project_root: Path):
    mlm_train = load_mlm_train_module(project_root)
    data_path = tmp_path / "tiny.jsonl.gz"
    data_path.write_text("", encoding="utf-8")

    cfg = mlm_train.parse_args(
        [
            "--data-path", str(data_path),
            "--lr-schedule", "cosine",
            "--min-lr-ratio", "0.1",
            "--early-stopping-patience", "3",
            "--early-stopping-min-delta", "0.01",
            "--tensorboard",
        ]
    )

    assert cfg.lr_schedule == "cosine"
    assert cfg.min_lr_ratio == pytest.approx(0.1)
    assert cfg.early_stopping_patience == 3
    assert cfg.early_stopping_min_delta == pytest.approx(0.01)
    assert cfg.tensorboard is True


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"lr_schedule": "bogus"}, "lr_schedule"),
        ({"min_lr_ratio": 1.5}, "min_lr_ratio"),
        ({"min_lr_ratio": -0.1}, "min_lr_ratio"),
        ({"early_stopping_patience": -1}, "early_stopping_patience"),
        ({"early_stopping_min_delta": -0.5}, "early_stopping_min_delta"),
    ],
)
def test_config_validate_rejects_bad_schedule_fields(project_root: Path, overrides, match):
    mlm_train = load_mlm_train_module(project_root)
    cfg = mlm_train.TrainConfig(data_path="x", **overrides)
    with pytest.raises(ValueError, match=match):
        cfg.validate()


# --------------------------------------------------------------------------- #
# LR scheduler: constant path unchanged; cosine decays to the floor.
# --------------------------------------------------------------------------- #
def test_constant_schedule_matches_legacy_warmup_ramp(project_root: Path):
    """The default (constant) schedule must be byte-for-byte the old warmup ramp."""
    mlm_train = load_mlm_train_module(project_root)
    cfg = mlm_train.TrainConfig(data_path="x", learning_rate=0.1, warmup_steps=4)
    opt = mlm_train.build_optimizer(torch.nn.Linear(3, 3), cfg)
    sched = mlm_train.build_lr_scheduler(opt, cfg)  # no total_steps -> legacy call
    assert sched is not None
    assert _lr_trace(sched, opt, 5) == pytest.approx([0.025, 0.05, 0.075, 0.1, 0.1])


def test_constant_no_warmup_still_returns_none(project_root: Path):
    mlm_train = load_mlm_train_module(project_root)
    cfg = mlm_train.TrainConfig(data_path="x", warmup_steps=0)
    opt = mlm_train.build_optimizer(torch.nn.Linear(2, 2), cfg)
    assert mlm_train.build_lr_scheduler(opt, cfg) is None


def test_cosine_warms_up_then_decays_to_floor(project_root: Path):
    mlm_train = load_mlm_train_module(project_root)
    cfg = mlm_train.TrainConfig(
        data_path="x",
        learning_rate=1.0,
        warmup_steps=2,
        lr_schedule="cosine",
        min_lr_ratio=0.0,
    )
    opt = mlm_train.build_optimizer(torch.nn.Linear(3, 3), cfg)
    sched = mlm_train.build_lr_scheduler(opt, cfg, total_steps=6)
    assert sched is not None

    seen = _lr_trace(sched, opt, 7)
    # Warmup: step0 -> 1/2, step1 -> 2/2 (=peak at end of warmup).
    assert seen[0] == pytest.approx(0.5)
    assert seen[1] == pytest.approx(1.0)
    # At the start of the decay window (step == warmup) the multiplier is 1.0.
    assert seen[2] == pytest.approx(1.0)
    # Half-cosine midpoint (step 4, progress 0.5) sits at 0.5 of peak.
    assert seen[4] == pytest.approx(0.5)
    # End of horizon reaches the floor (min_lr_ratio == 0).
    assert seen[6] == pytest.approx(0.0, abs=1e-9)
    # Monotonic non-increasing across the decay window.
    assert all(a + 1e-9 >= b for a, b in zip(seen[2:], seen[3:]))


def test_cosine_respects_nonzero_floor(project_root: Path):
    mlm_train = load_mlm_train_module(project_root)
    cfg = mlm_train.TrainConfig(
        data_path="x", learning_rate=1.0, warmup_steps=0,
        lr_schedule="cosine", min_lr_ratio=0.1,
    )
    opt = mlm_train.build_optimizer(torch.nn.Linear(3, 3), cfg)
    sched = mlm_train.build_lr_scheduler(opt, cfg, total_steps=4)
    seen = _lr_trace(sched, opt, 5)
    assert seen[0] == pytest.approx(1.0)          # progress 0 -> peak
    assert seen[-1] == pytest.approx(0.1, abs=1e-9)  # never drops below the floor


def test_cosine_without_horizon_falls_back_to_constant(project_root: Path):
    """Cosine requested but no usable total_steps must not divide by zero; it
    degrades to warmup-then-constant."""
    mlm_train = load_mlm_train_module(project_root)
    cfg = mlm_train.TrainConfig(
        data_path="x", learning_rate=0.1, warmup_steps=2, lr_schedule="cosine",
    )
    opt = mlm_train.build_optimizer(torch.nn.Linear(3, 3), cfg)
    sched = mlm_train.build_lr_scheduler(opt, cfg, total_steps=None)
    seen = _lr_trace(sched, opt, 4)
    assert seen == pytest.approx([0.05, 0.1, 0.1, 0.1])


# --------------------------------------------------------------------------- #
# Early stopping decision logic.
# --------------------------------------------------------------------------- #
def test_early_stopping_disabled_when_patience_zero(project_root: Path):
    mlm_train = load_mlm_train_module(project_root)
    count, stop = mlm_train.early_stopping_decision(
        val_loss=9.9, best_val_loss=0.1, epochs_without_improvement=5,
        patience=0, min_delta=0.0,
    )
    assert (count, stop) == (0, False)


def test_early_stopping_resets_on_improvement_and_fires_after_patience(project_root: Path):
    mlm_train = load_mlm_train_module(project_root)
    # Improvement resets the counter.
    count, stop = mlm_train.early_stopping_decision(
        val_loss=0.4, best_val_loss=0.5, epochs_without_improvement=1,
        patience=2, min_delta=0.0,
    )
    assert (count, stop) == (0, False)

    # Two consecutive non-improving epochs reach patience=2 and fire.
    count, stop = mlm_train.early_stopping_decision(
        val_loss=0.6, best_val_loss=0.5, epochs_without_improvement=0,
        patience=2, min_delta=0.0,
    )
    assert (count, stop) == (1, False)
    count, stop = mlm_train.early_stopping_decision(
        val_loss=0.6, best_val_loss=0.5, epochs_without_improvement=count,
        patience=2, min_delta=0.0,
    )
    assert (count, stop) == (2, True)


def test_early_stopping_min_delta_requires_meaningful_gain(project_root: Path):
    mlm_train = load_mlm_train_module(project_root)
    # A gain smaller than min_delta does NOT count as improvement.
    count, stop = mlm_train.early_stopping_decision(
        val_loss=0.499, best_val_loss=0.5, epochs_without_improvement=0,
        patience=1, min_delta=0.01,
    )
    assert (count, stop) == (1, True)


def test_early_stopping_non_finite_loss_counts_as_no_improvement(project_root: Path):
    mlm_train = load_mlm_train_module(project_root)
    count, stop = mlm_train.early_stopping_decision(
        val_loss=float("nan"), best_val_loss=0.5, epochs_without_improvement=0,
        patience=1, min_delta=0.0,
    )
    assert (count, stop) == (1, True)


# --------------------------------------------------------------------------- #
# TensorBoard writer builder.
# --------------------------------------------------------------------------- #
def test_tensorboard_disabled_returns_none(project_root: Path, tmp_path: Path):
    mlm_train = load_mlm_train_module(project_root)
    cfg = mlm_train.TrainConfig(data_path="x", tensorboard=False)
    assert mlm_train.build_tensorboard_writer(cfg, tmp_path) is None


def test_tensorboard_enabled_writes_under_output_dir(project_root: Path, tmp_path: Path):
    pytest.importorskip("tensorboard", reason="optional 'tb' extra not installed")
    mlm_train = load_mlm_train_module(project_root)
    cfg = mlm_train.TrainConfig(data_path="x", tensorboard=True)
    writer = mlm_train.build_tensorboard_writer(cfg, tmp_path)
    assert writer is not None
    try:
        # log_epoch_scalars with a real writer must not raise; None is a no-op.
        mlm_train.log_epoch_scalars(
            writer, 1, {"loss": 0.5}, {"loss": 0.6, "mlm_acc": 0.4}, 1e-4
        )
        mlm_train.log_epoch_scalars(None, 1, {"loss": 0.5}, {"loss": 0.6}, 1e-4)
        writer.flush()
    finally:
        writer.close()
    assert (tmp_path / "tb").exists()


# --------------------------------------------------------------------------- #
# Checkpoint config field + atomic write + intra-epoch checkpointing.
# --------------------------------------------------------------------------- #
def test_checkpoint_every_steps_defaults_off_and_validates(tmp_path: Path, project_root: Path):
    mlm_train = load_mlm_train_module(project_root)
    data_path = tmp_path / "tiny.jsonl.gz"
    data_path.write_text("", encoding="utf-8")

    cfg = mlm_train.parse_args(["--data-path", str(data_path)])
    assert cfg.checkpoint_every_steps == 0  # disabled by default

    cfg = mlm_train.parse_args(["--data-path", str(data_path), "--checkpoint-every-steps", "500"])
    assert cfg.checkpoint_every_steps == 500

    with pytest.raises(ValueError, match="checkpoint_every_steps"):
        mlm_train.TrainConfig(data_path="x", checkpoint_every_steps=-1).validate()


def test_save_checkpoint_is_atomic_and_leaves_no_tmp(project_root: Path, tmp_path: Path):
    mlm_train = load_mlm_train_module(project_root)
    cfg = mlm_train.TrainConfig(data_path="x")
    model = torch.nn.Linear(4, 4)
    opt = mlm_train.build_optimizer(model, cfg)

    ckpt = tmp_path / "last.pt"
    mlm_train.save_checkpoint(ckpt, model, opt, cfg, epoch=2, val_loss=0.3)

    assert ckpt.exists()
    # os.replace() moves the temp onto the final path -> no leftover .tmp.
    assert not (tmp_path / "last.pt.tmp").exists()
    payload = torch.load(ckpt, map_location="cpu")
    assert payload["epoch"] == 2
    assert payload["val_loss"] == pytest.approx(0.3)


def _tiny_base_dataset(write_processed_jsonl_gz, tmp_path):
    import random as _random

    rng = _random.Random(0)
    aa = "ACDEFGHIKLMNPQRSTVWY"

    def rec():
        seq = "".join(rng.choice(aa) for _ in range(30))
        return {"sequence": seq, "locus": "IGH", "chain_group": "heavy",
                "split": "train", "length": 30}

    records = [rec() for _ in range(16)]
    return write_processed_jsonl_gz(tmp_path / "tiny.jsonl.gz", records)


def _tiny_base_cfg(mlm_train, data_path, **overrides):
    params = dict(
        data_path=str(data_path), training_stage="base", epochs=1,
        batch_size=4, eval_batch_size=4, max_length=64,
        d_model=32, n_heads=4, n_layers=1, d_ff=64, dropout=0.0,
        hcdr3_span_probability=0.0, learning_rate=0.01,
    )
    params.update(overrides)
    return mlm_train.TrainConfig(**params)


def test_intra_epoch_checkpoint_writes_last_with_epoch_index_and_best_loss(
    project_root: Path, tmp_path: Path, tokenizer, write_processed_jsonl_gz
):
    mlm_train = load_mlm_train_module(project_root)
    from smallAntibodyGen.data.MLMCollator import OASSequenceDataset

    data_path = _tiny_base_dataset(write_processed_jsonl_gz, tmp_path)
    cfg = _tiny_base_cfg(mlm_train, data_path, checkpoint_every_steps=2)
    device = torch.device("cpu")
    train_ds = OASSequenceDataset(str(data_path), split="train")
    model = mlm_train.build_model(tokenizer, cfg, device)
    opt = mlm_train.build_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    sched = mlm_train.build_lr_scheduler(opt, cfg)

    out_dir = tmp_path / "run"
    out_dir.mkdir()
    mlm_train.train_one_epoch(
        model=model, train_dataset=train_ds, tokenizer=tokenizer, optimizer=opt,
        scaler=scaler, scheduler=sched, cfg=cfg, device=device, epoch=0,
        output_dir=out_dir, best_val_loss=0.42,
    )

    last = out_dir / "last.pt"
    assert last.exists()
    assert not (out_dir / "last.pt.tmp").exists()
    payload = torch.load(last, map_location="cpu")
    # 0-based epoch index -> a resume re-enters this epoch from batch 0.
    assert payload["epoch"] == 0
    # best_val_loss carried through so resume keeps best-tracking, not inf.
    assert payload["val_loss"] == pytest.approx(0.42)


def test_intra_epoch_checkpoint_disabled_when_steps_zero(
    project_root: Path, tmp_path: Path, tokenizer, write_processed_jsonl_gz
):
    mlm_train = load_mlm_train_module(project_root)
    from smallAntibodyGen.data.MLMCollator import OASSequenceDataset

    data_path = _tiny_base_dataset(write_processed_jsonl_gz, tmp_path)
    cfg = _tiny_base_cfg(mlm_train, data_path, checkpoint_every_steps=0)  # default: off
    device = torch.device("cpu")
    train_ds = OASSequenceDataset(str(data_path), split="train")
    model = mlm_train.build_model(tokenizer, cfg, device)
    opt = mlm_train.build_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    sched = mlm_train.build_lr_scheduler(opt, cfg)

    out_dir = tmp_path / "run"
    out_dir.mkdir()
    # Even with an output_dir passed, steps==0 must write nothing mid-epoch.
    mlm_train.train_one_epoch(
        model=model, train_dataset=train_ds, tokenizer=tokenizer, optimizer=opt,
        scaler=scaler, scheduler=sched, cfg=cfg, device=device, epoch=0,
        output_dir=out_dir, best_val_loss=0.42,
    )
    assert not (out_dir / "last.pt").exists()


# --------------------------------------------------------------------------- #
# new_module_lr_multiplier + best_checkpoint_metric (ported from mirror)
# --------------------------------------------------------------------------- #
def _lr_knob_cfg(mlm_train, tmp_path: Path, *extra: str):
    data_path = tmp_path / "tiny.jsonl.gz"
    data_path.write_text("", encoding="utf-8")
    ckpt = tmp_path / "init.pt"
    ckpt.write_text("", encoding="utf-8")
    return mlm_train.parse_args(
        [
            "--data-path",
            str(data_path),
            "--training-stage",
            "antigen_real_label_refine",
            "--init-checkpoint",
            str(ckpt),
            *extra,
        ]
    )


def _dual_model(mlm_train, norm_first: bool = False):
    from smallAntibodyGen.models.mlm import AntibodyAntigenCrossAttention, MLMConfig

    return AntibodyAntigenCrossAttention(
        MLMConfig(
            vocab_size=40,
            pad_token_id=0,
            max_length=64,
            d_model=32,
            n_heads=4,
            n_layers=1,
            d_ff=64,
            dropout=0.0,
            norm_first=norm_first,
        )
    )


def test_new_module_lr_prefixes_match_the_warm_start_missing_keys(project_root: Path):
    """The prefix tuple is not eyeballed: it must equal the set of modules the
    real translation function leaves randomly initialized, in BOTH norm modes.

    If someone renames a fusion module, this test fails instead of the multiplier
    silently applying to nothing.
    """
    from smallAntibodyGen.models.mlm import AntibodyMLM, MLMConfig

    mlm_train = load_mlm_train_module(project_root)
    base = dict(
        vocab_size=40,
        pad_token_id=0,
        max_length=64,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
    )
    single = AntibodyMLM(MLMConfig(**base))
    translated = mlm_train.build_antigen_refine_init_state_dict(single.state_dict())

    observed: set[str] = set()
    for norm_first in (False, True):
        dual = _dual_model(mlm_train, norm_first=norm_first)
        incompatible = dual.load_state_dict(translated, strict=False)
        observed |= {key.split(".")[0] + "." for key in incompatible.missing_keys}

    assert observed == set(mlm_train.NEW_MODULE_LR_PREFIXES)


def test_default_multiplier_keeps_the_historical_two_group_optimizer(
    tmp_path: Path, project_root: Path
):
    mlm_train = load_mlm_train_module(project_root)
    cfg = _lr_knob_cfg(mlm_train, tmp_path)
    assert cfg.new_module_lr_multiplier == 1.0
    optimizer = mlm_train.build_optimizer(_dual_model(mlm_train), cfg)
    assert len(optimizer.param_groups) == 2
    assert all(group["lr"] == cfg.learning_rate for group in optimizer.param_groups)


def test_multiplier_splits_four_groups_with_the_scaled_lr(
    tmp_path: Path, project_root: Path
):
    mlm_train = load_mlm_train_module(project_root)
    cfg = _lr_knob_cfg(mlm_train, tmp_path, "--new-module-lr-multiplier", "4.0")
    model = _dual_model(mlm_train)
    optimizer = mlm_train.build_optimizer(model, cfg)
    assert len(optimizer.param_groups) == 4
    base_lr = cfg.learning_rate
    assert [group["lr"] for group in optimizer.param_groups] == [
        base_lr,
        base_lr,
        base_lr * 4.0,
        base_lr * 4.0,
    ]
    assert [group["weight_decay"] for group in optimizer.param_groups] == [
        cfg.weight_decay,
        0.0,
        cfg.weight_decay,
        0.0,
    ]
    # Every trainable parameter lands in exactly one group, none is dropped.
    grouped = sum(len(group["params"]) for group in optimizer.param_groups)
    assert grouped == sum(1 for p in model.parameters() if p.requires_grad)


def test_multiplier_routes_exactly_the_new_modules(tmp_path: Path, project_root: Path):
    mlm_train = load_mlm_train_module(project_root)
    cfg = _lr_knob_cfg(mlm_train, tmp_path, "--new-module-lr-multiplier", "4.0")
    model = _dual_model(mlm_train)
    optimizer = mlm_train.build_optimizer(model, cfg)

    scaled_ids = {
        id(p) for group in optimizer.param_groups[2:] for p in group["params"]
    }
    expected_ids = {
        id(p)
        for name, p in model.named_parameters()
        if p.requires_grad and name.startswith(mlm_train.NEW_MODULE_LR_PREFIXES)
    }
    assert scaled_ids == expected_ids
    assert scaled_ids  # the set is non-empty, i.e. the test is not vacuous


def test_multiplier_fails_loud_when_no_targeted_modules_exist(
    tmp_path: Path, project_root: Path
):
    """A single-stream model has none of the targeted modules; silently applying
    the knob to nothing would make an arm look like it ran when it did not."""
    from smallAntibodyGen.models.mlm import AntibodyMLM, MLMConfig

    mlm_train = load_mlm_train_module(project_root)
    cfg = _lr_knob_cfg(mlm_train, tmp_path, "--new-module-lr-multiplier", "4.0")
    single = AntibodyMLM(
        MLMConfig(
            vocab_size=40,
            pad_token_id=0,
            max_length=64,
            d_model=32,
            n_heads=4,
            n_layers=1,
            d_ff=64,
            dropout=0.0,
        )
    )
    with pytest.raises(ValueError, match="none of the targeted modules"):
        mlm_train.build_optimizer(single, cfg)


def test_multiplier_rejected_off_antigen_stages_and_when_non_positive(
    tmp_path: Path, project_root: Path
):
    mlm_train = load_mlm_train_module(project_root)
    data_path = tmp_path / "tiny.jsonl.gz"
    data_path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="only supported for antigen stages"):
        mlm_train.parse_args(
            ["--data-path", str(data_path), "--new-module-lr-multiplier", "4.0"]
        )
    with pytest.raises(ValueError, match="new_module_lr_multiplier must be > 0"):
        _lr_knob_cfg(mlm_train, tmp_path, "--new-module-lr-multiplier", "0.0")


def test_select_checkpoint_metric_value_defaults_to_combined_loss(
    tmp_path: Path, project_root: Path
):
    mlm_train = load_mlm_train_module(project_root)
    cfg = _lr_knob_cfg(mlm_train, tmp_path)
    assert cfg.best_checkpoint_metric == "val_loss"
    metrics = {"loss": 1.25, "compatibility_loss": 0.5}
    assert mlm_train.select_checkpoint_metric_value(cfg, metrics) == 1.25


def test_select_checkpoint_metric_value_can_key_on_the_compat_term(
    tmp_path: Path, project_root: Path
):
    mlm_train = load_mlm_train_module(project_root)
    cfg = _lr_knob_cfg(mlm_train, tmp_path, "--best-checkpoint-metric", "val_compat_loss")
    metrics = {"loss": 1.25, "compatibility_loss": 0.5}
    assert mlm_train.select_checkpoint_metric_value(cfg, metrics) == 0.5


def test_best_checkpoint_metric_rejected_off_antigen_stages(
    tmp_path: Path, project_root: Path
):
    mlm_train = load_mlm_train_module(project_root)
    data_path = tmp_path / "tiny.jsonl.gz"
    data_path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="only supported for antigen"):
        mlm_train.parse_args(
            ["--data-path", str(data_path), "--best-checkpoint-metric", "val_compat_loss"]
        )


def test_mechanism_search_knob_defaults_are_pinned(tmp_path: Path, project_root: Path):
    """One cross-knob default pin: any of these flipping by accident silently
    changes every existing run."""
    mlm_train = load_mlm_train_module(project_root)
    data_path = tmp_path / "tiny.jsonl.gz"
    data_path.write_text("", encoding="utf-8")
    cfg = mlm_train.parse_args(["--data-path", str(data_path)])
    assert cfg.mlm_loss_weight == 1.0
    assert cfg.compat_readout == "cls"
    assert cfg.new_module_lr_multiplier == 1.0
    assert cfg.best_checkpoint_metric == "val_loss"


# --------------------------------------------------------------------------- #
# J03: run fingerprints and checkpoint lineage, end to end through main().
#
# `test_experiment_fingerprints.py` pins the pure hashing contract. These pin
# the WIRING: that every checkpoint carries a fingerprint, that a resume against
# an edited config/corpus/architecture is refused BEFORE any state is loaded,
# and that a legacy checkpoint is a hard error for resume but only a warning for
# warm start.
# --------------------------------------------------------------------------- #
def _fp_dataset(write_processed_jsonl_gz, tmp_path: Path, *, salt: str = "a"):
    import random as _random

    rng = _random.Random(0)
    aa = "ACDEFGHIKLMNPQRSTVWY"
    records = []
    for i in range(16):
        seq = salt[0] + "".join(rng.choice(aa) for _ in range(29))
        records.append({
            "sequence": seq, "locus": "IGH", "chain_group": "heavy",
            "split": "train" if i < 12 else "val", "length": 30,
        })
    return write_processed_jsonl_gz(tmp_path / "tiny.jsonl.gz", records)


def _fp_cfg(mlm_train, data_path, out_dir, **overrides):
    params = dict(
        data_path=str(data_path), training_stage="base", output_dir=str(out_dir),
        epochs=1, batch_size=4, eval_batch_size=4, max_length=64,
        d_model=32, n_heads=4, n_layers=1, d_ff=64, dropout=0.0,
        hcdr3_span_probability=0.0, shuffle_pair_probability=0.0,
        pair_loss_weight=0.0, learning_rate=1e-3, device="cpu",
        show_progress=False, resume_from_last=False,
    )
    params.update(overrides)
    return mlm_train.TrainConfig(**params)


def _drive_main(mlm_train, cfg):
    """Run main() with `cfg` substituted for CLI parsing."""
    original = mlm_train.parse_args
    mlm_train.parse_args = lambda *a, **k: cfg
    try:
        mlm_train.main()
    finally:
        mlm_train.parse_args = original


def test_save_checkpoint_records_the_run_fingerprint_when_given_one(
    project_root: Path, tmp_path: Path
):
    mlm_train = load_mlm_train_module(project_root)
    from smallAntibodyGen import experiment

    cfg = mlm_train.TrainConfig(data_path="x")
    model = torch.nn.Linear(4, 4)
    opt = mlm_train.build_optimizer(model, cfg)
    fingerprint = {"run_hash": "f" * 64, "components": {}, "manifests": {}}

    ckpt = tmp_path / "last.pt"
    mlm_train.save_checkpoint(ckpt, model, opt, cfg, epoch=1, val_loss=0.1,
                              run_fingerprint=fingerprint)
    payload = torch.load(ckpt, map_location="cpu")
    assert payload[experiment.RUN_FINGERPRINT_KEY]["run_hash"] == "f" * 64

    # No fingerprint supplied -> the legacy shape, which resume then refuses.
    legacy = tmp_path / "legacy.pt"
    mlm_train.save_checkpoint(legacy, model, opt, cfg, epoch=1, val_loss=0.1)
    assert experiment.RUN_FINGERPRINT_KEY not in torch.load(legacy, map_location="cpu")


def test_main_writes_the_fingerprint_into_checkpoints_and_beside_them(
    project_root: Path, tmp_path: Path, write_processed_jsonl_gz
):
    import json

    from smallAntibodyGen import experiment

    mlm_train = load_mlm_train_module(project_root)
    data_path = _fp_dataset(write_processed_jsonl_gz, tmp_path)
    out_dir = tmp_path / "run"
    _drive_main(mlm_train, _fp_cfg(mlm_train, data_path, out_dir))

    sidecar = json.loads((out_dir / "run_fingerprint.json").read_text(encoding="utf-8"))
    assert set(sidecar["components"]) == {
        "architecture", "objective", "tokenizer", "data", "contracts", "source"
    }
    for name in ("last.pt", "best.pt"):
        payload = torch.load(out_dir / name, map_location="cpu")
        stored = payload[experiment.RUN_FINGERPRINT_KEY]
        assert stored["run_hash"] == sidecar["run_hash"]
        assert stored["parent_checkpoint_hash"] is None  # from-scratch run
        assert stored["worktree_dirty"] in (True, False)


def test_run_architecture_manifest_covers_the_mlmconfig_only_fields(
    project_root: Path, tmp_path: Path, write_processed_jsonl_gz
):
    """`asdict(TrainConfig)` cannot see these four; the fingerprint must."""
    mlm_train = load_mlm_train_module(project_root)
    data_path = _fp_dataset(write_processed_jsonl_gz, tmp_path)
    cfg = _fp_cfg(mlm_train, data_path, tmp_path / "run")
    fingerprint = mlm_train.build_run_fingerprint(cfg, mlm_train.build_tokenizer(), None)

    model_config = fingerprint["manifests"]["architecture"]["model_config"]
    for field in ("activation", "tie_weights", "initializer_range", "scale_residual_init"):
        assert field in model_config, field
        assert field not in cfg.__dataclass_fields__, f"{field} unexpectedly on TrainConfig"
    assert model_config["vocab_size"] == 35
    assert fingerprint["manifests"]["architecture"]["model_class"] == "AntibodyMLM"


@pytest.mark.parametrize("overrides, expected_field", [
    ({"mask_probability": 0.4}, "mask_probability"),
    ({"seed": 1234}, "seed"),
    ({"d_model": 64}, "d_model"),
    ({"use_amp": True}, "use_amp"),
    ({"eval_batch_size": 8}, "eval_batch_size"),
    ({"train_num_workers": 1}, "train_num_workers"),
])
def test_resume_is_refused_before_any_state_is_loaded_when_a_field_changed(
    project_root: Path, tmp_path: Path, write_processed_jsonl_gz, capsys,
    overrides, expected_field,
):
    from smallAntibodyGen import experiment

    mlm_train = load_mlm_train_module(project_root)
    data_path = _fp_dataset(write_processed_jsonl_gz, tmp_path)
    out_dir = tmp_path / "run"
    _drive_main(mlm_train, _fp_cfg(mlm_train, data_path, out_dir))
    capsys.readouterr()

    edited = _fp_cfg(mlm_train, data_path, out_dir, epochs=2, resume_from_last=True,
                     **overrides)
    with pytest.raises(experiment.ResumeFingerprintMismatch) as excinfo:
        _drive_main(mlm_train, edited)

    message = str(excinfo.value)
    assert expected_field in message
    # Field names and both values, not two opaque hashes.
    assert "checkpoint=" in message and "run=" in message
    # And nothing was restored: load_checkpoint never ran.
    assert "[checkpoint] loaded <-" not in capsys.readouterr().out


def test_resume_is_refused_when_the_corpus_content_changed(
    project_root: Path, tmp_path: Path, write_processed_jsonl_gz
):
    """Same filename, different bytes -- the dataset hash is what catches it."""
    from smallAntibodyGen import experiment

    mlm_train = load_mlm_train_module(project_root)
    data_path = _fp_dataset(write_processed_jsonl_gz, tmp_path, salt="a")
    out_dir = tmp_path / "run"
    _drive_main(mlm_train, _fp_cfg(mlm_train, data_path, out_dir))

    _fp_dataset(write_processed_jsonl_gz, tmp_path, salt="c")  # regenerated in place
    with pytest.raises(experiment.ResumeFingerprintMismatch, match="data"):
        _drive_main(mlm_train, _fp_cfg(mlm_train, data_path, out_dir, epochs=2,
                                       resume_from_last=True))


def test_resume_against_a_legacy_checkpoint_is_a_hard_error(
    project_root: Path, tmp_path: Path, write_processed_jsonl_gz, capsys
):
    """All six checkpoints on disk predate J03: 5 top-level keys, no fingerprint."""
    from smallAntibodyGen import experiment

    mlm_train = load_mlm_train_module(project_root)
    data_path = _fp_dataset(write_processed_jsonl_gz, tmp_path)
    out_dir = tmp_path / "run"
    _drive_main(mlm_train, _fp_cfg(mlm_train, data_path, out_dir))

    payload = torch.load(out_dir / "last.pt", map_location="cpu")
    payload.pop(experiment.RUN_FINGERPRINT_KEY)
    torch.save(payload, out_dir / "last.pt")
    capsys.readouterr()

    with pytest.raises(experiment.LegacyCheckpointResumeError) as excinfo:
        _drive_main(mlm_train, _fp_cfg(mlm_train, data_path, out_dir, epochs=2,
                                       resume_from_last=True))
    message = str(excinfo.value)
    assert experiment.RUN_FINGERPRINT_KEY in message
    assert "--init-checkpoint" in message  # points at the supported alternative
    assert "[checkpoint] loaded <-" not in capsys.readouterr().out


def test_operational_only_edits_still_permit_a_resume(
    project_root: Path, tmp_path: Path, write_processed_jsonl_gz, capsys
):
    """The approved exclusion list must actually buy something."""
    mlm_train = load_mlm_train_module(project_root)
    data_path = _fp_dataset(write_processed_jsonl_gz, tmp_path)
    out_dir = tmp_path / "run"
    _drive_main(mlm_train, _fp_cfg(mlm_train, data_path, out_dir, show_progress=False))
    capsys.readouterr()

    # show_progress, tensorboard and report_masked_fraction_bins are
    # operational-only, and resume_from_last is the flag the check gates on.
    _drive_main(mlm_train, _fp_cfg(
        mlm_train, data_path, out_dir,
        resume_from_last=True, show_progress=True, report_masked_fraction_bins=True,
    ))
    assert "[checkpoint] loaded <-" in capsys.readouterr().out


def test_warm_start_from_a_legacy_checkpoint_warns_loudly_but_proceeds(
    project_root: Path, tmp_path: Path, capsys
):
    """Owner decision: a legacy warm start is allowed, still validated, and loud."""
    from smallAntibodyGen import experiment

    mlm_train = load_mlm_train_module(project_root)
    cfg = mlm_train.TrainConfig(
        data_path="x", max_length=64, d_model=32, n_heads=4, n_layers=1, d_ff=64,
        dropout=0.0,
    )
    model = mlm_train.build_model(mlm_train.build_tokenizer(), cfg, torch.device("cpu"))
    ckpt = tmp_path / "legacy.pt"
    mlm_train.save_checkpoint(ckpt, model, mlm_train.build_optimizer(model, cfg), cfg,
                              epoch=1, val_loss=1.0)  # no run_fingerprint -> legacy
    assert experiment.RUN_FINGERPRINT_KEY not in torch.load(ckpt, map_location="cpu")

    mlm_train.validate_init_checkpoint_compatibility(cfg, ckpt)  # must not raise
    out = capsys.readouterr().out
    assert "UNVERIFIABLE" in out
    assert experiment.RUN_FINGERPRINT_KEY in out

    # The existing architecture validator still runs on the same legacy payload.
    with pytest.raises(ValueError, match="d_model"):
        mlm_train.validate_init_checkpoint_compatibility(
            mlm_train.TrainConfig(data_path="x", max_length=64, d_model=64, n_heads=4,
                                  n_layers=1, d_ff=64, dropout=0.0),
            ckpt,
        )


def test_require_clean_worktree_defaults_off_and_is_parseable(
    tmp_path: Path, project_root: Path
):
    mlm_train = load_mlm_train_module(project_root)
    data_path = tmp_path / "tiny.jsonl.gz"
    data_path.write_text("", encoding="utf-8")

    assert mlm_train.parse_args(["--data-path", str(data_path)]).require_clean_worktree is False
    assert mlm_train.parse_args(
        ["--data-path", str(data_path), "--require-clean-worktree"]
    ).require_clean_worktree is True
    with pytest.raises(ValueError, match="require_clean_worktree"):
        mlm_train.TrainConfig(data_path="x", require_clean_worktree="yes").validate()
