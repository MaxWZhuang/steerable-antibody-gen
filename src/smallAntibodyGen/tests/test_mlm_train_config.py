from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def load_mlm_train_module(project_root: Path):
    script_path = project_root / "scripts" / "mlm_train.py"
    spec = importlib.util.spec_from_file_location("mlm_train", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_args_accepts_saved_yaml_config(tmp_path: Path, project_root: Path):
    mlm_train = load_mlm_train_module(project_root)
    data_path = tmp_path / "tiny.jsonl.gz"
    data_path.write_text("", encoding="utf-8")

    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"data_path: {data_path}",
                "seed: 7",
                "batch_size: 16",
                "num_workers: 2",
                "max_length: 128",
                "mixed_precision: true",
                "epochs: 3",
                "warmup_steps: 500",
                "model:",
                "  d_model: 192",
                "  n_heads: 6",
                "  n_layers: 4",
                "  d_ff: 768",
                "  dropout: 0.2",
                "optimizer:",
                "  betas: [0.9, 0.999]",
                "logging:",
                "  eval_every: 500",
            ]
        ),
        encoding="utf-8",
    )

    cfg = mlm_train.parse_args(["--config", str(config_path)])

    assert cfg.data_path == str(data_path)
    assert cfg.seed == 7
    assert cfg.batch_size == 16
    assert cfg.train_num_workers == 2
    assert cfg.eval_num_workers == 2
    assert cfg.max_length == 128
    assert cfg.use_amp is True
    assert cfg.epochs == 3
    assert cfg.d_model == 192
    assert cfg.n_heads == 6
    assert cfg.n_layers == 4
    assert cfg.d_ff == 768
    assert cfg.dropout == 0.2


def test_parse_args_cli_values_override_config(tmp_path: Path, project_root: Path):
    mlm_train = load_mlm_train_module(project_root)
    data_path = tmp_path / "tiny.jsonl.gz"
    data_path.write_text("", encoding="utf-8")

    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"data_path: {data_path}",
                "batch_size: 16",
                "mixed_precision: false",
                "model:",
                "  d_model: 192",
            ]
        ),
        encoding="utf-8",
    )

    cfg = mlm_train.parse_args(
        [
            "--config",
            str(config_path),
            "--batch-size",
            "64",
            "--use-amp",
            "--d-model",
            "256",
        ]
    )

    assert cfg.batch_size == 64
    assert cfg.use_amp is True
    assert cfg.d_model == 256


def test_parse_args_requires_data_path_when_not_in_config(tmp_path: Path, project_root: Path):
    mlm_train = load_mlm_train_module(project_root)
    config_path = tmp_path / "train.yaml"
    config_path.write_text("batch_size: 16\n", encoding="utf-8")

    with pytest.raises(SystemExit):
        mlm_train.parse_args(["--config", str(config_path)])


def test_parse_args_paired_refine_defaults_to_separate_output_dir(tmp_path: Path, project_root: Path):
    mlm_train = load_mlm_train_module(project_root)
    data_path = tmp_path / "tiny.jsonl.gz"
    data_path.write_text("", encoding="utf-8")
    init_ckpt = tmp_path / "best.pt"
    init_ckpt.write_text("placeholder", encoding="utf-8")

    cfg = mlm_train.parse_args(
        [
            "--data-path",
            str(data_path),
            "--training-stage",
            "paired_refine",
            "--init-checkpoint",
            str(init_ckpt),
        ]
    )

    assert cfg.training_stage == "paired_refine"
    assert cfg.output_dir == "checkpoints/mlm_paired_refine"


def test_parse_args_paired_refine_requires_init_checkpoint(tmp_path: Path, project_root: Path):
    mlm_train = load_mlm_train_module(project_root)
    data_path = tmp_path / "tiny.jsonl.gz"
    data_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="requires `init_checkpoint`"):
        mlm_train.parse_args(
            [
                "--data-path",
                str(data_path),
                "--training-stage",
                "paired_refine",
            ]
        )
