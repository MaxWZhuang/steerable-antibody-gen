from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def load_mlm_train_module(project_root: Path):
    script_path = project_root.parents[1] / "scripts" / "mlm_train.py"
    spec = importlib.util.spec_from_file_location("mlm_train", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_args_accepts_saved_yaml_config(tmp_path: Path, project_root: Path):
    mlm_train = load_mlm_train_module(project_root)
    if mlm_train.yaml is None:
        pytest.skip("PyYAML not installed in test environment")
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
    if mlm_train.yaml is None:
        pytest.skip("PyYAML not installed in test environment")
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
    if mlm_train.yaml is None:
        pytest.skip("PyYAML not installed in test environment")
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


def test_parse_args_antigen_refine_defaults_to_separate_output_dir(tmp_path: Path, project_root: Path):
    mlm_train = load_mlm_train_module(project_root)
    data_path = tmp_path / "tiny_antigen.jsonl.gz"
    data_path.write_text("", encoding="utf-8")

    cfg = mlm_train.parse_args(
        [
            "--data-path",
            str(data_path),
            "--training-stage",
            "antigen_refine",
        ]
    )

    assert cfg.training_stage == "antigen_refine"
    assert cfg.output_dir == "checkpoints/mlm_antigen_refine"


def test_antigen_refine_uses_dual_stream_collator_and_model(tmp_path: Path, project_root: Path):
    mlm_train = load_mlm_train_module(project_root)
    from smallAntibodyGen.models.mlm import AntibodyAntigenCrossAttention

    data_path = tmp_path / "tiny_antigen.jsonl.gz"
    record = {
        "record_id": "antigen-1",
        "sequence": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVS",
        "sequence_heavy": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVS",
        "sequence_antigen": "MKTIIALSYIFCLVFADYKDDDDK",
        "locus": "PAIRED_ANTIGEN",
        "chain_group": "paired_antigen",
        "split": "train",
        "length": 56,
        "target_key": "uniprot:p11111",
        "target_name": "test_target",
        "target_pdb": "1abc",
        "target_uniprot": "P12345",
        "dataset": "asd-test",
        "confidence": "very_high",
        "affinity_type": "bool",
        "affinity_raw": "1.0",
        "processed_measurement_raw": "1.0",
        "processed_measurement_float": 1.0,
        "binder_label": 1,
        "is_strong_binder": True,
        "is_nanobody": True,
        "scfv": False,
        "cdr3_aa_heavy": "CARDRST",
        "cdr3_start_aa_heavy": 10,
        "cdr3_end_aa_heavy": 17,
        "heavy_locus": "IGH",
        "light_locus": None,
        "is_paired": False,
        "antigen_length": 24,
        "metadata": {},
        "source_file": "tiny_antigen.parquet",
    }
    with gzip.open(data_path, "wt", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
        f.write(json.dumps({**record, "record_id": "antigen-2", "target_key": "uniprot:p22222", "sequence_antigen": "ACDEFGHIKLMNPQRSTVWY"}) + "\n")

    cfg = mlm_train.parse_args(
        [
            "--data-path",
            str(data_path),
            "--training-stage",
            "antigen_refine",
            "--batch-size",
            "2",
            "--eval-batch-size",
            "2",
            "--max-length",
            "64",
        ]
    )
    tokenizer = mlm_train.build_tokenizer()
    train_dataset, _ = mlm_train.build_datasets(cfg)
    loader = mlm_train.build_train_loader(train_dataset, tokenizer, cfg, epoch=0)
    batch = next(iter(loader))
    model = mlm_train.build_model(tokenizer, cfg, device=mlm_train.torch.device("cpu"))

    assert isinstance(model, AntibodyAntigenCrossAttention)
    assert "antibody_input_ids" in batch
    assert "antigen_input_ids" in batch
    assert "compatibility_labels" in batch
