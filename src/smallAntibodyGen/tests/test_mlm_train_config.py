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
    init_ckpt = tmp_path / "best.pt"
    init_ckpt.write_text("placeholder", encoding="utf-8")

    cfg = mlm_train.parse_args(
        [
            "--data-path",
            str(data_path),
            "--training-stage",
            "antigen_refine",
            "--init-checkpoint",
            str(init_ckpt),
        ]
    )

    assert cfg.training_stage == "antigen_refine"
    assert cfg.output_dir == "checkpoints/mlm_antigen_refine"


def test_parse_args_antigen_refine_requires_init_checkpoint(tmp_path: Path, project_root: Path):
    mlm_train = load_mlm_train_module(project_root)
    data_path = tmp_path / "tiny_antigen.jsonl.gz"
    data_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="requires `init_checkpoint`"):
        mlm_train.parse_args(
            [
                "--data-path",
                str(data_path),
                "--training-stage",
                "antigen_refine",
            ]
        )


def test_parse_args_antigen_real_label_refine_defaults_to_separate_output_dir(
    tmp_path: Path,
    project_root: Path,
):
    mlm_train = load_mlm_train_module(project_root)
    data_path = tmp_path / "tiny_antigen.jsonl.gz"
    data_path.write_text("", encoding="utf-8")
    init_ckpt = tmp_path / "best.pt"
    init_ckpt.write_text("placeholder", encoding="utf-8")

    cfg = mlm_train.parse_args(
        [
            "--data-path",
            str(data_path),
            "--training-stage",
            "antigen_real_label_refine",
            "--init-checkpoint",
            str(init_ckpt),
        ]
    )

    assert cfg.training_stage == "antigen_real_label_refine"
    assert cfg.output_dir == "checkpoints/mlm_antigen_real_label_refine"


def test_parse_args_antigen_real_label_refine_requires_init_checkpoint(
    tmp_path: Path,
    project_root: Path,
):
    mlm_train = load_mlm_train_module(project_root)
    data_path = tmp_path / "tiny_antigen.jsonl.gz"
    data_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="requires `init_checkpoint`"):
        mlm_train.parse_args(
            [
                "--data-path",
                str(data_path),
                "--training-stage",
                "antigen_real_label_refine",
            ]
        )


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
    init_ckpt = tmp_path / "best.pt"
    init_ckpt.write_text("placeholder", encoding="utf-8")

    cfg = mlm_train.parse_args(
        [
            "--data-path",
            str(data_path),
            "--training-stage",
            "antigen_refine",
            "--init-checkpoint",
            str(init_ckpt),
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
    assert "target_keys" in batch
    assert "dataset_names" in batch
    assert "antibody_format_groups" in batch
    assert "antigen_length_buckets" in batch


def test_antigen_real_label_refine_filters_to_binary_labels_and_uses_dual_stream(
    tmp_path: Path,
    project_root: Path,
):
    mlm_train = load_mlm_train_module(project_root)
    from smallAntibodyGen.models.mlm import AntibodyAntigenCrossAttention

    data_path = tmp_path / "tiny_antigen_real_label.jsonl.gz"
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
        f.write(json.dumps({**record, "record_id": "antigen-2", "target_key": "uniprot:p22222", "binder_label": 0, "is_strong_binder": False}) + "\n")
        f.write(json.dumps({**record, "record_id": "antigen-3", "target_key": "uniprot:p33333", "binder_label": None, "is_strong_binder": False}) + "\n")
    init_ckpt = tmp_path / "best.pt"
    init_ckpt.write_text("placeholder", encoding="utf-8")

    cfg = mlm_train.parse_args(
        [
            "--data-path",
            str(data_path),
            "--training-stage",
            "antigen_real_label_refine",
            "--init-checkpoint",
            str(init_ckpt),
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

    assert len(train_dataset.records) == 2
    assert isinstance(model, AntibodyAntigenCrossAttention)
    assert batch["compatibility_mask"].tolist() == [True, True]
    assert sorted(batch["compatibility_labels"].tolist()) == [0, 1]
    assert batch["is_shuffled_antigen"].tolist() == [False, False]


def test_compatibility_binary_metrics_are_deterministic(project_root: Path):
    mlm_train = load_mlm_train_module(project_root)

    metrics = mlm_train.compatibility_binary_metrics(
        labels=[1, 0, 1, 0],
        scores=[0.9, 0.8, 0.4, 0.1],
        preds=[1, 1, 0, 0],
    )

    assert metrics["compatibility_labeled_count"] == 4
    assert metrics["compatibility_positive_rate"] == pytest.approx(0.5)
    assert metrics["compatibility_precision"] == pytest.approx(0.5)
    assert metrics["compatibility_recall"] == pytest.approx(0.5)
    assert metrics["compatibility_specificity"] == pytest.approx(0.5)
    assert metrics["compatibility_balanced_acc"] == pytest.approx(0.5)
    assert metrics["compatibility_mcc"] == pytest.approx(0.0)
    assert metrics["compatibility_auroc"] == pytest.approx(0.75)
    assert metrics["compatibility_auprc"] == pytest.approx(5 / 6)


def test_antigen_refine_baseline_diagnostics_return_expected_keys(tmp_path: Path, project_root: Path):
    mlm_train = load_mlm_train_module(project_root)

    data_path = tmp_path / "tiny_antigen_diag.jsonl.gz"
    base_record = {
        "sequence": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVS",
        "sequence_heavy": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVS",
        "sequence_antigen": "MKTIIALSYIFCLVFADYKDDDDK",
        "locus": "PAIRED_ANTIGEN",
        "chain_group": "paired_antigen",
        "split": "train",
        "length": 56,
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
        "is_nanobody": False,
        "scfv": False,
        "cdr3_aa_heavy": "CARDRST",
        "cdr3_start_aa_heavy": 10,
        "cdr3_end_aa_heavy": 17,
        "heavy_locus": "IGH",
        "light_locus": None,
        "is_paired": False,
        "metadata": {},
        "source_file": "tiny_antigen.parquet",
    }
    records = [
        {
            **base_record,
            "record_id": "train-1",
            "target_key": "uniprot:p11111",
            "antigen_length": 24,
            "split": "train",
        },
        {
            **base_record,
            "record_id": "train-2",
            "target_key": "uniprot:p22222",
            "dataset": "asd-alt",
            "sequence_antigen": "ACDEFGHIKLMNPQRSTVWYACDE",
            "antigen_length": 24,
            "split": "train",
        },
        {
            **base_record,
            "record_id": "val-1",
            "target_key": "uniprot:p33333",
            "sequence_antigen": "QRSTVWYACDEFGHIKLMNPQRST",
            "antigen_length": 24,
            "split": "val",
        },
        {
            **base_record,
            "record_id": "val-2",
            "target_key": "uniprot:p44444",
            "dataset": "asd-alt",
            "sequence_antigen": "LMNPQRSTVWYACDEFGHIKLMNP",
            "antigen_length": 24,
            "split": "val",
        },
    ]
    with gzip.open(data_path, "wt", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    init_ckpt = tmp_path / "best.pt"
    init_ckpt.write_text("placeholder", encoding="utf-8")

    cfg = mlm_train.parse_args(
        [
            "--data-path",
            str(data_path),
            "--training-stage",
            "antigen_refine",
            "--init-checkpoint",
            str(init_ckpt),
            "--batch-size",
            "2",
            "--eval-batch-size",
            "2",
            "--max-length",
            "64",
        ]
    )
    tokenizer = mlm_train.build_tokenizer()
    train_dataset, val_dataset = mlm_train.build_datasets(cfg)

    baselines = mlm_train.fit_group_majority_baselines(train_dataset, tokenizer, cfg)
    metrics = mlm_train.evaluate_group_majority_baselines(val_dataset, tokenizer, cfg, baselines)

    assert baselines["fit_records"] == len(train_dataset)
    assert baselines["fit_labeled_examples"] > 0
    assert "majority_maps" in baselines
    assert int(metrics["labeled_examples"]) > 0
    assert 0.0 <= metrics["always_positive_acc"] <= 1.0
    assert 0.0 <= metrics["target_key_majority_acc"] <= 1.0
    assert 0.0 <= metrics["dataset_majority_acc"] <= 1.0
    assert 0.0 <= metrics["format_majority_acc"] <= 1.0
    assert 0.0 <= metrics["antigen_bucket_majority_acc"] <= 1.0


def test_build_antigen_refine_init_state_dict_clones_encoder_into_both_branches(project_root: Path):
    mlm_train = load_mlm_train_module(project_root)

    checkpoint_state_dict = {
        "sequence_encoder.token_embedding.weight": mlm_train.torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "sequence_encoder.final_norm.weight": mlm_train.torch.tensor([5.0, 6.0]),
        "sequence_encoder.final_norm.bias": mlm_train.torch.tensor([7.0, 8.0]),
        "lm_head.weight": mlm_train.torch.tensor([[9.0, 10.0], [11.0, 12.0]]),
        "pair_head.weight": mlm_train.torch.tensor([[13.0, 14.0], [15.0, 16.0]]),
    }

    translated = mlm_train.build_antigen_refine_init_state_dict(checkpoint_state_dict)

    assert "antibody_encoder.token_embedding.weight" in translated
    assert "antigen_encoder.token_embedding.weight" in translated
    assert "antibody_encoder.final_norm.weight" in translated
    assert "antigen_encoder.final_norm.bias" in translated
    assert "lm_head.weight" in translated
    assert "pair_head.weight" not in translated
    assert mlm_train.torch.equal(
        translated["antibody_encoder.token_embedding.weight"],
        checkpoint_state_dict["sequence_encoder.token_embedding.weight"],
    )
    assert mlm_train.torch.equal(
        translated["antigen_encoder.token_embedding.weight"],
        checkpoint_state_dict["sequence_encoder.token_embedding.weight"],
    )


def test_build_antigen_refine_init_state_dict_accepts_bare_encoder_keys(project_root: Path):
    mlm_train = load_mlm_train_module(project_root)

    checkpoint_state_dict = {
        "token_embedding.weight": mlm_train.torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "final_norm.bias": mlm_train.torch.tensor([5.0, 6.0]),
        "lm_head.weight": mlm_train.torch.tensor([[7.0, 8.0], [9.0, 10.0]]),
    }

    translated = mlm_train.build_antigen_refine_init_state_dict(checkpoint_state_dict)

    assert mlm_train.torch.equal(
        translated["antibody_encoder.token_embedding.weight"],
        checkpoint_state_dict["token_embedding.weight"],
    )
    assert mlm_train.torch.equal(
        translated["antigen_encoder.final_norm.bias"],
        checkpoint_state_dict["final_norm.bias"],
    )
    assert mlm_train.torch.equal(
        translated["lm_head.weight"],
        checkpoint_state_dict["lm_head.weight"],
    )


def test_initialize_antigen_refine_from_checkpoint_loads_both_encoders_and_lm_head(
    tmp_path: Path,
    project_root: Path,
):
    mlm_train = load_mlm_train_module(project_root)
    from smallAntibodyGen.models.mlm import AntibodyAntigenCrossAttention, AntibodyMLM, MLMConfig

    config = MLMConfig(
        vocab_size=32,
        pad_token_id=0,
        max_length=16,
        d_model=8,
        n_heads=2,
        n_layers=1,
        d_ff=16,
        dropout=0.0,
    )
    source_model = AntibodyMLM(config)
    for _, param in source_model.named_parameters():
        if param.requires_grad:
            param.data.fill_(0.25)

    ckpt_path = tmp_path / "paired_init.pt"
    mlm_train.torch.save(
        {
            "model_state_dict": source_model.state_dict(),
            "train_config": {
                "d_model": 8,
                "n_heads": 2,
                "n_layers": 1,
                "d_ff": 16,
                "dropout": 0.0,
                "max_length": 16,
            },
        },
        ckpt_path,
    )

    target_model = AntibodyAntigenCrossAttention(config)
    for _, param in target_model.named_parameters():
        if param.requires_grad:
            param.data.zero_()

    mlm_train.initialize_antigen_refine_from_checkpoint(
        path=ckpt_path,
        model=target_model,
        map_location="cpu",
    )

    assert mlm_train.torch.allclose(
        target_model.antibody_encoder.token_embedding.weight,
        source_model.sequence_encoder.token_embedding.weight,
    )
    assert mlm_train.torch.allclose(
        target_model.antigen_encoder.token_embedding.weight,
        source_model.sequence_encoder.token_embedding.weight,
    )
    assert mlm_train.torch.allclose(
        target_model.antibody_encoder.final_norm.weight,
        source_model.sequence_encoder.final_norm.weight,
    )
    assert mlm_train.torch.allclose(
        target_model.antigen_encoder.final_norm.bias,
        source_model.sequence_encoder.final_norm.bias,
    )
    assert mlm_train.torch.allclose(
        target_model.lm_head.weight,
        source_model.lm_head.weight,
    )
