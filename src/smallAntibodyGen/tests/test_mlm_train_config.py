from __future__ import annotations

import gzip
import importlib.util
import json
import math
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
    assert cfg.warmup_steps == 500  # honored as a real field, not silently dropped
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


def test_parse_args_antigen_hcdr3_infill_defaults_to_full_span_settings(
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
            "antigen_hcdr3_infill_refine",
            "--init-checkpoint",
            str(init_ckpt),
        ]
    )

    assert cfg.training_stage == "antigen_hcdr3_infill_refine"
    assert cfg.output_dir == "checkpoints/mlm_antigen_hcdr3_infill_refine"
    assert cfg.hcdr3_mask_mode == "full_span"
    assert cfg.mask_replacement_strategy == "always_mask"
    assert cfg.compatibility_loss_weight == 0.0
    assert cfg.shuffle_antigen_probability == 0.0


def test_parse_args_antigen_hcdr3_infill_requires_init_checkpoint(
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
                "antigen_hcdr3_infill_refine",
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


def test_antigen_hcdr3_infill_filters_to_positive_valid_spans_and_full_masks(
    tmp_path: Path,
    project_root: Path,
):
    mlm_train = load_mlm_train_module(project_root)
    from smallAntibodyGen.models.mlm import AntibodyAntigenCrossAttention

    data_path = tmp_path / "tiny_hcdr3_infill.jsonl.gz"
    heavy_sequence = "EVQLVESGGGCARDRSTWGQGTLV"
    cdr3_start = heavy_sequence.index("CARDRST")
    cdr3_end = cdr3_start + len("CARDRST")
    base_record = {
        "record_id": "antigen-1",
        "sequence": heavy_sequence,
        "sequence_heavy": heavy_sequence,
        "sequence_antigen": "MKTIIALSYIFCLVFADYKDDDDK",
        "locus": "PAIRED_ANTIGEN",
        "chain_group": "paired_antigen",
        "split": "train",
        "length": len(heavy_sequence),
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
        "cdr3_aa": "CARDRST",
        "cdr3_aa_heavy": "CARDRST",
        "cdr3_start_aa": cdr3_start,
        "cdr3_end_aa": cdr3_end,
        "cdr3_start_aa_heavy": cdr3_start,
        "cdr3_end_aa_heavy": cdr3_end,
        "heavy_locus": "IGH",
        "light_locus": None,
        "is_paired": False,
        "antigen_length": 24,
        "metadata": {},
        "source_file": "tiny_antigen.parquet",
    }
    with gzip.open(data_path, "wt", encoding="utf-8") as f:
        f.write(json.dumps(base_record) + "\n")
        # Non-strong binder: excluded by the is_strong_binder infill gate. (Must
        # override the inherited is_strong_binder=True from base_record, else it
        # would be a contradictory non-binder still flagged as a strong binder.)
        f.write(json.dumps({**base_record, "record_id": "neg", "binder_label": 0, "is_strong_binder": False}) + "\n")
        f.write(json.dumps({**base_record, "record_id": "missing-antigen", "sequence_antigen": ""}) + "\n")
        f.write(json.dumps({**base_record, "record_id": "bad-span", "cdr3_end_aa": None, "cdr3_end_aa_heavy": None}) + "\n")
    init_ckpt = tmp_path / "best.pt"
    init_ckpt.write_text("placeholder", encoding="utf-8")

    cfg = mlm_train.parse_args(
        [
            "--data-path",
            str(data_path),
            "--training-stage",
            "antigen_hcdr3_infill_refine",
            "--init-checkpoint",
            str(init_ckpt),
            "--batch-size",
            "1",
            "--eval-batch-size",
            "1",
            "--max-length",
            "64",
        ]
    )
    tokenizer = mlm_train.build_tokenizer()
    train_dataset, _ = mlm_train.build_datasets(cfg)
    loader = mlm_train.build_train_loader(train_dataset, tokenizer, cfg, epoch=0)
    batch = next(iter(loader))
    model = mlm_train.build_model(tokenizer, cfg, device=mlm_train.torch.device("cpu"))

    targeted_positions = {
        idx for idx, value in enumerate(batch["antibody_labels"][0].tolist()) if value != -100
    }
    expected_positions = set(range(2 + cdr3_start, 2 + cdr3_end))

    assert len(train_dataset.records) == 1
    assert isinstance(model, AntibodyAntigenCrossAttention)
    assert targeted_positions == expected_positions
    assert batch["hcdr3_target_mask"][0].nonzero().flatten().tolist() == sorted(expected_positions)
    assert all(batch["antibody_input_ids"][0, pos].item() == tokenizer.mask_id for pos in expected_positions)


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


def test_hcdr3_metric_counts_and_finalization(project_root: Path):
    mlm_train = load_mlm_train_module(project_root)
    logits = mlm_train.torch.zeros((1, 5, 4), dtype=mlm_train.torch.float32)
    labels = mlm_train.torch.full((1, 5), -100, dtype=mlm_train.torch.long)
    labels[0, 1] = 2
    labels[0, 2] = 3
    logits[0, 1, 2] = 5.0
    logits[0, 2, 1] = 5.0
    target_mask = mlm_train.torch.zeros((1, 5), dtype=mlm_train.torch.bool)
    target_mask[0, 1:3] = True

    counts = mlm_train.hcdr3_metric_counts(
        logits,
        labels,
        target_mask,
        mlm_train.torch.tensor([1]),
        mlm_train.torch.tensor([3]),
        mlm_train.torch.tensor([True]),
    )
    metrics = mlm_train.finalize_hcdr3_metrics(counts)

    assert counts["hcdr3_target_tokens"] == 2
    assert counts["hcdr3_correct_tokens"] == 1
    assert counts["hcdr3_valid_spans"] == 1
    assert counts["hcdr3_exact_matches"] == 0
    assert metrics["hcdr3_token_acc"] == pytest.approx(0.5)
    assert metrics["hcdr3_span_exact_match"] == pytest.approx(0.0)


def test_hcdr3_metric_finalization_handles_empty_counts(project_root: Path):
    mlm_train = load_mlm_train_module(project_root)
    metrics = mlm_train.finalize_hcdr3_metrics({})

    assert math.isnan(metrics["hcdr3_token_acc"])
    assert math.isnan(metrics["hcdr3_span_exact_match"])
    assert metrics["hcdr3_target_tokens"] == 0.0
    assert metrics["hcdr3_valid_spans"] == 0.0


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


def test_parse_args_antigen_encoder_defaults_to_scratch(tmp_path: Path, project_root: Path):
    """Stage 0 must be inert: a plain run keeps the original scratch antigen stream."""
    mlm_train = load_mlm_train_module(project_root)
    data_path = tmp_path / "tiny.jsonl.gz"
    data_path.write_text("", encoding="utf-8")

    cfg = mlm_train.parse_args(["--data-path", str(data_path)])

    assert cfg.antigen_encoder_type == "scratch"
    assert cfg.esm_model_name == "facebook/esm2_t6_8M_UR50D"
    assert cfg.antigen_max_length == 512
    assert cfg.antigen_encoder_finetune == "frozen"
    assert cfg.lora_r == 8
    assert cfg.lora_alpha == 16
    assert cfg.lora_dropout == 0.05


def test_parse_args_accepts_antigen_encoder_section(tmp_path: Path, project_root: Path):
    mlm_train = load_mlm_train_module(project_root)
    if mlm_train.yaml is None:
        pytest.skip("PyYAML not installed in test environment")
    data_path = tmp_path / "tiny_antigen.jsonl.gz"
    data_path.write_text("", encoding="utf-8")
    init_ckpt = tmp_path / "best.pt"
    init_ckpt.write_text("placeholder", encoding="utf-8")

    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"data_path: {data_path}",
                "training_stage: antigen_real_label_refine",
                f"init_checkpoint: {init_ckpt}",
                "antigen_encoder:",
                "  type: esm",
                "  esm_model_name: facebook/esm2_t12_35M_UR50D",
                "  antigen_max_length: 256",
                "  finetune: lora",
                "  lora_r: 16",
                "  lora_alpha: 32",
                "  lora_dropout: 0.1",
            ]
        ),
        encoding="utf-8",
    )

    cfg = mlm_train.parse_args(["--config", str(config_path)])

    assert cfg.antigen_encoder_type == "esm"
    assert cfg.esm_model_name == "facebook/esm2_t12_35M_UR50D"
    assert cfg.antigen_max_length == 256
    assert cfg.antigen_encoder_finetune == "lora"
    assert cfg.lora_r == 16
    assert cfg.lora_alpha == 32
    assert cfg.lora_dropout == 0.1


def test_parse_args_cli_overrides_antigen_encoder_section(tmp_path: Path, project_root: Path):
    mlm_train = load_mlm_train_module(project_root)
    if mlm_train.yaml is None:
        pytest.skip("PyYAML not installed in test environment")
    data_path = tmp_path / "tiny_antigen.jsonl.gz"
    data_path.write_text("", encoding="utf-8")
    init_ckpt = tmp_path / "best.pt"
    init_ckpt.write_text("placeholder", encoding="utf-8")

    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"data_path: {data_path}",
                "training_stage: antigen_real_label_refine",
                f"init_checkpoint: {init_ckpt}",
                "antigen_encoder:",
                "  type: esm",
                "  antigen_max_length: 256",
            ]
        ),
        encoding="utf-8",
    )

    cfg = mlm_train.parse_args(
        ["--config", str(config_path), "--antigen-max-length", "384"]
    )

    assert cfg.antigen_encoder_type == "esm"
    assert cfg.antigen_max_length == 384  # CLI wins over the config section


def test_parse_args_rejects_esm_on_non_antigen_stage(tmp_path: Path, project_root: Path):
    mlm_train = load_mlm_train_module(project_root)
    data_path = tmp_path / "tiny.jsonl.gz"
    data_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="only applies to antigen stages"):
        mlm_train.parse_args(
            ["--data-path", str(data_path), "--antigen-encoder-type", "esm"]
        )


def test_parse_args_rejects_out_of_range_antigen_max_length(tmp_path: Path, project_root: Path):
    mlm_train = load_mlm_train_module(project_root)
    data_path = tmp_path / "tiny.jsonl.gz"
    data_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="antigen_max_length must be in"):
        mlm_train.parse_args(
            ["--data-path", str(data_path), "--antigen-max-length", "4096"]
        )


def test_build_model_scratch_carries_antigen_encoder_config(tmp_path: Path, project_root: Path):
    """Passthrough check: build_model wires the new fields onto MLMConfig without
    changing which model class is built (Stage 0 = no behavior change)."""
    mlm_train = load_mlm_train_module(project_root)
    from smallAntibodyGen.models.mlm import AntibodyMLM

    data_path = tmp_path / "tiny.jsonl.gz"
    data_path.write_text("", encoding="utf-8")

    cfg = mlm_train.parse_args(
        ["--data-path", str(data_path), "--antigen-max-length", "256"]
    )
    tokenizer = mlm_train.build_tokenizer()
    model = mlm_train.build_model(tokenizer, cfg, device=mlm_train.torch.device("cpu"))

    assert isinstance(model, AntibodyMLM)  # scratch/base path unchanged
    assert model.config.antigen_encoder_type == "scratch"
    assert model.config.antigen_max_length == 256  # field carried through


def test_mlmconfig_rejects_bad_antigen_encoder_fields(project_root: Path):
    from smallAntibodyGen.models.mlm import MLMConfig

    base = dict(vocab_size=32, pad_token_id=0, max_length=16)

    with pytest.raises(ValueError, match="antigen_encoder_type"):
        MLMConfig(**base, antigen_encoder_type="bogus").validate()
    with pytest.raises(ValueError, match="antigen_encoder_finetune"):
        MLMConfig(**base, antigen_encoder_finetune="bogus").validate()
    with pytest.raises(ValueError, match="antigen_max_length"):
        MLMConfig(**base, antigen_max_length=0).validate()
    with pytest.raises(ValueError, match="lora_dropout"):
        MLMConfig(**base, lora_dropout=1.5).validate()
    # A valid ESM-typed config validates cleanly.
    MLMConfig(**base, antigen_encoder_type="esm", antigen_encoder_finetune="lora").validate()


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


def test_esm_antigen_hcdr3_infill_train_step_end_to_end(tmp_path: Path, project_root: Path):
    """Smoke test the full ESM antigen path: collator emits ESM antigen ids, the ESM
    dual-stream model consumes them, and one backward step reaches the projection while
    leaving the frozen ESM backbone without gradients."""
    pytest.importorskip("transformers", reason="optional 'esm' extra not installed")
    mlm_train = load_mlm_train_module(project_root)
    torch = mlm_train.torch

    data_path = tmp_path / "tiny_esm_infill.jsonl.gz"
    heavy_sequence = "EVQLVESGGGCARDRSTWGQGTLV"
    cdr3_start = heavy_sequence.index("CARDRST")
    cdr3_end = cdr3_start + len("CARDRST")
    record = {
        "record_id": "esm-1",
        "sequence": heavy_sequence,
        "sequence_heavy": heavy_sequence,
        "sequence_antigen": "MKTIIALSYIFCLVFADYKDDDDK",
        "locus": "PAIRED_ANTIGEN",
        "chain_group": "paired_antigen",
        "split": "train",
        "length": len(heavy_sequence),
        "target_key": "uniprot:p11111",
        "dataset": "asd-test",
        "affinity_type": "bool",
        "processed_measurement_float": 1.0,
        "binder_label": 1,
        "is_strong_binder": True,
        "is_nanobody": True,
        "cdr3_aa": "CARDRST",
        "cdr3_aa_heavy": "CARDRST",
        "cdr3_start_aa": cdr3_start,
        "cdr3_end_aa": cdr3_end,
        "cdr3_start_aa_heavy": cdr3_start,
        "cdr3_end_aa_heavy": cdr3_end,
        "heavy_locus": "IGH",
        "antigen_length": 24,
    }
    with gzip.open(data_path, "wt", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
        f.write(json.dumps({**record, "record_id": "esm-2", "target_key": "uniprot:p22222"}) + "\n")
    init_ckpt = tmp_path / "best.pt"
    init_ckpt.write_text("placeholder", encoding="utf-8")

    try:
        cfg = mlm_train.parse_args(
            [
                "--data-path", str(data_path),
                "--training-stage", "antigen_hcdr3_infill_refine",
                "--init-checkpoint", str(init_ckpt),
                "--antigen-encoder-type", "esm",
                "--antigen-max-length", "32",
                "--max-length", "64",
                "--batch-size", "2",
                "--eval-batch-size", "2",
                "--d-model", "32",
                "--n-heads", "4",
                "--n-layers", "1",
                "--d-ff", "64",
                "--device", "cpu",
            ]
        )
        tokenizer = mlm_train.build_tokenizer()
        train_dataset, _ = mlm_train.build_datasets(cfg)
        loader = mlm_train.build_train_loader(train_dataset, tokenizer, cfg, epoch=0)
        model = mlm_train.build_model(tokenizer, cfg, device=torch.device("cpu"))
    except OSError:
        pytest.skip("ESM weights unavailable (offline)")

    batch = next(iter(loader))
    mlm_logits, compatibility_logits = model(
        antibody_input_ids=batch["antibody_input_ids"],
        antibody_attention_mask=batch["antibody_attention_mask"],
        antigen_input_ids=batch["antigen_input_ids"],
        antigen_attention_mask=batch["antigen_attention_mask"],
    )
    loss = model.compute_mlm_loss(mlm_logits, batch["antibody_labels"])
    loss.backward()

    from smallAntibodyGen.models.esm_antigen_encoder import ESMAntigenEncoder

    assert isinstance(model.antigen_encoder, ESMAntigenEncoder)
    assert torch.isfinite(loss)
    # Antigen tokens are ESM ids (vocab ~33), distinct from the AA tokenizer stream.
    assert int(batch["antigen_input_ids"].max()) < 33
    # The frozen ESM backbone gets no gradient; the projection does.
    assert model.antigen_encoder.projection.weight.grad is not None
    assert all(p.grad is None for p in model.antigen_encoder.esm.parameters())


def test_warm_start_into_esm_model_keeps_backbone_and_loads_antibody(tmp_path: Path, project_root: Path):
    """A scratch dual-stream checkpoint must warm-start the antibody encoder of an ESM
    model while leaving the ESM backbone at its pretrained weights (its scratch
    antigen_encoder.* keys are dropped, not force-loaded)."""
    pytest.importorskip("transformers", reason="optional 'esm' extra not installed")
    mlm_train = load_mlm_train_module(project_root)
    torch = mlm_train.torch
    from smallAntibodyGen.models.mlm import AntibodyAntigenCrossAttention, MLMConfig

    common = dict(
        vocab_size=mlm_train.build_tokenizer().vocab_size,
        pad_token_id=mlm_train.build_tokenizer().pad_id,
        max_length=32,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
    )
    scratch_model = AntibodyAntigenCrossAttention(MLMConfig(**common))
    for _, p in scratch_model.named_parameters():
        if p.requires_grad:
            p.data.fill_(0.3)
    ckpt_path = tmp_path / "scratch_dual.pt"
    torch.save({"model_state_dict": scratch_model.state_dict()}, ckpt_path)

    try:
        esm_model = AntibodyAntigenCrossAttention(
            MLMConfig(**common, antigen_encoder_type="esm", antigen_max_length=16)
        )
    except OSError:
        pytest.skip("ESM weights unavailable (offline)")

    esm_before = esm_model.antigen_encoder.esm.embeddings.word_embeddings.weight.detach().clone()

    mlm_train.initialize_antigen_refine_from_checkpoint(
        path=ckpt_path, model=esm_model, map_location="cpu"
    )

    # Antibody encoder warm-started from the checkpoint...
    assert torch.allclose(
        esm_model.antibody_encoder.token_embedding.weight,
        scratch_model.antibody_encoder.token_embedding.weight,
    )
    # ...while the ESM backbone is untouched (kept at pretrained init).
    assert torch.allclose(
        esm_model.antigen_encoder.esm.embeddings.word_embeddings.weight, esm_before
    )
