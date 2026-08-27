"""Regression tests for the per-epoch metric accumulators in `mlm_train`.

`train_one_epoch` declared `length_correct` / `length_total` / `length_nll_sum` /
`strength_pred_all` / `strength_target_all` and then read them at the end of the
epoch -- but never wrote to them. `evaluate` had the matching accumulation block;
the training loop did not. Every run with those heads enabled therefore reported
`train.length_acc = NaN` over `train.length_eligible_rows = 0`, which reads as
"measured, no signal" rather than "never measured".

The bug is invisible to a shape/smoke test: the keys were present and the values
were of the right type. Only asserting on the CONTENT catches it.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from smallAntibodyGen.data.MLMCollator import OASSequenceDataset


ANTIGEN = "MKTFFVLLLACTIVAALYPQGSHMRVDPTLQNAWKGCEDFIRTLPQVSAYNKLMDEGHRWQ"


def load_mlm_train_module(project_root: Path):
    script_path = project_root.parents[1] / "scripts" / "mlm_train.py"
    spec = importlib.util.spec_from_file_location("mlm_train", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _antigen_dataset(write_processed_jsonl_gz, tmp_path, heavy_seq, heavy_cdr3):
    """A tiny antigen-stage corpus: real binder labels, valid HCDR3 spans, and
    graded strength quantiles, so the length AND strength heads both have rows."""
    start = heavy_seq.index(heavy_cdr3)
    records = []
    for idx in range(12):
        records.append(
            {
                "sequence": heavy_seq,
                "locus": "PAIRED_ANTIGEN",
                "chain_group": "paired_antigen",
                "split": "train" if idx < 8 else "val",
                "length": len(heavy_seq),
                "cdr3_start_aa": start,
                "cdr3_end_aa": start + len(heavy_cdr3),
                "cdr3_aa": heavy_cdr3,
                "cdr3_start_aa_heavy": start,
                "cdr3_end_aa_heavy": start + len(heavy_cdr3),
                "cdr3_aa_heavy": heavy_cdr3,
                "sequence_heavy": heavy_seq,
                "heavy_locus": "IGH",
                "sequence_antigen": ANTIGEN,
                "antigen_length": len(ANTIGEN),
                "record_id": f"r{idx}",
                "target_key": f"uniprot:p{idx % 3}",
                "dataset": "ds",
                "affinity_type": "bool",
                "binder_label": idx % 2,
                "affinity_strength_quantile": (idx % 5) / 4.0,
            }
        )
    return write_processed_jsonl_gz(tmp_path / "antigen.jsonl.gz", records)


def _cfg(mlm_train, data_path, init_ckpt, **overrides):
    params = dict(
        data_path=str(data_path),
        training_stage="antigen_real_label_refine",
        init_checkpoint=str(init_ckpt),
        epochs=1,
        batch_size=4,
        eval_batch_size=4,
        max_length=192,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
        hcdr3_span_probability=0.0,
        learning_rate=0.01,
        length_head_max=32,
    )
    params.update(overrides)
    return mlm_train.TrainConfig(**params)


def _run_one_epoch(mlm_train, cfg, tokenizer, data_path):
    device = torch.device("cpu")
    train_ds = OASSequenceDataset(str(data_path), split="train")
    model = mlm_train.build_model(tokenizer, cfg, device)
    optimizer = mlm_train.build_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    scheduler = mlm_train.build_lr_scheduler(optimizer, cfg)
    return mlm_train.train_one_epoch(
        model=model,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
        cfg=cfg,
        device=device,
        epoch=0,
    )


def test_train_epoch_reports_real_length_metrics(
    project_root: Path, tmp_path: Path, tokenizer, write_processed_jsonl_gz,
    heavy_seq, heavy_cdr3,
):
    mlm_train = load_mlm_train_module(project_root)
    data_path = _antigen_dataset(write_processed_jsonl_gz, tmp_path, heavy_seq, heavy_cdr3)
    cfg = _cfg(
        mlm_train, data_path, tmp_path / "init.pt", length_loss_weight=1.0
    )
    metrics = _run_one_epoch(mlm_train, cfg, tokenizer, data_path)

    # The dataset has valid, in-range, non-shuffled spans on every row, so the
    # length head MUST have seen rows. Zero here is the original bug.
    assert metrics["length_eligible_rows"] > 0
    assert 0.0 <= metrics["length_acc"] <= 1.0
    assert metrics["length_nll"] > 0.0


def test_train_epoch_reports_real_strength_metrics(
    project_root: Path, tmp_path: Path, tokenizer, write_processed_jsonl_gz,
    heavy_seq, heavy_cdr3,
):
    mlm_train = load_mlm_train_module(project_root)
    data_path = _antigen_dataset(write_processed_jsonl_gz, tmp_path, heavy_seq, heavy_cdr3)
    cfg = _cfg(
        mlm_train, data_path, tmp_path / "init.pt", strength_loss_weight=1.0
    )
    metrics = _run_one_epoch(mlm_train, cfg, tokenizer, data_path)

    assert metrics["strength_eligible_rows"] > 0
    # Spearman is defined only with >= 2 non-constant pairs; the corpus above
    # spans five distinct quantiles, so a NaN here means nothing was collected.
    assert metrics["val_strength_spearman"] == metrics["val_strength_spearman"]


def test_length_and_strength_keys_stay_absent_when_the_heads_are_off(
    project_root: Path, tmp_path: Path, tokenizer, write_processed_jsonl_gz,
    heavy_seq, heavy_cdr3,
):
    """The accumulation fix must not leak keys into default runs."""
    mlm_train = load_mlm_train_module(project_root)
    data_path = _antigen_dataset(write_processed_jsonl_gz, tmp_path, heavy_seq, heavy_cdr3)
    cfg = _cfg(mlm_train, data_path, tmp_path / "init.pt")
    metrics = _run_one_epoch(mlm_train, cfg, tokenizer, data_path)

    for key in (
        "length_eligible_rows",
        "length_acc",
        "length_nll",
        "strength_eligible_rows",
        "val_strength_spearman",
    ):
        assert key not in metrics
