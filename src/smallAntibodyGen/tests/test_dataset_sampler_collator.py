from __future__ import annotations

import json
from pathlib import Path

import torch

from smallAntibodyGen.data.MLMCollator import (
    ChainLengthBucketBatchSampler,
    MLMCollator,
    OASSequenceDataset
)
from smallAntibodyGen.tokenizer import AminoAcidTokenizer


def make_processed_record(tokenizer, sequence, locus, chain_group, split="train", cdr3_aa=None, cdr3_start_aa=None, cdr3_end_aa=None):
    token_ids = tokenizer.encode_sequence(sequence, locus=locus, max_length=192)
    return {
        "sequence": sequence,
        "token_ids": token_ids,
        "locus": locus,
        "chain_group": chain_group,
        "split": split,
        "length": len(sequence),
        "token_length": len(token_ids),
        "cdr3_aa": cdr3_aa,
        "cdr3_start_aa": cdr3_start_aa,
        "cdr3_end_aa": cdr3_end_aa,
        "v_call": None,
        "d_call": None,
        "j_call": None,
        "redundancy": 1,
        "source_file": "tiny.csv.gz",
        "metadata": {},
    }


def test_dataset_loads_and_filters_by_split(tmp_path: Path, tokenizer, write_processed_jsonl_gz):
    records = [
        make_processed_record(tokenizer, "CARDRST", "IGH", "heavy", split="train"),
        make_processed_record(tokenizer, "QQYNSY", "IGK", "light", split="val"),
    ]
    data_path = write_processed_jsonl_gz(tmp_path / "processed.jsonl.gz", records)

    train_ds = OASSequenceDataset(data_path, split="train")
    val_ds = OASSequenceDataset(data_path, split="val")

    assert len(train_ds) == 1
    assert len(val_ds) == 1
    assert train_ds[0].chain_group == "heavy"
    assert val_ds[0].chain_group == "light"


def test_sampler_batches_are_chain_homogeneous_and_length_bucketed(tmp_path: Path, tokenizer, write_processed_jsonl_gz):
    records = [
        make_processed_record(tokenizer, "A" * 100, "IGH", "heavy"),
        make_processed_record(tokenizer, "A" * 103, "IGH", "heavy"),
        make_processed_record(tokenizer, "A" * 116, "IGH", "heavy"),
        make_processed_record(tokenizer, "A" * 119, "IGH", "heavy"),
        make_processed_record(tokenizer, "A" * 80, "IGK", "light"),
        make_processed_record(tokenizer, "A" * 83, "IGK", "light"),
    ]
    data_path = write_processed_jsonl_gz(tmp_path / "processed.jsonl.gz", records)
    ds = OASSequenceDataset(data_path, split="train")

    sampler = ChainLengthBucketBatchSampler(
        ds,
        batch_size=2,
        bucket_width=8,
        drop_last=False,
        seed=42,
    )

    for batch_indices in sampler:
        recs = [ds[i] for i in batch_indices]
        assert len({r.chain_group for r in recs}) == 1
        assert len({r.length // 8 for r in recs}) == 1


def test_collator_returns_expected_shapes(tmp_path: Path, tokenizer, write_processed_jsonl_gz):
    records = [
        make_processed_record(tokenizer, "CARDRST", "IGH", "heavy"),
        make_processed_record(tokenizer, "QQYNSY", "IGK", "light"),
    ]
    data_path = write_processed_jsonl_gz(tmp_path / "processed.jsonl.gz", records)
    ds = OASSequenceDataset(data_path, split="train")

    collator = MLMCollator(
        tokenizer=tokenizer,
        max_length=64,
        mask_probability=0.15,
        hcdr3_span_probability=0.0,
        rng_seed=42,
    )
    batch = collator([ds[0], ds[1]])

    assert set(batch.keys()) == {"input_ids", "attention_mask", "labels"}
    assert batch["input_ids"].shape == batch["attention_mask"].shape == batch["labels"].shape
    assert batch["input_ids"].dtype == torch.long
    assert batch["attention_mask"].dtype == torch.long
    assert batch["labels"].dtype == torch.long


def test_collator_never_targets_special_positions(tmp_path: Path, tokenizer, write_processed_jsonl_gz):
    records = [
        make_processed_record(tokenizer, "CARDRST", "IGH", "heavy"),
        make_processed_record(tokenizer, "QQYNSY", "IGK", "light"),
    ]
    data_path = write_processed_jsonl_gz(tmp_path / "processed.jsonl.gz", records)
    ds = OASSequenceDataset(data_path, split="train")

    collator = MLMCollator(
        tokenizer=tokenizer,
        max_length=64,
        mask_probability=1.0,
        hcdr3_span_probability=0.0,
        rng_seed=42,
    )
    batch = collator([ds[0], ds[1]])

    for i, record in enumerate([ds[0], ds[1]]):
        token_len = record.token_length
        labels = batch["labels"][i]

        # [CLS] and chain token
        assert labels[0].item() == -100
        assert labels[1].item() == -100

        # [EOS]
        assert labels[token_len - 1].item() == -100

        # padding (if any)
        for j in range(token_len, labels.size(0)):
            assert labels[j].item() == -100


def test_hcdr3_masking_hits_hcdr3_positions(tmp_path: Path, tokenizer, write_processed_jsonl_gz):
    sequence = "AAAAABCDEFGHIIII"
    cdr3 = "ABCDEFGH"
    record = make_processed_record(
        tokenizer,
        sequence=sequence,
        locus="IGH",
        chain_group="heavy",
        cdr3_aa=cdr3,
        cdr3_start_aa=4,
        cdr3_end_aa=12,
    )
    data_path = write_processed_jsonl_gz(tmp_path / "processed.jsonl.gz", [record])
    ds = OASSequenceDataset(data_path, split="train")

    collator = MLMCollator(
        tokenizer=tokenizer,
        max_length=64,
        mask_probability=0.15,
        hcdr3_span_probability=1.0,
        hcdr3_span_min=3,
        hcdr3_span_max=3,
        rng_seed=42,
    )
    batch = collator([ds[0]])
    labels = batch["labels"][0]

    hcdr3_token_positions = set(range(2 + 4, 2 + 12))
    targeted_positions = {j for j, v in enumerate(labels.tolist()) if v != -100}

    assert len(hcdr3_token_positions.intersection(targeted_positions)) > 0


def test_attention_mask_matches_padding(tmp_path: Path, tokenizer, write_processed_jsonl_gz):
    records = [
        make_processed_record(tokenizer, "CARDRST", "IGH", "heavy"),
        make_processed_record(tokenizer, "QQY", "IGK", "light"),
    ]
    data_path = write_processed_jsonl_gz(tmp_path / "processed.jsonl.gz", records)
    ds = OASSequenceDataset(data_path, split="train")

    collator = MLMCollator(
        tokenizer=tokenizer,
        max_length=64,
        mask_probability=0.15,
        hcdr3_span_probability=0.0,
        rng_seed=42,
    )
    batch = collator([ds[0], ds[1]])

    for i, record in enumerate([ds[0], ds[1]]):
        token_len = record.token_length
        mask = batch["attention_mask"][i].tolist()
        assert mask[:token_len] == [1] * token_len
        assert mask[token_len:] == [0] * (len(mask) - token_len)
        