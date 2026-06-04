from __future__ import annotations

import json
from pathlib import Path

import torch

from smallAntibodyGen.data.MLMCollator import (
    AntibodyAntigenCollator,
    AntibodyAntigenRealLabelCollator,
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


def make_processed_pair_record(
    heavy_sequence,
    light_sequence,
    heavy_locus="IGH",
    light_locus="IGK",
    split="train",
):
    return {
        "pair_id": "pair-1",
        "is_paired": True,
        "sequence": heavy_sequence,
        "sequence_heavy": heavy_sequence,
        "sequence_light": light_sequence,
        "heavy_locus": heavy_locus,
        "light_locus": light_locus,
        "locus": "PAIRED",
        "chain_group": "paired",
        "split": split,
        "length": len(heavy_sequence) + len(light_sequence),
        "token_length": len(heavy_sequence) + len(light_sequence) + 5,
        "cdr3_aa_heavy": None,
        "cdr3_start_aa_heavy": None,
        "cdr3_end_aa_heavy": None,
        "cdr3_aa_light": None,
        "cdr3_start_aa_light": None,
        "cdr3_end_aa_light": None,
        "metadata": {},
        "source_file": "tiny_paired.csv.gz",
    }


def make_processed_antibody_antigen_record(
    heavy_sequence,
    antigen_sequence,
    *,
    light_sequence=None,
    split="train",
    dataset="asd-test",
    affinity_type="bool",
    affinity_raw="1.0",
    processed_measurement_raw="1.0",
    processed_measurement_float=1.0,
    binder_label=1,
    is_strong_binder=None,
    target_key="uniprot:p12345",
    record_id="antigen-1",
):
    record = {
        "record_id": record_id,
        "sequence": heavy_sequence,
        "sequence_heavy": heavy_sequence,
        "sequence_light": light_sequence,
        "sequence_antigen": antigen_sequence,
        "locus": "PAIRED_ANTIGEN",
        "chain_group": "paired_antigen",
        "split": split,
        "length": len(heavy_sequence) + len(light_sequence or ""),
        "target_key": target_key,
        "target_name": "test_target",
        "target_pdb": "1abc",
        "target_uniprot": "P12345",
        "dataset": dataset,
        "confidence": "very_high",
        "affinity_type": affinity_type,
        "affinity_raw": affinity_raw,
        "processed_measurement_raw": processed_measurement_raw,
        "processed_measurement_float": processed_measurement_float,
        "binder_label": binder_label,
        "is_nanobody": light_sequence is None,
        "scfv": False,
        "cdr3_aa_heavy": "CARDRST",
        "cdr3_start_aa_heavy": 10,
        "cdr3_end_aa_heavy": 17,
        "cdr3_aa_light": "QQYNSY" if light_sequence else None,
        "cdr3_start_aa_light": 20 if light_sequence else None,
        "cdr3_end_aa_light": 26 if light_sequence else None,
        "heavy_locus": "IGH",
        "light_locus": "IGK" if light_sequence else None,
        "is_paired": bool(light_sequence),
        "metadata": {},
        "source_file": "tiny_antigen.parquet",
    }
    if is_strong_binder is not None:
        record["is_strong_binder"] = is_strong_binder
    return record


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


def test_dataset_loads_antibody_antigen_fields_without_breaking_existing_schema(
    tmp_path: Path,
    write_processed_jsonl_gz,
):
    records = [
        make_processed_antibody_antigen_record(
            heavy_sequence="EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVS",
            light_sequence=None,
            antigen_sequence="MKTIIALSYIFCLVFADYKDDDDK",
        ),
        make_processed_antibody_antigen_record(
            heavy_sequence="QVQLQESGGGLVQAGGSLRLSCAASGFTFSSYAMGWFRQAPGKEREFVA",
            light_sequence="DIQMTQSPSSLSASVGDRVTITCQASQDINNYLNWYQQKPGKAPKLLIY",
            antigen_sequence="ACDEFGHIKLMNPQRSTVWY",
            split="val",
            binder_label=0,
            is_strong_binder=False,
        ),
    ]
    data_path = write_processed_jsonl_gz(tmp_path / "antibody_antigen.jsonl.gz", records)

    train_ds = OASSequenceDataset(data_path, split="train")
    val_ds = OASSequenceDataset(data_path, split="val")

    train_record = train_ds[0]
    val_record = val_ds[0]

    assert train_record.sequence_antigen == "MKTIIALSYIFCLVFADYKDDDDK"
    assert train_record.target_key == "uniprot:p12345"
    assert train_record.dataset_name == "asd-test"
    assert train_record.binder_label == 1
    assert train_record.is_strong_binder is True
    assert train_record.affinity_family == "binary_binding"
    assert train_record.affinity_strength_label == 1
    assert train_record.affinity_strength_score == 1.0
    assert train_record.is_nanobody is True
    assert train_record.is_paired is False

    assert val_record.sequence_light is not None
    assert val_record.sequence_antigen == "ACDEFGHIKLMNPQRSTVWY"
    assert val_record.binder_label == 0
    assert val_record.is_strong_binder is False
    assert val_record.affinity_family == "binary_binding"
    assert val_record.affinity_strength_label == 0
    assert val_record.is_nanobody is False
    assert val_record.is_paired is True


def test_dataset_derives_bool_and_kd_threshold_affinity_strength_annotations(
    tmp_path: Path,
    write_processed_jsonl_gz,
):
    kd_strong = make_processed_antibody_antigen_record(
        heavy_sequence="EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVS",
        light_sequence=None,
        antigen_sequence="MKTIIALSYIFCLVFADYKDDDDK",
        dataset="flab-test",
        affinity_type="kd",
        affinity_raw="1e-10",
        processed_measurement_raw="1e-10",
        processed_measurement_float=1e-10,
        binder_label=None,
    )
    kd_not_strong = make_processed_antibody_antigen_record(
        heavy_sequence="EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVT",
        light_sequence=None,
        antigen_sequence="MKTIIALSYIFCLVFADYKDDDDA",
        dataset="flab-test",
        affinity_type="kd",
        affinity_raw="1e-8",
        processed_measurement_raw="1e-8",
        processed_measurement_float=1e-8,
        binder_label=None,
    )
    neg_log_kd_strong = make_processed_antibody_antigen_record(
        heavy_sequence="QVQLQESGGGLVQAGGSLRLSCAASGFTFSSYAMGWFRQAPGKEREFVA",
        light_sequence=None,
        antigen_sequence="ACDEFGHIKLMNPQRSTVWY",
        dataset="flab-test",
        affinity_type="-log KD",
        affinity_raw="9.5",
        processed_measurement_raw="9.5",
        processed_measurement_float=9.5,
        binder_label=None,
    )
    neg_log_kd_not_strong = make_processed_antibody_antigen_record(
        heavy_sequence="QVQLQESGGGLVQAGGSLRLSCAASGFTFSSYAMGWFRQAPGKEREFVG",
        light_sequence=None,
        antigen_sequence="ACDEFGHIKLMNPQRSTVWF",
        dataset="flab-test",
        affinity_type="-log KD",
        affinity_raw="8.5",
        processed_measurement_raw="8.5",
        processed_measurement_float=8.5,
        binder_label=None,
    )
    fuzzy_high = make_processed_antibody_antigen_record(
        heavy_sequence="QVQLQESGGGLVQAGGSLRLSCAASGFTFSSYAMGWFRQAPGKEREFVY",
        light_sequence=None,
        antigen_sequence="ACDEFGHIKLMNPQRSTVWG",
        dataset="buzz-test",
        affinity_type="fuzzy",
        affinity_raw="h",
        processed_measurement_raw="h",
        processed_measurement_float=None,
        binder_label=None,
        is_strong_binder=False,
    )
    records = [kd_strong, kd_not_strong, neg_log_kd_strong, neg_log_kd_not_strong, fuzzy_high]
    data_path = write_processed_jsonl_gz(tmp_path / "antibody_antigen_affinity.jsonl.gz", records)

    ds = OASSequenceDataset(data_path, split="train")
    kd_records = [record for record in ds.records if record.affinity_type == "kd"]
    assert kd_records[0].affinity_family == "ranking_regression"
    assert kd_records[0].affinity_strength_label == 1
    assert kd_records[0].affinity_strength_score == 10.0
    assert kd_records[1].affinity_strength_label is None
    assert kd_records[1].affinity_strength_score == 8.0

    neg_log_records = [record for record in ds.records if record.affinity_type == "-log KD"]
    assert neg_log_records[0].affinity_strength_label == 1
    assert neg_log_records[0].affinity_strength_score == 9.5
    assert neg_log_records[1].affinity_strength_label is None
    assert neg_log_records[1].affinity_strength_score == 8.5

    fuzzy_record = next(record for record in ds.records if record.affinity_type == "fuzzy")
    assert fuzzy_record.affinity_family == "unknown"
    assert fuzzy_record.affinity_strength_label is None
    assert fuzzy_record.affinity_strength_score is None


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

    assert set(batch.keys()) == {
        "input_ids",
        "attention_mask",
        "labels",
        "pair_labels",
        "pair_mask",
        "affinity_strength_labels",
        "affinity_strength_mask",
        "affinity_strength_scores",
        "affinity_strength_score_mask",
        "affinity_family_ids",
    }
    assert batch["input_ids"].shape == batch["attention_mask"].shape == batch["labels"].shape
    assert batch["input_ids"].dtype == torch.long
    assert batch["attention_mask"].dtype == torch.long
    assert batch["labels"].dtype == torch.long
    assert batch["affinity_strength_labels"].dtype == torch.long
    assert batch["affinity_strength_mask"].dtype == torch.bool
    assert batch["affinity_strength_scores"].dtype == torch.float32
    assert batch["affinity_strength_score_mask"].dtype == torch.bool
    assert batch["affinity_family_ids"].dtype == torch.long


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


def test_sampler_groups_paired_records_by_paired_bucket(tmp_path: Path, write_processed_jsonl_gz):
    records = [
        make_processed_pair_record("A" * 100, "C" * 90),
        make_processed_pair_record("A" * 101, "C" * 91),
        make_processed_pair_record("A" * 115, "C" * 100),
    ]
    data_path = write_processed_jsonl_gz(tmp_path / "paired.jsonl.gz", records)
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
        assert {r.chain_group for r in recs} == {"paired"}


def test_collator_can_shuffle_paired_examples(tmp_path: Path, tokenizer, write_processed_jsonl_gz):
    records = [
        make_processed_pair_record("H" * 10 + "AAAA", "L" * 10 + "CCCC", light_locus="IGK"),
        make_processed_pair_record("H" * 10 + "BBBB", "L" * 10 + "DDDD", light_locus="IGL"),
    ]
    data_path = write_processed_jsonl_gz(tmp_path / "paired.jsonl.gz", records)
    ds = OASSequenceDataset(data_path, split="train")

    collator = MLMCollator(
        tokenizer=tokenizer,
        max_length=64,
        mask_probability=0.3,
        hcdr3_span_probability=0.0,
        shuffle_pair_probability=1.0,
        rng_seed=42,
    )
    batch = collator([ds[0], ds[1]])

    assert batch["pair_mask"].tolist() == [True, True]
    assert batch["pair_labels"].tolist() == [0, 0]
    assert batch["input_ids"].shape == batch["labels"].shape


def test_collator_returns_affinity_strength_tensors(tmp_path: Path, tokenizer, write_processed_jsonl_gz):
    records = [
        make_processed_antibody_antigen_record(
            heavy_sequence="EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVS",
            light_sequence=None,
            antigen_sequence="MKTIIALSYIFCLVFADYKDDDDK",
            affinity_type="bool",
            binder_label=1,
        ),
        make_processed_antibody_antigen_record(
            heavy_sequence="QVQLQESGGGLVQAGGSLRLSCAASGFTFSSYAMGWFRQAPGKEREFVA",
            light_sequence=None,
            antigen_sequence="ACDEFGHIKLMNPQRSTVWY",
            dataset="flab-test",
            affinity_type="kd",
            affinity_raw="1e-8",
            processed_measurement_raw="1e-8",
            processed_measurement_float=1e-8,
            binder_label=None,
            is_strong_binder=False,
        ),
    ]
    data_path = write_processed_jsonl_gz(tmp_path / "antibody_antigen_batch.jsonl.gz", records)
    ds = OASSequenceDataset(data_path, split="train")

    collator = MLMCollator(
        tokenizer=tokenizer,
        max_length=64,
        mask_probability=0.15,
        hcdr3_span_probability=0.0,
        rng_seed=42,
    )
    batch = collator([ds[0], ds[1]])

    assert batch["affinity_strength_labels"].tolist() == [1, 0]
    assert batch["affinity_strength_mask"].tolist() == [True, False]
    assert batch["affinity_strength_score_mask"].tolist() == [True, True]
    assert batch["affinity_family_ids"].tolist() == [1, 3]


def test_antibody_antigen_collator_returns_dual_stream_batch(tmp_path: Path, tokenizer, write_processed_jsonl_gz):
    records = [
        make_processed_antibody_antigen_record(
            heavy_sequence="EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVS",
            light_sequence=None,
            antigen_sequence="MKTIIALSYIFCLVFADYKDDDDK",
            target_key="uniprot:p11111",
            record_id="antigen-1",
        ),
        make_processed_antibody_antigen_record(
            heavy_sequence="QVQLQESGGGLVQAGGSLRLSCAASGFTFSSYAMGWFRQAPGKEREFVA",
            light_sequence=None,
            antigen_sequence="ACDEFGHIKLMNPQRSTVWY",
            target_key="uniprot:p22222",
            record_id="antigen-2",
        ),
    ]
    data_path = write_processed_jsonl_gz(tmp_path / "antibody_antigen_dual.jsonl.gz", records)
    ds = OASSequenceDataset(data_path, split="train")

    collator = AntibodyAntigenCollator(
        tokenizer=tokenizer,
        max_length=64,
        mask_probability=0.15,
        hcdr3_span_probability=0.0,
        shuffle_antigen_probability=0.0,
        rng_seed=42,
    )
    batch = collator([ds[0], ds[1]])

    assert set(batch.keys()) == {
        "antibody_input_ids",
        "antibody_attention_mask",
        "antibody_labels",
        "antigen_input_ids",
        "antigen_attention_mask",
        "compatibility_labels",
        "compatibility_mask",
        "is_shuffled_antigen",
        "record_ids",
        "target_keys",
        "dataset_names",
        "antibody_format_groups",
        "antigen_length_buckets",
    }
    assert batch["antibody_input_ids"].shape == batch["antibody_attention_mask"].shape == batch["antibody_labels"].shape
    assert batch["antigen_input_ids"].shape == batch["antigen_attention_mask"].shape
    assert batch["compatibility_labels"].tolist() == [1, 1]
    assert batch["compatibility_mask"].tolist() == [True, True]
    assert batch["is_shuffled_antigen"].tolist() == [False, False]


def test_antibody_antigen_real_label_collator_uses_binder_labels_without_shuffling(
    tmp_path: Path,
    tokenizer,
    write_processed_jsonl_gz,
):
    records = [
        make_processed_antibody_antigen_record(
            heavy_sequence="EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVS",
            light_sequence=None,
            antigen_sequence="MKTIIALSYIFCLVFADYKDDDDK",
            target_key="uniprot:p11111",
            record_id="antigen-1",
            binder_label=1,
            is_strong_binder=True,
        ),
        make_processed_antibody_antigen_record(
            heavy_sequence="QVQLQESGGGLVQAGGSLRLSCAASGFTFSSYAMGWFRQAPGKEREFVA",
            light_sequence=None,
            antigen_sequence="ACDEFGHIKLMNPQRSTVWY",
            target_key="uniprot:p22222",
            record_id="antigen-2",
            binder_label=0,
            is_strong_binder=False,
        ),
    ]
    data_path = write_processed_jsonl_gz(tmp_path / "antibody_antigen_real_labels.jsonl.gz", records)
    ds = OASSequenceDataset(data_path, split="train")

    collator = AntibodyAntigenRealLabelCollator(
        tokenizer=tokenizer,
        max_length=64,
        mask_probability=0.15,
        hcdr3_span_probability=0.0,
        shuffle_antigen_probability=1.0,
        rng_seed=42,
    )
    batch = collator([ds[0], ds[1]])

    assert batch["compatibility_mask"].tolist() == [True, True]
    assert batch["compatibility_labels"].tolist() == [1, 0]
    assert batch["is_shuffled_antigen"].tolist() == [False, False]
    assert batch["target_keys"] == ["uniprot:p11111", "uniprot:p22222"]


def test_antibody_antigen_collator_uses_configurable_shuffle_fraction(tmp_path: Path, tokenizer, write_processed_jsonl_gz):
    records = [
        make_processed_antibody_antigen_record(
            heavy_sequence="EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVS",
            light_sequence=None,
            antigen_sequence="MKTIIALSYIFCLVFADYKDDDDK",
            target_key="uniprot:p11111",
            record_id="antigen-1",
        ),
        make_processed_antibody_antigen_record(
            heavy_sequence="QVQLQESGGGLVQAGGSLRLSCAASGFTFSSYAMGWFRQAPGKEREFVA",
            light_sequence=None,
            antigen_sequence="ACDEFGHIKLMNPQRSTVWY",
            target_key="uniprot:p22222",
            record_id="antigen-2",
        ),
    ]
    data_path = write_processed_jsonl_gz(tmp_path / "antibody_antigen_shuffle.jsonl.gz", records)
    ds = OASSequenceDataset(data_path, split="train")

    collator = AntibodyAntigenCollator(
        tokenizer=tokenizer,
        max_length=64,
        mask_probability=0.15,
        hcdr3_span_probability=0.0,
        shuffle_antigen_probability=1.0,
        rng_seed=42,
    )
    batch = collator([ds[0], ds[1]])

    assert batch["compatibility_mask"].tolist() == [True, True]
    assert batch["compatibility_labels"].tolist() == [0, 0]
    assert batch["is_shuffled_antigen"].tolist() == [True, True]
    assert batch["target_keys"][0] == "uniprot:p22222"
    assert batch["target_keys"][1] == "uniprot:p11111"


def test_antibody_antigen_collator_only_uses_strong_binders_for_compatibility(
    tmp_path: Path,
    tokenizer,
    write_processed_jsonl_gz,
):
    records = [
        make_processed_antibody_antigen_record(
            heavy_sequence="EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVS",
            light_sequence=None,
            antigen_sequence="MKTIIALSYIFCLVFADYKDDDDK",
            target_key="uniprot:p11111",
            record_id="antigen-1",
            is_strong_binder=True,
        ),
        make_processed_antibody_antigen_record(
            heavy_sequence="QVQLQESGGGLVQAGGSLRLSCAASGFTFSSYAMGWFRQAPGKEREFVA",
            light_sequence=None,
            antigen_sequence="ACDEFGHIKLMNPQRSTVWY",
            target_key="uniprot:p22222",
            record_id="antigen-2",
            is_strong_binder=False,
            binder_label=0,
        ),
    ]
    data_path = write_processed_jsonl_gz(tmp_path / "antibody_antigen_strong_only.jsonl.gz", records)
    ds = OASSequenceDataset(data_path, split="train")

    collator = AntibodyAntigenCollator(
        tokenizer=tokenizer,
        max_length=64,
        mask_probability=0.15,
        hcdr3_span_probability=0.0,
        shuffle_antigen_probability=1.0,
        rng_seed=42,
    )
    batch = collator([ds[0], ds[1]])

    assert batch["compatibility_mask"].tolist() == [True, False]
    assert batch["compatibility_labels"].tolist() == [1, 0]
    assert batch["is_shuffled_antigen"].tolist() == [False, False]
