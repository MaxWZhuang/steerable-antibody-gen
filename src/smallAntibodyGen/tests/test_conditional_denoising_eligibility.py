"""
Contract tests for conditional-denoising eligibility (J22c).

Contract:  specs/conditional_denoising_eligibility.md
Rationale: specs/decisions/0001-conditional-denoising-eligibility.md

A row's eligibility decides whether it contributes antigen-conditioned MLM
targets. It is independent of whether that row contributes compatibility or
strength supervision, and it never changes the corrupted input.

The defect being fixed: `AntibodyAntigenRealLabelCollator` overrides only
`_build_antibody_antigen_batch`, so the inherited `__call__` masked tokens for
every row. A measured NONBINDER therefore became a positive reconstruction
target for the policy under the antigen it is labeled not to bind.

These tests are written to fail against the pre-fix behavior.

One structural note on the "byte-identical to pre-change behavior" requirements.
Pre-change behavior is exactly `all_filtered_rows`: every row eligible, no labels
cleared. So the checkable form of that requirement is that the affected rows are
byte-identical BETWEEN the two policies under one seed -- which also proves the
RNG stream is untouched, since `_mask_tokens` draws per selected position.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from smallAntibodyGen.data.MLMCollator import (
    CONDITIONAL_DENOISING_ELIGIBILITY_POLICIES,
    MLM_IGNORE_INDEX,
    AntibodyAntigenCollator,
    AntibodyAntigenRealLabelCollator,
    OASSequenceDataset,
)

HEAVY = "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYAMHWVRQAPGQGLEWMGWINAGNGNTKYSQKFQGRVTITRDTSASTAYMELSSLRSEDTAVYYCARDRSTFDYWGQGTLVTVSS"
ANTIGEN = "MKTIIALSYIFCLVFADYKDDDDKGSHMTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHQYREQIKRVKDSDDVPMVLVGNKCDL"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _load_script(project_root: Path, name: str):
    """Import a scripts/*.py module by name (scripts/ added to sys.path)."""
    scripts_dir = project_root.parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location(name, scripts_dir / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _antigen_record(
    record_id: str,
    *,
    binder_label: int | None,
    affinity_type: str = "bool",
    processed_measurement: object = None,
    strength_quantile: float | None = None,
    is_strong_binder: bool | None = None,
    heavy: str = HEAVY,
) -> dict:
    """
    One processed antibody-antigen row.

    `binder_label` is populated by the producer ONLY for `affinity_type == "bool"`
    rows, so a KD / -log KD / fuzzy row legitimately carries `binder_label=None`
    while still being a real measured strong binder. That asymmetry is the whole
    reason the policy is per-stage rather than a single predicate.
    """
    record = {
        "record_id": record_id,
        "sequence": heavy,
        "sequence_heavy": heavy,
        "sequence_light": None,
        "sequence_antigen": ANTIGEN,
        "locus": "PAIRED_ANTIGEN",
        "chain_group": "paired_antigen",
        "split": "train",
        "length": len(heavy),
        "target_key": "uniprot:p12345",
        "target_name": "test_target",
        "target_pdb": "1abc",
        "target_uniprot": "P12345",
        "dataset": "asd-test",
        "confidence": "very_high",
        "affinity_type": affinity_type,
        "affinity_raw": None,
        "processed_measurement_raw": (
            None if processed_measurement is None else str(processed_measurement)
        ),
        "processed_measurement_float": (
            processed_measurement if isinstance(processed_measurement, float) else None
        ),
        "binder_label": binder_label,
        "is_nanobody": True,
        "scfv": False,
        "cdr3_aa_heavy": "ARDRSTFDY",
        "cdr3_start_aa_heavy": 98,
        "cdr3_end_aa_heavy": 107,
        "cdr3_aa_light": None,
        "cdr3_start_aa_light": None,
        "cdr3_end_aa_light": None,
        "heavy_locus": "IGH",
        "light_locus": None,
        "is_paired": False,
        "metadata": {},
        "source_file": "tiny_antigen.parquet",
    }
    if strength_quantile is not None:
        record["affinity_strength_quantile"] = strength_quantile
    if is_strong_binder is not None:
        record["is_strong_binder"] = is_strong_binder
    return record


def _dataset(write_processed_jsonl_gz, tmp_path, records, name="antigen.jsonl.gz"):
    return OASSequenceDataset(
        write_processed_jsonl_gz(tmp_path / name, records), split="train"
    )


def _real_label_collator(tokenizer, policy, **overrides):
    params = dict(
        tokenizer=tokenizer,
        max_length=192,
        mask_probability=0.15,
        hcdr3_span_probability=0.0,
        shuffle_antigen_probability=0.0,
        rng_seed=42,
        conditional_denoising_eligibility=policy,
    )
    params.update(overrides)
    return AntibodyAntigenRealLabelCollator(**params)


# --------------------------------------------------------------------------- #
# Policy value contract
# --------------------------------------------------------------------------- #


def test_policy_names_are_exactly_the_two_approved_values():
    # A named, serializable value -- never a callable -- so it can be recorded in
    # a config, saved in a checkpoint, and hashed into a run fingerprint.
    assert CONDITIONAL_DENOISING_ELIGIBILITY_POLICIES == (
        "all_filtered_rows",
        "binary_binders_only",
    )


def test_collator_rejects_an_unknown_policy(tokenizer):
    with pytest.raises(ValueError, match="conditional_denoising_eligibility"):
        _real_label_collator(tokenizer, "binders_only")


def test_base_collator_defaults_to_all_filtered_rows(tokenizer):
    # The default exists for backward compatibility only; production sites pass
    # the policy explicitly (see the wiring tests below).
    collator = AntibodyAntigenCollator(tokenizer=tokenizer, max_length=192)
    assert collator.conditional_denoising_eligibility == "all_filtered_rows"


# --------------------------------------------------------------------------- #
# binary_binders_only
# --------------------------------------------------------------------------- #


def test_nonbinder_loses_mlm_but_keeps_compatibility_and_its_corrupted_input(
    tmp_path, tokenizer, write_processed_jsonl_gz
):
    # THE defect. Pre-fix, row 1 (a measured nonbinder) carried real MLM labels:
    # a positive reconstruction target under the antigen it does not bind.
    records = [
        _antigen_record("binder", binder_label=1, processed_measurement=1.0),
        _antigen_record("nonbinder", binder_label=0, processed_measurement=0.0),
    ]
    ds = _dataset(write_processed_jsonl_gz, tmp_path, records)
    rows = [ds[0], ds[1]]

    strict = _real_label_collator(tokenizer, "binary_binders_only")(rows)
    legacy = _real_label_collator(tokenizer, "all_filtered_rows")(rows)

    # Row 1 has no conditional MLM signal at all.
    assert bool(strict["conditional_denoising_eligible"][0]) is True
    assert bool(strict["conditional_denoising_eligible"][1]) is False
    assert torch.equal(
        strict["antibody_labels"][1],
        torch.full_like(strict["antibody_labels"][1], MLM_IGNORE_INDEX),
    )
    assert not strict["hcdr3_target_mask"][1].any()
    # Pre-fix it did have targets -- the test is not vacuous.
    assert (legacy["antibody_labels"][1] != MLM_IGNORE_INDEX).any()

    # ...but it keeps its compatibility supervision,
    assert bool(strict["compatibility_mask"][1]) is True
    assert int(strict["compatibility_labels"][1]) == 0
    # ...and its corrupted input is untouched, so compatibility still trains on
    # the same noisy state it would have seen before.
    assert torch.equal(strict["antibody_input_ids"], legacy["antibody_input_ids"])


def test_binder_row_is_byte_identical_across_the_policy_change(
    tmp_path, tokenizer, write_processed_jsonl_gz
):
    # Eligible rows must be bit-for-bit what they were before the knob existed.
    # That holds only because eligibility is applied AFTER `_mask_tokens`, which
    # draws per selected position: filtering rows earlier would shift every
    # subsequent draw.
    records = [
        _antigen_record("binder-a", binder_label=1, processed_measurement=1.0),
        _antigen_record("nonbinder", binder_label=0, processed_measurement=0.0),
        _antigen_record("binder-b", binder_label=1, processed_measurement=1.0),
    ]
    ds = _dataset(write_processed_jsonl_gz, tmp_path, records)
    rows = [ds[0], ds[1], ds[2]]

    strict_collator = _real_label_collator(tokenizer, "binary_binders_only")
    legacy_collator = _real_label_collator(tokenizer, "all_filtered_rows")
    strict = strict_collator(rows)
    legacy = legacy_collator(rows)

    for eligible_row in (0, 2):
        for key in ("antibody_input_ids", "antibody_labels", "hcdr3_target_mask"):
            assert torch.equal(strict[key][eligible_row], legacy[key][eligible_row]), (
                key,
                eligible_row,
            )

    # Same RNG position after the call, not just the same output. An output can
    # match while the stream has drifted.
    assert strict_collator.rng.getstate() == legacy_collator.rng.getstate()


def test_unlabeled_row_receives_neither_objective(
    tmp_path, tokenizer, write_processed_jsonl_gz
):
    records = [
        _antigen_record("binder", binder_label=1, processed_measurement=1.0),
        _antigen_record("unlabeled", binder_label=None, affinity_type="kd"),
    ]
    ds = _dataset(write_processed_jsonl_gz, tmp_path, records)
    batch = _real_label_collator(tokenizer, "binary_binders_only")([ds[0], ds[1]])

    assert bool(batch["conditional_denoising_eligible"][1]) is False
    assert bool(batch["compatibility_mask"][1]) is False
    assert torch.equal(
        batch["antibody_labels"][1],
        torch.full_like(batch["antibody_labels"][1], MLM_IGNORE_INDEX),
    )


def test_graded_strength_row_without_a_binary_label_keeps_its_strength_target(
    tmp_path, tokenizer, write_processed_jsonl_gz
):
    # Stage-3 rows admitted through `include_strength_rows` carry a quantile but
    # NO binary label. They are ineligible for conditional MLM; their strength
    # targets are unaffected. Eligibility and strength are separate questions.
    records = [
        _antigen_record("binder", binder_label=1, processed_measurement=1.0),
        _antigen_record(
            "graded", binder_label=None, affinity_type="kd", strength_quantile=0.9
        ),
    ]
    ds = _dataset(write_processed_jsonl_gz, tmp_path, records)
    assert ds[1].affinity_strength_quantile == 0.9  # fixture is not vacuous
    batch = _real_label_collator(tokenizer, "binary_binders_only")([ds[0], ds[1]])

    assert bool(batch["conditional_denoising_eligible"][1]) is False
    assert torch.equal(
        batch["antibody_labels"][1],
        torch.full_like(batch["antibody_labels"][1], MLM_IGNORE_INDEX),
    )
    assert bool(batch["strength_mask"][1]) is True
    assert float(batch["strength_targets"][1]) == pytest.approx(0.9)


def test_all_nonbinder_batch_is_a_finite_zero_and_does_not_raise(
    tmp_path, tokenizer, write_processed_jsonl_gz
):
    # An all-nonbinder batch is LEGITIMATE under this policy and must still
    # contribute compatibility supervision. It is counted, not raised.
    from smallAntibodyGen.models.mlm import AntibodyAntigenCrossAttention, MLMConfig

    records = [
        _antigen_record("nb-a", binder_label=0, processed_measurement=0.0),
        _antigen_record("nb-b", binder_label=0, processed_measurement=0.0),
    ]
    ds = _dataset(write_processed_jsonl_gz, tmp_path, records)
    batch = _real_label_collator(tokenizer, "binary_binders_only")([ds[0], ds[1]])

    assert not batch["conditional_denoising_eligible"].any()
    assert bool(batch["compatibility_mask"].all()) is True

    model = AntibodyAntigenCrossAttention(
        MLMConfig(
            vocab_size=tokenizer.vocab_size,
            pad_token_id=tokenizer.pad_id,
            max_length=192,
            d_model=32,
            n_heads=4,
            n_layers=1,
            d_ff=64,
            dropout=0.0,
        )
    )
    labels = batch["antibody_labels"]
    logits = torch.randn(labels.shape[0], labels.shape[1], tokenizer.vocab_size)
    loss = model.compute_mlm_loss(logits, labels)
    assert torch.isfinite(loss)  # a finite zero, NOT NaN -- which is why the
    assert float(loss) == 0.0  # failure is invisible without an explicit count.


# --------------------------------------------------------------------------- #
# all_filtered_rows
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "affinity_type,measurement",
    [("kd", 0.4), ("-log KD", 9.5), ("fuzzy", "H")],
)
def test_unlabeled_assay_types_stay_fully_eligible(
    tmp_path, tokenizer, write_processed_jsonl_gz, affinity_type, measurement
):
    # Its own fixture, deliberately: Stage 4 filters on `is_strong_binder`, and
    # `binder_label` is populated only for `affinity_type == "bool"` rows. If a
    # single `binder_label == 1` predicate were used for both stages, every row
    # here would silently lose all MLM signal while the loss curve stayed
    # plausible. This must not be an incidental consequence of another test.
    records = [
        _antigen_record(
            f"strong-{affinity_type}",
            binder_label=None,
            affinity_type=affinity_type,
            processed_measurement=measurement,
            is_strong_binder=True,
        )
    ]
    ds = _dataset(write_processed_jsonl_gz, tmp_path, records)
    assert ds[0].binder_label is None
    assert ds[0].is_strong_binder is True

    batch = _real_label_collator(tokenizer, "all_filtered_rows")([ds[0]])
    assert bool(batch["conditional_denoising_eligible"][0]) is True
    assert (batch["antibody_labels"][0] != MLM_IGNORE_INDEX).any()

    # ...and the same row under the Stage-3 policy would lose everything.
    strict = _real_label_collator(tokenizer, "binary_binders_only")([ds[0]])
    assert not strict["conditional_denoising_eligible"][0]


def test_zero_eligible_rows_under_all_filtered_rows_raises_immediately(
    tmp_path, tokenizer, write_processed_jsonl_gz
):
    # Under this policy every admitted row is eligible by construction, so zero
    # eligible rows in a nonempty batch can only mean incorrect wiring.
    records = [_antigen_record("r", binder_label=1, processed_measurement=1.0)]
    ds = _dataset(write_processed_jsonl_gz, tmp_path, records)

    collator = _real_label_collator(tokenizer, "all_filtered_rows")
    # Force the only way this state is reachable: a broken predicate.
    collator._is_conditional_denoising_eligible = lambda item: False
    with pytest.raises(ValueError, match="incorrect wiring"):
        collator([ds[0]])


def test_base_antibody_antigen_collator_is_unchanged(
    tmp_path, tokenizer, write_processed_jsonl_gz
):
    # The synthetic shuffled-antigen path is explicitly out of scope: under
    # `all_filtered_rows` those rows stay eligible.
    records = []
    for i in range(4):
        record = _antigen_record(
            f"r{i}", binder_label=1, processed_measurement=1.0, is_strong_binder=True
        )
        # Distinct targets AND distinct antigen sequences: `_find_antigen_donor`
        # excludes the row's own target and its own antigen, so identical rows
        # have no valid donor and nothing would be shuffled.
        record["target_key"] = f"uniprot:p0000{i}"
        record["target_uniprot"] = f"P0000{i}"
        record["sequence_antigen"] = ANTIGEN[: len(ANTIGEN) - i] + "A" * i
        records.append(record)
    ds = _dataset(write_processed_jsonl_gz, tmp_path, records)
    rows = [ds[i] for i in range(4)]

    collator = AntibodyAntigenCollator(
        tokenizer=tokenizer,
        max_length=192,
        hcdr3_span_probability=0.0,
        shuffle_antigen_probability=1.0,
        rng_seed=42,
    )
    batch = collator(rows)
    assert bool(batch["conditional_denoising_eligible"].all()) is True
    assert bool(batch["is_shuffled_antigen"].any()) is True
    # Shuffled rows keep their conditional MLM targets, unchanged from before.
    for row in range(4):
        if batch["is_shuffled_antigen"][row]:
            assert (batch["antibody_labels"][row] != MLM_IGNORE_INDEX).any()


# --------------------------------------------------------------------------- #
# Shared: metadata coupling
# --------------------------------------------------------------------------- #


def test_hcdr3_metadata_agrees_with_the_post_eligibility_target_mask(
    tmp_path, tokenizer, write_processed_jsonl_gz
):
    # Clearing labels without clearing the target mask would leave HCDR3 target
    # counts and mask-fraction bins reporting positions that contribute no loss,
    # silently corrupting the calibration bins guide training depends on.
    records = [
        _antigen_record("binder", binder_label=1, processed_measurement=1.0),
        _antigen_record("nonbinder", binder_label=0, processed_measurement=0.0),
    ]
    ds = _dataset(write_processed_jsonl_gz, tmp_path, records)
    batch = _real_label_collator(
        tokenizer, "binary_binders_only", hcdr3_span_probability=1.0
    )([ds[0], ds[1]])

    labelled = batch["antibody_labels"] != MLM_IGNORE_INDEX
    # Every reported HCDR3 target position must actually carry a label.
    assert torch.equal(batch["hcdr3_target_mask"] & labelled, batch["hcdr3_target_mask"])
    assert not batch["hcdr3_target_mask"][1].any()
    # The eligible row still has HCDR3 targets, so the assertion above is not
    # trivially satisfied by an all-false mask.
    assert batch["hcdr3_target_mask"][0].any()


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "training_stage,expected_class",
    [
        ("antigen_real_label_refine", AntibodyAntigenRealLabelCollator),
        ("antigen_hcdr3_infill_refine", AntibodyAntigenRealLabelCollator),
        ("antigen_refine", AntibodyAntigenCollator),
    ],
)
@pytest.mark.parametrize("policy", CONDITIONAL_DENOISING_ELIGIBILITY_POLICIES)
def test_train_and_eval_loader_builders_both_pass_the_policy_explicitly(
    project_root, tmp_path, tokenizer, write_processed_jsonl_gz, training_stage, expected_class, policy
):
    # A default reaching production through omission is a defect, so BOTH builders
    # must forward the config value for BOTH collator classes.
    mlm_train = _load_script(project_root, "mlm_train")
    records = [
        _antigen_record(f"r{i}", binder_label=1, processed_measurement=1.0, is_strong_binder=True)
        for i in range(4)
    ]
    data_path = write_processed_jsonl_gz(tmp_path / "antigen.jsonl.gz", records)
    cfg = mlm_train.TrainConfig(
        data_path=str(data_path),
        training_stage=training_stage,
        batch_size=2,
        eval_batch_size=2,
        max_length=192,
        conditional_denoising_eligibility=policy,
    )
    dataset = OASSequenceDataset(str(data_path), split="train")

    train_loader = mlm_train.build_train_loader(dataset, tokenizer, cfg, epoch=0)
    eval_loader = mlm_train.build_eval_loader(dataset, tokenizer, cfg)
    for loader in (train_loader, eval_loader):
        assert isinstance(loader.collate_fn, expected_class)
        assert loader.collate_fn.conditional_denoising_eligibility == policy


def test_stage_three_defaults_to_binary_binders_only_and_stage_four_does_not(
    project_root, tmp_path
):
    mlm_train = _load_script(project_root, "mlm_train")
    data_path = tmp_path / "tiny.jsonl.gz"
    data_path.write_text("", encoding="utf-8")

    stage3 = mlm_train.parse_args(
        [
            "--data-path", str(data_path),
            "--training-stage", "antigen_real_label_refine",
            "--init-checkpoint", "parent.pt",
        ]
    )
    assert stage3.conditional_denoising_eligibility == "binary_binders_only"

    stage4 = mlm_train.parse_args(
        [
            "--data-path", str(data_path),
            "--training-stage", "antigen_hcdr3_infill_refine",
            "--init-checkpoint", "parent.pt",
        ]
    )
    assert stage4.conditional_denoising_eligibility == "all_filtered_rows"

    # An explicit CLI value still wins over the stage default.
    override = mlm_train.parse_args(
        [
            "--data-path", str(data_path),
            "--training-stage", "antigen_real_label_refine",
            "--init-checkpoint", "parent.pt",
            "--conditional-denoising-eligibility", "all_filtered_rows",
        ]
    )
    assert override.conditional_denoising_eligibility == "all_filtered_rows"


def test_config_rejects_an_unknown_policy(project_root):
    mlm_train = _load_script(project_root, "mlm_train")
    cfg = mlm_train.TrainConfig(
        data_path="x", conditional_denoising_eligibility="binders_only"
    )
    with pytest.raises(ValueError, match="conditional_denoising_eligibility"):
        cfg.validate()


def test_policy_round_trips_through_config_json_and_checkpoint(project_root, tmp_path):
    # The policy is a TrainConfig field, so it flows into `train_config` in every
    # saved checkpoint and into the run's config JSON without further work. Two
    # policies produce different training populations from otherwise
    # identical-looking configs, so this is what a fingerprint check reads.
    import json
    from dataclasses import asdict

    mlm_train = _load_script(project_root, "mlm_train")
    cfg = mlm_train.TrainConfig(
        data_path="x",
        training_stage="antigen_real_label_refine",
        init_checkpoint="parent.pt",
        conditional_denoising_eligibility="binary_binders_only",
    )
    cfg.validate()

    assert asdict(cfg)["conditional_denoising_eligibility"] == "binary_binders_only"
    assert (
        json.loads(json.dumps(asdict(cfg)))["conditional_denoising_eligibility"]
        == "binary_binders_only"
    )

    model = torch.nn.Linear(4, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt_path = tmp_path / "last.pt"
    mlm_train.save_checkpoint(ckpt_path, model, opt, cfg, epoch=1, val_loss=0.5)
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert (
        payload["train_config"]["conditional_denoising_eligibility"]
        == "binary_binders_only"
    )


def test_stage_three_config_file_sets_the_policy_explicitly(project_root):
    # The approved contract requires production sites to be explicit rather than
    # relying on the stage default.
    yaml = pytest.importorskip("yaml")
    config_path = project_root.parents[1] / "configs" / "refine_antigen_real_label.yaml"
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert data["conditional_denoising_eligibility"] == "binary_binders_only"


# --------------------------------------------------------------------------- #
# Zero-eligible: preflight and whole-epoch
# --------------------------------------------------------------------------- #


def test_preflight_summary_counts_eligible_rows_per_policy(
    project_root, tmp_path, write_processed_jsonl_gz
):
    mlm_train = _load_script(project_root, "mlm_train")
    records = [
        _antigen_record("nb-a", binder_label=0, processed_measurement=0.0),
        _antigen_record("nb-b", binder_label=0, processed_measurement=0.0),
        _antigen_record("b", binder_label=1, processed_measurement=1.0),
    ]
    ds = _dataset(write_processed_jsonl_gz, tmp_path, records)

    strict = mlm_train.summarize_conditional_denoising_eligibility(
        ds, "binary_binders_only"
    )
    assert strict == {"total": 3, "eligible": 1}

    legacy = mlm_train.summarize_conditional_denoising_eligibility(ds, "all_filtered_rows")
    assert legacy == {"total": 3, "eligible": 3}


def test_preflight_summary_agrees_with_the_collator_predicate(
    project_root, tmp_path, tokenizer, write_processed_jsonl_gz
):
    # A divergence would mean preflight blesses a population the collator then
    # discards.
    mlm_train = _load_script(project_root, "mlm_train")
    records = [
        _antigen_record("b", binder_label=1, processed_measurement=1.0),
        _antigen_record("nb", binder_label=0, processed_measurement=0.0),
        _antigen_record("kd", binder_label=None, affinity_type="kd", is_strong_binder=True),
    ]
    ds = _dataset(write_processed_jsonl_gz, tmp_path, records)
    rows = [ds[0], ds[1], ds[2]]

    for policy in CONDITIONAL_DENOISING_ELIGIBILITY_POLICIES:
        summary = mlm_train.summarize_conditional_denoising_eligibility(ds, policy)
        batch = _real_label_collator(tokenizer, policy)(rows)
        assert summary["eligible"] == int(
            batch["conditional_denoising_eligible"].sum()
        ), policy


def test_an_all_nonbinder_epoch_fails(
    project_root, tmp_path, tokenizer, write_processed_jsonl_gz
):
    # Per BATCH this is legitimate. Per EPOCH it means the conditional policy
    # learned nothing, and the loss curve would not show it.
    mlm_train = _load_script(project_root, "mlm_train")
    records = [
        _antigen_record(f"nb{i}", binder_label=0, processed_measurement=0.0)
        for i in range(8)
    ]
    data_path = write_processed_jsonl_gz(tmp_path / "nonbinders.jsonl.gz", records)
    cfg = mlm_train.TrainConfig(
        data_path=str(data_path),
        training_stage="antigen_real_label_refine",
        conditional_denoising_eligibility="binary_binders_only",
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
    )
    device = torch.device("cpu")
    dataset = OASSequenceDataset(str(data_path), split="train")
    model = mlm_train.build_model(tokenizer, cfg, device)
    out_dir = tmp_path / "run"
    out_dir.mkdir()

    with pytest.raises(ValueError, match="eligible for antigen-conditioned MLM"):
        mlm_train.train_one_epoch(
            model=model,
            train_dataset=dataset,
            tokenizer=tokenizer,
            optimizer=mlm_train.build_optimizer(model, cfg),
            scaler=torch.amp.GradScaler("cuda", enabled=False),
            scheduler=mlm_train.build_lr_scheduler(
                mlm_train.build_optimizer(model, cfg), cfg
            ),
            cfg=cfg,
            device=device,
            epoch=0,
            output_dir=out_dir,
            best_val_loss=float("inf"),
        )


def test_a_binder_bearing_epoch_reports_its_eligible_census(
    project_root, tmp_path, tokenizer, write_processed_jsonl_gz
):
    mlm_train = _load_script(project_root, "mlm_train")
    records = [
        _antigen_record(
            f"r{i}", binder_label=(1 if i % 2 == 0 else 0), processed_measurement=float(i % 2 == 0)
        )
        for i in range(8)
    ]
    data_path = write_processed_jsonl_gz(tmp_path / "mixed.jsonl.gz", records)
    cfg = mlm_train.TrainConfig(
        data_path=str(data_path),
        training_stage="antigen_real_label_refine",
        conditional_denoising_eligibility="binary_binders_only",
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
    )
    device = torch.device("cpu")
    dataset = OASSequenceDataset(str(data_path), split="train")
    model = mlm_train.build_model(tokenizer, cfg, device)
    optimizer = mlm_train.build_optimizer(model, cfg)
    out_dir = tmp_path / "run"
    out_dir.mkdir()

    metrics = mlm_train.train_one_epoch(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        scheduler=mlm_train.build_lr_scheduler(optimizer, cfg),
        cfg=cfg,
        device=device,
        epoch=0,
        output_dir=out_dir,
        best_val_loss=float("inf"),
    )

    assert metrics["conditional_denoising_policy_rows"] == 8.0
    assert metrics["conditional_denoising_eligible_rows"] == 4.0
    assert metrics["conditional_denoising_eligible_tokens"] > 0.0


# --------------------------------------------------------------------------- #
# Findings from independent review of the first implementation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_all_ignored_mlm_loss_is_a_finite_zero_even_in_fp16(tokenizer, dtype):
    """
    The contract's claim that an all-ignored batch yields "a finite zero, not
    NaN" is true in fp32 only, unless the guard promotes explicitly.

    Under AMP the LM head emits fp16, and fp16 saturates at 65504. A Stage-3
    batch is 16 x 192 x 35 = 107,520 logits, so once the head acquires a mean
    logit magnitude above ~0.61 the guard's `logits.sum()` overflows to `inf`
    and `inf * 0.0` is NaN -- which `train_one_epoch` then turns into a
    FloatingPointError, killing the run mid-training rather than at smoke time.

    `binary_binders_only` is what makes this branch routine on Stage 3: before
    it, every row carried MLM targets and the branch was unreachable there.
    """
    from smallAntibodyGen.models.mlm import AntibodyAntigenCrossAttention, MLMConfig

    model = AntibodyAntigenCrossAttention(
        MLMConfig(
            vocab_size=tokenizer.vocab_size,
            pad_token_id=tokenizer.pad_id,
            max_length=192,
            d_model=32,
            n_heads=4,
            n_layers=1,
            d_ff=64,
            dropout=0.0,
        )
    )
    # A systematic offset, as a trained head acquires. Overflows the fp16 sum.
    logits = (torch.randn(16, 192, tokenizer.vocab_size) + 1.2).to(dtype)
    labels = torch.full((16, 192), MLM_IGNORE_INDEX, dtype=torch.long)

    loss = model.compute_mlm_loss(logits, labels)
    assert torch.isfinite(loss), f"{dtype}: pre-fix this was NaN via inf * 0.0"
    assert float(loss) == 0.0


def test_stage_default_holds_for_every_construction_path(project_root, tmp_path):
    """
    Resolving the stage default in `parse_args` alone is not enough: a Stage-3
    `TrainConfig` built directly -- in a test, or by
    `hcdr3_infill.config_from_checkpoint` -- would silently keep pre-fix
    behavior. The contract is explicit that a default reaching production
    through omission is a defect.
    """
    mlm_train = _load_script(project_root, "mlm_train")

    direct = mlm_train.TrainConfig(
        data_path="x",
        training_stage="antigen_real_label_refine",
        init_checkpoint="parent.pt",
    )
    assert direct.conditional_denoising_eligibility == "binary_binders_only"

    # An explicit value still wins over the stage default.
    explicit = mlm_train.TrainConfig(
        data_path="x",
        training_stage="antigen_real_label_refine",
        init_checkpoint="parent.pt",
        conditional_denoising_eligibility="all_filtered_rows",
    )
    assert explicit.conditional_denoising_eligibility == "all_filtered_rows"

    # Every other stage keeps all_filtered_rows.
    for stage in ("base", "paired_refine", "antigen_refine", "antigen_hcdr3_infill_refine"):
        cfg = mlm_train.TrainConfig(
            data_path="x",
            training_stage=stage,
            init_checkpoint=None if stage == "base" else "parent.pt",
        )
        assert cfg.conditional_denoising_eligibility == "all_filtered_rows", stage

    # And the resolved value -- never the `None` sentinel -- is what serializes.
    from dataclasses import asdict

    assert asdict(direct)["conditional_denoising_eligibility"] == "binary_binders_only"


def test_binary_binders_only_is_rejected_on_a_stage_with_no_antigen(project_root):
    mlm_train = _load_script(project_root, "mlm_train")
    cfg = mlm_train.TrainConfig(
        data_path="x",
        training_stage="paired_refine",
        init_checkpoint="parent.pt",
        conditional_denoising_eligibility="binary_binders_only",
    )
    with pytest.raises(ValueError, match="only .* for antigen stages"):
        cfg.validate()


def test_base_collator_output_is_byte_identical_to_the_pre_change_collator(
    project_root, tmp_path, tokenizer, write_processed_jsonl_gz
):
    """
    The weaker form of this test -- "all rows are eligible" on an all-binder
    fixture -- passes even if the base default is flipped to
    `binary_binders_only`. This one loads the collator as it existed at HEAD~
    and compares tensors, so it fails on any behavioral drift.
    """
    import importlib.util
    import subprocess

    repo_root = project_root.parents[1]
    previous = subprocess.run(
        ["git", "show", "HEAD:src/smallAntibodyGen/data/MLMCollator.py"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if previous.returncode != 0:
        pytest.skip("pre-change MLMCollator.py is not retrievable from git")

    legacy_path = tmp_path / "legacy_collator.py"
    legacy_path.write_text(previous.stdout, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("legacy_collator", legacy_path)
    legacy = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = legacy
    spec.loader.exec_module(legacy)

    records = []
    for i in range(4):
        record = _antigen_record(
            f"r{i}", binder_label=(1 if i % 2 == 0 else 0), processed_measurement=1.0,
            is_strong_binder=True,
        )
        record["target_key"] = f"uniprot:p0000{i}"
        record["sequence_antigen"] = ANTIGEN[: len(ANTIGEN) - i] + "A" * i
        records.append(record)
    ds = _dataset(write_processed_jsonl_gz, tmp_path, records)
    rows = [ds[i] for i in range(4)]

    kwargs = dict(
        tokenizer=tokenizer,
        max_length=192,
        hcdr3_span_probability=0.5,
        shuffle_antigen_probability=1.0,
        rng_seed=42,
    )
    current_collator = AntibodyAntigenCollator(**kwargs)
    legacy_collator = legacy.AntibodyAntigenCollator(**kwargs)
    current = current_collator(rows)
    before = legacy_collator(rows)

    # The only keys the working tree is allowed to ADD over HEAD: the
    # eligibility mask this ticket contributes, and the alias-resolved target
    # grouping J02 contributes. Everything else must still match byte for byte,
    # which the loop below and the RNG-cursor check enforce. J02's donor
    # matching reads `canonical_target_id or target_key`, and this fixture's
    # rows predate the canonical field, so the fallback must reproduce the
    # legacy donor choices exactly -- that is what the tensor comparison proves.
    assert set(current) - set(before) == {
        "conditional_denoising_eligible",
        "canonical_target_ids",
    }
    for key, value in before.items():
        other = current[key]
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, other), key
        else:
            assert value == other, key
    # Same RNG cursor: the stream was not perturbed.
    assert current_collator.rng.getstate() == legacy_collator.rng.getstate()


def test_an_all_nonbinder_batch_inside_a_healthy_epoch_is_counted_not_raised(
    project_root, tmp_path, tokenizer, write_processed_jsonl_gz
):
    """
    Per BATCH an all-nonbinder batch is legitimate and must still contribute
    compatibility supervision. Only the whole-epoch total is fatal. With
    batch_size=2 and the binders grouped at the end, at least one batch is
    all-nonbinder while the epoch as a whole is fine.
    """
    mlm_train = _load_script(project_root, "mlm_train")
    records = [
        _antigen_record(f"nb{i}", binder_label=0, processed_measurement=0.0)
        for i in range(6)
    ] + [
        _antigen_record(f"b{i}", binder_label=1, processed_measurement=1.0)
        for i in range(2)
    ]
    data_path = write_processed_jsonl_gz(tmp_path / "mostly_nonbinders.jsonl.gz", records)
    cfg = mlm_train.TrainConfig(
        data_path=str(data_path),
        training_stage="antigen_real_label_refine",
        init_checkpoint="parent.pt",
        epochs=1,
        batch_size=2,
        eval_batch_size=2,
        max_length=192,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
        hcdr3_span_probability=0.0,
        learning_rate=0.01,
    )
    assert cfg.conditional_denoising_eligibility == "binary_binders_only"
    dataset = OASSequenceDataset(str(data_path), split="train")
    model = mlm_train.build_model(tokenizer, cfg, torch.device("cpu"))
    optimizer = mlm_train.build_optimizer(model, cfg)
    out_dir = tmp_path / "run"
    out_dir.mkdir()

    metrics = mlm_train.train_one_epoch(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        scheduler=mlm_train.build_lr_scheduler(optimizer, cfg),
        cfg=cfg,
        device=torch.device("cpu"),
        epoch=0,
        output_dir=out_dir,
        best_val_loss=float("inf"),
    )
    assert metrics["conditional_denoising_policy_rows"] == 8.0
    assert metrics["conditional_denoising_eligible_rows"] == 2.0
    assert metrics["conditional_denoising_eligible_tokens"] > 0.0


def test_evaluate_reports_the_eligibility_census(
    project_root, tmp_path, tokenizer, write_processed_jsonl_gz
):
    # Under `binary_binders_only` validation MLM loss measures a different
    # population than it did before. Nothing else in the val record says so.
    mlm_train = _load_script(project_root, "mlm_train")
    records = [
        _antigen_record(
            f"r{i}",
            binder_label=(1 if i % 2 == 0 else 0),
            processed_measurement=1.0,
            heavy=HEAVY,
        )
        for i in range(4)
    ]
    for record in records:
        record["split"] = "val"
    data_path = write_processed_jsonl_gz(tmp_path / "val.jsonl.gz", records)
    cfg = mlm_train.TrainConfig(
        data_path=str(data_path),
        training_stage="antigen_real_label_refine",
        init_checkpoint="parent.pt",
        epochs=1,
        batch_size=2,
        eval_batch_size=2,
        max_length=192,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
        hcdr3_span_probability=0.0,
    )
    dataset = OASSequenceDataset(str(data_path), split="val")
    model = mlm_train.build_model(tokenizer, cfg, torch.device("cpu"))

    metrics = mlm_train.evaluate(
        model=model,
        val_dataset=dataset,
        tokenizer=tokenizer,
        cfg=cfg,
        device=torch.device("cpu"),
    )
    assert metrics["conditional_denoising_policy_rows"] == 4.0
    assert metrics["conditional_denoising_eligible_rows"] == 2.0


def test_zero_eligible_tokens_across_an_epoch_fails(
    project_root, tmp_path, tokenizer, write_processed_jsonl_gz
):
    """
    The rows-eligible and tokens-eligible guards are distinct failures. Eligible
    rows can still produce zero MLM targets, in which case the conditional policy
    receives no gradient while the loss reports a healthy finite zero.

    `sampled_span` and `full_span` cannot reach this state -- `_select_target_positions`
    floors the budget at `max(1, ...)`. `partial_span` can: it draws
    `k ~ uniform{0..L}` over the HCDR3, and `k == 0` is a legitimate outcome (the
    fully-visible decode endpoint). Seed 165 is the smallest seed for which both
    rows of this two-row corpus draw `k == 0`, so the guard is exercised through
    the real code path rather than a monkeypatch.
    """
    mlm_train = _load_script(project_root, "mlm_train")
    records = [
        _antigen_record(f"b{i}", binder_label=1, processed_measurement=1.0)
        for i in range(2)
    ]
    data_path = write_processed_jsonl_gz(tmp_path / "no_targets.jsonl.gz", records)
    cfg = mlm_train.TrainConfig(
        data_path=str(data_path),
        training_stage="antigen_real_label_refine",
        init_checkpoint="parent.pt",
        epochs=1,
        batch_size=2,
        eval_batch_size=2,
        max_length=192,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
        hcdr3_mask_mode="partial_span",
        mask_replacement_strategy="always_mask",
        hcdr3_span_probability=0.0,
        seed=165,
        learning_rate=0.01,
    )
    dataset = OASSequenceDataset(str(data_path), split="train")
    model = mlm_train.build_model(tokenizer, cfg, torch.device("cpu"))
    optimizer = mlm_train.build_optimizer(model, cfg)
    out_dir = tmp_path / "run"
    out_dir.mkdir()

    with pytest.raises(ValueError, match="0 target tokens"):
        mlm_train.train_one_epoch(
            model=model,
            train_dataset=dataset,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scaler=torch.amp.GradScaler("cuda", enabled=False),
            scheduler=mlm_train.build_lr_scheduler(optimizer, cfg),
            cfg=cfg,
            device=torch.device("cpu"),
            epoch=0,
            output_dir=out_dir,
            best_val_loss=float("inf"),
        )


def test_preflight_raises_end_to_end_on_an_all_nonbinder_corpus(
    project_root, tmp_path, tokenizer, write_processed_jsonl_gz, monkeypatch
):
    """
    Drives the real `main()`. Without this, deleting the entire preflight block
    from `scripts/mlm_train.py` leaves the whole suite green.
    """
    mlm_train = _load_script(project_root, "mlm_train")

    records = [
        _antigen_record(f"nb{i}", binder_label=0, processed_measurement=0.0)
        for i in range(8)
    ]
    for i, record in enumerate(records):
        record["split"] = "train" if i < 6 else "val"
    data_path = write_processed_jsonl_gz(tmp_path / "nonbinders.jsonl.gz", records)

    # A real parent checkpoint, because `validate_checkpoint_plan` requires one.
    parent = tmp_path / "parent.pt"
    parent_cfg = mlm_train.TrainConfig(
        data_path=str(data_path),
        training_stage="base",
        max_length=192,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
    )
    parent_model = mlm_train.build_model(tokenizer, parent_cfg, torch.device("cpu"))
    mlm_train.save_checkpoint(
        parent,
        parent_model,
        mlm_train.build_optimizer(parent_model, parent_cfg),
        parent_cfg,
        epoch=1,
        val_loss=1.0,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlm_train",
            "--data-path", str(data_path),
            "--training-stage", "antigen_real_label_refine",
            "--init-checkpoint", str(parent),
            "--output-dir", str(tmp_path / "run"),
            "--device", "cpu",
            "--no-resume-from-last",
            "--no-progress",
            "--epochs", "1",
            "--batch-size", "2",
            "--eval-batch-size", "2",
            "--max-length", "192",
            "--d-model", "32",
            "--n-heads", "4",
            "--n-layers", "1",
            "--d-ff", "64",
            "--dropout", "0.0",
        ],
    )
    with pytest.raises(ValueError, match="eligible for antigen-conditioned"):
        mlm_train.main()
