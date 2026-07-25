"""Tests for the schedule-covering corruption knobs.

Three opt-in knobs, all defaulting to the historical behavior byte-for-byte:

- ``mask_rate_schedule`` on the collator (``fixed`` | ``uniform``),
- ``eval_mask_rate_schedule`` (``""`` = inherit the train schedule),
- ``report_masked_fraction_bins`` (per-masked-fraction MLM accuracy bins).

Ported from the sibling mirror repo, including the boundary-misbin fix its
adversarial verifier found after the original landing.
"""
from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import pytest
import torch

from smallAntibodyGen.data.MLMCollator import MLMCollator, OASRecord
from smallAntibodyGen.tokenizer import AminoAcidTokenizer


def _load_mlm_train(project_root: Path):
    script_path = project_root.parents[1] / "scripts" / "mlm_train.py"
    spec = importlib.util.spec_from_file_location("mlm_train", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(tokenizer: AminoAcidTokenizer, sequence: str) -> OASRecord:
    token_ids = tokenizer.encode_sequence(sequence, locus="IGH", max_length=64)
    return OASRecord(
        sequence=sequence,
        token_ids=token_ids,
        locus="IGH",
        chain_group="heavy",
        split="train",
        length=len(sequence),
        token_length=len(token_ids),
    )


HEAVY_A = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQ"
HEAVY_B = "QVQLQESGPGLVKPSETLSLTCTVSGGSISSYYWSWIRQ"


def _collator(tokenizer, **kwargs) -> MLMCollator:
    base = dict(
        tokenizer=tokenizer,
        max_length=64,
        mask_probability=0.15,
        hcdr3_span_probability=0.0,
        rng_seed=1234,
    )
    base.update(kwargs)
    return MLMCollator(**base)


# ------------------------------------------------------------------ collator


def test_fixed_schedule_is_the_default_and_is_byte_identical(tokenizer):
    """The default must not merely produce the same masking *rate* -- it must
    consume the identical RNG stream, so batches are bit-for-bit what they were
    before the knob existed."""
    records = [_record(tokenizer, HEAVY_A), _record(tokenizer, HEAVY_B)]
    default = _collator(tokenizer)
    explicit = _collator(tokenizer, mask_rate_schedule="fixed")
    assert default.mask_rate_schedule == "fixed"

    batch_default = default(records)
    batch_explicit = explicit(records)
    assert list(batch_default.keys()) == list(batch_explicit.keys())
    for key, value in batch_default.items():
        other = batch_explicit[key]
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, other), key
        else:
            assert value == other, key
    # Same RNG position after the call, not just the same output.
    assert default.rng.getstate() == explicit.rng.getstate()


def test_uniform_schedule_varies_the_masked_fraction_across_rows(tokenizer):
    """Under ``uniform`` the per-row masked fraction is drawn, so a batch of
    identical-length rows must NOT all land on the same target count."""
    records = [_record(tokenizer, HEAVY_A) for _ in range(24)]
    collator = _collator(tokenizer, mask_rate_schedule="uniform")
    batch = collator(records)
    per_row = (batch["labels"] != -100).sum(dim=1).tolist()
    assert len(set(per_row)) > 1


def test_uniform_schedule_ignores_mask_probability(tokenizer):
    """Two collators differing ONLY in ``mask_probability`` produce identical
    uniform batches -- the knob is documented as ignored, so pin it."""
    records = [_record(tokenizer, HEAVY_A) for _ in range(12)]
    low = _collator(tokenizer, mask_rate_schedule="uniform", mask_probability=0.05)
    high = _collator(tokenizer, mask_rate_schedule="uniform", mask_probability=0.95)
    batch_low = low(records)
    batch_high = high(records)
    assert torch.equal(batch_low["labels"], batch_high["labels"])


def test_uniform_schedule_consumes_exactly_one_extra_draw_per_selection(tokenizer):
    """The rate draw is `1 - random()`, exactly one per `_select_target_positions`
    call, taken immediately before the budget computation.

    Counted on the selection method alone rather than the whole batch: downstream
    BERT corruption draws scale with the number of targets, so a whole-batch count
    would not isolate the schedule draw.
    """
    record = _record(tokenizer, HEAVY_A)

    def draws_consumed(schedule: str) -> int:
        probe = _collator(tokenizer, mask_rate_schedule=schedule)
        row = torch.tensor(record.token_ids, dtype=torch.long)
        counter = {"n": 0}
        real_random = probe.rng.random

        def counting_random():
            counter["n"] += 1
            return real_random()

        probe.rng.random = counting_random  # type: ignore[method-assign]
        probe._select_target_positions(row, record)
        return counter["n"]

    assert draws_consumed("uniform") - draws_consumed("fixed") == 1


def test_uniform_schedule_is_inert_in_full_span_mode(tokenizer, heavy_seq, heavy_cdr3):
    """``full_span`` returns before the budget path, so the schedule cannot
    change it -- and must not perturb the RNG stream either."""
    start = heavy_seq.index(heavy_cdr3)
    record = OASRecord(
        sequence=heavy_seq,
        token_ids=tokenizer.encode_sequence(heavy_seq, locus="IGH", max_length=128),
        locus="IGH",
        chain_group="heavy",
        split="train",
        length=len(heavy_seq),
        token_length=len(heavy_seq) + 3,
        cdr3_start_aa=start,
        cdr3_end_aa=start + len(heavy_cdr3),
    )
    fixed = _collator(
        tokenizer, max_length=128, hcdr3_mask_mode="full_span", mask_rate_schedule="fixed"
    )
    uniform = _collator(
        tokenizer,
        max_length=128,
        hcdr3_mask_mode="full_span",
        mask_rate_schedule="uniform",
    )
    batch_fixed = fixed([record])
    batch_uniform = uniform([record])
    assert torch.equal(batch_fixed["labels"], batch_uniform["labels"])
    assert fixed.rng.getstate() == uniform.rng.getstate()


def test_invalid_mask_rate_schedule_is_rejected(tokenizer):
    with pytest.raises(ValueError, match="mask_rate_schedule"):
        _collator(tokenizer, mask_rate_schedule="linear")


# -------------------------------------------------------- masked-fraction bins


def _bin_inputs(n_targets: int, n_eligible: int, vocab: int = 40):
    """One row with ``n_eligible`` non-special positions, ``n_targets`` of them
    labelled, and logits that always predict the right token."""
    seq_len = n_eligible + 2
    labels = torch.full((1, seq_len), -100, dtype=torch.long)
    input_ids = torch.zeros((1, seq_len), dtype=torch.long)  # 0 == pad == special
    input_ids[0, 1 : 1 + n_eligible] = 10
    labels[0, 1 : 1 + n_targets] = 10
    logits = torch.zeros((1, seq_len, vocab))
    logits[0, :, 10] = 5.0
    return logits, labels, input_ids


def test_masked_fraction_bin_counts_by_hand(project_root: Path, tokenizer):
    """Two rows, hand-computed: 2/10 -> [0.2,0.4), 9/10 -> [0.8,1.0]."""
    mlm_train = _load_mlm_train(project_root)
    logits_a, labels_a, ids_a = _bin_inputs(2, 10)
    logits_b, labels_b, ids_b = _bin_inputs(9, 10)
    logits = torch.cat([logits_a, logits_b])
    labels = torch.cat([labels_a, labels_b])
    input_ids = torch.cat([ids_a, ids_b])

    counts = mlm_train.masked_fraction_bin_counts(
        logits, labels, input_ids, tokenizer.special_ids
    )
    assert counts == {"frac_20_40": [2, 2], "frac_80_100": [9, 9]}


def test_masked_fraction_boundary_bins_are_exact(project_root: Path, tokenizer):
    """Boundary regression: frac == 3/5 must land in [0.6, 0.8).

    The float form ``int(0.6 / 0.2)`` gives 2 because 0.6/0.2 evaluates to
    2.9999999999999996; the integer form gives 3. Every eligible-20 row with 12
    targets was landing one bin too low -- systematically distorting exactly the
    bins a corruption-coverage comparison reads.
    """
    mlm_train = _load_mlm_train(project_root)
    cases = [
        (3, 20, "frac_0_20"),  # 0.15
        (4, 20, "frac_20_40"),  # 0.20 edge
        (8, 20, "frac_40_60"),  # 0.40 edge
        (3, 5, "frac_60_80"),  # 0.60 edge -- THE defect case
        (12, 20, "frac_60_80"),  # 0.60 edge, larger denominator
        (16, 20, "frac_80_100"),  # 0.80 edge
        (5, 5, "frac_80_100"),  # 1.0, right-closed last bin
    ]
    for n_targets, n_eligible, expected_key in cases:
        logits, labels, input_ids = _bin_inputs(n_targets, n_eligible)
        counts = mlm_train.masked_fraction_bin_counts(
            logits, labels, input_ids, tokenizer.special_ids
        )
        assert list(counts.keys()) == [expected_key], (
            f"{n_targets}/{n_eligible} -> {list(counts.keys())}, expected {expected_key}"
        )
        assert counts[expected_key] == [n_targets, n_targets]


def test_finalize_masked_fraction_bins_emits_every_bin(project_root: Path):
    """Empty bins are emitted as NaN accuracy with 0 tokens, never dropped --
    a missing key in one arm and not the other silently breaks a comparison."""
    mlm_train = _load_mlm_train(project_root)
    metrics = mlm_train.finalize_masked_fraction_bins({"frac_40_60": [3, 4]})
    assert metrics["mlm_acc_frac_40_60"] == pytest.approx(0.75)
    assert metrics["mlm_tokens_frac_40_60"] == 4.0
    for key in ("frac_0_20", "frac_20_40", "frac_60_80", "frac_80_100"):
        assert metrics[f"mlm_acc_{key}"] != metrics[f"mlm_acc_{key}"]  # NaN
        assert metrics[f"mlm_tokens_{key}"] == 0.0


# ------------------------------------------------------------- config plumbing


def test_schedule_config_defaults_and_validation(tmp_path: Path, project_root: Path):
    mlm_train = _load_mlm_train(project_root)
    data_path = tmp_path / "tiny.jsonl.gz"
    data_path.write_text("", encoding="utf-8")

    cfg = mlm_train.parse_args(["--data-path", str(data_path)])
    assert cfg.mask_rate_schedule == "fixed"
    assert cfg.eval_mask_rate_schedule == ""
    assert cfg.report_masked_fraction_bins is False

    cfg = mlm_train.parse_args(
        [
            "--data-path",
            str(data_path),
            "--mask-rate-schedule",
            "uniform",
            "--eval-mask-rate-schedule",
            "uniform",
            "--report-masked-fraction-bins",
        ]
    )
    assert cfg.mask_rate_schedule == "uniform"
    assert cfg.eval_mask_rate_schedule == "uniform"
    assert cfg.report_masked_fraction_bins is True

    cfg.mask_rate_schedule = "linear"
    with pytest.raises(ValueError, match="mask_rate_schedule"):
        cfg.validate()


def test_eval_schedule_inherits_when_blank(tmp_path: Path, project_root: Path):
    """`""` means inherit, which is what lets arms that differ in TRAIN schedule
    still share one arm-independent eval protocol when it is set explicitly."""
    mlm_train = _load_mlm_train(project_root)
    data_path = tmp_path / "tiny.jsonl.gz"
    data_path.write_text("", encoding="utf-8")
    cfg = mlm_train.parse_args(
        ["--data-path", str(data_path), "--mask-rate-schedule", "uniform"]
    )
    assert (cfg.eval_mask_rate_schedule or cfg.mask_rate_schedule) == "uniform"

    cfg = mlm_train.parse_args(
        [
            "--data-path",
            str(data_path),
            "--mask-rate-schedule",
            "uniform",
            "--eval-mask-rate-schedule",
            "fixed",
        ]
    )
    assert (cfg.eval_mask_rate_schedule or cfg.mask_rate_schedule) == "fixed"


def test_report_masked_fraction_bins_negation_flag(tmp_path: Path, project_root: Path):
    mlm_train = _load_mlm_train(project_root)
    data_path = tmp_path / "tiny.jsonl.gz"
    data_path.write_text("", encoding="utf-8")
    config_path = tmp_path / "train.yaml"
    if mlm_train.yaml is None:
        pytest.skip("PyYAML not installed in test environment")
    config_path.write_text(
        f"data_path: {data_path}\nreport_masked_fraction_bins: true\n", encoding="utf-8"
    )
    cfg = mlm_train.parse_args(["--config", str(config_path)])
    assert cfg.report_masked_fraction_bins is True
    cfg = mlm_train.parse_args(
        ["--config", str(config_path), "--no-report-masked-fraction-bins"]
    )
    assert cfg.report_masked_fraction_bins is False


def test_unused_random_import_guard():
    """`random` is imported for the RNG-draw counting test above; keep it used
    so linters do not remove the import the test depends on."""
    assert isinstance(random.Random(0).random(), float)
