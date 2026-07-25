"""Tests for the learned HCDR3-length posterior, ported from the sibling mirror.

The load-bearing property is the NO-LEAK contract: the length query must be the
collapsed-span encoding (exactly one ``[MASK]``), because on the ordinary
encoding the number of masks IS the answer and a head trained there would learn
to count masks rather than predict a length.
"""
from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import pytest
import torch

from smallAntibodyGen.data.MLMCollator import AntibodyAntigenCollator, OASRecord
from smallAntibodyGen.infill.hcdr3 import (
    FixedLengthHCDR3Infiller,
    HCDR3Span,
    LearnedLengthProposal,
    encode_masked_hcdr3_ids,
)
from smallAntibodyGen.models.mlm import (
    AntibodyAntigenCrossAttention,
    MLMConfig,
    class_index_to_length,
    length_to_class_index,
)

ANTIGEN = "MKTIIALSYIFCLVFADYKDDDDKAMDIGINSDPYQ"


def _load_script(project_root: Path, name: str):
    scripts_dir = project_root.parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location(name, scripts_dir / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _record(tokenizer, heavy_seq: str, heavy_cdr3: str, **overrides) -> OASRecord:
    start = heavy_seq.index(heavy_cdr3)
    base = dict(
        sequence=heavy_seq,
        token_ids=tokenizer.encode_sequence(heavy_seq, locus="IGH", max_length=192),
        locus="PAIRED_ANTIGEN",
        chain_group="paired_antigen",
        split="train",
        length=len(heavy_seq),
        token_length=len(heavy_seq) + 3,
        cdr3_start_aa=start,
        cdr3_end_aa=start + len(heavy_cdr3),
        cdr3_aa=heavy_cdr3,
        sequence_heavy=heavy_seq,
        heavy_locus="IGH",
        sequence_antigen=ANTIGEN,
        record_id="r0",
        target_key="uniprot:p1",
        dataset_name="ds",
        is_strong_binder=True,
    )
    base.update(overrides)
    return OASRecord(**base)


def _model(tokenizer, *, length_head_max: int = 32, use_length_head: bool = True):
    torch.manual_seed(5)
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
            use_length_head=use_length_head,
            length_head_max=length_head_max,
        )
    )
    model.eval()
    return model


# ------------------------------------------------------------ class mapping


def test_length_class_mapping_round_trips_and_fails_loud():
    assert length_to_class_index(1, 32) == 0
    assert length_to_class_index(32, 32) == 31
    assert class_index_to_length(0, 32) == 1
    assert class_index_to_length(31, 32) == 32
    for length in (1, 7, 32):
        assert class_index_to_length(length_to_class_index(length, 32), 32) == length
    # No silent clamp: an out-of-range length is a mis-registered length_head_max,
    # and clamping it would train the head on a wrong class forever.
    with pytest.raises(ValueError, match="out of range"):
        length_to_class_index(33, 32)
    with pytest.raises(ValueError, match="out of range"):
        length_to_class_index(0, 32)


# --------------------------------------------------------------- the head


def test_length_head_off_draws_zero_extra_init_rng(tokenizer):
    torch.manual_seed(4242)
    off = AntibodyAntigenCrossAttention(
        MLMConfig(
            vocab_size=tokenizer.vocab_size,
            pad_token_id=tokenizer.pad_id,
            max_length=64,
            d_model=32,
            n_heads=4,
            n_layers=1,
            d_ff=64,
            dropout=0.0,
        )
    )
    torch.manual_seed(4242)
    on = AntibodyAntigenCrossAttention(
        MLMConfig(
            vocab_size=tokenizer.vocab_size,
            pad_token_id=tokenizer.pad_id,
            max_length=64,
            d_model=32,
            n_heads=4,
            n_layers=1,
            d_ff=64,
            dropout=0.0,
            use_length_head=True,
        )
    )
    assert "length_head.weight" not in off.state_dict()
    assert "length_head.weight" in on.state_dict()
    for key, value in off.state_dict().items():
        assert torch.equal(value, on.state_dict()[key]), key


def test_predict_length_logits_without_a_head_fails_loud(tokenizer):
    model = _model(tokenizer, use_length_head=False)
    with pytest.raises(RuntimeError, match="use_length_head=True"):
        model.predict_length_logits(torch.randn(2, 32))


def test_length_loss_is_masked_and_differentiable_at_zero(tokenizer):
    model = _model(tokenizer)
    logits = torch.randn(3, 32, requires_grad=True)
    labels = torch.tensor([0, 5, 9])
    empty = model.compute_length_loss(logits, labels, torch.zeros(3, dtype=torch.bool))
    assert float(empty) == 0.0
    assert empty.requires_grad
    empty.backward()

    partial = model.compute_length_loss(
        logits, labels, torch.tensor([True, False, True])
    )
    reference = torch.nn.functional.cross_entropy(
        logits[[0, 2]], labels[[0, 2]]
    )
    assert float(partial) == pytest.approx(float(reference))


# ------------------------------------------------------- the no-leak contract


def _length_collator(tokenizer, length_head_max: int = 32) -> AntibodyAntigenCollator:
    return AntibodyAntigenCollator(
        tokenizer=tokenizer,
        max_length=192,
        shuffle_antigen_probability=0.0,
        rng_seed=3,
        build_length_query=True,
        length_head_max=length_head_max,
    )


def test_length_query_has_exactly_one_mask_regardless_of_true_length(
    tokenizer, heavy_seq, heavy_cdr3
):
    """The whole no-leak contract in one assertion: two records differing ONLY in
    HCDR3 length produce length queries with the same number of masks."""
    collator = _length_collator(tokenizer)
    short = _record(tokenizer, heavy_seq, heavy_cdr3[:4])
    long = _record(tokenizer, heavy_seq, heavy_cdr3)
    batch = collator([short, long])
    mask_counts = (batch["length_query_input_ids"] == tokenizer.mask_id).sum(dim=1)
    assert mask_counts.tolist() == [1, 1]
    # ...and the labels still differ, i.e. the test is not vacuous.
    assert batch["length_labels"][0].item() != batch["length_labels"][1].item()


def test_length_query_bytes_match_the_shared_encoder(tokenizer, heavy_seq, heavy_cdr3):
    """Parity: the collator's length query must equal the infiller's
    ``proposed_length=1`` encoding, because they are the same function."""
    collator = _length_collator(tokenizer)
    record = _record(tokenizer, heavy_seq, heavy_cdr3)
    batch = collator([record])
    expected, _, _, _ = encode_masked_hcdr3_ids(
        tokenizer, record, HCDR3Span.from_record(record), proposed_length=1
    )
    actual = batch["length_query_input_ids"][0].tolist()
    assert actual[: len(expected)] == expected


def test_length_labels_use_the_shared_class_mapping(tokenizer, heavy_seq, heavy_cdr3):
    collator = _length_collator(tokenizer)
    record = _record(tokenizer, heavy_seq, heavy_cdr3)
    batch = collator([record])
    assert batch["length_labels"][0].item() == length_to_class_index(len(heavy_cdr3), 32)
    assert bool(batch["length_label_mask"][0])


def test_out_of_range_lengths_are_masked_not_clamped(tokenizer, heavy_seq, heavy_cdr3):
    """A length beyond length_head_max is excluded from supervision. Clamping it
    would teach the head that every long HCDR3 has exactly the maximum length."""
    collator = _length_collator(tokenizer, length_head_max=4)
    record = _record(tokenizer, heavy_seq, heavy_cdr3)
    assert len(heavy_cdr3) > 4
    batch = collator([record])
    assert not bool(batch["length_label_mask"][0])
    # And nothing raised.


def test_invalid_span_rows_are_masked(tokenizer, heavy_seq):
    collator = _length_collator(tokenizer)
    record = OASRecord(
        sequence=heavy_seq,
        locus="PAIRED_ANTIGEN",
        chain_group="paired_antigen",
        split="train",
        length=len(heavy_seq),
        token_length=len(heavy_seq) + 3,
        sequence_heavy=heavy_seq,
        heavy_locus="IGH",
        sequence_antigen=ANTIGEN,
        record_id="r1",
        is_strong_binder=True,
    )
    batch = collator([record])
    assert not bool(batch["length_label_mask"][0])


def test_shuffled_antigen_rows_are_excluded_from_length_supervision(
    tokenizer, heavy_seq, heavy_cdr3
):
    """A donor antigen makes the record's true HCDR3 length a statement about a
    pairing that never existed."""
    collator = AntibodyAntigenCollator(
        tokenizer=tokenizer,
        max_length=192,
        shuffle_antigen_probability=1.0,
        rng_seed=3,
        build_length_query=True,
        length_head_max=32,
    )
    batch = collator(
        [
            _record(
                tokenizer,
                heavy_seq,
                heavy_cdr3,
                record_id="a",
                target_key="uniprot:p1",
                sequence_antigen=ANTIGEN,
            ),
            _record(
                tokenizer,
                heavy_seq,
                heavy_cdr3,
                record_id="b",
                target_key="uniprot:p2",
                sequence_antigen="AMDIGINSDPYQNVKLLTQFGWKA",
            ),
        ]
    )
    shuffled = batch["is_shuffled_antigen"]
    assert bool(shuffled.any())
    assert not bool((batch["length_label_mask"] & shuffled).any())


def test_length_query_keys_are_absent_by_default(tokenizer, heavy_seq, heavy_cdr3):
    collator = AntibodyAntigenCollator(
        tokenizer=tokenizer, max_length=192, shuffle_antigen_probability=0.0, rng_seed=3
    )
    batch = collator([_record(tokenizer, heavy_seq, heavy_cdr3)])
    assert "length_query_input_ids" not in batch
    assert "length_labels" not in batch


def test_length_query_consumes_no_collator_rng(tokenizer, heavy_seq, heavy_cdr3):
    """Turning the length query on must not perturb the masking stream."""
    record = _record(tokenizer, heavy_seq, heavy_cdr3)
    off = AntibodyAntigenCollator(
        tokenizer=tokenizer, max_length=192, shuffle_antigen_probability=0.0, rng_seed=3
    )
    on = _length_collator(tokenizer)
    batch_off = off([record])
    batch_on = on([record])
    assert torch.equal(batch_off["antibody_labels"], batch_on["antibody_labels"])
    assert off.rng.getstate() == on.rng.getstate()


# ------------------------------------------------------- LearnedLengthProposal


def _infiller(tokenizer, model) -> FixedLengthHCDR3Infiller:
    return FixedLengthHCDR3Infiller(model, tokenizer, max_length=192, device="cpu")


def test_learned_proposal_requires_a_length_head(tokenizer):
    model = _model(tokenizer, use_length_head=False)
    with pytest.raises(ValueError, match="use_length_head=True"):
        LearnedLengthProposal(_infiller(tokenizer, model), length_head_max=32)


def test_learned_proposal_rejects_a_max_beyond_the_checkpoint_head(tokenizer):
    """The checkpoint's head size wins; asking for more used to die deep inside
    proposal with a bare IndexError."""
    model = _model(tokenizer, length_head_max=8)
    with pytest.raises(ValueError, match="exceeds the model's length-head size"):
        LearnedLengthProposal(_infiller(tokenizer, model), length_head_max=32)


def test_learned_proposal_samples_only_feasible_lengths(
    tokenizer, heavy_seq, heavy_cdr3
):
    model = _model(tokenizer, length_head_max=16)
    infiller = _infiller(tokenizer, model)
    proposal = LearnedLengthProposal(infiller, length_head_max=16)
    lengths = proposal.propose_lengths(
        _record(tokenizer, heavy_seq, heavy_cdr3), num_lengths=25, rng=random.Random(0)
    )
    assert len(lengths) == 25
    assert all(1 <= length <= 16 for length in lengths)


def test_learned_proposal_posterior_is_renormalized(tokenizer, heavy_seq, heavy_cdr3):
    model = _model(tokenizer, length_head_max=16)
    proposal = LearnedLengthProposal(_infiller(tokenizer, model), length_head_max=16)
    record = _record(tokenizer, heavy_seq, heavy_cdr3)
    lengths, probs = proposal._renormalized_posterior(
        record, HCDR3Span.from_record(record)
    )
    assert lengths
    assert sum(probs) == pytest.approx(1.0)


def test_learned_proposal_returns_empty_when_nothing_fits(
    tokenizer, heavy_seq, heavy_cdr3
):
    """An overflowing scaffold must return [] rather than raising: the CLI calls
    propose_lengths outside its per-length try/except, so a raise would abort the
    entire generation run over one record."""
    model = _model(tokenizer, length_head_max=16)
    infiller = FixedLengthHCDR3Infiller(model, tokenizer, max_length=4, device="cpu")
    proposal = LearnedLengthProposal(infiller, length_head_max=16)
    assert (
        proposal.propose_lengths(
            _record(tokenizer, heavy_seq, heavy_cdr3),
            num_lengths=3,
            rng=random.Random(0),
        )
        == []
    )


def test_learned_proposal_top_k_is_deterministic(tokenizer, heavy_seq, heavy_cdr3):
    model = _model(tokenizer, length_head_max=16)
    proposal = LearnedLengthProposal(
        _infiller(tokenizer, model), length_head_max=16, mode="top_k"
    )
    record = _record(tokenizer, heavy_seq, heavy_cdr3)
    a = proposal.propose_lengths(record, num_lengths=4, rng=random.Random(1))
    b = proposal.propose_lengths(record, num_lengths=4, rng=random.Random(999))
    assert a == b
    assert len(set(a)) == len(a)


# --------------------------------------------------------------- config knobs


def test_length_knobs_default_off(tmp_path: Path, project_root: Path):
    mlm_train = _load_script(project_root, "mlm_train")
    data_path = tmp_path / "tiny.jsonl.gz"
    data_path.write_text("", encoding="utf-8")
    cfg = mlm_train.parse_args(["--data-path", str(data_path)])
    assert cfg.length_loss_weight == 0.0
    assert cfg.length_head_max == 32
    with pytest.raises(ValueError, match="only supported for antigen stages"):
        mlm_train.parse_args(
            ["--data-path", str(data_path), "--length-loss-weight", "0.5"]
        )


# ------------------------------------------------------------- length census


def test_length_census_coverage_math(project_root: Path):
    census = _load_script(project_root, "length_census")
    from collections import Counter

    counts = Counter({5: 10, 10: 10, 40: 5})
    assert census.coverage_at(counts, 10) == pytest.approx(20 / 25)
    assert census.coverage_at(counts, 40) == pytest.approx(1.0)
    assert census.percentile(counts, 0.5) == 10
    assert census.percentile(counts, 0.99) == 40
