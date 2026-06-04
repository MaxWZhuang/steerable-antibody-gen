from __future__ import annotations

import random

import torch

from smallAntibodyGen.data.MLMCollator import OASRecord
from smallAntibodyGen.infill.hcdr3 import (
    AntigenCompatibilityScorer,
    EmpiricalHCDR3LengthPrior,
    FixedLengthHCDR3Infiller,
)
from smallAntibodyGen.models.mlm import AntibodyAntigenCrossAttention, MLMConfig


def make_antigen_record(
    *,
    heavy_sequence: str = "AAAACARDRSTYYYY",
    antigen_sequence: str = "MKTIIALSYIFCLVFADYKDDDDK",
    binder_label: int = 1,
) -> OASRecord:
    cdr3 = "CARDRST"
    start = heavy_sequence.index(cdr3)
    end = start + len(cdr3)
    return OASRecord(
        sequence=heavy_sequence,
        locus="PAIRED_ANTIGEN",
        chain_group="paired_antigen",
        split="train",
        length=len(heavy_sequence),
        sequence_heavy=heavy_sequence,
        sequence_antigen=antigen_sequence,
        heavy_locus="IGH",
        binder_label=binder_label,
        cdr3_aa=cdr3,
        cdr3_start_aa=start,
        cdr3_end_aa=end,
        cdr3_aa_heavy=cdr3,
        cdr3_start_aa_heavy=start,
        cdr3_end_aa_heavy=end,
        record_id="record-1",
        target_key="uniprot:p12345",
    )


def make_tiny_antigen_model(tokenizer) -> AntibodyAntigenCrossAttention:
    config = MLMConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_id,
        max_length=64,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
    )
    return AntibodyAntigenCrossAttention(config)


def test_fixed_length_infiller_preserves_framework_and_changes_only_hcdr3(tokenizer):
    torch.manual_seed(0)
    model = make_tiny_antigen_model(tokenizer)
    record = make_antigen_record()
    infiller = FixedLengthHCDR3Infiller(model, tokenizer, max_length=64, device="cpu")

    candidates = infiller.infill(record, num_samples=1, temperature=1.0, top_k=1)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.length == len("CARDRST")
    assert len(candidate.generated_hcdr3) == len("CARDRST")
    assert candidate.heavy_sequence.startswith("AAAA")
    assert candidate.heavy_sequence.endswith("YYYY")
    assert candidate.heavy_sequence == "AAAA" + candidate.generated_hcdr3 + "YYYY"


def test_fixed_length_infiller_accepts_proposed_unknown_length(tokenizer):
    torch.manual_seed(0)
    model = make_tiny_antigen_model(tokenizer)
    record = make_antigen_record()
    infiller = FixedLengthHCDR3Infiller(model, tokenizer, max_length=64, device="cpu")

    candidates = infiller.infill(record, length=5, num_samples=1, temperature=1.0, top_k=1)

    assert len(candidates) == 1
    assert candidates[0].length == 5
    assert len(candidates[0].generated_hcdr3) == 5
    assert candidates[0].heavy_sequence == "AAAA" + candidates[0].generated_hcdr3 + "YYYY"


def test_empirical_hcdr3_length_prior_is_deterministic(tokenizer):
    records = [
        make_antigen_record(heavy_sequence="AAAACARDRSTYYYY", binder_label=1),
        make_antigen_record(heavy_sequence="AAAACARDRSTYYYY", binder_label=1),
        make_antigen_record(heavy_sequence="GGGGCARDRSTTTTT", binder_label=0),
    ]
    prior = EmpiricalHCDR3LengthPrior.fit(records, positive_only=True)

    first = prior.propose_lengths(records[0], num_lengths=5, rng=random.Random(42))
    second = prior.propose_lengths(records[0], num_lengths=5, rng=random.Random(42))

    assert first == second
    assert first == [len("CARDRST")] * 5


def test_antigen_compatibility_scorer_returns_probability(tokenizer):
    torch.manual_seed(0)
    model = make_tiny_antigen_model(tokenizer)
    record = make_antigen_record()
    scorer = AntigenCompatibilityScorer(model, tokenizer, max_length=64, device="cpu")

    score = scorer.score(record, heavy_sequence=record.sequence_heavy or record.sequence)

    assert 0.0 <= score <= 1.0
