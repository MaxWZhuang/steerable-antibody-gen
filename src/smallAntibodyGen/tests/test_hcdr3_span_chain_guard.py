"""Regression test: a LIGHT chain's CDR3 must never be reported as the HCDR3.

`prepare_oas.py` writes `cdr3_start_aa`/`cdr3_end_aa` for IGK/IGL records too, and
`cdr3_*_heavy` is absent on them, so `_heavy_hcdr3_aa_span`'s generic fallback used
to take the LIGHT CDR3 and hand it back as the heavy one. Consequences:

- `full_span` / `partial_span`: the light CDR3 became the infilling TARGET, i.e.
  the model was trained to infill a light CDR3 under the name HCDR3.
- every mode: the span was counted in `hcdr3_valid_mask` / `hcdr3_target_mask` /
  `hcdr3_original`, so `hcdr3_token_acc`, `hcdr3_span_exact_match` and
  `hcdr3_valid_spans` folded light-CDR3 tokens into the reported "HCDR3" number.

The `sampled_span` MASKING branch was already guarded; the fallback and the
metadata builder were not. Any run over `oas_igk`/`oas_igl`, or a locus-balanced
`oas_all`, reported a corrupted HCDR3 metric.
"""
from __future__ import annotations

import pytest

from smallAntibodyGen.data.MLMCollator import MLMCollator, OASRecord


LIGHT_SEQ = (
    "DIQMTQSPSSLSASVGDRVTITCQASQDINNYLNWYQQKPGKAPKLLIYYTSRLHSGVPSRFSGSGSGTDFT"
    "LTISSLQPEDFATYYCQQYNSYPWTFGQGTKVEIK"
)
LIGHT_CDR3 = "QQYNSYPWT"


def _light_record(**overrides) -> OASRecord:
    start = LIGHT_SEQ.index(LIGHT_CDR3)
    base = dict(
        sequence=LIGHT_SEQ, locus="IGK", chain_group="light", split="train",
        length=len(LIGHT_SEQ), cdr3_aa=LIGHT_CDR3,
        cdr3_start_aa=start, cdr3_end_aa=start + len(LIGHT_CDR3),
    )
    base.update(overrides)
    return OASRecord(**base)


def _heavy_record(heavy_seq: str, heavy_cdr3: str) -> OASRecord:
    start = heavy_seq.index(heavy_cdr3)
    return OASRecord(
        sequence=heavy_seq, locus="IGH", chain_group="heavy", split="train",
        length=len(heavy_seq), cdr3_aa=heavy_cdr3,
        cdr3_start_aa=start, cdr3_end_aa=start + len(heavy_cdr3),
    )


def _collator(tokenizer, mode: str) -> MLMCollator:
    return MLMCollator(
        tokenizer, max_length=192, hcdr3_span_probability=1.0,
        hcdr3_mask_mode=mode, rng_seed=0,
    )


@pytest.mark.parametrize("mode", ["sampled_span", "full_span", "partial_span"])
def test_light_chain_is_never_a_valid_hcdr3_span(tokenizer, mode: str):
    batch = _collator(tokenizer, mode)([_light_record()])
    assert batch["hcdr3_valid_mask"].tolist() == [False]
    assert batch["hcdr3_original"] == [None], "the LIGHT CDR3 leaked into hcdr3_original"
    assert int(batch["hcdr3_target_mask"].sum()) == 0


@pytest.mark.parametrize("mode", ["full_span", "partial_span"])
def test_light_chain_contributes_no_infilling_targets(tokenizer, mode: str):
    """The strict HCDR3 modes must produce ZERO targets for a light-only record."""
    batch = _collator(tokenizer, mode)([_light_record()])
    assert int(batch["labels"].ne(-100).sum()) == 0


def test_light_chain_still_gets_ordinary_mlm_masking(tokenizer):
    """The guard must not stop light chains from being trained on at all.

    `sampled_span` is the general MLM mode; a light record should still receive
    random residue targets, just not be counted as an HCDR3 span.
    """
    batch = _collator(tokenizer, "sampled_span")([_light_record()])
    assert int(batch["labels"].ne(-100).sum()) > 0
    assert batch["hcdr3_valid_mask"].tolist() == [False]


@pytest.mark.parametrize("mode", ["sampled_span", "full_span", "partial_span"])
def test_heavy_chain_is_unaffected(tokenizer, heavy_seq, heavy_cdr3, mode: str):
    batch = _collator(tokenizer, mode)([_heavy_record(heavy_seq, heavy_cdr3)])
    assert batch["hcdr3_valid_mask"].tolist() == [True]
    assert batch["hcdr3_original"] == [heavy_cdr3]


def test_heavy_specific_fields_still_win_regardless_of_chain_group(tokenizer):
    """The guard applies ONLY to the generic fallback.

    A record carrying explicit `cdr3_*_heavy` is unambiguous, so it must be honored
    even when `chain_group` is an unfamiliar value.
    """
    start = LIGHT_SEQ.index(LIGHT_CDR3)
    record = _light_record(
        chain_group="something_new",
        cdr3_aa_heavy=LIGHT_CDR3,
        cdr3_start_aa_heavy=start,
        cdr3_end_aa_heavy=start + len(LIGHT_CDR3),
    )
    batch = _collator(tokenizer, "full_span")([record])
    assert batch["hcdr3_valid_mask"].tolist() == [True]


def test_paired_and_antigen_records_keep_the_generic_fallback(tokenizer, heavy_seq, heavy_cdr3):
    """Paired / antigen records legitimately carry their HEAVY CDR3 generically."""
    start = heavy_seq.index(heavy_cdr3)
    for chain_group, is_paired in (("paired_antigen", False), ("paired", True)):
        record = OASRecord(
            sequence=heavy_seq, locus="PAIRED_ANTIGEN", chain_group=chain_group,
            split="train", length=len(heavy_seq), cdr3_aa=heavy_cdr3,
            cdr3_start_aa=start, cdr3_end_aa=start + len(heavy_cdr3),
            sequence_heavy=heavy_seq, heavy_locus="IGH", is_paired=is_paired,
        )
        batch = _collator(tokenizer, "full_span")([record])
        assert batch["hcdr3_valid_mask"].tolist() == [True], chain_group
        assert batch["hcdr3_original"] == [heavy_cdr3], chain_group
