"""Tests for the order-averaged evidence estimator (ported from the mirror).

The two properties worth protecting are:

1. the batched ``K``-orders-at-once implementation equals the naive sequential
   loop bit-for-bit, and
2. the logits are read while the position is still masked -- a reveal-first bug
   would look like a spectacularly good model.
"""
from __future__ import annotations

import math

import pytest
import torch

from smallAntibodyGen.data.MLMCollator import OASRecord
from smallAntibodyGen.infill import evidence as ev
from smallAntibodyGen.infill.hcdr3 import FixedLengthHCDR3Infiller, HCDR3Span
from smallAntibodyGen.models.mlm import AntibodyAntigenCrossAttention, MLMConfig

ANTIGEN = "MKTIIALSYIFCLVFADYKDDDDKAMDIGINSDPYQ"


def _record(tokenizer, heavy_seq: str, heavy_cdr3: str) -> OASRecord:
    start = heavy_seq.index(heavy_cdr3)
    return OASRecord(
        sequence=heavy_seq,
        token_ids=tokenizer.encode_sequence(heavy_seq, locus="IGH", max_length=192),
        locus="IGH",
        chain_group="heavy",
        split="val",
        length=len(heavy_seq),
        token_length=len(heavy_seq) + 3,
        cdr3_start_aa=start,
        cdr3_end_aa=start + len(heavy_cdr3),
        cdr3_aa=heavy_cdr3,
        sequence_heavy=heavy_seq,
        heavy_locus="IGH",
        sequence_antigen=ANTIGEN,
        record_id="r0",
        is_strong_binder=True,
    )


def _infiller(tokenizer) -> FixedLengthHCDR3Infiller:
    torch.manual_seed(17)
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
    model.eval()
    return FixedLengthHCDR3Infiller(model, tokenizer, max_length=192, device="cpu")


# --------------------------------------------------------------- pure stats


def test_summarize_orders_is_hand_verified():
    summary = ev.summarize_orders([1.0, 3.0, 5.0, 7.0])
    assert summary.mean == pytest.approx(4.0)
    # sample sd (ddof=1) of [1,3,5,7] is sqrt(20/3); se = sd/sqrt(4)
    assert summary.se == pytest.approx(math.sqrt(20.0 / 3.0) / 2.0)
    assert summary.half1_mean == pytest.approx(2.0)
    assert summary.half2_mean == pytest.approx(6.0)
    assert summary.half_delta == pytest.approx(4.0)


def test_single_order_reports_nan_not_zero():
    """A single order cannot self-check; reporting 0.0 would read as
    "measured, no error"."""
    summary = ev.summarize_orders([2.5])
    assert summary.mean == pytest.approx(2.5)
    assert math.isnan(summary.se)
    assert math.isnan(summary.half_delta)
    assert summary.half_within_2se is False


def test_summarize_orders_rejects_empty():
    with pytest.raises(ValueError):
        ev.summarize_orders([])


def test_sample_orders_is_deterministic_and_permutes():
    a = ev.sample_orders(6, 4, seed=11)
    b = ev.sample_orders(6, 4, seed=11)
    c = ev.sample_orders(6, 4, seed=12)
    assert a == b
    assert a != c
    for order in a:
        assert sorted(order) == list(range(6))
    assert ev.sample_orders(1, 3, seed=1) == [[0], [0], [0]]


def test_sample_orders_requires_a_seed_source():
    with pytest.raises(ValueError, match="deterministic"):
        ev.sample_orders(4, 2)


# ------------------------------------------------------------- the estimator


def test_batched_orders_equal_the_sequential_loop(tokenizer, heavy_seq, heavy_cdr3):
    """The whole speed argument is that K orders can advance as one batch. If the
    batched result ever differed from the sequential one, E-hat would be a
    different quantity from the one documented."""
    infiller = _infiller(tokenizer)
    record = _record(tokenizer, heavy_seq, heavy_cdr3)
    candidate = heavy_cdr3
    orders = ev.sample_orders(len(candidate), 5, seed=3)

    batched = ev.path_step_logprobs(infiller, record, candidate, orders)
    sequential = torch.cat(
        [ev.path_step_logprobs(infiller, record, candidate, [order]) for order in orders]
    )
    assert torch.equal(batched, sequential)


def test_teacher_forcing_reads_logits_while_still_masked(
    tokenizer, heavy_seq, heavy_cdr3
):
    """Independent reconstruction of step 0 under left-to-right order: the model
    must be scored on the fully-masked state, not on a state where the position
    has already been revealed."""
    infiller = _infiller(tokenizer)
    record = _record(tokenizer, heavy_seq, heavy_cdr3)
    candidate = heavy_cdr3
    order = list(range(len(candidate)))

    step = ev.path_step_logprobs(infiller, record, candidate, [order])

    span = HCDR3Span.from_record(record)
    base_ids, base_attn, mask_positions, _, _ = (
        infiller._encode_antibody_with_masked_hcdr3(
            record, span, proposed_length=len(candidate)
        )
    )
    antigen_ids, antigen_attn = infiller._encode_antigen(record)
    with torch.no_grad():
        logits, _ = infiller.model(
            antibody_input_ids=base_ids,
            antibody_attention_mask=base_attn,
            antigen_input_ids=antigen_ids,
            antigen_attention_mask=antigen_attn,
        )
    canonical = torch.tensor(infiller.canonical_token_ids)
    row = torch.log_softmax(logits[0, mask_positions[0]][canonical], dim=-1)
    residues = [infiller.tokenizer.id_to_token[t] for t in infiller.canonical_token_ids]
    expected = float(row[residues.index(candidate[0])])
    assert float(step[0, 0]) == pytest.approx(expected, abs=1e-6)


def test_evidence_is_order_invariant_in_expectation(tokenizer, heavy_seq, heavy_cdr3):
    """Two independent draws of many orders should agree within a few standard
    errors -- the property single-path scores lack."""
    infiller = _infiller(tokenizer)
    record = _record(tokenizer, heavy_seq, heavy_cdr3)
    a = ev.estimate_evidence(infiller, record, heavy_cdr3, num_orders=16, seed=1)
    b = ev.estimate_evidence(infiller, record, heavy_cdr3, num_orders=16, seed=2)
    combined_se = math.hypot(a.evidence_se, b.evidence_se)
    assert abs(a.evidence - b.evidence) <= 4.0 * combined_se


def test_evidence_is_reproducible_from_a_seed(tokenizer, heavy_seq, heavy_cdr3):
    infiller = _infiller(tokenizer)
    record = _record(tokenizer, heavy_seq, heavy_cdr3)
    a = ev.estimate_evidence(infiller, record, heavy_cdr3, num_orders=4, seed=9)
    b = ev.estimate_evidence(infiller, record, heavy_cdr3, num_orders=4, seed=9)
    assert a == b


def test_orders_must_be_permutations(tokenizer, heavy_seq, heavy_cdr3):
    infiller = _infiller(tokenizer)
    record = _record(tokenizer, heavy_seq, heavy_cdr3)
    with pytest.raises(ValueError, match="permutation"):
        ev.path_step_logprobs(infiller, record, heavy_cdr3, [[0, 0, 1]])
    with pytest.raises(ValueError, match="non-empty"):
        ev.path_step_logprobs(infiller, record, heavy_cdr3, [])


def test_non_canonical_residue_fails_loud(tokenizer, heavy_seq, heavy_cdr3):
    infiller = _infiller(tokenizer)
    record = _record(tokenizer, heavy_seq, heavy_cdr3)
    bad = "X" * len(heavy_cdr3)
    with pytest.raises(ValueError, match="not one of the canonical"):
        ev.estimate_evidence(infiller, record, bad, num_orders=2, seed=0)


def test_identical_candidates_get_identical_evidence(tokenizer, heavy_seq, heavy_cdr3):
    """Seeded by CONTENT, not by list position: two identical candidates must tie
    exactly, or Monte-Carlo noise leaks into the rankings a comparison reports."""
    infiller = _infiller(tokenizer)
    record = _record(tokenizer, heavy_seq, heavy_cdr3)
    a = ev.estimate_evidence(infiller, record, heavy_cdr3, num_orders=4, seed=5)
    b = ev.estimate_evidence(infiller, record, heavy_cdr3, num_orders=4, seed=5)
    assert a.evidence == b.evidence


# --------------------------------------------------------- ranking helpers


def test_rank_topk_is_stable_under_ties():
    scores = [1.0, 3.0, 3.0, 2.0]
    assert ev.rank_topk_stable(scores, 2) == [1, 2]
    assert ev.rank_topk_stable(scores, 10) == [1, 2, 3, 0]
    assert ev.rank_topk_stable(scores, 0) == []


def test_boundary_tie_count_counts_the_split_group():
    assert ev.boundary_tie_count([5.0, 3.0, 3.0, 1.0], 2) == 2
    assert ev.boundary_tie_count([5.0, 4.0, 3.0, 1.0], 2) == 1
    assert ev.boundary_tie_count([5.0, 4.0], 2) == 0
    assert ev.boundary_tie_count([5.0, 4.0], 0) == 0


def test_demotion_rate_is_hand_verified():
    judge = [10.0, 9.0, 8.0, 7.0]
    evidence = [7.0, 8.0, 9.0, 10.0]
    fraction, judge_ties, evidence_ties = ev.demotion_rate_for_record(judge, evidence, 2)
    # Judge top-2 = {0, 1}; evidence top-2 = {3, 2}: both demoted.
    assert fraction == pytest.approx(1.0)
    assert judge_ties == 1
    assert evidence_ties == 1
    # Perfect agreement demotes nothing.
    assert ev.demotion_rate_for_record(judge, judge, 2)[0] == pytest.approx(0.0)
    assert ev.demotion_rate_for_record([], [], 2)[0] == 0.0
    with pytest.raises(ValueError):
        ev.demotion_rate_for_record([1.0], [1.0, 2.0], 1)


def test_json_sanitizer_replaces_non_finite_only():
    payload = {
        "a": float("nan"),
        "b": float("inf"),
        "c": 1.5,
        "d": [float("-inf"), 2, True, "x"],
        "e": {"f": float("nan")},
    }
    assert ev.to_json_safe(payload) == {
        "a": None,
        "b": None,
        "c": 1.5,
        "d": [None, 2, True, "x"],
        "e": {"f": None},
    }


# ------------------------------------------------------- score_candidates.py


def _load_scorer(project_root):
    import importlib.util
    import sys
    from pathlib import Path

    scripts_dir = Path(project_root).parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location(
        "score_candidates", scripts_dir / "score_candidates.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_decision_score_is_hand_verified(project_root):
    scorer = _load_scorer(project_root)
    score, omitted = scorer.decision_score(
        compatibility=0.8,
        evidence_value=-12.0,
        length=4,
        w_match=2.0,
        w_evidence=0.5,
    )
    # 2.0 * 0.8 + 0.5 * (-12.0 / 4) = 1.6 - 1.5 = 0.1
    assert omitted == []
    assert score == pytest.approx(0.1)


def test_missing_compatibility_is_marked_not_silently_dropped(project_root):
    """A partially-computed score that looks complete is worse than no score."""
    scorer = _load_scorer(project_root)
    score, omitted = scorer.decision_score(
        compatibility=None,
        evidence_value=-12.0,
        length=4,
        w_match=1.0,
        w_evidence=1.0,
    )
    assert score is None
    assert omitted == ["compatibility"]


def test_content_seed_is_stable_and_content_dependent(project_root):
    scorer = _load_scorer(project_root)
    assert scorer.content_seed("r0", "ARDY") == scorer.content_seed("r0", "ARDY")
    assert scorer.content_seed("r0", "ARDY") != scorer.content_seed("r0", "ARDW")
    assert scorer.content_seed("r0", "ARDY") != scorer.content_seed("r1", "ARDY")
