from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import torch

from smallAntibodyGen.data.MLMCollator import OASRecord
from smallAntibodyGen.infill.hcdr3 import (
    AntigenCompatibilityScorer,
    EmpiricalHCDR3LengthPrior,
    FixedLengthHCDR3Infiller,
    HCDR3Span,
)
from smallAntibodyGen.models.mlm import AntibodyAntigenCrossAttention, MLMConfig


def make_antigen_record(
    *,
    heavy_sequence: str = "AAAACARDRSTYYYY",
    antigen_sequence: str = "MKTIIALSYIFCLVFADYKDDDDK",
    binder_label: int = 1,
    is_strong_binder: bool | None = None,
) -> OASRecord:
    cdr3 = "CARDRST"
    start = heavy_sequence.index(cdr3)
    end = start + len(cdr3)
    # The HCDR3 infill stage and the empirical length prior gate on
    # is_strong_binder; default it to the binder_label so positive fixtures are
    # also strong binders unless a test overrides it explicitly.
    if is_strong_binder is None:
        is_strong_binder = binder_label == 1
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
        is_strong_binder=is_strong_binder,
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


def test_infiller_runs_one_forward_for_multiple_samples(tokenizer):
    # The masked input is identical across samples, so infill() must run the
    # model forward exactly once and reuse the shared logits for every draw,
    # rather than re-encoding and re-running the model per sample.
    torch.manual_seed(0)
    inner = make_tiny_antigen_model(tokenizer)

    class CountingModel(torch.nn.Module):
        def __init__(self, model: torch.nn.Module) -> None:
            super().__init__()
            self.model = model
            self.forward_calls = 0

        def forward(self, **kwargs):
            self.forward_calls += 1
            return self.model(**kwargs)

    counting = CountingModel(inner)
    record = make_antigen_record()
    infiller = FixedLengthHCDR3Infiller(counting, tokenizer, max_length=64, device="cpu")

    candidates = infiller.infill(record, num_samples=5, temperature=1.0)

    assert len(candidates) == 5
    assert counting.forward_calls == 1  # one shared forward, not one per sample
    for candidate in candidates:
        assert len(candidate.generated_hcdr3) == len("CARDRST")


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


def test_empirical_length_prior_gates_on_is_strong_binder(tokenizer):
    # The prior follows is_strong_binder, not binder_label: a KD-style strong
    # binder with no bool binder_label still contributes, and a bool positive
    # that is not a strong binder is excluded.
    records = [
        make_antigen_record(binder_label=0, is_strong_binder=True),
        make_antigen_record(binder_label=1, is_strong_binder=False),
    ]
    prior = EmpiricalHCDR3LengthPrior.fit(records, positive_only=True)

    # Only the strong binder (HCDR3 length 7) is in the histogram.
    assert prior.length_counts == {len("CARDRST"): 1}
    assert prior.propose_lengths(records[0], num_lengths=3, rng=random.Random(0)) == [7, 7, 7]


def test_antigen_compatibility_scorer_returns_probability(tokenizer):
    torch.manual_seed(0)
    model = make_tiny_antigen_model(tokenizer)
    record = make_antigen_record()
    scorer = AntigenCompatibilityScorer(model, tokenizer, max_length=64, device="cpu")

    score = scorer.score(record, heavy_sequence=record.sequence_heavy or record.sequence)

    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# ProteinGuide-style guidance primitive (v1)
#
# `_guided_position_scores` is the order-independent core of guided infilling:
# given one fixed partially-filled working sequence and a single masked token
# position, it returns the guided per-residue base logits used to sample that
# position. Guidance reweights the model's own MLM marginal by the compatibility
# head's binder log-probability for each candidate residue, evaluated by
# batched enumeration over the ~20 canonical amino acids:
#
#     guided_logits[a] = log p_MLM(a | x) + gamma * log p(binder | x with pos=a)
#
# These tests pin the primitive independently of any unmasking schedule: the
# schedule only decides *which* working state and position are passed in.
# ---------------------------------------------------------------------------


def _masked_working_state(infiller, record):
    """Build a fully HCDR3-masked antibody/antigen state and its first mask pos."""
    span = HCDR3Span.from_record(record)
    (
        antibody_input_ids,
        antibody_attention_mask,
        mask_positions,
        _prefix,
        _suffix,
    ) = infiller._encode_antibody_with_masked_hcdr3(record, span, proposed_length=span.length)
    antigen_input_ids, antigen_attention_mask = infiller._encode_antigen(record)
    return (
        antibody_input_ids,
        antibody_attention_mask,
        antigen_input_ids,
        antigen_attention_mask,
        mask_positions,
    )


def test_guided_position_scores_match_sequential_reference(tokenizer):
    # The batched-enumeration primitive must equal a from-scratch sequential
    # reference: MLM marginal at the position, plus gamma times the per-candidate
    # binder log-probability obtained by placing each canonical residue and
    # re-running the model one candidate at a time.
    torch.manual_seed(0)
    model = make_tiny_antigen_model(tokenizer)
    model.eval()
    record = make_antigen_record()
    infiller = FixedLengthHCDR3Infiller(model, tokenizer, max_length=64, device="cpu")

    (
        antibody_input_ids,
        antibody_attention_mask,
        antigen_input_ids,
        antigen_attention_mask,
        mask_positions,
    ) = _masked_working_state(infiller, record)
    position = mask_positions[0]
    gamma = 2.0

    guided, unguided = infiller._guided_position_scores(
        antibody_input_ids,
        antibody_attention_mask,
        antigen_input_ids,
        antigen_attention_mask,
        position,
        guidance_strength=gamma,
        guidance_target=1,
    )

    canonical_ids = infiller.canonical_token_ids
    with torch.no_grad():
        mlm_logits, _ = model(
            antibody_input_ids=antibody_input_ids,
            antibody_attention_mask=antibody_attention_mask,
            antigen_input_ids=antigen_input_ids,
            antigen_attention_mask=antigen_attention_mask,
        )
    ref_unguided = torch.log_softmax(mlm_logits[0, position][canonical_ids], dim=-1)

    ref_binder = []
    for token_id in canonical_ids:
        candidate = antibody_input_ids.clone()
        candidate[0, position] = token_id
        with torch.no_grad():
            _, compatibility_logits = model(
                antibody_input_ids=candidate,
                antibody_attention_mask=antibody_attention_mask,
                antigen_input_ids=antigen_input_ids,
                antigen_attention_mask=antigen_attention_mask,
            )
        ref_binder.append(torch.log_softmax(compatibility_logits, dim=-1)[0, 1])
    ref_binder = torch.stack(ref_binder)
    ref_guided = ref_unguided + gamma * ref_binder

    assert guided.shape == (len(canonical_ids),)
    assert unguided.shape == (len(canonical_ids),)
    assert torch.allclose(unguided, ref_unguided, atol=1e-5)
    assert torch.allclose(guided, ref_guided, atol=1e-5)


def test_guided_position_scores_zero_strength_ignores_classifier(tokenizer):
    # At gamma == 0 the classifier must have no effect: guided base logits equal
    # the unguided MLM marginal exactly, so sampling reduces to plain MLM
    # decoding regardless of what the compatibility head predicts.
    torch.manual_seed(0)
    model = make_tiny_antigen_model(tokenizer)
    model.eval()
    record = make_antigen_record()
    infiller = FixedLengthHCDR3Infiller(model, tokenizer, max_length=64, device="cpu")

    (
        antibody_input_ids,
        antibody_attention_mask,
        antigen_input_ids,
        antigen_attention_mask,
        mask_positions,
    ) = _masked_working_state(infiller, record)

    guided, unguided = infiller._guided_position_scores(
        antibody_input_ids,
        antibody_attention_mask,
        antigen_input_ids,
        antigen_attention_mask,
        mask_positions[0],
        guidance_strength=0.0,
        guidance_target=1,
    )

    assert torch.allclose(guided, unguided, atol=1e-6)


# ---------------------------------------------------------------------------
# guided_infill: the iterative, easy-first, one-position-per-step sampler that
# threads the guidance primitive through an unmasking loop.
# ---------------------------------------------------------------------------


def test_guided_infill_preserves_framework_and_reports_unguided_logprob(tokenizer):
    # Guided generation only edits the HCDR3 interval, and the reported
    # log-probability stays the per-position mean of the model's own (unguided)
    # marginal so candidates remain comparable to plain infill() output.
    torch.manual_seed(0)
    model = make_tiny_antigen_model(tokenizer)
    record = make_antigen_record()
    infiller = FixedLengthHCDR3Infiller(model, tokenizer, max_length=64, device="cpu")

    candidates = infiller.guided_infill(record, num_samples=2, guidance_strength=1.0)

    assert len(candidates) == 2
    for candidate in candidates:
        assert candidate.length == len("CARDRST")
        assert len(candidate.generated_hcdr3) == len("CARDRST")
        assert candidate.heavy_sequence == "AAAA" + candidate.generated_hcdr3 + "YYYY"
        assert candidate.mean_log_probability <= 0.0
        assert abs(candidate.log_probability / candidate.length - candidate.mean_log_probability) < 1e-6


def test_guided_infill_zero_strength_ignores_classifier(tokenizer):
    # At gamma == 0 the compatibility head must not influence sampling. Perturbing
    # only that head between two identically-seeded runs must leave the generated
    # HCDR3s unchanged, proving guidance is genuinely disabled (not just weak).
    model = make_tiny_antigen_model(tokenizer)
    record = make_antigen_record()
    infiller = FixedLengthHCDR3Infiller(model, tokenizer, max_length=64, device="cpu")

    torch.manual_seed(0)
    first = infiller.guided_infill(record, num_samples=3, guidance_strength=0.0, temperature=1.0)

    with torch.no_grad():
        for parameter in model.compatibility_head.parameters():
            parameter.add_(torch.randn_like(parameter) * 5.0)

    torch.manual_seed(0)
    second = infiller.guided_infill(record, num_samples=3, guidance_strength=0.0, temperature=1.0)

    assert [c.generated_hcdr3 for c in first] == [c.generated_hcdr3 for c in second]


def test_guided_infill_confidence_greedy_matches_primitive_first_step(tokenizer):
    # End-to-end wiring check for the default schedule: with strong guidance and
    # greedy sampling (top_k=1), the first residue committed must be (a) at the
    # lowest-entropy position (easy-first order) and (b) the argmax of the tested
    # guidance primitive at that position. This ties the loop to the primitive
    # deterministically, with no reliance on sampling noise.
    torch.manual_seed(0)
    model = make_tiny_antigen_model(tokenizer)
    model.eval()
    record = make_antigen_record()
    infiller = FixedLengthHCDR3Infiller(model, tokenizer, max_length=64, device="cpu")

    (
        antibody_input_ids,
        antibody_attention_mask,
        antigen_input_ids,
        antigen_attention_mask,
        mask_positions,
    ) = _masked_working_state(infiller, record)

    canonical_ids = infiller.canonical_token_ids
    with torch.no_grad():
        mlm_logits, _ = model(
            antibody_input_ids=antibody_input_ids,
            antibody_attention_mask=antibody_attention_mask,
            antigen_input_ids=antigen_input_ids,
            antigen_attention_mask=antigen_attention_mask,
        )
    marginals = torch.stack([mlm_logits[0, pos][canonical_ids] for pos in mask_positions])
    logprobs = torch.log_softmax(marginals, dim=-1)
    entropy = -(logprobs.exp() * logprobs).sum(dim=-1)
    star = int(torch.argmin(entropy).item())
    first_position = mask_positions[star]

    guided, _unguided = infiller._guided_position_scores(
        antibody_input_ids,
        antibody_attention_mask,
        antigen_input_ids,
        antigen_attention_mask,
        first_position,
        guidance_strength=1000.0,
        guidance_target=1,
    )
    expected_residue = tokenizer.id_to_token[canonical_ids[int(torch.argmax(guided).item())]]

    candidates = infiller.guided_infill(
        record,
        num_samples=1,
        guidance_strength=1000.0,
        top_k=1,
        order="confidence",
    )
    assert candidates[0].generated_hcdr3[star] == expected_residue


def test_guided_infill_supports_all_orders_and_rejects_unknown(tokenizer):
    # The order parameter is a real, validated knob: each supported schedule
    # produces a well-formed candidate, and an unknown order fails loudly.
    torch.manual_seed(0)
    model = make_tiny_antigen_model(tokenizer)
    record = make_antigen_record()
    infiller = FixedLengthHCDR3Infiller(model, tokenizer, max_length=64, device="cpu")

    for order in ("confidence", "random", "left_to_right"):
        candidates = infiller.guided_infill(
            record,
            num_samples=1,
            guidance_strength=1.0,
            order=order,
            rng=random.Random(0),
        )
        assert len(candidates) == 1
        assert candidates[0].heavy_sequence == "AAAA" + candidates[0].generated_hcdr3 + "YYYY"

    try:
        infiller.guided_infill(record, num_samples=1, order="spiral")
    except ValueError:
        pass
    else:  # pragma: no cover - explicit failure path
        raise AssertionError("guided_infill must reject an unknown order")


# ---------------------------------------------------------------------------
# Regression: the generation/scoring CLI must propagate the checkpoint's antigen
# encoder settings into the infiller and scorer. Dropping them made an ESM-trained
# checkpoint tokenize the antigen with the scratch tokenizer and feed scratch ids
# into an ESM encoder (a silent train/inference mismatch). These tests pin the
# wiring without needing ESM weights or a real checkpoint by stubbing the
# constructors and capturing the kwargs they receive.
# ---------------------------------------------------------------------------


def _load_hcdr3_infill_module():
    scripts_dir = Path(__file__).resolve().parents[3] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    script_path = scripts_dir / "hcdr3_infill.py"
    spec = importlib.util.spec_from_file_location("hcdr3_infill", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _esm_cfg(module):
    return module.TrainConfig(
        data_path="unused.jsonl.gz",
        training_stage="antigen_hcdr3_infill_refine",
        init_checkpoint="ckpt.pt",
        max_length=192,
        antigen_encoder_type="esm",
        esm_model_name="facebook/esm2_t6_8M_UR50D",
        antigen_max_length=333,
    )


def test_build_infiller_threads_antigen_encoder_config(tokenizer, monkeypatch):
    module = _load_hcdr3_infill_module()
    captured: dict = {}

    class _StubInfiller:
        def __init__(self, model, tok, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(module, "FixedLengthHCDR3Infiller", _StubInfiller)
    module.build_infiller(object(), tokenizer, _esm_cfg(module), "cpu")

    assert captured["antigen_encoder_type"] == "esm"
    assert captured["esm_model_name"] == "facebook/esm2_t6_8M_UR50D"
    assert captured["antigen_max_length"] == 333
    assert captured["max_length"] == 192


def test_build_compatibility_scorer_threads_antigen_encoder_config(tokenizer, monkeypatch):
    module = _load_hcdr3_infill_module()
    captured: dict = {}

    class _StubScorer:
        def __init__(self, model, tok, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(module, "AntigenCompatibilityScorer", _StubScorer)
    module.build_compatibility_scorer(object(), tokenizer, _esm_cfg(module), "cpu")

    assert captured["antigen_encoder_type"] == "esm"
    assert captured["esm_model_name"] == "facebook/esm2_t6_8M_UR50D"
    assert captured["antigen_max_length"] == 333
    assert captured["max_length"] == 192


def test_build_infiller_scratch_default_uses_scratch_tokenizer(tokenizer):
    # The scratch path must stay byte-identical: a real infiller built from a
    # scratch config encodes the antigen with the repo tokenizer at max_length.
    module = _load_hcdr3_infill_module()
    cfg = module.TrainConfig(
        data_path="unused.jsonl.gz",
        training_stage="antigen_real_label_refine",
        init_checkpoint="ckpt.pt",
        max_length=192,
    )
    infiller = module.build_infiller(object(), tokenizer, cfg, "cpu")
    assert infiller._antigen_encode_max_length == 192
    # Scratch antigen encoding is defined to equal encode_sequence(locus=None).
    antigen = "MKTIIALSYIFCLVFA"
    assert infiller.antigen_tokenizer.encode(antigen, 192) == tokenizer.encode_sequence(
        antigen, locus=None, max_length=192
    )


def test_generation_run_with_no_candidates_fails_loudly():
    # Regression: a run that skips every proposed length (e.g. every record is
    # missing its antigen sequence, or the wrong data file was passed) must not
    # write an empty JSONL and exit 0 — that silent no-op looks like success.
    module = _load_hcdr3_infill_module()
    import pytest as _pytest

    with _pytest.raises(SystemExit):
        module.assert_generated_any([], skipped_count=8)

    # A run that produced at least one candidate is fine (no raise).
    module.assert_generated_any([{"record_id": "r"}], skipped_count=3)
