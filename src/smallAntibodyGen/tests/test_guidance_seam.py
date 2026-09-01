"""Tests for the guidance plumbing ported from the sibling mirror repo:

- ``hcdr3_mask_mode="partial_span"`` — the "noisy" masking schedule that matches
  the partially-filled states guided decoding actually queries the compatibility
  head on.
- The external ``guidance_model`` seam plus its ``--guidance-checkpoint`` CLI
  flag, which lets the steering head be a different head from the generator's.
- The amortized antigen encoding in ``guided_infill``, which must be bit-exact.

Everything here is opt-in; the first tests pin that the defaults are unchanged.
"""
from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from smallAntibodyGen.data.MLMCollator import MLMCollator, OASRecord
from smallAntibodyGen.infill.hcdr3 import FixedLengthHCDR3Infiller, HCDR3Span
from smallAntibodyGen.models.mlm import AntibodyAntigenCrossAttention, MLMConfig
from smallAntibodyGen.tokenizer import AminoAcidTokenizer


def _load_hcdr3_infill(project_root: Path):
    scripts_dir = project_root.parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    script_path = scripts_dir / "hcdr3_infill.py"
    spec = importlib.util.spec_from_file_location("hcdr3_infill", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _span_record(tokenizer, heavy_seq: str, heavy_cdr3: str, antigen: str) -> OASRecord:
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
        sequence_antigen=antigen,
        record_id="r0",
        is_strong_binder=True,
    )


ANTIGEN = "MKTIIALSYIFCLVFADYKDDDDKAMDIGINSDPYQ"


def _dual_model(tokenizer, seed: int = 11) -> AntibodyAntigenCrossAttention:
    torch.manual_seed(seed)
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
    return model


# ------------------------------------------------------- partial_span masking


def _partial_collator(tokenizer, seed: int) -> MLMCollator:
    return MLMCollator(
        tokenizer=tokenizer,
        max_length=192,
        mask_probability=0.15,
        hcdr3_span_probability=0.5,
        hcdr3_mask_mode="partial_span",
        rng_seed=seed,
    )


def test_partial_span_targets_only_hcdr3_positions(tokenizer, heavy_seq, heavy_cdr3):
    """No non-span position may ever become a target: the mode exists to
    reproduce a partially-filled HCDR3, not to add global corruption."""
    record = _span_record(tokenizer, heavy_seq, heavy_cdr3, ANTIGEN)
    for seed in range(20):
        collator = _partial_collator(tokenizer, seed)
        batch = collator([record])
        row = torch.tensor(record.token_ids, dtype=torch.long)
        span_positions = set(collator._heavy_hcdr3_positions(row, record))
        target_positions = set((batch["labels"][0] != -100).nonzero().flatten().tolist())
        assert target_positions <= span_positions


def test_partial_span_reaches_both_endpoints_of_k(tokenizer, heavy_seq, heavy_cdr3):
    """k ~ uniform{0..L}: k == 0 (fully visible, the finished-decode endpoint)
    and k == L (the full_span state) must both be reachable, or the schedule does
    not cover the states decoding visits."""
    record = _span_record(tokenizer, heavy_seq, heavy_cdr3, ANTIGEN)
    probe = _partial_collator(tokenizer, 0)
    row = torch.tensor(record.token_ids, dtype=torch.long)
    span_len = len(probe._heavy_hcdr3_positions(row, record))
    assert span_len > 0

    observed = set()
    for seed in range(300):
        collator = _partial_collator(tokenizer, seed)
        batch = collator([record])
        observed.add(int((batch["labels"][0] != -100).sum().item()))
    assert 0 in observed
    assert span_len in observed
    assert observed <= set(range(span_len + 1))


def test_partial_span_is_strict_about_invalid_spans(tokenizer, heavy_seq):
    """Like ``full_span``: a record with no valid span contributes zero targets
    rather than silently falling back to random masking."""
    record = OASRecord(
        sequence=heavy_seq,
        token_ids=tokenizer.encode_sequence(heavy_seq, locus="IGH", max_length=192),
        locus="IGH",
        chain_group="heavy",
        split="train",
        length=len(heavy_seq),
        token_length=len(heavy_seq) + 3,
    )
    collator = _partial_collator(tokenizer, 3)
    batch = collator([record])
    assert int((batch["labels"] != -100).sum().item()) == 0


def test_partial_span_ignores_mask_probability(tokenizer, heavy_seq, heavy_cdr3):
    record = _span_record(tokenizer, heavy_seq, heavy_cdr3, ANTIGEN)
    low = MLMCollator(
        tokenizer=tokenizer,
        max_length=192,
        mask_probability=0.01,
        hcdr3_mask_mode="partial_span",
        rng_seed=7,
    )
    high = MLMCollator(
        tokenizer=tokenizer,
        max_length=192,
        mask_probability=0.99,
        hcdr3_mask_mode="partial_span",
        rng_seed=7,
    )
    assert torch.equal(low([record])["labels"], high([record])["labels"])


def test_default_mask_mode_is_unchanged(tokenizer):
    assert MLMCollator(tokenizer=tokenizer, max_length=64).hcdr3_mask_mode == "sampled_span"


# ------------------------------------------------------------ guidance seam


def _infiller(tokenizer, model, **kwargs) -> FixedLengthHCDR3Infiller:
    return FixedLengthHCDR3Infiller(model, tokenizer, max_length=192, device="cpu", **kwargs)


def test_guidance_seam_is_provably_inert_at_gamma_zero(
    tokenizer, heavy_seq, heavy_cdr3
):
    """A DELIBERATELY perturbed guidance model attached at gamma == 0 must change
    nothing: gamma == 0 takes the unguided branch and no classifier is consulted."""
    record = _span_record(tokenizer, heavy_seq, heavy_cdr3, ANTIGEN)
    model = _dual_model(tokenizer)
    poisoned = copy.deepcopy(model)
    with torch.no_grad():
        for param in poisoned.parameters():
            param.add_(torch.randn_like(param) * 5.0)

    torch.manual_seed(99)
    plain = _infiller(tokenizer, model).guided_infill(
        record, num_samples=2, guidance_strength=0.0
    )
    torch.manual_seed(99)
    seamed = _infiller(tokenizer, model, guidance_model=poisoned).guided_infill(
        record, num_samples=2, guidance_strength=0.0
    )
    assert [c.generated_hcdr3 for c in plain] == [c.generated_hcdr3 for c in seamed]
    assert [c.log_probability for c in plain] == [c.log_probability for c in seamed]


def test_attached_guidance_model_supplies_the_binder_term(
    tokenizer, heavy_seq, heavy_cdr3
):
    """At gamma > 0 the attached classifier -- not the generation model's own
    head -- must drive the binder term.

    Pinned at the scores level (the values `_binder_logprobs_by_candidate`
    returns), not at the sampled-string level: two random heads can happen to
    rank residues the same way, so a string comparison is a flaky proxy for the
    routing contract this test exists to protect.
    """
    record = _span_record(tokenizer, heavy_seq, heavy_cdr3, ANTIGEN)
    model = _dual_model(tokenizer, seed=11)
    other = _dual_model(tokenizer, seed=4242)

    plain = _infiller(tokenizer, model)
    seamed = _infiller(tokenizer, model, guidance_model=other)

    antigen_ids, antigen_mask = plain._encode_antigen(record)
    span = HCDR3Span.from_record(record)
    ab_ids, ab_mask, mask_positions, _, _ = plain._encode_antibody_with_masked_hcdr3(
        record, span, proposed_length=span.length
    )
    position = mask_positions[0]

    with torch.no_grad():
        self_scores = plain._binder_logprobs_by_candidate(
            ab_ids, ab_mask, antigen_ids, antigen_mask, position
        )
        seam_scores = seamed._binder_logprobs_by_candidate(
            ab_ids, ab_mask, antigen_ids, antigen_mask, position
        )
        # What the OTHER model returns when asked directly.
        reference = _infiller(tokenizer, other)._binder_logprobs_by_candidate(
            ab_ids, ab_mask, antigen_ids, antigen_mask, position
        )

    assert torch.equal(seam_scores, reference)
    assert not torch.allclose(seam_scores, self_scores)


def test_guided_decoding_actually_calls_the_attached_guidance_model(
    tokenizer, heavy_seq, heavy_cdr3
):
    """Routing pin over a full decode: the enumeration forwards go to the
    guidance model and the ordering forwards stay with the generation model."""
    record = _span_record(tokenizer, heavy_seq, heavy_cdr3, ANTIGEN)
    model = _dual_model(tokenizer, seed=11)
    other = _dual_model(tokenizer, seed=4242)
    calls = {"generation": 0, "guidance": 0}

    def counting(module, key):
        original = module.forward

        def wrapped(*args, **kwargs):
            calls[key] += 1
            return original(*args, **kwargs)

        module.forward = wrapped

    counting(model, "generation")
    counting(other, "guidance")

    span = HCDR3Span.from_record(record)
    torch.manual_seed(3)
    _infiller(tokenizer, model, guidance_model=other).guided_infill(
        record, num_samples=1, guidance_strength=1.0, order="left_to_right"
    )
    # One ordering forward and one enumeration forward per unmasked position.
    assert calls["generation"] == span.length
    assert calls["guidance"] == span.length


def test_guidance_model_fails_loud_on_state_longer_than_its_max_length(
    tokenizer, heavy_seq, heavy_cdr3
):
    """The scorer truncates long antigens with a warning; the guidance model must
    NOT inherit that -- overrunning its positional embeddings would corrupt every
    steering decision silently."""
    record = _span_record(tokenizer, heavy_seq, heavy_cdr3, ANTIGEN)
    model = _dual_model(tokenizer)
    infiller = _infiller(
        tokenizer, model, guidance_model=copy.deepcopy(model), guidance_max_length=8
    )
    with pytest.raises(ValueError, match="exceeds the guidance checkpoint's max_length"):
        infiller.guided_infill(record, num_samples=1, guidance_strength=1.0)


def test_guidance_antigen_encoding_uses_the_guidance_tokenizer(
    tokenizer, heavy_seq, heavy_cdr3
):
    """The guidance model tokenizes antigens with ITS OWN effective max length,
    not the generation model's.

    Scope note: this varies only ``guidance_max_length``, i.e. it pins the
    "``guidance_antigen_max_length`` is None, so inherit the guidance ANTIBODY
    budget" half of the rule. The ``guidance_antigen_max_length`` knob itself is
    covered by the two AB-07 tests below -- this one passes either way.
    """
    record = _span_record(tokenizer, heavy_seq, heavy_cdr3, ANTIGEN)
    model = _dual_model(tokenizer)
    infiller = _infiller(
        tokenizer, model, guidance_model=copy.deepcopy(model), guidance_max_length=16
    )
    gen_ids, _ = infiller._encode_antigen(record)
    guide_ids, _ = infiller._encode_guidance_antigen(record)
    assert guide_ids.size(1) <= 16
    assert gen_ids.size(1) != guide_ids.size(1)


# A long antigen, so every budget under test actually truncates and the observed
# token count is the budget rather than the sequence length.
LONG_ANTIGEN = ANTIGEN * 8  # 288 residues -> 290 tokens


def test_guidance_antigen_budget_is_honoured_for_a_scratch_encoder(
    tokenizer, heavy_seq, heavy_cdr3
):
    """AB-07 at its fourth call site.

    The guidance seam landed one commit before the AB-07 fix and kept a private
    copy of the pre-AB-07 expression, whose ``scratch`` branch clamped the antigen
    to the guidance model's ANTIBODY budget -- making
    ``guidance_antigen_max_length`` inert exactly where the shipped chain sets it
    (max_length 288, antigen_max_length 1024). The resolved budget must not depend
    on the encoder type.
    """
    record = _span_record(tokenizer, heavy_seq, heavy_cdr3, LONG_ANTIGEN)
    model = _dual_model(tokenizer)
    infiller = _infiller(
        tokenizer,
        model,
        guidance_model=copy.deepcopy(model),
        guidance_antigen_encoder_type="scratch",
        guidance_max_length=160,
        guidance_antigen_max_length=96,
    )
    # Pre-fix both of these were 160 -- the guidance ANTIBODY budget.
    assert infiller._guidance_antigen_encode_max_length == 96
    guide_ids, _ = infiller._encode_guidance_antigen(record)
    assert guide_ids.size(1) == 96


def test_guided_decoding_feeds_the_guidance_model_its_own_antigen_budget(
    tokenizer, heavy_seq, heavy_cdr3
):
    """The budget must survive to the enumeration forward, not just to the
    private encode helper.

    ``guided_infill`` builds the ~20-way enumeration antigen state once from the
    guidance antigen stream, so a budget clamped back to the antibody length
    silently truncates the antigen behind EVERY steering decision -- with no
    error and no shape mismatch, because the guidance encoder's positional table
    is larger, not smaller.
    """
    record = _span_record(tokenizer, heavy_seq, heavy_cdr3, LONG_ANTIGEN)
    model = _dual_model(tokenizer, seed=11)
    guide = _dual_model(tokenizer, seed=4242)

    seen_antigen_lengths: list[int] = []
    original_encode_antigen = guide.encode_antigen

    def spy(antigen_input_ids, *args, **kwargs):
        seen_antigen_lengths.append(int(antigen_input_ids.size(1)))
        return original_encode_antigen(antigen_input_ids, *args, **kwargs)

    guide.encode_antigen = spy
    infiller = _infiller(
        tokenizer,
        model,
        guidance_model=guide,
        guidance_antigen_encoder_type="scratch",
        guidance_max_length=192,
        guidance_antigen_max_length=64,
    )
    infiller.guided_infill(
        record, num_samples=1, guidance_strength=1.0, order="left_to_right"
    )
    # One enumeration state per guided_infill call, at the guidance ANTIGEN
    # budget. Pre-fix this was [192] -- the guidance antibody budget.
    assert seen_antigen_lengths == [64]


# ------------------------------------------------- amortized antigen encoding


def test_antigen_state_cache_is_bit_exact(tokenizer, heavy_seq, heavy_cdr3):
    """The cached pre-fuse antigen encoding must reproduce the uncached forward
    exactly -- it is a determinism optimization, not an approximation."""
    record = _span_record(tokenizer, heavy_seq, heavy_cdr3, ANTIGEN)
    model = _dual_model(tokenizer)
    infiller = _infiller(tokenizer, model)
    antigen_ids, antigen_mask = infiller._encode_antigen(record)
    span = HCDR3Span.from_record(record)
    ab_ids, ab_mask, _, _, _ = infiller._encode_antibody_with_masked_hcdr3(
        record, span, proposed_length=span.length
    )
    with torch.no_grad():
        uncached = model(
            antibody_input_ids=ab_ids,
            antibody_attention_mask=ab_mask,
            antigen_input_ids=antigen_ids,
            antigen_attention_mask=antigen_mask,
        )
        state = model.encode_antigen(antigen_ids, antigen_mask)
        cached = model(
            antibody_input_ids=ab_ids,
            antibody_attention_mask=ab_mask,
            antigen_input_ids=antigen_ids,
            antigen_attention_mask=antigen_mask,
            antigen_state=state,
        )
    for a, b in zip(uncached, cached):
        assert torch.equal(a, b)


def test_binder_logprobs_cache_matches_uncached(tokenizer, heavy_seq, heavy_cdr3):
    record = _span_record(tokenizer, heavy_seq, heavy_cdr3, ANTIGEN)
    model = _dual_model(tokenizer)
    infiller = _infiller(tokenizer, model)
    antigen_ids, antigen_mask = infiller._encode_antigen(record)
    span = HCDR3Span.from_record(record)
    ab_ids, ab_mask, mask_positions, _, _ = infiller._encode_antibody_with_masked_hcdr3(
        record, span, proposed_length=span.length
    )
    position = mask_positions[0]
    n = len(infiller.canonical_token_ids)
    with torch.no_grad():
        uncached = infiller._binder_logprobs_by_candidate(
            ab_ids, ab_mask, antigen_ids, antigen_mask, position
        )
        state = model.encode_antigen(
            antigen_ids.repeat(n, 1), antigen_mask.repeat(n, 1)
        )
        cached = infiller._binder_logprobs_by_candidate(
            ab_ids, ab_mask, antigen_ids, antigen_mask, position, antigen_state=state
        )
    assert torch.equal(uncached, cached)


def test_guided_infill_is_unchanged_by_the_cache(tokenizer, heavy_seq, heavy_cdr3):
    """End-to-end: the amortized loop reproduces the sequences the uncached loop
    would have produced, at the same torch seed."""
    record = _span_record(tokenizer, heavy_seq, heavy_cdr3, ANTIGEN)
    model = _dual_model(tokenizer)
    infiller = _infiller(tokenizer, model)

    torch.manual_seed(1234)
    with_cache = infiller.guided_infill(
        record, num_samples=2, guidance_strength=2.0, order="left_to_right"
    )

    # Reconstruct the pre-cache behavior. Both sites must drop the cache
    # together: _binder_logprobs_by_candidate decides whether to repeat the
    # antigen tensors based on whether it was HANDED a cache, so clearing the
    # cache only inside forward would leave a batch-1 antigen against a batch-20
    # antibody.
    original_forward = type(model).forward
    original_binder = type(infiller)._binder_logprobs_by_candidate

    def forward_without_cache(self, *args, **kwargs):
        kwargs["antigen_state"] = None
        return original_forward(self, *args, **kwargs)

    def binder_without_cache(self, *args, **kwargs):
        kwargs["antigen_state"] = None
        return original_binder(self, *args, **kwargs)

    type(model).forward = forward_without_cache
    type(infiller)._binder_logprobs_by_candidate = binder_without_cache
    try:
        torch.manual_seed(1234)
        without_cache = infiller.guided_infill(
            record, num_samples=2, guidance_strength=2.0, order="left_to_right"
        )
    finally:
        type(model).forward = original_forward
        type(infiller)._binder_logprobs_by_candidate = original_binder

    assert [c.generated_hcdr3 for c in with_cache] == [
        c.generated_hcdr3 for c in without_cache
    ]
    assert [c.log_probability for c in with_cache] == [
        c.log_probability for c in without_cache
    ]


# ------------------------------------------------------------------- CLI seam


def test_guidance_checkpoint_at_gamma_zero_is_a_parser_error(project_root: Path):
    """Accepting a guidance checkpoint that provably never runs would label sweep
    rows with a classifier that touched nothing."""
    hcdr3_infill = _load_hcdr3_infill(project_root)
    with pytest.raises(SystemExit):
        hcdr3_infill.main(
            [
                "--checkpoint",
                "does_not_matter.pt",
                "--data-path",
                "does_not_matter.jsonl.gz",
                "--guidance-checkpoint",
                "guide.pt",
                "--guidance-strength",
                "0",
            ]
        )


def test_candidate_row_records_the_guidance_checkpoint(project_root: Path, tokenizer):
    hcdr3_infill = _load_hcdr3_infill(project_root)

    class _Candidate:
        length = 5
        generated_hcdr3 = "ARDYW"
        heavy_sequence = "XXARDYWXX"
        log_probability = -1.0
        mean_log_probability = -0.2
        compatibility_score = None

    class _Record:
        record_id = "r0"
        target_key = "uniprot:p1"
        target_name = "t"
        split = "val"

    class _Span:
        original_hcdr3 = "ARDYW"
        length = 5

    row = hcdr3_infill.candidate_to_json(
        record=_Record(),
        true_span=_Span(),
        length_mode="fixed",
        candidate=_Candidate(),
        guidance_strength=2.0,
        guidance_order="confidence",
        guidance_checkpoint="checkpoints/guide/best.pt",
    )
    assert row["guidance_checkpoint"] == "checkpoints/guide/best.pt"

    off = hcdr3_infill.candidate_to_json(
        record=_Record(),
        true_span=_Span(),
        length_mode="fixed",
        candidate=_Candidate(),
        guidance_strength=0.0,
        guidance_checkpoint="checkpoints/guide/best.pt",
    )
    assert off["guidance_checkpoint"] is None


# ------------------------------------------ stale-compat-head guidance guard


def _stage4_checkpoint(hcdr3_infill, tmp_path, *, compatibility_loss_weight, name):
    """
    Write a real, strictly-loadable dual-stream checkpoint for the infill stage.

    Built through the CLI's OWN ``build_model``/``TrainConfig`` so the saved
    state dict is exactly what ``load_dual_stream_model`` reconstructs; the guard
    under test runs after that load, so a hand-rolled state dict would fail
    earlier for an unrelated reason.
    """
    from dataclasses import asdict

    merged = hcdr3_infill._train_config_defaults()
    merged.update(
        training_stage="antigen_hcdr3_infill_refine",
        data_path="unused.jsonl.gz",
        # The stage validator requires a warm-start source; never read here.
        init_checkpoint="unused_stage3.pt",
        max_length=64,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
        compatibility_loss_weight=compatibility_loss_weight,
    )
    cfg = hcdr3_infill.TrainConfig(**merged)
    model = hcdr3_infill.build_model(
        hcdr3_infill.build_tokenizer(), cfg, torch.device("cpu")
    )
    path = tmp_path / name
    torch.save({"model_state_dict": model.state_dict(), "train_config": asdict(cfg)}, path)
    return path


def _run_guided_cli(hcdr3_infill, checkpoint, *extra):
    """Invoke the CLI far enough to reach the guard, returning the raised error."""
    with pytest.raises(BaseException) as excinfo:
        hcdr3_infill.main(
            [
                "--checkpoint",
                str(checkpoint),
                "--data-path",
                "no_such_dataset.jsonl.gz",
                "--guidance-strength",
                "1.0",
                "--no-score",
                *extra,
            ]
        )
    return str(excinfo.value)


def test_guidance_refuses_a_compat_head_that_got_zero_loss_weight(
    project_root: Path, tmp_path: Path
):
    """The shipped stage-4 config sets compatibility_loss_weight: 0.0, so that
    checkpoint's compat head received no gradient while the encoder feeding it
    kept training. Steering by it silently produces a stale readout, and a gamma
    sweep against it cannot separate 'guidance does not help' from 'the steerer
    is not a classifier'. Refuse rather than emit the numbers."""
    hcdr3_infill = _load_hcdr3_infill(project_root)
    stale = _stage4_checkpoint(
        hcdr3_infill, tmp_path, compatibility_loss_weight=0.0, name="stage4_stale.pt"
    )
    message = _run_guided_cli(hcdr3_infill, stale)
    assert "compatibility_loss_weight = 0" in message
    assert "--allow-untrained-guidance-head" in message


def test_guidance_guard_passes_when_the_compat_head_was_trained(
    project_root: Path, tmp_path: Path
):
    """Positive control: the guard must be keyed on the loss weight, not on
    'guidance is on'. A head that was actually trained gets through, and the run
    proceeds to fail on the (deliberately missing) dataset instead."""
    hcdr3_infill = _load_hcdr3_infill(project_root)
    trained = _stage4_checkpoint(
        hcdr3_infill, tmp_path, compatibility_loss_weight=1.0, name="stage4_trained.pt"
    )
    message = _run_guided_cli(hcdr3_infill, trained)
    assert "refusing to guide" not in message


def test_guidance_guard_is_overridable_for_a_deliberate_negative_control(
    project_root: Path, tmp_path: Path
):
    """The override exists so a negative control is still runnable; without it the
    guard would be a wall rather than a speed bump."""
    hcdr3_infill = _load_hcdr3_infill(project_root)
    stale = _stage4_checkpoint(
        hcdr3_infill, tmp_path, compatibility_loss_weight=0.0, name="stage4_override.pt"
    )
    message = _run_guided_cli(
        hcdr3_infill, stale, "--allow-untrained-guidance-head"
    )
    assert "refusing to guide" not in message


def test_guidance_guard_is_not_consulted_at_gamma_zero(
    project_root: Path, tmp_path: Path
):
    """At gamma == 0 no classifier is consulted, so the head's provenance is
    irrelevant and the guard must not fire -- otherwise the zero-weight stage-4
    checkpoint could not be used for ordinary unguided infilling at all."""
    hcdr3_infill = _load_hcdr3_infill(project_root)
    stale = _stage4_checkpoint(
        hcdr3_infill, tmp_path, compatibility_loss_weight=0.0, name="stage4_gamma0.pt"
    )
    with pytest.raises(BaseException) as excinfo:
        hcdr3_infill.main(
            [
                "--checkpoint",
                str(stale),
                "--data-path",
                "no_such_dataset.jsonl.gz",
                "--guidance-strength",
                "0",
                "--no-score",
            ]
        )
    assert "refusing to guide" not in str(excinfo.value)


def test_guidance_guard_also_covers_an_external_guidance_checkpoint(
    project_root: Path, tmp_path: Path
):
    """The guard is keyed on the checkpoint that actually supplies the binder
    term, so routing a zero-weight head in through --guidance-checkpoint is not a
    way around it."""
    hcdr3_infill = _load_hcdr3_infill(project_root)
    trained = _stage4_checkpoint(
        hcdr3_infill, tmp_path, compatibility_loss_weight=1.0, name="gen_trained.pt"
    )
    stale_guide = _stage4_checkpoint(
        hcdr3_infill, tmp_path, compatibility_loss_weight=0.0, name="guide_stale.pt"
    )
    message = _run_guided_cli(
        hcdr3_infill, trained, "--guidance-checkpoint", str(stale_guide)
    )
    assert "--guidance-checkpoint" in message
    assert "compatibility_loss_weight = 0" in message
