"""Tests for the frozen-input evaluation artifact.

The load-bearing tests are the positional ones. A partner probe compares the
same predicted chain under different partners; if the predicted chain MOVES
between conditions, the measured "partner effect" is partly a positional effect
and the number is worthless. `test_light_direction_rejects_a_length_changing_donor`
and `test_verify_catches_a_drifted_case` are the two that make that failure
impossible to ship.
"""
from __future__ import annotations

import copy

import pytest
import torch

from smallAntibodyGen.data.MLMCollator import MLMCollator, OASRecord
from smallAntibodyGen.evaluation import frozen_inputs as fz
from smallAntibodyGen.tokenizer import AminoAcidTokenizer


HEAVY_A = "QVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVS"
HEAVY_B = "EVQLVESGGGLVQPGRSLRLSCAASGFTFDDYAMHWVRQAPGKGLEWVS"   # same length as A
HEAVY_SHORT = "QVQLVESGGGLVQPGGSLRLSCAAS"                          # deliberately shorter
LIGHT_A = "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIY"
LIGHT_B = "EIVLTQSPGTLSLSPGERATLSCRASQSVSSSYLAWYQQKPGQAPRLLIY"  # different length, fine


def _tokenizer() -> AminoAcidTokenizer:
    return AminoAcidTokenizer()


def _collator(**overrides) -> MLMCollator:
    base = dict(
        tokenizer=_tokenizer(),
        max_length=256,
        mask_probability=0.15,
        hcdr3_span_probability=0.0,   # deterministic-ish: plain random masking
        shuffle_pair_probability=0.0,  # freezing must not shuffle behind our back
        rng_seed=7,
    )
    base.update(overrides)
    return MLMCollator(**base)


def _full_span_collator() -> MLMCollator:
    """Full-span probes need BOTH knobs; selection alone leaves part visible."""
    return _collator(hcdr3_mask_mode="full_span", mask_replacement_strategy="always_mask")


def _pool(records):
    return [(fz.derive_artifact_id(r)[0], r) for r in records]


def _paired(record_id: str, heavy: str, light: str) -> OASRecord:
    return OASRecord(
        sequence=heavy,
        locus="IGH",
        chain_group="heavy",
        split="val",
        length=len(heavy),
        sequence_heavy=heavy,
        sequence_light=light,
        heavy_locus="IGH",
        light_locus="IGK",
        is_paired=True,
        record_id=record_id,
        cdr3_aa_heavy="ARGGYYYYDY",
        cdr3_start_aa_heavy=20,
        cdr3_end_aa_heavy=30,
    )


def _single(record_id: str, seq: str) -> OASRecord:
    return OASRecord(
        sequence=seq,
        locus="IGH",
        chain_group="heavy",
        split="val",
        length=len(seq),
        record_id=record_id,
    )


# --------------------------------------------------------------------------- #
# Encoding layout -- the fact every positional claim rests on
# --------------------------------------------------------------------------- #
def test_heavy_span_is_independent_of_the_light_partner():
    collator = _collator()
    record = _paired("r0", HEAVY_A, LIGHT_A)
    swapped = copy.copy(record)
    swapped.sequence_light = LIGHT_B

    assert fz.heavy_span(record) == fz.heavy_span(swapped)
    ids_native = torch.tensor(collator._encode_record(record))
    ids_swapped = torch.tensor(collator._encode_record(swapped))
    start, end = fz.heavy_span(record)
    assert torch.equal(ids_native[start:end], ids_swapped[start:end])


def test_light_span_moves_when_the_heavy_partner_changes_length():
    """The trap, stated as a test: this is why exact-length matching is required."""
    collator = _collator()
    record = _paired("r0", HEAVY_A, LIGHT_A)
    shorter = copy.copy(record)
    shorter.sequence_heavy = HEAVY_SHORT

    assert fz.light_span(record) != fz.light_span(shorter)


# --------------------------------------------------------------------------- #
# Donor selection
# --------------------------------------------------------------------------- #
def test_light_direction_rejects_a_length_changing_donor():
    collator = _collator()
    record = _paired("r0", HEAVY_A, LIGHT_A)
    pool = [record, _paired("r1", HEAVY_SHORT, LIGHT_B)]

    assert not fz.eligible_donors(
        record, _pool(pool), fz.LIGHT_GIVEN_HEAVY,
        require_exact_length=True, policy=fz.DonorPolicy(),
    ), "a shorter heavy donor would shift every light position"

    pool.append(_paired("r2", HEAVY_B, LIGHT_B))
    donors = fz.eligible_donors(
        record, _pool(pool), fz.LIGHT_GIVEN_HEAVY,
        require_exact_length=True, policy=fz.DonorPolicy(),
    )
    assert [d[0] for d in donors] == ["r2"]


def test_donor_identical_to_the_native_partner_is_refused():
    """An identical donor makes the condition a no-op and deflates the effect."""
    collator = _collator()
    record = _paired("r0", HEAVY_A, LIGHT_A)
    twin = _paired("r1", HEAVY_B, LIGHT_A)  # same LIGHT as the native record
    assert not fz.eligible_donors(
        record, _pool([twin]), fz.HEAVY_GIVEN_LIGHT,
        require_exact_length=False, policy=fz.DonorPolicy(),
    )


def test_neutral_filler_preserves_partner_length():
    record = _paired("r0", HEAVY_A, LIGHT_A)
    filled = fz._synthetic_filler_donor(record, fz.LIGHT_GIVEN_HEAVY)
    assert len(filled.sequence_heavy) == len(HEAVY_A)
    assert set(filled.sequence_heavy) == {fz.DEFAULT_FILLER_RESIDUE}


def test_multi_character_filler_is_refused():
    """`tokenizer.mask_token` is '[MASK]' and would expand to seven tokens."""
    record = _paired("r0", HEAVY_A, LIGHT_A)
    with pytest.raises(fz.FrozenInputsError, match="single character"):
        fz._synthetic_filler_donor(record, fz.LIGHT_GIVEN_HEAVY, filler="[MASK]")


# --------------------------------------------------------------------------- #
# The frozen partner cases
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "direction", [fz.HEAVY_GIVEN_LIGHT, fz.LIGHT_GIVEN_HEAVY]
)
def test_partner_cases_present_the_predicted_chain_identically(direction: str):
    collator = _collator()
    records = [
        _paired("r0", HEAVY_A, LIGHT_A),
        _paired("r1", HEAVY_B, LIGHT_B),
    ]
    cases = fz.build_partner_cases(records, collator, direction)
    assert cases, "no cases built; the probe would be vacuous"

    by_group: dict[str, dict[str, fz.FrozenCase]] = {}
    for case in cases:
        by_group.setdefault(case.notes["group"], {})[case.condition] = case

    for group, members in by_group.items():
        assert set(members) == set(fz.CONDITIONS)
        ref = members[fz.NATIVE]
        idx = list(ref.target_positions)
        for condition, case in members.items():
            assert case.target_positions == ref.target_positions
            assert torch.equal(case.input_ids[idx], ref.input_ids[idx])
            assert torch.equal(case.labels[idx], ref.labels[idx])


def test_partner_conditions_actually_differ_off_the_predicted_chain():
    """Fault check: if the partner never changed, the probe measures nothing."""
    collator = _collator()
    records = [_paired("r0", HEAVY_A, LIGHT_A), _paired("r1", HEAVY_B, LIGHT_B)]
    cases = fz.build_partner_cases(records, collator, fz.HEAVY_GIVEN_LIGHT)
    groups: dict[str, dict[str, fz.FrozenCase]] = {}
    for case in cases:
        groups.setdefault(case.notes["group"], {})[case.condition] = case

    group = next(iter(groups.values()))
    native, alt = group[fz.NATIVE], group[fz.MATCHED_ALTERNATIVE]
    assert not torch.equal(native.input_ids, alt.input_ids)


def test_absent_mechanism_is_recorded_and_direction_specific():
    collator = _collator()
    records = [_paired("r0", HEAVY_A, LIGHT_A), _paired("r1", HEAVY_B, LIGHT_B)]

    heavy_cases = fz.build_partner_cases(records, collator, fz.HEAVY_GIVEN_LIGHT)
    light_cases = fz.build_partner_cases(records, collator, fz.LIGHT_GIVEN_HEAVY)

    heavy_absent = [c for c in heavy_cases if c.condition == fz.ABSENT]
    light_absent = [c for c in light_cases if c.condition == fz.ABSENT]
    assert heavy_absent and light_absent
    # Removing the light chain preserves heavy positions; removing the heavy
    # chain would not, so the two directions MUST use different mechanisms.
    assert all(c.absent_mechanism == fz.ABSENT_CHAIN_REMOVED for c in heavy_absent)
    assert all(c.absent_mechanism == fz.ABSENT_SYNTHETIC_FILLER for c in light_absent)


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def _tiny_payload(collator: MLMCollator) -> dict:
    records = [_paired("r0", HEAVY_A, LIGHT_A), _paired("r1", HEAVY_B, LIGHT_B)]
    return fz.build_frozen_benchmark(
        stage1_records=[_single("s0", HEAVY_A), _single("s1", HEAVY_B)],
        paired_records=records,
        retention_collator=collator,
        partner_collator=_collator(),
        full_span_collator=_full_span_collator(),
        provenance={"note": "unit test"},
    )


def test_build_verifies_and_digests():
    payload = _tiny_payload(_collator())
    assert payload["schema"] == fz.FROZEN_SCHEMA
    assert payload["tensor_digest"] == fz.tensor_digest(payload)
    assert payload["semantic_digest"] == fz.semantic_digest(payload)
    fz.verify_frozen(payload)


def test_verify_catches_a_drifted_case():
    """Corrupt one condition's predicted positions; verification must refuse it."""
    payload = _tiny_payload(_collator())
    for entry in payload["probes"][fz.PROBE_PARTNER]:
        if entry["condition"] == fz.MATCHED_ALTERNATIVE:
            pos = entry["target_positions"][0]
            entry["input_ids"][pos] = (entry["input_ids"][pos] + 1) % 30
            break
    with pytest.raises(fz.PositionDriftError):
        fz.verify_frozen(payload)


def test_verify_catches_labels_outside_the_declared_targets():
    payload = _tiny_payload(_collator())
    entry = payload["probes"][fz.PROBE_STAGE1_RETENTION][0]
    stray = next(
        i for i in range(entry["labels"].numel()) if i not in entry["target_positions"]
    )
    entry["labels"][stray] = 5
    with pytest.raises(fz.FrozenInputsError, match="outside the declared targets"):
        fz.verify_frozen(payload)


def test_round_trip_preserves_the_digest(tmp_path):
    payload = _tiny_payload(_collator())
    path = fz.save_frozen(tmp_path / "frozen.pt", payload)
    reloaded = fz.load_frozen(path)
    assert reloaded["tensor_digest"] == payload["tensor_digest"]
    assert reloaded["semantic_digest"] == payload["semantic_digest"]


def test_load_rejects_a_tampered_artifact(tmp_path):
    payload = _tiny_payload(_collator())
    path = fz.save_frozen(tmp_path / "frozen.pt", payload)
    tampered = torch.load(path, map_location="cpu", weights_only=False)
    entry = tampered["probes"][fz.PROBE_STAGE1_RETENTION][0]
    pos = entry["target_positions"][0]
    entry["input_ids"][pos] = (entry["input_ids"][pos] + 1) % 30
    torch.save(tampered, path)
    with pytest.raises(fz.FrozenInputsError, match="tensor_digest mismatch"):
        fz.load_frozen(path)


def test_full_span_probe_targets_whole_hcdr3():
    """Full-span recovery must be its own probe, not ordinary MLM accuracy."""
    records = [_paired("r0", HEAVY_A, LIGHT_A), _paired("r1", HEAVY_B, LIGHT_B)]
    cases = fz.build_single_chain_cases(
        records, _full_span_collator(), fz.PROBE_HCDR3_FULL_SPAN
    )
    assert cases
    for case in cases:
        # Contiguous span, which ordinary 15% random masking would not produce.
        positions = list(case.target_positions)
        assert positions == list(range(positions[0], positions[-1] + 1))
