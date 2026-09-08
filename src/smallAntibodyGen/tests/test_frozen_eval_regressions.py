"""Regressions from the schema-v1 review of the frozen-eval artifact.

Every test here FAILED to exist when the corresponding defect shipped, and each
defect passed the v1 suite. Kept in their own module so the list of things that
once slipped through stays legible.

The five:

1. Donor self-exclusion compared `record_id`, which is `None` on real processed
   paired rows -- `None == None` rejected every candidate, the partner probe was
   empty, and verification passed because every check over zero cases is true.
2. `hcdr3_mask_mode="full_span"` selects the span but does not HIDE it; BERT
   80/10/10 replacement left part of the "hidden" HCDR3 readable.
3. Region spans were not frozen, so the artifact could not answer the
   HCDR3-specific questions it was built for.
4. Verification compared only target positions, so context elsewhere in the
   predicted chain -- and the attention mask -- could drift undetected; and the
   digest ignored direction, donor, grouping and absent mechanism.
5. Donor choice was "first eligible" with no length bound: order dependent, and
   it accepted a 3-residue partner against a 49-residue native one.
"""
from __future__ import annotations

import random as _random

import pytest

from smallAntibodyGen.evaluation import frozen_inputs as fz
from smallAntibodyGen.tests.test_frozen_eval_inputs import (
    HEAVY_A,
    HEAVY_B,
    LIGHT_A,
    LIGHT_B,
    _collator,
    _full_span_collator,
    _paired,
    _pool,
    _tiny_payload,
    _tokenizer,
)


def _no_id_paired(pair_id, heavy, light):
    """A record shaped like a real processed OAS row: pair_id, no record_id."""
    record = _paired("unused", heavy, light)
    record.record_id = None
    record.pair_id = pair_id
    return record


# --------------------------------------------------------------------------- #
# 1. Identity
# --------------------------------------------------------------------------- #
def test_artifact_id_falls_back_when_record_id_is_absent():
    assert fz.derive_artifact_id(_paired("r0", HEAVY_A, LIGHT_A)) == (
        "r0",
        fz.ID_FROM_RECORD_ID,
    )
    assert fz.derive_artifact_id(_no_id_paired("p0", HEAVY_A, LIGHT_A)) == (
        "p0",
        fz.ID_FROM_PAIR_ID,
    )

    anonymous = _no_id_paired(None, HEAVY_A, LIGHT_A)
    ident, source = fz.derive_artifact_id(anonymous)
    assert source == fz.ID_FROM_CONTENT and len(ident) == 24
    assert fz.derive_artifact_id(anonymous)[0] == ident, "must be stable"
    assert fz.derive_artifact_id(_no_id_paired(None, HEAVY_B, LIGHT_B))[0] != ident


def test_records_without_record_id_still_produce_partner_cases():
    """The defect itself: `None == None` emptied the probe, silently."""
    records = [
        _no_id_paired("p0", HEAVY_A, LIGHT_A),
        _no_id_paired("p1", HEAVY_B, LIGHT_B),
    ]
    cases = fz.build_partner_cases(records, _collator(), fz.HEAVY_GIVEN_LIGHT)
    assert cases, "records with no record_id must still yield partner cases"
    assert all(c.artifact_id_source == fz.ID_FROM_PAIR_ID for c in cases)


def test_empty_required_probe_is_a_failure_not_a_pass():
    payload = {
        "schema": fz.FROZEN_SCHEMA,
        "probes": {p: [] for p in fz.PROBES},
        "exclusions": {"heavy_given_light:no_admissible_donor": 128},
    }
    with pytest.raises(fz.EmptyProbeError, match="no cases"):
        fz.verify_frozen(payload)


# --------------------------------------------------------------------------- #
# 2. Full-span means hidden
# --------------------------------------------------------------------------- #
def test_full_span_probe_refuses_bert_replacement():
    records = [_paired("r0", HEAVY_A, LIGHT_A)]
    with pytest.raises(fz.FrozenInputsError, match="always_mask"):
        fz.build_single_chain_cases(
            records, _collator(hcdr3_mask_mode="full_span"), fz.PROBE_HCDR3_FULL_SPAN
        )


def test_full_span_probe_refuses_an_ordinary_sampled_collator():
    records = [_paired("r0", HEAVY_A, LIGHT_A)]
    with pytest.raises(fz.FrozenInputsError, match="full_span"):
        fz.build_single_chain_cases(records, _collator(), fz.PROBE_HCDR3_FULL_SPAN)


def test_full_span_targets_are_actually_hidden():
    records = [_paired("r0", HEAVY_A, LIGHT_A), _paired("r1", HEAVY_B, LIGHT_B)]
    cases = fz.build_single_chain_cases(
        records, _full_span_collator(), fz.PROBE_HCDR3_FULL_SPAN
    )
    assert cases
    mask_id = _tokenizer().mask_id
    for case in cases:
        assert all(int(case.input_ids[p]) == mask_id for p in case.target_positions)


# --------------------------------------------------------------------------- #
# 3. Region metadata
# --------------------------------------------------------------------------- #
def test_region_metadata_is_frozen_with_the_case():
    records = [_paired("r0", HEAVY_A, LIGHT_A), _paired("r1", HEAVY_B, LIGHT_B)]
    cases = fz.build_partner_cases(records, _collator(), fz.HEAVY_GIVEN_LIGHT)
    assert cases
    for case in cases:
        assert set(case.regions) >= {"heavy", "light", "hcdr3_heavy", "hcdr3_light"}
        # Provenance of each span is explicit: one is the production helper, the
        # other mirrors its rule at the light offset.
        assert case.regions["hcdr3_heavy"]["derivation"].startswith("production:")
        assert case.regions["hcdr3_light"]["derivation"].startswith("mirrored:")
        for info in case.regions.values():
            assert "targets_in_region" in info


# --------------------------------------------------------------------------- #
# 4. Verification actually enforces the stated invariant
# --------------------------------------------------------------------------- #
def test_verify_catches_a_change_outside_the_targets_but_inside_the_chain():
    payload = _tiny_payload(_collator())
    for entry in payload["probes"][fz.PROBE_PARTNER]:
        if entry["condition"] != fz.MATCHED_ALTERNATIVE:
            continue
        lo, hi = entry["predicted_span"]
        targets = set(entry["target_positions"])
        spare = next(p for p in range(lo, hi) if p not in targets)
        entry["input_ids"][spare] = int(entry["input_ids"][spare]) % 20 + 6
        break
    with pytest.raises(fz.PositionDriftError, match="outside the target positions"):
        fz.verify_frozen(payload)


def test_verify_catches_a_changed_attention_mask():
    payload = _tiny_payload(_collator())
    for entry in payload["probes"][fz.PROBE_PARTNER]:
        if entry["condition"] == fz.ABSENT:
            entry["attention_mask"][entry["predicted_span"][0]] = 0
            break
    with pytest.raises(fz.PositionDriftError, match="attention mask"):
        fz.verify_frozen(payload)


def test_verify_catches_a_missing_condition():
    payload = _tiny_payload(_collator())
    entries = payload["probes"][fz.PROBE_PARTNER]
    entries.remove(next(e for e in entries if e["condition"] == fz.ABSENT))
    with pytest.raises(fz.FrozenInputsError, match="exactly one of each"):
        fz.verify_frozen(payload)


def test_verify_catches_a_duplicate_case_id():
    payload = _tiny_payload(_collator())
    entries = payload["probes"][fz.PROBE_STAGE1_RETENTION]
    entries.append(dict(entries[0]))
    with pytest.raises(fz.FrozenInputsError, match="duplicate case_id"):
        fz.verify_frozen(payload)


def test_semantic_digest_binds_the_analysis_fields():
    """Relabelling a donor changes no tensor, but changes what the number means."""
    payload = _tiny_payload(_collator())
    tensor_before = fz.tensor_digest(payload)
    semantic_before = fz.semantic_digest(payload)

    for entry in payload["probes"][fz.PROBE_PARTNER]:
        if entry["condition"] == fz.MATCHED_ALTERNATIVE:
            entry["donor_artifact_id"] = "some-other-donor"
            break

    assert fz.tensor_digest(payload) == tensor_before
    assert fz.semantic_digest(payload) != semantic_before


def test_semantic_digest_binds_direction():
    payload = _tiny_payload(_collator())
    before = fz.semantic_digest(payload)
    payload["probes"][fz.PROBE_PARTNER][0]["direction"] = "relabelled"
    assert fz.semantic_digest(payload) != before


# --------------------------------------------------------------------------- #
# 5. Donor policy
# --------------------------------------------------------------------------- #
def test_donor_policy_rejects_a_wildly_mismatched_partner_length():
    record = _paired("r0", HEAVY_A, LIGHT_A)
    stubby = _paired("r1", HEAVY_B, "DIQ")
    assert not fz.eligible_donors(
        record, _pool([stubby]), fz.HEAVY_GIVEN_LIGHT,
        require_exact_length=False, policy=fz.DonorPolicy(),
    )
    # Relaxing the DECLARED bound admits it, which is what declaring it buys.
    assert fz.eligible_donors(
        record, _pool([stubby]), fz.HEAVY_GIVEN_LIGHT,
        require_exact_length=False,
        policy=fz.DonorPolicy(max_length_ratio_delta=10.0),
    )


def test_donor_choice_is_seeded_not_first_eligible():
    record = _paired("r0", HEAVY_A, LIGHT_A)
    pool = _pool(
        [_paired(f"d{i}", HEAVY_B, LIGHT_A[:-1] + c) for i, c in enumerate("ACDEFGHIKL")]
    )
    picks = {
        fz.choose_donor(
            record, pool, fz.HEAVY_GIVEN_LIGHT,
            require_exact_length=False, policy=fz.DonorPolicy(),
            rng=_random.Random(seed),
        )[0]
        for seed in range(12)
    }
    assert len(picks) > 1, "selection must depend on the seed, not on pool order"


def test_build_reports_donor_usage_and_exclusions():
    payload = _tiny_payload(_collator())
    assert payload["donor_policy"]["name"] == "seeded_length_matched"
    usage = payload["donor_usage"]
    assert usage["distinct_donors"] >= 1
    assert usage["groups_by_direction"]
    assert isinstance(payload["exclusions"], dict)
