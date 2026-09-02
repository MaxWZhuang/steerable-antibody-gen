"""
Tests for the claim and release planes.

The point of a conformance validator is that it FAILS. A validator that accepts
everything it is given is a rubber stamp with a schema, so most of what is here
is fault injection: each Level-1 requirement is removed in turn and the validator
must name the requirement that went missing.
"""
from __future__ import annotations

import json

import pytest

from smallAntibodyGen.entity_resolution import conformance as cf


def _claim() -> cf.Claim:
    return cf.Claim(
        claim_class="closed-book unseen-broader-group",
        target_population="a snapshot",
        unit_of_generalisation="biological target",
        allowed_evaluation_context=("the item under test",),
        prohibited_exposure_relationships=("the same target on both sides",),
        permitted_exposure_relationships=("the same antibody on both sides",),
        exclusions=("rows with no antigen",),
        estimand="performance on unseen targets",
    )


def _operationalisation() -> cf.Operationalisation:
    return cf.Operationalisation(
        relation_classes=("exact", "lexical-near-duplicate", "metadata/container",
                          "pairwise-containment"),
        unassessed_relation_classes=("semantic", "structural", "aggregate-mosaic"),
        detectors={"exact": "sha256"},
        thresholds={"family_identity": 0.9},
        masks=(),
        promotion_rules=({"evidence": "E1", "action": "must-link"},),
        known_blind_spots=("structure",),
        calibration_population="synthetic, seed 20260901",
        model_capability_envelope={"learned_components_in_grouping": []},
    )


def _artifacts(**overrides):
    """Build a complete, valid Level-1 artifact set."""
    claim_manifest = cf.build_claim_manifest(
        name="t", claim=_claim(), operationalisation=_operationalisation(),
        population={"distinct_antigens": 100, "max_split_group_row_share": 0.70},
        provenance={"git_commit": "abc123"},
    )
    split_manifest = cf.build_split_manifest(
        claim_manifest_sha256=claim_manifest["manifest_sha256"],
        claim_name="t",
        assignments={"g1": "train", "g2": "val"},
        group_of={"r1": "g1", "r2": "g1", "r3": "g2"},
        group_evidence={"g1": {"constructs": 2}, "g2": {"constructs": 1}},
        target_val_fraction=0.1,
        row_weights={"r1": 10, "r2": 10, "r3": 30},
    )
    guard_report = cf.build_guard_report(
        claim_manifest_sha256=claim_manifest["manifest_sha256"],
        resolution_stats={
            "component_min_pairwise_identity": 0.93,
            "component_min_pairwise_coverage": 0.88,
            "target_test_ineligible_constructs": 2,
        },
        calibration={"pairs": 100, "positive_pairs": 10, "negative_pairs": 90},
        audit={"pairs": 20, "positive_pairs": 5, "negative_pairs": 15},
        blocking_recall={"qualifying_pairs": 12, "missed": [], "recall": 1.0},
    )
    unsupported = ["opaque_upstream_pretraining"]
    supported, _ = cf.reduce_claim(_claim().claim_class, unsupported)
    line = cf.LeakageLine(
        claim=supported,
        relations=_operationalisation().relation_classes,
        unassessed=_operationalisation().unassessed_relation_classes,
        conditioning="declared training corpus only",
        population="100 antigens",
        conformance="Level 1",
        result_identity="claim x, split y",
        status="valid for declared channels",
    ).as_dict()
    bundle = {
        "claim_manifest": claim_manifest,
        "split_manifest": split_manifest,
        "guard_report": guard_report,
        "leakage_line": line,
        "unsupported_channels": unsupported,
    }
    bundle.update(overrides)
    return bundle


def _validate(bundle):
    return cf.validate_level_1(
        bundle["claim_manifest"], bundle["split_manifest"],
        bundle["guard_report"], bundle["leakage_line"],
        unsupported_channels=bundle["unsupported_channels"],
    )


def test_a_complete_artifact_set_reaches_level_1():
    """The happy path, so that every failure below is a real signal."""
    attestation = _validate(_artifacts())
    assert attestation.failures == ()
    assert attestation.level == 1
    assert attestation.conformant
    assert attestation.supported_claim == "closed-book unseen-broader-group"


def test_claim_class_must_come_from_the_controlled_vocabulary():
    """Free-form leakage statements are too easy to weaken invisibly."""
    with pytest.raises(cf.ConformanceError, match="controlled"):
        cf.Claim(
            claim_class="basically held out", target_population="p",
            unit_of_generalisation="u", allowed_evaluation_context=(),
            prohibited_exposure_relationships=(),
            permitted_exposure_relationships=(), exclusions=(), estimand="e",
        )


def test_every_relation_class_must_be_declared_assessed_or_not():
    """Listing only what was checked reads as if everything was checked."""
    with pytest.raises(cf.ConformanceError, match="unaccounted for"):
        cf.Operationalisation(
            relation_classes=("exact",), unassessed_relation_classes=("semantic",),
            detectors={}, thresholds={}, masks=(), promotion_rules=(),
            known_blind_spots=(), calibration_population="p",
            model_capability_envelope={},
        )


def test_a_relation_cannot_be_both_assessed_and_unassessed():
    with pytest.raises(cf.ConformanceError, match="both assessed and unassessed"):
        cf.Operationalisation(
            relation_classes=cf.RELATION_CLASSES,
            unassessed_relation_classes=("exact",),
            detectors={}, thresholds={}, masks=(), promotion_rules=(),
            known_blind_spots=(), calibration_population="p",
            model_capability_envelope={},
        )


def test_a_monitor_rule_without_an_escalation_is_rejected():
    """`monitor` without owner, metric, threshold, deadline and escalation is
    an undocumented acceptance of leakage wearing a policy's name."""
    with pytest.raises(cf.ConformanceError, match="undocumented acceptance"):
        cf.Operationalisation(
            relation_classes=cf.RELATION_CLASSES, unassessed_relation_classes=(),
            detectors={}, thresholds={}, masks=(),
            promotion_rules=({"evidence": "E4", "action": "monitor"},),
            known_blind_spots=(), calibration_population="p",
            model_capability_envelope={},
        )
    ok = cf.Operationalisation(
        relation_classes=cf.RELATION_CLASSES, unassessed_relation_classes=(),
        detectors={}, thresholds={}, masks=(),
        promotion_rules=({
            "evidence": "E4", "action": "monitor", "owner": "max",
            "metric": "straddling pairs", "threshold": 0, "deadline": "2026-12-01",
            "escalation": "promote to must-link",
        },),
        known_blind_spots=(), calibration_population="p",
        model_capability_envelope={},
    )
    assert ok.promotion_rules[0]["action"] == "monitor"


def test_a_claim_manifest_without_provenance_seals_nothing():
    with pytest.raises(cf.ConformanceError, match="provenance"):
        cf.build_claim_manifest(
            name="t", claim=_claim(), operationalisation=_operationalisation(),
            population={}, provenance={},
        )


def test_tampering_with_a_sealed_manifest_is_detected():
    """The seal is tamper-evident, which is the whole of what it claims to be."""
    bundle = _artifacts()
    bundle["claim_manifest"] = dict(bundle["claim_manifest"])
    bundle["claim_manifest"]["population"] = {"distinct_antigens": 999999}
    attestation = _validate(bundle)
    assert any("seal" in failure for failure in attestation.failures)
    assert attestation.level == 0


def test_a_split_not_bound_to_its_claim_is_rejected():
    """A split that does not name its claim can be quoted under any claim."""
    bundle = _artifacts()
    bundle["split_manifest"] = dict(bundle["split_manifest"])
    bundle["split_manifest"]["claim_manifest_sha256"] = "0" * 64
    attestation = _validate(bundle)
    assert any("not bound" in failure for failure in attestation.failures)


def test_an_audit_over_zero_pairs_is_not_evidence():
    """Zero errors over zero pairs is a clean scorecard that measured nothing."""
    bundle = _artifacts()
    bundle["guard_report"] = dict(bundle["guard_report"])
    bundle["guard_report"]["audit"] = {"pairs": 0}
    attestation = _validate(bundle)
    assert any("scored no pairs" in failure for failure in attestation.failures)


def test_an_audit_with_no_negative_pairs_gets_its_false_merge_count_for_free():
    bundle = _artifacts()
    bundle["guard_report"] = dict(bundle["guard_report"])
    bundle["guard_report"]["audit"] = {
        "pairs": 10, "positive_pairs": 10, "negative_pairs": 0
    }
    attestation = _validate(bundle)
    assert any("is free" in failure for failure in attestation.failures)


def test_missing_component_minima_fail_level_1():
    """A partition nobody can inspect cannot support a held-out claim."""
    bundle = _artifacts()
    bundle["guard_report"] = dict(bundle["guard_report"])
    bundle["guard_report"]["resolution"] = {"target_test_ineligible_constructs": 0}
    attestation = _validate(bundle)
    assert any("component_min_pairwise_identity" in f for f in attestation.failures)


def test_a_leakage_line_stronger_than_the_reducer_allows_is_rejected():
    """The line cannot claim more than the sealed reduction supports."""
    bundle = _artifacts()
    bundle["leakage_line"] = dict(bundle["leakage_line"])
    bundle["leakage_line"]["claim"] = "closed-book unseen-entity"
    attestation = _validate(bundle)
    assert any("sealed reducer supports" in f for f in attestation.failures)


def test_the_reducer_moves_only_downward_and_never_invents_a_rescue():
    """A missing channel produces a predetermined weaker claim, not an argument."""
    supported, unmapped = cf.reduce_claim(
        "closed-book unseen-entity", ["opaque_upstream_pretraining"]
    )
    assert supported == "closed-book unseen-broader-group"
    assert unmapped == ()

    supported, unmapped = cf.reduce_claim(
        "closed-book unseen-entity",
        ["opaque_upstream_pretraining", "antibody_axis_not_held_out"],
    )
    assert supported == "record-random", "the weakest applicable reduction wins"

    # A channel with no sealed mapping makes the result uninterpretable rather
    # than quietly weaker. That is deliberately the expensive outcome.
    supported, unmapped = cf.reduce_claim(
        "closed-book unseen-entity", ["a channel nobody predeclared"]
    )
    assert unmapped == ("a channel nobody predeclared",)
    attestation = cf.ConformanceAttestation(
        level=1, claimed_level=1, supported_claim=supported,
        uninterpretable_channels=unmapped, failures=(), notes=(),
    )
    assert not attestation.conformant


def test_a_dominant_group_fails_the_level_even_when_the_ratio_is_hit():
    """Hitting a 10% val target with one group holding most rows is not a split."""
    claim_manifest = cf.build_claim_manifest(
        name="t", claim=_claim(), operationalisation=_operationalisation(),
        population={"max_split_group_row_share": 0.70},
        provenance={"git_commit": "abc"},
    )
    split_manifest = cf.build_split_manifest(
        claim_manifest_sha256=claim_manifest["manifest_sha256"], claim_name="t",
        assignments={"big": "train", "small": "val"},
        group_of={f"r{i}": ("big" if i < 90 else "small") for i in range(100)},
        group_evidence={}, target_val_fraction=0.1,
        row_weights={f"r{i}": 1 for i in range(100)},
    )
    assert split_manifest["achieved_val_fraction"] == pytest.approx(0.10)
    assert split_manifest["largest_group_share"] == pytest.approx(0.90)
    bundle = _artifacts(claim_manifest=claim_manifest, split_manifest=split_manifest)
    bundle["guard_report"] = cf.build_guard_report(
        claim_manifest_sha256=claim_manifest["manifest_sha256"],
        resolution_stats=bundle["guard_report"]["resolution"],
        calibration={}, audit=bundle["guard_report"]["audit"],
        blocking_recall=bundle["guard_report"]["blocking_recall"],
    )
    attestation = _validate(bundle)
    # 90% of rows in one group is over the 70% the claim predeclared, so this is
    # a FAILURE and not a note. Concentration that a reader has to notice in a
    # footnote is concentration nobody notices.
    assert attestation.level == 0
    assert any("percolated" in f for f in attestation.failures)
    assert any("effective number of independent groups" in n for n in attestation.notes)


def test_the_split_manifest_seals_the_assignment_itself():
    """Sealing parameters is not sealing a split.

    Reconstructing an assignment from parameters requires the code and the corpus
    to stay byte-identical forever. Recording the assignment does not.
    """
    manifest = _artifacts()["split_manifest"]
    assert manifest["assignments"] == {"g1": "train", "g2": "val"}
    assert manifest["records"] == 3
    assert manifest["groups"] == 2


def test_a_record_in_an_unassigned_group_is_refused():
    claim_manifest = _artifacts()["claim_manifest"]
    with pytest.raises(cf.ConformanceError, match="unassigned group"):
        cf.build_split_manifest(
            claim_manifest_sha256=claim_manifest["manifest_sha256"], claim_name="t",
            assignments={"g1": "train"}, group_of={"r1": "g1", "r2": "g_missing"},
            group_evidence={}, target_val_fraction=0.1,
        )


def test_a_missing_blocking_recall_measurement_fails_the_level():
    """The derivation is not the evidence, and the validator has to say so.

    `blocking.py` is explicit that its recall guarantee is derived and that the
    MEASUREMENT, not the derivation, is what an artifact quotes. A validator that
    never reads the measurement is quoting the derivation.
    """
    bundle = _artifacts()
    bundle["guard_report"] = dict(bundle["guard_report"])
    bundle["guard_report"]["blocking_recall"] = None
    attestation = _validate(bundle)
    assert any("blocking-recall" in f for f in attestation.failures)
    assert attestation.level == 0


def test_a_blocking_recall_over_zero_qualifying_pairs_fails():
    """Recall 1.0 over nothing is free, exactly like an audit over zero pairs."""
    bundle = _artifacts()
    bundle["guard_report"] = dict(bundle["guard_report"])
    bundle["guard_report"]["blocking_recall"] = {
        "qualifying_pairs": 0, "missed": [], "recall": 1.0
    }
    attestation = _validate(bundle)
    assert any("no qualifying pairs" in f for f in attestation.failures)


def test_a_lost_qualifying_pair_fails_the_level():
    """A pair the blocker dropped is silent leakage, not a tuning problem."""
    bundle = _artifacts()
    bundle["guard_report"] = dict(bundle["guard_report"])
    bundle["guard_report"]["blocking_recall"] = {
        "qualifying_pairs": 12, "missed": [["a", "b"]], "recall": 11 / 12
    }
    attestation = _validate(bundle)
    assert any("silent leakage" in f for f in attestation.failures)


def test_concentration_without_a_predeclared_bound_fails():
    """A result cannot pass a gate the claim never set.

    Without a predeclared bound there is no number this run could have failed,
    and a concentration figure nobody can fail is a statistic rather than a gate.
    """
    bundle = _artifacts()
    manifest = cf.build_claim_manifest(
        name="t", claim=_claim(), operationalisation=_operationalisation(),
        population={"distinct_antigens": 100},  # no max_split_group_row_share
        provenance={"git_commit": "abc123"},
    )
    split = cf.build_split_manifest(
        claim_manifest_sha256=manifest["manifest_sha256"], claim_name="t",
        assignments={"g1": "train", "g2": "val"},
        group_of={"r1": "g1", "r2": "g2"}, group_evidence={},
        target_val_fraction=0.1, row_weights={"r1": 1, "r2": 1},
    )
    guard = cf.build_guard_report(
        claim_manifest_sha256=manifest["manifest_sha256"],
        resolution_stats=bundle["guard_report"]["resolution"],
        calibration={}, audit=bundle["guard_report"]["audit"],
        blocking_recall=bundle["guard_report"]["blocking_recall"],
    )
    attestation = cf.validate_level_1(
        manifest, split, guard, bundle["leakage_line"],
        unsupported_channels=bundle["unsupported_channels"],
    )
    assert any("predeclares no max_split_group_row_share" in f
               for f in attestation.failures)


def test_concentration_is_gated_in_rows_not_records():
    """The unit is the whole point of the gate.

    One target in this corpus carries 63% of the rows while being one record
    among 9,574. A concentration share measured in records is blind to exactly
    the failure the gate exists for, so this pins that the two units disagree and
    that the gate follows the rows.
    """
    manifest = cf.build_claim_manifest(
        name="t", claim=_claim(), operationalisation=_operationalisation(),
        population={"max_split_group_row_share": 0.50},
        provenance={"git_commit": "abc"},
    )
    # Two groups, evenly split by RECORD, wildly uneven by ROW.
    split = cf.build_split_manifest(
        claim_manifest_sha256=manifest["manifest_sha256"], claim_name="t",
        assignments={"heavy": "train", "light": "val"},
        group_of={"r1": "heavy", "r2": "light"}, group_evidence={},
        target_val_fraction=0.1, row_weights={"r1": 900, "r2": 100},
    )
    assert split["largest_group_share"] == pytest.approx(0.5)
    assert split["largest_group_row_share"] == pytest.approx(0.9)
    bundle = _artifacts()
    guard = cf.build_guard_report(
        claim_manifest_sha256=manifest["manifest_sha256"],
        resolution_stats=bundle["guard_report"]["resolution"],
        calibration={}, audit=bundle["guard_report"]["audit"],
        blocking_recall=bundle["guard_report"]["blocking_recall"],
    )
    attestation = cf.validate_level_1(
        manifest, split, guard, bundle["leakage_line"],
        unsupported_channels=bundle["unsupported_channels"],
    )
    assert any("percolated" in f for f in attestation.failures), (
        "an even record split hiding a 90% row concentration must fail"
    )


def test_the_leakage_line_renders_every_controlled_field():
    """The line is generated from machine-readable fields, not written by hand."""
    line = _artifacts()["leakage_line"]
    rendered = line["rendered"]
    for label in ("Claim:", "Relations:", "Unassessed:", "Conditioning:",
                  "Population:", "Conformance:", "Result identity:", "Status:"):
        assert label in rendered
    assert json.loads(json.dumps(line)) == line, "the line must be JSON-round-trippable"
