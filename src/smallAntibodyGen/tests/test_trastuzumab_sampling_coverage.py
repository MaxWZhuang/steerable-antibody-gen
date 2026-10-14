"""Analytic exposure coverage must match the family/source/pair sampler."""
from dataclasses import asdict, replace

import pytest

from smallAntibodyGen.experiments.trastuzumab_poc.artifacts import canonical_bytes
from smallAntibodyGen.experiments.trastuzumab_poc.data import (
    GroupingDeclaration, InputRecord, ManifestDeclaration, QualificationRule,
    SourceDeclaration, qualify_manifest,
)
from smallAntibodyGen.experiments.trastuzumab_poc.inputs import coverage_report
from smallAntibodyGen.experiments.trastuzumab_poc.sampling import FamilyFirstSampler, SamplerConfig


def _row(index, family, target, *, source="s1", partition="train", eligible=1):
    # Distinct canonical sequences without requiring real biological data.
    heavy = "A" * (index + 4)
    antigen = {"x": "D", "y": "E", "z": "F", "val": "G", "test": "I"}[target]
    return InputRecord(str(index), source, str(index), heavy, "CC", antigen, "paired",
                       family, target, partition, tuple(range(eligible)), None, None,
                       "synthetic", "synthetic", "positive", "positive")


def _population(rows, *, priority=("s1", "s2", "unused")):
    declaration = ManifestDeclaration(
        tuple(SourceDeclaration(source, "synthetic", i) for i, source in enumerate(priority)),
        (QualificationRule("positive", "synthetic"),), priority,
        GroupingDeclaration(("antibody_family", "target_family")),
    )
    return qualify_manifest(declaration, rows)


def test_equal_antibody_family_mass_can_concentrate_one_target():
    rows = [_row(i, "x-large", "x") for i in range(4)]
    rows += [_row(4, "x-small-1", "x"), _row(5, "x-small-2", "x")]
    rows += [_row(i, "y-large", "y") for i in range(6, 26)]
    report = coverage_report(_population(rows))
    sampling = report["sampling"]
    assert sampling["probability_mass_by_target_family"] == {"x": 0.75, "y": 0.25}
    assert sampling["effective_target_families"] == pytest.approx(1.6)
    assert sampling["maximum_target_family_probability"] == 0.75
    assert sampling["probability_mass_by_antibody_family"] == {
        "x-large": 0.25, "x-small-1": 0.25, "x-small-2": 0.25, "y-large": 0.25,
    }
    assert sampling["probability_mass_by_target_family"]["x"] != pytest.approx(6 / 26)


def test_sources_are_uniform_within_family_not_globally_or_by_row():
    rows = [_row(0, "a", "x", source="s1")]
    rows += [_row(i, "a", "y", source="s2") for i in range(1, 4)]
    rows += [_row(4, "b", "z", source="s2")]
    sampling = coverage_report(_population(rows))["sampling"]
    assert sampling["probability_mass_by_target_family"] == {"x": 0.25, "y": 0.25, "z": 0.5}
    assert sampling["probability_mass_by_owning_source"] == {"s1": 0.25, "s2": 0.75}
    assert sampling["effective_target_families"] == pytest.approx(8 / 3)


def test_duplicate_provenance_uses_only_deduplicated_owner():
    row = _row(0, "a", "x", source="s1")
    duplicate = replace(row, record_id="duplicate", source_id="s2", source_record_id="duplicate")
    population = _population([row, duplicate], priority=("s2", "s1", "unused"))
    sampling = coverage_report(population)["sampling"]
    assert sampling["train_records"] == 1
    assert sampling["probability_mass_by_owning_source"] == {"s2": 1.0}
    assert len(population.records[0].supporting_refs) == 2


def test_validation_and_test_do_not_inflate_training_diversity():
    rows = [_row(0, "a", "x"), _row(1, "v", "val", partition="val"),
            _row(2, "t", "test", partition="test")]
    population = _population(rows)
    report = coverage_report(population)
    assert report["coverage"]["unique_ag"] == 3  # Preserve the existing full-population field.
    assert report["sampling"]["train_unique_antigens"] == 1
    assert report["sampling"]["probability_mass_by_target_family"] == {"x": 1.0}
    assert report["sampling"]["effective_target_families"] == 1.0
    assert report["sampling"]["scope"] == "qualified_train_partition"


def test_eligible_budget_and_uniform_mask_reference_use_sampling_mass():
    one = _row(0, "a", "x", eligible=1)
    three = replace(_row(1, "b", "y", eligible=2), l_eligible=(0,))
    report = coverage_report(_population([one, three]))["sampling"]
    assert report["probability_mass_by_eligible_residue_count"] == {"1": 0.5, "3": 0.5}
    assert report["expected_eligible_residues"] == 2.0
    assert report["uniform_masking_reference"]["mask_count_probability_mass"] == pytest.approx({
        "1": 2 / 3, "2": 1 / 6, "3": 1 / 6,
    })
    assert report["uniform_masking_reference"]["expected_masked_residues"] == 1.5
    assert "span_weight=full_region_weight=0" in report["uniform_masking_reference"]["assumption"]


def test_empty_training_population_remains_a_valid_coverage_report():
    report = coverage_report(_population([_row(0, "v", "val", partition="val")]))["sampling"]
    assert report["status"] == "no_train_records"
    assert report["train_records"] == 0
    assert report["probability_mass_by_target_family"] == {}
    assert report["effective_target_families"] is None
    assert report["expected_eligible_residues"] is None
    assert report["uniform_masking_reference"]["mask_count_probability_mass"] == {}
    assert report["uniform_masking_reference"]["expected_masked_residues"] is None


def test_report_is_row_order_invariant_preserves_old_fields_and_is_json_safe():
    rows = [_row(0, "a", "x"), _row(1, "a", "y", source="s2"), _row(2, "b", "z")]
    population = _population(rows)
    report = coverage_report(population)
    assert canonical_bytes(report) == canonical_bytes(coverage_report(_population(list(reversed(rows)))))
    assert report["schema"] == "trastuzumab-coverage/1"
    assert report["manifest_digest"] == population.manifest_digest
    assert report["population_digest"] == population.population_digest
    assert report["coverage"] == asdict(population.coverage)
    assert report["excluded"] == [asdict(row) for row in population.excluded]
    for key, value in report["sampling"].items():
        if key.startswith("probability_mass_by_"):
            assert sum(value.values()) == pytest.approx(1.0)


def test_analytic_mass_agrees_with_the_sampler_choice_tree_without_monte_carlo():
    rows = [_row(0, "a", "x"), _row(1, "a", "y", source="s2"),
            _row(2, "a", "y", source="s2"), _row(3, "b", "z")]
    population = _population(rows)
    sampler = FamilyFirstSampler(population, SamplerConfig(1, "family_first_v1", 0))
    record_probabilities = {}

    class EnumeratedChoice:
        def __init__(self, choices):
            self.choices = iter(choices)
            self.probability = 1.0

        def choice(self, values):
            self.probability /= len(values)
            selected = next(self.choices)
            assert selected in values
            return selected

    for row in population.train_records():
        rng = EnumeratedChoice((row.antibody_family, row.owning_source_id, row.record_id))
        assert sampler._draw_one(rng) == row.record_id
        record_probabilities[row.record_id] = rng.probability
    by_target = {}
    for row in population.train_records():
        by_target[row.target_family] = by_target.get(row.target_family, 0.0) + record_probabilities[row.record_id]
    assert coverage_report(population)["sampling"]["probability_mass_by_target_family"] == pytest.approx(by_target)
