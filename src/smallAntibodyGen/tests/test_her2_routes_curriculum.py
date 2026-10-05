"""The training-view layer: summaries, bounded contrasts and typed measurements.

Offline and fixture-free. These are the checks that the curriculum hands the
objective something it can actually learn a measured quantity from, rather than a
masked window and a coefficient:

* replicate rows collapse into one construct summary that still carries them;
* contrast enumeration is bounded and completes on a mutational scan, and the
  quartet it emits carries the numeric interaction;
* the measurement payload keeps the four measurement kinds apart;
* a head view is the FULL variant sequence at ``bin=null``.

No HER2 reserved outcome is read anywhere here: every record is synthetic.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from smallAntibodyGen.experiments import her2_routes_data as data_lib
from smallAntibodyGen.experiments import her2_routes_serial as serial
from smallAntibodyGen.experiments.her2_routes_conditions import (
    BIN_HIGH,
    BIN_NULL,
    Condition,
    ConditionSpec,
)
from smallAntibodyGen.experiments.her2_routes_sources import (
    CENSOR_LEFT,
    CENSOR_NONE,
    CENSOR_RIGHT,
    FAMILY_AFFINITY,
    KIND_INTERVAL,
    KIND_ORDINAL,
    KIND_POINT,
    Observation,
)

PARENT_HEAVY = "ACDEFGHIKLMNPQRSTVWY"
PARENT_LIGHT = "YWVTSRQPNMLKIHGFEDCA"
#: The derived assay identity of the synthetic source: ``study:family:target``.
ASSAY_KEY = "synthetic:affinity:target"
ASSAY_INDEX = {ASSAY_KEY: 0}


@pytest.fixture
def bound_vocabulary():
    """Bind a p-IgGen-shaped residue table and restore the previous one."""
    previous = dict(serial.PIGGEN_ID_OF_RESIDUE)
    vocabulary = {residue: index + 3 for index, residue in enumerate(
        "ACDEFGHIKLMNPQRSTVWY")}
    vocabulary.update({"1": serial.PIGGEN_START_ID, "2": serial.PIGGEN_END_ID,
                       "B": serial.PIGGEN_VOCAB_SIZE - 1})
    serial.bind_piggen_vocabulary(vocabulary)
    try:
        yield vocabulary
    finally:
        serial.PIGGEN_ID_OF_RESIDUE.clear()
        serial.PIGGEN_ID_OF_RESIDUE.update(previous)


def observation(heavy, light=PARENT_LIGHT, *, value, replicate=None, kind=KIND_POINT,
                censoring=CENSOR_NONE, interval=None, assay="synthetic", target="target",
                source="synthetic", parent="synthetic::parent", uncertainty=None,
                family=FAMILY_AFFINITY):
    sites = serial.edit_map([("H", index) for index, (a, b) in
                             enumerate(zip(heavy, PARENT_HEAVY)) if a != b] or [("H", 0)])
    construct = serial.Construct(heavy=heavy, light=light, sites=sites, source=source,
                                 record_id=heavy[:4])
    return Observation(construct=construct, source=source, family=family, kind=kind,
                       target=target, assay=assay, parent=parent, scale="pKD",
                       value=value,
                       interval=interval if interval is not None else (value, value),
                       censoring=censoring, replicate=replicate, uncertainty=uncertainty)


def mutate(position, residue, *, heavy=PARENT_HEAVY):
    buffer = list(heavy)
    buffer[position] = residue
    return "".join(buffer)


# --- summaries ------------------------------------------------------------

def test_replicates_collapse_into_one_summary_that_still_carries_them():
    rows = [observation(PARENT_HEAVY, value=8.0, replicate=1),
            observation(PARENT_HEAVY, value=8.4, replicate=2),
            observation(mutate(3, "W"), value=7.0, replicate=1)]
    summaries = data_lib.summarise_observations(rows)
    assert len(summaries) == 2
    parent = next(row for row in summaries if row.construct.heavy == PARENT_HEAVY)
    assert parent.support == 2
    assert parent.values == (8.0, 8.4)
    assert parent.value == pytest.approx(8.2)
    assert parent.replicates == (1, 2)
    # Each replicate survives with its OWN censoring, which is what the head
    # likelihood consumes. Collapsing them here is what made a mixture of one floor
    # replicate and two points train as a point at the mean of the two.
    assert len(parent.measurements) == 2
    assert {row.censoring for row in parent.measurements} == {CENSOR_NONE}
    assert parent.replicate_spread == pytest.approx(0.2828427, rel=1e-4)
    assert parent.uncertainty is None, "no SE was reported; the spread is a different quantity"
    # The key a bin attaches to has no replicate in it, so two replicates of one
    # construct cannot land in two different bins.
    assert len(parent.population_key) == 4
    assert parent.population_key == rows[0].population_key


def test_a_mixed_censored_construct_is_not_reported_as_an_exact_point():
    """One floor replicate and two points is not a point at the mean of the two."""
    rows = [observation(PARENT_HEAVY, value=None, replicate="a", kind=KIND_INTERVAL,
                        censoring=CENSOR_LEFT, interval=(None, 6.0)),
            observation(PARENT_HEAVY, value=8.0, replicate="b", kind=KIND_INTERVAL),
            observation(PARENT_HEAVY, value=8.4, replicate="c", kind=KIND_INTERVAL)]
    summary = data_lib.summarise_observations(rows)[0]
    assert summary.censoring == data_lib.CENSOR_MIXED
    assert summary.value == pytest.approx(8.2), "the point estimate averages the uncensored only"
    assert summary.support == 2 and summary.censored_support == 1
    # The pooled bound is the UNION, so a construct whose replicates straddle a cut
    # is unbinnable rather than placed by whichever replicate was read first.
    assert summary.interval == (None, 8.4)


def test_a_wholly_censored_construct_has_no_point_value_at_all():
    rows = [observation(PARENT_HEAVY, value=None, replicate="a", kind=KIND_INTERVAL,
                        censoring=CENSOR_RIGHT, interval=(9.5, None))]
    summary = data_lib.summarise_observations(rows)[0]
    assert summary.value is None, "a censored replicate has no point value to average"
    assert summary.support == 0
    assert summary.interval == (9.5, None)


# --- bounded contrast enumeration ----------------------------------------

def test_contrasts_are_enumerated_on_every_measured_background():
    """The neighbour graph, not a scan around one modal wild type.

    Five measured constructs: the parent, two singles at different positions, a third
    single at one of them, and the double. The single-substitution neighbour graph has
    five edges where one endpoint carries the parent residue -- including
    ``W3 -> double`` and ``A7 -> double``, which are the SAME substitutions measured on
    a MUTATED background. Those two are the contextual signal, and the earlier
    revision produced neither: it anchored every effect on the modal wild type, so a
    substitution's effect in another background was never a contrast at all.
    """
    background = observation(PARENT_HEAVY, value=8.0)
    single_a = observation(mutate(3, "W"), value=7.5)
    single_b = observation(mutate(7, "A"), value=8.2)
    double = observation(mutate(7, "A", heavy=mutate(3, "W")), value=8.6)
    alternative = observation(mutate(3, "M"), value=7.9)
    summaries = data_lib.summarise_observations(
        [background, single_a, single_b, double, alternative])
    block = data_lib.build_local_contrasts(summaries)
    by_kind = {}
    for contrast in block["contrasts"]:
        by_kind.setdefault(contrast.kind, []).append(contrast)
    assert len(by_kind[data_lib.CONTRAST_SINGLE]) == 5
    assert len(by_kind[data_lib.CONTRAST_ALTERNATIVES]) == 1
    assert len(by_kind[data_lib.CONTRAST_COUPLED]) == 1
    # The effect of I7A on the W3 background: a single-site pair whose background is
    # itself a mutant. This is the pair the modal-wild-type anchoring could not form.
    contextual = [c for c in by_kind[data_lib.CONTRAST_SINGLE]
                  if c.right.construct.heavy == mutate(3, "W")]
    assert contextual, "the effect of a substitution on a MUTATED background is missing"
    assert contextual[0].left.construct.heavy == mutate(7, "A", heavy=mutate(3, "W"))
    assert contextual[0].margin == pytest.approx(8.6 - 7.5)
    # margin is left - right for EVERY kind, and for a single the left is the variant,
    # so the margin is the measured effect of the edit.
    single = next(c for c in by_kind[data_lib.CONTRAST_SINGLE]
                  if c.left.construct.heavy == mutate(3, "W")
                  and c.right.construct.heavy == PARENT_HEAVY)
    assert single.margin == pytest.approx(-0.5)
    assert single.margin == pytest.approx(single.left_value - single.right_value)
    assert single.sites == (("H", 3),)
    assert single.substitution == ("H", 3, "E", "W")
    pair = by_kind[data_lib.CONTRAST_ALTERNATIVES][0]
    assert pair.sites == (("H", 3),)
    # M (7.9) against W (7.5) at one position, canonically ordered by residue, so the
    # same unordered pair always yields the same signed margin: +0.4.
    assert pair.margin == pytest.approx(0.4)
    assert pair.margin == pytest.approx(pair.left_value - pair.right_value)
    quartet = by_kind[data_lib.CONTRAST_COUPLED][0]
    # (8.6 - 8.0) - ((7.5 - 8.0) + (8.2 - 8.0)) = 0.6 - (-0.3) = +0.9. The plan's own
    # worked example, computed from the four measurements rather than quoted.
    assert quartet.interaction == pytest.approx(0.9)
    assert quartet.double_effect == pytest.approx(0.6)
    # The quartet's margin obeys the same rule as every other kind; the double's own
    # effect is a separately named field rather than a third meaning for one name.
    assert quartet.margin == pytest.approx(quartet.left_value - quartet.right_value)
    assert len(quartet.alternatives()) == 4
    assert quartet.alternatives()[0] is quartet.background
    assert quartet.pair_rows() == (1, 2)
    assert quartet.coupled


def test_a_mutational_scan_does_not_blow_up_quadratically():
    """1,000 single mutants of one parent are pairwise at distance two.

    A complete all-pairs enumeration at ``max_sites=2`` therefore emits about half
    a million contrasts for this one group -- quadratic in its OUTPUT, not only in
    its runtime. The masked-context neighbour indexing emits a bounded number,
    reservoir-samples down to the cap, and finishes.
    """
    rows = [observation(PARENT_HEAVY, value=8.0)]
    for index in range(1000):
        position = index % len(PARENT_HEAVY)
        residue = "ACDEFGHIKLMNPQRSTVWY"[(index // len(PARENT_HEAVY)) % 20]
        heavy = mutate(position, residue)
        if heavy == PARENT_HEAVY:
            continue
        rows.append(observation(heavy, value=8.0 + index * 0.001))
    summaries = data_lib.summarise_observations(rows)
    started = time.perf_counter()
    block = data_lib.build_local_contrasts(summaries, max_per_background=256)
    elapsed = time.perf_counter() - started
    assert elapsed < 30.0, f"the bounded enumeration took {elapsed:.1f}s"
    assert len(block["contrasts"]) <= 256 + block["quartets_available"]
    assert block["pairs_available"] > 256
    assert block["pairs_dropped"] == block["pairs_available"] - 256
    assert block["max_per_background"] == 256
    assert block["masked_contexts"] > 0


def test_contrasts_never_cross_an_assay_or_a_target():
    left = observation(mutate(3, "W"), value=7.0, assay="a")
    right = observation(mutate(3, "M"), value=9.0, assay="b")
    background_a = observation(PARENT_HEAVY, value=8.0, assay="a")
    background_b = observation(PARENT_HEAVY, value=8.0, assay="b")
    summaries = data_lib.summarise_observations([left, right, background_a, background_b])
    block = data_lib.build_local_contrasts(summaries)
    for contrast in block["contrasts"]:
        assert contrast.assay in {row.assay for row in summaries}
        alternatives = {row.assay for row in contrast.alternatives()}
        assert len(alternatives) == 1, "a contrast pooled two assays"
    # No same-position pair exists, because the two alternatives are in two assays.
    assert not [c for c in block["contrasts"] if c.kind == data_lib.CONTRAST_ALTERNATIVES]


def test_a_family_with_no_measured_wild_type_still_yields_its_alternatives():
    """Four single mutants at two positions, no measured parent.

    The earlier revision returned NOTHING here: it required a measured wild type in
    the group and skipped the family. But two residues measured at one position on one
    shared background is a preference about that coordinate whether or not the
    background itself was assayed, and it is the contrast a position-only rule cannot
    answer. The reference residue is still used -- to LABEL the pair as competing
    alternatives -- and it never gates enumeration.
    """
    rows = [observation(mutate(3, "W"), value=7.0), observation(mutate(3, "M"), value=7.4),
            observation(mutate(7, "A"), value=8.0), observation(mutate(7, "V"), value=8.3)]
    summaries = data_lib.summarise_observations(rows)
    block = data_lib.build_local_contrasts(summaries)
    kinds = {contrast.kind for contrast in block["contrasts"]}
    assert kinds == {data_lib.CONTRAST_ALTERNATIVES}
    assert len(block["contrasts"]) == 2, "one competing pair at each of the two positions"
    for contrast in block["contrasts"]:
        assert contrast.margin == pytest.approx(
            contrast.left_value - contrast.right_value)
    assert block["skipped"]["no_value"] == 0


def test_a_declared_parent_sequence_decides_which_residue_an_edit_is_measured_against():
    """With the parent supplied, the per-column mode is not consulted at all.

    The mode is a reasonable default and it is a *vote*: in a saturated scan every
    residue at a fully mutated coordinate appears about equally often, so which one
    wins depends on a tie-break rather than on biology, and a pair can be labelled a
    competing alternative when it is a measured single-site effect. Supplying the
    parent removes the vote from the labelling entirely, which is what this asserts.
    """
    rows = [observation(mutate(3, residue), value=7.0 + index * 0.1)
            for index, residue in enumerate("WMFYK")]
    rows.append(observation(PARENT_HEAVY, value=8.0))
    summaries = data_lib.summarise_observations(rows)
    with_parent = data_lib.build_local_contrasts(
        summaries, parents={"synthetic::parent": (PARENT_HEAVY, PARENT_LIGHT)},
        max_per_background=4096)
    singles = [c for c in with_parent["contrasts"] if c.kind == data_lib.CONTRAST_SINGLE]
    alternatives = [c for c in with_parent["contrasts"]
                    if c.kind == data_lib.CONTRAST_ALTERNATIVES]
    assert len(singles) == 5, "every mutant is a single-site effect against the parent"
    assert len(alternatives) == 10, "and the mutants also compete pairwise at that coordinate"
    for contrast in singles:
        assert contrast.right.construct.heavy == PARENT_HEAVY
        assert contrast.substitution[2] == PARENT_HEAVY[3]


def test_held_out_rows_never_enter_the_contrast_enumeration():
    rows = data_lib.summarise_observations(
        [observation(PARENT_HEAVY, value=8.0), observation(mutate(3, "W"), value=7.0)])
    held = data_lib.ConstructSummary(
        construct=rows[1].construct, study=rows[1].study, assay=rows[1].assay,
        target=rows[1].target, parent=rows[1].parent, family=rows[1].family,
        kind=rows[1].kind, measurements=rows[1].measurements, region=rows[1].region,
        partition=data_lib.PARTITION_DEVELOPMENT, scale=rows[1].scale, source=rows[1].source)
    block = data_lib.build_local_contrasts([rows[0], held])
    assert block["contrasts"] == ()
    assert block["skipped"]["not_training_partition"] == 1


# --- typed measurements ---------------------------------------------------

def test_the_payload_keeps_the_four_measurement_kinds_apart():
    rows = [
        observation(PARENT_HEAVY, value=8.0, kind=KIND_POINT),
        observation(mutate(1, "W"), value=None, kind=KIND_INTERVAL, censoring=CENSOR_RIGHT,
                    interval=(9.5, None)),
        observation(mutate(2, "W"), value=None, kind=KIND_INTERVAL, censoring=CENSOR_LEFT,
                    interval=(None, 5.0)),
        observation(mutate(3, "W"), value=2.0, kind=KIND_ORDINAL),
    ]
    summaries = data_lib.summarise_observations(rows)
    assert {row.assay for row in summaries} == {ASSAY_KEY}
    payload = data_lib.measurement_payload(summaries, assay_index=ASSAY_INDEX)
    assert payload.rows() == 4
    document = payload.document()
    assert document["by_kind"][KIND_INTERVAL] == 2
    assert document["by_kind"][KIND_POINT] == 1
    # A right-censored row has no upper bound and a left-censored row has no lower one.
    right = [index for index, row in enumerate(summaries) if row.censoring == CENSOR_RIGHT][0]
    left = [index for index, row in enumerate(summaries) if row.censoring == CENSOR_LEFT][0]
    assert payload.finite_lower[right] and not payload.finite_upper[right]
    assert payload.lower[right] == pytest.approx(9.5)
    assert not payload.finite_lower[left] and payload.finite_upper[left]
    assert payload.upper[left] == pytest.approx(5.0)
    # A point row is a degenerate interval, so one code path serves both.
    point = [index for index, row in enumerate(summaries) if row.kind == KIND_POINT][0]
    assert payload.lower[point] == payload.upper[point] == pytest.approx(8.0)
    assert int(payload.kind_mask(KIND_INTERVAL).sum()) == 2


def test_an_unknown_assay_is_refused_rather_than_given_another_sources_scale():
    summaries = data_lib.summarise_observations([observation(PARENT_HEAVY, value=8.0)])
    with pytest.raises(ValueError, match="not in the frozen assay index"):
        data_lib.measurement_payload(summaries, assay_index={"something_else": 0})


# --- head views ----------------------------------------------------------

def spec():
    return ConditionSpec(assays=(ASSAY_KEY,))


def test_a_head_view_is_the_full_variant_sequence_at_null_quality(bound_vocabulary):
    """The head predicts the outcome of the molecule that was measured.

    Masking the edit window would ask it to predict the outcome of a molecule that
    does not exist, which is what reusing the whole-window serialisation did.
    """
    summaries = data_lib.summarise_observations(
        [observation(mutate(3, "W"), value=8.0), observation(mutate(7, "A"), value=7.0)])
    condition_spec = spec()
    views = data_lib.head_views(summaries, condition_spec=condition_spec, route="causal")
    assert len(views) == 2
    construct, tokens, condition, row = views[0]
    # 1 + heavy + light + 1, with NO mask token anywhere.
    assert len(tokens) == 1 + len(construct.heavy) + len(construct.light) + 1
    assert serial.PIGGEN_MASK_ID not in tokens
    assert tokens[0] == serial.PIGGEN_START_ID and tokens[-1] == serial.PIGGEN_END_ID
    assert condition.bin == BIN_NULL
    assert row.value is not None
    batch = data_lib.collate_heads(views, pad_id=serial.PIGGEN_MASK_ID,
                                   condition_spec=condition_spec, assay_index=ASSAY_INDEX)
    assert batch.tokens.shape == (2, len(tokens))
    assert batch.measurement.rows() == 2
    expected = {mutate(3, "W"): 8.0, mutate(7, "A"): 7.0}
    assert np.allclose(batch.measurement.value,
                       [expected[view[0].heavy] for view in views])
    assert "the full variant construct, unmasked" in batch.document()["input"]
    # The masked-editor layout has no mask token either.
    masked = data_lib.head_views(summaries, condition_spec=condition_spec, route="masked")
    assert serial.ABLANG_MASK_ID not in masked[0][1]


def test_a_head_batch_with_a_quality_bin_is_refused(bound_vocabulary):
    summaries = data_lib.summarise_observations([observation(mutate(3, "W"), value=8.0)])
    condition_spec = spec()
    views = list(data_lib.head_views(summaries, condition_spec=condition_spec,
                                     route="causal"))
    construct, tokens, condition, row = views[0]
    leaking = (construct, tokens, Condition(family=condition.family, bin=BIN_HIGH,
                                            assay=condition.assay), row)
    with pytest.raises(ValueError, match="answer-free stream"):
        data_lib.collate_heads([leaking], pad_id=serial.PIGGEN_MASK_ID,
                               condition_spec=condition_spec, assay_index=ASSAY_INDEX)
