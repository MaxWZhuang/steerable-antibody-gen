"""Regression tests for the S1 functional-foundation repairs (2 October 2026).

Each test drives the real function with the input that broke it and asserts behaviour:

* **exact-measurement eligibility** -- quantitative pairs are decided by the replicates (every
  one an exact finite uncensored pKD point), not by ``summary.kind``; a mixed-censored
  construct's uncensored-only mean is never differenced;
* **pair identity** -- a pair carries its own assay/target/family/scale; the background is the
  full masked context, independent of which endpoint was chosen; ties and refusals are counted;
* **objectives** -- ``weighted_conditional_nll`` is a different loss from ``numeric_margin_match``
  (distinct gradients when the alternatives hold the right odds but less than unit mass), the
  retired ``conditional_fit`` is refused before any model loads, and the config contract is
  versioned;
* **deterministic scoring** -- the policy is scored in eval mode with gradients on;
* **hooks** -- mean ablation needs an explicit frozen donor mean and is batch-invariant, export
  and patch carry their layout (square tensors included), and causal helpers restore flags and
  RNG state and never touch CUDA on a CPU run;
* **a known mechanism** -- a toy model whose output reads one feature, so the interventions have
  an answer that is known in advance;
* **the review repairs** (section 10) -- a capture is a copy, reference-mean provenance is frozen
  and cannot override the record, a pair's measured target has its own digest that binds into the
  checkpoint identity, conflicting duplicate pairs refuse the pool, imported pairs must back up
  their margin and digests, draws are background-first, and the design metadata quotes the
  configured winner-likelihood.

Everything is synthetic and CPU-only; nothing is downloaded or trained.
"""
from __future__ import annotations

import dataclasses
import json
import math
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from smallAntibodyGen.experiments import her2_routes_checkpoints as checkpoints
from smallAntibodyGen.experiments import her2_routes_controls as controls
from smallAntibodyGen.experiments import her2_routes_data as data_lib
from smallAntibodyGen.experiments import her2_routes_interp as interp
from smallAntibodyGen.experiments import her2_routes_pairs as pairs_lib
from smallAntibodyGen.experiments import her2_routes_posttrain as posttrain
from smallAntibodyGen.experiments import her2_routes_serial as serial
from smallAntibodyGen.experiments import her2_routes_services as services_mod
from smallAntibodyGen.experiments import her2_routes_spec as spec
from smallAntibodyGen.experiments.her2_routes_conditions import BIN_HIGH, Condition
from smallAntibodyGen.experiments.her2_routes_sources import (
    CENSOR_LEFT,
    CENSOR_NONE,
    CENSOR_RIGHT,
    FAMILY_AFFINITY,
    KIND_INTERVAL,
    KIND_ORDINAL,
    KIND_POINT,
    SCALE_ENRICHMENT,
    SCALE_PKD,
    Observation,
)
from smallAntibodyGen.models import her2_routes_hooks as hooks_lib
from smallAntibodyGen.tests.test_her2_routes_dplm_services import (  # noqa: F401  (fixtures)
    PANEL,
    ROUTE_LORA,
    complete_stage,
    harness,
    make_pairs,
    produce_bakeoff_best,
)

ROOT = Path(__file__).resolve().parents[3]
HEAVY = "ACDEFGHIKLMNPQRSTVWY" * 2       # 40 residues: positions 30..33 lie past a 24-residue prefix
LIGHT = "YWVTSRQPNMLKIHGFEDCA"
PARENTS = {"synthetic::parent": (HEAVY, LIGHT)}
ASSAY = "synthetic:affinity:target"
OTHER_ASSAY = "other:affinity:T2"


def mutate(*edits, heavy=HEAVY):
    buffer = list(heavy)
    for position, residue in edits:
        buffer[position] = residue
    return "".join(buffer)


def observation(heavy, *, value, light=LIGHT, kind=KIND_POINT, censoring=CENSOR_NONE,
                interval="auto", replicate=None, source="synthetic", target="target",
                family=FAMILY_AFFINITY, scale=SCALE_PKD, parent="synthetic::parent"):
    sites = serial.edit_map([("H", index) for index, (a, b) in enumerate(zip(heavy, HEAVY))
                             if a != b] or [("H", 0)])
    construct = serial.Construct(heavy=heavy, light=light, sites=sites, source=source,
                                 record_id=heavy[:4])
    if isinstance(interval, str):
        interval = (value, value)
    return Observation(construct=construct, source=source, family=family, kind=kind,
                       target=target, assay=source, parent=parent, scale=scale, value=value,
                       interval=interval, censoring=censoring, replicate=replicate)


def summary_of(measurements, *, kind=KIND_POINT, scale=SCALE_PKD, assay=ASSAY, target="target",
               family=FAMILY_AFFINITY, partition=data_lib.PARTITION_TRAIN):
    construct = serial.Construct(heavy=HEAVY, light=LIGHT, sites=(("H", 3),),
                                 source="synthetic", record_id="s")
    return data_lib.ConstructSummary(
        construct=construct, study="synthetic", assay=assay, target=target,
        parent="synthetic::parent", family=family, kind=kind, measurements=tuple(measurements),
        region=(("H", 3),), partition=partition, scale=scale, source="synthetic")


def replicate(value, *, lower="auto", upper="auto", censoring=CENSOR_NONE):
    if isinstance(lower, str):
        lower = value
    if isinstance(upper, str):
        upper = value
    return data_lib.ReplicateMeasurement(value=value, lower=lower, upper=upper,
                                         censoring=censoring)


def contrasts_of(rows, *, max_sites=1):
    summaries = data_lib.summarise_observations(rows)
    block = data_lib.build_local_contrasts(summaries, max_sites=max_sites, parents=PARENTS,
                                           max_per_background=4096, max_per_context=64)
    return summaries, block


def pool_of(rows, *, enrolled=(ASSAY,), violates=lambda row: False):
    _summaries, block = contrasts_of(rows)
    return pairs_lib.build_measured_pairs(block["contrasts"], violates=violates,
                                          enrolled_assays=enrolled)


def one_site_rows(*, e=8.0, w=7.0, m=9.0, f=7.0, **kwargs):
    """The parent (E at position 3) and three alternatives at that one coordinate."""
    return [observation(HEAVY, value=e, **kwargs),
            observation(mutate((3, "W")), value=w, **kwargs),
            observation(mutate((3, "M")), value=m, **kwargs),
            observation(mutate((3, "F")), value=f, **kwargs)]


# ---------------------------------------------------------------------------
# 1. exact-measurement eligibility
# ---------------------------------------------------------------------------

def test_an_interval_kind_row_whose_replicates_are_all_exact_points_is_quantitative():
    """Adams publishes KIND_INTERVAL rows whose in-window values are degenerate intervals."""
    rows = one_site_rows(kind=KIND_INTERVAL)
    summaries = data_lib.summarise_observations(rows)
    assert {row.kind for row in summaries} == {KIND_INTERVAL}
    for row in summaries:
        assert row.exact_status == "exact"
        assert pairs_lib.quantitative_status(row) == (True, "exact_pkd")
    assert len(pool_of(rows)["pairs"]) == 5, "a kind == point gate would have dropped every one"


def test_a_point_with_no_declared_bounds_is_still_exact():
    """Absent explicit bounds are not grounds to refuse a finite uncensored point."""
    rows = one_site_rows(interval=(None, None))
    summaries = data_lib.summarise_observations(rows)
    assert all(row.measurements[0].lower is None for row in summaries)
    assert all(row.exact_status == "exact" for row in summaries)
    assert len(pool_of(rows)["pairs"]) == 5


def test_every_non_exact_construct_is_excluded_for_its_own_counted_reason():
    exact = summary_of([replicate(8.0), replicate(8.4)])
    assert exact.exact_value == pytest.approx(8.2) and exact.exact_status == "exact"

    mixed = summary_of([replicate(None, lower=None, upper=6.0, censoring=CENSOR_LEFT),
                        replicate(8.0), replicate(8.4)], kind=KIND_INTERVAL)
    assert mixed.value == pytest.approx(8.2), "the qualified mean still exists"
    assert mixed.exact_value is None and mixed.exact_status == "mixed_censoring"

    censored = summary_of([replicate(None, lower=9.5, upper=None, censoring=CENSOR_RIGHT)])
    assert censored.exact_value is None and censored.exact_status == "censored"

    nonfinite = summary_of([replicate(float("nan"))])
    assert nonfinite.exact_value is None and nonfinite.exact_status == "nonfinite"

    loose = summary_of([replicate(8.0, lower=7.0, upper=9.0)])
    assert loose.exact_value is None and loose.exact_status == "nondegenerate_bounds"

    empty = summary_of([])
    assert empty.exact_value is None and empty.exact_status == "empty"

    ordinal = summary_of([replicate(2.0)], kind=KIND_ORDINAL)
    assert ordinal.exact_value is None and ordinal.exact_status == "ordinal"

    reasons = {name: pairs_lib.quantitative_status(row) for name, row in
               dict(mixed=mixed, censored=censored, nonfinite=nonfinite, loose=loose,
                    empty=empty, ordinal=ordinal).items()}
    assert reasons == {"mixed": (False, "mixed_censoring"), "censored": (False, "censored"),
                       "nonfinite": (False, "nonfinite"),
                       "loose": (False, "nondegenerate_bounds"), "empty": (False, "empty"),
                       "ordinal": (False, "ordinal")}


def test_exact_value_keeps_enrichment_semantics_and_the_pair_gate_enforces_pkd():
    """exact_value is unit-agnostic; only the pair gate insists on pKD."""
    enrichment = summary_of([replicate(3.2)], kind="enrichment", scale=SCALE_ENRICHMENT)
    assert enrichment.exact_value == pytest.approx(3.2)
    assert pairs_lib.quantitative_status(enrichment) == (False, "scale_not_pkd")
    missing = summary_of([replicate(8.0)], assay="")
    assert pairs_lib.quantitative_status(missing) == (False, "missing_metadata")


def test_a_mixed_censored_construct_is_never_a_margin_or_a_quartet_target():
    base = [observation(HEAVY, value=8.0), observation(mutate((3, "W")), value=7.0)]
    mixed_m = [observation(mutate((3, "M")), value=None, kind=KIND_INTERVAL,
                           censoring=CENSOR_LEFT, interval=(None, 6.0), replicate="a"),
               observation(mutate((3, "M")), value=9.0, kind=KIND_INTERVAL, replicate="b"),
               observation(mutate((3, "M")), value=9.2, kind=KIND_INTERVAL, replicate="c")]
    summaries, block = contrasts_of(base + mixed_m)
    mixed = next(row for row in summaries if row.construct.heavy == mutate((3, "M")))
    assert mixed.value == pytest.approx(9.1) and mixed.exact_value is None
    assert block["skipped"]["not_exact_value"] == 1 and block["skipped"]["no_value"] == 0
    assert block["not_exact_by_status"] == {"mixed_censoring": 1}
    assert all(mutate((3, "M")) not in (c.left.construct.heavy, c.right.construct.heavy)
               for c in block["contrasts"]), "a margin was differenced from a mixed row"
    assert len(block["contrasts"]) == 1

    # A quartet whose DOUBLE is mixed-censored is not formed; with an exact double it is.
    def quartet_rows(double_rows):
        return ([observation(HEAVY, value=8.0), observation(mutate((3, "W")), value=7.5),
                 observation(mutate((7, "A")), value=8.2)] + double_rows)
    double_edit = mutate((3, "W"), (7, "A"))
    mixed_double = [observation(double_edit, value=None, kind=KIND_INTERVAL,
                                censoring=CENSOR_LEFT, interval=(None, 6.0), replicate="a"),
                    observation(double_edit, value=8.6, kind=KIND_INTERVAL, replicate="b")]
    _s, blocked = contrasts_of(quartet_rows(mixed_double), max_sites=2)
    assert not [c for c in blocked["contrasts"] if c.kind == data_lib.CONTRAST_COUPLED]
    _s, formed = contrasts_of(quartet_rows([observation(double_edit, value=8.6)]), max_sites=2)
    quartets = [c for c in formed["contrasts"] if c.kind == data_lib.CONTRAST_COUPLED]
    assert len(quartets) == 1 and quartets[0].interaction == pytest.approx(0.9)


def test_the_teacher_training_bank_reads_exact_values_only():
    mixed = [observation(mutate((3, "M")), value=None, kind=KIND_INTERVAL, censoring=CENSOR_LEFT,
                         interval=(None, 6.0), replicate="a", source="alphabind_her2_validation"),
             observation(mutate((3, "M")), value=9.0, kind=KIND_INTERVAL, replicate="b",
                         source="alphabind_her2_validation")]
    rows = [observation(HEAVY, value=8.0, source="alphabind_her2_validation")] + mixed
    summaries = data_lib.summarise_observations(rows)
    closure = types.SimpleNamespace(violates=lambda *, window, identity: None)
    stub = types.SimpleNamespace(closure=lambda: closure, summaries=lambda: summaries)
    windows, values = services_mod.FlightServices.her2_training_bank(stub)
    assert values == [8.0], "the mixed-censored construct's uncensored mean 9.0 was a target"
    assert len(windows) == 1


# ---------------------------------------------------------------------------
# 2. pair identity: both classes, ties, metadata, the full masked context
# ---------------------------------------------------------------------------

def test_a_pool_has_both_contrast_classes_and_counts_its_ties():
    result = pool_of(one_site_rows())
    pairs, counts = result["pairs"], result["counts"]
    # E=8, W=7, M=9, F=7 at one coordinate: three mutant-vs-background edges and three
    # competing pairs, of which W-F is a measured tie.
    assert counts["eligible"] == 5 and counts["ties"] == 1
    assert counts["excluded"] == {"tie": 1}
    assert counts["by_contrast_kind"] == {data_lib.CONTRAST_ALTERNATIVES: 2,
                                          data_lib.CONTRAST_SINGLE: 3}
    for pair in pairs:
        assert pair["chosen_value"] > pair["rejected_value"]
        assert pair["margin"] == pytest.approx(pair["chosen_value"] - pair["rejected_value"])
        assert (pair["assay"], pair["target"], pair["family"], pair["scale"]) == (
            ASSAY, "target", FAMILY_AFFINITY, SCALE_PKD)
        assert pair["version"] == pairs_lib.PAIRS_VERSION
    # The background wins some edges (E beats W and F): chosen is the higher value, not the mutant.
    assert any(pair["contrast_kind"] == data_lib.CONTRAST_SINGLE
               and pair["chosen"].construct.heavy == HEAVY for pair in pairs)
    assert any(pair["contrast_kind"] == data_lib.CONTRAST_SINGLE
               and pair["rejected"].construct.heavy == HEAVY for pair in pairs)
    assert pairs_lib.require_pair_metadata(pairs) == 5
    assert counts["by_assay"] == {ASSAY: {"pairs": 5, "backgrounds": 1, "rows": 10}}


def test_a_mixed_censored_endpoint_never_reaches_the_pool_and_is_counted_upstream():
    rows = one_site_rows()[:2] + [
        observation(mutate((3, "M")), value=None, kind=KIND_INTERVAL, censoring=CENSOR_LEFT,
                    interval=(None, 6.0), replicate="a"),
        observation(mutate((3, "M")), value=9.0, kind=KIND_INTERVAL, replicate="b")]
    _summaries, block = contrasts_of(rows)
    assert block["skipped"]["not_exact_value"] == 1
    result = pool_of(rows)
    assert result["counts"]["eligible"] == 1, "only E against W survives"
    assert all(pair["chosen"].construct.heavy != mutate((3, "M"))
               and pair["rejected"].construct.heavy != mutate((3, "M"))
               for pair in result["pairs"])


def test_an_assay_that_is_not_enrolled_is_refused_not_relabelled_null():
    result = pool_of(one_site_rows(), enrolled=("some:other:assay",))
    assert result["pairs"] == ()
    # all six neighbour pairs are refused for their assay, the measured tie included: the
    # enrolment check runs before the tie check, so nothing reaches the pool relabelled 'null'
    assert result["counts"]["excluded"] == {"assay_not_enrolled": 6}
    assert "null" not in {pair["assay"] for pair in result["pairs"]}
    with pytest.raises(ValueError, match="enrolled assay"):
        pairs_lib.build_measured_pairs([], violates=lambda row: False, enrolled_assays=())


def test_closure_violations_on_either_endpoint_exclude_the_pair():
    rejected_heavy = mutate((3, "W"))
    result = pool_of(one_site_rows(), violates=lambda row: row.construct.heavy == rejected_heavy)
    assert result["counts"]["excluded"].get("closure", 0) >= 1
    for pair in result["pairs"]:
        assert rejected_heavy not in (pair["chosen"].construct.heavy,
                                      pair["rejected"].construct.heavy)


def test_endpoints_that_disagree_on_assay_or_scale_are_not_a_pair():
    _summaries, block = contrasts_of(one_site_rows())
    contrast = next(c for c in block["contrasts"] if c.kind == data_lib.CONTRAST_SINGLE)
    other = dataclasses.replace(contrast.right, assay="elsewhere:affinity:target")
    mismatched = dataclasses.replace(contrast, right=other)
    result = pairs_lib.build_measured_pairs([mismatched], violates=lambda row: False,
                                            enrolled_assays=(ASSAY, "elsewhere:affinity:target"))
    assert result["pairs"] == () and result["counts"]["excluded"] == {"metadata_mismatch": 1}
    other_scale = dataclasses.replace(contrast.right, scale=SCALE_ENRICHMENT)
    result = pairs_lib.build_measured_pairs(
        [dataclasses.replace(contrast, right=other_scale)], violates=lambda row: False,
        enrolled_assays=(ASSAY,))
    assert result["counts"]["excluded"] == {"scale_not_pkd": 1}


def test_the_background_digest_is_orientation_independent_and_complete():
    _summaries, block = contrasts_of(one_site_rows())
    contrast = next(c for c in block["contrasts"] if c.kind == data_lib.CONTRAST_SINGLE)
    swapped = dataclasses.replace(
        contrast, left=contrast.right, right=contrast.left, left_value=contrast.right_value,
        right_value=contrast.left_value, margin=-contrast.margin)
    forward, backward = (pairs_lib.build_measured_pairs(
        [item], violates=lambda row: False, enrolled_assays=(ASSAY,))["pairs"][0]
        for item in (contrast, swapped))
    assert forward["background"] == backward["background"], "orientation changed the context"
    assert forward["pair_digest"] == backward["pair_digest"]
    assert forward["chosen"].construct.heavy == backward["chosen"].construct.heavy
    assert len(forward["background"]) == 64, "the digest is the full SHA-256, not a prefix"

    heavy, light = pairs_lib.masked_context(forward["chosen"].construct, forward["sites"])
    assert pairs_lib.masked_context(forward["rejected"].construct, forward["sites"]) == (heavy, light)
    assert pairs_lib.MASK_CHARACTER in heavy
    base = dict(heavy=heavy, light=light, assay=ASSAY, target="target", family="affinity",
                scale=SCALE_PKD, sites=(("H", 3),))
    reference = pairs_lib.background_digest(**base)
    for change in (dict(assay="x:affinity:target"), dict(target="t2"), dict(family="display"),
                   dict(scale=SCALE_ENRICHMENT), dict(sites=(("H", 4),)),
                   dict(light=light[:-1] + "W"), dict(heavy=heavy[:-1] + "W")):
        assert pairs_lib.background_digest(**{**base, **change}) != reference, change


def test_four_distinct_backgrounds_are_four_strata_not_one_sequence_prefix():
    """The first pool keyed backgrounds on ``identity()[:24]``: the first 24 heavy residues."""
    rows = []
    for k in range(4):
        background = mutate((30 + k, "W"))
        rows.append(observation(background, value=8.0 + 0.1 * k))
        rows.append(observation(mutate((3, "M"), heavy=background), value=7.0))
    result = pool_of(rows)
    pairs = result["pairs"]
    assert len(pairs) == 4
    legacy = {pair["rejected"].construct.identity()[:24] for pair in pairs}
    assert len(legacy) == 1, "the legacy key collapses four contexts into one stratum"
    assert len({pair["background"] for pair in pairs}) == 4
    assert len({pair["pair_digest"] for pair in pairs}) == 4
    bank = posttrain.PairBank(pairs=tuple(pairs), reuse_cap=2)
    drawn, record = bank.draw(4, seed=1)
    assert record["backgrounds_available"] == 4 and record["backgrounds_drawn"] == 4
    assert len(drawn) == 4


def test_a_binary_landscape_edge_set_has_one_digest_per_edge_and_either_side_can_win():
    """Two binary sites -> four genotypes -> four single-substitution edges."""
    rows = [observation(mutate(), value=8.0),                       # E, I
            observation(mutate((3, "W")), value=8.5),               # W, I   (mutant wins)
            observation(mutate((7, "A")), value=7.0),               # E, A   (background wins)
            observation(mutate((3, "W"), (7, "A")), value=6.5)]     # W, A
    result = pool_of(rows)
    pairs = result["pairs"]
    assert len(pairs) == 4
    assert len({pair["background"] for pair in pairs}) == 4, "each edge masks a different context"
    assert {pair["contrast_kind"] for pair in pairs} == {data_lib.CONTRAST_SINGLE}
    wins_by_mutant = [p for p in pairs if p["chosen"].construct.heavy != HEAVY
                      and p["rejected"].construct.heavy == HEAVY]
    wins_by_background = [p for p in pairs if p["chosen"].construct.heavy == HEAVY]
    assert wins_by_mutant and wins_by_background
    assert {pair["sites"] for pair in pairs} == {(("H", 3),), (("H", 7),)}


# ---------------------------------------------------------------------------
# 3. conditions, populations and samples
# ---------------------------------------------------------------------------

def test_a_pair_condition_is_derived_from_the_pair_and_never_defaulted():
    pair = pool_of(one_site_rows())["pairs"][0]
    assert pairs_lib.pair_condition(pair) == Condition(
        family=FAMILY_AFFINITY, bin=BIN_HIGH, assay=ASSAY)
    explicit = Condition(family=FAMILY_AFFINITY, bin="null", assay=ASSAY)
    assert pairs_lib.pair_condition(pair, explicit) is explicit
    with pytest.raises(ValueError, match="incompatible"):
        pairs_lib.pair_condition(pair, Condition(family=FAMILY_AFFINITY, bin=BIN_HIGH,
                                                 assay="alphabind_her2:affinity:HER2"))
    with pytest.raises(ValueError, match="incompatible"):
        pairs_lib.pair_condition(pair, Condition(family="display", bin=BIN_HIGH, assay=ASSAY))
    with pytest.raises(ValueError, match="Unsupported pair condition"):
        pairs_lib.pair_condition(pair, "prior_only")
    stripped = {key: value for key, value in pair.items() if key not in ("assay", "family")}
    with pytest.raises(ValueError, match="never defaulted to HER2"):
        pairs_lib.pair_condition(stripped)
    with pytest.raises(ValueError, match="lacks"):
        pairs_lib.require_pair_metadata([stripped])


def test_a_population_is_explicit_counted_and_has_no_her2_default():
    rows = one_site_rows() + one_site_rows(source="other", target="T2")
    result = pool_of(rows, enrolled=(ASSAY, OTHER_ASSAY))
    pairs = result["pairs"]
    assert {pair["assay"] for pair in pairs} == {ASSAY, OTHER_ASSAY}, "the pool is pooled"
    everything = pairs_lib.select_population(pairs)
    assert everything["selected"] == 10 and everything["outside_population"] == 0
    chosen = pairs_lib.select_population(pairs, {"assay": OTHER_ASSAY, "family": "affinity"})
    assert chosen["selected"] == 5 and chosen["outside_population"] == 5
    assert {pair["assay"] for pair in chosen["pairs"]} == {OTHER_ASSAY}
    with pytest.raises(ValueError, match="Unknown population key"):
        pairs_lib.select_population(pairs, {"asay": ASSAY})
    with pytest.raises(ValueError, match="empty population"):
        pairs_lib.select_population(pairs, {})


def test_the_content_addressed_sample_ignores_enumeration_order():
    pairs = pool_of(one_site_rows())["pairs"]
    first, record = pairs_lib.sample_pairs(pairs, 3)
    second, _ = pairs_lib.sample_pairs(tuple(reversed(pairs)), 3)
    assert [p["pair_digest"] for p in first] == [p["pair_digest"] for p in second]
    assert record["drawn"] == 3 and record["available"] == 5
    assert pairs_lib.pair_pool_digest(pairs) == pairs_lib.pair_pool_digest(reversed(pairs))


def stub_services(rows, *, assays, posttrain_block=None):
    summaries = data_lib.summarise_observations(rows)
    closure = types.SimpleNamespace(violates=lambda *, window, identity: None)
    return types.SimpleNamespace(
        _cache={},
        config={"curriculum": {"max_contrasts_per_background": 4096,
                               "max_contrasts_per_context": 32},
                "posttrain": dict(posttrain_block or {})},
        eligible_summaries=lambda: summaries, parent_sequences=lambda: dict(PARENTS),
        closure=lambda: closure,
        condition_spec=types.SimpleNamespace(assay_order=(*sorted(assays), "null")))


def test_the_service_pool_is_pooled_filterable_and_never_silently_her2():
    rows = one_site_rows() + one_site_rows(source="other", target="T2")
    stub = stub_services(rows, assays=(ASSAY, OTHER_ASSAY))
    pooled = services_mod.FlightServices.measured_pairs(stub)
    assert {pair["assay"] for pair in pooled} == {ASSAY, OTHER_ASSAY} and len(pooled) == 10
    only = services_mod.FlightServices.measured_pairs(stub, population={"assay": ASSAY})
    assert len(only) == 5 and {pair["assay"] for pair in only} == {ASSAY}
    inventory = stub._cache["pair_inventory"]
    assert inventory["outside_population"] == 5 and inventory["version"] == pairs_lib.PAIRS_VERSION
    assert inventory["counts"]["by_assay"][ASSAY]["pairs"] == 5
    with pytest.raises(ValueError, match="No eligible measured preference pair"):
        services_mod.FlightServices.measured_pairs(stub, population={"assay": "nothing:here"})
    assert services_mod.FlightServices.measured_pairs(
        stub, population={"assay": "nothing:here"}, allow_empty=True) == []


def test_the_config_population_is_an_explicit_optional_filter():
    rows = one_site_rows() + one_site_rows(source="other", target="T2")
    stub = stub_services(rows, assays=(ASSAY, OTHER_ASSAY),
                         posttrain_block={"population": {"assay": OTHER_ASSAY}})
    selected = services_mod.FlightServices.measured_pairs(stub)
    assert {pair["assay"] for pair in selected} == {OTHER_ASSAY}
    assert stub._cache["pair_inventory"]["population"] == {"assay": OTHER_ASSAY}
    assert stub._cache["pair_inventory"]["outside_population"] == 5


def test_an_unenrolled_assay_in_the_service_pool_is_counted_not_mapped_to_null():
    rows = one_site_rows() + one_site_rows(source="other", target="T2")
    stub = stub_services(rows, assays=(ASSAY,))
    pairs = services_mod.FlightServices.measured_pairs(stub)
    assert {pair["assay"] for pair in pairs} == {ASSAY}
    excluded = stub._cache["pair_inventory"]["counts"]["excluded"]
    assert excluded == {"assay_not_enrolled": 6, "tie": 1}, (
        "the other assay's six neighbour pairs are refused; the enrolled assay's tie is counted")


# ---------------------------------------------------------------------------
# 4. the objectives
# ---------------------------------------------------------------------------

def pair_logits(probabilities):
    """A parameter whose softmax is ``probabilities`` and two adjacent rows reading it.

    float64, so "exactly the right odds" is exact to ~1e-16 rather than float32 rounding.
    """
    parameter = torch.nn.Parameter(torch.log(torch.as_tensor(probabilities, dtype=torch.float64)))
    log_probs = torch.log_softmax(parameter, dim=-1).reshape(1, 1, -1).expand(2, 1, -1)
    return parameter, log_probs


TARGETS = torch.tensor([[0], [1]])           # chosen = residue 0, rejected = residue 1
MASK = torch.ones(2, 1, dtype=torch.long)


def test_weighted_nll_matches_the_hand_computation_over_the_full_vocabulary():
    logits = torch.tensor([0.3, -0.2, 1.1, 0.0, -0.7])
    log_probs = torch.log_softmax(logits, dim=-1).reshape(1, 1, -1).expand(2, 1, -1)
    kappa, delta = 2.0, 0.5
    loss, record = posttrain.weighted_conditional_nll_loss(
        log_probs, TARGETS, MASK, deltas=[delta], kappa=kappa)
    t = torch.sigmoid(torch.tensor(kappa * delta))
    expected = -(t * torch.log_softmax(logits, -1)[0] + (1 - t) * torch.log_softmax(logits, -1)[1])
    assert float(loss) == pytest.approx(float(expected), rel=1e-6)
    q = torch.softmax(logits, -1)
    assert record["mean_alternative_mass"] == pytest.approx(float(q[0] + q[1]), rel=1e-6)
    assert record["vocabulary"] == 5 and record["method"] == posttrain.METHOD_WEIGHTED_CONDITIONAL_NLL
    assert "NOT an exact tilted-prior fit" in record["note"]


def test_equal_odds_on_less_than_unit_mass_has_distinct_gradients():
    """The two arms are different objectives: the log-odds loss cannot see the missing mass."""
    kappa, delta, mass = 2.0, 0.6, 0.5
    t = 1.0 / (1.0 + math.exp(-kappa * delta))
    # q(chosen)/q(rejected) = exp(kappa * delta) exactly, but the pair holds only half the mass.
    probabilities = [mass * t, mass * (1 - t)] + [(1 - mass) / 4] * 4

    parameter, log_probs = pair_logits(probabilities)
    margin, margin_record = posttrain.margin_match_loss(
        log_probs[0, 0, 0:1], log_probs[1, 0, 1:2], [delta], kappa=kappa)
    margin.backward()
    margin_gradient = parameter.grad.clone()
    assert float(margin) == pytest.approx(0.0, abs=1e-12)
    assert float(margin_gradient.abs().max()) < 1e-9, "right odds -> no log-odds gradient at all"
    assert margin_record["mean_alternative_mass"] == pytest.approx(mass, rel=1e-5)

    parameter, log_probs = pair_logits(probabilities)
    nll, record = posttrain.weighted_conditional_nll_loss(
        log_probs, TARGETS, MASK, deltas=[delta], kappa=kappa)
    nll.backward()
    gradient = parameter.grad
    assert float(gradient.abs().max()) > 0.1
    # The likelihood pulls mass onto BOTH measured alternatives and off the unmeasured ones.
    assert float(gradient[0]) < 0 and float(gradient[1]) < 0 and bool((gradient[2:] > 0).all())
    assert record["mean_alternative_mass"] == pytest.approx(mass, rel=1e-5)
    assert record["mean_abs_log_odds_residual"] == pytest.approx(0.0, abs=1e-5)


def test_the_minimiser_of_the_pair_local_nll_puts_all_mass_on_the_two_alternatives():
    """The record's claim: this is not a tilted-prior fit, which would keep prior mass elsewhere."""
    kappa, delta = 2.0, 0.6
    t = 1.0 / (1.0 + math.exp(-kappa * delta))
    parameter, log_probs = pair_logits([t, 1 - t, 1e-20, 1e-20, 1e-20])
    loss, _record = posttrain.weighted_conditional_nll_loss(
        log_probs, TARGETS, MASK, deltas=[delta], kappa=kappa)
    loss.backward()
    assert float(parameter.grad.abs().max()) < 1e-5, "q = (t, 1 - t, 0, ...) is stationary"


def test_the_weighted_loss_refuses_malformed_input_instead_of_coercing_it():
    _p, good = pair_logits([0.25, 0.25, 0.25, 0.25])
    call = dict(deltas=[0.5], kappa=2.0)
    posttrain.weighted_conditional_nll_loss(good, TARGETS, MASK, **call)
    with pytest.raises(ValueError, match="FULL vocabulary"):
        posttrain.weighted_conditional_nll_loss(good - 1.0, TARGETS, MASK, **call)
    with pytest.raises(ValueError, match="even number of rows"):
        posttrain.weighted_conditional_nll_loss(good[:1], TARGETS[:1], MASK[:1], **call)
    with pytest.raises(ValueError, match="not aligned"):
        posttrain.weighted_conditional_nll_loss(good, TARGETS[:, :0], MASK, **call)
    with pytest.raises(ValueError, match="finite and positive"):
        posttrain.weighted_conditional_nll_loss(good, TARGETS, MASK, deltas=[0.5], kappa=0.0)
    with pytest.raises(ValueError, match="not finite"):
        posttrain.weighted_conditional_nll_loss(good, TARGETS, MASK, deltas=[float("nan")],
                                                kappa=2.0)
    with pytest.raises(ValueError, match="measured differences"):
        posttrain.weighted_conditional_nll_loss(good, TARGETS, MASK, deltas=[0.5, 0.5], kappa=2.0)
    with pytest.raises(ValueError, match="same residue"):
        posttrain.weighted_conditional_nll_loss(good, torch.tensor([[2], [2]]), MASK, **call)
    with pytest.raises(ValueError, match="outside the 4-residue vocabulary"):
        posttrain.weighted_conditional_nll_loss(good, torch.tensor([[0], [9]]), MASK, **call)
    two_sites = torch.log_softmax(torch.zeros(2, 2, 4), dim=-1)
    with pytest.raises(ValueError, match="exactly one masked coordinate"):
        posttrain.weighted_conditional_nll_loss(
            two_sites, torch.tensor([[0, 1], [1, 0]]), torch.ones(2, 2, dtype=torch.long), **call)
    bad = good.clone().detach()
    bad[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="no finite probability mass"):
        posttrain.weighted_conditional_nll_loss(bad, TARGETS, MASK, **call)


def test_weights_and_per_assay_diagnostics_are_recorded():
    log_probs = torch.log_softmax(torch.randn(4, 1, 6, generator=torch.Generator().manual_seed(1)),
                                  dim=-1)
    targets = torch.tensor([[0], [1], [2], [3]])
    mask = torch.ones(4, 1, dtype=torch.long)
    loss, record = posttrain.weighted_conditional_nll_loss(
        log_probs, targets, mask, deltas=[0.5, 1.0], kappa=2.0, assays=["a", "b"],
        weights=[1.0, 3.0])
    assert record["weights"] == "supplied" and set(record["by_assay"]) == {"a", "b"}
    assert record["by_assay"]["a"]["pairs"] == 1
    uniform, _ = posttrain.weighted_conditional_nll_loss(
        log_probs, targets, mask, deltas=[0.5, 1.0], kappa=2.0)
    assert float(loss) != pytest.approx(float(uniform))
    exposure = posttrain.AssayExposure()
    exposure.add(["a", "a", "b"], masses=[0.1, 0.3, 0.5])
    document = exposure.document()
    assert document["a"]["pairs_drawn"] == 2 and document["b"]["pairs_drawn"] == 1
    assert document["a"]["mean_alternative_mass"] == pytest.approx(0.2)
    assert document["b"]["mean_alternative_mass"] == pytest.approx(0.5)


def test_the_deprecated_conditional_fit_is_the_margin_match_loss_under_another_name():
    """Why it was retired: the v1 comparison ran one objective twice."""
    log_probs = torch.log_softmax(torch.randn(4, 1, 6, generator=torch.Generator().manual_seed(2)),
                                  dim=-1)
    targets = torch.tensor([[0], [1], [2], [3]])
    mask = torch.ones(4, 1, dtype=torch.long)
    values = [8.0, 7.0, 9.0, 6.5]
    legacy, _ = posttrain.conditional_fit_loss(log_probs, targets, mask, values=values, kappa=2.0)
    picked = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1).sum(dim=1)
    margin, _ = posttrain.margin_match_loss(picked[0::2], picked[1::2], [1.0, 2.5], kappa=2.0)
    assert float(legacy) == pytest.approx(float(margin), rel=1e-6)
    new, _ = posttrain.weighted_conditional_nll_loss(
        log_probs, targets, mask, deltas=[1.0, 2.5], kappa=2.0)
    assert float(new) != pytest.approx(float(margin), rel=1e-3)


def test_resolve_methods_refuses_the_legacy_name_unknowns_and_duplicates():
    assert posttrain.resolve_methods(["weighted_conditional_nll", "local_ipo"]) == (
        "weighted_conditional_nll", "local_ipo")
    assert set(posttrain.METHODS) == {"weighted_conditional_nll", "numeric_margin_match",
                                      "local_ipo", "local_dpo"}
    with pytest.raises(ValueError, match="weighted_conditional_nll") as legacy:
        posttrain.resolve_methods(["conditional_fit", "local_ipo"])
    assert "numeric_margin_match" in str(legacy.value) and "Historical" in str(legacy.value)
    with pytest.raises(ValueError, match="unknown method"):
        posttrain.resolve_methods(["local_ipo", "made_up"])
    with pytest.raises(ValueError, match="more than once"):
        posttrain.resolve_methods(["local_ipo", "local_ipo"])
    with pytest.raises(ValueError, match="non-empty"):
        posttrain.resolve_methods([])
    with pytest.raises(ValueError, match="legacy|conditional_fit"):
        posttrain.design(methods=["conditional_fit"])
    document = posttrain.design(methods=["weighted_conditional_nll", "numeric_margin_match"])
    assert document["contract_version"] == 2
    assert "one-site" in document["objectives"]["numeric_margin_match"]
    assert "tilted" in document["objectives"]["weighted_conditional_nll"]


def test_a_stale_posttrain_config_fails_before_any_model_is_loaded():
    touched = []

    def boom(*args, **kwargs):
        touched.append(args)
        raise AssertionError("a model, checkpoint or pair pool was touched")
    stub = types.SimpleNamespace(load_for_scoring=boom, checkpoint_lineage=boom,
                                 measured_pairs=boom, development_panel_outcomes=boom,
                                 context=None, config={})
    stub.config = {"posttrain": {"methods": ["weighted_conditional_nll"]}}
    with pytest.raises(ValueError, match="contract_version"):
        services_mod.FlightServices.run_posttrain(stub)
    stub.config = {"posttrain": {"contract_version": 1, "methods": ["weighted_conditional_nll"]}}
    with pytest.raises(ValueError, match="contract_version"):
        services_mod.FlightServices.run_posttrain(stub)
    stub.config = {"posttrain": {"contract_version": 2,
                                 "methods": ["conditional_fit", "local_ipo"]}}
    with pytest.raises(ValueError, match="'conditional_fit'"):
        services_mod.FlightServices.run_posttrain(stub)
    stub.config = {"posttrain": {"contract_version": 2, "methods": ["local_ipo", "local_ipo"]}}
    with pytest.raises(ValueError, match="more than once"):
        services_mod.FlightServices.run_posttrain(stub)
    assert touched == []
    assert posttrain.arm_identity("local_ipo") == "local_ipo.v2"
    with pytest.raises(ValueError, match="not a v2 method"):
        posttrain.arm_identity("conditional_fit")


def test_the_same_pool_and_draw_seeds_give_every_arm_the_same_draws():
    pairs = pool_of(one_site_rows())["pairs"]

    def schedule():
        bank = posttrain.PairBank(pairs=tuple(pairs), reuse_cap=2)
        return [[p["pair_digest"] for p in bank.draw(2, seed=100 + step)[0]]
                for step in range(1, 5)]
    first, second = schedule(), schedule()
    assert first == second
    assert posttrain.schedule_digest(first) == posttrain.schedule_digest(second)
    assert posttrain.schedule_digest(first) != posttrain.schedule_digest(list(reversed(first)))


# ---------------------------------------------------------------------------
# 5. deterministic scoring with gradients on
# ---------------------------------------------------------------------------

class _Dropper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.drop = torch.nn.Dropout(p=0.9)

    def forward(self, hidden):
        return self.drop(self.linear(hidden))


def test_deterministic_scoring_is_eval_mode_with_gradients_and_restores_flags():
    torch.manual_seed(0)
    model = _Dropper()
    model.train()
    model.drop.eval()               # a module deliberately left in another mode
    hidden = torch.randn(3, 4)
    with posttrain.deterministic_scoring(model):
        assert posttrain.require_deterministic_scoring(model)
        first, second = model(hidden), model(hidden)
        assert torch.equal(first, second), "dropout made two scorings of one input differ"
        assert first.requires_grad and torch.is_grad_enabled()
        first.sum().backward()
        assert model.linear.weight.grad is not None
    assert model.training and model.linear.training and not model.drop.training
    model.drop.train()
    with pytest.raises(ValueError, match="training mode"):
        posttrain.require_deterministic_scoring(model, label="policy")
    model.eval()
    with torch.no_grad():
        with pytest.raises(ValueError, match="gradients are disabled"):
            posttrain.require_deterministic_scoring(model)
        assert posttrain.require_deterministic_scoring(model, gradients=False)


# ---------------------------------------------------------------------------
# 6. the real stage on the tiny DPLM harness
# ---------------------------------------------------------------------------

def test_pair_views_score_each_pair_under_its_own_condition(harness):
    services = harness.services
    pairs = make_pairs(services)
    views = services.pair_views(ROUTE_LORA, pairs)
    assert len(views) == 8
    assert {view[2] for view in views} == {Condition(FAMILY_AFFINITY, BIN_HIGH, "a1")}
    assert services.collate(ROUTE_LORA, views).condition_indices.shape == (8, 3)
    with pytest.raises(ValueError, match="incompatible"):
        services.pair_views(ROUTE_LORA, pairs, condition=Condition(
            FAMILY_AFFINITY, BIN_HIGH, "alphabind_her2:affinity:HER2"))
    with pytest.raises(ValueError, match="Unsupported pair condition"):
        services.pair_views(ROUTE_LORA, pairs, condition="prior_only")
    foreign = [dict(pairs[0], assay="zzz")]
    with pytest.raises(ValueError, match="not enrolled"):
        services.collate(ROUTE_LORA, services.pair_views(ROUTE_LORA, foreign))
    with pytest.raises(ValueError, match="never defaulted to HER2"):
        services.pair_views(ROUTE_LORA, [{k: v for k, v in pairs[0].items() if k != "assay"}])


def test_run_posttrain_runs_every_v2_arm_on_one_pool_with_real_updates(harness, monkeypatch):
    services, context = harness.services, harness.context
    harness.config["posttrain"]["methods"] = [
        "weighted_conditional_nll", "numeric_margin_match", "local_ipo", "local_dpo"]
    produce_bakeoff_best(harness, ROUTE_LORA)
    complete_stage(context, "route_gate", winner=ROUTE_LORA)
    monkeypatch.setattr(services, "measured_pairs", lambda: make_pairs(services))
    monkeypatch.setattr(services, "development_panel_outcomes", lambda: PANEL)
    calls = []
    original = services_mod.FlightServices.canonical_log_probs

    def recording(self, route, model, batch):
        calls.append({"training": any(module.training for module in model.modules()),
                      "grad": torch.is_grad_enabled(),
                      "trainable": any(p.requires_grad for p in model.parameters())})
        return original(self, route, model, batch)
    monkeypatch.setattr(services_mod.FlightServices, "canonical_log_probs", recording)
    result = services.run_posttrain()

    assert result["contract_version"] == 2
    assert set(result["methods"]) == {"weighted_conditional_nll", "numeric_margin_match",
                                      "local_ipo", "local_dpo"}
    pool = pairs_lib.pair_pool_digest(make_pairs(services))
    assert result["pairs"]["pool_sha256"] == pool
    assert result["matched_draws"]["identical_across_arms"] is True
    assert len(set(result["matched_draws"]["draw_schedule_sha256"].values())) == 1
    for method, block in result["methods"].items():
        assert block["contract_version"] == 2 and block["pair_pool_sha256"] == pool
        assert block["objective"] == posttrain.OBJECTIVES[method]
        assert block["parameter_movement"]["updated"] is True, (
            f"{method} drew pairs but no update reached the weights")
        assert block["updates"] == 2 and block["pairs_drawn"] == 4
        assert block["assay_exposure"]["a1"]["pairs_drawn"] == 4
        assert block["assay_exposure"]["a1"]["mean_alternative_mass"] is not None
        assert all(np.isfinite(entry["loss"]) for entry in block["history"])
        payload = torch.load(str(context.path("posttrain", method, "last.pt")),
                             map_location="cpu", weights_only=False)
        assert payload["identity"]["arm"] == posttrain.arm_identity(method)
        assert payload["posttrain"]["contract_version"] == 2
        assert payload["posttrain"]["pair_pool_sha256"] == pool
    nll = result["methods"]["weighted_conditional_nll"]["history"][0]
    assert 0.0 < nll["mean_alternative_mass"] < 1.0 and nll["vocabulary"] == 20

    policy = [call for call in calls if call["trainable"] and call["grad"]]
    reference = [call for call in calls if not call["trainable"]]
    assert len(policy) >= 8, "four arms x two updates of gradient-enabled policy scoring"
    assert not any(call["training"] for call in policy), "the policy was scored in training mode"
    assert reference and not any(call["training"] or call["grad"] for call in reference), (
        "the frozen reference must be eval and no_grad")


def test_run_posttrain_refuses_a_legacy_config_before_loading_a_model(harness, monkeypatch):
    services = harness.services
    monkeypatch.setattr(services, "load_for_scoring", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("a model was loaded")))
    harness.config["posttrain"]["methods"] = ["conditional_fit", "local_ipo"]
    with pytest.raises(ValueError, match="weighted_conditional_nll"):
        services.run_posttrain()
    harness.config["posttrain"]["methods"] = ["local_ipo"]
    del harness.config["posttrain"]["contract_version"]
    with pytest.raises(ValueError, match="contract_version"):
        services.run_posttrain()


def test_e4_requests_its_population_explicitly_and_records_what_it_sampled(harness, monkeypatch):
    services = harness.services
    best = produce_bakeoff_best(harness, ROUTE_LORA)
    seen = {}

    def fake(*, population=None, allow_empty=False):
        seen.update(population=population, allow_empty=allow_empty)
        return make_pairs(services)
    monkeypatch.setattr(services, "measured_pairs", fake)
    panel = services.panel_local_ordering(ROUTE_LORA, best)
    assert seen == {"population": {"assay": "a1", "family": "affinity"}, "allow_empty": True}
    assert panel["supported"] and panel["population"] == 4
    assert 0.0 <= panel["value"] <= 1.0
    assert panel["pairs_version"] == pairs_lib.PAIRS_VERSION
    assert panel["requested_population"] == {"assay": "a1", "family": "affinity"}
    assert panel["sampling"]["drawn"] == 4 and "content-addressed" in panel["sampling"]["rule"]

    monkeypatch.setattr(services, "measured_pairs", lambda **kw: [])
    monkeypatch.setattr(services, "load_for_scoring", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("scored with no pairs")))
    empty = services.panel_local_ordering(ROUTE_LORA, best)
    assert empty["supported"] is False and empty["value"] is None
    assert "population" in empty["blocking_reason"]


# ---------------------------------------------------------------------------
# 7. the T4-v2 config, through the real loader
# ---------------------------------------------------------------------------

CONFIGS = ROOT / "configs" / "experiments"
V2_PATH = CONFIGS / "her2_generative_routes_t4v2.json"


def test_the_t4v2_config_loads_through_the_real_loader_and_cannot_point_at_an_existing_flight(
        tmp_path):
    config = spec.load_config(V2_PATH)
    assert config["config_path"] == "configs/experiments/her2_generative_routes_t4v2.json"
    others = [json.loads(path.read_text(encoding="utf-8"))
              for path in CONFIGS.glob("her2_generative_routes*.json") if path != V2_PATH]
    assert others, "no sibling config found to compare against"
    for other in others:
        assert other.get("run_root_leaf") != config["run_root_leaf"]
        assert other.get("campaign_id") != config["campaign_id"]
        assert other.get("config_path") != config["config_path"]
        assert (other.get("revisions") or {}).get("current") != config["revisions"]["current"]
    base_path = CONFIGS / "her2_generative_routes.json"
    context = spec.resolve_context(ROOT, config_path=V2_PATH, artifact_root=tmp_path,
                                   ensure=False)
    legacy = spec.resolve_context(ROOT, config_path=base_path, artifact_root=tmp_path,
                                  ensure=False)
    assert context.run_root != legacy.run_root
    assert context.run_root == tmp_path / "runs" / "t4v2_rev01" / "flight_t4v2"
    assert not context.run_root.exists(), "resolving a context must not create the run root"


def test_the_t4v2_config_differs_from_the_legacy_template_only_where_it_says_it_does():
    new = json.loads(V2_PATH.read_text(encoding="utf-8"))
    old = json.loads((CONFIGS / "her2_generative_routes.json").read_text(encoding="utf-8"))
    for document in (new, old):
        for key in ("campaign_id", "config_path", "run_root_leaf", "opt_in_note", "posttrain"):
            document.pop(key, None)
        document["revisions"] = {k: v for k, v in document["revisions"].items()
                                 if k not in ("current", "reason")}
        document["curriculum"].pop("contrast_rule")
        document["gates"].pop("local_pairs_note", None)
    assert new == old, "the copy drifted from the legacy template outside the declared differences"


def test_the_t4v2_posttrain_block_is_a_valid_v2_contract_and_the_legacy_one_is_refused():
    new = spec.load_config(V2_PATH)["posttrain"]
    assert posttrain.require_contract(new) == 2
    methods = posttrain.resolve_methods(new["methods"])
    assert methods[0] == "weighted_conditional_nll" and set(methods) == set(posttrain.METHODS)
    assert "population" not in new, "the pool is pooled; a HER2 restriction must be declared"
    legacy = spec.load_config(CONFIGS / "her2_generative_routes.json")["posttrain"]
    assert "conditional_fit" in legacy["methods"], "the legacy template must stay unchanged"
    with pytest.raises(ValueError, match="contract_version"):
        posttrain.require_contract(legacy)
    with pytest.raises(ValueError, match="conditional_fit"):
        posttrain.resolve_methods(legacy["methods"])


# ---------------------------------------------------------------------------
# 8. hooks: a toy known mechanism
# ---------------------------------------------------------------------------

class _Scale(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))

    def forward(self, hidden):
        return hidden * self.scale


class _Mechanism(torch.nn.Module):
    """``output = 2 * h[:, 1, 0]``: token 1, channel 0 is the ONLY feature the output reads."""

    def __init__(self):
        super().__init__()
        self.trunk = _Scale()
        self.drop = torch.nn.Dropout(p=0.9)

    def forward(self, tokens):
        return 2.0 * self.drop(self.trunk(tokens))[:, 1, 0]


def tokens(feature, *, rows=2, seed=0):
    values = torch.randn(rows, 3, 4, generator=torch.Generator().manual_seed(seed))
    values[:, 1, 0] = torch.as_tensor(feature, dtype=values.dtype)
    return values


def batch_of(feature, *, ids=None, rows=2, seed=0):
    return {"x": tokens(feature, rows=rows, seed=seed),
            "ids": list(ids or [f"row{index}" for index in range(rows)])}


def run_model(model, batch):
    return model(batch["x"])


def toy_registry(model):
    registry = hooks_lib.HookRegistry(route="toy")
    name = registry.register("route/toy/block0/resid_post", model.trunk, site="resid_post",
                             layout=hooks_lib.LAYOUT_TOKENS_FIRST)
    return registry, name


def donor_reference(model, registry, name, *, values=(2.0, 4.0), ids=("donor-a", "donor-b")):
    batches = [batch_of(value, rows=1, ids=[identity], seed=10 + index)
               for index, (value, identity) in enumerate(zip(values, ids))]
    return interp.build_reference_mean(
        model, registry, donor_batches=batches, name=name, forward=run_model,
        ids_of=lambda batch: batch["ids"], provenance={"donor_set": "toy"})


def test_a_known_mechanism_is_recovered_by_ablation_patch_rescue_and_controls():
    model = _Mechanism()
    model.train()
    registry, name = toy_registry(model)
    clean, corrupted, strong = batch_of(1.0), batch_of(0.0, seed=1), batch_of(5.0, seed=2)
    rng_before = torch.get_rng_state()

    def output(batch, **plan):
        if not plan:
            return interp.capture(model, registry, batch=batch, forward=run_model)["output"]
        return interp.intervene(model, registry, batch=batch, name=name, forward=run_model,
                                **plan)["output"]
    baseline = interp.capture(model, registry, batch=clean, forward=run_model)
    assert not baseline["output"].requires_grad, "capture must not build a graph"
    assert torch.allclose(baseline["output"], torch.full((2,), 2.0))
    assert torch.allclose(output(corrupted), torch.zeros(2))

    # necessary feature: zeroing token 1 removes the output; zeroing token 0 (control) does not
    assert torch.allclose(output(clean, kind="zero_ablate", positions=[1]), torch.zeros(2))
    assert torch.allclose(output(clean, kind="zero_ablate", positions=[0]), torch.full((2,), 2.0))
    assert torch.allclose(output(clean, kind="zero_ablate", positions=[2]), torch.full((2,), 2.0))

    # donor patch: a strong donor at token 1 drives the output to the donor's; token 0 does not
    donor = interp.capture(model, registry, batch=strong, names=[name],
                           forward=run_model)["exports"][name]
    assert output(clean, kind="patch", source=donor, positions=[1]).tolist() == [10.0, 10.0]
    assert torch.allclose(output(clean, kind="patch", source=donor, positions=[0]),
                          torch.full((2,), 2.0))

    # rescue: patching the CLEAN activation into the corrupted run restores the output at the
    # necessary position and does nothing at the irrelevant one
    healthy = interp.capture(model, registry, batch=clean, names=[name],
                             forward=run_model)["exports"][name]
    assert torch.allclose(output(corrupted, kind="patch", source=healthy, positions=[1]),
                          torch.full((2,), 2.0))
    assert torch.allclose(output(corrupted, kind="patch", source=healthy, positions=[0]),
                          torch.zeros(2))

    # nothing leaked: hooks off, plans cleared, no grads, training mode and RNG state unchanged
    assert registry.handles == [] and registry.interventions == {}
    assert model.training and model.drop.training and model.trunk.training
    assert all(parameter.grad is None for parameter in model.parameters())
    assert torch.equal(torch.get_rng_state(), rng_before)


def test_mean_ablation_uses_a_frozen_donor_mean_and_is_batch_invariant():
    model = _Mechanism()
    registry, name = toy_registry(model)
    reference = donor_reference(model, registry, name)
    assert reference.donor_ids == ("donor-a", "donor-b")
    assert reference.document()["donor_set"] == "toy"
    assert float(reference.mean[0, 1, 0]) == pytest.approx(3.0, abs=1e-5)
    full = {"x": tokens([1.0, 5.0, 9.0], rows=3, seed=4), "ids": ["r0", "r1", "r2"]}

    def ablated(batch):
        return interp.intervene(model, registry, batch=batch, name=name, kind="mean_ablate",
                                forward=run_model, positions=[1], reference=reference)["output"]
    together = ablated(full)
    assert torch.allclose(together, torch.full((3,), 6.0)), "2 * the donor mean 3.0"
    for index in range(3):
        alone = ablated({"x": full["x"][index:index + 1], "ids": [full["ids"][index]]})
        assert torch.allclose(alone, together[index:index + 1]), "the batch changed the control"
    # What the retired behaviour would have produced for this batch: the mean of the batch itself.
    assert float(2.0 * full["x"][:, 1, 0].mean()) == pytest.approx(10.0) != 6.0


def test_mean_ablation_without_a_reference_raises_and_there_is_no_self_mean_fallback():
    model = _Mechanism()
    registry, name = toy_registry(model)
    with pytest.raises(ValueError, match="explicit frozen donor"):
        registry.plan(name, kind="mean_ablate")
    with pytest.raises(ValueError, match="explicit frozen donor"):
        registry.plan(name, kind="mean_ablate", reference=torch.zeros(1, 3, 4))
    with pytest.raises(ValueError, match="takes no reference mean"):
        registry.plan(name, kind="zero_ablate", reference=donor_reference(model, registry, name))
    with pytest.raises(TypeError):
        interp.cross_condition_experiment(
            model, registry, batch_a=batch_of(1.0), batch_b=batch_of(0.0), name=name,
            forward=run_model, metric=lambda out: float(out.mean()))
    with pytest.raises(ValueError, match="explicit frozen ReferenceMean"):
        interp.cross_condition_experiment(
            model, registry, batch_a=batch_of(1.0), batch_b=batch_of(0.0), name=name,
            forward=run_model, metric=lambda out: float(out.mean()), mean_reference=None)
    # a hook that somehow carries a mean_ablate plan with no reference refuses too
    registry.interventions[name] = {"kind": "mean_ablate", "positions": None, "source": None,
                                    "reference": None}
    registry.enable([name])
    with pytest.raises(ValueError, match="no batch-mean fallback"):
        model.trunk(torch.zeros(1, 3, 4))
    registry.disable()


def test_a_reference_must_match_its_tap_layout_dimensions_and_digest():
    model = _Mechanism()
    registry, name = toy_registry(model)
    reference = donor_reference(model, registry, name)
    other = hooks_lib.reference_mean(torch.randn(2, 3, 4), name="route/toy/block9/resid_post",
                                     layout=hooks_lib.LAYOUT_TOKENS_FIRST)
    with pytest.raises(ValueError, match="was built for"):
        registry.plan(name, kind="mean_ablate", reference=other)
    wrong_layout = hooks_lib.reference_mean(torch.randn(2, 3, 4), name=name,
                                            layout=hooks_lib.LAYOUT_TOKENS_LAST)
    with pytest.raises(ValueError, match="this tap is"):
        registry.plan(name, kind="mean_ablate", reference=wrong_layout)
    registry.plan(name, kind="mean_ablate", reference=reference)
    registry.enable([name])
    with pytest.raises(ValueError, match="non-batch shape"):
        model.trunk(torch.zeros(1, 5, 4))          # a 5-token row against a 3-token reference
    registry.disable()
    reference.mean.add_(1.0)                       # an in-place edit after freezing
    with pytest.raises(ValueError, match="no longer matches its frozen digest"):
        registry.plan(name, kind="mean_ablate", reference=reference)


def test_the_reference_is_detached_cloned_and_one_value_per_position():
    donors = torch.randn(4, 3, 5, requires_grad=True)
    reference = hooks_lib.reference_mean(donors, name="route/toy/block0/resid_post",
                                         layout=hooks_lib.LAYOUT_TOKENS_FIRST)
    assert reference.mean.shape == (1, 3, 5) and not reference.mean.requires_grad
    snapshot = reference.mean.clone()
    with torch.no_grad():
        donors.add_(100.0)
    assert torch.equal(reference.mean, snapshot), "the reference aliases its donors"
    assert reference.verify()
    produced = reference.value_for(torch.zeros(7, 3, 5))
    assert produced.shape == (7, 3, 5)
    produced.add_(1.0)
    assert torch.equal(reference.mean, snapshot), "writing through the broadcast edited the mean"
    assert reference.counts.tolist() == [4, 4, 4]


def test_padding_is_excluded_from_the_reference_and_an_unreached_position_is_refused():
    name, layout = "route/toy/block0/resid_post", hooks_lib.LAYOUT_TOKENS_FIRST
    activations = torch.zeros(2, 4, 2)
    activations[0] = torch.tensor([[1., 1.], [2., 2.], [3., 3.], [4., 4.]])
    activations[1] = torch.tensor([[3., 3.], [4., 4.], [1e6, 1e6], [1e6, 1e6]])   # padding garbage
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    reference = hooks_lib.reference_mean(activations, name=name, layout=layout, mask=mask)
    assert reference.mean[0, :, 0].tolist() == [2.0, 3.0, 3.0, 4.0]
    assert reference.counts.tolist() == [2, 2, 1, 1]
    assert reference.document()["padding_masked"] is True
    with pytest.raises(ValueError, match="no real donor row"):
        hooks_lib.reference_mean(activations, name=name, layout=layout,
                                 mask=torch.tensor([[1, 1, 0, 1], [1, 1, 0, 1]]))
    with pytest.raises(ValueError, match="padding mask"):
        hooks_lib.reference_mean(activations, name=name, layout=layout, mask=torch.ones(2, 3))
    with pytest.raises(ValueError, match="not finite"):
        bad = activations.clone()
        bad[0, 0, 0] = float("nan")
        hooks_lib.reference_mean(bad, name=name, layout=layout)


def test_held_out_rows_cannot_be_donors_and_donor_provenance_is_frozen():
    model = _Mechanism()
    registry, name = toy_registry(model)
    reference = donor_reference(model, registry, name)
    stored = reference.document()
    assert stored["donor_ids"] == ["donor-a", "donor-b"] and stored["donor_rows"] == 2
    assert stored["donor_ids_named"] == 2
    assert stored["sha256"] == reference.sha256 and stored["ids_supplied"] is True
    hooks_lib.reference_mean(torch.randn(2, 3, 4), name=name, layout=hooks_lib.LAYOUT_TOKENS_FIRST,
                             donor_ids=["a", "b"], evaluation_ids=["c"])
    with pytest.raises(ValueError, match="also donors"):
        hooks_lib.reference_mean(torch.randn(2, 3, 4), name=name,
                                 layout=hooks_lib.LAYOUT_TOKENS_FIRST, donor_ids=["a", "b"],
                                 evaluation_ids=["b"])
    with pytest.raises(ValueError, match="carry no ids"):
        hooks_lib.reference_mean(torch.randn(2, 3, 4), name=name,
                                 layout=hooks_lib.LAYOUT_TOKENS_FIRST, evaluation_ids=["x"])
    with pytest.raises(ValueError, match="more than once"):
        hooks_lib.reference_mean(torch.randn(2, 3, 4), name=name,
                                 layout=hooks_lib.LAYOUT_TOKENS_FIRST, donor_ids=["a", "a"])
    with pytest.raises(ValueError, match="also donors"):
        interp.cross_condition_experiment(
            model, registry, batch_a=batch_of(1.0), batch_b=batch_of(0.0), name=name,
            forward=run_model, metric=lambda out: float(out.mean()), mean_reference=reference,
            evaluation_ids=["donor-a"])


def test_the_cross_condition_experiment_threads_the_reference_and_claims_no_pathway():
    model = _Mechanism()
    model.train()
    model.drop.eval()
    registry, name = toy_registry(model)
    reference = donor_reference(model, registry, name)
    rng_before = torch.get_rng_state()
    result = interp.cross_condition_experiment(
        model, registry, batch_a=batch_of(1.0), batch_b=batch_of(0.0, seed=1), name=name,
        forward=run_model, metric=lambda out: float(out.mean()), mean_reference=reference,
        positions=[1], evaluation_ids=["held-out-1"])
    assert result["patch"]["difference"] == pytest.approx(2.0)          # 0 -> 2: condition A's value
    assert result["zero_ablate"]["difference"] == pytest.approx(0.0)    # already zero
    assert result["mean_ablate"]["difference"] == pytest.approx(6.0)    # 2 * donor mean 3.0
    assert result["mean_reference"]["sha256"] == reference.sha256
    assert result["mean_reference"]["donor_ids"] == ["donor-a", "donor-b"]
    assert "does not isolate a pathway" in result["reading"]
    assert "isolates the conditioning pathway" not in result["reading"]
    assert model.training and not model.drop.training and model.trunk.training
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert registry.handles == [] and registry.interventions == {}


def test_controlled_evaluation_never_touches_cuda_on_a_cpu_run(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CUDA was touched in a CPU process")
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    for name in ("get_rng_state", "set_rng_state", "get_rng_state_all", "set_rng_state_all",
                 "device_count", "manual_seed", "manual_seed_all", "current_device"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    model = _Mechanism()
    model.train()
    before = torch.get_rng_state()
    with interp.controlled_evaluation(model, seed=11):
        assert not model.training and not torch.is_grad_enabled()
        torch.rand(3)                           # consumes the forked generator only
    assert model.training and torch.equal(torch.get_rng_state(), before)
    with interp.controlled_evaluation():
        assert not torch.is_grad_enabled()


# ---------------------------------------------------------------------------
# 9. layouts: export, write, read and patch
# ---------------------------------------------------------------------------

class _Tap(torch.nn.Module):
    def forward(self, hidden):
        return hidden + 0.0


def two_layout_registry():
    first, second = _Tap(), _Tap()
    registry = hooks_lib.HookRegistry(route="toy")
    first_name = registry.register("route/toy/block0/resid_post", first, site="resid_post",
                                   layout=hooks_lib.LAYOUT_TOKENS_FIRST)
    second_name = registry.register("route/toy/block1/resid_post", second, site="resid_post",
                                    layout=hooks_lib.LAYOUT_TOKENS_LAST)
    return registry, (first, first_name), (second, second_name)


def test_a_square_tensor_is_ablated_on_the_declared_axis_for_each_layout():
    registry, (first, first_name), (second, second_name) = two_layout_registry()
    square = torch.ones(2, 4, 4)
    registry.plan(first_name, kind="zero_ablate", positions=[1])
    registry.plan(second_name, kind="zero_ablate", positions=[1])
    registry.enable()
    tokens_first, tokens_last = first(square), second(square)
    registry.disable()
    assert float(tokens_first[:, 1, :].abs().sum()) == 0.0 and float(tokens_first[:, :, 1].sum()) > 0
    assert float(tokens_last[:, :, 1].abs().sum()) == 0.0 and float(tokens_last[:, 1, :].sum()) > 0
    assert registry.position_axis(first_name) == 1 and registry.position_axis(second_name) == 2


def test_patch_sources_carry_their_layout_so_a_square_patch_cannot_transpose():
    registry, (first, first_name), (second, second_name) = two_layout_registry()
    asymmetric = torch.arange(32, dtype=torch.float32).reshape(2, 4, 4)
    registry.enable([first_name])
    first(asymmetric)
    registry.disable()
    exported = registry.export(first_name)
    assert exported["layout"] == hooks_lib.LAYOUT_TOKENS_FIRST and exported["position_axis"] == 1
    with pytest.raises(ValueError, match="must carry its layout"):
        registry.plan(first_name, kind="patch", source=asymmetric)       # a bare tensor
    with pytest.raises(ValueError, match="captured as"):
        registry.plan(second_name, kind="patch", source=exported)       # square: shapes agree
    with pytest.raises(ValueError, match="needs a declared layout"):
        hooks_lib.PatchSource(tensor=asymmetric, layout="whatever")
    with pytest.raises(ValueError, match="contradicts layout"):
        registry.plan(first_name, kind="patch",
                      source={"tensor": asymmetric, "layout": "tokens_first", "position_axis": 2})
    registry.plan(first_name, kind="patch", source=exported)
    registry.enable([first_name])
    patched = first(torch.zeros(2, 4, 4))
    registry.disable()
    assert torch.equal(patched, asymmetric), "the layout-matched patch must be the identity copy"
    assert registry.interventions[first_name]["source"].tensor is not exported["tensor"]


def test_export_never_guesses_a_layout():
    registry, (first, first_name), _second = two_layout_registry()
    with pytest.raises(ValueError, match="has not been captured"):
        registry.export(first_name)
    registry.captured[first_name] = torch.zeros(1, 2, 3)        # captured outside the registry
    with pytest.raises(ValueError, match="without a recorded layout"):
        registry.export(first_name)
    registry.captured_layout[first_name] = "nonsense"
    with pytest.raises(ValueError, match="unknown layout"):
        registry.export(first_name)
    registry.captured_layout.clear()
    registry.enable([first_name])
    first(torch.zeros(1, 2, 3))
    registry.disable()
    assert set(registry.export_all()) == {first_name}


def token_rows(count):
    return [{"index": index} for index in range(count)]


def test_write_and_read_roundtrip_both_layouts_including_square_dimensions(tmp_path):
    registry, (first, first_name), (second, second_name) = two_layout_registry()
    square = torch.arange(32, dtype=torch.float32).reshape(2, 4, 4)
    registry.enable()
    first(square)
    second(square)
    registry.disable()
    exports = registry.export_all()
    document = {"record_kind": "activations", "route": "toy", "token_map": token_rows(4)}
    record = interp.write_activations(
        tmp_path, "square", exports, document, run_root=tmp_path, logical_prefix="acts/square",
        token_map=token_rows(4))
    assert record["activation_layouts"][first_name] == {
        "layout": "tokens_first", "position_axis": 1, "shape": [2, 4, 4], "stored_dtype": "float32"}
    assert record["activation_layouts"][second_name]["position_axis"] == 2
    read, stored = interp.read_activations(tmp_path, "square")
    assert stored["activation_layouts"] == record["activation_layouts"]
    for name in (first_name, second_name):
        assert np.array_equal(read[name]["tensor"], square.numpy())
        assert read[name]["layout"] == exports[name]["layout"]
        assert read[name]["position_axis"] == exports[name]["position_axis"]


def test_the_token_map_is_checked_against_the_position_axis_not_dim_one(tmp_path):
    channels_first = torch.randn(2, 3, 5)                 # (batch, channels=3, positions=5)
    document = {"record_kind": "activations", "route": "toy", "token_map": token_rows(5)}
    exported = {"name": "conv", "tensor": channels_first, "layout": "tokens_last",
                "position_axis": 2, "shape": [2, 3, 5]}
    interp.write_activations(tmp_path, "conv", {"conv": exported}, document, run_root=tmp_path,
                             logical_prefix="acts/conv", token_map=token_rows(5))
    read, _record = interp.read_activations(tmp_path, "conv")
    assert read["conv"]["shape"] == [2, 3, 5] and read["conv"]["position_axis"] == 2
    # The OLD check read shape[1] == 3: a three-row map would have been accepted for a tensor
    # whose residue axis has five positions. It is refused now.
    with pytest.raises(ValueError, match="mapped indices against a tensor of length 5"):
        interp.write_activations(
            tmp_path, "conv_wrong", {"conv": exported}, dict(document, token_map=token_rows(3)),
            run_root=tmp_path, logical_prefix="acts/conv_wrong", token_map=token_rows(3))
    tokens_first = {"tensor": torch.randn(2, 5, 3), "layout": "tokens_first"}
    interp.write_activations(tmp_path, "tf", {"tf": tokens_first},
                             dict(document, token_map=token_rows(5)), run_root=tmp_path,
                             logical_prefix="acts/tf", token_map=token_rows(5))


def test_the_writer_refuses_missing_unknown_or_contradictory_layouts(tmp_path):
    document = {"record_kind": "activations", "token_map": token_rows(3)}
    common = dict(run_root=tmp_path, token_map=token_rows(3))
    with pytest.raises(ValueError, match="no layout was supplied"):
        interp.write_activations(tmp_path, "bare", {"x": torch.zeros(2, 3, 4)}, document,
                                 logical_prefix="acts/bare", **common)
    with pytest.raises(ValueError, match="unknown layout"):
        interp.write_activations(tmp_path, "unknown", {"x": {"tensor": torch.zeros(2, 3, 4),
                                                              "layout": "channels_first"}},
                                 document, logical_prefix="acts/unknown", **common)
    with pytest.raises(ValueError, match="puts position on axis"):
        interp.write_activations(
            tmp_path, "contradict",
            {"x": {"tensor": torch.zeros(2, 3, 4), "layout": "tokens_first", "position_axis": 2}},
            document, logical_prefix="acts/contradict", **common)
    interp.write_activations(tmp_path, "supplied", {"x": torch.zeros(2, 3, 4)}, document,
                             logical_prefix="acts/supplied", layouts={"x": "tokens_first"}, **common)


def test_the_writer_detaches_and_moves_tensors_to_cpu_safely(tmp_path):
    grad = torch.randn(2, 3, 4, requires_grad=True)
    half = torch.randn(2, 3, 4).to(torch.bfloat16)
    document = {"record_kind": "activations", "token_map": token_rows(3)}
    interp.write_activations(
        tmp_path, "safe", {"grad": {"tensor": grad * 2.0, "layout": "tokens_first"},
                           "half": {"tensor": half, "layout": "tokens_first"}},
        document, run_root=tmp_path, logical_prefix="acts/safe", token_map=token_rows(3))
    read, _record = interp.read_activations(tmp_path, "safe")
    assert read["grad"]["tensor"].dtype == np.float32 and read["half"]["tensor"].dtype == np.float32
    assert np.allclose(read["grad"]["tensor"], (grad * 2.0).detach().numpy())


def test_a_shard_with_no_recorded_layouts_is_not_read_with_a_guessed_one(tmp_path):
    from smallAntibodyGen.experiments import her2_support_paths as paths
    paths.write_shard(tmp_path, "legacy", {"x": np.zeros((2, 3, 4), dtype=np.float32)},
                      {"record_kind": "activations", "token_map": token_rows(3)},
                      order="capture order", logical_prefix="acts/legacy", run_root=tmp_path)
    with pytest.raises(ValueError, match="cannot be read with a guessed one"):
        interp.read_activations(tmp_path, "legacy")


# ---------------------------------------------------------------------------
# 10. review repairs
# ---------------------------------------------------------------------------

# -- the T4-v2 config's inherited prose ---------------------------------------------------------

def test_the_t4v2_external_sources_note_keeps_the_legacy_closure_history():
    """A copy test that compares the whole config would also pass if both sides drifted; this one
    pins the sentence that a hand-typed copy dropped."""
    new = json.loads(V2_PATH.read_text(encoding="utf-8"))["artifacts"]["external_sources_note"]
    old = json.loads((CONFIGS / "her2_generative_routes.json").read_text(
        encoding="utf-8"))["artifacts"]["external_sources_note"]
    assert new == old
    assert ("The earlier hand-written list named ablang2/__init__.py; the pinned package is at "
            "upstream/ablang2/ablang2/__init__.py, so the acquisition verification failed against "
            "its own note that the closure was three paths.") in new


# -- hooks: a capture is a copy ------------------------------------------------------------------

REF_NAME = "route/toy/block0/resid_post"


def identity_registry(*, site="resid_post"):
    """One tap on ``torch.nn.Identity``, which hands back the very tensor it was given."""
    tap = torch.nn.Identity()
    registry = hooks_lib.HookRegistry(route="toy")
    name = registry.register(f"route/toy/block0/{site}", tap, site=site,
                             layout=hooks_lib.LAYOUT_TOKENS_FIRST)
    return registry, tap, name


@pytest.mark.parametrize("site", ["resid_post", "resid_pre"])
def test_a_capture_is_a_copy_taken_at_the_tap_not_a_view_of_what_the_model_keeps(site):
    registry, tap, name = identity_registry(site=site)
    registry.enable([name])
    x = torch.ones(2, 3, 4)
    y = tap(x)
    assert y is x, "Identity returns its input: this is the aliasing case"
    y.add_(5)                                         # an in-place edit AFTER the tap
    registry.disable()
    exported = registry.export(name)
    assert torch.equal(exported["tensor"], torch.ones(2, 3, 4)), (
        f"{site}: the capture followed an in-place edit made after the tap")
    assert exported["tensor"].data_ptr() != x.data_ptr()
    assert torch.equal(x, torch.full((2, 3, 4), 6.0))


def test_a_capture_under_an_intervention_is_the_pre_intervention_activation_and_stays_so():
    registry, tap, name = identity_registry()
    registry.plan(name, kind="zero_ablate", positions=[1])
    registry.enable([name])
    x = torch.ones(2, 3, 4)
    y = tap(x)
    registry.disable()
    assert float(y[:, 1].abs().sum()) == 0.0 and float(y[:, 0].sum()) > 0.0
    x.add_(5)                                         # the caller edits its own input afterwards
    assert torch.equal(registry.export(name)["tensor"], torch.ones(2, 3, 4))


# -- hooks: reference-mean provenance ------------------------------------------------------------

def donors_of(rows=3):
    return torch.randn(rows, 2, 4, generator=torch.Generator().manual_seed(rows))


def test_provenance_cannot_override_the_record_it_describes():
    layout = hooks_lib.LAYOUT_TOKENS_FIRST
    attack = {"sha256": "overridden", "donor_ids": ["eval"]}
    with pytest.raises(ValueError, match="sha256"):
        hooks_lib.reference_mean(donors_of(), name=REF_NAME, layout=layout,
                                 donor_ids=["a", "b", "c"], provenance=attack)
    for key in sorted(hooks_lib.RESERVED_DOCUMENT_KEYS | hooks_lib.RESERVED_BUILDER_KEYS):
        with pytest.raises(ValueError, match="may not set"):
            hooks_lib.reference_mean(donors_of(), name=REF_NAME, layout=layout,
                                     provenance={key: "x"})
    honest = hooks_lib.reference_mean(donors_of(), name=REF_NAME, layout=layout,
                                      donor_ids=["a", "b", "c"], provenance={"donor_set": "toy"})
    document = honest.document()
    assert document["sha256"] == honest.sha256 and document["donor_ids"] == ["a", "b", "c"]
    assert document["donor_set"] == "toy"
    # building the record directly cannot smuggle the same keys in either
    with pytest.raises(ValueError, match="reserved key"):
        dataclasses.replace(honest, provenance={"sha256": "forged"})


def test_provenance_is_frozen_canonicalised_and_covered_by_the_digest():
    layout = hooks_lib.LAYOUT_TOKENS_FIRST
    provenance = {"donor_set": {"studies": ["s1", "s2"], "n": 2}, "tags": ["a"]}
    reference = hooks_lib.reference_mean(donors_of(), name=REF_NAME, layout=layout,
                                         provenance=provenance)
    digest, document = reference.sha256, reference.document()
    # external nested mutation of the containers the caller passed in
    provenance["donor_set"]["studies"].append("evil")
    provenance["donor_set"]["n"] = 99
    provenance["tags"].append("evil")
    provenance["added"] = True
    assert reference.document() == document
    assert reference.digest() == digest == reference.sha256 and reference.verify()
    # mutation of the frozen object itself
    with pytest.raises(TypeError):
        reference.provenance["tags"] = ()
    with pytest.raises(TypeError):
        reference.provenance["donor_set"]["n"] = 1
    assert isinstance(reference.provenance["tags"], tuple)
    assert not hasattr(reference.provenance["tags"], "append")
    with pytest.raises(dataclasses.FrozenInstanceError):
        reference.provenance = {}
    with pytest.raises(dataclasses.FrozenInstanceError):
        reference.donor_ids = ("x",)
    # the document is a copy: editing it does not reach the reference
    copy = reference.document()
    copy["donor_set"]["studies"].append("x")
    copy["tags"].append("x")
    copy["donor_ids"].append("x")
    copy["sha256"] = "z"
    assert reference.document() == document
    # provenance is part of the digest, so a swap made behind the frozen API is caught too
    other = hooks_lib.reference_mean(donors_of(), name=REF_NAME, layout=layout,
                                     provenance={"donor_set": "different"})
    assert other.sha256 != reference.sha256
    object.__setattr__(reference, "provenance", other.provenance)
    with pytest.raises(ValueError, match="no longer matches its frozen digest"):
        reference.verify()
    named = hooks_lib.reference_mean(donors_of(), name=REF_NAME, layout=layout,
                                     donor_ids=["a", "b", "c"])
    object.__setattr__(named, "donor_ids", ("eval", "b", "c"))
    with pytest.raises(ValueError, match="no longer matches its frozen digest"):
        named.verify()


def test_provenance_must_be_plain_json_data():
    layout = hooks_lib.LAYOUT_TOKENS_FIRST
    for bad, message in ((torch.zeros(2), "plain JSON data"),
                         ({"k": float("nan")}, "not finite"),
                         ({1: "x"}, "non-string key"),
                         ({"k": {1, 2}}, "plain JSON data")):
        with pytest.raises(ValueError, match=message):
            hooks_lib.reference_mean(donors_of(), name=REF_NAME, layout=layout, provenance=bad)


def test_the_actual_donor_row_count_is_distinct_from_the_named_ids():
    layout = hooks_lib.LAYOUT_TOKENS_FIRST
    unnamed = hooks_lib.reference_mean(donors_of(3), name=REF_NAME, layout=layout)
    document = unnamed.document()
    assert document["donor_rows"] == 3 and document["donor_ids_named"] == 0
    assert document["donor_ids"] == [] and unnamed.donor_ids == ()
    named = hooks_lib.reference_mean(donors_of(3), name=REF_NAME, layout=layout,
                                     donor_ids=["c", "a", "b"])
    assert named.donor_ids == ("a", "b", "c") and isinstance(named.donor_ids, tuple)
    assert named.document()["donor_rows"] == 3 == named.document()["donor_ids_named"]
    builder = hooks_lib.ReferenceMeanBuilder(name=REF_NAME, layout=layout)
    builder.add(donors_of(2)).add(donors_of(5))
    assert builder.build().donor_rows == 7, "rows add up across donor batches"
    with pytest.raises(ValueError, match="bare string"):
        dataclasses.replace(named, donor_ids="abc")
    with pytest.raises(ValueError, match="named donor ids"):
        dataclasses.replace(named, donor_ids=("a", "b"))
    with pytest.raises(ValueError, match="more than once"):
        dataclasses.replace(named, donor_ids=("a", "a", "b"))


def test_reference_building_capture_and_ablation_stay_on_the_cpu(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CUDA was initialised in a CPU process")
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    registry, tap, name = identity_registry()
    reference = hooks_lib.reference_mean(
        donors_of(3), name=name, layout=hooks_lib.LAYOUT_TOKENS_FIRST, donor_ids=["a", "b", "c"],
        evaluation_ids=["held-out"], provenance={"donor_set": "toy"})
    registry.plan(name, kind="mean_ablate", reference=reference)
    registry.enable([name])
    produced = tap(torch.ones(2, 2, 4))
    registry.disable()
    assert produced.device.type == "cpu" and reference.verify()
    assert registry.export(name)["tensor"].device.type == "cpu"
    assert torch.equal(produced[0], reference.mean[0]), "mean ablation replaced the activation"


# -- pairs: the measured target, duplicates and imported-pair content ----------------------------

def single_contrast(rows=None):
    _summaries, block = contrasts_of(rows or one_site_rows())
    return next(c for c in block["contrasts"] if c.kind == data_lib.CONTRAST_SINGLE)


def with_value(summary, value, **changes):
    """A copy of a construct summary whose one replicate is a different measured number."""
    return dataclasses.replace(summary, measurements=(replicate(value),), **changes)


def pool_from(contrasts, *, enrolled=(ASSAY,)):
    return pairs_lib.build_measured_pairs(list(contrasts), violates=lambda row: False,
                                          enrolled_assays=enrolled)


def swapped_orientation(contrast):
    return dataclasses.replace(
        contrast, left=contrast.right, right=contrast.left, left_value=contrast.right_value,
        right_value=contrast.left_value, margin=-contrast.margin)


def test_the_target_digest_moves_with_values_winners_and_provenance_but_structural_ids_do_not():
    base = pool_of(one_site_rows())
    for label, other in (("a changed value", pool_of(one_site_rows(e=8.2))),
                         ("a flipped winner", pool_of(one_site_rows(e=6.5)))):
        assert other["counts"]["pool_sha256"] == base["counts"]["pool_sha256"], label
        assert other["counts"]["targets_sha256"] != base["counts"]["targets_sha256"], label
        before = {pair["id"]: pair for pair in base["pairs"]}
        after = {pair["id"]: pair for pair in other["pairs"]}
        assert set(before) == set(after), f"{label} changed which structural pairs exist"
        for identity, pair in after.items():
            involves_parent = HEAVY in (pair["chosen"].construct.heavy,
                                        pair["rejected"].construct.heavy)
            same = pair["target_digest"] == before[identity]["target_digest"]
            assert same != involves_parent, (
                f"{label}: a pair's target digest must change exactly when one of its own "
                f"endpoints' measurements did ({identity})")
    assert pairs_lib.pair_targets_digest(base["pairs"]) == base["counts"]["targets_sha256"]
    assert pairs_lib.pair_targets_digest(base["pairs"]) == pairs_lib.pair_targets_digest(
        reversed(base["pairs"])), "the pool target digest depends on enumeration order"

    contrast = single_contrast()
    reference = pool_from([contrast])["pairs"][0]
    reprovenanced = pool_from([dataclasses.replace(
        contrast, right=dataclasses.replace(contrast.right, study="a-different-study"))])["pairs"][0]
    assert reprovenanced["pair_digest"] == reference["pair_digest"]
    assert reprovenanced["target_digest"] != reference["target_digest"], "provenance is evidence"
    mirrored = pool_from([swapped_orientation(contrast)])["pairs"][0]
    assert mirrored["target_digest"] == reference["target_digest"], "orientation moved the target"


def test_identical_duplicate_contrasts_are_deduplicated_and_counted_in_any_order():
    contrast = single_contrast()
    twin = dataclasses.replace(contrast)
    swapped = swapped_orientation(contrast)
    single = pool_from([contrast])
    for ordering in ([contrast, twin], [twin, contrast], [contrast, swapped], [swapped, contrast],
                     [contrast, twin, swapped]):
        result = pool_from(ordering)
        assert len(result["pairs"]) == 1
        assert result["counts"]["excluded"] == {"duplicate": len(ordering) - 1}
        assert result["counts"]["pool_sha256"] == single["counts"]["pool_sha256"]
        assert result["counts"]["targets_sha256"] == single["counts"]["targets_sha256"]
        assert result["pairs"][0]["target_digest"] == single["pairs"][0]["target_digest"]


def test_conflicting_duplicate_contrasts_refuse_the_pool_whichever_came_first():
    contrast = single_contrast()
    left_value, right_value = float(contrast.left.exact_value), float(contrast.right.exact_value)
    variants = {
        "a different value": dataclasses.replace(
            contrast, right=with_value(contrast.right, right_value + 0.25)),
        "a flipped winner": dataclasses.replace(
            contrast, right=with_value(contrast.right, left_value + (left_value - right_value))),
        "a measured tie": dataclasses.replace(contrast, right=with_value(contrast.right,
                                                                         left_value)),
        "different provenance": dataclasses.replace(
            contrast, right=dataclasses.replace(contrast.right, study="a-different-study")),
    }
    for label, other in variants.items():
        messages = []
        for ordering in ([contrast, other], [other, contrast]):
            with pytest.raises(ValueError, match="conflicting ways") as caught:
                pool_from(ordering)
            messages.append(str(caught.value))
        assert messages[0] == messages[1], f"{label}: the refusal depended on enumeration order"
        assert "pair:" in messages[0] and "resolve the duplicated measurements" in messages[0].lower()
    # a tie that stands alone is still counted and enrolled nowhere; only a CONFLICT refuses
    tie_only = pool_from([variants["a measured tie"]])
    assert tie_only["pairs"] == () and tie_only["counts"]["ties"] == 1


def test_imported_pairs_must_back_up_their_margin_values_context_and_digests():
    pairs = pool_of(one_site_rows())["pairs"]
    pair = pairs[0]
    assert pairs_lib.require_pair_metadata(pairs) == 5
    assert pairs_lib.require_pair_metadata([dict(pair, margin=pair["margin"] + 1e-12)]) == 1
    censored = dataclasses.replace(
        pair["rejected"], measurements=(replicate(7.0, lower=None, upper=7.0,
                                                  censoring=CENSOR_LEFT),))
    wrong = {
        "a margin that is not the measured difference":
            (dict(pair, margin=pair["margin"] + 0.5), "not chosen_value - rejected_value"),
        "a NaN margin": (dict(pair, margin=float("nan")), "finite number"),
        "an infinite margin": (dict(pair, margin=float("inf")), "finite number"),
        "a string margin": (dict(pair, margin=str(pair["margin"])), "finite number"),
        "a boolean margin": (dict(pair, margin=True), "finite number"),
        "a non-finite value": (dict(pair, chosen_value=float("inf")), "finite number"),
        "a tie": (dict(pair, rejected_value=pair["chosen_value"], margin=0.0), "strictly higher"),
        "an inverted pair": (dict(pair, chosen_value=pair["rejected_value"],
                                  rejected_value=pair["chosen_value"], margin=-pair["margin"]),
                             "strictly higher"),
        "a value its endpoint never measured":
            (dict(pair, chosen_value=pair["chosen_value"] + 1.0, margin=pair["margin"] + 1.0),
             "exact measured value"),
        "a context digest that is not its own": (dict(pair, background="f" * 64), "context digest"),
        "another pair's identity": (dict(pair, pair_digest=pairs[1]["pair_digest"]), "pair_digest"),
        "another pair's measured target":
            (dict(pair, target_digest=pairs[1]["target_digest"]), "target_digest"),
        "an assay its endpoints were not measured under":
            (dict(pair, assay="other:affinity:T2"), "was measured as assay"),
        "an endpoint that is only a stub":
            (dict(pair, chosen=types.SimpleNamespace(construct=pair["chosen"].construct)),
             "not a measured construct summary"),
        "a censored endpoint": (dict(pair, rejected=censored), "not an exact pKD measurement"),
        "a non-training endpoint": (
            dict(pair, chosen=dataclasses.replace(pair["chosen"],
                                                  partition=data_lib.PARTITION_DEVELOPMENT)),
            "not a training-partition row"),
    }
    for label, (broken, message) in wrong.items():
        with pytest.raises(ValueError, match=message):
            pairs_lib.require_pair_metadata([broken])


def test_the_t4_margin_rule_is_not_imposed_on_background_permuted_contrasts():
    """``permute_background_effects`` moves effect bundles between backgrounds on purpose, so a
    permuted contrast's stored margin disagrees with its own endpoints. T4 reads the endpoints."""
    rows = []
    for k in range(4):
        background = mutate((30 + k, "W"))
        rows.append(observation(background, value=8.0 + 0.1 * k))
        rows.append(observation(mutate((3, "M"), heavy=background), value=7.0))
    _summaries, block = contrasts_of(rows)
    contrasts = tuple(block["contrasts"])
    permuted = None
    for seed in range(32):
        candidate, record = controls.permute_background_effects(contrasts, seed=seed)
        if record["effect_bundles_moved"]:
            permuted = candidate
            break
    assert permuted is not None, "no seed moved an effect bundle"
    violators = [c for c in permuted if abs(float(c.margin) - (
        float(c.left.exact_value) - float(c.right.exact_value))) > 1e-9]
    assert violators, "the permutation left every margin equal to its endpoints' difference"
    direct, through_permuted = pool_from(contrasts), pool_from(permuted)
    assert through_permuted["counts"]["targets_sha256"] == direct["counts"]["targets_sha256"]
    assert pairs_lib.require_pair_metadata(through_permuted["pairs"]) == len(direct["pairs"]) == 4


def test_run_posttrain_refuses_an_unbacked_pair_before_loading_any_model(harness, monkeypatch):
    services, context = harness.services, harness.context
    produce_bakeoff_best(harness, ROUTE_LORA)
    complete_stage(context, "route_gate", winner=ROUTE_LORA)
    tampered = [dict(pair, margin=99.0) for pair in make_pairs(services)]
    monkeypatch.setattr(services, "measured_pairs", lambda: tampered)
    monkeypatch.setattr(services, "load_for_scoring", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("a model was loaded before the pairs were checked")))
    with pytest.raises(ValueError, match="not chosen_value - rejected_value"):
        services.run_posttrain()


# -- the bank: background-first draws ------------------------------------------------------------

def bank_pairs(layout):
    """Synthetic pair records from ``{background: [substitution, ...]}``, one pair per entry."""
    pairs = []
    for background, substitutions in layout.items():
        for index, substitution in enumerate(substitutions):
            pairs.append({"id": f"{background}/{substitution}/{index}", "background": background,
                          "substitution": substitution,
                          "pair_digest": f"{background}|{substitution}|{index}",
                          "target_digest": f"t|{background}|{substitution}|{index}"})
    return tuple(pairs)


#: One crowded background and three sparse ones: 16 pairs, 12 of them in B0.
UNEVEN = {"B0": [f"S{i}" for i in range(12)], "B1": ["S0", "S1"], "B2": ["S0"], "B3": ["S1"]}


def test_a_draw_is_background_first_not_uniform_over_pairs():
    pairs = bank_pairs(UNEVEN)
    for seed in range(25):
        bank = posttrain.PairBank(pairs=pairs, reuse_cap=3)
        drawn, record = bank.draw(4, seed=seed)
        assert sorted(pair["background"] for pair in drawn) == ["B0", "B1", "B2", "B3"], seed
        assert record["backgrounds_available"] == 4 and record["backgrounds_drawn"] == 4
        assert record["max_pairs_from_one_background"] == 1
        assert record["substitution_cells_available"] == 16
    # six pairs: the two single-pair backgrounds are used up after the first round, so the
    # second round goes to the two backgrounds that still hold pairs
    bank = posttrain.PairBank(pairs=pairs, reuse_cap=3)
    drawn, record = bank.draw(6, seed=0)
    tally = {name: sum(1 for pair in drawn if pair["background"] == name)
             for name in ("B0", "B1", "B2", "B3")}
    assert tally == {"B0": 2, "B1": 2, "B2": 1, "B3": 1}
    assert record["max_pairs_from_one_background"] == 2 and record["drawn"] == 6
    assert record["pair_digests"] == [pair["pair_digest"] for pair in drawn]
    assert record["target_digests"] == [pair["target_digest"] for pair in drawn]


def test_the_substitution_marginal_is_balanced_within_a_background_but_not_across_the_pool():
    """The documented limit, measured: substitution X is in every background, Y only in B0."""
    pairs = bank_pairs({"B0": ["X", "Y"], "B1": ["X"], "B2": ["X"], "B3": ["X"]})
    x_total = y_total = y_in_b0 = 0
    for seed in range(40):
        bank = posttrain.PairBank(pairs=pairs, reuse_cap=1000)
        drawn, _record = bank.draw(4, seed=seed)
        x_total += sum(1 for pair in drawn if pair["substitution"] == "X")
        y_total += sum(1 for pair in drawn if pair["substitution"] == "Y")
        y_in_b0 += sum(1 for pair in drawn if pair["background"] == "B0"
                       and pair["substitution"] == "Y")
    assert x_total >= 120 and y_total <= 40, (
        "every draw takes one pair per background, so X is drawn at least three times in four")
    assert 5 <= y_in_b0 <= 35, "inside B0 the two substitutions should alternate roughly evenly"
    rule = posttrain.PAIR_SAMPLING_RULE
    assert "NOT BALANCED" in rule and "substitution marginal across the pool" in rule
    assert posttrain.design(methods=["local_ipo"])["pair_sampling"] == rule


def test_reuse_caps_and_exhaustion_are_honoured_by_background_first_draws():
    pairs = bank_pairs(UNEVEN)
    bank = posttrain.PairBank(pairs=pairs, reuse_cap=2)
    taken, partial = [], None
    while True:
        try:
            drawn, record = bank.draw(5, seed=len(taken))
        except ValueError as error:
            assert "reuse cap" in str(error)
            break
        digests = [pair["pair_digest"] for pair in drawn]
        assert len(set(digests)) == len(digests), "a pair appeared twice in one draw"
        taken.extend(digests)
        if record["exhausted"]:
            partial = record
    assert len(taken) == 2 * len(pairs) == 32
    assert set(taken) == {pair["pair_digest"] for pair in pairs}
    assert max(taken.count(digest) for digest in set(taken)) == 2
    assert partial is not None and partial["drawn"] < partial["requested"]
    document = bank.document()
    assert document["pairs_at_cap"] == 16 and document["pairs_available"] == 0
    assert document["draws_taken"] == 32


def test_the_same_seed_pool_and_caps_give_every_objective_the_same_draws():
    pairs = bank_pairs(UNEVEN)

    def schedule(pool, seeds):
        bank = posttrain.PairBank(pairs=pool, reuse_cap=2)
        return [[pair["pair_digest"] for pair in bank.draw(5, seed=seed)[0]] for seed in seeds]
    first = schedule(pairs, range(100, 106))
    assert first == schedule(pairs, range(100, 106))
    assert first == schedule(tuple(reversed(pairs)), range(100, 106)), (
        "enumeration order changed the draws")
    assert schedule(pairs, range(200, 206)) != first, "the seed does not matter"


def test_difficulty_orders_candidates_inside_a_cell_but_never_overrides_background_first():
    pairs = bank_pairs({"B0": ["S"] * 3 + ["T"] * 7, "B1": ["S"]})
    difficulty = {position: (100.0 if pair["background"] == "B0" else 0.0)
                  for position, pair in enumerate(pairs)}
    bank = posttrain.PairBank(pairs=pairs, reuse_cap=1)
    drawn, _record = bank.draw(2, seed=3, difficulty=difficulty)
    assert {pair["background"] for pair in drawn} == {"B0", "B1"}, (
        "high difficulty concentrated the draw on one background")
    cell = bank_pairs({"B0": ["S", "S", "S", "S"]})
    scores = {0: 1.0, 1: 5.0, 2: 3.0, 3: 5.0}
    bank = posttrain.PairBank(pairs=cell, reuse_cap=1)
    order = [bank.draw(1, seed=0, difficulty=scores)[0][0]["pair_digest"] for _ in range(4)]
    assert order == ["B0|S|1", "B0|S|3", "B0|S|2", "B0|S|0"], (
        "highest difficulty first, ties in canonical (digest) order, each pair once under cap 1")


def test_the_bank_records_exposure_and_states_which_marginal_it_balances():
    bank = posttrain.PairBank(pairs=bank_pairs(UNEVEN), reuse_cap=3)
    for step in range(5):
        bank.draw(4, seed=step)
    document = bank.document()
    exposure = document["exposure"]
    assert document["draws_taken"] == 20
    assert exposure["backgrounds"]["in_pool"] == 4 and exposure["backgrounds"]["touched"] == 4
    assert exposure["backgrounds"]["mean_draws"] * 4 == 20
    assert exposure["backgrounds"]["min_draws"] >= 1
    assert exposure["substitutions"]["in_pool"] == 12      # S0..S11 by label across backgrounds
    assert "background" in exposure["balanced_marginal"]
    assert "substitution marginal across the pool" in exposure["not_balanced"]
    assert document["sampling"] == posttrain.PAIR_SAMPLING_RULE


# -- objectives: the corrected claim -------------------------------------------------------------

def test_the_weighted_nll_docstring_claims_a_shared_optimum_not_the_same_objective():
    doc = posttrain.weighted_conditional_nll_loss.__doc__
    assert "same objective as :func:`margin_match_loss` again" not in doc
    assert "OPTIMUM" in doc and "not the same objective" in doc
    kappa, delta = 2.0, 1.5
    target = kappa * delta
    soft = torch.sigmoid(torch.tensor(target, dtype=torch.float64))

    def gradients(offset):
        odds = torch.tensor(target + offset, dtype=torch.float64, requires_grad=True)
        logistic = -(soft * torch.nn.functional.logsigmoid(odds)
                     + (1.0 - soft) * torch.nn.functional.logsigmoid(-odds))
        zero = torch.zeros(1, dtype=torch.float64)
        margin_loss, _ = posttrain.margin_match_loss(odds.reshape(1), zero, [delta], kappa=kappa)
        return (float(torch.autograd.grad(logistic, odds)[0]),
                float(torch.autograd.grad(margin_loss, odds)[0]))
    at_optimum = gradients(0.0)
    assert abs(at_optimum[0]) < 1e-12 and abs(at_optimum[1]) < 1e-12, "the optimum is shared"
    logistic, squared = gradients(1.0)
    assert squared == pytest.approx(2.0) and logistic == pytest.approx(
        float(torch.sigmoid(torch.tensor(target + 1.0)) - soft), abs=1e-6)
    assert abs(logistic - squared) > 1.0, "the gradients are different, so the objectives are"


# -- identity and metadata through the real stage -----------------------------------------------

def test_the_training_targets_digest_distinguishes_every_component():
    base = dict(pair_pool_sha256="a" * 64, pair_targets_sha256="b" * 64,
                draw_schedule_sha256="c" * 64, target_schedule_sha256="d" * 64)
    reference = posttrain.training_targets_digest(**base)
    assert posttrain.training_targets_digest(**base) == reference
    for key in base:
        assert posttrain.training_targets_digest(**{**base, key: "e" * 64}) != reference, key


def test_changed_training_targets_change_the_checkpoint_identity_and_the_draw_evidence(
        harness, monkeypatch):
    services, context = harness.services, harness.context
    produce_bakeoff_best(harness, ROUTE_LORA)
    complete_stage(context, "route_gate", winner=ROUTE_LORA)
    monkeypatch.setattr(services, "development_panel_outcomes", lambda: PANEL)

    def run(**values):
        monkeypatch.setattr(services, "measured_pairs", lambda: make_pairs(services, **values))
        result = services.run_posttrain()
        payloads = {method: torch.load(str(context.path("posttrain", method, "last.pt")),
                                       map_location="cpu", weights_only=False)
                    for method in result["methods"]}
        return result, payloads
    first, first_payloads = run()
    second, second_payloads = run(chosen_value=9.0)
    # the structural pool and the drawn pairs are unchanged ...
    assert first["pairs"]["pool_sha256"] == second["pairs"]["pool_sha256"]
    assert (first["matched_draws"]["draw_schedule_sha256"]
            == second["matched_draws"]["draw_schedule_sha256"])
    # ... the measured targets are not
    assert first["pairs"]["targets_sha256"] != second["pairs"]["targets_sha256"]
    assert first["matched_draws"]["identical_across_arms"] is True
    assert len({block["target_schedule_sha256"] for block in first["methods"].values()}) == 1
    for method, block in first["methods"].items():
        other = second["methods"][method]
        assert block["pair_targets_sha256"] == first["pairs"]["targets_sha256"]
        assert block["target_schedule_sha256"] != other["target_schedule_sha256"]
        assert block["training_targets_sha256"] != other["training_targets_sha256"]
        record = block["draw_records"][0]
        assert len(record["target_digests"]) == len(record["pair_digests"]) == record["drawn"]
        identity, other_identity = (first_payloads[method]["identity"],
                                    second_payloads[method]["identity"])
        assert (identity["schedule_sha256"] == block["training_targets_sha256"]
                == first_payloads[method]["posttrain"]["training_targets_sha256"])
        assert identity["schedule_sha256"] != other_identity["schedule_sha256"]
        assert identity["identity_sha256"] != other_identity["identity_sha256"]
        with pytest.raises(ValueError, match="schedule_sha256"):
            checkpoints.require_identity(identity, other_identity, path="posttrain")


def test_the_design_metadata_quotes_the_configured_winner_likelihood(harness, monkeypatch):
    services, context = harness.services, harness.context
    produce_bakeoff_best(harness, ROUTE_LORA)
    complete_stage(context, "route_gate", winner=ROUTE_LORA)
    monkeypatch.setattr(services, "measured_pairs", lambda: make_pairs(services))
    monkeypatch.setattr(services, "development_panel_outcomes", lambda: PANEL)
    harness.config["posttrain"]["winner_likelihood"] = 0.37
    result = services.run_posttrain()
    ipo, nll = result["methods"]["local_ipo"], result["methods"]["weighted_conditional_nll"]
    assert ipo["design"]["winner_likelihood_coefficient"] == pytest.approx(0.37)
    assert ipo["history"] and all(
        entry["winner_likelihood_coefficient"] == pytest.approx(0.37) for entry in ipo["history"]
    ), "the executed coefficient is not the one the metadata quotes"
    assert ipo["design"]["winner_likelihood_applied_to"] == ["local_ipo"]
    assert nll["design"]["winner_likelihood_coefficient"] == pytest.approx(0.37)
    assert nll["design"]["winner_likelihood_applied_to"] == [], "the likelihood arm never uses it"
    saved = torch.load(str(context.path("posttrain", "local_ipo", "last.pt")),
                       map_location="cpu", weights_only=False)
    assert saved["posttrain"]["winner_likelihood"] == pytest.approx(0.37)
    # absent from the config the executed value is zero and the metadata says so, not 0.1
    del harness.config["posttrain"]["winner_likelihood"]
    result = services.run_posttrain()
    ipo = result["methods"]["local_ipo"]
    assert ipo["design"]["winner_likelihood_coefficient"] == 0.0
    assert all("winner_likelihood_coefficient" not in entry for entry in ipo["history"])
