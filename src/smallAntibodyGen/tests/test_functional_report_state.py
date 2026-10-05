"""Measurement reports, acquired subsets and the contrast cache (turn 13 repairs).

Each test drives the real function with the input that broke it and asserts behaviour. The data are
REAL ``Observation`` records collapsed by the real ``summarise_observations`` (or real
``ConstructSummary`` objects), not arbitrary stubs, so the keys, replicate lists and region tuples
are the ones production passes around. Services objects are small ``SimpleNamespace`` stand-ins
that carry only the methods the function under test calls; nothing trains, loads a model or
touches CUDA, and no test requires a model to win or a biological gain.

* **acquired units (E13)** -- an acquired subset is selected by the full population key
  ``(identity, study, assay, target)``, never by identity alone, so choosing one assay does not
  admit the other assay or target measured on the same construct; the contrasts enumerated over
  it have only acquired endpoints; the accounting reports population keys, constructs, parents
  and replicates separately and takes presentations from the ledger;
* **contrast cache** -- the cache answers an identical request and rebuilds for a different pool,
  label, replicate, long sequence, site coordinate, parent, cap or seed, whatever the caller's
  key says;
* **E15** -- controls are SEEN-TRAINING rows of the held-out row's own target/assay/readout/scale
  stratum, selected without reading an outcome or the row order, and a stratum without one has an
  explicit absent comparison, not a substitute;
* **E16** -- a role comes from the scale, joins live inside one declared or uniquely inferred
  assay pair, ambiguity is refused or reported, and the two readouts have their own tolerances.
"""
from __future__ import annotations

import random
import types

import pytest

from smallAntibodyGen.experiments import her2_routes_data as data_lib
from smallAntibodyGen.experiments import her2_routes_report as report_lib
from smallAntibodyGen.experiments import her2_routes_services as services_mod
from smallAntibodyGen.experiments.her2_routes_serial import HEAVY as CHAIN_H, Construct
from smallAntibodyGen.experiments.her2_routes_sources import (
    CENSOR_NONE,
    KIND_ENRICHMENT,
    KIND_INTERVAL,
    KIND_POINT,
    SCALE_ENRICHMENT,
    SCALE_EXPRESSION,
    SCALE_PKD,
    Observation,
)

BASE = "A" * 40
LIGHT = "CC"
FLIGHT = services_mod.FlightServices


# ---------------------------------------------------------------------------
# builders: real Observation records -> real summaries
# ---------------------------------------------------------------------------

def observation(residue, value, *, assay="a1", target="T", parent="parent", replicate=None,
                sites=((CHAIN_H, 34),), heavy=None, source="synthetic", scale=SCALE_PKD,
                family="affinity", kind=KIND_POINT, measurement_id=None):
    heavy = heavy or (BASE[:34] + residue + BASE[35:])
    raw = {} if measurement_id is None else {"measurement_id": measurement_id}
    return Observation(
        construct=Construct(heavy=heavy, light=LIGHT, sites=tuple(sites), source=source),
        source=source, family=family, kind=kind, target=target, assay=assay, parent=parent,
        scale=scale, value=float(value), replicate=replicate, raw=raw)


def summarise(observations):
    return list(data_lib.summarise_observations(observations))


RESIDUES = "CDEFGHIKLM"


def library(count, *, assay="a1", target="T", replicates=1, base_value=1.0, **extra):
    """``count`` single-site variants of one parent, each measured ``replicates`` times."""
    out = []
    for index in range(count):
        for replicate in range(replicates):
            out.append(observation(
                RESIDUES[index], base_value + index + 0.01 * replicate, assay=assay,
                target=target, replicate=replicate + 1,
                measurement_id=f"{assay}:{target}:{index}:{replicate}", **extra))
    return out


# ---------------------------------------------------------------------------
# E13: an acquired unit is the full population key
# ---------------------------------------------------------------------------

def test_a_summary_is_one_full_population_key_and_keeps_every_replicate():
    rows = summarise(library(3, replicates=4))
    assert len(rows) == 3
    for row in rows:
        identity, study, assay, target = row.population_key
        assert identity == f"{row.construct.heavy}|{row.construct.light}"
        assert (study, target) == ("synthetic", "T") and assay == "synthetic:affinity:T:a1"
        assert len(row.measurements) == 4


def subset_services(pool):
    """A services stand-in for ``subset_stream``: the real method, a recording ``curriculum_stream``."""
    holder = types.SimpleNamespace()
    holder.training_pools = lambda: {"summaries": {
        data_lib.FAMILY_HER2: (), data_lib.FAMILY_OTHER: tuple(pool),
        data_lib.FAMILY_REPLAY: ()}}
    holder.requests = []

    def curriculum_stream(route, **kwargs):
        holder.requests.append(kwargs)
        return kwargs["permutation"]
    holder.curriculum_stream = curriculum_stream
    return holder


def restriction(pool, subset):
    holder = subset_services(pool)
    restrict = FLIGHT.subset_stream(holder, "route", subset, seed=3, updates=2)
    return restrict, holder


def two_assays_two_targets():
    """The SAME five constructs measured in two assays of one target, and in a second target."""
    observations = (library(5, assay="a1", replicates=3) + library(5, assay="a2", replicates=3)
                    + library(5, assay="a1", target="T2", replicates=3))
    return summarise(observations)


def test_selecting_one_assay_does_not_admit_the_other_assay_or_target_of_the_same_construct():
    pool = two_assays_two_targets()
    assert len(pool) == 15 and len({row.identity for row in pool}) == 5
    acquired = [row for row in pool if row.assay.endswith(":a1") and row.target == "T"]
    assert len(acquired) == 5
    legacy = [row for row in pool if row.identity in {item.identity for item in acquired}]
    assert len(legacy) == 15, "identity alone admits every assay and target on these constructs"
    restrict, holder = restriction(pool, acquired)
    kept = restrict(list(pool))
    assert sorted(row.population_key for row in kept) == sorted(row.population_key
                                                                for row in acquired)
    assert {(row.assay, row.target) for row in kept} == {("synthetic:affinity:T:a1", "T")}
    assert holder.requests[0]["seed"] == 3 and holder.requests[0]["updates"] == 2


def test_every_replicate_of_an_acquired_summary_travels_with_it():
    pool = two_assays_two_targets()
    acquired = [row for row in pool if row.assay.endswith(":a2")][:2]
    restrict, _ = restriction(pool, acquired)
    kept = restrict(list(pool))
    assert len(kept) == 2 and all(len(row.measurements) == 3 for row in kept)
    assert all(row in acquired for row in kept), "the very summaries, with every replicate"


def test_an_acquired_subset_is_validated_before_a_stream_exists():
    pool = two_assays_two_targets()
    one = pool[0]
    with pytest.raises(ValueError, match="twice"):
        restriction(pool, [one, one])
    with pytest.raises(ValueError, match="at least one"):
        restriction(pool, [])
    with pytest.raises(ValueError, match="four non-empty strings"):
        restriction(pool, [types.SimpleNamespace(population_key=("", "s", "a", "t"))])
    with pytest.raises(ValueError, match="four non-empty strings"):
        restriction(pool, [types.SimpleNamespace(population_key=("i", "s", "a"))])
    foreign = summarise(library(1, assay="never-enrolled"))
    with pytest.raises(ValueError, match="not in the measured training pools"):
        restriction(pool, [one, foreign[0]])


def test_contrasts_over_an_acquired_subset_have_only_acquired_endpoints():
    """Integration: the real restriction, then the real enumeration. Local contrasts stay inside
    the acquired population: no endpoint belongs to another assay or target of a chosen construct."""
    pool = two_assays_two_targets()
    acquired = [row for row in pool if row.assay.endswith(":a1") and row.target == "T"]
    restrict, _ = restriction(pool, acquired)
    kept = restrict(list(pool))
    holder = types.SimpleNamespace(
        _cache={}, config={"curriculum": {"max_contrast_sites": 2}},
        parent_sequences=lambda: {"parent": (BASE, LIGHT)})
    contrasts = FLIGHT.contrast_pool(holder, kept, seed=11, cache_key=("other_functional", 11,
                                                                       "full", True, len(kept)))
    assert contrasts, "five variants of one parent have measured single-site neighbours"
    allowed = {row.population_key for row in acquired}
    for contrast in contrasts:
        for endpoint in contrast.alternatives():
            assert endpoint.population_key in allowed
            assert (endpoint.assay, endpoint.target) == ("synthetic:affinity:T:a1", "T")
    # the legacy identity selection would have mixed assays and targets into the enumeration
    legacy = FLIGHT.contrast_pool(
        types.SimpleNamespace(_cache={}, config={"curriculum": {"max_contrast_sites": 2}},
                              parent_sequences=lambda: {"parent": (BASE, LIGHT)}),
        [row for row in pool if row.identity in {item.identity for item in acquired}], seed=11)
    assert {(item.assay, item.target) for contrast in legacy
            for item in contrast.alternatives()} > {("synthetic:affinity:T:a1", "T")}


# ---------------------------------------------------------------------------
# the contrast cache is decided from the inputs, not from the caller's label
# ---------------------------------------------------------------------------

KEY = ("other_functional", 7, "full", True, 2)


class BuilderSpy:
    """Stands in for ``build_local_contrasts`` and counts how often it was asked."""

    def __init__(self):
        self.calls = []

    def __call__(self, rows, **kwargs):
        self.calls.append({"rows": list(rows), "kwargs": dict(kwargs)})
        return {"contrasts": (len(self.calls),), "groups": 1,
                "by_kind": {data_lib.CONTRAST_COUPLED: 0}}


@pytest.fixture
def spy(monkeypatch):
    recorder = BuilderSpy()
    monkeypatch.setattr(data_lib, "build_local_contrasts", recorder)
    return recorder


def cache_services(parents=None, **curriculum):
    holder = types.SimpleNamespace(
        _cache={}, config={"curriculum": {"max_contrast_sites": 2, **curriculum}},
        parents=dict({"parent": (BASE, LIGHT)} if parents is None else parents))
    holder.parent_sequences = lambda: holder.parents
    return holder


def pool(holder, rows, *, seed=7, key=KEY):
    return FLIGHT.contrast_pool(holder, rows, seed=seed, cache_key=key)


def four():
    return summarise(library(4))


def test_an_unchanged_request_is_answered_from_the_cache(spy):
    holder, rows = cache_services(), four()
    first = pool(holder, rows[:2])
    assert pool(holder, rows[:2]) == first and len(spy.calls) == 1
    again = summarise(library(4))[:2]            # equal content, different objects
    assert pool(holder, again) == first and len(spy.calls) == 1


def test_a_same_count_different_subset_is_rebuilt_not_handed_the_first_subsets_endpoints(spy):
    """The probe-13 defect: first/same/different built 1/1/1 where the right answer is 1/1/2."""
    holder, rows = cache_services(), four()
    builds = []
    for chosen in (rows[:2], rows[:2], rows[2:]):
        pool(holder, chosen)
        builds.append(len(spy.calls))
    assert builds == [1, 1, 2]
    assert [call["rows"] for call in spy.calls] == [rows[:2], rows[2:]]


def eight_replicates(*, last=None, swap=False):
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    if last is not None:
        values[-1] = last
    if swap:
        values[6], values[7] = values[7], values[6]
    return summarise([observation("C", value, replicate=index + 1, measurement_id=f"r{index}")
                      for index, value in enumerate(values)]
                     + [observation("D", 9.0, replicate=1, measurement_id="d")])


def test_a_label_change_beyond_the_sixth_replicate_rebuilds(spy):
    holder = cache_services()
    base, changed, swapped = eight_replicates(), eight_replicates(last=80.0), eight_replicates(
        swap=True)
    assert len(base[0].measurements) == 8
    # ConstructSummary.document() keeps six replicates, so it cannot see the seventh or eighth:
    assert base[0].document()["replicates"] == swapped[0].document()["replicates"]
    pool(holder, base)
    pool(holder, base)
    assert len(spy.calls) == 1
    pool(holder, changed)
    assert len(spy.calls) == 2, "the eighth replicate's value changed"
    pool(holder, swapped)
    assert len(spy.calls) == 3, "replicates 7 and 8 swapped: the mean is unchanged, the data is not"


def test_a_label_change_on_a_single_replicate_rebuilds(spy):
    holder = cache_services()
    rows = four()
    relabelled = summarise([observation(RESIDUES[i], (9.0 if i == 0 else 1.0 + i),
                                        replicate=1, measurement_id=f"a1:T:{i}:0")
                            for i in range(4)])
    pool(holder, rows)
    pool(holder, relabelled)
    assert len(spy.calls) == 2, "same sequences and key, different margins, must not reuse them"


def test_a_sequence_change_beyond_the_first_32_characters_rebuilds(spy):
    holder = cache_services()
    one = summarise([observation("C", 1.0, heavy=BASE[:38] + "C" + BASE[39:]),
                     observation("D", 2.0)])
    other = summarise([observation("C", 1.0, heavy=BASE[:38] + "D" + BASE[39:]),
                       observation("D", 2.0)])
    assert one[0].identity[:32] == other[0].identity[:32] and one[0].identity != other[0].identity
    assert one[0].document()["identity"] == other[0].document()["identity"], (
        "document() truncates the identity at 32 characters and cannot tell these apart")
    pool(holder, one)
    pool(holder, other)
    assert len(spy.calls) == 2


def test_a_site_coordinate_change_rebuilds(spy):
    holder = cache_services()
    here = summarise([observation("C", 1.0, sites=((CHAIN_H, 34),)),
                      observation("D", 2.0, sites=((CHAIN_H, 34),))])
    there = summarise([observation("C", 1.0, sites=((CHAIN_H, 35),)),
                       observation("D", 2.0, sites=((CHAIN_H, 35),))])
    assert here[0].document()["region_sites"] == there[0].document()["region_sites"] == 1, (
        "document() reduces the sites to a count, so these two summaries share a document")
    pool(holder, here)
    pool(holder, there)
    assert len(spy.calls) == 2


def test_a_parent_sequence_change_rebuilds(spy):
    holder, rows = cache_services(), four()
    pool(holder, rows)
    pool(holder, rows)
    assert len(spy.calls) == 1
    holder.parents = {"parent": (BASE[:10] + "W" + BASE[11:], LIGHT)}
    pool(holder, rows)
    assert len(spy.calls) == 2
    holder.parents = {}
    pool(holder, rows)
    assert len(spy.calls) == 3


@pytest.mark.parametrize("change", [{"max_contrast_sites": 1}, {"max_contrasts_per_background": 3},
                                    {"max_contrasts_per_context": 3},
                                    {"max_quartets_per_background": 3}])
def test_an_enumeration_cap_change_rebuilds(spy, change):
    holder, rows = cache_services(), four()
    pool(holder, rows)
    holder.config["curriculum"].update(change)
    pool(holder, rows)
    assert len(spy.calls) == 2
    assert {name: spy.calls[1]["kwargs"][name] for name in (
        "max_sites", "max_per_background", "max_per_context", "max_quartets_per_background")
        } != {name: spy.calls[0]["kwargs"][name] for name in (
            "max_sites", "max_per_background", "max_per_context", "max_quartets_per_background")}


def test_a_seed_change_rebuilds_and_an_uncached_call_never_populates_the_cache(spy):
    holder, rows = cache_services(), four()
    pool(holder, rows, seed=7)
    pool(holder, rows, seed=8)
    assert len(spy.calls) == 2
    pool(holder, rows, seed=7)
    assert len(spy.calls) == 2, "the first seed's entry is still there"
    pool(holder, rows, seed=9, key=None)
    pool(holder, rows, seed=9, key=None)
    assert len(spy.calls) == 4 and len(holder._cache["contrast_pools"]) == 2


def test_the_pool_is_materialised_once_so_a_generator_is_not_consumed_by_the_digest(spy):
    holder, rows = cache_services(), four()
    first = pool(holder, (row for row in rows))
    assert len(spy.calls[0]["rows"]) == 4, "the builder saw every row, not an exhausted iterator"
    assert pool(holder, list(rows)) == first and len(spy.calls) == 1


def test_the_digest_binds_the_order_the_enumeration_consumes(spy):
    """The enumeration draws one seeded stream group by group in pool order, so the same rows in a
    different order are a different request; the digest does not claim otherwise."""
    holder, rows = cache_services(), four()
    pool(holder, rows)
    pool(holder, list(reversed(rows)))
    assert len(spy.calls) == 2
    assert holder._cache["contrast_inventory"]["input_binding"]["order_sensitive"] is True


def test_the_inventory_carries_the_versioned_binding_and_a_hit_restores_it(spy):
    holder, rows = cache_services(), four()
    pool(holder, rows[:2])
    first = dict(holder._cache["contrast_inventory"]["input_binding"])
    assert first["version"] == services_mod.CONTRAST_INPUT_DIGEST_VERSION
    assert first["records"] == 2 and len(first["sha256"]) == 64
    pool(holder, rows[2:])
    assert holder._cache["contrast_inventory"]["input_binding"]["sha256"] != first["sha256"]
    pool(holder, rows[:2])
    assert len(spy.calls) == 2
    assert holder._cache["contrast_inventory"]["input_binding"] == first, (
        "a hit puts ITS inventory back, so the E8 panel never reads another pool's counts")


def test_the_binding_is_decided_inside_contrast_pool_for_every_caller_key(spy):
    holder, rows = cache_services(), four()
    for key in (KEY, ("E8", 1, 2), ("profile", 5)):
        pool(holder, rows[:2], key=key)
        pool(holder, rows[2:], key=key)
    assert len(spy.calls) == 6, "three caller keys x two different pools of the same size"


def test_e8_rebuilds_for_a_different_same_size_pool_and_reuses_an_identical_one(spy):
    rows = four()
    state = {"eligible": rows[:2]}
    holder = types.SimpleNamespace(
        _cache={}, config={"curriculum": {"max_contrast_sites": 2}, "split": {"seed": 5}},
        eligible_summaries=lambda: state["eligible"],
        parent_sequences=lambda: {"parent": (BASE, LIGHT)})
    holder.contrast_pool = lambda eligible, **kwargs: FLIGHT.contrast_pool(holder, eligible,
                                                                           **kwargs)
    FLIGHT.panel_interaction(holder)
    FLIGHT.panel_interaction(holder)
    assert len(spy.calls) == 1
    state["eligible"] = rows[2:]
    record = FLIGHT.panel_interaction(holder)
    assert len(spy.calls) == 2 and record["evidence_kind"] == "training_data_interaction_diagnostic"


# ---------------------------------------------------------------------------
# E13: honest accounting
# ---------------------------------------------------------------------------

class E13Services:
    """The services ``e13_learning_curve`` calls, recording what it asks for."""

    def __init__(self, tmp_path, rows, *, her2=(), score=None, display_rows=1234):
        self.rows, self.her2 = list(rows), list(her2)
        self.context = types.SimpleNamespace(path=lambda *parts: tmp_path.joinpath(*parts))
        self.display_rows, self.score = display_rows, score or (lambda call: [0, 1, 2, 3, 4, 5])
        self.subsets, self.directories, self.trained = [], [], []
        self._cache = {"bins": {("k", "s", "a", "t"): "high"}}

    def development_panel_outcomes(self):
        return [{"window": f"W{index}", "value": float(index)} for index in range(6)]

    def training_pools(self):
        return {"summaries": {data_lib.FAMILY_HER2: tuple(self.her2),
                              data_lib.FAMILY_OTHER: tuple(self.rows),
                              data_lib.FAMILY_REPLAY: ()},
                "display_rows": self.display_rows,
                "counts": {data_lib.FAMILY_REPLAY: 777}}

    def subset_stream(self, route, subset, *, seed, updates):
        self.subsets.append(list(subset))
        return object()

    def development_metric(self, route):
        return None

    def train(self, route, *, stream, updates, seed, directory, arm, heartbeat, evaluate):
        self.directories.append(directory)
        call = len(self.trained)
        self.trained.append(call)
        return {"updates": updates, "best": {"metric": 0.1, "step": updates},
                "exposure": {"totals": {"measurement_presentations": 1000 + call,
                                        "microbatch_rows": 64},
                             "unique_measured_constructs": 7 + call}}

    def score_panel(self, route, checkpoint, rows):
        return list(self.score(len(self.trained)))


def one_parent_library():
    """Five constructs x two assays x three replicates, ONE parent: ten summaries."""
    return summarise(library(5, assay="a1", replicates=3) + library(5, assay="a2", replicates=3))


def test_the_label_budget_is_population_keys_not_the_parent_count(tmp_path):
    services = E13Services(tmp_path, one_parent_library())
    record = report_lib.e13_learning_curve(services, "route", fractions=(0.2, 0.8, 1.0),
                                           updates=3, seed=4)
    small, large, full = record["curve"]["rows"]
    assert (small["population_keys"], large["population_keys"], full["population_keys"]) == (
        2, 8, 10), "2 versus 8 labels over one parent is 2 versus 8 keys, not 1 versus 1 parent"
    assert small["parents"] == large["parents"] == full["parents"] == 1
    assert small["summary_outcomes"] == 2 and large["summary_outcomes"] == 8
    assert full["distinct_constructs"] == 5, "ten keys over five constructs"
    assert 1 <= small["distinct_constructs"] <= 2
    assert 4 <= large["distinct_constructs"] <= 5, "eight keys over constructs that own two each"
    assert full["replicate_observations_available"] == 30
    assert small["replicate_observations_available"] == 6
    assert len(small["selected_keys_sha256"]) == 64
    assert small["selected_keys_sha256"] != large["selected_keys_sha256"]
    assert record["label_budget_unit"] == "population_key"


def test_repeated_presentations_come_from_the_exposure_ledger_not_the_replicate_count(tmp_path):
    services = E13Services(tmp_path, one_parent_library())
    record = report_lib.e13_learning_curve(services, "route", fractions=(0.2, 0.8), updates=3,
                                           seed=4)
    first, second = record["curve"]["rows"]
    assert first["presentations"]["measurement_presentations"] == 1000
    assert second["presentations"]["measurement_presentations"] == 1001
    assert first["presentations"]["unique_measured_constructs"] == 7
    assert first["presentations"]["measurement_presentations"] != first[
        "replicate_observations_available"]
    assert "ledger" in first["presentations"]["source"]
    assert "measurement_presentations" not in {key for key in first if key != "presentations"}, (
        "the pre-fit replicate count is no longer labelled a presentation count")


def test_a_ledger_that_does_not_carry_a_figure_reports_none_not_zero(tmp_path):
    services = E13Services(tmp_path, one_parent_library())
    services.train = lambda route, **kwargs: {"updates": 3, "best": {}, "exposure": {}}
    record = report_lib.e13_learning_curve(services, "route", fractions=(0.5,), updates=3, seed=4)
    assert record["curve"]["rows"][0]["presentations"]["measurement_presentations"] is None


def test_the_curve_is_labelled_incremental_and_binds_the_fixed_supervision(tmp_path):
    services = E13Services(tmp_path, one_parent_library())
    record = report_lib.e13_learning_curve(services, "route", fractions=(0.2, 1.0), updates=3,
                                           seed=4)
    scope = record["supervision_scope"]
    assert scope["kind"] == "incremental_summary_supervision"
    assert "INCREMENTAL summary supervision" in scope["label"]
    assert "not a clean total-N learning curve" in scope["label"]
    assert scope["thresholds_refit_per_fraction"] is False and scope["clean_total_n_claim"] is None
    assert record["curve"]["scope"] == scope
    fixed = record["fixed_supervision"]
    assert fixed["display_rows"] == 1234 and fixed["display_subset"] is False
    assert fixed["replay_rows"] == 777 and fixed["replay_subset"] is False
    assert fixed["quality_bin_entries"] == 1
    assert fixed["quality_thresholds_refit_per_fraction"] is False
    assert "full training pool" in fixed["quality_thresholds_fit_on"]


def test_selected_families_count_the_her2_and_other_pools_separately(tmp_path):
    rows = one_parent_library()
    services = E13Services(tmp_path, rows[5:], her2=rows[:5])
    record = report_lib.e13_learning_curve(services, "route", fractions=(1.0,), updates=3, seed=4)
    full, = record["curve"]["rows"]
    assert full["selected_by_family"] == {"her2": 5, "other_functional": 5}
    assert full["population_keys"] == 10


def test_a_curve_with_no_defined_score_is_unknown_not_decreasing(tmp_path):
    services = E13Services(tmp_path, one_parent_library(), score=lambda call: [1.0] * 6)
    record = report_lib.e13_learning_curve(services, "route", fractions=(0.2, 0.8, 1.0),
                                           updates=3, seed=4)
    assert [row["score"] for row in record["curve"]["rows"]] == [None, None, None]
    assert record["curve"]["monotone"] is None and record["curve"]["scored_points"] == 0
    assert "unknown" in record["curve"]["monotone_reason"]
    assert record["value"] is None, "a constant scorer has no Spearman; it is not a score of zero"
    diagnosis = report_lib.no_gain_diagnosis(panels={}, controls={}, exposure={},
                                             learning_curve=record["curve"])
    assert not any("did not rise" in item["explanation"] for item in diagnosis["diagnosis"])


def test_output_directories_are_unique_and_keep_the_historical_whole_percent_names(tmp_path):
    services = E13Services(tmp_path, one_parent_library())
    report_lib.e13_learning_curve(services, "route", fractions=(0.2, 0.8, 1.0), updates=3, seed=4)
    assert [path.name for path in services.directories] == ["fraction020", "fraction080",
                                                            "fraction100"]
    assert all(path.parent.name == "e13" for path in services.directories)
    assert report_lib.fraction_directory_name(0.105) != report_lib.fraction_directory_name(0.1)
    assert report_lib.fraction_directory_name(0.001) == "fraction_0p001"


@pytest.mark.parametrize("bad", [[0.0], [-0.5], [1.5], [float("nan")], [float("inf")], [True],
                                 ["0.5"], [None], [], [0.5, 0.5]])
def test_invalid_fractions_are_refused_before_any_fit(tmp_path, bad):
    services = E13Services(tmp_path, one_parent_library())
    with pytest.raises(ValueError):
        report_lib.e13_learning_curve(services, "route", fractions=bad, updates=3, seed=4)
    assert services.trained == [] and services.subsets == []


def test_fractions_whose_directories_or_subsets_coincide_are_refused(tmp_path):
    services = E13Services(tmp_path, one_parent_library())
    with pytest.raises(ValueError, match="collide on their output directory names"):
        report_lib.e13_learning_curve(services, "route", fractions=(0.1, 0.1 + 1e-12), updates=3,
                                      seed=4)
    small = E13Services(tmp_path, summarise(library(3)))
    with pytest.raises(ValueError, match="same number of summaries"):
        report_lib.e13_learning_curve(small, "route", fractions=(0.5, 0.6), updates=3, seed=4)
    assert services.trained == [] and small.trained == []


def test_a_pool_that_names_one_population_key_twice_is_refused(tmp_path):
    rows = one_parent_library()
    services = E13Services(tmp_path, rows + rows[:1])
    with pytest.raises(ValueError, match="population key twice"):
        report_lib.e13_learning_curve(services, "route", fractions=(0.5,), updates=3, seed=4)


# ---------------------------------------------------------------------------
# E15: SEEN-TRAINING controls of the held-out row's own stratum
# ---------------------------------------------------------------------------

def row(index, value, *, target="T", assay=None, study="s", scale=SCALE_PKD, kind=KIND_POINT,
        family="affinity", partition=data_lib.PARTITION_TRAIN, parent="p", region=(("H", 3),)):
    """A real summary whose full identity is fixed by ``index`` (two positions are edited)."""
    pattern = list("ACDEFGHIKLMNPQRSTVWY" * 2)
    for position in (index % 20, 20 + index // 20):
        pattern[position] = "P" if pattern[position] != "P" else "Q"
    construct = Construct(heavy="".join(pattern), light="YWVTSRQPNMLKIHGFEDCA",
                          sites=tuple(region), source="synthetic", record_id=f"r{index}")
    return data_lib.ConstructSummary(
        construct=construct, study=study, assay=assay or f"{study}:{family}:{target}:a",
        target=target, parent=parent, family=family, kind=kind,
        measurements=(data_lib.ReplicateMeasurement(
            value=value, lower=value, upper=value, censoring=CENSOR_NONE, replicate=1,
            measurement_id=f"m{index}"),),
        region=tuple(region), partition=partition, scale=scale, source="synthetic")


def transfer_services(rows, score_of, *, vocabulary=None):
    scored = []

    def summary_scores(route, model, chosen):
        scored.append([item.population_key for item in chosen])
        return [float(score_of[item.population_key]) for item in chosen]
    holder = types.SimpleNamespace(
        summaries=lambda: list(rows),
        withheld_marker=lambda item: None if item.is_training else "study:held",
        load_for_scoring=lambda route, checkpoint: (object(), object(), {"checkpoint": "stub"}),
        summary_scores=summary_scores)
    if vocabulary is not None:
        holder.condition_spec = types.SimpleNamespace(assays=tuple(vocabulary))
    return holder, scored


class Fixture:
    """Rows plus the score each one receives; ``add`` keeps the two aligned."""

    def __init__(self):
        self.rows, self.score, self.groups = [], {}, {}

    def add(self, group, index, value, score, **kwargs):
        item = row(index, value, **kwargs)
        self.rows.append(item)
        self.score[item.population_key] = score
        self.groups.setdefault(group, []).append(item)
        return item

    def keys(self, *groups):
        return {item.population_key for group in groups for item in self.groups[group]}


def root_fixture():
    """Held y [1,2,3] scored [1,3,2] (rho 0.5); the right controls y [4,5,6] scored [4,5,6] (rho 1);
    wrong-target, wrong-assay and wrong-scale controls y [4,5,6] scored in REVERSE (rho -1). The
    wrong ones come FIRST in the row list, so a first-N pick of 'the same kind' takes them."""
    fx = Fixture()
    for offset, (y, s) in enumerate(zip((4, 5, 6), (6, 5, 4))):
        fx.add("wrong_target", 20 + offset, y, s, target="T2")
        fx.add("wrong_assay", 30 + offset, y, s, assay="s:affinity:T:b")
        # the SAME assay string as the held-out rows, but another measured readout and scale
        fx.add("wrong_scale", 40 + offset, y, s, assay="s:affinity:T:a", scale=SCALE_EXPRESSION)
    for offset, (y, s) in enumerate(zip((4, 5, 6), (4, 5, 6))):
        fx.add("right", 10 + offset, y, s)
    for offset, (y, s) in enumerate(zip((1, 2, 3), (1, 3, 2))):
        fx.add("held", offset, y, s, partition=data_lib.PARTITION_DEVELOPMENT)
    return fx


def test_the_root_fixture_gives_minus_one_half_and_never_scores_a_wrong_stratum_row():
    fx = root_fixture()
    holder, scored = transfer_services(fx.rows, fx.score)
    record = report_lib.e15_transfer(holder, "route", "checkpoint")
    block = record["by_heldout"]["study:held"]
    assert block["spearman"] == pytest.approx(0.5)
    assert block["seen_training_control_spearman"] == pytest.approx(1.0)
    assert block["seen_training_control_rows"] == 3
    assert block["delta"] == pytest.approx(-0.5)
    assert record["value"] == pytest.approx(-0.5), "a wrong-assay control would have given +1.5"
    seen = {key for batch in scored for key in batch}
    assert seen == fx.keys("held", "right"), "no wrong-target/assay/scale row was ever scored"
    stratum, = block["strata"]
    assert stratum["stratum"] == {"target": "T", "assay": "s:affinity:T:a", "readout": "affinity",
                                  "scale": SCALE_PKD}
    comparator = stratum["comparator"]
    assert comparator["status"] == "available" and comparator["size_matched"] is True
    assert comparator["compatible_training_rows"] == comparator["training_rows_in_stratum"] == 3
    assert comparator["control_rows"] == comparator["heldout_rows"] == 3
    assert len(comparator["control_ids"]) == 3 and len(comparator["control_ids_sha256"]) == 64
    assert "no outcome and no input order" in comparator["selection"]
    assert record["strata_with_comparator"] == 1 and record["strata_without_comparator"] == 0


def test_the_controls_are_named_seen_training_not_independent_validation():
    fx = root_fixture()
    holder, _ = transfer_services(fx.rows, fx.score)
    record = report_lib.e15_transfer(holder, "route", "checkpoint")
    assert record["control_policy"].startswith("SEEN-TRAINING controls, not an independent")
    assert "in_distribution" not in str(sorted(record["by_heldout"]["study:held"]))
    assert "SEEN-TRAINING" in record["reading"] and "not independent" in record["reading"]
    condition = record["scoring_condition"]
    assert condition["bin"].startswith("high") and "null" in condition["assay"]


def test_a_reordering_of_irrelevant_rows_changes_nothing():
    fx = root_fixture()
    holder, _ = transfer_services(fx.rows, fx.score)
    baseline = report_lib.e15_transfer(holder, "route", "c")
    for ordering in (list(reversed(fx.rows)), random.Random(5).sample(fx.rows, len(fx.rows)),
                     sorted(fx.rows, key=lambda item: item.population_key, reverse=True)):
        holder, _ = transfer_services(ordering, fx.score)
        again = report_lib.e15_transfer(holder, "route", "c")
        assert again["value"] == baseline["value"]
        left, right = again["by_heldout"]["study:held"], baseline["by_heldout"]["study:held"]
        assert left["strata"][0]["comparator"]["control_ids_sha256"] == right["strata"][0][
            "comparator"]["control_ids_sha256"]
        assert left["strata"][0]["comparator"]["control_ids"] == right["strata"][0][
            "comparator"]["control_ids"]


def test_control_selection_reads_no_outcome():
    """Five compatible training rows, three wanted: the same three whatever their outcomes are."""
    def build(values):
        fx = Fixture()
        for offset in range(5):
            fx.add("pool", 10 + offset, values[offset], float(offset))
        for offset in range(3):
            fx.add("held", offset, float(offset), float(offset),
                   partition=data_lib.PARTITION_DEVELOPMENT)
        return fx
    chosen = []
    for values in ((0.0, 1.0, 2.0, 3.0, 4.0), (4.0, 3.0, 2.0, 1.0, 0.0), (9.0, 1.0, 7.0, 1.0, 3.0)):
        fx = build(values)
        holder, scored = transfer_services(fx.rows, fx.score)
        report_lib.e15_transfer(holder, "route", "c")
        controls = [batch for batch in scored if set(batch) <= fx.keys("pool")]
        assert len(controls) == 1 and len(controls[0]) == 3
        chosen.append(sorted(controls[0]))
    assert chosen[0] == chosen[1] == chosen[2]


def test_an_exact_point_from_an_interval_source_matches_the_same_endpoint_semantics():
    fx = Fixture()
    for offset, (y, s) in enumerate(zip((4, 5, 6), (4, 5, 6))):
        fx.add("interval", 10 + offset, y, s, kind=KIND_INTERVAL)
    for offset, (y, s) in enumerate(zip((1, 2, 3), (1, 3, 2))):
        fx.add("held", offset, y, s, partition=data_lib.PARTITION_DEVELOPMENT, kind=KIND_POINT)
    holder, scored = transfer_services(fx.rows, fx.score)
    record = report_lib.e15_transfer(holder, "route", "c")
    block = record["by_heldout"]["study:held"]
    assert block["strata"][0]["heldout_kinds"] == [KIND_POINT]
    assert block["seen_training_control_rows"] == 3 and record["value"] == pytest.approx(-0.5)
    assert {key for batch in scored for key in batch} == fx.keys("held", "interval")


def test_a_withheld_study_with_no_compatible_training_assay_has_no_delta_only_its_own_score():
    fx = Fixture()
    for offset, (y, s) in enumerate(zip((4, 5, 6), (6, 5, 4))):
        fx.add("elsewhere", 10 + offset, y, s, study="other", target="T")
    for offset, (y, s) in enumerate(zip((1, 2, 3), (1, 3, 2))):
        fx.add("held", offset, y, s, partition=data_lib.PARTITION_DEVELOPMENT, study="newstudy")
    holder, scored = transfer_services(fx.rows, fx.score)
    record = report_lib.e15_transfer(holder, "route", "c")
    block = record["by_heldout"]["study:held"]
    assert block["spearman"] == pytest.approx(0.5), "the standalone held-out score is shown"
    assert block["delta"] is None and block["seen_training_control_spearman"] is None
    comparator = block["strata"][0]["comparator"]
    assert comparator["status"] == "absent" and comparator["control_rows"] == 0
    assert "no control is substituted from another stratum" in comparator["reason"]
    assert record["supported"] is False and record["value"] is None
    assert record["population"] == 3 and record["strata_without_comparator"] == 1
    assert "no held-out stratum has a compatible seen-training comparison" in record[
        "blocking_reason"]
    assert {key for batch in scored for key in batch} == fx.keys("held"), (
        "no unrelated training row was scored as a stand-in")
    report_lib.require_no_placeholder(record)


def test_one_compatible_training_row_is_not_a_comparison():
    fx = Fixture()
    fx.add("lone", 10, 5.0, 5.0)
    for offset in range(3):
        fx.add("held", offset, float(offset), float(2 - offset),
               partition=data_lib.PARTITION_DEVELOPMENT)
    holder, _ = transfer_services(fx.rows, fx.score)
    record = report_lib.e15_transfer(holder, "route", "c")
    comparator = record["by_heldout"]["study:held"]["strata"][0]["comparator"]
    assert comparator["status"] == "absent" and comparator["compatible_training_rows"] == 1
    assert record["value"] is None


def test_fewer_controls_than_held_out_rows_is_disclosed_as_not_size_matched():
    fx = Fixture()
    for offset, (y, s) in enumerate(zip((4, 5), (4, 5))):
        fx.add("pool", 10 + offset, y, s)
    for offset, (y, s) in enumerate(zip((1, 2, 3), (1, 3, 2))):
        fx.add("held", offset, y, s, partition=data_lib.PARTITION_DEVELOPMENT)
    holder, _ = transfer_services(fx.rows, fx.score)
    record = report_lib.e15_transfer(holder, "route", "c")
    comparator = record["by_heldout"]["study:held"]["strata"][0]["comparator"]
    assert comparator["status"] == "available" and comparator["size_matched"] is False
    assert comparator["control_rows"] == 2 and comparator["heldout_rows"] == 3
    assert record["value"] == pytest.approx(0.5 - 1.0)


def test_a_held_out_population_spanning_two_strata_is_never_pooled_into_one_spearman():
    fx = Fixture()
    for tag, assay, base in (("a", "s:affinity:T:a", 10), ("b", "s:affinity:T:b", 20)):
        for offset, (y, s) in enumerate(zip((4, 5, 6), (4, 5, 6))):
            fx.add("pool_" + tag, base + offset, y, s, assay=assay)
    for offset, (y, s) in enumerate(zip((1, 2, 3), (1, 3, 2))):
        fx.add("held_a", offset, y, s, partition=data_lib.PARTITION_DEVELOPMENT,
               assay="s:affinity:T:a")
    for offset, (y, s) in enumerate(zip((1, 2, 3), (3, 2, 1))):
        fx.add("held_b", 3 + offset, y, s, partition=data_lib.PARTITION_DEVELOPMENT,
               assay="s:affinity:T:b")
    holder, _ = transfer_services(fx.rows, fx.score)
    record = report_lib.e15_transfer(holder, "route", "c")
    block = record["by_heldout"]["study:held"]
    assert len(block["strata"]) == 2 and block["rows"] == 6
    assert block["spearman"] is None and "no scale" in block["spearman_note"]
    deltas = sorted(item["delta"] for item in block["strata"])
    assert deltas == pytest.approx([-2.0, -0.5]), "rho 0.5-1 in assay a, rho -1-1 in assay b"
    assert record["value"] == pytest.approx(-1.25) and record["strata_with_comparator"] == 2
    assert block["delta"] == pytest.approx(-1.25)


def test_the_assay_conditioning_of_each_stratum_is_disclosed_when_the_vocabulary_is_known():
    fx = root_fixture()
    holder, _ = transfer_services(fx.rows, fx.score, vocabulary=("s:affinity:T:a",))
    record = report_lib.e15_transfer(holder, "route", "c")
    assert record["by_heldout"]["study:held"]["strata"][0]["assay_conditioned_as"] == "own"
    holder, _ = transfer_services(fx.rows, fx.score, vocabulary=("something:else",))
    record = report_lib.e15_transfer(holder, "route", "c")
    assert record["by_heldout"]["study:held"]["strata"][0]["assay_conditioned_as"] == "null"


# ---------------------------------------------------------------------------
# E16: roles from the scale, joins inside one declared assay pair
# ---------------------------------------------------------------------------

AFF, EXP = "s:affinity:T:aff", "s:enrichment:T:exp"


def readouts(affinities, expressions, *, extra=(), study="s", target="T", exp_scale=SCALE_EXPRESSION,
             exp_target=None, exp_study=None):
    rows = [row(i, y, assay=AFF, study=study, target=target) for i, y in enumerate(affinities)]
    rows += [row(i, y, assay=EXP, study=exp_study or study, target=exp_target or target,
                 scale=exp_scale, family="enrichment", kind=KIND_ENRICHMENT)
             for i, y in enumerate(expressions)]
    return list(rows) + list(extra)


def run16(rows, **kwargs):
    return report_lib.e16_readouts(types.SimpleNamespace(summaries=lambda: list(rows)), **kwargs)


def test_the_hand_fixture_has_one_expression_matched_and_one_affinity_matched_pair():
    rows = readouts([1.0, 3.0, 3.05], [0.0, 0.05, 1.0])
    for kwargs in ({"tolerance": 0.1}, {"affinity_tolerance": 0.1, "expression_tolerance": 0.1}):
        record = run16(rows, **kwargs)
        assert record["population"] == 3 and record["supported"] is True
        assert record["expression_matched_pairs"] == 1 and record["affinity_matched_pairs"] == 1
        assert record["value"] == 1.0
        assert record["residual_expression_imbalance"] == pytest.approx(0.05)
        assert record["residual_affinity_imbalance"] == pytest.approx(0.05)
        assert record["affinity_spread_among_expression_matched"] == pytest.approx(2.0)
        stratum, = record["strata"]
        assert stratum["selection"] == "inferred_unique" and stratum["constructs_with_both"] == 3
        assert (stratum["affinity_assay"], stratum["expression_assay"]) == (AFF, EXP)
        assert (stratum["affinity_scale"], stratum["expression_scale"]) == (SCALE_PKD,
                                                                            SCALE_EXPRESSION)


def test_the_join_is_independent_of_row_order():
    rows = readouts([1.0, 3.0, 3.05], [0.0, 0.05, 1.0])
    expected = run16(rows, tolerance=0.1)
    for ordering in (list(reversed(rows)), random.Random(2).sample(rows, len(rows))):
        again = run16(ordering, tolerance=0.1)
        assert again["strata"] == expected["strata"] and again["value"] == expected["value"]


def test_the_tolerances_are_separate_and_the_legacy_one_is_a_recorded_broadcast():
    rows = readouts([1.0, 3.0, 3.05], [0.0, 0.05, 1.0])
    legacy = run16(rows, tolerance=0.1)
    assert legacy["tolerances"]["source"] == "legacy_broadcast"
    assert legacy["tolerances"]["affinity"] == legacy["tolerances"]["expression"] == 0.1
    assert "different units" in legacy["tolerances"]["note"]
    named = run16(rows, affinity_tolerance=0.01, expression_tolerance=0.1)
    assert named["tolerances"]["source"] == "named"
    assert named["tolerances"]["units"] == {"affinity": SCALE_PKD, "expression": SCALE_EXPRESSION}
    assert named["affinity_matched_pairs"] == 0 and named["expression_matched_pairs"] == 1
    assert named["strata"][0]["tolerance_units"] == {"affinity": SCALE_PKD,
                                                     "expression": SCALE_EXPRESSION}


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf"), None, True, "x"])
def test_a_bad_tolerance_is_refused(bad):
    rows = readouts([1.0, 3.0], [0.0, 0.05])
    with pytest.raises(ValueError):
        run16(rows, affinity_tolerance=bad, expression_tolerance=0.1)
    with pytest.raises(ValueError):
        run16(rows, affinity_tolerance=0.1, expression_tolerance=bad)
    with pytest.raises(ValueError):
        run16(rows, tolerance=bad)


def test_the_two_tolerance_forms_cannot_be_mixed_or_half_declared():
    rows = readouts([1.0, 3.0], [0.0, 0.05])
    with pytest.raises(ValueError, match="not both"):
        run16(rows, tolerance=0.1, affinity_tolerance=0.1, expression_tolerance=0.1)
    with pytest.raises(ValueError, match="needs both"):
        run16(rows, affinity_tolerance=0.1)
    with pytest.raises(ValueError, match="needs both"):
        run16(rows)


def test_a_binding_enrichment_is_never_an_expression_readout():
    rows = readouts([1.0, 3.0, 3.05], [0.0, 0.05, 1.0], exp_scale=SCALE_ENRICHMENT)
    record = run16(rows, tolerance=0.1)
    assert record["supported"] is False and record["value"] is None and record["population"] == 0
    assert record["unsupported_roles"] == {"binding_enrichment": 3}
    assert "binding enrichment is never an expression readout" in record["blocking_reason"]
    assert report_lib.readout_role(rows[-1]) == "binding_enrichment"
    assert report_lib.readout_role(rows[0]) == "affinity"


def test_a_binding_enrichment_next_to_a_real_expression_readout_creates_no_ambiguity():
    binding = [row(i, float(i), assay="s:enrichment:T:bind", scale=SCALE_ENRICHMENT,
                   family="enrichment", kind=KIND_ENRICHMENT) for i in range(3)]
    record = run16(readouts([1.0, 3.0, 3.05], [0.0, 0.05, 1.0], extra=binding), tolerance=0.1)
    assert record["supported"] is True and record["stratum_count"] == 1
    assert record["unsupported_roles"] == {"binding_enrichment": 3}
    assert record["strata"][0]["expression_assay"] == EXP


def test_a_cross_target_or_cross_study_join_is_unsupported():
    cross_target = run16(readouts([1.0, 3.0, 3.05], [0.0, 0.05, 1.0], exp_target="T2"),
                         tolerance=0.1)
    assert cross_target["supported"] is False and cross_target["population"] == 0
    assert len(cross_target["groups_with_one_role_only"]) == 2
    cross_study = run16(readouts([1.0, 3.0, 3.05], [0.0, 0.05, 1.0], exp_study="elsewhere"),
                        tolerance=0.1)
    assert cross_study["supported"] is False and cross_study["strata"] == []


def test_several_expression_assays_are_ambiguous_never_chosen_by_position_or_order():
    second = [row(i, y, assay="s:enrichment:T:exp2", scale=SCALE_EXPRESSION, family="enrichment",
                  kind=KIND_ENRICHMENT) for i, y in enumerate([5.0, 5.05, 9.0])]
    rows = readouts([1.0, 3.0, 3.05], [0.0, 0.05, 1.0], extra=second)
    outcomes = []
    for ordering in (rows, list(reversed(rows)), random.Random(9).sample(rows, len(rows))):
        record = run16(ordering, tolerance=0.1)
        assert record["supported"] is False and record["value"] is None
        group, = record["unsupported_groups"]
        assert [entry["assay"] for entry in group["inventory"]["expression"]] == [EXP,
                                                                                  "s:enrichment:T:exp2"]
        assert [entry["assay"] for entry in group["inventory"]["affinity"]] == [AFF]
        assert "never chosen by position" in group["reason"]
        outcomes.append(record["unsupported_groups"])
    assert outcomes[0] == outcomes[1] == outcomes[2], "order cannot select another assay"


def ambiguous_rows():
    second = [row(i, y, assay="s:enrichment:T:exp2", scale=SCALE_EXPRESSION, family="enrichment",
                  kind=KIND_ENRICHMENT) for i, y in enumerate([5.0, 5.05, 9.0])]
    return readouts([1.0, 3.0, 3.05], [0.0, 0.05, 1.0], extra=second)


def pair(expression_assay, **extra):
    return {"study": "s", "target": "T", "affinity_assay": AFF,
            "expression_assay": expression_assay, **extra}


def test_an_explicit_pair_resolves_the_ambiguity_and_its_choice_is_what_is_used():
    rows = ambiguous_rows()
    first = run16(rows, tolerance=0.1, assay_pairs=[pair(EXP)])
    other = run16(list(reversed(rows)), tolerance=0.1, assay_pairs=[pair("s:enrichment:T:exp2")])
    assert first["supported"] is True and first["strata"][0]["selection"] == "explicit"
    assert first["strata"][0]["expression_assay"] == EXP
    assert first["expression_matched_pairs"] == 1
    assert other["strata"][0]["expression_assay"] == "s:enrichment:T:exp2"
    assert other["expression_matched_pairs"] == 1
    assert first["unsupported_groups"] == []


def test_several_explicit_strata_stay_separate_and_are_never_pooled():
    record = run16(ambiguous_rows(), tolerance=0.1,
                   assay_pairs=[pair(EXP), pair("s:enrichment:T:exp2")])
    assert record["stratum_count"] == 2 and record["population"] == 6
    assert [item["expression_assay"] for item in record["strata"]] == [
        EXP, "s:enrichment:T:exp2"]
    assert [item["pairs"] for item in record["strata"]] == [3, 3]
    assert record["value"] is None and "withheld" in record["headline_value"]
    assert all(item["selection"] == "explicit" for item in record["strata"])


def test_a_bad_explicit_selection_is_an_error_not_a_fallback():
    rows = ambiguous_rows()
    with pytest.raises(ValueError, match="does not name exactly one"):
        run16(rows, tolerance=0.1, assay_pairs=[pair("s:enrichment:T:missing")])
    with pytest.raises(ValueError, match="holds no exact affinity or expression summary"):
        run16(rows, tolerance=0.1, assay_pairs=[{**pair(EXP), "target": "nowhere"}])
    with pytest.raises(ValueError, match="declared twice"):
        run16(rows, tolerance=0.1, assay_pairs=[pair(EXP), pair(EXP)])
    with pytest.raises(ValueError, match="four non-empty strings"):
        run16(rows, tolerance=0.1, assay_pairs=[{"study": "s"}])
    with pytest.raises(ValueError, match="four non-empty strings"):
        run16(rows, tolerance=0.1, assay_pairs=[pair("")])


def test_two_summaries_of_one_construct_in_one_role_are_refused():
    duplicate = [row(0, 1.5, assay=AFF)]
    with pytest.raises(ValueError, match="more than one summary of the same full construct"):
        run16(readouts([1.0, 3.0, 3.05], [0.0, 0.05, 1.0], extra=duplicate), tolerance=0.1)


def test_only_exact_scalars_join_and_the_rest_are_counted_by_role():
    mixed = data_lib.ConstructSummary(
        construct=row(7, 0.0).construct, study="s", assay=EXP, target="T", parent="p",
        family="enrichment", kind=KIND_ENRICHMENT,
        measurements=(
            data_lib.ReplicateMeasurement(value=None, lower=None, upper=1.0, censoring="left",
                                          replicate=1),
            data_lib.ReplicateMeasurement(value=2.0, lower=2.0, upper=2.0, censoring="none",
                                          replicate=2)),
        region=(("H", 3),), scale=SCALE_EXPRESSION, source="synthetic")
    record = run16(readouts([1.0, 3.0, 3.05], [0.0, 0.05, 1.0], extra=[mixed]), tolerance=0.1)
    assert record["excluded_not_exact"] == {"affinity": {}, "expression": {"mixed_censoring": 1}}
    assert record["population"] == 3


def test_the_e16_record_is_a_data_support_summary_with_no_model_prediction():
    record = run16(readouts([1.0, 3.0, 3.05], [0.0, 0.05, 1.0]), tolerance=0.1)
    assert record["evidence_kind"] == "data_support_summary" and record["model_predictions"] is None
    assert "No model prediction enters" in record["reading"]
    report_lib.require_no_placeholder(record)
    empty = run16(readouts([1.0], [0.0]), tolerance=0.1)
    assert empty["supported"] is False and empty["evidence_kind"] == "data_support_summary"
    report_lib.require_no_placeholder(empty)


def test_the_pair_cap_is_recorded_when_it_bites(monkeypatch):
    monkeypatch.setattr(report_lib, "E16_PAIR_CAP", 2)
    record = run16(readouts([1.0, 3.0, 3.05], [0.0, 0.05, 1.0]), tolerance=0.1)
    stratum, = record["strata"]
    assert stratum["candidate_pairs"] == 3 and stratum["pairs"] == 2
    assert stratum["pairs_capped"] is True


def test_an_empty_report_input_is_unsupported_not_an_error():
    record = run16([], tolerance=0.1)
    assert record["supported"] is False and record["population"] == 0
    assert record["strata"] == [] and record["value"] is None
