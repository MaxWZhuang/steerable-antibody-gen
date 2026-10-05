"""Regression tests for S2a: data, exposure, shape and scientific-control correctness (2 Oct 2026).

Each test drives the real function with the input that broke it and asserts behaviour:

* **exposure** -- ``[a, a, b]`` is three presentations of two constructs; a batch drawn across
  sources is attributed row by row, extra whole-window preference rows and a quartet joint's
  forwards land on the source that owns them; every additive bucket counter sums to its global;
  a versioned state restores every bucket exactly; a version-1 state keeps its globals and says
  its identity-derived breakdown is unavailable instead of inventing zeros;
* **shapes** -- ``selected_shapes`` follows ``profile.json`` as it appears and changes, a
  microbatch must divide the effective batch exactly and carry the declared local recipe (an
  even size of at least four when interactions are requested), every control arm is refused
  before a model loads, the profiler records incompatible candidates as rejected and carries on,
  and a draw that does not hold the candidate's number of rows is never accepted;
* **exact scalars** -- a hand-built contrast with a mixed-censored, ordinal, cross-assay or
  cross-scale endpoint is refused before any forward, an interval-kind row of degenerate points
  is valid, permuted effect bundles stay valid, enrichment ordering is sign-only; condition
  cuts, enrichment ranking, the development and reserved panels and E15/E16 read exact values and
  count what they exclude; E8 never pools interactions across units;
* **geometry and E19** -- the adaptation bank is the eligible population independent of exact
  values, the nearest-training distance is exact, a reversal needs a real context reversal, a
  score tie is half credit, and a required gate passes only on an affirmative ``True``;
* **reference copies** -- a planned mean ablation owns its reference, an export is an
  independent copy, and caller provenance cannot override the donor-batch count;
* **review repairs (section G)** -- every per-row exposure count is validated before it is
  aggregated; a legacy ledger that recorded no family infers its unattributed rows instead of
  failing family additivity; duplicate reversal pairs agree exactly and a score tolerance must be
  finite and non-negative; the control gate fails NaN rankings and invalid margins and an
  uncomparable E19 is an unsupported panel with a reason; ``capture`` returns caller-owned
  copies; the local profile searches every family for a quartet the joint really runs on,
  records the joints that executed, and qualification refuses a full recipe without them;
  ``effective_batch`` is judged as supplied.

Everything is synthetic and CPU-only; nothing is downloaded or trained, and no reserved file is
opened (the reserved read is a stub).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import types

import numpy as np
import pytest
import torch

from smallAntibodyGen.experiments import her2_routes_controls as controls
from smallAntibodyGen.experiments import her2_routes_data as data_lib
from smallAntibodyGen.experiments import her2_routes_interp as interp
from smallAntibodyGen.experiments import her2_routes_qualify as qualify
from smallAntibodyGen.experiments import her2_routes_report as report_lib
from smallAntibodyGen.experiments import her2_routes_serial as serial
from smallAntibodyGen.experiments import her2_routes_services as services_mod
from smallAntibodyGen.experiments import her2_routes_spec as spec
from smallAntibodyGen.experiments import her2_routes_tasks as tasks_lib
from smallAntibodyGen.experiments import her2_support_paths as paths
from smallAntibodyGen.experiments.her2_routes_conditions import (
    BIN_HIGH,
    fit_bin_thresholds,
)
from smallAntibodyGen.experiments.her2_routes_sources import (
    CENSOR_LEFT,
    CENSOR_NONE,
    CENSOR_RIGHT,
    KIND_ENRICHMENT,
    KIND_INTERVAL,
    KIND_POINT,
    SCALE_ENRICHMENT,
    SCALE_EXPRESSION,
    SCALE_PKD,
    Observation,
)
from smallAntibodyGen.models import her2_routes_hooks as hooks_lib
from smallAntibodyGen.tests.test_her2_functional_repairs import (
    _Mechanism,
    batch_of,
    donor_reference,
    replicate,
    run_model,
    toy_registry,
)
from smallAntibodyGen.tests.test_her2_routes_dplm_services import (  # noqa: F401  (fixtures)
    HEAVY as DPLM_HEAVY,
    LIGHT as DPLM_LIGHT,
    ROUTE_FULL,
    fake_scaffold,
    harness,
    train as train_with,
)
from smallAntibodyGen.tests.test_her2_routes_tasks import (  # noqa: F401  (fixtures)
    ASSAY,
    HEAVY,
    LIGHT,
    mutate,
    pieces,
    summary,
)

ROUTE = "ablang2_masked_editor"


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------

def contribution(source, identities, *, forwards=None, targets=0, measured=True,
                 family=data_lib.FAMILY_HER2):
    identities = tuple(identities)
    return data_lib.ExposureContribution(
        source=source, rows=len(identities),
        forwards=len(identities) if forwards is None else forwards, targets=targets,
        identities=identities, measured=measured, family=family)


def row_of(index, measurements, *, assay=ASSAY, kind=KIND_POINT, scale=SCALE_PKD,
           family="affinity", partition=data_lib.PARTITION_TRAIN, source="synthetic",
           study="synthetic", parent="synthetic::parent", region=(("H", 3),)):
    """A construct summary whose identity is fixed by ``index`` (so two readouts can join)."""
    heavy = list(HEAVY)
    for position in (index % 20, 20 + index // 20):
        heavy[position] = "P" if heavy[position] != "P" else "Q"
    construct = serial.Construct(heavy="".join(heavy), light=LIGHT, sites=tuple(region),
                                 source=source, record_id=f"r{index}")
    return data_lib.ConstructSummary(
        construct=construct, study=study, assay=assay, target="target", parent=parent,
        family=family, kind=kind, measurements=tuple(measurements), region=tuple(region),
        partition=partition, scale=scale, source=source)


def mixed(*points, floor=6.0):
    """One floor (left-censored) replicate and exact points: ``exact_value`` is ``None``."""
    return [replicate(None, lower=None, upper=floor, censoring=CENSOR_LEFT),
            *[replicate(point) for point in points]]


def her2_summary(core, measurements, *, partition=data_lib.PARTITION_TRAIN, kind=KIND_POINT,
                 source="alphabind_her2_validation"):
    construct = serial.her2_construct(fake_scaffold(), core, record_id=core)
    return data_lib.ConstructSummary(
        construct=construct, study="alphabind_her2", assay="a1", target="HER2",
        parent="alphabind::wild_type", family="affinity", kind=kind,
        measurements=tuple(measurements), region=construct.sites, partition=partition,
        scale=SCALE_PKD, source=source)


def with_source(row, source):
    construct = serial.Construct(heavy=row.construct.heavy, light=row.construct.light,
                                 sites=row.construct.sites, source=source,
                                 record_id=row.construct.record_id)
    return dataclasses.replace(row, construct=construct, source=source)


def pair_contrast(background, *, left_value, right_value, margin=None, kind=None):
    left = summary(mutate(background, 3, "W"), value=left_value, region=(("H", 3),))
    right = summary(mutate(background, 3, "M"), value=right_value, region=(("H", 3),))
    return data_lib.LocalContrast(
        left=left, right=right, sites=(("H", 3),), left_value=left_value, right_value=right_value,
        assay=ASSAY, target="target", same_assay=True, coupled=False,
        kind=data_lib.CONTRAST_ALTERNATIVES,
        margin=left_value - right_value if margin is None else margin, scale="pKD")


def quartet_contrast(*, double=None, sources=None):
    region = (("H", 3), ("H", 7))
    background = summary(HEAVY, value=8.0, region=region)
    left = summary(mutate(HEAVY, 3, "W"), value=7.5, region=region)
    right = summary(mutate(HEAVY, 7, "A"), value=8.2, region=region)
    both = double if double is not None else summary(
        mutate(mutate(HEAVY, 3, "W"), 7, "A"), value=8.6, region=region)
    if sources:
        background, left, right = (with_source(row, sources[name]) for row, name in
                                   ((background, "background"), (left, "left"), (right, "right")))
        both = with_source(both, sources["double"])
    return data_lib.LocalContrast(
        left=left, right=right, sites=region, left_value=left.value, right_value=right.value,
        assay=ASSAY, target="target", same_assay=True, coupled=True,
        kind=data_lib.CONTRAST_COUPLED, background=background, double=both,
        margin=left.value - right.value, effect_left=-0.5, effect_right=0.2, double_effect=0.6,
        interaction=0.9, scale="pKD")


def whole_views(services, rows, *, bins):
    return data_lib.whole_window_views(rows, condition_spec=services.condition_spec, bins=bins,
                                       route="masked", max_positions=512)


# ---------------------------------------------------------------------------
# A. the exposure ledger
# ---------------------------------------------------------------------------

def test_identity_multiplicity_is_presentations_and_distinct_is_a_separate_count():
    ledger = data_lib.ExposureLedger()
    ledger.record(task=data_lib.TASK_HEADS, contributions=[contribution("s", ["a", "a", "b"])],
                  update=True)
    document = ledger.document()
    assert document["totals"]["measurement_presentations"] == 3
    assert document["totals"]["unique_constructs"] == 2
    for kind, name in (("by_source", "s"), ("by_task", data_lib.TASK_HEADS),
                       ("by_family", data_lib.FAMILY_HER2)):
        bucket = document[kind][name]
        assert bucket["measurement_presentations"] == 3, kind
        assert bucket["unique_constructs"] == 2 and bucket["unique_measured_constructs"] == 2, kind
    # An unmeasured presentation presents the construct and measures nothing.
    ledger.record(task=data_lib.TASK_RECONSTRUCTION,
                  contributions=[contribution("s", ["a", "c"], measured=False)])
    document = ledger.document()
    assert document["totals"]["measurement_presentations"] == 3
    assert document["by_source"]["s"]["unique_constructs"] == 3
    assert document["by_source"]["s"]["unique_measured_constructs"] == 2
    assert document["unique_measured_constructs"] == 2


def test_a_mixed_batch_is_attributed_row_by_row_and_never_to_its_first_row():
    contributions = data_lib.contributions_by_source(
        sources=["alpha", "alpha", "beta"], identities=["a", "a", "b"], forwards=[1, 1, 2],
        targets=[5, 5, 7], measured=True, family=data_lib.FAMILY_OTHER)
    assert [entry.source for entry in contributions] == ["alpha", "beta"]
    ledger = data_lib.ExposureLedger().record(
        task=data_lib.TASK_LOCAL, contributions=contributions, update=True)
    document = ledger.document()
    alpha, beta = document["by_source"]["alpha"], document["by_source"]["beta"]
    assert (alpha["microbatch_rows"], alpha["student_forwards"], alpha["supervised_targets"]) == (
        2, 2, 10)
    assert (beta["microbatch_rows"], beta["student_forwards"], beta["supervised_targets"]) == (
        1, 2, 7)
    assert alpha["measurement_presentations"] == 2 and alpha["unique_constructs"] == 1
    assert beta["measurement_presentations"] == 1
    totals = document["totals"]
    assert (totals["microbatch_rows"], totals["student_forwards"],
            totals["supervised_targets"], totals["measurement_presentations"]) == (3, 4, 17, 3)
    assert ledger.require_additive() is True
    for kind in ("task", "source", "family"):
        for key, block in document["additivity"][kind].items():
            assert block["additive"] is True, (kind, key, block)


def test_a_construct_in_two_sources_and_two_tasks_is_one_global_construct():
    ledger = data_lib.ExposureLedger()
    ledger.record(task=data_lib.TASK_LOCAL, contributions=[contribution("alpha", ["x"])])
    ledger.record(task=data_lib.TASK_WHOLE, contributions=[contribution("beta", ["x"])])
    document = ledger.document()
    assert document["totals"]["unique_constructs"] == 1
    assert document["by_source"]["alpha"]["unique_constructs"] == 1
    assert document["by_source"]["beta"]["unique_constructs"] == 1, (
        "distinct counts are per bucket: the same construct counts in each source that presented it")
    assert document["constructs_reused_across_tasks"] == 1
    assert document["totals"]["measurement_presentations"] == 2
    assert document["additivity"]["source"]["measurement_presentations"]["additive"] is True
    assert "unique_constructs" not in document["additivity"]["source"], (
        "distinct counts are not additive and are not claimed to be")
    assert "optimizer_updates" not in document["by_source"]["alpha"], (
        "an update is drawn across sources; no bucket owns one")


def populate(ledger):
    ledger.record(task=data_lib.TASK_LOCAL, update=True,
                  contributions=data_lib.contributions_by_source(
                      sources=["alpha", "alpha", "beta"], identities=["a", "b", "a"],
                      forwards=[1, 1, 2], targets=[4, 4, 2], measured=True,
                      family=data_lib.FAMILY_HER2))
    ledger.record(task=data_lib.TASK_RECONSTRUCTION, update=True,
                  contributions=[contribution("oas", ["o1", "o1", "o2"], measured=False,
                                              family=data_lib.FAMILY_REPLAY, targets=6)])
    ledger.record(task=data_lib.TASK_WHOLE, source="beta", rows=2, forwards=3, targets=9,
                  identities=["a", "z"], measured=True, family=data_lib.FAMILY_OTHER)
    return ledger


def test_a_versioned_state_restores_every_bucket_exactly_and_resumes_identically():
    original = populate(data_lib.ExposureLedger())
    state = json.loads(json.dumps(original.state()))           # plain data, JSON round-trippable
    assert state["version"] == data_lib.LEDGER_STATE_VERSION
    restored = data_lib.ExposureLedger.restore(state)
    assert restored.document() == original.document()
    assert (restored.by_task, restored.by_source, restored.by_family) == (
        original.by_task, original.by_source, original.by_family)
    assert restored.constructs == original.constructs
    assert restored.measured_constructs == original.measured_constructs
    # An identity set that survived the round trip keeps counting correctly afterwards: "z" and
    # "a" are already known to beta, so presenting them again adds presentations, not constructs.
    for ledger in (original, restored):
        ledger.record(task=data_lib.TASK_WHOLE,
                      contributions=[contribution("beta", ["a", "z", "z", "new"],
                                                  family=data_lib.FAMILY_OTHER)])
    assert restored.document() == original.document()
    assert restored.state() == original.state()
    assert restored.document()["by_source"]["beta"]["unique_constructs"] == 3
    assert restored.document()["by_source"]["beta"]["measurement_presentations"] == 1 + 2 + 4

    corrupt = json.loads(json.dumps(original.state()))
    corrupt["by_source"]["alpha"]["unique_constructs"] += 1
    with pytest.raises(ValueError, match="inconsistent"):
        data_lib.ExposureLedger.restore(corrupt)
    corrupt["version"] = 99
    with pytest.raises(ValueError, match="Unknown exposure-ledger state version"):
        data_lib.ExposureLedger.restore(corrupt)


LEGACY_BLOCK = {"optimizer_updates": 0, "microbatch_rows": 12, "student_forwards": 20,
                "unique_constructs": 0, "measurement_presentations": 0, "supervised_targets": 90}
# The task blocks partition the global rows/forwards/targets (7+5, 12+8, 50+40), as a real
# version-1 ledger's did; a block repeating the whole global in each task would sum to double it.
LEGACY_TASK_BLOCKS = {
    data_lib.TASK_WHOLE: {**LEGACY_BLOCK, "microbatch_rows": 7, "student_forwards": 12,
                          "supervised_targets": 50},
    data_lib.TASK_LOCAL: {**LEGACY_BLOCK, "microbatch_rows": 5, "student_forwards": 8,
                          "supervised_targets": 40},
}
LEGACY_STATE = {
    "totals": {"optimizer_updates": 3, "microbatch_rows": 12, "student_forwards": 20,
               "unique_constructs": 3, "measurement_presentations": 7,
               "supervised_targets": 90},
    "by_task": {task: dict(block) for task, block in LEGACY_TASK_BLOCKS.items()},
    "by_source": {"alpha": dict(LEGACY_BLOCK)},
    "by_family": {data_lib.FAMILY_HER2: dict(LEGACY_BLOCK)},
    "constructs": ["a", "b", "c"], "measured_constructs": ["a", "b"],
    "presentations": [[data_lib.TASK_WHOLE, "a", 3], [data_lib.TASK_WHOLE, "b", 2],
                      [data_lib.TASK_LOCAL, "a", 1], [data_lib.TASK_LOCAL, "c", 1]]}


def test_a_version_one_state_keeps_its_globals_and_marks_the_breakdown_unavailable():
    ledger = data_lib.ExposureLedger.restore(LEGACY_STATE)
    assert ledger.totals == LEGACY_STATE["totals"]
    document = ledger.document()
    assert document["breakdown_complete"] == {"task": False, "source": False, "family": False}
    assert document["restored_from"]["version"] == 1
    alpha = document["by_source"]["alpha"]
    assert alpha["microbatch_rows"] == 12 and alpha["supervised_targets"] == 90, (
        "what the old state did record carries over")
    for key in ("measurement_presentations", "unique_constructs", "unique_measured_constructs"):
        assert alpha[key] is None, f"{key} must be unavailable, not an invented zero"
    assert document["by_task"][data_lib.TASK_WHOLE]["measurement_presentations"] is None
    assert document["constructs_reused_across_tasks"] == 1, (
        "the per-task presentations of version 1 still say which construct was reused")
    # Recording goes on; the unavailable figures stay unavailable rather than turning into
    # undercounts, and the new rows are added to the old ones.
    ledger.record(task=data_lib.TASK_WHOLE, contributions=[contribution("alpha", ["e", "e"])],
                  update=True)
    document = ledger.document()
    assert document["by_source"]["alpha"]["microbatch_rows"] == 14
    assert document["by_source"]["alpha"]["unique_constructs"] is None
    assert document["totals"]["measurement_presentations"] == 9
    assert document["additivity"]["source"]["measurement_presentations"]["additive"] is None
    assert document["additivity"]["source"]["microbatch_rows"]["additive"] is True
    # A saved-and-restored legacy ledger is still incomplete.
    again = data_lib.ExposureLedger.restore(json.loads(json.dumps(ledger.state())))
    assert again.document()["breakdown_complete"] == {"task": False, "source": False,
                                                       "family": False}
    assert again.document()["by_source"]["alpha"]["unique_constructs"] is None


def test_a_task_outcome_must_agree_with_its_own_contributions():
    contributions = data_lib.contributions_by_source(
        sources=["alpha", "alpha", "beta"], identities=["a", "a", "b"], forwards=[1, 1, 2],
        targets=[3, 3, 1], measured=True, family=data_lib.FAMILY_OTHER)
    outcome = tasks_lib.TaskOutcome(
        loss=torch.zeros(()), rows=3, forwards=4, targets=7, identities=("a", "a", "b"),
        measured=True, source="alpha", family=data_lib.FAMILY_OTHER, contributions=contributions)
    assert sum(entry.rows for entry in outcome.exposure()) == 3
    assert outcome.document()["contributions"][1]["source"] == "beta"
    with pytest.raises(ValueError, match="sum to 4 forwards but the outcome reports 5"):
        dataclasses.replace(outcome, forwards=5).exposure()
    with pytest.raises(ValueError, match="not the identities it presented"):
        dataclasses.replace(outcome, identities=("a", "b", "b")).exposure()
    bare = tasks_lib.TaskOutcome(loss=torch.zeros(()), rows=2, forwards=2, targets=4,
                                 identities=("a", "b"), source="solo",
                                 family=data_lib.FAMILY_HER2)
    assert [entry.source for entry in bare.exposure()] == ["solo"]


def test_whole_window_attributes_each_row_and_its_preference_forward_to_its_own_source(pieces):
    services, model, _heads = pieces
    first = summary(HEAVY, value=8.0, region=(("H", 3), ("H", 7)))
    second = with_source(summary(mutate(HEAVY, 11, "W"), value=7.0,
                                 region=(("H", 3), ("H", 7))), "beta")
    bins = {first.population_key: BIN_HIGH, second.population_key: BIN_HIGH}
    views = whole_views(services, [first, second], bins=bins)
    outcome = tasks_lib.whole_window_outcome(
        services, ROUTE, model, views, rng=np.random.default_rng(3), preference_weight=1.0)
    assert outcome.diagnostics["preference_rows"] == 2
    parts = {entry.source: entry for entry in outcome.exposure()}
    assert set(parts) == {"synthetic", "beta"}
    assert parts["synthetic"].rows == parts["beta"].rows == 1
    assert parts["synthetic"].forwards == parts["beta"].forwards == 2, (
        "one forward for the assigned condition and one for its contrast, per row")
    assert sum(entry.forwards for entry in outcome.exposure()) == outcome.forwards == 4
    assert sum(entry.targets for entry in outcome.exposure()) == outcome.targets
    ledger = data_lib.ExposureLedger().record_outcome(task=data_lib.TASK_WHOLE, outcome=outcome,
                                                      update=True)
    assert ledger.require_additive() is True
    by_source = ledger.document()["by_source"]
    assert by_source["synthetic"]["student_forwards"] == by_source["beta"]["student_forwards"] == 2
    assert ledger.totals["student_forwards"] == 4 and ledger.totals["microbatch_rows"] == 2


def test_a_quartet_joints_forwards_belong_to_the_backgrounds_source(pieces):
    services, model, _heads = pieces
    contrast = quartet_contrast(sources={"background": "bg_source", "left": "edit_source",
                                         "right": "edit_source", "double": "edit_source"})
    outcome = tasks_lib.local_outcome(
        services, ROUTE, model, [contrast], kappa=2.0, reconstruction_weight=0.25,
        interaction_weight=0.5, max_joint=1)
    assert outcome.diagnostics["interaction"]["quartets_trained"] == 1
    assert outcome.forwards == 4 + 3, "four endpoint rows plus the joint's three scored rows"
    parts = {entry.source: entry for entry in outcome.exposure()}
    assert parts["bg_source"].rows == 1 and parts["bg_source"].forwards == 1 + 3
    assert parts["edit_source"].rows == 3 and parts["edit_source"].forwards == 3
    assert sum(entry.targets for entry in outcome.exposure()) == outcome.targets
    ledger = data_lib.ExposureLedger().record_outcome(task=data_lib.TASK_LOCAL, outcome=outcome)
    assert ledger.require_additive() is True


def test_a_diffusion_reconstruction_batch_keeps_each_rows_source(harness):
    services = harness.services
    model = services.build_route(ROUTE_FULL, seed=1)

    def outcome_for(sources):
        constructs = [serial.Construct(heavy=DPLM_HEAVY, light=DPLM_LIGHT,
                                       sites=((serial.HEAVY, 3),), source=source,
                                       record_id=f"r{index}")
                      for index, source in enumerate(sources)]
        views = data_lib.reconstruction_views(
            constructs, rng=np.random.default_rng(0), mask_fraction=0.15,
            condition_spec=services.condition_spec, family="display", route="masked",
            max_positions=256)
        return tasks_lib.reconstruction_outcome(
            services, ROUTE_FULL, model, views, family=data_lib.FAMILY_REPLAY,
            rng=np.random.default_rng(3))

    single = outcome_for(["oas", "oas"])
    assert [(entry.source, entry.rows) for entry in single.exposure()] == [("oas", 2)]
    assert single.exposure()[0].targets == single.targets > 0

    several = outcome_for(["oas", "oas", "other"])
    parts = {entry.source: entry for entry in several.exposure()}
    assert parts["oas"].rows == 2 and parts["other"].rows == 1
    assert parts["oas"].targets == parts["other"].targets == 0, (
        "the model reports one corrupted-position total per batch; it is not split by guess")
    batch_total = parts[data_lib.UNATTRIBUTED_SOURCE]
    assert batch_total.rows == 0 and batch_total.targets == several.targets > 0
    assert sum(entry.targets for entry in several.exposure()) == several.targets
    ledger = data_lib.ExposureLedger().record_outcome(task=data_lib.TASK_RECONSTRUCTION,
                                                      outcome=several)
    assert ledger.require_additive() is True


def test_training_records_per_source_exposure_and_resumes_it_exactly(harness, tmp_path):
    directory = tmp_path / "train"
    first = train_with(harness, ROUTE_FULL, updates=2, directory=directory, resume=False)
    exposure = first["exposure"]
    assert exposure["totals"]["microbatch_rows"] == 4
    source = exposure["by_source"]["synthetic"]
    assert source["microbatch_rows"] == 4 and source["student_forwards"] == 4
    assert source["measurement_presentations"] == 0, "reconstruction presents, it does not measure"
    assert source["unique_constructs"] == 1, "two identical sequences are one distinct construct"
    assert exposure["breakdown_complete"] == {"task": True, "source": True, "family": True}
    assert first["shapes"]["provenance"]["profile_present"] is False
    resumed = train_with(harness, ROUTE_FULL, updates=3, directory=directory, resume=True)
    after = resumed["exposure"]
    assert after["totals"]["microbatch_rows"] == 6
    assert after["by_source"]["synthetic"]["microbatch_rows"] == 6
    assert after["by_source"]["synthetic"]["unique_constructs"] == 1, (
        "the restored identity set knew the construct; a lost one would count it again")
    for kind in ("task", "source", "family"):
        for key, block in after["additivity"][kind].items():
            assert block["additive"] is True, (kind, key)


# ---------------------------------------------------------------------------
# B. shape and profile contracts
# ---------------------------------------------------------------------------

QUARTET_RECIPE = {"max_contrast_sites": 2, "local_quartet_probability": 0.25,
                  "interaction_weight": 0.5}
PAIR_RECIPE = {"max_contrast_sites": 1, "local_quartet_probability": 0.0,
               "interaction_weight": 0.0}


@pytest.mark.parametrize("size", [1, 2, 3, 5, 7])
def test_a_recipe_requesting_interactions_refuses_an_odd_or_small_microbatch(size):
    with pytest.raises(ValueError, match="cannot carry the declared local recipe"):
        data_lib.validate_microbatch(size, curriculum=QUARTET_RECIPE)


@pytest.mark.parametrize("size", [4, 6, 8, 12])
def test_a_recipe_requesting_interactions_accepts_an_even_microbatch_of_four_or_more(size):
    checked = data_lib.validate_microbatch(size, effective_batch=24, curriculum=QUARTET_RECIPE)
    assert checked["microbatch"] == size and checked["accumulation"] == 24 // size
    assert checked["local_recipe"]["mode"] == "interactions"


@pytest.mark.parametrize("size,accepted", [(1, False), (3, False), (2, True), (4, True)])
def test_a_pair_only_recipe_needs_an_even_microbatch_of_two_or_more(size, accepted):
    if accepted:
        assert data_lib.validate_microbatch(size, curriculum=PAIR_RECIPE)["microbatch"] == size
    else:
        with pytest.raises(ValueError, match="cannot carry the declared local recipe"):
            data_lib.validate_microbatch(size, curriculum=PAIR_RECIPE)


def test_the_recipe_requirement_comes_from_the_declaration_not_from_a_draw():
    # max_contrast_sites 2 with a zero quartet probability and zero interaction weight is pair-only.
    recipe = {"max_contrast_sites": 2, "local_quartet_probability": 0.0, "interaction_weight": 0.0}
    assert data_lib.local_shape_recipe(recipe)["mode"] == "pairs"
    assert data_lib.local_shape_recipe({**recipe, "interaction_weight": 0.5})["mode"] == \
        "interactions"
    # No local share, or no declaration, asks nothing beyond a positive size.
    unscheduled = {**QUARTET_RECIPE, "shares": {data_lib.TASK_LOCAL: 0.0}}
    assert data_lib.local_shape_recipe(unscheduled)["mode"] == "not_scheduled"
    assert data_lib.validate_microbatch(1, curriculum=unscheduled)["microbatch"] == 1
    assert data_lib.local_shape_recipe({})["mode"] == "undeclared"


@pytest.mark.parametrize("bad", [0, -2, 2.5, 4.0, "4", True, None])
def test_a_microbatch_must_be_a_positive_integer(bad):
    with pytest.raises(ValueError, match="positive integer"):
        data_lib.validate_microbatch(bad)


def test_a_microbatch_must_divide_the_effective_batch_exactly():
    with pytest.raises(ValueError, match="does not divide the effective batch 10 exactly"):
        data_lib.validate_microbatch(4, effective_batch=10)
    with pytest.raises(ValueError, match="not the declared effective batch"):
        data_lib.validate_microbatch(4, effective_batch=64, accumulation=15)
    assert data_lib.validate_microbatch(4, effective_batch=64, accumulation=16)["accumulation"] == 16


def test_selected_shapes_follow_the_profile_file_as_it_appears_and_changes(harness):
    services = harness.services
    declared = services.selected_shapes()
    assert declared["source"].startswith("the config's declared shapes")
    assert (declared["microbatch"], declared["accumulation"]) == (2, 1)
    assert declared["provenance"]["profile_present"] is False
    target = harness.context.path(spec.PROFILE_JSON)
    first_digest = paths.write_json(target, {"route": {"chosen_microbatch": 1}})
    shaped = services.selected_shapes()
    assert (shaped["microbatch"], shaped["accumulation"], shaped["effective_batch"]) == (1, 2, 2)
    assert shaped["source"].startswith("the smallest microbatch the profile accepted")
    assert shaped["provenance"]["profile_present"] is True
    assert shaped["provenance"]["profile_sha256"] == first_digest == paths.sha256_file(target)
    assert shaped["provenance"]["profile_chosen_microbatch"] == {"route": 1}
    second_digest = paths.write_json(target, {"route": {"chosen_microbatch": 2},
                                              "other": {"chosen_microbatch": 2}})
    changed = services.selected_shapes()
    assert (changed["microbatch"], changed["accumulation"]) == (2, 1)
    assert changed["provenance"]["profile_sha256"] == second_digest != first_digest
    target.unlink()
    assert services.selected_shapes()["source"].startswith("the config's declared shapes"), (
        "a deleted profile is a fallback again, not a remembered shape")


def test_a_profile_choosing_a_shape_the_recipe_cannot_run_is_refused(harness):
    harness.config["curriculum"].update(QUARTET_RECIPE)
    harness.config["optimization"].update(microbatch=4, accumulation=2, effective_batch=8)
    services = harness.services
    assert services.selected_shapes()["microbatch"] == 4
    target = harness.context.path(spec.PROFILE_JSON)
    paths.write_json(target, {"route": {"chosen_microbatch": 3}})
    with pytest.raises(ValueError, match="does not divide the effective batch 8 exactly"):
        services.selected_shapes()
    paths.write_json(target, {"route": {"chosen_microbatch": 2}})
    with pytest.raises(ValueError, match="cannot carry the declared local recipe"):
        services.selected_shapes()
    paths.write_json(target, {"route": {"chosen_microbatch": 8}})
    assert services.selected_shapes()["accumulation"] == 1


def test_an_unrunnable_declared_shape_is_refused_before_any_model_is_built(harness, monkeypatch):
    harness.config["curriculum"].update(QUARTET_RECIPE)          # declared microbatch stays 2
    services = harness.services

    def built(*args, **kwargs):
        raise AssertionError("a model was built for a shape that cannot run")
    monkeypatch.setattr(services, "build_route", built)
    with pytest.raises(ValueError, match="cannot carry the declared local recipe"):
        services.selected_shapes()
    with pytest.raises(ValueError, match="cannot carry the declared local recipe"):
        services.train(ROUTE_FULL, stream=lambda *args, **kwargs: None, updates=1, seed=1,
                       directory=harness.root / "never", resume=False)
    # Every control arm shares this curriculum, so the stage refuses once, before the first arm.
    with pytest.raises(ValueError, match="cannot carry the declared local recipe"):
        services.run_controls()
    # A declared microbatch that does not divide the effective batch is refused too.
    harness.config["curriculum"].clear()
    harness.config["optimization"].update(microbatch=3, accumulation=1, effective_batch=4)
    with pytest.raises(ValueError, match="not the declared effective batch|does not divide"):
        services.selected_shapes()


def reconstruction_draw(services, count):
    constructs = [serial.Construct(heavy=DPLM_HEAVY, light=DPLM_LIGHT, sites=((serial.HEAVY, 3),),
                                   source="synthetic", record_id=f"r{index}")
                  for index in range(count)]
    views = data_lib.reconstruction_views(
        constructs, rng=np.random.default_rng(0), mask_fraction=0.15,
        condition_spec=services.condition_spec, family="display", route="masked",
        max_positions=256)
    return tasks_lib.TaskDraw(task=data_lib.TASK_RECONSTRUCTION, family=data_lib.FAMILY_REPLAY,
                              views=views, source="synthetic")


def test_the_profile_rejects_incompatible_candidates_and_continues_to_a_valid_one(harness):
    harness.config["curriculum"].update(QUARTET_RECIPE)
    harness.config["optimization"].update(microbatch=4, accumulation=2, effective_batch=8)
    services = harness.services
    profile = services.profile(
        ROUTE_FULL, microbatches=(1, 2, 3, 4), steps=2,
        draws={data_lib.TASK_RECONSTRUCTION: reconstruction_draw(services, 4)})
    rejected = {entry["microbatch"]: entry for entry in profile["rejected_shapes"]}
    assert sorted(rejected) == [1, 2, 3]
    assert "cannot carry the declared local recipe" in rejected[1]["reason"]
    assert "cannot carry the declared local recipe" in rejected[2]["reason"]
    assert "does not divide the effective batch 8 exactly" in rejected[3]["reason"]
    assert profile["candidates_requested"] == [1, 2, 3, 4]
    assert {row["microbatch"] for row in profile["rows"]} == {4}
    assert profile["accepted_microbatches"] == [4] and profile["chosen_microbatch"] == 4
    assert profile["accumulation_steps"] == 2 and profile["effective_batch"] == 8
    row = profile["rows"][0]
    assert row["requested_microbatch"] == 4 and row["rows_measured"] == 4
    assert row["failure"] is None and row["accepted"] is True


def test_a_candidate_is_accepted_only_when_its_draw_holds_that_many_rows(harness):
    harness.config["curriculum"].update(QUARTET_RECIPE)
    harness.config["optimization"].update(microbatch=4, accumulation=2, effective_batch=8)
    services = harness.services
    profile = services.profile(
        ROUTE_FULL, microbatches=(4, 8), steps=2,
        draws={data_lib.TASK_RECONSTRUCTION: reconstruction_draw(services, 4)})
    by_size = {row["microbatch"]: row for row in profile["rows"]}
    assert by_size[4]["accepted"] is True and by_size[4]["rows_measured"] == 4
    assert by_size[8]["accepted"] is False and by_size[8]["rows_measured"] == 4
    assert by_size[8]["failure"].startswith("shape_mismatch")
    assert by_size[8]["failure_detail"] == {"requested": 8, "formed": 4}
    assert by_size[8]["completed_steps"] == 0, "a mislabelled shape is not timed"
    assert profile["accepted_microbatches"] == [4] and profile["chosen_microbatch"] == 4


def test_a_profile_with_no_valid_candidate_refuses_before_building_a_model(harness, monkeypatch):
    harness.config["curriculum"].update(QUARTET_RECIPE)
    harness.config["optimization"].update(microbatch=4, accumulation=2, effective_batch=8)
    services = harness.services

    def built(*args, **kwargs):
        raise AssertionError("a model was built before the candidates were judged")
    monkeypatch.setattr(services, "build_route", built)
    with pytest.raises(ValueError, match="rejected before any model was built"):
        services.profile(ROUTE_FULL, microbatches=(1, 2, 3), steps=2,
                         draws={data_lib.TASK_RECONSTRUCTION: reconstruction_draw(services, 2)})


class _Pool:
    """A contrast pool stand-in: ``len(alternatives())`` endpoint rows of one kind.

    A coupled one carries what ``data_lib.joint_executable`` reads -- a background, a double, two
    sites, a numeric interaction and margin -- on a point scale by default. ``scale_kind`` makes its
    endpoints enrichment-scale, which ``local_outcome`` orders and never fits; ``interaction=None``
    is a quartet whose interaction was not measured.
    """

    def __init__(self, kind, rows, source="s", *, scale_kind=KIND_POINT, interaction=0.5,
                 margin=0.1):
        self.kind = kind
        self._members = [types.SimpleNamespace(
            construct=types.SimpleNamespace(identity=lambda: "id"), source=source)
            for _ in range(rows)]
        self.left = types.SimpleNamespace(source=source, kind=scale_kind)
        coupled = kind == data_lib.CONTRAST_COUPLED
        self.background = object() if coupled else None
        self.double = object() if coupled else None
        self.sites = (("H", 3), ("H", 7)) if coupled else (("H", 3),)
        self.interaction = interaction if coupled else None
        self.margin = margin

    def alternatives(self):
        return self._members


def profile_stub(bank, curriculum, *, by_family=None):
    """A services stand-in for ``_profile_local_draw``.

    ``bank`` is the one pool every non-empty family returns; ``by_family`` maps each family to its
    OWN pool, which is what a first-family-pairs / later-family-quartets search needs.
    """
    members = {data_lib.FAMILY_OTHER: ("other_row",), data_lib.FAMILY_HER2: ("her2_row",)}
    if by_family is None:
        members = {data_lib.FAMILY_OTHER: ("row",), data_lib.FAMILY_HER2: ()}
        by_family = {data_lib.FAMILY_OTHER: bank}
    owner = {rows: family for family, rows in members.items() if rows}
    stub = types.SimpleNamespace(
        config={"curriculum": curriculum},
        contrast_pool=lambda rows, **kwargs: tuple(by_family.get(owner[tuple(rows)], ())),
        contrast_views=lambda contrasts, route: tuple(
            object() for contrast in contrasts for _ in contrast.alternatives()))
    return stub, {"summaries": dict(members)}


@pytest.mark.parametrize("size", [4, 6, 8])
def test_the_local_profile_draw_holds_exactly_the_requested_rows(size):
    quartet, pair = _Pool(data_lib.CONTRAST_COUPLED, 4), _Pool(data_lib.CONTRAST_SINGLE, 2)
    stub, pools = profile_stub([quartet, pair], QUARTET_RECIPE)
    notes = {}
    draw, reason = services_mod.FlightServices._profile_local_draw(stub, pools, ROUTE, size, 1,
                                                                   notes)
    assert reason is None and draw.rows() == size
    kinds = [contrast.kind for contrast in draw.contrasts]
    assert kinds.count(data_lib.CONTRAST_COUPLED) == size // 4, "quartets first: the worst shape"
    if size == 8:
        assert "local_contrasts_repeated" in notes, "one quartet was used twice, and it says so"


def test_the_local_profile_draw_refuses_what_the_pool_cannot_tile():
    quartet = _Pool(data_lib.CONTRAST_COUPLED, 4)
    stub, pools = profile_stub([quartet], QUARTET_RECIPE)
    draw, reason = services_mod.FlightServices._profile_local_draw(stub, pools, ROUTE, 6, 1, {})
    assert draw is None and "cannot tile 6 endpoint rows" in reason
    stub, pools = profile_stub([], QUARTET_RECIPE)
    draw, reason = services_mod.FlightServices._profile_local_draw(stub, pools, ROUTE, 4, 1, {})
    assert draw is None
    # A pair-only pool cannot stand in for a recipe that requests interaction quartets: the old
    # fallback profiled the cheaper pair computation and certified it. It is unavailable now.
    pairs_only, pools = profile_stub([_Pool(data_lib.CONTRAST_SINGLE, 2)], QUARTET_RECIPE)
    notes = {}
    draw, reason = services_mod.FlightServices._profile_local_draw(pairs_only, pools, ROUTE, 4, 1,
                                                                   notes)
    assert draw is None and "quartet that executes it" in reason
    assert notes["local_quartet_unavailable"] == reason


class _Closure:
    radius = 2

    def require_allowed(self, **kwargs):
        return True

    def violates(self, **kwargs):
        return None


def stream_for(harness, monkeypatch, bank, *, microbatch):
    config = harness.config
    config["curriculum"].update(QUARTET_RECIPE)
    config["optimization"].update(microbatch=microbatch, accumulation=1, effective_batch=microbatch)
    config["routes"][ROUTE_FULL]["updates"] = 4
    services = harness.services
    row = row_of(0, [replicate(8.0)])
    monkeypatch.setattr(services, "closure", lambda: _Closure())
    monkeypatch.setattr(services, "closure_window", lambda construct: "WINDOW")
    monkeypatch.setattr(services, "training_pools", lambda: {
        "summaries": {data_lib.FAMILY_HER2: (), data_lib.FAMILY_OTHER: (row,),
                      data_lib.FAMILY_REPLAY: ()},
        "display": None, "counts": {}, "closure_radius": 2})
    monkeypatch.setattr(services, "contrast_pool", lambda eligible, **kwargs: tuple(bank))
    monkeypatch.setattr(services, "assay_index", lambda: {})
    monkeypatch.setattr(services, "contrast_views", lambda contrasts, route: tuple(
        object() for contrast in contrasts for _ in contrast.alternatives()))
    return services


def local_draw(services, **kwargs):
    stream = services.curriculum_stream(ROUTE_FULL, seed=3, updates=2, **kwargs)
    return stream(1, 0, np.random.default_rng(0), task_override=data_lib.TASK_LOCAL,
                  family_override=data_lib.FAMILY_OTHER)


def test_the_local_stream_emits_exactly_the_validated_microbatch(harness, monkeypatch):
    quartet = _Pool(data_lib.CONTRAST_COUPLED, 4)
    services = stream_for(harness, monkeypatch, [quartet], microbatch=4)
    assert local_draw(services).rows() == 4
    pairs = [_Pool(data_lib.CONTRAST_SINGLE, 2), _Pool(data_lib.CONTRAST_ALTERNATIVES, 2)]
    services = stream_for(harness, monkeypatch, pairs, microbatch=6)
    assert local_draw(services).rows() == 6
    # A pool that cannot tile the batch says so instead of presenting a shorter one.
    services = stream_for(harness, monkeypatch, [quartet], microbatch=6)
    with pytest.raises(ValueError, match="emitted 4 endpoint row.*microbatch of 6"):
        local_draw(services)


def test_the_stream_refuses_a_microbatch_its_recipe_cannot_tile(harness, monkeypatch):
    services = stream_for(harness, monkeypatch, [_Pool(data_lib.CONTRAST_SINGLE, 2)],
                          microbatch=4)
    for size in (1, 2, 3):
        with pytest.raises(ValueError, match="cannot carry the declared local recipe"):
            services.curriculum_stream(ROUTE_FULL, seed=3, updates=2, microbatch_override=size)
    harness.config["optimization"].update(microbatch=3, accumulation=1, effective_batch=3)
    with pytest.raises(ValueError, match="cannot carry the declared local recipe"):
        services.curriculum_stream(ROUTE_FULL, seed=3, updates=2)


# --- qualification judges the shape too ------------------------------------

def qualify_rows(size, **changes):
    from smallAntibodyGen.tests.test_her2_routes_qualify import _row, _tasks
    return [_row(task, size, **changes) for task in _tasks()]


def qualify_profile(rows, chosen, **changes):
    from smallAntibodyGen.tests.test_her2_routes_qualify import _profile
    return _profile(rows, chosen, **changes)


def analyse(profile, **kwargs):
    return qualify.analyse_profile(profile, route="dplm_150m_masked", tasks=data_lib.TASKS,
                                   diffusion=True, gpu_required=True, **kwargs)


def test_qualification_needs_the_measured_rows_to_equal_the_chosen_microbatch():
    rows = qualify_rows(4)
    status, reason, evidence = analyse(qualify_profile(rows, 4))
    assert status is None, reason
    rows[0]["rows_measured"] = 2
    status, reason, _ = analyse(qualify_profile(rows, 4))
    assert status == qualify.VERDICT_FAILED
    assert "the draw held 2 row(s) but the selected microbatch is 4" in reason
    # An accepted larger size that was not measured at that size is evidence, not a pass.
    rows = qualify_rows(4) + qualify_rows(8, rows_measured=4)
    status, reason, evidence = analyse(qualify_profile(rows, 4))
    assert status is None, reason
    assert {entry["microbatch"] for entry in evidence["shape_mismatches"]} == {8}


#: What a real profile row records for a local batch that trained one quartet joint under
#: ``QUARTET_RECIPE`` (interaction_weight 0.5, the default max_joint_per_update of 1).
JOINT_EVIDENCE = {"coupled_contrasts": 1, "joint_executable_contrasts": 1, "quartets_available": 1,
                  "quartets_trained": 1, "steps_observed": 3, "interaction_weight": 0.5,
                  "max_joint_per_update": 1}


def with_local_evidence(rows, evidence=JOINT_EVIDENCE, **changes):
    """``rows`` whose local row carries ``evidence`` (``None`` records none) with ``changes``."""
    for row in rows:
        if row["task"] == data_lib.TASK_LOCAL:
            row["interaction_evidence"] = None if evidence is None else {**evidence, **changes}
    return rows


def test_qualification_checks_divisibility_and_the_declared_recipe_when_they_are_known():
    rows = qualify_rows(4)
    status, reason, _ = analyse(qualify_profile(rows, 4, effective_batch=10, accumulation_steps=2))
    assert status == qualify.VERDICT_FAILED and "does not divide the effective batch" in reason
    status, reason, _ = analyse(qualify_profile(rows, 4, effective_batch=64,
                                                accumulation_steps=15))
    assert status == qualify.VERDICT_FAILED and "not the declared effective batch" in reason
    status, reason, _ = analyse(qualify_profile(with_local_evidence(rows), 4, effective_batch=64,
                                                accumulation_steps=16),
                                curriculum=QUARTET_RECIPE)
    assert status is None, reason
    small = with_local_evidence(qualify_rows(2))
    status, reason, _ = analyse(qualify_profile(small, 2, effective_batch=64,
                                                accumulation_steps=32),
                                curriculum=QUARTET_RECIPE)
    assert status == qualify.VERDICT_FAILED
    assert "cannot carry the declared local recipe" in reason


# ---------------------------------------------------------------------------
# C. exact scalar supervision and reporting
# ---------------------------------------------------------------------------

def counting(services):
    calls = []
    original = services.canonical_log_probs

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    services.canonical_log_probs = counted
    return calls


def mixed_summary(heavy, region, *, points=(8.0, 8.4), floor=6.0):
    base = summary(heavy, value=8.0, region=region)
    return dataclasses.replace(base, kind=KIND_INTERVAL, measurements=(
        data_lib.ReplicateMeasurement(value=None, lower=None, upper=floor, censoring="left",
                                      replicate="a"),
        *[data_lib.ReplicateMeasurement(value=point, lower=point, upper=point, censoring="none",
                                        replicate=str(index)) for index, point in
          enumerate(points)]))


def test_a_hand_built_quartet_with_a_mixed_censored_endpoint_is_refused_before_any_forward(
        pieces):
    services, model, _heads = pieces
    calls = counting(services)
    region = (("H", 3), ("H", 7))
    double = mixed_summary(mutate(mutate(HEAVY, 3, "W"), 7, "A"), region)
    assert double.value == pytest.approx(8.2) and double.exact_value is None
    contrast = quartet_contrast(double=double)
    with pytest.raises(ValueError, match="double endpoint is mixed_censoring, not an exact scalar"):
        tasks_lib.local_outcome(services, ROUTE, model, [contrast], kappa=2.0,
                                reconstruction_weight=0.25, interaction_weight=0.5, max_joint=1)
    with pytest.raises(ValueError, match="not an exact scalar target"):
        tasks_lib.quartet_joint_outcome(services, ROUTE, model, contrast, kappa=2.0)
    assert calls == [], "a forward ran before the endpoints were checked"
    trained = tasks_lib.local_outcome(
        services, ROUTE, model, [quartet_contrast()], kappa=2.0, reconstruction_weight=0.25,
        interaction_weight=0.5, max_joint=1)
    assert trained.diagnostics["interaction"]["quartets_trained"] == 1 and calls


@pytest.mark.parametrize("change,phrase", [
    (lambda c: setattr(c, "left", dataclasses.replace(c.left, kind="ordinal")),
     "left endpoint is ordinal"),
    (lambda c: setattr(c, "right", dataclasses.replace(
        c.right, measurements=(data_lib.ReplicateMeasurement(
            value=None, lower=9.5, upper=None, censoring="right", replicate=1),))),
     "right endpoint is censored"),
    (lambda c: setattr(c, "left", dataclasses.replace(c.left, assay="another:assay")),
     "left endpoint is assay 'another:assay'"),
    (lambda c: setattr(c, "right", dataclasses.replace(c.right, target="another")),
     "right endpoint is target 'another'"),
    (lambda c: setattr(c, "right", dataclasses.replace(c.right, scale="enrichment")),
     "endpoints span scales"),
    (lambda c: setattr(c, "scale", "other-scale"), "the contrast declares 'other-scale'"),
    (lambda c: setattr(c, "left", dataclasses.replace(c.left, family="enrichment")),
     "endpoints span families")])
def test_a_contrast_whose_endpoints_are_not_one_exact_scalar_target_is_refused(pieces, change,
                                                                               phrase):
    services, model, _heads = pieces
    calls = counting(services)
    contrast = pair_contrast(HEAVY, left_value=8.0, right_value=7.0)
    change(contrast)
    with pytest.raises(ValueError, match=phrase):
        tasks_lib.local_outcome(services, ROUTE, model, [contrast], kappa=2.0,
                                reconstruction_weight=0.25)
    assert calls == []


def test_an_interval_kind_row_of_degenerate_points_is_a_valid_endpoint(pieces):
    services, model, _heads = pieces
    contrast = pair_contrast(HEAVY, left_value=8.0, right_value=7.0)
    contrast.left = dataclasses.replace(contrast.left, kind=KIND_INTERVAL)
    contrast.right = dataclasses.replace(contrast.right, kind=KIND_INTERVAL)
    assert data_lib.scalar_contrast_problems(contrast) == []
    outcome = tasks_lib.local_outcome(services, ROUTE, model, [contrast], kappa=2.0,
                                      reconstruction_weight=0.25)
    assert torch.isfinite(outcome.loss)


def test_background_permuted_contrasts_stay_valid_and_still_train(pieces):
    services, model, _heads = pieces
    first = pair_contrast(HEAVY, left_value=8.0, right_value=6.0)
    second = pair_contrast(mutate(HEAVY, 11, "P"), left_value=7.0, right_value=8.5)
    for seed in range(40):
        changed, audit = controls.permute_background_effects([first, second], seed=seed)
        if audit["effect_bundles_moved"]:
            break
    assert audit["effect_bundles_moved"] == 2, "no seed moved a bundle; the control did nothing"
    for contrast in changed:
        assert data_lib.scalar_contrast_problems(contrast) == []
    # The permuted margin is deliberately NOT the endpoints' difference: that is the control.
    assert changed[0].margin != pytest.approx(changed[0].left.exact_value
                                              - changed[0].right.exact_value)
    outcome = tasks_lib.local_outcome(services, ROUTE, model, list(changed), kappa=2.0,
                                      reconstruction_weight=0.25)
    assert torch.isfinite(outcome.loss)


def enrichment_contrast(margin, background=HEAVY):
    contrast = pair_contrast(background, left_value=3.0, right_value=2.0, margin=margin)
    for name in ("left", "right"):
        setattr(contrast, name, dataclasses.replace(getattr(contrast, name),
                                                    kind=KIND_ENRICHMENT, scale=SCALE_ENRICHMENT))
    contrast.scale = SCALE_ENRICHMENT
    return contrast


def test_enrichment_ordering_is_sign_only_and_numeric_margins_are_not(pieces):
    services, model, _heads = pieces

    def loss_of(contrasts):
        return tasks_lib.local_outcome(services, ROUTE, model, contrasts, kappa=2.0,
                                       reconstruction_weight=0.25).loss.detach()
    other = mutate(HEAVY, 11, "P")
    small = loss_of([enrichment_contrast(0.5), enrichment_contrast(-2.0, other)])
    large = loss_of([enrichment_contrast(50.0), enrichment_contrast(-200.0, other)])
    assert torch.allclose(small, large), "a monotone rescale of enrichment margins moved the loss"
    numeric_small = loss_of([pair_contrast(HEAVY, left_value=8.0, right_value=7.5)])
    numeric_large = loss_of([pair_contrast(HEAVY, left_value=8.0, right_value=-42.0)])
    assert not torch.allclose(numeric_small, numeric_large), (
        "a pKD margin is fitted to kappa x margin; it is not sign-only")


def indexed_pool(count):
    return [row_of(index, [replicate(float(index + 1))]) for index in range(count)]


def test_condition_cuts_use_exact_values_and_bin_other_rows_by_their_whole_interval():
    rows = indexed_pool(30)                                       # exact values 1 .. 30
    mixed_high = row_of(30, [replicate(None, lower=None, upper=25.0, censoring=CENSOR_LEFT),
                             replicate(99.0), replicate(101.0)], kind=KIND_INTERVAL)
    inside = row_of(31, [replicate(15.0, lower=12.0, upper=18.0)])
    straddling = row_of(32, [replicate(15.0, lower=8.0, upper=22.0)])
    above = row_of(33, [replicate(None, lower=25.0, upper=None, censoring=CENSOR_RIGHT)],
                   kind=KIND_INTERVAL)
    held_out = row_of(34, [replicate(500.0)], partition=data_lib.PARTITION_DEVELOPMENT)
    summaries = rows + [mixed_high, inside, straddling, above, held_out]
    assert mixed_high.value == pytest.approx(100.0) and mixed_high.exact_value is None
    thresholds, bins, support = services_mod.fit_condition_bins(
        summaries, minimum_support=10, fit=fit_bin_thresholds)
    key = next(iter(thresholds))
    values = np.arange(1.0, 31.0)
    assert thresholds[key]["lower"] == pytest.approx(float(np.quantile(values, 1 / 3)))
    assert thresholds[key]["upper"] == pytest.approx(float(np.quantile(values, 2 / 3))), (
        "a non-exact row's uncensored-only mean moved a cut")
    assert support[key]["exact_rows_fitted"] == 30 and support[key]["train_rows"] == 33
    assert support[key]["excluded_by_exact_status"] == {"mixed_censoring": 1,
                                                        "nondegenerate_bounds": 2}
    assert bins[rows[0].population_key] == "low" and bins[rows[29].population_key] == "high"
    assert bins[inside.population_key] == "mid", "an interval inside one bin is binned"
    assert straddling.population_key not in bins, (
        "its value 15 is in the middle bin, but its interval 8-22 straddles both cuts")
    assert mixed_high.population_key not in bins, "its union interval spans every bin"
    assert bins[above.population_key] == "high", "a right-censored row above the upper cut"
    assert held_out.population_key in bins, "the cuts are applied to held-out rows too"


def test_scalar_ordering_uses_exact_values_but_keeps_every_row_in_the_pool():
    ranked = [row_of(0, [replicate(1.0)], kind=KIND_ENRICHMENT, family="enrichment",
                     assay="enrichment:assay", scale=SCALE_ENRICHMENT),
              row_of(1, [replicate(2.0)], kind=KIND_ENRICHMENT, family="enrichment",
                     assay="enrichment:assay", scale=SCALE_ENRICHMENT),
              row_of(2, mixed(9.0, 9.5), kind=KIND_ENRICHMENT, family="enrichment",
                     assay="enrichment:assay", scale=SCALE_ENRICHMENT)]
    assert ranked[2].value == pytest.approx(9.25) and ranked[2].exact_value is None
    groups = services_mod.exact_value_groups(ranked)
    assert sorted(groups) == [1.0, 2.0], "the mixed row's invented 9.25 is not a rank key"
    payload = data_lib.measurement_payload(ranked, assay_index={"enrichment:assay": 0})
    assert not np.isnan(payload.value[0]) and np.isnan(payload.value[2])
    assert payload.replicates() == 1 + 1 + 3, "every replicate still reaches the likelihood"
    assert int(np.asarray(payload.replicate_censored).sum()) == 1
    pairs = tasks_lib._enrichment_pairs(payload, payload.kind_mask(KIND_ENRICHMENT))
    assert pairs["left"] == [1] and pairs["right"] == [0], "row 2 has no scalar order"


def test_the_development_panel_reads_exact_values_and_counts_what_it_excludes():
    exact = her2_summary("WGGDGFYAMD", [replicate(8.0), replicate(8.4)])
    mixed_row = her2_summary("WGGDGFYAMA", mixed(9.0, 9.4), kind=KIND_INTERVAL)
    censored = her2_summary("WGGDGFYAMV",
                            [replicate(None, lower=9.5, upper=None, censoring=CENSOR_RIGHT)],
                            kind=KIND_INTERVAL)
    training = her2_summary("WGGDGFYAMC", [replicate(7.0)])
    split = types.SimpleNamespace(
        development=("WGGDGFYAMV", "WGGDGFYAMA", "WGGDGFYAMD", "WGGDGFYAMK"))
    stub = types.SimpleNamespace(split=lambda: split, _cache={},
                                 summaries=lambda: [exact, mixed_row, censored, training])
    assert mixed_row.value == pytest.approx(9.2), "the old panel would have carried this number"
    rows = services_mod.FlightServices.development_panel_outcomes(stub)
    assert [row["window"] for row in rows] == ["WGGDGFYAMD"]
    assert rows[0]["value"] == pytest.approx(8.2)
    accounting = services_mod.FlightServices.development_panel_accounting(stub)
    assert accounting["frozen_development_windows"] == 4
    assert accounting["windows_with_a_measured_summary"] == 3
    assert accounting["windows_with_an_exact_outcome"] == 1
    assert accounting["windows_without_an_exact_outcome"] == 2
    assert accounting["summaries_excluded_by_exact_status"] == {"censored": 1,
                                                                "mixed_censoring": 1}
    nothing = types.SimpleNamespace(split=lambda: split, _cache={},
                                    summaries=lambda: [mixed_row, censored])
    with pytest.raises(ValueError, match="No development window carries an exact outcome"):
        services_mod.FlightServices.development_panel_outcomes(nothing)


def test_the_reserved_panel_qualifies_construct_level_values_and_opens_no_file(monkeypatch):
    from smallAntibodyGen.experiments import her2_routes_sources as sources

    def observation_of(core, value, *, replicate_id, censoring=CENSOR_NONE, interval=None):
        return Observation(
            construct=serial.her2_construct(fake_scaffold(), core, record_id=core),
            source="alphabind_her2_validation", family="affinity", kind=KIND_INTERVAL,
            target="HER2", assay="alphabind_her2_validation", parent="alphabind::wild_type",
            scale=SCALE_PKD, value=value, interval=interval or (value, value),
            censoring=censoring, replicate=replicate_id,
            raw={"measurement_id": f"{core}:{replicate_id}"})
    observations = [
        observation_of("WGGDGFYAMD", 8.0, replicate_id="a"),
        observation_of("WGGDGFYAMD", 8.4, replicate_id="b"),
        observation_of("WGGDGFYAMA", None, replicate_id="a", censoring=CENSOR_LEFT,
                       interval=(None, 6.0)),
        observation_of("WGGDGFYAMA", 9.0, replicate_id="b"),
        observation_of("WGGDGFYAMV", 5.0, replicate_id="a")]
    calls = []

    def fake_loader(path, **kwargs):
        calls.append((path, kwargs))
        return {"observations": observations, "reserved_unlock": {"spent": True},
                "window_gate": {"gated": False}, "rows": len(observations)}
    monkeypatch.setattr(sources, "load_alphaseq_observations", fake_loader)
    stub = types.SimpleNamespace(
        split=lambda: types.SimpleNamespace(reserved=("WGGDGFYAMD", "WGGDGFYAMA")),
        source_paths=lambda: {"alphabind_her2_validation": "never-opened.tsv"},
        scaffold=fake_scaffold)
    panel, record = services_mod.FlightServices.reserved_panel(stub, unlock="the-unlock")
    assert [row["window"] for row in panel] == ["WGGDGFYAMD"]
    assert panel[0]["value"] == pytest.approx(8.2)
    assert record["reserved_observations"] == 4 and record["reserved_summaries"] == 2
    assert record["summaries_excluded_by_exact_status"] == {"mixed_censoring": 1}
    assert record["unlock"] == {"spent": True}
    (path, kwargs), = calls
    assert path == "never-opened.tsv" and kwargs["reserved_unlock"] == "the-unlock"
    assert kwargs["allowed_windows"] is None, "the single-use unlock is what lifts the gate"


def test_e16_joins_exact_values_only_and_counts_the_rest():
    def readouts(*, second_expression):
        # The expression readout carries the EXPRESSION scale: a role comes from the scale, and a
        # binding-enrichment scale here would (correctly) make it no expression readout at all.
        rows = [row_of(0, [replicate(8.0)]), row_of(1, [replicate(9.0)]),
                row_of(0, [replicate(3.0)], family="enrichment", kind=KIND_ENRICHMENT,
                       assay="enrichment:assay", scale=SCALE_EXPRESSION),
                row_of(1, second_expression, family="enrichment", kind=KIND_ENRICHMENT,
                       assay="enrichment:assay", scale=SCALE_EXPRESSION)]
        return types.SimpleNamespace(summaries=lambda: rows)
    paired = report_lib.e16_readouts(readouts(second_expression=[replicate(3.5)]), tolerance=1.0)
    assert paired["population"] == 1 and paired["value"] == 1.0
    assert paired["excluded_not_exact"] == {"affinity": {}, "expression": {}}
    assert paired["residual_expression_imbalance"] == pytest.approx(0.5)
    assert paired["tolerances"]["source"] == "legacy_broadcast", "the broadcast is recorded"
    assert paired["evidence_kind"] == "data_support_summary"
    assert [stratum["selection"] for stratum in paired["strata"]] == ["inferred_unique"]
    # With the second construct's expression only a mixed-censored mean, the old join (on
    # ``value``) would still have paired it; on exact values there is nothing to compare.
    blocked = report_lib.e16_readouts(readouts(second_expression=mixed(3.4, 3.6)), tolerance=1.0)
    assert blocked["supported"] is False and blocked["population"] == 0
    assert blocked["excluded_not_exact"]["expression"] == {"mixed_censoring": 1}


def test_e15_scores_exact_withheld_rows_and_counts_the_rest():
    held = [row_of(index, [replicate(7.0 + index)], partition=data_lib.PARTITION_DEVELOPMENT)
            for index in range(3)]
    held_mixed = row_of(3, mixed(9.0, 9.4), partition=data_lib.PARTITION_DEVELOPMENT,
                        kind=KIND_INTERVAL)
    held_censored = row_of(
        4, [replicate(None, lower=9.5, upper=None, censoring=CENSOR_RIGHT)],
        partition=data_lib.PARTITION_DEVELOPMENT, kind=KIND_INTERVAL)
    training = [row_of(10 + index, [replicate(float(index))]) for index in range(5)]
    scored = []

    def summary_scores(route, model, rows):
        scored.append([row.exact_value for row in rows])
        return [float(row.exact_value) for row in rows]
    stub = types.SimpleNamespace(
        summaries=lambda: held + [held_mixed, held_censored] + training,
        withheld_marker=lambda row: None if row.is_training else "study:held",
        load_for_scoring=lambda route, checkpoint: (object(), object(), {"checkpoint": "stub"}),
        summary_scores=summary_scores)
    record = report_lib.e15_transfer(stub, ROUTE, "checkpoint")
    block = record["by_heldout"]["study:held"]
    assert block["rows"] == 3 and block["excluded"] == {"censored": 1, "mixed_censoring": 1}
    assert record["excluded_by_heldout"] == {"study:held": {"censored": 1,
                                                            "mixed_censoring": 1}}
    assert block["spearman"] == pytest.approx(1.0)
    assert block["seen_training_control_rows"] == 3, "named for what it is: seen-training rows"
    assert "SEEN-TRAINING" in record["control_policy"]
    assert all(value is not None for values in scored for value in values)
    assert record["value"] == pytest.approx(0.0)
    every = types.SimpleNamespace(summaries=lambda: [held_mixed, held_censored],
                                  withheld_marker=stub.withheld_marker)
    blocked = report_lib.e15_transfer(every, ROUTE, "checkpoint")
    assert blocked["supported"] is False
    assert blocked["excluded_by_heldout"]["study:held"] == {"censored": 1, "mixed_censoring": 1}


def test_e8_never_pools_interactions_across_units():
    def quartet(interaction, scale):
        return types.SimpleNamespace(kind=data_lib.CONTRAST_COUPLED, interaction=interaction,
                                     scale=scale)

    def stub_for(contrasts):
        return types.SimpleNamespace(
            eligible_summaries=lambda: [types.SimpleNamespace(is_training=True)],
            config={"split": {"seed": 1}}, contrast_pool=lambda rows, **kwargs: contrasts,
            _cache={"contrast_inventory": {"by_kind": {
                data_lib.CONTRAST_COUPLED: len(contrasts)}}})
    both = [quartet(0.4, SCALE_PKD), quartet(-0.6, SCALE_PKD), quartet(50.0, SCALE_ENRICHMENT)]
    record = services_mod.FlightServices.panel_interaction(stub_for(both))
    assert record["population"] == 2
    assert record["value"] == pytest.approx(0.5), "a pooled mean over three would be 17.0"
    assert record["primary_scale"] == SCALE_PKD
    assert record["by_scale"][SCALE_ENRICHMENT]["mean_absolute_interaction"] == 50.0
    assert record["by_scale"][SCALE_PKD]["quartets"] == 2
    only_enrichment = services_mod.FlightServices.panel_interaction(
        stub_for([quartet(50.0, SCALE_ENRICHMENT)]))
    assert only_enrichment["supported"] is False and only_enrichment["value"] is None
    assert "not pooled with pKD" in only_enrichment["blocking_reason"]
    assert only_enrichment["by_scale"][SCALE_ENRICHMENT]["quartets"] == 1


def test_e8_is_labelled_a_training_data_diagnostic_with_no_predictive_claim():
    """The interactions are measured TRAINING quartets; no model prediction enters this API."""
    def quartet(interaction):
        return types.SimpleNamespace(kind=data_lib.CONTRAST_COUPLED, interaction=interaction,
                                     scale=SCALE_PKD)

    def stub_for(contrasts, count):
        return types.SimpleNamespace(
            eligible_summaries=lambda: [types.SimpleNamespace(is_training=True)],
            config={"split": {"seed": 1}}, contrast_pool=lambda rows, **kwargs: contrasts,
            _cache={"contrast_inventory": {"by_kind": {data_lib.CONTRAST_COUPLED: count}}})
    measured = services_mod.FlightServices.panel_interaction(
        stub_for([quartet(0.4), quartet(-0.6)], 2))
    empty = services_mod.FlightServices.panel_interaction(stub_for([], 0))
    no_pkd = services_mod.FlightServices.panel_interaction(stub_for(
        [types.SimpleNamespace(kind=data_lib.CONTRAST_COUPLED, interaction=2.0,
                               scale=SCALE_ENRICHMENT)], 1))
    for record in (measured, empty, no_pkd):
        assert record["evidence_kind"] == "training_data_interaction_diagnostic"
        assert record["model_predictions"] is None
        assert record["population_scope"] == "measured training quartets, not held-out data"
        endpoint = record["predictive_endpoint"]
        assert endpoint["status"] == "not_evaluated"
        assert "no model interaction accuracy" in endpoint["scope"]
        report_lib.require_no_placeholder(record)
    assert measured["value"] == pytest.approx(0.5) and measured["supported"] is True
    # The data-unsupported reasons describe DATA only; the predictive endpoint is its own field.
    for record in (empty, no_pkd):
        assert record["supported"] is False and record["value"] is None
        assert "quartet" in record["blocking_reason"]
        assert "predictive" not in record["blocking_reason"]
        assert "model" not in record["blocking_reason"]


# ---------------------------------------------------------------------------
# D. geometry and orientation
# ---------------------------------------------------------------------------

def test_the_adaptation_bank_is_the_eligible_population_independent_of_exact_values():
    exact = her2_summary("WGGDGFYAMD", [replicate(8.0)])
    censored = her2_summary("WGGDGFYAMA",
                            [replicate(None, lower=9.5, upper=None, censoring=CENSOR_RIGHT)],
                            kind=KIND_INTERVAL)
    mixed_row = her2_summary("WGGDGFYAMV", mixed(9.0, 9.4), kind=KIND_INTERVAL)
    development = her2_summary("WGGDGFYAMC", [replicate(7.0)],
                               partition=data_lib.PARTITION_DEVELOPMENT)
    refused = her2_summary("WGGDGFYAMK", [replicate(7.5)])
    closure = types.SimpleNamespace(
        violates=lambda *, window, identity: "radius" if window == "WGGDGFYAMK" else None)
    display = types.SimpleNamespace(cores=("AAAAAAAAAA", "WGGDGFYAMD"))
    stub = types.SimpleNamespace(
        summaries=lambda: [exact, censored, mixed_row, development, refused],
        closure=lambda: closure, display_pool=lambda: display, _cache={})
    stub.adaptation_bank = lambda: services_mod.FlightServices.adaptation_bank(stub)
    stub.her2_training_bank = lambda: services_mod.FlightServices.her2_training_bank(stub)

    bank = services_mod.FlightServices.adaptation_bank(stub)
    assert bank == ("AAAAAAAAAA", "WGGDGFYAMA", "WGGDGFYAMD", "WGGDGFYAMV"), (
        "censored and mixed-censored constructs still train the head and whole-window tasks")
    windows, values = services_mod.FlightServices.her2_training_bank(stub)
    assert windows == ["WGGDGFYAMD"] and values == [8.0], "the exact bank is a different thing"
    record = services_mod.FlightServices.adaptation_bank_record(stub)
    assert record["windows"] == 4 and record["measured_her2_windows"] == 3
    assert record["display_windows"] == 2 and record["display_only_windows"] == 1
    assert record["measured_without_exact_value"] == 2
    assert record["counts"]["not_training_partition"] == 1
    assert record["counts"]["closure_refused"] == 1
    assert record["counts"]["excluded_by_exact_status"] == {"censored": 1, "mixed_censoring": 1}
    assert record["sha256"] == hashlib.sha256(("\n".join(bank) + "\n").encode()).hexdigest()
    assert "carries no labels" in record["semantics"]
    labelled = services_mod.FlightServices.training_label_bank(stub)
    assert labelled["windows"] == ["WGGDGFYAMD"] and labelled["values"] == [8.0]
    assert labelled["provenance"]["kind"] == "exact_labelled_training_bank"
    assert labelled["provenance"]["rows"] == 1 and labelled["provenance"]["sha256"]
    assert "never a display class" in labelled["provenance"]["label_semantics"]


def test_reversal_profiles_use_the_exact_nearest_adaptation_distance():
    windows = ["AAAAAAAAAA", "CAAAAAAAAA", "WWWWWWWWWW"]
    panel = [{"window": window, "value": value} for window, value in zip(windows, (2.0, 1.0, 0.0))]
    stub = types.SimpleNamespace(
        development_panel_outcomes=lambda: panel,
        score_panel=lambda route, checkpoint, rows: [0.5, 0.25, 0.0],
        config={"her2": {"wild_type_core": "DDDDDDDDDD"}},
        adaptation_bank=lambda: ("AAAAAAAAWW",))
    pairs = services_mod.FlightServices.reversal_pairs(stub, ROUTE, "checkpoint")
    assert len(pairs) == 1, "only the one-site pair is a substitution group"
    pair = pairs[0]
    assert pair["left_profile"] == {"to_wild_type": 10, "to_nearest_training": 2}
    assert pair["right_profile"] == {"to_wild_type": 10, "to_nearest_training": 3}, (
        "the neighbouring development window (distance 1) is not a training sequence")
    assert pair["outcome_delta"] == pytest.approx(1.0) and pair["score_delta"] == pytest.approx(0.25)


def test_nearest_hamming_is_exact_chunked_and_never_a_sentinel():
    rng = np.random.default_rng(7)
    alphabet = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
    bank = ["".join(rng.choice(alphabet, 10)) for _ in range(200)]
    queries = ["".join(rng.choice(alphabet, 10)) for _ in range(17)]
    expected = [min(controls.hamming(query, member) for member in bank) for query in queries]
    assert controls.nearest_hamming(queries, bank) == expected
    assert controls.nearest_hamming(queries, bank, budget=64) == expected, (
        "a tiny budget forces many chunks and must not change a distance")
    assert controls.nearest_hamming(["WWWWWWWWWW"], ["DDDDDDDDDD"]) == [10], (
        "farther than any radius is a distance of ten, not -1")
    assert controls.nearest_hamming([], bank) == []
    with pytest.raises(ValueError, match="empty bank"):
        controls.nearest_hamming(queries, [])
    with pytest.raises(ValueError, match="window length"):
        controls.nearest_hamming(["AAAA"], bank)


def reversal_pair(left, right, outcome, score, *, left_counts=(2, 2), right_counts=(2, 2)):
    """One supplied pair; ``*_counts`` are ``(distance to wild type, to nearest training)``."""
    return {"left_window": left, "right_window": right,
            "left_profile": {"to_wild_type": left_counts[0], "to_nearest_training": left_counts[1]},
            "right_profile": {"to_wild_type": right_counts[0],
                              "to_nearest_training": right_counts[1]},
            "outcome_delta": outcome, "score_delta": score}


def test_a_pure_position_effect_is_not_a_reversal():
    """A is better than C at position 0 in BOTH backgrounds: no context reversal exists."""
    pairs = [reversal_pair("AA", "CA", +1.0, +1.0), reversal_pair("CC", "AC", -1.0, -1.0)]
    block = controls.matched_reversals(pairs)
    assert block["groups_seen"] == 1 and block["groups_in_both_directions"] == 0
    assert block["support_gated"] is True and block["macro_accuracy"] is None
    assert block["reason"] and "not an accuracy of zero" in block["reason"]


def test_a_true_context_reversal_is_retained_and_swapping_inputs_changes_nothing():
    pairs = [reversal_pair("AA", "CA", +1.0, +1.0),     # A better than C in the first context
             reversal_pair("AB", "CB", -1.0, -1.0)]     # C better than A in the second
    block = controls.matched_reversals(pairs)
    assert block["groups_in_both_directions"] == 1
    assert block["macro_accuracy"] == pytest.approx(1.0)
    swapped = [reversal_pair("CA", "AA", -1.0, -1.0), reversal_pair("CB", "AB", +1.0, +1.0)]
    assert controls.matched_reversals(swapped) == block
    assert controls.matched_reversals(list(reversed(pairs))) == block
    additive = controls.matched_reversals([reversal_pair("AA", "CA", +1.0, +1.0),
                                           reversal_pair("AB", "CB", -1.0, +1.0)])
    assert additive["macro_accuracy"] == pytest.approx(0.5), "the 0.5 ceiling of an additive rule"


def test_canonicalisation_flips_the_profiles_and_both_signs_together():
    asymmetric = reversal_pair("CA", "AA", -1.0, -0.25, left_counts=(3, 5), right_counts=(1, 4))
    canonical = controls.canonical_reversal_pair(asymmetric)
    assert (canonical["left_window"], canonical["right_window"]) == ("AA", "CA")
    assert canonical["outcome_delta"] == 1.0 and canonical["score_delta"] == 0.25
    assert dict(canonical["left_profile"]) == {"to_nearest_training": 4, "to_wild_type": 1}
    assert dict(canonical["right_profile"]) == {"to_nearest_training": 5, "to_wild_type": 3}
    assert controls.canonical_reversal_pair(reversal_pair("AAA", "CCA", 1.0, 1.0)) is None
    # Endpoint profiles are NOT sorted independently: the same substitution with different
    # profiles on the two sides is a different group, never a reversal of one another.
    different = [reversal_pair("AA", "CA", +1.0, +1.0, left_counts=(2, 2), right_counts=(2, 3)),
                 reversal_pair("AB", "CB", -1.0, -1.0, left_counts=(2, 3), right_counts=(2, 2))]
    assert controls.matched_reversals(different)["groups_in_both_directions"] == 0


def test_physical_pairs_are_deduplicated_and_measured_ties_are_excluded():
    pairs = [reversal_pair("AA", "CA", +1.0, +1.0), reversal_pair("CA", "AA", -1.0, -1.0),
             reversal_pair("AB", "CB", 0.0, 0.5), reversal_pair("AC", "CC", -1.0, -1.0)]
    block = controls.matched_reversals(pairs)
    assert block["pairs_supplied"] == 4 and block["duplicate_physical_pairs"] == 1
    assert block["measured_ties_excluded"] == 1 and block["pairs_used"] == 2
    assert block["groups_in_both_directions"] == 1
    with pytest.raises(ValueError, match="supplied twice with different measurements"):
        controls.matched_reversals([reversal_pair("AA", "CA", +1.0, +1.0),
                                    reversal_pair("AA", "CA", -1.0, -1.0)])
    with pytest.raises(ValueError, match="non-finite"):
        controls.matched_reversals([reversal_pair("AA", "CA", float("nan"), 1.0)])


# ---------------------------------------------------------------------------
# E. E19 ties, support and the required gate
# ---------------------------------------------------------------------------

def test_a_score_tie_earns_half_credit_and_a_measured_tie_is_excluded():
    outcomes = [1.0, 2.0, 3.0, 4.0]
    flat = controls.stratified_concordance([5.0, 5.0, 7.0, 7.0], outcomes, ["a", "a", "b", "b"])
    assert flat["per_stratum"]["a"]["concordance"] == 0.5
    assert flat["macro_average"] == 0.5 and flat["score_ties"] == 2
    perfect = controls.stratified_concordance([1.0, 2.0, 3.0, 4.0], outcomes, ["a", "a", "b", "b"])
    assert perfect["macro_average"] == 1.0 and perfect["score_ties"] == 0
    # The difference the fabricated zero used to create: model 1.0 minus a distance-only 0.5.
    assert perfect["macro_average"] - flat["macro_average"] == 0.5
    tied = controls.stratified_concordance([1.0, 2.0, 3.0], [4.0, 4.0, 5.0], ["a", "a", "a"])
    assert tied["measured_ties_excluded"] == 1 and tied["within_stratum_pairs"] == 2
    assert tied["per_stratum"]["a"]["concordance"] == 1.0
    mixed_ties = controls.stratified_concordance([1.0, 1.0, 2.0], [1.0, 2.0, 3.0], ["a", "a", "a"])
    assert mixed_ties["per_stratum"]["a"]["concordance"] == pytest.approx((0.5 + 1.0 + 1.0) / 3)
    tolerant = controls.stratified_concordance([1.0, 1.001], [1.0, 2.0], ["a", "a"], tolerance=0.01)
    assert tolerant["per_stratum"]["a"]["concordance"] == 0.5


def test_no_comparable_pair_is_none_with_a_reason_never_zero():
    block = controls.stratified_concordance([1.0, 2.0], [3.0, 3.0], ["a", "a"])
    assert block["macro_average"] is None and block["within_stratum_concordance"] is None
    assert "not a concordance of zero" in block["reason"]
    assert block["per_stratum"]["a"]["measured_ties_excluded"] == 1
    with pytest.raises(ValueError, match="non-finite"):
        controls.stratified_concordance([float("nan"), 1.0], [1.0, 2.0], ["a", "a"])


def test_strata_are_joint_and_exact_or_declared_coarsened():
    labels, scheme = controls.joint_distance_strata([10, 10, 9], [2, 3, 2])
    assert labels == ["wt=10|adaptation=2", "wt=10|adaptation=3", "wt=9|adaptation=2"]
    assert scheme["exact"] is True and "nothing is pooled" in scheme["binning"]
    coarse, scheme = controls.joint_distance_strata([10, 2], [7, 1], edges=(0, 1, 2, 3))
    assert coarse == ["wt=>3|adaptation=>3", "wt=2|adaptation=1"]
    assert scheme["exact"] is False and "larger distances pooled" in scheme["binning"]


E19_MARGINS = {"distance_stratified": 0.02}


def e19_stub(*, scores, outcomes=(5.0, 6.0, 4.0)):
    windows = ["AAAAAAAAAA", "AAAAAAAAAC", "WWWWWWWWWW"]
    panel = [{"window": window, "value": value} for window, value in zip(windows, outcomes)]
    return types.SimpleNamespace(
        development_panel_outcomes=lambda: panel,
        score_panel=lambda route, checkpoint, rows: list(scores),
        config={"her2": {"wild_type_core": "DDDDDDDDDD"}}), windows


def e19_banks():
    return (
        {"windows": ["AAAAAAAAAW", "WWWWWWWWWA"], "values": [5.5, 3.0],
         "provenance": {"kind": "exact_labelled_training_bank", "rows": 2, "sha256": "x" * 64}},
        {"windows": ["AAAAAAAAWW"], "values": None,
         "provenance": {"kind": "eligible_adaptation_bank", "windows": 1, "sha256": "y" * 64}})


def test_e19_uses_joint_strata_a_training_labelled_baseline_and_half_credit_for_distance():
    stub, _windows = e19_stub(scores=[1.0, 2.0, 0.0])
    labelled, geometry = e19_banks()
    record = report_lib.e19_shortcuts(stub, ROUTE, "checkpoint", margins=E19_MARGINS,
                                      training_bank=labelled, adaptation_bank=geometry)
    # Windows 1 and 2 share the joint stratum wt=10, adaptation=2 and nothing else has a partner.
    assert record["strata_scheme"]["exact"] is True
    assert record["stratified_concordance"]["macro_average"] == 1.0
    assert record["distance_only_stratified"]["macro_average"] == 0.5, (
        "constant distance inside a stratum is a coin flip, not zero")
    assert record["value"] == pytest.approx(0.5) and record["passes"] is True
    baseline = record["training_neighbour_baseline"]
    assert baseline["bank"] == labelled["provenance"]
    assert [entry["prediction"] for entry in baseline["predictions"]] == [5.5, 5.5, 3.0], (
        "the neighbours are TRAINING windows with their exact labels, not the panel itself")
    assert baseline["stratified"]["macro_average"] == 0.5, "the neighbour tie is half credit"
    assert record["adaptation_geometry_bank"] == geometry["provenance"]


def test_e19_requires_both_banks_and_refuses_a_leaky_or_labelled_geometry_bank():
    stub, _windows = e19_stub(scores=[1.0, 2.0, 0.0])
    labelled, geometry = e19_banks()
    with pytest.raises(TypeError):
        report_lib.e19_shortcuts(stub, ROUTE, "checkpoint", margins=E19_MARGINS)
    with pytest.raises(ValueError, match="provenance"):
        report_lib.e19_shortcuts(stub, ROUTE, "checkpoint", margins=E19_MARGINS,
                                 training_bank={"windows": ["AAAAAAAAAW"], "values": [1.0]},
                                 adaptation_bank=geometry)
    with pytest.raises(ValueError, match="geometry only"):
        report_lib.e19_shortcuts(stub, ROUTE, "checkpoint", margins=E19_MARGINS,
                                 training_bank=labelled,
                                 adaptation_bank={**geometry, "values": [1.0]})
    leaky = {**labelled, "windows": ["AAAAAAAAAA", "WWWWWWWWWA"]}
    with pytest.raises(ValueError, match="contains 1 development window"):
        report_lib.e19_shortcuts(stub, ROUTE, "checkpoint", margins=E19_MARGINS,
                                 training_bank=leaky, adaptation_bank=geometry)
    with pytest.raises(ValueError, match="labels for"):
        report_lib.e19_shortcuts(stub, ROUTE, "checkpoint", margins=E19_MARGINS,
                                 training_bank={**labelled, "values": [1.0]},
                                 adaptation_bank=geometry)


def test_e19_with_no_comparable_stratum_has_no_value_and_does_not_pass():
    stub, _windows = e19_stub(scores=[1.0, 2.0, 0.0], outcomes=(5.0, 5.0, 4.0))
    labelled, geometry = e19_banks()
    record = report_lib.e19_shortcuts(stub, ROUTE, "checkpoint", margins=E19_MARGINS,
                                      training_bank=labelled, adaptation_bank=geometry)
    assert record["value"] is None and record["passes"] is None
    assert "no stratum holds a comparable pair" in record["reason"]


def test_a_required_gate_passes_only_on_an_affirmative_true():
    good = {"passes": True}
    rankings = {"full": 0.5, "reconstruction_only": 0.1, "head_only": 0.2}
    failed, reasons = controls.control_gate(rankings, required=["reconstruction_only", "head_only"],
                                            margin=0.02, e19=good)
    assert failed == [] and reasons == {}
    # A required arm that never ran fails; it is not "an arm the full arm beat".
    failed, reasons = controls.control_gate(rankings, required=["reconstruction_only", "missing"],
                                            margin=0.02, e19=good)
    assert failed == ["missing"] and "no ranking" in reasons["missing"]
    # An undefined correlation is missing evidence, not zero.
    failed, reasons = controls.control_gate({**rankings, "head_only": None},
                                            required=["head_only"], margin=0.02, e19=good)
    assert failed == ["head_only"] and "not a ranking of zero" in reasons["head_only"]
    failed, reasons = controls.control_gate({**rankings, "full": None},
                                            required=["head_only"], margin=0.02, e19=good)
    assert "full_ranking_undefined" in failed and "head_only" in failed
    # An arm the full arm does not beat by the margin fails.
    failed, _ = controls.control_gate({"full": 0.40, "head_only": 0.399}, required=["head_only"],
                                      margin=0.02, e19=good)
    assert failed == ["head_only"]
    # E19: None (no comparable stratum) and False both fail; only True passes.
    for passes in (None, False):
        failed, reasons = controls.control_gate(rankings, required=[], margin=0.02,
                                                e19={"passes": passes})
        assert failed == ["distance_stratified_concordance"], passes
    failed, _ = controls.control_gate(rankings, required=[], margin=0.02, e19=None)
    assert failed == ["distance_stratified_concordance"]


# ---------------------------------------------------------------------------
# F. the three reference copies
# ---------------------------------------------------------------------------

def test_a_planned_mean_ablation_owns_its_reference():
    model = _Mechanism()
    model.train()
    registry, name = toy_registry(model)
    reference = donor_reference(model, registry, name)           # mean 3.0 at token 1, channel 0
    registry.plan(name, kind="mean_ablate", positions=[1], reference=reference)
    planned = registry.interventions[name]["reference"]
    assert planned is not reference
    assert planned.mean.data_ptr() != reference.mean.data_ptr()
    assert planned.counts.data_ptr() != reference.counts.data_ptr()
    assert planned.sha256 == reference.sha256 and planned.document() == reference.document()
    rng_before = torch.get_rng_state()
    reference.mean.add_(5.0)                  # the caller edits its own object AFTER planning
    reference.counts.add_(1)
    registry.enable([name])
    with interp.controlled_evaluation(model):
        output = model(batch_of(1.0)["x"])
    registry.disable()
    registry.clear_plans()
    assert torch.allclose(output, torch.full((2,), 6.0)), (
        "2 x the frozen donor mean 3.0; the post-plan edit made it sixteen")
    assert planned.verify() is True
    with pytest.raises(ValueError, match="no longer matches its frozen digest"):
        reference.verify()
    assert model.training and model.drop.training and torch.equal(torch.get_rng_state(), rng_before)
    assert registry.handles == [] and registry.interventions == {}


def test_the_intervention_runs_on_the_clone_without_touching_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CUDA was touched in a CPU process")
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    for name in ("get_rng_state", "set_rng_state", "get_rng_state_all", "set_rng_state_all",
                 "device_count", "manual_seed", "manual_seed_all", "current_device"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    model = _Mechanism()
    registry, name = toy_registry(model)
    reference = donor_reference(model, registry, name)
    result = interp.intervene(model, registry, batch=batch_of(1.0), name=name,
                              kind="mean_ablate", forward=run_model, positions=[1],
                              reference=reference)
    assert torch.allclose(result["output"], torch.full((2,), 6.0))
    assert result["reference_sha256"] == reference.sha256
    assert registry.export_all() is not None


def test_an_export_is_an_independent_copy_of_the_capture():
    model = _Mechanism()
    registry, name = toy_registry(model)
    captured = interp.capture(model, registry, batch=batch_of(1.0), names=[name],
                              forward=run_model)
    snapshot = registry.captured[name].clone()
    exported = registry.export(name)
    assert exported["tensor"].data_ptr() != registry.captured[name].data_ptr()
    exported["tensor"].add_(5.0)                       # mutate the exported artifact in place
    assert torch.equal(registry.captured[name], snapshot), "the registry's capture was rewritten"
    again = registry.export(name)
    assert torch.equal(again["tensor"], snapshot)
    captured["exports"][name]["tensor"].add_(1.0)      # and the exports capture() returned
    assert torch.equal(registry.captured[name], snapshot)
    assert [exported["layout"], exported["position_axis"]] == [again["layout"],
                                                             again["position_axis"]]
    assert registry.export_all([name])[name]["tensor"].data_ptr() != \
        registry.captured[name].data_ptr()


def test_reference_provenance_cannot_override_the_donor_batch_count():
    model = _Mechanism()
    registry, name = toy_registry(model)
    batches = [batch_of(2.0, rows=1, ids=["d0"], seed=10),
               batch_of(4.0, rows=1, ids=["d1"], seed=11)]
    ran = []

    def forward(module, batch):
        ran.append(1)
        return module(batch["x"])
    with pytest.raises(ValueError, match="may not set 'donor_batches'"):
        interp.build_reference_mean(model, registry, donor_batches=batches, name=name,
                                    forward=forward, ids_of=lambda batch: batch["ids"],
                                    provenance={"donor_batches": 99})
    assert ran == [], "the override was refused before any donor was captured"
    reference = interp.build_reference_mean(
        model, registry, donor_batches=batches, name=name, forward=forward,
        ids_of=lambda batch: batch["ids"], provenance={"donor_set": "toy"})
    document = reference.document()
    assert document["donor_batches"] == 2 and document["donor_set"] == "toy"
    assert document["donor_rows"] == 2 and document["donor_ids"] == ["d0", "d1"]
    assert hooks_lib.ReferenceMean.frozen_copy(reference).sha256 == reference.sha256


# ---------------------------------------------------------------------------
# G. repairs after the independent review of the first S2a implementation
# ---------------------------------------------------------------------------

# --- exposure counts are validated one element at a time -------------------

@pytest.mark.parametrize("forwards", [
    [.5, .5], [-1, 2], [1, True], [1, float("nan")], [2, float("inf")], [1, "2"],
    .5, -1, True, float("nan"), "3", None])
def test_every_per_row_exposure_count_is_validated_before_any_aggregation(forwards):
    # ``int(.5)`` is 0 and ``-1 + 2`` is 1: both used to become a plausible forward count.
    with pytest.raises(ValueError, match="integer|negative"):
        data_lib.contributions_by_source(sources=["s", "s"], identities=["a", "b"],
                                         forwards=forwards, targets=1, measured=True)
    with pytest.raises(ValueError, match="integer|negative"):
        data_lib.contributions_by_source(sources=["s", "s"], identities=["a", "b"],
                                         forwards=1, targets=forwards, measured=True)


def test_a_bad_count_names_its_element_and_numpy_integers_are_still_accepted():
    with pytest.raises(ValueError, match=r"forwards\[1\]"):
        data_lib.contributions_by_source(sources=["s", "s"], identities=["a", "b"],
                                         forwards=[1, .5], targets=0, measured=True)
    with pytest.raises(ValueError, match=r"cannot be negative: targets\[0\] is -3"):
        data_lib.contributions_by_source(sources=["s", "s"], identities=["a", "b"],
                                         forwards=1, targets=[-3, 4], measured=True)
    with pytest.raises(ValueError, match="integer"):
        data_lib.contributions_by_source(
            sources=["s", "s"], identities=["a", "b"], forwards=np.array([1.0, 2.0]), targets=0,
            measured=True)
    parts = data_lib.contributions_by_source(
        sources=["s", "s", "t"], identities=["a", "b", "c"], forwards=np.array([1, 2, 3]),
        targets=np.int64(2), measured=True)
    assert [(part.source, part.rows, part.forwards, part.targets) for part in parts] == [
        ("s", 2, 3, 4), ("t", 1, 3, 2)]
    (only,) = data_lib.contributions_by_source(
        sources=["s"], identities=["a"], forwards=np.array(5), targets=np.uint8(1), measured=True)
    assert (only.forwards, only.targets) == (5, 1)


# --- a legacy ledger that recorded no family -------------------------------

def legacy_state_without_family():
    """A genuine old ledger: it recorded rows under a source and a task and named no family."""
    ledger = data_lib.ExposureLedger().record(
        task=data_lib.TASK_HEADS, source="s", rows=1, forwards=1, targets=1, identities=["a"],
        measured=True)
    state = ledger.state()
    legacy = {key: state[key] for key in ("totals", "by_task", "by_source", "by_family")}
    legacy.update(constructs=["a"], measured_constructs=["a"],
                  presentations=[[data_lib.TASK_HEADS, "a", 1]])
    assert legacy["by_family"] == {}
    return legacy


def test_a_valid_legacy_ledger_with_no_family_infers_its_unattributed_rows():
    old = data_lib.ExposureLedger.restore(legacy_state_without_family())
    assert old.family_unattributed_rows == 1 and old.by_family == {}
    family = old.additivity()["family"]
    assert all(block["additive"] is None for block in family.values()), (
        "no family owns the row, so the family sum is unanswerable -- not a broken sum")
    assert old.require_additive() is True
    assert old.additivity()["source"]["microbatch_rows"]["additive"] is True
    restored_from = old.document()["restored_from"]
    assert restored_from["family_unattributed_rows"] == 1 and "no family" in restored_from["note"]
    # New records continue: a family-named one lands in its family, an unnamed one in the
    # unattributed count, and neither is ever split across families by guess.
    old.record(task=data_lib.TASK_WHOLE,
               contributions=[contribution("s", ["b", "b"], family=data_lib.FAMILY_HER2)])
    assert old.family_unattributed_rows == 1
    assert old.by_family[data_lib.FAMILY_HER2]["microbatch_rows"] == 2
    old.record(task=data_lib.TASK_WHOLE, source="s", rows=3, forwards=3, targets=0,
               identities=["c", "c", "c"], measured=True)
    assert old.family_unattributed_rows == 4
    assert old.additivity()["family"]["microbatch_rows"] == {
        "sum_of_buckets": 2, "total": 6, "additive": None}
    assert old.require_additive() is True
    again = data_lib.ExposureLedger.restore(json.loads(json.dumps(old.state())))
    assert again.family_unattributed_rows == 4 and again.state() == old.state()
    assert again.additivity()["family"]["microbatch_rows"]["additive"] is None
    assert again.require_additive() is True


def test_an_ordinary_legacy_ledger_with_families_has_nothing_unattributed():
    old = data_lib.ExposureLedger.restore(LEGACY_STATE)
    assert old.family_unattributed_rows == 0
    assert old.document()["restored_from"]["family_unattributed_rows"] == 0
    additive = ("microbatch_rows", "student_forwards", "supervised_targets")
    family = old.additivity()["family"]
    assert [family[key]["additive"] for key in additive] == [True, True, True]
    assert family["measurement_presentations"]["additive"] is None, (
        "version 1 cannot say how many measured presentations each family had")
    old.record(task=data_lib.TASK_WHOLE,
               contributions=[contribution("alpha", ["e"], family=data_lib.FAMILY_HER2)])
    family = old.additivity()["family"]
    assert [family[key]["additive"] for key in additive] == [True, True, True]
    assert old.require_additive() is True


def test_a_legacy_ledger_with_partial_family_attribution_is_not_invented_into_one():
    partial = {**LEGACY_STATE, "by_family": {data_lib.FAMILY_HER2: {
        **LEGACY_BLOCK, "microbatch_rows": 8, "student_forwards": 14, "supervised_targets": 60}}}
    old = data_lib.ExposureLedger.restore(partial)
    assert old.family_unattributed_rows == 4 and old.totals == LEGACY_STATE["totals"]
    assert old.by_family[data_lib.FAMILY_HER2]["microbatch_rows"] == 8, "nothing was invented"
    assert all(block["additive"] is None for block in old.additivity()["family"].values())
    assert old.require_additive() is True


@pytest.mark.parametrize("key,value", [("microbatch_rows", 13), ("student_forwards", 21),
                                       ("supervised_targets", 91)])
def test_a_legacy_family_total_above_the_global_total_is_refused(key, value):
    broken = {**LEGACY_STATE, "by_family": {data_lib.FAMILY_HER2: {**LEGACY_BLOCK, key: value}}}
    with pytest.raises(ValueError, match="inconsistent"):
        data_lib.ExposureLedger.restore(broken)


# --- E5: duplicates agree exactly; a tolerance must be a real one ----------

def test_a_nearzero_duplicate_cannot_change_reversal_support_with_its_order():
    zero = reversal_pair("AA", "CA", 0.0, 0.0)
    tiny = reversal_pair("AA", "CA", 5e-13, 5e-13)
    reverse = reversal_pair("AC", "CC", -1.0, -1.0)
    # 5e-13 used to count as the same measurement as 0 (a 1e-12 tolerance), so the first-seen copy
    # decided between a tie and a direction and the supported-group count depended on input order.
    for ordered in ([zero, tiny, reverse], [tiny, zero, reverse], [reverse, tiny, zero]):
        with pytest.raises(ValueError, match="supplied twice with different measurements"):
            controls.matched_reversals(ordered)
    # Exact repeats are still repeats, and a measured tie stays an exact class either way round.
    flipped_tie = reversal_pair("CA", "AA", -0.0, 0.0)
    blocks = [controls.matched_reversals(ordered)
              for ordered in ([zero, flipped_tie, reverse], [flipped_tie, zero, reverse])]
    assert blocks[0] == blocks[1]
    assert blocks[0]["duplicate_physical_pairs"] == 1 and blocks[0]["measured_ties_excluded"] == 1
    assert blocks[0]["groups_in_both_directions"] == 0
    exact = [reversal_pair("AA", "CA", 1.0, 1.0), reversal_pair("CA", "AA", -1.0, -1.0),
             reversal_pair("AC", "CC", -1.0, -1.0)]
    assert controls.matched_reversals(exact)["groups_in_both_directions"] == 1
    assert controls.matched_reversals(list(reversed(exact))) == controls.matched_reversals(exact)


@pytest.mark.parametrize("bad", [-1.0, -1e-9, float("nan"), float("inf"), True, "0.1", None])
def test_a_score_tolerance_must_be_finite_and_non_negative_in_both_panels(bad):
    with pytest.raises(ValueError, match="finite non-negative"):
        controls.stratified_concordance([1.0, 1.0], [1.0, 2.0], ["s", "s"], tolerance=bad)
    with pytest.raises(ValueError, match="finite non-negative"):
        controls.matched_reversals([reversal_pair("AA", "CA", 1.0, 1.0)], tolerance=bad)


def test_a_constant_predictor_is_half_credit_and_a_valid_tolerance_is_reported():
    block = controls.stratified_concordance([1.0, 1.0], [1.0, 2.0], ["s", "s"])
    assert block["macro_average"] == 0.5 and block["tolerance"] == 0.0
    block = controls.stratified_concordance([1.0, 1.0], [1.0, 2.0], ["s", "s"],
                                            tolerance=np.float64(0.25))
    assert block["macro_average"] == 0.5 and block["tolerance"] == 0.25
    assert controls.matched_reversals([], tolerance=0)["tolerance"] == 0.0


# --- the control gate and E19 ----------------------------------------------

def test_the_control_gate_fails_finite_invalid_rankings_before_comparing_them():
    good = {"passes": True}
    nan, inf, negative_inf = float("nan"), float("inf"), float("-inf")
    # ``full - value < margin`` is False for NaN, so a NaN used to PASS the gate.
    failed, reasons = controls.control_gate({"full": nan, "head_only": .4}, required=["head_only"],
                                            margin=.02, e19=good)
    assert "full_ranking_undefined" in failed and "head_only" in failed
    assert "not a finite number" in reasons["full_ranking_undefined"]
    failed, reasons = controls.control_gate({"full": inf, "head_only": .4}, required=["head_only"],
                                            margin=.02, e19=good)
    assert "full_ranking_undefined" in failed and "head_only" in failed
    for value in (nan, inf, negative_inf, True, "0.4"):
        failed, reasons = controls.control_gate({"full": .5, "head_only": value},
                                                required=["head_only"], margin=.02, e19=good)
        assert failed == ["head_only"], value
        assert "not a finite number" in reasons["head_only"]
    assert controls.control_gate({"full": .5, "head_only": .4}, required=["head_only"],
                                 margin=.02, e19=good) == ([], {})


@pytest.mark.parametrize("margin", [float("nan"), float("inf"), float("-inf"), -0.01, True, None,
                                    "0.02"])
def test_the_control_gate_fails_an_invalid_margin_instead_of_comparing_with_it(margin):
    good = {"passes": True}
    failed, reasons = controls.control_gate({"full": .5, "head_only": .1}, required=["head_only"],
                                            margin=margin, e19=good)
    assert "control_margin_invalid" in failed and "head_only" in failed
    assert "finite non-negative" in reasons["control_margin_invalid"]
    # A negative margin would let an arm that BEATS the full arm pass.
    failed, _ = controls.control_gate({"full": .1, "head_only": .5}, required=["head_only"],
                                      margin=margin, e19=good)
    assert "control_margin_invalid" in failed and "head_only" in failed
    # Zero is a valid margin ("not worse than"), and the arm still has to meet it.
    assert controls.control_gate({"full": .5, "head_only": .5}, required=["head_only"],
                                 margin=0.0, e19=good)[0] == []
    assert controls.control_gate({"full": .4, "head_only": .5}, required=["head_only"],
                                 margin=0.0, e19=good)[0] == ["head_only"]


@pytest.mark.parametrize("record", [
    {"passes": True, "margin": float("nan")}, {"passes": True, "value": float("inf")},
    {"passes": True, "margin": -0.1}, {"passes": True, "supported": False},
    {"passes": 1}, {"passes": "yes"}, {"passes": False}, {"passes": None}])
def test_the_control_gate_does_not_pass_an_e19_record_that_is_not_an_affirmative_result(record):
    failed, _ = controls.control_gate({"full": .5}, required=[], margin=.02, e19=record)
    assert failed == ["distance_stratified_concordance"]
    ok = {"passes": True, "value": 0.1, "margin": 0.02, "supported": True}
    assert controls.control_gate({"full": .5}, required=[], margin=.02, e19=ok)[0] == []


def test_e19_without_a_comparable_stratum_is_an_unsupported_panel_with_a_real_reason():
    stub, _windows = e19_stub(scores=[1.0, 2.0, 0.0], outcomes=(5.0, 5.0, 4.0))
    labelled, geometry = e19_banks()
    record = report_lib.e19_shortcuts(stub, ROUTE, "checkpoint", margins=E19_MARGINS,
                                      training_bank=labelled, adaptation_bank=geometry)
    assert record["supported"] is False, "the report's unsupported list reads this field"
    assert record["value"] is None and record["passes"] is None
    assert "no stratum holds a comparable pair" in record["blocking_reason"]
    assert record["blocking_reason"] == record["reason"]
    assert report_lib.require_no_placeholder(record) is record, (
        "the reason is a data fact, not unfinished code")
    failed, _ = controls.control_gate({"full": .5}, required=[], margin=.02, e19=record)
    assert failed == ["distance_stratified_concordance"]
    # A comparable stratum is a supported panel with a value.
    stub, _windows = e19_stub(scores=[1.0, 2.0, 0.0])
    supported = report_lib.e19_shortcuts(stub, ROUTE, "checkpoint", margins=E19_MARGINS,
                                         training_bank=labelled, adaptation_bank=geometry)
    assert supported["supported"] is True and supported["blocking_reason"] is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, None])
def test_e19_refuses_an_invalid_margin_before_it_scores_anything(bad):
    stub, _windows = e19_stub(scores=[1.0, 2.0, 0.0])
    scored = []
    stub.score_panel = lambda route, checkpoint, rows: scored.append(1) or [1.0, 2.0, 0.0]
    labelled, geometry = e19_banks()
    with pytest.raises(ValueError, match="margin"):
        report_lib.e19_shortcuts(stub, ROUTE, "checkpoint", margins={"distance_stratified": bad},
                                 training_bank=labelled, adaptation_bank=geometry)
    assert scored == []


# --- capture hands out caller-owned activations ----------------------------

def test_capture_returns_caller_owned_activations_that_later_captures_do_not_rewrite():
    model = _Mechanism()
    registry, name = toy_registry(model)
    first = interp.capture(model, registry, batch=batch_of(1.0), names=[name], forward=run_model)
    held = registry.captured[name].clone()
    returned = first["activations"][name]
    assert returned.data_ptr() != registry.captured[name].data_ptr()
    returned.add_(7.0)                                 # the caller edits what capture() returned
    assert torch.equal(registry.captured[name], held), "the registry's own capture was rewritten"
    assert torch.equal(registry.export(name)["tensor"], held)
    edited = first["activations"][name].clone()
    second = interp.capture(model, registry, batch=batch_of(2.0), names=[name], forward=run_model)
    assert second["activations"][name].data_ptr() != registry.captured[name].data_ptr()
    snapshot = registry.captured[name].clone()
    second["activations"][name].zero_()
    assert torch.equal(registry.captured[name], snapshot)
    assert torch.equal(first["activations"][name], edited), "a later capture rewrote an earlier one"


# --- the actual interaction profile ----------------------------------------

PROBABILITY_ONLY_RECIPE = {"max_contrast_sites": 2, "local_quartet_probability": 0.25,
                           "interaction_weight": 0.0}


def local_profile_draw(curriculum, by_family, size):
    stub, pools = profile_stub(None, curriculum, by_family=by_family)
    notes = {}
    draw, reason = services_mod.FlightServices._profile_local_draw(stub, pools, ROUTE, size, 1,
                                                                   notes)
    return draw, reason, notes


def test_the_joint_requirement_comes_from_the_declared_recipe():
    need = data_lib.local_joint_requirement(QUARTET_RECIPE)
    assert (need["mode"], need["coupled_rows_required"], need["joint_required"]) == (
        "interactions", True, True)
    assert need["interaction_weight"] == 0.5 and need["max_joint_per_update"] == 1
    need = data_lib.local_joint_requirement(PROBABILITY_ONLY_RECIPE)
    assert (need["coupled_rows_required"], need["joint_required"]) == (True, False)
    need = data_lib.local_joint_requirement(PAIR_RECIPE)
    assert (need["coupled_rows_required"], need["joint_required"]) == (False, False)
    need = data_lib.local_joint_requirement({**QUARTET_RECIPE, "max_joint_per_update": 0})
    assert (need["coupled_rows_required"], need["joint_required"]) == (True, False), (
        "no joint is allowed per update, so none can be required of the profile")
    need = data_lib.local_joint_requirement({**QUARTET_RECIPE,
                                             "shares": {data_lib.TASK_LOCAL: 0.0}})
    assert (need["coupled_rows_required"], need["joint_required"]) == (False, False)


def test_the_predicate_matches_what_local_outcome_actually_executes(pieces):
    services, model, _heads = pieces
    numeric = quartet_contrast()
    enrichment = quartet_contrast()
    for name in ("left", "right", "background", "double"):
        setattr(enrichment, name, dataclasses.replace(
            getattr(enrichment, name), kind=KIND_ENRICHMENT, scale=SCALE_ENRICHMENT))
    enrichment.scale = SCALE_ENRICHMENT
    unmeasured = quartet_contrast()
    unmeasured.interaction = None
    pair = pair_contrast(HEAVY, left_value=8.0, right_value=7.0)
    for contrast, executes in ((numeric, 1), (enrichment, 0), (unmeasured, 0), (pair, 0)):
        outcome = tasks_lib.local_outcome(
            services, ROUTE, model, [contrast], kappa=2.0, reconstruction_weight=0.25,
            interaction_weight=0.5, max_joint=1)
        assert outcome.diagnostics["interaction"]["quartets_trained"] == executes, contrast.kind
        assert data_lib.joint_executable(contrast) is bool(executes), contrast.kind


def test_the_profile_searches_every_family_for_the_quartet_a_later_family_holds():
    other = [_Pool(data_lib.CONTRAST_SINGLE, 2)]
    her2 = [_Pool(data_lib.CONTRAST_COUPLED, 4), _Pool(data_lib.CONTRAST_SINGLE, 2)]
    draw, reason, notes = local_profile_draw(
        QUARTET_RECIPE, {data_lib.FAMILY_OTHER: other, data_lib.FAMILY_HER2: her2}, 4)
    assert reason is None and draw.rows() == 4
    assert draw.family == data_lib.FAMILY_HER2 == notes["local_draw_family"]
    assert [contrast.kind for contrast in draw.contrasts] == [data_lib.CONTRAST_COUPLED], (
        "the first non-empty bank holds only pairs; stopping there profiled cheaper work")
    # An enrichment quartet in the first family is not the work the joint runs; a later numeric
    # one is.
    enrichment = [_Pool(data_lib.CONTRAST_COUPLED, 4, scale_kind=KIND_ENRICHMENT)]
    draw, reason, _ = local_profile_draw(
        QUARTET_RECIPE, {data_lib.FAMILY_OTHER: enrichment, data_lib.FAMILY_HER2: her2}, 4)
    assert reason is None and draw.family == data_lib.FAMILY_HER2
    assert all(data_lib.joint_executable(contrast) for contrast in draw.contrasts)


def test_a_recipe_that_fits_the_joint_is_not_profiled_with_enrichment_quartets_or_no_quartets():
    pair = _Pool(data_lib.CONTRAST_SINGLE, 2)
    for bank in ([_Pool(data_lib.CONTRAST_COUPLED, 4, scale_kind=KIND_ENRICHMENT), pair],
                 [_Pool(data_lib.CONTRAST_COUPLED, 4, interaction=None), pair],
                 [pair, _Pool(data_lib.CONTRAST_ALTERNATIVES, 2)]):
        draw, reason, notes = local_profile_draw(QUARTET_RECIPE,
                                                 {data_lib.FAMILY_OTHER: bank}, 4)
        assert draw is None, "pairs are a cheaper computation than the declared recipe"
        assert "quartet that executes it" in reason and "not profiled with pairs" in reason
        assert notes["local_quartet_unavailable"] == reason
    # The size is part of it: a size that holds no quartet cannot fill up with pairs either.
    draw, reason, _ = local_profile_draw(
        QUARTET_RECIPE, {data_lib.FAMILY_OTHER: [_Pool(data_lib.CONTRAST_COUPLED, 4), pair]}, 2)
    assert draw is None and reason


def test_a_size_eight_local_profile_holds_exactly_eight_rows_of_real_quartets():
    bank = [_Pool(data_lib.CONTRAST_COUPLED, 4), _Pool(data_lib.CONTRAST_SINGLE, 2)]
    draw, reason, notes = local_profile_draw(QUARTET_RECIPE, {data_lib.FAMILY_OTHER: bank}, 8)
    assert reason is None and draw.rows() == 8 and len(draw.views) == 8
    assert [contrast.kind for contrast in draw.contrasts] == [data_lib.CONTRAST_COUPLED] * 2
    assert "local_contrasts_repeated" in notes, "one quartet served twice, and it says so"
    assert "local_joint_not_executed" not in notes


def test_a_probability_only_coupled_recipe_profiles_its_declared_cheaper_computation_honestly():
    enrichment = [_Pool(data_lib.CONTRAST_COUPLED, 4, scale_kind=KIND_ENRICHMENT)]
    draw, reason, notes = local_profile_draw(PROBABILITY_ONLY_RECIPE,
                                             {data_lib.FAMILY_OTHER: enrichment}, 4)
    assert reason is None and draw.rows() == 4
    assert "no sequential joint runs" in notes["local_joint_not_executed"]
    # The recipe still requests quartets, so pairs alone are not its computation.
    draw, reason, _ = local_profile_draw(
        PROBABILITY_ONLY_RECIPE, {data_lib.FAMILY_OTHER: [_Pool(data_lib.CONTRAST_SINGLE, 2)]}, 4)
    assert draw is None and "coupled contrast" in reason


def test_an_explicit_pair_only_recipe_profiles_pairs_and_needs_no_quartet():
    pairs = [_Pool(data_lib.CONTRAST_SINGLE, 2), _Pool(data_lib.CONTRAST_ALTERNATIVES, 2)]
    draw, reason, notes = local_profile_draw(PAIR_RECIPE, {data_lib.FAMILY_OTHER: pairs}, 4)
    assert reason is None and draw.rows() == 4
    assert all(contrast.kind != data_lib.CONTRAST_COUPLED for contrast in draw.contrasts)
    assert "local_quartet_unavailable" not in notes and "local_joint_not_executed" not in notes
    only_quartets = [_Pool(data_lib.CONTRAST_COUPLED, 4)]
    draw, reason, _ = local_profile_draw(PAIR_RECIPE, {data_lib.FAMILY_OTHER: only_quartets}, 4)
    assert draw is None and "cannot tile 4 endpoint rows" in reason


# --- the profile records what actually ran, and qualification demands it ---

def local_profile_with_dispatch(harness, monkeypatch, *, recipe, bank, trained):
    """A real ``profile()`` over one local draw whose dispatch reports ``trained`` per step."""
    harness.config["curriculum"].update(recipe)
    harness.config["optimization"].update(microbatch=4, accumulation=2, effective_batch=8)
    services = harness.services
    steps = iter(trained)

    def reporting(svc, route, model, heads, draw, **kwargs):
        # A bank of pairs holds no quartet to train, so it cannot report one as available.
        available = sum(1 for contrast in (draw.contrasts or ())
                        if contrast.kind == data_lib.CONTRAST_COUPLED)
        outcome = tasks_lib.TaskOutcome(
            loss=torch.zeros((), requires_grad=True), term="local_effect", rows=draw.rows(),
            diagnostics={"interaction": {"quartets_available": available,
                                         "quartets_trained": next(steps)}})
        outcome.scheduled_task = draw.task
        return outcome
    monkeypatch.setattr(tasks_lib, "dispatch", reporting)
    draw = tasks_lib.TaskDraw(
        task=data_lib.TASK_LOCAL, family=data_lib.FAMILY_OTHER,
        views=tuple(object() for contrast in bank for _ in contrast.alternatives()),
        contrasts=tuple(bank), source="s")
    profile = services.profile(ROUTE_FULL, microbatches=(4,), steps=len(trained),
                               draws={data_lib.TASK_LOCAL: draw})
    return services, json.loads(json.dumps(qualify.json_safe(profile)))


def analyse_local(services, profile, curriculum):
    return qualify.analyse_profile(profile, route=ROUTE_FULL, tasks=(data_lib.TASK_LOCAL,),
                                   diffusion=False, gpu_required=False, curriculum=curriculum)


def test_the_profile_records_the_joints_that_ran_and_qualification_requires_them(
        harness, monkeypatch):
    bank = [_Pool(data_lib.CONTRAST_COUPLED, 4)]
    services, profile = local_profile_with_dispatch(harness, monkeypatch, recipe=QUARTET_RECIPE,
                                                    bank=bank, trained=[1, 1, 1])
    (row,) = profile["rows"]
    assert row["interaction_evidence"] == {
        "coupled_contrasts": 1, "joint_executable_contrasts": 1, "quartets_available": 1,
        "quartets_trained": 1, "steps_observed": 3, "interaction_weight": 0.5,
        "max_joint_per_update": 1}
    assert profile["local_joint_requirement"]["joint_required"] is True
    curriculum = dict(services.config["curriculum"])
    status, reason, evidence = analyse_local(services, profile, curriculum)
    assert status is None, reason
    assert evidence["rows"][0]["interaction_evidence"]["quartets_trained"] == 1


def test_one_step_without_a_joint_is_not_a_qualified_interaction_profile(harness, monkeypatch):
    services, profile = local_profile_with_dispatch(
        harness, monkeypatch, recipe=QUARTET_RECIPE, bank=[_Pool(data_lib.CONTRAST_COUPLED, 4)],
        trained=[1, 0, 1])
    assert profile["rows"][0]["interaction_evidence"]["quartets_trained"] == 0, (
        "the minimum over the steps is recorded, so one skipped joint cannot hide")
    status, reason, _ = analyse_local(services, profile, dict(services.config["curriculum"]))
    assert status == qualify.VERDICT_FAILED and "no joint executed" in reason


def test_a_pair_profile_cannot_certify_an_interaction_recipe_but_can_certify_its_own(
        harness, monkeypatch):
    pairs = [_Pool(data_lib.CONTRAST_SINGLE, 2), _Pool(data_lib.CONTRAST_SINGLE, 2)]
    services, profile = local_profile_with_dispatch(
        harness, monkeypatch, recipe=PAIR_RECIPE, bank=pairs, trained=[0, 0])
    evidence = profile["rows"][0]["interaction_evidence"]
    assert evidence["coupled_contrasts"] == 0 and evidence["quartets_trained"] == 0
    status, reason, _ = analyse_local(services, profile, dict(services.config["curriculum"]))
    assert status is None, f"an explicit pair-only recipe legitimately qualifies: {reason}"
    # The same pair profile is not evidence for a recipe that fits quartet joints.
    status, reason, _ = analyse_local(services, profile, {**QUARTET_RECIPE})
    assert status == qualify.VERDICT_FAILED and "interaction_weight 0.0" in reason


def test_the_generic_interaction_diagnostic_is_not_joint_evidence():
    # Pairs report an ``interaction`` diagnostic too; only the executed-joint record counts.
    rows = qualify_rows(4)
    for row in rows:
        row["dispatched"]["diagnostic_keys"] = ["interaction", "mean_timestep", "num_timesteps"]
    status, reason, _ = analyse(qualify_profile(with_local_evidence(rows, None), 4,
                                                effective_batch=64, accumulation_steps=16),
                                curriculum=QUARTET_RECIPE)
    assert status == qualify.VERDICT_FAILED and "no executed interaction evidence" in reason


@pytest.mark.parametrize("changes,phrase", [
    ({"quartets_trained": 0}, "no joint executed"),
    ({"coupled_contrasts": 0, "quartets_available": 0, "quartets_trained": 0},
     "held no coupled contrast"),
    ({"quartets_trained": 2, "quartets_available": 1}, "contradictory"),
    ({"quartets_available": 3}, "contradictory"),
    ({"interaction_weight": 0.0}, "profiled under interaction_weight 0.0"),
    ({"max_joint_per_update": 4}, "max_joint_per_update 4"),
    ({"interaction_weight": None}, "profiled under"),
    ({"quartets_trained": True}, "non-negative integer"),
    ({"quartets_trained": -1}, "non-negative integer"),
    ({"coupled_contrasts": 1.0}, "non-negative integer"),
    ({"quartets_trained": None}, "non-negative integer")])
def test_forged_or_contradictory_joint_evidence_is_refused_under_a_full_recipe(changes, phrase):
    rows = with_local_evidence(qualify_rows(4), **changes)
    status, reason, _ = analyse(qualify_profile(rows, 4, effective_batch=64,
                                                accumulation_steps=16),
                                curriculum=QUARTET_RECIPE)
    assert status == qualify.VERDICT_FAILED and phrase in reason, reason


def test_a_missing_evidence_record_is_refused_but_a_narrow_fixture_is_still_judged_alone():
    rows = with_local_evidence(qualify_rows(4), None)
    profile = qualify_profile(rows, 4, effective_batch=64, accumulation_steps=16)
    status, reason, _ = analyse(profile, curriculum=QUARTET_RECIPE)
    assert status == qualify.VERDICT_FAILED and "no executed interaction evidence" in reason
    # No curriculum: the measured shape alone is judged, so a reconstruction-style fixture is
    # not asked for joints it never claimed to run.
    status, reason, _ = analyse(profile)
    assert status is None, reason
    # An explicit pair-only recipe asks for none either.
    status, reason, _ = analyse(profile, curriculum=PAIR_RECIPE)
    assert status is None, reason


def test_a_probability_only_recipe_needs_coupled_rows_and_runs_no_joint():
    rows = with_local_evidence(qualify_rows(4), quartets_available=0, quartets_trained=0,
                               interaction_weight=0.0)
    profile = qualify_profile(rows, 4, effective_batch=64, accumulation_steps=16)
    status, reason, _ = analyse(profile, curriculum=PROBABILITY_ONLY_RECIPE)
    assert status is None, reason
    bare = with_local_evidence(qualify_rows(4), quartets_available=0, quartets_trained=0,
                               coupled_contrasts=0, interaction_weight=0.0)
    status, reason, _ = analyse(qualify_profile(bare, 4, effective_batch=64,
                                                accumulation_steps=16),
                                curriculum=PROBABILITY_ONLY_RECIPE)
    assert status == qualify.VERDICT_FAILED and "held no coupled contrast" in reason
    trained = with_local_evidence(qualify_rows(4), interaction_weight=0.0)
    status, reason, _ = analyse(qualify_profile(trained, 4, effective_batch=64,
                                                accumulation_steps=16),
                                curriculum=PROBABILITY_ONLY_RECIPE)
    assert status == qualify.VERDICT_FAILED and "runs none" in reason


#: What a real profile row records for a pair-only local batch under ``PAIR_RECIPE``
#: (interaction_weight 0.0, the default max_joint_per_update of 1): no quartet, no joint.
PAIR_EVIDENCE = {"coupled_contrasts": 0, "joint_executable_contrasts": 0, "quartets_available": 0,
                 "quartets_trained": 0, "steps_observed": 3, "interaction_weight": 0.0,
                 "max_joint_per_update": 1}


def judge_pair_only(evidence=PAIR_EVIDENCE, **changes):
    rows = with_local_evidence(qualify_rows(4), evidence, **changes)
    return analyse(qualify_profile(rows, 4, effective_batch=64, accumulation_steps=16),
                   curriculum=PAIR_RECIPE)


def test_evidence_that_agrees_with_a_pair_only_recipe_qualifies_and_missing_evidence_may_stay_missing():
    status, reason, _ = judge_pair_only()
    assert status is None, reason
    status, reason, _ = judge_pair_only(None)
    assert status is None, f"a pair-only recipe asks for no interaction evidence: {reason}"
    # A fixture may record only some of it, and a real profile writes None for what it could not
    # observe; neither is a contradiction, so neither is refused.
    status, reason, _ = judge_pair_only({"steps_observed": 3})
    assert status is None, reason
    status, reason, _ = judge_pair_only(quartets_available=None, quartets_trained=None)
    assert status is None, reason


@pytest.mark.parametrize("changes,phrase", [
    ({"coupled_contrasts": 1, "quartets_available": 1, "quartets_trained": 1},
     "reported trained although the declared recipe runs none"),
    ({"quartets_trained": 1}, "contradictory interaction evidence"),
    ({"quartets_available": 2}, "contradictory interaction evidence"),
    ({"coupled_contrasts": 2}, "pair-only and draws none"),
    ({"interaction_weight": 0.5}, "profiled under interaction_weight 0.5"),
    ({"interaction_weight": float("nan")}, "profiled under interaction_weight nan"),
    ({"interaction_weight": True}, "profiled under interaction_weight True"),
    ({"max_joint_per_update": 4}, "max_joint_per_update 4"),
    ({"max_joint_per_update": 1.0}, "max_joint_per_update 1.0"),
    ({"quartets_trained": True}, "non-negative integer"),
    ({"quartets_available": -1}, "non-negative integer"),
    ({"coupled_contrasts": 0.0}, "non-negative integer"),
    ({"quartets_trained": "0"}, "non-negative integer")])
def test_recorded_evidence_is_judged_against_a_pair_only_recipe_even_though_none_is_required(
        changes, phrase):
    status, reason, _ = judge_pair_only(**changes)
    assert status == qualify.VERDICT_FAILED and phrase in reason, reason


def test_a_partial_record_is_judged_for_the_fields_it_does_carry():
    status, reason, _ = judge_pair_only({"interaction_weight": 0.5})
    assert status == qualify.VERDICT_FAILED and "profiled under interaction_weight 0.5" in reason
    status, reason, _ = judge_pair_only({"quartets_trained": 1})
    assert status == qualify.VERDICT_FAILED and "runs none" in reason, (
        "a trained joint is refused under a recipe that runs none, whatever else is unrecorded")
    status, reason, _ = judge_pair_only({"quartets_available": 1})
    assert status is None, "one count alone cannot contradict the others"
    status, reason, _ = judge_pair_only({"quartets_available": 1, "coupled_contrasts": 0})
    assert status == qualify.VERDICT_FAILED and "contradictory" in reason


def test_an_undeclared_or_unscheduled_recipe_cannot_report_trained_joints_either():
    trained = {**PAIR_EVIDENCE, "coupled_contrasts": 1, "quartets_available": 1,
               "quartets_trained": 1, "interaction_weight": 0.5}
    row = {"task": data_lib.TASK_LOCAL, "interaction_evidence": trained}
    unscheduled = {**QUARTET_RECIPE, "shares": {data_lib.TASK_LOCAL: 0.0}}
    undeclared = {"interaction_weight": 0.5}
    for curriculum in (unscheduled, undeclared):
        need = data_lib.local_joint_requirement(curriculum)
        assert need["coupled_rows_required"] is False and need["joint_required"] is False
        defects = qualify.joint_evidence_defects(row, need)
        assert any("reported trained although the declared recipe runs none" in defect
                   for defect in defects), (need["mode"], defects)
        assert qualify.joint_evidence_defects({**row, "interaction_evidence": None}, need) == []
        assert qualify.joint_evidence_defects({"task": data_lib.TASK_LOCAL}, need) == []


def test_malformed_evidence_is_a_defect_under_every_recipe_and_the_full_recipe_is_unweakened():
    for curriculum, phrase in ((PAIR_RECIPE, "not the record of executed work"),
                               (QUARTET_RECIPE, "no executed interaction evidence")):
        need = data_lib.local_joint_requirement(curriculum)
        for malformed in ("trained", [1], 3):
            defects = qualify.joint_evidence_defects(
                {"task": data_lib.TASK_LOCAL, "interaction_evidence": malformed}, need)
            assert len(defects) == 1 and phrase in defects[0], (curriculum, malformed, defects)
    need = data_lib.local_joint_requirement(QUARTET_RECIPE)
    for absent in ({"task": data_lib.TASK_LOCAL},
                   {"task": data_lib.TASK_LOCAL, "interaction_evidence": None},
                   {"task": data_lib.TASK_LOCAL, "interaction_evidence": {}}):
        assert qualify.joint_evidence_defects(absent, need), (
            "a recipe that requests quartets still needs every field of the evidence")
    # And a complete, consistent record under the full recipe still passes untouched.
    assert qualify.joint_evidence_defects(
        {"task": data_lib.TASK_LOCAL, "interaction_evidence": JOINT_EVIDENCE}, need) == []


# --- effective_batch is judged as supplied ---------------------------------

@pytest.mark.parametrize("bad", [64.5, 64.0, True, -64, 0, "64"])
def test_qualification_refuses_a_raw_invalid_effective_batch_instead_of_dropping_it(bad):
    rows = qualify_rows(4)
    # A float used to become "no divisibility check" and the profile qualified.
    status, reason, _ = analyse(qualify_profile(rows, 4, effective_batch=bad,
                                                accumulation_steps=16))
    assert status == qualify.VERDICT_FAILED and "effective_batch" in reason
    assert "positive integer" in reason, reason


def test_a_full_recipe_needs_the_profile_to_record_its_effective_batch():
    rows = with_local_evidence(qualify_rows(4))
    status, reason, _ = analyse(qualify_profile(rows, 4), curriculum=QUARTET_RECIPE)
    assert status == qualify.VERDICT_FAILED and "records no effective_batch" in reason
    status, reason, _ = analyse(qualify_profile(rows, 4))
    assert status is None, "without a recipe the measured shape is judged alone"


@pytest.mark.parametrize("bad", [64.5, True, -8, 0, "8", None, "missing"])
def test_profile_refuses_a_raw_invalid_effective_batch_before_building_a_model(
        harness, monkeypatch, bad):
    if bad == "missing":
        harness.config["optimization"].pop("effective_batch")
    else:
        harness.config["optimization"]["effective_batch"] = bad
    services = harness.services

    def built(*args, **kwargs):
        raise AssertionError("a model was built for an invalid effective batch")
    monkeypatch.setattr(services, "build_route", built)
    # The injected-draws path is validated too: the draw does not matter, the declaration does.
    with pytest.raises(ValueError, match="effective_batch must be a positive integer"):
        services.profile(ROUTE_FULL, microbatches=(2,), steps=2,
                         draws={data_lib.TASK_RECONSTRUCTION: reconstruction_draw(services, 2)})
