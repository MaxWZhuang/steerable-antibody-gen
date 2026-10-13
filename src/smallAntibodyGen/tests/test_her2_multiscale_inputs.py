"""Frozen inputs: restore the split, load the sources, validate, and build no model.

The synthetic world (``her2_multiscale_world``) is sixteen train cores, two development cores, one
reserved and two excluded, with a real frozen record, vocabulary and inventory on disk. Every
refusal below is checked against what had NOT yet happened: no outcome converted, no model built.
"""
from __future__ import annotations

import ast
import copy
import dataclasses
import json
from pathlib import Path

import pytest

from smallAntibodyGen.experiments import her2_routes_data as data_lib
from smallAntibodyGen.experiments import her2_routes_serial as serial
from smallAntibodyGen.experiments import her2_routes_services as services_mod
from smallAntibodyGen.experiments import her2_routes_sources as sources
from smallAntibodyGen.experiments import her2_routes_split as split_lib
from smallAntibodyGen.experiments import her2_support_paths as paths
from smallAntibodyGen.experiments.her2_multiscale import inputs as inputs_lib
from smallAntibodyGen.experiments.her2_multiscale import membership as membership_lib
from smallAntibodyGen.tests import her2_multiscale_world as world_lib

ASSAY = world_lib.ASSAY


@pytest.fixture
def world(tmp_path, monkeypatch):
    return world_lib.build_world(tmp_path, monkeypatch)


def digest_tree(root):
    return {str(path.relative_to(root)): paths.sha256_file(path)
            for path in sorted(Path(root).rglob("*")) if path.is_file()}


class Spy:
    """Counts the numeric conversions of the AlphaSeq outcome column."""

    def __init__(self, monkeypatch):
        self.values = []
        real = sources.pkd_from_log10_nm

        def spy(value):
            self.values.append(float(value))
            return real(value)
        monkeypatch.setattr(sources, "pkd_from_log10_nm", spy)


def raw_value(index, replicate):
    return 9.0 - (7.0 + 0.1 * index + 0.02 * replicate)


# --- the restored membership ------------------------------------------------------------------

def test_the_frozen_world_loads_and_builds_no_model(world):
    loaded, context = world.load()
    assert world.loads == []
    assert loaded.membership.counts == world.counts
    assert loaded.membership.split_sha256 == world.policy["expected_split_sha256"]
    assert len(loaded.training_keys) == 16 and len(loaded.pool) == 16
    assert [row["window"] for row in loaded.panel] == list(world_lib.DEVELOPMENT)
    observed = loaded.validation["observations"]
    assert observed["observations"] == 36 and observed["populations"] == 18
    assert observed["her2_by_partition"] == {"development": 4, "train": 32}
    assert loaded.validation["summaries"] == {"summaries": 18,
                                              "by_partition": {"development": 2, "train": 16}}
    assert sorted(loaded.services.condition_spec.assays) == [ASSAY]
    assert "typed loaded observations" in loaded.document()["boundary"]
    assert set(loaded.frozen) == {"split", "vocabulary", "inventory"}
    for record in loaded.frozen.values():
        assert len(record["file_sha256"]) == 64 and len(record["document_sha256"]) == 64
    # The survivor scope is stated, with the duplicate counts it rests on (equal here: the world
    # has no duplicate evidence), and the bin policy is recorded with what it froze.
    assert observed["validation_scope"] == "post_legacy_closure_typed_survivors"
    assert observed["discarded_duplicate_evidence_validated"] is False
    assert observed["duplicate_observations_dropped_by_closure"] == {"current": 0, "frozen": 0}
    assert loaded.bins["policy"] == "current_exact_training"
    assert loaded.bins["historical_replay"] is False
    assert loaded.bins["differences"]["populations_changed"] == 0
    assert loaded.document()["bin_policy"] == loaded.bins
    identity = loaded.bin_identity
    assert identity["policy"] == "current_exact_training" and identity["bins_assigned"] > 0
    assert len(identity["thresholds_sha256"]) == 64 and len(identity["support_sha256"]) == 64


def test_the_train_set_is_the_candidates_minus_everything_held_out_or_excluded(world):
    loaded, _ = world.load()
    membership = loaded.membership
    assert membership.train == frozenset(world_lib.lattice_cores())
    assert membership.development == world_lib.DEVELOPMENT
    assert membership.reserved == world_lib.RESERVED
    assert set(membership.assignment) == set(world.candidates)
    assert {core for core, kind in membership.assignment.items() if kind == split_lib.EXCLUDED} \
        == set(world_lib.EXCLUDED)


def test_no_split_builder_is_called(world, monkeypatch):
    def forbidden(name):
        def refuse(*args, **kwargs):
            raise AssertionError(f"{name} was called")
        return refuse
    for name in ("build_hard_split", "freeze_split", "select_batches", "select_batched",
                 "split_certificate"):
        monkeypatch.setattr(split_lib, name, forbidden(name))
    monkeypatch.setattr(services_mod.FlightServices, "freeze_hard_split",
                        forbidden("FlightServices.freeze_hard_split"))
    loaded, _ = world.load("no-builder")
    assert loaded.membership.counts == world.counts
    with pytest.raises(ValueError, match="never freezes"):
        loaded.services.freeze_hard_split()
    with pytest.raises(ValueError, match="never freezes"):
        loaded.services.freeze_hard_split(persist=False)
    source = Path(membership_lib.__file__).read_text(encoding="utf-8")
    called = {node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
              for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)}
    assert not called & {"build_hard_split", "freeze_split", "select_batches", "select_batched"}


def test_the_services_override_exactly_the_split_closure_and_freeze():
    own = {name for name, value in vars(inputs_lib.FrozenInputServices).items()
           if callable(value) and not name.startswith("__")}
    assert own == {"adopt", "bin_assignments", "split", "closure", "freeze_hard_split"}


def test_a_services_object_with_no_adopted_split_never_derives_one(world):
    roots = world.reserve("bare")
    context = world.context(roots, world.pilot())
    services = inputs_lib.FrozenInputServices(context, device="cpu")
    with pytest.raises(ValueError, match="adopt"):
        services.split()
    with pytest.raises(ValueError, match="adopt"):
        services.closure()


def test_reserved_and_excluded_outcomes_are_never_converted(world, monkeypatch):
    spy = Spy(monkeypatch)
    loaded, _ = world.load("gate")
    allowed = {raw_value(index, replicate) for index in range(18) for replicate in range(2)}
    held_back = {raw_value(index, replicate) for index in (18, 19, 20) for replicate in range(2)}
    assert set(spy.values) == allowed and not set(spy.values) & held_back
    windows = {row.construct.residues(serial.her2_core_edit_map())
               for row in loaded.services.observations()}
    assert windows == set(world_lib.lattice_cores()) | set(world_lib.DEVELOPMENT)


@pytest.mark.parametrize("key", ["expected_split_sha256", "expected_sequence_sha256"])
def test_a_tampered_pin_is_refused_before_any_outcome_or_model(world, monkeypatch, key):
    spy = Spy(monkeypatch)
    with pytest.raises(ValueError, match="pins"):
        world.load("tampered", policy=dict(world.policy, **{key: "0" * 64}))
    assert spy.values == [] and world.loads == []


def test_a_changed_frozen_record_no_longer_matches_the_pinned_digest(world, monkeypatch):
    spy = Spy(monkeypatch)
    record = copy.deepcopy(world.record)
    record["development"] = list(reversed(record["development"]))
    paths.write_json(world.frozen / "hard_split_freeze.json", record)
    with pytest.raises(ValueError, match="split record digests"):
        world.load("changed")
    assert spy.values == [] and world.loads == []


def test_changed_expected_counts_are_refused_before_any_outcome(world, monkeypatch):
    spy = Spy(monkeypatch)
    pilot = world.pilot(world.write_pilot(
        "counts", lambda doc: doc["inputs"]["expected_counts"].update(excluded=3)))
    with pytest.raises(ValueError, match="counts"):
        world.load("counts", pilot=pilot)
    assert spy.values == [] and world.loads == []


def restore(world, record, **changes):
    arguments = dict(record_sha256=paths.digest_document(record), candidates=world.candidates,
                     expected_split_sha256=paths.digest_document(record),
                     expected_sequence_sha256=split_lib.sequence_digest(world.candidates),
                     split_settings=world.settings, expected_counts=world.counts)
    arguments.update(changes)
    return membership_lib.restore_membership(record, **arguments)


def tampered(world, mutate):
    record = copy.deepcopy(world.record)
    mutate(record)
    return record


def test_the_untouched_record_restores(world):
    membership = restore(world, copy.deepcopy(world.record))
    assert membership.counts == world.counts and len(membership.train) == 16


TAMPERINGS = {
    "development and reserved overlap": (
        lambda record: record.update(reserved=[world_lib.DEVELOPMENT[0]]),
        "both development and reserved"),
    "a repeated development window": (
        lambda record: record.update(development=[world_lib.DEVELOPMENT[0]] * 2),
        "repeats a window"),
    "a window that is not a candidate": (
        lambda record: record.update(development=["WWWWWWWWWW", world_lib.DEVELOPMENT[1]]),
        "not candidates"),
    "a window of the wrong length": (
        lambda record: record.update(reserved=["EEEE"]), "10-residue window"),
    "an excluded held-out window": (
        lambda record: record["excluded"].update(
            within_2_edits_of_a_held_out_core=[world_lib.DEVELOPMENT[0]]),
        "contains a development or reserved window"),
    "a window excluded for two reasons": (
        lambda record: record["excluded"].update(another=[world_lib.EXCLUDED[0]]),
        "repeats a window of another reason"),
    "a recorded count that the sets do not give": (
        lambda record: record["counts"].update(development=3), "counts\\['development'\\]"),
    "a recorded purge count that the sets do not give": (
        lambda record: record["counts"].update(purged_within_radius=7), "purged_within_radius"),
    "changed split settings": (
        lambda record: record["settings"].update(purge_radius=3), "split settings differ"),
    "a missing key": (lambda record: record.pop("selection"), "missing"),
    "a record sequence digest that is not the pinned one": (
        lambda record: record.update(sequence_sha256="0" * 64), "record's sequence digest"),
    "a train window moved into exclusion": (
        lambda record: record["excluded"]["within_2_edits_of_a_held_out_core"].append(
            world_lib.lattice_cores()[0]), "restored membership counts"),
}


@pytest.mark.parametrize("name", sorted(TAMPERINGS))
def test_a_tampered_record_is_refused_even_when_its_digest_is_re_pinned(world, name):
    mutate, message = TAMPERINGS[name]
    with pytest.raises(ValueError, match=message):
        restore(world, tampered(world, mutate))


def test_a_candidate_set_that_is_not_the_pinned_one_is_refused(world):
    with pytest.raises(ValueError, match="candidate windows digest"):
        restore(world, copy.deepcopy(world.record), candidates=world.candidates[:-1])


def test_the_closure_must_agree_with_the_record_for_every_train_window(world):
    """A near neighbour of a held-out core left in train means the record and the radius disagree."""
    near = world_lib.EXCLUDED[0]
    record = tampered(world, lambda value: (
        value["excluded"].update(within_2_edits_of_a_held_out_core=[world_lib.EXCLUDED[1]]),
        value["counts"].update(retained_for_training=17, purged_within_radius=1, excluded=1)))
    counts = dict(world.counts, excluded=1, retained_for_training=17)
    membership = restore(world, record, expected_counts=counts)
    assert near in membership.train
    scaffold = world_lib.fake_scaffold()
    with pytest.raises(ValueError, match="do not clear the closure"):
        membership_lib.build_closure(membership, scaffold=scaffold)


def test_the_closure_guards_the_held_out_windows_their_identities_and_their_neighbours(world):
    loaded, _ = world.load()
    closure, scaffold = loaded.closure, world.scaffold
    assert closure.radius == 2
    for core in (*world_lib.DEVELOPMENT, *world_lib.RESERVED):
        assert closure.violates(window=core) is not None
        assert closure.violates(identity=serial.her2_construct(scaffold, core).identity())
    assert closure.violates(window=world_lib.EXCLUDED[0]) is not None   # one edit from held out
    assert all(closure.violates(window=core) is None for core in world_lib.lattice_cores())


# --- validation of the typed observations -----------------------------------------------------

def observation(world, core, *, partition="train", parent="p", family="affinity",
                scale=sources.SCALE_PKD, replicate=None, measurement="a", sites=None,
                source=inputs_lib.HER2_SOURCE, window=None, value=7.0, kind="point",
                ordinal_class=None, censoring="none"):
    heavy = world.scaffold.with_core(core)
    construct = serial.Construct(heavy=heavy, light=world.scaffold.light,
                                 sites=sites or serial.her2_core_edit_map(), source=source,
                                 record_id=measurement)
    return sources.Observation(
        construct=construct, source=source, family=family, kind=kind, target="HER2",
        assay=source, parent=parent, scale=scale, value=value, interval=(value, value),
        censoring=censoring, ordinal_class=ordinal_class, replicate=replicate,
        raw={"partition": partition, "window": window or core, "measurement_id": measurement})


@pytest.fixture
def membership(world):
    return restore(world, copy.deepcopy(world.record))


def test_consistent_observations_pass_and_are_counted(world, membership):
    train = world_lib.lattice_cores()[0]
    rows = [observation(world, train, measurement="a"), observation(world, train, measurement="b"),
            observation(world, world_lib.DEVELOPMENT[0], partition="development")]
    report, populations = inputs_lib.validate_observations(rows, membership)
    assert populations == 2 and report["observations"] == 3
    assert report["her2_by_partition"] == {"train": 2, "development": 1}


@pytest.mark.parametrize("change, field", [
    ({"parent": "other"}, "parent"), ({"scale": sources.SCALE_ENRICHMENT}, "scale"),
    ({"partition": "development"}, "partition"), ({"sites": (("H", 98), ("H", 99))}, "region")])
def test_one_population_must_agree_on_its_metadata_partition_and_region(
        world, membership, change, field):
    train = world_lib.lattice_cores()[0]
    rows = [observation(world, train, measurement="a"),
            observation(world, train, measurement="b", **change)]
    with pytest.raises(ValueError, match="disagree within one population") as raised:
        inputs_lib.validate_observations(rows, membership)
    assert field in str(raised.value)


def test_a_repeated_observation_is_refused(world, membership):
    train = world_lib.lattice_cores()[0]
    rows = [observation(world, train, measurement="a"), observation(world, train, measurement="a")]
    with pytest.raises(ValueError, match="duplicate"):
        inputs_lib.validate_observations(rows, membership)


@pytest.mark.parametrize("core, partition, window", [
    (world_lib.RESERVED[0], "train", None),                    # a reserved window was materialised
    (world_lib.EXCLUDED[0], "train", None),                    # an excluded window was materialised
    (world_lib.DEVELOPMENT[0], "train", None),                 # a development window labelled train
    (world_lib.lattice_cores()[1], "development", None),       # a train window labelled development
    (world_lib.lattice_cores()[1], "train", world_lib.WILD)])  # the row's own window disagrees
def test_a_her2_row_must_sit_where_the_restored_split_put_its_window(
        world, membership, core, partition, window):
    rows = [observation(world, core, partition=partition, window=window)]
    with pytest.raises(ValueError, match="disagree with the restored split"):
        inputs_lib.validate_observations(rows, membership)


def test_rows_of_other_sources_do_not_consult_the_her2_split(world, membership):
    rows = [observation(world, world_lib.RESERVED[0], source="some_other_source")]
    report, _ = inputs_lib.validate_observations(rows, membership)
    assert report["her2_rows"] == 0


# --- measurement kind and ordinal class of the survivors ---------------------------------------------

@pytest.mark.parametrize("first, second, differs", [
    ({"kind": "interval"}, {"kind": "ordinal", "ordinal_class": 1}, ["kind", "ordinal_class"]),
    ({"kind": "point"}, {"kind": "interval"}, ["kind"]),
    ({"kind": "interval"}, {"kind": "point"}, ["kind"]),
    ({"kind": "ordinal", "ordinal_class": 0}, {"kind": "ordinal", "ordinal_class": 2},
     ["ordinal_class"])])
def test_one_population_must_agree_on_its_measurement_kind_and_ordinal_class(
        world, membership, first, second, differs):
    """The summary keeps the first record's kind and class; a mixed population would train on it."""
    train = world_lib.lattice_cores()[0]
    rows = [observation(world, train, measurement="a", **first),
            observation(world, train, measurement="b", **second)]
    with pytest.raises(ValueError, match="disagree within one population") as raised:
        inputs_lib.validate_observations(rows, membership)
    for name in differs:
        assert name in str(raised.value)


def test_a_change_of_censoring_is_not_a_change_of_measurement_type(world, membership):
    """Exact and floor-censored replicates of one interval population are one measurement type."""
    train = world_lib.lattice_cores()[0]
    rows = [observation(world, train, measurement="a", kind="interval", censoring="none"),
            observation(world, train, measurement="b", kind="interval", censoring="left"),
            observation(world, train, measurement="c", kind="interval", censoring="right")]
    report, populations = inputs_lib.validate_observations(rows, membership)
    assert populations == 1 and report["observations"] == 3


@pytest.mark.parametrize("kwargs", [
    {"kind": "ordinal", "ordinal_class": None}, {"kind": "ordinal", "ordinal_class": -1},
    {"kind": "ordinal", "ordinal_class": True}, {"kind": "ordinal", "ordinal_class": 1.0},
    {"kind": "point", "ordinal_class": 1}, {"kind": "interval", "ordinal_class": 0},
    {"kind": "enrichment", "ordinal_class": 2}, {"kind": "mystery"}])
def test_the_ordinal_class_must_match_the_kind_and_the_kind_must_be_known(
        world, membership, kwargs):
    rows = [observation(world, world_lib.lattice_cores()[0], **kwargs)]
    with pytest.raises(ValueError, match="unknown measurement kind or an ordinal class"):
        inputs_lib.validate_observations(rows, membership)


@pytest.mark.parametrize("kwargs", [
    {"kind": "ordinal", "ordinal_class": 0}, {"kind": "ordinal", "ordinal_class": 2},
    {"kind": "point"}, {"kind": "interval"}, {"kind": "enrichment"}])
def test_every_known_kind_with_its_matching_class_passes(world, membership, kwargs):
    rows = [observation(world, world_lib.lattice_cores()[0], **kwargs)]
    report, _ = inputs_lib.validate_observations(rows, membership)
    assert report["observations"] == 1


def test_the_validation_scope_is_stated_in_the_report(world, membership):
    rows = [observation(world, world_lib.lattice_cores()[0])]
    report, _ = inputs_lib.validate_observations(rows, membership)
    assert report["validation_scope"] == inputs_lib.SURVIVOR_SCOPE \
        == "post_legacy_closure_typed_survivors"
    assert report["discarded_duplicate_evidence_validated"] is False


def test_summaries_are_checked_against_the_same_split(world, membership):
    rows = [observation(world, world_lib.lattice_cores()[0], measurement="a"),
            observation(world, world_lib.DEVELOPMENT[0], partition="development")]
    summaries = data_lib.summarise_observations(rows)
    report = inputs_lib.validate_summaries(summaries, membership, populations=2)
    assert report == {"summaries": 2, "by_partition": {"development": 1, "train": 1}}
    with pytest.raises(ValueError, match="distinct populations"):
        inputs_lib.validate_summaries(summaries, membership, populations=3)
    broken = copy.deepcopy(summaries)
    broken[0].partition = "unlisted"
    with pytest.raises(ValueError, match="unknown partition"):
        inputs_lib.validate_summaries(broken, membership, populations=2)
    wrong = copy.deepcopy(summaries)
    wrong[0].partition = "development" if wrong[0].partition == "train" else "train"
    with pytest.raises(ValueError, match="differs from the restored split"):
        inputs_lib.validate_summaries(wrong, membership, populations=2)
    regionless = copy.deepcopy(summaries)
    regionless[0].region = ()
    with pytest.raises(ValueError, match="region differs"):
        inputs_lib.validate_summaries(regionless, membership, populations=2)


def inject(monkeypatch, change):
    """Wrap the real HER2 loader so the observations it returns carry ``change``."""
    real = sources.load_alphaseq_observations

    def wrapped(*args, **kwargs):
        block = real(*args, **kwargs)
        observations = list(block["observations"])
        observations = change(observations) or observations
        return dict(block, observations=tuple(observations))
    monkeypatch.setattr(sources, "load_alphaseq_observations", wrapped)


def test_a_partition_conflict_in_the_loaded_rows_stops_the_run_before_any_model(
        world, monkeypatch):
    """Swap one train row's partition with one development row's: the vocabulary counts (train rows
    per assay) are unchanged, so the refusal comes from the population check, not from the counts."""
    def swap(observations):
        train = next(item for item in observations if item.raw["partition"] == "train")
        development = next(item for item in observations
                           if item.raw["partition"] == "development")
        train.raw["partition"], development.raw["partition"] = "development", "train"
    inject(monkeypatch, swap)
    with pytest.raises(ValueError, match="disagree within one population"):
        world.load("partition-conflict")
    assert world.loads == []


def test_a_metadata_conflict_in_the_loaded_rows_stops_the_run_before_any_model(
        world, monkeypatch):
    def second_parent(observations):
        first = observations[0]
        return observations + [dataclasses.replace(
            first, parent="another_parent",
            raw=dict(first.raw, measurement_id="extra", partition="development"))]
    inject(monkeypatch, second_parent)
    with pytest.raises(ValueError, match="disagree within one population") as raised:
        world.load("metadata-conflict")
    assert "parent" in str(raised.value)
    assert world.loads == []


def test_a_row_in_a_reserved_window_stops_the_run(world, monkeypatch):
    def reserved_row(observations):
        first = observations[0]
        core = world_lib.RESERVED[0]
        construct = serial.Construct(
            heavy=world.scaffold.with_core(core), light=world.scaffold.light,
            sites=first.construct.sites, source=first.source, record_id="leak")
        return observations + [dataclasses.replace(
            first, construct=construct, raw=dict(first.raw, window=core, partition="development",
                                                 measurement_id="leak"))]
    inject(monkeypatch, reserved_row)
    with pytest.raises(ValueError, match="disagree with the restored split"):
        world.load("reserved-leak")
    assert world.loads == []


# --- the frozen vocabulary and inventory -------------------------------------------------------

def rewrite(path, mutate):
    document = paths.read_json(path)
    mutate(document)
    paths.write_json(path, document)


def test_the_frozen_vocabulary_is_the_one_in_use(world):
    loaded, context = world.load("vocabulary")
    frozen = world.frozen / "condition_vocabulary.json"
    assert (context.run_root / "condition_vocabulary.json").read_bytes() == frozen.read_bytes()
    spec = loaded.services.condition_spec
    assert spec.assay_order == (ASSAY, "null") and spec.sizes["assay"] == 2


@pytest.mark.parametrize("mutate", [
    lambda document: document["counts"].update({ASSAY: document["counts"][ASSAY] + 1}),
    lambda document: document.update(assays=[ASSAY, "a_spurious_assay"]),
    lambda document: document.update(assays=[])])
def test_a_derived_vocabulary_that_differs_is_refused_not_resized(world, mutate):
    rewrite(world.frozen / "condition_vocabulary.json", mutate)
    with pytest.raises(ValueError):
        world.load("vocabulary-changed")
    assert world.loads == []


POLICY = "current_exact_training"


@pytest.fixture(scope="module")
def derived(tmp_path_factory):
    """The inventory the REAL services derive over the synthetic world, as plain JSON.

    Built once; every test deep-copies it, so the comparison tests are pure data and fast.
    """
    with pytest.MonkeyPatch.context() as patch:
        built = world_lib.build_world(tmp_path_factory.mktemp("inventory"), patch)
        return json.loads((built.frozen / "source_inventory.json").read_text(encoding="utf-8"))


def changed(derived, mutate):
    document = copy.deepcopy(derived)
    mutate(document)
    return document


def compare(new, frozen, policy=POLICY):
    return inputs_lib.compare_inventory(new, frozen, bin_policy=policy)


def sources_row(document, **changes):
    document["sources"][0].update(changes)


def test_the_world_inventory_is_the_services_own_complete_output(derived):
    assert derived and inputs_lib.inventory_structure(derived, label="derived", derived=True) == []
    assert derived["sources"][0]["source"] == inputs_lib.HER2_SOURCE
    assert len(derived["sources"][0]["sha256"]) == 64
    assert derived["display_library"] is None and "bin_threshold_support" in derived
    assert derived["paired_oas"]["sha256"] == world_lib.OAS_SHA256
    problems, evidence = compare(derived, copy.deepcopy(derived))
    assert problems == [] and evidence["differences"]["populations_changed"] == 0


#: name -> (mutation of the FROZEN inventory, text the refusal must contain). Each field is one the
#: old comparison either skipped (it only compared what the frozen record happened to carry) or
#: compared conditionally.
MUTATIONS = {
    "summaries": (lambda d: d.update(summaries=d["summaries"] + 1), "summaries"),
    "training_summaries": (lambda d: d.update(training_summaries=d["training_summaries"] + 1),
                           "training_summaries"),
    "partitions": (lambda d: d["partitions"].update(train=1), "partitions.train"),
    "source sha256": (lambda d: sources_row(d, sha256="0" * 64), "sha256"),
    "source observations": (lambda d: sources_row(d, observations=1), "observations"),
    "source unique_constructs": (lambda d: sources_row(d, unique_constructs=1),
                                 "unique_constructs"),
    "source rows_in_file": (lambda d: sources_row(d, rows_in_file=d["sources"][0]["rows_in_file"]
                                                  + 1), "rows_in_file"),
    "source excluded": (lambda d: sources_row(d, excluded={"x": 1}), "excluded"),
    "source missing_outcome": (lambda d: sources_row(d, missing_outcome=99), "missing_outcome"),
    "source unalignable": (lambda d: sources_row(d, unalignable=99), "unalignable"),
    "source family": (lambda d: sources_row(d, family="other"), "family"),
    "source target": (lambda d: sources_row(d, target="other"), "target"),
    "source enrolled": (lambda d: sources_row(d, enrolled=False), "enrolled"),
    "source measurement_kind": (lambda d: sources_row(d, measurement_kind="other"),
                                "measurement_kind"),
    "closure kept": (lambda d: d["closure"].update(kept=1), "closure.kept"),
    "closure duplicates": (lambda d: d["closure"].update(duplicates=8191), "closure.duplicates"),
    "closure unique": (lambda d: d["closure"].update(unique_constructs=1),
                       "closure.unique_constructs"),
    "closure shared": (lambda d: d["closure"].update(constructs_in_multiple_sources=4),
                       "closure.constructs_in_multiple_sources"),
    "totals observations": (lambda d: d["totals"].update(observations=1), "totals.observations"),
    "totals unique": (lambda d: d["totals"].update(unique_constructs=1),
                      "totals.unique_constructs"),
    "vocabulary counts": (lambda d: d["assay_vocabulary"].update(counts={ASSAY: 1}),
                          "assay_vocabulary.counts"),
    "vocabulary assays": (lambda d: d["assay_vocabulary"].update(assays=[ASSAY, "extra"]),
                          "assay_vocabulary.assays"),
    "condition vocabulary counts": (lambda d: d["condition_vocabulary"].update(counts={ASSAY: 1}),
                                    "condition_vocabulary.counts"),
    "condition vocabulary assays": (lambda d: d["condition_vocabulary"].update(
        assays=["extra", ASSAY]), "condition_vocabulary.assays"),
    "heldout label": (lambda d: d["declared_heldouts"].update(partition_label="other"),
                      "declared_heldouts.partition_label"),
    "heldout parents": (lambda d: d["declared_heldouts"].update(withheld_parents=["p"]),
                        "declared_heldouts.withheld_parents"),
    "heldout studies": (lambda d: d["declared_heldouts"].update(withheld_studies=["s"]),
                        "declared_heldouts.withheld_studies"),
    "heldout rows moved": (lambda d: d["declared_heldouts"].update(
        rows_moved={"parent": 3, "study": 0}), "declared_heldouts.rows_moved"),
    "kinds by kind": (lambda d: d["measurement_kinds"]["by_kind"].update(ordinal=5),
                      "measurement_kinds.by_kind"),
    "censoring": (lambda d: d["measurement_kinds"]["by_censoring"].update(
        left=d["measurement_kinds"]["by_censoring"].get("left", 0) + 1),
        "measurement_kinds.by_censoring"),
    "oas sha256": (lambda d: d["paired_oas"].update(sha256="1" * 64), "paired_oas.sha256"),
    "oas member": (lambda d: d["paired_oas"].update(member="other.jsonl.gz"), "paired_oas.member"),
    "oas seed": (lambda d: d["paired_oas"].update(seed=8), "paired_oas.seed"),
    "oas limit": (lambda d: d["paired_oas"].update(limit_per_split=5),
                  "paired_oas.limit_per_split"),
    "oas splits": (lambda d: d["paired_oas"].update(splits=["train"]), "paired_oas.splits"),
    "oas rows": (lambda d: d["paired_oas"].update(rows_in_file=5), "paired_oas.rows_in_file"),
    "oas seen": (lambda d: d["paired_oas"].update(rows_seen_by_split={"train": 5}),
                 "paired_oas.rows_seen_by_split"),
    "oas excluded": (lambda d: d["paired_oas"].update(excluded={"x": 1}), "paired_oas.excluded"),
    "oas other splits": (lambda d: d["paired_oas"].update(other_splits_present={"test": 1}),
                         "paired_oas.other_splits_present"),
    "oas studies": (lambda d: d["paired_oas"].update(studies=3), "paired_oas.studies"),
    "oas by split": (lambda d: d["paired_oas_by_split"].update(train=3), "paired_oas_by_split"),
    "region": (lambda d: d["region_geometry"].update(extra=[["H", 1]]), "region_geometry.extra"),
    "supersession added": (lambda d: d["supersessions"].update(
        ghost={"superseded_by": "x", "study": "s"}), "supersessions.ghost"),
    "parent sequence": (lambda d: d["parent_sequences"].append("ghost::parental"),
                        "parent_sequences"),
    "display library": (lambda d: d.update(display_library={
        "rows": 3, "split": "train", "source": "her2_display", "assay": "a", "ordinal_classes": 3,
        "by_class": {"0": 1, "1": 1, "2": 1}, "positives_only_would_see": 1, "rows_in_split": 3,
        "refused_by_closure": 0, "positives_only": False}), "display_library"),
}


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_every_scientific_field_of_the_frozen_inventory_is_compared(derived, name):
    mutate, message = MUTATIONS[name]
    problems, evidence = compare(derived, changed(derived, mutate))
    assert problems and any(message in problem for problem in problems), problems
    assert all("frozen" in problem for problem in problems)
    assert evidence is None or "historical_replay" in evidence


def test_the_projection_is_every_required_block_except_the_bin_fields(derived):
    """What is compared is what is required; the bin fields are governed by the policy instead."""
    assert set(inputs_lib.scientific_projection(derived)) == (
        set(inputs_lib.INVENTORY_SHAPE) - set(inputs_lib.BIN_FIELDS)) | {"display_library"}
    assert set(inputs_lib.BIN_FIELDS) == {"bin_thresholds", "bin_threshold_support",
                                          "bins_assigned"}
    nested = {name: {field for field, _ in fields}
              for name, (_, fields) in inputs_lib.INVENTORY_SHAPE.items() if fields}
    projected = inputs_lib.scientific_projection(derived)
    for name, fields in nested.items():
        assert set(projected[name]) == fields, name


# --- completeness: nothing compares nothing --------------------------------------------------------

TOP_LEVEL = sorted(inputs_lib.INVENTORY_SHAPE) + ["display_library"]
NESTED = sorted((name, field) for name, (_, nested) in inputs_lib.INVENTORY_SHAPE.items()
                for field, _ in nested)


def test_an_empty_frozen_inventory_names_every_required_field(derived):
    problems, evidence = compare(derived, {})
    assert evidence is None and problems
    listed = inputs_lib.inventory_structure({}, label="frozen", derived=False)
    for name in TOP_LEVEL:
        assert any(f"{name} is missing" in problem for problem in listed), name
    assert inputs_lib.inventory_structure([], label="frozen", derived=False) == [
        "frozen is not an object"]


@pytest.mark.parametrize("name", TOP_LEVEL)
def test_a_frozen_inventory_without_a_required_field_is_refused(derived, name):
    problems, evidence = compare(derived, changed(derived, lambda d: d.pop(name)))
    assert evidence is None
    assert any("the frozen inventory" in problem and f"{name} is missing" in problem
               for problem in problems), problems


@pytest.mark.parametrize("name", TOP_LEVEL)
def test_a_derived_inventory_without_a_required_field_is_refused(derived, name):
    problems, evidence = compare(changed(derived, lambda d: d.pop(name)), derived)
    assert evidence is None
    assert any("the derived inventory" in problem and f"{name} is missing" in problem
               for problem in problems), problems


@pytest.mark.parametrize("name, field", NESTED)
def test_a_frozen_block_without_a_required_nested_field_is_refused(derived, name, field):
    problems, evidence = compare(derived, changed(derived, lambda d: d[name].pop(field)))
    assert evidence is None
    assert any(f"{name}.{field} is missing" in problem for problem in problems), problems


@pytest.mark.parametrize("name", sorted({name for name, _ in NESTED}))
def test_a_frozen_block_emptied_to_braces_is_refused_not_skipped(derived, name):
    problems, evidence = compare(derived, changed(derived, lambda d: d.update({name: {}})))
    assert evidence is None and any(f"{name}." in problem for problem in problems), problems


@pytest.mark.parametrize("name, value", [
    ("summaries", "18"), ("summaries", True), ("sources", {}), ("partitions", []),
    ("closure", []), ("parent_sequences", {}), ("supersessions", []), ("bins_assigned", 1.5),
    ("bin_thresholds", [])])
def test_a_required_field_of_the_wrong_kind_is_refused(derived, name, value):
    problems, evidence = compare(derived, changed(derived, lambda d: d.update({name: value})))
    assert evidence is None and any(f"{name} must be" in problem for problem in problems)


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d["sources"][0].pop("sha256"), "lacks ['sha256']"),
    (lambda d: d["sources"][0].pop("excluded"), "lacks ['excluded']"),
    (lambda d: sources_row(d, sha256=None), "no SHA-256"),
    (lambda d: sources_row(d, sha256="abc"), "no SHA-256"),
    (lambda d: sources_row(d, sha256="A" * 64), "no SHA-256"),
    (lambda d: sources_row(d, source=""), "no source name"),
    (lambda d: sources_row(d, source=None), "no source name"),
    (lambda d: sources_row(d, observations="3"), "observations must be an integer"),
    (lambda d: d["sources"].append(copy.deepcopy(d["sources"][0])), "repeat"),
    (lambda d: d["sources"].__setitem__(0, "not a row"), "is not an object"),
    (lambda d: d.update(sources=[]), "sources is empty")])
def test_source_rows_must_be_complete_named_unique_and_hashed(derived, mutate, message):
    for new, frozen in ((derived, changed(derived, mutate)), (changed(derived, mutate), derived)):
        problems, evidence = compare(new, frozen)
        assert evidence is None and any(message in problem for problem in problems), problems


def test_source_membership_is_exact_and_order_does_not_matter(derived):
    def add(document):
        document["sources"].append({**document["sources"][0], "source": "second_source"})
    two = changed(derived, add)
    problems, _ = compare(two, derived)
    assert any("sources.second_source was produced now" in problem for problem in problems)
    problems, _ = compare(derived, two)
    assert any("sources.second_source is in the frozen inventory and was not produced now"
               in problem for problem in problems)
    reordered = changed(two, lambda d: d["sources"].reverse())
    assert compare(two, reordered)[0] == [] and compare(reordered, two)[0] == []
    renamed = changed(two, lambda d: d["sources"][1].update(source="third_source"))
    problems, _ = compare(two, renamed)
    assert len(problems) == 2


def test_nullable_source_metadata_is_a_value_not_an_invalid_record(derived):
    """A source with no enrolled spec (the historical Petersen S1 alias) carries None metadata."""
    def add(document):
        document["sources"].append({
            **document["sources"][0], "source": "PetersenS1", "enrolled": None, "family": None,
            "target": None, "measurement_kind": None, "sha256": "b" * 64})
    both = changed(derived, add)
    assert inputs_lib.inventory_structure(both, label="x", derived=True) == []
    assert compare(both, copy.deepcopy(both))[0] == []
    other = changed(both, lambda d: d["sources"][1].update(family="affinity"))
    problems, _ = compare(both, other)
    assert any("sources.PetersenS1.family" in problem for problem in problems)


def test_bin_threshold_support_is_required_of_the_derived_side_and_optional_historically(derived):
    problems, evidence = compare(changed(derived, lambda d: d.pop("bin_threshold_support")),
                                 derived)
    assert evidence is None and any("bin_threshold_support is missing" in p for p in problems)
    problems, evidence = compare(derived, changed(derived, lambda d: d.pop("bin_threshold_support")))
    assert problems == [] and evidence["historical"]["support_recorded"] is False
    problems, evidence = compare(derived, changed(
        derived, lambda d: d.update(bin_threshold_support=[])))
    assert evidence is None and any("bin_threshold_support must be dict" in p for p in problems)


def test_the_display_library_is_present_and_compared_when_enrolled(derived):
    library = {"rows": 3, "split": "train", "source": "her2_display", "assay": "a",
               "ordinal_classes": 3, "by_class": {"0": 1, "1": 1, "2": 1},
               "positives_only_would_see": 1, "rows_in_split": 3, "refused_by_closure": 0,
               "positives_only": False}
    enrolled = changed(derived, lambda d: d.update(display_library=library))
    assert compare(enrolled, copy.deepcopy(enrolled))[0] == []
    other = changed(enrolled, lambda d: d["display_library"].update(rows=4))
    problems, _ = compare(enrolled, other)
    assert any("display_library.rows" in problem for problem in problems)
    assert any("display_library" in problem for problem in compare(derived, enrolled)[0])
    missing = changed(enrolled, lambda d: d["display_library"].pop("rows_in_split"))
    problems, evidence = compare(enrolled, missing)
    assert evidence is None and any("display_library.rows_in_split is missing" in p
                                    for p in problems)
    problems, evidence = compare(enrolled, changed(derived, lambda d: d.update(display_library=3)))
    assert evidence is None and any("display_library must be an object or None" in p
                                    for p in problems)


# --- what is ignored: prose and one operational flag, nothing else ---------------------------------

def test_reused_cache_alone_is_not_a_difference(derived):
    frozen = changed(derived, lambda d: d["paired_oas"].update(reused_cache=True))
    problems, evidence = compare(derived, frozen)
    assert problems == [] and evidence is not None
    assert compare(frozen, derived)[0] == []


def test_prose_and_detail_listings_are_the_only_other_fields_left_out(derived):
    def prose(document):
        document.update(note="a", ordering="b", partition_rule="c", threshold_rule="d")
        document["totals"]["note"] = "e"
        document["paired_oas"].update(sampling="f", outcome="g", top_studies={"z": 9})
        document["closure"].update(rule="h", duplicate_detail=[{"x": 1}],
                                   shared_examples={"k": ["a", "b"]})
        document["supersessions"]["ghost"] = {"superseded_by": "x", "study": "s", "reason": "r"}
    base = changed(derived, lambda d: d["supersessions"].update(
        ghost={"superseded_by": "x", "study": "s", "reason": "other words"}))
    assert compare(base, changed(derived, prose))[0] == []
    renamed = changed(base, lambda d: d["supersessions"]["ghost"].update(superseded_by="y"))
    problems, _ = compare(base, renamed)
    assert any("supersessions.ghost.superseded_by" in problem for problem in problems)
    dropped = changed(base, lambda d: d["supersessions"]["ghost"].pop("study"))
    assert any("supersessions.ghost.study" in problem for problem in compare(base, dropped)[0])


# --- the declared bin policy ---------------------------------------------------------------------------

def a_fitted_population(derived):
    key = next((key for key, value in sorted(derived["bin_thresholds"].items())
                if value is not None), None)
    assert key is not None, "the synthetic world fits no bin threshold"
    return key


def test_an_unsupported_bin_policy_is_refused_before_any_comparison(derived):
    for policy in ("historical_replay", "", None, "current"):
        with pytest.raises(ValueError, match="unsupported bin policy"):
            compare(derived, derived, policy)


def test_historical_bin_drift_is_permitted_and_reported_under_the_declared_policy(derived):
    key = a_fitted_population(derived)

    def historical(document):
        document["bin_thresholds"][key]["lower"] -= 1.0       # the old row.value cuts
        document["bin_thresholds"][key]["upper"] -= 1.0
        document["bins_assigned"] += 5
        document.pop("bin_threshold_support")                 # never recorded historically
    problems, evidence = compare(derived, changed(derived, historical))
    assert problems == []
    differences = evidence["differences"]
    assert differences["populations_changed"] == 1 and differences["unlisted"] == 0
    listed = differences["listed"][0]
    assert listed["population"] == key
    assert listed["historical"]["lower"] == listed["current"]["lower"] - 1.0
    count = derived["bins_assigned"]
    assert differences["bins_assigned"] == {"historical": count + 5, "current": count,
                                           "difference": -5}
    assert evidence["historical"]["support_recorded"] is False
    assert evidence["current"]["support_recorded"] is True
    assert evidence["historical"]["thresholds_sha256"] != evidence["current"]["thresholds_sha256"]
    assert evidence["policy"] == POLICY and evidence["historical_replay"] is False
    assert "not a replay" in evidence["claim"]
    # row.value was the mean of the UNCENSORED replicates of summaries that also held censored
    # ones; the claim does not say the censored replicates were in the mean.
    assert "uncensored replicates" in evidence["claim"]
    assert "censored replicates included" not in evidence["claim"]


def test_the_frozen_bin_identity_is_the_current_one(derived):
    key = a_fitted_population(derived)
    frozen = changed(derived, lambda d: d["bin_thresholds"][key].update(lower=-9.0))
    _, evidence = compare(derived, frozen)
    identity = evidence["frozen_identity"]
    assert identity == {"policy": POLICY,
                        "thresholds_sha256": paths.digest_document(derived["bin_thresholds"]),
                        "support_sha256": paths.digest_document(derived["bin_threshold_support"]),
                        "bins_assigned": derived["bins_assigned"]}
    assert identity["thresholds_sha256"] == evidence["current"]["thresholds_sha256"]
    assert identity["thresholds_sha256"] != evidence["historical"]["thresholds_sha256"]


def test_populations_present_on_one_side_only_are_listed_not_dropped(derived):
    frozen = changed(derived, lambda d: d["bin_thresholds"].update({"affinity::ghost::*": None}))
    _, evidence = compare(derived, frozen)
    assert evidence["differences"]["populations_only_historical"] == ["affinity::ghost::*"]
    assert evidence["differences"]["populations_only_current"] == []
    _, evidence = compare(frozen, derived)
    assert evidence["differences"]["populations_only_current"] == ["affinity::ghost::*"]
    assert evidence["differences"]["populations_changed"] == 0


def test_bin_drift_never_excuses_a_difference_in_anything_else(derived):
    key = a_fitted_population(derived)

    def both(document):
        document["bin_thresholds"][key]["lower"] -= 1.0
        document["summaries"] += 1
    problems, evidence = compare(derived, changed(derived, both))
    assert any(problem.startswith("summaries") for problem in problems)
    assert not any("bin" in problem for problem in problems)
    assert evidence is not None


# --- the actual bin assignments: fingerprint, record, identity --------------------------------------

def fingerprint(mapping, summaries, count=None):
    return inputs_lib.fingerprint_bin_assignments(
        mapping, summaries=summaries, expected_count=len(mapping) if count is None else count)


def test_the_assignment_fingerprint_is_the_actual_mapping_and_ignores_insertion_order(world):
    loaded, _ = world.load("assignments")
    services = loaded.services
    mapping, summaries = services.bin_assignments(), services.summaries()
    assert mapping == services._cache["bins"], "the getter reads the mapping the services use"
    assert len(mapping) == loaded.bins["current"]["bins_assigned"] > 1
    first = fingerprint(mapping, summaries)
    reordered = dict(reversed(list(mapping.items())))
    assert reordered == mapping and list(reordered) != list(mapping)
    assert fingerprint(reordered, summaries) == first
    identity = loaded.bin_identity
    assert identity["assignments_schema"] == inputs_lib.ASSIGNMENT_SCHEMA == first["schema"]
    assert identity["assignments_sha256"] == first["sha256"] and len(first["sha256"]) == 64
    assert identity["assignments_count"] == first["count"] == identity["bins_assigned"]
    assert sum(first["by_bin"].values()) == first["count"] == sum(first["by_partition"].values())
    assert set(first["by_bin"]) <= set(inputs_lib.ASSIGNED_BINS)
    services.bin_assignments().clear()
    assert len(services.bin_assignments()) == first["count"], "the getter returns a copy"


def test_a_label_swap_that_keeps_every_bin_count_changes_the_fingerprint(world):
    loaded, _ = world.load("swap")
    summaries = loaded.services.summaries()
    keys = [summary.population_key for summary in summaries]
    mapping = {key: ("low", "high")[index % 2] for index, key in enumerate(keys)}
    swapped = dict(mapping)
    swapped[keys[0]], swapped[keys[1]] = mapping[keys[1]], mapping[keys[0]]
    one, two = fingerprint(mapping, summaries), fingerprint(swapped, summaries)
    assert one["by_bin"] == two["by_bin"] and one["count"] == two["count"]
    assert one["sha256"] != two["sha256"]


def test_an_unknown_population_bin_or_count_is_refused(world):
    loaded, _ = world.load("refusals")
    mapping, summaries = loaded.services.bin_assignments(), loaded.services.summaries()
    key = next(iter(mapping))
    ghost = ("f" * 64, "ghost_study", "ghost_assay", "HER2")
    cases = [
        ("belong to no loaded summary", {**mapping, ghost: "mid"}, None),
        ("keyed by", {**mapping, key[:3]: "mid"}, None),
        ("keyed by", {**mapping, "not a tuple": "mid"}, None),
        ("outside", {**mapping, key: "null"}, None),
        ("outside", {**mapping, key: "medium"}, None),
        ("outside", {**mapping, key: None}, None),
        ("inventory reports", mapping, len(mapping) + 1),
        ("not a mapping", list(mapping.items()), None)]
    for message, bad, count in cases:
        with pytest.raises(ValueError, match=message):
            fingerprint(bad, summaries, count)


def test_an_assignment_without_a_loaded_summary_stops_the_load_before_any_model(
        world, monkeypatch):
    real = inputs_lib.FrozenInputServices.bin_assignments

    def with_ghost(self):
        return {**real(self), ("f" * 64, "ghost_study", "ghost_assay", "HER2"): "mid"}
    monkeypatch.setattr(inputs_lib.FrozenInputServices, "bin_assignments", with_ghost)
    with pytest.raises(ValueError, match="belong to no loaded summary"):
        world.load("ghost")
    assert world.loads == []


def test_the_current_bins_record_is_complete_written_once_and_bound_to_the_identity(world):
    loaded, context = world.load("bins-file")
    frozen = paths.read_json(world.frozen / "source_inventory.json")
    record, identity = loaded.current_bins, loaded.bin_identity
    assert record["bin_thresholds"] == frozen["bin_thresholds"]
    assert record["bin_threshold_support"] == frozen["bin_threshold_support"]
    assert record["policy"] == POLICY and record["historical_replay"] is False
    assert record["assignments"]["sha256"] == identity["assignments_sha256"]
    assert identity["current_bins_sha256"] == paths.digest_document(record)
    target = context.run_root / inputs_lib.CURRENT_BINS_FILE
    assert not target.exists(), "loading writes nothing; the runner persists the record"
    assert loaded.document()["current_bins_file"] == {}
    written = inputs_lib.write_current_bins(loaded, context.run_root)
    assert written == {"path": "current_bins.json", "file_sha256": paths.sha256_file(target),
                       "content_sha256": identity["current_bins_sha256"]}
    assert paths.read_json(target) == record
    assert loaded.document()["current_bins_file"] == written
    with pytest.raises(ValueError, match="already exists"):
        inputs_lib.write_current_bins(loaded, context.run_root)


def test_the_bound_record_keeps_every_cut_and_its_digest_follows_each_part(derived, tmp_path):
    new = changed(derived, lambda d: d["bin_thresholds"].update({"affinity::ghost::*": None}))
    _, evidence = compare(new, derived)
    assignments = {"schema": inputs_lib.ASSIGNMENT_SCHEMA, "count": new["bins_assigned"],
                   "sha256": "a" * 64}
    bound, record = inputs_lib.bind_current_bins(evidence, inventory=new, assignments=assignments)
    assert record["bin_thresholds"] == new["bin_thresholds"]
    assert record["bin_thresholds"]["affinity::ghost::*"] is None
    assert record["bin_threshold_support"] == new["bin_threshold_support"]
    target = tmp_path / "current_bins.json"
    digest = paths.write_json(target, record)
    assert paths.read_json(target) == record and digest == paths.digest_document(record)
    identity = bound["frozen_identity"]
    assert identity["current_bins_sha256"] == digest and identity["assignments_sha256"] == "a" * 64
    assert identity["assignments_count"] == new["bins_assigned"] == identity["bins_assigned"]
    assert identity["thresholds_sha256"] == paths.digest_document(new["bin_thresholds"])
    assert evidence["frozen_identity"].keys() < identity.keys(), "the original is not mutated"

    def record_digest(inventory, assigned=assignments):
        _, shifted = compare(inventory, derived)
        return paths.digest_document(inputs_lib.bind_current_bins(
            shifted, inventory=inventory, assignments=assigned)[1])
    key = a_fitted_population(new)
    cut = changed(new, lambda d: d["bin_thresholds"][key].update(
        lower=d["bin_thresholds"][key]["lower"] - 1.0))
    assert record_digest(cut) != digest, "a cut is part of the record"
    assert record_digest(new, dict(assignments, sha256="b" * 64)) != digest, \
        "the assignment digest is part of the record"
    assert record_digest(new) == digest, "the digest is a function of the content alone"
    with pytest.raises(ValueError, match="differ from the ones"):
        inputs_lib.bind_current_bins(evidence, inventory=cut, assignments=assignments)
    with pytest.raises(ValueError, match="bin evidence reports"):
        inputs_lib.bind_current_bins(evidence, inventory=new,
                                     assignments=dict(assignments, count=assignments["count"] + 1))


# --- the same comparison through load_inputs -----------------------------------------------------------

def load_expecting(world, name, message):
    """``load_inputs`` over a freshly reserved root, which must refuse with ``message``."""
    roots = world.reserve(name)
    context = world.context(roots, world.pilot())
    with pytest.raises(ValueError, match=message):
        inputs_lib.load_inputs(world.pilot(), context, policy=world.policy,
                               frozen_dir=world.frozen, device="cpu")
    return context


def test_an_empty_frozen_inventory_is_refused_before_the_vocabulary_or_any_source(
        world, monkeypatch):
    """There is no bootstrap mode: ``{}`` is incomplete, not "nothing to compare"."""
    spy = Spy(monkeypatch)

    def forbidden(self, *args, **kwargs):
        raise AssertionError("a source was opened for an incomplete frozen inventory")
    monkeypatch.setattr(services_mod.FlightServices, "build_sources", forbidden)
    paths.write_json(world.frozen / "source_inventory.json", {})
    context = load_expecting(world, "empty-inventory", "incomplete and nothing was loaded")
    assert list(context.run_root.iterdir()) == [], "the vocabulary was installed before the refusal"
    assert spy.values == [] and world.loads == []


@pytest.mark.parametrize("name", ["closure", "paired_oas", "declared_heldouts",
                                  "measurement_kinds", "region_geometry", "display_library",
                                  "assay_vocabulary", "totals"])
def test_a_frozen_inventory_missing_a_semantic_block_is_refused_before_any_source(
        world, monkeypatch, name):
    spy = Spy(monkeypatch)
    rewrite(world.frozen / "source_inventory.json", lambda document: document.pop(name))
    context = load_expecting(world, "missing-block", f"{name} is missing")
    assert list(context.run_root.iterdir()) == [] and spy.values == [] and world.loads == []


def test_an_emptied_semantic_block_is_refused_before_any_source(world, monkeypatch):
    spy = Spy(monkeypatch)
    rewrite(world.frozen / "source_inventory.json", lambda document: document.update(
        closure={}, paired_oas={}, measurement_kinds={}))
    context = load_expecting(world, "emptied", "closure.kept is missing")
    assert list(context.run_root.iterdir()) == [] and spy.values == []


@pytest.mark.parametrize("mutate, message", [
    (lambda document: document.update(summaries=document["summaries"] + 1), "summaries"),
    (lambda document: document["partitions"].update(train=1), "partitions.train"),
    (lambda document: document["sources"][0].update(sha256="0" * 64), "sha256"),
    (lambda document: document["sources"][0].update(observations=1), "observations"),
    (lambda document: document["closure"].update(kept=1), "closure.kept"),
    (lambda document: document["paired_oas"].update(sha256="1" * 64), "paired_oas.sha256"),
    (lambda document: document["paired_oas"].update(seed=99), "paired_oas.seed"),
    (lambda document: document["declared_heldouts"].update(withheld_parents=["p"]),
     "declared_heldouts.withheld_parents"),
    (lambda document: document["measurement_kinds"]["by_censoring"].update(left=3),
     "measurement_kinds.by_censoring"),
    (lambda document: document["region_geometry"].update(extra=[["H", 1]]),
     "region_geometry.extra"),
    (lambda document: document["sources"].append(dict(document["sources"][0], source="extra")),
     "sources.extra"),
    (lambda document: document["assay_vocabulary"].update(counts={ASSAY: 1}),
     "assay_vocabulary.counts")])
def test_the_loaded_sources_must_reproduce_the_frozen_inventory(world, mutate, message):
    rewrite(world.frozen / "source_inventory.json", mutate)
    load_expecting(world, "inventory-changed", message)
    assert world.loads == []


def test_reused_cache_and_prose_do_not_stop_a_load(world):
    def operational(document):
        document["paired_oas"]["reused_cache"] = True
        document["note"] = "prose"
        document["paired_oas"]["sampling"] = "other words"
    rewrite(world.frozen / "source_inventory.json", operational)
    loaded, _ = world.load("operational")
    assert loaded.bins["differences"]["populations_changed"] == 0


def test_historical_bins_load_under_the_declared_policy_and_the_current_ones_are_used(world):
    """The frozen thresholds are the old row.value cuts; the services fit the current ones."""
    current = paths.read_json(world.frozen / "source_inventory.json")
    key = a_fitted_population(current)

    def historical(document):
        document["bin_thresholds"][key]["lower"] -= 1.0
        document["bin_thresholds"][key]["upper"] -= 1.0
        document["bins_assigned"] += 5
        document.pop("bin_threshold_support")
    rewrite(world.frozen / "source_inventory.json", historical)
    loaded, _ = world.load("historical-bins")
    differences = loaded.bins["differences"]
    assert differences["populations_changed"] == 1
    assert differences["bins_assigned"]["difference"] == -5
    assert loaded.bins["historical_replay"] is False
    used = inputs_lib.plain(loaded.services._cache["source_inventory"])
    assert used["bin_thresholds"] == current["bin_thresholds"]
    assert loaded.bin_identity["thresholds_sha256"] == paths.digest_document(
        current["bin_thresholds"])
    assert loaded.document()["bin_policy"]["differences"]["listed"][0]["population"] == key
    assert len(loaded.services._cache["bins"]) == current["bins_assigned"]


def test_a_pilot_without_the_declared_policy_cannot_load(world):
    pilot = world.pilot()
    object.__setattr__(pilot, "bin_policy", "historical_replay")
    roots = world.reserve("no-policy")
    context = world.context(roots, pilot)
    with pytest.raises(ValueError, match="bin policy"):
        inputs_lib.load_inputs(pilot, context, policy=world.policy, frozen_dir=world.frozen,
                               device="cpu")
    assert list(context.run_root.iterdir()) == []


# --- freshness and cache isolation --------------------------------------------------------------

def test_caches_stay_inside_the_fresh_run_root_and_no_input_root_is_written(world):
    before = {name: digest_tree(getattr(world, name))
              for name in ("code", "data", "artifacts", "frozen")}
    loaded, context = world.load("cache")
    run_root = context.run_root
    assert loaded.services.source_cache_root() == run_root / "source_cache"
    assert (run_root / "source_cache").is_dir()
    assert not (world.artifacts / "cache").exists() and not (world.artifacts / "runs").exists()
    after = {name: digest_tree(getattr(world, name))
             for name in ("code", "data", "artifacts", "frozen")}
    assert after == before
    assert sorted(path.name for path in run_root.iterdir()) == [
        "condition_vocabulary.json", "source_cache"]


def test_a_run_root_is_used_once(world):
    world.load("once")
    with pytest.raises(ValueError, match="already exists"):
        world.load("once")


def test_every_declared_path_resolves_under_the_right_root(world):
    loaded, context = world.load("roots")
    services = loaded.services
    assert services.source_paths()[inputs_lib.HER2_SOURCE] == (
        world.artifacts / world_lib.SOURCE_LOGICAL)
    assert context.repository(context.config["dplm"]["pins"]) == (
        world.code / "configs" / "experiments" / "dplm_pins.json")
    assert context.repository(context.config["roots"]["her2_raw"]).is_relative_to(world.data)
    assert services.dplm_pins()["variants"]["dplm_150m"]["revision"] == "a" * 40


# --- source identities ----------------------------------------------------------------------------

def test_source_identities_are_verified_where_a_manifest_describes_them(world):
    loaded, _ = world.load("identities")
    record = loaded.identities
    assert record["manifest_files_verified"] == 1 and len(record["manifest_files_digest"]) == 64
    assert record["sources"] == [{"source": inputs_lib.HER2_SOURCE, "root": "artifact",
                                  "logical": world_lib.SOURCE_LOGICAL,
                                  "verification": "acquisition_manifest", "manifest_files": 1}]
    assert record["dplm"]["bound"] is True and record["dplm"]["dplm_variants"] == ["dplm_150m"]
    assert record["manifests"]["dplm_acquisition_manifest"]["present"] is True
    assert "no single marker" in record["note"]


def test_a_source_file_that_no_longer_matches_its_manifest_is_refused(world):
    with (world.artifacts / world_lib.SOURCE_LOGICAL).open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="hashes"):
        world.load("changed-source")
    assert world.loads == []


def test_a_dplm_manifest_that_disagrees_with_the_pins_is_refused(world):
    manifest = world.artifacts / "data" / "dplm_acquisition_manifest.json"
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["files"][0]["sha256"] = "1" * 64
    manifest.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="DPLM binding failed"):
        world.load("bad-dplm-manifest")


def test_a_missing_dplm_manifest_is_refused(world):
    (world.artifacts / "data" / "dplm_acquisition_manifest.json").unlink()
    with pytest.raises(ValueError, match="DPLM acquisition manifest"):
        world.load("no-dplm-manifest")


def add_manifest_entries(world, files):
    """Write ``{logical: bytes}`` under the artifact root and list each in the main manifest."""
    manifest = world.artifacts / "data" / "acquisition_manifest.json"
    document = json.loads(manifest.read_text(encoding="utf-8"))
    for logical, payload in files.items():
        target = world.artifacts / logical
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        document["files"].append({"path": logical, "sha256": paths.sha256_file(target),
                                  "bytes": len(payload)})
    manifest.write_text(json.dumps(document), encoding="utf-8")


def identity_context(world, name, block):
    context = world.context(world.reserve(name), world.pilot())
    context.config["sources"]["paths"].update(block)
    return context


def test_every_member_of_a_multi_file_source_is_verified_not_only_the_first(world):
    """The loaders hash only one member of some sources; the manifest loop covers all of them."""
    members = {"data/extra/pp489_a.csv": b"id,x\n1,2\n", "data/extra/pp489_b.csv": b"id,x\n3,4\n"}
    add_manifest_entries(world, members)
    block = {"extra_source": {"root": "artifact", "logicals": sorted(members)}}
    record = inputs_lib.verify_source_identities(identity_context(world, "members", block))
    rows = [row for row in record["sources"] if row["source"] == "extra_source"]
    assert [row["logical"] for row in rows] == sorted(members)
    assert all(row["verification"] == "acquisition_manifest" and row["manifest_files"] == 1
               for row in rows)
    assert record["manifest_files_verified"] == 3
    (world.artifacts / "data" / "extra" / "pp489_b.csv").write_bytes(b"id,x\n3,5\n")
    with pytest.raises(ValueError, match="hashes"):
        inputs_lib.verify_source_identities(identity_context(world, "members-changed", block))


def test_a_directory_source_expands_to_its_manifest_children_and_each_is_verified(world):
    children = {"data/dirsource/replicate_1.csv": b"a\n", "data/dirsource/replicate_2.csv": b"b\n"}
    add_manifest_entries(world, children)
    block = {"dir_source": {"root": "artifact", "logical": "data/dirsource"}}
    record = inputs_lib.verify_source_identities(identity_context(world, "directory", block))
    row = next(row for row in record["sources"] if row["source"] == "dir_source")
    assert row["verification"] == "acquisition_manifest" and row["manifest_files"] == 2
    assert record["manifest_files_verified"] == 3
    (world.artifacts / "data" / "dirsource" / "replicate_2.csv").write_bytes(b"changed\n")
    with pytest.raises(ValueError, match="hashes"):
        inputs_lib.verify_source_identities(identity_context(world, "directory-changed", block))


def test_a_source_no_manifest_describes_is_recorded_as_loader_digest_only(world):
    target = world.artifacts / "data" / "undescribed.csv"
    target.write_bytes(b"x\n")
    block = {"undescribed": {"root": "artifact", "logical": "data/undescribed.csv"}}
    record = inputs_lib.verify_source_identities(identity_context(world, "undescribed", block))
    row = next(row for row in record["sources"] if row["source"] == "undescribed")
    assert row["verification"] == "loader_digest_only" and row["manifest_files"] == 0
    assert record["manifest_files_verified"] == 1


# --- narrowing to the one route --------------------------------------------------------------------

def test_the_config_is_narrowed_to_the_150m_route_and_the_base_is_untouched(world):
    base = json.loads(world.base_path.read_text(encoding="utf-8"))
    assert base["routes"]["enabled"] == ["dplm_150m_masked", "dplm_650m_lora_masked"]
    narrowed = inputs_lib.narrowed_config(base)
    assert narrowed["routes"]["enabled"] == ["dplm_150m_masked"]
    assert "dplm_650m_lora_masked" not in narrowed["routes"]
    assert narrowed["routes"]["dplm_150m_masked"]["optimizer"]["name"] == "adafactor"
    assert "dplm_650m_lora_masked" in base["routes"]
    without = copy.deepcopy(base)
    del without["routes"]["dplm_150m_masked"]
    with pytest.raises(ValueError, match="does not enable"):
        inputs_lib.narrowed_config(without)


def test_a_missing_frozen_record_is_refused(world):
    (world.frozen / "source_inventory.json").unlink()
    with pytest.raises(ValueError, match="frozen inventory record"):
        world.load("missing-record")
    assert world.loads == []
