"""Outcome-free functional budget, schedule, replay and restricted measurements (M1)."""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import random
from collections import Counter
from fractions import Fraction

import pytest
import torch

from smallAntibodyGen.experiments.trastuzumab_poc import functional_data as fd
from smallAntibodyGen.experiments.trastuzumab_poc import functional_protocol as fp
from smallAntibodyGen.tests.test_trastuzumab_functional_data import standard_qualified, standard_world

SEED = "budget-seed-1"


def P(value):
    return fp.Measurement("point", value=value)


def I(lower, upper):
    return fp.Measurement("interval", lower=lower, upper=upper)


def LB(lower):
    return fp.Measurement("lower_bound", lower=lower)


def UB(upper):
    return fp.Measurement("upper_bound", upper=upper)


# ---------------------------------------------------------------------------
# Interval comparison
# ---------------------------------------------------------------------------

HIGHER_BETTER = [
    (P(1), P(2), "b_better"), (P(2), P(1), "a_better"), (P(1), P(1), "tie"),
    (P(0.0), P(-0.0), "tie"),
    (I(1, 2), I(3, 4), "b_better"), (I(3, 4), I(1, 2), "a_better"),
    (I(1, 2), I(2, 3), "unresolved"), (I(1, 3), I(2, 4), "unresolved"),
    (I(1, 4), I(2, 3), "unresolved"), (I(1, 2), I(1, 2), "unresolved"),
    (P(2), I(1, 3), "unresolved"), (P(2), I(2, 3), "unresolved"),
    (P(1), I(2, 3), "b_better"), (I(2, 3), P(1), "a_better"),
    (LB(10), P(5), "a_better"), (LB(10), P(10), "unresolved"), (LB(10), P(11), "unresolved"),
    (LB(10), LB(20), "unresolved"), (P(5), LB(10), "b_better"),
    (UB(5), P(10), "b_better"), (UB(5), P(5), "unresolved"), (UB(5), P(1), "unresolved"),
    (UB(5), UB(3), "unresolved"),
    (LB(10), UB(5), "a_better"), (UB(5), LB(10), "b_better"), (LB(10), UB(10), "unresolved"),
    (I(1, 2), LB(3), "b_better"), (I(1, 2), UB(0), "a_better"),
]
FLIP = {"a_better": "b_better", "b_better": "a_better", "tie": "tie", "unresolved": "unresolved"}


@pytest.mark.parametrize("a, b, expected", HIGHER_BETTER)
def test_compare_matrix_higher_better(a, b, expected):
    assert fp.compare(a, b, "higher_better") == expected


@pytest.mark.parametrize("a, b, expected", HIGHER_BETTER)
def test_compare_matrix_reverses_with_direction_and_is_antisymmetric(a, b, expected):
    assert fp.compare(a, b, "lower_better") == FLIP[expected]
    for direction in ("higher_better", "lower_better"):
        assert fp.compare(b, a, direction) == FLIP[fp.compare(a, b, direction)]


def test_compare_rejects_a_bad_direction_or_operand():
    with pytest.raises(ValueError):
        fp.compare(P(1), P(2), "up")
    with pytest.raises(TypeError):
        fp.compare(1.0, P(2), "higher_better")


@pytest.mark.parametrize("kwargs", [
    dict(kind="range", value=1.0), dict(kind="point"), dict(kind="point", value=1.0, lower=0.0),
    dict(kind="point", value=float("nan")), dict(kind="point", value=float("inf")),
    dict(kind="point", value=-float("inf")), dict(kind="point", value=True),
    dict(kind="point", value="1.0"), dict(kind="point", value=10 ** 400),
    dict(kind="interval", lower=1.0), dict(kind="interval", lower=2.0, upper=1.0),
    dict(kind="interval", lower=1.0, upper=1.0), dict(kind="interval", lower=float("-inf"), upper=1.0),
    dict(kind="lower_bound", lower=1.0, upper=2.0), dict(kind="lower_bound", lower=float("inf")),
    dict(kind="upper_bound"),
])
def test_invalid_measurements_are_rejected(kwargs):
    with pytest.raises(ValueError):
        fp.Measurement(**kwargs)


@pytest.mark.parametrize("document", [
    {}, {"kind": "point"}, {"kind": "point", "value": 1, "extra": 2},
    {"kind": "lower_bound", "lower": "Infinity"}, {"kind": "upper_bound", "upper": None},
    {"kind": "interval", "lower": 1}, "point", None,
    {"kind": "lower_bound", "lower": 1, "upper": 2}, {"kind": ["point"], "value": 1}])
def test_measurement_documents_are_strict(document):
    with pytest.raises(ValueError):
        fp.Measurement.from_document(document)


def test_measurements_round_trip_without_json_infinity():
    for measurement in (P(3), I(1.5, 2.5), LB(10), UB(-4)):
        document = measurement.document()
        assert fp.Measurement.from_document(json.loads(json.dumps(document, allow_nan=False))) == measurement
    assert isinstance(P(3).value, float) and P(-0.0).value == 0.0
    assert set(LB(10).document()) == {"kind", "lower"} and set(UB(1).document()) == {"kind", "upper"}


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

def _budget(n=8, seed=SEED):
    q = standard_qualified()
    return q, fp.select_functional_budget(q, n, seed)


def test_budgets_nest_as_prefixes_of_one_seeded_order():
    q, full = _budget()
    assert set(full.selected) == set(q.train_observations())
    for n in range(1, 9):
        assert fp.select_functional_budget(q, n, SEED).selected == full.selected[:n]
    keys = [hashlib.sha256(f"{SEED}\x00{g}\x00{c}".encode("utf-8")).hexdigest() for g, c in full.selected]
    assert keys == sorted(keys)
    assert fp.select_functional_budget(q, 8, "another-seed").selected != full.selected


def test_the_budget_does_not_depend_on_document_row_order():
    decl, records, doc = standard_world()
    shuffled = copy.deepcopy(doc)
    shuffled["constructs"].reverse()
    shuffled["groups"].reverse()
    other = fd.qualify_functional(fd.parse_functional_constructs(shuffled), decl, tuple(reversed(records)))
    assert fp.select_functional_budget(other, 8, SEED) == _budget()[1]


@pytest.mark.parametrize("n", [0, -1, True, 9, 2.0, "3", None])
def test_budget_size_must_be_a_positive_integer_within_the_train_observations(n):
    with pytest.raises(ValueError):
        fp.select_functional_budget(standard_qualified(), n, SEED)


@pytest.mark.parametrize("seed", ["", None, 5, "a\x00b"])
def test_budget_seed_must_be_nonempty_text(seed):
    with pytest.raises(ValueError):
        fp.select_functional_budget(standard_qualified(), 3, seed)
    with pytest.raises(TypeError):
        fp.select_functional_budget("qualified", 3, SEED)


def test_budget_identity_round_trips_through_json():
    _, budget = _budget()
    identity = fp.functional_budget_identity(budget)
    assert fp.validate_functional_budget_identity(json.loads(json.dumps(identity))) == budget
    assert "measurement" not in json.dumps(identity)


TAMPER = {
    "selection_digest": lambda i: i.update(selection_digest="0" * 64),
    "budget_n_up": lambda i: i.update(budget_n=9),
    "budget_n_down": lambda i: i.update(budget_n=7),
    "budget_n_bool": lambda i: i.update(budget_n=True),
    "seed": lambda i: i.update(budget_seed="another-seed"),
    "empty_seed": lambda i: i.update(budget_seed=""),
    "qualified_digest": lambda i: i.update(qualified_digest="1" * 64),
    "contract_digest": lambda i: i.update(contract_digest="2" * 64),
    "drop_one": lambda i: i["selected"].pop(),
    "duplicate_one": lambda i: i["selected"].__setitem__(1, list(i["selected"][0])),
    "rotate_order": lambda i: i["selected"].insert(0, i["selected"].pop()),
    "replace_id": lambda i: i["selected"][0].__setitem__(1, "t99"),
    "row_wrong_shape": lambda i: i["selected"].__setitem__(0, ["g-a"]),
    "selected_not_a_list": lambda i: i.update(selected="x"),
    "extra_key": lambda i: i.update(values=[1]),
    "missing_key": lambda i: i.pop("budget_seed"),
    "wrong_schema": lambda i: i.update(schema="trastuzumab-functional-budget/2"),
    "digest_not_hex": lambda i: i.update(selection_digest="zz"),
}


@pytest.mark.parametrize("name", sorted(TAMPER))
def test_tampered_budget_identities_are_rejected_without_any_document(name):
    identity = fp.functional_budget_identity(_budget()[1])
    TAMPER[name](identity)
    with pytest.raises(ValueError):
        fp.validate_functional_budget_identity(identity)


def test_a_forger_who_recomputes_the_digest_still_cannot_reorder_the_selection():
    identity = fp.functional_budget_identity(_budget()[1])
    identity["selected"] = identity["selected"][::-1]
    identity["selection_digest"] = fp._selection_digest(
        identity["qualified_digest"], identity["contract_digest"], identity["budget_seed"],
        identity["budget_n"], [tuple(row) for row in identity["selected"]])
    with pytest.raises(ValueError, match="seeded selection order"):
        fp.validate_functional_budget_identity(identity)


def _forged(q, observations, seed=SEED):
    ordered = sorted(observations, key=lambda o: (fp._selection_key(seed, *o), o))
    return fp.FunctionalBudget(q.digest, q.contract_digest, seed, len(ordered), tuple(ordered),
                               fp._selection_digest(q.digest, q.contract_digest, seed,
                                                    len(ordered), ordered))


def test_a_budget_must_belong_to_the_qualified_document_and_its_train_observations(tmp_path):
    q = standard_qualified()
    decl, records, doc = standard_world()
    doc["study_contract"]["assertion"] = "another curator claim"
    other = fd.qualify_functional(fd.parse_functional_constructs(doc), decl, records)
    with pytest.raises(ValueError, match="not selected from this qualified"):
        fp.functional_exposure_plan(q, fp.select_functional_budget(other, 8, SEED))
    for outside in ([("g-a", "t0"), ("g-a", "t99")], [("g-a", "t0"), ("g-val", "v0")]):
        forged = _forged(q, outside)
        with pytest.raises(ValueError, match="outside the qualified train observations"):
            fp.functional_exposure_plan(q, forged)
        # Nothing is opened for a budget that does not match: no FileNotFoundError.
        with pytest.raises(ValueError, match="outside the qualified train observations"):
            fp.load_restricted_measurements(tmp_path / "missing.json", q, forged)


NON_PREFIXES = {
    "skips_the_first": lambda selected: selected[1:3],
    "skips_the_second": lambda selected: selected[:1] + selected[2:3],
    "later_window": lambda selected: selected[2:5],
}


@pytest.mark.parametrize("name", sorted(NON_PREFIXES))
def test_a_self_consistent_budget_that_is_not_the_seeded_prefix_is_rejected_before_any_file_opens(name, tmp_path):
    q, full = _budget()
    forged = _forged(q, NON_PREFIXES[name](full.selected))
    # Every check that needs no document passes: train membership, uniqueness, hash order and a
    # recomputed digest.  Only the full candidate list shows these are not the first n.
    assert forged.selected_set <= set(q.train_observations())
    assert fp.validate_functional_budget_identity(fp.functional_budget_identity(forged)) == forged
    assert forged != fp.select_functional_budget(q, forged.budget_n, SEED)
    document = labels_document(forged, ORDERED)
    for call in (lambda: fp.functional_exposure_plan(q, forged),
                 lambda: fp.load_restricted_measurements(tmp_path / "missing.json", q, forged),
                 lambda: fp.RestrictedFunctionalData.from_document(document, q, forged)):
        with pytest.raises(ValueError, match="not the seeded prefix"):
            call()


@pytest.mark.parametrize("seed", [SEED, "another-seed"])
def test_every_genuine_prefix_matches_its_qualified_document(seed):
    q = standard_qualified()
    for n in range(1, 9):
        fp._check_budget_matches(q, fp.select_functional_budget(q, n, seed))


# ---------------------------------------------------------------------------
# Schedule, replay, coverage
# ---------------------------------------------------------------------------

def test_a_budget_without_a_pair_eligible_group_cannot_be_scheduled_or_planned():
    q, budget = _budget(1)
    for call in (lambda: fp.FunctionalSchedule(budget), lambda: fp.functional_exposure_plan(q, budget),
                 lambda: fp.replay_schedule(budget, 0, 1, 1), lambda: fp.scheduled_coverage(budget, 0, 1, 1)):
        with pytest.raises(ValueError, match="no group has two or more"):
            call()


def test_draws_are_deterministic_identity_only_and_never_replaced():
    _, budget = _budget()
    schedule = fp.FunctionalSchedule(budget)
    members = budget.group_members()
    groups = set()
    random.seed(11)
    python_state, torch_state = random.getstate(), torch.get_rng_state().clone()
    first = [schedule.draw(5, u, k) for u in range(30) for k in range(4)]
    for draw in first:
        assert draw.left_id != draw.right_id
        assert draw.left_id in members[draw.group_id] and draw.right_id in members[draw.group_id]
        groups.add(draw.group_id)
    assert first == [schedule.draw(5, u, k) for u in range(30) for k in range(4)]
    assert (first[7].update, first[7].index) == (1, 3)
    assert groups == {"g-a", "g-b"}          # g-single is a singleton; g-val is not in the budget
    assert first != [schedule.draw(6, u, k) for u in range(30) for k in range(4)]
    assert random.getstate() == python_state and torch.equal(torch.get_rng_state(), torch_state)


@pytest.mark.parametrize("args", [(-1, 0, 0), (2 ** 63, 0, 0), ("5", 0, 0), (True, 0, 0),
                                  (5, -1, 0), (5, 0, -1), (5, True, 0), (5, 0, 1.0)])
def test_draw_keys_are_validated(args):
    schedule = fp.FunctionalSchedule(_budget()[1])
    with pytest.raises(ValueError):
        schedule.draw(*args)


@pytest.mark.parametrize("args", [(0, 0, 0, "p"), (5, 3, 2, "functional-pair"), (2 ** 63 - 1, 10, 1, "x")])
def test_keyed_seed_is_the_torch_side_keyed_seed(args):
    from smallAntibodyGen.experiments.trastuzumab_poc import randomness
    assert fp.RNG_ALGORITHM == randomness.RNG_ALGORITHM
    assert fp._keyed_seed(*args) == randomness.keyed_seed(*args)


def test_replay_matches_direct_enumeration_and_its_state_is_bounded_by_the_selection():
    _, budget = _budget()
    schedule = fp.FunctionalSchedule(budget)
    expected = Counter()
    for update in range(7):
        for k in range(3):
            expected.update(schedule.draw(9, update, k).observations)
    replay = fp.replay_schedule(budget, 9, 3, 7)
    assert replay.pair_opportunities == 21
    assert dict(replay.exposures) == {o: expected.get(o, 0) for o in budget.selected}
    assert sum(n for _, n in replay.exposures) == 42 and sum(n for _, n in replay.group_draws) == 21
    assert replay.compared == frozenset(expected)
    assert len(fp.replay_schedule(budget, 9, 3, 300).exposures) == budget.budget_n == len(replay.exposures)


def test_scheduled_coverage_is_an_exact_fraction_that_counts_ineligible_singletons():
    _, budget = _budget()
    one = fp.scheduled_coverage(budget, 9, 1, 1)
    assert isinstance(one, Fraction) and one == Fraction(2, 8)
    # 7 of 8 observations sit in groups with two or more selected members.
    assert fp.scheduled_coverage(budget, 9, 4, 100) == Fraction(7, 8)


def test_coverage_requirement_is_a_declared_threshold_compared_exactly():
    assert fp.coverage_requirement_met(Fraction(9, 10), 0.9) is True
    assert fp.coverage_requirement_met(Fraction(89, 100), 0.9) is False
    assert fp.coverage_requirement_met(Fraction(7, 8), 1.0) is False
    assert fp.coverage_requirement_met(Fraction(1), 1.0) is True
    assert fp.coverage_requirement_met(Fraction(0), 0) is True
    # A schedule can legitimately leave observations unseen; only the declared bar applies.
    _, budget = _budget()
    coverage = fp.scheduled_coverage(budget, 9, 4, 100)
    assert fp.coverage_requirement_met(coverage, 0.875) and not fp.coverage_requirement_met(coverage, 0.9)
    for bad in (-0.1, 1.1, float("nan"), True, "0.5"):
        with pytest.raises(ValueError):
            fp.coverage_requirement_met(Fraction(1, 2), bad)
    with pytest.raises(TypeError):
        fp.coverage_requirement_met(0.5, 0.5)


def test_exposure_plan_reports_structure_without_reading_outcomes():
    q, budget = _budget()
    plan = fp.functional_exposure_plan(q, budget)
    assert plan["selected_observations"] == 8
    assert plan["unique_selected_construct_ids"] == plan["unique_physical_constructs"] == 6
    assert plan["unique_antibodies"] == 6 and plan["unique_antigens"] == 1
    assert plan["repeated_construct_ids"] == ["t0", "t5"]
    assert plan["singleton_group_ids"] == ["g-single"] and plan["pair_eligible_group_ids"] == ["g-a", "g-b"]
    assert (plan["pair_eligible_observations"], plan["pair_ineligible_observations"]) == (7, 1)
    assert plan["possible_pair_count"] == 9 and "schedule" not in plan
    assert plan["groups"]["g-a"] == {"selected_members": 4, "pair_eligible": True, "possible_pairs": 6}
    assert plan["groups"]["g-single"]["pair_eligible"] is False
    json.dumps(plan)


def test_exposure_plan_reports_unseen_observations_and_zero_coverage():
    q, budget = _budget()
    schedule = fp.functional_exposure_plan(q, budget, seed=9, pairs_per_update=1, total_updates=1)["schedule"]
    assert schedule["pair_opportunities"] == 1
    assert (schedule["seen_observation_count"], schedule["unseen_observation_count"]) == (2, 6)
    assert len(schedule["unseen_observations"]) == 6
    assert (schedule["minimum_count_all_selected"], schedule["maximum_count_all_selected"]) == (0, 1)
    # Two of eight observations is the reduced fraction 1/4.
    assert schedule["compared_fraction"] == {"numerator": 1, "denominator": 4, "value": 0.25}
    with pytest.raises(ValueError, match="together"):
        fp.functional_exposure_plan(q, budget, seed=9)


@pytest.mark.parametrize("seed", ["s0", "s1", "s2", "s3", "s4", "s5"])
@pytest.mark.parametrize("n", [2, 3, 5, 7])
def test_plan_agrees_with_an_independent_count_of_the_selection(seed, n):
    q, budget = _budget(n, seed)
    by_group = {}
    for group, construct in budget.selected:
        by_group.setdefault(group, set()).add(construct)
    eligible = sorted(g for g, members in by_group.items() if len(members) >= 2)
    if not eligible:
        with pytest.raises(ValueError, match="no group has two or more"):
            fp.functional_exposure_plan(q, budget)
        return
    plan = fp.functional_exposure_plan(q, budget)
    ids = [c for _, c in budget.selected]
    assert plan["pair_eligible_group_ids"] == eligible
    assert plan["singleton_group_ids"] == sorted(g for g, m in by_group.items() if len(m) == 1)
    assert plan["possible_pair_count"] == sum(len(by_group[g]) * (len(by_group[g]) - 1) // 2 for g in eligible)
    assert plan["selected_observations"] == n and plan["unique_selected_construct_ids"] == len(set(ids))
    assert plan["repeated_construct_ids"] == sorted(c for c in set(ids) if ids.count(c) > 1)


# ---------------------------------------------------------------------------
# Restricted measurements
# ---------------------------------------------------------------------------

def point(value):
    return {"kind": "point", "value": value}


def interval(lower, upper):
    return {"kind": "interval", "lower": lower, "upper": upper}


def at_least(lower):
    return {"kind": "lower_bound", "lower": lower}


def at_most(upper):
    return {"kind": "upper_bound", "upper": upper}


ORDERED = {
    ("g-a", "t0"): point(1.0), ("g-a", "t1"): point(5.0), ("g-a", "t2"): point(5.0),
    ("g-a", "t3"): interval(4.0, 6.0),
    ("g-b", "t0"): at_least(100.0), ("g-b", "t4"): point(10.0), ("g-b", "t5"): point(50.0),
    ("g-single", "t5"): point(7.0),
}


def negate(measurement):
    kind = measurement["kind"]
    if kind == "point":
        return point(-measurement["value"])
    if kind == "interval":
        return interval(-measurement["upper"], -measurement["lower"])
    return at_most(-measurement["lower"]) if kind == "lower_bound" else at_least(-measurement["upper"])


WORLDS = {
    "ordered": ORDERED,
    "negated": {key: negate(value) for key, value in ORDERED.items()},
    "tied": {key: point(1.0) for key in ORDERED},
    "uncertain": {key: interval(0.0, 10.0) for key in ORDERED},
}


def labels_document(budget, values):
    return {"schema": fp.MEASUREMENTS_SCHEMA, "selection_digest": budget.selection_digest,
            "measurements": [{"group_id": g, "construct_id": c, "measurement": copy.deepcopy(values[(g, c)])}
                             for g, c in budget.selected]}


def restricted(q, budget, values):
    return fp.RestrictedFunctionalData.from_document(labels_document(budget, values), q, budget)


def test_classification_follows_oriented_interval_rules():
    q, budget = _budget()
    data = restricted(q, budget, ORDERED)
    assert data.directions == {"g-a": "lower_better", "g-b": "higher_better", "g-single": "lower_better"}
    cases = [("g-a", "t0", "t1", "ordered", "t0", "t1"), ("g-a", "t1", "t0", "ordered", "t0", "t1"),
             ("g-a", "t1", "t2", "tie", None, None), ("g-a", "t3", "t1", "unresolved", None, None),
             ("g-a", "t0", "t3", "ordered", "t0", "t3"), ("g-b", "t0", "t4", "ordered", "t0", "t4"),
             ("g-b", "t4", "t5", "ordered", "t5", "t4")]
    for group, left, right, outcome, better, worse in cases:
        result = data.classify(fp.PairDraw(0, 0, group, left, right))
        assert (result.outcome, result.better_id, result.worse_id) == (outcome, better, worse)
        assert result.informative == (outcome == "ordered")
    for bad in (fp.PairDraw(0, 0, "g-zzz", "t0", "t1"), fp.PairDraw(0, 0, "g-a", "t0", "t0"),
                fp.PairDraw(0, 0, "g-a", "t0", "t4"), fp.PairDraw(0, 0, "g-single", "t5", "t5")):
        with pytest.raises(ValueError, match="outside the fixed budget"):
            data.classify(bad)
    with pytest.raises(TypeError):
        data.classify(("g-a", "t0", "t1"))


BAD_LABEL_DOCUMENTS = {
    "wrong_schema": lambda d: d.update(schema="trastuzumab-functional-measurements/2"),
    "extra_top_level_key": lambda d: d.update(note="x"),
    "missing_measurements": lambda d: d.pop("measurements"),
    "measurements_not_a_list": lambda d: d.update(measurements={}),
    "row_extra_key": lambda d: d["measurements"][0].update(direction="lower_better"),
    "row_not_an_object": lambda d: d["measurements"].append(3),
    "duplicate_row": lambda d: d["measurements"].append(copy.deepcopy(d["measurements"][0])),
    "missing_observation": lambda d: d["measurements"].pop(),
    "heldout_observation": lambda d: d["measurements"].append(
        {"group_id": "g-val", "construct_id": "v0", "measurement": point(1.0)}),
    "unknown_group": lambda d: d["measurements"][0].update(group_id="g-zzz"),
    "unknown_kind": lambda d: d["measurements"][0]["measurement"].update(kind="range"),
    "bool_value": lambda d: d["measurements"][0].update(measurement=point(True)),
    "string_infinity": lambda d: d["measurements"][0].update(measurement=at_least("Infinity")),
    "reversed_interval": lambda d: d["measurements"][0].update(measurement=interval(2.0, 1.0)),
}


@pytest.mark.parametrize("name", sorted(BAD_LABEL_DOCUMENTS))
def test_measurements_must_be_exactly_the_selected_observations_and_well_formed(name):
    q, budget = _budget()
    document = labels_document(budget, ORDERED)
    BAD_LABEL_DOCUMENTS[name](document)
    with pytest.raises(ValueError):
        fp.RestrictedFunctionalData.from_document(document, q, budget)


def test_the_selection_digest_is_compared_before_any_value_is_interpreted():
    q, budget = _budget()
    document = labels_document(budget, ORDERED)
    document["selection_digest"] = "0" * 64
    document["measurements"] = "this is not even a list"
    with pytest.raises(ValueError, match="selection digest"):
        fp.RestrictedFunctionalData.from_document(document, q, budget)
    document["selection_digest"] = budget.selection_digest
    with pytest.raises(ValueError, match="measurements must be a JSON list"):
        fp.RestrictedFunctionalData.from_document(document, q, budget)


def test_measurements_are_loaded_from_a_strict_file_for_a_fixed_budget(tmp_path):
    q, budget = _budget()
    document = labels_document(budget, ORDERED)
    path = tmp_path / "measurements.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    loaded = fp.load_restricted_measurements(path, q, budget)
    assert loaded.data_digest() == restricted(q, budget, ORDERED).data_digest()
    assert loaded.data_digest() != restricted(q, budget, WORLDS["negated"]).data_digest()
    assert loaded.measurement("g-a", "t3") == I(4.0, 6.0)
    path.write_text(json.dumps(document).replace('"value": 1.0', '"value": Infinity', 1), encoding="utf-8")
    with pytest.raises(ValueError, match="nonfinite JSON"):
        fp.load_restricted_measurements(path, q, budget)
    path.write_text(json.dumps(document)[:-1] + ', "schema": "again"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        fp.load_restricted_measurements(path, q, budget)


def test_the_schedule_and_compared_set_do_not_depend_on_labels():
    q, budget = _budget()
    schedule = fp.FunctionalSchedule(budget)
    reference = [schedule.draw(4, u, k) for u in range(12) for k in range(3)]
    replay = fp.replay_schedule(budget, 4, 3, 12)
    outcomes = {}
    for name, values in WORLDS.items():
        outcomes[name] = restricted(q, budget, values).replay_outcomes(4, 3, 12)
        assert outcomes[name].pair_opportunities == 36 == replay.pair_opportunities
        assert outcomes[name].compared == replay.compared
        total = outcomes[name].ordered_pairs + outcomes[name].tie_pairs + outcomes[name].unresolved_pairs
        assert total == 36
        assert [schedule.draw(4, u, k) for u in range(12) for k in range(3)] == reference
    assert outcomes["tied"].tie_pairs == 36 and outcomes["tied"].ordered_pairs == 0
    assert outcomes["uncertain"].unresolved_pairs == 36 and outcomes["uncertain"].ordered_observations == frozenset()
    for field in ("ordered_pairs", "tie_pairs", "unresolved_pairs", "ordered_observations"):
        assert getattr(outcomes["ordered"], field) == getattr(outcomes["negated"], field)
    ordered, negated = restricted(q, budget, ORDERED), restricted(q, budget, WORLDS["negated"])
    for draw in reference:
        a, b = ordered.classify(draw), negated.classify(draw)
        assert a.outcome == b.outcome and (a.better_id, a.worse_id) == (b.worse_id, b.better_id)


def test_no_pair_is_ever_redrawn_even_when_every_pair_ties(monkeypatch):
    q, budget = _budget()
    data = restricted(q, budget, WORLDS["tied"])
    calls = []
    original = fp.FunctionalSchedule.draw

    def counting(self, seed, update, k):
        calls.append((update, k))
        return original(self, seed, update, k)

    monkeypatch.setattr(fp.FunctionalSchedule, "draw", counting)
    outcome = data.replay_outcomes(4, 3, 12)
    assert len(calls) == len(set(calls)) == 36 and outcome.tie_pairs == 36


def test_data_free_validators_never_consult_labels(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("labels were consulted")

    q, budget = _budget()
    monkeypatch.setattr(fp.RestrictedFunctionalData, "classify", refuse)
    monkeypatch.setattr(fp, "compare", refuse)
    fp.validate_functional_budget_identity(fp.functional_budget_identity(budget))
    fp.functional_exposure_plan(q, budget, seed=1, pairs_per_update=2, total_updates=5)
    replay = fp.replay_schedule(budget, 1, 2, 5)
    fp.scheduled_coverage(budget, 1, 2, 5)
    fp.check_counter_invariants({"pair_opportunities": 10},
                                {"compared": [list(o) for o in replay.compared]}, replay, 2, 5)


def test_classification_membership_is_a_hash_lookup_not_a_member_scan(monkeypatch):
    q, budget = _budget()
    data = restricted(q, budget, ORDERED)
    expected = data.replay_outcomes(4, 3, 12)

    def refuse(*args, **kwargs):
        raise AssertionError("classify scanned the group's member tuple")

    monkeypatch.setattr(fp.FunctionalBudget, "members_of", refuse)
    assert data.classify(fp.PairDraw(0, 0, "g-a", "t0", "t1")).outcome == "ordered"
    assert data.replay_outcomes(4, 3, 12) == expected
    for bad in (fp.PairDraw(0, 0, "g-zzz", "t0", "t1"), fp.PairDraw(0, 0, "g-a", "t0", "t0"),
                fp.PairDraw(0, 0, "g-a", "t0", "t4"), fp.PairDraw(0, 0, "g-a", "t4", "t0"),
                fp.PairDraw(0, 0, "g-a", "t4", "t5"), fp.PairDraw(0, 0, "g-single", "t5", "t5")):
        with pytest.raises(ValueError, match="outside the fixed budget"):
            data.classify(bad)


# ---------------------------------------------------------------------------
# Immutable restricted data
# ---------------------------------------------------------------------------

def test_restricted_data_is_an_immutable_snapshot_of_its_inputs_and_of_its_own_fields():
    q, budget = _budget()
    directions = {g: q.direction(g) for g in budget.group_members()}
    values = {key: fp.Measurement.from_document(ORDERED[key]) for key in budget.selected}
    data = fp.RestrictedFunctionalData(budget, directions, values)
    digest, replay = data.data_digest(), data.replay_outcomes(4, 3, 12)
    # Serialization is unchanged by the snapshot type.
    assert digest == restricted(q, budget, ORDERED).data_digest()
    # The caller's original dicts are no longer connected to the supervision.
    directions["g-a"] = "higher_better"
    values[("g-a", "t0")] = P(99.0)
    del values[("g-b", "t4")]
    # Neither are the public fields writable, item by item or as a whole.
    for mapping, key, value in ((data.directions, "g-a", "higher_better"),
                                (data.values, ("g-a", "t0"), P(99.0))):
        assert not isinstance(mapping, dict)
        with pytest.raises(TypeError):
            mapping[key] = value
        with pytest.raises(TypeError):
            del mapping[key]
        for mutator in ("update", "pop", "popitem", "clear", "setdefault"):
            assert not hasattr(mapping, mutator)
        # A copy is detached from the snapshot.
        mapping.copy()[key] = value
    for field in ("budget", "directions", "values"):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(data, field, {})
    with pytest.raises(dataclasses.FrozenInstanceError):
        data.measurement("g-a", "t0").value = 99.0
    assert data.directions == {"g-a": "lower_better", "g-b": "higher_better", "g-single": "lower_better"}
    assert data.measurement("g-a", "t0") == P(1.0) and data.measurement("g-b", "t4") == P(10.0)
    assert data.data_digest() == digest and data.replay_outcomes(4, 3, 12) == replay


def test_restricted_data_still_validates_its_construction():
    q, budget = _budget()
    directions = {g: q.direction(g) for g in budget.group_members()}
    values = {key: fp.Measurement.from_document(ORDERED[key]) for key in budget.selected}
    fp.RestrictedFunctionalData(budget, directions, values)
    with pytest.raises(ValueError, match="directions"):
        fp.RestrictedFunctionalData(budget, {**directions, "g-a": "up"}, values)
    with pytest.raises(ValueError, match="directions"):
        fp.RestrictedFunctionalData(budget, {g: d for g, d in directions.items() if g != "g-b"}, values)
    with pytest.raises(ValueError, match="measurements"):
        fp.RestrictedFunctionalData(budget, directions, {k: v for k, v in values.items() if k != ("g-a", "t0")})
    with pytest.raises(ValueError, match="measurements"):
        fp.RestrictedFunctionalData(budget, directions, {**values, ("g-val", "v0"): P(1.0)})
    with pytest.raises(ValueError, match="measurements"):
        fp.RestrictedFunctionalData(budget, directions, {**values, ("g-a", "t0"): 1.0})
    with pytest.raises(TypeError):
        fp.RestrictedFunctionalData("budget", directions, values)


def test_restricted_data_can_be_bound_to_the_qualified_budget_and_group_directions():
    q, budget = _budget()
    data = restricted(q, budget, ORDERED)
    assert data.verify_against(q) is None
    flip = {"higher_better": "lower_better", "lower_better": "higher_better"}
    flipped = fp.RestrictedFunctionalData(budget, {g: flip[d] for g, d in data.directions.items()}, data.values)
    with pytest.raises(ValueError, match="directions differ"):
        flipped.verify_against(q)
    # A directly constructed object cannot smuggle in a self-consistent non-prefix budget either.
    non_prefix = _forged(q, budget.selected[1:3])
    direct = fp.RestrictedFunctionalData(
        non_prefix, {g: q.direction(g) for g in non_prefix.group_members()},
        {key: data.values[key] for key in non_prefix.selected})
    with pytest.raises(ValueError, match="not the seeded prefix"):
        direct.verify_against(q)
    decl, records, doc = standard_world()
    doc["study_contract"]["assertion"] = "another curator claim"
    other = fd.qualify_functional(fd.parse_functional_constructs(doc), decl, records)
    with pytest.raises(ValueError, match="not selected from this qualified"):
        data.verify_against(other)
    with pytest.raises(TypeError):
        data.verify_against("qualified")


# ---------------------------------------------------------------------------
# Zero-update horizons and ordered updates
# ---------------------------------------------------------------------------

MIXED = {
    ("g-a", "t0"): point(1.0), ("g-a", "t1"): interval(0.0, 10.0), ("g-a", "t2"): interval(0.0, 10.0),
    ("g-a", "t3"): interval(0.0, 10.0),
    # Only g-b's t0/t4 pair is ordered; each draw lands on it with probability 1/2 * 1/3.
    ("g-b", "t0"): point(1.0), ("g-b", "t4"): point(2.0), ("g-b", "t5"): interval(0.0, 10.0),
    ("g-single", "t5"): point(7.0),
}


def ordered_pairs_per_update(data, budget, seed, pairs, updates):
    """Independent of ``replay_outcomes``: only the pure draw and the pair classifier."""
    schedule = fp.FunctionalSchedule(budget)
    return [sum(data.classify(schedule.draw(seed, u, k)).outcome == "ordered" for k in range(pairs))
            for u in range(updates)]


def test_a_zero_update_horizon_is_valid_and_everything_is_zero():
    q, budget = _budget()
    replay = fp.replay_schedule(budget, 4, 3, 0)
    assert (replay.updates, replay.pairs_per_update, replay.pair_opportunities) == (0, 3, 0)
    assert dict(replay.exposures) == dict.fromkeys(budget.selected, 0) and len(replay.exposures) == 8
    assert all(n == 0 for _, n in replay.group_draws) and replay.compared == frozenset()
    assert fp.scheduled_coverage(budget, 4, 3, 0) == Fraction(0)
    plan = fp.functional_exposure_plan(q, budget, seed=4, pairs_per_update=3, total_updates=0)["schedule"]
    assert (plan["pair_opportunities"], plan["seen_observation_count"], plan["unseen_observation_count"]) == (0, 0, 8)
    assert (plan["minimum_count_all_selected"], plan["maximum_count_all_selected"]) == (0, 0)
    assert plan["compared_fraction"] == {"numerator": 0, "denominator": 1, "value": 0.0}
    zero = {"pair_opportunities": 0, "ordered_pairs": 0, "tie_pairs": 0, "unresolved_pairs": 0,
            "active_pairs": 0, "ordered_updates": 0}
    seen = {"compared": [], "ordered": []}
    for world in WORLDS.values():
        outcome = restricted(q, budget, world).replay_outcomes(4, 3, 0)
        assert (outcome.updates, outcome.pair_opportunities) == (0, 0)
        assert (outcome.ordered_pairs, outcome.tie_pairs, outcome.unresolved_pairs, outcome.ordered_updates) == (0, 0, 0, 0)
        assert outcome.compared == frozenset() and outcome.ordered_observations == frozenset()
        for horizon in (outcome, replay):
            fp.check_counter_invariants(dict(zero), seen, horizon, 3, 0)
    fp.check_counter_invariants({"pair_opportunities": 0}, {"compared": []}, replay, 3, 0)
    for forged in (dict(zero, pair_opportunities=1), dict(zero, ordered_pairs=1),
                   dict(zero, ordered_pairs=1, ordered_updates=1), dict(zero, ordered_updates=1),
                   dict(zero, tie_pairs=1), dict(zero, active_pairs=1)):
        with pytest.raises(ValueError):
            fp.check_counter_invariants(forged, seen, replay, 3, 0)
    with pytest.raises(ValueError, match="compared observations differ"):
        fp.check_counter_invariants(zero, {"compared": [["g-a", "t0"]], "ordered": []}, replay, 3, 0)
    with pytest.raises(ValueError):
        fp.check_counter_invariants(zero, {"compared": [], "ordered": [["g-a", "t0"], ["g-a", "t1"]]}, replay, 3, 0)


@pytest.mark.parametrize("updates", [True, False, -1, 1.0, "3", None])
def test_updates_must_be_a_non_negative_integer_everywhere(updates):
    q, budget = _budget()
    replay = fp.replay_schedule(budget, 4, 3, 0)
    data = restricted(q, budget, ORDERED)
    for call in (lambda: fp.replay_schedule(budget, 4, 3, updates),
                 lambda: fp.scheduled_coverage(budget, 4, 3, updates),
                 lambda: fp.functional_exposure_plan(q, budget, seed=4, pairs_per_update=3, total_updates=updates),
                 lambda: data.replay_outcomes(4, 3, updates),
                 lambda: fp.check_counter_invariants({"pair_opportunities": 0}, {"compared": []}, replay, 3, updates)):
        with pytest.raises(ValueError):
            call()


@pytest.mark.parametrize("pairs", [0, -1, True, 2.0, None])
def test_pairs_per_update_stays_a_positive_integer_even_for_zero_updates(pairs):
    q, budget = _budget()
    replay = fp.replay_schedule(budget, 4, 3, 0)
    data = restricted(q, budget, ORDERED)
    for call in (lambda: fp.replay_schedule(budget, 4, pairs, 0), lambda: data.replay_outcomes(4, pairs, 0),
                 lambda: fp.check_counter_invariants({"pair_opportunities": 0}, {"compared": []}, replay, pairs, 0)):
        with pytest.raises(ValueError):
            call()


def test_ordered_updates_is_zero_when_every_pair_ties_or_is_unresolved():
    q, budget = _budget()
    for name, resolved in (("tied", "tie_pairs"), ("uncertain", "unresolved_pairs")):
        outcome = restricted(q, budget, WORLDS[name]).replay_outcomes(4, 3, 12)
        assert getattr(outcome, resolved) == 36
        assert (outcome.ordered_pairs, outcome.ordered_updates) == (0, 0)


def test_ordered_updates_counts_updates_not_pairs_and_does_not_depend_on_hinge_activity():
    q, budget = _budget()
    data = restricted(q, budget, ORDERED)
    outcome = data.replay_outcomes(6, 50, 3)
    per_update = ordered_pairs_per_update(data, budget, 6, 50, 3)
    assert max(per_update) >= 2                       # many ordered pairs share one update
    assert outcome.ordered_pairs == sum(per_update) > 3
    assert outcome.ordered_updates == sum(1 for n in per_update if n) <= 3 < outcome.ordered_pairs


def test_ordered_updates_in_a_world_where_ordered_and_unresolved_updates_mix():
    q, budget = _budget()
    data = restricted(q, budget, MIXED)
    outcome = data.replay_outcomes(4, 2, 80)
    per_update = ordered_pairs_per_update(data, budget, 4, 2, 80)
    assert outcome.ordered_pairs == sum(per_update)
    assert outcome.ordered_updates == sum(1 for n in per_update if n)
    assert 0 < outcome.ordered_updates < 80                       # ordered AND all-unresolved updates occur
    assert outcome.tie_pairs == 0 and outcome.unresolved_pairs == 160 - outcome.ordered_pairs
    assert outcome.ordered_observations == frozenset({("g-b", "t0"), ("g-b", "t4")})
    # A prefix of the horizon is the replay of that prefix.
    shorter = data.replay_outcomes(4, 2, 30)
    assert shorter.ordered_updates == sum(1 for n in per_update[:30] if n)
    assert shorter.ordered_pairs == sum(per_update[:30])


# ---------------------------------------------------------------------------
# Counter invariants
# ---------------------------------------------------------------------------

def _honest(values=None, pairs=3, updates=12):
    q, budget = _budget()
    data = restricted(q, budget, values or ORDERED)
    outcome = data.replay_outcomes(4, pairs, updates)
    counters = {"pair_opportunities": outcome.pair_opportunities, "ordered_pairs": outcome.ordered_pairs,
                "tie_pairs": outcome.tie_pairs, "unresolved_pairs": outcome.unresolved_pairs,
                "active_pairs": 0, "ordered_updates": outcome.ordered_updates}
    seen = {"compared": sorted(list(o) for o in outcome.compared),
            "ordered": sorted(list(o) for o in outcome.ordered_observations)}
    return budget, outcome, counters, seen


def test_honest_counters_pass_against_both_replays():
    budget, outcome, counters, seen = _honest()
    fp.check_counter_invariants(counters, seen, outcome, 3, 12)
    fp.check_counter_invariants(counters, seen, fp.replay_schedule(budget, 4, 3, 12), 3, 12)
    fp.check_counter_invariants({"pair_opportunities": 36}, {"compared": seen["compared"]},
                                fp.replay_schedule(budget, 4, 3, 12), 3, 12)
    with pytest.raises(ValueError, match="outcome counters are required"):
        fp.check_counter_invariants({"pair_opportunities": 36}, {"compared": seen["compared"]}, outcome, 3, 12)


COUNTER_TAMPER = {
    "opportunities_plus_one": lambda c, s: c.update(pair_opportunities=c["pair_opportunities"] + 1),
    "opportunities_bool": lambda c, s: c.update(pair_opportunities=True),
    "compared_missing": lambda c, s: s["compared"].pop(),
    "compared_extra": lambda c, s: s["compared"].append(["g-val", "v0"]),
    "compared_malformed": lambda c, s: s["compared"].append(["g-a"]),
    "compared_absent": lambda c, s: s.pop("compared"),
    "partition_broken": lambda c, s: c.update(tie_pairs=c["tie_pairs"] + 1),
    "negative_counter": lambda c, s: c.update(tie_pairs=-1),
    "counter_missing": lambda c, s: c.pop("unresolved_pairs"),
    "active_exceeds_ordered": lambda c, s: c.update(active_pairs=c["ordered_pairs"] + 1),
    "ordered_not_compared": lambda c, s: s["ordered"].append(["g-val", "v0"]),
}


@pytest.mark.parametrize("name", sorted(COUNTER_TAMPER))
def test_forged_counters_are_rejected_by_the_schedule_replay(name):
    budget, outcome, counters, seen = _honest()
    COUNTER_TAMPER[name](counters, seen)
    for replay in (outcome, fp.replay_schedule(budget, 4, 3, 12)):
        with pytest.raises(ValueError):
            fp.check_counter_invariants(counters, seen, replay, 3, 12)


def test_the_replay_horizon_must_match_the_recorded_horizon():
    budget, outcome, counters, seen = _honest()
    with pytest.raises(ValueError, match="different horizon"):
        fp.check_counter_invariants(counters, seen, outcome, 3, 13)
    with pytest.raises(ValueError, match="different horizon"):
        fp.check_counter_invariants(counters, seen, outcome, 4, 9)
    with pytest.raises(TypeError):
        fp.check_counter_invariants(counters, seen, "replay", 3, 12)


def test_outcome_dependent_counters_get_bounds_from_data_free_replay_and_exactness_from_labels():
    budget, outcome, counters, seen = _honest()
    moved = dict(counters, ordered_pairs=counters["ordered_pairs"] - 1, tie_pairs=counters["tie_pairs"] + 1)
    fp.check_counter_invariants(moved, seen, fp.replay_schedule(budget, 4, 3, 12), 3, 12)   # bounds only
    with pytest.raises(ValueError, match="differ from the label-dependent replay"):
        fp.check_counter_invariants(moved, seen, outcome, 3, 12)
    fewer = dict(seen, ordered=seen["ordered"][1:])
    fp.check_counter_invariants(counters, fewer, fp.replay_schedule(budget, 4, 3, 12), 3, 12)
    with pytest.raises(ValueError, match="differ from the label-dependent replay"):
        fp.check_counter_invariants(counters, fewer, outcome, 3, 12)


def test_an_ordered_pair_needs_two_distinct_ordered_observations():
    _, budget = _budget()
    replay = fp.replay_schedule(budget, 4, 1, 1)                  # exactly one scheduled pair
    pair = sorted(list(o) for o in replay.compared)
    assert len(pair) == 2
    compared = {"compared": pair}
    one_ordered = {"pair_opportunities": 1, "ordered_pairs": 1, "tie_pairs": 0, "unresolved_pairs": 0}
    # Zero, one (twice, or either one) observed ID cannot account for an ordered pair.
    for ordered in ([], pair[:1], pair[1:], [pair[0], pair[0]]):
        with pytest.raises(ValueError, match="two distinct ordered observations"):
            fp.check_counter_invariants(dict(one_ordered), dict(compared, ordered=ordered), replay, 1, 1)
    with pytest.raises(ValueError, match="two distinct ordered observations"):
        fp.check_counter_invariants(dict(one_ordered), compared, replay, 1, 1)       # absent == empty
    fp.check_counter_invariants(dict(one_ordered), dict(compared, ordered=pair), replay, 1, 1)
    # With no ordered pair no ordered observation is allowed, and none is required.
    none_ordered = dict(one_ordered, ordered_pairs=0, tie_pairs=1)
    fp.check_counter_invariants(dict(none_ordered), dict(compared, ordered=[]), replay, 1, 1)
    fp.check_counter_invariants(dict(none_ordered), compared, replay, 1, 1)
    for ordered in (pair[:1], pair):
        with pytest.raises(ValueError, match="exceed"):
            fp.check_counter_invariants(dict(none_ordered), dict(compared, ordered=ordered), replay, 1, 1)


def test_ordered_observations_keep_their_upper_bound_and_may_leave_available_labels_unreached():
    _, budget = _budget()
    replay = fp.replay_schedule(budget, 4, 3, 12)
    compared = sorted(list(o) for o in replay.compared)
    assert len(compared) >= 3
    counters = {"pair_opportunities": 36, "ordered_pairs": 1, "tie_pairs": 35, "unresolved_pairs": 0}
    seen = {"compared": compared}
    # One ordered pair reached two labels; every other compared label is legitimately unreached.
    fp.check_counter_invariants(dict(counters), dict(seen, ordered=compared[:2]), replay, 3, 12)
    with pytest.raises(ValueError, match="exceed"):
        fp.check_counter_invariants(dict(counters), dict(seen, ordered=compared[:3]), replay, 3, 12)


def test_ordered_updates_bounds_are_the_ceiling_of_pairs_over_pairs_per_update_and_the_horizon():
    _, budget = _budget()
    replay = fp.replay_schedule(budget, 4, 3, 12)
    seen = {"compared": sorted(list(o) for o in replay.compared)}
    seen["ordered"] = seen["compared"][:2]
    four = {"pair_opportunities": 36, "ordered_pairs": 4, "tie_pairs": 32, "unresolved_pairs": 0}
    # Four ordered pairs at three pairs per update need ceil(4 / 3) = 2 updates and at most 4.
    for value in (2, 3, 4):
        fp.check_counter_invariants(dict(four, ordered_updates=value), seen, replay, 3, 12)
    for value in (0, 1, 5):
        with pytest.raises(ValueError, match="ordered_updates"):
            fp.check_counter_invariants(dict(four, ordered_updates=value), seen, replay, 3, 12)
    # The horizon caps it too: six ordered pairs in two updates can only be two ordered updates.
    short = fp.replay_schedule(budget, 4, 3, 2)
    seen = {"compared": sorted(list(o) for o in short.compared)}
    seen["ordered"] = seen["compared"][:2]
    six = {"pair_opportunities": 6, "ordered_pairs": 6, "tie_pairs": 0, "unresolved_pairs": 0}
    fp.check_counter_invariants(dict(six, ordered_updates=2), seen, short, 3, 2)
    for value in (1, 3):
        with pytest.raises(ValueError, match="ordered_updates"):
            fp.check_counter_invariants(dict(six, ordered_updates=value), seen, short, 3, 2)


ORDERED_UPDATES_TAMPER = {
    "one_too_many": lambda c: c.update(ordered_updates=min(c["ordered_pairs"], 12) + 1),
    "below_the_ceiling": lambda c: c.update(ordered_updates=-(-c["ordered_pairs"] // 3) - 1),
    "negative": lambda c: c.update(ordered_updates=-1),
    "bool": lambda c: c.update(ordered_updates=True),
    "float": lambda c: c.update(ordered_updates=float(c["ordered_updates"])),
    "null": lambda c: c.update(ordered_updates=None),
    "text": lambda c: c.update(ordered_updates=str(c["ordered_updates"])),
}


@pytest.mark.parametrize("name", sorted(ORDERED_UPDATES_TAMPER))
def test_forged_ordered_updates_are_rejected_by_both_replays(name):
    budget, outcome, counters, seen = _honest()
    ORDERED_UPDATES_TAMPER[name](counters)
    for replay in (outcome, fp.replay_schedule(budget, 4, 3, 12)):
        with pytest.raises(ValueError):
            fp.check_counter_invariants(counters, seen, replay, 3, 12)


@pytest.mark.parametrize("values, pairs, updates", [(ORDERED, 3, 12), (MIXED, 2, 80)], ids=["ordered", "mixed"])
def test_ordered_updates_is_pinned_exactly_by_labels_and_only_bounded_by_the_schedule(values, pairs, updates):
    budget, outcome, counters, seen = _honest(values, pairs, updates)
    schedule_only = fp.replay_schedule(budget, 4, pairs, updates)
    for replay in (outcome, schedule_only):
        fp.check_counter_invariants(counters, seen, replay, pairs, updates)
    low, high = -(-counters["ordered_pairs"] // pairs), min(counters["ordered_pairs"], updates)
    other = next(value for value in range(low, high + 1) if value != counters["ordered_updates"])
    moved = dict(counters, ordered_updates=other)
    fp.check_counter_invariants(moved, seen, schedule_only, pairs, updates)             # inside the bounds
    with pytest.raises(ValueError, match="differ from the label-dependent replay"):
        fp.check_counter_invariants(moved, seen, outcome, pairs, updates)


def test_ordered_updates_needs_the_outcome_counters_like_the_other_ordered_records():
    budget, _, _, seen = _honest()
    replay = fp.replay_schedule(budget, 4, 3, 12)
    with pytest.raises(ValueError, match="need the ordered, tie and unresolved"):
        fp.check_counter_invariants({"pair_opportunities": 36, "ordered_updates": 3},
                                    {"compared": seen["compared"]}, replay, 3, 12)


def test_seen_observations_must_be_a_subset_of_the_budget_not_equal_to_it():
    _, budget = _budget()
    selected = [list(o) for o in budget.selected]
    assert fp.check_seen_within_budget([], budget) == frozenset()
    assert fp.check_seen_within_budget(selected[:3], budget) == frozenset(budget.selected[:3])
    assert fp.check_seen_within_budget(selected, budget) == budget.selected_set
    with pytest.raises(ValueError, match="outside the fixed budget"):
        fp.check_seen_within_budget(selected[:3] + [["g-val", "v0"]], budget)
    with pytest.raises(ValueError):
        fp.check_seen_within_budget([["g-a"]], budget)
