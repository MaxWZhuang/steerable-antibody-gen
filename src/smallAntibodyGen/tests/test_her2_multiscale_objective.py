"""The pair objective, against stub models whose every conditional is computed independently.

The stubs here are *coherent*: each conditional the model returns is derived, inside the stub, by
summing an explicit joint table over the hidden sites given the tokens it is shown. The code under
test never sees the table; it can only get the right answer by serialising the pair, revealing each
endpoint's own residues one site at a time through the real ``window_log_probs``, and summing.

Two joints are used, both with a small supported alphabet (the stub puts probability zero on the
rest, so nothing here enumerates 20^3 outcomes):

* two sites, three symbols, weights ``[[6,2,2],[1,3,6],[3,5,2]] / 30``: both marginals are uniform,
  yet ``log P(11) - log P(00) = -ln 2``;
* three binary sites with parity structure: every even outcome has probability 0.20 and every odd
  one 0.05, so all single and pair marginals are uniform while ``log P(111) - log P(000) = -ln 4``.

A scorer that masked the sites simultaneously and multiplied marginals would report a gap of zero
for both; only a sequential reveal gets the joint. The order invariance of these joints says
nothing about whether a real masked language model is consistent in that way -- it only means any
order the objective picks must give the same number here, so a difference would be a bug.
"""
from __future__ import annotations

import dataclasses
import itertools
import json
import math

import numpy as np
import pytest
import torch

from smallAntibodyGen.experiments import her2_routes_data as data_lib
from smallAntibodyGen.experiments import her2_routes_serial as serial
from smallAntibodyGen.experiments import her2_routes_tasks as tasks_lib
from smallAntibodyGen.experiments.her2_multiscale import objective as objective_lib
from smallAntibodyGen.experiments.her2_multiscale import pairs as pairs_lib
from smallAntibodyGen.experiments.her2_routes_conditions import BINS, ConditionSpec
from smallAntibodyGen.experiments.her2_routes_services import ROUTE_DPLM_150M, FlightServices
from smallAntibodyGen.experiments.her2_routes_sources import SCALE_PKD
from smallAntibodyGen.experiments.her2_routes_split import ExclusionClosure
from smallAntibodyGen.tests.test_her2_routes_tasks import TinyEditor, TinyServices
from smallAntibodyGen.tests.test_her2_routes_tasks import _contrast as legacy_contrast

ROUTE = "dplm_150m_masked"
ASSAY = "synthetic:affinity:target"
HEAVY = "ACDEFGHIKLMNPQRSTVWY" * 2
LIGHT = "YWVTSRQPNMLKIHGFEDCA"
OTHER_BASE = HEAVY[:20] + "W" + HEAVY[21:]
CANONICAL = "ACDEFGHIKLMNPQRSTVWY"
MASK = serial.ABLANG_MASK_ID
OPEN = ExclusionClosure.build()

SITES1 = (("H", 3),)
SITES2 = (("H", 3), ("H", 7))
SITES3 = (("H", 3), ("H", 5), ("L", 2))
TABLE1 = {(0,): 0.7, (1,): 0.3}
THREE_SYMBOL = {(i, j): weight / 30
                for i, row in enumerate([[6, 2, 2], [1, 3, 6], [3, 5, 2]])
                for j, weight in enumerate(row)}
PARITY = {bits: 0.20 if sum(bits) % 2 == 0 else 0.05
          for bits in itertools.product((0, 1), repeat=3)}
CASES = {1: (SITES1, "AC", TABLE1, "A", "C"),
         2: (SITES2, "ACD", THREE_SYMBOL, "AA", "CC"),
         3: (SITES3, "AC", PARITY, "AAA", "CCC")}


def token_position(site):
    """Where the paired ``<VH>|<VL>`` layout puts a site, derived here from the layout itself."""
    chain, index = site
    return 1 + index if chain == "H" else len(HEAVY) + 4 + index


class JointOracle:
    """A masked 'model' that answers every query from an explicit joint over the edited sites."""

    def __init__(self, sites, alphabet, table):
        self.sites = tuple(sites)
        self.alphabet = alphabet
        self.table = dict(table)
        self.positions = [token_position(site) for site in self.sites]
        self.symbol_of_id = {serial.ABLANG_RESIDUE_IDS[residue]: symbol
                             for symbol, residue in enumerate(alphabet)}
        self.calls, self.probes = [], []

    def conditional(self, queried, visible):
        """``P(x_queried = s | visible)`` for every symbol: the joint summed over what is hidden."""
        weights = [0.0] * len(self.alphabet)
        for symbols, probability in self.table.items():
            if all(symbols[index] == symbol for index, symbol in visible.items()):
                weights[symbols[queried]] += probability
        total = sum(weights)
        return [weight / total for weight in weights]

    def canonical_log_probs(self, tokens, query_slots, *, condition_indices=None):
        tokens, slots = np.asarray(tokens), np.asarray(query_slots)
        self.calls.append((tokens.copy(), slots.copy()))
        log_probs = torch.full((tokens.shape[0], 1, 20), -60.0, dtype=torch.float64)
        for row in range(tokens.shape[0]):
            queried = self.positions.index(int(slots[row, 0]))
            assert int(tokens[row, slots[row, 0]]) == MASK, "a query must be a hidden site"
            visible = {index: self.symbol_of_id[int(tokens[row, position])]
                       for index, position in enumerate(self.positions)
                       if index != queried and int(tokens[row, position]) != MASK}
            for symbol, probability in enumerate(self.conditional(queried, visible)):
                log_probs[row, 0, CANONICAL.index(self.alphabet[symbol])] = math.log(probability)
        probe = torch.zeros_like(log_probs, requires_grad=True)
        self.probes.append(probe)
        return log_probs + probe


class Services:
    """The slice of ``FlightServices`` the objective calls, with the REAL sequential reveal."""

    device = "cpu"

    def __init__(self, *, assay_dropout=0.0):
        self.condition_spec = ConditionSpec(assays=(ASSAY,), assay_dropout=assay_dropout)
        self.batches = []

    def collate(self, route, views):
        batch = data_lib.collate_masked(views, pad_id=serial.ABLANG_PAD_ID,
                                        condition_spec=self.condition_spec)
        self.batches.append(batch)
        return batch

    def window_log_probs(self, route, model, batch):
        return FlightServices.window_log_probs(self, route, model, batch)


class Scripted:
    """A generator whose uniforms and permutations can be scripted and whose calls are counted."""

    def __init__(self, *, uniforms=(0.5,), permutations=None, seed=0):
        self._uniforms = itertools.cycle(uniforms)
        self._permutations = itertools.cycle(permutations) if permutations else None
        self._inner = np.random.default_rng(seed)
        self.random_calls = 0

    def random(self):
        self.random_calls += 1
        return next(self._uniforms)

    def permutation(self, size):
        if self._permutations is None:
            return self._inner.permutation(size)
        order = next(self._permutations)
        assert len(order) == size
        return np.asarray(order)


def summary(edits, *, value, region, base=HEAVY, partition="train"):
    heavy, light = list(base), list(LIGHT)
    for (chain, index), residue in edits.items():
        (heavy if chain == "H" else light)[index] = residue
    heavy, light = "".join(heavy), "".join(light)
    construct = serial.Construct(heavy=heavy, light=light, sites=tuple(region),
                                 source="synthetic", record_id=heavy[3:8])
    return data_lib.ConstructSummary(
        construct=construct, study="synthetic", assay=ASSAY, target="target",
        parent="synthetic::parent", family="affinity", kind="point",
        measurements=(data_lib.ReplicateMeasurement(
            value=value, lower=value, upper=value, censoring="none", replicate=1,
            measurement_id=f"m:{heavy}|{light}"),),
        region=tuple(region), mutated_sites=(), partition=partition, scale=SCALE_PKD,
        source="synthetic")


def pair_of(sites, residues_a, residues_b, *, value_a=7.0, value_b=8.0, base=HEAVY):
    """The pair whose first endpoint carries ``residues_a`` at ``sites`` and second ``residues_b``."""
    first = summary(dict(zip(sites, residues_a)), value=value_a, region=sites, base=base)
    second = summary(dict(zip(sites, residues_b)), value=value_b, region=sites, base=base)
    return pairs_lib.pair_between(first, second)


def no_window(construct):
    """``window_of`` for constructs outside the closure's edit space."""
    return None


def run(pair_list, model, *, order=None, kappa=2.0, services=None, rng=None, closure=OPEN,
        route=ROUTE, window_of=no_window, keys=None):
    services = services if services is not None else Services()
    if rng is None:
        rng = Scripted(permutations=None if order is None else [order])
    if keys is None:  # the training population: exactly the endpoints under test
        keys = frozenset(key for pair in pair_list if isinstance(pair, pairs_lib.MeasuredPair)
                         for key in pair.population_keys)
    outcome = objective_lib.pair_outcome(services, route, model, pair_list, closure=closure,
                                         window_of=window_of, training_population_keys=keys,
                                         kappa=kappa, rng=rng)
    return outcome, services


def marginal(table, fixed):
    return sum(probability for symbols, probability in table.items()
               if all(symbols[index] == symbol for index, symbol in fixed.items()))


# --- the fixtures are what the tests say they are --------------------------------------------------

def test_the_joints_have_the_properties_the_tests_rely_on():
    for index in (0, 1):
        for symbol in range(3):
            assert marginal(THREE_SYMBOL, {index: symbol}) == pytest.approx(1 / 3)
    assert sum(THREE_SYMBOL.values()) == pytest.approx(1.0)
    assert math.log(THREE_SYMBOL[(1, 1)]) - math.log(THREE_SYMBOL[(0, 0)]) \
        == pytest.approx(-math.log(2))
    assert sum(PARITY.values()) == pytest.approx(1.0)
    for size in (1, 2):
        for indices in itertools.combinations(range(3), size):
            for values in itertools.product((0, 1), repeat=size):
                assert marginal(PARITY, dict(zip(indices, values))) == pytest.approx(0.5 ** size)
    assert math.log(PARITY[(1, 1, 1)]) - math.log(PARITY[(0, 0, 0)]) == pytest.approx(-math.log(4))


# --- the sequential gap equals the joint log-ratio, in every order ------------------------------------

@pytest.mark.parametrize("order", list(itertools.permutations(range(2))))
def test_two_site_gap_matches_the_joint_in_every_order(order):
    pair = pair_of(SITES2, "AA", "CC", value_a=7.0, value_b=8.0)
    outcome, _ = run([pair], JointOracle(SITES2, "ACD", THREE_SYMBOL), order=order)
    joint_gap = math.log(THREE_SYMBOL[(1, 1)]) - math.log(THREE_SYMBOL[(0, 0)])
    assert outcome.diagnostics["predicted_gaps"][0] == pytest.approx(joint_gap, abs=1e-9)
    assert outcome.diagnostics["orders"] == [[list(SITES2[index]) for index in order]]
    assert float(outcome.loss.detach()) == pytest.approx((joint_gap - 2.0 * (8.0 - 7.0)) ** 2)


@pytest.mark.parametrize("order", list(itertools.permutations(range(2))))
def test_a_target_order_mistake_cannot_hide_behind_repeated_residues(order):
    """AC against CA: the endpoints differ in WHICH site carries which residue.

    With ``AA`` against ``CC`` a scorer that swapped the two sites' targets would still see the
    same pair of repeated residues. Here the swap changes the answer: the joint weights are
    ``P(A,C) = 2/30`` and ``P(C,A) = 1/30`` on uniform marginals, so ``J(CA) - J(AC) = -ln 2``,
    and a scorer that reveals sites in the wrong order or against the wrong residue reads another
    number (a simultaneous product of marginals reads zero).
    """
    pair = pair_of(SITES2, "AC", "CA", value_a=7.0, value_b=8.0)
    outcome, _ = run([pair], JointOracle(SITES2, "ACD", THREE_SYMBOL), order=order)
    assert THREE_SYMBOL[(1, 0)] / THREE_SYMBOL[(0, 1)] == pytest.approx(0.5)
    assert outcome.diagnostics["predicted_gaps"][0] == pytest.approx(-math.log(2), abs=1e-9)
    assert outcome.diagnostics["orders"] == [[list(SITES2[index]) for index in order]]
    reverse, _ = run([pair.oriented(pair.second.identity)], JointOracle(SITES2, "ACD", THREE_SYMBOL),
                     order=order)
    assert reverse.diagnostics["predicted_gaps"][0] == pytest.approx(math.log(2), abs=1e-9)


@pytest.mark.parametrize("order", list(itertools.permutations(range(3))))
def test_three_site_gap_matches_the_joint_in_every_order(order):
    pair = pair_of(SITES3, "AAA", "CCC", value_a=6.0, value_b=7.5)
    outcome, _ = run([pair], JointOracle(SITES3, "AC", PARITY), order=order)
    joint_gap = math.log(0.05) - math.log(0.20)
    assert outcome.diagnostics["predicted_gaps"][0] == pytest.approx(joint_gap, abs=1e-9)
    assert outcome.diagnostics["orders"] == [[list(SITES3[index]) for index in order]]
    assert outcome.diagnostics["target_gaps"] == [pytest.approx(2.0 * 1.5)]


def test_distance_one_is_the_conditional_log_odds():
    pair = pair_of(SITES1, "A", "C", value_a=7.0, value_b=7.5)
    outcome, _ = run([pair], JointOracle(SITES1, "AC", TABLE1))
    assert outcome.diagnostics["predicted_gaps"][0] \
        == pytest.approx(math.log(0.3) - math.log(0.7), abs=1e-9)


# --- what the model is shown at each step -------------------------------------------------------------------

@pytest.mark.parametrize("order", list(itertools.permutations(range(3))))
def test_step_one_is_shared_and_every_later_step_reveals_each_endpoints_own_residues(order):
    pair = pair_of(SITES3, "AAA", "CCC")
    model = JointOracle(SITES3, "AC", PARITY)
    run([pair], model, order=order)
    positions = [token_position(site) for site in SITES3]
    own = (serial.ABLANG_RESIDUE_IDS["A"], serial.ABLANG_RESIDUE_IDS["C"])
    assert len(model.calls) == 3
    for step, (tokens, slots) in enumerate(model.calls):
        assert tokens.shape[0] == 2  # row 0 is the first endpoint, row 1 the second
        queried = positions[order[step]]
        assert list(slots[:, 0]) == [queried, queried]
        for index in order[:step]:  # every predecessor, with that endpoint's own residue
            assert tokens[0, positions[index]] == own[0] and tokens[1, positions[index]] == own[1]
        for index in order[step:]:  # the queried site and everything after it still hidden
            assert tokens[0, positions[index]] == MASK and tokens[1, positions[index]] == MASK
        background = [column for column in range(tokens.shape[1]) if column not in positions]
        assert np.array_equal(tokens[0, background], tokens[1, background])
        assert not (tokens[:, background] == MASK).any()  # no hidden unchanged background
    first_step = model.calls[0][0]
    assert np.array_equal(first_step[0], first_step[1])


# --- orientation and gradients --------------------------------------------------------------------------------

def test_reversing_the_endpoints_negates_the_gap_and_keeps_the_loss():
    pair = pair_of(SITES3, "AAA", "CCC", value_a=6.5, value_b=8.1)
    forward, _ = run([pair], JointOracle(SITES3, "AC", PARITY), order=(2, 0, 1))
    backward, _ = run([pair.oriented(pair.second.identity)], JointOracle(SITES3, "AC", PARITY),
                      order=(2, 0, 1))
    assert backward.diagnostics["predicted_gaps"] \
        == pytest.approx([-gap for gap in forward.diagnostics["predicted_gaps"]])
    assert backward.diagnostics["target_gaps"] \
        == pytest.approx([-gap for gap in forward.diagnostics["target_gaps"]])
    torch.testing.assert_close(backward.loss, forward.loss)


def test_gradient_reaches_every_sequential_conditional_at_the_right_residue():
    pair = pair_of(SITES3, "AAA", "CCC", value_a=7.0, value_b=8.0)
    model = JointOracle(SITES3, "AC", PARITY)
    outcome, _ = run([pair], model, order=(1, 2, 0), kappa=2.0)
    outcome.loss.backward()
    residual = (math.log(0.05) - math.log(0.20)) - 2.0 * (8.0 - 7.0)
    assert len(model.probes) == 3  # one conditional per scored site, all in the graph
    for probe in model.probes:
        assert probe.grad is not None
        for row, residue, sign in ((0, "A", -1.0), (1, "C", +1.0)):
            expected = np.zeros(20)
            expected[CANONICAL.index(residue)] = sign * 2.0 * residual
            np.testing.assert_allclose(probe.grad[row, 0].numpy(), expected, atol=1e-9)


# --- parity with the existing one-site objective ------------------------------------------------------------------

def test_distance_one_matches_the_legacy_local_effect_loss():
    services = Services()
    torch.manual_seed(20260927)
    model = TinyEditor(condition_sizes=services.condition_spec.sizes)
    margin, region = 0.5, (("H", 3), ("H", 7))
    legacy = tasks_lib.local_outcome(
        TinyServices(), "ablang2_masked_editor", model, [legacy_contrast(margin)], kappa=2.0,
        reconstruction_weight=0.0)
    lower = summary({("H", 3): "M"}, value=8.0 - margin, region=region)
    higher = summary({("H", 3): "W"}, value=8.0, region=region)
    outcome, _ = run([pairs_lib.pair_between(lower, higher)], model, services=services)
    torch.testing.assert_close(outcome.loss, legacy.loss)
    assert outcome.term == legacy.term == "local_effect"


# --- conditioning --------------------------------------------------------------------------------------------------------

def test_dropout_is_drawn_once_per_pair_and_shared_by_both_endpoints():
    pair_list = [pair_of(SITES1, "A", "C"), pair_of(SITES1, "A", "C", base=OTHER_BASE)]
    services = Services(assay_dropout=0.99)
    rng = Scripted(uniforms=(0.0, 0.995))  # the first pair drops its assay, the second keeps it
    run(pair_list, JointOracle(SITES1, "AC", TABLE1), services=services, rng=rng)
    assert rng.random_calls == 2
    indices = services.batches[0].condition_indices
    order = services.condition_spec.assay_order
    assert indices[0, 2] == indices[1, 2] == order.index("null")
    assert indices[2, 2] == indices[3, 2] == order.index(ASSAY)
    assert (indices[:, 1] == BINS.index("null")).all()  # quality is null for every row
    assert (indices[:, 0] == indices[0, 0]).all()


# --- accounting ----------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("distance", [1, 2, 3])
def test_a_pair_costs_two_d_forwards_and_two_d_targets(distance):
    sites, alphabet, table, left, right = CASES[distance]
    pair_list = [pair_of(sites, left, right), pair_of(sites, left, right, base=OTHER_BASE)]
    model = JointOracle(sites, alphabet, table)
    outcome, _ = run(pair_list, model)
    assert outcome.rows == 4
    assert outcome.forwards == 2 * distance * 2 and outcome.targets == 2 * distance * 2
    assert len(model.calls) == distance and all(tokens.shape[0] == 4 for tokens, _ in model.calls)
    contributions = outcome.exposure()
    assert sum(item.forwards for item in contributions) == outcome.forwards
    assert sum(item.targets for item in contributions) == outcome.targets
    assert sum(item.rows for item in contributions) == outcome.rows
    assert outcome.identities == tuple(identity for pair in pair_list
                                       for identity in pair.identities)
    assert outcome.measured and outcome.family == data_lib.FAMILY_HER2
    assert outcome.diagnostics["distance"] == distance
    json.dumps(outcome.document())


def test_the_outcome_names_the_numeric_term_and_nothing_else():
    pair = pair_of(SITES2, "AA", "CC")
    outcome, _ = run([pair], JointOracle(SITES2, "ACD", THREE_SYMBOL))
    assert outcome.term == objective_lib.TERM == "local_effect"
    assert set(outcome.terms) == {"local_effect"}
    assert outcome.terms["local_effect"] == pytest.approx(float(outcome.loss.detach()))
    assert outcome.diagnostics["mode"] == "absolute"


def test_the_loss_is_the_mean_over_pairs():
    pair_list = [pair_of(SITES2, "AA", "CC", value_a=7.0, value_b=8.0),
                 pair_of(SITES2, "AA", "CC", value_a=7.0, value_b=5.0, base=OTHER_BASE)]
    outcome, _ = run([*pair_list], JointOracle(SITES2, "ACD", THREE_SYMBOL))
    gap = -math.log(2)
    expected = ((gap - 2.0) ** 2 + (gap + 4.0) ** 2) / 2
    assert float(outcome.loss.detach()) == pytest.approx(expected)


# --- the reveal order ------------------------------------------------------------------------------------------------------------------

def test_the_reveal_order_comes_from_the_explicit_generator():
    def order_for(seed):
        pair = pair_of(SITES3, "AAA", "CCC")
        outcome, _ = run([pair], JointOracle(SITES3, "AC", PARITY),
                         rng=np.random.default_rng(seed))
        return tuple(tuple(site) for site in outcome.diagnostics["orders"][0])
    assert order_for(5) == order_for(5)
    assert {order_for(seed) for seed in range(60)} \
        == {tuple(SITES3[index] for index in order) for order in itertools.permutations(range(3))}


def test_no_global_generator_is_read_or_advanced():
    pair = pair_of(SITES3, "AAA", "CCC")
    numpy_before, torch_before = np.random.get_state(), torch.get_rng_state()
    run([pair], JointOracle(SITES3, "AC", PARITY), rng=np.random.default_rng(1))
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0] and np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert torch.equal(torch_before, torch.get_rng_state())


# --- refusals ------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("route", ["ablang2_masked_editor", "piggen_causal_infiller",
                                   "dwjs_energy", "dplm_650m_lora_masked"])
def test_only_the_dplm_150m_route_is_wired(route):
    with pytest.raises(ValueError, match="wired for"):
        run([pair_of(SITES2, "AA", "CC")], JointOracle(SITES2, "ACD", THREE_SYMBOL), route=route)


def test_the_route_constant_matches_the_services():
    assert objective_lib.ROUTE_DPLM_150M == ROUTE_DPLM_150M == ROUTE


def test_a_distance_above_three_is_refused_before_any_forward():
    pair = dataclasses.replace(pair_of(SITES3, "AAA", "CCC"), distance=4)
    model = JointOracle(SITES3, "AC", PARITY)
    with pytest.raises(ValueError, match="distance"):
        run([pair], model)
    assert model.calls == []


def test_one_microbatch_has_one_distance():
    with pytest.raises(ValueError, match="one distance"):
        run([pair_of(SITES1, "A", "C"), pair_of(SITES2, "AA", "CC")],
            JointOracle(SITES2, "ACD", THREE_SYMBOL))


@pytest.mark.parametrize("kappa", [0, 0.0, -1.0, float("nan"), float("inf"), True, "2", None])
def test_kappa_must_be_a_finite_positive_number(kappa):
    with pytest.raises(ValueError, match="kappa"):
        run([pair_of(SITES2, "AA", "CC")], JointOracle(SITES2, "ACD", THREE_SYMBOL), kappa=kappa)


def test_the_generator_is_required_and_the_batch_is_not_empty():
    model = JointOracle(SITES2, "ACD", THREE_SYMBOL)
    services = Services()
    pair = pair_of(SITES2, "AA", "CC")
    with pytest.raises(ValueError, match="explicit random generator"):
        objective_lib.pair_outcome(services, ROUTE, model, [pair], closure=OPEN,
                                   window_of=no_window,
                                   training_population_keys=frozenset(pair.population_keys),
                                   kappa=2.0, rng=None)
    with pytest.raises(ValueError, match="no pairs"):
        run([], model)
    with pytest.raises(ValueError, match="MeasuredPair"):
        run([{"distance": 2}], model)


def test_a_stale_pair_is_refused_before_any_forward():
    pair = pair_of(SITES2, "AA", "CC")
    pair.second.partition = "development"
    model, services = JointOracle(SITES2, "ACD", THREE_SYMBOL), Services()
    with pytest.raises(ValueError, match="stale"):
        run([pair], model, services=services)
    assert model.calls == [] and services.batches == []


def test_the_closure_is_applied_at_consumption():
    pair = pair_of(SITES2, "AA", "CC")
    closure = ExclusionClosure.build(reserved_identities=[pair.first.identity])
    model = JointOracle(SITES2, "ACD", THREE_SYMBOL)
    with pytest.raises(ValueError, match="stale"):
        run([pair], model, closure=closure)
    assert model.calls == []


def test_the_closure_window_comes_from_the_callers_callback():
    pair = pair_of(SITES2, "AA", "CC")
    window = pair.first.construct.residues(SITES2)
    closure = ExclusionClosure.build(reserved_windows=[window])
    run([pair], JointOracle(SITES2, "ACD", THREE_SYMBOL), closure=closure)  # no window: not tested
    model = JointOracle(SITES2, "ACD", THREE_SYMBOL)
    with pytest.raises(ValueError, match="closure_refused"):
        run([pair], model, closure=closure,
            window_of=lambda construct: construct.residues(SITES2))
    assert model.calls == []


def test_a_train_labelled_pair_outside_the_training_population_is_refused_before_any_forward():
    pair = pair_of(SITES2, "AA", "CC")
    assert pair.first.partition == pair.second.partition == "train"
    model, services = JointOracle(SITES2, "ACD", THREE_SYMBOL), Services()
    keys = frozenset(pair.population_keys) - {pair.second.population_key}
    with pytest.raises(ValueError, match="not_training_population"):
        run([pair], model, services=services, keys=keys)
    assert model.calls == [] and services.batches == []
    with pytest.raises(TypeError):  # the boundary is a required argument, never a default
        objective_lib.pair_outcome(services, ROUTE, model, [pair], closure=OPEN,
                                   window_of=no_window, kappa=2.0, rng=Scripted())
