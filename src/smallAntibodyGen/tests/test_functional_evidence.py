"""Build B contracts: the exact finite-world teacher, the frozen q7 split and the function families.

The mathematics is RE-DERIVED here with independent enumeration (plain loops and lambdas, never the
module's own helpers) and then compared with the module. These tests qualify the teacher and the frozen
world; they say nothing about any learned model, and every task is synthetic.
"""
from __future__ import annotations

import itertools
import json
import math
from fractions import Fraction

import numpy as np
import pytest

from smallAntibodyGen.experiments import functional_evidence as fe

OBS = fe.Observation


# ---------------------------------------------------------------------------
# independent references
# ---------------------------------------------------------------------------

def bit(world, i):
    return (world >> i) & 1


def brute_posterior(events, prior=None):
    """Apply every event one at a time to each of the 16 worlds with exact arithmetic."""
    seen = {}
    for e in events:
        if e.event_id in seen:
            assert seen[e.event_id] == (e.fact, e.bit, e.reliability)
            continue
        seen[e.event_id] = (e.fact, e.bit, e.reliability)
    weights = []
    for w in range(16):
        weight = Fraction(1, 16) if prior is None else Fraction(prior[w])
        for fact, b, r in seen.values():
            r = Fraction(r)
            weight *= r if bit(w, fact) == b else 1 - r
        weights.append(weight)
    total = sum(weights)
    return [x / total for x in weights]


def run(node, world):
    """An independent evaluator of the typed AST."""
    if isinstance(node, fe.Read):
        return bit(world, node.index)
    if isinstance(node, fe.Not):
        return 1 - run(node.child, world)
    if isinstance(node, fe.And):
        return run(node.left, world) * run(node.right, world)
    if isinstance(node, fe.Xor):
        return (run(node.left, world) + run(node.right, world)) % 2
    raise AssertionError(node)


def table_of(function):
    return sum(function(tuple(bit(w, i) for i in range(4))) << w for w in range(16))


def renamed(table, perm):
    out = 0
    for w in range(16):
        y = sum(bit(w, perm[i]) << i for i in range(4))
        out |= bit(table, y) << w
    return out


def orbit(table):
    return min(renamed(table, p) for p in itertools.permutations(range(4)))


def distinct_variables(count):
    return itertools.permutations(range(4), count)


# the families, written independently as plain Boolean lambdas over the four fact bits
INDEPENDENT = {
    "source": {
        "READ": (1, lambda x, v: x[v[0]]),
        "NOT_READ": (1, lambda x, v: 1 - x[v[0]]),
        "AND2": (2, lambda x, v: x[v[0]] & x[v[1]]),
        "XOR2": (2, lambda x, v: x[v[0]] ^ x[v[1]]),
        "NOT_XOR2": (2, lambda x, v: 1 - (x[v[0]] ^ x[v[1]])),
        "AND3": (3, lambda x, v: x[v[0]] & x[v[1]] & x[v[2]]),
        "XOR3": (3, lambda x, v: x[v[0]] ^ x[v[1]] ^ x[v[2]]),
        "AND4": (4, lambda x, v: x[v[0]] & x[v[1]] & x[v[2]] & x[v[3]]),
        "XOR4": (4, lambda x, v: x[v[0]] ^ x[v[1]] ^ x[v[2]] ^ x[v[3]]),
        "AND_ANDNOT": (3, lambda x, v: x[v[0]] & x[v[1]] & (1 - x[v[2]]))},
    "primary": {
        "AND_XOR_READ": (3, lambda x, v: (x[v[0]] ^ x[v[1]]) & x[v[2]]),
        "AND_XOR_XOR": (4, lambda x, v: (x[v[0]] ^ x[v[1]]) & (x[v[2]] ^ x[v[3]])),
        "AND_XOR_NOT": (3, lambda x, v: (x[v[0]] ^ x[v[1]]) & (1 - x[v[2]]))},
    "validation": {"NOT_AND2": (2, lambda x, v: 1 - (x[v[0]] & x[v[1]]))},
    "transfer": {"XOR_AND_READ": (3, lambda x, v: (x[v[0]] & x[v[1]]) ^ x[v[2]])}}


def independent_tables():
    out = {}
    for role, families in INDEPENDENT.items():
        out[role] = {}
        for name, (arity, function) in families.items():
            out[role][name] = {table_of(lambda x, v=v, f=function: f(x, v))
                               for v in distinct_variables(arity)}
    return out


def independent_q7():
    centers = [(a, b, (a + b) % 7, (a + 2 * b) % 7) for a in range(7) for b in range(7)]

    def distance(s, c):
        return sum(1 for p, q in zip(s, c) if p != q)
    everything = list(itertools.product(range(7), repeat=4))
    excluded = {s for s in everything if min(distance(s, c) for c in centers) <= 1}
    source = [s for s in everything if s not in excluded]
    return centers, excluded, source, distance


# ---------------------------------------------------------------------------
# the teacher: hand cases
# ---------------------------------------------------------------------------

def test_hand_cases_are_exact_equalities():
    read0, and01 = fe.Read(0), fe.And(fe.Read(0), fe.Read(1))
    assert fe.program_probability(read0, ()) == Fraction(1, 2)
    assert fe.program_probability(and01, ()) == Fraction(1, 4)
    one = [OBS("a", 0, 1, 0.75)]
    assert fe.program_probability(read0, one) == Fraction(3, 4)
    two = [OBS("a", 0, 1, 0.75), OBS("b", 0, 1, 0.75)]
    assert fe.program_probability(read0, two) == Fraction(9, 10)
    opposed = [OBS("a", 0, 1, 0.75), OBS("b", 0, 0, 0.75)]
    assert fe.program_probability(read0, opposed) == Fraction(1, 2)
    assert fe.program_probability(read0, [OBS("a", 0, 1, 0.5)]) == Fraction(1, 2)
    assert fe.world_posterior([OBS("a", 0, 1, 0.5), OBS("b", 2, 0, 0.5)]) == fe.UNIFORM_PRIOR.weights


def test_a_cancelling_pair_preserves_the_whole_posterior():
    base = [OBS("a", 0, 1, 0.75), OBS("z", 3, 0, 0.75)]
    padded = base + [OBS("p", 0, 1, 0.75), OBS("q", 0, 0, 0.75), OBS("n", 1, 1, 0.5)]
    assert fe.world_posterior(base) == fe.world_posterior(padded)


def test_the_posterior_matches_an_independent_brute_force_on_random_records():
    rng = np.random.default_rng(3)
    for trial in range(40):
        events = [OBS(f"e{trial}.{i}", int(rng.integers(0, 4)), int(rng.integers(0, 2)),
                      float(rng.choice([0.25, 0.5, 0.75, 0.9, 0.6]))) for i in range(int(rng.integers(0, 9)))]
        assert list(fe.world_posterior(events)) == brute_posterior(events)


def test_a_non_uniform_prior_is_applied_exactly():
    weights = [Fraction(i + 1, 136) for i in range(16)]
    prior = fe.explicit_prior([float(w) for w in weights])
    events = [OBS("a", 1, 1, 0.75), OBS("b", 2, 0, 0.9)]
    expected = brute_posterior(events, prior.weights)
    assert list(fe.world_posterior(events, prior)) == expected


def test_duplicate_ids_count_once_and_conflicts_refuse():
    one = [OBS("a", 0, 1, 0.75)]
    assert fe.world_posterior(one + one) == fe.world_posterior(one)
    assert fe.world_posterior(one + [OBS("b", 0, 1, 0.75)]) != fe.world_posterior(one)
    with pytest.raises(ValueError, match="conflicting"):
        fe.world_posterior(one + [OBS("a", 0, 0, 0.75)])
    with pytest.raises(ValueError, match="conflicting"):
        fe.world_posterior(one + [OBS("a", 0, 1, 0.9)])
    assert len(fe.canonical_events(one + one)) == 1


@pytest.mark.parametrize("fields", [
    ("", 0, 1, 0.5), (" a", 0, 1, 0.5), (3, 0, 1, 0.5), ("a", True, 1, 0.5), ("a", 4, 1, 0.5),
    ("a", -1, 1, 0.5), ("a", 0, True, 0.5), ("a", 0, 2, 0.5), ("a", 0, 1.0, 0.5),
    ("a", 0, 1, True), ("a", 0, 1, -0.1), ("a", 0, 1, 1.1), ("a", 0, 1, float("nan")),
    ("a", 0, 1, float("inf")), ("a", 0, 1, "0.5"), ("a", 0.0, 1, 0.5)])
def test_observations_are_validated_strictly(fields):
    with pytest.raises(ValueError):
        fe.Observation(*fields)


def test_numpy_integers_are_accepted_and_numpy_bools_refused():
    assert fe.Observation("a", np.int64(2), np.int8(1), np.float32(0.5)).fact == 2
    with pytest.raises(ValueError):
        fe.Observation("a", np.True_, 1, 0.5)


def test_hard_evidence_is_handled_and_inconsistent_hard_evidence_refuses():
    assert fe.program_probability(fe.Read(0), [OBS("a", 0, 1, 1.0)]) == 1
    assert fe.program_probability(fe.Read(0), [OBS("a", 0, 1, 0.0)]) == 0
    with pytest.raises(ValueError, match="inconsistent"):
        fe.world_posterior([OBS("a", 0, 1, 1.0), OBS("b", 0, 0, 1.0)])
    # evidence that contradicts a prior's structural zero is also refused, never a default
    only_even_x0 = fe.explicit_prior([Fraction(1, 8) if w % 2 == 0 else Fraction(0)
                                      for w in range(16)])
    with pytest.raises(ValueError, match="inconsistent"):
        fe.world_posterior([OBS("a", 0, 1, 1.0)], only_even_x0)


def test_realized_entropy_can_increase_after_contradiction():
    def entropy(p):
        p = float(p)
        return -(p * math.log(p) + (1 - p) * math.log(1 - p))
    confident = [OBS(f"p{i}", 0, 1, 0.75) for i in range(3)]
    contradicted = confident + [OBS(f"n{i}", 0, 0, 0.75) for i in range(3)]
    before = fe.program_probability(fe.Read(0), confident)
    after = fe.program_probability(fe.Read(0), contradicted)
    assert before == Fraction(27, 28) and after == Fraction(1, 2)
    assert entropy(after) > entropy(before)


def test_prior_validation():
    with pytest.raises(ValueError):
        fe.explicit_prior([0.1] * 16)
    with pytest.raises(ValueError):
        fe.explicit_prior([1.1, -0.1] + [0.0] * 14)          # sums to one but one weight is negative
    with pytest.raises(ValueError):
        fe.explicit_prior([True] + [0.0] * 15)
    with pytest.raises(ValueError):
        fe.explicit_prior([float("nan")] * 16)
    with pytest.raises(ValueError):
        fe.explicit_prior([0.5, 0.5])
    exact = fe.explicit_prior([1 / 16] * 16)
    assert sum(exact.weights) == 1


# ---------------------------------------------------------------------------
# correlated evidence and the factor teacher
# ---------------------------------------------------------------------------

def test_a_correlated_prior_has_half_marginals_but_and_is_one_half_not_one_quarter():
    prior = fe.correlated_prior_x0_equals_x1()
    marginals = fe.marginals(prior.weights)
    assert marginals == (Fraction(1, 2),) * 4
    assert fe.program_probability(fe.And(fe.Read(0), fe.Read(1)), (), prior) == Fraction(1, 2)
    assert fe.program_probability(fe.And(fe.Read(0), fe.Read(1)), ()) == Fraction(1, 4)
    assert not fe.is_product_distribution(prior.weights)
    assert fe.is_product_distribution(fe.UNIFORM_PRIOR.weights)


def test_the_factor_teacher_refuses_a_correlated_prior_and_never_multiplies_marginals_silently():
    prior = fe.correlated_prior_x0_equals_x1()
    with pytest.raises(ValueError, match="not a product"):
        fe.interchange_probability((), (), 0, fe.And(fe.Read(0), fe.Read(1)), prior)
    with pytest.raises(ValueError, match="not a product"):
        fe.require_independent_factor_regime(prior)


def test_the_factor_teacher_equals_the_full_teacher_on_the_raw_hybrid_for_product_priors():
    # a product prior with P(X0 = 1) = 1/3 and the other facts uniform, passed as EXACT fractions
    nonuniform = fe.explicit_prior([(Fraction(1, 3) if bit(w, 0) else Fraction(2, 3)) * Fraction(1, 8)
                                    for w in range(16)])
    for prior in (fe.UNIFORM_PRIOR, nonuniform):
        assert fe.is_product_distribution(prior.weights)
        base = fe.raw_events((0, 4, 2, 6), 1, prefix="base")
        donor = fe.raw_events((6, 1, 3, 2), 2, prefix="donor")
        for site in range(4):
            for program in (fe.And(fe.Read(0), fe.Read(1)), fe.Xor(fe.Read(site), fe.Read(2)),
                            fe.Not(fe.Read(site))):
                hybrid = fe.hybrid_events(base, donor, site)
                assert (fe.interchange_probability(base, donor, site, program, prior)
                        == fe.program_probability(program, hybrid, prior))


# ---------------------------------------------------------------------------
# programs
# ---------------------------------------------------------------------------

def test_program_nodes_are_typed_with_strict_arity_and_indices():
    for bad in (True, 4, -1, 1.0, "0"):
        with pytest.raises(ValueError):
            fe.Read(bad)
    with pytest.raises(ValueError):
        fe.Not(3)
    with pytest.raises(ValueError):
        fe.And(fe.Read(0), "x")
    with pytest.raises(TypeError):
        fe.And(fe.Read(0))
    with pytest.raises(ValueError):
        fe.program_from_document({"op": "NOT", "args": [{"op": "READ", "index": 0},
                                                         {"op": "READ", "index": 1}]})
    with pytest.raises(ValueError):
        fe.program_from_document({"op": "AND", "args": [{"op": "READ", "index": 0}]})
    with pytest.raises(ValueError):
        fe.program_from_document({"op": "OR", "args": []})
    with pytest.raises(ValueError):
        fe.program_from_document({"op": "READ", "index": 0, "extra": 1})
    parsed = fe.program_from_document({"op": "XOR", "args": [
        {"op": "READ", "index": 2}, {"op": "NOT", "args": [{"op": "READ", "index": 3}]}]})
    assert parsed == fe.Xor(fe.Read(2), fe.Not(fe.Read(3)))


def test_known_truth_tables_and_the_executor_agree_with_the_independent_one():
    assert fe.truth_table(fe.Read(0)) == 0xAAAA
    assert fe.truth_table(fe.Not(fe.Read(0))) == 0x5555
    assert fe.truth_table(fe.And(fe.Read(0), fe.Read(1))) == 0x8888
    assert fe.truth_table(fe.Xor(fe.Read(0), fe.Read(1))) == 0x6666
    program = fe.And(fe.Xor(fe.Read(0), fe.Read(1)), fe.Not(fe.Read(2)))
    assert fe.truth_table(program) == sum(run(program, w) << w for w in range(16))


def test_token_serialization_round_trips_and_malformed_sequences_refuse():
    for function in fe.function_registry().functions:
        assert fe.program_from_tokens(function.tokens) == function.program
        assert fe.program_from_tokens(list(function.tokens)) == function.program
    for bad in ([], [fe.TOKEN_AND, 1], [1, 1], [fe.TOKEN_NOT], [8], [True]):
        with pytest.raises(ValueError):
            fe.program_from_tokens(bad)


def test_depends_on_is_semantic():
    assert fe.depends_on(fe.truth_table(fe.Read(1)), 1)
    assert not fe.depends_on(fe.truth_table(fe.Read(1)), 0)
    # XOR(x0, x0) never reads another fact and is the constant 0: no dependence at all
    constant = fe.truth_table(fe.Xor(fe.Read(0), fe.Read(0)))
    assert constant == 0 and not any(fe.depends_on(constant, k) for k in range(4))


# ---------------------------------------------------------------------------
# function families: counts, orbits, no leakage
# ---------------------------------------------------------------------------

def test_function_families_match_an_independent_enumeration():
    registry = fe.function_registry()
    tables = independent_tables()
    for role, families in tables.items():
        for name, expected in families.items():
            got = {item.table for item in registry.of_family(name)}
            assert got == expected, name
        union = set().union(*families.values())
        assert registry.tables(role) == union
        assert registry.orbits(role) == {orbit(t) for t in union}
    assert {role: len(registry.role(role)) for role in fe.FUNCTION_ROLES} == {
        "source": 48, "primary": 27, "validation": 6, "transfer": 12}
    assert {role: len(registry.orbits(role)) for role in fe.FUNCTION_ROLES} == {
        "source": 10, "primary": 3, "validation": 1, "transfer": 1}


def test_no_truth_table_or_orbit_leaks_between_roles_and_aliases_stay_in_their_family():
    registry = fe.function_registry()
    for left, right in itertools.combinations(fe.FUNCTION_ROLES, 2):
        assert not registry.tables(left) & registry.tables(right)
        assert not registry.orbits(left) & registry.orbits(right)
    owner = {}
    for family, tables in registry.alias_tables.items():
        for table in tables:
            assert owner.setdefault(table, family) == family
    for family, programs in registry.alias_programs.items():
        for program in programs:
            table = sum(run(program, w) << w for w in range(16))
            assert table in registry.alias_tables[family] and owner[table] == family
    assert len(registry.alias_programs["AND4"]) > len(registry.alias_tables["AND4"])


def test_the_registry_builds_and_a_cross_family_alias_refuses_with_its_own_message(monkeypatch):
    """The first table of the first family used to raise ``KeyError`` from an eagerly evaluated error
    message; the cross-family check itself must still refuse, with the intended ValueError."""
    built = fe._build_registry()
    assert len(built.functions) == 93 and built.program_width == fe.function_registry().program_width
    assert [f.table for f in built.functions] == [f.table for f in fe.function_registry().functions]
    twin = fe.FamilyDef("READ_TWIN", fe.ROLE_SOURCE, 1, 4, fe._b_read)
    monkeypatch.setattr(fe, "FAMILY_DEFS", fe.FAMILY_DEFS + (twin,))
    with pytest.raises(ValueError, match="belongs to both READ and READ_TWIN"):
        fe._build_registry()


def test_complementing_inputs_or_negating_the_output_is_not_an_equivalence():
    registry = fe.function_registry()
    xor2 = {f.table for f in registry.of_family("XOR2")}
    xnor = {f.table for f in registry.of_family("NOT_XOR2")}
    assert xor2.isdisjoint(xnor)
    assert {orbit(t) for t in xor2}.isdisjoint({orbit(t) for t in xnor})
    and2 = {f.table for f in registry.of_family("AND2")}
    nand = {f.table for f in registry.of_family("NOT_AND2")}
    assert {orbit(t) for t in and2}.isdisjoint({orbit(t) for t in nand})


def test_every_primitive_occurs_in_the_source_set_and_canonical_programs_are_unambiguous():
    registry = fe.function_registry()
    kinds = {type(node).__name__ for f in registry.role("source") for node in _nodes(f.program)}
    assert {"Read", "Not", "And", "Xor"} <= kinds
    assert len({f.tokens for f in registry.functions}) == len(registry.functions)
    assert len({f.function_id for f in registry.functions}) == len(registry.functions)
    assert [f.index for f in registry.functions] == list(range(93))
    assert registry.program_width == max(len(f.tokens) for f in registry.functions)
    for f in registry.functions:
        assert f.table == sum(run(f.program, w) << w for w in range(16))


def _nodes(program):
    yield program
    for child in (getattr(program, "child", None), getattr(program, "left", None),
                  getattr(program, "right", None)):
        if child is not None:
            yield from _nodes(child)


def test_orbit_function_agrees_with_the_independent_orbit():
    for function in fe.function_registry().functions:
        assert function.orbit == orbit(function.table)
        assert fe.orbit_key(function.table) == orbit(function.table)


# ---------------------------------------------------------------------------
# the q7 split
# ---------------------------------------------------------------------------

def test_q7_counts_match_an_independent_enumeration():
    centers, excluded, source, distance = independent_q7()
    world = fe.q7_world()
    assert len(centers) == 49 and set(world.centers) == set(centers)
    assert min(distance(a, b) for a, b in itertools.combinations(centers, 2)) == 3
    assert len(excluded) == 1225 and world.excluded == frozenset(excluded)
    assert len(source) == 1176 and list(world.source) == source
    for site in range(4):
        for level in range(7):
            assert sum(1 for s in source if s[site] == level) == 168
    source_set = set(source)
    legal_counts = []
    for s in source:
        for site in range(4):
            legal = [l for l in range(7) if s[:site] + (l,) + s[site + 1:] in source_set]
            legal_counts.append(len(legal))
            assert s[site] in legal and tuple(legal) == world.legal_levels(s, site)
    assert set(legal_counts) == {4} and len(legal_counts) == 4704
    summary = fe.verify_q7()
    assert summary["source"] == 1176 and summary["excluded"] == 1225


def test_q7_strata_are_defined_by_t_and_each_level_is_balanced():
    centers, _, _, _ = independent_q7()
    world = fe.q7_world()
    expected = {0: "primary", 1: "primary", 2: "validation", 3: "transfer_fit", 4: "transfer_fit",
                5: "transfer_eval", 6: "transfer_eval"}
    for center in centers:
        t = (center[1] - 2 * center[0]) % 7
        assert world.split_of_center[center] == expected[t]
    sizes = {name: len(members) for name, members in world.centers_by_split.items()}
    assert sizes == {"primary": 14, "validation": 7, "transfer_fit": 14, "transfer_eval": 14}
    for name, members in world.centers_by_split.items():
        for site in range(4):
            counts = {level: sum(1 for c in members if c[site] == level) for level in range(7)}
            assert set(counts.values()) == {len(members) // 7}


def test_all_hybrids_stay_in_the_source_for_source_bases_and_leave_it_for_primary_bases():
    world = fe.q7_world()
    source = set(world.source)
    for base in world.source[::7]:
        for site in range(4):
            legal = set(world.legal_levels(base, site))
            for donor in world.source[::53]:
                hybrid = fe.hybrid_signature(base, donor, site)
                assert (hybrid in source) == (donor[site] in legal)
    for base in world.centers_by_split["primary"]:
        for donor in world.centers_by_split["primary"]:
            for site in range(4):
                assert fe.hybrid_signature(base, donor, site) not in source
    assert len(world.centers_by_split["primary"]) ** 2 * 4 * 27 == 21168


def test_the_split_holds_out_labelled_contexts_not_permutation_orbits():
    world = fe.q7_world()
    assert (1, 2, 3, 5) in world.centers
    assert (1, 2, 5, 3) in set(world.source)
    assert (1, 2, 5, 3) not in world.excluded
    document = fe.world_document()
    assert "ordered by fact identity" in document["claim_boundary"]


def test_panel_counts_are_derived_from_the_frozen_world_and_verified_before_training():
    counts = fe.panel_counts()
    assert counts == {
        "source_context_query_pairs": 56448, "validation_new_function": 42,
        "validation_source_function": 336, "primary_composition": 378,
        "primary_source_preservation": 672, "transfer_fit_candidates": 168,
        "transfer_eval_candidates": 168, "strong_primary_interventions": 21168}
    summary = fe.verify_world()
    assert summary["panels"] == counts and summary["source"] == 1176 and summary["excluded"] == 1225


def test_declared_panel_counts():
    registry = fe.function_registry()
    n = {role: len(registry.role(role)) for role in fe.FUNCTION_ROLES}
    world = fe.q7_world()
    c = {name: len(members) for name, members in world.centers_by_split.items()}
    assert len(world.source) * n["source"] == 56448
    assert c["validation"] * n["validation"] == 42 and c["validation"] * n["source"] == 336
    assert c["primary"] * n["primary"] == 378 and c["primary"] * n["source"] == 672
    assert c["transfer_fit"] * n["transfer"] == 168 and c["transfer_eval"] * n["transfer"] == 168


# ---------------------------------------------------------------------------
# raw evidence
# ---------------------------------------------------------------------------

def test_every_raw_variant_has_the_posterior_and_signature_of_its_label():
    world = fe.q7_world()
    for signature in world.source[::37]:
        expected = tuple(fe.POSTERIOR_LEVELS[l] for l in signature)
        for variant in range(5):
            events = fe.raw_events(signature, variant)
            assert fe.signature_of_events(events) == signature
            assert fe.marginals(fe.world_posterior(events)) == expected
            assert len({e.event_id for e in events}) == len(events)


def test_event_count_is_not_a_sufficient_statistic_of_the_signature():
    signature = (3, 3, 3, 3)
    counts = {len(fe.raw_events(signature, v)) for v in range(12)}
    assert 0 in counts and len(counts) > 2
    assert len(fe.raw_events((0, 0, 0, 0), 0)) == 12
    mixed = {tuple(sorted((e.fact, e.bit, e.reliability) for e in fe.raw_events((4, 2, 1, 5), v)))
             for v in range(6)}
    assert len(mixed) > 3
    assert max(len(fe.raw_events(s, v, cancel_max=1, null_max=2)) for s in fe.q7_world().source[::11]
               for v in range(4)) <= fe.max_events(1, 2) == 28


def test_raw_events_are_deterministic_and_variant_zero_is_minimal():
    assert fe.raw_events((6, 0, 3, 2), 3) == fe.raw_events((6, 0, 3, 2), 3)
    assert fe.raw_events((6, 0, 3, 2), 1) != fe.raw_events((6, 0, 3, 2), 2)
    plain = fe.raw_events((6, 0, 3, 2), 0)
    assert len(plain) == 3 + 3 + 0 + 1 and all(e.reliability == 0.75 for e in plain)


def test_hybrid_events_are_the_donor_site_plus_the_other_base_facts():
    base, donor = fe.raw_events((1, 2, 3, 4), 1), fe.raw_events((5, 5, 5, 5), 2)
    for site in range(4):
        hybrid = fe.hybrid_events(base, donor, site)
        expected = list((1, 2, 3, 4))
        expected[site] = 5
        assert fe.signature_of_events(hybrid) == tuple(expected)
        assert {e.event_id for e in hybrid if e.fact == site} == {e.event_id for e in donor
                                                                  if e.fact == site}
    assert fe.restrict_events(base, [0, 2]) == tuple(e for e in base if e.fact in (0, 2))


def test_non_q7_records_have_no_signature():
    with pytest.raises(ValueError):
        fe.signature_of_events([OBS("a", 0, 1, 0.9)])
    with pytest.raises(ValueError):
        fe.signature_of_events([OBS(f"a{i}", 0, 1, 0.75) for i in range(4)])


# ---------------------------------------------------------------------------
# the Build A identity
# ---------------------------------------------------------------------------

def test_identity_sequences_are_fixed_width_unique_per_context_and_program_and_round_trip():
    registry = fe.function_registry()
    world = fe.q7_world()
    width = fe.identity_length()
    assert width == 4 + registry.program_width == 11
    assert len(set(fe.IDENTITY_ALPHABET)) == len(fe.IDENTITY_ALPHABET) == 15
    seen = {}
    contexts = list(world.centers_by_split["transfer_fit"]) + list(world.source[::97])
    for signature in contexts:
        for function in registry.functions:
            text = fe.identity_sequence(signature, function.program)
            assert len(text) == width and set(text) <= set(fe.IDENTITY_ALPHABET)
            assert text not in seen, (signature, function.function_id)
            seen[text] = (signature, function.function_id)
            assert fe.parse_identity_sequence(text) == (signature, function.program)
    # a context alone is not an identity: its twelve queries differ only in the program part
    signature = world.centers_by_split["transfer_fit"][0]
    texts = [fe.identity_sequence(signature, f.program) for f in registry.role("transfer")]
    assert len(set(texts)) == 12 and len({t[:4] for t in texts}) == 1
    assert len({fe.group_id(signature)}) == 1
    ids = {fe.candidate_id(signature, f) for f in registry.role("transfer")}
    assert len(ids) == 12


@pytest.mark.parametrize("text", ["", "ABCD", "ABCDWQQQQQQ" + "Q", "ABCDQQQQQQQ", "ABCDAAAAAAA",
                                  "ABCDWWQWQQQ", "ABCZWQQQQQQ", "abcdWQQQQQQ"])
def test_malformed_identities_refuse(text):
    with pytest.raises(ValueError):
        fe.parse_identity_sequence(text)


def test_world_document_is_plain_json_and_its_digest_is_stable():
    document = fe.world_document()
    assert json.loads(json.dumps(document)) == document
    assert fe.world_sha256() == fe.world_sha256()
    assert len(document["functions"]) == 93 and document["q7"]["counts"]["source"] == 1176
    assert fe.stable_seed("a", 1) == fe.stable_seed("a", 1) != fe.stable_seed("a", 2)
