"""Typed SET/SWAP programs, the provenance teacher, frozen tapes and the cohort boundary.

The provenance executor under test is validated against an INDEPENDENT oracle written here: a plain
mutable list that applies operations in place. Provenance is recovered by running that same value
oracle on a sequence of DISTINCT tokens (a token that ends up somewhere is where it came from), so no
test compares an implementation with itself. Hand fixtures are written out by hand. Nothing here
asserts a scientific result; these are mechanics and contracts.
"""
from __future__ import annotations

import copy
import dataclasses
import itertools
import types
from pathlib import Path

import numpy as np
import pytest

from smallAntibodyGen.experiments import _functional_transfer_synthetic as synth
from smallAntibodyGen.experiments import functional_baseline as fb
from smallAntibodyGen.experiments import functional_composition as fc
from smallAntibodyGen.experiments import functional_native_export as fne

L4 = fc.Layout(4, 0, "ACDE")                 # four heavy positions: the hand fixtures
L42 = fc.Layout(4, 2, "ACDE")                # six coordinates: the smallest generation layout


def H(index):
    return fc.Coord("H", index)


def L(index):
    return fc.Coord("L", index)


# ---------------------------------------------------------------------------
# the independent oracle: a mutable list, nothing shared with the implementation
# ---------------------------------------------------------------------------

def plain_ops(program):
    out = []
    for op in program.ops:
        if isinstance(op, fc.SetOp):
            out.append(("SET", (op.coord.chain, op.coord.index), op.symbol))
        else:
            out.append(("SWAP", (op.a.chain, op.a.index), (op.b.chain, op.b.index)))
    return out


def oracle_cells(heavy, light, ops):
    cells = list(heavy) + list(light)

    def slot(coord):
        chain, index = coord
        return index if chain == "H" else len(heavy) + index
    for op in ops:
        if op[0] == "SET":
            cells[slot(op[1])] = op[2]
        else:
            a, b = slot(op[1]), slot(op[2])
            cells[a], cells[b] = cells[b], cells[a]
    return cells, slot


def oracle_provenance(n_heavy, n_light, ops):
    """Where each final position's content CAME FROM: run the oracle over distinct tokens."""
    heavy = [("orig", i) for i in range(n_heavy)]
    light = [("orig", n_heavy + j) for j in range(n_light)]
    cells, slot = oracle_cells(heavy, light, ops)
    return cells, slot


def all_programs(layout):
    ops = ([fc.SetOp(coord, symbol) for coord in layout.coords() for symbol in layout.alphabet]
           + [fc.SwapOp(a, b) for a, b in itertools.combinations(layout.coords(), 2)])
    yield fc.Program(())
    for op in ops:
        yield fc.Program((op,))
    for first, second in itertools.product(ops, repeat=2):
        yield fc.Program((first, second))


def all_queries(layout):
    for coord in layout.coords():
        yield fc.ResQuery(coord)
    for a, b in itertools.combinations(layout.coords(), 2):
        yield fc.EqQuery(a, b)


def chained_by_sets(program):
    """Independent restatement of the held-out class with plain sets."""
    swaps = [op for op in program.ops if isinstance(op, fc.SwapOp)]
    if len(program.ops) != 2 or len(swaps) != 2:
        return False
    return len(set(swaps[0].coords()) & set(swaps[1].coords())) == 1


# ---------------------------------------------------------------------------
# hand fixtures
# ---------------------------------------------------------------------------

def values_after(program, heavy="ACDA"):
    return [fc.evaluate_resolution(fc.resolve(L4, program, fc.ResQuery(H(i))), heavy, "")
            for i in range(4)]


def test_two_swap_chain_hand_fixture_values_and_origins():
    chain = fc.Program((fc.SwapOp(H(0), H(1)), fc.SwapOp(H(1), H(2))))
    assert values_after(chain) == ["C", "D", "A", "A"]
    eq = fc.resolve(L4, chain, fc.EqQuery(H(2), H(3)))
    assert fc.evaluate_resolution(eq, "ACDA", "") is True
    assert [cell.coord for cell in eq.operands] == [H(0), H(3)]
    assert fc.classify(eq) == fc.STRATUM_EQ_PRIMARY
    assert fc.answer(L4, "ACDA", "", chain, fc.ResQuery(H(2))) == "A"


def test_reversed_order_hand_fixture_changes_the_answer():
    reverse = fc.Program((fc.SwapOp(H(1), H(2)), fc.SwapOp(H(0), H(1))))
    assert values_after(reverse) == ["D", "A", "C", "A"]
    assert fc.answer(L4, "ACDA", "", reverse, fc.ResQuery(H(2))) == "C"
    assert fc.answer(L4, "ACDA", "", reverse, fc.EqQuery(H(2), H(3))) is False


def test_changing_original_coordinate_three_flips_eq_but_not_res():
    chain = fc.Program((fc.SwapOp(H(0), H(1)), fc.SwapOp(H(1), H(2))))
    assert fc.answer(L4, "ACDC", "", chain, fc.ResQuery(H(2))) == "A"      # unchanged
    assert fc.answer(L4, "ACDC", "", chain, fc.EqQuery(H(2), H(3))) is False   # was true


def test_set_then_swap_makes_a_literal_derived_secondary_event():
    program = fc.Program((fc.SetOp(H(0), "E"), fc.SwapOp(H(0), H(1))))
    resolution = fc.resolve(L4, program, fc.ResQuery(H(1)))
    assert fc.evaluate_resolution(resolution, "ACDA", "") == "E"
    assert resolution.operands[0].is_literal and fc.classify(resolution) == fc.STRATUM_RES_LITERAL


def test_disjoint_swaps_resolve_eq_to_original_zero_and_two():
    program = fc.Program((fc.SwapOp(H(0), H(1)), fc.SwapOp(H(2), H(3))))
    resolution = fc.resolve(L4, program, fc.EqQuery(H(1), H(3)))
    assert [cell.coord for cell in resolution.operands] == [H(0), H(2)]


def test_secondary_strata_are_labelled_separately():
    two_literals = fc.Program((fc.SetOp(H(0), "E"), fc.SetOp(H(1), "E")))
    assert fc.classify(fc.resolve(L4, two_literals, fc.EqQuery(H(0), H(1)))) \
        == fc.STRATUM_EQ_LITERAL_LITERAL
    one_literal = fc.Program((fc.SetOp(H(0), "E"),))
    assert fc.classify(fc.resolve(L4, one_literal, fc.EqQuery(H(0), H(1)))) \
        == fc.STRATUM_EQ_ORIGINAL_LITERAL
    same = fc.Resolution(fc.EqQuery(H(0), H(1)), (fc.Cell.original(H(2)), fc.Cell.original(H(2))))
    assert fc.classify(same) == fc.STRATUM_EQ_SAME_ORIGIN
    assert fc.classify(fc.resolve(L4, fc.Program(()), fc.EqQuery(H(0), H(1)))) \
        == fc.STRATUM_EQ_PRIMARY


def test_both_swaps_are_required_for_the_hand_example_and_each_counterfactual_changes_the_pair():
    chain = fc.Program((fc.SwapOp(H(0), H(1)), fc.SwapOp(H(1), H(2))))
    query = fc.EqQuery(H(2), H(3))
    assert fc.depends_on_both(L4, chain, query)
    assert fc.dependency_tag(L4, chain, query) == fc.DEP_BOTH
    base = {cell.coord for cell in fc.resolve(L4, chain, query).operands}
    for variant in fc.counterfactual_programs(chain).values():
        assert {cell.coord for cell in fc.resolve(L4, variant, query).operands} != base
    # a query that ignores the program is not a composition test
    far = fc.Layout(6, 0, "ACDE")
    unrelated = fc.EqQuery(H(4), H(5))
    assert not fc.depends_on_both(far, chain, unrelated)
    assert fc.dependency_tag(far, chain, unrelated) == fc.DEP_IGNORES


def test_counterfactual_answer_flags_match_hand_values():
    chain = fc.Program((fc.SwapOp(H(0), H(1)), fc.SwapOp(H(1), H(2))))
    flags = fc.counterfactual_answer_changes(L4, "ACDA", "", chain, fc.EqQuery(H(2), H(3)))
    # base EQ(2,3) is true (A, A). By hand: only SWAP(1,2) gives A D C A (C vs A: false); only
    # SWAP(0,1) gives C A D A (D vs A: false); reversed order gives D A C A (C vs A: false)
    assert flags == {"delete_first": True, "delete_second": True, "reversed": True}
    same = fc.counterfactual_answer_changes(L4, "AAAA", "", chain, fc.EqQuery(H(2), H(3)))
    assert same == {"delete_first": False, "delete_second": False, "reversed": False}


# ---------------------------------------------------------------------------
# independent oracles over bounded exhaustive programs and sequences
# ---------------------------------------------------------------------------

def test_teacher_values_equal_the_mutable_list_oracle_for_every_short_program_query_and_sequence():
    layout = fc.Layout(2, 2, "AC")
    sequences = [("".join(h), "".join(l)) for h in itertools.product("AC", repeat=2)
                 for l in itertools.product("AC", repeat=2)]
    checked = 0
    for program in all_programs(layout):
        ops = plain_ops(program)
        for query in all_queries(layout):
            resolution = fc.resolve(layout, program, query)
            for heavy, light in sequences:
                cells, slot = oracle_cells(heavy, light, ops)
                if isinstance(query, fc.ResQuery):
                    expected = cells[slot((query.coord.chain, query.coord.index))]
                else:
                    expected = (cells[slot((query.a.chain, query.a.index))]
                                == cells[slot((query.b.chain, query.b.index))])
                assert fc.evaluate_resolution(resolution, heavy, light) == expected
                checked += 1
    assert checked == 211 * 10 * 16


def test_teacher_provenance_equals_the_oracle_run_over_distinct_tokens():
    layout = fc.Layout(2, 2, "AC")
    for program in all_programs(layout):
        cells, slot = oracle_provenance(2, 2, plain_ops(program))
        for query in all_queries(layout):
            resolution = fc.resolve(layout, program, query)
            for coord, cell in zip(query.coords(), resolution.operands):
                held = cells[slot((coord.chain, coord.index))]
                if isinstance(held, tuple):
                    assert not cell.is_literal
                    origin = held[1]
                    assert cell.coord == (H(origin) if origin < 2 else L(origin - 2))
                else:
                    assert cell.is_literal and cell.symbol == held


def test_depends_on_both_equals_an_oracle_pair_comparison_for_every_two_swap_program():
    layout = fc.Layout(4, 2, "ACDE")

    def pair(ops, coords):
        cells, slot = oracle_provenance(4, 2, ops)
        return frozenset(cells[slot((c.chain, c.index))][1] for c in coords)
    for program in all_programs(layout):
        if not (len(program.ops) == 2 and all(isinstance(op, fc.SwapOp) for op in program.ops)):
            continue
        first, second = program.ops
        variants = [[plain_ops(fc.Program((second,)))[0]], [plain_ops(fc.Program((first,)))[0]],
                    plain_ops(fc.Program((second, first)))]
        for a, b in itertools.combinations(layout.coords(), 2):
            base = pair(plain_ops(program), (a, b))
            expected = len(base) == 2 and all(pair(v, (a, b)) != base for v in variants)
            assert fc.depends_on_both(layout, program, fc.EqQuery(a, b)) == expected


# ---------------------------------------------------------------------------
# validation: typed, strict, never coerced
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("chain,index", [("H", True), ("H", 1.0), ("X", 0), ("h", 0), ("H", -1),
                                         (None, 0), ("H", "1")])
def test_coordinates_refuse_bools_floats_unknown_chains_and_negatives(chain, index):
    with pytest.raises(ValueError):
        fc.Coord(chain, index)


def test_numpy_integers_are_accepted_as_indices_and_bounds_are_the_layouts():
    assert fc.Coord("H", np.int64(3)).index == 3
    with pytest.raises(ValueError, match="outside"):
        L4.check_coord(H(4))
    with pytest.raises(ValueError, match="outside"):
        L4.check_coord(L(0))                 # the light chain has no positions in this layout


def test_symbols_and_arity_are_validated():
    with pytest.raises(ValueError):
        fc.SetOp(H(0), "")
    with pytest.raises(ValueError):
        fc.SetOp(H(0), "AC")
    with pytest.raises(ValueError, match="alphabet"):
        L4.check_program(fc.Program((fc.SetOp(H(0), "W"),)))
    with pytest.raises(ValueError, match="distinct"):
        fc.SwapOp(H(0), H(0))
    with pytest.raises(ValueError, match="distinct"):
        fc.EqQuery(H(1), H(1))
    with pytest.raises(ValueError, match="at most"):
        fc.Program((fc.SwapOp(H(0), H(1)),) * 3)
    for bad in ([["SET", ["H", 0]]], [["SWAP", ["H", 0]]], [["SET", ["H", True], "A"]],
                [["SET", ["H", 0], "A", "C"]], [["ROTATE", ["H", 0]]], [[]], "SET"):
        with pytest.raises(ValueError):
            fc.parse_program(bad, L4)
    for bad in (["RES"], ["EQ", ["H", 0]], ["EQ", ["H", 0], ["H", 0]], ["NOT", ["H", 0]]):
        with pytest.raises(ValueError):
            fc.parse_query(bad, L4)


def test_program_and_query_documents_round_trip_and_swap_is_symmetric():
    program = fc.Program((fc.SetOp(H(3), "E"), fc.SwapOp(H(2), H(0))))
    assert fc.parse_program(fc.program_document(program), L4) == program
    assert fc.SwapOp(H(2), H(0)) == fc.SwapOp(H(0), H(2))
    assert fc.EqQuery(H(3), H(1)) == fc.EqQuery(H(1), H(3))
    assert fc.program_key(program) == "SET(H3=E);SWAP(H0,H2)"
    assert fc.program_key(fc.Program(())) == "NOP"
    assert fc.event_id("c1", program, fc.EqQuery(H(3), H(1))) == "c1|SET(H3=E);SWAP(H0,H2)|EQ(H1,H3)"
    with pytest.raises(ValueError):
        fc.event_id("a|b", program, fc.ResQuery(H(0)))


# ---------------------------------------------------------------------------
# template classes and the held-out dependency
# ---------------------------------------------------------------------------

def test_template_ids_are_invariant_under_coordinate_renaming_and_operand_symmetry():
    rng = np.random.default_rng(3)
    coords = list(L42.coords())
    programs = [p for p in itertools.islice(all_programs(L42), 0, None, 37)]
    for program in programs:
        for query in itertools.islice(all_queries(L42), 0, None, 5):
            base = fc.template_id(program, query)
            for _ in range(4):
                renaming = dict(zip(coords, rng.permutation(len(coords)).tolist()))
                renaming = {c: coords[i] for c, i in renaming.items()}

                def rename(op):
                    if isinstance(op, fc.SetOp):
                        return fc.SetOp(renaming[op.coord], op.symbol)
                    return fc.SwapOp(renaming[op.a], renaming[op.b])
                renamed_program = fc.Program(tuple(rename(op) for op in program.ops))
                if isinstance(query, fc.ResQuery):
                    renamed_query = fc.ResQuery(renaming[query.coord])
                else:
                    renamed_query = fc.EqQuery(renaming[query.a], renaming[query.b])
                assert fc.template_id(renamed_program, renamed_query) == base


def test_set_symbols_are_abstracted_and_the_two_orders_of_a_chain_are_one_template():
    assert fc.template_id(fc.Program((fc.SetOp(H(0), "A"),))) \
        == fc.template_id(fc.Program((fc.SetOp(H(0), "E"),)))
    one = fc.Program((fc.SwapOp(H(0), H(1)), fc.SwapOp(H(1), H(2))))
    other = fc.Program((fc.SwapOp(H(1), H(2)), fc.SwapOp(H(0), H(1))))
    assert fc.template_id(one) == fc.template_id(other)


def test_every_chained_two_swap_program_is_one_program_template_distinct_from_the_others():
    layout = fc.Layout(4, 2, "ACDE")
    by_class = {"chained": set(), "disjoint": set(), "same": set()}
    for program in all_programs(layout):
        swaps = [op for op in program.ops if isinstance(op, fc.SwapOp)]
        if len(program.ops) != 2 or len(swaps) != 2:
            assert not fc.is_chained_swap_pair(program) and not chained_by_sets(program)
            continue
        overlap = len(set(swaps[0].coords()) & set(swaps[1].coords()))
        assert fc.is_chained_swap_pair(program) == (overlap == 1) == chained_by_sets(program)
        by_class[{1: "chained", 0: "disjoint", 2: "same"}[overlap]].add(fc.template_id(program))
    assert all(len(templates) == 1 for templates in by_class.values())
    assert len(set().union(*by_class.values())) == 3


def test_program_classes_name_the_held_out_and_degenerate_swap_pairs():
    chain = fc.Program((fc.SwapOp(H(0), H(1)), fc.SwapOp(H(1), H(2))))
    disjoint = fc.Program((fc.SwapOp(H(0), H(1)), fc.SwapOp(H(2), H(3))))
    same = fc.Program((fc.SwapOp(H(0), H(1)), fc.SwapOp(H(1), H(0))))
    assert [fc.program_class(p) for p in (chain, disjoint, same)] == [
        fc.CLASS_SWAP_CHAINED, fc.CLASS_SWAP_DISJOINT, fc.CLASS_SWAP_SAME]
    assert fc.program_class(fc.Program((fc.SetOp(H(0), "A"), fc.SwapOp(H(0), H(1))))) \
        == fc.CLASS_SET_SWAP


# ---------------------------------------------------------------------------
# generated event tables
# ---------------------------------------------------------------------------

def random_rows(count, seed, prefix="s", heavy=4, light=2):
    """Random sequences of one shape. A constant sequence has no complete instance (every EQ over it
    is true) and is refused by the generator, so it is redrawn here; for any other draw the rows are
    exactly what they always were."""
    rng = np.random.default_rng(seed)
    rows = []
    while len(rows) < count:
        h = "".join("ACDE"[j] for j in rng.integers(0, 4, size=heavy))
        l = "".join("ACDE"[j] for j in rng.integers(0, 4, size=light))
        if len(set(h + l)) > 1:
            rows.append((f"{prefix}{len(rows):03d}", h, l))
    return rows


@pytest.fixture(scope="module")
def train_table():
    rows = random_rows(24, 11)
    return rows, fc.generate_events(L42, rows, universe=fc.UNIVERSE_TRAIN,
                                    instances_per_sequence=3, seed=5)


@pytest.fixture(scope="module")
def held_table():
    rows = random_rows(40, 12, "h")
    return rows, fc.generate_events(L42, rows, universe=fc.UNIVERSE_HELD_OUT,
                                    instances_per_sequence=2, seed=6)


def test_training_events_exclude_every_chained_two_swap_program_and_cover_the_primitives(train_table):
    rows, table = train_table
    assert len(table) > 0 and table.config["nominal_events"] == 24 * 3 * 4
    assert not any(chained_by_sets(event.program) for event in table.events)
    assert not any(event.held_out for event in table.events)
    assert fc.coverage_report(table)["complete"]
    chained_templates = {fc.template_id(p) for p in all_programs(L42) if chained_by_sets(p)}
    assert not any(fc.template_id(event.program) in chained_templates for event in table.events)
    assert fc.verify_events_against(table, rows)
    assert len(set(table.ids())) == len(table)


def test_training_answers_equal_the_mutable_list_oracle(train_table):
    rows, table = train_table
    sequences = {cid: (h, l) for cid, h, l in rows}
    for event in table.events:
        heavy, light = sequences[event.candidate_id]
        cells, slot = oracle_cells(heavy, light, plain_ops(event.program))
        if isinstance(event.query, fc.ResQuery):
            expected = cells[slot((event.query.coord.chain, event.query.coord.index))]
        else:
            a, b = event.query.a, event.query.b
            expected = cells[slot((a.chain, a.index))] == cells[slot((b.chain, b.index))]
        assert event.answer == expected


def test_primary_eq_events_compare_two_distinct_original_origins(train_table):
    _, table = train_table
    primary = [e for e in table.events if e.stratum == fc.STRATUM_EQ_PRIMARY]
    assert primary
    for event in primary:
        coords = [cell.coord for cell in event.resolution.operands]
        assert None not in coords and coords[0] != coords[1]
    secondary = {e.stratum for e in table.events} - {fc.STRATUM_EQ_PRIMARY, fc.STRATUM_RES_ORIGINAL}
    assert secondary <= {fc.STRATUM_RES_LITERAL, fc.STRATUM_EQ_ORIGINAL_LITERAL,
                         fc.STRATUM_EQ_LITERAL_LITERAL}


def test_eq_true_false_balance_holds_per_sequence_and_program_or_is_recorded_unsupported(train_table):
    _, table = train_table
    units = {}
    for event in table.events:
        if event.output_type == fc.OUTPUT_EQUALITY and event.stratum == fc.STRATUM_EQ_PRIMARY:
            units.setdefault((event.candidate_id, fc.program_key(event.program)), [0, 0])[
                0 if event.answer else 1] += 1
    noted = {(note["candidate"], note["program"]) for note in table.unsupported if "program" in note}
    for unit, (true, false) in units.items():
        assert true == false or unit in noted, f"{unit} is unbalanced and not recorded"
    report = fc.balance_report(table)
    primary = [e for e in table.events
               if e.output_type == fc.OUTPUT_EQUALITY and e.stratum == fc.STRATUM_EQ_PRIMARY]
    assert sum(t + f for t, f in ((v["true"], v["false"]) for v in report["templates"].values())) \
        == len(primary)
    assert report["units_with_equal_true_and_false"] + report["unbalanced_unit_count"] \
        == report["primary_eq_units"]


def test_held_out_events_need_both_swaps_and_the_counterfactual_oracle_agrees(held_table):
    rows, table = held_table
    assert fc.coverage_report(table)["complete"] and fc.verify_events_against(table, rows)
    sequences = {cid: (h, l) for cid, h, l in rows}
    both = [e for e in table.events if e.dependency == fc.DEP_BOTH]
    assert both and all(chained_by_sets(e.program) and e.held_out for e in table.events)
    saw_changed = saw_unchanged = False
    for event in both:
        if event.output_type != fc.OUTPUT_EQUALITY:
            continue
        first, second = plain_ops(event.program)
        variants = {"delete_first": [second], "delete_second": [first], "reversed": [second, first]}
        base_cells, slot = oracle_provenance(4, 2, [first, second])
        coords = event.query.coords()
        base_pair = frozenset(base_cells[slot((c.chain, c.index))][1] for c in coords)
        assert len(base_pair) == 2                                   # two DISTINCT original origins
        heavy, light = sequences[event.candidate_id]
        base_value = oracle_cells(heavy, light, [first, second])
        for name, ops in variants.items():
            cells, _ = oracle_provenance(4, 2, ops)
            assert frozenset(cells[slot((c.chain, c.index))][1] for c in coords) != base_pair
        flags = {}
        for name, ops in variants.items():
            values, _ = oracle_cells(heavy, light, ops)
            values_base, _ = base_value
            a, b = (slot((c.chain, c.index)) for c in coords)
            flags[name] = (values[a] == values[b]) != (values_base[a] == values_base[b])
        assert event.flags == flags
        saw_changed |= any(flags.values())
        saw_unchanged |= not any(flags.values())
    assert saw_changed and saw_unchanged, "both answer-changing cases and unchanged-answer controls"
    assert {e.dependency for e in table.events} >= {fc.DEP_BOTH, fc.DEP_IGNORES}


def test_events_are_deterministic_and_the_seed_changes_them(train_table):
    rows, table = train_table
    again = fc.generate_events(L42, rows, universe=fc.UNIVERSE_TRAIN, instances_per_sequence=3,
                               seed=5)
    other = fc.generate_events(L42, rows, universe=fc.UNIVERSE_TRAIN, instances_per_sequence=3,
                               seed=6)
    assert again.sha256 == table.sha256 and other.sha256 != table.sha256
    assert table.verify()


def test_generation_refuses_unfit_layouts_sequences_and_universes():
    with pytest.raises(ValueError, match="at least"):
        fc.generate_events(L4, random_rows(2, 1), universe=fc.UNIVERSE_TRAIN,
                           instances_per_sequence=1, seed=0)
    with pytest.raises(ValueError, match="layout"):
        fc.generate_events(L42, [("x", "ACD", "AC")], universe=fc.UNIVERSE_TRAIN,
                           instances_per_sequence=1, seed=0)
    with pytest.raises(ValueError, match="universe"):
        fc.generate_events(L42, random_rows(1, 1), universe="held", instances_per_sequence=1, seed=0)


def test_event_documents_round_trip_and_a_doctored_answer_is_refused(train_table):
    rows, table = train_table
    document = copy.deepcopy(table.document())
    rebuilt = fc.parse_event_document(document, L42)
    assert rebuilt.sha256 == table.sha256
    document["events"][0]["answer"] = (not document["events"][0]["answer"]
                                       if isinstance(document["events"][0]["answer"], bool)
                                       else "W")
    with pytest.raises(ValueError):
        fc.parse_event_document(document, L42)             # digest no longer matches
    doctored = copy.deepcopy(table)
    object.__setattr__(doctored.events[0], "answer",
                       (not doctored.events[0].answer) if isinstance(doctored.events[0].answer, bool)
                       else "W")
    with pytest.raises(ValueError, match="not the teacher's"):
        fc.verify_events_against(doctored, rows)


# ---------------------------------------------------------------------------
# the shuffled-target control
# ---------------------------------------------------------------------------

def test_shuffled_targets_preserve_each_strata_multiset_and_change_real_values(train_table):
    _, table = train_table
    codes = table.codes()
    shuffled, report = fc.shuffle_targets(table, seed=0)
    assert not shuffled.flags.writeable
    strata = {}
    for position, event in enumerate(table.events):
        strata.setdefault((event.output_type, event.template_id, event.stratum), []).append(position)
    total_changed = 0
    for entry in report["strata"]:
        members = np.array(strata[(entry["output"], entry["template"], entry["stratum"])])
        before, after = codes[members], shuffled[members]
        assert sorted(before.tolist()) == sorted(after.tolist())          # a permutation
        changed = int(np.sum(before != after))
        assert changed == entry["changed_values"]                         # VALUES, not row ids
        total_changed += changed
        if entry["status"].startswith("unsupported"):
            assert changed == 0
            repeated = max(entry["value_counts"].values())
            assert {"unsupported_constant": entry["distinct_values"] < 2,
                    "unsupported_small": entry["events"] < fc.MIN_STRATUM,
                    "unsupported_singleton_classes": repeated < 2,
                    "unsupported_degenerate": True}[entry["status"]]
        else:
            assert 1 <= changed <= entry["events"] - 1                    # never a fixed inversion
            # and never a class renaming: some repeated original class is split across new targets
            assert any(len(set(after[before == value].tolist())) > 1
                       for value in set(before.tolist()) if int((before == value).sum()) >= 2)
            assert entry["split_original_classes"] >= 1 and entry["deterministic_class_mapping"] is False
    assert total_changed == report["changed_values"] == int(np.sum(codes != shuffled)) > 0
    assert 0.0 < report["changed_fraction"] < 1.0


def test_a_constant_stratum_is_reported_unsupported_and_left_unshuffled():
    events = [types.SimpleNamespace(output_type="equality", template_id="t", stratum="eq_primary")
              for _ in range(8)]
    table = types.SimpleNamespace(events=events, codes=lambda: np.ones(8, dtype=np.int64))
    shuffled, report = fc.shuffle_targets(table, seed=1)
    assert shuffled.tolist() == [1] * 8 and report["changed_values"] == 0
    assert report["strata"][0]["status"] == "unsupported_constant"
    assert report["unsupported_strata"]


def test_a_balanced_boolean_stratum_is_not_a_perfect_inversion():
    events = [types.SimpleNamespace(output_type="equality", template_id="t", stratum="eq_primary")
              for _ in range(8)]
    codes = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.int64)
    table = types.SimpleNamespace(events=events, codes=lambda: codes)
    for seed in range(10):
        shuffled, report = fc.shuffle_targets(table, seed=seed)
        changed = int(np.sum(shuffled != codes))
        assert 1 <= changed <= 7 and sorted(shuffled.tolist()) == sorted(codes.tolist())
        assert report["strata"][0]["changed_values"] == changed


def test_shuffle_is_deterministic_per_seed_and_uses_no_outside_state(train_table):
    _, table = train_table
    first, _ = fc.shuffle_targets(table, seed=3)
    second, _ = fc.shuffle_targets(table, seed=3)
    third, _ = fc.shuffle_targets(table, seed=4)
    assert np.array_equal(first, second) and not np.array_equal(first, third)


# ---------------------------------------------------------------------------
# instruction rows: typed, answer-free; R's direct form is sequence-free
# ---------------------------------------------------------------------------

def test_composed_instruction_rows_for_a_hand_program():
    space = L42.space()
    program = fc.Program((fc.SetOp(H(3), "E"), fc.SwapOp(H(0), H(1))))
    rows = fc.compile_instruction(space, L42, program, fc.EqQuery(H(2), L(1)))
    assert rows == ((fc.KIND_SET, 3, -1, 3, -1), (fc.KIND_SWAP, 0, 1, -1, -1),
                    (fc.KIND_EQ, 2, 5, -1, -1))
    assert fc.compile_instruction(space, L42, fc.Program(()), fc.ResQuery(L(0))) \
        == ((fc.KIND_RES, 4, -1, -1, -1),)
    assert fc.FUNCTION_INSTRUCTION == ((fc.KIND_FUNCTION, -1, -1, -1, -1),)


def test_direct_instructions_resolve_operands_and_carry_no_sequence(train_table):
    space = L42.space()
    chain = fc.Program((fc.SwapOp(H(0), H(1)), fc.SwapOp(H(1), H(2))))
    resolved = fc.resolve(L42, chain, fc.EqQuery(H(2), H(3)))
    assert fc.compile_direct(space, L42, resolved) == ((fc.KIND_REL_ORIGINAL_ORIGINAL, 0, 3, -1, -1),)
    literal = fc.resolve(L42, fc.Program((fc.SetOp(H(0), "E"), fc.SwapOp(H(0), H(1)))),
                         fc.ResQuery(H(1)))
    assert fc.compile_direct(space, L42, literal) == ((fc.KIND_READ_LITERAL, -1, -1, 3, -1),)
    original = fc.resolve(L42, chain, fc.ResQuery(H(2)))
    assert fc.compile_direct(space, L42, original) == ((fc.KIND_READ_ORIGINAL, 0, -1, -1, -1),)
    mixed = fc.resolve(L42, fc.Program((fc.SetOp(H(0), "C"),)), fc.EqQuery(H(0), H(1)))
    assert fc.compile_direct(space, L42, mixed) == ((fc.KIND_REL_ORIGINAL_LITERAL, 1, -1, 1, -1),)
    both = fc.resolve(L42, fc.Program((fc.SetOp(H(0), "A"), fc.SetOp(H(1), "D"))),
                      fc.EqQuery(H(0), H(1)))
    assert fc.compile_direct(space, L42, both) == ((fc.KIND_REL_LITERAL_LITERAL, -1, -1, 0, 2),)
    _, table = train_table
    for event in table.events:
        assert len(fc.compile_direct(space, L42, event.resolution)) == 1


def test_resolution_is_independent_of_the_candidate_sequence():
    chain = fc.Program((fc.SwapOp(H(0), H(1)), fc.SwapOp(H(1), H(2))))
    query = fc.EqQuery(H(2), H(3))
    assert fc.resolve(L4, chain, query) == fc.resolve(L4, chain, query)
    # the resolution API takes no sequence at all; only evaluation reads one
    first = fc.evaluate_resolution(fc.resolve(L4, chain, query), "ACDA", "")
    second = fc.evaluate_resolution(fc.resolve(L4, chain, query), "ECDE", "")
    assert first is True and second is True


# ---------------------------------------------------------------------------
# frozen tapes
# ---------------------------------------------------------------------------

def make_tape(table, *, seed=0, steps=6, extra=0, batch=4, aux_batch=5):
    ids = tuple(f"f{i}" for i in range(8))
    return fc.build_seed_tape(
        seed=seed, steps=steps, functional_steps=steps + extra, batch=batch, aux_batch=aux_batch,
        functional_ids=ids, functional_labels=[float(i) / 3 for i in range(8)], table=table,
        mask_binding={"padded_tokens": 14, "padded_instructions": 3,
                      "token_lengths": {cid: 14 for cid in ids}})


def test_tapes_are_canonical_read_only_hashed_and_replay_identically(train_table):
    _, table = train_table
    tape = make_tape(table)
    again = make_tape(table)
    assert tape.sha256 == again.sha256 and tape.verify()
    for array in (tape.functional_labels, tape.functional_batches, tape.aux_batches, tape.q_targets):
        assert not array.flags.writeable
    assert make_tape(table, seed=1).sha256 != tape.sha256
    restored = fc.SeedTape.from_document(copy.deepcopy(tape.document()))
    assert restored.sha256 == tape.sha256
    tape.functional_batches.setflags(write=True)
    tape.functional_batches[0, 0] += 1
    with pytest.raises(ValueError, match="no longer matches"):
        tape.verify()


@pytest.mark.parametrize("edit", [
    lambda d: d["functional_batches"][0].__setitem__(0, (d["functional_batches"][0][0] + 1) % 8),
    lambda d: d["aux_batches"][1].__setitem__(0, d["aux_batches"][1][0] + 1),
    lambda d: d["functional_labels"].__setitem__(0, d["functional_labels"][0] + 0.5),
    lambda d: d["q_targets"].__setitem__(0, 1 - d["q_targets"][0]),
    lambda d: d["mask_binding"]["token_lengths"].__setitem__("f0", 13),
    lambda d: d.__setitem__("seed", d["seed"] + 1)])
def test_a_tampered_tape_document_is_refused(train_table, edit):
    _, table = train_table
    document = copy.deepcopy(make_tape(table).document())
    edit(document)
    with pytest.raises(ValueError, match="recorded digest"):
        fc.SeedTape.from_document(document)


def test_the_functional_stream_extends_and_the_auxiliary_stream_is_one_pass_when_it_divides(train_table):
    _, table = train_table
    short, long = make_tape(table, steps=3, batch=4), make_tape(table, steps=3, extra=4, batch=4)
    assert np.array_equal(short.functional_batches, long.functional_batches[:3])
    n = len(table)
    one_pass = fc.build_seed_tape(
        seed=2, steps=n // 4, functional_steps=n // 4, batch=4, aux_batch=4,
        functional_ids=("a", "b", "c", "d"), functional_labels=[0.0, 1.0, 2.0, 3.0], table=table,
        mask_binding={"padded_tokens": 1, "padded_instructions": 3, "token_lengths": {}})
    flat = one_pass.aux_batches.reshape(-1)
    assert len(set(flat.tolist())) == len(flat) and int(flat.max()) < n    # one epoch: no repeat


def test_tape_construction_uses_no_target_or_held_out_information(train_table):
    # the builder's signature has no evaluation argument at all, and the shuffled targets come from
    # the source-only table alone
    import inspect
    names = set(inspect.signature(fc.build_seed_tape).parameters)
    assert names == {"seed", "steps", "functional_steps", "batch", "aux_batch", "functional_ids",
                     "functional_labels", "table", "mask_binding"}


# ---------------------------------------------------------------------------
# cohorts: the data boundary on a real (tiny, random) native token cache
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def world_inputs(tmp_path_factory):
    world = synth.build_world(synth.SELFTEST_SPEC)
    paths = synth.write_inputs(world, tmp_path_factory.mktemp("cohort_world") / "inputs",
                               model_seed=0)
    return world, paths


@pytest.fixture(scope="module")
def cache(world_inputs):
    return fne.load_token_cache(world_inputs[1]["token_header"])


def manifest_document(world):
    return copy.deepcopy(world.cohort_document())


def export_variant(tmp_path, world, mutate):
    document = world.export_plan_document()
    mutate(document["rows"])
    plan = fne.parse_export_plan(document)
    fne.export_dplm_tokens(fne.tiny_model(0), plan, tmp_path / "t.json", tmp_path / "t.npz",
                           batch_size=64)
    return fne.load_token_cache(tmp_path / "t.json")


def test_valid_cohorts_pass_and_report_layout_and_target_identity_length(world_inputs, cache):
    world, _ = world_inputs
    cohorts = fc.validate_cohorts(cache, fc.parse_cohort_manifest(manifest_document(world)))
    assert cohorts.layout == fc.Layout(4, 4, "ACDE") and cohorts.target_identity_length == 9
    assert set(cohorts.roles) == set(fc.ROLES)
    summary = cohorts.summary(cache)
    assert summary["cache_sha256"] == cache.cache_sha256
    assert all(summary["roles"][role]["candidates"] == len(world.roles[role]) for role in fc.ROLES)


def test_source_aux_defaults_to_source_train_and_hashes_the_same_as_explicit(world_inputs):
    world, _ = world_inputs
    document = manifest_document(world)
    document["cohorts"]["source_aux"] = list(document["cohorts"]["source_train"])
    explicit = fc.parse_cohort_manifest(document)
    del document["cohorts"]["source_aux"]
    defaulted = fc.parse_cohort_manifest(document)
    assert defaulted.aux_defaulted and not explicit.aux_defaulted
    assert defaulted.roles["source_aux"] == defaulted.roles["source_train"]
    assert defaulted.sha256() == explicit.sha256()


@pytest.mark.parametrize("edit,message", [
    (lambda d: d.__setitem__("surprise", 1), "unknown"),
    (lambda d: d["cohorts"].__setitem__("source_dev", []), "non-empty"),
    (lambda d: d["cohorts"].__setitem__("target_eval", d["cohorts"]["target_eval"][:2] * 2),
     "repeats"),
    (lambda d: d["cohorts"].__setitem__("extra_role", ["x"]), "unknown"),
    (lambda d: d["cohorts"].pop("target_train"), "missing"),
    (lambda d: d["source"].pop("units"), "missing"),
    (lambda d: d["target"].pop("useful_threshold"), "missing"),
    (lambda d: d.__setitem__("alphabet", "ACDEc"), "alphabet"),
    (lambda d: d.__setitem__("schema", "other/1"), "not a")])
def test_the_cohort_manifest_parser_is_strict(world_inputs, edit, message):
    document = manifest_document(world_inputs[0])
    edit(document)
    with pytest.raises(ValueError, match=message):
        fc.parse_cohort_manifest(document)


def validate(world, cache, edit):
    document = manifest_document(world)
    edit(document["cohorts"])
    return fc.validate_cohorts(cache, fc.parse_cohort_manifest(document))


def test_cache_rows_outside_the_declared_union_are_refused(world_inputs, cache):
    world, _ = world_inputs
    with pytest.raises(ValueError, match="outside the declared cohort union"):
        validate(world, cache, lambda c: c.__setitem__("target_eval", c["target_eval"][1:]))


def test_declared_ids_the_cache_lacks_are_refused(world_inputs, cache):
    world, _ = world_inputs
    with pytest.raises(ValueError, match="does not hold"):
        validate(world, cache, lambda c: c["source_dev"].append("ghost"))


def test_partial_overlap_of_source_train_and_source_aux_is_refused(world_inputs, cache):
    world, _ = world_inputs

    def edit(c):
        c["source_aux"] = [cid for cid in c["source_aux"] if cid != c["source_train"][0]]
    with pytest.raises(ValueError, match="only partially"):
        validate(world, cache, edit)


def test_source_aux_may_not_include_source_dev_or_a_target_candidate(world_inputs, cache):
    world, _ = world_inputs
    with pytest.raises(ValueError, match="share"):
        validate(world, cache, lambda c: c["source_aux"].append(c["source_dev"][0]))
    with pytest.raises(ValueError, match="share"):
        validate(world, cache, lambda c: c["source_aux"].append(c["target_train"][0]))


def test_swapped_target_roles_fail_the_expected_cache_split_check(world_inputs, cache):
    world, _ = world_inputs

    def edit(c):
        c["target_train"], c["target_eval"] = c["target_eval"], c["target_train"]
    with pytest.raises(ValueError, match="cache split"):
        validate(world, cache, edit)


def test_a_group_shared_between_cohorts_is_refused(tmp_path, world_inputs):
    world, _ = world_inputs
    aux_group = next(row["group_id"] for row in world.rows if row["cohort"] == "source")
    target = next(row["candidate_id"] for row in world.rows if row["cohort"] == "target_train")

    def mutate(rows):
        for row in rows:
            if row["candidate_id"] == target:
                row["group_id"] = aux_group
    shared = export_variant(tmp_path, world, mutate)
    with pytest.raises(ValueError, match="share group"):
        fc.validate_cohorts(shared, fc.parse_cohort_manifest(manifest_document(world)))


def test_a_target_roster_with_mixed_full_identity_lengths_is_refused_not_truncated(tmp_path,
                                                                                   world_inputs):
    world, _ = world_inputs
    target = next(row["candidate_id"] for row in world.rows if row["cohort"] == "target_eval")

    def mutate(rows):
        for row in rows:
            if row["candidate_id"] == target:
                row["light"] = row["light"] + "A"
    mixed = export_variant(tmp_path, world, mutate)
    with pytest.raises(ValueError, match="full-identity lengths"):
        fc.validate_cohorts(mixed, fc.parse_cohort_manifest(manifest_document(world)))


def test_an_auxiliary_cohort_with_two_layouts_keeps_each_rows_own_layout(tmp_path, world_inputs):
    """The accepted data boundary permits variable source/auxiliary lengths: a real tiny-native cache
    with H4/L4 and H5/L4 source rows is accepted and every row keeps its own layout."""
    world, _ = world_inputs
    aux = next(row["candidate_id"] for row in world.rows if row["cohort"] == "source")

    def mutate(rows):
        for row in rows:
            if row["candidate_id"] == aux:
                row["heavy"] = row["heavy"] + "A"
    odd = export_variant(tmp_path, world, mutate)
    cohorts = fc.validate_cohorts(odd, fc.parse_cohort_manifest(manifest_document(world)))
    assert cohorts.layout is None and isinstance(cohorts.layouts, fc.LayoutMap)
    assert cohorts.layouts.classes() == ((4, 4), (5, 4))
    assert cohorts.layouts.layout_for(aux) == fc.Layout(5, 4, "ACDE")
    other = next(cid for cid in cohorts.roles["source_aux"] if cid != aux)
    assert cohorts.layouts.layout_for(other) == fc.Layout(4, 4, "ACDE")
    summary = cohorts.summary(odd)["auxiliary_layouts"]
    assert summary["variable"] is True and summary["shapes"] == {
        "H4L4": len(cohorts.roles["source_aux"]) - 1, "H5L4": 1}
    assert summary["sha256"] == fc.layouts_sha256(cohorts.layouts) \
        != fc.layouts_sha256(fc.Layout(4, 4, "ACDE"))


def test_an_auxiliary_row_below_the_minimum_coordinates_is_refused_by_name(tmp_path, world_inputs):
    world, _ = world_inputs
    aux = next(row["candidate_id"] for row in world.rows if row["cohort"] == "source")

    def mutate(rows):
        for row in rows:
            if row["candidate_id"] == aux:
                row["heavy"], row["light"] = "AC", "DE"
    tiny = export_variant(tmp_path, world, mutate)
    with pytest.raises(ValueError, match="fewer than 6 coordinates") as raised:
        fc.validate_cohorts(tiny, fc.parse_cohort_manifest(manifest_document(world)))
    assert aux in str(raised.value) and "Remove them" in str(raised.value)


def test_h4l4_and_h5l3_target_rows_coexist_when_the_full_identity_length_agrees(tmp_path,
                                                                              world_inputs):
    """The matrix joins ONE full H|L identity length: 4+1+4 and 5+1+3 are both 9. Nothing is padded or
    truncated, and a different total is still refused (see the mixed-identity-length test)."""
    world, _ = world_inputs
    target = next(row["candidate_id"] for row in world.rows if row["cohort"] == "target_eval")

    def mutate(rows):
        for row in rows:
            if row["candidate_id"] == target:
                row["heavy"], row["light"] = row["heavy"] + "A", row["light"][:3]
    mixed = export_variant(tmp_path, world, mutate)
    cohorts = fc.validate_cohorts(mixed, fc.parse_cohort_manifest(manifest_document(world)))
    assert cohorts.target_identity_length == 9
    position = cohorts.position[target]
    assert (len(mixed.heavy[position]), len(mixed.light[position])) == (5, 3)


def test_outcome_files_must_name_exactly_the_cohort(tmp_path, world_inputs):
    world, paths = world_inputs
    ids = world.roles["source_train"]
    assert set(fc.read_role_outcomes(paths["source_train_outcomes"], ids,
                                     label="source training outcomes")) == set(ids)
    with pytest.raises(ValueError, match="lacks"):
        fc.read_role_outcomes(paths["source_train_outcomes"], ids + ("ghost",), label="x")
    with pytest.raises(ValueError, match="outside"):
        fc.read_role_outcomes(paths["source_train_outcomes"], ids[1:], label="x")
    bad = tmp_path / "nan.csv"
    bad.write_text("candidate_id,utility\n" + f"{ids[0]},nan\n", encoding="utf-8")
    with pytest.raises(ValueError):
        fc.read_role_outcomes(bad, ids[:1], label="x")


def test_the_synthetic_world_matches_its_declaration(world_inputs):
    world, _ = world_inputs
    spec = synth.SELFTEST_SPEC
    assert {role: len(ids) for role, ids in world.roles.items()} == spec.totals()
    pilot = synth.build_world(synth.PILOT_SPEC)
    assert {role: len(ids) for role, ids in pilot.roles.items()} == {
        "source_train": 128, "source_aux": 512, "source_dev": 64, "target_train": 128,
        "target_eval": 128}
    assert set(pilot.roles["source_train"]) <= set(pilot.roles["source_aux"])
    # boundary positions are irrelevant to both declared functions
    for row in world.rows[:20]:
        heavy, light = row["heavy"], row["light"]
        for boundary in "ACDE":
            flipped_h, flipped_l = boundary + heavy[1:], light[:3] + boundary
            assert synth.source_function(heavy, light) == synth.source_function(flipped_h, flipped_l)
            assert synth.target_function(heavy, light) == synth.target_function(flipped_h, flipped_l)
    assert synth.source_function("AEEC", "DDAA") == pytest.approx(1.0 + 0.5 + 0.2 * (1.0 / 3.0))
    assert synth.target_function("AEDC", "DDDA") == pytest.approx(1.0 + 0.0 + 0.2 * (2.0 / 3.0))


def test_baseline_helpers_used_by_the_boundary_exist_with_their_contracts():
    assert fb.SPLIT_TRAIN == "train" and fb.SPLIT_EVAL == "eval"
    assert fc.EXPECTED_SPLIT["source_dev"] == fb.SPLIT_EVAL


# ---------------------------------------------------------------------------
# canonical event identity: one question, one id
# ---------------------------------------------------------------------------

def hand_event(layout, cid, heavy, light, program, query):
    resolution = fc.resolve(layout, program, query)
    value = fc.answer(layout, heavy, light, program, query)
    return fc._with_flags(layout, heavy, light,
                          fc._make_event(layout, cid, program, query, resolution, value))


def test_an_event_id_must_be_the_canonical_id_of_its_candidate_program_and_query(train_table):
    _, table = train_table
    original = table.events[0]
    with pytest.raises(ValueError, match="canonical id"):
        dataclasses.replace(original, event_id="alias")             # refused at construction
    mutated = copy.deepcopy(original)
    object.__setattr__(mutated, "event_id", "alias")
    with pytest.raises(ValueError, match="canonical id"):
        fc.EventTable(L42, (original, mutated), fc.UNIVERSE_TRAIN, {}, ())   # and when counted
    document = copy.deepcopy(table.document())
    document["events"][0]["id"] = "alias"
    body = {key: value for key, value in document.items() if key != "event_table_sha256"}
    document["event_table_sha256"] = fb.digest(body)                # a self-consistently re-hashed doc
    with pytest.raises(ValueError, match="canonical id"):
        fc.parse_event_document(document, L42)
    assert len(set(fc.event_id(e.candidate_id, e.program, e.query) for e in table.events)) \
        == len(table)                                               # unique answers = unique questions


# ---------------------------------------------------------------------------
# matched true/false pairs by FULL program+query template; balance reporting
# ---------------------------------------------------------------------------

def test_true_and_false_eq_events_of_an_instance_share_one_full_program_and_query_template():
    layout = fc.Layout(3, 3, "ACDE")
    program = fc.Program((fc.SwapOp(H(0), H(1)),))
    template = "SWAP(v0,v1)|EQ(v0,v2)"
    hand = [(fc.EqQuery(H(0), L(0)), True), (fc.EqQuery(H(0), H(2)), False),
            (fc.EqQuery(H(2), L(0)), False)]
    for query, truth in hand:
        assert fc.answer(layout, "ACA", "CDE", program, query) is truth
    assert fc.template_id(program, hand[0][0]) == fc.template_id(program, hand[1][0]) == template
    assert fc.template_id(program, hand[2][0]) != template          # positions the program never touches
    for seed in range(32):
        notes = []
        events = fc._train_instance(layout, "cross", "ACA", "CDE", program, fc.STRATUM_EQ_PRIMARY,
                                    np.random.default_rng(seed), notes)
        eq = [e for e in events if e.output_type == fc.OUTPUT_EQUALITY]
        assert len(events) == 4 and len(eq) == 2 and {e.answer for e in eq} == {True, False}
        assert eq[0].template_id == eq[1].template_id == template, seed
        assert not notes


def test_held_out_balance_counts_the_both_swap_events_and_reports_the_ignoring_control_apart():
    layout = fc.Layout(4, 4, "ACDE")
    program = fc.Program((fc.SwapOp(H(0), H(1)), fc.SwapOp(H(1), H(2))))
    queries = [fc.EqQuery(H(2), H(3)), fc.EqQuery(H(2), L(0)), fc.EqQuery(H(3), L(0))]
    events = tuple(hand_event(layout, "held", "ACDA", "CAAA", program, q) for q in queries)
    assert [e.answer for e in events] == [True, False, False]
    assert [e.dependency for e in events] == [fc.DEP_BOTH, fc.DEP_BOTH, fc.DEP_IGNORES]
    report = fc.balance_report(fc.EventTable(layout, events, fc.UNIVERSE_HELD_OUT, {}, ()))
    assert report["primary_eq_units"] == 1 and report["units_with_equal_true_and_false"] == 1
    assert report["unbalanced_unit_count"] == 0 and report["templates_with_only_one_answer"] == []
    (template, counts), = report["templates"].items()
    assert (counts["true"], counts["false"]) == (1, 1)
    assert report["program_ignoring_and_other_controls"] == {
        fc.DEP_IGNORES: {"true": 0, "false": 1, "events": 1}}


def test_balance_is_reported_per_candidate_and_full_template_with_support_counts(train_table):
    _, table = train_table
    report = fc.balance_report(table)
    primary = [e for e in table.events
               if e.output_type == fc.OUTPUT_EQUALITY and e.stratum == fc.STRATUM_EQ_PRIMARY]
    assert report["unit"].startswith("(candidate, full canonical program+query template)")
    assert sum(v["true"] + v["false"] for v in report["templates"].values()) == len(primary)
    assert all(v["candidates"] >= 1 for v in report["templates"].values())
    assert report["unbalanced_unit_count"] == 0                    # every accepted instance is matched
    assert "infeasible_matches" in report and "eq_by_stratum" in report


# ---------------------------------------------------------------------------
# exact budgets: replace an incomplete instance, never shorten the roster
# ---------------------------------------------------------------------------

def test_the_two_pilot_instances_that_were_incomplete_have_no_false_primary_eq():
    layout = fc.Layout(4, 4, "ACDE")
    for cid, heavy, light, program in (
            ("sa-g-h2-l2-070", "DADD", "DCDD", fc.Program((fc.SetOp(L(1), "A"), fc.SetOp(H(1), "C")))),
            ("sa-g-h3-l3-124", "EEEE", "EDEE", fc.Program((fc.SetOp(L(1), "A"), fc.SetOp(L(0), "C"))))):
        notes = []
        events = fc._train_instance(layout, cid, heavy, light, program, fc.STRATUM_EQ_PRIMARY,
                                    np.random.default_rng(0), notes)
        assert events == [] and "no primary EQ pair" in notes[-1]["reason"], cid


def test_a_rejected_draw_is_replaced_in_the_same_instance_and_logged(monkeypatch):
    layout = fc.Layout(4, 4, "ACDE")
    bad = fc.Program((fc.SetOp(L(1), "A"), fc.SetOp(H(1), "C")))
    real, calls = fc._draw_train_program, []

    def first_draw_is_bad(klass, layout_, decks, rng):
        calls.append(klass)
        return bad if len(calls) == 1 else real(klass, layout_, decks, rng)
    monkeypatch.setattr(fc, "_draw_train_program", first_draw_is_bad)
    table = fc.generate_events(layout, [("sa-1", "DADD", "DCDD")], universe=fc.UNIVERSE_TRAIN,
                               instances_per_sequence=2, seed=0)
    assert len(table) == 8 and len(set(table.ids())) == 8          # four unique events per instance
    replacement = table.config["replacement"]
    assert replacement["replaced_instances"] == 1 and replacement["rejected_attempts"] == 1
    assert list(replacement["rejection_reasons"].values()) == [1]
    assert all(event.program != bad for event in table.events)


def test_a_sequence_that_cannot_supply_a_complete_instance_is_refused_never_shortened():
    layout = fc.Layout(4, 4, "ACDE")
    rows = random_rows(6, 3, heavy=4, light=4) + [("zz-constant", "AAAA", "AAAA")]
    with pytest.raises(ValueError, match="cannot supply a complete train instance") as raised:
        fc.generate_events(layout, rows, universe=fc.UNIVERSE_TRAIN, instances_per_sequence=2, seed=0)
    assert "zz-constant" in str(raised.value) and "never shortened" in str(raised.value)
    table = fc.generate_events(layout, rows, universe=fc.UNIVERSE_TRAIN, instances_per_sequence=2,
                               seed=0, strict=False)
    assert "zz-constant" not in set(table.candidate_ids()) and len(table) == 6 * 2 * 4
    skipped = [note for note in table.unsupported if note.get("candidate") == "zz-constant"]
    assert len(skipped) == 2 and all("no complete instance" in note["reason"] for note in skipped)


def test_the_synthetic_world_never_selects_a_constant_auxiliary_sequence():
    for seed in range(3):
        world = synth.build_world(dataclasses.replace(synth.SELFTEST_SPEC, seed=seed))
        sequences = {row["candidate_id"]: row["heavy"] + row["light"] for row in world.rows}
        assert all(len(set(sequences[cid])) > 1 for cid in world.roles["source_aux"])
        assert isinstance(world.excluded, tuple)
    assert synth._is_constant(0, 0, 0) and not synth._is_constant(1, 0, 0)
    assert synth._replace_constant([0, 5, 9], 0, 0, np.random.default_rng(0))[1] == [0]


# ---------------------------------------------------------------------------
# variable layouts: per-candidate layouts, one shared coordinate space
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mixed_table():
    rows = random_rows(28, 31, "a", 4, 2) + random_rows(24, 32, "b", 5, 3)
    layouts = fc.layouts_for_rows(rows, "ACDE")
    return rows, layouts, fc.generate_events(layouts, rows, universe=fc.UNIVERSE_TRAIN,
                                             instances_per_sequence=3, seed=7)


def test_a_layout_map_generates_per_layout_events_checked_against_each_rows_own_bounds(mixed_table):
    rows, layouts, table = mixed_table
    assert isinstance(layouts, fc.LayoutMap) and layouts.classes() == ((4, 2), (5, 3))
    assert table.config["layout_shapes"] == {"H4L2": 28, "H5L3": 24} and len(table) == 52 * 3 * 4
    assert fc.verify_events_against(table, rows)
    sequences = {cid: (h, l) for cid, h, l in rows}
    for event in table.events:
        heavy, light = sequences[event.candidate_id]
        coords = [c for op in event.program.ops for c in op.coords()] + list(event.query.coords())
        assert all(c.index < (len(heavy) if c.chain == "H" else len(light)) for c in coords)
        cells, slot = oracle_cells(heavy, light, plain_ops(event.program))      # independent oracle
        if isinstance(event.query, fc.ResQuery):
            expected = cells[slot((event.query.coord.chain, event.query.coord.index))]
        else:
            a, b = event.query.a, event.query.b
            expected = cells[slot((a.chain, a.index))] == cells[slot((b.chain, b.index))]
        assert event.answer == expected
    report = fc.coverage_report(table)
    assert report["complete"] and set(report["by_layout"]) == {"H4L2", "H5L3"}
    assert all(entry["complete"] for entry in report["by_layout"].values())
    again = fc.generate_events(layouts, rows, universe=fc.UNIVERSE_TRAIN, instances_per_sequence=3,
                               seed=7)
    assert again.sha256 == table.sha256                                          # deterministic


def test_the_layout_map_is_part_of_the_event_table_identity(mixed_table):
    rows, layouts, table = mixed_table
    document = copy.deepcopy(table.document())
    assert document["layout"] == layouts.document()
    assert len(document["layout"]["candidates"]) == len(rows)
    assert fc.parse_event_document(copy.deepcopy(document), layouts).sha256 == table.sha256
    shapes = list(layouts.shapes)
    shapes[0] = (shapes[0][0], shapes[0][1] + 1, shapes[0][2])
    other = fc.LayoutMap("ACDE", tuple(shapes))
    assert fc.layouts_sha256(other) != fc.layouts_sha256(layouts)
    with pytest.raises(ValueError, match="layout differs"):
        fc.parse_event_document(copy.deepcopy(document), other)


def test_light_coordinates_use_the_shared_heavy_offset_and_no_row_gets_an_out_of_range_one():
    space = fc.CoordSpace(5, 3)
    short = fc.Layout(4, 2, "ACDE")
    assert fc.compile_instruction(space, short, fc.Program(()), fc.ResQuery(L(0))) \
        == ((fc.KIND_RES, 5, -1, -1, -1),)               # the SPACE's heavy size, not this row's 4
    swap = fc.compile_instruction(space, short, fc.Program((fc.SwapOp(H(3), L(1)),)),
                                  fc.EqQuery(H(0), L(0)))
    assert swap[0][1:3] == (3, 6)
    with pytest.raises(ValueError, match="outside"):
        fc.compile_instruction(space, short, fc.Program(()), fc.ResQuery(H(4)))   # fits the space
    with pytest.raises(ValueError, match="outside"):
        fc.compile_instruction(space, short, fc.Program(()), fc.ResQuery(L(2)))
    long = fc.Layout(5, 3, "ACDE")
    assert fc.compile_instruction(space, long, fc.Program(()), fc.ResQuery(H(4)))[0][1] == 4
    direct = fc.resolve(short, fc.Program(()), fc.ResQuery(L(1)))
    assert fc.compile_direct(space, short, direct) == ((fc.KIND_READ_ORIGINAL, 6, -1, -1, -1),)


def test_a_table_cannot_hold_an_event_out_of_range_on_its_own_shorter_row():
    big = fc.Layout(5, 2, "ACDE")
    program, query = fc.Program(()), fc.ResQuery(H(4))
    event = fc._make_event(big, "x", program, query, fc.resolve(big, program, query),
                           fc.answer(big, "ACDEA", "AC", program, query))
    with pytest.raises(ValueError, match="outside"):
        fc.EventTable(fc.LayoutMap("ACDE", (("x", 4, 2),)), (event,), fc.UNIVERSE_TRAIN, {}, ())
    assert len(fc.EventTable(fc.LayoutMap("ACDE", (("x", 5, 2),)), (event,), fc.UNIVERSE_TRAIN,
                             {}, ())) == 1


def test_a_sparse_layout_shape_reports_its_own_incomplete_coverage_while_others_complete():
    rows = random_rows(30, 41, "a", 4, 2) + random_rows(1, 42, "z", 5, 3)
    layouts = fc.layouts_for_rows(rows, "ACDE")
    table = fc.generate_events(layouts, rows, universe=fc.UNIVERSE_TRAIN, instances_per_sequence=2,
                               seed=1)
    report = fc.coverage_report(table)
    missing = fc.incomplete_coverage(report)
    assert report["complete"] is False and set(missing) == {"H5L3"}
    assert "every_coordinate_in_a_SET" in missing["H5L3"]
    assert report["by_layout"]["H4L2"]["complete"] is True


def test_layouts_for_rows_returns_one_layout_when_uniform_and_refuses_rows_below_the_minimum():
    uniform = random_rows(5, 51, "u", 4, 2)
    assert fc.layouts_for_rows(uniform, "ACDE") == fc.Layout(4, 2, "ACDE")
    with pytest.raises(ValueError, match="fewer than 6 coordinates"):
        fc.layouts_for_rows(uniform + [("tiny", "AC", "DE")], "ACDE")
    with pytest.raises(ValueError, match="at least 6 coordinates"):
        fc.generate_events(fc.LayoutMap("ACDE", (("tiny", 2, 2),)), [("tiny", "AC", "DE")],
                           universe=fc.UNIVERSE_TRAIN, instances_per_sequence=1, seed=0)


# ---------------------------------------------------------------------------
# Q must corrupt the relation, not rename classes
# ---------------------------------------------------------------------------

def fake_table(codes):
    events = [types.SimpleNamespace(output_type="equality", template_id="t", stratum="eq_primary")
              for _ in codes]
    array = np.array(codes, dtype=np.int64)
    return types.SimpleNamespace(events=events, codes=lambda: array)


def splits_a_repeated_class(before, after):
    return any(len(set(after[before == value].tolist())) > 1
               for value in set(before.tolist()) if int((before == value).sum()) >= 2)


def test_four_distinct_labels_cannot_supply_a_corruption_control_only_a_renaming():
    layout = fc.Layout(4, 4, "ACDE")
    events = [hand_event(layout, f"q{i}", symbol + "AAA", "AAAA", fc.Program(()), fc.ResQuery(H(0)))
              for i, symbol in enumerate("ACDE")]
    table = fc.EventTable(layout, tuple(events), fc.UNIVERSE_TRAIN, {}, ())
    assert table.codes().tolist() == [0, 1, 2, 3]
    for seed in range(5):
        shuffled, report = fc.shuffle_targets(table, seed)
        assert shuffled.tolist() == [0, 1, 2, 3] and report["changed_values"] == 0
        assert report["strata"][0]["status"] == "unsupported_singleton_classes"
        assert len(report["unsupported_strata"]) == 1


@pytest.mark.parametrize("codes,status", [([0, 1, 1], "unsupported_small"),
                                          ([2, 2, 2, 2, 2], "unsupported_constant"),
                                          ([0, 1, 2, 3], "unsupported_singleton_classes")])
def test_unsupported_strata_are_named_and_left_unshuffled(codes, status):
    shuffled, report = fc.shuffle_targets(fake_table(codes), 0)
    assert shuffled.tolist() == codes and report["changed_values"] == 0
    assert report["strata"][0]["status"] == status and report["unsupported_strata"]


def test_a_valid_corruption_keeps_the_histogram_changes_and_keeps_values_and_splits_a_class():
    codes = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    for seed in range(20):
        shuffled, report = fc.shuffle_targets(fake_table(codes.tolist()), seed)
        assert sorted(shuffled.tolist()) == sorted(codes.tolist())
        changed = int(np.sum(shuffled != codes))
        assert 1 <= changed <= 7 and report["changed_values"] == changed
        assert splits_a_repeated_class(codes, shuffled), seed
        entry = report["strata"][0]
        assert entry["status"] in ("shuffled", "shuffled_best_effort")
        assert entry["split_original_classes"] >= 1 and entry["deterministic_class_mapping"] is False


def test_a_boolean_inversion_and_any_other_class_renaming_is_never_accepted():
    codes = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.int64)
    for seed in range(50):
        shuffled, _ = fc.shuffle_targets(fake_table(codes.tolist()), seed)
        assert shuffled.tolist() != (1 - codes).tolist()
        assert splits_a_repeated_class(codes, shuffled)
    tiny = np.array([1, 1, 0, 0], dtype=np.int64)
    for seed in range(20):
        shuffled, report = fc.shuffle_targets(fake_table(tiny.tolist()), seed)
        assert shuffled.tolist() not in ([0, 0, 1, 1], [1, 1, 0, 0])      # inversion / no change
        assert report["strata"][0]["status"].startswith("shuffled")


# ---------------------------------------------------------------------------
# a tape snapshot is independent and verified while it is made
# ---------------------------------------------------------------------------

def test_a_tape_snapshot_is_independent_and_verifies_while_it_is_made(train_table):
    _, table = train_table
    tape = make_tape(table)
    snapshot = tape.snapshot()
    assert snapshot.sha256 == tape.sha256 and snapshot.functional_labels is not tape.functional_labels
    tape.functional_labels.setflags(write=True)
    tape.functional_labels[0] += 5.0
    assert snapshot.verify()                          # the snapshot never saw the edit
    with pytest.raises(ValueError, match="recorded digest"):
        tape.snapshot()                               # a mutated tape cannot be snapshotted


# ===========================================================================
# repair 15c
# ===========================================================================

# ---------------------------------------------------------------------------
# C1: a retry must be a DIFFERENT draw
# ---------------------------------------------------------------------------

def pinned_decks(layout, coordinates, symbols, seed=0):
    """Decks exactly as ``generate_events`` builds them, with the SET deck queues left pending."""
    rng = np.random.default_rng(seed)
    decks = {"set_coord": fc._Deck(layout.coords(), rng), "swap_coord": fc._Deck(layout.coords(), rng),
             "chain_coord": fc._Deck(layout.coords(), rng),
             "set_symbol": fc._Deck(tuple(layout.alphabet), rng)}
    decks["set_coord"].queue = list(coordinates)
    decks["set_symbol"].queue = list(symbols)
    return decks, rng


def test_a_deferred_deck_item_returns_later_in_the_next_cycle_and_never_at_once():
    deck = fc._Deck(range(6), np.random.default_rng(0))
    deck.queue = [3]                                   # one pending item pins the next draw
    deck.begin()
    assert deck.draw() == 3
    deck.defer()                                       # the attempt was rejected: 3 is owed, not handed back
    assert deck.owed == [3]
    deck.begin()
    cycle = [deck.draw() for _ in range(6)]
    assert cycle[0] != 3 and cycle[-1] == 3 and sorted(cycle) == list(range(6))
    deck.begin()
    deck.draw()                                        # an ACCEPTED draw is simply consumed
    assert deck.owed == []


@pytest.mark.parametrize("heavy,light,odd", [("EEEE", "EDEE", 5), ("DDDD", "ADDD", 4)])
def test_a_pending_deck_cannot_pin_every_retry_to_the_same_infeasible_program(heavy, light, odd):
    """The seed-0 pilot refused sa-g-h3-l3-124 (EEEE/EDEE) and the selftest sa-g-h2-l2-002 (DDDD/ADDD):
    a pending SET queue gave SET(odd, A); SET(odd, A) on every one of 200 attempts, which erases the only
    differing residue, and restoring the queues made advancing the RNG irrelevant. A feasible program of
    the SAME class (two SETs elsewhere) leaves the odd residue and a true/false pair."""
    layout = fc.Layout(4, 4, "ACDE")
    decks, rng = pinned_decks(layout, [odd, odd], [0, 0])
    made, rejected, reasons = fc._draw_instance(fc.UNIVERSE_TRAIN, layout, "witness", heavy, light, 10,
                                                decks, rng, set(), [])
    assert made is not None and len(made) == 4 and rejected == 1
    assert any("no primary EQ pair" in reason for reason in reasons)
    program = made[0].program
    assert fc.program_class(program) == fc.CLASS_SET_SET      # the scheduled class is kept
    assert layout.coords()[odd] not in {op.coord for op in program.ops}
    eq = [event for event in made if event.output_type == fc.OUTPUT_EQUALITY]
    assert {event.answer for event in eq} == {True, False}
    assert eq[0].template_id == eq[1].template_id
    for event in made:
        assert event.program == program
        assert event.answer == fc.answer(layout, heavy, light, event.program, event.query)


def test_the_l42_fixture_roster_is_complete_including_the_row_the_old_retry_refused(train_table):
    rows, table = train_table
    assert dict((cid, (h, l)) for cid, h, l in rows)["s015"] == ("DDDC", "EA")
    assert len(table) == 24 * 3 * 4 and table.unique_sequences() == 24
    assert len([event for event in table.events if event.candidate_id == "s015"]) == 12
    program = fc.Program((fc.SetOp(H(3), "A"), fc.SetOp(L(0), "C")))     # independently feasible
    events = fc._train_instance(L42, "s015", "DDDC", "EA", program, fc.STRATUM_EQ_PRIMARY,
                                np.random.default_rng(0), [])
    assert len(events) == 4


# ---------------------------------------------------------------------------
# C2: the one-sided literal-literal control is kept, not deleted
# ---------------------------------------------------------------------------

def diverse_rows(count, seed, heavy=4, light=2):
    """Random sequences with at least three distinct residues (every class is plainly feasible)."""
    rng = np.random.default_rng(seed)
    rows = []
    while len(rows) < count:
        h = "".join("ACDE"[j] for j in rng.integers(0, 4, size=heavy))
        l = "".join("ACDE"[j] for j in rng.integers(0, 4, size=light))
        if len(set(h + l)) >= 3:
            rows.append((f"v{len(rows):03d}", h, l))
    return rows


@pytest.mark.parametrize("second,expected", [("C", False), ("A", True)])
def test_a_literal_literal_goal_keeps_a_one_sided_control_and_a_balanced_primary_pair(second,
                                                                                      expected):
    layout = fc.Layout(4, 4, "ACDE")
    program = fc.Program((fc.SetOp(H(0), "A"), fc.SetOp(H(1), second)))
    for seed in range(16):
        notes = []
        events = fc._train_instance(layout, "lit", "ACDE", "ACDE", program,
                                    fc.STRATUM_EQ_LITERAL_LITERAL, np.random.default_rng(seed), notes)
        assert len(events) == 4 and len({event.event_id for event in events}) == 4
        eq = [event for event in events if event.output_type == fc.OUTPUT_EQUALITY]
        res = [event for event in events if event.output_type == fc.OUTPUT_RESIDUE]
        assert len(eq) == 3 and len(res) == 1
        control = [event for event in eq if event.stratum == fc.STRATUM_EQ_LITERAL_LITERAL]
        primary = [event for event in eq if event.stratum == fc.STRATUM_EQ_PRIMARY]
        assert len(control) == 1 and control[0].answer is expected
        assert control[0].query == fc.EqQuery(H(0), H(1))
        assert len(primary) == 2 and {event.answer for event in primary} == {True, False}
        assert primary[0].template_id == primary[1].template_id
        assert "ONE-SIDED" in notes[-1]["reason"] and "unsupported" in notes[-1]["reason"]


def test_a_literal_literal_goal_without_a_literal_literal_pair_uses_the_primary_stratum():
    layout = fc.Layout(4, 4, "ACDE")
    program = fc.Program((fc.SetOp(H(0), "A"), fc.SetOp(H(0), "C")))     # both SETs hit one coordinate
    notes = []
    events = fc._train_instance(layout, "lit", "ACDE", "ACDE", program, fc.STRATUM_EQ_LITERAL_LITERAL,
                                np.random.default_rng(0), notes)
    assert len(events) == 4 and len([e for e in events if e.output_type == fc.OUTPUT_RESIDUE]) == 2
    assert all(event.stratum != fc.STRATUM_EQ_LITERAL_LITERAL for event in events)
    assert "no literal-literal EQ exists" in notes[-1]["reason"]


def test_generated_tables_keep_the_literal_control_and_report_the_actual_mix():
    rows = diverse_rows(60, 21)
    table = fc.generate_events(L42, rows, universe=fc.UNIVERSE_TRAIN, instances_per_sequence=4, seed=3)
    shapes = table.config["instance_shapes"]
    assert set(shapes) <= {"2RES+2EQ", "1RES+3EQ"} and sum(shapes.values()) == 60 * 4
    assert len(table) == 60 * 4 * 4 and len(set(table.ids())) == len(table)   # 4 unique answers each
    literal = [event for event in table.events if event.stratum == fc.STRATUM_EQ_LITERAL_LITERAL]
    assert literal and len(literal) == shapes["1RES+3EQ"]
    report = fc.balance_report(table)
    assert report["unbalanced_unit_count"] == 0               # every PRIMARY pair is still matched
    output = report["events_by_output"]
    assert output["residue"] + output["equality"] == len(table)
    assert output["residue"] == 2 * shapes.get("2RES+2EQ", 0) + shapes["1RES+3EQ"]
    status = report["secondary_eq_balance"]["eq_literal_literal"]
    assert status["true"] + status["false"] == len(literal)
    assert status["balance"] == ("both_answers_present" if status["true"] and status["false"]
                                 else "one_sided_unsupported")
    assert fc.coverage_report(table)["complete"] and fc.verify_events_against(table, rows)


# ---------------------------------------------------------------------------
# C5: bounded constant filtering and a PUBLISHED selection record
# ---------------------------------------------------------------------------

def test_capacity_exceeding_source_counts_are_refused_up_front_not_by_hanging():
    with pytest.raises(ValueError, match="source_aux"):
        synth.SyntheticSpec(source_train=1, source_aux=4096, source_dev=1, target_train=1,
                            target_eval=1)
    spec = synth.SyntheticSpec(source_train=1, source_aux=4095, source_dev=1, target_train=1,
                               target_eval=1)
    assert spec.totals()["source_aux"] == 4 * 4095
    with pytest.raises(ValueError, match="non-constant"):                  # and the draw itself is bounded
        synth._replace_constant(list(range(4096)), 0, 0, np.random.default_rng(0))


def test_the_constant_replacement_draws_directly_from_the_available_codes():
    chosen = [code for code in range(4096) if code != 5]      # every interior but one non-constant code
    codes, replaced = synth._replace_constant(chosen, 0, 0, np.random.default_rng(0))
    assert replaced == [0] and codes == list(range(1, 4096))


def test_write_inputs_publishes_the_selection_record(world_inputs):
    world, paths = world_inputs
    assert Path(paths["provenance"]).name == synth.PROVENANCE_NAME
    record = fb.read_json(paths["provenance"])
    spec = world.spec
    assert record["schema"] == synth.PROVENANCE_SCHEMA
    assert record["spec"] == {"source_train": spec.source_train, "source_aux": spec.source_aux,
                              "source_dev": spec.source_dev, "target_train": spec.target_train,
                              "target_eval": spec.target_eval, "seed": spec.seed}
    assert record["totals"] == spec.totals()
    assert record["excluded"] == [dict(entry) for entry in world.excluded]
    assert record["excluded_count"] == len(world.excluded)
    assert record["cohort_document_sha256"] == fb.digest(fb.read_json(paths["cohorts"]))
    assert record["roster_sha256"]["source_aux"] == fb.digest(list(world.roles["source_aux"]))
    assert "default_rng" in record["selection_rule"] and "not a measured label" in record["kind"]


def test_a_real_exclusion_and_its_replacement_are_published_in_the_record(tmp_path):
    spec = synth.SyntheticSpec(source_train=1, source_aux=4095, source_dev=1, target_train=1,
                               target_eval=1, seed=0)
    world = synth.build_world(spec)                # 4095 of 4096 interiors per group: constants are drawn
    assert world.excluded, "drawing 4095 of 4096 interiors per group selects a constant one"
    path = synth.write_provenance(world, tmp_path / synth.PROVENANCE_NAME)
    record = fb.read_json(path)
    assert record["excluded_count"] == len(world.excluded) == len(record["excluded"]) >= 1
    sequences = {(row["heavy"], row["light"]) for row in world.rows}
    for entry in record["excluded"]:
        h0 = int(entry["group"].split("-")[1][1:])
        assert entry["cohort"] == "source" and entry["group"] == f"g-h{h0}-l{h0}"
        assert entry["interior_code"] == h0 * 1365                      # all six interior symbols equal H0
        heavy, light = entry["excluded_sequence"]["heavy"], entry["excluded_sequence"]["light"]
        assert len(set(heavy + light)) == 1 and (heavy, light) not in sequences
        assert not synth._is_constant(entry["replacement_interior_code"], h0, h0)
        assert synth._sequence_of(entry["replacement_interior_code"], h0, h0) in sequences
    assert all(len(set(row["heavy"] + row["light"])) > 1 for row in world.rows
               if row["cohort"] == "source")
    assert len(world.roles["source_aux"]) == 4 * 4095              # the declared count is unchanged
    with pytest.raises(ValueError, match="already exists"):
        synth.write_provenance(world, path)                        # written once, never replaced
