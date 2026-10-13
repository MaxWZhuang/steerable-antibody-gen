"""Planning, budget and scoring of the visible-only masked diagnostics, against closed-form oracles.

All sequences, ids and values are invented; the analytic backends are toys (see the world module).
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest

from smallAntibodyGen.experiments.her2_multiscale.masks import MODE_FULL, MODE_PARTIAL, draw_mask
from smallAntibodyGen.experiments.interaction_qualification import diagnostics as dg
from smallAntibodyGen.models.antibody_antigen.views import CANONICAL_RESIDUES, mask_complete
from smallAntibodyGen.tests import aa_qualification_world as world
from smallAntibodyGen.tests.aa_qualification_diagnostics_world import (
    COLUMN_SUM,
    LN20,
    ROW_SUM,
    TOTAL,
    AntigenPullBackend,
    JointTwoSiteBackend,
    drow,
    make_config,
    plan_for,
    signature,
)
from smallAntibodyGen.tests.aa_world import FixedLogitsBackend, RecordingBackend

FOUR = [("H", 0), ("H", 1), ("H", 2), ("H", 3)]
OFF = {"enabled": False, "max_pairs": 1, "kappa": 1.0}


def on(max_pairs: int = 8, kappa: float = 1.0):
    return {"enabled": True, "max_pairs": max_pairs, "kappa": kappa}


def execute(backend, rows, **config):
    """Plan, then score under a counting wrapper whose limit is the plan: every call must be a planned one."""
    plan = plan_for(rows, make_config(**config))
    counting = dg.CountingBackend(backend, {dg.PHASE_DIAGNOSTICS: plan.budget.diagnostics})
    with counting.phase(dg.PHASE_DIAGNOSTICS):
        cells = dg.run_singleton_tasks(counting, plan.singleton_tasks)
        pairs = dg.run_pair_tasks(counting, plan.pair_tasks)
    counting.verify()
    return plan, cells, pairs


# --- budget formulas and counts ------------------------------------------------------------------------


def test_budget_formulas_match_the_derivation():
    assert [dg.order_count(n) for n in (1, 2, 3)] == [1, 2, 2]
    assert [dg.singleton_cell_cost(k) for k in (1, 2, 3, 4)] == [6, 14, 20, 26]
    assert [dg.pair_task_cost(d) for d in (1, 2, 3)] == [4, 16, 24]


def test_two_antibodies_one_antigen_cost_exactly_233_forwards():
    rows = [drow("a", "ACDE"), drow("b", "ACFG")]
    plan = plan_for(rows, make_config(mask_counts=[1, 2, 3, "full"], geometries=["suffix", "subset"], local_gap=on()))
    assert plan.budget.feature == 5  # two joint, two antibody-only, ONE shared antigen view
    assert [task.cost for task in plan.singleton_tasks] == [106, 106]
    assert plan.budget.singleton == 212 and plan.budget.pair == 16 and plan.budget.total == 233
    assert len(plan.pair_tasks) == 1 and plan.pair_tasks[0].distance == 2
    for task in plan.singleton_tasks:
        assert [(c.count, c.geometry) for c in task.cases] == [
            (1, "suffix"), (1, "subset"), (2, "suffix"), (2, "subset"), (3, "suffix"), (3, "subset"), (4, "full"),
        ]  # fmt: skip
    assert plan.budget.within_budget is True  # the default max_forwards is far above 233
    with pytest.raises(dg.BudgetExceeded):
        dg.require_within_budget(dg.ForwardBudget(5, 212, 16, 232))
    dg.require_within_budget(dg.ForwardBudget(5, 212, 16, 233))


def test_the_exact_plan_is_what_a_backend_is_actually_called_with():
    rows = [drow("a", "ACDE"), drow("b", "ACFG")]
    spy = RecordingBackend(FixedLogitsBackend())
    plan, _, _ = execute(spy, rows, mask_counts=[1, 2, 3, "full"], geometries=["suffix", "subset"], local_gap=on())
    assert len(spy.views) == plan.budget.diagnostics == 228  # 212 + 16, features excluded
    # diagnostics only ever forward masked or progressively revealed states, never a full unmasked view
    assert all(view.hidden_sites() for view in spy.views)


def test_feasible_cases_dedupe_skip_and_run_the_full_mask_once():
    both = ["subset", "suffix"]  # listed out of order on purpose
    cases, skipped = dg.feasible_cases(4, [1, 2, 3, "full"], both)
    assert cases == [(1, "suffix"), (1, "subset"), (2, "suffix"), (2, "subset"), (3, "suffix"), (3, "subset"), (4, "full")]
    assert skipped == []
    assert dg.feasible_cases(4, [4, "full"], both) == ([(4, "full")], [])
    assert dg.feasible_cases(2, [1, 5, "full"], ["suffix"]) == ([(1, "suffix"), (2, "full")], [5])
    assert dg.feasible_cases(1, [1, 2, 3], both) == ([(1, "full")], [2, 3])
    assert dg.feasible_cases(3, [7], ["suffix"]) == ([], [7])


def test_counts_above_the_region_are_skipped_and_reported_never_truncated():
    plan = plan_for([drow("a", "ACDE", region=[("H", 0), ("H", 1)])], make_config(mask_counts=[1, 5, "full"], local_gap=OFF))
    (task,) = plan.singleton_tasks
    assert task.skipped_counts == (5,)
    assert [(c.count, c.geometry) for c in task.cases] == [(1, "suffix"), (2, "full")]
    assert plan.singleton_stats["skipped_count_entries"] == 1


# --- masks and orders -------------------------------------------------------------------------------------


def test_fixed_count_geometry_matches_the_existing_draw_helper():
    region = [("H", 1), ("H", 3), ("H", 4), ("L", 0), ("L", 2)]
    lengths = dict(heavy_length=6, light_length=4)
    suffixes = {}
    for seed in range(300):
        draw = draw_mask(
            region,
            count_rng=np.random.default_rng(seed),
            position_rng=np.random.default_rng(10_000 + seed),
            geometry="suffix",
            full_probability=0.0,
            **lengths,
        )
        assert draw.mode == MODE_PARTIAL
        suffixes[draw.count] = draw.sites
    assert sorted(suffixes) == [1, 2, 3, 4]
    for count, sites in suffixes.items():
        mine = dg.fixed_count_mask(region, count, "suffix", seed=0, content_key="c", **lengths)
        assert (mine.sites, mine.count, mine.size, mine.mode, mine.geometry) == (sites, count, 5, MODE_PARTIAL, "suffix")
    full = dg.fixed_count_mask(region, 5, "subset", seed=0, content_key="c", **lengths)
    assert full.sites == tuple(region) and full.mode == MODE_FULL and full.geometry == "full" and full.count == 5
    scrambled = dg.fixed_count_mask(list(reversed(region)), 2, "suffix", seed=0, content_key="c", **lengths)
    assert scrambled.sites == (("L", 0), ("L", 2))  # the suffix lives in canonical coordinates
    reference = draw_mask(
        region, count_rng=np.random.default_rng(1), position_rng=np.random.default_rng(2), geometry="subset",
        full_probability=0.0, **lengths,
    )  # fmt: skip
    mine = dg.fixed_count_mask(region, reference.count, "subset", seed=0, content_key="c", **lengths)
    rank = lambda coord: (coord[0] != "H", coord[1])  # noqa: E731
    assert list(mine.sites) == sorted(mine.sites, key=rank) and set(mine.sites) <= set(region)
    assert len(mine.sites) == reference.count and list(reference.sites) == sorted(reference.sites, key=rank)


def test_fixed_count_mask_refuses_bad_counts_and_geometries():
    lengths = dict(heavy_length=4, light_length=0, seed=0, content_key="c")
    for bad in (0, 5, True, 1.0, "2"):
        with pytest.raises(ValueError):
            dg.fixed_count_mask(FOUR, bad, "suffix", **lengths)
    with pytest.raises(ValueError):
        dg.fixed_count_mask(FOUR, 2, "prefix", **lengths)


def test_subset_masks_are_uniform_without_replacement_and_depend_on_content_and_seed_only():
    seen, positions = Counter(), Counter()
    for index in range(600):
        draw = dg.fixed_count_mask(FOUR, 2, "subset", seed=3, content_key=f"content-{index}", heavy_length=4, light_length=0)
        assert len(set(draw.sites)) == draw.count == 2
        seen[draw.sites] += 1
        positions.update(draw.sites)
    assert len(seen) == 6
    assert all(0.35 < positions[coord] / 600 < 0.65 for coord in FOUR)
    again = dg.fixed_count_mask(FOUR, 2, "subset", seed=3, content_key="content-7", heavy_length=4, light_length=0)
    assert again.sites == dg.fixed_count_mask(FOUR, 2, "subset", seed=3, content_key="content-7", heavy_length=4, light_length=0).sites
    assert any(
        dg.fixed_count_mask(FOUR, 2, "subset", seed=1, content_key=f"k{i}", heavy_length=4, light_length=0).sites
        != dg.fixed_count_mask(FOUR, 2, "subset", seed=2, content_key=f"k{i}", heavy_length=4, light_length=0).sites
        for i in range(50)
    )


def test_reveal_order_comes_from_visible_content_never_from_truth_and_reverse_is_literal():
    plan = plan_for(
        [drow("x", "ACDE"), drow("y", "ACFG")], make_config(mask_counts=[1, 2], geometries=["suffix"], local_gap=OFF)
    )
    by_heavy = {task.heavy: task for task in plan.singleton_tasks}
    two_x = [c for c in by_heavy["ACDE"].cases if c.count == 2][0]
    two_y = [c for c in by_heavy["ACFG"].cases if c.count == 2][0]
    # same visible background (AC--) and antigen, different hidden truth (DE vs FG)
    assert two_x.sites == two_y.sites == (("H", 2), ("H", 3))
    assert two_x.view_digest == two_y.view_digest and two_x.order == two_y.order
    assert two_x.reverse_order == tuple(reversed(two_x.order)) and set(two_x.order) == set(two_x.sites)
    one = [c for c in by_heavy["ACDE"].cases if c.count == 1][0]
    assert one.reverse_order is None and len(one.order) == 1
    sites = tuple(FOUR)
    orders = {dg.explicit_order(sites, "digest", seed) for seed in range(30)}
    assert len(orders) > 1 and all(set(order) == set(sites) for order in orders)
    assert dg.explicit_order(sites, "digest", 5) == dg.explicit_order(list(reversed(sites)), "digest", 5)
    assert len({dg.explicit_order(sites, f"digest-{i}", 5) for i in range(30)}) > 1


# --- selection --------------------------------------------------------------------------------------------


def test_one_task_per_content_and_region_lists_every_record_that_measured_it():
    rows = [
        drow("r1", "ACDE", assay="a1"),
        drow("r2", "ACDE", assay="a2"),
        drow("r3", "ACDE", assay="a3", region=[("H", 1), ("H", 2), ("H", 3)]),  # same content, another region
        drow("r4", "ACFG", assay="a1"),
    ]
    plan = plan_for(rows, make_config(mask_counts=[1], local_gap=OFF))
    assert sorted(task.record_ids for task in plan.singleton_tasks) == [("r1", "r2"), ("r3",), ("r4",)]
    two_assays = [task for task in plan.singleton_tasks if task.record_ids == ("r1", "r2")][0]
    assert two_assays.sources_assays == (("s1", "a1"), ("s1", "a2"))


def test_records_without_a_region_are_counted_and_never_planned():
    rows = [drow("r1", "ACDE", region=None), drow("r2", "ACFG")]
    plan = plan_for(rows, make_config(mask_counts=[1], local_gap=OFF))
    assert [task.record_ids for task in plan.singleton_tasks] == [("r2",)]
    assert plan.singleton_stats["records_missing_region"] == 1


def test_max_examples_keeps_the_lowest_content_priorities_whatever_the_names_and_values():
    heavies = ["ACDE", "ACDF", "ACDG", "ACDH", "ACDI", "ACDK"]
    rows = [drow(f"id-{i}", h, value=7.0 + i) for i, h in enumerate(heavies)]
    everything = plan_for(rows, make_config(max_examples=6, mask_counts=[1], local_gap=OFF))
    capped = plan_for(rows, make_config(max_examples=2, mask_counts=[1], local_gap=OFF))
    assert [t.task_id for t in capped.singleton_tasks] == [t.task_id for t in everything.singleton_tasks[:2]]
    assert capped.singleton_stats["cap_binding"] is True and everything.singleton_stats["cap_binding"] is False
    renamed = [drow(f"zz-{i}", h, source="other", assay="other-assay", value=1.0) for i, h in reversed(list(enumerate(heavies)))]
    again = plan_for(renamed, make_config(max_examples=2, mask_counts=[1], local_gap=OFF))
    assert [t.task_id for t in again.singleton_tasks] == [t.task_id for t in capped.singleton_tasks]
    other_seed = plan_for(rows, make_config(max_examples=6, seed=1, mask_counts=[1], local_gap=OFF))
    assert [t.task_id for t in other_seed.singleton_tasks] != [t.task_id for t in everything.singleton_tasks]


def test_configured_partitions_filter_diagnostics_and_the_sealed_test_never_reaches_the_planner():
    rows = [
        drow("t", "ACDE", partition="train", antigen="EFGK"),
        drow("d", "ACFG", partition="development", antigen="LMNP"),
        drow("x", "WWWW", partition="test", antigen="QRST"),
    ]
    plan = plan_for(rows, make_config(partitions=["development"], mask_counts=[1], local_gap=OFF))
    assert [task.partition for task in plan.singleton_tasks] == ["development"]
    unsealed = plan_for(rows, make_config(evaluate_test=True, partitions=["test"], mask_counts=[1], local_gap=OFF))
    assert [task.partition for task in unsealed.singleton_tasks] == ["test"]
    panel = world.panel_of(rows)
    with pytest.raises(ValueError, match="sealed test"):
        dg.plan_diagnostics(panel.records, make_config().diagnostics, 0)


# --- pair tasks ---------------------------------------------------------------------------------------------


def test_a_pair_task_gathers_every_same_assay_contrast_and_never_builds_a_cross_assay_one():
    rows = [
        drow("a1", "ACDE", assay="a1"), drow("b1", "ACDF", assay="a1"),
        drow("a2", "ACDE", assay="a2"), drow("b2", "ACDF", assay="a2"),
        drow("c3", "ACDG", assay="a3"),  # alone in its assay: its neighbours were measured elsewhere
        drow("z1", "WCDE", assay="a4"), drow("z2", "ACDE", assay="a5"),  # near-identical across assays only
    ]  # fmt: skip
    plan = plan_for(rows, make_config(local_gap=on()))
    (task,) = plan.pair_tasks
    assert [(c.source, c.assay, c.record_id_a, c.record_id_b) for c in task.contrasts] == [
        ("s1", "a1", "a1", "b1"), ("s1", "a2", "a2", "b2"),
    ]  # fmt: skip
    assert plan.pair_stats["eligible_contrasts_scanned"] == 2 and plan.pair_stats["selected_contrasts"] == 2
    assert task.a.heavy == "ACDE" and task.b.heavy == "ACDF" and task.changed == (("H", 3),)


@pytest.mark.parametrize(
    "rows, reason",
    [
        ([drow("a", "ACDE"), drow("b", "ACDF", region=[("H", 0), ("H", 1)])], "region_mismatch"),
        ([drow("a", "ACDEF", region=[("H", 0), ("H", 1), ("H", 2)]), drow("b", "ACDEG", region=[("H", 0), ("H", 1), ("H", 2)])], "edits_outside_region"),
        ([drow("a", "ACDE"), drow("b", "ACDEF", region=FOUR)], "chain_or_length_mismatch"),
        ([drow("a", "ACDE"), drow("b", "ACDE", light="KL", region=FOUR)], "chain_or_length_mismatch"),
        ([drow("a", "ACDE"), drow("b", "FGHI")], "distance_above_3"),
    ],
)  # fmt: skip
def test_unsupported_pairs_are_counted_not_built(rows, reason):
    plan = plan_for(rows, make_config(local_gap=on()))
    assert plan.pair_tasks == () and plan.pair_stats["refused"] == {reason: 1}
    assert plan.pair_stats["screened_pairs"] == 1 and plan.budget.pair == 0


def test_distance_three_is_the_largest_supported_edit():
    (task,) = plan_for([drow("a", "ACDE"), drow("b", "FGHE")], make_config(local_gap=on())).pair_tasks
    assert task.distance == 3 and task.cost == dg.pair_task_cost(3) == 24
    assert task.changed == (("H", 0), ("H", 1), ("H", 2))


def test_the_pair_cap_keeps_the_global_lowest_priorities_not_the_first_encountered():
    rows = [drow(f"r{i}", "ACD" + residue) for i, residue in enumerate("EFGHI")]  # 10 single-edit pairs
    everything = plan_for(rows, make_config(local_gap=on(max_pairs=20)))
    assert len(everything.pair_tasks) == 10 and everything.pair_stats["cap_binding"] is False
    assert everything.pair_stats["screened_pairs"] == 10 == everything.pair_stats["eligible_contrasts_scanned"]
    ranked = sorted(everything.pair_tasks, key=lambda t: (t.priority, t.a_fingerprint, t.b_fingerprint))
    assert [t.task_id for t in everything.pair_tasks] == [t.task_id for t in ranked]
    for cap in (1, 3, 7):
        capped = plan_for(rows, make_config(local_gap=on(max_pairs=cap)))
        assert [t.task_id for t in capped.pair_tasks] == [t.task_id for t in ranked[:cap]]
        assert capped.pair_stats["cap_binding"] is True
        reversed_rows = plan_for(list(reversed(rows)), make_config(local_gap=on(max_pairs=cap)))
        assert [t.task_id for t in reversed_rows.pair_tasks] == [t.task_id for t in capped.pair_tasks]


def test_competing_pairs_in_several_assays_are_decided_by_content_not_by_names():
    def build(assays):
        first, second = assays
        return [
            drow("p1", "ACDE", assay=first), drow("p2", "ACDF", assay=first), drow("p3", "ACDG", assay=first),
            drow("q1", "ACDE", assay=second), drow("q2", "ACDF", assay=second),
        ]  # fmt: skip

    config = make_config(local_gap=on(max_pairs=1))
    one = plan_for(build(("alpha", "beta")), config)
    swapped = plan_for(build(("beta", "alpha")), config)
    assert [t.task_id for t in one.pair_tasks] == [t.task_id for t in swapped.pair_tasks]
    assert [len(t.contrasts) for t in one.pair_tasks] == [len(t.contrasts) for t in swapped.pair_tasks]


def test_plan_is_invariant_to_row_order_ids_names_and_measured_values():
    heavies = ["ACDE", "ACDF", "ACDG", "ACFG", "WCDE"]
    base = [
        drow(f"r{i}-{a}", h, assay=a, value=6.0 + i)
        for i, h in enumerate(heavies)
        for a in ("a1", "a2")
        if a == "a1" or i < 3
    ]
    variant = [
        drow(f"q{index}", row["heavy"], source="zeta", assay={"a1": "b-late", "a2": "a-early"}[row["assay"]], value=9.5 - index)
        for index, row in reversed(list(enumerate(base)))
    ]
    config = make_config(mask_counts=[1, 2, "full"], geometries=["suffix", "subset"], max_examples=3, local_gap=on(max_pairs=2))
    one, two = plan_for(base, config), plan_for(variant, config)
    assert signature(one) == signature(two)
    assert [len(t.contrasts) for t in one.pair_tasks] == [len(t.contrasts) for t in two.pair_tasks]
    assert one.singleton_stats["cap_binding"] and one.pair_stats["cap_binding"]


# --- counting backend -----------------------------------------------------------------------------------------


def test_counting_backend_refuses_overruns_and_confirms_the_planned_totals():
    inner = RecordingBackend(FixedLogitsBackend())
    counting = dg.CountingBackend(inner, {"features": 1, "diagnostics": 2})
    view, _ = mask_complete({"H": "ACD"})
    with pytest.raises(dg.BudgetOverrun):
        counting.forward(view)  # outside a budgeted phase
    with counting.phase("features"):
        counting.forward(view)
        with pytest.raises(dg.BudgetOverrun):
            counting.forward(view)
        with pytest.raises(RuntimeError):
            with counting.phase("diagnostics"):
                pass
    with pytest.raises(dg.BudgetMismatch):
        counting.verify()
    with counting.phase("diagnostics"):
        counting.forward(view)
        counting.forward(view)
    counting.verify()
    assert counting.calls == {"features": 1, "diagnostics": 2} and counting.total_calls == 3 == len(inner.views)
    with pytest.raises(KeyError):
        counting.phase("elsewhere").__enter__()


def test_a_backend_that_was_never_built_cannot_be_called():
    counting = dg.CountingBackend(None, {"features": 0})
    counting.verify()
    with counting.phase("features"):
        with pytest.raises(dg.BudgetOverrun):
            counting.forward(mask_complete({"H": "ACD"})[0])


# --- oracle A: uniform logits ---------------------------------------------------------------------------------


def test_uniform_logits_give_closed_form_scores_and_the_mass_changes_only_the_mass():
    rows = [drow("a", "ACDE"), drow("b", "ACDF"), drow("c", "ACFG")]
    config = dict(mask_counts=[1, 2, "full"], geometries=["suffix", "subset"], local_gap=on())
    plan, high, pairs_high = execute(FixedLogitsBackend(mass=0.9), rows, **config)
    _, low, pairs_low = execute(FixedLogitsBackend(mass=0.2), rows, **config)
    assert len(high) == len(low) == 2 * plan.n_cells and pairs_high
    for hi, lo in zip(high, low):
        k = hi["count"]
        assert hi["nll"]["sum"] == pytest.approx(k * LN20, rel=1e-5)
        assert hi["nll"]["token_mean"] == pytest.approx(LN20, rel=1e-5)
        assert hi["sequential"]["forward"]["sum"] == pytest.approx(-k * LN20, rel=1e-5)
        assert hi["sequential"]["forward"]["mean"] == pytest.approx(-LN20, rel=1e-5)
        if k > 1:
            assert hi["sequential"]["reverse"]["sum"] == pytest.approx(-k * LN20, rel=1e-5)
            assert hi["order_sensitivity"]["sum_difference"] == pytest.approx(0.0, abs=1e-5)
            assert hi["order_sensitivity"]["max_abs_per_site"] == pytest.approx(0.0, abs=1e-5)
        else:
            assert hi["sequential"]["reverse"] is None and hi["order_sensitivity"] is None
        assert hi["nll"]["sum"] == lo["nll"]["sum"] and hi["sequential"]["forward"]["sum"] == lo["sequential"]["forward"]["sum"]
        assert hi["nll"]["canonical_mass"] == pytest.approx([0.9] * k, rel=1e-5)
        assert lo["nll"]["canonical_mass"] == pytest.approx([0.2] * k, rel=1e-5)
        assert lo["sequential"]["forward"]["step_canonical_mass"] == pytest.approx([0.2] * k, rel=1e-5)
        assert hi["completion"]["total"] == k and len(hi["completion"]["residues"]) == k
    for result in pairs_high + pairs_low:
        for arm in dg.ARMS:
            for name in dg.ORDER_NAMES:
                if name in result["arms"][arm]:
                    assert result["arms"][arm][name]["gap"] == pytest.approx(0.0, abs=1e-6)
            diff = result["arms"][arm]["order_gap_difference"]
            assert diff is None or diff == pytest.approx(0.0, abs=1e-6)


def test_native_mask_ratio_differs_between_arms_because_the_antigen_adds_unmasked_tokens():
    rows = [drow("a", "ACDE", antigen="KLM")]
    _, cells, _ = execute(FixedLogitsBackend(), rows, mask_counts=[2], geometries=["suffix"], local_gap=OFF)
    joint, alone = [c for c in cells if c["arm"] == "joint"][0], [c for c in cells if c["arm"] == "antibody_only"][0]
    assert joint["nll"]["mask_ratio"] == pytest.approx(2 / 7) and alone["nll"]["mask_ratio"] == pytest.approx(2 / 4)
    summary = dg.summarize_singletons(cells)
    (paired,) = summary["paired_joint_minus_antibody_only"]
    assert paired["native_mask_ratio"]["joint_over_antibody_only_mean"] == pytest.approx(4 / 7)
    assert summary["antigen_specificity_of_joint_gap"] == "unestablished"
    assert "antigen-specificity claim" in summary["native_confound_note"]


def test_completion_preserves_known_residues_and_recovery_scores_the_hidden_sites_only():
    rows = [drow("aa", "CAAA"), drow("other", "CDEF")]
    plan, cells, _ = execute(FixedLogitsBackend(), rows, mask_counts=[2], geometries=["suffix"], local_gap=OFF)
    task_of = {task.heavy: task.task_id for task in plan.singleton_tasks}
    for cell in cells:
        assert cell["completion"]["residues"] == "AA" and cell["completion"]["known_residues_preserved"] is True
        assert sorted(cell["completion"]["order"]) == ["H:2", "H:3"]  # only hidden coordinates are ever filled
    easy = [c for c in cells if c["task_id"] == task_of["CAAA"]]
    hard = [c for c in cells if c["task_id"] == task_of["CDEF"]]
    assert all((c["completion"]["correct"], c["completion"]["total"], c["completion"]["exact"]) == (2, 2, True) for c in easy)
    assert all((c["completion"]["correct"], c["completion"]["total"], c["completion"]["exact"]) == (0, 2, False) for c in hard)
    groups = {g["arm"]: g for g in dg.summarize_singletons(cells)["groups"]}
    assert groups["joint"]["recovery"]["macro_case_mean_rate"] == pytest.approx(0.5)
    assert groups["joint"]["recovery"]["token_weighted_rate"] == pytest.approx(0.5)
    assert groups["joint"]["recovery"]["exact_fraction"] == pytest.approx(0.5)


# --- oracle B: a consistent two-site joint ------------------------------------------------------------------------


def _two_site_rows(layout):
    if layout == "heavy_heavy":
        return (("H", 0), ("H", 1)), (lambda rid, residue: drow(rid, residue * 2))
    return (("H", 0), ("L", 0)), (
        lambda rid, residue: drow(rid, residue, light=residue, region=[("H", 0), ("L", 0)])
    )  # fmt: skip


@pytest.mark.parametrize("layout", ["heavy_heavy", "heavy_light"])
def test_a_consistent_joint_gives_closed_form_scores_the_same_in_both_orders(layout):
    sites, make = _two_site_rows(layout)
    rows = [make("aa", "A"), make("cc", "C")]
    backend = JointTwoSiteBackend(sites)
    plan, cells, pairs = execute(backend, rows, mask_counts=["full"], geometries=["suffix"], local_gap=on())
    task_of = {task.pair_fingerprint: task for task in plan.singleton_tasks}
    ids = {("A" if "A" in t.heavy else "C"): t.task_id for t in task_of.values()}
    j_aa, j_cc = math.log(4 / TOTAL), math.log(8 / TOTAL)
    nll_aa = math.log(TOTAL**2 / (ROW_SUM["A"] * COLUMN_SUM["A"]))  # = ln(411^2 / (23 * 24)), NOT the joint NLL
    nll_cc = math.log(TOTAL**2 / (ROW_SUM["C"] * COLUMN_SUM["C"]))
    assert nll_aa == pytest.approx(math.log(411**2 / (23 * 24)))
    assert nll_aa != pytest.approx(-j_aa, abs=1e-3)
    paths = {}
    for cell in cells:
        aa = cell["task_id"] == ids["A"]
        assert cell["nll"]["sum"] == pytest.approx(nll_aa if aa else nll_cc, abs=1e-5)
        for name in ("forward", "reverse"):
            assert cell["sequential"][name]["sum"] == pytest.approx(j_aa if aa else j_cc, abs=1e-5)
        assert cell["order_sensitivity"]["sum_difference"] == pytest.approx(0.0, abs=1e-5)
        assert cell["order_sensitivity"]["max_abs_per_site"] > 0.1  # the per-site terms do depend on the order
        completion = cell["completion"]
        assert completion["residues"] == "CC"  # the greedy path is the same for both truths
        paths.setdefault(cell["arm"], set()).add((tuple(completion["order"]), completion["residues"]))
        assert (completion["correct"], completion["total"]) == ((0, 2) if aa else (2, 2))
        assert completion["exact"] is (not aa)
    assert all(len(found) == 1 for found in paths.values())  # identical path before external truth scoring
    (result,) = pairs
    assert result["distance"] == 2
    for arm in dg.ARMS:
        for name in dg.ORDER_NAMES:
            entry = result["arms"][arm][name]
            assert entry["gap"] == pytest.approx(math.log(2.0), abs=1e-5)
            assert entry["j_a"] == pytest.approx(j_aa, abs=1e-5) and entry["j_b"] == pytest.approx(j_cc, abs=1e-5)
        assert result["arms"][arm]["order_gap_difference"] == pytest.approx(0.0, abs=1e-5)


def test_a_two_site_pair_costs_sixteen_forwards_and_each_task_is_scored_once_per_arm_and_order():
    sites, make = _two_site_rows("heavy_heavy")
    backend = JointTwoSiteBackend(sites)
    plan = plan_for([make("aa", "A"), make("cc", "C")], make_config(mask_counts=["full"], local_gap=on()))
    counting = dg.CountingBackend(backend, {dg.PHASE_DIAGNOSTICS: plan.budget.pair})
    with counting.phase(dg.PHASE_DIAGNOSTICS):
        dg.run_pair_tasks(counting, plan.pair_tasks)
    counting.verify()
    assert plan.budget.pair == 16 == backend.forwards


# --- oracle C: a visible antigen changes the answer (plumbing only) -----------------------------------------------


def near(value):
    return pytest.approx(value, rel=1e-5, abs=1e-6)


def test_an_antigen_reading_backend_gives_nonzero_joint_minus_antibody_only_differences_on_the_right_arm():
    """Synthetic plumbing evidence: each arm's number lands on that arm. It says nothing about antigen specificity."""
    ln19 = math.log(19.0)
    rows = [drow("aa", "ACDA", value=6.0), drow("cc", "ACDC", value=7.0)]  # the one differing site is H3: A -> C
    plan, cells, pairs = execute(AntigenPullBackend(), rows, mask_counts=[1], geometries=["suffix"], local_gap=on())
    task_of = {task.heavy: task.task_id for task in plan.singleton_tasks}
    by_cell = {(cell["task_id"], cell["arm"]): cell for cell in cells}
    # (heavy, arm) -> (NLL of the gold residue at H3, greedy residue at H3, greedy residue correct?)
    expected = {
        ("ACDC", "joint"): (math.log(2.0), "C", True),  # gold C: p = 1/2 with the antigen
        ("ACDC", "antibody_only"): (LN20, "A", False),  # ... but 1/20 without it, and the tie goes to column A
        ("ACDA", "joint"): (math.log(38.0), "C", False),  # gold A: p = 1/38 with the antigen
        ("ACDA", "antibody_only"): (LN20, "A", True),
    }
    assert len(cells) == len(expected)
    for (heavy, arm), (nll, greedy, right) in expected.items():
        cell = by_cell[(task_of[heavy], arm)]
        assert cell["nll"]["sum"] == near(nll) and cell["nll"]["site"] == near([nll])
        assert cell["sequential"]["forward"]["sum"] == near(-nll)
        completion = cell["completion"]
        assert completion["residues"] == greedy and (completion["correct"], completion["exact"]) == (int(right), right)
    gold_c = [cell for cell in cells if cell["task_id"] == task_of["ACDC"]]
    summary = dg.summarize_singletons(gold_c)
    groups = {group["arm"]: group for group in summary["groups"]}
    assert groups["joint"]["nll"]["token_weighted_mean"] == near(math.log(2.0))
    assert groups["antibody_only"]["nll"]["token_weighted_mean"] == near(LN20)
    assert groups["joint"]["recovery"]["exact_fraction"] == 1.0
    assert groups["antibody_only"]["recovery"]["exact_fraction"] == 0.0
    (paired,) = summary["paired_joint_minus_antibody_only"]
    assert paired["nll_difference"]["token_weighted_mean"] == near(-math.log(10.0))  # ln 2 - ln 20, joint minus alone
    assert paired["nll_difference"]["macro_case_mean"] == near(-math.log(10.0))
    assert paired["sequential_forward_difference"]["token_weighted_mean"] == near(math.log(10.0))
    (both,) = dg.summarize_singletons(cells)["paired_joint_minus_antibody_only"]
    assert both["nll_difference"]["token_weighted_mean"] == near((math.log(38.0) + math.log(2.0) - 2 * LN20) / 2)

    (result,) = pairs  # endpoints A -> C at H3: oriented by sorted content, so A is "ACDA" and B is "ACDC"
    assert result["distance"] == 1 and result["changed"] == ["H:3"] and result["reverse_order"] is None
    joint, alone = result["arms"]["joint"]["forward"], result["arms"]["antibody_only"]["forward"]
    assert joint["gap"] == near(ln19) and alone["gap"] == near(0.0)  # a SUM over the one changed site
    assert (joint["j_a"], joint["j_b"]) == near((-math.log(38.0), -math.log(2.0)))
    assert (alone["j_a"], alone["j_b"]) == near((-LN20, -LN20))
    assert all(result["arms"][arm]["order_gap_difference"] is None for arm in dg.ARMS)

    report = dg.summarize_pairs(pairs, plan.pair_tasks, world.panel_of(rows).by_id(), plan.kappa)
    by_arm = {(group["arm"], group["order"]): group for group in report["groups"]}
    assert sorted(by_arm) == [("antibody_only", "forward"), ("joint", "forward")]
    joint, alone = by_arm[("joint", "forward")], by_arm[("antibody_only", "forward")]
    assert joint["mean_gap"] == near(ln19) and alone["mean_gap"] == near(0.0)
    assert joint["mean_delta_pkd"] == near(1.0) and alone["mean_delta_pkd"] == near(1.0)  # kappa is 1, so target = delta
    assert joint["mean_residual"] == near(ln19 - 1.0) and alone["mean_residual"] == near(-1.0)
    assert joint["sign_agreement"] == {"n_nonzero": 1, "n_agree": 1}
    assert alone["sign_agreement"] == {"n_nonzero": 0, "n_agree": 0}  # a gap of exactly zero has no sign
    (difference,) = report["paired_joint_minus_antibody_only"]
    assert difference["order"] == "forward" and difference["n_contrasts"] == 1
    assert difference["mean_gap_difference"] == near(ln19)
    assert difference["mean_abs_residual_difference"] == near((ln19 - 1.0) - 1.0)
    (contrast,) = report["contrasts"]
    assert contrast["arms"]["joint"]["forward"]["gap"] == near(ln19)
    assert contrast["arms"]["antibody_only"]["forward"]["gap"] == near(0.0)
    assert contrast["arms"]["joint"]["forward"]["residual"] == near(ln19 - 1.0)
    assert contrast["arms"]["antibody_only"]["forward"]["residual"] == near(-1.0)
    assert report["order_reversal"] == []  # one changed site: there is no reverse order to compare


# --- aggregation ------------------------------------------------------------------------------------------------


def _cell(task_id, count, nll_sum, *, geometry="full", arm="joint", mask_ratio=0.5):
    return {
        "task_id": task_id, "pair_fingerprint": task_id, "partition": "development", "count": count,
        "geometry": geometry, "arm": arm, "coords": [],
        "nll": {"sum": nll_sum, "token_mean": nll_sum / count, "site": [], "canonical_mass": [0.9] * count, "mask_ratio": mask_ratio},
        "sequential": {"forward": {"sum": -nll_sum, "mean": -nll_sum / count}, "reverse": None},
        "order_sensitivity": None,
        "completion": {"correct": 0, "total": count, "exact": False},
    }  # fmt: skip


def test_macro_and_token_weighted_means_are_reported_separately_per_group_and_never_pooled():
    cells = [
        _cell("t1", 1, 3.0), _cell("t2", 3, 3.0),  # full masks of different region sizes
        _cell("t1", 1, 5.0, geometry="suffix"), _cell("t1", 1, 7.0, geometry="subset"),
        _cell("t1", 1, 2.0, arm="antibody_only"), _cell("t2", 3, 3.0, arm="antibody_only"),
    ]  # fmt: skip
    summary = dg.summarize_singletons(cells)
    groups = {(g["count_label"], g["geometry"], g["arm"]): g for g in summary["groups"]}
    full = groups[("full", "full", "joint")]
    assert full["mask_sizes"] == [1, 3] and full["n_views"] == 2 and full["n_distinct_content_tasks"] == 2
    assert full["nll"]["macro_case_mean"] == pytest.approx(2.0)  # mean of 3.0 and 1.0
    assert full["nll"]["token_weighted_mean"] == pytest.approx(1.5)  # 6.0 over 4 masked sites
    assert groups[(1, "suffix", "joint")]["nll"]["macro_case_mean"] == pytest.approx(5.0)
    assert groups[(1, "subset", "joint")]["nll"]["token_weighted_mean"] == pytest.approx(7.0)
    assert len(summary["groups"]) == 4 and all(g["count_label"] is not None for g in summary["groups"])  # no pooled row
    (paired,) = summary["paired_joint_minus_antibody_only"]
    assert paired["count_label"] == "full" and paired["n_views"] == 2
    assert paired["nll_difference"]["token_weighted_mean"] == pytest.approx((3.0 - 2.0 + 3.0 - 3.0) / 4)
    assert paired["nll_difference"]["macro_case_mean"] == pytest.approx(((3.0 - 2.0) + (1.0 - 1.0)) / 2)
    assert "independent" in summary["independence_note"]


def test_repeated_masks_of_one_molecule_are_one_content_task_in_a_group():
    cells = [_cell("t1", 2, 4.0, geometry="suffix"), _cell("t1", 2, 6.0, geometry="suffix")]
    (group,) = dg.summarize_singletons(cells)["groups"]
    assert group["n_views"] == 2 and group["n_distinct_content_tasks"] == 1 and group["n_distinct_sequence_pairs"] == 1


def test_pair_summary_reports_the_gap_sum_target_and_residual_with_separate_pair_and_contrast_counts():
    logits = {("H", 2): [0.0] * 20, ("H", 3): [0.0] * 20}
    logits[("H", 2)][CANONICAL_RESIDUES.index("F")] = 2.0
    logits[("H", 3)][CANONICAL_RESIDUES.index("E")] = 1.0
    logits[("H", 3)][CANONICAL_RESIDUES.index("G")] = 3.0
    rows = [
        drow("a1", "ACDE", assay="a1", value=6.0), drow("b1", "ACFG", assay="a1", value=7.0),
        drow("a2", "ACDE", assay="a2", value=8.0), drow("b2", "ACFG", assay="a2", value=7.0),
    ]  # fmt: skip
    config = make_config(mask_counts=[1], local_gap=on(kappa=2.0))
    plan = plan_for(rows, config)
    counting = dg.CountingBackend(FixedLogitsBackend(logits), {dg.PHASE_DIAGNOSTICS: plan.budget.pair})
    with counting.phase(dg.PHASE_DIAGNOSTICS):
        results = dg.run_pair_tasks(counting, plan.pair_tasks)
    counting.verify()
    summary = dg.summarize_pairs(results, plan.pair_tasks, world.panel_of(rows).by_id(), plan.kappa)
    assert summary["unique_pair_tasks"] == 1 and summary["measurement_contrasts"] == 2
    # additive logits: gap = (2.0 - 0.0) + (3.0 - 1.0) = 4.0, a SUM over two changed sites (not 2.0 per site)
    by_assay = {(g["assay"], g["arm"], g["order"]): g for g in summary["groups"]}
    assert len(by_assay) == 2 * 2 * 2
    first, second = by_assay[("a1", "joint", "forward")], by_assay[("a2", "joint", "reverse")]
    assert (first["n_pairs"], first["n_contrasts"], first["mean_gap"]) == (1, 1, pytest.approx(4.0))
    assert first["mean_delta_pkd"] == pytest.approx(1.0) and first["mean_target"] == pytest.approx(2.0)  # kappa * (yB - yA)
    assert first["mean_residual"] == pytest.approx(2.0) and first["mean_abs_residual"] == pytest.approx(2.0)
    assert second["mean_delta_pkd"] == pytest.approx(-1.0) and second["mean_target"] == pytest.approx(-2.0)
    assert second["mean_residual"] == pytest.approx(6.0)
    assert first["sign_agreement"] == {"n_nonzero": 1, "n_agree": 1}
    assert all(g["distance"] == 2 and g["source"] == "s1" for g in summary["groups"])
    assert [row["arms"]["joint"]["forward"]["gap"] for row in summary["contrasts"]] == [pytest.approx(4.0)] * 2
    for entry in summary["order_reversal"]:
        assert entry["mean_forward_minus_reverse"] == pytest.approx(0.0, abs=1e-6)
    assert all(e["mean_gap_difference"] == pytest.approx(0.0, abs=1e-6) for e in summary["paired_joint_minus_antibody_only"])
    assert summary["antigen_specificity_of_joint_gap"] == "unestablished" and "epistasis" in summary["note"]
