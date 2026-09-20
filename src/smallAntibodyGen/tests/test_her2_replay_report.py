"""Reporting: nothing absent is filled in, and no winner comes out of it.

Each test here is a way a report could look complete while being false: a stopped
trajectory that simply vanishes from a table, a delta formed against an endpoint
that was never reached, a Pareto claim that reads as superiority, or an
affirmative sentence about affinity that nobody meant to write.
"""
from __future__ import annotations

import numpy as np
import pytest

from smallAntibodyGen.experiments import her2_replay_report as report


def trajectory(name, *, task="ipo", lam=1.0, seed=20260918, status="completed", records=None,
               stop_reason=None):
    return {"trajectory": name, "arm_id": f"{task}_lambda{lam}", "task": task,
            "replay_lambda": lam, "seed": seed, "status": status, "stop_reason": stop_reason,
            "updates": 3750, "checks": 151, "exposures": {"chosen": 240000},
            "endpoint_records": records or {}}


def strata(ap):
    """Per-stratum metrics in the shape ``her2_eval.stratified_metrics`` produces."""
    return {"0": {"n": 40, "average_precision": ap + 0.05},
            "1": {"n": 30, "average_precision": ap + 0.01},
            "2": {"n": 20, "average_precision": ap},
            ">=3": {"n": 10, "average_precision": ap - 0.01}}


def endpoint(update, *, ap, kl, tenfold=0.02, eligible=True, passed=True, per_stratum=None):
    """One endpoint record, with the worst stratum DERIVED from the strata it carries.

    ``her2_replay_report.endpoint_evaluation`` takes the worst stratum as the
    minimum over the selection strata (1, 2, >=3) of the same per-stratum table it
    publishes. A fixture that supplied an unrelated ``ap - 0.02`` made the two
    fields disagree, and an assertion written against the strata then failed on the
    fixture's own arithmetic rather than on anything the report does.
    """
    per_stratum = strata(ap) if per_stratum is None else per_stratum
    worst = min(per_stratum[key]["average_precision"] for key in ("1", "2", ">=3"))
    return {"record_kind": "exposure_endpoint", "update": update,
            "macro_average_precision": ap, "worst_stratum_average_precision": worst,
            "val_strata": per_stratum,
            "gate_passed": passed, "diversity_eligible": eligible,
            "diversity_failed_gates": [] if eligible else ["unique_fraction"],
            "preservation": {"forward_kl_mean": kl, "tenfold_fraction": tenfold,
                             "hundredfold_fraction": tenfold / 10,
                             "conditional_kl_mean": kl * 1.01}}


def state(trajectories):
    return {"schema_version": "her2-parent-replay/1", "record_kind": "campaign_state",
            "trajectories": trajectories, "total": len(trajectories),
            "counts": {}, "generated_at": "2026-09-20T00:00:00+00:00"}


# ---------------------------------------------------------------------------
# rows: absent is absent
# ---------------------------------------------------------------------------

def test_a_stopped_trajectory_has_a_row_that_says_so():
    rows = report.endpoint_rows(
        state([trajectory("t", status="stopped_by_gate",
                          stop_reason="parent_relative_likelihood_breach")]),
        endpoint_updates=[1000, 2000], batch_rows=64)
    assert len(rows) == 2
    assert all(row["reached"] is False for row in rows)
    assert all("stopped_by_gate" in row["missing_reason"] for row in rows)
    assert all("none is substituted" in row["missing_reason"] for row in rows)
    assert rows[0]["chosen_exposures"] == 64000


def test_every_declared_endpoint_of_every_trajectory_gets_a_row():
    rows = report.endpoint_rows(
        state([trajectory("a"), trajectory("b", lam=0.0)]),
        endpoint_updates=[1000, 2000, 3750], batch_rows=64)
    assert len(rows) == 6


def test_a_reached_endpoint_carries_its_measured_numbers():
    rows = report.endpoint_rows(
        state([trajectory("a", records={"1000": endpoint(1000, ap=0.88, kl=0.31)})]),
        endpoint_updates=[1000], batch_rows=64)
    assert rows[0]["reached"] and rows[0]["macro_average_precision"] == 0.88
    assert rows[0]["forward_kl"] == 0.31 and rows[0]["tenfold_fraction"] == 0.02
    assert rows[0]["conditional_kl"] == pytest.approx(0.3131)


# ---------------------------------------------------------------------------
# matched deltas
# ---------------------------------------------------------------------------

def matched_rows():
    control = trajectory("ipo_lambda0_seed1", lam=0.0, seed=1,
                         records={"1000": endpoint(1000, ap=0.85, kl=0.60)})
    treated = trajectory("ipo_lambda1_seed1", lam=1.0, seed=1,
                         records={"1000": endpoint(1000, ap=0.86, kl=0.30)})
    return report.endpoint_rows(state([control, treated]), endpoint_updates=[1000],
                                batch_rows=64)


def test_a_matched_delta_is_formed_only_when_both_sides_reached_the_same_endpoint():
    deltas = report.matched_control_deltas(matched_rows())
    assert len(deltas["matched"]) == 1
    row = deltas["matched"][0]
    assert row["delta_macro_average_precision"] == pytest.approx(0.01)
    assert row["delta_forward_kl"] == pytest.approx(-0.30)
    assert row["control_trajectory"] == "ipo_lambda0_seed1"
    assert deltas["matched_on"] == "same task, same parent seed, same exact exposure endpoint"


def test_a_matched_delta_carries_every_stratum_not_only_the_macro_average():
    """A macro average can improve while one training-distance stratum falls."""
    control = trajectory("ipo_lambda0_seed1", lam=0.0, seed=1,
                         records={"1000": endpoint(1000, ap=0.85, kl=0.60)})
    treated = trajectory("ipo_lambda1_seed1", lam=1.0, seed=1,
                         records={"1000": endpoint(
                             1000, ap=0.86, kl=0.30,
                             per_stratum={"0": {"n": 40, "average_precision": 0.95},
                                          "1": {"n": 30, "average_precision": 0.90},
                                          "2": {"n": 20, "average_precision": 0.86},
                                          ">=3": {"n": 10, "average_precision": 0.80}})})
    rows = report.endpoint_rows(state([control, treated]), endpoint_updates=[1000], batch_rows=64)
    deltas = report.matched_control_deltas(rows)
    row = deltas["matched"][0]
    assert deltas["strata"] == ["1", "2", ">=3"]
    assert row["delta_stratum_average_precision"][">=3"] == pytest.approx(0.80 - 0.84)
    assert row["delta_stratum_average_precision"]["1"] == pytest.approx(0.90 - 0.86)
    assert row["delta_stratum_average_precision"][">=3"] < 0 < \
        row["delta_macro_average_precision"], (
        "this is the case the per-stratum columns exist for")
    # The worst stratum is >=3 on both sides: 0.80 treated against 0.84 control.
    assert row["worst_stratum_average_precision"] == pytest.approx(0.80)
    assert row["control_worst_stratum_average_precision"] == pytest.approx(0.84)
    assert row["delta_worst_stratum_average_precision"] == pytest.approx(0.80 - 0.84)


def test_an_empty_stratum_is_unavailable_rather_than_a_zero():
    row = {"val_strata": {"1": {"n": 0, "note": "empty_stratum"},
                          "2": {"n": 5, "average_precision": 0.4}}}
    assert report.stratum_average_precision(row, "1") is None
    assert report.stratum_average_precision(row, "2") == 0.4
    assert report.stratum_average_precision(row, ">=3") is None


def test_no_delta_is_formed_without_a_matched_control():
    treated = trajectory("ipo_lambda1_seed1", lam=1.0, seed=1,
                         records={"1000": endpoint(1000, ap=0.86, kl=0.30)})
    rows = report.endpoint_rows(state([treated]), endpoint_updates=[1000], batch_rows=64)
    deltas = report.matched_control_deltas(rows)
    assert deltas["matched"] == []
    assert deltas["unavailable"][0]["control_reached"] is False
    assert "no matched zero-replay endpoint" in deltas["unavailable"][0]["reason"]
    assert "no earlier endpoint" in deltas["substitution_policy"]


def test_a_delta_never_crosses_a_seed_or_an_exposure_endpoint():
    control = trajectory("ipo_lambda0_seed2", lam=0.0, seed=2,
                         records={"1000": endpoint(1000, ap=0.85, kl=0.60)})
    treated = trajectory("ipo_lambda1_seed1", lam=1.0, seed=1,
                         records={"1000": endpoint(1000, ap=0.86, kl=0.30)})
    rows = report.endpoint_rows(state([control, treated]), endpoint_updates=[1000],
                                batch_rows=64)
    deltas = report.matched_control_deltas(rows)
    assert deltas["matched"] == [], "a control at another seed is not a matched control"


def test_eligibility_flags_travel_with_every_delta():
    control = trajectory("ipo_lambda0_seed1", lam=0.0, seed=1,
                         records={"1000": endpoint(1000, ap=0.85, kl=0.60)})
    treated = trajectory("ipo_lambda1_seed1", lam=1.0, seed=1,
                         records={"1000": endpoint(1000, ap=0.9, kl=0.30, eligible=False)})
    rows = report.endpoint_rows(state([control, treated]), endpoint_updates=[1000],
                                batch_rows=64)
    row = report.matched_control_deltas(rows)["matched"][0]
    assert row["both_diversity_eligible"] is False and row["both_gate_eligible"] is True


# ---------------------------------------------------------------------------
# Pareto: observed, not established
# ---------------------------------------------------------------------------

def test_the_pareto_front_is_within_a_seed_and_an_endpoint_and_is_labelled_descriptive():
    rows = report.endpoint_rows(state([
        trajectory("a", lam=0.0, seed=1, records={"1000": endpoint(1000, ap=0.80, kl=0.90)}),
        trajectory("b", lam=1.0, seed=1, records={"1000": endpoint(1000, ap=0.85, kl=0.40)}),
        trajectory("c", lam=10.0, seed=1, records={"1000": endpoint(1000, ap=0.70, kl=0.50)}),
        trajectory("d", lam=1.0, seed=2, records={"1000": endpoint(1000, ap=0.60, kl=0.10)}),
    ]), endpoint_updates=[1000], batch_rows=64)
    front = report.pareto_front(rows)
    names = {point["trajectory"] for point in front["points"]}
    assert names == {"b", "d"}, "a is dominated by b; c is dominated by b; d is another seed"
    assert "not a winner" in front["status"]
    assert front["grouped_within"] == "one parent seed and one exposure endpoint"


def test_an_unreached_point_is_never_on_the_front():
    rows = report.endpoint_rows(state([trajectory("a", status="incomplete")]),
                                endpoint_updates=[1000], batch_rows=64)
    assert report.pareto_front(rows)["points"] == []


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------

def test_coverage_counts_what_is_missing_and_says_it_is_missing_coverage():
    rows = report.endpoint_rows(state([
        trajectory("a", records={"1000": endpoint(1000, ap=0.8, kl=0.2)}),
        trajectory("b", status="stopped_by_gate", stop_reason="breach")]),
        endpoint_updates=[1000, 2000], batch_rows=64)
    coverage = report.coverage_summary(state([trajectory("a"), trajectory("b")]), rows)
    assert coverage["declared_endpoint_rows"] == 4 and coverage["reached_endpoint_rows"] == 1
    assert coverage["missing_endpoint_rows"] == 3 and coverage["complete"] is False
    assert "never becomes a preservation finding" in coverage["note"]


# ---------------------------------------------------------------------------
# the rendered narrative
# ---------------------------------------------------------------------------

CONFIG = {"protocol": "specs/her2_support_preservation_plan.md",
          "config_path": "configs/experiments/her2_parent_replay.json",
          "screen": {"tasks": ["continued_sft", "ipo"],
                     "replay_lambdas": [0.0, 0.01, 0.1, 1.0, 10.0, 100.0],
                     "parent_seeds": [20260918, 20260919, 20260920],
                     "chosen_per_update": 64, "endpoint_updates": [1000, 2000, 3750]}}


def render(trajectories, *, endpoint_updates=(1000,), costs=None):
    block = state(trajectories)
    rows = report.endpoint_rows(block, endpoint_updates=list(endpoint_updates), batch_rows=64)
    deltas = report.matched_control_deltas(rows)
    pareto = report.pareto_front(rows)
    coverage = report.coverage_summary(block, rows)
    return report.render_report(
        config=CONFIG, state=block, rows=rows, deltas=deltas, pareto=pareto, coverage=coverage,
        banks={"replay_rows": 100000, "monitor_rows": 10000, "overlap_summary": "seed 1: 12"},
        freeze={"commit": "abc123", "frozen_at": "2026-09-20T00:00:00+00:00",
                "audit_decision_outcome": "escalate",
                "audit_source_freeze_commit": "be302f8"},
        costs={"optimizer": 12.0, "gate": 30.0} if costs is None else costs, figures={})


def test_the_report_lists_every_trajectory_and_every_missing_endpoint():
    text = render([
        trajectory("ipo_lambda0_seed1", lam=0.0, seed=1,
                   records={"1000": endpoint(1000, ap=0.85, kl=0.6)}),
        trajectory("ipo_lambda1_seed1", lam=1.0, seed=1, status="stopped_by_gate",
                   stop_reason="parent_relative_likelihood_breach"),
        trajectory("ipo_lambda10_seed1", lam=10.0, seed=1, status="incomplete")])
    assert "ipo_lambda0_seed1" in text and "ipo_lambda1_seed1" in text
    assert "ipo_lambda10_seed1" in text
    assert "not reached" in text
    assert "never resumed" in text
    assert "No global winner" in text
    assert "untouched final test" in text


def test_the_report_states_the_maximum_exposure_arithmetic():
    """Three trajectories at the 3,750-update endpoint is 3 x 240,000 chosen examples."""
    text = render([trajectory("a"), trajectory("b"), trajectory("c")])
    assert "720,000" in text
    assert "early stops reduce the actual total" in text


def test_the_report_passes_its_own_claim_guard():
    assert report.require_no_forbidden_claim(render([trajectory("a")]))


@pytest.mark.parametrize("sentence", [
    "This independently confirms the selected arm.",
    "Replay improves affinity at lambda 10.",
    "lambda=1 is the global winner.",
    "This is confirmatory evidence for preservation."])
def test_an_affirmative_claim_this_screen_may_not_make_is_refused(sentence):
    with pytest.raises(ValueError, match="does not confirm affinity"):
        report.require_no_forbidden_claim(sentence)


def test_the_negated_forms_the_report_actually_uses_are_not_refused():
    assert report.require_no_forbidden_claim(
        "No global winner is produced. This is not an affinity measurement and it is not an "
        "independent confirmation of a previously selected arm.")


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def test_the_report_prints_every_stratum_and_its_matched_difference():
    text = render([
        trajectory("ipo_lambda0_seed1", lam=0.0, seed=1,
                   records={"1000": endpoint(1000, ap=0.85, kl=0.6)}),
        trajectory("ipo_lambda1_seed1", lam=1.0, seed=1,
                   records={"1000": endpoint(1000, ap=0.86, kl=0.3)})])
    assert "## Validation AP in each training-distance stratum" in text
    assert "AP d=1" in text and "AP d=2" in text and "AP d=>=3" in text
    assert "Δ AP d=>=3" in text
    # The macro average of the strata columns, printed beside them.
    assert "0.8600" in text and "0.8700" in text
    assert "Stratum 0 is reported by the" in text


def test_the_report_names_the_teacher_cache_cost_rather_than_folding_it_away():
    """The measured rows, and the sentence with its line wrapping normalized away.

    The narrative is wrapped to a column width, so ``Bank preparation`` spans a
    newline in the rendered Markdown. Asserting the phrase against the raw text was
    asserting the wrapping; what matters is that the teacher cache is a row of its
    own and is not folded into bank inference.
    """
    text = render([trajectory("ipo_lambda0_seed1", lam=0.0, seed=1)],
                  costs={"optimizer": 12.0, "gate": 30.0, "generation": 44.0,
                         "teacher_cache": 91.5, "io": 3.0})
    assert "| teacher_cache | 91.5 |" in text, "its own measured row, not part of generation"
    assert "| generation | 44.0 |" in text
    flowed = " ".join(text.split())
    assert "teacher-cache build is its own category" in flowed
    assert "Bank preparation is included above" in flowed


def test_the_tradeoff_figures_are_faceted_by_exposure_and_seed():
    """108 points in one frame with 4.5pt labels is not a figure anybody can read."""
    pytest.importorskip("matplotlib")
    import tempfile
    from pathlib import Path as _Path
    trajectories = []
    for seed in (1, 2):
        for lam in (0.0, 1.0):
            trajectories.append(trajectory(
                f"ipo_lambda{lam}_seed{seed}", lam=lam, seed=seed,
                records={"1000": endpoint(1000, ap=0.8 + lam / 100, kl=0.3 + lam / 10),
                         "2000": endpoint(2000, ap=0.82, kl=0.4, eligible=lam == 0.0)}))
    block = state(trajectories)
    rows = report.endpoint_rows(block, endpoint_updates=[1000, 2000], batch_rows=64)
    with tempfile.TemporaryDirectory() as directory:
        figures = report.render_figures(_Path(directory), rows)
        assert set(figures) == {"her2-parent-replay-forward-kl.png",
                                "her2-parent-replay-tenfold.png",
                                "her2-parent-replay-hundredfold.png"}
        entry = figures["her2-parent-replay-forward-kl.png"]
        assert entry["written"] and entry["panels"]["rows"] == "exposure endpoint"
        assert entry["panels"]["columns"] == "parent seed"
        assert entry["panels"]["points_per_panel"] == 2, (
            "one task and two lambdas in this fixture; the declared grid gives 2 x 6 = 12")
        assert "one panel per exposure endpoint and parent seed" in entry["caption"]
        assert "never plotted as zeros" in entry["caption"]


def test_a_panel_with_no_reached_endpoint_says_so_rather_than_being_empty():
    pytest.importorskip("matplotlib")
    import tempfile
    from pathlib import Path as _Path
    block = state([trajectory("ipo_lambda1_seed1", lam=1.0, seed=1, status="stopped_by_gate",
                              stop_reason="parent_relative_likelihood_breach")])
    rows = report.endpoint_rows(block, endpoint_updates=[1000], batch_rows=64)
    with tempfile.TemporaryDirectory() as directory:
        figures = report.render_figures(_Path(directory), rows)
        assert figures["her2-parent-replay-forward-kl.png"]["written"] is True


def test_endpoint_records_are_recovered_for_a_trajectory_that_is_still_running(tmp_path):
    """A reached endpoint must not disappear because no terminal status exists yet."""
    from smallAntibodyGen.experiments import her2_replay_campaign as campaign
    import json as _json
    directory = tmp_path / "trajectories" / "ipo_lambda1_seed1"
    directory.mkdir(parents=True)
    (directory / campaign.ENDPOINTS_JSONL).write_text(
        _json.dumps({"record_kind": "exposure_endpoint", "update": 1000,
                     "macro_average_precision": 0.9,
                     "checkpoint": {"sha256": "a" * 64}}) + "\n",
        encoding="utf-8", newline="\n")
    block = state([dict(trajectory("ipo_lambda1_seed1", lam=1.0, seed=1, status="running"),
                        endpoint_records=None)])
    enriched = report.attach_endpoint_records(tmp_path, block)
    row = enriched["trajectories"][0]
    assert sorted(row["endpoint_records"]) == ["1000"]
    assert row["endpoint_records_from"] == "the append-only endpoint journal"
    rows = report.endpoint_rows(enriched, endpoint_updates=[1000], batch_rows=64)
    assert rows[0]["reached"] is True and rows[0]["macro_average_precision"] == 0.9


def test_compact_lambda_labels_stay_short_enough_to_sit_beside_a_point():
    assert report.compact_lambda(0.0) == "λ0"
    assert report.compact_lambda(0.01) == "λ.01"
    assert report.compact_lambda(100.0) == "λ100"


def test_task_styles_are_stable_across_every_panel_and_every_figure():
    assert set(report.TASK_STYLES) == {"continued_sft", "ipo"}
    markers = {name: style["marker"] for name, style in report.TASK_STYLES.items()}
    assert len(set(markers.values())) == len(markers), "one marker per method"


def test_the_figure_writer_passes_an_explicit_format_on_the_comparison_path():
    """The legacy renderer inferred the format from a '.png.rerender' suffix and raised."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tempfile
    from pathlib import Path as _Path
    with tempfile.TemporaryDirectory() as directory:
        target = _Path(directory) / "figure.png"
        figure, axes = plt.subplots()
        axes.plot([0, 1], [0, 1])
        name, first = report.save_figure(figure, target)
        assert first["written"] and target.is_file()
        _, second = report.save_figure(figure, target)
        plt.close(figure)
        assert second["sha256"] == first["sha256"]
        assert not list(_Path(directory).glob("*.rerender*")), "the comparison file is removed"


def test_the_run_directory_figure_is_replaced_as_the_campaign_progresses():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tempfile
    from pathlib import Path as _Path
    with tempfile.TemporaryDirectory() as directory:
        target = _Path(directory) / "figure.png"
        first_figure, axes = plt.subplots()
        axes.plot([0, 1], [0, 1])
        report.save_figure(first_figure, target, replace=True)
        plt.close(first_figure)
        second_figure, axes = plt.subplots()
        axes.plot([0, 1], [1, 0])
        _, entry = report.save_figure(second_figure, target, replace=True)
        plt.close(second_figure)
        assert "still running" in entry["replaced"]


def test_endpoint_evaluation_uses_the_inherited_macro_average_statistic():
    strata = np.array(["1"] * 6 + ["2"] * 6 + [">=3"] * 6)
    scores = np.linspace(0, 1, 18)
    positives = np.array([1, 0] * 9, dtype=bool)
    cores = np.array([f"core{i:02d}" for i in range(18)])
    block = report.endpoint_evaluation(scored_mean_log_probability=scores, positives=positives,
                                       cores=cores, strata=strata,
                                       categories=("0", "1", "2", ">=3"))
    assert set(block["strata_used"]) == {"1", "2", ">=3"}
    assert 0.0 <= block["macro_average_precision"] <= 1.0
    assert block["inputs"] == ["validation"]
    assert "no reserved test label" in block["note"]
