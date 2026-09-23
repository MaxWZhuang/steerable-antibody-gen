"""Opt-in native checks against the pinned weights and the real dataset.

These are **explicitly opt-in**: they need the raw HER2 release, the pinned
p-IgGen weights and the historical campaign roots, none of which are in the
repository. Enable with::

    .venv/Scripts/python.exe -m pytest src/smallAntibodyGen/tests/test_her2_nf_native.py -m her2_nf_native -q

The environment variable HER2_NF_NATIVE=1 remains an alternative opt-in.

What they establish that the CPU suite cannot: the probability contract holds on
the *real* weights, the historical streams re-derive from the recorded pairing
seed, and the miniature end-to-end orchestration path -- resume, a deliberate
gate stop, a wrong-identity refusal, a corrupt-state refusal and the forbidden-row
guard -- runs against the production code rather than a synthetic seam.

The native gradient tolerance is NOT the CPU one. Float32 kernels differ by
reduction order; the measured maximum on this architecture is about 2e-5, and
that number is a measurement recorded here, never a reason to loosen the float64
identity asserted in ``test_her2_nf_objectives.py``.
"""
from __future__ import annotations

import os

import pytest

from smallAntibodyGen.experiments import her2_nf_campaign as campaign
from smallAntibodyGen.experiments import her2_nf_spec as spec
from smallAntibodyGen.experiments import her2_support_paths as paths

pytestmark = pytest.mark.her2_nf_native


@pytest.fixture(scope="module", autouse=True)
def native_opt_in(request):
    if (os.environ.get("HER2_NF_NATIVE") != "1"
            and request.config.getoption("markexpr") != "her2_nf_native"):
        pytest.skip("native fixtures require -m her2_nf_native or HER2_NF_NATIVE=1")

CONFIG_PATH = "configs/experiments/her2_next_flight.json"

#: Measured native tolerances. Separate from, and looser than, the CPU float64
#: identity; both are declared so neither can be quietly substituted.
NATIVE_GRADIENT_ATOL = 1e-4
NATIVE_GRADIENT_RTOL = 1e-3


@pytest.fixture(scope="module")
def flight(tmp_path_factory):
    import smallAntibodyGen
    root = paths.Path(smallAntibodyGen.__file__).resolve().parents[2]
    return spec.resolve_context(root, config_path=root / CONFIG_PATH,
                                run_root=tmp_path_factory.mktemp("nf_native"))


@pytest.fixture(scope="module")
def services(flight):
    device = "cuda" if _cuda() else "cpu"
    return campaign.build_services(flight, device=device)


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:                                         # pragma: no cover
        return False


def test_the_actual_configured_paths_resolve_on_the_real_tree(flight, services):
    """The preflight failure a synthetic seam cannot produce: a wrong raw root.

    ``FlightServices.policy`` and ``.vocab`` used to append ``piggen`` to a root
    the inherited loaders already append it to, so every native call asked for
    ``.../piggen/piggen/config.json``. Nothing in the CPU suite touches that path.
    """
    raw = services.raw_root()
    assert (raw / "piggen" / "config.json").is_file(), raw
    assert (raw / "piggen" / "tokenizer.json").is_file()
    vocab = services.vocab()
    assert len(vocab) > 20
    scaffold = services.scaffold()
    assert len(scaffold.prefix) == 99 and scaffold.prefix.endswith("YYCSR")
    for seed in flight.config["block_a"]["parent_seeds"]:
        checkpoint, entry = services.parent_checkpoint(seed)
        assert checkpoint.is_file(), checkpoint
        assert paths.sha256_file(checkpoint)
        assert entry["state_sha256"]
    assert (services.historical_root() / "banks_manifest.json").is_file()


def test_the_probability_contract_holds_on_the_pinned_weights(services):
    block = services.probability_contract()
    assert block["sampler_versus_scorer"]["rows_above_atol"] == 0
    assert block["train_versus_eval_max_abs"] == 0.0, "zero hidden/attention dropout"
    assert block["support"]["size"] == 20 and block["support"]["core_length"] == 10
    assert block["parent"]["observed_state_sha256"] == block["parent"]["expected_state_sha256"]
    # The modelled event is the ten editable residues under a 99-token prompt of
    # start sentinel + VH[:98]: no light chain, no right Tyr, no EOS.
    assert block["prompt"]["matches"] is True
    assert block["prompt"]["observed"]["prompt_tokens"] == 99
    assert block["prompt"]["observed"]["light_chain_residues"] == 0
    assert block["prompt"]["observed"]["right_anchor_included"] is False
    assert "sequence_log_probs called once in train mode" in block["train_versus_eval_basis"]


def test_the_historical_streams_are_loaded_and_also_re_derive(services):
    block = services.verify_historical_streams()
    for seed, entry in block["per_seed"].items():
        if not entry["available"]:
            pytest.skip(f"the historical stream record for seed {seed} is not reachable")
        assert entry["loaded"] is True, seed
        assert entry["replay_order_reused"] is True, seed
        # The SAVED replay order is what the loop consumes, and it matches the
        # digest the completed campaign recorded for it.
        assert entry["replay_order_sha256"] == entry["replay_order_recorded"], seed
        for name, comparison in entry["comparison"].items():
            assert comparison["matches"], (seed, name, comparison)
    assert block["all_match"] is True


def test_both_historical_kernels_reproduce_their_losses_and_monitor_vectors(services):
    """IPO0 AND IPO+FKL10: agreeing on one says nothing about the replay term."""
    block = services.verify_update_kernel_parity()
    if not block["available"]:
        pytest.skip(str(block))
    assert set(block["arms"]) == {"ipo_lambda0", "ipo_lambda10"}
    for name, arm in block["arms"].items():
        assert arm["available"], name
        assert arm["passed"], (name, arm["rows"][:3])
        assert arm["updates"] >= 1
        # The saved artefacts, not re-derived ones.
        assert arm["sources"]["stream"] == "saved", name
        assert arm["sources"]["reference_cache"] == "saved", name
        if arm["preservation"] == "fkl":
            assert arm["sources"]["replay_bank"] == "saved", name
        if arm["monitor_vector"] is not None:
            assert arm["monitor_vector"]["passed"], (name, arm["monitor_vector"])
    assert block["passed"] is True
    assert "not evidence of bitwise identity" in block["measurement_note"]


def test_the_validation_gate_population_is_the_inherited_pair_set(services):
    pairs = services.validation_pairs()
    assert int(pairs["pairs"]) == 25_722
    assert pairs["chosen_index"].shape[0] == pairs["rejected_index"].shape[0] == 25_722
    assert pairs["chosen_index"].shape[1] == 10


def test_the_saved_banks_and_reference_cache_load_with_their_declared_shapes(services,
                                                                             flight):
    from smallAntibodyGen.experiments import her2_nf_reuse as reuse
    from smallAntibodyGen.experiments import her2_preferences as preferences
    seed = int(flight.config["block_a"]["parent_seeds"][0])
    replay = reuse.load_replay_bank(services.historical_root(), seed, role="replay",
                                    expected_rows=100_000)
    assert replay["index"].shape == (100_000, 10)
    assert replay["teacher_probabilities_array"].shape == (100_000, 10, 20)
    monitor = reuse.load_replay_bank(services.historical_root(), seed, role="monitor",
                                     expected_rows=10_000)
    assert monitor["index"].shape == (10_000, 10)
    population = preferences.build_population(services.split("train"), "train")
    cache = reuse.load_reference_cache(services.historical_root(), seed,
                                       population=population,
                                       scaffold_prefix=services.scaffold().prefix)
    assert cache["rows"] == 234_007
    assert cache["chosen"].size == 120_477 and cache["rejected"].size == 113_530
    assert cache["source"] == "saved"


def test_a_historical_endpoint_loads_under_its_own_schema(services, flight):
    from smallAntibodyGen.experiments import her2_nf_reuse as reuse
    seed = int(flight.config["block_a"]["parent_seeds"][0])
    target = (services.historical_root() / "trajectories" / f"ipo_lambda0_seed{seed}"
              / "endpoint_update1000.pt")
    if not target.is_file():
        pytest.skip(f"{target} is not reachable")
    metadata = reuse.checkpoint_metadata(target)
    assert metadata["schema_version"] == reuse.HISTORICAL_ENDPOINT_SCHEMA
    policy = services.policy(target)          # strict-loads, digest-checked
    assert policy.model is not None


def test_the_geometry_reproduces_the_declared_counts(services):
    result = services.build_geometry()
    assert result["feasibility"]["passed"], result["attempts"]
    assert result["certificate"]["zero_violations"] is True
    counts = result["feasibility"]["counts"]
    # The declared construction: T0 < T is the C-neighbourhood purge.
    assert counts["t0_rows"] < counts["train_rows"]
    assert counts["purge_rows"] <= counts["t0_rows"]
    assert counts["match_rows"] == counts["purge_rows"]
    assert counts["retained_high_rows"] >= 50_000
    # The independently measured native geometry.
    assert counts["t0_rows"] == 234_890
    assert counts["purge_rows"] == 190_951
    assert counts["retained_high_rows"] == 61_019


def test_the_miniature_full_workflow_passes_with_numbers_in_it(services, flight):
    """Train, interrupt, resume, refuse, score, decompose, screen and report."""
    result = services.smoke(namespace="smoke_native")
    assert result["passed"], result["failures"]
    names = {entry["check"] for entry in result["checks"]}
    assert {"end_to_end_completes", "intermediate_checkpoint_written",
            "every_declared_endpoint_published", "interruption_actually_fired",
            "pending_endpoint_recorded", "resume_reproduces_uninterrupted_weights",
            "interrupted_endpoint_was_published",
            "resume_reconciled_the_pending_endpoint",
            "resumed_optimizer_matches", "resumed_scheduler_matches", "resumed_rng_matches",
            "whole_payload_digest_present", "truncated_journal_repaired_before_append",
            "wrong_identity_refused", "corrupt_weights_refused",
            "corrupt_optimizer_half_refused", "deliberate_gate_stop",
            "evaluation_rows_refused_in_training", "row_scores_are_numeric_and_finite",
            "class_mass_is_exact_and_positive", "coupling_statistics_are_finite",
            "decomposition_identity_holds", "drift_satisfies_B_le_T",
            "comparator_fit_selects_a_regularization",
            "numeric_report_block_is_nonempty"} <= names
    for field in ("updates", "checks", "exposures", "stream_position", "replay_position"):
        entry = next(block for block in result["checks"]
                     if block["check"] == f"resume_matches_uninterrupted_{field}")
        assert entry["passed"], entry
    analysis = result["analysis"]
    assert analysis["ranking"]["average_precision"] is not None
    assert analysis["class_mass"]["_panel"]["mass"] > 0
    assert len(analysis["mixture_curve"]["points"]) == 9
    assert analysis["coupling"]["total_correlation"]["total_correlation"] >= 0.0


def test_the_smoke_namespace_is_never_erased_to_make_a_run_pass(services, flight):
    """A namespace that already holds results is a collision, not a thing to delete."""
    directory = flight.path("smoke_occupied")
    directory.mkdir(parents=True, exist_ok=True)
    paths.write_json(directory / "someone_elses_result.json", {"keep": True})
    with pytest.raises(ValueError, match="never erases an existing run"):
        services.smoke(namespace="smoke_occupied")
    assert (directory / "someone_elses_result.json").is_file()


def test_the_profile_measures_every_cost_category(services):
    profile = services.profile()
    assert profile.missing() == []
    for category, block in profile.document()["measurements"].items():
        assert block["measured"] is True and block["seconds_per_unit"] >= 0, category
    sentinel = profile.measurements["sentinel"]["detail"]
    assert sentinel["sides"] == 2, "both pair halves, not just the chosen one"
    assert sentinel["parent_draws"] == 1024
    scoring = profile.measurements["score_50k_pass"]
    assert scoring["units"] > 1, "measured per ROW; the 50k total is a forecast quantity"
    comparator = profile.measurements["comparator_fit"]["detail"]
    assert comparator["cnn_seconds"] > 0 and comparator["pairwise_seconds"] > 0
    assert comparator["probes"]["pairwise"]["features"] == 190 + 16245
    stage_one = profile.measurements["stage1_fit"]["detail"]
    assert stage_one["effective_batch"] == 128
    assert stage_one["rows_per_epoch"] == 61_019, "the ACTUAL high rows of the purged population"
    assert stage_one["steps_per_fit"] == 5 * 477
    assert set(profile.measurements["optimizer_update"]["detail"]["seconds_per_update_by_family"]) == {
        "ipo_none", "ipo_fkl", "ipo_tail", "dpo_none", "dpo_fkl", "dpo_tail"}
    generation = profile.measurements["generation_draw"]["detail"]
    assert generation["retained_shape"][1:] == [10, 20]


def test_the_runtime_closure_check_really_runs_for_the_production_entry_point(flight):
    """Exercised in an isolated subprocess, where the imported set is production's.

    The pytest process legitimately imports test modules, so the check reports
    itself skipped there. This is where it is not skipped.
    """
    import subprocess
    import sys
    from smallAntibodyGen.experiments import her2_nf_spec as spec_lib
    spec_lib.source_snapshot(flight)
    program = (
        "import sys, json;"
        f"sys.path.insert(0, {str(flight.repository_root / 'src')!r});"
        "from smallAntibodyGen.experiments import her2_nf_spec as spec;"
        f"context = spec.resolve_context({str(flight.repository_root)!r},"
        f" config_path={str(flight.config_path)!r}, run_root={str(flight.run_root)!r});"
        "print(json.dumps(spec.verify_source_snapshot(context, label='subprocess')))")
    completed = subprocess.run([sys.executable, "-c", program], capture_output=True,
                               text=True, cwd=str(flight.repository_root), check=False)
    assert completed.returncode == 0, completed.stderr[-2000:]
    import json as json_lib
    block = json_lib.loads(completed.stdout.strip().splitlines()[-1])
    assert block["runtime_closure_checked"] is True
    assert block["non_production_imports"] == []


def test_no_unresolved_coefficient_can_enter_production(flight, services):
    """A frozen record that refused a family leaves its arms unrunnable."""
    from smallAntibodyGen.experiments import her2_nf_campaign as campaign_lib
    queue = campaign_lib.build_queue(
        flight.config, frozen={"dpo_beta": None, "dpo_fkl.lambda": None,
                               "ipo_tail.lambda": None, "dpo_tail.lambda": None},
        require_frozen=False)
    pending = [row for row in queue if row.status == "coefficients_pending"]
    assert pending, "the arms that cite a refused family are visible, not omitted"
    with pytest.raises(ValueError, match="never started at an invented coefficient"):
        campaign_lib.build_queue(flight.config, frozen={}, require_frozen=True)


def test_native_calibration_continuation_and_tail_trajectory(services, flight, monkeypatch):
    from smallAntibodyGen.experiments import her2_nf_calibration as calibration
    from smallAntibodyGen.experiments import her2_nf_trajectory as trajectory
    original = services.run_one_trajectory

    def miniature(**kwargs):
        return original(**kwargs, bank_rows=128, monitor_rows=64, namespace="native_calibration")

    monkeypatch.setattr(services, "run_one_trajectory", miniature)
    monkeypatch.setattr(services, "CALIBRATION_HORIZON", 4)
    seed = flight.config["block_a"]["parent_seeds"][0]
    base = calibration.CalibrationEntry("L1:native_dpo_fkl", "dpo_fkl", "dpo", "fkl",
                                         {"beta": 0.1, "lambda": 10.0}, 2)
    first = services._run_pilot(base, seed=seed, budget_seconds=1800)
    assert first["status"] == "completed", first
    extended = calibration.CalibrationEntry("EXT:" + base.entry_id, "dpo_fkl", "dpo", "fkl",
                                             dict(base.coefficients), 4)
    final = services._run_pilot(extended, seed=seed, resume_from=base.entry_id, budget_seconds=1800)
    assert final["status"] == "completed", final
    assert final["cost"]["updates_actually_run"] == 2
    directory = services._pilot_directory(base.entry_id)
    progress = trajectory.durable_progress(directory)
    assert progress["updates"] == 4 and progress["resume_boundaries"]
    assert first["cost"]["gpu_seconds"] > 0 and final["cost"]["gpu_seconds"] > 0
    tail = calibration.CalibrationEntry("L2:native_ipo_tail", "ipo_tail", "ipo", "tail",
                                         {"tau": 0.1, "lambda": 0.1}, 2)
    result = services._run_pilot(tail, seed=seed, budget_seconds=1800)
    assert result["status"] == "completed", result


def test_native_fresh_parent_interruption_and_high_c_gate(services, flight):
    import numpy as np
    from smallAntibodyGen.experiments import her2_nf_monitor as monitor
    from smallAntibodyGen.experiments import her2_nf_stage1 as stage1
    from smallAntibodyGen.experiments import her2_nf_storage as storage
    populations = services.load_populations()
    train_rows = populations["purge_rows"]
    high = (services.split("train").iloc[train_rows]["class"].to_numpy() == "high")
    index = services.index("train")[train_rows[high]][:256]
    c_rows = populations["calibration_rows"]
    c_index = services.index("val")[c_rows]
    labels = services.split("val").iloc[c_rows]["class"].to_numpy() == "high"
    plan = dict(flight.config["block_b"]["stage1"], epochs=1, checkpoints=[1], resume_interval_updates=1)
    directory = flight.path("native_fresh_parent")

    def fit(callback=None):
        policy = services.policy()
        try:
            return stage1.fit(policy, index=index, seed=20260918, directory=directory,
                              plan_config=plan, optimization=plan["optimization"],
                              selection_index=c_index, selection_labels=labels,
                              source_sha256=services._snapshot_digest(), after_update=callback)
        finally:
            del policy
            storage.collect_unused()

    def crash(update):
        if update == 1:
            raise RuntimeError("native stage1 interruption")

    with pytest.raises(RuntimeError, match="native stage1 interruption"):
        fit(crash)
    result = fit()
    assert result["selected_epoch"] == 1
    selected = result["checkpoints"][1]
    policy = services.policy(selected["path"])
    scores = policy.score(c_index[labels])["sum_log_probability"]
    gate = monitor.HighRowGate(row_index=c_index[labels], parent_log_probability=scores,
                               threshold=1.0, label="native high-C gate")
    observed = gate.evaluate(policy, update=1, reason="native")
    assert observed["passed"] and abs(observed["D"]) < 1e-6
    assert len(scores) == 958 and np.isfinite(scores).all()
    del policy
    services.release_transient()


def test_native_numeric_report_and_verification_roundtrip(services, flight):
    import numpy as np
    from types import SimpleNamespace
    from smallAntibodyGen.experiments import her2_nf_metrics as metrics
    from smallAntibodyGen.experiments import her2_nf_report as report
    analysis = paths.read_json(flight.path("smoke_native", "smoke_analysis.json"))
    with np.load(flight.path("smoke_native", "row_scores.npz")) as arrays:
        summed = arrays["sum_log_probability"].copy()
        positive = arrays["positive"].astype(bool)
    policy = services.policy(services.parent_checkpoint(20260918)[0])
    parent = policy.score(services.index("val")[:len(summed)])["sum_log_probability"]
    del policy
    services.release_transient()
    evidence = {
        "yield_curves": {"native_smoke": {"log_probabilities": summed[positive],
                                          "control_log_probabilities": parent[positive],
                                          "control_label": "native parent"}},
        "rankings": {"native_smoke": metrics.stratified_rank_block(
            summed / 10, positive, np.full(len(summed), "smoke"), categories=("smoke",))},
        "class_mass": {"native_smoke": analysis["class_mass"]},
        "coupling": [{"model": "native_smoke", "repeats": [{
            "rows": analysis["coupling"]["bank"]["rows"],
            "pairwise_mi_sum": analysis["coupling"]["pairwise_mi_sum"],
            "total_correlation": analysis["coupling"]["total_correlation"]}]}]}
    context = spec.resolve_context(flight.repository_root, config_path=flight.config_path,
                                    run_root=flight.path("native_report_roundtrip"))
    result = campaign.run_stage(context, "report", SimpleNamespace(report_evidence=lambda: evidence))
    assert result["status"] == "completed"
    document = paths.read_json(context.path("report.json"))
    assert document["yield"] and document["class_mass"]
    assert len(document["primary_contrasts"]) == 6
    assert not document["completion"]["complete"], "a native miniature is not a completed flight"
    verified = campaign.run_stage(context, "verify", SimpleNamespace())
    assert verified["checks"]["numeric_report_sections"]["passed"]
    assert not verified["passed"], "missing full-flight stages must remain visible"
