"""Queue completeness, stage ordering, gates, the measured forecast, and launch health.

Orchestration is tested against a stub runtime. That is deliberate: the property
being checked is that a feasibility failure stops the *dependent* stages and
nothing else, that a required stage which did not run is visible as such, and
that no forecast can be written from costs nobody measured. None of that needs a
GPU, and all of it would be expensive to discover during a twenty-hour run.
"""
from __future__ import annotations

import json

import pytest

from smallAntibodyGen.experiments import her2_nf_campaign as campaign
from smallAntibodyGen.experiments import her2_nf_contract as contract
from smallAntibodyGen.experiments import her2_nf_spec as spec
from smallAntibodyGen.experiments import her2_support_paths as paths

CONFIG_PATH = "configs/experiments/her2_next_flight.json"


@pytest.fixture
def flight(tmp_path):
    """The real shipped configuration, pointed at a scratch run root.

    The declared 20 GiB floor is for the approved output volume; a scratch
    directory on the system disk would trip the readiness gate in every test, so
    it is lowered here and exercised on its own below.
    """
    import smallAntibodyGen
    root = paths.Path(smallAntibodyGen.__file__).resolve().parents[2]
    context = spec.resolve_context(root, config_path=root / CONFIG_PATH,
                                   run_root=tmp_path / "run")
    context.config["storage"]["min_free_bytes"] = 1
    return context


class StubServices:
    """Records which stage methods ran, and can be told to fail one of them."""

    def __init__(self, *, fail=None, geometry_passes=True, all_terminal=True):
        self.calls = []
        self.fail = fail
        self.geometry_passes = geometry_passes
        self.all_terminal = all_terminal

    def _note(self, name):
        self.calls.append(name)
        if self.fail == name:
            raise campaign.ReadinessGate(f"stubbed readiness failure in {name}")

    def probability_contract(self):
        self._note("probability_contract")
        return {"sampler_versus_scorer": {"max_absolute_error": 8.9e-6}}

    def verify_historical_streams(self):
        self._note("verify_historical_streams")
        return {"all_match": True, "per_seed": {}}

    def verify_update_kernel_parity(self):
        self._note("verify_update_kernel_parity")
        return {"passed": True, "max_absolute_difference": 1e-6}

    def build_geometry(self):
        self._note("build_geometry")
        feasibility = {"passed": bool(self.geometry_passes), "radius": 2,
                       "counts": {"purge_rows": 190951}, "checks": {}}
        return {"radius": 2, "panel_size": 1000, "feasibility": feasibility,
                "certificate": {"zero_violations": True, "record_kind": "neighbor_certificate"},
                "manifest": {"record_kind": "split_manifest"}}

    def mixture_rescore(self):
        self._note("mixture_rescore")
        return {"curves": [{"label": "a"}], "grid": [0.0, 1.0]}

    def profile(self):
        self._note("profile")
        profile = campaign.Profile()
        for category in campaign.COST_CATEGORIES:
            profile.record(category, seconds=1.0, units=10)
        return profile

    def calibrate(self):
        self._note("calibrate")
        from smallAntibodyGen.experiments import her2_nf_calibration as calibration
        return {"frozen": {"dpo_beta": 0.1, "dpo_fkl": {"coefficients": {"lambda": 10.0}},
                           "ipo_tail": {"coefficients": {"lambda": 0.1}},
                           "dpo_tail": {"coefficients": {"lambda": 0.1}}},
                "family_outcomes": {},
                "ledger": calibration.CalibrationLedger().document()}

    def run_production(self, *, max_trajectories=None, heartbeat=None):
        self._note("run_production")
        return {"started": 0, "remaining": [] if self.all_terminal else ["A_DPO_0_seed20260918"],
                "all_terminal": self.all_terminal, "any_terminal": True, "completed": 30,
                "stopped_by_gate": 0, "incomplete": 0, "failed": 0}

    def freeze_finalists(self):
        self._note("freeze_finalists")
        return {"named_checkpoints": [{"name": "IPO_FKL@u1000_seed20260918"}],
                "coupling_finalists": [{"name": "IPO_FKL@u1000_seed20260918"}],
                "tail_family": {"family": "ipo_tail", "feasible": True}}

    def run_audits(self):
        self._note("run_audits")
        return {"models": [], "banks": []}

    def run_coupling(self):
        self._note("run_coupling")
        return {"screens": [], "finalists": [], "banks": 0}

    def smoke(self, *, namespace=None):
        self._note("smoke")
        return {"passed": True, "checks": [], "failures": []}


# ---------------------------------------------------------------------------
# the queue
# ---------------------------------------------------------------------------

def test_the_queue_carries_every_declared_cell_including_the_optional_one(flight):
    queue = campaign.build_queue(flight.config, frozen={"dpo_beta": 0.1,
                                                        "dpo_fkl.lambda": 10.0,
                                                        "dpo_tail.lambda": 0.1,
                                                        "ipo_tail.lambda": 0.1})
    block_a = [row for row in queue if row.block == "A"]
    block_b = [row for row in queue if row.block == "B"]
    assert len(block_a) == 7 * 3, "seven declared Block-A arms at three seeds"
    assert len(block_b) == 2 * 3 * 3, "two regimes x three seeds x three arms"
    optional = [row for row in queue if row.optional]
    assert len(optional) == 3
    assert all(row.status == "deferred_optional" for row in optional)
    assert "listed so its absence is visible" in optional[0].note


def test_reused_historical_arms_are_not_factorial_cells_until_parity_passes(flight):
    frozen = {"dpo_beta": 0.1, "dpo_fkl.lambda": 10.0, "dpo_tail.lambda": 0.1,
              "ipo_tail.lambda": 0.1}
    pending = campaign.build_queue(flight.config, frozen=frozen, reuse_verified=False)
    verified = campaign.build_queue(flight.config, frozen=frozen, reuse_verified=True)
    pending_reuse = [row for row in pending if row.reuse]
    assert pending_reuse and all(row.status == "reuse_pending_parity" for row in pending_reuse)
    assert "not a factorial cell" in pending_reuse[0].note
    assert all(row.status == "reuse_verified" for row in verified if row.reuse)


def test_a_production_cell_is_never_started_at_an_invented_coefficient(flight):
    with pytest.raises(ValueError, match="never started at an invented coefficient"):
        campaign.build_queue(flight.config, frozen={})


def test_dpo_arms_declare_that_they_consume_a_rejected_row(flight):
    queue = campaign.build_queue(flight.config, frozen={"dpo_beta": 0.1,
                                                        "dpo_fkl.lambda": 10.0,
                                                        "dpo_tail.lambda": 0.1,
                                                        "ipo_tail.lambda": 0.1})
    dpo = [row for row in queue if row.task == "dpo"]
    assert dpo and all(row.document()["uses_rejected"] for row in dpo)


def test_comparators_are_declared_entries_not_extra_preference_cells():
    ids = {entry["id"] for entry in campaign.comparator_rows()}
    assert {"parent", "continued_sft_lambda0p1", "cnn_single", "cnn_ensemble",
            "additive_linear", "interaction_classifier", "mixture_alpha",
            "historical_ipo0", "historical_ipo_fkl"} <= ids
    cnn = next(entry for entry in campaign.comparator_rows() if entry["id"] == "cnn_single")
    assert "different training signal" in cnn["note"]


def test_campaign_state_gives_every_row_a_status(flight):
    frozen = {"dpo_beta": 0.1, "dpo_fkl.lambda": 10.0, "dpo_tail.lambda": 0.1,
              "ipo_tail.lambda": 0.1}
    queue = campaign.build_queue(flight.config, frozen=frozen)
    state = campaign.campaign_state(flight.run_root, queue)
    assert len(state["rows"]) == len(queue)
    assert sum(state["counts"].values()) == len(queue)
    assert all("observed_status" in row for row in state["rows"])
    assert "never an omission" in state["completeness_rule"]


# ---------------------------------------------------------------------------
# profiling and the forecast
# ---------------------------------------------------------------------------

def test_a_profile_entry_must_be_measured():
    profile = campaign.Profile()
    assert profile.missing() == list(campaign.COST_CATEGORIES)
    profile.record("optimizer_update", seconds=29.5, units=100)
    assert profile.measurements["optimizer_update"]["seconds_per_unit"] == pytest.approx(0.295)
    assert profile.measurements["optimizer_update"]["measured"] is True
    with pytest.raises(ValueError, match="Unknown cost category"):
        profile.record("wishful_thinking", seconds=1.0, units=1)
    with pytest.raises(ValueError, match="positive unit count"):
        profile.record("full_gate", seconds=1.0, units=0)


def test_a_forecast_refuses_an_incomplete_profile(flight):
    profile = campaign.Profile()
    profile.record("optimizer_update", seconds=1.0, units=10)
    queue = campaign.build_queue(flight.config, require_frozen=False)
    with pytest.raises(ValueError, match="never written in its place"):
        campaign.runtime_forecast(profile, queue=queue,
                                  banks_plan=campaign.banks_plan(flight.config))


def test_a_measured_forecast_reports_categories_and_the_declared_drop_order(flight):
    profile = campaign.Profile()
    for category in campaign.COST_CATEGORIES:
        profile.record(category, seconds=1.0, units=1000)
    plan = campaign.banks_plan(flight.config)
    verified = campaign.runtime_forecast(
        profile, queue=campaign.build_queue(flight.config, require_frozen=False,
                                            reuse_verified=True), banks_plan=plan)
    assert verified["measured"] is True
    assert set(campaign.COST_CATEGORIES) <= set(verified["seconds"])
    assert verified["seconds"]["calibration_reserved_cap"] == 3 * 3600
    assert verified["total_hours"] > 0
    # With the two historical arms reused, the declared new post-training work is
    # 12 x 3750 plus 18 x 1000 = 63,000 updates.
    assert verified["queue_updates"] == 12 * 3750 + 18 * 1000 == 63_000
    # Until parity establishes the reuse, those six paths are counted as work --
    # the conservative direction for a schedule.
    pending = campaign.runtime_forecast(
        profile, queue=campaign.build_queue(flight.config, require_frozen=False,
                                            reuse_verified=False), banks_plan=plan)
    assert pending["queue_updates"] == 18 * 3750 + 18 * 1000
    assert verified["drop_order"][0].startswith("context-averaged")
    assert "not permission to relabel" in verified["basis"]
    # The extrapolations live in the FORECAST, with their units named, not
    # inside the profile as a measurement with units=1.
    assert set(verified["extrapolations"]) == {"score_50k_pass", "stage1_fit", "comparator_fit"}
    assert verified["scored_rows"] > 0


def test_the_banks_plan_keeps_the_required_finalist_counts(flight):
    plan = campaign.banks_plan(flight.config)
    assert plan["finalist_models"] == 15 and plan["finalist_banks_per_model"] == 2
    assert plan["finalist_draws"] == 15 * 2 * 50_000 == 1_500_000
    assert plan["stage1_fits"] == 6, "six fresh challenge parents"
    assert plan["screen_models"] == 147
    assert plan["audit_models"] == 69
    # A comparator SET per challenge regime plus one for the original split.
    assert plan["comparator_sets"] == 3
    assert plan["validation_rows"] == 78_652
    assert plan["scored_rows"] >= plan["finalist_draws"]


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

def test_run_all_advances_through_every_stage_in_order(flight):
    """Real stages, stubbed runtime: the orchestration itself is what is under test."""
    services = StubServices()
    document = campaign.run_all(flight, services)
    assert list(document["stages"]) == list(campaign.STAGES)
    assert document["stages"] == {stage: ("failed" if stage == "verify" else "completed")
                                  for stage in campaign.STAGES}, \
        document["stages"]
    assert document["all_required_completed"] is False
    assert services.calls[0] == "probability_contract"
    assert services.calls.index("profile") < services.calls.index("calibrate")
    assert services.calls.index("calibrate") < services.calls.index("run_production")
    assert services.calls.index("run_production") < services.calls.index("run_audits")
    # The miniature smoke path runs before the long queue, not after it.
    assert services.calls.index("smoke") < services.calls.index("run_production")
    assert "different facts" in document["claim"]
    # The freeze really ran: a content-addressed snapshot exists and the protocol
    # has no unresolved fields left.
    snapshot = paths.read_json(flight.path("source_snapshot.json"))
    assert len(snapshot["snapshot_sha256"]) == 64
    assert paths.read_json(flight.path("resolved_protocol.json"))["unresolved"] == []
    # And the report and verification are on disk with every queued row present.
    report = paths.read_json(flight.path("report.json"))
    assert report["rows"] > 0
    # This seam supplies empty audits: strict verification must reject those
    # even though its orchestration stub claims the stages completed.
    assert paths.read_json(flight.path("verification.json"))["passed"] is False


def test_a_feasibility_gate_stops_only_the_dependent_stages(flight):
    services = StubServices(geometry_passes=False)
    document = campaign.run_all(flight, services)
    assert document["stages"]["geometry"] == "blocked_feasibility"
    assert document["stages"]["mixtures"] == "completed", "an independent stage still runs"
    assert document["stages"]["profile"] == "completed"
    assert document["stages"]["calibrate"] == "completed"
    # The freeze refuses to resolve a protocol over an uncertified geometry, and
    # everything downstream of it stays unrun rather than running on one.
    assert document["stages"]["freeze"] == "blocked_readiness"
    assert document["stages"]["production"] == "not_run"
    assert document["all_required_completed"] is False
    record = campaign.read_stage_record(flight.run_root, "geometry")
    assert "predeclared radius-1" in record["reason"]
    assert "independent stages still run" in record["consequence"]
    blocked = campaign.read_stage_record(flight.run_root, "production")
    assert "blocked by geometry" in blocked["reason"]


def test_a_readiness_gate_is_classified_separately_from_a_feasibility_gate(flight):
    services = StubServices(fail="probability_contract")
    record = campaign.run_stage(flight, "preflight", services)
    assert record["status"] == "blocked_readiness"
    assert "no scientific claim is made" in record["consequence"]


def test_every_required_stage_writes_a_record_even_when_it_did_not_run(flight):
    services = StubServices(geometry_passes=False)
    campaign.run_all(flight, services)
    for stage in campaign.REQUIRED_STAGES:
        assert campaign.read_stage_record(flight.run_root, stage) is not None, stage


def test_production_refuses_to_start_without_a_resolved_protocol(flight):
    services = StubServices()
    with pytest.raises(ValueError):
        campaign.stage_production(flight, services)


def test_a_failing_smoke_path_blocks_the_production_queue(flight):
    class Failing(StubServices):
        def smoke(self):
            self._note("smoke")
            return {"passed": False, "checks": [{"check": "resume", "passed": False}],
                    "failures": ["resume"]}

    services = Failing()
    document = campaign.run_all(flight, services)
    assert document["stages"]["production"] == "blocked_readiness"
    record = campaign.read_stage_record(flight.run_root, "production")
    assert "smoke path did not pass" in record["reason"]
    assert campaign.read_stage_record(flight.run_root, "smoke")["status"] == "failed"
    assert document["stages"]["audit"] == "not_run"


def test_a_completed_stage_is_reused_rather_than_rerun(flight):
    """Rerunning production would discard the queue it just spent twenty hours on."""
    services = StubServices()
    campaign.run_all(flight, services)
    before = list(services.calls)
    again = StubServices()
    campaign.run_all(flight, again)
    assert again.calls == [], "every stage was reused under an unchanged identity"
    assert "run_production" in before
    record = campaign.read_stage_record(flight.run_root, "production")
    assert record["identity"]["config_sha256"]
    reusable, reason = campaign.stage_is_reusable(flight, "production")
    assert reusable and "same source and configuration" in reason


def test_a_changed_configuration_makes_a_completed_stage_non_reusable(flight, tmp_path):
    services = StubServices()
    campaign.run_all(flight, services)
    assert campaign.stage_is_reusable(flight, "geometry")[0] is True
    moved = tmp_path / "changed_config.json"
    moved.write_text(json.dumps(dict(flight.config, campaign_id="other")), encoding="utf-8")
    other = spec.FlightContext(repository_root=flight.repository_root,
                               run_root=flight.run_root, config=flight.config,
                               config_path=moved)
    reusable, reason = campaign.stage_is_reusable(other, "geometry")
    assert reusable is False and "identity changed" in reason


def test_partial_production_still_lets_the_audit_and_report_run(flight):
    """An interrupted queue's finished endpoints are the only output it has."""
    services = StubServices(all_terminal=False)
    document = campaign.run_all(flight, services)
    assert document["stages"]["production"] == "partial"
    assert document["stages"]["audit"] == "completed"
    assert document["stages"]["report"] == "completed"
    assert document["all_required_completed"] is False
    report = paths.read_json(flight.path("report.json"))
    assert report["completion"]["complete"] is False
    assert report["completion"]["rows_without_terminal_status"]


def test_the_heartbeat_advances_with_the_stages(flight):
    beats = []

    class Beat:
        def beat(self, **fields):
            beats.append(fields)

    campaign.run_all(flight, StubServices(), heartbeat=Beat())
    stages = [entry.get("stage") for entry in beats if entry.get("phase") == "stage"]
    assert stages == list(campaign.STAGES)


def test_launch_health_counts_calibration_and_stage_one_work(flight):
    """A flight two hours into calibration has advanced, whatever the queue says."""
    before = campaign.launch_health(flight)
    assert before["work"]["calibration_updates"] == 0
    assert before["health"] == "not_running"
    directory = flight.path("calibration", "L1_dpo_fkl_beta0.1_lambda3")
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "updates.jsonl").open("w", encoding="utf-8") as stream:
        for update in range(1, 6):
            stream.write(json.dumps({"record_kind": "update", "update": update}) + "\n")
    parents = flight.path("parents", "purge_seed20260918")
    parents.mkdir(parents=True, exist_ok=True)
    paths.write_json(parents / "stage1_history.json", [{"epoch": 1}, {"epoch": 2}])
    after = campaign.launch_health(flight)
    assert after["work"]["calibration_updates"] == 5
    assert after["work"]["stage_one_epochs"] == 2
    assert after["journalled_updates_total"] == 7
    assert after["progress_observed"] is True


def test_insufficient_storage_is_a_readiness_gate_and_never_shrinks_the_banks(flight):
    flight.config["storage"]["min_free_bytes"] = 1 << 60      # a petabyte
    record = campaign.run_stage(flight, "preflight", StubServices())
    assert record["status"] == "blocked_readiness"
    assert "NOT reduced to fit" in record["reason"]
    # The declared bank counts are unchanged by the failure.
    assert campaign.banks_plan(flight.config)["finalist_draws"] == 1_500_000


def test_a_queue_built_before_calibration_marks_its_pending_coefficients(flight):
    queue = campaign.build_queue(flight.config, require_frozen=False)
    pending = [row for row in queue if row.status == "coefficients_pending"]
    assert pending, "the tail and DPO arms cannot be resolved before the freeze"
    assert "refused by production" in pending[0].note
    # And production still refuses to build one.
    with pytest.raises(ValueError, match="invented coefficient"):
        campaign.build_queue(flight.config, require_frozen=True)


# ---------------------------------------------------------------------------
# launch health
# ---------------------------------------------------------------------------

def test_launch_health_separates_running_from_progress_and_from_completion(flight):
    block = campaign.launch_health(flight)
    assert block["worker_running"] is False
    assert block["health"] == "not_running"
    assert "never inferred from a successful launch" in block["claim"]


def test_a_stage_record_round_trips(flight):
    campaign.write_stage_record(flight.run_root, "recover", {"status": "completed"})
    record = campaign.read_stage_record(flight.run_root, "recover")
    assert record["stage"] == "recover" and record["record_kind"] == "stage_record"
    summary = campaign.stage_summary(flight.run_root)
    assert summary["recover"]["ran"] is True
    assert summary["geometry"]["status"] == "not_run"


def test_the_reporting_queue_inherits_the_recorded_preflight_reuse_decision(flight):
    """Reproduced: production skipped the six verified-reuse rows without writing a
    terminal record, and the reporting queue -- built with the default
    ``reuse_verified=False`` -- left them ``reuse_pending_parity`` forever. The
    report could never call the flight ready and ``production_has_no_pending_rows``
    failed after a completed production run."""
    campaign.write_stage_record(flight.run_root, "preflight",
                                {"status": "completed", "reuse_verified": True})
    queue = campaign.queue_for_reporting(flight)
    reused = [row for row in queue if row.reuse]
    assert reused and all(row.status == "reuse_verified" for row in reused)
    state = campaign.campaign_state(flight.run_root, queue)
    assert not [row for row in state["rows"]
                if row["observed_status"] in campaign.NON_TERMINAL_STATUSES
                and row["reuse"]]


def test_the_reporting_queue_keeps_reuse_pending_until_parity_is_recorded(flight):
    campaign.write_stage_record(flight.run_root, "preflight",
                                {"status": "completed", "reuse_verified": False})
    queue = campaign.queue_for_reporting(flight)
    reused = [row for row in queue if row.reuse]
    assert reused and all(row.status == "reuse_pending_parity" for row in reused)
    # And with no preflight record at all, the conservative default still holds.
    campaign.stage_record_path(flight.run_root, "preflight").unlink()
    assert all(row.status == "reuse_pending_parity"
               for row in campaign.queue_for_reporting(flight) if row.reuse)


def test_partial_production_does_not_freeze_the_downstream_stages(flight):
    """Reproduced: ``run-all --max-trajectories`` completed audit, coupling and
    report over a partial queue. The next invocation advanced production, but the
    downstream stage identities were unchanged, so they were reused -- the newly
    produced checkpoints were never audited and never reached the report."""
    first = StubServices(all_terminal=False)
    campaign.run_all(flight, first)
    assert campaign.read_stage_record(flight.run_root, "production")["status"] == "partial"
    assert "run_audits" in first.calls, "partial production is still audited on what exists"

    second = StubServices(all_terminal=True)
    campaign.run_all(flight, second)
    assert campaign.read_stage_record(flight.run_root, "production")["status"] == "completed"
    assert "run_audits" in second.calls, "production advanced, so the audit cannot be reused"
    assert "run_coupling" in second.calls
    assert "reused" not in campaign.read_stage_record(flight.run_root, "audit")

    # A third pass over an unchanged, terminal production reuses everything again:
    # the invalidation is bound to what production produced, not to the clock.
    third = StubServices(all_terminal=True)
    campaign.run_all(flight, third)
    assert third.calls == []


def test_the_production_digest_tracks_the_outcome_and_not_the_clock(flight):
    assert campaign.production_outcome_digest(flight.run_root) is None
    campaign.write_stage_record(flight.run_root, "production",
                                {"status": "partial", "completed": 3, "remaining": ["A_x"]})
    partial = campaign.production_outcome_digest(flight.run_root)
    campaign.write_stage_record(flight.run_root, "production",
                                {"status": "partial", "completed": 3, "remaining": ["A_x"],
                                 "wall_seconds": 91.2, "started": 3})
    assert campaign.production_outcome_digest(flight.run_root) == partial, \
        "a rerun that produced nothing new must not invalidate the downstream stages"
    campaign.write_stage_record(flight.run_root, "production",
                                {"status": "completed", "completed": 4, "remaining": []})
    assert campaign.production_outcome_digest(flight.run_root) != partial
