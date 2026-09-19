"""The guarded loop: what it charges, what it saves, and what it refuses to write.

Everything here runs on the CPU against an injected deterministic clock, which is
the only way to assert exact budget arithmetic without a CUDA device. The gate is
live in every test: there is no disable flag in production and none is simulated.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from smallAntibodyGen.experiments import her2_data as data
from smallAntibodyGen.experiments import her2_guard as guard
from smallAntibodyGen.experiments import her2_guarded_trajectory as loop
from smallAntibodyGen.experiments import her2_objectives as arms
from smallAntibodyGen.experiments import her2_preferences as preferences
from smallAntibodyGen.experiments.her2_runtime import GpuBudgetClock

PAIRS = 8


def cores(seed, count):
    rng = np.random.default_rng(seed)
    letters = np.array(list(data.CANONICAL))
    seen, rows = set(), []
    while len(rows) < count:
        core = "".join(rng.choice(letters, size=data.CORE_LENGTH))
        if core not in seen:
            seen.add(core)
            rows.append(core)
    return rows


def instrumented_clock(step_seconds):
    """t, t, t+dt, t+dt, ... and a flag saying whether a segment is currently open."""
    state = {"now": 0.0, "open": False}

    def clock():
        if not state["open"]:
            state["open"] = True
            return state["now"]
        state["open"] = False
        state["now"] += step_seconds
        return state["now"]
    clock.state = state
    return clock


class CountingPolicy:
    """A toy policy that records which rows it scored and whether a clock segment was open."""

    def __init__(self, chosen_index, rejected_index, clock=None, seed=0):
        generator = torch.Generator().manual_seed(seed)
        self.weight = torch.nn.Parameter(torch.randn(data.CORE_LENGTH, 20, generator=generator)
                                         * 0.1)
        self.model = torch.nn.Module()
        self.model.weight = self.weight
        self.device = torch.device("cpu")
        self.chosen_key = np.asarray(chosen_index).tobytes()
        self.rejected_key = np.asarray(rejected_index).tobytes()
        self.clock = clock
        self.rejected_forwards_inside_segments = 0
        self.rejected_forwards = 0

    def position_log_probs(self, index):
        values = torch.as_tensor(np.asarray(index), dtype=torch.long)
        logits = self.weight.unsqueeze(0).expand(values.shape[0], -1, -1)
        return torch.log_softmax(logits, dim=-1).gather(2, values.unsqueeze(-1)).squeeze(-1)

    def sequence_log_probs(self, index):
        if np.asarray(index).tobytes() == self.rejected_key:
            self.rejected_forwards += 1
            if self.clock is not None and self.clock.state["open"]:
                self.rejected_forwards_inside_segments += 1
        return self.position_log_probs(index).sum(dim=1)


class Harness:
    """One trajectory's moving parts, wired the way the runner wires them."""

    def __init__(self, tmp_path, *, step_seconds=1.0, warmup=3, drops=None, arm="dpo",
                 precharge=0.0, update_interval=1):
        self.directory = Path(tmp_path) / "trajectory"
        self.chosen_index = data.encode_cores(cores(1, PAIRS))
        self.rejected_index = data.encode_cores(cores(2, PAIRS))
        self.pairs = {"chosen_index": self.chosen_index, "rejected_index": self.rejected_index,
                      "pairs": PAIRS}
        self.clock_fn = instrumented_clock(step_seconds)
        self.clock = GpuBudgetClock(clock=self.clock_fn)
        if precharge:
            self.clock.charge_reused(precharge, reason="reused frozen reference cache")
        self.monitor_clock = GpuBudgetClock(clock=instrumented_clock(0.1))
        self.policy = CountingPolicy(self.chosen_index, self.rejected_index, clock=self.clock_fn)
        self.optimizer = torch.optim.SGD([self.policy.weight], lr=0.1)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: min(1.0, (step + 1) / warmup))
        self.arm = arm
        self.drops = drops or {}
        self.state = {"drop": 0.0}
        self.reference_chosen = torch.tensor(
            self.policy.sequence_log_probs(self.chosen_index).detach().numpy(),
            dtype=torch.float32)
        self.reference_rejected = torch.tensor(
            self.policy.sequence_log_probs(self.rejected_index).detach().numpy(),
            dtype=torch.float32)
        identity = guard.parent_reference_identity(
            parent_checkpoint_sha256="a" * 64, parent_state_sha256="b" * 64,
            config_sha256="c" * 64, scaffold_prefix="1AAA",
            chosen_index=self.chosen_index, rejected_index=self.rejected_index)
        parent_chosen = np.array(self.reference_chosen.numpy(), dtype=np.float64)
        parent_rejected = np.array(self.reference_rejected.numpy(), dtype=np.float64)
        parent_chosen.setflags(write=False)
        parent_rejected.setflags(write=False)
        self.reference = guard.ParentValidationReference(
            identity=identity, chosen=parent_chosen, rejected=parent_rejected,
            gpu_seconds=0.0, wall_seconds=0.0)
        self.monitor = guard.GateMonitor(
            self.policy, self.reference, directory=self.directory,
            schedule=guard.MonitorSchedule(update_interval=update_interval,
                                           gpu_second_interval=100.0),
            score_sequences_fn=self._score, monitor_clock=self.monitor_clock)
        self.checks = []
        # Building the reference scored both sides once. That was setup, not the
        # trajectory, so the counters start from zero at the first update.
        self.policy.rejected_forwards = 0
        self.policy.rejected_forwards_inside_segments = 0

    def _score(self, policy, index, *, batch_size=256, progress_every=0):
        """The monitored scores: the parent's, minus whatever drop the test declared."""
        key = np.asarray(index).tobytes()
        base = self.reference.chosen if key == self.chosen_key() else self.reference.rejected
        return np.asarray(base, dtype=np.float64) - self.state["drop"]

    def chosen_key(self):
        return np.asarray(self.chosen_index).tobytes()

    def monitor_check(self, *, update, gpu_seconds, reason):
        self.state["drop"] = float(self.drops.get(update, 0.0))
        record = self.monitor.check(self.pairs, update=update, gpu_seconds=gpu_seconds,
                                    reason=reason)
        self.checks.append(record)
        return record

    def step(self, update):
        if self.arm == "continued_sft":
            scores = self.policy.sequence_log_probs(self.chosen_index)
            loss, diagnostics = arms.batch_loss("continued_sft", policy_chosen=scores)
            return loss, {"cycle": 0, "sequences": PAIRS, "pairs": 0,
                          "batch_unique_chosen": PAIRS, "per_pair": diagnostics}
        chosen = self.policy.sequence_log_probs(self.chosen_index)
        rejected = self.policy.sequence_log_probs(self.rejected_index)
        loss, diagnostics = arms.batch_loss(
            self.arm, policy_chosen=chosen, policy_rejected=rejected,
            reference_chosen=self.reference_chosen, reference_rejected=self.reference_rejected,
            coefficients={"beta": 0.1})
        return loss, {"cycle": 0, "batch": update - 1, "sequences": 2 * PAIRS, "pairs": PAIRS,
                      "batch_unique_chosen": PAIRS, "batch_unique_rejected": PAIRS,
                      "per_pair": diagnostics,
                      "pair_cores": (self.chosen_index, self.rejected_index)}

    def run(self, budgets, *, max_updates=None, on_budget=None, step=None, saver=None):
        saved = []

        def budget_callback(budget, record):
            saved.append(budget)
            path = self.directory / f"budget_{int(budget)}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"checkpoint")
            return {"checkpoint": str(path)}

        document = loop.run_guarded_trajectory(
            step=step or self.step, budgets=budgets, clock=self.clock,
            optimizer=self.optimizer, scheduler=self.scheduler, model=self.policy.model,
            gradient_clip=1.0, monitor_check=self.monitor_check, directory=self.directory,
            parent_state={"checkpoint": "checkpoints/parent.pt", "sha256": "d" * 64},
            on_budget=on_budget or budget_callback, exposure_fields=("sequences", "pairs"),
            cumulative_exposures=lambda: {"distinct_chosen_rows": PAIRS,
                                          "chosen_population": PAIRS},
            max_updates=max_updates, schedule=self.monitor.schedule, saver=saver)
        return document, saved

    def journal(self, name):
        path = self.directory / name
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def verdicts(self):
        """The gate verdict lines: one per check, written before any snapshot I/O."""
        return [line for line in self.journal("monitor.jsonl")
                if line["record_kind"] == "gate_verdict"]

    def snapshots(self):
        """The snapshot outcome lines: what was written to disk after each verdict."""
        return [line for line in self.journal("monitor.jsonl")
                if line["record_kind"] == "snapshot"]


# ---------------------------------------------------------------------------
# the passing path
# ---------------------------------------------------------------------------

def test_a_passing_trajectory_reaches_every_budget_and_rolls_the_last_passing_file(tmp_path):
    harness = Harness(tmp_path)
    document, saved = harness.run([2.0, 4.0])
    assert saved == [2.0, 4.0]
    assert document["status"] == "completed"
    assert sorted(document["budgets"]) == ["2.0", "4.0"]
    assert document["budgets_not_reached"] == {}
    assert document["last_passing"]["kind"] == "rolling_file"
    assert document["last_passing"]["update"] == 4
    assert (harness.directory / "last_passing.pt").is_file()
    assert not (harness.directory / "failed_state.pt").exists()
    # One journal line per update; per check, a verdict line and then a snapshot line.
    updates = harness.journal("updates.jsonl")
    assert [entry["update"] for entry in updates] == [1, 2, 3, 4]
    assert len(harness.verdicts()) == len(harness.snapshots()) == len(harness.checks)
    assert all(record["passed"] for record in harness.verdicts())
    assert all(record["last_passing"]["sha256"] for record in harness.snapshots())
    # The verdict is on disk before the bytes it describes.
    lines = harness.journal("monitor.jsonl")
    assert [line["record_kind"] for line in lines[:2]] == ["gate_verdict", "snapshot"]
    assert lines[1]["verdict_journal_line"] == 0


def test_each_passing_check_records_the_hash_of_the_bytes_it_wrote(tmp_path):
    """The rolling file is a latest-state artifact; earlier hashes are history, not promises."""
    harness = Harness(tmp_path)
    document, _ = harness.run([4.0])
    hashes = [record["last_passing"]["sha256"] for record in harness.snapshots()]
    assert len(set(hashes)) > 1, "the weights move, so the rolling file's bytes move with them"
    from smallAntibodyGen.experiments.her2_runtime import sha256
    current = sha256(harness.directory / "last_passing.pt")
    assert current == hashes[-1]
    assert current != hashes[0], "the first check's hash no longer resolves at that path"
    assert "rolling latest-state artifact" in document["rolling_checkpoint_note"]


def test_the_budget_checkpoint_exists_because_the_gate_check_before_it_passed(tmp_path):
    harness = Harness(tmp_path)
    document, _ = harness.run([2.0])
    record = document["budgets"]["2.0"]
    assert record["reached"] is True
    assert record["gate_check"] == len(harness.checks) - 1
    assert harness.checks[record["gate_check"]]["passed"] is True
    assert harness.checks[record["gate_check"]]["reason"] == "budget_crossing"


def test_the_journal_separates_the_learning_rate_used_from_the_next_one(tmp_path):
    harness = Harness(tmp_path, warmup=4)
    harness.run([3.0])
    updates = harness.journal("updates.jsonl")
    first = updates[0]
    assert first["learning_rate_used"] < first["learning_rate_next"], "warmup is still climbing"
    assert first["learning_rate_used"] == pytest.approx(0.1 * 1 / 4)
    assert first["learning_rate_next"] == pytest.approx(0.1 * 2 / 4)


def test_the_journal_carries_the_per_pair_distributions_and_the_pair_distance(tmp_path):
    harness = Harness(tmp_path, arm="dpop")
    harness.run([2.0], step=lambda update: _dpop_step(harness, update))
    entry = harness.journal("updates.jsonl")[0]
    assert set(entry["margin"]["quantiles"]) == {str(q) for q in guard.QUANTILES}
    assert entry["coefficient"]["mean"] is not None
    assert entry["modified_margin"]["mean"] is not None
    assert entry["modified_coefficient"]["mean"] is not None
    assert 0.0 <= entry["hinge_active_fraction"] <= 1.0
    assert entry["within_pair_hamming"]["pairs"] == PAIRS
    assert entry["distinct_exposures"]["chosen_population"] == PAIRS
    assert entry["exposures"]["sequences"] == 2 * PAIRS


def _dpop_step(harness, update):
    chosen = harness.policy.sequence_log_probs(harness.chosen_index)
    rejected = harness.policy.sequence_log_probs(harness.rejected_index)
    loss, diagnostics = arms.batch_loss(
        "dpop", policy_chosen=chosen, policy_rejected=rejected,
        reference_chosen=harness.reference_chosen, reference_rejected=harness.reference_rejected,
        coefficients={"beta": 0.1, "lambda": 1.0})
    return loss, {"cycle": 0, "batch": update - 1, "sequences": 2 * PAIRS, "pairs": PAIRS,
                  "per_pair": diagnostics,
                  "pair_cores": (harness.chosen_index, harness.rejected_index)}


# ---------------------------------------------------------------------------
# the stop
# ---------------------------------------------------------------------------

def test_a_breach_stops_before_the_next_update_and_before_the_next_checkpoint(tmp_path):
    harness = Harness(tmp_path, drops={3: 5.0})
    document, saved = harness.run([2.0, 6.0])
    assert document["status"] == "stopped"
    assert document["stop_reason"] == "parent_relative_likelihood_breach"
    assert document["updates"] == 3, "the update that tripped it ran; the next one did not"
    assert [entry["update"] for entry in harness.journal("updates.jsonl")] == [1, 2, 3]
    assert saved == [2.0], "the 6 s budget was never reached, so no callback ran"
    assert document["budgets_not_reached"]["6.0"]["reached"] is False
    assert document["budgets_not_reached"]["6.0"]["stopped_at_update"] == 3
    assert "checkpoint" not in document["budgets_not_reached"]["6.0"]
    assert not (harness.directory / "budget_6.pt").exists()


def test_the_failing_weights_go_to_their_own_file_and_never_to_last_passing(tmp_path):
    harness = Harness(tmp_path, drops={3: 5.0})
    document, _ = harness.run([2.0, 6.0])
    assert (harness.directory / "failed_state.pt").is_file()
    assert (harness.directory / "optimizer_state.pt").is_file()
    assert (harness.directory / "scheduler_state.pt").is_file()
    failed = torch.load(harness.directory / "failed_state.pt", weights_only=False)
    passing = torch.load(harness.directory / "last_passing.pt", weights_only=False)
    assert failed["kind"] == "failed_state" and passing["kind"] == "last_passing"
    assert failed["update"] == 3 and passing["update"] == 2
    assert not torch.equal(failed["state"]["weight"], passing["state"]["weight"])
    assert document["last_passing"]["update"] == 2


def test_a_breach_at_the_very_first_check_leaves_the_parent_as_the_last_passing_state(tmp_path):
    harness = Harness(tmp_path, drops={1: 5.0})
    document, saved = harness.run([2.0])
    assert document["updates"] == 1
    assert saved == []
    assert document["last_passing"]["kind"] == "verified_parent"
    assert document["last_passing"]["sha256"] == "d" * 64
    assert document["last_passing"]["update"] == 0
    assert not (harness.directory / "last_passing.pt").exists()


def test_the_stop_record_reports_the_two_observed_states_and_dates_nothing(tmp_path):
    """The interval is not a bracket: an earlier excursion that recovered looks the same."""
    harness = Harness(tmp_path, drops={4: 5.0}, update_interval=3)
    document, _ = harness.run([10.0], max_updates=6)
    stop = document["stop"]
    assert stop["stop_check"]["update"] == 4
    assert stop["last_passing_check"]["update"] == 1
    assert stop["interval_updates"] == 3
    assert stop["interval_gpu_seconds"] == pytest.approx(3.0)
    assert "did not observe the first crossing" in stop["bracket_note"]
    assert "not a bracket on it" in stop["bracket_note"]
    assert "No onset is claimed" in stop["bracket_note"]
    assert "run" in stop and "snapshot" in stop
    assert json.loads((harness.directory / "stop.json").read_text(encoding="utf-8")) == stop


def test_a_nonfinite_validation_score_stops_the_trajectory(tmp_path):
    harness = Harness(tmp_path, drops={2: float("nan")})
    document, _ = harness.run([6.0])
    assert document["status"] == "stopped"
    assert document["stop_reason"] == "nonfinite_validation_score"
    assert document["updates"] == 2


def test_the_last_passing_state_is_not_selectable_at_a_budget_it_never_reached(tmp_path):
    harness = Harness(tmp_path, drops={3: 5.0})
    document, _ = harness.run([2.0, 6.0])
    assert document["last_passing_is_selectable"] is False
    assert document["selectable_endpoints"] == ["2.0"]
    assert "never relabelled as a budget it did not reach" in document["selection_note"]


def test_the_rolling_state_is_not_selectable_even_when_every_budget_was_reached(tmp_path):
    """Completing the budgets says nothing about the rolling file. Endpoints are selected."""
    harness = Harness(tmp_path)
    document, _ = harness.run([2.0, 4.0])
    assert document["status"] == "completed"
    assert document["budgets_not_reached"] == {}
    assert document["last_passing_is_selectable"] is False
    assert document["selectable_endpoints"] == ["2.0", "4.0"]
    assert "not selectable even when every budget WAS reached" in document["selection_note"]


def test_an_artificially_capped_run_with_budgets_outstanding_is_not_completed(tmp_path):
    """A max_updates cap is an artificial end, and `completed` would misreport it."""
    harness = Harness(tmp_path, update_interval=10)
    document, saved = harness.run([2.0, 50.0], max_updates=3)
    assert saved == [2.0]
    assert document["status"] == "incomplete"
    assert document["update_cap"] == 3 and document["update_cap_reached"] is True
    assert document["updates"] == 3
    assert document["budgets_not_reached"]["50.0"]["reached"] is False
    assert document["last_passing_is_selectable"] is False


# ---------------------------------------------------------------------------
# durability
# ---------------------------------------------------------------------------

def test_an_exception_inside_the_step_still_leaves_complete_evidence(tmp_path):
    harness = Harness(tmp_path)

    def exploding_step(update):
        if update == 3:
            raise RuntimeError("synthetic device failure")
        return harness.step(update)

    with pytest.raises(RuntimeError, match="synthetic device failure"):
        harness.run([10.0], step=exploding_step)
    document = json.loads((harness.directory / "trajectory.json").read_text(encoding="utf-8"))
    assert document["status"] == "failed"
    assert document["failure"]["type"] == "RuntimeError"
    # Two updates completed and a third was attempted. "3 updates" would claim a
    # journal line that does not exist.
    assert document["failure"]["attempted_update"] == 3
    assert document["failure"]["updates"] == 2
    assert document["updates"] == 2 and document["attempted_updates"] == 3
    assert document["failed_attempts"] == 1
    assert document["failure"]["optimizer_stepped"] is False
    # The failed attempt's device time was really spent, so it is charged and reported.
    assert document["cost"]["training_gpu_seconds"] == pytest.approx(3.0)
    assert document["failed_work_gpu_seconds"] == pytest.approx(1.0)
    assert [entry["update"] for entry in harness.journal("updates.jsonl")] == [1, 2]
    assert (harness.directory / "stop.json").is_file()
    assert (harness.directory / "failed_state.pt").is_file()
    assert document["budgets_not_reached"]["10.0"]["stop_reason"] == "exception"


class BrokenSaver(loop.StateSaver):
    """A saver whose disk fails at exactly the wrong moment."""

    def __init__(self, *args, fail_on, **kwargs):
        super().__init__(*args, **kwargs)
        self.fail_on = fail_on

    def _write(self, name, payload):
        if name.startswith(self.fail_on):
            raise OSError("synthetic disk failure")
        return super()._write(name, payload)


def test_a_snapshot_that_cannot_be_written_does_not_take_the_verdict_with_it(tmp_path):
    """The stop rests on the measurement, not on the diagnostic bytes beside it."""
    harness = Harness(tmp_path, drops={2: 5.0})
    saver = BrokenSaver(harness.directory, harness.policy.model, fail_on="failed_state")
    document, _ = harness.run([6.0], saver=saver)
    assert document["status"] == "stopped"
    assert document["stop_reason"] == "parent_relative_likelihood_breach"
    verdicts = harness.verdicts()
    assert len(verdicts) == 2 and verdicts[-1]["passed"] is False
    assert verdicts[-1]["D"] == pytest.approx(5.0)
    assert verdicts[-1]["scores"]["chosen_values_sha256"]
    snapshot = harness.snapshots()[-1]
    assert "synthetic disk failure" in snapshot["save_failed"]
    assert snapshot["D"] == pytest.approx(5.0)
    # The measured D and the score hashes survive into the stop record.
    assert document["stop"]["detail"]["D"] == pytest.approx(5.0)
    assert document["stop"]["detail"]["scores"]["sha256"]
    assert document["snapshot_errors"]
    assert not (harness.directory / "failed_state.pt").exists()


def test_a_failed_rolling_save_stops_the_trajectory_before_another_update(tmp_path):
    """CX-33: the loop may not run on past a passing state it could not save.

    Continuing would move the weights beyond a state nothing recorded, and would
    then write a nominal endpoint whose rolling companion is already stale. The
    verdict and the previous passing bytes survive; the run does not.
    """
    harness = Harness(tmp_path, update_interval=1)
    saver = BrokenSaver(harness.directory, harness.policy.model, fail_on="last_passing")
    document, saved = harness.run([2.0, 4.0], saver=saver)
    assert document["updates"] == 1, "the first failed save ends it; no second update ran"
    assert len(harness.checks) == 1
    assert saved == [], "and no nominal endpoint was written after it"
    assert document["status"] == "failed", "an artifact failure is not a completion"
    assert document["stop_reason"] == "last_passing_snapshot_failure"
    # The measured verdict is intact and so is the previous passing state.
    verdict = harness.verdicts()[-1]
    assert verdict["passed"] is True and verdict["D"] is not None
    assert verdict["scores"]["chosen_values_sha256"]
    assert document["last_passing"]["kind"] == "verified_parent"
    assert document["last_passing"]["sha256"] == "d" * 64
    assert not (harness.directory / "last_passing.pt").exists()
    assert len(document["snapshot_errors"]) == 1
    assert "synthetic disk failure" in document["snapshot_errors"][0]["save_failed"]
    assert document["failure"]["where"] == "save_last_passing"
    assert document["failure"]["gate_verdict"]["passed"] is True
    # Both budgets are still accounted for, as budgets nothing reached.
    assert sorted(document["budgets_not_reached"]) == ["2.0", "4.0"]
    assert document["budgets"] == {}
    # The evidence is on disk and inspectable.
    assert (harness.directory / "stop.json").is_file()
    assert json.loads((harness.directory / "trajectory.json").read_text(
        encoding="utf-8"))["status"] == "failed"


def test_a_failed_nominal_checkpoint_write_does_not_lose_the_budget(tmp_path):
    """CX-34: a budget leaves the outstanding list only after its endpoint is durable."""
    harness = Harness(tmp_path)

    def failing_budget(budget, record):
        raise OSError("synthetic nominal save failure")

    with pytest.raises(OSError, match="synthetic nominal save failure"):
        harness.run([1.0, 3.0], on_budget=failing_budget)
    document = json.loads((harness.directory / "trajectory.json").read_text(encoding="utf-8"))
    assert document["budgets"] == {}, "nothing was durably written, so nothing was reached"
    assert sorted(document["budgets_not_reached"]) == ["1.0", "3.0"], "and none vanished"
    assert document["status"] == "failed"
    lines = harness.journal("budgets.jsonl")
    assert [line["record_kind"] for line in lines] == ["budget_checkpoint_failed"]
    assert lines[0]["target_gpu_seconds"] == 1.0 and lines[0]["reached"] is False
    progress = json.loads((harness.directory / "trajectory_progress.json").read_text(
        encoding="utf-8"))
    assert 1.0 in progress["budgets_outstanding"]


def test_a_failing_state_save_on_a_real_breach_stays_best_effort(tmp_path):
    """CX-33's other half: the gate stop is not downgraded by a diagnostic write failing."""
    harness = Harness(tmp_path, drops={2: 5.0})
    saver = BrokenSaver(harness.directory, harness.policy.model, fail_on="failed_state")
    document, _ = harness.run([6.0], saver=saver)
    assert document["status"] == "stopped"
    assert document["stop_reason"] == "parent_relative_likelihood_breach"
    assert document["snapshot_errors"]


def test_the_controller_advances_the_cadence_without_help_from_the_callback(tmp_path):
    """A pure monitor callback must not change when the next check is due."""
    harness = Harness(tmp_path)
    seen = []

    def pure_check(*, update, gpu_seconds, reason):
        seen.append((update, reason))
        return {"passed": True, "check": len(seen) - 1, "D": 0.0}

    schedule = guard.MonitorSchedule(update_interval=25, gpu_second_interval=100.0)
    document = loop.run_guarded_trajectory(
        step=harness.step, budgets=[1000.0], clock=harness.clock, optimizer=harness.optimizer,
        scheduler=harness.scheduler, model=harness.policy.model, gradient_clip=1.0,
        monitor_check=pure_check, directory=harness.directory,
        parent_state={"checkpoint": "p.pt", "sha256": "d" * 64}, max_updates=30,
        schedule=schedule, exposure_fields=("sequences", "pairs"))
    assert [update for update, _ in seen] == [1, 26, 30]
    assert [reason for _, reason in seen] == ["first_update", "update_interval", "final_state"]
    assert schedule.last_update == 30
    assert document["status"] == "incomplete", "the 1000 s budget was never reached"


def test_the_budget_mapping_is_recoverable_from_disk_before_the_run_ends(tmp_path):
    """A kill at update 1,201 must not lose which budget the checkpoint beside it is."""
    harness = Harness(tmp_path)
    seen = {}

    def on_budget(budget, record):
        path = harness.directory / "trajectory_progress.json"
        seen[budget] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
        return {}

    document, _ = harness.run([2.0, 4.0], on_budget=on_budget)
    assert seen[2.0]["budgets"] == {}, "the first crossing is not recorded before it happens"
    assert sorted(seen[4.0]["budgets"]) == ["2.0"], "by the second, the first is already on disk"
    assert seen[4.0]["budgets"]["2.0"]["reached"] is True
    assert seen[4.0]["budgets_outstanding"] == [4.0]
    lines = harness.journal("budgets.jsonl")
    assert [line["target_gpu_seconds"] for line in lines] == [2.0, 4.0]
    assert all(line["record_kind"] == "budget_reached" for line in lines)
    final = json.loads((harness.directory / "trajectory_progress.json").read_text(
        encoding="utf-8"))
    assert sorted(final["budgets"]) == ["2.0", "4.0"] and final["budgets_outstanding"] == []
    assert final["updates"] == document["updates"]


def test_the_per_pair_loss_summary_does_not_replace_the_scalar_batch_loss(tmp_path):
    """The optimizer stepped on a number; a distribution may not land on its name."""
    harness = Harness(tmp_path)
    harness.run([2.0])
    entry = harness.journal("updates.jsonl")[0]
    assert isinstance(entry["loss"], float)
    assert isinstance(entry["per_pair_loss"], dict)
    assert entry["per_pair_loss"]["mean"] == pytest.approx(entry["loss"], rel=1e-5)
    assert entry["sign_accuracy"] is not None and 0.0 <= entry["sign_accuracy"] <= 1.0
    assert entry["rejected_log_probability"]["mean"] is not None
    assert entry["chosen_nll_per_residue"]["mean"] is not None
    assert entry["rejected_change"]["quantiles"]


def test_the_journals_are_readable_after_a_stop_without_a_clean_close(tmp_path):
    """Every line is flushed and fsynced as it is written, not at close."""
    harness = Harness(tmp_path, drops={2: 5.0})
    harness.run([6.0])
    lines = (harness.directory / "updates.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert all(json.loads(line)["update"] for line in lines)


# ---------------------------------------------------------------------------
# cost accounting
# ---------------------------------------------------------------------------

def test_monitoring_never_pollutes_the_training_clock(tmp_path):
    harness = Harness(tmp_path, update_interval=1)
    document, _ = harness.run([4.0])
    assert len(harness.checks) == 4, "one check per update at this cadence"
    assert document["cost"]["training_gpu_seconds"] == pytest.approx(4.0)
    assert harness.monitor.gpu_seconds == pytest.approx(0.4)
    assert document["cost"]["diagnostic_wall_seconds"] >= 0.0
    assert document["cost"]["checkpoint_wall_seconds"] > 0.0
    updates = harness.journal("updates.jsonl")
    assert sum(entry["update_gpu_seconds"] for entry in updates) == pytest.approx(4.0)


def test_continued_sft_performs_no_rejected_inference_inside_a_charged_segment(tmp_path):
    harness = Harness(tmp_path, arm="continued_sft")
    harness.run([3.0])
    assert harness.policy.rejected_forwards_inside_segments == 0
    assert harness.policy.rejected_forwards == 0, "the injected monitor scorer does not forward"
    entry = harness.journal("updates.jsonl")[0]
    assert "margin" not in entry and "coefficient" not in entry
    assert entry["nll_per_residue"]["mean"] is not None
    assert entry["pairs"] == 0


def test_the_gate_applies_to_continued_sft_on_the_same_code_path(tmp_path):
    harness = Harness(tmp_path, arm="continued_sft", drops={2: 5.0})
    document, _ = harness.run([6.0])
    assert document["status"] == "stopped"
    assert document["stop_reason"] == "parent_relative_likelihood_breach"


def test_a_precharge_that_eats_the_first_budget_fails_honestly(tmp_path):
    harness = Harness(tmp_path, precharge=4.0)
    with pytest.raises(ValueError, match="already charged"):
        harness.run([3.0, 6.0])


def test_a_precharge_inside_the_first_budget_is_charged_not_forgiven(tmp_path):
    harness = Harness(tmp_path, precharge=1.5)
    document, _ = harness.run([3.0])
    assert document["cost"]["precharged_gpu_seconds"] == pytest.approx(1.5)
    assert document["updates"] == 2
    assert document["budgets"]["3.0"]["actual_gpu_seconds"] == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# the fork does not drift from the original loop
# ---------------------------------------------------------------------------

def test_the_guarded_loop_agrees_with_the_original_on_the_scientific_records(tmp_path):
    """Same synthetic input, a live gate that genuinely passes: same budget bookkeeping.

    Timing dictionaries are deliberately not compared -- they are different
    objects with different fields. What must agree is what a result depends on:
    where each budget was crossed, how many updates it took, whether the
    overshoot stayed inside one update, and what was consumed.
    """
    budgets = [3.0, 7.0]
    harness = Harness(tmp_path / "guarded")
    guarded, _ = harness.run(budgets)

    original_clock = GpuBudgetClock(clock=instrumented_clock(1.0))
    policy = CountingPolicy(harness.chosen_index, harness.rejected_index, seed=0)
    optimizer = torch.optim.SGD([policy.weight], lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / 3))
    steps = {"count": 0}

    def step(update):
        steps["count"] += 1
        chosen = policy.sequence_log_probs(harness.chosen_index)
        rejected = policy.sequence_log_probs(harness.rejected_index)
        loss, _ = arms.batch_loss("dpo", policy_chosen=chosen, policy_rejected=rejected,
                                  reference_chosen=harness.reference_chosen,
                                  reference_rejected=harness.reference_rejected,
                                  coefficients={"beta": 0.1})
        return loss, {"sequences": 2 * PAIRS, "pairs": PAIRS}

    original = preferences.run_budgeted_trajectory(
        step=step, budgets=budgets, clock=original_clock, on_budget=lambda budget, record: {},
        optimizer=optimizer, scheduler=scheduler, model=policy.model, gradient_clip=1.0,
        exposure_fields=("sequences", "pairs"))

    assert guarded["updates"] == original["updates"] == steps["count"]
    assert sorted(guarded["budgets"]) == sorted(original["budgets"])
    for key in sorted(original["budgets"]):
        mine, theirs = guarded["budgets"][key], original["budgets"][key]
        assert mine["updates"] == theirs["updates"]
        assert mine["actual_gpu_seconds"] == pytest.approx(theirs["actual_gpu_seconds"])
        assert mine["overshoot_within_one_update"] == theirs["overshoot_within_one_update"]
        assert mine["exposures"] == theirs["exposures"]
        assert mine["core_token_exposures"] == theirs["core_token_exposures"]
    assert guarded["exposures"] == original["exposures"]
    assert all(record["passed"] for record in harness.checks), "the gate was live and passed"


def test_the_guarded_loop_needs_a_budget_and_a_cadence(tmp_path):
    harness = Harness(tmp_path)
    with pytest.raises(ValueError, match="at least one budget"):
        harness.run([])
    with pytest.raises(ValueError, match="cadence must be supplied"):
        loop.run_guarded_trajectory(
            step=harness.step, budgets=[1.0], clock=harness.clock, optimizer=harness.optimizer,
            scheduler=harness.scheduler, model=harness.policy.model, gradient_clip=1.0,
            monitor_check=harness.monitor_check, directory=harness.directory,
            parent_state={"checkpoint": "p.pt", "sha256": "d" * 64})


def test_within_pair_hamming_is_the_pairs_own_distance():
    left = data.encode_cores(["ACDEFGHIKL", "ACDEFGHIKL"])
    right = data.encode_cores(["ACDEFGHIKM", "WCDEFGHIKM"])
    summary = loop.within_pair_hamming(left, right)
    assert summary["mean"] == pytest.approx(1.5)
    assert summary["min"] == 1 and summary["max"] == 2
