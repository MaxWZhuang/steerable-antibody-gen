"""The loop: exact resume, de-duplicated journals, and the difference between
stopping and crashing.

The resume test compares full state digests, not loss curves. A loss curve
hides the two failures that matter -- an AdamW ``step`` counter or a
``LambdaLR`` ``last_epoch`` restored one off, which shifts the warmup by a
factor of two at the first continued update and is invisible in four decimal
places.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from smallAntibodyGen.experiments import her2_nf_monitor as monitor
from smallAntibodyGen.experiments import her2_nf_objectives as objectives
from smallAntibodyGen.experiments import her2_nf_trajectory as trajectory
from smallAntibodyGen.tests.her2_nf_support import (TinyPolicy, TinyStream, state_digest,
                                                    tiny_cores, tiny_optimizer)

ROWS = 64


class Harness:
    """One arm wired to the tiny policy; the only thing a test varies is the gate."""

    def __init__(self, tmp_path, *, updates=8, batch_rows=8, micro=4, gate=None,
                 preservation="none", lam=0.0, seed=0):
        self.cores = tiny_cores(ROWS, seed=1)
        self.rejected = tiny_cores(ROWS, seed=2)
        self.policy = TinyPolicy(seed=seed)
        self.optimizer, self.scheduler = tiny_optimizer(self.policy)
        self.reference_chosen = self.policy.sequence_log_probs(self.cores).detach()
        self.reference_rejected = self.policy.sequence_log_probs(self.rejected).detach()
        self.parent_bank = tiny_cores(32, seed=3)
        self.parent_scores = self.policy.score(self.parent_bank)["sum_log_probability"]
        self.stream = TinyStream(rows=ROWS, batch_rows=batch_rows, updates=updates, seed=4)
        self.replay_order = np.tile(np.arange(32), 200)[:batch_rows * updates]
        self.updates = updates
        self.batch_rows = batch_rows
        self.micro = micro
        self.preservation = preservation
        self.lam = lam
        self.gate = gate or (lambda **kwargs: {"passed": True, "D": 0.0})
        self.directory = tmp_path
        self.identity = {"trajectory": "tiny", "parent_state_sha256": "p"}
        self.endpoint_calls = []

    def task_batch(self, chosen_rows, rejected_rows):
        return objectives.task_term(
            "ipo", policy_chosen=self.policy.sequence_log_probs(self.cores[chosen_rows]),
            policy_rejected=self.policy.sequence_log_probs(self.rejected[rejected_rows]),
            reference_chosen=self.reference_chosen[chosen_rows],
            reference_rejected=self.reference_rejected[rejected_rows],
            coefficients={"tau": 0.1})

    def preservation_batch(self, rows):
        parent = torch.as_tensor(self.parent_scores[rows])
        return objectives.tail_term(parent, self.policy.sequence_log_probs(
            self.parent_bank[rows]))

    def endpoint(self, **kwargs):
        self.endpoint_calls.append(int(kwargs["update"]))
        return {"update": int(kwargs["update"]), "exposures": dict(kwargs["exposures"])}

    def run(self, *, updates=None, resume=True, endpoints=None, on_endpoint=None,
            checkpoints=(), on_checkpoint=None, restore_sentinel=None):
        total = int(updates or self.updates)
        endpoints = tuple(endpoints or (total,))
        plan = monitor.MonitorPlan(endpoints=endpoints, checkpoints=tuple(checkpoints),
                                   sentinel_interval=2, full_interval=2)
        return trajectory.run_trajectory(
            row={"trajectory": "tiny", "block": "A", "task": "ipo", "seed": 1},
            directory=self.directory, policy=self.policy, optimizer=self.optimizer,
            scheduler=self.scheduler, stream=self.stream, replay_order=self.replay_order,
            plan=plan, endpoints=endpoints, batch_rows=self.batch_rows,
            microbatch_rows=self.micro, task_batch=self.task_batch,
            preservation_batch=(self.preservation_batch if self.preservation != "none"
                                else None),
            full_check=self.gate,
            sentinel_check=lambda **kwargs: {"record_kind": "sentinel_check",
                                             "request_full_check": False, "D": 0.0,
                                             "update": kwargs["update"]},
            on_endpoint=on_endpoint or self.endpoint, gradient_clip=1.0,
            identity=self.identity, checkpoints=tuple(checkpoints),
            on_checkpoint=on_checkpoint, restore_sentinel=restore_sentinel,
            uses_rejected=True, preservation_family=self.preservation,
            preservation_lambda=self.lam, max_updates=total, resume=resume,
            stream_digests=self.stream.document())


def test_a_completed_run_reports_its_endpoints_and_exposures(tmp_path):
    harness = Harness(tmp_path, updates=6, batch_rows=8)
    document = harness.run()
    assert document["status"] == trajectory.STATUS_COMPLETED
    assert document["updates"] == 6
    assert document["exposures"] == {"chosen": 48, "rejected": 48, "replay": 0, "tail": 0}
    assert sorted(document["endpoints_reached"]) == ["6"]
    assert document["endpoints_not_reached"] == {}


def test_exposures_are_a_pure_function_of_the_update_number():
    assert trajectory.exposures_at(10, batch_rows=64, uses_rejected=True, replay_rows=64,
                                   tail_rows=0) == {"chosen": 640, "rejected": 640,
                                                    "replay": 640, "tail": 0}


def test_interrupted_intermediate_checkpoint_is_published_before_advancing(tmp_path):
    calls = []

    def publish(**kwargs):
        calls.append(kwargs["update"])
        if len(calls) == 1:
            raise RuntimeError("checkpoint publication interruption")
        return {"checkpoint_update": kwargs["update"]}

    first = Harness(tmp_path, updates=4)
    with pytest.raises(RuntimeError, match="checkpoint publication interruption"):
        first.run(checkpoints=(2,), on_checkpoint=publish)
    resumed = Harness(tmp_path, updates=4)
    result = resumed.run(checkpoints=(2,), on_checkpoint=publish)
    assert result["status"] == trajectory.STATUS_COMPLETED
    assert calls == [2, 2]
    assert result["checkpoints_saved"] == [2]
    assert result["stream_position"] == 32
    assert result["updates"] == 4


def test_a_durable_gate_stop_cannot_resume_training(tmp_path):
    calls = []

    def gate(**kwargs):
        calls.append(kwargs["update"])
        return {"passed": kwargs["update"] < 2, "D": 2.0,
                "stop_reason": "deliberate gate stop"}

    first = Harness(tmp_path, updates=4, gate=gate)
    assert first.run()["status"] == trajectory.STATUS_STOPPED
    before = list(calls)
    resumed = Harness(tmp_path, updates=4, gate=gate)
    result = resumed.run()
    assert result["status"] == trajectory.STATUS_STOPPED
    assert result["updates"] == 2
    assert calls == before
    for key, value in first.policy.model.state_dict().items():
        assert torch.equal(value, resumed.policy.model.state_dict()[key])
    # continued-SFT-style arms consume no rejected row; DPO and SimPO do.
    assert trajectory.exposures_at(3, batch_rows=8, uses_rejected=False, replay_rows=0,
                                   tail_rows=8)["rejected"] == 0


def test_resume_reproduces_the_uninterrupted_state_exactly(tmp_path):
    straight = Harness(tmp_path / "straight", updates=8, batch_rows=8, seed=5)
    straight.run(updates=8)
    interrupted = Harness(tmp_path / "resumed", updates=8, batch_rows=8, seed=5)
    interrupted.run(updates=4, endpoints=(4,))
    (interrupted.directory / trajectory.STATUS_JSON).unlink()
    interrupted.run(updates=8, endpoints=(8,))
    assert state_digest(interrupted.policy) == state_digest(straight.policy)


def test_resume_restores_the_optimizer_and_scheduler_rather_than_rebuilding_them(tmp_path):
    harness = Harness(tmp_path, updates=4, batch_rows=8, seed=6)
    harness.run(updates=4)
    store = trajectory.TrajectoryStateStore(tmp_path, identity=harness.identity)
    payload = torch.load(store.path(), map_location="cpu", weights_only=False)
    assert payload["optimizer"]["state"], "AdamW's moment state must be persisted"
    assert payload["scheduler"]["last_epoch"] == 4
    assert payload["scaler"] is None
    assert "GradScaler" in payload["scaler_reason"]
    assert set(payload["rng"]) >= {"python", "numpy", "torch_cpu", "torch_cuda"}


def test_a_resumed_run_does_not_double_count_updates_or_endpoints(tmp_path):
    harness = Harness(tmp_path, updates=6, batch_rows=8, seed=7)
    harness.run(updates=4, endpoints=(2, 4))
    (harness.directory / trajectory.STATUS_JSON).unlink()
    document = harness.run(updates=6, endpoints=(2, 4, 6))
    progress = trajectory.durable_progress(harness.directory)
    assert progress["updates"] == 6
    assert progress["distinct_update_records"] == 6
    assert sorted(int(value) for value in progress["endpoints_reached_updates"]) == [2, 4, 6]
    assert document["exposures"]["chosen"] == 48
    assert progress["resume_boundaries"], "the replay is disclosed, not silent"


def test_a_state_ahead_of_its_journal_is_refused(tmp_path):
    harness = Harness(tmp_path, updates=4, batch_rows=8, seed=8)
    harness.run(updates=4)
    (harness.directory / trajectory.UPDATES_JSONL).write_text("", encoding="utf-8")
    (harness.directory / trajectory.STATUS_JSON).unlink()
    with pytest.raises(ValueError, match="ahead of its own journal"):
        harness.run(updates=6)


def test_a_state_from_a_different_trajectory_is_refused(tmp_path):
    harness = Harness(tmp_path, updates=2, batch_rows=8, seed=9)
    harness.run(updates=2)
    store = trajectory.TrajectoryStateStore(tmp_path, identity={"trajectory": "someone_else"})
    policy = TinyPolicy(seed=9)
    optimizer, scheduler = tiny_optimizer(policy)
    with pytest.raises(ValueError, match="different trajectory"):
        store.load(policy=policy, optimizer=optimizer, scheduler=scheduler)


def test_a_corrupted_state_is_refused(tmp_path):
    harness = Harness(tmp_path, updates=2, batch_rows=8, seed=10)
    harness.run(updates=2)
    store = trajectory.TrajectoryStateStore(tmp_path, identity=harness.identity)
    payload = torch.load(store.path(), map_location="cpu", weights_only=False)
    payload["state"]["table"] = payload["state"]["table"] + 1.0
    torch.save(payload, store.path())
    policy = TinyPolicy(seed=10)
    optimizer, scheduler = tiny_optimizer(policy)
    with pytest.raises(ValueError, match="corrupted state is not resumed"):
        store.load(policy=policy, optimizer=optimizer, scheduler=scheduler)


def test_a_state_from_different_streams_is_refused(tmp_path):
    harness = Harness(tmp_path, updates=2, batch_rows=8, seed=11)
    harness.run(updates=2)
    store = trajectory.TrajectoryStateStore(tmp_path, identity=harness.identity)
    policy = TinyPolicy(seed=11)
    optimizer, scheduler = tiny_optimizer(policy)
    with pytest.raises(ValueError, match="different data streams"):
        store.load(policy=policy, optimizer=optimizer, scheduler=scheduler,
                   expect_stream_digests={"chosen": "not-the-same"})


def test_a_gate_breach_is_a_stopped_outcome_with_a_diagnostic_snapshot(tmp_path):
    def gate(**kwargs):
        if int(kwargs["update"]) >= 4:
            return {"passed": False, "D": 9.9, "stop_reason": "parent_relative_likelihood_breach"}
        return {"passed": True, "D": 0.1}

    harness = Harness(tmp_path, updates=8, batch_rows=8, gate=gate, seed=12)
    document = harness.run()
    assert document["status"] == trajectory.STATUS_STOPPED
    assert document["stop_reason"] == "parent_relative_likelihood_breach"
    assert (tmp_path / trajectory.FAILED_STATE).is_file()
    assert document["endpoints_not_reached"]
    assert "no checkpoint exists and none is substituted" in \
        document["endpoints_not_reached"]["8"]["reason"]


def test_a_crash_is_failed_is_journal_counted_and_is_re_raised(tmp_path):
    class Boom(RuntimeError):
        pass

    def gate(**kwargs):
        if int(kwargs["update"]) >= 4:
            raise Boom("scorer exploded")
        return {"passed": True}

    harness = Harness(tmp_path, updates=8, batch_rows=8, gate=gate, seed=13)
    with pytest.raises(Boom):
        harness.run()
    status = trajectory.read_terminal_status(tmp_path)
    assert status["status"] == trajectory.STATUS_FAILED
    assert status["updates"] == 4 and status["attempted_updates"] == 4
    assert "re-raised" in status["failure"]["consequence"]
    assert "never presented as completed work" in status["failure"]["counters_from"]


def test_the_check_restores_the_training_generators(tmp_path):
    """Evaluation must not advance the training RNG streams."""
    observed = {}

    def gate(**kwargs):
        torch.rand(5)                     # a check that consumes randomness
        np.random.rand(5)
        observed["ran"] = True
        return {"passed": True}

    torch.manual_seed(1234)
    np.random.seed(99)
    harness = Harness(tmp_path, updates=2, batch_rows=8, gate=gate, seed=14)
    before_torch = torch.get_rng_state().clone()
    before_numpy = np.random.get_state(legacy=True)
    harness.run(updates=2)
    assert observed["ran"]
    assert torch.equal(torch.get_rng_state(), before_torch)
    after_numpy = np.random.get_state(legacy=True)
    # Keys AND position: advancing the position alone is enough to change every
    # later draw, so comparing the key array by itself would pass on a real leak.
    assert np.array_equal(after_numpy[1], before_numpy[1])
    assert after_numpy[2] == before_numpy[2]


def test_a_sentinel_can_only_request_and_the_full_check_decides(tmp_path):
    calls = {"full": 0, "sentinel": 0}

    def gate(**kwargs):
        calls["full"] += 1
        return {"passed": True}

    harness = Harness(tmp_path, updates=8, batch_rows=8, gate=gate, seed=15)

    def sentinel(**kwargs):
        calls["sentinel"] += 1
        return {"record_kind": "sentinel_check", "request_full_check": True, "D": 0.7,
                "update": kwargs["update"], "trigger_reasons": ["chosen_side_D"]}

    plan = monitor.MonitorPlan(endpoints=(8,), checkpoints=(), sentinel_interval=2,
                               full_interval=4)
    trajectory.run_trajectory(
        row={"trajectory": "tiny", "block": "A", "task": "ipo", "seed": 1},
        directory=tmp_path, policy=harness.policy, optimizer=harness.optimizer,
        scheduler=harness.scheduler, stream=harness.stream, replay_order=harness.replay_order,
        plan=plan, endpoints=(8,), batch_rows=8, microbatch_rows=4,
        task_batch=harness.task_batch, preservation_batch=None, full_check=gate,
        sentinel_check=sentinel, on_endpoint=harness.endpoint, gradient_clip=1.0,
        identity=harness.identity, uses_rejected=True, preservation_family="none",
        preservation_lambda=0.0, max_updates=8, resume=False,
        stream_digests=harness.stream.document())
    # Every sentinel request escalated to a full check, and the sentinel itself
    # never stopped anything.
    assert calls["sentinel"] >= 1 and calls["full"] >= calls["sentinel"]


def test_a_preservation_arm_journals_its_term_diagnostics(tmp_path):
    harness = Harness(tmp_path, updates=4, batch_rows=8, preservation="tail", lam=0.5, seed=16)
    harness.run(updates=4)
    records = [json.loads(line) for line in
               (tmp_path / trajectory.UPDATES_JSONL).read_text(encoding="utf-8").splitlines()
               if line.strip()]
    assert records[0]["preservation_family"] == "tail"
    assert records[0]["preservation_lambda"] == 0.5
    assert records[0]["tail_active_fraction"] is not None
    assert records[0]["preservation_mean"] is not None


def test_journal_records_stay_json_writable_even_with_numpy_values(tmp_path):
    def gate(**kwargs):
        return {"passed": True, "D": np.float64(0.25), "rows": np.int64(7),
                "scores": np.zeros(3), "quantiles": {"0.5": np.float64(1.0)}}

    harness = Harness(tmp_path, updates=2, batch_rows=8, gate=gate, seed=17)
    harness.run(updates=2)
    records = [json.loads(line) for line in
               (tmp_path / trajectory.MONITOR_JSONL).read_text(encoding="utf-8").splitlines()
               if line.strip()]
    verdicts = [record for record in records if record["record_kind"] == "gate_verdict"]
    assert verdicts and verdicts[0]["D"] == 0.25
    assert "scores" not in verdicts[0]


def test_an_interrupted_endpoint_publication_is_completed_by_the_resume(tmp_path):
    """The reproduced defect: endpoints 2 and 4, a kill inside endpoint 2's
    publication, and a resume that started at 3 and skipped endpoint 2 forever.
    """
    published = []

    def exploding(**kwargs):
        if int(kwargs["update"]) == 2 and not published:
            published.append("attempted")
            raise RuntimeError("killed inside the endpoint publication")
        published.append(int(kwargs["update"]))
        return {"update": int(kwargs["update"]), "exposures": dict(kwargs["exposures"])}

    harness = Harness(tmp_path, updates=4, batch_rows=8, seed=30)
    with pytest.raises(RuntimeError, match="killed inside"):
        harness.run(updates=4, endpoints=(2, 4), on_endpoint=exploding)
    state = torch.load(tmp_path / trajectory.RESUME_STATE, map_location="cpu",
                       weights_only=False)
    assert state["progress"]["update"] == 2
    assert state["progress"]["pending_endpoint"]["update"] == 2

    (tmp_path / trajectory.STATUS_JSON).unlink()
    document = harness.run(updates=4, endpoints=(2, 4), on_endpoint=exploding)
    assert document["status"] == trajectory.STATUS_COMPLETED
    assert sorted(int(value) for value in document["endpoints_reached"]) == [2, 4]
    assert document["resume_reconciliation"]["published_endpoint"] == 2
    assert document["endpoints_not_reached"] == {}
    assert published[1] == 2, "endpoint 2 was published from the weights it belongs to"


def test_an_interrupted_run_matches_the_uninterrupted_one_in_every_recorded_field(tmp_path):
    straight = Harness(tmp_path / "straight", updates=6, batch_rows=8, seed=31)
    reference = straight.run(updates=6, endpoints=(2, 6), checkpoints=(4,),
                             on_checkpoint=lambda **kwargs: {"update": kwargs["update"]})
    fired = {"done": False}

    def exploding(**kwargs):
        if int(kwargs["update"]) == 2 and not fired["done"]:
            fired["done"] = True
            raise RuntimeError("killed inside the endpoint publication")
        return {"update": int(kwargs["update"]), "exposures": dict(kwargs["exposures"])}

    interrupted = Harness(tmp_path / "resumed", updates=6, batch_rows=8, seed=31)
    with pytest.raises(RuntimeError):
        interrupted.run(updates=6, endpoints=(2, 6), checkpoints=(4,),
                        on_endpoint=exploding,
                        on_checkpoint=lambda **kwargs: {"update": kwargs["update"]})
    (interrupted.directory / trajectory.STATUS_JSON).unlink()
    resumed = interrupted.run(updates=6, endpoints=(2, 6), checkpoints=(4,),
                              on_endpoint=exploding,
                              on_checkpoint=lambda **kwargs: {"update": kwargs["update"]})
    assert state_digest(interrupted.policy) == state_digest(straight.policy)
    for field in ("updates", "checks", "sentinels", "clipped_updates", "exposures",
                  "stream_position", "replay_position", "checkpoints_saved"):
        assert resumed[field] == reference[field], field
    assert sorted(resumed["endpoints_reached"]) == sorted(reference["endpoints_reached"])
    straight_state = torch.load(straight.directory / trajectory.RESUME_STATE,
                                map_location="cpu", weights_only=False)
    resumed_state = torch.load(interrupted.directory / trajectory.RESUME_STATE,
                               map_location="cpu", weights_only=False)
    assert (trajectory._optimizer_step_count(resumed_state["optimizer"])
            == trajectory._optimizer_step_count(straight_state["optimizer"]))
    assert resumed_state["scheduler"]["last_epoch"] == straight_state["scheduler"]["last_epoch"]


def test_a_tampered_optimizer_half_is_refused_even_with_intact_weights(tmp_path):
    """A model digest alone certifies the weights and nothing else."""
    harness = Harness(tmp_path, updates=2, batch_rows=8, seed=32)
    harness.run(updates=2)
    store = trajectory.TrajectoryStateStore(tmp_path, identity=harness.identity)
    payload = torch.load(store.path(), map_location="cpu", weights_only=False)
    payload["scheduler"]["last_epoch"] = int(payload["scheduler"]["last_epoch"]) + 5
    torch.save(payload, store.path())
    policy = TinyPolicy(seed=32)
    optimizer, scheduler = tiny_optimizer(policy)
    with pytest.raises(ValueError, match="does not reproduce its recorded digest"):
        store.load(policy=policy, optimizer=optimizer, scheduler=scheduler)


def test_a_state_without_a_payload_digest_is_refused(tmp_path):
    harness = Harness(tmp_path, updates=2, batch_rows=8, seed=33)
    harness.run(updates=2)
    store = trajectory.TrajectoryStateStore(tmp_path, identity=harness.identity)
    payload = torch.load(store.path(), map_location="cpu", weights_only=False)
    payload.pop("payload_sha256")
    torch.save(payload, store.path())
    policy = TinyPolicy(seed=33)
    optimizer, scheduler = tiny_optimizer(policy)
    with pytest.raises(ValueError, match="no whole-payload digest"):
        store.load(policy=policy, optimizer=optimizer, scheduler=scheduler)


def test_a_truncated_final_journal_line_is_repaired_before_the_next_append(tmp_path):
    """Appending onto a half-written line makes it a corrupt INTERIOR record."""
    harness = Harness(tmp_path, updates=2, batch_rows=8, seed=34)
    harness.run(updates=2)
    journal = tmp_path / trajectory.UPDATES_JSONL
    with journal.open("a", encoding="utf-8") as stream:
        stream.write('{"record_kind": "update", "update": 3, "exposu')
    repair = trajectory.repair_truncated_journal(journal)
    assert repair["repaired"] is True
    assert repair["dropped_truncated_final_line"] is True
    progress = trajectory.durable_progress(tmp_path)
    assert progress["updates"] == 2
    # And a complete final record that merely lost its newline keeps the record.
    with journal.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"record_kind": "update", "update": 3, "exposures": {}}))
    second = trajectory.repair_truncated_journal(journal)
    assert second["added_missing_newline"] is True
    assert trajectory.durable_progress(tmp_path)["updates"] == 3


def test_the_resume_repairs_the_journal_rather_than_refusing_to_open_it(tmp_path):
    harness = Harness(tmp_path, updates=4, batch_rows=8, seed=35)
    harness.run(updates=2, endpoints=(2,))
    with (tmp_path / trajectory.UPDATES_JSONL).open("a", encoding="utf-8") as stream:
        stream.write('{"record_kind": "update", "upd')
    (tmp_path / trajectory.STATUS_JSON).unlink()
    document = harness.run(updates=4, endpoints=(4,))
    assert document["status"] == trajectory.STATUS_COMPLETED
    assert document["journal_repairs"], "the repair is disclosed, not silent"


def test_tail_max_drop_is_a_maximum_not_a_mean_of_microbatch_maxima(tmp_path):
    """Averaging four microbatch maxima reports a number no replay row produced."""
    harness = Harness(tmp_path, updates=2, batch_rows=8, micro=2, preservation="tail",
                      lam=0.5, seed=36)
    seen = []
    original = harness.preservation_batch

    def spying(rows):
        mean, block = original(rows)
        seen.append(float(block["max_drop_nats"]))
        return mean, block

    harness.preservation_batch = spying
    harness.run(updates=2)
    records = [json.loads(line) for line in
               (tmp_path / trajectory.UPDATES_JSONL).read_text(encoding="utf-8").splitlines()
               if line.strip()]
    first_update = seen[:4]
    assert records[0]["tail_max_drop"] == pytest.approx(max(first_update))
    assert records[0]["tail_max_drop"] >= sum(first_update) / len(first_update)


def test_the_stream_and_replay_cursors_travel_with_the_state(tmp_path):
    replay = Harness(tmp_path / "replay", updates=4, batch_rows=8, preservation="tail",
                     lam=0.5, seed=37)
    document = replay.run(updates=4)
    assert document["stream_position"] == 32
    assert document["replay_position"] == 32
    control = Harness(tmp_path / "control", updates=4, batch_rows=8, seed=37)
    other = control.run(updates=4)
    assert other["stream_position"] == 32
    # A no-preservation control moves no replay stream at all, which is what
    # makes its update identical to one produced by code with no replay term.
    assert other["replay_position"] == 0


def test_the_previous_sentinel_value_is_handed_back_on_resume(tmp_path):
    restored = []
    harness = Harness(tmp_path, updates=4, batch_rows=8, seed=38)
    harness.run(updates=2, endpoints=(2,))
    (tmp_path / trajectory.STATUS_JSON).unlink()
    harness.run(updates=4, endpoints=(4,), restore_sentinel=restored.append)
    assert restored, "the resumed gate was handed the previous sentinel value"


def test_the_no_scaler_reason_describes_float32_training(tmp_path):
    assert "trains entirely in float32" in trajectory.NO_SCALER_REASON
    assert "diagnostic accumulation" in trajectory.NO_SCALER_REASON
    assert "float64 training" not in trajectory.NO_SCALER_REASON


def test_terminal_status_is_not_silently_rewritten(tmp_path):
    harness = Harness(tmp_path, updates=2, batch_rows=8, seed=18)
    document = harness.run(updates=2)
    with pytest.raises(ValueError, match="not rewritten"):
        trajectory.write_terminal_status(tmp_path, dict(document, status="completed",
                                                        updates=999))
