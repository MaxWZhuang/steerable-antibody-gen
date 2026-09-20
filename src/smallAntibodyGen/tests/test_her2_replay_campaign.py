"""The campaign: one writer, exact exposures, and statuses that cannot flatter a run.

The failures covered here are the ones that produce a finished-looking campaign:
a second writer interleaving into the same directory, an interrupted trajectory
quietly resumed, a breached gate that still leaves a selectable endpoint, a
checkpoint marker written before the bytes were validated, and a "healthy" launch
claimed from a process that merely started.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from smallAntibodyGen.experiments import her2_replay as replay
from smallAntibodyGen.experiments import her2_replay_campaign as campaign
from smallAntibodyGen.experiments import her2_replay_report as report
from smallAntibodyGen.experiments import her2_replay_streams as streams
from smallAntibodyGen.experiments import her2_support_paths as paths


CONFIG = {"screen": {"tasks": ["continued_sft", "ipo"],
                     "replay_lambdas": [0.0, 0.01, 0.1, 1.0, 10.0, 100.0],
                     "parent_seeds": [20260918, 20260919, 20260920]}}


# ---------------------------------------------------------------------------
# the queue
# ---------------------------------------------------------------------------

def test_the_queue_is_the_declared_grid_with_controls_first_per_seed():
    queue = campaign.build_queue(CONFIG)
    assert len(queue) == 36
    first_seed = [row for row in queue if row["seed"] == 20260918]
    assert first_seed[0]["replay_lambda"] == 0.0 and first_seed[0]["is_control"]
    assert first_seed[1]["replay_lambda"] == 0.0
    assert [row["replay_lambda"] for row in first_seed][:4] == [0.0, 0.0, 0.01, 0.01]
    assert len({row["trajectory"] for row in queue}) == 36


def test_the_queue_order_is_a_fixed_function_of_the_config():
    assert campaign.build_queue(CONFIG) == campaign.build_queue(CONFIG)


def test_a_grid_without_the_matched_control_is_refused():
    broken = {"screen": dict(CONFIG["screen"], replay_lambdas=[0.1, 1.0])}
    with pytest.raises(ValueError, match="lambda = 0 control"):
        campaign.build_queue(broken)


def test_arm_ids_are_filename_safe_and_stable():
    assert campaign.arm_id("ipo", 0.01) == "ipo_lambda0p01"
    assert campaign.arm_id("continued_sft", 10.0) == "continued_sft_lambda10"
    assert campaign.trajectory_id("ipo", 0.0, 20260918) == "ipo_lambda0_seed20260918"


# ---------------------------------------------------------------------------
# one writer at a time
# ---------------------------------------------------------------------------

def test_a_second_writer_is_refused_while_the_lock_is_held(tmp_path):
    """Also asserts the lock is NOT re-entrant within one process.

    That is the property the campaign depends on: ``owner_is_gone`` decides
    whether a begun trajectory is marked incomplete, and a re-entrant lock would
    make a live campaign's trajectories look abandoned. A platform where this
    fails must fail here, loudly, rather than in the run directory.
    """
    first = campaign.CampaignLock(tmp_path / "campaign.lock")
    with first.held():
        second = campaign.CampaignLock(tmp_path / "campaign.lock")
        with pytest.raises(ValueError, match="One writer at a time"):
            second.acquire()
        assert campaign.owner_is_gone(tmp_path / "campaign.lock") is False
    assert campaign.owner_is_gone(tmp_path / "campaign.lock") is True


def test_the_lock_records_its_owner_as_evidence_not_as_the_liveness_test(tmp_path):
    lock = campaign.CampaignLock(tmp_path / "campaign.lock")
    with lock.held():
        owner = lock.recorded_owner()
    assert owner["pid"] and owner["session"] and owner["acquired_at"]
    assert "Liveness is decided by the OS lock" in owner["note"]


def test_observing_the_lock_creates_nothing_and_rewrites_nothing(tmp_path):
    """A status probe must not become a write into the run it is looking at.

    The reproduction: ``owner_is_gone`` used to call ``acquire``, which creates the
    lock file if it is missing and replaces ``.owner.json`` with the *observer's*
    session. A dashboard refreshing every five seconds would rewrite the recorded
    provenance of the campaign it was displaying.
    """
    lock_path = tmp_path / "campaign.lock"
    assert campaign.lock_state(lock_path)["state"] == campaign.LOCK_MISSING
    assert not lock_path.exists(), "observing a missing lock does not create it"

    holder = campaign.CampaignLock(lock_path)
    with holder.held():
        owner = dict(holder.recorded_owner())
        owner_bytes = holder.owner_path().read_bytes()
        owner_mtime = holder.owner_path().stat().st_mtime_ns
        lock_mtime = lock_path.stat().st_mtime_ns
        observed = campaign.lock_state(lock_path)
        assert observed["state"] == campaign.LOCK_HELD
        assert observed["recorded_owner"]["session"] == owner["session"]
        assert campaign.owner_is_gone(lock_path) is False
        assert holder.owner_path().read_bytes() == owner_bytes
        assert holder.owner_path().stat().st_mtime_ns == owner_mtime
        assert lock_path.stat().st_mtime_ns == lock_mtime
    released = campaign.lock_state(lock_path)
    assert released["state"] == campaign.LOCK_RELEASED
    assert released["recorded_owner"]["session"] == owner["session"], (
        "the owner record still names the process that held it, not the observer")


def test_an_unreadable_lock_is_never_reported_as_nobody_being_there(tmp_path, monkeypatch):
    lock_path = tmp_path / "campaign.lock"
    lock_path.write_bytes(b"\0")

    def refuse(*args, **kwargs):
        raise PermissionError("locked by another user")

    monkeypatch.setattr(Path, "open", refuse)
    observed = campaign.lock_state(lock_path)
    assert observed["state"] == campaign.LOCK_INACCESSIBLE
    monkeypatch.undo()
    assert campaign.owner_is_gone(lock_path) is True
    monkeypatch.setattr(Path, "open", refuse)
    assert campaign.owner_is_gone(lock_path) is False, (
        "'I could not tell' must not become 'the owner is gone': that is the answer that "
        "marks a live campaign's trajectories incomplete")


def test_a_displaced_owner_record_is_preserved_before_the_next_one_replaces_it(tmp_path):
    lock_path = tmp_path / "campaign.lock"
    first = campaign.CampaignLock(lock_path)
    with first.held():
        original = dict(first.recorded_owner())
    second = campaign.CampaignLock(lock_path)
    with second.held():
        assert second.recorded_owner()["session"] != original["session"]
        preserved = second.previous_owners()
    assert [row["session"] for row in preserved] == [original["session"]]
    assert preserved[0]["record_kind"] == "displaced_campaign_lock_owner"


def test_the_heartbeat_carries_wall_and_monotonic_time(tmp_path):
    beat = campaign.Heartbeat(tmp_path / "heartbeat.json", owner=campaign.owner_identity(),
                              every=0.0)
    record = beat.beat(trajectory="x", update=3)
    assert record["update"] == 3 and record["monotonic"] > 0 and record["heartbeat_at"]


# ---------------------------------------------------------------------------
# state files
# ---------------------------------------------------------------------------

class TinyModule(torch.nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.prefix = torch.nn.Parameter(torch.randn(1, generator=generator))
        self.table = torch.nn.Parameter(torch.randn(10, 20, generator=generator))

    def logits(self, index):
        rows = torch.as_tensor(np.asarray(index), dtype=torch.long)
        shift = torch.nn.functional.one_hot(rows, num_classes=20).float()
        return self.table.unsqueeze(0) + self.prefix * shift


class TinyPolicy:
    """The ``CorePolicy`` seam the campaign actually uses, on a two-parameter model."""

    core_length = 10

    def __init__(self, model):
        self.model = model
        self.device = torch.device("cpu")

    def core_index(self, index):
        return torch.as_tensor(np.asarray(index), dtype=torch.long)

    def token_ids(self, index):
        return self.core_index(index)

    def core_logits(self, core_ids):
        return self.model.logits(core_ids)

    def position_log_probs(self, index):
        values = self.core_index(index)
        return torch.log_softmax(self.core_logits(values), dim=-1).gather(
            2, values.unsqueeze(-1)).squeeze(-1)

    def sequence_log_probs(self, index):
        return self.position_log_probs(index).sum(dim=1)


def test_the_saver_writes_different_files_and_validates_the_bytes(tmp_path):
    saver = campaign.ReplayStateSaver(tmp_path, TinyModule(), metadata={"trajectory": "t"})
    passing = saver.save_last_passing(update=25, exposures={"chosen": 1600}, check=2)
    failed = saver.save_failed(update=26, exposures={"chosen": 1664}, stop_reason="breach")
    endpoint = saver.save_endpoint(update=1000, exposures={"chosen": 64000})
    assert passing["file"] == "last_passing.pt" and failed["file"] == "failed_state.pt"
    assert endpoint["file"] == "endpoint_update1000.pt"
    assert (tmp_path / "last_passing.pt").is_file() and (tmp_path / "failed_state.pt").is_file()
    for record in (passing, failed, endpoint):
        assert record["state_sha256"] and record["sha256"] and record["bytes"] > 0
        # The behaviour, not the sentence: the digest in the record is the one the
        # bytes on disk produce when they are loaded and re-hashed.
        restored = torch.load(tmp_path / record["file"], map_location="cpu", weights_only=True)
        assert campaign.state_dict_digest(restored["state"]) == record["state_sha256"]
        assert restored["state_sha256"] == record["state_sha256"]
        assert "re-digested from the restored tensors" in record["validated"]


def test_a_failure_save_cannot_land_in_the_file_whose_name_says_passing(tmp_path):
    """The one lie this design exists to prevent, checked by behaviour not by reading."""
    saver = campaign.ReplayStateSaver(tmp_path, TinyModule())
    saver.save_failed(update=3, exposures={"chosen": 12}, stop_reason="breach")
    assert (tmp_path / campaign.FAILED_STATE).is_file()
    assert not (tmp_path / campaign.LAST_PASSING).exists(), (
        "save_failed writes only the failure slot; there is no argument that redirects it")
    payload = torch.load(tmp_path / campaign.FAILED_STATE, map_location="cpu", weights_only=True)
    assert payload["kind"] == "failed_state" and payload["stop_reason"] == "breach"
    assert Path(campaign.__file__).read_text(encoding="utf-8").count(
        "def save_failed") == 1


def test_a_truncated_checkpoint_never_gets_a_completion_record(tmp_path, monkeypatch):
    """The marker is written after the bytes re-read, not after ``torch.save`` returns."""
    saver = campaign.ReplayStateSaver(tmp_path, TinyModule())
    real_save = torch.save                       # retained BEFORE the patch, or this recurses

    def broken_save(payload, path):
        real_save({"schema_version": payload["schema_version"], "kind": payload["kind"],
                   "update": payload["update"], "exposures": payload["exposures"],
                   "state": {}, "state_sha256": "0" * 64}, path)

    monkeypatch.setattr(torch, "save", broken_save)
    with pytest.raises(ValueError, match="did not validate before publication"):
        saver.save_endpoint(update=1000, exposures={"chosen": 64000})
    assert not (tmp_path / "endpoint_update1000.pt").exists(), (
        "a checkpoint that failed validation is never published to its destination")


def test_a_corrupted_tensor_with_an_intact_digest_field_is_refused(tmp_path, monkeypatch):
    """The reproduction the reviewer ran: valid bytes, changed weights, same digest string.

    Comparing the stored ``state_sha256`` string to the expected string proves only
    that the string was copied. This save serializes perfectly well and carries a
    tensor that is not the one the update produced.
    """
    model = TinyModule(seed=7)
    saver = campaign.ReplayStateSaver(tmp_path, model)
    real_save = torch.save

    def corrupting_save(payload, path):
        state = {name: tensor.clone() for name, tensor in payload["state"].items()}
        first = sorted(state)[0]
        state[first] = state[first] + 1.0                 # a different tensor, same digest field
        real_save(dict(payload, state=state), path)

    monkeypatch.setattr(torch, "save", corrupting_save)
    with pytest.raises(ValueError, match="did not validate before publication"):
        saver.save_endpoint(update=2000, exposures={"chosen": 128000})
    assert not (tmp_path / "endpoint_update2000.pt").exists()


def test_a_checkpoint_whose_metadata_names_another_trajectory_is_refused(tmp_path, monkeypatch):
    """The reviewer's second reproduction: perfect tensors, rewritten identity.

    ``_inspect`` excluded metadata from its comparison, so a save that kept every
    tensor while replacing ``trajectory`` and ``freeze`` with ``WRONG`` validated
    and was published. Tensors do not say whose they are.
    """
    model = TinyModule(seed=11)
    saver = campaign.ReplayStateSaver(
        tmp_path, model, metadata={"trajectory": "ipo_lambda1_seed20260918", "freeze": "abc123",
                                   "started_at": "2026-09-20T00:00:00+00:00"})
    real_save = torch.save

    def rewriting_save(payload, path):
        real_save(dict(payload, metadata={"trajectory": "WRONG", "freeze": "WRONG",
                                          "started_at": "2026-09-20T00:00:00+00:00"}), path)

    monkeypatch.setattr(torch, "save", rewriting_save)
    with pytest.raises(ValueError, match="claims a different identity"):
        saver.save_endpoint(update=3000, exposures={"chosen": 192000})
    assert not (tmp_path / "endpoint_update3000.pt").exists()


def test_write_once_reverification_compares_the_identity_and_ignores_the_clock(tmp_path):
    """A retry stays idempotent; another trajectory reusing the path does not pass."""
    model = TinyModule(seed=12)
    identity = {"trajectory": "ipo_lambda1_seed20260918", "freeze": "abc123"}
    first = campaign.ReplayStateSaver(
        tmp_path, model, metadata=dict(identity, started_at="2026-09-20T00:00:00+00:00")
    ).save_endpoint(update=1000, exposures={"chosen": 64000})
    # The same trajectory under a later process clock. ``started_at`` says when the
    # fit began, not which fit it is, so it is not an identity change.
    again = campaign.ReplayStateSaver(
        tmp_path, model, metadata=dict(identity, started_at="2026-09-20T09:00:00+00:00")
    ).save_endpoint(update=1000, exposures={"chosen": 64000})
    assert again["sha256"] == first["sha256"] and "never replaced" in again["validated"]
    other = campaign.ReplayStateSaver(
        tmp_path, model, metadata=dict(identity, trajectory="ipo_lambda10_seed20260918"))
    with pytest.raises(ValueError, match="claims a different identity"):
        other.save_endpoint(update=1000, exposures={"chosen": 64000})
    assert paths.sha256_file(tmp_path / "endpoint_update1000.pt") == first["sha256"]


def test_a_failed_readback_preserves_the_previous_passing_checkpoint(tmp_path, monkeypatch):
    """``os.replace`` after validation, not before: a failed save loses nothing."""
    model = TinyModule(seed=8)
    saver = campaign.ReplayStateSaver(tmp_path, model)
    first = saver.save_last_passing(update=25, exposures={"chosen": 1600}, check=1)
    original = (tmp_path / campaign.LAST_PASSING).read_bytes()

    with torch.no_grad():
        model.table.add_(0.5)                             # the model moved on
    real_load = torch.load
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: (_ for _ in ()).throw(
        OSError("disk read failed")))
    with pytest.raises(ValueError, match="was not touched"):
        saver.save_last_passing(update=50, exposures={"chosen": 3200}, check=2)
    monkeypatch.setattr(torch, "load", real_load)
    assert (tmp_path / campaign.LAST_PASSING).read_bytes() == original, (
        "the previous passing checkpoint must survive a save that could not be read back")
    assert paths.sha256_file(tmp_path / campaign.LAST_PASSING) == first["sha256"]
    assert not list(tmp_path.glob("*.tmp")), "the failed temporary is cleaned up"


def test_an_endpoint_snapshot_is_write_once_and_identical_reverification_is_allowed(tmp_path):
    model = TinyModule(seed=9)
    saver = campaign.ReplayStateSaver(tmp_path, model)
    first = saver.save_endpoint(update=1000, exposures={"chosen": 64000})
    again = saver.save_endpoint(update=1000, exposures={"chosen": 64000})
    assert again["sha256"] == first["sha256"] and "never replaced" in again["validated"]
    with torch.no_grad():
        model.prefix.add_(1.0)
    with pytest.raises(ValueError, match="not the weights this update produced"):
        saver.save_endpoint(update=1000, exposures={"chosen": 64000})
    assert paths.sha256_file(tmp_path / "endpoint_update1000.pt") == first["sha256"]


def test_the_rolling_last_passing_state_is_the_only_one_that_is_replaced(tmp_path):
    model = TinyModule(seed=10)
    saver = campaign.ReplayStateSaver(tmp_path, model)
    first = saver.save_last_passing(update=25, exposures={"chosen": 1600}, check=1)
    with torch.no_grad():
        model.table.add_(0.25)
    second = saver.save_last_passing(update=50, exposures={"chosen": 3200}, check=2)
    assert second["sha256"] != first["sha256"] and second["rolling"] is True
    assert "rolling latest-passing state" in second["note"]


def test_the_saver_registers_write_once_checkpoints_and_not_the_rolling_one(tmp_path):
    """What a later verification can be held to: endpoints and failures, not the roller.

    One boundary, in the saver: it knows which write is rolling and does not offer
    it to the ledger at all. The production callback in ``fit_one`` asserts the
    same thing from the other side instead of silently skipping, so the two halves
    cannot drift into describing different contracts.
    """
    registered = []
    saver = campaign.ReplayStateSaver(
        tmp_path, TinyModule(seed=6),
        register=lambda **fields: registered.append((fields["name"], fields["kind"])))
    saver.save_last_passing(update=25, exposures={"chosen": 1600}, check=1)
    saver.save_endpoint(update=1000, exposures={"chosen": 64000})
    saver.save_failed(update=1001, exposures={"chosen": 64064}, stop_reason="breach")
    assert ("endpoint_update1000.pt", "exposure_endpoint") in registered
    assert (campaign.FAILED_STATE, "failed_state") in registered
    assert all(name != campaign.LAST_PASSING for name, _ in registered), (
        "the rolling state is not offered to a write-once ledger, which would refuse its second "
        "write; the saver is where that boundary lives")
    # And a second rolling save still offers nothing, which is the case a ledger
    # would actually reject.
    with torch.no_grad():
        saver.model.table.add_(0.5)
    saver.save_last_passing(update=50, exposures={"chosen": 3200}, check=2)
    assert len(registered) == 2


def test_the_restored_digest_rule_is_the_one_the_policy_module_uses():
    """The recomputed digest has to be comparable to the one recorded at save time."""
    from smallAntibodyGen.experiments import her2_policy as policy_lib
    module = TinyModule(seed=5)
    assert campaign.state_dict_digest(module.state_dict()) == policy_lib.state_digest(module)


# ---------------------------------------------------------------------------
# statuses
# ---------------------------------------------------------------------------

def terminal(status, **fields):
    return dict({"schema_version": campaign.TRAJECTORY_SCHEMA,
                 "record_kind": "trajectory_status", "trajectory": "t", "status": status,
                 "updates": 10, "checks": 1, "exposures": {"chosen": 640}}, **fields)


def test_a_terminal_status_is_written_once_and_never_contradicted(tmp_path):
    campaign.write_terminal_status(tmp_path, terminal(campaign.STATUS_COMPLETED))
    campaign.write_terminal_status(tmp_path, terminal(campaign.STATUS_COMPLETED))
    with pytest.raises(ValueError, match="already records a different terminal outcome"):
        campaign.write_terminal_status(tmp_path, terminal(campaign.STATUS_STOPPED))


def test_a_non_terminal_status_cannot_be_written_as_terminal(tmp_path):
    with pytest.raises(ValueError, match="not a terminal status"):
        campaign.write_terminal_status(tmp_path, terminal(campaign.STATUS_RUNNING))


def begun(directory, *, updates, checks, endpoints=(), reached_updates=(), trailing=""):
    """A trajectory directory as a killed process would leave it."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / campaign.IDENTITY_JSON).write_text(json.dumps(
        {"trajectory": directory.name, "arm_id": "ipo_lambda1", "task": "ipo",
         "replay_lambda": 1.0, "seed": 20260918}), encoding="utf-8")
    (directory / campaign.PROGRESS_JSON).write_text(json.dumps(
        {"update": updates["progress"], "checks": checks,
         "exposures": {"chosen": updates["progress"] * 64},
         "endpoints_reached": [str(value) for value in reached_updates],
         "declared_endpoints": list(endpoints)}), encoding="utf-8")
    lines = [json.dumps({"record_kind": "update", "update": value,
                         "exposures": {"chosen": value * 64, "rejected": value * 64,
                                       "replay": value * 64}})
             for value in range(1, updates["journal"] + 1)]
    (directory / campaign.UPDATES_JSONL).write_text(
        "\n".join(lines) + "\n" + trailing, encoding="utf-8", newline="\n")
    (directory / campaign.MONITOR_JSONL).write_text(
        "".join(json.dumps({"record_kind": "gate_verdict", "update": value, "passed": True}) + "\n"
                for value in range(1, checks + 1)), encoding="utf-8", newline="\n")
    (directory / campaign.ENDPOINTS_JSONL).write_text(
        "".join(json.dumps({"record_kind": "exposure_endpoint", "update": value,
                            "macro_average_precision": 0.5,
                            "checkpoint": {"sha256": "a" * 64, "state_sha256": "b" * 64}}) + "\n"
                for value in reached_updates), encoding="utf-8", newline="\n")


def test_durable_progress_reads_the_journals_rather_than_the_progress_file(tmp_path):
    """The reviewer's reproduction: progress says 1,000 and the journal says 1,024."""
    directory = tmp_path / "trajectories" / "ipo_lambda1_seed20260918"
    begun(directory, updates={"progress": 1000, "journal": 1024}, checks=41,
          endpoints=(1000, 2000), reached_updates=(1000,))
    durable = campaign.durable_progress(directory)
    assert durable["updates"] == 1024
    assert durable["exposures"]["chosen"] == 1024 * 64
    assert sorted(durable["endpoints_reached"]) == ["1000"]
    assert durable["truncated_trailing_update_line"] is False


def test_a_partial_final_journal_line_is_dropped_and_reported(tmp_path):
    directory = tmp_path / "trajectories" / "t"
    begun(directory, updates={"progress": 4, "journal": 8}, checks=1,
          trailing='{"record_kind": "update", "update": 9, "exposu')
    durable = campaign.durable_progress(directory)
    assert durable["updates"] == 8, "an update whose record never finished writing is not completed"
    assert durable["truncated_trailing_update_line"] is True


def test_a_complete_final_record_without_its_newline_is_kept_and_reported_as_such(tmp_path):
    """Two different states, reported as two.

    A final line that parses whole is a completed record and is counted; the
    missing terminator says the writer was killed between the object and the
    newline. Reporting that kept record under "a partial final line is dropped"
    described the file wrongly, while the count itself was right.
    """
    directory = tmp_path / "trajectories" / "t"
    begun(directory, updates={"progress": 4, "journal": 8}, checks=1,
          trailing=json.dumps({"record_kind": "update", "update": 9,
                               "exposures": {"chosen": 576}}))
    durable = campaign.durable_progress(directory)
    assert durable["updates"] == 9, "the record is whole, so the update completed"
    assert durable["truncated_trailing_update_line"] is False
    assert durable["unterminated_final_update_line"] is True
    assert "is kept" in durable["basis"] and "unterminated" in durable["basis"]


def test_a_journal_damaged_in_the_middle_is_corruption_not_an_interrupted_write(tmp_path):
    directory = tmp_path / "trajectories" / "t"
    begun(directory, updates={"progress": 2, "journal": 3}, checks=1)
    path = directory / campaign.UPDATES_JSONL
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[1] = "{broken"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="not the final line"):
        campaign.durable_progress(directory)


def test_an_interrupted_trajectory_with_no_journalled_update_reports_zero_not_the_last_check(
        tmp_path):
    """The progress file is not promoted when the journals hold nothing.

    "The journals recorded no completed update" and "the last check said 1,000"
    are different statements, and only the first is durable evidence. Both are
    reported, in fields that say which is which.
    """
    directory = tmp_path / "trajectories" / "t"
    begun(directory, updates={"progress": 1000, "journal": 0}, checks=41, endpoints=(1000,))
    document = campaign.mark_begun_incomplete(directory, reason="the owner is gone")
    assert document["updates"] == 0 and document["exposures"] == {}
    assert document["progress_file_updates"] == 1000
    assert document["progress_file_exposures"]["chosen"] == 1000 * 64
    assert document["checks"] == 41 and document["progress_file_checks"] == 41
    assert "including when they recovered nothing" in document["recovery"]


def test_a_begun_trajectory_without_a_terminal_status_becomes_incomplete(tmp_path):
    directory = tmp_path / "trajectories" / "ipo_lambda1_seed20260918"
    begun(directory, updates={"progress": 1000, "journal": 1024}, checks=41,
          endpoints=(1000, 2000, 3750), reached_updates=(1000,))
    document = campaign.mark_begun_incomplete(directory, reason="the owner is gone")
    assert document["status"] == campaign.STATUS_INCOMPLETE
    assert document["updates"] == 1024, "the completed updates are the journalled ones"
    assert document["progress_file_updates"] == 1000
    assert document["exposures"]["chosen"] == 1024 * 64
    # The reached endpoint survives as a RECORD, with its checkpoint identity.
    assert sorted(document["endpoints_reached"]) == ["1000"]
    assert document["endpoints_reached"]["1000"]["checkpoint"]["state_sha256"] == "b" * 64
    assert document["endpoints_reached"]["1000"]["record_kind"] == "exposure_endpoint"
    assert sorted(document["endpoints_not_reached"]) == ["2000", "3750"]
    # No resume, asserted as behaviour and as a stated policy rather than by wording.
    assert campaign.mark_begun_incomplete(directory, reason="again") is None, (
        "a trajectory with a terminal status is not re-marked")
    assert "not restarted" in document["resume_policy"]
    assert "no partial state is adopted" in document["resume_policy"]
    assert document["status"] not in (campaign.STATUS_COMPLETED, campaign.STATUS_RUNNING)


def test_an_interrupted_endpoint_checkpoint_alone_is_not_a_reached_endpoint(tmp_path):
    """A ``.pt`` left behind by an interrupted save has no record and is not counted."""
    directory = tmp_path / "trajectories" / "t"
    begun(directory, updates={"progress": 1000, "journal": 1000}, checks=41,
          endpoints=(1000,), reached_updates=())
    (directory / "endpoint_update1000.pt").write_bytes(b"not a record")
    document = campaign.mark_begun_incomplete(directory, reason="killed")
    assert document["endpoints_reached"] == {}
    assert sorted(document["endpoints_not_reached"]) == ["1000"]


def test_an_unstarted_trajectory_is_not_marked_incomplete(tmp_path):
    directory = tmp_path / "trajectories" / "never_started"
    directory.mkdir(parents=True)
    assert campaign.mark_begun_incomplete(directory, reason="x") is None


def test_campaign_state_reads_queued_running_and_interrupted_from_artifacts(tmp_path):
    queue = campaign.build_queue(CONFIG)[:3]
    directory = tmp_path / "trajectories" / queue[0]["trajectory"]
    directory.mkdir(parents=True)
    (directory / campaign.PROGRESS_JSON).write_text(json.dumps({"update": 5, "checks": 1}),
                                                     encoding="utf-8")
    campaign.write_terminal_status(tmp_path / "trajectories" / queue[1]["trajectory"],
                                   terminal(campaign.STATUS_STOPPED,
                                            stop_reason="parent_relative_likelihood_breach"))
    lock = tmp_path / "campaign.lock"
    state = campaign.campaign_state(tmp_path, queue, lock_path=lock)
    rows = {row["trajectory"]: row for row in state["trajectories"]}
    assert rows[queue[0]["trajectory"]]["status"] == campaign.STATUS_INCOMPLETE
    assert rows[queue[1]["trajectory"]]["status"] == campaign.STATUS_STOPPED
    assert rows[queue[2]["trajectory"]]["status"] == campaign.STATUS_QUEUED
    assert state["writer_alive"] is False

    holder = campaign.CampaignLock(lock)
    with holder.held():
        live = campaign.campaign_state(tmp_path, queue, lock_path=lock)
    assert live["trajectories"][0]["status"] == campaign.STATUS_RUNNING
    assert live["lock"]["state"] == campaign.LOCK_HELD


def test_campaign_state_counts_a_live_trajectory_from_its_journals(tmp_path):
    queue = campaign.build_queue(CONFIG)[:1]
    directory = tmp_path / "trajectories" / queue[0]["trajectory"]
    begun(directory, updates={"progress": 1000, "journal": 1024}, checks=41,
          endpoints=(1000, 2000), reached_updates=(1000,))
    lock = tmp_path / "campaign.lock"
    holder = campaign.CampaignLock(lock)
    with holder.held():
        state = campaign.campaign_state(tmp_path, queue, lock_path=lock)
    row = state["trajectories"][0]
    assert row["status"] == campaign.STATUS_RUNNING
    assert row["updates"] == 1024 and row["exposures"]["chosen"] == 1024 * 64
    assert row["endpoints_reached"] == ["1000"]
    assert row["counters_from"] == "its append-only journals"


def test_launch_evidence_is_not_displayed_as_current_health_after_a_failure():
    evidence = {"healthy": True, "first_update_completed": 1, "first_check_update": 1,
                "basis": "read back from the trajectory's own journals"}
    running = campaign.launch_health(evidence, writer_state=campaign.LOCK_HELD)
    assert running["status"] == campaign.HEALTH_RUNNING
    failed = campaign.launch_health(evidence, writer_state=campaign.LOCK_HELD,
                                    failed_trajectories=1)
    assert failed["status"] == campaign.HEALTH_FAILED_AFTER_LAUNCH
    assert failed["first_update_completed"] == 1, "the historical evidence is preserved"
    dead = campaign.launch_health(evidence, writer_state=campaign.LOCK_RELEASED)
    assert dead["status"] == campaign.HEALTH_STOPPED_AFTER_LAUNCH
    unknown = campaign.launch_health(evidence, writer_state=campaign.LOCK_INACCESSIBLE)
    assert unknown["status"] == campaign.HEALTH_LAUNCH_VERIFIED
    nothing = campaign.launch_health({"healthy": False, "basis": "no update yet"},
                                     writer_state=campaign.LOCK_HELD)
    assert nothing["status"] == campaign.HEALTH_NOT_ESTABLISHED


def test_health_needs_a_completed_update_and_a_passed_check(tmp_path):
    queue = campaign.build_queue(CONFIG)[:1]
    directory = tmp_path / "trajectories" / queue[0]["trajectory"]
    directory.mkdir(parents=True)
    assert campaign.first_update_and_check(tmp_path, queue)["healthy"] is False
    (directory / campaign.UPDATES_JSONL).write_text(
        json.dumps({"record_kind": "update", "update": 1, "weighted_total": 1.5}) + "\n",
        encoding="utf-8")
    (directory / campaign.MONITOR_JSONL).write_text(
        json.dumps({"record_kind": "gate_verdict", "update": 1, "passed": False,
                    "D": 4.2}) + "\n", encoding="utf-8")
    failed = campaign.first_update_and_check(tmp_path, queue)
    assert failed["first_update_completed"] == 1 and failed["healthy"] is False
    (directory / campaign.MONITOR_JSONL).write_text(
        json.dumps({"record_kind": "gate_verdict", "update": 1, "passed": True,
                    "D": 0.02}) + "\n", encoding="utf-8")
    healthy = campaign.first_update_and_check(tmp_path, queue)
    assert healthy["healthy"] is True and healthy["first_check_passed"] is True


# ---------------------------------------------------------------------------
# the trajectory loop
# ---------------------------------------------------------------------------

def tiny_run(tmp_path, *, replay_lambda, gate_passes, endpoints=(2, 4), updates=4,
             batch_rows=4, microbatch_rows=3):
    """Run the real loop on a two-parameter model and a synthetic stream."""
    policy = TinyPolicy(TinyModule(seed=1))
    optimizer = torch.optim.AdamW(policy.model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    generator = np.random.default_rng(5)
    cores = generator.integers(0, 20, size=(64, 10)).astype(np.int8)
    logs = torch.log_softmax(torch.randn(64, 10, 20, generator=torch.Generator().manual_seed(2)),
                             dim=-1)
    probabilities = logs.exp()

    class Stream:
        cycle_of_position = np.zeros(updates * batch_rows, dtype=np.int32)

        def batch(self, update):
            start = (update - 1) * batch_rows
            rows = np.arange(start, start + batch_rows) % 64
            return rows, rows

    def task_batch(chosen_rows, rejected_rows):
        return replay.task_term("continued_sft",
                                policy_chosen=policy.sequence_log_probs(cores[chosen_rows]))

    def replay_batch(rows):
        student = replay.student_log_probabilities(policy, cores[rows])
        return replay.replay_term(probabilities[rows], logs[rows], student)

    checks = []

    def gate_check(*, update, reason, exposures):
        passed = gate_passes(update)
        checks.append(update)
        return {"passed": passed, "D": 0.1 if passed else 4.2, "update": update,
                "reason": reason,
                "stop_reason": None if passed else "parent_relative_likelihood_breach"}

    def preservation(*, update, reason):
        return {"available": True, "forward_kl": {"mean": 0.01}}

    reached = []

    def on_endpoint(*, update, exposures, checkpoint, gate, preservation, ledger):
        reached.append(update)
        return {"record_kind": "exposure_endpoint", "update": update,
                "chosen_exposures": exposures["chosen"], "checkpoint": checkpoint}

    row = {"trajectory": "t", "arm_id": "a", "task": "continued_sft",
           "replay_lambda": replay_lambda, "seed": 1}
    document = campaign.run_trajectory(
        row=row, directory=tmp_path, policy=policy, optimizer=optimizer, scheduler=scheduler,
        stream=Stream(), replay_order=np.arange(updates * batch_rows) % 64,
        cadence=streams.UpdateCadence(first_update=1, interval=2, endpoints=endpoints),
        endpoints=endpoints, batch_rows=batch_rows, microbatch_rows=microbatch_rows,
        task_batch=task_batch, replay_batch=replay_batch, gate_check=gate_check,
        preservation=preservation, on_endpoint=on_endpoint, gradient_clip=1.0,
        identity={"trajectory": "t"}, max_updates=updates)
    return document, checks, reached


def test_a_completed_trajectory_reaches_every_endpoint_with_exact_exposures(tmp_path):
    document, checks, reached = tiny_run(tmp_path, replay_lambda=1.0,
                                         gate_passes=lambda update: True)
    assert document["status"] == campaign.STATUS_COMPLETED
    assert document["updates"] == 4
    assert document["exposures"]["chosen"] == 16
    assert document["exposures"]["replay"] == 16
    assert document["exposures"]["rejected"] == 0, "continued SFT performs no rejected inference"
    assert reached == [2, 4] and sorted(document["endpoints_reached"]) == ["2", "4"]
    assert document["endpoints_not_reached"] == {}


def test_a_zero_lambda_control_consumes_no_replay_rows(tmp_path):
    document, _, _ = tiny_run(tmp_path, replay_lambda=0.0, gate_passes=lambda update: True)
    assert document["exposures"]["replay"] == 0
    lines = [json.loads(line) for line in
             (tmp_path / campaign.UPDATES_JSONL).read_text(encoding="utf-8").splitlines()]
    assert all(row["replay_loss"] is None for row in lines)
    assert all(row["replay_lambda"] == 0.0 for row in lines)


def test_a_gate_breach_stops_the_trajectory_and_reaches_no_endpoint(tmp_path):
    document, checks, reached = tiny_run(tmp_path, replay_lambda=1.0,
                                         gate_passes=lambda update: update < 2)
    assert document["status"] == campaign.STATUS_STOPPED
    assert document["stop_reason"] == "parent_relative_likelihood_breach"
    assert reached == [], "a failed gate never produces a selectable nominal endpoint"
    assert sorted(document["endpoints_not_reached"]) == ["2", "4"]
    assert (tmp_path / campaign.FAILED_STATE).is_file()
    assert not (tmp_path / "endpoint_update2.pt").is_file()
    snapshots = [json.loads(line) for line in
                 (tmp_path / campaign.MONITOR_JSONL).read_text(encoding="utf-8").splitlines()]
    breach = [row for row in snapshots if row.get("record_kind") == "snapshot"][-1]
    assert breach["failed_state"]["file"] == campaign.FAILED_STATE
    assert "never a reached endpoint" in breach["consequence"]


def test_the_cadence_and_the_endpoints_are_checked_without_duplicates(tmp_path):
    """Interval 2 with endpoints at 2 and 4: update 3 is off cadence and 2/4 coincide."""
    _, checks, _ = tiny_run(tmp_path, replay_lambda=0.0, gate_passes=lambda update: True)
    assert checks == [1, 2, 4] and len(checks) == len(set(checks))


def test_progress_is_on_disk_before_the_trajectory_ends(tmp_path):
    tiny_run(tmp_path, replay_lambda=0.0, gate_passes=lambda update: True)
    progress = json.loads((tmp_path / campaign.PROGRESS_JSON).read_text(encoding="utf-8"))
    assert progress["endpoints_reached"] == ["2", "4"]
    assert progress["exposures"]["chosen"] == 16


def failing_run(tmp_path, *, fail_at, endpoints=(2,), updates=8, interval=5, journal_factory=None):
    """The real loop with a task callback that raises at one update."""
    policy = TinyPolicy(TinyModule(seed=1))
    optimizer = torch.optim.AdamW(policy.model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    cores = np.random.default_rng(5).integers(0, 20, size=(64, 10)).astype(np.int8)
    batch_rows = 4

    class Stream:
        cycle_of_position = np.zeros(updates * batch_rows, dtype=np.int32)

        def batch(self, update):
            rows = np.arange((update - 1) * batch_rows, update * batch_rows) % 64
            return rows, rows

    state = {"update": 0}

    class CountingStream(Stream):
        def batch(self, update):
            state["update"] = int(update)
            return Stream.batch(self, update)

    def task_batch(chosen_rows, rejected_rows):
        if state["update"] == fail_at:
            raise RuntimeError("the backend fell over")
        return replay.task_term("continued_sft",
                                policy_chosen=policy.sequence_log_probs(cores[chosen_rows]))

    def gate_check(*, update, reason, exposures):
        return {"passed": True, "D": 0.1, "update": update, "reason": reason}

    def on_endpoint(*, update, exposures, checkpoint, gate, preservation, ledger):
        return {"record_kind": "exposure_endpoint", "update": update,
                "chosen_exposures": exposures["chosen"], "checkpoint": checkpoint,
                "macro_average_precision": 0.5}

    row = {"trajectory": "t", "arm_id": "a", "task": "continued_sft", "replay_lambda": 0.0,
           "seed": 1}
    return campaign.run_trajectory(
        row=row, directory=tmp_path, policy=policy, optimizer=optimizer, scheduler=scheduler,
        stream=CountingStream(), replay_order=np.arange(updates * batch_rows) % 64,
        cadence=streams.UpdateCadence(first_update=1, interval=interval, endpoints=endpoints),
        endpoints=endpoints, batch_rows=batch_rows, microbatch_rows=2,
        task_batch=task_batch, replay_batch=lambda rows: None, gate_check=gate_check,
        preservation=lambda **kwargs: {"available": True}, on_endpoint=on_endpoint,
        gradient_clip=1.0, identity={"trajectory": "t"}, max_updates=updates,
        journal_factory=journal_factory)


def test_a_failure_between_checks_is_a_failed_trajectory_with_journalled_counters(tmp_path):
    """Not `incomplete`, not `stopped_by_gate`, and it fail-stops the campaign.

    The failure is at update 8 while the last check was at update 5, so the
    progress file is four updates behind. The terminal status must report the seven
    updates that completed -- never the eighth, which was attempted.
    """
    with pytest.raises(RuntimeError, match="fell over"):
        failing_run(tmp_path, fail_at=8)
    terminal = campaign.read_terminal_status(tmp_path)
    assert terminal["status"] == campaign.STATUS_FAILED
    assert terminal["updates"] == 7, "seven updates completed and were journalled"
    assert terminal["attempted_updates"] == 8
    assert terminal["exposures"]["chosen"] == 7 * 4
    assert "RuntimeError" in terminal["failure"]["error"]
    progress = json.loads((tmp_path / campaign.PROGRESS_JSON).read_text(encoding="utf-8"))
    assert progress["update"] == 7 and progress["attempted_update"] == 8


def test_a_failure_on_the_first_journal_append_reports_zero_rather_than_the_loop_counters(tmp_path):
    """Confirmed counters come from the journals, including when they hold nothing.

    The reviewer injected an ``OSError`` on the very first update-journal append and
    got ``updates: 0`` paired with ``exposures: {chosen: 4}`` under a document
    saying the counters came from the journals. Zero and zero is the truthful pair;
    what the loop had counted is kept in its own fields and is never presented as
    completed work.
    """
    from smallAntibodyGen.experiments.her2_guarded_trajectory import JsonlJournal

    class RefusingJournal(JsonlJournal):
        def append(self, record):
            if record.get("record_kind") == "update":
                raise OSError("injected first update journal append failure")
            return super().append(record)

    with pytest.raises(OSError, match="injected"):
        failing_run(tmp_path, fail_at=None, journal_factory=RefusingJournal)
    terminal = campaign.read_terminal_status(tmp_path)
    assert terminal["status"] == campaign.STATUS_FAILED
    assert terminal["updates"] == 0 and terminal["exposures"] == {}
    assert terminal["attempted_updates"] == 1
    assert terminal["attempted_exposures"]["chosen"] == 4
    assert "are the journalled ones and are zero" in terminal["failure"]["counters_from"]


def test_a_failure_after_a_completed_endpoint_keeps_that_endpoint(tmp_path):
    with pytest.raises(RuntimeError, match="fell over"):
        failing_run(tmp_path, fail_at=6, endpoints=(2,))
    terminal = campaign.read_terminal_status(tmp_path)
    assert terminal["status"] == campaign.STATUS_FAILED
    assert sorted(terminal["endpoints_reached"]) == ["2"]
    assert terminal["endpoints_reached"]["2"]["record_kind"] == "exposure_endpoint"
    assert terminal["endpoints_reached"]["2"]["checkpoint"]["file"] == "endpoint_update2.pt"
    assert campaign.durable_progress(tmp_path)["endpoints_reached"]["2"]["update"] == 2


def test_a_reached_endpoint_is_journalled_as_a_record_while_the_run_is_still_going(tmp_path):
    tiny_run(tmp_path, replay_lambda=0.0, gate_passes=lambda update: True)
    lines = [json.loads(line) for line in
             (tmp_path / campaign.ENDPOINTS_JSONL).read_text(encoding="utf-8").splitlines()]
    assert [row["update"] for row in lines] == [2, 4]
    assert all(row["record_kind"] == "exposure_endpoint" for row in lines)
    assert all(row["checkpoint"]["state_sha256"] for row in lines)


def test_the_loop_refuses_a_stream_batch_of_the_wrong_size(tmp_path):
    policy = TinyPolicy(TinyModule())

    class ShortStream:
        cycle_of_position = np.zeros(8, dtype=np.int32)

        def batch(self, update):
            return np.arange(2), np.arange(2)

    with pytest.raises(ValueError, match="not the declared"):
        campaign.run_trajectory(
            row={"trajectory": "t", "arm_id": "a", "task": "continued_sft",
                 "replay_lambda": 0.0, "seed": 1},
            directory=tmp_path, policy=policy,
            optimizer=torch.optim.AdamW(policy.model.parameters(), lr=1e-3),
            scheduler=torch.optim.lr_scheduler.LambdaLR(
                torch.optim.AdamW(policy.model.parameters(), lr=1e-3), lambda step: 1.0),
            stream=ShortStream(), replay_order=np.arange(8),
            cadence=streams.UpdateCadence(endpoints=(2,)), endpoints=(2,), batch_rows=4,
            microbatch_rows=2, task_batch=lambda a, b: None, replay_batch=lambda r: None,
            gate_check=lambda **kwargs: {"passed": True}, preservation=lambda **kwargs: {},
            on_endpoint=lambda **kwargs: {}, gradient_clip=1.0, identity={}, max_updates=2)


# ---------------------------------------------------------------------------
# preservation is a diagnostic, never a stopping rule
# ---------------------------------------------------------------------------

def test_a_preservation_failure_is_recorded_and_does_not_stop_anything(monkeypatch):
    from smallAntibodyGen.experiments import her2_support_scoring as scoring
    monkeypatch.setattr(scoring, "strict_sequence_log_probabilities",
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            ValueError("nonfinite logit")))
    monitor = campaign.PreservationMonitor(
        index=np.zeros((4, 10), dtype=np.int8), parent_log_probability=np.zeros(4),
        teacher_probabilities=torch.zeros(4, 10, 20),
        teacher_log_probabilities=torch.zeros(4, 10, 20))
    block = monitor.measure(TinyPolicy(TinyModule()))
    assert block["available"] is False and "nonfinite" in block["reason"]
    assert "introduce no stopping rule" in block["consequence"]
    assert block["recognized_as"].startswith("nonfinite")


def test_a_programming_error_in_the_monitor_stops_the_campaign_instead(monkeypatch):
    """Only a recognized nonfinite observation follows the diagnostic policy.

    A shape or ordering failure is not an observation about the model. Recording it
    as "preservation unavailable" would let a broken monitor run 36 trajectories and
    let the campaign call itself complete with no preservation numbers at all.
    """
    from smallAntibodyGen.experiments import her2_support_scoring as scoring
    monkeypatch.setattr(scoring, "strict_sequence_log_probabilities",
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            ValueError("expected (N, 10) canonical cores, got (4, 9)")))
    monitor = campaign.PreservationMonitor(
        index=np.zeros((4, 10), dtype=np.int8), parent_log_probability=np.zeros(4),
        teacher_probabilities=torch.zeros(4, 10, 20),
        teacher_log_probabilities=torch.zeros(4, 10, 20))
    with pytest.raises(ValueError, match="not a recognized nonfinite-score observation"):
        monitor.measure(TinyPolicy(TinyModule()))


def test_preservation_reports_the_drop_and_the_conditional_kl_apart():
    policy = TinyPolicy(TinyModule(seed=2))
    index = np.random.default_rng(1).integers(0, 20, size=(8, 10)).astype(np.int8)
    logs = torch.log_softmax(torch.randn(8, 10, 20, generator=torch.Generator().manual_seed(3)),
                             dim=-1)
    monitor = campaign.PreservationMonitor(
        index=index, parent_log_probability=np.full(8, -30.0),
        teacher_probabilities=logs.exp(), teacher_log_probabilities=logs)
    block = monitor.measure(policy, conditional_batch=3)
    assert block["available"] is True
    assert block["forward_kl"]["rows"] == 8
    assert block["conditional_kl"]["rows"] == 8
    assert "not interchangeable" in block["conditional_kl"]["distinct_from_drop"]
    assert set(block["tails"]["counts"]) == {"gt1", "gt5", "tenfold", "hundredfold"}


# ---------------------------------------------------------------------------
# an endpoint record cannot be built from a failed gate
# ---------------------------------------------------------------------------

def test_an_endpoint_record_is_refused_for_a_check_that_did_not_pass():
    with pytest.raises(ValueError, match="never produces a selectable nominal endpoint"):
        report.trajectory_endpoint_record(
            row={"trajectory": "t", "arm_id": "a", "task": "ipo", "replay_lambda": 1.0,
                 "seed": 1},
            update=1000, exposures={"chosen": 64000}, checkpoint={},
            gate={"passed": False, "D": 4.0}, preservation={}, evaluation={},
            diversity={"eligible": True})


def test_a_diversity_ineligible_endpoint_is_still_a_recorded_observation():
    record = report.trajectory_endpoint_record(
        row={"trajectory": "t", "arm_id": "a", "task": "ipo", "replay_lambda": 1.0, "seed": 1},
        update=1000, exposures={"chosen": 64000}, checkpoint={"sha256": "x"},
        gate={"passed": True, "D": 0.1},
        preservation={"forward_kl": {"mean": 0.2},
                      "tails": {"counts": {"tenfold": {"fraction": 0.03}}}},
        evaluation={"macro_average_precision": 0.87, "worst_stratum_average_precision": 0.8,
                    "val_metrics": {}, "val_strata": {}},
        diversity={"eligible": False, "failed_gates": ["unique_fraction"]})
    assert record["diversity_eligible"] is False
    assert record["diversity_failed_gates"] == ["unique_fraction"]
    assert "distinct observation from a likelihood breach" in record["eligibility_note"]
    assert record["preservation"]["tenfold_fraction"] == 0.03


# ---------------------------------------------------------------------------
# cost accounting
# ---------------------------------------------------------------------------

def test_costs_are_measured_per_category_and_the_remainder_is_reported():
    ledger = campaign.CostLedger()
    with ledger.segment("optimizer"):
        pass
    with ledger.segment("gate"):
        pass
    document = ledger.document()
    assert document["segments"]["optimizer"] == 1 and document["segments"]["gate"] == 1
    assert document["unattributed_seconds"] >= 0
    assert "not assumed to be twice" in document["basis"]
    with pytest.raises(ValueError, match="Unknown cost category"):
        with ledger.segment("training"):
            pass
