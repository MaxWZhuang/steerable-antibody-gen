"""The common evaluation, slice B: the CUDA guard, the checkpoint policy, hashed copies and
the scoring of one subject, against a fake services object.

What is real: the guard's effect on ``torch.cuda``, the file copy and hashing, the policy
table, and the document each subject produces. What is replaced: ``FlightServices`` (a
recording fake), because the point of these tests is the orchestration -- which bin is asked
for, from which file, with what refusals -- not the model. Nothing here says anything about a
trained route.
"""
from __future__ import annotations

import hashlib
import os
import sys
import types
from pathlib import Path

import pytest
import torch

from smallAntibodyGen.experiments import her2_routes_common_eval as ce
from smallAntibodyGen.experiments import her2_support_paths as paths
from smallAntibodyGen.experiments.her2_routes_conditions import BIN_HIGH, BIN_NULL
from smallAntibodyGen.tests.test_her2_routes_common_eval import panel_rows

# ---------------------------------------------------------------------------
# the in-process CUDA guard
# ---------------------------------------------------------------------------


def cuda_attributes():
    return (torch.cuda._lazy_init, torch.cuda.is_available, torch.cuda.device_count)


def test_the_guard_makes_cuda_unreachable_and_restores_every_attribute():
    """``CUDA_VISIBLE_DEVICES=-1`` is NOT used: on this machine's torch build any value that
    hides the device makes a CPU backward end in a native access violation (measured: exit 139
    for ``-1``, ``99`` and a nonexistent UUID; exit 0 for an empty or absent variable). The
    guard is in-process instead, and it must be fail-closed, not merely quiet.
    """
    before = cuda_attributes()
    with ce.forbid_cuda():
        assert torch.cuda.is_available() is False
        assert torch.cuda.device_count() == 0
        with torch.random.fork_rng():                 # the call that initialises every device
            pass
        with pytest.raises(RuntimeError, match="forbidden"):
            torch.cuda.init()
        with pytest.raises(RuntimeError, match="forbidden"):
            torch.zeros(1, device="cuda")
    assert cuda_attributes() == before


def test_the_guard_restores_even_when_the_body_raises():
    before = cuda_attributes()
    with pytest.raises(KeyError):
        with ce.forbid_cuda():
            raise KeyError("boom")
    assert cuda_attributes() == before


def test_the_guard_refuses_a_process_that_already_initialised_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(RuntimeError, match="already initialised"):
        with ce.forbid_cuda():
            pass


# ---------------------------------------------------------------------------
# which checkpoints may be read
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("allow, eligible", [(False, False), (True, True)])
def test_the_live_name_fallback_needs_an_acknowledgement_and_nothing_else(tmp_path, allow,
                                                                          eligible):
    """This policy governs ONLY the fallback (no hard link possible); the normal route never opens
    the live name at all. It is an acknowledgement, not a liveness test: a one-time probe goes
    stale during a long evaluation, and a restarted supervisor re-saves ``last.pt`` within seconds.
    MEASURED on this machine: an ordinary read handle makes the trainer's ``Path.replace`` raise
    ``WinError 5``, and a delete-sharing handle does not help."""
    verdict = ce.checkpoint_policy(tmp_path / "last.pt", allow_live=allow)
    assert verdict["eligible"] is eligible and verdict["reason"]
    if not eligible:
        assert "replace" in verdict["reason"] and "hard-link" in verdict["reason"]


def test_there_are_no_stable_file_or_liveness_exemptions(tmp_path):
    """A ``stepN.pt`` milestone and a route that already has ``run.json`` are NOT safe to open on
    a live flight (a resumed bakeoff re-runs ``train()`` for finished routes and re-saves
    ``last.pt``; a crash between the milestone save and the ``last.pt`` save makes resume rewrite
    the milestone), and a cached "not running" answer goes stale mid-evaluation."""
    directory = tmp_path / "route"
    directory.mkdir()
    (directory / "run.json").write_text("{}", encoding="utf-8")
    for name in ("step5000.pt", "best.pt", "last.pt"):
        assert ce.checkpoint_policy(directory / name, allow_live=False)["eligible"] is False
    assert not hasattr(ce, "route_completed")
    import inspect
    assert "flight_running" not in inspect.signature(ce.checkpoint_policy).parameters
    assert "flight_running" not in inspect.signature(ce.pinned_checkpoint).parameters


# ---------------------------------------------------------------------------
# a hashed, stable copy
# ---------------------------------------------------------------------------

def test_a_copy_is_byte_identical_hashed_and_leaves_no_temporary(tmp_path):
    source = tmp_path / "ckpt.pt"
    payload = bytes(range(256)) * 4099
    source.write_bytes(payload)
    destination = tmp_path / "scratch" / "copy.pt"
    record = ce.copy_with_hash(source, destination, chunk=1 << 12)
    assert destination.read_bytes() == payload
    assert record == {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
    assert not list(destination.parent.glob("*.tmp"))


def test_a_copy_that_would_not_fit_is_refused_before_any_byte_is_written(tmp_path, monkeypatch):
    source = tmp_path / "ckpt.pt"
    source.write_bytes(b"x" * 1000)
    monkeypatch.setattr(ce.shutil, "disk_usage",
                        lambda _path: types.SimpleNamespace(total=10, used=9, free=100))
    with pytest.raises(ValueError, match="free"):
        ce.copy_with_hash(source, tmp_path / "scratch" / "copy.pt")
    assert not (tmp_path / "scratch" / "copy.pt").exists()


def test_a_source_replaced_while_it_was_being_read_is_refused(tmp_path, monkeypatch):
    source = tmp_path / "ckpt.pt"
    source.write_bytes(b"x" * 5000)
    signatures = iter([(5000, 1), (5000, 2)])
    monkeypatch.setattr(ce, "_signature", lambda _path: next(signatures))
    with pytest.raises(ValueError, match="changed while"):
        ce.copy_with_hash(source, tmp_path / "scratch" / "copy.pt")
    assert not (tmp_path / "scratch" / "copy.pt").exists()
    assert not list((tmp_path / "scratch").glob("*.tmp"))


# ---------------------------------------------------------------------------
# pinning: the hazard is real on this machine, and a hard link avoids it
# ---------------------------------------------------------------------------

def can_link(tmp_path):
    probe = tmp_path / "probe.src"
    probe.write_bytes(b"x")
    try:
        os.link(probe, tmp_path / "probe.dst")
        return True
    except OSError:
        return False


@pytest.mark.skipif(sys.platform != "win32", reason="the replace hazard is Windows-specific")
def test_an_ordinary_read_handle_blocks_the_trainers_replace(tmp_path):
    """The MEASURED premise of the whole pinning design. If this stops raising, the premise (and
    the cost of pinning) deserves a second look."""
    original = tmp_path / "last.pt"
    original.write_bytes(b"old" * 100)
    temporary = tmp_path / "last.pt.tmp"
    temporary.write_bytes(b"new" * 100)
    with open(original, "rb") as handle:
        handle.read(5)
        with pytest.raises(PermissionError):
            temporary.replace(original)


def test_a_pinned_checkpoint_never_blocks_the_trainers_replace_and_keeps_its_bytes(tmp_path):
    if not can_link(tmp_path):
        pytest.skip("hard links are unavailable on this filesystem")
    original = tmp_path / "flight" / "last.pt"
    original.parent.mkdir()
    payload = b"old" * 5000
    original.write_bytes(payload)
    with ce.pinned_checkpoint(original, tmp_path / "scratch", "k", allow_live=False) as pin:
        assert pin["method"] == "hardlink"
        assert pin["sha256"] == hashlib.sha256(payload).hexdigest() and pin["bytes"] == len(payload)
        with open(pin["path"], "rb") as reader:            # a read in progress ON THE PIN
            reader.read(7)
            temporary = original.with_name("last.pt.tmp")
            temporary.write_bytes(b"new" * 5000)
            temporary.replace(original)                    # the trainer saves now: must not raise
            assert reader.read() == payload[7:]
        assert pin["path"].read_bytes() == payload, "the pin keeps the OLD bytes"
        assert original.read_bytes() == b"new" * 5000
    assert not pin["path"].exists(), "the pin is removed afterwards"


def test_the_pin_is_removed_when_the_body_raises_and_foreign_files_are_left_alone(tmp_path):
    if not can_link(tmp_path):
        pytest.skip("hard links are unavailable on this filesystem")
    original = tmp_path / "ckpt.pt"
    original.write_bytes(b"w" * 100)
    stale = tmp_path / "scratch" / "k.pt"
    stale.parent.mkdir()
    stale.write_bytes(b"left behind by a crashed run")
    with pytest.raises(KeyError):
        with ce.pinned_checkpoint(original, tmp_path / "scratch", "k",
                                  allow_live=False) as pin:
            assert pin["path"].read_bytes() == b"w" * 100
            raise KeyError("boom")
    assert stale.exists(), "another invocation's directory is never deleted as if it were stale"
    assert not pin["path"].exists() and not pin["path"].parent.exists()


def refuse_links(*_args, **_kwargs):
    raise OSError(17, "The system cannot move the file to a different disk drive")


def test_without_a_link_a_live_checkpoint_needs_the_acknowledgement_and_is_then_copied(
        tmp_path, monkeypatch):
    original = tmp_path / "ckpt.pt"
    original.write_bytes(b"w" * 100)
    monkeypatch.setattr(ce.os, "link", refuse_links)
    with pytest.raises(ValueError, match="replace"):
        with ce.pinned_checkpoint(original, tmp_path / "scratch", "k", allow_live=False):
            pytest.fail("a refused pin must not yield")
    with ce.pinned_checkpoint(original, tmp_path / "scratch", "k", allow_live=True) as pin:
        assert pin["method"] == "copy" and pin["path"].read_bytes() == b"w" * 100
        assert pin["sha256"] == hashlib.sha256(b"w" * 100).hexdigest()
    assert not pin["path"].exists()
    assert ce.leftover_pins(tmp_path / "scratch") == [], "the refused attempt left nothing behind"


def test_two_invocations_sharing_a_scratch_root_and_a_key_cannot_touch_each_others_pins(
        tmp_path):
    """Review finding: with one shared pin name the second invocation deleted the first's active
    pin as if it were stale, so the first could hash the old checkpoint and load the new one --
    a wrong SHA256 beside the scores, with no identity error (identity is constant across
    training steps). Each call now gets a private directory."""
    if not can_link(tmp_path):
        pytest.skip("hard links are unavailable on this filesystem")
    original = tmp_path / "flight" / "best.pt"
    original.parent.mkdir()
    original.write_bytes(b"first" * 1000)
    scratch = tmp_path / "scratch"
    with ce.pinned_checkpoint(original, scratch, "same_key", allow_live=False) as first:
        temporary = original.with_name("best.pt.tmp")
        temporary.write_bytes(b"second" * 1000)
        temporary.replace(original)                     # the trainer saves between the two calls
        with ce.pinned_checkpoint(original, scratch, "same_key", allow_live=False) as second:
            assert first["path"] != second["path"]
            assert first["path"].read_bytes() == b"first" * 1000
            assert second["path"].read_bytes() == b"second" * 1000
            assert first["sha256"] == hashlib.sha256(b"first" * 1000).hexdigest()
            assert second["sha256"] == hashlib.sha256(b"second" * 1000).hexdigest()
        assert first["path"].exists(), "leaving the second pin must not remove the first"
        assert first["path"].read_bytes() == b"first" * 1000
    assert not first["path"].exists() and not second["path"].exists()


def test_leftover_pin_directories_are_reported_and_never_deleted(tmp_path):
    scratch = tmp_path / "scratch"
    (scratch / "pin_old_run_abc").mkdir(parents=True)
    (scratch / "pin_old_run_abc" / "x.pt").write_bytes(b"w")
    (scratch / "unrelated").mkdir()
    assert ce.leftover_pins(scratch) == [scratch / "pin_old_run_abc"]
    assert ce.leftover_pins(tmp_path / "missing") == []
    assert (scratch / "pin_old_run_abc" / "x.pt").exists()


def test_only_one_invocation_may_hold_an_output_directory(tmp_path):
    out = tmp_path / "out"
    with ce.exclusive_invocation(out):
        with pytest.raises(ValueError, match="another evaluator invocation"):
            with ce.exclusive_invocation(out):
                pytest.fail("the second invocation must not enter")
    with ce.exclusive_invocation(out):          # released again afterwards
        pass


# ---------------------------------------------------------------------------
# scoring one subject
# ---------------------------------------------------------------------------

PANEL_ROWS = ce.validate_panel(panel_rows(6),
                               development_windows=[row["window"] for row in panel_rows(6)])
DIGEST = "d" * 64


class FakeServices:
    """Stands in for ``FlightServices``: records every call, scripts every score."""

    device = "cpu"

    def __init__(self, *, identity=None, high=None, null=None, step=7000):
        self.calls = []
        self.identity = identity or {"route": "ablang2_masked_editor", "arm": "full",
                                     "seed": 20260927}
        self.high = list(high if high is not None else [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0])
        self.null = list(null if null is not None else [-1.5, -1.0, -3.5, -2.0, -6.0, -4.0])
        self.step = step

    def load_for_scoring(self, route, checkpoint, *, seed=0, arm=None):
        self.calls.append(("load", route, None if checkpoint is None else Path(checkpoint)))
        if checkpoint is None:
            return "model", "heads", {"checkpoint": None, "trained": False}
        return "model", "heads", {"path": str(checkpoint), "step": self.step, "trained": True,
                                  "training_identity": dict(self.identity)}

    def score_panel(self, route, checkpoint, panel, *, batch_size=16, model=None, heads=None,
                    arm="full", seed=0, requested_bin=None):
        self.calls.append(("score", route, checkpoint, requested_bin, model, len(panel)))
        return list(self.null if requested_bin == BIN_NULL else self.high)


def checkpoint_file(tmp_path, name="step5000.pt", size=3000):
    path = tmp_path / "flight" / "bakeoff" / "ablang2_masked_editor" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"w" * size)
    return path


def subject(path=None, label="step5000", route="ablang2_masked_editor", flight="t1"):
    return ce.Subject(flight=flight, route=route, label=label, checkpoint=path)


def score(services, tmp_path, who, **overrides):
    arguments = dict(scratch=tmp_path / "scratch", panel_digest=DIGEST, allow_live=False,
                     batch_size=2)
    arguments.update(overrides)
    return ce.score_subject(services, who, PANEL_ROWS, **arguments)


def test_a_subject_key_is_stable_and_names_flight_route_and_label():
    assert subject().key == "t1__ablang2_masked_editor__step5000"
    assert subject(None, label="zero_shot").kind == "zero_shot"
    assert subject(Path("x.pt")).kind == "checkpoint"


def test_a_checkpoint_is_scored_from_its_hashed_copy_under_both_bins(tmp_path):
    original = checkpoint_file(tmp_path)
    services = FakeServices()
    document = score(services, tmp_path, subject(original))
    loads = [call for call in services.calls if call[0] == "load"]
    assert len(loads) == 1 and loads[0][2] != original, "scoring must read the frozen copy"
    scores = [call for call in services.calls if call[0] == "score"]
    assert [call[3] for call in scores] == [BIN_HIGH, BIN_NULL]
    assert all(call[2] is None and call[4] == "model" for call in scores), (
        "the loaded model is reused; score_panel must not reload a checkpoint per bin")
    assert document["high"] == services.high and document["null"] == services.null
    assert document["checkpoint"]["sha256"] == hashlib.sha256(original.read_bytes()).hexdigest()
    assert document["checkpoint"]["step"] == 7000
    assert document["checkpoint"]["training_identity"]["arm"] == "full"
    assert document["checkpoint"]["original_path"] == str(original)
    assert document["panel_digest"] == DIGEST and document["kind"] == "checkpoint"
    assert not list((tmp_path / "scratch").rglob("*.pt")), "the scratch copy is removed once scored"
    paths.canonical_json(document)                 # allow_nan=False: no NaN anywhere


def test_the_zero_shot_backbone_loads_no_checkpoint_and_is_labelled(tmp_path):
    services = FakeServices()
    document = score(services, tmp_path, subject(None, label="zero_shot"))
    assert services.calls[0] == ("load", "ablang2_masked_editor", None)
    assert document["kind"] == "zero_shot" and document["checkpoint"] is None


@pytest.mark.parametrize("identity, message", [
    ({"route": "piggen_causal_infiller", "arm": "full", "seed": 1}, "route"),
    ({"route": "ablang2_masked_editor", "arm": "head_only", "seed": 1}, "arm"),
])
def test_a_checkpoint_of_another_route_or_arm_is_refused(tmp_path, identity, message):
    original = checkpoint_file(tmp_path)
    with pytest.raises(ValueError, match=message):
        score(FakeServices(identity=identity), tmp_path, subject(original))


@pytest.mark.parametrize("field, bad, message", [
    ("high", [float("nan")] + [-1.0] * 5, "finite"),
    ("null", [-1.0] * 5, "6 rows"),
    ("high", [-1.0] * 6 + [-2.0], "6 rows"),
])
def test_invalid_scores_are_refused_before_anything_is_written(tmp_path, field, bad, message):
    original = checkpoint_file(tmp_path)
    with pytest.raises(ValueError, match=message):
        score(FakeServices(**{field: bad}), tmp_path, subject(original))


def test_a_live_checkpoint_is_scored_from_a_pin_without_any_acknowledgement(tmp_path):
    if not can_link(tmp_path):
        pytest.skip("hard links are unavailable on this filesystem")
    original = checkpoint_file(tmp_path, name="last.pt")
    document = score(FakeServices(), tmp_path, subject(original, label="last"))
    assert document["checkpoint"]["snapshot_method"] == "hardlink"


def test_a_live_checkpoint_that_cannot_be_pinned_is_refused_without_the_acknowledgement(
        tmp_path, monkeypatch):
    original = checkpoint_file(tmp_path, name="last.pt")
    monkeypatch.setattr(ce.os, "link", refuse_links)
    services = FakeServices()
    with pytest.raises(ValueError, match="replace"):
        score(services, tmp_path, subject(original, label="last"))
    assert services.calls == [], "a refused subject must not even be loaded"


def test_scoring_requires_a_cpu_services_object(tmp_path):
    services = FakeServices()
    services.device = "cuda"
    with pytest.raises(ValueError, match="cpu"):
        score(services, tmp_path, subject(None, label="zero_shot"))


def test_the_evaluator_identity_hashes_the_code_that_produced_the_scores():
    identity = ce.evaluator_identity()
    assert identity["schema"] == ce.SCHEMA
    for name in ("her2_routes_common_eval.py", "her2_routes_services.py", "her2_routes_eval.py"):
        assert len(identity["sources"][name]) == 64
    assert identity["sources"]["her2_routes_common_eval.py"] == paths.sha256_file(ce.__file__)
    assert identity["python"] and identity["numpy"]


def test_the_evaluator_identity_covers_every_module_scoring_can_execute(tmp_path, monkeypatch):
    """The services run the working tree's ``models/dplm_policy.py``, ``her2_routes_serial.py`` and
    more, and the frozen-flight verification does not cover them (it runs with the runtime-closure
    check off). Hashing three named files let an edit to any other module slip past the resume
    gate, so the identity hashes every source file of the package -- and not the tests."""
    real = ce.evaluator_identity()
    assert real["package_files"] > 50, "the real package inventory must be non-trivial"
    assert set(real["dependencies"]) == set(ce.DEPENDENCIES), (
        "the installed versions of the native scoring dependencies are recorded without "
        "importing them (p-IgGen and DPLM run transformers code that no hash above covers)")
    moved = {**real, "dependencies": {**real["dependencies"], "transformers": "0.0.0-other"}}
    assert ce.evaluator_differences(real, moved) == ["dependencies"]
    root = tmp_path / "pkg"
    (root / "models").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "__pycache__").mkdir()
    (root / "models" / "dplm_policy.py").write_text("a = 1\n", encoding="utf-8")
    (root / "tests" / "test_x.py").write_text("x = 1\n", encoding="utf-8")
    (root / "__pycache__" / "junk.py").write_text("j = 1\n", encoding="utf-8")
    monkeypatch.setattr(ce, "PACKAGE_ROOT", root)
    before = ce.evaluator_identity()
    assert before["package_files"] == 1
    (root / "tests" / "test_x.py").write_text("x = 2\n", encoding="utf-8")
    (root / "__pycache__" / "junk.py").write_text("j = 2\n", encoding="utf-8")
    assert ce.evaluator_identity()["package_sha256"] == before["package_sha256"], (
        "tests and bytecode directories are not part of the scoring implementation")
    (root / "models" / "dplm_policy.py").write_text("a = 2\n", encoding="utf-8")
    after = ce.evaluator_identity()
    assert ce.evaluator_differences(before, after) == ["package_sha256"]
    (root / "models" / "new_module.py").write_text("n = 1\n", encoding="utf-8")
    assert ce.evaluator_identity()["package_sha256"] != after["package_sha256"], (
        "a new module is a change too, not only an edited one")


# ---------------------------------------------------------------------------
# resuming
# ---------------------------------------------------------------------------

FLIGHT_MARKERS = {name: f"{index}" * 64 for index, name in enumerate(ce.MARKERS)}


def ensure(services, tmp_path, who, **overrides):
    arguments = dict(out_dir=tmp_path / "out", scratch=tmp_path / "scratch",
                     panel_digest=DIGEST, allow_live=False, batch_size=2,
                     flight_markers=FLIGHT_MARKERS)
    arguments.update(overrides)
    return ce.ensure_subject(services, who, PANEL_ROWS, **arguments)


def other_evaluator(monkeypatch):
    """The same evaluator after its code changed: one source hash differs."""
    real = ce.evaluator_identity()
    changed = {**real, "sources": {**real["sources"], "her2_routes_common_eval.py": "f" * 64}}
    monkeypatch.setattr(ce, "evaluator_identity", lambda: changed)


@pytest.mark.parametrize("with_checkpoint", [True, False])
def test_a_saved_subject_is_not_reused_after_the_evaluator_changed(
        tmp_path, monkeypatch, with_checkpoint):
    """Resuming after an edit would score the remaining subjects with the NEW code and keep the
    old ones, and the summary would contrast scores that were never produced by one
    implementation. The saved evaluator identity must equal the current one."""
    original = checkpoint_file(tmp_path)
    who = subject(original) if with_checkpoint else subject(None, label="zero_shot")
    ensure(FakeServices(), tmp_path, who)
    other_evaluator(monkeypatch)
    second = FakeServices()
    with pytest.raises(ValueError, match="evaluator"):
        ensure(second, tmp_path, who)
    assert second.calls == [], "a refused resume must not score anything"


@pytest.mark.parametrize("with_checkpoint", [True, False])
def test_a_saved_subject_is_not_reused_when_the_flight_markers_differ(tmp_path, with_checkpoint):
    """A zero-shot document carries no checkpoint hash, so nothing but the flight's identity
    markers ties it to the flight it was scored against."""
    original = checkpoint_file(tmp_path)
    who = subject(original) if with_checkpoint else subject(None, label="zero_shot")
    ensure(FakeServices(), tmp_path, who)
    moved = {**FLIGHT_MARKERS, "external_sources.json": "e" * 64}
    with pytest.raises(ValueError, match="markers"):
        ensure(FakeServices(), tmp_path, who, flight_markers=moved)


def test_a_saved_subject_without_a_marker_record_is_not_reused(tmp_path):
    who = subject(None, label="zero_shot")
    document = ensure(FakeServices(), tmp_path, who)
    document.pop("flight_markers")
    paths.write_json(tmp_path / "out" / f"subject_{who.key}.json", document)
    with pytest.raises(ValueError, match="markers"):
        ensure(FakeServices(), tmp_path, who)


def test_a_new_subject_records_the_markers_of_the_flight_it_was_scored_against(tmp_path):
    document = ensure(FakeServices(), tmp_path, subject(None, label="zero_shot"))
    assert document["flight_markers"] == FLIGHT_MARKERS


def test_a_finished_subject_is_reused_only_when_its_content_and_panel_still_match(tmp_path):
    original = checkpoint_file(tmp_path)
    document = ensure(FakeServices(), tmp_path, subject(original))
    assert (tmp_path / "out" / f"subject_{subject(original).key}.json").is_file()
    second = FakeServices()
    again = ensure(second, tmp_path, subject(original))
    assert again == document and second.calls == [], "a matching subject is not rescored"
    with pytest.raises(ValueError, match="panel"):
        ensure(FakeServices(), tmp_path, subject(original), panel_digest="e" * 64)
    original.write_bytes(b"different weights" * 100)
    with pytest.raises(ValueError, match="content"):
        ensure(FakeServices(), tmp_path, subject(original))


# ---------------------------------------------------------------------------
# mutable status files are never opened
# ---------------------------------------------------------------------------

MUTABLE = ("heartbeat.json", "progress.json", "campaign_status.json", "run_all.json")


def forbid_opening(monkeypatch, names=MUTABLE):
    """Fail the test if anything OPENS a file the trainer replaces in place.

    ``Path.is_file()`` and ``stat`` take no lasting handle and are allowed; ``open``,
    ``read_text`` and ``read_bytes`` are not. On Windows a reader's handle can make the
    trainer's ``os.replace`` of ``progress.json`` / ``heartbeat.json`` raise, and those writes
    have no retry.
    """
    real_open = Path.open

    def guarded(self, *args, **kwargs):
        if self.name in names:
            raise AssertionError(f"opened mutable status file {self}")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: guarded(self).read())
    monkeypatch.setattr(Path, "read_bytes", lambda self: guarded(self, "rb").read())


def test_pinning_never_opens_a_mutable_status_file(tmp_path, monkeypatch):
    if not can_link(tmp_path):
        pytest.skip("hard links are unavailable on this filesystem")
    original = checkpoint_file(tmp_path, name="last.pt")
    for name in MUTABLE:
        (original.parent / name).write_text("{not json", encoding="utf-8")
    forbid_opening(monkeypatch)            # armed AFTER the fixtures, which write these files
    with ce.pinned_checkpoint(original, tmp_path / "scratch", "k", allow_live=False) as pin:
        assert pin["method"] == "hardlink"
