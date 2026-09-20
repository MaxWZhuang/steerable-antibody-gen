"""Portable names, hash-probed roots, LF bytes and artifacts that cannot be refreshed.

Every test here is about a failure mode that produces a *plausible* result rather
than an error: a Windows path that resolves to a different file on Linux, a
partial shard that reads as a completed one, a rerun that rewrites its own
timings, or a host directory that reaches published evidence through an exception
message. None of them raise on their own.
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from smallAntibodyGen.experiments import her2_support_paths as paths


# ---------------------------------------------------------------------------
# recorded paths: flavor, traversal, drive-relative, anchors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("recorded,flavor", [
    ("C:\\Users\\x\\outputs\\camp\\stage1\\f.pt", paths.WINDOWS_ABSOLUTE),
    ("C:/Users/x/outputs/camp/stage1/f.pt", paths.WINDOWS_ABSOLUTE),
    ("/home/x/outputs/camp/stage1/f.pt", paths.POSIX_ABSOLUTE),
    ("outputs\\camp\\stage1\\f.pt", paths.WINDOWS_RELATIVE),
    ("outputs/camp/stage1/f.pt", paths.RELATIVE),
    ("\\\\server\\share\\f.pt", paths.WINDOWS_UNC),
    ("C:untrusted/camp/stage1/f.pt", paths.WINDOWS_DRIVE_RELATIVE),
])
def test_recorded_flavor_is_reported_not_guessed(recorded, flavor):
    assert paths.recorded_path_flavor(recorded) == flavor


def test_windows_and_posix_records_resolve_to_the_same_logical_artifact():
    """The same file recorded by two runs must give one logical name."""
    windows = "C:\\Users\\big\\outputs\\her2_guarded_20260918\\stage1\\run\\last_passing.pt"
    posix = "/mnt/f/outputs/her2_guarded_20260918/stage1/run/last_passing.pt"
    relative = "outputs/her2_guarded_20260918/stage1/run/last_passing.pt"
    suffixes = {paths.campaign_relative(value, anchor="her2_guarded_20260918")
                for value in (windows, posix, relative)}
    assert suffixes == {"stage1/run/last_passing.pt"}


@pytest.mark.parametrize("recorded,match", [
    # Drive-relative: names the current directory OF DRIVE C, which is process state.
    ("C:untrusted/her2_guarded_20260918/stage1/f.pt", "drive-relative"),
    # Traversal anywhere, not only in the suffix below the anchor.
    ("C:/outside/../her2_guarded_20260918/stage1/f.pt", "traverses"),
    ("outputs/her2_guarded_20260918/../../etc/f.pt", "traverses"),
    ("\\\\server\\her2_guarded_20260918\\f.pt", "UNC"),
])
def test_ambiguous_or_traversing_records_are_refused(recorded, match):
    with pytest.raises(ValueError, match=match):
        paths.campaign_relative(recorded, anchor="her2_guarded_20260918")


def test_a_repeated_anchor_is_not_resolved_by_taking_the_last_one():
    doubled = "outputs/camp/nested/camp/stage1/f.pt"
    with pytest.raises(ValueError, match="2 times"):
        paths.campaign_relative(doubled, anchor="camp")


def test_case_colliding_anchor_is_a_collision_not_a_match():
    with pytest.raises(ValueError, match="only by case"):
        paths.campaign_relative("outputs/CAMP/stage1/f.pt", anchor="camp")


def test_anchor_directory_itself_is_not_a_file():
    with pytest.raises(ValueError, match="names the anchor directory"):
        paths.campaign_relative("outputs/camp", anchor="camp")


@pytest.mark.parametrize("logical", [
    "/absolute/name", "C:/drive/name", "back\\slash", "has/../traversal", "trailing/space ",
    "empty//component", "dot/.", "colon/na:me"])
def test_logical_names_reject_everything_that_is_not_one(logical):
    with pytest.raises(ValueError):
        paths.require_logical_name(logical)


def test_case_collisions_between_logical_names_fail_closed():
    with pytest.raises(ValueError, match="case folding"):
        paths.require_no_case_collisions(["a/B.pt", "a/b.pt"], where="test")
    assert paths.require_no_case_collisions(["a/b.pt", "a/c.pt"], where="test")


def test_resolve_under_refuses_to_escape_its_root(tmp_path):
    assert paths.resolve_under(tmp_path, "a/b.json") == Path(tmp_path, "a", "b.json")
    with pytest.raises(ValueError):
        paths.resolve_under(tmp_path, "../outside.json")


# ---------------------------------------------------------------------------
# LF bytes and canonical documents
# ---------------------------------------------------------------------------

def test_new_json_is_lf_sorted_and_hashes_as_its_own_document(tmp_path):
    target = tmp_path / "doc.json"
    document = {"b": 2, "a": [1, 2], "nested": {"z": None, "y": "text"}}
    digest = paths.write_json(target, document)
    raw = target.read_bytes()
    assert b"\r\n" not in raw
    assert digest == paths.digest_document(document)
    assert list(json.loads(raw.decode("utf-8"))) == ["a", "b", "nested"]


def test_nonfinite_values_cannot_be_written(tmp_path):
    with pytest.raises(ValueError):
        paths.write_json(tmp_path / "bad.json", {"value": float("nan")})


# ---------------------------------------------------------------------------
# shards: partial, tampered, and reruns that must not refresh
# ---------------------------------------------------------------------------

def write_example(directory, values=(1.0, 2.0, 3.0), run_root=None):
    return paths.write_shard(
        directory, "scores", {"drop": np.asarray(values, dtype=np.float64),
                              "draw_index": np.arange(len(values), dtype=np.int64)},
        {"record_kind": "sequence_scores", "identity": {"state": "abc"},
         "timings": {"inference_seconds": 1.25, "wall_seconds": 2.0}},
        order="bank draw order 0..N-1", logical_prefix="scores/example",
        run_root=directory if run_root is None else run_root)


def test_a_partial_shard_is_never_a_completed_one(tmp_path):
    write_example(tmp_path)
    (tmp_path / f"scores{paths.SHARD_RECORD}").unlink()
    assert not paths.shard_is_complete(tmp_path, "scores")
    with pytest.raises(ValueError, match="partial write"):
        paths.read_shard(tmp_path, "scores")


@pytest.mark.parametrize("field,value,match", [
    ("dtype", "float32", "record declares"),
    ("shape", [999], "record declares"),
    ("order", "wrong", "row order"),
    ("content_sha256", "0" * 64, "content digest"),
])
def test_edited_array_metadata_is_rejected_not_only_the_contents(tmp_path, field, value, match):
    """Equal numbers under an edited description are still a changed artifact."""
    write_example(tmp_path)
    record_path = tmp_path / f"scores{paths.SHARD_RECORD}"
    record = paths.read_json(record_path)
    record["arrays"]["drop"][field] = value
    paths.write_json(record_path, record)
    with pytest.raises(ValueError, match=match):
        paths.read_shard(tmp_path, "scores")


def test_a_rewritten_container_with_equal_arrays_is_still_a_changed_file(tmp_path):
    """Identical numbers, different archive bytes: a changed file, and it is reported.

    ``numpy.savez`` is byte-deterministic on modern numpy, so rewriting the archive
    the same way proves nothing. The rewrite here is what a real re-export looks
    like -- the same arrays through a *compressed* zip, with a different member
    timestamp -- and the container hash has to notice.
    """
    write_example(tmp_path)
    arrays = paths.read_arrays(tmp_path / "scores.npz")
    original = (tmp_path / "scores.npz").read_bytes()
    with zipfile.ZipFile(tmp_path / "scores.npz", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, values in sorted(arrays.items()):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asarray(values), allow_pickle=False)
            archive.writestr(zipfile.ZipInfo(f"{name}.npy", date_time=(2001, 1, 1, 0, 0, 0)),
                             buffer.getvalue())
    rewritten = (tmp_path / "scores.npz").read_bytes()
    assert rewritten != original, "the rewrite must actually change the file bytes"
    assert paths.read_arrays(tmp_path / "scores.npz").keys() == arrays.keys()
    for name, values in arrays.items():
        assert np.array_equal(paths.read_arrays(tmp_path / "scores.npz")[name], values)
    with pytest.raises(ValueError, match="container hashes"):
        paths.read_shard(tmp_path, "scores")


def test_extra_array_in_the_container_is_rejected(tmp_path):
    write_example(tmp_path)
    arrays = paths.read_arrays(tmp_path / "scores.npz")
    arrays["surprise"] = np.zeros(3)
    paths.write_arrays(tmp_path / "scores.npz", arrays)
    with pytest.raises(ValueError, match="container hashes|record declares|container holds"):
        paths.read_shard(tmp_path, "scores")


def test_rerun_verifies_a_completed_shard_and_keeps_its_original_timings(tmp_path):
    first = write_example(tmp_path)
    again = paths.write_shard(
        tmp_path, "scores", {"drop": np.asarray([1.0, 2.0, 3.0]),
                             "draw_index": np.arange(3, dtype=np.int64)},
        {"record_kind": "sequence_scores", "identity": {"state": "abc"},
         "timings": {"inference_seconds": 99.0, "wall_seconds": 120.0}},
        order="bank draw order 0..N-1", logical_prefix="scores/example", run_root=tmp_path)
    assert again["rerun"] == "verified_without_rewrite"
    on_disk = paths.read_json(tmp_path / f"scores{paths.SHARD_RECORD}")
    assert on_disk["timings"] == first["timings"] == {"inference_seconds": 1.25,
                                                      "wall_seconds": 2.0}


def test_rerun_with_different_numbers_refuses_to_mutate_the_artifact(tmp_path):
    write_example(tmp_path)
    with pytest.raises(ValueError, match="no longer reproduce"):
        paths.write_shard(
            tmp_path, "scores", {"drop": np.asarray([1.0, 2.0, 3.5]),
                                 "draw_index": np.arange(3, dtype=np.int64)},
            {"record_kind": "sequence_scores", "identity": {"state": "abc"},
             "timings": {}}, order="bank draw order 0..N-1", logical_prefix="scores/example",
            run_root=tmp_path)


def test_rerun_over_a_corrupt_container_is_not_accepted_by_comparing_json_only(tmp_path):
    """The reuse path must verify the npz itself, not just its description."""
    write_example(tmp_path)
    (tmp_path / "scores.npz").write_bytes(b"not an npz")
    with pytest.raises(Exception):
        write_example(tmp_path)


# ---------------------------------------------------------------------------
# the completion manifest is the saved authority
# ---------------------------------------------------------------------------

def test_completion_manifest_detects_an_edited_artifact_after_the_fact(tmp_path):
    target = tmp_path / "summary.json"
    document = {"value": 1, "timings": {"wall_seconds": 3.0}}
    paths.write_json(target, document)
    paths.record_completion(tmp_path, "run/summary.json", target, kind="summary",
                            scientific_digest=paths.digest_document({"value": 1}),
                            timings=document["timings"])
    report = paths.verify_completions(tmp_path, resolve=lambda name: target)
    assert report["problems"] == [] and report["artifacts_checked"] == 1
    paths.write_json(target, {"value": 2, "timings": {"wall_seconds": 0.5}})
    report = paths.verify_completions(tmp_path, resolve=lambda name: target)
    assert report["problems"] and "bytes differ" in report["problems"][0]["problem"]


def test_recording_a_second_different_result_under_one_name_is_refused(tmp_path):
    target = tmp_path / "summary.json"
    paths.write_json(target, {"value": 1})
    paths.record_completion(tmp_path, "run/summary.json", target, kind="summary",
                            scientific_digest="a" * 64)
    with pytest.raises(ValueError, match="different scientific content"):
        paths.record_completion(tmp_path, "run/summary.json", target, kind="summary",
                                scientific_digest="b" * 64)


def test_a_missing_completed_artifact_is_a_problem_not_a_silent_pass(tmp_path):
    target = tmp_path / "summary.json"
    paths.write_json(target, {"value": 1})
    paths.record_completion(tmp_path, "run/summary.json", target, kind="summary",
                            scientific_digest="a" * 64)
    report = paths.verify_completions(tmp_path, resolve=lambda name: None)
    assert report["problems"][0]["problem"].startswith("recorded complete but absent")


# ---------------------------------------------------------------------------
# a shard is bound into the ledger, so a deleted one cannot vanish quietly
# ---------------------------------------------------------------------------

def shard_resolver(tmp_path):
    return lambda logical: tmp_path / Path(logical).name


def test_writing_a_shard_registers_both_its_record_and_its_container(tmp_path):
    write_example(tmp_path)
    artifacts = paths.read_completion_manifest(tmp_path)["artifacts"]
    assert sorted(artifacts) == ["scores/example/scores.complete.json",
                                 "scores/example/scores.npz"]
    assert artifacts["scores/example/scores.npz"]["kind"] == "shard_container"
    assert artifacts["scores/example/scores.complete.json"]["timings"] == {
        "inference_seconds": 1.25, "wall_seconds": 2.0}


def test_a_deleted_shard_is_reported_even_when_both_files_are_gone(tmp_path):
    """The failure a directory scan cannot see: nothing is left to scan."""
    write_example(tmp_path)
    expected = list(paths.shard_logicals("scores/example", "scores"))
    (tmp_path / "scores.npz").unlink()
    (tmp_path / f"scores{paths.SHARD_RECORD}").unlink()
    report = paths.verify_completions(tmp_path, resolve=shard_resolver(tmp_path),
                                      expected=expected)
    assert {problem["artifact"] for problem in report["problems"]} == set(expected)
    assert any("claimed by a summary" in problem["problem"] for problem in report["problems"])


def test_an_output_a_summary_claims_but_the_ledger_never_saw_is_reported(tmp_path):
    write_example(tmp_path)
    report = paths.verify_completions(
        tmp_path, resolve=shard_resolver(tmp_path),
        expected=["scores/never_ran/sequence_scores.npz"])
    assert any("never recorded it" in problem["problem"] for problem in report["problems"])


def test_a_rerun_over_an_unregistered_completed_shard_does_not_adopt_it(tmp_path):
    """Registering it now would make the rerun the authority for its own inputs."""
    write_example(tmp_path)
    paths.completion_manifest_path(tmp_path).unlink()
    with pytest.raises(ValueError, match="never recorded it"):
        write_example(tmp_path)


def test_a_rerun_over_an_edited_completion_record_is_refused_not_re_registered(tmp_path):
    write_example(tmp_path)
    record_path = tmp_path / f"scores{paths.SHARD_RECORD}"
    record = paths.read_json(record_path)
    record["timings"] = {"inference_seconds": 0.0, "wall_seconds": 0.0}
    paths.write_json(record_path, record)
    with pytest.raises(ValueError, match="completion manifest recorded"):
        write_example(tmp_path)


def test_a_rerun_of_an_intact_shard_still_verifies_without_rewriting(tmp_path):
    first = write_example(tmp_path)
    again = write_example(tmp_path)
    assert again["rerun"] == "verified_without_rewrite"
    assert again["timings"] == first["timings"]
    assert paths.read_completion_manifest(tmp_path)["artifacts"][
        "scores/example/scores.npz"]["sha256"] == paths.sha256_file(tmp_path / "scores.npz")


# ---------------------------------------------------------------------------
# host paths never reach tracked evidence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "C:\\Users\\big DAWG\\outputs\\camp\\f.pt is missing",
    "failed to open F:/her2/stage1/f.pt",
    "/home/reviewer/outputs/camp/f.pt did not verify",
])
def test_host_paths_are_scrubbed_from_nested_reasons(text):
    document = {"records": [{"status_reason": text, "nested": {"detail": [text]}}]}
    scrubbed = paths.scrub_host_paths(document)
    assert paths.host_path_leaks(scrubbed) == []
    assert paths.host_path_leaks(document), "the fixture must actually contain a host path"


def test_logical_names_are_not_mistaken_for_host_paths():
    document = {"logical_path": "outputs/her2_guarded_20260918/stage1/f.pt"}
    assert paths.scrub_host_paths(document) == document


# ---------------------------------------------------------------------------
# roots: accepted by hash, and disjoint from what they read
# ---------------------------------------------------------------------------

def test_a_root_that_merely_exists_is_not_accepted(tmp_path):
    empty, real = tmp_path / "empty", tmp_path / "real"
    (real / "validation").mkdir(parents=True)
    (real / "validation/stage1.json").write_bytes(b"content")
    empty.mkdir()
    probes = {"validation/stage1.json": paths.sha256_bytes(b"content")}
    with pytest.raises(ValueError, match="hash probes"):
        paths.resolve_root("guarded", logical="outputs/camp", candidates=[empty], probes=probes)
    resolved = paths.resolve_root("guarded", logical="outputs/camp",
                                  candidates=[empty, real], probes=probes)
    assert resolved.local_path == real
    assert "local_path" not in resolved.document()


def test_a_root_with_different_bytes_is_rejected(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "f.json").write_bytes(b"different")
    with pytest.raises(ValueError, match="hash probes"):
        paths.resolve_root("guarded", logical="outputs/camp", candidates=[root],
                           probes={"f.json": paths.sha256_bytes(b"content")})


def test_a_root_cannot_be_accepted_without_a_probe(tmp_path):
    with pytest.raises(ValueError, match="at least one hash probe"):
        paths.resolve_root("guarded", logical="outputs/camp", candidates=[tmp_path], probes={})


@pytest.mark.parametrize("relative", ["", "sub", ".."])
def test_a_run_root_overlapping_a_protected_location_is_refused(tmp_path, relative):
    protected = tmp_path / "historical"
    (protected / "sub").mkdir(parents=True)
    run_root = protected if relative == "" else (
        protected / "sub" if relative == "sub" else tmp_path)
    with pytest.raises(ValueError, match="overlaps protected locations"):
        paths.require_disjoint_run_root(run_root, {"historical": protected})


def test_a_disjoint_run_root_is_allowed(tmp_path):
    protected = tmp_path / "historical"
    protected.mkdir()
    assert paths.require_disjoint_run_root(tmp_path / "audit", {"historical": protected})


def test_run_paths_can_bind_without_creating_anything(tmp_path):
    run = paths.RunPaths.create(tmp_path, tmp_path / "audit", logical="outputs/audit",
                                create=False)
    assert not run.run_root.exists()
    run.ensure()
    assert run.run_root.is_dir()
    assert run.logical("a/b.json") == "outputs/audit/a/b.json"


# ---------------------------------------------------------------------------
# progress and timings tell the truth about how a stage ended
# ---------------------------------------------------------------------------

def test_an_unknown_total_stays_null_rather_than_being_invented(tmp_path):
    progress = paths.StageProgress(tmp_path / "p.json", stage="inventory")
    progress.start()
    progress.advance("first")
    record = paths.read_json(tmp_path / "p.json")
    assert record["total"] is None and record["completed"] == 1
    assert "denominator" in record["total_note"]
    with pytest.raises(ValueError):
        paths.StageProgress(tmp_path / "q.json", stage="x", total=0)


def test_a_raise_records_failed_and_an_interrupt_records_interrupted(tmp_path):
    failing = paths.StageProgress(tmp_path / "f.json", stage="score")
    with pytest.raises(RuntimeError):
        with failing.guard():
            raise RuntimeError("scoring refused")
    assert paths.read_json(tmp_path / "f.json")["status"] == "failed"

    stopped = paths.StageProgress(tmp_path / "i.json", stage="score")
    with pytest.raises(KeyboardInterrupt):
        with stopped.guard():
            raise KeyboardInterrupt
    record = paths.read_json(tmp_path / "i.json")
    assert record["status"] == "interrupted" and record["error"] == "KeyboardInterrupt"


def test_an_interrupted_stage_leaves_no_completed_marker(tmp_path):
    progress = paths.StageProgress(tmp_path / "s.json", stage="ches")
    with pytest.raises(SystemExit):
        with progress.guard():
            raise SystemExit(1)
    assert paths.read_json(tmp_path / "s.json")["status"] != "completed"


def test_timings_are_measured_apart_and_the_remainder_is_reported():
    clock = paths.StageClock()
    with clock.segment("inference"):
        pass
    clock.charge("analysis", 0.5)
    document = clock.document()
    assert document["analysis_seconds"] == pytest.approx(0.5)
    assert document["unattributed_seconds"] >= 0.0
    with pytest.raises(ValueError):
        clock.charge("gpu", 1.0)


def test_the_scientific_projection_drops_operational_keys_only():
    node = {"mean": 1.0, "timings": {"wall_seconds": 3}, "generated_at": "now",
            "nested": {"value": 2, "wall_seconds": 9}}
    assert paths.scientific_projection(node) == {"mean": 1.0, "nested": {"value": 2}}
