"""Checkpoint admission: the pure half (pins, origin verification, copy, record, per-load check).

What is asserted, and what is not:

* a pin set is a closed schema, and the three identity fields an admitted checkpoint may carry from
  its origin flight are exactly ``config_sha256``, ``snapshot_sha256`` and ``external_sha256``;
* every refusal the plan's two reviews named is exercised: a swapped file, a wrong step, a drifted
  route block, a drifted external file, a different vocabulary or split, a record that disagrees
  with the pins, and a record that merely exists;
* the origin flight is only ever read.

Nothing here loads a model. The services-level behaviour (the foreign-identity branch of
``load_for_scoring``, the mixed bakeoff, stage reuse) is in ``test_her2_routes_admission_services``.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from smallAntibodyGen.experiments import her2_routes_admission as admission
from smallAntibodyGen.experiments import her2_support_paths as paths

HEX = "a" * 64
ROUTE = "ablang2_masked_editor"
ROUTES = ("ablang2_masked_editor", "piggen_causal_infiller", "dplm_150m_masked")


def sha(text):
    return paths.sha256_text(text)


# ---------------------------------------------------------------------------
# a synthetic origin flight and a live config that matches it
# ---------------------------------------------------------------------------

ORIGIN_CONFIG = {
    "schema_version": "her2-generative-routes/1",
    "routes": {"enabled": [ROUTE],
               ROUTE: {"updates": 20000, "seed": 7, "max_positions": 256,
                       "why": "origin prose", "compute_note": "origin note"}},
    "her2": {"wild_type_core": "WGGDGFYAMD", "assay": "a1", "note": "origin her2 prose"},
}
CONFIG_LOGICAL = "configs/experiments/origin.json"
EXTERNAL = {"data/a.py": sha("a"), "data/b.py": sha("b"), "models/ablang2/model.pt": sha("m")}


def live_config():
    """The new flight's config: same science, different prose, one extra DPLM-only block."""
    config = copy.deepcopy(ORIGIN_CONFIG)
    config["routes"][ROUTE]["why"] = "live prose is different"
    config["routes"][ROUTE]["compute_note"] = "live note"
    config["her2"]["note"] = "live her2 prose"
    config["dplm"] = {"pins": "dplm_pins.json"}
    return config


def make_origin(tmp_path, *, name="origin"):
    """Write a finished flight's identity markers, frozen config copy and a checkpoint file."""
    root = tmp_path / name
    (root / "source_snapshot" / "configs" / "experiments").mkdir(parents=True)
    config_path = root / "source_snapshot" / CONFIG_LOGICAL
    config_path.write_text(json.dumps(ORIGIN_CONFIG, sort_keys=True), encoding="utf-8")
    paths.write_json(root / "source_snapshot.json", {"snapshot_sha256": sha("snapshot")})
    paths.write_json(root / "external_sources.json",
                     {"external_sha256": admission.external_aggregate(EXTERNAL),
                      "files": dict(EXTERNAL)})
    (root / "condition_vocabulary.json").write_text('{"bins": ["high"]}\n', encoding="utf-8")
    paths.write_json(root / "hard_split_freeze.json", {"sequence_sha256": "s" * 64, "n": 100})
    checkpoint = root / "bakeoff" / ROUTE / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"origin checkpoint bytes" * 100)
    return root, checkpoint


def live_values(origin_root):
    """What the new root's own markers say. Equal to the origin's for a compatible flight."""
    external = dict(EXTERNAL)
    external["models/dplm_150m/config.json"] = sha("dplm")  # an allowed addition
    return {"config": live_config(),
            "condition_sha256": paths.sha256_file(origin_root / "condition_vocabulary.json"),
            "split_sha256": paths.digest_document(
                paths.read_json(origin_root / "hard_split_freeze.json")),
            "external_files": external}


def make_pin(origin_root, checkpoint, **overrides):
    pin = admission.derive_pin(
        origin_root, route=ROUTE, origin_config=CONFIG_LOGICAL, checkpoint=checkpoint,
        checkpoint_step=9500, arm="full", seed=7, schedule_sha256=None,
        selector="development_affinity_spearman_high", live_config=live_config())
    pin.update(overrides)
    return pin


# ---------------------------------------------------------------------------
# block digests
# ---------------------------------------------------------------------------

def test_block_digest_ignores_prose_and_key_order_but_not_science():
    a = {"updates": 5, "why": "x", "seed": 1}
    assert admission.block_digest(a) == admission.block_digest({"seed": 1, "updates": 5, "why": "y"})
    assert admission.block_digest(a) != admission.block_digest({"updates": 6, "seed": 1})


def test_block_digest_strips_note_keys_at_every_depth():
    nested = {"opt": {"lr": 1, "note": "a"}, "compute_note": "b"}
    assert admission.block_digest(nested) == admission.block_digest({"opt": {"lr": 1}})
    assert admission.block_digest(nested) != admission.block_digest({"opt": {"lr": 2}})


def test_config_block_digests_cover_route_her2_and_only_blocks_that_exist():
    digests = admission.config_block_digests(ORIGIN_CONFIG, ROUTE)
    assert set(digests) == {"route", "her2"}
    assert set(admission.config_block_digests(live_config(), ROUTE)) == {"route", "her2", "dplm"}
    assert (admission.config_block_digests(live_config(), ROUTE)["route"] == digests["route"]
            and admission.config_block_digests(live_config(), ROUTE)["her2"] == digests["her2"])


def test_a_missing_route_block_is_refused():
    with pytest.raises(ValueError, match="no routes"):
        admission.config_block_digests({"routes": {"enabled": []}}, ROUTE)


# ---------------------------------------------------------------------------
# the pin schema
# ---------------------------------------------------------------------------

def good_pin():
    return {"origin_flight": str(Path("/tmp/origin").resolve()), "origin_config": CONFIG_LOGICAL,
            "origin_config_sha256": HEX, "origin_snapshot_sha256": HEX,
            "origin_external_sha256": HEX, "arm": "full", "seed": 7, "schedule_sha256": None,
            "checkpoint_sha256": HEX, "checkpoint_bytes": 100, "checkpoint_step": 9500,
            "block_digests": {"route": HEX, "her2": HEX}, "selector": "legacy"}


def test_parse_pins_accepts_a_complete_closed_block():
    block = {route: good_pin() for route in ROUTES}
    assert set(admission.parse_pins(block, routes=ROUTES)) == set(ROUTES)


@pytest.mark.parametrize("mutate,match", [
    (lambda b: b.pop("piggen_causal_infiller"), "route set"),
    (lambda b: b.__setitem__("dwjs_energy", good_pin()), "route set"),
    (lambda b: b[ROUTE].pop("checkpoint_sha256"), "missing"),
    (lambda b: b[ROUTE].__setitem__("extra", 1), "unexpected"),
    (lambda b: b[ROUTE].__setitem__("checkpoint_sha256", "xyz"), "64 hex"),
    (lambda b: b[ROUTE].__setitem__("origin_config_sha256", HEX.upper()), "64 hex"),
    (lambda b: b[ROUTE].__setitem__("seed", True), "seed"),
    (lambda b: b[ROUTE].__setitem__("checkpoint_step", 0), "checkpoint_step"),
    (lambda b: b[ROUTE].__setitem__("checkpoint_bytes", -1), "checkpoint_bytes"),
    (lambda b: b[ROUTE].__setitem__("origin_flight", "relative/path"), "absolute"),
    (lambda b: b[ROUTE].__setitem__("block_digests", {"route": HEX}), "her2"),
    (lambda b: b[ROUTE].__setitem__("block_digests", {"route": HEX, "her2": HEX, "zzz": HEX}),
     "block_digests"),
    (lambda b: b[ROUTE].__setitem__("schedule_sha256", "short"), "schedule_sha256"),
])
def test_parse_pins_refuses_every_malformed_pin(mutate, match):
    block = {route: good_pin() for route in ROUTES}
    mutate(block)
    with pytest.raises(ValueError, match=match):
        admission.parse_pins(block, routes=ROUTES)


def test_parse_pins_refuses_a_non_mapping_and_an_empty_block():
    for bad in (None, [], {}):
        with pytest.raises(ValueError):
            admission.parse_pins(bad, routes=ROUTES)


def test_exactly_three_identity_fields_come_from_the_origin():
    overrides = admission.pinned_identity_fields(good_pin())
    assert set(overrides) == {"config_sha256", "snapshot_sha256", "external_sha256"}
    assert admission.OVERRIDDEN_IDENTITY_FIELDS == tuple(sorted(overrides))


def test_the_pin_digest_moves_with_any_field():
    pin = good_pin()
    other = dict(pin, checkpoint_step=9501)
    assert admission.pin_digest(pin) != admission.pin_digest(other)
    assert admission.pin_digest(pin) == admission.pin_digest(dict(reversed(list(pin.items()))))


# ---------------------------------------------------------------------------
# external-table compatibility (the narrow check that replaces an aggregate override)
# ---------------------------------------------------------------------------

def test_an_identical_table_with_an_allowed_dplm_addition_is_compatible():
    new = dict(EXTERNAL, **{"models/dplm_150m/config.json": "1" * 64,
                            "models/dplm_650m/vocab.txt": "2" * 64})
    assert admission.external_problems(EXTERNAL, new) == []


def test_missing_changed_and_unexpected_files_are_each_named():
    new = dict(EXTERNAL)
    del new["data/a.py"]
    new["data/b.py"] = "f" * 64
    new["data/new.py"] = "e" * 64
    problems = admission.external_problems(EXTERNAL, new)
    assert any("data/a.py" in p and "missing" in p for p in problems)
    assert any("data/b.py" in p and "changed" in p for p in problems)
    assert any("data/new.py" in p and "unexpected" in p for p in problems)


def test_an_empty_origin_table_is_refused_not_trivially_compatible():
    assert admission.external_problems({}, EXTERNAL)


# ---------------------------------------------------------------------------
# deriving a pin and verifying an origin
# ---------------------------------------------------------------------------

def test_a_derived_pin_is_valid_and_binds_the_real_files(tmp_path):
    root, checkpoint = make_origin(tmp_path)
    pin = make_pin(root, checkpoint)
    parsed = admission.parse_pins({ROUTE: pin}, routes=(ROUTE,))[ROUTE]
    assert parsed["checkpoint_sha256"] == paths.sha256_file(checkpoint)
    assert parsed["checkpoint_bytes"] == checkpoint.stat().st_size
    assert parsed["origin_external_sha256"] == admission.external_aggregate(EXTERNAL)
    assert parsed["origin_snapshot_sha256"] == sha("snapshot")
    assert parsed["origin_config_sha256"] == paths.sha256_file(
        root / "source_snapshot" / CONFIG_LOGICAL)
    assert set(parsed["block_digests"]) == {"route", "her2"}


def test_a_compatible_origin_has_no_problems(tmp_path):
    root, checkpoint = make_origin(tmp_path)
    pin = make_pin(root, checkpoint)
    assert admission.origin_problems(pin, ROUTE, live=live_values(root)) == []


def test_the_origin_is_only_read(tmp_path):
    root, checkpoint = make_origin(tmp_path)
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    mtimes = {p: p.stat().st_mtime_ns for p in before}
    admission.origin_problems(make_pin(root, checkpoint), ROUTE, live=live_values(root))
    assert {p: p.read_bytes() for p in root.rglob("*") if p.is_file()} == before
    assert {p: p.stat().st_mtime_ns for p in before} == mtimes


@pytest.mark.parametrize("edit,match", [
    (lambda pin, root, ck, live: ck.write_bytes(b"swapped"), "checkpoint"),
    (lambda pin, root, ck, live: pin.update(checkpoint_sha256="b" * 64), "checkpoint"),
    (lambda pin, root, ck, live: pin.update(checkpoint_bytes=pin["checkpoint_bytes"] + 1), "bytes"),
    (lambda pin, root, ck, live: pin.update(origin_config_sha256="b" * 64), "config"),
    (lambda pin, root, ck, live: pin.update(origin_snapshot_sha256="b" * 64), "snapshot"),
    (lambda pin, root, ck, live: pin.update(origin_external_sha256="b" * 64), "external"),
    (lambda pin, root, ck, live: live.update(condition_sha256="b" * 64), "vocabulary"),
    (lambda pin, root, ck, live: live.update(split_sha256="b" * 64), "split"),
    (lambda pin, root, ck, live: live["config"]["routes"][ROUTE].update(updates=1), "route"),
    (lambda pin, root, ck, live: live["config"]["her2"].update(assay="other"), "her2"),
    (lambda pin, root, ck, live: live["external_files"].pop("data/a.py"), "external"),
    (lambda pin, root, ck, live: pin["block_digests"].update(route="b" * 64), "route"),
])
def test_each_drift_is_named(tmp_path, edit, match):
    root, checkpoint = make_origin(tmp_path)
    pin, live = make_pin(root, checkpoint), live_values(root)
    edit(pin, root, checkpoint, live)
    problems = admission.origin_problems(pin, ROUTE, live=live)
    assert problems and any(match in problem for problem in problems), problems


def test_a_missing_origin_file_is_a_problem_not_a_crash(tmp_path):
    root, checkpoint = make_origin(tmp_path)
    pin, live = make_pin(root, checkpoint), live_values(root)
    (root / "external_sources.json").unlink()
    assert any("external_sources.json" in p for p in admission.origin_problems(pin, ROUTE, live=live))


# ---------------------------------------------------------------------------
# copying, the record, and the check made on every load
# ---------------------------------------------------------------------------

def test_copy_verified_copies_exact_bytes(tmp_path):
    source = tmp_path / "s.bin"
    source.write_bytes(b"x" * 5000)
    target = tmp_path / "d" / "t.bin"
    out = admission.copy_verified(source, target, sha256=paths.sha256_file(source), nbytes=5000)
    assert out == target and target.read_bytes() == source.read_bytes()
    assert not list(target.parent.glob("*.tmp"))


def test_copy_verified_refuses_a_mismatch_and_leaves_nothing_behind(tmp_path):
    source = tmp_path / "s.bin"
    source.write_bytes(b"x" * 100)
    target = tmp_path / "d" / "t.bin"
    with pytest.raises(ValueError, match="sha256"):
        admission.copy_verified(source, target, sha256="0" * 64, nbytes=100)
    assert not target.exists() and not list(target.parent.glob("*"))


def test_copy_verified_never_overwrites_different_bytes(tmp_path):
    source = tmp_path / "s.bin"
    source.write_bytes(b"x" * 100)
    target = tmp_path / "t.bin"
    target.write_bytes(b"other")
    with pytest.raises(ValueError, match="already exists"):
        admission.copy_verified(source, target, sha256=paths.sha256_file(source), nbytes=100)
    assert target.read_bytes() == b"other"


def test_copy_verified_is_idempotent_for_identical_bytes(tmp_path):
    source = tmp_path / "s.bin"
    source.write_bytes(b"x" * 100)
    target = tmp_path / "t.bin"
    digest = paths.sha256_file(source)
    admission.copy_verified(source, target, sha256=digest, nbytes=100)
    admission.copy_verified(source, target, sha256=digest, nbytes=100)
    assert target.read_bytes() == source.read_bytes()


def admitted(tmp_path):
    """An origin, its pin, and a destination holding the verified copy and its record."""
    root, checkpoint = make_origin(tmp_path)
    pin = make_pin(root, checkpoint)
    dest = tmp_path / "dest" / "bakeoff" / ROUTE
    target = admission.copy_verified(checkpoint, dest / "best.pt", sha256=pin["checkpoint_sha256"],
                                     nbytes=pin["checkpoint_bytes"])
    record = admission.build_record(ROUTE, pin, origin_root=root, destination=target,
                                    best={"best_step": 9500, "best_metric": -0.68})
    admission.write_record(dest / admission.ADMISSION_FILE, record)
    return root, pin, dest, record


def test_the_record_round_trips_and_says_it_did_not_train(tmp_path):
    _root, pin, dest, record = admitted(tmp_path)
    again = admission.read_record(dest / admission.ADMISSION_FILE)
    assert again == record
    assert record["admitted"] is True and record["trained"] is False
    assert record["pin_sha256"] == admission.pin_digest(pin)
    assert record["selected_on"] == "development"
    assert record["best"] == {"best_step": 9500, "best_metric": -0.68}


def test_an_edited_record_is_refused_on_read(tmp_path):
    _root, _pin, dest, record = admitted(tmp_path)
    target = dest / admission.ADMISSION_FILE
    document = json.loads(target.read_text(encoding="utf-8"))
    document["best"]["best_step"] = 1
    target.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="record_sha256"):
        admission.read_record(target)


def test_an_admitted_checkpoint_that_is_intact_has_no_load_problems(tmp_path):
    _root, pin, dest, _record = admitted(tmp_path)
    assert admission.loadable_problems(ROUTE, dest / "best.pt", pin, live_config=live_config()) == []


def test_a_swapped_file_of_the_same_size_is_refused_on_every_load(tmp_path):
    _root, pin, dest, _record = admitted(tmp_path)
    target = dest / "best.pt"
    target.write_bytes(bytes(reversed(target.read_bytes())))
    assert target.stat().st_size == pin["checkpoint_bytes"]
    problems = admission.loadable_problems(ROUTE, target, pin, live_config=live_config())
    assert any("sha256" in p for p in problems)


def test_the_mere_presence_of_a_record_does_not_authorise_a_different_pin(tmp_path):
    _root, pin, dest, _record = admitted(tmp_path)
    forged = dict(pin, checkpoint_step=1)
    problems = admission.loadable_problems(ROUTE, dest / "best.pt", forged, live_config=live_config())
    assert any("pin" in p for p in problems), problems


def test_a_missing_record_and_a_record_for_another_route_are_refused(tmp_path):
    _root, pin, dest, _record = admitted(tmp_path)
    (dest / admission.ADMISSION_FILE).unlink()
    assert any("admission.json" in p for p in
               admission.loadable_problems(ROUTE, dest / "best.pt", pin, live_config=live_config()))
    record = admission.build_record("piggen_causal_infiller", pin, origin_root=tmp_path / "origin",
                                    destination=dest / "best.pt", best={})
    admission.write_record(dest / admission.ADMISSION_FILE, record)
    assert any("route" in p for p in
               admission.loadable_problems(ROUTE, dest / "best.pt", pin, live_config=live_config()))


def test_a_drifted_live_route_block_is_refused_on_every_load(tmp_path):
    _root, pin, dest, _record = admitted(tmp_path)
    config = live_config()
    config["routes"][ROUTE]["max_positions"] = 512
    problems = admission.loadable_problems(ROUTE, dest / "best.pt", pin, live_config=config)
    assert any("route" in p for p in problems)


def test_a_missing_checkpoint_file_is_a_problem(tmp_path):
    _root, pin, dest, _record = admitted(tmp_path)
    (dest / "best.pt").unlink()
    assert any("best.pt" in p for p in
               admission.loadable_problems(ROUTE, dest / "best.pt", pin, live_config=live_config()))


def test_the_digest_cache_rehashes_when_the_file_changes(tmp_path):
    cache = admission.DigestCache()
    target = tmp_path / "f.bin"
    target.write_bytes(b"a" * 10)
    first = cache.sha256(target)
    assert cache.sha256(target) == first
    target.write_bytes(b"b" * 11)
    assert cache.sha256(target) != first


# ---------------------------------------------------------------------------
# astra's review of the finished code (2026-10-07): findings 2 and 4 and the overlap guard
# ---------------------------------------------------------------------------

def _redigest(path, mutate):
    """Edit a record the way an attacker would: change it, then recompute its own digest."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    document.pop("record_sha256")
    mutate(document)
    document["record_sha256"] = paths.digest_document(document)
    Path(path).write_text(json.dumps(document), encoding="utf-8")


def test_the_aggregate_is_the_one_the_spec_computes():
    files = {"b": "2" * 64, "a": "1" * 64}
    assert admission.external_aggregate(files) == paths.sha256_text(f"a {'1' * 64}\nb {'2' * 64}")


def test_an_origin_table_that_does_not_aggregate_to_its_digest_is_refused(tmp_path):
    """An edited ``files`` mapping under an unchanged claimed digest must not be trusted."""
    root, checkpoint = make_origin(tmp_path)
    pin, live = make_pin(root, checkpoint), live_values(root)
    marker = paths.read_json(root / "external_sources.json")
    marker["files"]["data/a.py"] = "e" * 64            # the origin's table is edited ...
    paths.write_json(root / "external_sources.json", marker)   # ... its claimed digest is not
    live["external_files"]["data/a.py"] = "e" * 64     # and the destination is made to agree
    problems = admission.origin_problems(pin, ROUTE, live=live)
    assert any("aggregate" in problem for problem in problems), problems


def test_a_record_whose_embedded_pin_was_edited_and_redigested_is_refused(tmp_path):
    _root, pin, dest, _record = admitted(tmp_path)
    _redigest(dest / admission.ADMISSION_FILE, lambda d: d["pin"].update(arm="null"))
    problems = admission.loadable_problems(ROUTE, dest / "best.pt", pin, live_config=live_config())
    assert any("embedded pin" in problem for problem in problems), problems


def test_record_best_must_match_the_verified_checkpoint_payload(tmp_path):
    _root, pin, dest, _record = admitted(tmp_path)
    payload = {"step": pin["checkpoint_step"], "best_step": 9500, "best_metric": -0.68}
    assert admission.record_payload_problems(dest / admission.ADMISSION_FILE, payload) == []
    _redigest(dest / admission.ADMISSION_FILE, lambda d: d["best"].update(best_step=1))
    problems = admission.record_payload_problems(dest / admission.ADMISSION_FILE, payload)
    assert any("best_step" in problem for problem in problems), problems


def test_a_payload_with_another_step_than_the_record_is_refused(tmp_path):
    _root, pin, dest, _record = admitted(tmp_path)
    payload = {"step": pin["checkpoint_step"] + 1, "best_step": 9500, "best_metric": -0.68}
    assert any("step" in p for p in
               admission.record_payload_problems(dest / admission.ADMISSION_FILE, payload))


@pytest.mark.parametrize("run_root", ["{origin}", "{origin}/nested/run", "{parent}"])
def test_a_destination_overlapping_an_origin_flight_is_refused(tmp_path, run_root):
    origin = tmp_path / "runs" / "t1" / "flight"
    origin.mkdir(parents=True)
    pins = {ROUTE: dict(good_pin(), origin_flight=str(origin))}
    target = run_root.format(origin=origin, parent=origin.parent)
    with pytest.raises(ValueError, match="overlaps"):
        admission.require_disjoint(Path(target), pins)


def test_a_disjoint_destination_is_accepted(tmp_path):
    origin = tmp_path / "runs" / "t1" / "flight"
    origin.mkdir(parents=True)
    pins = {ROUTE: dict(good_pin(), origin_flight=str(origin))}
    admission.require_disjoint(tmp_path / "runs" / "admitted" / "flight", pins)
    admission.require_disjoint(tmp_path / "runs" / "t1" / "flight_other", pins)


# ---------------------------------------------------------------------------
# astra round 2: no memoised digest may vouch for bytes that were replaced
# ---------------------------------------------------------------------------

def test_a_same_size_replacement_with_a_restored_mtime_is_still_detected(tmp_path):
    import os
    target = tmp_path / "f.bin"
    target.write_bytes(b"a" * 4096)
    stat = target.stat()
    cache = admission.DigestCache()
    first = cache.sha256(target)
    target.write_bytes(b"b" * 4096)                       # same length ...
    os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns))   # ... and the old timestamp
    assert target.stat().st_size == stat.st_size and target.stat().st_mtime_ns == stat.st_mtime_ns
    assert cache.sha256(target) != first


def test_an_admitted_checkpoint_replaced_with_its_timestamp_restored_is_refused(tmp_path):
    import os
    _root, pin, dest, _record = admitted(tmp_path)
    target = dest / "best.pt"
    assert admission.loadable_problems(ROUTE, target, pin, live_config=live_config()) == []
    stat = target.stat()
    data = bytearray(target.read_bytes())
    data[0] ^= 0xFF
    target.write_bytes(bytes(data))
    os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    problems = admission.loadable_problems(ROUTE, target, pin, live_config=live_config())
    assert any("sha256" in problem for problem in problems), problems


# ---------------------------------------------------------------------------
# sol round 3: a link planted at a temporary name must never be written through
# ---------------------------------------------------------------------------

def _link(kind, link, target):
    """A hard link (no privilege on NTFS) or a symbolic link (skipped when not permitted)."""
    import os
    if kind == "hard":
        os.link(target, link)
        return
    try:
        os.symlink(target, link)
    except OSError as error:
        pytest.skip(f"cannot create a symbolic link here: {error}")


@pytest.mark.parametrize("kind", ["hard", "symbolic"])
def test_copy_verified_never_writes_through_a_link_at_its_temporary_name(tmp_path, kind):
    origin_file = tmp_path / "origin" / "best.pt"
    origin_file.parent.mkdir()
    origin_file.write_bytes(b"ORIGIN" * 100)
    source = tmp_path / "s.bin"
    source.write_bytes(b"x" * 5000)
    target = tmp_path / "d" / "t.bin"
    target.parent.mkdir()
    _link(kind, target.with_name(target.name + ".tmp"), origin_file)
    admission.copy_verified(source, target, sha256=paths.sha256_file(source), nbytes=5000)
    assert origin_file.read_bytes() == b"ORIGIN" * 100, "the origin file was written through a link"
    assert target.read_bytes() == source.read_bytes()


@pytest.mark.parametrize("kind", ["hard", "symbolic"])
def test_write_record_never_writes_through_a_link_at_its_temporary_name(tmp_path, kind):
    root, pin, dest, record = admitted(tmp_path)
    origin_file = tmp_path / "origin_record.json"
    origin_file.write_bytes(b"ORIGIN-RECORD" * 50)
    record_path = dest / admission.ADMISSION_FILE
    record_path.unlink()
    _link(kind, record_path.with_name(record_path.name + ".tmp"), origin_file)
    admission.write_record(record_path, record)
    assert origin_file.read_bytes() == b"ORIGIN-RECORD" * 50, "the origin file was written through a link"
    assert admission.read_record(record_path)["route"] == ROUTE


# ---------------------------------------------------------------------------
# sol round 7: the run root is checked ONCE for every precondition of a redirected write
# ---------------------------------------------------------------------------

def _junction_to(link, target):
    import subprocess
    done = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                          capture_output=True, text=True)
    if done.returncode:
        pytest.skip(f"cannot create a directory junction here: {done.stdout} {done.stderr}")


def _tree(tmp_path):
    root = tmp_path / "run"
    (root / "stages").mkdir(parents=True)
    (root / "stages" / "profile.json").write_text("{}", encoding="utf-8")
    (root / "bakeoff" / "x").mkdir(parents=True)
    (root / "bakeoff" / "x" / "best.pt").write_bytes(b"checkpoint")
    return root


def test_a_clean_run_root_and_an_absent_one_have_no_problems(tmp_path):
    assert admission.run_root_problems(tmp_path / "absent") == []
    assert admission.run_root_problems(_tree(tmp_path)) == []


def test_a_hard_linked_file_anywhere_in_the_run_root_is_found(tmp_path):
    import os
    root = _tree(tmp_path)
    origin_file = tmp_path / "origin" / "best.pt"
    origin_file.parent.mkdir()
    origin_file.write_bytes(b"ORIGIN")
    os.link(origin_file, root / "campaign.owners.jsonl")
    problems = admission.run_root_problems(root)
    assert len(problems) == 1 and "campaign.owners.jsonl" in problems[0] and "hard link" in problems[0]
    with pytest.raises(ValueError, match="hard link"):
        admission.require_clean_run_root(root)


def test_a_junction_is_found_and_never_traversed(tmp_path):
    import os
    root = _tree(tmp_path)
    elsewhere = tmp_path / "origin" / "stages"
    elsewhere.mkdir(parents=True)
    os.link(root / "stages" / "profile.json", elsewhere / "linked.json")   # inside the target
    _junction_to(root / "source_snapshot", elsewhere)
    try:
        problems = admission.run_root_problems(root)
        assert any("source_snapshot" in line and "junction or symbolic link" in line
                   for line in problems), problems
        assert not any("linked.json" in line for line in problems), (
            "the walk followed a junction into its target")
    finally:
        os.rmdir(root / "source_snapshot")


def test_a_symbolic_link_is_found(tmp_path):
    import os
    root = _tree(tmp_path)
    target = tmp_path / "origin_file"
    target.write_bytes(b"x")
    try:
        os.symlink(target, root / "link")
    except OSError as error:
        pytest.skip(f"cannot create a symbolic link here: {error}")
    assert any("link" in line and "junction or symbolic link" in line
               for line in admission.run_root_problems(root))


def test_a_run_root_that_is_itself_a_junction_is_found(tmp_path):
    import os
    real = tmp_path / "elsewhere"
    real.mkdir()
    link = tmp_path / "run"
    _junction_to(link, real)
    try:
        assert any("the run root itself" in line for line in admission.run_root_problems(link))
    finally:
        os.rmdir(link)


@pytest.mark.parametrize("writer", ["copy_verified", "write_record"])
def test_the_admission_writers_release_the_descriptor_when_fdopen_fails(tmp_path, writer, monkeypatch):
    import tempfile
    root, pin, dest, record = admitted(tmp_path)
    created = []
    real = tempfile.mkstemp

    def recording(*args, **kwargs):
        descriptor, name = real(*args, **kwargs)
        created.append(descriptor)
        return descriptor, name
    monkeypatch.setattr(tempfile, "mkstemp", recording)

    def refuse(*_args, **_kwargs):
        raise MemoryError("simulated failure while wrapping the descriptor")
    import os
    monkeypatch.setattr(os, "fdopen", refuse)
    if writer == "write_record":
        target = dest / "second" / admission.ADMISSION_FILE
        call = lambda: admission.write_record(target, record)
    else:
        target = dest / "second" / "best.pt"
        checkpoint = admission.origin_checkpoint_path(pin, ROUTE)
        call = lambda: admission.copy_verified(checkpoint, target, sha256=pin["checkpoint_sha256"],
                                               nbytes=pin["checkpoint_bytes"])
    with pytest.raises(MemoryError, match="simulated"):
        call()
    assert len(created) == 1, created
    with pytest.raises(OSError):
        os.fstat(created[0])
    assert not target.exists() and not list(target.parent.iterdir())

