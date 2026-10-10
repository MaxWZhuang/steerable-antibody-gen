"""Admission through the real services: a tiny DPLM trained in one root, admitted into another.

The origin and the destination are two different run roots with different config files and
different freeze markers, so an admitted checkpoint genuinely carries a foreign identity. What is
asserted:

* the foreign-identity branch accepts exactly the admitted ``best.pt`` of an admitted route, and
  refuses the same bytes at any other path, another file in the same directory, a swapped file
  (including another checkpoint of the SAME training run), a wrong step, a drifted live route
  block and a missing or edited record;
* only the three pinned identity fields come from the origin; route, arm, seed, vocabulary and
  split stay live-checked;
* ``admit_route`` refuses an incompatible origin before it writes anything, and never writes into
  the origin;
* the mixed bakeoff admits the three finished routes and trains only dWJS, and nothing trains over
  an admitted checkpoint.

Nothing here says anything about a trained DPLM's quality; the models are random-init and tiny.
"""
from __future__ import annotations

import copy
import json
import shutil
import types
from pathlib import Path

import pytest
import torch

from smallAntibodyGen.experiments import her2_routes_admission as admission
from smallAntibodyGen.experiments import her2_routes_services as services_mod
from smallAntibodyGen.experiments import her2_routes_spec as spec
from smallAntibodyGen.experiments import her2_support_paths as paths
from smallAntibodyGen.models import dplm_policy as dplm
from smallAntibodyGen.tests.test_her2_routes_admission import good_pin
from smallAntibodyGen.tests.test_her2_routes_dplm_services import (
    ROUTE_FULL,
    Context,
    fake_scaffold,
    make_config,
    reconstruction_stream,
    tiny_native_config,
    write_pins,
)

VOCABULARY = '{"bins": ["high", "null"]}\n'
SPLIT = {"sequence_sha256": "s" * 64, "n": 100}
SELECTOR = "development_affinity_spearman_high"
GATES = json.loads(json.dumps(spec.ADMISSION_GATE_LOCK))


def write_markers(run_root, *, snapshot, external_files):
    run_root.mkdir(parents=True, exist_ok=True)
    paths.write_json(run_root / "source_snapshot.json", {"snapshot_sha256": snapshot})
    paths.write_json(run_root / "external_sources.json",
                     {"external_sha256": admission.external_aggregate(external_files),
                      "files": dict(external_files)})
    (run_root / "condition_vocabulary.json").write_text(VOCABULARY, encoding="utf-8")
    paths.write_json(run_root / "hard_split_freeze.json", dict(SPLIT))


def build_services(root, config, monkeypatch):
    def fake_loader(weights_dir, *, pins, variant, device="cpu", storage_dtype=torch.float32,
                    dropout=None):
        return dplm.NativeDPLM.random_init(tiny_native_config(), seed=3, std=0.3, label=variant,
                                           storage_dtype=storage_dtype)
    monkeypatch.setattr(dplm, "load_pinned_dplm", fake_loader)
    Path(root).mkdir(parents=True, exist_ok=True)
    write_pins(root)
    context = Context(root, config)
    services = services_mod.FlightServices(context, device="cpu")
    services._cache["scaffold"] = fake_scaffold()
    return services, context


def train_origin(tmp_path, monkeypatch):
    """A finished origin flight: a trained tiny DPLM with a best.pt (step 2) and a last.pt (step 4)."""
    root = tmp_path / "origin"
    config = make_config()
    services, context = build_services(root, config, monkeypatch)
    write_markers(context.run_root, snapshot="a" * 64,
                  external_files={"data/a.py": paths.sha256_text("a")})
    snapshot = context.run_root / "source_snapshot" / "configs" / "experiments"
    snapshot.mkdir(parents=True)
    shutil.copyfile(context.config_path, snapshot / "origin.json")
    directory = context.path("bakeoff", ROUTE_FULL)
    services.train(ROUTE_FULL, stream=reconstruction_stream(services), updates=4, seed=5,
                   directory=directory, evaluate=lambda model, heads, step: abs(step - 2) + 1.0)
    return context, directory, config


def destination_config(origin_config, pin):
    config = make_config()
    config["routes"]["enabled"] = list(spec.ADMISSION_PHASE_ROUTES)
    for route in ("ablang2_masked_editor", "piggen_causal_infiller"):
        config["routes"][route] = {"seed": 1}
    config["routes"]["dwjs_energy"] = {"seed": 1, "denoiser_updates": 1, "energy_updates": 1}
    # ROUTE_FULL is the string "dplm_150m_masked": the destination's block IS the origin's.
    config["routes"][ROUTE_FULL] = copy.deepcopy(origin_config["routes"][ROUTE_FULL])
    config["routes"].pop("dplm_650m_lora_masked", None)
    config["gates"] = dict(GATES, generation_endpoint="prose")
    config["decoding"] = {"seed": 1}
    config["revisions"] = {"kind": "development"}
    config["admission_phase"] = spec.admission_phase_declaration()
    config["admission"] = {"ablang2_masked_editor": good_pin(),
                           "piggen_causal_infiller": good_pin(),
                           "dplm_150m_masked": pin}
    return config


ROUTE = ROUTE_FULL


@pytest.fixture
def world(tmp_path, monkeypatch):
    origin_context, origin_dir, origin_config = train_origin(tmp_path, monkeypatch)
    probe = destination_config(origin_config, good_pin())
    pin = admission.derive_pin(
        origin_context.run_root, route=ROUTE_FULL, origin_config="configs/experiments/origin.json",
        checkpoint=origin_dir / "best.pt",
        checkpoint_step=int(torch.load(origin_dir / "best.pt", map_location="cpu", mmap=True,
                                       weights_only=False)["step"]),
        arm="full", seed=5, schedule_sha256=None, selector=SELECTOR, live_config=probe)
    config = destination_config(origin_config, pin)
    services, context = build_services(tmp_path / "dest", config, monkeypatch)
    write_markers(context.run_root, snapshot="b" * 64,
                  external_files={"data/a.py": paths.sha256_text("a"),
                                  "models/dplm_150m/config.json": "1" * 64})
    return types.SimpleNamespace(services=services, context=context, config=config, pin=pin,
                                 origin=origin_context, origin_dir=origin_dir, root=tmp_path)


def admitted_path(world):
    return world.context.path("bakeoff", ROUTE, "best.pt")


def admit(world):
    return world.services.admit_route(ROUTE, world.pin)


# ---------------------------------------------------------------------------
# the pinned identity
# ---------------------------------------------------------------------------

def test_the_pin_overrides_exactly_three_identity_fields(world):
    services = world.services
    plain = services.training_identity(ROUTE, arm="full", seed=5)
    pinned = services.training_identity(ROUTE, arm="full", seed=5, pinned=world.pin)
    moved = {key for key in set(plain) | set(pinned) if plain.get(key) != pinned.get(key)}
    assert moved == {"config_sha256", "snapshot_sha256", "external_sha256", "identity_sha256"}
    assert pinned["config_sha256"] == world.pin["origin_config_sha256"]
    for key in ("route", "arm", "seed", "condition_vocabulary_sha256", "split_sha256",
                "schedule_sha256"):
        assert pinned[key] == plain[key], key
    assert pinned.get("native_sha256") == plain.get("native_sha256")


def test_a_flight_without_the_phase_has_no_admitted_paths(tmp_path, monkeypatch):
    services, context = build_services(tmp_path / "legacy", make_config(), monkeypatch)
    assert services.admission_policy() is None
    assert services.admitted_pin(ROUTE_FULL, context.path("bakeoff", ROUTE_FULL, "best.pt")) is None


# ---------------------------------------------------------------------------
# admitting
# ---------------------------------------------------------------------------

def test_admission_copies_the_pinned_bytes_and_writes_a_record(world):
    record = admit(world)
    target = admitted_path(world)
    assert target.is_file() and paths.sha256_file(target) == world.pin["checkpoint_sha256"]
    assert record["admitted"] is True and record["trained"] is False
    assert record["step"] == world.pin["checkpoint_step"] and record["selected_on"] == "development"
    written = admission.read_record(target.parent / admission.ADMISSION_FILE)
    assert written["pin_sha256"] == admission.pin_digest(world.pin)
    assert not (target.parent / "run.json").exists(), "a record, not a fake training run"


def test_the_origin_flight_is_never_written(world):
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns)
              for p in world.origin.run_root.rglob("*") if p.is_file()}
    admit(world)
    world.services.load_for_scoring(ROUTE, admitted_path(world))
    assert {p: (p.read_bytes(), p.stat().st_mtime_ns)
            for p in world.origin.run_root.rglob("*") if p.is_file()} == before


def test_an_incompatible_vocabulary_refuses_before_anything_is_written(world):
    world.context.path("condition_vocabulary.json").write_text('{"bins": ["other"]}\n',
                                                               encoding="utf-8")
    with pytest.raises(ValueError, match="vocabulary"):
        admit(world)
    assert not world.context.path("bakeoff").exists()


def test_a_missing_external_file_refuses_before_anything_is_written(world):
    paths.write_json(world.context.path("external_sources.json"),
                     {"external_sha256": "x" * 64, "files": {}})
    with pytest.raises(ValueError, match="external table"):
        admit(world)
    assert not world.context.path("bakeoff").exists()


def test_the_checkpoints_own_identity_must_match_the_expected_one(world):
    pin = dict(world.pin, arm="null")
    with pytest.raises(ValueError, match="different identity.*arm|arm"):
        world.services.admit_route(ROUTE, pin)
    assert not admitted_path(world).exists()


def test_a_wrong_step_in_the_pin_is_refused(world):
    pin = dict(world.pin, checkpoint_step=world.pin["checkpoint_step"] + 1)
    with pytest.raises(ValueError, match="step"):
        world.services.admit_route(ROUTE, pin)
    assert not admitted_path(world).exists()


def test_admitting_twice_is_idempotent(world):
    admit(world)
    first = paths.sha256_file(admitted_path(world))
    admit(world)
    assert paths.sha256_file(admitted_path(world)) == first


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def test_the_admitted_checkpoint_loads_under_its_foreign_identity(world):
    admit(world)
    model, heads, record = world.services.load_for_scoring(ROUTE, admitted_path(world))
    assert record["trained"] is True and record["step"] == world.pin["checkpoint_step"]
    assert not model.training and heads is not None


def test_the_same_bytes_at_any_other_path_are_refused_by_the_ordinary_identity(world):
    admit(world)
    elsewhere = world.context.path("elsewhere", "best.pt")
    elsewhere.parent.mkdir(parents=True)
    shutil.copyfile(admitted_path(world), elsewhere)
    with pytest.raises(ValueError, match="different identity"):
        world.services.load_for_scoring(ROUTE, elsewhere)


def test_another_file_in_the_admitted_directory_is_not_admitted(world):
    admit(world)
    sibling = admitted_path(world).with_name("last.pt")
    shutil.copyfile(admitted_path(world), sibling)
    assert world.services.admitted_pin(ROUTE, sibling) is None
    with pytest.raises(ValueError, match="different identity"):
        world.services.load_for_scoring(ROUTE, sibling)


def test_another_checkpoint_of_the_same_training_run_is_refused(world):
    """Same identity, different step: the exact swap an identity-only check would let through."""
    admit(world)
    last = world.origin_dir / "last.pt"
    assert int(torch.load(last, map_location="cpu", mmap=True, weights_only=False)["step"]) == 4
    shutil.copyfile(last, admitted_path(world))
    with pytest.raises(ValueError, match="not the admitted"):
        world.services.load_for_scoring(ROUTE, admitted_path(world))


def test_a_wrong_payload_step_is_refused_even_when_the_hash_agrees(world, monkeypatch):
    admit(world)
    monkeypatch.setattr(admission, "loadable_problems", lambda *a, **k: [])
    world.config["admission"][ROUTE] = dict(world.pin,
                                            checkpoint_step=world.pin["checkpoint_step"] + 1)
    with pytest.raises(ValueError, match="payload is step"):
        world.services.load_for_scoring(ROUTE, admitted_path(world))


def test_a_drifted_live_route_block_is_refused_on_load(world):
    admit(world)
    world.services.load_for_scoring(ROUTE, admitted_path(world))
    world.config["routes"][ROUTE]["max_positions"] = 128
    with pytest.raises(ValueError, match="route block"):
        world.services.load_for_scoring(ROUTE, admitted_path(world))


def test_a_changed_pin_in_the_config_is_not_authorised_by_the_record(world):
    admit(world)
    world.config["admission"][ROUTE] = dict(world.pin, selector="something else")
    with pytest.raises(ValueError, match="pin"):
        world.services.load_for_scoring(ROUTE, admitted_path(world))


def test_a_missing_or_edited_record_is_refused_on_load(world):
    admit(world)
    record = admitted_path(world).with_name(admission.ADMISSION_FILE)
    document = json.loads(record.read_text(encoding="utf-8"))
    document["best"] = {"best_step": 1}
    record.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="record_sha256"):
        world.services.load_for_scoring(ROUTE, admitted_path(world))
    record.unlink()
    with pytest.raises(ValueError, match="admission.json"):
        world.services.load_for_scoring(ROUTE, admitted_path(world))


def test_the_zero_shot_backbone_still_loads_without_any_checkpoint(world):
    model, _heads, record = world.services.load_for_scoring(ROUTE, None)
    assert record["trained"] is False and model is not None


# ---------------------------------------------------------------------------
# the gate's view of an admitted route
# ---------------------------------------------------------------------------

def test_the_gate_reads_best_from_the_record_and_trained_routes_from_run_json(world):
    admit(world)
    record = world.services.bakeoff_record(ROUTE)
    assert record["admitted"] is True and record["trained"] is False
    assert record["best"]["best_step"] == world.pin["checkpoint_step"]
    trained = world.context.path("bakeoff", "dwjs_energy")
    trained.mkdir(parents=True)
    paths.write_json(trained / "run.json", {"best": {"step": 7}})
    assert world.services.bakeoff_record("dwjs_energy") == {"best": {"step": 7}}


# ---------------------------------------------------------------------------
# the mixed bakeoff
# ---------------------------------------------------------------------------

def stub_energy(world, monkeypatch, calls):
    def fake_train_energy(*, directory, conditioned, population, seed, heartbeat=None, **_):
        calls.append(("energy", Path(directory).name, bool(conditioned)))
        return {"updates": 2, "phase_updates": {"denoiser": 1, "energy": 1}, "exposure": {},
                "identity": {}, "energy_contract_sha256": "c" * 64, "exposure_trace": {}}
    monkeypatch.setattr(world.services, "train_energy", fake_train_energy)
    monkeypatch.setattr(services_mod.energy_contract, "require_matched_controls",
                        lambda full, baseline: {"matched": True})
    monkeypatch.setattr(world.services, "train",
                        lambda *a, **k: pytest.fail("no likelihood route may train in this flight"))
    monkeypatch.setattr(spec, "verify_training_phase_split", lambda *a, **k: None)


def test_the_mixed_bakeoff_admits_three_routes_and_trains_only_dwjs(world, monkeypatch):
    calls = []
    stub_energy(world, monkeypatch, calls)
    admitted = []

    def fake_admit(route, pin):
        admitted.append(route)
        return {"route": route, "admitted": True, "trained": False}
    monkeypatch.setattr(world.services, "admit_route", fake_admit)
    result = world.services.run_bakeoff()
    assert admitted == ["ablang2_masked_editor", "piggen_causal_infiller", "dplm_150m_masked"]
    assert calls == [("energy", "dwjs_energy", True), ("energy", "density_only", False)]
    assert result["all_terminal"] is True and result["remaining"] == []
    assert set(result["completed"]) == set(spec.ADMISSION_PHASE_ROUTES)
    assert result["admission_phase"]["phase_id"] == "admitted_route_gate"
    assert "training_phase" not in result


def test_a_trained_route_directory_that_holds_an_admission_record_is_refused(world, monkeypatch):
    calls = []
    stub_energy(world, monkeypatch, calls)
    monkeypatch.setattr(world.services, "admit_route",
                        lambda route, pin: {"route": route, "admitted": True})
    directory = world.context.path("bakeoff", "dwjs_energy")
    directory.mkdir(parents=True)
    (directory / admission.ADMISSION_FILE).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="admitted checkpoint"):
        world.services.run_bakeoff()
    assert calls == []


def test_nothing_trains_over_an_admitted_checkpoint(world):
    admit(world)
    directory = admitted_path(world).parent
    before = paths.sha256_file(admitted_path(world))
    with pytest.raises(ValueError, match="admitted checkpoint"):
        world.services.train(ROUTE, stream=reconstruction_stream(world.services), updates=1,
                             seed=5, directory=directory)
    with pytest.raises(ValueError, match="admitted checkpoint"):
        world.services.train_energy(directory=directory, seed=1, population="x", conditioned=True)
    assert paths.sha256_file(admitted_path(world)) == before


def test_a_legacy_bakeoff_is_unchanged_by_the_new_branches(tmp_path, monkeypatch):
    services, context = build_services(tmp_path / "legacy", make_config(), monkeypatch)
    trained = []
    monkeypatch.setattr(services, "train", lambda route, **k: trained.append(route) or
                        {"updates": int(k["updates"])})
    monkeypatch.setattr(services, "curriculum_stream", lambda *a, **k: None)
    monkeypatch.setattr(services, "selection_metric", lambda route: None)
    result = services.run_bakeoff()
    assert trained == [ROUTE_FULL, "dplm_650m_lora_masked"]
    assert "admission_phase" not in result and result["all_terminal"] is True


# ---------------------------------------------------------------------------
# the forecast counts imports as imports
# ---------------------------------------------------------------------------

def test_admitted_routes_are_forecast_as_imports_not_training_schedules(world):
    per_route = {"ablang2_masked_editor": {"hours": 29.0, "microbatch": 4},
                 "dplm_150m_masked": {"hours": 101.0, "microbatch": 4},
                 "dwjs_energy": {"hours": 1.3, "microbatch": 8}}
    out = world.services.forecast_imports(per_route)
    assert out["dwjs_energy"] == per_route["dwjs_energy"]
    for route in ("ablang2_masked_editor", "dplm_150m_masked"):
        assert out[route]["imported"] is True and out[route]["hours"] == 0.0
        assert out[route]["hours_if_trained_here"] == per_route[route]["hours"]
    assert sum(block["hours"] for block in out.values()) == pytest.approx(1.3)


def test_an_imported_route_carries_every_counter_the_profile_report_prints(world):
    """astra, final pass: scripts/her2_routes_gpu_profile.py prints hours, optimizer_updates and
    row_presentations for EVERY route; an imported block that omitted one crashed the report with a
    KeyError after the profile had finished."""
    per_route = {"ablang2_masked_editor": {"hours": 29.0, "microbatch": 4, "optimizer_updates": 20000,
                                           "row_presentations": 1280000},
                 "dwjs_energy": {"hours": 1.3, "microbatch": 8, "optimizer_updates": 12000,
                                 "row_presentations": 768000}}
    out = world.services.forecast_imports(per_route)
    for route, block in sorted(out.items()):
        line = (f"  {route}: {block['hours']:.2f} h  {block['optimizer_updates']} updates  "
                f"{block['row_presentations']} row presentations")
        assert "row presentations" in line
    imported = out["ablang2_masked_editor"]
    assert imported["row_presentations"] == 0 and imported["optimizer_updates"] == 0
    assert imported.get("basis"), "the report prints accounting-or-basis; an import must explain itself"
    assert out["dwjs_energy"]["row_presentations"] == 768000          # a trained route is unchanged


def test_a_flight_without_the_phase_forecasts_every_route_as_before(tmp_path, monkeypatch):
    services, _context = build_services(tmp_path / "legacy", make_config(), monkeypatch)
    per_route = {"ablang2_masked_editor": {"hours": 29.0}}
    assert services.forecast_imports(per_route) is per_route


# ---------------------------------------------------------------------------
# an agreeing second freeze must not move the split digest every identity is bound to
# ---------------------------------------------------------------------------
#
# Found on the first real admission flight: ``split()`` re-derives the split in every fresh process
# and ``freeze_hard_split(persist=True)`` rewrote the record with a ``reconfirmed`` stamp, which
# changed its digest (3955a598... -> b6342175...) and orphaned every checkpoint bound to the
# pinned one. The content of the split was identical; only the digest moved.

def test_the_first_freeze_is_written(tmp_path):
    target = tmp_path / "hard_split_freeze.json"
    freeze = {"sequence_sha256": "a" * 64, "development": ["x"], "reserved": ["y"]}
    services_mod.FlightServices.persist_split_freeze(target, freeze, existing=None)
    assert paths.read_json(target) == freeze


def test_an_agreeing_second_freeze_leaves_the_record_byte_for_byte_alone(tmp_path):
    target = tmp_path / "hard_split_freeze.json"
    original = {"sequence_sha256": "a" * 64, "development": ["x"], "reserved": ["y"]}
    paths.write_json(target, original)
    before = (target.read_bytes(), target.stat().st_mtime_ns, paths.digest_document(original))
    stamped = dict(original, reconfirmed=True)
    services_mod.FlightServices.persist_split_freeze(target, stamped, existing=original)
    assert (target.read_bytes(), target.stat().st_mtime_ns) == before[:2]
    assert paths.digest_document(paths.read_json(target)) == before[2]
    assert "reconfirmed" not in paths.read_json(target)


def test_a_dry_run_never_writes_and_the_flag_is_honoured(tmp_path):
    target = tmp_path / "hard_split_freeze.json"
    assert services_mod.FlightServices.persist_split_freeze(
        target, {"a": 1}, existing=None, persist=False) is False
    assert not target.exists()


# ---------------------------------------------------------------------------
# astra's review of the finished code (2026-10-07)
# ---------------------------------------------------------------------------

def test_an_explicit_arm_that_contradicts_the_pin_is_refused(world):
    """The ordinary branch refuses a wrong arm; admission must not disable that check."""
    admit(world)
    with pytest.raises(ValueError, match="arm"):
        world.services.load_for_scoring(ROUTE, admitted_path(world), arm="null")
    world.services.load_for_scoring(ROUTE, admitted_path(world), arm="full")
    world.services.load_for_scoring(ROUTE, admitted_path(world))


def _redigest(path, mutate):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    document.pop("record_sha256")
    mutate(document)
    document["record_sha256"] = paths.digest_document(document)
    Path(path).write_text(json.dumps(document), encoding="utf-8")


def test_a_redigested_record_with_a_fabricated_best_step_cannot_be_published(world):
    """``best`` is published by the gate; it must come from the verified checkpoint, not the record."""
    admit(world)
    record = admitted_path(world).with_name(admission.ADMISSION_FILE)
    _redigest(record, lambda d: d["best"].update(best_step=1))
    with pytest.raises(ValueError, match="best_step"):
        world.services.load_for_scoring(ROUTE, admitted_path(world))


def test_a_redigested_record_with_an_edited_embedded_pin_is_refused_by_the_gate_view(world):
    admit(world)
    record = admitted_path(world).with_name(admission.ADMISSION_FILE)
    _redigest(record, lambda d: d["pin"].update(arm="null"))
    with pytest.raises(ValueError, match="embedded pin"):
        world.services.bakeoff_record(ROUTE)


def test_admitting_into_the_origin_flight_is_refused_before_anything_is_written(world):
    pin = dict(world.pin, origin_flight=str(world.context.run_root))
    with pytest.raises(ValueError, match="overlaps"):
        world.services.admit_route(ROUTE, pin)
    assert not world.context.path("bakeoff").exists()


# ---------------------------------------------------------------------------
# astra round 2: a junction inside the destination must not redirect the write into an origin
# ---------------------------------------------------------------------------

def _junction(link, target):
    import subprocess
    done = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                          capture_output=True, text=True)
    if done.returncode:
        pytest.skip(f"cannot create a directory junction here: {done.stdout} {done.stderr}")


def test_a_junctioned_bakeoff_directory_cannot_redirect_the_record_into_the_origin(world):
    import os
    bakeoff = world.context.path("bakeoff")
    bakeoff.mkdir(parents=True)
    link = bakeoff / ROUTE
    _junction(link, world.origin_dir)
    try:
        before = sorted(p.name for p in world.origin_dir.iterdir())
        with pytest.raises(ValueError, match="overlaps"):
            admit(world)
        assert sorted(p.name for p in world.origin_dir.iterdir()) == before
        assert not (world.origin_dir / admission.ADMISSION_FILE).exists()
    finally:
        os.rmdir(link)        # removes the link only, never the origin's contents


# ---------------------------------------------------------------------------
# sol round 3: the destination is checked against EVERY pinned origin, not only its own route's
# ---------------------------------------------------------------------------

def test_a_bakeoff_directory_junctioned_into_another_routes_origin_is_refused(world):
    import os
    other_origin = world.root / "other_origin"
    unused = other_origin / "bakeoff" / "unused"
    unused.mkdir(parents=True)
    # another route's finished flight, as the committed config pins it
    world.config["admission"]["ablang2_masked_editor"]["origin_flight"] = str(other_origin)
    assert spec.admission_pins(world.services.config)["ablang2_masked_editor"][
        "origin_flight"] == str(other_origin), "the services must read the pins this test edits"
    bakeoff = world.context.path("bakeoff")
    bakeoff.mkdir(parents=True)
    link = bakeoff / ROUTE
    _junction(link, unused)
    try:
        with pytest.raises(ValueError, match="overlaps"):
            admit(world)
        assert list(unused.iterdir()) == [], "a file was written into another route's origin"
    finally:
        os.rmdir(link)        # removes the link only


# ---------------------------------------------------------------------------
# astra confirming pass: a TRAINED route's directory is checked against every pinned origin too
# ---------------------------------------------------------------------------

def _junctioned_into(link, target):
    """Make ``link`` a junction to ``target`` (a directory inside a pinned origin flight)."""
    link.parent.mkdir(parents=True, exist_ok=True)
    _junction(link, target)


def test_a_retrained_routes_bakeoff_directory_junctioned_into_an_origin_is_refused(world, monkeypatch):
    import os
    calls = []
    stub_energy(world, monkeypatch, calls)
    monkeypatch.setattr(world.services, "admit_route",
                        lambda route, pin: {"route": route, "admitted": True})
    unused = world.origin.run_root / "bakeoff" / "unused_dwjs"
    unused.mkdir(parents=True)
    link = world.context.path("bakeoff", "dwjs_energy")
    _junctioned_into(link, unused)
    try:
        # Refused either by the run-root scan (a junction inside the run root) or by the resolved
        # destination check; which fires first is not the property. Nothing may train or be written.
        with pytest.raises(ValueError, match="overlaps|junction or symbolic link"):
            world.services.run_bakeoff()
        assert calls == [], "the dWJS retraining reached a destination inside a pinned origin"
        assert list(unused.iterdir()) == []
    finally:
        os.rmdir(link)        # removes the link only


def test_the_density_only_child_junctioned_into_an_origin_is_refused(world):
    import os
    unused = world.origin.run_root / "bakeoff" / "unused_density_only"
    unused.mkdir(parents=True)
    parent = world.context.path("bakeoff", "dwjs_energy")
    link = parent / "density_only"
    _junctioned_into(link, unused)
    try:
        with pytest.raises(ValueError, match="overlaps"):
            world.services.train_energy(directory=link, seed=1, population="x", conditioned=False)
        assert list(unused.iterdir()) == [], "train_energy wrote into a pinned origin"
    finally:
        os.rmdir(link)


def test_train_into_a_directory_junctioned_into_an_origin_is_refused(world):
    import os
    unused = world.origin.run_root / "bakeoff" / "unused_train"
    unused.mkdir(parents=True)
    link = world.context.path("bakeoff", "some_other_route")
    _junctioned_into(link, unused)
    try:
        with pytest.raises(ValueError, match="overlaps"):
            world.services.train(ROUTE, stream=reconstruction_stream(world.services), updates=1,
                                 seed=5, directory=link)
        assert list(unused.iterdir()) == [], "train wrote into a pinned origin"
    finally:
        os.rmdir(link)


def test_an_ordinary_trained_route_directory_is_still_accepted(world, monkeypatch):
    calls = []
    stub_energy(world, monkeypatch, calls)
    monkeypatch.setattr(world.services, "admit_route",
                        lambda route, pin: {"route": route, "admitted": True})
    world.services.run_bakeoff()
    assert calls == [("energy", "dwjs_energy", True), ("energy", "density_only", False)]


# ---------------------------------------------------------------------------
# sol round 8: the writable SOURCE CACHE lives outside the run root and must be checked too
# ---------------------------------------------------------------------------

def test_a_source_cache_table_directory_junctioned_into_an_origin_is_refused(world):
    import os
    cache = world.services.source_cache_root()          # a clean (absent) cache is fine
    unused = world.origin.run_root / "bakeoff" / "unused_cache"
    unused.mkdir(parents=True)
    cache.mkdir(parents=True)
    link = cache / "paired_oas"
    _junction(link, unused)
    try:
        with pytest.raises(ValueError, match="junction or symbolic link"):
            world.services.source_cache_root()
        assert list(unused.iterdir()) == [], "the cache writer would have created files in an origin"
    finally:
        os.rmdir(link)


def test_a_hard_linked_cache_table_is_refused_and_the_origin_file_is_untouched(world):
    import os
    origin_file = world.origin.run_root / "bakeoff" / "rows_stand_in.tsv"
    origin_file.parent.mkdir(parents=True, exist_ok=True)
    origin_file.write_bytes(b"ORIGIN-BYTES" * 10)
    table = world.services.source_cache_root() / "paired_oas" / "rows.tsv"
    table.parent.mkdir(parents=True)
    os.link(origin_file, table)
    with pytest.raises(ValueError, match="hard link"):
        world.services.source_cache_root()
    assert origin_file.read_bytes() == b"ORIGIN-BYTES" * 10


def test_a_cache_root_that_itself_resolves_into_an_origin_is_refused(world):
    import os
    unused = world.origin.run_root / "bakeoff" / "unused_cache_root"
    unused.mkdir(parents=True)
    cache = world.services.source_cache_root()
    cache.parent.mkdir(parents=True)
    _junction(cache, unused)
    try:
        with pytest.raises(ValueError, match="overlaps|junction or symbolic link"):
            world.services.source_cache_root()
    finally:
        os.rmdir(cache)


def test_a_clean_cache_and_an_absent_one_are_returned_unchanged(world):
    cache = world.services.source_cache_root()
    assert cache == world.context.artifact("cache/sources")
    (cache / "paired_oas").mkdir(parents=True)
    (cache / "paired_oas" / "rows.tsv").write_text("x\n", encoding="utf-8")
    assert world.services.source_cache_root() == cache


def test_an_explicit_scratch_cache_is_checked_like_the_default(world, tmp_path):
    import os
    scratch = tmp_path / "scratch_cache"
    scratch.mkdir()
    victim = tmp_path / "victim.bin"
    victim.write_bytes(b"V")
    os.link(victim, scratch / "linked.tsv")
    world.services._source_cache = scratch
    with pytest.raises(ValueError, match="hard link"):
        world.services.source_cache_root()


def test_a_flight_without_the_phase_does_not_check_its_cache(tmp_path, monkeypatch):
    import os
    services, context = build_services(tmp_path / "legacy_cache", make_config(), monkeypatch)
    cache = services.source_cache_root()
    cache.mkdir(parents=True)
    victim = tmp_path / "victim.bin"
    victim.write_bytes(b"V")
    os.link(victim, cache / "linked.tsv")
    assert services.source_cache_root() == cache


def test_the_cache_path_is_spelled_in_exactly_one_place():
    """Every reader follows source_cache_root(); a second literal is how one reader would bypass its check."""
    source = Path(services_mod.__file__).read_text(encoding="utf-8")
    assert source.count('"cache/sources"') == 1
