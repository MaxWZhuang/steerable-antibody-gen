"""The code-held ADMISSION phase: a closed declaration, held stages, locked gate numbers.

The phase exists so the held ``route_gate`` stage can run on the routes that finished in other
flights, and for nothing else. What is asserted:

* the config's ``admission_phase`` block is a closed schema compared with constants in code, so
  neither the route set, the stage list, the held stages, the split pin nor the OWNER'S GATE
  NUMBERS can be relaxed by editing a config;
* a four-route config with no ``competition`` block does not escape through the five-way hold:
  every stage after ``route_gate`` is refused centrally, ``verify`` and ``report`` included;
* the phase can be combined with neither the training phase nor a competition config, and never
  reports ``completed`` or a complete flight;
* the pins in the config are validated structurally, and dWJS (which retrains under energy
  contract v2) can never be admitted.

Not asserted here: loading an admitted checkpoint (see ``test_her2_routes_admission_services``).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from smallAntibodyGen.experiments import her2_routes_campaign as campaign
from smallAntibodyGen.experiments import her2_routes_spec as spec
from smallAntibodyGen.experiments import her2_support_paths as paths
from smallAntibodyGen.tests.test_her2_routes_admission import good_pin

ROOT = Path(__file__).resolve().parents[3]
CONFIGS = ROOT / "configs" / "experiments"
JOINT = json.loads((CONFIGS / "her2_generative_routes_joint.json").read_text(encoding="utf-8"))
ADMITTED = spec.ADMISSION_ADMITTED_ROUTES


def admission_config():
    """A valid admission-phase config, built from the joint config the way the real one is."""
    config = copy.deepcopy(JOINT)
    config.pop("competition", None)
    config["routes"]["enabled"] = list(spec.ADMISSION_PHASE_ROUTES)
    config["routes"].pop("dplm_650m_lora_masked", None)
    for key in ("posttrain", "confirmation", "controls", "priors"):
        config.pop(key, None)
    config["revisions"] = dict(config.get("revisions") or {}, kind="development",
                               supersedes=None, current="admitted_gate_rev01")
    config[spec.ADMISSION_PHASE_KEY] = spec.admission_phase_declaration()
    config[spec.ADMISSION_PINS_KEY] = {route: good_pin() for route in ADMITTED}
    return config


class SimpleContext:
    def __init__(self, root, config):
        self.run_root = Path(root) / "run"
        self.artifact_root = Path(root)
        self.repository_root = Path(root)
        self.config = config
        self.config_path = Path(root) / "config.json"
        self.revision = "rev01"
        self.campaign_id = "probe"
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(json.dumps(config), encoding="utf-8")

    def path(self, *parts):
        return self.run_root.joinpath(*[str(part) for part in parts])


# ---------------------------------------------------------------------------
# the declaration
# ---------------------------------------------------------------------------

def test_the_declaration_fixes_routes_stages_holds_and_the_owners_gate_numbers():
    declared = spec.admission_phase_declaration()
    assert declared["phase_id"] == "admitted_route_gate"
    assert declared["kind"] == "admission" and declared["admission"] == "route_gate_only"
    assert declared["routes"] == ["ablang2_masked_editor", "piggen_causal_infiller",
                                  "dwjs_energy", "dplm_150m_masked"]
    assert declared["admitted_routes"] == ["ablang2_masked_editor", "piggen_causal_infiller",
                                           "dplm_150m_masked"]
    assert declared["trained_routes"] == ["dwjs_energy"]
    assert declared["rule"] == "best"
    assert declared["stages"][-1] == "route_gate"
    assert declared["stages"] == ["acquire", "preflight", "split", "sources", "profile", "smoke",
                                  "freeze", "bakeoff", "route_gate"]
    assert set(declared["held_stages"]) == {"controls", "priors", "posttrain", "decoding",
                                            "confirmation_freeze", "confirm", "report", "verify"}
    assert set(declared["stages"]).isdisjoint(declared["held_stages"])
    assert set(declared["stages"]) | set(declared["held_stages"]) == set(campaign.STAGES)
    assert declared["gates"] == {"delta": 0.05, "readout_tolerance": 0.25,
                                 "generation_draws": 512, "retention_rows": 128,
                                 "local_pairs": 256,
                                 "margins": {"backbone": 0.02, "distance": 0.02, "generation": 0.0}}
    assert declared["expected_split_sha256"] == spec.TRAINING_PHASE_SPLIT_SHA256
    assert declared["expected_sequence_sha256"] == spec.TRAINING_PHASE_SEQUENCE_SHA256


def test_dwjs_is_trained_and_never_admitted():
    assert "dwjs_energy" not in spec.ADMISSION_ADMITTED_ROUTES
    assert set(spec.ADMISSION_ADMITTED_ROUTES) | set(spec.ADMISSION_TRAINED_ROUTES) == set(
        spec.ADMISSION_PHASE_ROUTES)


def test_a_valid_config_resolves_to_a_policy_with_a_digest():
    policy = spec.admission_phase(admission_config())
    assert policy["kind"] == "admission" and policy["admission"] == "route_gate_only"
    assert policy["policy_sha256"] == paths.digest_document(spec.admission_phase_declaration())
    assert len(policy["pins_sha256"]) == 64


def test_a_config_with_no_phase_key_is_not_an_admission_config():
    assert spec.admission_phase(copy.deepcopy(JOINT)) is None


# ---------------------------------------------------------------------------
# a closed schema
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", sorted(spec.admission_phase_declaration()))
def test_every_declared_value_is_pinned_in_code(key):
    config = admission_config()
    block = config[spec.ADMISSION_PHASE_KEY]
    block[key] = "tampered" if not isinstance(block[key], list) else block[key] + ["x"]
    with pytest.raises(ValueError, match="cannot be relaxed|closed schema"):
        spec.admission_phase(config)


def test_a_missing_or_extra_key_is_refused():
    config = admission_config()
    config[spec.ADMISSION_PHASE_KEY].pop("held_stages")
    with pytest.raises(ValueError, match="closed schema"):
        spec.admission_phase(config)
    config = admission_config()
    config[spec.ADMISSION_PHASE_KEY]["extra"] = 1
    with pytest.raises(ValueError, match="closed schema"):
        spec.admission_phase(config)


@pytest.mark.parametrize("bad", [None, {}, [], "x"])
def test_a_malformed_phase_block_is_refused_not_read_as_no_phase(bad):
    config = admission_config()
    config[spec.ADMISSION_PHASE_KEY] = bad
    with pytest.raises(ValueError, match="nonempty mapping"):
        spec.admission_phase(config)


def test_shrinking_the_held_stages_is_refused():
    config = admission_config()
    config[spec.ADMISSION_PHASE_KEY]["held_stages"] = ["confirm", "verify"]
    with pytest.raises(ValueError, match="cannot be relaxed"):
        spec.admission_phase(config)


# ---------------------------------------------------------------------------
# the owner's gate numbers are locked
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("edit", [
    lambda g: g.__setitem__("delta", 0.0),
    lambda g: g["margins"].__setitem__("generation", 0.1),
    lambda g: g["margins"].__setitem__("backbone", 0.0),
    lambda g: g["margins"].__setitem__("distance", 0.5),
    lambda g: g.__setitem__("generation_draws", 8),
    lambda g: g.__setitem__("retention_rows", 1),
    lambda g: g.__setitem__("local_pairs", 1),
    lambda g: g.__setitem__("readout_tolerance", 0.9),
    lambda g: g["margins"].__setitem__("generation", 0),  # an int is not the declared 0.0
])
def test_relaxed_gate_numbers_are_refused_even_with_the_declaration_untouched(edit):
    config = admission_config()
    edit(config["gates"])
    with pytest.raises(ValueError, match="gates"):
        spec.admission_phase(config)


def test_prose_in_the_gates_block_may_change():
    config = admission_config()
    config["gates"]["generation_endpoint"] = "reworded prose is not a threshold"
    assert spec.admission_phase(config) is not None


def test_a_config_with_no_gates_is_refused():
    config = admission_config()
    config.pop("gates")
    with pytest.raises(ValueError, match="gates"):
        spec.admission_phase(config)


# ---------------------------------------------------------------------------
# routes, revision, exclusions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("enabled", [
    ["ablang2_masked_editor", "piggen_causal_infiller", "dwjs_energy"],
    ["piggen_causal_infiller", "ablang2_masked_editor", "dwjs_energy", "dplm_150m_masked"],
    ["ablang2_masked_editor", "piggen_causal_infiller", "dwjs_energy", "dplm_150m_masked",
     "dplm_650m_lora_masked"],
])
def test_the_enabled_routes_must_be_exactly_the_four_in_order(enabled):
    config = admission_config()
    config["routes"]["enabled"] = enabled
    with pytest.raises(ValueError, match="routes.enabled"):
        spec.admission_phase(config)


def test_a_competition_block_is_refused():
    config = admission_config()
    config["competition"] = {"required_routes": list(spec.ADMISSION_PHASE_ROUTES)}
    with pytest.raises(ValueError, match="competition"):
        spec.admission_phase(config)


def test_the_two_phases_cannot_be_combined():
    config = admission_config()
    config[spec.TRAINING_PHASE_KEY] = spec.training_phase_declaration()
    with pytest.raises(ValueError, match="both"):
        spec.admission_phase(config)
    with pytest.raises(ValueError, match="both"):
        spec.training_phase(config)
    with pytest.raises(ValueError, match="both"):
        spec.active_phase(config)


def test_a_non_development_revision_is_refused():
    config = admission_config()
    config["revisions"]["kind"] = "production"
    with pytest.raises(ValueError, match="development"):
        spec.admission_phase(config)


def test_active_phase_dispatches_and_leaves_the_legacy_config_alone():
    assert spec.active_phase(copy.deepcopy(JOINT)) is None
    assert spec.active_phase(admission_config())["kind"] == "admission"
    train = json.loads((CONFIGS / "her2_generative_routes_dplm_train.json").read_text("utf-8"))
    assert spec.active_phase(train)["phase_id"] == "dplm_explicit_training"
    assert "kind" not in spec.active_phase(train)


# ---------------------------------------------------------------------------
# the pins block
# ---------------------------------------------------------------------------

def test_the_pins_are_required_and_validated():
    config = admission_config()
    config.pop(spec.ADMISSION_PINS_KEY)
    with pytest.raises(ValueError, match="pins"):
        spec.admission_phase(config)
    config = admission_config()
    config[spec.ADMISSION_PINS_KEY]["ablang2_masked_editor"]["checkpoint_sha256"] = "nope"
    with pytest.raises(ValueError, match="64 hex"):
        spec.admission_phase(config)


def test_a_pin_for_dwjs_is_refused_because_it_is_retrained():
    config = admission_config()
    config[spec.ADMISSION_PINS_KEY]["dwjs_energy"] = good_pin()
    with pytest.raises(ValueError, match="route set"):
        spec.admission_phase(config)


def test_the_pins_digest_moves_with_any_pin():
    first = spec.admission_phase(admission_config())["pins_sha256"]
    config = admission_config()
    config[spec.ADMISSION_PINS_KEY]["piggen_causal_infiller"]["checkpoint_step"] += 1
    assert spec.admission_phase(config)["pins_sha256"] != first


def test_admission_pins_returns_the_validated_mapping():
    pins = spec.admission_pins(admission_config())
    assert set(pins) == set(ADMITTED)


# ---------------------------------------------------------------------------
# resolution and the central stage hold
# ---------------------------------------------------------------------------

def test_the_resolved_protocol_records_the_phase_and_needs_gates_and_decoding(tmp_path):
    context = SimpleContext(tmp_path, admission_config())
    document = spec.resolved_protocol(
        context, snapshot={"snapshot_sha256": "s"}, external={"external_sha256": "e"},
        split={"a": 1}, profile={"b": 1}, forecast={"c": 1},
        revision={"revision": "r", "kind": "development"})
    assert document["unresolved"] == []
    assert document[spec.ADMISSION_PHASE_KEY]["phase_id"] == "admitted_route_gate"
    assert document[spec.ADMISSION_PINS_KEY].keys() == set(ADMITTED)
    broken = admission_config()
    broken.pop("decoding")
    document = spec.resolved_protocol(
        SimpleContext(tmp_path / "b", broken), snapshot={"snapshot_sha256": "s"},
        external={"external_sha256": "e"}, split={"a": 1}, profile={"b": 1}, forecast={"c": 1},
        revision={"revision": "r", "kind": "development"})
    assert "config.decoding" in document["unresolved"]


def test_an_invalid_phase_is_unresolved_not_silently_ignored(tmp_path):
    config = admission_config()
    config[spec.ADMISSION_PHASE_KEY]["rule"] = "last"
    document = spec.resolved_protocol(
        SimpleContext(tmp_path, config), snapshot={"snapshot_sha256": "s"},
        external={"external_sha256": "e"}, split={"a": 1}, profile={"b": 1}, forecast={"c": 1},
        revision={"revision": "r", "kind": "development"})
    assert "admission_phase.invalid" in document["unresolved"]


def test_every_held_stage_is_refused_and_every_phase_stage_is_allowed(tmp_path):
    context = SimpleContext(tmp_path, admission_config())
    for stage in spec.ADMISSION_PHASE_HELD_STAGES:
        with pytest.raises(ValueError, match="held"):
            campaign.require_stage_allowed(context, stage)
    for stage in spec.ADMISSION_PHASE_STAGES:
        assert campaign.require_stage_allowed(context, stage)["kind"] == "admission"


def test_the_refusal_names_the_admission_phase_not_the_training_phase(tmp_path):
    context = SimpleContext(tmp_path, admission_config())
    with pytest.raises(ValueError) as caught:
        campaign.require_stage_allowed(context, "controls")
    message = str(caught.value)
    assert "admitted_route_gate" in message and "admission" in message
    assert "training-only" not in message


def test_a_direct_held_stage_call_refuses_before_reuse_and_any_effect(tmp_path, monkeypatch):
    context = SimpleContext(tmp_path, admission_config())
    called = []
    monkeypatch.setattr(campaign, "STAGE_FUNCTIONS",
                        {name: (lambda *a, _n=name, **k: called.append(_n) or {"status": "completed"})
                         for name in campaign.STAGES})
    campaign.write_stage_record(context, "controls", {"status": "completed", "identity": {}})
    for stage in ("controls", "verify", "report", "confirm"):
        with pytest.raises(ValueError, match="held"):
            campaign.run_stage(context, stage, services=object())
    assert called == []


def test_run_all_defaults_to_the_phase_stages_and_never_claims_a_complete_flight(tmp_path,
                                                                                monkeypatch):
    context = SimpleContext(tmp_path, admission_config())
    launched = []
    monkeypatch.setattr(campaign, "STAGE_FUNCTIONS",
                        {name: (lambda c, s, _n=name, **k: launched.append(_n) or {"status": "completed"})
                         for name in campaign.STAGES})
    monkeypatch.setattr(campaign, "stage_identity", lambda *a, **k: {})
    document = campaign.run_all(context, services=object())
    assert launched == list(spec.ADMISSION_PHASE_STAGES)
    assert document["scope"] == "admission_phase"
    assert document["admission_phase_completed"] is True
    assert document["complete_flight"] is False
    assert document["admission_phase"]["held_stages"] == list(spec.ADMISSION_PHASE_HELD_STAGES)
    assert "training_phase" not in document


def test_run_all_with_an_explicit_held_stage_refuses_the_whole_list_first(tmp_path, monkeypatch):
    context = SimpleContext(tmp_path, admission_config())
    launched = []
    monkeypatch.setattr(campaign, "STAGE_FUNCTIONS",
                        {name: (lambda c, s, _n=name, **k: launched.append(_n) or {"status": "completed"})
                         for name in campaign.STAGES})
    with pytest.raises(ValueError, match="held"):
        campaign.run_all(context, services=object(), stages=("acquire", "controls"))
    assert launched == []


def test_the_status_word_is_not_completed(tmp_path, monkeypatch):
    assert campaign.STATUS_ADMISSION_PHASE_COMPLETED == "admission_phase_completed"
    assert campaign.STATUS_ADMISSION_PHASE_COMPLETED != campaign.STATUS_COMPLETED
    assert campaign.STATUS_ADMISSION_PHASE_COMPLETED in campaign.SUCCESS_STATUSES


def test_select_stages_honours_the_phase_bounds(tmp_path):
    context = SimpleContext(tmp_path, admission_config())
    assert campaign.select_stages(context) == spec.ADMISSION_PHASE_STAGES
    assert campaign.select_stages(context, through_stage="bakeoff")[-1] == "bakeoff"
    with pytest.raises(ValueError):
        campaign.select_stages(context, through_stage="controls")
    with pytest.raises(ValueError):
        campaign.select_stages(context, from_stage="verify")


# ---------------------------------------------------------------------------
# the real config file
# ---------------------------------------------------------------------------

def test_the_legacy_configs_are_untouched_by_the_new_phase():
    for name in ("her2_generative_routes.json", "her2_generative_routes_dplm.json",
                 "her2_generative_routes_joint.json", "her2_generative_routes_t4v2.json",
                 "her2_generative_routes_dplm_train.json"):
        config = json.loads((CONFIGS / name).read_text(encoding="utf-8"))
        assert spec.ADMISSION_PHASE_KEY not in config and spec.ADMISSION_PINS_KEY not in config
        assert spec.admission_phase(config) is None


# ---------------------------------------------------------------------------
# stage identity: reuse must notice a different checkpoint
# ---------------------------------------------------------------------------

def _write_checkpoints(context, *, which=None):
    for route in which or spec.ADMISSION_PHASE_ROUTES:
        target = context.path("bakeoff", route, "best.pt")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(f"checkpoint for {route}".encode())


def _freeze_marker(context, digest="a" * 64):
    paths.write_json(context.path(spec.FREEZE_MARKER), {"snapshot_sha256": digest})


def test_downstream_identity_binds_the_whole_selected_checkpoint_manifest(tmp_path):
    context = SimpleContext(tmp_path, admission_config())
    _write_checkpoints(context)
    _freeze_marker(context)
    before = campaign.stage_identity(context, "route_gate")
    assert len(before["admitted_manifest_sha256"]) == 64
    assert before["snapshot_sha256"] == "a" * 64, "the real freeze marker, not a dead filename"
    context.path("bakeoff", "dplm_150m_masked", "best.pt").write_bytes(b"a different checkpoint")
    after = campaign.stage_identity(context, "route_gate")
    assert after["admitted_manifest_sha256"] != before["admitted_manifest_sha256"]


def test_the_manifest_covers_every_route_including_the_trained_one(tmp_path):
    context = SimpleContext(tmp_path, admission_config())
    _write_checkpoints(context)
    _freeze_marker(context)
    base = campaign.stage_identity(context, "route_gate")["admitted_manifest_sha256"]
    for route in spec.ADMISSION_PHASE_ROUTES:
        target = context.path("bakeoff", route, "best.pt")
        original = target.read_bytes()
        target.write_bytes(original + b"!")
        assert campaign.stage_identity(context, "route_gate")["admitted_manifest_sha256"] != base, route
        target.write_bytes(original)
    assert campaign.stage_identity(context, "route_gate")["admitted_manifest_sha256"] == base


def test_a_missing_checkpoint_changes_the_manifest_instead_of_being_skipped(tmp_path):
    context = SimpleContext(tmp_path, admission_config())
    _write_checkpoints(context)
    full = campaign.stage_identity(context, "route_gate")["admitted_manifest_sha256"]
    context.path("bakeoff", "dwjs_energy", "best.pt").unlink()
    assert campaign.stage_identity(context, "route_gate")["admitted_manifest_sha256"] != full


def test_the_manifest_moves_with_the_committed_pins(tmp_path):
    config = admission_config()
    one = SimpleContext(tmp_path / "one", config)
    _write_checkpoints(one)
    other_config = admission_config()
    other_config[spec.ADMISSION_PINS_KEY]["piggen_causal_infiller"]["checkpoint_step"] += 1
    two = SimpleContext(tmp_path / "two", other_config)
    _write_checkpoints(two)
    assert (campaign.stage_identity(one, "route_gate")["admitted_manifest_sha256"]
            != campaign.stage_identity(two, "route_gate")["admitted_manifest_sha256"])


def test_a_stale_route_gate_record_is_not_reusable_after_a_checkpoint_changes(tmp_path):
    context = SimpleContext(tmp_path, admission_config())
    _write_checkpoints(context)
    _freeze_marker(context)
    paths.write_json(context.path("external_sources.json"), {"external_sha256": "e" * 64})
    campaign.write_stage_record(context, "bakeoff", {
        "status": "completed", "completed": [], "remaining": [],
        "identity": campaign.stage_identity(context, "bakeoff")})
    campaign.write_stage_record(context, "route_gate", {
        "status": "completed", "identity": campaign.stage_identity(context, "route_gate")})
    assert campaign.stage_is_reusable(context, "route_gate")[0] is True
    context.path("bakeoff", "ablang2_masked_editor", "best.pt").write_bytes(b"swapped")
    reusable, reason = campaign.stage_is_reusable(context, "route_gate")
    assert reusable is False and "admitted_manifest_sha256" in reason


def test_pre_freeze_stage_identities_keep_the_legacy_shape_so_resume_does_not_rerun_them(tmp_path):
    context = SimpleContext(tmp_path, admission_config())
    _write_checkpoints(context)
    _freeze_marker(context)
    early = campaign.stage_identity(context, "profile")
    assert "admitted_manifest_sha256" not in early and early["snapshot_sha256"] is None


def test_legacy_and_training_phase_identities_are_unchanged(tmp_path):
    legacy = SimpleContext(tmp_path / "legacy", json.loads(
        (CONFIGS / "her2_generative_routes.json").read_text("utf-8")))
    paths.write_json(legacy.path(spec.FREEZE_MARKER), {"snapshot_sha256": "a" * 64})
    identity = campaign.stage_identity(legacy, "route_gate")
    assert set(identity) == {"stage", "revision", "config_sha256", "external_sha256",
                             "snapshot_sha256", "bakeoff_outcome_sha256"}
    assert identity["snapshot_sha256"] is None, "the legacy lookup is preserved on purpose"
    training = SimpleContext(tmp_path / "train", json.loads(
        (CONFIGS / "her2_generative_routes_dplm_train.json").read_text("utf-8")))
    assert "admitted_manifest_sha256" not in campaign.stage_identity(training, "bakeoff")


# ---------------------------------------------------------------------------
# the bakeoff record says what was imported and what was trained
# ---------------------------------------------------------------------------

def test_the_bakeoff_record_names_the_admitted_and_the_trained_routes(tmp_path, monkeypatch):
    context = SimpleContext(tmp_path, admission_config())

    class Services:
        def run_bakeoff(self, heartbeat=None):
            routes = {route: {"admitted": True, "trained": False} for route in ADMITTED}
            routes["dwjs_energy"] = {"updates": 5, "trained": True}
            return {"routes": routes, "completed": sorted(routes), "remaining": [],
                    "all_terminal": True}

    monkeypatch.setattr(campaign.spec, "verify_source_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(campaign.spec, "verify_external_sources", lambda *a, **k: None)
    monkeypatch.setattr(campaign.spec, "require_resolved", lambda *a, **k: None)
    monkeypatch.setattr(campaign.spec, "require_revision_discipline", lambda *a, **k: None)
    monkeypatch.setattr(campaign, "_require_phase_split", lambda *a, **k: None)
    monkeypatch.setattr(campaign, "_require_admission_ready", lambda *a, **k: None)
    record = campaign.stage_bakeoff(context, Services())
    block = record["admission_phase"]
    assert block["admitted"] == sorted(ADMITTED) and block["trained"] == ["dwjs_energy"]
    assert block["phase_id"] == "admitted_route_gate" and len(block["pins_sha256"]) == 64
    assert "training_phase" not in record


# ---------------------------------------------------------------------------
# a direct verify() call is held too
# ---------------------------------------------------------------------------

def test_a_direct_verify_call_is_refused_under_the_admission_phase(tmp_path):
    context = SimpleContext(tmp_path, admission_config())
    with pytest.raises(ValueError, match="held"):
        campaign.verify(context)


# ---------------------------------------------------------------------------
# a synthetic world: three origin flights, a destination root, valid pins
# ---------------------------------------------------------------------------

from smallAntibodyGen.experiments import her2_routes_admission as admission  # noqa: E402

CONFIG_LOGICAL = "configs/experiments/origin.json"
EXTERNAL = {"data/a.py": paths.sha256_text("a"), "data/b.py": paths.sha256_text("b")}
VOCABULARY = '{"bins": ["high", "null"]}\n'
SPLIT = {"sequence_sha256": "s" * 64, "n": 100}


def build_world(tmp_path, *, steps=None):
    """``(context, config)`` whose pins verify against three synthetic finished origin flights."""
    config = admission_config()
    pins = {}
    for index, route in enumerate(ADMITTED):
        root = tmp_path / f"origin_{route}"
        (root / "source_snapshot" / "configs" / "experiments").mkdir(parents=True)
        origin = {"schema_version": spec.ROUTES_SCHEMA,
                  "routes": {"enabled": [route], route: copy.deepcopy(config["routes"][route])},
                  "her2": copy.deepcopy(config["her2"])}
        if isinstance(config.get("dplm"), dict):
            origin["dplm"] = copy.deepcopy(config["dplm"])
        (root / "source_snapshot" / CONFIG_LOGICAL).write_text(
            json.dumps(origin, sort_keys=True), encoding="utf-8")
        paths.write_json(root / "source_snapshot.json", {"snapshot_sha256": paths.sha256_text(f"s{index}")})
        paths.write_json(root / "external_sources.json",
                         {"external_sha256": admission.external_aggregate(EXTERNAL),
                          "files": dict(EXTERNAL)})
        (root / "condition_vocabulary.json").write_text(VOCABULARY, encoding="utf-8")
        paths.write_json(root / "hard_split_freeze.json", dict(SPLIT))
        checkpoint = root / "bakeoff" / route / "best.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(f"origin checkpoint of {route}".encode() * 50)
        pins[route] = admission.derive_pin(
            root, route=route, origin_config=CONFIG_LOGICAL, checkpoint=checkpoint,
            checkpoint_step=(steps or {}).get(route, 100 + index), arm="full",
            seed=int(config["routes"][route]["seed"]), schedule_sha256=None,
            selector="development_affinity_spearman_high", live_config=config)
    config[spec.ADMISSION_PINS_KEY] = pins
    context = SimpleContext(tmp_path, config)
    paths.write_json(context.path(spec.EXTERNAL_MARKER),
                     {"external_sha256": "x" * 64,
                      "files": dict(EXTERNAL, **{"models/dplm_150m/config.json": "1" * 64})})
    context.path("condition_vocabulary.json").write_text(VOCABULARY, encoding="utf-8")
    paths.write_json(context.path(spec.SPLIT_FREEZE), dict(SPLIT))
    return context, config


def test_a_compatible_world_is_ready_and_reads_only(tmp_path):
    context, _config = build_world(tmp_path)
    policy = campaign._require_admission_ready(context)
    assert policy["kind"] == "admission"


def test_readiness_is_a_no_op_for_legacy_and_training_configs(tmp_path, monkeypatch):
    monkeypatch.setattr(admission, "origin_problems",
                        lambda *a, **k: pytest.fail("origin_problems must not run"))
    for name in ("her2_generative_routes.json", "her2_generative_routes_dplm_train.json"):
        config = json.loads((CONFIGS / name).read_text(encoding="utf-8"))
        context = SimpleContext(tmp_path / name, config)
        assert campaign._require_admission_ready(context) is None


def test_a_different_vocabulary_stops_everything_before_any_gpu_time(tmp_path):
    context, _config = build_world(tmp_path)
    context.path("condition_vocabulary.json").write_text('{"bins": ["other"]}\n', encoding="utf-8")
    with pytest.raises(campaign.ReadinessGate, match="vocabulary") as caught:
        campaign._require_admission_ready(context)
    assert "no GPU time was spent" in str(caught.value)
    for route in ADMITTED:
        assert route in str(caught.value)


def test_a_different_split_is_refused(tmp_path):
    context, _config = build_world(tmp_path)
    paths.write_json(context.path(spec.SPLIT_FREEZE), {"sequence_sha256": "z" * 64, "n": 100})
    with pytest.raises(campaign.ReadinessGate, match="split"):
        campaign._require_admission_ready(context)


def test_a_missing_destination_marker_is_a_readiness_gate_not_a_crash(tmp_path):
    context, _config = build_world(tmp_path)
    context.path("condition_vocabulary.json").unlink()
    with pytest.raises(campaign.ReadinessGate, match="absent"):
        campaign._require_admission_ready(context)


def test_a_swapped_origin_checkpoint_is_refused(tmp_path):
    context, config = build_world(tmp_path)
    pin = config[spec.ADMISSION_PINS_KEY]["piggen_causal_infiller"]
    admission.origin_checkpoint_path(pin, "piggen_causal_infiller").write_bytes(b"swapped")
    with pytest.raises(campaign.ReadinessGate, match="piggen_causal_infiller"):
        campaign._require_admission_ready(context)


def test_an_external_table_missing_an_origin_file_is_refused(tmp_path):
    context, _config = build_world(tmp_path)
    paths.write_json(context.path(spec.EXTERNAL_MARKER),
                     {"external_sha256": "x" * 64, "files": {"data/a.py": EXTERNAL["data/a.py"]}})
    with pytest.raises(campaign.ReadinessGate, match="external table"):
        campaign._require_admission_ready(context)


def test_the_guarded_wrapper_checks_readiness_before_running_the_stage_body(tmp_path, monkeypatch):
    context, _config = build_world(tmp_path)
    order = []
    for name in ("verify_source_snapshot", "verify_external_sources", "require_resolved",
                 "require_revision_discipline"):
        monkeypatch.setattr(campaign.spec, name, lambda *a, _n=name, **k: order.append(_n))
    monkeypatch.setattr(campaign, "_require_phase_split", lambda *a, **k: order.append("split"))
    original = campaign._require_admission_ready
    monkeypatch.setattr(campaign, "_require_admission_ready",
                        lambda c: order.append("ready") or original(c))
    assert campaign._guarded(context, "bakeoff", lambda: order.append("body") or "done") == "done"
    assert order[-3:] == ["split", "ready", "body"]
    context.path("condition_vocabulary.json").write_text("changed", encoding="utf-8")
    with pytest.raises(campaign.ReadinessGate):
        campaign._guarded(context, "bakeoff", lambda: pytest.fail("the body must not run"))


def test_profile_smoke_and_freeze_all_check_readiness_first(tmp_path, monkeypatch):
    context, _config = build_world(tmp_path)
    context.path("condition_vocabulary.json").write_text("changed", encoding="utf-8")
    monkeypatch.setattr(campaign, "_require_phase_split", lambda *a, **k: None)
    boom = lambda *a, **k: pytest.fail("no GPU stage work may start")  # noqa: E731
    monkeypatch.setattr(campaign, "profile_routes", boom)
    for stage in (campaign.stage_profile, campaign.stage_smoke, campaign.stage_freeze):
        with pytest.raises(campaign.ReadinessGate):
            stage(context, object())


# ---------------------------------------------------------------------------
# the committed config
# ---------------------------------------------------------------------------

REAL = CONFIGS / "her2_generative_routes_admitted_gate.json"
ORIGIN_LOGICAL = {"ablang2_masked_editor": "configs/experiments/her2_generative_routes.json",
                  "piggen_causal_infiller": "configs/experiments/her2_generative_routes.json",
                  "dplm_150m_masked": "configs/experiments/her2_generative_routes_dplm_train.json"}


def real_config():
    return spec.load_config(REAL)


def test_the_committed_config_is_a_valid_admission_config():
    config = real_config()
    policy = spec.admission_phase(config)
    assert policy["phase_id"] == "admitted_route_gate"
    assert config["routes"]["enabled"] == list(spec.ADMISSION_PHASE_ROUTES)
    assert config["config_path"] == "configs/experiments/her2_generative_routes_admitted_gate.json"
    assert config["revisions"]["kind"] == "development"
    assert config["revisions"]["current"] == "admitted_gate_rev02"
    assert config["revisions"]["supersedes"] == "admitted_gate_rev01"
    assert config["campaign_id"] != JOINT["campaign_id"] and config["run_root_leaf"] != JOINT["run_root_leaf"]


def test_the_committed_config_has_no_competition_block_and_no_held_only_blocks():
    config = real_config()
    for key in ("competition", "posttrain", "confirmation", "controls", "priors"):
        assert key not in config, key
    assert spec.competition_unresolved(config) == [], (
        "the five-way hold does not apply to this config; the admission phase is what holds the "
        "later stages")


def test_the_committed_config_keeps_the_owners_gate_numbers():
    gates = real_config()["gates"]
    assert gates["delta"] == 0.05
    assert gates["margins"] == {"backbone": 0.02, "distance": 0.02, "generation": 0.0}
    assert paths.canonical_json(gates["margins"]["generation"]).strip() == "0.0"


def test_the_committed_pins_describe_each_routes_own_best_checkpoint():
    pins = spec.admission_pins(real_config())
    assert set(pins) == set(ADMITTED)
    assert {route: pin["checkpoint_step"] for route, pin in pins.items()} == {
        "ablang2_masked_editor": 9500, "piggen_causal_infiller": 2000, "dplm_150m_masked": 15500}
    for route, pin in pins.items():
        assert pin["origin_config"] == ORIGIN_LOGICAL[route]
        assert pin["arm"] == "full" and pin["seed"] == 20260927 and pin["schedule_sha256"] is None
    assert pins["dplm_150m_masked"]["selector"].startswith("hr17_")
    assert pins["ablang2_masked_editor"]["selector"] == "development_affinity_spearman_high"
    assert set(pins["dplm_150m_masked"]["block_digests"]) == {"route", "her2", "dplm"}
    assert set(pins["ablang2_masked_editor"]["block_digests"]) == {"route", "her2"}


def test_the_committed_route_blocks_equal_the_origins_apart_from_prose():
    """The architecture an admitted checkpoint is scored under is the one it was trained under."""
    config = real_config()
    for name, origin in (("her2_generative_routes.json", ("ablang2_masked_editor",
                                                          "piggen_causal_infiller", "dwjs_energy")),
                         ("her2_generative_routes_dplm_train.json", ("dplm_150m_masked",))):
        base = json.loads((CONFIGS / name).read_text(encoding="utf-8"))
        for route in origin:
            assert (admission.block_digest(config["routes"][route])
                    == admission.block_digest(base["routes"][route])), route


FLIGHTS = Path("F:/smallAntibodyGen/her2_generative_routes_20260927/runs")


@pytest.mark.skipif(not FLIGHTS.is_dir(), reason="the finished origin flights are not on this machine")
def test_the_committed_pins_verify_against_the_finished_origin_flights():
    config = real_config()
    for route, pin in spec.admission_pins(config).items():
        checkpoint = admission.origin_checkpoint_path(pin, route)
        assert checkpoint.is_file() and checkpoint.stat().st_size == pin["checkpoint_bytes"], route
        frozen = Path(pin["origin_flight"]) / "source_snapshot" / pin["origin_config"]
        assert paths.sha256_file(frozen) == pin["origin_config_sha256"], route
        assert (paths.read_json(Path(pin["origin_flight"]) / "source_snapshot.json")["snapshot_sha256"]
                == pin["origin_snapshot_sha256"]), route
        assert (paths.read_json(Path(pin["origin_flight"]) / "external_sources.json")["external_sha256"]
                == pin["origin_external_sha256"]), route


# ---------------------------------------------------------------------------
# astra's review: the run root may not be, or sit inside, a pinned origin flight
# ---------------------------------------------------------------------------

def test_resolve_context_refuses_a_run_root_inside_a_pinned_origin(tmp_path):
    config = real_config()
    origin = Path(spec.admission_pins(config)["ablang2_masked_editor"]["origin_flight"])
    # Separate repository and artifact roots, so the OLD repository-overlap guard cannot be what
    # fires; the message must be the new origin guard's own.
    for run_root in (origin, origin / "nested", origin.parent):
        with pytest.raises(ValueError, match="pinned origin flight"):
            spec.resolve_context(tmp_path / "repo", config_path=REAL, run_root=run_root,
                                 artifact_root=tmp_path / "artifacts", ensure=False)


def test_resolve_context_accepts_the_default_disjoint_run_root(tmp_path):
    context = spec.resolve_context(tmp_path / "repo", config_path=REAL,
                                   artifact_root=tmp_path / "artifacts", ensure=False)
    assert context.run_root == (tmp_path / "artifacts" / "runs" / "admitted_gate_rev02"
                                / "flight_admitted_gate")


def test_the_readiness_check_also_refuses_an_overlapping_destination(tmp_path):
    context, config = build_world(tmp_path)
    pin = config[spec.ADMISSION_PINS_KEY]["ablang2_masked_editor"]
    config[spec.ADMISSION_PINS_KEY]["ablang2_masked_editor"] = dict(pin, origin_flight=str(context.run_root))
    with pytest.raises(campaign.ReadinessGate, match="pinned origin flight"):
        campaign._require_admission_ready(context)


# ---------------------------------------------------------------------------
# astra round 2: stage reuse must notice a missing or edited admission record
# ---------------------------------------------------------------------------

def _record_path(context, route):
    return context.path("bakeoff", route, "admission.json")


def test_downstream_identity_binds_the_admission_records_too(tmp_path):
    context = SimpleContext(tmp_path, admission_config())
    _write_checkpoints(context)
    for route in spec.ADMISSION_ADMITTED_ROUTES:
        _record_path(context, route).write_text(json.dumps({"route": route}), encoding="utf-8")
    base = campaign.stage_identity(context, "route_gate")["admitted_manifest_sha256"]
    record = _record_path(context, "dplm_150m_masked")
    record.write_text(json.dumps({"route": "dplm_150m_masked", "edited": True}), encoding="utf-8")
    assert campaign.stage_identity(context, "route_gate")["admitted_manifest_sha256"] != base
    record.unlink()
    assert campaign.stage_identity(context, "route_gate")["admitted_manifest_sha256"] != base


def test_the_trained_route_has_no_record_to_bind(tmp_path):
    context = SimpleContext(tmp_path, admission_config())
    _write_checkpoints(context)
    for route in spec.ADMISSION_ADMITTED_ROUTES:
        _record_path(context, route).write_text("{}", encoding="utf-8")
    base = campaign.stage_identity(context, "route_gate")["admitted_manifest_sha256"]
    _record_path(context, "dwjs_energy").write_text("{}", encoding="utf-8")   # not an admitted route
    assert campaign.stage_identity(context, "route_gate")["admitted_manifest_sha256"] == base


def test_a_completed_route_gate_is_not_reused_after_an_admission_record_is_deleted(tmp_path):
    context = SimpleContext(tmp_path, admission_config())
    _write_checkpoints(context)
    _freeze_marker(context)
    paths.write_json(context.path("external_sources.json"), {"external_sha256": "e" * 64})
    for route in spec.ADMISSION_ADMITTED_ROUTES:
        _record_path(context, route).write_text(json.dumps({"route": route}), encoding="utf-8")
    campaign.write_stage_record(context, "bakeoff", {
        "status": "completed", "completed": [], "remaining": [],
        "identity": campaign.stage_identity(context, "bakeoff")})
    campaign.write_stage_record(context, "route_gate", {
        "status": "completed", "identity": campaign.stage_identity(context, "route_gate")})
    assert campaign.stage_is_reusable(context, "route_gate")[0] is True
    # (The bakeoff stage's own identity includes its own outcome record, so it is never reusable on a
    # resume in any flight; re-running it is idempotent and is what repairs a deleted record.)
    before = campaign.stage_identity(context, "bakeoff")["admitted_manifest_sha256"]
    _record_path(context, "ablang2_masked_editor").unlink()
    reusable, reason = campaign.stage_is_reusable(context, "route_gate")
    assert reusable is False and "admitted_manifest_sha256" in reason, reason
    assert campaign.stage_identity(context, "bakeoff")["admitted_manifest_sha256"] != before


# ---------------------------------------------------------------------------
# sol round 6: a junctioned DIRECTORY inside the run root must not redirect any write into an origin
# ---------------------------------------------------------------------------

def _real_context(tmp_path, origins):
    """The REAL ``RoutesContext`` (not a test double) over an admission config pinned to ``origins``."""
    config = admission_config()
    for route, origin in origins.items():
        config[spec.ADMISSION_PINS_KEY][route] = dict(good_pin(), origin_flight=str(origin))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    run_root = tmp_path / "run"
    run_root.mkdir()
    return spec.RoutesContext(repository_root=tmp_path, run_root=run_root, artifact_root=tmp_path,
                              config=config, config_path=config_path), run_root


def _junction(link, target):
    import subprocess
    done = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                          capture_output=True, text=True)
    if done.returncode:
        pytest.skip(f"cannot create a directory junction here: {done.stdout} {done.stderr}")


def test_a_stage_record_directory_junctioned_into_an_origin_is_never_written(tmp_path):
    import os
    origin = tmp_path / "origin_flight"
    (origin / campaign.STAGE_RECORDS).mkdir(parents=True)
    context, run_root = _real_context(tmp_path, {"ablang2_masked_editor": origin})
    link = run_root / campaign.STAGE_RECORDS
    _junction(link, origin / campaign.STAGE_RECORDS)
    try:
        with pytest.raises(ValueError, match="overlaps"):
            campaign.write_stage_record(context, "profile", {"status": "blocked"})
        assert list((origin / campaign.STAGE_RECORDS).iterdir()) == [], (
            "a stage record was written into a pinned origin flight")
    finally:
        os.rmdir(link)        # removes the link only


def test_every_run_root_path_is_checked_against_every_pinned_origin(tmp_path):
    import os
    origin = tmp_path / "other_origin"
    (origin / "bakeoff").mkdir(parents=True)
    context, run_root = _real_context(tmp_path, {"piggen_causal_infiller": origin})
    link = run_root / "bakeoff"
    _junction(link, origin / "bakeoff")
    try:
        for parts in (("bakeoff",), ("bakeoff", "dwjs_energy"), ("bakeoff", "dwjs_energy", "density_only")):
            with pytest.raises(ValueError, match="overlaps"):
                context.path(*parts)
    finally:
        os.rmdir(link)


def test_ordinary_run_root_paths_are_still_returned_under_the_phase(tmp_path):
    origin = tmp_path / "origin_flight"
    origin.mkdir()
    context, run_root = _real_context(tmp_path, {"ablang2_masked_editor": origin})
    assert context.path("stages", "profile.json") == run_root / "stages" / "profile.json"
    assert context.path() == run_root


def test_a_flight_without_the_phase_pays_nothing_for_the_check(tmp_path):
    config = {"campaign_id": "legacy"}
    run_root = tmp_path / "run"
    run_root.mkdir()
    context = spec.RoutesContext(repository_root=tmp_path, run_root=run_root, artifact_root=tmp_path,
                                 config=config, config_path=tmp_path / "c.json")
    assert context.path("bakeoff", "x") == run_root / "bakeoff" / "x"


# ---------------------------------------------------------------------------
# sol round 7: the snapshot adapter and the lock-owner journal bypass RoutesContext.path
# ---------------------------------------------------------------------------

def test_a_snapshot_directory_junctioned_into_an_origin_stops_the_guarded_stages(tmp_path):
    import os
    origin = tmp_path / "origin_flight"
    (origin / "empty").mkdir(parents=True)
    context, run_root = _real_context(tmp_path, {"ablang2_masked_editor": origin})
    link = run_root / "source_snapshot"
    _junction(link, origin / "empty")
    try:
        with pytest.raises(campaign.ReadinessGate, match="junction or symbolic link"):
            campaign._require_admission_ready(context)
        assert list((origin / "empty").iterdir()) == []
    finally:
        os.rmdir(link)


def _resolve(tmp_path, run_root, **kwargs):
    repository = tmp_path / "repo"
    artifact = tmp_path / "art"
    repository.mkdir(exist_ok=True)
    artifact.mkdir(exist_ok=True)
    config = admission_config()
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return spec.resolve_context(repository, config_path=config_path, run_root=run_root,
                                artifact_root=artifact, **kwargs)


def test_a_hard_linked_owner_journal_is_refused_before_the_lock_can_append(tmp_path):
    import os
    origin_file = tmp_path / "origin" / "markers.json"
    origin_file.parent.mkdir()
    origin_file.write_bytes(b"ORIGIN")
    run_root = tmp_path / "art" / "run_x"
    run_root.mkdir(parents=True)
    os.link(origin_file, run_root / "campaign.owners.jsonl")
    with pytest.raises(ValueError, match="hard link"):
        _resolve(tmp_path, run_root)
    assert origin_file.read_bytes() == b"ORIGIN"


def test_the_read_only_form_does_not_scan_and_a_clean_root_is_accepted(tmp_path):
    import os
    origin_file = tmp_path / "origin" / "markers.json"
    origin_file.parent.mkdir()
    origin_file.write_bytes(b"ORIGIN")
    run_root = tmp_path / "art" / "run_x"
    run_root.mkdir(parents=True)
    os.link(origin_file, run_root / "campaign.owners.jsonl")
    assert _resolve(tmp_path, run_root, ensure=False).run_root == run_root      # status stays usable
    (run_root / "campaign.owners.jsonl").unlink()
    assert _resolve(tmp_path, run_root).run_root == run_root


def test_a_flight_without_the_phase_is_not_scanned(tmp_path):
    import os
    origin_file = tmp_path / "origin" / "markers.json"
    origin_file.parent.mkdir()
    origin_file.write_bytes(b"ORIGIN")
    run_root = tmp_path / "art" / "run_x"
    run_root.mkdir(parents=True)
    os.link(origin_file, run_root / "campaign.owners.jsonl")
    repository, artifact = tmp_path / "repo", tmp_path / "art"
    repository.mkdir()
    legacy = admission_config()
    legacy.pop(spec.ADMISSION_PHASE_KEY)
    legacy.pop(spec.ADMISSION_PINS_KEY)
    config_path = tmp_path / "legacy.json"
    config_path.write_text(json.dumps(legacy), encoding="utf-8")
    context = spec.resolve_context(repository, config_path=config_path, run_root=run_root,
                                   artifact_root=artifact)
    assert context.run_root == run_root
