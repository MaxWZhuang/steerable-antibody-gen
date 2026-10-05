"""The explicit DPLM TRAINING phase: closed declaration, stage holds, pinned split, the HR17
training selector, persisted diagnostics and the phase verification.

Scope of what is asserted, and what is not:

* the phase is training-only. It is not competition admission and never reports a complete
  flight, and ``spec.IMPLEMENTED_AMENDMENTS`` stays empty (HR-17 is NOT marked implemented);
* the config's ``training_phase`` block is a closed schema compared with constants in code, so
  mutations of the phase, policy, split pin or schedule are refused;
* the HR17 selector is exercised on the real tiny DPLM harness of
  ``test_her2_routes_dplm_services`` (CPU, random-init trunk, no pretrained weights). The
  ranking-reversal cases script the per-window log probabilities, because a random-init model
  says nothing about the quality of a trained one.

Nothing here says anything about a trained DPLM, GPU fit, or pretrained numerical equivalence.
The CLI entry points are not driven end to end (they need the artifact volume and the
environment bootstrap); the stage-selection logic they call is.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from smallAntibodyGen.experiments import her2_routes_campaign as campaign
from smallAntibodyGen.experiments import her2_routes_services as services_mod
from smallAntibodyGen.experiments import her2_routes_spec as spec
from smallAntibodyGen.experiments import her2_support_paths as paths
from smallAntibodyGen.experiments.her2_routes_conditions import BIN_HIGH, BIN_NULL, Condition
from smallAntibodyGen.tests.test_her2_routes_dplm_services import (  # noqa: F401  (fixtures)
    PANEL,
    ROUTE_FULL,
    ROUTE_LORA,
    complete_stage,
    harness,
    reconstruction_stream,
    train,
)

ROOT = Path(__file__).resolve().parents[3]
CONFIGS = ROOT / "configs" / "experiments"
SPLIT_SHA = "3955a5981c225e94ba2b08e97e90175b5a3b6cf35393adea9b018c5542fd5d49"
SEQUENCE_SHA = "a742b610f1ea120a8d669b2d240252cb11c185dd4574978cea13314b8a575b73"


def read_config(name):
    return json.loads((CONFIGS / name).read_text(encoding="utf-8"))


NEW = read_config("her2_generative_routes_dplm_train.json")
BASE = read_config("her2_generative_routes.json")
DPLM = read_config("her2_generative_routes_dplm.json")
JOINT = read_config("her2_generative_routes_joint.json")
NOTES = ("why", "compute_note")


def without_notes(block):
    return {key: value for key, value in block.items() if key not in NOTES}


class SimpleContext:
    """The slice of ``RoutesContext`` the campaign machinery touches."""

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

    def artifact(self, logical):
        return self.artifact_root / str(logical)

    def repository(self, logical):
        return self.repository_root / str(logical)


def recorders(monkeypatch, launched, *, statuses=None):
    def make(name):
        def body(_context, _services, **_kwargs):
            launched.append(name)
            return {"status": dict(statuses or {}).get(name, "completed")}
        return body
    monkeypatch.setattr(campaign, "STAGE_FUNCTIONS", {name: make(name) for name in campaign.STAGES})


# ---------------------------------------------------------------------------
# the new config and the code-held declaration
# ---------------------------------------------------------------------------

def test_the_new_config_is_a_valid_independent_phase_config():
    document = spec.load_config(CONFIGS / "her2_generative_routes_dplm_train.json")
    policy = spec.training_phase(document)
    assert policy is not None and policy["admission"] == "none"
    assert policy["phase_id"] == "dplm_explicit_training"
    assert policy["routes"] == ["dplm_150m_masked", "dplm_650m_lora_masked"]
    assert policy["updates"] == 20000 and policy["milestones"] == [5000, 10000, 15000]
    assert policy["expected_split_sha256"] == SPLIT_SHA
    assert policy["expected_sequence_sha256"] == SEQUENCE_SHA
    assert policy["policy_sha256"] == paths.digest_document(spec.training_phase_declaration())
    assert policy["selection_metric"] == spec.SELECTION_METRIC_TRAINING
    assert policy["selection_metric"] != spec.SELECTION_METRIC_LEGACY
    for other in (BASE, DPLM, JOINT):
        assert document["campaign_id"] != other["campaign_id"]
        assert document["run_root_leaf"] != other["run_root_leaf"]
        assert document["revisions"]["current"] != other["revisions"]["current"]
        assert document["config_path"] != other["config_path"]
    assert document["revisions"]["kind"] == "development"
    # HR-17's training selector is not HR-17: the joint hold and its guard are untouched
    assert spec.IMPLEMENTED_AMENDMENTS == ()
    assert "hr17_high_null_ratio" not in spec.IMPLEMENTED_AMENDMENTS
    assert spec.competition_unresolved(document) == []
    assert spec.competition_unresolved(JOINT)          # still held


def test_the_route_blocks_and_schedule_are_the_qualified_ones_unchanged():
    for route in spec.TRAINING_PHASE_ROUTES:
        assert without_notes(NEW["routes"][route]) == without_notes(JOINT["routes"][route])
        assert without_notes(NEW["routes"][route]) == without_notes(DPLM["routes"][route])
    assert NEW["routes"]["enabled"] == list(spec.TRAINING_PHASE_ROUTES)
    # every non-prose value (numbers, lists, nested numeric blocks) is the joint config's
    for key in ("curriculum", "optimization", "smoke", "profile", "storage", "runtime",
                "conditions", "her2"):
        ours = {name: value for name, value in NEW[key].items() if not isinstance(value, str)}
        theirs = JOINT[key]
        assert ours, key
        for name, value in ours.items():
            assert theirs[name] == value, (key, name)
    assert NEW["optimization"]["selection_metric"] != JOINT["optimization"]["selection_metric"]


def test_the_split_block_is_verbatim_because_its_prose_is_part_of_the_pinned_digest():
    """``freeze_split`` records the WHOLE split block; different prose is a different digest."""
    assert NEW["split"] == JOINT["split"] == BASE["split"]


def test_downstream_sections_are_absent_so_removing_the_phase_key_cannot_resolve():
    for key in ("gates", "controls", "priors", "posttrain", "decoding", "confirmation",
                "competition"):
        assert key not in NEW


def test_the_legacy_configs_carry_no_phase_and_keep_their_stage_list():
    for document in (BASE, DPLM, JOINT):
        assert spec.TRAINING_PHASE_KEY not in document
        assert spec.training_phase(document) is None
        assert campaign.stages_for(document) == campaign.STAGES
    assert campaign.stages_for(NEW) == spec.TRAINING_PHASE_STAGES
    assert set(spec.TRAINING_PHASE_STAGES) | set(spec.TRAINING_PHASE_HELD_STAGES) == set(
        campaign.STAGES)
    assert not set(spec.TRAINING_PHASE_STAGES) & set(spec.TRAINING_PHASE_HELD_STAGES)


def mutate(path, value, *, delete=False):
    document = copy.deepcopy(NEW)
    target = document
    for part in path[:-1]:
        target = target[part]
    if delete:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    return document


MUTATIONS = {
    "empty_mapping": lambda: mutate((spec.TRAINING_PHASE_KEY,), {}),
    "null_block": lambda: mutate((spec.TRAINING_PHASE_KEY,), None),
    "list_block": lambda: mutate((spec.TRAINING_PHASE_KEY,), ["x"]),
    "missing_key": lambda: mutate((spec.TRAINING_PHASE_KEY, "held_stages"), None, delete=True),
    "extra_key": lambda: mutate((spec.TRAINING_PHASE_KEY, "relax"), True),
    "phase_id": lambda: mutate((spec.TRAINING_PHASE_KEY, "phase_id"), "other"),
    "admission": lambda: mutate((spec.TRAINING_PHASE_KEY, "admission"), "competition"),
    "route_order": lambda: mutate((spec.TRAINING_PHASE_KEY, "routes"),
                                  list(reversed(spec.TRAINING_PHASE_ROUTES))),
    "one_route": lambda: mutate((spec.TRAINING_PHASE_KEY, "routes"),
                                [spec.TRAINING_PHASE_ROUTES[0]]),
    "phase_updates": lambda: mutate((spec.TRAINING_PHASE_KEY, "updates"), 1000),
    "phase_milestones": lambda: mutate((spec.TRAINING_PHASE_KEY, "milestones"), [5000]),
    "stages_without_verify": lambda: mutate((spec.TRAINING_PHASE_KEY, "stages"),
                                            list(spec.TRAINING_PHASE_STAGES[:-1])),
    "held_relaxed": lambda: mutate((spec.TRAINING_PHASE_KEY, "held_stages"),
                                   list(spec.TRAINING_PHASE_HELD_STAGES[:-1])),
    "selection_metric": lambda: mutate((spec.TRAINING_PHASE_KEY, "selection_metric"),
                                       spec.SELECTION_METRIC_LEGACY),
    "head_diagnostics": lambda: mutate((spec.TRAINING_PHASE_KEY, "head_diagnostics"), "complete"),
    "split_pin": lambda: mutate((spec.TRAINING_PHASE_KEY, "expected_split_sha256"), "0" * 64),
    "sequence_pin": lambda: mutate((spec.TRAINING_PHASE_KEY, "expected_sequence_sha256"),
                                   "0" * 64),
    "route_set": lambda: mutate(("routes", "enabled"), list(spec.COMPETITION_ROUTES)),
    "enabled_order": lambda: mutate(("routes", "enabled"),
                                    list(reversed(spec.TRAINING_PHASE_ROUTES))),
    "route_updates": lambda: mutate(("routes", "dplm_150m_masked", "updates"), 19999),
    "route_updates_type": lambda: mutate(("routes", "dplm_650m_lora_masked", "updates"), 20000.0),
    "route_milestones": lambda: mutate(("routes", "dplm_650m_lora_masked", "milestones"),
                                       [5000, 10000]),
    "competition_graft": lambda: mutate(("competition",), dict(JOINT["competition"])),
    "confirmation_revision": lambda: mutate(("revisions", "kind"), "confirmation"),
}


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_a_mutated_phase_policy_split_or_schedule_is_refused(name):
    document = MUTATIONS[name]()
    with pytest.raises(ValueError):
        spec.training_phase(document)


def test_the_phase_cannot_be_grafted_onto_the_five_way_config():
    grafted = copy.deepcopy(JOINT)
    grafted[spec.TRAINING_PHASE_KEY] = spec.training_phase_declaration()
    with pytest.raises(ValueError):
        spec.training_phase(grafted)


def resolve(tmp_path, config):
    context = SimpleContext(tmp_path, config)
    return context, spec.resolved_protocol(
        context, snapshot={"snapshot_sha256": "s"}, external={"external_sha256": "e"},
        split={"a": 1}, profile={"a": 1}, forecast={"a": 1},
        revision={"revision": "rev01", "kind": "development"})


def test_a_complete_phase_resolves_and_records_its_policy(tmp_path):
    context, document = resolve(tmp_path, copy.deepcopy(NEW))
    assert document["unresolved"] == []
    assert document[spec.TRAINING_PHASE_KEY]["policy_sha256"] == \
        spec.training_phase(NEW)["policy_sha256"]
    assert "gates" not in document and "confirmation" not in document
    assert spec.require_resolved(context)["unresolved"] == []


def test_removing_the_phase_key_fails_production_resolution(tmp_path):
    stripped = copy.deepcopy(NEW)
    del stripped[spec.TRAINING_PHASE_KEY]
    context, document = resolve(tmp_path, stripped)
    assert {"config.gates", "config.decoding", "config.posttrain",
            "config.confirmation"} <= set(document["unresolved"])
    with pytest.raises(ValueError, match="unresolved fields"):
        spec.require_resolved(context)


@pytest.mark.parametrize("name", ["phase_id", "split_pin", "held_relaxed", "route_updates"])
def test_a_malformed_phase_is_unresolved_not_silently_ignored(tmp_path, name):
    context, document = resolve(tmp_path, MUTATIONS[name]())
    assert "training_phase.invalid" in document["unresolved"]
    assert document["training_phase_error"]
    with pytest.raises(ValueError, match="unresolved fields"):
        spec.require_resolved(context)


def test_the_joint_config_still_resolves_to_its_hold(tmp_path):
    grafted = copy.deepcopy(JOINT)
    grafted[spec.TRAINING_PHASE_KEY] = spec.training_phase_declaration()
    _context, document = resolve(tmp_path, grafted)
    assert "training_phase.invalid" in document["unresolved"]
    assert "competition.amendment::hr17_high_null_ratio" in document["unresolved"]
    _context, plain = resolve(tmp_path / "plain", copy.deepcopy(JOINT))
    assert "competition.amendment::hr17_high_null_ratio" in plain["unresolved"]
    assert spec.TRAINING_PHASE_KEY not in plain


# ---------------------------------------------------------------------------
# stage holds: direct stage, run_all, supervise, stage lists
# ---------------------------------------------------------------------------

def test_every_held_stage_is_refused_and_every_phase_stage_is_allowed(tmp_path):
    context = SimpleContext(tmp_path, copy.deepcopy(NEW))
    for stage in spec.TRAINING_PHASE_HELD_STAGES:
        with pytest.raises(ValueError, match="held by"):
            campaign.require_stage_allowed(context, stage)
    for stage in spec.TRAINING_PHASE_STAGES:
        assert campaign.require_stage_allowed(context, stage)["admission"] == "none"
    legacy = SimpleContext(tmp_path / "legacy", copy.deepcopy(JOINT))
    for stage in campaign.STAGES:
        assert campaign.require_stage_allowed(legacy, stage) is None


def test_a_direct_held_stage_refuses_before_reuse_and_before_any_effect(tmp_path, monkeypatch):
    context = SimpleContext(tmp_path, copy.deepcopy(NEW))
    launched = []
    recorders(monkeypatch, launched)
    # a completed, identity-matching record would be REUSED for a legacy stage; here it is not
    campaign.write_stage_record(context, "controls", {
        "status": "completed", "identity": campaign.stage_identity(context, "controls")})
    before = paths.read_json(campaign.stage_record_path(context, "controls"))
    with pytest.raises(ValueError, match="held by"):
        campaign.run_stage(context, "controls", None)
    with pytest.raises(ValueError, match="held by"):
        campaign.run_stage(context, "report", None, reuse=False)
    assert launched == []
    assert paths.read_json(campaign.stage_record_path(context, "controls")) == before
    assert not campaign.stage_record_path(context, "report").exists()


def test_run_all_defaults_to_the_phase_stages_and_never_claims_a_complete_flight(
        tmp_path, monkeypatch):
    context = SimpleContext(tmp_path, copy.deepcopy(NEW))
    launched = []
    recorders(monkeypatch, launched)
    document = campaign.run_all(context, services=None, reuse=False)
    assert launched == list(spec.TRAINING_PHASE_STAGES)
    assert document["required_stages"] == list(spec.TRAINING_PHASE_STAGES)
    assert document["all_required_completed"] is True
    assert document["scope"] == "training_phase"
    assert document["training_phase_completed"] is True and document["complete_flight"] is False
    for stage in spec.TRAINING_PHASE_HELD_STAGES:
        assert stage not in document["stages"]


def test_run_all_with_an_explicit_held_stage_refuses_the_whole_list_first(tmp_path, monkeypatch):
    context = SimpleContext(tmp_path, copy.deepcopy(NEW))
    launched = []
    recorders(monkeypatch, launched)
    with pytest.raises(ValueError, match="held by"):
        campaign.run_all(context, services=None, reuse=False,
                         stages=("acquire", "bakeoff", "route_gate"))
    assert launched == [], "not even the allowed stages ahead of the held one ran"
    assert not context.path(campaign.STAGE_RECORDS).exists()
    assert not context.path("run_all.json").exists()


def test_legacy_run_all_default_is_still_the_full_flight(tmp_path, monkeypatch):
    context = SimpleContext(tmp_path, {"revisions": {"current": "rev01"}})
    launched = []
    recorders(monkeypatch, launched)
    document = campaign.run_all(context, services=None, reuse=False)
    assert launched == list(campaign.STAGES)
    assert "training_phase" not in document and "complete_flight" not in document
    assert document["required_stages"] == list(campaign.REQUIRED_STAGES)


def test_supervise_defaults_to_the_phase_and_reports_a_distinct_status(tmp_path, monkeypatch):
    context = SimpleContext(tmp_path, copy.deepcopy(NEW))
    launched = []
    recorders(monkeypatch, launched)
    record = campaign.supervise(context, None, heartbeat_seconds=30.0, reuse=False)
    assert launched == list(spec.TRAINING_PHASE_STAGES)
    assert record["status"] == campaign.STATUS_TRAINING_PHASE_COMPLETED != "completed"
    assert record["status"] in campaign.SUCCESS_STATUSES
    assert record["run_all"]["complete_flight"] is False


def test_supervise_with_a_held_stage_refuses_before_the_lock_or_any_status(tmp_path, monkeypatch):
    context = SimpleContext(tmp_path, copy.deepcopy(NEW))
    launched = []
    recorders(monkeypatch, launched)
    with pytest.raises(ValueError, match="held by"):
        campaign.supervise(context, None, stages=("decoding",), reuse=False)
    assert launched == []
    for name in (campaign.LOCK_FILE, campaign.HEARTBEAT_JSON, campaign.STATUS_JSON):
        assert not context.path(name).exists()


def test_select_stages_honours_the_phase_and_keeps_the_legacy_slice(tmp_path):
    phase = SimpleContext(tmp_path, copy.deepcopy(NEW))
    assert campaign.select_stages(phase) == spec.TRAINING_PHASE_STAGES
    through = campaign.select_stages(phase, through_stage="bakeoff")
    assert through[-1] == "bakeoff" and "verify" not in through, (
        "--through-stage bakeoff truthfully stops short of the phase verification")
    assert campaign.select_stages(phase, from_stage="bakeoff") == ("bakeoff", "verify")
    for bound in spec.TRAINING_PHASE_HELD_STAGES:
        with pytest.raises(ValueError, match="not one of this config's stages"):
            campaign.select_stages(phase, through_stage=bound)
        with pytest.raises(ValueError, match="not one of this config's stages"):
            campaign.select_stages(phase, from_stage=bound)
    legacy = SimpleContext(tmp_path / "legacy", copy.deepcopy(JOINT))
    assert campaign.select_stages(legacy) == campaign.STAGES
    assert campaign.select_stages(legacy, from_stage="freeze", through_stage="decoding") == \
        campaign.STAGES[campaign.STAGES.index("freeze"):campaign.STAGES.index("decoding") + 1]


# ---------------------------------------------------------------------------
# the pinned split: refused before anything is written or trained
# ---------------------------------------------------------------------------

CERTIFICATE = {"zero_violations": True, "violations": [],
               "counts": {"reserved_meets_minimum": True, "retained_for_training": 1,
                          "purged_within_radius": 1, "panel_eligible": 1, "reserved": 1,
                          "development": 1},
               "power": {}, "reserved_status": "ok"}
FREEZE = {"record_kind": "hard_split_freeze", "sequence_sha256": "seq", "development": ["A"]}


class SplitServices:
    def __init__(self, freeze):
        self.freeze = freeze
        self.persist = []

    def freeze_hard_split(self, *, persist=True):
        self.persist.append(persist)
        return {"certificate": copy.deepcopy(CERTIFICATE), "freeze": copy.deepcopy(self.freeze)}


def pinned_context(tmp_path, monkeypatch, *, pin=None, sequence="seq"):
    monkeypatch.setattr(spec, "TRAINING_PHASE_SPLIT_SHA256",
                        pin or paths.digest_document(FREEZE))
    monkeypatch.setattr(spec, "TRAINING_PHASE_SEQUENCE_SHA256", sequence)
    config = copy.deepcopy(NEW)
    config[spec.TRAINING_PHASE_KEY] = spec.training_phase_declaration()
    return SimpleContext(tmp_path, config)


def test_a_split_that_is_not_the_pinned_one_writes_nothing(tmp_path, monkeypatch):
    context = pinned_context(tmp_path, monkeypatch, pin="0" * 64)
    services = SplitServices(FREEZE)
    record = campaign.run_stage(context, "split", services, reuse=False)
    assert record["status"] == "blocked_readiness" and "pins" in record["reason"]
    assert services.persist == [False], "the split must not be persisted before the comparison"
    assert not context.path(spec.SPLIT_FREEZE).exists()
    assert not context.path("split_certificate.json").exists()
    with pytest.raises(ValueError, match="absent"):
        spec.verify_training_phase_split(context)


def test_a_wrong_sequence_digest_is_refused_even_with_the_right_document_digest(
        tmp_path, monkeypatch):
    context = pinned_context(tmp_path, monkeypatch, sequence="other")
    record = campaign.run_stage(context, "split", SplitServices(FREEZE), reuse=False)
    assert record["status"] == "blocked_readiness"
    assert not context.path(spec.SPLIT_FREEZE).exists()


def test_the_pinned_split_is_written_and_a_rerun_keeps_the_pinned_bytes(tmp_path, monkeypatch):
    context = pinned_context(tmp_path, monkeypatch)
    record = campaign.run_stage(context, "split", SplitServices(FREEZE), reuse=False)
    assert record["status"] == "completed"
    target = context.path(spec.SPLIT_FREEZE)
    assert paths.digest_document(paths.read_json(target)) == spec.TRAINING_PHASE_SPLIT_SHA256
    first = target.read_bytes()
    # freeze_split stamps ``reconfirmed`` on a second freeze; that would move the digest every
    # training identity is bound to, so the pinned record is kept
    again = dict(FREEZE, reconfirmed=True)
    record = campaign.run_stage(context, "split", SplitServices(again), reuse=False)
    assert record["status"] == "completed" and target.read_bytes() == first
    assert spec.verify_training_phase_split(context)["split_sha256"] == \
        spec.TRAINING_PHASE_SPLIT_SHA256


def test_the_freeze_and_smoke_stages_recheck_the_pin_and_a_legacy_config_is_untouched(
        tmp_path, monkeypatch):
    context = pinned_context(tmp_path, monkeypatch)
    with pytest.raises(campaign.ReadinessGate, match="absent"):
        campaign.stage_freeze(context, None)
    with pytest.raises(campaign.ReadinessGate, match="absent"):
        campaign.stage_smoke(context, None)
    paths.write_json(context.path(spec.SPLIT_FREEZE), dict(FREEZE, development=["B"]))
    with pytest.raises(campaign.ReadinessGate, match="pins"):
        campaign._require_phase_split(context)
    legacy = SimpleContext(tmp_path / "legacy", copy.deepcopy(JOINT))
    assert campaign._require_phase_split(legacy) is None


# ---------------------------------------------------------------------------
# the HR17 training selector on the real tiny DPLM harness
# ---------------------------------------------------------------------------

def phase_harness(harness, monkeypatch, *, write_split=True):
    """Switch the harness config to a valid tiny phase: constants patched, panel and stream faked."""
    document = {"record_kind": "hard_split_freeze", "sequence_sha256": "seq"}
    if write_split:
        paths.write_json(harness.context.path(spec.SPLIT_FREEZE), document)
    monkeypatch.setattr(spec, "TRAINING_PHASE_SPLIT_SHA256", paths.digest_document(document))
    monkeypatch.setattr(spec, "TRAINING_PHASE_SEQUENCE_SHA256", "seq")
    monkeypatch.setattr(spec, "TRAINING_PHASE_UPDATES", 4)
    monkeypatch.setattr(spec, "TRAINING_PHASE_MILESTONES", (2,))
    config = harness.config
    config["revisions"] = {"kind": spec.REVISION_DEVELOPMENT}
    for route in (ROUTE_FULL, ROUTE_LORA):
        config["routes"][route]["milestones"] = [2]
    config[spec.TRAINING_PHASE_KEY] = spec.training_phase_declaration()
    services = harness.services
    services.development_panel_outcomes = lambda: copy.deepcopy(PANEL)
    services.curriculum_stream = lambda route, **_kw: reconstruction_stream(services)
    # The condition tables start at zero, where high and null coincide and the ratio is
    # identically zero (refused, by design). Real training moves them; here every model the
    # services build starts with a deterministic nonzero bin table so the selector is informative.
    build_route = services.build_route

    def informative(route, *, seed=0):
        model = build_route(route, seed=seed)
        perturb_conditions(model)
        return model
    services.build_route = informative
    return spec.training_phase(config)


class Scripted:
    """Stands in for ``window_log_probs``: per-row values chosen by the requested bin."""

    def __init__(self, services, high, null):
        encode = services.condition_spec.encode
        self.high_index = tuple(encode(Condition(family="affinity", bin=BIN_HIGH, assay="a1")))
        self.null_index = tuple(encode(Condition(family="affinity", bin=BIN_NULL, assay="a1")))
        self.values = {"high": list(high), "null": list(null)}
        self.seen = {"high": 0, "null": 0}
        self.batches = []

    def __call__(self, route, model, batch):
        index = tuple(int(value) for value in batch.condition_indices[0])
        label = "high" if index == self.high_index else "null"
        assert index in (self.high_index, self.null_index)
        rows = int(batch.tokens.shape[0])
        take = self.values[label][self.seen[label]:self.seen[label] + rows]
        self.seen[label] += rows
        self.batches.append((label, batch))
        return torch.tensor([[value] * 10 for value in take], dtype=torch.float32)


def build(harness, route=ROUTE_FULL):
    model = harness.services.build_route(route, seed=1)
    heads = harness.services.build_heads(route, model)
    return model, heads


def perturb_conditions(model, seed=11):
    """Move the zero-initialised condition tables so high and null genuinely differ."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.startswith("condition_"):
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.5)


def test_the_high_only_and_ratio_rankings_reverse_and_the_sign_is_negative_spearman(
        harness, monkeypatch):
    phase_harness(harness, monkeypatch)
    services = harness.services
    model, heads = build(harness)
    outcomes = [row["value"] for row in PANEL]                    # 8, 7, 6, 5
    assert outcomes == sorted(outcomes, reverse=True)
    # high-only ranks perfectly with pKD, but the null term explains it away: the RATIO reverses
    scripted = Scripted(services, high=[-1.0, -2.0, -3.0, -4.0], null=[-0.5, -3.0, -6.0, -9.0])
    monkeypatch.setattr(services, "window_log_probs", scripted)
    diagnostics = services.hr17_selection(ROUTE_FULL, model, heads, PANEL, step=3)
    assert diagnostics["spearman_high"] == pytest.approx(1.0)
    assert diagnostics["spearman_ratio"] == pytest.approx(-1.0)
    assert diagnostics["metric"] == pytest.approx(1.0), "the loop minimises: -rho of the ratio"
    assert diagnostics["ratio"] == pytest.approx([-0.5, 1.0, 3.0, 5.0])
    assert diagnostics["high"] == pytest.approx([-1.0, -2.0, -3.0, -4.0])
    assert diagnostics["null"] == pytest.approx([-0.5, -3.0, -6.0, -9.0])
    assert diagnostics["divisor"] == 10 and diagnostics["rows"] == 4
    # the same panel with an aligned ratio is the better (lower) metric
    aligned = Scripted(services, high=[-1.0, -2.0, -3.0, -4.0], null=[-0.8, -1.6, -2.4, -3.2])
    monkeypatch.setattr(services, "window_log_probs", aligned)
    better = services.hr17_selection(ROUTE_FULL, model, heads, PANEL, step=3)
    assert better["spearman_ratio"] == pytest.approx(1.0)
    assert better["metric"] == pytest.approx(-1.0)
    assert better["metric"] < diagnostics["metric"]
    paths.canonical_json(diagnostics)                # JSON-compatible, no NaN, no numpy


def test_undefined_diagnostic_correlations_are_none_never_nan(harness, monkeypatch):
    phase_harness(harness, monkeypatch)
    services = harness.services
    model, heads = build(harness)
    # a constant high term (undefined high-only rho) with a varying null term: ratio is defined
    scripted = Scripted(services, high=[-1.0] * 4, null=[-1.0, -2.0, -3.0, -5.0])
    monkeypatch.setattr(services, "window_log_probs", scripted)
    diagnostics = services.hr17_selection(ROUTE_FULL, model, heads, PANEL, step=1)
    assert diagnostics["spearman_high"] is None
    assert diagnostics["spearman_ratio"] == pytest.approx(-1.0)   # ratio [0, 1, 2, 4] vs 8..5
    assert diagnostics["metric"] == pytest.approx(1.0)
    assert diagnostics["spearman_null"] == pytest.approx(1.0)
    text = paths.canonical_json(diagnostics)          # allow_nan=False would raise on a NaN
    assert "NaN" not in text


def test_the_two_terms_differ_only_in_the_bin_and_nothing_is_mutated(harness, monkeypatch):
    phase_harness(harness, monkeypatch)
    services = harness.services
    model, heads = build(harness)
    panel = copy.deepcopy(PANEL)
    scaffold = copy.deepcopy(services.scaffold())
    scripted = Scripted(services, high=[-1.0, -2.0, -3.0, -4.0], null=[-1.5, -1.0, -5.0, -2.0])
    monkeypatch.setattr(services, "window_log_probs", scripted)
    services.hr17_selection_terms(ROUTE_FULL, model, heads, panel, batch_size=2)
    assert [label for label, _batch in scripted.batches] == ["high", "null", "high", "null"]
    for (label_a, high), (label_b, null) in zip(scripted.batches[0::2], scripted.batches[1::2]):
        assert (label_a, label_b) == ("high", "null")
        for name in ("tokens", "query_slots", "targets", "attention_mask", "target_mask"):
            assert np.array_equal(getattr(high, name), getattr(null, name)), name
        assert high.identities == null.identities
        moved = np.flatnonzero((high.condition_indices != null.condition_indices).any(axis=0))
        assert len(moved) == 1, "the requested bin is the only condition column that differs"
        assert np.array_equal(high.condition_indices[:, moved[0]],
                              np.full(len(high.condition_indices),
                                      services.condition_spec.encode(
                                          Condition(family="affinity", bin=BIN_HIGH,
                                                    assay="a1"))[moved[0]]))
    assert panel == PANEL and services.scaffold() == scaffold


def test_a_constant_tied_or_nonfinite_evidence_is_refused_with_no_fallback(harness, monkeypatch):
    phase_harness(harness, monkeypatch)
    services = harness.services
    model, heads = build(harness)

    def run(high, null, panel=None):
        monkeypatch.setattr(services, "window_log_probs", Scripted(services, high, null))
        return services.hr17_selection(ROUTE_FULL, model, heads, panel or PANEL, step=1)
    with pytest.raises(ValueError, match="undefined"):
        run([-1.0, -2.0, -3.0, -4.0], [-1.0, -2.0, -3.0, -4.0])        # ratio identically zero
    with pytest.raises(ValueError, match="undefined"):
        run([-1.0, -2.0, -3.0, -4.0], [-1.5, -2.5, -3.5, -4.5])        # a constant, tied ratio
    with pytest.raises(ValueError, match="nonfinite"):
        run([-1.0, float("nan"), -3.0, -4.0], [-1.0, -2.0, -3.0, -5.0])
    with pytest.raises(ValueError, match="nonfinite"):
        run([-1.0, -2.0, float("-inf"), -4.0], [-1.0, -2.0, -3.0, -5.0])
    nan_outcome = [dict(row) for row in PANEL]
    nan_outcome[1]["value"] = float("nan")
    with pytest.raises(ValueError):
        run([-1.0, -2.0, -3.0, -4.0], [-1.0, -3.0, -3.5, -9.0], nan_outcome)
    # a partially tied ratio is still a defined ranking (average ranks) and is accepted
    assert run([-1.0, -2.0, -3.0, -4.0], [-2.0, -3.0, -3.0, -7.0])["rows"] == 4


def test_malformed_and_mismatched_panels_are_refused_before_any_forward(harness, monkeypatch):
    phase_harness(harness, monkeypatch)
    services = harness.services
    model, heads = build(harness)
    calls = []
    monkeypatch.setattr(services, "window_log_probs", lambda *a: calls.append(a))
    with pytest.raises(ValueError, match="empty"):
        services.hr17_selection(ROUTE_FULL, model, heads, [], step=1)
    for bad in ([{"window": "WGGDGFYAMD"}], [{"value": 1.0}], ["WGGDGFYAMD"],
                [{"window": "WGGDGFYAM", "value": 1.0}],        # nine residues
                [{"window": "WGGDGFYAMDA", "value": 1.0}],      # eleven residues
                [{"window": "WGGDGFYAMB", "value": 1.0}],       # not a canonical residue
                [{"window": None, "value": 1.0}]):
        with pytest.raises(ValueError):
            services.hr17_selection(ROUTE_FULL, model, heads, bad, step=1)
    monkeypatch.setattr(services_mod, "her2_core_edit_map",
                        lambda: serial_sites()[:9])
    with pytest.raises(ValueError, match="not ten"):
        services.hr17_selection(ROUTE_FULL, model, heads, PANEL, step=1)
    with pytest.raises(ValueError, match="DPLM routes"):
        services.hr17_selection("ablang2_masked_editor", model, heads, PANEL, step=1)
    assert calls == []


def serial_sites():
    from smallAntibodyGen.experiments import her2_routes_serial as serial
    return serial.her2_core_edit_map()


@pytest.mark.parametrize("route", [ROUTE_FULL, ROUTE_LORA])
def test_selection_is_deterministic_and_leaves_rng_weights_and_gradients_alone(
        harness, monkeypatch, route):
    phase_harness(harness, monkeypatch)
    services = harness.services
    model, heads = build(harness, route)
    perturb_conditions(model)
    original = services.window_log_probs

    def consuming(route_name, model_, batch):
        torch.rand(3)                                    # would advance the global generator
        return original(route_name, model_, batch)
    monkeypatch.setattr(services, "window_log_probs", consuming)
    for parameter in list(model.parameters()) + list(heads.parameters()):
        if parameter.requires_grad:
            parameter.grad = torch.full_like(parameter, 0.25)
    weights = {key: value.clone() for key, value in model.state_dict().items()}
    grads = [None if p.grad is None else p.grad.clone()
             for p in list(model.parameters()) + list(heads.parameters())]
    rng = torch.get_rng_state()
    first = services.hr17_selection(route, model, heads, PANEL, step=5)
    assert torch.equal(rng, torch.get_rng_state()), "the evaluation consumed generator state"
    second = services.hr17_selection(route, model, heads, PANEL, step=5)
    assert first == second, "the selection is deterministic"
    assert len(set(first["ratio"])) > 1, "the perturbed tables must make the ratio informative"
    assert all(torch.equal(weights[key], value) for key, value in model.state_dict().items())
    after = [None if p.grad is None else p.grad
             for p in list(model.parameters()) + list(heads.parameters())]
    assert all((a is None and b is None) or torch.equal(a, b) for a, b in zip(grads, after))


@pytest.mark.parametrize("initial", ["train", "eval", "mixed"])
@pytest.mark.parametrize("fail", [False, True])
def test_model_and_heads_are_evaluated_in_eval_and_their_modes_are_restored(
        harness, monkeypatch, initial, fail):
    phase_harness(harness, monkeypatch)
    services = harness.services
    model, heads = build(harness)
    perturb_conditions(model)
    model.train(initial == "train")
    heads.train(initial == "train")
    if initial == "mixed":
        model.train()
        heads.eval()
        list(model.modules())[3].training = False
    before = ([(m, m.training) for m in model.modules()], [(m, m.training) for m in heads.modules()])
    original = services.window_log_probs
    observed = []

    def watching(route_name, model_, batch):
        observed.append((model_.training, heads.training,
                         any(m.training for m in model_.modules()),
                         any(m.training for m in heads.modules())))
        if fail and len(observed) == 2:
            raise RuntimeError("a forward failed")
        return original(route_name, model_, batch)
    monkeypatch.setattr(services, "window_log_probs", watching)
    if fail:
        with pytest.raises(RuntimeError, match="a forward failed"):
            services.hr17_selection(ROUTE_FULL, model, heads, PANEL, step=1)
    else:
        services.hr17_selection(ROUTE_FULL, model, heads, PANEL, step=1)
    assert observed and all(row == (False, False, False, False) for row in observed), (
        "model AND heads must be in eval for every forward")
    assert all(module.training is flag for module, flag in before[0] + before[1])


def test_the_selection_metric_dispatch_keeps_the_legacy_default_unchanged(harness, monkeypatch):
    services = harness.services
    legacy = services.selection_metric(ROUTE_FULL)          # the real legacy callback
    assert callable(legacy) and legacy.__name__ == "metric"
    assert getattr(legacy, "selection_metric", None) is None
    assert getattr(legacy, "last_diagnostics", None) is None
    assert services.hr17_training_metric(ROUTE_FULL).selection_metric == \
        spec.SELECTION_METRIC_TRAINING
    sentinel = object()
    monkeypatch.setattr(services, "development_metric", lambda route: (sentinel, route))
    assert services.selection_metric(ROUTE_FULL) == (sentinel, ROUTE_FULL)
    assert services.selection_metric("ablang2_masked_editor") == (sentinel, "ablang2_masked_editor")


def test_under_the_phase_the_dispatch_selects_hr17_and_refuses_other_routes(harness,
                                                                          monkeypatch):
    phase_harness(harness, monkeypatch)
    services = harness.services
    monkeypatch.setattr(services, "development_metric",
                        lambda route: pytest.fail("the legacy selector must not run"))
    for route in (ROUTE_FULL, ROUTE_LORA):
        callback = services.selection_metric(route)
        assert callback.selection_metric == spec.SELECTION_METRIC_TRAINING
        assert callback.last_diagnostics is None
    for route in ("ablang2_masked_editor", "piggen_causal_infiller", "dwjs_energy"):
        with pytest.raises(ValueError, match="selects only"):
            services.selection_metric(route)


def test_a_legacy_train_run_keeps_its_shape_and_grows_no_phase_fields(harness):
    directory = harness.root / "legacy_run"
    record = train(harness, ROUTE_FULL, updates=2, directory=directory,
                   evaluate=lambda model, heads, step: 1.0 / (step + 1))
    assert "training_phase" not in record
    assert all("selection_diagnostics" not in entry for entry in record["history"])
    assert "training_phase" not in paths.read_json(directory / "run.json")
    assert sorted(record["identity"]) == sorted(
        harness.services.training_identity(ROUTE_FULL, arm="full", seed=5))


# ---------------------------------------------------------------------------
# training under the phase: policy in the record, diagnostics persisted through a resume
# ---------------------------------------------------------------------------

def test_training_under_the_phase_needs_the_phase_selector_and_its_route(harness, monkeypatch):
    phase_harness(harness, monkeypatch)
    for bad in (None, lambda model, heads, step: 0.5):
        with pytest.raises(ValueError, match="selects checkpoints with"):
            train(harness, ROUTE_FULL, updates=1, directory=harness.root / "bad", evaluate=bad)
    with pytest.raises(ValueError, match="trains only"):
        harness.services.train("ablang2_masked_editor", stream=None, updates=1, seed=1,
                               directory=harness.root / "other",
                               evaluate=harness.services.selection_metric(ROUTE_FULL))
    assert not (harness.root / "bad").exists() and not (harness.root / "other").exists()


@pytest.mark.parametrize("route", [ROUTE_FULL, ROUTE_LORA])
def test_policy_and_diagnostics_persist_through_a_real_train_and_resume(
        harness, monkeypatch, route):
    policy = phase_harness(harness, monkeypatch)
    services = harness.services
    # the shared ``train`` helper takes no milestones, so call the service with (2,) requested
    full = services.train(route, stream=reconstruction_stream(services), updates=4, seed=5,
                          directory=harness.root / "full", resume=False, milestones=(2,),
                          evaluate=services.selection_metric(route))
    assert full["training_phase"] == {
        "phase_id": policy["phase_id"], "policy_sha256": policy["policy_sha256"],
        "admission": "none", "selection_metric": spec.SELECTION_METRIC_TRAINING,
        "head_diagnostics": "deferred", "note": full["training_phase"]["note"]}
    written = paths.read_json(harness.root / "full" / "run.json")
    assert written["training_phase"]["policy_sha256"] == policy["policy_sha256"]
    evaluated = [entry for entry in full["history"] if "selection_diagnostics" in entry]
    assert [entry["step"] for entry in evaluated] == [1, 2, 3, 4]      # eval_every == 1
    for entry in evaluated:
        block = entry["selection_diagnostics"]
        assert block["metric_name"] == spec.SELECTION_METRIC_TRAINING
        assert block["metric"] == pytest.approx(entry["development_metric"])
        assert len(block["high"]) == len(block["null"]) == len(block["ratio"]) == len(PANEL)
        assert block["divisor"] == 10 and block["step"] == entry["step"]
    paths.canonical_json(full)
    # identity is not weakened or extended by the phase: the same fields as a legacy identity
    assert sorted(full["identity"]) == sorted(services.training_identity(route, arm="full",
                                                                         seed=5))
    train(harness, route, updates=2, directory=harness.root / "interrupted",
          evaluate=services.selection_metric(route))
    resumed = train(harness, route, updates=4, directory=harness.root / "interrupted",
                    resume=True, evaluate=services.selection_metric(route))
    assert resumed["resume"] is not None and resumed["resume"]["step"] == 2
    assert resumed["training_phase"]["policy_sha256"] == policy["policy_sha256"]
    steps = [entry["step"] for entry in resumed["history"] if "selection_diagnostics" in entry]
    assert steps == [1, 2, 3, 4], "steps 1-2 come back out of the checkpoint history"
    for left, right in zip(evaluated, [e for e in resumed["history"]
                                       if "selection_diagnostics" in e]):
        assert left["selection_diagnostics"]["metric"] == pytest.approx(
            right["selection_diagnostics"]["metric"], abs=1e-5)
    # evaluation consumed no generator state, so the resumed path is the uninterrupted one
    from smallAntibodyGen.experiments import her2_routes_checkpoints as checkpoints
    report = checkpoints.resumed_trajectory_report(full["history"], resumed["history"],
                                                   tolerance=1e-6)
    assert report["identical"], report["offenders"]
    assert (harness.root / "full" / "best.pt").is_file()
    assert (harness.root / "full" / "step2.pt").is_file()


# ---------------------------------------------------------------------------
# bakeoff and verification on the real harness
# ---------------------------------------------------------------------------

ARTIFACTS = ("native_evidence.json", "source_inventory.json", "split_certificate.json",
             spec.PROFILE_JSON, spec.FORECAST_JSON, spec.REVISION_MARKER)


def prepare_phase_run(harness, monkeypatch):
    """Frozen-looking artifacts first (identities are built from them), then a real bakeoff."""
    policy = phase_harness(harness, monkeypatch)
    context = harness.context
    for logical in ARTIFACTS:
        paths.write_json(context.path(logical), {"present": True})
    paths.write_json(context.path(spec.EXTERNAL_MARKER),
                     {"external_sha256": "e" * 64, "files": {}})
    paths.write_json(context.path("resolved_protocol.json"),
                     {spec.TRAINING_PHASE_KEY: policy, "unresolved": []})
    monkeypatch.setattr(spec, "verify_source_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(spec, "verify_external_sources", lambda *a, **k: None)
    result = harness.services.run_bakeoff()
    assert result["all_terminal"] and result["training_phase"]["policy_sha256"] == \
        policy["policy_sha256"]
    for stage in spec.TRAINING_PHASE_STAGES[:-1]:
        complete_stage(context, stage)
    return policy


def test_a_bakeoff_under_the_wrong_split_stops_before_the_first_update(harness, monkeypatch):
    phase_harness(harness, monkeypatch, write_split=False)
    with pytest.raises(ValueError, match="absent"):
        harness.services.run_bakeoff()
    assert not harness.context.path("bakeoff").exists()
    paths.write_json(harness.context.path(spec.SPLIT_FREEZE),
                     {"record_kind": "hard_split_freeze", "sequence_sha256": "different"})
    with pytest.raises(ValueError, match="pins"):
        harness.services.run_bakeoff()
    assert not harness.context.path("bakeoff").exists()


def test_the_bakeoff_stage_records_the_policy_and_re_verifies_the_split(harness, monkeypatch):
    policy = phase_harness(harness, monkeypatch)
    monkeypatch.setattr(spec, "verify_source_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(spec, "verify_external_sources", lambda *a, **k: None)
    monkeypatch.setattr(spec, "require_resolved", lambda *a, **k: None)
    monkeypatch.setattr(spec, "require_revision_discipline", lambda *a, **k: None)
    record = campaign.stage_bakeoff(harness.context, harness.services)
    assert record["status"] == "completed"
    assert record["training_phase"]["policy_sha256"] == policy["policy_sha256"]
    assert record["training_phase"]["selection_metric"] == spec.SELECTION_METRIC_TRAINING
    for route in (ROUTE_FULL, ROUTE_LORA):
        assert (harness.context.path("bakeoff", route, "last.pt")).is_file()


def check_named(document, name):
    return next(row for row in document["results"] if row["check"] == name)


def test_a_complete_phase_verifies_and_is_never_a_complete_flight(harness, monkeypatch):
    policy = prepare_phase_run(harness, monkeypatch)
    context, services = harness.context, harness.services
    document = campaign.verify(context, services)
    assert document["passed"], document["problems"]
    assert document["training_phase_completed"] is True and document["complete_flight"] is False
    assert document["training_phase"]["policy_sha256"] == policy["policy_sha256"]
    assert "NOT a complete flight" in document["asserts"]
    assert document["verified_stages"] == list(spec.TRAINING_PHASE_STAGES[:-1])
    for route in (ROUTE_FULL, ROUTE_LORA):
        for name in ("last.pt", "best.pt", "step2.pt"):
            assert check_named(document, f"phase::checkpoint::{route}::{name}")["passed"]
        assert check_named(document, f"phase::run_record::{route}")["passed"]
    assert len(paths.canonical_json(campaign._plain(document))) < 200_000, (
        "small scalars only: no tensors in the verification JSON")
    record = campaign.stage_verify(context, services)
    assert record["status"] == "completed" and record["training_phase_completed"] is True
    assert record["complete_flight"] is False
    assert paths.read_json(context.path("verification.json"))["complete_flight"] is False
    # verify is not "reused" from a stale record under the phase
    policy_context = context
    campaign.write_stage_record(policy_context, "verify", {
        "status": "completed", "identity": campaign.stage_identity(policy_context, "verify")})
    rerun = campaign.run_stage(policy_context, "verify", services)
    assert "reused" not in rerun


TAMPER = {
    "route": lambda identity: "ablang2_masked_editor",
    "arm": lambda identity: "reconstruction_only",
    "seed": lambda identity: int(identity["seed"]) + 1,
    "config_sha256": lambda identity: "1" * 64,
    "snapshot_sha256": lambda identity: "2" * 64,
    "external_sha256": lambda identity: "3" * 64,
    "condition_vocabulary_sha256": lambda identity: "4" * 64,
    "split_sha256": lambda identity: "5" * 64,
    "schedule_sha256": lambda identity: "6" * 64,
    "native_sha256": lambda identity: "7" * 64,
}


def test_verify_checks_every_checkpoint_against_the_full_expected_identity(harness, monkeypatch):
    prepare_phase_run(harness, monkeypatch)
    context, services = harness.context, harness.services
    target = context.path("bakeoff", ROUTE_LORA, "step2.pt")
    original = target.read_bytes()
    label = f"phase::checkpoint::{ROUTE_LORA}::step2.pt"
    for field, change in sorted(TAMPER.items()):
        payload = torch.load(str(target), map_location="cpu", weights_only=False)
        payload["identity"][field] = change(payload["identity"])
        torch.save(payload, target)
        document = campaign.verify(context, services)
        row = check_named(document, label)
        assert not row["passed"] and field in row["identity_differences"], field
        assert not document["passed"] and document["training_phase_completed"] is False
        target.write_bytes(original)
    payload = torch.load(str(target), map_location="cpu", weights_only=False)
    del payload["identity"]["native_sha256"]                      # a missing field is a mismatch
    torch.save(payload, target)
    assert "native_sha256" in check_named(campaign.verify(context, services),
                                          label)["identity_differences"]
    payload["identity"] = dict(services.training_identity(ROUTE_LORA, arm="full", seed=5))
    payload["step"] += 1                                            # right identity, wrong step
    torch.save(payload, target)
    assert not check_named(campaign.verify(context, services), label)["passed"]
    target.write_bytes(original)
    assert campaign.verify(context, services)["passed"]


def test_verify_refuses_a_swapped_missing_or_unbound_checkpoint(harness, monkeypatch):
    prepare_phase_run(harness, monkeypatch)
    context, services = harness.context, harness.services
    full_last = context.path("bakeoff", ROUTE_FULL, "last.pt")
    lora_last = context.path("bakeoff", ROUTE_LORA, "last.pt")
    saved = lora_last.read_bytes()
    lora_last.write_bytes(full_last.read_bytes())                   # another route's checkpoint
    row = check_named(campaign.verify(context, services),
                      f"phase::checkpoint::{ROUTE_LORA}::last.pt")
    assert not row["passed"] and {"route", "native_sha256"} <= set(row["identity_differences"])
    lora_last.write_bytes(saved)
    lora_last.rename(lora_last.with_name("moved.pt"))
    document = campaign.verify(context, services)
    assert not check_named(document, f"phase::checkpoint::{ROUTE_LORA}::last.pt")["passed"]
    lora_last.with_name("moved.pt").rename(lora_last)
    run_path = context.path("bakeoff", ROUTE_FULL, "run.json")
    run = paths.read_json(run_path)
    run["training_phase"]["policy_sha256"] = "0" * 64                # a record under another policy
    paths.write_json(run_path, run)
    assert not check_named(campaign.verify(context, services),
                           f"phase::run_record::{ROUTE_FULL}")["passed"]
    assert not campaign.verify(context)["passed"], "no services means no identity to compare"


def test_verify_fails_when_a_held_stage_or_a_reserved_opening_appears(harness, monkeypatch):
    prepare_phase_run(harness, monkeypatch)
    context, services = harness.context, harness.services
    complete_stage(context, "report")
    document = campaign.verify(context, services)
    assert not check_named(document, "phase::held::report")["passed"]
    assert not document["passed"] and document["training_phase_completed"] is False
    campaign.stage_record_path(context, "report").unlink()
    paths.write_json(context.path("reserved_opening.json"), {"frozen_selection_sha256": "x"})
    document = campaign.verify(context, services)
    assert not check_named(document, "phase::absent::reserved_opening.json")["passed"]


@pytest.mark.parametrize("status", ["failed", "partial"])
def test_verify_fails_on_a_failed_or_partial_held_stage_record(harness, monkeypatch, status):
    prepare_phase_run(harness, monkeypatch)
    context, services = harness.context, harness.services
    assert campaign.verify(context, services)["passed"]
    record_path = campaign.stage_record_path(context, "route_gate")
    campaign.write_stage_record(context, "route_gate", {"status": status})
    document = campaign.verify(context, services)
    row = check_named(document, "phase::held::route_gate")
    assert not row["passed"] and row["status"] == status
    assert not document["passed"] and document["training_phase_completed"] is False
    assert record_path.is_file(), "evidence is reported, never deleted"


ORPHAN_FILES = ("route_gate.json", "controls.json", "priors.json", "posttrain.json",
                "decoding.json", "confirmation.json", "report.json", "frozen_recipe.json",
                "reserved_opening.json")
ORPHAN_DIRECTORY_FILES = (("controls", "full", "best.pt"), ("priors", "x", "last.pt"),
                          ("posttrain", "conditional_fit", "last.pt"),
                          ("decoding", "raw_draws.json"), ("confirm", "seed1", "best.pt"),
                          ("report", "figure.png"))


@pytest.mark.parametrize("logical", ORPHAN_FILES)
def test_verify_fails_on_an_orphan_held_stage_file(harness, monkeypatch, logical):
    prepare_phase_run(harness, monkeypatch)
    context, services = harness.context, harness.services
    target = context.path(logical)
    paths.write_json(target, {"orphan": True})
    document = campaign.verify(context, services)
    assert not check_named(document, f"phase::absent::{logical}")["passed"]
    assert not document["passed"] and document["training_phase_completed"] is False
    assert target.is_file(), "evidence is reported, never deleted"


@pytest.mark.parametrize("parts", ORPHAN_DIRECTORY_FILES)
def test_verify_fails_on_an_orphan_held_stage_directory(harness, monkeypatch, parts):
    prepare_phase_run(harness, monkeypatch)
    context, services = harness.context, harness.services
    target = context.path(*parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"orphan")
    document = campaign.verify(context, services)
    assert not check_named(document, f"phase::absent::{parts[0]}/")["passed"]
    assert not document["passed"] and document["training_phase_completed"] is False
    assert target.is_file(), "evidence is reported, never deleted"


def test_verify_fails_when_the_split_on_disk_is_not_the_pinned_one(harness, monkeypatch):
    prepare_phase_run(harness, monkeypatch)
    context, services = harness.context, harness.services
    paths.write_json(context.path(spec.SPLIT_FREEZE),
                     {"record_kind": "hard_split_freeze", "sequence_sha256": "seq", "x": 1})
    document = campaign.verify(context, services)
    assert not check_named(document, "phase::split_pinned")["passed"]
    assert not check_named(document, f"phase::identity_split_pinned::{ROUTE_FULL}")["passed"]
    assert not check_named(document, f"phase::checkpoint::{ROUTE_FULL}::last.pt")["passed"]
    assert document["complete_flight"] is False and document["training_phase_completed"] is False


def test_verify_fails_on_an_invalid_declaration_and_claims_nothing_complete(harness, monkeypatch):
    prepare_phase_run(harness, monkeypatch)
    context, services = harness.context, harness.services
    monkeypatch.setattr(spec, "TRAINING_PHASE_SPLIT_SHA256", "0" * 64)    # config block now differs
    document = campaign.verify(context, services)
    assert not check_named(document, "phase::declaration_valid")["passed"]
    assert not document["passed"]
    assert document["complete_flight"] is False and document["training_phase_completed"] is False


def test_the_completed_phase_status_word_is_not_completed():
    assert campaign.STATUS_TRAINING_PHASE_COMPLETED != campaign.STATUS_COMPLETED
    assert set(campaign.SUCCESS_STATUSES) == {"completed", "training_phase_completed"}
