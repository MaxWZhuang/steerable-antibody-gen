"""The stage machine: self-exclusion, strict prerequisites, and the derived freeze.

Three orchestration defects, all of which produced a *record* rather than a crash:

* ``verify`` required its own record, which does not exist while it is running. A
  complete flight therefore failed on its first verification and passed on its
  second, which is a property of the check.
* a partial selection did not stop the confirmation. A bake-off may legitimately be
  partial and let the independent downstream stages run on what exists; a partial
  selection cannot let the confirmation freeze a recipe that no selection produced.
* the confirmation revision's ``frozen_after`` list was hand-written in a config, so
  a confirmation could be frozen before a selection it claimed to follow.

Orchestration only: no torch, no model, no artifact volume. The stage functions are
replaced by recorders, because what is under test is the machine.
"""
from __future__ import annotations

import json

import pytest

from smallAntibodyGen.experiments import her2_routes_campaign as campaign
from smallAntibodyGen.experiments import her2_support_paths as paths


class Context:
    """The slice of ``RoutesContext`` the machine touches."""

    def __init__(self, root, *, config=None):
        self.run_root = root
        self.artifact_root = root
        self.repository_root = root
        self.config = config or {"revisions": {"current": "rev01"}}
        self.config_path = root / "config.json"
        root.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        self.revision = "rev01"
        self.campaign_id = "probe"

    def path(self, *parts):
        return self.run_root.joinpath(*[str(part) for part in parts])

    def artifact(self, logical):
        return self.artifact_root / str(logical)

    def repository(self, logical):
        return self.repository_root / str(logical)


def complete(context, *stages):
    for stage in stages:
        campaign.write_stage_record(context, stage, {"status": "completed"})


# --- verify excludes itself ------------------------------------------------

def test_verify_does_not_require_its_own_record(tmp_path):
    assert "verify" not in campaign.VERIFIED_STAGES
    assert set(campaign.VERIFIED_STAGES) == set(campaign.REQUIRED_STAGES) - {"verify"}


def test_verify_reports_the_stages_that_did_not_complete(tmp_path):
    context = Context(tmp_path / "run")
    complete(context, "acquire", "preflight")
    document = campaign.verify(context)
    assert not document["passed"]
    assert any("split is not_run" in problem for problem in document["problems"])
    named = {row["check"] for row in document["results"]}
    assert "stage::verify" not in named
    assert "verify_excludes_itself" in named
    assert "COMPLETENESS, not correctness" in document["asserts"]


def test_verify_checks_the_reserved_opening_is_bound_to_the_frozen_recipe(tmp_path):
    context = Context(tmp_path / "run")
    paths.write_json(context.path("frozen_recipe.json"), {"sha256": "abc", "route": "r"})
    paths.write_json(context.path("reserved_opening.json"),
                     {"frozen_selection_sha256": "abc"})
    document = campaign.verify(context)
    bound = next(row for row in document["results"]
                 if row["check"] == "reserved_opening_bound_to_frozen_selection")
    assert bound["passed"]
    paths.write_json(context.path("reserved_opening.json"),
                     {"frozen_selection_sha256": "different"})
    document = campaign.verify(context)
    bound = next(row for row in document["results"]
                 if row["check"] == "reserved_opening_bound_to_frozen_selection")
    assert not bound["passed"]
    assert any("not bound to the frozen recipe" in problem for problem in document["problems"])


def test_an_unopened_reserved_panel_adds_no_binding_check(tmp_path):
    context = Context(tmp_path / "run")
    document = campaign.verify(context)
    assert document["reserved_opened"] is False
    assert not [row for row in document["results"]
                if row["check"] == "reserved_opening_bound_to_frozen_selection"]


# --- strict prerequisites --------------------------------------------------

def test_the_confirmation_freeze_is_held_until_every_selection_completed(tmp_path):
    context = Context(tmp_path / "run")
    complete(context, "route_gate", "controls", "priors")
    assert campaign.unmet_prerequisites(context, "confirmation_freeze") == \
        ["posttrain", "decoding"]
    complete(context, "posttrain", "decoding")
    assert campaign.unmet_prerequisites(context, "confirmation_freeze") == []


def test_the_confirmation_is_held_until_the_freeze_completed(tmp_path):
    context = Context(tmp_path / "run")
    assert campaign.unmet_prerequisites(context, "confirm") == ["confirmation_freeze"]
    complete(context, "confirmation_freeze")
    assert campaign.unmet_prerequisites(context, "confirm") == []


def test_a_stage_with_no_strict_prerequisites_is_never_held(tmp_path):
    context = Context(tmp_path / "run")
    for stage in ("acquire", "bakeoff", "report", "verify"):
        assert campaign.unmet_prerequisites(context, stage) == []


def test_the_bakeoff_may_be_partial_and_the_confirmation_may_not():
    """A partial bake-off is a usable partial; a partial selection is not."""
    assert "bakeoff" not in campaign.STRICT_PREREQUISITES
    assert "confirmation_freeze" in campaign.STRICT_PREREQUISITES
    assert "confirm" in campaign.STRICT_PREREQUISITES


def test_run_all_does_not_launch_a_stage_whose_prerequisites_are_absent(tmp_path, monkeypatch):
    """Resuming a flight AT the confirmation must not freeze an absent selection.

    ``--from-stage`` makes this reachable without any upstream failure: the dependency
    blocking only fires when an upstream stage ran and did not complete, so a flight
    started at the freeze has nothing blocking it and the strict prerequisite is the
    only thing that holds it.
    """
    context = Context(tmp_path / "run")
    launched = []

    def recorder(name):
        def body(_context, _services, **_kwargs):
            launched.append(name)
            return {"status": "completed"}
        return body
    monkeypatch.setattr(campaign, "STAGE_FUNCTIONS",
                        {name: recorder(name) for name in campaign.STAGES})
    document = campaign.run_all(context, services=None,
                                stages=("confirmation_freeze", "confirm", "report"),
                                reuse=False)
    assert launched == [], (
        "the freeze is held and it blocks its declared dependants, so nothing in this "
        "subset runs and the queue says so")
    assert document["stages"]["confirmation_freeze"] == "not_run"
    assert document["stages"]["confirm"] == "not_run"
    assert document["stages"]["report"] == "not_run"
    record = campaign.read_stage_record(context, "confirmation_freeze")
    assert "strict prerequisites have not completed" in record["reason"]
    assert "a selection nothing selected" in record["consequence"]


def test_run_all_blocks_the_downstream_of_a_failed_stage(tmp_path, monkeypatch):
    context = Context(tmp_path / "run")
    launched = []

    def recorder(name):
        def body(_context, _services, **_kwargs):
            launched.append(name)
            return {"status": "completed"}
        return body
    functions = {name: recorder(name) for name in campaign.STAGES}

    def failing(_context, _services, **_kwargs):
        return {"status": "failed", "reason": "a declared failure"}
    functions["decoding"] = failing
    monkeypatch.setattr(campaign, "STAGE_FUNCTIONS", functions)
    document = campaign.run_all(context, services=None, reuse=False)
    assert "confirmation_freeze" not in launched and "confirm" not in launched
    assert document["stages"]["confirmation_freeze"] == "not_run"
    assert not document["all_required_completed"]


# --- the derived frozen_after list ----------------------------------------

def test_the_frozen_after_list_is_derived_from_the_stage_records(tmp_path):
    context = Context(tmp_path / "run")
    complete(context, "route_gate", "controls")
    block = campaign.confirmation_prerequisites(context)
    assert block["frozen_after"] == ["route_gate", "controls"]
    assert block["missing"] == ["priors", "posttrain", "decoding"]
    assert "derived from the stage records rather than declared" in block["rule"]


def test_the_freeze_stage_refuses_when_a_selection_is_missing(tmp_path):
    context = Context(tmp_path / "run")
    complete(context, "route_gate")
    with pytest.raises(campaign.ReadinessGate, match="have not completed"):
        campaign.stage_confirmation_freeze(context, services=None)


# --- the stage order ------------------------------------------------------

def test_the_split_precedes_the_sources_and_gates_them(tmp_path):
    order = list(campaign.STAGES)
    assert order.index("split") < order.index("sources"), (
        "sources opens outcome columns and the split is a function of sequences alone")
    assert "sources" in campaign.DEPENDENTS["split"]
    assert "confirmation_freeze" in campaign.DEPENDENTS["decoding"]
    assert campaign.DEPENDENTS["report"] == ("verify",)


def test_every_declared_stage_has_a_function():
    assert set(campaign.STAGE_FUNCTIONS) == set(campaign.STAGES)
