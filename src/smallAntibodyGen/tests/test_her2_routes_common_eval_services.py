"""The two opt-in additions to ``FlightServices`` that the common evaluation needs.

Both are Class-K changes (a new knob whose default is the old behaviour exactly):

* ``score_panel(..., requested_bin=None)`` -- ``None`` means the high-affinity bin, which is
  what the method always scored. A null-bin score for ANY route then comes out of the same
  serialization and collation path rather than a second copy of it (encoding-parity rule).
* ``FlightServices(..., source_cache=None)`` -- where the source readers keep their parsed
  caches. ``None`` is the shared ``<artifact>/cache/sources`` it always used. Several readers
  rewrite their cache non-atomically on a miss, so a read-only caller must be able to point
  them at a scratch copy instead of the directory a live flight shares.

What is real here: the tiny DPLM harness's forward passes (the parity test), and the real
``collate``/serialization for the non-DPLM routes' request path. What is replaced: only
``window_log_probs`` where a test needs scripted per-bin values, and ``collate`` where a test
only needs to see which condition was requested. Nothing here says anything about a trained
model.
"""
from __future__ import annotations

import inspect
import types
from pathlib import Path

import pytest
import torch

from smallAntibodyGen.experiments import her2_routes_serial as serial
from smallAntibodyGen.experiments import her2_routes_services as services_mod
from smallAntibodyGen.experiments.her2_routes_conditions import (
    BIN_HIGH,
    BIN_LOW,
    BIN_NULL,
    Condition,
)
from smallAntibodyGen.tests.test_her2_routes_dplm_services import (  # noqa: F401  (fixtures)
    PANEL,
    ROUTE_FULL,
    harness,
)
from smallAntibodyGen.tests.test_her2_routes_dplm_training_phase import (
    Scripted,
    build,
    perturb_conditions,
)
from smallAntibodyGen.tests.test_her2_routes_serial import VOCAB

ROUTE_MASKED = services_mod.ROUTE_MASKED
ROUTE_CAUSAL = services_mod.ROUTE_CAUSAL
ROUTE_ENERGY = services_mod.ROUTE_ENERGY


# ---------------------------------------------------------------------------
# score_panel(requested_bin=...)
# ---------------------------------------------------------------------------

def scripted_scores(harness, monkeypatch, requested_bin="unset"):
    services = harness.services
    model, heads = build(harness)
    scripted = Scripted(services, high=[-1.0, -2.0, -3.0, -4.0], null=[-1.5, -1.0, -5.0, -2.0])
    monkeypatch.setattr(services, "window_log_probs", scripted)
    kwargs = {} if requested_bin == "unset" else {"requested_bin": requested_bin}
    scores = services.score_panel(ROUTE_FULL, None, PANEL, batch_size=2, model=model,
                                  heads=heads, **kwargs)
    return scores, scripted


def test_the_default_bin_is_high_and_is_what_the_method_always_scored(harness, monkeypatch):
    scores, scripted = scripted_scores(harness, monkeypatch)
    assert scores == pytest.approx([-1.0, -2.0, -3.0, -4.0])
    assert {label for label, _batch in scripted.batches} == {"high"}


def test_an_explicit_high_request_is_identical_to_the_default(harness, monkeypatch):
    default, _ = scripted_scores(harness, monkeypatch)
    explicit, scripted = scripted_scores(harness, monkeypatch, requested_bin=BIN_HIGH)
    assert explicit == default
    assert {label for label, _batch in scripted.batches} == {"high"}


def test_a_null_request_scores_the_null_condition_on_the_same_rows(harness, monkeypatch):
    scores, scripted = scripted_scores(harness, monkeypatch, requested_bin=BIN_NULL)
    assert scores == pytest.approx([-1.5, -1.0, -5.0, -2.0])
    assert {label for label, _batch in scripted.batches} == {"null"}


def test_an_unknown_bin_is_refused_rather_than_scored_as_high(harness, monkeypatch):
    services = harness.services
    model, heads = build(harness)
    monkeypatch.setattr(services, "window_log_probs",
                        lambda *a, **k: pytest.fail("a refused request must not reach a forward"))
    with pytest.raises(ValueError, match="Unknown condition bin"):
        services.score_panel(ROUTE_FULL, None, PANEL, model=model, heads=heads,
                             requested_bin="bogus")


def test_the_energy_route_has_no_bin_and_refuses_one(harness, monkeypatch):
    services = harness.services
    monkeypatch.setattr(services, "energy_panel_scores",
                        lambda *a, **k: pytest.fail("a refused request must not be scored"))
    with pytest.raises(ValueError, match="no bin"):
        services.score_panel(ROUTE_ENERGY, None, PANEL, model=object(), heads=None,
                             requested_bin=BIN_NULL)


def test_score_panel_high_and_null_equal_the_hr17_terms_on_a_real_forward(harness):
    """Two independent code paths, one answer, on non-degenerate (perturbed) tables.

    ``hr17_selection_terms`` is the DPLM training selector and builds its own views; the new
    request path must reproduce its per-row high and null terms exactly. The tables are
    perturbed because at zero initialisation high and null coincide and ``0 == 0`` would pass
    even if the new argument were ignored.
    """
    services = harness.services
    model, heads = build(harness)
    perturb_conditions(model)
    terms = services.hr17_selection_terms(ROUTE_FULL, model, heads, PANEL, batch_size=2)
    high = services.score_panel(ROUTE_FULL, None, PANEL, batch_size=2, model=model, heads=heads,
                                requested_bin=BIN_HIGH)
    null = services.score_panel(ROUTE_FULL, None, PANEL, batch_size=2, model=model, heads=heads,
                                requested_bin=BIN_NULL)
    assert high == pytest.approx(terms["high"], abs=1e-6)
    assert null == pytest.approx(terms["null"], abs=1e-6)
    assert max(abs(h - n) for h, n in zip(high, null)) > 1e-4, (
        "the perturbed tables must make the two bins differ, or this test proves nothing")


@pytest.fixture
def bound_piggen_vocabulary():
    """The causal serializer refuses an unbound residue table; bind one and restore the old one.

    ``bind_piggen_vocabulary`` writes a module-global, so leaving a fake behind would make every
    later test in the session depend on collection order.
    """
    previous = dict(serial.PIGGEN_ID_OF_RESIDUE)
    serial.bind_piggen_vocabulary(VOCAB)
    try:
        yield
    finally:
        serial.PIGGEN_ID_OF_RESIDUE.clear()
        serial.PIGGEN_ID_OF_RESIDUE.update(previous)


@pytest.mark.parametrize("route", [ROUTE_MASKED, ROUTE_CAUSAL])
def test_the_requested_bin_reaches_collation_for_the_non_dplm_routes(
        harness, monkeypatch, bound_piggen_vocabulary, route):
    """AbLang2 and p-IgGen cannot be built in a unit test; what must hold is the request path.

    Every view handed to ``collate`` carries exactly the requested condition (same family and
    assay as the default), for each route's own serialization.
    """
    services = harness.services
    harness.config["routes"].setdefault(route, {})["max_positions"] = 256
    seen = []

    def collate(requested_route, views):
        seen.append((requested_route, list(views)))
        return types.SimpleNamespace(rows=len(views))

    monkeypatch.setattr(services, "collate", collate)
    monkeypatch.setattr(services, "window_log_probs",
                        lambda r, m, batch: torch.zeros((batch.rows, 10)))
    for requested_bin in (None, BIN_NULL, BIN_LOW):
        seen.clear()
        services.score_panel(route, None, PANEL, batch_size=3, model=object(), heads=None,
                             requested_bin=requested_bin)
        views = [view for _route, batch in seen for view in batch]
        assert {r for r, _batch in seen} == {route}
        assert len(views) == len(PANEL)
        expected = Condition(family="affinity", bin=requested_bin or BIN_HIGH,
                             assay=harness.config["her2"]["assay"])
        assert {view[2] for view in views} == {expected}


# ---------------------------------------------------------------------------
# FlightServices(source_cache=...)
# ---------------------------------------------------------------------------

def test_the_source_cache_defaults_to_the_shared_artifact_directory(harness):
    assert harness.services.source_cache_root() == harness.context.artifact("cache/sources")


def test_a_source_cache_override_redirects_every_reader(harness, tmp_path):
    scratch = tmp_path / "scratch_cache"
    redirected = services_mod.FlightServices(harness.context, device="cpu", source_cache=scratch)
    assert redirected.source_cache_root() == Path(scratch)
    assert harness.services.source_cache_root() == harness.context.artifact("cache/sources")


def test_the_cache_location_has_one_definition_not_one_per_reader():
    """Both readers (``build_sources`` and ``paired_oas``) route through the helper.

    A literal copy of the path at a call site is how a read-only caller would redirect one
    reader and still write through the other, so the literal may appear once, in the helper.
    """
    source = inspect.getsource(services_mod)
    assert source.count('"cache/sources"') == 1
    assert "source_cache_root()" in inspect.getsource(services_mod.FlightServices.build_sources)
    assert "source_cache_root()" in inspect.getsource(services_mod.FlightServices.paired_oas)


def test_the_default_construction_signature_is_unchanged_for_existing_callers():
    parameters = inspect.signature(services_mod.FlightServices.__init__).parameters
    assert list(parameters)[:3] == ["self", "context", "device"]
    assert parameters["device"].default == "cuda"
    assert parameters["source_cache"].default is None
    assert parameters["source_cache"].kind is inspect.Parameter.KEYWORD_ONLY
    score = inspect.signature(services_mod.FlightServices.score_panel).parameters
    assert score["requested_bin"].default is None
    assert score["requested_bin"].kind is inspect.Parameter.KEYWORD_ONLY
