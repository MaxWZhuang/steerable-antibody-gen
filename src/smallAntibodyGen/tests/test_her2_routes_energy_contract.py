"""The energy route's version-2 contract, resume identity, runtime mode and route capabilities.

What is real here: ``FlightServices.train_energy`` (its sampler, both phases, optimizer, ledger,
checkpoints and resume), ``load_for_scoring``, ``score_panel``, ``energy_panel_scores`` and
``generate.energy_generate`` over a real, tiny ``WalkJumpModel`` on CPU, with a synthetic cohort of
``ConstructSummary`` rows installed in the services' eligible-summary cache. What is replaced:
only that cache and the bin mapping (the real sources are neither in the repository nor read by a
test), and, in the refusal tests, spies that fail the test if a loader, optimizer, directory or
state is touched before a refusal.

Nothing here says anything about a trained energy model's quality or about GPU behaviour. It
checks what the contract binds, what a resume refuses, that the null arm is never given a context,
and that the generic token-likelihood operations refuse the energy route early.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import shutil
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from smallAntibodyGen.experiments import her2_routes_capabilities as capabilities
from smallAntibodyGen.experiments import her2_routes_checkpoints as checkpoints
from smallAntibodyGen.experiments import her2_routes_data as data_lib
from smallAntibodyGen.experiments import her2_routes_energy_contract as energy_contract
from smallAntibodyGen.experiments import her2_routes_generate as generate
from smallAntibodyGen.experiments import her2_routes_lineage as lineage
from smallAntibodyGen.experiments import her2_routes_serial as serial
from smallAntibodyGen.experiments import her2_routes_services as services_mod
from smallAntibodyGen.experiments import her2_routes_spec as spec
from smallAntibodyGen.experiments import her2_support_paths as paths
from smallAntibodyGen.experiments.her2_data import CANONICAL
from smallAntibodyGen.models import her2_routes_energy as energy_lib
from smallAntibodyGen.tests.test_her2_routes_dplm_services import (  # noqa: F401  (helpers)
    HEAVY,
    LIGHT,
    Context,
    fake_scaffold,
    make_config,
)

ROUTE = services_mod.ROUTE_ENERGY
ALL = energy_contract.POPULATION_ALL_ELIGIBLE
HER2_ONLY = energy_contract.POPULATION_HER2_ONLY
HER2_REGION = tuple((serial.HEAVY, index) for index in (3, 4, 5))
OTHER_REGION = tuple((serial.HEAVY, index) for index in (10, 11, 12))
PANEL = [{"window": "WGGDGFYAMD", "value": 1.0}, {"window": "WGGDGFYAMA", "value": 2.0},
         {"window": "WAGDGFYAMD", "value": 3.0}]

#: Tiny and fast: three edit sites, two walk steps, three updates per phase.
ENERGY_BLOCK = {"window": 3, "sigma": 0.5, "scale": 1.0, "reg_l2_norm": 0.1, "context_width": 16,
                "microbatch": 2, "walk_steps": 2, "denoiser_updates": 3, "energy_updates": 3,
                "seed": 11, "profile_walk_steps": [2]}


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def summary(variant, *, source, region, study, assay="a1", target="T",
            partition=data_lib.PARTITION_TRAIN):
    heavy = list(HEAVY)
    for offset, (_chain, index) in enumerate(region):
        heavy[index] = CANONICAL[(variant + 3 * offset) % len(CANONICAL)]
    construct = serial.Construct(heavy="".join(heavy), light=LIGHT, sites=region, source=source,
                                 record_id=f"{source}{variant}")
    return data_lib.ConstructSummary(
        construct=construct, study=study, assay=assay, target=target, parent=f"{study}::parent",
        family="affinity", kind="point", measurements=(), region=region, partition=partition,
        scale="pKD", source=source)


def make_rows():
    her2 = [summary(variant, source=energy_contract.HER2_SOURCE, region=HER2_REGION,
                    study="alphabind_her2") for variant in range(6)]
    other = [summary(variant, source="other_source", region=OTHER_REGION, study="other_study")
             for variant in range(6, 12)]
    held_out = summary(12, source=energy_contract.HER2_SOURCE, region=HER2_REGION,
                       study="alphabind_her2", partition=data_lib.PARTITION_DEVELOPMENT)
    return her2 + other + [held_out]


@pytest.fixture
def energy(tmp_path):
    config = make_config()
    config["conditions"]["bin_dropout"] = 0.5
    config["routes"]["enabled"] = [ROUTE]
    config["routes"][ROUTE] = dict(ENERGY_BLOCK)
    context = Context(tmp_path, config)
    services = services_mod.FlightServices(context, device="cpu")
    services._cache["scaffold"] = fake_scaffold()
    rows = make_rows()
    services._cache["eligible"] = tuple(rows)
    services._cache["bins"] = {row.population_key: ("high" if index % 2 else "low")
                               for index, row in enumerate(rows)}
    return types.SimpleNamespace(services=services, context=context, config=config,
                                 root=tmp_path, rows=rows)


def train(h, name, *, conditioned, population=ALL, resume=False, seed=11, **kwargs):
    return h.services.train_energy(directory=h.root / name, seed=seed, population=population,
                                   conditioned=conditioned, resume=resume, **kwargs)


def snapshot(directory):
    directory = Path(directory)
    if not directory.exists():
        return None
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.iterdir())}


def tamper(path, mutate):
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    mutate(payload)
    torch.save(payload, str(path))


def instrument(monkeypatch, services):
    """Count every way a context could be built or a context tower run."""
    counts = {"energy_context": 0, "build_energy_context": 0, "context_tower": 0}
    original_context = services.energy_context

    def counted_context(*args, **kwargs):
        counts["energy_context"] += 1
        return original_context(*args, **kwargs)
    monkeypatch.setattr(services, "energy_context", counted_context)
    original_build = energy_lib.build_energy_context

    def counted_build(*args, **kwargs):
        counts["build_energy_context"] += 1
        return original_build(*args, **kwargs)
    monkeypatch.setattr(energy_lib, "build_energy_context", counted_build)
    original_forward = energy_lib.ContextEncoder.forward

    def counted_forward(self, context):
        counts["context_tower"] += 1
        return original_forward(self, context)
    monkeypatch.setattr(energy_lib.ContextEncoder, "forward", counted_forward)
    return counts


def capture_draws(monkeypatch):
    """Every trace entry, in order, as the sampler produced it."""
    log = []
    original = energy_contract.ExposureTrace.record

    def recording(self, **kwargs):
        log.append({key: kwargs[key] for key in ("phase", "step", "family", "group", "tile",
                                                 "sites", "members", "identities")})
        return original(self, **kwargs)
    monkeypatch.setattr(energy_contract.ExposureTrace, "record", recording)
    return log


def forbid_mutation(monkeypatch, services):
    """Fail the test if anything is built, restored, created or optimised before a refusal."""
    def refuse(name):
        def refused(*args, **kwargs):
            pytest.fail(f"{name} ran before the request was refused")
        return refused
    monkeypatch.setattr(services, "build_route", refuse("build_route"))
    monkeypatch.setattr(services, "build_heads", refuse("build_heads"))
    monkeypatch.setattr(services, "fresh_state", refuse("fresh_state"))
    monkeypatch.setattr(services, "energy_generator", refuse("energy_generator"))
    monkeypatch.setattr(services, "load_for_scoring", refuse("load_for_scoring"))
    monkeypatch.setattr(checkpoints, "restore", refuse("checkpoints.restore"))
    monkeypatch.setattr(torch.optim, "AdamW", refuse("torch.optim.AdamW"))
    monkeypatch.setattr(Path, "mkdir", refuse("Path.mkdir"))


# ---------------------------------------------------------------------------
# the pure contract, trace and runtime helpers
# ---------------------------------------------------------------------------

def fake_row(identity, *, source="other_source", assay="a1", target="T", region=None):
    region = region or ((serial.HEAVY, 1), (serial.HEAVY, 2))
    return types.SimpleNamespace(identity=identity, source=source, study="s", assay=assay,
                                 target=target, parent="p", family="affinity",
                                 partition="train", region=region,
                                 population_key=(identity, "s", assay, target))


def fake_rows():
    return [fake_row("a|x"), fake_row("b|x"), fake_row("c|x"),
            fake_row("d|x", source=energy_contract.HER2_SOURCE),
            fake_row("e|x", source=energy_contract.HER2_SOURCE, region=((serial.HEAVY, 5),))]


SPEC_DOCUMENT = {"families": ["affinity"], "bins": ["low", "high", "null"],
                 "assays": ["a1", "null"], "bin_dropout": 0.1, "assay_dropout": 0.2}


WALK_STEPS = 2


def make_contract(arm=energy_contract.ARM_FULL, *, population=ALL, microbatch=2, rows=None,
                  budget=None, max_query_sites=48):
    """A contract with every field the production one records that the exposure arithmetic reads:
    the resolved microbatch, the tile rule, the cohort's region lengths, the walk and its ledger."""
    groups = energy_contract.canonical_groups(rows if rows is not None else fake_rows())
    assignment = (None if arm == energy_contract.ARM_NULL else
                  energy_contract.condition_assignment_document(
                      groups, {}, default_bin="null", spec_document=SPEC_DOCUMENT))
    return energy_contract.build_contract(
        arm=arm, population=population, cohort=energy_contract.cohort_document(groups),
        sampling={"seed": 1, "microbatch": microbatch,
                  "tile_rule": {"max_query_sites": max_query_sites}},
        initialization={"seed": 1},
        phases={"budget": dict(budget or {"denoiser": 3, "energy": 3}),
                "walk": {"steps": WALK_STEPS}},
        forward_ledger=energy_contract.forward_ledger_document(WALK_STEPS),
        condition_assignment=assignment)


def build_progress(contract, steps, *, sites=((serial.HEAVY, 1),), shift=0):
    """The trace and the exposure ledger of ``steps`` = ``{phase: count}``, recorded the way the
    sampler records them: ``microbatch`` rows per step, one tile of ``sites``, and a ledger entry
    of ``forwards_per_row`` forwards and ``len(sites)`` targets per row. ``shift`` changes WHICH
    rows are drawn, so two otherwise identical runs can hold different draws."""
    microbatch = contract["sampling"]["microbatch"]
    pool = ("a|x", "b|x", "c|x")
    trace, ledger = energy_contract.ExposureTrace(), data_lib.ExposureLedger()
    for name in energy_contract.PHASES:
        for number in range(1, steps.get(name, 0) + 1):
            members = [index + shift for index in range(microbatch)]
            identities = [pool[member % len(pool)] for member in members]
            trace.record(phase=name, step=number, family="other", group=0, tile=0, sites=sites,
                         members=members, identities=identities)
            ledger.record(
                task=data_lib.TASK_WHOLE, update=True,
                contributions=data_lib.contributions_by_source(
                    sources=["other_source"] * microbatch, identities=identities,
                    forwards=energy_contract.forwards_per_row(name, WALK_STEPS),
                    targets=len(sites), measured=True, family=data_lib.FAMILY_OTHER))
    return trace, ledger


def synthetic_payload(contract, *, phase="denoiser", step=2, steps=None):
    budget = contract["phases"]["budget"]
    trace, ledger = build_progress(contract, steps or {
        energy_contract.PHASE_DENOISER: step if phase == "denoiser" else budget["denoiser"],
        energy_contract.PHASE_ENERGY: 0 if phase == "denoiser" else step})
    return {"identity": {"route": ROUTE, "arm": contract["arm"],
                         "schedule_sha256": contract["energy_contract_sha256"]},
            "energy_contract": copy_of(contract), "energy_phase": phase, "step": step,
            "phase_budget": dict(budget), "optimizer": {"state": {}},
            "energy_trace": trace.state(), "exposure": ledger.state()}


def rehash(payload):
    """Re-derive every hash of a payload's contract and the identity's, as a forger would."""
    contract = payload["energy_contract"]
    contract["common_sha256"] = energy_contract.common_digest(contract)
    contract["energy_contract_sha256"] = energy_contract.contract_digest(contract)
    payload["identity"]["schedule_sha256"] = contract["energy_contract_sha256"]
    return payload


def copy_of(document):
    return json.loads(json.dumps(document))


def test_the_arms_share_a_common_digest_and_differ_in_the_contract_hash():
    full, null = make_contract(energy_contract.ARM_FULL), make_contract(energy_contract.ARM_NULL)
    assert full["common_sha256"] == null["common_sha256"], "only the conditioning differs"
    assert full["energy_contract_sha256"] != null["energy_contract_sha256"]
    assert full["conditioning"]["mode"] == "conditioned" and null["conditioning"]["mode"] == "null"
    assert null["conditioning"]["condition_assignment"] is None
    assert null["conditioning"]["requested_quality_bin"] == "not applicable"
    for contract in (full, null):
        assert energy_contract.contract_digest(contract) == contract["energy_contract_sha256"]
        assert energy_contract.common_digest(contract) == contract["common_sha256"]
        assert contract == json.loads(json.dumps(contract)), "plain JSON, never tuples"
    assert make_contract(microbatch=3)["common_sha256"] != full["common_sha256"]
    assert make_contract(budget={"denoiser": 4, "energy": 3})["common_sha256"] != (
        full["common_sha256"])
    assert make_contract(population=HER2_ONLY)["common_sha256"] != full["common_sha256"]


def test_the_conditioning_assignment_is_required_by_the_full_arm_and_forbidden_for_the_null():
    groups = energy_contract.canonical_groups(fake_rows())
    cohort = energy_contract.cohort_document(groups)
    common = dict(population=ALL, cohort=cohort, sampling={}, initialization={},
                  phases={"budget": {"denoiser": 1, "energy": 1}},
                  forward_ledger=energy_contract.forward_ledger_document(1))
    with pytest.raises(ValueError, match="needs the condition assignment"):
        energy_contract.build_contract(arm=energy_contract.ARM_FULL, **common)
    with pytest.raises(ValueError, match="no condition assignment to bind"):
        energy_contract.build_contract(arm=energy_contract.ARM_NULL, condition_assignment={},
                                       **common)
    with pytest.raises(ValueError, match="Unknown energy arm"):
        energy_contract.build_contract(arm="density_only", **common)


def test_the_cohort_digest_binds_the_content_not_a_record_id_or_the_enumeration_order():
    base = energy_contract.cohort_document(energy_contract.canonical_groups(fake_rows()))
    shuffled = energy_contract.cohort_document(
        energy_contract.canonical_groups(list(reversed(fake_rows()))))
    assert shuffled["sha256"] == base["sha256"], "the same set of rows is the same cohort"
    for change in ({"target": "other"}, {"source": "elsewhere"}, {"assay": "a2"},
                   {"region": ((serial.HEAVY, 9), (serial.HEAVY, 10))}):
        rows = fake_rows()
        rows[0] = fake_row("a|x", **{key: value for key, value in change.items()})
        changed = energy_contract.cohort_document(energy_contract.canonical_groups(rows))
        assert changed["sha256"] != base["sha256"], change
    rows = fake_rows()
    rows[0] = fake_row("a|DIFFERENT-CONSTRUCT")
    assert energy_contract.cohort_document(
        energy_contract.canonical_groups(rows))["sha256"] != base["sha256"]
    assert base["rows"] == 5 and base["families"] == {
        "her2": {"groups": 2, "members": 2}, "other": {"groups": 1, "members": 3}}


def test_the_condition_assignment_binds_the_bin_each_member_is_built_from():
    groups = energy_contract.canonical_groups(fake_rows())
    plain = energy_contract.condition_assignment_document(
        groups, {}, default_bin="null", spec_document=SPEC_DOCUMENT)
    binned = energy_contract.condition_assignment_document(
        groups, {fake_rows()[0].population_key: "high"}, default_bin="null",
        spec_document=SPEC_DOCUMENT)
    assert plain["sha256"] != binned["sha256"]
    assert plain["bins"] == {"null": 5} and binned["bins"] == {"high": 1, "null": 4}
    assert binned["dropout"] == {"bin": 0.1, "assay": 0.2}


def test_only_a_family_that_exists_is_drawn_and_the_law_is_recorded():
    import numpy as np
    both = np.random.default_rng(3)
    reference = np.random.default_rng(3)
    families = [energy_contract.draw_family(("her2", "other"), both) for _ in range(300)]
    expected = ["her2" if reference.random() < 2 / 3 else "other" for _ in range(300)]
    assert families == expected, "the declared 2/3 law, one draw per step, exactly as before"
    assert {"her2", "other"} == set(families)
    single, untouched = np.random.default_rng(3), np.random.default_rng(3)
    assert energy_contract.draw_family(("other",), single) == "other"
    assert energy_contract.draw_family(("her2",), single) == "her2"
    assert single.bit_generator.state == untouched.bit_generator.state, (
        "a lone family is chosen without consuming a draw")
    law = energy_contract.family_law(["her2", "other"])
    assert law["her2_probability"] == pytest.approx(2 / 3) and law["draw_consumed_per_step"]
    assert energy_contract.family_law(["other"])["draw_consumed_per_step"] is False
    with pytest.raises(ValueError, match="no family"):
        energy_contract.draw_family((), np.random.default_rng(0))


def test_forwards_are_row_forwards_one_for_the_denoiser_and_walk_plus_two_for_the_energy():
    assert energy_contract.forwards_per_row("denoiser", 7) == 1
    assert energy_contract.forwards_per_row("energy", 7) == 9
    ledger = energy_contract.forward_ledger_document(7)
    assert ledger["energy_phase_per_row"] == 9 and ledger["denoiser_phase_per_row"] == 1
    assert "ROW-FORWARDS" in ledger["unit"] and "Not batched calls" in ledger["unit"]
    with pytest.raises(ValueError, match="Unknown energy phase"):
        energy_contract.forwards_per_row("both", 7)


def test_the_trace_is_a_cumulative_ordered_chain_that_state_restores_exactly():
    def run(draws):
        trace = energy_contract.ExposureTrace()
        for number, (members, identities) in enumerate(draws, start=1):
            trace.record(phase="denoiser", step=number, family="her2", group=0, tile=0,
                         sites=[(serial.HEAVY, 1)], members=members, identities=identities)
        return trace
    a = run([([0], ["x"]), ([1], ["y"])])
    assert run([([0], ["x"]), ([1], ["y"])]).state() == a.state()
    assert run([([1], ["y"]), ([0], ["x"])]).state()["chain_sha256"] != a.state()["chain_sha256"], (
        "the ORDER is part of the evidence")
    assert run([([0], ["x"]), ([1], ["z"])]).state()["chain_sha256"] != a.state()["chain_sha256"]
    assert a.phases["denoiser"] == {"steps": 2, "rows": 2, "site_exposures": 2}
    restored = energy_contract.ExposureTrace.from_state(copy_of(a.state()))
    assert restored.state() == a.state()
    a.record(phase="denoiser", step=3, family="her2", group=0, tile=0,
             sites=[(serial.HEAVY, 1)], members=[0], identities=["x"])
    restored.record(phase="denoiser", step=3, family="her2", group=0, tile=0,
                    sites=[(serial.HEAVY, 1)], members=[0], identities=["x"])
    assert restored.state() == a.state(), "a restored trace continues; it does not reset"
    with pytest.raises(ValueError, match="neither reset nor skipped"):
        restored.record(phase="denoiser", step=9, family="her2", group=0, tile=0,
                        sites=[(serial.HEAVY, 1)], members=[0], identities=["x"])
    document = a.document()
    assert document["total"] == {"steps": 3, "rows": 3, "site_exposures": 3}


@pytest.mark.parametrize("damage", [
    lambda s: s.update(schema="other"), lambda s: s.update(chain_sha256="zz"),
    lambda s: s["phases"].pop("energy"), lambda s: s["phases"]["energy"].update(steps=-1),
    lambda s: s["phases"]["energy"].update(steps=True), lambda s: s["phases"]["energy"].pop("rows")])
def test_a_malformed_trace_state_is_refused(damage):
    state = energy_contract.ExposureTrace().state()
    damage(state)
    with pytest.raises(ValueError, match="exposure trace|trace count|trace counts"):
        energy_contract.ExposureTrace.from_state(state)


def test_the_legacy_checkpoint_is_refused_with_an_incompatibility_error():
    contract = make_contract()
    payload = synthetic_payload(contract)
    payload.pop("energy_contract")
    payload["identity"]["schedule_sha256"] = None
    for refuse in (lambda: energy_contract.require_resumable(payload, contract=contract,
                                                              path="legacy.pt"),
                   lambda: energy_contract.inference_binding(payload, path="legacy.pt")):
        with pytest.raises(energy_contract.EnergyCheckpointIncompatible,
                           match="cannot be resumed, scored or relabelled") as caught:
            refuse()
        assert isinstance(caught.value, lineage.LineageError), (
            "a campaign stage records blocked_readiness for it instead of crashing")
        assert isinstance(caught.value, ValueError)
    assert "version 2" in energy_contract.LEGACY_REFUSAL
    assert "preserved untouched" in energy_contract.LEGACY_REFUSAL


def test_a_valid_state_is_resumable_and_a_completed_phase_is_a_legitimate_transition():
    contract = make_contract()
    middle = energy_contract.require_resumable(synthetic_payload(contract, step=2),
                                               contract=contract, path="m.pt")
    assert (middle["phase"], middle["step"], middle["completed_phase"]) == ("denoiser", 2, False)
    boundary = energy_contract.require_resumable(synthetic_payload(contract, step=3),
                                                 contract=contract, path="b.pt")
    assert boundary["completed_phase"] is True and boundary["phase"] == "denoiser"
    energy_middle = energy_contract.require_resumable(
        synthetic_payload(contract, phase="energy", step=1), contract=contract, path="e.pt")
    assert (energy_middle["phase"], energy_middle["step"]) == ("energy", 1)
    assert energy_middle["trace"].phases["denoiser"]["steps"] == 3
    done = energy_contract.require_resumable(
        synthetic_payload(contract, phase="energy", step=3), contract=contract, path="d.pt")
    assert done["completed_phase"] is True


@pytest.mark.parametrize("mutate,phrase", [
    (lambda p: p.update(energy_phase=""), "records energy_phase"),
    (lambda p: p.pop("energy_phase"), "records energy_phase"),
    (lambda p: p.update(energy_phase="both"), "records energy_phase"),
    (lambda p: p.update(step=4), "against a denoiser budget of 3"),
    (lambda p: p.update(step=-1), "against a denoiser budget of 3"),
    (lambda p: p.update(step=True), "against a denoiser budget of 3"),
    (lambda p: p.update(step=None), "against a denoiser budget of 3"),
    (lambda p: p.update(phase_budget={"denoiser": 3, "energy": 4}), "different experiment"),
    (lambda p: p.pop("phase_budget"), "different experiment"),
    (lambda p: p.update(optimizer=None), "no optimizer state"),
    (lambda p: p.pop("optimizer"), "no optimizer state"),
    (lambda p: p.pop("energy_trace"), "no energy_trace"),
    (lambda p: p["energy_trace"]["phases"]["energy"].update(steps=1),
     "the trace and the weights disagree")])
def test_a_resume_needs_a_recognised_phase_a_bounded_step_exact_budgets_and_state(mutate, phrase):
    contract = make_contract()
    payload = synthetic_payload(contract)
    mutate(payload)
    with pytest.raises(ValueError, match=phrase):
        energy_contract.require_resumable(payload, contract=contract, path="x.pt")


def test_a_tampered_contract_mode_or_digest_is_refused_by_recomputation():
    contract = make_contract()
    edited = synthetic_payload(contract)
    edited["energy_contract"]["conditioning"]["mode"] = "null"
    with pytest.raises(ValueError, match="does not hash"):
        energy_contract.stored_contract(edited, path="x.pt")
    swapped = synthetic_payload(contract)
    swapped["identity"]["schedule_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="not the identity's schedule_sha256"):
        energy_contract.stored_contract(swapped, path="x.pt")
    # A forgery that re-hashes the document AND the identity still cannot make a full arm null.
    forged = synthetic_payload(contract)
    forged["energy_contract"]["conditioning"]["mode"] = "null"
    forged["energy_contract"]["energy_contract_sha256"] = energy_contract.contract_digest(
        forged["energy_contract"])
    forged["identity"]["schedule_sha256"] = forged["energy_contract"]["energy_contract_sha256"]
    with pytest.raises(ValueError, match="inconsistent"):
        energy_contract.stored_contract(forged, path="x.pt")
    # ... nor can it relabel the arm in the identity alone.
    relabelled = synthetic_payload(contract)
    relabelled["identity"]["arm"] = "null"
    with pytest.raises(ValueError, match="identity's arm"):
        energy_contract.stored_contract(relabelled, path="x.pt")
    # ... nor edit the common fields and leave the common digest behind.
    moved = synthetic_payload(contract)
    moved["energy_contract"]["sampling"]["microbatch"] = 99
    moved["energy_contract"]["energy_contract_sha256"] = energy_contract.contract_digest(
        moved["energy_contract"])
    moved["identity"]["schedule_sha256"] = moved["energy_contract"]["energy_contract_sha256"]
    with pytest.raises(ValueError, match="common digest"):
        energy_contract.stored_contract(moved, path="x.pt")
    assert energy_contract.stored_contract(synthetic_payload(contract), path="ok.pt") == contract


def test_a_changed_request_names_what_changed():
    stored = make_contract()
    payload = synthetic_payload(stored)
    for requested, fragment in (
            (make_contract(microbatch=3), "sampling.microbatch"),
            (make_contract(budget={"denoiser": 4, "energy": 3}), "phases.budget.denoiser"),
            (make_contract(energy_contract.ARM_NULL), "conditioning.mode"),
            (make_contract(population=HER2_ONLY), "population.name")):
        with pytest.raises(ValueError, match="different energy contract") as caught:
            energy_contract.require_resumable(payload, contract=requested, path="x.pt")
        assert fragment in str(caught.value)


def test_inference_binding_validates_the_stored_contract_and_the_requested_arm():
    contract = make_contract()
    payload = synthetic_payload(contract, phase="energy", step=3)
    stored, terminal = energy_contract.inference_binding(payload, path="best.pt")
    assert stored["arm"] == "full" and terminal is True
    _stored, terminal = energy_contract.inference_binding(synthetic_payload(contract, step=2),
                                                          path="last.pt")
    assert terminal is False
    energy_contract.inference_binding(payload, path="best.pt", arm="full")
    with pytest.raises(ValueError, match="'null' arm was requested"):
        energy_contract.inference_binding(payload, path="best.pt", arm="null")
    payload["energy_phase"] = "bogus"
    with pytest.raises(ValueError, match="records energy_phase"):
        energy_contract.inference_binding(payload, path="best.pt")


def test_the_tile_and_microbatch_bounds_are_bound_in_the_contract_digest():
    assert energy_contract.tile_lengths(7, 3) == [1, 3]
    assert energy_contract.tile_lengths(6, 3) == [3]
    assert energy_contract.tile_lengths(3, 48) == [3]
    assert energy_contract.tile_lengths(5, 0) == [5], "a non-positive tile size leaves it whole"
    cohort = energy_contract.cohort_document(energy_contract.canonical_groups(fake_rows()))
    assert cohort["region_lengths"] == [1, 2]
    bounds = energy_contract.exposure_bounds(make_contract(), where="test")
    assert bounds["microbatch"] == 2 and bounds["tiles"] == [1, 2]
    assert bounds["budget"] == {"denoiser": 3, "energy": 3}
    assert bounds["forwards_per_row"] == {"denoiser": 1, "energy": WALK_STEPS + 2}
    assert energy_contract.exposure_bounds(make_contract(max_query_sites=1),
                                           where="test")["tiles"] == [1]
    base = make_contract()["common_sha256"]
    assert make_contract(max_query_sites=1)["common_sha256"] != base
    assert make_contract(rows=[fake_row("a|x"), fake_row("b|x")])["common_sha256"] != base
    narrow = make_contract(max_query_sites=1)
    wide, ledger = build_progress(narrow, {"denoiser": 2},
                                  sites=((serial.HEAVY, 1), (serial.HEAVY, 2)))
    payload = synthetic_payload(narrow)
    payload["energy_trace"], payload["exposure"] = wide.state(), ledger.state()
    with pytest.raises(ValueError, match="site exposures"):
        energy_contract.require_resumable(payload, contract=narrow, path="x.pt")


def _edit_phase(phase, **changes):
    def edit(payload):
        payload["energy_trace"]["phases"][phase].update(changes)
    return edit


def _totals(**changes):
    def edit(payload):
        payload["exposure"]["totals"].update(changes)
    return edit


def _shift_ledger(key):
    """Move one counter by one in the totals AND in every bucket kind, so the ledger stays
    additive and internally consistent while no longer agreeing with the trace."""
    def edit(payload):
        ledger = payload["exposure"]
        ledger["totals"][key] += 1
        if key != "optimizer_updates":
            for kind in ("by_task", "by_source", "by_family"):
                next(iter(ledger[kind].values()))[key] += 1
    return edit


def _break_additivity(payload):
    next(iter(payload["exposure"]["by_source"].values()))["microbatch_rows"] += 1


def _corrupt_identity_table(payload):
    payload["exposure"]["identity_table"] = []


PAYLOAD_DAMAGE = [
    ("zero_rows_and_sites_for_positive_steps", _edit_phase("denoiser", rows=0, site_exposures=0),
     "makes 4"),
    ("rows_off_by_one", _edit_phase("denoiser", rows=3), "makes 4"),
    ("rows_doubled", _edit_phase("denoiser", rows=8), "makes 4"),
    ("no_sites_for_positive_steps", _edit_phase("denoiser", site_exposures=0), "site exposures"),
    ("sites_not_a_multiple_of_the_microbatch", _edit_phase("denoiser", site_exposures=5),
     "site exposures"),
    ("more_sites_than_the_largest_tile_allows", _edit_phase("denoiser", site_exposures=10),
     "site exposures"),
    ("rows_in_a_zero_step_phase", _edit_phase("energy", rows=2), "makes 0"),
    ("sites_in_a_zero_step_phase", _edit_phase("energy", site_exposures=2), "site exposures"),
    ("no_ledger", lambda p: p.pop("exposure"), "no exposure ledger"),
    ("ledger_not_a_mapping", lambda p: p.update(exposure="ledger"), "no exposure ledger"),
    ("ledger_rows", _shift_ledger("microbatch_rows"), "exposure ledger records"),
    ("ledger_forwards", _shift_ledger("student_forwards"), "exposure ledger records"),
    ("ledger_targets", _shift_ledger("supervised_targets"), "exposure ledger records"),
    ("ledger_updates", _shift_ledger("optimizer_updates"), "exposure ledger records"),
    ("ledger_totals_alone_edited", _totals(microbatch_rows=5), "do not sum to its totals"),
    ("ledger_not_additive", _break_additivity, "do not sum to its totals"),
    ("ledger_identity_table_corrupt", _corrupt_identity_table, "does not restore"),
    ("ledger_unknown_version", lambda p: p["exposure"].update(version=99), "does not restore")]


@pytest.mark.parametrize("name,mutate,phrase", PAYLOAD_DAMAGE, ids=[row[0] for row in PAYLOAD_DAMAGE])
def test_a_trace_or_ledger_that_does_not_follow_from_the_contract_is_refused(name, mutate, phrase):
    contract = make_contract()
    payload = synthetic_payload(contract)             # denoiser step 2: 4 rows, 4 site exposures
    mutate(payload)
    for refuse in (lambda: energy_contract.require_resumable(payload, contract=contract,
                                                              path="x.pt"),
                   lambda: energy_contract.inference_binding(payload, path="x.pt")):
        with pytest.raises(ValueError, match=phrase):
            refuse()


def test_the_unmodified_fixture_is_exactly_what_those_damaged_copies_depart_from():
    contract = make_contract()
    payload = synthetic_payload(contract)
    assert payload["energy_trace"]["phases"]["denoiser"] == {"steps": 2, "rows": 4,
                                                             "site_exposures": 4}
    totals = payload["exposure"]["totals"]
    assert (totals["optimizer_updates"], totals["microbatch_rows"], totals["supervised_targets"],
            totals["student_forwards"]) == (2, 4, 4, 4), "the ledger agrees with the trace"
    through_energy = synthetic_payload(contract, phase="energy", step=1)
    assert through_energy["exposure"]["totals"]["student_forwards"] == (
        3 * 2 * 1 + 1 * 2 * (WALK_STEPS + 2)), "denoiser rows x1 + energy rows x (walk + 2)"
    for payload in (payload, through_energy):
        energy_contract.inference_binding(payload, path="x.pt")


def test_validating_a_checkpoint_never_changes_the_payload_it_reads():
    contract = make_contract()
    for payload in (synthetic_payload(contract, step=3),
                    synthetic_payload(contract, phase="energy", step=2)):
        before = copy.deepcopy(payload)
        energy_contract.require_resumable(payload, contract=contract, path="x.pt")
        energy_contract.inference_binding(payload, path="x.pt")
        assert payload == before, "the ledger restore and the trace are read, not written"


@pytest.mark.parametrize("strip,phrase", [
    (lambda c: c["population"]["cohort"].pop("region_lengths"), "records no"),
    (lambda c: c["sampling"].pop("tile_rule"), "records no"),
    (lambda c: c["sampling"].update(microbatch=0), "not a positive integer"),
    (lambda c: c["sampling"].update(microbatch=True), "not a positive integer"),
    (lambda c: c["forward_ledger"].pop("energy_phase_per_row"), "records no"),
    (lambda c: c["phases"]["walk"].update(steps=5), "not the row-forwards")])
def test_a_contract_that_records_no_usable_bounds_cannot_vouch_for_a_trace(strip, phrase):
    contract = make_contract()
    payload = synthetic_payload(contract)
    strip(payload["energy_contract"])
    rehash(payload)           # every hash is true, so only the bounds themselves are in question
    forged = payload["energy_contract"]
    energy_contract.stored_contract(payload, path="x.pt")
    with pytest.raises(ValueError, match=phrase):
        energy_contract.require_resumable(payload, contract=forged, path="x.pt")
    with pytest.raises(ValueError, match=phrase):
        energy_contract.inference_binding(payload, path="x.pt")


def test_the_runtime_is_immutable_and_its_arm_and_mode_must_agree():
    runtime = energy_contract.checkpoint_free_runtime("null")
    assert (runtime.arm, runtime.mode, runtime.conditioned) == ("null", "null", False)
    assert runtime.initialization == "random_untrained" and runtime.source == "checkpoint_free"
    with pytest.raises(dataclasses.FrozenInstanceError):
        runtime.mode = "conditioned"
    with pytest.raises(ValueError, match="inconsistent"):
        energy_contract.EnergyRuntime(arm="full", mode="null", source="checkpoint",
                                      initialization="trained_checkpoint")
    with pytest.raises(ValueError, match="Unknown energy arm"):
        energy_contract.checkpoint_free_runtime("density_only")
    assert "not applicable" in runtime.document()["requested_quality_bin"]
    assert energy_contract.checkpoint_free_runtime("full").conditioned is True


def run_record(arm, steps=None, *, shift=0, **contract_options):
    """A run record as ``train_energy`` returns it: contract, trace, ledger document and identity."""
    contract = make_contract(arm, **contract_options)
    counts = dict(steps or {"denoiser": 3, "energy": 0})
    trace, ledger = build_progress(contract, counts, shift=shift)
    return {"energy_contract": contract, "exposure_trace": trace.document(),
            "phase_updates": {name: counts.get(name, 0) for name in energy_contract.PHASES},
            "exposure": ledger.document(), "arm": arm,
            "conditioned": arm == energy_contract.ARM_FULL,
            "energy_contract_sha256": contract["energy_contract_sha256"],
            "common_sha256": contract["common_sha256"],
            "identity": {"route": ROUTE, "arm": arm,
                         "schedule_sha256": contract["energy_contract_sha256"]}}


def test_the_matched_control_check_recomputes_both_contracts_and_compares_the_draws():
    matched = energy_contract.require_matched_controls(run_record("full"), run_record("null"))
    assert matched["common_contract_matched"] is True
    assert matched["same_draws"] is True and matched["exposure_matched"] is True
    assert matched["differs_only_in"] == "conditioning (context given or None)"
    assert matched["phase_updates"] == {"full": {"denoiser": 3, "energy": 0},
                                        "null": {"denoiser": 3, "energy": 0}}
    through_energy = {"denoiser": 3, "energy": 2}
    both = energy_contract.require_matched_controls(run_record("full", through_energy),
                                                    run_record("null", through_energy))
    assert both["same_draws"] is True and both["phase_updates"]["null"] == through_energy
    with pytest.raises(ValueError, match="one full and one null"):
        energy_contract.require_matched_controls(run_record("null"), run_record("null"))


def test_a_changed_cohort_under_the_old_hashes_is_not_a_matched_control():
    for victim in ("full", "null"):
        records = {"full": run_record("full"), "null": run_record("null")}
        records[victim]["energy_contract"]["population"]["cohort"]["sha256"] = "0" * 64
        with pytest.raises(ValueError, match="does not hash") as caught:
            energy_contract.require_matched_controls(records["full"], records["null"])
        assert f"the {victim} run record" in str(caught.value), "the message names the bad record"
    # Re-deriving the contract's own hash is not enough: the common digest still describes the
    # old fields, so a forgery has to leave it stale or recompute it too.
    stale_common = run_record("null")
    stale_common["energy_contract"]["population"]["cohort"]["sha256"] = "0" * 64
    stale_common["energy_contract"]["energy_contract_sha256"] = energy_contract.contract_digest(
        stale_common["energy_contract"])
    with pytest.raises(ValueError, match="common digest"):
        energy_contract.require_matched_controls(run_record("full"), stale_common)


def test_a_self_consistently_rehashed_but_different_common_contract_is_refused():
    other_rows = fake_rows()
    other_rows[0] = fake_row("a|x", target="a-different-target")
    different = run_record("null", rows=other_rows)
    # Every hash of it is true: it is its own, internally consistent, different experiment.
    energy_contract.validated_contract(different["energy_contract"], where="the forgery")
    assert different["common_sha256"] != run_record("full")["common_sha256"]
    with pytest.raises(ValueError, match="do not share a common contract") as caught:
        energy_contract.require_matched_controls(run_record("full"), different)
    assert "population.cohort.sha256" in str(caught.value)


def test_a_changed_trace_at_equal_progress_is_refused_not_reported_as_matched():
    forged = run_record("null")
    forged["exposure_trace"]["chain_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="different draws"):
        energy_contract.require_matched_controls(run_record("full"), forged)
    # A trace that is internally true -- real rows, real counts -- but of other rows.
    other_draws = run_record("null", shift=1)
    assert other_draws["exposure_trace"]["phases"] == run_record("full")["exposure_trace"]["phases"]
    with pytest.raises(ValueError, match="different draws"):
        energy_contract.require_matched_controls(run_record("full"), other_draws)


def test_unequal_progress_matches_the_common_contract_but_makes_no_exposure_claim():
    ahead = energy_contract.require_matched_controls(
        run_record("full", {"denoiser": 3, "energy": 0}),
        run_record("null", {"denoiser": 2, "energy": 0}))
    assert ahead["common_contract_matched"] is True
    assert ahead["same_draws"] is None and ahead["exposure_matched"] is None
    assert ahead["differs_only_in"] is None, "no unconditional only-conditioning claim"
    assert "unverified" in ahead["claim"]
    assert ahead["phase_updates"]["full"] != ahead["phase_updates"]["null"]


def _edit_trace(**changes):
    def edit(record):
        record["exposure_trace"]["phases"]["denoiser"].update(changes)
        record["exposure_trace"].pop("total")
    return edit


@pytest.mark.parametrize("mutate,phrase", [
    (lambda r: r["phase_updates"].update(denoiser=2), "records phase_updates"),
    (lambda r: r.pop("phase_updates"), "phase_updates must count"),
    (lambda r: r["phase_updates"].update(denoiser=True), "phase_updates must count"),
    (_edit_trace(rows=0, site_exposures=0), "makes 6"),     # positive steps, nothing presented
    (_edit_trace(rows=5), "makes 6"),
    (_edit_trace(site_exposures=0), "site exposures"),
    (_edit_trace(site_exposures=7), "site exposures"),      # not a multiple of the microbatch
    (_edit_trace(site_exposures=13), "site exposures"),     # more than the largest tile allows
    (lambda r: r["exposure_trace"]["total"].update(rows=1), "add up to"),
    (lambda r: r["exposure"]["totals"].update(microbatch_rows=5), "exposure ledger records"),
    (lambda r: r["exposure"]["totals"].update(student_forwards=1), "exposure ledger records"),
    (lambda r: r["exposure"]["totals"].update(supervised_targets=1), "exposure ledger records"),
    (lambda r: r["exposure"]["totals"].update(optimizer_updates=1), "exposure ledger records"),
    (lambda r: r["identity"].update(arm="full"), "identity has arm"),
    (lambda r: r["identity"].update(schedule_sha256="0" * 64), "identity has schedule_sha256"),
    (lambda r: r.update(energy_contract_sha256="0" * 64), "records energy_contract_sha256"),
    (lambda r: r.update(common_sha256="0" * 64), "records common_sha256"),
    (lambda r: r.update(conditioned=True), "records conditioned"),
    (lambda r: r.update(arm="full"), "records arm"),
    (lambda r: r.pop("exposure_trace"), "exposure trace"),
    (lambda r: r.pop("energy_contract"), "not a mapping")])
def test_a_run_record_is_held_to_its_own_contract_trace_and_ledger(mutate, phrase):
    damaged = run_record("null")
    mutate(damaged)
    with pytest.raises(ValueError, match=phrase):
        energy_contract.require_matched_controls(run_record("full"), damaged)


def test_a_run_record_cannot_overrun_its_budget_or_start_the_energy_phase_early():
    with pytest.raises(ValueError, match="against a budget of 3"):
        energy_contract.require_matched_controls(
            run_record("full"), run_record("null", {"denoiser": 4, "energy": 0}))
    with pytest.raises(ValueError, match="starts only after the denoiser phase is complete"):
        energy_contract.require_matched_controls(
            run_record("full", {"denoiser": 2, "energy": 1}),
            run_record("null", {"denoiser": 2, "energy": 1}))


def test_the_model_binds_a_runtime_once_and_its_architecture_cannot_stand_in_for_it():
    model = energy_lib.WalkJumpModel(window=3, context_width=16,
                                     condition_sizes={"family": 3, "bin": 4, "assay": 2})
    assert model.runtime is None
    with pytest.raises(ValueError, match="no validated runtime contract"):
        energy_contract.require_runtime(model, where="test")
    runtime = energy_contract.checkpoint_free_runtime("null")
    assert model.bind_runtime(runtime) is runtime and model.runtime is runtime
    model.bind_runtime(runtime)                           # the same binding again is a no-op
    with pytest.raises(ValueError, match="already bound"):
        model.bind_runtime(energy_contract.checkpoint_free_runtime("full"))
    with pytest.raises(AttributeError):
        model.runtime = None
    assert "runtime" not in model.state_dict()
    document = model.document()["context"]
    assert document["has_condition_tables"] is True and "how_trained" in document, (
        "the null arm owns condition tables, so the architecture says nothing about the arm")
    assert energy_contract.require_runtime(model, where="test", arm="null") is runtime
    with pytest.raises(ValueError, match="'full' arm was requested"):
        energy_contract.require_runtime(model, where="test", arm="full")
    with pytest.raises(ValueError, match="'conditioned' mode was requested"):
        energy_contract.require_runtime(model, where="test", mode="conditioned")
    assert energy_contract.require_runtime(model, where="test", mode="null") is runtime
    assert "context=None" in model.density_only_arm()["difference_from_full"]
    assert "NOT identical" in model.density_only_arm()["difference_from_full"]


# ---------------------------------------------------------------------------
# training: the same experiment except for the conditioning
# ---------------------------------------------------------------------------

def test_train_energy_takes_the_population_and_the_conditioning_as_separate_required_arguments(
        energy):
    directory = energy.root / "never"
    with pytest.raises(TypeError):
        energy.services.train_energy(directory=directory, seed=1)
    with pytest.raises(TypeError):
        energy.services.train_energy(directory=directory, seed=1, population=ALL)
    with pytest.raises(TypeError):
        energy.services.train_energy(directory=directory, seed=1, conditioned=True)
    for bad in (None, 1, "full", "null"):
        with pytest.raises(ValueError, match="True or False"):
            energy.services.train_energy(directory=directory, seed=1, population=ALL,
                                         conditioned=bad)
    for name in ("measured_extension", "density_only", ""):
        with pytest.raises(ValueError, match="Unknown energy training population"):
            energy.services.train_energy(directory=directory, seed=1, population=name,
                                         conditioned=True)
    assert not directory.exists(), "a refused request creates nothing"


def test_a_full_and_a_null_run_share_the_cohort_and_the_actual_ordered_draws(energy, monkeypatch):
    draws = capture_draws(monkeypatch)
    cleans = []
    for name in ("denoiser_loss", "energy_loss"):
        original = getattr(energy_lib.WalkJumpModel, name)

        def wrapper(self, clean, *args, _original=original, **kwargs):
            cleans.append(clean.detach().clone())
            return _original(self, clean, *args, **kwargs)
        monkeypatch.setattr(energy_lib.WalkJumpModel, name, wrapper)

    full = train(energy, "full", conditioned=True)
    full_draws, full_cleans = list(draws), list(cleans)
    draws.clear()
    cleans.clear()
    null = train(energy, "null", conditioned=False)

    assert len(full_draws) == 6 and full_draws == list(draws), (
        "the same family, group, tile, members, identities and sites at every step")
    assert len(full_cleans) == len(cleans) == 6
    assert all(torch.equal(left, right) for left, right in zip(full_cleans, cleans)), (
        "the clean windows the two arms were trained on are identical, step by step")
    assert full["exposure_trace"] == null["exposure_trace"]
    assert full["exposure_trace"]["total"] == {"steps": 6, "rows": 12, "site_exposures": 36}
    assert full["common_sha256"] == null["common_sha256"]
    assert full["energy_contract_sha256"] != null["energy_contract_sha256"]
    assert (full["energy_contract"]["population"] == null["energy_contract"]["population"])
    assert full["training_rows"] == null["training_rows"] == 12, (
        "the held-out development row is not in either cohort")
    assert full["exposure"]["totals"] == null["exposure"]["totals"]
    assert ({name: block["microbatch_rows"] for name, block in
             full["exposure"]["by_family"].items()}
            == {name: block["microbatch_rows"] for name, block in
                null["exposure"]["by_family"].items()})
    assert (full["identity"]["arm"], null["identity"]["arm"]) == ("full", "null")
    for record in (full, null):
        assert record["identity"]["schedule_sha256"] == record["energy_contract_sha256"]
        assert record["energy_contract"]["version"] == 2
    matched = energy_contract.require_matched_controls(full, null)
    assert matched["same_draws"] is True


def test_conditioning_dropout_consumes_the_mask_generator_and_never_the_samplers(energy):
    train(energy, "full", conditioned=True)
    train(energy, "null", conditioned=False)
    full = torch.load(str(energy.root / "full" / "last.pt"), map_location="cpu",
                      weights_only=False)["rng"]
    null = torch.load(str(energy.root / "null" / "last.pt"), map_location="cpu",
                      weights_only=False)["rng"]
    assert full.numpy["sampler_rng"] == null.numpy["sampler_rng"], (
        "the sampler consumed exactly the same draws in both arms")
    assert full.numpy["mask_rng"] != null.numpy["mask_rng"], (
        "the conditioned arm's dropout advanced the mask generator; the null arm's never did")


def test_the_population_is_independent_of_the_conditioning(energy):
    all_full = train(energy, "all_full", conditioned=True, population=ALL)
    her2_null = train(energy, "her2_null", conditioned=False, population=HER2_ONLY)
    all_null = train(energy, "all_null", conditioned=False, population=ALL)
    her2_full = train(energy, "her2_full", conditioned=True, population=HER2_ONLY)
    assert all_full["training_rows"] == all_null["training_rows"] == 12
    assert her2_null["training_rows"] == her2_full["training_rows"] == 6
    assert all_full["common_sha256"] == all_null["common_sha256"]
    assert her2_full["common_sha256"] == her2_null["common_sha256"]
    assert all_full["common_sha256"] != her2_full["common_sha256"]
    assert her2_full["energy_contract"]["sampling"]["family_law"]["present"] == ["her2"]
    assert all_full["energy_contract"]["sampling"]["family_law"]["present"] == ["her2", "other"]
    assert all_full["population"] == ALL and all_null["conditioned"] is False


def test_only_a_family_that_exists_is_drawn_so_an_other_only_cohort_trains(energy, monkeypatch):
    energy.services._cache["eligible"] = tuple(
        row for row in energy.rows if row.source != energy_contract.HER2_SOURCE)
    draws = capture_draws(monkeypatch)
    record = train(energy, "other_only", conditioned=True)
    assert {entry["family"] for entry in draws} == {"other"}
    law = record["energy_contract"]["sampling"]["family_law"]
    assert law["present"] == ["other"] and law["draw_consumed_per_step"] is False
    assert set(record["exposure"]["by_family"]) == {data_lib.FAMILY_OTHER}


def test_the_cohort_order_is_canonical_so_enumeration_order_changes_nothing(energy):
    first = train(energy, "first", conditioned=True)
    energy.services._cache["eligible"] = tuple(reversed(energy.rows))
    second = train(energy, "second", conditioned=True)
    assert second["energy_contract_sha256"] == first["energy_contract_sha256"]
    assert second["exposure_trace"] == first["exposure_trace"]
    energy.services._cache["eligible"] = tuple(energy.rows[1:])
    smaller = train(energy, "smaller", conditioned=True)
    assert smaller["common_sha256"] != first["common_sha256"], "a different cohort is a different run"


def test_the_denoiser_diagnostic_reports_what_the_arm_was_actually_given(energy):
    full = train(energy, "full", conditioned=True)
    null = train(energy, "null", conditioned=False)
    assert all(entry["conditioned"] is True for entry in full["phases"]["denoiser"]["history"])
    assert all(entry["conditioned"] is False for entry in null["phases"]["denoiser"]["history"])
    assert all(entry["conditioned"] is True for entry in full["phases"]["energy"]["history"])
    assert all(entry["conditioned"] is False for entry in null["phases"]["energy"]["history"])


def test_null_training_never_invokes_a_context_builder_or_a_context_tower(energy, monkeypatch):
    counts = instrument(monkeypatch, energy.services)
    train(energy, "null", conditioned=False)
    assert counts == {"energy_context": 0, "build_energy_context": 0, "context_tower": 0}
    train(energy, "full", conditioned=True)
    assert counts["energy_context"] > 0 and counts["build_energy_context"] > 0
    assert counts["context_tower"] > 0, "the same instrumentation does see the conditioned arm"


def test_the_ledger_counts_row_forwards_honestly(energy):
    energy.config["routes"][ROUTE]["energy_updates"] = 2
    record = train(energy, "counted", conditioned=True)
    totals = record["exposure"]["totals"]
    # denoiser: 3 updates x 2 rows x 1 forward; energy: 2 updates x 2 rows x (2 walk + 2) forwards.
    assert totals["optimizer_updates"] == 5 and totals["microbatch_rows"] == 10
    assert totals["student_forwards"] == 3 * 2 * 1 + 2 * 2 * (2 + 2) == 22
    assert totals["supervised_targets"] == 10 * 3
    ledger = record["forward_ledger"]
    assert ledger["denoiser_phase_per_row"] == 1 and ledger["energy_phase_per_row"] == 4
    assert "ROW-FORWARDS" in ledger["unit"]
    assert record["energy_contract"]["forward_ledger"] == ledger


def test_the_profile_selected_microbatch_is_resolved_and_bound_before_the_identity(energy):
    declared = train(energy, "declared", conditioned=True)
    assert declared["shape"]["microbatch"] == 2 and declared["shape"]["microbatch_source"] == (
        "declared")
    profile = energy.context.path(spec.PROFILE_JSON)
    profile.parent.mkdir(parents=True, exist_ok=True)
    profile.write_text(json.dumps({ROUTE: {"chosen_microbatch": 3}}), encoding="utf-8")
    profiled = train(energy, "profiled", conditioned=True)
    assert profiled["shape"]["microbatch"] == 3
    assert profiled["shape"]["microbatch_source"] == "profile"
    assert profiled["shape"]["profile"]["sha256"] == paths.sha256_file(profile)
    assert profiled["exposure"]["totals"]["microbatch_rows"] == 6 * 3
    assert profiled["energy_contract"]["sampling"]["microbatch"] == 3
    assert profiled["energy_contract_sha256"] != declared["energy_contract_sha256"]
    for bad in ({ROUTE: {"chosen_microbatch": 2.5}}, {ROUTE: {"chosen_microbatch": True}},
                {ROUTE: "not a mapping"}, ["not", "a", "mapping"]):
        profile.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(ValueError):
            energy.services.energy_run_shape()


BAD_SIZES = [0, False, "", -1, 2.5, True, "3", []]


@pytest.mark.parametrize("bad", BAD_SIZES, ids=[repr(value) for value in BAD_SIZES])
def test_a_supplied_falsey_or_invalid_size_is_refused_before_any_work(energy, monkeypatch, bad):
    services = energy.services
    forbid_mutation(monkeypatch, services)
    monkeypatch.setattr(services, "energy_rows", lambda **kwargs: pytest.fail(
        "the cohort was read before the size was refused"))
    directory = energy.root / "never"
    with pytest.raises(ValueError, match="updates must be a positive integer"):
        services.energy_run_shape(updates=bad)
    with pytest.raises(ValueError, match="updates must be a positive integer"):
        services.train_energy(directory=directory, seed=11, population=ALL, conditioned=True,
                              updates=bad)
    with pytest.raises(ValueError, match="stop_after_updates must be a positive integer"):
        services.train_energy(directory=directory, seed=11, population=ALL, conditioned=True,
                              stop_after_updates=bad)
    assert not directory.exists(), "a refused request creates nothing"


def test_only_none_means_unspecified_and_a_positive_stop_still_stops(energy):
    shape = energy.services.energy_run_shape(updates=None)
    assert shape["budget"] == {"denoiser": 3, "energy": 3}
    assert shape["budget_source"] == "the declared phase updates"
    assert energy.services.energy_run_shape(updates=5)["budget"] == {"denoiser": 5, "energy": 5}
    one = train(energy, "one", conditioned=True, stop_after_updates=1)
    assert one["phase_updates"] == {"denoiser": 1, "energy": 0}, "one step, as asked"
    assert train(energy, "all", conditioned=True, stop_after_updates=None)["phase_updates"] == {
        "denoiser": 3, "energy": 3}


def test_a_falsey_profile_entry_is_invalid_not_absent(energy):
    profile = energy.context.path(spec.PROFILE_JSON)
    profile.parent.mkdir(parents=True, exist_ok=True)
    for bad in (0, False, "", [], "2", -3, 1.5):
        profile.write_text(json.dumps({ROUTE: {"chosen_microbatch": bad}}), encoding="utf-8")
        with pytest.raises(ValueError, match="chosen_microbatch must be a positive integer"):
            energy.services.energy_run_shape()
    for bad in (0, False, "", []):
        profile.write_text(json.dumps({ROUTE: bad}), encoding="utf-8")
        with pytest.raises(ValueError, match="not a mapping"):
            energy.services.energy_run_shape()
    for unspecified in ({}, {ROUTE: None}, {ROUTE: {}}, {ROUTE: {"chosen_microbatch": None}}):
        profile.write_text(json.dumps(unspecified), encoding="utf-8")
        shape = energy.services.energy_run_shape()
        assert shape["microbatch"] == 2 and shape["microbatch_source"] == "declared", unspecified
    profile.write_text(json.dumps({ROUTE: {"chosen_microbatch": 3}}), encoding="utf-8")
    assert energy.services.energy_run_shape()["microbatch_source"] == "profile"


# ---------------------------------------------------------------------------
# resume: continuity, and refusal before any mutation
# ---------------------------------------------------------------------------

def test_an_interrupted_run_resumes_the_trace_the_state_and_the_ledger_exactly(energy,
                                                                              monkeypatch):
    draws = capture_draws(monkeypatch)
    whole = train(energy, "whole", conditioned=True)
    reference = list(draws)
    # (stop, progress at the stop, phases whose loss history the resumed run still carries). A run
    # resumed inside the energy phase reports the finished denoiser phase as completed, not re-run.
    for stop, expected, compared in (
            (2, {"denoiser": 2, "energy": 0}, ("denoiser", "energy")),   # inside the denoiser
            (3, {"denoiser": 3, "energy": 0}, ("denoiser", "energy")),   # exactly the boundary
            (4, {"denoiser": 3, "energy": 1}, ("energy",))):             # inside the energy phase
        draws.clear()
        name = f"split{stop}"
        partial = train(energy, name, conditioned=True, stop_after_updates=stop)
        assert partial["phase_updates"] == expected
        assert partial["exposure_trace"]["phases"]["denoiser"]["steps"] == expected["denoiser"]
        assert partial["exposure_trace"]["phases"]["energy"]["steps"] == expected["energy"]
        saved = torch.load(str(energy.root / name / "last.pt"), map_location="cpu",
                           weights_only=False)
        assert saved["energy_trace"]["phases"]["denoiser"]["steps"] == expected["denoiser"]
        taken = list(draws)
        draws.clear()
        resumed = train(energy, name, conditioned=True, resume=True)
        assert resumed["resume"] is not None
        assert resumed["phase_updates"] == whole["phase_updates"] == {"denoiser": 3, "energy": 3}
        assert resumed["exposure_trace"] == whole["exposure_trace"], (
            "the resumed run's trace continued from the checkpoint; it did not reset or skip")
        assert taken + list(draws) == reference, (
            "the interrupted-then-resumed draws are exactly the uninterrupted draws")
        assert resumed["exposure"]["totals"] == whole["exposure"]["totals"]
        assert resumed["energy_contract_sha256"] == whole["energy_contract_sha256"]
        if stop == 4:
            assert resumed["phases"]["denoiser"] == {"updates": 3, "resumed_completed_phase": True}
        for phase in compared:
            report = checkpoints.resumed_trajectory_report(
                whole["phases"][phase]["history"], resumed["phases"][phase]["history"],
                tolerance=1e-5)
            assert report["identical"] and report["compared_steps"] > 0, (stop, phase, report)


def test_the_stored_checkpoints_all_carry_the_contract_and_the_trace(energy):
    record = train(energy, "run", conditioned=True)
    directory = energy.root / "run"
    saved = {}
    for name in ("last.pt", "best.pt"):
        saved[name] = torch.load(str(directory / name), map_location="cpu", weights_only=False)
        assert saved[name]["energy_contract"] == record["energy_contract"]
        assert saved[name]["identity"]["schedule_sha256"] == record["energy_contract_sha256"]
        assert saved[name]["energy_trace"]["chain_sha256"] == (
            record["exposure_trace"]["chain_sha256"])
        assert saved[name]["energy_phase"] == "energy" and saved[name]["step"] == 3
    assert saved["best.pt"]["optimizer"] is None and saved["last.pt"]["optimizer"] is not None
    assert paths.read_json(directory / "run.json")["energy_contract"] == record["energy_contract"]
    partial = train(energy, "partial", conditioned=True, stop_after_updates=2)
    periodic = torch.load(str(energy.root / "partial" / "last.pt"), map_location="cpu",
                          weights_only=False)
    assert periodic["energy_contract"] == partial["energy_contract"]
    assert periodic["energy_phase"] == "denoiser" and periodic["step"] == 2


def test_resuming_a_completed_run_is_a_no_op_that_keeps_the_trace(energy):
    first = train(energy, "done", conditioned=True)
    again = train(energy, "done", conditioned=True, resume=True)
    assert again["phase_updates"] == first["phase_updates"]
    assert again["exposure_trace"] == first["exposure_trace"]
    assert again["energy_contract_sha256"] == first["energy_contract_sha256"]
    assert again["exposure"]["totals"] == first["exposure"]["totals"]


def _victim(energy, *, conditioned=True, stop=2):
    train(energy, "victim", conditioned=conditioned, stop_after_updates=stop)
    return energy.root / "victim"


def _profile_microbatch(energy):
    profile = energy.context.path(spec.PROFILE_JSON)
    profile.parent.mkdir(parents=True, exist_ok=True)
    profile.write_text(json.dumps({ROUTE: {"chosen_microbatch": 3}}), encoding="utf-8")
    return {}


@pytest.mark.parametrize("name,request_change,fragment", [
    ("population", lambda h: {"population": HER2_ONLY}, "population.name"),
    ("budgets", lambda h: {"updates": 5}, "phases.budget.denoiser"),
    ("microbatch", _profile_microbatch, "sampling.microbatch"),
    ("conditioning", lambda h: {"conditioned": False}, "conditioning.mode"),
    ("seed", lambda h: {"seed": 12}, "sampling.seed")])
def test_a_changed_experiment_refuses_the_resume_before_anything_is_mutated(
        energy, monkeypatch, name, request_change, fragment):
    victim = _victim(energy)
    arguments = {"seed": 11, "population": ALL, "conditioned": True, "resume": True}
    arguments.update(request_change(energy))
    before = snapshot(victim)
    forbid_mutation(monkeypatch, energy.services)
    with pytest.raises(ValueError, match="different energy contract") as caught:
        energy.services.train_energy(directory=victim, **arguments)
    assert fragment in str(caught.value), (name, str(caught.value))
    assert snapshot(victim) == before, "the checkpoint directory is untouched"


def _legacy(payload):
    payload.pop("energy_contract")
    payload["identity"]["schedule_sha256"] = None


def _forged_mode(payload):
    contract = payload["energy_contract"]
    contract["conditioning"]["mode"] = energy_contract.MODE_NULL
    contract["energy_contract_sha256"] = energy_contract.contract_digest(contract)
    payload["identity"]["schedule_sha256"] = contract["energy_contract_sha256"]


def _trace_behind(payload):
    payload["energy_trace"]["phases"]["denoiser"]["steps"] = 1


@pytest.mark.parametrize("name,mutate,phrase", [
    ("legacy", _legacy, "cannot be resumed, scored or relabelled"),
    ("missing_phase", lambda p: p.update(energy_phase=""), "records energy_phase"),
    ("unknown_phase", lambda p: p.update(energy_phase="bogus"), "records energy_phase"),
    ("step_past_budget", lambda p: p.update(step=99), "against a denoiser budget of 3"),
    ("budgets_edited", lambda p: p.update(phase_budget={"denoiser": 9, "energy": 9}),
     "different experiment"),
    ("no_optimizer", lambda p: p.update(optimizer=None), "no optimizer state"),
    ("mode_edited", lambda p: p["energy_contract"]["conditioning"].update(mode="null"),
     "does not hash"),
    ("digest_swapped", lambda p: p["identity"].update(schedule_sha256="0" * 64),
     "not the identity's schedule_sha256"),
    ("mode_forged", _forged_mode, "inconsistent"),
    ("no_trace", lambda p: p.pop("energy_trace"), "no energy_trace"),
    ("trace_behind_weights", _trace_behind, "the trace and the weights disagree"),
    # The trace and the ledger must follow from the contract's microbatch and tiles.
    ("trace_rows_zeroed",
     lambda p: p["energy_trace"]["phases"]["denoiser"].update(rows=0, site_exposures=0),
     "makes 4"),
    ("trace_sites_beyond_the_tiles",
     lambda p: p["energy_trace"]["phases"]["denoiser"].update(site_exposures=99),
     "site exposures"),
    ("no_ledger", lambda p: p.pop("exposure"), "no exposure ledger"),
    ("ledger_updates_edited", _shift_ledger("optimizer_updates"), "exposure ledger records"),
    ("ledger_rows_edited", _shift_ledger("microbatch_rows"), "exposure ledger records"),
    ("ledger_not_additive", _break_additivity, "do not sum to its totals"),
    # The full-checkpoint schema, required keys and RNG bundle are read before a model exists.
    ("wrong_schema", lambda p: p.update(schema_version="her2-routes-checkpoint/1"),
     "not her2-routes-checkpoint/3"),
    ("no_sampler_position", lambda p: p.pop("sampler_position"), "missing 'sampler_position'"),
    ("no_rng", lambda p: p.pop("rng"), "missing 'rng'"),
    ("rng_not_a_bundle", lambda p: p.update(rng={"torch_cpu": 1}),
     "where the RNG bundle belongs"),
    ("rng_lacks_the_sampler", lambda p: p["rng"].numpy.pop("sampler_rng"),
     "no state for the generator"),
    ("rng_lacks_the_mask_generator", lambda p: p["rng"].numpy.pop("mask_rng"),
     "no state for the generator"),
    ("rng_lacks_the_energy_generator", lambda p: p["rng"].torch_generators.pop("energy_rng"),
     "no state for the generator"),
    ("rng_lacks_the_cpu_state", lambda p: setattr(p["rng"], "torch_cpu", None),
     "torch CPU random state"),
    ("rng_lacks_the_python_state", lambda p: setattr(p["rng"], "python_random", None),
     "Python random state"),
    ("rng_state_of_the_wrong_kind",
     lambda p: p["rng"].torch_generators.update(sampler_rng=p["rng"].numpy.pop("sampler_rng")),
     "is a NumPy generator here")])
def test_a_damaged_checkpoint_refuses_the_resume_before_anything_is_mutated(
        energy, monkeypatch, name, mutate, phrase):
    victim = _victim(energy)
    tamper(victim / "last.pt", mutate)
    before = snapshot(victim)
    forbid_mutation(monkeypatch, energy.services)
    with pytest.raises(ValueError, match=phrase):
        energy.services.train_energy(directory=victim, seed=11, population=ALL, conditioned=True,
                                     resume=True)
    assert snapshot(victim) == before, "the checkpoint is preserved, not repaired or overwritten"


def test_a_legacy_artifact_is_preserved_whether_the_start_is_a_resume_or_not(energy,
                                                                            monkeypatch):
    legacy = energy.root / "legacy"
    legacy.mkdir()
    for name in ("last.pt", "best.pt"):
        torch.save({"schema_version": checkpoints.CHECKPOINT_SCHEMA, "step": 4,
                    "identity": {"arm": "full", "route": ROUTE, "schedule_sha256": None},
                    "energy_phase": "energy"}, str(legacy / name))
    only_best = energy.root / "only_best"
    only_best.mkdir()
    shutil.copy(legacy / "best.pt", only_best / "best.pt")
    before = snapshot(legacy)
    forbid_mutation(monkeypatch, energy.services)
    for resume in (True, False):
        with pytest.raises(ValueError, match="cannot be resumed, scored or relabelled"):
            energy.services.train_energy(directory=legacy, seed=11, population=ALL,
                                         conditioned=True, resume=resume)
        assert snapshot(legacy) == before
    with pytest.raises(ValueError, match="cannot be resumed, scored or relabelled"):
        energy.services.train_energy(directory=only_best, seed=11, population=ALL,
                                     conditioned=False, resume=True)
    assert [path.name for path in only_best.iterdir()] == ["best.pt"]


def test_a_terminal_best_is_for_inference_and_is_not_a_resumable_state(energy, monkeypatch):
    train(energy, "done", conditioned=True)
    best = energy.root / "done" / "best.pt"
    model, _heads, record = energy.services.load_for_scoring(ROUTE, best)
    assert record["energy_runtime"]["terminal"] is True and model.runtime.arm == "full"
    victim = energy.root / "copied"
    victim.mkdir()
    shutil.copy(best, victim / "last.pt")
    before = snapshot(victim)
    forbid_mutation(monkeypatch, energy.services)
    with pytest.raises(ValueError, match="no optimizer state"):
        energy.services.train_energy(directory=victim, seed=11, population=ALL, conditioned=True,
                                     resume=True)
    assert snapshot(victim) == before


def test_a_mid_training_checkpoint_loads_for_inference_but_is_not_terminal(energy):
    victim = _victim(energy)
    model, _heads, record = energy.services.load_for_scoring(ROUTE, victim / "last.pt")
    assert record["energy_runtime"]["terminal"] is False and record["energy_runtime"]["step"] == 2
    assert model.runtime.source == "checkpoint"


def test_the_resume_rng_inventory_is_what_a_real_energy_state_carries(energy):
    state = energy.services.fresh_state(1)
    state.energy_rng = energy.services.energy_generator(1)
    kinds = {name: ("torch" if isinstance(generator, torch.Generator) else "numpy")
             for name, generator in state.generators().items()}
    assert kinds == services_mod.ENERGY_RNG_INVENTORY
    train(energy, "run", conditioned=True)
    for name in ("last.pt", "best.pt"):
        payload = torch.load(str(energy.root / "run" / name), map_location="cpu",
                             weights_only=False)
        assert checkpoints.require_restorable_metadata(
            payload, path=name, generators=services_mod.ENERGY_RNG_INVENTORY) is True


def test_the_metadata_preflight_reads_the_payload_and_builds_nothing(energy, monkeypatch):
    train(energy, "run", conditioned=True)
    payload = torch.load(str(energy.root / "run" / "last.pt"), map_location="cpu",
                         weights_only=False)
    forbid_mutation(monkeypatch, energy.services)
    before = copy.deepcopy({key: payload[key] for key in ("exposure", "energy_trace", "step")})
    assert checkpoints.require_restorable_metadata(
        payload, path="last.pt", generators=services_mod.ENERGY_RNG_INVENTORY) is True
    assert {key: payload[key] for key in before} == before
    with pytest.raises(ValueError, match="'numpy' or 'torch'"):
        checkpoints.require_restorable_metadata(payload, path="last.pt",
                                                generators={"mask_rng": "cuda"})
    with pytest.raises(ValueError, match="no state for the generator"):
        checkpoints.require_restorable_metadata(
            payload, path="last.pt", generators={**services_mod.ENERGY_RNG_INVENTORY,
                                                 "extra_rng": "numpy"})


# The scalars ``restore`` indexes after it has loaded state. Each one used to pass the preflight
# whether it was missing or malformed, and then failed (or, for a bool, silently coerced) midway
# through a resume that had already built a model and seeded the global generators.
_BAD_COUNTERS = (-1, True, np.bool_(True), 2.0, 1.5, "3", None, float("nan"))
_BAD_METRICS = (float("nan"), float("inf"), float("-inf"), True, np.bool_(False), "0.5", [0.5],
                10 ** 400)
_MALFORMED_SCALARS = (
    [pytest.param("best_metric", value, id=f"best_metric={value!r}"[:40])
     for value in _BAD_METRICS]
    + [pytest.param(key, value, id=f"{key}={value!r}")
       for key in ("best_step", "plateau_reductions", "plateau_bad_evaluations",
                   "sampler_position")
       for value in _BAD_COUNTERS])


def test_the_scalar_fields_restore_reads_are_named_by_the_preflight():
    assert set(checkpoints.RESTORE_SCALAR_KEYS) == {"best_metric", "best_step",
                                                    "plateau_reductions",
                                                    "plateau_bad_evaluations"}
    assert not set(checkpoints.RESTORE_SCALAR_KEYS) & set(checkpoints.REQUIRED_KEYS), (
        "REQUIRED_KEYS is the list a resume reports as restored; the scalars are checked beside it")
    assert set(checkpoints.RESTORE_COUNTER_KEYS) == {
        "step", "sampler_position", "best_step", "plateau_reductions", "plateau_bad_evaluations"}


def test_the_scalar_preflight_accepts_exactly_the_forms_restore_can_read(energy):
    train(energy, "run", conditioned=True)
    payload = torch.load(str(energy.root / "run" / "last.pt"), map_location="cpu",
                         weights_only=False)

    def accepted(**changes):
        return checkpoints.require_restorable_metadata(
            {**payload, **changes}, path="last.pt", generators=services_mod.ENERGY_RNG_INVENTORY)
    assert accepted() is True
    # The metric is None (nothing evaluated yet) or any finite real; ints and numpy scalars are real.
    for metric in (None, 0.0, -3.5, 2, np.float32(0.25), np.float64(1.5)):
        assert accepted(best_metric=metric) is True, metric
    for key in checkpoints.RESTORE_COUNTER_KEYS:
        for counter in (0, 7, np.int64(3)):
            assert accepted(**{key: counter}) is True, (key, counter)
    for metric in _BAD_METRICS:
        with pytest.raises(ValueError, match="'best_metric' is"):
            accepted(best_metric=metric)
    for key in checkpoints.RESTORE_COUNTER_KEYS:
        for counter in _BAD_COUNTERS:
            with pytest.raises(ValueError, match=f"'{key}' is"):
                accepted(**{key: counter})
    # The refusal is about the field, so a payload that DID carry the field is not blamed for it.
    for key in checkpoints.RESTORE_SCALAR_KEYS:
        damaged = dict(payload)
        del damaged[key]
        with pytest.raises(ValueError, match=f"missing '{key}'"):
            checkpoints.require_restorable_metadata(
                damaged, path="last.pt", generators=services_mod.ENERGY_RNG_INVENTORY)


@pytest.mark.parametrize("field", checkpoints.RESTORE_SCALAR_KEYS)
def test_a_checkpoint_missing_a_scalar_restore_reads_is_refused_before_anything_is_built(
        energy, monkeypatch, field):
    victim = _victim(energy)
    tamper(victim / "last.pt", lambda payload: payload.pop(field))
    before = snapshot(victim)
    forbid_mutation(monkeypatch, energy.services)
    with pytest.raises(ValueError, match=f"missing '{field}'"):
        energy.services.train_energy(directory=victim, seed=11, population=ALL, conditioned=True,
                                     resume=True)
    assert snapshot(victim) == before, "the checkpoint is preserved, not repaired or overwritten"


@pytest.mark.parametrize("field,value", _MALFORMED_SCALARS)
def test_a_malformed_scalar_is_refused_before_anything_is_built(energy, monkeypatch, field,
                                                                value):
    victim = _victim(energy)
    tamper(victim / "last.pt", lambda payload: payload.update({field: value}))
    before = snapshot(victim)
    forbid_mutation(monkeypatch, energy.services)
    with pytest.raises(ValueError, match=f"'{field}' is .*it must be"):
        energy.services.train_energy(directory=victim, seed=11, population=ALL, conditioned=True,
                                     resume=True)
    assert snapshot(victim) == before, "the checkpoint is preserved, not repaired or overwritten"


# ---------------------------------------------------------------------------
# inference: the mode survives every path
# ---------------------------------------------------------------------------

def test_null_scoring_and_sampling_never_build_a_context_and_survive_model_reuse(energy,
                                                                                monkeypatch):
    train(energy, "null", conditioned=False)
    best = energy.root / "null" / "best.pt"
    counts = instrument(monkeypatch, energy.services)
    model, heads, record = energy.services.load_for_scoring(ROUTE, best)
    assert (model.runtime.arm, model.runtime.mode) == ("null", "null")
    assert record["energy_runtime"]["mode"] == "null" and record["trained"] is True
    assert model.document()["context"]["conditioned"] is True, (
        "the architecture owns condition tables; only the runtime contract knows the arm")
    owned = energy.services.score_panel(ROUTE, best, PANEL)
    reused = energy.services.score_panel(ROUTE, None, PANEL, model=model, heads=heads)
    assert owned == pytest.approx(reused), "a model= reuse path keeps the validated null mode"
    result = generate.energy_generate(energy.services, model, draws=3, context_rows=(), seed=5,
                                      batch_size=2, sampling_options={"steps": 2})
    assert len(result.windows) == 3 and all(len(window) == 3 for window in result.windows)
    assert result.diagnostics["mode"] == "null" and result.diagnostics["arm"] == "null"
    assert result.diagnostics["requested_bin"] is None
    assert result.diagnostics["requested_bin_applicable"] is False
    assert result.diagnostics["context_built"] is False
    assert counts == {"energy_context": 0, "build_energy_context": 0, "context_tower": 0}
    # a conditioned model through the very same paths DOES build and run its context
    train(energy, "full", conditioned=True)
    full, full_heads, _ = energy.services.load_for_scoring(ROUTE, energy.root / "full" / "best.pt")
    energy.services.score_panel(ROUTE, None, PANEL, model=full, heads=full_heads)
    generate.energy_generate(energy.services, full, draws=2, context_rows=energy.rows[:2],
                             seed=5, batch_size=2, sampling_options={"steps": 2})
    assert counts["energy_context"] > 0 and counts["context_tower"] > 0


def test_the_null_arm_scores_differently_from_the_conditioned_arm_on_the_same_panel(energy):
    train(energy, "full", conditioned=True)
    train(energy, "null", conditioned=False)
    full = energy.services.score_panel(ROUTE, energy.root / "full" / "best.pt", PANEL)
    null = energy.services.score_panel(ROUTE, energy.root / "null" / "best.pt", PANEL)
    assert len(full) == len(null) == 3 and full != pytest.approx(null)


def test_an_explicit_contradictory_arm_is_refused_before_a_model_is_built(energy, monkeypatch):
    train(energy, "null", conditioned=False)
    train(energy, "full", conditioned=True)
    null_best, full_best = energy.root / "null" / "best.pt", energy.root / "full" / "best.pt"
    model, heads, _record = energy.services.load_for_scoring(ROUTE, null_best)
    with pytest.raises(ValueError, match="'full' arm was requested"):
        energy.services.score_panel(ROUTE, None, PANEL, model=model, heads=heads, arm="full")
    energy.services.score_panel(ROUTE, None, PANEL, model=model, heads=heads, arm="null")
    monkeypatch.setattr(energy.services, "build_route",
                        lambda *a, **k: pytest.fail("a model was built for a contradicted arm"))
    with pytest.raises(ValueError, match="'full' arm was requested"):
        energy.services.load_for_scoring(ROUTE, null_best, arm="full")
    with pytest.raises(ValueError, match="'null' arm was requested"):
        energy.services.score_panel(ROUTE, full_best, PANEL, arm="null")
    with pytest.raises(ValueError, match="'density_only' arm was requested"):
        energy.services.load_for_scoring(ROUTE, full_best, arm="density_only")


def test_a_model_with_no_validated_runtime_is_refused_by_every_inference_path(energy):
    bare = energy.services.build_route(ROUTE, seed=1)
    with pytest.raises(ValueError, match="no validated runtime contract"):
        energy.services.score_panel(ROUTE, None, PANEL, model=bare, heads=None)
    with pytest.raises(ValueError, match="no validated runtime contract"):
        energy.services.energy_panel_scores(bare, PANEL)
    with pytest.raises(ValueError, match="no validated runtime contract"):
        generate.energy_generate(energy.services, bare, draws=1, context_rows=energy.rows[:1],
                                 seed=1)
    with pytest.raises(ValueError, match="no validated runtime contract"):
        energy.services.energy_representation(bare, energy.rows[:1])


def test_a_contradictory_mode_or_bin_is_refused_at_generation(energy):
    train(energy, "null", conditioned=False)
    train(energy, "full", conditioned=True)
    null, _, _ = energy.services.load_for_scoring(ROUTE, energy.root / "null" / "best.pt")
    full, _, _ = energy.services.load_for_scoring(ROUTE, energy.root / "full" / "best.pt")
    with pytest.raises(ValueError, match="'conditioned' mode was requested"):
        generate.energy_generate(energy.services, null, draws=1, context_rows=energy.rows[:1],
                                 seed=1, mode="conditioned")
    with pytest.raises(ValueError, match="'null' mode was requested"):
        generate.energy_generate(energy.services, full, draws=1, context_rows=energy.rows[:1],
                                 seed=1, mode="null")
    with pytest.raises(ValueError, match="not applicable"):
        generate.energy_generate(energy.services, null, draws=1, context_rows=(), seed=1,
                                 requested_bin="high")
    with pytest.raises(ValueError, match="needs measured rows"):
        generate.energy_generate(energy.services, full, draws=1, context_rows=(), seed=1)
    with pytest.raises(ValueError, match="no pooled context vector"):
        energy.services.energy_representation(null, energy.rows[:1])
    pooled = energy.services.energy_representation(full, energy.rows[:1])
    assert tuple(pooled.shape) == (1, 16)
    assert generate.energy_generate(
        energy.services, full, draws=1, context_rows=energy.rows[:1], seed=1,
        sampling_options={"steps": 2}).diagnostics["requested_bin"] == "high"


@pytest.mark.parametrize("name,mutate,phrase", [
    ("legacy", _legacy, "cannot be resumed, scored or relabelled"),
    ("mode_edited", lambda p: p["energy_contract"]["conditioning"].update(mode="null"),
     "does not hash"),
    ("digest_swapped", lambda p: p["identity"].update(schedule_sha256="0" * 64),
     "not the identity's schedule_sha256"),
    ("mode_forged", _forged_mode, "inconsistent"),
    ("unknown_phase", lambda p: p.update(energy_phase="bogus"), "records energy_phase")])
def test_load_for_scoring_recomputes_the_stored_contract_before_building_a_model(
        energy, monkeypatch, name, mutate, phrase):
    train(energy, "run", conditioned=True)
    best = energy.root / "run" / "best.pt"
    tamper(best, mutate)
    monkeypatch.setattr(energy.services, "build_route",
                        lambda *a, **k: pytest.fail("a model was built for a damaged checkpoint"))
    monkeypatch.setattr(energy.services, "build_heads",
                        lambda *a, **k: pytest.fail("heads were built for a damaged checkpoint"))
    with pytest.raises(ValueError, match=phrase):
        energy.services.load_for_scoring(ROUTE, best)
    with pytest.raises(ValueError, match=phrase):
        energy.services.score_panel(ROUTE, best, PANEL)


def test_a_named_but_absent_energy_checkpoint_is_refused_before_a_model_is_built(energy,
                                                                                monkeypatch):
    monkeypatch.setattr(energy.services, "build_route",
                        lambda *a, **k: pytest.fail("a model was built for an absent checkpoint"))
    with pytest.raises(ValueError, match="is absent"):
        energy.services.load_for_scoring(ROUTE, energy.root / "no_such.pt")


def test_a_checkpoint_free_energy_model_is_labelled_random_and_has_an_explicit_mode(energy):
    model, _heads, record = energy.services.load_for_scoring(ROUTE, None)
    assert model.runtime.source == "checkpoint_free" and model.runtime.mode == "conditioned"
    assert record["initialization"] == "random_untrained" and record["trained"] is False
    assert "NOT a frozen zero-shot backbone" in record["label"]
    null, _heads, null_record = energy.services.load_for_scoring(ROUTE, None, arm="null")
    assert null.runtime.mode == "null" and null_record["energy_runtime"]["mode"] == "null"
    with pytest.raises(ValueError, match="Unknown energy arm"):
        energy.services.load_for_scoring(ROUTE, None, arm="density_only")
    scores = energy.services.score_panel(ROUTE, None, PANEL)
    assert len(scores) == 3
    assert energy.services.score_panel(ROUTE, None, PANEL, arm="null") != pytest.approx(scores)


def endpoint_spies(energy, monkeypatch):
    """Record every model load and every energy generation the raw-generation endpoint makes.

    Only the independent assessor is replaced (it fits a display library the test does not have);
    loading, the runtime contract and ``energy_generate`` are the real ones.
    """
    services = energy.services
    monkeypatch.setattr(services, "independent_assessor", lambda: {
        "score": lambda windows: [float(len(set(window))) for window in windows],
        "fit": {}, "independence": "synthetic"})
    loads, generations = [], []
    original_load = services.load_for_scoring

    def load(route, checkpoint, **kwargs):
        loaded = original_load(route, checkpoint, **kwargs)
        loads.append({"checkpoint": checkpoint, "kwargs": dict(kwargs), "model": loaded[0]})
        return loaded
    monkeypatch.setattr(services, "load_for_scoring", load)
    original_generate = generate.energy_generate

    def spy(services_argument, model, **kwargs):
        generations.append({"arm": model.runtime.arm, "mode": kwargs.get("mode"),
                            "context_rows": len(kwargs["context_rows"]),
                            "initialization": model.runtime.initialization})
        return original_generate(services_argument, model, **kwargs)
    monkeypatch.setattr(generate, "energy_generate", spy)
    return loads, generations


def test_a_null_checkpoints_endpoint_comparator_is_null_and_starts_from_the_trained_seed(
        energy, monkeypatch):
    train(energy, "null", conditioned=False, seed=5)
    best = energy.root / "null" / "best.pt"
    loads, generations = endpoint_spies(energy, monkeypatch)
    counts = instrument(monkeypatch, energy.services)
    monkeypatch.setattr(energy.services, "energy_rows", lambda **kwargs: pytest.fail(
        "context rows were read for an endpoint in which no arm is given a context"))
    result = energy.services.generation_endpoint(ROUTE, best, draws=2, seed=3)
    assert counts == {"energy_context": 0, "build_energy_context": 0, "context_tower": 0}, (
        "neither endpoint arm builds a context or runs a context tower")
    assert generations == [
        {"arm": "null", "mode": "null", "context_rows": 0, "initialization": "trained_checkpoint"},
        {"arm": "null", "mode": "null", "context_rows": 0, "initialization": "random_untrained"}]
    assert loads[0]["checkpoint"] == best and loads[0]["kwargs"] == {}
    assert loads[1]["checkpoint"] is None
    assert loads[1]["kwargs"] == {"seed": 5, "arm": "null"}, (
        "the comparator keeps the trained arm and the checkpoint's recorded initialization seed")
    expected = energy.services.build_route(ROUTE, seed=5).state_dict()
    comparator = loads[1]["model"].state_dict()
    assert all(torch.equal(comparator[name], tensor) for name, tensor in expected.items()), (
        "the comparator IS the trained model's own random starting point")
    other = energy.services.build_route(ROUTE, seed=0).state_dict()
    assert any(not torch.equal(comparator[name], tensor) for name, tensor in other.items()), (
        "the seed is what makes it so: the default seed builds different weights")
    arms = result["arms"]
    assert arms["trained"]["checkpoint"]["energy_runtime"]["mode"] == "null"
    assert arms["backbone"]["checkpoint"]["energy_runtime"]["mode"] == "null"
    assert arms["backbone"]["checkpoint"]["initialization"] == "random_untrained"
    assert arms["backbone"]["comparator"] == {
        "arm": "null", "mode": "null", "initialization": "random_untrained",
        "initialization_seed": 5,
        "initialization_seed_source": "the trained checkpoint's recorded identity seed",
        "sampling_seed": 3}
    assert "comparator" not in arms["trained"]
    assert result["value"] is not None
    assert "RANDOMLY INITIALISED" in result["backbone_note"]
    assert "does not include a change of conditioning" in result["backbone_note"]
    assert "predicted quality" in result["definition"], "predicted, not measured, quality"
    assert "nothing here has been experimentally assayed" in result["not_measured"]


def test_a_full_checkpoints_endpoint_comparator_is_still_conditioned(energy, monkeypatch):
    train(energy, "full", conditioned=True, seed=5)
    loads, generations = endpoint_spies(energy, monkeypatch)
    counts = instrument(monkeypatch, energy.services)
    result = energy.services.generation_endpoint(ROUTE, energy.root / "full" / "best.pt", draws=2,
                                                 seed=3)
    assert [(entry["arm"], entry["mode"]) for entry in generations] == [
        ("full", "conditioned"), ("full", "conditioned")]
    assert all(entry["context_rows"] > 0 for entry in generations)
    assert [entry["initialization"] for entry in generations] == [
        "trained_checkpoint", "random_untrained"]
    assert loads[1]["kwargs"] == {"seed": 5, "arm": "full"}
    assert counts["energy_context"] > 0 and counts["context_tower"] > 0
    assert result["arms"]["backbone"]["comparator"]["mode"] == "conditioned"
    assert result["arms"]["backbone"]["comparator"]["initialization_seed"] == 5


def test_a_checkpoint_free_endpoint_keeps_the_default_arm_and_initialization_seed(
        energy, monkeypatch):
    loads, generations = endpoint_spies(energy, monkeypatch)
    result = energy.services.generation_endpoint(ROUTE, None, draws=2, seed=3)
    assert loads[0]["checkpoint"] is None and loads[0]["kwargs"] == {}
    assert loads[1]["kwargs"] == {"seed": 0, "arm": "full"}
    assert [(entry["arm"], entry["mode"]) for entry in generations] == [
        ("full", "conditioned"), ("full", "conditioned")]
    comparator = result["arms"]["backbone"]["comparator"]
    assert comparator["initialization_seed"] == 0
    assert comparator["initialization_seed_source"].startswith("the default initialization seed")


def test_the_energy_route_still_refuses_a_requested_bin(energy):
    model, heads, _record = energy.services.load_for_scoring(ROUTE, None)
    with pytest.raises(ValueError, match="no bin"):
        energy.services.score_panel(ROUTE, None, PANEL, model=model, heads=heads,
                                    requested_bin="null")


def test_score_panel_keeps_the_old_full_default_for_the_likelihood_routes(monkeypatch, energy):
    seen = []

    def loader(route, checkpoint, *, seed=0, arm=None):
        seen.append((route, arm))
        raise RuntimeError("stop after recording the request")
    monkeypatch.setattr(energy.services, "load_for_scoring", loader)
    for route, expected in ((services_mod.ROUTE_MASKED, "full"), (ROUTE, None)):
        with pytest.raises(RuntimeError, match="stop after recording"):
            energy.services.score_panel(route, None, PANEL)
        assert seen[-1] == (route, expected)
    with pytest.raises(RuntimeError, match="stop after recording"):
        energy.services.score_panel(services_mod.ROUTE_MASKED, None, PANEL, arm="distilled")
    assert seen[-1] == (services_mod.ROUTE_MASKED, "distilled")


# ---------------------------------------------------------------------------
# the bakeoff and the smoke use the matched experiment
# ---------------------------------------------------------------------------

def test_the_bakeoff_trains_both_arms_on_the_same_declared_population(energy):
    result = energy.services.run_bakeoff()
    record = result["routes"][ROUTE]
    baseline = record["density_only_baseline"]
    assert result["all_terminal"] is True
    assert record["population"] == ALL and record["conditioned"] is True
    assert baseline["arm"] == "null" and baseline["matched_to_full"]["same_draws"] is True
    null_run = paths.read_json(energy.context.path("bakeoff", ROUTE, "density_only", "run.json"))
    assert null_run["population"] == ALL and null_run["conditioned"] is False
    assert (null_run["energy_contract"]["population"]["cohort"]["rows"]
            == record["energy_contract"]["population"]["cohort"]["rows"] == 12), (
        "the null arm sees the same 12 rows, not the 6 HER2 rows the old coupling gave it")
    assert null_run["common_sha256"] == record["common_sha256"]
    assert baseline["energy_contract_sha256"] == null_run["energy_contract_sha256"]
    assert record["training_rows"] == null_run["training_rows"] == 12


def test_the_smoke_resume_check_passes_and_includes_the_trace(energy):
    result = energy.services.smoke_energy_resume(directory=energy.root / "smoke", seed=11, steps=2)
    assert result["passed"], result["failures"]
    assert result["detail"]["exposure_trace_continued"] is True
    assert result["full"] == result["resumed"] == {"denoiser": 4, "energy": 4}


# ---------------------------------------------------------------------------
# route capabilities: refuse early, before any expensive work
# ---------------------------------------------------------------------------

def test_the_capability_error_is_a_lineage_error_and_the_route_names_agree():
    assert issubclass(capabilities.RouteCapabilityError, lineage.LineageError)
    assert issubclass(capabilities.RouteCapabilityError, ValueError)
    assert (capabilities.ENERGY_ROUTE == services_mod.ROUTE_ENERGY == energy_contract.ENERGY_ROUTE)
    assert capabilities.supports_token_likelihood(services_mod.ROUTE_MASKED)
    assert capabilities.supports_token_likelihood(services_mod.ROUTE_CAUSAL)
    assert capabilities.supports_token_likelihood(services_mod.ROUTE_DPLM_150M)
    assert not capabilities.supports_token_likelihood(ROUTE)
    assert capabilities.require_token_likelihood(services_mod.ROUTE_MASKED, "anything") is True
    record = capabilities.capabilities(ROUTE)
    assert record["token_likelihood"] is False and record["energy_scoring"] is True
    assert capabilities.capabilities(services_mod.ROUTE_MASKED)["token_likelihood"] is True
    with pytest.raises(capabilities.RouteCapabilityError, match="FlightServices.train"):
        capabilities.require_token_likelihood(ROUTE, "FlightServices.train")


def _confirmation_stub(monkeypatch):
    monkeypatch.setattr(services_mod.lineage, "consume_frozen_recipe",
                        lambda path, freeze_record=None: {"route": ROUTE})
    monkeypatch.setattr(services_mod.lineage, "require_replayable",
                        lambda recipe, where=None: {})


OPERATIONS = {
    "train": lambda h: h.services.train(ROUTE, stream=None, updates=1, seed=1,
                                         directory=h.root / "never"),
    "curriculum_stream": lambda h: h.services.curriculum_stream(ROUTE, seed=1, updates=1),
    "window_log_probs": lambda h: h.services.window_log_probs(ROUTE, object(), object()),
    "canonical_log_probs": lambda h: h.services.canonical_log_probs(ROUTE, object(), object()),
    "representation": lambda h: h.services.representation(ROUTE, object(), object()),
    "collate": lambda h: h.services.collate(ROUTE, []),
    "window_scores": lambda h: h.services.window_scores(ROUTE, object(), ["WGGDGFYAMD"],
                                                         condition=None),
    "joint_support": lambda h: h.services.joint_support(ROUTE, object(), None, sites=(),
                                                         condition=None),
    "distil_teacher": lambda h: h.services.distil_teacher(ROUTE, object(), bank=[], panel=[]),
    "run_controls": lambda h: h.services.run_controls(),
    "run_priors": lambda h: h.services.run_priors(),
    "run_posttrain": lambda h: h.services.run_posttrain(),
    "run_confirmation": lambda h: h.services.run_confirmation(),
}


@pytest.mark.parametrize("operation", sorted(OPERATIONS))
def test_a_token_likelihood_operation_refuses_the_energy_route_before_any_work(
        energy, monkeypatch, operation):
    _confirmation_stub(monkeypatch)
    services = energy.services
    forbid_mutation(monkeypatch, services)
    for name in ("training_pools", "development_panel_outcomes", "her2_training_bank",
                 "measured_pairs", "adaptation_bank", "build_optimizer", "paired_oas",
                 "checkpoint_lineage"):
        monkeypatch.setattr(services, name, lambda *a, _name=name, **k: pytest.fail(
            f"{_name} ran before the capability refusal"))
    with pytest.raises(capabilities.RouteCapabilityError) as caught:
        OPERATIONS[operation](energy)
    assert isinstance(caught.value, lineage.LineageError)
    assert f"FlightServices.{operation}" in str(caught.value)
    assert "no likelihood metric is invented" in str(caught.value)
    assert not (energy.root / "never").exists()
    assert not energy.context.path("controls").exists()
    assert not energy.context.path("priors").exists()
    assert not energy.context.path("posttrain").exists()
    assert not energy.context.path("confirm").exists()


def test_a_likelihood_route_is_not_refused_by_the_guards(energy):
    # These calls fail (or not) for their own reasons on made-up arguments; what matters is that the
    # capability guard is not among them.
    for call in (lambda: energy.services.window_log_probs(services_mod.ROUTE_MASKED, None,
                                                           types.SimpleNamespace(tokens=None)),
                 lambda: energy.services.collate(services_mod.ROUTE_MASKED, [])):
        try:
            call()
        except capabilities.RouteCapabilityError:
            pytest.fail("a likelihood route was refused by the energy capability guard")
        except Exception:
            pass


def test_the_retention_panel_reports_the_energy_route_as_unsupported_without_reading_anything(
        energy, monkeypatch):
    monkeypatch.setattr(energy.services, "paired_oas",
                        lambda *a, **k: pytest.fail("the repertoire was read for the energy route"))
    monkeypatch.setattr(energy.services, "load_for_scoring",
                        lambda *a, **k: pytest.fail("a model was loaded for the energy route"))
    panel = energy.services.panel_prior_retention(ROUTE, None)
    assert panel["panel"] == "E1" and panel["supported"] is False and panel["value"] is None
    assert "no per-token likelihood" in panel["blocking_reason"]


def test_the_energy_route_keeps_its_own_training_scoring_and_sampling(energy):
    record = train(energy, "own", conditioned=True)
    assert record["updates"] == 6
    best = energy.root / "own" / "best.pt"
    assert len(energy.services.score_panel(ROUTE, best, PANEL)) == 3
    model, _heads, _record = energy.services.load_for_scoring(ROUTE, best)
    result = generate.energy_generate(energy.services, model, draws=2,
                                      context_rows=energy.rows[:2], seed=3, batch_size=2,
                                      sampling_options={"steps": 2})
    assert len(result.windows) == 2 and result.log_probabilities is None
    assert result.diagnostics["requested_bin"] == "high" and result.diagnostics["mode"] == (
        "conditioned")
