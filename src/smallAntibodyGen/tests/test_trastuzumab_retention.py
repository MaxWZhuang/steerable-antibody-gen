"""Frozen held-out reconstruction retention (M3): the plan, the lineage binding and the readout.

Model-facing tests run the production tiny ``FusedDPLM`` (nonzero native and adapter dropout,
activation checkpointing) behind real ``StageTrainer`` checkpoints loaded through
``load_inference_checkpoint``.  Planning tests are model-free.
"""
from __future__ import annotations

import copy
import hashlib
import random
import re
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from smallAntibodyGen.experiments.trastuzumab_poc import checkpoint as ck
from smallAntibodyGen.experiments.trastuzumab_poc import cli
from smallAntibodyGen.experiments.trastuzumab_poc import functional_data as fd
from smallAntibodyGen.experiments.trastuzumab_poc import functional_protocol as fp
from smallAntibodyGen.experiments.trastuzumab_poc import inputs, retention
from smallAntibodyGen.experiments.trastuzumab_poc.artifacts import digest, read_json
from smallAntibodyGen.experiments.trastuzumab_poc.data import GroupingDeclaration, qualify_manifest
from smallAntibodyGen.experiments.trastuzumab_poc.objectives import UniformMaskSpec, draw_mask_uniform
from smallAntibodyGen.experiments.trastuzumab_poc.parent import export_parent, load_parent
from smallAntibodyGen.experiments.trastuzumab_poc.post_budget import selection_key
from smallAntibodyGen.experiments.trastuzumab_poc.protocol import (
    EndpointConstruct, EndpointEntry, EndpointManifest,
)
from smallAntibodyGen.experiments.trastuzumab_poc.randomness import keyed_seed
from smallAntibodyGen.experiments.trastuzumab_poc.runner import StageTrainer, TrainingConfig
from smallAntibodyGen.experiments.trastuzumab_poc.transfer_guard import qualify_transfer
from smallAntibodyGen.models.antibody_antigen.dynamic_context import DynamicContextProvider
from smallAntibodyGen.tests.test_trastuzumab_cli import mid_config, run, sha, write
from smallAntibodyGen.tests.test_trastuzumab_exports import _panel_document
from smallAntibodyGen.tests.test_trastuzumab_functional_data import (
    AG, AG2, KD, broad, construct_row, declaration, group_row, heavy, imported_names, qualify,
)
from smallAntibodyGen.tests.test_trastuzumab_functional_runner import (
    SMALL_VALUES, build_trainer, fresh_models, functional_run, small_document, training_config,
)
from smallAntibodyGen.tests.test_trastuzumab_functional_transfer import (
    FAMILIES, SOURCE, endpoint_manifest, post_data, post_fusion,
)
from smallAntibodyGen.tests.test_trastuzumab_runner import equal

AG3 = "GSGSGSGSGS"
AG4 = "WVWVWVWVWV"
FILES = {"input_sha256": "1" * 64, "endpoint_manifest_sha256": "2" * 64, "functional_constructs_sha256": None}
# What ``load_broad_context`` puts in every command-line midtraining identity; the trainer adds budget/functional.
INPUT_IDENTITY = {"input_sha256": FILES["input_sha256"]}
POST_CONFIG = TrainingConfig(stage="post", arm="paired", seed=19, total_updates=1, examples_per_update=1,
                             learning_rate=0.001, ordering_weight=1.0, minimum_ordering_endpoint_coverage=0.0)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


# ---------------------------------------------------------------------------
# Worlds
# ---------------------------------------------------------------------------

def world_records():
    """Three training rows and three held-out rows: paired, paired with ONE eligible site, heavy-only."""
    return (
        broad("t0", heavy(0)),
        broad("t1", heavy(10)),
        broad("t2", heavy(11), family="famB"),
        broad("h0", heavy(12), Ag=AG2, family="famH0", target="tgt-h0", partition="val"),
        replace(broad("h1", heavy(13), Ag=AG3, family="famH1", target="tgt-h1", partition="test"),
                h_eligible=(0,), l_eligible=None),
        broad("hv", heavy(14), L=None, Ag=AG2, family="famHV", target="tgt-h0", partition="val"),
    )


def spec_document(*, selection=None, masks=None, protected=None, functional_path=None):
    document = {
        "schema": retention.SPEC_SCHEMA, "pairs_path": "pairs.json", "functional_constructs_path": functional_path,
        "protected": {"endpoint_manifest_path": "endpoints.json",
                      "excluded_antibody_families": list(FAMILIES["excluded_antibody_families"]),
                      "excluded_target_families": list(FAMILIES["excluded_target_families"]),
                      "family_axes": ["antibody_family", "target_family"]},
        "selection": {"partitions": ["val", "test"], "record_ids": None, "limit": None},
        "masks": {"seed": 5, "per_record": 3}}
    document["selection"].update(selection or {})
    document["masks"].update(masks or {})
    document["protected"].update(protected or {})
    return document


def make_spec(**changes):
    return retention.parse_spec(spec_document(**changes))


def functional_doc(assertion=None):
    document = small_document(declaration())
    if assertion is not None:
        document["study_contract"]["assertion"] = assertion
    return fd.parse_functional_constructs(document)


def build(records=None, *, decl=None, spec=None, functional=None, manifest=None):
    decl = declaration() if decl is None else decl
    records = world_records() if records is None else tuple(records)
    files = {**FILES, "functional_constructs_sha256": None if functional is None else "3" * 64}
    return retention.build_plan(spec or make_spec(), declaration=decl, records=records,
                                population=qualify_manifest(decl, records),
                                manifest=manifest or endpoint_manifest(), functional=functional, files=files)


def seal(plan):
    plan["plan_sha256"] = digest({k: v for k, v in plan.items() if k != "plan_sha256"})
    return plan


def evaluate(item, plan, arm="paired"):
    return retention.evaluate_retention(item.policy, item.encoder, plan, arm=arm, provenance=item.provenance)


def mask_sets(plan):
    return [(case, mask, {tuple(site) for site in mask}) for case in plan["cases"] for mask in case["masks"]]


# ---------------------------------------------------------------------------
# Trained checkpoints: mid with and without the functional arm, post from each, direct-release post
# ---------------------------------------------------------------------------

def loaded(saved, encoder, make_policy):
    policy = make_policy()
    provenance = ck.load_inference_checkpoint(policy, saved["path"], encoder=encoder,
                                              expected_sha256=saved["sha256"], source_identity=SOURCE)
    return SimpleNamespace(policy=policy, encoder=encoder, provenance=provenance)


@pytest.fixture(scope="module")
def lineage(tmp_path_factory):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        directory = tmp_path_factory.mktemp("retention")
        decl, records = declaration(), world_records()
        population = qualify_manifest(decl, records)
        qualified = qualify(small_document(decl), decl, records, population=population)
        manifest = endpoint_manifest()
        world = SimpleNamespace(mid={}, post={})
        for aux in (False, True):
            certificate = qualify_transfer(population, manifest, functional=qualified if aux else None, **FAMILIES)
            base, encoder = fresh_models("lora")
            mid = build_trainer(base, encoder, population, training_config(total_updates=2),
                                functional=functional_run(qualified, SMALL_VALUES["ordered"]) if aux else None,
                                certificate=certificate, data_identity=INPUT_IDENTITY)
            if not aux:
                world.update0 = loaded(ck.save_checkpoint(mid, directory / "mid-0.pt"), encoder,
                                       lambda: fresh_models("lora")[0])
            mid.run()
            world.mid[aux] = loaded(ck.save_checkpoint(mid, directory / f"mid-{aux}.pt"), encoder,
                                    lambda: fresh_models("lora")[0])
            exported = export_parent(world.mid[aux].policy, directory / f"parent-{aux}.pt",
                                     provenance=world.mid[aux].provenance)

            def parent(aux=aux, exported=exported, encoder=encoder):
                return load_parent(directory / f"parent-{aux}.pt", expected_sha256=exported["sha256"],
                                   encoder_identity=encoder.encoder_identity, fusion_config=post_fusion(),
                                   device="cpu", seed=5)

            policy = parent()
            post = StageTrainer(policy, encoder, POST_CONFIG, post_data=post_data(manifest),
                                data_identity=SOURCE, source_identity=SOURCE,
                                initialization_identity={"kind": "parent", "sha256": exported["sha256"],
                                                         "base": policy.base_identity})
            post.run()
            world.post[aux] = loaded(ck.save_checkpoint(post, directory / f"post-{aux}.pt"), encoder, parent)
        base, encoder = fresh_models("lora")
        direct = StageTrainer(base, encoder, POST_CONFIG, post_data=post_data(manifest), data_identity=SOURCE,
                              source_identity=SOURCE,
                              initialization_identity={"kind": "release", "base": base.base_identity})
        direct.run()
        world.direct = loaded(ck.save_checkpoint(direct, directory / "direct.pt"), encoder,
                              lambda: fresh_models("lora")[0])
        return world
    finally:
        torch.set_num_threads(previous)


# ---------------------------------------------------------------------------
# The specification
# ---------------------------------------------------------------------------

SPEC_PROBLEMS = {
    "schema": lambda d: d.update(schema="trastuzumab-retention-spec/2"),
    "extra_key": lambda d: d.update(outcome=1.0),
    "missing_key": lambda d: d.pop("masks"),
    "bool_seed": lambda d: d["masks"].update(seed=True),
    "zero_masks": lambda d: d["masks"].update(per_record=0),
    "train_partition": lambda d: d["selection"].update(partitions=["train"]),
    "empty_partitions": lambda d: d["selection"].update(partitions=[]),
    "duplicate_partition": lambda d: d["selection"].update(partitions=["val", "val"]),
    "ids_and_limit": lambda d: d["selection"].update(record_ids=["h0"], limit={"count": 1, "seed": "s"}),
    "duplicate_ids": lambda d: d["selection"].update(record_ids=["h0", "h0"]),
    "empty_ids": lambda d: d["selection"].update(record_ids=[]),
    "limit_extra_key": lambda d: d["selection"].update(limit={"count": 1, "seed": "s", "rank": 1}),
    "limit_zero": lambda d: d["selection"].update(limit={"count": 0, "seed": "s"}),
    "no_axes": lambda d: d["protected"].update(family_axes=[]),
    "unknown_axis": lambda d: d["protected"].update(family_axes=["species"]),
    "no_protected_families": lambda d: d["protected"].update(excluded_target_families=[]),
    "empty_path": lambda d: d.update(pairs_path=""),
}


@pytest.mark.parametrize("problem", sorted(SPEC_PROBLEMS))
def test_the_specification_is_strict(problem):
    document = spec_document()
    SPEC_PROBLEMS[problem](document)
    with pytest.raises((ValueError, TypeError)):
        retention.parse_spec(document)


def test_the_example_specification_parses_under_the_current_schema():
    path = Path(__file__).resolve().parents[3] / "configs" / "trastuzumab_lowresource" / "retention" / "heldout_spec.json"
    spec = retention.parse_spec(read_json(path))
    assert spec["family_axes"] == ["antibody_family", "target_family"] and spec["selection"]["limit"] is None


# ---------------------------------------------------------------------------
# Planning: selection, determinism, literal masks
# ---------------------------------------------------------------------------

def test_the_plan_is_repeatable_and_freezes_literal_cases_with_independently_derived_masks():
    plan = build()
    assert plan == build() and retention.validate_plan(plan) == plan
    assert [case["record_id"] for case in plan["cases"]] == ["h0", "h1", "hv"]
    assert plan["selection"]["pool"] == {"test": 1, "val": 2} and plan["selection"]["selected"] == 3
    by_id = {r.record_id: r for r in world_records()}
    for case in plan["cases"]:
        record = by_id[case["record_id"]]
        assert (case["H"], case["L"], case["Ag"]) == (record.H, record.L, record.Ag)
        assert (case["antibody_family"], case["target_family"]) == (record.antibody_family, record.target_family)
        union = [("H", i) for i in record.h_eligible] + [("L", i) for i in record.l_eligible or ()]
        assert case["eligible"] == [list(c) for c in union]
        expected = []
        for index in range(3):
            rng = random.Random(keyed_seed(5, index, 0, "heldout-mask/" + case["record_id"]))
            expected.append([list(s) for s in draw_mask_uniform(union, rng, UniformMaskSpec()).sites])
        assert case["masks"] == expected
    heavy_only = plan["cases"][2]
    assert heavy_only["L"] is None and heavy_only["antibody_format"] == "heavy_only"
    assert all(role == "H" for mask in heavy_only["masks"] for role, _ in mask)
    assert plan["counts"] == {"records": 3, "masks": 9,
                              "masked_residues": sum(len(m) for c in plan["cases"] for m in c["masks"])}
    assert plan["plan_sha256"] == digest({k: v for k, v in plan.items() if k != "plan_sha256"})
    assert build(spec=make_spec(masks={"seed": 6}))["plan_sha256"] != plan["plan_sha256"]


def test_the_plan_binds_the_declaration_population_raw_input_manifest_and_families():
    decl, records = declaration(), world_records()
    population = qualify_manifest(decl, records)
    plan = build(functional=functional_doc())
    qualified = qualify(small_document(decl), decl, records, population=population)
    assert plan["broad"]["declaration_digest"] == population.manifest_digest
    assert plan["broad"]["population_digest"] == population.population_digest
    assert plan["broad"]["raw_input_digest"] == qualified.broad_input_digest and plan["broad"]["raw_rows"] == 6
    assert plan["broad"]["input_sha256"] == FILES["input_sha256"]
    assert plan["functional"] == {
        "constructs_sha256": "3" * 64, "qualified_digest": qualified.digest, "contract_digest": qualified.contract_digest,
        "broad_input_digest": qualified.broad_input_digest, "declaration_digest": population.manifest_digest,
        "population_digest": population.population_digest, "train_constructs": 2}
    assert plan["protected"]["endpoint_manifest_digest"] == endpoint_manifest().manifest_digest()
    assert plan["protected"]["excluded_target_families"] == ["heldout-target"]
    assert plan["guard"]["family_axes"] == ["antibody_family", "target_family"]
    assert plan["guard"]["against"] == {"denoising_train_raw_rows": 3, "functional_train_constructs": 2,
                                        "protected_endpoint_entries": 3}
    assert "not binding retention" in plan["scope"] and plan["assumptions"]
    # Any bound identity changes the digest.
    other = build(functional=functional_doc("another curator claim"))
    assert other["functional"]["qualified_digest"] != plan["functional"]["qualified_digest"]
    assert other["plan_sha256"] != plan["plan_sha256"]
    assert build()["functional"] is None and build()["plan_sha256"] != plan["plan_sha256"]


def test_a_limit_or_explicit_ids_select_deterministically_and_leave_each_cases_masks_unchanged():
    full = build()
    spec = make_spec(selection={"limit": {"count": 2, "seed": "s1"}})
    limited = build(spec=spec)
    assert limited == build(spec=spec) and limited["selection"]["limit"] == {"count": 2, "seed": "s1"}
    expected = sorted(sorted(("h0", "h1", "hv"), key=lambda i: selection_key("s1", i))[:2])
    assert [c["record_id"] for c in limited["cases"]] == expected
    explicit = build(spec=make_spec(selection={"record_ids": ["hv", "h0"]}))
    assert [c["record_id"] for c in explicit["cases"]] == ["h0", "hv"]
    assert explicit["cases"][0] == full["cases"][0] and explicit["selection"]["rule"] != full["selection"]["rule"]
    assert [c["record_id"] for c in build(spec=make_spec(selection={"partitions": ["val"]}))["cases"]] == ["h0", "hv"]
    with pytest.raises(ValueError, match="limit.count exceeds"):
        build(spec=make_spec(selection={"limit": {"count": 4, "seed": "s1"}}))


def test_an_empty_selection_or_an_absent_training_population_is_refused():
    with pytest.raises(ValueError, match="no record in the requested held-out partitions"):
        build([r for r in world_records() if r.partition == "train"])
    with pytest.raises(ValueError, match="training population"):
        build([r for r in world_records() if r.partition != "train"])


def test_excluded_duplicate_and_training_rows_cannot_be_reintroduced():
    decl = declaration(excluded_record_ids=("x",))
    excluded = broad("x", heavy(15), Ag=AG3, family="famX", target="tgt-h1", partition="val")
    duplicate = broad("h0-dup", heavy(12), Ag=AG2, family="famH0", target="tgt-h0", partition="val",
                      source_record_id="zz")
    records = [*world_records(), excluded, duplicate]
    assert [c["record_id"] for c in build(records, decl=decl)["cases"]] == ["h0", "h1", "hv"]
    for record_id in ("x", "h0-dup", "t0"):
        with pytest.raises(ValueError, match="never selectable"):
            build(records, decl=decl, spec=make_spec(selection={"record_ids": [record_id]}))


def test_an_excluded_raw_training_row_still_blocks_an_overlapping_heldout_record():
    decl = declaration(excluded_record_ids=("tx",))
    hidden_train = broad("tx", heavy(16), Ag=AG3, family="famTX", target="tgt-h1", partition="train")
    assert "tx" not in {r.record_id for r in qualify_manifest(decl, [*world_records(), hidden_train]).records}
    with pytest.raises(ValueError, match="h1 shares its antigen with a denoising training row"):
        build([*world_records(), hidden_train], decl=decl)


# ---------------------------------------------------------------------------
# Planning: the disjointness guard
# ---------------------------------------------------------------------------

def target_only():
    return declaration(grouping=GroupingDeclaration(separation_axes=("target_family",)))


def target_only_document(decl):
    """``small_document`` with its val constructs moved off the training target, which a target axis requires."""
    document = small_document(decl)
    for row in document["constructs"]:
        if row["partition"] == "val":
            row.update(Ag=AG2, target_family="tgt-h0")
    for group in document["groups"]:
        if group["group_id"] == "g-val":
            group["antigen_sha256"] = hashlib.sha256(AG2.encode("ascii")).hexdigest()
    return document


def with_construct(document, row, group):
    document["constructs"].append(row)
    document["groups"].append(group)
    return fd.parse_functional_constructs(document)


def overlap_world(name):
    """``(declaration, records, functional document, message)`` with exactly one overlap built in."""
    decl, records, functional = declaration(), list(world_records()), None
    if name == "protected_antibody_other_antigen":
        records.append(broad("p", "ACACDEFGHIKLDE", L="FG", Ag=AG3, family="famP", target="tgt-h1", partition="val"))
        message = "shares its (H, L) with a protected endpoint"
    elif name == "protected_antigen_other_antibody":
        records.append(broad("p", heavy(15), Ag="HIK", family="famP", target="tgt-p", partition="val"))
        message = "shares its antigen with a protected endpoint"
    elif name == "protected_antibody_family":
        records.append(broad("p", heavy(15), Ag=AG4, family="heldout-ab", target="tgt-p", partition="val"))
        message = "is in a protected antibody family"
    elif name == "protected_target_family":
        records.append(broad("p", heavy(15), Ag=AG4, family="famP", target="heldout-target", partition="val"))
        message = "is in a protected target family"
    elif name == "train_antibody_other_antigen":
        decl = target_only()          # the declared separation axis alone would let this through
        records.append(broad("p", heavy(0), Ag=AG4, family="famF", target="tgt-p", partition="val"))
        message = "shares its (H, L) with a denoising training row"
    elif name == "train_antigen_other_antibody":
        records.append(broad("p", heavy(15), Ag=AG, family="famP", target="her2", partition="val"))
        message = "shares its antigen with a denoising training row"
    elif name == "train_target_family":
        records.append(broad("p", heavy(15), Ag=AG4, family="famP", target="her2", partition="val"))
        message = "shares its target_family with a denoising training row"
    elif name == "train_antibody_family":
        decl = target_only()
        records.append(broad("p", heavy(15), Ag=AG4, family="famF", target="tgt-p", partition="val"))
        message = "shares its antibody_family with a denoising training row"
    elif name == "functional_antibody_other_antigen":
        decl = target_only()
        functional = with_construct(target_only_document(decl),
                                    construct_row("z", heavy(12), Ag=AG4, family="famH0", target="tgt-z"),
                                    group_row("g-z", "ctx-a", KD, ["z"], Ag=AG4))
        message = "shares its (H, L) with a functional training construct"
    elif name == "functional_antigen_other_antibody":
        functional = with_construct(small_document(decl),
                                    construct_row("z", heavy(15), Ag=AG2, family="famZ", target="tgt-h0"),
                                    group_row("g-z", "ctx-a", KD, ["z"], Ag=AG2))
        message = "shares its antigen with a functional training construct"
    else:
        functional = with_construct(small_document(decl),
                                    construct_row("z", heavy(15), Ag=AG4, family="famZ", target="tgt-h0"),
                                    group_row("g-z", "ctx-a", KD, ["z"], Ag=AG4))
        message = "shares its target_family with a functional training construct"
    return decl, records, functional, message


OVERLAPS = ["protected_antibody_other_antigen", "protected_antigen_other_antibody", "protected_antibody_family",
            "protected_target_family", "train_antibody_other_antigen", "train_antigen_other_antibody",
            "train_target_family", "train_antibody_family", "functional_antibody_other_antigen",
            "functional_antigen_other_antibody", "functional_target_family"]


@pytest.mark.parametrize("name", OVERLAPS)
def test_every_overlap_axis_is_rejected_not_only_the_complete_triple(name):
    decl, records, functional, message = overlap_world(name)
    with pytest.raises(ValueError, match=re.escape(message)):
        build(records, decl=decl, functional=functional)
    # The same world without the overlapping row or construct plans fine (the rejection is the overlap).
    assert build(decl=decl)["counts"]["records"] == 3


@pytest.mark.parametrize("name", [n for n in OVERLAPS if n.startswith("functional")])
def test_a_functional_training_construct_blocks_the_plan_even_when_no_budget_selects_it(name):
    decl, records, functional, message = overlap_world(name)
    qualified = fd.qualify_functional(functional, decl, tuple(records), population=qualify_manifest(decl, records))
    seed = next(s for s in (f"s{i}" for i in range(120))
                if fp.select_functional_budget(qualified, 2, s).selected_set == {("g-x", "a"), ("g-x", "b")})
    assert ("g-z", "z") not in fp.select_functional_budget(qualified, 2, seed).selected_set
    with pytest.raises(ValueError, match=re.escape(message)):
        build(records, decl=decl, functional=functional)


def test_the_declared_family_axes_are_what_is_checked_and_what_is_reported():
    decl, records, _, _ = overlap_world("train_target_family")
    with pytest.raises(ValueError, match="target_family"):
        build(records, decl=decl)
    relaxed = build(records, decl=decl, spec=make_spec(protected={"family_axes": ["antibody_family"]}))
    assert relaxed["guard"]["family_axes"] == ["antibody_family"] and relaxed["counts"]["records"] == 4
    assert "not independently established homology" in relaxed["guard"]["assertion"]


@pytest.mark.parametrize("index", range(3))
def test_every_endpoint_role_protects_its_construct(index):
    cores, roles = ("ACDEFGHIKL", "LMNPQRSTVW", "AAAAAAAAAA"), ("post_pool", "development", "reserved")
    manifest = EndpointManifest(tuple(
        EndpointEntry(str(i), EndpointConstruct("AC", core, "DE", "FG", "HIK"), role, f"n{i}", "assay", "synthetic")
        for i, (core, role) in enumerate(zip(cores, roles))))
    row = broad("p", "AC" + cores[index] + "DE", L="FG", Ag=AG3, family="famP", target="tgt-h1", partition="val")
    with pytest.raises(ValueError, match=re.escape("shares its (H, L) with a protected endpoint")):
        build([*world_records(), row], manifest=manifest)
    assert build(manifest=manifest)["protected"]["roles"] == ["development", "post_pool", "reserved"]


def test_a_full_raw_cross_manifest_contradiction_is_caught_even_for_an_excluded_broad_row():
    decl = declaration(excluded_record_ids=("x",))
    excluded = broad("x", heavy(15), Ag=AG3, family="famX", target="tgt-h1", partition="val")
    functional = with_construct(small_document(decl),
                                construct_row("z", heavy(15), Ag=AG3, family="famOther", target="tgt-h1"),
                                group_row("g-z", "ctx-a", KD, ["z"], Ag=AG3))
    with pytest.raises(ValueError, match="disagrees with the broad manifest"):
        build([*world_records(), excluded], decl=decl, functional=functional)


def test_the_population_is_not_trusted():
    decl, records = declaration(), world_records()
    with pytest.raises(ValueError, match="not the qualification of the raw records"):
        retention.build_plan(make_spec(), declaration=decl, records=records[:-1],
                             population=qualify_manifest(decl, records), manifest=endpoint_manifest(),
                             files=FILES)


# ---------------------------------------------------------------------------
# Plan validation precedes every forward
# ---------------------------------------------------------------------------

def forge_mask(plan):
    case = plan["cases"][0]
    original = case["masks"][0]
    case["masks"][0] = next([c] for c in case["eligible"] if [c] != original)


def add_case(plan, record_id):
    """A copy of the first case under another record id, with its masks re-derived and every count re-aligned.

    Only the duplicated physical content is left contradictory, so a rejection is about that and nothing else.
    """
    case = copy.deepcopy(plan["cases"][0])
    case["record_id"] = record_id
    case["masks"] = retention.draw_masks(record_id, [tuple(c) for c in case["eligible"]], plan["masks"]["seed"],
                                         plan["masks"]["per_record"])
    plan["cases"].insert(1, case)                  # "h0" < "h0b" < "h1": the cases stay sorted
    plan["selection"]["selected"] += 1
    plan["selection"]["pool"][case["partition"]] += 1
    plan["broad"]["raw_rows"] += 1
    plan["counts"] = retention._counts(plan["cases"])


def three_roles_two_entries(plan):
    plan["protected"].update(roles=["development", "post_pool", "reserved"], entries=2)
    plan["guard"]["against"].update(protected_endpoint_entries=2)


TAMPERED = {
    "unsealed_mask": (lambda p: p["cases"][0]["masks"][0].__setitem__(0, ["H", 99]), False, "digest mismatch"),
    "unsealed_extra_key": (lambda p: p.update(outcome=1.0), False, "expected exactly"),
    "ineligible_coordinate": (lambda p: p["cases"][0]["masks"][0].__setitem__(0, ["H", 99]), True, "not eligible"),
    "duplicate_coordinate": (lambda p: p["cases"][0]["masks"][0].append(list(p["cases"][0]["masks"][0][0])),
                             True, "duplicate"),
    "forged_but_valid_mask": (forge_mask, True, "not the recipe's masks"),
    "edited_sequence": (lambda p: p["cases"][0].update(H=p["cases"][0]["H"][::-1]), True, "content key"),
    "case_extra_key": (lambda p: p["cases"][0].update(outcome=1.0), True, "retention case"),
    "counts": (lambda p: p["counts"].update(masks=p["counts"]["masks"] + 1), True, "counts disagree"),
    "case_order": (lambda p: p["cases"].reverse(), True, "sorted by record_id"),
    "recipe": (lambda p: p["masks"]["recipe"].update(kind="other"), True, "mask recipe"),
    "protected_families": (lambda p: p["protected"].update(excluded_target_families=[]), True, "excluded_target_families"),
    "functional_mismatch": (lambda p: p["guard"]["against"].update(functional_train_constructs=2), True, "disagrees"),
    # A re-sealed plan whose declared state contradicts itself (the digest is valid; the content is not).
    "pool_below_selected": (lambda p: p["selection"]["pool"].update(val=1), True, "more records than its qualified pool"),
    "all_selection_below_pool": (lambda p: (p["selection"]["pool"].update(test=2), p["broad"].update(raw_rows=99)),
                                 True, "whole pool of each partition"),
    "same_content_other_id": (lambda p: add_case(p, "h0b"), True, "same physical"),
    "protected_antibody_family_case": (lambda p: p["cases"][1].update(
        antibody_family=p["protected"]["excluded_antibody_families"][0]), True, "protected family"),
    "protected_target_family_case": (lambda p: p["cases"][2].update(
        target_family=p["protected"]["excluded_target_families"][0]), True, "protected family"),
    "no_roles_with_entries": (lambda p: p["protected"].update(roles=[]), True, "protected roles disagree"),
    "more_roles_than_entries": (three_roles_two_entries, True, "protected roles disagree"),
    "fewer_raw_rows_than_accounted": (lambda p: p["broad"].update(raw_rows=5), True, "raw_rows is smaller"),
}


@pytest.mark.parametrize("problem", sorted(TAMPERED))
def test_tampering_is_rejected_before_any_model_forward(lineage, monkeypatch, problem):
    change, reseal, message = TAMPERED[problem]
    item = lineage.mid[False]
    policy, encoder = copy.deepcopy(item.policy), copy.deepcopy(item.encoder)
    forwards = []
    monkeypatch.setattr(policy, "reconstruction_nll", lambda *a, **k: forwards.append("policy"))
    monkeypatch.setattr(encoder, "extract_exact", lambda *a, **k: forwards.append("encoder"))
    plan = copy.deepcopy(build())
    change(plan)
    if reseal:
        seal(plan)
    with pytest.raises(ValueError, match=message):
        retention.validate_plan(plan)
    with pytest.raises(ValueError, match=message):
        retention.evaluate_retention(policy, encoder, plan, arm="paired", provenance=item.provenance)
    assert forwards == []


# ---------------------------------------------------------------------------
# The readout
# ---------------------------------------------------------------------------

def test_every_mask_is_scored_with_a_fresh_provider_on_the_current_masked_view_for_both_arms(lineage, monkeypatch):
    item, plan = lineage.mid[False], build()
    outcomes = {}
    for arm in ("paired", "abonly"):
        policy, encoder = copy.deepcopy(item.policy), copy.deepcopy(item.encoder)
        calls, real_nll, real_extract = [], policy.reconstruction_nll, encoder.extract_exact

        def nll(views, contexts, targets, calls=calls, real=real_nll):
            assert len(views) == len(contexts) == len(targets) == 1
            calls.append(("policy", views[0], contexts[0], targets[0]))
            return real(views, contexts, targets)

        def extract(view, calls=calls, real=real_extract):
            calls.append(("encoder", view))
            return real(view)

        monkeypatch.setattr(policy, "reconstruction_nll", nll)
        monkeypatch.setattr(encoder, "extract_exact", extract)
        result = retention.evaluate_retention(policy, encoder, plan, arm=arm, provenance=item.provenance)
        outcomes[arm] = result
        policy_calls = [c for c in calls if c[0] == "policy"]
        encoder_calls = [c for c in calls if c[0] == "encoder"]
        assert result["status"] == "complete" and len(policy_calls) == len(encoder_calls) == 9
        assert len({id(c[2]) for c in policy_calls}) == 9           # one provider per mask, never reused
        for (_, query, provider, target), (_, seen), (case, mask, sites) in zip(policy_calls, encoder_calls,
                                                                                mask_sets(plan)):
            assert isinstance(provider, DynamicContextProvider)
            assert set(target.coords()) == set(query.hidden_sites()) == set(provider.hidden_union) == sites
            assert query.roles == (("H",) if case["L"] is None else ("H", "L"))
            # The encoder sees the CURRENT masked view: gold is absent, and the antigen only in the paired arm.
            assert seen.roles == query.roles + (("Ag",) if arm == "paired" else ())
            assert set(seen.hidden_sites()) == sites and all(seen.token(site) is None for site in sites)
    assert outcomes["paired"]["aggregate"] != outcomes["abonly"]["aggregate"]
    assert outcomes["paired"]["plan"] == outcomes["abonly"]["plan"] and outcomes["abonly"]["arm"] == "abonly"


def test_heavy_only_and_repeated_masks_are_evaluated_and_counted(lineage):
    result = evaluate(lineage.mid[False], build())
    by_id = {case["record_id"]: case for case in result["cases"]}
    assert result["status"] == "complete" and result["counts"] == {
        "records": 3, "masks": 9, "masked_residues": result["counts"]["masked_residues"], "failed_masks": 0}
    # h1 has ONE eligible site, so its three masks are identical draws and all three are scored.
    assert [m["sites"] for m in by_id["h1"]["masks"]] == [1, 1, 1]
    assert all(m["nll"] is not None and m["failure"] is None for c in result["cases"] for m in c["masks"])
    assert len(by_id["hv"]["masks"]) == 3 and by_id["hv"]["mean_nll"] is not None


def test_the_aggregate_is_the_mean_of_mask_means_per_record_then_the_mean_of_records(lineage, monkeypatch):
    item, plan = lineage.mid[False], build()
    policy, encoder = copy.deepcopy(item.policy), copy.deepcopy(item.encoder)
    monkeypatch.setattr(policy, "reconstruction_nll",
                        lambda views, contexts, targets: torch.tensor(float(len(targets[0])) ** 2))
    result = retention.evaluate_retention(policy, encoder, plan, arm="paired", provenance=item.provenance)
    record_means = [sum(len(m) ** 2 for m in case["masks"]) / len(case["masks"]) for case in plan["cases"]]
    assert result["aggregate"]["mean_record_nll"] == pytest.approx(sum(record_means) / len(record_means), rel=1e-12)
    assert [c["mean_nll"] for c in result["cases"]] == pytest.approx(record_means, rel=1e-12)
    assert result["counts"]["masked_residues"] == sum(len(m) for c in plan["cases"] for m in c["masks"])


def test_summarize_has_a_known_answer_for_unequal_masks_and_sites_per_record():
    plan = {"cases": [{"record_id": "a", "masks": [[["H", 0]]]},
                      {"record_id": "b", "masks": [[["H", 0]], [["H", 0], ["H", 1]], [["H", 0], ["H", 1], ["H", 2]]]}]}
    cases, aggregate, failed = retention.summarize(plan, [(1.0, None), (3.0, None), (5.0, None), (7.0, None)])
    # Records weigh equally: (1 + (3 + 5 + 7) / 3) / 2 = 3, not the pooled masks (4) nor the site-weighted 5.
    assert [c["mean_nll"] for c in cases] == [1.0, 5.0] and aggregate == {"mean_record_nll": 3.0} and failed == 0
    assert [m["sites"] for m in cases[1]["masks"]] == [1, 2, 3]
    cases, aggregate, failed = retention.summarize(plan, [(1.0, None), (3.0, None), (None, "FloatingPointError: x"),
                                                          (7.0, None)])
    assert aggregate is None and failed == 1
    assert cases[0]["mean_nll"] == 1.0 and cases[1]["mean_nll"] is None
    assert [m["nll"] for m in cases[1]["masks"]] == [3.0, None, 7.0]


def test_a_numerical_or_memory_failure_is_recorded_and_leaves_no_complete_aggregate(lineage, monkeypatch):
    item, plan = lineage.mid[False], build()
    policy, encoder = copy.deepcopy(item.policy), copy.deepcopy(item.encoder)
    state = {"calls": 0}

    def flaky(views, contexts, targets):
        state["calls"] += 1
        if state["calls"] == 2:
            raise FloatingPointError("injected")
        if state["calls"] == 4:
            raise torch.cuda.OutOfMemoryError("injected out of memory")
        if state["calls"] == 5:
            return torch.tensor(float("nan"))
        return torch.tensor(1.5)

    monkeypatch.setattr(policy, "reconstruction_nll", flaky)
    result = retention.evaluate_retention(policy, encoder, plan, arm="paired", provenance=item.provenance)
    assert result["status"] == "incomplete" and result["aggregate"] is None
    assert result["counts"]["failed_masks"] == 3 and result["counts"]["masks"] == 9
    masks = [m for c in result["cases"] for m in c["masks"]]
    assert len(masks) == 9 and [m["nll"] is None for m in masks].count(True) == 3
    assert [m["failure"].split(":")[0] for m in masks if m["failure"]] == [
        "FloatingPointError", "OutOfMemoryError", "FloatingPointError"]
    assert "nonfinite" in masks[4]["failure"]
    assert [c["mean_nll"] for c in result["cases"]] == [None, None, 1.5]      # only a fully scored record has a mean


def test_a_structural_error_aborts_instead_of_becoming_an_omitted_record(lineage, monkeypatch):
    item, plan = lineage.mid[False], build()
    policy, encoder = copy.deepcopy(item.policy), copy.deepcopy(item.encoder)

    def broken(views, contexts, targets):
        raise RuntimeError("structural")

    monkeypatch.setattr(policy, "reconstruction_nll", broken)
    with pytest.raises(RuntimeError, match="structural"):
        retention.evaluate_retention(policy, encoder, plan, arm="paired", provenance=item.provenance)


def disturb(policy, encoder):
    """Mixed module flags, stale gradients and seeded generators; returns what evaluation must give back."""
    for index, module in enumerate(list(policy.modules())):
        module.training = index % 3 != 1
    for index, module in enumerate(list(encoder.modules())):
        module.training = index % 2 == 0
    for index, parameter in enumerate(p for p in policy.parameters() if p.requires_grad):
        parameter.grad = torch.full_like(parameter, 0.25 + index)
    random.seed(11)
    torch.manual_seed(12)
    return SimpleNamespace(
        flags=[(module, module.training) for module in (*policy.modules(), *encoder.modules())],
        values=[(p, p.detach().clone(), p.requires_grad, None if p.grad is None else p.grad.clone())
                for p in (*policy.parameters(), *encoder.parameters())],
        python=random.getstate(), torch=torch.get_rng_state())


def assert_undisturbed(before):
    assert {flag for _, flag in before.flags} == {True, False}                 # the disturbance was real
    assert all(module.training is flag for module, flag in before.flags)
    assert random.getstate() == before.python and torch.equal(torch.get_rng_state(), before.torch)
    assert torch.is_grad_enabled()
    for parameter, value, requires_grad, grad in before.values:
        assert torch.equal(parameter.detach(), value) and parameter.requires_grad is requires_grad
        assert parameter.grad is None if grad is None else torch.equal(parameter.grad, grad)


def poison_encoder(encoder, monkeypatch, calls=None):
    """The REAL frozen encoder returns NaN representations, on the given 1-based forwards (default: all)."""
    original, layer, state = encoder.model.forward, encoder.model.num_layers, {"calls": 0}

    def poisoned(*args, **kwargs):
        state["calls"] += 1
        output = original(*args, **kwargs)
        if calls is None or state["calls"] in calls:
            output["representations"][layer] = torch.full_like(output["representations"][layer], float("nan"))
        return output

    monkeypatch.setattr(encoder.model, "forward", poisoned)
    return state


def test_nonfinite_encoder_representations_fail_only_their_masks_and_leave_no_aggregate(lineage, monkeypatch):
    item, plan = lineage.mid[False], build()
    policy, encoder = copy.deepcopy(item.policy), copy.deepcopy(item.encoder)
    clean = retention.evaluate_retention(policy, encoder, plan, arm="paired", provenance=item.provenance)
    before = disturb(policy, encoder)
    forwards = poison_encoder(encoder, monkeypatch, calls={2, 5})
    result = retention.evaluate_retention(policy, encoder, plan, arm="paired", provenance=item.provenance)
    assert forwards["calls"] == 9                                              # one encoder forward per planned mask
    assert result["status"] == "incomplete" and result["aggregate"] is None
    assert result["counts"]["failed_masks"] == 2 and result["counts"]["masks"] == 9
    masks = [m for c in result["cases"] for m in c["masks"]]
    clean_masks = [m for c in clean["cases"] for m in c["masks"]]
    assert [i for i, m in enumerate(masks) if m["nll"] is None] == [1, 4]
    assert {m["failure"] for m in masks if m["failure"]} == {"ValueError: features must all be finite"}
    # The other masks are scored exactly as in the clean run: a failed mask leaves no residue behind.
    assert [m for i, m in enumerate(masks) if i not in (1, 4)] == [m for i, m in enumerate(clean_masks) if i not in (1, 4)]
    assert [c["mean_nll"] for c in result["cases"]] == [None, None, clean["cases"][2]["mean_nll"]]
    assert_undisturbed(before)


def test_every_mask_nonfinite_is_an_incomplete_result_that_changes_nothing_afterwards(lineage, monkeypatch):
    item, plan = lineage.mid[False], build()
    policy, encoder = copy.deepcopy(item.policy), copy.deepcopy(item.encoder)
    clean = retention.evaluate_retention(policy, encoder, plan, arm="paired", provenance=item.provenance)
    before = disturb(policy, encoder)
    poison_encoder(encoder, monkeypatch)
    result = retention.evaluate_retention(policy, encoder, plan, arm="paired", provenance=item.provenance)
    assert result["status"] == "incomplete" and result["aggregate"] is None and result["counts"]["failed_masks"] == 9
    assert all(m["nll"] is None and m["failure"] == "ValueError: features must all be finite"
               for c in result["cases"] for m in c["masks"])
    assert all(c["mean_nll"] is None for c in result["cases"])
    assert result["result_sha256"] == digest({k: v for k, v in result.items() if k != "result_sha256"})
    assert_undisturbed(before)
    monkeypatch.undo()                                                         # the encoder is healthy again
    assert retention.evaluate_retention(policy, encoder, plan, arm="paired", provenance=item.provenance) == clean


def test_a_structural_value_error_from_the_encoder_still_aborts_and_restores_state(lineage, monkeypatch):
    item, plan = lineage.mid[False], build()
    policy, encoder = copy.deepcopy(item.policy), copy.deepcopy(item.encoder)
    original, layer = encoder.model.forward, encoder.model.num_layers

    def one_residue_short(*args, **kwargs):
        output = original(*args, **kwargs)
        output["representations"][layer] = output["representations"][layer][:, :-1]
        return output

    monkeypatch.setattr(encoder.model, "forward", one_residue_short)
    before = disturb(policy, encoder)
    with pytest.raises(ValueError, match="features must have shape"):
        retention.evaluate_retention(policy, encoder, plan, arm="paired", provenance=item.provenance)
    assert_undisturbed(before)


def test_a_policy_other_than_the_checkpoints_is_refused_before_any_forward(lineage, monkeypatch):
    item, plan = lineage.mid[False], build()
    other, _ = fresh_models("full")                       # same encoder identity, a different update mode
    encoder, forwards = copy.deepcopy(item.encoder), []
    monkeypatch.setattr(other, "reconstruction_nll", lambda *a, **k: forwards.append("policy"))
    monkeypatch.setattr(encoder, "extract_exact", lambda *a, **k: forwards.append("encoder"))
    assert other.encoder_identity == encoder.encoder_identity
    assert other.identity() != item.provenance["training"]["model"]
    with pytest.raises(ValueError, match="model identity differs"):
        retention.evaluate_retention(other, encoder, plan, arm="paired", provenance=item.provenance)
    assert forwards == []


def test_every_view_is_preflighted_before_the_first_forward(lineage, monkeypatch):
    item, plan = lineage.mid[False], build()
    policy, encoder = copy.deepcopy(item.policy), copy.deepcopy(item.encoder)
    real, state, forwards = policy.preflight, {"calls": 0}, []

    def preflight(views, contexts):
        state["calls"] += 1
        if state["calls"] == 4:
            raise ValueError("antibody exceeds native position limit")
        return real(views, contexts)

    monkeypatch.setattr(policy, "preflight", preflight)
    monkeypatch.setattr(policy, "reconstruction_nll", lambda *a, **k: forwards.append(1))
    monkeypatch.setattr(encoder, "extract_exact", lambda *a, **k: forwards.append(1))
    with pytest.raises(ValueError, match="native position limit"):
        retention.evaluate_retention(policy, encoder, plan, arm="paired", provenance=item.provenance)
    assert state["calls"] == 4 and forwards == []


def test_update_zero_and_the_final_checkpoint_are_scored_on_one_frozen_plan(lineage):
    plan = build()
    before, after = evaluate(lineage.update0, plan), evaluate(lineage.mid[False], plan)
    assert before["checkpoint"]["next_update"] == 0 and after["checkpoint"]["next_update"] == 2
    assert before["plan"] == after["plan"] and before["status"] == after["status"] == "complete"
    assert before["model_sha256"] == after["model_sha256"] and before["source_sha256"] == after["source_sha256"]
    assert before["checkpoint"]["checkpoint_sha256"] != after["checkpoint"]["checkpoint_sha256"]
    assert before["aggregate"]["mean_record_nll"] != after["aggregate"]["mean_record_nll"]
    for result in (before, after):
        assert result["result_sha256"] == digest({k: v for k, v in result.items() if k != "result_sha256"})
        assert result["readout"]["unit"] == "nats per masked site" and "not binding retention" in result["scope"]
        assert result["encoder_runtime_sha256"] and result["lineage"]["stage"] == "mid"


def test_the_readout_is_deterministic_in_eval_mode_whatever_the_outer_mode_and_generators(lineage):
    item, plan = lineage.mid[False], build()
    policy, encoder = copy.deepcopy(item.policy), copy.deepcopy(item.encoder)
    policy.train()
    torch.manual_seed(1)
    random.seed(2)
    first = retention.evaluate_retention(policy, encoder, plan, arm="paired", provenance=item.provenance)
    policy.eval()
    torch.manual_seed(3)
    random.seed(4)
    assert retention.evaluate_retention(policy, encoder, plan, arm="paired", provenance=item.provenance) == first


# ---------------------------------------------------------------------------
# No hidden effect on the model, the generators or the next update
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("outcome", ["success", "failure"])
def test_modes_generators_parameters_and_gradients_are_restored_exactly(lineage, monkeypatch, outcome):
    item, plan = lineage.mid[False], build()
    policy, encoder = copy.deepcopy(item.policy), copy.deepcopy(item.encoder)
    policy_modules, encoder_modules = list(policy.modules()), list(encoder.modules())
    for index, module in enumerate(policy_modules):
        module.training = index % 3 != 1
    for index, module in enumerate(encoder_modules):
        module.training = index % 2 == 0
    flags = [(module, module.training) for module in policy_modules + encoder_modules]
    assert {flag for _, flag in flags} == {True, False}                       # genuinely mixed
    trainable = [p for p in policy.parameters() if p.requires_grad]
    for index, parameter in enumerate(trainable):
        parameter.grad = torch.full_like(parameter, 0.25 + index)
    values = {id(p): (p.detach().clone(), p.requires_grad) for p in (*policy.parameters(), *encoder.parameters())}
    grads = {id(p): p.grad.clone() for p in trainable}
    random.seed(11)
    torch.manual_seed(12)
    python_state, torch_state = random.getstate(), torch.get_rng_state()
    state, eval_in_policy, eval_in_encoder = {"calls": 0}, [], []
    real = policy.reconstruction_nll

    def disturbing(views, contexts, targets):
        state["calls"] += 1
        random.random()
        torch.rand(4)
        if outcome == "failure" and state["calls"] == 3:
            policy.up.training = True
            raise RuntimeError("injected failure")
        value = real(views, contexts, targets)
        if state["calls"] == 9:                      # leave flags flipped at the very end
            policy.up.training = True
            encoder.model.training = True
        return value

    handles = [
        policy.backbone.register_forward_pre_hook(
            lambda module, args: eval_in_policy.append(all(not m.training for m in policy.modules()))),
        encoder.model.register_forward_pre_hook(
            lambda module, args: eval_in_encoder.append(all(not m.training for m in encoder.modules())))]
    monkeypatch.setattr(policy, "reconstruction_nll", disturbing)
    try:
        if outcome == "failure":
            with pytest.raises(RuntimeError, match="injected failure"):
                retention.evaluate_retention(policy, encoder, plan, arm="paired", provenance=item.provenance)
        else:
            result = retention.evaluate_retention(policy, encoder, plan, arm="paired", provenance=item.provenance)
            assert result["status"] == "complete"
            assert len(eval_in_policy) == 9 and all(eval_in_policy)       # eval inside, whatever the outer modes
            assert len(eval_in_encoder) == 9 and all(eval_in_encoder)
    finally:
        for handle in handles:
            handle.remove()
    assert all(module.training is flag for module, flag in flags)
    assert random.getstate() == python_state and torch.equal(torch.get_rng_state(), torch_state)
    assert torch.is_grad_enabled()
    for parameter in (*policy.parameters(), *encoder.parameters()):
        value, requires_grad = values[id(parameter)]
        assert torch.equal(parameter.detach(), value) and parameter.requires_grad is requires_grad
    for parameter in trainable:
        assert parameter.grad is not None and torch.equal(parameter.grad, grads[id(parameter)])


def test_a_retention_evaluation_between_two_updates_leaves_the_next_update_bit_identical(lineage):
    decl, records = declaration(), world_records()
    population = qualify_manifest(decl, records)
    certificate = qualify_transfer(population, endpoint_manifest(), **FAMILIES)
    base, encoder = fresh_models("lora")
    twin_policy, twin_encoder = copy.deepcopy(base), copy.deepcopy(encoder)
    plain = build_trainer(base, encoder, population, training_config(total_updates=3), certificate=certificate)
    probed = build_trainer(twin_policy, twin_encoder, population, training_config(total_updates=3),
                           certificate=certificate)
    assert plain.step() == probed.step()
    random_state = random.getstate()
    result = retention.evaluate_retention(probed.policy, probed.encoder, build(), arm="paired",
                                          provenance=lineage.mid[False].provenance)
    assert result["status"] == "complete" and random.getstate() == random_state
    assert all(parameter.grad is None for _, parameter in probed.parameters)
    assert plain.step() == probed.step()
    equal(plain.policy.state_dict(), probed.policy.state_dict())
    equal(plain.optimizer.state_dict(), probed.optimizer.state_dict())
    assert plain.progress() == probed.progress() and probed.next_update == 2


# ---------------------------------------------------------------------------
# Lineage
# ---------------------------------------------------------------------------

def test_a_common_plan_binds_midtraining_with_and_without_the_functional_arm(lineage):
    plain, shared = build(), build(functional=functional_doc())
    no_aux, aux = lineage.mid[False], lineage.mid[True]
    base = ["broad_input_file", "population", "endpoint_manifest", "protected_families"]
    assert retention.check_lineage(plain, no_aux.provenance) == {
        "stage": "mid", "origin": "release", "bound": [*base, "no_functional_arm_in_run"]}
    # Deliberate: a common plan may guard functional train data that this run never trained on.  Only the
    # identity fields are compared, never the whole certificate digest.
    assert retention.check_lineage(shared, no_aux.provenance)["bound"] == [*base, "no_functional_arm_in_run"]
    assert retention.check_lineage(shared, aux.provenance)["bound"] == [*base, "functional_qualification"]
    for plan in (plain, build(functional=functional_doc("another curator claim"))):
        with pytest.raises(ValueError, match="functional qualification"):
            retention.check_lineage(plan, aux.provenance)


@pytest.mark.parametrize("name", ["families", "manifest", "population"])
def test_drift_between_the_plan_and_the_midtraining_run_is_rejected(lineage, name):
    if name == "families":
        plan, message = build(spec=make_spec(protected={"excluded_target_families": ["another-target"]})), \
            "protect different excluded_target_families"
    elif name == "manifest":
        plan, message = build(manifest=EndpointManifest(endpoint_manifest().entries[:2])), "different endpoint manifest"
    else:
        extra = broad("hx", heavy(16), Ag=AG2, family="famHX", target="tgt-h0", partition="val")
        plan, message = build([*world_records(), extra]), "population differs"
    for item in (lineage.mid[False], lineage.mid[True]):
        with pytest.raises(ValueError, match=message):
            retention.check_lineage(plan, item.provenance)


@pytest.mark.parametrize("aux", [False, True])
def test_post_from_midtraining_binds_the_current_budget_and_the_inherited_parent(lineage, aux):
    item = lineage.post[aux]
    plan = build(functional=functional_doc() if aux else None)
    record = retention.check_lineage(plan, item.provenance)
    assert record["stage"] == "post" and record["origin"] == "parent" and record["bound"][0] == "post_budget_manifest"
    assert "parent_population" in record["bound"] and "parent_protected_families" in record["bound"]
    assert "parent_broad_input_file" in record["bound"] and "inherited_transfer_certificate" in record["bound"]
    # The CURRENT fitting budget's manifest.
    with pytest.raises(ValueError, match="post fitting budget belongs to a different endpoint manifest"):
        retention.check_lineage(build(manifest=EndpointManifest(endpoint_manifest().entries[:2])), item.provenance)
    # The inherited certificate.
    with pytest.raises(ValueError, match="protect different excluded_antibody_families"):
        retention.check_lineage(build(spec=make_spec(protected={"excluded_antibody_families": ["another-ab"]})),
                                item.provenance)
    # The inherited functional qualification.
    if aux:
        assert record["bound"][-1] == "parent_functional_qualification"
        for drifted in (build(), build(functional=functional_doc("another curator claim"))):
            with pytest.raises(ValueError, match="functional qualification"):
                retention.check_lineage(drifted, item.provenance)
    else:
        assert record["bound"][-1] == "parent_no_functional_arm_in_run"
        assert retention.check_lineage(build(functional=functional_doc()), item.provenance)["origin"] == "parent"


def test_direct_release_post_binds_only_the_budget_manifest_and_invents_no_broad_ancestry(lineage):
    item = lineage.direct
    assert retention.check_lineage(build(), item.provenance) == {
        "stage": "post", "origin": "release", "bound": ["post_budget_manifest"]}
    extra = broad("hx", heavy(16), Ag=AG2, family="famHX", target="tgt-h0", partition="val")
    assert retention.check_lineage(build([*world_records(), extra]), item.provenance)["origin"] == "release"
    with pytest.raises(ValueError, match="post fitting budget belongs to a different endpoint manifest"):
        retention.check_lineage(build(manifest=EndpointManifest(endpoint_manifest().entries[:2])), item.provenance)
    forged = copy.deepcopy(item.provenance)
    forged["training"]["transfer_certificate"] = copy.deepcopy(
        lineage.mid[False].provenance["training"]["transfer_certificate"])
    with pytest.raises(ValueError, match="no broad transfer certificate"):
        retention.check_lineage(build(), forged)
    result = evaluate(item, build())
    assert result["status"] == "complete" and result["lineage"]["origin"] == "release"


def mutate_parent(provenance, change):
    for base in (provenance["training"]["model"]["base"], provenance["training"]["initialization"]["base"]):
        change(base["parent_provenance"])


def test_tampered_or_incomplete_provenance_is_rejected(lineage):
    plan, mid, post = build(), lineage.mid[False], lineage.post[False]
    cases = [
        (mid, lambda p: p["training"].update(transfer_certificate=None), "no transfer certificate"),
        (mid, lambda p: p["training"]["transfer_certificate"].update(excluded_target_families=["x"]),
         "certificate digest"),
        (mid, lambda p: p["training"].update(data={"manifest": "0" * 64, "population": "1" * 64}), "population differs"),
        (mid, lambda p: p["training"]["config"].update(stage="other"), "unsupported training stage"),
        (mid, lambda p: p.pop("progress"), "checkpoint provenance"),
        (post, lambda p: p["training"]["input"]["budget"].update(selection_digest="f" * 64), "selection digest"),
        (post, lambda p: mutate_parent(p, lambda parent: parent["progress"].update(next_update=1)),
         "completed, unmeasured"),
        (post, lambda p: mutate_parent(p, lambda parent: parent["training"]["transfer_certificate"].update(
            excluded_target_families=["x"])), "certificate digest"),
        (post, lambda p: p["training"]["initialization"].update(sha256="0" * 64), "parent initialization"),
    ]
    for item, change, message in cases:
        provenance = copy.deepcopy(item.provenance)
        change(provenance)
        with pytest.raises(ValueError, match=message):
            retention.check_lineage(plan, provenance)


RECORDED_INPUT = {
    "missing": (lambda t: t["input"].pop("input_sha256", None), "no broad input file hash"),
    "not_an_object": (lambda t: t.update(input=None), "no broad input file hash"),
    "other_file": (lambda t: t["input"].update(input_sha256="9" * 64), "different broad input file"),
    "not_a_hash": (lambda t: t["input"].update(input_sha256="not-a-sha"), "lowercase SHA-256"),
    "uppercase": (lambda t: t["input"].update(input_sha256="A" * 64), "lowercase SHA-256"),
    "not_text": (lambda t: t["input"].update(input_sha256=None), "lowercase SHA-256"),
}


@pytest.mark.parametrize("problem", sorted(RECORDED_INPUT))
@pytest.mark.parametrize("stage", ["mid", "post"])
@pytest.mark.parametrize("aux", [False, True])
def test_the_checkpoints_recorded_raw_input_file_must_be_the_plans_input_file(lineage, aux, stage, problem):
    item = getattr(lineage, stage)[aux]
    plan = build(functional=functional_doc() if aux else None)
    change, message = RECORDED_INPUT[problem]
    expected = "broad_input_file" if stage == "mid" else "parent_broad_input_file"
    assert expected in retention.check_lineage(plan, item.provenance)["bound"]        # the unmodified record binds
    provenance = copy.deepcopy(item.provenance)
    if stage == "mid":
        change(provenance["training"])
    else:                                                  # the parent's identity is what the post checkpoint inherits
        mutate_parent(provenance, lambda parent: change(parent["training"]))
    with pytest.raises(ValueError, match=message):
        retention.check_lineage(plan, provenance)


def test_a_plan_over_another_raw_file_is_refused_by_every_checkpoint_before_any_forward(lineage, monkeypatch):
    elsewhere = copy.deepcopy(build())
    elsewhere["broad"]["input_sha256"] = "9" * 64
    seal(elsewhere)                                    # a valid plan, over a different raw file
    for item in (lineage.mid[False], lineage.post[False]):
        policy, encoder = copy.deepcopy(item.policy), copy.deepcopy(item.encoder)
        forwards = []
        monkeypatch.setattr(policy, "reconstruction_nll", lambda *a, **k: forwards.append("policy"))
        monkeypatch.setattr(encoder, "extract_exact", lambda *a, **k: forwards.append("encoder"))
        with pytest.raises(ValueError, match="different broad input file"):
            retention.evaluate_retention(policy, encoder, elsewhere, arm="paired", provenance=item.provenance)
        assert forwards == []


def malformed_parent_hash(provenance):
    for base in (provenance["training"]["model"]["base"], provenance["training"]["initialization"]["base"]):
        base["parent_sha256"] = "not-a-sha"
    provenance["training"]["initialization"]["sha256"] = "not-a-sha"


def reseal_certificate(certificate):
    certificate["certificate_sha256"] = digest({k: v for k, v in certificate.items() if k != "certificate_sha256"})
    return certificate


def forged_but_valid_certificate(certificate):
    """A certificate whose digest verifies and that differs from the parent's in one recorded count."""
    certificate = copy.deepcopy(certificate)
    certificate["coverage"]["train_records"] += 1
    return reseal_certificate(certificate)


def digest_broken_certificate(certificate):
    certificate = copy.deepcopy(certificate)
    certificate["excluded_target_families"] = ["x"]
    return certificate


INHERITED = {
    # The post identity's OWN certificate against the parent's actual certificate.
    "malformed_certificate": (lambda p, other: p["training"].update(transfer_certificate={"malformed": "unrelated"}),
                              "transfer certificate: expected exactly"),
    "absent_certificate": (lambda p, other: p["training"].update(transfer_certificate=None),
                           "records no transfer certificate"),
    "digest_broken_certificate": (lambda p, other: p["training"].update(
        transfer_certificate=digest_broken_certificate(p["training"]["transfer_certificate"])),
        "certificate digest mismatch"),
    "valid_but_forged_certificate": (lambda p, other: p["training"].update(
        transfer_certificate=forged_but_valid_certificate(p["training"]["transfer_certificate"])),
        "not the certificate its midtraining parent"),
    "another_runs_certificate": (lambda p, other: p["training"].update(transfer_certificate=copy.deepcopy(other)),
                                 "not the certificate its midtraining parent"),
    # The parent conditions the trainer's own lineage guard enforces (exact integers, horizon, zero exposure).
    "parent_hash_not_a_hash": (lambda p, other: malformed_parent_hash(p), "lowercase SHA-256"),
    "parent_checkpoint_hash_not_a_hash": (lambda p, other: mutate_parent(
        p, lambda parent: parent.update(checkpoint_sha256="not-a-sha")), "lowercase SHA-256"),
    "next_update_is_a_float": (lambda p, other: mutate_parent(
        p, lambda parent: parent["progress"].update(next_update=float(parent["training"]["config"]["total_updates"]))),
        "completed, unmeasured"),
    "fitting_exposure_is_false": (lambda p, other: mutate_parent(
        p, lambda parent: parent["progress"].update(fitting_endpoints_available=False)), "completed, unmeasured"),
    "selection_exposure_is_a_float": (lambda p, other: mutate_parent(
        p, lambda parent: parent["progress"].update(selection_labels=0.0)), "completed, unmeasured"),
    "inherited_exposure": (lambda p, other: mutate_parent(
        p, lambda parent: parent["progress"].update(inherited_exposure={"fitting": 1})), "completed, unmeasured"),
    "zero_horizon": (lambda p, other: mutate_parent(
        p, lambda parent: (parent["training"]["config"].update(total_updates=0),
                           parent["progress"].update(next_update=0))), "completed, unmeasured"),
}


@pytest.mark.parametrize("problem", sorted(INHERITED))
@pytest.mark.parametrize("aux", [False, True])
def test_a_post_checkpoint_must_carry_its_parents_certificate_and_a_parent_the_trainer_would_accept(
        lineage, aux, problem):
    item, plan = lineage.post[aux], build(functional=functional_doc() if aux else None)
    change, message = INHERITED[problem]
    other = lineage.mid[not aux].provenance["training"]["transfer_certificate"]         # valid, but not this lineage's
    assert other != item.provenance["training"]["transfer_certificate"]
    assert item.provenance["training"]["transfer_certificate"] == lineage.mid[aux].provenance["training"][
        "transfer_certificate"]                                                          # the unmodified post carries it
    provenance = copy.deepcopy(item.provenance)
    change(provenance, other)
    with pytest.raises(ValueError, match=message):
        retention.check_lineage(plan, provenance)


# ---------------------------------------------------------------------------
# Static: planning and evaluation never reach a measurement or a label
# ---------------------------------------------------------------------------

def test_the_retention_module_cannot_reach_measurements_or_labels():
    forbidden = {"load_post", "load_functional_measurements", "load_restricted_measurements", "RestrictedPostData",
                 "RestrictedFunctionalData", "MeasuredEndpoint", "select_budget", "select_functional_budget",
                 "resume_checkpoint", "build_models", "StageTrainer"}
    assert not [name for name in imported_names(retention) if name.rsplit(".", 1)[-1] in forbidden]
    assert "smallAntibodyGen.experiments.trastuzumab_poc.functional_protocol" not in {
        name.rsplit(".", 1)[0] for name in imported_names(retention)}


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def pairs_document():
    return {"schema": "trastuzumab-pairs/1",
            "declaration": {"sources": [{"source_id": "broad", "description": "broad source", "priority": 0}],
                            "rules": [{"rule_name": "rule", "description": "positive evidence"}],
                            "source_priority_order": ["broad"], "grouping": {"separation_axes": ["antibody_family"]}},
            "records": [asdict(record) for record in world_records()]}


def forbid(patch):
    def boom(*args, **kwargs):
        raise AssertionError("opened a model, a weight, a label or a measurement")

    patch.setattr(cli, "models", boom)
    patch.setattr(cli, "load_inference_checkpoint", boom)
    for owner, name in ((inputs, "load_functional_measurements"), (inputs, "load_post"),
                        (fp, "load_restricted_measurements"), (fp.RestrictedFunctionalData, "from_document")):
        patch.setattr(owner, name, boom)


def test_cli_plans_without_models_trains_and_evaluates_without_any_training_input(tmp_path, monkeypatch):
    config_path, _ = mid_config(tmp_path, updates=2)
    write(tmp_path / "pairs.json", pairs_document())
    declaration_, *_ = inputs.load_broad_context(tmp_path / "pairs.json")
    write(tmp_path / "constructs.json", small_document(declaration_))
    write(tmp_path / "spec.json", spec_document(functional_path="constructs.json", masks={"seed": 5, "per_record": 2}))
    plan_path = tmp_path / "retention-plan.json"
    with monkeypatch.context() as patch:
        forbid(patch)
        run("plan-retention", "--spec", tmp_path / "spec.json", "--output", plan_path)
        with pytest.raises(ValueError, match="output already exists"):
            run("plan-retention", "--spec", tmp_path / "spec.json", "--output", plan_path)
    plan = read_json(plan_path)
    assert retention.validate_plan(plan) == plan and plan["counts"]["records"] == 3 and plan["counts"]["masks"] == 6
    assert plan["broad"]["input_sha256"] == sha(tmp_path / "pairs.json")
    assert plan["functional"]["constructs_sha256"] == sha(tmp_path / "constructs.json")
    assert plan["protected"]["endpoint_manifest_sha256"] == sha(tmp_path / "endpoints.json")

    run("train", "--config", config_path, "--output", tmp_path / "full")
    trained = read_json(tmp_path / "full" / "result.json")
    checkpoint_hash = trained["checkpoint"]["sha256"]
    for name in ("pairs.json", "constructs.json", "endpoints.json", "spec.json"):
        (tmp_path / name).unlink()                       # evaluation needs only plan, model and checkpoint
    common = ("--config", config_path, "--plan", plan_path, "--checkpoint", tmp_path / "full" / "checkpoint.pt",
              "--checkpoint-sha256", checkpoint_hash)
    with monkeypatch.context() as patch:
        for owner, name in ((inputs, "load_functional_measurements"), (inputs, "load_post"),
                            (inputs, "load_broad_context"), (fp, "load_restricted_measurements")):
            patch.setattr(owner, name, lambda *a, **k: pytest.fail("evaluation reopened a training input"))
        run("evaluate-retention", *common, "--arm", "paired", "--output", tmp_path / "paired.json")
        run("evaluate-retention", *common, "--arm", "abonly", "--output", tmp_path / "abonly.json")
        with pytest.raises(ValueError, match="output already exists"):
            run("evaluate-retention", *common, "--arm", "paired", "--output", tmp_path / "paired.json")
    paired, abonly = read_json(tmp_path / "paired.json"), read_json(tmp_path / "abonly.json")
    for result, arm in ((paired, "paired"), (abonly, "abonly")):
        assert result["schema"] == "trastuzumab-retention-result/1" and result["status"] == "complete"
        assert result["arm"] == arm and result["plan"]["plan_sha256"] == plan["plan_sha256"]
        assert result["checkpoint"]["checkpoint_sha256"] == checkpoint_hash and result["checkpoint"]["next_update"] == 2
        assert result["counts"] == {**plan["counts"], "failed_masks": 0}
        assert result["lineage"]["bound"][0] == "broad_input_file"                # the real identity carries the hash
        assert result["lineage"]["bound"][-1] == "no_functional_arm_in_run"       # common plan, no-aux run
        assert result["result_sha256"] == digest({k: v for k, v in result.items() if k != "result_sha256"})
    assert paired["aggregate"] != abonly["aggregate"] and paired["model_sha256"] == abonly["model_sha256"]
    # Existing commands are unchanged on the same checkpoint.
    queries = write(tmp_path / "queries.json", _panel_document())
    run("score", "--config", config_path, "--checkpoint", tmp_path / "full" / "checkpoint.pt",
        "--checkpoint-sha256", checkpoint_hash, "--queries", queries, "--arm", "paired", "--scorer-id", "tiny",
        "--orders", 1, "--order-seed", 3, "--output", tmp_path / "scores")
    assert (tmp_path / "scores" / "scores.json").exists()


def test_cli_planning_failure_and_a_tampered_plan_never_reach_a_model_or_write_output(tmp_path, monkeypatch):
    config_path, _ = mid_config(tmp_path, updates=1)
    write(tmp_path / "pairs.json", pairs_document())
    write(tmp_path / "spec.json", spec_document(protected={"excluded_antibody_families": ["famH0"]}))
    with monkeypatch.context() as patch:
        forbid(patch)
        with pytest.raises(ValueError, match="h0 is in a protected antibody family"):
            run("plan-retention", "--spec", tmp_path / "spec.json", "--output", tmp_path / "rejected.json")
        assert not (tmp_path / "rejected.json").exists()
        write(tmp_path / "spec.json", spec_document())
        run("plan-retention", "--spec", tmp_path / "spec.json", "--output", tmp_path / "plan.json")
        plan = read_json(tmp_path / "plan.json")
        plan["cases"][0]["masks"][0][0] = ["H", 99]
        write(tmp_path / "tampered.json", plan)
        with pytest.raises(ValueError, match="digest mismatch"):
            run("evaluate-retention", "--config", config_path, "--plan", tmp_path / "tampered.json",
                "--checkpoint", tmp_path / "never-opened.pt", "--checkpoint-sha256", "a" * 64, "--arm", "paired",
                "--output", tmp_path / "result.json")
        assert not (tmp_path / "result.json").exists()
