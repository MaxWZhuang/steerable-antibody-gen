"""Transfer certificate, lineage guard and the parent hand-off with the functional arm (M2).

The first half is outcome-free and API-level.  The second half trains an actual functional
midtraining run (LoRA and full), exports it, and starts a fresh post-training run from it.
"""
from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from smallAntibodyGen.experiments.trastuzumab_poc import checkpoint as ck
from smallAntibodyGen.experiments.trastuzumab_poc import functional_protocol as fp
from smallAntibodyGen.experiments.trastuzumab_poc import parent as parent_module
from smallAntibodyGen.experiments.trastuzumab_poc.data import qualify_manifest
from smallAntibodyGen.experiments.trastuzumab_poc.parent import export_parent, load_parent
from smallAntibodyGen.experiments.trastuzumab_poc.protocol import (
    EndpointConstruct, EndpointEntry, EndpointManifest, MeasuredEndpoint, RestrictedPostData,
    SelectionAllowance, select_budget,
)
from smallAntibodyGen.experiments.trastuzumab_poc.runner import StageTrainer, TrainingConfig
from smallAntibodyGen.experiments.trastuzumab_poc.transfer_guard import (
    functional_binding, qualify_transfer, validate_stage_lineage,
)
from smallAntibodyGen.models.antibody_antigen.dynamic_context import DynamicContextProvider
from smallAntibodyGen.models.antibody_antigen.fusion import FusionConfig
from smallAntibodyGen.models.antibody_antigen.views import mask_complete
from smallAntibodyGen.tests.test_trastuzumab_functional_data import (
    AG, AG2, KD, LIGHT, broad_records, construct_row, declaration, group_row, heavy, qualify,
)
from smallAntibodyGen.tests.test_trastuzumab_functional_runner import (
    MODES, SMALL_VALUES, build_trainer, fresh_models, functional_run, small_document, small_world,
    training_config,
)

FAMILIES = {"excluded_antibody_families": ["heldout-ab"], "excluded_target_families": ["heldout-target"]}
SOURCE = {"source": "synthetic"}


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def endpoint_manifest():
    entries = tuple(EndpointEntry(str(i), EndpointConstruct("AC", core, "DE", "FG", "HIK"), "post_pool",
                                  "n", "assay", "synthetic affinity")
                    for i, core in enumerate(("ACDEFGHIKL", "LMNPQRSTVW", "AAAAAAAAAA")))
    return EndpointManifest(entries)


def post_data(manifest):
    budget = select_budget(manifest, 3, "shared")
    labels = tuple(MeasuredEndpoint(e.endpoint_id, float(i), "higher_better", "assay")
                   for i, e in enumerate(manifest.entries))
    return RestrictedPostData(manifest=manifest, budget=budget, measured=labels, fitting_count=3,
                              development_labels=(), selection_allowance=SelectionAllowance(0, ()))


def release():
    base = {"variant": "dplm_650m", "pretrained": True}
    return SimpleNamespace(base_identity=base), {"kind": "release", "base": copy.deepcopy(base)}


# ---------------------------------------------------------------------------
# The certificate
# ---------------------------------------------------------------------------

def test_the_certificate_binds_the_full_functional_qualification_and_leaves_the_population_alone():
    decl, records, population, qualified = small_world()
    manifest = endpoint_manifest()
    digest_before = population.population_digest
    plain = qualify_transfer(population, manifest, **FAMILIES)
    bound = qualify_transfer(population, manifest, functional=qualified, **FAMILIES)
    assert plain["schema"] == bound["schema"] == "trastuzumab-transfer-qualification/2"
    assert plain["functional"] is None and bound["functional"] == functional_binding(qualified)
    assert bound["functional"]["qualified_digest"] == qualified.digest
    assert bound["functional"]["contract_digest"] == qualified.contract_digest
    assert bound["functional"]["train_constructs"] == 2 and bound["functional"]["train_unique_antigens"] == 1
    assert bound["certificate_sha256"] != plain["certificate_sha256"]
    assert {k: v for k, v in bound.items() if k not in ("functional", "certificate_sha256")} == {
        k: v for k, v in plain.items() if k not in ("functional", "certificate_sha256")}
    assert population.population_digest == digest_before and bound["coverage"]["train_records"] == 3
    # Another qualification (here only a different curator assertion) binds differently.
    document = small_document(decl)
    document["study_contract"]["assertion"] = "another curator claim"
    other = qualify(document, decl, records, population=population)
    other_bound = qualify_transfer(population, manifest, functional=other, **FAMILIES)
    assert other_bound["functional"]["qualified_digest"] != bound["functional"]["qualified_digest"]
    assert other_bound["certificate_sha256"] != bound["certificate_sha256"]


def forbidden_world(case):
    decl, records, document = declaration(), broad_records(), None
    document = small_document(decl)
    # The family-only case needs its OWN antigen (AG2, not the protected "HIK"): the shared AG
    # already carries "her2" in the existing constructs, and one sequence cannot carry two families.
    row = {"exact_antibody": dict(H="ACACDEFGHIKLDE", L="FG"), "exact_antigen": dict(H=heavy(3), Ag="HIK"),
           "antibody_family": dict(H=heavy(3), family="heldout-ab"),
           "target_family": dict(H=heavy(3), Ag=AG2, target="heldout-target")}[case]
    H = row.pop("H")
    document["constructs"].append(construct_row("z", H, **row))
    document["groups"].append(group_row("g-z", "ctx-a", KD, ["z"], Ag=row.get("Ag", AG)))
    population = qualify_manifest(decl, records)
    return population, qualify(document, decl, records, population=population)


@pytest.mark.parametrize("case", ["exact_antibody", "exact_antigen", "antibody_family", "target_family"])
def test_a_forbidden_train_construct_is_rejected_even_when_the_budget_never_selects_it(case):
    population, qualified = forbidden_world(case)
    seed = next(s for s in (f"s{i}" for i in range(120))
                if fp.select_functional_budget(qualified, 2, s).selected_set == {("g-x", "a"), ("g-x", "b")})
    budget = fp.select_functional_budget(qualified, 2, seed)
    assert ("g-z", "z") not in budget.selected_set
    assert fp.functional_exposure_plan(qualified, budget)["pair_eligible_group_ids"] == ["g-x"]
    with pytest.raises(ValueError, match="functional training: z"):
        qualify_transfer(population, endpoint_manifest(), functional=qualified, **FAMILIES)
    # The denoising population alone is still acceptable: the rejection is the functional check.
    assert qualify_transfer(population, endpoint_manifest(), **FAMILIES)["functional"] is None


def test_validation_constructs_are_not_training_exposure_and_inputs_are_checked():
    decl, records, population, _ = small_world()
    document = small_document(decl)
    document["constructs"].append(construct_row("zv", heavy(4), Ag="HIK", family="famV", partition="val"))
    document["groups"].append(group_row("g-zv", "ctx-a", KD, ["zv"], Ag="HIK"))
    qualified = qualify(document, decl, records, population=population)
    certificate = qualify_transfer(population, endpoint_manifest(), functional=qualified, **FAMILIES)
    assert certificate["functional"]["train_constructs"] == 2
    other_population = qualify_manifest(decl, records[:2])
    with pytest.raises(ValueError, match="different broad population"):
        qualify_transfer(other_population, endpoint_manifest(), functional=qualified, **FAMILIES)
    with pytest.raises(TypeError):
        qualify_transfer(population, endpoint_manifest(), functional="qualified", **FAMILIES)


# ---------------------------------------------------------------------------
# The lineage guard
# ---------------------------------------------------------------------------

def test_a_midtraining_run_needs_a_certificate_that_binds_exactly_its_functional_qualification():
    decl, records, population, qualified = small_world()
    policy, origin = release()
    config = SimpleNamespace(stage="mid")
    plain = qualify_transfer(population, endpoint_manifest(), **FAMILIES)
    bound = qualify_transfer(population, endpoint_manifest(), functional=qualified, **FAMILIES)
    assert validate_stage_lineage(policy, config, origin, transfer_certificate=bound, functional=qualified) == bound
    assert validate_stage_lineage(policy, config, origin, transfer_certificate=plain) == plain
    for certificate, functional in ((plain, qualified), (bound, None)):
        with pytest.raises(ValueError, match="functional qualification"):
            validate_stage_lineage(policy, config, origin, transfer_certificate=certificate, functional=functional)
    document = small_document(decl)
    document["study_contract"]["assertion"] = "another curator claim"
    other = qualify(document, decl, records, population=population)
    with pytest.raises(ValueError, match="functional qualification"):
        validate_stage_lineage(policy, config, origin, transfer_certificate=bound, functional=other)
    with pytest.raises(ValueError, match="requires transfer qualification"):
        validate_stage_lineage(policy, config, origin, functional=qualified)
    tampered = copy.deepcopy(bound)
    tampered["functional"]["qualified_digest"] = "f" * 64
    with pytest.raises(ValueError, match="certificate digest"):
        validate_stage_lineage(policy, config, origin, transfer_certificate=tampered, functional=qualified)
    tampered = copy.deepcopy(bound)
    tampered["functional"] = None
    with pytest.raises(ValueError, match="certificate digest"):
        validate_stage_lineage(policy, config, origin, transfer_certificate=tampered, functional=qualified)
    with pytest.raises(TypeError):
        validate_stage_lineage(policy, config, origin, transfer_certificate=plain, functional="qualified")


def test_post_training_takes_no_functional_supervision():
    _, _, _, qualified = small_world()
    policy, origin = release()
    with pytest.raises(ValueError, match="no functional supervision"):
        validate_stage_lineage(policy, SimpleNamespace(stage="post"), origin,
                               post_data=post_data(endpoint_manifest()), functional=qualified)


@pytest.mark.parametrize("certified", ["functional", "none"])
def test_the_trainer_rejects_a_certificate_that_does_not_bind_its_functional_run(certified):
    _, _, population, qualified = small_world()
    base, encoder = fresh_models("lora")
    certificate = qualify_transfer(population, endpoint_manifest(),
                                   functional=qualified if certified == "functional" else None, **FAMILIES)
    run = functional_run(qualified, SMALL_VALUES["ordered"]) if certified == "none" else None
    with pytest.raises(ValueError, match="functional qualification"):
        build_trainer(base, encoder, population, training_config(), functional=run, certificate=certificate)


# ---------------------------------------------------------------------------
# The parent hand-off, with actual training
# ---------------------------------------------------------------------------

def post_fusion():
    return FusionConfig(context_dim=8, bottleneck=6, lora_rank=2, lora_alpha=4, lora_dropout=0.3,
                        update_mode="lora", activation_checkpointing=True)


@pytest.fixture(scope="module", params=MODES)
def chain(request, tmp_path_factory):
    """An actual functional midtraining run, its checkpoint, and the parent exported from it."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        mode = request.param
        directory = tmp_path_factory.mktemp(f"chain-{mode}")
        _, _, population, qualified = small_world()
        manifest = endpoint_manifest()
        certificate = qualify_transfer(population, manifest, functional=qualified, **FAMILIES)
        base, encoder = fresh_models(mode)
        mid = build_trainer(base, encoder, population, training_config(total_updates=3),
                            functional=functional_run(qualified, SMALL_VALUES["ordered"]),
                            certificate=certificate)
        mid.run()
        saved = ck.save_checkpoint(mid, directory / "mid.pt")
        policy, _ = fresh_models(mode)
        provenance = ck.load_inference_checkpoint(policy, saved["path"], encoder=encoder,
                                                  expected_sha256=saved["sha256"], source_identity=SOURCE)
        exported = export_parent(policy, directory / "parent.pt", provenance=provenance)
        return SimpleNamespace(mode=mode, mid=mid, encoder=encoder, manifest=manifest, certificate=certificate,
                               provenance=provenance, parent_path=directory / "parent.pt",
                               parent_sha=exported["sha256"], checkpoint=saved)
    finally:
        torch.set_num_threads(previous)


def load_chain_parent(chain):
    return load_parent(chain.parent_path, expected_sha256=chain.parent_sha,
                       encoder_identity=chain.encoder.encoder_identity, fusion_config=post_fusion(),
                       device="cpu", seed=5)


def post_trainer(chain, policy=None, manifest=None):
    policy = load_chain_parent(chain) if policy is None else policy
    origin = {"kind": "parent", "sha256": chain.parent_sha, "base": policy.base_identity}
    config = TrainingConfig(stage="post", arm="paired", seed=19, total_updates=1, examples_per_update=1,
                            learning_rate=0.001, ordering_weight=1.0, minimum_ordering_endpoint_coverage=0.0)
    return StageTrainer(policy, chain.encoder, config, post_data=post_data(manifest or chain.manifest),
                        data_identity=SOURCE, source_identity=SOURCE, initialization_identity=origin)


def test_the_midtraining_run_trained_the_head_and_recorded_the_supervision(chain):
    counters = chain.mid.progress()["functional"]["counters"]
    assert chain.mid.next_update == 3 and counters["ordered_updates"] == 3 and counters["ordered_pairs"] == 6
    assert float(next(iter(chain.mid.head_optimizer.state.values()))["step"]) == 3.0
    assert chain.provenance["training"]["transfer_certificate"] == chain.certificate
    assert chain.certificate["functional"]["qualified_digest"] == chain.mid.identity["functional"]["qualified_digest"]


def test_the_exported_parent_preserves_logits_and_features_at_the_merge_tolerance(chain):
    parent = load_chain_parent(chain)
    view, _ = mask_complete({"H": heavy(0), "L": LIGHT, "Ag": AG}, (("H", 3), ("H", 4)))
    chain.mid.policy.eval()
    parent.eval()
    outputs = []
    try:
        for policy in (chain.mid.policy, parent):
            with torch.no_grad():
                logits, hidden, _ = policy.forward_views([view.without_ag()], [DynamicContextProvider(chain.encoder, view)])
            outputs.append((logits, hidden))
    finally:
        chain.mid.policy.train()
    torch.testing.assert_close(outputs[0][0], outputs[1][0], rtol=0, atol=1e-4)
    torch.testing.assert_close(outputs[0][1], outputs[1][1], rtol=0, atol=1e-4)
    assert parent.config.update_mode == "lora" and parent.base_identity["variant"] == "materialized-parent"


def test_the_parent_drops_the_head_and_both_optimizers_but_keeps_the_supervision_provenance(chain):
    payload = torch.load(chain.parent_path, weights_only=True)
    assert set(payload) == set(parent_module._PAYLOAD_KEYS)
    assert set(payload["provenance"]) == {"checkpoint_sha256", "training", "progress"}
    assert payload["provenance"]["checkpoint_sha256"] == chain.checkpoint["sha256"]
    assert payload["provenance"]["training"]["functional"] == chain.mid.identity["functional"]
    assert payload["provenance"]["progress"]["functional"] == chain.mid.progress()["functional"]
    assert payload["provenance"]["training"]["functional"]["head"]["parent_behavior"].startswith("discarded")
    assert set(payload["fusion_weights"]) == {"norm.weight", "norm.bias", "down.weight", "down.bias",
                                             "up.weight", "up.bias"}
    assert chain.mid.head is not None and chain.mid.head_optimizer.state


def test_a_fresh_post_run_starts_without_the_head_or_any_optimizer_state(chain):
    post = post_trainer(chain)
    assert post.head is None and post.head_optimizer is None and post.functional is None
    assert post.identity["functional"] is None and post.progress()["functional"] is None
    assert not post.optimizer.state and post.policy.config.update_mode == "lora"
    adapters = {name: p.detach().clone() for name, p in post.policy.named_parameters() if "lora_B" in name}
    assert adapters and all(not bool(tensor.any()) for tensor in adapters.values())     # fresh, zero-output
    assert post.identity["transfer_certificate"] == chain.certificate
    provenance = post.identity["initialization"]["base"]["parent_provenance"]
    assert set(provenance) == {"checkpoint_sha256", "training", "progress"}
    inherited = post.progress()["inherited_exposure"]["parent_progress"]
    assert inherited["next_update"] == 3 and inherited["functional"] == chain.mid.progress()["functional"]
    assert post.identity["input"]["budget"]["selected_ids"] == list(post.post_data.budget.selected_ids)
    assert post.identity["input"]["functional"] is None
    post.step()
    assert any(not torch.equal(adapters[name], p) for name, p in post.policy.named_parameters() if name in adapters)
    assert post.progress()["fitting_endpoints_available"] == 3


def test_the_post_run_needs_the_protected_endpoint_manifest_of_the_functional_parent(chain):
    old = chain.manifest
    entries = list(old.entries)
    entries[2] = replace(entries[2], construct=replace(entries[2].construct, core="CCCCCCCCCC"))
    with pytest.raises(ValueError, match="different endpoint manifest"):
        post_trainer(chain, manifest=EndpointManifest(tuple(entries)))


def put(container, key, value):
    container[key] = value


TAMPERS = {
    "counter": lambda t, p: put(p["functional"]["counters"], "ordered_pairs", p["functional"]["counters"]["ordered_pairs"] + 1),
    "seen": lambda t, p: p["functional"]["seen"]["compared"].pop(),
    "ordered_updates": lambda t, p: put(p["functional"]["counters"], "ordered_updates", 0),
    "horizon": lambda t, p: put(p, "next_update", 2),
    "inherited_exposure": lambda t, p: put(p, "inherited_exposure", {"invented": 1}),
    "progress_forgotten": lambda t, p: put(p, "functional", None),
    "ancestor_digest": lambda t, p: put(t["functional"], "qualified_digest", "0" * 64),
    "ancestor_budget": lambda t, p: put(t["input"]["functional"]["budget"], "budget_seed", "other"),
    "ancestor_forgotten": lambda t, p: put(t, "functional", None),
    "certificate_digest": lambda t, p: put(t["transfer_certificate"]["functional"], "qualified_digest", "f" * 64),
    "certificate_removed": lambda t, p: put(t["transfer_certificate"], "functional", None),
    "old_training_schema": lambda t, p: put(t, "schema", "trastuzumab-training/1"),
}


@pytest.mark.parametrize("name", sorted(TAMPERS))
def test_tampered_parent_provenance_is_rejected_before_the_first_post_update(chain, name):
    policy = load_chain_parent(chain)
    identity = copy.deepcopy(policy.base_identity)
    provenance = identity["parent_provenance"]
    TAMPERS[name](provenance["training"], provenance["progress"])
    policy.base_identity = identity
    with pytest.raises(ValueError):
        post_trainer(chain, policy=policy)
