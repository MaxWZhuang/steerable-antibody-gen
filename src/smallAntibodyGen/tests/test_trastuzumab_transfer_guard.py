"""Adversarial cross-stage leakage and inherited measurement-budget checks."""
import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from smallAntibodyGen.experiments.trastuzumab_poc.data import (
    GroupingDeclaration, InputRecord, ManifestDeclaration, QualificationRule,
    SourceDeclaration, qualify_manifest,
)
from smallAntibodyGen.experiments.trastuzumab_poc.protocol import (
    EndpointConstruct, EndpointEntry, EndpointManifest, MeasuredEndpoint,
    RestrictedPostData, SelectionAllowance, select_budget,
)
from smallAntibodyGen.experiments.trastuzumab_poc.transfer_guard import (
    qualify_transfer, validate_stage_lineage,
)


def _manifest():
    construct = EndpointConstruct("AC", "ACDEFGHIKL", "DE", "MN", "QRST")
    return EndpointManifest((
        EndpointEntry("fit", construct, "post_pool", "fit-group", "synthetic", "synthetic score"),
        EndpointEntry("reserved", replace(construct, core="CCDEFGHIKL"), "reserved",
                      "reserved-group", "synthetic", "synthetic score"),
    ))


def _record(**changes):
    row = InputRecord("broad", "source", "source-row", "AAAA", "CCCC", "DDDD", "paired",
                      "broad-antibody", "broad-target", "train", (0, 1), (0,), None,
                      "synthetic reference", "synthetic assay", "positive", "positive")
    return replace(row, **changes)


def _population(*records):
    declaration = ManifestDeclaration(
        (SourceDeclaration("source", "synthetic", 0),),
        (QualificationRule("positive", "synthetic positive"),), ("source",),
        GroupingDeclaration(("antibody_family", "target_family")),
    )
    return qualify_manifest(declaration, records or (_record(),))


def _qualify(population=None, manifest=None, **kwargs):
    arguments = {"excluded_antibody_families": ["downstream-antibody"],
                 "excluded_target_families": ["downstream-target"]}
    arguments.update(kwargs)
    return qualify_transfer(population or _population(), manifest or _manifest(), **arguments)


def _post(manifest=None):
    manifest = manifest or _manifest()
    budget = select_budget(manifest, 1, "shared")
    return RestrictedPostData(
        manifest=manifest, budget=budget,
        measured=(MeasuredEndpoint("fit", 1.0, "higher_better", "synthetic"),),
        fitting_count=1, development_labels=(), selection_allowance=SelectionAllowance(0, ()),
    )


def _release(tiny=False):
    base = {"variant": "synthetic-test-only" if tiny else "dplm_650m", "pretrained": not tiny}
    return SimpleNamespace(base_identity=base), {"kind": "release", "base": copy.deepcopy(base)}


def _parent(mode="lora"):
    release, origin = _release()
    certificate = _qualify()
    source = {"base": release.base_identity, "update_mode": mode}
    training = {
        "schema": "trastuzumab-training/1", "selection": "fixed-last",
        "config": {"stage": "mid", "total_updates": 2}, "model": copy.deepcopy(source),
        "initialization": origin, "transfer_certificate": certificate,
        "data": {"manifest": certificate["declaration_digest"], "population": certificate["population_digest"]},
    }
    progress = {"next_update": 2, "fitting_endpoints_available": 0,
                "selection_labels": 0, "inherited_exposure": None}
    base = {"variant": "materialized-parent", "parent_sha256": "a" * 64,
            "source_policy_identity": source,
            "parent_provenance": {"checkpoint_sha256": "b" * 64,
                                  "training": training, "progress": progress}}
    return SimpleNamespace(base_identity=base), {"kind": "parent", "sha256": "a" * 64,
                                               "base": copy.deepcopy(base)}


@pytest.mark.parametrize("case", ("exact", "antibody_other_target", "target_other_antibody",
                                  "antibody_family", "target_fragment_family"))
def test_transfer_rejects_downstream_exposure_across_documents(case):
    heldout = _manifest().by_id("reserved").construct
    changes = {
        "exact": {"H": heldout.full_heavy(), "L": heldout.light, "Ag": heldout.Ag},
        "antibody_other_target": {"H": heldout.full_heavy(), "L": heldout.light},
        "target_other_antibody": {"Ag": heldout.Ag},
        "antibody_family": {"antibody_family": "downstream-antibody"},
        "target_fragment_family": {"Ag": "QRS", "target_family": "downstream-target"},
    }
    with pytest.raises(ValueError, match="downstream"):
        _qualify(_population(_record(**changes[case])))


def test_transfer_is_outcome_free_canonical_and_copies_family_lists():
    families = ["z", "downstream-antibody"]
    first = _qualify(excluded_antibody_families=families)
    second = _qualify(excluded_antibody_families=list(reversed(families)))
    assert first == second
    families.clear()
    assert first["excluded_antibody_families"] == ["downstream-antibody", "z"]
    assert first["endpoint_manifest_digest"] == _manifest().manifest_digest()
    assert first["coverage"]["protected_endpoint_roles"] == ["post_pool", "reserved"]


@pytest.mark.parametrize("bad", (None, [], "a", [""], [" x"], ["a", "a"], [1]))
@pytest.mark.parametrize("field", ("excluded_antibody_families", "excluded_target_families"))
def test_transfer_requires_explicit_valid_curated_family_lists(field, bad):
    with pytest.raises(ValueError):
        _qualify(**{field: bad})


def test_transfer_coverage_describes_only_train_population_and_eligibility():
    heldout = _manifest().by_id("reserved").construct
    validation = _record(record_id="val", source_record_id="val", partition="val",
                         H=heldout.full_heavy(), L=heldout.light, Ag=heldout.Ag,
                         antibody_family="downstream-antibody", target_family="downstream-target")
    certificate = _qualify(_population(_record(), validation))
    assert certificate["coverage"]["train_records"] == 1
    assert certificate["coverage"]["train_unique_antigens"] == 1
    assert certificate["coverage"]["eligible_residues_min"] == 3
    assert certificate["coverage"]["records_with_light_eligible"] == 1


def test_transfer_rejects_empty_train_population():
    with pytest.raises(ValueError, match="nonempty training"):
        _qualify(_population(_record(partition="val")))


def test_mid_released_requires_certificate_and_tiny_smoke_may_omit_it():
    policy, origin = _release()
    with pytest.raises(ValueError, match="requires transfer qualification"):
        validate_stage_lineage(policy, SimpleNamespace(stage="mid"), origin)
    certificate = _qualify()
    assert validate_stage_lineage(policy, SimpleNamespace(stage="mid"), origin,
                                  transfer_certificate=certificate) == certificate
    tiny, tiny_origin = _release(tiny=True)
    assert validate_stage_lineage(tiny, SimpleNamespace(stage="mid"), tiny_origin) is None


def test_direct_release_post_needs_no_broad_certificate():
    policy, origin = _release()
    assert validate_stage_lineage(policy, SimpleNamespace(stage="post"), origin, post_data=_post()) is None


@pytest.mark.parametrize("mode", ("lora", "full"))
def test_completed_certified_mid_parent_accepts_same_manifest_post(mode):
    policy, origin = _parent(mode)
    certificate = validate_stage_lineage(policy, SimpleNamespace(stage="post"), origin, post_data=_post())
    assert certificate == _qualify()
    certificate["excluded_antibody_families"].clear()
    assert policy.base_identity["parent_provenance"]["training"]["transfer_certificate"]["excluded_antibody_families"]


@pytest.mark.parametrize("case", ("post_parent", "incomplete", "fitting", "selection", "inherited",
                                  "uncertified", "changed_model", "changed_population", "recursive"))
def test_parent_cannot_smuggle_labels_or_unsupported_ancestry(case):
    policy, _ = _parent()
    base = policy.base_identity
    training, progress = base["parent_provenance"]["training"], base["parent_provenance"]["progress"]
    if case == "post_parent":
        training["config"]["stage"] = "post"
    elif case == "incomplete":
        progress["next_update"] = 1
    elif case == "fitting":
        progress["fitting_endpoints_available"] = 1000
    elif case == "selection":
        progress["selection_labels"] = 1
    elif case == "inherited":
        progress["inherited_exposure"] = {"parent_progress": {"fitting_endpoints_available": 1000}}
    elif case == "uncertified":
        training["transfer_certificate"] = None
    elif case == "changed_model":
        training["model"]["update_mode"] = "full"
    elif case == "changed_population":
        training["data"]["population"] = "f" * 64
    else:
        training["initialization"] = {"kind": "parent", "base": training["model"]["base"], "sha256": "c" * 64}
    origin = {"kind": "parent", "sha256": "a" * 64, "base": copy.deepcopy(base)}
    with pytest.raises(ValueError):
        validate_stage_lineage(policy, SimpleNamespace(stage="post"), origin, post_data=_post())


def test_parent_rejects_changed_downstream_manifest_before_post():
    policy, origin = _parent()
    old = _manifest()
    changed = EndpointManifest((old.entries[0], replace(old.entries[1], endpoint_id="another-reserved")))
    with pytest.raises(ValueError, match="different endpoint manifest"):
        validate_stage_lineage(policy, SimpleNamespace(stage="post"), origin, post_data=_post(changed))


def test_initialization_metadata_cannot_claim_release_for_materialized_parent():
    policy, _ = _parent()
    false_origin = {"kind": "release", "base": copy.deepcopy(policy.base_identity)}
    with pytest.raises(ValueError, match="hide a materialized parent"):
        validate_stage_lineage(policy, SimpleNamespace(stage="post"), false_origin, post_data=_post())


def test_mid_cannot_start_from_parent_even_with_valid_certificate():
    policy, origin = _parent()
    with pytest.raises(ValueError, match="release initialization"):
        validate_stage_lineage(policy, SimpleNamespace(stage="mid"), origin, transfer_certificate=_qualify())


def test_origin_must_bind_actual_backbone_and_parent_hash():
    policy, origin = _parent()
    origin["sha256"] = "c" * 64
    with pytest.raises(ValueError, match="actual parent"):
        validate_stage_lineage(policy, SimpleNamespace(stage="post"), origin, post_data=_post())
    origin["base"] = {"variant": "dplm_650m", "pretrained": True}
    with pytest.raises(ValueError, match="actual policy backbone"):
        validate_stage_lineage(policy, SimpleNamespace(stage="post"), origin, post_data=_post())


def test_certificate_tampering_cannot_be_promoted():
    policy, origin = _release()
    certificate = _qualify()
    certificate["endpoint_manifest_digest"] = "f" * 64
    with pytest.raises(ValueError, match="certificate digest"):
        validate_stage_lineage(policy, SimpleNamespace(stage="mid"), origin, transfer_certificate=certificate)
