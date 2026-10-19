"""Outcome-free transfer qualification and the fixed release -> mid -> post lineage.

These checks protect this experiment, not the generic parent serialization API.
Sequence equality is a backstop; biological family assignments remain curated.
No helper discovers data, opens labels, or infers family identities from names.
"""
from __future__ import annotations

import copy

from smallAntibodyGen.experiments.trastuzumab_poc.artifacts import digest, require_keys
from smallAntibodyGen.experiments.trastuzumab_poc.data import QualifiedPopulation
from smallAntibodyGen.experiments.trastuzumab_poc.functional_checkpoint import validate_functional_training
from smallAntibodyGen.experiments.trastuzumab_poc.functional_data import QualifiedFunctional
from smallAntibodyGen.experiments.trastuzumab_poc.protocol import EndpointManifest, RestrictedPostData


# The training-identity version that this lineage guard accepts for a parent.
TRAINING_SCHEMA = "trastuzumab-training/2"
_SCHEMA = "trastuzumab-transfer-qualification/2"
_CERTIFICATE_KEYS = (
    "schema", "population_digest", "declaration_digest", "endpoint_manifest_digest",
    "excluded_antibody_families", "excluded_target_families", "coverage", "functional",
    "certificate_sha256",
)
_FUNCTIONAL_DIGESTS = ("qualified_digest", "contract_digest", "declaration_digest",
                       "population_digest", "broad_input_digest")
_FUNCTIONAL_KEYS = (*_FUNCTIONAL_DIGESTS, "train_constructs", "train_unique_antibodies",
                    "train_unique_antigens", "train_antibody_families", "train_target_families")


def _families(values, name):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{name} must be an explicit nonempty list of curated family IDs")
    if any(not isinstance(v, str) or not v or v.strip() != v for v in values):
        raise ValueError(f"{name} must contain nonempty family IDs without surrounding whitespace")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicate family IDs")
    return sorted(values)


def _sha(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA256")


def functional_binding(qualified):
    """The part of a certificate that binds one functional qualification, budget-independently."""
    train = qualified.train_constructs()
    return {
        "qualified_digest": qualified.digest, "contract_digest": qualified.contract_digest,
        "declaration_digest": qualified.declaration_digest,
        "population_digest": qualified.population_digest,
        "broad_input_digest": qualified.broad_input_digest,
        "train_constructs": len(train),
        "train_unique_antibodies": len({(c.H, c.L) for c in train}),
        "train_unique_antigens": len({c.Ag for c in train}),
        "train_antibody_families": sorted({c.antibody_family for c in train}),
        "train_target_families": sorted({c.target_family for c in train}),
    }


def qualify_transfer(population, endpoint_manifest, *, excluded_antibody_families,
                     excluded_target_families, functional=None):
    """Reject downstream leakage in the actual broad TRAIN population.

    All endpoint roles protect their exact H/L construct and antigen sequence.
    Explicit family exclusions additionally protect unlisted variants and target
    fragments when curators have assigned their families consistently. Previously
    excluded records and validation/test records are not training exposure.

    ``functional`` (a ``QualifiedFunctional``) extends the same checks to EVERY functional
    train construct, selected for the observation budget or not, so the certificate does not
    depend on which budget is drawn.  The certificate then binds the full functional
    qualification identity.  The denoising population is never touched by it.
    """
    if not isinstance(population, QualifiedPopulation) or not isinstance(endpoint_manifest, EndpointManifest):
        raise TypeError("transfer qualification requires a qualified population and endpoint manifest")
    if functional is not None and not isinstance(functional, QualifiedFunctional):
        raise TypeError("functional must be a QualifiedFunctional")
    ab_families = _families(excluded_antibody_families, "excluded_antibody_families")
    ag_families = _families(excluded_target_families, "excluded_target_families")
    protected_antibodies = {(e.construct.full_heavy(), e.construct.light) for e in endpoint_manifest.entries}
    protected_antigens = {e.construct.Ag for e in endpoint_manifest.entries}
    train = population.train_records()
    if not train:
        raise ValueError("transfer qualification requires a nonempty training population")
    for row in train:
        if (row.H, row.L) in protected_antibodies:
            raise ValueError(f"downstream antibody construct appears in broad training: {row.record_id}")
        if row.Ag in protected_antigens:
            raise ValueError(f"downstream antigen sequence appears in broad training: {row.record_id}")
        if row.antibody_family in ab_families:
            raise ValueError(f"excluded downstream antibody family appears in broad training: {row.record_id}")
        if row.target_family in ag_families:
            raise ValueError(f"excluded downstream target family appears in broad training: {row.record_id}")
    binding = None
    if functional is not None:
        if (functional.declaration_digest != population.manifest_digest
                or functional.population_digest != population.population_digest):
            raise ValueError("functional qualification belongs to a different broad population")
        if not functional.train_constructs():
            raise ValueError("functional qualification has no training construct")
        for item in functional.train_constructs():
            if (item.H, item.L) in protected_antibodies:
                raise ValueError(f"downstream antibody construct appears in functional training: {item.construct_id}")
            if item.Ag in protected_antigens:
                raise ValueError(f"downstream antigen sequence appears in functional training: {item.construct_id}")
            if item.antibody_family in ab_families:
                raise ValueError(f"excluded downstream antibody family appears in functional training: {item.construct_id}")
            if item.target_family in ag_families:
                raise ValueError(f"excluded downstream target family appears in functional training: {item.construct_id}")
        binding = functional_binding(functional)
    eligible = [len(row.eligible_union()) for row in train]
    certificate = {
        "schema": _SCHEMA,
        "population_digest": population.population_digest,
        "declaration_digest": population.manifest_digest,
        "endpoint_manifest_digest": endpoint_manifest.manifest_digest(),
        "excluded_antibody_families": ab_families,
        "excluded_target_families": ag_families,
        "coverage": {
            "scope": "qualified_train_partition",
            "train_records": len(train),
            "train_unique_antibodies": len({(r.H, r.L) for r in train}),
            "train_unique_antigens": len({r.Ag for r in train}),
            "train_antibody_families": sorted({r.antibody_family for r in train}),
            "train_target_families": sorted({r.target_family for r in train}),
            "eligible_residues_min": min(eligible),
            "eligible_residues_max": max(eligible),
            "records_with_light_eligible": sum(bool(r.l_eligible) for r in train),
            "records_with_aux_region": sum(bool(r.aux_region) for r in train),
            "protected_endpoint_roles": sorted({e.role for e in endpoint_manifest.entries}),
        },
        "functional": binding,
    }
    certificate["certificate_sha256"] = digest(certificate)
    return certificate


def _functional_binding(value, certificate):
    if value is None:
        return
    require_keys(value, _FUNCTIONAL_KEYS, "functional transfer binding")
    for name in _FUNCTIONAL_DIGESTS:
        _sha(value[name], name)
    if (value["declaration_digest"] != certificate["declaration_digest"]
            or value["population_digest"] != certificate["population_digest"]):
        raise ValueError("functional binding does not match the certificate's broad population")
    for name in ("train_constructs", "train_unique_antibodies", "train_unique_antigens"):
        if type(value[name]) is not int or value[name] < 1:
            raise ValueError("functional binding counts must be positive integers")
    if (value["train_unique_antibodies"] > value["train_constructs"]
            or value["train_unique_antigens"] > value["train_constructs"]):
        raise ValueError("inconsistent functional binding")
    for name in ("train_antibody_families", "train_target_families"):
        if value[name] != _families(value[name], name):
            raise ValueError("functional binding families must be canonical sorted lists")


def _certificate(value):
    require_keys(value, _CERTIFICATE_KEYS, "transfer certificate")
    if value["schema"] != _SCHEMA:
        raise ValueError("unsupported transfer certificate schema")
    for name in ("population_digest", "declaration_digest", "endpoint_manifest_digest", "certificate_sha256"):
        _sha(value[name], name)
    for name in ("excluded_antibody_families", "excluded_target_families"):
        if value[name] != _families(value[name], name):
            raise ValueError("transfer certificate family IDs must be canonical sorted lists")
    coverage = value["coverage"]
    require_keys(coverage, ("scope", "train_records", "train_unique_antibodies", "train_unique_antigens",
                           "train_antibody_families", "train_target_families", "eligible_residues_min",
                           "eligible_residues_max", "records_with_light_eligible", "records_with_aux_region",
                           "protected_endpoint_roles"), "transfer coverage")
    if coverage["scope"] != "qualified_train_partition":
        raise ValueError("transfer coverage must describe the training partition")
    for name in ("train_records", "train_unique_antibodies", "train_unique_antigens",
                 "eligible_residues_min", "eligible_residues_max"):
        if type(coverage[name]) is not int or coverage[name] < 1:
            raise ValueError("transfer coverage counts must be positive integers")
    for name in ("records_with_light_eligible", "records_with_aux_region"):
        if type(coverage[name]) is not int or not 0 <= coverage[name] <= coverage["train_records"]:
            raise ValueError("invalid transfer eligibility coverage")
    if (coverage["train_unique_antibodies"] > coverage["train_records"]
            or coverage["train_unique_antigens"] > coverage["train_records"]
            or coverage["eligible_residues_min"] > coverage["eligible_residues_max"]):
        raise ValueError("inconsistent transfer coverage")
    for name in ("train_antibody_families", "train_target_families"):
        if coverage[name] != _families(coverage[name], name):
            raise ValueError("transfer coverage families must be canonical sorted lists")
    roles = coverage["protected_endpoint_roles"]
    if (not isinstance(roles, list) or not roles or roles != sorted(set(roles))
            or any(r not in ("post_pool", "development", "reserved") for r in roles)):
        raise ValueError("invalid protected endpoint roles")
    _functional_binding(value["functional"], value)
    unsigned ={k: v for k, v in value.items() if k != "certificate_sha256"}
    if digest(unsigned) != value["certificate_sha256"]:
        raise ValueError("transfer certificate digest mismatch")
    return copy.deepcopy(value)


def _parent_functional(training, progress, certificate):
    """A parent's functional records are explicit, internally consistent and certified.

    The derived ``input`` records are required even without a functional arm: a midtraining
    parent has no post fitting budget, and an absent arm is ``null`` in the identity, the
    progress, the input AND the certificate (never merely some of them).  Other keys of
    ``input`` are informational and are not constrained here.
    """
    recorded_input = training.get("input")
    if ("functional" not in training or "functional" not in progress
            or not isinstance(recorded_input, dict)
            or not {"budget", "functional"} <= set(recorded_input)):
        raise ValueError("parent training records lack the explicit budget and functional fields")
    if recorded_input["budget"] is not None:
        raise ValueError("midtraining parent records a post fitting budget")
    try:
        functional = validate_functional_training(training, progress)
    except (KeyError, TypeError) as error:
        raise ValueError(f"parent functional records are malformed: {error!r}") from error
    binding = certificate["functional"]
    if functional is None:
        if binding is not None:
            raise ValueError("parent certificate binds a functional qualification that its "
                             "identity does not declare")
        return
    recorded = recorded_input["functional"]["budget"]
    if (binding is None or binding["qualified_digest"] != recorded["qualified_digest"]
            or binding["contract_digest"] != recorded["contract_digest"]):
        raise ValueError("parent transfer certificate does not bind its functional qualification")


def _release_origin(origin, base):
    require_keys(origin, ("kind", "base"), "release initialization")
    if origin["kind"] != "release" or origin["base"] != base:
        raise ValueError("release initialization does not match the actual backbone")
    if not isinstance(base, dict) or base.get("variant") == "materialized-parent":
        raise ValueError("release initialization cannot hide a materialized parent")
    if base.get("variant") != "synthetic-test-only" and base.get("pretrained") is not True:
        raise ValueError("release initialization requires a pinned pretrained base")


def validate_stage_lineage(policy, config, initialization_identity, *, transfer_certificate=None,
                           post_data=None, functional=None):
    """Validate this experiment's ancestry, returning its qualification certificate.

    Generic parent exports remain reusable elsewhere. This experiment permits
    only release -> mid -> post or release -> direct post. A measured parent must
    never masquerade as a smaller fresh measurement budget.

    ``functional`` is the ``QualifiedFunctional`` of a midtraining run; its certificate must
    bind exactly that qualification.  Post-training takes no functional supervision, and a
    functional parent must carry a certificate that binds the functional qualification
    recorded in its own training identity.
    """
    base = policy.base_identity
    if not isinstance(initialization_identity, dict) or initialization_identity.get("base") != base:
        raise ValueError("initialization identity does not match the actual policy backbone")
    kind = initialization_identity.get("kind")
    if config.stage not in ("mid", "post"):
        raise ValueError("unsupported transfer stage")
    if functional is not None and not isinstance(functional, QualifiedFunctional):
        raise TypeError("functional must be a QualifiedFunctional")
    if config.stage == "mid":
        _release_origin(initialization_identity, base)
        if transfer_certificate is None:
            if base.get("variant") != "synthetic-test-only":
                raise ValueError("released midtraining requires transfer qualification")
            return None
        certificate = _certificate(transfer_certificate)
        if certificate["functional"] != (None if functional is None else functional_binding(functional)):
            raise ValueError("transfer certificate does not bind this run's functional qualification")
        return certificate
    if functional is not None:
        raise ValueError("post-training takes no functional supervision")
    if not isinstance(post_data, RestrictedPostData):
        raise TypeError("post-training lineage requires restricted selected data")
    if kind == "release":
        _release_origin(initialization_identity, base)
        if transfer_certificate is not None:
            raise ValueError("direct release adaptation has no broad transfer certificate")
        return None
    require_keys(initialization_identity, ("kind", "base", "sha256"), "parent initialization")
    if (kind != "parent" or base.get("variant") != "materialized-parent"
            or initialization_identity["sha256"] != base.get("parent_sha256")):
        raise ValueError("parent initialization does not match the actual parent")
    _sha(initialization_identity["sha256"], "parent_sha256")
    provenance = base.get("parent_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("parent lacks recorded training provenance")
    require_keys(provenance, ("checkpoint_sha256", "training", "progress"), "parent provenance")
    _sha(provenance["checkpoint_sha256"], "parent checkpoint_sha256")
    training, progress = provenance["training"], provenance["progress"]
    if not isinstance(training, dict) or not isinstance(progress, dict):
        raise ValueError("parent training and progress must be objects")
    source = base.get("source_policy_identity")
    if not isinstance(source, dict) or training.get("model") != source:
        raise ValueError("parent training model differs from the materialized source policy")
    parent_config = training.get("config")
    if (training.get("schema") != TRAINING_SCHEMA
            or training.get("selection") != "fixed-last"
            or not isinstance(parent_config, dict) or parent_config.get("stage") != "mid"):
        raise ValueError("post-training requires a midtraining parent, never a measured post-training parent")
    _release_origin(training.get("initialization"), source.get("base"))
    horizon = parent_config.get("total_updates")
    if (type(horizon) is not int or horizon < 1 or type(progress.get("next_update")) is not int
            or progress["next_update"] != horizon):
        raise ValueError("post-training requires a completed midtraining parent")
    if (type(progress.get("fitting_endpoints_available")) is not int
            or progress["fitting_endpoints_available"] != 0
            or type(progress.get("selection_labels")) is not int or progress["selection_labels"] != 0
            or progress.get("inherited_exposure", "missing") is not None):
        raise ValueError("midtraining parent has inherited fitting or selection exposure")
    certificate = _certificate(training.get("transfer_certificate"))
    if training.get("data") != {"manifest": certificate["declaration_digest"],
                                "population": certificate["population_digest"]}:
        raise ValueError("parent transfer certificate does not bind its actual training population")
    if certificate["endpoint_manifest_digest"] != post_data.budget.manifest_digest:
        raise ValueError("parent transfer qualification protects a different endpoint manifest")
    _parent_functional(training, progress, certificate)
    if transfer_certificate is not None and _certificate(transfer_certificate) != certificate:
        raise ValueError("requested transfer certificate differs from the trained parent")
    return certificate
