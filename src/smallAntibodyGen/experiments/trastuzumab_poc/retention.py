"""Frozen held-out reconstruction retention (M3): an outcome-free plan and a no-update readout.

A fixed set of broad val/test records, each with fixed literal masks, is scored by the same
mean-site reconstruction NLL that midtraining minimizes, with the same dynamic frozen-encoder
context (a fresh provider on the CURRENT masked combined view for every mask).  Run on one plan
before and after midtraining or posttraining it shows how much broad masked reconstruction a
checkpoint kept or lost.  It is software machinery: it is NOT binding retention, NOT evidence of
biological success, and it opens no model, weight, label or measurement while planning.

Planning (``plan_retention``) reads only the explicitly named files.  Records come from the
QUALIFIED population's val/test partitions, never from raw, excluded or duplicate rows.  Every
selected record is checked against ALL raw denoising training rows, ALL functional training
constructs (selected for a budget or not) and ALL protected endpoint entries on the exact (H, L)
pair, the exact antigen, and the declared family axes.  Family identities are curator assertions:
the plan reports which axes were checked and never claims independent homology.

Evaluation (``evaluate_retention``) preflights every view before the first forward, then runs
under ``frozen_evaluation``: policy eval mode, no gradients, no optimizer, and exact restoration
of every module's training flag and of the Python and Torch generators.

Trust model.  ``validate_plan`` checks a serialized plan's shape, digest and internal consistency
(counts, roles, duplicate content, protected families) and re-derives every mask.  It does NOT
re-establish, from digests alone, that the cases are qualified records of their partitions, that a
limit's ranking chose them, or that they are disjoint from the training and protected populations:
the plan carries digests, not those populations, so that was checked once, by ``build_plan``, against
the full inputs.  Pin the ``plan_sha256`` of the plan ``plan-retention`` wrote; an author who
recomputes every digest can write a self-consistent plan that validation alone cannot tell apart.
``check_lineage`` binds a checkpoint to the plan only through what the checkpoint recorded about its
own training: the raw broad input file hash, the population digests, the transfer certificate and the
functional qualification.
"""
from __future__ import annotations

import copy
import math
import random
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import torch

from smallAntibodyGen.experiments.trastuzumab_poc.artifacts import (
    canonical_bytes, digest, file_digest, positive_count, read_json, require_keys, seed_value, text_value,
)
from smallAntibodyGen.experiments.trastuzumab_poc.data import _content_key, qualify_manifest
from smallAntibodyGen.experiments.trastuzumab_poc.functional_data import (
    FunctionalConstructs, load_functional_constructs, qualify_functional, sha256_hex,
)
from smallAntibodyGen.experiments.trastuzumab_poc.inputs import load_broad_context, load_endpoint_manifest
from smallAntibodyGen.experiments.trastuzumab_poc.objectives import UniformMaskSpec, draw_mask_uniform
from smallAntibodyGen.experiments.trastuzumab_poc.post_budget import selection_key, validate_post_budget_identity
from smallAntibodyGen.experiments.trastuzumab_poc.protocol import ENDPOINT_ROLES, EndpointManifest
from smallAntibodyGen.experiments.trastuzumab_poc.randomness import RNG_ALGORITHM, keyed_seed
# The lineage validators are private to the transfer guard because they only mean something inside
# this campaign's release -> mid -> post chain; retention reuses them rather than restating them.
from smallAntibodyGen.experiments.trastuzumab_poc.transfer_guard import (
    TRAINING_SCHEMA, _certificate, _families, _parent_functional, _release_origin,
)
from smallAntibodyGen.models.antibody_antigen.dynamic_context import DynamicContextProvider
from smallAntibodyGen.models.antibody_antigen.views import CANONICAL_RESIDUES, mask_complete

SPEC_SCHEMA = "trastuzumab-retention-spec/1"
PLAN_SCHEMA = "trastuzumab-retention-plan/1"
RESULT_SCHEMA = "trastuzumab-retention-result/1"

SCOPE = ("frozen held-out masked-reconstruction NLL; software machinery, not binding retention and "
         "not evidence of biological success")
ASSUMPTIONS = (
    "partitions and family identities are curator-declared; exact sequence equality is the only "
    "independent check this plan makes",
    "chronology, source independence and external sequence-family truth are not established",
    "the frozen encoder's own pretraining data is not controlled: held-out records may have been "
    "seen by it",
    "homology between a held-out record and a training or protected record is not tested",
)
MASK_PURPOSE = "heldout-mask/"
MASK_RECIPE = {
    "kind": "uniform-k/1", "span_weight": 0.0, "full_region_weight": 0.0, "rng": RNG_ALGORITHM,
    "seed": "keyed_seed(seed, mask_index, 0, 'heldout-mask/' + record_id)",
    "union": "the record's eligible coordinates in qualified order (H ascending, then L ascending)",
    "repeats": "kept: an identical mask drawn twice is evaluated and counted twice"}
READOUT = {
    "name": "heldout_masked_reconstruction_nll", "unit": "nats per masked site",
    "orientation": "lower_is_better",
    "per_mask": "mean over the masked sites (the production reconstruction_nll)",
    "per_record": "arithmetic mean over the record's masks",
    "aggregate": "arithmetic mean over records; sites and masks are never pooled",
    "context": "fresh DynamicContextProvider on the current masked combined view for every mask",
    "mode": "policy eval, no_grad; no optimizer step, no gradient write; module flags and generators restored exactly"}

_PARTITIONS = ("val", "test")
_AXES = ("antibody_family", "target_family")
_SELECTION_RULES = ("every qualified record of the requested partitions",
                    "explicit record_ids drawn from the qualified requested partitions",
                    "first limit.count of the qualified pool by sha256(limit.seed NUL record_id)")
_GUARD_NOTE = ("exact (H, L), exact antigen and the declared family axes against the listed populations; "
               "family identities are curator assertions, not independently established homology")
_PLAN_KEYS = ("schema", "scope", "assumptions", "selection", "broad", "functional", "protected", "guard",
              "masks", "cases", "counts", "plan_sha256")
_CASE_KEYS = ("record_id", "partition", "antibody_format", "H", "L", "Ag", "antibody_family",
              "target_family", "content_key", "eligible", "masks")
_CANON = frozenset(CANONICAL_RESIDUES)
_ROLE_RANK = {"H": 0, "L": 1}
# ``ExactViewFeatures`` rejects NaN or infinite frozen-encoder representations with this ValueError.
# It is the one ValueError that is a numerical outcome rather than a structural error.
_NONFINITE_FEATURES = "features must all be finite"


# ---------------------------------------------------------------------------
# Specification
# ---------------------------------------------------------------------------

def _strings(value, name, allowed=None):
    if not isinstance(value, list) or not value or any(not isinstance(v, str) or not v for v in value):
        raise ValueError(f"{name} must be a nonempty list of nonempty strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} contains duplicates")
    if allowed is not None and set(value) - set(allowed):
        raise ValueError(f"{name} must be drawn from {list(allowed)}")
    return sorted(value)


def parse_spec(document):
    """Strict, path-preserving parse of a retention specification into its canonical form."""
    require_keys(document, ("schema", "pairs_path", "functional_constructs_path", "protected",
                            "selection", "masks"), "retention spec")
    if document["schema"] != SPEC_SCHEMA:
        raise ValueError("unsupported retention spec schema")
    protected, selection, masks = document["protected"], document["selection"], document["masks"]
    require_keys(protected, ("endpoint_manifest_path", "excluded_antibody_families",
                             "excluded_target_families", "family_axes"), "retention protected")
    require_keys(selection, ("partitions", "record_ids", "limit"), "retention selection")
    require_keys(masks, ("seed", "per_record"), "retention masks")
    text_value(document["pairs_path"], "pairs_path")
    text_value(protected["endpoint_manifest_path"], "endpoint_manifest_path")
    if document["functional_constructs_path"] is not None:
        text_value(document["functional_constructs_path"], "functional_constructs_path")
    limit, record_ids = selection["limit"], selection["record_ids"]
    if limit is not None:
        require_keys(limit, ("count", "seed"), "retention limit")
        positive_count(limit["count"], "limit count")
        text_value(limit["seed"], "limit seed")
        limit = {"count": limit["count"], "seed": limit["seed"]}
    if record_ids is not None:
        record_ids = _strings(record_ids, "record_ids")
        if limit is not None:
            raise ValueError("record_ids and limit are mutually exclusive")
    seed_value(masks["seed"])
    positive_count(masks["per_record"], "masks per_record")
    return {
        "pairs_path": document["pairs_path"], "functional_constructs_path": document["functional_constructs_path"],
        "endpoint_manifest_path": protected["endpoint_manifest_path"],
        "excluded_antibody_families": _families(protected["excluded_antibody_families"],
                                                "excluded_antibody_families"),
        "excluded_target_families": _families(protected["excluded_target_families"],
                                              "excluded_target_families"),
        "family_axes": _strings(protected["family_axes"], "family_axes", _AXES),
        "selection": {"partitions": _strings(selection["partitions"], "partitions", _PARTITIONS),
                      "record_ids": record_ids, "limit": limit},
        "masks": {"seed": masks["seed"], "per_record": masks["per_record"]},
    }


def load_spec(path):
    """Parse a specification file; relative paths are relative to the file, like a run config."""
    path = Path(path).resolve()
    spec = parse_spec(read_json(path))
    for key in ("pairs_path", "functional_constructs_path", "endpoint_manifest_path"):
        if spec[key] is not None:
            spec[key] = str((path.parent / spec[key]).resolve())
    return spec


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def _select(population, selection):
    partitions = selection["partitions"]
    pool = sorted((r for p in partitions for r in population.by_partition(p)), key=lambda r: r.record_id)
    if not pool:
        raise ValueError("the qualified population has no record in the requested held-out partitions")
    ids, limit = selection["record_ids"], selection["limit"]
    if ids is not None:
        known = {r.record_id: r for r in pool}
        unknown = [i for i in ids if i not in known]
        if unknown:
            raise ValueError(f"not qualified records of partitions {partitions} (excluded, duplicate, "
                             f"training and unknown rows are never selectable): {unknown}")
        chosen, rule = [known[i] for i in ids], _SELECTION_RULES[1]
    elif limit is not None:
        if limit["count"] > len(pool):
            raise ValueError("limit.count exceeds the qualified held-out pool")
        chosen = sorted(pool, key=lambda r: selection_key(limit["seed"], r.record_id))[:limit["count"]]
        rule = _SELECTION_RULES[2]
    else:
        chosen, rule = pool, _SELECTION_RULES[0]
    return sorted(chosen, key=lambda r: r.record_id), {p: len(population.by_partition(p)) for p in partitions}, rule


def check_disjoint(selected, *, train_rows, functional_train, manifest, excluded_antibody_families,
                   excluded_target_families, axes):
    """Fail closed when a selected record overlaps a training or protected population.

    ``train_rows`` are ALL raw denoising training rows (excluded ones too) and
    ``functional_train`` ALL functional training constructs, so no budget or exclusion can hide an
    overlap.  Exact (H, L) is checked regardless of the antigen and the exact antigen regardless
    of (H, L); the listed family axes are checked on top.  Protected endpoint entries of every
    role protect their exact antibody, antigen and both explicit family lists.
    """
    populations = {"denoising training row": train_rows, "functional training construct": functional_train}
    for label, rows in populations.items():
        antibodies, antigens = {(r.H, r.L) for r in rows}, {r.Ag for r in rows}
        families = {axis: {getattr(r, axis) for r in rows} for axis in axes}
        for record in selected:
            if (record.H, record.L) in antibodies:
                raise ValueError(f"held-out {record.record_id} shares its (H, L) with a {label}")
            if record.Ag in antigens:
                raise ValueError(f"held-out {record.record_id} shares its antigen with a {label}")
            for axis, values in families.items():
                if getattr(record, axis) in values:
                    raise ValueError(f"held-out {record.record_id} shares its {axis} with a {label}")
    antibodies = {(e.construct.full_heavy(), e.construct.light) for e in manifest.entries}
    antigens = {e.construct.Ag for e in manifest.entries}
    for record in selected:
        if (record.H, record.L) in antibodies:
            raise ValueError(f"held-out {record.record_id} shares its (H, L) with a protected endpoint")
        if record.Ag in antigens:
            raise ValueError(f"held-out {record.record_id} shares its antigen with a protected endpoint")
        if record.antibody_family in excluded_antibody_families:
            raise ValueError(f"held-out {record.record_id} is in a protected antibody family")
        if record.target_family in excluded_target_families:
            raise ValueError(f"held-out {record.record_id} is in a protected target family")


def draw_masks(record_id, union, seed, count):
    """The literal masks of one record; ``validate_plan`` re-derives them with this same function."""
    masks = []
    for index in range(count):
        rng = random.Random(keyed_seed(seed, index, 0, MASK_PURPOSE + record_id))
        masks.append([list(site) for site in draw_mask_uniform(union, rng, UniformMaskSpec()).sites])
    return masks


def _case(record, seed, count):
    union = record.eligible_union()
    return {"record_id": record.record_id, "partition": record.partition,
            "antibody_format": record.antibody_format, "H": record.H, "L": record.L, "Ag": record.Ag,
            "antibody_family": record.antibody_family, "target_family": record.target_family,
            "content_key": record.content_key, "eligible": [list(c) for c in union],
            "masks": draw_masks(record.record_id, union, seed, count)}


def build_plan(spec, *, declaration, records, population, manifest, functional=None, files):
    """The sealed, outcome-free plan.  No model, weight, label or measurement is touched.

    ``spec`` is :func:`parse_spec` output, ``records`` ALL raw broad rows, ``functional`` an optional
    :class:`FunctionalConstructs` and ``files`` the byte identities of the three input files.  The
    supplied population is not trusted: it must equal the qualification of ``records``.
    """
    if not isinstance(manifest, EndpointManifest):
        raise TypeError("manifest must be an EndpointManifest")
    if functional is not None and not isinstance(functional, FunctionalConstructs):
        raise TypeError("functional must be a FunctionalConstructs")
    require_keys(files, ("input_sha256", "endpoint_manifest_sha256", "functional_constructs_sha256"), "files")
    for name in ("input_sha256", "endpoint_manifest_sha256"):
        sha256_hex(files[name], name)
    if files["functional_constructs_sha256"] is not None:
        sha256_hex(files["functional_constructs_sha256"], "functional_constructs_sha256")
    if (functional is None) != (files["functional_constructs_sha256"] is None):
        raise ValueError("a functional document and its file identity are supplied together")
    records = tuple(records)
    if qualify_manifest(declaration, records) != population:
        raise ValueError("the population is not the qualification of the raw records")
    qualified = None if functional is None else qualify_functional(functional, declaration, records,
                                                                    population=population)
    raw_digest = digest(sorted((asdict(r) for r in records), key=lambda r: r["record_id"]))
    train_rows = [r for r in records if r.partition == "train"]
    if not train_rows:
        raise ValueError("held-out retention needs a training population to be held out from")
    selected, pool, rule = _select(population, spec["selection"])
    functional_train = () if qualified is None else qualified.train_constructs()
    check_disjoint(selected, train_rows=train_rows, functional_train=functional_train, manifest=manifest,
                   excluded_antibody_families=set(spec["excluded_antibody_families"]),
                   excluded_target_families=set(spec["excluded_target_families"]), axes=spec["family_axes"])
    masks = spec["masks"]
    cases = [_case(r, masks["seed"], masks["per_record"]) for r in selected]
    plan = {
        "schema": PLAN_SCHEMA, "scope": SCOPE, "assumptions": list(ASSUMPTIONS),
        "selection": {**copy.deepcopy(spec["selection"]), "rule": rule, "pool": pool, "selected": len(cases)},
        "broad": {"input_sha256": files["input_sha256"], "raw_input_digest": raw_digest,
                  "raw_rows": len(records), "declaration_digest": population.manifest_digest,
                  "population_digest": population.population_digest},
        "functional": None if qualified is None else {
            "constructs_sha256": files["functional_constructs_sha256"], "qualified_digest": qualified.digest,
            "contract_digest": qualified.contract_digest, "broad_input_digest": qualified.broad_input_digest,
            "declaration_digest": qualified.declaration_digest, "population_digest": qualified.population_digest,
            "train_constructs": len(functional_train)},
        "protected": {"endpoint_manifest_sha256": files["endpoint_manifest_sha256"],
                      "endpoint_manifest_digest": manifest.manifest_digest(),
                      "roles": sorted({e.role for e in manifest.entries}), "entries": len(manifest.entries),
                      "excluded_antibody_families": list(spec["excluded_antibody_families"]),
                      "excluded_target_families": list(spec["excluded_target_families"])},
        "guard": {"exact_antibody": True, "exact_antigen": True, "family_axes": list(spec["family_axes"]),
                  "against": {"denoising_train_raw_rows": len(train_rows),
                              "functional_train_constructs": None if qualified is None else len(functional_train),
                              "protected_endpoint_entries": len(manifest.entries)},
                  "assertion": _GUARD_NOTE},
        "masks": {"seed": masks["seed"], "per_record": masks["per_record"], "recipe": copy.deepcopy(MASK_RECIPE)},
        "cases": cases,
        "counts": _counts(cases),
    }
    plan["plan_sha256"] = digest(plan)
    return plan


def plan_retention(spec_path):
    """File-level planning: only the three explicitly named input files are read."""
    spec = load_spec(spec_path)
    declaration, records, population, identity = load_broad_context(spec["pairs_path"])
    functional_path = spec["functional_constructs_path"]
    return build_plan(
        spec, declaration=declaration, records=records, population=population,
        manifest=load_endpoint_manifest(spec["endpoint_manifest_path"]),
        functional=None if functional_path is None else load_functional_constructs(functional_path),
        files={"input_sha256": identity["input_sha256"],
               "endpoint_manifest_sha256": file_digest(spec["endpoint_manifest_path"]),
               "functional_constructs_sha256": None if functional_path is None else file_digest(functional_path)})


def _counts(cases):
    return {"records": len(cases), "masks": sum(len(c["masks"]) for c in cases),
            "masked_residues": sum(len(m) for c in cases for m in c["masks"])}


# ---------------------------------------------------------------------------
# Plan validation (before any model is touched)
# ---------------------------------------------------------------------------

def _count(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _sequence(value, name, optional=False):
    if value is None and optional:
        return
    if not isinstance(value, str) or not value or set(value) - _CANON:
        raise ValueError(f"{name} must be a nonempty canonical amino-acid string")


def _coords(value, name):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty list of [role, index] coordinates")
    for coord in value:
        if (not isinstance(coord, list) or len(coord) != 2 or coord[0] not in _ROLE_RANK
                or type(coord[1]) is not int or coord[1] < 0):
            raise ValueError(f"{name} holds an invalid coordinate: {coord!r}")
    if len({tuple(c) for c in value}) != len(value):
        raise ValueError(f"{name} holds a duplicate coordinate")
    return [tuple(c) for c in value]


def _validate_case(case, masks, partitions):
    require_keys(case, _CASE_KEYS, "retention case")
    name = case["record_id"]
    text_value(name, "record_id")
    for key in ("antibody_family", "target_family"):
        text_value(case[key], f"{name} {key}")
    if case["partition"] not in partitions:
        raise ValueError(f"case {name} is outside the selected partitions")
    _sequence(case["H"], f"{name} H")
    _sequence(case["L"], f"{name} L", optional=True)
    _sequence(case["Ag"], f"{name} Ag")
    if case["antibody_format"] != ("paired" if case["L"] is not None else "heavy_only"):
        raise ValueError(f"case {name} format and light chain disagree")
    if case["content_key"] != _content_key(case["H"], case["L"], case["Ag"]):
        raise ValueError(f"case {name} content key does not match its sequences")
    lengths = {"H": len(case["H"]), "L": 0 if case["L"] is None else len(case["L"])}
    eligible = _coords(case["eligible"], f"{name} eligible")
    if any(i >= lengths[role] for role, i in eligible) or eligible != sorted(
            eligible, key=lambda c: (_ROLE_RANK[c[0]], c[1])):
        raise ValueError(f"case {name} eligible coordinates are out of bounds or not in qualified order")
    if not isinstance(case["masks"], list) or len(case["masks"]) != masks["per_record"]:
        raise ValueError(f"case {name} does not hold exactly masks.per_record masks")
    for mask in case["masks"]:
        if not set(_coords(mask, f"{name} mask")) <= set(eligible):
            raise ValueError(f"case {name} masks a coordinate that is not eligible")
    if case["masks"] != draw_masks(name, eligible, masks["seed"], masks["per_record"]):
        raise ValueError(f"case {name} masks are not the recipe's masks")


def _validate_selection(selection):
    """The recorded selection rule is internally consistent; returns its partitions."""
    require_keys(selection, ("partitions", "record_ids", "limit", "rule", "pool", "selected"), "plan selection")
    partitions = _strings(selection["partitions"], "partitions", _PARTITIONS)
    ids, limit, pool = selection["record_ids"], selection["limit"], selection["pool"]
    if ids is not None:
        _strings(ids, "record_ids")
    if limit is not None:
        require_keys(limit, ("count", "seed"), "plan limit")
        positive_count(limit["count"], "limit count")
        text_value(limit["seed"], "limit seed")
    expected = 1 if ids is not None else 2 if limit is not None else 0
    if (selection["partitions"] != partitions or (ids is not None and limit is not None)
            or selection["rule"] != _SELECTION_RULES[expected]):
        raise ValueError("invalid plan selection record")
    require_keys(pool, partitions, "plan selection pool")
    for count in pool.values():
        _count(count, "selection pool", minimum=0)
    _count(selection["selected"], "selected")
    return partitions


def _validate_population(document, partitions):
    """Checks that need only the plan itself: the declared counts, content and families agree.

    These are contradictions a re-sealed plan can contain.  They are NOT a proof of membership:
    the pools are recorded counts, so a plan whose cases are consistent with its own counts but
    were never drawn from the real qualified population passes (see :func:`validate_plan`).
    """
    cases, selection, protected = document["cases"], document["selection"], document["protected"]
    pool = selection["pool"]
    selected = {p: sum(case["partition"] == p for case in cases) for p in partitions}
    if any(selected[p] > pool[p] for p in partitions):
        raise ValueError("a partition selects more records than its qualified pool holds")
    if selection["rule"] == _SELECTION_RULES[0] and selected != pool:
        raise ValueError("selecting every qualified record must select the whole pool of each partition")
    keys = [case["content_key"] for case in cases]
    if len(set(keys)) != len(keys):
        raise ValueError("two cases hold the same physical (H, L, Ag) content under different record ids")
    for case in cases:
        if (case["antibody_family"] in protected["excluded_antibody_families"]
                or case["target_family"] in protected["excluded_target_families"]):
            raise ValueError(f"case {case['record_id']} is in a protected family")
    accounted = document["guard"]["against"]["denoising_train_raw_rows"] + sum(pool.values())
    if document["broad"]["raw_rows"] < accounted:
        raise ValueError("raw_rows is smaller than the training rows plus the qualified held-out pools")


def validate_plan(document):
    """Strict shape, hash, structure and internal-consistency validation; returns a private copy.

    Every mask is re-derived from the recipe, so a literal mask edited together with the plan
    digest is still rejected, and the declared counts, roles, content keys and protected families
    must agree with each other.  Nothing here reads a file or touches a model.

    This does NOT reestablish, from hashes alone, that each case is a member of the qualified
    population, that a limit's seeded ranking chose these cases, or that no case overlaps a
    training or protected population (the plan holds digests of those populations, not the
    populations).  Those guarantees come from the original :func:`build_plan` run and from pinning
    the frozen plan's ``plan_sha256``; a re-sealed plan that is consistent with itself is accepted.
    """
    require_keys(document, _PLAN_KEYS, "retention plan")
    canonical_bytes(document)
    if document["schema"] != PLAN_SCHEMA or document["scope"] != SCOPE or document["assumptions"] != list(ASSUMPTIONS):
        raise ValueError("unsupported retention plan schema, scope or assumptions")
    sha256_hex(document["plan_sha256"], "plan_sha256")
    if digest({k: v for k, v in document.items() if k != "plan_sha256"}) != document["plan_sha256"]:
        raise ValueError("retention plan digest mismatch")
    broad, functional, protected = document["broad"], document["functional"], document["protected"]
    require_keys(broad, ("input_sha256", "raw_input_digest", "raw_rows", "declaration_digest",
                         "population_digest"), "plan broad")
    for key in ("input_sha256", "raw_input_digest", "declaration_digest", "population_digest"):
        sha256_hex(broad[key], key)
    _count(broad["raw_rows"], "raw_rows")
    if functional is not None:
        require_keys(functional, ("constructs_sha256", "qualified_digest", "contract_digest", "broad_input_digest",
                                  "declaration_digest", "population_digest", "train_constructs"), "plan functional")
        for key in ("constructs_sha256", "qualified_digest", "contract_digest", "broad_input_digest",
                    "declaration_digest", "population_digest"):
            sha256_hex(functional[key], key)
        _count(functional["train_constructs"], "functional train_constructs")
        if (functional["broad_input_digest"] != broad["raw_input_digest"]
                or functional["declaration_digest"] != broad["declaration_digest"]
                or functional["population_digest"] != broad["population_digest"]):
            raise ValueError("the plan's functional qualification belongs to a different broad input")
    require_keys(protected, ("endpoint_manifest_sha256", "endpoint_manifest_digest", "roles", "entries",
                             "excluded_antibody_families", "excluded_target_families"), "plan protected")
    sha256_hex(protected["endpoint_manifest_sha256"], "endpoint_manifest_sha256")
    sha256_hex(protected["endpoint_manifest_digest"], "endpoint_manifest_digest")
    _count(protected["entries"], "protected entries")
    if protected["roles"] != sorted(set(protected["roles"])) or set(protected["roles"]) - set(ENDPOINT_ROLES):
        raise ValueError("invalid protected endpoint roles")
    if not protected["roles"] or len(protected["roles"]) > protected["entries"]:
        raise ValueError("protected roles disagree with the protected endpoint entry count")
    for key in ("excluded_antibody_families", "excluded_target_families"):
        if protected[key] != _families(protected[key], key):
            raise ValueError(f"{key} must be a canonical sorted list")
    guard = document["guard"]
    require_keys(guard, ("exact_antibody", "exact_antigen", "family_axes", "against", "assertion"), "plan guard")
    axes = _strings(guard["family_axes"], "family_axes", _AXES)
    if (guard["exact_antibody"] is not True or guard["exact_antigen"] is not True
            or guard["assertion"] != _GUARD_NOTE or guard["family_axes"] != axes):
        raise ValueError("invalid plan guard record")
    against = guard["against"]
    require_keys(against, ("denoising_train_raw_rows", "functional_train_constructs",
                           "protected_endpoint_entries"), "plan guard against")
    _count(against["denoising_train_raw_rows"], "denoising_train_raw_rows")
    if (against["protected_endpoint_entries"] != protected["entries"]
            or against["functional_train_constructs"] != (None if functional is None
                                                         else functional["train_constructs"])):
        raise ValueError("plan guard record disagrees with the protected and functional records")
    masks, selection = document["masks"], document["selection"]
    require_keys(masks, ("seed", "per_record", "recipe"), "plan masks")
    seed_value(masks["seed"])
    positive_count(masks["per_record"], "masks per_record")
    if masks["recipe"] != MASK_RECIPE:
        raise ValueError("unsupported mask recipe")
    cases = document["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("a retention plan needs at least one case")
    partitions = _validate_selection(selection)
    for case in cases:
        _validate_case(case, masks, partitions)
    ids = [case["record_id"] for case in cases]
    if ids != sorted(set(ids)):
        raise ValueError("cases must be unique and sorted by record_id")
    if selection["record_ids"] is not None and selection["record_ids"] != ids:
        raise ValueError("the cases are not the explicitly selected record_ids")
    if (selection["selected"] != len(cases) or document["counts"] != _counts(cases)
            or (selection["limit"] is not None and selection["limit"]["count"] != len(cases))):
        raise ValueError("plan counts disagree with its cases")
    _validate_population(document, partitions)
    return copy.deepcopy(document)


def load_plan(path):
    return validate_plan(read_json(path))


# ---------------------------------------------------------------------------
# Lineage: what the checkpoint says it trained on, against what the plan protects
# ---------------------------------------------------------------------------

def _bind_mid(plan, training, progress):
    """A midtraining identity against the plan: the input file, the actual data, the certificate, the functional arm."""
    certificate = training["transfer_certificate"]
    if certificate is None:
        raise ValueError("the midtraining identity has no transfer certificate; the protected split "
                         "cannot be verified")
    certificate = _certificate(certificate)
    broad, protected = plan["broad"], plan["protected"]
    population = {"manifest": broad["declaration_digest"], "population": broad["population_digest"]}
    if (training["data"] != population or certificate["declaration_digest"] != population["manifest"]
            or certificate["population_digest"] != population["population"]):
        raise ValueError("the midtraining population differs from the plan's broad population")
    # The plan promises the whole raw file, not only its qualified population: excluded and duplicate
    # rows live only in the file.  Every CLI midtraining identity records it (``load_broad_context``).
    recorded_input = training["input"]
    if not isinstance(recorded_input, dict) or "input_sha256" not in recorded_input:
        raise ValueError("the midtraining identity records no broad input file hash; the plan's raw "
                         "input file cannot be verified")
    sha256_hex(recorded_input["input_sha256"], "the midtraining identity's input_sha256")
    if recorded_input["input_sha256"] != broad["input_sha256"]:
        raise ValueError("the midtraining run read a different broad input file than the plan")
    if certificate["endpoint_manifest_digest"] != protected["endpoint_manifest_digest"]:
        raise ValueError("the midtraining certificate protects a different endpoint manifest than the plan")
    for key in ("excluded_antibody_families", "excluded_target_families"):
        if certificate[key] != protected[key]:
            raise ValueError(f"the midtraining certificate and the plan protect different {key}")
    _parent_functional(training, progress, certificate)       # identity and certificate agree on the arm
    binding, qualified = certificate["functional"], plan["functional"]
    if binding is not None and (qualified is None or any(
            binding[key] != qualified[key] for key in ("qualified_digest", "contract_digest", "broad_input_digest"))):
        raise ValueError("the midtraining functional qualification is not the plan's functional qualification")
    # A common plan may guard a functional population that this run never trained on (binding is None).
    return ["broad_input_file", "population", "endpoint_manifest", "protected_families",
            "functional_qualification" if binding is not None else "no_functional_arm_in_run"]


def _lineage(plan, provenance):
    require_keys(provenance, ("checkpoint_sha256", "training", "progress"), "checkpoint provenance")
    sha256_hex(provenance["checkpoint_sha256"], "checkpoint_sha256")
    training, progress = provenance["training"], provenance["progress"]
    if training["schema"] != TRAINING_SCHEMA:
        raise ValueError("unsupported training identity schema")
    stage, base = training["config"]["stage"], training["model"]["base"]
    initialization = training["initialization"]
    if stage == "mid":
        _release_origin(initialization, base)
        return {"stage": "mid", "origin": "release", "bound": _bind_mid(plan, training, progress)}
    if stage != "post":
        raise ValueError("unsupported training stage")
    budget = training["input"]["budget"]
    validate_post_budget_identity(budget)
    if budget["manifest_digest"] != plan["protected"]["endpoint_manifest_digest"]:
        raise ValueError("the post fitting budget belongs to a different endpoint manifest than the plan protects")
    bound = ["post_budget_manifest"]
    if initialization["kind"] == "release":
        _release_origin(initialization, base)
        if training["transfer_certificate"] is not None:
            raise ValueError("direct release adaptation has no broad transfer certificate")
        return {"stage": "post", "origin": "release", "bound": bound}
    # The parent checks below mirror ``transfer_guard.validate_stage_lineage`` (exact integer types, the
    # completed horizon, zero fitting and selection exposure) so that this evaluator is not weaker than
    # the trainer that wrote the identity.  No label or ``RestrictedPostData`` is reopened.
    require_keys(initialization, ("kind", "base", "sha256"), "parent initialization")
    if (initialization["kind"] != "parent" or initialization["base"] != base
            or base["variant"] != "materialized-parent" or initialization["sha256"] != base["parent_sha256"]):
        raise ValueError("parent initialization does not match the actual parent")
    sha256_hex(initialization["sha256"], "parent_sha256")
    parent = base["parent_provenance"]
    require_keys(parent, ("checkpoint_sha256", "training", "progress"), "parent provenance")
    sha256_hex(parent["checkpoint_sha256"], "parent checkpoint_sha256")
    parent_training, parent_progress = parent["training"], parent["progress"]
    if not isinstance(parent_training, dict) or not isinstance(parent_progress, dict):
        raise ValueError("parent training and progress must be objects")
    source, horizon = base["source_policy_identity"], parent_training["config"]["total_updates"]
    if (not isinstance(source, dict) or parent_training["schema"] != TRAINING_SCHEMA
            or parent_training["selection"] != "fixed-last" or parent_training["config"]["stage"] != "mid"
            or parent_training["model"] != source or type(horizon) is not int or horizon < 1
            or any(type(parent_progress[key]) is not int
                   for key in ("next_update", "fitting_endpoints_available", "selection_labels"))
            or parent_progress["next_update"] != horizon
            or parent_progress["fitting_endpoints_available"] != 0 or parent_progress["selection_labels"] != 0
            or parent_progress["inherited_exposure"] is not None):
        raise ValueError("a post checkpoint must descend from a completed, unmeasured midtraining parent")
    _release_origin(parent_training["initialization"], parent_training["model"]["base"])
    inherited = _bind_mid(plan, parent_training, parent_progress)
    # StageTrainer writes the parent's certificate, verbatim, on the post identity.  A post identity
    # that carries anything else was not produced by this lineage, whatever its parent says.
    current = training["transfer_certificate"]
    if current is None:
        raise ValueError("the post checkpoint records no transfer certificate although its parent was certified")
    if _certificate(current) != _certificate(parent_training["transfer_certificate"]):
        raise ValueError("the post checkpoint's transfer certificate is not the certificate its midtraining "
                         "parent was trained under")
    return {"stage": "post", "origin": "parent",
            "bound": bound + ["inherited_transfer_certificate"] + ["parent_" + name for name in inherited]}


def check_lineage(plan, provenance):
    """Bind a loaded checkpoint's recorded ancestry to the plan; returns what was compared.

    ``provenance`` is what ``load_inference_checkpoint`` returned.  A midtraining checkpoint must
    record the plan's raw broad input file hash (an absent or malformed record is refused), its
    population, a valid transfer certificate on the plan's manifest and families, and the plan's
    functional qualification when it trained on one.  A post checkpoint binds the CURRENT fitting
    budget's manifest and, when it descends from a midtraining parent, that parent's records as
    above, a parent that satisfies the same completed-and-unmeasured conditions as the trainer's
    lineage guard, and a CURRENT transfer certificate equal to the parent's.  A direct-release
    post checkpoint has no broad ancestry to bind and none is invented.
    """
    try:
        return _lineage(plan, provenance)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"malformed checkpoint provenance: {error!r}") from error


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@contextmanager
def frozen_evaluation(policy, encoder):
    """Policy eval and no_grad, then EXACT restoration, on success and on failure alike.

    ``FrozenESMPPI.train`` and its forward force every encoder module to eval, and
    ``artifacts.evaluating`` restores only the root, so each module's flag is snapshotted and put
    back by direct assignment (shared modules once).  The Python generator, the Torch CPU
    generator and the generator of every CUDA device holding a parameter are restored too.
    Parameters, gradients and optimizer state are never touched.
    """
    seen, flags = set(), []
    for root in (policy, encoder):
        for module in root.modules():
            if id(module) not in seen:
                seen.add(id(module))
                flags.append((module, module.training))
    python_state, cpu_state = random.getstate(), torch.get_rng_state()
    devices = sorted({p.device for root in (policy, encoder) for p in root.parameters()
                      if p.device.type == "cuda"}, key=str)
    cuda_states = [(device, torch.cuda.get_rng_state(device)) for device in devices]
    try:
        policy.eval()
        with torch.no_grad():
            yield
    finally:
        for module, training in flags:
            module.training = training
        random.setstate(python_state)
        torch.set_rng_state(cpu_state)
        for device, state in cuda_states:
            torch.cuda.set_rng_state(state, device)


def _jobs(policy, encoder, plan, arm):
    """Every view of every mask, preflighted; a structural error aborts before the first forward."""
    jobs = []
    for case in plan["cases"]:
        sequences = {"H": case["H"]}
        if case["L"] is not None:
            sequences["L"] = case["L"]
        if arm == "paired":
            sequences["Ag"] = case["Ag"]
        for mask in case["masks"]:
            combined, target = mask_complete(sequences, [tuple(site) for site in mask])
            provider = DynamicContextProvider(encoder, combined)
            encoder.validate_view(combined)
            query = combined.without_ag()
            policy.preflight([query], [provider])
            jobs.append((query, provider, target))
    return jobs


def _forward(policy, encoder, jobs):
    """One outcome per job.  Numerical and memory failures are outcomes; everything else propagates.

    A frozen encoder that returns NaN or infinite representations is rejected by ``ExactViewFeatures``
    with a ``ValueError`` (not a ``FloatingPointError``).  It is a numerical failure of one mask, so
    exactly that message is recorded; any other ``ValueError`` (shape, identity, view) is structural.
    """
    outcomes = []
    with frozen_evaluation(policy, encoder):
        for query, provider, target in jobs:
            try:
                value = float(policy.reconstruction_nll([query], [provider], [target]))
                if not math.isfinite(value):
                    raise FloatingPointError("nonfinite reconstruction NLL")
                outcomes.append((value, None))
            except (torch.cuda.OutOfMemoryError, FloatingPointError, ValueError) as error:
                if isinstance(error, ValueError) and str(error) != _NONFINITE_FEATURES:
                    raise
                outcomes.append((None, f"{type(error).__name__}: {error}"))
    return outcomes


def _mean(values):
    return math.fsum(values) / len(values)


def summarize(plan, outcomes):
    """Per-mask results, per-record means and the aggregate; incomplete means NO aggregate at all."""
    cases, position = [], 0
    for case in plan["cases"]:
        masks = []
        for index, mask in enumerate(case["masks"]):
            value, failure = outcomes[position]
            position += 1
            masks.append({"index": index, "sites": len(mask), "nll": value, "failure": failure})
        complete = all(m["nll"] is not None for m in masks)
        cases.append({"record_id": case["record_id"], "masks": masks,
                      "mean_nll": _mean([m["nll"] for m in masks]) if complete else None})
    failed = sum(m["nll"] is None for c in cases for m in c["masks"])
    aggregate = None if failed else {"mean_record_nll": _mean([c["mean_nll"] for c in cases])}
    return cases, aggregate, failed


def evaluate_retention(policy, encoder, plan, *, arm, provenance):
    """Score every planned mask of every planned record at one checkpoint.

    ``provenance`` is the ``load_inference_checkpoint`` result.  The plan, the lineage, the encoder
    and model identities (checked here as well, so a direct call is as strict as the command line)
    and every view are validated before the first forward.  A numerical failure of one mask (a
    nonfinite NLL, nonfinite frozen-encoder representations, or CUDA out-of-memory) is recorded
    and leaves the result ``incomplete`` with no aggregate; any other error aborts.  The same plan
    evaluated for the paired and ab-only arms differs only in the antigen.
    """
    if arm not in ("paired", "abonly"):
        raise ValueError("arm must be paired or abonly")
    plan = validate_plan(plan)
    canonical_bytes(provenance)
    lineage = check_lineage(plan, provenance)
    training = provenance["training"]
    if encoder.encoder_identity != policy.encoder_identity:
        raise ValueError("encoder and policy identity differ")
    if training["encoder_runtime"] != encoder.runtime_identity():
        raise ValueError("checkpoint encoder runtime differs from the evaluating encoder")
    if training["model"] != policy.identity():
        raise ValueError("checkpoint model identity differs from the evaluating policy")
    jobs = _jobs(policy, encoder, plan, arm)
    cases, aggregate, failed = summarize(plan, _forward(policy, encoder, jobs))
    result = {
        "schema": RESULT_SCHEMA, "scope": SCOPE, "assumptions": list(ASSUMPTIONS),
        "readout": copy.deepcopy(READOUT),
        "status": "incomplete" if failed else "complete", "arm": arm,
        "plan": {"plan_sha256": plan["plan_sha256"], "declaration_digest": plan["broad"]["declaration_digest"],
                 "population_digest": plan["broad"]["population_digest"]},
        "lineage": lineage,
        "checkpoint": {"checkpoint_sha256": provenance["checkpoint_sha256"], "training_sha256": digest(training),
                       "next_update": provenance["progress"]["next_update"]},
        "model_sha256": digest(policy.identity()), "encoder_runtime_sha256": digest(training["encoder_runtime"]),
        "source_sha256": digest(training["source"]),
        "counts": {**plan["counts"], "failed_masks": failed},
        "aggregate": aggregate, "cases": cases,
    }
    result["result_sha256"] = digest(result)
    return result
