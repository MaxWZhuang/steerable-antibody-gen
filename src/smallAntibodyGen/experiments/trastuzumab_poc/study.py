"""Preregistered downstream study (M4): an outcome-free plan and a selection over supplied score exports.

The question is a fair comparison of low-resource post-adaptation after different broad stages: direct
release, broad denoising only, and the detached and shared functional arms.  Every arm shares one
post recipe (only the learning rate and the post seed vary), one nested family of fitting budgets, one
fixed horizon with declared boundary updates, one development panel and one readout.  Nothing here
trains, schedules, launches or removes a checkpoint, and no model or weight is opened.  The planner opens
no label or measurement at all; the selector opens exactly one measurement file, the declared development
labels, and only after every export has validated.  Each input file is read once into a byte snapshot that
is both hashed and parsed, so a recorded file hash always names the document that was used.

``plan-study`` reads the specification and the outcome-free endpoint manifest only.  It derives the
query panel from ALL development entries, the literal reveal orders of the readout, the nested budgets
(through ``select_budget`` and ``post_budget_identity``) and the full inventory of trials, and seals them
under ``plan_sha256``.  The two opaque label pins of every budget (``selected_data_sha256`` and
``selected_labels_sha256``) are SUPPLIED by the caller from authorized restricted label preparation:
the planner never opens a label file and never claims to derive them.  Preparation obtains them from the
existing loader, outside planning, with no model and no training::

    restricted, identity = load_post(manifest_path, labels_path, budget_n=n, budget_seed=seed)
    pins = {"budget_n": n, "selected_data_sha256": identity["selected_data"],
            "selected_labels_sha256": identity["selected_labels_sha256"]}

The plan embeds the exact ``trastuzumab-queries/1`` document (``plan["queries"]``) to pass as
``--queries``, and the ``query`` section of the specification gives the ``score`` flags.  Scoring itself
is the existing ``score`` command at every declared boundary of every declared trial.

``select-study`` consumes a results index, one record per (trial, boundary): either the path and manifest
hash of a score export, or an explicit failure with its reason.  It rederives the plan, then validates
the whole inventory and EVERY export (byte hashes, payload and metadata links, the full panel, the
readout, exact case IDs, the checkpoint's training and progress identity, source, fusion, encoder,
runtime, origin and lineage, the available fitting budget and both label pins) BEFORE the development
label document is opened.  Any missing, extra, duplicated or inconsistent record blocks selection.  Only
then are the development labels read; a learning rate is chosen per (arm, budget) at the FINAL horizon by
the mean ordinal concordance over ALL planned (origin, post seed) trials, never from a single checkpoint
and never from an intermediate boundary.

Trust model.  Digests establish that the supplied records agree with each other and with the plan.  They
do not establish unreported history, the chronology of the preregistration, or that the label pins came
from the authorized preparation: an author who recomputes every digest can write a self-consistent set
of records.  Pin the ``plan_sha256`` that ``plan-study`` wrote.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from bisect import bisect_left, bisect_right, insort
from collections import Counter
from dataclasses import asdict, fields
from fractions import Fraction
from pathlib import Path

from smallAntibodyGen.experiments.trastuzumab_poc.artifacts import (
    Query, QueryPanel, canonical_bytes, digest, positive_count, read_json, require_keys, seed_value, text_value,
)
from smallAntibodyGen.experiments.trastuzumab_poc.functional_data import sha256_hex
from smallAntibodyGen.experiments.trastuzumab_poc.inputs import record, sequence
from smallAntibodyGen.experiments.trastuzumab_poc.post_budget import post_budget_identity
from smallAntibodyGen.experiments.trastuzumab_poc.protocol import (
    ENDPOINT_ROLES, DevelopmentLabel, EndpointConstruct, EndpointEntry, EndpointManifest, select_budget,
)
from smallAntibodyGen.experiments.trastuzumab_poc.randomness import RNG_ALGORITHM
from smallAntibodyGen.experiments.trastuzumab_poc.runner import TrainingConfig
# The lineage validators are private to the transfer guard because they only mean something inside this
# campaign's release -> mid -> post chain; the study reuses them, like retention, rather than restating them.
from smallAntibodyGen.experiments.trastuzumab_poc.transfer_guard import (
    TRAINING_SCHEMA, _certificate, _parent_functional, _release_origin,
)
from smallAntibodyGen.models.antibody_antigen.fusion import FusionConfig

SPEC_SCHEMA = "trastuzumab-study-spec/1"
PLAN_SCHEMA = "trastuzumab-study-plan/1"
RESULTS_SCHEMA = "trastuzumab-study-results/1"
LABELS_SCHEMA = "trastuzumab-study-development-labels/1"
SELECTION_SCHEMA = "trastuzumab-study-selection/1"
# Schemas written by the existing export path, which the study must read back unchanged.
EXPORT_SCHEMA = "trastuzumab-export/1"
INFERENCE_SCHEMA = "trastuzumab-inference/1"
SCORE_SCHEMA = "her2-benchmark/1"
SOURCE_SCHEMA = "trastuzumab-source/1"
READOUT_NAME = "whole_core_mean_sequential_log_probability"

SCOPE = ("preregistered comparison of low-resource post-adaptation over supplied score exports; software "
         "machinery, not evidence of biological success")
ASSUMPTIONS = (
    "the label pins are supplied by the caller from authorized label preparation; the planner never opens a "
    "label file and cannot derive them",
    "source, config, label and manifest digests establish that the supplied records agree with each other, "
    "not unreported history or the chronology of the preregistration",
    "the execution contract, the encoder, the origins and the development panel are declared by the caller; "
    "the selector verifies that every export records exactly them",
    "the development labels are a small fixed set: the selected learning rate is optimistic for those labels "
    "and says nothing about other endpoints",
)
METRIC = {
    "name": "development_ordinal_concordance",
    "definition": "over unordered pairs of development endpoints with different measured values, the fraction "
                  "whose score order agrees with the measured order; a pair with equal scores counts one half",
    "orientation": "measured values are oriented so that larger is better (lower_better values are negated); "
                   "scores are higher_is_better log probabilities",
    "ties": "pairs whose oriented measured values are exactly equal are excluded; no tolerance is applied",
    "undefined": "no comparable pair means the metric is undefined and selection is refused",
    "range": "[0, 1]; 0.5 is the expectation of an uninformative score",
    "units": "dimensionless ordinal agreement; not an affinity, a correlation or a calibrated probability",
}
SELECTION_RULE = {
    "horizon": "the learning rate is chosen at the FINAL boundary only; an intermediate boundary never rescues "
               "a failed final candidate",
    "aggregation": "mean metric over ALL planned (origin, post seed) trials of an (arm, budget, learning rate), "
                   "taken exactly over the integer pair counts of each trial (floating point is for reporting "
                   "only); the denominator is never reduced",
    "eligibility": "a learning rate is eligible only if every planned trial has a valid final score export "
                   "(all queries finite); one failed query invalidates the whole candidate",
    "choice": "highest mean metric among eligible learning rates",
    "tie_rule": "an exact tie of the mean metric, as an exact rational number, chooses the smallest learning "
                "rate; no tolerance is applied and no seed is ever chosen",
}
RETROSPECTIVE = ("RETROSPECTIVE: this learning rate was chosen by the final-horizon development metric, so its "
                 "trajectory is optimistic as evidence of adaptation speed")
PROSPECTIVE = ("PROSPECTIVE: this learning rate was fixed in the plan before any development label; it is the "
               "curve to use for adaptation-speed comparisons")

_SPEC_KEYS = ("schema", "study_id", "development", "query", "recipe", "budgets", "identity", "arms")
_TEMPLATE_KEYS = tuple(f.name for f in fields(TrainingConfig) if f.name not in ("learning_rate", "seed"))
_FUSION_KEYS = tuple(f.name for f in fields(FusionConfig))
_MID_ORIGIN_KEYS = ("origin_id", "parent_sha256", "parent_checkpoint_sha256", "parent_provenance_sha256")
_PLAN_KEYS = ("schema", "scope", "assumptions", "spec", "manifest", "queries", "queries_sha256", "scaffold_id",
              "antigen_sha256", "development", "readout", "budgets", "boundaries", "trials", "metric",
              "selection_rule", "counts", "plan_sha256")
_EXPORTED_KEYS = ("trial_id", "boundary", "status", "export_path", "manifest_sha256")
_FAILED_KEYS = ("trial_id", "boundary", "status", "reason")
_EXPORT_FILES = ("manifest.json", "metadata.json", "scores.json")
_METADATA_KEYS = ("schema", "checkpoint", "model", "encoder_runtime", "queries_sha256", "scaffold_id",
                  "assay_id", "antigen_sha256", "context_arm", "query_roles_or_outcomes_available", "readout")
_PAYLOAD_KEYS = ("schema", "record_kind", "scorer_id", "readout", "orientation", "provenance", "scores")
# The exact key contracts and fixed recipe of ``checkpoint._validate``.  They are restated because that module
# loads tensors; an export carries this metadata only, so only the metadata is checked (no tensor, no optimizer
# state, no replay of the draws).
_TRAINING_KEYS = ("schema", "config", "model", "encoder_runtime", "data", "input", "source", "initialization",
                  "rng", "runtime", "optimizer", "selection", "parameter_names", "transfer_certificate",
                  "ordering_readout", "functional")
_PROGRESS_KEYS = ("next_update", "counters", "seen", "unique", "fitting_endpoints_available", "selection_labels",
                  "inherited_exposure", "functional")
_COUNTER_KEYS = ("examples", "masked_residues", "teacher_forced_states", "pair_draws", "informative_pairs", "ties",
                 "antigen_examples")
_ENDPOINT_KEYS = ("endpoints", "reconstruction_endpoints", "ordering_endpoints", "informative_ordering_endpoints")
_SEEN_KEYS = ("records", "antibody_families", "target_families", "sources", *_ENDPOINT_KEYS)
_OPTIMIZER = {"kind": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8, "foreach": False, "fused": False,
              "capturable": False}
_ORDERING_READOUT = "dropout-disabled-with-checkpointing/1"


def _plain(value):
    """Round-trip through canonical JSON so a built document equals the same document read back."""
    return json.loads(canonical_bytes(value))


# ---------------------------------------------------------------------------
# Specification
# ---------------------------------------------------------------------------

def _strictly_increasing(values, name, check):
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a nonempty list")
    checked = [check(v) for v in values]
    if any(a >= b for a, b in zip(checked, checked[1:])):
        raise ValueError(f"{name} must be strictly increasing (ordered and unique)")
    return checked


def _positive_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def _development(value):
    require_keys(value, ("assay_id", "orientation"), "study development")
    text_value(value["assay_id"], "development assay_id")
    if value["orientation"] not in ("higher_better", "lower_better"):
        raise ValueError("development orientation must be higher_better or lower_better")
    return {"assay_id": value["assay_id"], "orientation": value["orientation"]}


def _query(value):
    require_keys(value, ("protocol_id", "scorer_id", "context_arm", "order_seed", "orders"), "study query")
    text_value(value["protocol_id"], "protocol_id")
    text_value(value["scorer_id"], "scorer_id")
    if value["context_arm"] not in ("paired", "abonly"):
        raise ValueError("context_arm must be paired or abonly")
    seed_value(value["order_seed"])
    positive_count(value["orders"], "orders")
    return dict(value)


def _recipe(value):
    require_keys(value, ("training", "fusion", "boundaries", "learning_rates", "prospective_learning_rate",
                         "post_seeds"), "study recipe")
    require_keys(value["training"], _TEMPLATE_KEYS, "post training template")
    template = copy.deepcopy(value["training"])
    if template["stage"] != "post" or template["arm"] != "paired":
        raise ValueError("the study recipe is the paired post-training stage")
    require_keys(value["fusion"], _FUSION_KEYS, "study fusion")
    fusion = asdict(FusionConfig(**value["fusion"]))
    if fusion["update_mode"] != "lora":
        raise ValueError("the fixed post recipe adapts a fresh LoRA")
    rates = _strictly_increasing(value["learning_rates"], "learning_rates",
                                 lambda v: _positive_number(v, "learning rate"))
    seeds = _strictly_increasing(value["post_seeds"], "post_seeds", seed_value)
    prospective = _positive_number(value["prospective_learning_rate"], "prospective_learning_rate")
    if prospective not in rates:
        raise ValueError("prospective_learning_rate must be a member of learning_rates")
    for rate in rates:
        for seed in seeds:
            TrainingConfig(**template, learning_rate=rate, seed=seed)       # the template is a valid recipe
    boundaries = _strictly_increasing(value["boundaries"], "boundaries", lambda v: positive_count(v, "boundary"))
    if boundaries[-1] != template["total_updates"]:
        raise ValueError("the final boundary must be the declared total_updates horizon")
    return {"training": template, "fusion": fusion, "boundaries": boundaries, "learning_rates": rates,
            "prospective_learning_rate": prospective, "post_seeds": seeds}


def _budgets(value):
    require_keys(value, ("budget_seed", "entries"), "study budgets")
    text_value(value["budget_seed"], "budget_seed")
    entries = value["entries"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("budgets.entries must be a nonempty list")
    for entry in entries:
        require_keys(entry, ("budget_n", "selected_data_sha256", "selected_labels_sha256"), "budget entry")
        sha256_hex(entry["selected_data_sha256"], "selected_data_sha256")
        sha256_hex(entry["selected_labels_sha256"], "selected_labels_sha256")
    _strictly_increasing([e["budget_n"] for e in entries], "budget sizes", lambda v: positive_count(v, "budget_n"))
    return {"budget_seed": value["budget_seed"], "entries": [dict(entry) for entry in entries]}


def _identity(value, fusion):
    require_keys(value, ("release_base_sha256", "source_sha256", "encoder", "execution"), "study identity")
    sha256_hex(value["release_base_sha256"], "release_base_sha256")
    sha256_hex(value["source_sha256"], "source_sha256")
    encoder, execution = value["encoder"], value["execution"]
    require_keys(encoder, ("identity", "runtime"), "study encoder")
    require_keys(execution, ("hardware", "software", "parameters"), "study execution")
    for name, part in (("encoder identity", encoder["identity"]), ("encoder runtime", encoder["runtime"]),
                       *((f"execution {key}", execution[key]) for key in execution)):
        if not isinstance(part, dict) or not part:
            raise ValueError(f"{name} must be a nonempty object")
    if encoder["runtime"].get("encoder") != encoder["identity"]:
        raise ValueError("the encoder runtime does not embed the declared encoder identity")
    if encoder["runtime"].get("compute_dtype") != fusion["compute_dtype"]:
        raise ValueError("the encoder compute dtype differs from the fusion compute dtype")
    canonical_bytes(value)
    return copy.deepcopy(value)


def _arms(value):
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError("a comparison needs at least two logical arms")
    arms, names, hashes = [], set(), []
    for arm in value:
        require_keys(arm, ("arm_id", "kind", "functional_mode", "origins"), "study arm")
        text_value(arm["arm_id"], "arm_id")
        if arm["arm_id"] in names:
            raise ValueError(f"duplicate arm_id {arm['arm_id']!r}")
        names.add(arm["arm_id"])
        kind, mode = arm["kind"], arm["functional_mode"]
        if kind not in ("release", "mid") or mode not in (None, "detached", "shared") or (
                kind == "release" and mode is not None):
            raise ValueError(f"arm {arm['arm_id']!r}: kind must be release or mid, and only a mid arm "
                             "declares a functional_mode (detached or shared)")
        origins = arm["origins"]
        if not isinstance(origins, list) or not origins:
            raise ValueError(f"arm {arm['arm_id']!r} declares no origin")
        if kind == "release" and len(origins) != 1:
            raise ValueError("a direct-release arm has exactly one origin; its post seeds are the replicates")
        keys = ("origin_id",) if kind == "release" else _MID_ORIGIN_KEYS
        ids = set()
        for origin in origins:
            require_keys(origin, keys, f"arm {arm['arm_id']!r} origin")
            text_value(origin["origin_id"], "origin_id")
            if origin["origin_id"] in ids:
                raise ValueError(f"arm {arm['arm_id']!r}: duplicate origin_id {origin['origin_id']!r}")
            ids.add(origin["origin_id"])
            hashes.extend(sha256_hex(origin[name], name) for name in keys[1:])
        arms.append(copy.deepcopy(arm))
    if sum(arm["kind"] == "release" for arm in arms) > 1:
        raise ValueError("at most one direct-release arm")
    if len(set(hashes)) != len(hashes):
        raise ValueError("parent SHA256 and provenance hashes must be distinct across all origins and arms")
    return arms


def normalize_spec(spec):
    """Strict canonical form of a path-free specification; idempotent, and never aliases its input."""
    require_keys(spec, _SPEC_KEYS, "study spec")
    if spec["schema"] != SPEC_SCHEMA:
        raise ValueError("unsupported study spec schema")
    text_value(spec["study_id"], "study_id")
    recipe = _recipe(spec["recipe"])
    return {"schema": SPEC_SCHEMA, "study_id": spec["study_id"], "development": _development(spec["development"]),
            "query": _query(spec["query"]), "recipe": recipe, "budgets": _budgets(spec["budgets"]),
            "identity": _identity(spec["identity"], recipe["fusion"]), "arms": _arms(spec["arms"])}


def parse_spec(document):
    """``(spec, endpoint_manifest_path)``: strict, with the manifest path preserved verbatim."""
    if not isinstance(document, dict) or "endpoint_manifest_path" not in document:
        raise ValueError("study spec: expected an object with endpoint_manifest_path")
    rest = {key: value for key, value in document.items() if key != "endpoint_manifest_path"}
    return normalize_spec(rest), text_value(document["endpoint_manifest_path"], "endpoint_manifest_path")


def load_spec(path):
    """Parse a specification file; the manifest path is relative to the file, like a run config."""
    path = Path(path).resolve()
    spec, manifest_path = parse_spec(read_json(path))
    return spec, str((path.parent / manifest_path).resolve())


def _strict_json(data):
    """Parse exact bytes under ``read_json``'s rules: a BOM and pretty-printing are fine, a duplicate key or a
    non-finite number is not."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def nonfinite(constant):
        raise ValueError(f"nonfinite JSON: {constant}")

    value = json.loads(data.decode("utf-8-sig"), object_pairs_hook=pairs, parse_constant=nonfinite)
    canonical_bytes(value)
    return value


def read_snapshot(path):
    """``(document, sha256)`` of ONE read of ``path``: the hash names exactly the bytes that were parsed.

    Parsing a file and then hashing it by a second read lets a replacement in between pair one document
    with another file's hash.  Here the bytes are captured once, then hashed and parsed from that capture.
    """
    data = Path(path).read_bytes()
    return _strict_json(data), hashlib.sha256(data).hexdigest()


def _endpoint_manifest(document):
    """``inputs.load_endpoint_manifest`` over an already parsed document (same contract, no second read)."""
    require_keys(document, ("schema", "entries"), "endpoint manifest")
    if document["schema"] != "trastuzumab-endpoints/1":
        raise ValueError("unsupported endpoint schema")
    return EndpointManifest(tuple(record(EndpointEntry, row, {
        "construct": lambda x: record(EndpointConstruct, x),
    }) for row in sequence(document["entries"])))


def _manifest_snapshot(path):
    """``(manifest, sha256)`` from one read of the endpoint manifest file."""
    document, sha256 = read_snapshot(path)
    return _endpoint_manifest(document), sha256


# ---------------------------------------------------------------------------
# Planning (outcome-free: the specification and the endpoint manifest only)
# ---------------------------------------------------------------------------

def _panel(spec, manifest):
    """The query panel of ALL development entries, case_id = endpoint_id, ids by the current formulas."""
    development = sorted(manifest.development_entries(), key=lambda e: e.endpoint_id)
    if len(development) < 2:
        raise ValueError("a development ordering needs at least two development endpoints")
    assay = spec["development"]["assay_id"]
    if {e.assay_id for e in development} != {assay}:
        raise ValueError(f"every development endpoint must belong to the one declared assay {assay!r}")
    first = development[0].construct
    # EndpointManifest guarantees ONE scaffold and antigen across every entry.  construct_id covers the
    # scaffold segments and the core but NOT the antigen, so the antigen is bound by the panel digest.
    scaffold_id = digest({"kind": "scaffold", "segments": [first.heavy_prefix, first.heavy_suffix, first.light]})
    queries = tuple(Query(e.endpoint_id, e.construct.core,
                          digest({"kind": "construct", "scaffold": scaffold_id, "core": e.construct.core}))
                    for e in development)
    # QueryPanel re-derives every construct_id and rejects an empty heavy prefix or suffix.
    return QueryPanel(spec["query"]["protocol_id"], assay, first.heavy_prefix, first.heavy_suffix, first.light,
                      first.Ag, queries)


def _readout(panel, query):
    """The literal readout ``score_queries`` records: reveal orders from ``random.Random(order_seed)``."""
    view, _ = panel.masked("A" * 10, query["context_arm"])
    coords = tuple(view.hidden_sites())
    rng = random.Random(query["order_seed"])
    schedules = [tuple(rng.sample(coords, len(coords))) for _ in range(query["orders"])]
    return {"name": READOUT_NAME, "canonical_residues": 20, "order_seed": query["order_seed"],
            "orders": [[list(c) for c in order] for order in schedules],
            "averaging": "arithmetic mean over complete orders; nats per site",
            "is_sampler_likelihood": False, "is_affinity": False}


def _plan_budgets(spec, manifest):
    """Canonical, nested budgets from the FULL manifest, each with the caller's opaque label pins."""
    seed, budgets, previous = spec["budgets"]["budget_seed"], [], []
    for entry in spec["budgets"]["entries"]:
        identity = post_budget_identity(select_budget(manifest, entry["budget_n"], seed))
        if identity["selected_ids"][:len(previous)] != previous:
            raise ValueError("fitting budgets must be nested prefixes of one seeded order")
        previous = identity["selected_ids"]
        budgets.append({"budget_n": entry["budget_n"], "identity": identity,
                        "selected_data_sha256": entry["selected_data_sha256"],
                        "selected_labels_sha256": entry["selected_labels_sha256"]})
    return budgets


def _trials(spec):
    """Every trial of every real origin, in one deterministic order; a trial id hashes its coordinates."""
    recipe, trials = spec["recipe"], []
    for arm in spec["arms"]:
        for origin in arm["origins"]:
            for budget in spec["budgets"]["entries"]:
                for rate in recipe["learning_rates"]:
                    for seed in recipe["post_seeds"]:
                        trial = {"arm_id": arm["arm_id"], "origin_id": origin["origin_id"],
                                 "budget_n": budget["budget_n"], "learning_rate": rate, "post_seed": seed}
                        trials.append({"trial_id": digest(trial), **trial})
    return trials


def build_plan(spec, manifest, manifest_sha256):
    """The sealed, outcome-free plan.  No label, model, weight or measurement is touched."""
    if not isinstance(manifest, EndpointManifest):
        raise TypeError("manifest must be an EndpointManifest")
    sha256_hex(manifest_sha256, "manifest_sha256")
    spec = normalize_spec(spec)
    panel = _panel(spec, manifest)
    budgets = _plan_budgets(spec, manifest)
    trials = _trials(spec)
    development_ids = [q.case_id for q in panel.queries]
    fitting_ids = budgets[-1]["identity"]["selected_ids"]
    union = set(fitting_ids) | set(development_ids)
    if len(union) != len(fitting_ids) + len(development_ids):
        raise ValueError("the largest fitting budget overlaps the development endpoints")
    boundaries = spec["recipe"]["boundaries"]
    plan = _plain({
        "schema": PLAN_SCHEMA, "scope": SCOPE, "assumptions": list(ASSUMPTIONS), "spec": spec,
        "manifest": {"file_sha256": manifest_sha256, "digest": manifest.manifest_digest(),
                     "entries": len(manifest.entries),
                     "roles": {role: sum(e.role == role for e in manifest.entries) for role in ENDPOINT_ROLES}},
        "queries": panel.document(), "queries_sha256": digest(panel.document()),
        "scaffold_id": panel.scaffold_id, "antigen_sha256": panel.antigen_sha256,
        "development": {**spec["development"], "endpoint_ids": development_ids,
                        "endpoint_semantics": manifest.by_id(development_ids[0]).endpoint_semantics},
        "readout": _readout(panel, spec["query"]), "budgets": budgets, "boundaries": boundaries,
        "trials": trials, "metric": METRIC, "selection_rule": SELECTION_RULE,
        "counts": {"arms": len(spec["arms"]), "trials": len(trials), "exports": len(trials) * len(boundaries),
                   "development_labels": len(development_ids), "largest_fitting_budget": len(fitting_ids),
                   "unique_label_union": len(union)},
    })
    plan["plan_sha256"] = digest(plan)
    return plan


def plan_study(spec_path):
    """File-level planning: only the specification and the endpoint manifest are read."""
    spec, manifest_path = load_spec(spec_path)
    manifest, manifest_sha256 = _manifest_snapshot(manifest_path)
    return build_plan(spec, manifest, manifest_sha256)


def validate_plan(plan, manifest, manifest_sha256):
    """A plan is accepted only if it IS the plan its embedded specification and the manifest derive.

    Rederiving rejects an omitted, extra or duplicated trial or budget, an edited panel, readout or
    count, and a changed manifest, whatever the plan digest says.  It cannot show that the caller's
    label pins are the real ones (they are opaque), or that the plan predates any result.
    """
    require_keys(plan, _PLAN_KEYS, "study plan")
    canonical_bytes(plan)
    if plan["schema"] != PLAN_SCHEMA:
        raise ValueError("unsupported study plan schema")
    sha256_hex(plan["plan_sha256"], "plan_sha256")
    if digest({k: v for k, v in plan.items() if k != "plan_sha256"}) != plan["plan_sha256"]:
        raise ValueError("study plan digest mismatch")
    if build_plan(plan["spec"], manifest, manifest_sha256) != plan:
        raise ValueError("the study plan is not the plan its specification and manifest derive")
    return copy.deepcopy(plan)


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def oriented(value, orientation):
    """Measured values oriented so that larger is better."""
    return value if orientation == "higher_better" else -value


def ordinal_concordance(scores, values):
    """Exact ordinal concordance of ``scores`` with ``values`` (larger is better in both).

    A comparable pair has exactly different values.  It is concordant if the scores order it the same
    way, counts one half if the scores are equal, and is discordant otherwise.  ``value`` is
    ``(concordant + tied / 2) / comparable`` or ``None`` when no pair is comparable.  Sorting by value and
    keeping the scores of strictly worse groups in a sorted list gives every count in O(n log n)
    comparisons.
    """
    if len(scores) != len(values):
        raise ValueError("scores and values must be parallel")
    scores = [float(_finite(s, "score")) for s in scores]
    values = [_finite(v, "measured value") for v in values]
    order = sorted(range(len(values)), key=lambda i: values[i])
    worse, concordant, tied, comparable, start = [], 0, 0, 0, 0
    while start < len(order):
        stop = start
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        group = [scores[i] for i in order[start:stop]]
        for score in group:
            below = bisect_left(worse, score)
            concordant += below
            tied += bisect_right(worse, score) - below
        comparable += len(group) * len(worse)
        for score in group:
            insort(worse, score)
        start = stop
    return {"comparable_pairs": comparable, "concordant": concordant, "tied_predictions": tied,
            "discordant": comparable - concordant - tied,
            "value": None if not comparable else (2 * concordant + tied) / (2 * comparable)}


# ---------------------------------------------------------------------------
# Export validation (before any development label is opened)
# ---------------------------------------------------------------------------

def _read_document(path, *, sha256=None, size=None, name):
    """Parse a file whose exact bytes were hashed: the parse must be of the canonical bytes that were hashed."""
    data = Path(path).read_bytes()
    if (sha256 is not None and hashlib.sha256(data).hexdigest() != sha256) or (
            size is not None and len(data) != size):
        raise ValueError(f"{name} does not match the recorded SHA256 and size")
    document = _strict_json(data)
    if canonical_bytes(document) + b"\n" != data:
        raise ValueError(f"{name} is not the canonical encoding of the bytes that were hashed")
    return document


def _binding(value, name):
    require_keys(value, ("sha256", "bytes"), name)
    sha256_hex(value["sha256"], name)
    positive_count(value["bytes"], f"{name} bytes")
    return value


def _load_export(directory, declared_sha256):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"export directory {str(directory)!r} does not exist")
    if sorted(p.name for p in directory.iterdir()) != list(_EXPORT_FILES):
        raise ValueError(f"an export holds exactly {list(_EXPORT_FILES)}")
    manifest = _read_document(directory / "manifest.json", sha256=declared_sha256, name="export manifest")
    require_keys(manifest, ("schema", "files", "checkpoint", "queries_sha256"), "export manifest")
    if manifest["schema"] != EXPORT_SCHEMA:
        raise ValueError("unsupported export schema")
    files = manifest["files"]
    require_keys(files, ("scores.json", "metadata.json"), "export files")
    documents = {}
    for filename in ("scores.json", "metadata.json"):
        binding = _binding(files[filename], f"export {filename}")
        documents[filename] = _read_document(directory / filename, sha256=binding["sha256"],
                                             size=binding["bytes"], name=f"export {filename}")
    return manifest, documents["scores.json"], documents["metadata.json"]


def _scores(plan, metadata, payload):
    """Panel, readout and case inventory; returns ``(scores or None, failures)``."""
    require_keys(metadata, _METADATA_KEYS, "export metadata")
    query = plan["spec"]["query"]
    expected = {"schema": INFERENCE_SCHEMA, "queries_sha256": plan["queries_sha256"],
                "scaffold_id": plan["scaffold_id"], "assay_id": plan["development"]["assay_id"],
                "antigen_sha256": plan["antigen_sha256"], "context_arm": query["context_arm"],
                "readout": plan["readout"]}
    for key, value in expected.items():
        if metadata[key] != value:
            raise ValueError(f"export metadata {key} differs from the plan")
    # ``0 == False`` in Python: the flags must be the booleans themselves.
    if (metadata["query_roles_or_outcomes_available"] is not False
            or metadata["readout"]["is_sampler_likelihood"] is not False
            or metadata["readout"]["is_affinity"] is not False):
        raise ValueError("export metadata must record that no outcomes were available and that the readout "
                         "is neither a sampler likelihood nor an affinity")
    require_keys(payload, _PAYLOAD_KEYS, "export scores")
    if (payload["schema"] != SCORE_SCHEMA or payload["record_kind"] != "score_table"
            or payload["scorer_id"] != query["scorer_id"] or payload["readout"] != READOUT_NAME
            or payload["orientation"] != "higher_is_better"):
        raise ValueError("export scores are not the declared score table")
    if payload["provenance"] != {"queries": plan["queries_sha256"], "metadata": digest(metadata)}:
        raise ValueError("export scores do not link to the plan's panel and to this metadata")
    rows = payload["scores"]
    if not isinstance(rows, list):
        raise ValueError("export scores must be a list")
    for row in rows:
        require_keys(row, ("case_id", "score", "failure"), "score row")
    ids = [row["case_id"] for row in rows]
    declared = plan["development"]["endpoint_ids"]
    if any(not isinstance(i, str) for i in ids) or len(set(ids)) != len(ids) or sorted(ids) != sorted(declared):
        raise ValueError("export case IDs are not exactly the declared development endpoints "
                         f"(missing {sorted(set(declared) - set(ids))}, extra {sorted(set(ids) - set(declared))}, "
                         f"duplicated {sorted(i for i, c in Counter(ids).items() if c > 1)})")
    scores, failures = {}, []
    for row in rows:
        score, failure = row["score"], row["failure"]
        if failure is None:
            scores[row["case_id"]] = float(_finite(score, f"score of {row['case_id']}"))
        elif score is not None or not isinstance(failure, str) or not failure.strip():
            raise ValueError(f"a failed case ({row['case_id']}) has a null score and a nonempty failure reason")
        else:
            failures.append({"case_id": row["case_id"], "failure": failure})
    # One failed query invalidates the whole candidate: the surviving subset is never scored.
    return (None if failures else scores), sorted(failures, key=lambda f: f["case_id"])


def _release_lineage(ctx, training):
    base = training["model"]["base"]
    _release_origin(training["initialization"], base)
    if digest(base) != ctx["spec"]["identity"]["release_base_sha256"]:
        raise ValueError("the direct-release checkpoint was not adapted from the declared release backbone")
    if training["transfer_certificate"] is not None:
        raise ValueError("direct release adaptation has no broad transfer certificate")
    return None, None


def _parent_lineage(ctx, arm, origin, training):
    """Bind a post checkpoint to its declared real parent; mirrors the trainer's own lineage guard."""
    base, initialization = training["model"]["base"], training["initialization"]
    require_keys(initialization, ("kind", "base", "sha256"), "parent initialization")
    if (initialization["kind"] != "parent" or initialization["base"] != base
            or base["variant"] != "materialized-parent" or initialization["sha256"] != base["parent_sha256"]):
        raise ValueError("parent initialization does not match the actual parent")
    if initialization["sha256"] != origin["parent_sha256"]:
        raise ValueError(f"the checkpoint was adapted from a parent other than origin {origin['origin_id']!r}")
    parent = base["parent_provenance"]
    require_keys(parent, ("checkpoint_sha256", "training", "progress"), "parent provenance")
    if (parent["checkpoint_sha256"] != origin["parent_checkpoint_sha256"]
            or digest(parent) != origin["parent_provenance_sha256"]):
        raise ValueError(f"the recorded parent provenance is not the one declared for origin {origin['origin_id']!r}")
    parent_training, parent_progress = parent["training"], parent["progress"]
    source, horizon = base["source_policy_identity"], parent_training["config"]["total_updates"]
    if (not isinstance(source, dict) or parent_training["schema"] != TRAINING_SCHEMA
            or parent_training["selection"] != "fixed-last" or parent_training["config"]["stage"] != "mid"
            or parent_training["model"] != source or type(horizon) is not int or horizon < 1
            or any(type(parent_progress[key]) is not int
                   for key in ("next_update", "fitting_endpoints_available", "selection_labels"))
            or parent_progress["next_update"] != horizon
            or parent_progress["fitting_endpoints_available"] != 0 or parent_progress["selection_labels"] != 0
            or parent_progress["inherited_exposure"] is not None):
        raise ValueError("a post checkpoint must descend from a completed, unmeasured midtraining parent, "
                         "never from a post-trained checkpoint")
    _release_origin(parent_training["initialization"], parent_training["model"]["base"])
    if digest(source["base"]) != ctx["spec"]["identity"]["release_base_sha256"]:
        raise ValueError("the midtraining parent was not trained from the declared release backbone")
    if parent_training["encoder_runtime"] != training["encoder_runtime"]:
        raise ValueError("the parent's encoder runtime differs from the post checkpoint's")
    certificate = _certificate(parent_training["transfer_certificate"])
    if parent_training["data"] != {"manifest": certificate["declaration_digest"],
                                   "population": certificate["population_digest"]}:
        raise ValueError("the parent's transfer certificate does not bind its actual training population")
    if certificate["endpoint_manifest_digest"] != ctx["manifest_digest"]:
        raise ValueError("the parent's transfer certificate protects a different endpoint manifest")
    _parent_functional(parent_training, parent_progress, certificate)
    functional = parent_training["functional"]
    mode = None if functional is None else functional["config"]["gradient_mode"]
    if mode != arm["functional_mode"]:
        raise ValueError(f"arm {arm['arm_id']!r} declares functional_mode {arm['functional_mode']!r} but the "
                         f"parent trained with {mode!r}")
    if training["transfer_certificate"] is None or _certificate(training["transfer_certificate"]) != certificate:
        raise ValueError("the post checkpoint's transfer certificate is not its parent's")
    supervision = None if functional is None else parent_training["input"]["functional"]["budget"]
    broad = {"functional_mode": mode, "functional_labels": 0 if supervision is None else supervision["budget_n"],
             "functional_selection_digest": None if supervision is None else supervision["selection_digest"]}
    return {"parent_progress": copy.deepcopy(parent_progress), "external_declaration": None}, broad


def _count(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _exposure(progress, config, update, budget_ids):
    """The recorded counters and observed IDs of a post checkpoint are consistent with each other.

    The bounded, pure-JSON rules of ``checkpoint._validate``: exact counter equations for ``update`` updates,
    sorted unique nonempty ID lists with matching counts, and an endpoint exposure that is a SUBSET of the
    available budget.  It is a consistency check of supplied records: it never requires every available endpoint
    to have been seen and never replays the draw history.
    """
    require_keys(progress["counters"], _COUNTER_KEYS, "checkpoint counters")
    counters = {key: _count(value, f"counter {key}") for key, value in progress["counters"].items()}
    examples = update * config.examples_per_update
    draws = examples if config.ordering_weight else 0
    if (counters["examples"] != examples or counters["masked_residues"] < examples
            or counters["antigen_examples"] != (examples if config.arm == "paired" else 0)
            or counters["pair_draws"] != draws
            or counters["informative_pairs"] + counters["ties"] != draws
            or counters["teacher_forced_states"] != 20 * counters["informative_pairs"]):
        raise ValueError("the checkpoint's training counters are inconsistent with its update and recipe")
    require_keys(progress["seen"], _SEEN_KEYS, "checkpoint seen IDs")
    require_keys(progress["unique"], _SEEN_KEYS, "checkpoint unique counts")
    for key in _SEEN_KEYS:
        values = progress["seen"][key]
        if (not isinstance(values, list) or any(not isinstance(x, str) or not x for x in values)
                or values != sorted(set(values))):
            raise ValueError(f"checkpoint seen {key} must be a sorted list of unique nonempty IDs")
        if (type(progress["unique"][key]) is not int or progress["unique"][key] != len(values)
                or len(values) > examples + 2 * draws):
            raise ValueError(f"checkpoint unique count of {key} does not match its seen IDs and exposure")
    if any(progress["seen"][key] for key in _SEEN_KEYS if key not in _ENDPOINT_KEYS):
        raise ValueError("a post checkpoint records only endpoint exposure, no broad record or family")
    seen = {key: set(progress["seen"][key]) for key in _ENDPOINT_KEYS}
    if seen["endpoints"] - set(budget_ids):
        raise ValueError("the checkpoint saw endpoints outside its available fitting budget")
    if (seen["endpoints"] != seen["reconstruction_endpoints"] | seen["ordering_endpoints"]
            or not seen["informative_ordering_endpoints"] <= seen["ordering_endpoints"]
            or len(seen["reconstruction_endpoints"]) > examples
            or len(seen["ordering_endpoints"]) > 2 * draws
            or len(seen["informative_ordering_endpoints"]) > 2 * counters["informative_pairs"]):
        raise ValueError("inconsistent reconstruction and measurement exposure")


def _checkpoint_fields(ctx, trial, arm, origin, boundary, checkpoint):
    spec = ctx["spec"]
    recipe, identity = spec["recipe"], spec["identity"]
    require_keys(checkpoint, ("checkpoint_sha256", "training", "progress"), "checkpoint provenance")
    sha256_hex(checkpoint["checkpoint_sha256"], "checkpoint_sha256")
    training, progress = checkpoint["training"], checkpoint["progress"]
    require_keys(training, _TRAINING_KEYS, "checkpoint training identity")
    require_keys(progress, _PROGRESS_KEYS, "checkpoint progress")
    if training["schema"] != TRAINING_SCHEMA or training["selection"] != "fixed-last":
        raise ValueError("a study checkpoint is a fixed-last training identity of the current schema")
    # canonical_bytes keeps ``0`` from passing for ``False`` and ``1`` for ``True`` in the recorded optimizer flags.
    if (canonical_bytes(training["optimizer"]) != canonical_bytes(_OPTIMIZER) or training["rng"] != RNG_ALGORITHM
            or training["ordering_readout"] != _ORDERING_READOUT):
        raise ValueError("a study checkpoint records the fixed AdamW recipe, the current random algorithm and "
                         "the dropout-disabled ordering readout")
    config = TrainingConfig(**recipe["training"], learning_rate=trial["learning_rate"], seed=trial["post_seed"])
    try:
        TrainingConfig(**training["config"])     # the dict equality below would read ``True`` as ``1``
    except ValueError as error:
        raise ValueError(f"the checkpoint's training config is not a valid training config: {error}") from error
    if training["config"] != asdict(config):
        raise ValueError("the checkpoint's training config is not the declared post recipe at this learning "
                         "rate, seed and horizon")
    model, encoder_runtime, runtime = training["model"], training["encoder_runtime"], training["runtime"]
    if (model["fusion"] != recipe["fusion"] or model["update_mode"] != recipe["fusion"]["update_mode"]
            or model["compute_dtype"] != recipe["fusion"]["compute_dtype"]):
        raise ValueError("the checkpoint's fusion config differs from the declared common fusion config")
    if model["encoder"] != identity["encoder"]["identity"] or encoder_runtime != identity["encoder"]["runtime"]:
        raise ValueError("the checkpoint's encoder identity or runtime differs from the declared encoder")
    if encoder_runtime["encoder"] != model["encoder"]:
        raise ValueError("the checkpoint's encoder runtime and model name different encoders")
    if runtime["model"] != model or runtime["encoder"] != encoder_runtime:
        raise ValueError("the checkpoint's runtime record names a different model or encoder than its training")
    if {key: runtime[key] for key in ("hardware", "software", "parameters")} != identity["execution"]:
        raise ValueError("the checkpoint's execution (hardware, software, parameters) differs from the "
                         "declared execution contract")
    source = training["source"]
    require_keys(source, ("schema", "files", "sha256"), "training source identity")
    if (source["schema"] != SOURCE_SCHEMA or digest(source["files"]) != source["sha256"]
            or source["sha256"] != identity["source_sha256"]):
        raise ValueError("the checkpoint's source identity is not the declared source")
    budget, recorded = ctx["budgets"][trial["budget_n"]], training["input"]
    if training["data"] != {"selected_data": budget["selected_data_sha256"]} or (
            recorded["selected_data"] != budget["selected_data_sha256"]):
        raise ValueError("the checkpoint does not carry the declared selected_data SHA256 for this budget")
    if recorded["selected_labels_sha256"] != budget["selected_labels_sha256"]:
        raise ValueError("the checkpoint does not carry the declared selected_labels_sha256 for this budget")
    if recorded["endpoint_manifest_sha256"] != ctx["manifest_sha256"] or recorded["manifest"] != ctx["manifest_digest"]:
        raise ValueError("the checkpoint was trained against a different endpoint manifest")
    if recorded["budget"] != budget["identity"]:
        raise ValueError("the checkpoint's fitting budget is not the canonical selection for this budget size")
    if training["functional"] is not None or recorded["functional"] is not None or progress["functional"] is not None:
        raise ValueError("post-training takes no functional supervision")
    for key, expected in (("next_update", boundary), ("fitting_endpoints_available", trial["budget_n"]),
                          ("selection_labels", 0)):
        if type(progress[key]) is not int or progress[key] != expected:
            raise ValueError(f"checkpoint progress {key} is {progress[key]!r}, expected {expected}")
    _exposure(progress, config, boundary, budget["identity"]["selected_ids"])
    inherited, broad = (_release_lineage(ctx, training) if arm["kind"] == "release"
                        else _parent_lineage(ctx, arm, origin, training))
    if progress["inherited_exposure"] != inherited:
        raise ValueError("the checkpoint's inherited exposure is not its declared parent's progress")
    return {"checkpoint_sha256": checkpoint["checkpoint_sha256"], "training_sha256": digest(training),
            "available": progress["fitting_endpoints_available"], "seen": len(progress["seen"]["endpoints"]),
            "broad": broad}


def _check_checkpoint(ctx, trial, arm, origin, boundary, checkpoint):
    """Everything an export's recorded checkpoint provenance must say; nothing is loaded or trained."""
    try:
        return _checkpoint_fields(ctx, trial, arm, origin, boundary, checkpoint)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"malformed checkpoint provenance: {error!r}") from error


def _check_export(ctx, trial, boundary, directory, declared_sha256):
    plan = ctx["plan"]
    arm = ctx["arms"][trial["arm_id"]]
    origin = next(o for o in arm["origins"] if o["origin_id"] == trial["origin_id"])
    manifest, payload, metadata = _load_export(directory, declared_sha256)
    scores, failures = _scores(plan, metadata, payload)           # also fixes the shape of the metadata
    if manifest["checkpoint"] != metadata["checkpoint"] or manifest["queries_sha256"] != metadata["queries_sha256"]:
        raise ValueError("the export manifest does not link to its metadata")
    checkpoint = _check_checkpoint(ctx, trial, arm, origin, boundary, metadata["checkpoint"])
    training = metadata["checkpoint"]["training"]
    if metadata["model"] != training["model"] or metadata["encoder_runtime"] != training["encoder_runtime"]:
        raise ValueError("the export metadata names a different model or encoder runtime than the checkpoint")
    return {"status": "exported", **checkpoint, "scores": scores, "failures": failures,
            "manifest_sha256": declared_sha256}


def _context(plan, manifest):
    spec = plan["spec"]
    return {"plan": plan, "spec": spec, "manifest_sha256": plan["manifest"]["file_sha256"],
            "manifest_digest": manifest.manifest_digest(),
            "trials": {t["trial_id"]: t for t in plan["trials"]},
            "arms": {a["arm_id"]: a for a in spec["arms"]},
            "budgets": {b["budget_n"]: b for b in plan["budgets"]}}


def _inventory(plan, results):
    require_keys(results, ("schema", "plan_sha256", "records"), "study results index")
    if results["schema"] != RESULTS_SCHEMA:
        raise ValueError("unsupported study results schema")
    if results["plan_sha256"] != plan["plan_sha256"]:
        raise ValueError("the results index belongs to a different plan")
    rows = results["records"]
    if not isinstance(rows, list):
        raise ValueError("results records must be a list")
    for row in rows:
        if not isinstance(row, dict) or row.get("status") not in ("exported", "failed"):
            raise ValueError("every results record has status exported or failed")
        exported = row["status"] == "exported"
        require_keys(row, _EXPORTED_KEYS if exported else _FAILED_KEYS, "results record")
        text_value(row["trial_id"], "trial_id")
        if type(row["boundary"]) is not int:
            raise ValueError("a results record boundary must be an integer")
        if exported:
            text_value(row["export_path"], "export_path")
            sha256_hex(row["manifest_sha256"], "manifest_sha256")
        else:
            text_value(row["reason"], "failure reason")
    seen = Counter((row["trial_id"], row["boundary"]) for row in rows)
    planned = {(t["trial_id"], b) for t in plan["trials"] for b in plan["boundaries"]}
    missing, extra = sorted(planned - set(seen)), sorted(set(seen) - planned)
    duplicated = sorted(key for key, count in seen.items() if count > 1)
    if missing or extra or duplicated:
        raise ValueError(
            f"the results index is not the planned inventory: {len(missing)} missing (first {missing[:2]}), "
            f"{len(extra)} unplanned (first {extra[:2]}), {len(duplicated)} duplicated (first {duplicated[:2]})")
    return {(row["trial_id"], row["boundary"]): row for row in rows}


def _check_results(plan, manifest, results, base):
    """Inventory, then EVERY export; no development label exists in this function's reach."""
    ctx = _context(plan, manifest)
    rows = _inventory(plan, results)
    items = {}
    for trial in plan["trials"]:
        for boundary in plan["boundaries"]:
            row = rows[(trial["trial_id"], boundary)]
            if row["status"] == "failed":
                items[(trial["trial_id"], boundary)] = {"status": "export_failed", "reason": row["reason"]}
            else:
                items[(trial["trial_id"], boundary)] = _check_export(
                    ctx, trial, boundary, Path(base) / row["export_path"], row["manifest_sha256"])
    exported = [(key, item) for key, item in items.items() if item["status"] == "exported"]
    counts = Counter(item["checkpoint_sha256"] for _, item in exported)
    if any(count > 1 for count in counts.values()):
        raise ValueError("two records report the same checkpoint SHA256; every boundary is its own checkpoint")
    declared = {value for arm in plan["spec"]["arms"] for origin in arm["origins"]
                for name, value in origin.items() if name != "origin_id"}
    if set(counts) & declared:
        raise ValueError("a post checkpoint reports the SHA256 of a declared parent; a post-trained "
                         "checkpoint is never a parent")
    trained, broad = {}, {}
    for (trial_id, _), item in exported:
        if trained.setdefault(trial_id, item["training_sha256"]) != item["training_sha256"]:
            raise ValueError("a trial's boundaries record different training identities")
        trial = ctx["trials"][trial_id]
        if broad.setdefault((trial["arm_id"], trial["origin_id"]), item["broad"]) != item["broad"]:
            raise ValueError("an origin's records disagree about its broad supervision")
    return items, broad


# ---------------------------------------------------------------------------
# Selection (after validation: the development labels are opened here and not before)
# ---------------------------------------------------------------------------

def _development_labels(plan, document):
    require_keys(document, ("schema", "plan_sha256", "assay_id", "orientation", "labels"), "development labels")
    if document["schema"] != LABELS_SCHEMA:
        raise ValueError("unsupported development label schema")
    if document["plan_sha256"] != plan["plan_sha256"]:
        raise ValueError("the development labels belong to a different plan")
    development = plan["development"]
    if document["assay_id"] != development["assay_id"] or document["orientation"] != development["orientation"]:
        raise ValueError("the development labels declare another assay or orientation than the plan")
    if not isinstance(document["labels"], list):
        raise ValueError("development labels must be a list")
    labels = [record(DevelopmentLabel, row) for row in document["labels"]]
    for label in labels:
        if label.assay_id != development["assay_id"] or label.orientation != development["orientation"]:
            raise ValueError("every development label carries the plan's one assay and one orientation")
    ids = [label.endpoint_id for label in labels]
    declared = set(development["endpoint_ids"])
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate development label")
    if set(ids) != declared:
        raise ValueError("development labels must cover exactly the declared development endpoints "
                         f"(missing {sorted(declared - set(ids))}, extra {sorted(set(ids) - declared)})")
    return {label.endpoint_id: oriented(float(label.value), label.orientation) for label in labels}


def _exact(row):
    """A scored row's metric as the exact fraction its integer pair counts give.  Floats are for reporting only:
    two different decompositions of one rational mean can differ in the last float bit."""
    pairs = row["pairs"]
    return Fraction(2 * pairs["concordant"] + pairs["tied_predictions"], 2 * pairs["comparable_pairs"])


def _exact_mean(rows):
    return sum((_exact(row) for row in rows), Fraction(0)) / len(rows)


def _row(trial, boundary, item, values, ids):
    row = {"trial_id": trial["trial_id"], "boundary": boundary, "status": item["status"], "metric": None,
           "pairs": None, "reason": item.get("reason"), "failures": [], "checkpoint_sha256": None,
           "available_endpoints": None, "seen_endpoints": None}
    if item["status"] == "export_failed":
        return row
    row.update(checkpoint_sha256=item["checkpoint_sha256"], available_endpoints=item["available"],
               seen_endpoints=item["seen"], failures=item["failures"])
    if item["failures"]:
        row["status"] = "invalid_scores"
        return row
    result = ordinal_concordance([item["scores"][i] for i in ids], [values[i] for i in ids])
    row.update(status="scored", metric=result["value"],
               pairs={k: result[k] for k in ("comparable_pairs", "concordant", "tied_predictions", "discordant")})
    return row


def _curve(plan, rows, trial_ids, rate, label):
    trials = {t["trial_id"]: t for t in plan["trials"]}
    points = []
    for boundary in plan["boundaries"]:
        members, scored = [], []
        for trial_id in trial_ids:
            row = rows[(trial_id, boundary)]
            members.append({"trial_id": trial_id, "origin_id": trials[trial_id]["origin_id"],
                            "post_seed": trials[trial_id]["post_seed"], "status": row["status"],
                            "metric": row["metric"], "reason": row["reason"], "failures": row["failures"]})
            if row["status"] == "scored":
                scored.append(row)
        points.append({"boundary": boundary, "planned": len(members), "valid": len(scored),
                       "mean_metric": float(_exact_mean(scored)) if len(scored) == len(members) else None,
                       "members": members})
    return {"learning_rate": rate, "label": label, "points": points}


def _evaluate(plan, items, values):
    """Per-trial rows, the final-horizon candidates, the winners and both curves; ``values`` are oriented."""
    ids, spec = plan["development"]["endpoint_ids"], plan["spec"]
    ties = sum(count * (count - 1) // 2 for count in Counter(values[i] for i in ids).values())
    if len(ids) * (len(ids) - 1) // 2 == ties:
        raise ValueError("the development metric is undefined: no pair of development endpoints has "
                         "different measured values")
    trials = {t["trial_id"]: t for t in plan["trials"]}
    rows = {(t["trial_id"], b): _row(t, b, items[(t["trial_id"], b)], values, ids)
            for t in plan["trials"] for b in plan["boundaries"]}
    groups = {}
    for trial in plan["trials"]:
        groups.setdefault((trial["arm_id"], trial["budget_n"], trial["learning_rate"]), []).append(trial["trial_id"])
    final, rates = plan["boundaries"][-1], spec["recipe"]["learning_rates"]
    candidates, selected, curves = [], [], []
    for arm in spec["arms"]:
        for budget in plan["budgets"]:
            best = None
            for rate in rates:
                members = groups[(arm["arm_id"], budget["budget_n"], rate)]
                finals = [rows[(trial_id, final)] for trial_id in members]
                scored = [r for r in finals if r["status"] == "scored"]
                eligible = len(scored) == len(finals)
                mean = _exact_mean(scored) if eligible else None          # an exact Fraction, compared exactly
                candidates.append({
                    "arm_id": arm["arm_id"], "budget_n": budget["budget_n"], "learning_rate": rate,
                    "planned_trials": len(finals), "valid_final_trials": len(scored), "eligible": eligible,
                    "mean_final_metric": None if mean is None else float(mean),
                    "ineligible": [{"trial_id": r["trial_id"], "status": r["status"], "reason": r["reason"],
                                    "failures": r["failures"]} for r in finals if r["status"] != "scored"]})
                # Rates ascend, so a strict improvement keeps the smallest rate on an exact tie.
                if eligible and (best is None or mean > best[1]):
                    best = (rate, mean)
            chosen = None if best is None else best[0]
            winners = [] if chosen is None else [
                {"trial_id": trial_id, "origin_id": trials[trial_id]["origin_id"],
                 "post_seed": trials[trial_id]["post_seed"], "metric": rows[(trial_id, final)]["metric"],
                 "checkpoint_sha256": rows[(trial_id, final)]["checkpoint_sha256"]}
                for trial_id in groups[(arm["arm_id"], budget["budget_n"], chosen)]]
            selected.append({
                "arm_id": arm["arm_id"], "budget_n": budget["budget_n"], "selected_learning_rate": chosen,
                "mean_final_metric": None if best is None else float(best[1]), "final_boundary": final,
                "reason": None if best is not None else "no learning rate has a valid final score for every planned trial",
                "winning_trials": winners})
            curves.append({
                "arm_id": arm["arm_id"], "budget_n": budget["budget_n"],
                "retrospective_selected": None if chosen is None else _curve(
                    plan, rows, groups[(arm["arm_id"], budget["budget_n"], chosen)], chosen, RETROSPECTIVE),
                "prospective_fixed": _curve(
                    plan, rows, groups[(arm["arm_id"], budget["budget_n"], spec["recipe"]["prospective_learning_rate"])],
                    spec["recipe"]["prospective_learning_rate"], PROSPECTIVE)})
    return rows, candidates, selected, curves


def _broad_supervision(plan, broad):
    """One entry per DECLARED midtraining origin, never only the origins that happen to have an export.

    ``verified`` means the origin's exported records agree with each other and with the declared parent (supplied-
    record consistency, not proof of execution).  An origin whose records all failed has no such export, so its
    functional supervision is ``unavailable`` and its counts are unknown (null), not zero.  A direct-release
    arm has no broad stage and no entry.
    """
    entries = []
    for arm in plan["spec"]["arms"]:
        if arm["kind"] != "mid":
            continue
        for origin in arm["origins"]:
            identity = {"arm_id": arm["arm_id"], "origin_id": origin["origin_id"],
                        "declared_functional_mode": arm["functional_mode"]}
            value = broad.get((arm["arm_id"], origin["origin_id"]))
            if value is None:
                entries.append({**identity, "status": "unavailable", "functional_mode": None,
                                "functional_labels": None, "functional_selection_digest": None,
                                "reason": "no export of this origin was recorded, so its parent's functional "
                                          "supervision was not verified"})
            else:
                entries.append({**identity, "status": "verified", **value})
    return sorted(entries, key=lambda e: (e["arm_id"], e["origin_id"]))


def _accounting(plan, broad):
    sizes = [b["budget_n"] for b in plan["budgets"]]
    largest = plan["budgets"][-1]["identity"]["selected_ids"]
    development = plan["development"]["endpoint_ids"]
    union = set(largest) | set(development)
    if len(union) != len(largest) + len(development):
        raise ValueError("the largest fitting budget overlaps the development endpoints")
    return {
        "fitting_labels_by_budget": {str(n): n for n in sizes},
        "unique_fitting_labels": len(largest),
        "fitting_budgets": "nested prefixes of one seeded order; the largest contains every smaller budget",
        "development_labels": len(development),
        "unique_label_union": len(union),
        "development_labels_used_in_training": 0,
        "broad_supervision": _broad_supervision(plan, broad),
        "broad_supervision_note": "functional supervision labels of a midtraining parent, counted separately "
                                  "from the post fitting and development labels above; one entry per declared "
                                  "midtraining origin, null where no export verified it",
    }


def select_study(plan, manifest, results, load_labels, *, base, files):
    """Validate everything, then open the development labels, then select.

    ``load_labels`` is called exactly once and only after the plan, the inventory and every export
    validated; it returns ``(document, file_sha256)``.  ``base`` resolves relative export paths.
    ``files`` carries the SHA256 of the plan, manifest and results index files.  The result is a new
    document; nothing is written and no checkpoint is removed.
    """
    require_keys(files, ("plan", "manifest", "results"), "input file hashes")
    for name, value in files.items():
        sha256_hex(value, f"{name} file SHA256")
    plan = validate_plan(plan, manifest, files["manifest"])
    items, broad = _check_results(plan, manifest, results, base)
    document, labels_sha256 = load_labels()
    sha256_hex(labels_sha256, "development labels file SHA256")
    rows, candidates, selected, curves = _evaluate(plan, items, _development_labels(plan, document))
    spec, arms, budgets = plan["spec"], plan["spec"]["arms"], plan["budgets"]
    decisions = len(arms) * len(budgets)
    result = _plain({
        "schema": SELECTION_SCHEMA, "scope": SCOPE, "assumptions": list(ASSUMPTIONS),
        "inputs": {"plan_sha256": plan["plan_sha256"], "plan_file_sha256": files["plan"],
                   "manifest_file_sha256": files["manifest"], "results_file_sha256": files["results"],
                   "results_digest": digest(results), "development_labels_file_sha256": labels_sha256,
                   "development_labels_digest": digest(document)},
        "metric": plan["metric"], "selection_rule": plan["selection_rule"],
        "counts": {"trials": len(plan["trials"]), "exports": len(rows),
                   "failed_exports": sum(r["status"] == "export_failed" for r in rows.values()),
                   "invalid_scores": sum(r["status"] == "invalid_scores" for r in rows.values()),
                   "scored": sum(r["status"] == "scored" for r in rows.values())},
        "label_accounting": _accounting(plan, broad),
        "opportunities": {
            "selection_decisions": decisions, "learning_rates_compared_per_decision": len(spec["recipe"]["learning_rates"]),
            "candidate_learning_rates": decisions * len(spec["recipe"]["learning_rates"]),
            "final_horizon_scorings": len(plan["trials"]), "all_boundary_scorings": len(rows),
            "note": "every scoring reuses the same development labels; the selected rate is optimistic for them"},
        "candidates": candidates, "selected": selected, "curves": curves,
        "evaluations": [rows[(t["trial_id"], b)] for t in plan["trials"] for b in plan["boundaries"]],
    })
    result["result_sha256"] = digest(result)
    return result


def select_study_files(plan_path, manifest_path, results_path, labels_path):
    """File-level selection over one byte snapshot per file; the labels file is not opened until everything
    else validated.  Every recorded file hash is the hash of the bytes the document was parsed from."""
    plan, plan_sha256 = read_snapshot(plan_path)
    manifest, manifest_sha256 = _manifest_snapshot(manifest_path)
    results, results_sha256 = read_snapshot(results_path)
    return select_study(plan, manifest, results, lambda: read_snapshot(labels_path),
                        base=Path(results_path).resolve().parent,
                        files={"plan": plan_sha256, "manifest": manifest_sha256, "results": results_sha256})
