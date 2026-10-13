"""Frozen inputs for the pilot: restore the decisions, load the sources, validate, then (only then) train.

The production flight already decided three things and wrote them down: the split
(``hard_split_freeze.json``), the condition vocabulary (``condition_vocabulary.json``) and what each
source contributed (``source_inventory.json``). This module reuses those decisions and reuses the
existing source loaders; it does not freeze, split, acquire or prepare anything.

**One narrow subclass.** :class:`FrozenInputServices` overrides exactly ``split``, ``closure`` and
``freeze_hard_split``: the first two return the restored membership and the closure built from it,
and the third refuses. It adds ``adopt`` (installs those two) and ``bin_assignments`` (a read-only
copy of the population-to-bin mapping the inherited services keep in a private cache and expose
nowhere). The inherited ``build_sources(retain_observations=True)``,
``training_pools`` and ``curriculum_stream`` then run unchanged against a split nobody can
re-derive. The frozen vocabulary is copied into the fresh run root before the services are built, so
the inherited ``freeze_condition_vocabulary`` compares what the sources derive with the frozen
vocabulary (assays AND per-assay counts) and refuses a difference; it never resizes a table.

**Order is the point.** ``load_inputs`` (1) reads and digests the three frozen files, (2) restores
and validates the membership from the sequence-only candidate set, (3) verifies source identities
that an acquisition manifest describes, (4) calls ``build_sources`` (whose loaders already gate the
HER2 outcome read by the restored windows before any value is converted), (5) validates the typed
observations and the summaries it produced, and (6) compares the derived inventory with the frozen
one. A model is built only after all of that, by a caller that holds the returned
:class:`LoadedInputs`.

**The frozen inventory is checked for completeness first.** A frozen inventory missing a field the
comparison reads (or one that is ``{}``) is refused before the vocabulary is installed and before any
source is opened: a comparison that is conditional on what the frozen record happens to carry
compares nothing. :func:`compare_inventory` then projects both inventories onto the same explicit set
of scientific fields (:data:`INVENTORY_SHAPE`) and compares the projections exactly. Descriptive prose
and the paired-OAS ``reused_cache`` flag are the only fields left out, and each is named.

**Bins are a declared policy, not a replay.** The production flight fitted its quality-bin thresholds
on ``row.value``, the mean of a summary's UNCENSORED replicates, over summaries that ALSO held
censored replicates; the current source code fits them on the train rows' exact points only and so
excludes those summaries. Same source bytes, different cuts. The pilot declares
``inputs.bin_policy = current_exact_training``: every arm uses the CURRENT thresholds, they are frozen
here and the historical thresholds, support and assigned count are kept as evidence with every
difference reported. Nothing claims a historical replay.

**What the current bins are, as numbers.** Digests alone would not say which population sits in which
bin, so :func:`fingerprint_bin_assignments` hashes the actual mapping the inherited services built
(every loaded assigned row, train and development; the cuts are fitted on train only and no reserved
HER2 window is loaded), :func:`bind_current_bins` puts that fingerprint and the COMPLETE current thresholds
and support into one small record, and :func:`write_current_bins` writes it as ``current_bins.json``
before any model exists. Its content digest and the assignment digest and count are part of
``LoadedInputs.bin_identity``, hence of the run identity and the snapshot's. The assignment rows
themselves are not saved, and the bounded list of changed populations in ``LoadedInputs.bins`` is
a readability aid, never the only numerical record.

**What "validated" covers.** The boundary is the typed, loaded :class:`Observation` and
:class:`ConstructSummary` records that SURVIVED the legacy closure -- their population key, family,
parent, scale, partition, declared region, measurement kind, ordinal class and duplicate evidence --
and the loader digests and counts in the inventory. ``observation_closure`` drops duplicate
observations before this module sees them and offers no hook, so the discarded rows are NOT
validated (``discarded_duplicate_evidence_validated`` is ``false`` in the record, beside the frozen
and current duplicate counts); the source hashes, the source-code closure and the inventory bounds
are population evidence, not proof that a discarded row agrees with its survivor. A column no loader
reads is not examined, and a loader that parses a value wrongly is outside what this can see.
"""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import her2_routes_data as data_lib
from .. import her2_routes_sources as sources_lib
from .. import her2_routes_spec as spec
from .. import her2_routes_split as split_lib
from .. import her2_support_paths as paths
from ..her2_routes_conditions import BIN_HIGH, BIN_LOW, BIN_MID
from ..her2_routes_serial import her2_core_edit_map
from ..her2_routes_services import FlightServices
from ..her2_runtime import require
from . import membership as membership_lib
from .config import BIN_POLICIES, ROUTE

HER2_SOURCE = "alphabind_her2_validation"
SURVIVOR_SCOPE = "post_legacy_closure_typed_survivors"
FROZEN_FILES = {"split": spec.SPLIT_FREEZE, "vocabulary": "condition_vocabulary.json",
                "inventory": "source_inventory.json"}
VOCABULARY_FILE = FROZEN_FILES["vocabulary"]
#: The record of the bins every arm uses, written under the run root before any model exists.
CURRENT_BINS_FILE = "current_bins.json"
CURRENT_BINS_SCHEMA = "her2-multiscale-current-bins/1"
#: Version of the canonical record form the assignment digest is taken over (see
#: :func:`fingerprint_bin_assignments`); a change of form is a change of this string.
ASSIGNMENT_SCHEMA = "her2-multiscale-bin-assignments/1"
#: The bins ``fit_condition_bins`` can assign; ``null`` is a dropout, never an assignment.
ASSIGNED_BINS = (BIN_LOW, BIN_MID, BIN_HIGH)


class FrozenInputServices(FlightServices):
    """``FlightServices`` whose split, closure and freeze are supplied, never derived."""

    def adopt(self, membership, closure):
        """Install the restored decisions. Nothing reads a split before this has been called."""
        self._cache["split"] = membership
        self._cache["closure"] = closure
        return self

    def bin_assignments(self):
        """A copy of the population -> bin mapping ``build_sources`` assigned.

        Keys are ``(full construct identity, study, assay, target)``, values ``low``/``mid``/``high``.
        The inherited services keep this in a private cache (``curriculum_stream`` reads the same
        mapping unchanged) and offer no getter; this one reads it and changes nothing.
        """
        require("bins" in self._cache,
                "no bin assignment exists yet (build_sources has not run)")
        return dict(self._cache["bins"])

    def split(self):
        require("split" in self._cache,
                "the restored split membership has not been installed (adopt() first); the "
                "services never derive one")
        return self._cache["split"]

    def closure(self):
        require("closure" in self._cache,
                "the closure of the restored split has not been installed (adopt() first)")
        return self._cache["closure"]

    def freeze_hard_split(self, *, persist=True):
        raise ValueError("The pilot restores the frozen split from its record and never freezes, "
                         "rebuilds or rewrites one")


@dataclass
class LoadedInputs:
    """Everything the runner needs from the input stage, all of it already validated."""

    services: object
    membership: object
    closure: object
    frozen: dict
    identities: dict
    validation: dict
    training_keys: frozenset
    pool: tuple
    panel: list
    narrowed_routes: dict = field(default_factory=dict)
    bins: dict = field(default_factory=dict)
    #: The complete current-bins record (:func:`bind_current_bins`); persisted by
    #: :func:`write_current_bins`, which also fills ``bins_file``.
    current_bins: dict = field(default_factory=dict)
    bins_file: dict = field(default_factory=dict)

    @property
    def window_of(self):
        return self.services.closure_window

    @property
    def bin_identity(self):
        """The bin policy and the digests of what it froze: thresholds, support and assignments."""
        return dict(self.bins["frozen_identity"])

    def document(self):
        return {"frozen_files": self.frozen, "membership": self.membership.document(),
                "source_identities": self.identities, "validation": self.validation,
                "bin_policy": self.bins, "current_bins_file": self.bins_file,
                "training_population_keys": len(self.training_keys),
                "her2_training_pool_summaries": len(self.pool),
                "development_panel_rows": len(self.panel),
                "configured_routes": self.narrowed_routes,
                "boundary": ("typed loaded observations and summaries, the loaders' own digests "
                             "and counts, and the manifest digests listed above; not every "
                             "raw-file field")}


def narrowed_config(base):
    """The base config reduced to the one route the pilot trains.

    The services read ``routes.enabled`` and each route's block; a second route listed there, or its
    block left in place, would be a route the pilot silently carries. Everything else is the base's.
    """
    config = copy.deepcopy(base)
    routes = config["routes"]
    require(ROUTE in routes and ROUTE in routes.get("enabled", ()),
            f"the base config does not enable {ROUTE}")
    for name in [name for name in routes if name.startswith("dplm_") and name != ROUTE]:
        del routes[name]
    routes["enabled"] = [ROUTE]
    return config


def read_frozen_records(frozen_dir):
    """The three frozen files of the production flight, each with its two digests.

    ``file_sha256`` is the byte digest (what is copied or compared as a file); ``document_sha256`` is
    ``paths.digest_document`` of the parsed JSON, the digest the production flight's checkpoint
    identities and the base config's pins use.
    """
    records = {}
    for name, filename in FROZEN_FILES.items():
        target = Path(frozen_dir) / filename
        require(target.is_file(), f"the frozen {name} record {target} is missing")
        document = paths.read_json(target)
        records[name] = {"path": str(target), "file_sha256": paths.sha256_file(target),
                         "document_sha256": paths.digest_document(document),
                         "document": document}
    return records


def install_frozen_vocabulary(context, frozen):
    """Copy the frozen vocabulary byte-for-byte into the fresh run root.

    The services build their condition spec from ``context.path('condition_vocabulary.json')`` when
    it exists, and ``freeze_condition_vocabulary`` compares what the sources derive with that file
    instead of writing a new one. Putting the frozen file there is the whole mechanism.
    """
    target = context.path(VOCABULARY_FILE)
    require(not target.exists(), f"{target} already exists in what should be a fresh run root")
    shutil.copyfile(frozen["vocabulary"]["path"], target)
    require(paths.sha256_file(target) == frozen["vocabulary"]["file_sha256"],
            "the copied vocabulary does not hash to the frozen one")
    return target


# ---------------------------------------------------------------------------
# source identities
# ---------------------------------------------------------------------------

def verify_source_identities(context):
    """Re-hash every configured source file an acquisition manifest describes; record the rest.

    There are several acquisition manifests and no single one covers all the bulk data: files and
    directories under the artifact root are verified here against the merged manifest table;
    repository-root files are hashed and recorded; whatever neither can describe is recorded as such
    and left to the loader's own digest, which ``compare_inventory`` checks against the frozen
    inventory. The DPLM files are bound to their pins through the DPLM manifest, and the pinned native
    loader verifies the same digests again when the model is built.
    """
    config = context.config
    table, per_manifest = spec.read_all_manifests(context)
    closure = spec.external_closure(
        context, table=table, extra=tuple(config["artifacts"].get("external_sources_extra") or ()))
    dplm = spec.require_dplm_binding(context, table=table, per_manifest=per_manifest,
                                     closure=closure)
    rows, wanted = [], []
    for name, entry in sorted(dict(config["sources"]["paths"]).items()):
        logicals = entry.get("logicals") or ([entry["logical"]] if entry.get("logical") else [])
        root = entry.get("root", "artifact")
        for logical in logicals:
            row = {"source": name, "root": root, "logical": logical}
            if root == "artifact":
                key = spec.normalise_artifact_path(logical, context.artifact_root)
                members = [key] if key in table else sorted(
                    item for item in table if item.startswith(key + "/"))
                target = context.artifact(logical)
                require(target.exists(), f"the configured source {name}: {target} is missing")
                row.update(verification="acquisition_manifest" if members else "loader_digest_only",
                           manifest_files=len(members))
                wanted.extend(members)
            else:
                target = context.repository(logical)
                require(target.is_file(), f"the configured source {name}: {target} is missing")
                row.update(verification="file_sha256", sha256=paths.sha256_file(target))
            rows.append(row)
    digests = spec.external_source_digests(context, {}, logical_paths=wanted, table=table)
    dependency = config["artifacts"].get("dependency_manifest")
    dependency_path = context.artifact(dependency) if dependency else None
    return {"manifests": per_manifest, "dplm": dplm, "sources": rows,
            "dependency_manifest": {
                "logical": dependency,
                "sha256": (paths.sha256_file(dependency_path)
                           if dependency_path is not None and dependency_path.is_file() else None)},
            "manifest_files_verified": len(digests),
            "manifest_files_digest": paths.sha256_text(
                "\n".join(f"{name} {digests[name]}" for name in sorted(digests))),
            "note": ("several manifests, no single marker: an artifact-root source is verified "
                     "only where a manifest describes it; the rest carry the loader's digest, "
                     "compared with the frozen inventory")}


# ---------------------------------------------------------------------------
# validation of what was loaded
# ---------------------------------------------------------------------------

def plain(node):
    """JSON types for a record: numpy scalars and arrays coerced, tuples to lists, keys to text."""
    if isinstance(node, dict):
        return {str(key): plain(value) for key, value in node.items()}
    if isinstance(node, (list, tuple, set, frozenset)):
        return [plain(value) for value in node]
    if isinstance(node, np.generic):
        return node.item()
    if isinstance(node, np.ndarray):
        return node.tolist()
    return node


def _short(key):
    return hashlib.blake2b(json.dumps(key, separators=(",", ":")).encode("utf-8"),
                           digest_size=16).digest()


def validate_observations(observations, membership):
    """Population-level consistency of the typed observations, and the HER2 rows against the split.

    Per population key (identity, study, assay, target): family, parent, scale, partition, declared
    region, measurement ``kind`` and ``ordinal_class`` must agree across its observations, because
    the summary takes the first and would silently carry its values; no observation may repeat; and
    every HER2 row must sit in a train or development window whose restored assignment equals the
    row's own partition. Raises with the first conflicts, before any model exists.

    ``kind`` is compared exactly, with no point/interval compatibility: a point and an interval
    reading of one population are different measurement types, and the summary would keep the first
    one's. Censoring (left, right, none, mixed) is NOT part of the facts -- one interval population
    legitimately holds exact and floor-censored replicates -- so a change of censoring is never read
    as a change of measurement type. ``ordinal_class`` is a non-negative integer for an ordinal
    observation and ``None`` for every other kind.

    **Scope: ``post_legacy_closure_typed_survivors``.** ``observation_closure`` drops duplicate
    observations before the typed records reach this function and exposes no hook, so what is
    checked is the survivors; the discarded duplicates are not (``discarded_duplicate_evidence_
    validated`` is ``False`` in the report). A pre-deduplication evidence audit is deferred.
    """
    sites = her2_core_edit_map()
    populations, seen, conflicts, her2_problems, semantics = {}, set(), [], [], []
    her2 = {"rows": 0, "by_partition": {}, "windows": set()}
    kinds = (sources_lib.KIND_ORDINAL, sources_lib.KIND_INTERVAL, sources_lib.KIND_POINT,
             sources_lib.KIND_ENRICHMENT)
    for record in observations:
        key = _short(record.population_key)
        partition = str(record.raw.get("partition") or data_lib.PARTITION_TRAIN)
        ordinal = record.ordinal_class
        facts = (record.family, record.parent, record.scale, partition,
                 tuple(record.construct.sites), record.kind, ordinal)
        known = populations.setdefault(key, facts)
        if known != facts and len(conflicts) < 20:
            names = ("family", "parent", "scale", "partition", "region", "kind", "ordinal_class")
            conflicts.append({"study": record.study, "assay": record.assay_key,
                              "identity": record.identity[:24],
                              "differs_in": [name for name, one, two in zip(names, known, facts)
                                             if one != two]})
        if record.kind not in kinds:
            valid = False
        elif record.kind == sources_lib.KIND_ORDINAL:
            valid = isinstance(ordinal, int) and not isinstance(ordinal, bool) and ordinal >= 0
        else:
            valid = ordinal is None
        if not valid and len(semantics) < 20:
            semantics.append({"study": record.study, "assay": record.assay_key,
                              "identity": record.identity[:24], "kind": record.kind,
                              "ordinal_class": ordinal})
        evidence = _short(record.observation_identity)
        if evidence in seen and len(conflicts) < 20:
            conflicts.append({"study": record.study, "assay": record.assay_key,
                              "identity": record.identity[:24], "differs_in": ["duplicate"]})
        seen.add(evidence)
        if record.source == HER2_SOURCE:
            window = record.construct.residues(sites)
            assigned = membership.assignment.get(window)
            her2["rows"] += 1
            her2["by_partition"][partition] = her2["by_partition"].get(partition, 0) + 1
            her2["windows"].add(window)
            if (assigned not in (split_lib.TRAIN, split_lib.DEVELOPMENT) or assigned != partition
                    or record.raw.get("window") != window) and len(her2_problems) < 20:
                her2_problems.append({"window": window, "assigned": assigned,
                                      "row_partition": partition,
                                      "row_window": record.raw.get("window")})
    require(not conflicts,
            f"typed observations disagree within one population: {conflicts[:5]}. The summary "
            "keeps the first and would train on it")
    require(not semantics,
            f"typed observations carry an unknown measurement kind or an ordinal class that does "
            f"not match their kind: {semantics[:5]}")
    require(not her2_problems,
            f"HER2 observations disagree with the restored split: {her2_problems[:5]}. A reserved "
            "or excluded window was materialised, or a row carries another partition than its "
            "window's")
    return {"validation_scope": SURVIVOR_SCOPE, "discarded_duplicate_evidence_validated": False,
            "observations": len(seen), "populations": len(populations),
            "her2_rows": her2["rows"], "her2_by_partition": dict(sorted(her2["by_partition"].items())),
            "her2_windows": len(her2["windows"])}, len(populations)


def validate_summaries(summaries, membership, *, populations):
    """The summaries that ``summarise_observations`` produced, against the same split."""
    sites = her2_core_edit_map()
    keys, problems, partitions = set(), [], {}
    for summary in summaries:
        keys.add(_short(summary.population_key))
        partitions[summary.partition] = partitions.get(summary.partition, 0) + 1
        if summary.partition not in data_lib.PARTITIONS and len(problems) < 20:
            problems.append(("unknown partition", summary.partition))
        if tuple(summary.region) != tuple(summary.construct.sites) and len(problems) < 20:
            problems.append(("region differs from the declared edit map", summary.identity[:24]))
        if summary.source == HER2_SOURCE:
            assigned = membership.assignment.get(summary.construct.residues(sites))
            if assigned != summary.partition and len(problems) < 20:
                problems.append(("HER2 summary partition differs from the restored split",
                                 summary.construct.residues(sites), assigned, summary.partition))
    require(not problems, f"loaded summaries are inconsistent: {problems[:5]}")
    require(len(keys) == len(summaries) == populations,
            f"{len(summaries)} summaries over {len(keys)} distinct populations, but the typed "
            f"observations hold {populations} populations")
    return {"summaries": len(summaries), "by_partition": dict(sorted(partitions.items()))}


#: The columns of one ``sources`` row of an inventory. Presence is required; the values may be
#: ``None`` (a source without an enrolled spec, such as the historical Petersen S1 alias, carries
#: ``None`` for enrolled/family/target/measurement_kind, and that is a legitimate value).
SOURCE_FIELDS = ("source", "enrolled", "family", "target", "measurement_kind", "rows_in_file",
                 "observations", "unique_constructs", "missing_outcome", "unalignable", "excluded",
                 "sha256")

#: The fields of a non-null ``display_library`` block that identify the library.
DISPLAY_FIELDS = (("rows", "int"), ("split", "str"), ("source", "str"), ("assay", "str"),
                  ("ordinal_classes", "int"), ("by_class", "dict"),
                  ("positives_only_would_see", "int"), ("rows_in_split", "int"),
                  ("refused_by_closure", "int"), ("positives_only", "bool"))

#: Every top-level field of a complete inventory, its container kind, and the nested fields the
#: comparison reads. A missing name, a wrong kind or a missing nested field makes an inventory
#: incomplete -- on the frozen side before any source is opened, on the derived side before a model
#: is built -- so an empty ``{}`` frozen record can never pass by having nothing to compare.
#: Nested fields are also exactly what is compared; nothing else of the block is.
INVENTORY_SHAPE = {
    "sources": ("list", ()),
    "summaries": ("int", ()),
    "training_summaries": ("int", ()),
    "bins_assigned": ("int", ()),
    "partitions": ("dict", ()),
    "totals": ("dict", (("observations", "int"), ("unique_constructs", "int"))),
    "closure": ("dict", (("kept", "int"), ("duplicates", "int"), ("unique_constructs", "int"),
                         ("constructs_in_multiple_sources", "int"))),
    "assay_vocabulary": ("dict", (("assays", "list"), ("counts", "dict"))),
    "condition_vocabulary": ("dict", (("assays", "list"), ("counts", "dict"))),
    "declared_heldouts": ("dict", (("partition_label", "str"), ("withheld_parents", "list"),
                                   ("withheld_studies", "list"), ("rows_moved", "dict"))),
    "measurement_kinds": ("dict", (("by_kind", "dict"), ("by_censoring", "dict"))),
    # ``reused_cache`` is the one operational field (whether a parsed cache was read) and the
    # prose (sampling, outcome) is left out on purpose. ``top_studies`` is a secondary listing and is
    # not compared either: the file SHA-256, the sampler policy (member, seed, limit, splits) and
    # the counts (rows, rows seen by split, studies) are what bind the selection, and all of them
    # are compared.
    "paired_oas": ("dict", (("sha256", "str"), ("member", "any"), ("seed", "int"),
                            ("limit_per_split", "int"), ("splits", "list"),
                            ("rows_in_file", "int"), ("rows_seen_by_split", "dict"),
                            ("excluded", "dict"), ("other_splits_present", "dict"),
                            ("studies", "int"))),
    "paired_oas_by_split": ("dict", ()),
    "region_geometry": ("dict", ()),
    "supersessions": ("dict", ()),
    "bin_thresholds": ("dict", ()),
    "parent_sequences": ("list", ()),
}

#: Descriptive prose of a supersession entry; its identity fields are compared, the reason is not.
SUPERSESSION_PROSE = ("reason",)
#: Fields the bin policy governs instead of an equality comparison (see :func:`bin_evidence`).
BIN_FIELDS = ("bin_thresholds", "bin_threshold_support", "bins_assigned")
#: Cap on the per-population differences LISTED in the evidence; the counts are never capped.
LISTED_BIN_DIFFERENCES = 200
MAX_PROBLEMS = 40


def _kind_ok(value, kind):
    if kind == "any":
        return True
    if kind == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, {"bool": bool, "str": str, "dict": dict, "list": list}[kind])


def _is_digest(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _check_fields(problems, label, name, block, fields):
    for field_name, kind in fields:
        if field_name not in block:
            problems.append(f"{label}: {name}.{field_name} is missing")
        elif not _kind_ok(block[field_name], kind):
            problems.append(f"{label}: {name}.{field_name} must be {kind}, got "
                            f"{type(block[field_name]).__name__}")


def inventory_structure(inventory, *, label, derived):
    """Why ``inventory`` is not complete enough to compare (an empty list when it is).

    Every :data:`INVENTORY_SHAPE` field must be present with its container kind; every source row
    must carry every :data:`SOURCE_FIELDS` column, a non-empty unique name and a SHA-256; and
    ``display_library`` must be PRESENT (``None`` is its explicit "not enrolled" value). Values are
    not required to be truthy: an empty ``region_geometry`` or a ``None`` ``target`` is a value.
    ``bin_threshold_support`` is required of a freshly derived inventory and optional in a
    historical one, which never recorded it.
    """
    if not isinstance(inventory, dict):
        return [f"{label} is not an object"]
    problems = []
    for name, (kind, nested) in INVENTORY_SHAPE.items():
        if name not in inventory:
            problems.append(f"{label}: {name} is missing")
        elif not _kind_ok(inventory[name], kind):
            problems.append(f"{label}: {name} must be {kind}, got {type(inventory[name]).__name__}")
        elif nested:
            _check_fields(problems, label, name, inventory[name], nested)
    rows = inventory.get("sources")
    if isinstance(rows, list):
        if not rows:
            problems.append(f"{label}: sources is empty")
        names = []
        for position, row in enumerate(rows):
            if not isinstance(row, dict):
                problems.append(f"{label}: sources[{position}] is not an object")
                continue
            absent = [name for name in SOURCE_FIELDS if name not in row]
            if absent:
                problems.append(f"{label}: sources[{position}] lacks {absent}")
                continue
            if not isinstance(row["source"], str) or not row["source"].strip():
                problems.append(f"{label}: sources[{position}] has no source name")
            names.append(row["source"])
            for name in ("observations", "unique_constructs"):
                if not _kind_ok(row[name], "int"):
                    problems.append(f"{label}: sources[{position}].{name} must be an integer")
            if not _is_digest(row["sha256"]):
                problems.append(f"{label}: source {row['source']!r} has no SHA-256")
        repeated = sorted({name for name in names if names.count(name) > 1}, key=str)
        if repeated:
            problems.append(f"{label}: source name(s) {repeated} repeat")
    supersessions = inventory.get("supersessions")
    if isinstance(supersessions, dict):
        for name, entry in supersessions.items():
            if not isinstance(entry, dict):
                problems.append(f"{label}: supersessions.{name} is not an object")
    if "display_library" not in inventory:
        problems.append(f"{label}: display_library is missing (None is the explicit 'not "
                        "enrolled' value)")
    elif inventory["display_library"] is not None:
        library = inventory["display_library"]
        if not isinstance(library, dict):
            problems.append(f"{label}: display_library must be an object or None")
        else:
            _check_fields(problems, label, "display_library", library, DISPLAY_FIELDS)
    if "bin_threshold_support" in inventory:
        if not isinstance(inventory["bin_threshold_support"], dict):
            problems.append(f"{label}: bin_threshold_support must be dict")
    elif derived:
        problems.append(f"{label}: bin_threshold_support is missing")
    return problems


def scientific_projection(inventory):
    """The compared content of a structurally complete inventory (see :data:`INVENTORY_SHAPE`).

    Source rows are keyed by name, supersessions lose their prose, parent sequences are a sorted
    name list, and the bin fields are absent: the bin policy governs those.
    """
    projection = {"sources": {row["source"]: {name: row[name] for name in SOURCE_FIELDS
                                              if name != "source"}
                              for row in inventory["sources"]}}
    for name, (_, nested) in INVENTORY_SHAPE.items():
        if name in projection or name in BIN_FIELDS:
            continue
        if nested:
            projection[name] = {field_name: inventory[name][field_name]
                                for field_name, _ in nested}
        else:
            projection[name] = inventory[name]
    projection["supersessions"] = {
        name: {key: value for key, value in entry.items() if key not in SUPERSESSION_PROSE}
        for name, entry in inventory["supersessions"].items()}
    projection["parent_sequences"] = sorted(str(name) for name in inventory["parent_sequences"])
    library = inventory["display_library"]
    projection["display_library"] = (None if library is None else
                                     {name: library[name] for name, _ in DISPLAY_FIELDS})
    return projection


def _brief(value):
    text = json.dumps(value, sort_keys=True, default=str)
    return text if len(text) <= 90 else text[:87] + "..."


def _differences(path, new, frozen, problems):
    if isinstance(new, dict) and isinstance(frozen, dict):
        for key in sorted(set(new) | set(frozen), key=str):
            if key not in frozen:
                problems.append(f"{path}.{key} was produced now and is not in the frozen inventory")
            elif key not in new:
                problems.append(f"{path}.{key} is in the frozen inventory and was not produced now")
            else:
                _differences(f"{path}.{key}", new[key], frozen[key], problems)
    elif new != frozen:
        problems.append(f"{path}: {_brief(new)} now against the frozen {_brief(frozen)}")


def _threshold_digest(document):
    return paths.digest_document(document)


def _bin_summary(thresholds, support, assigned):
    return {"populations": len(thresholds),
            "fitted_populations": sum(1 for value in thresholds.values() if value is not None),
            "bins_assigned": int(assigned),
            "thresholds_sha256": _threshold_digest(thresholds),
            "support_recorded": support is not None,
            "support_sha256": None if support is None else _threshold_digest(support)}


def bin_evidence(new, frozen, policy):
    """The declared bin policy, what it froze, and every difference from the historical bins.

    ``current_exact_training``: the CURRENT thresholds (train rows' exact points only) are the ones
    every arm uses. The historical thresholds, support and assigned count are kept as evidence
    beside them, with the populations that changed listed (to a cap; the counts are exact) -- the
    differences are reported, not ignored, and ``historical_replay`` is ``False``.
    """
    require(policy in BIN_POLICIES, f"unsupported bin policy {policy!r}; one of {BIN_POLICIES}")
    old_cuts, new_cuts = frozen["bin_thresholds"], new["bin_thresholds"]
    old_support, new_support = frozen.get("bin_threshold_support"), new["bin_threshold_support"]
    historical = _bin_summary(old_cuts, old_support, frozen["bins_assigned"])
    current = _bin_summary(new_cuts, new_support, new["bins_assigned"])
    changed, only_historical, only_current = [], [], []
    for key in sorted(set(old_cuts) | set(new_cuts)):
        if key not in new_cuts:
            only_historical.append(key)
        elif key not in old_cuts:
            only_current.append(key)
        elif old_cuts[key] != new_cuts[key]:
            changed.append({"population": key, "historical": old_cuts[key],
                            "current": new_cuts[key]})
    return {
        "policy": policy, "historical_replay": False,
        "claim": ("every arm uses thresholds re-fitted by the current source code on the train "
                  "rows' exact points, over the frozen split and the same source bytes; the "
                  "production flight fitted on row.value, the mean of a summary's uncensored "
                  "replicates, over summaries that also held censored replicates, which the "
                  "exact-only rule now excludes, so the two differ and this is not a replay of "
                  "the historical bins"),
        "historical": historical, "current": current,
        "differences": {
            "populations_changed": len(changed),
            "populations_only_historical": only_historical,
            "populations_only_current": only_current,
            "bins_assigned": {"historical": historical["bins_assigned"],
                              "current": current["bins_assigned"],
                              "difference": current["bins_assigned"]
                              - historical["bins_assigned"]},
            "listed": changed[:LISTED_BIN_DIFFERENCES],
            "unlisted": max(0, len(changed) - LISTED_BIN_DIFFERENCES)},
        "frozen_identity": {"policy": policy, "thresholds_sha256": current["thresholds_sha256"],
                            "support_sha256": current["support_sha256"],
                            "bins_assigned": current["bins_assigned"]}}


def _json_text(node):
    return json.dumps(node, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _population_text(key):
    """Canonical JSON text of a population key; refuses anything but a four-part JSON-able key."""
    require(isinstance(key, (tuple, list)) and len(key) == 4,
            f"a bin assignment is keyed by (identity, study, assay, target), got {key!r}")
    try:
        return _json_text(plain(key))
    except (TypeError, ValueError) as error:
        raise ValueError(f"the bin assignment key {key!r} is not canonical JSON: {error}") from None


def fingerprint_bin_assignments(assignments, *, summaries, expected_count):
    """The digest of the ACTUAL population -> bin mapping, after checking it against what was loaded.

    ``assignments`` is :meth:`FrozenInputServices.bin_assignments`; ``summaries`` the loaded
    :class:`ConstructSummary` records; ``expected_count`` the derived inventory's ``bins_assigned``.
    Refused, with the first offenders named: a key that is not a four-part canonical-JSON population
    key, two keys with one canonical form, a key with no loaded summary (an unknown population), a
    bin that is not ``low``/``mid``/``high``, and a count that differs from the inventory's.

    The digest is SHA-256 over :data:`ASSIGNMENT_SCHEMA` and then one line per assignment,
    ``[<population key as a JSON array>,"<bin>"]``, in the order of those key texts (a total order on
    the full population key, so the digest does not depend on dict insertion order). Two mappings
    with the same keys and the same bin COUNTS but different labels therefore differ. ``by_bin`` and
    ``by_partition`` are informational counts beside it.
    """
    require(isinstance(assignments, dict), "the bin assignments are not a mapping")
    partitions = {_population_text(summary.population_key): summary.partition
                  for summary in summaries}
    records, unknown, illegitimate = {}, [], []
    for key, name in assignments.items():
        text = _population_text(key)
        require(text not in records,
                f"two bin assignment keys have one canonical form: {text[:80]}")
        records[text] = name
        if text not in partitions:
            unknown.append(text[:80])
        if name not in ASSIGNED_BINS:
            illegitimate.append((text[:80], name))
    require(not unknown,
            f"{len(unknown)} bin assignment(s) belong to no loaded summary, e.g. {unknown[:3]}")
    require(not illegitimate,
            f"{len(illegitimate)} bin assignment(s) name a bin outside {ASSIGNED_BINS}, "
            f"e.g. {illegitimate[:3]}")
    require(len(records) == int(expected_count),
            f"{len(records)} bin assignments, but the derived inventory reports {expected_count}")
    digest = hashlib.sha256()
    digest.update((ASSIGNMENT_SCHEMA + "\n").encode("ascii"))
    by_bin, by_partition = {}, {}
    for text in sorted(records):
        digest.update(f"[{text},{_json_text(records[text])}]\n".encode("ascii"))
        by_bin[records[text]] = by_bin.get(records[text], 0) + 1
        by_partition[partitions[text]] = by_partition.get(partitions[text], 0) + 1
    return {"schema": ASSIGNMENT_SCHEMA, "count": len(records), "sha256": digest.hexdigest(),
            "by_bin": dict(sorted(by_bin.items())),
            "by_partition": dict(sorted(by_partition.items())),
            "scope": ("every loaded population that was assigned a bin, counted by partition above; "
                      "the cuts are fitted on the train rows' exact points only and no reserved "
                      "HER2 window is loaded")}


def bind_current_bins(evidence, *, inventory, assignments):
    """``(evidence, record)``: the assignment fingerprint bound into the bin identity, and the record.

    ``record`` is the complete current-bins document: the policy, the FULL current thresholds
    (``None`` entries included) and the FULL current support, and the assignment fingerprint. Its
    thresholds and support must hash to the digests ``evidence`` already carries. The returned
    evidence adds the fingerprint to ``current`` and, to ``frozen_identity``, the assignment schema,
    count and digest and the record's content digest, so a change to any cut, to its support or to
    any population's bin changes the identity.
    """
    inventory = plain(inventory)
    cuts = json.loads(json.dumps(inventory["bin_thresholds"], allow_nan=False))
    support = json.loads(json.dumps(inventory["bin_threshold_support"], allow_nan=False))
    current = evidence["current"]
    require(_threshold_digest(cuts) == current["thresholds_sha256"]
            and _threshold_digest(support) == current["support_sha256"],
            "the current thresholds or support differ from the ones the bin evidence digests")
    require(assignments["count"] == current["bins_assigned"],
            f"{assignments['count']} assignments, but the bin evidence reports "
            f"{current['bins_assigned']}")
    record = {"schema_version": CURRENT_BINS_SCHEMA, "policy": evidence["policy"],
              "historical_replay": False,
              "scope": ("the bins every arm uses: thresholds fitted by the current source code on the "
                        "train rows' exact points, their support, and the fingerprint of the "
                        "population-to-bin mapping they produced (the mapping itself is not saved)"),
              "bin_thresholds": cuts, "bin_threshold_support": support,
              "assignments": dict(assignments)}
    content = paths.digest_document(record)
    bound = copy.deepcopy(evidence)
    bound["current"]["assignments"] = dict(assignments)
    bound["current"]["record_sha256"] = content
    bound["frozen_identity"].update(
        assignments_schema=assignments["schema"], assignments_count=assignments["count"],
        assignments_sha256=assignments["sha256"], current_bins_sha256=content)
    return bound, record


def write_current_bins(loaded, directory):
    """Write ``current_bins.json`` into the run root; record its run-relative path and file SHA-256.

    Called by the runner right after the inputs are loaded and before any model is built. The file is
    the canonical JSON of ``loaded.current_bins``, so its file SHA-256 must equal the content digest
    the identity already carries.
    """
    require(loaded.current_bins, "there is no current-bins record to write")
    target = Path(directory) / CURRENT_BINS_FILE
    require(not target.exists(), f"{target} already exists in what should be a fresh run root")
    digest = paths.write_json(target, loaded.current_bins)
    expected = loaded.bin_identity["current_bins_sha256"]
    require(digest == expected,
            f"{target} hashes to {digest}, not to the content digest {expected} in the identity")
    loaded.bins_file = {"path": CURRENT_BINS_FILE, "file_sha256": digest, "content_sha256": expected}
    return dict(loaded.bins_file)


def compare_inventory(new, frozen, *, bin_policy):
    """``(problems, bin evidence)``: the inventory just derived against the frozen one.

    Both inventories must first be complete (:func:`inventory_structure`); either one failing is the
    whole answer and the evidence is ``None``. Then the scientific projections are compared exactly,
    field by field, with every difference named. The bin fields are not compared for equality: the
    declared ``bin_policy`` governs them and :func:`bin_evidence` reports how the current bins differ
    from the historical ones. An empty problem list means the inputs are the frozen ones under that
    policy.
    """
    require(bin_policy in BIN_POLICIES,
            f"unsupported bin policy {bin_policy!r}; one of {BIN_POLICIES}. A bin difference is "
            "only ever accepted under a declared policy")
    new = json.loads(json.dumps(plain(new), sort_keys=True, default=str))
    frozen = json.loads(json.dumps(plain(frozen), sort_keys=True, default=str))
    problems = (inventory_structure(frozen, label="the frozen inventory", derived=False)
                + inventory_structure(new, label="the derived inventory", derived=True))
    if problems:
        return problems[:MAX_PROBLEMS], None
    problems = []
    now, then = scientific_projection(new), scientific_projection(frozen)
    for name in now:
        _differences(name, now[name], then[name], problems)
    if len(problems) > MAX_PROBLEMS:
        problems = problems[:MAX_PROBLEMS] + [f"... and {len(problems) - MAX_PROBLEMS} more"]
    return problems, bin_evidence(new, frozen, bin_policy)


# ---------------------------------------------------------------------------
# the whole stage
# ---------------------------------------------------------------------------

def load_inputs(pilot, context, *, policy, frozen_dir, device):
    """Restore, load, validate. Returns :class:`LoadedInputs`; builds no model, trains nothing.

    ``policy`` carries the pins (``expected_split_sha256``, ``expected_sequence_sha256``), normally
    ``spec.training_phase(base_config)``. ``context`` is a :class:`~.roots.PilotContext`, already
    over a freshly reserved run root.
    """
    frozen = read_frozen_records(frozen_dir)
    # Before the vocabulary is installed, before a source is opened: a frozen inventory that cannot
    # be compared field by field (or is empty) is not an inventory to load against.
    incomplete = inventory_structure(frozen["inventory"]["document"],
                                     label="the frozen inventory", derived=False)
    require(not incomplete,
            "the frozen source inventory is incomplete and nothing was loaded: "
            + "; ".join(incomplete[:8]))
    require(pilot.bin_policy in BIN_POLICIES,
            f"the pilot's bin policy {pilot.bin_policy!r} is not one of {BIN_POLICIES}")
    install_frozen_vocabulary(context, frozen)
    services = FrozenInputServices(context, device=device,
                                   source_cache=context.roots.source_cache)
    scaffold = services.scaffold()
    sequence_file = services.source_paths().get(HER2_SOURCE)
    require(sequence_file is not None, f"the config declares no {HER2_SOURCE} source path")
    candidates = membership_lib.candidate_windows(sequence_file, scaffold=scaffold)
    membership = membership_lib.restore_membership(
        frozen["split"]["document"], record_sha256=frozen["split"]["document_sha256"],
        candidates=candidates["windows"], expected_split_sha256=policy["expected_split_sha256"],
        expected_sequence_sha256=policy["expected_sequence_sha256"],
        split_settings=context.config["split"], expected_counts=pilot.expected_counts)
    closure = membership_lib.build_closure(membership, scaffold=scaffold)
    services.adopt(membership, closure)
    identities = verify_source_identities(context)
    inventory = services.build_sources(retain_observations=True)
    observed, populations = validate_observations(services.observations(), membership)
    summaries = validate_summaries(services.summaries(), membership, populations=populations)
    problems, bins = compare_inventory(inventory, frozen["inventory"]["document"],
                                       bin_policy=pilot.bin_policy)
    require(not problems, "the loaded sources are not the frozen ones: " + "; ".join(problems))
    assignments = fingerprint_bin_assignments(
        services.bin_assignments(), summaries=services.summaries(),
        expected_count=bins["current"]["bins_assigned"])
    bins, current_bins = bind_current_bins(bins, inventory=inventory, assignments=assignments)
    observed["duplicate_observations_dropped_by_closure"] = {
        "current": int(plain(inventory)["closure"]["duplicates"]),
        "frozen": int(frozen["inventory"]["document"]["closure"]["duplicates"])}
    require(sorted(services.condition_spec.assays)
            == sorted(frozen["vocabulary"]["document"]["assays"]),
            "the condition vocabulary in use is not the frozen one")
    pools = services.training_pools()
    pool = tuple(pools["summaries"][data_lib.FAMILY_HER2])
    require(pool, "the HER2 training pool is empty")
    sites = her2_core_edit_map()
    train = membership.train
    keys = frozenset(summary.population_key for summary in services.summaries()
                     if summary.source == HER2_SOURCE
                     and summary.construct.residues(sites) in train)
    require(keys, "no HER2 summary falls in a train window of the restored split")
    outside = [summary for summary in pool if summary.population_key not in keys]
    require(not outside,
            f"{len(outside)} HER2 training-pool summaries are not in the restored training "
            "population")
    panel = services.development_panel_outcomes()
    return LoadedInputs(
        services=services, membership=membership, closure=closure,
        frozen={name: {key: value for key, value in record.items() if key != "document"}
                for name, record in frozen.items()},
        identities=identities,
        validation={"observations": observed, "summaries": summaries,
                    "inventory_compared": sorted(scientific_projection(plain(inventory))),
                    "inventory_governed_by_bin_policy": list(BIN_FIELDS),
                    "inventory_not_compared": {
                        "prose": ["note", "ordering", "partition_rule", "threshold_rule",
                                  "supersessions.*.reason", "paired_oas.sampling",
                                  "paired_oas.outcome"],
                        "operational": ["paired_oas.reused_cache"],
                        "detail_listings": ["paired_oas.top_studies", "closure.duplicate_detail",
                                            "closure.shared_examples"]},
                    "candidates": {key: value for key, value in candidates.items()
                                   if key != "windows"},
                    "pool_counts": dict(pools["counts"])},
        training_keys=keys, pool=pool, panel=panel,
        narrowed_routes={"enabled": list(context.config["routes"]["enabled"])}, bins=bins,
        current_bins=current_bins)
