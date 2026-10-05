"""Config, context, the external source freeze, and the two frozen revisions.

Two things this campaign needs that the inherited freeze machinery does not provide,
and one it does.

**It does provide the repository half.** ``her2_replay_spec.source_closure`` is an
AST walk that states outright that *function-level imports are included because the
walk is syntactic*, and ``her2_nf_spec.source_snapshot`` copies the closure's bytes
into the run root, hashes them and re-verifies both the archive and the worktree at
every stage. So a lazy import inside an eval module is already frozen, and a
mid-flight edit already fails. That machinery is called unchanged; nothing here adds
an ``--allow-dirty`` path.

**It does not cover bytes outside the repository.** The pinned AbLang2 package is
imported from the artifact volume and is production code. ``source_closure`` records
externals by *version*, and an upstream package loaded from a directory has no
distribution metadata, so its version record is ``unresolved`` -- the bytes would be
unfrozen. :func:`external_source_digests` reads them from the acquisition manifest,
:func:`verify_external_sources` re-hashes them at every stage, and both are folded
into the aggregate identity. No digest is hard-coded in this repository.

**Two frozen identities, not one.** ``rev01`` is a declared **development**
revision: its numbers may be read for curves and selection. If a round finds a real
training-path bug it is superseded by ``rev02`` under a new provenance identity, with
the old root retained read-only and **never resumed**. The **confirmation** revision
is frozen last, after architecture, objectives, mix, decoding and selection are all
frozen, and nothing in its closure is edited afterwards. This is explicit transfer
under a new identity, not a silent resume; :func:`require_revision_discipline`
enforces the distinction.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from . import her2_nf_spec as nf_spec
from . import her2_replay_spec as replay_spec
from . import her2_support_paths as paths
from .her2_routes_conditions import ConditionSpec
from .her2_runtime import canonical_json, require

ROUTES_SCHEMA = "her2-generative-routes/1"

REVISION_DEVELOPMENT = "development"
REVISION_CONFIRMATION = "confirmation"
REVISION_KINDS = (REVISION_DEVELOPMENT, REVISION_CONFIRMATION)

#: Every stage that SELECTS something the confirmation freezes. ``confirmation_prerequisites``
#: and ``STRICT_PREREQUISITES`` already treated all six as selections; the revision guard
#: refused only the first three, so priors, post-training and decoding could be re-run under
#: the confirmation identity after ``frozen_recipe.json`` had bound their outcome (HR-04).
SELECTION_STAGES = ("bakeoff", "route_gate", "controls", "priors", "posttrain", "decoding")

#: The acquisition manifests merged into the external closure, in order. The DPLM manifest
#: is optional: a config that does not name it (the current three-route config) is unchanged.
MANIFEST_NAMES = ("acquisition_manifest", "primary_acquisition_manifest",
                  "primary_review_manifest", "primary_corrections_manifest",
                  "dplm_acquisition_manifest")

FREEZE_MARKER = nf_spec.FREEZE_MARKER
EXTERNAL_MARKER = "external_sources.json"
REVISION_MARKER = "revision.json"
SPLIT_FREEZE = "hard_split_freeze.json"
PROFILE_JSON = "gpu_profile.json"
FORECAST_JSON = "runtime_forecast.json"


def load_config(path):
    """Read the flight config and refuse any unresolved placeholder."""
    path = Path(path)
    require(path.is_file(), f"The routes config {path} is missing")
    document = json.loads(path.read_text(encoding="utf-8"))
    require(document.get("schema_version") == ROUTES_SCHEMA,
            f"{path} is not a {ROUTES_SCHEMA} config (it declares "
            f"{document.get('schema_version')!r})")
    holes = replay_spec.placeholders(document)
    require(not holes,
            f"{path} still carries unresolved placeholders at {holes}. Every scientific setting is "
            "complete before the freeze; a stage running against one would record a null where a "
            "declared value belongs.")
    return document


@dataclass(frozen=True)
class RoutesContext:
    """Repository, run root, artifact root and the resolved config."""

    repository_root: Path
    run_root: Path
    artifact_root: Path
    config: dict
    config_path: Path
    revision: str = "rev01"

    @property
    def campaign_id(self):
        return str(self.config["campaign_id"])

    @property
    def revision_root(self):
        return self.artifact_root / "runs" / self.revision

    def path(self, *parts):
        return self.run_root.joinpath(*[str(part) for part in parts])

    def artifact(self, logical):
        return paths.resolve_under(self.artifact_root, str(logical))

    def repository(self, logical):
        return paths.resolve_under(self.repository_root, str(logical))

    def condition_spec(self):
        block = self.config["conditions"]
        return ConditionSpec(assays=tuple(block["assays"]),
                             bin_dropout=float(block["bin_dropout"]),
                             assay_dropout=float(block["assay_dropout"]))

    def ensure(self):
        self.run_root.mkdir(parents=True, exist_ok=True)
        return self


def resolve_context(repository_root, *, config_path, run_root=None, artifact_root=None,
                    revision=None, ensure=True):
    """Bind the context, resolving the artifact root by hash probe where possible.

    ``ensure=False`` is the read-only form: it binds the same paths and creates nothing,
    which is what a status page needs -- asking "what ran?" must not create the run root it
    is asking about.
    """
    repository_root = Path(repository_root)
    config = load_config(config_path)
    declared = artifact_root or config["artifacts"]["root"]
    artifact = Path(declared)
    require(artifact.is_absolute(),
            f"The artifact root {declared!r} must be absolute; a relative one resolves against "
            "the working directory, which differs between the launcher and the supervisor")
    revision = str(revision or config["revisions"]["current"])
    root = (Path(run_root) if run_root is not None
            else artifact / "runs" / revision / config["run_root_leaf"])
    context = RoutesContext(repository_root=repository_root, run_root=root,
                            artifact_root=artifact, config=config,
                            config_path=Path(config_path), revision=revision)
    protected = {"repository": repository_root,
                 "raw_her2": repository_root / config["roots"]["her2_raw"],
                 "paired_oas": repository_root / config["roots"]["paired_oas"]}
    for name, logical in dict(config["roots"].get("historical") or {}).items():
        protected[f"historical_{name}"] = repository_root / logical
    paths.require_disjoint_run_root(root, protected)
    return context.ensure() if ensure else context


# ---------------------------------------------------------------------------
# the external source freeze
# ---------------------------------------------------------------------------

def read_acquisition_manifest(context, *, logical=None):
    """Load and shape-check the acquisition manifest. The schema is validated as observed."""
    target = context.artifact(logical or context.config["artifacts"]["acquisition_manifest"])
    require(target.is_file(),
            f"{target} is absent. The acquire stage writes it, and every external digest in this "
            "campaign comes from it rather than from a constant in the repository.")
    document = paths.read_json(target)
    require(str(document.get("schema", "")).startswith("her2-routes-acquisition/"),
            f"{target} declares schema {document.get('schema')!r}, which this loader does not "
            "recognise. The schema is validated as observed, not guessed.")
    for key in ("files", "extracted_weights", "pins", "retrieved_utc", "outcomes_opened"):
        require(key in document, f"{target} is missing the {key!r} block")
    require(isinstance(document["files"], list) and document["files"],
            f"{target} records no files")
    for entry in list(document["files"]) + list(document["extracted_weights"]):
        for key in ("path", "sha256", "bytes"):
            require(key in entry, f"{target}: a manifest entry is missing {key!r}: {entry}")
    failures = list(document.get("failures") or [])
    require(not failures,
            f"{target} records {len(failures)} acquisition failure(s): {failures[:5]}. A partial "
            "acquisition is not a frozen input set.")
    return document, {"logical": str(logical or context.config['artifacts']
                                     ['acquisition_manifest']),
                      "sha256": paths.sha256_file(target),
                      "files": len(document["files"]),
                      "extracted_weights": len(document["extracted_weights"]),
                      "pins": dict(document["pins"]),
                      "retrieved_utc": document["retrieved_utc"],
                      "outcomes_opened": bool(document["outcomes_opened"])}


def read_all_manifests(context):
    """Every acquisition manifest, merged into one ``{normalised path: sha256}`` table.

    Three manifests exist and they were retrieved separately on purpose: the main
    acquisition (110 files plus two extracted weight sets), the first primary
    acquisition (four Petersen supplements) and the primary review acquisition (17
    pinned files, whose paths are **absolute** on the artifact volume). Merging them
    into one file would leave none of them verifiable alone; reading only the first,
    which is what the earlier closure did, leaves the primary sources outside the
    frozen identity entirely.

    Paths are normalised to logical names under the artifact root, because one of the
    three records absolute paths and a table keyed on both forms silently holds two
    entries for one file.
    """
    block = context.config["artifacts"]
    root = context.artifact_root
    table, per_manifest = {}, {}
    for name in MANIFEST_NAMES:
        logical = block.get(name)
        if not logical:
            continue
        target = context.artifact(logical)
        if not target.is_file():
            per_manifest[name] = {"present": False, "entries": 0, "logical": str(logical)}
            continue
        document = paths.read_json(target)
        entries = list(document.get("files") or []) + list(
            document.get("extracted_weights") or [])
        count = 0
        for entry in entries:
            path, digest = entry.get("path"), entry.get("sha256")
            if not path or not digest:
                continue
            table[normalise_artifact_path(path, root)] = str(digest)
            count += 1
        per_manifest[name] = {"present": True, "entries": count, "logical": str(logical),
                              "sha256": paths.sha256_file(target),
                              "retrieved_utc": document.get("retrieved_utc"),
                              "outcomes_opened": bool(document.get("outcomes_opened"))}
    require(table,
            "No acquisition manifest described a single file. Every external digest in this "
            "campaign comes from these manifests rather than from a constant in the repository.")
    return table, per_manifest


def normalise_artifact_path(path, root):
    """A logical name under the artifact root, whichever form the manifest used."""
    text = str(path).replace("\\", "/")
    prefix = str(root).replace("\\", "/").rstrip("/") + "/"
    if text.lower().startswith(prefix.lower()):
        return text[len(prefix):]
    return text.lstrip("/")


def external_source_digests(context, manifest, *, logical_paths, table=None):
    """``{logical: sha256}`` for the external production sources, from the manifests.

    Every digest comes from an acquisition manifest and is then re-hashed against the
    file on disk. A path no manifest describes is refused: an external source with no
    recorded digest is exactly the gap this table exists to close, and accepting it
    would make the freeze silently partial.
    """
    recorded = dict(table) if table is not None else read_all_manifests(context)[0]
    for entry in list(manifest.get("files") or []) + list(
            manifest.get("extracted_weights") or []):
        if entry.get("path") and entry.get("sha256"):
            recorded.setdefault(
                normalise_artifact_path(entry["path"], context.artifact_root),
                str(entry["sha256"]))
    digests, problems = {}, []
    for logical in sorted(set(str(value) for value in logical_paths)):
        key = normalise_artifact_path(logical, context.artifact_root)
        target = context.artifact(key)
        expected = recorded.get(key)
        if expected is None:
            problems.append(f"{logical}: not described by any acquisition manifest")
            continue
        if not target.is_file():
            problems.append(f"{logical}: described by a manifest but absent on disk")
            continue
        observed = paths.sha256_file(target)
        if observed != expected:
            problems.append(f"{logical}: hashes {observed}, the manifest recorded {expected}")
            continue
        digests[key] = observed
    require(not problems,
            "The external source table cannot be built: " + "; ".join(problems)
            + ". A missing or mismatched digest BLOCKS the freeze; it does not degrade to a "
            "warning, and nothing in this repository hard-codes one.")
    return digests


def external_closure(context, *, table, extra=()):
    """Every external logical path the frozen identity must cover. Derived, not listed.

    Derived from the manifests themselves rather than from a hand-written list of
    three paths. The earlier list named ``ablang2/__init__.py``, which no manifest
    describes -- the pinned package is at ``upstream/ablang2/ablang2/__init__.py`` --
    so the acquisition verification failed against its own note that the closure was
    three paths.

    What is selected: every pinned native package module, the native weights and
    hyperparameters, the locally stored p-IgGen release, and every primary source
    file. Bulk data that the loaders hash themselves is covered by the per-source
    digests in the inventory and is named in ``covered_elsewhere``.
    """
    wanted, covered = [], []
    for logical in sorted(table):
        text = str(logical)
        if text.startswith("upstream/") and text.endswith(".py"):
            wanted.append(text)
        elif text.startswith("models/"):
            wanted.append(text)
        elif text.startswith("data/primary/"):
            wanted.append(text)
        elif text.startswith("data/raw/her2_functional"):
            wanted.append(text)
        else:
            covered.append(text)
    for logical in extra:
        key = normalise_artifact_path(logical, context.artifact_root)
        if key not in wanted:
            wanted.append(key)
    return {"logical_paths": sorted(set(wanted)),
            "covered_elsewhere": len(covered),
            "selection": ("pinned native package modules, native weights and hyperparameters, the "
                          "locally stored generator release, and every primary source file. The "
                          "bulk curated releases are hashed by their own loaders and recorded in "
                          "the source inventory."),
            "derived_from": "the acquisition manifests, not a hand-written path list"}


def dplm_variants(config):
    """The DPLM variants the enabled routes need. Empty for the current three-route config."""
    routes = dict(config.get("routes") or {})
    names = []
    for route in list(routes.get("enabled") or ()):
        variant = dict(routes.get(route) or {}).get("variant")
        if variant and variant not in names:
            names.append(str(variant))
    return names


def require_dplm_binding(context, *, table, per_manifest, closure):
    """A flight that enables a DPLM route must have its bytes in the frozen external closure.

    Adding the weight paths to a config is not binding: the closure is derived from the
    acquisition manifests, so a DPLM route whose manifest is absent would freeze an identity
    that never mentions its weights. Refused here, and each file's manifest digest must equal
    the digest the pins file declares (where the pin declares a sha256).
    """
    variants = dplm_variants(context.config)
    if not variants:
        return {"dplm_variants": [], "bound": False}
    block = per_manifest.get("dplm_acquisition_manifest") or {}
    require(block.get("present"),
            "This config enables a DPLM route but the DPLM acquisition manifest "
            f"({context.config['artifacts'].get('dplm_acquisition_manifest')!r}) is absent. Run "
            "scripts/her2_routes_verify_dplm.py --write-manifest after placing the pinned files; "
            "adding paths to a config does not bind their bytes.")
    from . import her2_routes_dplm_pins as pins_lib
    pins = pins_lib.load_pins(context.repository(context.config["dplm"]["pins"]))
    bound, problems = {}, []
    wanted = set(closure["logical_paths"])
    for variant in variants:
        entry = pins_lib.variant_entry(pins, variant)
        for name in pins_lib.FILES:
            logical = f"{entry['logical_dir']}/{name}"
            if logical not in wanted:
                problems.append(f"{logical}: not in the external closure")
                continue
            pinned = entry["files"][name].get("sha256")
            if pinned is not None and table.get(logical) != pinned:
                problems.append(f"{logical}: manifest records {table.get(logical)}, pins say "
                                f"{pinned}")
        bound[variant] = {"repo": entry["repo"], "revision": entry["revision"]}
    require(not problems, "DPLM binding failed: " + "; ".join(problems))
    return {"dplm_variants": variants, "bound": True, "pins_sha256": pins_lib.pins_digest(pins),
            "variants": bound}


def write_external_marker(context, *, manifest_record, digests, native_report=None):
    aggregate = paths.sha256_text("\n".join(f"{name} {digests[name]}" for name in sorted(digests)))
    document = {"schema_version": ROUTES_SCHEMA, "record_kind": "external_sources",
                "acquisition_manifest": dict(manifest_record),
                "files": dict(sorted(digests.items())), "file_count": len(digests),
                "external_sha256": aggregate,
                "native": dict(native_report or {}),
                "why": ("her2_replay_spec.source_closure walks repository-local modules and records "
                        "externals by version. The pinned AbLang2 package is imported from the "
                        "artifact volume and has no distribution metadata, so its bytes would be "
                        "outside the frozen identity without this table."),
                "frozen_at": paths.utc_now()}
    paths.write_json(context.path(EXTERNAL_MARKER), document)
    return document


def verify_external_sources(context, *, label="stage"):
    """Re-hash every external source against the marker. A change stops the stage."""
    target = context.path(EXTERNAL_MARKER)
    require(target.is_file(),
            f"{target} is absent: every stage after the external freeze runs against a recorded "
            "external source table, and there is none.")
    marker = paths.read_json(target)
    changed, missing = [], []
    for logical, expected in sorted(dict(marker["files"]).items()):
        path = context.artifact(logical)
        if not path.is_file():
            missing.append(logical)
            continue
        observed = paths.sha256_file(path)
        if observed != expected:
            changed.append({"file": logical, "frozen_sha256": expected,
                            "observed_sha256": observed})
    aggregate = paths.sha256_text("\n".join(f"{name} {marker['files'][name]}"
                                           for name in sorted(marker["files"])))
    require(aggregate == marker["external_sha256"],
            f"{EXTERNAL_MARKER}: the recorded per-file digests aggregate to {aggregate} and the "
            f"marker claims {marker['external_sha256']}")
    require(not missing and not changed,
            f"{label}: the frozen external sources changed. Missing {missing}; changed "
            f"{[entry['file'] for entry in changed]}. The recorded numbers are attributed to those "
            "bytes, so a stage does not run against different ones.")
    return {"files": len(marker["files"]), "external_sha256": aggregate, "verified": True}


# ---------------------------------------------------------------------------
# the combined snapshot
# ---------------------------------------------------------------------------

def _nf_context(context):
    """An adapter so the inherited snapshot machinery is called unchanged.

    ``her2_nf_spec.source_snapshot`` needs ``config["source"]["entry_points"]``,
    ``config["config_path"]`` and ``config["campaign_id"]``; the routes config
    carries all three, so no field is synthesised here.
    """
    for key in ("source", "config_path", "campaign_id"):
        require(key in context.config,
                f"The routes config is missing {key!r}, which the inherited source snapshot needs")
    paths.require_logical_name(str(context.config["config_path"]))
    return nf_spec.FlightContext(repository_root=context.repository_root,
                                 run_root=context.run_root, config=context.config,
                                 config_path=context.config_path)


def source_snapshot(context):
    """The repository closure snapshot, through the inherited, untouched machinery."""
    return nf_spec.source_snapshot(_nf_context(context))


def verify_source_snapshot(context, *, label="stage", check_runtime_closure=None):
    """Re-hash the frozen worktree and archive against the snapshot marker.

    ``check_runtime_closure=None`` keeps the inherited default (checked outside pytest). A
    read-only consumer that runs from a DIFFERENT checkout than the frozen one -- the common
    evaluation -- passes ``False``: the closure check refuses every unfrozen production module
    the importing process has loaded, which is correct for a stage of the flight and wrong for
    an auditor, while the byte-for-byte re-hash of the frozen files still runs.
    """
    return nf_spec.verify_source_snapshot(_nf_context(context), label=label,
                                          check_runtime_closure=check_runtime_closure)


def combined_identity(context):
    """One digest over the repository snapshot and the external source table.

    The campaign's identity is this pair. Recording only the repository half would
    attribute numbers produced by a particular upstream package to a snapshot that
    does not mention it.
    """
    repository = paths.read_json(context.path(nf_spec.FREEZE_MARKER))
    external = paths.read_json(context.path(EXTERNAL_MARKER))
    aggregate = hashlib.sha256(canonical_json({
        "repository_snapshot_sha256": repository["snapshot_sha256"],
        "external_sha256": external["external_sha256"],
        "config_sha256": paths.sha256_file(context.config_path)}).encode("utf-8")).hexdigest()
    return {"schema_version": ROUTES_SCHEMA, "record_kind": "combined_identity",
            "repository_snapshot_sha256": repository["snapshot_sha256"],
            "repository_files": repository["file_count"],
            "external_sha256": external["external_sha256"],
            "external_files": external["file_count"],
            "config_sha256": paths.sha256_file(context.config_path),
            "combined_sha256": aggregate,
            "identity_is": ("the combined digest. This flight is authorized to build and train and "
                            "NOT to commit, so the identity is content-addressed rather than a "
                            "commit, and it spans both the repository closure and the pinned "
                            "external sources.")}


# ---------------------------------------------------------------------------
# revisions
# ---------------------------------------------------------------------------

def write_revision_marker(context, *, kind, supersedes=None, reason=None, frozen_after=()):
    require(kind in REVISION_KINDS, f"Unknown revision kind {kind!r}")
    if kind == REVISION_CONFIRMATION:
        require(frozen_after,
                "A confirmation revision must name what was frozen before it: architecture, "
                "objectives, mix, decoding and checkpoint selection. A confirmation frozen before "
                "selection is not a confirmation.")
    document = {"schema_version": ROUTES_SCHEMA, "record_kind": "revision",
                "revision": context.revision, "kind": kind,
                "supersedes": supersedes, "reason": reason,
                "frozen_after": sorted(str(value) for value in frozen_after),
                "identity": combined_identity(context),
                "rules": {
                    "development": ("numbers may be read for curves and for selection. May be "
                                    "superseded by a new revision if a round finds a real "
                                    "training-path bug."),
                    "confirmation": ("frozen last, after every selection. Nothing in its closure "
                                     "is edited afterwards."),
                    "supersession": ("the superseded root is retained READ-ONLY and is never "
                                     "resumed. This is explicit transfer under a new provenance "
                                     "identity, not a silent resume.")},
                "frozen_at": paths.utc_now()}
    paths.write_json(context.path(REVISION_MARKER), document)
    return document


def read_revision_marker(context):
    target = context.path(REVISION_MARKER)
    require(target.is_file(), f"{target} is absent; every stage runs under a declared revision")
    return paths.read_json(target)


def require_revision_discipline(context, *, stage, selection_frozen=False):
    """Refuse a confirmation stage under a development revision, and the reverse.

    The one rule evidence-driven correction depends on: development numbers are
    allowed to inform selection, and confirmation numbers are not allowed to be
    produced before selection is frozen.
    """
    marker = read_revision_marker(context)
    kind = marker["kind"]
    if stage in ("confirm", "report_confirmation"):
        require(kind == REVISION_CONFIRMATION,
                f"{stage} requires a {REVISION_CONFIRMATION!r} revision and this run is "
                f"{kind!r}. A confirmation produced under the development identity would be "
                "evidence about a specification that selection had already seen.")
        require(selection_frozen,
                f"{stage}: selection is not frozen. Architecture, objectives, mix, decoding and "
                "checkpoint selection are all frozen before the confirmation revision is created.")
    if stage in SELECTION_STAGES and kind == REVISION_CONFIRMATION:
        require(False,
                f"{stage} is a selection stage and this run is the confirmation revision. "
                "Selecting under the confirmation identity is what the two-revision rule exists "
                "to prevent.")
    return {"revision": context.revision, "kind": kind, "stage": stage,
            "selection_frozen": bool(selection_frozen)}


def require_not_resumed(context, *, superseded_roots):
    """A superseded revision root is read-only. Refuse to write into one."""
    conflicts = []
    for logical in superseded_roots:
        candidate = Path(logical)
        if candidate.exists() and paths.overlapping(context.run_root, candidate):
            conflicts.append(str(candidate.name))
    require(not conflicts,
            f"The run root overlaps the superseded revision root(s) {conflicts}. A superseded root "
            "is retained read-only and never resumed: Services-style helpers write into the run "
            "root, and a follow-up aimed at a landed flight has already broken a manifest once.")
    return True


#: What a five-way competition needs, held in CODE. A config-only pending list could be
#: satisfied by deleting its entries, so the config's ``competition`` block is compared
#: against these and can only ever ADD to what is unresolved.
COMPETITION_ROUTES = ("ablang2_masked_editor", "piggen_causal_infiller", "dwjs_energy",
                      "dplm_150m_masked", "dplm_650m_lora_masked")
COMPETITION_AMENDMENTS = ("hr17_high_null_ratio", "hr18_bank_and_distance_strata",
                          "originals_checkpoint_policy")
#: Edited only when the code for an amendment lands. Empty: none is implemented, so a
#: config that declares a competition cannot pass the resolved-protocol hold.
IMPLEMENTED_AMENDMENTS = ()


def competition_unresolved(config):
    """The competition fields that keep a joint config from resolving.

    ``[]`` for a config that neither carries a ``competition`` block nor enables every
    competition route (the two existing configs). A config that enables all five routes is a
    competition config whether or not its block survived, so deleting the block does not clear
    the hold.
    """
    enabled = list((config.get("routes") or {}).get("enabled") or ())
    block = config.get("competition")
    if not block and not set(COMPETITION_ROUTES) <= set(enabled):
        return []
    block = dict(block or {})
    unresolved = []
    required = list(block.get("required_routes") or ())
    if (enabled != required or enabled != list(COMPETITION_ROUTES)
            or required != list(COMPETITION_ROUTES)):
        unresolved.append("competition.route_set")
    if list(block.get("required_amendments") or ()) != list(COMPETITION_AMENDMENTS):
        unresolved.append("competition.amendment_list")
    for name in COMPETITION_AMENDMENTS:
        if name not in IMPLEMENTED_AMENDMENTS:
            unresolved.append(f"competition.amendment::{name}")
    return unresolved


#: The explicit DPLM TRAINING phase. It is a training-only selector, NOT competition admission:
#: it does not touch ``IMPLEMENTED_AMENDMENTS`` (the five-way joint config and its hold are
#: unchanged), it implements the DPLM training selector and not complete HR17 (head diagnostics
#: and the all-routes comparison are deferred), and HR18 plus checkpoint import/admission need a
#: frozen protocol of their own before any joint selection. Every constant below is held in CODE:
#: the config's ``training_phase`` block is compared against them and can neither relax nor
#: extend them.
TRAINING_PHASE_KEY = "training_phase"
PHASE_DPLM_TRAINING = "dplm_explicit_training"
PHASE_ADMISSION_NONE = "none"
TRAINING_PHASE_ROUTES = ("dplm_150m_masked", "dplm_650m_lora_masked")
TRAINING_PHASE_UPDATES = 20000
TRAINING_PHASE_MILESTONES = (5000, 10000, 15000)
TRAINING_PHASE_STAGES = ("acquire", "preflight", "split", "sources", "profile", "smoke", "freeze",
                         "bakeoff", "verify")
TRAINING_PHASE_HELD_STAGES = ("route_gate", "controls", "priors", "posttrain", "decoding",
                              "confirmation_freeze", "confirm", "report")
SELECTION_METRIC_LEGACY = "development_affinity_spearman_high"
SELECTION_METRIC_TRAINING = "hr17_fixed_path_high_minus_null_negative_spearman"
#: Canonical digest (``paths.digest_document``) of the hard-split freeze the original t1 flight and
#: both qualification runs wrote. Identical across all three, measured; the split is not invented
#: or changed here.
TRAINING_PHASE_SPLIT_SHA256 = "3955a5981c225e94ba2b08e97e90175b5a3b6cf35393adea9b018c5542fd5d49"
TRAINING_PHASE_SEQUENCE_SHA256 = "a742b610f1ea120a8d669b2d240252cb11c185dd4574978cea13314b8a575b73"


def training_phase_declaration():
    """The complete phase declaration. The config block must equal this exactly."""
    return {"phase_id": PHASE_DPLM_TRAINING, "admission": PHASE_ADMISSION_NONE,
            "routes": list(TRAINING_PHASE_ROUTES), "updates": int(TRAINING_PHASE_UPDATES),
            "milestones": [int(value) for value in TRAINING_PHASE_MILESTONES],
            "stages": list(TRAINING_PHASE_STAGES),
            "held_stages": list(TRAINING_PHASE_HELD_STAGES),
            "selection_metric": SELECTION_METRIC_TRAINING,
            "head_diagnostics": "deferred",
            "expected_split_sha256": TRAINING_PHASE_SPLIT_SHA256,
            "expected_sequence_sha256": TRAINING_PHASE_SEQUENCE_SHA256}


def training_phase(config):
    """``None`` for a config with no phase key; else the validated declaration plus its digest.

    Presence, not truthiness, decides: ``training_phase: {}`` or ``null`` is a malformed phase and
    is refused rather than read as "no phase". The block is a closed schema (no missing key, no
    extra key, every value equal to the code's), the route set and schedule must be exactly the
    declared ones, and a config that carries a ``competition`` block is refused, so the phase
    cannot be grafted onto the five-way config to clear its hold.
    """
    if TRAINING_PHASE_KEY not in config:
        return None
    block = config[TRAINING_PHASE_KEY]
    require(isinstance(block, dict) and block,
            f"{TRAINING_PHASE_KEY} must be a nonempty mapping (found {block!r})")
    declared = training_phase_declaration()
    require(set(block) == set(declared),
            f"{TRAINING_PHASE_KEY} keys differ from the closed schema: missing "
            f"{sorted(set(declared) - set(block))}, unexpected {sorted(set(block) - set(declared))}")
    for key, value in declared.items():
        require(block[key] == value and type(block[key]) is type(value),
                f"{TRAINING_PHASE_KEY}.{key} is {block[key]!r}; the code fixes it at {value!r}. "
                "The phase's routes, schedule, stage and held-stage declarations, policy and "
                "split pin cannot be relaxed by a config.")
    routes = dict(config.get("routes") or {})
    require(list(routes.get("enabled") or ()) == list(TRAINING_PHASE_ROUTES),
            f"routes.enabled must be exactly {list(TRAINING_PHASE_ROUTES)} in that order")
    for route in TRAINING_PHASE_ROUTES:
        route_block = dict(routes.get(route) or {})
        updates = route_block.get("updates")
        require(type(updates) is int and updates == TRAINING_PHASE_UPDATES,
                f"routes.{route}.updates must be {TRAINING_PHASE_UPDATES}")
        require([int(v) for v in route_block.get("milestones") or ()]
                == list(TRAINING_PHASE_MILESTONES),
                f"routes.{route}.milestones must be {list(TRAINING_PHASE_MILESTONES)}")
    require("competition" not in config and not set(COMPETITION_ROUTES) <= set(routes["enabled"]),
            f"{TRAINING_PHASE_KEY} cannot be combined with a competition config: it is not "
            "competition admission and does not clear the joint hold")
    require(str((config.get("revisions") or {}).get("kind")) == REVISION_DEVELOPMENT,
            f"{TRAINING_PHASE_KEY} runs under a development revision")
    return dict(declared, policy_sha256=paths.digest_document(declared))


def verify_training_phase_split(context, policy=None):
    """Refuse a split freeze that is not the pinned one. ``None`` when there is no phase.

    Called at the split and freeze stages and again before production training, so a wrong split
    stops the run before any update instead of being found by the final verification.
    """
    policy = policy if policy is not None else training_phase(context.config)
    if policy is None:
        return None
    target = context.path(SPLIT_FREEZE)
    require(target.is_file(), f"{target} is absent; the pinned split cannot be verified")
    document = paths.read_json(target)
    observed = paths.digest_document(document)
    require(observed == policy["expected_split_sha256"],
            f"the frozen split digests to {observed}; the phase pins "
            f"{policy['expected_split_sha256']}. The split is not invented or changed here.")
    require(document.get("sequence_sha256") == policy["expected_sequence_sha256"],
            f"the frozen split's sequence digest is {document.get('sequence_sha256')}; the phase "
            f"pins {policy['expected_sequence_sha256']}")
    return {"split_sha256": observed, "sequence_sha256": document["sequence_sha256"]}


#: Config sections a legacy flight resolves against. A training phase needs only the first two.
DOWNSTREAM_KEYS = ("gates", "decoding", "posttrain", "confirmation")


def resolved_protocol(context, *, snapshot, external, split, profile, forecast, revision):
    """Every field production runs against, with the unresolved ones named."""
    config = context.config
    unresolved = []
    try:
        policy = training_phase(config)
    except ValueError as error:
        policy = None
        unresolved.append("training_phase.invalid")
        phase_error = str(error)
    else:
        phase_error = None
    for name, value in (("split", split), ("profile", profile), ("forecast", forecast)):
        if not value:
            unresolved.append(name)
    required = ("routes", "curriculum") if policy is not None else (
        "routes", "curriculum", *DOWNSTREAM_KEYS)
    for key in required:
        if not config.get(key):
            unresolved.append(f"config.{key}")
    unresolved.extend(competition_unresolved(config))
    document = {"schema_version": ROUTES_SCHEMA, "record_kind": "resolved_protocol",
                "campaign_id": context.campaign_id, "revision": dict(revision),
                "identity": {"repository_snapshot_sha256": snapshot.get("snapshot_sha256"),
                             "external_sha256": (external or {}).get("external_sha256")},
                "split": dict(split or {}), "profile": dict(profile or {}),
                "forecast": dict(forecast or {}),
                "routes": dict(config.get("routes") or {}),
                "curriculum": dict(config.get("curriculum") or {})}
    for key in DOWNSTREAM_KEYS:
        if policy is None or key in config:
            document[key] = dict(config.get(key) or {})
    if policy is not None:
        document[TRAINING_PHASE_KEY] = dict(policy)
    if phase_error is not None:
        document["training_phase_error"] = phase_error
    document["unresolved"] = sorted(set(unresolved))
    document["rule"] = ("production runs against a filled protocol. An unresolved field is a "
                        "readiness gate, not a default.")
    paths.write_json(context.path("resolved_protocol.json"), document)
    return document


def require_resolved(context, *, label="production"):
    target = context.path("resolved_protocol.json")
    require(target.is_file(), f"{label}: {target} is absent; run the freeze stage first")
    document = paths.read_json(target)
    require(not document["unresolved"],
            f"{label}: the resolved protocol still has unresolved fields "
            f"{document['unresolved']}")
    return document
