"""Configuration, verified roots, the computed source closure and the training freeze.

The audit was frozen before it scored; this screen is frozen before it fits, and
the second specification is **informed by the first** -- the escalation the audit
recorded is what justifies the grid, and that dependence is stated in the marker
rather than implied by a date. Nothing here trains, samples or commits.

Five properties are enforced here rather than left to convention:

* **The source closure is computed, not declared.** ``AUDIT_SOURCE_FILES`` is a
  hand-maintained tuple whose own docstring says the rule is the module-level
  import closure: the rule was right and its enforcement was a human. This module
  walks the closure with :mod:`ast` from the actual CLI entry points, which also
  catches function-level imports that a ``sys.modules`` walk at freeze time would
  miss, and it includes every package ``__init__.py`` on the path. External
  dependencies are recorded from the environment; the virtualenv is never read.
* **Worktree bytes are compared to committed bytes.** A clean ``git status`` and a
  matching blob id both survive newline translation, because git normalizes what
  it stores. The digests this campaign records are of files on disk, so the freeze
  compares those to the bytes the commit actually holds.
* **A root is accepted by hash.** Every input root is probed with digests taken
  from the *published, tracked* audit evidence, so "this directory holds the
  campaign the config names" is a measurement rather than an existence check.
* **The banks do not exist yet at freeze time, and that is deliberate.** The
  freeze binds their sampling rule, their seeds, their counts and their dtypes.
  They are generated afterwards and their completed draw-order and content digests
  are bound to the freeze then, in a separate banks manifest that names it.
* **The frozen fields are immutable at fit time.** A CLI flag that would change
  the device, the microbatch, the effective batch or a seed is rejected, not
  applied: a run that quietly overrode one would not be the run that was reviewed.
"""
from __future__ import annotations

import ast
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from . import her2_support as support
from . import her2_support_paths as paths
from .her2_runtime import canonical_json, require

REPLAY_SCHEMA = "her2-parent-replay/1"
#: Written by the freeze stage after the review commit. Ignored by Git; it names
#: the real HEAD rather than being part of it.
FREEZE_MARKER = "training_spec_frozen.json"

#: Logical names inside the replay run root.
RESOLVED_CONFIG_JSON = "replay_config.json"
INPUT_MANIFEST_JSON = "input_manifest.json"
PREFLIGHT_JSON = "preflight.json"
BANKS_MANIFEST_JSON = "banks_manifest.json"
CAMPAIGN_STATUS_JSON = "campaign_status.json"
QUEUE_JSON = "queue.json"

#: New text this campaign hashes. These paths get narrow ``text eol=lf`` rules in
#: ``.gitattributes`` *before* any digest is taken, so a clone on another platform
#: reproduces the bytes. Pre-existing sources in the closure are not renormalized:
#: for them the worktree-versus-HEAD comparison is the binding guarantee, and their
#: observed newline state is recorded rather than corrected.
REPLAY_LF_PINNED = (
    "configs/experiments/her2_parent_replay.json",
    "scripts/posttrain_her2_replay.py",
    "scripts/rebuild_her2_support_report.py",
    "src/smallAntibodyGen/experiments/her2_replay.py",
    "src/smallAntibodyGen/experiments/her2_replay_banks.py",
    "src/smallAntibodyGen/experiments/her2_replay_campaign.py",
    "src/smallAntibodyGen/experiments/her2_replay_report.py",
    "src/smallAntibodyGen/experiments/her2_replay_spec.py",
    "src/smallAntibodyGen/experiments/her2_replay_streams.py",
)

#: The environment fields the freeze pins and every gated stage re-checks once.
#: Deliberately the library versions that decide a number -- not the host name, not
#: the driver string, which change for reasons that are not the computation.
RUNTIME_FIELDS = ("python", "torch_version", "transformers_version", "numpy_version",
                  "pandas_version")

#: Package initializers that already hold CRLF/mixed bytes on disk while Git
#: stores LF. They are in this screen's computed source closure, so the freeze
#: compares their worktree bytes to HEAD; narrow ``-text whitespace=cr-at-eol``
#: rules in ``.gitattributes`` make Git store exactly the bytes that are there.
#: Their contents are not edited and nothing is broadly renormalized.
LEGACY_RAW_INITIALIZERS = (
    "src/smallAntibodyGen/__init__.py",
    "src/smallAntibodyGen/benchmarks/__init__.py",
    "src/smallAntibodyGen/experiments/__init__.py",
)

#: Free bytes the run volume must have before the campaign is allowed to start.
#: 20 GiB against a ~16 GB budget leaves room for the transient second copy an
#: atomic checkpoint write needs. The observed figure is recorded either way.
MIN_FREE_BYTES = 20 * 1024 ** 3


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

def placeholders(node, prefix=""):
    """Any ``TODO``/``FIXME``/``TBD``/``PLACEHOLDER`` left in a scientific field."""
    found = []
    if isinstance(node, dict):
        for key, value in node.items():
            found.extend(placeholders(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(node, list):
        for position, value in enumerate(node):
            found.extend(placeholders(value, f"{prefix}[{position}]"))
    elif isinstance(node, str) and node.strip().upper() in ("TODO", "FIXME", "TBD", "PLACEHOLDER",
                                                            "NULL", "CHANGEME"):
        found.append(prefix)
    elif node is None and prefix.split(".")[-1] in ("seed", "lambda", "tau", "device"):
        found.append(prefix)
    return found


def load_config(path):
    """Read the replay config, refuse a launch-time placeholder, return its digests."""
    path = Path(path)
    require(path.is_file(), f"Replay config {path} is missing")
    document = paths.read_json(path)
    require(document.get("schema_version") == REPLAY_SCHEMA,
            f"{path} is not a {REPLAY_SCHEMA} config")
    blanks = sorted(placeholders(document))
    require(not blanks,
            f"{path} still carries launch-time placeholders at {blanks}. Every scientific setting "
            "is complete at commit; mechanically generated input bindings live in the tracked "
            "input manifest the config links to.")
    return document, paths.sha256_file(path), paths.digest_document(document)


@dataclass
class ReplayContext:
    """Everything a stage needs that does not depend on which stage it is."""

    repository_root: Path
    run: paths.RunPaths
    config: dict
    config_path: Path
    config_sha256: str
    config_digest: str
    roots: dict = field(default_factory=dict)
    audit: dict = field(default_factory=dict)

    @property
    def seeds(self):
        return [int(seed) for seed in self.config["screen"]["parent_seeds"]]

    @property
    def batch_rows(self):
        return int(self.config["screen"]["chosen_per_update"])

    @property
    def endpoint_updates(self):
        return [int(value) for value in self.config["screen"]["endpoint_updates"]]

    def root(self, name):
        require(name in self.roots, f"Root {name!r} was not resolved")
        return self.roots[name]

    def progress(self, stage, *, total=None, note=None, every=1):
        return paths.StageProgress(self.run.path(f"progress/{stage}.json"), stage=stage,
                                   total=total, note=note, every=every)

    def stage_status(self, stage):
        path = self.run.path(f"progress/{stage}.json")
        return paths.read_json(path) if path.is_file() else None

    def relative(self, path):
        return support._relative(path, self.repository_root)


def published_audit(repository_root, config):
    """The tracked, immutable half of the completed audit this screen is conditional on.

    Deliberately the *published* documents and the completion marker, never the
    run directory's ``verification.json`` or its ``progress/*.json``: those are
    operational, they are rewritten by every later verify, and binding one as a
    perpetual input would make this campaign's identity depend on when somebody
    last re-checked the audit.

    Three checks, and the first two used to be missing. Hashing the published files
    that happen to be present proves only that they are readable; each digest is
    compared to the **publication manifest** that published them, and the manifest
    itself is bound by digest so it cannot be the thing that moved. The published
    verification summary must record ``immutable: true`` with no problems: a screen
    conditional on a completed audit is not conditional on one whose own
    immutability check failed. The decision outcome is then compared to the one
    this config declares it is conditional on.
    """
    root = Path(repository_root)
    manifest_relative = config["audit"]["publication_manifest"]
    manifest_path = root / manifest_relative
    require(manifest_path.is_file(),
            f"{manifest_relative} is absent. The publication manifest is the authority the "
            "published audit files are checked against; hashing whatever is on disk and comparing "
            "it to itself would prove nothing.")
    manifest = paths.read_json(manifest_path)
    require(manifest.get("record_kind") == "publication_manifest",
            f"{manifest_relative} is not a publication manifest")
    publication_root = str(manifest.get("root") or "").rstrip("/")
    published_files = dict(manifest.get("files") or {})
    block, unbound = {}, []
    for name, relative in sorted(dict(config["audit"]["published"]).items()):
        target = root / relative
        require(target.is_file(),
                f"{relative} is absent. This screen is conditional on the completed audit; its "
                "published decision and verification summary are inputs, not background.")
        observed = paths.sha256_file(target)
        key = (relative[len(publication_root) + 1:]
               if publication_root and relative.startswith(publication_root + "/") else relative)
        recorded = published_files.get(key)
        if recorded is None:
            unbound.append(relative)
        elif recorded != observed:
            unbound.append(f"{relative} hashes {observed}, the publication manifest recorded "
                           f"{recorded}")
        block[name] = {"logical_path": relative, "sha256": observed,
                       "publication_key": key, "manifest_sha256": recorded}
    require(not unbound,
            f"These consulted audit files are not bound to their publication manifest: {unbound}. "
            "A published input is accepted because the manifest that published it records these "
            "exact bytes, never because a file with that name exists.")
    verification = paths.read_json(root / config["audit"]["published"]["verification"])
    require(verification.get("immutable") is True and not (verification.get("problems") or []),
            "The published audit verification does not record immutable=true with no problems "
            f"({verification.get('immutable')!r}, {len(verification.get('problems') or [])} "
            "problems). This screen is conditional on a completed audit, and an audit whose own "
            "immutability check did not pass is not one.")
    decision = paths.read_json(root / config["audit"]["published"]["decision"])
    require(decision.get("outcome") == config["audit"]["required_outcome"],
            f"The published audit decision is {decision.get('outcome')!r} and this screen is "
            f"declared conditional on {config['audit']['required_outcome']!r}. A screen whose "
            "precondition did not hold is not run under a config that says it did.")
    return {"published": block, "publication_manifest": dict(manifest),
            "publication_manifest_binding": {
                "logical_path": manifest_relative,
                "sha256": paths.sha256_file(manifest_path),
                "digest": paths.digest_document(manifest),
                "file_count": len(published_files),
                "role": ("the authority every published digest above was compared to; bound here "
                         "so the authority itself cannot be the thing that changed")},
            "published_verification": {"immutable": verification.get("immutable"),
                                       "problems": len(verification.get("problems") or []),
                                       "shards_checked": verification.get("shards_checked")},
            "decision_outcome": decision.get("outcome"),
            "decision_methods": {name: entry.get("outcome")
                                 for name, entry in sorted((decision.get("methods") or {}).items())},
            "audit_source_freeze_commit": config["audit"]["source_freeze_commit"],
            "dependence": ("this training specification was written AFTER the audit and is "
                           "informed by it. Nothing here was fixed before the audit ran, and the "
                           "artifacts say so.")}


def require_audit_completion(config, completion, *, audit):
    """Tie the claimed audit freeze and outcome to the completion marker's own evidence.

    The config *says* which commit froze the audit and which outcome this screen is
    conditional on. The completion marker is where the audit recorded both, after
    its own requirement checks and its immutability verification passed. Comparing
    them is what stops this screen from being conditional on a completion it
    describes rather than one that happened.
    """
    require(completion.get("record_kind") == "audit_complete",
            "The audit completion marker is not an audit_complete record")
    declared = config["audit"]["source_freeze_commit"]
    observed = completion.get("frozen_commit")
    require(observed == declared,
            f"The audit completion marker was written under freeze {observed} and this config "
            f"declares the audit source freeze {declared}. The commit this screen names as its "
            "precondition is not the one the completed audit ran under.")
    require(completion.get("decision_outcome") == config["audit"]["required_outcome"],
            f"The completion marker records outcome {completion.get('decision_outcome')!r} and "
            f"this screen is conditional on {config['audit']['required_outcome']!r}")
    marker_verification = dict(completion.get("verification") or {})
    require(marker_verification.get("immutable") is True,
            "The audit completion marker does not record a passing immutability verification")
    require(marker_verification.get("shards_checked")
            == audit["published_verification"]["shards_checked"],
            f"The completion marker verified {marker_verification.get('shards_checked')} shards "
            f"and the published verification summary records "
            f"{audit['published_verification']['shards_checked']}. Two different completion "
            "claims about one audit is a finding, not a rounding difference.")
    return {"frozen_commit": observed, "decision_outcome": completion.get("decision_outcome"),
            "completed_at": completion.get("completed_at"),
            "shards_checked": marker_verification.get("shards_checked"),
            "stages": dict(completion.get("stages") or {}),
            "binding": ("the config's declared audit freeze and required outcome are compared to "
                        "the completion marker's own record of both, and the marker's shard count "
                        "to the published verification summary")}


def audit_input_manifest(repository_root, config):
    """The audit's committed input manifest: the source of every verified digest."""
    relative = config["audit"]["input_manifest"]
    target = Path(repository_root) / relative
    require(target.is_file(), f"{relative} is absent; the audit's input manifest is this "
                              "campaign's source of verified input digests")
    document = paths.read_json(target)
    require(document.get("record_kind") == "input_manifest", f"{relative} is not an input manifest")
    return document, {"logical_path": relative, "sha256": paths.sha256_file(target)}


def _probes_from(manifest, *, root_name, kinds=None, logical_paths=None):
    """``{logical_path: sha256}`` for the audit inputs under one root."""
    probes = {}
    for entry in (manifest.get("inputs") or {}).values():
        if entry.get("root") != root_name:
            continue
        if kinds is not None and entry.get("kind") not in kinds:
            continue
        if logical_paths is not None and entry.get("logical_path") not in logical_paths:
            continue
        probes[entry["logical_path"]] = entry["sha256"]
    return probes


def resolve_context(repository_root, *, config_path, run_root=None, root_overrides=None):
    """Load the config and accept every input root by re-hashing published digests."""
    repository_root = Path(repository_root)
    config, config_sha256, config_digest = load_config(config_path)
    run = paths.RunPaths.create(repository_root,
                                run_root if run_root is not None else
                                repository_root / config["run_root"],
                                logical=config["run_root"], create=False)
    local_roots = paths.load_local_roots(run.run_root) if run.run_root.is_dir() else {}
    overrides = dict(root_overrides or {})
    manifest, manifest_entry = audit_input_manifest(repository_root, config)
    audit = published_audit(repository_root, config)
    audit["input_manifest"] = manifest_entry

    roots = {}
    for name in ("original", "guarded", "raw"):
        settings = config["roots"][name]
        logical = settings["logical"]
        probes = _probes_from(manifest, root_name=name,
                              kinds=tuple(settings["probe_kinds"]) if settings.get("probe_kinds")
                              else None,
                              logical_paths=set(settings["probe_files"])
                              if settings.get("probe_files") else None)
        require(probes,
                f"No published audit input under root {name!r} matches the declared probe "
                f"selection {settings.get('probe_files') or settings.get('probe_kinds')}; a root "
                "is accepted by re-hashing a published digest, never by existing")
        roots[name] = paths.resolve_root(
            name, logical=logical,
            candidates=paths.root_candidates(repository_root=repository_root, logical=logical,
                                             explicit=overrides.get(name),
                                             local_roots=local_roots),
            probes=probes)
    audit_settings = config["roots"]["audit_run"]
    roots["audit_run"] = paths.resolve_root(
        "audit_run", logical=audit_settings["logical"],
        candidates=paths.root_candidates(repository_root=repository_root,
                                         logical=audit_settings["logical"],
                                         explicit=overrides.get("audit_run"),
                                         local_roots=local_roots),
        probes={audit_settings["probe_file"]:
                audit["published"][audit_settings["probe_matches_published"]]["sha256"]})

    paths.require_disjoint_run_root(run.run_root, {
        "guarded_campaign": roots["guarded"].local_path,
        "original_campaign": roots["original"].local_path,
        "raw_release": roots["raw"].local_path,
        "support_audit_run": roots["audit_run"].local_path,
        "tracked_evidence": repository_root / config["evidence_root"]})
    run.ensure()
    return ReplayContext(repository_root=repository_root, run=run, config=config,
                         config_path=Path(config_path), config_sha256=config_sha256,
                         config_digest=config_digest, roots=roots, audit=audit)


# ---------------------------------------------------------------------------
# capacity
# ---------------------------------------------------------------------------

def capacity_record(path, *, minimum_bytes=MIN_FREE_BYTES, required=True):
    """Observed free capacity on the volume that will hold the checkpoints.

    Recorded whether or not it passes, because "how much room was actually there"
    is the number a later interruption has to be read against. The minimum is
    frozen in the config; it is not recomputed from whatever is free today.
    """
    usage = shutil.disk_usage(str(Path(path)))
    record = {"free_bytes": int(usage.free), "total_bytes": int(usage.total),
              "free_gib": usage.free / 1024 ** 3, "total_gib": usage.total / 1024 ** 3,
              "required_free_bytes": int(minimum_bytes),
              "required_free_gib": int(minimum_bytes) / 1024 ** 3,
              "sufficient": bool(usage.free >= int(minimum_bytes)),
              "basis": ("the declared checkpoint and bank budget plus the transient second copy "
                        "an atomic write needs; the volume may be a junction to another disk")}
    if required:
        require(record["sufficient"],
                f"The run volume has {record['free_gib']:.1f} GiB free and this campaign declares "
                f"{record['required_free_gib']:.1f} GiB. Provision the mapped output root before "
                "starting; a campaign that fills its disk at trajectory 30 loses the trajectory "
                "and leaves a partial checkpoint behind.")
    return record


# ---------------------------------------------------------------------------
# the computed source closure
# ---------------------------------------------------------------------------

def module_name_for(repository_root, relative):
    """The dotted module name of a repo-relative file, or ``None`` for a script."""
    parts = str(relative).split("/")
    if parts[0] != "src":
        return None
    dotted = parts[1:]
    require(dotted[-1].endswith(".py"), f"{relative} is not a python source file")
    dotted[-1] = dotted[-1][:-3]
    if dotted[-1] == "__init__":
        dotted = dotted[:-1]
    return ".".join(dotted)


def module_relative(repository_root, dotted):
    """The repo-relative file implementing ``dotted``, or ``None`` if it is external."""
    parts = str(dotted).split(".")
    if not parts or not parts[0]:
        return None
    base = Path(repository_root) / "src"
    candidate = base.joinpath(*parts)
    if (candidate / "__init__.py").is_file():
        return "/".join(["src", *parts, "__init__.py"])
    if candidate.with_suffix(".py").is_file():
        return "/".join(["src", *parts[:-1], f"{parts[-1]}.py"])
    return None


def package_initializers(repository_root, dotted):
    """Every ``__init__.py`` on the package path of ``dotted`` that exists in the repo."""
    parts = str(dotted).split(".")
    found = []
    for depth in range(1, len(parts) + 1):
        candidate = Path(repository_root) / "src" / Path(*parts[:depth]) / "__init__.py"
        if candidate.is_file():
            found.append("/".join(["src", *parts[:depth], "__init__.py"]))
    return found


def imported_names(tree, *, module_name, is_package=False):
    """Every dotted name a module imports, relative imports resolved to absolutes.

    ``ast.walk`` visits function bodies too, so a lazy import inside a function --
    which this repository uses for torch, pandas and matplotlib, and which a
    ``sys.modules`` snapshot at freeze time would miss entirely -- is caught here.

    ``is_package`` is not a detail. :func:`module_name_for` gives an ``__init__.py``
    the dotted name of the package it *is*, so resolving ``from . import leaf``
    inside it by the ordinary module rule strips the package's own last component
    and yields ``leaf`` instead of ``pkg.leaf`` -- a file the initializer executes,
    missing from the closure the freeze binds. Inside a package initializer, level 1
    is the package itself.
    """
    names = set()
    package = (module_name or "") if is_package else (
        (module_name or "").rsplit(".", 1)[0] if module_name else None)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                require(module_name is not None,
                        "A relative import appears in a file that is not inside the package")
                owner = module_name.split(".")
                # ``from . import x`` resolves against the importer's package. For an
                # ordinary module that is its parent; for a package initializer the
                # package is the module itself, so one fewer component is stripped.
                depth = node.level - 1 if is_package else node.level
                base = owner[:len(owner) - depth] if depth <= len(owner) else []
                prefix = ".".join(base + ([node.module] if node.module else []))
            else:
                prefix = node.module or ""
            if prefix:
                names.add(prefix)
            for alias in node.names:
                names.add(f"{prefix}.{alias.name}" if prefix else alias.name)
    if package:
        names.add(package)
    return names


def external_versions(names):
    """Installed versions of the external top-level packages, from the environment.

    Read through :mod:`importlib.metadata` and module attributes; the virtualenv
    tree is never walked. A package whose version cannot be established is recorded
    as ``null`` with the reason rather than omitted.
    """
    from importlib import metadata
    out = {}
    for name in sorted({str(value).split(".")[0] for value in names}):
        if name in sys.builtin_module_names or name in getattr(sys, "stdlib_module_names", ()):
            out[name] = {"version": None, "kind": "standard_library"}
            continue
        try:
            out[name] = {"version": metadata.version(name), "kind": "distribution"}
        except Exception as error:                              # noqa: BLE001 - recorded
            module = sys.modules.get(name)
            version = getattr(module, "__version__", None) if module is not None else None
            out[name] = {"version": None if version is None else str(version),
                         "kind": "imported_module" if version else "unresolved",
                         "reason": type(error).__name__}
    return out


def source_closure(repository_root, entry_points):
    """The transitive repo-local import closure of ``entry_points``, plus externals.

    Static rather than dynamic on purpose. A ``sys.modules`` walk reports what one
    execution happened to import, which is a function of which branches ran; the
    closure a freeze needs is what the code *can* import, and that is what the
    syntax says.
    """
    repository_root = Path(repository_root)
    pending = [str(entry) for entry in entry_points]
    files, external = {}, set()
    while pending:
        relative = pending.pop()
        if relative in files:
            continue
        target = repository_root / relative
        require(target.is_file(),
                f"{relative} is part of the computed source closure but is missing from the "
                "working tree")
        module_name = module_name_for(repository_root, relative)
        is_package = relative.endswith("/__init__.py")
        tree = ast.parse(target.read_text(encoding="utf-8"), filename=relative)
        files[relative] = module_name
        for dotted in sorted(imported_names(tree, module_name=module_name,
                                            is_package=is_package)):
            resolved = module_relative(repository_root, dotted)
            if resolved is None:
                parent = dotted.rsplit(".", 1)[0] if "." in dotted else None
                resolved = module_relative(repository_root, parent) if parent else None
                if resolved is None:
                    external.add(dotted)
                    continue
            for initializer in package_initializers(repository_root, dotted):
                if initializer not in files:
                    pending.append(initializer)
            if resolved not in files:
                pending.append(resolved)
    return {"entry_points": sorted(str(entry) for entry in entry_points),
            "files": sorted(files),
            "modules": {name: files[name] for name in sorted(files)},
            "external": sorted(external),
            "external_versions": external_versions(external),
            "rule": ("transitive module-level AST import closure over repository-local modules, "
                     "seeded from the actual CLI entry points. Function-level imports are included "
                     "because the walk is syntactic; package __init__.py files are included, and a "
                     "relative import inside one resolves against that package rather than its "
                     "parent, so a module an initializer executes is in the closure; external "
                     "dependencies are recorded from the environment and the virtualenv tree is "
                     "never read.")}


def require_runtime_closure(closure, *, repository_root, label="fit"):
    """Every repo-local module currently imported must appear in the frozen closure.

    The static closure is the authority. This is the complementary check at run
    time: a module that got imported by some path the syntax did not reveal -- a
    plugin, an ``importlib`` call, a test shim -- would otherwise contribute to the
    computation without being part of the frozen identity.
    """
    root = Path(repository_root).resolve()
    frozen = set(closure["files"])
    extra = []
    for name, module in sorted(sys.modules.items()):
        origin = getattr(module, "__file__", None)
        if not origin:
            continue
        try:
            relative = Path(origin).resolve().relative_to(root)
        except ValueError:
            continue
        logical = str(relative).replace("\\", "/")
        if logical.startswith("src/") and logical.endswith(".py") and logical not in frozen:
            extra.append(logical)
    require(not extra,
            f"{label}: these repository modules are imported but are not in the frozen source "
            f"closure: {sorted(set(extra))}. The frozen closure is what the recorded numbers are "
            "attributed to; an unfrozen module that participates in the computation is a gap in "
            "that attribution, not a detail.")
    return sorted(frozen)


# ---------------------------------------------------------------------------
# the freeze marker
# ---------------------------------------------------------------------------

def freeze_marker_path(context):
    return context.run.path(FREEZE_MARKER)


def read_freeze_marker(context):
    path = freeze_marker_path(context)
    require(path.is_file(),
            f"{path} is absent: fitting requires a recorded training_spec_frozen marker naming the "
            "commit that fixed the sources, config and inputs. Run the freeze stage after the "
            "review commit; there is no bypass and no --allow-dirty.")
    marker = paths.read_json(path)
    require(marker.get("record_kind") == "training_spec_frozen", f"{path} is not a freeze marker")
    require(marker.get("schema_version") == REPLAY_SCHEMA, f"{path} is not a {REPLAY_SCHEMA} marker")
    return marker


def entry_points(context):
    return list(context.config["source"]["entry_points"])


def newline_report(repository_root, logicals, *, pinned=REPLAY_LF_PINNED):
    """CRLF state of every frozen file, refused only where a narrow LF rule exists.

    The binding guarantee for a pre-existing source is that its worktree bytes
    equal its committed bytes, which is checked separately. Demanding LF from files
    this campaign did not introduce would either fail on a legacy source that is
    deliberately ``-text``, or push this campaign into renormalizing history it has
    no business touching. New files, whose ``.gitattributes`` rules are added
    before any digest is taken, are held to LF.
    """
    observed, offending = {}, []
    legacy = set(support.LEGACY_RAW_SOURCE_FILES) | set(LEGACY_RAW_INITIALIZERS)
    for logical in sorted(set(logicals)):
        target = Path(repository_root) / logical
        crlf = b"\r\n" in target.read_bytes()
        observed[logical] = "crlf" if crlf else "lf"
        if crlf and logical in set(pinned) and logical not in legacy:
            offending.append(logical)
    require(not offending,
            f"These new frozen files carry CRLF bytes: {offending}. Their narrow .gitattributes "
            "eol=lf rules exist so a clone on another platform reproduces the digests; a CRLF "
            "checkout hashes differently while `git status` stays clean.")
    return {"newlines": observed, "lf_pinned": sorted(set(pinned)),
            "legacy_raw": sorted(legacy),
            "note": ("pre-existing sources keep whatever bytes they were committed with; the "
                     "worktree-versus-HEAD comparison is what makes them reproducible here. The "
                     "three package initializers hold CRLF on disk and carry narrow -text rules "
                     "so Git stores exactly those bytes; they are preserved, never renormalized "
                     "and never edited by this screen.")}


def preflight_binding(context, closure):
    """What a preflight result is evidence *about*. Re-derived and compared at freeze."""
    screen = context.config["screen"]
    return {
        "source_sha256": paths.digest_document(
            {logical: paths.sha256_file(context.repository_root / logical)
             for logical in closure["files"]}),
        "config_sha256": context.config_sha256,
        "audit_published": paths.digest_document(context.audit["published"]),
        "screen": paths.digest_document({
            "tasks": list(screen["tasks"]), "lambdas": list(screen["replay_lambdas"]),
            "parent_seeds": list(screen["parent_seeds"]),
            "chosen_per_update": int(screen["chosen_per_update"]),
            "microbatch_rows": int(screen["microbatch_rows"]),
            "endpoint_updates": list(screen["endpoint_updates"])}),
        "tolerances": paths.digest_document(dict(context.config["tolerances"])),
        "environment": paths.digest_document(support.environment_record())}


def run_freeze(context, *, allow_missing_preflight=False):
    """Verify the committed sources, config, evidence and inputs, then write the marker.

    No commit, no dirty-tree bypass, and no bank digests: the banks are generated
    after this marker exists, under the rule it binds, and are bound to it
    afterwards by :mod:`her2_replay_banks`.

    ``allow_missing_preflight`` defaults to ``False`` and the CLI never passes it:
    in production a freeze without a completed preflight is refused. It exists so a
    fixture can exercise the rest of the freeze without a GPU, and
    :mod:`test_her2_replay_spec` asserts that the CLI does not reach for it.
    """
    marker_path = freeze_marker_path(context)
    require(not marker_path.is_file(),
            f"{marker_path} already exists and is not replaced. A freeze names the commit that "
            "fixed this screen; re-freezing over it would re-point fitted artifacts at a different "
            "specification. A different specification is a new run directory.")
    state = support.git_state(context.repository_root)
    require(not state["dirty"],
            "The working tree is dirty:\n  " + "\n  ".join(state["dirty_entries"][:20])
            + "\nThe freeze records a commit somebody reviewed. Commit or stash first.")
    closure = source_closure(context.repository_root, entry_points(context))
    evidence_root = context.config["evidence_root"]
    evidence_files = sorted(
        context.relative(path)
        for path in (context.repository_root / evidence_root).rglob("*") if path.is_file())
    require(evidence_files,
            f"{evidence_root} holds no evidence; run the prepare stage and commit its output "
            "before freezing")
    # Evidence is local-only research material and is authenticated against
    # committed digests rather than by being tracked; the sources and the config
    # are code and stay tracked. Requiring evidence at HEAD would mean publishing
    # it to be able to freeze it.
    evidence_manifest = support.require_evidence_matches_manifest(
        context.repository_root, evidence_files, context.config["campaign_id"])
    tracked = sorted(set(closure["files"])
                     | {context.relative(context.config_path)})
    untracked = [logical for logical in tracked
                 if not support.git_tracked(context.repository_root, logical)]
    require(not untracked,
            f"These files are not tracked at HEAD and cannot be frozen: {untracked}. An untracked "
            "new module would be part of the computation and absent from its identity.")
    newlines = newline_report(context.repository_root, tracked)
    drifted = support.worktree_matches_head(context.repository_root, tracked)
    require(not drifted,
            "These frozen files differ from their committed bytes at HEAD: "
            + canonical_json(drifted).strip()
            + " A checkout that re-expands newlines produces different digests from the same "
              "commit. Add a narrow .gitattributes eol rule and re-checkout.")

    preflight_path = context.run.path(PREFLIGHT_JSON)
    binding = preflight_binding(context, closure)
    preflight_record = None
    if preflight_path.is_file():
        preflight = paths.read_json(preflight_path)
        require(preflight.get("status") == "completed",
                "The recorded preflight did not complete; the freeze binds a finished probe")
        recorded = preflight.get("binding") or {}
        differing = sorted(key for key in set(recorded) | set(binding)
                           if recorded.get(key) != binding.get(key))
        require(not differing,
                f"The recorded preflight was run against different {differing}. Re-run the "
                "preflight stage against the sources, config and screen being frozen.")
        preflight_record = {"path": context.run.logical(PREFLIGHT_JSON),
                            "sha256": paths.sha256_file(preflight_path),
                            "binding": dict(binding),
                            "native": preflight.get("native"),
                            "gradient_control": preflight.get("gradient_control")}
    else:
        require(allow_missing_preflight,
                f"{preflight_path} is absent. A native inference-only parity probe on the real "
                "parents and a synthetic-weight gradient control on the native architecture are "
                "both part of this freeze; neither is optional and neither is a fit.")

    manifest_path = context.repository_root / context.config["input_manifest"]
    require(manifest_path.is_file(), f"{manifest_path} is absent; run the prepare stage first")
    input_manifest = paths.read_json(manifest_path)
    inputs = input_manifest["inputs"]
    input_hashes = {}
    for logical, entry in sorted(inputs.items()):
        observed = input_hash(context, entry)
        require(observed == entry["sha256"],
                f"{logical} hashes {observed}, the committed input manifest recorded "
                f"{entry['sha256']}")
        input_hashes[logical] = entry["sha256"]

    screen = context.config["screen"]
    marker = {
        "schema_version": REPLAY_SCHEMA, "record_kind": "training_spec_frozen",
        "campaign_id": context.config["campaign_id"], "protocol": context.config["protocol"],
        "frozen_at": paths.utc_now(),
        "git": {"commit": state["commit"], "dirty": False,
                "blobs": {logical: support.git_blob(context.repository_root, logical)
                          for logical in tracked},
                "head_bytes_sha256": {
                    logical: support.head_blob_sha256(context.repository_root, logical)
                    for logical in tracked}},
        "source": {"closure": closure,
                   "sha256": {logical: paths.sha256_file(context.repository_root / logical)
                              for logical in closure["files"]},
                   "newlines": newlines},
        "config": {"path": context.relative(context.config_path),
                   "sha256": context.config_sha256, "digest": context.config_digest,
                   "resolved": context.config},
        "evidence_manifest": evidence_manifest,
        "evidence_sha256": {logical: paths.sha256_file(context.repository_root / logical)
                            for logical in evidence_files},
        "inputs": dict(sorted(inputs.items())), "input_sha256": input_hashes,
        "input_count": len(input_hashes),
        "input_manifest": {"path": context.config["input_manifest"],
                           "sha256": paths.sha256_file(manifest_path),
                           "identity_sha256": (input_manifest.get("streams") or {}).get(
                               "identity_sha256")},
        "preflight": preflight_record,
        "audit": dict(context.audit),
        "screen": dict(screen),
        "optimization": dict(context.config["optimization"]),
        "gate": dict(context.config["gate"]),
        "banks": dict(context.config["banks"]),
        "tolerances": dict(context.config["tolerances"]),
        "environment": support.environment_record(),
        "capacity": capacity_record(context.run.run_root,
                                    minimum_bytes=int(context.config["storage"]
                                                      ["min_free_bytes"]),
                                    required=False),
        "immutable_fields": sorted(context.config["screen"]["immutable_fields"]),
        "banks_note": ("the replay and monitoring banks do not exist yet. This marker binds their "
                       "sampling rule, seeds, counts and dtypes; the banks stage generates them "
                       "afterwards and binds their completed draw-order and content digests to "
                       "this marker in banks_manifest.json."),
        "note": ("this marker is ignored by Git on purpose: it names the commit, it is not part of "
                 "it. This CLI never commits, and fitting refuses to run without it.")}
    paths.write_json(marker_path, marker)
    return marker


def input_hash(context, entry):
    root = context.roots.get(entry["root"]) if entry.get("root") else None
    target = (root.path(entry["logical_path"]) if root is not None
              else context.repository_root / entry["logical_path"])
    return paths.sha256_file(target) if target.is_file() else None


def runtime_contract(marker):
    """The device, dtypes and dependency versions the freeze recorded."""
    resolved = ((marker.get("config") or {}).get("resolved") or {})
    inference = dict(resolved.get("inference") or {})
    environment = dict(marker.get("environment") or {})
    closure = ((marker.get("source") or {}).get("closure") or {})
    return {"device": inference.get("device"),
            "forward_dtype": inference.get("forward_dtype"),
            "reduction_dtype": inference.get("reduction_dtype"),
            "environment": {name: environment.get(name) for name in RUNTIME_FIELDS},
            "external_versions": dict(closure.get("external_versions") or {})}


def require_runtime_contract(marker, *, device=None, label="fit"):
    """Check the recorded runtime once, before the expensive work, never per update.

    A freeze that pinned the sources and the inputs but let the run happen on a
    different device or a different torch would attribute numbers produced by one
    runtime to a specification that recorded another. The check is deliberately
    cheap and deliberately once: it is a precondition, not a loop invariant.

    The device is the sharp case. ``--device`` is an operational flag, and a run
    that quietly accepted ``cpu`` while the frozen config says ``cuda`` would go on
    citing the config's native probe for numbers a different backend produced.
    """
    contract = runtime_contract(marker)
    differing = []
    if device is not None and contract["device"] is not None and str(device) != contract["device"]:
        differing.append({"field": "device", "frozen": contract["device"],
                          "observed": str(device)})
    observed = support.environment_record()
    for field in RUNTIME_FIELDS:
        expected = contract["environment"].get(field)
        current = observed.get(field)
        if expected is not None and str(expected) != str(current):
            differing.append({"field": field, "frozen": expected, "observed": current})
    from importlib import metadata
    for name, entry in sorted(contract["external_versions"].items()):
        if entry.get("kind") != "distribution" or entry.get("version") is None:
            continue
        try:
            current = metadata.version(name)
        except Exception:                                       # noqa: BLE001 - recorded below
            current = None
        if current != entry["version"]:
            differing.append({"field": f"dependency::{name}", "frozen": entry["version"],
                              "observed": current})
    require(not differing,
            f"{label}: the runtime does not match the frozen contract: "
            + canonical_json(differing).strip()
            + " The freeze is what the recorded numbers are attributed to; a device or a "
              "dependency that changed under it produces numbers from a different runtime.")
    return dict(contract, observed_device=device, checked="once, before banks or fitting")


def require_frozen_identity(context, *, check_inputs=True, device=None, label="fit"):
    """Re-hash everything the marker pinned. A mismatch stops the stage."""
    marker = read_freeze_marker(context)
    require_runtime_contract(marker, device=device, label=label)
    differing = []
    for logical, expected in sorted((marker["source"].get("sha256") or {}).items()):
        target = context.repository_root / logical
        observed = paths.sha256_file(target) if target.is_file() else None
        if observed != expected:
            differing.append({"file": logical, "expected": expected, "observed": observed})
    if context.config_sha256 != marker["config"]["sha256"]:
        differing.append({"file": marker["config"]["path"],
                          "expected": marker["config"]["sha256"],
                          "observed": context.config_sha256})
    for logical, expected in sorted((marker.get("evidence_sha256") or {}).items()):
        target = context.repository_root / logical
        observed = paths.sha256_file(target) if target.is_file() else None
        if observed != expected:
            differing.append({"file": logical, "expected": expected, "observed": observed})
    if check_inputs:
        for logical, expected in sorted((marker.get("input_sha256") or {}).items()):
            observed = input_hash(context, marker["inputs"][logical])
            if observed != expected:
                differing.append({"file": logical, "expected": expected, "observed": observed})
    require(not differing,
            "The frozen identity no longer holds: " + canonical_json(differing).strip()
            + " Fitting under changed sources, config or inputs would attribute new numbers to the "
              "frozen specification.")
    return marker


def frozen_values(marker):
    """The resolved value of every field the freeze declared immutable."""
    resolved = dict(marker.get("screen") or {})
    inference = ((marker.get("config") or {}).get("resolved") or {}).get("inference") or {}
    resolved.setdefault("device", inference.get("device"))
    return resolved


def require_immutable_fields(marker, overrides):
    """Refuse a CLI override that would change a field the freeze pinned."""
    frozen = set(marker.get("immutable_fields") or ())
    values = frozen_values(marker)
    offending = sorted(name for name, value in dict(overrides or {}).items()
                       if value is not None and name in frozen and value != values.get(name))
    require(not offending,
            f"These frozen fields cannot be overridden at fit time: {offending}. The freeze is "
            "what the recorded numbers are attributed to; a run that silently changed the device, "
            "the microbatch, the effective batch or a seed would not be the reviewed run.")
    return True
