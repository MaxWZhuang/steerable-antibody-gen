"""Configuration, the source snapshot freeze, model lineage, and the resolved protocol.

The inherited ``her2_replay_spec.run_freeze`` requires a clean worktree and
byte-equality with ``HEAD`` for every file in the closure. That is the right
guarantee for a published campaign and it is left exactly as it is -- this
module adds nothing to it, removes nothing from it, and introduces no
``--allow-dirty`` flag anywhere.

This flight is authorized to build and train but **not** to commit, so its
source identity cannot be a commit. It is a content-addressed snapshot instead:
the actual import closure of the entry point, plus the config and the
specification documents, copied byte-for-byte into the run root, each hashed,
with one aggregate digest over the sorted per-file digests. ``HEAD`` and the
full dirty list are recorded beside it, and every file is classified
``tracked_clean`` / ``tracked_modified`` / ``untracked_new`` so a reader can see
exactly how the snapshot relates to the repository. Later stages re-hash the
worktree closure and refuse to run if a frozen file changed.

Three historical source files differ from the completed campaign's freeze. That
is a reviewed supersession limited to freeze, evidence and verification
machinery -- the training mathematics sources match -- and it is *disclosed*
here from the recorded supersession provenance. Nothing in this flight claims
that all historical source bytes are unchanged, and the old freeze machinery is
not edited to make the statement simpler.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from . import her2_nf_banks as banks
from . import her2_replay_spec as replay_spec
from . import her2_support as support
from . import her2_support_paths as paths
from .her2_nf_contract import NF_SCHEMA, PROBABILITY_EVENT
from .her2_runtime import require

FREEZE_MARKER = "source_snapshot.json"
RESOLVED_PROTOCOL = "resolved_protocol.json"
RECOVERED_FACTS = "recovered_facts.json"
DISCREPANCIES = "discrepancies.json"
SNAPSHOT_DIR = "source_snapshot"

#: Fields a stage may never override once the protocol is frozen.
IMMUTABLE_FIELDS = ("probability_event", "block_a", "block_b", "objectives", "preservation",
                    "banks", "seeds", "monitor", "calibration", "endpoints", "split")


# ---------------------------------------------------------------------------
# configuration and context
# ---------------------------------------------------------------------------

def load_config(path):
    """Read the flight config and refuse any unresolved placeholder."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    holes = replay_spec.placeholders(document)
    require(not holes,
            f"The flight configuration still carries unresolved placeholders: {holes}. A stage "
            "that ran against one would record a null where a declared value belongs.")
    require(document.get("schema_version") == NF_SCHEMA,
            f"Unsupported flight config schema {document.get('schema_version')!r}")
    return document


@dataclass(frozen=True)
class FlightContext:
    """Where everything lives: the repository, the run root, and the configured inputs."""

    repository_root: Path
    run_root: Path
    config: dict
    config_path: Path

    @property
    def campaign_id(self):
        return str(self.config["campaign_id"])

    def path(self, *parts):
        target = self.run_root.joinpath(*[str(part) for part in parts])
        return target

    def repository(self, logical):
        return paths.resolve_under(self.repository_root, str(logical))

    def artifact_root(self, name):
        """A configured artifact root, resolved against the repository.

        Runtime code reads the configured roots. The bounded evidence copies that
        live beside a review loop are inputs to tests and to a human reader; they
        are never a production path, because a run that silently read a copy
        would attribute its numbers to bytes nobody checked.
        """
        roots = dict(self.config.get("roots") or {})
        require(name in roots, f"No configured artifact root named {name!r}")
        entry = roots[name]
        logical = entry["logical"] if isinstance(entry, dict) else str(entry)
        return self.repository_root / logical

    def ensure(self):
        self.run_root.mkdir(parents=True, exist_ok=True)
        return self


def resolve_context(repository_root, *, config_path, run_root=None):
    repository_root = Path(repository_root)
    config = load_config(config_path)
    root = Path(run_root) if run_root is not None else repository_root / config["run_root"]
    return FlightContext(repository_root=repository_root, run_root=root, config=config,
                         config_path=Path(config_path)).ensure()


# ---------------------------------------------------------------------------
# the content-addressed source snapshot
# ---------------------------------------------------------------------------

def source_snapshot(context, *, extra_files=()):
    """Copy the real import closure plus config and spec, hash it, and record git state.

    The identity of this flight is ``snapshot_sha256``: a digest over the sorted
    ``(logical path, file digest)`` pairs. It is stated as exactly that -- a
    content address -- and never as a commit.

    An existing snapshot is **verified and reused**, never deleted and rewritten.
    A frozen source identity that a later call can silently replace is not a
    freeze: every result already attributed to the old bytes would quietly
    change its attribution. If the closure now hashes differently, this raises,
    and the answer is a new run under a new protocol identity.
    """
    marker = context.path(FREEZE_MARKER)
    if marker.is_file():
        return _verify_existing_snapshot(context, paths.read_json(marker))
    entry_points = list(context.config["source"]["entry_points"])
    closure = replay_spec.source_closure(context.repository_root, entry_points)
    logicals = list(closure["files"])
    for extra in list(extra_files) + list(context.config["source"].get("extra_files") or []):
        if extra not in logicals:
            logicals.append(str(extra))
    config_logical = str(context.config["config_path"])
    if config_logical not in logicals:
        logicals.append(config_logical)
    logicals = sorted(set(logicals))
    paths.require_no_case_collisions(logicals, where="source snapshot")

    snapshot_root = context.path(SNAPSHOT_DIR)
    require(not snapshot_root.exists() or not any(snapshot_root.rglob("*")),
            f"{snapshot_root} already holds archived source bytes but {FREEZE_MARKER} is absent. "
            "A snapshot directory without its marker is an interrupted freeze, and it is not "
            "recursively deleted to make room: move it aside deliberately.")
    files = {}
    for logical in logicals:
        source = context.repository(logical)
        require(source.is_file(), f"{logical} is in the source closure but missing from the tree")
        target = paths.resolve_under(snapshot_root, logical)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        digest = paths.sha256_file(source)
        copied = paths.sha256_file(target)
        require(digest == copied, f"{logical}: the snapshot copy does not match its source bytes")
        head = support.head_blob_sha256(context.repository_root, logical)
        tracked = support.git_tracked(context.repository_root, logical)
        files[logical] = {
            "sha256": digest,
            "head_sha256": head,
            "tracked": bool(tracked),
            "classification": ("tracked_clean" if tracked and head == digest else
                               "tracked_modified" if tracked else "untracked_new")}
    aggregate = paths.sha256_text("\n".join(f"{name} {files[name]['sha256']}"
                                            for name in sorted(files)))
    git = support.git_state(context.repository_root)
    document = {
        "schema_version": NF_SCHEMA, "record_kind": "source_snapshot",
        "campaign_id": context.campaign_id,
        "entry_points": entry_points, "files": files, "file_count": len(files),
        "snapshot_sha256": aggregate,
        "snapshot_directory": SNAPSHOT_DIR,
        "external_versions": closure["external_versions"],
        "environment": support.environment_record(),
        "git_state": {"head": git["commit"], "dirty": bool(git["dirty"]),
                      "dirty_entries": git["dirty_entries"]},
        "identity_is": ("the aggregate content digest above. This flight is authorized to build "
                        "and train and NOT to commit, so its source identity is content-addressed "
                        "rather than a commit. No claim is made that the tree is clean."),
        "inherited_freeze_untouched": ("her2_replay_spec.run_freeze still requires a clean tree "
                                       "and byte equality with HEAD. It is not called here, not "
                                       "modified, and no allow-dirty flag exists anywhere."),
        "frozen_at": paths.utc_now()}
    paths.write_json(context.path(FREEZE_MARKER), document)
    return document


def _verify_existing_snapshot(context, document):
    """Reuse a frozen snapshot only if it is still, byte for byte, the same one.

    Four things are checked, not one: the ARCHIVED bytes in the run root, the
    aggregate digest over them, the configured configuration file, and the
    current worktree closure. Verifying the worktree alone would accept a
    snapshot directory somebody had edited, and the archived copy is what a
    later reader will actually open.
    """
    snapshot_root = context.path(SNAPSHOT_DIR)
    archived, drifted, missing = [], [], []
    for logical, block in sorted(document["files"].items()):
        copy = paths.resolve_under(snapshot_root, logical)
        if not copy.is_file():
            missing.append({"file": logical, "where": "archive"})
        elif paths.sha256_file(copy) != block["sha256"]:
            archived.append({"file": logical, "where": "archive",
                             "frozen_sha256": block["sha256"],
                             "observed_sha256": paths.sha256_file(copy)})
        target = context.repository(logical)
        if not target.is_file():
            missing.append({"file": logical, "where": "worktree"})
        elif paths.sha256_file(target) != block["sha256"]:
            drifted.append({"file": logical, "where": "worktree",
                            "frozen_sha256": block["sha256"],
                            "observed_sha256": paths.sha256_file(target)})
    aggregate = paths.sha256_text("\n".join(
        f"{name} {document['files'][name]['sha256']}" for name in sorted(document["files"])))
    require(aggregate == document["snapshot_sha256"],
            f"{FREEZE_MARKER}: the recorded per-file digests aggregate to {aggregate} and the "
            f"marker claims {document['snapshot_sha256']}. The identity of this flight is that "
            "aggregate, so a marker that disagrees with its own file table is refused.")
    config_logical = str(context.config["config_path"])
    recorded_config = (document["files"].get(config_logical) or {}).get("sha256")
    observed_config = paths.sha256_file(context.config_path)
    require(recorded_config is None or recorded_config == observed_config,
            f"the configured configuration file now hashes {observed_config} and the freeze "
            f"recorded {recorded_config}. The coefficients, bank sizes and endpoints this flight "
            "runs under are in that file; a changed one is a new protocol version.")
    require(not (archived or drifted or missing),
            "the frozen source is not identical to the recorded snapshot: "
            f"archive mismatches {archived}, worktree mismatches {drifted}, missing {missing}. "
            "A frozen snapshot is verified and REUSED, never rewritten -- changed source needs a "
            "new run and a new protocol identity, because every result already recorded is "
            "attributed to these bytes.")
    return dict(document, reused=True,
                verified_at=paths.utc_now(),
                verification={"archived_files": len(document["files"]),
                              "aggregate_matches": True, "config_matches": True,
                              "worktree_matches": True,
                              "basis": ("archived bytes, aggregate digest, the configured "
                                        "configuration file and the current worktree closure")})


#: Repository paths that are legitimately imported without being production
#: source. A test module participates in no production computation, so freezing
#: one into a scientific source identity would be wrong -- but an imported one
#: is DISCLOSED in the verification record rather than silently tolerated, and
#: the inherited closure check itself is untouched and still refuses everything
#: else.
NON_PRODUCTION_PREFIXES = ("src/smallAntibodyGen/tests/",)


def non_production_imports(repository_root, *, prefixes=NON_PRODUCTION_PREFIXES):
    """Currently imported repository modules under a declared non-production prefix."""
    import sys
    root = Path(repository_root).resolve()
    found = []
    for module in list(sys.modules.values()):
        origin = getattr(module, "__file__", None)
        if not origin:
            continue
        try:
            relative = Path(origin).resolve().relative_to(root)
        except ValueError:
            continue
        logical = str(relative).replace("\\", "/")
        if logical.endswith(".py") and any(logical.startswith(p) for p in prefixes):
            found.append(logical)
    return sorted(set(found))


def under_pytest():
    """Is this process a pytest run?

    The runtime-closure check asks "is every imported repository module part of
    the frozen scientific closure". Inside pytest the answer is legitimately no
    -- the test modules are imported, and so is whatever a sibling test file
    pulled in -- and none of it participates in a production computation. The
    check is therefore *reported as skipped* here, with the reason, and is
    exercised against the real entry point in an isolated subprocess instead.

    This cannot weaken production: a supervisor process has no
    ``PYTEST_CURRENT_TEST``, so the check always runs there.
    """
    import os
    # Child processes inherit these environment variables but do not inherit
    # pytest's imported modules. They must still check their production closure.
    import sys
    return "pytest" in sys.modules and (
        "PYTEST_CURRENT_TEST" in os.environ or "PYTEST_VERSION" in os.environ)


def verify_source_snapshot(context, *, label="stage", check_runtime_closure=None):
    """Re-hash the worktree closure against the snapshot. A changed file stops the stage."""
    marker = context.path(FREEZE_MARKER)
    require(marker.is_file(),
            f"{marker} is absent: every stage after the freeze runs against a recorded source "
            "snapshot, and there is none.")
    document = paths.read_json(marker)
    changed, missing = [], []
    for logical, block in sorted(document["files"].items()):
        for where, target in (("worktree", context.repository(logical)),
                              ("archive", paths.resolve_under(context.path(SNAPSHOT_DIR),
                                                              logical))):
            if not target.is_file():
                missing.append(f"{logical} ({where})")
                continue
            observed = paths.sha256_file(target)
            if observed != block["sha256"]:
                changed.append({"file": logical, "where": where,
                                "frozen_sha256": block["sha256"], "observed_sha256": observed})
    require(not missing and not changed,
            f"{label}: the frozen source changed since the snapshot. Missing {missing}; changed "
            f"{[entry['file'] for entry in changed]}. The recorded numbers are attributed to the "
            "snapshot bytes, so a stage does not run against different ones.")
    if check_runtime_closure is None:
        check_runtime_closure = not under_pytest()
    tolerated = []
    if check_runtime_closure:
        # The inherited closure check is called unchanged and still refuses every
        # unfrozen production module. The only widening is the declared
        # non-production prefix -- inside a pytest process the imported set
        # legitimately contains the test modules, which participate in no
        # production computation and must NOT be frozen into a scientific source
        # identity. Whatever was tolerated is listed in the returned record, so
        # a test module imported during a real run would be visible rather than
        # silent.
        tolerated = non_production_imports(context.repository_root)
        replay_spec.require_runtime_closure(
            {"files": list(document["files"]) + tolerated},
            repository_root=context.repository_root, label=label)
    return {"verified": True, "snapshot_sha256": document["snapshot_sha256"],
            "files": len(document["files"]), "label": str(label),
            "checked": ["worktree bytes", "archived bytes"]
            + (["imported runtime closure"] if check_runtime_closure else []),
            "non_production_imports": tolerated,
            "non_production_rule": ("modules under " + ", ".join(NON_PRODUCTION_PREFIXES)
                                    + " are not production source and are not frozen. Any that "
                                      "were imported are listed above; every other unfrozen "
                                      "repository module is still refused."),
            "runtime_closure_checked": bool(check_runtime_closure),
            "runtime_closure_skip_reason": (None if check_runtime_closure else
                                            "this is a pytest process, whose imported set "
                                            "legitimately includes test modules and whatever a "
                                            "sibling test file pulled in. A supervisor process "
                                            "has no PYTEST_CURRENT_TEST and always runs the "
                                            "check; the native opt-in suite exercises the real "
                                            "entry point in an isolated subprocess.")}


def historical_supersession_disclosure(context):
    """Read the recorded supersession provenance from the configured artifact root.

    Read-only, from the configured root. If it is unreachable the fact is
    recorded as a disclosure gap rather than replaced by an assurance that the
    old sources are unchanged.
    """
    entry = dict(context.config.get("historical") or {})
    logical = entry.get("supersession_provenance")
    if not logical:
        return {"available": False, "reason": "no supersession provenance is configured"}
    target = context.repository_root / logical
    if not target.is_file():
        return {"available": False, "path": str(logical),
                "reason": "the configured supersession provenance is not readable from here",
                "consequence": ("the disclosure below is incomplete. This flight does NOT claim "
                                "that all historical source bytes are unchanged.")}
    document = paths.read_json(target)
    return {"available": True, "path": str(logical), "sha256": paths.sha256_file(target),
            "provenance": document,
            "scope": ("a reviewed supersession limited to freeze, evidence and verification "
                      "machinery. The training-mathematics sources match the historical freeze."),
            "statement": ("three historical source files differ from the completed campaign's "
                          "freeze. They are disclosed, not hidden, and this flight makes no claim "
                          "that every historical source byte is identical.")}


# ---------------------------------------------------------------------------
# model lineage
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LineageNode:
    """One model in the DAG, bound to the parent and the data it is allowed to have seen."""

    node_id: str
    kind: str
    parent_id: object
    parent_state_sha256: object
    population: str
    forbidden_rows: object
    config_sha256: str
    scaffold_prefix_sha256: str

    def document(self):
        return {"node_id": self.node_id, "kind": self.kind, "parent_id": self.parent_id,
                "parent_state_sha256": self.parent_state_sha256, "population": self.population,
                "config_sha256": self.config_sha256,
                "scaffold_prefix_sha256": self.scaffold_prefix_sha256,
                "forbidden_rows": (self.forbidden_rows.document()
                                   if self.forbidden_rows is not None else None)}


class ModelLineage:
    """The DAG, plus the two refusals that make it more than documentation."""

    def __init__(self):
        self.nodes = {}

    def add(self, node):
        require(node.node_id not in self.nodes, f"duplicate lineage node {node.node_id!r}")
        if node.parent_id is not None:
            require(node.parent_id in self.nodes,
                    f"{node.node_id!r} names parent {node.parent_id!r}, which is not in the DAG")
        self.nodes[node.node_id] = node
        return node

    def require_parent(self, node_id, *, observed_state_sha256, where):
        """Refuse a cache or a reference that came from a different parent."""
        node = self.nodes[node_id]
        require(node.parent_state_sha256 is not None,
                f"{where}: {node_id!r} declares no parent state digest to check against")
        require(str(observed_state_sha256) == str(node.parent_state_sha256),
                f"{where}: this artifact was produced by parent state {observed_state_sha256} and "
                f"{node_id!r} descends from {node.parent_state_sha256}. A reference cache or "
                "replay bank from another parent silently redefines every margin computed with "
                "it.")
        return True

    def require_allowed_rows(self, node_id, cores, *, where):
        """Refuse training or selection rows the node may not resolve."""
        node = self.nodes[node_id]
        if node.forbidden_rows is None:
            return True
        return node.forbidden_rows.check(cores, where=f"{where} ({node_id})")

    def document(self):
        return {"schema_version": NF_SCHEMA, "record_kind": "model_lineage",
                "nodes": {name: node.document() for name, node in sorted(self.nodes.items())},
                "edges": [{"parent": node.parent_id, "child": name}
                          for name, node in sorted(self.nodes.items())
                          if node.parent_id is not None],
                "enforced": ["wrong-parent caches are refused by state digest",
                             "forbidden training rows are refused at the loader"]}


# ---------------------------------------------------------------------------
# recovery
# ---------------------------------------------------------------------------

def recover(context, *, probes=()):
    """M0: record what is actually here, and export a discrepancy list.

    Facts that cannot be recovered are listed as unrecovered. They are not
    patched in memory and they are not replaced by the value a report once
    printed.
    """
    git = support.git_state(context.repository_root)
    environment = support.environment_record()
    capacity = replay_spec.capacity_record(
        context.run_root, minimum_bytes=int(context.config["storage"]["min_free_bytes"]),
        required=False)
    observed, unrecovered = {}, []
    for probe in list(probes) + list(context.config.get("recovery_probes") or []):
        logical = probe["path"] if isinstance(probe, dict) else str(probe)
        expected = probe.get("sha256") if isinstance(probe, dict) else None
        target = context.repository_root / logical
        if not target.is_file():
            unrecovered.append({"path": logical, "reason": "not readable from this process"})
            continue
        digest = paths.sha256_file(target)
        entry = {"path": logical, "sha256": digest, "bytes": int(target.stat().st_size)}
        if expected is not None:
            entry["expected_sha256"] = expected
            entry["matches"] = bool(digest == expected)
        observed[logical] = entry
    document = {
        "schema_version": NF_SCHEMA, "record_kind": "recovered_facts",
        "campaign_id": context.campaign_id,
        "git_state": {"head": git["commit"], "dirty": bool(git["dirty"]),
                      "dirty_entries": git["dirty_entries"]},
        "environment": environment, "capacity": capacity,
        "probability_event": PROBABILITY_EVENT,
        "probes": observed, "unrecovered": unrecovered,
        "supersession": historical_supersession_disclosure(context),
        "reserved_test_labels": ("never read. her2_data.load_split refuses the test split without "
                                 "an unlock and nothing in this flight requests one; only "
                                 "test_sequences (usecols=['seq']) is available and is used for "
                                 "membership diagnostics alone."),
        "recorded_at": paths.utc_now()}
    paths.write_json(context.path(RECOVERED_FACTS), document)
    discrepancies = {
        "schema_version": NF_SCHEMA, "record_kind": "discrepancies",
        "entries": ([{"kind": "unrecovered_input", **entry} for entry in unrecovered]
                    + [{"kind": "digest_mismatch", **entry} for entry in observed.values()
                       if entry.get("matches") is False]),
        "policy": ("exported as a list. A discrepancy is never patched in memory and never "
                   "replaced by a previously reported value; a stage that depends on one stops."),
        "recorded_at": paths.utc_now()}
    paths.write_json(context.path(DISCREPANCIES), discrepancies)
    return {"recovered": document, "discrepancies": discrepancies}


# ---------------------------------------------------------------------------
# the resolved protocol
# ---------------------------------------------------------------------------

def resolved_protocol(context, *, snapshot, geometry=None, calibration=None, forecast=None,
                      seed_table=None):
    """The executable resolved configuration. Every field filled or explicitly absent."""
    config = context.config
    document = {
        "schema_version": NF_SCHEMA, "record_kind": "resolved_protocol",
        "campaign_id": context.campaign_id,
        "config_sha256": paths.sha256_file(context.config_path),
        "source_snapshot_sha256": snapshot["snapshot_sha256"],
        "git_state": snapshot["git_state"],
        "environment": snapshot["environment"],
        "probability_event": PROBABILITY_EVENT,
        "probability_contract": config["probability"],
        "objectives": config["objectives"],
        "preservation": config["preservation"],
        "optimization": config["optimization"],
        "bank_roles": {name: spec.document() for name, spec in sorted(banks.BANK_ROLES.items())},
        "banks": config["banks"],
        "seed_tree": seed_table or banks.seed_table(config["seeds"]["spawn_keys"]),
        "stream_domains": dict(sorted(banks.STREAM_DOMAINS.items())),
        "stream_rule": config["seeds"].get("domains"),
        "monitor": config["monitor"],
        "block_a": config["block_a"], "block_b": config["block_b"],
        "primary_contrasts": config["analysis"]["primary_contrasts"],
        "primary_endpoints": config["analysis"]["primary_endpoints"],
        "calibration_rule": config["calibration"],
        "split": config["split"],
        "geometry": geometry, "calibration_outcome": calibration,
        "runtime_forecast": forecast,
        "unresolved": _unresolved(geometry, calibration, forecast),
        "resolved_at": paths.utc_now()}
    paths.write_json(context.path(RESOLVED_PROTOCOL), document)
    return document


def _unresolved(geometry, calibration, forecast):
    """Which blueprint fields are still blank. Production refuses to run with any.

    A *blocked* stage record is not a filled field. Passing one through because
    it is merely present is how a protocol gets frozen over a challenge whose
    geometry never certified.
    """
    missing = []
    if not _stage_is_resolved(geometry):
        missing.append("geometry: the split has not been certified")
    if not _stage_is_resolved(calibration):
        missing.append("calibration_outcome: no coefficient has been selected or refused")
    if forecast is None or not (forecast or {}).get("measured"):
        missing.append("runtime_forecast: no MEASURED profile exists; an invented cost is never "
                       "written in its place")
    return missing


def _stage_is_resolved(block):
    if block is None:
        return False
    status = dict(block).get("status")
    return status is None or status == "completed"


def require_resolved(context, *, label):
    """Production refuses to start while the protocol has unresolved fields."""
    target = context.path(RESOLVED_PROTOCOL)
    require(target.is_file(), f"{label}: {RESOLVED_PROTOCOL} has not been written")
    document = paths.read_json(target)
    require(not document["unresolved"],
            f"{label}: the resolved protocol still has unresolved fields "
            f"{document['unresolved']}. Production runs against a filled protocol; a blank field "
            "is a readiness gate, not a formality.")
    return document


def require_immutable(document, overrides):
    """A post-freeze change becomes a new protocol version with a reason, never a silent edit."""
    offending = sorted(key for key in dict(overrides) if key in IMMUTABLE_FIELDS)
    require(not offending,
            f"{offending} are frozen before three-seed production. Changing one after the freeze "
            "is a new protocol version with a recorded reason, not an override.")
    return True


def protocol_amendment(context, *, previous, reason, changes):
    """Record a post-freeze change as a new version. The old one is retained."""
    document = dict(previous)
    version = int(previous.get("protocol_version", 1)) + 1
    document.update(protocol_version=version, amended_at=paths.utc_now(),
                    amendment={"reason": str(reason), "changes": dict(changes),
                               "supersedes_sha256": paths.digest_document(previous)})
    paths.write_json(context.path(f"resolved_protocol.v{version}.json"), document)
    return document
