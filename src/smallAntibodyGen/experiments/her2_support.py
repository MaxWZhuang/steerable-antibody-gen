"""Stage coordinator for the HER2 support-preservation audit.

One module owns the order of operations, because the order is the protocol:
``inventory -> prepare -> preflight -> freeze -> score -> ches -> report -> decide``,
with ``verify`` available at any point to re-check what has already been written.

The contract between the stages:

* **Nothing scientific is scored before the freeze.** ``inventory`` and ``prepare``
  verify artifacts and write the portable evidence a reviewer commits; ``preflight``
  runs bounded numerical probes on new code only. ``score`` and ``ches`` refuse to
  start without ``audit_spec_frozen`` and re-hash the pinned sources, config and
  inputs it names. There is no ``--allow-dirty`` and no placeholder to fill in later.
* **This CLI never commits.** The freeze stage verifies an *existing* commit and
  writes an ignored marker pointing at the real HEAD. A tool that committed its own
  inputs would be hashing a tree it had just created.
* **A rerun verifies; it does not refresh.** Completed shards carry content digests
  and a timing block, and re-running compares them. Changing either means a new
  revision directory, not a mutated artifact with a newer elapsed time.
* **Missing is missing.** Every absent artifact, unverifiable identity or
  incomplete stage travels to the decision as missing coverage. It never becomes a
  negative preservation finding.
"""
from __future__ import annotations

import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .her2_runtime import canonical_json, require
from . import her2_ches as ches_lib
from . import her2_support_inventory as inventory_lib
from . import her2_support_paths as paths
from . import her2_support_scoring as scoring

STAGES = ("inventory", "prepare", "preflight", "freeze", "score", "ches", "report", "decide",
          "verify")
#: Stages whose measured cost describes the measurement itself. ``report``, ``decide``
#: and ``verify`` are excluded from the published cost table on purpose: a report that
#: quoted its own elapsed time would change every time it was rebuilt.
COSTED_STAGES = ("inventory", "prepare", "preflight", "freeze", "score", "ches")
#: Clear aliases, so an operator's shorthand cannot silently select another stage.
STAGE_ALIASES = {"inventory-only": "inventory", "verify-only": "verify", "verification": "verify",
                 "freeze-spec": "freeze", "decision": "decide", "publish": "report"}

#: Source files whose bytes define what this audit computes. Hashed at freeze and
#: re-hashed by every gated stage. Documentation and the report are deliberately
#: absent: a wording fix must not invalidate a scored artifact, a formula change must.
#:
#: The list is not limited to the new modules. The number this audit publishes comes
#: out of ``CorePolicy``, the core encoding, the pairing and the trajectory schema as
#: much as it comes out of the code written for the audit, so those files are part of
#: its identity too. They are *pinned*, never modified: a change in any of them means
#: the scored artifacts describe a different computation and the freeze must be redone.
#:
#: The rule is the **module-level import closure** of the files this audit executes,
#: not a hand-picked shortlist. ``her2_guarded_eval.expected_trajectories`` decides
#: which trajectories exist and it resolves arm identities through ``her2_objectives``;
#: the raw root is accepted by digests parsed by ``benchmarks/provenance``. Leaving
#: either out meant a change in them could silently change what was enumerated or
#: which tree was scored while the freeze still verified.
AUDIT_SOURCE_FILES = (
    "src/smallAntibodyGen/benchmarks/provenance.py",
    "src/smallAntibodyGen/experiments/dpo.py",
    "src/smallAntibodyGen/experiments/her2_ches.py",
    "src/smallAntibodyGen/experiments/her2_data.py",
    "src/smallAntibodyGen/experiments/her2_eval.py",
    "src/smallAntibodyGen/experiments/her2_guard.py",
    "src/smallAntibodyGen/experiments/her2_guarded_eval.py",
    "src/smallAntibodyGen/experiments/her2_guarded_trajectory.py",
    "src/smallAntibodyGen/experiments/her2_lineage.py",
    "src/smallAntibodyGen/experiments/her2_objectives.py",
    "src/smallAntibodyGen/experiments/her2_policy.py",
    "src/smallAntibodyGen/experiments/her2_preferences.py",
    "src/smallAntibodyGen/experiments/her2_runtime.py",
    "src/smallAntibodyGen/experiments/her2_support.py",
    "src/smallAntibodyGen/experiments/her2_support_inventory.py",
    "src/smallAntibodyGen/experiments/her2_support_paths.py",
    "src/smallAntibodyGen/experiments/her2_support_scoring.py",
    "scripts/audit_her2_support.py",
)

# These historical files already contain CRLF/mixed endings. The -text rules
# preserve their exact existing bytes in Git, including on another platform.
LEGACY_RAW_SOURCE_FILES = (
    "src/smallAntibodyGen/benchmarks/provenance.py",
    "src/smallAntibodyGen/experiments/her2_guard.py",
)

#: Logical names inside the run root.
INVENTORY_JSON = "inventory.json"
AUDIT_CONFIG_JSON = "audit_config.json"
MANIFEST_JSON = "manifest.json"
PREFLIGHT_JSON = "preflight.json"
DECISION_JSON = "decision.json"
COMPLETE_JSON = "audit_complete.json"
COVERAGE_JSON = "summaries/coverage.json"
CHECKPOINTS_JSON = "summaries/checkpoints.json"
CHES_JSON = "summaries/ches.json"
REPORT_MD = "report/her2-support-audit.md"


# ---------------------------------------------------------------------------
# configuration and context
# ---------------------------------------------------------------------------

def load_config(path):
    """Read the audit config and refuse a launch-time placeholder."""
    path = Path(path)
    require(path.is_file(), f"Audit config {path} is missing")
    document = paths.read_json(path)
    require(document.get("schema_version") == paths.AUDIT_SCHEMA,
            f"{path} is not an {paths.AUDIT_SCHEMA} config")
    blanks = sorted(_placeholders(document))
    require(not blanks,
            f"{path} still carries launch-time placeholders at {blanks}. Every scientific setting "
            "is complete at commit; mechanically generated input bindings live in the tracked "
            "input manifest the config links to.")
    return document, paths.sha256_file(path), paths.digest_document(document)


def _placeholders(node, prefix=""):
    """Any ``TODO``/``FIXME``/``TBD``/``PLACEHOLDER`` marker left in a scientific field."""
    found = []
    if isinstance(node, dict):
        for key, value in node.items():
            found.extend(_placeholders(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(node, list):
        for position, value in enumerate(node):
            found.extend(_placeholders(value, f"{prefix}[{position}]"))
    elif isinstance(node, str) and node.strip().upper() in ("TODO", "FIXME", "TBD", "PLACEHOLDER"):
        found.append(prefix)
    return found


@dataclass
class AuditContext:
    """Everything a stage needs that does not depend on which stage it is."""

    repository_root: Path
    run: paths.RunPaths
    config: dict
    config_path: Path
    config_sha256: str
    config_digest: str
    roots: dict = field(default_factory=dict)
    historical: dict = field(default_factory=dict)

    # -- convenience -------------------------------------------------------
    @property
    def seeds(self):
        return [int(seed) for seed in self.config["populations"]["seeds"]]

    def root(self, name):
        require(name in self.roots, f"Root {name!r} was not resolved")
        return self.roots[name]

    def progress(self, stage, *, total=None, note=None, every=1):
        return paths.StageProgress(self.run.path(f"progress/{stage}.json"), stage=stage,
                                   total=total, note=note, every=every)

    def stage_status(self, stage):
        path = self.run.path(f"progress/{stage}.json")
        return paths.read_json(path) if path.is_file() else None


def resolve_context(repository_root, *, config_path, run_root=None, root_overrides=None):
    """Load the config, resolve every historical root by hash probe, and bind them."""
    repository_root = Path(repository_root)
    config, config_sha256, config_digest = load_config(config_path)
    run = paths.RunPaths.create(repository_root,
                                run_root if run_root is not None else
                                repository_root / config["run_root"],
                                logical=config["run_root"], create=False)
    local_roots = paths.load_local_roots(run.run_root) if run.run_root.is_dir() else {}
    overrides = dict(root_overrides or {})

    historical = {}
    guarded_manifest_logical = config["historical"]["guarded_manifest"]
    historical["guarded_manifest"] = paths.read_json(repository_root / guarded_manifest_logical)
    historical["launch_manifest"] = paths.read_json(
        repository_root / config["historical"]["launch_manifest"])
    historical["campaign_status"] = paths.read_json(
        repository_root / config["historical"]["campaign_status"])
    historical["guarded_config"] = paths.read_json(
        repository_root / config["historical"]["guarded_config"])
    historical["original_config"] = paths.read_json(
        repository_root / config["historical"]["original_config"])

    roots = {}
    guarded_logical = config["roots"]["guarded"]["logical"]
    guarded_anchor = guarded_logical.rsplit("/", 1)[-1]
    roots["guarded"] = paths.resolve_root(
        "guarded", logical=guarded_logical,
        candidates=paths.root_candidates(
            repository_root=repository_root, logical=guarded_logical,
            explicit=overrides.get("guarded"), local_roots=local_roots,
            extra=(historical["campaign_status"].get("physical_output"),)),
        probes=inventory_lib.guarded_root_probes(historical["guarded_manifest"],
                                                 anchor=guarded_anchor))

    # The v1 root is accepted by the digests the *verified* guarded campaign recorded
    # for the parents it inherited. The launch manifest's source files are producer
    # code, not artifacts under this root, so they cannot probe it.
    guarded_stages = {int(stage["stage"]): paths.read_json(
                          roots["guarded"].path(f"validation/stage{int(stage['stage'])}"
                                                "_endpoints.json"))
                      for stage in historical["guarded_config"]["stages"]}
    historical["guarded_stage_documents"] = guarded_stages
    original_logical = config["roots"]["original"]["logical"]
    original_anchor = original_logical.rsplit("/", 1)[-1]
    roots["original"] = resolve_original_root(
        logical=original_logical, anchor=original_anchor,
        candidates=paths.root_candidates(
            repository_root=repository_root, logical=original_logical,
            explicit=overrides.get("original"), local_roots=local_roots),
        base_relative=config["historical"]["base_selection_relative"],
        expected_parents=inventory_lib.parent_digests_from_stages(guarded_stages))

    raw_logical = config["roots"]["raw"]["logical"]
    roots["raw"] = paths.resolve_root(
        "raw", logical=raw_logical,
        candidates=paths.root_candidates(
            repository_root=repository_root, logical=raw_logical,
            explicit=overrides.get("raw"), local_roots=local_roots),
        probes=_raw_root_probes(repository_root, raw_logical, config))

    # Nothing has been written yet. Refuse a run root that is, contains or sits
    # inside any location this audit only reads -- including through the junction,
    # which is why the comparison is on fully resolved paths.
    paths.require_disjoint_run_root(run.run_root, {
        "guarded_campaign": roots["guarded"].local_path,
        "original_campaign": roots["original"].local_path,
        "raw_release": roots["raw"].local_path,
        "launch_archive": repository_root / config["historical"]["source_snapshot"],
        "tracked_evidence": repository_root / config["evidence_root"]})
    run.ensure()

    historical["base_selection"] = paths.read_json(
        roots["original"].path(config["historical"]["base_selection_relative"]))
    historical["guarded_anchor"] = guarded_anchor
    historical["original_anchor"] = original_anchor
    historical["raw_anchor"] = raw_logical.rsplit("/", 1)[-1]
    return AuditContext(repository_root=repository_root, run=run, config=config,
                        config_path=Path(config_path), config_sha256=config_sha256,
                        config_digest=config_digest, roots=roots, historical=historical)


def resolve_original_root(*, logical, anchor, candidates, base_relative, expected_parents):
    """Accept the v1 root whose selected parents hash to the guarded record.

    The probe is built per candidate, because the *paths* of the parents live in
    that candidate's own ``base_selection.json`` -- but the *digests* they must
    reproduce come from the already-verified guarded campaign. A candidate whose
    selection file names parents with different bytes is a different campaign, and
    a candidate that merely has a file at the expected place is not accepted at all.
    """
    attempts = []
    for candidate in candidates:
        if candidate in (None, ""):
            continue
        entry = {"candidate": str(candidate), "accepted": False, "probes": []}
        try:
            selection_path = paths.resolve_under(candidate, base_relative)
        except ValueError as error:
            attempts.append(dict(entry, reason=str(error)))
            continue
        if not selection_path.is_file():
            attempts.append(dict(entry, reason=f"{base_relative} is absent under this candidate"))
            continue
        try:
            selection = paths.read_json(selection_path)
        except (OSError, ValueError) as error:
            attempts.append(dict(entry, reason=f"{base_relative} is unreadable: {error}"))
            continue
        selected = selection.get("selected") or {}
        probes, missing = {}, []
        for seed, digest in sorted(expected_parents.items()):
            record = selected.get(f"policy_sft_seed{int(seed)}")
            if not isinstance(record, dict) or not record.get("checkpoint"):
                missing.append(seed)
                continue
            probes[paths.campaign_suffix(record["checkpoint"], anchor=anchor)] = digest
        if missing or not probes:
            attempts.append(dict(entry, reason=f"the selection names no parent for seeds {missing}"))
            continue
        ok, results = paths.probe_root(candidate, probes)
        attempts.append({"candidate": str(candidate), "accepted": ok,
                         "probes": [dict(result) for result in results]})
        if ok:
            return paths.ResolvedRoot(name="original", logical=logical,
                                      local_path=Path(candidate), probes=results,
                                      candidates=tuple(attempts))
    detail = "; ".join(f"{item['candidate']}: "
                       f"{item.get('reason') or [p.get('reason') for p in item['probes'] if not p['matched']]}"
                       for item in attempts) or "no candidate was supplied"
    raise ValueError(
        f"No candidate root for 'original' ({logical}) holds the selected parents the guarded "
        f"campaign inherited. Tried {detail}. Record the machine-local mapping in the ignored "
        "local_roots.json; this is a missing or different artifact, not a reason to skip "
        "verification.")


def _raw_root_probes(repository_root, raw_logical, config):
    """Probe the raw root with the pinned p-IgGen weights and the two readable splits.

    The tracked benchmark manifests under ``specs/benchmarks/`` are the authority:
    they are in Git, a reviewer can diff them, and they already record these digests.
    """
    from ..benchmarks import provenance as prov
    wanted = set(config["roots"]["raw"]["probe_files"])
    probes = {}
    for relative in config["source_manifests"]:
        document = prov.load_manifest_document(Path(repository_root) / relative)
        # ``relative_path`` is already expressed relative to the raw root, which is
        # exactly the suffix a root probe needs.
        for entry in document.validated().files:
            if entry.relative_path in wanted:
                probes[entry.relative_path] = entry.sha256
    missing = sorted(wanted - set(probes))
    require(not missing,
            f"The tracked benchmark manifests do not pin {missing}; the raw root cannot be "
            "accepted by hash without them")
    return probes


# ---------------------------------------------------------------------------
# git, environment and the freeze marker
# ---------------------------------------------------------------------------

def git(repository_root, *arguments):
    """Read-only git query. This module never runs a command that writes history."""
    forbidden = {"commit", "add", "push", "reset", "checkout", "merge", "rebase", "tag", "clean"}
    require(arguments and arguments[0] not in forbidden,
            f"The audit CLI never runs 'git {arguments[0] if arguments else ''}'; the freeze "
            "verifies a commit somebody else made")
    result = subprocess.run(["git", *arguments], cwd=str(repository_root), capture_output=True,
                            text=True)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def git_state(repository_root):
    code, head, error = git(repository_root, "rev-parse", "HEAD")
    require(code == 0, f"git rev-parse HEAD failed: {error}")
    code, status, error = git(repository_root, "status", "--porcelain")
    require(code == 0, f"git status failed: {error}")
    entries = [line for line in status.splitlines() if line.strip()]
    return {"commit": head, "dirty": bool(entries), "dirty_entries": entries}


def git_bytes(repository_root, *arguments):
    """Read-only git query whose stdout is raw bytes, with no newline translation.

    ``text=True`` would decode and universal-newline the output, which is exactly
    the difference a CRLF checkout introduces -- so the byte comparison that is
    supposed to detect it would pass on every platform.
    """
    require(arguments and arguments[0] in ("cat-file", "show", "rev-parse"),
            "git_bytes reads objects; it does not run a command that writes history")
    result = subprocess.run(["git", *arguments], cwd=str(repository_root), capture_output=True)
    return result.returncode, result.stdout


def git_blob(repository_root, logical):
    code, blob, _ = git(repository_root, "rev-parse", f"HEAD:{logical}")
    return blob if code == 0 else None


def head_blob_sha256(repository_root, logical):
    """sha256 of the **committed bytes** of ``logical`` at HEAD, or ``None``.

    The blob *id* is not this: git hashes the normalized content it stores, so a
    working tree that ``core.autocrlf`` re-expanded to CRLF on checkout has the same
    blob id and different bytes on disk. The audit hashes files on disk, so the
    freeze has to compare those bytes to what the commit actually holds.
    """
    code, payload = git_bytes(repository_root, "cat-file", "blob", f"HEAD:{logical}")
    return paths.sha256_bytes(payload) if code == 0 else None


def worktree_matches_head(repository_root, logicals):
    """``[{file, worktree_sha256, head_sha256}]`` for files whose bytes differ from HEAD."""
    differing = []
    for logical in sorted(set(logicals)):
        target = Path(repository_root) / logical
        observed = paths.sha256_file(target) if target.is_file() else None
        committed = head_blob_sha256(repository_root, logical)
        if observed != committed:
            differing.append({"file": logical, "worktree_sha256": observed,
                              "head_sha256": committed})
    return differing


def git_tracked(repository_root, logical):
    code, _, _ = git(repository_root, "ls-files", "--error-unmatch", "--", logical)
    return code == 0


#: Campaign evidence is local-only research material, so it cannot be
#: authenticated by asking "is this tracked, and equal to HEAD?" -- that question
#: requires publishing it. A committed manifest of digests gives the same
#: tamper-detection without the bytes: the repository carries the hashes, the
#: machine carries the evidence. A freeze that silently accepted whatever
#: evidence happened to be on disk would certify nothing.
EVIDENCE_MANIFEST_DIR = "configs/evidence_manifests"

#: A freeze binds a campaign to exact source bytes, which is what makes its
#: numbers attributable. That contract has no room for "the file changed but it
#: is fine" -- so when a source genuinely must migrate after a campaign is
#: complete, the migration is recorded here rather than waved through.
#:
#: Each record authorizes ONE transition: this file, from this digest, to this
#: digest, in this commit, for this reason. A later edit to the same file breaks
#: the identity again until somebody records that transition too, and a record
#: whose current digest does not match the tree authorizes nothing. It is tracked
#: so the authorization is reviewable, and stages report identity held "via
#: recorded supersession" rather than silently passing as though bytes matched.
SOURCE_SUPERSESSIONS = "configs/source_supersessions.json"


def load_source_supersessions(repository_root):
    """``{(logical, superseded_sha256): record}`` for deliberately migrated sources."""
    path = Path(repository_root) / SOURCE_SUPERSESSIONS
    if not path.is_file():
        return {}
    records = paths.read_json(path).get("supersessions") or []
    return {(record["file"], record["superseded_sha256"]): record for record in records}


def classify_source_drift(repository_root, logical, expected, observed, *, table=None):
    """The supersession authorizing exactly this transition, or ``None``.

    ``None`` means the drift is unexplained and the caller must fail. Both ends
    are checked: a record that names the old digest but not the bytes actually on
    disk is not an authorization for whatever happens to be there now.
    """
    table = load_source_supersessions(repository_root) if table is None else table
    record = table.get((logical, expected))
    if record is None or record.get("current_sha256") != observed:
        return None
    return {"file": logical, "superseded_sha256": expected, "current_sha256": observed,
            "commit": record.get("commit"), "reason": record.get("reason")}


def evidence_manifest_path(repository_root, campaign_id):
    return Path(repository_root) / EVIDENCE_MANIFEST_DIR / f"{campaign_id}.json"


def evidence_digests(repository_root, evidence_files):
    """``{logical_path: sha256}`` over the evidence as it exists on disk."""
    return {logical: paths.sha256_file(Path(repository_root) / logical)
            for logical in sorted(evidence_files)}


def require_evidence_matches_manifest(repository_root, evidence_files, campaign_id):
    """Verify local evidence against its committed digests, or refuse the freeze."""
    manifest_path = evidence_manifest_path(repository_root, campaign_id)
    logical_manifest = _relative(manifest_path, repository_root)
    require(manifest_path.is_file(),
            f"{logical_manifest} is absent. Campaign evidence is local-only, so the freeze "
            "authenticates it against committed digests rather than against HEAD. Write the "
            f"manifest with:  python scripts/pin_evidence_manifest.py --config <config> ")
    require(git_tracked(repository_root, logical_manifest),
            f"{logical_manifest} is not tracked at HEAD. An untracked manifest is not evidence "
            "about anything -- anyone could rewrite it beside the files it certifies.")
    manifest = paths.read_json(manifest_path)
    recorded = manifest.get("sha256") or {}
    observed = evidence_digests(repository_root, evidence_files)
    missing = sorted(set(recorded) - set(observed))
    extra = sorted(set(observed) - set(recorded))
    require(not missing and not extra,
            f"The evidence on disk does not match {logical_manifest}. Absent here: {missing}. "
            f"Not in the manifest: {extra}. Re-pin the manifest deliberately; a freeze over a "
            "changed evidence set would attribute the run to a specification nobody reviewed.")
    differing = [{"file": logical, "manifest": recorded[logical], "observed": observed[logical]}
                 for logical in sorted(recorded) if recorded[logical] != observed[logical]]
    require(not differing,
            "These evidence files differ from their committed digests: "
            + canonical_json(differing).strip())
    return {"path": logical_manifest, "sha256": paths.sha256_file(manifest_path),
            "file_count": len(recorded),
            "authenticated_by": ("committed digests, not tracked bytes: the evidence is "
                                 "local-only research material and is deliberately not published")}


def environment_record():
    record = {"python": sys.version.split()[0], "platform": platform.platform(),
              "executable": Path(sys.executable).name}
    try:
        import torch
        record.update(torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                      cuda_available=bool(torch.cuda.is_available()),
                      gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except ImportError:                                     # pragma: no cover - torch is a dep
        record.update(torch_version=None, cuda_available=False, gpu=None)
    try:
        import transformers
        record["transformers_version"] = str(transformers.__version__)
    except ImportError:
        record["transformers_version"] = None
    record["numpy_version"] = str(np.__version__)
    import pandas
    record["pandas_version"] = str(pandas.__version__)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True)
        record["gpu_driver"] = sorted(set(result.stdout.strip().splitlines()))
        record["gpu_driver_reason"] = None
    except (OSError, subprocess.SubprocessError) as error:
        record["gpu_driver"] = None
        record["gpu_driver_reason"] = type(error).__name__
    return record


def freeze_marker_path(context):
    return context.run.path(paths.FREEZE_MARKER)


def read_freeze_marker(context):
    path = freeze_marker_path(context)
    require(path.is_file(),
            f"{path} is absent: production scoring requires a recorded audit_spec_frozen marker "
            "pointing at the commit that fixed the sources, config and inputs. Run the freeze "
            "stage after the review commit; there is no bypass.")
    marker = paths.read_json(path)
    require(marker.get("schema_version") == paths.AUDIT_SCHEMA, f"{path} is not an audit marker")
    return marker


def require_frozen_identity(context):
    """Re-hash everything the marker pinned. A mismatch stops the stage."""
    marker = read_freeze_marker(context)
    differing = []
    superseded = []
    table = load_source_supersessions(context.repository_root)
    for logical, expected in sorted((marker.get("source_sha256") or {}).items()):
        observed = paths.sha256_file(context.repository_root / logical)
        if observed == expected:
            continue
        # Only SOURCE may be superseded, and only by a reviewed record. Evidence,
        # inputs and the config still fail on any drift: those are what the run
        # measured, not the code that measured it.
        migration = classify_source_drift(context.repository_root, logical, expected,
                                          observed, table=table)
        if migration is None:
            differing.append({"file": logical, "expected": expected, "observed": observed})
        else:
            superseded.append(migration)
    if context.config_sha256 != marker["config"]["sha256"]:
        differing.append({"file": marker["config"]["path"], "expected": marker["config"]["sha256"],
                          "observed": context.config_sha256})
    for logical, expected in sorted((marker.get("evidence_sha256") or {}).items()):
        target = context.repository_root / logical
        observed = paths.sha256_file(target) if target.is_file() else None
        if observed != expected:
            differing.append({"file": logical, "expected": expected, "observed": observed})
    for logical, expected in sorted((marker.get("input_sha256") or {}).items()):
        entry = marker["inputs"][logical]
        observed = _input_hash(context, entry)
        if observed != expected:
            differing.append({"file": logical, "expected": expected, "observed": observed})
    require(not differing,
            "The frozen identity no longer holds: " + canonical_json(differing).strip()
            + " Scoring under changed sources, config or inputs would attribute new numbers to "
              "the frozen specification.")
    marker["superseded_sources"] = superseded
    return marker


def _input_hash(context, entry):
    root = context.roots.get(entry["root"]) if entry.get("root") else None
    target = (root.path(entry["logical_path"]) if root is not None
              else context.repository_root / entry["logical_path"])
    return paths.sha256_file(target) if target.is_file() else None


# ---------------------------------------------------------------------------
# stage: inventory
# ---------------------------------------------------------------------------

def stage_documents(context):
    """The guarded validation documents, one per stage, from the resolved root.

    Read once during root resolution -- they are what accepts the v1 root -- and
    reused here so every later stage reads the same bytes.
    """
    cached = context.historical.get("guarded_stage_documents")
    if cached:
        return dict(cached)
    guarded = context.root("guarded")
    documents = {}
    for stage in context.historical["guarded_config"]["stages"]:
        number = int(stage["stage"])
        path = guarded.path(f"validation/stage{number}_endpoints.json")
        require(path.is_file(),
                f"validation/stage{number}_endpoints.json is absent under the verified guarded "
                "root; the endpoint identities have no authority without it")
        documents[number] = paths.read_json(path)
    return documents


def trajectory_documents(context):
    """``{(stage, trajectory): summary}`` for every declared fitted trajectory."""
    from .her2_guarded_eval import expected_trajectories
    guarded = context.root("guarded")
    config = context.historical["guarded_config"]
    summaries, stops = {}, {}
    for stage in config["stages"]:
        number = int(stage["stage"])
        for name in expected_trajectories(config, number):
            key = (number, name)
            summary_path = guarded.path(f"stage{number}/{name}/summary.json")
            require(summary_path.is_file(),
                    f"stage{number}/{name}/summary.json is absent; a declared trajectory with no "
                    "record is an absence of evidence, not a stopped trajectory")
            summaries[key] = paths.read_json(summary_path)
            stop_path = guarded.path(f"stage{number}/{name}/stop.json")
            if stop_path.is_file():
                stops[key] = paths.read_json(stop_path)
    return summaries, stops


def v1_validation_records_relative(context):
    """Logical name of the one global v1 validation document, under the original root."""
    return v1_validation_authority(context)["validation_records"]["logical_path"]


def v1_validation_authority(context):
    """Verify the original report's complete validation provenance, without reading outcomes."""
    original = context.root("original")
    historical = context.config["historical"]
    integrity_relative = historical["v1_final_integrity"]
    integrity = paths.read_json(context.repository_root / integrity_relative)
    freeze_relative = historical["v1_selection_frozen_relative"]
    freeze_path = original.path(freeze_relative)
    freeze_sha = paths.sha256_file(freeze_path)
    require(freeze_sha == integrity["selection_freeze_sha256"],
            "The v1 selection freeze differs from the published final integrity record")
    freeze = paths.read_json(freeze_path)
    require(freeze.get("assay_outcomes_read") is False,
            "The v1 selection freeze must precede reserved assay outcomes")
    authority = {"selection_frozen": {"logical_path": freeze_relative, "sha256": freeze_sha}}
    for name in ("validation_records", "numerical_evaluation"):
        entry = freeze[name]
        relative = paths.campaign_suffix(entry["path"],
                                         anchor=context.historical["original_anchor"])
        digest = paths.sha256_file(original.path(relative))
        require(digest == entry["sha256"], f"The frozen v1 {name} bytes no longer match")
        if name == "numerical_evaluation":
            require(digest == integrity["numerical_manifest_sha256"],
                    "The v1 numerical manifest differs from the published integrity record")
        authority[name] = {"logical_path": relative, "sha256": digest}
    return authority


def v1_summaries(context):
    """The v1 continuation summaries, keyed ``{method}_seed{seed}``, and the global records.

    The per-run budget entries carry checkpoints and exposure counts. Validation,
    pair, generation and diversity numbers come from the complete MATH-backend
    re-evaluation named by the original selection freeze. The continuation-level
    validation document predates that numerical amendment and is incomplete.
    """
    original = context.root("original")
    relative = context.config["historical"]["original_continuation_relative"]
    out = {}
    for method in context.config["populations"]["v1_methods"]:
        for seed in context.seeds:
            key = f"{method}_seed{seed}"
            path = original.path(f"{relative}/{key}/summary.json")
            require(path.is_file(), f"{relative}/{key}/summary.json is absent under the verified "
                                    "original root")
            out[key] = paths.read_json(path)
    records_path = original.path(v1_validation_records_relative(context))
    records = paths.read_json(records_path)
    return out, records


def build_inventory(context, *, reader, progress=None):
    """Enumerate, verify and deduplicate every state the audit will score."""
    clock = paths.StageClock()
    guarded = context.root("guarded")
    original = context.root("original")
    anchor = context.historical["guarded_anchor"]
    populations = context.config["populations"]

    with clock.segment("io"):
        stages = stage_documents(context)
        summaries, stops = trajectory_documents(context)
        v1, v1_validation = v1_summaries(context)

    parents = inventory_lib.parent_records(
        context.historical["base_selection"], seeds=context.seeds,
        logical_root=original.logical, anchor=context.historical["original_anchor"])
    parents_by_seed = {record["seed"]: record["file_sha256"] for record in parents}

    endpoints = inventory_lib.guarded_endpoint_records(stages, anchor=anchor,
                                                       logical_root=guarded.logical)
    reuse = inventory_lib.reused_control_aliases(stages, endpoints)
    for record in endpoints:
        record["parent_sha256"] = parents_by_seed.get(record["seed"])
        # A required binding, not an annotation: the guarded payload records the
        # parent it inherited, and an endpoint whose recorded parent is not the one
        # this audit scores it against measures a drop from the wrong distribution.
        record["contract"]["parent_sha256"] = parents_by_seed.get(record["seed"])

    stopped_keys = {key for key in summaries
                    if str(summaries[key].get("status")) == "stopped" or key in stops}
    diagnostics = inventory_lib.stopped_snapshot_records(
        {key: stops[key] for key in sorted(stopped_keys) if key in stops},
        summaries, anchor=anchor, logical_root=guarded.logical)
    v1_records = inventory_lib.v1_endpoint_records(
        v1, logical_root=original.logical, parents_by_seed=parents_by_seed,
        anchor=context.historical["original_anchor"], validation_records=v1_validation,
        validation_source=v1_validation_records_relative(context))

    records = parents + endpoints + diagnostics + v1_records
    paths.require_no_case_collisions([record["id"] for record in records], where="inventory ids")
    paths.require_no_case_collisions(
        [f"{record['root']}/{record['logical_path']}" for record in records
         if record.get("logical_path")], where="inventory logical paths")

    if progress is not None:
        progress.start(total=len(records))
    for record in records:
        if record.get("status") == inventory_lib.STATUS_MISSING:
            if progress is not None:
                progress.advance(record["id"])
            continue
        root = context.root("guarded") if record["root"] == guarded.logical else original
        logical = record["logical_path"]
        try:
            with clock.segment("io"):
                target = root.path(logical)
                if record["payload_schema"] == inventory_lib.POLICY_SCHEMA:
                    payload = reader.read_policy(target, logical=f"{record['root']}/{logical}",
                                                 expected_file_sha256=record["file_sha256"])
                else:
                    payload = reader.read_trajectory(target, logical=f"{record['root']}/{logical}",
                                                     expected_file_sha256=record["file_sha256"])
            contract = inventory_lib.validate_payload_contract(
                payload, record.get("contract") or {}, role=record["role"],
                logical=f"{record['root']}/{logical}",
                compare_if_present=record.get("compare_contract") or {})
        except (ValueError, OSError) as error:
            record["status"] = inventory_lib.STATUS_FAILED
            record["status_reason"] = f"{type(error).__name__}: {error}"
            if progress is not None:
                progress.advance(record["id"])
            continue
        record.update(payload.document())
        record["payload_contract"] = contract
        record["status"] = inventory_lib.STATUS_VERIFIED
        record["status_reason"] = None
        recorded_by_manifest = record.get("state_sha256_recorded_by_manifest")
        if recorded_by_manifest:
            require(recorded_by_manifest == payload.state_digest_audit_computed,
                    f"{record['id']}: the manifest binds state digest {recorded_by_manifest} but a "
                    f"strict reload produces {payload.state_digest_audit_computed}")
        if record["role"] in (inventory_lib.ROLE_LAST_PASSING, inventory_lib.ROLE_FAILED_STATE):
            stage, trajectory = record["stage"], record["trajectory"]
            exposures, reason = inventory_lib.exposures_at_update(
                guarded.path(f"stage{stage}/{trajectory}/updates.jsonl"), record.get("updates"))
            record["exposures"] = exposures
            record["exposures_reason"] = reason
        if progress is not None:
            progress.advance(record["id"])

    with clock.segment("analysis"):
        banks = verify_banks(context, stages, parents_by_seed)
        for record in records:
            bank = banks.get(record.get("seed"))
            record["population_id"] = None if bank is None else f"parent_bank::{bank.parent_id}"
            record["bank_sha256"] = None if bank is None else bank.file_sha256
            record["probability_convention"] = inventory_lib.CORE_CONTRACT[
                "probability_convention"]
            record["dedup_key"] = (
                None if record["status"] != inventory_lib.STATUS_VERIFIED
                else inventory_lib.dedup_key(record, bank=bank))
        assignment, dedup = inventory_lib.deduplicate(records)
        for record in records:
            record["deduplication"] = assignment.get(record["id"])
        coverage = inventory_lib.coverage_table(records, expected={
            "total": int(populations["expected_total_states"]),
            "by_role": {
                inventory_lib.ROLE_PARENT: int(populations["expected_parents"]),
                inventory_lib.ROLE_ENDPOINT: int(populations["expected_guarded_endpoints"]),
                inventory_lib.ROLE_LAST_PASSING: int(populations["expected_stopped_trajectories"]),
                inventory_lib.ROLE_FAILED_STATE: int(populations["expected_stopped_trajectories"]),
                inventory_lib.ROLE_V1_ENDPOINT: int(populations["expected_v1_dpo_endpoints"])
                                                + int(populations["expected_v1_sft_endpoints"])}})

    return {"schema_version": paths.AUDIT_SCHEMA, "record_kind": "inventory",
            "protocol": context.config["protocol"],
            "audit_id": context.config["audit_id"],
            "roots": {name: root.document() for name, root in sorted(context.roots.items())},
            "probability_contract": dict(inventory_lib.CORE_CONTRACT),
            "records": sorted(records, key=lambda item: item["id"]),
            "reused_controls": reuse, "deduplication": dedup, "coverage": coverage,
            "parent_banks": {str(seed): bank.document() for seed, bank in sorted(banks.items())},
            "bank_cross_stage": inventory_lib.cross_stage_bank_consistency(stages),
            "trajectories": {"declared": len(summaries),
                             "stopped": len(stopped_keys),
                             "completed": len(summaries) - len(stopped_keys)},
            "timings": clock.document()}, banks


def verify_banks(context, stages, parents_by_seed):
    """Independently re-verify each seed's persisted 10,000-row temperature-1 bank."""
    populations = context.config["populations"]
    guarded = context.root("guarded")
    references = stages[min(stages)].get("parent_draw_references") or {}
    banks = {}
    for seed in context.seeds:
        reference = references.get(str(seed))
        require(reference is not None,
                f"The validation document names no parent draw bank for seed {seed}; a missing "
                "bank appears in the coverage table and is never regenerated silently")
        logical = paths.campaign_suffix(reference["path"],
                                        anchor=context.historical["guarded_anchor"])
        banks[seed] = inventory_lib.verify_parent_bank(
            dict(reference, seed=seed), path=guarded.path(logical),
            expected_rows=int(populations["bank_rows"]),
            expected_temperature=float(populations["bank_temperature"]),
            parent_sha256=parents_by_seed[seed],
            parent_id=f"parent::policy_sft_seed{seed}")
    return banks


# ---------------------------------------------------------------------------
# stage: prepare (portable evidence, written before the review commit)
# ---------------------------------------------------------------------------

def portable_inventory(document):
    """The tracked half of the inventory: logical names, hashes, statuses, no local paths.

    ``recorded_path`` is deliberately **not** carried. The legacy records hold
    ``C:\\Users\\...`` and ``F:\\...`` strings, and publishing them would put one
    machine's directory layout into reviewed evidence for no scientific gain. What
    the recorded path is *evidence of* -- that a legacy Windows absolute path was
    the source of this row, and which exact string it was -- survives as the flavor
    and a digest of the original text, which a holder of the local artifacts can
    re-derive and compare. The strings themselves stay in the ignored run record.

    The whole document then goes through :func:`her2_support_paths.scrub_host_paths`,
    because the field nobody designed -- an exception message captured into
    ``status_reason`` -- is how a host path actually reaches published evidence.
    """
    keep = ("id", "role", "name", "campaign", "arm_id", "objective", "coefficients", "seed",
            "stage", "trajectory", "parent_id", "parent_sha256", "nominal_budget_gpu_seconds",
            "nominal_budget_reason", "actual_gpu_seconds", "updates", "updates_reason",
            "exposures", "exposures_reason", "recorded_path_flavor",
            "logical_path", "root", "file_sha256", "payload_schema", "payload_kind",
            "source_manifest", "state_digest_recorded", "state_digest_recorded_reason",
            "state_digest_audit_computed", "status", "status_reason", "population_id",
            "bank_sha256", "probability_convention", "dedup_key", "deduplication", "aliases",
            "gate_passed", "diversity_eligible", "stop_reason", "rolling_note",
            "historical_metrics_source", "payload_contract")
    records = []
    for record in document["records"]:
        row = {key: record.get(key) for key in keep if key in record}
        recorded = record.get("recorded_path")
        row["recorded_path_sha256"] = (None if not recorded else paths.sha256_text(recorded))
        row["recorded_path_note"] = (
            "the host-specific recorded string is kept only in the ignored run record; its flavor "
            "and digest are published so a holder of the local artifacts can re-derive it")
        records.append(row)
    portable = {"schema_version": paths.AUDIT_SCHEMA, "record_kind": "portable_inventory",
                "protocol": document["protocol"], "audit_id": document["audit_id"],
                "roots": document["roots"],
                "probability_contract": document["probability_contract"],
                "coverage": document["coverage"], "deduplication": document["deduplication"],
                "reused_controls": document["reused_controls"],
                "parent_banks": document["parent_banks"],
                "bank_cross_stage": document["bank_cross_stage"],
                "trajectories": document["trajectories"],
                "records": records,
                "note": ("logical names and content hashes only. Machine-local roots live in the "
                         "ignored local_roots.json and never appear in this file.")}
    scrubbed = paths.scrub_host_paths(portable)
    leaks = paths.host_path_leaks(scrubbed)
    require(not leaks,
            f"Host paths survived scrubbing in the portable inventory at "
            f"{[entry['field'] for entry in leaks[:5]]}")
    return scrubbed


def build_pair_populations(context):
    """The fixed validation pairs and the declared v1 cycle-0 training prefixes.

    Both are built from the *frozen historical* pairing settings, so the IDs are a
    function of the committed configuration and not of anything this audit observes.
    """
    from .her2_data import load_split
    from .her2_preferences import PreferencePairing, build_population
    raw = context.root("raw").local_path
    populations = context.config["populations"]
    original = context.historical["original_config"]["continuation"]["pairing"]
    require(int(original["seed_base"]) == int(populations["pairing_seed_base"]),
            "The audit config's pairing seed base disagrees with the original campaign config")
    require(int(original["validation_seed"]) == int(populations["validation_pairing_seed"]),
            "The audit config's validation pairing seed disagrees with the original campaign")

    train = load_split(raw, "train")
    val = load_split(raw, "val")
    train_population = build_population(train, "train")
    val_population = build_population(val, "val")
    val_pairing = PreferencePairing(val_population, seed=int(original["validation_seed"]))
    validation = val_pairing.fixed_validation_pairs(original["validation_pairs"])
    require(int(validation["pairs"]) == int(populations["fixed_validation_pairs"]),
            f"The fixed validation pairing yields {validation['pairs']} pairs, the audit declares "
            f"{populations['fixed_validation_pairs']}")
    validation_rows, validation_partners = val_pairing.cycle_rows(0)
    if original["validation_pairs"] is not None:
        validation_rows = validation_rows[:int(original["validation_pairs"])]
        validation_partners = validation_partners[:int(original["validation_pairs"])]

    training = {}
    for seed in context.seeds:
        pairing = PreferencePairing(train_population,
                                    seed=int(original["seed_base"]) + int(seed))
        training[seed] = ches_lib.training_pair_prefix(
            pairing, cycle=int(populations["v1_training_cycle"]),
            count=int(populations["v1_training_pairs_per_seed"]))
    return {
        "train_population": train_population, "val_population": val_population,
        "train_frame": train, "val_frame": val,
        "validation": dict(validation, chosen_rows=validation_rows,
                           rejected_rows=validation_partners),
        "training": training,
        "identity": {
            "fixed_validation": ches_lib.pair_identity(
                validation["chosen_index"], validation["rejected_index"],
                population_id="fixed_validation_pairs",
                construction={"pairing_seed": int(original["validation_seed"]),
                              "cycle": 0, "count": original["validation_pairs"],
                              "rule": "PreferencePairing(val_population).fixed_validation_pairs"}),
            "v1_training": {
                str(seed): ches_lib.pair_identity(
                    block["chosen_index"], block["rejected_index"],
                    population_id=f"v1_training_cycle{block['cycle']}_seed{seed}",
                    construction={"pairing_seed": int(original["seed_base"]) + int(seed),
                                  "cycle": block["cycle"], "count": block["count"],
                                  "cycle_length": block["cycle_length"],
                                  "rule": "first N ordered pairs of PreferencePairing.cycle_rows"})
                for seed, block in sorted(training.items())}}}


def render_pair_evidence(context, pairs):
    """``{logical: text}`` for the frozen pair-ID CSVs. Renders, writes nothing."""
    rendered = {}
    lines = ["pair_index,chosen_row,rejected_row\n"]
    for position, (chosen, rejected) in enumerate(zip(pairs["validation"]["chosen_rows"],
                                                      pairs["validation"]["rejected_rows"])):
        lines.append(f"{position},{int(chosen)},{int(rejected)}\n")
    rendered[f"{context.config['evidence_root']}/pairs/fixed_validation_pairs.csv"] = "".join(lines)
    for seed, block in sorted(pairs["training"].items()):
        lines = ["pair_index,cycle,chosen_row,rejected_row\n"]
        for position, row in enumerate(block["pair_ids"]):
            lines.append(f"{position},{int(row[0])},{int(row[1])},{int(row[2])}\n")
        name = f"v1_training_cycle{block['cycle']}_seed{seed}.csv"
        rendered[f"{context.config['evidence_root']}/pairs/{name}"] = "".join(lines)
    return rendered


def verify_pair_evidence(context, rendered):
    """``{logical: sha256}`` for the rendered pair CSVs, refusing any that already differ.

    Nothing is written here. A rerun that would change a committed, frozen pair file
    has to stop *before* the first write: a check that runs afterwards reports a
    violation whose damage is already done.
    """
    digests, differing = {}, []
    for logical, text in sorted(rendered.items()):
        target = context.repository_root / logical
        digests[logical] = paths.sha256_text(text)
        if target.is_file() and paths.sha256_file(target) != digests[logical]:
            differing.append(logical)
    require(not differing,
            f"These committed pair files would change: {differing}. The pair identities are fixed "
            "before scoring; a different pairing is a different measurement and belongs in a new "
            "revision directory.")
    return digests


def write_pair_evidence(context, rendered):
    """Write the already-verified pair CSVs; identical files are left untouched."""
    written = {}
    for logical, text in sorted(rendered.items()):
        target = context.repository_root / logical
        digest = paths.sha256_text(text)
        if target.is_file() and paths.sha256_file(target) == digest:
            written[logical] = digest
            continue
        written[logical] = paths.write_text(target, text)
    return written


def raw_input_entry(raw_logical, *, anchor, recorded, digest):
    """``(manifest key, entry)`` for one pinned raw source, without doubling the root.

    ``her2_data.source_digests`` returns these keyed by the path the retrieval record
    wrote, which already contains the raw root:
    ``data/raw/her2_functional_20260918/absci/LICENSE``. Prefixing the resolved root
    onto that produced ``data/raw/.../data/raw/...`` -- a key that resolves to a file
    that does not exist, so 23 of these entries failed their own hash check and no
    freeze could ever pass. The anchor is stripped exactly once: the key stays the
    single repository-logical name a reviewer recognizes, and ``logical_path`` is the
    suffix that resolves under whichever raw root was selected, including a
    ``--raw-root`` override.
    """
    suffix = paths.campaign_suffix(recorded, anchor=anchor)
    return paths.logical_join(raw_logical, suffix), {
        "root": "raw", "logical_path": suffix, "sha256": digest,
        "kind": "pinned_raw_source", "recorded_key": recorded,
        "note": ("the retrieval record writes this path with the raw root inside it; the "
                 "logical_path below the root is what resolves against --raw-root")}


def build_input_manifest(context, inventory, pairs, pair_files):
    """Every input hash this audit depends on, as one tracked, linkable document.

    Mechanically generated bindings live here rather than in the config, so the
    config stays a hand-reviewed statement of defaults, tolerances and rules, and
    neither document hashes itself.
    """
    from .her2_data import source_digests, verify_all_sources
    raw = context.root("raw").local_path
    problems = verify_all_sources(raw, context.repository_root, context.config["source_manifests"])
    require(not problems, "Pinned raw sources do not verify: " + "; ".join(problems))
    sources = source_digests(context.repository_root, raw, context.config["source_manifests"])

    snapshot = inventory_lib.verify_source_snapshot(
        context.historical["launch_manifest"],
        context.repository_root / context.config["historical"]["source_snapshot"],
        repository_root=context.repository_root)

    inputs = {}
    for record in inventory["records"]:
        if not record.get("logical_path") or not record.get("file_sha256"):
            continue
        logical = f"{record['root']}/{record['logical_path']}"
        inputs[logical] = {"root": ("guarded" if record["root"] == context.root("guarded").logical
                                    else "original"),
                           "logical_path": record["logical_path"],
                           "sha256": record["file_sha256"], "kind": record["role"]}
    for seed, bank in sorted(inventory["parent_banks"].items()):
        relative = paths.campaign_suffix(bank["logical_path"],
                                         anchor=context.historical["guarded_anchor"])
        inputs[f"{context.root('guarded').logical}/{relative}"] = {
            "root": "guarded", "logical_path": relative,
            "sha256": bank["file_sha256"], "kind": "parent_bank"}
    manifest_relatives = set(context.config["source_manifests"])
    raw_root = context.root("raw")
    raw_anchor = context.historical["raw_anchor"]
    for logical, digest in sorted(sources.items()):
        # Two different kinds of key come back from source_digests: the tracked
        # manifests, which are repository-relative, and the pinned data/weight files,
        # whose retrieval records already write them out in full -- including the
        # raw root itself, ``data/raw/her2_functional_20260918/absci/LICENSE``.
        # Filing them under root=null would re-hash them against the repository and
        # ignore --raw-root entirely; prefixing the raw root onto a key that already
        # carries it produced ``data/raw/.../data/raw/...`` and made 23 entries fail
        # their own hash verification, which no freeze can pass. The anchor is
        # stripped once, so the entry resolves *under* the selected raw root while
        # the manifest key stays the single repository-logical name.
        if logical in manifest_relatives:
            inputs[logical] = {"root": None, "logical_path": logical, "sha256": digest,
                               "kind": "tracked_manifest"}
            continue
        key, entry = raw_input_entry(raw_root.logical, anchor=raw_anchor, recorded=logical,
                                     digest=digest)
        inputs[key] = entry

    # The historical documents this audit takes its identities from are inputs too.
    # An endpoint hash is only authoritative because the stage document that carries
    # it is the one that was reviewed.
    for name, relative in sorted(_historical_input_paths(context).items()):
        target = context.repository_root / relative
        if target.is_file():
            inputs[relative] = {"root": None, "logical_path": relative,
                                "sha256": paths.sha256_file(target),
                                "kind": f"historical_manifest::{name}"}
    guarded = context.root("guarded")
    for name, relative in sorted(_historical_campaign_documents(context).items()):
        target = guarded.path(relative)
        if target.is_file():
            inputs[f"{guarded.logical}/{relative}"] = {
                "root": "guarded", "logical_path": relative,
                "sha256": paths.sha256_file(target),
                "kind": f"historical_manifest::{name}"}
    original = context.root("original")
    base_relative = context.config["historical"]["base_selection_relative"]
    inputs[f"{original.logical}/{base_relative}"] = {
        "root": "original", "logical_path": base_relative,
        "sha256": paths.sha256_file(original.path(base_relative)),
        "kind": "historical_manifest::base_selection"}
    continuation = context.config["historical"]["original_continuation_relative"]
    for method in context.config["populations"]["v1_methods"]:
        for seed in context.seeds:
            relative = f"{continuation}/{method}_seed{seed}/summary.json"
            target = original.path(relative)
            if target.is_file():
                inputs[f"{original.logical}/{relative}"] = {
                    "root": "original", "logical_path": relative,
                    "sha256": paths.sha256_file(target),
                    "kind": "historical_manifest::v1_run"}
    # The v1 validation numbers this audit publishes come out of ONE global document,
    # so that document is an input in its own right.
    for name, entry in v1_validation_authority(context).items():
        inputs[f"{original.logical}/{entry['logical_path']}"] = {
            "root": "original", **entry, "kind": f"historical_manifest::v1_{name}"}
    integrity_relative = context.config["historical"]["v1_final_integrity"]
    inputs[integrity_relative] = {
        "root": None, "logical_path": integrity_relative,
        "sha256": paths.sha256_file(context.repository_root / integrity_relative),
        "kind": "historical_manifest::v1_final_integrity"}

    scaffold_prefix = _scaffold_prefix(context)
    return {"schema_version": paths.AUDIT_SCHEMA, "record_kind": "input_manifest",
            "audit_id": context.config["audit_id"], "protocol": context.config["protocol"],
            "config": {"path": _relative(context.config_path, context.repository_root),
                       "sha256": context.config_sha256, "digest": context.config_digest},
            "probability_contract": dict(inventory_lib.CORE_CONTRACT),
            "scaffold": {"prefix_length": len(scaffold_prefix),
                         "prefix_sha256": paths.sha256_text(scaffold_prefix),
                         "source": "row 0 of the submitted design table via her2_data.load_scaffold",
                         "note": "no reserved test label or SPR outcome is read to obtain this"},
            "inputs": dict(sorted(inputs.items())),
            "input_count": len(inputs),
            "pair_populations": pairs["identity"],
            "pair_files": dict(sorted(pair_files.items())),
            "historical_source_snapshot": snapshot,
            "note": ("this file is generated by the inventory/prepare stages and committed. The "
                     "config links to it and does not restate it, so neither document hashes "
                     "itself.")}


def _historical_input_paths(context):
    """Repository-relative historical documents whose bytes this audit relies on."""
    historical = context.config["historical"]
    return {"guarded_manifest": historical["guarded_manifest"],
            "launch_manifest": historical["launch_manifest"],
            "campaign_status": historical["campaign_status"],
            "guarded_config": historical["guarded_config"],
            "original_config": historical["original_config"]}


def _historical_campaign_documents(context):
    """Guarded-root documents that supply endpoint, stop and bank identities."""
    from .her2_guarded_eval import expected_trajectories
    config = context.historical["guarded_config"]
    out = {}
    for stage in config["stages"]:
        number = int(stage["stage"])
        out[f"stage{number}_endpoints"] = f"validation/stage{number}_endpoints.json"
        out[f"stage{number}_freeze"] = f"validation/stage{number}_freeze.json"
        for name in expected_trajectories(config, number):
            out[f"{name}_summary"] = f"stage{number}/{name}/summary.json"
            out[f"{name}_stop"] = f"stage{number}/{name}/stop.json"
    return out


def _scaffold_prefix(context):
    from .her2_data import load_scaffold
    return load_scaffold(context.root("raw").local_path).prefix


def _relative(path, root):
    return str(Path(path).resolve().relative_to(Path(root).resolve())).replace("\\", "/")


# ---------------------------------------------------------------------------
# stage: preflight
# ---------------------------------------------------------------------------

def run_preflight(context, *, policy, bank, rows, batch_rows, batch_sizes):
    """Bounded numerical probes on NEW code only. No sampling, no full-bank scoring.

    Every probe is on the first ``rows`` persisted bank rows of one seed, which
    already exist. Nothing here draws from a model and nothing here scores a
    population the audit will later report.
    """
    import torch
    tolerances = context.config["tolerances"]
    probe = bank.index[:int(rows)]
    wide = bank.index[:int(batch_rows)]
    clock = paths.StageClock()
    report = {"schema_version": paths.AUDIT_SCHEMA, "record_kind": "preflight",
              "probe_rows": int(rows), "batch_probe_rows": int(batch_rows),
              "bank": bank.document(),
              "note": ("bounded probes on persisted rows; no new sampling and no production "
                       "scoring happens before the freeze")}
    with clock.segment("inference"):
        report["head_reconstruction_full"] = ches_lib.head_reconstruction(
            policy, policy.token_ids(probe),
            atol=float(tolerances["head_reconstruction_atol"]),
            rtol=float(tolerances["head_reconstruction_rtol"]), cached=False)
        report["head_reconstruction_cached"] = ches_lib.head_reconstruction(
            policy, policy.token_ids(probe),
            atol=float(tolerances["head_reconstruction_atol"]),
            rtol=float(tolerances["head_reconstruction_rtol"]), cached=True)
        full = ches_lib.ches_batch(policy, probe, probe[::-1].copy(), cached=False)
        cached = ches_lib.ches_batch(policy, probe, probe[::-1].copy(), cached=True)
        identical = ches_lib.ches_batch(policy, probe, probe, cached=False)
        scores = {}
        for size in batch_sizes:
            with torch.inference_mode():
                scores[int(size)] = policy.score(wide, batch_size=int(size))[
                    "sum_log_probability"]
    difference = np.abs(full - cached)
    allowance = float(tolerances["ches_dot_atol"]) + float(tolerances["ches_dot_rtol"]) * np.abs(
        cached)
    report["ches_cached_vs_full"] = {
        "rows": int(full.size), "max_abs_error": float(difference.max()),
        "atol": float(tolerances["ches_dot_atol"]), "rtol": float(tolerances["ches_dot_rtol"]),
        "within_tolerance": bool((difference <= allowance).all())}
    require(report["ches_cached_vs_full"]["within_tolerance"],
            f"CHES cached/full parity failed: max error {difference.max():.6g}. Investigate; do "
            "not retry at a smaller batch until the discrepancy disappears.")
    report["ches_identical_inputs"] = {
        "rows": int(identical.size), "max_abs_ches": float(np.abs(identical).max()),
        "atol": float(tolerances["ches_dot_atol"]),
        "within_tolerance": bool(np.abs(identical).max()
                                 <= float(tolerances["ches_dot_atol"])
                                 + float(tolerances["ches_dot_rtol"])
                                 * float(np.abs(identical).max()))}
    require(report["ches_identical_inputs"]["within_tolerance"],
            "CHES of identical chosen/rejected inputs is not zero within the declared tolerance")
    report["ches_direct_double_sum"] = _tiny_double_sum_control(
        atol=float(tolerances["ches_dot_atol"]))

    from .her2_policy import compare_sum_log_probabilities
    reference_size = int(batch_sizes[-1])
    report["batch_size_parity"] = {}
    for size, values in sorted(scores.items()):
        if size == reference_size:
            continue
        report["batch_size_parity"][str(size)] = compare_sum_log_probabilities(
            values, scores[reference_size], label=f"batch {size} vs {reference_size}",
            atol=float(tolerances["sum_log_probability_atol"]),
            rtol=float(tolerances["sum_log_probability_rtol"]))
    # The audit's own scoring route, on both forward paths and twice on one path.
    # Head reconstruction proves the states; this proves the number the audit
    # actually publishes is the same on the cached and the full route, and that a
    # repeat of it is bitwise identical rather than merely close.
    with clock.segment("inference"):
        cached_lp = scoring.strict_sequence_log_probabilities(
            policy, probe, batch_size=reference_size, label="preflight cached", cached=True)
        full_lp = scoring.strict_sequence_log_probabilities(
            policy, probe, batch_size=reference_size, label="preflight full", cached=False)
        repeat_lp = scoring.strict_sequence_log_probabilities(
            policy, probe, batch_size=reference_size, label="preflight repeat", cached=True)
    report["sum_log_probability_cached_vs_full"] = compare_sum_log_probabilities(
        cached_lp["sum_log_probability"], full_lp["sum_log_probability"],
        label="audit scorer: cached versus full teacher forcing",
        atol=float(tolerances["sum_log_probability_atol"]),
        rtol=float(tolerances["sum_log_probability_rtol"]))
    repeat_error = float(np.abs(repeat_lp["sum_log_probability"]
                                - cached_lp["sum_log_probability"]).max())
    report["repeat_determinism"] = {
        "rows": int(probe.shape[0]), "max_abs_error": repeat_error,
        "bitwise_identical": repeat_error == 0.0,
        "requirement": ("the same weights, rows and batch size must produce bitwise identical "
                        "sums; a nonzero difference is nondeterminism in the route, not noise")}
    require(report["repeat_determinism"]["bitwise_identical"],
            f"Repeating the audit scorer on identical inputs moved the result by {repeat_error:g}. "
            "The scoring route is nondeterministic and the audit stops here.")
    report["logit_checks"] = cached_lp["checks"]
    with clock.segment("inference"):
        started = time.perf_counter()
        with torch.inference_mode():
            policy.score(wide, batch_size=reference_size)
        elapsed = time.perf_counter() - started
    from .her2_runtime import free_vram_mib
    free = free_vram_mib()
    report["operational"] = {"free_vram_mib": None if free is None else float(free),
                             "rows_scored": int(wide.shape[0]),
                             "seconds_for_probe_batch": elapsed,
                             "batch_size": reference_size,
                             "note": ("inference batch size is a recorded operational parameter, "
                                      "not a scientific arm, and only while parity holds")}
    report["environment"] = environment_record()
    report["timings"] = clock.document()
    report["status"] = "completed"
    return report


def _tiny_double_sum_control(*, atol):
    """The formula against a deliberately naive reference on tensors small enough to trust."""
    generator = np.random.default_rng(20260919)
    plus = generator.normal(size=(3, 10, 4))
    minus = generator.normal(size=(3, 10, 4))
    fast = ches_lib.ches_from_states(plus, minus)
    slow = ches_lib.ches_reference(plus, minus)
    error = float(np.abs(fast - slow).max())
    require(error <= atol,
            f"The vectorized CHES disagrees with the direct double sum by {error:.3g}")
    return {"rows": int(fast.size), "max_abs_error": error, "atol": float(atol)}


# ---------------------------------------------------------------------------
# stage: freeze
# ---------------------------------------------------------------------------

def preflight_binding(context):
    """What a preflight result is evidence *about*: sources, config, inputs, probes.

    Recorded inside the preflight report and re-derived at freeze time. The two
    must agree, or the probe describes a different computation than the one about
    to be scored.
    """
    inventory_path = context.run.path(INVENTORY_JSON)
    inventory = paths.read_json(inventory_path) if inventory_path.is_file() else {}
    parents = sorted((record["id"], record.get("state_digest_audit_computed"))
                     for record in inventory.get("records", [])
                     if record.get("role") == inventory_lib.ROLE_PARENT)
    banks = sorted((str(seed), block.get("file_sha256"))
                   for seed, block in (inventory.get("parent_banks") or {}).items())
    settings = context.config["inference"]
    return {
        "source_sha256": paths.digest_document(
            inventory_lib.audit_source_identity(context.repository_root, AUDIT_SOURCE_FILES)),
        "config_sha256": context.config_sha256,
        "parents": paths.digest_document(parents),
        "parent_banks": paths.digest_document(banks),
        "probe_order": paths.digest_document({
            "rows": int(settings["preflight_rows"]),
            "batch_rows": int(settings["preflight_batch_rows"]),
            "batch_sizes": [int(size) for size in settings["preflight_batch_sizes"]],
            "tolerances": dict(context.config["tolerances"])}),
        "environment": paths.digest_document(environment_record())}


def require_committed_inventory(context, inventory):
    """The runtime inventory must be the committed one, projection for projection.

    Production reads the ignored ``inventory.json`` in the run directory; the
    freeze hashes the tracked portable copy. Without this check the two can drift
    and every later number would be attributed to the reviewed file.
    """
    committed_path = context.repository_root / context.config["inventory_evidence"]
    require(committed_path.is_file(),
            f"{committed_path} is absent; the committed portable inventory is the authority the "
            "runtime inventory is checked against")
    committed = paths.read_json(committed_path)
    rebuilt = portable_inventory(inventory)
    if paths.scientific_projection(committed) == paths.scientific_projection(rebuilt):
        return {"matches_committed_inventory": True,
                "committed_sha256": paths.sha256_file(committed_path)}
    differing = sorted({key for key in set(committed) | set(rebuilt)
                        if paths.scientific_projection(committed.get(key))
                        != paths.scientific_projection(rebuilt.get(key))})
    raise ValueError(
        f"The run-directory inventory no longer projects onto the committed portable inventory "
        f"({committed_path}); they differ at {differing}. Production scoring would attribute its "
        "numbers to a reviewed file that does not describe it.")


def run_freeze(context):
    """Verify the committed sources, config, evidence and inputs, then write the marker.

    This stage does not commit and has no dirty-tree bypass. It reads the tree that
    somebody already reviewed and committed, proves that the bytes it is about to
    score are those bytes, and records the actual HEAD.
    """
    marker_path = freeze_marker_path(context)
    require(not marker_path.is_file(),
            f"{marker_path} already exists and is not replaced. A freeze names the commit that "
            "fixed this audit; re-freezing over it would silently re-point scored artifacts at a "
            "different specification. Delete the run directory to start a new revision.")
    state = git_state(context.repository_root)
    require(not state["dirty"],
            "The working tree is dirty:\n  " + "\n  ".join(state["dirty_entries"][:20])
            + "\nThe freeze records a commit somebody reviewed. Commit or stash first; there is "
              "no --allow-dirty for this stage.")
    evidence_root = context.config["evidence_root"]
    tracked = list(AUDIT_SOURCE_FILES) + [_relative(context.config_path, context.repository_root)]
    evidence_files = sorted(
        _relative(path, context.repository_root)
        for path in (context.repository_root / evidence_root).rglob("*") if path.is_file())
    require(evidence_files,
            f"{evidence_root} holds no evidence; run the inventory/prepare stages and commit their "
            "output before freezing")
    # Evidence is authenticated against committed digests instead of being
    # required to be tracked; see require_evidence_matches_manifest. Sources and
    # the config are code and stay tracked.
    evidence_manifest = require_evidence_matches_manifest(
        context.repository_root, evidence_files, context.config["audit_id"])
    untracked = [logical for logical in tracked
                 if not git_tracked(context.repository_root, logical)]
    require(not untracked,
            f"These files are not tracked at HEAD and cannot be frozen: {untracked}")
    crlf = [logical for logical in tracked
            if logical not in LEGACY_RAW_SOURCE_FILES
            if b"\r\n" in (context.repository_root / logical).read_bytes()]
    require(not crlf,
            f"These frozen files carry CRLF bytes: {crlf}. The narrow .gitattributes rules make "
            "this tree LF on every platform; a CRLF file hashes differently after a clone and the "
            "freeze digests would not reproduce.")
    # A clean ``git status`` and a matching blob *id* both survive newline
    # translation: git normalizes what it stores. The audit hashes bytes on disk, so
    # the only check that means anything is a comparison of those bytes to the bytes
    # the commit actually holds. With core.autocrlf=true and no eol attribute, a
    # fresh checkout of these same commits would hash differently and this stops it.
    drifted = worktree_matches_head(context.repository_root, tracked)
    require(not drifted,
            "These frozen files differ from their committed bytes at HEAD: "
            + canonical_json(drifted).strip()
            + " A checkout that re-expands newlines produces different hashes from the same "
              "commit. Add a narrow .gitattributes eol rule for the path and re-checkout; the "
              "freeze is not taken on bytes that a clone would not reproduce.")

    preflight_path = context.run.path(PREFLIGHT_JSON)
    require(preflight_path.is_file(),
            f"{preflight_path} is absent; the native preflight evidence is part of the freeze")
    preflight = paths.read_json(preflight_path)
    require(preflight.get("status") == "completed", "The recorded preflight did not complete")
    # A preflight is evidence only about the code, config and artifacts it ran
    # against. Hashing whichever preflight happens to be on disk would let a probe
    # from an earlier revision certify the current one.
    binding = preflight.get("binding") or {}
    current = preflight_binding(context)
    differing = sorted(key for key in set(binding) | set(current)
                       if binding.get(key) != current.get(key))
    require(not differing,
            f"The recorded preflight was run against different {differing}. Re-run the preflight "
            "stage against the sources, config, parents and banks being frozen.")

    manifest_path = context.repository_root / context.config["input_manifest"]
    require(manifest_path.is_file(), f"{manifest_path} is absent; run the prepare stage first")
    input_manifest = paths.read_json(manifest_path)
    inputs = input_manifest["inputs"]
    input_hashes = {}
    for logical, entry in sorted(inputs.items()):
        observed = _input_hash(context, entry)
        require(observed == entry["sha256"],
                f"{logical} hashes {observed}, the committed input manifest recorded "
                f"{entry['sha256']}")
        input_hashes[logical] = entry["sha256"]

    marker = {
        "schema_version": paths.AUDIT_SCHEMA, "record_kind": "audit_spec_frozen",
        "audit_id": context.config["audit_id"], "protocol": context.config["protocol"],
        "frozen_at": paths.utc_now(),
        "git": {"commit": state["commit"], "dirty": False,
                "blobs": {logical: git_blob(context.repository_root, logical)
                          for logical in sorted(tracked + evidence_files)},
                "head_bytes_sha256": {logical: head_blob_sha256(context.repository_root, logical)
                                      for logical in sorted(tracked + evidence_files)},
                "legacy_raw_sources": list(LEGACY_RAW_SOURCE_FILES),
                "byte_note": ("Every frozen file matches HEAD bytes exactly. Two legacy sources "
                              "use -text to preserve existing CRLF/mixed endings without "
                              "changing their historical bytes. Other frozen text uses LF. "
                              "source_sha256 always hashes the actual file bytes.")},
        "source_sha256": inventory_lib.audit_source_identity(context.repository_root,
                                                             AUDIT_SOURCE_FILES),
        "config": {"path": _relative(context.config_path, context.repository_root),
                   "sha256": context.config_sha256, "digest": context.config_digest},
        "evidence_manifest": evidence_manifest,
        "evidence_sha256": {logical: paths.sha256_file(context.repository_root / logical)
                            for logical in evidence_files},
        "inputs": dict(sorted(inputs.items())), "input_sha256": input_hashes,
        "input_count": len(input_hashes),
        "preflight": {"path": context.run.logical(PREFLIGHT_JSON),
                      "sha256": paths.sha256_file(preflight_path),
                      "binding": dict(current)},
        "tolerances": dict(context.config["tolerances"]),
        "statistics": dict(context.config["statistics"]),
        "decision": dict(context.config["decision"]),
        "environment": environment_record(),
        "note": ("this marker is ignored by Git on purpose: it names the commit, it is not part "
                 "of it. The CLI never commits; scoring refuses to run without this file and "
                 "re-hashes everything it names.")}
    paths.write_json(freeze_marker_path(context), marker)
    return marker


# ---------------------------------------------------------------------------
# stage: score
# ---------------------------------------------------------------------------

def strata_for_bank(context, bank, parent_log_probability, train_frame):
    """The three predeclared stratifications, all independent of post-training scores."""
    from .her2_data import encode_cores, nearest_training_labels
    statistics = context.config["statistics"]
    train_index = encode_cores(train_frame.seq)
    labels = (train_frame["class"] == "high").to_numpy().astype(np.float64)
    lookup = nearest_training_labels(bank.index, train_index, labels, max_distance=2)
    nearest = lookup.strata().astype(object)
    membership = scoring.membership_labels(lookup.distance == 0)
    quartiles, quartile_meta = scoring.quartile_labels(
        parent_log_probability, parts=int(statistics["parent_lp_quartiles"]))
    return {
        "nearest_training_distance": {
            "labels": nearest,
            "categories": list(statistics["nearest_training_bins"]),
            "definition": ("exact Hamming distance to the nearest training core, capped at >=3. "
                           "Not the within-pair chosen/rejected Hamming used by the CHES "
                           "analysis.")},
        "training_catalogue_membership": {
            "labels": membership, "categories": ["in_training", "not_in_training"],
            "definition": ("distance-zero presence in the training catalogue. Identities absent "
                           "from the catalogue stay unlabelled; no reserved test or SPR outcome "
                           "is consulted.")},
        "parent_log_probability_quartile": {
            "labels": quartiles,
            "categories": [f"q{k}" for k in range(1, int(statistics["parent_lp_quartiles"]) + 1)],
            "binning": quartile_meta,
            "definition": "quartiles of the parent's own sequence log probability, ties together"}}


def score_one(context, *, record, drop, index_matrix, strata, historical):
    """Statistics, strata and the labelled historical join for one checkpoint."""
    statistics = context.config["statistics"]
    thresholds = {name: float(value) for name, value in statistics["drop_thresholds"].items()}
    block = scoring.drop_statistics(
        drop, index_matrix, thresholds=thresholds,
        quantiles=tuple(statistics["quantiles"]), confidence=float(statistics["confidence"]),
        z=float(statistics["wilson_z"]), label=record["id"])
    block["strata"] = {
        name: dict(scoring.stratum_statistics(
            drop, entry["labels"], index_matrix, categories=entry["categories"],
            thresholds=thresholds, confidence=float(statistics["confidence"]),
            z=float(statistics["wilson_z"])), definition=entry["definition"])
        for name, entry in sorted(strata.items())}
    block["historical_metrics"] = historical
    return block


def historical_join(record):
    """Historical numbers, each carried with the population it was measured on.

    The values are read from ``historical_metrics_source``, which the enumeration
    copied out of the endpoint document. Reading them from top-level record keys --
    which is what an earlier version did -- produced a table of ``null`` for every
    checkpoint, because no enumerator ever wrote those keys: the unique fraction,
    the ranking and the reverse-KL column were all silently empty.
    """
    source = dict(record.get("historical_metrics_source") or {})
    missing_reason = source.get("reason")
    per_field = dict(source.get("not_recorded_by_row") or {})

    def entry(key, population, **extra):
        value = source.get(key)
        block = {"value": value, "population": population, **extra}
        if value is None:
            block["reason"] = (per_field.get(key) or missing_reason or
                               f"the {record.get('role')} document for this state records no "
                               f"{key!r}")
        if source.get("key"):
            block["source_row"] = source["key"]
        return block

    return {
        "parent_kl": entry("parent_kl",
                           "10,000 temperature-1 draws from THIS checkpoint",
                           direction="KL(policy || parent), reverse of this audit's estimand"),
        "generation": entry("generation",
                            "10,000 temperature-1 draws from this checkpoint"),
        "diversity": entry("diversity",
                           "10,000 temperature-1 draws from this checkpoint"),
        "ranking": entry("ranking",
                         "the campaign's own ranking key, where one exists"),
        "validation": entry("val_metrics",
                            "all validation rows, mean log probability per residue"),
        "chosen_drop": entry("val_pair_metrics",
                             "the fixed validation preference pairs"),
        "validation_strata": entry("val_strata",
                                   "the fixed validation rows, by declared stratum"),
        "eligibility": {"gate_passed": record.get("gate_passed"),
                        "diversity_eligible": record.get("diversity_eligible"),
                        "population": "the declared endpoint eligibility criteria"},
        "note": ("these columns come from the historical campaigns and are labelled by their own "
                 "population. They are not this audit's forward-KL measurement.")}


def paired_method_differences(drops, *, inventory_records, statistics, matrices):
    """Same-bank, same-rows differences between methods at matched endpoints.

    ``drops`` maps checkpoint id to its per-row drop vector. Two checkpoints are
    compared only when they share a seed, a budget and therefore a bank and a row
    order: the difference is then per-row, and the bootstrap resamples both sides
    on the one index matrix belonging to that seed. That is what makes the interval
    a paired one rather than two independent intervals subtracted.
    """
    by_id = {record["id"]: record for record in inventory_records}
    groups = {}
    for identifier in sorted(drops):
        record = by_id.get(identifier) or {}
        if record.get("role") != inventory_lib.ROLE_ENDPOINT:
            continue
        budget = record.get("nominal_budget_gpu_seconds")
        if budget is None:
            continue
        groups.setdefault((int(record["seed"]), float(budget)), []).append(identifier)
    out = {}
    for (seed, budget), members in sorted(groups.items()):
        matrix = matrices.get(seed)
        if matrix is None or len(members) < 2:
            continue
        for position, left in enumerate(sorted(members)):
            for right in sorted(members)[position + 1:]:
                if (by_id[left].get("arm_id") == by_id[right].get("arm_id")):
                    continue
                label = f"seed{seed}::budget{int(budget)}::{left}__minus__{right}"
                out[label] = dict(scoring.paired_difference(
                    drops[left], drops[right], matrix,
                    confidence=float(statistics["confidence"])),
                    seed=seed, budget_gpu_seconds=budget, left=left, right=right,
                    left_arm=by_id[left].get("arm_id"), right_arm=by_id[right].get("arm_id"),
                    definition=("mean per-row difference of the signed parent-minus-policy drop "
                                "between two methods at the same seed, budget, bank and row "
                                "order"))
    return {"comparisons": out, "count": len(out),
            "note": ("only same-seed, same-budget, same-bank pairs are compared. Cross-seed "
                     "differences are not formed: the rows are different objects and a shared "
                     "bootstrap index would be meaningless.")}


# ---------------------------------------------------------------------------
# summaries, decision inputs and the report
# ---------------------------------------------------------------------------

def decision_inputs(checkpoints, *, settings, inventory_records):
    """Per-method, per-seed 600 s endpoints, with unusable seeds named as unusable."""
    by_id = {record["id"]: record for record in inventory_records}
    endpoint = float(settings["endpoint_gpu_seconds"])
    out = {}
    for method in settings["methods"]:
        rows = []
        for seed in settings["seeds"]:
            match = None
            for identifier, block in sorted(checkpoints.items()):
                record = by_id.get(identifier)
                if record is None:
                    continue
                if (record.get("arm_id") == method and int(record.get("seed", -1)) == int(seed)
                        and record.get("role") == inventory_lib.ROLE_ENDPOINT
                        and float(record.get("nominal_budget_gpu_seconds") or -1) == endpoint):
                    match = (identifier, block)
                    break
            if match is None:
                rows.append({"seed": int(seed), "usable": False,
                             "reason": (f"no verified {method} endpoint at {endpoint:g} GPU "
                                        f"seconds for seed {seed}")})
                continue
            identifier, block = match
            counts = block["tails"]["counts"]
            rows.append({"seed": int(seed), "usable": True, "checkpoint_id": identifier,
                         "tenfold_wilson_lower": counts["tenfold"]["lower"],
                         "hundredfold_wilson_lower": counts["hundredfold"]["lower"],
                         "tenfold_fraction": counts["tenfold"]["fraction"],
                         "hundredfold_fraction": counts["hundredfold"]["fraction"],
                         "forward_kl": block["forward_kl"]["mean"]})
        out[method] = rows
    return out


def support_versus_diversity(checkpoints, inventory_records):
    """One row per scored endpoint: tail fraction against the historical uniqueness."""
    by_id = {record["id"]: record for record in inventory_records}
    rows = []
    for identifier, block in sorted(checkpoints.items()):
        record = by_id.get(identifier) or {}
        generation = ((block.get("historical_metrics") or {}).get("generation") or {}).get("value")
        rows.append({
            "checkpoint_id": identifier, "role": record.get("role"),
            "arm_id": record.get("arm_id"), "seed": record.get("seed"),
            "nominal_budget_gpu_seconds": record.get("nominal_budget_gpu_seconds"),
            "forward_kl": block["forward_kl"]["mean"],
            "forward_kl_ci_low": block["forward_kl"]["ci_low"],
            "forward_kl_ci_high": block["forward_kl"]["ci_high"],
            "tenfold_fraction": block["tails"]["counts"]["tenfold"]["fraction"],
            "tenfold_wilson_lower": block["tails"]["counts"]["tenfold"]["lower"],
            "hundredfold_fraction": block["tails"]["counts"]["hundredfold"]["fraction"],
            "unique_fraction": (generation or {}).get("unique_fraction"),
            "diversity_eligible": record.get("diversity_eligible"),
            "diversity_population": ("10,000 temperature-1 draws from the checkpoint; the support "
                                     "columns are parent draws and are a different population")})
    return rows


def render_report(context, *, inventory, checkpoints, ches_summary, decision, coverage, timings,
                  figures, paired=None, publication=None):
    """The published narrative. Every 159 semantic states appear, not only the winners.

    ``publication`` is the publication manifest when this text is the copy destined
    for ``reference/``: the data section then links the published files relative to
    the published report, instead of naming run-root paths a reader cannot open.
    """
    config = context.config
    lines = [
        "# HER2 support-preservation audit", "",
        f"Protocol: [{config['protocol']}](../{config['protocol']}).",
        "",
        "This is an inference-only, retrospective audit of parent-support preservation in HER2",
        "post-training. It measures forward KL from persisted temperature-1 parent draws and the",
        "tails of the per-sequence log-probability drop, and it relates parent CHES to later",
        "chosen-likelihood displacement. Nothing here is a binding-affinity measurement, and",
        "nothing here is evidence about newly generated designs.", "",
        "## What was inventoried", "",
        f"- Declared states: **{coverage['expected'].get('total')}**; "
        f"verified: **{coverage['by_status'].get('verified', 0)}**.",
        f"- Distinct computations after state deduplication: "
        f"**{inventory['deduplication']['distinct_computations']}**.",
        f"- Guarded trajectories: {inventory['trajectories']['completed']} completed, "
        f"{inventory['trajectories']['stopped']} stopped.",
        "- Diagnostic snapshots are never relabelled as nominal-budget endpoints.", "",
        "| role | enumerated | verified | expected |", "| --- | --- | --- | --- |"]
    for role, count in sorted(coverage["by_role"].items()):
        lines.append(f"| {role} | {count} | "
                     f"{coverage.get('verified_by_role', {}).get(role, 0)} | "
                     f"{(coverage['expected'].get('by_role') or {}).get(role, '—')} |")
    lines += ["", "## Coverage and missing states", ""]
    if coverage["shortfalls"]:
        lines.append("| role | kind | expected | observed |")
        lines.append("| --- | --- | --- | --- |")
        for entry in coverage["shortfalls"]:
            lines.append(f"| {entry['role']} | {entry.get('kind', 'enumeration')} | "
                         f"{entry['expected']} | {entry['observed']} |")
        lines.append("")
        lines.append("A shortfall of kind `verification` means the row was enumerated but its")
        lines.append("artifact did not verify. That is missing coverage, not a measurement.")
    else:
        lines.append("Every declared population was located **and verified**: the enumerated and")
        lines.append("verified counts both equal the declared ones for every role.")
    unverified = [record for record in inventory["records"]
                  if record.get("status") != inventory_lib.STATUS_VERIFIED]
    lines += ["", f"Unverified or missing states: **{len(unverified)}**."]
    for record in unverified:
        lines.append(f"- `{record['id']}` — {record.get('status')}: {record.get('status_reason')}")

    # Every semantic state, not only the primaries. An alias is a real declared
    # endpoint; leaving it out of the table is how a reused control disappears from
    # a report that claims full coverage.
    lines += ["", "## Every semantic state", "",
              f"All **{len(inventory['records'])}** enumerated states, including the reuse "
              "aliases that are scored once under another id.", "",
              "| id | role | arm | seed | budget s | status | scored as |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for record in sorted(inventory["records"], key=lambda item: item["id"]):
        deduplication = record.get("deduplication") or {}
        budget = record.get("nominal_budget_gpu_seconds")
        scored_as = deduplication.get("scored_as")
        if record.get("status") != inventory_lib.STATUS_VERIFIED:
            scored_as = f"not scored ({record.get('status')})"
        elif scored_as and not deduplication.get("is_primary"):
            scored_as = f"`{scored_as}` (alias)"
        else:
            scored_as = "itself"
        lines.append(
            f"| `{record['id']}` | {record.get('role')} | {record.get('arm_id')} | "
            f"{record.get('seed')} | {'—' if budget is None else format(float(budget), 'g')} | "
            f"{record.get('status')} | {scored_as} |")

    lines += ["", "## Every scored computation", "",
              "| id | role | arm | seed | budget s | forward KL | 95% CI | >ln10 | Wilson lower |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    by_id = {record["id"]: record for record in inventory["records"]}
    for identifier, block in sorted(checkpoints.items()):
        record = by_id.get(identifier, {})
        kl = block["forward_kl"]
        tail = block["tails"]["counts"]["tenfold"]
        budget = record.get("nominal_budget_gpu_seconds")
        budget_text = "—" if budget is None else format(float(budget), "g")
        lines.append(
            f"| `{identifier}` | {record.get('role')} | {record.get('arm_id')} | "
            f"{record.get('seed')} | {budget_text} | "
            f"{kl['mean']:.4f} | [{kl['ci_low']:.4f}, {kl['ci_high']:.4f}] | "
            f"{tail['fraction']:.5f} | {tail['lower']:.5f} |")

    lines += ["", "## Support versus diversity", "",
              "Diversity numbers are historical and were measured on **policy** draws; the support",
              "columns beside them are **parent** draws. They are different populations and the",
              "table never averages across them.", ""]
    lines += ["| checkpoint | >ln10 fraction | unique fraction (historical) | diversity eligible |",
              "| --- | --- | --- | --- |"]
    for row in support_versus_diversity(checkpoints, inventory["records"]):
        if row["role"] != inventory_lib.ROLE_ENDPOINT:
            continue
        unique = row["unique_fraction"]
        unique_text = "—" if unique is None else format(float(unique), ".4f")
        lines.append(f"| `{row['checkpoint_id']}` | {row['tenfold_fraction']:.5f} | "
                     f"{unique_text} | {row['diversity_eligible']} |")

    comparisons = (paired or {}).get("comparisons") or {}
    if comparisons:
        lines += ["", "## Paired method differences", "",
                  "Same seed, same budget, same bank rows, same bootstrap indices on both sides.",
                  "Cross-seed differences are not formed.", "",
                  "| comparison | mean difference | 95% CI |", "| --- | --- | --- |"]
        for label, block in sorted(comparisons.items()):
            lines.append(f"| `{label}` | {block['mean']:.4f} | "
                         f"[{block['ci_low']:.4f}, {block['ci_high']:.4f}] |")

    lines += ["", "## CHES", ""]
    if ches_summary is None:
        lines.append("The CHES stage has not been run in this revision.")
    else:
        overall = (ches_summary.get("parent") or {})
        lines.append("Parent CHES is computed first, on both declared populations, and related to")
        lines.append("later `log p_parent(chosen) - log p_checkpoint(chosen)`. Associations are")
        lines.append("descriptive: training changed both the representations and the likelihoods.")
        lines.append("")
        lines.append("| seed | population | pairs | mean CHES |")
        lines.append("| --- | --- | --- | --- |")
        for key, block in sorted(overall.items()):
            summary = block.get("ches_summary") or {}
            mean = summary.get("mean")
            lines.append(f"| {block.get('seed')} | {block.get('population_id')} | "
                         f"{summary.get('rows')} | "
                         f"{'—' if mean is None else format(mean, '.4f')} |")

        associations = ches_summary.get("associations") or {}
        if associations:
            lines += ["", "### Parent CHES versus displacement", "",
                      "| endpoint / population | pairs | Spearman (average ranks) | "
                      "mean displacement |",
                      "| --- | --- | --- | --- |"]
            for key, block in sorted(associations.items()):
                whole = block.get("overall") or {}
                rho = whole.get("spearman")
                mean = (((ches_summary.get("endpoints") or {}).get(key) or {})
                        .get("displacement") or {}).get("mean")
                lines.append(
                    f"| `{key}` | {whole.get('n')} | "
                    f"{'—' if rho is None else format(rho, '.4f')} | "
                    f"{'—' if mean is None else format(mean, '.4f')} |")
            lines.append("")
            lines.append("No p-value is attached: " + str(
                (ches_summary.get("spearman") or {}).get("p_value_reason")))

        increments = ches_summary.get("increments") or {}
        if increments:
            lines += ["", "### Early-checkpoint CHES and subsequent increments", "",
                      "| run / population / step | pairs | mean increment | "
                      "early-CHES Spearman | control |",
                      "| --- | --- | --- | --- | --- |"]
            for key, block in sorted(increments.items()):
                association = block.get("early_ches_association") or {}
                rho = (association.get("overall") or {}).get("spearman")
                control = block.get("control") or block.get("control_matched") or {}
                label = ("matched" if control.get("matched") else
                         f"uncontrolled ({control.get('reason')})")
                mean_difference = control.get("mean_difference")
                if mean_difference is not None:
                    label += f", difference {mean_difference:.4f}"
                lines.append(f"| `{key}` | {block.get('pairs')} | "
                             f"{block.get('mean_increment'):.4f} | "
                             f"{'—' if rho is None else format(rho, '.4f')} | {label} |")
        gaps = ches_summary.get("increment_gaps") or []
        if gaps:
            lines += ["", "Increments that could not be formed, reported rather than skipped:"]
            for gap in gaps:
                lines.append(f"- `{gap.get('label')}` — {gap.get('reason')}")
        lines.append("")
        lines.append("Missing by construction: the v1 continued-SFT control exists only at 180,")
        lines.append("360 and 600 seconds, so the 1200 s and 1800 s DPO increments are reported")
        lines.append("**uncontrolled**. No control was invented for them.")
        lines.append("")
        lines.append(f"Machine-readable: `{context.run.logical(CHES_JSON)}` and the per-endpoint")
        lines.append("`ches/<checkpoint>/<population>.npz` shards with their completion records.")

    lines += ["", "## Decision", "",
              f"Outcome: **{decision['outcome']}**.", "",
              "> " + decision["rule"], ""]
    lines.append("| method | seeds usable | seeds crossing | outcome |")
    lines.append("| --- | --- | --- | --- |")
    for method, block in sorted(decision["methods"].items()):
        lines.append(f"| {method} | {block['seeds_usable']}/{block['seeds_declared']} | "
                     f"{block['seeds_crossing']} | {block['outcome']} |")
    for method, block in sorted(decision["methods"].items()):
        lines += ["", f"**{method}** — {block['outcome_note']}", "",
                  "| seed | usable | >ln10 Wilson lower | >ln100 Wilson lower | crosses |",
                  "| --- | --- | --- | --- | --- |"]
        for row in block["seeds"]:
            lower10 = row.get("tenfold_wilson_lower")
            lower100 = row.get("hundredfold_wilson_lower")
            lines.append(
                f"| {row.get('seed')} | {row.get('usable')} | "
                f"{'—' if lower10 is None else format(lower10, '.5f')} | "
                f"{'—' if lower100 is None else format(lower100, '.5f')} | "
                f"{row.get('crosses') if row.get('usable') else row.get('reason')} |")
    if decision.get("blocking"):
        lines += ["", "This audit cannot support a preservation finding yet:"]
        for reason in decision["blocking"]:
            lines.append(f"- {reason}")
    lines += ["", "### Limitations", "",
              f"- {decision['resolution_caveat']}",
              f"- {decision['retrospective_caveat']}",
              f"- {decision['biology_caveat']}",
              f"- {decision['ches_note']}",
              "- Bootstrap intervals characterize finite-bank sampling variation at fixed models.",
              "  They do not repair historical selection and give no simultaneous coverage over",
              "  the whole checkpoint screen.", ""]
    if decision["outcome"] == scoring.DECISION_NO_ESCALATION:
        lines.append("Summary phrase for this result: **no detected abandonment at the resolution")
        lines.append("of this audit**, qualified by the measured tails and coverage above.")
        lines.append("")
    # Only the stages that produced the measurements. The report's own cost is not
    # knowable while it is being rendered, and reading it on a *second* run would
    # make the report a function of how many times it had been rebuilt.
    lines += ["## Cost", "",
              "| stage | inference s | I/O s | analysis s | wall s |",
              "| --- | --- | --- | --- | --- |"]
    for stage in COSTED_STAGES:
        block = timings.get(stage)
        if block is None:
            continue
        lines.append(f"| {stage} | {block.get('inference_seconds', 0):.1f} | "
                     f"{block.get('io_seconds', 0):.1f} | "
                     f"{block.get('analysis_seconds', 0):.1f} | "
                     f"{block.get('wall_seconds', 0):.1f} |")
    lines += ["", "The reporting and verification stages are excluded: they measure the cost of "
                  "rendering, not of the measurement, and including them would make this table "
                  "depend on how often the report was rebuilt.", ""]
    if figures:
        lines += ["", "## Figures", ""]
        for key, entry in sorted(figures.items()):
            name = entry.get("name", key)
            if entry.get("written"):
                lines.append(f"![{entry.get('caption')}](figures/{name})")
                lines.append("")
                lines.append(f"*{name}* — {entry.get('caption')}")
                lines.append("")
            else:
                lines.append(f"- `{name}` — not rendered: {entry.get('reason')}")
    if publication:
        lines += ["", "## Data behind this report", "",
                  "Every link below is relative to this file.", ""]
        for logical in sorted(publication["files"]):
            lines.append(f"- [{logical}]({logical})")
        lines += ["", "The full per-row score vectors, the shard records and the completion "
                      "manifest stay in the ignored run directory; what is published here is the "
                      "table this report cites.", ""]
    else:
        lines += ["", "## Data behind this report", "",
                  "Numeric artifacts for this run, under the ignored run root "
                  f"`{context.run.logical_run_root}`:", ""]
        for logical in (CHECKPOINTS_JSON, CHES_JSON, COVERAGE_JSON, DECISION_JSON,
                        "verification.json"):
            lines.append(f"- `{logical}`")
        lines.append("")
    lines += ["", "## Rebuilding this report", "",
              "```", f"python scripts/audit_her2_support.py report --config "
                     f"{_relative(context.config_path, context.repository_root)}", "```", ""]
    return "\n".join(lines) + "\n"


#: Metadata matplotlib would otherwise stamp into every export. Without this a
#: second render of the *same numbers* is a different file, so an immutable figure
#: could never be verified or published byte-for-byte.
FIGURE_METADATA = {"Software": None, "Creation Time": None, "Date": None}


def save_figure(figure, path, *, dpi=150, context=None):
    """Write one figure deterministically, once, and return its entry.

    Deterministic because the export metadata matplotlib would otherwise stamp in
    is suppressed, and *once* because a figure already on disk is never overwritten:
    a published figure is a deliverable somebody may have cited. If a re-render
    disagrees byte for byte, the original stays and the disagreement is reported --
    it is a rendering difference, not a new measurement, and it must not be able to
    replace a figure in place. A figure edited after the fact is caught by the
    completion manifest, not by this comparison.
    """
    path = Path(path)
    keys = ("Date",) if path.suffix.lower() == ".svg" else ("Software", "Creation Time")
    metadata = {key: FIGURE_METADATA[key] for key in keys}
    entry = {"name": path.name, "written": True}
    if path.is_file():
        temporary = path.with_name(path.name + ".rerender")
        figure.savefig(temporary, dpi=dpi, metadata=metadata)
        rendered = paths.sha256_file(temporary)
        temporary.unlink()
        entry["sha256"] = paths.sha256_file(path)
        if rendered != entry["sha256"]:
            entry["rerendered_bytes_differ"] = {
                "rendered_sha256": rendered, "kept": "the original file",
                "reason": ("a completed figure is not replaced in place. If this differs the "
                           "renderer is not byte-reproducible here; the numbers behind it are in "
                           "the published tables either way")}
        return path.name, entry
    figure.savefig(path, dpi=dpi, metadata=metadata)
    entry["sha256"] = paths.sha256_file(path)
    if context is not None:
        paths.record_completion(context.run.run_root,
                                context.run.logical(_relative(path, context.run.run_root)),
                                path, kind="figure", scientific_digest=entry["sha256"])
    return path.name, entry


def render_figures(context, *, checkpoints, inventory, ches_summary):
    """Small matplotlib figures. Absent matplotlib is recorded, never silently skipped.

    Each entry carries the run-relative ``logical`` it was written to and the bare
    ``name`` the report links to, so the same markdown resolves beside the report in
    the run directory and beside the published copy under ``reference/figures``.
    """
    figures = {}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        return {"her2-support-forward-kl.png": {
            "written": False, "name": "her2-support-forward-kl.png",
            "reason": f"matplotlib unavailable: {error}"}}
    directory = context.run.path("report/figures")
    directory.mkdir(parents=True, exist_ok=True)
    rows = [row for row in support_versus_diversity(checkpoints, inventory["records"])
            if row["role"] == inventory_lib.ROLE_ENDPOINT and row["forward_kl"] is not None]

    figure, axes = plt.subplots(figsize=(7.0, 4.2))
    if rows:
        axes.scatter([row["tenfold_fraction"] for row in rows],
                     [row["unique_fraction"] or float("nan") for row in rows], s=18)
    axes.set_xlabel("fraction of parent draws with drop > ln 10 (parent draws)")
    axes.set_ylabel("historical unique fraction (policy draws)")
    axes.set_title("Support versus diversity: two different populations")
    figure.tight_layout()
    name, entry = save_figure(figure, directory / "her2-support-vs-diversity.png",
                              context=context)
    plt.close(figure)
    figures[name] = dict(entry, logical=f"report/figures/{name}", caption=(
        "tenfold-drop fraction on parent draws against the historical uniqueness of policy "
        "draws; axes are different populations and the plot says so"))

    figure, axes = plt.subplots(figsize=(7.0, 4.2))
    if rows:
        order = sorted(rows, key=lambda item: item["forward_kl"])
        axes.errorbar(range(len(order)), [row["forward_kl"] for row in order],
                      yerr=[[row["forward_kl"] - row["forward_kl_ci_low"] for row in order],
                            [row["forward_kl_ci_high"] - row["forward_kl"] for row in order]],
                      fmt="o", markersize=3, linewidth=0.8)
    axes.set_xlabel("scored endpoints, ordered by estimate")
    axes.set_ylabel("forward KL (nats / ten-residue sequence)")
    axes.set_title("Forward KL from persisted parent draws, 95% bootstrap intervals")
    figure.tight_layout()
    name, entry = save_figure(figure, directory / "her2-support-forward-kl.png", context=context)
    plt.close(figure)
    figures[name] = dict(entry, logical=f"report/figures/{name}",
                         caption="per-endpoint forward KL with paired-row bootstrap intervals")

    if ches_summary:
        figure, axes = plt.subplots(figsize=(7.0, 4.2))
        plotted = 0
        for key, block in sorted((ches_summary.get("associations") or {}).items()):
            deciles = ((block.get("by_parent_ches_decile") or {}).get("mean_displacement") or {})
            values = [deciles.get(f"d{k}", {}).get("mean") for k in range(1, 11)]
            if any(value is not None for value in values):
                axes.plot(range(1, 11), [np.nan if v is None else v for v in values],
                          marker="o", markersize=3, linewidth=0.9, label=str(key))
                plotted += 1
        axes.set_xlabel("parent CHES decile (ties kept together)")
        axes.set_ylabel("mean chosen-likelihood displacement (nats)")
        axes.set_title("Parent CHES decile versus later displacement")
        if plotted:
            axes.legend(fontsize=6, ncol=2)
        figure.tight_layout()
        name, entry = save_figure(figure, directory / "her2-support-ches.png", context=context)
        plt.close(figure)
        figures[name] = dict(entry, logical=f"report/figures/{name}",
                             caption="displacement by parent-CHES decile, per endpoint")
    return figures


# ---------------------------------------------------------------------------
# immutability
# ---------------------------------------------------------------------------

def shard_directory(kind, identifier):
    """The run-relative directory one checkpoint's shards of ``kind`` live in."""
    return f"{kind}/{identifier.replace('::', '_')}"


def expected_run_outputs(context):
    """Every output the run's own summaries claim, as completion-manifest keys.

    Derived from the summaries and the inventory rather than from a directory scan.
    A scan answers "what is still here", which is the one question that cannot
    detect a deleted output: the missing shard simply is not in the listing, and the
    completion summary keeps claiming it.
    """
    expected = []
    for logical in (INVENTORY_JSON, AUDIT_CONFIG_JSON, PREFLIGHT_JSON, DECISION_JSON,
                    COVERAGE_JSON, CHECKPOINTS_JSON, CHES_JSON):
        if context.run.path(logical).is_file():
            expected.append(context.run.logical(logical))

    def shard_keys(kind, identifier, name):
        prefix = shard_directory(kind, identifier)
        return list(paths.shard_logicals(prefix, name))

    summary_path = context.run.path(CHECKPOINTS_JSON)
    if summary_path.is_file():
        summary = paths.read_json(summary_path)
        for identifier in sorted(summary.get("checkpoints") or {}):
            expected.extend(shard_keys("scores", identifier, "sequence_scores"))
        for identifier in sorted(summary.get("self_controls") or {}):
            expected.extend(shard_keys("parent_banks", identifier, "parent_scores"))
    ches_path = context.run.path(CHES_JSON)
    if ches_path.is_file():
        ches_document = paths.read_json(ches_path)
        for key in sorted(list(ches_document.get("parent") or {})
                          + list(ches_document.get("endpoints") or {})):
            checkpoint_id, _, population_id = str(key).rpartition("::")
            if checkpoint_id and population_id:
                expected.extend(shard_keys("ches", checkpoint_id, population_id))
    report_path = context.run.path(REPORT_MD)
    if report_path.is_file():
        expected.append(context.run.logical(REPORT_MD))
    return sorted(set(expected))


def verify_outputs(context):
    """Re-check every completed artifact against the **saved completion manifest**.

    Two things are verified, and they are different claims:

    * every shard still reproduces its recorded content digests, dtypes, shapes,
      order and container hash (:func:`her2_support_paths.read_shard`);
    * every artifact the completion manifest recorded still has exactly the bytes,
      and therefore the original timing block, it had when it completed.

    The second one is the point. Re-hashing a file against its own canonical
    re-serialization, or comparing a timing block to the same block read a moment
    later, proves only that the file is internally consistent -- which a tampered
    file also is.
    """
    run_root = context.run.run_root
    checked, problems = [], []
    for record_path in sorted(run_root.rglob(f"*{paths.SHARD_RECORD}")):
        record = paths.read_json(record_path)
        name = record.get("shard")
        logical = str(record_path.relative_to(run_root)).replace("\\", "/")
        try:
            paths.read_shard(record_path.parent, name)
        except (ValueError, OSError) as error:
            problems.append({"shard": logical, "problem": f"{type(error).__name__}: {error}"})
            continue
        entry = {"shard": logical, "arrays": sorted(record.get("arrays") or {}),
                 "record_bytes_canonical": (paths.sha256_file(record_path)
                                            == paths.digest_document(record)),
                 "has_timing_block": isinstance(record.get("timings"), dict),
                 "timings": record.get("timings")}
        if not entry["record_bytes_canonical"]:
            problems.append({"shard": logical,
                             "problem": ("the completion record's raw bytes are not its canonical "
                                         "form; it was edited or newline-translated")})
        if not entry["has_timing_block"]:
            problems.append({"shard": logical,
                             "problem": "the completion record carries no timing block"})
        checked.append(entry)
    documents = {}
    for logical in (INVENTORY_JSON, AUDIT_CONFIG_JSON, MANIFEST_JSON, PREFLIGHT_JSON,
                    DECISION_JSON, COVERAGE_JSON, CHECKPOINTS_JSON, CHES_JSON, COMPLETE_JSON):
        path = context.run.path(logical)
        if not path.is_file():
            continue
        document = paths.read_json(path)
        observed = paths.sha256_file(path)
        expected = paths.digest_document(document)
        documents[logical] = {"sha256": observed, "document_digest": expected,
                              "canonical_bytes": observed == expected}
        if observed != expected:
            problems.append({"artifact": logical,
                             "problem": ("raw file bytes do not equal the canonical document "
                                         "digest; newline translation or an external edit")})

    def resolve(logical):
        relative = logical[len(context.run.logical_run_root) + 1:] \
            if logical.startswith(context.run.logical_run_root + "/") else logical
        try:
            return context.run.path(relative)
        except ValueError:
            return None

    expected = expected_run_outputs(context)
    completion = paths.verify_completions(run_root, resolve=resolve, expected=expected)
    problems.extend(completion["problems"])
    return {"schema_version": paths.AUDIT_SCHEMA, "record_kind": "verification",
            "run_root": context.run.logical_run_root, "shards_checked": len(checked),
            "shards": checked, "documents": documents,
            "expected_outputs": expected,
            "expected_output_count": len(expected),
            "completion_manifest": {"artifacts_checked": completion["artifacts_checked"],
                                    "artifacts": completion["artifacts"]},
            "problems": problems,
            "immutable": not problems,
            "note": ("completed artifacts are verified against the saved completion manifest, "
                     "never rewritten to refresh their elapsed time. A changed result goes to a "
                     "new revision directory.")}


def require_new_or_identical(path, document, *, what, context=None, kind=None):
    """Write ``document`` once; on a rerun verify the **scientific** projection.

    Comparing whole documents is what made a rerun impossible: every summary
    carries ``generated_at`` and a timing block, so a second run that reproduced
    the science exactly would still be rejected -- after paying for the GPU work.
    The scientific projection is compared instead, and on agreement the original
    file, including its original timings, stays exactly as it was written. A
    difference in the science is still refused.
    """
    path = Path(path)
    digest = paths.digest_document(paths.scientific_projection(document))
    if not path.is_file():
        written = paths.write_json(path, document)
        if context is not None:
            paths.record_completion(context.run.run_root,
                                    context.run.logical(_relative(path, context.run.run_root)),
                                    path, kind=kind or what, scientific_digest=digest,
                                    timings=document.get("timings"))
        return written, "written"
    existing = paths.read_json(path)
    if paths.scientific_projection(existing) == paths.scientific_projection(document):
        if context is not None:
            paths.record_completion(context.run.run_root,
                                    context.run.logical(_relative(path, context.run.run_root)),
                                    path, kind=kind or what, scientific_digest=digest,
                                    timings=existing.get("timings"))
        return paths.sha256_file(path), "verified_identical_scientific_content"
    raise ValueError(
        f"{what} already exists at {path} with different scientific content. A completed artifact "
        "is not mutated to refresh it; choose a new revision directory for a different result.")


def require_new_or_identical_text(path, text, *, what, context=None, kind=None):
    """The text-file twin of :func:`require_new_or_identical`.

    A report has no scientific projection to take -- it is the rendered bytes -- so
    the comparison is exact. A rerun that produces the same report keeps the
    original file and its original completion record; one that produces different
    text is refused rather than overwriting a published deliverable in place.
    """
    path = Path(path)
    digest = paths.sha256_text(text)
    if path.is_file():
        observed = paths.sha256_file(path)
        require(observed == digest,
                f"{what} already exists at {path} with different bytes ({observed} vs {digest}). "
                "A completed report is not rewritten; a different result belongs in a new revision "
                "directory.")
    else:
        paths.write_text(path, text)
    if context is not None:
        paths.record_completion(context.run.run_root,
                                context.run.logical(_relative(path, context.run.run_root)),
                                path, kind=kind or what, scientific_digest=digest)
    return digest


def completed_stage(context, outputs):
    """Documents of an already-completed stage, or ``None`` if it has not completed.

    "Completed" means: every declared output exists, the completion manifest
    recorded it, and the bytes on disk still match that record. It is deliberately
    not "the progress file says completed" -- a status is a claim about a process,
    and this has to be a claim about artifacts. When it holds, the caller reuses the
    recorded science instead of repeating the GPU work that produced it.
    """
    artifacts = (paths.read_completion_manifest(context.run.run_root).get("artifacts") or {})
    documents, problems = {}, []
    for name, logical in sorted(outputs.items()):
        path = context.run.path(logical)
        if not path.is_file():
            return None
        entry = artifacts.get(context.run.logical(logical))
        if entry is None:
            problems.append(f"{logical} exists but was never recorded complete")
            continue
        observed = paths.sha256_file(path)
        if observed != entry["sha256"]:
            problems.append(f"{logical} hashes {observed}, the completion manifest recorded "
                            f"{entry['sha256']}")
            continue
        documents[name] = paths.read_json(path)
    require(not problems,
            "A stage output is on disk but does not match the saved completion manifest: "
            + "; ".join(problems)
            + ". A rerun verifies the recorded artifacts; it does not adopt whatever is there now.")
    return documents


def ches_identity(block, record):
    """What a CHES shard is bound to: the frozen pair order AND the weights scored.

    The pair population alone is not enough. Two runs can share a population and
    differ in the state that was loaded, and a shard reused across that difference
    would attribute one checkpoint's displacement to another.
    """
    return dict(block["identity"], state_digest=record.get("state_digest_audit_computed"))


def reusable_shard(context, kind, identifier, name, *, identity=None):
    """``(arrays, record)`` for a completed, ledger-bound shard, else ``None``.

    This is what makes an interrupted stage restartable: the states that finished
    are validated and reused, and only the unfinished ones are computed. The
    identity is compared before anything is reused, so a shard produced under a
    different parent, bank or convention is recomputed rather than adopted.
    """
    prefix = shard_directory(kind, identifier)
    directory = context.run.path(prefix)
    if not paths.shard_is_complete(directory, name):
        registered = paths.read_completion_manifest(context.run.run_root).get("artifacts") or {}
        if any(key in registered for key in paths.shard_logicals(prefix, name)):
            paths.require_registered_shard(context.run.run_root, directory, name,
                                           logical_prefix=prefix)
        return None
    paths.require_registered_shard(context.run.run_root, directory, name, logical_prefix=prefix)
    arrays, record = paths.read_shard(directory, name)
    if identity is not None:
        require(record.get("identity") == identity,
                f"{prefix}/{name} was produced under a different identity; a cached result is "
                "bound to the state, the bank and the probability convention it was measured "
                "under and is not reused across any of them")
    return arrays, record


#: The small numeric documents ``--publish`` copies beside the report.
PUBLISHED_DATA = ("decision.json", "coverage.json", "verification.json", "checkpoints.json",
                  "ches.json")


def publication_plan(context, *, figures, ches_document):
    """What ``--publish`` will write, named before anything is written.

    The report links to these paths, and the manifest records their digests, so the
    list has to exist before the report text does. Returning it separately keeps the
    two from disagreeing: the publisher refuses to write a set of files that is not
    exactly this one.
    """
    settings = context.config["publication"]
    data = [name for name in PUBLISHED_DATA if name != "ches.json" or ches_document]
    figure_names = sorted(entry["name"] for entry in figures.values() if entry.get("written"))
    return {"root": settings["root"],
            "data_directory": settings["data_directory"],
            "figure_directory": settings["figure_directory"],
            "report": _relative(context.repository_root / context.config["published_report"],
                                context.repository_root / settings["root"]),
            "files": sorted([f"{settings['data_directory']}/{name}" for name in data]
                            + [f"{settings['figure_directory']}/{name}"
                               for name in figure_names])}


def publish_deliverable(context, *, render, decision, coverage, checkpoints_document,
                        ches_document, verification, figures):
    """Copy the report, the small numeric summaries and the figures into ``reference``.

    Local, tracked evidence only: files are written, nothing is committed and
    nothing leaves the machine. The links in the published report are relative to
    the published report, so they resolve in the repository rather than pointing at
    an ignored run directory. Figure bytes are copied verbatim -- a re-render would
    produce a different file for the same numbers, and the published figure has to
    be the one the digests describe.

    Everything here is write-once: a published file that already exists must hold
    exactly these bytes, or the publication is refused rather than replaced.
    """
    settings = context.config["publication"]
    root = context.repository_root / settings["root"]
    plan = publication_plan(context, figures=figures, ches_document=ches_document)
    published = {}

    def publish_json(name, document):
        logical = f"{settings['data_directory']}/{name}"
        target = root / logical
        if target.is_file():
            existing = paths.read_json(target)
            require(paths.scientific_projection(existing)
                    == paths.scientific_projection(document),
                    f"{target} is already published with different scientific content; a changed "
                    "result belongs in a new revision directory")
            published[logical] = paths.sha256_file(target)
            return
        published[logical] = paths.write_json(target, document)

    publish_json("decision.json", decision)
    publish_json("coverage.json", coverage)
    publish_json("verification.json", {
        "schema_version": paths.AUDIT_SCHEMA, "record_kind": "published_verification",
        "immutable": verification["immutable"], "shards_checked": verification["shards_checked"],
        "problems": verification["problems"],
        "note": ("the immutability verdict and the problems it found. The per-artifact hashes and "
                 "the expected-output list stay in the run record, because that list grows as "
                 "later stages complete and a published file does not change")})
    publish_json("checkpoints.json", published_checkpoint_table(context, checkpoints_document))
    if ches_document:
        publish_json("ches.json", published_ches_tables(ches_document))

    for entry in sorted(figures.values(), key=lambda item: str(item.get("name"))):
        if not entry.get("written"):
            continue
        logical = f"{settings['figure_directory']}/{entry['name']}"
        target = root / logical
        payload = context.run.path(entry["logical"]).read_bytes()
        if target.is_file():
            require(target.read_bytes() == payload,
                    f"{target} is already published with different bytes; a re-rendered figure is "
                    "a new artifact and belongs in a new revision directory")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        published[logical] = paths.sha256_bytes(payload)

    require(sorted(published) == plan["files"],
            f"The publisher wrote {sorted(published)} but the report was rendered against "
            f"{plan['files']}; the links in a published report have to be the files beside it")

    report_text = render(plan)
    report_path = context.repository_root / context.config["published_report"]
    published[plan["report"]] = require_new_or_identical_text(
        report_path, report_text, what="the published report")

    manifest = {"schema_version": paths.AUDIT_SCHEMA, "record_kind": "publication_manifest",
                "audit_id": context.config["audit_id"],
                "root": settings["root"],
                "files": dict(sorted(published.items())),
                "file_count": len(published),
                "links": ("every path above is relative to the published report at "
                          f"{context.config['published_report']}"),
                "note": ("local tracked evidence. This CLI performs no network or git action; "
                         "committing these files is a separate, human decision.")}
    manifest_path = root / settings["manifest"]
    if manifest_path.is_file():
        existing = paths.read_json(manifest_path)
        require(paths.scientific_projection(existing) == paths.scientific_projection(manifest),
                f"{manifest_path} already publishes a different set of files")
    else:
        paths.write_json(manifest_path, manifest)
    return manifest


def published_checkpoint_table(context, checkpoints_document):
    """The small, linkable numeric table: one row per scored computation."""
    rows = {}
    for identifier, block in sorted((checkpoints_document.get("checkpoints") or {}).items()):
        tail = block["tails"]["counts"]
        rows[identifier] = {
            "forward_kl": block["forward_kl"]["mean"],
            "forward_kl_ci_low": block["forward_kl"]["ci_low"],
            "forward_kl_ci_high": block["forward_kl"]["ci_high"],
            "tenfold_fraction": tail["tenfold"]["fraction"],
            "tenfold_wilson_lower": tail["tenfold"]["lower"],
            "hundredfold_fraction": tail["hundredfold"]["fraction"],
            "hundredfold_wilson_lower": tail["hundredfold"]["lower"],
            "rows": block["tails"]["rows"]}
    return {"schema_version": paths.AUDIT_SCHEMA, "record_kind": "published_checkpoint_table",
            "audit_id": context.config["audit_id"],
            "probability_contract": dict(inventory_lib.CORE_CONTRACT),
            "scored": len(rows), "checkpoints": rows,
            "per_seed_then_seeds": checkpoints_document.get("per_seed_then_seeds"),
            "paired_method_differences": checkpoints_document.get("paired_method_differences"),
            "note": ("the per-row score vectors and the full statistics stay in the run record; "
                     "this is the summary table the report cites")}


def published_ches_tables(ches_document):
    """The CHES, control and increment numbers, kept numeric rather than narrated."""
    return {"schema_version": paths.AUDIT_SCHEMA, "record_kind": "published_ches_tables",
            "extraction": ches_document.get("extraction"),
            "extraction_path": ches_document.get("extraction_path"),
            "positions": ches_document.get("positions"),
            "coverage": ches_document.get("coverage"),
            "consumption": ches_document.get("consumption"),
            "parent": ches_document.get("parent"),
            "endpoints": ches_document.get("endpoints"),
            "associations": ches_document.get("associations"),
            "increments": ches_document.get("increments"),
            "increment_gaps": ches_document.get("increment_gaps"),
            "missing_controls": ches_document.get("missing_controls"),
            "note": ("the numeric CHES, matched-control and increment tables as measured; the "
                     "per-pair vectors stay in the run record")}


def stage_timings(context):
    """The recorded per-stage timing blocks, for the report's cost table."""
    out = {}
    for stage in STAGES:
        path = context.run.path(f"timings/{stage}.json")
        if path.is_file():
            out[stage] = paths.read_json(path)
    return out


def record_timings(context, stage, clock):
    """Record a stage's measured seconds without overwriting the original ones.

    The first completed run's timings are the ones that describe the work that
    produced the artifacts. A rerun that verified those artifacts did different
    work -- mostly hashing -- so its numbers go to a ``.rerun.json`` beside them
    instead of replacing a cost the report attributes to the measurement.
    """
    return record_timing_block(context, stage, clock.document())


def record_timing_block(context, stage, block):
    """The writer behind :func:`record_timings`, for stages that time themselves."""
    document = dict(block, stage=stage, recorded_at=paths.utc_now(),
                    schema_version=paths.AUDIT_SCHEMA)
    target = context.run.path(f"timings/{stage}.json")
    if target.is_file():
        document["note"] = ("a rerun of a stage whose original timings are already recorded; "
                            "those are preserved and this file records the rerun's own cost")
        paths.write_json(context.run.path(f"timings/{stage}.rerun.json"), document)
        return document
    paths.write_json(target, document)
    return document
