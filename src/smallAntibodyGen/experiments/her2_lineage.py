"""Two digest blocks, never one: what produced the parents vs what this revision is.

``read_base_selection`` in the original runner compares
``code_digests(ROOT, HER2_CODE_FILES)`` byte-for-byte against what
``base_selection.json`` recorded. That check is the reason this revision adds
files instead of editing them: the moment one of the ten original scientific
sources changes, the initial-SFT freeze stops being readable and the parents lose
their provenance.

So identity here has two honestly named halves:

``inherited``
    the original config hash, the original code digests, the base-selection hash
    and the parent checkpoints. These describe the code and config that
    **produced** the parents. This revision did not produce them and does not
    claim to.
``revision``
    this revision's own config and code digests, including the original sources
    as read-only inputs.

The freeze this revision writes is deliberately **not** a ``her2-selection/1``
document. It carries its own schema and a stage marker outside
:data:`~.her2_data.SELECTION_STAGES`, so :func:`~.her2_data.read_selection_freeze`
refuses it and no artifact produced here can unlock a reserved test label.

One more refusal lives here: an interrupted guarded run directory is **never**
overwritten. The original ledger offers ``--discard-incomplete``; this one does
not, because a guarded trajectory's partial evidence -- the journals, the monitor
scores, the stop record -- is the result when the gate fires, and a restart that
reclaims the directory destroys it.
"""
from __future__ import annotations

from pathlib import Path

from . import her2_data as data
from .her2_runtime import (HER2_CODE_FILES, code_digests, digest_document, load_json,
                           relative_key, require, save_json, sha256)

GUARDED_IDENTITY_SCHEMA = "her2-guarded-continuation/1"
#: Deliberately not ``her2-selection/1``. A different schema means
#: ``read_selection_freeze`` refuses the document outright rather than reasoning
#: about its stage, so this revision cannot open a reserved label at all.
GUARDED_SELECTION_SCHEMA = "her2-guarded-selection/1"
#: Also outside ``SELECTION_STAGES``, so even a schema-blind reader finds no
#: stage it recognizes.
GUARDED_STAGE = "guarded_continuation_stage"

#: This revision's own scientific sources. Kept separate from
#: ``HER2_CODE_FILES``: adding a file to that tuple would change the identity the
#: original freeze recorded and break the parents' provenance.
GUARDED_CODE_FILES = (
    "src/smallAntibodyGen/experiments/her2_objectives.py",
    "src/smallAntibodyGen/experiments/her2_guard.py",
    "src/smallAntibodyGen/experiments/her2_guarded_trajectory.py",
    "src/smallAntibodyGen/experiments/her2_guarded_eval.py",
    "src/smallAntibodyGen/experiments/her2_lineage.py",
    "src/smallAntibodyGen/experiments/her2_history.py",
    "src/smallAntibodyGen/experiments/her2_absci_audit.py",
    "scripts/posttrain_her2_guarded.py",
    "scripts/audit_absci_her2.py",
)

INHERITED_NOTE = ("these hashes describe the code and config that PRODUCED the parents; this "
                  "revision did not produce them and does not claim to")


def guarded_identity(root, *, config_path, original_config_path, base_selection_path,
                     source_digests, device, batch_sequences, revision_files=GUARDED_CODE_FILES):
    """The two-block identity every guarded artifact carries."""
    root = Path(root)
    return {
        "schema_version": GUARDED_IDENTITY_SCHEMA,
        "revision": {
            "config_sha256": sha256(config_path),
            "config_digest": digest_document(load_json(config_path)),
            "code_digests": code_digests(root, tuple(revision_files) + HER2_CODE_FILES),
            "note": ("the current revision's own sources, plus the original modules it imports "
                     "read-only")},
        "inherited": {
            "base_selection_path": relative_key(base_selection_path, root),
            "base_selection_sha256": sha256(base_selection_path),
            "original_config_sha256": sha256(original_config_path),
            "original_code_digests": code_digests(root, HER2_CODE_FILES),
            "note": INHERITED_NOTE},
        "source_digests": dict(source_digests),
        "device": device,
        "batch_sequences": int(batch_sequences),
    }


def verify_inherited_code(root, identity):
    """The tripwire: the original scientific sources must still hash as recorded.

    If this fails, somebody edited a file the initial-SFT freeze names. The right
    response is to restore it, not to re-record the new hash: the parents on disk
    were produced by the old bytes either way.
    """
    current = code_digests(Path(root), HER2_CODE_FILES)
    recorded = identity["inherited"]["original_code_digests"]
    differing = sorted(key for key in set(current) | set(recorded)
                       if current.get(key) != recorded.get(key))
    require(not differing,
            f"Original HER2 scientific sources changed since this identity was built: {differing}. "
            "The initial-SFT freeze binds those hashes, so the parents would no longer be "
            "readable under their own provenance.")
    return current


def read_original_freeze(path, *, root, identity, expected_selected, seeds):
    """Verify the initial-SFT freeze against the INHERITED block and return the parents.

    Everything checkable is checked before this revision writes anything: schema,
    the base stage marker, the original config hash, the original code hashes, the
    pinned source hashes, the exact expected artifact names and every named
    checkpoint's content hash. The unlock the reader mints is discarded -- it is a
    base-stage token and opens nothing.
    """
    inherited = identity["inherited"]
    document, _ = data.read_selection_freeze(
        path, root=root, expected_config_sha256=inherited["original_config_sha256"],
        expected_stage=data.SELECTION_STAGE_BASE, expected_selected=sorted(expected_selected),
        expected_code_digests=inherited["original_code_digests"])
    stored_sources = document.get("source_digests") or {}
    differing = sorted(key for key in set(stored_sources) | set(identity["source_digests"])
                       if stored_sources.get(key) != identity["source_digests"].get(key))
    require(not differing,
            f"{path} was produced against different pinned sources; the continuation would be "
            f"reading data the parents were not fitted on. Differing entries: {differing[:5]}")
    parents = {}
    for seed in seeds:
        key = f"policy_sft_seed{seed}"
        require(key in document["selected"],
                f"{path} does not name an initial SFT checkpoint for seed {seed}")
        entry = document["selected"][key]
        target = Path(root) / entry["checkpoint"]
        require(sha256(target) == entry["sha256"],
                f"{key}: the parent checkpoint on disk does not match the freeze")
        parents[int(seed)] = dict(entry, name=key)
    return document, parents


def bind_parents(identity, parents):
    """Attach the verified parents to the inherited block, by path and hash."""
    bound = json_copy(identity)
    bound["inherited"]["parents"] = {
        str(seed): {"name": entry["name"], "checkpoint": entry["checkpoint"],
                    "sha256": entry["sha256"]}
        for seed, entry in sorted(parents.items())}
    return bound


def json_copy(document):
    import copy
    return copy.deepcopy(document)


def assert_cannot_unlock(document):
    """A guarded artifact must be unreadable as a selection freeze. Checked, not assumed."""
    require(document.get("schema_version") == GUARDED_SELECTION_SCHEMA,
            "A guarded freeze carries the guarded schema")
    require(document.get("schema_version") != data.SELECTION_SCHEMA,
            "A guarded freeze must not claim the selection schema")
    require(document.get("stage") not in data.SELECTION_STAGES,
            f"A guarded freeze must not carry a selection stage marker: {document.get('stage')!r}")
    return True


def artifact_record(path, root):
    """One artifact's identity: repository-relative path, content hash and size."""
    path = Path(path)
    require(path.is_file(), f"Missing artifact {path}")
    return {"path": relative_key(path, root), "sha256": sha256(path),
            "bytes": int(path.stat().st_size)}


def verify_artifact(record, root):
    """Re-hash one named artifact. A changed file is evidence about other bytes."""
    path = Path(root) / record["path"]
    require(path.is_file(), f"Frozen artifact is missing: {path}")
    digest = sha256(path)
    require(digest == record["sha256"],
            f"Frozen artifact {record['path']} changed on disk: {digest} != {record['sha256']}")
    return True


class GuardedRunLedger:
    """One trajectory directory, its identity, and the refusal to reclaim it.

    Three outcomes and no fourth: a matching completed run is skipped, a fresh
    directory is claimed, and **anything else raises**. In particular an
    interrupted run is refused rather than discarded, because its journals,
    monitor scores and stop record are the evidence about what the gate saw.
    """

    def __init__(self, directory, identity, *, metadata=None):
        require(isinstance(identity, dict) and identity, "A run needs a non-empty identity")
        self.directory = Path(directory)
        self.identity = identity
        self.metadata = dict(metadata or {})
        self.path = self.directory / "run.json"

    def existing(self):
        return load_json(self.path) if self.path.is_file() else None

    def claim(self):
        record = self.existing()
        if record is not None:
            require(record["identity"] == self.identity,
                    f"{self.path} exists with a different identity; choose a fresh run id rather "
                    "than reusing one across configurations. The revision code hashes are part of "
                    "the identity, so edited code is a different run.")
            if record["status"] == "completed":
                return "completed"
            raise ValueError(
                f"{self.path} is marked {record['status']!r}. A guarded run directory is never "
                "overwritten: its journals, monitor score vectors and stop record are the "
                "evidence about what the gate observed. Name a new run id and leave this one in "
                "place.")
        self.directory.mkdir(parents=True, exist_ok=True)
        save_json(self.path, {"identity": self.identity, "status": "running",
                              "metadata": self.metadata})
        return "start"

    def finish(self, summary, *, status="completed"):
        record = self.existing() or {}
        metadata = dict(record.get("metadata") or self.metadata)
        save_json(self.path, {"identity": self.identity, "status": status,
                              "metadata": metadata, "summary": summary})
        return status
