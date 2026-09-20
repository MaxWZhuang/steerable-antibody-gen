"""Read-only inventory of the HER2 post-training states, their provenance and banks.

This module enumerates the 159 model states the support audit scores -- three
selected SFT parents, 72 reached guarded endpoints, 60 diagnostic snapshots from
the 30 early-stopped guarded trajectories, and 24 original-v1 endpoints -- and it
proves, rather than assumes, that each one is the artifact its manifest names.

The decisions that are easy to get wrong, and are therefore fixed here:

* **Two schemas, two adapters, no downgrade.** Normal policy checkpoints are
  ``her2-core-policy/1`` and carry ``state_sha256``. Guarded rolling and failure
  snapshots are ``her2-guarded-trajectory/1``, carry ``metadata={"identity": ...}``
  and carry **no** recorded state digest at all. Passing one through the other's
  reader raises with a message that says which reader to use; it never suppresses
  the schema error, and it never rewrites a payload into a convenient shape.
* **The file is hashed before it is unpickled.** ``torch.load(weights_only=True)``
  is a much smaller attack surface than a full unpickle, but it is still executed
  against bytes. The manifest digest is checked first, so the loader only ever
  opens a file the campaign's own frozen records vouch for.
* **Manifest authority is the frozen historical JSON, not the publication CSV.**
  Identities and raw hashes come from ``base_selection.json``, the guarded
  ``validation/stage{N}_endpoints.json`` documents, the trajectory ``stop.json``
  records and the v1 ``summary.json`` files. The published CSV tables are
  derived views and are never the source of a hash.
* **Producer identity and audit dependency are separate facts.** The 38 archived
  sources under the launch snapshot are verified against the launch manifest, and
  the current working tree is compared to them and reported -- as drift, not as a
  reason to disable a check and not as this audit's own source identity.
* **Alias states are named, deduplicated for computation and never recounted.**
  The three continued-SFT controls are reused by stages 2 and 3; the same bytes
  therefore appear under several semantic names. They are scored once.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .her2_data import CANONICAL, CORE_LENGTH, encode_cores
from .her2_policy import POLICY_SCHEMA, state_digest
from .her2_preferences import PROBABILITY_CONVENTION, core_digest
from .her2_guarded_trajectory import TRAJECTORY_SCHEMA
from .her2_runtime import require
from . import her2_support_paths as paths

#: Roles this audit distinguishes. A diagnostic snapshot is never a nominal endpoint.
ROLE_PARENT = "selected_parent"
ROLE_ENDPOINT = "reached_endpoint"
ROLE_LAST_PASSING = "last_passing"
ROLE_FAILED_STATE = "failed_state"
ROLE_V1_ENDPOINT = "v1_endpoint"
ROLES = (ROLE_PARENT, ROLE_ENDPOINT, ROLE_LAST_PASSING, ROLE_FAILED_STATE, ROLE_V1_ENDPOINT)

#: Statuses an inventory row may carry. Only ``verified`` may be scored.
STATUS_VERIFIED = "verified"
STATUS_MISSING = "missing"
STATUS_FAILED = "verification_failed"

#: The identity fields each role's payload MUST bind, and the manifest must declare.
#:
#: These are not aspirational. A strict load of all 159 real campaign states records
#: every one of these fields on every payload of that role, so an absence is a
#: provenance failure in *this* run rather than a gap in the historical writers. The
#: earlier revision moved every field into the optional tier, where a payload missing
#: its seed and its epoch still passed as long as one unrelated field agreed -- which
#: is the check that was supposed to stop the wrong checkpoint being scored.
#:
#: ``parent_sha256`` is required of the guarded endpoints as well as the diagnostic
#: snapshots: the reached endpoints carry ``identity.inherited.parents[seed].sha256``,
#: and an endpoint whose recorded parent is not the parent this audit scores it
#: against is measuring a drop from the wrong distribution.
REQUIRED_BINDINGS = {
    ROLE_PARENT: ("arm_id", "epoch", "seed"),
    ROLE_ENDPOINT: ("arm_id", "budget_seconds", "parent_sha256", "seed", "stage", "update"),
    ROLE_LAST_PASSING: ("arm_id", "parent_sha256", "seed", "stage", "update"),
    ROLE_FAILED_STATE: ("arm_id", "parent_sha256", "seed", "stage", "update"),
    ROLE_V1_ENDPOINT: ("budget_seconds", "method", "parent_sha256", "seed", "update"),
}

#: The one probability contract this audit scores under, recorded on every row so a
#: later join cannot silently mix a mean-per-residue column into a sum-of-ten one.
CORE_CONTRACT = {"core_length": CORE_LENGTH, "canonical_residues": CANONICAL,
                 "renormalization": "20way_canonical",
                 "probability_convention": PROBABILITY_CONVENTION}


# ---------------------------------------------------------------------------
# checkpoint adapters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CheckpointPayload:
    """One loaded checkpoint: its bytes, its metadata and the state it really holds."""

    logical: str
    file_sha256: str
    schema_version: str
    kind: str
    payload_keys: tuple
    metadata: dict
    tensors: dict
    state_digest_recorded: object
    state_digest_recorded_reason: object
    state_digest_audit_computed: str

    def document(self):
        """Facts about the payload. Deliberately *not* keyed ``logical_path``.

        The inventory row already carries the campaign-relative ``logical_path``
        that resolves under its root; a ``record.update(payload.document())`` that
        also wrote that key would replace it with the root-prefixed name the reader
        was called with, and the next stage would resolve ``root/root/...``.
        """
        return {"verified_logical_name": self.logical, "file_sha256": self.file_sha256,
                "payload_schema": self.schema_version, "payload_kind": self.kind,
                "payload_keys": list(self.payload_keys),
                "tensor_count": len(self.tensors),
                "state_digest_recorded": self.state_digest_recorded,
                "state_digest_recorded_reason": self.state_digest_recorded_reason,
                "state_digest_audit_computed": self.state_digest_audit_computed}


class CheckpointReader:
    """Strict, CPU-first reader for both checkpoint schemas.

    One container model is built and reused. That is safe precisely because every
    load is ``strict=True``: the key set must match exactly, so every parameter and
    persistent buffer is replaced and nothing from the previous checkpoint can
    survive into the recomputed digest. Building 159 fresh 22M-parameter containers
    instead would cost minutes of random initialization to prove the same thing.
    """

    def __init__(self, model_factory):
        self._factory = model_factory
        self._container = None

    @property
    def container(self):
        if self._container is None:
            self._container = self._factory()
            self._container.eval()
        return self._container

    # -- the shared strict path ------------------------------------------
    def _load(self, path, *, logical, expected_file_sha256, expect_schema, reader_name):
        import torch
        path = Path(path)
        require(path.is_file(),
                f"{logical} is missing at the resolved location; a missing state is reported in "
                "the coverage table, never substituted by a neighbouring checkpoint")
        require(isinstance(expected_file_sha256, str) and len(expected_file_sha256) == 64,
                f"{logical}: a trusted sha256 from the frozen manifests is required before this "
                "file is opened")
        # Hash first. The loader only ever unpickles bytes the campaign's own frozen
        # records already vouch for.
        observed = paths.sha256_file(path)
        require(observed == expected_file_sha256,
                f"{logical}: file sha256 {observed} does not match the manifest digest "
                f"{expected_file_sha256}; the artifact on disk is not the one that was frozen")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        require(isinstance(payload, dict), f"{logical}: payload is not a mapping")
        schema = payload.get("schema_version")
        require(schema == expect_schema,
                f"{logical}: payload schema is {schema!r}; {reader_name} accepts {expect_schema!r} "
                "only. Use the adapter for that schema -- this reader does not downgrade a "
                "payload or suppress its schema error.")
        require("state" in payload, f"{logical}: payload carries no 'state' block")
        state = payload["state"]
        require(isinstance(state, dict) and state, f"{logical}: 'state' is not a tensor mapping")
        tensors = self._verify_tensors(state, logical=logical)
        self.container.load_state_dict({k: v for k, v in state.items()}, strict=True)
        computed = state_digest(self.container)
        metadata = {key: value for key, value in payload.items() if key != "state"}
        return payload, metadata, tensors, computed, observed

    def _verify_tensors(self, state, *, logical):
        """Expected keys, exact dtypes, expected shapes, finite floats.

        The dtype is checked as strictly as the shape. ``load_state_dict`` casts a
        float16 tensor into a float32 parameter without complaint, so a checkpoint
        saved at reduced precision would load, score and produce numbers that are
        not the ones the campaign recorded -- with no error anywhere.
        """
        import torch
        expected = self.container.state_dict()
        missing = sorted(set(expected) - set(state))
        unexpected = sorted(set(state) - set(expected))
        require(not missing, f"{logical}: state is missing tensors {missing[:5]}")
        require(not unexpected, f"{logical}: state carries unexpected tensors {unexpected[:5]}")
        summary = {}
        for name in sorted(state):
            tensor = state[name]
            require(isinstance(tensor, torch.Tensor), f"{logical}: {name} is not a tensor")
            require(tensor.dtype == expected[name].dtype,
                    f"{logical}: {name} is {tensor.dtype}, the pinned architecture expects "
                    f"{expected[name].dtype}; load_state_dict would cast it silently")
            require(tuple(tensor.shape) == tuple(expected[name].shape),
                    f"{logical}: {name} has shape {tuple(tensor.shape)}, the pinned architecture "
                    f"expects {tuple(expected[name].shape)}")
            if tensor.is_floating_point():
                require(bool(torch.isfinite(tensor).all()),
                        f"{logical}: {name} carries a nonfinite value; a checkpoint that cannot "
                        "produce finite logits is a failure to report, not a row to drop")
            summary[name] = {"dtype": str(tensor.dtype), "shape": list(tensor.shape)}
        return summary

    # -- the two public adapters -----------------------------------------
    def read_policy(self, path, *, logical, expected_file_sha256):
        """``her2-core-policy/1``: the recorded state digest must reproduce."""
        payload, metadata, tensors, computed, observed = self._load(
            path, logical=logical, expected_file_sha256=expected_file_sha256,
            expect_schema=POLICY_SCHEMA, reader_name="the policy-checkpoint reader")
        recorded = payload.get("state_sha256")
        require(isinstance(recorded, str) and recorded,
                f"{logical}: a policy checkpoint must record state_sha256")
        require(recorded == computed,
                f"{logical}: recorded state digest {recorded} is not reproduced by a strict "
                f"reload ({computed})")
        return CheckpointPayload(
            logical=logical, file_sha256=observed, schema_version=POLICY_SCHEMA,
            kind=str(metadata.get("kind") or "policy"), payload_keys=tuple(sorted(payload)),
            metadata=_json_safe(metadata), tensors=tensors, state_digest_recorded=recorded,
            state_digest_recorded_reason=None, state_digest_audit_computed=computed)

    def read_trajectory(self, path, *, logical, expected_file_sha256):
        """``her2-guarded-trajectory/1``: no recorded digest exists, and none is invented."""
        payload, metadata, tensors, computed, observed = self._load(
            path, logical=logical, expected_file_sha256=expected_file_sha256,
            expect_schema=TRAJECTORY_SCHEMA, reader_name="the diagnostic-snapshot reader")
        kind = payload.get("kind")
        require(kind in (ROLE_LAST_PASSING, ROLE_FAILED_STATE),
                f"{logical}: diagnostic snapshot kind is {kind!r}, expected "
                f"{ROLE_LAST_PASSING!r} or {ROLE_FAILED_STATE!r}")
        require("state_sha256" not in payload,
                f"{logical}: this schema records no state digest, but the payload carries one; "
                "the snapshot writer changed and the audit's expectations are stale")
        metadata_block = payload.get("metadata")
        require(isinstance(metadata_block, dict) and isinstance(metadata_block.get("identity"),
                                                                dict),
                f"{logical}: diagnostic snapshots must carry metadata={{'identity': ...}}; an "
                "absent identity is a recorded provenance failure, not something to coerce")
        return CheckpointPayload(
            logical=logical, file_sha256=observed, schema_version=TRAJECTORY_SCHEMA,
            kind=str(kind), payload_keys=tuple(sorted(payload)), metadata=_json_safe(metadata),
            tensors=tensors, state_digest_recorded=None,
            state_digest_recorded_reason=("the her2-guarded-trajectory/1 rolling writer records no "
                                          "state digest; the audit's own recomputed digest is the "
                                          "only one that exists"),
            state_digest_audit_computed=computed)


def _json_safe(value):
    """Drop tensors and coerce numpy scalars so metadata survives canonical JSON."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()
                if not _is_tensor(item)}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value if not _is_tensor(item)]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _is_tensor(value):
    return type(value).__module__.startswith("torch") and hasattr(value, "detach")


# ---------------------------------------------------------------------------
# payload contracts
# ---------------------------------------------------------------------------

#: Where each schema keeps its identity block. A diagnostic snapshot nests it one
#: level down under ``metadata``; the policy checkpoint carries it at the top. The
#: reader adapts by declared schema instead of trying both and hoping.
IDENTITY_LOCATION = {POLICY_SCHEMA: ("identity",),
                     TRAJECTORY_SCHEMA: ("metadata", "identity")}


def payload_identity(payload):
    """The identity block of a payload, located by its declared schema."""
    location = IDENTITY_LOCATION.get(payload.schema_version)
    require(location is not None,
            f"{payload.logical}: no identity location is declared for schema "
            f"{payload.schema_version!r}")
    node = dict(payload.metadata)
    for key in location:
        node = node.get(key) if isinstance(node, dict) else None
        if node is None:
            break
    require(isinstance(node, dict) and node,
            f"{payload.logical}: schema {payload.schema_version} keeps its identity at "
            f"{'.'.join(location)} and the payload has nothing there; an absent identity is a "
            "recorded provenance failure, not something to look for somewhere else")
    return node


def guarded_parent_sha256(identity, seed):
    """The parent a guarded trajectory actually inherited, for ``seed``.

    ``identity.inherited.base_selection_sha256`` is the digest of the *selection
    manifest* -- one value shared by all three seeds. The parent weights of a seed
    are ``identity.inherited.parents[str(seed)].sha256``, and confusing the two
    would bind every trajectory to the same non-checkpoint hash and still "match".
    """
    inherited = (identity or {}).get("inherited") or {}
    parents = inherited.get("parents") or {}
    entry = parents.get(str(seed)) if isinstance(parents, dict) else None
    if isinstance(entry, dict) and entry.get("sha256"):
        return str(entry["sha256"])
    return None


def validate_payload_contract(payload, expected, *, role, logical, compare_if_present=None):
    """Check the declared identity/seed/arm/parent/budget/update/core bindings.

    Two tiers, because "the writer never recorded this field" and "the writer
    recorded a different value" are different findings:

    * ``expected`` is the binding contract. An absent field is a provenance gap
      and a differing one is the wrong checkpoint; both stop the record.
    * ``compare_if_present`` is everything else the manifest knows. A *mismatch*
      is still a hard failure -- that is the whole point of comparing it -- but an
      absence is recorded, named and carried into the evidence rather than
      inventing a failure over a field the historical writer never wrote.

    :data:`REQUIRED_BINDINGS` decides which tier a field belongs to, and it is
    enforced here rather than left to the caller: a caller that passed ``{}`` as the
    contract and moved everything into the optional tier would otherwise turn every
    hard binding into an advisory one without anything noticing.
    """
    require(role in ROLES, f"Unknown inventory role {role!r}")
    demoted = sorted(field for field in REQUIRED_BINDINGS[role]
                     if (expected or {}).get(field) is None)
    require(not demoted,
            f"{logical}: role {role!r} must bind {demoted} in its required contract. Every real "
            f"payload of this role records them, so a manifest that cannot declare one -- or a "
            f"caller that moved it into the compare-if-present tier -- is a provenance gap, not a "
            f"softer check.")
    metadata = dict(payload.metadata)
    identity = payload_identity(payload)
    nested = metadata.get("metadata") if isinstance(metadata.get("metadata"), dict) else {}
    progress = _first(metadata.get("progress"), nested.get("progress")) or {}
    observed = {
        "seed": _first(metadata.get("seed"), nested.get("seed"), identity.get("seed")),
        "arm_id": _arm_id_of(metadata, identity) or _arm_id_of(nested, identity),
        "stage": _first(metadata.get("stage"), nested.get("stage"), identity.get("stage")),
        "budget_seconds": _as_float(_first(metadata.get("budget_seconds"),
                                           nested.get("budget_seconds"),
                                           identity.get("budget_seconds"))),
        "update": _first(metadata.get("update"), nested.get("update"), progress.get("updates"),
                         progress.get("update")),
        "epoch": _first(metadata.get("epoch"), identity.get("epoch")),
        "method": _first(metadata.get("method"), nested.get("method"), identity.get("method"),
                         (identity.get("arm") or {}).get("objective")
                         if isinstance(identity.get("arm"), dict) else None),
        "kind": _first(metadata.get("kind"), nested.get("kind")),
        "parent_sha256": _first(
            identity.get("parent_sha256"),
            (identity.get("parent") or {}).get("sha256")
            if isinstance(identity.get("parent"), dict) else None,
            guarded_parent_sha256(identity, _first(metadata.get("seed"), nested.get("seed"),
                                                   identity.get("seed")))),
        "exposures": _first(metadata.get("exposures"), nested.get("exposures"),
                            progress.get("exposures")),
    }
    checks, unchecked = {}, []
    for field, wanted in sorted((expected or {}).items()):
        if wanted is None:
            unchecked.append(field)
            continue
        require(field in observed, f"{logical}: contract field {field!r} is not one this audit "
                                   f"knows how to read from a payload")
        actual = observed[field]
        require(actual is not None,
                f"{logical}: the payload carries no {field!r}, but the inventory declares "
                f"{wanted!r} for it; an absent binding is a provenance gap, not a default")
        matched = _equalish(actual, wanted)
        checks[field] = {"expected": wanted, "observed": actual, "matched": matched}
        require(matched,
                f"{logical}: payload {field} is {actual!r} but the manifest binds this artifact to "
                f"{wanted!r}; this is the wrong checkpoint for that record")
    absent = {}
    for field, wanted in sorted((compare_if_present or {}).items()):
        if wanted is None:
            unchecked.append(field)
            continue
        require(field in observed,
                f"{logical}: contract field {field!r} is not one this audit knows how to read")
        actual = observed[field]
        if actual is None:
            absent[field] = {"expected": wanted,
                             "reason": (f"the {payload.schema_version} payload records no "
                                        f"{field!r}; the manifest value is not confirmed by the "
                                        "checkpoint itself")}
            continue
        matched = _equalish(actual, wanted)
        checks[field] = {"expected": wanted, "observed": actual, "matched": matched}
        require(matched,
                f"{logical}: payload {field} is {actual!r} but the manifest binds this artifact to "
                f"{wanted!r}; this is the wrong checkpoint for that record")
    declared = {key: value for key, value in
                dict(dict(expected or {}), **dict(compare_if_present or {})).items()
                if value is not None}
    require(not declared or checks,
            f"{logical}: the payload confirmed none of the declared bindings {sorted(declared)}. "
            "An artifact whose identity cannot be confirmed from its own bytes at all is a "
            "provenance failure, not a verified state.")
    # The payload's own kind has to agree with the role the inventory assigned it.
    # A failure snapshot filed as a last-passing state would otherwise be scored as
    # a state that passed its gate.
    if role in (ROLE_LAST_PASSING, ROLE_FAILED_STATE):
        require(payload.kind == role,
                f"{logical}: the payload calls itself {payload.kind!r} but the stop record files "
                f"it as {role!r}; a diagnostic snapshot is never relabelled")
    else:
        require(payload.schema_version == POLICY_SCHEMA,
                f"{logical}: role {role!r} requires a {POLICY_SCHEMA} payload, not "
                f"{payload.schema_version!r}")
    return {"role": role, "checked": checks, "unchecked_fields": sorted(unchecked),
            "required_bindings": list(REQUIRED_BINDINGS[role]),
            "not_recorded_by_payload": absent,
            "payload_kind": payload.kind, "payload_schema": payload.schema_version,
            "observed": observed, "core_contract": dict(CORE_CONTRACT)}


def _first(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _arm_id_of(metadata, identity):
    for holder in (metadata or {}, identity or {}):
        if holder.get("arm_id"):
            return str(holder["arm_id"])
        arm = holder.get("arm")
        if isinstance(arm, dict) and arm.get("arm_id"):
            return str(arm["arm_id"])
        if isinstance(arm, str) and arm:
            return arm
    return None


def _as_float(value):
    return None if value is None else float(value)


def _equalish(actual, wanted):
    if isinstance(wanted, float) or isinstance(actual, float):
        try:
            return float(actual) == float(wanted)
        except (TypeError, ValueError):
            return False
    return actual == wanted


# ---------------------------------------------------------------------------
# historical manifests and source provenance
# ---------------------------------------------------------------------------

def guarded_root_probes(published_manifest, *, anchor):
    """``{logical suffix: sha256}`` for the guarded root, from the published manifest.

    The frozen stage markers are the probe because the published manifest -- which
    is tracked, reviewable and already hash-pinned by ``.gitattributes`` -- records
    their digests. A directory that reproduces all three is the completed campaign.
    """
    frozen = published_manifest.get("frozen_stages") or {}
    require(frozen, "The published guarded manifest records no frozen stages to probe with")
    probes = {}
    for stage, entry in sorted(frozen.items()):
        probes[paths.campaign_suffix(entry["path"], anchor=anchor)] = entry["sha256"]
    return probes


def parent_digests_from_stages(stage_documents):
    """``{seed: parent sha256}`` as the guarded campaign recorded them.

    Taken from ``parent_draw_references``, where every stage records the digest of
    the parent each bank was drawn from. That makes the *verified* guarded root the
    authority for what the v1 parents are, which is what lets the original root be
    accepted by hash instead of by existing at the expected place.
    """
    digests = {}
    for stage in sorted(stage_documents):
        references = stage_documents[stage].get("parent_draw_references") or {}
        for seed, entry in sorted(references.items()):
            digest = entry.get("parent_sha256")
            if not digest:
                continue
            previous = digests.setdefault(int(seed), digest)
            require(previous == digest,
                    f"The guarded stages record different parents for seed {seed}: {previous} "
                    f"and {digest}")
        identity = stage_documents[stage].get("identity") or {}
        parents = ((identity.get("inherited") or {}).get("parents") or {})
        for seed, entry in sorted(parents.items()):
            digest = entry.get("sha256") if isinstance(entry, dict) else None
            if not digest:
                continue
            previous = digests.setdefault(int(seed), digest)
            require(previous == digest,
                    f"Stage {stage} inherits a different parent for seed {seed}: {previous} "
                    f"and {digest}")
    require(digests, "No stage document records a parent digest to probe the original root with")
    return digests


def manifest_probes(manifest_files, *, anchor, prefix):
    """``{logical suffix: sha256}`` for entries of a manifest under one logical prefix."""
    probes = {}
    for logical, entry in sorted((manifest_files or {}).items()):
        if logical.startswith(prefix + "/"):
            probes[paths.campaign_relative(logical, anchor=anchor)] = entry["sha256"]
    require(probes, f"No manifest entry lives under {prefix!r} to probe that root with")
    return probes


def verify_source_snapshot(launch_manifest, snapshot_root, *, repository_root):
    """Re-hash the archived producer sources and compare the current tree to them.

    Three separate facts come out of this, and they are kept separate: the archive
    still matches what the launch recorded (provenance intact), the current working
    tree agrees or does not (drift), and neither of those is this audit's own source
    identity.
    """
    declared = dict(launch_manifest.get("source_files") or {})
    require(declared, "The launch manifest declares no source files")
    archived, drift, missing = {}, [], []
    for logical, entry in sorted(declared.items()):
        target = paths.resolve_under(snapshot_root, logical)
        if not target.is_file():
            missing.append(logical)
            archived[logical] = {"sha256": None, "matched": False, "reason": "absent from archive"}
            continue
        digest = paths.sha256_file(target)
        archived[logical] = {"sha256": digest, "matched": digest == entry["sha256"],
                             "declared_sha256": entry["sha256"]}
        if not archived[logical]["matched"]:
            archived[logical]["reason"] = "archived bytes differ from the launch manifest"
    current = {}
    for logical, entry in sorted(declared.items()):
        target = Path(repository_root) / logical
        digest = paths.sha256_file(target) if target.is_file() else None
        current[logical] = {"sha256": digest, "matches_producer": digest == entry["sha256"]}
        if digest != entry["sha256"]:
            drift.append(logical)
    mismatched = sorted(key for key, entry in archived.items() if not entry["matched"])
    require(not mismatched,
            f"The archived producer sources {mismatched[:5]} do not match the launch manifest. "
            "The provenance of the historical artifacts is unproven, which is a fail-closed "
            "result for this audit, not a note in a report.")
    return {"declared_sources": len(declared),
            "archived_verified": sum(1 for e in archived.values() if e["matched"]),
            "archive_missing": missing,
            "archive_mismatched": mismatched,
            "archived": archived, "current_working_tree": current,
            "current_drift": sorted(drift),
            "note": ("the archive is the producer identity of the historical artifacts. Working-"
                     "tree drift is reported, never repaired, and it is not this audit's own "
                     "source identity, which is recorded separately.")}


def audit_source_identity(repository_root, relative_paths):
    """This audit's own sources, hashed as committed bytes. Never mixed with producers."""
    digests = {}
    for logical in sorted(relative_paths):
        target = Path(repository_root) / logical
        require(target.is_file(),
                f"{logical} is part of the audit's own source identity but is missing")
        digests[logical] = paths.sha256_file(target)
    return digests


# ---------------------------------------------------------------------------
# population enumeration
# ---------------------------------------------------------------------------

def _epoch_keyed(value):
    """Is this table entry a ``{epoch: row}`` mapping rather than a single row?

    Tested by the *keys*, not by "some value is a dict": an ordinary row carries
    nested blocks of its own (an exposure count, a metric block), and treating one
    of those as an epoch row is how a parent ends up bound to a sub-object.
    """
    return (isinstance(value, dict) and bool(value)
            and all(str(key).lstrip("-").isdigit() and isinstance(inner, dict)
                    for key, inner in value.items()))


def checkpoint_table_rows(table):
    """``[(key, epoch, row)]`` over a ``checkpoint_table`` in any of its shapes."""
    rows = []
    if isinstance(table, dict):
        for key, value in sorted(table.items()):
            if _epoch_keyed(value):
                rows.extend((str(key), inner_key, inner) for inner_key, inner in sorted(
                    value.items()))
            elif isinstance(value, dict):
                rows.append((str(key), None, value))
            elif isinstance(value, list):
                rows.extend((str(key), None, item) for item in value if isinstance(item, dict))
    elif isinstance(table, list):
        rows.extend((None, None, item) for item in table if isinstance(item, dict))
    return rows


def _row_identifies(row, *, checkpoint, checkpoint_sha256):
    """Does this table row describe exactly the selected checkpoint's bytes?"""
    digest = _first(row.get("checkpoint_sha256"), row.get("sha256"))
    if checkpoint_sha256 and digest:
        return (str(digest) == str(checkpoint_sha256)), "checkpoint_sha256"
    path = row.get("checkpoint")
    if checkpoint and path:
        return (str(path).replace("\\", "/") == str(checkpoint).replace("\\", "/")), "checkpoint"
    return False, None


def parent_progress(base_selection, name, epoch, *, checkpoint=None, checkpoint_sha256=None):
    """``(updates, exposures, reason, binding)`` for a selected parent, from its own table.

    The selection record keeps a ``checkpoint_table`` with one row per saved epoch,
    so the parent's step and exposure counts are recorded facts rather than
    unknowable ones. Only the row for the *selected* epoch is used: a neighbouring
    epoch's counts describe different weights.

    The table is keyed by **run** name -- ``sft_seed20260918`` -- while the selected
    parent is named ``policy_sft_seed20260918``, so looking the selected name up in
    the table finds nothing and all three parents publish null counts against rows
    that are sitting right there. The row is therefore located by the identity the
    two documents genuinely share: the selected checkpoint's own digest, or its
    path where a row records no digest. Stripping a name prefix to make the keys
    line up would be a guess about naming; a hash is not.
    """
    table = base_selection.get("checkpoint_table")
    candidates, seen_epochs = [], set()
    for key, epoch_key, row in checkpoint_table_rows(table):
        row_epoch = _first(row.get("epoch"), epoch_key)
        if row_epoch is None:
            continue
        seen_epochs.add(str(row_epoch))
        if int(row_epoch) != int(epoch):
            continue
        matched, matched_by = _row_identifies(row, checkpoint=checkpoint,
                                              checkpoint_sha256=checkpoint_sha256)
        if matched:
            candidates.append((key, row, matched_by))
    if not candidates:
        return None, None, (
            f"no checkpoint_table row is identified by the selected {name} epoch {epoch} "
            f"checkpoint (digest "
            f"{str(checkpoint_sha256)[:12] if checkpoint_sha256 else 'absent'}); epochs present "
            f"in the table are {sorted(seen_epochs)}. The counts are not recovered from a "
            "neighbouring epoch or from a row that merely shares a name."), None
    require(len({paths.digest_document(row) for _, row, _ in candidates}) == 1,
            f"{name} epoch {epoch} matches {len(candidates)} different checkpoint_table rows; an "
            "ambiguous progress binding is refused rather than resolved by position")
    key, row, matched_by = candidates[0]
    binding = {"table_key": key, "epoch": int(epoch), "matched_by": matched_by,
               "checkpoint": row.get("checkpoint"),
               "checkpoint_sha256": _first(row.get("checkpoint_sha256"), row.get("sha256")),
               "state_sha256": row.get("state_sha256")}
    updates = _first(row.get("updates"), row.get("steps"))
    exposures = _first(row.get("exposures"), row.get("sequences"))
    if updates is None and exposures is None:
        return None, None, (f"the {name} epoch {epoch} row carries no update or exposure "
                            "counts"), binding
    return (None if updates is None else int(updates)), exposures, None, binding


def parent_records(base_selection, *, seeds, logical_root, anchor):
    """The three selected epoch-3 SFT parents, from the frozen base selection.

    ``anchor`` turns the selection record's repository-relative checkpoint path
    into the campaign-relative suffix every other row uses. One convention: a
    ``logical_path`` is always resolved *under* its root, so a row that kept the
    root inside its own path would resolve to ``root/root/...``.
    """
    selected = base_selection.get("selected") or {}
    records = []
    for seed in seeds:
        name = f"policy_sft_seed{int(seed)}"
        entry = selected.get(name)
        require(isinstance(entry, dict),
                f"{name} is absent from the frozen base selection; the audit will not guess a parent")
        require(entry.get("kind") == "policy", f"{name} is not a policy checkpoint")
        require(int(entry.get("epoch", -1)) == 3,
                f"{name} is epoch {entry.get('epoch')}, but the selected parent is epoch 3")
        updates, exposures, progress_reason, progress_binding = parent_progress(
            base_selection, name, 3, checkpoint=entry.get("checkpoint"),
            checkpoint_sha256=entry.get("sha256"))
        records.append({
            "id": f"parent::{name}", "role": ROLE_PARENT, "name": name,
            "campaign": "her2_posttrain_20260918", "arm_id": "sft", "objective": "sft",
            "coefficients": {}, "seed": int(seed), "stage": None,
            "parent_id": f"parent::{name}", "parent_sha256": entry["sha256"],
            "nominal_budget_gpu_seconds": None, "updates": updates,
            "updates_reason": (None if updates is not None else progress_reason),
            "exposures": exposures,
            "exposures_reason": (None if exposures is not None else progress_reason),
            "recorded_path": entry["checkpoint"],
            "logical_path": paths.campaign_suffix(entry["checkpoint"], anchor=anchor),
            "recorded_path_flavor": paths.recorded_path_flavor(entry["checkpoint"]),
            "root": logical_root, "file_sha256": entry["sha256"],
            "payload_schema": POLICY_SCHEMA,
            "source_manifest": "base_selection.json",
            "progress_binding": progress_binding,
            # The selection table records the parent's own state digest beside its
            # counts, so the audit's recomputed digest has a second, independent
            # witness instead of only the payload it was read from.
            "state_sha256_recorded_by_manifest": (progress_binding or {}).get("state_sha256"),
            "contract": {"epoch": 3, "seed": int(seed), "arm_id": "sft"},
            "compare_contract": {},
            "historical_metrics_source": {},
            "aliases": []})
    return records


def guarded_endpoint_records(stage_documents, *, anchor, logical_root):
    """The reached guarded endpoints, counted once with their reuse aliases named.

    A continued-SFT control is fitted in stage 1 and *declared* again in stages 2
    and 3. The later declarations are aliases of one artifact; counting them again
    would turn three controls into nine observations.
    """
    records, index = [], {}
    for stage in sorted(stage_documents):
        document = stage_documents[stage]
        for entry in document.get("endpoints") or []:
            name = entry["name"]
            digest = entry["checkpoint_sha256"]
            binding = dict(entry.get("checkpoint_binding") or {})
            identifier = f"guarded::stage{int(stage)}::{entry['trajectory']}::budget" \
                         f"{int(float(entry['nominal_budget']))}"
            if name in index:
                index[name]["aliases"].append(
                    {"stage": int(stage), "name": name, "kind": "restated_endpoint",
                     "source_manifest": f"validation/stage{int(stage)}_endpoints.json"})
                require(index[name]["file_sha256"] == digest,
                        f"{name} is restated in stage {stage} with a different checkpoint digest")
                continue
            record = {
                "id": identifier, "role": ROLE_ENDPOINT, "name": name,
                "campaign": "her2_guarded_20260918", "arm_id": entry["arm_id"],
                "objective": entry["objective"], "coefficients": dict(entry.get("coefficients") or {}),
                "seed": int(entry["seed"]), "stage": int(stage),
                "trajectory": entry["trajectory"],
                "parent_id": f"parent::policy_sft_seed{int(entry['seed'])}",
                "parent_sha256": None,
                "nominal_budget_gpu_seconds": float(entry["nominal_budget"]),
                "actual_gpu_seconds": entry.get("actual_gpu_seconds"),
                # The binding block repeats some fields and not others; the endpoint
                # row's own top-level count is used when it does not.
                "updates": _first(binding.get("updates"), entry.get("updates")),
                "updates_reason": (None if _first(binding.get("updates"), entry.get("updates"))
                                   is not None else
                                   "neither checkpoint_binding nor the endpoint row records an "
                                   "update count"),
                "exposures": dict(entry.get("exposures") or {}) or None,
                "exposures_reason": (None if entry.get("exposures") else
                                     "the endpoint record carries no exposure block"),
                "recorded_path": entry["checkpoint"],
                "logical_path": paths.campaign_suffix(entry["checkpoint"], anchor=anchor),
                "recorded_path_flavor": paths.recorded_path_flavor(entry["checkpoint"]),
                "root": logical_root, "file_sha256": digest,
                "payload_schema": POLICY_SCHEMA,
                "source_manifest": f"validation/stage{int(stage)}_endpoints.json",
                "state_sha256_recorded_by_manifest": binding.get("state_sha256"),
                "gate_passed": entry.get("gate_passed"),
                "diversity_eligible": entry.get("diversity_eligible"),
                # ``parent_sha256`` is filled in by the caller from the verified
                # parent bank and is a REQUIRED binding: the payload records the
                # parent it actually inherited under identity.inherited.parents.
                "contract": {
                    "seed": int(entry["seed"]), "arm_id": entry["arm_id"], "stage": int(stage),
                    "budget_seconds": float(entry["nominal_budget"]),
                    "update": _first(binding.get("updates"), entry.get("updates"))},
                "compare_contract": {},
                # The historical numbers travel with the row that recorded them.
                # Reading them at score time from a document nobody kept is how the
                # published table ends up with an empty diversity column.
                "historical_metrics_source": {
                    "parent_kl": entry.get("parent_kl"),
                    "generation": entry.get("generation"),
                    "diversity": entry.get("diversity"),
                    "val_metrics": entry.get("val_metrics"),
                    "val_pair_metrics": entry.get("val_pair_metrics"),
                    "val_strata": entry.get("val_strata"),
                    "ranking": entry.get("ranking"),
                    "gate_evidence": entry.get("gate_evidence"),
                    "parent_draws": entry.get("parent_draws")},
                "aliases": []}
            index[name] = record
            records.append(record)
    return records


def _reused_control_entries(node):
    """``[(key, entry)]`` for a stage's ``reused_controls``, whichever shape it has.

    Stage 1 records an empty container; stages 2 and 3 record a **list** of endpoint
    objects. An earlier version of this function assumed a mapping and died on
    ``.items()`` against the real campaign, so both shapes are handled explicitly
    and anything else is refused rather than iterated by accident.
    """
    if not node:
        return []
    if isinstance(node, dict):
        return [(str(key), value) for key, value in sorted(node.items())]
    require(isinstance(node, list),
            f"reused_controls is a {type(node).__name__}; this audit reads the recorded list of "
            "endpoint objects or a mapping keyed by name, and guesses at nothing else")
    entries = []
    for position, value in enumerate(node):
        key = value.get("name") if isinstance(value, dict) else None
        entries.append((str(key if key is not None else position), value))
    return entries


#: The tuple a ``reused_controls`` entry uses to point at an already-fitted endpoint.
REUSED_CONTROL_REFERENCE = ("reused_from_stage", "arm_id", "seed", "nominal_budget")


def reused_control_reference(entry):
    """``((stage, arm_id, seed, budget), None)`` or ``(None, reason)`` for one entry.

    The real ``reused_controls`` entries carry **no name and no checkpoint digest**.
    They record ``reused_from_stage``, ``arm_id``, ``seed``, ``nominal_budget``,
    ``objective``, ``coefficients``, the gate evidence and the metrics -- a
    *reference* to an endpoint fitted in an earlier stage, not a copy of it. An
    earlier revision matched on ``entry["name"]``, which no such entry has, so all
    eighteen declarations stayed unmatched and alias resolution could never
    complete. The four-field tuple is the identity the documents actually share.
    """
    if not isinstance(entry, dict):
        return None, f"the entry is a {type(entry).__name__}, not an endpoint reference object"
    values = {"reused_from_stage": entry.get("reused_from_stage"),
              "arm_id": entry.get("arm_id"),
              "seed": entry.get("seed"),
              "nominal_budget": _first(entry.get("nominal_budget"), entry.get("budget_seconds"),
                                       entry.get("nominal_budget_gpu_seconds"))}
    missing = sorted(field for field in REUSED_CONTROL_REFERENCE if values[field] is None)
    if missing:
        return None, (f"the entry records no {missing}; a reused control is resolved by "
                      f"{list(REUSED_CONTROL_REFERENCE)} and by nothing else")
    return (int(values["reused_from_stage"]), str(values["arm_id"]), int(values["seed"]),
            float(values["nominal_budget"])), None


def reused_control_aliases(stage_documents, records):
    """Attach each stage's declared reused controls to the endpoint they alias.

    Resolution is by the reached-endpoint tuple the entry names, an ambiguous match
    is refused rather than resolved by position, and the declaring document stays
    on the alias so a reviewer can see which stage restated the control.
    """
    by_endpoint = {}
    for record in records:
        if record.get("role") != ROLE_ENDPOINT or record.get("nominal_budget_gpu_seconds") is None:
            continue
        by_endpoint.setdefault((int(record["stage"]), str(record["arm_id"]), int(record["seed"]),
                                float(record["nominal_budget_gpu_seconds"])), []).append(record)
    attached, unmatched, conflicts, ambiguous = 0, [], [], []
    for stage in sorted(stage_documents):
        source_manifest = f"validation/stage{int(stage)}_endpoints.json"
        for key, entry in _reused_control_entries(
                stage_documents[stage].get("reused_controls")):
            reference, reason = reused_control_reference(entry)
            if reference is None:
                unmatched.append({"stage": int(stage), "key": key, "reason": reason})
                continue
            described = dict(zip(("reused_from_stage", "arm_id", "seed", "nominal_budget"),
                                 reference))
            candidates = by_endpoint.get(reference) or []
            if not candidates:
                unmatched.append({"stage": int(stage), "key": key, "reference": described,
                                  "reason": ("no reached endpoint was enumerated at that stage, "
                                             "arm, seed and nominal budget")})
                continue
            if len(candidates) > 1:
                ambiguous.append({"stage": int(stage), "reference": described,
                                  "endpoints": sorted(record["id"] for record in candidates)})
                continue
            target = candidates[0]
            digest = _first(entry.get("checkpoint_sha256"), entry.get("sha256"))
            if digest and str(digest) != str(target["file_sha256"]):
                conflicts.append({"stage": int(stage), "reference": described,
                                  "declared_sha256": digest,
                                  "endpoint_sha256": target["file_sha256"]})
                continue
            target["aliases"].append(dict(described, stage=int(stage), kind="reused_control",
                                          key=key, endpoint_id=target["id"],
                                          source_manifest=source_manifest,
                                          declared_sha256=digest))
            attached += 1
    require(not conflicts,
            f"A stage reuses a control under a different checkpoint digest: {conflicts}. The same "
            "semantic control pointing at different bytes is not an alias.")
    require(not ambiguous,
            f"A reused-control reference resolves to more than one reached endpoint: {ambiguous}. "
            "An ambiguous alias is refused; it is not assigned to the first match.")
    return {"attached": attached, "unmatched": unmatched, "digest_conflicts": conflicts,
            "ambiguous": ambiguous,
            "resolved_by": list(REUSED_CONTROL_REFERENCE),
            "note": ("a reused control is one artifact under several semantic names; it is scored "
                     "once and never counted as an independent observation")}


def stopped_snapshot_records(stop_documents, summaries, *, anchor, logical_root):
    """The retained last-passing and failure states of the early-stopped trajectories."""
    records = []
    for key in sorted(stop_documents):
        stage, trajectory = key
        stop = stop_documents[key]
        summary = summaries.get(key) or {}
        arm = dict(summary.get("arm") or {})
        seed = int(summary.get("seed", 0)) or _seed_from(trajectory)
        parent = dict(summary.get("parent") or {})
        common = {"campaign": "her2_guarded_20260918", "arm_id": arm.get("arm_id"),
                  "objective": arm.get("objective"),
                  "coefficients": dict(arm.get("coefficients") or {}),
                  "seed": seed, "stage": int(stage), "trajectory": trajectory,
                  "parent_id": f"parent::policy_sft_seed{seed}",
                  "parent_sha256": parent.get("sha256"),
                  "nominal_budget_gpu_seconds": None,
                  "nominal_budget_reason": ("a diagnostic snapshot is not a nominal-budget "
                                            "endpoint and is never relabelled as one"),
                  "root": logical_root, "payload_schema": TRAJECTORY_SCHEMA,
                  "stop_reason": stop.get("stop_reason"),
                  "historical_metrics_source": {
                      "stop_check": stop.get("stop_check"),
                      "run": (stop.get("run") or {}) if isinstance(stop.get("run"), dict) else None,
                      "parent_kl": summary.get("parent_kl"),
                      "generation": summary.get("generation")},
                  "aliases": []}

        last = dict(stop.get("last_passing") or {})
        if last.get("kind") == "rolling_file" and last.get("checkpoint"):
            records.append(dict(common, **{
                "id": f"guarded::stage{int(stage)}::{trajectory}::last_passing",
                "role": ROLE_LAST_PASSING,
                "name": f"{trajectory}_last_passing",
                "updates": last.get("update"),
                "recorded_path": last["checkpoint"],
                "logical_path": paths.campaign_suffix(last["checkpoint"], anchor=anchor),
                "recorded_path_flavor": paths.recorded_path_flavor(last["checkpoint"]),
                "file_sha256": last["sha256"],
                "source_manifest": f"stage{int(stage)}/{trajectory}/stop.json",
                "rolling_note": ("last_passing.pt holds only its final retained state; the "
                                 "journalled digests of earlier checks describe bytes that were "
                                 "overwritten"),
                "contract": {"seed": seed, "arm_id": arm.get("arm_id"),
                             "stage": int(stage), "update": last.get("update"),
                             "parent_sha256": parent.get("sha256")},
                "compare_contract": {}}))
        else:
            records.append(dict(common, **{
                "id": f"guarded::stage{int(stage)}::{trajectory}::last_passing",
                "role": ROLE_LAST_PASSING, "name": f"{trajectory}_last_passing",
                "updates": last.get("update"), "recorded_path": last.get("checkpoint"),
                "logical_path": None, "recorded_path_flavor": None, "file_sha256": None,
                "source_manifest": f"stage{int(stage)}/{trajectory}/stop.json",
                "status": STATUS_MISSING,
                "status_reason": (f"the retained last-passing state is {last.get('kind')!r}, not a "
                                  "rolling file: this trajectory stopped before any passing check "
                                  "wrote bytes, so the parent itself is the last passing state"),
                "contract": {}}))

        failed = dict((stop.get("snapshot") or {}).get("failed_state") or {})
        if failed.get("path"):
            records.append(dict(common, **{
                "id": f"guarded::stage{int(stage)}::{trajectory}::failed_state",
                "role": ROLE_FAILED_STATE,
                "name": f"{trajectory}_failed_state",
                "updates": failed.get("update"),
                "recorded_path": failed["path"],
                "logical_path": paths.campaign_suffix(failed["path"], anchor=anchor),
                "recorded_path_flavor": paths.recorded_path_flavor(failed["path"]),
                "file_sha256": failed["sha256"],
                "source_manifest": f"stage{int(stage)}/{trajectory}/stop.json",
                "contract": {"seed": seed, "arm_id": arm.get("arm_id"),
                             "stage": int(stage), "update": failed.get("update"),
                             "parent_sha256": parent.get("sha256")},
                "compare_contract": {}}))
        else:
            records.append(dict(common, **{
                "id": f"guarded::stage{int(stage)}::{trajectory}::failed_state",
                "role": ROLE_FAILED_STATE, "name": f"{trajectory}_failed_state",
                "updates": None, "recorded_path": None, "logical_path": None,
                "recorded_path_flavor": None, "file_sha256": None,
                "source_manifest": f"stage{int(stage)}/{trajectory}/stop.json",
                "status": STATUS_MISSING,
                "status_reason": "the stop record names no failure snapshot; the save failed or "
                                 "the stop was not a gate breach",
                "contract": {}}))
    return records


def _seed_from(trajectory):
    marker = "_seed"
    require(marker in trajectory, f"Cannot read a seed out of trajectory name {trajectory!r}")
    return int(trajectory.rsplit(marker, 1)[1])


def v1_endpoint_records(summaries, *, logical_root, parents_by_seed, anchor,
                        validation_records=None, validation_source="validation_math/validation_records.json"):
    """The original-v1 DPO and continued-SFT endpoints named by their own summaries.

    ``validation_records`` is the complete, frozen global validation document,
    document, keyed ``{method}_seed{seed}_budget{int(budget)}``: that is where the v1
    campaign kept its validation, pair, generation and diversity numbers. The budget
    entries in the per-run summaries carry none of them, so an endpoint whose row is
    not found says so explicitly instead of publishing an empty column.
    """
    records = []
    for run_key in sorted(summaries):
        summary = summaries[run_key]
        method = summary.get("method")
        seed = int(summary.get("seed"))
        require(method in ("dpo", "continued_sft"), f"{run_key}: unexpected v1 method {method!r}")
        for budget_key in sorted((summary.get("budgets") or {}), key=float):
            entry = summary["budgets"][budget_key]
            if not entry.get("checkpoint"):
                continue
            budget = float(budget_key)
            historical = v1_historical_metrics(
                validation_records, method=method, seed=seed, budget=budget,
                checkpoint=entry.get("checkpoint"),
                checkpoint_sha256=entry.get("checkpoint_sha256"), source=validation_source)
            records.append({
                "id": f"v1::{run_key}::budget{int(budget)}", "role": ROLE_V1_ENDPOINT,
                "name": f"v1_{method}_seed{seed}_budget{int(budget)}",
                "campaign": "her2_posttrain_20260918", "arm_id": method, "objective": method,
                "coefficients": {}, "seed": seed, "stage": None, "trajectory": run_key,
                "parent_id": f"parent::policy_sft_seed{seed}",
                "parent_sha256": parents_by_seed.get(seed),
                "nominal_budget_gpu_seconds": budget,
                "actual_gpu_seconds": entry.get("actual_gpu_seconds"),
                "updates": entry.get("updates"),
                "exposures": dict(entry.get("exposures") or {}) or None,
                "exposures_reason": (None if entry.get("exposures") else
                                     "the v1 budget record carries no exposure block"),
                "recorded_path": entry["checkpoint"],
                "logical_path": paths.campaign_suffix(entry["checkpoint"], anchor=anchor),
                "recorded_path_flavor": paths.recorded_path_flavor(entry["checkpoint"]),
                "root": logical_root, "file_sha256": entry["checkpoint_sha256"],
                "payload_schema": POLICY_SCHEMA,
                "state_sha256_recorded_by_manifest": entry.get("state_sha256"),
                "source_manifest": f"continuation/{run_key}/summary.json",
                "contract": {"seed": seed, "method": method, "budget_seconds": budget,
                             "update": entry.get("updates"),
                             "parent_sha256": parents_by_seed.get(seed)},
                "compare_contract": {},
                "diversity_eligible": historical.get("diversity_eligible"),
                "historical_metrics_source": historical,
                "aliases": []})
    return records


#: Fields the audit carries out of a v1 validation row, and where they live on it.
V1_METRIC_FIELDS = ("val_metrics", "val_pair_metrics", "val_strata", "generation", "parent_kl",
                    "ranking", "diversity", "val_positive_nll_per_residue")


def v1_record_key(method, seed, budget):
    """The key the global v1 validation document files this endpoint under."""
    return f"{method}_seed{int(seed)}_budget{int(float(budget))}"


def _v1_row(document, key):
    """``(row, reason)`` for one key of the global validation document."""
    if isinstance(document, dict):
        nested = document.get("records")
        if isinstance(nested, dict):
            document = nested
        elif isinstance(nested, list):
            document = {str(row.get("name")): row for row in nested if isinstance(row, dict)}
    elif isinstance(document, list):
        document = {str(row.get("name")): row for row in document if isinstance(row, dict)}
    else:
        return None, (f"validation_records.json is a {type(document).__name__}; this audit reads "
                      "the recorded mapping of run-budget keys to rows")
    row = document.get(key)
    if not isinstance(row, dict):
        return None, (f"validation_records.json holds no {key!r} row; a neighbouring budget's "
                      "metrics describe different weights and are not substituted")
    return row, None


def v1_historical_metrics(document, *, method, seed, budget, checkpoint=None,
                          checkpoint_sha256=None, source="validation_math/validation_records.json"):
    """The v1 validation row for exactly this run and budget, or a named absence.

    One global validation document holds every row, keyed
    ``{method}_seed{seed}_budget{int(budget)}``. Looking for a per-run file beside
    each summary -- which is what an earlier revision did -- found nothing, so all
    24 v1 endpoints published empty validation, pair, generation and diversity
    columns while the numbers sat in a document nobody opened.

    The key is exact, and the row is then *confirmed* against the endpoint's own
    method, seed, budget and checkpoint digest. A row that disagrees is not silently
    preferred to the summary: it is reported as a conflict, because two documents
    describing the same endpoint differently is a finding.
    """
    key = v1_record_key(method, seed, budget)
    if not document:
        return {"key": key,
                "reason": (f"no {source} was read, so no historical "
                           f"metrics are attached to the {budget:g} s endpoint")}
    row, reason = _v1_row(document, key)
    if row is None:
        return {"key": key, "reason": reason}
    conflicts, notes = {}, {}
    digests_agree = (checkpoint_sha256 is not None and row.get("sha256") is not None
                     and _equalish(row["sha256"], checkpoint_sha256))
    for field, wanted in (("method", method), ("seed", int(seed)),
                          ("budget_seconds", float(budget)),
                          ("sha256", checkpoint_sha256), ("checkpoint", checkpoint)):
        observed = row.get(field)
        if wanted is None or observed is None:
            continue
        if field == "checkpoint":
            # One recorded path convention: a row written with backslashes and a
            # summary written with forward ones name the same file.
            observed = str(observed).replace("\\", "/")
            wanted = str(wanted).replace("\\", "/")
        if _equalish(observed, wanted):
            continue
        if field == "checkpoint" and digests_agree:
            # Same bytes under two recorded spellings. That is a naming difference,
            # and it is recorded rather than treated as a different artifact -- the
            # digest is the identity.
            notes[field] = {"row": observed, "endpoint": wanted,
                            "resolved_by": "the checkpoint digests agree"}
            continue
        conflicts[field] = {"row": observed, "endpoint": wanted}
    require(not conflicts,
            f"The v1 validation row {key!r} disagrees with the endpoint it is joined to at "
            f"{conflicts}. Two documents describing the same endpoint differently is a finding, "
            "not a column to fill from whichever one was read last.")
    block = {"key": key, "source": source,
             "joined_on": ["method", "seed", "budget_seconds"],
             "recorded_differences": notes,
             "confirmed_by": sorted(field for field in ("sha256", "checkpoint")
                                    if row.get(field) is not None and field not in notes)}
    absent = {}
    for field in V1_METRIC_FIELDS:
        value = row.get(field)
        if field == "generation":
            value = _first(row.get("generation"), row.get("generation_summary"))
        if value is None:
            absent[field] = (f"the {key!r} validation row records no {field!r}; it is reported "
                             "absent rather than filled from another row")
        block[field] = value
    diversity = row.get("diversity")
    block["diversity_eligible"] = (diversity.get("eligible")
                                   if isinstance(diversity, dict) else None)
    if block["diversity_eligible"] is None:
        absent["diversity_eligible"] = (f"the {key!r} validation row records no diversity "
                                        "eligibility verdict")
    block["not_recorded_by_row"] = absent
    return block


# ---------------------------------------------------------------------------
# diagnostic exposure recovery
# ---------------------------------------------------------------------------

def exposures_at_update(journal_path, update):
    """The exposure counts recorded at exactly ``update`` in ``updates.jsonl``.

    An interpolation, a nearest match or the final line would all produce a
    plausible number for the wrong state. Only the exact record counts; anything
    else returns ``None`` with the reason, which the inventory records verbatim.
    """
    journal_path = Path(journal_path)
    if update is None:
        return None, "the snapshot records no update number"
    if not journal_path.is_file():
        return None, f"{journal_path.name} is absent for this trajectory"
    target = int(update)
    with journal_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if int(row.get("update", -1)) == target:
                exposures = row.get("exposures")
                if isinstance(exposures, dict) and exposures:
                    return {str(k): int(v) for k, v in sorted(exposures.items())}, None
                return None, f"the update {target} record carries no exposure block"
    return None, (f"no update {target} record exists in {journal_path.name}; a nearby update's "
                  "counts describe different weights and are not substituted")


# ---------------------------------------------------------------------------
# parent draw banks
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParentBank:
    """One seed's persisted 10,000 temperature-1 parent draws, independently verified."""

    parent_id: str
    seed: int
    name: str
    logical_path: str
    file_sha256: str
    rows: int
    temperature: float
    draw_seed: int
    parent_sha256: str
    index: np.ndarray
    order_sha256: str

    def document(self):
        return {"parent_id": self.parent_id, "seed": self.seed, "name": self.name,
                "logical_path": self.logical_path, "file_sha256": self.file_sha256,
                "rows": self.rows, "temperature": self.temperature, "draw_seed": self.draw_seed,
                "parent_sha256": self.parent_sha256, "core_order_sha256": self.order_sha256,
                "unique_cores": int(np.unique(self.index, axis=0).shape[0]),
                "duplicates_retained": True,
                "probability_convention": PROBABILITY_CONVENTION,
                "note": ("repeated stage references to this deterministic bank are not additional "
                         "samples")}


def verify_parent_bank(reference, *, path, expected_rows, expected_temperature, parent_sha256,
                       parent_id):
    """Re-derive every claim the validation document makes about one bank.

    File hash, row count, ``draw_index`` in exact 0..N-1 order, ten legal canonical
    residues per row, temperature, and the parent the draws came from. Duplicates
    stay: they are the sampling law.
    """
    import pandas as pd
    path = Path(path)
    require(path.is_file(), f"Parent bank {reference['path']} is missing at its resolved location")
    digest = paths.sha256_file(path)
    require(digest == reference["sha256"],
            f"Parent bank {reference['path']} hashes {digest}, the validation document recorded "
            f"{reference['sha256']}")
    frame = pd.read_csv(path, dtype={"core": str}, keep_default_na=False)
    require(list(frame.columns) == ["draw_index", "core"],
            f"{reference['path']} columns are {list(frame.columns)}, expected draw_index,core")
    require(len(frame) == int(expected_rows),
            f"{reference['path']} has {len(frame)} rows, expected {expected_rows}")
    order = frame.draw_index.to_numpy()
    require(bool((order == np.arange(len(frame))).all()),
            f"{reference['path']} draw_index is not the exact 0..{len(frame) - 1} order; the bank "
            "is a sequence, and a re-sorted file is a different measurement")
    index = encode_cores(frame.core)
    require(index.shape == (int(expected_rows), CORE_LENGTH),
            f"{reference['path']} did not decode to ({expected_rows}, {CORE_LENGTH}) cores")
    require(float(reference["temperature"]) == float(expected_temperature),
            f"{reference['path']} was drawn at temperature {reference['temperature']}, not "
            f"{expected_temperature}; a different temperature is a different sampling "
            "distribution and needs its own estimator")
    require(reference["parent_sha256"] == parent_sha256,
            f"{reference['path']} is bound to parent {reference['parent_sha256']}, not "
            f"{parent_sha256}")
    require(int(reference["draws"]) == int(expected_rows),
            f"{reference['path']} declares {reference['draws']} draws but holds {expected_rows}")
    return ParentBank(parent_id=parent_id, seed=int(reference["seed"]), name=reference["name"],
                      logical_path=reference["path"], file_sha256=digest, rows=int(len(frame)),
                      temperature=float(reference["temperature"]),
                      draw_seed=int(reference["draw_seed"]), parent_sha256=parent_sha256,
                      index=index, order_sha256=core_digest(index))


def cross_stage_bank_consistency(stage_documents):
    """Every stage must name the same bank bytes for the same seed."""
    seen, problems = {}, []
    for stage in sorted(stage_documents):
        for seed, entry in sorted((stage_documents[stage].get("parent_draw_references")
                                   or {}).items()):
            key = str(seed)
            previous = seen.get(key)
            if previous is None:
                seen[key] = {"stage": int(stage), "path": entry["path"],
                             "sha256": entry["sha256"], "draw_seed": entry["draw_seed"],
                             "temperature": entry["temperature"],
                             "parent_sha256": entry["parent_sha256"]}
                continue
            for field in ("path", "sha256", "draw_seed", "temperature", "parent_sha256"):
                if previous[field] != entry[field]:
                    problems.append(
                        f"seed {key}: stage {stage} records {field}={entry[field]!r} but stage "
                        f"{previous['stage']} recorded {previous[field]!r}")
    require(not problems,
            "The stages disagree about the parent draw banks: " + "; ".join(problems))
    return {"seeds": sorted(seen), "consistent_across_stages": True,
            "banks": {key: dict(value) for key, value in sorted(seen.items())}}


# ---------------------------------------------------------------------------
# deduplication and coverage
# ---------------------------------------------------------------------------

def dedup_key(record, *, bank):
    """Identity of a *computation*, not of a file.

    Two records share a key only when the weights, the parent, the bank bytes, the
    scored population, its row order and the probability contract are all the same.
    Anything less would merge two measurements that are not the same measurement.
    """
    payload = {
        "state_digest": record.get("state_digest_audit_computed"),
        "parent_id": record.get("parent_id"),
        "bank_sha256": None if bank is None else bank.file_sha256,
        "population_id": None if bank is None else f"parent_bank::{bank.parent_id}",
        "row_order_sha256": None if bank is None else bank.order_sha256,
        "contract": dict(CORE_CONTRACT)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def deduplicate(records):
    """Group records by computation key; the first id in sorted order is the scorer."""
    groups = {}
    for record in records:
        key = record.get("dedup_key")
        if key is None:
            continue
        groups.setdefault(key, []).append(record["id"])
    assignment = {}
    for key, members in groups.items():
        members.sort()
        primary = members[0]
        for identifier in members:
            assignment[identifier] = {"dedup_key": key, "scored_as": primary,
                                      "is_primary": identifier == primary,
                                      "alias_of": None if identifier == primary else primary}
    duplicates = {key: members for key, members in sorted(groups.items()) if len(members) > 1}
    return assignment, {"distinct_computations": len(groups), "duplicate_groups": duplicates,
                        "note": ("identical states are computed once and every semantic alias is "
                                 "retained; no alias is counted as an independent observation")}


def coverage_table(records, *, expected):
    """Counts by role and status, semantic *and* verified, against the protocol.

    A row that exists is not coverage. An enumerated state whose verification
    failed, whose file is missing or whose identity could not be confirmed is a
    gap, and it is counted as one here: the earlier version of this function
    reported ``complete`` for a population in which the only declared parent had
    failed to load, because the row was present in the list.
    """
    by_role, by_status, verified_by_role = {}, {}, {}
    for record in records:
        role = record["role"]
        by_role[role] = by_role.get(role, 0) + 1
        status = record.get("status", STATUS_MISSING)
        by_status[status] = by_status.get(status, 0) + 1
        if status == STATUS_VERIFIED:
            verified_by_role[role] = verified_by_role.get(role, 0) + 1
    observed_total = len(records)
    verified_total = by_status.get(STATUS_VERIFIED, 0)
    shortfalls = []
    for role, wanted in sorted((expected.get("by_role") or {}).items()):
        got = by_role.get(role, 0)
        if got != wanted:
            shortfalls.append({"role": role, "kind": "enumeration", "expected": wanted,
                               "observed": got})
        confirmed = verified_by_role.get(role, 0)
        if confirmed != wanted:
            shortfalls.append({"role": role, "kind": "verification", "expected": wanted,
                               "observed": confirmed,
                               "detail": [record["id"] for record in records
                                          if record["role"] == role
                                          and record.get("status") != STATUS_VERIFIED]})
    total_expected = expected.get("total")
    if total_expected is not None and observed_total != total_expected:
        shortfalls.append({"role": "total", "kind": "enumeration", "expected": total_expected,
                           "observed": observed_total})
    if total_expected is not None and verified_total != total_expected:
        shortfalls.append({"role": "total", "kind": "verification", "expected": total_expected,
                           "observed": verified_total})
    return {"by_role": dict(sorted(by_role.items())),
            "verified_by_role": dict(sorted(verified_by_role.items())),
            "by_status": dict(sorted(by_status.items())),
            "total": observed_total, "verified_total": verified_total,
            "expected": dict(expected), "shortfalls": shortfalls,
            "complete": not shortfalls,
            "note": ("semantic enumeration and verified count are reported apart. A shortfall in "
                     "either is a coverage gap: it never becomes a no-go preservation finding and "
                     "it never narrows the population silently.")}


#: Everything a *complete* audit has to be able to point at. A stage that exited is
#: not one of them; each entry is an artifact or a count somebody can re-check.
COMPLETION_REQUIREMENTS = ("inventory_coverage", "state_verification", "parent_banks",
                           "scored_computations", "alias_resolution", "pair_populations",
                           "ches_populations", "numerical_controls", "decision_inputs",
                           "stage_status")


def audit_completion(requirements):
    """Fold the declared requirement checks into one truthful completion verdict.

    ``requirements`` maps each name in :data:`COMPLETION_REQUIREMENTS` to
    ``{"satisfied": bool, "detail": ...}``. A missing entry is *not* satisfied:
    an audit cannot become complete by forgetting to evaluate one of its own
    requirements, which is exactly how "the process exited" turns into "the audit
    passed".
    """
    rows, unmet = {}, []
    for name in COMPLETION_REQUIREMENTS:
        entry = dict(requirements.get(name) or {})
        satisfied = bool(entry.get("satisfied", False))
        if name not in requirements:
            entry = {"satisfied": False, "detail": "this requirement was never evaluated"}
            satisfied = False
        entry["satisfied"] = satisfied
        rows[name] = entry
        if not satisfied:
            unmet.append(name)
    return {"requirements": rows, "unmet": unmet, "complete": not unmet,
            "note": ("a complete audit is one where every declared requirement was evaluated and "
                     "satisfied. Missing CHES, a missing bank, a numerical failure or incomplete "
                     "scoring leaves the audit incomplete, and an incomplete audit cannot produce "
                     "a preservation finding.")}
