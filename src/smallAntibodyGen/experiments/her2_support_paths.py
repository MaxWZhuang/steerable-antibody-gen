"""Portable paths, hash-probed roots, LF artifacts and run bookkeeping for the audit.

Nothing here is scientific. It exists because the HER2 support audit reads two
completed campaigns whose artifacts were written on one Windows box, through a
junction, with absolute paths baked into some of their JSON -- and it has to
publish evidence that a reviewer on another machine can verify byte for byte.
Five properties are enforced here rather than left to convention:

* **A root is accepted by hash, never by existence.** ``outputs/her2_guarded_20260918``
  is a directory junction to another volume on the recording box and an ordinary
  directory (or nothing at all) elsewhere. :func:`resolve_root` therefore probes
  candidate roots by re-hashing files whose digests a *published* manifest already
  fixed. ``Path.exists()`` proves that something is there, which is not the claim
  the audit needs.
* **Recorded paths keep their flavor.** Legacy records carry
  ``C:\\Users\\...\\outputs\\her2_guarded_20260918\\stage1\\...``; new records carry
  ``outputs/her2_guarded_20260918/...``. :func:`split_recorded` parses the flavor
  explicitly. ``Path(recorded).name`` on POSIX returns the whole Windows string,
  so a basename search is not a fallback -- and it is not allowed anyway, because
  two campaigns contain files called ``summary.json``.
* **Containment is checked without resolving.** ``Path.resolve()`` follows the
  junction onto ``F:``, at which point every containment test against the logical
  root fails. :func:`resolve_under` normalizes lexically instead, so the local
  mapping stays irrelevant to the logical name.
* **New artifacts are LF, sorted, finite and atomic.** ``her2_runtime.save_json``
  writes through ``Path.write_text``, which translates ``\\n`` to ``\\r\\n`` on
  Windows; every historical artifact is therefore CRLF and stays that way. The
  writers here open with ``newline=""`` so a new artifact's raw sha256 equals
  :func:`digest_document` of the same document on every platform.
* **Progress is recorded, and a total that is not known is ``None``.** A stage
  that cannot know its denominator in advance says so; the dashboard renders a
  count, not a fabricated percentage.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np

from .her2_runtime import canonical_json, require, sha256

#: Schema tag on every artifact this audit writes.
AUDIT_SCHEMA = "her2-support-audit/1"
#: New output root. The audit reads the original and guarded roots and writes here.
AUDIT_RUN_ROOT = "outputs/her2_support_audit_20260919"
#: Tracked, published evidence. Narrow ``text eol=lf`` rules cover this whole tree.
EVIDENCE_ROOT = "reference/evidence/her2-support-audit-2026-09-19"
#: Machine-local root mapping. Ignored by Git; never referenced by a scientific field.
LOCAL_ROOTS_FILE = "local_roots.json"
#: Written by the freeze stage after Codex commits. Ignored; points at the real HEAD.
FREEZE_MARKER = "audit_spec_frozen.json"

WINDOWS_ABSOLUTE = "windows_absolute"
WINDOWS_UNC = "windows_unc"
WINDOWS_RELATIVE = "windows_relative"
WINDOWS_DRIVE_RELATIVE = "windows_drive_relative"
POSIX_ABSOLUTE = "posix_absolute"
RELATIVE = "relative"

_DRIVE = re.compile(r"^[A-Za-z]:[\\/]")
_BARE_DRIVE = re.compile(r"^[A-Za-z]:$")
#: ``C:untrusted\x`` is *drive-relative*: it names the current directory of drive C:,
#: which is process state. Two runs resolve it to two different files, so it is never
#: mapped onto a root here.
_DRIVE_RELATIVE = re.compile(r"^[A-Za-z]:(?![\\/])")
_CONTROL = re.compile(r"[\x00-\x1f]")


# ---------------------------------------------------------------------------
# recorded paths and logical names
# ---------------------------------------------------------------------------

def recorded_path_flavor(recorded):
    """Which path grammar a *recorded* string was written in.

    Reported in every inventory row. A path that came out of a Windows run and a
    path that came out of this audit are different kinds of evidence, and the
    difference is not recoverable once both have been through ``Path()``.
    """
    require(isinstance(recorded, str) and recorded.strip() != "",
            "A recorded path must be a non-empty string")
    if recorded.startswith("\\\\") or recorded.startswith("//?"):
        return WINDOWS_UNC
    if _DRIVE.match(recorded):
        return WINDOWS_ABSOLUTE
    if _BARE_DRIVE.match(recorded) or _DRIVE_RELATIVE.match(recorded):
        return WINDOWS_DRIVE_RELATIVE
    if recorded.startswith("/"):
        return POSIX_ABSOLUTE
    if "\\" in recorded:
        return WINDOWS_RELATIVE
    return RELATIVE


def split_recorded(recorded):
    """``(flavor, parts)`` for a recorded path, with the anchor dropped.

    Three grammars are refused outright rather than normalized, because each of
    them resolves to different bytes depending on *where* it is resolved:

    * a UNC path is a network location, and a network path in a historical record
      is a provenance question, not something to map onto a local directory;
    * ``C:untrusted\\x`` is drive-relative -- it names the per-drive current
      directory, which is process state;
    * a ``..`` (or ``.``) anywhere in the path means the recorded string and the
      file it names are not in one-to-one correspondence, so an anchored suffix
      taken from it would be a guess. ``C:/outside/../campaign/stage1/f.pt`` is a
      path *into* the campaign only if ``outside`` happens to exist.
    """
    flavor = recorded_path_flavor(recorded)
    require(flavor != WINDOWS_UNC,
            f"{recorded!r} is a UNC path; this audit resolves local campaign roots only")
    require(flavor != WINDOWS_DRIVE_RELATIVE,
            f"{recorded!r} is drive-relative: it names the current directory of that drive, which "
            "is process state, so it does not identify a file across runs")
    if flavor in (WINDOWS_ABSOLUTE, WINDOWS_RELATIVE):
        pure = PureWindowsPath(recorded)
    elif flavor == POSIX_ABSOLUTE:
        pure = PurePosixPath(recorded)
    else:
        pure = PurePosixPath(recorded.replace("\\", "/"))
    parts = [part for part in pure.parts if part not in ("", "/", "\\")]
    if parts and (_BARE_DRIVE.match(parts[0]) or parts[0].endswith((":\\", ":/"))):
        parts = parts[1:]
    traversal = [part for part in parts if part in (".", "..")]
    require(not traversal,
            f"{recorded!r} traverses with {traversal}; a recorded path that walks up or through "
            "the tree is rejected wherever the component appears, not only at its suffix")
    require(parts, f"{recorded!r} names no path components")
    return flavor, tuple(parts)


def require_logical_name(logical):
    """Validate one forward-slash relative logical name and return its components.

    A logical name identifies an artifact across machines. Absolute roots, drive
    letters, backslashes, ``..`` and empty components are all refused rather than
    normalized: every one of them is a case where two different strings would
    resolve to the same file on one machine and different files on another.
    """
    require(isinstance(logical, str) and logical != "", "A logical name must be a non-empty string")
    require("\\" not in logical, f"{logical!r} uses backslashes; logical names are forward-slash")
    require(not logical.startswith("/"), f"{logical!r} is absolute; logical names are relative")
    require(not _DRIVE.match(logical) and not _BARE_DRIVE.match(logical),
            f"{logical!r} carries a drive letter")
    require(not _CONTROL.search(logical), f"{logical!r} carries a control character")
    parts = tuple(logical.split("/"))
    for part in parts:
        require(part not in ("", ".", ".."),
                f"{logical!r} has an empty or traversing component {part!r}")
        require(":" not in part, f"{logical!r} component {part!r} carries a colon")
        require(part == part.strip() and not part.endswith("."),
                f"{logical!r} component {part!r} has trailing whitespace or a trailing dot")
    return parts


def logical_join(*parts):
    """Build a logical name from components, validating the result."""
    joined = "/".join(str(part).strip("/") for part in parts if str(part) != "")
    require_logical_name(joined)
    return joined


def campaign_relative(recorded, *, anchor):
    """The suffix of ``recorded`` below the directory named ``anchor``.

    This is the only supported way to map a legacy absolute path onto a resolved
    root. The anchor must appear exactly once: a basename search would happily
    match ``summary.json`` in either campaign, and a last-occurrence rule would
    quietly prefer a nested copy. A component that differs from the anchor only by
    case is reported as a collision rather than accepted, because the recording
    filesystem was case-insensitive and the verifying one may not be.
    """
    require(isinstance(anchor, str) and anchor != "", "An anchor directory name is required")
    _, parts = split_recorded(recorded)
    exact = [index for index, part in enumerate(parts) if part == anchor]
    folded = [index for index, part in enumerate(parts)
              if part.casefold() == anchor.casefold() and part != anchor]
    require(not folded,
            f"{recorded!r} contains {[parts[i] for i in folded]}, which differs from the anchor "
            f"{anchor!r} only by case; this audit will not resolve a case-colliding path")
    require(len(exact) == 1,
            f"{recorded!r} contains the anchor {anchor!r} {len(exact)} times; a recorded path is "
            "resolved by its single anchored suffix, never by a basename search")
    suffix = "/".join(parts[exact[0] + 1:])
    require(suffix != "", f"{recorded!r} names the anchor directory itself, not a file under it")
    require_logical_name(suffix)
    return suffix


def campaign_suffix(recorded, *, anchor):
    """The campaign-relative suffix of a recorded path, whichever way it was written.

    The historical campaigns recorded some paths with the campaign root inside
    them (absolute Windows paths, and repository-relative ones) and some already
    relative to that root. Both are resolved here to the single convention every
    inventory row uses -- a suffix that is joined *under* a separately resolved
    root -- and nothing else is accepted: an absolute path that does not contain
    the anchor is a path into some other tree, and it is refused rather than
    reinterpreted. This is not a basename search; the anchored form still has to
    contain the anchor exactly once.
    """
    flavor, parts = split_recorded(recorded)
    if anchor in parts:
        return campaign_relative(recorded, anchor=anchor)
    folded = [part for part in parts if part.casefold() == anchor.casefold()]
    require(not folded,
            f"{recorded!r} contains {folded}, which differs from the anchor {anchor!r} only by "
            "case; this audit will not resolve a case-colliding path")
    require(flavor in (RELATIVE, WINDOWS_RELATIVE),
            f"{recorded!r} is an absolute path that does not pass through {anchor!r}; it names a "
            "file in another tree and is not resolved under this campaign root")
    suffix = "/".join(parts)
    require_logical_name(suffix)
    return suffix


def case_collisions(names):
    """Groups of logical names that differ only by case, as ``{folded: [names]}``."""
    groups = {}
    for name in names:
        groups.setdefault(str(name).casefold(), []).append(str(name))
    return {folded: sorted(set(group)) for folded, group in groups.items()
            if len(set(group)) > 1}


def require_no_case_collisions(names, *, where):
    collisions = case_collisions(names)
    require(not collisions,
            f"{where}: logical names collide under case folding {collisions}. Two names that are "
            "the same file on Windows and different files on Linux cannot both be artifacts.")
    return True


def resolve_under(root, logical):
    """Join a logical name under a local root, refusing anything that escapes it.

    Lexical normalization only. ``Path.resolve()`` would follow the junction that
    maps the guarded output directory onto another volume, and every containment
    check against the logical root would then fail on the machine the campaign
    actually ran on.
    """
    parts = require_logical_name(logical)
    base = os.path.normpath(os.path.abspath(str(root)))
    candidate = os.path.normpath(os.path.join(base, *parts))
    require(candidate == base or candidate.startswith(base + os.sep),
            f"{logical!r} resolves outside its root: {candidate!r} is not under {base!r}")
    return Path(candidate)


# ---------------------------------------------------------------------------
# hashes and array records
# ---------------------------------------------------------------------------

def sha256_file(path):
    return sha256(path)


def sha256_bytes(payload):
    return hashlib.sha256(bytes(payload)).hexdigest()


def sha256_text(text):
    """UTF-8 digest of text exactly as given. No newline translation happens here."""
    return sha256_bytes(str(text).encode("utf-8"))


def digest_document(document):
    """Hash of the canonical JSON *bytes* this module would write for ``document``."""
    return sha256_text(canonical_json(document))


def array_digest(array):
    """Content digest of one array: dtype, shape and C-ordered bytes.

    Container bytes are not part of it. ``numpy.savez`` stamps the current time
    into the zip entries, so two byte-different ``.npz`` files can hold identical
    numbers; the scientific identity of a saved vector is this digest, and the file
    hash beside it is recorded as the container fact it is.
    """
    values = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(str(values.shape).encode("ascii"))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def array_record(array, *, logical, order):
    """The per-array block every completed output manifest carries."""
    values = np.asarray(array)
    return {"file": logical, "dtype": str(values.dtype), "shape": list(values.shape),
            "order": order, "content_sha256": array_digest(values)}


# ---------------------------------------------------------------------------
# atomic LF writers
# ---------------------------------------------------------------------------

def write_text(path, text):
    """Write UTF-8 text with literal LF newlines, atomically. Returns its sha256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    os.replace(temporary, path)
    return sha256_file(path)


def write_json(path, document):
    """Sorted-key, ``allow_nan=False``, LF JSON. The file hash is the document digest."""
    text = canonical_json(document)
    digest = write_text(path, text)
    require(digest == sha256_text(text),
            f"{path} did not round-trip its own bytes; newline translation is active")
    return digest


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_arrays(path, arrays):
    """Atomically write a ``.npz`` of named arrays; returns the container sha256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    with temporary.open("wb") as stream:
        np.savez(stream, **{str(name): np.asarray(value) for name, value in arrays.items()})
    os.replace(temporary, path)
    return sha256_file(path)


def read_arrays(path):
    with np.load(Path(path)) as handle:
        return {name: handle[name] for name in handle.files}


# ---------------------------------------------------------------------------
# shards: a completed output is the record, not the bytes beside it
# ---------------------------------------------------------------------------

#: Suffix of the completion record written *after* a shard's arrays validate.
SHARD_RECORD = ".complete.json"

#: Keys whose values are operational, not scientific. A rerun is allowed to produce
#: different ones and a completed artifact is never rewritten to carry the new
#: values: elapsed time and an ``npz`` container hash (which embeds archive
#: timestamps) say nothing about what was measured.
MUTABLE_KEYS = frozenset({
    "generated_at", "recorded_at", "heartbeat_at", "frozen_at", "completed_at",
    "timings", "container", "probe_on_reuse", "rerun", "reused", "reused_completed_shard",
    "wall_seconds", "elapsed_wall_seconds", "inference_seconds", "io_seconds",
    "analysis_seconds", "unattributed_seconds", "seconds_for_probe_batch", "free_vram_mib"})


def scientific_projection(node, mutable=MUTABLE_KEYS):
    """``node`` with every operational key removed, recursively. The comparison basis."""
    if isinstance(node, dict):
        return {key: scientific_projection(value, mutable)
                for key, value in node.items() if key not in mutable}
    if isinstance(node, list):
        return [scientific_projection(value, mutable) for value in node]
    return node


def shard_logicals(logical_prefix, name):
    """``(record_logical, container_logical)`` for one shard, as ledger keys."""
    return f"{logical_prefix}/{name}{SHARD_RECORD}", f"{logical_prefix}/{name}.npz"


def register_shard(run_root, directory, name, record, *, logical_prefix):
    """Bind a completed shard's record **and** its container into the completion manifest.

    Both files, not just the JSON. A completion summary that claims a shard whose
    ``.npz`` has since been deleted is exactly the state the ledger exists to catch,
    and a directory scan cannot catch it: it only sees what is still there.
    """
    directory = Path(directory)
    record_logical, container_logical = shard_logicals(logical_prefix, name)
    entries = [record_completion(
        run_root, record_logical, directory / f"{name}{SHARD_RECORD}", kind="shard_record",
        scientific_digest=digest_document(scientific_projection(record)),
        timings=record.get("timings"))]
    container = record.get("container") or {}
    entries.append(record_completion(
        run_root, container_logical, directory / f"{name}.npz", kind="shard_container",
        scientific_digest=digest_document(record.get("arrays") or {}),
        timings=record.get("timings")))
    return entries


def require_registered_shard(run_root, directory, name, *, logical_prefix):
    """On a rerun, the ledger must already hold this shard; it is not re-bound now.

    Registering a completed-looking shard *during* the rerun would bootstrap trust
    in whatever is on disk at that moment: an edited record would simply be
    recorded as the truth. The entries have to predate the rerun, and the bytes have
    to still match them.
    """
    directory = Path(directory)
    artifacts = (read_completion_manifest(run_root).get("artifacts") or {})
    problems = []
    for logical, leaf in zip(shard_logicals(logical_prefix, name),
                             (f"{name}{SHARD_RECORD}", f"{name}.npz")):
        entry = artifacts.get(logical)
        if entry is None:
            problems.append(f"{logical} is complete on disk but the completion manifest never "
                            "recorded it")
            continue
        target = directory / leaf
        observed = sha256_file(target) if target.is_file() else None
        if observed != entry["sha256"]:
            problems.append(f"{logical} hashes {observed}, the completion manifest recorded "
                            f"{entry['sha256']}")
    require(not problems,
            "A completed shard does not match the saved completion manifest: "
            + "; ".join(problems)
            + ". The manifest is the authority a rerun is checked against; it is not re-written "
              "from the files it is supposed to be checking.")
    return True


def write_shard(directory, name, arrays, document, *, order, logical_prefix, run_root):
    """Write one array shard and, only after it validates, its completion record.

    A killed process leaves the ``.npz`` and no record. :func:`read_shard` refuses
    that, so a partial shard can never be mistaken for a completed one -- which is
    the failure mode a reruns-are-cheap audit walks into first.

    A shard that is already complete is **verified, not rewritten**. The recomputed
    arrays must reproduce the recorded content digests; if they do, the original
    record -- including its original timing block -- stays on disk untouched.

    ``run_root`` is required, not optional: a shard that is not registered in the
    run's completion manifest is invisible to every later verification, so its
    container could be deleted and the audit would still report the output as
    produced.
    """
    directory = Path(directory)
    existing_path = directory / f"{name}{SHARD_RECORD}"
    registered = read_completion_manifest(run_root).get("artifacts") or {}
    if any(key in registered for key in shard_logicals(logical_prefix, name)):
        # A deleted completion marker is corruption, not permission to overwrite
        # the remaining completed container as if it were an interrupted first write.
        require_registered_shard(run_root, directory, name, logical_prefix=logical_prefix)
    if existing_path.is_file():
        require_registered_shard(run_root, directory, name, logical_prefix=logical_prefix)
        # Verify the shard that is actually on disk -- container included -- before
        # comparing it to the recomputed arrays. Comparing the new arrays only to the
        # JSON would let a corrupt or truncated .npz survive a "verified" rerun.
        on_disk, existing = read_shard(directory, name)
        recorded = existing.get("arrays") or {}
        mismatched = sorted(key for key in set(recorded) | set(arrays)
                            if key not in recorded or key not in arrays
                            or recorded[key]["content_sha256"] != array_digest(arrays[key]))
        require(not mismatched,
                f"{existing_path} is a completed shard whose arrays {mismatched} no longer "
                "reproduce. A different result goes to a new revision directory; a completed "
                "artifact is not mutated.")
        require(sorted(on_disk) == sorted(arrays),
                f"{existing_path}: the container holds {sorted(on_disk)} but this run produced "
                f"{sorted(arrays)}")
        require(scientific_projection(existing) == scientific_projection(
                    dict(document, schema_version=AUDIT_SCHEMA, shard=name, order=order,
                         arrays=recorded, container=existing.get("container"),
                         status="completed")),
                f"{existing_path} is a completed shard recorded under different scientific "
                "content; choose a new revision directory")
        return dict(existing, rerun="verified_without_rewrite")
    container = directory / f"{name}.npz"
    file_sha256 = write_arrays(container, arrays)
    reloaded = read_arrays(container)
    records = {}
    for key in sorted(arrays):
        require(key in reloaded, f"{container}: array {key!r} did not survive the write")
        require(array_digest(reloaded[key]) == array_digest(arrays[key]),
                f"{container}: array {key!r} changed across the write")
        records[key] = array_record(arrays[key], logical=f"{logical_prefix}/{name}.npz",
                                    order=order)
    record = dict(document, schema_version=AUDIT_SCHEMA, shard=name, order=order,
                  arrays=records, container={"file": f"{logical_prefix}/{name}.npz",
                                             "sha256": file_sha256,
                                             "note": ("npz container bytes carry archive "
                                                      "timestamps; the per-array content digests "
                                                      "above are the scientific identity")},
                  status="completed")
    write_json(existing_path, record)
    register_shard(run_root, directory, name, record, logical_prefix=logical_prefix)
    return record


def read_shard(directory, name, *, require_container_hash=True):
    """Load a shard only if every recorded claim about it still holds.

    Content digests alone are not enough. An edited record whose declared dtype,
    shape, order or container hash no longer describes the archive is a changed
    artifact even when the numbers inside happen to agree, so each of those is
    re-derived here and compared. The container hash is reported separately from
    the array contents because ``numpy.savez`` stamps archive timestamps into the
    zip: equal arrays in a rewritten container are a *changed file*, which this
    function refuses rather than silently accepting.
    """
    directory = Path(directory)
    record_path = directory / f"{name}{SHARD_RECORD}"
    require(record_path.is_file(),
            f"{record_path} is absent: {name} has no completion record, so the .npz beside it is "
            "a partial write and is not a completed shard")
    record = read_json(record_path)
    require(record.get("schema_version") == AUDIT_SCHEMA,
            f"{record_path} is not an {AUDIT_SCHEMA} shard record")
    require(record.get("shard") == name,
            f"{record_path} records shard {record.get('shard')!r}, not {name!r}")
    require(record.get("status") == "completed", f"{record_path} is not marked completed")
    container_path = directory / f"{name}.npz"
    require(container_path.is_file(), f"{container_path} is absent but its record claims completed")
    container = record.get("container") or {}
    if require_container_hash:
        observed = sha256_file(container_path)
        require(observed == container.get("sha256"),
                f"{name}: the container hashes {observed}, the completion record recorded "
                f"{container.get('sha256')}. The recorded file changed; a rewritten archive with "
                "equal arrays is still a different file and is reported, not accepted.")
    arrays = read_arrays(container_path)
    recorded = record.get("arrays") or {}
    require(sorted(recorded) == sorted(arrays),
            f"{name}: the container holds {sorted(arrays)} but the record declares "
            f"{sorted(recorded)}; an extra or missing array is a different artifact")
    for key, entry in sorted(recorded.items()):
        values = np.asarray(arrays[key])
        require(str(values.dtype) == entry.get("dtype"),
                f"{name}: array {key!r} is {values.dtype}, the record declares {entry.get('dtype')}")
        require(list(values.shape) == list(entry.get("shape") or []),
                f"{name}: array {key!r} has shape {list(values.shape)}, the record declares "
                f"{entry.get('shape')}")
        require(entry.get("order") == record.get("order", entry.get("order")),
                f"{name}: array {key!r} declares row order {entry.get('order')!r}, the shard "
                f"declares {record.get('order')!r}")
        require(array_digest(values) == entry.get("content_sha256"),
                f"{name}: array {key!r} no longer matches its recorded content digest")
    return arrays, record


def shard_is_complete(directory, name):
    return (Path(directory) / f"{name}{SHARD_RECORD}").is_file()


# ---------------------------------------------------------------------------
# the completion manifest: the saved authority a rerun is checked against
# ---------------------------------------------------------------------------

#: Grows as stages complete. It is the authority a later verification compares to,
#: which is the thing a file compared against *itself* can never be.
COMPLETION_MANIFEST = "completion_manifest.json"


def completion_manifest_path(run_root):
    return Path(run_root) / COMPLETION_MANIFEST


def read_completion_manifest(run_root):
    path = completion_manifest_path(run_root)
    if not path.is_file():
        return {"schema_version": AUDIT_SCHEMA, "record_kind": "completion_manifest",
                "artifacts": {}}
    document = read_json(path)
    require(document.get("record_kind") == "completion_manifest",
            f"{path} is not a completion manifest")
    return document


def record_completion(run_root, logical, path, *, kind, scientific_digest, timings=None):
    """Bind one finished artifact's bytes, science and timings into the manifest.

    On a rerun the artifact is *checked against this record*, not against a fresh
    re-serialization of itself: a file compared to its own canonical form proves
    only that it is canonical, and a timing block compared to the same block read a
    moment later proves nothing at all. A completed entry is never overwritten.
    """
    path = Path(path)
    require(path.is_file(), f"{path} cannot be recorded complete: it does not exist")
    document = read_completion_manifest(run_root)
    artifacts = document.setdefault("artifacts", {})
    observed = sha256_file(path)
    entry = {"logical": logical, "kind": kind, "sha256": observed,
             "scientific_sha256": scientific_digest,
             "timings": None if timings is None else dict(timings),
             "recorded_at": utc_now()}
    previous = artifacts.get(logical)
    if previous is not None:
        require(previous["scientific_sha256"] == scientific_digest,
                f"{logical} was completed earlier with different scientific content "
                f"({previous['scientific_sha256']} vs {scientific_digest}); a changed result goes "
                "to a new revision directory")
        require(previous["sha256"] == observed,
                f"{logical} was completed earlier as {previous['sha256']} but the file on disk is "
                f"now {observed}; the original bytes, including the original timing block, are "
                "the record")
        return previous
    artifacts[logical] = entry
    document.update(schema_version=AUDIT_SCHEMA, record_kind="completion_manifest",
                    updated_at=utc_now(),
                    note=("every completed artifact, bound to its bytes, its scientific "
                          "projection and the timings measured when it was produced"))
    write_json(completion_manifest_path(run_root), document)
    return entry


def missing_expected_outputs(run_root, expected, *, resolve):
    """Declared outputs that are absent from the ledger or from disk.

    ``expected`` is the list of logical names a *summary* claims were produced. A
    scan of the surviving completion records cannot produce this list -- it can only
    enumerate what is still there -- so an output whose record and container were
    both deleted disappears from the scan and from the problem list with it.
    """
    artifacts = read_completion_manifest(run_root).get("artifacts") or {}
    problems = []
    for logical in sorted(set(expected)):
        entry = artifacts.get(logical)
        if entry is None:
            problems.append({"artifact": logical,
                             "problem": ("a summary claims this output but the completion "
                                         "manifest never recorded it")})
            continue
        path = resolve(logical)
        if path is None or not Path(path).is_file():
            problems.append({"artifact": logical,
                             "problem": ("recorded complete and claimed by a summary, but the "
                                         "file is gone from the run directory")})
    return problems


def verify_completions(run_root, *, resolve, expected=()):
    """Re-check every manifest entry against the saved authority. Returns problems."""
    document = read_completion_manifest(run_root)
    checked, problems = [], list(missing_expected_outputs(run_root, expected, resolve=resolve))
    for logical, entry in sorted((document.get("artifacts") or {}).items()):
        path = resolve(logical)
        if path is None or not Path(path).is_file():
            problems.append({"artifact": logical,
                             "problem": "recorded complete but absent from the run directory"})
            continue
        observed = sha256_file(path)
        row = {"artifact": logical, "kind": entry.get("kind"),
               "sha256_matches": observed == entry["sha256"],
               "recorded_sha256": entry["sha256"], "observed_sha256": observed,
               "timings": entry.get("timings")}
        if not row["sha256_matches"]:
            problems.append({"artifact": logical,
                             "problem": ("bytes differ from the completion manifest; the original "
                                         "timings and content are the record")})
        checked.append(row)
    return {"artifacts_checked": len(checked), "artifacts": checked, "problems": problems,
            "expected_checked": sorted(set(expected))}


# ---------------------------------------------------------------------------
# host paths never reach tracked evidence
# ---------------------------------------------------------------------------

_HOST_PATH = re.compile(r"(?:[A-Za-z]:[\\/]|\\\\|/(?:home|Users|mnt|media)/)[^\s'\"]*")


def scrub_host_paths(node, *, replacement="[host path removed]"):
    """Replace anything that looks like a machine-local path, recursively.

    Applied to every tracked document, including nested status reasons and error
    strings, which is where a resolved ``C:\\Users\\...`` or ``F:\\...`` actually
    reaches published evidence: not through a field somebody designed, but through
    an exception message that got recorded verbatim.
    """
    if isinstance(node, dict):
        return {key: scrub_host_paths(value, replacement=replacement)
                for key, value in node.items()}
    if isinstance(node, list):
        return [scrub_host_paths(value, replacement=replacement) for value in node]
    if isinstance(node, str):
        return _HOST_PATH.sub(replacement, node)
    return node


def host_path_leaks(node, prefix=""):
    """Every ``(json path, text)`` in ``node`` that still carries a host path."""
    found = []
    if isinstance(node, dict):
        for key, value in node.items():
            found.extend(host_path_leaks(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(node, list):
        for position, value in enumerate(node):
            found.extend(host_path_leaks(value, f"{prefix}[{position}]"))
    elif isinstance(node, str) and _HOST_PATH.search(node):
        found.append({"field": prefix, "text": node})
    return found


# ---------------------------------------------------------------------------
# root disjointness
# ---------------------------------------------------------------------------

def _real(path):
    """Fully resolved location, junctions followed. Only used for *containment*."""
    return os.path.normcase(os.path.realpath(str(path)))


def overlapping(first, second):
    """True when two local directories are the same, or one contains the other."""
    a, b = _real(first), _real(second)
    return a == b or a.startswith(b + os.sep) or b.startswith(a + os.sep)


def require_disjoint_run_root(run_root, protected):
    """Refuse a run root that is, contains or sits inside a protected location.

    Checked *before* the first write and against fully resolved paths, because the
    guarded campaign root is a junction: a run root pointed at the junction's target
    would be lexically unrelated to it and would still write into the historical
    campaign.
    """
    conflicts = []
    for name, candidate in sorted((protected or {}).items()):
        if candidate in (None, ""):
            continue
        if not Path(candidate).exists():
            continue
        if overlapping(run_root, candidate):
            conflicts.append({"protected": name, "resolved_conflict": True})
    require(not conflicts,
            f"The run root {Path(run_root).name!r} overlaps protected locations "
            f"{[entry['protected'] for entry in conflicts]} once junctions are resolved. The audit "
            "writes only into its own new directory; historical campaigns, raw inputs and archives "
            "are read-only.")
    return True


# ---------------------------------------------------------------------------
# roots
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResolvedRoot:
    """One campaign root, the candidate that won, and the probes that decided it."""

    name: str
    logical: str
    local_path: Path
    probes: tuple
    candidates: tuple

    def path(self, logical):
        return resolve_under(self.local_path, logical)

    def document(self):
        """Scientific identity only. The local path is deliberately absent."""
        return {"name": self.name, "logical_root": self.logical,
                "probes": [dict(probe) for probe in self.probes],
                "acceptance": "every probe file re-hashed to its published digest"}

    def local_document(self):
        """Machine-local mapping, for the ignored ``local_roots.json`` half."""
        return {"name": self.name, "logical_root": self.logical,
                "local_path": str(self.local_path),
                "candidates_tried": [str(candidate) for candidate in self.candidates]}


def probe_root(candidate, probes):
    """Re-hash every probe under ``candidate``; return ``(ok, results)``."""
    results, ok = [], True
    for logical, expected in sorted(probes.items()):
        entry = {"logical": logical, "expected_sha256": expected}
        try:
            target = resolve_under(candidate, logical)
        except ValueError as error:
            entry.update(observed_sha256=None, matched=False, reason=str(error))
            results.append(entry)
            ok = False
            continue
        if not target.is_file():
            entry.update(observed_sha256=None, matched=False, reason="absent under this candidate")
            results.append(entry)
            ok = False
            continue
        observed = sha256_file(target)
        entry.update(observed_sha256=observed, matched=observed == expected)
        if not entry["matched"]:
            entry["reason"] = "content differs from the published digest"
            ok = False
        results.append(entry)
    return ok, tuple(results)


def resolve_root(name, *, logical, candidates, probes):
    """Accept the first candidate whose every probe re-hashes to its published digest.

    ``probes`` maps a logical suffix to the sha256 a *published, tracked* manifest
    recorded for it. A candidate that merely exists is rejected, and so is one whose
    probe files exist with different bytes: the point of the probe is to prove that
    this directory holds the campaign the audit claims to be reading.
    """
    require(probes, f"{name}: a root cannot be accepted without at least one hash probe")
    attempts = []
    for candidate in candidates:
        if candidate is None:
            continue
        ok, results = probe_root(candidate, probes)
        attempts.append({"candidate": str(candidate), "accepted": ok,
                         "probes": [dict(entry) for entry in results]})
        if ok:
            return ResolvedRoot(name=name, logical=logical, local_path=Path(candidate),
                                probes=results, candidates=tuple(attempts))
    detail = "; ".join(f"{entry['candidate']}: "
                       f"{[p.get('reason') for p in entry['probes'] if not p['matched']]}"
                       for entry in attempts) or "no candidate was supplied"
    raise ValueError(
        f"No candidate root for {name!r} ({logical}) passed its hash probes. Tried {detail}. "
        "Record the machine-local mapping in the ignored local_roots.json; this is a missing or "
        "different artifact, not a reason to skip verification.")


def root_candidates(*, repository_root, logical, explicit=None, local_roots=None, extra=()):
    """Candidate local directories for one logical root, in the order they are tried.

    An explicit CLI value wins, then the ignored local mapping, then anything the
    historical campaign status recorded, then the logical location inside the
    checkout. Nothing here decides correctness; :func:`resolve_root` does.
    """
    ordered = []
    for candidate in (explicit, (local_roots or {}).get(logical), *extra,
                      Path(repository_root) / logical):
        if candidate in (None, ""):
            continue
        resolved = Path(candidate)
        if resolved not in ordered:
            ordered.append(resolved)
    return tuple(ordered)


def load_local_roots(run_root):
    """The ignored ``{logical_root: local_path}`` mapping, or an empty mapping."""
    path = Path(run_root) / LOCAL_ROOTS_FILE
    if not path.is_file():
        return {}
    document = read_json(path)
    roots = document.get("roots") if isinstance(document, dict) else None
    require(isinstance(roots, dict), f"{path} must carry an object under 'roots'")
    return {str(key): str(value) for key, value in roots.items()}


def write_local_roots(run_root, mapping):
    return write_json(Path(run_root) / LOCAL_ROOTS_FILE, {
        "schema_version": AUDIT_SCHEMA,
        "note": ("machine-local mapping from logical roots to this box's directories. Ignored by "
                 "Git and never referenced by a scientific field."),
        "roots": {str(key): str(value) for key, value in sorted(mapping.items())}})


# ---------------------------------------------------------------------------
# run directory, progress, timings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunPaths:
    """The audit's own output tree, addressed by logical name."""

    repository_root: Path
    run_root: Path
    logical_run_root: str

    @classmethod
    def create(cls, repository_root, run_root=None, *, logical=AUDIT_RUN_ROOT, create=True):
        """Bind the run directory. ``create=False`` binds without touching the disk.

        The disjointness guard has to run before the first write, and it needs the
        resolved historical roots, which are read from inside this directory. So the
        object is built first, checked, and only then asked to :meth:`ensure`.
        """
        repository_root = Path(repository_root)
        run_root = Path(run_root) if run_root is not None else repository_root / logical
        paths = cls(repository_root=repository_root, run_root=run_root, logical_run_root=logical)
        if create:
            paths.ensure()
        return paths

    def ensure(self):
        self.run_root.mkdir(parents=True, exist_ok=True)
        return self

    def path(self, logical):
        return resolve_under(self.run_root, logical)

    def logical(self, logical):
        require_logical_name(logical)
        return f"{self.logical_run_root}/{logical}"

    def repository(self, logical):
        return resolve_under(self.repository_root, logical)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


#: Every status a stage may report. ``interrupted`` and ``failed`` are distinct:
#: one is a process that died, the other is a check that refused.
STAGE_STATUSES = ("pending", "running", "completed", "failed", "interrupted")


class StageProgress:
    """Per-stage progress, heartbeat and truthful terminal status.

    ``total`` is ``None`` whenever the denominator is genuinely unknown. The one
    thing this class must not do is invent one: a progress bar that reads 40% of a
    made-up total is worse than a count, because it looks like knowledge.
    """

    def __init__(self, path, *, stage, total=None, note=None, every=1):
        require(total is None or (isinstance(total, int) and total > 0),
                "A stage total is a positive integer or None when it is not known")
        self.path = Path(path)
        self.stage = str(stage)
        self.total = total
        self.note = note
        self.every = max(1, int(every))
        self.completed = 0
        self.current = None
        self.status = "pending"
        self.error = None
        self.started_at = None
        self.started = None

    def _write(self):
        elapsed = None if self.started is None else time.perf_counter() - self.started
        write_json(self.path, {
            "schema_version": AUDIT_SCHEMA, "record_kind": "stage_progress",
            "stage": self.stage, "status": self.status, "total": self.total,
            "completed": self.completed, "current": self.current,
            "started_at": self.started_at, "heartbeat_at": utc_now(),
            "elapsed_wall_seconds": elapsed, "error": self.error, "note": self.note,
            "total_note": (None if self.total is not None else
                           "this stage cannot know its denominator in advance; the count is the "
                           "whole claim")})

    def start(self, total=None):
        if total is not None:
            require(isinstance(total, int) and total > 0, "A supplied total must be positive")
            self.total = total
        self.status = "running"
        self.started_at = utc_now()
        self.started = time.perf_counter()
        self._write()
        return self

    def advance(self, current=None, *, completed=None):
        self.completed = self.completed + 1 if completed is None else int(completed)
        self.current = current
        if self.completed % self.every == 0 or self.completed == self.total:
            self._write()
        return self.completed

    def heartbeat(self, current=None):
        if current is not None:
            self.current = current
        self._write()

    def finish(self, status="completed", *, error=None):
        require(status in STAGE_STATUSES, f"Unknown stage status {status!r}")
        self.status = status
        self.error = error
        self.current = None
        self._write()
        return self.status

    @contextmanager
    def guard(self):
        """Run a stage body; a raise records ``failed``, a kill leaves ``running``.

        A ``KeyboardInterrupt`` or a ``SystemExit`` is recorded as ``interrupted``
        rather than ``failed``: an operator stopping a long scoring pass has not
        found a problem with the audit, and the two must not read the same on the
        dashboard.
        """
        self.start()
        try:
            yield self
        except (KeyboardInterrupt, SystemExit) as stop:
            self.finish("interrupted", error=f"{type(stop).__name__}")
            raise
        except BaseException as error:                     # noqa: BLE001 - recorded, re-raised
            self.finish("failed", error=f"{type(error).__name__}: {error}")
            raise
        else:
            self.finish("completed")


@dataclass
class StageClock:
    """Separately measured inference, I/O and analysis seconds inside one wall clock.

    The three are reported apart because they answer different questions and
    because their sum is not the wall time: model loading, imports and the Python
    that glues the stage together sit in the remainder, and folding that into
    "inference" would overstate what the GPU cost.
    """

    started: float = field(default_factory=time.perf_counter)
    seconds: dict = field(default_factory=lambda: {"inference": 0.0, "io": 0.0, "analysis": 0.0})

    @contextmanager
    def segment(self, kind):
        require(kind in self.seconds, f"Unknown timing segment {kind!r}")
        start = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[kind] += time.perf_counter() - start

    def charge(self, kind, seconds):
        require(kind in self.seconds, f"Unknown timing segment {kind!r}")
        require(seconds >= 0, "Cannot charge a negative duration")
        self.seconds[kind] += float(seconds)

    @property
    def wall_seconds(self):
        return time.perf_counter() - self.started

    def document(self):
        wall = self.wall_seconds
        measured = sum(self.seconds.values())
        return {"inference_seconds": self.seconds["inference"], "io_seconds": self.seconds["io"],
                "analysis_seconds": self.seconds["analysis"], "wall_seconds": wall,
                "unattributed_seconds": max(0.0, wall - measured),
                "note": ("measured segments are disjoint; the remainder is model construction, "
                         "imports and glue, and is reported rather than folded into inference")}
