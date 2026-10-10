"""Checkpoint admission: score a route that a different, finished flight trained.

Why this exists. A checkpoint is bound to the run root that wrote it: its stored identity carries
that root's config, source-snapshot and external-table digests, and ``require_identity`` demands
exact equality with the root that reads it. That is the right rule for resuming training and a
wall for importing a finished route into a new flight. This module is the narrow, auditable way
through the wall, and it is opt-in: nothing here is reachable from a config that does not carry
the code-held admission phase (``her2_routes_spec.admission_phase``).

What is deliberately NOT done.

* **No aggregate override.** Only three identity fields may come from the origin flight
  (:data:`OVERRIDDEN_IDENTITY_FIELDS`). Route, arm, seed, condition vocabulary, split, schedule
  and native base stay live-checked against the destination on every load.
* **The external table is compared file by file**, not replaced: every file the origin froze must
  be present with the same hash in the destination's table (a DPLM asset may be added). A pinned
  digest of the whole table proves nothing about the tokenizer or release files a route reads.
* **A record is not authority.** ``admission.json`` is evidence and is self-digested, but an
  editor can recompute a digest. The authority is the pin in the committed config; the record
  must agree with it, and the file's SHA-256 and step are re-checked against the pin on EVERY
  load, so replacing ``best.pt`` with another checkpoint from the same training run (same
  identity, different step) is refused.
* **Route science is bound by content, not by the config digest.** ``native_identity`` is
  ``None`` for the three original routes, so ``config_sha256`` is the only identity field that
  covers their route blocks, and it is the one field overridden. The route block (and the
  ``her2`` / ``dplm`` blocks that exist) is therefore digested with prose stripped and compared
  with the pin on every load.
* **The origin flight is only read.**

The guarantee, and its limits. Through the supported admission flight entry points
(``scripts/her2_routes_*.py``), the artifact writes the flight makes into two destinations are
guarded against overlap with the pinned origin flights and against pre-existing symbolic links,
junctions or other reparse points, and files with more than one hard link
(:func:`tree_problems`). The two are checked at DIFFERENT points:

* the destination RUN ROOT is scanned in ``resolve_context`` before the lock, its owner journal or
  any marker is written, again in ``_require_admission_ready`` before every guarded stage and at
  the top of ``run_bakeoff``; every path built through ``RoutesContext.path`` is also checked
  against the pins at the moment it is built;
* the shared SOURCE CACHE (``FlightServices.source_cache_root``) is checked lazily, each time that
  accessor hands the location to a reader, so before the cache is first read or written, and not
  before the lock or at each stage boundary.

Every atomic writer creates its temporary file exclusively (``her2_support_paths.atomic_open``), so
a link at a temporary name is never written through. The entry scripts disable Python bytecode
generation before any project import (direct imports of the package, such as a test harness, are
outside that statement). This protection assumes the filesystem topology does not change between a
check and the write that follows it: a link created by accidental concurrent tooling while a stage
runs defeats it as surely as a deliberate one.

This is NOT process-wide write isolation. Runtime setup (``her2_routes_env.prepare`` points
``TEMP``/``TMP`` at ``<artifact_root>/tmp`` and the Hugging Face cache variables at
``<artifact_root>/cache/*``), the dependency, compiler and GPU caches (constructing a torch
optimiser initialises a compiler-cache directory), interpreter start-up and OS-generated files are
outside the checked destinations. Operators remain responsible for keeping those locations free of
aliases into a finished flight; they were found clean on the production artifact root when this was
written (2,502 files under ``tmp``, 25 under ``cache``, none a link or multiply linked), a
point-in-time observation and not a guard. Guarding them further was considered and declined: any
list of ambient locations (CUDA kernel caches, crash dumps, OS error reports) stays incomplete, so it
would not establish process-wide isolation, only a longer list of named exceptions to it.

Torch-free on purpose: the payload's stored identity is compared by the services, which already
own the checkpoint loader. Everything here can be exercised on plain dictionaries and small files.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path

from . import her2_support_paths as paths
from .her2_runtime import require

ADMISSION_SCHEMA = "her2-generative-routes/admission/1"
ADMISSION_FILE = "admission.json"

#: Keys that carry prose, not science. Stripped (at every depth) before a block is digested, so a
#: reworded ``why`` or ``note`` is not a drift and a changed ``updates`` or ``seed`` is.
NOTE_KEYS = ("why", "compute_note", "note")
#: Config blocks digested beside the route block, when the config has them.
BLOCK_NAMES = ("her2", "dplm")
#: The ONLY identity fields an admitted checkpoint may take from its origin flight.
OVERRIDDEN_IDENTITY_FIELDS = ("config_sha256", "external_sha256", "snapshot_sha256")
#: A destination external table may add these (the DPLM pinned assets) and nothing else.
EXTERNAL_ALLOWED_ADDITIONS = ("models/dplm_150m/", "models/dplm_650m/")

PIN_KEYS = ("origin_flight", "origin_config", "origin_config_sha256", "origin_snapshot_sha256",
            "origin_external_sha256", "arm", "seed", "schedule_sha256", "checkpoint_sha256",
            "checkpoint_bytes", "checkpoint_step", "block_digests", "selector")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_ORIGIN_MARKERS = {"snapshot": "source_snapshot.json", "external": "external_sources.json",
                   "vocabulary": "condition_vocabulary.json", "split": "hard_split_freeze.json"}


# ---------------------------------------------------------------------------
# digests
# ---------------------------------------------------------------------------

def _strip_notes(value):
    if isinstance(value, dict):
        return {key: _strip_notes(item) for key, item in value.items() if key not in NOTE_KEYS}
    if isinstance(value, list):
        return [_strip_notes(item) for item in value]
    return value


def block_digest(block):
    """SHA-256 of a config block's canonical JSON with prose keys removed at every depth."""
    return paths.sha256_text(paths.canonical_json(_strip_notes(block)))


def config_block_digests(config, route):
    """``{"route", "her2", "dplm"}`` digests for the blocks of ``config`` that exist."""
    routes = (config or {}).get("routes") or {}
    require(isinstance(routes.get(route), dict), f"the config has no routes.{route} block")
    digests = {"route": block_digest(routes[route])}
    for name in BLOCK_NAMES:
        if isinstance((config or {}).get(name), dict):
            digests[name] = block_digest(config[name])
    return digests


def external_aggregate(files):
    """The digest a frozen external table claims: ``sha256`` over sorted ``"<path> <sha256>"`` lines.

    The same rule ``her2_routes_spec.verify_external_sources`` applies. An origin's marker carries
    both the per-file table and this aggregate; the pin binds the aggregate, so the table must be
    recomputed to it before any file in it is believed.
    """
    return paths.sha256_text("\n".join(f"{name} {files[name]}" for name in sorted(files)))


def require_disjoint(run_root, pins):
    """Refuse a run root that is, sits inside, or contains a pinned origin flight.

    Admission reads the origin and writes only into the destination. A destination equal to an
    origin would make ``copy_verified`` accept the origin's own file and then write
    ``admission.json`` into a finished flight; one inside it, or above it, is no better.
    """
    root = Path(run_root).resolve()
    for route, pin in sorted(dict(pins).items()):
        origin = Path(pin["origin_flight"]).resolve()
        overlap = root == origin or origin in root.parents or root in origin.parents
        require(not overlap,
                f"the run root {root} overlaps the pinned origin flight of {route} ({origin}); "
                "admission reads a finished flight and never writes into, under or above it")


#: ``st_file_attributes`` bit set on symbolic links, junctions and every other reparse point.
REPARSE_POINT = 0x400


def _is_redirect(info):
    """A symbolic link, a Windows junction or any other reparse point, from an ``lstat`` result."""
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & REPARSE_POINT)


def tree_problems(root, *, label="the run root", limit=25):
    """Every entry under ``root`` that could carry a write somewhere else. Read-only.

    Every way found so far of sending an admission flight's write into a finished flight needs
    one of three things to exist INSIDE a tree the flight writes to: a junction or symbolic link
    (a redirected directory such as ``stages`` or ``source_snapshot``, or a redirected file), or a
    file with more than one hard link (a second name for an origin's bytes, opened for append by
    the lock's owner journal or for writing by any writer that reuses a name). Checking the
    destination of each write one site at a time kept leaving the next site open; this checks
    the precondition once, wholesale, before the flight touches anything and again before each
    guarded stage. A fresh tree contains none of them: checkpoints are copied, not linked.

    The trees are the run root and the shared source cache under the artifact root (the one
    place a flight writes outside its run root: ``FlightServices.source_cache_root``). The walk
    never follows a link, so a redirected directory is reported and not traversed. An absent
    tree has no problems. Residual window, stated: an entry created AFTER this check needs write
    access to the tree while a stage runs, which is the process's own trust boundary.
    """
    root = Path(root)
    try:
        info = os.lstat(root)
    except FileNotFoundError:
        return []
    if _is_redirect(info):
        return [f"{label} itself is a junction or symbolic link: {root}"]
    problems, stack = [], [root]
    while stack and len(problems) < limit:
        directory = stack.pop()
        with os.scandir(directory) as entries:
            found = sorted(entries, key=lambda entry: entry.name)
        for entry in found:
            info = os.lstat(entry.path)
            if _is_redirect(info):
                problems.append(f"a junction or symbolic link: {entry.path}")
            elif stat.S_ISDIR(info.st_mode):
                stack.append(Path(entry.path))
            elif stat.S_ISREG(info.st_mode) and info.st_nlink > 1:
                problems.append(f"a file with {info.st_nlink} hard links: {entry.path}")
    return problems[:limit]


def run_root_problems(run_root, *, limit=25):
    """:func:`tree_problems` for the run root."""
    return tree_problems(run_root, label="the run root", limit=limit)


def require_clean_tree(root, *, label):
    """Refuse a tree that holds a link, a junction or a multiply-linked file."""
    problems = tree_problems(root, label=label)
    require(not problems,
            f"{label} holds an entry that could redirect a write into another tree, so an "
            "admission flight will not use it (it reads finished flights and never writes into "
            "one): " + "; ".join(problems))


def require_clean_run_root(run_root):
    """Refuse a run root that holds a link, a junction or a multiply-linked file."""
    require_clean_tree(run_root, label="the run root")


class DigestCache:
    """File SHA-256, computed from the bytes on disk EVERY time. It memoises nothing.

    An earlier version memoised by ``(path, size, mtime_ns)`` to avoid re-reading 700 MB on each of
    the several loads a stage makes. A replacement of the same length whose modification time is
    restored (``touch -r``, an archive extractor, a tool that preserves timestamps) then kept the old
    digest and was accepted: neither the identity nor the record comparison looks at tensor values.
    A hash is a few seconds for the largest checkpoint, which is nothing next to the scoring it
    guards, so correctness wins. The class name is kept so callers and tests keep one seam.
    Residual window, stated: a writer that swaps the file between this hash and the loader's read
    needs write access to the destination run root during the stage, which is the same trust
    boundary as the process itself.
    """

    def sha256(self, path):
        return paths.sha256_file(Path(path))


DIGESTS = DigestCache()


# ---------------------------------------------------------------------------
# the pin schema
# ---------------------------------------------------------------------------

def _hex(value, label, *, nullable=False):
    if nullable and value is None:
        return
    require(isinstance(value, str) and _HEX64.fullmatch(value),
            f"{label} must be 64 hex characters (lowercase), found {value!r}")


def _positive_int(value, label):
    require(type(value) is int and value > 0, f"{label} must be a positive int, found {value!r}")


def parse_pins(block, *, routes):
    """The validated pin mapping ``route -> pin``. A closed schema; any other shape is refused.

    The pins live in the committed config, so they are part of the run's config digest and every
    stage identity. They are NOT read from the record the admission writes.
    """
    require(isinstance(block, dict) and block,
            f"the admission pins must be a nonempty mapping, found {block!r}")
    require(set(block) == set(routes),
            f"the admission pins' route set {sorted(block)} differs from the admitted routes "
            f"{sorted(routes)}")
    parsed = {}
    for route, pin in block.items():
        require(isinstance(pin, dict), f"admission pin for {route} is not a mapping")
        missing, extra = sorted(set(PIN_KEYS) - set(pin)), sorted(set(pin) - set(PIN_KEYS))
        require(not missing, f"admission pin for {route} is missing {missing}")
        require(not extra, f"admission pin for {route} has unexpected keys {extra}")
        for key in ("origin_config_sha256", "origin_snapshot_sha256", "origin_external_sha256",
                    "checkpoint_sha256"):
            _hex(pin[key], f"{route}.{key}")
        _hex(pin["schedule_sha256"], f"{route}.schedule_sha256", nullable=True)
        require(type(pin["seed"]) is int, f"{route}.seed must be an int, found {pin['seed']!r}")
        _positive_int(pin["checkpoint_step"], f"{route}.checkpoint_step")
        _positive_int(pin["checkpoint_bytes"], f"{route}.checkpoint_bytes")
        for key in ("arm", "selector", "origin_config", "origin_flight"):
            require(isinstance(pin[key], str) and pin[key],
                    f"{route}.{key} must be a nonempty string")
        require(Path(pin["origin_flight"]).is_absolute(),
                f"{route}.origin_flight must be an absolute path, found {pin['origin_flight']!r}")
        logical = Path(pin["origin_config"])
        require(not logical.is_absolute() and ".." not in logical.parts,
                f"{route}.origin_config must be a relative path inside the origin's snapshot")
        digests = pin["block_digests"]
        require(isinstance(digests, dict), f"{route}.block_digests must be a mapping")
        unknown = sorted(set(digests) - {"route", *BLOCK_NAMES})
        require(not unknown, f"{route}.block_digests has unexpected names {unknown}")
        for needed in ("route", "her2"):
            require(needed in digests, f"{route}.block_digests must include {needed!r}")
        for name, value in digests.items():
            _hex(value, f"{route}.block_digests.{name}")
        parsed[route] = dict(pin)
    return parsed


def pin_digest(pin):
    return paths.digest_document(dict(pin))


def pinned_identity_fields(pin):
    """The three identity fields that come from the origin flight. Nothing else may."""
    return {"config_sha256": pin["origin_config_sha256"],
            "snapshot_sha256": pin["origin_snapshot_sha256"],
            "external_sha256": pin["origin_external_sha256"]}


# ---------------------------------------------------------------------------
# reading an origin flight (read-only) and comparing external tables
# ---------------------------------------------------------------------------

def origin_checkpoint_path(pin, route):
    return Path(pin["origin_flight"]) / "bakeoff" / route / "best.pt"


def _read_marker(root, key):
    path = Path(root) / _ORIGIN_MARKERS[key]
    require(path.is_file(), f"the origin flight has no {_ORIGIN_MARKERS[key]} ({path})")
    return path


def external_problems(origin_files, new_files, *, allowed=EXTERNAL_ALLOWED_ADDITIONS):
    """Why ``new_files`` is not a compatible superset of ``origin_files``; ``[]`` if it is.

    Both are ``{logical path: sha256}``. Every origin file must be present and byte-identical.
    The only additions allowed are the pinned DPLM assets. An empty origin table is a problem:
    it would make every destination trivially compatible.
    """
    if not origin_files:
        return ["external table: the origin's table is empty, which proves nothing"]
    problems = []
    for name in sorted(origin_files):
        if name not in new_files:
            problems.append(f"external table: {name} is missing from the destination")
        elif new_files[name] != origin_files[name]:
            problems.append(f"external table: {name} changed "
                            f"({str(origin_files[name])[:12]} -> {str(new_files[name])[:12]})")
    for name in sorted(set(new_files) - set(origin_files)):
        if not any(name.startswith(prefix) for prefix in allowed):
            problems.append(f"external table: {name} is an unexpected addition")
    return problems


def derive_pin(origin_root, *, route, origin_config, checkpoint, checkpoint_step, arm, seed,
               schedule_sha256, selector, live_config):
    """Build a pin from the finished origin flight's own files. Refuses one that would not verify.

    ``origin_config`` is relative to the origin's ``source_snapshot`` directory: the frozen copy
    the flight actually ran, not whatever the repository's file of the same name says today.
    """
    root = Path(origin_root)
    config_path = root / "source_snapshot" / origin_config
    require(config_path.is_file(), f"{config_path} does not exist")
    origin = json.loads(config_path.read_text(encoding="utf-8"))
    origin_digests = config_block_digests(origin, route)
    live_digests = config_block_digests(live_config, route)
    for name, digest in origin_digests.items():
        require(live_digests.get(name) == digest,
                f"the {name} block of the live config does not equal the origin's "
                f"(prose excluded); this checkpoint would be scored under another architecture")
    checkpoint = Path(checkpoint)
    pin = {"origin_flight": str(root.resolve()), "origin_config": str(origin_config),
           "origin_config_sha256": paths.sha256_file(config_path),
           "origin_snapshot_sha256": paths.read_json(_read_marker(root, "snapshot"))["snapshot_sha256"],
           "origin_external_sha256": paths.read_json(_read_marker(root, "external"))["external_sha256"],
           "arm": str(arm), "seed": int(seed), "schedule_sha256": schedule_sha256,
           "checkpoint_sha256": paths.sha256_file(checkpoint),
           "checkpoint_bytes": int(checkpoint.stat().st_size),
           "checkpoint_step": int(checkpoint_step), "block_digests": origin_digests,
           "selector": str(selector)}
    return pin


def live_values(context):
    """The destination root's own values an origin flight is compared against (read-only).

    ``context`` needs only ``path`` and ``config``. Every marker is required to exist: the
    destination must have run its stages up to ``sources`` (external table, vocabulary, split)
    before anything can be compared, and a missing one is a plain error, not an empty value.
    """
    external = context.path(_ORIGIN_MARKERS["external"])
    vocabulary = context.path(_ORIGIN_MARKERS["vocabulary"])
    split = context.path(_ORIGIN_MARKERS["split"])
    for target in (external, vocabulary, split):
        require(target.is_file(), f"{target} is absent; run the stages before it first")
    return {"config": context.config, "condition_sha256": paths.sha256_file(vocabulary),
            "split_sha256": paths.digest_document(paths.read_json(split)),
            "external_files": dict(paths.read_json(external).get("files") or {})}


def origin_problems(pin, route, *, live):
    """Every reason the origin flight and the live destination are not compatible; ``[]`` if none.

    ``live`` carries the destination's own values: ``config`` (the parsed live config),
    ``condition_sha256``, ``split_sha256`` and ``external_files`` (``{path: sha256}``). The origin
    flight is only read.
    """
    problems = []
    root = Path(pin["origin_flight"])
    if not root.is_dir():
        return [f"origin flight {root} does not exist"]
    config_path = root / "source_snapshot" / pin["origin_config"]
    origin_config = None
    if not config_path.is_file():
        problems.append(f"origin config {config_path} is missing")
    else:
        if paths.sha256_file(config_path) != pin["origin_config_sha256"]:
            problems.append("origin config sha256 differs from the pin")
        origin_config = json.loads(config_path.read_text(encoding="utf-8"))
    markers = {}
    for key in _ORIGIN_MARKERS:
        path = root / _ORIGIN_MARKERS[key]
        if path.is_file():
            markers[key] = path
        else:
            problems.append(f"origin marker {_ORIGIN_MARKERS[key]} is missing")
    if "snapshot" in markers and (paths.read_json(markers["snapshot"]).get("snapshot_sha256")
                                  != pin["origin_snapshot_sha256"]):
        problems.append("origin snapshot digest differs from the pin")
    if "external" in markers:
        external = paths.read_json(markers["external"])
        if external.get("external_sha256") != pin["origin_external_sha256"]:
            problems.append("origin external digest differs from the pin")
        origin_files = external.get("files") or {}
        if external_aggregate(origin_files) != external.get("external_sha256"):
            problems.append("origin external table does not aggregate to its recorded digest; "
                            "its per-file digests cannot be trusted")
        problems.extend(external_problems(origin_files, live.get("external_files") or {}))
    if "vocabulary" in markers and (paths.sha256_file(markers["vocabulary"])
                                    != live.get("condition_sha256")):
        problems.append("the condition vocabulary differs between the origin and the destination")
    if "split" in markers and (paths.digest_document(paths.read_json(markers["split"]))
                               != live.get("split_sha256")):
        problems.append("the frozen split differs between the origin and the destination")
    pinned = pin["block_digests"]
    if origin_config is not None:
        problems.extend(_digest_problems(pinned, config_block_digests(origin_config, route),
                                         where="origin config"))
    problems.extend(_digest_problems(pinned, config_block_digests(live["config"], route),
                                     where="live config"))
    checkpoint = origin_checkpoint_path(pin, route)
    if not checkpoint.is_file():
        problems.append(f"origin checkpoint {checkpoint} is missing")
    else:
        if checkpoint.stat().st_size != pin["checkpoint_bytes"]:
            problems.append("origin checkpoint bytes differ from the pin")
        elif DIGESTS.sha256(checkpoint) != pin["checkpoint_sha256"]:
            problems.append("origin checkpoint sha256 differs from the pin")
    return problems


def _digest_problems(pinned, observed, *, where):
    problems = []
    for name, digest in pinned.items():
        if name not in observed:
            problems.append(f"the {where} has no {name} block, which the pin covers")
        elif observed[name] != digest:
            problems.append(f"the {name} block of the {where} differs from the pin "
                            "(prose excluded)")
    return problems


# ---------------------------------------------------------------------------
# copying, the record, and the check made on every load
# ---------------------------------------------------------------------------

def copy_verified(source, target, *, sha256, nbytes, chunk=1 << 22):
    """Copy ``source`` to ``target`` byte for byte, verified, never overwriting other bytes.

    Written to a temporary name and renamed only when the digest and length match, so a crash or a
    mismatch leaves nothing that could be mistaken for the admitted file. Repeating the call for
    identical bytes is a no-op. The temporary file is created EXCLUSIVELY under a fresh random
    name: a fixed ``<name>.tmp`` opened for writing would follow a hard link or symbolic link
    already planted there and truncate whatever it points at, an origin checkpoint included,
    before any digest is compared.
    """
    source, target = Path(source), Path(target)
    require(source.is_file(), f"{source} does not exist")
    require(source.stat().st_size == int(nbytes),
            f"{source} is {source.stat().st_size} bytes, the pin says {nbytes}")
    if target.exists():
        require(target.stat().st_size == int(nbytes) and paths.sha256_file(target) == sha256,
                f"{target} already exists with different bytes; admission never overwrites")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    digest, written = hashlib.sha256(), 0
    with source.open("rb") as reader, paths.atomic_open(target, "wb") as writer:
        for block in iter(lambda: reader.read(chunk), b""):
            digest.update(block)
            written += len(block)
            writer.write(block)
        # Raising here leaves the context WITHOUT the rename and removes the temporary.
        require(digest.hexdigest() == sha256 and written == int(nbytes),
                f"the copy of {source.name} has sha256 {digest.hexdigest()[:12]}, not the pinned "
                f"{sha256[:12]}")
    return target


def build_record(route, pin, *, origin_root, destination, best):
    """The self-digested record of one admission. Evidence, not authority (see the module note)."""
    destination = Path(destination)
    record = {"schema_version": ADMISSION_SCHEMA, "record_kind": "admission", "route": route,
              "admitted": True, "trained": False, "selected_on": "development", "rule": "best",
              "selector": pin["selector"], "step": int(pin["checkpoint_step"]),
              "best": dict(best), "pin": dict(pin), "pin_sha256": pin_digest(pin),
              "origin_flight": str(origin_root),
              "destination": {"path": str(destination), "bytes": int(destination.stat().st_size),
                              "sha256": paths.sha256_file(destination)},
              "admitted_at": paths.utc_now(),
              "note": ("this route was NOT trained in this flight. Its checkpoint was verified "
                       "against the pin committed in the config and copied byte for byte; every "
                       "number computed from it is selected on the development panel.")}
    record["record_sha256"] = paths.digest_document(record)
    return record


def write_record(path, record):
    """Write the record atomically. ``paths.write_json`` creates its temporary exclusively."""
    paths.write_json(path, record)
    return path


def read_record(path):
    """The record, refused if its own digest does not match what is on disk."""
    document = paths.read_json(path)
    claimed = document.get("record_sha256")
    body = {key: value for key, value in document.items() if key != "record_sha256"}
    require(claimed == paths.digest_document(body),
            f"{path}: record_sha256 does not match the record's contents; it was edited")
    return document


def record_payload_problems(record_path, payload):
    """Why the record's reported ``best`` and ``step`` are not the verified checkpoint's own.

    ``best`` is published by the gate beside every number it computes. It is therefore taken from
    the checkpoint, not trusted from a record whose digest an editor can recompute.
    """
    if not Path(record_path).is_file():
        return [f"{ADMISSION_FILE} is absent"]
    try:
        record = read_record(record_path)
    except ValueError as error:
        return [str(error)]
    problems = []
    best = record.get("best") or {}
    for key in ("best_step", "best_metric"):
        if best.get(key) != payload.get(key):
            problems.append(f"the record's best.{key} is {best.get(key)!r}, the checkpoint's own "
                            f"is {payload.get(key)!r}")
    if record.get("step") != payload.get("step"):
        problems.append(f"the record's step is {record.get('step')!r}, the checkpoint's own is "
                        f"{payload.get('step')!r}")
    return problems


def loadable_problems(route, path, pin, *, live_config, cache=None):
    """Why an admitted checkpoint may not be loaded now; ``[]`` if it may.

    Run on EVERY load. The pin (from the committed config) is the authority: the record must agree
    with it, the file's length and SHA-256 must equal it, and the live route, ``her2`` and ``dplm``
    blocks must digest to the pinned values. The payload's step is compared by the caller, which
    has the payload.
    """
    path = Path(path)
    cache = cache or DIGESTS
    problems = []
    if not path.is_file():
        return [f"{path} is absent"]
    record_path = path.parent / ADMISSION_FILE
    if not record_path.is_file():
        problems.append(f"{ADMISSION_FILE} is absent beside {path.name}")
    else:
        try:
            record = read_record(record_path)
        except ValueError as error:
            problems.append(str(error))
        else:
            if record.get("route") != route:
                problems.append(f"the record is for route {record.get('route')!r}, not {route!r}")
            if record.get("pin_sha256") != pin_digest(pin):
                problems.append("the record's pin digest does not match the pin committed in "
                                "the config")
            if record.get("pin") != dict(pin):
                problems.append("the record's embedded pin differs from the pin committed in "
                                "the config")
    if path.stat().st_size != pin["checkpoint_bytes"]:
        problems.append(f"{path.name} has {path.stat().st_size} bytes, the pin says "
                        f"{pin['checkpoint_bytes']}")
    elif cache.sha256(path) != pin["checkpoint_sha256"]:
        problems.append(f"{path.name} sha256 differs from the pin: this is not the admitted "
                        "checkpoint")
    try:
        observed = config_block_digests(live_config, route)
    except ValueError as error:
        problems.append(str(error))
    else:
        problems.extend(_digest_problems(pin["block_digests"], observed, where="live config"))
    return problems
