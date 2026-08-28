#!/usr/bin/env python
"""Machine-readable inventory of the local corpora and checkpoints — the Gate-0 asset list.

Why this exists
---------------
"Which corpora and which checkpoints does this machine actually have?" is a
question that gets answered by hand, in prose, once — and is wrong by the next
week. A hand-maintained list cannot tell you that ``oas_all.jsonl.gz`` under
``oas_5k`` is a different byte sequence than it was when a checkpoint trained on
it, and it cannot tell you which stage a ``.pt`` file came out of. This script
answers both from the files themselves.

Three properties make the output usable as evidence rather than as a snapshot:

1. **Byte-identical on rerun.** Sorted walk, sorted keys, no timestamps, no
   absolute paths, no machine identifiers. Two runs over unchanged inputs
   produce the same bytes, so ``diff`` is a meaningful integrity check.
2. **Machine-independent paths.** Every path in the JSON is relative to a root
   the caller *labelled*; the absolute prefix that located it never appears. The
   same corpus checked out at a different absolute path inventories identically.
3. **Bounded memory.** Hashes stream in chunks and checkpoint metadata is read
   out of the ``.pt`` archive's pickle stream *without* materializing tensors, so
   inventorying a 500 MB corpus or a 40 MB checkpoint costs one chunk of RAM.

Usage::

    python scripts/inventory_training_assets.py \\
      --data-root data/processed --checkpoint-root checkpoints \\
      --output-json outputs/training-inventory.json

    # An absolute root needs an explicit label, because an absolute path is
    # exactly the machine-specific prefix the JSON must not contain:
    python scripts/inventory_training_assets.py \\
      --data-root corpora=/mnt/scratch/processed \\
      --output-json outputs/training-inventory.json
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import pickle
import sys
import zipfile
from collections import OrderedDict
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping, Sequence

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# `is_absolute_on_any_platform` lives with the fingerprint code because both
# need the same host-independent answer; defining it twice is how the two
# copies drift.
from smallAntibodyGen import experiment  # noqa: E402

# A fixed literal, never a generated value: the schema version is part of the
# contract, not part of the run.
SCHEMA_VERSION = "training-inventory/1"

HASH_CHUNK_BYTES = 1 << 20
CHECKPOINT_SUFFIX = ".pt"
METRICS_FILENAME = "metrics.jsonl"
CORPUS_SUFFIXES = (".jsonl.gz", ".jsonl")
MANIFEST_SUFFIX = ".json"

# Finder droppings differ between machines and carry no training content, and a
# ``*.tmp`` file is an atomic write caught mid-flight rather than an artifact.
# Both are excluded so the inventory describes the assets, not the workstation.
IGNORED_FILENAMES = frozenset({".DS_Store"})
IGNORED_SUFFIXES = (".tmp",)


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def iter_file_chunks(path: Path, chunk_bytes: int = HASH_CHUNK_BYTES) -> Iterator[bytes]:
    """
    Yield the file's bytes in ``chunk_bytes``-sized blocks, last block short.

    Split out from :func:`sha256_file` so the chunking is directly observable in
    a test: "does this stream or does it slurp" is the property that keeps a
    500 MB corpus from becoming 500 MB of resident memory, and it is not visible
    from a digest alone.
    """
    if chunk_bytes <= 0:
        raise ValueError(f"chunk_bytes must be > 0, got {chunk_bytes}")
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                return
            yield block


def sha256_file(path: Path, chunk_bytes: int = HASH_CHUNK_BYTES) -> str:
    """Hex SHA-256 of the file's contents, streamed."""
    digest = hashlib.sha256()
    for block in iter_file_chunks(path, chunk_bytes):
        digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Walking and path normalization
# ---------------------------------------------------------------------------


def iter_files(root: Path) -> Iterator[Path]:
    """
    Yield every inventoried file under ``root``, in a filesystem-independent order.

    ``Path.iterdir`` returns entries in whatever order the directory happens to
    store them, which differs between machines and between two checkouts on one
    machine. Sorting at every level makes the traversal a function of the names
    alone. Symlinked directories are skipped: their targets are machine-specific
    and they can close a cycle that the recursion cannot see.
    """
    if not root.is_dir():
        return
    for entry in sorted(root.iterdir(), key=lambda item: item.name):
        if entry.name in IGNORED_FILENAMES:
            continue
        if entry.is_dir():
            if entry.is_symlink():
                continue
            yield from iter_files(entry)
        elif entry.is_file():
            if entry.name.endswith(IGNORED_SUFFIXES):
                continue
            yield entry


def logical_path(path: Path, root: Path) -> str:
    """
    The path as recorded in the JSON: relative to ``root``, POSIX separators.

    "Logical" means *what the asset is called inside its root*, not where the
    root happens to live. ``/Users/x/repo/data/processed/oas_5k/oas_all.jsonl.gz``
    under root ``data/processed`` is ``oas_5k/oas_all.jsonl.gz`` here and on any
    other machine holding the same corpus, which is what makes two inventories
    comparable at all.
    """
    return path.relative_to(root).as_posix()


def is_corpus_file(path: Path) -> bool:
    return path.name.endswith(CORPUS_SUFFIXES)


def is_manifest_file(path: Path) -> bool:
    return path.name.endswith(MANIFEST_SUFFIX) and not is_corpus_file(path)


def stats_manifest_candidates(path: Path) -> list[Path]:
    """
    The places a corpus file's stats manifest is allowed to live, in priority order.

    Both layouts this repo actually uses are covered, plus the two in-directory
    forms, because ``prepare_oas.py``'s ``--stats-output`` is caller-chosen and
    has landed in more than one place:

    - ``oas_5k/oas_all.jsonl.gz`` -> ``oas_5k/oas_all.stats.json`` (in-dir sibling)
    - ``oas_5k/oas_all.jsonl.gz`` -> ``oas_5k/stats.json`` (in-dir)
    - ``oas_5k/oas_all.jsonl.gz`` -> ``oas_5k.stats.json`` (sibling of the dir)
    - ``oas_paired_all/oas_all.jsonl.gz`` -> ``oas_paired_all.json`` (sibling, no infix)
    """
    stem = path.name
    for suffix in CORPUS_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    parent = path.parent
    return [
        parent / f"{stem}.stats.json",
        parent / "stats.json",
        parent.parent / f"{parent.name}.stats.json",
        parent.parent / f"{parent.name}{MANIFEST_SUFFIX}",
    ]


def find_stats_manifest(path: Path, root: Path) -> Path | None:
    """First existing manifest candidate for ``path`` that is still inside ``root``."""
    for candidate in stats_manifest_candidates(path):
        try:
            candidate.relative_to(root)
        except ValueError:
            # A candidate that escapes the root would put an un-labelled,
            # machine-specific path into the JSON. Refuse it.
            continue
        if candidate.is_file():
            return candidate
    return None


# ---------------------------------------------------------------------------
# JSON sanitizing
# ---------------------------------------------------------------------------


def _type_name(value: Any) -> str:
    if isinstance(value, _OpaqueObject):
        return value.qualname
    kind = type(value)
    return f"{kind.__module__}.{kind.__qualname__}"


def json_safe(value: Any) -> Any:
    """
    Coerce a value read out of a checkpoint into something strict JSON accepts.

    Two hazards, both real in saved training configs. ``NaN``/``Infinity`` are
    accepted by ``json.dumps`` by default but are *not* JSON, so they are mapped
    to ``null`` the way ``mlm_train.py`` already does for ``metrics.jsonl``.
    Anything that is not a JSON primitive becomes an explicit
    ``{"__unserializable__": "<type>"}`` marker rather than disappearing: a
    field that was there is worth recording even when its value is not
    representable.
    """
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return {"__unserializable__": _type_name(value)}


# ---------------------------------------------------------------------------
# Checkpoint metadata, read without materializing tensors
# ---------------------------------------------------------------------------


class _OpaqueObject:
    """
    Stand-in for a pickled value the metadata reader deliberately did not rebuild.

    Tolerates the mutation opcodes (``BUILD``, ``SETITEM``, ``APPEND``) so a
    tensor or an optimizer state entry can be *skipped* rather than
    reconstructed, while the surrounding dict structure still loads.
    """

    __slots__ = ("qualname",)

    def __init__(self, qualname: str) -> None:
        self.qualname = qualname

    def __setstate__(self, state: Any) -> None:
        return None

    def __setitem__(self, key: Any, value: Any) -> None:
        return None

    def append(self, value: Any) -> None:
        return None

    def extend(self, values: Any) -> None:
        return None

    def __repr__(self) -> str:
        return f"<opaque {self.qualname}>"


# Rebuilt for real, because the checkpoint's own structure is made of these.
_SAFE_CLASSES: dict[tuple[str, str], Any] = {
    ("collections", "OrderedDict"): OrderedDict,
    ("builtins", "dict"): dict,
    ("builtins", "list"): list,
    ("builtins", "set"): set,
    ("builtins", "frozenset"): frozenset,
    ("builtins", "tuple"): tuple,
    ("builtins", "bytearray"): bytearray,
    ("builtins", "complex"): complex,
}

_opaque_factories: dict[str, Any] = {}


def _opaque_factory(qualname: str) -> Any:
    """A callable that satisfies both the ``REDUCE`` and ``NEWOBJ`` opcodes."""
    factory = _opaque_factories.get(qualname)
    if factory is None:

        class _OpaqueFactory:
            def __new__(cls, *args: Any, **kwargs: Any) -> _OpaqueObject:
                return _OpaqueObject(qualname)

        factory = _OpaqueFactory
        _opaque_factories[qualname] = factory
    return factory


class _MetadataUnpickler(pickle.Unpickler):
    """
    Loads a ``.pt`` payload's structure while refusing to import or build anything.

    This is what makes "read the metadata without loading tensors" true rather
    than aspirational. ``persistent_load`` is where ``torch.load`` would go read
    the storage blobs out of the archive; returning a placeholder instead means
    the tensor bytes are never touched, so a 40 MB checkpoint costs the size of
    its pickle stream. It is also strictly safer than
    ``torch.load(weights_only=False)``, which unpickles arbitrary classes: every
    global outside :data:`_SAFE_CLASSES` resolves to an inert placeholder rather
    than to an imported object.
    """

    def find_class(self, module: str, name: str) -> Any:
        safe = _SAFE_CLASSES.get((module, name))
        if safe is not None:
            return safe
        return _opaque_factory(f"{module}.{name}")

    def persistent_load(self, pid: Any) -> Any:
        return _OpaqueObject("torch.storage")


def _find_pickle_member(names: Sequence[str]) -> str:
    """The shallowest ``data.pkl`` in a ``torch.save`` zip archive."""
    candidates = [name for name in names if name == "data.pkl" or name.endswith("/data.pkl")]
    if not candidates:
        raise ValueError("archive contains no data.pkl member")
    return min(candidates, key=lambda name: (name.count("/"), name))


def read_zip_payload(path: Path) -> Mapping[str, Any]:
    """Read a ``torch.save`` archive's top-level payload, tensor storages excluded."""
    with zipfile.ZipFile(path) as archive:
        raw = archive.read(_find_pickle_member(archive.namelist()))
    payload = _MetadataUnpickler(io.BytesIO(raw)).load()
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload is not a mapping")
    return payload


def read_torch_payload(path: Path) -> Mapping[str, Any]:
    """
    Fallback for archives the metadata reader cannot parse (e.g. legacy formats).

    ``torch`` is imported here rather than at module scope on purpose: the
    fallback is rare, and a top-level import would make the whole inventory —
    including the corpus half, which needs nothing but stdlib — unusable and
    untestable anywhere torch is not installed.
    """
    import torch  # noqa: PLC0415 - deliberately lazy; see docstring

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload is not a mapping")
    return payload


def read_checkpoint_payload(path: Path) -> tuple[Mapping[str, Any] | None, str | None, str | None]:
    """
    Return ``(payload, source, error)`` for one checkpoint; never raises.

    A corrupt or truncated ``.pt`` must produce a *record* saying so, not abort
    the inventory: the one moment you most need the asset list is the moment an
    artifact has gone bad, and an inventory that dies on the bad file tells you
    nothing about the other twelve.

    ``error`` carries the exception class names only. The messages are dropped
    deliberately — ``torch`` and ``zipfile`` both embed the absolute path they
    were handed into their exception text, which would put a machine-specific
    prefix straight into the JSON and break rerun-identity across machines.
    """
    try:
        return read_zip_payload(path), "zip_pickle", None
    except Exception as exc:  # noqa: BLE001 - any failure means "try the fallback"
        zip_error = type(exc).__name__
    try:
        return read_torch_payload(path), "torch_load", None
    except Exception as exc:  # noqa: BLE001 - any failure means "record and move on"
        return None, None, f"zip_pickle:{zip_error}; torch_load:{type(exc).__name__}"


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def describe_file(path: Path, root: Path) -> dict[str, Any]:
    """Identity of any inventoried file: what it is called, how big, what it hashes to."""
    return {
        "logical_path": logical_path(path, root),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def describe_corpus(path: Path, root: Path) -> dict[str, Any]:
    """A corpus file plus its adjacent stats manifest, or an explicit ``null``."""
    record = describe_file(path, root)
    manifest = find_stats_manifest(path, root)
    record["stats_manifest"] = None if manifest is None else describe_file(manifest, root)
    return record


def describe_checkpoint(path: Path, root: Path) -> dict[str, Any]:
    """
    A checkpoint file plus the provenance fields needed to place it in the stage chain.

    ``training_stage_source`` exists because ``null`` is ambiguous here. The
    checkpoints on this workstation predate the ``training_stage`` config field,
    so "the config says no stage" and "the config has no such field" are
    different facts about lineage and must not collapse into one ``null``.
    """
    record = describe_file(path, root)
    payload, source, error = read_checkpoint_payload(path)

    raw_config = payload.get("train_config") if payload is not None else None
    config = json_safe(raw_config) if isinstance(raw_config, Mapping) else None

    if config is None:
        stage_source = "unavailable"
    elif "training_stage" in config:
        stage_source = "saved_config"
    else:
        stage_source = "absent_from_saved_config"

    record.update(
        {
            "metadata_source": source,
            "metadata_error": error,
            "epoch": json_safe(payload.get("epoch")) if payload is not None else None,
            "val_loss": json_safe(payload.get("val_loss")) if payload is not None else None,
            "best_val_loss": (
                json_safe(payload.get("best_val_loss")) if payload is not None else None
            ),
            "training_stage": config.get("training_stage") if config else None,
            "training_stage_source": stage_source,
            "parent_checkpoint": config.get("init_checkpoint") if config else None,
            "saved_config": config,
        }
    )
    return record


# ---------------------------------------------------------------------------
# Roots
# ---------------------------------------------------------------------------


def inventory_data_root(label: str, root: Path) -> dict[str, Any]:
    """Inventory one corpus root: corpora, their manifests, and anything else present."""
    corpora: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    other_files: list[dict[str, Any]] = []
    for path in iter_files(root):
        if is_corpus_file(path):
            corpora.append(describe_corpus(path, root))
        elif is_manifest_file(path):
            manifests.append(describe_file(path, root))
        else:
            other_files.append(describe_file(path, root))
    return {
        "label": label,
        "present": root.is_dir(),
        "corpora": sorted(corpora, key=lambda item: item["logical_path"]),
        "manifests": sorted(manifests, key=lambda item: item["logical_path"]),
        "other_files": sorted(other_files, key=lambda item: item["logical_path"]),
    }


def iter_run_directories(root: Path) -> list[Path]:
    """
    The run directories under a checkpoint root, sorted.

    Enumerated from the directory listing rather than inferred from the files
    found, so a run directory holding nothing at all still appears — an empty
    run is a fact about the workstation, and inferring runs from files would
    silently drop it.
    """
    if not root.is_dir():
        return []
    return sorted(
        (
            entry
            for entry in root.iterdir()
            if entry.is_dir()
            and not entry.is_symlink()
            and entry.name not in IGNORED_FILENAMES
        ),
        key=lambda entry: entry.name,
    )


def inventory_run_directory(run_dir: Path, root: Path) -> dict[str, Any]:
    """
    Inventory one run directory: its checkpoints, its other files, its metrics log.

    ``metrics_jsonl`` is always present as a key and is ``null`` when the log is
    missing. Omitting the field would make "this run kept no metrics" and "the
    inventory did not look" indistinguishable in the JSON.
    """
    checkpoints: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    for path in iter_files(run_dir):
        if path.name.endswith(CHECKPOINT_SUFFIX):
            checkpoints.append(describe_checkpoint(path, root))
        elif path.name == METRICS_FILENAME and path.parent == run_dir:
            continue  # recorded under metrics_jsonl below
        else:
            files.append(describe_file(path, root))

    metrics_path = run_dir / METRICS_FILENAME
    return {
        "logical_path": logical_path(run_dir, root),
        "metrics_jsonl": describe_file(metrics_path, root) if metrics_path.is_file() else None,
        "checkpoints": sorted(checkpoints, key=lambda item: item["logical_path"]),
        "files": sorted(files, key=lambda item: item["logical_path"]),
    }


def inventory_checkpoint_root(label: str, root: Path) -> dict[str, Any]:
    """Inventory one checkpoint root, grouped by run directory."""
    runs = [inventory_run_directory(run_dir, root) for run_dir in iter_run_directories(root)]
    root_files = [
        describe_file(path, root) for path in iter_files(root) if path.parent == root
    ]
    return {
        "label": label,
        "present": root.is_dir(),
        "runs": sorted(runs, key=lambda item: item["logical_path"]),
        "root_files": sorted(root_files, key=lambda item: item["logical_path"]),
    }


def build_inventory(
    data_roots: Sequence[tuple[str, Path]],
    checkpoint_roots: Sequence[tuple[str, Path]],
) -> dict[str, Any]:
    """
    Assemble the whole inventory payload.

    Roots keep the caller's order rather than being sorted: the caller chose the
    labels, and re-ordering them would make the JSON depend on something the
    caller did not ask for. Everything *inside* a root is sorted by logical path.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "data_roots": [inventory_data_root(label, root) for label, root in data_roots],
        "checkpoint_roots": [
            inventory_checkpoint_root(label, root) for label, root in checkpoint_roots
        ],
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def render_json(payload: Mapping[str, Any]) -> str:
    """
    Serialize strictly and deterministically.

    ``allow_nan=False`` is the strictness: it turns a non-finite float into a
    loud failure instead of the ``NaN`` literal that no conforming JSON parser
    accepts. ``sort_keys=True`` is the determinism, and it is what lets the
    record-building code use whatever insertion order reads best.
    """
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_json_atomic(payload: Mapping[str, Any], path: Path) -> None:
    """
    Write the inventory via a same-directory temp file plus ``os.replace``.

    Serializing *before* opening the temp file matters: a payload that cannot be
    rendered then fails without having created anything, so a failed run leaves
    neither a stale destination nor a half-written temp file behind. ``os.replace``
    is atomic on POSIX and Windows, so an interrupted run leaves the previous
    inventory intact rather than truncating it — the same discipline
    ``mlm_train.py`` uses for checkpoints.
    """
    text = render_json(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_root_argument(raw: str) -> tuple[str, Path]:
    """
    Parse a ``PATH`` or ``LABEL=PATH`` root specification into ``(label, path)``.

    An absolute path may only be supplied as ``LABEL=PATH``. That is the whole
    mechanism behind "no machine-specific prefixes in the JSON": the label is the
    only root identifier that reaches the output, so an absolute root without a
    label has no name that would survive being moved to another machine, and is
    rejected rather than silently leaked.
    """
    label, separator, location = raw.partition("=")
    if not separator:
        label, location = raw, raw
    if not label:
        raise ValueError(f"empty root label in {raw!r}")
    if not location:
        raise ValueError(f"empty root path in {raw!r}")

    path = Path(location)
    if experiment.is_absolute_on_any_platform(location) and not separator:
        raise ValueError(
            f"absolute root {location!r} must be given as LABEL=PATH so the JSON "
            "records a machine-independent label instead of the absolute prefix"
        )
    normalized = PurePosixPath(label.replace("\\", "/")).as_posix()
    if PurePosixPath(normalized).is_absolute():
        raise ValueError(f"root label {label!r} must not be an absolute path")
    return normalized, path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-root",
        action="append",
        default=None,
        metavar="[LABEL=]PATH",
        help="Corpus root to inventory; repeatable. Absolute paths require LABEL=PATH.",
    )
    parser.add_argument(
        "--checkpoint-root",
        action="append",
        default=None,
        metavar="[LABEL=]PATH",
        help="Checkpoint root to inventory; repeatable. Absolute paths require LABEL=PATH.",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        type=str,
        help="Destination for the inventory JSON (written atomically).",
    )
    return parser


def _parse_roots(parser: argparse.ArgumentParser, raws: Sequence[str] | None, option: str):
    roots: list[tuple[str, Path]] = []
    for raw in raws or ():
        try:
            roots.append(parse_root_argument(raw))
        except ValueError as exc:
            parser.error(f"{option}: {exc}")
    return roots


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    data_roots = _parse_roots(parser, args.data_root, "--data-root")
    checkpoint_roots = _parse_roots(parser, args.checkpoint_root, "--checkpoint-root")

    payload = build_inventory(data_roots, checkpoint_roots)
    output_path = Path(args.output_json)
    write_json_atomic(payload, output_path)

    for root in payload["data_roots"]:
        if not root["present"]:
            print(f"[inventory] data root {root['label']}: MISSING", file=sys.stderr)
            continue
        print(
            f"[inventory] data root {root['label']}: "
            f"{len(root['corpora'])} corpora, {len(root['manifests'])} manifests"
        )
    for root in payload["checkpoint_roots"]:
        if not root["present"]:
            print(f"[inventory] checkpoint root {root['label']}: MISSING", file=sys.stderr)
            continue
        checkpoints = sum(len(run["checkpoints"]) for run in root["runs"])
        without_metrics = [run["logical_path"] for run in root["runs"] if run["metrics_jsonl"] is None]
        print(
            f"[inventory] checkpoint root {root['label']}: "
            f"{len(root['runs'])} runs, {checkpoints} checkpoints, "
            f"{len(without_metrics)} runs without {METRICS_FILENAME}"
        )
    print(f"[inventory] wrote {output_path}")


if __name__ == "__main__":
    main()
