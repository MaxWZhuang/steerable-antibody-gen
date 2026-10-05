"""Pinned DPLM weights, configs and vocabulary: verify, record, and (only on request) fetch.

Torch-free on purpose, so the acquisition manifest, the verify command and the tests can
run without importing a model. Nothing in this module runs as a side effect of building a
model: :func:`verify_variant_files` only reads, :func:`build_manifest` only describes
files that are already on disk, and :func:`fetch_variant` is a separate, explicit call
that needs a destination root and never overwrites a file.

What is pinned, and where it came from
--------------------------------------
``configs/experiments/dplm_pins.json`` carries, per variant, the Hugging Face repository,
the **commit** the bytes were read at, and for each file its size plus at least one
digest:

* ``pytorch_model.bin`` -- the LFS ``sha256`` the hub API reports;
* ``config.json`` -- the sha256 of the pinned bytes and the hub's git blob id;
* ``vocab.txt`` -- the git blob id (the two variants list the SAME blob id, which is how
  "same bytes" is established) and, where recorded, its sha256.

A digest of ``null`` is refused: it exists in this module only as a guard against a
corrupted pins file, never as a state a production run may proceed in.

The weights ship as a pickle-only ``pytorch_model.bin`` (no safetensors), so the loader
in ``models/dplm_policy.py`` reads them with ``torch.load(weights_only=True)`` after the
digest below has matched.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import her2_support_paths as paths
from .her2_runtime import require

PINS_SCHEMA = "her2-routes-dplm-pins/1"
MANIFEST_SCHEMA = "her2-routes-acquisition/dplm-1"
FILES = ("config.json", "vocab.txt", "pytorch_model.bin")
DIGEST_KEYS = ("sha256", "git_blob_sha1")
#: Verified-file cache: ``{(resolved path, size, mtime_ns): digests}``. Hashing a 2.6 GB
#: checkpoint on every ``build_route`` would dominate a profile that builds the route
#: several times; a changed size or mtime invalidates the entry.
_VERIFIED = {}


def load_pins(path):
    """Read and shape-check the pins file. Returns the document."""
    path = Path(path)
    require(path.is_file(), f"The DPLM pins file {path} is missing")
    document = json.loads(path.read_text(encoding="utf-8"))
    return validate_pins(document, label=str(path))


def validate_pins(document, *, label="pins"):
    require(isinstance(document, dict) and document.get("schema") == PINS_SCHEMA,
            f"{label} is not a {PINS_SCHEMA} document")
    variants = document.get("variants")
    require(isinstance(variants, dict) and variants, f"{label} declares no variants")
    for name, entry in variants.items():
        for key in ("repo", "revision", "logical_dir", "files", "architecture"):
            require(key in entry, f"{label}: variant {name!r} is missing {key!r}")
        require(len(str(entry["revision"])) == 40,
                f"{label}: variant {name!r} revision {entry['revision']!r} is not a full commit id")
        require(set(entry["files"]) == set(FILES),
                f"{label}: variant {name!r} must pin exactly {FILES}, got {sorted(entry['files'])}")
        for file_name, block in entry["files"].items():
            require(isinstance(block.get("bytes"), int) and block["bytes"] > 0,
                    f"{label}: {name}/{file_name} has no byte size")
            present = [key for key in DIGEST_KEYS if block.get(key)]
            require(present,
                    f"{label}: {name}/{file_name} pins no digest. A null placeholder is not a "
                    "pin and production refuses to proceed on one.")
            for key in DIGEST_KEYS:
                if key in block:
                    require(isinstance(block[key], str) and len(block[key]) in (40, 64),
                            f"{label}: {name}/{file_name}.{key} is not a hex digest")
    return document


def pins_digest(document):
    """The digest of the pins document. Folded into the training identity."""
    return paths.digest_document(document)


def variant_entry(document, variant):
    variants = document["variants"]
    require(variant in variants,
            f"Unknown DPLM variant {variant!r}; the pins file declares {sorted(variants)}")
    return variants[variant]


def git_blob_sha1(path):
    """The git blob id of a file: ``sha1('blob <size>\\0' + bytes)``."""
    path = Path(path)
    digest = hashlib.sha1()
    digest.update(f"blob {path.stat().st_size}\0".encode("ascii"))
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _observe(path, block):
    """Digests of one file, for the digest kinds the pin declares. Cached by stat."""
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    cached = _VERIFIED.setdefault(key, {})
    if "sha256" in block and "sha256" not in cached:
        cached["sha256"] = paths.sha256_file(path)
    if "git_blob_sha1" in block and "git_blob_sha1" not in cached:
        cached["git_blob_sha1"] = git_blob_sha1(path)
    return {name: cached[name] for name in DIGEST_KEYS if name in block}


def verification_report(directory, entry):
    """Compare the files under ``directory`` with the pins. Read-only, never raises.

    ``ok`` is true only when every pinned file is present, has the pinned size and matches
    every digest the pin declares. Each file row carries the observed digests, so a pin
    that has to be corrected is one command away.
    """
    directory = Path(directory)
    rows, problems = [], []
    for name in FILES:
        block = entry["files"][name]
        target = directory / name
        row = {"file": name, "path": str(target), "expected_bytes": int(block["bytes"]),
               "present": target.is_file()}
        if not row["present"]:
            problems.append(f"{name}: absent under {directory}")
            rows.append(row)
            continue
        row["observed_bytes"] = target.stat().st_size
        if row["observed_bytes"] != int(block["bytes"]):
            problems.append(f"{name}: {row['observed_bytes']} bytes, pinned {block['bytes']}")
            rows.append(row)
            continue
        observed = _observe(target, block)
        row["observed"] = observed
        for kind, value in observed.items():
            if value != block[kind]:
                problems.append(f"{name}: {kind} {value}, pinned {block[kind]}")
        rows.append(row)
    return {"directory": str(directory), "revision": entry["revision"], "repo": entry["repo"],
            "files": rows, "problems": problems, "ok": not problems}


def verify_variant_files(directory, entry):
    """Like :func:`verification_report` but refusing. Used by the loader."""
    report = verification_report(directory, entry)
    require(report["ok"],
            f"DPLM files under {directory} do not match the pins for {entry['repo']}@"
            f"{entry['revision']}: {report['problems']}. Nothing is downloaded or repaired here; "
            "see docs/DPLM-usage.md for the acquisition command.")
    return report


def build_manifest(artifact_root, document, variants, *, retrieved_utc=None):
    """An acquisition manifest for the DPLM files that are on disk and match the pins.

    The shape is the one ``her2_routes_spec.read_all_manifests`` already merges:
    ``files`` entries with a logical ``path``, ``sha256`` and ``bytes``. Every file's
    sha256 is computed from the disk bytes (a file whose pin is a git blob id only is
    hashed too), so the freeze compares the same kind of digest for every file.
    """
    root = Path(artifact_root)
    files, pins_block = [], {}
    for variant in variants:
        entry = variant_entry(document, variant)
        directory = root / entry["logical_dir"]
        verify_variant_files(directory, entry)
        for name in FILES:
            target = directory / name
            files.append({"path": f"{entry['logical_dir']}/{name}",
                          "sha256": paths.sha256_file(target),
                          "bytes": int(target.stat().st_size),
                          "repo": entry["repo"], "revision": entry["revision"]})
        pins_block[variant] = {"repo": entry["repo"], "revision": entry["revision"]}
    require(files, "No DPLM variant was named, so there is nothing to describe")
    return {"schema": MANIFEST_SCHEMA, "files": sorted(files, key=lambda row: row["path"]),
            "extracted_weights": [], "pins": pins_block,
            "pins_file_sha256": pins_digest(document),
            "retrieved_utc": retrieved_utc or paths.utc_now(), "outcomes_opened": False,
            "failures": [],
            "note": ("describes files already on disk that matched dplm_pins.json. The bytes were "
                     "placed by an explicit acquisition step; this manifest does not download.")}


def fetch_url(entry, name):
    return f"https://huggingface.co/{entry['repo']}/resolve/{entry['revision']}/{name}"


def fetch_variant(destination_root, entry, *, opener=None, chunk=1 << 20):
    """Download one variant's three files into ``destination_root / logical_dir``.

    Explicit and never implicit: the caller names the destination, an existing file is
    never overwritten, each file is streamed to a ``.part`` name and renamed only after
    its size and digests match the pin, and a mismatch deletes the partial and raises.
    ``opener`` is injectable so the tests exercise this without a network.
    """
    import urllib.request
    opener = opener or urllib.request.urlopen
    directory = Path(destination_root) / entry["logical_dir"]
    directory.mkdir(parents=True, exist_ok=True)
    done = []
    for name in FILES:
        block = entry["files"][name]
        target = directory / name
        require(not target.exists(),
                f"{target} already exists. A fetch never overwrites; verify it with "
                "scripts/her2_routes_verify_dplm.py or remove it deliberately.")
        partial = directory / (name + ".part")
        sha256, sha1 = hashlib.sha256(), hashlib.sha1()
        size = 0
        sha1.update(f"blob {int(block['bytes'])}\0".encode("ascii"))
        with opener(fetch_url(entry, name)) as response, partial.open("wb") as stream:
            for piece in iter(lambda: response.read(chunk), b""):
                stream.write(piece)
                sha256.update(piece)
                sha1.update(piece)
                size += len(piece)
        observed = {"sha256": sha256.hexdigest(), "git_blob_sha1": sha1.hexdigest()}
        problems = []
        if size != int(block["bytes"]):
            problems.append(f"{size} bytes, pinned {block['bytes']}")
        for kind in DIGEST_KEYS:
            if kind in block and observed[kind] != block[kind]:
                problems.append(f"{kind} {observed[kind]}, pinned {block[kind]}")
        if problems:
            partial.unlink(missing_ok=True)
            require(False, f"{name} from {fetch_url(entry, name)} did not match the pin: "
                           f"{problems}. The partial file was removed.")
        partial.replace(target)
        done.append({"file": name, "path": str(target), "bytes": size, **observed})
    return {"repo": entry["repo"], "revision": entry["revision"], "files": done}
