"""Artifact roots, cache redirection and dependency paths -- importable before torch.

This module has no third-party imports on purpose. Every entry point of this
campaign calls :func:`prepare_environment` **first**, because ``transformers``
reads ``HF_HOME`` at import time and ``torch`` reads ``TORCH_HOME`` at import
time: setting them afterwards leaves a cache directory in the user profile that
no manifest describes, and the run still succeeds. A module that imported torch
in order to offer this function could not be imported early enough to do its job.

The artifact root is addressed by **logical name with hash probes**, reusing
``her2_support_paths.resolve_root``, so "this directory holds the acquisition the
config names" is a measurement. A junction is not created and nothing is copied:
the large read-only inputs on the repository volume stay where they are.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from .her2_runtime import require

#: Environment variables redirected under the artifact root's ``cache`` directory.
CACHE_VARIABLES = ("HF_HOME", "TRANSFORMERS_CACHE", "HUGGINGFACE_HUB_CACHE", "TORCH_HOME",
                   "XDG_CACHE_HOME", "HF_HUB_OFFLINE")

#: Subdirectories the campaign writes under the artifact root.
ARTIFACT_SUBDIRECTORIES = ("data", "cache", "tmp", "upstream", "python_deps", "runs", "logs")

#: Logical names inside the artifact root. Nothing else is hard-coded.
NATIVE_PACKAGE_ROOT = "upstream/ablang2"
NATIVE_WEIGHTS_ROOT = "models/ABLANG-2"
WALKJUMP_PACKAGE_ROOT = "upstream/walkjump"
DEPENDENCY_ROOT = "python_deps"
ACQUISITION_MANIFEST = "data/acquisition_manifest.json"
#: The first primary acquisition (the four Petersen supplements).
PRIMARY_ACQUISITION_MANIFEST = "data/primary_acquisition_manifest.json"
#: The review acquisition: Adams replicate tables, the Petersen author parent
#: export, the AlphaSeq primary release and their licences. Separate from the
#: manifest above because it was retrieved under a separate pin, and merging the
#: two would make either one unverifiable on its own.
PRIMARY_REVIEW_MANIFEST = "data/primary_review_acquisition_manifest.json"
DEPENDENCY_MANIFEST = "data/native_dependencies_install.json"
_PREPARED_ROOT = None
#: Roots of the primary (non-curated) sources under the artifact volume.
PRIMARY_ADAMS_ROOT = "data/primary/adams2017"
PRIMARY_PETERSEN_ROOT = "data/primary/petersen2024"
PRIMARY_PETERSEN_PARENTS = ("data/primary/petersen2024_parent_export/data/whitehead/petersen/"
                            "original/AntibodySequences2.csv")
PRIMARY_ALPHASEQ_ROOT = "data/primary/alphaseq"


def prepare_environment(artifact_root, *, dependency_paths=None, offline=True):
    """Redirect every model cache under the artifact root and add the pinned deps.

    ``offline=True`` sets ``HF_HOME``-adjacent variables *and* ``HF_HUB_OFFLINE``,
    so an accidental hub call fails instead of silently downloading bytes nobody
    hashed. This campaign loads only explicitly pinned local weights.
    """
    global _PREPARED_ROOT
    root = Path(artifact_root)
    require(root.is_absolute(),
            f"The artifact root must be an absolute path, got {artifact_root!r}. A relative root "
            "would resolve against the working directory, which differs between the launcher and "
            "the detached supervisor.")
    late = sorted(name for name in ("torch", "transformers") if name in sys.modules)
    already_prepared = _PREPARED_ROOT == str(root)
    cache = root / "cache"
    temporary = root / "tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = os.environ["TMP"] = str(temporary)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    assigned = {}
    for name in CACHE_VARIABLES:
        value = "1" if name == "HF_HUB_OFFLINE" else str(cache / name.lower())
        if name == "HF_HUB_OFFLINE" and not offline:
            continue
        os.environ[name] = value
        assigned[name] = value
    if dependency_paths is None:
        dependency_paths = [root / DEPENDENCY_ROOT, root / NATIVE_PACKAGE_ROOT]
    inserted, absent = [], []
    for candidate in dependency_paths:
        path = str(Path(candidate))
        if not Path(path).is_dir():
            absent.append(path)
            continue
        if path not in sys.path:
            sys.path.insert(0, path)
            inserted.append(path)
    if not late:
        _PREPARED_ROOT = str(root)
    return {"artifact_root_configured": True, "cache_variables": assigned,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "sys_path_inserted": inserted, "declared_paths_absent": absent,
            "imported_before_this_call": [] if already_prepared else late,
            "already_prepared_before_scientific_imports": already_prepared,
            "ordering_rule": ("torch and transformers read their cache variables at import time. "
                              "This call must precede the first import of either; the list above "
                              "names any that were already imported, and preflight refuses to "
                              "continue when it is non-empty."),
            "offline": bool(offline)}


def bootstrap_cli(repository_root, argv, *,
                  default_config="configs/experiments/her2_generative_routes.json"):
    """Configure storage before importing the CLI's scientific dependencies.

    ``default_config`` must be the entry point's OWN ``--config`` default, so that a run with no
    ``--config`` prepares the root of the config it will actually load.
    """
    import argparse
    import json
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--artifact-root")
    args, _ = parser.parse_known_args(argv)
    config = Path(repository_root) / args.config
    document = json.loads(config.read_text(encoding="utf-8-sig"))
    root = args.artifact_root or document["artifacts"]["root"]
    record = prepare_environment(root)
    require_environment_ordering(record, label="CLI bootstrap")
    return record


def require_environment_ordering(record, *, label="preflight"):
    """Refuse to proceed when a cache variable was assigned after the import it governs."""
    late = list(record.get("imported_before_this_call") or [])
    require(not late,
            f"{label}: {late} were imported before the cache variables were assigned, so a cache "
            "directory may already exist outside the artifact root. Move prepare_environment to "
            "the top of the entry point, before any scientific import.")
    absent = list(record.get("declared_paths_absent") or [])
    require(not absent,
            f"{label}: the declared dependency path(s) {absent} do not exist. The pinned native "
            "dependencies live under the artifact root and are not installed into the repository "
            "virtualenv; a missing path means the acquire stage has not run.")
    return True


def home_cache_paths(home=None):
    base = Path(home) if home is not None else Path.home()
    return tuple(base / name for name in (".cache/huggingface", ".cache/torch"))


def home_cache_baseline(*, home=None):
    """A snapshot of the pre-existing profile caches, taken before anything loads.

    Both of this machine's default caches existed long before this campaign and
    belong to unrelated work. The question is not whether they exist -- it is
    whether **this flight** writes into them, so the check needs a baseline to
    compare against.
    """
    record = {}
    for path in home_cache_paths(home):
        if not path.exists():
            record[str(path)] = {"exists": False, "entries": 0, "newest_mtime": None}
            continue
        entries = sorted(child.name for child in path.iterdir())
        newest = max((child.stat().st_mtime for child in path.rglob("*")), default=None)
        record[str(path)] = {"exists": True, "entries": len(entries),
                             "newest_mtime": newest, "top_level": entries[:20]}
    return {"record_kind": "home_cache_baseline", "paths": record,
            "rule": ("existence is not a finding. A pre-existing cache belongs to other work and "
                     "is neither moved nor deleted; what is refused is a NEW write by this "
                     "flight.")}


def assert_no_home_cache(*, home=None, baseline=None, tolerance_seconds=1.0):
    """Refuse to continue if **this flight** wrote into a profile cache.

    The failure this catches is silent: a default download lands in the profile,
    the run succeeds, and the artifact volume's manifest does not describe the
    bytes that were loaded. Refusing merely because an unrelated cache *exists*
    catches nothing and asks the user to delete other people's data; both of this
    machine's caches predate the campaign. So the check is differential, and with
    no baseline it reports what it found rather than failing.
    """
    current = home_cache_baseline(home=home)
    paths = current["paths"]
    if baseline is None:
        return {"home_caches": sorted(name for name, block in paths.items() if block["exists"]),
                "checked": [str(path) for path in home_cache_paths(home)],
                "baseline_supplied": False, "new_writes": [],
                "meaning": ("pre-existing caches are recorded, not refused. Without a baseline "
                            "this call cannot distinguish an old cache from a new write, and it "
                            "says so instead of asserting either.")}
    grown = []
    for name, block in paths.items():
        before = dict(baseline["paths"]).get(name, {"exists": False, "entries": 0,
                                                    "newest_mtime": None})
        if block["exists"] and not before.get("exists"):
            grown.append({"path": name, "problem": "created during this flight"})
            continue
        if not block["exists"]:
            continue
        if int(block["entries"]) > int(before.get("entries") or 0):
            grown.append({"path": name, "problem": "gained top-level entries",
                          "before": before.get("entries"), "after": block["entries"]})
            continue
        newest, previous = block["newest_mtime"], before.get("newest_mtime")
        if newest is not None and previous is not None and \
                newest > float(previous) + float(tolerance_seconds):
            grown.append({"path": name, "problem": "a file was modified during this flight",
                          "before_mtime": previous, "after_mtime": newest})
    require(not grown,
            f"This flight wrote into the user-profile model cache: {grown}. Every weight it loads "
            "is a pinned local file under the artifact root; a default download is the one route "
            "by which unhashed bytes enter the computation. The pre-existing caches are left "
            "alone -- what is refused is the new write.")
    return {"home_caches": sorted(name for name, block in paths.items() if block["exists"]),
            "checked": [str(path) for path in home_cache_paths(home)],
            "baseline_supplied": True, "new_writes": [],
            "meaning": ("no profile cache directory gained an entry or a modified file between "
                        "the baseline and now. Existing unrelated caches are untouched.")}


def artifact_probes(manifest, *, limit=6):
    """``{logical path: sha256}`` hash probes taken from the acquisition manifest.

    The probes are a *selection* of recorded files, sorted for determinism, so
    root resolution is a re-hash of digests the manifest already fixed rather than
    an existence check. ``extracted_weights`` entries are preferred: the weights
    are the artifact a wrong root would most damagingly substitute.
    """
    entries = list(manifest.get("extracted_weights") or []) + list(manifest.get("files") or [])
    require(entries, "The acquisition manifest records no files to probe a root with")
    probes = {}
    for entry in sorted(entries, key=lambda item: str(item.get("path"))):
        path, digest = entry.get("path"), entry.get("sha256")
        if not path or not digest:
            continue
        probes[str(path)] = str(digest)
        if len(probes) >= int(limit):
            break
    require(probes, "No manifest entry carried both a path and a sha256")
    return probes
