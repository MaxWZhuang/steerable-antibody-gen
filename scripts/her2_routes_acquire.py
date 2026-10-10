#!/usr/bin/env python
"""Acquire and verify the pinned external inputs, and record their digests.

Two modes, and the separation matters:

``--verify`` (default) reads the acquisition manifest Codex produced, re-hashes
every file it names against the bytes on disk, and writes the external-source table
the freeze binds. **No digest is ever hard-coded in this repository**: a missing or
mismatched one blocks the freeze rather than degrading to a warning.

``--fetch`` downloads the public releases the manifest pins, at the pinned
revisions, into the artifact root. It refuses to overwrite a file whose recorded
digest already matches, and it refuses to write anywhere outside the artifact root.
Nothing is fetched into the repository virtualenv or the user profile.

The HER2 outcome columns are **not** opened here. ``--verify`` reads sequence and
metadata columns only, and it asserts ``outcomes_opened: false`` in the manifest
before the split stage is allowed to freeze.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

# An admission flight must not write into a source tree it imports from: bytecode would land in
# a ``__pycache__`` that the run-root scan and the source-cache check do not cover. Set before
# any project import (tests/test_her2_routes_entry_scripts.py enforces it for every entry script).
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smallAntibodyGen.experiments import her2_routes_env as env  # noqa: E402
if __name__ == "__main__":
    env.bootstrap_cli(ROOT, sys.argv[1:])
from smallAntibodyGen.experiments import her2_routes_spec as spec  # noqa: E402
from smallAntibodyGen.experiments import her2_support_paths as paths  # noqa: E402
from smallAntibodyGen.experiments.her2_runtime import require, sha256  # noqa: E402

DEFAULT_CONFIG = "configs/experiments/her2_generative_routes.json"


def build_parser():
    parser = argparse.ArgumentParser(prog="her2_routes_acquire.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--verify", action="store_true", default=True)
    parser.add_argument("--fetch", action="store_true",
                        help="download the pinned releases named by the manifest")
    parser.add_argument("--report", default=None, help="write the verification report here")
    return parser


def fetch_missing(root, manifest):
    """Download any manifest file that is absent or whose digest does not match."""
    fetched, skipped, problems = [], [], []
    for entry in list(manifest["files"]) + list(manifest.get("extracted_weights") or []):
        target = paths.resolve_under(root, str(entry["path"]))
        if target.is_file() and sha256(target) == entry["sha256"]:
            skipped.append(entry["path"])
            continue
        url = entry.get("url") or entry.get("source_url")
        if not url:
            problems.append(f"{entry['path']}: absent or changed, and the manifest records no URL "
                            "for it (an extracted weight is produced by the extraction step, not "
                            "downloaded)")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".part")
        with urllib.request.urlopen(url) as response, temporary.open("wb") as handle:
            handle.write(response.read())
        observed = sha256(temporary)
        if observed != entry["sha256"]:
            temporary.unlink(missing_ok=True)
            problems.append(f"{entry['path']}: downloaded bytes hash {observed}, the manifest "
                            f"pins {entry['sha256']}")
            continue
        temporary.replace(target)
        fetched.append(entry["path"])
    return {"fetched": fetched, "already_present": skipped, "problems": problems}


def main(argv=None):
    args = build_parser().parse_args(argv)
    config_path = ROOT / args.config
    document = spec.load_config(config_path)
    artifact_root = Path(args.artifact_root or document["artifacts"]["root"])
    ordering = env.prepare_environment(artifact_root)
    env.require_environment_ordering(ordering, label="acquire")
    context = spec.resolve_context(ROOT, config_path=config_path, artifact_root=artifact_root)

    manifest, record = spec.read_acquisition_manifest(context)
    fetch = None
    if args.fetch:
        fetch = fetch_missing(artifact_root, manifest)
        require(not fetch["problems"],
                "Acquisition problems: " + "; ".join(fetch["problems"]))
    logicals = list(document["artifacts"]["external_sources"])
    digests = spec.external_source_digests(context, manifest, logical_paths=logicals)
    external = spec.write_external_marker(context, manifest_record=record, digests=digests)
    report = {"acquisition": record, "fetch": fetch,
              "external_sources": {"count": external["file_count"],
                                   "external_sha256": external["external_sha256"]},
              "outcomes_opened": record["outcomes_opened"],
              "rule": ("every digest comes from the manifest and is re-hashed against disk. A "
                       "missing or mismatched digest blocks the freeze; nothing is hard-coded in "
                       "this repository.")}
    if args.report:
        paths.write_json(Path(args.report), report)
    print(paths.canonical_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
