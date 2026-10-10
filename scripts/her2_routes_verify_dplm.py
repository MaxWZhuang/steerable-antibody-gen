#!/usr/bin/env python
"""Verify, describe or (only on request) fetch the pinned DPLM files.

Three separate modes, and the default is the harmless one:

* **verify** (default): hash the local files under ``--root`` against
  ``configs/experiments/dplm_pins.json`` and print every observed digest. Reads only. No
  network, no writes, no environment preparation, no install. Exit status 1 on any
  mismatch or missing file.
* ``--write-manifest PATH``: after a passing verify, write a DPLM acquisition manifest to
  ``PATH`` (in the shape ``her2_routes_spec.read_all_manifests`` merges). This is what binds
  the DPLM bytes into a flight's frozen external identity; adding paths to a config does not.
* ``--fetch``: download the pinned files into ``--root`` from the Hugging Face URL at the
  pinned commit. This is the ONLY mode that touches the network. It never overwrites an
  existing file, streams to a ``.part`` name, checks size and digests before renaming, and
  deletes a partial that does not match.

Model construction never calls any of this. See docs/DPLM-usage.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# An admission flight must not write into a source tree it imports from: bytecode would land in
# a ``__pycache__`` that the run-root scan and the source-cache check do not cover. Set before
# any project import (tests/test_her2_routes_entry_scripts.py enforces it for every entry script).
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smallAntibodyGen.experiments import her2_routes_dplm_pins as pins_lib  # noqa: E402

DEFAULT_PINS = "configs/experiments/dplm_pins.json"


def build_parser():
    parser = argparse.ArgumentParser(prog="her2_routes_verify_dplm.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pins", default=DEFAULT_PINS)
    parser.add_argument("--root", required=True,
                        help="directory that CONTAINS models/dplm_150m and models/dplm_650m "
                             "(the artifact root, or any scratch root for a CPU validation)")
    parser.add_argument("--variant", action="append", default=None,
                        help="dplm_150m and/or dplm_650m (default: both)")
    parser.add_argument("--write-manifest", default=None, metavar="PATH",
                        help="after a passing verify, write an acquisition manifest here")
    parser.add_argument("--fetch", action="store_true",
                        help="EXPLICITLY download the pinned files (network). Never overwrites.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    pins_path = Path(args.pins)
    pins = pins_lib.load_pins(pins_path if pins_path.is_absolute() else ROOT / pins_path)
    variants = args.variant or sorted(pins["variants"])
    root = Path(args.root)
    if args.fetch:
        for variant in variants:
            entry = pins_lib.variant_entry(pins, variant)
            print(f"fetching {entry['repo']}@{entry['revision']} into "
                  f"{root / entry['logical_dir']}", flush=True)
            result = pins_lib.fetch_variant(root, entry)
            for row in result["files"]:
                print(f"  {row['file']}: {row['bytes']} bytes sha256={row['sha256']}", flush=True)
    failed = False
    for variant in variants:
        entry = pins_lib.variant_entry(pins, variant)
        report = pins_lib.verification_report(root / entry["logical_dir"], entry)
        print(f"== {variant}  {entry['repo']}@{entry['revision']}  ok={report['ok']}")
        for row in report["files"]:
            print(f"  {row['file']:<18} present={row['present']} "
                  f"bytes={row.get('observed_bytes')}/{row['expected_bytes']} "
                  f"observed={row.get('observed')}")
        for problem in report["problems"]:
            print(f"  PROBLEM: {problem}")
        failed = failed or not report["ok"]
    if args.write_manifest:
        if failed:
            print("not writing a manifest: verification failed", file=sys.stderr)
            return 1
        manifest = pins_lib.build_manifest(root, pins, variants)
        target = Path(args.write_manifest)
        if target.exists():
            print(f"{target} already exists; a manifest is never overwritten", file=sys.stderr)
            return 1
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {target}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
