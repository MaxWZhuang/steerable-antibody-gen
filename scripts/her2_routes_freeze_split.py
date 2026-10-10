#!/usr/bin/env python
"""Freeze the sequence-only hard split, before any outcome is opened.

Reads sequences and descriptors only. It refuses to run if the acquisition manifest
records ``outcomes_opened: true``, because a split frozen after outcomes were read is
not a reserved population and re-freezing would launder that.

The certificate is printed and written **whether or not** the reserved panel meets
its declared minimum. The power arithmetic is
``SE = (1 - rho^2) / sqrt(n - 1)``: at rho ~ 0.5 the 95% half-width is about 0.148 at
n = 100 and 0.042 at n = 322, against a +0.05 ambition. An underpowered panel is
reported with its actual n and this arithmetic; the separation is never relaxed after
seeing the answer.
"""
from __future__ import annotations

import argparse
import sys
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

DEFAULT_CONFIG = "configs/experiments/her2_generative_routes.json"


def build_parser():
    parser = argparse.ArgumentParser(prog="her2_routes_freeze_split.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="compute and print the certificate without writing the freeze")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config_path = ROOT / args.config
    document = spec.load_config(config_path)
    artifact_root = args.artifact_root or document["artifacts"]["root"]
    ordering = env.prepare_environment(artifact_root)
    env.require_environment_ordering(ordering, label="freeze_split")

    from smallAntibodyGen.experiments import her2_routes_campaign as campaign
    context = spec.resolve_context(ROOT, config_path=config_path, artifact_root=artifact_root,
                                   revision=args.revision)
    services = campaign.build_services(context, device="cpu")
    result = services.freeze_hard_split(persist=not args.dry_run)
    certificate = result["certificate"]
    print(paths.canonical_json({
        "compatible_windows": result["compatible_windows"],
        "unique_windows": result["unique_windows"],
        "rejected": result["rejected"],
        "derived_parent": {"linker_length": len(result["parent"]["linker"]),
                           "light_length": len(result["parent"]["light"]),
                           "shares": result["parent"]["shares"],
                           "distinct_linkers": result["parent"]["distinct_linkers"],
                           "distinct_lights": result["parent"]["distinct_lights"]},
        "old_library": result["old_library"],
        "counts": certificate["counts"], "power": certificate["power"],
        "zero_violations": certificate["zero_violations"],
        "reserved_status": certificate["reserved_status"],
        "selection": result["freeze"]["selection"]}))
    if args.dry_run:
        print("\ndry run: nothing was written", flush=True)
        return 0
    paths.write_json(context.path("split_certificate.json"), certificate)
    paths.write_json(context.path(spec.SPLIT_FREEZE), result["freeze"])
    print(f"\nwrote {context.path(spec.SPLIT_FREEZE)}", flush=True)
    return 0 if certificate["zero_violations"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
