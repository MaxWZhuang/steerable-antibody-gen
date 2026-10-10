#!/usr/bin/env python
"""Qualify ONE native DPLM route on this GPU, in its own new run root.

    .venv/Scripts/python.exe scripts/her2_routes_qualify_dplm.py \\
        --route dplm_150m_masked --revision joint_qual_150m

Runs, for that route only: source snapshot, acquire, preflight (native evidence on the real
model), split, sources, profile (all four tasks, the finite/clip check and the three spill
signals), a six-update smoke with a resume, and a native diffusion decode from both smoke
checkpoints. It writes ``qualification.json`` under
``<artifact root>/runs/<revision>/<run_root_leaf>``.

This is not a training launch, a route selection or a competition stage. ``--revision`` is
required and must start ``joint_qual_``; the root must not exist (or hold only lock files). Every
refusal happens before anything is written and before any model or CUDA work.

Exit status: 0 qualified; 1 failed; 2 refused_by_hardware; 3 refused before any write;
4 cpu_check_only (a code check that never certifies a GPU fit).

``prepare_environment`` runs before any scientific import, exactly as in the flight script. Use
``./.venv/Scripts/python.exe``, never bare ``python``.
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

DEFAULT_CONFIG = "configs/experiments/her2_generative_routes_joint.json"


def bootstrap(argv):
    """Prepare the storage root of the config THIS script will load, before scientific imports."""
    return env.bootstrap_cli(ROOT, argv, default_config=DEFAULT_CONFIG)


if __name__ == "__main__":
    bootstrap(sys.argv[1:])
from smallAntibodyGen.experiments import her2_routes_spec as spec  # noqa: E402

EXIT = {"qualified": 0, "failed": 1, "refused_by_hardware": 2, "cpu_check_only": 4}


def build_parser():
    parser = argparse.ArgumentParser(
        prog="her2_routes_qualify_dplm.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--route", required=True,
                        choices=tuple(route for route in spec.COMPETITION_ROUTES
                                      if route.startswith("dplm_")))
    parser.add_argument("--revision", required=True,
                        help="a NEW revision directory under <artifact root>/runs; must start "
                             "joint_qual_")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="the joint config (default: %(default)s)")
    parser.add_argument("--artifact-root", default=None,
                        help="override the configured artifact root (must be absolute)")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"),
                        help="cuda is the only device that can qualify; cpu is a code check")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config_path = ROOT / args.config
    document = spec.load_config(config_path)
    artifact_root = args.artifact_root or document["artifacts"]["root"]
    ordering = env.prepare_environment(artifact_root)
    env.require_environment_ordering(ordering, label="qualify_dplm")

    from smallAntibodyGen.experiments import her2_routes_campaign as campaign
    from smallAntibodyGen.experiments import her2_routes_qualify as qualify
    try:
        record = qualify.run_qualification(
            ROOT, config_path=config_path, artifact_root=artifact_root, revision=args.revision,
            route=args.route, device=args.device, build_services=campaign.build_services,
            environment=ordering)
    except qualify.QualificationRefused as error:
        print(f"REFUSED before any write: {error}", flush=True)
        return 3
    print(f"verdict: {record['verdict']}  gpu_qualified: {record['gpu_qualified']}  "
          f"route: {record['route']}  revision: {record['revision']}", flush=True)
    if record.get("reason"):
        print(f"reason: {record['reason']}", flush=True)
    for name, block in record.get("phases", {}).items():
        print(f"  {name:<12} {block['status']}", flush=True)
    if record["verdict"] == qualify.VERDICT_CPU_ONLY:
        print("a CPU run is a code check and does not certify a GPU fit.", flush=True)
    return EXIT.get(record["verdict"], 1)


if __name__ == "__main__":
    raise SystemExit(main())
