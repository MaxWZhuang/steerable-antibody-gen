#!/usr/bin/env python
"""The durable supervisor: single writer, journal heartbeat, resumable stages.

``--detach`` re-launches this script as a detached process and returns. Without it
this process *is* the supervisor and holds the lock for the whole writing life of
the run. Background jobs started inside an agent turn die at the turn boundary --
nanosecond-identical kills confirmed a group kill -- so an unattended run must be
detached, not backgrounded.

A PID is not progress. Liveness is the OS lock; the heartbeat is a progress signal;
scientific completion is the required stage list. ``her2_routes_flight.py health``
answers the first two and ``verify`` answers the third.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smallAntibodyGen.experiments import her2_routes_env as env  # noqa: E402
if __name__ == "__main__":
    env.bootstrap_cli(ROOT, sys.argv[1:])
from smallAntibodyGen.experiments import her2_routes_spec as spec  # noqa: E402
from smallAntibodyGen.experiments import her2_support_paths as paths  # noqa: E402

DEFAULT_CONFIG = "configs/experiments/her2_generative_routes.json"


def build_parser():
    parser = argparse.ArgumentParser(prog="her2_routes_supervisor.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--from-stage", default=None)
    parser.add_argument("--through-stage", default=None,
                        help="run through this stage and retain later stages as unfinished")
    parser.add_argument("--heartbeat-seconds", type=float, default=15.0)
    parser.add_argument("--max-restarts", type=int, default=0,
                        help="restarts after a crash. Default 0: a crash is reported, not retried "
                             "into a different trajectory.")
    parser.add_argument("--detach", action="store_true",
                        help="re-launch detached and return immediately")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config_path = ROOT / args.config
    document = spec.load_config(config_path)
    artifact_root = args.artifact_root or document["artifacts"]["root"]
    ordering = env.prepare_environment(artifact_root)
    env.require_environment_ordering(ordering, label="supervisor")

    from smallAntibodyGen.experiments import her2_routes_campaign as campaign
    context = spec.resolve_context(ROOT, config_path=config_path, artifact_root=artifact_root,
                                   revision=args.revision)
    # Refused before a detach, a lock, services or any stage: a declared training phase holds
    # every stage it does not list, including as the bound of --from/--through-stage.
    try:
        stages = campaign.select_stages(context, from_stage=args.from_stage,
                                        through_stage=args.through_stage)
    except ValueError as error:
        print(f"refused: {error}", flush=True)
        return 2
    if args.detach:
        arguments = ["--device", args.device,
                     "--heartbeat-seconds", str(args.heartbeat_seconds),
                     "--max-restarts", str(args.max_restarts)]
        if args.from_stage:
            arguments += ["--from-stage", args.from_stage]
        if args.through_stage:
            arguments += ["--through-stage", args.through_stage]
        record = campaign.launch_detached(context, arguments=arguments)
        print(f"detached pid {record['pid']}; log {record['log']}", flush=True)
        print("verify with: .venv/Scripts/python.exe scripts/her2_routes_flight.py health",
              flush=True)
        return 0

    services = campaign.build_services(context, device=args.device)
    record = campaign.supervise(context, services, stages=stages,
                               heartbeat_seconds=args.heartbeat_seconds,
                               max_restarts=args.max_restarts)
    print(paths.canonical_json({"status": record["status"], "stages": record["stages"]}))
    return 0 if record["status"] in campaign.SUCCESS_STATUSES else 2


if __name__ == "__main__":
    raise SystemExit(main())
