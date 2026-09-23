#!/usr/bin/env python
"""Durable supervisor for the HER2 next flight.

Runs the whole flight under a single-writer lock with a heartbeat, publishes a
truthful terminal status, and -- with ``--detach`` -- re-launches itself as a
process that outlives the shell that started it. Background jobs started inside
an interactive agent turn are killed at the turn boundary, so an unattended run
must be detached rather than backgrounded.

The status it writes distinguishes four things that are routinely conflated:

* the worker is **running** (the lock is held),
* work has **advanced** (trajectory journals gained records),
* the flight **stopped at a gate** (readiness or scientific feasibility),
* the flight is **complete** (every required stage completed and every queued
  row reached a terminal status).

Only the last of those is completion, and it is never inferred from the first.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smallAntibodyGen.experiments import her2_nf_campaign as campaign  # noqa: E402
from smallAntibodyGen.experiments import her2_nf_spec as spec  # noqa: E402
from smallAntibodyGen.experiments import her2_support_paths as paths  # noqa: E402

DEFAULT_CONFIG = "configs/experiments/her2_next_flight.json"


def build_parser():
    parser = argparse.ArgumentParser(
        prog="her2_next_flight_supervisor.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--from-stage", default=None, choices=campaign.STAGES)
    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument("--heartbeat-seconds", type=float, default=15.0)
    parser.add_argument("--max-restarts", type=int, default=0,
                        help="restarts after a crash. Default 0: a crash is a failure needing "
                             "diagnosis, not something to retry through.")
    parser.add_argument("--detach", action="store_true",
                        help="re-launch this supervisor as a detached process and return")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    context = spec.resolve_context(ROOT, config_path=ROOT / args.config, run_root=args.output)
    if args.detach:
        arguments = ["--device", args.device, "--heartbeat-seconds", str(args.heartbeat_seconds),
                     "--max-restarts", str(args.max_restarts)]
        if args.from_stage:
            arguments += ["--from-stage", args.from_stage]
        if args.max_trajectories is not None:
            arguments += ["--max-trajectories", str(args.max_trajectories)]
        record = campaign.launch_detached(
            context, script="scripts/her2_next_flight_supervisor.py", arguments=arguments)
        print(f"detached supervisor pid {record['pid']}; log {record['log']}", flush=True)
        print("verify with: python scripts/her2_next_flight.py health --config "
              f"{args.config}", flush=True)
        return 0

    stages = campaign.STAGES
    if args.from_stage:
        stages = campaign.STAGES[campaign.STAGES.index(args.from_stage):]
    services = campaign.build_services(context, device=args.device)
    status = campaign.supervise(context, services, stages=stages,
                                heartbeat_seconds=args.heartbeat_seconds,
                                max_restarts=args.max_restarts,
                                max_trajectories=args.max_trajectories)
    print(paths.canonical_json({"status": status["status"], "stages": status["stages"]}))
    return 0 if status["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
