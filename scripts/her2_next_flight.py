#!/usr/bin/env python
"""HER2 next flight: objective, preservation and proximity challenge.

One documented command advances the whole flight deterministically:

    .venv/Scripts/python.exe scripts/her2_next_flight.py run-all

That walks ``recover -> preflight -> geometry -> mixtures -> profile ->
calibrate -> freeze -> production -> audit -> couple -> report -> verify``
without any follow-up manual call. It may stop at a readiness gate (a missing
input, changed frozen source, insufficient storage) or at a scientific
feasibility gate (the proximity geometry fails at the declared radius, no
calibrated configuration qualifies). It never skips a required stage silently
and never reports a partial flight as complete: every stage writes a record, and
``verify`` checks the required list against what is on disk.

For an unattended run that outlives the launching shell, use the supervisor:

    .venv/Scripts/python.exe scripts/her2_next_flight_supervisor.py --detach

Individual stages are available by name for review and for re-running one piece
after a repair. ``smoke`` is the miniature native end-to-end path and is meant
to be run before the first full launch.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smallAntibodyGen.experiments import her2_nf_campaign as campaign  # noqa: E402
from smallAntibodyGen.experiments import her2_nf_report as report_lib  # noqa: E402
from smallAntibodyGen.experiments import her2_nf_spec as spec  # noqa: E402
from smallAntibodyGen.experiments import her2_support_paths as paths  # noqa: E402

DEFAULT_CONFIG = "configs/experiments/her2_next_flight.json"
CHOICES = tuple(campaign.STAGES) + ("smoke", "status", "run-all", "launch", "health")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="her2_next_flight.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=CHOICES, help="stage to run, or run-all")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="flight configuration (default: %(default)s)")
    parser.add_argument("--output", default=None,
                        help="run-root override; defaults to the config's run_root")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"),
                        help="inference/training device (default: %(default)s)")
    parser.add_argument("--max-trajectories", type=int, default=None,
                        help="operational bound on how many queued trajectories one production "
                             "session starts. It does not change the declared grid; the rest stay "
                             "queued and appear in every table.")
    parser.add_argument("--from-stage", default=None, choices=campaign.STAGES,
                        help="run-all: begin at this stage, keeping the declared order")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    context = spec.resolve_context(ROOT, config_path=ROOT / args.config, run_root=args.output)
    print(f"her2 next flight: stage {args.stage} -> {context.config['run_root']}", flush=True)

    if args.stage == "status":
        # The REPORTING queue: it lists every declared row before calibration has
        # frozen anything. Building the production queue here would refuse to
        # print a status until the flight was already runnable.
        queue = campaign.queue_for_reporting(context)
        state = campaign.campaign_state(context.run_root, queue)
        document = {"stages": campaign.stage_summary(context.run_root),
                    "trajectories": state["counts"],
                    "launch": campaign.launch_health(context)}
        print(paths.canonical_json(document))
        return 0
    if args.stage == "health":
        print(paths.canonical_json(campaign.launch_health(context)))
        return 0

    services = campaign.build_services(context, device=args.device)
    if args.stage == "launch":
        arguments = ["--device", args.device]
        if args.from_stage:
            arguments += ["--from-stage", args.from_stage]
        if args.max_trajectories is not None:
            arguments += ["--max-trajectories", str(args.max_trajectories)]
        record = campaign.launch_detached(context, arguments=arguments)
        print(f"launched pid {record['pid']}; log {record['log']}", flush=True)
        print("a PID is not progress: check `health` for the lock, the heartbeat and journalled "
              "updates before describing this flight as running.", flush=True)
        return 0
    if args.stage == "run-all":
        stages = campaign.STAGES
        if args.from_stage:
            stages = campaign.STAGES[campaign.STAGES.index(args.from_stage):]
        document = campaign.supervise(context, services, stages=stages,
                                      max_trajectories=args.max_trajectories)
        print(paths.canonical_json(document["stages"]))
        return 0 if document["status"] == "completed" else 2

    lock = campaign.CampaignLock(context.path(campaign.LOCK_FILE),
                                 owner=campaign.owner_identity())
    with lock.held():
        record = campaign.run_stage(context, args.stage, services,
                                    max_trajectories=args.max_trajectories)
    print(paths.canonical_json({k: v for k, v in record.items()
                                if k in ("stage", "status", "reason")}))
    if args.stage == "verify":
        document = report_lib.verify(context)
        return 0 if document["passed"] else 2
    return 0 if record["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
