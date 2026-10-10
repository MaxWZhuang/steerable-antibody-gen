#!/usr/bin/env python
"""HER2 generative routes: one command advances the whole flight deterministically.

    .venv/Scripts/python.exe scripts/her2_routes_flight.py run-all

That walks ``acquire -> preflight -> sources -> split -> profile -> smoke ->
freeze -> bakeoff -> route_gate -> controls -> priors -> posttrain -> decoding ->
confirm -> report -> verify`` with no follow-up manual call. It may stop at a
readiness gate (a missing input, a changed snapshot, insufficient disk) or a
feasibility gate (an underpowered hard split, no route clearing G1, a gain that
does not survive the controls). It never skips a required stage silently and never
reports a partial flight as complete.

For an unattended run that outlives the launching shell, use the supervisor:

    .venv/Scripts/python.exe scripts/her2_routes_supervisor.py --detach

``prepare_environment`` runs **before** any scientific import, because torch and
transformers read their cache variables at import time. Use
``./.venv/Scripts/python.exe``, never bare ``python``: the system interpreter fakes
missing-module failures on this box.
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

# The cache and dependency environment is prepared before torch is imported, so
# this import block is deliberately split and deliberately ordered.
from smallAntibodyGen.experiments import her2_routes_env as env  # noqa: E402
if __name__ == "__main__":
    env.bootstrap_cli(ROOT, sys.argv[1:])
from smallAntibodyGen.experiments import her2_routes_spec as spec  # noqa: E402
from smallAntibodyGen.experiments import her2_support_paths as paths  # noqa: E402

DEFAULT_CONFIG = "configs/experiments/her2_generative_routes.json"


def build_parser():
    from smallAntibodyGen.experiments import her2_routes_campaign as campaign
    choices = tuple(campaign.STAGES) + campaign.EXTRA_STAGES
    parser = argparse.ArgumentParser(
        prog="her2_routes_flight.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=choices, help="stage to run, or run-all")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="flight configuration (default: %(default)s)")
    parser.add_argument("--artifact-root", default=None,
                        help="override the configured artifact root (must be absolute)")
    parser.add_argument("--output", default=None, help="run-root override")
    parser.add_argument("--revision", default=None,
                        help="revision directory under <artifact root>/runs (default: the "
                             "config's revisions.current)")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--from-stage", default=None, choices=campaign.STAGES,
                        help="run-all: begin at this stage, keeping the declared order")
    parser.add_argument("--through-stage", default=None, choices=campaign.STAGES,
                        help="stop this worker after this stage; later stages remain unfinished")
    parser.add_argument("--no-reuse", action="store_true",
                        help="rerun a completed stage instead of reusing it. Rerunning bakeoff "
                             "discards the queue it just spent hours on, so this is explicit.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config_path = ROOT / args.config
    document = spec.load_config(config_path)
    artifact_root = args.artifact_root or document["artifacts"]["root"]
    ordering = env.prepare_environment(artifact_root)
    env.require_environment_ordering(ordering, label=f"flight::{args.stage}")

    from smallAntibodyGen.experiments import her2_routes_campaign as campaign
    context = spec.resolve_context(ROOT, config_path=config_path, run_root=args.output,
                                   artifact_root=artifact_root, revision=args.revision)
    print(f"her2 generative routes: stage {args.stage} -> {context.run_root} "
          f"(revision {context.revision})", flush=True)

    # status and health stay usable whatever the config declares. Everything else is refused
    # here -- before the environment record, services or any stage -- if a declared training
    # phase holds the requested stage or the requested bounds.
    if args.stage not in ("status", "health"):
        try:
            if args.stage in ("run-all", "launch"):
                stages = campaign.select_stages(context, from_stage=args.from_stage,
                                                through_stage=args.through_stage)
            else:
                campaign.require_stage_allowed(context, args.stage)
        except ValueError as error:
            print(f"refused: {error}", flush=True)
            return 2
    paths.write_json(context.path("environment.json"), ordering)

    if args.stage == "status":
        try:
            phase = campaign.phase_policy(context.config)
            phase = None if phase is None else {
                "phase_id": phase["phase_id"], "policy_sha256": phase["policy_sha256"],
                "admission": phase["admission"], "held_stages": phase["held_stages"],
                "complete_flight": False}
        except ValueError as error:
            phase = {"invalid": str(error)}
        print(paths.canonical_json({
            "stages": campaign.stage_summary(context),
            "launch": campaign.launch_health(context),
            "revision": context.revision, "training_phase": phase}))
        return 0
    if args.stage == "health":
        print(paths.canonical_json(campaign.launch_health(context)))
        return 0

    services = campaign.build_services(context, device=args.device)
    if args.stage == "launch":
        arguments = ["--device", args.device]
        if args.from_stage:
            arguments += ["--from-stage", args.from_stage]
        if args.through_stage:
            arguments += ["--through-stage", args.through_stage]
        record = campaign.launch_detached(context, arguments=arguments)
        print(f"launched pid {record['pid']}; log {record['log']}", flush=True)
        print("a PID is not progress: check `health` for the lock, the heartbeat and the "
              "journalled steps before describing this flight as running.", flush=True)
        return 0
    if args.stage == "run-all":
        record = campaign.supervise(context, services, stages=stages)
        print(paths.canonical_json(record["stages"]))
        return 0 if record["status"] in campaign.SUCCESS_STATUSES else 2

    lock = campaign.CampaignLock(context.path(campaign.LOCK_FILE),
                                 owner=campaign.owner_identity())
    with lock.held():
        record = campaign.run_stage(context, args.stage, services, reuse=not args.no_reuse)
    print(paths.canonical_json({key: record.get(key) for key in
                                ("stage", "status", "reason", "reused")}))
    return 0 if record["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
