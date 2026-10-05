#!/usr/bin/env python
"""Truthful status: what ran, what is blocked, whether a writer is alive.

Three separate facts, printed as three separate things, because conflating them is
how a launch gets reported as a result:

* **stages** -- which required stages completed, which are blocked and why;
* **liveness** -- whether the OS lock is held. The heartbeat's age is a *progress*
  signal and is never the liveness test. An inaccessible lock is reported as UNKNOWN,
  never as "stopped";
* **completion** -- asserted only by the verify stage against the required stage
  list, and it asserts COMPLETENESS rather than correctness.

**Read-only.** This script does not prepare the environment, create the artifact ``tmp``
directory, create the run root or take or write a lock. It prints the run root it resolved
and, when that root holds no stage record, names the sibling revision roots that do exist,
so an empty ``0/17`` is never mistaken for "no flight".
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smallAntibodyGen.experiments import her2_routes_spec as spec  # noqa: E402
from smallAntibodyGen.experiments import her2_support_paths as paths  # noqa: E402

DEFAULT_CONFIG = "configs/experiments/her2_generative_routes.json"


def build_parser():
    parser = argparse.ArgumentParser(prog="her2_routes_status.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--json", action="store_true", help="print the record and nothing else")
    return parser


def status_record(context):
    """The whole status document for ``context``. No writes."""
    from smallAntibodyGen.experiments import her2_routes_campaign as campaign
    stages = campaign.stage_summary(context)
    health = campaign.launch_health(context)
    verification = context.path("verification.json")
    verified = paths.read_json(verification) if verification.is_file() else None
    return {"campaign_id": context.campaign_id, "revision": context.revision,
            "run_root": campaign.describe_run_root(context),
            "stages": stages, "launch": health,
            "verification": (None if verified is None else
                             {"checks": verified["checks"], "passed": verified["passed"],
                              "problems": verified["problems"],
                              "asserts": verified["asserts"]}),
            "completion": {
                "required_stages_completed": sum(
                    1 for block in stages.values() if block["status"] == "completed"),
                "required_stages": len(campaign.REQUIRED_STAGES),
                "complete": all(stages[name]["status"] == "completed"
                                for name in campaign.REQUIRED_STAGES)}}


def main(argv=None):
    args = build_parser().parse_args(argv)
    config_path = ROOT / args.config
    document = spec.load_config(config_path)
    artifact_root = args.artifact_root or document["artifacts"]["root"]
    from smallAntibodyGen.experiments import her2_routes_campaign as campaign
    context = spec.resolve_context(ROOT, config_path=config_path, artifact_root=artifact_root,
                                   revision=args.revision, ensure=False)
    record = status_record(context)
    health, root = record["launch"], record["run_root"]
    if args.json:
        print(paths.canonical_json(record))
        return 0
    print(f"campaign {record['campaign_id']} revision {record['revision']}")
    print(f"  run root: {root['run_root']}  (exists: {root['exists']}, "
          f"{root['stage_records']} stage record(s))")
    if root["note"]:
        print(f"  NOTE: {root['note']}")
    print(f"  liveness: {health['health']}  (lock {health['lock']['state']}, "
          f"{health['journalled_steps']} journalled steps, "
          f"{health['stages_completed']} stages completed)")
    if health["worker_running"] is None:
        print("    the lock could not be observed: this is UNKNOWN, not 'stopped'. Do not "
              "launch a second writer on the strength of it.")
    for name in campaign.REQUIRED_STAGES:
        block = record["stages"][name]
        marker = {"completed": "ok", "not_run": "--"}.get(block["status"], "!!")
        print(f"  [{marker}] {name:<12} {block['status']}"
              + (f"  {block['reason']}" if block["reason"] else ""))
    print(f"  complete: {record['completion']['complete']} "
          f"({record['completion']['required_stages_completed']}/"
          f"{record['completion']['required_stages']} required stages)")
    verified = record["verification"]
    if verified is not None:
        print(f"  verification: {verified['checks']} checks, passed={verified['passed']}")
        print(f"    {verified['asserts']}")
    print("\na running worker, a launched PID and a complete flight are three different facts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
