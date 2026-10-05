#!/usr/bin/env python
"""Measure the real GPU cost of each route on this card, then forecast from it.

Three spill signals, because one is not enough on this box: the driver's free-VRAM
reading, the allocator's peak, **and** the per-row step time. A 5.6 GB configuration
once returned ``ok=True`` on this 4 GB card because Windows CUDA spills to system RAM, so
the absence of an OutOfMemoryError is not evidence that a configuration fits.

The forecast multiplies measured per-row costs by the declared schedule and reports
the six exposure counters separately. It does not turn "20,000 updates at effective
batch 64" into an exposure count: that arithmetic gives row presentations only, and
the student-forward count is strictly larger because a local contrast and a
preference pair each consume two forwards.

This is a WRITER of ``gpu_profile.json`` and ``runtime_forecast.json`` and it behaves like
one. It reads the config's own ``profile`` budget and schedule through the same function
the profile stage uses, takes the campaign's real lock for the whole run, and refuses -- with
no override flag -- a run root whose profile stage completed, whose freeze marker exists, or
whose lock is held or cannot be observed. The refusals happen before CUDA or a model is
touched.
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

DEFAULT_CONFIG = "configs/experiments/her2_generative_routes.json"


def build_parser():
    parser = argparse.ArgumentParser(prog="her2_routes_gpu_profile.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--route", default=None, help="profile one route instead of all enabled")
    return parser


def announce(route, block):
    print(f"\n== {route} ==", flush=True)
    for row in block["rows"]:
        print(f"  microbatch {row['microbatch']:>3} "
              f"step={row['median_step_seconds']} "
              f"per_row={row['seconds_per_row']} "
              f"peak={row['peak_allocated_mib']} MiB "
              f"free={row['driver_free_mib']} MiB "
              f"accepted={row['accepted']} "
              f"signals={row['signals']}", flush=True)
    print(f"  chosen microbatch: {block['chosen_microbatch']}", flush=True)


def main(argv=None):
    args = build_parser().parse_args(argv)
    config_path = ROOT / args.config
    document = spec.load_config(config_path)
    artifact_root = args.artifact_root or document["artifacts"]["root"]
    ordering = env.prepare_environment(artifact_root)
    env.require_environment_ordering(ordering, label="gpu_profile")

    from smallAntibodyGen.experiments import her2_routes_campaign as campaign
    context = spec.resolve_context(ROOT, config_path=config_path, artifact_root=artifact_root,
                                   revision=args.revision)
    _profiles, forecast = campaign.standalone_profile(
        context, lambda: campaign.build_services(context, device=args.device),
        routes=[args.route] if args.route else None, announce=announce)
    print("\n== forecast ==", flush=True)
    for route, block in sorted(forecast["per_route"].items()):
        print(f"  {route}: {block['hours']:.2f} h  "
              f"{block['optimizer_updates']} updates  "
              f"{block['row_presentations']} row presentations  "
              f"{block.get('student_forwards', 'n/a')} student forwards", flush=True)
        print(f"    {block.get('accounting') or block.get('basis')}", flush=True)
    print(f"  total {forecast['total_hours']:.2f} h, "
          f"with reserve {forecast['with_reserve_hours']:.2f} h", flush=True)
    print(f"  drop order: {forecast['drop_order']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
