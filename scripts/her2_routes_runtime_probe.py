#!/usr/bin/env python
"""Exercise the runtime before the flight: objectives, gradients, resume, shapes.

    .venv/Scripts/python.exe scripts/her2_routes_runtime_probe.py --stage all

One command that runs the parts of the production code a bad launch would only
discover hours in, and writes every measured number to
``<run root>/runtime_probe.json``:

* ``objectives`` -- one microbatch of each scheduled task through the real
  dispatcher, with the per-task forward count, the named loss terms and the
  per-task share of the gradient norm. A task that reports a loss and no gradient
  is visible here.
* ``heads`` -- the observation-head objective on a real head batch, with the
  measured answer to "does the head's gradient reach the trunk".
* ``sensitivity`` -- the numeric objective against a perturbed measurement and a
  perturbed context. An objective that ignores its target still produces a falling
  curve, so this is a difference and not a shape.
* ``shapes`` -- the profile at tiny microbatches plus the measured
  forwards-per-row ratio, and the shape ``selected_shapes`` would hand the loop.
* ``resume`` -- a short fit, an interruption and a resume, with the trajectory
  compared step for step.

It reads no reserved outcome: the split freeze gates the HER2 loader and this
script never mints a ``ReservedUnlock``. It writes only under the artifact root.

``prepare_environment`` runs before any scientific import, because torch and
transformers read their cache variables at import time. Use
``./.venv/Scripts/python.exe``, never bare ``python``.
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
STAGES = ("objectives", "heads", "sensitivity", "shapes", "resume", "all")


def build_parser():
    parser = argparse.ArgumentParser(prog="her2_routes_runtime_probe.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", default="all", choices=STAGES)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--route", default=None,
                       help="probe one route instead of every enabled one")
    parser.add_argument("--updates", type=int, default=6,
                       help="updates for the resume probe (default: %(default)s)")
    parser.add_argument("--microbatches", default="1,2",
                       help="comma-separated microbatches for the shape probe")
    return parser


def probe_objectives(services, route):
    """One microbatch of every scheduled task, dispatched through the real path."""
    import numpy as np
    import torch
    from smallAntibodyGen.experiments import her2_routes_data as data_lib
    from smallAntibodyGen.experiments import her2_routes_tasks as tasks_lib
    model = services.build_route(route)
    heads = services.build_heads(route, model)
    stream = services.curriculum_stream(route, seed=int(services.config["split"]["seed"]),
                                        updates=64)
    settings = services.task_settings()
    rng = np.random.default_rng([int(services.config["split"]["seed"]), 101])
    seen, rows, losses = {}, [], {}
    for step in range(1, 97):
        draw = stream(step, 0, rng)
        if draw is None or draw.rows() == 0 or draw.task in seen:
            continue
        outcome = tasks_lib.dispatch(services, route, model, heads, draw, arm="full",
                                     settings=settings, rng=rng)
        seen[draw.task] = True
        losses[draw.task] = outcome.loss
        rows.append(outcome.document())
        if len(seen) == len(data_lib.TASKS):
            break
    parameters = [p for p in list(model.parameters()) + list(heads.parameters())
                  if p.requires_grad]
    fractions = tasks_lib.gradient_fractions(losses, parameters)
    record = {"route": route, "tasks_observed": sorted(seen),
              "tasks_not_observed": sorted(set(data_lib.TASKS) - set(seen)),
              "microbatches": rows, "gradient_fractions": fractions,
              "declared_task_shares": dict(services.config["curriculum"]["shares"]),
              "reading": ("compare each task's gradient fraction with its share of updates. A "
                          "task with thirty per cent of the updates and three per cent of the "
                          "gradient is not contributing thirty per cent.")}
    del model, heads
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return record


def probe_heads(services, route):
    """The head objective, and whether its gradient reaches the trunk."""
    import torch
    model = services.build_route(route)
    heads = services.build_heads(route, model)
    record = services.gradient_reach(route, model, heads=heads)
    del model, heads
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return record


def probe_sensitivity(services, route):
    """Does the numeric objective move when the measurement moves? Measured, not assumed."""
    import torch
    from smallAntibodyGen.experiments import her2_routes_data as data_lib
    from smallAntibodyGen.experiments import her2_routes_tasks as tasks_lib
    model = services.build_route(route)
    contrasts = [contrast for contrast in services.contrast_pool(
        [row for row in services.eligible_summaries() if row.is_training],
        seed=int(services.config["split"]["seed"]))
        if contrast.margin is not None][:2]
    if not contrasts:
        del model
        return {"supported": False,
                "reason": ("no measured contrast with a finite margin is enrolled, so there is "
                           "nothing for a numeric sensitivity probe to perturb")}
    kappa = float(services.config["curriculum"]["kappa"])

    def build(scale):
        perturbed = []
        for contrast in contrasts:
            clone = data_lib.LocalContrast(**{**contrast.__dict__})
            clone.margin = float(contrast.margin) * float(scale) - 1.0
            perturbed.append(clone)
        outcome = tasks_lib.local_outcome(services, route, model, perturbed, kappa=kappa,
                                         reconstruction_weight=0.25)
        return float(outcome.loss.detach())
    measurement = tasks_lib.require_sensitive(lambda: build(1.0), perturb=lambda: build(-3.0),
                                              label="numeric margin")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"supported": True, "measurement": measurement,
            "contrasts": [contrast.document() for contrast in contrasts]}


def probe_shapes(services, route, *, microbatches):
    profile = services.profile(route, microbatches=microbatches, steps=3)
    return {"profile": profile, "selected_shapes": services.selected_shapes(),
            "ledger_sample": services.ledger_sample()}


def probe_resume(services, route, *, directory, updates):
    return services.smoke(route, directory=directory, updates=int(updates))


def main(argv=None):
    args = build_parser().parse_args(argv)
    config_path = ROOT / args.config
    document = spec.load_config(config_path)
    artifact_root = args.artifact_root or document["artifacts"]["root"]
    ordering = env.prepare_environment(artifact_root)
    env.require_environment_ordering(ordering, label="runtime_probe")

    from smallAntibodyGen.experiments import her2_routes_campaign as campaign
    context = spec.resolve_context(ROOT, config_path=config_path,
                                   artifact_root=artifact_root, revision=args.revision)
    services = campaign.build_services(context, device=args.device)
    routes = [args.route] if args.route else list(context.config["routes"]["enabled"])
    microbatches = tuple(int(value) for value in str(args.microbatches).split(","))
    wanted = (("objectives", "heads", "sensitivity", "shapes", "resume")
              if args.stage == "all" else (args.stage,))
    record = {"schema_version": spec.ROUTES_SCHEMA, "record_kind": "runtime_probe",
              "stages": list(wanted), "routes": routes, "device": args.device,
              "reserved_outcomes_read": False,
              "guarantee": ("the split freeze gates the HER2 outcome loader and this script never "
                            "mints a ReservedUnlock, so no reserved measurement is materialised "
                            "by any probe here."),
              "results": {}}
    for route in routes:
        block = {}
        if "objectives" in wanted and route != "dwjs_energy":
            block["objectives"] = probe_objectives(services, route)
        if "heads" in wanted and route != "dwjs_energy":
            block["heads"] = probe_heads(services, route)
        if "sensitivity" in wanted and route != "dwjs_energy":
            block["sensitivity"] = probe_sensitivity(services, route)
        if "shapes" in wanted:
            block["shapes"] = (probe_shapes(services, route, microbatches=microbatches)
                               if route != "dwjs_energy"
                               else {"profile": services.profile_energy()})
        if "resume" in wanted:
            block["resume"] = probe_resume(
                services, route, directory=context.path("runtime_probe", route),
                updates=args.updates)
        record["results"][route] = block
    target = context.path("runtime_probe.json")
    paths.write_json(target, campaign._plain(record))
    failures = sorted(
        f"{route}::{name}" for route, block in record["results"].items()
        for name, value in block.items()
        if isinstance(value, dict) and value.get("passed") is False)
    print(paths.canonical_json({"written": str(target), "routes": routes,
                                "failures": failures}))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
