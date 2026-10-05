#!/usr/bin/env python
"""Build B: learned evidence, composition and interchange-intervention pilot on a finite synthetic world.

A file-based CLI over ``smallAntibodyGen.experiments.functional_causal_learning`` (the learner, tapes,
evaluation, transfer and Build A bridge) and ``functional_evidence`` (the exact teacher, the q7 split and
the function families). Everything is synthetic; nothing here loads a pretrained model, downloads data or
touches CUDA, and nothing selects a winning arm.

Subcommands:

  world     print (or write) the explicit frozen world: the q7 code, split and counts (verified), the 93
            functions with truth tables and S4 orbits, and the Build A identity convention. Exit 0 only if
            every frozen count holds.
  tape      build the frozen SOURCE-only tape and shared initial state of each seed into a NEW output
            directory (the prepare stage) and print what was built, including the measured corrupted
            supervision (value changes, unchanged fraction, unsupported strata).
  run       run the stages in order (prepare, train, freeze, evaluate, transfer, export, report). Resumable:
            finished stages are verified and adopted, never recomputed; partial or foreign state refuses.
            A completed transfer or export stage re-verifies the adapters, caches and matrix artifacts it
            refers to on every entry. The held-out panels open only in ``evaluate``/``transfer``, after
            every source state is frozen, and the transfer-evaluation outcomes only after every adapter
            is frozen. ``export`` follows transfer, so it REUSES that already scored panel for a
            secondary analysis with settings fixed in advance: it is not a fresh confirmation.
  inspect   read an existing output directory: the stage state and, with --verify, every digest the
            stages recorded (tapes, frozen states, tracked adapters).
  selftest  a much smaller GENUINE end-to-end run (real tapes, every arm trained, hidden interventions,
            adapters, a real Build A matrix). Exit 0 only if every check holds.
  pilot     the bounded pilot (3 paired seeds, 1000 source updates, batch 64). An explicit command: nothing
            starts it implicitly. Time the selftest first. A negative result is a valid result.

CPU only; threads are pinned to one for ``selftest``/``pilot`` (and ``run --single-thread``). The output
path is caller-owned and may be on the F: development artifacts drive; an existing directory is only
resumed (same configuration, world and code), never emptied or deleted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smallAntibodyGen.experiments import functional_baseline as fb  # noqa: E402
from smallAntibodyGen.experiments import functional_causal_learning as fcl  # noqa: E402
from smallAntibodyGen.experiments import functional_evidence as fe  # noqa: E402
from smallAntibodyGen.experiments import functional_shared_transfer as fst  # noqa: E402


def _formatter(prog):
    return argparse.RawDescriptionHelpFormatter(prog, max_help_position=34, width=100)


def _add_config(parser):
    parser.add_argument("--config", default=None,
                        help="functional-causal-config JSON (strict; every key optional; defaults are "
                             "the bounded pilot)")
    parser.add_argument("--steps", type=int, default=None, help="override config.steps")
    parser.add_argument("--batch-size", type=int, default=None, help="override config.batch")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="override config.seeds")
    parser.add_argument("--arms", nargs="+", default=None, choices=list(fcl.ARMS),
                        help="override config.arms (F is required)")


def build_parser():
    parser = argparse.ArgumentParser(prog="functional_causal_learning.py", description=__doc__,
                                     formatter_class=_formatter)
    commands = parser.add_subparsers(dest="command", required=True,
                                     metavar="{world,tape,run,inspect,selftest,pilot}")

    world = commands.add_parser("world", formatter_class=_formatter,
                                help="print the explicit frozen world")
    world.add_argument("--output", default=None, help="also write the world document to this new file")

    tape = commands.add_parser("tape", formatter_class=_formatter,
                               help="build the source tapes and shared initial states")
    tape.add_argument("--output-dir", required=True)
    tape.add_argument("--deep-verify", action="store_true",
                      help="recompute every update digest and teacher label from the arrays")
    _add_config(tape)

    run = commands.add_parser("run", formatter_class=_formatter, help="run the stages",
                              description="Run the stages in order; see the module docstring.")
    run.add_argument("--output-dir", required=True,
                     help="pipeline directory (created; an existing one is resumed only if it belongs "
                          "to the same configuration, world and code)")
    run.add_argument("--stages", nargs="+", default=list(fcl.STAGES), choices=list(fcl.STAGES),
                     help="stages to run, in the order given (default: all)")
    run.add_argument("--single-thread", action="store_true", help="pin CPU threads to one")
    _add_config(run)

    inspect = commands.add_parser("inspect", formatter_class=_formatter,
                                  help="read an existing output directory")
    inspect.add_argument("--output-dir", required=True)
    inspect.add_argument("--verify", action="store_true",
                         help="re-verify the tapes, shared inputs and every frozen state")
    _add_config(inspect)

    selftest = commands.add_parser("selftest", formatter_class=_formatter,
                                   help="small genuine end-to-end check")
    selftest.add_argument("--output-dir", required=True)
    selftest.add_argument("--seed", type=int, default=0)

    pilot = commands.add_parser("pilot", formatter_class=_formatter,
                                help="the bounded pilot (explicit, never implicit)")
    pilot.add_argument("--output-dir", required=True)
    pilot.add_argument("--deep-verify", action="store_true")
    return parser


def _print(document):
    print(json.dumps(fb._builtin(document), indent=2, sort_keys=True, allow_nan=False))


def _config_from(args):
    document = fb.read_json(args.config) if getattr(args, "config", None) else {}
    overrides = {"steps": args.steps, "batch": args.batch_size, "seeds": args.seeds,
                 "arms": args.arms}
    for key, value in overrides.items():
        if value is not None:
            document[key] = value
    return fcl.CausalConfig.from_document(document)


def _run_world(args):
    counts = fe.verify_q7()
    document = fe.world_document()
    if args.output:
        path = Path(args.output)
        if path.exists():
            raise ValueError(f"{path} already exists; the world command never overwrites a file")
        fb.write_json(path, document)
    _print({"verified": counts, "functions": {role: len(fe.function_registry().role(role))
                                              for role in fe.FUNCTION_ROLES},
            "orbits": {role: len(fe.function_registry().orbits(role)) for role in fe.FUNCTION_ROLES},
            "world_sha256": fe.world_sha256(), "written": args.output,
            "identity": {"alphabet": fe.IDENTITY_ALPHABET, "length": fe.identity_length(),
                         "convention": fe.IDENTITY_CONVENTION},
            "claim_boundary": document["claim_boundary"]})
    return 0


def _run_tape(args):
    config = _config_from(args)
    with fst.single_thread():
        run = fcl.CausalRun.open(args.output_dir, config)
        record = run.prepare(deep_verify=args.deep_verify)
    _print({"output_dir": str(args.output_dir), "prepare": record,
            "note": ("only SOURCE signatures were labelled; the corrupted Ic targets are measured "
                     "(value changes, unchanged fraction, unsupported strata), not assumed")})
    return 0


def _run_run(args):
    config = _config_from(args)
    if args.single_thread:
        with fst.single_thread():
            run = fcl.CausalRun.open(args.output_dir, config)
            run.run(args.stages)
    else:
        run = fcl.CausalRun.open(args.output_dir, config)
        run.run(args.stages)
    _print({"output_dir": str(args.output_dir), "stages_run": list(args.stages),
            "stage_state": run.state["stages"],
            "report": (str(Path(args.output_dir) / fcl.REPORT_NAME) if "report" in args.stages
                       else None),
            "note": "a negative or inconclusive result is a valid result; no arm is selected"})
    return 0


def _run_inspect(args):
    config = _config_from(args)
    if not (Path(args.output_dir) / fcl.STATE_NAME).is_file():
        raise ValueError(f"{args.output_dir} is not a Build B output directory (no {fcl.STATE_NAME}); "
                         "inspect never creates one")
    run = fcl.CausalRun.open(args.output_dir, config)
    out = {"output_dir": str(args.output_dir),
           "stages": {name: bool(entry.get("complete")) for name, entry in run.state["stages"].items()}}
    if args.verify:
        verified = {}
        if run._complete("prepare"):
            run.preflight()
            verified["prepare"] = "tapes, bank, program table and initial states verified"
        if run._complete("freeze"):
            models = run.frozen_models()
            verified["freeze"] = {rid: model.state_sha256 for rid, model in models.items()}
            if run._complete("transfer"):
                stored = run._read_artifact("transfer/transfer.json",
                                            run.state["stages"]["transfer"]["transfer_sha256"])
                run._verify_transfer_artifacts(models, stored)
                verified["transfer"] = "tracked adapters verified (headers, weights, bindings)"
        out["verified"] = verified
    _print(out)
    return 0


def _run_selftest(args):
    report = fcl.selftest(args.output_dir, seed=args.seed)
    for check in report["checks"]:
        print(f"  [{'ok' if check['holds'] else '!!'}] {check['name']}")
    held = sum(1 for check in report["checks"] if check["holds"])
    print(f"selftest {'PASSED' if report['passed'] else 'FAILED'}: {held}/{len(report['checks'])} "
          f"checks in {report['cpu_seconds']:.1f} CPU s; report at "
          f"{Path(report['work_directory']) / fcl.SELFTEST_REPORT_NAME}")
    print(report["note"])
    return 0 if report["passed"] else 1


def _run_pilot(args):
    _print(fcl.pilot(args.output_dir, deep_verify=args.deep_verify))
    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "world":
            return _run_world(args)
        if args.command == "tape":
            return _run_tape(args)
        if args.command == "run":
            return _run_run(args)
        if args.command == "inspect":
            return _run_inspect(args)
        if args.command == "selftest":
            return _run_selftest(args)
        return _run_pilot(args)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
