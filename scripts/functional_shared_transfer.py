#!/usr/bin/env python
"""Build D: shared-parameter functional transfer from verified computations, on a frozen native token cache.

A file-based CLI over ``smallAntibodyGen.experiments.functional_shared_transfer``. It consumes ONE
verified native token cache (written by ``scripts/functional_native_export.py`` / the Python export
API: this CLI never discovers, downloads or deserialises a model) plus a cohort manifest naming the
exact candidate ids of source_train, source_aux (default: source_train), source_dev, target_train and
target_eval, and four outcome CSVs (``candidate_id,utility``; one declared source assay/scale and one
target assay/scale; finite exact numbers).

Subcommands:

  inspect    validate the cohorts against the cache (rows outside the declared union, overlapping
             sequences or groups, wrong cache splits, a mixed-length target roster are refused) and
             print a summary. Exit 0 only if everything holds, else 2 with the reason.
  run        run the stages in order (prepare, train, features, freeze, evaluate, panels, report).
             Resumable: finished runs and complete freezes are verified and adopted, never refitted;
             partial or foreign state is refused. The target-EVALUATION outcomes are opened only by
             the ``evaluate`` stage, after one combined matrix over native + every arm x source seed
             has been frozen and verified, so ``--stages prepare train features freeze`` can run
             before that file exists.
  synthetic  write a NEW directory with the synthetic pilot's inputs: a real token cache from a tiny
             RANDOMLY initialised CPU DPLM, the cohort manifest and four outcome CSVs (declared
             computational functions, not biology). ``--withhold-eval`` writes the evaluation
             outcomes under ``held/`` so they can be placed after the freeze.
  selftest   a much smaller end-to-end run (real tiny DPLM cache, all five arms, two paired seeds, exact
             F/S parity, resume, gradient wiring, one combined freeze). Exit 0 only if every check holds.
  pilot      the bounded synthetic pilot (256 macrosteps of 32, three paired seeds, budgets 16/64). It
             is an explicit command: nothing starts it implicitly. A negative result is a valid result.

Hashes are integrity, not authenticity. Nothing here selects a winning method.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smallAntibodyGen.experiments import _functional_transfer_synthetic as synth  # noqa: E402
from smallAntibodyGen.experiments import functional_baseline as fb  # noqa: E402
from smallAntibodyGen.experiments import functional_composition as fc  # noqa: E402
from smallAntibodyGen.experiments import functional_native_export as fne  # noqa: E402
from smallAntibodyGen.experiments import functional_shared_transfer as fst  # noqa: E402


def _formatter(prog):
    return argparse.RawDescriptionHelpFormatter(prog, max_help_position=34, width=100)


def _add_inputs(parser):
    parser.add_argument("--token-header", required=True, help="native token cache header JSON")
    parser.add_argument("--token-npz", default=None,
                        help="explicit NPZ of the token cache (default: its declared plain name)")
    parser.add_argument("--cohorts", required=True, help="functional-cohort-manifest/1 JSON")


def _add_outcomes(parser):
    parser.add_argument("--source-train-outcomes", required=True,
                        help="CSV candidate_id,utility for exactly the source_train ids")
    parser.add_argument("--source-dev-outcomes", required=True,
                        help="CSV for exactly the source_dev ids (used only to score source "
                             "preservation after training)")
    parser.add_argument("--target-train-outcomes", required=True,
                        help="CSV for exactly the target_train ids (read by the freeze stage)")
    parser.add_argument("--target-eval-outcomes", required=True,
                        help="CSV for exactly the target_eval ids (opened ONLY by the evaluate "
                             "stage; it may not exist yet when earlier stages run)")


def build_parser():
    parser = argparse.ArgumentParser(prog="functional_shared_transfer.py", description=__doc__,
                                     formatter_class=_formatter)
    commands = parser.add_subparsers(dest="command", required=True,
                                     metavar="{inspect,run,synthetic,selftest,pilot}")

    inspect = commands.add_parser("inspect", formatter_class=_formatter,
                                  help="validate the cohorts against the token cache",
                                  description="Validate the data boundary and print a summary.")
    _add_inputs(inspect)

    run = commands.add_parser(
        "run", formatter_class=_formatter, help="run the transfer pipeline",
        description=("Run the stages in order. See the module docstring for what each stage does "
                     "and which files it opens."))
    _add_inputs(run)
    _add_outcomes(run)
    run.add_argument("--output-dir", required=True,
                     help="pipeline directory (created; an existing one is resumed only if it "
                          "belongs to the same inputs and configuration)")
    run.add_argument("--config", default=None,
                     help="functional-shared-transfer config JSON (strict; every key optional; "
                          "defaults are the bounded pilot)")
    run.add_argument("--stages", nargs="+", default=list(fst.STAGES), choices=list(fst.STAGES),
                     help="stages to run, in the order given (default: all)")
    run.add_argument("--steps", type=int, default=None, help="override config.steps")
    run.add_argument("--batch-size", type=int, default=None, help="override config.batch")
    run.add_argument("--source-seeds", type=int, nargs="+", default=None,
                     help="override config.source_seeds")
    run.add_argument("--acquisition-seeds", type=int, nargs="+", default=None,
                     help="override config.acquisition_seeds")
    run.add_argument("--label-budgets", type=int, nargs="+", default=None,
                     help="override config.label_budgets")
    run.add_argument("--continued-f-extra-steps", type=int, default=None,
                     help="add the optional continued-F control with this many extra macrosteps "
                          "(its extra functional presentations are disclosed)")
    run.add_argument("--single-thread", action="store_true",
                     help="pin CPU threads to one for the run (bit-reproducible parity checks)")

    synthetic = commands.add_parser(
        "synthetic", formatter_class=_formatter,
        help="write the synthetic pilot's inputs (real tiny random CPU DPLM token cache)",
        description="Write a NEW directory of inputs for `run`. Declared computational functions.")
    synthetic.add_argument("--output-dir", required=True, help="a directory that must not exist")
    synthetic.add_argument("--size", choices=("pilot", "selftest"), default="pilot",
                           help="pilot: 128/512/64/128/128 sequences; selftest: a tiny world")
    synthetic.add_argument("--seed", type=int, default=0)
    synthetic.add_argument("--withhold-eval", action="store_true",
                           help="write the target-evaluation outcomes under held/ instead")

    selftest = commands.add_parser(
        "selftest", formatter_class=_formatter, help="small end-to-end check on a tiny random model",
        description=("Real tiny random CPU DPLM cache, all five arms, two paired seeds, exact F/S "
                     "parity, resume, gradient wiring and one combined freeze. A new directory is "
                     "created below --output-dir."))
    selftest.add_argument("--output-dir", required=True)
    selftest.add_argument("--seed", type=int, default=0)

    pilot = commands.add_parser(
        "pilot", formatter_class=_formatter, help="the bounded synthetic pilot (explicit, never implicit)",
        description=("Build the pilot's synthetic inputs and run every stage. Not started by any "
                     "other command. A negative result is a valid result."))
    pilot.add_argument("--output-dir", required=True)
    pilot.add_argument("--seed", type=int, default=0)
    return parser


def _print(document):
    print(json.dumps(fb._builtin(document), indent=2, sort_keys=True, allow_nan=False))


def _config_from(args):
    document = fb.read_json(args.config) if args.config else {}
    overrides = {"steps": args.steps, "batch": args.batch_size, "source_seeds": args.source_seeds,
                 "acquisition_seeds": args.acquisition_seeds, "label_budgets": args.label_budgets,
                 "continued_f_extra_steps": args.continued_f_extra_steps}
    for key, value in overrides.items():
        if value is not None:
            document[key] = value
    return fst.TransferConfig.from_document(document)


def _run_inspect(args):
    cache = fne.load_token_cache(args.token_header, npz_path=args.token_npz)
    manifest = fc.read_cohort_manifest(args.cohorts)
    cohorts = fc.validate_cohorts(cache, manifest)
    _print({"valid": True, "summary": cohorts.summary(cache),
            "note": ("rows outside the declared union, overlapping sequences or groups, wrong cache "
                     "splits and a mixed-length target roster are refused; hashes are integrity, "
                     "not authenticity")})
    return 0


def _inputs(args):
    return fst.TransferInputs(
        token_header=args.token_header, token_npz=args.token_npz, cohorts=args.cohorts,
        source_train_outcomes=args.source_train_outcomes,
        source_dev_outcomes=args.source_dev_outcomes,
        target_train_outcomes=args.target_train_outcomes,
        target_eval_outcomes=args.target_eval_outcomes)


def _run_run(args):
    config = _config_from(args)
    if args.single_thread:
        with fst.single_thread():
            transfer = fst.TransferRun.open(_inputs(args), args.output_dir, config)
            results = transfer.run(args.stages)
    else:
        transfer = fst.TransferRun.open(_inputs(args), args.output_dir, config)
        results = transfer.run(args.stages)
    _print({"output_dir": str(args.output_dir), "stages_run": list(args.stages),
            "stage_state": transfer.state["stages"],
            "report": (str(Path(args.output_dir) / fst.REPORT_NAME) if "report" in args.stages
                       else None),
            "note": ("a negative or inconclusive result is a valid result; no method is selected")})
    return 0


def _run_synthetic(args):
    base = synth.PILOT_SPEC if args.size == "pilot" else synth.SELFTEST_SPEC
    world = synth.build_world(dataclasses.replace(base, seed=args.seed))
    paths = synth.write_inputs(world, args.output_dir, model_seed=args.seed,
                               withhold_eval=args.withhold_eval)
    _print({"written": paths, "counts": base.totals(),
            "declaration": synth.DECLARATION,
            "note": ("a real token cache from a tiny RANDOMLY initialised CPU DPLM; the functions "
                     "are computational, not biological")})
    return 0


def _run_selftest(args):
    report = fst.selftest(args.output_dir, seed=args.seed)
    for check in report["checks"]:
        print(f"  [{'ok' if check['holds'] else '!!'}] {check['name']}")
    held = sum(1 for check in report["checks"] if check["holds"])
    print(f"selftest {'PASSED' if report['passed'] else 'FAILED'}: {held}/{len(report['checks'])} "
          f"checks; report at {Path(report['work_directory']) / fst.SELFTEST_REPORT_NAME}")
    print(report["note"])
    return 0 if report["passed"] else 1


def _run_pilot(args):
    _print(fst.pilot(args.output_dir, seed=args.seed))
    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "inspect":
            return _run_inspect(args)
        if args.command == "run":
            return _run_run(args)
        if args.command == "synthetic":
            return _run_synthetic(args)
        if args.command == "selftest":
            return _run_selftest(args)
        return _run_pilot(args)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
