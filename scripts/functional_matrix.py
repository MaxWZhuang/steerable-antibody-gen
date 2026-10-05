#!/usr/bin/env python
"""Acquired-only frozen-feature comparison matrix: a runnable CPU check of cache -> freeze -> score.

NumPy only. No native model is imported, discovered or loaded, no weights are downloaded, no GPU is
touched: every input and output path is explicit. Several FROZEN feature representations of one
candidate roster (each a versioned cache: a strict JSON header plus a numeric NPZ) are crossed with
fixed readout maps (linear, and a train-center RBF map, each followed by the baseline's ridge), label
budgets and seeds. Every cell is fitted by the accepted finite-support baseline (ridge reward,
linear-energy DPO, exact tilt, SIR), frozen into a hashed artifact, and only then scored against the
held-out outcomes. This is an opt-in comparison harness: it is not native generation, it makes no
claim that any representation or readout wins, and it selects no winning cell. Constant columns are
EXACT (bit-identical acquired values), never tolerance-classified, so a small but real variation is
not discarded for its units; the dense ridge/DPO systems are bounded in dimension before they are
allocated (a bound on one system, not a peak-memory guarantee).

Subcommands:

  selftest      write synthetic complete sequences, three synthetic caches (an informative one-hot
                encoding, a random projection of it and a pure-noise null control), a manifest and a
                plan, run a real 24-cell matrix, re-run it with SHUFFLED evaluation outcomes and
                with permuted CSV/cache row order, and check that no pre-evaluation byte changed,
                that acquisitions are paired and nested, that identities hold and that a tampered
                cell is refused. Writes matrix_selftest_report.json. Exit 0 only if every named
                check holds. It qualifies the machinery on a known synthetic truth, nothing more.
  run           freeze every cell from explicit CSV/JSON/cache paths, verify the freeze from disk,
                and only then open the evaluation outcomes. Writes cells/cell_*.json,
                matrix_inputs.json, matrix_freeze_index.json and matrix_evaluation_report.json.
                Exit 0 unless a cell FAILED (an unsupported cell is reported, not an error).
  evaluate      score a freeze that `run` left unscored (for example because the evaluation file
                was missing): re-verifies the freeze FROM DISK, then opens the evaluation outcomes
                and writes matrix_evaluation_report.json. Refuses an evaluation file that aliases
                a frozen file or the report, and an existing report unless --overwrite.
  verify-cache  load one cache and print its summary; refuses a stale, tampered, non-finite,
                object-dtype or inconsistent cache.

Input contracts (strict JSON: a duplicate key at any nesting level, NaN and unknown keys are refused):

  candidates CSV         candidate_id, sequence, group_id, split ("train" or "eval"),
                         prior_log_weight (finite, or the explicit spelling -inf). The sequence is
                         the COMPLETE fixed-length physical construct. This file carries the
                         reference prior; a cache cannot override it.
  training outcomes CSV  candidate_id, utility
  evaluation outcomes    candidate_id, utility (must cover EVERY eval candidate; opened last)
  manifest JSON          schema functional-matrix-manifest/1: target, assay, utility, grouping_rationale,
                         candidates {identity_kind "complete_sequence", alphabet, length}, prior {kind,
                         semantics, provenance}
  plan JSON              schema functional-matrix-plan/1: representations, maps, label_budgets, seeds,
                         baseline (BaselineConfig options minus seed/label budget; tau required),
                         reference {representation, map}, bootstrap {replicates, seed, confidence,
                         min_groups}. Everything is fixed before anything is fitted; nothing is tuned.
  feature cache          schema functional-feature-cache/1 (see the module docstring of
                         smallAntibodyGen.experiments.functional_matrix): complete-sequence identity,
                         per-row ids/sequences/groups/splits, the FeatureSpace with the feature
                         SupervisionHistory, checkpoint/source hashes, activation/pooling/token-map,
                         answer-free conditioning (quality explicitly null), and content hashes. The
                         header names its NPZ by a plain file name beside it, or pass --cache-npz.

Readout maps. Centering, scaling, RBF centers and the bandwidth are computed from the ACQUIRED
feature rows only (the acquired labels are drawn from the seed before any fit); every offered and
held-out row is then mapped. Trace normalisation gives each readout one overall scale and does NOT
equalise regularisation geometry: compare each cell's reported output dimension and effective ridge
df. DPO is linear in the mapped basis. Constant, duplicate and all-identical inputs are handled
explicitly and recorded; an undefined cell is recorded as unsupported with its reason.

Output safety. Any input path (including cache headers and NPZ files) that resolves to any output or
its .tmp staging file is refused before anything is written, even with --overwrite. A cells/
directory holding files this plan does not produce is refused (never cleaned). Inputs are never
modified. The evaluation outcomes are not opened or hashed until every cell and the freeze index are
written, reloaded and verified, and changing them changes no pre-evaluation file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smallAntibodyGen.experiments import functional_matrix as fm  # noqa: E402


def _formatter(prog):
    return argparse.RawDescriptionHelpFormatter(prog, max_help_position=34, width=100)


def build_parser():
    parser = argparse.ArgumentParser(prog="functional_matrix.py", description=__doc__,
                                     formatter_class=_formatter)
    commands = parser.add_subparsers(dest="command", required=True,
                                     metavar="{selftest,run,evaluate,verify-cache}")

    selftest = commands.add_parser(
        "selftest", formatter_class=_formatter,
        help="run a real synthetic matrix with caches and check what must hold",
        description=("Write a synthetic world, run a 24-cell matrix, re-run it with shuffled "
                     "evaluation outcomes and permuted row order, and write "
                     "matrix_selftest_report.json under --output-dir. Exit status 0 only if every "
                     "named check holds."))
    selftest.add_argument("--output-dir", required=True,
                          help="directory for the synthetic inputs, matrices and the report")
    selftest.add_argument("--seed", type=int, default=0,
                          help="seed of the synthetic world (default: 0)")

    run = commands.add_parser(
        "run", formatter_class=_formatter,
        help="freeze, verify and then score the matrix from explicit files",
        description=("Freeze every cell, write and reload and verify the cells and the freeze "
                     "index, and only then open the evaluation outcomes. See the module help for "
                     "the input contracts."))
    for flag, text in (("--candidates", "candidate CSV: candidate_id, sequence, group_id, split, "
                                        "prior_log_weight"),
                       ("--train-outcomes", "training outcome CSV: candidate_id, utility"),
                       ("--eval-outcomes", "evaluation outcome CSV (opened only after the freeze "
                                           "is verified): candidate_id, utility"),
                       ("--manifest", "matrix manifest JSON"),
                       ("--plan", "matrix plan JSON (frozen representations, maps, budgets, seeds)"),
                       ("--output-dir", "directory for cells/, the input record, the freeze index "
                                        "and the evaluation report")):
        run.add_argument(flag, required=True, help=text)
    run.add_argument("--cache", action="append", required=True, metavar="ID=HEADER",
                     help="a planned representation id and its cache header JSON; repeat once per "
                          "planned representation (exactly the planned set)")
    run.add_argument("--cache-npz", action="append", default=[], metavar="ID=NPZ",
                     help="an explicit NPZ for one cache, instead of the plain file name its "
                          "header declares beside it")
    run.add_argument("--overwrite", action="store_true",
                     help="replace existing outputs (default: refuse). Never replaces an input: a "
                          "path alias between an input and an output is refused before anything "
                          "is written")

    evaluate = commands.add_parser(
        "evaluate", formatter_class=_formatter,
        help="verify an existing freeze from disk, then score it",
        description=("Re-verify the freeze in --output-dir from disk (no state from any earlier "
                     "run is trusted), then open the evaluation outcomes and write "
                     "matrix_evaluation_report.json. Refuses an evaluation file that resolves to "
                     "the report, its .tmp file or any frozen file, and an existing report "
                     "unless --overwrite."))
    evaluate.add_argument("--output-dir", required=True,
                          help="directory holding a freeze written by `run`")
    evaluate.add_argument("--eval-outcomes", required=True,
                          help="evaluation outcome CSV: candidate_id, utility")
    evaluate.add_argument("--overwrite", action="store_true",
                          help="replace an existing evaluation report (default: refuse)")

    verify = commands.add_parser(
        "verify-cache", formatter_class=_formatter,
        help="load one feature cache and print its summary",
        description="Load and verify one cache (strict header, file/content/canonical hashes, "
                    "finite numeric arrays) and print its summary as JSON.")
    verify.add_argument("--header", required=True, help="cache header JSON")
    verify.add_argument("--npz", default=None,
                        help="explicit NPZ path (default: the plain file name in the header)")
    return parser


def _pairs(items, flag):
    out = {}
    for item in items or ():
        rep, separator, path = item.partition("=")
        if not separator or not rep or not path:
            raise ValueError(f"{flag} expects ID=PATH, got {item!r}")
        if rep in out:
            raise ValueError(f"{flag} names {rep!r} twice")
        out[rep] = path
    return out


def _run_selftest(args):
    report = fm.selftest(args.output_dir, seed=args.seed)
    for check in report["checks"]:
        print(f"  [{'ok' if check['holds'] else '!!'}] {check['name']}")
    print(f"selftest {'PASSED' if report['passed'] else 'FAILED'}: "
          f"{sum(1 for c in report['checks'] if c['holds'])}/{len(report['checks'])} checks; "
          f"report at {Path(args.output_dir) / fm.SELFTEST_REPORT_NAME}")
    print(report["note"])
    return 0 if report["passed"] else 1


def _run_matrix(args):
    summary = fm.run_matrix(
        candidates=args.candidates, train_outcomes=args.train_outcomes,
        eval_outcomes=args.eval_outcomes, manifest=args.manifest, plan=args.plan,
        caches=_pairs(args.cache, "--cache"), cache_npz=_pairs(args.cache_npz, "--cache-npz"),
        output_dir=args.output_dir, overwrite=args.overwrite)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 1 if summary["status_counts"][fm.STATUS_FAILED] else 0


def _run_evaluate(args):
    results = fm.evaluate_matrix(args.output_dir, args.eval_outcomes, overwrite=args.overwrite)
    summary = fm.summarize_evaluation(args.output_dir, results)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 1 if summary["status_counts"][fm.STATUS_FAILED] else 0


def _run_verify_cache(args):
    cache = fm.load_feature_cache(args.header, npz_path=args.npz)
    print(json.dumps({"summary": cache.summary(), "raw_file_digests": cache.raw},
                     indent=2, sort_keys=True, allow_nan=False))
    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "selftest":
            return _run_selftest(args)
        if args.command == "run":
            return _run_matrix(args)
        if args.command == "evaluate":
            return _run_evaluate(args)
        return _run_verify_cache(args)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
