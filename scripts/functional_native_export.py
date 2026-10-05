#!/usr/bin/env python
"""Native token-feature export: inspect/verify a plan or a token cache, pool it for the matrix, selftest.

The EXPORT itself is a Python API, not a command: it needs a native model instance that only you can
construct (this CLI does not discover, download or deserialise any checkpoint, and invents no
generic model factory). Typical use::

    from smallAntibodyGen.experiments import functional_native_export as fne
    model = ...                                              # your ConditionedDPLM
    plan = fne.read_export_plan("plan.json")
    fne.export_dplm_tokens(model, plan, "tok.json", "tok.npz", batch_size=8,
                           protected_input_paths=["candidates.csv", "manifest.json"])

Everything after that is file-based and lives here. No CUDA is initialised; every path is explicit.

Subcommands:

  inspect   print a summary of ONE input: --plan (a functional-native-export/1 plan: strictly parsed,
            canonical, no model needed) or --header (a token cache: fully loaded and verified).
  verify    load a token cache (strict header, every semantic binding, NPZ file/content hashes,
            offsets against the chains, canonical digest) and re-derive its digests from the held
            arrays. Exit 0 only if everything holds; exit 2 with the reason otherwise.
  pool      pool a verified token cache into a functional-feature-cache/1 the matrix loads
            unchanged (--pooling all_nonpad_mean | residue_mean). Joins an exact roster (the
            candidate CSV and the matrix manifest), refuses variable full-identity length, never
            trims, pads or drops, and refuses any output that already exists. The candidate CSV, the
            manifest, the token header/NPZ and every --protect path are protected from aliasing.
  selftest  export from a real tiny RANDOMLY initialised CPU DPLM, pool two ways, run ONE small
            matrix, and write a report under a fresh directory created below --output-dir. Exit 0
            only if every named check holds. It qualifies the machinery; it is not a pretrained or
            biological validation and it selects no best method.

Hashes are integrity, not authenticity: a cache whose WHOLE chain was rewritten consistently is not
detected, and the checkpoint/source hashes in a plan are caller assertions.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smallAntibodyGen.experiments import functional_baseline as fb  # noqa: E402
from smallAntibodyGen.experiments import functional_matrix as fm  # noqa: E402
from smallAntibodyGen.experiments import functional_native_export as fne  # noqa: E402


def _formatter(prog):
    return argparse.RawDescriptionHelpFormatter(prog, max_help_position=34, width=100)


def build_parser():
    parser = argparse.ArgumentParser(prog="functional_native_export.py", description=__doc__,
                                     formatter_class=_formatter)
    commands = parser.add_subparsers(dest="command", required=True,
                                     metavar="{inspect,verify,pool,selftest}")

    inspect = commands.add_parser(
        "inspect", formatter_class=_formatter,
        help="summarise a plan or a token cache",
        description="Print a JSON summary of exactly one of --plan or --header.")
    group = inspect.add_mutually_exclusive_group(required=True)
    group.add_argument("--plan", help="a functional-native-export/1 plan JSON")
    group.add_argument("--header", help="a token cache header JSON (loaded and verified)")
    inspect.add_argument("--npz", default=None,
                         help="explicit NPZ for --header (default: the plain file name it declares)")

    verify = commands.add_parser(
        "verify", formatter_class=_formatter,
        help="fully verify a token cache",
        description="Load a token cache and re-derive its digests; exit 0 only if all hold.")
    verify.add_argument("--header", required=True, help="token cache header JSON")
    verify.add_argument("--npz", default=None,
                        help="explicit NPZ path (default: the plain file name in the header)")

    pool = commands.add_parser(
        "pool", formatter_class=_formatter,
        help="pool a token cache into a matrix feature cache",
        description=("Pool a verified token cache into a functional-feature-cache/1. The roster "
                     "join is exact; outputs that exist are refused; no overwrite is offered."))
    for flag, text in (("--token-header", "token cache header JSON"),
                       ("--candidates", "candidate CSV: candidate_id, sequence (the full H|L "
                                        "identity), group_id, split, prior_log_weight"),
                       ("--manifest", "functional-matrix-manifest/1 JSON"),
                       ("--output-header", "pooled cache header to write"),
                       ("--output-npz", "pooled cache NPZ to write (same directory, plain name)"),
                       ("--representation-id", "safe id of the pooled representation")):
        pool.add_argument(flag, required=True, help=text)
    pool.add_argument("--token-npz", default=None,
                      help="explicit NPZ of the token cache (default: its declared plain name)")
    pool.add_argument("--pooling", required=True, choices=list(fne.POOLINGS),
                      help="all_nonpad_mean (every real token, specials included: the native "
                           "representation semantics) or residue_mean (residue tokens only)")
    pool.add_argument("--protect", action="append", default=[], metavar="PATH",
                      help="an extra input path to protect from aliasing (repeatable); the "
                           "candidate CSV, manifest and token cache files are always protected")

    selftest = commands.add_parser(
        "selftest", formatter_class=_formatter,
        help="end-to-end check on a tiny random CPU DPLM",
        description=("Export from a real tiny random CPU DPLM, pool two ways, run one small "
                     "matrix, and write native_export_selftest_report.json under a fresh "
                     "directory below --output-dir. Exit status 0 only if every check holds."))
    selftest.add_argument("--output-dir", required=True,
                          help="parent directory; a NEW subdirectory is created inside it and an "
                               "existing directory is never deleted or reused")
    selftest.add_argument("--seed", type=int, default=0,
                          help="seed of the synthetic roster and model (default: 0)")
    return parser


def _print(document):
    print(json.dumps(fb._builtin(document), indent=2, sort_keys=True, allow_nan=False))


def _run_inspect(args):
    if args.plan is not None:
        plan = fne.read_export_plan(args.plan)
        rows = plan.rows
        _print({"kind": "plan", "path": str(plan.path), "representation_id": plan.representation_id,
                "conditioning_mode": plan.conditioning_mode, "activation": plan.activation,
                "extractor": plan.extractor, "feature_supervision": plan.supervision.document(),
                "rows": len(rows), "train_rows": sum(1 for row in rows if row.split == "train"),
                "eval_rows": sum(1 for row in rows if row.split == "eval"),
                "heavy_only_rows": sum(1 for row in rows if not row.light),
                "plan_sha256": plan.fingerprint()})
        return 0
    cache = fne.load_token_cache(args.header, npz_path=args.npz)
    _print({"kind": "token_cache", **cache.summary()})
    return 0


def _run_verify(args):
    cache = fne.load_token_cache(args.header, npz_path=args.npz)
    cache.verify()
    _print({"verified": True, "cache_sha256": cache.cache_sha256,
            "file_sha256": cache.file_sha256, "runtime_sha256": cache.runtime_sha256,
            "token_map_sha256": cache.token_map_sha256, "rows": len(cache.ids),
            "note": "hashes are integrity, not authenticity"})
    return 0


def _run_pool(args):
    cache = fne.load_token_cache(args.token_header, npz_path=args.token_npz)
    manifest = fm.load_matrix_manifest(args.manifest)
    table = fb.read_candidates(args.candidates, alphabet=manifest.alphabet, length=manifest.length)
    summary = fne.write_pooled_matrix_cache(
        cache, table, manifest, args.output_header, args.output_npz,
        representation_id=args.representation_id, pooling=args.pooling,
        protected_input_paths=[args.candidates, args.manifest, *args.protect])
    _print(summary)
    return 0


def _run_selftest(args):
    report = fne.selftest(args.output_dir, seed=args.seed)
    for check in report["checks"]:
        print(f"  [{'ok' if check['holds'] else '!!'}] {check['name']}")
    for cell in report["cell_results"]:
        mse = "n/a" if cell["readout_mse"] is None else f"{cell['readout_mse']:.6g}"
        print(f"  cell {cell['cell_id']:<34} {cell['status']:<12} readout_mse={mse}")
    held = sum(1 for check in report["checks"] if check["holds"])
    print(f"selftest {'PASSED' if report['passed'] else 'FAILED'}: {held}/{len(report['checks'])} "
          f"checks; report at {Path(report['work_directory']) / fne.SELFTEST_REPORT_NAME}")
    print(report["note"])
    return 0 if report["passed"] else 1


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "inspect":
            return _run_inspect(args)
        if args.command == "verify":
            return _run_verify(args)
        if args.command == "pool":
            return _run_pool(args)
        return _run_selftest(args)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
