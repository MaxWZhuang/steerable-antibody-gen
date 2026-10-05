#!/usr/bin/env python
"""Score every route's checkpoints on the shared HER2 development panel -- read-only, CPU-only.

Three commands, run in order (``summarize`` is pure statistics and needs no flight):

    panel      extract the development panel through the production code and save it
    score      score each planned subject (a route's checkpoints + a zero-shot comparator)
    summarize  turn the saved scores into summary.json, per_row_scores.csv and report.md
    selfcheck  prove this interpreter is CPU-guarded (no flight needed)

``--flight NAME=RUN_ROOT`` is the flight's own run root (the directory holding ``launch.json``);
the config, worktree and artifact root are recovered from that launch record. Give every flight
whose routes you want compared, e.g. the bakeoff and the DPLM training phase:

    python scripts/her2_routes_common_eval.py panel --out-dir F:/.../common_eval_1 \\
        --flight t1=F:/.../runs/t1_rev01/flight \\
        --flight dplm=F:/.../runs/dplm_train_rev01/flight_dplm_train

What it guarantees, and how (each is tested; see ``her2_routes_common_eval``):

* It writes nothing into a flight. Services run on a scratch run root seeded with byte-copies of
  the identity markers and a scratch copy of the source cache, and the originals' markers are
  re-hashed afterwards.
* It never touches the GPU. A live training job owns it. The guard is in-process, because on this
  machine ``CUDA_VISIBLE_DEVICES=-1`` makes the torch build crash natively after a CPU backward.
* It never opens the live name of a checkpoint. On Windows an ordinary read handle makes the
  trainer's ``replace`` raise (measured: WinError 5, and a delete-sharing handle does not help),
  so each checkpoint is first pinned with a HARD LINK, which the trainer's replace does not
  block; the link is read and hashed, then removed. Put ``--out-dir`` on the same NTFS volume as
  the flight (F:) so links are possible.
* It opens no reserved outcome. The development panel is selection data, so the numbers are NOT
  a held-out result.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# A read-only auditor writes no bytecode into any source tree it imports from.
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smallAntibodyGen.experiments import her2_routes_common_eval as ce  # noqa: E402  (no torch)


def parse_flight(text):
    """``NAME=RUN_ROOT`` -> ``(name, Path)``."""
    name, separator, run_root = str(text).partition("=")
    if not separator or not name or not run_root or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError(f"--flight expects NAME=RUN_ROOT with a filename-safe NAME, got {text!r}")
    return name, Path(run_root)


def build_parser():
    parser = argparse.ArgumentParser(prog="her2_routes_common_eval.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    def common(sub, *, flights):
        sub.add_argument("--out-dir", required=True,
                         help="a NEW directory of the evaluator's own (put it on F:, not C:)")
        if flights:
            sub.add_argument("--flight", action="append", required=True, metavar="NAME=RUN_ROOT")
            sub.add_argument("--scratch-root", default=None,
                             help="default: <out-dir>/scratch. Temporary checkpoint copies, the "
                                  "scratch run roots and the scratch source cache")
            sub.add_argument("--threads", type=int, default=2,
                             help="torch CPU threads (default 2; the trainer needs the rest)")

    panel = commands.add_parser("panel", help="extract and save the shared development panel")
    common(panel, flights=True)
    score = commands.add_parser("score", help="score each flight's planned subjects")
    common(score, flights=True)
    score.add_argument("--route", action="append", default=None,
                       help="restrict to a route (repeatable); default: every supported route")
    score.add_argument("--labels", default=None,
                       help="comma-separated: best,last,stepN,zero_shot. Default: every "
                            "checkpoint plus zero_shot; an explicit label that does not exist "
                            "is an error")
    score.add_argument("--allow-live-checkpoints", action="store_true",
                       help="ONLY matters when a hard-link snapshot is impossible (the scratch "
                            "directory is on another volume): then read a still-training "
                            "route's file with an ordinary handle anyway. That can make the "
                            "trainer's atomic replace raise (WinError 5) and kill a run "
                            "launched with --max-restarts 0. Without this flag such a "
                            "checkpoint is refused")
    score.add_argument("--batch-size", type=int, default=4)
    summarize = commands.add_parser("summarize", help="statistics, CSV and report from saved scores")
    common(summarize, flights=False)
    commands.add_parser("selfcheck", help="prove this interpreter is CPU-guarded")
    return parser


def selfcheck():
    """A fresh-interpreter proof: CUDA unreachable, never initialised, CPU autograd works."""
    import torch
    with ce.forbid_cuda():
        x = torch.randn(8, 8, requires_grad=True)
        (x @ x).sum().backward()
        report = {"guard_active": bool(torch.cuda.device_count() == 0
                                       and not torch.cuda.is_available()),
                  "cuda_device_count": int(torch.cuda.device_count()),
                  "cpu_backward_ok": x.grad is not None, "torch": torch.__version__,
                  "threads": int(torch.get_num_threads())}
    report["cuda_initialized"] = bool(torch.cuda.is_initialized())
    print(json.dumps(report, sort_keys=True))
    return 0


def prepare_environment(flights, scratch_root):
    """Point every model cache and temp directory at the scratch root BEFORE torch is imported.

    The pinned native dependencies (``einops``, the AbLang2 package) live under the flights'
    artifact root and are added to ``sys.path`` from there, read-only. Every flight must share
    one artifact root.
    """
    from smallAntibodyGen.experiments import her2_routes_env as env
    roots = {flight.artifact_root for flight in flights}
    if len(roots) != 1:
        raise ValueError(f"the flights name different artifact roots {sorted(map(str, roots))}")
    artifact = roots.pop()
    record = env.prepare_environment(
        scratch_root, dependency_paths=[artifact / env.DEPENDENCY_ROOT,
                                        artifact / env.NATIVE_PACKAGE_ROOT])
    env.require_environment_ordering(record, label="common evaluation")
    return record


def log(message):
    print(message, flush=True)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "selfcheck":
        return selfcheck()
    if args.command == "summarize":
        summary = ce.summarize(args.out_dir)
        print(f"wrote summary.json, per_row_scores.csv and report.md in {args.out_dir} "
              f"({len(summary['subjects'])} subjects, {len(summary['contrasts'])} contrasts)")
        return 0
    named = [parse_flight(text) for text in args.flight]
    out_dir = Path(args.out_dir).resolve()
    scratch_root = Path(args.scratch_root or out_dir / "scratch").resolve()
    # Nothing below may create or read anything until the destinations are known to be outside
    # every flight. Capturing a flight creates pin directories (under <out-dir>/pins), and
    # environment preparation creates <scratch>/tmp, so the output directory is made recognisably
    # ours (an empty spec) first.
    ce.refuse_destinations_near_flights([run_root for _name, run_root in named],
                                        [out_dir, scratch_root])
    ce.ensure_output_directory(out_dir, [])
    flights = [ce.flight_from_run_root(name, run_root, pin_parent=out_dir / "pins")
               for name, run_root in named]
    ce.validate_destinations(flights, out_dir, scratch_root)
    ce.ensure_output_directory(out_dir, flights)
    prepare_environment(flights, scratch_root)
    with ce.forbid_cuda():
        if args.command == "panel":
            panels = ce.run_panel(flights, out_dir, scratch_root, threads=args.threads, log=log)
            print(f"saved {len(panels)} panel file(s); digest {panels[0]['digest'][:12]}")
        else:
            labels = None if args.labels is None else [
                label.strip() for label in args.labels.split(",") if label.strip()]
            result = ce.run_score(flights, out_dir, scratch_root, routes=args.route,
                                  labels=labels, allow_live=args.allow_live_checkpoints,
                                  batch_size=args.batch_size, threads=args.threads, log=log)
            print(f"scored {len(result['scored'])} subject(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
