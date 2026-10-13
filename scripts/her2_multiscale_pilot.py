#!/usr/bin/env python
"""Multiscale functional adaptation pilot: one arm of mixed whole-window masks and measured pairs.

    .venv/Scripts/python.exe scripts/her2_multiscale_pilot.py \\
        --frozen-inputs <artifact root>/runs/dplm_train_rev01/flight_dplm_train \\
        --data-repo-root <data repository> --run-root <fresh directory> --arm combined

Trains the native DPLM 150M route from its pinned weights for the config's bounded number of
updates, on the base config's four-task, three-family stream, with one of four arms:
``baseline`` (suffix masks, distance-1 pairs), ``masks`` (subset masks, distance-1 pairs), ``pairs``
(suffix masks, distances 1-3) and ``combined`` (subset masks, distances 1-3). The four arms are
comparisons inside this pilot, not a replay of an earlier flight, and a finished run is evidence about
the training signal only -- it does not claim the adaptation improves biology.

It reads the production flight's frozen split, condition vocabulary and source inventory
(``--frozen-inputs``) and restores them; it never freezes, splits, acquires or downloads. The
run root must not exist: it is created exclusively, before anything else, and must be disjoint from
the code, data, artifact and frozen-input roots AND from every data and historical location the base
config declares (the historical flight directories are junctions to other volumes, so they are
resolved first). Everything the run writes is under it, including the parsed-source caches, the
model caches and ``runtime.json`` (the prepared environment and the interpreter and library
versions actually imported). Pinned weights that are missing are fatal.

``prepare_environment`` runs before torch is imported. ``--help`` creates nothing and imports no
scientific module.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Bytecode must not land in a ``__pycache__`` beside the code a run is reading; this is set before
# any project import, as in the other entry scripts (tests/test_her2_routes_entry_scripts.py).
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_CONFIG = "configs/experiments/her2_multiscale_pilot.json"
ARMS = ("baseline", "masks", "pairs", "combined")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="her2_multiscale_pilot.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="pilot configuration (default: %(default)s)")
    parser.add_argument("--arm", choices=ARMS, default=None,
                        help="override the config's arm; recorded in pilot_config.json")
    parser.add_argument("--frozen-inputs", required=True,
                        help="the production flight directory holding hard_split_freeze.json, "
                             "condition_vocabulary.json and source_inventory.json (read-only)")
    parser.add_argument("--data-repo-root", required=True,
                        help="the existing data repository the declared data paths live under")
    parser.add_argument("--artifact-root", default=None,
                        help="the existing artifact volume (default: the base config's)")
    parser.add_argument("--run-root", required=True,
                        help="a fresh, nonexistent, absolute output directory")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    # Nothing above this line has created a directory or imported a scientific module. The next
    # three imports are standard-library only.
    from smallAntibodyGen.experiments import her2_routes_env as env
    from smallAntibodyGen.experiments.her2_multiscale import config as config_lib
    from smallAntibodyGen.experiments.her2_multiscale import roots as roots_lib
    config_path = Path(args.config)
    pilot = config_lib.load_pilot_config(
        config_path if config_path.is_absolute() else ROOT / config_path, code_root=ROOT)
    if args.arm is not None:
        pilot = pilot.with_arm(args.arm)
    base = config_lib.read_json_strict(pilot.base_path)
    artifact_root = Path(args.artifact_root or base["artifacts"]["root"])
    # The historical flights and data directories the base config declares are resolved (they can be
    # junctions to volumes outside the four roots) and protected BEFORE the run root is made.
    declared = roots_lib.declared_input_roots(base, data_repo_root=Path(args.data_repo_root))
    roots = roots_lib.reserve_run_root(
        args.run_root, code_root=ROOT, data_repo_root=Path(args.data_repo_root),
        artifact_root=artifact_root, frozen_inputs=Path(args.frozen_inputs),
        declared_inputs=declared)
    record = env.prepare_environment(
        roots.runtime_root, dependency_paths=[artifact_root / "python_deps",
                                              artifact_root / "upstream" / "ablang2"],
        offline=True)
    env.require_environment_ordering(record, label="multiscale pilot")
    from smallAntibodyGen.experiments.her2_multiscale import runner
    # The environment record is handed on so the run persists it (runtime.json) beside the code
    # identity; it is not a byte pin of the installed dependencies.
    result = runner.run_pilot(pilot, roots, device=args.device, environment=record, log=print)
    print(json.dumps({"status": result["status"], "arm": result["arm"],
                      "run_root": str(roots.run_root), "snapshot": result["snapshot"]["path"]}))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ValueError as error:
        print(f"refused: {error}", file=sys.stderr)
        sys.exit(2)
