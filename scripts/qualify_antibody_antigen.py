#!/usr/bin/env python
"""Qualify frozen native MINT features on explicit exact-pKD observations.

Full run (explicit local paths; nothing is downloaded or discovered):

    .venv/Scripts/python.exe scripts/qualify_antibody_antigen.py \\
        --config <config.json> --manifest <manifest.json> --run-root <fresh directory> \\
        --mint-source <pinned upstream checkout> --mint-checkpoint <pinned checkpoint file> [--device cpu]

Dry run: loads the config and manifest, checks the split, applies the shared length cap and prints the
masked-diagnostic plan and its exact forward budget as JSON. It imports no torch, reads no model files and
writes nothing (``--run-root`` / ``--mint-*`` are ignored); it exits 2 if the plan exceeds ``max_forwards``:

    .venv/Scripts/python.exe scripts/qualify_antibody_antigen.py \\
        --config <config.json> --manifest <manifest.json> --admission-only

A full run compares frozen pinned-MINT features of three controls (joint antibody-antigen, antibody-only,
independently encoded antibody + antigen) through one fixed train-only ridge, and scores visible-only masked
likelihood diagnostics with the joint and antibody-only views. It trains, distils and fuses nothing, and a
finished run asserts no scientific qualification: the report's scientific_status stays unestablished /
unverified / not_audited / not_assessed. The run root must not exist; it is reserved atomically after every
admission, budget and path check, and must be disjoint from the code checkout and from the config, manifest,
records, MINT source and checkpoint directories.

``--help`` creates nothing and imports no scientific module.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Bytecode must not land beside the code a run is reading; set before any project import.
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def build_parser():
    parser = argparse.ArgumentParser(
        prog="qualify_antibody_antigen.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="closed-schema qualification config (aa-qualification/1)")
    parser.add_argument("--manifest", required=True, help="explicit panel manifest (aa-manifest/1); the only data input")
    parser.add_argument("--run-root", default=None, help="a fresh, nonexistent, absolute output directory (full run)")
    parser.add_argument("--mint-source", default=None, help="the pinned upstream MINT checkout, a local directory")
    parser.add_argument("--mint-checkpoint", default=None, help="the pinned MINT checkpoint, a local file")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"), help="default: %(default)s")
    parser.add_argument(
        "--admission-only",
        action="store_true",
        help="print the admission report and the planned forward budget; no torch, no model files, no writes",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    # Nothing above this line has created a file or imported a scientific module.
    if args.admission_only:
        from smallAntibodyGen.experiments.interaction_qualification import runner

        pf = runner.preflight(args.config, args.manifest)
        sys.stdout.write(runner.canonical_text(runner.admission_report(pf)))
        if not pf.plan.budget.within_budget:
            runner.require_budget(pf)
        return 0
    missing = [
        flag
        for flag, value in (
            ("--run-root", args.run_root),
            ("--mint-source", args.mint_source),
            ("--mint-checkpoint", args.mint_checkpoint),
        )
        if value is None
    ]
    if missing:
        parser.error(f"a full run needs explicit local paths: missing {', '.join(missing)}")
    from smallAntibodyGen.experiments.interaction_qualification import runner

    result = runner.run_qualification(
        args.config,
        args.manifest,
        args.run_root,
        mint_source=args.mint_source,
        mint_checkpoint=args.mint_checkpoint,
        device=args.device,
        code_root=ROOT,
    )
    status = result["status"]
    print(
        json.dumps(
            {
                "state": status["state"],
                "run_root": result["run_root"],
                "report_sha256": status["report_sha256"],
                "released_execution": status["execution"]["released_execution"],
                "scientific_status": status["scientific_status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ValueError as error:
        print(f"refused: {error}", file=sys.stderr)
        sys.exit(2)
    except RuntimeError as error:
        # The pinned loader's own refusal (source/checkpoint do not match the pins); anything else is a bug.
        if type(error).__name__ != "MintLoadError":
            raise
        print(f"refused: {error}", file=sys.stderr)
        sys.exit(2)
