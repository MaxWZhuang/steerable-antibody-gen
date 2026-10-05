#!/usr/bin/env python
"""Load, qualify and inventory every enrolled source. CPU only; no model is built.

Runs before training and answers one question per source: how many rows are in the
file, how many survived qualification, what was excluded and why. Read-only on every
input.

The qualification decisions this prints are the ones that matter most:

* ``Pred_affinity`` is enrolled as a MEASURED log10(KD/nM) column despite its name;
* Koenig is enrolled as ENRICHMENT, and its KD-named columns train ordering only;
* Adams is doubly censored at the primary paper's detection window, and the
  ``format=Fab`` curation label is carried beside the primary scFv experiment rather
  than resolved;
* a blank outcome is a missing number and is counted, never converted to a negative;
* the CR9114 H1 columns are refused structurally, not filtered afterwards.
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


def build_parser():
    parser = argparse.ArgumentParser(prog="her2_routes_prepare_sources.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output", default=None, help="write the inventory here")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config_path = ROOT / args.config
    document = spec.load_config(config_path)
    artifact_root = args.artifact_root or document["artifacts"]["root"]
    ordering = env.prepare_environment(artifact_root)
    env.require_environment_ordering(ordering, label="prepare_sources")

    from smallAntibodyGen.experiments import her2_routes_campaign as campaign
    context = spec.resolve_context(ROOT, config_path=config_path, artifact_root=artifact_root,
                                   revision=args.revision)
    services = campaign.build_services(context, device="cpu")
    inventory = services.build_sources()
    target = Path(args.output) if args.output else context.path("source_inventory.json")
    paths.write_json(target, inventory)
    for row in inventory["sources"]:
        print(f"{row['source']:<32} rows={row['rows_in_file']!s:>8} "
              f"observations={row['observations']!s:>8} "
              f"unique={row['unique_constructs']!s:>8} "
              f"missing={row['missing_outcome']!s:>6} "
              f"enrolled={row['enrolled']}", flush=True)
    print(f"\nclosure: {inventory['closure']['kept']} observations, "
          f"{inventory['closure']['duplicates']} duplicates collapsed, "
          f"{inventory['closure']['unique_constructs']} unique constructs, "
          f"{inventory['closure']['constructs_in_multiple_sources']} appearing in more than one "
          f"source", flush=True)
    print(f"measurement kinds: {inventory['measurement_kinds']['by_kind']}", flush=True)
    print(f"censoring: {inventory['measurement_kinds']['by_censoring']}", flush=True)
    print(f"bins assigned: {inventory['bins_assigned']}", flush=True)
    print(f"\nwrote {target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
