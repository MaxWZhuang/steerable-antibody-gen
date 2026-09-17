#!/usr/bin/env python
"""
Prepare ESM-IF1 encoder inputs from a declared local structure file.

Reads one structural-input manifest, reads the local PDB/mmCIF file it pins by
SHA-256, and writes two files: a portable prepared artifact (the packed N/CA/C
coordinates, the confidence vector, the validated manifest, and digests over all
of them) and a deterministic preparation report.

Local files only. Nothing is fetched and no weights are loaded, so every
model-integration check in the report is recorded as NOT RUN with a reason rather
than as a pass.

The report's structural checks are re-verified while the report is built: the
manifest file is re-read and re-hashed and the structure file is re-read and the
whole preparation re-run, so a report that says "pass" is one whose checks have
just run against the files it names.

Nothing is ever overwritten. There is no --force: an existing output is an error,
the inputs are never written to, and any file this run created is removed again
if a later step fails. Failures print one actionable line on stderr and exit 2.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from smallAntibodyGen.structure import (  # noqa: E402
    StructureAdapterError,
    build_report,
    load_manifest,
    prepare_structure,
    prepared_to_document,
    structure_path_for,
    write_outputs,
)

EXAMPLE = """\
example. The fixture writer below generates a clearly SYNTHETIC toy complex --
a straight line of CA atoms 3.8 A apart, with no side chains and no chemistry.
It is not a deposited structure and nothing measured from it is evidence about
any protein:

  python src/smallAntibodyGen/tests/fixtures_esmif1_structure.py outputs/scratch/toy

  python scripts/prepare_esmif1_structure.py \\
      --manifest outputs/scratch/toy/toy.manifest.json \\
      --output-dir outputs/scratch/toy/prepared \\
      --name toy
"""


def display(path: Path) -> str:
    """Repo-relative when possible, absolute otherwise -- never a crash."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.strip().splitlines()[0],
        epilog=EXAMPLE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="structural-input manifest JSON (schema esmif1-structure-manifest/1)",
    )
    parser.add_argument(
        "--structure-root",
        type=Path,
        default=None,
        help="directory the manifest's relative_path is resolved against "
             "(default: the manifest's own directory)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory to write <name>.prepared.json and <name>.report.json into; "
             "created if absent, never written over",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="output file stem; must be a plain name with no path separators",
    )
    args = parser.parse_args(argv)

    try:
        manifest = load_manifest(args.manifest)
        structure_root = (
            args.structure_root
            if args.structure_root is not None
            else args.manifest.resolve().parent
        )
        structure_path = structure_path_for(manifest, structure_root)
        prepared = prepare_structure(manifest, structure_path)

        artifact_document = prepared_to_document(prepared)
        report = build_report(
            prepared,
            manifest_path=args.manifest,
            structure_path=structure_path,
            artifact_filename=f"{args.name}.prepared.json",
            artifact_content_sha256=artifact_document["identity"]["content_sha256"],
        )
        artifact_path, report_path = write_outputs(
            artifact_document,
            report,
            output_dir=args.output_dir,
            name=args.name,
            input_paths=(args.manifest, structure_path),
        )
    except (StructureAdapterError, OSError) as error:
        # Expected input failures only: every refusal this package raises, plus the
        # OS errors a path argument can cause (an unreadable directory, a name the
        # filesystem rejects). A programming error is not caught here on purpose --
        # it should print its traceback.
        print(f"error: {error}", file=sys.stderr)
        return 2

    span = prepared.decoded_span
    print(
        f"prepared {display(structure_path)} "
        f"(model ordinal {manifest.source.model_ordinal}, "
        f"chains {'+'.join(manifest.chains.order)})"
    )
    print(
        f"  decoded chain {span.chain_id}: {span.num_residues} residues at rows "
        f"0..{span.end_row - 1} of {prepared.num_rows} packed rows"
    )
    print(
        f"  sites declared {list(manifest.declared_site_ids)} -> policy order "
        f"{list(manifest.policy_site_ids)}"
    )
    if prepared.findings.numbering_gaps:
        print(
            f"  finding: {len(prepared.findings.numbering_gaps)} author-numbering gap(s) "
            "in the decoded chain; reported only, no spatial or completeness claim"
        )
    if prepared.findings.declared_sequence_mismatches:
        print(
            f"  finding: {len(prepared.findings.declared_sequence_mismatches)} declared "
            "sequence mismatch(es) between the structural template and the edited sequence"
        )
    print(f"  wrote {display(artifact_path)}")
    print(f"  wrote {display(report_path)}")
    print("  structural checks: re-verified against the named manifest and structure")
    print("  model integration checks: NOT RUN (no model supplied, no weights loaded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
