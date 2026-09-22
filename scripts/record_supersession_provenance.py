#!/usr/bin/env python
"""Record that a completed campaign now verifies against superseded sources.

A freeze pins exact source bytes, and the records a campaign wrote say things
like "reproduces byte for byte from the frozen sources". When a pinned source is
later migrated under a reviewed supersession, those words stop being exact: the
artifacts are unchanged and the reproduction is real, but it ran against the
reviewed SUCCESSORS of those files.

Rewriting the finished records to say so would be worse than leaving them. They
are write-once on purpose, and a completed artifact that changes whenever it is
re-verified is no longer evidence of when it was made -- several of them also pin
the sha256 of the very tool that wrote them, so editing that tool to add a
sentence is itself a change of identity. This writes a SEPARATE record beside
them instead, naming what it qualifies.

Usage::

    python scripts/record_supersession_provenance.py --run-dir outputs/her2_support_audit_20260919_r2
    python scripts/record_supersession_provenance.py --run-dir <dir> --check
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smallAntibodyGen.experiments import her2_support as support  # noqa: E402

RECORD = "supersession_provenance.json"

#: Each campaign family names its freeze marker differently and nests the pinned
#: source digests differently. Both are read here so one tool covers both.
MARKERS = (
    ("audit_spec_frozen.json", lambda m: m.get("source_sha256") or {}),
    ("training_spec_frozen.json", lambda m: (m.get("source") or {}).get("sha256") or {}),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def find_marker(run_dir: Path):
    for name, extract in MARKERS:
        candidate = run_dir / name
        if candidate.is_file():
            return candidate, extract
    raise SystemExit(f"no freeze marker in {run_dir}; nothing to qualify")


def accepted_supersessions(root: Path, run_dir: Path):
    marker_path, extract = find_marker(run_dir)
    pinned = extract(json.loads(marker_path.read_text(encoding="utf-8")))
    table = support.load_source_supersessions(root)
    accepted, unexplained = [], []
    for logical, expected in sorted(pinned.items()):
        target = root / logical
        observed = sha256_file(target) if target.is_file() else None
        if observed == expected:
            continue
        record = support.classify_source_drift(root, logical, expected, observed, table=table)
        (accepted if record is not None else unexplained).append(record or {
            "file": logical, "expected": expected, "observed": observed})
    return marker_path, accepted, unexplained


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="a completed campaign run directory holding a freeze marker")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--check", action="store_true",
                        help="report without writing")
    args = parser.parse_args(argv)

    marker_path, accepted, unexplained = accepted_supersessions(args.root, args.run_dir)
    if unexplained:
        print("These pinned sources differ and are NOT covered by a recorded supersession:",
              file=sys.stderr)
        for row in unexplained:
            print(f"  {row['file']}", file=sys.stderr)
        print("The campaign does not verify at all in this tree; that is a separate problem "
              "from provenance wording.", file=sys.stderr)
        return 1

    if not accepted:
        print(f"{args.run_dir}: every pinned source still matches its frozen bytes; "
              "there is nothing to qualify")
        return 0

    target = args.run_dir / RECORD
    document = {
        "schema_version": "1",
        "record_kind": "supersession_provenance",
        "run_directory": str(args.run_dir).replace("\\", "/"),
        "freeze_marker": marker_path.name,
        "qualifies": ("every record in this run directory that describes its provenance as the "
                      "frozen sources, including the report compatibility record"),
        "superseded_sources": accepted,
        "statement": (
            f"{len(accepted)} source file(s) pinned by this campaign's freeze have since been "
            f"migrated under reviewed supersessions recorded in {support.SOURCE_SUPERSESSIONS}. "
            "Verification and byte-for-byte reproduction in this tree therefore run against the "
            "reviewed SUCCESSORS of those files, not against the frozen bytes themselves. The "
            "measured artifacts are unchanged and the reproductions are real. The finished "
            "records in this directory predate the migration and are deliberately left exactly "
            "as they were written; this record is the qualification, not a correction."),
    }
    if args.check:
        print(json.dumps(document, indent=2))
        return 0
    target.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"qualified {len(accepted)} superseded source file(s) -> {target}")
    for row in accepted:
        print(f"  {row['file']}  {row['superseded_sha256'][:12]} -> {row['current_sha256'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
