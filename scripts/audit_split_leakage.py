#!/usr/bin/env python
"""
Read-only leakage audit of the PROCESSED corpora, within and across stages.

This exists because a split's validity is a property of the generated artifact,
not of the code that generated it. `scripts/prepare_oas.py` was corrected after
`data/processed/oas_paired_all/` was written, and a code fix does not rewrite a
corpus -- so the only way to know what a checkpoint's validation split actually
measures is to read the file the checkpoint trained on.

What it measures, per corpus and across the two:

- **exact record overlap** -- a val row whose full encoded sequence appears in
  train. This is the strict definition of a leaked evaluation row.
- **component overlap** -- a val row that SHARES a chain (heavy or light) or an
  HCDR3 with some training row without being identical to it. This is what the
  pair-connected-component split (BUGLOG Mirror BUG-03 + BUG-22) exists to
  prevent, and it is invisible to exact-record deduplication.
- **cross-stage overlap** -- a stage-2 validation component that stage 1 already
  trained on. Stage 1 is unsupervised, so this is not label leakage; it IS
  contamination the moment anyone claims a stage-2 validation antibody was
  unseen by the model, because the model is warm-started from stage 1.

Nothing here is a pass/fail gate. The numbers decide which CLAIM the splits can
support, and that is an owner decision -- a component-sharing split is perfectly
valid for "pairing a known component", and invalid for "generating a novel
antibody". The audit's job is to make the choice explicit rather than implicit.

Memory: sequences are stored as 8-byte BLAKE2b digests rather than strings, so
the two corpora fit in a few hundred MB instead of several GB. Collisions at
these set sizes are ~1e-8 and would only ever OVERSTATE overlap, never hide it.

Usage:
    python scripts/audit_split_leakage.py [--output-json specs/evidence/split-leakage-audit.json]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "split-leakage-audit/1"

UNPAIRED = PROJECT_ROOT / "data/processed/oas_unpaired_3m/oas_all.jsonl.gz"
PAIRED = PROJECT_ROOT / "data/processed/oas_paired_all/oas_paired.jsonl.gz"


def digest(text: str | None) -> bytes | None:
    """8-byte content digest, or None for an absent field."""
    if not text:
        return None
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()


def _field(record: dict[str, Any], key: str | tuple[str, ...]) -> str | None:
    """Read one field, or the concatenation of several (a full VH/VL pair)."""
    if isinstance(key, tuple):
        parts = [record.get(k) for k in key]
        return "".join(parts) if all(parts) else None
    return record.get(key)


def read_records(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def file_sha256(path: Path) -> str:
    """Bind the audit to the exact bytes it read; a corpus is regenerable."""
    digest_ = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest_.update(chunk)
    return digest_.hexdigest()


def collect(path: Path, fields: dict[str, str]) -> dict[str, Any]:
    """
    One pass over a corpus, bucketing each requested field by split.

    Args:
        path: processed JSONL(.gz).
        fields: label -> record key to digest.

    Returns:
        ``{"train": {label: set[bytes]}, "val": {label: [bytes|None]}, ...}``.
        Train is a SET (membership only); val is a LIST because the denominator
        is rows, not distinct values -- collapsing val to a set would silently
        change what the percentage means.
    """
    train = {label: set() for label in fields}
    val = {label: [] for label in fields}
    counts = {"train": 0, "val": 0, "other": 0}

    for record in read_records(path):
        split = record.get("split")
        if split == "train":
            counts["train"] += 1
            for label, key in fields.items():
                value = digest(_field(record, key))
                if value is not None:
                    train[label].add(value)
        elif split == "val":
            counts["val"] += 1
            for label, key in fields.items():
                val[label].append(digest(_field(record, key)))
        else:
            counts["other"] += 1

    return {"train": train, "val": val, "counts": counts}


def overlap(val_values: list[bytes | None], train_set: set[bytes]) -> dict[str, Any]:
    """Val ROWS (not distinct values) whose component appears in `train_set`."""
    present = [v for v in val_values if v is not None]
    hits = sum(1 for v in present if v in train_set)
    total = len(present)
    return {
        "val_rows_with_field": total,
        "val_rows_seen_in_train": hits,
        "fraction": (hits / total) if total else None,
        "distinct_val_values": len(set(present)),
        "distinct_train_values": len(train_set),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "specs/evidence/split-leakage-audit.json",
    )
    args = parser.parse_args()

    for path in (UNPAIRED, PAIRED):
        if not path.exists():
            print(f"REFUSED: missing corpus {path}")
            return 1

    print("reading stage-1 (unpaired) corpus ...")
    stage1 = collect(UNPAIRED, {"sequence": "variable_aa", "hcdr3": "cdr3_aa"})
    print(f"  train={stage1['counts']['train']:,}  val={stage1['counts']['val']:,}")

    print("reading stage-2 (paired) corpus ...")
    # NOT `variable_aa`. In the paired corpus that field holds the HEAVY CHAIN
    # ONLY (verified 2026-08-29: 100% of a 400,000-record sample, median length
    # 122), even though `length` records the paired length. Auditing it silently
    # reproduces the heavy-chain number under a "full sequence" label -- which is
    # exactly the mistake this comment exists to stop. The full pair is built
    # from the two fields the collator itself encodes
    # (`MLMCollator._encode_record`), so the audit and the training path agree.
    stage2 = collect(
        PAIRED,
        {
            "full_pair": ("sequence_heavy", "sequence_light"),
            "heavy": "sequence_heavy",
            "light": "sequence_light",
            "hcdr3": "cdr3_aa_heavy",
        },
    )
    print(f"  train={stage2['counts']['train']:,}  val={stage2['counts']['val']:,}")

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "corpora": {
            "stage1_unpaired": {
                "path": UNPAIRED.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": file_sha256(UNPAIRED),
                "rows": stage1["counts"],
            },
            "stage2_paired": {
                "path": PAIRED.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": file_sha256(PAIRED),
                "rows": stage2["counts"],
            },
        },
        "within_stage1": {
            "exact_sequence": overlap(stage1["val"]["sequence"], stage1["train"]["sequence"]),
            "hcdr3": overlap(stage1["val"]["hcdr3"], stage1["train"]["hcdr3"]),
        },
        "within_stage2": {
            "exact_full_pair": overlap(stage2["val"]["full_pair"], stage2["train"]["full_pair"]),
            "heavy_chain": overlap(stage2["val"]["heavy"], stage2["train"]["heavy"]),
            "light_chain": overlap(stage2["val"]["light"], stage2["train"]["light"]),
            "hcdr3": overlap(stage2["val"]["hcdr3"], stage2["train"]["hcdr3"]),
        },
        # Stage 1 is unsupervised, so this is not label leakage. It IS
        # contamination for any claim that a stage-2 validation antibody was
        # unseen, because stage 2 is warm-started from stage 1's weights.
        "stage2_val_seen_in_stage1_train": {
            "heavy_chain": overlap(stage2["val"]["heavy"], stage1["train"]["sequence"]),
            "light_chain": overlap(stage2["val"]["light"], stage1["train"]["sequence"]),
            "hcdr3": overlap(stage2["val"]["hcdr3"], stage1["train"]["hcdr3"]),
        },
    }

    def show(title: str, block: dict[str, Any]) -> None:
        print(f"\n{title}")
        for name, stats in block.items():
            frac = stats["fraction"]
            pct = "n/a" if frac is None else f"{frac:7.2%}"
            print(
                f"  {name:16} {stats['val_rows_seen_in_train']:>9,} / "
                f"{stats['val_rows_with_field']:>9,}  {pct}"
            )

    show("WITHIN STAGE 1 -- val rows whose component appears in stage-1 train",
         report["within_stage1"])
    show("WITHIN STAGE 2 -- val rows whose component appears in stage-2 train",
         report["within_stage2"])
    show("CROSS-STAGE -- stage-2 val components stage 1 already trained on",
         report["stage2_val_seen_in_stage1_train"])

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8", newline="") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"\nwrote {args.output_json.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
