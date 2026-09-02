#!/usr/bin/env python
"""
Find real A-B-C similarity chains that are OPEN at a given operating point.

WHY THIS SCRIPT EXISTS
----------------------
`src/smallAntibodyGen/tests/test_target_identity_acceptance.py` ships one
anti-percolation test carrying a prominent powerlessness warning: at the
thresholds an implementation is likely to choose, the three influenza-B
haemagglutinins it uses form no construct edge at all, so the assertion
``construct(A) != construct(C)`` holds trivially and the cluster criterion is
never exercised. The warning says what the repair is -- rewrite the test against
a fixture that closes a triangle AT THE OPERATING THRESHOLDS and assert
positively that the A~B and B~C edges exist.

That repair needs a real triple, and the honest way to get one is to look for it
rather than to move the operating point until the existing fixture works. This
script scans every distinct antigen sequence in the raw shards and reports every
triple (A, B, C) such that, at the given thresholds:

    A ~ B  admitted
    B ~ C  admitted
    A ~ C  REFUSED

Those are exactly the triples where single-linkage and a bounded-diameter
criterion give different answers, so a fixture built from one has power by
construction -- and the script prints the measured identity, coverage and overlap
of all three edges so the fixture can pin them.

Nothing here decides anything. It emits candidates and their measurements; which
triple becomes a fixture, and with what provenance, is a human decision recorded
in the fixture module.

USAGE
-----
    python scripts/scan_anti_percolation_triples.py \
        --identity 0.99 --coverage 0.95 --overlap 30 \
        --output outputs/anti-percolation-triples-construct.json

The first run over the 20 shards takes a few minutes to extract the antigen
universe; ``--cache`` stores it so later runs at other thresholds are quick.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from smallAntibodyGen.entity_resolution.alignment import (  # noqa: E402
    AlignmentTooLarge,
    align_pair,
)
from smallAntibodyGen.entity_resolution.blocking import CandidateIndex  # noqa: E402


def extract_antigen_universe(shard_dir: Path) -> Dict[str, Dict[str, object]]:
    """Read every shard and return the distinct antigen sequences with annotations.

    Identity is observed over EVERY shard row, including rows a downstream filter
    will drop, because a curator writing an accession on a row is evidence about
    the target whether or not that particular antibody survives filtering.

    Args:
        shard_dir: Directory of ``*.parquet`` shards.

    Returns:
        Mapping from antigen sequence to ``{"rows", "annotations"}`` where each
        annotation is a ``(name, pdb, uniprot)`` triple as the producer
        normalises them.
    """
    import pandas as pd

    scripts_dir = REPO_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import prepare_antibody_antigen as paa

    universe: Dict[str, Dict[str, object]] = {}
    shards = sorted(shard_dir.glob("*.parquet"))
    if not shards:
        raise SystemExit(f"no parquet shards under {shard_dir}")
    for shard in shards:
        frame = pd.read_parquet(shard, columns=["antigen_sequence", "metadata"])
        for raw_sequence, raw_metadata in zip(
            frame["antigen_sequence"], frame["metadata"]
        ):
            sequence = paa.clean_aa_sequence(raw_sequence)
            if not sequence:
                continue
            entry = universe.setdefault(sequence, {"rows": 0, "annotations": set()})
            entry["rows"] += 1
            fields = paa.extract_target_fields(paa.normalize_metadata_dict(raw_metadata))
            entry["annotations"].add((
                paa.normalize_target_name(fields["target_name"]),
                paa.canonicalize_accession(fields["target_pdb"]),
                paa.canonicalize_accession(fields["target_uniprot"]),
            ))
        print(f"  {shard.name}: {len(universe)} distinct", file=sys.stderr, flush=True)
    return universe


def load_universe(shard_dir: Path, cache: Path | None) -> Dict[str, Dict[str, object]]:
    """Load the antigen universe, using a JSON cache when one is available."""
    if cache is not None and cache.exists():
        payload = json.loads(cache.read_text())
        return {
            entry["sequence"]: {
                "rows": entry["rows"],
                "annotations": {tuple(a) for a in entry["annotations"]},
            }
            for entry in payload
        }
    universe = extract_antigen_universe(shard_dir)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps([
            {
                "sequence": sequence,
                "rows": entry["rows"],
                "annotations": sorted(entry["annotations"]),
            }
            for sequence, entry in sorted(universe.items())
        ]))
    return universe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path,
                        default=REPO_ROOT / "data" / "raw" / "asd-antibody-antigen")
    parser.add_argument("--cache", type=Path,
                        default=REPO_ROOT / "outputs" / "antigen-universe.json")
    parser.add_argument("--identity", type=float, required=True)
    parser.add_argument("--coverage", type=float, required=True)
    parser.add_argument("--overlap", type=int, default=30)
    parser.add_argument("--max-cells", type=int, default=50_000_000)
    parser.add_argument("--limit", type=int, default=40,
                        help="Most triples to report, best-separated first.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.time()
    universe = load_universe(args.shards, args.cache)
    print(f"universe: {len(universe)} distinct antigens "
          f"({time.time() - started:.0f}s)", file=sys.stderr)

    # Digest keys keep the index deterministic and the report readable.
    import hashlib

    def digest(sequence: str) -> str:
        return hashlib.sha256(sequence.encode("utf-8")).hexdigest()[:32]

    sequences = {digest(sequence): sequence for sequence in universe}
    annotations = {digest(s): sorted(e["annotations"]) for s, e in universe.items()}
    rows_of = {digest(s): e["rows"] for s, e in universe.items()}

    index = CandidateIndex(sequences)
    candidates, stats = index.candidate_pairs(args.identity, args.coverage)
    print(f"candidates: {len(candidates)} "
          f"(band {stats.length_band_pairs}) at {time.time() - started:.0f}s",
          file=sys.stderr)

    admitted: Dict[str, List[str]] = collections.defaultdict(list)
    measured: Dict[Tuple[str, str], object] = {}
    refused_large = 0
    for position, (left, right) in enumerate(candidates):
        if position % 2000 == 0:
            print(f"  aligning {position}/{len(candidates)} "
                  f"({time.time() - started:.0f}s)", file=sys.stderr, flush=True)
        try:
            result = align_pair(
                sequences[left], sequences[right], max_cells=args.max_cells
            )
        except AlignmentTooLarge:
            refused_large += 1
            continue
        measured[(left, right)] = result
        if (
            result.identity >= args.identity
            and result.min_coverage >= args.coverage
            and result.overlap >= args.overlap
        ):
            admitted[left].append(right)
            admitted[right].append(left)

    print(f"admitted edges: {sum(len(v) for v in admitted.values()) // 2}; "
          f"refused-too-large: {refused_large}", file=sys.stderr)

    def relation(one: str, other: str):
        key = (one, other) if one < other else (other, one)
        return measured.get(key)

    def qualifies(one: str, other: str) -> bool:
        result = relation(one, other)
        return bool(
            result
            and result.identity >= args.identity
            and result.min_coverage >= args.coverage
            and result.overlap >= args.overlap
        )

    triples = []
    for middle in sorted(admitted):
        neighbours = sorted(admitted[middle])
        for position, left in enumerate(neighbours):
            for right in neighbours[position + 1:]:
                if qualifies(left, right):
                    continue
                closing = relation(left, right)
                edge_ab = relation(left, middle)
                edge_bc = relation(middle, right)
                closing_identity = closing.identity if closing else 0.0
                triples.append({
                    "a": left, "b": middle, "c": right,
                    "a_rows": rows_of[left], "b_rows": rows_of[middle],
                    "c_rows": rows_of[right],
                    "a_length": len(sequences[left]),
                    "b_length": len(sequences[middle]),
                    "c_length": len(sequences[right]),
                    "a_annotations": annotations[left],
                    "b_annotations": annotations[middle],
                    "c_annotations": annotations[right],
                    "ab": _edge(edge_ab),
                    "bc": _edge(edge_bc),
                    "ac": _edge(closing),
                    "closing_measured": closing is not None,
                    "window": min(edge_ab.identity, edge_bc.identity) - closing_identity,
                    "rows": rows_of[left] + rows_of[middle] + rows_of[right],
                })

    # A triple whose closing edge was actually aligned and then REFUSED is
    # stronger evidence than one whose closing edge the blocker never proposed:
    # the first exercises the cluster criterion, the second only shows that two
    # sequences are far apart. Measured refusals therefore sort first, and within
    # them the ones with the most source rows behind them, so a fixture built
    # from the top of this list is about a part of the corpus that matters.
    triples.sort(key=lambda t: (
        not t["closing_measured"], -t["rows"], -t["window"], t["a"], t["b"], t["c"]
    ))
    payload = {
        "schema": "anti-percolation-triples/1",
        "operating_point": {
            "identity": args.identity,
            "coverage": args.coverage,
            "overlap": args.overlap,
        },
        "blocking": stats.as_dict(),
        "universe": len(sequences),
        "candidates": len(candidates),
        "alignments_refused_too_large": refused_large,
        "triples_found": len(triples),
        "triples": triples[: args.limit],
        "elapsed_seconds": round(time.time() - started, 1),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"{len(triples)} open triples -> {args.output}", file=sys.stderr)


def _edge(result) -> Dict[str, object]:
    """Render one measured relation for the report."""
    if result is None:
        return {"measured": False}
    return {
        "measured": True,
        "identity": result.identity,
        "cov_left": result.cov_left,
        "cov_right": result.cov_right,
        "min_coverage": result.min_coverage,
        "overlap": result.overlap,
        "columns": result.columns,
    }


if __name__ == "__main__":
    main()
