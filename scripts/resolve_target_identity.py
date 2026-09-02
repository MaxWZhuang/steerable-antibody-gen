#!/usr/bin/env python
"""
Run the typed-identity engine over a corpus and emit the Level-1 evidence.

WHAT THIS PRODUCES
------------------
The four artifacts a Level-1 conformance claim is checkable from, plus the
validator's attestation. All of them land in ``outputs/``, which is git-ignored,
because producing evidence and publishing it are different acts:

    outputs/claim-manifest-<name>.json     the sealed claim
    outputs/split-manifest-<name>.json     the exact assignment
    outputs/guard-report-<name>.json       component health and error rates
    outputs/leakage-line-<name>.json       the standard line
    outputs/conformance-<name>.json        the validator's verdict

It also prints the standard leakage line, which is the thing that belongs beside
every number quoted from this split.

Pass ``--evidence-dir specs/evidence`` to seal the claim manifest into the
tracked tree, which is what sealing a publishable benchmark version means.

WHAT IT IS FOR
--------------
Two jobs. The first is to produce the evidence. The second, and the reason it
exists as a script rather than a library call, is the percolation measurement: a
previous redesign of target identity looked correct on sixteen fixture records
and percolated to a 77% component on the relation the split actually keys on.
Sixteen records cannot show that. This runs the engine over all 9,574 distinct
antigen sequences and publishes the largest-group share, the component-size
distribution, the high-degree bridges that were refused, and the test-ineligible
count -- so the failure mode is measured rather than hoped past.

It does NOT write a corpus. Adopting the engine as the producer's default is a
separate decision with a data generation and a retraining behind it; this script
lets that decision be made against numbers.

COST, MEASURED
--------------
This is expensive and the number should be in front of you before you start it.
At the shipped operating point the similarity pass proposes **450,381** candidate
pairs over the 9,574 distinct antigens, and one alignment of a typical pair
(~420 x 430 residues) costs about **25 ms**, so a full run is roughly **3 to 4
hours** of CPU. Reading the shards costs another ~2 minutes.

The cost is call-overhead in the per-row numpy loop, not cells: 7.4 million
DP cells per second at 420x430. Two ways to bring it down, neither taken here:

- Align LAZILY inside the agglomeration rather than up front. Edges are processed
  strongest-first and most within-cluster edges turn out redundant, so a large
  fraction of those 450,381 alignments are never needed. This changes the
  clustering loop, and changing the clustering loop under time pressure is
  exactly how the two previous attempts at this failed.
- Move the inner loop to a compiled extension. That is a new dependency.

``--max-antigens N`` runs a deterministic prefix of the antigen universe, which
is what to use for a same-day answer. A subsample UNDER-estimates percolation, so
a clean subsample result is suggestive and a dirty one is conclusive.

USAGE
-----
    python scripts/resolve_target_identity.py --name asd-typed-v1

    # a same-day answer over the first 3,000 distinct antigens
    python scripts/resolve_target_identity.py --name asd-typed-3k --max-antigens 3000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from smallAntibodyGen import target_identity as ti  # noqa: E402
from smallAntibodyGen.entity_resolution import blocking, conformance  # noqa: E402


def deterministic_split(key: str, val_percent: int = 10) -> str:
    """Assign a group to train or val, by hash of the group id.

    Byte-identical to the producer's `deterministic_split`, deliberately: the
    split RULE is not what this change is about, and reimplementing it slightly
    differently would make the two engines' outputs incomparable for a reason
    that has nothing to do with identity.

    Args:
        key: The group id.
        val_percent: Fraction routed to validation.

    Returns:
        ``"train"`` or ``"val"``.
    """
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return "val" if int(digest[:8], 16) % 100 < val_percent else "train"


def iter_shard_rows(shard_dir: Path, limit: int | None) -> Iterator[Mapping[str, object]]:
    """Yield identity views of every shard row.

    Every row is observed, including rows `keep_record` will later drop: a
    curator writing an accession is evidence about the target whether or not
    that antibody survives filtering, and dropping it would make identity depend
    on filter thresholds.
    """
    import pandas as pd

    scripts_dir = REPO_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import prepare_antibody_antigen as paa

    seen_antigens: set = set()
    emitted = 0
    for shard in sorted(shard_dir.glob("*.parquet")):
        frame = pd.read_parquet(shard, columns=["antigen_sequence", "metadata"])
        for raw_sequence, raw_metadata in zip(
            frame["antigen_sequence"], frame["metadata"]
        ):
            sequence = paa.clean_aa_sequence(raw_sequence)
            if not sequence:
                continue
            if limit is not None:
                if sequence not in seen_antigens:
                    if len(seen_antigens) >= limit:
                        continue
                    seen_antigens.add(sequence)
            fields = paa.extract_target_fields(
                paa.normalize_metadata_dict(raw_metadata)
            )
            emitted += 1
            yield {
                "antigen_sequence": sequence,
                "metadata": {
                    "target_name": fields["target_name"],
                    "target_pdb": fields["target_pdb"],
                    "target_uniprot": fields["target_uniprot"],
                },
            }
        print(f"  {shard.name}: {emitted} rows", file=sys.stderr, flush=True)


def _git_commit() -> str:
    """The commit this ran at, or a marker saying it could not be determined."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def build_claim(point: ti.OperatingPoint) -> conformance.Claim:
    """The claim this split actually supports, stated before the numbers.

    Written as a claim and not as a threshold, and in particular written so that
    the ANTIBODY axis appears explicitly in ``permitted_exposure_relationships``.
    That is not a loophole, it is the honest statement: the split groups on the
    antigen, so the same antibody may appear on both sides, and measured on the
    shipped corpus it usually does. A claim that named only the antigen axis
    would let a 78.6% antibody-side overlap read as a passing leakage audit.
    """
    return conformance.Claim(
        claim_class="closed-book unseen-broader-group",
        target_population=(
            "distinct antigen sequences in data/raw/asd-antibody-antigen, all 20 "
            "shards, observed before keep_record filtering"
        ),
        unit_of_generalisation="biological target, quarantine-closed",
        allowed_evaluation_context=(
            "the antibody sequence under evaluation",
            "the antigen sequence it is conditioned on",
        ),
        prohibited_exposure_relationships=(
            "the same biological target on both sides of the split",
            "a construct and a construct that contains it on opposite sides",
            "two constructs sharing a structural container on opposite sides",
            "two constructs sharing an unapproved curator name on opposite sides",
        ),
        permitted_exposure_relationships=(
            "the same antibody against two different targets, one per side -- "
            "this split does NOT hold out antibodies, and any claim about unseen "
            "antibodies needs a different unit of generalisation",
            "structural or functional similarity between targets whose sequences "
            "do not align above the family threshold",
        ),
        exclusions=(
            "rows carrying no antigen sequence",
            "pairs too large to align, which are counted rather than dropped",
        ),
        estimand=(
            "performance on antibody-antigen pairs whose biological target was "
            "not present in training, under the declared sequence relations"
        ),
    )


def build_operationalisation(point: ti.OperatingPoint) -> conformance.Operationalisation:
    """The declared approximation of the claim, with its blind spots named."""
    return conformance.Operationalisation(
        relation_classes=(
            "exact", "lexical-near-duplicate", "metadata/container",
            "pairwise-containment",
        ),
        unassessed_relation_classes=("semantic", "structural", "aggregate-mosaic"),
        detectors={
            "exact": "SHA-256 of the cleaned antigen sequence",
            "lexical-near-duplicate": (
                "local affine Smith-Waterman, BLOSUM62, gap open 11 extend 1, "
                "with a k-mer and length-band blocker whose recall is derived and "
                "measured"
            ),
            "metadata/container": "shared normalised PDB accession",
            "pairwise-containment": (
                "one-sided coverage above threshold with the other side below it"
            ),
        },
        thresholds=point.as_dict(),
        masks=(),
        promotion_rules=(
            {"evidence": "E1 byte-identical antigen", "action": "must-link"},
            {"evidence": "E2 near-identical construct", "action": "must-link"},
            {"evidence": "E3 family-level similarity", "action": "must-link"},
            {"evidence": "E4 local containment", "action": "must-link"},
            {"evidence": "E5 shared UniProt accession", "action": "must-link"},
            {"evidence": "E6 shared PDB entry", "action": "must-link"},
            {"evidence": "E7 approved name", "action": "must-link"},
            {"evidence": "E8 unapproved name", "action": "must-link"},
            {"evidence": "concatenated construct (poly-G linker)",
             "action": "test-ineligible"},
            {"evidence": "container spanning more families than max_container_span",
             "action": "test-ineligible"},
        ),
        known_blind_spots=(
            "structural or epitope-level similarity between sequences that do not "
            "align: two targets can share a binding surface and no residues",
            "the antigen encoder's upstream pretraining corpus is opaque, so no "
            "data-side claim here can assert non-exposure through it",
            "the ANTIBODY axis is not held out at all under this claim",
            "aggregate mosaic exposure -- a construct reconstructable from the "
            "union of several others -- is not computed",
        ),
        calibration_population=(
            "entity_resolution.synthetic, seed 20260901, planted ground truth, "
            "sharing no code with the resolver; disjoint from the five curated "
            "audit families by construction"
        ),
        model_capability_envelope={
            "note": (
                "these thresholds are properties of the sequence relations and "
                "not of a model, so the envelope is unconstrained -- but that is "
                "only true because no learned component participates in "
                "grouping. A future embedding-based relation would need a real "
                "envelope here."
            ),
            "learned_components_in_grouping": [],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Benchmark version name.")
    parser.add_argument("--shards", type=Path,
                        default=REPO_ROOT / "data" / "raw" / "asd-antibody-antigen")
    parser.add_argument("--max-antigens", type=int, default=None,
                        help="Cap distinct antigens, for a quick check.")
    parser.add_argument("--val-percent", type=int, default=10)
    parser.add_argument("--max-cells", type=int, default=50_000_000)
    parser.add_argument("--max-container-span", type=int, default=8)
    parser.add_argument(
        "--max-split-group-row-share", type=float, default=0.70,
        help=(
            "Predeclared bound on the largest split group's share of SOURCE "
            "ROWS. Sealed into the claim manifest before the numbers exist, and "
            "gated by the validator: a run above it has percolated and its "
            "held-out claim is not supported. The default is chosen knowing that "
            "one target already carries 63.4%% of this corpus's rows on its own, "
            "so the bound is about what the QUARANTINE relation adds to that."
        ),
    )
    parser.add_argument("--blocking-audit-size", type=int, default=120,
                        help="Population size for the exhaustive blocking-recall audit.")
    # Everything defaults into `outputs/`, which is git-ignored. Sealing a claim
    # manifest into tracked `specs/evidence/` publishes it, and that is a
    # decision for whoever is sealing a benchmark version rather than a side
    # effect of running the shadow build. Pass
    # `--evidence-dir specs/evidence` deliberately when that is the intent.
    parser.add_argument("--evidence-dir", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs")
    args = parser.parse_args()

    started = time.time()
    point = ti.DEFAULT_OPERATING_POINT

    print("reading shards...", file=sys.stderr)
    rows = list(iter_shard_rows(args.shards, args.max_antigens))
    print(f"{len(rows)} rows in {time.time() - started:.0f}s", file=sys.stderr)

    def progress(done: int, total: int, label: str) -> None:
        print(f"  aligning {label} {done}/{total} "
              f"({time.time() - started:.0f}s)", file=sys.stderr, flush=True)

    resolution = ti.resolve_target_identity(
        rows,
        operating_point=point,
        max_cells=args.max_cells,
        max_container_span=args.max_container_span,
        progress=progress,
    )
    stats = resolution.stats()
    print(f"resolved in {time.time() - started:.0f}s", file=sys.stderr)

    # --- the split ------------------------------------------------------- #
    group_of: Dict[str, str] = {}
    rows_of: Dict[str, int] = {}
    for digest in resolution.antigen_digests():
        group_of[digest] = resolution.split_group_id(digest)
        rows_of[digest] = resolution.rows_for(digest)
    assignments = {
        group: deterministic_split(group, args.val_percent)
        for group in sorted(set(group_of.values()))
    }
    group_evidence = {
        group: {
            "constructs": sum(1 for g in group_of.values() if g == group),
            "test_ineligible": sum(
                1 for d, g in group_of.items()
                if g == group and resolution.test_ineligible(d)
            ),
        }
        for group in sorted(set(group_of.values()))
    }

    # --- the measured blocking-recall audit ------------------------------- #
    digests = resolution.antigen_digests()
    sample = {
        digest: resolution.antigen_sequence(digest)
        for digest in digests[: args.blocking_audit_size]
    }
    # Audited at the configuration the similarity pass actually runs, band on
    # and overlap floor applied, because a recall figure for a blocker nobody
    # runs is evidence about nothing.
    recall = blocking.blocking_recall_report(
        sample, point.family_identity, point.family_coverage,
        min_overlap=point.family_overlap,
    )
    print(f"blocking recall on {recall.population} sequences: {recall.recall}",
          file=sys.stderr)

    # --- the artifacts ---------------------------------------------------- #
    claim = build_claim(point)
    operationalisation = build_operationalisation(point)
    claim_manifest = conformance.build_claim_manifest(
        name=args.name,
        claim=claim,
        operationalisation=operationalisation,
        population={
            "source": str(args.shards.relative_to(REPO_ROOT)),
            "rows_observed": stats["target_rows_seen"],
            "rows_without_antigen": stats["target_rows_without_antigen"],
            "distinct_antigens": stats["target_distinct_antigens"],
            "constructs": stats["target_constructs"],
            "biological_targets": stats["target_families"],
            "split_groups": stats["target_split_groups"],
            "test_ineligible_constructs": stats["target_test_ineligible_constructs"],
            "max_antigens_cap": args.max_antigens,
            "max_split_group_row_share": args.max_split_group_row_share,
        },
        provenance={
            "git_commit": _git_commit(),
            "engine": "smallAntibodyGen.target_identity",
            "operating_point": point.as_dict(),
            "max_container_span": args.max_container_span,
            "max_cells": args.max_cells,
            "val_percent": args.val_percent,
        },
    )
    split_manifest = conformance.build_split_manifest(
        claim_manifest_sha256=claim_manifest["manifest_sha256"],
        claim_name=args.name,
        assignments=assignments,
        group_of=group_of,
        group_evidence=group_evidence,
        target_val_fraction=args.val_percent / 100.0,
        row_weights=rows_of,
    )
    guard_report = conformance.build_guard_report(
        claim_manifest_sha256=claim_manifest["manifest_sha256"],
        resolution_stats=stats,
        calibration=resolution.calibration_report().as_dict(),
        audit=resolution.audit_report().as_dict(),
        blocking_recall=recall.as_dict(),
    )

    # The channels this run cannot support, named before the reducer runs.
    unsupported = ["opaque_upstream_pretraining", "unassessed_structural_equivalence"]
    supported, unmapped = conformance.reduce_claim(claim.claim_class, unsupported)
    line = conformance.LeakageLine(
        claim=supported,
        relations=operationalisation.relation_classes,
        unassessed=(
            operationalisation.unassessed_relation_classes
            + ("opaque upstream pretraining of the ESM antigen encoder",
               "the antibody axis, which this claim does not hold out")
        ),
        conditioning=(
            "declared training corpus only; no retrieval index, no live tools, "
            "no few-shot context"
        ),
        population=(
            f"{stats['target_distinct_antigens']} distinct antigens over "
            f"{stats['target_rows_seen']} rows; "
            f"{stats['target_split_groups']} split groups; largest holds "
            f"{split_manifest['largest_group_share']:.2%} of antigens; "
            f"{stats['target_test_ineligible_constructs']} constructs "
            f"test-ineligible"
        ),
        conformance="Level 1",
        result_identity=(
            f"claim {claim_manifest['manifest_sha256'][:16]}, "
            f"split {split_manifest['manifest_sha256'][:16]}, "
            f"commit {_git_commit()[:12]}"
        ),
        status=(
            "valid for the declared static channels; upstream pretraining "
            "exposure unknown"
        ),
    )
    attestation = conformance.validate_level_1(
        claim_manifest, split_manifest, guard_report, line.as_dict(),
        unsupported_channels=unsupported,
    )

    conformance.write_artifact(
        args.evidence_dir / f"claim-manifest-{args.name}.json", claim_manifest)
    conformance.write_artifact(
        args.output_dir / f"split-manifest-{args.name}.json", split_manifest)
    conformance.write_artifact(
        args.output_dir / f"guard-report-{args.name}.json", guard_report)
    conformance.write_artifact(
        args.output_dir / f"leakage-line-{args.name}.json", line.as_dict())
    conformance.write_artifact(
        args.output_dir / f"conformance-{args.name}.json", attestation.as_dict())

    print()
    print(line.render())
    print()
    print(f"conformance: level {attestation.level} "
          f"(claimed {attestation.claimed_level})")
    for failure in attestation.failures:
        print(f"  FAILURE: {failure}")
    for note in attestation.notes:
        print(f"  note: {note}")
    print()
    for key in ("target_distinct_antigens", "target_constructs", "target_families",
                "target_split_groups", "target_test_ineligible_constructs",
                "target_accession_conflicts", "target_names_approved",
                "target_names_quarantined", "target_quarantine_edges",
                "target_construct_merges_refused", "target_family_merges_refused",
                "component_min_pairwise_identity", "component_min_pairwise_coverage",
                "largest_family_row_share", "largest_split_group_row_share",
                "largest_split_group_constructs",
                "target_alignments_measured", "target_alignments_refused_too_large"):
        print(f"  {key}: {stats[key]}")
    print(f"  achieved_val_fraction: {split_manifest['achieved_val_fraction']:.4f}")
    print(f"  largest_group_share:   {split_manifest['largest_group_share']:.4f}")
    print(f"  group_size_distribution: {split_manifest['group_size_distribution']}")
    print(f"  high_degree_bridges refused: {len(stats['high_degree_bridges'])}")
    print(f"\ntotal {time.time() - started:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
