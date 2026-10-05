#!/usr/bin/env python
"""Verify and bundle the provenance a publicly quotable result needs.

The flight's 24/24 verify checks assert **completeness** -- that every declared
stage ran and every declared artifact exists. They do not re-derive a single
digest. This does, for the five things a reader of a quoted number would have to
take on trust:

1. **The run manifest.** All 718 manifested files are re-hashed against their
   recorded sha256. A mismatch means the artifact a number came from has changed
   since the manifest was written.
2. **The frozen source.** The archived bytes under ``source_snapshot/`` are
   re-hashed against the per-file digests in ``source_snapshot.json``, and the
   aggregate is re-derived by the same rule the freeze used --
   ``sha256("\\n".join(f"{logical} {digest}"))`` over sorted logical paths -- and
   compared with the recorded ``snapshot_sha256`` that every trajectory's
   ``identity.source_snapshot_sha256`` points at.
3. **Bank identities.** Every bank this flight drew, with its role, draw seed,
   index digest and sampler-versus-scorer parity, collected from the places they
   are scattered across, plus a check that no two banks declared independent
   share a draw seed.
4. **Per-seed score files.** The row-score and preservation arrays every table is
   computed from, with digests and row counts, including the follow-up
   experiments' files, which carry no manifest of their own.
5. **The interval convention**, stated per reported statistic rather than in
   prose, because three different intervals appear in these tables and they
   answer three different questions.

Read-only. Nothing here writes into a landed run root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smallAntibodyGen.experiments import her2_support_paths as paths  # noqa: E402

#: Which interval belongs to which reported number. Prose in a methods section
#: is not enough: the same table carries all three.
INTERVAL_CONVENTION = {
    "across trained seeds": {
        "statistic": ("any arm-versus-arm or arm-minus-parent contrast: macro-AP, audit mean "
                      "drop, one-step pair ordering, affinity Spearman"),
        "interval": "paired t over the three parent seeds, df = 2",
        "function": "her2_nf_metrics.paired_t",
        "reading": ("the only interval here that speaks about training-seed variability. "
                    "df = 2 is weak: failure to exclude zero is NOT equivalence."),
    },
    "within one bank, across rows": {
        "statistic": "a difference computed on one fixed set of rows or clones",
        "interval": "paired row/clone bootstrap, 2,000 draws, shared resample across vectors",
        "function": "her2_nf_metrics.paired_bootstrap, scripts/her2_affinity_scoring.bootstrap_rows",
        "reading": ("Monte Carlo sensitivity to which rows the bank happens to contain. "
                    "Rows are not extra models and this is never pooled with the seed interval."),
    },
    "a rate out of a fixed denominator": {
        "statistic": "tenfold and hundredfold tail rates, off-library fraction, hit rates",
        "interval": "two-sided 95% Wilson score interval, no continuity correction",
        "function": "her2_nf_metrics.wilson_interval",
        "reading": "binomial uncertainty at one checkpoint on one bank; not a seed statement.",
    },
    "tail event convention": {
        "statistic": "tenfold and hundredfold counts",
        "interval": "not an interval: the counting rule",
        "function": "drop >= ln(10), drop >= ln(100)",
        "reading": ("INCLUSIVE. The historical post-hoc audit counted drop > threshold "
                    "(strict). The two are reported side by side and never averaged. The "
                    "likelihood gate's own stop rule is the inherited STRICT D > 1 "
                    "nat/sequence on a different population and is a different statistic."),
    },
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(run_root):
    """Re-hash every manifested file. Completeness is not integrity."""
    manifest = json.loads((run_root / "artifact_manifest.json").read_text(encoding="utf-8"))
    checked, mismatched, missing, size_only = 0, [], [], []
    for entry in manifest["files"]:
        target = run_root / entry["file"]
        if not target.is_file():
            missing.append(entry["file"])
            continue
        checked += 1
        if target.stat().st_size != entry["bytes"]:
            size_only.append(entry["file"])
        observed = sha256_file(target)
        if observed != entry["sha256"]:
            mismatched.append({"file": entry["file"], "recorded": entry["sha256"],
                               "observed": observed})
    return {"declared": manifest["count"], "checked": checked,
            "mismatched": mismatched, "missing": missing, "size_mismatch": size_only,
            "excluded_count": manifest["excluded_count"],
            "exclusion_rule": manifest["exclusion_rule"],
            "diagnosis": diagnose_mismatches(run_root, mismatched),
            "passed": not mismatched and not missing}


def diagnose_mismatches(run_root, mismatched):
    """Say WHAT changed, not just that something did.

    A stale digest with purely appended content is a different problem from a
    rewritten artifact, and the remedy differs: the first is a disclosure, the
    second invalidates whatever was computed from the file.
    """
    notes = []
    for entry in mismatched:
        if entry["file"] != "stream_registry.json":
            notes.append({"file": entry["file"], "classification": "unclassified",
                          "note": "not diagnosed; treat as a content change until shown otherwise"})
            continue
        document = json.loads((run_root / "stream_registry.json").read_text(encoding="utf-8"))
        streams = document["streams"]
        foreign = {name: record for name, record in streams.items()
                   if "fixed_vs_fresh" in name or record["descriptor"].get("role") == "audit"}
        seeds = [record["seed"] for record in streams.values()]
        notes.append({
            "file": entry["file"],
            "classification": "appended_by_a_follow_up_experiment",
            "total_streams": len(streams),
            "appended": sorted(foreign),
            "appended_count": len(foreign),
            "duplicate_seeds_across_the_whole_registry": len(seeds) != len(set(seeds)),
            "note": ("scripts/her2_fixed_vs_fresh_bank.py resolved its context with the LANDED "
                     "flight run root, so services.stream_seed appended its audit-bank "
                     "descriptors to the flight's registry and rewrote the file. The addition is "
                     "purely additive -- no pre-existing descriptor or seed changed, and the "
                     "registry's own collision check still passes -- but the landed run root is "
                     "no longer byte-identical to its manifest."),
            "remedy": ("disclose in a separate supersession record. Do NOT rewrite "
                       "artifact_manifest.json: a manifest edited to match a later mutation "
                       "certifies nothing. Follow-up experiments must resolve their own run "
                       "root."),
        })
    return notes


def verify_source_snapshot(run_root):
    """Re-hash the archived bytes and re-derive the aggregate by the freeze's own rule."""
    document = json.loads((run_root / "source_snapshot.json").read_text(encoding="utf-8"))
    snapshot_root = run_root / document["snapshot_directory"]
    files = document["files"]
    mismatched, missing = [], []
    for logical, record in sorted(files.items()):
        target = snapshot_root / logical
        if not target.is_file():
            missing.append(logical)
            continue
        observed = sha256_file(target)
        if observed != record["sha256"]:
            mismatched.append({"file": logical, "recorded": record["sha256"],
                               "observed": observed})
    aggregate = paths.sha256_text("\n".join(f"{name} {files[name]['sha256']}"
                                            for name in sorted(files)))
    classification = {}
    for record in files.values():
        classification[record["classification"]] = classification.get(
            record["classification"], 0) + 1
    return {"file_count": document["file_count"], "archived_checked": len(files) - len(missing),
            "archived_mismatched": mismatched, "archived_missing": missing,
            "recorded_aggregate": document["snapshot_sha256"],
            "rederived_aggregate": aggregate,
            "aggregate_matches": aggregate == document["snapshot_sha256"],
            "git_state": document["git_state"],
            "classification": classification,
            "identity_is": document["identity_is"],
            "passed": (aggregate == document["snapshot_sha256"]
                       and not mismatched and not missing)}


def collect_banks(run_root, extra_roots):
    """Every bank identity, and a collision check over draw seeds."""
    banks = []

    def absorb(document, source):
        for record in document:
            banks.append({
                "source": source, "role": record.get("role"),
                "parent_id": record.get("parent_id"),
                "rows": record.get("rows"), "draw_seed": record.get("draw_seed"),
                "index_sha256": record.get("index_sha256"),
                "sum_log_probability_sha256": record.get("sum_log_probability_sha256"),
                "sampled_by_state_sha256": record.get("sampled_by_state_sha256"),
                "temperature": record.get("temperature"),
                "parity": record.get("parity"),
                "duplicates_retained": record.get("duplicates_retained")})

    audit = run_root / "preservation_audit.json"
    if audit.is_file():
        absorb(json.loads(audit.read_text(encoding="utf-8")).get("banks", []),
               "preservation_audit.json")
    coupling = run_root / "coupling.json"
    if coupling.is_file():
        document = json.loads(coupling.read_text(encoding="utf-8"))
        for entry in document.get("finalists", []) + document.get("entries", []):
            absorb(entry.get("banks", []), "coupling.json")
    for root in extra_roots:
        fill = root / "fill.json"
        if fill.is_file():
            for record in json.loads(fill.read_text(encoding="utf-8")).get("banks", []):
                banks.append({"source": f"{root.name}/fill.json", "role": "q_generation_finalist",
                              "parent_id": f"parent::seed{record['seed']}",
                              "rows": record["rows"], "draw_seed": record["draw_seed"],
                              "index_sha256": record["index_sha256"],
                              "parity": {"rows_above_atol": record["parity_rows_above_atol"]},
                              "redrawn_from_flight": record.get("redrawn_from_flight")})

    by_seed = {}
    collisions = []
    for bank in banks:
        seed = bank.get("draw_seed")
        if seed is None:
            continue
        previous = by_seed.get(seed)
        # A redraw that deliberately reuses a descriptor is expected to match.
        if previous is not None and previous["index_sha256"] != bank["index_sha256"]:
            collisions.append({"draw_seed": seed, "a": previous["index_sha256"],
                               "b": bank["index_sha256"]})
        by_seed[seed] = bank
    # The pass rule is |difference| <= atol + rtol*|expected|, enforced by a raise
    # inside compare_sum_log_probabilities, so a bank that exists already passed.
    # rows_above_atol counts rows over the BARE atol and is a diagnostic, not the
    # criterion -- at |ell| ~ 27 the rtol term roughly doubles the allowance.
    failed_parity, above_atol = [], []
    for bank in banks:
        parity = bank.get("parity") or {}
        if not parity:
            continue
        worst, allowance = parity.get("max_abs_error"), parity.get("worst_allowance")
        if worst is not None and allowance is not None and worst > allowance:
            failed_parity.append(bank)
        elif (parity.get("rows_above_atol") or 0) > 0:
            above_atol.append({"parent_id": bank["parent_id"], "source": bank["source"],
                               "index_sha256": bank["index_sha256"],
                               "rows_above_atol": parity["rows_above_atol"],
                               "rows": parity.get("rows"),
                               "max_abs_error": worst, "worst_allowance": allowance})
    return {"count": len(banks), "distinct_draw_seeds": len(by_seed),
            "seed_collisions_with_different_content": collisions,
            "banks_failing_sampler_scorer_parity": failed_parity,
            "banks_with_rows_over_bare_atol": above_atol,
            "parity_rule": ("|sampled - rescored| <= atol + rtol*|rescored|, atol 5e-5, "
                            "rtol 2e-6, enforced by a raise at draw time"),
            "parity_note": ("rows_above_atol counts rows over the BARE atol and is a "
                            "diagnostic, not the pass criterion. At |sum log p| ~ 27 the rtol "
                            "term roughly doubles the allowance, so a row can exceed atol and "
                            "still be well inside tolerance."),
            "banks": banks,
            "passed": not collisions and not failed_parity}


def collect_score_files(directories):
    """The arrays every reported table is computed from, with digests."""
    out = []
    for label, directory in directories:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.npz")):
            out.append({"group": label, "file": path.name,
                        "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="outputs/her2_next_flight_20260922/production_03")
    parser.add_argument("--experiment", default="outputs/her2_fixed_vs_fresh_20260923")
    parser.add_argument("--matched", default="outputs/her2_matched_table_20260923")
    parser.add_argument("--output", default="outputs/her2_provenance_20260923")
    args = parser.parse_args(argv)

    run_root = ROOT / args.run_root
    experiment, matched = ROOT / args.experiment, ROOT / args.matched
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)

    print("re-hashing the run manifest ...", flush=True)
    manifest = verify_manifest(run_root)
    print("  declared %d, checked %d, mismatched %d, missing %d -> %s" % (
        manifest["declared"], manifest["checked"], len(manifest["mismatched"]),
        len(manifest["missing"]), "PASS" if manifest["passed"] else "FAIL"), flush=True)

    print("re-hashing the frozen source snapshot ...", flush=True)
    snapshot = verify_source_snapshot(run_root)
    print("  %d files, aggregate %s -> %s" % (
        snapshot["file_count"],
        "matches" if snapshot["aggregate_matches"] else "DIFFERS",
        "PASS" if snapshot["passed"] else "FAIL"), flush=True)
    print("  recorded  %s" % snapshot["recorded_aggregate"], flush=True)
    print("  rederived %s" % snapshot["rederived_aggregate"], flush=True)

    print("collecting bank identities ...", flush=True)
    banks = collect_banks(run_root, [experiment, matched])
    print("  %d banks, %d distinct draw seeds, %d parity failures, %d with a row over the "
          "bare atol (diagnostic) -> %s" % (
              banks["count"], banks["distinct_draw_seeds"],
              len(banks["banks_failing_sampler_scorer_parity"]),
              len(banks["banks_with_rows_over_bare_atol"]),
              "PASS" if banks["passed"] else "FAIL"), flush=True)

    print("digesting per-seed score files ...", flush=True)
    scores = collect_score_files([
        ("flight row_scores", run_root / "row_scores"),
        ("flight preservation", run_root / "preservation"),
        ("fixed_vs_fresh banks", experiment / "banks"),
        ("matched row_scores", matched / "row_scores"),
        ("matched preservation", matched / "preservation"),
        ("matched banks", matched / "banks"),
    ])
    groups = {}
    for record in scores:
        groups[record["group"]] = groups.get(record["group"], 0) + 1
    for label, count in sorted(groups.items()):
        print("  %-24s %4d files" % (label, count), flush=True)

    document = {
        "record_kind": "provenance_bundle",
        "run_root": args.run_root,
        "purpose": ("what a reader of a quoted number would otherwise take on trust. The "
                    "flight's verify stage asserts completeness; this re-derives digests."),
        "manifest": manifest,
        "source_snapshot": snapshot,
        "banks": banks,
        "score_files": scores,
        "score_file_counts": groups,
        "interval_convention": INTERVAL_CONVENTION,
        "passed": manifest["passed"] and snapshot["passed"] and banks["passed"],
        "recorded_at": paths.utc_now(),
    }
    paths.write_json(out / "provenance.json", document)
    print("\noverall: %s" % ("PASS" if document["passed"] else "FAIL"))
    print(f"wrote {out / 'provenance.json'}")
    return 0 if document["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
