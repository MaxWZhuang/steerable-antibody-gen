#!/usr/bin/env python
"""
Audit canonical target identity and split integrity on a PROCESSED corpus (J02).

J02 built and tested the canonicalization machinery on fixtures; what stayed
blocked was the number it exists to produce -- the real alias merge rate, and how
much of the stored train/val split those merges would move. This measures both
without regenerating anything.

Read-only on purpose. Re-running `prepare_antibody_antigen.py` to answer "how
many aliases merge" would rewrite the processed corpus and reassign splits, which
is a Class-B change costing a stage-3/stage-4 retrain. Change control's rule for
exactly this situation is to measure the straddle rate on real data BEFORE paying
that cost, so this script reads the existing corpus and reports what the change
would do.

The identity graph is built with the production functions imported from
`scripts/prepare_antibody_antigen.py` -- `extract_target_nodes` and
`TargetIdentityIndex` -- not a reimplementation. A second copy of union-find that
agreed with the first on fixtures and disagreed on real data is precisely the
failure this audit is supposed to detect.

One mapping detail: the preparer reads identity fields out of the raw parquet
row's nested `metadata`, while a processed record carries `target_name`,
`target_pdb`, and `target_uniprot` at top level under those same keys. So a
processed record is passed directly as the mapping, and `extract_target_fields`
finds the same fields by the same names.

Usage:

    python scripts/audit_target_identity.py \\
        --data-path data/processed/antibody_antigen/antibody_antigen.jsonl.gz \\
        --output-json outputs/target-identity-audit.json
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def load_preparer():
    """
    Import `prepare_antibody_antigen.py` as a module.

    It is a script, not a package module, so it is loaded by path -- the same
    mechanism the test suite uses for it.
    """
    script_path = Path(__file__).resolve().parent / "prepare_antibody_antigen.py"
    spec = importlib.util.spec_from_file_location("prepare_antibody_antigen", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    """Stream a JSONL(.gz) corpus one record at a time."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


# Each training stage trains on a DIFFERENT subset of this one file, and the
# leakage question is only meaningful about the rows a stage actually sees. The
# gating rules mirror the stage configs and CLAUDE.md's strong-binder rule: stage
# 4 gates on `is_strong_binder` (which covers boolean positives AND KD/-logKD/
# fuzzy strong binders), stage 3 on the binary `binder_label` population.
POPULATIONS: dict[str, Any] = {
    "all_rows": lambda record: True,
    "stage3_binary_labeled": lambda record: record.get("binder_label") in (0, 1),
    "stage4_strong_binders": lambda record: bool(record.get("is_strong_binder")),
}


def audit(data_path: Path, preparer) -> dict[str, Any]:
    """
    Two passes over the corpus: build the identity graph, then attribute splits.

    Two passes for the same reason the preparer needs them -- a row's component
    is only known once every row has been observed, so nothing can be attributed
    to a component during the first pass.
    """
    # Pass 1: identity only. Labels are never read, so supervision cannot
    # influence the partition.
    index = preparer.TargetIdentityIndex()
    rows = 0
    for record in iter_records(data_path):
        rows += 1
        nodes = preparer.extract_target_nodes(record, record.get("sequence_antigen") or "")
        index.observe(nodes)
    index.finalize()

    # Pass 2: attribute each row to its canonical component and legacy key, once
    # per population. The identity GRAPH is built over every row above -- identity
    # is a property of the corpus, and the preparer takes the same position
    # ("rows later dropped by keep_record still contribute their identifiers") --
    # but the split attribution is per-population, because a target that appears
    # in both splits of the full file may sit entirely in train once a stage's
    # filter is applied, or vice versa.
    per_population: dict[str, Any] = {}
    splits_by_canonical: dict[str, Counter] = defaultdict(Counter)
    splits_by_legacy: dict[str, Counter] = defaultdict(Counter)
    canonical_to_legacy: dict[str, set] = defaultdict(set)
    rows_by_split: Counter = Counter()
    rows_without_canonical = 0

    population_state = {
        name: (defaultdict(Counter), Counter())
        for name in POPULATIONS
        if name != "all_rows"
    }

    for record in iter_records(data_path):
        split = str(record.get("split") or "unknown")
        rows_by_split[split] += 1
        nodes = preparer.extract_target_nodes(record, record.get("sequence_antigen") or "")
        canonical = index.canonical_id(nodes) if nodes else ""
        legacy = str(record.get("target_key") or "")
        if not canonical:
            rows_without_canonical += 1
        else:
            splits_by_canonical[canonical][split] += 1
            canonical_to_legacy[canonical].add(legacy)
        if legacy:
            splits_by_legacy[legacy][split] += 1

        for name, predicate in POPULATIONS.items():
            if name == "all_rows":
                continue
            if predicate(record):
                groups, split_counts = population_state[name]
                split_counts[split] += 1
                if canonical:
                    groups[canonical][split] += 1

    def straddle_report(groups: dict[str, Counter]) -> dict[str, Any]:
        """A group straddles when its rows appear in more than one split."""
        straddling = {key: counts for key, counts in groups.items() if len(counts) > 1}
        straddling_rows = sum(sum(counts.values()) for counts in straddling.values())
        total_rows = sum(sum(counts.values()) for counts in groups.values())
        return {
            "groups": len(groups),
            "straddling_groups": len(straddling),
            "straddling_rows": straddling_rows,
            "total_rows": total_rows,
            "straddling_row_fraction": (
                round(straddling_rows / total_rows, 6) if total_rows else 0.0
            ),
            "largest_straddling_groups": [
                {"group": key, "splits": dict(counts)}
                for key, counts in sorted(
                    straddling.items(), key=lambda kv: -sum(kv[1].values())
                )[:10]
            ],
        }

    # How many legacy keys did canonicalization actually fuse? A component
    # covering more than one legacy key is an alias that the old split treated
    # as two independent targets.
    merged_components = {
        canonical: sorted(keys)
        for canonical, keys in canonical_to_legacy.items()
        if len({key for key in keys if key}) > 1
    }

    # Per-split leakage. The straddle fraction above is dominated by TRAIN rows
    # and understates what matters for evaluation: a val row is compromised when
    # its canonical target also carries train rows, because the model has then
    # seen that biological target under a different alias. The complement --
    # val rows on a target absent from train -- is the only part of val that
    # supports a generalization claim.
    def leakage_by_split(groups: dict[str, Counter]) -> dict[str, Any]:
        report: dict[str, Any] = {}
        splits = sorted({split for counts in groups.values() for split in counts})
        for split in splits:
            leaked = clean = 0
            clean_targets: set = set()
            leaked_targets: set = set()
            for canonical, counts in groups.items():
                here = counts.get(split, 0)
                if not here:
                    continue
                if sum(v for k, v in counts.items() if k != split):
                    leaked += here
                    leaked_targets.add(canonical)
                else:
                    clean += here
                    clean_targets.add(canonical)
            total = leaked + clean
            report[split] = {
                "rows": total,
                "rows_on_a_target_seen_in_another_split": leaked,
                "rows_on_a_target_unique_to_this_split": clean,
                "leaked_row_fraction": round(leaked / total, 6) if total else 0.0,
                "targets_shared_with_another_split": len(leaked_targets),
                "targets_unique_to_this_split": len(clean_targets),
            }
        return report

    # Antibody-side leakage: a SEPARATE axis from the one the split controls.
    # The split is keyed on the target, but stage 4 reconstructs the antibody's
    # HCDR3 -- so a val row whose heavy chain (or whose HCDR3) already appears in
    # train is memorization-exposed no matter how cleanly its antigen was held
    # out. A target-disjoint split does not make an antibody-disjoint split, and
    # the plan's Gate-3 claim depends on the second, not the first.
    antibody_side: dict[str, Any] = {}
    for name, predicate in POPULATIONS.items():
        train_heavy: set = set()
        train_hcdr3: set = set()
        train_pairs: set = set()
        val_rows = []
        for record in iter_records(data_path):
            if not predicate(record):
                continue
            heavy = (record.get("sequence_heavy") or record.get("heavy_variable_aa") or "")
            hcdr3 = record.get("cdr3_aa_heavy") or ""
            antigen = record.get("sequence_antigen") or ""
            split = str(record.get("split") or "unknown")
            if split == "train":
                if heavy:
                    train_heavy.add(heavy)
                if hcdr3:
                    train_hcdr3.add(hcdr3)
                if heavy and antigen:
                    train_pairs.add((heavy, antigen))
            elif split == "val":
                val_rows.append((heavy, hcdr3, antigen))

        total = len(val_rows)
        heavy_seen = sum(1 for heavy, _, _ in val_rows if heavy and heavy in train_heavy)
        hcdr3_seen = sum(1 for _, hcdr3, _ in val_rows if hcdr3 and hcdr3 in train_hcdr3)
        pair_seen = sum(
            1
            for heavy, _, antigen in val_rows
            if heavy and antigen and (heavy, antigen) in train_pairs
        )
        antibody_side[name] = {
            "val_rows": total,
            "val_rows_whose_heavy_chain_is_in_train": heavy_seen,
            "val_rows_whose_hcdr3_is_in_train": hcdr3_seen,
            "val_rows_whose_exact_antibody_antigen_pair_is_in_train": pair_seen,
            "heavy_chain_leak_fraction": round(heavy_seen / total, 6) if total else 0.0,
            "hcdr3_leak_fraction": round(hcdr3_seen / total, 6) if total else 0.0,
            "distinct_train_heavy_chains": len(train_heavy),
            "distinct_train_hcdr3": len(train_hcdr3),
        }

    per_split_leakage = leakage_by_split(splits_by_canonical)
    for name, (groups, split_counts) in population_state.items():
        per_population[name] = {
            "rows_by_split": dict(sorted(split_counts.items())),
            "distinct_canonical_targets": len(groups),
            "leakage": leakage_by_split(groups),
        }

    # Concentration: how dominated is the corpus by its largest target?
    rows_per_canonical = Counter(
        {key: sum(counts.values()) for key, counts in splits_by_canonical.items()}
    )
    total_attributed = sum(rows_per_canonical.values())
    top = rows_per_canonical.most_common(10)

    return {
        "schema_version": "target-identity-audit/1",
        "data_path": data_path.as_posix(),
        "rows": rows,
        "rows_by_split": dict(sorted(rows_by_split.items())),
        "rows_without_canonical_identity": rows_without_canonical,
        "index_stats": index.stats(),
        "canonical": straddle_report(splits_by_canonical),
        "per_split_leakage": per_split_leakage,
        "per_population": per_population,
        "antibody_side_leakage": antibody_side,
        "legacy_target_key": straddle_report(splits_by_legacy),
        "components_merging_multiple_legacy_keys": {
            "count": len(merged_components),
            "examples": [
                {"canonical_target_id": key, "legacy_target_keys": value}
                for key, value in sorted(merged_components.items())[:10]
            ],
        },
        "concentration": {
            "distinct_canonical_targets": len(rows_per_canonical),
            "rows_attributed": total_attributed,
            "top_targets": [
                {
                    "canonical_target_id": key,
                    "rows": count,
                    "fraction": round(count / total_attributed, 6)
                    if total_attributed
                    else 0.0,
                    "splits": dict(splits_by_canonical[key]),
                }
                for key, count in top
            ],
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    preparer = load_preparer()
    report = audit(args.data_path, preparer)

    print(f"[audit] {report['data_path']}")
    print(f"[audit] rows: {report['rows']}  splits: {report['rows_by_split']}")
    print(f"[audit] index stats: {json.dumps(report['index_stats'], sort_keys=True)}")
    for label in ("canonical", "legacy_target_key"):
        block = report[label]
        print(
            f"[{label}] groups={block['groups']} "
            f"straddling={block['straddling_groups']} "
            f"rows_in_straddling_groups={block['straddling_rows']} "
            f"({block['straddling_row_fraction']:.2%})"
        )
    for split, block in report["per_split_leakage"].items():
        print(
            f"[leakage/{split}] {block['rows_on_a_target_seen_in_another_split']}"
            f"/{block['rows']} rows ({block['leaked_row_fraction']:.2%}) sit on a target "
            f"seen in another split; {block['rows_on_a_target_unique_to_this_split']} rows "
            f"on {block['targets_unique_to_this_split']} targets are unique to it"
        )
    for name, block in report["per_population"].items():
        val = block["leakage"].get("val")
        if val is None:
            print(f"[{name}] no val rows")
            continue
        print(
            f"[{name}] val {val['rows_on_a_target_seen_in_another_split']}"
            f"/{val['rows']} rows ({val['leaked_row_fraction']:.2%}) sit on a target "
            f"also in train; {val['rows_on_a_target_unique_to_this_split']} clean val rows "
            f"on {val['targets_unique_to_this_split']} targets"
        )
    for name, block in report["antibody_side_leakage"].items():
        print(
            f"[antibody/{name}] of {block['val_rows']} val rows: "
            f"{block['val_rows_whose_heavy_chain_is_in_train']} "
            f"({block['heavy_chain_leak_fraction']:.2%}) have a heavy chain seen in train, "
            f"{block['val_rows_whose_hcdr3_is_in_train']} "
            f"({block['hcdr3_leak_fraction']:.2%}) have an HCDR3 seen in train"
        )
    merged = report["components_merging_multiple_legacy_keys"]["count"]
    print(f"[audit] components fusing >1 legacy target_key: {merged}")
    conc = report["concentration"]
    print(f"[audit] distinct canonical targets: {conc['distinct_canonical_targets']}")
    for entry in conc["top_targets"][:5]:
        print(
            f"    {entry['fraction']:>7.2%}  {entry['rows']:>8} rows  "
            f"{entry['splits']}  {entry['canonical_target_id']}"
        )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8", newline="") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"[audit] wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
