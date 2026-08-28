#!/usr/bin/env python
"""
Ruling 4: how well does the ANTIBODY ALONE predict its target?

The plan's Gate-3 question is whether the antigen changes the policy. That
question is only answerable if the antibody does not already identify the
target. If it does, a model can appear antigen-conditioned while having learned
"this framework/HCDR3 shape goes with that target", and the antigen input is
decoration. Ruling 4 puts this in Gate 0 for that reason: high predictability
does not disprove antigen conditioning, it caps how much any such claim can be
worth.

The probe is deliberately WEAK and dependency-free -- hashed k-mer multinomial
naive Bayes in numpy, no sklearn. That direction of error is the safe one: a weak
probe that still predicts the target well is strong evidence for the shortcut,
whereas a strong probe scoring well would leave "maybe the probe is just good"
open. Read a high number as a lower bound on predictability.

Three splits, because the gap between them IS the result:

- `random`      row-wise. Memorization is available; this is the ceiling.
- `clone`       grouped by exact HCDR3, so no clonal lineage spans the boundary.
- `heavy`       grouped by exact heavy chain.

Four models, because a number without its floor is not a result (change-control
Rule 4):

- `majority`      always the most frequent target. THE FLOOR. This corpus is
                  dominated by one target, so the floor is high and any model
                  scoring near it has learned nothing.
- `hcdr3_lookup`  exact HCDR3 -> most common target for it in train. Measures
                  memorization directly rather than inferring it.
- `nb_heavy`      naive Bayes over hashed 3-mers of the heavy chain.
- `nb_framework`  the same, with the HCDR3 span REMOVED from the heavy chain, so
                  a high score means the germline framework alone identifies the
                  target.

Usage:

    python scripts/probe_antibody_target_predictability.py \\
        --data-path data/processed/antibody_antigen/antibody_antigen.jsonl.gz \\
        --output-json outputs/ruling4-antibody-target-predictability.json
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

KMER_SIZE = 3
HASH_BUCKETS = 4096
SMOOTHING = 1.0


def load_preparer():
    """Import the preparer by path so identity uses the PRODUCTION functions."""
    script_path = Path(__file__).resolve().parent / "prepare_antibody_antigen.py"
    spec = importlib.util.spec_from_file_location("prepare_antibody_antigen", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


# Memoized because there are at most a few thousand distinct 3-mers over the
# residue alphabet, while the corpus contains tens of millions of k-mer
# occurrences. Bounded by the alphabet, so this cannot grow with corpus size.
_BUCKET_CACHE: dict[str, int] = {}


def stable_bucket(text: str) -> int:
    """
    Hash a k-mer to a bucket with a STABLE hash.

    Python's builtin `hash` is salted per process (PYTHONHASHSEED), so using it
    would make the probe's features -- and therefore its accuracy -- differ
    between runs of the same command. A recorded number that cannot be
    reproduced is not evidence.
    """
    cached = _BUCKET_CACHE.get(text)
    if cached is None:
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=4).digest()
        cached = int.from_bytes(digest, "big") % HASH_BUCKETS
        _BUCKET_CACHE[text] = cached
    return cached


def bucket_ids(sequence: str) -> np.ndarray:
    """
    The hashed bucket of every k-mer in `sequence`, computed once per row.

    Kept as ids rather than a dense 4096-wide vector because the dense form is
    16 KB per row and would not fit for a corpus this size; `np.bincount`
    expands it on demand at C speed.
    """
    return np.fromiter(
        (
            stable_bucket(sequence[i : i + KMER_SIZE])
            for i in range(len(sequence) - KMER_SIZE + 1)
        ),
        dtype=np.int32,
        count=max(0, len(sequence) - KMER_SIZE + 1),
    )


def featurize(buckets: np.ndarray) -> np.ndarray:
    """Dense hashed-k-mer count vector from precomputed bucket ids."""
    return np.bincount(buckets, minlength=HASH_BUCKETS).astype(np.float64)


def remove_span(sequence: str, start: Any, end: Any) -> str:
    """Drop the HCDR3 span, leaving framework + other CDRs."""
    try:
        start_i, end_i = int(start), int(end)
    except (TypeError, ValueError):
        return sequence
    if 0 <= start_i < end_i <= len(sequence):
        return sequence[:start_i] + sequence[end_i:]
    return sequence


def assign_split(group_key: str, holdout_fraction: float) -> str:
    """
    Deterministic group-wise holdout by hash of the grouping key.

    Hashing the GROUP (not the row) is what makes the split grouped: every row
    sharing a key lands on the same side, so a clonal lineage cannot span the
    boundary.
    """
    digest = hashlib.blake2b(group_key.encode("utf-8"), digest_size=8).digest()
    return "val" if (int.from_bytes(digest, "big") % 10_000) / 10_000 < holdout_fraction else "train"


def evaluate(rows: list[dict], split_name: str, holdout_fraction: float) -> dict[str, Any]:
    """Fit every model on this split's train side and score its val side."""
    if split_name == "random":
        key_of = lambda row: row["record_id"]  # noqa: E731
    elif split_name == "clone":
        key_of = lambda row: row["hcdr3"] or row["record_id"]  # noqa: E731
    elif split_name == "heavy":
        key_of = lambda row: row["heavy_sequence"] or row["record_id"]  # noqa: E731
    else:
        raise ValueError(f"unknown split {split_name!r}")

    train, val = [], []
    for row in rows:
        (val if assign_split(key_of(row), holdout_fraction) == "val" else train).append(row)

    if not train or not val:
        return {"split": split_name, "error": "empty side", "train": len(train), "val": len(val)}

    labels = sorted({row["target"] for row in rows})
    label_index = {label: i for i, label in enumerate(labels)}

    # --- majority: the floor -------------------------------------------------
    train_counts = Counter(row["target"] for row in train)
    majority_label = train_counts.most_common(1)[0][0]
    majority_correct = sum(1 for row in val if row["target"] == majority_label)

    # --- exact-HCDR3 lookup: memorization, measured --------------------------
    hcdr3_votes: dict[str, Counter] = defaultdict(Counter)
    for row in train:
        if row["hcdr3"]:
            hcdr3_votes[row["hcdr3"]][row["target"]] += 1
    lookup_table = {k: v.most_common(1)[0][0] for k, v in hcdr3_votes.items()}
    lookup_correct = 0
    lookup_hits = 0
    for row in val:
        predicted = lookup_table.get(row["hcdr3"])
        if predicted is not None:
            lookup_hits += 1
        else:
            predicted = majority_label
        lookup_correct += predicted == row["target"]

    # --- naive Bayes over hashed k-mers --------------------------------------
    def fit_predict(field: str) -> dict[str, Any]:
        counts = np.full((len(labels), HASH_BUCKETS), SMOOTHING, dtype=np.float64)
        priors = np.zeros(len(labels), dtype=np.float64)
        for row in train:
            i = label_index[row["target"]]
            counts[i] += featurize(row[field])
            priors[i] += 1.0
        log_theta = np.log(counts) - np.log(counts.sum(axis=1, keepdims=True))
        log_prior = np.log(np.maximum(priors, 1e-12)) - np.log(priors.sum())

        correct = 0
        correct_unseen = 0
        unseen_total = 0
        correct_minority = 0
        minority_total = 0
        train_hcdr3 = {row["hcdr3"] for row in train if row["hcdr3"]}
        chunk = 512
        for start in range(0, len(val), chunk):
            batch = val[start : start + chunk]
            features = np.stack([featurize(row[field]) for row in batch])
            scores = features @ log_theta.T + log_prior
            predictions = scores.argmax(axis=1)
            for row, prediction in zip(batch, predictions):
                hit = labels[prediction] == row["target"]
                correct += hit
                if row["hcdr3"] and row["hcdr3"] not in train_hcdr3:
                    unseen_total += 1
                    correct_unseen += hit
                # Plain accuracy is a weak metric on a corpus where one class is
                # ~79% of the rows: a model can look good while getting every
                # other target wrong. Restricting to rows NOT on the majority
                # target asks the question that actually matters -- can the
                # antibody identify its target when "guess the big one" is not
                # available? -- and its own floor is 0 by construction.
                if row["target"] != majority_label:
                    minority_total += 1
                    correct_minority += hit
        return {
            "accuracy": round(correct / len(val), 6),
            "accuracy_on_unseen_hcdr3": (
                round(correct_unseen / unseen_total, 6) if unseen_total else None
            ),
            "unseen_hcdr3_val_rows": unseen_total,
            "accuracy_excluding_majority_target": (
                round(correct_minority / minority_total, 6) if minority_total else None
            ),
            "non_majority_val_rows": minority_total,
        }

    return {
        "split": split_name,
        "train_rows": len(train),
        "val_rows": len(val),
        "distinct_targets": len(labels),
        "majority": {
            "label": majority_label,
            "accuracy": round(majority_correct / len(val), 6),
        },
        "hcdr3_lookup": {
            "accuracy": round(lookup_correct / len(val), 6),
            "val_rows_with_a_known_hcdr3": lookup_hits,
            "known_hcdr3_fraction": round(lookup_hits / len(val), 6),
        },
        "nb_heavy": fit_predict("heavy"),
        "nb_framework": fit_predict("framework"),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument(
        "--population",
        choices=["all_rows", "stage3_binary_labeled", "stage4_strong_binders"],
        default="stage4_strong_binders",
    )
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Cap rows for speed (0 = all). Deterministic: takes the first N.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    preparer = load_preparer()

    predicate = {
        "all_rows": lambda r: True,
        "stage3_binary_labeled": lambda r: r.get("binder_label") in (0, 1),
        "stage4_strong_binders": lambda r: bool(r.get("is_strong_binder")),
    }[args.population]

    # Pass 1: identity graph over EVERY row (identity is a property of the
    # corpus, not of the population being probed).
    index = preparer.TargetIdentityIndex()
    for record in iter_records(args.data_path):
        index.observe(
            preparer.extract_target_nodes(record, record.get("sequence_antigen") or "")
        )
    index.finalize()

    rows: list[dict] = []
    for record in iter_records(args.data_path):
        if not predicate(record):
            continue
        heavy = record.get("sequence_heavy") or record.get("heavy_variable_aa") or ""
        if not heavy:
            continue
        nodes = preparer.extract_target_nodes(record, record.get("sequence_antigen") or "")
        target = index.canonical_id(nodes) if nodes else ""
        if not target:
            continue
        rows.append(
            {
                "record_id": str(record.get("record_id") or len(rows)),
                "heavy": bucket_ids(heavy),
                "framework": bucket_ids(
                    remove_span(
                        heavy,
                        record.get("cdr3_start_aa_heavy"),
                        record.get("cdr3_end_aa_heavy"),
                    )
                ),
                "heavy_sequence": heavy,
                "hcdr3": record.get("cdr3_aa_heavy") or "",
                "target": target,
            }
        )
        if args.max_rows and len(rows) >= args.max_rows:
            break

    print(f"[ruling4] population={args.population} rows={len(rows)}")
    counts = Counter(row["target"] for row in rows)
    print(f"[ruling4] distinct canonical targets: {len(counts)}")
    top_label, top_count = counts.most_common(1)[0]
    print(
        f"[ruling4] corpus majority target {top_label} covers "
        f"{top_count}/{len(rows)} ({top_count / len(rows):.2%})"
    )

    results = [
        evaluate(rows, split, args.holdout_fraction)
        for split in ("random", "clone", "heavy")
    ]
    for block in results:
        if "error" in block:
            print(f"[{block['split']}] {block['error']}")
            continue
        print(
            f"[{block['split']}] val={block['val_rows']}  "
            f"floor(majority)={block['majority']['accuracy']:.4f}  "
            f"hcdr3_lookup={block['hcdr3_lookup']['accuracy']:.4f}  "
            f"nb_heavy={block['nb_heavy']['accuracy']:.4f}  "
            f"nb_framework={block['nb_framework']['accuracy']:.4f}"
        )
        unseen = block["nb_heavy"]["accuracy_on_unseen_hcdr3"]
        if unseen is not None:
            print(
                f"    nb_heavy on the {block['nb_heavy']['unseen_hcdr3_val_rows']} val rows "
                f"whose HCDR3 is NOT in train: {unseen:.4f}"
            )
        minority = block["nb_heavy"]["accuracy_excluding_majority_target"]
        if minority is not None:
            print(
                f"    nb_heavy on the {block['nb_heavy']['non_majority_val_rows']} val rows "
                f"NOT on the majority target (floor 0.0): {minority:.4f}"
                f"  [framework-only: "
                f"{block['nb_framework']['accuracy_excluding_majority_target']:.4f}]"
            )

    payload = {
        "schema_version": "ruling4-antibody-target-predictability/1",
        "data_path": args.data_path.as_posix(),
        "population": args.population,
        "rows": len(rows),
        "distinct_canonical_targets": len(counts),
        "corpus_majority_target": {
            "canonical_target_id": top_label,
            "rows": top_count,
            "fraction": round(top_count / len(rows), 6),
        },
        "kmer_size": KMER_SIZE,
        "hash_buckets": HASH_BUCKETS,
        "holdout_fraction": args.holdout_fraction,
        "splits": results,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8", newline="") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"[ruling4] wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
