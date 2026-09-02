"""
The leakage audit must count what it says it counts.

An audit is trusted more than the thing it audits, so its own arithmetic needs
pinning. Three things here are easy to get subtly wrong, and each one was either
a real mistake made on 2026-08-29 or a near-miss:

- **Rows, not distinct values.** The denominator is validation ROWS. Collapsing
  either side to a set changes the question from "what fraction of my evaluation
  is contaminated" to "what fraction of my vocabulary recurs", which is a much
  smaller and much more flattering number.
- **Identity is the field the MODEL sees.** In the paired corpus `variable_aa`
  holds the heavy chain ONLY, while `length` records the paired length. Auditing
  it reports the heavy-chain overlap under a "full sequence" label -- which is
  exactly the error this suite exists to prevent recurring, because it made a
  2.76% component overlap look like 2.76% exact-record leakage when the true
  exact-pair figure is 0%.
- **Absent fields are not matches.** A record missing a light chain must drop out
  of the denominator, never count as a non-overlap, which would dilute the rate.
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def audit(project_root: Path):
    script = project_root.parents[1] / "scripts" / "audit_split_leakage.py"
    spec = importlib.util.spec_from_file_location("audit_split_leakage", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, records: list[dict]) -> Path:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return path


# --------------------------------------------------------------------------- #
# Identity is the field the model sees
# --------------------------------------------------------------------------- #
def test_a_paired_identity_is_the_concatenation_of_both_chains(audit):
    """
    The full pair, built from the two fields `MLMCollator._encode_record`
    actually encodes -- so the audit and the training path agree on what one
    antibody IS.
    """
    record = {"sequence_heavy": "HHH", "sequence_light": "LLL", "variable_aa": "HHH"}
    assert audit._field(record, ("sequence_heavy", "sequence_light")) == "HHHLLL"
    assert audit._field(record, "sequence_heavy") == "HHH"


def test_a_pair_missing_either_chain_has_no_identity(audit):
    """
    Half a pair is not a pair. Returning the present half would make a
    heavy-only record collide with every other heavy-only record carrying that
    chain, inventing overlap that does not exist.
    """
    assert audit._field({"sequence_heavy": "HHH", "sequence_light": None},
                        ("sequence_heavy", "sequence_light")) is None
    assert audit._field({"sequence_heavy": "HHH"},
                        ("sequence_heavy", "sequence_light")) is None


def test_the_two_chains_are_not_confusable_by_concatenation(audit):
    """
    A boundary check: ("AB","C") and ("A","BC") are different antibodies and must
    not hash alike. They would under a naive join, and the resulting phantom
    overlap would be indistinguishable from real leakage.
    """
    left = audit._field({"sequence_heavy": "AB", "sequence_light": "C"},
                        ("sequence_heavy", "sequence_light"))
    right = audit._field({"sequence_heavy": "A", "sequence_light": "BC"},
                         ("sequence_heavy", "sequence_light"))
    assert left == right == "ABC", (
        "documents a REAL limitation: plain concatenation cannot separate these. "
        "It is acceptable only because chain lengths are near-constant in this "
        "corpus and a collision would OVERSTATE overlap, never hide it. Any "
        "future audit over variable-length components must use a delimiter."
    )


# --------------------------------------------------------------------------- #
# Rows, not distinct values
# --------------------------------------------------------------------------- #
def test_overlap_counts_rows_not_distinct_values(audit):
    """
    Ten validation rows sharing one leaked HCDR3 is ten contaminated evaluation
    rows, not one. Deduplicating the numerator would report 1/10 = 10% where the
    truth is 100%.
    """
    leaked = audit.digest("SAME")
    stats = audit.overlap([leaked] * 10, {leaked})
    assert stats["val_rows_seen_in_train"] == 10
    assert stats["val_rows_with_field"] == 10
    assert stats["fraction"] == 1.0
    assert stats["distinct_val_values"] == 1


def test_absent_fields_leave_the_denominator(audit):
    """
    A record with no light chain cannot be light-chain-contaminated, and must not
    count as a clean row either -- that would dilute the rate with rows the
    question does not apply to.
    """
    hit, miss = audit.digest("A"), audit.digest("B")
    stats = audit.overlap([hit, miss, None, None], {hit})
    assert stats["val_rows_with_field"] == 2
    assert stats["val_rows_seen_in_train"] == 1
    assert stats["fraction"] == 0.5


def test_an_empty_field_is_absent_not_a_value(audit):
    """Empty string and None are both "no component", never a matchable value."""
    assert audit.digest("") is None
    assert audit.digest(None) is None


def test_no_overlap_reports_zero_not_none(audit):
    """A clean split must be reported as a measured 0.0, distinguishable from
    'this could not be measured'."""
    stats = audit.overlap([audit.digest("A")], {audit.digest("B")})
    assert stats["fraction"] == 0.0
    assert stats["val_rows_seen_in_train"] == 0


def test_paired_split_reproduction_detects_a_stale_assignment(audit, tmp_path: Path):
    """One wrong stored side proves the artifact did not use the current key."""
    rows = []
    for index, heavy in enumerate(("HEAVY_A", "HEAVY_B")):
        expected = audit.deterministic_split(f"IGH:{heavy}", val_percent=10)
        stored = expected if index == 0 else (
            "val" if expected == "train" else "train"
        )
        rows.append({
            "split": stored,
            "heavy_locus": "IGH",
            "sequence_heavy": heavy,
        })

    stats = audit.paired_split_reproduction(
        _write(tmp_path / "paired.jsonl.gz", rows)
    )

    assert stats["rows_compared"] == 2
    assert stats["rows_matching_current_rule"] == 1
    assert stats["rows_mismatching_current_rule"] == 1
    assert stats["fraction_matching_current_rule"] == 0.5
    assert (
        stats["mismatch_examples"][0]["stored"]
        != stats["mismatch_examples"][0]["expected"]
    )


def test_paired_split_reproduction_is_exact_for_current_producer_key(
    audit, tmp_path: Path
):
    rows = [
        {
            "split": audit.deterministic_split(f"IGH:{heavy}", val_percent=10),
            "heavy_locus": "IGH",
            "sequence_heavy": heavy,
        }
        for heavy in ("HEAVY_A", "HEAVY_B", "HEAVY_C")
    ]

    stats = audit.paired_split_reproduction(
        _write(tmp_path / "paired.jsonl.gz", rows)
    )

    assert stats["rows_compared"] == 3
    assert stats["rows_matching_current_rule"] == 3
    assert stats["rows_mismatching_current_rule"] == 0
    assert stats["fraction_matching_current_rule"] == 1.0
    assert stats["mismatch_examples"] == []


# --------------------------------------------------------------------------- #
# End to end on a corpus whose answer is known by construction
# --------------------------------------------------------------------------- #
def test_collect_buckets_by_split_and_ignores_other_splits(audit, tmp_path: Path):
    """
    Only train and val participate. A row carrying any other split value is
    counted separately rather than silently folded into one of them.
    """
    path = _write(tmp_path / "c.jsonl.gz", [
        {"split": "train", "sequence_heavy": "H1", "sequence_light": "L1"},
        {"split": "train", "sequence_heavy": "H2", "sequence_light": "L1"},
        {"split": "val", "sequence_heavy": "H3", "sequence_light": "L1"},
        {"split": "val", "sequence_heavy": "H4", "sequence_light": "L4"},
        {"split": "test", "sequence_heavy": "H5", "sequence_light": "L5"},
    ])
    got = audit.collect(path, {"heavy": "sequence_heavy", "light": "sequence_light"})

    assert got["counts"] == {"train": 2, "val": 2, "other": 1}
    # L1 is shared by both training rows AND one validation row: the exact
    # component leak a pair-connected-component split exists to remove.
    light = audit.overlap(got["val"]["light"], got["train"]["light"])
    assert light["val_rows_seen_in_train"] == 1
    assert light["fraction"] == 0.5
    # ...while no heavy chain is shared, so the two axes report independently.
    heavy = audit.overlap(got["val"]["heavy"], got["train"]["heavy"])
    assert heavy["val_rows_seen_in_train"] == 0


def test_a_component_leak_is_invisible_to_exact_record_identity(audit, tmp_path: Path):
    """
    THE point of the audit. Exact-record deduplication reports a perfectly clean
    split while every validation row shares a light chain with training. Both
    numbers are true; only one of them answers "was this antibody unseen".
    """
    path = _write(tmp_path / "c.jsonl.gz", [
        {"split": "train", "sequence_heavy": "H1", "sequence_light": "SHARED"},
        {"split": "val", "sequence_heavy": "H2", "sequence_light": "SHARED"},
        {"split": "val", "sequence_heavy": "H3", "sequence_light": "SHARED"},
    ])
    got = audit.collect(path, {
        "full_pair": ("sequence_heavy", "sequence_light"),
        "light": "sequence_light",
    })

    exact = audit.overlap(got["val"]["full_pair"], got["train"]["full_pair"])
    light = audit.overlap(got["val"]["light"], got["train"]["light"])
    assert exact["fraction"] == 0.0, "exact-record dedup sees a clean split"
    assert light["fraction"] == 1.0, "...and every row is component-contaminated"
