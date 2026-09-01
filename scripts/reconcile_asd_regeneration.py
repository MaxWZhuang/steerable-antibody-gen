#!/usr/bin/env python
"""
Read-only reconciliation of the OLD vs REGENERATED antibody-antigen (ASD) corpus.

WHY THIS EXISTS
---------------
`data/processed/antibody_antigen/` was written on 2026-06-11, before the producer
grew production canonical target identity. `data/processed/antibody_antigen_v2/`
was written on 2026-08-31 from byte-identical parquet inputs (hashes pinned in
`outputs/asd-regeneration-pin.json`) using ALL DEFAULT parameters. The question
this script answers is not "are they different" -- they obviously are -- but
"is EVERY difference accounted for by a change we intended".

The hard constraint recorded in the pin is that the OLD run wrote **no stats file
and no parameter record**, so its settings cannot be *proven* from an artifact
the producer emitted for that purpose. This script therefore refuses to assume
the two runs shared parameters. It instead does three things:

1. **Recovers the OLD run's own summary from its console log** and *validates*
   that log against the corpus it claims to describe (row count and per-split
   counts must match the file on disk). A log is weaker evidence than a stats
   file, so it is only used once it has been checked against the bytes.
2. **Re-derives the split from the corpus's own fields** using the producer's
   own `deterministic_split`, so the claim "OLD split followed `target_key`,
   NEW split follows `canonical_target_id`" is measured, not assumed. If a row's
   split does not follow the key we think it followed, that is reported as a
   violation rather than smoothed over.
3. **Attributes every split change to a cause**, and reports the residual -- the
   rows whose split moved for a reason canonicalization does not explain -- as
   its own headline number. A non-zero residual is a blocker, not a footnote.

WHAT EACH COMPARISON IS FOR
---------------------------
- **Row and split counts** (§1). The coarsest signal; a corpus that gained rows
  changed its filtering, and a corpus that moved its val fraction changed what
  every downstream number means.
- **Record-id sets** (§2). `record_id` is `"{shard_file}:{row_index}"`, derived
  from the input layout rather than from output order, so it *should* be stable
  across runs over identical inputs. "Should" is not "is": the script VERIFIES
  stability by checking that record ids present in both corpora carry identical
  biology, and reports the natural-key (heavy, light, antigen) comparison
  alongside it either way. If the verification fails, the natural key is the one
  to believe, because the producer itself deduplicates on exactly that triple.
- **Biological multisets** (§3). An identifier can churn without the biology
  moving, and biology can move while identifiers hold still. Comparing the
  multisets of heavy / light / antigen sequences and of label tuples separates
  the two. Multisets, not sets: a corpus that duplicated a sequence and a corpus
  that did not are different corpora, and set comparison hides that.
- **Canonical target identity** (§4). The whole point of the regeneration. The
  legacy `target_key` picks the FIRST available identifier out of four mutually
  exclusive branches, so one biological target annotated once by UniProt and
  once by PDB got two keys and two independent split draws. Canonicalization
  fuses those aliases into one component. We report how many legacy keys were
  absorbed, the largest fusions, and -- importantly -- any legacy key that
  SPLIT across several canonical components, which would be the opposite of the
  intended effect and is not something the design predicts.
- **Split and target churn** (§5). Cross-tabulated train<->val movement, and
  the attribution described above.
- **Producer drop accounting** (§6). Both runs printed their own drop reasons.
  Row-count arithmetic that closes exactly (`kept + duplicates + drops ==
  rows_seen`, on both sides) is what turns "the corpora differ by N rows" into
  "the producer says it kept N more rows, for these reasons".
- **Drop-reason replay** (§7, opt-in `--replay-drop-reasons`). The two drop
  tables are almost disjoint, which shows THAT filtering changed but not WHY.
  The replay re-derives both tables from the raw shards changing exactly one
  thing -- the `clean_aa_sequence` implementation -- and checks each against its
  own run's log. Reproducing the 2026-06-11 table with the pre-fix cleaner
  settles two questions at once: that one function accounts for the entire
  shift, and that the OLD run used the default filter parameters, which is
  exactly what the pin says cannot be proven from the artifacts it left behind.

HONEST HANDLING OF AMBIGUITY
----------------------------
Nothing here is a pass/fail gate on its own. The artifact records, for each
difference, whether the script could tie it to a cause it can name. Anything it
cannot tie to a cause is emitted under `unexplained` and echoed to stdout with a
loud banner. The reader -- not this script -- decides whether the corpus is fit
to be designated the Stage 3 source; the script's job is to make sure that
decision is made against complete arithmetic rather than a spot check.

READ-ONLY
---------
Both corpora are opened `"rt"` through gzip and never written. The only file
this script creates is its JSON artifact.

Usage:
    python scripts/reconcile_asd_regeneration.py
    python scripts/reconcile_asd_regeneration.py --old <path> --new <path> \
        --output-json outputs/asd-regeneration-reconciliation.json
"""
from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "asd-regeneration-reconciliation/1"

OLD_CORPUS = PROJECT_ROOT / "data/processed/antibody_antigen/antibody_antigen.jsonl.gz"
NEW_CORPUS = PROJECT_ROOT / "data/processed/antibody_antigen_v2/antibody_antigen.jsonl.gz"

# The OLD run left no stats file. This console log is the only surviving record
# of what it did. It is matched to the corpus by `validate_summary_against_corpus`
# (row count and per-split counts must agree) before any of its numbers are used.
OLD_LOG = PROJECT_ROOT / "logs/20260611_010330_preprocess_asd.log"
NEW_LOG = PROJECT_ROOT / "logs/asd_regen_20260831_132554.log"

PIN_JSON = PROJECT_ROOT / "outputs/asd-regeneration-pin.json"
PRODUCER = PROJECT_ROOT / "scripts/prepare_antibody_antigen.py"

# 8 bytes of BLAKE2b per sequence. At ~8.3e5 distinct values per field the
# birthday probability is ~1.9e-8, and a collision could only ever make two
# different sequences look EQUAL -- i.e. it would understate the difference
# between the corpora, never invent one. Any difference this script reports is
# therefore real; only its silence carries that (negligible) risk.
DIGEST_BYTES = 8
ABSENT = b"\x00" * DIGEST_BYTES

# Fields the producer itself deduplicates on (`write_record.dedupe_key`). The
# natural key MUST be this triple and not a prettier choice, or the "one row per
# biological triple" invariant we lean on below would not actually hold.
BIOLOGY_FIELDS = ("sequence_heavy", "sequence_light", "sequence_antigen")


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #
def file_sha256(path: Path) -> str:
    """Bind the report to the exact bytes it read; a corpus is regenerable."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rel(path: Path) -> str:
    """Repo-relative path when possible; absolute otherwise (tests use tmp dirs)."""
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def sequence_digest(value: object) -> bytes:
    """
    8-byte digest of one sequence field, mirroring the producer's coercion.

    ``str(value or "")`` is exactly what `write_record` does when it builds its
    dedupe key, so ``None`` and ``""`` collapse to the same thing HERE for the
    same reason they collapse THERE. A missing field maps to the ABSENT
    sentinel rather than to the digest of the empty string, so "no light chain"
    and "a light chain that happens to hash to zero" stay distinguishable.
    """
    text = str(value or "")
    if not text:
        return ABSENT
    return hashlib.blake2b(text.encode("utf-8"), digest_size=DIGEST_BYTES).digest()


def biology_key(record: Dict[str, Any]) -> bytes:
    """
    The natural key of one row: (heavy, light, antigen) as 24 digest bytes.

    This is the producer's own dedupe key, so within one corpus it is unique by
    construction -- an invariant `index_corpus` checks rather than trusts.
    """
    return b"".join(sequence_digest(record.get(field)) for field in BIOLOGY_FIELDS)


def label_signature(record: Dict[str, Any]) -> Tuple[Any, bool, str]:
    """
    The supervision a row carries, as a comparable tuple.

    `binder_label` is kept as-is (None / 0 / 1 are three different states, and
    coercing None to 0 would silently convert "unlabelled" into "negative").
    `is_strong_binder` is normalized to bool because that is how every consumer
    gates on it.
    """
    return (
        record.get("binder_label"),
        bool(record.get("is_strong_binder")),
        str(record.get("affinity_type") or ""),
    )


def read_records(path: Path) -> Iterator[Dict[str, Any]]:
    """Stream a processed JSONL(.gz). Read-only, one row in memory at a time."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def read_log_text(path: Path) -> str:
    """
    Decode a producer console log.

    The two logs were captured by different harnesses: the 2026-06-11 one is a
    PowerShell redirect (UTF-16 LE with a BOM), the 2026-08-31 one is raw bytes
    containing tqdm block-drawing characters that are not valid UTF-8. Both are
    evidence; neither is going to be re-captured. Decoding is therefore lenient
    on purpose -- the fields we parse are pure ASCII, so replacement characters
    elsewhere cannot corrupt them.
    """
    raw = path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    return raw.decode("utf-8", errors="replace")


def load_producer_module():
    """
    Import the producer so the split is re-derived with ITS function, not a copy.

    A local reimplementation of `deterministic_split` would be a second source
    of truth, and the whole point of §5 is to test the corpus against what the
    producer actually computes. Importing costs a pandas import; that is cheaper
    than an attribution built on a drifted copy.
    """
    spec = importlib.util.spec_from_file_location("_asd_producer", PRODUCER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# §1-3  Corpus indexing
# --------------------------------------------------------------------------- #
class CorpusIndex:
    """
    One pass over a corpus, retaining only what the comparisons need.

    Sequences are kept as digests and every repeated string (`target_key`,
    `canonical_target_id`, the label tuple) is interned to a small integer, so a
    ~832k-row corpus costs a few hundred MB instead of several GB. Nothing
    stored here can reconstruct a sequence, which is the point: this is a
    comparison index, not a second copy of the corpus.

    Attributes:
        rows: total rows read.
        by_split: Counter of the `split` field (including any unexpected value).
        by_rid: record_id -> (split, target_key, canonical_target_id, bio, label)
            with the three string-valued members interned as ints.
        bio_to_rid: biology triple -> record_id. One entry per triple; a repeat
            is counted in `duplicate_biology` instead of overwriting, because a
            corpus that violates the producer's dedupe invariant must be
            reported, not silently deduplicated a second time here.
        heavy / light / antigen / labels: multisets (Counter) for §3.
        rows_with_canonical: rows carrying a non-empty `canonical_target_id`.
        field_present: which optional fields appeared at least once.
    """

    def __init__(self) -> None:
        self.rows = 0
        self.by_split: Counter = Counter()
        self.by_rid: Dict[str, Tuple[int, int, int, bytes, int]] = {}
        self.bio_to_rid: Dict[bytes, str] = {}
        self.duplicate_biology = 0
        self.duplicate_record_id = 0
        self.heavy: Counter = Counter()
        self.light: Counter = Counter()
        self.antigen: Counter = Counter()
        self.labels: Counter = Counter()
        self.target_key_rows: Counter = Counter()
        self.canonical_rows: Counter = Counter()
        self.legacy_to_canonical: Counter = Counter()
        self.rows_with_canonical = 0
        self.field_present: Dict[str, bool] = {}
        # Interning tables. Index 0 is reserved for "" so an absent field and an
        # empty one share a slot deliberately.
        self._strings: List[str] = [""]
        self._string_ids: Dict[str, int] = {"": 0}
        self._labels: List[Tuple[Any, bool, str]] = []
        self._label_ids: Dict[Tuple[Any, bool, str], int] = {}

    def _intern(self, text: str) -> int:
        found = self._string_ids.get(text)
        if found is None:
            found = len(self._strings)
            self._strings.append(text)
            self._string_ids[text] = found
        return found

    def _intern_label(self, label: Tuple[Any, bool, str]) -> int:
        found = self._label_ids.get(label)
        if found is None:
            found = len(self._labels)
            self._labels.append(label)
            self._label_ids[label] = found
        return found

    def string(self, ident: int) -> str:
        return self._strings[ident]

    def label(self, ident: int) -> Tuple[Any, bool, str]:
        return self._labels[ident]

    def add(self, record: Dict[str, Any]) -> None:
        self.rows += 1
        rid = str(record.get("record_id") or "")
        split = str(record.get("split") or "")
        self.by_split[split] += 1

        target_key = str(record.get("target_key") or "")
        canonical = str(record.get("canonical_target_id") or "")
        if "canonical_target_id" in record:
            self.field_present["canonical_target_id"] = True
        if "target_identity_nodes" in record:
            self.field_present["target_identity_nodes"] = True
        if canonical:
            self.rows_with_canonical += 1

        bio = biology_key(record)
        label_id = self._intern_label(label_signature(record))

        if rid in self.by_rid:
            self.duplicate_record_id += 1
        else:
            self.by_rid[rid] = (
                self._intern(split),
                self._intern(target_key),
                self._intern(canonical),
                bio,
                label_id,
            )
        if bio in self.bio_to_rid:
            self.duplicate_biology += 1
        else:
            self.bio_to_rid[bio] = rid

        self.heavy[bio[:DIGEST_BYTES]] += 1
        self.light[bio[DIGEST_BYTES : 2 * DIGEST_BYTES]] += 1
        self.antigen[bio[2 * DIGEST_BYTES :]] += 1
        self.labels[label_id] += 1
        self.target_key_rows[self._intern(target_key)] += 1
        if canonical:
            self.canonical_rows[self._intern(canonical)] += 1
            self.legacy_to_canonical[(self._intern(target_key), self._intern(canonical))] += 1

    def split_of(self, rid: str) -> str:
        return self.string(self.by_rid[rid][0])

    def target_key_of(self, rid: str) -> str:
        return self.string(self.by_rid[rid][1])

    def canonical_of(self, rid: str) -> str:
        return self.string(self.by_rid[rid][2])

    def bio_of(self, rid: str) -> bytes:
        return self.by_rid[rid][3]

    def label_of(self, rid: str) -> Tuple[Any, bool, str]:
        return self.label(self.by_rid[rid][4])


def index_corpus(records: Iterable[Dict[str, Any]]) -> CorpusIndex:
    """Build a `CorpusIndex` from any iterable of processed records."""
    index = CorpusIndex()
    for record in records:
        index.add(record)
    return index


# --------------------------------------------------------------------------- #
# §2  Identifier sets and identifier stability
# --------------------------------------------------------------------------- #
def compare_key_sets(old_keys: Iterable, new_keys: Iterable) -> Dict[str, int]:
    """Three-way split of two key sets: only-old, only-new, in-both."""
    old_set = set(old_keys)
    new_set = set(new_keys)
    return {
        "only_old": len(old_set - new_set),
        "only_new": len(new_set - old_set),
        "in_both": len(old_set & new_set),
        "old_total": len(old_set),
        "new_total": len(new_set),
    }


def verify_record_id_stability(
    old: CorpusIndex, new: CorpusIndex, sample_limit: int = 20
) -> Dict[str, Any]:
    """
    Decide whether `record_id` is a trustworthy join key across the two runs.

    `record_id` is `"{shard_file}:{row_index}"`, so over byte-identical inputs it
    is a property of the SOURCE and should be reproducible. That is a claim about
    the producer, and this function turns it into a measurement: for every id in
    both corpora, does the biology agree? Any disagreement means the id is a
    run-local label and the natural key must be used instead.

    Returns:
        A verdict dict. `stable` is True only when EVERY shared id carries
        identical biology -- one mismatch is enough to disqualify a join key.
    """
    shared = old.by_rid.keys() & new.by_rid.keys()
    mismatches: List[str] = []
    total_mismatch = 0
    for rid in shared:
        if old.bio_of(rid) != new.bio_of(rid):
            total_mismatch += 1
            if len(mismatches) < sample_limit:
                mismatches.append(rid)
    return {
        "shared_record_ids": len(shared),
        "shared_ids_with_identical_biology": len(shared) - total_mismatch,
        "shared_ids_with_changed_biology": total_mismatch,
        "stable": total_mismatch == 0,
        "sample_mismatched_ids": mismatches,
        "key_used_for_row_join": "record_id" if total_mismatch == 0 else "biology_triple",
    }


# --------------------------------------------------------------------------- #
# §3  Multiset comparison
# --------------------------------------------------------------------------- #
def multiset_delta(old: Counter, new: Counter) -> Dict[str, int]:
    """
    Compare two multisets, reporting BOTH the set and the multiplicity delta.

    `rows_only_in_old` / `rows_only_in_new` are multiset differences (sums of
    positive parts), so a value present in both but 5x in one and 2x in the
    other contributes 3 to one side and 0 to the other. `distinct_*` are plain
    set sizes. Reporting only one of the two would let a pure duplication change
    (or a pure vocabulary change) pass unseen.
    """
    old_keys = set(old)
    new_keys = set(new)
    only_old_rows = sum(count - new.get(key, 0) for key, count in old.items() if count > new.get(key, 0))
    only_new_rows = sum(count - old.get(key, 0) for key, count in new.items() if count > old.get(key, 0))
    return {
        "distinct_old": len(old_keys),
        "distinct_new": len(new_keys),
        "distinct_shared": len(old_keys & new_keys),
        "distinct_only_old": len(old_keys - new_keys),
        "distinct_only_new": len(new_keys - old_keys),
        "rows_old": sum(old.values()),
        "rows_new": sum(new.values()),
        "rows_only_in_old": only_old_rows,
        "rows_only_in_new": only_new_rows,
        "multisets_equal": old == new,
    }


def compare_biology_on_shared_rows(
    old: CorpusIndex, new: CorpusIndex, join_key: str, sample_limit: int = 20
) -> Dict[str, Any]:
    """
    Per-row biology and label comparison over the rows present in both corpora.

    `join_key` selects what "the same row" means:
      - ``"record_id"``: same source shard and row index. Answers "did this
        source row's content change".
      - ``"biology_triple"``: same (heavy, light, antigen). Answers "did the
        LABELS attached to this biological pairing change", which is the only
        question left once biology is the join key.

    Both are computed by the caller when useful; the distinction matters because
    a row whose identifier churned is not a row whose biology changed, and the
    task is to report the second, not the first.
    """
    if join_key == "record_id":
        shared = old.by_rid.keys() & new.by_rid.keys()
        biology_changed: List[str] = []
        labels_changed: List[str] = []
        for rid in shared:
            if old.bio_of(rid) != new.bio_of(rid):
                biology_changed.append(rid)
            if old.label_of(rid) != new.label_of(rid):
                labels_changed.append(rid)
        return {
            "join_key": join_key,
            "shared_rows": len(shared),
            "rows_with_changed_biology": len(biology_changed),
            "rows_with_changed_labels": len(labels_changed),
            "sample_changed_biology": sorted(biology_changed)[:sample_limit],
            "sample_changed_labels": sorted(labels_changed)[:sample_limit],
        }

    shared = old.bio_to_rid.keys() & new.bio_to_rid.keys()
    labels_changed_bio: List[str] = []
    for bio in shared:
        old_rid = old.bio_to_rid[bio]
        new_rid = new.bio_to_rid[bio]
        if old.label_of(old_rid) != new.label_of(new_rid):
            labels_changed_bio.append(new_rid)
    return {
        "join_key": join_key,
        "shared_rows": len(shared),
        "rows_with_changed_biology": 0,  # equal by construction under this join
        "rows_with_changed_labels": len(labels_changed_bio),
        "sample_changed_labels": sorted(labels_changed_bio)[:sample_limit],
    }


def classify_membership_changes(
    old: CorpusIndex, new: CorpusIndex, sample_limit: int = 20
) -> Dict[str, Any]:
    """
    Split the appearing/disappearing rows into "biology moved" and "row moved".

    The producer keeps the FIRST row of each (heavy, light, antigen) triple and
    drops the rest as duplicates. Loosening an upstream filter can therefore
    change WHICH source row represents a triple without changing the triple, and
    that shows up as a record_id vanishing from one corpus and a different one
    appearing in the other while the biology is untouched. Conflating that with
    a genuine gain or loss of biology would be a serious misread, so the two are
    counted separately:

    - ``only_old_biology_still_present``: the row's triple is still in NEW under
      a different record_id. Representation churn, not data loss.
    - ``only_old_biology_gone``: the triple left the corpus entirely. This is
      real data loss and must be explained by the producer's drop reasons.
    - the mirrored pair for rows only in NEW.

    Returns:
        Counts plus small samples of each category.
    """
    only_old = old.by_rid.keys() - new.by_rid.keys()
    only_new = new.by_rid.keys() - old.by_rid.keys()

    old_gone = 0
    old_churn = 0
    old_gone_sample: List[str] = []
    for rid in only_old:
        if old.bio_of(rid) in new.bio_to_rid:
            old_churn += 1
        else:
            old_gone += 1
            if len(old_gone_sample) < sample_limit:
                old_gone_sample.append(rid)

    new_added = 0
    new_churn = 0
    new_added_sample: List[str] = []
    for rid in only_new:
        if new.bio_of(rid) in old.bio_to_rid:
            new_churn += 1
        else:
            new_added += 1
            if len(new_added_sample) < sample_limit:
                new_added_sample.append(rid)

    return {
        "only_old_rows": len(only_old),
        "only_old_biology_still_present": old_churn,
        "only_old_biology_gone": old_gone,
        "sample_only_old_biology_gone": sorted(old_gone_sample),
        "only_new_rows": len(only_new),
        "only_new_biology_already_present": new_churn,
        "only_new_biology_added": new_added,
        "sample_only_new_biology_added": sorted(new_added_sample),
    }


# --------------------------------------------------------------------------- #
# §4  Canonical target identity
# --------------------------------------------------------------------------- #
def fusion_report(
    pairs: Dict[Tuple[str, str], int], top_n: int = 15
) -> Dict[str, Any]:
    """
    Measure how canonicalization regrouped the legacy `target_key` space.

    Args:
        pairs: ``{(legacy_target_key, canonical_target_id): row_count}``.
        top_n: how many largest fusions to list.

    Returns:
        Counts of fused components and of the legacy keys they absorbed, the
        largest fusions, and -- separately -- any legacy key that landed in more
        than one canonical component. A "split" legacy key is NOT what the
        design predicts (canonicalization is supposed to be a coarsening of the
        legacy partition), so it is surfaced as its own field rather than
        buried, and a non-empty list belongs in the unexplained bucket.
    """
    canonical_to_legacy: Dict[str, set] = defaultdict(set)
    canonical_rows: Counter = Counter()
    legacy_to_canonical: Dict[str, set] = defaultdict(set)
    for (legacy, canonical), count in pairs.items():
        canonical_to_legacy[canonical].add(legacy)
        canonical_rows[canonical] += count
        legacy_to_canonical[legacy].add(canonical)

    fused = {c: keys for c, keys in canonical_to_legacy.items() if len(keys) > 1}
    split_legacy = {legacy: sorted(cs) for legacy, cs in legacy_to_canonical.items() if len(cs) > 1}

    largest = sorted(
        (
            {
                "canonical_target_id": canonical,
                "legacy_key_count": len(keys),
                "rows": canonical_rows[canonical],
                "legacy_keys": sorted(keys)[:12],
                "legacy_keys_truncated": len(keys) > 12,
            }
            for canonical, keys in fused.items()
        ),
        key=lambda item: (-item["legacy_key_count"], -item["rows"], item["canonical_target_id"]),
    )[:top_n]

    return {
        "distinct_legacy_keys": len(legacy_to_canonical),
        "distinct_canonical_ids": len(canonical_to_legacy),
        "fused_components": len(fused),
        "legacy_keys_absorbed_by_fusions": sum(len(keys) for keys in fused.values()),
        "rows_in_fused_components": sum(canonical_rows[c] for c in fused),
        "largest_fusions": largest,
        "legacy_keys_spanning_multiple_components": len(split_legacy),
        "sample_split_legacy_keys": dict(list(split_legacy.items())[:10]),
    }


# --------------------------------------------------------------------------- #
# §5  Split churn and its attribution
# --------------------------------------------------------------------------- #
def split_change_attribution(
    old: CorpusIndex,
    new: CorpusIndex,
    split_fn,
    val_percent: int,
    sample_limit: int = 20,
) -> Dict[str, Any]:
    """
    Cross-tabulate train<->val movement and tie each move to a cause.

    The causal claim under test is: the OLD split hashed the legacy
    `target_key`; the NEW split hashes `canonical_target_id`; therefore a row's
    split changes exactly when those two keys hash to different buckets. Each
    clause is measured:

    - `old_rows_following_legacy_key`: rows where `old.split ==
      split_fn(old.target_key)`. If this is not 100%, the OLD run did NOT use
      the parameters we assumed (a different `--val-percent`, or a different key)
      and every attribution below is void -- which is why it is reported first.
    - `new_rows_following_canonical_id`: the same test on the NEW side.
    - `attributable_to_canonicalization`: changed rows where
      `split_fn(canonical) != split_fn(legacy)`.
    - `unattributed_changes`: the residual. This is the number that blocks
      designation if it is not zero.

    `split_fn` is the producer's own `deterministic_split`, passed in rather
    than reimplemented, so a change to the producer's hashing shows up here as a
    failed check instead of as agreement between two stale copies.
    """
    shared = old.by_rid.keys() & new.by_rid.keys()
    transitions: Counter = Counter()
    old_follows = 0
    new_follows = 0
    attributable = 0
    unattributed: List[str] = []
    target_key_changed = 0
    target_key_changed_samples: List[str] = []
    canonical_differs_from_legacy = 0

    for rid in shared:
        old_split = old.split_of(rid)
        new_split = new.split_of(rid)
        old_key = old.target_key_of(rid)
        new_key = new.target_key_of(rid)
        canonical = new.canonical_of(rid)

        if old_split == split_fn(old_key, val_percent=val_percent):
            old_follows += 1
        if canonical and new_split == split_fn(canonical, val_percent=val_percent):
            new_follows += 1
        if old_key != new_key:
            target_key_changed += 1
            if len(target_key_changed_samples) < sample_limit:
                target_key_changed_samples.append(rid)
        if canonical and canonical != new_key:
            canonical_differs_from_legacy += 1

        transitions[f"{old_split}->{new_split}"] += 1
        if old_split != new_split:
            legacy_bucket = split_fn(new_key, val_percent=val_percent)
            canonical_bucket = split_fn(canonical, val_percent=val_percent) if canonical else legacy_bucket
            if legacy_bucket != canonical_bucket:
                attributable += 1
            elif len(unattributed) < sample_limit:
                unattributed.append(rid)

    changed = sum(count for key, count in transitions.items() if key.split("->")[0] != key.split("->")[1])
    return {
        "shared_rows": len(shared),
        "val_percent_assumed": val_percent,
        "old_rows_following_legacy_key": old_follows,
        "old_key_hypothesis_holds": old_follows == len(shared),
        "new_rows_following_canonical_id": new_follows,
        "new_key_hypothesis_holds": new_follows == len(shared),
        "transitions": dict(sorted(transitions.items())),
        "rows_with_changed_split": changed,
        "attributable_to_canonicalization": attributable,
        "unattributed_changes": changed - attributable,
        "sample_unattributed": unattributed,
        "rows_with_changed_target_key": target_key_changed,
        "sample_changed_target_key": target_key_changed_samples,
        "rows_where_canonical_differs_from_legacy_key": canonical_differs_from_legacy,
    }


# --------------------------------------------------------------------------- #
# §6  Producer self-reporting
# --------------------------------------------------------------------------- #
SUMMARY_HEADER = "=== PARQUET PREPROCESS SUMMARY ==="
_SCALAR_KEYS = (
    "files_seen",
    "rows_seen",
    "records_kept",
    "duplicates_dropped",
    "paired_records",
    "heavy_only_records",
    "nanobody_records",
    "binder_labelable",
    "binder_positive",
    "binder_negative",
    "numeric_measurement_rows",
    "cdr3_span_unresolved",
    "label_conflicts",
    "file_errors",
    "row_errors",
    "identity_row_errors",
    "target_identity_node_count",
    "target_components",
    "target_alias_merges",
    "target_sequence_merges",
    "target_rows_without_identifier",
    "target_rows_without_identity",
)


def parse_producer_summary(text: str) -> Dict[str, Any]:
    """
    Recover a run's own accounting from its console log.

    Deliberately narrow: it reads the integer scalars, the `kept_by_split` /
    `kept_by_confidence` / `kept_by_affinity_type` dicts, and the indented
    `drop_reasons:` block. Everything else in the log (tqdm frames, KD unit
    diagnostics, dataset tables) is ignored. Parsing more would mean guessing at
    formats that have already changed once between the two runs.

    A key the log does not contain is simply absent from the result -- the OLD
    log predates several counters, and inventing a 0 for them would fabricate a
    measurement the run never made.
    """
    if SUMMARY_HEADER not in text:
        return {}
    body = text.split(SUMMARY_HEADER, 1)[1]
    summary: Dict[str, Any] = {}

    for key in _SCALAR_KEYS:
        match = re.search(rf"^{re.escape(key)}:\s+(-?\d+)\s*$", body, re.M)
        if match:
            summary[key] = int(match.group(1))

    for key in ("kept_by_split", "kept_by_confidence", "kept_by_affinity_type"):
        match = re.search(rf"^{re.escape(key)}:\s+(\{{.*\}})\s*$", body, re.M)
        if match:
            try:
                summary[key] = ast.literal_eval(match.group(1))
            except (ValueError, SyntaxError):  # pragma: no cover - malformed log
                summary[key] = None

    drop_match = re.search(r"^drop_reasons:\s*$", body, re.M)
    if drop_match:
        reasons: Dict[str, int] = {}
        for line in body[drop_match.end():].splitlines()[1:]:
            item = re.match(r"^\s{2,}([a-z0-9_]+):\s+(\d+)\s*$", line)
            if not item:
                break
            reasons[item.group(1)] = int(item.group(2))
        summary["drop_reasons"] = reasons

    return summary


def check_row_accounting(summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    `rows_seen == records_kept + duplicates_dropped + sum(drop_reasons)`.

    This is what upgrades the drop-reason table from a printout to an audit: if
    the identity closes, every input row is accounted for by exactly one
    outcome, and the difference between two runs can be attributed reason by
    reason. If it does not close, the log is incomplete and its drop reasons
    cannot carry the explanation.
    """
    required = ("rows_seen", "records_kept", "duplicates_dropped", "drop_reasons")
    if not all(key in summary for key in required):
        return {"checkable": False}
    dropped = sum(summary["drop_reasons"].values())
    total = summary["records_kept"] + summary["duplicates_dropped"] + dropped
    return {
        "checkable": True,
        "rows_seen": summary["rows_seen"],
        "records_kept": summary["records_kept"],
        "duplicates_dropped": summary["duplicates_dropped"],
        "rows_dropped_by_filters": dropped,
        "accounted_total": total,
        "balances": total == summary["rows_seen"],
    }


def validate_summary_against_corpus(summary: Dict[str, Any], index: CorpusIndex) -> Dict[str, Any]:
    """
    Prove a log describes the corpus we actually read, before believing it.

    The OLD run left no stats file, so its log is the only surviving parameter
    record -- and a log lying around in `logs/` is not self-evidently the log of
    the file on disk. Two independent quantities must agree: the total row count
    and the per-split counts. Both matching is strong evidence of provenance;
    either failing means the log must not be used to explain this corpus.
    """
    if "records_kept" not in summary:
        return {"checkable": False}
    logged_split = summary.get("kept_by_split") or {}
    observed_split = {key: value for key, value in index.by_split.items()}
    return {
        "checkable": True,
        "logged_records_kept": summary["records_kept"],
        "observed_rows": index.rows,
        "rows_match": summary["records_kept"] == index.rows,
        "logged_kept_by_split": dict(sorted(logged_split.items())),
        "observed_by_split": dict(sorted(observed_split.items())),
        "splits_match": dict(sorted(logged_split.items())) == dict(sorted(observed_split.items())),
    }


def diff_drop_reasons(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Per-reason delta between two runs' drop tables, plus the totals."""
    old_reasons = old.get("drop_reasons") or {}
    new_reasons = new.get("drop_reasons") or {}
    keys = sorted(set(old_reasons) | set(new_reasons))
    return {
        "per_reason": {
            key: {
                "old": old_reasons.get(key, 0),
                "new": new_reasons.get(key, 0),
                "delta": new_reasons.get(key, 0) - old_reasons.get(key, 0),
            }
            for key in keys
        },
        "total_old": sum(old_reasons.values()),
        "total_new": sum(new_reasons.values()),
        "total_delta": sum(new_reasons.values()) - sum(old_reasons.values()),
    }


# --------------------------------------------------------------------------- #
# §7  Drop-reason replay against the raw parquet  (opt-in, expensive)
#
# The two runs' drop tables are almost disjoint -- `missing_heavy` fired 79,105
# times in one and never in the other, `light_length_out_of_range` collapsed
# from six figures to single digits. Comparing them side by side shows THAT the
# filtering changed but not WHY, and "why" is exactly what has to be settled
# before the new corpus can be trusted: a filter that got looser for a reason
# nobody can name is a corpus with unknown contents.
#
# The replay reproduces both tables from the raw shards, changing exactly one
# thing between the passes -- the `clean_aa_sequence` implementation. If the
# legacy pass reproduces the 2026-06-11 log and the current pass reproduces the
# 2026-08-31 log, then that one function accounts for the entire difference AND
# the OLD run's filter parameters are pinned to the defaults used here, which is
# precisely the thing the regeneration pin says cannot be proven from artifacts.
# --------------------------------------------------------------------------- #

# The pre-fix body of `clean_aa_sequence`, verbatim from 10d31de (2026-06-10),
# the producer revision in force when the OLD corpus was written. `str(seq or "")`
# looks like a null guard, but a pandas NaN is a FLOAT and floats are truthy, so
# a missing chain became the string "nan" -> "NAN" -> two valid residues plus
# one, i.e. a three-residue antibody chain. Every downstream check then judged a
# fabricated sequence instead of an absent one. Reproduced here rather than
# imported because it no longer exists in the tree.
LEGACY_VALID_AA = set("ACDEFGHIKLMNPQRSTVWYBXZOU")
LEGACY_AA_ONLY = re.compile(r"[^A-Z]")


def legacy_clean_aa_sequence(seq: object) -> str:
    """`clean_aa_sequence` as it stood at commit 10d31de (pre-`pd.isna` guard)."""
    text = str(seq or "").upper().replace(" ", "")
    text = LEGACY_AA_ONLY.sub("", text)
    return "".join(ch for ch in text if ch in LEGACY_VALID_AA)


def default_filter_args(val_percent: int = 10) -> Any:
    """
    The producer's documented DEFAULTS, as a hypothesis to test -- not a fact.

    The regeneration pin records that the NEW run used all defaults and that the
    OLD run's parameters were never written down. Feeding these values into the
    replay turns "we assume OLD used defaults" into a falsifiable prediction:
    the replay's legacy drop table must equal the OLD log's, digit for digit.
    """
    return argparse.Namespace(
        val_percent=val_percent,
        min_heavy=70,
        max_heavy=180,
        min_light=70,
        max_light=170,
        min_antigen=8,
        max_antigen=2048,
        allowed_confidence="high,very_high",
    )


def replay_drop_reasons(rows: Iterable[Dict[str, Any]], producer, args: Any) -> Counter:
    """
    Re-derive each input row's filter outcome under BOTH cleaning implementations.

    Only `clean_aa_sequence` differs between the passes; `build_chain_features`,
    `normalize_confidence` and `keep_record` are the producer's own current
    functions in both, and `keep_record` is byte-identical between 10d31de and
    HEAD (checked before writing this). Swapping the module-level name is what
    makes the comparison a one-variable experiment instead of two independent
    reimplementations that could each be wrong.

    Args:
        rows: raw parquet rows as dicts.
        producer: the imported `prepare_antibody_antigen` module.
        args: filter parameters (see `default_filter_args`).

    Returns:
        Counter over ``(legacy_reason, current_reason)`` pairs -- the cross-tab,
        which is strictly more informative than two marginal tables because it
        says which rows moved between which reasons.
    """
    current_clean = producer.clean_aa_sequence
    cross: Counter = Counter()

    def outcome(row: Dict[str, Any]) -> str:
        heavy = producer.build_chain_features(row, "heavy")["heavy_variable_aa"]
        light = producer.build_chain_features(row, "light")["light_variable_aa"]
        return producer.keep_record(
            heavy_variable_aa=heavy,
            light_variable_aa=light,
            antigen_sequence=producer.clean_aa_sequence(row.get("antigen_sequence")),
            confidence=producer.normalize_confidence(row.get("confidence")),
            args=args,
        )[1]

    try:
        for row in rows:
            producer.clean_aa_sequence = legacy_clean_aa_sequence
            legacy_reason = outcome(row)
            producer.clean_aa_sequence = current_clean
            current_reason = outcome(row)
            cross[(legacy_reason, current_reason)] += 1
    finally:
        producer.clean_aa_sequence = current_clean
    return cross


def summarize_replay(
    cross: Counter, old_summary: Dict[str, Any], new_summary: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Turn the cross-tab into marginals and check them against the two run logs.

    `legacy_matches_old_log` is the load-bearing assertion: it simultaneously
    confirms that the OLD run used the default filter parameters and that the
    pre-fix `clean_aa_sequence` is the whole story behind the drop-table shift.
    Either marginal failing to match means the replay has NOT explained the
    difference, and the report must say so.
    """
    legacy: Counter = Counter()
    current: Counter = Counter()
    for (legacy_reason, current_reason), count in cross.items():
        legacy[legacy_reason] += count
        current[current_reason] += count

    def drops(marginal: Counter) -> Dict[str, int]:
        return {reason: count for reason, count in sorted(marginal.items()) if reason != "kept"}

    legacy_drops = drops(legacy)
    current_drops = drops(current)
    return {
        "rows_replayed": sum(cross.values()),
        "legacy_clean_aa_outcomes": dict(sorted(legacy.items())),
        "current_clean_aa_outcomes": dict(sorted(current.items())),
        "cross_tab": {f"{a} -> {b}": c for (a, b), c in sorted(cross.items())},
        "rows_whose_outcome_changed": sum(c for (a, b), c in cross.items() if a != b),
        "legacy_matches_old_log": legacy_drops == (old_summary.get("drop_reasons") or None),
        "current_matches_new_log": current_drops == (new_summary.get("drop_reasons") or None),
    }


def iter_parquet_rows(parquet_dir: Path) -> Iterator[Dict[str, Any]]:
    """Stream raw shards in the producer's own sorted order. Read-only."""
    import pandas as pd

    paths = [parquet_dir] if parquet_dir.is_file() else sorted(parquet_dir.glob("*.parquet"))
    for path in paths:
        print(f"  replaying {path.name} ...", flush=True)
        for row in pd.read_parquet(path).to_dict("records"):
            yield row


# --------------------------------------------------------------------------- #
# Verdict assembly
# --------------------------------------------------------------------------- #
def collect_unexplained(report: Dict[str, Any]) -> List[str]:
    """
    Gather every difference the script could not tie to a named cause.

    Each entry is a sentence a reader can act on. An empty list is the only
    state in which the report supports designating the new corpus; anything here
    is a blocker until a human explains it, and the script deliberately does not
    offer a way to suppress an entry.
    """
    problems: List[str] = []

    ids = report["record_ids"]
    if not ids["stability"]["stable"]:
        problems.append(
            f"record_id is NOT stable across runs: "
            f"{ids['stability']['shared_ids_with_changed_biology']:,} shared ids carry different biology."
        )

    shared = report["per_row_biology"]["by_record_id"]
    if shared["rows_with_changed_biology"]:
        problems.append(
            f"{shared['rows_with_changed_biology']:,} rows kept the same record_id but changed "
            "heavy/light/antigen sequence."
        )
    if shared["rows_with_changed_labels"]:
        problems.append(
            f"{shared['rows_with_changed_labels']:,} rows kept the same record_id but changed "
            "(binder_label, is_strong_binder, affinity_type)."
        )

    bio_join = report["per_row_biology"]["by_biology_triple"]
    if bio_join["rows_with_changed_labels"]:
        problems.append(
            f"{bio_join['rows_with_changed_labels']:,} biological triples present in both corpora "
            "carry different labels."
        )

    attribution = report["split_churn"]
    if not attribution["old_key_hypothesis_holds"]:
        problems.append(
            "OLD splits do not all follow deterministic_split(target_key, val_percent="
            f"{attribution['val_percent_assumed']}): "
            f"{attribution['shared_rows'] - attribution['old_rows_following_legacy_key']:,} rows disagree. "
            "The OLD run's parameters are therefore NOT the assumed defaults."
        )
    if not attribution["new_key_hypothesis_holds"]:
        problems.append(
            "NEW splits do not all follow deterministic_split(canonical_target_id, val_percent="
            f"{attribution['val_percent_assumed']}): "
            f"{attribution['shared_rows'] - attribution['new_rows_following_canonical_id']:,} rows disagree."
        )
    if attribution["unattributed_changes"]:
        problems.append(
            f"{attribution['unattributed_changes']:,} rows changed split for a reason canonicalization "
            "does not explain."
        )

    fusions = report["canonical_identity"]["fusions"]
    if fusions["legacy_keys_spanning_multiple_components"]:
        problems.append(
            f"{fusions['legacy_keys_spanning_multiple_components']:,} legacy target_keys landed in more "
            "than one canonical component -- canonicalization is supposed to only coarsen the partition."
        )

    canonical = report["canonical_identity"]
    if canonical["new_rows_with_canonical_target_id"] != canonical["new_rows"]:
        problems.append(
            f"{canonical['new_rows'] - canonical['new_rows_with_canonical_target_id']:,} NEW rows carry no "
            "canonical_target_id."
        )

    for tag in ("old", "new"):
        accounting = report["producer_logs"][tag]["row_accounting"]
        if accounting.get("checkable") and not accounting["balances"]:
            problems.append(
                f"{tag.upper()} producer log does not balance: kept + duplicates + drops = "
                f"{accounting['accounted_total']:,} but rows_seen = {accounting['rows_seen']:,}."
            )
        provenance = report["producer_logs"][tag]["matches_corpus"]
        if provenance.get("checkable") and not (provenance["rows_match"] and provenance["splits_match"]):
            problems.append(
                f"{tag.upper()} producer log does not describe the {tag.upper()} corpus on disk "
                "(row count or per-split counts disagree); its numbers cannot explain that corpus."
            )

    membership = report["membership"]
    kept_delta = membership["producer_kept_delta"]
    net = membership["rows_only_in_new"] - membership["rows_only_in_old"]
    if kept_delta is None:
        problems.append(
            "the producer logs do not both report records_kept, so the membership delta cannot be "
            "checked against the producer's own accounting."
        )
    elif net != kept_delta:
        problems.append(
            f"membership delta does not match the producer's kept-count delta: observed net "
            f"{net:+,} rows but the logs report {kept_delta:+,}."
        )

    replay = report.get("drop_reason_replay")
    if replay:
        if not replay["legacy_matches_old_log"]:
            problems.append(
                "the drop-reason replay under the PRE-FIX clean_aa_sequence does not reproduce the "
                "2026-06-11 log, so the OLD run's filter parameters are NOT the defaults assumed here "
                "and the drop-table difference remains unexplained."
            )
        if not replay["current_matches_new_log"]:
            problems.append(
                "the drop-reason replay under the CURRENT clean_aa_sequence does not reproduce the "
                "2026-08-31 log; the replay does not describe the run that wrote the NEW corpus."
            )

    detail = membership["detail"]
    if detail["only_old_biology_gone"]:
        problems.append(
            f"{detail['only_old_biology_gone']:,} biological triples present in OLD are absent from NEW "
            "entirely (not merely represented by a different source row)."
        )

    return problems


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def build_report(
    old: CorpusIndex,
    new: CorpusIndex,
    old_summary: Dict[str, Any],
    new_summary: Dict[str, Any],
    split_fn,
    val_percent: int,
    replay: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the full comparison from two already-built indexes."""
    stability = verify_record_id_stability(old, new)
    rid_sets = compare_key_sets(old.by_rid.keys(), new.by_rid.keys())
    bio_sets = compare_key_sets(old.bio_to_rid.keys(), new.bio_to_rid.keys())

    legacy_pairs = {
        (new.string(legacy), new.string(canonical)): count
        for (legacy, canonical), count in new.legacy_to_canonical.items()
    }

    kept_delta = None
    if "records_kept" in old_summary and "records_kept" in new_summary:
        kept_delta = new_summary["records_kept"] - old_summary["records_kept"]

    report: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "counts": {
            "old": {"rows": old.rows, "by_split": dict(sorted(old.by_split.items()))},
            "new": {"rows": new.rows, "by_split": dict(sorted(new.by_split.items()))},
            "row_delta": new.rows - old.rows,
            "val_fraction_old": (old.by_split.get("val", 0) / old.rows) if old.rows else None,
            "val_fraction_new": (new.by_split.get("val", 0) / new.rows) if new.rows else None,
            "duplicate_record_ids_old": old.duplicate_record_id,
            "duplicate_record_ids_new": new.duplicate_record_id,
            "duplicate_biology_triples_old": old.duplicate_biology,
            "duplicate_biology_triples_new": new.duplicate_biology,
        },
        "record_ids": {"sets": rid_sets, "stability": stability},
        "biology_triples": {"sets": bio_sets},
        "multisets": {
            "heavy": multiset_delta(old.heavy, new.heavy),
            "light": multiset_delta(old.light, new.light),
            "antigen": multiset_delta(old.antigen, new.antigen),
            "labels": _label_multiset_delta(old, new),
        },
        "per_row_biology": {
            "by_record_id": compare_biology_on_shared_rows(old, new, "record_id"),
            "by_biology_triple": compare_biology_on_shared_rows(old, new, "biology_triple"),
        },
        "canonical_identity": {
            "old_rows": old.rows,
            "new_rows": new.rows,
            "old_has_canonical_field": bool(old.field_present.get("canonical_target_id")),
            "new_has_canonical_field": bool(new.field_present.get("canonical_target_id")),
            "old_distinct_target_key": len(old.target_key_rows),
            "new_distinct_target_key": len(new.target_key_rows),
            "new_distinct_canonical_target_id": len(new.canonical_rows),
            "new_rows_with_canonical_target_id": new.rows_with_canonical,
            "new_fraction_rows_with_canonical": (new.rows_with_canonical / new.rows) if new.rows else None,
            "fusions": fusion_report(legacy_pairs),
        },
        "split_churn": split_change_attribution(old, new, split_fn, val_percent),
        "producer_logs": {
            "old": {
                "summary": old_summary,
                "row_accounting": check_row_accounting(old_summary),
                "matches_corpus": validate_summary_against_corpus(old_summary, old),
            },
            "new": {
                "summary": new_summary,
                "row_accounting": check_row_accounting(new_summary),
                "matches_corpus": validate_summary_against_corpus(new_summary, new),
            },
            "drop_reason_delta": diff_drop_reasons(old_summary, new_summary),
        },
        "membership": {
            "rows_only_in_old": rid_sets["only_old"],
            "rows_only_in_new": rid_sets["only_new"],
            "producer_kept_delta": kept_delta,
            "detail": classify_membership_changes(old, new),
        },
    }
    if replay is not None:
        report["drop_reason_replay"] = replay
    report["unexplained"] = collect_unexplained(report)
    report["verdict"] = "CLEAN" if not report["unexplained"] else "BLOCKED"
    return report


def _label_multiset_delta(old: CorpusIndex, new: CorpusIndex) -> Dict[str, Any]:
    """
    Label multiset comparison, keyed by the label TUPLE rather than by intern id.

    The two indexes intern independently, so comparing raw ids would compare
    two unrelated numbering schemes and report a spurious total mismatch. This
    is the kind of powerless comparison the test suite pins.
    """
    old_counter = Counter({old.label(ident): count for ident, count in old.labels.items()})
    new_counter = Counter({new.label(ident): count for ident, count in new.labels.items()})
    delta = multiset_delta(old_counter, new_counter)
    per_label = {}
    for label in sorted(set(old_counter) | set(new_counter), key=lambda item: str(item)):
        before = old_counter.get(label, 0)
        after = new_counter.get(label, 0)
        if before != after:
            per_label[str(label)] = {"old": before, "new": after, "delta": after - before}
    delta["changed_label_signatures"] = per_label
    return delta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--old", type=Path, default=OLD_CORPUS)
    parser.add_argument("--new", type=Path, default=NEW_CORPUS)
    parser.add_argument("--old-log", type=Path, default=OLD_LOG)
    parser.add_argument("--new-log", type=Path, default=NEW_LOG)
    parser.add_argument("--val-percent", type=int, default=10,
                        help="Split parameter to TEST the corpora against (not to impose on them).")
    parser.add_argument(
        "--replay-drop-reasons",
        type=Path,
        default=None,
        metavar="PARQUET_DIR",
        help=(
            "Opt-in (~10 min, read-only): re-derive both runs' drop tables from the raw shards, "
            "changing only the clean_aa_sequence implementation, and check each against its log. "
            "This is what turns the near-disjoint drop tables from an observation into an explanation."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "outputs/asd-regeneration-reconciliation.json",
    )
    args = parser.parse_args()

    for path in (args.old, args.new):
        if not path.exists():
            print(f"REFUSED: missing corpus {path}")
            return 1

    producer = load_producer_module()

    print("reading OLD corpus ...")
    old = index_corpus(read_records(args.old))
    print(f"  rows={old.rows:,}  by_split={dict(old.by_split)}")
    print("reading NEW corpus ...")
    new = index_corpus(read_records(args.new))
    print(f"  rows={new.rows:,}  by_split={dict(new.by_split)}")

    old_summary = parse_producer_summary(read_log_text(args.old_log)) if args.old_log.exists() else {}
    new_summary = parse_producer_summary(read_log_text(args.new_log)) if args.new_log.exists() else {}

    replay = None
    if args.replay_drop_reasons is not None:
        if not args.replay_drop_reasons.exists():
            print(f"REFUSED: missing parquet input {args.replay_drop_reasons}")
            return 1
        print("replaying drop reasons from the raw shards (this is the slow part) ...")
        cross = replay_drop_reasons(
            iter_parquet_rows(args.replay_drop_reasons), producer, default_filter_args(args.val_percent)
        )
        replay = summarize_replay(cross, old_summary, new_summary)

    report = build_report(old, new, old_summary, new_summary,
                          producer.deterministic_split, args.val_percent, replay)

    report["corpora"] = {
        "old": {
            "path": _rel(args.old),
            "sha256": file_sha256(args.old),
            "bytes": args.old.stat().st_size,
            "log": _rel(args.old_log) if args.old_log.exists() else None,
        },
        "new": {
            "path": _rel(args.new),
            "sha256": file_sha256(args.new),
            "bytes": args.new.stat().st_size,
            "log": _rel(args.new_log) if args.new_log.exists() else None,
        },
    }
    report["provenance"] = {
        "producer": _rel(PRODUCER),
        "producer_sha256": file_sha256(PRODUCER),
        "source_commit": _git_commit(),
    }
    if PIN_JSON.exists():
        pin = json.loads(PIN_JSON.read_text(encoding="utf-8"))
        report["provenance"]["pin"] = {
            "path": _rel(PIN_JSON),
            "pinned_producer_sha256": pin.get("producer_sha256"),
            "pinned_source_commit": pin.get("source_commit"),
            "pinned_old_sha256": (pin.get("previous_corpus") or {}).get("sha256"),
            "producer_matches_pin": pin.get("producer_sha256") == report["provenance"]["producer_sha256"],
            "old_corpus_matches_pin": (pin.get("previous_corpus") or {}).get("sha256")
            == report["corpora"]["old"]["sha256"],
        }
        if not report["provenance"]["pin"]["old_corpus_matches_pin"]:
            report["unexplained"].append(
                "OLD corpus sha256 does not match the sha256 recorded in outputs/asd-regeneration-pin.json."
            )
        if not report["provenance"]["pin"]["producer_matches_pin"]:
            report["unexplained"].append(
                "producer sha256 does not match the pin; the NEW corpus was written by different code "
                "than the one this reconciliation re-derives splits with."
            )
        report["verdict"] = "CLEAN" if not report["unexplained"] else "BLOCKED"

    _print_report(report)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8", newline="") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    print(f"\nwrote {_rel(args.output_json)}")
    return 0


def _git_commit() -> Optional[str]:
    """Best-effort HEAD sha so the artifact names the code that produced it."""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, timeout=30, check=False,
        )
        return out.stdout.strip() or None
    except Exception:  # pragma: no cover - git absent
        return None


def _print_report(report: Dict[str, Any]) -> None:
    counts = report["counts"]
    print("\n=== 1. ROWS AND SPLITS ===")
    print(f"  OLD rows {counts['old']['rows']:>9,}  {counts['old']['by_split']}")
    print(f"  NEW rows {counts['new']['rows']:>9,}  {counts['new']['by_split']}")
    print(f"  delta    {counts['row_delta']:>+9,}   "
          f"val fraction {counts['val_fraction_old']:.4%} -> {counts['val_fraction_new']:.4%}")

    ids = report["record_ids"]
    print("\n=== 2. RECORD IDS ===")
    print(f"  only OLD {ids['sets']['only_old']:>9,}   only NEW {ids['sets']['only_new']:>9,}   "
          f"both {ids['sets']['in_both']:>9,}")
    print(f"  stability: {'STABLE' if ids['stability']['stable'] else 'NOT STABLE'} "
          f"({ids['stability']['shared_ids_with_changed_biology']:,} shared ids changed biology) "
          f"-> joining on {ids['stability']['key_used_for_row_join']}")
    bio = report["biology_triples"]["sets"]
    print(f"  biology triples: only OLD {bio['only_old']:,}  only NEW {bio['only_new']:,}  both {bio['in_both']:,}")

    print("\n=== 3. BIOLOGICAL MULTISETS ===")
    for name, block in report["multisets"].items():
        print(f"  {name:8} distinct {block['distinct_old']:>8,} -> {block['distinct_new']:>8,}   "
              f"rows only-OLD {block['rows_only_in_old']:>8,}  only-NEW {block['rows_only_in_new']:>8,}   "
              f"equal={block['multisets_equal']}")
    per_row = report["per_row_biology"]["by_record_id"]
    print(f"  shared record_ids with CHANGED biology: {per_row['rows_with_changed_biology']:,}")
    print(f"  shared record_ids with CHANGED labels:  {per_row['rows_with_changed_labels']:,}")
    print(f"  shared biology triples with CHANGED labels: "
          f"{report['per_row_biology']['by_biology_triple']['rows_with_changed_labels']:,}")

    identity = report["canonical_identity"]
    print("\n=== 4. CANONICAL TARGET IDENTITY ===")
    print(f"  distinct target_key      OLD {identity['old_distinct_target_key']:,}  "
          f"NEW {identity['new_distinct_target_key']:,}")
    print(f"  distinct canonical id    NEW {identity['new_distinct_canonical_target_id']:,}")
    print(f"  rows carrying canonical  NEW {identity['new_rows_with_canonical_target_id']:,} / "
          f"{identity['new_rows']:,} ({identity['new_fraction_rows_with_canonical']:.4%})")
    fus = identity["fusions"]
    print(f"  fused components {fus['fused_components']:,} absorbing "
          f"{fus['legacy_keys_absorbed_by_fusions']:,} legacy keys "
          f"({fus['rows_in_fused_components']:,} rows)")
    print(f"  legacy keys spanning >1 component: {fus['legacy_keys_spanning_multiple_components']:,}")
    for item in fus["largest_fusions"][:10]:
        print(f"    {item['canonical_target_id'][:48]:48} keys={item['legacy_key_count']:>4}  "
              f"rows={item['rows']:>8,}")

    churn = report["split_churn"]
    print("\n=== 5. SPLIT CHURN ===")
    print(f"  OLD split follows target_key:        {churn['old_rows_following_legacy_key']:,} / "
          f"{churn['shared_rows']:,}  -> {churn['old_key_hypothesis_holds']}")
    print(f"  NEW split follows canonical id:      {churn['new_rows_following_canonical_id']:,} / "
          f"{churn['shared_rows']:,}  -> {churn['new_key_hypothesis_holds']}")
    print(f"  transitions: {churn['transitions']}")
    print(f"  rows changing split: {churn['rows_with_changed_split']:,}  "
          f"(canonicalization explains {churn['attributable_to_canonicalization']:,}, "
          f"UNATTRIBUTED {churn['unattributed_changes']:,})")
    print(f"  rows changing target_key: {churn['rows_with_changed_target_key']:,}")

    print("\n=== 6. PRODUCER DROP ACCOUNTING ===")
    for tag in ("old", "new"):
        block = report["producer_logs"][tag]
        acct = block["row_accounting"]
        prov = block["matches_corpus"]
        if acct.get("checkable"):
            print(f"  {tag.upper()}: rows_seen {acct['rows_seen']:,} = kept {acct['records_kept']:,} + "
                  f"dup {acct['duplicates_dropped']:,} + drops {acct['rows_dropped_by_filters']:,}  "
                  f"balances={acct['balances']}")
        else:
            print(f"  {tag.upper()}: no parsable summary")
        if prov.get("checkable"):
            print(f"        log matches corpus on disk: rows={prov['rows_match']} splits={prov['splits_match']}")
    for reason, values in report["producer_logs"]["drop_reason_delta"]["per_reason"].items():
        print(f"    {reason:28} OLD {values['old']:>8,}  NEW {values['new']:>8,}  {values['delta']:>+8,}")

    membership = report["membership"]
    detail = membership["detail"]
    print("\n=== 6b. MEMBERSHIP ===")
    print(f"  only OLD {detail['only_old_rows']:,} "
          f"(biology still present under another row: {detail['only_old_biology_still_present']:,}; "
          f"biology GONE: {detail['only_old_biology_gone']:,})")
    print(f"  only NEW {detail['only_new_rows']:,} "
          f"(biology already present under another row: {detail['only_new_biology_already_present']:,}; "
          f"biology ADDED: {detail['only_new_biology_added']:,})")
    print(f"  producer kept delta from logs: {membership['producer_kept_delta']}")

    replay = report.get("drop_reason_replay")
    if replay:
        print("\n=== 7. DROP-REASON REPLAY (raw parquet, one variable: clean_aa_sequence) ===")
        print(f"  rows replayed: {replay['rows_replayed']:,}   "
              f"outcome changed for {replay['rows_whose_outcome_changed']:,}")
        print(f"  pre-fix clean_aa_sequence reproduces the 2026-06-11 log: "
              f"{replay['legacy_matches_old_log']}")
        print(f"  current clean_aa_sequence reproduces the 2026-08-31 log: "
              f"{replay['current_matches_new_log']}")
        for transition, count in replay["cross_tab"].items():
            before, after = transition.split(" -> ")
            if before != after:
                print(f"    {transition:62} {count:>9,}")

    print("\n=== VERDICT ===")
    if report["unexplained"]:
        print("  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("  !!  UNEXPLAINED DIFFERENCES -- DO NOT DESIGNATE AS STAGE 3 SOURCE  !!")
        print("  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        for problem in report["unexplained"]:
            print(f"    - {problem}")
    else:
        print("  every measured difference is attributable to a named cause.")


if __name__ == "__main__":
    raise SystemExit(main())
