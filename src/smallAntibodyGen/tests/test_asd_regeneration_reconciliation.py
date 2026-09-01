"""
The regeneration reconciliation must be able to FAIL.

This file exists because `scripts/reconcile_asd_regeneration.py` is the artifact
that decides whether the regenerated ASD corpus may be designated the Stage 3
source. A reconciliation that reports "no unexplained differences" because its
comparisons are powerless is worse than no reconciliation at all -- it launders
an unchecked corpus into a trusted one. Every test below therefore pins a way
the script could be silently permissive:

- **Multisets, not sets.** A corpus that duplicated a row and one that did not
  are different corpora. `multiset_delta` must see multiplicity, so the fixtures
  deliberately hold the DISTINCT values equal and change only the counts. A set
  comparison passes that fixture; a multiset comparison does not.
- **Absent is not empty.** A heavy-only row and a row whose light chain is the
  empty string must not be distinguishable from each other (the producer's
  dedupe collapses them), but neither may collide with a row that HAS a light
  chain. Both directions are pinned.
- **Interning is per-corpus.** Each `CorpusIndex` interns strings and label
  tuples into its own numbering, so any comparison that leaks an intern id
  across corpora compares two unrelated numbering schemes. `_label_multiset_delta`
  exists precisely to avoid that, and the fixture makes the two corpora intern
  the SAME label at DIFFERENT ids so a naive id comparison would report a
  spurious total mismatch.
- **record_id stability is measured, not assumed.** The fixture supplies a
  corpus pair where an id is reused for different biology, and the verifier must
  say NOT STABLE.
- **Attribution must leave a residual.** A split that moved for a reason
  canonicalization does not explain must land in `unattributed_changes`. The
  fixture constructs exactly that row, and separately constructs a row whose
  move IS explained, so the test cannot pass by classifying everything one way.
- **Fusion is a coarsening.** A legacy key that lands in two canonical
  components is the opposite of the intended effect; the report must count it.
- **The producer log is evidence, not proof.** `validate_summary_against_corpus`
  must reject a log whose counts do not describe the corpus on disk, or the OLD
  run's unproven parameters get laundered into the report as fact.
- **Row accounting must actually balance.** A drop table that does not add up
  cannot carry an explanation, and the checker must notice.

Fixture power: several tests assert on their own fixtures first (e.g. that the
two corpora really do intern the same label at different ids, that the fused
component really does contain two legacy keys). This repo has been bitten by
tests that passed against broken code because the fixture never reached the
condition under test.
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def recon(project_root: Path):
    script = project_root.parents[1] / "scripts" / "reconcile_asd_regeneration.py"
    spec = importlib.util.spec_from_file_location("reconcile_asd_regeneration", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_record(
    record_id: str,
    *,
    split: str = "train",
    target_key: str = "uniprot:p1",
    canonical: str | None = None,
    heavy: str = "HHHH",
    light: str | None = "LLLL",
    antigen: str = "AAAA",
    binder_label: int | None = 1,
    is_strong_binder: bool = True,
    affinity_type: str = "bool",
) -> dict:
    """One processed row, carrying only the fields the reconciliation reads."""
    record = {
        "record_id": record_id,
        "split": split,
        "target_key": target_key,
        "sequence_heavy": heavy,
        "sequence_light": light,
        "sequence_antigen": antigen,
        "binder_label": binder_label,
        "is_strong_binder": is_strong_binder,
        "affinity_type": affinity_type,
    }
    if canonical is not None:
        record["canonical_target_id"] = canonical
    return record


# --------------------------------------------------------------------------- #
# Absent is not empty; the natural key is the producer's dedupe triple
# --------------------------------------------------------------------------- #
def test_absent_and_empty_sequence_fields_collapse_like_the_producer(recon):
    """
    `write_record` builds its dedupe key with `str(x or "")`, so a `None` light
    chain and an `""` light chain ARE the same row to the producer. The
    reconciliation must agree, or it would report a phantom difference on every
    heavy-only record.
    """
    assert recon.sequence_digest(None) == recon.sequence_digest("")
    assert recon.sequence_digest(None) == recon.ABSENT


def test_present_sequence_never_collides_with_absent(recon):
    """The collapse above must not extend to a chain that actually exists."""
    assert recon.sequence_digest("L") != recon.ABSENT
    assert recon.sequence_digest("L") != recon.sequence_digest("K")


def test_biology_key_is_the_producer_dedupe_triple_and_is_order_sensitive(recon):
    """
    Heavy/light/antigen occupy fixed 8-byte slots, so swapping two chains must
    change the key. A key built by hashing an unordered set would not.
    """
    assert recon.BIOLOGY_FIELDS == ("sequence_heavy", "sequence_light", "sequence_antigen")
    straight = recon.biology_key(make_record("a", heavy="H", light="L", antigen="A"))
    swapped = recon.biology_key(make_record("a", heavy="L", light="H", antigen="A"))
    assert straight != swapped
    assert len(straight) == 3 * recon.DIGEST_BYTES
    # The heavy slot is exactly the heavy digest -- the slicing the multisets rely on.
    assert straight[: recon.DIGEST_BYTES] == recon.sequence_digest("H")


def test_label_signature_keeps_none_distinct_from_zero(recon):
    """
    `binder_label` is None for every non-bool affinity type. Coercing it to 0
    would convert "unlabelled" into "negative" and make a real label change
    invisible.
    """
    unlabelled = recon.label_signature(make_record("a", binder_label=None))
    negative = recon.label_signature(make_record("a", binder_label=0))
    assert unlabelled != negative
    assert unlabelled[0] is None and negative[0] == 0


# --------------------------------------------------------------------------- #
# Multisets, not sets
# --------------------------------------------------------------------------- #
def test_multiset_delta_sees_multiplicity_a_set_comparison_would_miss(recon):
    """
    Fixture power check first: the two corpora have IDENTICAL distinct value
    sets. Any set-based comparison reports "equal". Only counting multiplicity
    finds the difference.
    """
    from collections import Counter

    old = Counter({"a": 5, "b": 1})
    new = Counter({"a": 2, "b": 1})
    assert set(old) == set(new), "fixture is powerless if the value sets differ"

    delta = recon.multiset_delta(old, new)
    assert delta["distinct_only_old"] == 0
    assert delta["distinct_only_new"] == 0
    assert delta["multisets_equal"] is False
    assert delta["rows_only_in_old"] == 3
    assert delta["rows_only_in_new"] == 0


def test_multiset_delta_reports_equality_only_when_counts_match(recon):
    from collections import Counter

    same = Counter({"a": 5, "b": 1})
    assert recon.multiset_delta(same, Counter(same))["multisets_equal"] is True


def test_label_multiset_is_compared_by_tuple_not_by_intern_id(recon):
    """
    Each index interns independently. Here the two corpora see the same two
    label tuples in OPPOSITE order, so the shared label has intern id 0 in one
    and id 1 in the other. Comparing ids would report a mismatch; the real
    answer is that the multisets are equal.

    The two labels are given DIFFERENT multiplicities on purpose. With equal
    multiplicities an id-keyed comparison would coincidentally still come out
    equal, and this test would pass against exactly the bug it exists to catch.
    """
    def bool_row(rid: str) -> dict:
        return make_record(rid, binder_label=1, affinity_type="bool")

    def fuzzy_row(rid: str) -> dict:
        return make_record(rid, binder_label=None, is_strong_binder=False, affinity_type="fuzzy",
                           heavy=f"H{rid}")

    old = recon.index_corpus([bool_row("a"), bool_row("b"), fuzzy_row("c")])
    new = recon.index_corpus([fuzzy_row("c"), bool_row("a"), bool_row("b")])

    # Fixture power, part 1: the interning really is inverted between the indexes.
    assert old.label(0) != new.label(0), "fixture is powerless if both intern identically"
    assert old.label(0) == new.label(1)
    # Fixture power, part 2: the id-keyed multisets DISAGREE, so a comparison
    # that leaked intern ids across corpora would be caught here.
    from collections import Counter

    assert Counter(dict(old.labels)) != Counter(dict(new.labels))

    delta = recon._label_multiset_delta(old, new)
    assert delta["multisets_equal"] is True
    assert delta["changed_label_signatures"] == {}


def test_label_multiset_reports_a_real_label_change(recon):
    """The mirror of the test above: an actual change must survive interning."""
    old = recon.index_corpus([make_record("a", binder_label=1)])
    new = recon.index_corpus([make_record("a", binder_label=0, is_strong_binder=False)])
    delta = recon._label_multiset_delta(old, new)
    assert delta["multisets_equal"] is False
    assert len(delta["changed_label_signatures"]) == 2


# --------------------------------------------------------------------------- #
# record_id stability is measured
# --------------------------------------------------------------------------- #
def test_record_id_stability_flags_an_id_reused_for_different_biology(recon):
    old = recon.index_corpus([make_record("shard:0", heavy="HHHH")])
    new = recon.index_corpus([make_record("shard:0", heavy="WWWW")])
    verdict = recon.verify_record_id_stability(old, new)
    assert verdict["stable"] is False
    assert verdict["shared_ids_with_changed_biology"] == 1
    assert verdict["key_used_for_row_join"] == "biology_triple"
    assert verdict["sample_mismatched_ids"] == ["shard:0"]


def test_record_id_stability_accepts_ids_whose_biology_is_unchanged(recon):
    """
    A row whose SPLIT and TARGET moved but whose biology held still must not be
    called unstable -- that is exactly the intended effect of canonicalization,
    and flagging it would drown the real signal.
    """
    old = recon.index_corpus([make_record("shard:0", split="val", target_key="pdb:1abc")])
    new = recon.index_corpus(
        [make_record("shard:0", split="train", target_key="pdb:1abc", canonical="uniprot:p1")]
    )
    verdict = recon.verify_record_id_stability(old, new)
    assert verdict["stable"] is True
    assert verdict["key_used_for_row_join"] == "record_id"


def test_changed_biology_under_a_stable_id_is_reported_per_row(recon):
    old = recon.index_corpus([make_record("shard:0", antigen="AAAA"), make_record("shard:1")])
    new = recon.index_corpus([make_record("shard:0", antigen="CCCC"), make_record("shard:1")])
    result = recon.compare_biology_on_shared_rows(old, new, "record_id")
    assert result["shared_rows"] == 2
    assert result["rows_with_changed_biology"] == 1
    assert result["sample_changed_biology"] == ["shard:0"]


def test_changed_labels_are_reported_under_the_biology_join(recon):
    """
    Joining on biology answers the only question left once the sequences match:
    did the supervision attached to this pairing move?
    """
    old = recon.index_corpus([make_record("shard:0", binder_label=1, is_strong_binder=True)])
    new = recon.index_corpus([make_record("shard:9", binder_label=0, is_strong_binder=False)])
    result = recon.compare_biology_on_shared_rows(old, new, "biology_triple")
    assert result["shared_rows"] == 1
    assert result["rows_with_changed_labels"] == 1


# --------------------------------------------------------------------------- #
# Membership: representation churn is not data loss
# --------------------------------------------------------------------------- #
def test_membership_separates_representation_churn_from_lost_biology(recon):
    """
    Fixture: `shard:0` disappears but its triple survives as `shard:7` (dedupe
    chose a different representative), while `shard:1`'s triple leaves entirely.
    Collapsing those two into one "rows dropped" number is the misread this
    function exists to prevent.
    """
    old = recon.index_corpus(
        [
            make_record("shard:0", heavy="AAAA"),
            make_record("shard:1", heavy="BBBB"),
        ]
    )
    new = recon.index_corpus(
        [
            make_record("shard:7", heavy="AAAA"),
            make_record("shard:8", heavy="CCCC"),
        ]
    )
    detail = recon.classify_membership_changes(old, new)
    assert detail["only_old_rows"] == 2
    assert detail["only_old_biology_still_present"] == 1
    assert detail["only_old_biology_gone"] == 1
    assert detail["sample_only_old_biology_gone"] == ["shard:1"]
    assert detail["only_new_rows"] == 2
    assert detail["only_new_biology_already_present"] == 1
    assert detail["only_new_biology_added"] == 1


# --------------------------------------------------------------------------- #
# Fusion is a coarsening of the legacy partition
# --------------------------------------------------------------------------- #
def test_fusion_report_counts_absorbed_legacy_keys(recon):
    pairs = {
        ("uniprot:p1", "uniprot:p1"): 10,
        ("pdb:1abc", "uniprot:p1"): 5,
        ("name:spike", "uniprot:p1"): 1,
        ("uniprot:p9", "uniprot:p9"): 7,
    }
    report = recon.fusion_report(pairs)
    assert report["distinct_legacy_keys"] == 4
    assert report["distinct_canonical_ids"] == 2
    assert report["fused_components"] == 1
    assert report["legacy_keys_absorbed_by_fusions"] == 3
    assert report["rows_in_fused_components"] == 16
    assert report["largest_fusions"][0]["canonical_target_id"] == "uniprot:p1"
    assert report["largest_fusions"][0]["legacy_key_count"] == 3
    assert report["legacy_keys_spanning_multiple_components"] == 0


def test_fusion_report_flags_a_legacy_key_that_split_across_components(recon):
    """
    Canonicalization is supposed to only MERGE legacy keys. One legacy key
    landing in two components is unpredicted behaviour and must be surfaced,
    not averaged into the fusion counts.
    """
    pairs = {
        ("name:spike", "uniprot:p1"): 3,
        ("name:spike", "uniprot:p2"): 4,
    }
    report = recon.fusion_report(pairs)
    assert report["legacy_keys_spanning_multiple_components"] == 1
    assert report["sample_split_legacy_keys"]["name:spike"] == ["uniprot:p1", "uniprot:p2"]


def test_fusion_report_ignores_singleton_components(recon):
    """A canonical id covering one legacy key is not a fusion."""
    report = recon.fusion_report({("uniprot:p1", "uniprot:p1"): 3})
    assert report["fused_components"] == 0
    assert report["legacy_keys_absorbed_by_fusions"] == 0
    assert report["largest_fusions"] == []


# --------------------------------------------------------------------------- #
# Split attribution must leave a residual when one is due
# --------------------------------------------------------------------------- #
def _hash_split(key: str, val_percent: int = 10) -> str:
    """A stand-in for the producer's `deterministic_split` in unit tests."""
    import hashlib

    bucket = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "val" if bucket < val_percent else "train"


def _find_keys():
    """
    Search for concrete keys with the bucket behaviour each test needs.

    Constructing these by hand is impossible (the buckets come from SHA-1), and
    hard-coding them would silently rot if the hashing changed. Searching makes
    the fixture self-verifying: if no key with the required property exists the
    test errors out instead of quietly testing nothing.
    """
    legacy_val = next(k for k in (f"pdb:{i}" for i in range(10000)) if _hash_split(k) == "val")
    legacy_train = next(k for k in (f"pdb:{i}" for i in range(10000)) if _hash_split(k) == "train")
    canonical_train = next(
        k for k in (f"uniprot:{i}" for i in range(10000)) if _hash_split(k) == "train"
    )
    return legacy_val, legacy_train, canonical_train


def test_split_change_is_attributed_to_canonicalization_when_the_buckets_differ(recon):
    """
    Row whose legacy key hashes to val and whose canonical id hashes to train:
    the move IS explained. The fixture asserts the bucket difference exists
    before relying on it.
    """
    legacy_val, _, canonical_train = _find_keys()
    assert _hash_split(legacy_val) == "val" and _hash_split(canonical_train) == "train"

    old = recon.index_corpus([make_record("s:0", split="val", target_key=legacy_val)])
    new = recon.index_corpus(
        [make_record("s:0", split="train", target_key=legacy_val, canonical=canonical_train)]
    )
    result = recon.split_change_attribution(old, new, _hash_split, 10)
    assert result["rows_with_changed_split"] == 1
    assert result["attributable_to_canonicalization"] == 1
    assert result["unattributed_changes"] == 0
    assert result["transitions"] == {"val->train": 1}


def test_split_change_with_identical_buckets_is_left_unattributed(recon):
    """
    Row whose canonical id hashes to the SAME bucket as its legacy key but whose
    split moved anyway. Nothing about canonicalization explains that, and the
    residual must say so -- this is the number that blocks designation.
    """
    _, legacy_train, canonical_train = _find_keys()
    assert _hash_split(legacy_train) == _hash_split(canonical_train) == "train"

    old = recon.index_corpus([make_record("s:0", split="val", target_key=legacy_train)])
    new = recon.index_corpus(
        [make_record("s:0", split="train", target_key=legacy_train, canonical=canonical_train)]
    )
    result = recon.split_change_attribution(old, new, _hash_split, 10)
    assert result["rows_with_changed_split"] == 1
    assert result["attributable_to_canonicalization"] == 0
    assert result["unattributed_changes"] == 1
    assert result["sample_unattributed"] == ["s:0"]


def test_split_attribution_reports_when_the_old_key_hypothesis_fails(recon):
    """
    If the OLD corpus's split does not follow `deterministic_split(target_key)`,
    the OLD run did not use the parameters we assumed and every downstream
    attribution is void. That must be visible, not swallowed.
    """
    _, legacy_train, canonical_train = _find_keys()
    old = recon.index_corpus([make_record("s:0", split="val", target_key=legacy_train)])
    new = recon.index_corpus(
        [make_record("s:0", split="train", target_key=legacy_train, canonical=canonical_train)]
    )
    result = recon.split_change_attribution(old, new, _hash_split, 10)
    assert result["old_rows_following_legacy_key"] == 0
    assert result["old_key_hypothesis_holds"] is False
    assert result["new_key_hypothesis_holds"] is True


def test_split_attribution_confirms_both_hypotheses_when_they_hold(recon):
    """The positive control: a corpus pair that behaves exactly as designed."""
    legacy_val, _, canonical_train = _find_keys()
    old = recon.index_corpus([make_record("s:0", split="val", target_key=legacy_val)])
    new = recon.index_corpus(
        [make_record("s:0", split="train", target_key=legacy_val, canonical=canonical_train)]
    )
    result = recon.split_change_attribution(old, new, _hash_split, 10)
    assert result["old_key_hypothesis_holds"] is True
    assert result["new_key_hypothesis_holds"] is True
    assert result["rows_where_canonical_differs_from_legacy_key"] == 1


def test_split_attribution_counts_target_key_changes_separately(recon):
    """
    A row whose legacy `target_key` itself changed is a different phenomenon
    from a row that was re-grouped by canonicalization, and merging the two
    would hide an upstream change to `build_target_key`.
    """
    legacy_val, legacy_train, _ = _find_keys()
    old = recon.index_corpus([make_record("s:0", split="val", target_key=legacy_val)])
    new = recon.index_corpus(
        [make_record("s:0", split="val", target_key=legacy_train, canonical=legacy_train)]
    )
    result = recon.split_change_attribution(old, new, _hash_split, 10)
    assert result["rows_with_changed_target_key"] == 1
    assert result["sample_changed_target_key"] == ["s:0"]


# --------------------------------------------------------------------------- #
# The producer log is evidence, not proof
# --------------------------------------------------------------------------- #
SUMMARY_FIXTURE = """some tqdm noise here
=== PARQUET PREPROCESS SUMMARY ===
input:               data/raw/asd
files_seen:          20
rows_seen:           100
records_kept:        60
duplicates_dropped:  10
label_conflicts:     2
kept_by_split:       {'train': 55, 'val': 5}
kept_by_confidence:  {'very_high': 40, 'high': 20}
drop_reasons:
  confidence_filtered: 20
  missing_heavy: 10
committed:           data/processed/x.jsonl.gz
"""


def test_parse_producer_summary_reads_scalars_dicts_and_drop_reasons(recon):
    summary = recon.parse_producer_summary(SUMMARY_FIXTURE)
    assert summary["rows_seen"] == 100
    assert summary["records_kept"] == 60
    assert summary["duplicates_dropped"] == 10
    assert summary["kept_by_split"] == {"train": 55, "val": 5}
    assert summary["drop_reasons"] == {"confidence_filtered": 20, "missing_heavy": 10}


def test_parse_producer_summary_omits_counters_a_run_never_printed(recon):
    """
    The OLD log predates several counters. Defaulting them to 0 would fabricate
    a measurement, so an absent key must stay absent.
    """
    summary = recon.parse_producer_summary(SUMMARY_FIXTURE)
    assert "target_components" not in summary
    assert "identity_row_errors" not in summary


def test_parse_producer_summary_returns_nothing_without_the_header(recon):
    assert recon.parse_producer_summary("no summary in this log at all") == {}


def test_row_accounting_balances_only_when_every_input_row_is_accounted_for(recon):
    summary = recon.parse_producer_summary(SUMMARY_FIXTURE)
    assert recon.check_row_accounting(summary)["balances"] is True

    summary["drop_reasons"]["confidence_filtered"] = 19
    assert recon.check_row_accounting(summary)["balances"] is False


def test_row_accounting_is_not_checkable_without_a_drop_table(recon):
    assert recon.check_row_accounting({"rows_seen": 1})["checkable"] is False


def test_log_provenance_rejects_a_log_that_does_not_describe_the_corpus(recon):
    """
    The OLD run left no stats file, so its log is the only parameter record --
    and an unvalidated log is a guess. A log claiming counts the corpus on disk
    does not have must be refused before any of its numbers are used.
    """
    index = recon.index_corpus(
        [make_record("s:0", split="train"), make_record("s:1", split="val", heavy="WWWW")]
    )
    summary = recon.parse_producer_summary(SUMMARY_FIXTURE)
    assert summary["records_kept"] == 60 and index.rows == 2, "fixture must actually disagree"

    verdict = recon.validate_summary_against_corpus(summary, index)
    assert verdict["rows_match"] is False
    assert verdict["splits_match"] is False


def test_log_provenance_accepts_a_log_that_matches_on_rows_and_splits(recon):
    index = recon.index_corpus(
        [make_record("s:0", split="train"), make_record("s:1", split="val", heavy="WWWW")]
    )
    summary = {"records_kept": 2, "kept_by_split": {"train": 1, "val": 1}}
    verdict = recon.validate_summary_against_corpus(summary, index)
    assert verdict["rows_match"] is True
    assert verdict["splits_match"] is True


def test_log_provenance_rejects_a_matching_row_count_with_a_wrong_split_split(recon):
    """
    Two independent quantities, not one: a log with the right TOTAL but the
    wrong train/val breakdown describes a different run and must still fail.
    """
    index = recon.index_corpus(
        [make_record("s:0", split="train"), make_record("s:1", split="val", heavy="WWWW")]
    )
    summary = {"records_kept": 2, "kept_by_split": {"train": 2}}
    verdict = recon.validate_summary_against_corpus(summary, index)
    assert verdict["rows_match"] is True
    assert verdict["splits_match"] is False


def test_utf16_producer_log_is_decoded(recon, tmp_path: Path):
    """
    The 2026-06-11 log is a PowerShell redirect: UTF-16 LE with a BOM. Reading
    it as UTF-8 throws, and reading it as latin-1 interleaves NUL bytes that
    break every regex. Neither log will ever be re-captured, so the decoder has
    to handle both encodings.
    """
    path = tmp_path / "old.log"
    path.write_bytes(SUMMARY_FIXTURE.encode("utf-16"))
    summary = recon.parse_producer_summary(recon.read_log_text(path))
    assert summary["records_kept"] == 60


def test_invalid_utf8_producer_log_is_still_parsed(recon, tmp_path: Path):
    """The 2026-08-31 log carries raw tqdm bytes that are not valid UTF-8."""
    path = tmp_path / "new.log"
    path.write_bytes(b"\xe2\x96\x88\xff\xfe_not_a_bom_here\n" + SUMMARY_FIXTURE.encode("utf-8"))
    summary = recon.parse_producer_summary(recon.read_log_text(path))
    assert summary["records_kept"] == 60


def test_drop_reason_delta_covers_reasons_present_in_only_one_run(recon):
    """
    The two runs do not share a drop-reason vocabulary -- `missing_heavy` never
    fired in the OLD run. A delta that only walked the intersection would drop
    the single largest explained difference on the floor.
    """
    old = {"drop_reasons": {"light_length_out_of_range": 100}}
    new = {"drop_reasons": {"missing_heavy": 80, "light_length_out_of_range": 1}}
    delta = recon.diff_drop_reasons(old, new)
    assert delta["per_reason"]["missing_heavy"] == {"old": 0, "new": 80, "delta": 80}
    assert delta["per_reason"]["light_length_out_of_range"]["delta"] == -99
    assert delta["total_old"] == 100
    assert delta["total_new"] == 81


# --------------------------------------------------------------------------- #
# Drop-reason replay: one variable, and it must be the right one
# --------------------------------------------------------------------------- #
def test_legacy_clean_aa_sequence_fabricates_a_chain_from_a_pandas_nan(recon):
    """
    The whole explanation for the drop-table shift rests on this: `str(x or "")`
    is not a null guard for a float NaN, because NaN is truthy. The pre-fix
    function turned a missing chain into the two-plus-one residues of "NAN",
    and every length check downstream then judged a fabricated sequence.

    Fixture power: the test asserts the NaN really is truthy, because if some
    future pandas made it falsy the shim would silently agree with the fix and
    this test would pass while proving nothing.
    """
    import math

    nan = float("nan")
    assert bool(nan) is True, "fixture is powerless if NaN stops being truthy"
    assert math.isnan(nan)

    assert recon.legacy_clean_aa_sequence(nan) == "NAN"
    assert len(recon.legacy_clean_aa_sequence(nan)) == 3


def test_legacy_clean_aa_sequence_agrees_with_the_current_one_on_real_sequences(recon):
    """
    A shim that differed anywhere else would make the replay a two-variable
    experiment. It must diverge ONLY on the missing-value path.
    """
    producer = _load_producer(recon)
    for sequence in ("EVQLVESGGG", "evqlv esggg", "EVQL-VES*GGG", "", None):
        assert recon.legacy_clean_aa_sequence(sequence) == producer.clean_aa_sequence(sequence)
    assert recon.legacy_clean_aa_sequence(float("nan")) != producer.clean_aa_sequence(float("nan"))


def _load_producer(recon):
    return recon.load_producer_module()


def test_replay_filter_parameters_are_the_producer_own_defaults(recon):
    """
    The replay's claim is "the OLD run used the DEFAULTS". That is only a
    meaningful claim if the values fed to `keep_record` are the producer's
    actual defaults, so they are read back from the producer's own parser rather
    than trusted as a hand-copied list. A default changed upstream breaks this
    test instead of silently invalidating the replay's conclusion.
    """
    producer = _load_producer(recon)
    saved = sys.argv
    sys.argv = ["prepare_antibody_antigen.py"]
    try:
        producer_defaults = producer.parse_args()
    finally:
        sys.argv = saved

    assumed = recon.default_filter_args()
    for field in ("min_heavy", "max_heavy", "min_light", "max_light",
                  "min_antigen", "max_antigen", "allowed_confidence", "val_percent"):
        assert getattr(assumed, field) == getattr(producer_defaults, field), field


def test_replay_changes_exactly_one_thing_and_reproduces_both_outcomes(recon):
    """
    End-to-end on three hand-built rows, each pinning one arm of the mechanism:

    - a row with a NaN heavy chain: `heavy_length_out_of_range` under the legacy
      cleaner (because "NAN" is 3 residues), `missing_heavy` under the current one;
    - a row with a NaN light chain and an otherwise valid heavy chain: dropped as
      `light_length_out_of_range` under the legacy cleaner, KEPT under the current
      one -- the 121,490-row loosening in miniature;
    - a fully valid row: kept by both, so the replay cannot pass by rejecting
      everything.
    """
    producer = _load_producer(recon)
    nan = float("nan")
    heavy = "E" * 120
    antigen = "A" * 50

    rows = [
        {"heavy_sequence": nan, "light_sequence": nan, "antigen_sequence": antigen,
         "confidence": "high", "metadata": {}},
        {"heavy_sequence": heavy, "light_sequence": nan, "antigen_sequence": antigen,
         "confidence": "high", "metadata": {}},
        {"heavy_sequence": heavy, "light_sequence": "L" * 110, "antigen_sequence": antigen,
         "confidence": "high", "metadata": {}},
    ]
    cross = recon.replay_drop_reasons(rows, producer, recon.default_filter_args())
    assert cross[("heavy_length_out_of_range", "missing_heavy")] == 1
    assert cross[("light_length_out_of_range", "kept")] == 1
    assert cross[("kept", "kept")] == 1
    assert sum(cross.values()) == 3


def test_replay_restores_the_producer_module_it_monkeypatched(recon):
    """
    The replay swaps a module-level name to run its legacy pass. Leaving that
    swap in place would silently corrupt anything else importing the producer
    in the same process -- including the split re-derivation this same run does.
    """
    producer = _load_producer(recon)
    before = producer.clean_aa_sequence
    recon.replay_drop_reasons(
        [{"heavy_sequence": "E" * 120, "light_sequence": None,
          "antigen_sequence": "A" * 50, "confidence": "high", "metadata": {}}],
        producer,
        recon.default_filter_args(),
    )
    assert producer.clean_aa_sequence is before


def test_replay_restores_the_producer_module_even_when_the_legacy_pass_raises(recon):
    """
    The restore must be in a `finally`, or one bad row leaves a 2026-06-10
    function installed on the producer module for the rest of the process --
    including the split re-derivation this same run performs afterwards.

    The failure has to happen while the LEGACY cleaner is installed, or the test
    is powerless: on the happy path the loop body already swaps back before the
    next iteration, so a raise between rows proves nothing. The fixture forces
    the raise inside the legacy pass and then ASSERTS that is where it happened.
    """
    producer = _load_producer(recon)
    before = producer.clean_aa_sequence

    class ExplodingArgs:
        """Filter args that blow up on first use, recording the live cleaner."""

        def __init__(self) -> None:
            self.allowed_confidence = "high,very_high"
            self.cleaner_at_failure = None

        @property
        def min_heavy(self):
            self.cleaner_at_failure = producer.clean_aa_sequence
            raise RuntimeError("filter parameter unavailable")

    args = ExplodingArgs()
    row = {"heavy_sequence": "E" * 120, "light_sequence": None,
           "antigen_sequence": "A" * 50, "confidence": "high", "metadata": {}}
    with pytest.raises(RuntimeError):
        recon.replay_drop_reasons([row], producer, args)

    # Fixture power: the raise really did land inside the legacy pass.
    assert args.cleaner_at_failure is recon.legacy_clean_aa_sequence
    assert producer.clean_aa_sequence is before


def test_summarize_replay_confirms_a_match_against_both_logs(recon):
    from collections import Counter

    cross = Counter({
        ("heavy_length_out_of_range", "missing_heavy"): 5,
        ("light_length_out_of_range", "kept"): 3,
        ("kept", "kept"): 10,
    })
    summary = recon.summarize_replay(
        cross,
        {"drop_reasons": {"heavy_length_out_of_range": 5, "light_length_out_of_range": 3}},
        {"drop_reasons": {"missing_heavy": 5}},
    )
    assert summary["rows_replayed"] == 18
    assert summary["rows_whose_outcome_changed"] == 8
    assert summary["legacy_matches_old_log"] is True
    assert summary["current_matches_new_log"] is True


def test_summarize_replay_refuses_a_log_it_does_not_reproduce(recon):
    """
    If the replay's legacy marginal does not equal the OLD log's drop table, the
    OLD run did not use the parameters the replay assumed and nothing has been
    explained. A near-miss must fail exactly like a wild miss.
    """
    from collections import Counter

    cross = Counter({("heavy_length_out_of_range", "missing_heavy"): 5, ("kept", "kept"): 10})
    summary = recon.summarize_replay(
        cross,
        {"drop_reasons": {"heavy_length_out_of_range": 4}},  # off by one
        {"drop_reasons": {"missing_heavy": 5}},
    )
    assert summary["legacy_matches_old_log"] is False
    assert summary["current_matches_new_log"] is True


def test_summarize_replay_does_not_claim_a_match_against_a_missing_log(recon):
    """An absent drop table is not a match; it is an unavailable comparison."""
    from collections import Counter

    summary = recon.summarize_replay(Counter({("kept", "kept"): 1}), {}, {})
    assert summary["legacy_matches_old_log"] is False
    assert summary["current_matches_new_log"] is False


def test_replay_mismatch_blocks_the_verdict(recon):
    """The replay must be able to veto, not merely decorate the report."""
    old_rows, new_rows, summary_old, summary_new = _clean_pair(recon)
    report = recon.build_report(
        recon.index_corpus(old_rows),
        recon.index_corpus(new_rows),
        summary_old,
        summary_new,
        _hash_split,
        10,
        replay={"rows_replayed": 1, "rows_whose_outcome_changed": 0, "cross_tab": {},
                "legacy_clean_aa_outcomes": {}, "current_clean_aa_outcomes": {},
                "legacy_matches_old_log": False, "current_matches_new_log": True},
    )
    assert report["verdict"] == "BLOCKED"
    assert any("PRE-FIX clean_aa_sequence" in problem for problem in report["unexplained"])


def test_replay_match_leaves_the_verdict_clean(recon):
    """The positive control for the veto above."""
    old_rows, new_rows, summary_old, summary_new = _clean_pair(recon)
    report = recon.build_report(
        recon.index_corpus(old_rows),
        recon.index_corpus(new_rows),
        summary_old,
        summary_new,
        _hash_split,
        10,
        replay={"rows_replayed": 1, "rows_whose_outcome_changed": 0, "cross_tab": {},
                "legacy_clean_aa_outcomes": {}, "current_clean_aa_outcomes": {},
                "legacy_matches_old_log": True, "current_matches_new_log": True},
    )
    assert report["verdict"] == "CLEAN"
    assert report["drop_reason_replay"]["legacy_matches_old_log"] is True


# --------------------------------------------------------------------------- #
# Index bookkeeping
# --------------------------------------------------------------------------- #
def test_index_counts_rows_splits_and_canonical_coverage(recon):
    index = recon.index_corpus(
        [
            make_record("s:0", split="train", canonical="uniprot:p1"),
            make_record("s:1", split="val", heavy="WWWW", canonical="uniprot:p1"),
            make_record("s:2", split="train", heavy="YYYY"),
        ]
    )
    assert index.rows == 3
    assert dict(index.by_split) == {"train": 2, "val": 1}
    assert index.rows_with_canonical == 2
    assert index.field_present.get("canonical_target_id") is True


def test_index_flags_a_corpus_that_violates_the_producer_dedupe_invariant(recon):
    """
    The natural key is only a key if the producer's dedupe held. Two rows with
    the same triple must be counted, not silently collapsed -- otherwise a
    corpus with a broken dedupe would reconcile "cleanly" against one without.
    """
    index = recon.index_corpus([make_record("s:0"), make_record("s:1")])
    assert index.rows == 2
    assert index.duplicate_biology == 1
    assert len(index.bio_to_rid) == 1


def test_compare_key_sets_is_a_three_way_partition(recon):
    result = recon.compare_key_sets(["a", "b"], ["b", "c"])
    assert result == {
        "only_old": 1,
        "only_new": 1,
        "in_both": 1,
        "old_total": 2,
        "new_total": 2,
    }


# --------------------------------------------------------------------------- #
# End-to-end: the verdict must flip on a real problem
# --------------------------------------------------------------------------- #
def _write_corpus(path: Path, records: list[dict]) -> Path:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return path


def _clean_pair(recon):
    """
    A corpus pair with NO unexplained differences: identical biology, identical
    labels, splits that follow the key each run is supposed to hash.
    """
    legacy_val, _, canonical_train = _find_keys()
    old = [make_record("s:0", split="val", target_key=legacy_val)]
    new = [make_record("s:0", split="train", target_key=legacy_val, canonical=canonical_train)]
    summary_old = {
        "rows_seen": 3,
        "records_kept": 1,
        "duplicates_dropped": 1,
        "drop_reasons": {"missing_heavy": 1},
        "kept_by_split": {"val": 1},
    }
    summary_new = dict(summary_old, kept_by_split={"train": 1})
    return old, new, summary_old, summary_new


def test_end_to_end_verdict_is_clean_when_every_difference_is_attributable(recon):
    old_rows, new_rows, summary_old, summary_new = _clean_pair(recon)
    report = recon.build_report(
        recon.index_corpus(old_rows),
        recon.index_corpus(new_rows),
        summary_old,
        summary_new,
        _hash_split,
        10,
    )
    assert report["unexplained"] == []
    assert report["verdict"] == "CLEAN"
    assert report["split_churn"]["rows_with_changed_split"] == 1
    assert report["split_churn"]["attributable_to_canonicalization"] == 1


def test_end_to_end_verdict_blocks_on_a_changed_label(recon):
    """
    Same corpus pair as the clean case, plus one label flip. Nothing else moves,
    so a verdict that stays CLEAN means the label comparison is inert.
    """
    old_rows, new_rows, summary_old, summary_new = _clean_pair(recon)
    new_rows = [dict(new_rows[0], binder_label=0, is_strong_binder=False)]
    report = recon.build_report(
        recon.index_corpus(old_rows),
        recon.index_corpus(new_rows),
        summary_old,
        summary_new,
        _hash_split,
        10,
    )
    assert report["verdict"] == "BLOCKED"
    assert any("changed" in problem and "binder_label" in problem for problem in report["unexplained"])


def test_end_to_end_verdict_blocks_when_biology_disappears(recon):
    old_rows, new_rows, summary_old, summary_new = _clean_pair(recon)
    old_rows = old_rows + [make_record("s:1", heavy="WWWW", target_key="pdb:gone")]
    summary_old = dict(summary_old, records_kept=2, kept_by_split={"val": 1, "train": 1})
    report = recon.build_report(
        recon.index_corpus(old_rows),
        recon.index_corpus(new_rows),
        summary_old,
        summary_new,
        _hash_split,
        10,
    )
    assert report["verdict"] == "BLOCKED"
    assert report["membership"]["detail"]["only_old_biology_gone"] == 1
    assert any("absent from NEW" in problem for problem in report["unexplained"])


def test_end_to_end_verdict_blocks_when_a_new_row_lacks_a_canonical_id(recon):
    """
    The entire point of the regeneration is that `canonical_target_id` is
    populated. A NEW row without one is a regeneration that did not do its job.
    """
    old_rows, new_rows, summary_old, summary_new = _clean_pair(recon)
    new_rows = [{k: v for k, v in new_rows[0].items() if k != "canonical_target_id"}]
    report = recon.build_report(
        recon.index_corpus(old_rows),
        recon.index_corpus(new_rows),
        summary_old,
        summary_new,
        _hash_split,
        10,
    )
    assert report["verdict"] == "BLOCKED"
    assert any("carry no" in problem and "canonical_target_id" in problem
               for problem in report["unexplained"])


def test_end_to_end_verdict_blocks_when_membership_contradicts_the_logs(recon):
    """
    The corpora gained a row that the producer's own kept-count delta does not
    account for. That is an arithmetic contradiction between the artifact and
    its producer, and it must block.
    """
    old_rows, new_rows, summary_old, summary_new = _clean_pair(recon)
    new_rows = new_rows + [make_record("s:9", heavy="WWWW", target_key="pdb:x", canonical="pdb:x")]
    summary_new = dict(summary_new, records_kept=2, kept_by_split={"train": 2})
    # The logs now agree with the corpora on counts, but claim a kept delta of
    # +1 while... they do agree. Break only the log, so the contradiction is
    # unambiguous.
    summary_new = dict(summary_new, records_kept=5)
    report = recon.build_report(
        recon.index_corpus(old_rows),
        recon.index_corpus(new_rows),
        summary_old,
        summary_new,
        _hash_split,
        10,
    )
    assert report["verdict"] == "BLOCKED"
    assert any("kept-count delta" in problem for problem in report["unexplained"])


def test_cli_writes_a_json_artifact_and_never_touches_the_corpora(recon, tmp_path: Path):
    """
    Read-only is a hard requirement -- these corpora are expensive to
    regenerate. Pin it by hashing both inputs before and after a full run.
    """
    import hashlib

    legacy_val, _, canonical_train = _find_keys()
    old_path = _write_corpus(tmp_path / "old.jsonl.gz",
                             [make_record("s:0", split="val", target_key=legacy_val)])
    new_path = _write_corpus(
        tmp_path / "new.jsonl.gz",
        [make_record("s:0", split="train", target_key=legacy_val, canonical=canonical_train)],
    )
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (old_path, new_path)]

    out = tmp_path / "report.json"
    argv = [
        "reconcile_asd_regeneration.py",
        "--old", str(old_path),
        "--new", str(new_path),
        "--old-log", str(tmp_path / "missing-old.log"),
        "--new-log", str(tmp_path / "missing-new.log"),
        "--output-json", str(out),
    ]
    saved = sys.argv
    sys.argv = argv
    try:
        assert recon.main() == 0
    finally:
        sys.argv = saved

    after = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (old_path, new_path)]
    assert before == after, "the reconciliation must be read-only with respect to the corpora"

    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["schema_version"] == recon.SCHEMA_VERSION
    assert report["corpora"]["old"]["sha256"] == before[0]
    assert report["corpora"]["new"]["sha256"] == before[1]
    assert report["provenance"]["producer_sha256"]
    assert report["provenance"]["source_commit"]
