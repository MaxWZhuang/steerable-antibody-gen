"""
The quarantine mechanism, tested before any ancestor-contact outcome exists.

Every test here is about the MECHANISM. No result is asserted, because none has
been measured; that is the point of committing this separately.

Three defects shipped in earlier drafts of this path, and each has a test named
after it:

- **Pair-incompatible filter.** Queries were indexed under the budget implied by
  their OWN length, but the threshold normalizes by the LONGER member, so a
  longer ancestor grants more edits. A length-9 query and a length-10 ancestor at
  distance 2 are exactly 80% similar and shared no key -- the query was reported
  clean.
- **Light-chain ancestors.** ~44% of Stage-1 rows are light chains with `cdr3_aa`
  populated. Reading those as ancestor HCDR3 quarantines on the strength of
  LIGHT CDR3 similarity.
- **Powerless fixtures.** A brute-force comparison over random strings from a
  large alphabet contains almost no true neighbours, so it passes against a
  broken filter. Every equivalence test below therefore ASSERTS its fixture
  contains a declared number of true neighbours; a fixture that stops being
  adversarial fails loudly instead of silently proving nothing.
"""
from __future__ import annotations

import itertools
import random

import pytest

from smallAntibodyGen import ancestor_quarantine as aq


# --------------------------------------------------------------------------- #
# The 9-vs-10 counterexample
# --------------------------------------------------------------------------- #
def test_a_longer_ancestor_raises_the_edit_budget():
    """The asymmetry the old filter ignored: the threshold uses the LONGER member."""
    query, ancestor = "ABCDEFGHI", "ABXCDEYGHI"
    assert aq.bounded_levenshtein(query, ancestor, 5) == 2
    assert aq.max_edits(len(query)) == 1, "the query's own length allows only 1"
    assert aq.max_edits(len(ancestor)) == 2, "...but the pair is judged at length 10"
    assert aq.similarity(query, ancestor) == pytest.approx(0.8)


def test_the_nine_versus_ten_pair_is_found_by_the_filter():
    """
    The exact case that was reported clean. Indexing under `index_budget`
    rather than `max_edits(len(query))` is what fixes it.
    """
    query, ancestor = "ABCDEFGHI", "ABXCDEYGHI"
    assert aq.index_budget(len(query)) >= aq.max_edits(len(ancestor))
    index = aq.QueryIndex([query])
    assert index.candidates(ancestor), "filter produced no candidate"
    assert index.neighbors(ancestor) == aq.brute_force_neighbors([query], ancestor)


def test_the_compatible_length_ceiling_covers_every_longer_partner():
    """
    No ancestor longer than the ceiling can be within the band, and every one at
    or below it must be indexable.
    """
    for length in range(4, 30):
        ceiling = aq.compatible_length_ceiling(length)
        beyond = ceiling + 1
        assert beyond - length > aq.max_edits(beyond), (
            f"length {beyond} claimed incompatible with {length} but is reachable"
        )


# --------------------------------------------------------------------------- #
# Filter equals brute force, on fixtures with declared power
# --------------------------------------------------------------------------- #
def _dense_pool(seed: int, n: int, lo: int, hi: int, alphabet: str = "ABC") -> list[str]:
    """Small alphabet, narrow length range: a pool that actually contains neighbours."""
    rng = random.Random(seed)
    return sorted({
        "".join(rng.choice(alphabet) for _ in range(rng.randint(lo, hi)))
        for _ in range(n)
    })


@pytest.mark.parametrize("seed", range(25))
def test_filter_equals_brute_force_on_dense_mixed_length_fixtures(seed):
    """
    Mixed lengths, so indels and the longer-partner asymmetry are exercised, and
    the fixture is asserted to contain true neighbours so the comparison has power.
    """
    queries = _dense_pool(seed, 40, 5, 9)
    ancestors = _dense_pool(seed + 1000, 120, 5, 11)

    true_pairs = sum(len(aq.brute_force_neighbors(queries, a)) for a in ancestors)
    assert true_pairs >= 5, (
        f"fixture contains only {true_pairs} true neighbours and cannot detect a "
        "broken filter -- this assertion exists because an earlier 'verified "
        "complete, 0/40 seeds' check was near-vacuous"
    )

    index = aq.QueryIndex(queries)
    for ancestor in ancestors:
        assert sorted(index.neighbors(ancestor)) == sorted(
            aq.brute_force_neighbors(queries, ancestor)
        )


def test_filter_equals_brute_force_over_an_exhaustive_small_space():
    """Every binary string of length 5-7 against every other: complete coverage."""
    universe = ["".join(p) for L in (5, 6, 7) for p in itertools.product("AB", repeat=L)]
    queries = universe[::7]
    index = aq.QueryIndex(queries)
    true_pairs = 0
    for ancestor in universe:
        expected = aq.brute_force_neighbors(queries, ancestor)
        true_pairs += len(expected)
        assert sorted(index.neighbors(ancestor)) == sorted(expected)
    assert true_pairs > 100, "exhaustive space must be rich in neighbours"


def test_short_queries_fall_back_to_a_pair_compatible_length_gate():
    """
    The fallback gate must use max(len(query), len(ancestor)), not the ancestor
    alone -- the same asymmetry as the indexing bug.
    """
    assert aq.length_compatible("ABCDEFGHI", "ABXCDEYGHI") is True
    assert aq.length_compatible("ABC", "ABCDEFGHIJKL") is False

    # THE discriminating case: the QUERY is the longer member, so the budget must
    # come from it. Deriving the gate from the ancestor alone gives max_edits(8)=1
    # and rejects a pair that is genuinely within the band. Every other assertion
    # here has the query shorter, where both formulations agree -- so without this
    # line the ancestor-only mutation survives the whole suite.
    assert aq.max_edits(10) == 2 and aq.max_edits(8) == 1
    assert aq.length_compatible("ABCDEFGHIJ", "ABCDEFGH") is True

    # Length 1 cannot carry even a 2-mer key, so it takes the direct path.
    single = ["A", "B"]
    index = aq.QueryIndex(single)
    assert index.direct == [0, 1], "length-1 queries must be compared directly"
    for ancestor in ("A", "AB", "ZZZZ"):
        assert sorted(index.neighbors(ancestor)) == sorted(
            aq.brute_force_neighbors(single, ancestor))

    # Length 2 IS keyable (budget 0 -> one segment -> a 2-mer key), and must give
    # the same answer through the indexed path.
    tiny = ["AB", "AC"]
    keyed = aq.QueryIndex(tiny)
    assert keyed.direct == [], "length-2 queries are keyable at k=2"
    for ancestor in ("AB", "AC", "ABC", "ZZZZ"):
        assert sorted(keyed.neighbors(ancestor)) == sorted(
            aq.brute_force_neighbors(tiny, ancestor))


# --------------------------------------------------------------------------- #
# Stage-1 light chains are not ancestors of an HCDR3
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("chain_group,expected", [
    ("heavy", True), ("light", False), ("paired", False), (None, False), ("", False),
])
def test_only_heavy_stage1_records_supply_an_ancestor_hcdr3(chain_group, expected):
    assert aq.stage1_contributes_hcdr3({"chain_group": chain_group}) is expected


def test_light_chain_cdr3_cannot_quarantine_a_heavy_hcdr3():
    """
    A light record whose CDR3 is identical to an ASD HCDR3 must be ignored.
    ~44% of Stage-1 rows are light, so this is not a corner case.
    """
    rows = [
        {"chain_group": "light", "cdr3_aa": "CARDGGYYYFDYW"},
        {"chain_group": "heavy", "cdr3_aa": "QQQQQQQQQQQQQ"},
    ]
    usable = [r["cdr3_aa"] for r in rows if aq.stage1_contributes_hcdr3(r)]
    assert usable == ["QQQQQQQQQQQQQ"]


# --------------------------------------------------------------------------- #
# One contact quarantines the whole component
# --------------------------------------------------------------------------- #
def test_a_single_hcdr3_contact_quarantines_its_entire_component():
    membership = {("t", "c1"): {"hcdr3": {"AAA", "AAB", "AAC"}, "heavy": set()},
                  ("t", "c2"): {"hcdr3": {"ZZZ"}, "heavy": set()}}
    contacts = {h: aq.Contact(key=h) for h in ("AAA", "AAB", "AAC", "ZZZ")}
    contacts["AAB"].note("stage1", "train", 0.92)

    result = aq.quarantine_components(membership, contacts)
    assert result[("t", "c1")]["quarantined"] is True
    assert result[("t", "c1")]["stages"] == ["stage1_train"]
    assert result[("t", "c1")]["reason_hcdr3_ge_80"] is True
    assert result[("t", "c2")]["quarantined"] is False


def test_one_contact_fans_out_to_every_target_local_component():
    """
    Components are TARGET-LOCAL. In the real corpus 4,868 of 10,020 labelled
    HCDR3 occur under more than one target (one under 842), so a flat
    HCDR3 -> component dict cannot represent those memberships, and one contact
    must quarantine the component in EVERY target containing that sequence.
    """
    membership = {("t1", "c1"): {"hcdr3": {"AAA"}, "heavy": set()},
                  ("t2", "c9"): {"hcdr3": {"AAA"}, "heavy": set()},
                  ("t3", "c4"): {"hcdr3": {"ZZZ"}, "heavy": set()}}
    contacts = {h: aq.Contact(key=h) for h in ("AAA", "ZZZ")}
    contacts["AAA"].note("stage2", "train", 0.88)

    result = aq.quarantine_components(membership, contacts)
    assert result[("t1", "c1")]["quarantined"] is True
    assert result[("t2", "c9")]["quarantined"] is True
    assert result[("t3", "c4")]["quarantined"] is False


def test_exact_heavy_contact_quarantines_and_keeps_its_attribution():
    """
    Heavy identity gates on its own AND carries stage/split. An earlier version
    quarantined on heavy contact but reported `stages: []`, losing any record of
    which ancestor stage caused it.
    """
    membership = {("t", "c1"): {"hcdr3": {"AAA"}, "heavy": set()},
                  ("t", "c2"): {"hcdr3": {"ZZZ"}, "heavy": {"HEAVY1"}}}
    hcdr3 = {h: aq.Contact(key=h) for h in ("AAA", "ZZZ")}
    heavy = {"HEAVY1": aq.Contact(key="HEAVY1")}
    heavy["HEAVY1"].note_heavy("stage2", "selection")

    result = aq.quarantine_components(membership, hcdr3, heavy)
    assert result[("t", "c1")]["quarantined"] is False
    assert result[("t", "c2")]["quarantined"] is True
    assert result[("t", "c2")]["reason_exact_heavy"] is True
    assert result[("t", "c2")]["stages"] == ["stage2_selection"]


def test_an_unmeasured_sequence_quarantines_rather_than_passes():
    """
    Silence is not evidence of disjointness. A component containing a sequence
    the scan never covered cannot enter a final cohort.
    """
    membership = {("t", "c1"): {"hcdr3": {"AAA", "BBB"}, "heavy": set()}}
    contacts = {"AAA": aq.Contact(key="AAA")}  # BBB never measured
    result = aq.quarantine_components(membership, contacts)
    assert result[("t", "c1")]["unmeasured"] is True
    assert result[("t", "c1")]["quarantined"] is True


def test_selection_contact_is_recorded_separately_from_training_contact():
    """
    Validation rows selected best.pt, so they gate -- but the narrower
    training-only number must stay recoverable.
    """
    contact = aq.Contact(key="AAA")
    contact.note("stage2", "selection", 0.85)
    assert contact.touches_stage2_selection is True
    assert contact.touches_stage2_train is False
    assert contact.contacted is True


def test_clean_records_are_emitted_explicitly():
    """A clean query serializes as measured-and-clean, not as an absence."""
    payload = aq.Contact(key="AAA").to_json()
    assert payload["reason_exact_heavy"] is False
    assert payload["reason_hcdr3_ge_80"] is False
    assert payload["max_similarity"] == 0.0


# --------------------------------------------------------------------------- #
# Shard / resume validation
# --------------------------------------------------------------------------- #
def _header(**kw):
    base = dict(manifest_sha256="m", scanner_sha256="s",
                corpus_sha256={"stage1": "a", "stage2": "b"},
                shard_index=0, shard_count=2)
    base.update(kw)
    return aq.ShardHeader(**base)


def test_join_accepts_a_complete_consistent_shard_set():
    aq.validate_shards([_header(shard_index=0), _header(shard_index=1)])


@pytest.mark.parametrize("field,value", [
    ("manifest_sha256", "different"),
    ("scanner_sha256", "different"),
    ("corpus_sha256", {"stage1": "a", "stage2": "CHANGED"}),
])
def test_join_rejects_mixed_provenance(field, value):
    """Results from different universes cannot be joined."""
    with pytest.raises(aq.ShardMismatch, match="different contracts"):
        aq.validate_shards([_header(shard_index=0), _header(shard_index=1, **{field: value})])


def test_join_rejects_a_missing_shard():
    """A partial scan understates contact and overstates the retained cohort."""
    with pytest.raises(aq.ShardMismatch, match="missing shard"):
        aq.validate_shards([_header(shard_index=0)])


def test_join_rejects_duplicate_shards():
    with pytest.raises(aq.ShardMismatch, match="duplicate shard"):
        aq.validate_shards([_header(shard_index=0), _header(shard_index=0)])


def test_join_rejects_an_empty_shard_set():
    with pytest.raises(aq.ShardMismatch, match="no shards"):
        aq.validate_shards([])


def test_merging_shards_ors_every_flag():
    """A query contacted in one shard stays contacted after the merge."""
    a = aq.Contact(key="AAA"); a.note("stage1", "train", 0.85)
    b = aq.Contact(key="AAA"); b.note("stage2", "selection", 0.95)
    merged = aq.merge_contacts([[a], [b]])["AAA"]
    assert merged.touches_stage1_train and merged.touches_stage2_selection
    assert merged.max_similarity == pytest.approx(0.95)


# --------------------------------------------------------------------------- #
# Candidate universe
# --------------------------------------------------------------------------- #
def _target(**kw):
    base = {"measurement_method": "exact", "distinct_hcdr3": 100,
            "labels": {"binder_positive": 5, "binder_negative": 5}}
    base.update(kw)
    return base


def test_a_target_with_binders_nonbinders_and_a_map_is_included():
    included, failed = aq.evaluate_candidate(_target())
    assert included and failed == []


@pytest.mark.parametrize("override,expected_reason", [
    ({"measurement_method": "sampled_percolation"}, "exact_component_map_available"),
    ({"labels": {"binder_positive": 0, "binder_negative": 5}}, "has_binary_binder"),
    ({"labels": {"binder_positive": 5, "binder_negative": 0}}, "has_binary_nonbinder"),
    ({"distinct_hcdr3": 0}, "has_usable_hcdr3"),
])
def test_exclusions_are_reported_with_their_reason(override, expected_reason):
    included, failed = aq.evaluate_candidate(_target(**override))
    assert not included
    assert expected_reason in failed


def test_small_panels_are_not_excluded_by_a_component_floor():
    """
    No m_t >= 50 floor. Precision analysis has not justified one, so small panels
    are included and Commit C decides whether their intervals are useful.
    """
    included, _ = aq.evaluate_candidate(_target(distinct_hcdr3=3))
    assert included is True


def test_manifest_hash_is_content_addressed_and_order_independent():
    a = {"targets": {"x": 1, "y": 2}, "band": 80}
    b = {"band": 80, "targets": {"y": 2, "x": 1}}
    assert aq.manifest_hash(a) == aq.manifest_hash(b)
    assert aq.manifest_hash(a) != aq.manifest_hash({"band": 90, "targets": {"x": 1, "y": 2}})


@pytest.mark.parametrize("field", ["sizing_sha256", "component_map_sha256",
                                   "mechanism_sha256"])
def test_manifest_hash_binds_every_provenance_input(field):
    """
    An earlier version hashed the body and attached provenance AFTERWARDS, so the
    sizing, component-map and mechanism hashes could all change while the
    "frozen" manifest hash stayed identical.
    """
    base = {"included": {"t": 1},
            "provenance": {"sizing_sha256": "a", "component_map_sha256": "b",
                           "mechanism_sha256": "c"}}
    other = {"included": {"t": 1},
             "provenance": dict(base["provenance"], **{field: "CHANGED"})}
    assert aq.manifest_hash(base) != aq.manifest_hash(other)


def test_manifest_hash_ignores_only_its_own_field():
    body = {"included": {"t": 1}, "provenance": {"sizing_sha256": "a"}}
    assert aq.manifest_hash(body) == aq.manifest_hash(
        dict(body, manifest_sha256="whatever"))


def test_verify_manifest_round_trips_and_detects_tampering():
    body = {"included": {"t": 1}, "provenance": {"sizing_sha256": "a"}}
    body["manifest_sha256"] = aq.manifest_hash(body)
    aq.verify_manifest(body)
    body["included"]["t"] = 2
    with pytest.raises(aq.ManifestDamaged, match="hash mismatch"):
        aq.verify_manifest(body)


def test_verify_manifest_refuses_a_manifest_without_provenance():
    with pytest.raises(aq.ManifestDamaged, match="no provenance"):
        aq.verify_manifest({"included": {}, "manifest_sha256": "x"})


# --------------------------------------------------------------------------- #
# A damaged component map is an ERROR, never a scientific exclusion
# --------------------------------------------------------------------------- #
def _sizing_target():
    return {"distinct_hcdr3": 3, "bands": {"80": {"m_t": 2}}}


def test_a_consistent_component_map_reconciles():
    aq.reconcile_component_map("t", _sizing_target(), {"a": "a", "b": "a", "c": "c"})


def test_a_missing_component_map_refuses_rather_than_excluding():
    """
    An earlier version recorded this as the `exact_component_map_available`
    criterion failing, so a truncated map would silently SHRINK the frozen
    universe while the run reported success.
    """
    with pytest.raises(aq.ManifestDamaged, match="no component map"):
        aq.reconcile_component_map("t", _sizing_target(), None)


def test_a_truncated_component_map_refuses():
    with pytest.raises(aq.ManifestDamaged, match="HCDR3 against"):
        aq.reconcile_component_map("t", _sizing_target(), {"a": "a", "b": "a"})


def test_a_component_count_disagreeing_with_m_t_refuses():
    with pytest.raises(aq.ManifestDamaged, match="components against"):
        aq.reconcile_component_map("t", _sizing_target(),
                                   {"a": "a", "b": "b", "c": "c"})


def test_a_missing_band_refuses():
    with pytest.raises(aq.ManifestDamaged, match="no band"):
        aq.reconcile_component_map("t", {"distinct_hcdr3": 3, "bands": {}},
                                   {"a": "a", "b": "a", "c": "c"})


def test_component_map_hash_is_taken_on_decompressed_content():
    """
    gzip embeds an mtime, so hashing compressed bytes makes an identical logical
    map hash differently when regenerated.
    """
    rows = ['{"a":1}', '{"b":2}']
    assert aq.canonical_jsonl_sha256(rows) == aq.canonical_jsonl_sha256(list(rows))
    assert aq.canonical_jsonl_sha256(rows) != aq.canonical_jsonl_sha256(rows[::-1])
    assert aq.canonical_jsonl_sha256(rows) == aq.canonical_jsonl_sha256(
        [r + chr(10) for r in rows]), "trailing newlines must not change the hash"


# --------------------------------------------------------------------------- #
# Shard completeness, not just shard presence
# --------------------------------------------------------------------------- #
def test_join_rejects_a_present_but_incomplete_shard():
    """
    THE false-clean path. Both headers are valid, but shard 0 never scanned AAA;
    shard 1's clean record for AAA then survives the merge and its component is
    retained as clean, having never been compared against shard 0's ancestors.
    """
    with pytest.raises(aq.ShardMismatch, match="missing 1 candidate"):
        aq.validate_shards([_header(shard_index=0), _header(shard_index=1)],
                           shard_queries=[["BBB"], ["AAA", "BBB"]],
                           expected_queries=["AAA", "BBB"])


def test_join_rejects_records_outside_the_frozen_manifest():
    with pytest.raises(aq.ShardMismatch, match="outside the frozen manifest"):
        aq.validate_shards([_header(shard_index=0), _header(shard_index=1)],
                           shard_queries=[["AAA", "BBB", "EXTRA"], ["AAA", "BBB"]],
                           expected_queries=["AAA", "BBB"])


def test_join_rejects_a_shard_repeating_a_query():
    with pytest.raises(aq.ShardMismatch, match="repeats"):
        aq.validate_shards([_header(shard_index=0), _header(shard_index=1)],
                           shard_queries=[["AAA", "AAA", "BBB"], ["AAA", "BBB"]],
                           expected_queries=["AAA", "BBB"])


def test_join_rejects_an_out_of_range_shard_index():
    with pytest.raises(aq.ShardMismatch, match="outside"):
        aq.validate_shards([_header(shard_index=0), _header(shard_index=5)])


def test_join_accepts_complete_shards_covering_every_query():
    aq.validate_shards([_header(shard_index=0), _header(shard_index=1)],
                       shard_queries=[["AAA", "BBB"], ["AAA", "BBB"]],
                       expected_queries=["AAA", "BBB"])
