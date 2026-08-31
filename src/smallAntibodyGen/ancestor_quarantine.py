"""
Mechanism for quarantining ASD components that touch a Stage-1/2 ancestor (B1).

This module contains NO results. It is committed before any ancestor-contact
outcome exists, so that the measurement commit that follows can say: the universe
and the mechanism were frozen before the numbers were known.

The predeclared rule, applied per ASD component -- one contacting sequence
quarantines the whole component, because the component is the indivisible unit:

    ancestor_hcdr3 = stage1 train+selection HEAVY-chain CDR3
                   + stage2 train+selection heavy CDR3
    ancestor_heavy = stage1 train+selection heavy chains
                   + stage2 train+selection heavy chains

    quarantined(component) = any(
        exact_heavy_match(seq) or max_hcdr3_similarity(seq) >= 0.80
        for seq in component
    )

Validation rows are in the union deliberately: they received no gradients, but
they selected `best.pt`, so they participated in model selection and are not
untouched final evidence. Train and selection contact are recorded SEPARATELY so
the narrower optimizer-training-overlap number stays available.

**Fuzzy full-heavy similarity is explicitly DEFERRED.** It does not gate
eligibility here, and it is not computed here either -- claiming it in prose
while computing nothing was a defect in an earlier draft. Antibody frameworks
share substantial germline similarity, so gating on 80-90% whole-heavy identity
would silently redefine the task as generalization to unseen germline
frameworks. The framework controls (antibody-only and framework-only baselines,
matched antigen swaps, label-switching subsets) belong to the final evaluator.

The resulting holdout is therefore **HCDR3-neighbourhood-disjoint**, never
"unseen antibody families".
"""
from __future__ import annotations

import collections
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Sequence

BAND = 80
#: Longest k-mer used as a filter key. Shorter keys are used automatically when a
#: query's segments are too short to carry a 4-mer.
MAX_KMER = 4
#: Below this segment length no key is selective enough to be worth indexing, and
#: the query falls back to a direct length-gated comparison.
MIN_KMER = 2


def max_edits(length: int, band_pct: int = BAND) -> int:
    """
    Edits allowed at `band_pct` for a string of `length`, in integer arithmetic.

    Never `floor((1 - band/100) * length)`: `1.0 - 0.80` is 0.19999999999999996,
    which allows one edit too few at lengths 5, 10, 15, 20, ...
    """
    return ((100 - band_pct) * length) // 100


def accepts(distance: int, longest: int, band_pct: int = BAND) -> bool:
    """`distance` is within the band for a pair whose longer member is `longest`."""
    return 100 * distance <= (100 - band_pct) * longest


def bounded_levenshtein(a: str, b: str, limit: int) -> int:
    """Edit distance, abandoned once it provably exceeds `limit` (returns limit+1)."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    if a == b:
        return 0
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        lo, hi = max(1, i - limit), min(len(b), i + limit)
        if lo > 1:
            current.extend([limit + 1] * (lo - 1))
        for j in range(lo, hi + 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                              previous[j - 1] + (ca != b[j - 1])))
        current.extend([limit + 1] * (len(b) - hi))
        if min(current) > limit:
            return limit + 1
        previous = current
    return previous[len(b)]


def similarity(a: str, b: str, band_pct: int = BAND) -> float | None:
    """Normalized similarity if the pair is within the band, else None."""
    longest = max(len(a), len(b))
    if longest == 0:
        return 1.0
    limit = max_edits(longest, band_pct)
    distance = bounded_levenshtein(a, b, limit)
    if not accepts(distance, longest, band_pct):
        return None
    return 1.0 - distance / longest


# --------------------------------------------------------------------------- #
# Pair-compatible segment filter
# --------------------------------------------------------------------------- #
def compatible_length_ceiling(query_length: int, band_pct: int = BAND) -> int:
    """
    Longest ancestor that can still be within the band of this query.

    An ancestor LONGER than the query raises the edit budget, because the
    threshold normalizes by the longer member. The bound follows from requiring
    `La - Lq <= max_edits(La)`, i.e. `La * band <= Lq * 100`.
    """
    return (query_length * 100) // band_pct


def index_budget(query_length: int, band_pct: int = BAND) -> int:
    """
    Edit allowance a query must be indexed under: the MAXIMUM any compatible
    ancestor could grant, not the one its own length implies.

    Indexing a length-9 query under `max_edits(9) = 1` gives two segments, but a
    length-10 ancestor at distance 2 is genuinely within the band and can damage
    both -- so the pair shares no key and the query is falsely reported clean.
    That is the 9-vs-10 counterexample pinned in the tests.
    """
    return max_edits(compatible_length_ceiling(query_length, band_pct), band_pct)


def segments(s: str, parts: int) -> list[str]:
    """`parts` disjoint contiguous segments covering `s`."""
    if parts <= 1:
        return [s]
    width, out, start = len(s) // parts, [], 0
    for i in range(parts):
        end = len(s) if i == parts - 1 else start + width
        out.append(s[start:end])
        start = end
    return out


def query_keys(s: str, band_pct: int = BAND) -> tuple[list[tuple[int, str]], bool]:
    """
    Filter keys for one query, and whether the filter is complete for it.

    Returns `(keys, guaranteed)`. Each key is `(k, kmer)`; the k varies per query
    because a query indexed under a larger budget has shorter segments. When even
    a 2-mer will not fit, no key is selective and `guaranteed` is False -- such
    queries must be compared directly rather than silently passed as clean.
    """
    parts = index_budget(len(s), band_pct) + 1
    segs = segments(s, parts)
    if not segs:
        return [], False
    k = min(MAX_KMER, min(len(seg) for seg in segs))
    if k < MIN_KMER:
        return [], False
    return [(k, seg[:k]) for seg in segs], True


def ancestor_keys(s: str, ks: Iterable[int]) -> set[tuple[int, str]]:
    """Every k-mer of an ancestor, for each k any query was indexed under."""
    out: set[tuple[int, str]] = set()
    for k in ks:
        if len(s) >= k:
            out.update((k, s[j:j + k]) for j in range(len(s) - k + 1))
    return out


def length_compatible(query: str, ancestor: str, band_pct: int = BAND) -> bool:
    """
    Length gate for the direct-comparison fallback.

    Derived from `max(len(query), len(ancestor))`, never from the ancestor alone
    -- the same asymmetry that produced the 9-vs-10 miss.
    """
    longest = max(len(query), len(ancestor))
    return abs(len(query) - len(ancestor)) <= max_edits(longest, band_pct)


class QueryIndex:
    """Filter index over ASD HCDR3 queries; ancestors are streamed past it."""

    def __init__(self, queries: Sequence[str], band_pct: int = BAND) -> None:
        self.queries = list(queries)
        self.band_pct = band_pct
        self.index: dict[tuple[int, str], list[int]] = collections.defaultdict(list)
        self.direct: list[int] = []
        self.ks: set[int] = set()
        for i, q in enumerate(self.queries):
            keys, guaranteed = query_keys(q, band_pct)
            if not guaranteed:
                self.direct.append(i)
                continue
            for key in set(keys):
                self.index[key].append(i)
                self.ks.add(key[0])

    def candidates(self, ancestor: str) -> set[int]:
        """Complete candidate superset for one ancestor."""
        found: set[int] = set()
        for key in ancestor_keys(ancestor, self.ks):
            found.update(self.index.get(key, ()))
        for i in self.direct:
            if length_compatible(self.queries[i], ancestor, self.band_pct):
                found.add(i)
        return found

    def neighbors(self, ancestor: str) -> list[tuple[int, float]]:
        """Verified `(query_index, similarity)` pairs within the band."""
        out = []
        for i in self.candidates(ancestor):
            sim = similarity(self.queries[i], ancestor, self.band_pct)
            if sim is not None:
                out.append((i, sim))
        return out


def brute_force_neighbors(queries: Sequence[str], ancestor: str,
                          band_pct: int = BAND) -> list[tuple[int, float]]:
    """All-pairs reference. Tests assert the index agrees with this exactly."""
    out = []
    for i, q in enumerate(queries):
        sim = similarity(q, ancestor, band_pct)
        if sim is not None:
            out.append((i, sim))
    return out


# --------------------------------------------------------------------------- #
# Ancestor eligibility
# --------------------------------------------------------------------------- #
def stage1_contributes_hcdr3(record: dict[str, Any]) -> bool:
    """
    Only HEAVY-chain Stage-1 records supply an ancestor HCDR3.

    Stage 1 is unpaired OAS and roughly 44% of its rows are light chains with
    `cdr3_aa` populated. Reading those as ancestor HCDR3 quarantines ASD heavy
    CDR3s on the strength of similar LIGHT CDR3s, which is a different sequence
    population and not evidence of exposure.
    """
    return str(record.get("chain_group") or "").lower() == "heavy"


# --------------------------------------------------------------------------- #
# Contact records
# --------------------------------------------------------------------------- #
@dataclass
class Contact:
    """
    One query's contact state. Emitted for EVERY query, contacted or not.

    Clean records are explicit so a later join can tell "measured and clean" from
    "never measured" -- silence is not evidence of disjointness.
    """

    key: str
    max_similarity: float = 0.0
    touches_stage1_train: bool = False
    touches_stage1_selection: bool = False
    touches_stage2_train: bool = False
    touches_stage2_selection: bool = False
    reason_exact_heavy: bool = False
    reason_hcdr3_ge_80: bool = False

    @property
    def hcdr3(self) -> str:
        """Back-compatible alias; `key` is an HCDR3 or a heavy chain."""
        return self.key

    def stages(self) -> set[str]:
        """Which ancestor stage/split this record was contacted from."""
        return {
            name for name in ("stage1_train", "stage1_selection",
                              "stage2_train", "stage2_selection")
            if getattr(self, f"touches_{name}")
        }

    def note(self, stage: str, split: str, similarity_value: float) -> None:
        setattr(self, f"touches_{stage}_{split}", True)
        self.reason_hcdr3_ge_80 = True
        self.max_similarity = max(self.max_similarity, similarity_value)

    def note_heavy(self, stage: str, split: str) -> None:
        setattr(self, f"touches_{stage}_{split}", True)
        self.reason_exact_heavy = True

    @property
    def contacted(self) -> bool:
        return self.reason_exact_heavy or self.reason_hcdr3_ge_80

    def to_json(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "max_similarity": round(self.max_similarity, 6),
            "touches_stage1_train": self.touches_stage1_train,
            "touches_stage1_selection": self.touches_stage1_selection,
            "touches_stage2_train": self.touches_stage2_train,
            "touches_stage2_selection": self.touches_stage2_selection,
            "reason_exact_heavy": self.reason_exact_heavy,
            "reason_hcdr3_ge_80": self.reason_hcdr3_ge_80,
        }


class ShardMismatch(RuntimeError):
    """Shards disagree about their contract, or are present but incomplete."""


class ManifestDamaged(RuntimeError):
    """The frozen universe's inputs are inconsistent; an error, not an exclusion."""


@dataclass
class ShardHeader:
    """Provenance every shard carries, so a join cannot mix incompatible runs."""

    manifest_sha256: str
    scanner_sha256: str
    corpus_sha256: dict[str, str]
    shard_index: int
    shard_count: int

    def contract(self) -> tuple:
        return (self.manifest_sha256, self.scanner_sha256,
                tuple(sorted(self.corpus_sha256.items())), self.shard_count)


def validate_shards(headers: Sequence[ShardHeader],
                    shard_queries: Sequence[Sequence[str]] | None = None,
                    expected_queries: Iterable[str] | None = None) -> None:
    """
    Refuse a join that is incomplete, duplicated, or mixed-provenance.

    Header agreement is necessary but NOT sufficient. A shard can be present,
    correctly labelled, and still have scanned only part of the query set --
    whereupon another shard's clean record for the same query survives the merge
    and its component is retained as clean, having never been compared against
    the first shard's ancestors. So when `shard_queries` and `expected_queries`
    are given, EVERY shard must carry exactly one record for every candidate.

    Raises:
        ShardMismatch: contracts differ; an index repeats, is missing, or is out
            of range; or a shard's query set is not exactly the manifest's.
    """
    if not headers:
        raise ShardMismatch("no shards supplied")
    contracts = {h.contract() for h in headers}
    if len(contracts) != 1:
        raise ShardMismatch(
            "shards were produced under different contracts (manifest, scanner or "
            "corpus hashes differ); results from different universes cannot be joined"
        )
    expected_count = headers[0].shard_count
    seen = collections.Counter(h.shard_index for h in headers)
    duplicates = sorted(i for i, n in seen.items() if n > 1)
    if duplicates:
        raise ShardMismatch(f"duplicate shard index/indices {duplicates}")
    out_of_range = sorted(i for i in seen if not 0 <= i < expected_count)
    if out_of_range:
        raise ShardMismatch(
            f"shard index/indices {out_of_range} outside [0, {expected_count})")
    missing = sorted(set(range(expected_count)) - set(seen))
    if missing:
        raise ShardMismatch(
            f"missing shard index/indices {missing} of {expected_count}; a partial "
            "scan understates contact and overstates the retained cohort")

    if (shard_queries is None) != (expected_queries is None):
        raise ShardMismatch(
            "shard_queries and expected_queries must be supplied together; one "
            "without the other silently skips record-completeness checking, which "
            "is the check that prevents a false-clean component"
        )
    if shard_queries is None:
        return
    if len(shard_queries) != len(headers):
        raise ShardMismatch(
            f"{len(shard_queries)} query collection(s) for {len(headers)} shard "
            "header(s); zip would truncate silently and leave the surplus shards "
            "unchecked"
        )
    expected = set(expected_queries)
    for header, queries in zip(headers, shard_queries):
        keys = list(queries)
        counts = collections.Counter(keys)
        repeated = sorted(k for k, n in counts.items() if n > 1)
        if repeated:
            raise ShardMismatch(
                f"shard {header.shard_index} repeats {len(repeated)} query record(s)")
        got = set(keys)
        absent = expected - got
        unexpected = got - expected
        if absent:
            raise ShardMismatch(
                f"shard {header.shard_index} is missing {len(absent)} candidate query "
                "record(s); a present-but-incomplete shard yields FALSE-CLEAN "
                "components, because another shard's clean record survives the merge")
        if unexpected:
            raise ShardMismatch(
                f"shard {header.shard_index} carries {len(unexpected)} record(s) "
                "outside the frozen manifest")


def merge_contacts(shards: Iterable[Iterable[Contact]]) -> dict[str, Contact]:
    """Merge per-shard contact records; every flag is a logical OR."""
    merged: dict[str, Contact] = {}
    for shard in shards:
        for contact in shard:
            current = merged.get(contact.key)
            if current is None:
                merged[contact.key] = Contact(**dict(contact.__dict__))
                continue
            current.max_similarity = max(current.max_similarity, contact.max_similarity)
            for name in ("touches_stage1_train", "touches_stage1_selection",
                         "touches_stage2_train", "touches_stage2_selection",
                         "reason_exact_heavy", "reason_hcdr3_ge_80"):
                if getattr(contact, name):
                    setattr(current, name, True)
    return merged


# --------------------------------------------------------------------------- #
# Join: contact -> whole-component quarantine
# --------------------------------------------------------------------------- #
def quarantine_components(
    membership: dict[tuple[str, str], dict[str, set]],
    hcdr3_contacts: dict[str, Contact],
    heavy_contacts: dict[str, Contact] | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    """
    Quarantine every component containing at least one contacted sequence.

    Identity is `(canonical_target, component_id)`, never a bare component id and
    never a flat HCDR3 -> component dict. Components are TARGET-LOCAL, and in the
    real corpus 4,868 of 10,020 labelled HCDR3 occur under more than one target
    (one under 842). A single global dictionary cannot represent those different
    memberships, and one contact must fan out to every target-local component
    containing that sequence.

    Args:
        membership: (target, component) -> {"hcdr3": {...}, "heavy": {...}}.
        hcdr3_contacts: HCDR3 -> record, one per candidate query.
        heavy_contacts: heavy chain -> record. Exact heavy identity gates
            independently AND carries its own stage/split attribution; an earlier
            version quarantined on heavy contact but reported `stages: []`.

    Returns:
        (target, component) -> state. A sequence with no record is `unmeasured`
        and quarantines its component: silence is not evidence of disjointness.
    """
    heavy_contacts = heavy_contacts or {}
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for key, members in membership.items():
        state: dict[str, Any] = {
            "quarantined": False, "reason_exact_heavy": False,
            "reason_hcdr3_ge_80": False, "unmeasured": False,
            "stages": set(), "contacted_sequences": 0,
        }
        for axis, contacts in (("hcdr3", hcdr3_contacts), ("heavy", heavy_contacts)):
            for sequence in sorted(members.get(axis, ())):
                contact = contacts.get(sequence)
                if contact is None:
                    state["unmeasured"] = True
                    state["quarantined"] = True
                    continue
                if not contact.contacted:
                    continue
                state["quarantined"] = True
                state["contacted_sequences"] += 1
                if axis == "heavy":
                    state["reason_exact_heavy"] = True
                else:
                    state["reason_exact_heavy"] |= contact.reason_exact_heavy
                    state["reason_hcdr3_ge_80"] |= contact.reason_hcdr3_ge_80
                state["stages"].update(contact.stages())
        state["stages"] = sorted(state["stages"])
        out[key] = state
    return out


# --------------------------------------------------------------------------- #
# Candidate universe
# --------------------------------------------------------------------------- #
#: Deterministic inclusion. No m_t floor: no precision analysis justifies one, so
#: small panels are included and Commit C decides whether their intervals help.
INCLUSION_CRITERIA = (
    "exact_component_map_available",
    "has_binary_binder",
    "has_binary_nonbinder",
    "has_usable_hcdr3",
)


def evaluate_candidate(target: dict[str, Any]) -> tuple[bool, list[str]]:
    """Decide inclusion from structure alone, returning the FAILED criteria."""
    failed = []
    if target.get("measurement_method") != "exact":
        failed.append("exact_component_map_available")
    labels = target.get("labels") or {}
    if not labels.get("binder_positive"):
        failed.append("has_binary_binder")
    if not labels.get("binder_negative"):
        failed.append("has_binary_nonbinder")
    if not target.get("distinct_hcdr3"):
        failed.append("has_usable_hcdr3")
    return (not failed), failed


def reconcile_component_map(canonical: str, target: dict[str, Any],
                            mapping: dict[str, str] | None,
                            band_percent: int = BAND) -> None:
    """
    Check one target's component map against its sizing record.

    Every disagreement is a DAMAGED INPUT, not a scientific exclusion. An earlier
    version recorded a missing map as the `exact_component_map_available`
    criterion failing, so a truncated map would silently shrink the frozen
    universe while the run reported success.

    Raises:
        ManifestDamaged: the map is absent for an otherwise eligible target, or
            its band, HCDR3 count, or component count disagrees with the sizing.
    """
    if mapping is None:
        raise ManifestDamaged(
            f"{canonical}: eligible by structure but has no component map. A "
            "missing map is a damaged input, not an exclusion -- refusing rather "
            "than silently shrinking the frozen universe.")
    band = (target.get("bands") or {}).get(str(band_percent))
    if band is None:
        raise ManifestDamaged(f"{canonical}: sizing has no band {band_percent}")
    if len(mapping) != target.get("distinct_hcdr3"):
        raise ManifestDamaged(
            f"{canonical}: component map has {len(mapping):,} HCDR3 against the "
            f"sizing's {target.get('distinct_hcdr3'):,}")
    components = len(set(mapping.values()))
    if components != band.get("m_t"):
        raise ManifestDamaged(
            f"{canonical}: component map has {components:,} components against the "
            f"sizing's m_t={band.get('m_t'):,}")


def canonical_jsonl_sha256(rows: Iterable[str]) -> str:
    """
    Hash the DECOMPRESSED canonical lines, never the compressed bytes.

    gzip embeds an mtime by default, so an identical logical map regenerated
    later hashes differently and a frozen reference stops matching for reasons
    unrelated to content.
    """
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row.rstrip("\n").encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def manifest_hash(payload: dict[str, Any]) -> str:
    """
    Content hash of the frozen universe, excluding only `manifest_sha256`.

    Provenance MUST already be present. An earlier version hashed the body and
    attached provenance afterwards, so the sizing, component-map and mechanism
    hashes could all change while the "frozen" manifest hash stayed identical.
    """
    body = {k: v for k, v in payload.items() if k != "manifest_sha256"}
    text = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def verify_manifest(payload: dict[str, Any]) -> None:
    """
    Recompute and check a manifest's own hash.

    Raises:
        ManifestDamaged: no recorded hash, no provenance to bind, or a mismatch.
    """
    recorded = payload.get("manifest_sha256")
    if not recorded:
        raise ManifestDamaged("manifest carries no manifest_sha256")
    if not payload.get("provenance"):
        raise ManifestDamaged(
            "manifest carries no provenance; its hash would not bind the sizing, "
            "component map or mechanism it was derived from")
    actual = manifest_hash(payload)
    if actual != recorded:
        raise ManifestDamaged(
            f"manifest hash mismatch: recorded {recorded[:16]}..., actual {actual[:16]}...")
