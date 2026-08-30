#!/usr/bin/env python
"""
Size ASD as a SECONDARY benchmark: how much independent evidence survives the
>=0.80 component quarantine, for EVERY canonical target.

This computes and freezes the model-blind half of the contract -- the similarity
rule, the component ids it induces, and the per-target independent-component
count `m_t` a conditional sampler needs -- and deliberately stops there. Which
components become `selection` and which become `final` is NOT decided here; that
waits on J05c's common evaluation contract, so ASD is not fitted around whatever
happens to be left.

The exclusion rule (predeclared):

    excluded_as_related = exact_heavy_chain
                          or lineage_edge          # same IGHV/IGHJ, same len, >=0.80 identity
                          or model_neighbor_edge   # normalized Levenshtein >=0.80

`lineage_edge` is not computed, and does not need to be. It requires equal
length, and for equal-length strings Levenshtein <= Hamming, so
identity >= 0.80 implies normalized-Levenshtein >= 0.80: it is a strict SUBSET of
`model_neighbor_edge` and the union is unchanged. (ASD carries no V/J annotation
in either the processed corpus or the 11-column raw shards, so it could not have
been computed anyway -- but the union is exact, not an approximation.)

Thresholds are INTEGER arithmetic throughout. `floor((1.0 - 0.80) * length)` is
wrong: 1.0 - 0.80 evaluates to 0.19999999999999996, which allows one edit too few
at lengths 5, 10, 15, 20, ... -- fewer edges, more apparent components, and an
overstatement of independence exactly where it matters.

Bands are reported at exact / >=0.90 / >=0.80. The PRIMARY exclusion is >=0.80.
If that leaves too little, the honest outcome is insufficient final evidence, not
a looser cutoff chosen after seeing the sizes.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import importlib.util
import json
import math
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "asd-cohort-sizing/2"

ASD = PROJECT_ROOT / "data/processed/antibody_antigen/antibody_antigen.jsonl.gz"
CANONICALIZER = PROJECT_ROOT / "scripts/prepare_antibody_antigen.py"

#: Bands as INTEGER percent. Never a float threshold -- see the module docstring.
BAND_PERCENTS = (100, 90, 80)
PRIMARY_BAND = 80

#: Above this many distinct HCDR3 an exact component computation is not
#: affordable here; the target is measured by sampled percolation instead, with
#: the method recorded on the target. Only HER2 exceeds it.
EXACT_LIMIT = 60_000


def max_edits(length: int, band_pct: int) -> int:
    """Edits allowed at `band_pct` for a string of `length`, in integer arithmetic."""
    return ((100 - band_pct) * length) // 100


def bounded_levenshtein(a: str, b: str, limit: int) -> int:
    """
    Edit distance, abandoned as soon as it provably exceeds `limit`.

    Returns `limit + 1` on abandonment. The band is what makes this affordable:
    at >=0.80 on a 13-mer only distance <= 2 matters, so all but a narrow
    diagonal of the DP can be skipped.
    """
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
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (ca != b[j - 1]),
            ))
        current.extend([limit + 1] * (len(b) - hi))
        if min(current) > limit:
            return limit + 1
        previous = current
    return previous[len(b)]


def similar(a: str, b: str, band_pct: int) -> bool:
    """
    Normalized-Levenshtein similarity >= band, normalizing by the longer string.

    Stated as `100 * distance <= (100 - band_pct) * length` so the comparison is
    exact at the boundary rather than subject to binary floating point.
    """
    longest = max(len(a), len(b))
    if longest == 0:
        return True
    limit = max_edits(longest, band_pct)
    return 100 * bounded_levenshtein(a, b, limit) <= (100 - band_pct) * longest


class Union:
    """Disjoint set over an index space, path-halved."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def deletion_variants(s: str, d: int) -> set[str]:
    """
    Every string obtainable from `s` by deleting at most `d` characters.

    This is what makes the candidate search complete for Levenshtein. Positional
    blocking argues by pigeonhole, which holds only for substitutions: an indel
    shifts every later block, so two genuinely-close strings can share no aligned
    block and a real component gets split.

    Precisely: if `lev(a, b) <= d` then their d-deletion neighbourhoods intersect,
    so the intersection supplies a COMPLETE CANDIDATE SUPERSET. The converse does
    NOT hold -- intersecting neighbourhoods also produce candidates further apart
    than `d` -- which is harmless only because every candidate is then verified by
    exact bounded Levenshtein.
    """
    out = {s}
    frontier = {s}
    for _ in range(d):
        nxt = set()
        for t in frontier:
            for i in range(len(t)):
                nxt.add(t[:i] + t[i + 1:])
        out |= nxt
        frontier = nxt
    return out


def union_neighbors(seqs: list[str], band_pct: int, union: Union) -> dict[str, int]:
    """
    Union every pair within the band, verifying ALL candidate pairs.

    Each sequence collects every EARLIER sequence sharing any deletion variant,
    deduplicates them, and verifies each. An earlier version compared each bucket
    only against its first member, which is an anchor star rather than a graph:
    two non-anchor members can be neighbours when neither neighbours the anchor.
    Verified counterexample at >=0.80 -- {AAAAAB, AAAABA, AAAABB} is ONE component
    because AAAABB bridges, but an anchor star leaves AAAABA isolated.

    Skipping a pair already in the same component is still sound (the edge would
    change nothing) and is what keeps dense libraries affordable. No bucket is
    ever skipped for size.
    """
    index: dict[str, list[int]] = collections.defaultdict(list)
    verified = 0
    largest = 0
    for i, s in enumerate(seqs):
        limit = max_edits(len(s), band_pct)
        variants = deletion_variants(s, limit)
        candidates: set[int] = set()
        for variant in variants:
            bucket = index[variant]
            if bucket:
                largest = max(largest, len(bucket))
                candidates.update(bucket)
        for j in candidates:
            if union.find(i) == union.find(j):
                continue
            verified += 1
            if similar(seqs[i], seqs[j], band_pct):
                union.union(i, j)
        for variant in variants:
            index[variant].append(i)
    return {"index_keys": len(index), "largest_bucket": largest,
            "levenshtein_checks": verified, "buckets_skipped": 0}


def components(seqs: list[str], heavy_of: list[str], band_pct: int) -> tuple[list[int], dict]:
    """Component id per sequence under `exact_heavy OR neighbor(band)`."""
    union = Union(len(seqs))
    by_heavy: dict[str, int] = {}
    for i, h in enumerate(heavy_of):
        if not h:
            continue
        if h in by_heavy:
            union.union(i, by_heavy[h])
        else:
            by_heavy[h] = i
    stats = {"index_keys": 0, "largest_bucket": 0, "levenshtein_checks": 0, "buckets_skipped": 0}
    if band_pct < 100:
        stats = union_neighbors(seqs, band_pct, union)
    return [union.find(i) for i in range(len(seqs))], stats


def naive_components(seqs: list[str], heavy_of: list[str], band_pct: int) -> list[int]:
    """
    All-pairs reference implementation. O(n^2) and only for tests.

    The indexed graph must agree with this exactly; that equivalence is what the
    test suite checks exhaustively over a small alphabet.
    """
    union = Union(len(seqs))
    by_heavy: dict[str, int] = {}
    for i, h in enumerate(heavy_of):
        if not h:
            continue
        if h in by_heavy:
            union.union(i, by_heavy[h])
        else:
            by_heavy[h] = i
    if band_pct < 100:
        for i in range(len(seqs)):
            for j in range(i + 1, len(seqs)):
                if similar(seqs[i], seqs[j], band_pct):
                    union.union(i, j)
    return [union.find(i) for i in range(len(seqs))]


def shannon_effective(sizes: list[int]) -> float:
    """exp(Shannon entropy): a DIVERSITY statistic, not an effective sample size."""
    total = sum(sizes)
    if not total:
        return 0.0
    ent = -sum((s / total) * math.log(s / total) for s in sizes if s)
    return math.exp(ent)


def simpson_effective(sizes: list[int]) -> float:
    """
    1 / sum(q^2), the inverse participation ratio.

    Closer to an effective SAMPLE SIZE -- the number of equally-weighted draws
    with the same variance -- so it, not the perplexity, bounds achievable metric
    precision. The conservative of the two under skew.
    """
    total = sum(sizes)
    if not total:
        return 0.0
    return 1.0 / sum((s / total) ** 2 for s in sizes if s)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def load_preparer():
    """
    Import the production canonicalization, never a second copy of it.

    `canonical_target_id` does NOT exist in this corpus -- 0 of 828,315 rows
    carry it, because the file predates J02 -- so grouping on the stored
    `target_key` alone splits targets that canonicalization fuses (`pdb:aayl49`
    is 13,265 rows by legacy key against 17,585 as a canonical component). The
    identity graph is therefore BUILT here, from the same functions the preparer
    and the J02 audit use.
    """
    spec = importlib.util.spec_from_file_location("prepare_antibody_antigen", CANONICALIZER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def iter_asd() -> Iterator[dict[str, Any]]:
    with gzip.open(ASD, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def canonical_rows(preparer) -> list[dict[str, Any]]:
    """
    Two passes: a row's component is only known once every row is observed.

    Pass 1 reads IDENTITY ONLY -- never a label, affinity, or split -- so
    supervision cannot influence which rows are grouped together.
    """
    index = preparer.TargetIdentityIndex()
    for record in iter_asd():
        index.observe(preparer.extract_target_nodes(record, record.get("sequence_antigen") or ""))
    index.finalize()

    rows = []
    for record in iter_asd():
        nodes = preparer.extract_target_nodes(record, record.get("sequence_antigen") or "")
        canonical = (index.canonical_id(nodes) if nodes else "") or f"legacy:{record.get('target_key')}"
        rows.append({
            "canonical": canonical,
            "legacy": str(record.get("target_key") or ""),
            "hcdr3": record.get("cdr3_aa_heavy") or record.get("cdr3_aa"),
            "heavy": record.get("sequence_heavy") or "",
            "dataset": record.get("dataset"),
            "source_file": record.get("source_file"),
            "affinity_type": record.get("affinity_type"),
            "binder_label": record.get("binder_label"),
            "is_strong_binder": record.get("is_strong_binder"),
            "split": record.get("split"),
        })
    return rows


def percolation_curve(seqs: list[str], band_pct: int, seed: int = 0) -> list[dict[str, Any]]:
    """
    Largest-component fraction at increasing sample sizes.

    Used ONLY for targets too large to resolve exactly here. A rising curve is
    the percolation signature, so this MEASURES the canonical component rather
    than assuming it from a legacy subset -- but it is a sampled measurement and
    the report labels it as one.
    """
    rng = random.Random(seed)
    curve = []
    for n in (10_000, 40_000, 120_000):
        if n > len(seqs):
            break
        sample = rng.sample(seqs, n)
        comp, _ = components(sample, [""] * n, band_pct)
        sizes = collections.Counter(comp)
        curve.append({"n": n, "largest_fraction": sizes.most_common(1)[0][1] / n,
                      "components": len(sizes)})
    return curve


def size_target(recs: list[dict[str, Any]], exact: bool) -> dict[str, Any]:
    """Component structure, labels, provenance and component ids for one target."""
    distinct = sorted({r["hcdr3"] for r in recs if r["hcdr3"]})
    idx = {h: i for i, h in enumerate(distinct)}
    heavy_of = [""] * len(distinct)
    for r in recs:
        if r["hcdr3"] and r["heavy"] and not heavy_of[idx[r["hcdr3"]]]:
            heavy_of[idx[r["hcdr3"]]] = r["heavy"]

    labels = collections.Counter(str(r["binder_label"]) for r in recs)
    pos, neg = labels.get("1", 0), labels.get("0", 0)
    out: dict[str, Any] = {
        "rows": len(recs),
        "distinct_hcdr3": len(distinct),
        "legacy_keys_fused": sorted({r["legacy"] for r in recs}),
        "labels": {"binder_positive": pos, "binder_negative": neg,
                   "prevalence": (pos / (pos + neg)) if (pos + neg) else None,
                   "is_strong_binder": sum(1 for r in recs if r["is_strong_binder"])},
        "provenance": {
            "datasets": dict(collections.Counter(r["dataset"] for r in recs).most_common()),
            "affinity_types": dict(collections.Counter(r["affinity_type"] for r in recs).most_common()),
            "source_files": len({r["source_file"] for r in recs}),
        },
        "current_split": dict(collections.Counter(r["split"] for r in recs)),
        "bands": {},
    }

    if not exact:
        out["measurement_method"] = "sampled_percolation"
        out["bands"][str(PRIMARY_BAND)] = {
            "exact": False,
            "percolation_curve": percolation_curve(distinct, PRIMARY_BAND),
            "note": "too large to resolve exactly here; a rising largest-component "
                    "fraction is the percolation signature",
        }
        return out

    out["measurement_method"] = "exact"
    for band in BAND_PERCENTS:
        comp, stats = components(distinct, heavy_of, band)
        sizes = sorted(collections.Counter(comp).values(), reverse=True)
        rows_per: collections.Counter = collections.Counter()
        for r in recs:
            if r["hcdr3"]:
                rows_per[comp[idx[r["hcdr3"]]]] += 1
        # Stable id: the lexicographically smallest member of each component, so
        # ids depend on content alone -- not on iteration order, dict ordering,
        # or how many targets happened to run.
        reps: dict[int, str] = {}
        for h, i in idx.items():
            root = comp[i]
            if root not in reps or h < reps[root]:
                reps[root] = h
        out["bands"][str(band)] = {
            "exact": True,
            "m_t": len(sizes),
            "largest_component_fraction": (sizes[0] / len(distinct)) if distinct else None,
            "effective_components_shannon": round(shannon_effective(sizes), 1),
            "effective_components_simpson": round(simpson_effective(sizes), 1),
            "effective_components_simpson_by_rows": round(
                simpson_effective(list(rows_per.values())), 1),
            "singletons": sum(1 for x in sizes if x == 1),
            "unit": "components over DISTINCT HCDR3; *_by_rows weights by row count",
            "component_hash": hashlib.sha256(
                "\n".join(sorted(reps.values())).encode("utf-8")).hexdigest(),
            "search": stats,
        }
        if band == PRIMARY_BAND:
            out["_mapping"] = {h: reps[comp[i]] for h, i in idx.items()}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--output-json", type=Path,
                        default=PROJECT_ROOT / "specs/evidence/asd-cohort-sizing.json")
    parser.add_argument("--mapping-out", type=Path,
                        default=PROJECT_ROOT / "specs/evidence/asd-component-map.jsonl.gz")
    args = parser.parse_args()

    print("building the canonical identity graph (identity only, no labels read) ...")
    preparer = load_preparer()
    rows = canonical_rows(preparer)
    by_target: dict[str, list] = collections.defaultdict(list)
    for r in rows:
        by_target[r["canonical"]].append(r)
    print(f"  {len(rows):,} rows -> {len(by_target):,} canonical targets")

    targets: dict[str, Any] = {}
    mapping_lines: list[str] = []
    oversize: list[str] = []
    for n, (canonical, recs) in enumerate(sorted(by_target.items()), 1):
        distinct = len({r["hcdr3"] for r in recs if r["hcdr3"]})
        exact = distinct <= EXACT_LIMIT
        if not exact:
            oversize.append(canonical)
        sized = size_target(recs, exact)
        mapping = sized.pop("_mapping", None)
        if mapping:
            mapping_lines.append(json.dumps(
                {"canonical_target": canonical, "band_percent": PRIMARY_BAND,
                 "components": mapping}, sort_keys=True))
        targets[canonical] = sized
        if n % 400 == 0:
            print(f"  sized {n:,}/{len(by_target):,} targets ...")

    report = {
        "schema_version": SCHEMA_VERSION,
        "role": "SECONDARY benchmark. AVIDa is primary final evidence.",
        "membership_frozen": False,
        "frozen_here": "the similarity rule, the component ids it induces, and m_t "
                       "per canonical target (all model-blind)",
        "deferred": "selection vs final assignment -- awaits J05c's evaluation contract",
        "provenance": {
            "corpus_path": ASD.relative_to(PROJECT_ROOT).as_posix(),
            "corpus_sha256": file_sha256(ASD),
            "canonicalizer_path": CANONICALIZER.relative_to(PROJECT_ROOT).as_posix(),
            "canonicalizer_sha256": file_sha256(CANONICALIZER),
            "source_commit": git_commit(),
            "component_map": args.mapping_out.relative_to(PROJECT_ROOT).as_posix(),
        },
        "similarity_rule": {
            "primary_band_percent": PRIMARY_BAND,
            "bands_percent": list(BAND_PERCENTS),
            "arithmetic": "integer: accepted iff 100*distance <= (100-band)*max_len",
            "edges": "exact_heavy_chain OR normalized_levenshtein(hcdr3) >= band",
            "candidate_search": "deletion neighbourhoods, a complete candidate SUPERSET "
                                "for Levenshtein; every candidate verified exactly",
            "lineage_edge": "strict subset of the neighbor edge for equal-length pairs; "
                            "union unchanged. ASD carries no V/J annotation regardless",
        },
        "targets_measured_by_sampling": oversize,
        "targets": targets,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8", newline="") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with gzip.open(args.mapping_out, "wt", encoding="utf-8", newline="") as handle:
        for line in mapping_lines:
            handle.write(line + "\n")

    eligible = [(c, t) for c, t in targets.items()
                if t.get("measurement_method") == "exact"
                and t["bands"][str(PRIMARY_BAND)]["m_t"] >= 50]
    eligible.sort(key=lambda kv: -kv[1]["bands"][str(PRIMARY_BAND)]["effective_components_simpson"])
    print(f"\ntargets with m_t >= 50 independent components at >={PRIMARY_BAND}%: {len(eligible)}")
    for canonical, t in eligible[:12]:
        b = t["bands"][str(PRIMARY_BAND)]
        print(f"  {canonical[:40]:40} rows={t['rows']:>7,} m_t={b['m_t']:>6,} "
              f"simpson={b['effective_components_simpson']:>7.1f} "
              f"+{t['labels']['binder_positive']:,}/-{t['labels']['binder_negative']:,}")
    print(f"\nwrote {args.output_json.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"wrote {args.mapping_out.relative_to(PROJECT_ROOT).as_posix()}")
    print("membership NOT frozen: selection/final assignment awaits J05c")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
