"""Frozen, graded HCDR3 comparisons. This is a ranking/sensitivity benchmark.

Cross-target substitutions have no measured labels. Their results cannot
establish correct antigen conditioning. No training gate or source split is
changed here. Historical membership is a separate, checkpoint-specific audit.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

SCHEMA = "hcdr3-contrast-manifest/1"
CANONICAL_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
CORE_TARGETS = (
    "name:human_tigit", "name:sars_cov2_rbd", "pdb:aayl49", "name:human_pd_1",
)
HA_TARGETS = ("pdb:3gbm", "pdb:3gbn")


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def save_json(path: str | Path, payload: Any) -> None:
    """Write stable bytes, refusing to replace an existing different artifact."""
    path = Path(path)
    encoded = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if path.suffix == ".gz":
        encoded = gzip.compress(encoded, mtime=0)
    if path.exists():
        if path.read_bytes() == encoded:
            return
        raise FileExistsError(f"artifact already exists with different content: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(encoded)


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    return json.loads(raw)


def heavy_context(row: Mapping[str, Any]) -> tuple[str, str, str, str] | None:
    """Prefix, suffix, light chain, HCDR3; coordinates must agree with sequence."""
    heavy = row.get("sequence_heavy") or row.get("sequence") or ""
    start, end = row.get("cdr3_start_aa_heavy"), row.get("cdr3_end_aa_heavy")
    cdr3 = row.get("cdr3_aa_heavy")
    if start is None or end is None:
        start, end, cdr3 = row.get("cdr3_start_aa"), row.get("cdr3_end_aa"), row.get("cdr3_aa")
    if (not isinstance(heavy, str) or type(start) is not int or type(end) is not int
            or not 0 <= start < end <= len(heavy) or heavy[start:end] != cdr3):
        return None
    return heavy[:start], heavy[end:], row.get("sequence_light") or "", cdr3


def numeric_measurement(row: Mapping[str, Any]) -> float | None:
    value = row.get("processed_measurement_float")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def contrast_fold(background: tuple[str, str, str], seed: int) -> str:
    # The same background stays together across target, assay and loop length.
    bucket = int(digest([seed, list(background)])[:16], 16) % 100
    return "test" if bucket < 10 else "validation" if bucket < 20 else "train"


def summarize_groups(groups: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    groups = list(groups)
    rows = [r for g in groups for r in g["variants"]]
    return {
        "groups": len(groups), "rows": len(rows),
        "distinct_hcdr3s": len({r["hcdr3"] for r in rows}),
        "all_outside_strong_binder_gate": sum(not any(r["is_strong_binder"] for r in g["variants"])
                                              for g in groups),
        "source_splits": dict(sorted(Counter(r["source_split"] for r in rows).items())),
        "contrast_folds": dict(sorted(Counter(g["fold"] for g in groups).items())),
    }


def build_manifest(rows: Iterable[dict[str, Any]], *, canonicalize,
                   provenance: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize with the producer's finalized index, never a local alias rule."""
    by_context: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    excluded: Counter = Counter()
    targets = set(protocol["targets"])
    assay_rules = {(r["dataset"], r["affinity_type"]): r for r in protocol["assays"]}
    for rule in assay_rules.values():
        if rule["direction"] not in ("higher", "lower"):
            raise ValueError("assay direction must be higher or lower")
    cohort: dict[str, dict[str, Any]] = {}
    census: dict[tuple, list[tuple]] = defaultdict(list)
    for row in rows:
        target = canonicalize(row)
        context = heavy_context(row)
        # Reconciliation deliberately includes binary structural rows.
        if target in targets | set(HA_TARGETS):
            c = cohort.setdefault(target, {"rows": 0, "hcdr3s": set(), "backgrounds": set(),
                                          "antigens": set(), "antibodies": set(), "assays": Counter(),
                                          "graded_rows": 0, "graded_hcdr3s": set(),
                                          "graded_backgrounds": set(), "graded_antibodies": set()})
            c["rows"] += 1
            kind = str(row.get("affinity_type") or "").strip().lower()
            c["assays"][kind] += 1
            c["antigens"].add(row.get("sequence_antigen") or "")
            if context:
                pre, post, light, cdr3 = context
                c["hcdr3s"].add(cdr3); c["backgrounds"].add(context[:3])
                antibody = (pre + cdr3 + post, light)
                c["antibodies"].add(antibody)
                if kind == "-log kd" and numeric_measurement(row) is not None:
                    c["graded_rows"] += 1; c["graded_hcdr3s"].add(cdr3)
                    c["graded_backgrounds"].add(context[:3]); c["graded_antibodies"].add(antibody)
        kind = str(row.get("affinity_type") or "").strip().lower()
        value = numeric_measurement(row)
        # Broad non-binary census, separate from the approved benchmark predicate.
        if context and value is not None and kind not in ("bool", "fuzzy"):
            census[(context[:3], target, kind)].append((context[3], value, bool(row.get("is_strong_binder"))))
        if target not in targets:
            excluded["target_outside_protocol"] += 1
            continue
        dataset = str(row.get("dataset") or "")
        if (dataset, kind) not in assay_rules:
            excluded["assay_not_in_protocol"] += 1
            continue
        if context is None:
            excluded["invalid_or_inconsistent_hcdr3_span"] += 1
            continue
        if value is None:
            excluded["nonfinite_or_missing_measurement"] += 1
            continue
        if not set(context[3]) <= CANONICAL_AA:
            excluded["noncanonical_hcdr3_residue"] += 1
            continue
        antigen = row.get("sequence_antigen") or ""
        if not antigen:
            excluded["missing_antigen"] += 1
            continue
        # Assay/source strings are the available strata, not proof of identical
        # experimental conditions. That limitation is included in the artifact.
        key = (target, *context[:3], antigen, dataset, kind, row.get("source_url") or "",
               row.get("heavy_locus") or "IGH", row.get("light_locus") or "IGK")
        by_context[key].append({
            "record_id": str(row.get("record_id") or digest(row)), "hcdr3": context[3],
            "measurement": value, "measurement_raw": row.get("processed_measurement_raw"),
            "source_split": str(row.get("split") or "unknown"),
            "is_strong_binder": bool(row.get("is_strong_binder")),
        })
    groups = []
    for key, variants in sorted(by_context.items()):
        target, pre, post, light, antigen, dataset, kind, source, heavy_locus, light_locus = key
        measurements: dict[str, set[float]] = defaultdict(set)
        for r in variants:
            measurements[r["hcdr3"]].add(r["measurement"])
        if any(len(values) > 1 for values in measurements.values()):
            excluded["rows_in_group_with_conflicting_repeat_measurements"] += len(variants)
            continue
        if len(measurements) < 2 or len({r["measurement"] for r in variants}) < 2:
            excluded["rows_without_distinct_variant_measurement_contrast"] += len(variants)
            continue
        # Repeated identical observations do not receive extra statistical weight.
        unique = {}
        for r in sorted(variants, key=lambda x: (x["hcdr3"], x["record_id"])):
            kept = unique.setdefault(r["hcdr3"], dict(r, observation_ids=[]))
            kept["observation_ids"].append(r["record_id"])
        group = {
            "group_id": digest(key), "target": target,
            "prefix": pre, "suffix": post, "light": light, "antigen": antigen,
            "heavy_locus": heavy_locus, "light_locus": light_locus,
            "dataset": dataset, "affinity_type": kind, "source_url": source,
            "direction": assay_rules[(dataset, kind)]["direction"],
            "background_id": digest([pre, post, light]),
            "fold": contrast_fold((pre, post, light), protocol["seed"]),
            "variants": list(unique.values()),
        }
        groups.append(group)
    if not groups:
        raise ValueError("the contrast predicate produced no groups")
    # Freeze one alternative target sequence per group, picked without labels or scores.
    antigens = sorted({(g["target"], g["antigen"]) for g in groups})
    for g in groups:
        donors = [(t, a) for t, a in antigens if t != g["target"] and a != g["antigen"]]
        if donors:
            t, a = min(donors, key=lambda ta: (abs(len(ta[1]) - len(g["antigen"])), digest(ta)))
            g["antigen_control"] = {"target": t, "sequence": a,
                                     "mechanism": "cross_target_substitution_unmeasured"}
        else:
            g["antigen_control"] = None
    broad = {k: v for k, v in census.items()
             if k[1] not in HA_TARGETS and len({r[0] for r in v}) >= 2 and len({r[1] for r in v}) >= 2}
    reconciliation = {}
    for target, c in sorted(cohort.items()):
        reconciliation[target] = {k: len(v) if isinstance(v, set) else dict(v) if isinstance(v, Counter) else v
                                  for k, v in c.items()}
    if all(t in cohort for t in HA_TARGETS):
        a, b = (cohort[t] for t in HA_TARGETS)
        reconciliation["ha_shared"] = {k: len(a[k] & b[k]) for k in
                                        ("antibodies", "backgrounds", "graded_antibodies", "graded_backgrounds")}
    payload = {
        "schema": SCHEMA, "provenance": provenance, "protocol": protocol,
        "claims": {
            "measured": "within-context graded variant ranking under the recorded assay strata",
            "antigen_controls": "input sensitivity only; substituted pairs have no measured labels",
            "antigen_conditioning_correctness_established": False,
            "historical_holdout_established": False,
            "folds": "prospective background-disjoint folds within known targets; original splits retained",
            "neighborhood_disjointness_established": False,
            "assay_caveat": "dataset/type/source are available strata; batch, replicate uncertainty and censoring may be missing",
            "processed_corpus_caveat": "upstream sequence-triple deduplication may already have removed quantitative replicates",
        },
        "summary": summarize_groups(groups),
        "per_target": {t: summarize_groups(g for g in groups if g["target"] == t)
                       for t in sorted({g["target"] for g in groups})},
        "exclusions": dict(sorted(excluded.items())), "cohort_reconciliation": reconciliation,
        "broad_census_without_ha": {"predicate": "finite non-bool/non-fuzzy measurement, valid annotated HCDR3; group by prefix/suffix/light, canonical target, assay type",
                                    "groups": len(broad), "rows": sum(map(len, broad.values())),
                                    "distinct_hcdr3s": len({r[0] for v in broad.values() for r in v})},
        "groups": sorted(groups, key=lambda g: g["group_id"]),
    }
    payload["manifest_sha256"] = digest(payload)
    verify_manifest(payload)
    return payload


def verify_manifest(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != SCHEMA or not payload.get("groups"):
        raise ValueError("unsupported or empty contrast manifest")
    content = dict(payload)
    expected = content.pop("manifest_sha256", None)
    if not expected or digest(content) != expected:
        raise ValueError("contrast manifest checksum mismatch")
    seen = set()
    for g in payload["groups"]:
        if g["group_id"] in seen:
            raise ValueError("duplicate contrast group")
        seen.add(g["group_id"])
        if len({v["hcdr3"] for v in g["variants"]}) < 2:
            raise ValueError("contrast group has fewer than two variants")
        if len({v["measurement"] for v in g["variants"]}) < 2:
            raise ValueError("contrast group has no measurement difference")
        if len({v["hcdr3"] for v in g["variants"]}) != len(g["variants"]):
            raise ValueError("duplicate HCDR3 observation in contrast group")
        if g["fold"] != contrast_fold((g["prefix"], g["suffix"], g["light"]), payload["protocol"]["seed"]):
            raise ValueError("contrast group is assigned to an inconsistent fold")
        control = g["antigen_control"]
        if control and (control["target"] == g["target"] or control["sequence"] == g["antigen"]):
            raise ValueError("antigen control is not an alternative")


def concordance(values: list[float], scores: list[float], *, direction: str,
                score_tolerance: float = 1e-8) -> dict[str, Any]:
    """Pairwise order agreement; measured ties excluded, score ties count half."""
    if len(values) != len(scores) or direction not in ("higher", "lower"):
        raise ValueError("invalid concordance inputs")
    if not all(math.isfinite(x) for x in values + scores):
        raise ValueError("nonfinite concordance input")
    wins = losses = ties = measured_ties = 0
    sign = 1 if direction == "higher" else -1
    for i in range(len(values)):
        for j in range(i):
            difference = sign * (values[i] - values[j])
            if difference == 0:
                measured_ties += 1
                continue
            predicted = scores[i] - scores[j]
            if abs(predicted) <= score_tolerance:
                ties += 1
            elif difference * predicted > 0:
                wins += 1
            else:
                losses += 1
    n = wins + losses + ties
    return {"concordant": wins, "discordant": losses, "score_ties": ties,
            "measurement_ties": measured_ties, "comparable_pairs": n,
            "concordance": (wins + ties / 2) / n if n else None}
