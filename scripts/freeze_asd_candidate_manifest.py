#!/usr/bin/env python
"""
Freeze the ASD candidate universe (Commit B1), before any ancestor contact exists.

Inclusion is DETERMINISTIC and structural, never hand-picked. A target enters the
universe when all of these hold:

    exact component map available   (sampled-percolation targets are excluded --
                                     their component structure was never resolved)
    at least one binary binder
    at least one binary nonbinder
    a usable HCDR3

There is deliberately **no `m_t >= 50` floor**. No precision analysis justifies a
threshold yet, so small panels are included and Commit C decides whether their
intervals are useful. Imposing a floor now would be a guess dressed as a
criterion.

The manifest carries every candidate component and HCDR3, the reason each target
was included or excluded, corpus hashes, the component-map hash, and its own
content hash. It carries **no ancestor-contact information** -- that is the whole
point: the universe is fixed before the outcomes are known, so a later
restriction cannot become outcome-driven.

Fuzzy full-heavy similarity is DEFERRED, not silently dropped: it does not gate
eligibility, and the framework controls it would inform (antibody-only and
framework-only baselines, matched antigen swaps, label-switching subsets) belong
to the final evaluator.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from smallAntibodyGen import ancestor_quarantine as aq  # noqa: E402

SCHEMA_VERSION = "asd-candidate-manifest/1"
SIZING = PROJECT_ROOT / "specs/evidence/asd-cohort-sizing.json"
COMPONENT_MAP = PROJECT_ROOT / "specs/evidence/asd-component-map.jsonl.gz"


def display(path: Path) -> str:
    """Repo-relative when possible, absolute otherwise -- never a crash.

    `relative_to` raises for a path outside the repository, which is a legitimate
    choice for `--output-json` during testing, and a success message must not be
    able to fail a run that already succeeded.
    """
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--sizing", type=Path, default=SIZING)
    parser.add_argument("--component-map", type=Path, default=COMPONENT_MAP)
    parser.add_argument("--output-json", type=Path,
                        default=PROJECT_ROOT / "specs/evidence/asd-candidate-manifest.json")
    args = parser.parse_args()

    for path in (args.sizing, args.component_map):
        if not path.exists():
            print(f"REFUSED: {display(path)} does not exist. "
                  "The candidate universe is derived from the sizing artifact and its "
                  "component map; it cannot be invented.", file=sys.stderr)
            return 1

    sizing = json.loads(args.sizing.read_text(encoding="utf-8"))

    # Duplicate target rows are a DAMAGED MAP, not a last-one-wins situation. The
    # earlier version assigned into a dict, so a second row for a target silently
    # replaced the first and nothing reported it.
    raw_rows: list[str] = []
    components_by_target: dict[str, dict[str, str]] = {}
    with gzip.open(args.component_map, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw_rows.append(line)
            row = json.loads(line)
            target = row["canonical_target"]
            if target in components_by_target:
                print(f"REFUSED: component map contains duplicate rows for {target}. "
                      "A duplicate is a damaged input, not a last-one-wins choice.",
                      file=sys.stderr)
                return 1
            if row.get("band_percent") != aq.BAND:
                print(f"REFUSED: {target} mapped at band {row.get('band_percent')}, "
                      f"but this manifest freezes band {aq.BAND}.", file=sys.stderr)
                return 1
            components_by_target[target] = row["components"]

    included: dict[str, Any] = {}
    excluded: dict[str, list[str]] = {}
    try:
        for canonical, target in sorted(sizing.get("targets", {}).items()):
            ok, failed = aq.evaluate_candidate(target)
            if not ok:
                excluded[canonical] = failed
                continue
            # Structurally eligible, so the map MUST be present and consistent.
            # Any disagreement raises rather than quietly shrinking the universe.
            mapping = components_by_target.get(canonical)
            aq.reconcile_component_map(canonical, target, mapping, aq.BAND)
            by_component: dict[str, dict[str, list[str]]] = {}
            for hcdr3, component in sorted(mapping.items()):
                by_component.setdefault(component, {"hcdr3": []})["hcdr3"].append(hcdr3)
            included[canonical] = {
                "rows": target["rows"],
                "distinct_hcdr3": target["distinct_hcdr3"],
                "m_t": target["bands"][str(aq.BAND)]["m_t"],
                "labels": target["labels"],
                "component_hash": target["bands"][str(aq.BAND)]["component_hash"],
                "components": {c: {"hcdr3": sorted(v["hcdr3"])}
                               for c, v in sorted(by_component.items())},
            }
    except aq.ManifestDamaged as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1

    # Provenance goes in FIRST. The earlier version hashed the body and attached
    # provenance afterwards, so the sizing, component-map and mechanism hashes
    # could all change while the "frozen" manifest hash stayed identical.
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "band_percent": aq.BAND,
        "inclusion_criteria": list(aq.INCLUSION_CRITERIA),
        "no_component_floor": "deliberate: no m_t threshold is justified before "
                              "precision analysis, so small panels are included",
        "fuzzy_full_heavy": "DEFERRED -- does not gate eligibility; framework "
                            "controls belong to the final evaluator",
        "component_identity": "(canonical_target, component_id); components are "
                              "target-local and one HCDR3 may belong to several",
        "contains_ancestor_contact": False,
        "provenance": {
            "sizing_sha256": file_sha256(args.sizing),
            # Hashed on DECOMPRESSED canonical content: gzip embeds an mtime, so
            # hashing compressed bytes makes an identical logical map hash
            # differently when regenerated.
            "component_map_sha256": aq.canonical_jsonl_sha256(raw_rows),
            "mechanism_sha256": file_sha256(
                PROJECT_ROOT / "src/smallAntibodyGen/ancestor_quarantine.py"),
            "sizing_code_sha256": file_sha256(PROJECT_ROOT / "scripts/size_asd_cohorts.py"),
        },
        "included": included,
        "excluded": {k: sorted(v) for k, v in sorted(excluded.items())},
    }
    body["manifest_sha256"] = aq.manifest_hash(body)
    aq.verify_manifest(body)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8", newline="") as handle:
        json.dump(body, handle, indent=2, sort_keys=True)
        handle.write("\n")

    total_components = sum(len(t["components"]) for t in included.values())
    total_hcdr3 = sum(t["distinct_hcdr3"] for t in included.values())
    print(f"candidate targets included : {len(included):,}")
    print(f"          excluded         : {len(excluded):,}")
    print(f"candidate components       : {total_components:,}")
    print(f"candidate distinct HCDR3   : {total_hcdr3:,}")
    print(f"manifest_sha256            : {body['manifest_sha256']}")
    print(f"\nwrote {display(args.output_json)}")
    print("contains NO ancestor-contact information, by construction")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
