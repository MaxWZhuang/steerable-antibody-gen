"""Software-only fixtures for the qualification slice tests.

Every sequence, identifier, citation and value here is an invented test string. None of it is data,
a measurement, or a stand-in for either; no claim about any real antibody or antigen rests on it.
This module is torch-free so the records/config/probes tests do not need torch.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from smallAntibodyGen.experiments.interaction_qualification.records import (
    Panel,
    build_panel,
    parse_manifest,
    parse_records,
)

HEAVIES = ("ACD", "ACE", "ACF", "ACG", "ACH", "ACI")
ANTIGENS = ("EFG", "EFH", "EFI", "EFK", "EFL", "EFM")
SYNTHETIC_NOTE = "synthetic software-test declaration; not a measurement"


def row(
    record_id: str,
    heavy: str = "ACD",
    light: Optional[str] = None,
    antigen: str = "EFG",
    *,
    partition: str = "train",
    source: str = "study1",
    assay: str = "assay1",
    value: float = 7.3,
    antibody_group: Optional[str] = None,
    antigen_group: Optional[str] = None,
    region: Optional[Sequence[Sequence[Any]]] = None,
) -> Dict[str, Any]:
    """One ``aa-record/1`` row. Groups default to a name derived from the sequence, so they stay consistent."""
    return {
        "schema": "aa-record/1",
        "record_id": record_id,
        "heavy": heavy,
        "light": light,
        "antigen": antigen,
        "antibody_group": antibody_group if antibody_group is not None else f"abg-{heavy}-{light or 'none'}",
        "antigen_group": antigen_group if antigen_group is not None else f"agg-{antigen}",
        "partition": partition,
        "source": source,
        "assay": assay,
        "measurement": {"kind": "exact_pkd", "value": value, "measured": True},
        "region": None if region is None else [list(coord) for coord in region],
    }


def grid(
    partition: str,
    heavies: Iterable[str],
    antigens: Iterable[str],
    *,
    source: str = "study1",
    assay: str = "assay1",
    light: Optional[str] = None,
    value: float = 7.0,
) -> List[Dict[str, Any]]:
    """Every (heavy, antigen) cell of one partition/source/assay, with unique record ids."""
    return [
        row(
            f"{partition}-{source}-{assay}-{heavy}-{antigen}",
            heavy=heavy,
            light=light,
            antigen=antigen,
            partition=partition,
            source=source,
            assay=assay,
            value=value + 0.01 * index,
        )
        for index, (heavy, antigen) in enumerate(itertools.product(heavies, antigens))
    ]


def declarations(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """A provenance block declaring exactly the (source, assay) pairs the rows use."""
    sources: Dict[str, Any] = {}
    for item in rows:
        entry = sources.setdefault(item["source"], {"citation": f"synthetic citation for {item['source']}", "assays": {}})
        entry["assays"][item["assay"]] = f"synthetic exact equilibrium pKD for {item['assay']} ({SYNTHETIC_NOTE})"
    return sources


def jsonl_bytes(rows: Iterable[Dict[str, Any]]) -> bytes:
    return "".join(json.dumps(item) + "\n" for item in rows).encode("utf-8")


def manifest_dict(
    rows: Sequence[Dict[str, Any]],
    sha256: str,
    *,
    regime: str = "both",
    sources: Optional[Dict[str, Any]] = None,
    records_file: str = "panel.jsonl",
) -> Dict[str, Any]:
    return {
        "schema": "aa-manifest/1",
        "records_file": records_file,
        "records_sha256": sha256,
        "sources": declarations(rows) if sources is None else sources,
        "split_regime": regime,
        "split_basis": "synthetic: groups assigned by the test author",
    }


def panel_of(
    rows: Sequence[Dict[str, Any]], regime: str = "both", sources: Optional[Dict[str, Any]] = None
) -> Panel:
    """Parse and audit rows in memory (no files)."""
    data = jsonl_bytes(rows)
    sha = hashlib.sha256(data).hexdigest()
    manifest = parse_manifest(json.dumps(manifest_dict(rows, sha, regime=regime, sources=sources)))
    return build_panel(manifest, parse_records(data), sha)


def write_panel(
    directory: Path,
    rows: Sequence[Dict[str, Any]],
    *,
    regime: str = "both",
    sources: Optional[Dict[str, Any]] = None,
    records_name: str = "panel.jsonl",
    manifest_name: str = "manifest.json",
) -> Path:
    """Write a records file and its manifest next to each other; returns the manifest path."""
    data = jsonl_bytes(rows)
    (directory / records_name).write_bytes(data)
    manifest = manifest_dict(
        rows, hashlib.sha256(data).hexdigest(), regime=regime, sources=sources, records_file=records_name
    )
    path = directory / manifest_name
    path.write_bytes(json.dumps(manifest).encode("utf-8"))
    return path


def config_dict(
    *,
    evaluate_test: bool = False,
    max_tokens: int = 1000,
    ridge_alpha: float = 1.0,
    qualification_id: str = "synthetic-software-test",
    **diagnostics: Any,
) -> Dict[str, Any]:
    """An ``aa-qualification/1`` config; keyword overrides replace whole ``diagnostics`` entries."""
    diag: Dict[str, Any] = {
        "seed": 0,
        "partitions": ["development"],
        "mask_counts": [1, "full"],
        "geometries": ["suffix"],
        "max_examples": 8,
        "max_forwards": 100000,
        "local_gap": {"enabled": True, "max_pairs": 8, "kappa": 1.0},
    }
    diag.update(diagnostics)
    return {
        "schema": "aa-qualification/1",
        "qualification_id": qualification_id,
        "max_tokens": max_tokens,
        "ridge_alpha": ridge_alpha,
        "evaluate_test": evaluate_test,
        "diagnostics": diag,
    }


def write_inputs(
    directory: Path,
    rows: Sequence[Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None,
    *,
    regime: str = "both",
) -> Tuple[Path, Path]:
    """Write ``config.json``, ``panel.jsonl`` and ``manifest.json`` into ``directory``; returns (config, manifest)."""
    directory.mkdir(parents=True, exist_ok=True)
    manifest = write_panel(directory, rows, regime=regime)
    config_path = directory / "config.json"
    config_path.write_text(json.dumps(config_dict() if config is None else config), encoding="utf-8")
    return config_path, manifest
