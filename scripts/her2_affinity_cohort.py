#!/usr/bin/env python
"""Recover the independent measured-affinity cohort from the published benchmark.

Source: ``buzz/data/spr/spr_results.xlsx`` in the pinned HER2 release (the SPR
plate the benchmark reports). 768 assayed clones; this script keeps the ones
that are **in support** for this study's model -- a 120-residue heavy chain whose
fixed scaffold matches the released trastuzumab framework exactly, so that the
ten editable residues are the only thing that varies -- and that carry a
**finite measured KD**.

``-log10(KD)`` is used throughout, so higher always means stronger affinity.

Nothing here scores a model. It builds the cohort table and the two
model-free baselines (distance to wild type, nearest labelled-library
neighbour), so the GPU pass has a frozen row set to score against.

openpyxl is not installed and this deliberately does not install it while a
training run shares the interpreter: an xlsx is a zip of XML and is read here
directly.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smallAntibodyGen.experiments import her2_data as data_lib  # noqa: E402

SHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


def read_sheet(path):
    """Every row of the first worksheet, shared strings resolved."""
    archive = zipfile.ZipFile(path)
    shared = []
    if "xl/sharedStrings.xml" in archive.namelist():
        shared = ["".join(node.text or "" for node in item.iter(f"{{{SHEET_NS}}}t"))
                  for item in ET.fromstring(archive.read("xl/sharedStrings.xml"))]
    root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    rows = []
    for row in root.iter(f"{{{SHEET_NS}}}row"):
        values = []
        for cell in row.iter(f"{{{SHEET_NS}}}c"):
            value = cell.find(f"{{{SHEET_NS}}}v")
            if value is None:
                values.append("")
            elif cell.get("t") == "s":
                values.append(shared[int(value.text)])
            else:
                values.append(value.text)
        rows.append(values)
    return rows


def header_index(rows):
    for position, row in enumerate(rows):
        if any((cell or "").strip().startswith("KD") for cell in row):
            return position
    raise SystemExit("no KD header found in the workbook")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default="data/raw/her2_functional_20260918")
    parser.add_argument("--output", default="outputs/her2_biology_20260923")
    args = parser.parse_args(argv)

    raw = ROOT / args.raw_root
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    workbook = raw / "buzz" / "data" / "spr" / "spr_results.xlsx"

    rows = read_sheet(workbook)
    start = header_index(rows)
    header = [(cell or "").replace("\n", " ").strip() for cell in rows[start]]
    kd_column = next(i for i, name in enumerate(header) if name.startswith("KD"))
    heavy_column = header.index("H")
    label_column = header.index("LABEL")
    clone_column = 0

    scaffold = data_lib.load_scaffold(raw)
    prefix, suffix = scaffold.prefix, scaffold.suffix
    wt = data_lib.WT_CORE

    kept, dropped = [], {"non_finite_kd": 0, "heavy_length": 0, "scaffold_mismatch": 0,
                         "noncanonical_core": 0}
    canonical = set(data_lib.CANONICAL)
    for row in rows[start + 1:]:
        if len(row) <= max(kd_column, heavy_column, label_column):
            continue
        raw_kd, heavy = (row[kd_column] or "").strip(), (row[heavy_column] or "").strip()
        try:
            kd = float(raw_kd)
        except ValueError:
            dropped["non_finite_kd"] += 1
            continue
        if not np.isfinite(kd) or kd <= 0:
            dropped["non_finite_kd"] += 1
            continue
        if len(heavy) != data_lib.HEAVY_LENGTH:
            dropped["heavy_length"] += 1
            continue
        core = heavy[data_lib.CORE_START:data_lib.CORE_START + data_lib.CORE_LENGTH]
        if heavy[:data_lib.CORE_START] != prefix[1:] or \
                heavy[data_lib.CORE_START + data_lib.CORE_LENGTH:] != suffix:
            dropped["scaffold_mismatch"] += 1
            continue
        if not set(core) <= canonical:
            dropped["noncanonical_core"] += 1
            continue
        kept.append({"clone_id": row[clone_column], "core": core, "kd_molar": kd,
                     "neg_log10_kd": float(-np.log10(kd)),
                     "source_method": (row[label_column] or "").strip(),
                     "wt_distance": int(sum(a != b for a, b in zip(core, wt)))})

    # Model-free baselines, computed against the labelled library the flight
    # trained and evaluated on. Both are scorers in their own right and the GPU
    # pass must beat them, not merely beat chance.
    train = data_lib.load_split(raw, "train")
    val = data_lib.load_split(raw, "val")
    library = {}
    for frame in (train, val):
        for core, klass in zip(frame.seq, frame["class"]):
            library.setdefault(core, klass)
    library_cores = np.array(list(library))
    library_index = data_lib.encode_cores(library_cores)
    library_class = np.array([library[c] for c in library_cores])
    class_value = {"low": 0.0, "mid": 1.0, "high": 2.0}

    for record in kept:
        index = data_lib.encode_cores([record["core"]])[0]
        distances = (library_index != index[None, :]).sum(axis=1)
        nearest = int(distances.min())
        neighbours = library_class[distances == nearest]
        record["in_library"] = bool(nearest == 0)
        record["library_class"] = library[record["core"]] if nearest == 0 else ""
        record["nearest_library_distance"] = nearest
        record["nearest_neighbour_score"] = float(
            np.mean([class_value[c] for c in neighbours]))
        record["nearest_neighbour_support"] = int(neighbours.size)

    fields = ["clone_id", "core", "kd_molar", "neg_log10_kd", "source_method",
              "wt_distance", "in_library", "library_class", "nearest_library_distance",
              "nearest_neighbour_score", "nearest_neighbour_support"]
    with (out / "affinity_152.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(kept)

    by_method = {}
    for record in kept:
        by_method.setdefault(record["source_method"], []).append(record)
    summary = {
        "record_kind": "affinity_cohort",
        "source": str(workbook.relative_to(ROOT).as_posix()),
        "assayed_rows": len(rows) - start - 1,
        "kept": len(kept),
        "dropped": dropped,
        "in_support_rule": ("120-residue heavy chain whose scaffold prefix and suffix match "
                            "the released trastuzumab framework exactly, canonical core "
                            "residues, and a finite positive KD"),
        "affinity_convention": "-log10(KD); higher is stronger",
        "wild_type_core": wt,
        "by_source_method": {name: {
            "designs": len(items),
            "in_library": sum(1 for r in items if r["in_library"]),
            "neg_log10_kd_mean": float(np.mean([r["neg_log10_kd"] for r in items])),
            "neg_log10_kd_min": float(np.min([r["neg_log10_kd"] for r in items])),
            "neg_log10_kd_max": float(np.max([r["neg_log10_kd"] for r in items])),
        } for name, items in sorted(by_method.items())},
        "score_selection_exposure": (
            "designs whose core also appears in the labelled library are flagged "
            "in_library. Those rows were available to training and to the flight's "
            "evaluation panel, so a correlation computed over them is not independent "
            "of the adaptation data."),
        "recorded_at": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(),
    }
    (out / "affinity_cohort.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    print(f"assayed rows          {summary['assayed_rows']}")
    print(f"kept (in support, KD) {summary['kept']}")
    print(f"dropped               {dropped}")
    print()
    print("%-22s %7s %10s %8s %8s" % ("source method", "designs", "in library",
                                      "min", "max"))
    for name, block in summary["by_source_method"].items():
        print("%-22s %7d %10d %8.2f %8.2f" % (
            name or "(blank)", block["designs"], block["in_library"],
            block["neg_log10_kd_min"], block["neg_log10_kd_max"]))
    print()
    print(f"wrote {out / 'affinity_152.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
