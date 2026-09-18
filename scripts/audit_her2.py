#!/usr/bin/env python
"""Phase 0 CPU audit: HER2 provenance, split integrity, support, proximity, assay cohort.

Reads nothing it is not allowed to read. Train and validation labels are audited
in full; the test split is touched only through its ``seq`` column, and the assay
workbook only through its metadata columns. Both restrictions are enforced by the
loaders in ``her2_data``, not by this script remembering to behave.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smallAntibodyGen.benchmarks import provenance as prov  # noqa: E402
from smallAntibodyGen.experiments import her2_data as data  # noqa: E402
from smallAntibodyGen.experiments import her2_preferences as preferences  # noqa: E402
from smallAntibodyGen.experiments.her2_runtime import (  # noqa: E402
    digest_document, require, save_json, sha256)

TRASTUZUMAB_H1 = "GFNIKDTY"
TRASTUZUMAB_H2 = "IYPTNGYT"


def downloaded_index(raw_root, repository_root):
    """Relative path -> pinned (sha256, bytes) from BOTH untracked retrieval records."""
    entries = {}
    for entry in data.read_downloaded_manifest(raw_root)["files"]:
        relative = Path(entry["local_path"]).relative_to(Path(raw_root).relative_to(repository_root))
        entries[relative.as_posix()] = (entry["sha256"], entry["bytes"], entry["url"])
    for entry in data.read_piggen_manifest(raw_root)["files"]:
        relative = Path(entry["path"]).relative_to(Path(raw_root).relative_to(repository_root))
        entries[relative.as_posix()] = (entry["sha256"], entry["bytes"], entry["url"])
    return entries


def cross_check_manifests(config, raw_root):
    """Every tracked hash must equal the untracked retrieval record's hash.

    The retrieval record is read and never rewritten: it is the evidence that the
    tracked manifest was transcribed rather than invented.
    """
    pinned = downloaded_index(raw_root, ROOT)
    report = {}
    for relative in config["source_manifests"]:
        document = prov.load_manifest_document(ROOT / relative)
        manifest = document.validated()
        problems = list(prov.verify_manifest_files(manifest, raw_root))
        for entry in manifest.files:
            if entry.relative_path not in pinned:
                problems.append(f"{entry.relative_path}: absent from the retrieval record")
                continue
            digest, size, _ = pinned[entry.relative_path]
            if digest != entry.sha256 or size != entry.size_bytes:
                problems.append(f"{entry.relative_path}: tracked manifest disagrees with retrieval record")
        report[manifest.dataset_name] = {
            "manifest_path": relative, "manifest_sha256": sha256(ROOT / relative),
            "release_version": manifest.release_version, "license": manifest.license,
            "files": len(manifest.files), "problems": problems,
            "retrieval_record_urls": {entry.relative_path: pinned[entry.relative_path][2]
                                      for entry in manifest.files if entry.relative_path in pinned},
        }
        require(not problems, f"{manifest.dataset_name} provenance problems: {problems}")
    return report


def absci_support_mask(frame):
    """Rows that are genuinely inside this benchmark's support, jointly.

    "Inside the support" means all of it at once: HCDR3 length 13, the ``SR``
    left anchor and the ``Y`` right anchor in the positions this scaffold fixes,
    a canonical 10-mer between them, and the trastuzumab H1/H2 the Buzz library
    holds constant. Cropping every 13-mer and calling the middle a core is the
    thing this replaces -- it silently imports designs built on a different
    framework into a benchmark that fixes that framework.

    Returns ``(mask, notes)``; ``notes`` records whether the H1/H2 condition could
    be evaluated at all, because a file without those columns cannot support the
    joint claim and must not be scored as though it did.
    """
    h3 = frame["HCDR3"]
    inside = ((h3.str.len() == 13) & h3.str[:2].eq("SR") & h3.str[12:13].eq("Y")
              & h3.str[2:12].apply(lambda c: len(c) == data.CORE_LENGTH
                                   and set(c) <= set(data.CANONICAL)))
    notes = {"length_13": True, "sr_left_anchor": True, "y_right_anchor": True,
             "canonical_core": True}
    if "HCDR1" in frame.columns and "HCDR2" in frame.columns:
        inside = inside & frame.HCDR1.eq(TRASTUZUMAB_H1) & frame.HCDR2.eq(TRASTUZUMAB_H2)
        notes["fixed_h1_h2"] = True
    else:
        notes["fixed_h1_h2"] = False
        notes["h1_h2_note"] = ("this file carries no HCDR1/HCDR2 columns, so the fixed-framework "
                               "condition is unverified and the count below is an upper bound")
    return inside.to_numpy(), notes


def absci_diagnostic(raw_root, library_cores, train_cores):
    """Support-mismatch numbers for the AbSci release. Diagnostic only, never a headline."""
    report = {}
    for name in ("zero-shot-binders.csv", "spr-controls.csv"):
        frame = pd.read_csv(raw_root / "absci" / name, dtype=str, keep_default_na=False)
        h3 = frame["HCDR3"]
        lengths = h3.str.len()
        inside, notes = absci_support_mask(frame)
        supported = frame[inside]
        cores = supported.HCDR3.str[2:12]
        entry = {
            # Full-source counts, kept separate from the in-support counts so the
            # difference between "the release is large" and "the release overlaps
            # this task" stays visible.
            "rows": int(len(frame)), "unique_hcdr3": int(h3.nunique()),
            "length_counts": {str(k): int(v) for k, v in lengths.value_counts().items()},
            "length_13_rows": int((lengths == 13).sum()),
            "anchor_counts": {str(k): int(v) for k, v in
                              h3[lengths == 13].str[:2].value_counts().items()},
            "in_support_rows": int(inside.sum()),
            "in_support_conditions": notes,
            "in_support_unique_cores": int(cores.nunique()),
            "in_support_fraction": float(inside.mean()) if len(frame) else None,
            "cores_inside_buzz_library": int(cores.isin(library_cores).sum()),
            "cores_inside_training_split": int(cores.isin(train_cores).sum()),
            "overlap_note": ("overlap is computed over in-support rows only; a design outside the "
                             "support is not cropped into this benchmark and then compared"),
        }
        if "HCDR1" in frame.columns:
            entry["fixed_h1_h2_rows"] = int(((frame.HCDR1 == TRASTUZUMAB_H1)
                                             & (frame.HCDR2 == TRASTUZUMAB_H2)).sum())
        if "Binder" in frame.columns:
            entry["binder_counts"] = {str(k): int(v) for k, v in frame.Binder.value_counts().items()}
        report[name] = entry
    return report


def run(config_path, output):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    require(config["schema_version"] == "her2-posttrain/1", "Unsupported protocol schema")
    raw_root = ROOT / config["raw_root"]
    expected = config["expected_counts"]
    document = {"schema_version": "her2-audit/1", "config_sha256": sha256(config_path),
                "config_digest": digest_document(config), "raw_root": config["raw_root"],
                "reserved_test_labels_read": False, "assay_outcomes_read": False}

    print("1/6 verifying pinned downloads", flush=True)
    problems = data.verify_all_sources(raw_root, ROOT, config["source_manifests"])
    require(not problems, f"Pinned source files failed verification: {problems}")
    document["downloaded_files_verified"] = True
    document["manifests"] = cross_check_manifests(config, raw_root)

    print("2/6 auditing published splits", flush=True)
    train = data.load_split(raw_root, "train")
    val = data.load_split(raw_root, "val")
    test_seq = data.test_sequences(raw_root)
    require(len(test_seq) == expected["test_rows"], "Test split row count changed")
    document["splits"] = {"train": data.audit_split(train, "train"),
                          "val": data.audit_split(val, "val"),
                          "test": {"rows": int(len(test_seq)), "unique_cores": int(test_seq.nunique()),
                                   "labels_read": False}}
    sets = {"train": set(train.seq), "val": set(val.seq), "test": set(test_seq)}
    document["cross_split_overlap"] = {
        "train_val": len(sets["train"] & sets["val"]),
        "train_test": len(sets["train"] & sets["test"]),
        "val_test": len(sets["val"] & sets["test"]),
        "library_total": len(sets["train"] | sets["val"] | sets["test"])}
    require(document["cross_split_overlap"]["library_total"] == expected["library_total"],
            "Combined library size changed")
    require(max(document["cross_split_overlap"][k] for k in ("train_val", "train_test", "val_test")) == 0,
            "The published splits are no longer disjoint")

    print("3/6 support and scaffold", flush=True)
    train_index = data.encode_cores(train.seq)
    positives = train["class"] == data.POSITIVE_CLASS
    require(int(positives.sum()) == expected["train_high"], "Training high-bin count changed")
    high_index = train_index[positives.to_numpy()]
    counts = data.support_counts(high_index)
    document["support"] = {
        "training_high_rows": int(len(high_index)),
        "sites_with_all_20_residues": int((counts > 0).all(axis=1).sum()),
        "min_residue_count": int(counts.min()),
        "per_site_entropy_nats": data.site_entropy(high_index),
        "mean_pairwise_hamming_training_high": data.mean_pairwise_hamming(high_index),
        "note": ("all 20 canonical residues are available at all 10 positions; there is no "
                 "16-bit genotype restriction and no held-out-locus gate in this benchmark")}
    scaffold = data.load_scaffold(raw_root)
    document["scaffold"] = {
        "heavy_length": len(scaffold.heavy), "light_length": len(scaffold.light),
        "core_start": data.CORE_START, "prefix_length": len(scaffold.prefix),
        "prefix_tail": scaffold.prefix[-6:], "suffix": scaffold.suffix,
        "wild_type_core": data.WT_CORE,
        "context_note": ("the policy sees the start token plus VH[:98] only: framework 1-3 and "
                         "H1/H2, no FR4, no light chain, no antigen encoder")}

    print("4/6 proximity to the training set", flush=True)
    labels = positives.to_numpy().astype(np.float64)
    prior = float(labels.mean())
    document["proximity"] = {"training_positive_prior": prior}
    for name, frame_index in (("val", data.encode_cores(val.seq)),
                              ("test", data.encode_cores(test_seq))):
        lookup = data.nearest_training_labels(frame_index, train_index, labels, max_distance=2)
        strata, strata_counts = np.unique(lookup.strata(), return_counts=True)
        document["proximity"][name] = {
            "rows": int(frame_index.shape[0]),
            "min_train_hamming_counts": {str(k): int(v) for k, v in zip(strata, strata_counts)},
            "prior_fallback_rows": int((lookup.distance < 0).sum()),
            "mean_tied_neighbours": float(lookup.neighbour_count[lookup.distance >= 0].mean()),
            "labels_read": name != "test"}
    # Eligible preference populations, audited here so the counts the post-training
    # config asserts are evidence rather than numbers typed into a JSON file.
    train_population = preferences.build_population(train, "train")
    val_population = preferences.build_population(val, "val")
    document["preferences"] = {
        "train": train_population.document(), "val": val_population.document(),
        "reference_rows": int(train_population.chosen_index.shape[0]
                              + train_population.rejected_index.shape[0]),
        "rule": ("chosen = high bin, rejected = low bin at the SAME wild-type Hamming distance. "
                 "mid is retained for the three-class classifier and is never used as a rejected "
                 "example. High rows at a distance with no low partner are excluded and counted.")}
    document["proximity"]["interpretation"] = (
        "held-out ranking here is interpolation over a dense local library, not distant-family "
        "generalization; the strata above are the disclosure, not a fix")

    print("5/6 independent assay cohort (metadata only)", flush=True)
    library = sets["train"] | sets["val"] | sets["test"]
    cohort = data.assay_cohort(raw_root, scaffold, library)
    document["assay_cohort"] = cohort["counts"]
    document["assay_cohort"]["workbook_sheet"] = cohort["sheet_name"]
    document["assay_cohort"]["workbook_header_row"] = data.WORKBOOK_HEADER_ROW
    document["assay_cohort"]["workbook_headers"] = cohort["header"]
    document["assay_cohort"]["outcome_column"] = sorted(data.OUTCOME_COLUMNS)
    document["assay_cohort"]["assay_modality"] = (
        "SPR on a Carterra CMDP chip, which is what the workbook metadata declares. The workbook "
        "is the labelled source, so that is the modality this benchmark reports.")
    document["assay_cohort"]["assay_modality_discrepancy"] = (
        "the upstream repository README abstract says Biolayer Interferometry instead. The "
        "discrepancy is preserved rather than resolved; the workbook metadata is used because it "
        "is the file the outcomes come from.")
    # Expectations are checked and reported, not asserted: a drift here is a finding
    # about the release, and the audit should publish it rather than abort on it.
    document["assay_cohort"]["expectation_check"] = {
        key: {"expected": expected[key], "observed": cohort["counts"][observed],
              "matches": cohort["counts"][observed] == expected[key]}
        for key, observed in (("assay_workbook_rows", "workbook_rows"),
                              ("assay_same_scaffold_rows", "same_scaffold_rows"),
                              ("assay_design_method_rows", "design_method_rows"),
                              ("assay_primary_unique_cores", "primary_unique_cores"))}

    print("6/6 AbSci support mismatch", flush=True)
    document["absci"] = absci_diagnostic(raw_root, library, sets["train"])
    document["absci_role"] = ("diagnostic only: variable-length designs on a different context "
                              "are not cropped into this benchmark's support and called the same task")

    document["exposure"] = {
        "piggen_her2_paired_finetuning_scanned": False,
        "piggen_her2_unpaired_heavy_archive_scanned": False,
        "archive_to_checkpoint_lineage_reconciled": False,
        "prior_scan_note": ("an earlier paired-corpus scan in this repository queried the CR9114 "
                            "landscape, not HER2. It says nothing about this benchmark, and "
                            "reporting it as a HER2 scan was wrong."),
        "statement": ("HER2 pretraining exposure is UNSCANNED and UNRESOLVED, for both the paired "
                      "fine-tuning corpus and the unpaired heavy archive. No clean-exposure claim "
                      "is made, conditional on nothing. The matched random-initialization arm is "
                      "exposure-free in its initialization only, which is a statement about where "
                      "its weights came from and not about the supervised data both arms see.")}
    document["disclosure"] = (
        "Labels have been read for aggregate integrity checks, and two example rows per published "
        "split were printed, during the 2026-09-18 audit. That access is disclosed rather than "
        "described as sealed from the start. What is true is narrower and checkable: no "
        "outcome-based model selection has occurred, no final evaluation has been run, and this "
        "run read neither the reserved test labels nor the assay outcomes.")
    save_json(Path(output) / "audit.json", document)
    print(f"audit written to {Path(output) / 'audit.json'}", flush=True)
    return document


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/her2_posttrain.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/her2_posttrain_20260918/audit")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    run(args.config, args.output)
