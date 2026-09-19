"""Audit of the pinned Absci release: what its KD cells are, and what they are not.

Role, fixed before anything is counted: **diagnostic only.** This is a different
antibody context with variable-length HCDR3s. It is not an interchangeable larger
cohort, and the 152 finite-KD Buzz SPR measurements cannot be topped up by
pooling rows from it. The only sequences this module will hand back as cores are
those that already match the benchmark's fixed support exactly -- ``SR`` + ten
canonical residues + ``Y`` -- so there is no function here that crops a 12- or
15-mer into a 10-mer and no path by which the generator's support expands.

What the audit measures, per file, from the bytes:

* every KD cell's class -- finite, blank, censored (``<``/``>``/``<=``/``>=``),
  ``N.B.``, ``I.C.``, ``N/A``, zero-width-space contaminated, or unsupported --
  with counts. Nothing censored or unquantified becomes a number.
* the explicit ``Binder`` column, counted as **true / false / unknown**, inside
  every subset that is reported. A blank KD cell is not a negative: it is a cell
  with no number in it, and the only thing in this release that says a design did
  not bind is the boolean column saying so.
* HCDR3 lengths, uniqueness and duplication -- reported twice, because they are
  two different facts. Repeated HCDR3s are not repeated antibodies: the full
  (HCDR1, HCDR2, HCDR3) context is counted separately where those columns exist.
* CDR-context compatibility with the benchmark's fixed scaffold, and whether it
  is **verified** or an upper bound: verification needs HCDR1 *and* HCDR2 columns
  equal to trastuzumab's. That is compatibility of the CDR context and **not**
  proof of the whole VH/VL scaffold sequence, which these files do not contain.
  The zero-shot file has neither column, so its compatible count is an upper
  bound and says so.
* overlap between the files by HCDR3, because they describe overlapping
  measurements: counts across files are **not** additive;
* overlap with the Buzz splits by core, for the compatible rows only. Test
  *sequences* are read; no test label is.

Attribution: the supplied license adds a fourth clause absent from the SPDX
template -- any reference to or publication of these data must be attributed to
**Absci Corporation (2023)**.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .her2_data import CANONICAL, CORE_LENGTH
from .her2_runtime import require, sha256

AUDIT_SCHEMA = "her2-absci-audit/1"
ATTRIBUTION = "Absci Corporation (2023)"

#: Trastuzumab's IMGT CDRs as the Absci release states them in its own README.
TRASTUZUMAB_HCDR1 = "GFNIKDTY"
TRASTUZUMAB_HCDR2 = "IYPTNGYT"
TRASTUZUMAB_HCDR3 = "SRWGGDGFYAMDY"
#: The benchmark's fixed support: SR + ten editable residues + Y.
ANCHOR_LEFT = "SR"
ANCHOR_RIGHT = "Y"
SUPPORTED_HCDR3_LENGTH = len(ANCHOR_LEFT) + CORE_LENGTH + len(ANCHOR_RIGHT)

#: The three pinned data files, by the relative path the tracked manifest uses.
FILES = {
    "zero_shot_binders": "absci/zero-shot-binders.csv",
    "spr_controls": "absci/spr-controls.csv",
    "functionality_supplement": "absci/functionality-developability-cross_reactivity-data.csv",
}
PRIMARY_KD_COLUMN = "KD (nM)"
HCDR3_COLUMN = "HCDR3"

ZERO_WIDTH_SPACE = "​"

KD_FINITE = "finite_positive"
KD_BLANK = "blank"
KD_CENSORED = "censored"
KD_NON_BINDING = "non_binding"
KD_BINDING_UNQUANTIFIED = "binding_unquantified"
KD_NOT_AVAILABLE = "not_available"
KD_UNSUPPORTED = "unsupported"

_CENSOR_PREFIXES = ("<", ">", "≤", "≥", "<=", ">=")


def _normalize(text):
    """Upper case, whitespace and zero-width spaces removed. Both sides normalized."""
    return "".join(str(text).replace(ZERO_WIDTH_SPACE, "").split()).upper()


_NON_BINDING = frozenset(_normalize(t) for t in ("N.B.", "NB", "N.B", "NO BINDING", "NON-BINDER"))
_UNQUANTIFIED = frozenset(_normalize(t) for t in ("I.C.", "IC", "I.C"))
_NOT_AVAILABLE = frozenset(_normalize(t) for t in ("N/A", "NA", "N.A.", "NOT AVAILABLE", "-"))


def classify_kd(value):
    """``(class, kd_nanomolar or None)`` for one cell. Nothing is ever invented.

    A censored cell (``<0.1``) records a bound, not a measurement, so it returns
    no number. ``N.B.`` is a non-binding observation, ``I.C.`` is binding that was
    not quantified, and ``N/A`` is an absence -- three different facts, and none of
    them is a KD. An unrecognized cell is reported unsupported rather than guessed.
    """
    raw = "" if value is None else str(value)
    contaminated = ZERO_WIDTH_SPACE in raw
    text = raw.replace(ZERO_WIDTH_SPACE, "").strip()
    if not text:
        return (KD_BLANK, None, contaminated)
    upper = _normalize(text)
    if upper in _NON_BINDING:
        return (KD_NON_BINDING, None, contaminated)
    if upper in _UNQUANTIFIED:
        return (KD_BINDING_UNQUANTIFIED, None, contaminated)
    if upper in _NOT_AVAILABLE:
        return (KD_NOT_AVAILABLE, None, contaminated)
    if text.startswith(_CENSOR_PREFIXES):
        return (KD_CENSORED, None, contaminated)
    try:
        number = float(text)
    except ValueError:
        return (KD_UNSUPPORTED, None, contaminated)
    if not np.isfinite(number) or number <= 0:
        return (KD_UNSUPPORTED, None, contaminated)
    return (KD_FINITE, number, contaminated)


def classify_column(series):
    """Class counts for one KD column, plus the finite values in row order."""
    classes, values, contaminated = {}, [], 0
    for cell in series:
        name, number, dirty = classify_kd(cell)
        classes[name] = classes.get(name, 0) + 1
        contaminated += int(dirty)
        values.append(number)
    return {"classes": dict(sorted(classes.items())),
            "finite_positive": int(classes.get(KD_FINITE, 0)),
            "zero_width_space_cells": contaminated,
            "conversion_policy": ("censored, non-binding, unquantified and absent cells carry no "
                                  "number and are never converted into one"),
            "precision_note": ("a finite cell is a measured quantitative affinity carrying its own "
                               "experimental error. It is treated as a measurement, not as an "
                               "exact or noiseless value")}, values


def supported_core(hcdr3):
    """The ten-residue core of an HCDR3 that is ALREADY in support, else ``None``.

    "In support" means exactly ``SR`` + ten canonical residues + ``Y``. A 12-mer or
    a 15-mer returns ``None``: cropping one into a 10-mer would silently redefine
    the task, so this function has no branch that does it and no caller can ask it
    to. This is the only route from an Absci row to a benchmark core.
    """
    text = "" if hcdr3 is None else str(hcdr3).strip().upper()
    if len(text) != SUPPORTED_HCDR3_LENGTH:
        return None
    if not text.startswith(ANCHOR_LEFT) or not text.endswith(ANCHOR_RIGHT):
        return None
    core = text[len(ANCHOR_LEFT):len(ANCHOR_LEFT) + CORE_LENGTH]
    if set(core) - set(CANONICAL):
        return None
    return core


#: What the boolean column may say. Anything else -- blank included -- is
#: ``unknown``, and unknown is never folded into either class.
BINDER_TRUE = frozenset({"true", "t", "yes", "y", "1"})
BINDER_FALSE = frozenset({"false", "f", "no", "n", "0"})


def classify_binder(value):
    """``true`` / ``false`` / ``unknown`` for one explicit binder cell.

    There is no branch here that looks at a KD. A blank KD means the release
    published no number for that design; whether it bound is a different column's
    business, and inferring "negative" from "no number" would turn an absence of
    data into a labelled negative.
    """
    text = "" if value is None else str(value).replace(ZERO_WIDTH_SPACE, "").strip().lower()
    if text in BINDER_TRUE:
        return "true"
    if text in BINDER_FALSE:
        return "false"
    return "unknown"


def _flag_counts(records):
    counts = {"true": 0, "false": 0, "unknown": 0}
    for record in records:
        counts[record.get("binder", "unknown")] += 1
    return counts


def _length_histogram(series):
    lengths = [len(str(value).strip()) for value in series]
    unique, counts = np.unique(np.asarray(lengths, dtype=np.int64), return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(unique, counts)}


def verify_pinned_files(root, raw_root, manifest_path):
    """Re-hash the three files against the TRACKED manifest before reading them."""
    from ..benchmarks import provenance as prov
    manifest = prov.load_manifest_document(Path(root) / manifest_path).validated()
    problems = list(prov.verify_manifest_files(manifest, Path(raw_root)))
    require(not problems, f"Pinned Absci files failed verification: {problems}")
    return {"manifest": str(manifest_path), "manifest_sha256": sha256(Path(root) / manifest_path),
            "files_verified": len(manifest.files),
            "license_note": ("the supplied Clear BSD text adds a fourth clause: reference or "
                             "publication of these data must be attributed to "
                             f"{ATTRIBUTION}")}


def audit_file(path, *, name, hcdr3_column=HCDR3_COLUMN, kd_column=PRIMARY_KD_COLUMN,
               binder_column=None, library_cores=None):
    """One pinned file: KD classes, lengths, support compatibility, duplication."""
    path = Path(path)
    require(path.is_file(), f"Pinned Absci file is missing: {path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    require(hcdr3_column in frame.columns, f"{path.name} has no {hcdr3_column} column")
    kd_report = (classify_column(frame[kd_column])[0] if kd_column in frame.columns
                 else {"classes": {}, "finite_positive": 0, "note": f"no {kd_column} column"})
    cores = [supported_core(value) for value in frame[hcdr3_column]]
    in_support = np.array([core is not None for core in cores])
    has_h1 = "HCDR1" in frame.columns
    has_h2 = "HCDR2" in frame.columns
    cdr_context_verified = bool(has_h1 and has_h2)
    if cdr_context_verified:
        framework = ((frame["HCDR1"].str.strip() == TRASTUZUMAB_HCDR1)
                     & (frame["HCDR2"].str.strip() == TRASTUZUMAB_HCDR2)).to_numpy()
    else:
        framework = np.ones(len(frame), dtype=bool)
    compatible_mask = in_support & framework
    compatible = [core for core, keep in zip(cores, compatible_mask) if keep]
    cells = (list(frame[kd_column]) if kd_column in frame.columns else [""] * len(frame))
    has_binder_column = binder_column is not None and binder_column in frame.columns
    binder_cells = (list(frame[binder_column]) if has_binder_column else [None] * len(frame))
    binder_flags = [classify_binder(cell) if has_binder_column else "unknown"
                    for cell in binder_cells]
    compatible_records = []
    for core, cell, flag, keep in zip(cores, cells, binder_flags, compatible_mask):
        if not keep:
            continue
        kd_class, kd_value, _ = classify_kd(cell)
        compatible_records.append({"core": core, "kd_class": kd_class, "kd_nanomolar": kd_value,
                                   "binder": flag})
    document = {
        "name": name, "path": str(path), "sha256": sha256(path),
        "bytes": int(path.stat().st_size), "rows": int(len(frame)),
        "columns": list(frame.columns),
        "kd": kd_report,
        "hcdr3": {"unique": int(frame[hcdr3_column].nunique()),
                  "duplicate_rows": int(len(frame) - frame[hcdr3_column].nunique()),
                  "length_histogram": _length_histogram(frame[hcdr3_column]),
                  "cdr_context": _cdr_context_duplication(frame, hcdr3_column),
                  "duplicate_note": ("a repeated HCDR3 is not a repeated antibody. Where HCDR1 "
                                     "and HCDR2 are available, the full CDR-context duplication "
                                     "is counted beside it and the two numbers differ.")},
        "support": {
            "supported_hcdr3_length": SUPPORTED_HCDR3_LENGTH,
            "rule": f"{ANCHOR_LEFT} + {CORE_LENGTH} canonical residues + {ANCHOR_RIGHT}",
            "length_and_anchor_compatible_rows": int(in_support.sum()),
            "length_and_anchor_compatible_unique_cores":
                len({core for core, keep in zip(cores, in_support) if keep}),
            "compatible_rows": int(compatible_mask.sum()),
            "compatible_unique_cores": len(set(compatible)),
            "compatible_kd_classes": _class_counts(compatible_records),
            "compatible_binder_flags": _flag_counts(compatible_records),
            "cdr_context_verified": cdr_context_verified,
            "scaffold_proven": False,
            "scaffold_note": ("'compatible' here means the CDR context matches: the HCDR3 is in "
                              "this benchmark's support and, where the columns exist, HCDR1 and "
                              "HCDR2 equal trastuzumab's. These files carry no VH/VL sequence, so "
                              "nothing here proves the rest of the scaffold is the same."),
            "binder_flag_note": ("binder true/false/unknown comes from the explicit boolean "
                                 "column. A blank KD is a missing number, not a negative."),
            "hcdr1_column_present": has_h1, "hcdr2_column_present": has_h2,
            "cropping": "not performed; an out-of-support HCDR3 yields no core"},
    }
    if not cdr_context_verified:
        document["support"]["upper_bound"] = True
        document["support"]["reason"] = (
            "this file carries no HCDR1/HCDR2 columns, so CDR-context identity with the fixed "
            "scaffold cannot be established. The compatible count is an UPPER BOUND on "
            "compatibility, not a count of usable rows.")
    else:
        document["support"].update(
            upper_bound=False,
            fixed_framework_rows=int(framework.sum()),
            fixed_hcdr1=TRASTUZUMAB_HCDR1, fixed_hcdr2=TRASTUZUMAB_HCDR2)
    document["compatible_cores"] = sorted(set(compatible))
    if has_binder_column:
        counts = _flag_counts([{"binder": flag} for flag in binder_flags])
        document["binder_flag"] = {
            "column": binder_column, "true": counts["true"], "false": counts["false"],
            "unknown": counts["unknown"], "other": counts["unknown"],
            "discrimination_possible": bool(counts["true"] > 0 and counts["false"] > 0),
            "source": "the explicit boolean column, not the KD cells"}
    else:
        document["binder_flag"] = {
            "column": None, "true": None, "false": None, "unknown": int(len(frame)),
            "finite_kd_rows": int(kd_report.get("finite_positive", 0)),
            "discrimination_possible": False,
            "note": ("this file is published as a binder set and may be described as one. What it "
                     "does not carry is an explicit Binder column, so no row's flag is KNOWN here "
                     "and none is inferred from a KD cell. With no explicit negatives there is no "
                     "labelled contrast in this file, and no discrimination statistic -- AUROC or "
                     "average precision -- is computed from it: a ranking metric over labels the "
                     "file does not carry would describe the assumption, not these data.")}
    if library_cores is not None:
        document["library_overlap"] = _library_overlap(compatible_records, library_cores)
    return document


def _cdr_context_duplication(frame, hcdr3_column):
    """Duplication over the full (HCDR1, HCDR2, HCDR3) context where it is available.

    The controls file has 1,829 unique HCDR3s in 1,855 rows, and zero duplicated
    CDR contexts: the repeats are the same HCDR3 on different HCDR1/HCDR2. Calling
    those 26 rows "duplicate antibodies" would be wrong in both directions.
    """
    columns = [name for name in ("HCDR1", "HCDR2", hcdr3_column) if name in frame.columns]
    if len(columns) < 2:
        return {"columns": columns, "unique": None, "duplicate_rows": None,
                "note": ("no HCDR1/HCDR2 columns, so CDR-context duplication is not computable "
                         "here and is reported as unavailable rather than as zero")}
    stripped = frame[columns].apply(lambda column: column.str.strip())
    unique = int(len(stripped.drop_duplicates()))
    return {"columns": columns, "unique": unique,
            "duplicate_rows": int(len(frame) - unique),
            "note": "duplication over the full CDR context, not over HCDR3 alone"}


def _class_counts(records):
    counts = {}
    for record in records:
        counts[record["kd_class"]] = counts.get(record["kd_class"], 0) + 1
    return dict(sorted(counts.items()))


def _library_overlap(compatible_records, library_cores):
    """Overlap of the compatible cores with each Buzz split. Sequences only.

    The finite-KD count is reported separately from the non-binding one, because
    they are different observations and pooling them is how an availability
    problem disappears into a total.
    """
    unique = {record["core"] for record in compatible_records}
    overlap = {}
    for split, catalog in sorted(library_cores.items()):
        shared = sorted(unique & set(catalog))
        overlap[split] = {"unique_cores_in_split": len(shared), "cores": shared}
    union = set().union(*[set(catalog) for catalog in library_cores.values()]) \
        if library_cores else set()
    independent = sorted(unique - union)
    finite = sorted({record["core"] for record in compatible_records
                     if record["kd_class"] == KD_FINITE})
    independent_records = [record for record in compatible_records
                           if record["core"] in set(independent)]
    overlap["independent_of_every_split"] = independent
    overlap["independent_finite_kd_cores"] = sorted(set(independent) & set(finite))
    overlap["finite_kd_compatible_cores"] = len(finite)
    overlap["independent_rows"] = len(independent_records)
    overlap["independent_kd_classes"] = _class_counts(independent_records)
    overlap["independent_binder_flags"] = _flag_counts(independent_records)
    overlap["note"] = ("exact core matches against the published Buzz splits. Test SEQUENCES are "
                       "read for this comparison; no test label is.")
    overlap["counting_note"] = ("finite-KD rows and explicit non-binders are counted apart and "
                                "never pooled into one 'measured' total. Rows whose binder flag "
                                "is unknown are counted as unknown, not as either class.")
    return overlap


def cross_file_overlap(reports):
    """How the files overlap by HCDR3. This is why their counts are not additive."""
    sequences = {}
    for name, report in reports.items():
        frame = pd.read_csv(report["path"], dtype=str, keep_default_na=False)
        sequences[name] = set(frame[HCDR3_COLUMN].str.strip())
    names = sorted(sequences)
    pairs = {}
    for left in names:
        for right in names:
            if left < right:
                shared = sequences[left] & sequences[right]
                pairs[f"{left}|{right}"] = {
                    "shared_unique_hcdr3": len(shared),
                    "left_fraction": len(shared) / max(1, len(sequences[left])),
                    "right_fraction": len(shared) / max(1, len(sequences[right]))}
    return {"unique_hcdr3_per_file": {name: len(values) for name, values in sequences.items()},
            "pairwise": pairs,
            "additive": False,
            "note": ("the same designs are characterized in more than one file, so adding row "
                     "counts across files would count measurements twice")}


def audit(root, raw_root, *, manifest_path="specs/benchmarks/absci_denovo_her2.json",
          library_cores=None):
    """The whole audit: verify the pinned bytes, then report what they contain."""
    root, raw_root = Path(root), Path(raw_root)
    provenance = verify_pinned_files(root, raw_root, manifest_path)
    reports = {
        "zero_shot_binders": audit_file(raw_root / FILES["zero_shot_binders"],
                                        name="zero_shot_binders", library_cores=library_cores),
        "spr_controls": audit_file(raw_root / FILES["spr_controls"], name="spr_controls",
                                   binder_column="Binder", library_cores=library_cores),
        "functionality_supplement": audit_file(raw_root / FILES["functionality_supplement"],
                                               name="functionality_supplement",
                                               library_cores=library_cores),
    }
    controls = reports["spr_controls"]
    document = {
        "schema_version": AUDIT_SCHEMA,
        "attribution": ATTRIBUTION,
        "role": {"in_this_study": "diagnostic only",
                 "primary_cohort": False,
                 "merged_cohort": None,
                 "why": ("variable-length HCDR3s on a different context. Pooling them with the "
                         "152 finite-KD Buzz SPR rows would not enlarge that cohort, it would "
                         "replace it with a different measurement."),
                 "cropping": "never performed"},
        "provenance": provenance,
        "files": reports,
        "cross_file": cross_file_overlap({name: report for name, report in reports.items()}),
        "supplement_note": ("the functionality/developability file repeats characterization of "
                            "candidates that already appear in the binder file; it is not an "
                            "additional independent primary cohort"),
        "measurement_formats": ("the supplement reports Fab and mAb KD with standard deviations "
                                "and cross-reactivity as 'mean +/- SD' or NB. Those formats are "
                                "preserved as read; none is collapsed into a single KD."),
    }
    if library_cores is not None:
        overlap = controls.get("library_overlap", {})
        document["independent_compatible_controls"] = {
            "unique_cores": len(overlap.get("independent_of_every_split", [])),
            "finite_kd_unique_cores": len(overlap.get("independent_finite_kd_cores", [])),
            "compatible_kd_classes": controls["support"]["compatible_kd_classes"],
            "compatible_binder_flags": controls["support"]["compatible_binder_flags"],
            "independent_kd_classes": overlap.get("independent_kd_classes"),
            "independent_binder_flags": overlap.get("independent_binder_flags"),
            "note": ("CDR-context-compatible control cores that appear in no Buzz split. Finite-KD "
                     "rows and explicitly flagged non-binders are counted apart and are never "
                     "pooled into one 'measured' total; a blank KD is a missing number and is not "
                     "read as a negative, which is why the binder flags are counted separately.")}
    return document
