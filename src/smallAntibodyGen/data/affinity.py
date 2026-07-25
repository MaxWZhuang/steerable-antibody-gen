"""The single home of the strong-binder decision tree and the affinity-family map.

Why this module exists
----------------------
Before it, the "what counts as a strong binder" rules lived in TWO places that
had to be kept identical by hand:

- the PRODUCER, ``infer_is_strong_binder`` in ``scripts/prepare_antibody_antigen.py``,
  which writes ``is_strong_binder`` into the processed JSONL, and
- the READER, ``OASSequenceDataset._infer_is_strong_binder`` in
  ``data/MLMCollator.py``, the fallback for JSONL written before that field
  existed.

Two hand-mirrored copies of a threshold table is a high-drift shape: a change to
one silently reclassifies rows relative to the other, and nothing fails. Both
sites now delegate to :func:`is_strong_binder_from_fields`.

What is shared and what is NOT
------------------------------
This module owns the **decision tree and the thresholds** -- the part that
drifts. It deliberately does NOT own **coercion**, because the two sites read
different worlds and must keep their own:

- the producer coerces with pandas-aware helpers (``clean_text`` / ``safe_float``
  / ``parse_binder_label``), which parse numeric strings and treat ``pd.NA`` as
  missing;
- the reader coerces ``json.loads`` output, where a numeric field is already a
  number and a string measurement is not expected.

So every entry point here takes ALREADY-COERCED inputs. That is what makes the
consolidation behavior-preserving rather than merely intended to be.

The constants
-------------
``KD_MOLAR_NANOMOLAR_BOUNDARY`` (1e-3) is the magnitude disambiguator: a real
antibody KD expressed in molar is always well below 1 mM, so any value at or
above 1e-3 must already be in nanomolar. ``KD_STRONG_THRESHOLD_MOLAR`` (1e-9),
``KD_STRONG_THRESHOLD_NANOMOLAR`` (1.0) and ``NEG_LOG_KD_STRONG_THRESHOLD``
(9.0) are the same 1 nM cut expressed in the three unit systems the corpus uses.
"""
from __future__ import annotations

import math
from typing import Any, Dict

# 1 nM expressed three ways, plus the molar/nanomolar magnitude disambiguator.
KD_MOLAR_NANOMOLAR_BOUNDARY = 1e-3
KD_STRONG_THRESHOLD_MOLAR = 1e-9
KD_STRONG_THRESHOLD_NANOMOLAR = 1.0
NEG_LOG_KD_STRONG_THRESHOLD = 9.0

# Floor applied before -log10 so a zero/denormal KD cannot produce -inf.
KD_LOG_FLOOR_MOLAR = 1e-12

AFFINITY_FAMILY_IDS = {
    "unknown": 0,
    "binary_binding": 1,
    # Present in the id table but never produced by `affinity_family_for_type`.
    # Kept so stored ids stay stable if a graded-strength family is introduced.
    "ordered_strength": 2,
    "ranking_regression": 3,
    "mutation_effect": 4,
}


def normalize_affinity_type(value: object) -> str:
    """Return a stable lowercase affinity-type key ("" when missing)."""
    return str(value or "").strip().lower()


def affinity_family_for_type(affinity_type: object) -> str:
    """
    Map heterogeneous affinity types into conservative supervision families.

    The families -- not the raw assay names -- are what training conditions on,
    because the corpus mixes booleans, KD in two unit systems, -log KD, ddG and
    ELISA ratios, and only some of those are comparable to each other.
    """
    normalized = normalize_affinity_type(affinity_type)
    if normalized == "bool":
        return "binary_binding"
    if normalized in {"kd", "-log kd"}:
        return "ranking_regression"
    if normalized in {"ddg", "elisa_mut_to_wt_ratio"}:
        return "mutation_effect"
    return "unknown"


def marker_text(value: object) -> str:
    """
    Normalize a stored qualitative marker to comparable text (reader-side).

    Follows ``clean_text`` in ``scripts/prepare_antibody_antigen.py``, which wrote
    these values: a missing marker becomes "", everything else is stringified and
    trimmed. Stringifying FIRST is what lets a caller distinguish "absent" from a
    present-but-falsy measurement such as ``0.0``.

    Only ``None`` and float NaN count as missing here, where ``clean_text``
    defers to ``pd.isna`` and so also catches pandas/numpy sentinels. Those
    cannot appear on the reader side -- it reads ``json.loads`` output -- and the
    difference fails safe (a sentinel stringifies to a non-"h" value, i.e. NOT
    strong) rather than inventing a strong binder.
    """
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def finite_float(value: object) -> float | None:
    """Reader-side coercion: a finite float, rejecting bools and non-numerics."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def is_strong_binder_from_fields(
    *,
    affinity_type: str,
    marker: str,
    measurement: float | None,
    binder_label: int | None,
) -> bool:
    """
    The strong-binder decision tree, expressed once, over pre-coerced inputs.

    Policy (unchanged from both former copies):

    - explicit boolean binder rows with label 1,
    - fuzzy assay rows whose qualitative marker is "h",
    - kd rows with KD <= 1 nM (molar values <= 1e-9, or nanomolar values <= 1.0),
    - -log KD rows with processed measurement >= 9.

    Args:
        affinity_type:
            Already lowercased/trimmed assay type.
        marker:
            The fuzzy branch's already-resolved qualitative marker: the cleaned
            ``processed_measurement`` when present, else the cleaned
            ``affinity_raw``, else "". Resolving it at the call site (rather than
            passing both fields) is what preserves each site's precedence rule,
            in which a present-but-falsy ``0.0`` must NOT fall through to
            ``affinity_raw``.
        measurement:
            Already-coerced finite measurement, or ``None``.
        binder_label:
            Already-parsed 0/1 boolean-assay label, or ``None``.
    """
    if affinity_type == "bool":
        return binder_label == 1

    if affinity_type == "fuzzy":
        return marker.lower() == "h"

    if measurement is None:
        return False

    if affinity_type == "kd":
        # KD may be stored in molar (e.g. 1e-9) or already in nanomolar (e.g.
        # 1.0); a strong binder is KD <= 1 nM. Disambiguate by magnitude.
        if measurement <= 0:
            return False
        if measurement >= KD_MOLAR_NANOMOLAR_BOUNDARY:
            return measurement <= KD_STRONG_THRESHOLD_NANOMOLAR
        return measurement <= KD_STRONG_THRESHOLD_MOLAR

    if affinity_type == "-log kd":
        return measurement >= NEG_LOG_KD_STRONG_THRESHOLD

    return False


def infer_is_strong_binder(record: Dict[str, Any]) -> bool:
    """
    Reader-side entry point: the strong-binder flag for one decoded JSONL row.

    A stored ``is_strong_binder`` wins outright; everything else is the legacy
    fallback for JSONL written before that field existed.
    """
    if "is_strong_binder" in record:
        return bool(record.get("is_strong_binder"))

    affinity_type = normalize_affinity_type(record.get("affinity_type"))
    marker = marker_text(record.get("processed_measurement_raw")) or marker_text(
        record.get("affinity_raw")
    )
    binder_label = record.get("binder_label")
    return is_strong_binder_from_fields(
        affinity_type=affinity_type,
        marker=marker,
        measurement=finite_float(record.get("processed_measurement_float")),
        binder_label=binder_label if binder_label in (0, 1) else None,
    )


def base_affinity_strength_score(record: Dict[str, Any]) -> float | None:
    """
    Convert one record into the scalar used by graded affinity supervision.

    Binary rows return their 0/1 label. Ranking/regression rows return a
    ``-log10(KD in molar)`` so that raw-molar and nanomolar encodings of the SAME
    measurement produce the SAME score -- without that normalization a
    nanomolar-encoded strong binder (1.0 nM -> score 0.0) would never clear the
    ``>= 9.0`` threshold. Every other family returns ``None``: a ddG and an ELISA
    ratio are not on a shared scale and must not be pooled into one target.
    """
    affinity_type = normalize_affinity_type(record.get("affinity_type"))
    family = affinity_family_for_type(affinity_type)
    binder_label = record.get("binder_label")

    if family == "binary_binding":
        if binder_label in (0, 1):
            return float(binder_label)
        return None

    if family != "ranking_regression":
        return None

    value = finite_float(record.get("processed_measurement_float"))
    if value is None:
        return None

    if affinity_type == "kd":
        if value <= 0:
            return None
        if value >= KD_MOLAR_NANOMOLAR_BOUNDARY:
            value = value * 1e-9
        return -math.log10(max(value, KD_LOG_FLOOR_MOLAR))
    return value


def annotate_affinity(record: Dict[str, Any]) -> dict[str, object]:
    """
    Derive the conservative affinity supervision fields for one record.
    """
    affinity_type_normalized = normalize_affinity_type(record.get("affinity_type"))
    affinity_family = affinity_family_for_type(affinity_type_normalized)
    strength_score = base_affinity_strength_score(record)
    strength_label: int | None = None

    if affinity_family == "binary_binding":
        if record.get("binder_label") in (0, 1):
            strength_label = int(record["binder_label"])
    elif affinity_family == "ranking_regression" and strength_score is not None:
        if strength_score >= NEG_LOG_KD_STRONG_THRESHOLD:
            strength_label = 1

    return {
        "affinity_type_normalized": affinity_type_normalized or None,
        "affinity_family": affinity_family,
        "affinity_strength_score": strength_score,
        "affinity_strength_label": strength_label,
    }
