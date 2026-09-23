"""The exact neighbourhood purge, its matched control, and the certificate.

The original random split is dense: most validation rows sit one mutation away
from a training row. Selecting the already-far rows would measure a different
population, so the intervention is built on proximity instead -- a frozen
evaluation panel ``E``, a calibration panel ``C`` certified away from ``E``, and
two training populations with identical ``(class, distance-to-WT)`` composition:

* ``T0`` = ``T`` minus everything within radius ``r`` of ``C``. Applied to both
  regimes, so ``C`` is separated from both and can monitor either.
* ``T_purge`` = ``T0`` minus everything within radius ``r`` of ``E``.
* ``T_match`` = a deterministic sample from ``T0`` matching ``T_purge``'s counts
  cell by cell. It may keep close neighbours of ``E``; that is the point.

``T0 < T`` is the ``C``-neighbourhood purge doing its job. It is not a duplicate
anomaly and there is no discrepancy to explain: the source splits carry unique
cores and no cross-split overlap, and the loader already refuses anything else.

The neighbour search is exact. Two ten-position cores are within distance ``d``
iff they agree on some projection that deletes ``d`` positions, so indexing every
mask of size ``r`` finds every pair within ``r`` and nothing else. Sweeping mask
sizes ``0, 1, ..., r`` in increasing order additionally makes the *first* hit the
exact distance. Beyond ``r`` the method certifies only a lower bound of ``r + 1``,
and that lower bound is stored under its own name -- ``certified_min_distance``
-- so it is never read as a measured nearest distance.
"""
from __future__ import annotations

import itertools

import numpy as np

from .her2_data import CORE_LENGTH, WT_CORE, encode_cores, hamming_to, pack_codes
from .her2_nf_contract import (NF_SCHEMA, core_hashes, hash_order, index_hashes)
from .her2_runtime import require

#: The primary challenge radius, and the predeclared fallback.
PRIMARY_RADIUS = 2
FALLBACK_RADIUS = 1

#: Panel sizes, largest first. The largest nested panel that passes is preferred.
PANEL_SIZES = (1000, 500, 250)

#: Deterministic cap on the calibration panel.
CALIBRATION_PANEL_ROWS = 3000

#: The matched-deletion stream. Recorded as a lineage, not as a bare integer.
MATCHED_DELETION_ENTROPY = 20260922
MATCHED_DELETION_SPAWN_KEY = (2, 1000)

#: Original proximity strata of a validation row, by exact distance to the
#: ORIGINAL training set. Frozen into the manifest at construction, because ``T``
#: itself changes under the purge and the label must not move with it.
ORIGINAL_STRATA = ("1", "2", ">=3")

#: Engineering feasibility thresholds. Proposed practicality criteria, not power
#: guarantees and not biological constants.
FEASIBILITY = {
    "min_retained_train_fraction": 0.50,
    "min_retained_high_rows": 50_000,
    "min_pairable_fraction": 0.95,
    "min_panel_high": 100,
    "min_panel_non_high": 100,
    "min_panel_mid_for_inference": 50,
    "min_panel_low_for_inference": 50,
    "min_formerly_close_fraction_of_panel": 0.50,
    "min_matched_control_retention": 0.50,
    "max_purge_retention": 0.0,
}


def masks_of_size(size):
    """Position tuples to DELETE. Size ``r`` retains ``10 - r`` positions."""
    return tuple(itertools.combinations(range(CORE_LENGTH), int(size)))


def mask_count(radius):
    return sum(len(masks_of_size(size)) for size in range(int(radius) + 1))


# ---------------------------------------------------------------------------
# the exact neighbour search
# ---------------------------------------------------------------------------

def exact_nearest_within(query_index, reference_index, *, radius):
    """Exact nearest distance up to ``radius``, with a retained witness.

    Masks are applied in increasing size and a query row leaves the pending set
    at its first hit, so the size at which it was found *is* its exact distance:
    a closer pair would have matched at a smaller mask, and a pair at distance
    exactly ``k`` agrees on all but those ``k`` sites and therefore matches the
    single mask deleting precisely them.

    Rows with no hit get ``distance = -1`` and ``certified_min_distance =
    radius + 1``. That is a certified lower bound, not the nearest distance.
    """
    queries = np.asarray(query_index)
    reference = np.asarray(reference_index)
    radius = int(radius)
    require(queries.ndim == 2 and queries.shape[1] == CORE_LENGTH, "Expected (N, 10) query cores")
    require(reference.ndim == 2 and reference.shape[1] == CORE_LENGTH,
            "Expected (M, 10) reference cores")
    require(0 <= radius <= 3, f"Unsupported radius {radius}")
    total = queries.shape[0]
    distance = np.full(total, -1, dtype=np.int64)
    witness = np.full(total, -1, dtype=np.int64)
    mask_used = np.full(total, -1, dtype=np.int64)
    pending = np.arange(total)
    mask_id = 0
    examined = 0
    if reference.shape[0] == 0:
        return _neighbour_record(queries, reference, radius, distance, witness, mask_used, 0, 0)
    for size in range(radius + 1):
        for positions in masks_of_size(size):
            if pending.size == 0:
                break
            reference_codes = pack_codes(reference, positions)
            order = np.argsort(reference_codes, kind="stable")
            ordered = reference_codes[order]
            probe = pack_codes(queries[pending], positions)
            left = np.searchsorted(ordered, probe, side="left")
            right = np.searchsorted(ordered, probe, side="right")
            found = right > left
            examined += int(pending.size)
            if found.any():
                rows = pending[found]
                partners = order[left[found]]
                observed = (queries[rows] != reference[partners]).sum(axis=1)
                require(bool((observed == size).all()),
                        f"A projection of size {size} matched a pair at Hamming distance "
                        f"{sorted(set(observed.tolist()))}; the packing is lossless, so this "
                        "would mean the mask table and the verification disagree")
                distance[rows] = size
                witness[rows] = partners
                mask_used[rows] = mask_id
                pending = pending[~found]
            mask_id += 1
    return _neighbour_record(queries, reference, radius, distance, witness, mask_used, mask_id,
                             examined)


def _neighbour_record(queries, reference, radius, distance, witness, mask_used, masks, examined):
    hit = distance >= 0
    return {"radius": int(radius), "query_rows": int(queries.shape[0]),
            "reference_rows": int(reference.shape[0]),
            "within_radius": hit, "distance": distance, "witness_row": witness,
            "witness_mask_id": mask_used,
            "certified_min_distance": np.where(hit, distance, int(radius) + 1),
            "hits": int(hit.sum()), "masks_examined": int(masks),
            "projection_probes": int(examined),
            "certified_note": ("for a row with no hit the method certifies distance >= radius + 1 "
                               "and does not recover the actual nearest distance. That bound is "
                               "stored as certified_min_distance and is never reported as an "
                               "exact distance.")}


def brute_force_within(query_index, reference_index, *, radius):
    """All-pairs reference implementation. Only for validation on small inputs."""
    queries = np.asarray(query_index)
    reference = np.asarray(reference_index)
    require(queries.shape[0] * max(reference.shape[0], 1) <= 4_000_000,
            "brute_force_within is a validation oracle and refuses a quadratic run this large")
    if reference.shape[0] == 0:
        return {"within_radius": np.zeros(queries.shape[0], dtype=bool),
                "distance": np.full(queries.shape[0], -1, dtype=np.int64)}
    distances = (queries[:, None, :] != reference[None, :, :]).sum(axis=2)
    nearest = distances.min(axis=1)
    return {"within_radius": nearest <= int(radius),
            "distance": np.where(nearest <= int(radius), nearest, -1).astype(np.int64)}


# ---------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------

def original_proximity_strata(val_index, train_index, *, radius=PRIMARY_RADIUS):
    """``"1"``/``"2"``/``">=3"`` per validation row, by exact distance to the ORIGINAL ``T``."""
    block = exact_nearest_within(val_index, train_index, radius=radius)
    distance = block["distance"]
    require(int((distance == 0).sum()) == 0,
            f"{int((distance == 0).sum())} validation core(s) are identical to a training core. "
            "The published splits are supposed to be overlap-removed; this stops construction "
            "rather than quietly assigning them a stratum.")
    labels = np.where(distance < 0, ">=3", distance.astype(str))
    counts = {name: int((labels == name).sum()) for name in ORIGINAL_STRATA}
    return {"labels": labels, "counts": counts, "radius": int(radius),
            "frozen_note": ("these labels describe distance to the ORIGINAL training set and are "
                            "frozen into the split manifest. T changes under the purge; this "
                            "label must not move with it.")}


def build_evaluation_panel(val_frame, strata, *, panel_size):
    """``E``: ``panel_size`` rows from each original proximity stratum, in hash order.

    Deterministic sequence hashes, never model outcomes: the panel is a function
    of the published split and the declared size, so it cannot be redrawn after
    seeing a result without that being visible as a different size.
    """
    cores = list(val_frame.seq)
    labels = np.asarray(strata)
    chosen = []
    per_stratum = {}
    for name in ORIGINAL_STRATA:
        rows = np.flatnonzero(labels == name)
        require(rows.size >= int(panel_size),
                f"stratum {name} holds {rows.size} rows and the panel asks for {panel_size}")
        ordered = rows[hash_order([cores[row] for row in rows])]
        take = ordered[: int(panel_size)]
        per_stratum[name] = take
        chosen.append(take)
    index = np.sort(np.concatenate(chosen))
    return {"rows": index, "per_stratum": {k: v.tolist() for k, v in per_stratum.items()},
            "panel_size_per_stratum": int(panel_size), "total": int(index.size),
            "order": "ascending sha256(core UTF-8) within each original stratum",
            "selection_inputs": "published split membership and the core strings only"}


def build_calibration_panel(val_frame, val_index, evaluation_rows, *, radius=PRIMARY_RADIUS,
                            cap=CALIBRATION_PANEL_ROWS):
    """``C``: the first hash-ordered ``V`` rows outside ``E`` certified beyond ``radius`` of ``E``."""
    cores = list(val_frame.seq)
    excluded = set(int(row) for row in np.asarray(evaluation_rows).tolist())
    candidates = np.asarray([row for row in range(len(cores)) if row not in excluded])
    ordered = candidates[hash_order([cores[row] for row in candidates])]
    panel = np.asarray(val_index)[np.asarray(evaluation_rows)]
    block = exact_nearest_within(np.asarray(val_index)[ordered], panel, radius=radius)
    eligible = ordered[~block["within_radius"]]
    require(eligible.size >= int(cap),
            f"only {eligible.size} validation rows sit beyond radius {radius} of E, and the "
            f"declared calibration cap is {cap}")
    rows = np.sort(eligible[: int(cap)])
    return {"rows": rows, "total": int(rows.size), "cap": int(cap), "radius": int(radius),
            "eligible_rows": int(eligible.size),
            "excluded_for_proximity": int(ordered.size - eligible.size),
            "order": "ascending sha256(core UTF-8) over V minus E",
            "separation": f"certified distance > {radius} from every E row"}


# ---------------------------------------------------------------------------
# the two training populations
# ---------------------------------------------------------------------------

def _cells(frame, index):
    classes = np.asarray(frame["class"])
    distance = hamming_to(np.asarray(index), encode_cores([WT_CORE])[0])
    return classes, distance


def build_populations(train_frame, train_index, *, evaluation_index, calibration_index,
                      radius=PRIMARY_RADIUS):
    """``T0``, ``T_purge`` and ``T_match`` with their witnesses and cell counts."""
    train_index = np.asarray(train_index)
    near_c = exact_nearest_within(train_index, np.asarray(calibration_index), radius=radius)
    t0_rows = np.flatnonzero(~near_c["within_radius"])
    near_e = exact_nearest_within(train_index[t0_rows], np.asarray(evaluation_index), radius=radius)
    purge_rows = t0_rows[~near_e["within_radius"]]
    classes, distance = _cells(train_frame, train_index)
    match_rows = matched_deletion(t0_rows, purge_rows, classes=classes, distance=distance)
    return {"schema_version": NF_SCHEMA, "record_kind": "training_populations",
            "radius": int(radius),
            "train_rows": int(train_index.shape[0]),
            "t0_rows": t0_rows, "purge_rows": purge_rows, "match_rows": match_rows["rows"],
            "removed_for_calibration_panel": int(near_c["hits"]),
            "removed_for_evaluation_panel": int(near_e["hits"]),
            "calibration_witnesses": {"witness_row": near_c["witness_row"],
                                      "distance": near_c["distance"]},
            "evaluation_witnesses": {"t0_row": t0_rows, "witness_row": near_e["witness_row"],
                                     "distance": near_e["distance"]},
            "matched_deletion": {k: v for k, v in match_rows.items() if k != "rows"},
            "t0_note": ("T0 removes the C neighbourhood from BOTH regimes, so C is separated from "
                        "both and can monitor either. T0 < T is that removal, not a duplicate "
                        "anomaly: the source rows are unique and non-overlapping by construction.")}


def matched_deletion(t0_rows, purge_rows, *, classes, distance,
                     entropy=MATCHED_DELETION_ENTROPY, spawn_key=MATCHED_DELETION_SPAWN_KEY):
    """Sample ``T_match`` from ``T0``, matching ``T_purge``'s ``(class, WT distance)`` counts.

    Deterministic: a recorded ``SeedSequence`` lineage, cells visited in sorted
    order, and candidates inside a cell ordered by their row index before the
    draw. It may retain close neighbours of ``E``, which is exactly what makes it
    the control for data quantity and coarse composition rather than for
    proximity.
    """
    sequence = np.random.SeedSequence(int(entropy), spawn_key=tuple(int(k) for k in spawn_key))
    generator = np.random.default_rng(sequence)
    t0_rows = np.asarray(t0_rows)
    purge_rows = np.asarray(purge_rows)
    wanted = {}
    for row in purge_rows:
        key = (str(classes[row]), int(distance[row]))
        wanted[key] = wanted.get(key, 0) + 1
    available = {}
    for row in t0_rows:
        key = (str(classes[row]), int(distance[row]))
        available.setdefault(key, []).append(int(row))
    picked, shortfall = [], {}
    for key in sorted(wanted):
        pool = np.asarray(sorted(available.get(key, [])), dtype=np.int64)
        count = int(wanted[key])
        if pool.size < count:
            shortfall[f"{key[0]}|{key[1]}"] = {"wanted": count, "available": int(pool.size)}
            picked.append(pool)
            continue
        chosen = generator.choice(pool.size, size=count, replace=False)
        picked.append(pool[np.sort(chosen)])
    require(not shortfall,
            f"the matched control cannot be filled in {sorted(shortfall)}; T0 holds fewer rows in "
            "those cells than T_purge does. Matching is what removes the quantity and composition "
            "confound, so an approximate match is not substituted.")
    rows = np.sort(np.concatenate(picked)) if picked else np.asarray([], dtype=np.int64)
    return {"rows": rows, "cells": len(wanted), "rows_drawn": int(rows.size),
            "seed_sequence": {"entropy": int(entropy), "spawn_key": [int(k) for k in spawn_key],
                              "generator": "numpy.random.default_rng(SeedSequence(...))"},
            "rule": ("per (class, distance-to-WT) cell, without replacement, cells in sorted order "
                     "and candidates in ascending row order before the draw"),
            "limitation": ("matching controls class and WT-distance composition. It does not "
                           "randomize every residue feature or library provenance.")}


# ---------------------------------------------------------------------------
# feasibility and the certificate
# ---------------------------------------------------------------------------

def pairability(frame, rows, *, index, positive_class="high", negative_class="low"):
    """Fraction of retained positives whose WT-distance cell holds a negative partner."""
    rows = np.asarray(rows)
    classes = np.asarray(frame["class"])[rows]
    distance = hamming_to(np.asarray(index)[rows], encode_cores([WT_CORE])[0])
    positives = classes == positive_class
    negative_distances = set(distance[classes == negative_class].tolist())
    pairable = np.asarray([int(value) in negative_distances for value in distance[positives]])
    return {"high_rows": int(positives.sum()), "pairable_high_rows": int(pairable.sum()),
            "pairable_fraction": float(pairable.mean()) if pairable.size else None,
            "unpairable_by_distance": {
                str(int(value)): int(count) for value, count in
                zip(*np.unique(distance[positives][~pairable], return_counts=True))},
            "rule": "a high row is pairable when its own WT distance has at least one low row"}


def panel_composition(frame, rows):
    classes = np.asarray(frame["class"])[np.asarray(rows)]
    counts = {name: int((classes == name).sum()) for name in ("low", "mid", "high")}
    counts["non_high"] = counts["low"] + counts["mid"]
    counts["rows"] = int(classes.size)
    return counts


def manipulation_strength(evaluation_index, *, train_index, purge_rows, match_rows,
                          radius=PRIMARY_RADIUS):
    """How many ``E`` rows lose a close neighbour under the purge and keep one under the control.

    The design identifies the intended intervention only if originally-close
    ``E`` rows keep a ``<= r`` neighbour in ``T_match`` and none does in
    ``T_purge``.
    """
    train_index = np.asarray(train_index)
    original = exact_nearest_within(evaluation_index, train_index, radius=radius)
    purged = exact_nearest_within(evaluation_index, train_index[np.asarray(purge_rows)],
                                  radius=radius)
    matched = exact_nearest_within(evaluation_index, train_index[np.asarray(match_rows)],
                                   radius=radius)
    close = original["within_radius"]
    return {"radius": int(radius),
            "panel_rows": int(np.asarray(evaluation_index).shape[0]),
            "originally_within_radius": int(close.sum()),
            "retained_in_matched_control": int(matched["within_radius"][close].sum()),
            "retained_in_purge": int(purged["within_radius"].sum()),
            "matched_control_retention": (float(matched["within_radius"][close].mean())
                                          if close.any() else None),
            "purge_retention": float(purged["within_radius"].mean()),
            "criterion": ("an engineering manipulation threshold: at least half of the originally "
                          "close panel rows keep a neighbour in the control and none does in the "
                          "purge. Not a biological constant.")}


def feasibility_report(*, train_frame, train_index, populations, evaluation_rows,
                       calibration_rows, val_frame, val_index, radius=PRIMARY_RADIUS,
                       strata_counts=None, thresholds=None):
    """Every declared criterion with its observed value and its verdict."""
    criteria = dict(thresholds or FEASIBILITY)
    purge_rows = np.asarray(populations["purge_rows"])
    match_rows = np.asarray(populations["match_rows"])
    train_rows = int(np.asarray(train_index).shape[0])
    classes = np.asarray(train_frame["class"])
    retained_high = int((classes[purge_rows] == "high").sum())
    pair_block = pairability(train_frame, purge_rows, index=train_index)
    evaluation_index = np.asarray(val_index)[np.asarray(evaluation_rows)]
    manipulation = manipulation_strength(evaluation_index, train_index=train_index,
                                         purge_rows=purge_rows, match_rows=match_rows,
                                         radius=radius)
    panel_e = panel_composition(val_frame, evaluation_rows)
    panel_c = panel_composition(val_frame, calibration_rows)
    formerly_close = None
    if strata_counts is not None:
        close_rows = int(strata_counts.get("1", 0) + strata_counts.get("2", 0))
        formerly_close = close_rows / float(panel_e["rows"]) if panel_e["rows"] else None
    checks = {
        "retained_train_fraction": _check(purge_rows.size / float(train_rows),
                                          criteria["min_retained_train_fraction"], ">="),
        "retained_high_rows": _check(retained_high, criteria["min_retained_high_rows"], ">="),
        "pairable_fraction": _check(pair_block["pairable_fraction"],
                                    criteria["min_pairable_fraction"], ">="),
        "evaluation_high": _check(panel_e["high"], criteria["min_panel_high"], ">="),
        "evaluation_non_high": _check(panel_e["non_high"], criteria["min_panel_non_high"], ">="),
        "calibration_high": _check(panel_c["high"], criteria["min_panel_high"], ">="),
        "calibration_non_high": _check(panel_c["non_high"], criteria["min_panel_non_high"], ">="),
        "evaluation_mid_inferential": _check(panel_e["mid"],
                                             criteria["min_panel_mid_for_inference"], ">="),
        "evaluation_low_inferential": _check(panel_e["low"],
                                             criteria["min_panel_low_for_inference"], ">="),
        "matched_control_retention": _check(manipulation["matched_control_retention"],
                                            criteria["min_matched_control_retention"], ">="),
        "purge_retention": _check(manipulation["purge_retention"],
                                  criteria["max_purge_retention"], "<="),
    }
    if formerly_close is not None:
        checks["formerly_close_fraction_of_panel"] = _check(
            formerly_close, criteria["min_formerly_close_fraction_of_panel"], ">=")
    return {"schema_version": NF_SCHEMA, "record_kind": "proximity_feasibility",
            "radius": int(radius), "criteria": criteria, "checks": checks,
            "passed": all(block["passed"] for block in checks.values()),
            "counts": {"train_rows": train_rows, "t0_rows": int(np.asarray(
                populations["t0_rows"]).size), "purge_rows": int(purge_rows.size),
                "match_rows": int(match_rows.size), "retained_high_rows": retained_high},
            "pairability": pair_block, "manipulation": manipulation,
            "evaluation_panel": panel_e, "calibration_panel": panel_c,
            "status": ("proposed practicality thresholds, published BEFORE any panel revision. "
                       "They are not power guarantees. If radius 2 fails, the predeclared "
                       f"radius-{FALLBACK_RADIUS} design is a DIFFERENT challenge and is named as "
                       "one.")}


def _check(observed, threshold, comparison):
    if observed is None:
        return {"observed": None, "threshold": threshold, "comparison": comparison,
                "passed": False, "reason": "not computable on this population"}
    passed = observed >= threshold if comparison == ">=" else observed <= threshold
    return {"observed": float(observed), "threshold": float(threshold),
            "comparison": comparison, "passed": bool(passed)}


def neighbour_certificate(*, train_frame, train_index, val_frame, val_index, populations,
                          evaluation_rows, calibration_rows, radius=PRIMARY_RADIUS,
                          source_digests=None):
    """Zero-violation certificate: the boundaries, the witnesses, and the counts.

    The three prohibitions checked here are the three that would silently
    invalidate the challenge: a ``T_purge`` row near ``E``, a training row near
    ``C`` in either regime, and any overlap between the two panels.
    """
    train_index = np.asarray(train_index)
    val_index = np.asarray(val_index)
    evaluation_index = val_index[np.asarray(evaluation_rows)]
    calibration_index = val_index[np.asarray(calibration_rows)]
    purge_rows = np.asarray(populations["purge_rows"])
    match_rows = np.asarray(populations["match_rows"])

    purge_near_e = exact_nearest_within(train_index[purge_rows], evaluation_index, radius=radius)
    purge_near_c = exact_nearest_within(train_index[purge_rows], calibration_index, radius=radius)
    match_near_c = exact_nearest_within(train_index[match_rows], calibration_index, radius=radius)
    panel_overlap = sorted(set(index_hashes(evaluation_index).tolist())
                           & set(index_hashes(calibration_index).tolist()))
    violations = {
        "purge_rows_within_radius_of_E": int(purge_near_e["hits"]),
        "purge_rows_within_radius_of_C": int(purge_near_c["hits"]),
        "matched_rows_within_radius_of_C": int(match_near_c["hits"]),
        "panel_rows_shared_between_E_and_C": len(panel_overlap)}
    require(violations["purge_rows_within_radius_of_E"] == 0,
            f"{violations['purge_rows_within_radius_of_E']} purged-population row(s) remain within "
            f"radius {radius} of the evaluation panel. The purge is the intervention; a nonzero "
            "count here means the challenge does not test what it claims to.")
    require(violations["purge_rows_within_radius_of_C"] == 0
            and violations["matched_rows_within_radius_of_C"] == 0,
            "a training row sits within the declared radius of the calibration panel. C monitors "
            "both regimes and must be separated from both.")
    require(violations["panel_rows_shared_between_E_and_C"] == 0,
            "E and C share a row; the calibration panel would then select on evaluation rows")
    witnesses = _witness_table(train_index, populations, evaluation_index, radius)
    return {"schema_version": NF_SCHEMA, "record_kind": "neighbor_certificate",
            "radius": int(radius), "masks_per_radius": {str(size): len(masks_of_size(size))
                                                        for size in range(int(radius) + 1)},
            "projection_rule": ("two cores are within distance d iff some projection deleting d "
                                "positions matches. Mask identity is part of the key; the full "
                                "Hamming distance is verified on every hit and the witness row is "
                                "retained."),
            "violations": violations, "zero_violations": all(v == 0 for v in violations.values()),
            "exclusion_witnesses": witnesses,
            "population_counts": {
                "train": int(train_index.shape[0]),
                "t0": int(np.asarray(populations["t0_rows"]).size),
                "purge": int(purge_rows.size), "match": int(match_rows.size)},
            "class_counts": {
                "purge": _class_distance_counts(train_frame, train_index, purge_rows),
                "match": _class_distance_counts(train_frame, train_index, match_rows)},
            "panel_row_ids": {
                "evaluation": index_hashes(evaluation_index).tolist(),
                "calibration": index_hashes(calibration_index).tolist()},
            "original_test_excluded": ("no original test row enters any construction and no "
                                       "reserved label is read at any point"),
            "source_digests": dict(source_digests or {}),
            "certified_bound": ("a nonmatch certifies distance >= radius + 1. It does not recover "
                                "the exact nearest distance, and certified_min_distance is stored "
                                "separately from any exact distance.")}


def _witness_table(train_index, populations, evaluation_index, radius, limit=50):
    """A readable sample of the exclusions, with the exact distance that caused each."""
    removed = np.setdiff1d(np.asarray(populations["t0_rows"]), np.asarray(populations["purge_rows"]))
    if removed.size == 0:
        return {"removed_rows": 0, "sample": []}
    block = exact_nearest_within(train_index[removed], evaluation_index, radius=radius)
    order = removed[: int(limit)]
    return {"removed_rows": int(removed.size),
            "distance_histogram": {str(int(value)): int(count) for value, count in
                                   zip(*np.unique(block["distance"], return_counts=True))},
            "sample": [{"train_row": int(row), "distance": int(block["distance"][position]),
                        "panel_witness_row": int(block["witness_row"][position])}
                       for position, row in enumerate(order)]}


def _class_distance_counts(frame, index, rows):
    rows = np.asarray(rows)
    classes = np.asarray(frame["class"])[rows]
    distance = hamming_to(np.asarray(index)[rows], encode_cores([WT_CORE])[0])
    out = {}
    for name, value in zip(classes.tolist(), distance.tolist()):
        key = f"{name}|{int(value)}"
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def split_manifest(*, train_frame, val_frame, train_index, val_index, evaluation_rows,
                   calibration_rows, populations, strata, radius, panel_size, source_digests=None):
    """The immutable identity of the challenge split. Written before any policy sees ``E``."""
    evaluation_rows = np.asarray(evaluation_rows)
    calibration_rows = np.asarray(calibration_rows)
    labels = np.asarray(strata["labels"])
    return {"schema_version": NF_SCHEMA, "record_kind": "split_manifest",
            "radius": int(radius), "panel_size_per_stratum": int(panel_size),
            "original_counts": {"train": int(len(train_frame)), "val": int(len(val_frame))},
            "original_stratum_counts": dict(strata["counts"]),
            "evaluation_panel": {
                "rows": int(evaluation_rows.size),
                "row_ids": core_hashes([val_frame.seq.iloc[int(row)]
                                        for row in evaluation_rows]).tolist(),
                "composition": panel_composition(val_frame, evaluation_rows),
                "original_stratum_counts": {
                    name: int((labels[evaluation_rows] == name).sum())
                    for name in ORIGINAL_STRATA},
                "original_train_distance": [int(value) if value != ">=3" else -1
                                            for value in labels[evaluation_rows].tolist()]},
            "calibration_panel": {
                "rows": int(calibration_rows.size),
                "row_ids": core_hashes([val_frame.seq.iloc[int(row)]
                                        for row in calibration_rows]).tolist(),
                "composition": panel_composition(val_frame, calibration_rows)},
            "populations": {
                "t0": int(np.asarray(populations["t0_rows"]).size),
                "purge": int(np.asarray(populations["purge_rows"]).size),
                "match": int(np.asarray(populations["match_rows"]).size),
                "purge_row_ids_sha256": _rows_digest(train_frame, populations["purge_rows"]),
                "match_row_ids_sha256": _rows_digest(train_frame, populations["match_rows"])},
            "matched_deletion": populations["matched_deletion"],
            "source_digests": dict(source_digests or {}),
            "panels_are_not_training_populations": (
                "neither E nor C is a task-training population. E never enters challenge training "
                "or challenge selection; C is the challenge's monitoring and selection population."),
            "disclosure": ("this is a newly enforced adaptation separation on EXISTING data, not a "
                           "newly collected blind benchmark. Block-A calibration uses the "
                           "historical validation population, which contains E; that inherited "
                           "exploratory exposure is disclosed and prevents any claim that E was "
                           "globally unseen. Unknown pretraining overlap remains unresolved.")}


def _rows_digest(frame, rows):
    import hashlib
    digest = hashlib.sha256()
    for row in np.sort(np.asarray(rows)):
        digest.update(str(frame.seq.iloc[int(row)]).encode("utf-8"))
    return digest.hexdigest()


def replay_collision_record(generated_cores, evaluation_cores):
    """Generated replay draws that happen to equal panel rows. Recorded, never filtered.

    Filtering them would change ``P``'s replay distribution to manufacture a
    clean-looking overlap count. They are unlabelled samples, not supervised
    label exposure, and they are counted here instead.
    """
    panel = set(core_hashes(evaluation_cores).tolist())
    hits = [digest for digest in core_hashes(generated_cores).tolist() if digest in panel]
    return {"generated_rows": len(list(generated_cores)), "panel_rows": len(panel),
            "collisions": len(hits), "distinct_collisions": len(set(hits)),
            "policy": ("recorded and retained. A generated sequence that equals a panel row is an "
                       "unlabelled sample; removing it would change the replay distribution.")}
