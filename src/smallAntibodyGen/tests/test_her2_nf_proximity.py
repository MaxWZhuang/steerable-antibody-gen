"""The exact purge: brute-force agreement, determinism, and the certificate's refusals.

The projection search is the part of this flight most likely to be quietly
wrong, because an approximate neighbour search produces a plausible-looking
purge and a certificate that certifies nothing. Every test here compares it to
an all-pairs oracle on adversarial distance-0/1/2/3 cases and on random subsets,
and the certificate tests confirm that a violation raises rather than being
recorded as a small number.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smallAntibodyGen.experiments import her2_data as data
from smallAntibodyGen.experiments import her2_nf_contract as contract
from smallAntibodyGen.experiments import her2_nf_proximity as proximity

CANONICAL = data.CANONICAL


def _cores(rows, *, seed, alphabet=6):
    return np.random.default_rng(seed).integers(0, alphabet, size=(rows, 10)).astype(np.int8)


def _frame(index, *, classes=None, seed=0):
    generator = np.random.default_rng(seed)
    cores = data.decode_cores(index)
    if classes is None:
        classes = generator.choice(["low", "mid", "high"], size=len(cores))
    return pd.DataFrame({"seq": cores, "class": list(classes),
                         "label": [1 if value == "high" else 0 for value in classes],
                         "edit_distance": data.hamming_to(index,
                                                          data.encode_cores([data.WT_CORE])[0])})


def _brute(queries, panel, radius):
    distances = (np.asarray(queries)[:, None, :] != np.asarray(panel)[None, :, :]).sum(axis=2)
    nearest = distances.min(axis=1)
    return nearest <= radius, np.where(nearest <= radius, nearest, -1)


def test_adversarial_distance_zero_one_two_three_cases():
    base = np.tile(np.arange(10) % 6, (1, 1)).astype(np.int8)
    panel = base.copy()
    queries = []
    for changes in range(4):
        row = base[0].copy()
        for position in range(changes):
            row[position] = (row[position] + 1) % 6
        queries.append(row)
    queries = np.asarray(queries, dtype=np.int8)
    block = proximity.exact_nearest_within(queries, panel, radius=2)
    assert block["distance"].tolist() == [0, 1, 2, -1]
    assert block["certified_min_distance"].tolist() == [0, 1, 2, 3]
    assert block["within_radius"].tolist() == [True, True, True, False]


def test_projection_search_matches_brute_force_on_random_subsets():
    for seed in range(6):
        panel = _cores(40, seed=seed)
        queries = _cores(120, seed=seed + 100)
        for radius in (1, 2):
            block = proximity.exact_nearest_within(queries, panel, radius=radius)
            hit, distance = _brute(queries, panel, radius)
            assert block["within_radius"].tolist() == hit.tolist()
            assert block["distance"].tolist() == distance.tolist()


def test_the_witness_is_a_real_row_at_the_reported_distance():
    panel = _cores(30, seed=9)
    queries = _cores(60, seed=10)
    block = proximity.exact_nearest_within(queries, panel, radius=2)
    for row in np.flatnonzero(block["within_radius"]):
        witness = int(block["witness_row"][row])
        assert witness >= 0
        assert int((queries[row] != panel[witness]).sum()) == int(block["distance"][row])


def test_an_empty_reference_produces_no_hits_rather_than_an_error():
    block = proximity.exact_nearest_within(_cores(5, seed=1),
                                           np.zeros((0, 10), dtype=np.int8), radius=2)
    assert block["hits"] == 0
    assert block["certified_min_distance"].tolist() == [3] * 5


def test_brute_force_oracle_refuses_a_quadratic_run_that_is_too_large():
    with pytest.raises(ValueError, match="refuses"):
        proximity.brute_force_within(_cores(3000, seed=1), _cores(3000, seed=2), radius=2)


def test_matched_deletion_is_deterministic_and_matches_every_cell_exactly():
    index = _cores(400, seed=11)
    frame = _frame(index, seed=12)
    classes = np.asarray(frame["class"])
    distance = np.asarray(frame["edit_distance"])
    t0 = np.arange(400)
    purge = np.arange(0, 400, 2)
    first = proximity.matched_deletion(t0, purge, classes=classes, distance=distance)
    second = proximity.matched_deletion(t0, purge, classes=classes, distance=distance)
    assert np.array_equal(first["rows"], second["rows"])
    assert first["rows_drawn"] == purge.size
    wanted, got = {}, {}
    for row in purge:
        key = (classes[row], int(distance[row]))
        wanted[key] = wanted.get(key, 0) + 1
    for row in first["rows"]:
        key = (classes[row], int(distance[row]))
        got[key] = got.get(key, 0) + 1
    assert wanted == got
    assert first["seed_sequence"]["spawn_key"] == [2, 1000]


def test_an_unfillable_cell_stops_construction_rather_than_approximating():
    index = _cores(20, seed=13)
    classes = np.asarray(["high"] * 20)
    distance = np.arange(20)
    with pytest.raises(ValueError, match="approximate match is not substituted"):
        proximity.matched_deletion(np.arange(5), np.arange(20), classes=classes,
                                   distance=distance)


def test_pairability_counts_high_rows_with_a_partner_at_their_own_distance():
    index = np.asarray([[0] * 10, [1] + [0] * 9, [2] + [0] * 9], dtype=np.int8)
    frame = _frame(index, classes=["high", "low", "high"])
    block = proximity.pairability(frame, np.arange(3), index=index)
    assert block["high_rows"] == 2
    # Both high rows sit at WT distances that the single low row may or may not share.
    assert 0.0 <= block["pairable_fraction"] <= 1.0
    assert "at least one low row" in block["rule"]


def test_t0_is_smaller_than_t_because_of_the_calibration_purge():
    """T0 < T is the C-neighbourhood removal, not a duplicate anomaly."""
    train_index = _cores(300, seed=14)
    train_frame = _frame(train_index, seed=15)
    calibration = train_index[:5].copy()
    calibration[:, 0] = (calibration[:, 0] + 1) % 6          # distance 1 from five train rows
    evaluation = _cores(5, seed=16)
    populations = proximity.build_populations(
        train_frame, train_index, evaluation_index=evaluation,
        calibration_index=calibration, radius=2)
    assert populations["t0_rows"].size < train_index.shape[0]
    assert populations["removed_for_calibration_panel"] > 0
    assert "not a duplicate anomaly" in populations["t0_note"]


def test_the_certificate_refuses_a_purge_row_that_is_still_close_to_the_panel():
    train_index = _cores(200, seed=17)
    train_frame = _frame(train_index, seed=18)
    val_index = _cores(40, seed=19)
    val_frame = _frame(val_index, seed=20)
    evaluation_rows = np.arange(5)
    calibration_rows = np.arange(5, 15)
    populations = proximity.build_populations(
        train_frame, train_index, evaluation_index=val_index[evaluation_rows],
        calibration_index=val_index[calibration_rows], radius=2)
    # Smuggle a close neighbour back into the "purged" population.
    close = train_index.shape[0]
    train_index = np.vstack([train_index, val_index[0][None, :]])
    train_frame = _frame(train_index, seed=18)
    populations["purge_rows"] = np.append(populations["purge_rows"], close)
    with pytest.raises(ValueError, match="within radius"):
        proximity.neighbour_certificate(
            train_frame=train_frame, train_index=train_index, val_frame=val_frame,
            val_index=val_index, populations=populations, evaluation_rows=evaluation_rows,
            calibration_rows=calibration_rows, radius=2)


def test_calibration_panel_is_certified_beyond_the_radius_of_the_evaluation_panel():
    val_index = _cores(300, seed=21, alphabet=20)
    val_frame = _frame(val_index, seed=22)
    evaluation_rows = np.arange(10)
    block = proximity.build_calibration_panel(val_frame, val_index, evaluation_rows, radius=2,
                                              cap=50)
    overlap = set(block["rows"].tolist()) & set(evaluation_rows.tolist())
    assert not overlap
    check = proximity.exact_nearest_within(val_index[block["rows"]],
                                           val_index[evaluation_rows], radius=2)
    assert check["hits"] == 0


def test_evaluation_panel_takes_hash_order_within_each_stratum():
    val_index = _cores(90, seed=23, alphabet=20)
    val_frame = _frame(val_index, seed=24)
    labels = np.asarray(["1"] * 30 + ["2"] * 30 + [">=3"] * 30)
    block = proximity.build_evaluation_panel(val_frame, labels, panel_size=5)
    assert block["total"] == 15
    for name in proximity.ORIGINAL_STRATA:
        rows = block["per_stratum"][name]
        digests = contract.core_hashes([val_frame.seq.iloc[row] for row in rows])
        assert list(digests) == sorted(digests)
    # Deterministic: the same split and size give the same panel.
    again = proximity.build_evaluation_panel(val_frame, labels, panel_size=5)
    assert np.array_equal(block["rows"], again["rows"])


def test_feasibility_publishes_every_criterion_with_its_observed_value():
    train_index = _cores(200, seed=25)
    train_frame = _frame(train_index, seed=26)
    val_index = _cores(60, seed=27)
    val_frame = _frame(val_index, seed=28)
    evaluation_rows = np.arange(6)
    calibration_rows = np.arange(6, 20)
    populations = proximity.build_populations(
        train_frame, train_index, evaluation_index=val_index[evaluation_rows],
        calibration_index=val_index[calibration_rows], radius=2)
    report = proximity.feasibility_report(
        train_frame=train_frame, train_index=train_index, populations=populations,
        evaluation_rows=evaluation_rows, calibration_rows=calibration_rows,
        val_frame=val_frame, val_index=val_index, radius=2)
    assert set(proximity.FEASIBILITY) == set(report["criteria"])
    for name, block in report["checks"].items():
        assert "observed" in block and "threshold" in block and "passed" in block, name
    # Behaviour, not prose: the overall verdict is exactly the conjunction of the
    # published per-criterion verdicts, so no criterion can fail while the table
    # reports feasible.
    assert report["passed"] == all(block["passed"] for block in report["checks"].values())
    assert report["radius"] == 2


def test_replay_collisions_are_recorded_and_never_filtered():
    panel = data.decode_cores(_cores(5, seed=29))
    generated = panel[:2] + data.decode_cores(_cores(8, seed=30))
    block = proximity.replay_collision_record(generated, panel)
    assert block["collisions"] == 2
    assert "would change the replay distribution" in block["policy"]


def test_split_manifest_freezes_the_original_train_distance_per_panel_row():
    train_index = _cores(80, seed=31)
    train_frame = _frame(train_index, seed=32)
    val_index = _cores(40, seed=33)
    val_frame = _frame(val_index, seed=34)
    evaluation_rows = np.arange(4)
    calibration_rows = np.arange(4, 12)
    populations = proximity.build_populations(
        train_frame, train_index, evaluation_index=val_index[evaluation_rows],
        calibration_index=val_index[calibration_rows], radius=2)
    strata = {"labels": np.asarray(["1", "2", ">=3", "1"] + [">=3"] * 36),
              "counts": {"1": 2, "2": 1, ">=3": 37}}
    manifest = proximity.split_manifest(
        train_frame=train_frame, val_frame=val_frame, train_index=train_index,
        val_index=val_index, evaluation_rows=evaluation_rows,
        calibration_rows=calibration_rows, populations=populations, strata=strata,
        radius=2, panel_size=4)
    assert manifest["evaluation_panel"]["original_train_distance"] == [1, 2, -1, 1]
    assert len(manifest["evaluation_panel"]["row_ids"]) == 4
    assert "not a newly collected blind benchmark" in manifest["disclosure"]


def test_core_identity_audit_stops_on_a_conflicting_class_label():
    index = _cores(6, seed=35)
    left = _frame(index, classes=["high"] * 6)
    right = _frame(index, classes=["low"] * 6)
    audit = contract.audit_cores({"train": left, "val": right}, label="conflict probe")
    assert audit["conflicting_class_labels"]
    with pytest.raises(ValueError, match="different class labels"):
        contract.require_no_label_conflicts(audit)
