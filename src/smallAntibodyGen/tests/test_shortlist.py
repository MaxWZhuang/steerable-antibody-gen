"""Selection does not see labels; geometry and calibration are independently checked."""
import itertools

import numpy as np
import pandas as pd
import pytest

from smallAntibodyGen.experiments.shortlist import calibrate_admission, portfolio_metrics, select_shortlist, sequence_diversity


def candidates():
    return pd.DataFrame({"genotype": [f"{i:016b}" for i in (0, 1, 3, 7, 15, 31, 63, 255)], "score": -np.arange(8.)})


def test_admission_and_deterministic_max_min_selection():
    frame = candidates()
    assert select_shortlist(frame, 2) == frame.genotype[:2].tolist()
    assert select_shortlist(frame, 2, 2) == [f"{0:016b}", f"{7:016b}"]
    assert select_shortlist(frame, 2, 4) == [f"{0:016b}", f"{255:016b}"]
    assert select_shortlist(frame.sample(frac=1, random_state=10), 4, 2) == select_shortlist(frame, 4, 2)


@pytest.mark.parametrize("change", ["labels", "duplicate", "short_id", "nan", "k", "multiplier"])
def test_rejects_leakage_and_malformed_inputs(change):
    frame = candidates()
    k, multiplier = 2, 2
    if change == "labels": frame["mean"] = 9.
    if change == "duplicate": frame.loc[1, "genotype"] = frame.loc[0, "genotype"]
    if change == "short_id": frame.loc[0, "genotype"] = "0"
    if change == "nan": frame.loc[0, "score"] = np.nan
    if change == "k": k = 9
    if change == "multiplier": multiplier = True
    with pytest.raises(ValueError): select_shortlist(frame, k, multiplier)


def test_tied_scores_order_by_identity():
    frame = candidates().assign(score=0.)
    assert select_shortlist(frame.iloc[::-1], 2) == frame.genotype[:2].tolist()


def test_distances_match_independent_pair_enumeration_and_components_are_transitive():
    ids = [f"{i:016b}" for i in (0, 1, 3, 65535)]
    values = [sum(a != b for a, b in zip(x, y)) for x, y in itertools.combinations(ids, 2)]
    metrics = sequence_diversity(ids)
    assert metrics["mean_hamming"] == pytest.approx(np.mean(values))
    assert metrics["minimum_hamming"] == 1
    assert metrics["hamming_le_one_components"] == 2
    assert metrics["largest_component"] == 3
    assert sequence_diversity([ids[0], ids[0]])["identity_collision_pair_fraction"] == 1.


def training():
    frame = candidates().drop(columns="score")
    return frame.assign(split="train", mean=np.linspace(10., 8., 8), effective_sem=.1, block=0, calibration_cohort=0)


def config(tolerance):
    return {"quality_threshold": 9., "budgets": [2, 4], "admission_multipliers": [1, 2], "affinity_loss_tolerance": tolerance}


def test_training_calibration_falls_back_when_quality_drops_and_can_expand_when_allowed():
    frame = training()
    scores = candidates().set_index("genotype").score
    assert calibrate_admission(frame, scores, config(0.))["selected_multiplier"] == 1
    assert calibrate_admission(frame, scores, config(2.))["selected_multiplier"] == 2


@pytest.mark.parametrize("split", ["development", "test"])
def test_calibration_rejects_held_out_labels(split):
    with pytest.raises(ValueError, match="training"):
        calibrate_admission(training().assign(split=split), candidates().set_index("genotype").score, config(0.))


def test_evaluation_joins_selected_identities_and_checks_quality_subset():
    frame = training().assign(split="development")
    selected = [frame.genotype[0], frame.genotype[7]]
    result = portfolio_metrics(frame.iloc[::-1], selected, 9.)
    assert result["mean_affinity"] == 9.
    assert result["quality_qualified_count"] == 1
    assert result["quality_qualified_diversity"] is None
    with pytest.raises(ValueError): portfolio_metrics(frame.assign(split="test"), selected, 9.)
