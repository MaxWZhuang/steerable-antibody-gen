"""The finite-support functional baseline, its CLI and the DPLM hook binder.

Part 1 tests ``experiments/functional_baseline.py``: exact tilts, the hand-computed analytic
cases, the bounded DPO fit, SIR accounting, the stage separation (fit / allocate / evaluate), the
hashed frozen artifact, the label/provenance guards and the planted causal-learning experiment.
Everything is NumPy on CPU. What it establishes is that the machinery computes what it says on
known truths; it says nothing about any native model or about biology.

Part 2 tests ``bind_conditioned_dplm`` on a tiny, randomly initialised, locally constructed
``DPLMEncoder``/``ConditionedDPLM`` (CPU, no weights, no downloads, public AbLang tokens built by
the real serialisation helpers). That is a synthetic CPU qualification of the tap sites, layouts and
interventions, not a claim about pretrained weights, biology or the memory of a full native run.

No test asserts on source text.
"""
from __future__ import annotations

import copy
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from smallAntibodyGen.experiments import functional_baseline as fb
from smallAntibodyGen.experiments import her2_routes_interp as interp
from smallAntibodyGen.experiments import her2_routes_serial as serial
from smallAntibodyGen.models import dplm_policy as dplm
from smallAntibodyGen.models import her2_routes_hooks as hooks_lib
from smallAntibodyGen.tests import test_dplm_policy as dplm_fixtures

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "functional_baseline.py"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_cli():
    spec = importlib.util.spec_from_file_location("functional_baseline_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def label_free():
    return fb.SupervisionHistory(status="label_free", exposure_status="none")


def unknown_history():
    return fb.SupervisionHistory(status="unknown")


def documented(*entries):
    return fb.SupervisionHistory(status="documented", documented=tuple(entries))


def entry(*, count, target="T", assay="A", ids=None):
    return {"assay": assay, "target": target, "label_count": count, "label_ids": ids}


def explicit_space(names=("x1", "x2"), supervision=None):
    return fb.FeatureSpace(kind="explicit", names=tuple(names), schema={"columns": list(names)},
                           history=("explicit columns",),
                           supervision=supervision or label_free())


def problem_kwargs(**overrides):
    base = dict(
        target="T", assay="A",
        utility=fb.UtilitySpec(units="u", scale="linear", higher_is_better=True,
                               useful_threshold=1.0),
        grouping_rationale="every candidate is its own group",
        space=explicit_space(),
        prior=fb.PriorSpec(kind=fb.PRIOR_SCORE, semantics="scores", supervision=label_free()),
        train_ids=tuple(f"t{i}" for i in range(6)),
        train_features=[[0, 0], [1, 0], [0, 1], [1, 1], [2, 0], [0, 2]],
        train_utilities=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        train_log_prior=[0.0] * 6,
        train_groups=tuple(f"gt{i}" for i in range(6)),
        support_ids=tuple(f"s{i}" for i in range(4)),
        support_features=[[0.5, 0.5], [1, 2], [2, 1], [0, 0.1]],
        support_log_prior=[0.0, -0.5, -math.inf, 0.3],
        support_groups=tuple(f"gs{i}" for i in range(4)))
    base.update(overrides)
    return base


def tiny_problem(**overrides):
    return fb.FiniteSupportProblem(**problem_kwargs(**overrides))


def tiny_config(**overrides):
    base = dict(tau=1.0, seed=3, ridge_alpha=0.1, sir_proposal_draws=50, final_draws=20)
    base.update(overrides)
    return fb.BaselineConfig(**base)


TRUTH = {"s0": 1.0, "s1": 3.0, "s2": 2.0, "s3": 0.5}


@pytest.fixture
def frozen():
    return fb.fit_and_allocate(tiny_problem(), tiny_config())


def write_csv(path, header, rows):
    Path(path).write_text(header + "\n" + "\n".join(rows) + ("\n" if rows else ""),
                          encoding="utf-8")
    return Path(path)


CANDIDATE_HEADER = "candidate_id,sequence,group_id,split,prior_log_weight"
GOOD_ROWS = ["c1,AC,g1,train,0.0", "c2,AD,g1,train,-1.0", "c3,CA,g2,eval,-inf",
             "c4,CD,g2,eval,-0.5"]


def manifest_document(**overrides):
    document = {
        "target": "T", "assay": "A",
        "utility": {"units": "pKD", "scale": "log10", "higher_is_better": True,
                    "useful_threshold": 1.0},
        "grouping_rationale": "groups are clones",
        "features": {"kind": "sequence_onehot", "alphabet": "ACD", "length": 2,
                     "provenance": {"status": "label_free"}},
        "prior": {"kind": "score_normalized", "semantics": "scores",
                  "provenance": {"status": "label_free"}}}
    document.update(overrides)
    return document


# ---------------------------------------------------------------------------
# exact finite-support mathematics: the hand-computed cases
# ---------------------------------------------------------------------------

def test_case_1_and_2_oracle_tilts_and_the_regret_at_the_prior():
    tau = 0.37
    star = np.exp(fb.log_tilt(np.log([0.9, 0.1]), [0.0, tau * math.log(9.0)], tau))
    assert star == pytest.approx([0.5, 0.5], abs=1e-12)
    flat = np.exp(fb.log_tilt(np.log([0.9, 0.1]), [2.0, 2.0], tau))
    assert flat == pytest.approx([0.9, 0.1], abs=1e-12), "a constant utility must preserve p"
    p, u = np.log([0.5, 0.5]), [0.0, math.log(3.0)]
    assert np.exp(fb.log_tilt(p, u, 1.0)) == pytest.approx([0.25, 0.75], abs=1e-12)
    outcome = fb.oracle_regret(p, p, u, 1.0)
    assert outcome["regret"] == pytest.approx(math.log(2.0) - 0.5 * math.log(3.0), abs=1e-12)
    assert outcome["tau_kl_to_oracle"] == pytest.approx(outcome["regret"], abs=1e-12)


def test_case_3_a_positive_predicted_gap_beside_a_true_improvement_of_log_three_halves():
    p = np.log([0.5, 0.5])
    true_u, predicted_u = np.array([math.log(3.0), 0.0]), np.array([0.0, math.log(3.0)])
    log_hat = fb.log_tilt(p, predicted_u, 1.0)
    assert np.exp(log_hat) == pytest.approx([0.25, 0.75], abs=1e-12)
    gap = fb.kl_divergence(p, log_hat)
    assert gap > 0 and gap == pytest.approx(math.log(2.0) - 0.5 * math.log(3.0), abs=1e-12)
    improvement = fb.objective(p, p, true_u, 1.0) - fb.objective(log_hat, p, true_u, 1.0)
    assert improvement == pytest.approx(math.log(1.5), abs=1e-12)
    cross = (fb.expected_value(p, true_u - predicted_u)
             - fb.expected_value(log_hat, true_u - predicted_u))
    assert cross == pytest.approx(0.5 * math.log(3.0), abs=1e-12)
    assert improvement == pytest.approx(-gap + cross, abs=1e-12)


def test_case_4_sir_weights_count_occurrences_and_ess_is_not_the_collapsed_ess():
    law, info = fb.sir_weighted_distribution(
        np.log([0.5, 0.5]), np.log([0.75, 0.25]), [0.0, 0.0], 1.0, np.array([3, 1]))
    assert np.exp(law) == pytest.approx([0.5, 0.5], abs=1e-12)
    assert info["effective_sample_size"] == pytest.approx(3.0, abs=1e-12)
    collapsed_ess = 1.0 / float(np.sum(np.exp(law) ** 2))
    assert collapsed_ess == pytest.approx(2.0) != info["effective_sample_size"]
    assert info["distinct_candidates"] == 2 and info["proposal_draws"] == 4
    repeated, repeated_info = fb.sir_weighted_distribution(
        np.log([0.5, 0.5]), np.log([0.5, 0.5]), [0.0, 0.0], 1.0, np.array([4, 0]))
    assert repeated_info["effective_sample_size"] == pytest.approx(4.0, abs=1e-12)
    assert repeated_info["distinct_candidates"] == 1
    assert np.exp(repeated) == pytest.approx([1.0, 0.0])


def test_case_5_an_underflowed_probability_gives_a_finite_kl_not_a_fake_infinity():
    p = np.log([0.5, 0.5])
    log_hat = fb.log_tilt(p, [0.0, -1000.0], 1.0)
    assert np.exp(log_hat[1]) == 0.0 and np.isfinite(log_hat[1])
    kl = fb.kl_divergence(p, log_hat)
    assert math.isfinite(kl) and kl == pytest.approx(500.0 - math.log(2.0), abs=1e-9)
    assert fb.kl_divergence(log_hat, p) == pytest.approx(math.log(2.0), abs=1e-9)


def test_case_6_the_dpo_optimum_and_its_effect_on_two_priors():
    space = explicit_space(("x",))
    policy = fb.fit_dpo([[-1.0], [1.0]], [1], [0], space, beta=1.0, l2=1.0 / math.log(3.0))
    theta = float(policy.theta[0])
    assert theta == pytest.approx(math.log(3.0) / 2.0, abs=1e-8)
    diagnostics = policy.diagnostics
    assert diagnostics["converged"] and diagnostics["stop_reason"] == "gradient_tolerance"
    assert diagnostics["initial_loss"] == pytest.approx(math.log(2.0), abs=1e-12)
    assert diagnostics["final_loss"] == pytest.approx(
        math.log(4.0 / 3.0) + math.log(3.0) / 8.0, abs=1e-9)
    assert diagnostics["updates"] >= 2 and diagnostics["final_loss"] < diagnostics["initial_loss"]
    energy = policy.energy([[-1.0], [1.0]])
    uniform = np.exp(fb.normalize_log_weights(np.log([0.5, 0.5]) + energy))
    skewed = np.exp(fb.normalize_log_weights(np.log([0.75, 0.25]) + energy))
    assert uniform == pytest.approx([0.25, 0.75], abs=1e-8)
    assert skewed == pytest.approx([0.5, 0.5], abs=1e-8)


def test_case_7_a_monotone_reward_rescaling_keeps_the_ranking_but_changes_allocation_and_j():
    p = np.log([0.25] * 4)
    u = np.array([0.0, 0.5, 1.0, 1.5])
    for transformed in (4.0 * u, np.exp(2.0 * u)):
        assert fb.spearman(u, transformed)["value"] == pytest.approx(1.0)
        assert fb.concordance(u, transformed)["value"] == pytest.approx(1.0)
    base, steep = fb.log_tilt(p, u, 1.0), fb.log_tilt(p, 4.0 * u, 1.0)
    assert fb.total_variation(base, steep) > 0.3, "hand value is about 0.41"
    j_base, j_steep = fb.objective(base, p, u, 1.0), fb.objective(steep, p, u, 1.0)
    assert j_base == pytest.approx(fb.oracle_objective_closed_form(p, u, 1.0), abs=1e-12)
    assert j_base - j_steep > 0.3, "hand values: J(q_u) is about 0.90, J(q_4u) about 0.49"


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_case_8_the_oracle_regret_identity_holds_for_random_q_with_structural_zeros(seed):
    rng = np.random.default_rng(seed)
    for _ in range(30):
        prior = np.where(rng.random(7) < 0.3, -np.inf, rng.normal(size=7))
        prior[0] = 0.0
        q_log = np.where(np.isfinite(prior), rng.normal(size=7), -np.inf)
        q_log = q_log - fb.logsumexp(q_log)
        outcome = fb.oracle_regret(q_log, prior, rng.normal(size=7), 0.6)
        assert math.isfinite(outcome["regret"])
        assert outcome["regret"] == pytest.approx(outcome["tau_kl_to_oracle"], abs=1e-9)
        assert outcome["regret"] >= -1e-12, "no policy beats the oracle on J"


def test_a_q_with_mass_outside_the_reference_support_has_infinite_regret_on_both_sides():
    prior = np.array([math.log(0.5), math.log(0.5), -np.inf])
    q_log = np.log([0.2, 0.3, 0.5])
    outcome = fb.oracle_regret(q_log, prior, [0.0, 1.0, 5.0], 1.0)
    assert math.isinf(outcome["regret"]) and math.isinf(outcome["tau_kl_to_oracle"])
    assert fb.objective(q_log, prior, [0.0, 1.0, 5.0], 1.0) == -math.inf


def test_both_kl_directions_are_named_and_the_sparse_policy_shows_the_asymmetry():
    sparse = np.array([0.0, -np.inf])
    learned = np.log([0.25, 0.75])
    assert fb.total_variation(sparse, learned) == pytest.approx(0.75)
    assert fb.kl_divergence(sparse, learned) == pytest.approx(math.log(4.0))
    assert math.isinf(fb.kl_divergence(learned, sparse))


def test_support_is_decided_by_the_log_weight_so_minus_10000_is_not_a_structural_zero():
    log_p = fb.normalize_log_weights([0.0, -10000.0])
    assert np.exp(log_p[1]) == 0.0 and int((log_p > -np.inf).sum()) == 2
    brought_back = np.exp(fb.log_tilt([0.0, -10000.0], [0.0, 10000.0], 1.0))
    assert brought_back == pytest.approx([0.5, 0.5], abs=1e-12)
    walled = np.exp(fb.log_tilt([0.0, -np.inf], [0.0, 10000.0], 1.0))
    assert walled == pytest.approx([1.0, 0.0]), "an explicit -inf is a structural zero"


@pytest.mark.parametrize("bad", [[0.0, np.nan], [0.0, np.inf], [-np.inf, -np.inf]])
def test_nan_plus_inf_and_an_all_zero_support_are_refused(bad):
    with pytest.raises(ValueError):
        fb.normalize_log_weights(bad)


def test_constants_and_candidate_order_do_not_change_the_normalised_results():
    p, u = np.array([-0.3, -1.2, -np.inf, 0.4]), np.array([0.1, 0.9, 5.0, -0.2])
    reference = fb.log_tilt(p, u, 0.8)
    assert fb.log_tilt(p + 7.5, u - 3.25, 0.8) == pytest.approx(reference, abs=1e-12)
    order = np.array([3, 0, 2, 1])
    assert fb.log_tilt(p[order], u[order], 0.8) == pytest.approx(reference[order], abs=1e-12)


def test_the_analytic_checks_shipped_with_the_selftest_all_hold():
    results = fb.analytic_checks()
    assert len(results) == 7
    assert all(item["holds"] for item in results), [i for i in results if not i["holds"]]


# ---------------------------------------------------------------------------
# ranking metrics
# ---------------------------------------------------------------------------

def test_spearman_and_concordance_hand_values_and_undefined_cases_carry_reasons():
    assert fb.spearman([1, 2, 3, 4], [1, 2, 3, 4])["value"] == pytest.approx(1.0)
    assert fb.spearman([1, 2, 3, 4], [4, 3, 2, 1])["value"] == pytest.approx(-1.0)
    assert fb.concordance([1, 2, 3, 4], [4, 3, 2, 1])["value"] == 0.0
    ties = fb.spearman([1, 2, 3], [1, 1, 2])
    assert ties["value"] == pytest.approx(math.sqrt(3.0) / 2.0)
    excluded = fb.concordance([1, 2, 3], [1, 1, 2])
    assert excluded["value"] == 1.0 and excluded["comparable_pairs"] == 2
    half = fb.concordance([1, 1, 2], [1, 2, 3])
    assert half["value"] == pytest.approx(2.5 / 3.0) and half["score_tied_pairs"] == 1
    for result, fragment in ((fb.spearman([1, 1, 1], [1, 2, 3]), "scores are constant"),
                             (fb.spearman([1, 2, 3], [5, 5, 5]), "outcomes are constant"),
                             (fb.spearman([1.0], [2.0]), "fewer than two")):
        assert result["value"] is None and fragment in result["reason"]
    none = fb.concordance([1, 2, 3], [4, 4, 4])
    assert none["value"] is None and "distinct outcomes" in none["reason"]


# ---------------------------------------------------------------------------
# the reward predictor and feature-space interventions
# ---------------------------------------------------------------------------

def test_ridge_matches_the_closed_form_and_is_frozen():
    space = explicit_space(("x",))
    model = fb.fit_reward([[0.0], [1.0], [2.0]], [1.0, 3.0, 5.0], space, alpha=1.0)
    assert model.weights[0] == pytest.approx(4.0 / 3.0) and model.intercept == pytest.approx(5.0 / 3.0)
    assert model.predict([[1.0]])[0] == pytest.approx(3.0)
    near = fb.fit_reward([[0.0], [1.0], [2.0]], [1.0, 3.0, 5.0], space, alpha=1e-9)
    assert near.weights[0] == pytest.approx(2.0, abs=1e-8)
    with pytest.raises(ValueError):
        model.weights[0] = 0.0
    with pytest.raises(ValueError, match="alpha"):
        fb.fit_reward([[0.0]], [1.0], space, alpha=0.0)


def test_raw_predictions_do_not_depend_on_the_other_rows_of_a_batch():
    rng = np.random.default_rng(1)
    space = explicit_space(("a", "b", "c"))
    model = fb.fit_reward(rng.normal(size=(10, 3)), rng.normal(size=10), space, alpha=0.5)
    batch = rng.normal(size=(7, 3))
    together = model.predict(batch)
    for index in range(7):
        assert model.predict(batch[index:index + 1])[0] == pytest.approx(together[index], abs=1e-12)
    order = rng.permutation(7)
    assert model.predict(batch[order]) == pytest.approx(together[order], abs=1e-12)


def test_ablation_uses_the_frozen_training_mean_never_the_batch():
    space = explicit_space(("x1", "x2"))
    reference = fb.FeatureReference.from_training([[0.0, 0.0], [2.0, 4.0], [4.0, 8.0]], space)
    assert reference.mean.tolist() == [2.0, 4.0]
    batch = np.array([[10.0, 10.0], [20.0, 20.0]])
    together = fb.intervene_features(batch, space, "ablate", columns=("x1",), reference=reference)
    alone = fb.intervene_features(batch[:1], space, "ablate", columns=("x1",), reference=reference)
    assert together.tolist() == [[2.0, 10.0], [2.0, 20.0]] and alone.tolist() == [[2.0, 10.0]]
    assert batch.tolist() == [[10.0, 10.0], [20.0, 20.0]], "the input is never edited in place"
    with pytest.raises(ValueError):
        reference.mean[0] = 1.0
    with pytest.raises(ValueError, match="explicit frozen"):
        fb.intervene_features(batch, space, "ablate", columns=("x1",))
    other = fb.FeatureReference.from_training([[0.0, 0.0]], explicit_space(("y1", "y2")))
    with pytest.raises(ValueError, match="other columns"):
        fb.intervene_features(batch, space, "ablate", columns=("x1",), reference=other)


def test_patch_and_identity_interventions_validate_their_arguments():
    space = explicit_space(("x1", "x2"))
    batch = np.array([[1.0, 2.0], [3.0, 4.0]])
    broadcast = fb.intervene_features(batch, space, "patch", columns=("x2",), donor=[9.0, 8.0])
    assert broadcast.tolist() == [[1.0, 8.0], [3.0, 8.0]]
    matched = fb.intervene_features(batch, space, "patch", columns=("x1",),
                                    donor=[[7.0, 0.0], [6.0, 0.0]])
    assert matched.tolist() == [[7.0, 2.0], [6.0, 4.0]]
    assert fb.intervene_features(batch, space, "identity").tolist() == batch.tolist()
    with pytest.raises(ValueError, match="matched donor"):
        fb.intervene_features(batch, space, "patch", columns=("x1",), donor=[[1.0, 2.0]])
    with pytest.raises(ValueError, match="needs an explicit donor"):
        fb.intervene_features(batch, space, "patch", columns=("x1",))
    with pytest.raises(ValueError, match="takes no"):
        fb.intervene_features(batch, space, "identity", columns=("x1",))
    with pytest.raises(ValueError, match="unknown feature column"):
        fb.intervene_features(batch, space, "patch", columns=("nope",), donor=[0.0, 0.0])
    with pytest.raises(ValueError, match="unknown intervention"):
        fb.intervene_features(batch, space, "swap")


def test_a_feature_space_cannot_be_built_without_its_provenance():
    with pytest.raises(TypeError):
        fb.FeatureSpace(kind="explicit", names=("x",), schema={}, history=("explicit",))
    with pytest.raises(ValueError, match="provenance is required"):
        fb.FeatureSpace(kind="explicit", names=("x",), schema={}, history=("explicit",),
                        supervision="label_free")
    with pytest.raises(ValueError, match="history"):
        fb.FeatureSpace(kind="explicit", names=("x",), schema={}, history=(),
                        supervision=label_free())
    with pytest.raises(ValueError, match="unique"):
        fb.FeatureSpace(kind="explicit", names=("x", "x"), schema={}, history=("e",),
                        supervision=label_free())
    sub, indices = explicit_space(("a", "b", "c")).select(("c", "a"))
    assert indices == [2, 0] and sub.names == ("c", "a")
    assert "select columns" in sub.history[-1] and sub.supervision.status == "label_free"


# ---------------------------------------------------------------------------
# preference pairs and the DPO fit
# ---------------------------------------------------------------------------

def test_pairs_exclude_ties_and_zero_prior_endpoints_and_count_both():
    everything = fb.preference_pairs([1.0, 1.0, 2.0, 3.0], [True] * 4, max_pairs=100, seed=0)
    assert everything.diagnostics["eligible_pairs"] == 5
    assert everything.diagnostics["ties_excluded"] == 1
    assert everything.diagnostics["pairs_used"] == 5
    restricted = fb.preference_pairs([1.0, 1.0, 2.0, 3.0], [True, True, True, False],
                                     max_pairs=100, seed=0)
    diagnostics = restricted.diagnostics
    assert diagnostics["excluded_zero_prior_labels"] == 1
    assert diagnostics["labelled_candidates"] == 4 and diagnostics["positive_support_labels"] == 3
    assert restricted.winners.tolist() == [2, 2] and restricted.losers.tolist() == [0, 1]
    assert 3 not in restricted.winners.tolist() + restricted.losers.tolist()


def test_a_finite_very_negative_prior_remains_eligible_and_only_minus_inf_is_excluded():
    problem = tiny_problem(train_log_prior=[0.0, 0.0, -10000.0, 0.0, 0.0, 0.0])
    fit = fb.fit_stage(problem, tiny_config())
    assert fit.pairs.diagnostics["excluded_zero_prior_labels"] == 0
    assert fit.pairs.diagnostics["pairs_used"] == 15
    zeroed = fb.fit_stage(tiny_problem(train_log_prior=[0.0, 0.0, -math.inf, 0.0, 0.0, 0.0]),
                          tiny_config())
    assert zeroed.pairs.diagnostics["excluded_zero_prior_labels"] == 1
    assert zeroed.pairs.diagnostics["eligible_pairs"] == 10
    used = set(zeroed.pairs.winners.tolist()) | set(zeroed.pairs.losers.tolist())
    assert 2 not in used, "a pair touched the zero-prior candidate"
    assert zeroed.reward.n_train == 6, "ridge may legitimately use the zero-prior label"


def test_the_pair_cap_samples_with_a_recorded_seed_and_keeps_the_label_count():
    utilities = np.arange(30.0)
    flags = np.ones(30, dtype=bool)
    capped = fb.preference_pairs(utilities, flags, max_pairs=10, seed=5)
    assert capped.diagnostics["pairs_used"] == 10 and capped.diagnostics["subsampled"] is True
    assert capped.diagnostics["method"] == "seeded_uniform_rank_sample"
    assert capped.diagnostics["labelled_candidates"] == 30
    assert capped.diagnostics["eligible_pairs"] == 435 and capped.diagnostics["seed"] == 5
    unordered = {frozenset(pair) for pair in zip(capped.winners.tolist(), capped.losers.tolist())}
    assert len(unordered) == 10, "a sampled pair repeated"
    assert all(utilities[w] > utilities[l] for w, l in zip(capped.winners, capped.losers))
    again = fb.preference_pairs(utilities, flags, max_pairs=10, seed=5)
    assert again.winners.tolist() == capped.winners.tolist()
    other = fb.preference_pairs(utilities, flags, max_pairs=10, seed=6)
    assert other.winners.tolist() != capped.winners.tolist() \
        or other.losers.tolist() != capped.losers.tolist()
    middle = fb.preference_pairs(utilities, flags, max_pairs=200, seed=5)
    assert middle.diagnostics["method"] == "seeded_uniform_rank_sample"
    assert middle.diagnostics["pairs_used"] == 200
    full = fb.preference_pairs(utilities, flags, max_pairs=500, seed=5)
    assert full.diagnostics["method"] == "all_eligible_pairs" and full.diagnostics["pairs_used"] == 435
    assert full.diagnostics["subsampled"] is False


def test_dpo_takes_real_steps_the_loss_decreases_and_the_optimum_is_a_true_minimum():
    rng = np.random.default_rng(3)
    features = rng.normal(size=(12, 4))
    utilities = features @ rng.normal(size=4)
    pairs = fb.preference_pairs(utilities, np.ones(12, dtype=bool), max_pairs=1000, seed=0)
    space = explicit_space(tuple(f"f{i}" for i in range(4)))
    policy = fb.fit_dpo(features, pairs.winners, pairs.losers, space, beta=2.0, l2=0.05)
    diagnostics = policy.diagnostics
    assert diagnostics["converged"] and diagnostics["updates"] >= 2
    assert diagnostics["final_loss"] < diagnostics["initial_loss"]
    history = diagnostics["loss_history"]
    assert len(history) == diagnostics["updates"] + 1
    assert all(later <= earlier + 1e-12 for earlier, later in zip(history, history[1:]))
    assert float(np.linalg.norm(policy.theta)) > 0.1, "the weights never moved"
    delta = features[pairs.winners] - features[pairs.losers]

    def regularised(theta):
        return float(np.mean(np.logaddexp(0.0, -2.0 * (delta @ theta))) + 0.025 * float(theta @ theta))
    best = regularised(np.asarray(policy.theta))
    for _ in range(20):
        assert regularised(np.asarray(policy.theta) + 1e-3 * rng.normal(size=4)) > best
    assert np.corrcoef(features @ policy.theta, utilities)[0, 1] > 0.5


def test_strictly_increasing_transforms_leave_dpo_unchanged_but_move_ridge_and_the_tilt():
    problem = tiny_problem()
    pairs = fb.preference_pairs(problem.train_utilities, np.ones(6, dtype=bool), max_pairs=100,
                                seed=1)
    transformed = np.exp(np.asarray(problem.train_utilities) / 3.0)
    moved = fb.preference_pairs(transformed, np.ones(6, dtype=bool), max_pairs=100, seed=1)
    assert moved.winners.tolist() == pairs.winners.tolist()
    assert moved.losers.tolist() == pairs.losers.tolist()
    space = problem.space
    original = fb.fit_dpo(problem.train_features, pairs.winners, pairs.losers, space)
    again = fb.fit_dpo(problem.train_features, moved.winners, moved.losers, space)
    assert np.array_equal(original.theta, again.theta)
    ridge_a = fb.fit_reward(problem.train_features, problem.train_utilities, space, alpha=0.1)
    ridge_b = fb.fit_reward(problem.train_features, transformed, space, alpha=0.1)
    assert not np.allclose(ridge_a.weights, ridge_b.weights)
    tilt_a = fb.log_tilt(problem.support_log_prior, ridge_a.predict(problem.support_features), 1.0)
    tilt_b = fb.log_tilt(problem.support_log_prior, ridge_b.predict(problem.support_features), 1.0)
    assert fb.total_variation(tilt_a, tilt_b) > 1e-3


def test_without_a_usable_pair_dpo_is_reported_unsupported_not_called_trained(frozen):
    tied = fb.fit_and_allocate(tiny_problem(train_utilities=[2.0] * 6), tiny_config())
    assert tied["fitted"]["dpo"]["status"] == "unsupported"
    assert "no usable preference pair" in tied["fitted"]["dpo"]["reason"]
    assert tied["predictions"]["dpo_energy"] is None
    assert not [name for name in tied["policies"]["log_probability"] if name.startswith("dpo_")]
    assert fb.verify_artifact(tied) is True
    report = fb.evaluate_frozen(tied, TRUTH)
    assert report["dpo"]["status"] == "unsupported" and "reason" in report["dpo"]
    constant = report["prediction_ranking"]["all_candidates"]["spearman"]
    assert constant["value"] is None and "constant" in constant["reason"]
    lonely = fb.fit_stage(tiny_problem(train_log_prior=[0.0] + [-math.inf] * 5), tiny_config())
    assert lonely.dpo is None and "no usable preference pair" in lonely.dpo_unsupported_reason
    assert frozen["fitted"]["dpo"]["status"] == "fit"


def test_dpo_rejects_malformed_pairs_and_an_unbounded_pair_matrix(monkeypatch):
    space = explicit_space(("x",))
    with pytest.raises(ValueError, match="at least one"):
        fb.fit_dpo([[0.0], [1.0]], [], [], space)
    with pytest.raises(ValueError, match="itself"):
        fb.fit_dpo([[0.0], [1.0]], [0], [0], space)
    with pytest.raises(ValueError, match="outside"):
        fb.fit_dpo([[0.0], [1.0]], [5], [0], space)
    with pytest.raises(ValueError, match="l2"):
        fb.fit_dpo([[0.0], [1.0]], [1], [0], space, l2=0.0)
    monkeypatch.setattr(fb, "MAX_PAIR_MATRIX_ELEMENTS", 3)
    with pytest.raises(ValueError, match="bound on the pair-difference matrix"):
        fb.fit_dpo([[0.0], [1.0]], [1, 1, 1, 1], [0, 0, 0, 0], space)


def test_the_dense_gram_and_hessian_are_bounded_before_anything_is_allocated(monkeypatch):
    # pairs x d does NOT bound d x d: one pair of 16384-wide rows would still ask for a 2 GiB Hessian.
    # The dimension check is integer arithmetic that precedes the width check on the features and
    # every allocation, so deliberately mismatched tiny features reach the dense-bound error first.
    wide = explicit_space(names=tuple(f"w{i}" for i in range(16384)))
    assert 16384 * 16384 > fb.MAX_DENSE_SYSTEM_ELEMENTS >= 4096 * 4096

    def refuse_allocation(*args, **kwargs):
        raise AssertionError("an identity matrix was allocated for a refused dense system")
    with monkeypatch.context() as patched:
        patched.setattr(np, "eye", refuse_allocation)
        with pytest.raises(ValueError, match="dense DPO Newton Hessian system is 16384 x 16384"):
            fb.fit_dpo([[0.0], [1.0]], [1], [0], wide)
        with pytest.raises(ValueError, match="dense ridge Gram system is 16384 x 16384"):
            fb.fit_reward([[0.0], [1.0]], [0.0, 1.0], wide, alpha=1.0)
        with pytest.raises(ValueError, match="dense ridge Gram system"):
            fb._ridge_solution(np.zeros((2, 16384)), np.zeros(2), 1.0)   # the verifier's path too
    # a small injected bound pins the boundary exactly: dimension 2 (4 elements) fits, 3 (9) does not
    monkeypatch.setattr(fb, "MAX_DENSE_SYSTEM_ELEMENTS", 4)
    two, three = explicit_space(("a", "b")), explicit_space(("a", "b", "c"))
    model = fb.fit_reward([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]], [1.0, 2.0, 3.0], two, alpha=1.0)
    assert model.n_train == 3
    assert fb.fit_dpo([[0.0, 0.0], [1.0, 1.0]], [1], [0], two).diagnostics["pairs"] == 1
    with pytest.raises(ValueError, match="3 x 3 = 9 elements"):
        fb.fit_reward(np.zeros((3, 3)), [0.0, 1.0, 2.0], three, alpha=1.0)
    with pytest.raises(ValueError, match="3 x 3 = 9 elements"):
        fb.fit_dpo(np.zeros((2, 3)), [1], [0], three)
    # the pair bound is a separate, still-enforced limit
    monkeypatch.setattr(fb, "MAX_PAIR_MATRIX_ELEMENTS", 3)
    with pytest.raises(ValueError, match="bound on the pair-difference matrix"):
        fb.fit_dpo([[0.0, 0.0], [1.0, 1.0]], [1, 1], [0, 0], two)


# ---------------------------------------------------------------------------
# SIR accounting
# ---------------------------------------------------------------------------

def test_sir_refuses_an_uncovered_support_and_a_zero_total_weight():
    with pytest.raises(ValueError, match="gives zero mass"):
        fb.sir_weighted_distribution(np.log([0.5, 0.5]), np.array([0.0, -np.inf]), [0.0, 0.0],
                                     1.0, np.array([2, 0]))
    with pytest.raises(ValueError, match="zero total sampled weight"):
        fb.sir_weighted_distribution(np.array([0.0, -np.inf]), np.log([0.5, 0.5]), [0.0, 0.0],
                                     1.0, np.array([0, 4]))
    with pytest.raises(ValueError, match="at least one proposal draw"):
        fb.sir_weighted_distribution(np.log([0.5, 0.5]), np.log([0.5, 0.5]), [0.0, 0.0], 1.0,
                                     np.array([0, 0]))
    with pytest.raises(ValueError, match="non-negative integer"):
        fb.sir_weighted_distribution(np.log([0.5, 0.5]), np.log([0.5, 0.5]), [0.0, 0.0], 1.0,
                                     np.array([1.5, 0.5]))


def test_sir_draws_keep_multiplicity_and_resource_accounting_is_honest(frozen):
    sir = frozen["policies"]["sir"]
    assert sum(sir["counts"]) == sir["proposal_draws"] == 50
    assert sum(sir["final_histogram"]) == sir["final_draws"] == 20
    accounting = sir["resource_accounting"]
    assert accounting["proposal_draws_K"] == 50 and accounting["final_draws_M"] == 20
    assert accounting["full_support_prediction_calls"] == 4
    assert accounting["unique_candidates_queried_by_sampler"] == sir["distinct_proposed"] \
        == sum(1 for n in sir["counts"] if n > 0)
    assert accounting["final_resample_new_prediction_calls"] == 0
    assert accounting["scores_precomputed_for_full_support"] is True
    assert accounting["scoring_cost_saving_claimed"] is False
    prior = np.exp(fb.decode_log(frozen["prior"]["log_probability"]))
    expected = float(np.sum(1.0 - (1.0 - prior) ** 50))
    assert accounting["expected_unique_queries_with_caching"] == pytest.approx(expected)
    assert 1.0 <= sir["effective_sample_size"] <= 50.0


def test_the_report_separates_the_weighted_law_from_the_final_histogram(frozen):
    report = fb.evaluate_frozen(frozen, TRUTH)
    sir = report["sir"]
    weighted = np.exp(fb.decode_log(frozen["policies"]["log_probability"]["sir_weighted_proposal"]))
    expected = float(np.sum(1.0 - (1.0 - weighted) ** 20))
    assert sir["weighted_proposal_law"]["expected_distinct_total"] == pytest.approx(expected)
    histogram = np.array(frozen["policies"]["sir"]["final_histogram"])
    realized = sir["final_resampled_histogram"]
    assert realized["realized_distinct"] == int((histogram > 0).sum())
    assert realized["draws_M"] == 20 and "second hypothetical" in realized["note"]
    assert sir["occurrence_effective_sample_size"] == pytest.approx(
        frozen["policies"]["sir"]["effective_sample_size"])
    assert sir["distinct_proposed_candidates"] == frozen["policies"]["sir"]["distinct_proposed"]


def test_the_sir_attribution_identity_holds_and_the_gap_is_nonnegative(frozen):
    report = fb.evaluate_frozen(frozen, TRUTH)
    attribution = report["sir"]["attribution_against_learned_exact_tilt"]
    assert attribution["status"] == "finite" and attribution["identity_holds"] is True
    assert attribution["predicted_gap_is_nonnegative"] is True
    assert attribution["predicted_objective_gap"] >= 0.0
    assert attribution["true_J_difference_from_learned_exact"] == pytest.approx(
        -attribution["predicted_objective_gap"] + attribution["residual_cross_term"], abs=1e-9)
    assert "not two nonnegative errors" in attribution["statement"]


def test_a_declared_proposal_is_used_and_must_be_declared_explicitly():
    config = tiny_config(sir_proposal="declared")
    ids = ("a", "b", "c")
    allocation = fb.allocate(ids, [0.0, 0.0, 0.0], [0.0, 0.5, 1.0], config,
                             proposal_log_weights=np.log([0.2, 0.3, 0.5]))
    assert allocation.sir["proposal_kind"] == "declared"
    assert np.exp(allocation.sir["log_proposal"]) == pytest.approx([0.2, 0.3, 0.5])
    with pytest.raises(ValueError, match="if and only if"):
        fb.allocate(ids, [0.0] * 3, [0.0] * 3, config)
    with pytest.raises(ValueError, match="if and only if"):
        fb.allocate(ids, [0.0] * 3, [0.0] * 3, tiny_config(), proposal_log_weights=[0.0] * 3)
    with pytest.raises(ValueError, match="declared SIR proposal"):
        fb.fit_and_allocate(tiny_problem(), config)
    uniform = fb.allocate(ids, [0.0, -math.inf, 0.0], [0.0] * 3,
                          tiny_config(sir_proposal="uniform_over_target_support"))
    assert np.exp(uniform.sir["log_proposal"]) == pytest.approx([0.5, 0.0, 0.5])


# ---------------------------------------------------------------------------
# stage separation and the frozen artifact
# ---------------------------------------------------------------------------

def test_fit_and_allocate_have_no_outcome_argument_and_refuse_one():
    for function in (fb.fit_reward, fb.fit_dpo, fb.preference_pairs, fb.fit_stage, fb.allocate,
                     fb.freeze_artifact, fb.fit_and_allocate):
        names = set(inspect.signature(function).parameters)
        assert not {name for name in names if "outcome" in name or "truth" in name}, function
    assert "outcomes" in inspect.signature(fb.evaluate_frozen).parameters
    with pytest.raises(TypeError):
        fb.allocate(("a", "b"), [0.0, 0.0], [0.0, 1.0], tiny_config(), outcomes={"a": 1.0})
    with pytest.raises(TypeError):
        fb.fit_stage(tiny_problem(), tiny_config(), outcomes=TRUTH)


def test_allocation_demands_canonical_order_and_every_policy_is_a_distribution():
    with pytest.raises(ValueError, match="canonical order"):
        fb.allocate(("b", "a"), [0.0, 0.0], [0.0, 1.0], tiny_config())
    allocation = fb.allocate(("a", "b", "c"), [0.0, -math.inf, 0.2], [0.0, 3.0, 1.0],
                             tiny_config(), dpo_energy=[0.0, 1.0, -1.0])
    assert set(allocation.policies) == {"prior", "ridge_tilt", "dpo_pressure_0.25",
                                        "dpo_pressure_0.5", "dpo_pressure_1", "dpo_pressure_2",
                                        "dpo_pressure_4", "sir_weighted_proposal"}
    for name, log_q in allocation.policies.items():
        assert float(np.exp(log_q).sum()) == pytest.approx(1.0, abs=1e-12), name
        assert log_q[1] == -np.inf, f"{name} gave mass to a structural zero"
        assert not log_q.flags.writeable


def test_the_artifact_is_deterministic_and_independent_of_input_row_order():
    config = tiny_config()
    first = fb.fit_and_allocate(tiny_problem(), config)
    second = fb.fit_and_allocate(tiny_problem(), config)
    assert first == second
    order_t, order_s = [4, 1, 5, 0, 3, 2], [2, 0, 3, 1]
    base = problem_kwargs()
    shuffled = problem_kwargs(
        train_ids=tuple(base["train_ids"][i] for i in order_t),
        train_features=[base["train_features"][i] for i in order_t],
        train_utilities=[base["train_utilities"][i] for i in order_t],
        train_log_prior=[base["train_log_prior"][i] for i in order_t],
        train_groups=tuple(base["train_groups"][i] for i in order_t),
        support_ids=tuple(base["support_ids"][i] for i in order_s),
        support_features=[base["support_features"][i] for i in order_s],
        support_log_prior=[base["support_log_prior"][i] for i in order_s],
        support_groups=tuple(base["support_groups"][i] for i in order_s))
    permuted = fb.fit_and_allocate(fb.FiniteSupportProblem(**shuffled), config)
    assert permuted == first, "candidate order changed an ID-aligned result or the seeded draws"


def test_mutating_evaluation_outcomes_leaves_every_frozen_hash_unchanged(frozen):
    snapshot = copy.deepcopy(frozen)
    first = fb.evaluate_frozen(frozen, TRUTH)
    flipped = {key: -value + 10.0 for key, value in TRUTH.items()}
    second = fb.evaluate_frozen(frozen, flipped)
    assert frozen == snapshot, "evaluating mutated the artifact"
    assert frozen["hashes"] == snapshot["hashes"]
    assert first["artifact_sha256"] == second["artifact_sha256"] == frozen["hashes"]["artifact_sha256"]
    assert first["oracle"] != second["oracle"], "the oracle must depend on the outcomes"
    assert first["prediction_ranking"] != second["prediction_ranking"]


def test_evaluation_demands_complete_finite_outcomes_for_exactly_the_heldout_support(frozen):
    incomplete = {key: value for key, value in TRUTH.items() if key != "s3"}
    with pytest.raises(ValueError, match="missing for 1 of 4"):
        fb.evaluate_frozen(frozen, incomplete)
    with pytest.raises(ValueError, match="outside the held-out support"):
        fb.evaluate_frozen(frozen, {**TRUTH, "t0": 1.0})
    with pytest.raises(ValueError, match="finite"):
        fb.evaluate_frozen(frozen, {**TRUTH, "s0": float("nan")})
    with pytest.raises(ValueError, match="finite"):
        fb.evaluate_frozen(frozen, {**TRUTH, "s0": float("inf")})


def test_structural_zeros_get_no_mass_but_still_need_their_outcome(frozen):
    ids = frozen["support"]["ids"]
    zero = ids.index("s2")
    for name, values in frozen["policies"]["log_probability"].items():
        assert values[zero] is None, f"{name} gave mass to the structural zero"
    assert frozen["prior"]["candidate_count"] == 4 and frozen["prior"]["reachable_count"] == 3
    assert frozen["prior"]["structural_zero_count"] == 1
    report = fb.evaluate_frozen(frozen, TRUTH)
    assert report["support"]["reachable_support"] == 3 and report["support"]["structural_zeros"] == 1
    assert report["policies"]["ridge_tilt"]["reachable_support"] == 3
    with pytest.raises(ValueError, match="missing"):
        fb.evaluate_frozen(frozen, {key: v for key, v in TRUTH.items() if key != "s2"})


SECTION_TAMPERS = {
    "manifest": lambda doc: doc["manifest"].update(target="other"),
    "config": lambda doc: doc["config"].update(tau=2.0),
    "features": lambda doc: doc["features"]["space"]["history"].append("sneaky step"),
    "training": lambda doc: doc["training"]["utilities"].__setitem__(0, 99.0),
    "prior": lambda doc: doc["prior"]["log_weight"].__setitem__(0, 5.0),
    "support": lambda doc: doc["support"]["ids"].reverse(),
    "fitted": lambda doc: doc["fitted"]["reward"]["weights"].__setitem__(0, 7.0),
    "predictions": lambda doc: doc["predictions"]["utility"].__setitem__(0, 7.0),
    "policies": lambda doc: doc["policies"]["log_probability"]["ridge_tilt"].__setitem__(0, -0.1),
    "provenance": lambda doc: doc["provenance"].update(declared_not_verified=False)}


def test_every_hashed_section_is_covered():
    assert set(SECTION_TAMPERS) == set(fb.HASHED_SECTIONS)


@pytest.mark.parametrize("section", sorted(SECTION_TAMPERS))
def test_editing_any_section_of_the_artifact_is_refused(frozen, section):
    tampered = copy.deepcopy(frozen)
    SECTION_TAMPERS[section](tampered)
    with pytest.raises(ValueError, match="modified after freezing"):
        fb.verify_artifact(tampered)
    with pytest.raises(ValueError):
        fb.evaluate_frozen(tampered, TRUTH)


def test_smuggled_keys_a_missing_hash_block_and_a_rehash_forgery_are_all_refused(frozen):
    smuggled = copy.deepcopy(frozen)
    smuggled["extra"] = {"note": "unhashed"}
    with pytest.raises(ValueError, match="outside the hash"):
        fb.verify_artifact(smuggled)
    headless = copy.deepcopy(frozen)
    del headless["hashes"]
    with pytest.raises(ValueError, match="differ from the hashed layout"):
        fb.verify_artifact(headless)
    forged_weights = copy.deepcopy(frozen)
    forged_weights["fitted"]["reward"]["weights"][0] += 1.0
    forged_weights["hashes"] = fb._section_hashes(forged_weights)
    with pytest.raises(ValueError, match="ridge solution"):
        fb.verify_artifact(forged_weights)
    forged_predictions = copy.deepcopy(frozen)
    forged_predictions["predictions"]["utility"][0] += 0.5
    forged_predictions["hashes"] = fb._section_hashes(forged_predictions)
    with pytest.raises(ValueError, match="exact tilt"):
        fb.verify_artifact(forged_predictions)
    leaked = copy.deepcopy(frozen)
    leaked["training"]["ids"][0] = "s0"
    leaked["hashes"] = fb._section_hashes(leaked)
    with pytest.raises(ValueError, match="overlap"):
        fb.verify_artifact(leaked)
    assert fb.verify_artifact(frozen) is True


def test_inputs_and_artifacts_are_copied_at_every_boundary():
    features = np.array(problem_kwargs()["train_features"], dtype=float)
    problem = tiny_problem(train_features=features)
    before = problem.train_features.copy()
    features[:] = 99.0
    assert np.array_equal(problem.train_features, before)
    for array in (problem.train_features, problem.support_features, problem.train_utilities,
                  problem.train_log_prior, problem.support_log_prior):
        assert not array.flags.writeable
    with pytest.raises(ValueError):
        problem.train_features[0, 0] = 5.0
    artifact = fb.fit_and_allocate(problem, tiny_config())
    artifact["training"]["features"][0][0] = 123.0
    artifact["predictions"]["utility"][0] = 123.0
    assert problem.train_features[0, 0] == 0.0
    again = fb.fit_and_allocate(problem, tiny_config())
    assert again["training"]["features"][0][0] == 0.0 and again["predictions"]["utility"][0] != 123.0


# ---------------------------------------------------------------------------
# the evaluation report
# ---------------------------------------------------------------------------

def test_the_report_shares_one_tau_keeps_the_oracle_on_top_and_every_identity_holds(frozen):
    report = fb.evaluate_frozen(frozen, TRUTH)
    assert report["tau"] == 1.0 and report["identities"]["all_hold"] is True
    best = report["oracle"]["objective_J"]["value"]
    assert report["oracle_objective_closed_form"] == pytest.approx(best, abs=1e-12)
    assert report["oracle"]["oracle_regret"]["value"] == pytest.approx(0.0, abs=1e-12)
    for name, entry in report["policies"].items():
        assert entry["objective_J"]["value"] <= best + 1e-12, name
        assert entry["oracle_regret"]["value"] >= -1e-12, name
        assert entry["regret_identity"]["holds"] is True, name
        residual = entry["oracle_regret"]["value"] \
            - report["tau"] * entry["kl_policy_to_oracle"]["value"]
        assert abs(residual) < 1e-9, name
    prior = report["policies"]["prior"]
    assert prior["kl_policy_to_prior"]["value"] == pytest.approx(0.0, abs=1e-12)
    assert prior["tv_to_prior"] == pytest.approx(0.0, abs=1e-12)
    assert "vs_learned_exact_tilt" in report["policies"]["sir_weighted_proposal"]
    assert "vs_learned_exact_tilt" not in report["policies"]["ridge_tilt"]
    assert report["truth_coverage"]["complete"] is True
    assert report["useful_threshold"] == 1.0 and report["support"]["useful_candidates"] == 3
    assert {"scope", "provenance", "oracle", "sir", "dpo_energy"} <= set(report["claims"])


def test_the_prediction_ranking_is_reported_apart_from_the_policy_probability_ranking(frozen):
    report = fb.evaluate_frozen(frozen, TRUTH)
    assert {"all_candidates", "reachable_support", "rmse", "mean_bias"} <= set(
        report["prediction_ranking"])
    policy = report["policies"]["dpo_pressure_1"]["ranking"]
    assert {"policy_log_probability_vs_utility",
            "policy_log_ratio_to_prior_vs_utility"} <= set(policy)
    constant = report["policies"]["prior"]["ranking"]["policy_log_ratio_to_prior_vs_utility"]
    assert constant["spearman"]["value"] is None and constant["spearman"]["reason"]


def test_the_dpo_pressure_curve_is_a_separate_true_j_and_kl_comparison(frozen):
    report = fb.evaluate_frozen(frozen, TRUTH)
    dpo = report["dpo"]
    assert dpo["status"] == "fit" and "not applicable" in dpo["attribution"]
    curve = dpo["pressure_curve"]
    assert [row["pressure"] for row in curve] == [0.25, 0.5, 1.0, 2.0, 4.0]
    kls = [row["kl_policy_to_prior"]["value"] for row in curve]
    assert all(later >= earlier - 1e-12 for earlier, later in zip(kls, kls[1:])), (
        "KL(q_c || p) is nondecreasing in the pressure c for an exponential tilt")
    assert dpo["training_diagnostics"]["converged"] is True
    assert report["dpo"]["note"] == report["claims"]["dpo_energy"]


def test_an_infinite_kl_is_a_status_and_the_json_is_strict():
    artifact = fb.fit_and_allocate(tiny_problem(), tiny_config(sir_proposal_draws=1, final_draws=1))
    report = fb.evaluate_frozen(artifact, TRUTH)
    law = report["policies"]["sir_weighted_proposal"]
    assert law["kl_prior_to_policy"] == {"status": "infinite", "value": None}
    assert law["kl_policy_to_prior"]["status"] == "finite"
    assert report["policies"]["sir_weighted_proposal"]["vs_learned_exact_tilt"][
        "kl_learned_to_policy"]["status"] == "infinite"
    assert report["sir"]["occurrence_effective_sample_size"] == pytest.approx(1.0)
    text = json.dumps(report, allow_nan=False)
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text) == report
    with pytest.raises(ValueError, match="NaN/Infinity"):
        fb.write_json("unused.json", {"bad": float("inf")})


def test_label_accounting_separates_budget_pairs_repeats_and_validation():
    problem = tiny_problem()
    config = tiny_config(label_budget=4, declared_validation_labels=2, seed=9)
    artifact = fb.fit_and_allocate(problem, config)
    accounting = artifact["provenance"]["label_accounting"]
    assert accounting["offered_labels"] == 6 and accounting["acquired_labels"] == 4
    assert accounting["expanded_preference_pairs"] == 6 != accounting["acquired_labels"]
    assert "labels_consumed_total" not in accounting, "acquired + validation is an invented sum"
    assert accounting["declared_validation_labels"] == 2
    assert accounting["validation_cost"]["declared_labels"] == 2
    assert accounting["validation_cost"]["disjointness_from_acquired_asserted"] is False
    assert accounting["selection"]["evaluation_outcomes_used"] is False
    chosen = artifact["training"]["ids"]
    assert len(chosen) == 4 and set(chosen) <= set(problem.train_ids)
    effective = artifact["provenance"]["effective_data"]
    assert effective["ridge"]["labels_used"] == 4
    assert effective["dpo"]["pairs_used"] == 6 and effective["dpo"]["eligible_pairs"] == 6
    assert artifact["provenance"]["split_counts"]["train_labelled_acquired"] == 4
    same = fb.fit_and_allocate(problem, config)
    assert same["training"]["ids"] == chosen
    other = [fb.fit_and_allocate(problem, tiny_config(label_budget=4, seed=seed))["training"]["ids"]
             for seed in range(10, 16)]
    assert any(ids != chosen for ids in other), "the seed never changed the selected labels"


def test_acquire_labels_is_seeded_sorted_and_refuses_to_invent_labels():
    chosen = fb.acquire_labels(10, 4, 5)
    assert chosen.tolist() == sorted(set(chosen.tolist())) and len(chosen) == 4
    assert chosen.max() < 10 and fb.acquire_labels(10, 4, 5).tolist() == chosen.tolist()
    assert fb.acquire_labels(10, None, 5).tolist() == list(range(10))
    with pytest.raises(ValueError, match="exceeds"):
        fb.acquire_labels(3, 4, 0)


# ---------------------------------------------------------------------------
# provenance and label-history guards
# ---------------------------------------------------------------------------

def test_clean_low_n_is_true_only_for_label_free_histories():
    clean = fb.clean_low_n_eligibility(
        target="T", assay="A", feature_history=label_free(), prior_history=label_free(),
        train_ids=["t"], support_ids=["s"], acquired_labels=24)
    assert clean["clean_low_n_eligible"] is True and clean["status"] == "eligible"
    assert clean["acquisition_replay_eligible"] is True and clean["declared_not_verified"] is True


def test_an_unknown_history_makes_the_verdict_unavailable_not_true():
    for features, prior in ((unknown_history(), label_free()), (label_free(), unknown_history())):
        verdict = fb.clean_low_n_eligibility(
            target="T", assay="A", feature_history=features, prior_history=prior,
            train_ids=["t"], support_ids=["s"], acquired_labels=24)
        assert verdict["clean_low_n_eligible"] is None and verdict["status"] == "unavailable"
        assert verdict["acquisition_replay_eligible"] is None
        assert verdict["label_cost"]["unknown_history_present"] is True


def test_a_checkpoint_trained_on_2146_target_labels_is_not_a_clean_24_shot_model():
    verdict = fb.clean_low_n_eligibility(
        target="T", assay="A", feature_history=label_free(),
        prior_history=documented(entry(count=2146)), train_ids=["t"], support_ids=["s"],
        acquired_labels=24)
    assert verdict["clean_low_n_eligible"] is False and verdict["status"] == "ineligible"
    assert any("24-shot" in reason for reason in verdict["reasons"])
    cost = verdict["label_cost"]
    assert "documented_effective_target_labels" not in cost, "no invented 2146 + 24 sum"
    unique = cost["unique_same_target_labels"]
    assert unique["exact"] is None and unique["status"] == "bounded_only"
    assert (unique["lower_bound"], unique["upper_bound"]) == (2146, 2170), (
        "without label ids the 24 new labels may or may not be among the 2146")
    assert cost["declared_label_uses"]["prior_history_same_target"] == 2146
    assert cost["declared_label_uses"]["acquired_current"] == 24


def test_overlapping_label_ids_are_leakage_or_double_payment_and_other_targets_are_transfer():
    leak = fb.clean_low_n_eligibility(
        target="T", assay="A", feature_history=documented(entry(count=1, target="X", ids=["s"])),
        prior_history=label_free(), train_ids=["t"], support_ids=["s"], acquired_labels=1)
    assert leak["clean_low_n_eligible"] is False and "leakage" in leak["reasons"][0]
    paid = fb.clean_low_n_eligibility(
        target="T", assay="A", feature_history=documented(entry(count=1, target="X", ids=["t"])),
        prior_history=label_free(), train_ids=["t"], support_ids=["s"], acquired_labels=1)
    assert paid["clean_low_n_eligible"] is False and "paid for twice" in paid["reasons"][0]
    transfer = fb.clean_low_n_eligibility(
        target="T", assay="A", feature_history=documented(entry(count=1, target="X", ids=["z"])),
        prior_history=label_free(), train_ids=["t"], support_ids=["s"], acquired_labels=1)
    assert transfer["clean_low_n_eligible"] is True and transfer["notes"]
    no_ids = fb.clean_low_n_eligibility(
        target="T", assay="A", feature_history=documented(entry(count=5, target="X")),
        prior_history=label_free(), train_ids=["t"], support_ids=["s"], acquired_labels=1)
    assert no_ids["clean_low_n_eligible"] is None
    mixed = fb.clean_low_n_eligibility(
        target="T", assay="A", feature_history=unknown_history(),
        prior_history=documented(entry(count=5)), train_ids=["t"], support_ids=["s"],
        acquired_labels=1)
    assert mixed["clean_low_n_eligible"] is False, "a definite violation outranks an unknown"


def test_sequence_only_exposure_is_recorded_separately_and_does_not_move_the_verdict():
    exposed = fb.SupervisionHistory(status="label_free", exposure_status="declared",
                                    exposure_description="pretrained on public sequences")
    verdict = fb.clean_low_n_eligibility(
        target="T", assay="A", feature_history=exposed, prior_history=label_free(),
        train_ids=["t"], support_ids=["s"], acquired_labels=1)
    assert verdict["clean_low_n_eligible"] is True
    assert verdict["sequence_only_exposure"]["features"]["status"] == "declared"
    assert verdict["sequence_only_exposure"]["prior"]["status"] == "none"


def test_the_pipeline_report_carries_the_verdict_for_the_declared_histories():
    problem = tiny_problem(prior=fb.PriorSpec(kind=fb.PRIOR_SCORE, semantics="scores",
                                              supervision=documented(entry(count=300))))
    artifact = fb.fit_and_allocate(problem, tiny_config())
    verdict = artifact["provenance"]["clean_low_n"]
    assert verdict["clean_low_n_eligible"] is False
    assert fb.evaluate_frozen(artifact, TRUTH)["clean_low_n"]["status"] == "ineligible"


@pytest.mark.parametrize("kwargs,message", [
    (dict(status="mystery"), "supervision status"),
    (dict(status="documented"), "at least one entry"),
    (dict(status="label_free", documented=(entry(count=1),)), "cannot list supervision"),
    (dict(status="documented", documented=(entry(count=2, ids=["a"]),)), "declares label_count=2"),
    (dict(status="documented", documented=(entry(count=2, ids=["a", "a"]),)), "repeat"),
    (dict(status="unknown", exposure_status="maybe"), "sequence_only_exposure")])
def test_supervision_histories_are_validated(kwargs, message):
    with pytest.raises(ValueError, match=message):
        fb.SupervisionHistory(**kwargs)


@pytest.mark.parametrize("override,message", [
    (dict(train_ids=("t0", "t1", "t2", "t3", "t4", "s0")), "overlap in candidate ids"),
    (dict(train_groups=("gt0", "gt1", "gt2", "gt3", "gt4", "gs0")), "share biological group"),
    (dict(train_ids=("t0", "t0", "t2", "t3", "t4", "t5")), "duplicate identifier"),
    (dict(train_utilities=[0.0, 1.0, float("nan"), 3.0, 4.0, 5.0]), "finite"),
    (dict(train_log_prior=[0.0, 0.0, float("nan"), 0.0, 0.0, 0.0]), "NaN"),
    (dict(support_log_prior=[0.0, float("inf"), 0.0, 0.0]), r"\+inf"),
    (dict(support_log_prior=[-math.inf] * 4), "zero mass"),
    (dict(train_sequences=("AA",) * 6, support_sequences=("AA", "AB", "AC", "AD")),
     "duplicate physical sequences"),
    (dict(train_sequences=("AA", "AB", "AC", "AD", "AE", "AF"),
          support_sequences=("ab", "XX", "YY", "ZZ")), "same physical sequence"),
    (dict(train_sequences=("AA",) * 6), "sequences for both")])
def test_the_problem_refuses_overlap_duplicates_and_nonfinite_values(override, message):
    with pytest.raises(ValueError, match=message):
        tiny_problem(**override)


def test_prior_semantics_are_validated_per_kind():
    assert fb.validate_prior_log_weights("uniform", [0.0, 0.0, 0.0]).tolist() == [0.0, 0.0, 0.0]
    for bad in ([0.0, -1.0], [0.0, -math.inf]):
        with pytest.raises(ValueError, match="identical finite"):
            fb.validate_prior_log_weights("uniform", bad)
    fb.validate_prior_log_weights("joint_log_probability",
                                  [math.log(0.5), math.log(0.25), -math.inf])
    with pytest.raises(ValueError, match="sum to"):
        fb.validate_prior_log_weights("joint_log_probability", [0.0, 0.0])
    fb.validate_prior_log_weights("score_normalized", [3.0, -math.inf, 1.0])
    for bad in ([0.0, float("nan")], [0.0, float("inf")]):
        with pytest.raises(ValueError):
            fb.validate_prior_log_weights("score_normalized", bad)
    with pytest.raises(ValueError, match="prior kind"):
        fb.validate_prior_log_weights("whatever", [0.0])
    with pytest.raises(ValueError, match="prior kind"):
        fb.PriorSpec(kind="whatever", semantics="x", supervision=label_free())
    with pytest.raises(TypeError):
        fb.PriorSpec(kind="uniform", semantics="x")


def test_the_utility_spec_declares_orientation_and_refuses_inferred_conversions():
    fb.UtilitySpec(units="u", scale="s", higher_is_better=True, useful_threshold=0.5)
    with pytest.raises(ValueError, match="higher_is_better=true"):
        fb.UtilitySpec(units="pKD", scale="s", higher_is_better=False, useful_threshold=0.5)
    with pytest.raises(ValueError, match="useful_threshold"):
        fb.UtilitySpec(units="u", scale="s", higher_is_better=True, useful_threshold=float("nan"))
    document = fb.UtilitySpec(units="u", scale="s", higher_is_better=True,
                              useful_threshold=0.5).document()
    assert "never fitted from evaluation quantiles" in document["threshold_rule"]


# ---------------------------------------------------------------------------
# CSV, outcome and manifest contracts
# ---------------------------------------------------------------------------

def test_the_candidate_reader_accepts_the_contract_and_returns_canonical_order(tmp_path):
    rows = list(reversed(GOOD_ROWS))
    table = fb.read_candidates(write_csv(tmp_path / "c.csv", CANDIDATE_HEADER, rows),
                               alphabet="ACD", length=2)
    assert table.ids == ("c1", "c2", "c3", "c4")
    assert table.splits == ("train", "train", "eval", "eval")
    assert table.prior_log_weight.tolist() == [0.0, -1.0, -math.inf, -0.5]
    assert not table.prior_log_weight.flags.writeable


@pytest.mark.parametrize("extra,message", [
    ("c1,DD,g9,train,0.0", "repeats a candidate_id"),
    ("c5,AC,g9,train,0.0", "same physical sequence"),
    ("c5,ac,g9,train,0.0", "same physical sequence"),
    ("c5,CA,g9,train,0.0", "train/evaluation overlap in canonical physical sequence"),
    ("c5,AX,g9,train,0.0", "outside the alphabet"),
    ("c5,ACD,g9,train,0.0", "length 3"),
    ("c5,DA,g1,eval,0.0", "appear in both"),
    ("c5,DA,g9,test,0.0", "split must be"),
    ("c5,DA,g9,train,nan", "NaN is not allowed"),
    ("c5,DA,g9,train,inf", "never allowed"),
    ("c5,DA,g9,train,abc", "is not a number"),
    ("c5,DA,g9,train,", "empty value"),
    ("c5,DA,,train,0.0", "empty group_id"),
    ("c5,DA,g9,train", "wrong number of fields")])
def test_the_candidate_reader_refuses_each_contract_violation(tmp_path, extra, message):
    path = write_csv(tmp_path / "c.csv", CANDIDATE_HEADER, GOOD_ROWS + [extra])
    with pytest.raises(ValueError, match=message):
        fb.read_candidates(path, alphabet="ACD", length=2)


def test_the_candidate_reader_refuses_a_changed_column_set(tmp_path):
    with_outcome = write_csv(tmp_path / "a.csv", CANDIDATE_HEADER + ",utility",
                             [row + ",1.0" for row in GOOD_ROWS])
    with pytest.raises(ValueError, match="unexpected column"):
        fb.read_candidates(with_outcome, alphabet="ACD", length=2)
    missing = write_csv(tmp_path / "b.csv", "candidate_id,sequence,group_id,split",
                        [row.rsplit(",", 1)[0] for row in GOOD_ROWS])
    with pytest.raises(ValueError, match="missing column"):
        fb.read_candidates(missing, alphabet="ACD", length=2)
    with pytest.raises(ValueError, match="does not exist"):
        fb.read_candidates(tmp_path / "nope.csv", alphabet="ACD", length=2)
    empty = write_csv(tmp_path / "e.csv", CANDIDATE_HEADER, [])
    with pytest.raises(ValueError, match="no rows"):
        fb.read_candidates(empty, alphabet="ACD", length=2)


def test_the_outcome_reader_refuses_repeats_nonfinite_values_and_extra_columns(tmp_path):
    good = write_csv(tmp_path / "o.csv", "candidate_id,utility", ["a,1.5", "b,-2"])
    assert fb.read_outcomes(good, label="outcomes") == {"a": 1.5, "b": -2.0}
    cases = {"dup": (["a,1.0", "a,2.0"], "two outcomes"), "nan": (["a,nan"], "NaN"),
             "inf": (["a,inf"], "not finite"), "neg": (["a,-inf"], "only allowed as an explicit")}
    for name, (rows, message) in cases.items():
        path = write_csv(tmp_path / f"{name}.csv", "candidate_id,utility", rows)
        with pytest.raises(ValueError, match=message):
            fb.read_outcomes(path, label="outcomes")
    extra = write_csv(tmp_path / "x.csv", "candidate_id,utility,group_id", ["a,1.0,g"])
    with pytest.raises(ValueError, match="unexpected column"):
        fb.read_outcomes(extra, label="outcomes")


def test_the_manifest_is_validated_and_infers_nothing():
    parsed = fb.parse_manifest(manifest_document())
    assert parsed.utility.useful_threshold == 1.0 and parsed.alphabet == "ACD"
    assert parsed.feature_supervision.status == "label_free"
    bad_orientation = manifest_document(utility={"units": "KD nM", "scale": "raw",
                                                 "higher_is_better": False,
                                                 "useful_threshold": 5.0})
    with pytest.raises(ValueError, match="higher_is_better=true"):
        fb.parse_manifest(bad_orientation)
    with pytest.raises(ValueError, match="missing"):
        fb.parse_manifest(manifest_document(utility={"units": "u", "scale": "s",
                                                     "higher_is_better": True}))
    with pytest.raises(ValueError, match="unknown key"):
        fb.parse_manifest({**manifest_document(), "extra": 1})
    with pytest.raises(ValueError, match="missing"):
        document = manifest_document()
        del document["grouping_rationale"]
        fb.parse_manifest(document)
    with pytest.raises(ValueError, match="sequence_onehot"):
        fb.parse_manifest(manifest_document(features={
            "kind": "learned", "alphabet": "ACD", "length": 2,
            "provenance": {"status": "label_free"}}))
    with pytest.raises(ValueError, match="at least one entry"):
        fb.parse_manifest(manifest_document(prior={
            "kind": "uniform", "semantics": "s", "provenance": {"status": "documented"}}))
    with pytest.raises(ValueError, match="prior kind"):
        fb.parse_manifest(manifest_document(prior={
            "kind": "mystery", "semantics": "s", "provenance": {"status": "label_free"}}))


def test_build_problem_enforces_the_training_label_contract(tmp_path):
    table = fb.read_candidates(write_csv(tmp_path / "c.csv", CANDIDATE_HEADER, GOOD_ROWS),
                               alphabet="ACD", length=2)
    manifest = fb.parse_manifest(manifest_document())
    problem = fb.build_problem(manifest, table, {"c1": 1.0, "c2": 2.0})
    assert problem.train_ids == ("c1", "c2") and problem.support_ids == ("c3", "c4")
    assert problem.space.kind == "sequence_onehot" and problem.train_features.shape == (2, 6)
    assert problem.train_features[0].tolist() == [1, 0, 0, 0, 1, 0], "A at pos0 and C at pos1"
    with pytest.raises(ValueError, match="evaluation-split candidate"):
        fb.build_problem(manifest, table, {"c1": 1.0, "c3": 2.0})
    with pytest.raises(ValueError, match="unknown candidate"):
        fb.build_problem(manifest, table, {"c1": 1.0, "zz": 2.0})
    with pytest.raises(ValueError, match="no rows"):
        fb.build_problem(manifest, table, {})
    uniform = fb.parse_manifest(manifest_document(prior={
        "kind": "uniform", "semantics": "flat", "provenance": {"status": "label_free"}}))
    with pytest.raises(ValueError, match="identical finite"):
        fb.build_problem(uniform, table, {"c1": 1.0, "c2": 2.0})


def test_one_hot_features_are_fixed_alphabet_and_fixed_length():
    space = fb.onehot_feature_space("ACD", 2, label_free())
    assert space.dimension == 6 and space.names[0] == "pos0=A" and space.names[-1] == "pos1=D"
    assert fb.onehot_encode(["ad", " CA "], space).tolist() == [
        [1, 0, 0, 0, 0, 1], [0, 1, 0, 1, 0, 0]]
    with pytest.raises(ValueError, match="outside the alphabet"):
        fb.onehot_encode(["AX"], space)
    with pytest.raises(ValueError, match="fixed length"):
        fb.onehot_encode(["ACD"], space)
    with pytest.raises(ValueError, match="distinct"):
        fb.onehot_feature_space("AAC", 2, label_free())


# ---------------------------------------------------------------------------
# the CSV pipeline and the CLI
# ---------------------------------------------------------------------------

def run_pipeline(tmp_path, *, eval_path=None, out="out", seed=0, **config):
    paths = fb.write_synthetic_inputs(tmp_path / "in", seed)
    settings = dict(tau=1.0, seed=seed, label_budget=30, sir_proposal_draws=128, final_draws=32)
    settings.update(config)
    summary = fb.run_finite_support(
        candidates=paths["candidates"], train_outcomes=paths["train_outcomes"],
        eval_outcomes=eval_path or paths["eval_outcomes"], manifest=paths["manifest"],
        output_dir=tmp_path / out, config=fb.BaselineConfig(**settings))
    return paths, summary


def test_the_evaluation_file_is_opened_only_after_the_frozen_artifact_exists(tmp_path,
                                                                            monkeypatch):
    seen = []
    original = fb.read_outcomes
    out = tmp_path / "out"

    def spy(path, *, label):
        seen.append((label, (out / fb.FROZEN_NAME).exists()))
        return original(path, label=label)
    monkeypatch.setattr(fb, "read_outcomes", spy)
    run_pipeline(tmp_path)
    assert seen == [("training outcomes", False), ("evaluation outcomes", True)]


def test_a_missing_evaluation_file_leaves_a_verified_frozen_artifact_that_never_saw_it(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        run_pipeline(tmp_path, eval_path=tmp_path / "absent.csv")
    frozen_path = tmp_path / "out" / fb.FROZEN_NAME
    assert frozen_path.is_file() and not (tmp_path / "out" / fb.REPORT_NAME).exists()
    assert fb.verify_artifact(fb.read_json(frozen_path)) is True
    assert fb.read_json(frozen_path)["provenance"]["evaluation_outcomes_used"] is False


def test_an_incomplete_evaluation_file_fails_and_never_renormalises(tmp_path):
    paths = fb.write_synthetic_inputs(tmp_path / "in")
    lines = paths["eval_outcomes"].read_text(encoding="utf-8").splitlines()
    short = tmp_path / "short.csv"
    short.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing for 1 of"):
        run_pipeline(tmp_path, eval_path=short)
    assert not (tmp_path / "out" / fb.REPORT_NAME).exists()


def test_different_evaluation_outcomes_give_a_byte_identical_frozen_artifact(tmp_path):
    paths, first = run_pipeline(tmp_path, out="first")
    shuffled = tmp_path / "shuffled.csv"
    fb._shuffled_outcomes(paths["eval_outcomes"], shuffled, 0)
    _, second = run_pipeline(tmp_path, out="second", eval_path=shuffled)
    assert first["artifact_sha256"] == second["artifact_sha256"]
    assert (tmp_path / "first" / fb.FROZEN_NAME).read_bytes() \
        == (tmp_path / "second" / fb.FROZEN_NAME).read_bytes()
    assert fb.read_json(tmp_path / "first" / fb.REPORT_NAME)["oracle"] \
        != fb.read_json(tmp_path / "second" / fb.REPORT_NAME)["oracle"]


def test_training_outcomes_naming_an_evaluation_candidate_are_refused(tmp_path):
    paths = fb.write_synthetic_inputs(tmp_path / "in")
    eval_line = paths["eval_outcomes"].read_text(encoding="utf-8").splitlines()[1]
    polluted = tmp_path / "polluted.csv"
    polluted.write_text(paths["train_outcomes"].read_text(encoding="utf-8") + eval_line + "\n",
                        encoding="utf-8")
    with pytest.raises(ValueError, match="evaluation-split candidate"):
        fb.run_finite_support(
            candidates=paths["candidates"], train_outcomes=polluted,
            eval_outcomes=paths["eval_outcomes"], manifest=paths["manifest"],
            output_dir=tmp_path / "out", config=fb.BaselineConfig(tau=1.0))
    assert not (tmp_path / "out" / fb.FROZEN_NAME).exists(), "nothing may be frozen from a bad join"


def test_an_existing_artifact_is_not_overwritten_without_permission(tmp_path):
    paths, _ = run_pipeline(tmp_path)
    with pytest.raises(ValueError, match="already exists"):
        run_pipeline(tmp_path)
    paths_again = dict(paths)
    summary = fb.run_finite_support(
        candidates=paths_again["candidates"], train_outcomes=paths_again["train_outcomes"],
        eval_outcomes=paths_again["eval_outcomes"], manifest=paths_again["manifest"],
        output_dir=tmp_path / "out", config=fb.BaselineConfig(tau=1.0, label_budget=30),
        overwrite=True)
    assert Path(summary["report"]).is_file()


def test_the_pipeline_reports_budget_support_zeros_dpo_exclusions_and_the_verdict(tmp_path):
    _, summary = run_pipeline(tmp_path)
    report = fb.read_json(summary["report"])
    accounting = report["label_accounting"]
    assert accounting["acquired_labels"] == 30 and accounting["offered_labels"] > 30
    assert report["split_counts"]["train_labelled_acquired"] == 30
    assert report["identities"]["all_hold"] is True and summary["all_identities_hold"] is True
    assert report["clean_low_n"]["status"] == "eligible"
    assert report["effective_data"]["ridge"]["labels_used"] == 30
    assert report["effective_data"]["dpo"]["pairs_used"] > 30, "pairs are not labels"
    assert report["manifest"]["utility"]["useful_threshold"] == 1.5
    assert report["support"]["candidates"] > report["support"]["structural_zeros"] >= 0
    json.dumps(report, allow_nan=False)


def test_the_cli_help_is_complete(capsys):
    cli = load_cli()
    with pytest.raises(SystemExit) as caught:
        cli.main(["--help"])
    assert caught.value.code == 0
    top = capsys.readouterr().out
    assert "selftest" in top and "finite-support" in top
    flags = {"selftest": ("--output-dir", "--seed"),
             "finite-support": ("--candidates", "--train-outcomes", "--eval-outcomes", "--manifest",
                                "--output-dir", "--tau", "--seed", "--ridge-alpha",
                                "--label-budget", "--declared-validation-labels", "--dpo-beta",
                                "--dpo-l2", "--dpo-max-pairs", "--dpo-max-iterations",
                                "--dpo-tolerance", "--dpo-pressures", "--sir-proposal",
                                "--sir-proposal-draws", "--final-draws", "--overwrite")}
    for command, names in flags.items():
        with pytest.raises(SystemExit) as caught:
            cli.main([command, "--help"])
        assert caught.value.code == 0
        text = capsys.readouterr().out
        for name in names:
            assert name in text, (command, name)
    assert "higher_is_better" in top and "frozen" in top


def test_the_cli_runs_the_pipeline_on_explicit_paths_and_prints_strict_json(tmp_path, capsys):
    cli = load_cli()
    paths = fb.write_synthetic_inputs(tmp_path / "in")
    argv = ["finite-support", "--candidates", str(paths["candidates"]),
            "--train-outcomes", str(paths["train_outcomes"]),
            "--eval-outcomes", str(paths["eval_outcomes"]), "--manifest", str(paths["manifest"]),
            "--output-dir", str(tmp_path / "out"), "--tau", "1.0", "--label-budget", "25",
            "--sir-proposal-draws", "64", "--final-draws", "16", "--dpo-pressures", "0.5", "2"]
    assert cli.main(argv) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["all_identities_hold"] is True and len(summary["artifact_sha256"]) == 64
    report = fb.read_json(tmp_path / "out" / fb.REPORT_NAME)
    assert [row["pressure"] for row in report["dpo"]["pressure_curve"]] == [0.5, 2.0]
    assert report["label_accounting"]["acquired_labels"] == 25
    assert cli.main(argv) == 2, "an existing artifact is refused without --overwrite"
    assert "already exists" in capsys.readouterr().err
    assert cli.main(argv + ["--overwrite"]) == 0


def test_the_cli_reports_a_missing_evaluation_file_as_an_error_after_freezing(tmp_path, capsys):
    cli = load_cli()
    paths = fb.write_synthetic_inputs(tmp_path / "in")
    argv = ["finite-support", "--candidates", str(paths["candidates"]),
            "--train-outcomes", str(paths["train_outcomes"]),
            "--eval-outcomes", str(tmp_path / "absent.csv"), "--manifest", str(paths["manifest"]),
            "--output-dir", str(tmp_path / "out"), "--tau", "1.0"]
    assert cli.main(argv) == 2
    assert "does not exist" in capsys.readouterr().err
    assert (tmp_path / "out" / fb.FROZEN_NAME).is_file()
    with pytest.raises(SystemExit):
        cli.main(argv[:-2])   # --tau is required: it is a scientific choice, not a default


def test_the_selftest_passes_every_named_check_and_writes_a_real_strict_report(tmp_path, capsys):
    cli = load_cli()
    out = tmp_path / "selftest"
    assert cli.main(["selftest", "--output-dir", str(out), "--seed", "0"]) == 0
    printed = capsys.readouterr().out
    assert "selftest PASSED" in printed and "!!" not in printed
    report_path = out / "selftest_report.json"
    text = report_path.read_text(encoding="utf-8")
    report = json.loads(text, parse_constant=lambda token: pytest.fail(f"literal {token}"))
    assert report["passed"] is True and all(check["holds"] for check in report["checks"])
    names = {check["name"] for check in report["checks"]}
    assert {"planted::additive_predictor_misses_the_context_reversal",
            "planted::interaction_ridge_learns_the_reversal",
            "planted::restoration_rescues_the_ablation",
            "pipeline::evaluation_outcomes_cannot_change_the_frozen_artifact",
            "pipeline::a_tampered_artifact_is_refused"} <= names
    assert "planted::donor_patch_flips_the_local_effect_and_removes_the_cross_context_reversal" \
        in names
    assert "planted::donor_patch_introduces_the_reversal" not in names
    assert report["planted_scope"]["unique_feature_contexts"] == 8
    assert report["planted_scope"]["training_rows"] == 24
    assert "not unseen-context generalisation" in report["note"]
    for relative in ("planted/additive_frozen_artifact.json",
                     "planted/interaction_frozen_artifact.json",
                     "finite_support/frozen_artifact.json", "finite_support/evaluation_report.json",
                     "synthetic/candidates.csv", "synthetic/manifest.json"):
        assert (out / relative).is_file(), relative
    for relative in ("planted/interaction_frozen_artifact.json", "finite_support/frozen_artifact.json"):
        assert fb.verify_artifact(fb.read_json(out / relative)) is True


# ---------------------------------------------------------------------------
# the planted causal-learning experiment
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def planted():
    return fb.run_planted_experiment(seed=0, permutations=60)


def test_an_additive_predictor_misses_the_context_reversal_and_interaction_ridge_learns_it(planted):
    effects = planted["a_effects_by_context"]
    assert effects["truth"] == {"b0": 1.0, "b1": -1.0}
    assert abs(effects["additive"]["b0"]) < 1e-6 and abs(effects["additive"]["b1"]) < 1e-6
    assert planted["additive_reversal_present"] is False
    assert effects["interaction"]["b0"] == pytest.approx(1.0, abs=1e-4)
    assert effects["interaction"]["b1"] == pytest.approx(-1.0, abs=1e-4)
    assert planted["interaction_reversal_present"] is True


def test_the_interaction_predictor_ranks_and_allocates_better_on_the_heldout_panel(planted):
    ranking = planted["prediction_ranking"]
    interaction = ranking["interaction"]["all_candidates"]
    additive = ranking["additive"]["all_candidates"]
    assert interaction["spearman"]["value"] > 0.9
    assert additive["spearman"]["value"] == pytest.approx(0.447, abs=0.01)
    assert interaction["concordance"]["value"] == pytest.approx(1.0)
    regret = {name: planted["policy_summary"][name]["ridge_tilt"]["oracle_regret"]["value"]
              for name in ("additive", "interaction")}
    assert regret["interaction"] < 1e-6 < regret["additive"]
    assert all(planted["identities_hold"].values())
    for name in ("additive", "interaction"):
        assert planted["dpo"][name]["excluded_zero_prior_labels"] == 1


def test_interventions_applied_before_prediction_show_necessity_sufficiency_and_rescue(planted):
    iv = planted["interventions"]
    assert iv["identity"]["bitwise_equal"] is True
    ablate = iv["ablate_interaction"]
    assert ablate["reversal_present"] is False
    assert ablate["a_effects"]["b0"] == pytest.approx(1.0, abs=1e-4)
    assert ablate["a_effects"]["b1"] == pytest.approx(1.0, abs=1e-4)
    nuisance = iv["ablate_nuisance"]
    assert nuisance["max_abs_prediction_change"] < 1e-9 and nuisance["reversal_present"] is True
    patch = iv["donor_patch_sufficiency"]
    assert "reversal_introduced" not in patch, "the old label named the opposite of the result"
    assert patch["local_negative_effect_introduced"] is True
    assert patch["a_effect_in_context_b0_before"] == pytest.approx(1.0, abs=1e-4)
    assert patch["a_effect_in_context_b0_after"] == pytest.approx(-1.0, abs=1e-4)
    assert patch["cross_context_reversal_before"] is True
    assert patch["cross_context_reversal_after"] is False, "a is now negative in BOTH contexts"
    assert patch["cross_context_reversal_removed"] is True
    assert patch["prediction_change"] == pytest.approx(-2.0, abs=1e-4)
    assert patch["prediction_change"] == pytest.approx(patch["interaction_weight_times_delta_ab"],
                                                       abs=1e-9)
    assert abs(patch["nuisance_patch_prediction_change"]) < 1e-9
    rescue = iv["rescue_after_ablation"]
    assert rescue["bitwise_equal_to_clean"] is True and rescue["max_abs_difference_from_clean"] == 0.0
    assert rescue["reversal_present"] is True
    assert "not a molecular mechanism" in iv["claim"]


def test_raw_predictions_ignore_unrelated_batch_composition_even_under_ablation(planted):
    independence = planted["batch_independence"]
    assert independence["batch_rows"] == 9
    for key in ("max_abs_difference_alone_vs_in_batch",
                "max_abs_difference_ablated_alone_vs_in_batch",
                "max_abs_difference_after_reordering"):
        assert independence[key] < 1e-12, key


def test_a_random_label_policy_is_judged_by_a_permutation_null_not_a_zero_gain_claim(planted):
    check = planted["permutation_check"]
    assert check["permutations"] == 60 and check["permuted_at_least_real"] <= 1
    assert check["permutation_p_value"] <= 0.05
    assert check["real_gain"] > check["permuted_gain_max"]
    assert check["permuted_gain_max"] - check["permuted_gain_min"] > 1e-6, (
        "realised random-label gains vary; they are not exactly zero")
    assert abs(check["permuted_prediction_spearman_mean"]) < 0.3
    assert check["real_prediction_spearman"] > 0.9
    assert "not a confidence interval" in check["note"]


def test_the_planted_design_is_balanced_so_the_nuisance_weight_is_zero():
    base = fb.planted_space()
    problem, truth = fb.planted_problem(base)
    model = fb.fit_reward(problem.train_features, problem.train_utilities, base, alpha=1e-6)
    weights = dict(zip(base.names, model.weights.tolist()))
    assert abs(weights["nuisance"]) < 1e-9
    assert weights["a"] == pytest.approx(1.0, abs=1e-4) and weights["ab"] == pytest.approx(-2.0, abs=1e-4)
    assert set(truth) == set(problem.support_ids) and len(truth) == 16
    sub, _ = base.select(("a", "b", "nuisance"))
    assert sub.names == ("a", "b", "nuisance")


# ---------------------------------------------------------------------------
# bounded correctness repair: label identity and cost
# ---------------------------------------------------------------------------

def cost_of(**overrides):
    base = dict(target="T", assay="A", feature_history=label_free(), prior_history=label_free(),
                train_ids=["t0", "t1", "t2"], support_ids=["s0"], acquired_labels=2,
                acquired_ids=["t0", "t1"])
    base.update(overrides)
    return fb.clean_low_n_eligibility(**base)["label_cost"]


def staged(problem, config):
    fit = fb.fit_stage(problem, config)
    allocation = fb.allocate(problem.support_ids, problem.support_log_prior, fit.predictions,
                             config, dpo_energy=fit.dpo_energy)
    return fit, allocation


def test_label_ids_shared_across_histories_and_the_acquisition_are_unioned_never_summed():
    verdict = fb.clean_low_n_eligibility(
        target="T", assay="A", feature_history=documented(entry(count=2, ids=["a", "b"])),
        prior_history=documented(entry(count=2, ids=["a", "b"])), train_ids=["t0", "t1"],
        support_ids=["s0"], acquired_labels=2, acquired_ids=["t0", "t1"])
    cost = verdict["label_cost"]
    unique = cost["unique_same_target_labels"]
    assert unique["status"] == "exact" and unique["exact"] == 4, "{a, b, t0, t1}, not 2 + 2 + 2"
    uses = cost["declared_label_uses"]
    assert (uses["acquired_current"], uses["feature_history_same_target"],
            uses["prior_history_same_target"]) == (2, 2, 2)
    assert "documented_effective_target_labels" not in cost
    assert "documented_same_target_historical_labels" not in cost
    assert verdict["clean_low_n_eligible"] is False
    same = fb.clean_low_n_eligibility(
        target="T", assay="A", feature_history=documented(entry(count=2, ids=["t0", "t1"])),
        prior_history=documented(entry(count=2, ids=["t0", "t1"])), train_ids=["t0", "t1"],
        support_ids=["s0"], acquired_labels=2, acquired_ids=["t0", "t1"])["label_cost"]
    assert same["unique_same_target_labels"]["exact"] == 2, "the same two labels are not six"
    assert same["declared_label_uses"]["feature_history_same_target"] == 2


def test_unique_totals_are_bounds_or_unavailable_where_identities_are_missing():
    hidden = cost_of(prior_history=documented(entry(count=5)))["unique_same_target_labels"]
    assert hidden["exact"] is None and hidden["status"] == "bounded_only"
    assert (hidden["lower_bound"], hidden["upper_bound"]) == (5, 7), "2 acquired ids, 5 hidden"
    unknown_cost = cost_of(feature_history=unknown_history())
    unknown = unknown_cost["unique_same_target_labels"]
    assert unknown["status"] == "lower_bound_only" and unknown["exact"] is None
    assert unknown["upper_bound"] is None and unknown["lower_bound"] == 2
    assert unknown_cost["unknown_history_present"] is True
    assert unknown_cost["unknown_history_sources"] == ["feature extractor"]
    anonymous = cost_of(acquired_ids=None,
                        prior_history=documented(entry(count=3, ids=["a", "b", "c"])))
    assert anonymous["acquired_ids_supplied"] is False
    bounds = anonymous["unique_same_target_labels"]
    assert bounds["exact"] is None and (bounds["lower_bound"], bounds["upper_bound"]) == (3, 5)
    lone = cost_of(acquired_ids=None)["unique_same_target_labels"]
    assert lone["exact"] == 2, "one component with no ids is still the whole set"


def test_an_unknown_history_stays_unknown_even_when_a_known_overlap_already_makes_it_false():
    verdict = fb.clean_low_n_eligibility(
        target="T", assay="A", feature_history=unknown_history(),
        prior_history=documented(entry(count=1, ids=["t0"])), train_ids=["t0"],
        support_ids=["s0"], acquired_labels=1, acquired_ids=["t0"])
    assert verdict["clean_low_n_eligible"] is False
    cost = verdict["label_cost"]
    assert cost["unknown_history_present"] is True
    assert cost["unknown_history_sources"] == ["feature extractor"]
    assert cost["unique_same_target_labels"]["exact"] is None
    assert cost["unique_same_target_labels"]["status"] == "lower_bound_only"


def test_the_same_id_on_another_target_is_not_the_same_label_but_the_overlap_check_stays():
    verdict = fb.clean_low_n_eligibility(
        target="T", assay="A",
        feature_history=documented(entry(count=1, target="OTHER", ids=["t0"])),
        prior_history=label_free(), train_ids=["t0", "t1"], support_ids=["s0"],
        acquired_labels=2, acquired_ids=["t0", "t1"])
    cost = verdict["label_cost"]
    assert cost["unique_same_target_labels"]["exact"] == 2, "OTHER's t0 is another measured outcome"
    uses = cost["declared_label_uses"]
    assert uses["feature_history_other_target_or_assay"] == 1
    assert uses["feature_history_same_target"] == 0
    assert verdict["clean_low_n_eligible"] is False
    assert any("paid for twice" in reason for reason in verdict["reasons"])
    assert "(target, assay, label id)" in cost["identity_convention"]


def test_declared_validation_cost_is_separate_and_disjointness_is_not_asserted():
    cost = cost_of(declared_validation_labels=3)
    validation = cost["validation"]
    assert validation["declared_labels"] == 3 and validation["ids_supplied"] is False
    assert validation["disjointness_asserted"] is False
    assert cost["unique_same_target_labels"]["exact"] == 2, "validation is never added in"
    including = validation["unique_same_target_labels_including_validation"]
    assert including["exact"] is None
    assert (including["lower_bound"], including["upper_bound"]) == (3, 5)
    assert cost_of()["validation"]["unique_same_target_labels_including_validation"]["exact"] == 2


def test_the_artifact_tracks_acquired_identities_for_the_unique_total():
    prior = fb.PriorSpec(kind=fb.PRIOR_SCORE, semantics="scores",
                         supervision=documented(entry(count=2, ids=["h0", "h1"])))
    artifact = fb.fit_and_allocate(tiny_problem(prior=prior),
                                   tiny_config(label_budget=4, declared_validation_labels=1))
    cost = artifact["provenance"]["clean_low_n"]["label_cost"]
    assert cost["acquired_ids_supplied"] is True
    unique = cost["unique_same_target_labels"]
    assert unique["status"] == "exact" and unique["exact"] == 6, "4 acquired + 2 historical"
    assert cost["validation"]["declared_labels"] == 1


# ---------------------------------------------------------------------------
# bounded correctness repair: snapshots and the fit-time binding
# ---------------------------------------------------------------------------

def test_editing_caller_owned_metadata_after_construction_or_fit_cannot_rewrite_provenance():
    schema = {"columns": ["x1", "x2"], "note": "original"}
    feature_history = documented(entry(count=2, target="OTHER", ids=["h0", "h1"]))
    space = fb.FeatureSpace(kind="explicit", names=("x1", "x2"), schema=schema,
                            history=("explicit columns",), supervision=feature_history)
    prior_history = documented(entry(count=1, target="OTHER", ids=["p0"]))
    prior = fb.PriorSpec(kind=fb.PRIOR_SCORE, semantics="scores", supervision=prior_history)
    problem = tiny_problem(space=space, prior=prior)
    assert problem.space is not space and problem.prior is not prior
    assert space.supervision is not feature_history and prior.supervision is not prior_history
    config = tiny_config()
    fit, allocation = staged(problem, config)
    before = fb.freeze_artifact(problem, config, fit, allocation)
    schema["note"] = "edited source dict"
    space.schema["note"] = "edited space"
    space.schema["columns"].append("sneaky")
    feature_history.documented[0]["label_ids"].append("sneaky")
    feature_history.documented[0]["label_count"] = 99
    prior_history.documented[0]["target"] = "T"
    assert problem.space.schema == {"columns": ["x1", "x2"], "note": "original"}
    assert fb.freeze_artifact(problem, config, fit, allocation) == before
    assert fb.fit_and_allocate(problem, config) == before
    documented_features = before["features"]["space"]["provenance"]["documented_supervision"]
    assert documented_features[0]["label_count"] == 2
    assert documented_features[0]["label_ids"] == ["h0", "h1"]
    assert before["prior"]["spec"]["provenance"]["documented_supervision"][0]["target"] == "OTHER"
    assert before["features"]["space"]["schema"]["note"] == "original"


def test_a_problem_or_config_that_changed_after_the_fit_is_refused_at_freeze():
    problem, config = tiny_problem(), tiny_config()
    fit, allocation = staged(problem, config)
    assert fb.freeze_artifact(problem, config, fit, allocation)["schema"] == fb.ARTIFACT_SCHEMA
    problem.space.schema["injected"] = "after the fit"
    with pytest.raises(ValueError, match="edited after the fit"):
        fb.freeze_artifact(problem, config, fit, allocation)
    fresh = tiny_problem()
    fresh_fit, fresh_allocation = staged(fresh, config)
    with pytest.raises(ValueError, match="different BaselineConfig"):
        fb.freeze_artifact(fresh, tiny_config(ridge_alpha=0.2), fresh_fit, fresh_allocation)
    other = tiny_problem(train_utilities=[0.0, 1.0, 2.0, 3.0, 4.0, 6.0])
    with pytest.raises(ValueError, match="different problem inputs"):
        fb.freeze_artifact(other, config, fresh_fit, fresh_allocation)


# ---------------------------------------------------------------------------
# bounded correctness repair: label uses
# ---------------------------------------------------------------------------

def test_dpo_label_uses_are_instrumented_and_reported_apart_from_acquired_labels():
    artifact = fb.fit_and_allocate(tiny_problem(), tiny_config())
    accounting = artifact["provenance"]["label_accounting"]
    diagnostics = artifact["fitted"]["dpo"]["diagnostics"]
    evaluations = diagnostics["evaluations"]
    assert evaluations["objective"] >= 2 and evaluations["gradient"] >= 2
    assert evaluations["hessian"] >= 1 and evaluations["difference_matrix_builds"] == 1
    passes = 1 + evaluations["objective"] + evaluations["gradient"] + evaluations["hessian"]
    assert diagnostics["full_pair_passes"] == passes
    assert diagnostics["pair_endpoint_presentations"] == 2 * diagnostics["pairs"] * passes
    uses = accounting["label_uses"]
    assert uses["independent_labels"] == accounting["acquired_labels"] == 6
    assert accounting["repeated_presentations"] == uses["repeated_uses_beyond_first"] \
        == diagnostics["pair_endpoint_presentations"] > 0
    assert uses["total_label_uses"] == 6 + accounting["repeated_presentations"]
    assert uses["dpo"]["status"] == "instrumented"
    tied = fb.fit_and_allocate(tiny_problem(train_utilities=[2.0] * 6), tiny_config())
    none = tied["provenance"]["label_accounting"]
    assert none["label_uses"]["dpo"]["status"] == "not_fit"
    assert none["repeated_presentations"] == 0, "no DPO fit, so a true (not unconditional) zero"
    assert none["acquired_labels"] == 6


# ---------------------------------------------------------------------------
# bounded correctness repair: the artifact binds the complete offered inputs
# ---------------------------------------------------------------------------

def test_changing_an_unselected_offered_row_changes_the_provenance_hash_but_not_the_fit():
    config = tiny_config(label_budget=3, seed=3)
    chosen = set(fb.acquire_labels(6, 3, 3).tolist())
    k = min(set(range(6)) - chosen)
    sequences = dict(train_sequences=("AA", "AB", "AC", "AD", "AE", "AF"),
                     support_sequences=("BA", "BB", "BC", "BD"))
    base_kwargs = problem_kwargs()
    base = fb.fit_and_allocate(tiny_problem(**sequences), config)
    assert base["training"]["acquired_indices"] == sorted(chosen)
    assert base["training"]["offered_ids"] == [f"t{i}" for i in range(6)]
    assert [base["training"]["offered_ids"][i] for i in base["training"]["acquired_indices"]] \
        == base["training"]["ids"]
    utilities = list(base_kwargs["train_utilities"])
    utilities[k] += 10.0
    priors = list(base_kwargs["train_log_prior"])
    priors[k] = -2.0
    features = [list(row) for row in base_kwargs["train_features"]]
    features[k] = [7.0, 7.0]
    groups = list(base_kwargs["train_groups"])
    groups[k] = "g-other"
    train_sequences = list(sequences["train_sequences"])
    train_sequences[k] = "ZZ"
    edits = {"outcome": dict(train_utilities=utilities), "prior": dict(train_log_prior=priors),
             "features": dict(train_features=features), "group": dict(train_groups=tuple(groups)),
             "sequence": dict(train_sequences=tuple(train_sequences))}
    for name, edit in edits.items():
        changed = fb.fit_and_allocate(tiny_problem(**{**sequences, **edit}), config)
        assert changed["hashes"]["artifact_sha256"] != base["hashes"]["artifact_sha256"], name
        assert changed["hashes"]["provenance"] != base["hashes"]["provenance"], name
        assert changed["predictions"] == base["predictions"], f"{name} leaked into the fit"
        assert changed["policies"] == base["policies"], f"{name} leaked into the allocation"
        assert changed["training"]["ids"] == base["training"]["ids"], name
        assert fb.verify_artifact(changed) is True, name
    support = list(sequences["support_sequences"])
    support[0] = "YY"
    moved = fb.fit_and_allocate(tiny_problem(train_sequences=sequences["train_sequences"],
                                             support_sequences=tuple(support)), config)
    assert moved["support"]["sequences_sha256"] != base["support"]["sequences_sha256"]
    assert moved["hashes"]["artifact_sha256"] != base["hashes"]["artifact_sha256"]
    assert moved["predictions"] == base["predictions"]
    assert base["support"]["sequences_sha256"] is not None


def test_rehashed_edits_of_the_roster_the_selection_or_the_binding_are_still_refused(frozen):
    def forge(edit, document=frozen):
        forged = copy.deepcopy(document)
        edit(forged)
        forged["hashes"] = fb._section_hashes(forged)
        return forged
    with pytest.raises(ValueError, match="acquired ids are not the offered roster"):
        fb.verify_artifact(forge(lambda doc: doc["training"]["offered_ids"].__setitem__(0, "t0x")))
    with pytest.raises(ValueError, match="selection rule disagrees"):
        fb.verify_artifact(forge(lambda doc: doc["training"]["selection"].update(seed=99)))
    with pytest.raises(ValueError, match="problem digest does not match"):
        fb.verify_artifact(forge(lambda doc: doc["provenance"]["input_binding"]["components"][
            "train"].update(utilities_sha256="0" * 64)))
    budgeted = fb.fit_and_allocate(tiny_problem(), tiny_config(label_budget=3))
    real = budgeted["training"]["acquired_indices"]
    other = [i for i in range(6) if i not in real]

    def swap(doc):
        doc["training"]["acquired_indices"] = other
        doc["training"]["ids"] = [doc["training"]["offered_ids"][i] for i in other]
    with pytest.raises(ValueError, match="seeded selection"):
        fb.verify_artifact(forge(swap, budgeted))
    assert fb.verify_artifact(budgeted) is True


def test_unlabelled_training_rows_are_bound_into_the_artifact(tmp_path):
    paths = fb.write_synthetic_inputs(tmp_path / "in")
    lines = paths["train_outcomes"].read_text(encoding="utf-8").splitlines()
    dropped = lines[1].split(",")[0]
    trimmed = tmp_path / "train_trimmed.csv"
    trimmed.write_text("\n".join(lines[:1] + lines[2:]) + "\n", encoding="utf-8")

    def run(candidates, out):
        return fb.run_finite_support(
            candidates=candidates, train_outcomes=trimmed, eval_outcomes=paths["eval_outcomes"],
            manifest=paths["manifest"], output_dir=tmp_path / out,
            config=fb.BaselineConfig(tau=1.0, label_budget=20, sir_proposal_draws=64,
                                     final_draws=16))
    base = run(paths["candidates"], "base")
    artifact = fb.read_json(tmp_path / "base" / fb.FROZEN_NAME)
    counts = artifact["provenance"]["split_counts"]
    assert counts["train_candidates_offered"] == counts["train_labelled_offered"] + 1
    components = artifact["provenance"]["input_binding"]["components"]
    assert components["unlabelled_train"]["count"] == 1
    assert components["candidate_roster"]["count"] > counts["train_labelled_offered"]
    rows = paths["candidates"].read_text(encoding="utf-8").splitlines()
    rows = [line.rsplit(",", 1)[0] + ",-30.0" if line.startswith(f"{dropped},") else line
            for line in rows]
    edited = tmp_path / "candidates_edited.csv"
    edited.write_text("\n".join(rows) + "\n", encoding="utf-8")
    changed = run(edited, "changed")
    assert changed["artifact_sha256"] != base["artifact_sha256"]
    other = fb.read_json(tmp_path / "changed" / fb.FROZEN_NAME)
    assert other["predictions"] == artifact["predictions"], "the unlabelled row never entered a fit"


# ---------------------------------------------------------------------------
# bounded correctness repair: the strict-pair sampler
# ---------------------------------------------------------------------------

def brute_force_strict_pairs(values):
    """``{(lower-outcome index, higher-outcome index)}`` by direct enumeration."""
    values = list(values)
    return {(a, b) if values[a] < values[b] else (b, a)
            for a in range(len(values)) for b in range(a + 1, len(values))
            if values[a] != values[b]}


@pytest.mark.parametrize("seed", range(5))
def test_the_rank_mapping_numbers_every_strict_pair_exactly_once(seed):
    rng = np.random.default_rng(seed)
    values = rng.integers(0, 4, size=14).astype(float)
    universe = fb._strict_pair_universe(values)
    total = int(universe[2][-1])
    losers, winners = fb._pairs_from_ranks(np.arange(total), universe)
    produced = list(zip(losers.tolist(), winners.tolist()))
    assert len(produced) == len(set(produced)) == total
    assert set(produced) == brute_force_strict_pairs(values.tolist())
    assert all(values[w] > values[l] for l, w in produced)


def test_a_tie_heavy_panel_of_999_ties_and_one_positive_samples_without_exhaustion():
    utilities = np.zeros(1000)
    utilities[417] = 1.0
    flags = np.ones(1000, dtype=bool)
    pairs = fb.preference_pairs(utilities, flags, max_pairs=100, seed=0)
    diagnostics = pairs.diagnostics
    assert diagnostics["eligible_pairs"] == 999 and diagnostics["pairs_used"] == 100
    assert diagnostics["ties_excluded"] == 498501 and diagnostics["subsampled"] is True
    assert diagnostics["method"] == "seeded_uniform_rank_sample"
    assert set(pairs.winners.tolist()) == {417}
    losers = pairs.losers.tolist()
    assert len(set(losers)) == 100 and 417 not in losers, "no replacement, no tie pair"
    assert fb.preference_pairs(utilities, flags, max_pairs=100, seed=0).losers.tolist() == losers
    assert fb.preference_pairs(utilities, flags, max_pairs=100, seed=1).losers.tolist() != losers
    every = fb.preference_pairs(utilities, flags, max_pairs=999, seed=0)
    assert every.diagnostics["method"] == "all_eligible_pairs"
    assert every.diagnostics["pairs_used"] == 999
    seen = set()
    for seed in range(300):
        seen.update(fb.preference_pairs(utilities, flags, max_pairs=100, seed=seed).losers.tolist())
    assert len(seen) == 999, "every one of the 999 strict pairs is reachable across seeds"


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_sparse_tie_heavy_panels_exclude_unsupported_labels_and_draw_without_replacement(seed):
    rng = np.random.default_rng(100 + seed)
    values = rng.choice([0.0, 0.0, 0.0, 1.0, 2.5], size=60)
    supported = rng.random(60) > 0.2
    pairs = fb.preference_pairs(values, supported, max_pairs=40, seed=seed)
    usable = np.flatnonzero(supported)
    universe = {(int(usable[l]), int(usable[w]))
                for l, w in brute_force_strict_pairs([values[i] for i in usable])}
    produced = list(zip(pairs.losers.tolist(), pairs.winners.tolist()))
    assert len(produced) == len(set(produced)) == min(40, len(universe))
    assert set(produced) <= universe
    assert pairs.diagnostics["eligible_pairs"] == len(universe)
    used = set(pairs.winners.tolist()) | set(pairs.losers.tolist())
    assert not used & set(np.flatnonzero(~supported).tolist()), "a zero-prior label was paired"
    again = fb.preference_pairs(values, supported, max_pairs=40, seed=seed)
    assert again.winners.tolist() == pairs.winners.tolist()


# ---------------------------------------------------------------------------
# bounded correctness repair: lossless DPO pressure names
# ---------------------------------------------------------------------------

def test_distinct_dpo_pressures_keep_distinct_lossless_names_through_every_stage(tmp_path):
    close = (1.0, 1.0000001)
    assert fb.pressure_name(1.0) == "dpo_pressure_1"
    assert fb.pressure_name(1.0) != fb.pressure_name(1.0000001)
    for pressure in (0.25, 1.0, 1.0000001, 3.0e-7, 12345.678, 4.0):
        text = fb.pressure_name(pressure).removeprefix("dpo_pressure_")
        assert float(text) == pressure, pressure
    artifact = fb.fit_and_allocate(tiny_problem(), tiny_config(dpo_pressures=close))
    names = [fb.pressure_name(pressure) for pressure in close]
    assert len(set(names)) == 2 and set(names) <= set(artifact["policies"]["log_probability"])
    assert fb.verify_artifact(artifact) is True
    fb.write_json(tmp_path / "frozen.json", artifact)
    assert fb.verify_artifact(fb.read_json(tmp_path / "frozen.json")) is True
    report = fb.evaluate_frozen(artifact, TRUTH)
    assert set(names) <= set(report["policies"])
    curve = report["dpo"]["pressure_curve"]
    assert [row["pressure"] for row in curve] == list(close)
    assert {row["policy"] for row in curve} == set(names)
    assert {row["policy"] for row in report["comparison"]} >= set(names)


# ---------------------------------------------------------------------------
# bounded correctness repair: the common-cohort policy ranking
# ---------------------------------------------------------------------------

def policy_report(q, p, u, tau=1.0):
    with np.errstate(divide="ignore"):
        log_q = np.log(np.asarray(q, dtype=float))
        log_p = fb.normalize_log_weights(np.log(np.asarray(p, dtype=float)))
    u = np.asarray(u, dtype=float)
    log_star = fb.log_tilt(log_p, u, tau)
    return fb._policy_report(log_q, log_p=log_p, utilities=u, tau=tau, draws=5, threshold=0.5,
                             log_star=log_star, best=fb.objective(log_star, log_p, u, tau))


def test_primary_ranking_uses_the_common_prior_reachable_cohort_with_zero_mass_tied_last():
    ranking = policy_report([0.9, 0.1, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 2.0])["ranking"]
    assert ranking["cohort"]["candidates"] == 3
    assert ranking["cohort"]["policy_zero_mass_in_cohort"] == 1
    for key in ("policy_log_probability_vs_utility", "policy_log_ratio_to_prior_vs_utility"):
        assert ranking[key]["concordance"]["value"] == pytest.approx(1.0 / 3.0), key
        assert ranking[key]["concordance"]["comparable_pairs"] == 3, key
        assert ranking[key]["spearman"]["value"] == pytest.approx(-0.5), key
    conditional = ranking["conditional_on_policy_support"]
    assert conditional["candidates"] == 2
    assert conditional["policy_log_probability_vs_utility"]["spearman"]["value"] == pytest.approx(1.0)
    assert "never the headline" in conditional["definition"]


def test_the_ratio_ranking_excludes_prior_zero_candidates_but_keeps_policy_zero_ones():
    ranking = policy_report([0.5, 0.5, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0],
                            [1.0, 0.5, 0.0, 9.0])["ranking"]
    assert ranking["cohort"]["candidates"] == 3, "the prior-zero candidate is outside the cohort"
    assert ranking["cohort"]["policy_zero_mass_in_cohort"] == 1
    for key in ("policy_log_probability_vs_utility", "policy_log_ratio_to_prior_vs_utility"):
        assert ranking[key]["spearman"]["value"] == pytest.approx(1.5 / math.sqrt(3.0)), key
        assert ranking[key]["concordance"]["value"] == pytest.approx(2.5 / 3.0), key


def test_ranking_keeps_underflowed_finite_logs_ordered_and_ties_neg_inf_by_equality():
    scores = np.array([-2000.0, -1000.0, -math.inf, -math.inf, -3000.0])
    outcomes = np.array([2.0, 3.0, 0.0, 1.0, 1.5])
    assert fb.spearman(scores, outcomes)["value"] == pytest.approx(9.5 / math.sqrt(95.0))
    result = fb.concordance(scores, outcomes)
    assert result["value"] == pytest.approx(0.95)
    assert result["comparable_pairs"] == 10 and result["score_tied_pairs"] == 1
    assert fb._average_ranks(scores).tolist() == [4.0, 5.0, 1.5, 1.5, 3.0]
    for bad in (np.array([0.0, np.inf]), np.array([0.0, np.nan])):
        with pytest.raises(ValueError, match=r"NaN or \+inf"):
            fb.spearman(bad, [1.0, 2.0])
        with pytest.raises(ValueError, match=r"NaN or \+inf"):
            fb.concordance(bad, [1.0, 2.0])


def test_a_sparse_policy_no_longer_ranks_perfectly_in_the_report(frozen):
    report = fb.evaluate_frozen(frozen, TRUTH)
    for name, entry in report["policies"].items():
        ranking = entry["ranking"]
        assert ranking["cohort"]["candidates"] == 3, name
        assert {"conditional_on_policy_support", "cohort"} <= set(ranking), name


# ---------------------------------------------------------------------------
# bounded correctness repair: the planted toy
# ---------------------------------------------------------------------------

REASON = "the scores are constant, so the rank correlation is undefined"


def test_an_undefined_permutation_spearman_stays_none_with_counts_and_a_conditional_mean(
        monkeypatch):
    problem, _ = fb.planted_problem(fb.planted_space())
    config = fb.BaselineConfig(tau=0.5, seed=0, ridge_alpha=1e-6)
    calls = {"n": 0}

    def scripted(scores, outcomes):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"value": 0.9, "n": 16, "reason": None}
        if calls["n"] % 2 == 0:
            return {"value": None, "n": 16, "reason": REASON}
        return {"value": 0.2, "n": 16, "reason": None}
    monkeypatch.setattr(fb, "spearman", scripted)
    result = fb._permutation_check(problem, config, 10, 0)
    summary = result["permuted_prediction_spearman"]
    assert summary["defined_permutations"] == 5 and summary["undefined_permutations"] == 5
    assert summary["undefined_reasons"] == {REASON: 5}
    assert summary["conditional_mean"] == pytest.approx(0.2), "an undefined one is not scored 0"
    assert result["permuted_prediction_spearman_mean"] == pytest.approx(0.2)
    assert summary["mean_unavailable_reason"] is None
    assert result["real_prediction_spearman"] == pytest.approx(0.9)
    monkeypatch.setattr(fb, "spearman",
                        lambda scores, outcomes: {"value": None, "n": 16, "reason": REASON})
    none = fb._permutation_check(problem, config, 6, 0)
    assert none["real_prediction_spearman"] is None
    assert none["real_prediction_spearman_reason"] == REASON
    summary = none["permuted_prediction_spearman"]
    assert summary["defined_permutations"] == 0 and summary["undefined_permutations"] == 6
    assert summary["conditional_mean"] is None and summary["mean_unavailable_reason"]
    assert none["permuted_prediction_spearman_mean"] is None


def test_the_planted_toy_is_labelled_a_fit_and_intervention_qualification(planted):
    scope = planted["design"]["scope"]
    assert scope["full_factorial"] is True and scope["unique_feature_contexts"] == 8
    assert scope["training_rows"] == 24 and scope["evaluation_rows"] == 16
    assert scope["supplied_interaction_coordinate"] == "ab"
    assert scope["evaluation_cells_are_training_cells"] is True
    assert scope["evaluation_contexts_unseen_in_training"] == 0
    assert "NOT unseen-context generalisation" in scope["label"]
    assert "24 rows" in planted["design"]["training"]
    reading = planted["interventions"]["donor_patch_sufficiency"]["reading"]
    assert "REMOVES the cross-context reversal" in reading


# ---------------------------------------------------------------------------
# bounded correctness repair: CSV and JSON loaders
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("literal", ["-1e999", "-1E400", "-9e9999"])
def test_an_overflowing_negative_literal_is_refused_not_turned_into_a_structural_zero(
        tmp_path, literal):
    path = write_csv(tmp_path / "c.csv", CANDIDATE_HEADER, GOOD_ROWS + [f"c5,DA,g9,train,{literal}"])
    with pytest.raises(ValueError, match="overflows to negative infinity"):
        fb.read_candidates(path, alphabet="ACD", length=2)
    outcomes = write_csv(tmp_path / "o.csv", "candidate_id,utility", [f"a,{literal}"])
    with pytest.raises(ValueError, match="overflows"):
        fb.read_outcomes(outcomes, label="outcomes")


@pytest.mark.parametrize("spelling", ["-inf", "-INF", "-Infinity"])
def test_only_an_explicitly_spelled_negative_infinity_declares_a_structural_zero(tmp_path,
                                                                                  spelling):
    rows = GOOD_ROWS[:2] + [f"c3,CA,g2,eval,{spelling}", "c4,CD,g2,eval,-0.5"]
    table = fb.read_candidates(write_csv(tmp_path / "c.csv", CANDIDATE_HEADER, rows),
                               alphabet="ACD", length=2)
    assert table.prior_log_weight.tolist() == [0.0, -1.0, -math.inf, -0.5]


def test_the_json_loader_rejects_duplicate_keys_at_every_nesting_level(tmp_path):
    cases = {"top": '{"status": "label_free", "status": "unknown"}',
             "nested": '{"a": {"b": {"c": 1, "c": 2}}}',
             "in_a_list": '{"a": [{"k": 1, "k": 2}]}'}
    for name, text in cases.items():
        path = tmp_path / f"{name}.json"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match="appears twice"):
            fb.read_json(path)
        with pytest.raises(ValueError, match="appears twice"):
            fb.load_manifest(path)
    good = tmp_path / "good.json"
    good.write_text('{"a": {"b": 1}, "c": [1, 2.5], "d": {"b": 2}}', encoding="utf-8")
    assert fb.read_json(good) == {"a": {"b": 1}, "c": [1, 2.5], "d": {"b": 2}}
    huge = tmp_path / "huge.json"
    huge.write_text('{"x": 1e999}', encoding="utf-8")
    with pytest.raises(ValueError, match="overflows to infinity"):
        fb.read_json(huge)
    nan = tmp_path / "nan.json"
    nan.write_text('{"x": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="strict JSON"):
        fb.read_json(nan)


# ---------------------------------------------------------------------------
# bounded correctness repair: no input may be destroyed by an output
# ---------------------------------------------------------------------------

def run_aliased(paths, out, overwrite, **replace):
    inputs = {**paths, **replace}
    return fb.run_finite_support(
        candidates=inputs["candidates"], train_outcomes=inputs["train_outcomes"],
        eval_outcomes=inputs["eval_outcomes"], manifest=inputs["manifest"], output_dir=out,
        config=fb.BaselineConfig(tau=1.0), overwrite=overwrite)


@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("alias_name", [fb.FROZEN_NAME, fb.REPORT_NAME, fb.FROZEN_NAME + ".tmp",
                                        fb.REPORT_NAME + ".tmp"])
@pytest.mark.parametrize("which", ["candidates", "train_outcomes", "eval_outcomes", "manifest"])
def test_an_input_aliasing_an_output_or_its_temp_file_is_refused_before_any_write(
        tmp_path, which, alias_name, overwrite):
    paths = fb.write_synthetic_inputs(tmp_path / "in")
    out = tmp_path / "out"
    out.mkdir()
    victim = out / alias_name
    victim.write_bytes(paths[which].read_bytes())
    original = victim.read_bytes()
    before = sorted(item.name for item in out.iterdir())
    with pytest.raises(ValueError, match="resolves to the same file"):
        run_aliased(paths, out, overwrite, **{which: victim})
    assert victim.read_bytes() == original, "an input was destroyed"
    assert sorted(item.name for item in out.iterdir()) == before, "a refusal must be write-free"


def test_a_dotdot_spelling_and_a_hard_link_of_an_output_are_still_aliases(tmp_path):
    paths = fb.write_synthetic_inputs(tmp_path / "in")
    out = tmp_path / "out"
    out.mkdir()
    victim = out / fb.FROZEN_NAME
    victim.write_bytes(paths["eval_outcomes"].read_bytes())
    original = victim.read_bytes()
    spelled = tmp_path / "out" / ".." / "out" / fb.FROZEN_NAME
    with pytest.raises(ValueError, match="resolves to the same file"):
        run_aliased(paths, out, True, eval_outcomes=spelled)
    link = tmp_path / "in" / "linked_eval.csv"
    try:
        os.link(victim, link)
    except (OSError, NotImplementedError, AttributeError):
        pytest.skip("hard links are unavailable on this filesystem")
    with pytest.raises(ValueError, match="resolves to the same file"):
        run_aliased(paths, out, True, eval_outcomes=link)
    assert victim.read_bytes() == original
    with pytest.raises(ValueError, match="not a directory"):
        run_aliased(paths, paths["candidates"], True)


def test_the_cli_refuses_an_aliasing_output_with_exit_2_even_with_overwrite(tmp_path, capsys):
    cli = load_cli()
    paths = fb.write_synthetic_inputs(tmp_path / "in")
    out = tmp_path / "out"
    out.mkdir()
    victim = out / fb.FROZEN_NAME
    victim.write_bytes(paths["eval_outcomes"].read_bytes())
    original = victim.read_bytes()
    argv = ["finite-support", "--candidates", str(paths["candidates"]),
            "--train-outcomes", str(paths["train_outcomes"]), "--eval-outcomes", str(victim),
            "--manifest", str(paths["manifest"]), "--output-dir", str(out), "--tau", "1.0",
            "--overwrite"]
    assert cli.main(argv) == 2
    assert "resolves to the same file" in capsys.readouterr().err
    assert victim.read_bytes() == original
    assert sorted(item.name for item in out.iterdir()) == [fb.FROZEN_NAME]


# ---------------------------------------------------------------------------
# bounded correctness repair: nested label budgets
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("offered,seed", [(12, 0), (12, 7), (40, 3), (6, 11)])
def test_budget_subsets_are_nested_for_a_fixed_roster_and_seed(offered, seed):
    chosen = {budget: set(fb.acquire_labels(offered, budget, seed).tolist())
              for budget in range(1, offered + 1)}
    for small in range(1, offered):
        assert chosen[small] <= chosen[small + 1], (small, seed)
    assert len(chosen[1]) == 1 and chosen[offered] == set(range(offered))
    expected = np.sort(np.random.default_rng([seed, fb.STREAM_BUDGET]).permutation(offered)[:5])
    assert fb.acquire_labels(offered, 5, seed).tolist() == expected.tolist()


def test_nested_budgets_hold_through_the_pipeline_and_the_seed_still_matters():
    ids = {budget: set(fb.fit_and_allocate(tiny_problem(),
                                           tiny_config(label_budget=budget, seed=7))["training"]["ids"])
           for budget in (2, 4, 6)}
    assert ids[2] <= ids[4] <= ids[6]
    assert set(fb.acquire_labels(8, 4, 7).tolist()) <= set(fb.acquire_labels(8, 8, 7).tolist())
    reference = set(fb.acquire_labels(30, 5, 0).tolist())
    assert any(set(fb.acquire_labels(30, 5, seed).tolist()) != reference for seed in range(1, 6))
    recorded = fb.fit_and_allocate(tiny_problem(), tiny_config(label_budget=3))["training"]
    assert "NESTED" in recorded["selection"]["method"]


# ---------------------------------------------------------------------------
# bounded correctness repair: malformed artifacts fail clearly; grouping prose is accurate
# ---------------------------------------------------------------------------

def test_a_rehashed_artifact_missing_a_field_the_verifier_reads_is_a_clear_value_error(frozen):
    forged = copy.deepcopy(frozen)
    del forged["support"]["identity_sha256"]
    forged["hashes"] = fb._section_hashes(forged)
    with pytest.raises(ValueError, match="malformed"):
        fb.verify_artifact(forged)
    with pytest.raises(ValueError, match="malformed"):
        fb.evaluate_frozen(forged, TRUTH)
    mistyped = copy.deepcopy(frozen)
    mistyped["hashes"] = "not a mapping"
    with pytest.raises(ValueError, match="malformed"):
        fb.verify_artifact(mistyped)
    assert fb.verify_artifact(frozen) is True


def test_a_rehashed_artifact_missing_a_field_only_the_evaluator_reads_is_a_clear_value_error(
        frozen):
    forged = copy.deepcopy(frozen)
    del forged["policies"]["sir"]["distinct_proposed"]
    forged["hashes"] = fb._section_hashes(forged)
    assert fb.verify_artifact(forged) is True      # verification does not read that field
    with pytest.raises(ValueError, match="malformed"):
        fb.evaluate_frozen(forged, TRUTH)
    # an outcome problem is still reported as an outcome problem, never blamed on the artifact
    with pytest.raises(ValueError, match="missing") as caught:
        fb.evaluate_frozen(frozen, {key: v for key, v in TRUTH.items() if key != "s1"})
    assert "malformed" not in str(caught.value)


def test_the_synthetic_grouping_prose_claims_no_sequence_distance_the_grouping_does_not_enforce(
        tmp_path):
    paths = fb.write_synthetic_inputs(tmp_path)
    rationale = fb.read_json(paths["manifest"])["grouping_rationale"].lower()
    assert "near-identical" not in rationale
    assert "does not enforce a minimum sequence distance" in rationale
    table = fb.read_candidates(paths["candidates"], alphabet=fb.SELFTEST_ALPHABET,
                               length=fb.SELFTEST_LENGTH)
    train = [s for s, split in zip(table.sequences, table.splits) if split == fb.SPLIT_TRAIN]
    held = [s for s, split in zip(table.sequences, table.splits) if split == fb.SPLIT_EVAL]
    closest = min(sum(a != b for a, b in zip(x, y)) for x in train for y in held)
    assert closest == 1, "the prose must not promise a distance: a one-residue neighbour straddles"


# ---------------------------------------------------------------------------
# bounded correctness repair: a large common offset must not erase probabilities
# ---------------------------------------------------------------------------

#: float64 spacing here is 2: ``BIG + 2`` is exact, ``BIG + 0.5`` is not. A per-pool constant of this
#: size (a DPO energy ``theta . phi`` on large features, a trace-normalised predictor) must leave
#: every probability identity intact, because only differences carry information.
BIG = 1e16


def hand_tilt(prior, exponent):
    """``p * exp(exponent)`` renormalised in plain probability space, independent of the module."""
    weights = np.asarray(prior, dtype=float) * np.exp(np.asarray(exponent, dtype=float))
    return weights / weights.sum()


@pytest.mark.parametrize("offset", [BIG, -BIG, 1e300, -1e300])
def test_normalising_equal_weights_at_a_large_common_offset_sums_to_one(offset):
    log_q = fb.normalize_log_weights([offset, offset])
    assert log_q == pytest.approx([-math.log(2.0)] * 2, abs=1e-12)
    assert float(np.exp(log_q).sum()) == pytest.approx(1.0, abs=1e-12)
    walled = fb.normalize_log_weights([offset, -np.inf, offset])
    assert np.exp(walled) == pytest.approx([0.5, 0.0, 0.5], abs=1e-12) and walled[1] == -np.inf


@pytest.mark.parametrize("offset", [BIG, -BIG])
def test_a_representable_gap_at_a_large_offset_gives_the_exact_odds(offset):
    log_q = fb.normalize_log_weights([offset, offset + 2.0])
    assert log_q[1] - log_q[0] == pytest.approx(2.0, abs=1e-12)
    assert np.exp(log_q) == pytest.approx(hand_tilt([1.0, 1.0], [0.0, 2.0]), abs=1e-12)


@pytest.mark.parametrize("offset", [0.0, BIG, -BIG, 1e300, -1e300])
@pytest.mark.parametrize("tau", [0.5, 1.0, 7.0])
def test_a_constant_utility_leaves_a_nonuniform_prior_untouched(offset, tau):
    prior = np.array([0.25, 0.75])
    q = np.exp(fb.log_tilt(np.log(prior), [offset, offset], tau))
    assert q == pytest.approx(prior, abs=1e-12)


def test_a_constant_utility_at_a_large_offset_keeps_structural_zeros_and_finite_low_weights():
    log_prior = np.array([math.log(0.25), -np.inf, math.log(0.75), -10000.0])
    reference = fb.normalize_log_weights(log_prior)
    for offset in (BIG, -BIG, 1e300):
        log_q = fb.log_tilt(log_prior, np.full(4, offset), 1.0)
        # rtol because float64 spacing at -10000 is ~2e-12: one rounding step there is not an error
        np.testing.assert_allclose(log_q, reference, atol=1e-12, rtol=1e-12)
        assert log_q[1] == -np.inf and np.isfinite(log_q[3]), offset


def test_a_finite_minus_10000_prior_is_revived_by_a_tilt_at_a_large_offset():
    brought_back = np.exp(fb.log_tilt([0.0, -10000.0], [BIG, BIG + 10000.0], 1.0))
    assert brought_back == pytest.approx([0.5, 0.5], abs=1e-12)
    # a masked candidate's own (huge) utility must neither revive it nor drag the others off scale
    walled = fb.log_tilt([math.log(0.5), -np.inf, math.log(0.5)], [BIG, 1e300, BIG], 1.0)
    assert np.exp(walled) == pytest.approx([0.5, 0.0, 0.5], abs=1e-12) and walled[1] == -np.inf


@pytest.mark.parametrize("offset", [BIG, -BIG])
def test_a_representable_utility_gap_at_a_large_offset_gives_the_prior_weighted_odds(offset):
    tau, prior = 2.0, np.array([0.2, 0.8])
    q = np.exp(fb.log_tilt(np.log(prior), [offset, offset + 2.0], tau))
    assert q[1] / q[0] == pytest.approx(4.0 * math.exp(1.0), rel=1e-12)
    assert q == pytest.approx(hand_tilt(prior, [0.0, 2.0 / tau]), abs=1e-12)


def test_the_same_dpo_energy_at_a_large_offset_leaves_the_prior_at_every_pressure():
    log_p = np.log([0.25, 0.75])
    for offset in (BIG, -BIG, 1e300):
        for pressure in (0.25, 1.0, 4.0):
            q = np.exp(fb.dpo_log_policy(log_p, [offset, offset], pressure, name="p"))
            assert q == pytest.approx([0.25, 0.75], abs=1e-12), (offset, pressure)
    allocation = fb.allocate(("a", "b"), log_p, [0.0, 0.0], tiny_config(), dpo_energy=[BIG, BIG])
    dpo = {name: log_q for name, log_q in allocation.policies.items()
           if name.startswith("dpo_pressure_")}
    assert len(dpo) == 5
    for name, log_q in dpo.items():
        assert np.exp(log_q) == pytest.approx([0.25, 0.75], abs=1e-12), name


def test_allocation_matches_the_hand_tilts_for_utilities_and_energies_at_a_large_offset():
    prior = np.array([0.2, 0.3, 0.5])
    gaps = np.array([0.0, 2.0, 0.0])
    config = tiny_config(tau=2.0)
    allocation = fb.allocate(("a", "b", "c"), np.log(prior), BIG + gaps, config,
                             dpo_energy=-BIG + gaps)
    policies = {name: np.exp(log_q) for name, log_q in allocation.policies.items()}
    assert policies["prior"] == pytest.approx(prior, abs=1e-12)
    assert policies["ridge_tilt"] == pytest.approx(hand_tilt(prior, gaps / 2.0), abs=1e-12)
    for pressure in config.dpo_pressures:
        assert policies[fb.pressure_name(pressure)] == pytest.approx(
            hand_tilt(prior, pressure * gaps), abs=1e-12), pressure
    sir = allocation.sir
    counts, proposal = np.asarray(sir["counts"], dtype=float), np.exp(sir["log_proposal"])
    ratio = (prior / proposal) * np.exp(gaps / 2.0)
    assert policies["sir_weighted_proposal"] == pytest.approx(
        counts * ratio / (counts * ratio).sum(), abs=1e-12)
    assert sir["effective_sample_size"] == pytest.approx(
        (counts * ratio).sum() ** 2 / (counts * ratio ** 2).sum(), rel=1e-9)
    for log_q in allocation.policies.values():
        assert float(np.exp(log_q).sum()) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("offset", [0.0, BIG, -BIG])
def test_sir_with_equal_utilities_returns_the_count_weighted_prior_ratio_and_the_hand_ess(offset):
    prior, proposal = np.log([0.25, 0.75]), np.log([0.5, 0.5])
    law, info = fb.sir_weighted_distribution(prior, proposal, [offset, offset], 1.0,
                                             np.array([1, 1]))
    assert np.exp(law) == pytest.approx([0.25, 0.75], abs=1e-12)
    assert info["effective_sample_size"] == pytest.approx(1.6, rel=1e-12)    # 2^2 / (0.25 + 2.25)
    # multiplicity: n = (2, 1) gives n*w = (1.0, 1.5), law (0.4, 0.6), ESS 2.5^2 / (2*0.25 + 2.25)
    law, info = fb.sir_weighted_distribution(prior, proposal, [offset, offset], 1.0,
                                             np.array([2, 1]))
    assert np.exp(law) == pytest.approx([0.4, 0.6], abs=1e-12)
    assert info["effective_sample_size"] == pytest.approx(25.0 / 11.0, rel=1e-12)


def test_sir_rebases_over_sampled_reachable_candidates_and_keeps_a_representable_gap():
    prior, proposal = np.array([0.2, 0.3, 0.5]), np.array([0.5, 0.25, 0.25])
    counts = np.array([3, 0, 2])
    # candidate 1 is never sampled, so its absurd utility must not set the reference
    law, info = fb.sir_weighted_distribution(np.log(prior), np.log(proposal),
                                             [BIG, 1e300, BIG + 2.0], 1.0, counts)
    per_draw = np.array([0.2 / 0.5 * math.exp(-2.0), 0.0, 0.5 / 0.25])     # p/r * exp(u'/tau)
    mass = counts * per_draw
    assert np.exp(law) == pytest.approx(mass / mass.sum(), abs=1e-12)
    assert info["effective_sample_size"] == pytest.approx(
        mass.sum() ** 2 / (counts * per_draw ** 2).sum(), rel=1e-12)
    assert info["distinct_candidates"] == 2 and info["positive_weight_distinct_candidates"] == 2


def test_sir_gives_a_sampled_structural_zero_no_weight_at_a_large_offset():
    law, info = fb.sir_weighted_distribution([math.log(0.5), math.log(0.5), -np.inf],
                                             np.log([1 / 3, 1 / 3, 1 / 3]), [BIG] * 3, 1.0,
                                             np.array([1, 1, 2]))
    assert np.exp(law) == pytest.approx([0.5, 0.5, 0.0], abs=1e-12) and law[2] == -np.inf
    assert info["effective_sample_size"] == pytest.approx(2.0, rel=1e-12)   # 3^2 / (2.25 + 2.25)
    assert info["distinct_candidates"] == 3 and info["positive_weight_distinct_candidates"] == 2


#: The proposal puts essentially all its mass on candidate 2, which the PRIOR gives none. The two
#: live candidates then carry the SAME tiny proposal log weight, so their importance ratio is the
#: prior ratio -- if nothing rounds ``log p`` away against the 1e16 on its way to the difference.
RARE_PROPOSALS = {"rare_live": [-1e16, -1e16, 0.0], "dominant_unsampled": [0.0, 0.0, 1e16]}


@pytest.mark.parametrize("prior, counts, utilities, expected_law, expected_ess", [
    # prior ratio alone: w ∝ p = (.25, .75); ESS = 1^2 / (.0625 + .5625)
    ([0.25, 0.75, 0.0], [1, 1, 0], [0.0, 0.0, 0.0], [0.25, 0.75, 0.0], 1.6),
    # prior ratio and multiplicity: n·w ∝ (2/3, 1) -> (.4, .6); ESS = (5/3)^2 / (2/9 + 1)
    ([0.25, 0.75, 0.0], [2, 1, 0], [0.0, 0.0, 0.0], [0.4, 0.6, 0.0], 25.0 / 11.0),
    # multiplicity alone: equal prior, n = (2, 1) -> (2/3, 1/3); ESS = 3^2 / (2 + 1)
    ([0.5, 0.5, 0.0], [2, 1, 0], [0.0, 0.0, 0.0], [2 / 3, 1 / 3, 0.0], 3.0),
    # a utility that exactly cancels the prior ratio (tau = 1, u0 = log 3): w ∝ (.75, .75)
    ([0.25, 0.75, 0.0], [1, 1, 0], [math.log(3.0), 0.0, 0.0], [0.5, 0.5, 0.0], 2.0),
    ([0.25, 0.75, 0.0], [2, 1, 0], [math.log(3.0), 0.0, 0.0], [2 / 3, 1 / 3, 0.0], 3.0),
])
@pytest.mark.parametrize("proposal", sorted(RARE_PROPOSALS))
def test_sir_keeps_prior_ratios_and_multiplicity_under_a_huge_common_proposal_offset(
        prior, counts, utilities, expected_law, expected_ess, proposal):
    with np.errstate(divide="ignore"):
        log_prior = np.log(np.asarray(prior))
    law, info = fb.sir_weighted_distribution(log_prior, RARE_PROPOSALS[proposal], utilities, 1.0,
                                             np.array(counts))
    assert np.exp(law) == pytest.approx(expected_law, abs=1e-12)
    assert law[2] == -np.inf, "the prior's structural zero must stay a structural zero"
    assert float(np.exp(law).sum()) == pytest.approx(1.0, abs=1e-12)
    assert info["effective_sample_size"] == pytest.approx(expected_ess, rel=1e-12)
    assert info["proposal_draws"] == sum(counts)
    assert info["distinct_candidates"] == 2 and info["positive_weight_distinct_candidates"] == 2


@pytest.mark.parametrize("unsampled_mass", [0.0, 5.0, 1e3, 1e8, 1e16, 1e100])
def test_sir_law_does_not_depend_on_how_much_proposal_mass_a_zero_prior_candidate_holds(
        unsampled_mass):
    """Candidate 3 is a structural zero of the prior, so it is never live however often it is
    drawn. Its proposal weight is a gauge on the live candidates: the law and the ESS must not
    move as it grows from comparable to 1e100. Dyadic log weights keep every input exact, so any
    drift is the function's, not the test's."""
    log_prior = np.array([0.0, 1.0, -0.5, -np.inf])
    live_proposal = np.array([0.5, -0.5, 1.0])
    utilities = np.array([0.5, 0.0, -1.0, 7.0])
    counts = np.array([3, 1, 2, 4])           # the zero-prior candidate is drawn four times
    tau = 0.5
    law, info = fb.sir_weighted_distribution(
        log_prior, np.append(live_proposal, unsampled_mass), utilities, tau, counts)
    # hand value on the live candidates only: n * w with w = (p / r) * exp(u / tau)
    w = np.exp(log_prior[:3]) / np.exp(live_proposal) * np.exp(utilities[:3] / tau)
    mass = counts[:3] * w
    assert np.exp(law) == pytest.approx(np.append(mass / mass.sum(), 0.0), abs=1e-12)
    assert law[3] == -np.inf
    assert info["effective_sample_size"] == pytest.approx(
        mass.sum() ** 2 / (counts[:3] * w ** 2).sum(), rel=1e-12)
    assert info["proposal_draws"] == 10 and info["distinct_candidates"] == 4
    assert info["positive_weight_distinct_candidates"] == 3


def test_sir_keeps_the_live_ratio_when_an_unsampled_prior_candidate_carries_a_huge_log_weight():
    """The prior is only declared up to a constant too: a finite, UNSAMPLED candidate with log
    weight 2^50 must not push the live candidates' prior ratio below the float spacing."""
    big = 2.0 ** 50
    law, info = fb.sir_weighted_distribution(
        [0.0, 1.0, big], [0.0, 0.5, 0.0], [0.0, 0.0, 0.0], 1.0, np.array([1, 2, 0]))
    # w ∝ p / r = (1 / 1, e^1 / e^0.5): n·w = (1, 2 e^0.5)
    weights = np.array([1.0, 2.0 * math.exp(0.5), 0.0])
    assert np.exp(law) == pytest.approx(weights / weights.sum(), abs=1e-12)
    assert law[2] == -np.inf
    # the finite-support guards survive the new centring: the proposal must cover the reachable
    # support, may not be asked for a draw it cannot make, and the sampled weight must be positive
    with pytest.raises(ValueError, match="zero mass to 1 candidate"):
        fb.sir_weighted_distribution(np.log([0.5, 0.5]), np.array([0.0, -np.inf]), [0.0, 0.0], 1.0,
                                     np.array([1, 1]))
    with pytest.raises(ValueError, match="cannot draw"):
        fb.sir_weighted_distribution(np.array([0.0, 0.0, -np.inf]), np.array([0.0, 0.0, -np.inf]),
                                     [0.0, 0.0, 0.0], 1.0, np.array([1, 1, 1]))
    with pytest.raises(ValueError, match="zero total sampled weight"):
        fb.sir_weighted_distribution(np.array([0.0, -np.inf]), np.array([0.0, 0.0]), [0.0, 0.0],
                                     1.0, np.array([0, 3]))


def test_verification_recomputes_offset_policies_with_the_helper_allocation_uses(frozen,
                                                                                  monkeypatch):
    """The stored tilts are the HAND values; verification must reach them through the same centring
    that allocation used, or an honest offset artifact would be refused (or a forged one pass)."""
    forged = copy.deepcopy(frozen)
    gaps = np.array([0.0, 2.0, 0.0, 2.0])
    tau = forged["config"]["tau"]
    prior = np.array([0.0 if item is None else math.exp(item)
                      for item in forged["prior"]["log_probability"]])

    def stored(weights):
        weights = np.asarray(weights, dtype=float)
        with np.errstate(divide="ignore"):
            return fb.encode_log(np.log(weights / weights.sum()))

    forged["predictions"]["utility"] = (BIG + gaps).tolist()
    forged["predictions"]["dpo_energy"] = (-BIG + gaps).tolist()
    log_policies = forged["policies"]["log_probability"]
    log_policies["ridge_tilt"] = stored(hand_tilt(prior, gaps / tau))
    for pressure in forged["config"]["dpo"]["pressures"]:
        log_policies[fb.pressure_name(pressure)] = stored(hand_tilt(prior, pressure * gaps))
    sir = forged["policies"]["sir"]
    counts = np.array(sir["counts"], dtype=float)
    proposal = np.array([0.0 if item is None else math.exp(item) for item in sir["log_proposal"]])
    ratio = np.divide(prior, proposal, out=np.zeros(4), where=proposal > 0)
    log_policies["sir_weighted_proposal"] = stored(counts * ratio * np.exp(gaps / tau))
    forged["hashes"] = fb._section_hashes(forged)
    calls = []
    real = fb.dpo_log_policy
    monkeypatch.setattr(fb, "dpo_log_policy", lambda *a, **k: calls.append(1) or real(*a, **k))
    assert fb.verify_artifact(forged) is True
    assert len(calls) == len(forged["config"]["dpo"]["pressures"]), (
        "verification must recompute each DPO policy through the shared helper")
    doctored = copy.deepcopy(forged)
    doctored["policies"]["log_probability"]["ridge_tilt"] = stored(prior)   # the prior, not the tilt
    doctored["hashes"] = fb._section_hashes(doctored)
    with pytest.raises(ValueError, match="stored exact tilt"):
        fb.verify_artifact(doctored)


def test_unrepresentable_differences_fail_explicitly_and_never_become_zero_mass():
    huge = 1.7e308
    with pytest.raises(ValueError, match="span more than a float64"):
        fb.normalize_log_weights([huge, -huge])
    assert np.isfinite(fb.normalize_log_weights([0.0, -huge])).all(), (
        "a finite weight that IS representable relative to the maximum stays finite")
    with pytest.raises(ValueError, match="rebasing them would overflow"):
        fb.log_tilt([0.0, 0.0], [huge, -huge], 1.0)
    with pytest.raises(ValueError, match="rebasing them would overflow"):
        fb.dpo_log_policy(np.log([0.5, 0.5]), [huge, -huge], 1.0, name="p")
    with pytest.raises(ValueError, match="overflowed"):
        fb.log_tilt([0.0, 0.0], [0.0, 1e300], 1e-10)
    with pytest.raises(ValueError, match="overflowed"):
        fb.dpo_log_policy(np.log([0.5, 0.5]), [0.0, 1e300], 1e10, name="p")
    with pytest.raises(ValueError, match="differ by more than a float64"):
        fb.sir_weighted_distribution(np.log([0.5, 0.5]), np.log([0.5, 0.5]), [huge, -huge], 1.0,
                                     np.array([1, 1]))
    with pytest.raises(ValueError, match="not representable"):
        fb.sir_weighted_distribution(np.log([0.5, 0.5]), np.log([0.5, 0.5]), [0.0, -1e300], 1e-10,
                                     np.array([1, 1]))


# ---------------------------------------------------------------------------
# Part 2: the DPLM hook binder, on a tiny random CPU model
# ---------------------------------------------------------------------------

def hook_model(layers=2, seed=1):
    return dplm_fixtures.make_model(seed=seed,
                                    config=dplm_fixtures.tiny_config(num_hidden_layers=layers))


def two_constructs():
    first = dplm_fixtures.construct(record="a")
    second = dplm_fixtures.construct(heavy=dplm_fixtures.LIGHT, light=dplm_fixtures.HEAVY,
                                     record="b")
    return first, second


def batch_for(*items, condition=None):
    tokens, slots = dplm_fixtures.public_batch(list(items))
    batch = {"tokens": tokens, "slots": slots}
    if condition is not None:
        batch["condition"] = condition
    return batch


def log_probs(model, batch):
    return model.canonical_log_probs(batch["tokens"], batch["slots"],
                                     condition_indices=batch.get("condition"))


def tap(index, site):
    return hooks_lib.activation_name("dplm", index, site)


def test_the_binder_registers_four_tokens_first_taps_on_the_real_modules():
    model = hook_model(layers=3)
    registry = hooks_lib.bind_conditioned_dplm(model)
    expected = sorted(tap(i, site) for i in range(3) for site in hooks_lib.SITES)
    assert registry.names() == expected and len(expected) == 12
    assert all(registry.layout(name) == hooks_lib.LAYOUT_TOKENS_FIRST for name in expected)
    assert all(registry.position_axis(name) == 1 for name in expected)
    layer = model.native.esm.encoder.layer[1]
    modules = {site: registry.taps[tap(1, site)]["module"] for site in hooks_lib.SITES}
    assert modules["resid_pre"] is layer and modules["resid_post"] is layer
    assert modules["attn_out"] is layer.attention.output.dropout
    assert modules["mlp_out"] is layer.output.dropout
    custom = hooks_lib.bind_conditioned_dplm(model, route="other")
    assert custom.route == "other" and custom.names()[0].startswith("route/other/")


def test_a_real_forward_fires_every_site_with_float32_tokens_first_activations():
    model = hook_model()
    registry = hooks_lib.bind_conditioned_dplm(model)
    batch = batch_for(*two_constructs())
    grabbed = interp.capture(model, registry, batch=batch, forward=log_probs)
    length, hidden = int(batch["tokens"].shape[1]), model.hidden_size
    assert grabbed["names"] == registry.names() and len(grabbed["exports"]) == 8
    for name, export in grabbed["exports"].items():
        assert export["layout"] == hooks_lib.LAYOUT_TOKENS_FIRST and export["position_axis"] == 1
        assert tuple(export["tensor"].shape) == (2, length, hidden), name
        assert export["tensor"].dtype == torch.float32, name
    assert registry.handles == [] and registry.interventions == {}


def test_enabling_the_taps_without_a_plan_leaves_the_forward_bit_identical():
    model = hook_model()
    registry = hooks_lib.bind_conditioned_dplm(model)
    batch = batch_for(*two_constructs())
    flag = model.training
    with interp.controlled_evaluation(model):
        plain = log_probs(model, batch)
    grabbed = interp.capture(model, registry, batch=batch, forward=log_probs)
    assert torch.equal(grabbed["output"], plain)
    assert model.training is flag, "capture must restore the caller's train/eval flag"
    again = interp.capture(model, registry, batch=batch, forward=log_probs)
    assert torch.equal(again["output"], plain)


def test_each_layer_adds_its_two_branches_to_the_residual_and_layers_chain_exactly():
    model = hook_model(layers=3)
    registry = hooks_lib.bind_conditioned_dplm(model)
    batch = batch_for(*two_constructs())
    grabbed = interp.capture(model, registry, batch=batch, forward=log_probs)
    acts = {name: export["tensor"] for name, export in grabbed["exports"].items()}
    for index in range(3):
        pre, attention, mlp, post = (acts[tap(index, site)] for site in hooks_lib.SITES)
        assert torch.allclose(post, pre + attention + mlp, atol=1e-5), index
        assert not torch.allclose(post, pre, atol=1e-4), "the layer changed nothing"
        assert float(attention.abs().max()) > 0 and float(mlp.abs().max()) > 0
        if index + 1 < 3:
            assert torch.equal(post, acts[tap(index + 1, "resid_pre")]), index
    native = dplm.ablang_to_dplm(batch["tokens"])
    last = acts[tap(2, "resid_post")]
    with interp.controlled_evaluation(model):
        final = model.native(native)["last_hidden_state"]
        normalised = model.native.esm.encoder.emb_layer_norm_after(last)
    assert torch.allclose(normalised, final, atol=1e-6), "last_hidden_state is the LayerNormed output"
    assert not torch.allclose(final, last, atol=1e-3), "it is NOT the final layer's resid_post"


def test_a_full_final_layer_patch_reproduces_the_donor_outputs_and_not_the_recipients():
    model = hook_model()
    registry = hooks_lib.bind_conditioned_dplm(model)
    first, second = two_constructs()
    recipient, donor = batch_for(first, second), batch_for(second, first)
    last = tap(1, "resid_post")
    donor_run = interp.capture(model, registry, batch=donor, names=[last], forward=log_probs)
    baseline = interp.capture(model, registry, batch=recipient, names=[last],
                              forward=log_probs)["output"]
    patched = interp.intervene(model, registry, batch=recipient, name=last, kind="patch",
                               forward=log_probs, source=donor_run["exports"][last])["output"]
    assert torch.allclose(patched, donor_run["output"], atol=1e-6)
    assert not torch.allclose(baseline, donor_run["output"], atol=1e-4)
    assert not torch.allclose(patched, baseline, atol=1e-4)


def test_a_resid_pre_patch_keeps_the_rotary_and_mask_arguments_alive():
    model = hook_model()
    registry = hooks_lib.bind_conditioned_dplm(model)
    first, second = two_constructs()
    recipient, donor = batch_for(first, second), batch_for(second, first)
    name = tap(0, "resid_pre")
    donor_run = interp.capture(model, registry, batch=donor, names=[name], forward=log_probs)
    patched = interp.intervene(model, registry, batch=recipient, name=name, kind="patch",
                               forward=log_probs, source=donor_run["exports"][name])["output"]
    assert torch.allclose(patched, donor_run["output"], atol=1e-6), (
        "the pre-hook must pass cos, sin and the key mask through untouched")


def test_ablations_and_positional_patches_change_the_real_forward_but_a_capture_plan_does_not():
    model = hook_model()
    registry = hooks_lib.bind_conditioned_dplm(model)
    first, second = two_constructs()
    batch = batch_for(first, second)
    baseline = interp.capture(model, registry, batch=batch, forward=log_probs)["output"]
    for site in ("attn_out", "mlp_out"):
        ablated = interp.intervene(model, registry, batch=batch, name=tap(0, site),
                                   kind="zero_ablate", forward=log_probs)["output"]
        assert not torch.allclose(ablated, baseline, atol=1e-6), site
    unchanged = interp.intervene(model, registry, batch=batch, name=tap(0, "attn_out"),
                                 kind="capture", forward=log_probs)["output"]
    assert torch.equal(unchanged, baseline)
    donor = interp.capture(model, registry, batch=batch_for(second, first),
                           names=[tap(1, "resid_pre")], forward=log_probs)["exports"][tap(1, "resid_pre")]
    slots = [int(s) for s in batch["slots"][0]]
    local = interp.intervene(model, registry, batch=batch, name=tap(1, "resid_pre"), kind="patch",
                             forward=log_probs, source=donor, positions=slots)["output"]
    assert not torch.allclose(local, baseline, atol=1e-6)
    registry.plan(tap(1, "resid_post"), kind="capture")
    registry.plan(tap(1, "resid_pre"), kind="patch", source=donor, positions=slots)
    registry.enable([tap(1, "resid_pre"), tap(1, "resid_post")])
    with interp.controlled_evaluation(model):
        log_probs(model, batch)
    registry.disable()
    pre = registry.captured[tap(1, "resid_pre")]
    assert torch.equal(pre, interp.capture(
        model, registry, batch=batch, names=[tap(1, "resid_pre")],
        forward=log_probs)["exports"][tap(1, "resid_pre")]["tensor"]), (
        "a capture under an intervention is the pre-intervention activation")


def test_mean_ablation_uses_the_frozen_donor_mean_and_ignores_the_batch():
    model = hook_model()
    registry = hooks_lib.bind_conditioned_dplm(model)
    first, second = two_constructs()
    name = tap(1, "resid_pre")
    reference = interp.build_reference_mean(
        model, registry, donor_batches=[batch_for(second, first)], name=name, forward=log_probs,
        ids_of=lambda batch: ["donor-0", "donor-1"], provenance={"donor_set": "synthetic"})
    assert reference.donor_rows == 2 and reference.document()["donor_set"] == "synthetic"
    together = interp.intervene(model, registry, batch=batch_for(first, second), name=name,
                                kind="mean_ablate", forward=log_probs, reference=reference)["output"]
    alone = interp.intervene(model, registry, batch=batch_for(first), name=name,
                             kind="mean_ablate", forward=log_probs, reference=reference)["output"]
    assert torch.allclose(alone, together[:1], atol=1e-6), "the batch changed the control"
    plain = interp.capture(model, registry, batch=batch_for(first), forward=log_probs)["output"]
    assert not torch.allclose(alone, plain, atol=1e-6)
    with pytest.raises(ValueError, match="explicit frozen donor"):
        registry.plan(name, kind="mean_ablate")


def test_a_capture_holds_only_the_latest_firing_and_exports_are_caller_owned():
    model = hook_model()
    registry = hooks_lib.bind_conditioned_dplm(model)
    first, second = two_constructs()
    name = tap(0, "resid_post")
    registry.enable([name])
    with interp.controlled_evaluation(model):
        log_probs(model, batch_for(first, second))
        earlier = registry.export(name)
        saved = earlier["tensor"].clone()
        log_probs(model, batch_for(second, first))
    registry.disable()
    later = registry.export(name)
    assert not torch.equal(later["tensor"], saved), "the registry kept the first firing"
    assert torch.equal(earlier["tensor"], saved), "a later forward rewrote a caller's export"
    earlier["tensor"].add_(1.0)
    assert torch.equal(registry.export(name)["tensor"], later["tensor"])
    later["tensor"].zero_()
    assert float(registry.export(name)["tensor"].abs().sum()) > 0


def test_padding_is_excluded_from_a_dplm_reference_mean():
    model = hook_model()
    registry = hooks_lib.bind_conditioned_dplm(model)
    long = dplm_fixtures.construct(record="long")
    short = dplm_fixtures.construct(heavy=dplm_fixtures.HEAVY[:18], record="short")
    rows, slots = [], []
    for item in (long, short):
        serialization = serial.ablang_serialization(item, masked_sites=item.sites, order=item.sites)
        rows.append(list(serialization.tokens))
        slots.append(list(serialization.query_slots))
    width = max(len(row) for row in rows)
    rows = [row + [serial.ABLANG_PAD_ID] * (width - len(row)) for row in rows]
    batch = {"tokens": torch.tensor(rows), "slots": torch.tensor(slots)}
    real = batch["tokens"] != serial.ABLANG_PAD_ID
    assert real.sum(dim=1).tolist() == [45, 43]
    name = tap(1, "resid_post")
    activation = interp.capture(model, registry, batch=batch, names=[name],
                                forward=log_probs)["exports"][name]["tensor"]
    masked = hooks_lib.reference_mean(activation, name=name, layout=hooks_lib.LAYOUT_TOKENS_FIRST,
                                      mask=real)
    unmasked = hooks_lib.reference_mean(activation, name=name,
                                        layout=hooks_lib.LAYOUT_TOKENS_FIRST)
    assert masked.counts[0].item() == 2 and masked.counts[43].item() == 1
    assert masked.document()["padding_masked"] is True
    assert torch.allclose(masked.mean[0, 43], activation[0, 43])
    assert torch.allclose(unmasked.mean[0, 43], (activation[0, 43] + activation[1, 43]) / 2)
    assert not torch.allclose(unmasked.mean[0, 43], activation[0, 43], atol=1e-6)


def test_absolute_positions_name_different_residues_once_heavy_lengths_differ():
    sites = ((serial.HEAVY, 3), (serial.LIGHT, 0), (serial.LIGHT, 2))
    long = dplm_fixtures.construct(sites=sites, record="long")
    short = dplm_fixtures.construct(heavy=dplm_fixtures.HEAVY[:18], sites=sites, record="short")
    p_long, p_short = dplm.dplm_positions(long), dplm.dplm_positions(short)
    assert p_long[(serial.HEAVY, 3)] == p_short[(serial.HEAVY, 3)] == 4
    assert p_long[(serial.LIGHT, 0)] == 24 and p_short[(serial.LIGHT, 0)] == 22
    assert p_long[(serial.LIGHT, 0)] == p_short[(serial.LIGHT, 2)], (
        "the same encoded position is light residue 0 in one row and light residue 2 in the other")


def test_condition_only_donors_are_identical_until_a_nonuniform_vector_is_planted():
    model = hook_model()
    registry = hooks_lib.bind_conditioned_dplm(model)
    first, second = two_constructs()
    name = tap(1, "resid_pre")
    zeros = torch.zeros(2, 3, dtype=torch.long)
    plain = interp.capture(model, registry, batch=batch_for(first, second, condition=zeros),
                           names=[name], forward=log_probs)
    conditioned = interp.capture(
        model, registry, batch=batch_for(first, second, condition=dplm_fixtures.CONDITIONS),
        names=[name], forward=log_probs)
    assert torch.equal(plain["output"], conditioned["output"])
    assert torch.equal(plain["exports"][name]["tensor"], conditioned["exports"][name]["tensor"])
    dplm_fixtures.randomise_tables(model)
    after = interp.capture(
        model, registry, batch=batch_for(first, second, condition=dplm_fixtures.CONDITIONS),
        names=[name], forward=log_probs)
    before = interp.capture(model, registry, batch=batch_for(first, second, condition=zeros),
                            names=[name], forward=log_probs)
    assert not torch.equal(after["exports"][name]["tensor"], before["exports"][name]["tensor"])
    assert not torch.allclose(after["output"], before["output"], atol=1e-6)
