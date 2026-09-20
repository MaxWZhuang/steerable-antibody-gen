"""CHES: the ten shifted states, the formula, the cache, and the temporal analysis.

The native tests build a two-layer GPTNeoX in memory -- the class the pinned
p-IgGen release uses -- so the state extraction and the output-head reconstruction
run against the real implementation rather than a stand-in. The formula itself is
checked against an independently written triple sum, not against a second copy of
the same factorization.
"""
from __future__ import annotations

import numpy as np
import pytest

from smallAntibodyGen.experiments import her2_ches as ches
from smallAntibodyGen.experiments import her2_data as data
from smallAntibodyGen.experiments import her2_policy as policy_lib

transformers = pytest.importorskip("transformers", reason="CHES tests need transformers")
torch = pytest.importorskip("torch")

VOCAB = {"<PAD>": 0}
VOCAB.update({residue: index + 1 for index, residue in enumerate(data.CANONICAL)})
VOCAB["1"] = len(VOCAB)
VOCAB["2"] = len(VOCAB)
PREFIX = "1" + "ACDEFGHIKLMNPQ"


@pytest.fixture(scope="module")
def policy():
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    torch.manual_seed(3)
    config = GPTNeoXConfig(vocab_size=len(VOCAB), hidden_size=32, num_hidden_layers=2,
                           num_attention_heads=4, intermediate_size=64,
                           max_position_embeddings=64, hidden_dropout=0.0,
                           attention_dropout=0.0, bos_token_id=0, eos_token_id=VOCAB["C"])
    model = GPTNeoXForCausalLM(config)
    model.eval()
    built = policy_lib.CorePolicy.from_prefix(model, PREFIX, VOCAB, device="cpu")
    try:
        built.core_logits(built.token_ids(np.zeros((2, data.CORE_LENGTH), dtype=np.int64)))
    except (AttributeError, TypeError) as error:    # pragma: no cover - old transformers
        pytest.skip(f"this transformers build has no batched prefix cache: {error}")
    return built


def cores(count, seed=0):
    return np.random.default_rng(seed).integers(
        0, 20, size=(count, data.CORE_LENGTH)).astype(np.int64)


# ---------------------------------------------------------------------------
# the formula
# ---------------------------------------------------------------------------

def test_the_vectorized_formula_matches_an_independent_triple_sum():
    generator = np.random.default_rng(1)
    plus = generator.normal(size=(4, 10, 5))
    minus = generator.normal(size=(4, 10, 5))
    assert np.allclose(ches.ches_from_states(plus, minus),
                       ches.ches_reference(plus, minus), atol=1e-9)


def test_identical_responses_give_exactly_zero():
    states = np.random.default_rng(2).normal(size=(3, 10, 6))
    assert np.abs(ches.ches_from_states(states, states)).max() == 0.0
    assert np.abs(ches.ches_reference(states, states)).max() < 1e-9


def test_the_score_is_not_symmetric_in_chosen_and_rejected():
    generator = np.random.default_rng(4)
    plus, minus = generator.normal(size=(2, 10, 3)), generator.normal(size=(2, 10, 3))
    forward = ches.ches_from_states(plus, minus)
    swapped = ches.ches_from_states(minus, plus)
    assert not np.allclose(forward, swapped)


def test_misaligned_state_blocks_are_refused():
    with pytest.raises(ValueError, match="aligned"):
        ches.ches_from_states(np.zeros((2, 10, 3)), np.zeros((2, 9, 3)))


# ---------------------------------------------------------------------------
# the ten states, proved by reconstruction
# ---------------------------------------------------------------------------

def test_exactly_ten_causally_shifted_states_are_extracted(policy):
    index = cores(3)
    states = ches.full_hidden_states(policy, policy.token_ids(index))
    assert states.shape[0] == 3 and states.shape[1] == data.CORE_LENGTH
    assert len(ches.EXTRACTED_POSITIONS) == data.CORE_LENGTH
    assert ches.EXTRACTED_POSITIONS[0].startswith("prefix[")
    assert ches.EXTRACTED_POSITIONS[-1] == f"core_input[{data.CORE_LENGTH - 2}]"


def test_the_extracted_states_reconstruct_the_models_own_logits(policy):
    report = ches.head_reconstruction(policy, policy.token_ids(cores(4)), atol=1e-3, rtol=1e-5)
    assert report["within_tolerance"] and report["max_abs_error"] < 1e-3
    assert report["path"] == "full"


def test_the_cached_and_full_extraction_paths_agree(policy):
    index = cores(4, seed=5)
    full = ches.full_hidden_states(policy, policy.token_ids(index)).detach().numpy()
    cached = ches.cached_hidden_states(policy, policy.token_ids(index)).detach().numpy()
    assert np.allclose(full, cached, atol=1e-4, rtol=1e-4)
    cached_report = ches.head_reconstruction(policy, policy.token_ids(index), atol=1e-3,
                                             rtol=1e-5, cached=True)
    assert cached_report["path"] == "cached" and cached_report["within_tolerance"]


def test_a_wrong_position_shift_breaks_the_reconstruction(policy):
    """The proof has to be able to fail: states rolled by one must not reconstruct."""
    index = cores(3, seed=6)
    core_ids = policy.token_ids(index)
    _, head = ches.backbone_and_head(policy.model)
    states = ches.full_hidden_states(policy, core_ids)
    rolled = torch.roll(states, shifts=1, dims=1)
    reference = policy.full_logits(core_ids)
    shifted_error = (head(rolled)[..., policy.canonical_ids].double()
                     - reference.double()).abs().max()
    assert float(shifted_error) > 1e-2


def test_chosen_and_rejected_never_share_a_mutated_prefix_cache(policy):
    """Scoring the rejected response must not depend on the chosen one preceding it."""
    chosen, rejected = cores(4, seed=7), cores(4, seed=8)
    alone = ches.ches_batch(policy, rejected, rejected, cached=True)
    after = ches.ches_batch(policy, chosen, rejected, cached=True)
    again = ches.ches_batch(policy, rejected, rejected, cached=True)
    assert np.array_equal(alone, again)
    assert np.abs(alone).max() == 0.0 and np.abs(after).max() > 0.0


def test_the_cached_and_full_ches_routes_agree(policy):
    chosen, rejected = cores(6, seed=9), cores(6, seed=10)
    full = ches.ches_batch(policy, chosen, rejected, cached=False)
    cached = ches.ches_batch(policy, chosen, rejected, cached=True)
    relative = np.abs(full - cached) / np.maximum(np.abs(full), 1e-6)
    assert relative.max() < 1e-3


def test_scoring_leaves_the_model_in_evaluation_mode(policy):
    policy.model.eval()
    ches.ches_batch(policy, cores(2), cores(2, seed=1))
    assert policy.model.training is False


def test_ches_scores_returns_aligned_log_probabilities_and_checks_logits(policy):
    chosen, rejected = cores(4, seed=11), cores(4, seed=12)
    block = ches.ches_scores(policy, chosen, rejected, batch_size=2, score_batch_size=4)
    assert block["ches"].shape == (4,)
    assert block["chosen_log_probability"].shape == (4,)
    assert block["logit_checks"]["chosen"]["mode"] == "eval + inference_mode"
    reference = policy.score(chosen, batch_size=4)["sum_log_probability"]
    assert np.allclose(block["chosen_log_probability"], reference, atol=5e-5, rtol=2e-6)


# ---------------------------------------------------------------------------
# populations, identity and consumption
# ---------------------------------------------------------------------------

def test_a_resorted_pair_population_is_a_different_identity():
    chosen, rejected = cores(5, seed=13), cores(5, seed=14)
    first = ches.pair_identity(chosen, rejected, population_id="p", construction={})
    shuffled = ches.pair_identity(chosen[::-1].copy(), rejected[::-1].copy(),
                                  population_id="p", construction={})
    assert first["chosen_core_order_sha256"] != shuffled["chosen_core_order_sha256"]
    assert first["pairs"] == 5


def test_consumption_is_verified_against_the_earliest_endpoint():
    verified = ches.verify_consumption(pairs_used=4096, cycle=0, cycle_length=25722,
                                       endpoint_pair_exposures={"a": 47424, "b": 61000},
                                       label="seed 1")
    assert verified["verified"] and verified["required_pair_exposures"] == 4096
    with pytest.raises(ValueError, match="fewer than the"):
        ches.verify_consumption(pairs_used=4096, cycle=0, cycle_length=25722,
                                endpoint_pair_exposures={"a": 1000}, label="seed 1")
    with pytest.raises(ValueError, match="do not fit inside"):
        ches.verify_consumption(pairs_used=99999, cycle=0, cycle_length=25722,
                                endpoint_pair_exposures={"a": 99999}, label="seed 1")
    with pytest.raises(ValueError, match="consumption is unverified"):
        ches.verify_consumption(pairs_used=10, cycle=0, cycle_length=100,
                                endpoint_pair_exposures={}, label="seed 1")


def test_repeated_cores_are_counted_because_they_block_a_naive_p_value():
    index = np.repeat(cores(3, seed=15), 2, axis=0)
    repeats = ches.repeated_core_identities(index)
    assert repeats["rows"] == 6 and repeats["distinct_cores"] == 3
    assert repeats["max_repeats"] == 2 and repeats["rows_in_repeated_cores"] == 6
    assert "not independent" in repeats["note"]


def test_within_pair_hamming_is_the_chosen_versus_rejected_distance():
    chosen = np.zeros((2, data.CORE_LENGTH), dtype=np.int64)
    rejected = chosen.copy()
    rejected[0, :3] = 1
    assert list(ches.within_pair_hamming(chosen, rejected)) == [3, 0]


# ---------------------------------------------------------------------------
# association and temporal analysis
# ---------------------------------------------------------------------------

def test_spearman_uses_average_ranks_and_publishes_no_p_value():
    x = np.asarray([1.0, 2.0, 2.0, 4.0])
    y = np.asarray([1.0, 3.0, 3.0, 4.0])
    block = ches.spearman_average_ranks(x, y)
    assert block["spearman"] == pytest.approx(1.0)
    assert block["p_value"] is None and "recur" in block["p_value_reason"]
    assert ches.spearman_average_ranks(np.zeros(4), np.arange(4.0))["spearman"] is None
    assert ches.spearman_average_ranks(np.zeros(2), np.zeros(2))["reason"] == \
        "fewer than three pairs"


def test_displacement_analysis_reports_every_declared_stratum():
    rows = 40
    generator = np.random.default_rng(16)
    parent_ches = generator.normal(size=rows)
    displacement = 0.5 * parent_ches + generator.normal(size=rows) * 0.01
    chosen = cores(rows, seed=17)
    rejected = chosen.copy()
    rejected[:, 0] = (rejected[:, 0] + 1) % 20
    block = ches.displacement_analysis(
        parent_ches=parent_ches, displacement=displacement, chosen_index=chosen,
        rejected_index=rejected, parent_chosen_log_probability=generator.normal(size=rows))
    assert block["overall"]["spearman"] > 0.9
    # set comparison: sorted() on these labels is lexical, so d10 sorts before d2
    assert set(block["by_parent_ches_decile"]["bins"]) == {f"d{k}" for k in range(1, 11)}
    hamming_bins = block["by_within_pair_hamming"]["bins"]
    assert hamming_bins["1"]["rows"] == rows and hamming_bins["0"]["rows"] == 0
    assert "not the nearest-training distance" in block["by_within_pair_hamming"]["definition"]


def test_the_increment_analysis_uses_early_checkpoint_ches_not_the_parents():
    rows = 30
    generator = np.random.default_rng(18)
    early = generator.normal(size=rows)
    later = early + 0.4 + generator.normal(size=rows) * 0.01
    early_ches = np.arange(rows, dtype=np.float64)
    chosen = cores(rows, seed=19)
    block = ches.increment_analysis(
        early, later, label="dpo_seed1::180->600", early_ches=early_ches,
        early_checkpoint={"id": "v1::dpo_seed1::budget180"}, chosen_index=chosen,
        rejected_index=chosen, parent_chosen_log_probability=generator.normal(size=rows),
        parent_ches=generator.normal(size=rows))
    assert block["mean_increment"] == pytest.approx(0.4, abs=0.02)
    assert block["early_checkpoint"]["id"] == "v1::dpo_seed1::budget180"
    assert block["early_ches_association"]["overall"]["n"] == rows
    # the parent comparison is reported beside the early one, never instead of it
    assert "parent_ches_versus_increment" in block
    assert "weaker analysis" in block["parent_ches_note"]


def test_an_increment_without_early_ches_says_so_rather_than_substituting():
    block = ches.increment_analysis(np.zeros(5), np.ones(5), label="x")
    assert block["early_ches_association"] is None
    assert "not substituted by the parent" in block["early_ches_reason"]


def test_a_matched_control_is_actually_subtracted():
    treated = np.asarray([1.0, 2.0, 3.0, 4.0])
    control = np.asarray([0.5, 0.5, 0.5, 0.5])
    block = ches.matched_control_comparison(
        treated, control, label="dpo minus sft", treated_id="dpo", control_id="sft", budget=600.0)
    assert block["matched"] is True
    assert block["mean_difference"] == pytest.approx(2.0)
    assert block["treated_mean"] == pytest.approx(2.5)
    with pytest.raises(ValueError, match="same pairs in the same order"):
        ches.matched_control_comparison(treated, control[:2], label="x", treated_id="a",
                                        control_id="b", budget=600.0)


def test_an_unavailable_control_is_named_and_never_called_matched():
    block = ches.unavailable_control(label="v1_continued_sft_1200", budget=1200.0,
                                     reason="not run: budgets are 180/360/600 only")
    assert block["matched"] is False
    assert "No control is synthesized" in block["consequence"]


def test_training_pair_prefixes_are_fixed_before_scoring():
    class Pairing:
        def cycle_rows(self, cycle):
            return np.arange(10), np.arange(10)[::-1].copy()

        def pair_cores(self, chosen_rows, rejected_rows):
            return (np.tile(chosen_rows[:, None], (1, data.CORE_LENGTH)) % 20,
                    np.tile(rejected_rows[:, None], (1, data.CORE_LENGTH)) % 20)

    block = ches.training_pair_prefix(Pairing(), cycle=0, count=4)
    assert block["count"] == 4 and block["cycle_length"] == 10
    assert block["pair_ids"].shape == (4, 3)
    assert list(block["pair_ids"][0]) == [0, 0, 9]
    with pytest.raises(ValueError, match="fewer than the declared"):
        ches.training_pair_prefix(Pairing(), cycle=0, count=99)
