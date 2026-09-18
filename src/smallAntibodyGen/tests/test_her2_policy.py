"""CorePolicy mechanics on a tiny native GPTNeoX: parity, sampling, seam, checkpoints.

Nothing here downloads a checkpoint. The model is a two-layer GPTNeoX built from a
config in memory, which is the same class the pinned p-IgGen release uses, so the
shared-prefix cache path is exercised against the real implementation rather than
a stand-in. If ``transformers`` is not installed the module skips.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from smallAntibodyGen.experiments import her2_data as data
from smallAntibodyGen.experiments import her2_policy as policy_lib

transformers = pytest.importorskip("transformers", reason="native policy tests need transformers")

#: A character-level vocabulary in the shape of the released one: pad, the twenty
#: canonical residues, then the two biological sentinels. Token id 2 is a residue.
VOCAB = {"<PAD>": 0}
VOCAB.update({residue: index + 1 for index, residue in enumerate(data.CANONICAL)})
VOCAB["1"] = len(VOCAB)
VOCAB["2"] = len(VOCAB)
PREFIX = "1" + "ACDEFGHIKLMNPQ"


def build_policy(seed=0, device="cpu"):
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    torch.manual_seed(seed)
    config = GPTNeoXConfig(vocab_size=len(VOCAB), hidden_size=32, num_hidden_layers=2,
                           num_attention_heads=4, intermediate_size=64,
                           max_position_embeddings=64, hidden_dropout=0.0, attention_dropout=0.0,
                           bos_token_id=0, eos_token_id=VOCAB["C"])
    model = GPTNeoXForCausalLM(config).to(device)
    return policy_lib.CorePolicy.from_prefix(model, PREFIX, VOCAB, device=device)


@pytest.fixture(scope="module")
def policy():
    built = build_policy()
    try:
        built.core_logits(built.token_ids(np.zeros((2, data.CORE_LENGTH), dtype=np.int64)))
    except (AttributeError, TypeError) as error:  # pragma: no cover - old transformers
        pytest.skip(f"this transformers build has no batched prefix cache: {error}")
    return built


def sample_index(count, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 20, size=(count, data.CORE_LENGTH)).astype(np.int64)


# ---------------------------------------------------------------------------
# the model seam refuses to coerce
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad,match", [
    (np.zeros((2, 9), dtype=np.int64), "Core width"),
    (np.zeros((2, 10, 1), dtype=np.int64), "Expected"),
    (np.zeros((0, 10), dtype=np.int64), "Empty core batch"),
    (np.full((1, 10), 3.7), "integral"),
    (np.full((1, 10), -1, dtype=np.int64), "outside"),
    (np.full((1, 10), 20, dtype=np.int64), "outside"),
])
def test_core_index_rejects_malformed_input(policy, bad, match):
    with pytest.raises(ValueError, match=match):
        policy.core_index(bad)


def test_a_negative_index_is_not_silently_the_last_residue(policy):
    """-1 would index residue Y and produce a plausible, wrong number."""
    good = np.full((1, data.CORE_LENGTH), 19, dtype=np.int64)
    bad = np.full((1, data.CORE_LENGTH), -1, dtype=np.int64)
    reference = policy.score(good)["sum_log_probability"]
    assert np.isfinite(reference).all()
    with pytest.raises(ValueError):
        policy.score(bad)


# ---------------------------------------------------------------------------
# cached shared prefix vs full teacher forcing
# ---------------------------------------------------------------------------

def gradients_of(policy, loss):
    policy.model.zero_grad(set_to_none=True)
    loss.backward()
    grads = {name: parameter.grad.detach().clone()
             for name, parameter in policy.model.named_parameters() if parameter.grad is not None}
    policy.model.zero_grad(set_to_none=True)
    return grads


def assert_parity(policy, index, tolerance=1e-4):
    cached_loss = policy.loss(index, cached=True)
    cached = gradients_of(policy, cached_loss)
    full_loss = policy.loss(index, cached=False)
    full = gradients_of(policy, full_loss)
    assert set(cached) == set(full), "the two paths touched different parameters"
    assert cached, "no gradients were produced at all"
    assert max(float(value.norm()) for value in full.values()) > 0, "all gradients are zero"
    assert float(cached_loss) == pytest.approx(float(full_loss), abs=tolerance)
    worst = max(float((cached[name] - full[name]).abs().max()) for name in cached)
    assert worst < tolerance, f"largest gradient disagreement {worst}"
    return float(full_loss)


def test_cached_and_full_paths_agree_on_loss_and_all_gradients(policy):
    assert_parity(policy, sample_index(6))


def test_parity_survives_a_real_optimizer_update():
    """Parity at initialization is not parity while training; take a real step."""
    policy = build_policy(seed=3)
    index = sample_index(6, seed=1)
    before = assert_parity(policy, index)
    optimizer = torch.optim.AdamW(policy.model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    policy.loss(index).backward()
    optimizer.step()
    after = assert_parity(policy, index)
    assert after != before, "the optimizer step did not change the loss"


def test_scoring_is_invariant_to_the_scoring_batch_size(policy):
    index = sample_index(11, seed=5)
    one = policy.score(index, batch_size=1)["sum_log_probability"]
    seven = policy.score(index, batch_size=7)["sum_log_probability"]
    # The production comparison, on the production tolerance: batching changes the
    # fp32 kernels, so this is a bounded-difference claim rather than an equality.
    report = policy_lib.compare_sum_log_probabilities(one, seven, label="batch 1 vs 7")
    assert report["max_abs_error"] < policy_lib.SUM_LOG_PROBABILITY_ATOL


def test_mean_is_the_sum_over_ten_positions(policy):
    index = sample_index(4, seed=6)
    scored = policy.score(index)
    assert np.allclose(scored["sum_log_probability"] / data.CORE_LENGTH,
                       scored["mean_log_probability"])
    sequence = policy.sequence_log_probs(index).detach().numpy()
    assert np.max(np.abs(sequence - scored["sum_log_probability"])) < 1e-5


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------

def test_sampler_and_scorer_agree_within_the_declared_tolerance(policy):
    """The sampler records the temperature-1 density; re-scoring must reproduce it.

    "Exactly" was never the claim available in fp32: the two routes run different
    kernels over the same weights. The production helper states the tolerance and
    records the measured error instead of hiding it behind a bare ``<``.
    """
    index, logps = policy.sample(24, seed=11, batch_size=8)
    rescored = policy.score(index)["sum_log_probability"]
    report = policy_lib.compare_sum_log_probabilities(rescored, logps,
                                                      label="sampler vs scorer")
    assert report["rows"] == 24
    assert report["max_abs_error"] < policy_lib.SUM_LOG_PROBABILITY_ATOL


def test_sampling_is_reproducible_and_temperature_does_not_move_the_recorded_density(policy):
    first, first_logps = policy.sample(16, seed=7)
    second, second_logps = policy.sample(16, seed=7)
    assert np.array_equal(first, second)
    assert np.allclose(first_logps, second_logps)
    hot, hot_logps = policy.sample(16, seed=7, temperature=2.0)
    rescored = policy.score(hot)["sum_log_probability"]
    assert np.max(np.abs(rescored - hot_logps)) < 1e-5, "the recorded density must be temperature 1"


def test_generation_always_produces_ten_residues_and_an_r_cannot_stop_it(policy):
    """The shipped sentinel ids collide with residues; decoding must ignore them."""
    index, _ = policy.sample(200, seed=13, batch_size=64)
    cores = data.decode_cores(index)
    assert all(len(core) == data.CORE_LENGTH for core in cores)
    assert index.shape == (200, data.CORE_LENGTH)
    assert any("R" in core for core in cores), "R never sampled; the test cannot say anything"
    assert policy.model.config.eos_token_id == VOCAB["C"]
    assert any("C" in core for core in cores), "the configured eos id is a residue and must decode"


def test_sample_rejects_a_nonpositive_count_or_temperature(policy):
    with pytest.raises(ValueError, match="Positive draw count"):
        policy.sample(0, seed=1)
    with pytest.raises(ValueError, match="Temperature"):
        policy.sample(4, seed=1, temperature=0.0)


# ---------------------------------------------------------------------------
# construction guards and checkpoints
# ---------------------------------------------------------------------------

def test_nonzero_dropout_is_refused_at_construction():
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    config = GPTNeoXConfig(vocab_size=len(VOCAB), hidden_size=32, num_hidden_layers=1,
                           num_attention_heads=4, intermediate_size=64,
                           max_position_embeddings=64, hidden_dropout=0.1, attention_dropout=0.0)
    with pytest.raises(ValueError, match="zero hidden/attention dropout"):
        policy_lib.CorePolicy.from_prefix(GPTNeoXForCausalLM(config), PREFIX, VOCAB)


def test_checkpoint_round_trip_is_strict_and_digest_checked(tmp_path):
    policy = build_policy(seed=4)
    index = sample_index(5, seed=9)
    before = policy.score(index)["sum_log_probability"]
    path = tmp_path / "epoch_1.pt"
    digest = policy_lib.save_checkpoint(path, policy, {"epoch": 1})
    restored = build_policy(seed=99)
    payload = policy_lib.load_checkpoint(path, restored.model)
    assert payload["state_sha256"] == digest
    after = restored.score(index)["sum_log_probability"]
    assert np.max(np.abs(before - after)) < 1e-6


def test_checkpoint_reload_rejects_a_foreign_architecture(tmp_path):
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    policy = build_policy(seed=4)
    path = tmp_path / "epoch_1.pt"
    policy_lib.save_checkpoint(path, policy, {"epoch": 1})
    config = GPTNeoXConfig(vocab_size=len(VOCAB), hidden_size=16, num_hidden_layers=1,
                           num_attention_heads=4, intermediate_size=32,
                           max_position_embeddings=64, hidden_dropout=0.0, attention_dropout=0.0)
    with pytest.raises(RuntimeError):
        policy_lib.load_checkpoint(path, GPTNeoXForCausalLM(config))


def test_checkpoint_schema_is_checked(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"schema_version": "something-else/9", "state": {}}, path)
    policy = build_policy(seed=4)
    with pytest.raises(ValueError, match="Unsupported checkpoint schema"):
        policy_lib.load_checkpoint(path, policy.model)


# ---------------------------------------------------------------------------
# schedules
# ---------------------------------------------------------------------------

def test_continuation_schedule_warms_up_then_stays_constant():
    assert policy_lib.continuation_learning_rate_scale(1, warmup_steps=100) == pytest.approx(0.01)
    assert policy_lib.continuation_learning_rate_scale(100, warmup_steps=100) == pytest.approx(1.0)
    for step in (101, 500, 20000):
        assert policy_lib.continuation_learning_rate_scale(step, warmup_steps=100) == 1.0


def test_continuation_schedule_does_not_depend_on_an_unknown_horizon():
    """A budget-stopped run has no total step count, so the schedule must not need one."""
    early = [policy_lib.continuation_learning_rate_scale(s, warmup_steps=10) for s in range(1, 40)]
    assert early[-1] == early[-2] == 1.0
