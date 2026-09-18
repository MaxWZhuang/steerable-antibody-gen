"""Preference pairing, the frozen reference cache, DPO mechanics and budget accounting.

Everything here runs on the CPU in a few seconds and downloads nothing. The GPU
budget clock is driven by an injected deterministic counter, which is the only way
to assert exact budget arithmetic without a CUDA device; production refuses the
injection and requires real CUDA events.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

from smallAntibodyGen.experiments import her2_data as data
from smallAntibodyGen.experiments import her2_policy as policy_lib
from smallAntibodyGen.experiments import her2_preferences as preferences
from smallAntibodyGen.experiments.her2_runtime import GpuBudgetClock


def cores(seed, count):
    rng = np.random.default_rng(seed)
    letters = np.array(list(data.CANONICAL))
    seen, rows = set(), []
    while len(rows) < count:
        core = "".join(rng.choice(letters, size=data.CORE_LENGTH))
        if core not in seen:
            seen.add(core)
            rows.append(core)
    return rows


def frame_with(rows):
    """rows: iterable of (core, class). Distances are whatever the cores imply."""
    return pd.DataFrame({"seq": [core for core, _ in rows],
                         "class": [name for _, name in rows]})


def mutate(core, positions):
    letters = list(core)
    for position in positions:
        letters[position] = data.CANONICAL[(data.CANONICAL.index(letters[position]) + 1) % 20]
    return "".join(letters)


@pytest.fixture
def population():
    """High and low rows at distances 1 and 2, plus one unmatchable high at distance 3."""
    rows = []
    for position in range(4):
        rows.append((mutate(data.WT_CORE, [position]), "high"))
    for position in range(4, 6):
        rows.append((mutate(data.WT_CORE, [position]), "low"))
    for position in range(4):
        rows.append((mutate(data.WT_CORE, [position, 7]), "high"))
    for position in range(4, 9):
        rows.append((mutate(data.WT_CORE, [position, 9]), "low"))
    rows.append((mutate(data.WT_CORE, [0, 1, 2]), "high"))   # distance 3, no low partner
    rows.append((mutate(data.WT_CORE, [3, 4]), "mid"))       # mid is never a rejected row
    return preferences.build_population(frame_with(rows), "train")


# ---------------------------------------------------------------------------
# eligibility and pairing
# ---------------------------------------------------------------------------

def test_population_excludes_high_rows_with_no_low_partner(population):
    assert population.chosen_index.shape[0] == 8
    assert population.rejected_index.shape[0] == 7
    assert population.excluded_chosen_distances == {3: 1}
    assert set(np.unique(population.chosen_distance)) == {1, 2}


def test_mid_is_never_a_rejected_row(population):
    rejected = set(data.decode_cores(population.rejected_index))
    assert mutate(data.WT_CORE, [3, 4]) not in rejected


def test_pairs_never_cross_distance_groups_or_splits(population):
    pairing = preferences.PreferencePairing(population, seed=11)
    for cycle in range(4):
        chosen_rows, rejected_rows = pairing.cycle_rows(cycle)
        assert (population.chosen_distance[chosen_rows]
                == population.rejected_distance[rejected_rows]).all()


def test_every_chosen_row_is_visited_exactly_once_per_cycle(population):
    pairing = preferences.PreferencePairing(population, seed=11)
    for cycle in range(3):
        chosen_rows, _ = pairing.cycle_rows(cycle)
        assert sorted(chosen_rows.tolist()) == list(range(population.chosen_index.shape[0]))


def test_the_shuffle_is_deterministic_and_seed_dependent(population):
    a = preferences.PreferencePairing(population, seed=11).cycle_rows(0)
    b = preferences.PreferencePairing(population, seed=11).cycle_rows(0)
    c = preferences.PreferencePairing(population, seed=12).cycle_rows(0)
    assert (a[0] == b[0]).all() and (a[1] == b[1]).all()
    assert not ((a[0] == c[0]).all() and (a[1] == c[1]).all())


def test_a_long_run_eventually_covers_every_low_row(population):
    """Coverage is a property of the rotation, not a hope about randomness."""
    pairing = preferences.PreferencePairing(population, seed=11)
    seen = set()
    for cycle in range(6):
        _, rejected_rows = pairing.cycle_rows(cycle)
        seen.update(rejected_rows.tolist())
    assert seen == set(range(population.rejected_index.shape[0]))


def test_partners_rotate_rather_than_repeat(population):
    """The distance-2 group has five lows and four chosen, so one cycle cannot reach all."""
    pairing = preferences.PreferencePairing(population, seed=11)
    first = set(pairing.cycle_rows(0)[1].tolist())
    second = set(pairing.cycle_rows(1)[1].tolist())
    assert first != second
    assert len(first | second) == population.rejected_index.shape[0]


def test_the_incomplete_tail_batch_is_included(population):
    pairing = preferences.PreferencePairing(population, seed=11)
    batches = list(pairing.batches(0, 3))
    assert [len(chosen) for chosen, _ in batches] == [3, 3, 2]
    assert sum(len(chosen) for chosen, _ in batches) == population.chosen_index.shape[0]


def test_validation_pairs_are_fixed(population):
    pairing = preferences.PreferencePairing(population, seed=99)
    first = pairing.fixed_validation_pairs()
    second = pairing.fixed_validation_pairs()
    assert first["digest"] == second["digest"]
    assert first["pairs"] == population.chosen_index.shape[0]


def test_a_population_with_no_low_rows_has_no_eligible_pairs():
    """With nothing to reject, every high row is unmatchable and pairing refuses."""
    frame = frame_with([(core, "high") for core in cores(3, 4)])
    population = preferences.build_population(frame, "train")
    assert population.chosen_index.shape[0] == 0
    assert sum(population.excluded_chosen_distances.values()) == 4
    with pytest.raises(ValueError, match="No chosen rows to pair"):
        preferences.PreferencePairing(population, seed=1)


# ---------------------------------------------------------------------------
# the toy policy used for cache and loss mechanics
# ---------------------------------------------------------------------------

class ToyPolicy:
    """A differentiable stand-in with the same log-probability contract as CorePolicy.

    ``scale=0`` is the uniform policy. It has to be built from a fixed random
    pattern rather than a constant fill, because log-softmax of any constant
    vector is the same distribution -- a "different" constant policy would score
    identically and could not detect a moved reference.
    """

    def __init__(self, scale=0.0, seed=0):
        generator = torch.Generator().manual_seed(seed)
        base = torch.randn(data.CORE_LENGTH, 20, generator=generator)
        self.weight = torch.nn.Parameter(base * float(scale))
        self.model = torch.nn.Module()
        self.model.weight = self.weight
        self.device = torch.device("cpu")

    def position_log_probs(self, index):
        values = torch.as_tensor(np.asarray(index), dtype=torch.long)
        logits = self.weight.unsqueeze(0).expand(values.shape[0], -1, -1)
        return torch.log_softmax(logits, dim=-1).gather(2, values.unsqueeze(-1)).squeeze(-1)

    def sequence_log_probs(self, index):
        return self.position_log_probs(index).sum(dim=1)

    def loss(self, index):
        return -self.position_log_probs(index).mean()


# ---------------------------------------------------------------------------
# reference cache
# ---------------------------------------------------------------------------

def make_cache(tmp_path, policy, index):
    identity = preferences.reference_identity(
        checkpoint_sha256="a" * 64, config_sha256="b" * 64, scaffold_prefix="1AAA", index=index)
    clock = GpuBudgetClock(clock=iter_clock([0.0, 2.5]))
    return preferences.build_reference_cache(policy, index, identity, clock=clock), identity


def iter_clock(values):
    """A deterministic monotonic counter: successive calls return successive values."""
    state = {"position": 0, "values": list(values)}

    def clock():
        value = state["values"][min(state["position"], len(state["values"]) - 1)]
        state["position"] += 1
        return value
    return clock


def test_reference_cache_is_immutable_and_identity_bound(tmp_path):
    index = data.encode_cores(cores(5, 12))
    policy = ToyPolicy()
    cache, identity = make_cache(tmp_path, policy, index)
    assert cache.values.shape == (12,)
    assert cache.creation_gpu_seconds == pytest.approx(2.5)
    with pytest.raises(ValueError):
        cache.values[0] = 0.0
    path = preferences.save_reference_cache(tmp_path / "reference_cache.npy", cache)
    with pytest.raises(ValueError, match="immutable"):
        preferences.save_reference_cache(path, cache)
    reloaded = preferences.load_reference_cache(path, identity)
    assert np.array_equal(reloaded.values, cache.values)
    assert reloaded.reused is True
    # The original creation cost is what the cold-start budget is charged.
    assert reloaded.creation_gpu_seconds == pytest.approx(2.5)
    # The warm reuse cost is measured around the completed read, digest and identity
    # check. Computing it in the caller's argument list -- the previous shape of this
    # API -- subtracted two readings taken before the load and always reported 0.
    assert reloaded.warm_reuse_wall_seconds > 0.0


def test_reference_cache_rejects_a_different_checkpoint_or_order(tmp_path):
    index = data.encode_cores(cores(5, 12))
    cache, identity = make_cache(tmp_path, ToyPolicy(), index)
    path = preferences.save_reference_cache(tmp_path / "reference_cache.npy", cache)
    other = dict(identity, checkpoint_sha256="c" * 64)
    with pytest.raises(ValueError, match="identity mismatch"):
        preferences.load_reference_cache(path, other)
    shuffled = preferences.reference_identity(
        checkpoint_sha256="a" * 64, config_sha256="b" * 64, scaffold_prefix="1AAA",
        index=index[::-1])
    with pytest.raises(ValueError, match="identity mismatch"):
        preferences.load_reference_cache(path, shuffled)


def test_reference_cache_rejects_tampered_contents(tmp_path):
    index = data.encode_cores(cores(5, 12))
    cache, identity = make_cache(tmp_path, ToyPolicy(), index)
    path = preferences.save_reference_cache(tmp_path / "reference_cache.npy", cache)
    np.save(path, np.zeros_like(cache.values) - 1.0)
    with pytest.raises(ValueError, match="do not match the digest"):
        preferences.load_reference_cache(path, identity)


def test_reference_parity_catches_a_moved_reference(tmp_path):
    index = data.encode_cores(cores(5, 12))
    cache, _ = make_cache(tmp_path, ToyPolicy(), index)
    report = preferences.verify_reference_parity(ToyPolicy(), cache, index, [0, 3, 7])
    assert report["max_abs_error"] == pytest.approx(0.0)
    # The declared tolerance is recorded with the measurement, not left implicit.
    assert report["atol"] == pytest.approx(policy_lib.SUM_LOG_PROBABILITY_ATOL)
    assert report["rtol"] == pytest.approx(policy_lib.SUM_LOG_PROBABILITY_RTOL)
    with pytest.raises(ValueError, match="disagrees with a fresh score"):
        preferences.verify_reference_parity(ToyPolicy(scale=1.0), cache, index, [0, 3, 7])


def test_reference_parity_still_rejects_a_small_but_real_disagreement(tmp_path):
    """The looser sum-log-p tolerance must not become 'anything close enough'."""
    index = data.encode_cores(cores(5, 12))
    cache, _ = make_cache(tmp_path, ToyPolicy(), index)
    moved = np.array(cache.values, dtype=np.float64)
    moved[2] -= 1e-3                                    # 20x the fp32 noise floor
    moved.setflags(write=False)
    nudged = preferences.ReferenceCache(identity=cache.identity, values=moved,
                                        creation_gpu_seconds=cache.creation_gpu_seconds)
    with pytest.raises(ValueError, match="exceeds the declared tolerance"):
        preferences.verify_reference_parity(ToyPolicy(), nudged, index, [0, 2, 7])


def test_reference_values_carry_no_gradient(tmp_path):
    index = data.encode_cores(cores(5, 8))
    cache, _ = make_cache(tmp_path, ToyPolicy(), index)
    tensor = cache.tensor([0, 1, 2], device=torch.device("cpu"))
    assert tensor.requires_grad is False


# ---------------------------------------------------------------------------
# DPO loss
# ---------------------------------------------------------------------------

def test_dpo_loss_is_log_two_at_the_reference():
    """At initialization the policy IS the reference, so every margin is zero."""
    policy = ToyPolicy()
    index = data.encode_cores(cores(9, 6))
    other = data.encode_cores(cores(10, 6))
    reference_chosen = policy.sequence_log_probs(index).detach()
    reference_rejected = policy.sequence_log_probs(other).detach()
    loss, statistics = preferences.dpo_batch(policy, index, other, reference_chosen,
                                             reference_rejected, beta=0.1)
    assert float(loss) == pytest.approx(math.log(2.0), abs=1e-6)
    assert statistics["pairs"] == 6
    assert statistics["mean_margin"] == pytest.approx(0.0, abs=1e-6)


def test_dpo_update_moves_the_policy_and_produces_gradients():
    policy = ToyPolicy()
    index = data.encode_cores(cores(9, 6))
    other = data.encode_cores(cores(10, 6))
    reference_chosen = policy.sequence_log_probs(index).detach()
    reference_rejected = policy.sequence_log_probs(other).detach()
    loss, _ = preferences.dpo_batch(policy, index, other, reference_chosen, reference_rejected,
                                    beta=0.1)
    loss.backward()
    assert policy.weight.grad is not None
    assert float(policy.weight.grad.abs().max()) > 0
    before = policy.weight.detach().clone()
    torch.optim.SGD([policy.weight], lr=1.0).step()
    assert not torch.equal(before, policy.weight.detach())
    after, _ = preferences.dpo_batch(policy, index, other, reference_chosen, reference_rejected,
                                     beta=0.1)
    assert float(after) < float(loss)


def test_dpo_refuses_a_reference_that_carries_gradient():
    policy = ToyPolicy()
    index = data.encode_cores(cores(9, 4))
    other = data.encode_cores(cores(10, 4))
    live = policy.sequence_log_probs(index)
    with pytest.raises(ValueError, match="frozen"):
        preferences.dpo_batch(policy, index, other, live, live, beta=0.1)


# ---------------------------------------------------------------------------
# budget accounting
# ---------------------------------------------------------------------------

def stepping_clock(step_seconds):
    """Returns t, t, t+dt, t+dt, ... so each segment measures exactly ``step_seconds``."""
    state = {"now": 0.0, "open": False}

    def clock():
        if not state["open"]:
            state["open"] = True
            return state["now"]
        state["open"] = False
        state["now"] += step_seconds
        return state["now"]
    return clock


def trajectory(budgets, step_seconds=1.0, max_updates=100, precharge=0.0, track_rows=False):
    policy = ToyPolicy()
    index = data.encode_cores(cores(4, 8))
    optimizer = torch.optim.SGD([policy.weight], lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    clock = GpuBudgetClock(clock=stepping_clock(step_seconds))
    if precharge:
        clock.charge_reused(precharge, reason="reused frozen reference cache")
    reached = []
    seen = set()

    def step(update):
        # Two rows per update, sweeping a population of four: the distinct count
        # grows and then stops, which a per-batch unique count cannot express.
        rows = [(2 * (update - 1)) % 4, (2 * update - 1) % 4]
        seen.update(rows)
        return policy.loss(index), {"sequences": 8, "batch_unique_chosen": len(set(rows))}

    return preferences.run_budgeted_trajectory(
        step=step, budgets=budgets, clock=clock, optimizer=optimizer, scheduler=scheduler,
        model=policy.model, gradient_clip=1.0, exposure_fields=("sequences",),
        cumulative_exposures=(lambda: {"distinct_chosen_rows": len(seen), "chosen_population": 4})
        if track_rows else None,
        on_budget=lambda budget, record: reached.append(budget) or {"checkpoint": f"b{budget}"},
        max_updates=max_updates), reached


def test_one_trajectory_passes_through_every_budget_without_restarting():
    result, reached = trajectory([3.0, 6.0, 10.0])
    assert reached == [3.0, 6.0, 10.0]
    assert result["updates"] == 10
    assert [record["updates"] for record in result["budgets"].values()] == [3, 6, 10]
    # The later budget is a continuation: its update count includes the earlier one's.
    assert result["budgets"]["10.0"]["updates"] > result["budgets"]["3.0"]["updates"]
    assert result["exposures"] == {"sequences": 80}


def test_a_budget_may_overshoot_by_at_most_one_update():
    result, _ = trajectory([2.5], step_seconds=1.0)
    record = result["budgets"]["2.5"]
    assert record["actual_gpu_seconds"] == pytest.approx(3.0)
    assert record["target_gpu_seconds"] == 2.5
    assert 0 < record["overshoot_seconds"] <= 1.0
    # The bound is checked and recorded, not only asserted in prose.
    assert record["overshoot_within_one_update"] is True
    assert record["last_update_gpu_seconds"] == pytest.approx(1.0)


def test_distinct_exposures_are_recorded_at_every_budget():
    """A per-batch unique count says nothing about what a budget consumed."""
    result, _ = trajectory([1.0, 3.0], track_rows=True)
    first = result["budgets"]["1.0"]["distinct_exposures"]
    second = result["budgets"]["3.0"]["distinct_exposures"]
    assert first["distinct_chosen_rows"] == 2
    assert second["distinct_chosen_rows"] == 4
    assert second["chosen_population"] == 4
    assert result["distinct_exposures"]["distinct_chosen_rows"] == 4
    # Sequence and core-token exposures are cumulative and are not the distinct count.
    assert result["budgets"]["3.0"]["exposures"]["sequences"] == 24
    assert result["budgets"]["3.0"]["core_token_exposures"] == 24 * data.CORE_LENGTH


def test_trajectory_reports_wall_time_and_the_costs_it_excludes():
    result, _ = trajectory([1.0, 2.0])
    assert result["trajectory_wall_seconds"] > 0.0
    assert result["excluded_evaluation_wall_seconds"] >= 0.0
    # Each budget's own checkpoint/validation cost is measured around the callback.
    assert all("budget_evaluation_wall_seconds" in record
               for record in result["budgets"].values())
    # ...and it is never charged to the GPU budget, which stays the injected count.
    assert result["gpu_seconds"] == pytest.approx(2.0)


def test_a_precharged_reference_that_eats_the_first_budget_fails_honestly():
    """If reference scoring alone exceeds the smallest budget, say so; do not label
    the first update's checkpoint '180 s' and call the overshoot one update."""
    with pytest.raises(ValueError, match="already charged"):
        trajectory([3.0, 6.0], precharge=4.0)


def test_a_precharge_inside_the_first_budget_is_charged_not_forgiven():
    result, _ = trajectory([3.0], precharge=1.5)
    assert result["precharged_gpu_seconds"] == pytest.approx(1.5)
    # 1.5 charged + two 1.0 s updates crosses 3.0; the cache cost is part of the budget.
    assert result["updates"] == 2
    assert result["budgets"]["3.0"]["actual_gpu_seconds"] == pytest.approx(3.5)


def test_checkpoint_work_is_not_charged_to_the_training_budget():
    """``on_budget`` runs outside the clock, so its cost cannot shorten training."""
    result, _ = trajectory([3.0])
    assert result["gpu_seconds"] == pytest.approx(3.0)
    assert result["budgets"]["3.0"]["updates"] == 3


def test_the_budget_clock_requires_cuda_in_production(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="measured GPU budget"):
        GpuBudgetClock()


def test_trajectory_needs_at_least_one_budget():
    with pytest.raises(ValueError, match="at least one budget"):
        trajectory([])


def test_validation_pair_metrics_report_accuracy_and_margins():
    policy = ToyPolicy()
    chosen = data.encode_cores(cores(21, 5))
    rejected = data.encode_cores(cores(22, 5))
    pairs = {"chosen_index": chosen, "rejected_index": rejected}
    reference_chosen = policy.sequence_log_probs(chosen).detach().numpy()
    reference_rejected = policy.sequence_log_probs(rejected).detach().numpy()
    document = preferences.validation_pair_metrics(
        policy, pairs, reference_chosen=reference_chosen, reference_rejected=reference_rejected,
        beta=0.1)
    assert document["pairs"] == 5
    assert document["mean_implicit_margin"] == pytest.approx(0.0, abs=1e-6)
    assert document["mean_dpo_loss"] == pytest.approx(math.log(2.0), abs=1e-6)
    # A uniform toy policy scores every core identically, so no pair is "won".
    assert document["pair_ties"] == 5
