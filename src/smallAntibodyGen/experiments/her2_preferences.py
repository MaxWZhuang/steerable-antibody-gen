"""Preference pairs, the frozen reference cache, and GPU-budgeted continuation.

This module is the post-training amendment: after the initial SFT selection, each
seed's selected checkpoint is cloned twice and continued under a **measured GPU
elapsed budget** -- once with more SFT on the same positives, once with DPO.

The things that are easy to get wrong, and are therefore fixed here:

* **A preference pair is high-vs-low at the same wild-type distance.** ``mid`` is
  kept for the three-class classifier and is *not* treated as a non-binder, so it
  never appears as a rejected example. Matching on Hamming distance to the
  wild-type core removes the trivial shortcut in which "chosen" simply means
  "closer to trastuzumab": within a pair that feature is constant.
* **Every chosen row is visited once per cycle, and the low set is covered
  eventually.** The chosen order is a seeded shuffle; the low partner for the
  k-th chosen inside a distance group is read from a fixed seeded permutation of
  that group's low rows at an offset that advances by the group's chosen count
  each cycle. So a group with ``m`` low rows and ``n`` chosen rows has covered
  every low row after ``ceil(m / n)`` cycles -- coverage is a property of the
  construction, not a hope about randomness.
* **The reference is a cache, not a callback.** DPO's reference term must be the
  *frozen* selected checkpoint. Scoring it with the live model would make the
  reference move with the policy and silently turn the objective into something
  else, so the reference log probabilities are computed once, bound to the
  checkpoint/scaffold/convention that produced them, and are immutable.
* **Budgets are device-elapsed seconds, checked between updates.** A trajectory
  runs once and drops a checkpoint as it passes each budget; it is never restarted
  per budget, so the 1800 s checkpoint is a genuine continuation of the 180 s one.
  The recorded number is the actual elapsed time, which may overshoot the target
  by at most one update.

No equal-update or equal-epoch claim is made anywhere. Two methods that get the
same measured GPU seconds take different numbers of steps, and that is the point.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .dpo import dpo_per_pair_loss
from .her2_data import CORE_LENGTH, WT_CORE, decode_cores, encode_cores, hamming_to
from .her2_policy import (SUM_LOG_PROBABILITY_ATOL, SUM_LOG_PROBABILITY_RTOL,
                          compare_sum_log_probabilities)
from .her2_runtime import canonical_json, load_json, require, save_json

PREFERENCE_SCHEMA = "her2-preferences/1"
REFERENCE_CACHE_SCHEMA = "her2-reference-cache/1"
#: Both continuation methods share this; it is recorded so the reference cache can
#: refuse to be reused under a different scoring contract.
PROBABILITY_CONVENTION = "sum_log_probability_over_10_core_positions_20way_renormalized"


def _digest(payload):
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def array_digest(values):
    """Content hash of a float array, at full precision and in order."""
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return hashlib.sha256(array.tobytes()).hexdigest()


def core_digest(index):
    """Content hash of a core index block. Order is part of the identity."""
    array = np.ascontiguousarray(np.asarray(index, dtype=np.int8))
    return hashlib.sha256(array.tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# eligible populations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PreferencePopulation:
    """One split's chosen/rejected rows, already filtered to matchable distances."""

    split: str
    chosen_index: np.ndarray
    chosen_distance: np.ndarray
    rejected_index: np.ndarray
    rejected_distance: np.ndarray
    excluded_chosen_distances: dict

    def document(self):
        return {"split": self.split, "chosen_rows": int(self.chosen_index.shape[0]),
                "rejected_rows": int(self.rejected_index.shape[0]),
                "excluded_chosen_rows": int(sum(self.excluded_chosen_distances.values())),
                "excluded_chosen_by_distance": {str(k): int(v) for k, v in
                                                sorted(self.excluded_chosen_distances.items())},
                "chosen_digest": core_digest(self.chosen_index),
                "rejected_digest": core_digest(self.rejected_index),
                "matched_on": "hamming distance to the wild-type core",
                "rejected_class": "low only; mid is never treated as a non-binder"}


def build_population(frame, split, *, positive_class="high", negative_class="low"):
    """Split a labelled frame into chosen/rejected rows that can actually be paired.

    A high-bin row at a distance where the split has **no** low-bin row cannot be
    matched without breaking the distance-matching rule, so it is dropped and the
    drop is counted by distance. Silently pairing it against some other distance
    would reintroduce exactly the confound the matching removes.
    """
    index = encode_cores(frame.seq)
    distance = hamming_to(index, encode_cores([WT_CORE])[0])
    chosen_mask = (frame["class"] == positive_class).to_numpy()
    rejected_mask = (frame["class"] == negative_class).to_numpy()
    rejected_distances = np.unique(distance[rejected_mask])
    matchable = np.isin(distance, rejected_distances)
    dropped = distance[chosen_mask & ~matchable]
    excluded = {int(d): int(c) for d, c in zip(*np.unique(dropped, return_counts=True))}
    keep = chosen_mask & matchable
    return PreferencePopulation(
        split=split, chosen_index=index[keep], chosen_distance=distance[keep],
        rejected_index=index[rejected_mask], rejected_distance=distance[rejected_mask],
        excluded_chosen_distances=excluded)


# ---------------------------------------------------------------------------
# pairing
# ---------------------------------------------------------------------------

class PreferencePairing:
    """Deterministic distance-matched pairing with per-cycle rotation of partners.

    One *cycle* visits every chosen row exactly once, in a seeded shuffle. Within
    a cycle, the chosen rows of one distance group are handed successive entries
    of that group's fixed permutation of low rows, starting at an offset that
    advances by the group's size each cycle. Two consequences, both checked by
    tests rather than asserted here: no chosen row is skipped or repeated inside a
    cycle, and after ``ceil(m / n)`` cycles every low row in a group of size ``m``
    with ``n`` chosen rows has been used at least once.
    """

    def __init__(self, population, *, seed):
        require(population.chosen_index.shape[0] > 0, "No chosen rows to pair")
        require(population.rejected_index.shape[0] > 0, "No rejected rows to pair")
        self.population = population
        self.seed = int(seed)
        self.groups = {}
        for value in sorted(set(population.chosen_distance.tolist())):
            rejected_rows = np.flatnonzero(population.rejected_distance == value)
            require(rejected_rows.size > 0,
                    f"Distance {value} has chosen rows but no rejected partner; the population "
                    "builder should have excluded them")
            permutation = np.random.default_rng([self.seed, 7, int(value)]).permutation(
                rejected_rows.size)
            self.groups[int(value)] = rejected_rows[permutation]

    @property
    def chosen_count(self):
        return int(self.population.chosen_index.shape[0])

    def cycle_rows(self, cycle):
        """``(chosen_rows, rejected_rows)`` for one full pass over the chosen rows."""
        require(isinstance(cycle, (int, np.integer)) and cycle >= 0, "Cycle must be >= 0")
        order = np.random.default_rng([self.seed, int(cycle)]).permutation(self.chosen_count)
        distances = self.population.chosen_distance[order]
        partners = np.empty(order.size, dtype=np.int64)
        for value, pool in self.groups.items():
            positions = np.flatnonzero(distances == value)
            if positions.size == 0:
                continue
            # The offset advances by the group's chosen count every cycle, so the
            # pool is swept rather than resampled: nothing is structurally unreachable.
            start = (int(cycle) * positions.size) % pool.size
            partners[positions] = pool[(start + np.arange(positions.size)) % pool.size]
        return order, partners

    def batches(self, cycle, batch_pairs):
        """Chunk one cycle into batches. The final short batch is yielded, not dropped."""
        require(isinstance(batch_pairs, int) and batch_pairs > 0, "Positive batch size required")
        order, partners = self.cycle_rows(cycle)
        for start in range(0, order.size, batch_pairs):
            yield order[start:start + batch_pairs], partners[start:start + batch_pairs]

    def stream(self, batch_pairs, *, start_cycle=0):
        """Endless ``(cycle, batch_number, chosen_rows, rejected_rows)`` batches."""
        cycle = int(start_cycle)
        while True:
            for number, (chosen, rejected) in enumerate(self.batches(cycle, batch_pairs)):
                yield cycle, number, chosen, rejected
            cycle += 1

    def pair_cores(self, chosen_rows, rejected_rows):
        return (self.population.chosen_index[chosen_rows],
                self.population.rejected_index[rejected_rows])

    def fixed_validation_pairs(self, count=None):
        """One frozen validation pairing, built once and never rotated."""
        order, partners = self.cycle_rows(0)
        if count is not None:
            order, partners = order[:count], partners[:count]
        chosen, rejected = self.pair_cores(order, partners)
        require(bool((self.population.chosen_distance[order]
                      == self.population.rejected_distance[partners]).all()),
                "A validation pair crossed distance groups")
        return {"chosen_index": chosen, "rejected_index": rejected,
                "pairs": int(order.size),
                "digest": _digest({"chosen": core_digest(chosen),
                                   "rejected": core_digest(rejected)})}


# ---------------------------------------------------------------------------
# frozen reference cache
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReferenceCache:
    """Sequence log probabilities of the frozen reference, bound to their identity."""

    identity: dict
    values: np.ndarray
    creation_gpu_seconds: float
    reused: bool = False
    warm_reuse_wall_seconds: float = 0.0

    def lookup(self, rows):
        """Detached reference values for a batch, as a plain read-only array."""
        return self.values[np.asarray(rows)]

    def tensor(self, rows, *, device, dtype=torch.float32):
        # Copy on the way out: the stored array is deliberately read-only, and
        # wrapping a read-only buffer in a tensor is exactly how an "immutable"
        # cache acquires an in-place write later.
        values = np.array(self.lookup(rows), dtype=np.float64)
        return torch.as_tensor(values, device=device).to(dtype)

    def document(self):
        return {"identity": self.identity, "rows": int(self.values.size),
                "creation_gpu_seconds": float(self.creation_gpu_seconds),
                "reused_from_disk": bool(self.reused),
                "warm_reuse_wall_seconds": float(self.warm_reuse_wall_seconds),
                "budget_note": ("the original creation cost is charged to the reported cold-start "
                                "budget even when the cache is reused; the warm reuse wall time is "
                                "reported separately and is not subtracted from anything")}


def reference_identity(*, checkpoint_sha256, config_sha256, scaffold_prefix, index,
                       convention=PROBABILITY_CONVENTION):
    """Everything that would change the cached numbers, in one comparable block."""
    return {"schema_version": REFERENCE_CACHE_SCHEMA,
            "checkpoint_sha256": checkpoint_sha256, "config_sha256": config_sha256,
            "scaffold_prefix_sha256": hashlib.sha256(scaffold_prefix.encode()).hexdigest(),
            "probability_convention": convention,
            "core_order_sha256": core_digest(index),
            "rows": int(np.asarray(index).shape[0])}


@torch.no_grad()
def score_sequences(policy, index, *, batch_size=256, progress_every=200):
    """``(N,)`` SUM log probability under the current weights, without gradients."""
    values = np.asarray(index)
    was_training = policy.model.training
    policy.model.eval()
    out = np.empty(values.shape[0], dtype=np.float64)
    for start in range(0, values.shape[0], batch_size):
        chunk = values[start:start + batch_size]
        out[start:start + len(chunk)] = (
            policy.sequence_log_probs(chunk).double().cpu().numpy())
        if progress_every and (start // batch_size) % progress_every == 0:
            print(f"  reference scored {min(start + len(chunk), values.shape[0])}/"
                  f"{values.shape[0]}", flush=True)
    if was_training:
        policy.model.train()
    return out


def build_reference_cache(policy, index, identity, *, clock, batch_size=256):
    """Score the union of eligible chosen and rejected cores exactly once.

    The cost is charged to the DPO arm's own budget, inside ``clock``, because it
    is work DPO needs and continued SFT does not. It is charged **once**: a rerun
    that reuses the saved cache still reports the original creation seconds as the
    cold-start cost, and reports the warm reuse separately.
    """
    with clock.segment() as segment:
        values = score_sequences(policy, index, batch_size=batch_size)
    require(bool(np.isfinite(values).all()), "Nonfinite reference log probability")
    require(bool((values <= 1e-6).all()), "Reference log probability above zero")
    bound = dict(identity, values_sha256=array_digest(values))
    array = np.array(values, dtype=np.float64)
    array.setflags(write=False)
    return ReferenceCache(identity=bound, values=array,
                          creation_gpu_seconds=float(segment.seconds))


def save_reference_cache(path, cache):
    """Write the cache next to its identity. An existing file is never overwritten."""
    path = Path(path)
    require(path.suffix == ".npy", "The reference cache array path must end in .npy")
    require(not path.is_file(), f"A reference cache already exists at {path}; it is immutable")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, cache.values)
    save_json(path.with_suffix(".json"), {"schema_version": REFERENCE_CACHE_SCHEMA,
                                          "identity": cache.identity,
                                          "creation_gpu_seconds": cache.creation_gpu_seconds})
    return path


def load_reference_cache(path, expected_identity):
    """Reload and re-verify. Any mismatch is a failure, never a silent rescore.

    The warm-reuse wall time is measured *around the work* -- the read, the digest
    and the identity check. Timing it in the caller's argument list, as this used
    to be, evaluates the subtraction before the load runs and always reports
    approximately zero.
    """
    started = time.perf_counter()
    path = Path(path)
    document = load_json(path.with_suffix(".json"))
    require(document["schema_version"] == REFERENCE_CACHE_SCHEMA,
            "Unsupported reference-cache schema")
    stored = document["identity"]
    values = np.load(path)
    require(array_digest(values) == stored["values_sha256"],
            "Reference cache contents do not match the digest recorded with them")
    expected = dict(expected_identity, values_sha256=stored.get("values_sha256"))
    require(stored == expected,
            f"Reference cache identity mismatch. Differing keys: "
            f"{sorted(k for k in set(stored) | set(expected) if stored.get(k) != expected.get(k))}")
    values.setflags(write=False)
    return ReferenceCache(identity=stored, values=values,
                          creation_gpu_seconds=float(document["creation_gpu_seconds"]),
                          reused=True,
                          warm_reuse_wall_seconds=float(time.perf_counter() - started))


def verify_reference_parity(policy, cache, index, rows, *, atol=SUM_LOG_PROBABILITY_ATOL,
                            rtol=SUM_LOG_PROBABILITY_RTOL, batch_size=64):
    """Re-score a few cached rows with the live reference weights and compare.

    Cheap, and it catches the failure that matters: a cache that survived its
    identity check but was produced by different weights, a different prefix or a
    different renormalization. The tolerance is the shared sum-log-probability one
    -- two fp32 routes over the same weights differ by fp32 rounding, and a cache
    from *different* weights differs by orders of magnitude more.
    """
    rows = np.asarray(rows)
    fresh = score_sequences(policy, np.asarray(index)[rows], batch_size=batch_size,
                            progress_every=0)
    report = compare_sum_log_probabilities(
        fresh, cache.values[rows], label="Reference cache disagrees with a fresh score",
        atol=atol, rtol=rtol)
    return dict(report, probe_rows=rows.tolist())


# ---------------------------------------------------------------------------
# objectives
# ---------------------------------------------------------------------------

def dpo_batch(policy, chosen_index, rejected_index, reference_chosen, reference_rejected, *, beta):
    """Mean DPO loss over one batch of pairs, plus the diagnostics worth recording.

    The policy term is a SUM over the ten core positions, matching the reference
    convention exactly. Both members of a pair go through the same shared-prefix
    forward path the reference used, so the ``beta * (pi - ref)`` differences are
    comparable quantities rather than two conventions subtracted from each other.
    """
    policy_chosen = policy.sequence_log_probs(chosen_index)
    policy_rejected = policy.sequence_log_probs(rejected_index)
    reference_chosen = reference_chosen.to(policy_chosen.dtype)
    reference_rejected = reference_rejected.to(policy_rejected.dtype)
    per_pair = dpo_per_pair_loss(policy_chosen, policy_rejected, reference_chosen,
                                 reference_rejected, beta=beta)
    with torch.no_grad():
        margin = ((policy_chosen - policy_rejected)
                  - (reference_chosen - reference_rejected)).double()
    return per_pair.mean(), {"pairs": int(policy_chosen.shape[0]),
                             "mean_margin": float(margin.mean()),
                             "accuracy": float((margin > 0).double().mean()),
                             "mean_policy_chosen": float(policy_chosen.detach().double().mean()),
                             "mean_policy_rejected": float(policy_rejected.detach().double().mean())}


@torch.no_grad()
def validation_pair_metrics(policy, pairs, *, batch_size=256, reference_chosen=None,
                            reference_rejected=None, beta=None):
    """Held-out preference metrics for one checkpoint, reported at every budget.

    ``pair_accuracy`` is the raw question -- does the policy give the high-bin
    member more probability than its distance-matched low-bin partner -- and needs
    no reference. The implicit-reward accuracy and mean margin do need one, and
    they are computed against the same frozen values DPO trained on, so a shifted
    reference cannot make the number look better.
    """
    chosen = score_sequences(policy, pairs["chosen_index"], batch_size=batch_size,
                             progress_every=0)
    rejected = score_sequences(policy, pairs["rejected_index"], batch_size=batch_size,
                               progress_every=0)
    document = {"pairs": int(chosen.size),
                "pair_accuracy": float((chosen > rejected).mean()),
                "pair_ties": int((chosen == rejected).sum()),
                "mean_chosen_sum_log_probability": float(chosen.mean()),
                "mean_rejected_sum_log_probability": float(rejected.mean()),
                "mean_log_probability_gap": float((chosen - rejected).mean()),
                "chosen_nll_per_residue": float(-chosen.mean() / CORE_LENGTH),
                "rejected_nll_per_residue": float(-rejected.mean() / CORE_LENGTH)}
    if reference_chosen is not None and reference_rejected is not None:
        margin = (chosen - rejected) - (np.asarray(reference_chosen)
                                        - np.asarray(reference_rejected))
        document["implicit_reward_accuracy"] = float((margin > 0).mean())
        document["mean_implicit_margin"] = float(margin.mean())
        if beta is not None:
            document["mean_dpo_loss"] = float(np.logaddexp(0.0, -float(beta) * margin).mean())
    return document


def continued_sft_batch(policy, chosen_index):
    """Mean per-residue NLL on chosen positives -- the same objective as initial SFT."""
    loss = policy.loss(chosen_index)
    return loss, {"sequences": int(np.asarray(chosen_index).shape[0]),
                  "nll_per_residue": float(loss.detach())}


# ---------------------------------------------------------------------------
# budgeted trajectory
# ---------------------------------------------------------------------------

def build_optimizer(model, optimization):
    return torch.optim.AdamW(model.parameters(), lr=optimization["learning_rate"],
                            betas=tuple(optimization["betas"]),
                            weight_decay=optimization["weight_decay"])


def run_budgeted_trajectory(*, step, budgets, clock, on_budget, optimizer, scheduler, model,
                            gradient_clip, progress=None, exposure_fields=(),
                            cumulative_exposures=None, max_updates=None):
    """Train one continuous trajectory, dropping a checkpoint as each budget passes.

    ``step(update_number)`` performs the forward pass and returns
    ``(loss, statistics)``; everything from the forward pass through
    ``optimizer.step()`` is charged to ``clock``. The budget is compared **after**
    an update completes, so the recorded elapsed time can exceed the target by at
    most one update -- and that bound is now *checked* per budget and recorded,
    rather than asserted in a docstring.

    Two honesty properties, both of which used to be claims rather than checks:

    * Work already charged to the clock before the first update (the DPO reference
      cache) must leave room for the smallest budget. If it does not, the first
      "180 s checkpoint" would be one update after an already-blown budget, and
      calling that a budgeted trajectory would be false. It raises instead.
    * ``on_budget(budget, record)`` runs outside the clock: saving a checkpoint and
      measuring validation are real costs, but they are not training. Their wall
      time is accumulated and reported as an explicitly excluded cost, next to the
      total trajectory wall time, instead of vanishing.

    ``cumulative_exposures()``, when supplied, returns the distinct-row counts seen
    so far; it is recorded at *every* budget, because a per-batch unique count says
    nothing about what a budget consumed.
    """
    remaining = sorted(float(b) for b in budgets)
    require(remaining, "A trajectory needs at least one budget")
    precharged = float(clock.elapsed_seconds)
    require(precharged < remaining[0],
            f"{precharged:.1f} GPU seconds were already charged to this trajectory (reference "
            f"cache creation) before the first update, which is at or beyond the smallest budget "
            f"of {remaining[0]:.1f} s. The budgeted trajectory cannot be reported honestly under "
            "these budgets; raise them or reduce the reference population rather than recording a "
            "checkpoint whose 'one update of overshoot' is a fiction.")
    started = time.perf_counter()
    history, reached, update = [], {}, 0
    totals = {name: 0 for name in exposure_fields}
    excluded_wall = 0.0
    while remaining:
        if max_updates is not None and update >= max_updates:
            break
        update += 1
        with clock.segment() as segment:
            optimizer.zero_grad(set_to_none=True)
            loss, statistics = step(update)
            require(bool(torch.isfinite(loss)), f"Nonfinite loss at update {update}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip,
                                                 error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
        for name in exposure_fields:
            totals[name] += int(statistics.get(name, 0))
        entry = {"update": update, "loss": float(loss.detach()),
                 "gradient_norm": float(norm),
                 "learning_rate": float(scheduler.get_last_lr()[0]),
                 "update_gpu_seconds": float(segment.seconds),
                 "cumulative_gpu_seconds": clock.elapsed_seconds, **statistics}
        history.append(entry)
        if progress is not None:
            progress.update(update, loss=entry["loss"],
                            gpu_seconds=entry["cumulative_gpu_seconds"])
        while remaining and clock.elapsed_seconds >= remaining[0]:
            budget = remaining.pop(0)
            overshoot = clock.elapsed_seconds - budget
            record = {"target_gpu_seconds": budget,
                      "actual_gpu_seconds": clock.elapsed_seconds,
                      "overshoot_seconds": overshoot,
                      "last_update_gpu_seconds": entry["update_gpu_seconds"],
                      "overshoot_within_one_update":
                          bool(overshoot <= entry["update_gpu_seconds"] + 1e-9),
                      "precharged_gpu_seconds": precharged,
                      "updates": update, "exposures": dict(totals),
                      "core_token_exposures": int(totals.get("sequences", 0)) * CORE_LENGTH,
                      "last_loss": entry["loss"],
                      "trajectory_wall_seconds": time.perf_counter() - started,
                      "excluded_evaluation_wall_seconds": excluded_wall}
            if cumulative_exposures is not None:
                record["distinct_exposures"] = dict(cumulative_exposures())
            if not record["overshoot_within_one_update"]:
                record["overshoot_note"] = (
                    "this budget and an earlier one were both passed inside the same update, so "
                    "the overshoot is bounded by that update only in aggregate")
            evaluation_started = time.perf_counter()
            record.update(on_budget(budget, record) or {})
            record["budget_evaluation_wall_seconds"] = time.perf_counter() - evaluation_started
            excluded_wall += record["budget_evaluation_wall_seconds"]
            reached[str(budget)] = record
    document = {"updates": update, "history": history, "budgets": reached,
                "exposures": dict(totals),
                "core_token_exposures": int(totals.get("sequences", 0)) * CORE_LENGTH,
                "gpu_seconds": clock.elapsed_seconds,
                "precharged_gpu_seconds": precharged,
                "trajectory_wall_seconds": time.perf_counter() - started,
                "excluded_evaluation_wall_seconds": excluded_wall,
                "budget_note": ("budgets are measured device-elapsed training seconds; they are "
                                "not FLOP or energy measurements, and no equal-update or "
                                "equal-epoch fairness is claimed. Wall time above includes the "
                                "excluded checkpoint/validation cost reported beside it.")}
    if cumulative_exposures is not None:
        document["distinct_exposures"] = dict(cumulative_exposures())
    return document


def cores_of(index):
    require(np.asarray(index).shape[1] == CORE_LENGTH, "Expected 10-residue cores")
    return decode_cores(index)
