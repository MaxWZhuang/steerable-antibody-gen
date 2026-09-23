"""The two-tier monitor: cheap sentinels that can only ask, full gates that decide.

The historical campaign ran the full 25,722-pair gate every 25 updates, which is
about 70% of its fit time. This flight keeps the same statistic, the same
population and the same threshold for the full check, and inserts a cheap
sentinel between full checks so the budget can cover 63,000 new updates:

* **sentinel** -- 512 preference pairs plus 1,024 parent draws, every 25 updates.
  It can *request* a full check and it can never stop a trajectory. Triggers are
  conservative and predeclared: chosen-side ``D >= .5``, any nonfinite score, a
  mean increase of ``>= .25`` over the previous sentinel, an inclusive tenfold
  point rate ``>= .02``, or ``>= 3`` inclusive hundredfold events.
* **full gate** -- the inherited ``her2_guard.LikelihoodGate`` on the whole fixed
  population, at update 1, every 100 updates, at every checkpoint and endpoint,
  at the final update, and whenever a sentinel asks.

No detection probability is claimed. A sentinel computed on 512 pairs has a
sampling distribution, and choosing its trigger from a historical distribution
would not make an interval on a future trajectory; the retrospective sensitivity
is quantified as a diagnostic and is not elevated to a guarantee. Monitoring
sensitivity between full checks is therefore *changed*, and that is disclosed
rather than described as equivalent to the old cadence.

Block B has no 25,722-pair population and must not point at the old validation
gate, which contains ``E``. Its gate is the chosen-side statistic over **all**
high ``C`` rows -- including rows in WT-distance cells with no eligible low
partner, which a pair-based gate would silently drop -- with the same summed-log-
probability convention, the same strict ``D > 1`` threshold and the same
nonfinite stop. Pair-based diagnostics on ``C`` are computed separately and
cannot change that population.
"""
from __future__ import annotations

import math

import numpy as np

from . import her2_guard as guard
from . import her2_support_scoring as scoring
from .her2_data import CORE_LENGTH
from .her2_nf_contract import LN10, LN100, NF_SCHEMA
from .her2_preferences import score_sequences
from .her2_runtime import require

#: Sentinel size and cadence.
SENTINEL_PAIRS = 512
SENTINEL_PARENT_DRAWS = 1024
SENTINEL_INTERVAL = 25
FULL_INTERVAL = 100

#: Predeclared, conservative sentinel triggers. Engineering choices.
SENTINEL_TRIGGERS = {
    "chosen_side_D_at_least": 0.5,
    "increase_since_previous_at_least": 0.25,
    "inclusive_tenfold_rate_at_least": 0.02,
    "inclusive_hundredfold_events_at_least": 3,
}

#: The inherited stop rule, unchanged.
STOP_THRESHOLD_NATS = 1.0


# ---------------------------------------------------------------------------
# cadence
# ---------------------------------------------------------------------------

class MonitorPlan:
    """Which check is due at a completed update, and why. A pure function of the update."""

    def __init__(self, *, endpoints=(), checkpoints=(), sentinel_interval=SENTINEL_INTERVAL,
                 full_interval=FULL_INTERVAL, first_update=1):
        require(int(sentinel_interval) > 0 and int(full_interval) > 0,
                "both intervals must be positive")
        require(int(full_interval) % int(sentinel_interval) == 0,
                "the full interval must be a multiple of the sentinel interval, so a full check "
                "always lands on a sentinel update and the two cadences cannot drift apart")
        self.sentinel_interval = int(sentinel_interval)
        self.full_interval = int(full_interval)
        self.first_update = int(first_update)
        self.endpoints = tuple(sorted(int(value) for value in endpoints))
        self.checkpoints = tuple(sorted(int(value) for value in checkpoints))

    def due(self, update, *, final=False, requested=False):
        """``(kind, reason)`` where kind is ``"full"``, ``"sentinel"`` or ``None``."""
        update = int(update)
        if update == self.first_update:
            return "full", "first_update"
        if update in self.endpoints:
            return "full", "exposure_endpoint"
        if update in self.checkpoints:
            return "full", "checkpoint"
        if final:
            return "full", "final_state"
        if requested:
            return "full", "sentinel_request"
        if update % self.full_interval == 0:
            return "full", "full_interval"
        if update % self.sentinel_interval == 0:
            return "sentinel", "sentinel_interval"
        return None, None

    def scheduled(self, total_updates):
        out = []
        for update in range(1, int(total_updates) + 1):
            kind, reason = self.due(update, final=(update == int(total_updates)))
            if kind is not None:
                out.append((update, kind, reason))
        return out

    def document(self):
        return {"sentinel_interval": self.sentinel_interval, "full_interval": self.full_interval,
                "first_update": self.first_update, "endpoints": list(self.endpoints),
                "checkpoints": list(self.checkpoints),
                "sentinel": {"pairs": SENTINEL_PAIRS, "parent_draws": SENTINEL_PARENT_DRAWS,
                             "triggers": dict(SENTINEL_TRIGGERS),
                             "authority": "may request a full check; may never stop a trajectory"},
                "full": {"authority": "the only check that can stop a trajectory",
                         "rule": "strict D > 1.0 nat/sequence, or any nonfinite score"},
                "amendment": ("this replaces the historical every-25 FULL gate. The full "
                              "statistic, population and threshold are unchanged; what changed is "
                              "how often the full statistic is computed."),
                "disclosure": ("monitoring sensitivity BETWEEN full checks is changed. No "
                               "detection probability is claimed and this is not equivalent to "
                               "the old cadence.")}


# ---------------------------------------------------------------------------
# the sentinel
# ---------------------------------------------------------------------------

class SentinelGate:
    """A cheap fixed probe that can only raise its hand.

    All three parts are fixed at construction: the same 512 preference PAIRS --
    both the chosen and the rejected half -- and the same 1,024 parent draws at
    every sentinel, so consecutive values are comparable and the "increase since
    the previous sentinel" trigger means something. Scoring only the chosen half
    would make this a 512-row probe of a 1,024-row population and would miss a
    policy that moved the rejected side.

    The parent reference arrays are required finite at construction. A nonfinite
    reference makes every later ``D`` a ``nan`` that compares false against every
    trigger, which is the one failure mode a monitor must not have.
    """

    def __init__(self, *, chosen_index, parent_chosen, parent_index, parent_log_probability,
                 rejected_index=None, parent_rejected=None, batch_size=256, triggers=None):
        self.chosen_index = np.asarray(chosen_index)
        self.parent_chosen = np.asarray(parent_chosen, dtype=np.float64)
        self.rejected_index = None if rejected_index is None else np.asarray(rejected_index)
        self.parent_rejected = (None if parent_rejected is None
                                else np.asarray(parent_rejected, dtype=np.float64))
        self.parent_index = np.asarray(parent_index)
        self.parent_log_probability = np.asarray(parent_log_probability, dtype=np.float64)
        require(self.chosen_index.shape[0] == self.parent_chosen.size,
                "one parent chosen score per sentinel pair")
        require(self.parent_index.shape[0] == self.parent_log_probability.size,
                "one parent score per sentinel draw")
        require((self.rejected_index is None) == (self.parent_rejected is None),
                "the rejected half needs both its rows and its parent scores")
        if self.rejected_index is not None:
            require(self.rejected_index.shape[0] == self.parent_rejected.size
                    == self.parent_chosen.size,
                    "the sentinel's rejected half must match its chosen half row for row")
        for name, values in (("parent_chosen", self.parent_chosen),
                             ("parent_rejected", self.parent_rejected),
                             ("parent_log_probability", self.parent_log_probability)):
            if values is None:
                continue
            require(bool(np.isfinite(values).all()),
                    f"the sentinel reference {name} carries a nonfinite value. Every later D "
                    "against it would be nan, and nan compares false against every trigger, so "
                    "the probe would report 'nothing to see' forever.")
        self.batch_size = int(batch_size)
        self.triggers = dict(triggers or SENTINEL_TRIGGERS)
        self.previous_D = None

    def evaluate(self, policy, *, update, reason):
        current = np.asarray(score_sequences(policy, self.chosen_index,
                                             batch_size=self.batch_size, progress_every=0),
                             dtype=np.float64)
        drawn = np.asarray(score_sequences(policy, self.parent_index,
                                           batch_size=self.batch_size, progress_every=0),
                           dtype=np.float64)
        rejected = None
        if self.rejected_index is not None:
            rejected = np.asarray(score_sequences(policy, self.rejected_index,
                                                  batch_size=self.batch_size, progress_every=0),
                                  dtype=np.float64)
        nonfinite_halves = sorted(
            name for name, values in (("chosen", current), ("rejected", rejected),
                                      ("parent_draws", drawn))
            if values is not None and not bool(np.isfinite(values).all()))
        finite = not nonfinite_halves
        record = {"schema_version": NF_SCHEMA, "record_kind": "sentinel_check",
                  "update": int(update), "reason": reason,
                  "pairs": int(self.chosen_index.shape[0]),
                  "rejected_rows": (0 if self.rejected_index is None
                                    else int(self.rejected_index.shape[0])),
                  "parent_draws": int(self.parent_index.shape[0]),
                  "nonfinite": not finite, "nonfinite_halves": nonfinite_halves,
                  "triggers": dict(self.triggers),
                  "authority": "request only; a sentinel never stops a trajectory"}
        if not finite:
            record.update(D=None, tenfold_rate=None, hundredfold_events=None,
                          request_full_check=True, trigger_reasons=["nonfinite_sentinel_score"],
                          consequence=("a nonfinite score on EITHER half requests the full "
                                       "authoritative gate; no trigger threshold is consulted "
                                       "because none of them is meaningful against nan."))
            return record
        value = float((self.parent_chosen - current).mean())
        drop = self.parent_log_probability - drawn
        tenfold = float((drop >= LN10).mean())
        hundredfold = int((drop >= LN100).sum())
        reasons = []
        if value >= float(self.triggers["chosen_side_D_at_least"]):
            reasons.append("chosen_side_D")
        if (self.previous_D is not None
                and value - self.previous_D >= float(
                    self.triggers["increase_since_previous_at_least"])):
            reasons.append("increase_since_previous")
        if tenfold >= float(self.triggers["inclusive_tenfold_rate_at_least"]):
            reasons.append("inclusive_tenfold_rate")
        if hundredfold >= int(self.triggers["inclusive_hundredfold_events_at_least"]):
            reasons.append("inclusive_hundredfold_events")
        record.update(D=value, D_per_residue=value / CORE_LENGTH,
                      previous_D=self.previous_D,
                      change_since_previous=None if self.previous_D is None
                      else value - self.previous_D,
                      tenfold_rate=tenfold, hundredfold_events=hundredfold,
                      max_drop=float(drop.max()),
                      request_full_check=bool(reasons), trigger_reasons=reasons,
                      tail_convention=">= (inclusive)")
        if rejected is not None:
            record.update(
                rejected_side_D=float((self.parent_rejected - rejected).mean()),
                pair_accuracy=float((current > rejected).mean()),
                implicit_margin_mean=float(((current - rejected)
                                            - (self.parent_chosen
                                               - self.parent_rejected)).mean()))
        self.previous_D = value
        return record

    def restore_previous_D(self, value):
        """Reinstate the last sentinel value after a resume.

        The "increase since the previous sentinel" trigger compares consecutive
        values, so a resumed trajectory that starts from ``None`` silently loses
        that trigger for one interval. The saved progress carries the value; this
        is what puts it back.
        """
        self.previous_D = None if value is None else float(value)
        return self.previous_D

    def document(self):
        return {"pairs": int(self.chosen_index.shape[0]),
                "rejected_rows": (0 if self.rejected_index is None
                                  else int(self.rejected_index.shape[0])),
                "parent_draws": int(self.parent_index.shape[0]),
                "triggers": dict(self.triggers),
                "population": ("both halves of the fixed preference pairs plus the fixed parent "
                               "draws"),
                "authority": "request only; a sentinel never stops a trajectory"}


def sentinel_sensitivity(per_pair_drop, *, pairs=SENTINEL_PAIRS, trigger, true_value,
                         draws=2000, seed):
    """Retrospective sensitivity of a sentinel trigger, as a diagnostic.

    Resamples an observed per-pair drop distribution to estimate how often a
    sentinel of this size would have asked for a full check at a given true mean.
    It describes a historical distribution, so it is evidence about this trigger's
    behaviour and not a bound on a future trajectory's detection probability.
    """
    values = np.asarray(per_pair_drop, dtype=np.float64)
    require(values.ndim == 1 and values.size > 0, "Expected a per-pair drop vector")
    centred = values - values.mean() + float(true_value)
    generator = np.random.default_rng(int(seed))
    fired = 0
    for _ in range(int(draws)):
        rows = generator.integers(0, centred.size, int(pairs))
        if centred[rows].mean() >= float(trigger):
            fired += 1
    return {"pairs": int(pairs), "trigger": float(trigger), "true_value": float(true_value),
            "draws": int(draws), "fire_rate": fired / float(draws), "seed": int(seed),
            "status": ("a retrospective resampling diagnostic on an observed distribution. It is "
                       "not a detection guarantee and is not a coverage statement about any "
                       "future trajectory.")}


# ---------------------------------------------------------------------------
# Block A: the inherited full gate, unchanged
# ---------------------------------------------------------------------------

class FullPairGate:
    """The historical fixed-validation-pair gate, delegated to unchanged."""

    def __init__(self, reference, *, threshold=STOP_THRESHOLD_NATS, batch_size=256):
        self.reference = reference
        self.gate = guard.LikelihoodGate(threshold_nats_per_sequence=float(threshold),
                                         core_positions=CORE_LENGTH)
        self.batch_size = int(batch_size)

    def evaluate(self, policy, pairs, *, update, reason):
        chosen = score_sequences(policy, pairs["chosen_index"], batch_size=self.batch_size,
                                 progress_every=0)
        rejected = score_sequences(policy, pairs["rejected_index"], batch_size=self.batch_size,
                                   progress_every=0)
        record = self.gate.evaluate(self.reference, chosen, rejected)
        record.update(record_kind="gate_verdict", kind="full", update=int(update), reason=reason,
                      population="fixed_validation_pairs",
                      block="A", scores={"chosen": np.asarray(chosen, dtype=np.float64),
                                         "rejected": np.asarray(rejected, dtype=np.float64)})
        return record

    def document(self):
        return dict(self.gate.document(), block="A", delegated_to="her2_guard.LikelihoodGate")


# ---------------------------------------------------------------------------
# Block B: the chosen-side gate over ALL high C rows
# ---------------------------------------------------------------------------

class HighRowGate:
    """Chosen-side ``D`` over every high row of ``C``. No pairability filter.

    Building C-internal preference pairs and gating on those would silently drop
    every high row whose WT distance has no eligible low partner in ``C``, which
    is a different and smaller population than the one the specification names.
    The pairs are still built -- as an independent diagnostic -- but they cannot
    change what this gate measures.
    """

    def __init__(self, *, row_index, parent_log_probability, threshold=STOP_THRESHOLD_NATS,
                 batch_size=256, label="all high C rows"):
        self.row_index = np.asarray(row_index)
        self.parent_log_probability = np.asarray(parent_log_probability, dtype=np.float64)
        require(self.row_index.shape[0] == self.parent_log_probability.size,
                "one parent score per gated row")
        require(self.row_index.shape[0] > 0, "the challenge gate needs at least one high row")
        require(bool(np.isfinite(self.parent_log_probability).all()),
                "the parent produced a nonfinite score on the gated population, so the gate would "
                "have no reference and every later D would be nan rather than large")
        self.threshold = float(threshold)
        self.batch_size = int(batch_size)
        self.label = str(label)

    def evaluate(self, policy, *, update, reason):
        scored = score_sequences(policy, self.row_index, batch_size=self.batch_size,
                                 progress_every=0)
        current = np.asarray(scored, dtype=np.float64)
        nonfinite = int((~np.isfinite(current)).sum())
        record = {"schema_version": NF_SCHEMA, "record_kind": "gate_verdict", "kind": "full",
                  "block": "B", "update": int(update), "reason": reason,
                  "population": self.label, "rows": int(current.size),
                  "nonfinite_scores": nonfinite,
                  "threshold_nats_per_sequence": self.threshold,
                  "threshold_nats_per_residue": self.threshold / CORE_LENGTH,
                  "statistic": ("D = mean(parent_chosen - current_chosen) over ALL high rows of "
                                "the calibration panel, in nats per sequence"),
                  "rule": "strict D > threshold stops; exact equality passes; nonfinite stops",
                  "pairability_note": ("every high row is included, including rows in WT-distance "
                                       "cells with no eligible low partner. Pair diagnostics are "
                                       "computed separately and cannot change this population."),
                  "scores": {"chosen": current}}
        if nonfinite:
            record.update(D=None, D_per_residue=None, passed=False,
                          stop_reason="nonfinite_validation_score")
            return record
        drop = self.parent_log_probability - current
        value = float(drop.mean())
        record.update(D=value, D_per_residue=value / CORE_LENGTH,
                      mean_current_chosen_nll_per_residue=float(-current.mean() / CORE_LENGTH),
                      chosen_drop=guard.quantile_summary(drop),
                      passed=bool(value <= self.threshold))
        if not record["passed"]:
            record["stop_reason"] = "parent_relative_likelihood_breach"
        return record

    def document(self):
        return {"block": "B", "population": self.label, "rows": int(self.row_index.shape[0]),
                "threshold_nats_per_sequence": self.threshold,
                "rule": "strict D > threshold, or any nonfinite score",
                "precision_note": ("this population is far smaller than Block A's 25,722 pairs, so "
                                   "the monitoring precision differs. The population size is "
                                   "reported with every verdict.")}


def pair_diagnostics(policy, pairs, *, parent_chosen, parent_rejected, batch_size=256):
    """Optional C-only pair statistics. Diagnostic; never the Block-B gate population."""
    chosen = np.asarray(score_sequences(policy, pairs["chosen_index"], batch_size=int(batch_size),
                                        progress_every=0), dtype=np.float64)
    rejected = np.asarray(score_sequences(policy, pairs["rejected_index"],
                                          batch_size=int(batch_size), progress_every=0),
                          dtype=np.float64)
    reference_chosen = np.asarray(parent_chosen, dtype=np.float64)
    reference_rejected = np.asarray(parent_rejected, dtype=np.float64)
    margin = (chosen - rejected) - (reference_chosen - reference_rejected)
    return {"record_kind": "challenge_pair_diagnostics", "pairs": int(chosen.size),
            "pair_accuracy": float((chosen > rejected).mean()),
            "mean_implicit_margin": float(margin.mean()),
            "chosen_side_D_on_pairs": float((reference_chosen - chosen).mean()),
            "status": ("a diagnostic on the pairable subset only. The Block-B gate runs on ALL "
                       "high rows and is not affected by this subset.")}


def monitor_population_report(*, high_rows, pairable_rows, excluded_rows, block):
    """What the gate population is, what the pair subset is, and what the gap costs."""
    return {"block": str(block), "high_rows": int(high_rows),
            "pairable_rows": int(pairable_rows), "excluded_from_pairs": int(excluded_rows),
            "gate_population": "all high rows",
            "precision_change": ("a smaller gate population has a wider sampling distribution for "
                                 "D than the historical 25,722-pair one. The change is reported, "
                                 "not corrected for, and no equivalence to the old precision is "
                                 "claimed."),
            "standard_error_scale": (math.sqrt(25722.0 / high_rows) if high_rows else None)}


def preservation_block(policy, bank, *, parent_log_probability, teacher_probabilities=None,
                       teacher_log_probabilities=None, batch_size=256, conditional_batch=64,
                       label="development preservation bank"):
    """Forward KL, inclusive tails and (optionally) conditional KL on a parent bank.

    A nonfinite score is a recognized observation about the policy and is recorded
    as one. Any other failure -- a shape mismatch, a wrong bank, a programming
    error -- is not an observation and is re-raised, because a monitor that
    reports "unavailable" for every trajectory would let a campaign call itself
    complete on nothing.
    """
    from .her2_nf_metrics import drop_block
    try:
        scored = scoring.strict_sequence_log_probabilities(
            policy, np.asarray(bank), batch_size=int(batch_size), label=label)
    except ValueError as error:
        text = str(error).lower()
        require("nonfinite" in text,
                f"{label} failed with an error that is not a recognized nonfinite-score "
                f"observation: {type(error).__name__}: {error}. A schema, ordering or programming "
                "failure stops the run instead of being recorded as a missing diagnostic.")
        return {"available": False, "reason": f"{type(error).__name__}: {error}",
                "recognized_as": "nonfinite score on the preservation bank"}
    block = {"available": True, "rows": int(np.asarray(bank).shape[0]),
             "drop": drop_block(parent_log_probability, scored["sum_log_probability"]),
             "policy_log_probability": scoring.log_probability_block(
                 scored["sum_log_probability"], label=label),
             "logit_checks": scored["checks"]}
    if teacher_probabilities is not None:
        import torch
        from . import her2_replay as replay_lib
        was_training = bool(policy.model.training)
        policy.model.eval()
        totals = []
        try:
            with torch.inference_mode():
                for start in range(0, int(np.asarray(bank).shape[0]), int(conditional_batch)):
                    rows = slice(start, start + int(conditional_batch))
                    student = replay_lib.student_log_probabilities(policy, np.asarray(bank)[rows])
                    totals.append(replay_lib.sequence_conditional_kl(
                        teacher_probabilities[rows].to(student.device),
                        teacher_log_probabilities[rows].to(student.device),
                        student).double().cpu().numpy())
        finally:
            if was_training:
                policy.model.train()
        conditional = np.concatenate(totals)
        block["conditional_kl"] = {
            "mean": float(conditional.mean()),
            "standard_error": float(conditional.std(ddof=1) / math.sqrt(conditional.size)),
            "rows": int(conditional.size),
            "difference_from_log_ratio_mean": float(
                conditional.mean() - block["drop"]["mean_log_ratio"]),
            "note": ("the summed per-position conditional KL and the mean sequence log ratio are "
                     "two estimators of KL(P||Q) with the same population mean and different "
                     "distributions. Both are saved with their difference; neither is adjusted to "
                     "match the other.")}
    return block
