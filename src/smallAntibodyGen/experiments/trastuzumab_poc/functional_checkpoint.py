"""Identity, progress and data-free validation of the functional arm inside training records (M2).

The trainer, the checkpoint reader and the lineage guard all read the same plain-JSON
records, so a counter, a budget or a head description has ONE definition here:

* ``FunctionalRun`` binds the three inputs of the arm (config, qualified document,
  restricted measurements) to each other BEFORE any update; a directly constructed
  ``RestrictedFunctionalData`` is re-verified against the qualified document.
* ``functional_input_identity`` / ``functional_identity`` build ``input.functional`` (the
  canonical outcome-free budget and availability counts) and ``identity.functional`` (config,
  head architecture and initialization, digests, head-step rule, discard-on-parent semantics).
  No measured value is part of either.
* ``functional_progress`` serializes the counters and the exact sorted unique observation IDs.
* ``validate_functional_training`` replays the outcome-free schedule to the exact
  opportunities and compared IDs and bounds the outcome counters.  Given the restricted
  measurements (resume, save) it replays the labels as well and pins the outcome counters and
  ``ordered_updates`` exactly; ``active_pairs`` depends on the model, so it is only bounded.

This module owns no optimizer: ``checkpoint.py`` validates the AdamW states, using
``ordered_updates`` from here as the exact head step count.  It never imports the runner.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from fractions import Fraction
from typing import Any, Dict, Optional

from smallAntibodyGen.experiments.trastuzumab_poc.functional_data import (
    QualifiedFunctional, exact_keys, sha256_hex,
)
from smallAntibodyGen.experiments.trastuzumab_poc.functional_protocol import (
    PAIR_PURPOSE, RNG_ALGORITHM, FunctionalBudget, RestrictedFunctionalData,
    check_counter_invariants, check_seen_within_budget, coverage_requirement_met,
    functional_budget_identity, functional_exposure_plan, replay_schedule,
    validate_functional_budget_identity,
)
from smallAntibodyGen.experiments.trastuzumab_poc.functional_train import (
    HEAD_INIT_PURPOSE, FunctionalConfig,
)
from smallAntibodyGen.experiments.trastuzumab_poc.randomness import keyed_seed

FUNCTIONAL_SCHEMA = "trastuzumab-functional-training/1"
INPUT_SCHEMA = "trastuzumab-functional-input/1"
HEAD_KIND = "linear-no-bias-fp32/1"
# Purpose key of the isolated Torch stream around the functional forwards and backwards.
FUNCTIONAL_PURPOSE = "functional-model"
COUNTER_KEYS = ("pair_opportunities", "ordered_pairs", "tie_pairs", "unresolved_pairs",
                "active_pairs", "ordered_updates", "feature_forwards")
SEEN_KEYS = ("compared", "ordered")
HEAD_STEP_RULE = ("one AdamW step per global update that backpropagated at least one ordered pair, "
                  "including updates whose hinges are all inactive; learning rate = training "
                  "schedule at the GLOBAL update times head_lr_scale, assigned on every update; "
                  "head step count == ordered_updates; separate gradient clipping")
PARENT_BEHAVIOR = ("discarded at parent export: head weights and both optimizer states are never "
                   "inherited; downstream starts a fresh adapter and optimizer; this record is "
                   "kept as supervision provenance")
_AVAILABLE_KEYS = ("qualified_train_observations", "qualified_train_physical_constructs",
                   "selected_observations", "selected_construct_ids", "selected_physical_constructs")


@dataclass(frozen=True, eq=False)
class FunctionalRun:
    """The inputs of one functional arm, bound to each other before the first update."""
    config: FunctionalConfig
    qualified: QualifiedFunctional
    data: RestrictedFunctionalData

    def __post_init__(self) -> None:
        for name, kind in (("config", FunctionalConfig), ("qualified", QualifiedFunctional),
                           ("data", RestrictedFunctionalData)):
            if not isinstance(getattr(self, name), kind):
                raise TypeError(f"{name} must be {kind.__name__}")
        # Objects built directly through the API are not trusted: budget AND directions.
        self.data.verify_against(self.qualified)


def planned_functional_exposure(run: FunctionalRun, *, seed: int, total_updates: int) -> Dict[str, Any]:
    """The outcome-free exposure plan for the declared horizon; the coverage requirement is checked."""
    return check_functional_exposure(functional_exposure_plan(
        run.qualified, run.data.budget, seed=seed, pairs_per_update=run.config.pairs_per_update,
        total_updates=total_updates), run.config)


def functional_coverage(plan) -> Fraction:
    fraction = plan["schedule"]["compared_fraction"]
    return Fraction(fraction["numerator"], fraction["denominator"])


def functional_coverage_met(plan, config: FunctionalConfig) -> bool:
    return coverage_requirement_met(functional_coverage(plan), config.minimum_compared_fraction)


def check_functional_exposure(plan, config: FunctionalConfig):
    if not functional_coverage_met(plan, config):
        raise ValueError(
            f"declared horizon compares only {float(functional_coverage(plan)):.1%} of the selected "
            f"functional observations; required {config.minimum_compared_fraction:.1%}. Inspect the "
            "outcome-free plan and preregister a longer horizon or an explicit coverage threshold.")
    return plan


# ---------------------------------------------------------------------------
# Identity records
# ---------------------------------------------------------------------------

def head_identity(hidden_size: int, seed: int) -> Dict[str, Any]:
    if type(hidden_size) is not int or hidden_size < 1:
        raise ValueError("head hidden_size must be a positive integer")
    return {"kind": HEAD_KIND, "hidden_size": hidden_size, "parameter_names": ["weight"],
            "initialization": {"purpose": HEAD_INIT_PURPOSE, "seed": seed,
                               "keyed_seed": keyed_seed(seed, 0, 0, HEAD_INIT_PURPOSE),
                               "distribution": "normal(0, hidden_size ** -0.5)",
                               "rng": "isolated-cpu-then-device"},
            "optimizer_step_rule": HEAD_STEP_RULE, "parent_behavior": PARENT_BEHAVIOR}


def _features() -> Dict[str, str]:
    return {"pooling": "mean-fused-hidden-over-antibody-slots/1",
            "context": "fresh-complete-view-provider/1",
            "dropout": "disabled-through-backward-with-checkpointing/1"}


def functional_input_identity(run: FunctionalRun) -> Dict[str, Any]:
    """``input.functional``: the canonical outcome-free budget and what was available."""
    budget, qualified = run.data.budget, run.qualified
    selected = {construct for _, construct in budget.selected}
    return {"schema": INPUT_SCHEMA, "budget": functional_budget_identity(budget),
            "available": {
                "qualified_train_observations": len(qualified.train_observations()),
                "qualified_train_physical_constructs":
                    len({c.content_key for c in qualified.train_constructs()}),
                "selected_observations": len(budget.selected),
                "selected_construct_ids": len(selected),
                "selected_physical_constructs":
                    len({qualified.construct(c).content_key for c in selected})}}


def functional_identity(run: FunctionalRun, *, hidden_size: int, seed: int) -> Dict[str, Any]:
    """``identity.functional``; digests only, never a measured value."""
    return {"schema": FUNCTIONAL_SCHEMA, "config": asdict(run.config),
            "head": head_identity(hidden_size, seed),
            "qualified_digest": run.qualified.digest, "contract_digest": run.qualified.contract_digest,
            "selection_digest": run.data.budget.selection_digest,
            "measurement_digest": run.data.data_digest(),
            "pair_schedule": {"rng": RNG_ALGORITHM, "purpose": PAIR_PURPOSE, "seed": seed,
                              "pairs_per_update": run.config.pairs_per_update},
            "weighting": "weight / pairs_per_update", "features": _features()}


# ---------------------------------------------------------------------------
# Progress record
# ---------------------------------------------------------------------------

def new_counters() -> Dict[str, int]:
    return dict.fromkeys(COUNTER_KEYS, 0)


def new_seen():
    return {key: set() for key in SEEN_KEYS}


def functional_progress(counters, seen) -> Dict[str, Any]:
    """Counters plus exact sorted unique ``[group_id, construct_id]`` lists."""
    return {"counters": dict(counters),
            "seen": {key: [list(item) for item in sorted(seen[key])] for key in SEEN_KEYS}}


def _identifier_pairs(values, where):
    if (not isinstance(values, list) or any(
            not isinstance(item, list) or len(item) != 2
            or any(not isinstance(part, str) or not part for part in item) for item in values)):
        raise ValueError(f"{where} must be a list of [group_id, construct_id] pairs")
    pairs = [tuple(item) for item in values]
    if pairs != sorted(set(pairs)):
        raise ValueError(f"{where} must be sorted and unique")
    return pairs


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validated_budget(identity, block) -> FunctionalBudget:
    supplied = identity["input"]["functional"]
    exact_keys(supplied, ("schema", "budget", "available"), "functional input")
    if supplied["schema"] != INPUT_SCHEMA:
        raise ValueError("unsupported functional input schema")
    budget = validate_functional_budget_identity(supplied["budget"])
    if functional_budget_identity(budget) != supplied["budget"]:
        raise ValueError("functional budget identity is not canonical")
    for name in ("qualified_digest", "contract_digest", "selection_digest"):
        if block[name] != getattr(budget, name):
            raise ValueError(f"functional identity {name} differs from the recorded budget")
    available = exact_keys(supplied["available"], _AVAILABLE_KEYS, "functional availability")
    if any(type(available[key]) is not int or available[key] < 1 for key in _AVAILABLE_KEYS):
        raise ValueError("functional availability counts must be positive integers")
    if (available["selected_observations"] != budget.budget_n
            or available["selected_construct_ids"] != len({c for _, c in budget.selected})
            or available["selected_physical_constructs"] > available["selected_construct_ids"]
            or available["selected_observations"] > available["qualified_train_observations"]
            or available["selected_physical_constructs"] > available["qualified_train_physical_constructs"]):
        raise ValueError("inconsistent functional availability counts")
    return budget


def validate_functional_training(identity, progress, *,
                                 restricted: Optional[RestrictedFunctionalData] = None):
    """Validate the functional parts of one training identity and progress record.

    Returns ``None`` when the run has no functional arm (and every functional record is
    explicitly null), otherwise ``{"config", "hidden_size", "budget", "ordered_updates"}``.
    Without ``restricted`` this is data-free; with it, the labels must hash to the recorded
    ``measurement_digest`` and then pin the outcome counters.
    """
    block, recorded = identity["functional"], progress["functional"]
    if block is None:
        if identity["input"]["functional"] is not None or recorded is not None:
            raise ValueError("functional records without a functional identity")
        return None
    if identity["config"]["stage"] != "mid":
        raise ValueError("functional supervision is a midtraining-only arm")
    if identity["input"]["functional"] is None or recorded is None:
        raise ValueError("a functional identity needs its input and progress records")
    exact_keys(block, ("schema", "config", "head", "qualified_digest", "contract_digest",
                       "selection_digest", "measurement_digest", "pair_schedule", "weighting",
                       "features"), "functional identity")
    if block["schema"] != FUNCTIONAL_SCHEMA:
        raise ValueError("unsupported functional identity schema")
    seed, updates = identity["config"]["seed"], progress["next_update"]
    exact_keys(block["config"], [f.name for f in fields(FunctionalConfig)], "functional config")
    config = FunctionalConfig(**block["config"])
    if asdict(config) != block["config"]:
        raise ValueError("functional config is not canonical")
    head = block["head"]
    if not isinstance(head, dict) or type(head.get("hidden_size")) is not int:
        raise ValueError("functional head identity needs an integer hidden_size")
    if head != head_identity(head["hidden_size"], seed):
        raise ValueError("functional head identity differs from its canonical record")
    sha256_hex(block["measurement_digest"], "functional measurement_digest")
    if (block["pair_schedule"] != {"rng": RNG_ALGORITHM, "purpose": PAIR_PURPOSE, "seed": seed,
                                   "pairs_per_update": config.pairs_per_update}
            or block["weighting"] != "weight / pairs_per_update" or block["features"] != _features()):
        raise ValueError("functional schedule, weighting or feature record differs from the recorded config")
    budget = _validated_budget(identity, block)

    exact_keys(recorded, ("counters", "seen"), "functional progress")
    counters, seen = recorded["counters"], recorded["seen"]
    exact_keys(counters, COUNTER_KEYS, "functional counters")
    exact_keys(seen, SEEN_KEYS, "functional seen")
    if any(type(counters[key]) is not int or counters[key] < 0 for key in COUNTER_KEYS):
        raise ValueError("functional counters must be nonnegative integers")
    observed = [pair for key in SEEN_KEYS for pair in _identifier_pairs(seen[key], f"seen.{key}")]
    check_seen_within_budget(observed, budget)
    if restricted is not None:
        if restricted.budget != budget:
            raise ValueError("restricted measurements belong to a different functional budget")
        # The same budget and order are not enough: the labels that replay the counters must be
        # the exact labels (and directions) the identity recorded. The digest binds them
        # BEFORE the classification replay below.
        if restricted.data_digest() != block["measurement_digest"]:
            raise ValueError("restricted measurements differ from the recorded measurement digest")
    pairs = config.pairs_per_update
    replay = (restricted.replay_outcomes(seed, pairs, updates) if restricted is not None
              else replay_schedule(budget, seed, pairs, updates))
    check_counter_invariants(counters, seen, replay, pairs, updates)
    if counters["feature_forwards"] != 2 * counters["ordered_pairs"]:
        raise ValueError("functional feature forwards must be two per ordered pair")
    return {"config": config, "hidden_size": head["hidden_size"], "budget": budget,
            "ordered_updates": counters["ordered_updates"]}
