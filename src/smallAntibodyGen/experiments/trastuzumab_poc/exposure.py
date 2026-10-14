"""Outcome-free accounting of the exact configured post-training draw schedule.

This is a schedule inspection, not a training run or a recommendation to change
the horizon. It reads endpoint identity, assay and role only. Pair-label exposure
means a value will be compared; whether that comparison is tied, informative, or
has an active ranking penalty cannot be known without outcomes and model scores.
"""
from __future__ import annotations

import math

from smallAntibodyGen.experiments.trastuzumab_poc.artifacts import (
    positive_count, seed_value, text_value,
)
from smallAntibodyGen.experiments.trastuzumab_poc.randomness import RNG_ALGORITHM, keyed_random


def _summary(counts, eligible_ids):
    """Include zero counts so an unused available measurement stays visible."""
    counts = {endpoint_id: counts[endpoint_id] for endpoint_id in sorted(counts)}
    seen = [endpoint_id for endpoint_id, count in counts.items() if count]
    unseen = [endpoint_id for endpoint_id, count in counts.items() if not count]
    eligible_counts = [counts[endpoint_id] for endpoint_id in eligible_ids]
    return {
        "per_endpoint_counts": counts,
        "seen_endpoint_ids": seen,
        "unseen_endpoint_ids": unseen,
        "seen_endpoint_count": len(seen),
        "unseen_endpoint_count": len(unseen),
        "fraction_of_selected": len(seen) / len(counts),
        "total_endpoint_exposures": sum(counts.values()),
        "minimum_count_all_selected": min(counts.values()),
        "maximum_count_all_selected": max(counts.values()),
        "eligible_endpoint_count": len(eligible_ids),
        "fraction_of_eligible": len(seen) / len(eligible_ids) if eligible_ids else None,
        "minimum_count_eligible": min(eligible_counts) if eligible_counts else None,
    }


def plan_post_exposure(selected_entries, config):
    """Enumerate the intended schedule without reading measurements or models.

    ``selected_entries`` must preserve the exact selected-budget order used by
    ``RestrictedPostData.selected_entries``. Reconstruction choices depend on that
    order; pair choices instead use sorted assay groups and sorted endpoint IDs,
    matching ``RestrictedPostData.draw_pair``. ``config`` supplies ``stage``,
    ``seed``, ``total_updates``, ``examples_per_update`` and ``ordering_weight``.

    The report covers the complete declared horizon, not just a stop/resume
    segment. Time is linear in training units plus sorting the selected IDs;
    memory is linear in selected endpoints. No draw history is retained, global
    RNG is unchanged, and neither the sampler nor the horizon is altered.
    """
    if config.stage != "post":
        raise ValueError("post exposure planning requires stage='post'")
    seed = seed_value(config.seed)
    updates = positive_count(config.total_updates, "total_updates")
    examples = positive_count(config.examples_per_update, "examples_per_update")
    weight = config.ordering_weight
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
        raise ValueError("ordering_weight must be a finite nonnegative number")

    entries = tuple(selected_entries)
    if not entries:
        raise ValueError("post exposure planning requires selected endpoints")
    ids = []
    by_assay = {}
    for entry in entries:
        endpoint_id = text_value(entry.endpoint_id, "endpoint_id")
        assay_id = text_value(entry.assay_id, "assay_id")
        if entry.role != "post_pool":
            raise ValueError("only selected post_pool endpoints may be planned")
        ids.append(endpoint_id)
        by_assay.setdefault(assay_id, []).append(endpoint_id)
    if len(set(ids)) != len(ids):
        raise ValueError("selected endpoint IDs must be unique")

    # Keep the same ordering and RNG calls as RestrictedPostData.draw_pair:
    # rng.sample(rng.choice(groups), 2). Selection never inspects label values.
    groups = tuple(tuple(sorted(group)) for _, group in sorted(by_assay.items()) if len(group) >= 2)
    singletons = sorted(group[0] for group in by_assay.values() if len(group) == 1)
    eligible = sorted(endpoint_id for group in groups for endpoint_id in group)
    ordering_enabled = weight > 0
    if ordering_enabled and not groups:
        raise ValueError("ordering needs at least two selected endpoints in the same assay")

    reconstruction_counts = dict.fromkeys(ids, 0)
    pair_counts = dict.fromkeys(ids, 0)
    for update in range(updates):
        for example in range(examples):
            chosen = keyed_random(seed, update, example, "endpoint").choice(ids)
            reconstruction_counts[chosen] += 1
            if ordering_enabled:
                rng = keyed_random(seed, update, example, "pair")
                left, right = rng.sample(rng.choice(groups), 2)
                pair_counts[left] += 1
                pair_counts[right] += 1

    units = updates * examples
    return {
        "schema": "trastuzumab-post-exposure/1",
        "rng": RNG_ALGORITHM,
        "seed": seed,
        "total_updates": updates,
        "examples_per_update": examples,
        "training_units": units,
        "selected_endpoint_count": len(ids),
        "selected_endpoint_ids_in_schedule_order": ids,
        "ordering_enabled": ordering_enabled,
        "pair_draws": units if ordering_enabled else 0,
        "pair_eligible_endpoint_ids": eligible,
        "pair_ineligible_singleton_ids": singletons,
        "reconstruction": _summary(reconstruction_counts, sorted(ids)),
        "pair_labels": _summary(pair_counts, eligible),
        "scope": "complete declared horizon; all updates assumed successful",
        "measurement_limit": "pair exposure counts comparisons, not informative labels or nonzero gradients; outcomes are not read",
    }
