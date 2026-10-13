"""Diagnostics for the pilot: labelled, bounded, and kept apart.

Run once before training and once after, on the same model object, in evaluation mode, under
``no_grad`` and a forked torch generator; every module's incoming mode is restored in ``finally``.
Nothing here draws from a global generator: every choice comes from a keyed ``numpy`` generator, so
the baseline and the final evaluation look at the same rows, the same masks and the same pairs.

Four blocks, never merged, because they answer different questions:

* ``development_ranking`` -- generalisation to the frozen DEVELOPMENT panel (the cores no training
  row or neighbour touched), as two labelled scores of the same fixed-path windows:
  ``high_only`` (mean log probability under ``bin=high``) and ``high_minus_null`` (that minus the
  same window under ``bin=null``), each with its Spearman against the qualified pKD. An undefined
  Spearman is ``null`` with a reason -- at the start the condition tables are zero-initialised, so
  the high-minus-null ratio is exactly constant and has no ranking to report. Only development
  windows are scored; no reserved or test label is read, and no per-row quality bin is used.
* ``completion_by_mask_count`` -- the masked-window negative log likelihood of the same development
  windows when ``k`` of the ten HER2 sites are hidden (suffix or a keyed random subset, ``k = 10``
  being the full window), under ``bin=null`` and ``bin=high``. This is the capability the mask arms
  train; it is not an affinity ranking.
* ``order_sensitivity`` -- the same full window scored with the sites revealed forwards and in
  reverse. A masked model that is not an exactly consistent joint gives different numbers; the gap
  and the agreement of the two orderings are reported with their support.
* ``training_fit_by_distance`` -- how well the model's sequentially revealed gap matches
  ``kappa * measured gap`` on a fixed sample of TRAINING pairs at each distance. This is a fit
  measure on the training population and is never a generalisation claim; for an arm that does not
  train a distance it measures transfer to that distance. It reports the arm's CONFIGURED
  replacement distances and, separately, the REALIZED replacement counts of the loop so far (empty
  before the first update) -- a configured weight is never evidence that a distance trained.

Held-out local-gap diagnostics are deferred: they need pairs among development cores scored through
a qualification of their own, and the pilot invents no number in their place.
"""
from __future__ import annotations

import math
from contextlib import contextmanager

import numpy as np
import torch

from .. import her2_routes_eval as eval_lib
from ..her2_routes_conditions import BIN_HIGH, BIN_NULL, Condition
from ..her2_routes_serial import ablang_serialization, her2_construct, her2_core_edit_map
from ..her2_runtime import require
from . import objective as objective_lib

#: Keys of the evaluation generators. Disjoint from the training keys in ``draws``.
KEY_SUBSET, KEY_FIT_PAIRS, KEY_FIT_ORDER = 91, 92, 93
GEOMETRIES = ("suffix", "subset")


@contextmanager
def isolated(services, *roots):
    """Eval mode, no gradient, forked generators; incoming modes restored whatever happens."""
    snapshot = [(module, module.training) for root in roots if root is not None
                for module in root.modules()]
    devices = ([torch.cuda.current_device()]
               if str(services.device).startswith("cuda") and torch.cuda.is_available() else [])
    try:
        for root in roots:
            if root is not None:
                root.eval()
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            yield
    finally:
        for module, flag in snapshot:
            module.training = flag


class _NoDropout:
    """A generator for the condition dropout that never drops, with real permutations."""

    def __init__(self, generator):
        self._generator = generator

    def random(self):
        return 1.0 - 2.0 ** -53

    def permutation(self, size):
        return self._generator.permutation(size)


def _finite(value, name):
    number = float(value)
    require(math.isfinite(number), f"{name} is not finite ({value!r}); training diverged or the "
                                   "evaluation is broken, and a non-finite number is never written")
    return number


def spearman_record(scores, outcomes):
    """``{"value", "reason", "rows"}``: Spearman's rho, or ``None`` with the reason it is undefined."""
    scores, outcomes = np.asarray(scores, dtype=np.float64), np.asarray(outcomes, dtype=np.float64)
    rows = int(scores.size)
    value = eval_lib.spearman(scores, outcomes)
    if value is not None and math.isfinite(value):
        return {"value": float(value), "reason": None, "rows": rows}
    if rows < 2:
        reason = "fewer than two rows"
    elif float(np.ptp(scores)) == 0.0:
        reason = "the scores are constant, so they rank nothing"
    elif float(np.ptp(outcomes)) == 0.0:
        reason = "the outcomes are constant"
    else:
        reason = "undefined"
    return {"value": None, "reason": reason, "rows": rows}


def _summary(values, name):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": _finite(values.mean(), f"mean {name}"), "min": _finite(values.min(), name),
            "max": _finite(values.max(), name), "rows": int(values.size)}


def development_ranking(services, route, model, heads, panel, *, batch_size):
    """High-only and high-minus-null scores of the development windows, each with its Spearman."""
    terms = services.hr17_selection_terms(route, model, heads, panel, batch_size=batch_size)
    high = np.asarray(terms["high"], dtype=np.float64)
    null = np.asarray(terms["null"], dtype=np.float64)
    require(high.shape == null.shape == (len(panel),),
            f"{high.shape} high and {null.shape} null scores for {len(panel)} panel rows")
    outcomes = [float(row["value"]) for row in panel]
    ratio = high - null
    return {
        "rows": len(panel), "divisor": int(terms["divisor"]), "assay": terms["assay"],
        "high_only": {"label": "mean fixed-path log probability of the window per site, bin=high",
                      "scores": _summary(high, "high"),
                      "spearman_vs_pkd": spearman_record(high, outcomes)},
        "high_minus_null": {
            "label": "the same, minus the same window under bin=null (the HR17 training selector's "
                     "ratio)",
            "scores": _summary(ratio, "high minus null"),
            "spearman_vs_pkd": spearman_record(ratio, outcomes)},
        "null_only": {"label": "diagnostic: bin=null alone",
                      "scores": _summary(null, "null"),
                      "spearman_vs_pkd": spearman_record(null, outcomes)},
        "note": ("development windows only, with their exact qualified pKD. A constant "
                 "high-minus-null (zero-initialised conditioning) has no Spearman and reports "
                 "null with its reason; no value is invented.")}


def _hidden(sites, count, geometry, rng):
    if geometry == "suffix":
        return tuple(sites[len(sites) - count:])
    chosen = sorted(int(position) for position in rng.permutation(len(sites))[:count])
    return tuple(sites[position] for position in chosen)


def completion_by_mask_count(services, route, model, panel, *, seed, rows, counts, batch_size):
    """Masked-window NLL per hidden site of development windows, by ``k``, geometry and bin."""
    sites = her2_core_edit_map()
    scaffold, assay = services.scaffold(), services.config["her2"]["assay"]
    constructs = [her2_construct(scaffold, row["window"], record_id=f"panel{index}")
                  for index, row in enumerate(panel[:int(rows)])]
    require(constructs, "the development panel is empty")
    result = {}
    for bin_name in (BIN_NULL, BIN_HIGH):
        condition = Condition(family="affinity", bin=bin_name, assay=assay)
        for geometry in GEOMETRIES:
            for count in counts:
                require(1 <= int(count) <= len(sites), f"mask count {count} is outside the window")
                views = []
                for index, construct in enumerate(constructs):
                    rng = np.random.default_rng([int(seed), KEY_SUBSET, index, int(count)])
                    hidden = _hidden(sites, int(count), geometry, rng)
                    views.append((construct,
                                  ablang_serialization(construct, masked_sites=hidden,
                                                       order=hidden), condition))
                per_row = []
                for start in range(0, len(views), int(batch_size)):
                    batch = services.collate(route, views[start:start + int(batch_size)])
                    log_probs = services.canonical_log_probs(route, model, batch)
                    targets = torch.as_tensor(batch.targets, dtype=torch.long,
                                              device=log_probs.device)
                    mask = torch.as_tensor(batch.target_mask, device=log_probs.device).to(
                        log_probs.dtype)
                    picked = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
                    per_row.extend((-(picked * mask).sum(dim=1) / mask.sum(dim=1)).tolist())
                result.setdefault(bin_name, {}).setdefault(geometry, {})[str(count)] = {
                    **_summary(per_row, "completion NLL"), "hidden_sites": int(count)}
    return {"unit": "mean negative log likelihood per hidden site", "window_sites": len(sites),
            "by_bin": result,
            "note": ("k = 10 hides the whole window and suffix = subset there; the table is on the "
                     "development windows and is not an affinity ranking")}


def order_sensitivity(services, route, model, panel, *, rows, batch_size):
    """The same full window scored with its sites revealed forwards and in reverse."""
    sites = her2_core_edit_map()
    scaffold, assay = services.scaffold(), services.config["her2"]["assay"]
    condition = Condition(family="affinity", bin=BIN_HIGH, assay=assay)
    window = panel[:int(rows)]
    constructs = [her2_construct(scaffold, row["window"], record_id=f"panel{index}")
                  for index, row in enumerate(window)]
    scores = {}
    for name, order in (("forward", tuple(sites)), ("reverse", tuple(reversed(sites)))):
        values = []
        for start in range(0, len(constructs), int(batch_size)):
            views = [(construct, ablang_serialization(construct, order=order), condition)
                     for construct in constructs[start:start + int(batch_size)]]
            batch = services.collate(route, views)
            values.extend((services.window_log_probs(route, model, batch).sum(dim=1)
                           / len(sites)).tolist())
        scores[name] = np.asarray(values, dtype=np.float64)
    outcomes = [float(row["value"]) for row in window]
    gap = np.abs(scores["forward"] - scores["reverse"])
    return {"rows": len(window), "unit": "mean log probability per site, bin=high",
            "mean_forward": _finite(scores["forward"].mean(), "forward"),
            "mean_reverse": _finite(scores["reverse"].mean(), "reverse"),
            "mean_abs_difference": _finite(gap.mean(), "order gap"),
            "max_abs_difference": _finite(gap.max(), "order gap"),
            "spearman_forward_vs_reverse": spearman_record(scores["forward"], scores["reverse"]),
            "spearman_forward_vs_pkd": spearman_record(scores["forward"], outcomes),
            "spearman_reverse_vs_pkd": spearman_record(scores["reverse"], outcomes)}


def training_fit_by_distance(services, route, model, pool, *, closure, window_of, keys, kappa,
                             seed, per_distance, batch_size):
    """Predicted against ``kappa * measured`` gap on a fixed sample of TRAINING pairs, per distance."""
    result = {}
    for distance in pool.distances:
        rng = np.random.default_rng([int(seed), KEY_FIT_PAIRS, int(distance)])
        pairs = [pool.draw(distance, rng) for _ in range(int(per_distance))]
        predicted, target = [], []
        for start in range(0, len(pairs), int(batch_size)):
            order = np.random.default_rng([int(seed), KEY_FIT_ORDER, int(distance), start])
            outcome = objective_lib.pair_outcome(
                services, route, model, pairs[start:start + int(batch_size)], closure=closure,
                window_of=window_of, training_population_keys=keys, kappa=kappa,
                rng=_NoDropout(order))
            predicted.extend(outcome.diagnostics["predicted_gaps"])
            target.extend(outcome.diagnostics["target_gaps"])
        predicted, target = np.asarray(predicted), np.asarray(target)
        residual = predicted - target
        signed = target != 0.0
        result[str(distance)] = {
            "pairs": len(pairs),
            "mean_predicted_gap": _finite(predicted.mean(), "predicted gap"),
            "mean_target_gap": _finite(target.mean(), "target gap"),
            "mean_abs_residual": _finite(np.abs(residual).mean(), "residual"),
            "rmse": _finite(math.sqrt(float((residual ** 2).mean())), "rmse"),
            "sign_agreement": (None if not signed.any() else _finite(
                (np.sign(predicted[signed]) == np.sign(target[signed])).mean(), "agreement")),
            "sign_agreement_pairs": int(signed.sum()),
            "spearman_predicted_vs_target": spearman_record(predicted, target)}
    return {"unit": "log-probability gap of the sequentially revealed joint, second endpoint minus "
                    "first, against kappa times the measured pKD gap",
            "population": "training pairs (a fit measure, not generalisation)",
            "by_distance": result}


def distance_reporting(configured_distances, realized):
    """The configured replacement distances and the realized counts, kept apart and labelled.

    ``configured_replacement_distances`` is what the arm's config would draw replacements at. It is
    present in the evaluation BEFORE any update as well, and is not evidence that a distance
    trained: only ``realized_training.replacement_distances`` (the loop's own tally at the time of
    this evaluation: empty before training) says what was drawn. The fit below scores the
    replacement pair objective, not the retained legacy single/pair and quartet local paths, whose
    counts are in ``realized_training.retained_legacy_local``.
    """
    return {
        "configured_replacement_distances": sorted(int(d) for d in configured_distances),
        "configured_note": ("the distances the arm is configured to replace local draws with; not "
                            "evidence that any of them trained"),
        "realized_training": realized,
        "objective_scope": ("the replacement measured-pair objective (objective.pair_outcome); the "
                            "retained legacy single/pair and quartet local paths are neither "
                            "scored nor distance-weighted here")}


def evaluate(services, route, model, heads, *, panel, pool, closure, window_of, keys, kappa, seed,
             diagnostics, configured_distances, realized):
    """All four blocks plus the explicit deferral, as one JSON-safe record.

    ``realized`` is the training loop's own tally so far (``runner.realized_training``), so the
    record states what was drawn and not only what was configured.
    """
    bound = int(diagnostics["batch_size"])
    with isolated(services, model, heads):
        return {
            "development_ranking": development_ranking(services, route, model, heads, panel,
                                                       batch_size=bound),
            "completion_by_mask_count": completion_by_mask_count(
                services, route, model, panel, seed=seed, rows=diagnostics["completion_rows"],
                counts=diagnostics["completion_counts"], batch_size=bound),
            "order_sensitivity": order_sensitivity(
                services, route, model, panel, rows=diagnostics["order_rows"], batch_size=bound),
            "training_fit_by_distance": {
                **training_fit_by_distance(
                    services, route, model, pool, closure=closure, window_of=window_of,
                    keys=keys, kappa=kappa, seed=seed,
                    per_distance=diagnostics["fit_pairs_per_distance"], batch_size=bound),
                **distance_reporting(configured_distances, realized)},
            "heldout_local_gap": {
                "status": "deferred",
                "reason": ("scoring pairs among development cores needs its own qualification "
                           "(pair eligibility, the closure, and a panel that never trains); the "
                           "pilot reports no number for it rather than a placeholder")}}
