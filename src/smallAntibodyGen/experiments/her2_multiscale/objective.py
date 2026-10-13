"""The pair objective: a sequentially revealed log-probability gap fitted to a measured pKD gap.

For a :class:`~.pairs.MeasuredPair` ``(A, B)`` at distance ``d`` (the endpoints differ at ``d``
coordinates ``S``) the objective is::

    J(X) = sum_t  log p( x_t | context, x_<t revealed )        over a revealed order of S
    loss = ( J(B) - J(A) - kappa * (y_B - y_A) ) ** 2          averaged over the pairs

with the existing :func:`~..her2_routes_objectives.fit_absolute_log_odds` (absolute log odds: no
reference model, no division by ``d``).

**What is scored.** Only the ``d`` coordinates the endpoints differ at. They are the only masked
tokens; every other residue stays visible and is identical in both endpoints, so the two rows
share one visible background and there is no hidden unchanged window. Both endpoints use the
*same* reveal order, drawn once per pair from the explicit ``rng`` argument (there is no global
generator). Step ``t`` queries ``order[t]`` with the endpoint's own residues revealed at
``order[:t]``, so step 1 has identical inputs for ``A`` and ``B`` and every later step sees all
predecessors. That is :meth:`FlightServices.window_log_probs`, which already reveals arbitrary
``d`` sequentially with gradients through every conditional; this module collates the views and
calls it rather than re-implementing the reveal. The first rows of ``A`` and ``B`` are identical
and are scored twice on purpose: a pair costs ``2 * d`` row-forwards and ``2 * d`` supervised
targets, which is what the outcome reports.

**Conditioning.** Quality is ``null`` for every row. The ``(family, null, assay)`` request goes
through the condition spec's dropout exactly once per pair and both endpoints share the result.
The ``rng`` is consumed per pair in a fixed order: the dropout draw(s), then the reveal order.

**What this is not.** The sum ``J`` is a scoring device. For a masked model that is not an exactly
consistent joint, ``J`` depends on the reveal order, which is why the order is randomised per
presentation rather than fixed. Two measured endpoints identify one number, ``y_B - y_A``; they do
not identify an interaction, an epistatic term, or the value of any intermediate genotype. This
is a term for mixed adaptation of the pretrained model, not a new sampler.

**Retained reconstruction.** The legacy local objective keeps the ordinary likelihood of the
alternatives beside the numeric term, because the fit alone can be satisfied by moving both
conditionals apart so that neither residue is likely. ``reconstruction_weight > 0`` keeps that term
on the SAME endpoints: one more forward over the same collated rows, every changed site hidden at
once, scored by the existing masked NLL and added to the loss as ``weight * nll``. It is reported
under its own name (``local_reconstruction``) and its rows, forwards and targets are counted in
the outcome. With the default ``0.0`` nothing extra runs and nothing extra is counted.

Rows of the collated batch alternate ``first, second`` per pair. Every pair in a call has the same
distance, so the batch is rectangular and no padded step is scored.
"""
from __future__ import annotations

import math
from numbers import Real

import numpy as np
import torch

from .. import her2_routes_data as data_lib
from .. import her2_routes_objectives as objectives_lib
from ..her2_routes_conditions import BIN_NULL, Condition
from ..her2_routes_serial import ablang_serialization
from ..her2_routes_tasks import TaskOutcome
from ..her2_runtime import require
from .pairs import MAX_DISTANCE, MeasuredPair, require_pair_current

#: ``her2_routes_services.ROUTE_DPLM_150M``, repeated because that module is the whole runtime and
#: this one must import without it. A test pins the two strings together.
ROUTE_DPLM_150M = "dplm_150m_masked"
SUPPORTED_ROUTES = (ROUTE_DPLM_150M,)

#: The coefficient key the training loop records. It names the objective that ran.
TERM = "local_effect"


def _kappa(value):
    require(isinstance(value, Real) and not isinstance(value, bool)
            and math.isfinite(float(value)) and float(value) > 0.0,
            f"kappa must be a finite positive number, got {value!r}")
    return float(value)


def _require_generator(rng):
    require(callable(getattr(rng, "random", None)) and callable(getattr(rng, "permutation", None)),
            "The pair objective needs an explicit random generator for the condition dropout and "
            "the reveal order; it never reads a global one")


def _common_distance(pairs):
    for pair in pairs:
        require(isinstance(pair, MeasuredPair), f"Expected a MeasuredPair, got {type(pair).__name__}")
    distances = {pair.distance for pair in pairs}
    require(len(distances) == 1,
            f"A pair microbatch has one distance so every row has the same number of scored "
            f"steps; got {sorted(distances)}")
    distance = distances.pop()
    require(isinstance(distance, int) and not isinstance(distance, bool)
            and 1 <= distance <= MAX_DISTANCE,
            f"A pair is at distance 1..{MAX_DISTANCE}, got {distance!r}")
    return distance


def _reconstruction_weight(value):
    require(isinstance(value, Real) and not isinstance(value, bool)
            and math.isfinite(float(value)) and float(value) >= 0.0,
            f"reconstruction_weight must be a finite non-negative number, got {value!r}")
    return float(value)


def pair_outcome(services, route, model, pairs, *, closure, window_of, training_population_keys,
                 kappa, rng, reconstruction_weight=0.0, family=data_lib.FAMILY_HER2):
    """Score ``pairs`` (all at one distance) and return the :class:`TaskOutcome`.

    ``closure`` (seen through ``window_of``) and ``training_population_keys`` are the consumer's
    boundary the pairs are re-validated against *now*, before any forward: a pair whose endpoints
    left the training partition or population, entered the closure or changed in value raises here
    however it was built. ``services`` supplies ``condition_spec``, ``collate`` and
    ``window_log_probs`` (and ``canonical_log_probs`` when ``reconstruction_weight`` is positive).
    """
    require(route in SUPPORTED_ROUTES,
            f"The pair objective is wired for {SUPPORTED_ROUTES} only, not {route!r}")
    kappa = _kappa(kappa)
    reconstruction_weight = _reconstruction_weight(reconstruction_weight)
    _require_generator(rng)
    pairs = tuple(pairs)
    require(pairs, "A pair microbatch with no pairs")
    distance = _common_distance(pairs)
    for pair in pairs:
        require_pair_current(pair, closure=closure, window_of=window_of,
                             training_population_keys=training_population_keys)
    views, orders = [], []
    for pair in pairs:
        condition = services.condition_spec.apply_dropout(
            Condition(family=pair.first.family, bin=BIN_NULL, assay=pair.first.assay), rng)
        order = tuple(pair.sites[int(index)] for index in rng.permutation(len(pair.sites)))
        orders.append(order)
        for summary in (pair.first, pair.second):
            construct = data_lib.with_sites(summary.construct, pair.sites)
            views.append((construct,
                          ablang_serialization(construct, masked_sites=pair.sites, order=order),
                          condition))
    batch = services.collate(route, views)
    per_site = services.window_log_probs(route, model, batch)
    require(tuple(per_site.shape) == (len(views), distance),
            f"Expected {len(views)} rows of {distance} sequential conditionals, got "
            f"{tuple(per_site.shape)}")
    targets = int(np.asarray(batch.target_mask).sum())
    require(targets == len(views) * distance,
            f"The batch supervises {targets} targets, expected {len(views) * distance}")
    totals = per_site.sum(dim=1)
    first_totals, second_totals = totals[0::2], totals[1::2]
    deltas = [pair.delta for pair in pairs]
    effect, fit = objectives_lib.fit_absolute_log_odds(second_totals, first_totals, deltas,
                                                       kappa=kappa)
    with torch.no_grad():
        predicted = (second_totals - first_totals).detach().cpu().tolist()
    terms = {TERM: float(effect.detach())}
    loss, row_forwards, row_targets = effect, distance, distance
    if reconstruction_weight > 0.0:
        log_probs = services.canonical_log_probs(route, model, batch)
        batch_targets = torch.as_tensor(batch.targets, dtype=torch.long, device=log_probs.device)
        batch_mask = torch.as_tensor(batch.target_mask, device=log_probs.device)
        nll, nll_diagnostics = objectives_lib.masked_nll(log_probs, batch_targets, batch_mask)
        loss = effect + reconstruction_weight * nll
        terms["local_reconstruction"] = float(nll.detach())
        fit["reconstruction_weight"] = reconstruction_weight
        fit["reconstruction_active_targets"] = int(nll_diagnostics["active_targets"])
        row_forwards += 1
        row_targets += distance
    identities = tuple(view[0].identity() for view in views)
    return TaskOutcome(
        loss=loss, term=TERM, terms=terms,
        diagnostics={**fit, "distance": distance,
                     "orders": [[[chain, int(index)] for chain, index in order]
                                for order in orders],
                     "target_gaps": [kappa * delta for delta in deltas],
                     "predicted_gaps": predicted,
                     "structural_digests": [pair.structural_digest for pair in pairs]},
        rows=len(views), forwards=len(views) * row_forwards, targets=len(views) * row_targets,
        identities=identities, measured=True, source=views[0][0].source, family=family,
        contributions=data_lib.contributions_by_source(
            sources=[view[0].source for view in views], identities=identities,
            forwards=row_forwards, targets=row_targets, measured=True, family=family))
