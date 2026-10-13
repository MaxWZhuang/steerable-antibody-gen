"""How an arm changes a base draw: whole-window masks, and measured pairs in place of local contrasts.

The base stream (``FlightServices.curriculum_stream``) is drawn first, in EVERY arm, exactly as the
production flight draws it. It owns which task and family each microbatch gets, which rows, which
condition, which tile, and every legacy quartet. Nothing here touches the generator it consumes, so
the four arms of one pilot see the same source, task and quartet sequence and differ only in what
this module does to a draw afterwards.

Every change draws from its OWN generator, ``numpy.random.default_rng([seed, step, micro, ...,
key])``, with a key per purpose. The arm is never part of a key, which is what makes two arms agree on
anything they share:

* **Masks** (whole-window task only). Each view of the base draw keeps its construct, its chunk of the
  common region, its condition, its normalisation and its preference term; only the serialisation
  changes, to ``masks.draw_mask`` over that chunk. The full/partial decision and ``k`` come from a
  count generator keyed by the view's row, the positions from a separate position generator, so a
  ``suffix`` arm and a ``subset`` arm hide the same ``k`` sites' worth of the same rows and differ
  only in which sites. The full probability is ``1 - masked_progressive_probability`` of the base
  config, because the base stream's progressive draw reveals a prefix with that probability. The
  ``suffix`` arm is therefore the same LAW as the base stream's progressive mask, not a replay of its
  random path: the base draws its mask from the view generator, this from keyed generators.
* **Pairs** (HER2 local draws that contain no quartet). With a declared probability the contrasts
  of such a draw are replaced by measured pairs from the pool: one distance for the microbatch, then
  ``rows / 2`` pairs at it. The same draws are replaced in every arm (the decision is keyed without
  the arm); the arms differ in which distances the replacement uses. Quartet-containing draws and
  other-functional local draws pass through untouched.
"""
from __future__ import annotations

import dataclasses
import hashlib

import numpy as np

from .. import her2_routes_data as data_lib
from ..her2_routes_serial import ablang_serialization
from ..her2_runtime import require
from . import masks as masks_lib
from . import objective as objective_lib

#: One key per purpose, so no change shares a stream with another or with the base stream.
KEY_COUNT, KEY_POSITION = 71, 72
KEY_REPLACE, KEY_DISTANCE, KEY_PAIRS, KEY_OBJECTIVE = 73, 74, 75, 76


def full_probability(curriculum):
    """``1 - masked_progressive_probability``: the share of whole-window draws that hide everything."""
    progressive = float(curriculum.get("masked_progressive_probability", 0.5))
    return masks_lib.validated_full_probability(1.0 - progressive)


def replace_masks(draw, *, seed, step, micro, geometry, probability):
    """``(draw, documents)``: the whole-window draw with every view's mask re-drawn.

    ``documents`` is one ``MaskDraw.document()`` per view (mode, ``k``, region size, geometry).
    """
    require(draw.task == data_lib.TASK_WHOLE,
            f"masks are replaced on whole-window draws only, not {draw.task!r}")
    views, documents = [], []
    for row, (construct, _serialization, condition) in enumerate(draw.views):
        count_rng = np.random.default_rng([int(seed), int(step), int(micro), row, KEY_COUNT])
        position_rng = np.random.default_rng([int(seed), int(step), int(micro), row, KEY_POSITION])
        mask = masks_lib.draw_mask(
            construct.sites, heavy_length=len(construct.heavy), light_length=len(construct.light),
            count_rng=count_rng, position_rng=position_rng, geometry=geometry,
            full_probability=probability)
        views.append((construct,
                      ablang_serialization(construct, masked_sites=mask.sites, order=mask.sites),
                      condition))
        documents.append(mask.document())
    return dataclasses.replace(draw, views=tuple(views)), documents


def summarise_masks(documents):
    """Counts of full and partial masks and the histogram of ``k``, for the progress record."""
    modes, counts = {}, {}
    for document in documents:
        modes[document["mode"]] = modes.get(document["mode"], 0) + 1
        counts[str(document["count"])] = counts.get(str(document["count"]), 0) + 1
    return {"views": len(documents), "modes": modes, "counts": dict(sorted(counts.items()))}


def choose_distance(weights, rng):
    """One distance from ``{distance: probability}``; the last one absorbs float rounding."""
    draw, total = float(rng.random()), 0.0
    ordered = sorted(weights.items())
    for distance, probability in ordered:
        total += float(probability)
        if draw < total:
            return int(distance)
    return int(ordered[-1][0])


def _identity_digest(identity):
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


class LocalPairs:
    """Replace eligible HER2 local draws with measured pairs, and score them.

    ``training_keys``, ``closure`` and ``window_of`` are the consumer's boundary; the objective
    re-validates every pair against them on every draw, so a pool built earlier cannot carry a
    row that has since left the training population.
    """

    def __init__(self, pool, *, seed, probability, distance_weights, closure, window_of,
                 training_keys, kappa, reconstruction_weight):
        require(0.0 <= float(probability) <= 1.0, "the replacement probability is in [0, 1]")
        require(distance_weights and all(distance in pool.distances
                                         for distance in distance_weights),
                f"distance weights {sorted(distance_weights)} name a distance the pool "
                f"({pool.distances}) does not hold; a missing distance is a preflight error and "
                "is never renormalised away")
        self.pool, self.seed, self.probability = pool, int(seed), float(probability)
        self.weights = dict(distance_weights)
        self.closure, self.window_of, self.keys = closure, window_of, training_keys
        self.kappa, self.reconstruction_weight = float(kappa), float(reconstruction_weight)

    @staticmethod
    def eligible(draw):
        """A HER2 local draw whose contrasts are all pairs (no quartet, nothing coupled)."""
        return (draw.task == data_lib.TASK_LOCAL and draw.family == data_lib.FAMILY_HER2
                and bool(draw.contrasts)
                and all(contrast.kind != data_lib.CONTRAST_COUPLED for contrast in draw.contrasts))

    def _rng(self, step, micro, key):
        return np.random.default_rng([self.seed, int(step), int(micro), key])

    def wants(self, draw, *, step, micro):
        """Is this eligible draw replaced? Keyed by (seed, step, micro) only: the same in every arm."""
        if not self.eligible(draw):
            return False
        return float(self._rng(step, micro, KEY_REPLACE).random()) < self.probability

    def sample(self, draw, *, step, micro):
        """``(distance, pairs)`` with the base draw's endpoint-row count: one distance, ``rows/2`` pairs."""
        rows = int(draw.rows())
        require(rows >= 2 and rows % 2 == 0, f"a pair replacement needs an even row count, got {rows}")
        distance = choose_distance(self.weights, self._rng(step, micro, KEY_DISTANCE))
        rng = self._rng(step, micro, KEY_PAIRS)
        return distance, [self.pool.draw(distance, rng) for _ in range(rows // 2)]

    def outcome(self, services, route, model, draw, *, step, micro):
        """``(outcome, document)`` of the replacement for ``draw``; the objective is the real one."""
        distance, pairs = self.sample(draw, step=step, micro=micro)
        outcome = objective_lib.pair_outcome(
            services, route, model, pairs, closure=self.closure, window_of=self.window_of,
            training_population_keys=self.keys, kappa=self.kappa,
            rng=self._rng(step, micro, KEY_OBJECTIVE),
            reconstruction_weight=self.reconstruction_weight, family=draw.family)
        require(outcome.rows == draw.rows(),
                f"the replacement presents {outcome.rows} endpoint rows, the base draw {draw.rows()}")
        document = {"kind": "measured_pairs", "distance": distance, "pairs": len(pairs),
                    "base_contrasts": len(draw.contrasts), "base_rows": int(draw.rows()),
                    "structural_digests": [pair.structural_digest for pair in pairs],
                    "identity_digests": [_identity_digest(identity)
                                         for pair in pairs for identity in pair.identities],
                    "row_forwards": int(outcome.forwards) // int(outcome.rows)}
        return outcome, document
