"""Whole-mask draws: how many sites of a common region to hide, and which.

One draw hides a subset of a declared common region ``R`` of ``n`` coordinates:

* with probability ``full_probability`` (default 0.5) it hides **all** ``n`` -- the whole-window
  case the model is later asked to generate;
* otherwise it hides ``k`` sites with ``k`` uniform on ``1..n-1``. A partial draw therefore never
  hides nothing and never duplicates the full mask. A one-coordinate region is always full.

Which ``k`` sites a partial draw hides is the only thing the *geometry* changes:
``"suffix"`` (the baseline) hides the last ``k`` coordinates of the region, ``"subset"`` hides
``k`` coordinates drawn uniformly without replacement.

**Region order.** A region arrives as any ordered tuple or list of ``(chain, index)`` pairs.
:func:`canonical_region` sorts it heavy-then-light, ascending, once, and everything below -- the
suffix, the subset, the returned sites -- is defined on that canonical order, so the order a caller
happened to list the coordinates in changes nothing. "Suffix" means the last ``k`` coordinates in
canonical order, not the C-terminal end of a loop.

**Two generators, constant consumption.** The full/partial decision and ``k`` come from
``count_rng``; the subset positions come from ``position_rng``. A call consumes the same amount
from each generator whichever geometry it runs and whichever mode it draws (two uniform floats
from ``count_rng``; one permutation of ``n`` from ``position_rng``, used only by a partial
``"subset"`` draw). Two runs of one seed that differ only in geometry therefore draw identical
modes and counts for every row, and leave both generators in identical states.

No label, value or bin is an argument: the draw cannot depend on a measurement.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from ..her2_routes_serial import CHAINS, HEAVY, LIGHT
from ..her2_runtime import require

DEFAULT_FULL_PROBABILITY = 0.5

GEOMETRY_SUFFIX = "suffix"
GEOMETRY_SUBSET = "subset"
GEOMETRIES = (GEOMETRY_SUFFIX, GEOMETRY_SUBSET)

MODE_FULL = "full"
MODE_PARTIAL = "partial"


def _length(value, name):
    require(isinstance(value, Integral) and not isinstance(value, bool) and int(value) >= 0,
            f"{name} must be a non-negative integer, got {value!r}")
    return int(value)


def canonical_region(region, *, heavy_length, light_length):
    """The region as a sorted, validated tuple of ``(chain, index)`` pairs.

    A region is a non-empty tuple or list of ``(chain, index)`` pairs, in any order. The chain is
    exactly ``"H"`` or ``"L"``; the index is an integer (``bool`` is not one), non-negative and
    inside that chain's length; no coordinate repeats. Nothing is coerced, so ``"1"``, ``1.0`` and
    ``True`` are refused rather than read as ``1``. The result is heavy-then-light, ascending.
    """
    lengths = {HEAVY: _length(heavy_length, "heavy_length"),
               LIGHT: _length(light_length, "light_length")}
    require(isinstance(region, (tuple, list)),
            f"A region must be a tuple or list of (chain, index) pairs, got {type(region).__name__}")
    require(len(region) > 0, "A region names no coordinates; there would be nothing to mask")
    seen = set()
    for site in region:
        require(isinstance(site, (tuple, list)) and len(site) == 2,
                f"A region coordinate must be a (chain, index) pair, got {site!r}")
        chain, index = site
        require(isinstance(chain, str) and chain in CHAINS,
                f"A region chain must be one of {CHAINS}, got {chain!r}")
        require(isinstance(index, Integral) and not isinstance(index, bool),
                f"A region index must be an integer, got {index!r}")
        index = int(index)
        require(index >= 0, f"A region index must be non-negative, got {(chain, index)}")
        require(index < lengths[chain],
                f"Region coordinate {(chain, index)} is outside the {lengths[chain]}-residue "
                f"{chain} chain")
        require((chain, index) not in seen, f"Duplicate coordinate {(chain, index)} in the region")
        seen.add((chain, index))
    return tuple(sorted(seen, key=lambda site: (CHAINS.index(site[0]), site[1])))


def validated_full_probability(value):
    """``value`` as a float in ``[0, 1]``. NaN, infinities, ``bool`` and non-numbers are refused."""
    require(isinstance(value, Real) and not isinstance(value, bool),
            f"full_probability must be a real number, got {value!r}")
    probability = float(value)
    require(math.isfinite(probability) and 0.0 <= probability <= 1.0,
            f"full_probability must be a finite number in [0, 1], got {value!r}")
    return probability


def _require_separate_generators(count_rng, position_rng):
    for name, rng in (("count_rng", count_rng), ("position_rng", position_rng)):
        require(isinstance(rng, np.random.Generator),
                f"{name} must be a numpy Generator, got {type(rng).__name__}")
    require(count_rng is not position_rng and count_rng.bit_generator is not position_rng.bit_generator,
            "count_rng and position_rng must be separate generators: a geometry that consumed "
            "position draws from the count stream would shift every later mask count")


@dataclass(frozen=True)
class MaskDraw:
    """One draw: the sites to hide (canonical order), how many, and how it was chosen."""

    sites: tuple
    count: int
    size: int
    mode: str
    geometry: str

    def document(self):
        """The loggable part of the draw: mode and ``k`` (``count``), not the coordinates."""
        return {"mode": self.mode, "count": int(self.count), "region_size": int(self.size),
                "geometry": self.geometry}


def draw_mask(region, *, heavy_length, light_length, count_rng, position_rng,
              geometry=GEOMETRY_SUFFIX, full_probability=DEFAULT_FULL_PROBABILITY):
    """Draw which sites of ``region`` to hide. See the module docstring for the contract.

    Everything is validated before either generator is touched, so a refused call leaves both
    streams exactly where they were.
    """
    sites = canonical_region(region, heavy_length=heavy_length, light_length=light_length)
    require(geometry in GEOMETRIES, f"geometry must be one of {GEOMETRIES}, got {geometry!r}")
    probability = validated_full_probability(full_probability)
    _require_separate_generators(count_rng, position_rng)
    size = len(sites)
    decision = float(count_rng.random())
    spread = float(count_rng.random())
    ranking = [int(position) for position in position_rng.permutation(size)]
    if size == 1 or decision < probability:
        return MaskDraw(sites=sites, count=size, size=size, mode=MODE_FULL, geometry=geometry)
    # int(spread * (n - 1)) is in 0..n-2; the clamp guards the last float below 1.0 rounding up.
    count = 1 + min(int(spread * (size - 1)), size - 2)
    if geometry == GEOMETRY_SUFFIX:
        chosen = sites[size - count:]
    else:
        chosen = tuple(sites[position] for position in sorted(ranking[:count]))
    return MaskDraw(sites=chosen, count=count, size=size, mode=MODE_PARTIAL, geometry=geometry)
