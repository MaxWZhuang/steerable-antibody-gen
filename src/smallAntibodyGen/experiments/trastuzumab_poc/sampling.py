"""Matched sample schedule for the trastuzumab PoC.

Algorithm: uniform antibody-family, then uniform owning-source WITHIN that family,
then uniform unique item within that (family, source) group.

Ab-only and paired arms select IDENTICAL qualified pair IDs, eligible masks, and
update batches.  The only difference is whether the resulting VisibleView includes Ag.

No Torch dependency.  RNG streams for masking are independent from model/dropout/eval.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from smallAntibodyGen.experiments.trastuzumab_poc.data import (
    QualifiedPopulation, QualifiedRecord,
)
from smallAntibodyGen.models.antibody_antigen.views import (
    Coord, mask_complete,
    MaskedTargets, VisibleView,
)


# ---------------------------------------------------------------------------
# Draw key and result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DrawKey:
    """Uniquely identifies one draw request.  Passed to deterministic draw API."""
    algorithm: str   # e.g. "family_first_v1"
    seed: int
    update: int
    example: int
    purpose: str     # "train_paired" | "train_abonly" | "mask" | etc.


@dataclass(frozen=True)
class SampledItem:
    """Result of one item draw (before masking)."""
    record_id: str
    source_id: str
    antibody_family: str
    target_family: str
    H: str
    L: Optional[str]
    Ag: str
    h_eligible: Tuple[int, ...]
    l_eligible: Optional[Tuple[int, ...]]
    aux_region: Optional[Tuple[Coord, ...]] = None

    def eligible_union(self) -> Tuple[Coord, ...]:
        coords: List[Coord] = [("H", i) for i in self.h_eligible]
        if self.l_eligible:
            coords += [("L", i) for i in self.l_eligible]
        return tuple(coords)


@dataclass(frozen=True)
class MaskSeed:
    """Seed and purpose for an independent masking RNG stream."""
    seed: int
    purpose: str


@dataclass(frozen=True)
class SampledBatch:
    """One update batch: paired_items and abonly_items share record_ids."""
    update: int
    paired_items: Tuple[SampledItem, ...]
    abonly_items: Tuple[SampledItem, ...]
    mask_seeds: Tuple[MaskSeed, ...]  # one per item, independent of model/arm

    def __post_init__(self) -> None:
        if len(self.paired_items) != len(self.abonly_items):
            raise ValueError("paired_items and abonly_items must have equal length")
        paired_ids = tuple(i.record_id for i in self.paired_items)
        abonly_ids = tuple(i.record_id for i in self.abonly_items)
        if paired_ids != abonly_ids:
            raise ValueError(
                "paired and abonly batches must select identical record_ids in identical order"
            )
        if len(self.mask_seeds) != len(self.paired_items):
            raise ValueError("mask_seeds must have one entry per item")


# ---------------------------------------------------------------------------
# Sampler configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SamplerConfig:
    """Configuration for the family-first balanced sampler.

    batch_size: number of items per update batch.
    algorithm: identifier string for the draw algorithm.
    base_seed: integer seed for the sampler (reproducible draws).
    mask_purpose: purpose label for the mask RNG stream.
    """
    batch_size: int
    algorithm: str
    base_seed: int
    mask_purpose: str = "uniform_mask"

    def __post_init__(self) -> None:
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int):
            raise TypeError("batch_size must be an int")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if not isinstance(self.algorithm, str) or not self.algorithm:
            raise ValueError("algorithm must be a nonempty string")
        if isinstance(self.base_seed, bool) or not isinstance(self.base_seed, int):
            raise TypeError("base_seed must be an int")
        if not isinstance(self.mask_purpose, str) or not self.mask_purpose:
            raise ValueError("mask_purpose must be a nonempty string")


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

class FamilyFirstSampler:
    """Family-first balanced sampler over a qualified population.

    Sampling is row-order invariant: shuffling the input QualifiedPopulation records
    produces identical draws for identical (update, example) keys.

    The paired and abonly arms select identical record_ids.  Masking RNG is independent
    from the item-selection RNG and from any model/dropout/eval RNG.

    Only train partition records are exposed.
    """

    ALGORITHM = "family_first_v1"

    def __init__(self, population: QualifiedPopulation, config: SamplerConfig) -> None:
        if not isinstance(population, QualifiedPopulation):
            raise TypeError("population must be a QualifiedPopulation")
        if not isinstance(config, SamplerConfig):
            raise TypeError("config must be a SamplerConfig")
        if config.algorithm != self.ALGORITHM:
            raise ValueError(
                f"SamplerConfig.algorithm {config.algorithm!r} != {self.ALGORITHM!r}"
            )

        train = population.train_records()
        if not train:
            raise ValueError("No train records in population; cannot create sampler")

        # Build stable sorted index structures (sorted canonical order = row-order invariant)
        # family -> source -> [record_id, ...]  — all lists sorted for determinism
        self._family_source_map: Dict[str, Dict[str, List[str]]] = {}
        self._record_by_id: Dict[str, QualifiedRecord] = {}

        for rec in train:
            self._record_by_id[rec.record_id] = rec
            fam = rec.antibody_family
            src = rec.owning_source_id
            self._family_source_map.setdefault(fam, {}).setdefault(src, []).append(rec.record_id)

        # Sort all lists for determinism
        self._families: Tuple[str, ...] = tuple(sorted(self._family_source_map.keys()))
        for fam in self._families:
            for src in self._family_source_map[fam]:
                self._family_source_map[fam][src].sort()
            # Sort sources within family for determinism
            self._family_source_map[fam] = dict(
                sorted(self._family_source_map[fam].items())
            )

        self._config = config
        self._population_digest = population.population_digest

    def draw_batch(self, update: int) -> SampledBatch:
        """Draw one update batch.  Identical (update) -> identical batch."""
        if isinstance(update, bool) or not isinstance(update, int) or update < 0:
            raise ValueError("update must be a non-negative int")

        # Item selection RNG: independent of mask RNG

        record_ids: List[str] = []
        for ex in range(self._config.batch_size):
            # Each example gets its own independent draw via re-seeded RNG
            ex_rng = self._make_rng(purpose="item_select", update=update, example=ex)
            rid = self._draw_one(ex_rng)
            record_ids.append(rid)

        # Build SampledItem for each arm (paired / abonly) — identical record IDs
        paired_items: List[SampledItem] = []
        abonly_items: List[SampledItem] = []
        mask_seeds: List[MaskSeed] = []

        for ex, rid in enumerate(record_ids):
            rec = self._record_by_id[rid]
            # Paired item includes Ag (caller builds view with Ag)
            paired_items.append(_record_to_item(rec))
            # Abonly item is identical (arm difference resolved at view-construction time)
            abonly_items.append(_record_to_item(rec))
            # Independent masking seed
            mask_seed = self._make_mask_seed(update=update, example=ex)
            mask_seeds.append(mask_seed)

        return SampledBatch(
            update=update,
            paired_items=tuple(paired_items),
            abonly_items=tuple(abonly_items),
            mask_seeds=tuple(mask_seeds),
        )

    def draw_item_keyed(self, key: DrawKey) -> SampledItem:
        """Deterministic single-item draw keyed by DrawKey."""
        if key.algorithm != self._config.algorithm or key.seed != self._config.base_seed:
            raise ValueError("draw key must match sampler algorithm and base seed")
        rng = self._make_rng(purpose=key.purpose, update=key.update, example=key.example)
        rid = self._draw_one(rng)
        return _record_to_item(self._record_by_id[rid])

    def mask_rng(self, update: int, example: int) -> random.Random:
        """Return an independent masking RNG for the given update/example.

        This RNG is separate from item-selection and from any model RNG.
        Calling it does NOT affect item-selection draws.
        """
        return self._make_rng_obj(
            purpose=self._config.mask_purpose,
            update=update,
            example=example,
        )

    def _draw_one(self, rng: random.Random) -> str:
        """Family-first draw: uniform family -> uniform source -> uniform item."""
        family = rng.choice(self._families)
        sources = tuple(sorted(self._family_source_map[family].keys()))
        source = rng.choice(sources)
        items = self._family_source_map[family][source]
        return rng.choice(items)

    def _make_rng(self, purpose: str, update: int, example: int) -> random.Random:
        return self._make_rng_obj(purpose=purpose, update=update, example=example)

    def _make_rng_obj(self, purpose: str, update: int, example: int) -> random.Random:
        if any(type(x) is not int or x < 0 for x in (update, example)):
            raise ValueError("update and example must be non-negative integers")
        seed_str = json.dumps({
            "alg": self._config.algorithm,
            "base_seed": self._config.base_seed,
            "pop_digest": self._population_digest,
            "purpose": purpose,
            "update": update,
            "example": example,
        }, sort_keys=True)
        digest = hashlib.sha256(seed_str.encode("utf-8")).digest()
        seed_int = int.from_bytes(digest[:8], "big")
        rng = random.Random(seed_int)
        return rng

    def _make_mask_seed(self, update: int, example: int) -> MaskSeed:
        seed_str = json.dumps({
            "alg": self._config.algorithm,
            "base_seed": self._config.base_seed,
            "pop_digest": self._population_digest,
            "purpose": self._config.mask_purpose,
            "update": update,
            "example": example,
        }, sort_keys=True)
        digest = hashlib.sha256(seed_str.encode("utf-8")).digest()
        seed_int = int.from_bytes(digest[:8], "big")
        return MaskSeed(seed=seed_int, purpose=self._config.mask_purpose)


# ---------------------------------------------------------------------------
# View construction helpers
# ---------------------------------------------------------------------------

def build_paired_view(
    item: SampledItem,
    masked_coords: Sequence[Coord],
) -> Tuple[VisibleView, MaskedTargets]:
    """Build the paired (H + L + Ag) view for a sampled item."""
    seqs: Dict[str, str] = {"H": item.H, "Ag": item.Ag}
    if item.L is not None:
        seqs["L"] = item.L
    return mask_complete(seqs, masked_coords)


def build_abonly_view(
    item: SampledItem,
    masked_coords: Sequence[Coord],
) -> Tuple[VisibleView, MaskedTargets]:
    """Build the ab-only (H + L, no Ag) view for a sampled item.

    Uses the same masked_coords as the paired arm — arm difference is Ag inclusion only.
    """
    seqs: Dict[str, str] = {"H": item.H}
    if item.L is not None:
        seqs["L"] = item.L
    return mask_complete(seqs, masked_coords)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _record_to_item(rec: QualifiedRecord) -> SampledItem:
    return SampledItem(
        record_id=rec.record_id,
        source_id=rec.owning_source_id,
        antibody_family=rec.antibody_family,
        target_family=rec.target_family,
        H=rec.H,
        L=rec.L,
        Ag=rec.Ag,
        h_eligible=rec.h_eligible,
        l_eligible=rec.l_eligible,
        aux_region=rec.aux_region,
    )
