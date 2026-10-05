"""Quality conditions as three independent tables, and the dropout that trains them.

The mistake this module exists to prevent is a single flat vocabulary of strings
like ``display_high`` and ``affinity_high``. With one table, dropping the assay
component of ``affinity_high:engelhart2022`` has to produce *some* token, and the
obvious choice -- ``high`` -- is the same token ``display_high`` collapses to. The
two are different questions: a display bin is a sorted FACS gate at one
concentration and an affinity bin is a third of a qualified KD distribution.
Collapsing them would train one embedding on both and then report a
quality-conditioned generation result that cannot distinguish them.

So conditions are a **product of three independent categoricals**:

* ``family`` in ``display`` / ``affinity`` / ``enrichment`` -- **never dropped**.
  The family says which kind of measurement is being requested, which is part of
  the question, not part of the answer.
* ``bin`` in ``low`` / ``mid`` / ``high`` / ``null``. ``null`` is the answer-free
  request and is what the observation heads always see. A recorded fraction of
  functional updates is switched to ``null`` so the model has a trained route for
  "generate under this assay, no quality requested".
* ``assay`` in the enrolled study identifiers, or ``null``. Dropped on its own
  recorded fraction, which is what creates a trained route for an *unseen* study.
  An untrained per-assay embedding is not that route, and this module cannot
  produce one: :meth:`ConditionSpec.encode` refuses an unknown assay name.

The three dropout rates are separate settings because they answer separate
questions, and both are applied from an explicit generator so a training view is
reproducible from ``(seed, update, row)`` rather than from global RNG state.
"""
from __future__ import annotations

from dataclasses import dataclass

from .her2_runtime import require

FAMILY_DISPLAY = "display"
FAMILY_AFFINITY = "affinity"
FAMILY_ENRICHMENT = "enrichment"
FAMILIES = (FAMILY_DISPLAY, FAMILY_AFFINITY, FAMILY_ENRICHMENT)

BIN_LOW = "low"
BIN_MID = "mid"
BIN_HIGH = "high"
BIN_NULL = "null"
BINS = (BIN_LOW, BIN_MID, BIN_HIGH, BIN_NULL)

ASSAY_NULL = "null"


@dataclass(frozen=True)
class Condition:
    """One resolved request: a family, a quality bin and an assay identity."""

    family: str
    bin: str
    assay: str

    def __post_init__(self):
        require(self.family in FAMILIES, f"Unknown condition family {self.family!r}")
        require(self.bin in BINS, f"Unknown condition bin {self.bin!r}")

    @property
    def is_answer_free(self):
        """True when no quality is requested. The observation heads see only these."""
        return self.bin == BIN_NULL

    def document(self):
        return {"family": self.family, "bin": self.bin, "assay": self.assay}


@dataclass(frozen=True)
class ConditionSpec:
    """The enrolled assay identities and the three dropout rates, frozen together."""

    assays: tuple
    bin_dropout: float = 0.10
    assay_dropout: float = 0.25
    families: tuple = FAMILIES

    def __post_init__(self):
        require(self.assays, "A condition spec with no enrolled assay cannot encode anything")
        require(len(set(self.assays)) == len(self.assays), "Duplicate assay identity")
        require(ASSAY_NULL not in self.assays,
                f"{ASSAY_NULL!r} is the reserved dropped-assay slot and cannot also be a study")
        for name, value in (("bin_dropout", self.bin_dropout),
                            ("assay_dropout", self.assay_dropout)):
            require(0.0 <= float(value) < 1.0, f"{name} must be in [0, 1), got {value!r}")

    @classmethod
    def from_enrolled(cls, vocabulary, *, bin_dropout=0.10, assay_dropout=0.25):
        """Build the spec from :func:`her2_routes_sources.derive_assay_vocabulary`.

        The alternative -- a hand-written list in a config -- is a second source
        of truth that drifts: the Li releases carry a per-row ``Assay`` column, so
        the identities the stream produces are a property of the data and not of
        the config. A name in the config that the stream never produces trains an
        embedding row on nothing; a name the stream produces that the config does
        not know raises inside the training loop.
        """
        names = tuple(dict(vocabulary)["assays"])
        require(names, "The enrolled observations produced no assay identity to condition on")
        return cls(assays=names, bin_dropout=float(bin_dropout),
                   assay_dropout=float(assay_dropout))

    @property
    def assay_order(self):
        """Assay identities in the frozen index order, with ``null`` last."""
        return (*sorted(self.assays), ASSAY_NULL)

    @property
    def sizes(self):
        return {"family": len(self.families), "bin": len(BINS), "assay": len(self.assay_order)}

    @property
    def rows(self):
        """Total embedding rows the model allocates for conditions."""
        return sum(self.sizes.values())

    def encode(self, condition):
        """``(family index, bin index, assay index)``. An unknown assay is refused."""
        require(isinstance(condition, Condition), "encode takes a Condition")
        require(condition.assay in self.assay_order,
                f"Assay {condition.assay!r} is not enrolled. An unseen study is served by the "
                f"dropped-assay route ({ASSAY_NULL!r}), never by an untrained embedding row.")
        return (self.families.index(condition.family), BINS.index(condition.bin),
                self.assay_order.index(condition.assay))

    def apply_dropout(self, condition, rng):
        """Independently drop the bin and the assay component. Family is untouched."""
        bin_name = condition.bin
        assay = condition.assay
        if bin_name != BIN_NULL and float(rng.random()) < float(self.bin_dropout):
            bin_name = BIN_NULL
        if assay != ASSAY_NULL and float(rng.random()) < float(self.assay_dropout):
            assay = ASSAY_NULL
        return Condition(family=condition.family, bin=bin_name, assay=assay)

    def answer_free(self, condition):
        """The same request with the quality answer removed. Used by every head batch."""
        return Condition(family=condition.family, bin=BIN_NULL, assay=condition.assay)

    def document(self):
        return {"families": list(self.families), "bins": list(BINS),
                "assays": list(self.assay_order), "sizes": dict(self.sizes),
                "embedding_rows": int(self.rows),
                "bin_dropout": float(self.bin_dropout),
                "assay_dropout": float(self.assay_dropout),
                "rule": ("family is never dropped; bin and assay are dropped independently at "
                         "the two declared rates. display_high and affinity_high cannot collapse "
                         "onto one another because the family component survives both dropouts.")}


# ---------------------------------------------------------------------------
# train-only bin thresholds
# ---------------------------------------------------------------------------

#: The population a set of thresholds is fitted over. A parental background is
#: part of it: a mutational scan's "high" third is high *relative to its own
#: parent*, and pooling two backgrounds inside one assay makes the bin a
#: statement about which parent a row came from.
PARENT_ANY = "*"


@dataclass(frozen=True)
class BinThresholds:
    """Frozen low/mid and mid/high cut points for one (family, assay, parent)."""

    family: str
    assay: str
    lower: float
    upper: float
    support: int
    scale: str
    direction: str
    parent: str = PARENT_ANY

    @property
    def population(self):
        return f"{self.family}::{self.assay}::{self.parent}"

    def document(self):
        return {"family": self.family, "assay": self.assay, "parent": self.parent,
                "population": self.population, "lower": float(self.lower),
                "upper": float(self.upper), "support": int(self.support), "scale": self.scale,
                "direction": self.direction}


def fit_bin_thresholds(values, *, family, assay, scale, direction="higher_is_better",
                       minimum_support=30, parent=PARENT_ANY):
    """Thirds of a **train-only** qualified distribution, with the tie rule fixed.

    The cut points are the 1/3 and 2/3 empirical quantiles of the finite,
    qualified, **train-only** values of one ``(family, assay, parent)``
    population. Train-only is the load-bearing word: thresholds fitted over every
    observation put the development and reserved rows' own values into the
    definition of the bin those rows are later conditioned on, which is a
    quantity the model can only get right by having seen them.

    Two further rules are frozen here rather than decided per call:

    * **ties.** A value exactly equal to a cut point falls in the *lower* bin
      (``<=`` lower is low, ``<= upper`` is mid). A dataset with a spike at the
      cut -- Adams' censored tails, Koenig's enrichment plateau -- otherwise
      distributes identical measurements across two bins depending on sort order.
    * **support.** Below ``minimum_support`` qualified values no thresholds are
      produced at all, and every row of that population trains with
      ``bin=null``. Thirds of eleven measurements is not a quality semantics.
    """
    import numpy as np
    require(direction in ("higher_is_better", "lower_is_better"),
            f"Unknown direction {direction!r}")
    finite = np.asarray([float(v) for v in values], dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size < int(minimum_support):
        return None
    lower, upper = (float(np.quantile(finite, 1.0 / 3.0)), float(np.quantile(finite, 2.0 / 3.0)))
    require(lower <= upper, "Quantile order inverted")
    if lower == upper:
        # A degenerate distribution has no thirds. Reporting that is the answer;
        # nudging one cut point would manufacture a bin boundary from nothing.
        return None
    return BinThresholds(family=family, assay=assay, parent=str(parent), lower=lower,
                         upper=upper, support=int(finite.size), scale=str(scale),
                         direction=direction)


def assign_bin(thresholds, *, value=None, interval=None):
    """The bin a measurement or a censoring interval falls in, or ``None``.

    A bin is assigned only when the whole measurement-or-censoring interval fits
    inside one bin. A right-censored Tite-Seq row at ``pKD >= 9.5`` is high only
    if ``9.5`` is already above the upper cut; a row whose interval straddles a
    cut is genuinely ambiguous and returns ``None`` -- it still trains the
    observation likelihood and the null-conditioned reconstruction.
    """
    if thresholds is None:
        return None
    low, high = (float(value), float(value)) if interval is None else (
        float("-inf") if interval[0] is None else float(interval[0]),
        float("inf") if interval[1] is None else float(interval[1]))
    require(low <= high, f"Inverted interval [{low}, {high}]")
    names = ((BIN_LOW, BIN_MID, BIN_HIGH) if thresholds.direction == "higher_is_better"
             else (BIN_HIGH, BIN_MID, BIN_LOW))
    edges = (thresholds.lower, thresholds.upper)
    band_of = lambda x: 0 if x <= edges[0] else (1 if x <= edges[1] else 2)
    if band_of(low) != band_of(high):
        return None
    return names[band_of(low)]
