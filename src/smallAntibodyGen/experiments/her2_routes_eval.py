"""The E1-E22 panels: numeric where supported, support-gated with counts otherwise.

Every panel reports population size, neighbourhood count, assay, distance
distribution and missing-data policy beside its score. A panel whose measurements
are unavailable is **explicitly untested with its blocking reason**, never quietly
dropped -- and "no measured HER2 counter-target exists" therefore produces no
specificity claim rather than a specificity claim with a caveat.

The frozen family clustering is the part most easily got wrong. Clusters are built
**once**, from the pooled unassigned unique raw candidates of *every* preregistered
arm at the *maximum* draw count, sorted lexicographically, greedily assigned at
radius one -- and then frozen for every arm and every nested draw prefix. Rebuilding
clusters at each sample count changes the measurement, so
:class:`FrozenClustering` refuses to be rebuilt.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .her2_routes_controls import hamming
from .her2_runtime import require

PANELS = {
    "E1": "antibody prior retention",
    "E2": "conditional completion",
    "E3": "display-group ranking",
    "E4": "local measured ordering",
    "E5": "opposite-direction substitutions",
    "E6": "quantitative affinity",
    "E7": "combination generalisation",
    "E8": "measured interaction or rescue",
    "E9": "raw generation and revision",
    "E10": "measured-catalogue behaviour",
    "E11": "structural contribution",
    "E12": "target specificity",
    "E13": "data efficiency",
    "E14": "repeatability",
    "E15": "functional transfer",
    "E16": "distinct functional readouts",
    "E17": "learning from context",
    "E18": "teacher contribution",
    "E19": "distance shortcuts",
    "E20": "antigen information",
    "E21": "local learning versus full generation",
    "E22": "generative mechanism",
}

#: Panels with no local measurement, and the exact reason. A claim is not made.
UNSUPPORTED = {
    "E12": ("no measured HER2 counter-target outcome exists in any enrolled source. A different "
            "antigen sequence, a swapped light chain or an unrelated binder is not automatically "
            "a measured negative, so NO specificity claim is made. The non-HER2 AlphaSeq releases "
            "DO carry an avg_off_target_affinity column and the Li release carries three named "
            "negative-control targets; both are preserved raw and counted in the source "
            "inventory, and neither is a HER2 counter-target. A specificity panel for THOSE "
            "targets is buildable from them and is not this panel."),
    "E20": ("the only multi-target measured resource here is the ASD antibody-antigen set, whose "
            "stage-4 split leaks 78.6% with 381 clean validation rows, and the germline framework "
            "alone identifies the target about 800x above chance under clone-grouped splits. A "
            "constant antigen also supplies no within-task variation for fixed-HER2. Reported as "
            "not runnable for a molecular-reasoning claim."),
}


def panel_record(panel, *, population, value=None, interval=None, assay=None, distances=None,
                 supported=True, reason=None, extra=None):
    """The uniform envelope every panel result is reported in."""
    require(panel in PANELS, f"Unknown panel {panel!r}")
    record = {"panel": panel, "name": PANELS[panel], "supported": bool(supported),
              "population": int(population), "assay": assay,
              "value": None if value is None else float(value),
              "interval": (None if interval is None
                           else [float(interval[0]), float(interval[1])]),
              "blocking_reason": reason or UNSUPPORTED.get(panel),
              "distance_distribution": (None if distances is None else
                                        {str(int(k)): int(v) for k, v in
                                         zip(*np.unique(np.asarray(distances), return_counts=True))}),
              "missing_data_policy": ("a blank outcome is a missing number, never a negative; "
                                      "rows without a qualified value are excluded and counted")}
    if extra:
        record.update(dict(extra))
    if not supported:
        require(record["blocking_reason"],
                f"{panel}: an unsupported panel must carry its blocking reason")
        require(record["value"] is None,
                f"{panel}: an unsupported panel cannot also report a value")
    return record


# ---------------------------------------------------------------------------
# ranking statistics
# ---------------------------------------------------------------------------

def finite_vector(values, name):
    """``values`` as a 1-D float vector, refusing None, NaN and +/-inf rather than ranking them."""
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a 1-D numeric sequence: {error}") from error
    require(array.ndim == 1, f"{name} must be 1-D, got shape {array.shape}")
    require(bool(np.isfinite(array).all()),
            f"{name} contains {int((~np.isfinite(array)).sum())} missing or non-finite "
            f"value(s); refusing to rank invalid evidence")
    return array


def finite_scalar(value, name):
    """``value`` as a finite float; None, NaN and +/-inf are refused with the field name."""
    require(value is not None, f"{name} is missing")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number, got {value!r}") from error
    require(np.isfinite(number), f"{name} must be finite, got {value!r}")
    return number


def spearman(scores, outcomes):
    """Spearman's rho with average ranks for ties. Returns ``None`` below two points.

    Scores and outcomes must be finite and aligned; invalid evidence raises ``ValueError``
    instead of being sorted as an ordinary rank. ``None`` still means a constant or
    undersupported (but valid) input.
    """
    scores = finite_vector(scores, "scores")
    outcomes = finite_vector(outcomes, "outcomes")
    require(scores.shape == outcomes.shape,
            f"Misaligned scores and outcomes: {scores.shape} vs {outcomes.shape}")
    if scores.size < 2:
        return None
    left, right = _average_ranks(scores), _average_ranks(outcomes)
    if left.std() == 0 or right.std() == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _average_ranks(values):
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    position = 0
    while position < values.size:
        end = position
        while end + 1 < values.size and values[order[end + 1]] == values[order[position]]:
            end += 1
        ranks[order[position:end + 1]] = 0.5 * (position + end) + 1.0
        position = end + 1
    return ranks


def censoring_aware_concordance(scores, lower, upper):
    """Pair concordance that skips pairs whose censored intervals overlap.

    A right-censored ``pKD >= 9.5`` and a point ``9.8`` are not orderable: the true
    value could be either side. Counting that pair as discordant penalises a model
    for the assay's detection limit.
    """
    scores = np.asarray(scores, dtype=np.float64)
    lower = np.asarray([-np.inf if v is None else float(v) for v in lower])
    upper = np.asarray([np.inf if v is None else float(v) for v in upper])
    pairs, concordant, skipped, ties = 0, 0.0, 0, 0
    for i in range(scores.size):
        for j in range(i + 1, scores.size):
            if lower[i] > upper[j]:
                ordered = 1.0
            elif lower[j] > upper[i]:
                ordered = -1.0
            else:
                skipped += 1
                continue
            pairs += 1
            difference = scores[i] - scores[j]
            if difference == 0.0:
                # A tied SCORE on an orderable pair is neither concordant nor
                # discordant. Counting it as discordant penalises a model for a
                # coincidence and makes a constant scorer score zero instead of the
                # 0.5 that a coin flip earns.
                concordant += 0.5
                ties += 1
            elif difference * ordered > 0:
                concordant += 1
    return {"pairs": pairs, "skipped_unorderable": skipped, "score_ties": ties,
            "concordance": None if pairs == 0 else concordant / pairs,
            "rule": ("a pair whose censored intervals overlap is not orderable and is skipped, not "
                     "counted as discordant. A tied SCORE on an orderable pair earns half credit, "
                     "so a constant scorer measures 0.5 rather than 0.")}


def paired_bootstrap(differences, *, seed, draws=2000, confidence=0.95):
    """Bootstrap interval over PAIRED per-unit differences of a supplied quantity.

    A general utility, and deliberately **not** the reserved endpoint: the caller
    supplies the per-unit differences, so what the interval means is whatever the
    caller differenced. Applied to raw log scores it is moved by an arbitrary additive
    offset in one arm and moves no ordering, which is why the reserved endpoint is
    :func:`paired_delta_spearman` -- a bootstrap of the difference of *correlations* on
    the same outcomes.

    Pairing is available because arms are scored on identical units, and it is much
    tighter than two independent intervals. The units are sequence neighbourhoods --
    linked contrasts and replicates travel together -- and **selection batches are
    not uncertainty units**.
    """
    values = np.asarray(differences, dtype=np.float64)
    require(values.size > 1, "A paired bootstrap needs more than one unit")
    rng = np.random.default_rng(int(seed))
    means = np.asarray([values[rng.integers(0, values.size, values.size)].mean()
                        for _ in range(int(draws))])
    alpha = (1.0 - float(confidence)) / 2.0
    low, high = float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))
    return {"units": int(values.size), "mean_difference": float(values.mean()),
            "interval": [low, high], "excludes_zero": bool(low > 0 or high < 0),
            "draws": int(draws), "confidence": float(confidence),
            "unit": "sequence neighbourhood; linked contrasts and replicates stay together",
            "not_a_unit": "a selection batch. Batch diameters were 6-10 edits; they are not "
                          "families and not bootstrap units.",
            "seed_variation_note": ("this interval is a within-seed row bootstrap and is NOT a "
                                    "statement about training-seed variability; the per-seed table "
                                    "is reported separately")}


def average_precision(scores, labels):
    """Macro-averagable AP for one class, **threshold-grouped**. Used by E3.

    Delegated to ``her2_nf_metrics.average_precision``, which is the repository's
    tie-grouped implementation and has been checked against scikit-learn on eighty
    tie-heavy cases to 1.11e-16. A third implementation here was order-dependent on
    tied scores: a constant scorer's AP changed with the input permutation, and a
    display panel is exactly where tied scores occur.
    """
    from . import her2_nf_metrics as metrics
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    require(scores.shape == labels.shape, "Misaligned scores and labels")
    if not labels.any():
        return None
    return float(metrics.average_precision(scores, labels))


def paired_delta_spearman(scores_a, scores_b, outcomes, *, groups, seed, draws=2000,
                          confidence=0.95):
    """Bootstrap the DIFFERENCE OF SPEARMANS over resampled neighbourhoods.

    The primary reserved endpoint. Two things it is not, both of which were
    previously reported under this name:

    * it is not a bootstrap of the mean **log-score** difference. An arbitrary
      additive offset in one arm's scores changes that quantity and changes nothing
      about the ordering it claims to measure, so a shifted scorer passed;
    * it is not a resample of rows. The unit is a sequence neighbourhood, so linked
      contrasts and replicates travel together; resampling rows treats two members
      of one neighbourhood as two independent observations.

    Each draw resamples whole groups with replacement, recomputes both Spearmans on
    the resampled units, and takes the difference. Groups whose resample yields
    fewer than two distinct outcomes contribute nothing to that draw's correlation,
    which is why the usable-draw count is reported.
    """
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    y = np.asarray(outcomes, dtype=np.float64)
    labels = np.asarray([str(value) for value in groups])
    require(a.shape == b.shape == y.shape == labels.shape,
            "Misaligned scores, outcomes and neighbourhood labels")
    unique = sorted(set(labels.tolist()))
    require(len(unique) > 1,
            f"A paired delta-Spearman bootstrap needs more than one neighbourhood; the panel "
            f"carries {len(unique)}. One unit is not an interval.")
    members = {name: np.flatnonzero(labels == name) for name in unique}
    rng = np.random.default_rng(int(seed))
    observed = (spearman(a, y), spearman(b, y))
    samples = []
    for _ in range(int(draws)):
        picked = [unique[int(index)] for index in rng.integers(0, len(unique), len(unique))]
        rows = np.concatenate([members[name] for name in picked])
        left, right = spearman(a[rows], y[rows]), spearman(b[rows], y[rows])
        if left is None or right is None:
            continue
        samples.append(left - right)
    require(samples,
            "No bootstrap draw produced a computable pair of correlations. The panel's outcomes "
            "are constant within too many neighbourhoods for this endpoint to have an interval, "
            "and that is reported rather than approximated.")
    values = np.asarray(samples, dtype=np.float64)
    alpha = (1.0 - float(confidence)) / 2.0
    low, high = float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))
    delta = (None if observed[0] is None or observed[1] is None
             else float(observed[0] - observed[1]))
    return {"neighbourhoods": len(unique), "rows": int(a.size),
            "spearman_a": observed[0], "spearman_b": observed[1],
            "delta": delta, "interval": [low, high],
            "excludes_zero": bool(low > 0 or high < 0),
            "draws": int(draws), "usable_draws": int(values.size),
            "confidence": float(confidence),
            "statistic": "difference of Spearman correlations on the same measured outcomes",
            "unit": "sequence neighbourhood; linked contrasts and replicates stay together",
            "not_a_unit": ("a selection batch. Batch diameters were 6-10 edits; they are not "
                           "families and not bootstrap units."),
            "not_this": ("a bootstrap of the mean log-score difference. An arbitrary additive "
                         "offset in one arm's scores moves that quantity and moves no ordering."),
            "seed_variation_note": ("this interval is a within-seed neighbourhood bootstrap and is "
                                   "NOT a statement about training-seed variability; the per-seed "
                                   "table is reported separately")}


# ---------------------------------------------------------------------------
# frozen family clustering
# ---------------------------------------------------------------------------

@dataclass
class FrozenClustering:
    """One global assignment, built once at the maximum draw count and never rebuilt."""

    medoids: tuple
    radius: int
    separation: int
    built_from: dict
    frozen: bool = True
    assignments: dict = field(default_factory=dict)

    def assign(self, window):
        """Nearest medoid within ``radius``, deterministic ties, or ``None``."""
        best, chosen = None, None
        for index, medoid in enumerate(self.medoids):
            distance = hamming(window, medoid)
            if distance > int(self.radius):
                continue
            if best is None or distance < best or (distance == best and medoid < chosen):
                best, chosen = distance, medoid
        if chosen is None:
            return None
        return self.medoids.index(chosen)

    def occupancy(self, windows):
        """Occupancy by deletion-key lookup, not by a medoid scan per window.

        The scan is ``len(windows) * len(medoids)`` Hamming computations. At the
        plan's maximum draw count that is the operation this whole measurement is
        blocked on, and the hash form is exact for the declared radius.
        """
        from . import her2_routes_split as split_lib
        return split_lib.assign_to_medoids(list(windows), self.medoids,
                                           radius=int(self.radius))

    def document(self):
        return {"medoids": len(self.medoids), "radius": int(self.radius),
                "separation": int(self.separation), "built_from": dict(self.built_from),
                "frozen": True,
                "rule": ("built once from the pooled unassigned unique raw candidates of every "
                         "preregistered arm at the MAXIMUM draw count, sorted lexicographically, "
                         "greedily assigned at radius one, then reused for every arm and every "
                         "nested draw prefix. Rebuilding per sample count changes the measurement."),
                "meaning": ("operational sequence clusters. They are not demonstrated binding "
                            "mechanisms and no family-level biological claim rests on them.")}


def training_medoids(windows, *, separation=3):
    """Frozen training medoids separated by at least ``separation`` substitutions.

    The separation is a strict floor, so the greedy cover is at radius
    ``separation - 1``: a candidate is rejected when it sits **within** that radius
    of an accepted medoid, which is the same statement as "at least ``separation``
    edits away" and is answerable by hashing.
    """
    from . import her2_routes_split as split_lib
    fresh, _existing = split_lib.greedy_medoids(windows, radius=max(0, int(separation) - 1))
    return fresh


def freeze_clustering(arm_windows, *, radius=1, separation=3, assigned=()):
    """Build the one global clustering. ``arm_windows`` is ``{arm: windows}``.

    Must be called with **every** preregistered arm's candidates at the maximum draw
    count. Calling it with a subset silently changes the measurement for every arm,
    so the arms that contributed are recorded in ``built_from``.

    The greedy cover is computed by deletion-key hashing. The Python scan it replaces
    is quadratic in the pooled candidate count, and the pooled count at the plan's
    maximum draw is 100,000 per arm.
    """
    from . import her2_routes_split as split_lib
    require(arm_windows, "A frozen clustering needs at least one arm's candidates")
    existing = set(str(window) for window in assigned)
    pooled = sorted({str(window) for windows in dict(arm_windows).values()
                     for window in windows} - existing)
    medoids, prior = split_lib.greedy_medoids(pooled, radius=int(radius), existing=existing)
    return FrozenClustering(
        medoids=tuple(prior) + tuple(medoids), radius=int(radius), separation=int(separation),
        built_from={"arms": sorted(dict(arm_windows)),
                    "candidates_per_arm": {arm: len(set(windows)) for arm, windows in
                                           sorted(dict(arm_windows).items())},
                    "pooled_unique_unassigned": len(pooled),
                    "pre_assigned_excluded": len(existing),
                    "known_training_medoids": len(prior),
                    "new_candidate_medoids": len(medoids),
                    "method": ("greedy cover by deletion-key hashing, exact at the declared "
                               "radius. The pairwise scan is quadratic in the pooled count.")})


# ---------------------------------------------------------------------------
# raw generation
# ---------------------------------------------------------------------------

def generation_report(windows, *, training, catalogue, clustering, wild_type, draw_prefixes,
                      mode="full_mask", nearest_radius=4):
    """Novelty, duplicates, recovery, occupancy and nearest-adaptation, at each prefix.

    Nested prefixes of one draw sequence, so the curves are comparable across arms at
    shared reported counts. An off-catalogue sample is an in-silico candidate and is
    labelled one: nothing generated here has been experimentally measured.
    """
    from . import her2_routes_split as split_lib
    windows = [str(window) for window in windows]
    training = set(str(window) for window in training)
    catalogue = set(str(window) for window in catalogue)
    # Nearest-training distance by the packed deletion-mask search, bounded at the
    # declared radius. The Python scan is len(unique) x len(training) Hamming
    # computations, and the training library here is the 367,042-row display split.
    rows = []
    for prefix in sorted(int(value) for value in draw_prefixes):
        require(prefix <= len(windows),
                f"A {prefix}-draw prefix was requested and only {len(windows)} draws exist. The "
                "shorter count is reported as the measured one rather than padded.")
        sample = windows[:prefix]
        unique = sorted(set(sample))
        novel = [window for window in unique if window not in training]
        nearest = (split_lib.min_distance_to_library(unique, sorted(training),
                                                     max_distance=int(nearest_radius))
                   if training and unique else np.full(len(unique), -1, dtype=np.int64))
        within = [int(value) for value in nearest if int(value) >= 0]
        rows.append({
            "draws": prefix, "unique": len(unique),
            "exact_duplicates": prefix - len(unique),
            "duplicate_rate": None if prefix == 0 else (prefix - len(unique)) / prefix,
            "novel_unique": len(novel),
            "novelty_rate": None if not unique else len(novel) / len(unique),
            "catalogue_recovered": len(set(unique) & catalogue),
            "catalogue_recovery_rate": (None if not catalogue
                                        else len(set(unique) & catalogue) / len(catalogue)),
            "within_nearest_radius": len(within),
            "beyond_nearest_radius": len(unique) - len(within),
            "nearest_radius": int(nearest_radius),
            "mean_nearest_training_distance": (float(np.mean(within)) if within else None),
            "nearest_distance_note": ("the mean is over the unique draws WITHIN the declared "
                                      "radius. A draw beyond it has no bounded distance and is "
                                      "counted separately rather than folded in at the radius."),
            "mean_wild_type_distance": (None if not unique else
                                        float(np.mean([hamming(w, wild_type) for w in unique]))),
            "occupancy": clustering.occupancy(unique)})
    return {"mode": mode, "rows": rows, "total_draws": len(windows),
            "labels": ("off-catalogue samples are in-silico candidates. Closed-catalogue recovery "
                       "measures allocation among known sequences and is not novel-design success. "
                       "Nothing generated by this project has been experimentally assayed.")}


def _finite_or_none(value):
    """``float(value)`` when it is a finite number, else ``None``. Never a stand-in zero."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def learning_curve(points, *, scope=None):
    """E13: development score at each nested data fraction, against distinct measured summaries.

    ``points`` maps a fraction to that fit's block (``score`` and the accounting beside it). A
    missing or non-finite score stays missing: it is listed under ``unscored_fractions`` and is
    never read as ``0``, which would turn a failed fit into a "decrease". Every returned row
    carries the SAME value the ordering used -- a finite float, or ``None`` for an unscored
    fraction -- so a ``NaN``/``inf`` never survives into ``rows[*].score`` and the block is strict
    JSON; the caller's own block is not rewritten. ``monotone`` is ``None``
    -- with the reason and the counts -- when fewer than two fractions carry a finite score, and a
    bool over the SCORED fractions otherwise. It is a statement about single-fit point estimates
    (no interval, one seed), so ``monotone is False`` is a description of the points and is not,
    by itself, evidence of a capacity limit.

    ``scope`` is the caller's account of what supervision the curve varies; it is carried through
    verbatim so the curve cannot be read without it.
    """
    entries = {}
    for fraction, block in dict(points).items():
        value = _finite_or_none(fraction)
        require(value is not None and 0.0 < value <= 1.0,
                f"a learning-curve fraction must be a finite number in (0, 1], got {fraction!r}")
        require(value not in entries, f"the learning-curve fraction {value!r} appears twice")
        entries[value] = dict(block)
    rows = [dict(block, fraction=fraction, score=_finite_or_none(block.get("score")))
            for fraction, block in sorted(entries.items())]
    scored = [(row["fraction"], row["score"]) for row in rows if row["score"] is not None]
    unscored = [row["fraction"] for row in rows if row["score"] is None]
    if len(scored) < 2:
        monotone = None
        reason = (f"{len(scored)} of {len(rows)} fraction(s) carry a finite score; an ordering "
                  "needs at least two, so monotonicity is unknown (not false)")
        max_decrease = None
    else:
        decreases = [earlier - later for (_, earlier), (_, later) in zip(scored, scored[1:])
                     if earlier > later]
        monotone = not decreases
        reason = None
        max_decrease = float(max(decreases)) if decreases else 0.0
    return {"fractions": [row["fraction"] for row in rows],
            "rows": rows,
            "scored_points": len(scored), "unscored_fractions": unscored,
            "monotone": monotone, "monotone_reason": reason,
            "monotone_basis": ("non-decreasing across the SCORED fractions in increasing order, "
                               "on single-fit point estimates; no interval and one seed, so a "
                               "decrease is not shown to exceed noise"),
            "largest_decrease": max_decrease,
            "x_axis": ("distinct measured POPULATION KEYS (study, assay, target, full construct) "
                       "-- the unit one acquired summary is; not parent families and not "
                       "replicate rows. Constructs, parents, replicate observations and the "
                       "ledger's presentations are reported beside it, separately."),
            "scope": scope,
            "protocol": "each subset is trained to its own best development result"}


def resolve_readout_tolerances(*, tolerance=None, affinity_tolerance=None,
                               expression_tolerance=None, units=None):
    """The two matching tolerances, validated, with where they came from and what they are in.

    Affinity (pKD units) and expression (a study-internal enrichment ratio) are numbers on
    different scales, so each has its own tolerance, finite and non-negative. The legacy single
    ``tolerance`` is still accepted and is broadcast to both, but the record says so: one number
    applied to two readouts does not make their units common.
    """
    named = (affinity_tolerance, expression_tolerance)
    if tolerance is not None:
        require(all(item is None for item in named),
                "give either the named affinity and expression tolerances or the legacy single "
                "tolerance, not both")
        affinity = expression = _nonnegative(tolerance, "the readout tolerance")
        source = "legacy_broadcast"
        note = ("the single legacy tolerance was broadcast to BOTH readouts as a declared "
                "convenience. The two readouts are in different units, so equal numbers are not "
                "equal tolerances in any physical sense")
    else:
        require(all(item is not None for item in named),
                "E16 needs both affinity_tolerance and expression_tolerance (separate units), or "
                "the legacy single tolerance to broadcast explicitly")
        affinity = _nonnegative(affinity_tolerance, "affinity_tolerance")
        expression = _nonnegative(expression_tolerance, "expression_tolerance")
        source = "named"
        note = "each readout's tolerance is its own, in its own units"
    given = dict(units or {})
    return {"affinity": affinity, "expression": expression, "source": source, "note": note,
            "units": {"affinity": str(given.get("affinity", "unspecified")),
                      "expression": str(given.get("expression", "unspecified"))}}


def _nonnegative(value, name):
    require(not isinstance(value, bool), f"{name} must be a number, got {value!r}")
    number = finite_scalar(value, name)
    require(number >= 0.0, f"{name} must be non-negative, got {number:g}")
    return number


def readout_separation(pairs, *, tolerance=None, affinity_tolerance=None,
                       expression_tolerance=None, units=None):
    """E16: affinity among expression-matched variants, and the converse. A DATA summary.

    Each pair carries two constructs' exact affinity and exact expression values from ONE
    declared assay pair. Matching is within each readout's own tolerance
    (:func:`resolve_readout_tolerances`), and the residual imbalance after approximate matching is
    reported -- matching eliminates differences only in the matched feature. The other readout's
    spread inside the matched set is reported too, since that spread is what the panel is about.

    No model prediction enters: the counts describe the measured data and say nothing about a
    model's ability to separate affinity from expression.
    """
    resolved = resolve_readout_tolerances(
        tolerance=tolerance, affinity_tolerance=affinity_tolerance,
        expression_tolerance=expression_tolerance, units=units)
    expression_matched, affinity_matched = [], []
    for pair in pairs:
        if abs(float(pair["expression_left"]) - float(pair["expression_right"])) \
                <= resolved["expression"]:
            expression_matched.append(pair)
        if abs(float(pair["affinity_left"]) - float(pair["affinity_right"])) \
                <= resolved["affinity"]:
            affinity_matched.append(pair)

    def residual(group, field):
        if not group:
            return None
        return float(np.mean([abs(float(p[f"{field}_left"]) - float(p[f"{field}_right"]))
                              for p in group]))
    block = {"expression_matched_pairs": len(expression_matched),
             "affinity_matched_pairs": len(affinity_matched),
             "affinity_tolerance": resolved["affinity"],
             "expression_tolerance": resolved["expression"],
             "tolerance_units": dict(resolved["units"]),
             "tolerance_source": resolved["source"], "tolerance_note": resolved["note"],
             "residual_expression_imbalance": residual(expression_matched, "expression"),
             "residual_affinity_imbalance": residual(affinity_matched, "affinity"),
             "affinity_spread_among_expression_matched": residual(expression_matched, "affinity"),
             "expression_spread_among_affinity_matched": residual(affinity_matched, "expression"),
             "matching": ("approximate, so the residual imbalance above is reported. Exact matching "
                          "would eliminate differences only in the matched features."),
             "thresholds": "from training and development only"}
    if resolved["source"] == "legacy_broadcast":
        block["tolerance"] = resolved["affinity"]
    return block


def gate_decision(*, route, ranking, generation, backbone_ranking, distance_ranking,
                  comparators, delta, margins):
    """G1: the predeclared Pareto criterion. Not 'beat every comparator'.

    A route advances only if it beats (a) its own frozen zero-shot backbone and
    (b) the distance-only control, on a ranking endpoint **and** a raw-generation
    endpoint, and is not dominated by more than ``delta`` by the best discriminative
    comparator. "Beat every discriminative comparator simultaneously" is the wrong
    gate for a generator: a CNN can out-rank a decoder without the decoder being bad.
    """
    who = f"gate_decision[{route}]"
    ranking = finite_scalar(ranking, f"{who} ranking")
    generation = finite_scalar(generation, f"{who} generation")
    backbone_ranking = finite_scalar(backbone_ranking, f"{who} backbone_ranking")
    distance_ranking = finite_scalar(distance_ranking, f"{who} distance_ranking")
    delta = finite_scalar(delta, f"{who} delta")
    margins = dict(margins)
    for key in ("backbone", "distance", "generation"):
        require(key in margins, f"{who} margins is missing {key!r}")
    margins = {k: finite_scalar(v, f"{who} margin {k!r}") for k, v in margins.items()}
    comparators = {name: finite_scalar(value, f"{who} comparator {name!r}")
                   for name, value in dict(comparators).items()}
    best_comparator = max(comparators.values(), default=float("-inf"))
    checks = {
        "beats_own_backbone": ranking - backbone_ranking >= margins["backbone"],
        "beats_distance_only": ranking - distance_ranking >= margins["distance"],
        "raw_generation_improves": generation >= margins["generation"],
        "not_dominated_beyond_delta": (best_comparator == float("-inf")
                                       or ranking >= best_comparator - delta),
    }
    return {"route": route, "checks": checks, "advances": all(checks.values()),
            "ranking": float(ranking), "generation": float(generation),
            "backbone_ranking": float(backbone_ranking),
            "distance_only_ranking": float(distance_ranking),
            "best_discriminative_comparator": (None if best_comparator == float("-inf")
                                               else best_comparator),
            "delta": float(delta), "margins": {k: float(v) for k, v in dict(margins).items()},
            "criterion": ("Pareto, predeclared before the bake-off. A discriminative comparator "
                          "out-ranking the decoder does not fail the gate by itself, because a "
                          "classifier has no raw-generation distribution to compare and a decoder "
                          "is not required to win a ranking-only contest."),
            "spr152": "reported beside this decision and never gated on"}
