"""T2 controls and the distance-shortcut harness (E17, E19).

These are the panels that decide whether a measured gain is a measured gain. They
matter more here than the plan implies, because on the 152-row SPR cohort
**-Hamming distance to wild type alone reaches Spearman 0.4312 [0.3000, 0.5609]
against the parent's 0.5634** -- most of the signal on that cohort is distance. So
E19 is a gate, not a panel, and G2 can end the campaign with a reported artefact,
which is a legitimate outcome.

The five matched-exposure controls, and what each one removes:

* ``reconstruction_only`` -- identical sequence exposure, every measured functional
  loss removed **including head supervision**. Isolates "did the functional signal
  do anything at all".
* ``head_only`` -- the observation heads train; the decoder's functional objectives
  do not. A better head is not a better decoder, and this arm is what makes that
  separable rather than asserted.
* ``outcome_permuted`` -- outcomes shuffled **within** assay and distance strata,
  so the marginal distribution of outcomes and its relationship to distance both
  survive and only the sequence-outcome correspondence is destroyed.
* ``background_permuted`` -- outcomes shuffled within consistently oriented
  substitution groups, so the substitution marginals survive and only the
  background dependence is destroyed.
* ``local_only`` -- local effects without the whole-window task, which is E21.

:func:`permutation_preserved_marginals` reports which marginals each permutation
actually preserved, because a permutation that destroyed more than intended makes
the control easier to beat and the gain look real.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from .her2_runtime import require

ARM_FULL = "full"
ARM_RECONSTRUCTION_ONLY = "reconstruction_only"
ARM_HEAD_ONLY = "head_only"
ARM_OUTCOME_PERMUTED = "outcome_permuted"
ARM_BACKGROUND_PERMUTED = "background_permuted"
ARM_LOCAL_ONLY = "local_only"
ARM_NO_FUNCTIONAL_LOSS = "no_functional_loss"
CONTROL_ARMS = (ARM_RECONSTRUCTION_ONLY, ARM_HEAD_ONLY, ARM_OUTCOME_PERMUTED,
                ARM_BACKGROUND_PERMUTED, ARM_LOCAL_ONLY, ARM_NO_FUNCTIONAL_LOSS)

#: Which loss terms each arm may switch on. The coefficients the trainer reads.
ARM_TERMS = {
    ARM_FULL: ("reconstruction", "local_effect", "whole_window", "heads"),
    ARM_RECONSTRUCTION_ONLY: ("reconstruction",),
    ARM_NO_FUNCTIONAL_LOSS: ("reconstruction",),
    ARM_HEAD_ONLY: ("reconstruction", "heads"),
    ARM_OUTCOME_PERMUTED: ("reconstruction", "local_effect", "whole_window", "heads"),
    ARM_BACKGROUND_PERMUTED: ("reconstruction", "local_effect", "whole_window", "heads"),
    ARM_LOCAL_ONLY: ("reconstruction", "local_effect", "heads"),
}


def arm_coefficients(arm, *, base):
    """Zero every term the arm excludes, keeping the rest at their declared values."""
    require(arm in ARM_TERMS, f"Unknown control arm {arm!r}")
    allowed = set(ARM_TERMS[arm])
    return {name: (float(value) if name in allowed else 0.0)
            for name, value in dict(base).items()}


def require_matched_exposure(ledgers, *, label="T2 controls", tolerance=0):
    """Every arm must have seen the same number of row presentations.

    A control that saw fewer rows is not a control: the comparison would confound
    the removed loss with a shorter schedule. The tolerance is zero by default.
    """
    rows = {name: int(dict(block)["totals"]["microbatch_rows"])
            for name, block in dict(ledgers).items()}
    require(rows, f"{label}: no ledgers to compare")
    spread = max(rows.values()) - min(rows.values())
    require(spread <= int(tolerance),
            f"{label}: row presentations differ by {spread} across arms {rows}. A control that saw "
            "fewer rows confounds the removed loss with a shorter schedule.")
    forwards = {name: int(dict(block)["totals"]["student_forwards"])
                for name, block in dict(ledgers).items()}
    return {"microbatch_rows": rows, "student_forwards": forwards, "row_spread": int(spread),
            "forward_spread": int(max(forwards.values()) - min(forwards.values())),
            "forward_note": ("student forwards legitimately DIFFER between arms: an arm without "
                             "local contrasts makes fewer two-forward calls. Rows are what is "
                             "matched; forwards are reported so the compute difference is visible "
                             "rather than hidden inside a matched-exposure claim.")}


# ---------------------------------------------------------------------------
# permutations
# ---------------------------------------------------------------------------

def distance_stratum(distance, *, edges=(0, 1, 2, 3)):
    """A coarse distance label. Strata are declared, not fitted to the data."""
    value = int(distance)
    for edge in edges:
        if value <= edge:
            return str(edge)
    return f">{edges[-1]}"


def permute_within_strata(values, strata, *, seed):
    """Shuffle ``values`` inside each stratum. Returns the permuted array.

    Within-stratum permutation is what preserves the marginal of the outcome and its
    relationship to the stratum variable while destroying the sequence-outcome
    correspondence. A global shuffle would destroy the distance relationship too,
    and an arm that beat *that* control would only have learned about distance.
    """
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray([str(label) for label in strata])
    require(values.shape[0] == labels.shape[0], "One stratum label per value")
    out = values.copy()
    rng = np.random.default_rng(int(seed))
    for label in sorted(set(labels.tolist())):
        rows = np.flatnonzero(labels == label)
        if rows.size < 2:
            continue
        out[rows] = values[rows][rng.permutation(rows.size)]
    return out


def permutation_preserved_marginals(original, permuted, strata, *, tolerance=1e-9):
    """Which marginals survived the permutation? Reported, not assumed."""
    original = np.asarray(original, dtype=np.float64)
    permuted = np.asarray(permuted, dtype=np.float64)
    labels = np.asarray([str(label) for label in strata])
    global_ok = bool(np.allclose(np.sort(original), np.sort(permuted), atol=tolerance))
    per_stratum = {}
    for label in sorted(set(labels.tolist())):
        rows = np.flatnonzero(labels == label)
        per_stratum[label] = {
            "n": int(rows.size),
            "multiset_preserved": bool(np.allclose(np.sort(original[rows]),
                                                   np.sort(permuted[rows]), atol=tolerance)),
            "mean_preserved": bool(abs(float(original[rows].mean() - permuted[rows].mean()))
                                   <= tolerance)}
    moved = int((original != permuted).sum())
    return {"global_multiset_preserved": global_ok, "per_stratum": per_stratum,
            "values_moved": moved, "values": int(original.size),
            "fraction_moved": None if original.size == 0 else moved / original.size,
            "destroyed": "the sequence-to-outcome correspondence within each stratum",
            "preserved": ("the outcome marginal overall and within every stratum, so an arm cannot "
                          "beat this control by learning the stratum variable")}


def oriented_substitution_group(left_window, right_window):
    """The DIRECTED substitution key, or ``None`` for a multi-site pair.

    ``(position, from_residue, to_residue)`` in the order the arguments were given, so
    swapping them gives the reverse key. That is deliberate: the direction is the
    thing E5 measures, and a pair seen in both directions has to produce two distinct
    keys for the both-directions requirement to mean anything.

    For *grouping*, use :func:`canonical_substitution_group`: a group key that depends
    on argument order puts the same unordered substitution in two groups depending on
    which member the enumeration read first, and then neither group has both
    directions.
    """
    require(len(left_window) == len(right_window), "Windows of different length")
    differing = [position for position, (a, b) in enumerate(zip(left_window, right_window))
                 if a != b]
    if len(differing) != 1:
        return None
    position = differing[0]
    return (position, left_window[position], right_window[position])


def permute_background_effects(contrasts, *, seed):
    """Move measured effect bundles between backgrounds of the same directed edit.

    Constructs, exposure and reconstruction targets stay in place. Only the local
    functional targets move; whole-region and head targets retain their measurements.
    A quartet's three effects and interaction move together to preserve algebra.
    """
    contrasts = tuple(contrasts)
    groups = {}
    for index, contrast in enumerate(contrasts):
        sites = tuple(contrast.sites)
        signature = tuple(tuple(row.construct.residue_at(site) for site in sites)
                          for row in contrast.alternatives())
        key = (contrast.assay, contrast.target, contrast.left.parent, contrast.kind,
               sites, signature, contrast.scale)
        groups.setdefault(key, []).append(index)
    rng = np.random.default_rng(int(seed))
    output = list(contrasts)
    fields = ("left_value", "right_value", "margin", "interaction", "double_effect",
              "effect_left", "effect_right")
    moved = 0
    supported = 0
    for indices in groups.values():
        if len(indices) < 2:
            continue
        supported += len(indices)
        donors = rng.permutation(indices)
        for index, donor in zip(indices, donors):
            values = {field: getattr(contrasts[int(donor)], field) for field in fields}
            output[index] = replace(contrasts[index], **values)
            moved += int(any(getattr(contrasts[index], name) != value
                             for name, value in values.items()))
    return tuple(output), {
        "contrasts": len(contrasts), "strata": len(groups),
        "contrasts_in_permutable_strata": supported, "effect_bundles_moved": moved,
        "preserved": "the complete effect-bundle multiset in each directed-edit stratum",
        "changed": "local effect association with the visible background",
        "unchanged": "constructs, head outcomes, whole-region conditions and reconstruction",
    }


def canonical_substitution_group(left_window, right_window):
    """The orientation-independent substitution key, or ``None``.

    ``(position, lower_residue, higher_residue)`` -- the same key whichever member is
    passed first. This is what a group key has to be: the directed key puts one
    unordered substitution in two groups depending on read order, and
    :func:`matched_reversals` then finds no group with both directions in it even when
    every pair was observed twice.
    """
    directed = oriented_substitution_group(left_window, right_window)
    if directed is None:
        return None
    position, first, second = directed
    return (position, min(first, second), max(first, second))


# ---------------------------------------------------------------------------
# E19: distance shortcuts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DistanceProfile:
    """The declared distance summaries every candidate carries."""

    to_wild_type: int
    to_nearest_training: int
    to_nearest_by_group: dict
    neighbourhood_density: int

    def document(self):
        return {"to_wild_type": int(self.to_wild_type),
                "to_nearest_training": int(self.to_nearest_training),
                "to_nearest_by_group": {k: int(v) for k, v in
                                        sorted(self.to_nearest_by_group.items())},
                "neighbourhood_density": int(self.neighbourhood_density)}


def hamming(left, right):
    require(len(left) == len(right), "Windows of different length")
    return sum(1 for a, b in zip(left, right) if a != b)


def distance_profile(window, *, wild_type, bank, groups=None, radius=2):
    """Distances to wild type, to the nearest bank member, per group, plus density.

    ``bank`` is ONE frozen eligible measured-training bank, shared across arms, so
    the primary comparison is not confounded by each arm's own exposures. Distance
    to an arm's actual adaptation exposures is reported separately.
    """
    bank = list(bank)
    require(bank, "An empty distance bank cannot produce a nearest neighbour")
    distances = [hamming(window, member) for member in bank]
    by_group = {}
    for name, members in dict(groups or {}).items():
        if members:
            by_group[name] = min(hamming(window, member) for member in members)
    return DistanceProfile(to_wild_type=hamming(window, wild_type),
                           to_nearest_training=min(distances),
                           to_nearest_by_group=by_group,
                           neighbourhood_density=sum(1 for d in distances if d <= int(radius)))


def out_of_fold_nearest(query, *, bank, values, exclude_self=True):
    """Nearest-neighbour outcome prediction with the query excluded from its own bank.

    Without the exclusion a training query retrieves itself at distance zero and the
    control scores near-perfectly, which makes it useless as a control and makes the
    model's own gain look larger by comparison.
    """
    best, chosen = None, []
    for member, value in zip(bank, values):
        if exclude_self and member == query:
            continue
        distance = hamming(query, member)
        if best is None or distance < best:
            best, chosen = distance, [float(value)]
        elif distance == best:
            chosen.append(float(value))
    if best is None:
        return {"distance": None, "prediction": None, "ties": 0,
                "reason": "the bank held nothing but the query itself"}
    return {"distance": int(best), "prediction": float(np.mean(chosen)), "ties": len(chosen)}


def nearest_hamming(queries, bank, *, budget=16_000_000):
    """Exact Hamming distance from each query to its nearest ``bank`` member. Chunked.

    Always a real distance: there is no radius, so there is no "beyond the radius" sentinel to be
    mistaken for one (``min_distance``-style ``-1`` answers mean "farther than the radius", which
    is not a distance). The bank is deduplicated and processed in chunks sized so that one
    ``queries x chunk x width`` comparison never exceeds ``budget`` bytes. All windows must have
    one length; an empty bank has no nearest neighbour and is refused.
    """
    queries = [str(query) for query in queries]
    members = sorted({str(member) for member in bank})
    require(members, "An empty bank has no nearest neighbour; refusing to invent a distance")
    width = len(members[0])
    require(all(len(member) == width for member in members),
            "The bank mixes window lengths, so a Hamming distance is undefined")
    require(all(len(query) == width for query in queries),
            f"A query is not {width} residues long, the bank's window length")
    if not queries:
        return []
    asked = np.frombuffer("".join(queries).encode("ascii"), dtype=np.uint8).reshape(
        len(queries), width)
    held = np.frombuffer("".join(members).encode("ascii"), dtype=np.uint8).reshape(
        len(members), width)
    best = np.full(len(queries), width + 1, dtype=np.int64)
    step = max(1, int(budget) // max(1, len(queries) * max(1, width)))
    for start in range(0, len(members), step):
        block = held[start:start + step]
        distance = (asked[:, None, :] != block[None, :, :]).sum(axis=2)
        best = np.minimum(best, distance.min(axis=1))
    return [int(value) for value in best]


def joint_distance_strata(wild_type_distances, nearest_distances, *, edges=None):
    """``(labels, scheme)``: joint wild-type x nearest-adaptation distance strata.

    A distance-only rule is constant within a stratum, so the strata are what the shortcut check
    holds fixed: a model that scores well inside them knows something distance does not. They are
    JOINT -- a window's wild-type distance AND its distance to the nearest adaptation sequence.
    With ``edges=None`` both are exact integers, which coarsens nothing; with ``edges`` each
    distance is binned by :func:`distance_stratum` and the scheme says so, so a coarsening is
    never silent.
    """
    wild_type_distances = [int(value) for value in wild_type_distances]
    nearest_distances = [int(value) for value in nearest_distances]
    require(len(wild_type_distances) == len(nearest_distances),
            "One nearest-adaptation distance is required per wild-type distance")

    def name(value):
        return str(value) if edges is None else distance_stratum(value, edges=tuple(edges))
    labels = [f"wt={name(a)}|adaptation={name(b)}"
              for a, b in zip(wild_type_distances, nearest_distances)]
    return labels, {
        "axes": ["distance to wild type", "distance to the nearest eligible adaptation sequence"],
        "binning": ("exact integer distances on both axes; nothing is pooled" if edges is None
                    else f"declared edges {list(edges)} on both axes; larger distances pooled"),
        "exact": edges is None}


def _is_number(value):
    """A real, non-boolean number (bool is an ``int`` subclass and is not a measurement)."""
    return (isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_)))


def score_tolerance(tolerance):
    """The score-tie tolerance as a float, refusing anything but a finite non-negative number.

    A NaN or negative tolerance makes ``abs(difference) <= tolerance`` False even for identical
    scores, so a constant predictor would be scored 0 instead of the half credit a tie earns --
    a fabricated loss, not a declared rule.
    """
    require(_is_number(tolerance) and math.isfinite(float(tolerance)) and float(tolerance) >= 0.0,
            f"The score tolerance must be a finite non-negative number, got {tolerance!r}")
    return float(tolerance)


def stratified_concordance(scores, outcomes, strata, *, tolerance=0.0):
    """Pair concordance within each declared distance stratum, macro-averaged.

    Pooling strata is how a model that only knows distance scores well: the pooled
    concordance is dominated by between-stratum comparisons it gets right for the
    wrong reason.

    A pair measured as a tie is excluded and counted. A pair the SCORE ties (``|difference| <=
    tolerance``) earns half credit: a rule that is constant within a stratum -- which is exactly a
    distance-only rule inside a distance stratum -- scores 0.5 there, not 0, and counting its ties
    as misses would add a fabricated half point to the model beside it. A stratum with no
    comparable pair has no concordance (``None``), never zero.
    """
    tolerance = score_tolerance(tolerance)
    scores = np.asarray(scores, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64)
    labels = np.asarray([str(label) for label in strata])
    require(scores.shape == outcomes.shape == labels.shape, "Misaligned inputs")
    require(bool(np.isfinite(scores).all() and np.isfinite(outcomes).all()),
            "A non-finite score or outcome has no ordering; refusing to count it as a tie")
    per_stratum = {}
    pooled_pairs, pooled_credit, pooled_score_ties, pooled_measured_ties = 0, 0.0, 0, 0
    for label in sorted(set(labels.tolist())):
        rows = np.flatnonzero(labels == label)
        pairs = score_ties = measured_ties = 0
        credit = 0.0
        for i in range(rows.size):
            for j in range(i + 1, rows.size):
                a, b = rows[i], rows[j]
                if outcomes[a] == outcomes[b]:
                    measured_ties += 1
                    continue
                pairs += 1
                difference = scores[a] - scores[b]
                if abs(difference) <= tolerance:
                    credit += 0.5
                    score_ties += 1
                elif difference * (outcomes[a] - outcomes[b]) > 0:
                    credit += 1.0
        per_stratum[label] = {"pairs": pairs, "n": int(rows.size), "score_ties": score_ties,
                              "measured_ties_excluded": measured_ties,
                              "concordance": None if pairs == 0 else credit / pairs}
        pooled_pairs += pairs
        pooled_credit += credit
        pooled_score_ties += score_ties
        pooled_measured_ties += measured_ties
    usable = [block["concordance"] for block in per_stratum.values()
              if block["concordance"] is not None]
    return {"per_stratum": per_stratum,
            "macro_average": None if not usable else float(np.mean(usable)),
            "within_stratum_pairs": pooled_pairs,
            "within_stratum_concordance": (None if pooled_pairs == 0
                                           else pooled_credit / pooled_pairs),
            "score_ties": pooled_score_ties,
            "measured_ties_excluded": pooled_measured_ties,
            "strata_with_support": len(usable), "strata": len(per_stratum),
            "reason": (None if usable else
                       "no stratum holds a pair with different measured outcomes, so there is no "
                       "comparison to score; this is missing support, not a concordance of zero"),
            "tie_rule": ("a score tie earns half credit, a measured tie is excluded and counted, "
                         "and a stratum with no comparable pair has no concordance"),
            "tolerance": tolerance,
            "note": ("macro-averaging over distance strata is not averaging over classes, and the "
                     "pooled figure is deliberately absent: it is dominated by between-stratum "
                     "comparisons a distance-only rule also gets right.")}


def _profile_key(profile):
    """A hashable, order-independent form of a distance-profile dict."""
    return tuple(sorted((str(name), int(value)) for name, value in dict(profile).items()))


def canonical_reversal_pair(pair):
    """One substitution pair in canonical orientation, or ``None`` for a multi-site pair.

    The canonical LEFT window carries the lexicographically lower residue at the differing
    position. A pair supplied the other way round is reoriented as a whole: the two windows swap,
    the two endpoint profiles swap WITH them, and the signed outcome and score deltas change sign.
    Canonicalising the substitution key alone (the earlier behaviour) kept every sign as
    supplied, so ``AA -> CA`` (``A`` better) and ``CC -> AC`` (``A`` better again) looked like a
    pair seen in opposite directions, and a pure position effect produced a perfect "reversal".
    """
    left, right = pair["left_window"], pair["right_window"]
    directed = oriented_substitution_group(left, right)
    if directed is None:
        return None
    position, from_residue, to_residue = directed
    outcome, score = float(pair["outcome_delta"]), float(pair["score_delta"])
    require(math.isfinite(outcome) and math.isfinite(score),
            f"The pair {left}/{right} carries a non-finite outcome or score delta")
    left_profile, right_profile = pair["left_profile"], pair["right_profile"]
    if from_residue > to_residue:
        left, right = right, left
        left_profile, right_profile = right_profile, left_profile
        outcome, score = -outcome, -score
    return {"substitution": (position, min(from_residue, to_residue),
                             max(from_residue, to_residue)),
            "left_window": left, "right_window": right,
            "left_profile": _profile_key(left_profile),
            "right_profile": _profile_key(right_profile),
            "outcome_delta": outcome, "score_delta": score}


def matched_reversals(pairs, *, tolerance=0.0):
    """E5/E19's reversal test: oriented substitution x both endpoints' distance profiles.

    Every pair is first put in canonical orientation (:func:`canonical_reversal_pair`), so the
    direction of a pair is the sign of its CANONICAL outcome delta -- whether the lower-residue
    side measured higher -- and the input order of its endpoints changes nothing. Unordered physical
    pairs are deduplicated (a pair supplied twice with different measurements is refused) and
    measured ties are excluded and counted.

    Within each exact group (substitution and the two canonical endpoints' profiles) seen in
    both directions, accuracy is averaged **per direction**, the two directions are averaged, and
    then groups are macro-averaged. A rule using only the matched features has a 0.5 ceiling under
    this grouping; matching starting backgrounds alone, or balancing only substitutions, does
    not give that guarantee. Score ties get half credit under the declared tolerance. A group
    whose pairs all point the same way -- a pure position effect -- is not a reversal.
    """
    tolerance = score_tolerance(tolerance)
    seen, groups = {}, {}
    supplied = ties = duplicates = multi_site = 0
    for pair in pairs:
        supplied += 1
        canonical = canonical_reversal_pair(pair)
        if canonical is None:
            multi_site += 1
            continue
        physical = (canonical["left_window"], canonical["right_window"])
        if physical in seen:
            earlier = seen[physical]
            # EXACT agreement, not a tolerance: a measured tie (``== 0``) and a sign are exact
            # classes below, so two copies of one physical pair that differ by 5e-13 around zero
            # are a tie in one order and a direction in the other. Negation under
            # canonicalisation is exact in floating point, so a genuine repeat always agrees.
            same = (earlier["left_profile"] == canonical["left_profile"]
                    and earlier["right_profile"] == canonical["right_profile"]
                    and earlier["outcome_delta"] == canonical["outcome_delta"]
                    and earlier["score_delta"] == canonical["score_delta"])
            require(same,
                    f"The physical pair {physical[0]}/{physical[1]} was supplied twice with "
                    "different measurements, scores or profiles (compared exactly: a tie and a "
                    "sign are exact classes). One pair has one measurement; it cannot stand in "
                    "two directions.")
            duplicates += 1
            continue
        seen[physical] = canonical
        outcome = canonical["outcome_delta"]
        if outcome == 0:
            ties += 1
            continue
        key = (canonical["substitution"], canonical["left_profile"], canonical["right_profile"])
        direction = "forward" if outcome > 0 else "reverse"
        margin = canonical["score_delta"]
        correct = (0.5 if abs(margin) <= tolerance
                   else float((margin > 0) == (outcome > 0)))
        groups.setdefault(key, {"forward": [], "reverse": []})[direction].append(correct)
    both = {key: block for key, block in groups.items() if block["forward"] and block["reverse"]}
    per_group = [0.5 * (float(np.mean(block["forward"])) + float(np.mean(block["reverse"])))
                 for block in both.values()]
    return {"pairs_supplied": supplied,
            "pairs_used": sum(len(block["forward"]) + len(block["reverse"])
                              for block in groups.values()),
            "measured_ties_excluded": ties, "duplicate_physical_pairs": duplicates,
            "multi_site_pairs_skipped": multi_site,
            "groups_seen": len(groups), "groups_in_both_directions": len(both),
            "participating_comparisons": sum(len(block["forward"]) + len(block["reverse"])
                                             for block in both.values()),
            "macro_accuracy": None if not per_group else float(np.mean(per_group)),
            "support_gated": len(both) == 0,
            "reason": (None if both else
                       "no canonical substitution group with matched endpoint profiles was "
                       "measured in both directions, so the ceiling argument has nothing to "
                       "test; this is missing support, not an accuracy of zero"),
            "orientation": ("canonical: the left window carries the lower residue, and a pair "
                            "supplied the other way round swaps its windows, its profiles and "
                            "the signs of both deltas together"),
            "ceiling_claim": ("a rule using only the matched features has a 0.5 ceiling under this "
                              "exact grouping. Matching starting backgrounds alone does not give "
                              "that guarantee."),
            "her2_caveat": ("sequence-only audits found 0-3 repeated development groups and 2-5 "
                            "repeated test groups on the strict HER2 panel, with 0-7 and 4-11 "
                            "participating comparisons. Treat HER2 reversals as exploratory and "
                            "support-gated; CR9114 carries the stronger capability test."),
            "tolerance": tolerance}


def control_gate(rankings, *, required, margin, e19):
    """``(failed, reasons)``: which required gates the controls stage fails. Missing is failure.

    A gate PASSES only on an affirmative result. Every other state is a failure with its reason
    named, because each of them is missing evidence rather than a pass:

    * a required control arm with no ranking at all (it did not run or was not scored);
    * a ranking that is ``None`` -- an undefined correlation, which is not a correlation of zero
      (``spearman(...) or 0.0`` once turned it into one, and the arm then "failed to beat" or
      "matched" the full arm by arithmetic);
    * a full arm whose own ranking is undefined, which makes every comparison with it impossible;
    * a ranking that is present but not a finite number (NaN, +/-inf, a bool or a string):
      ``full - value < margin`` is False for NaN, so it used to PASS;
    * a declared margin that is not a finite non-negative number -- NaN compares False with
      everything and a negative margin lets an arm that beats the full arm pass -- which fails
      the gate and every comparison it would have made;
    * an E19 ``passes`` that is anything but ``True`` -- ``None`` (no comparable stratum) is not a
      pass, and ``False`` is the shortcut artefact the gate exists to report -- or an E19 record
      that says it is unsupported, or carries a non-finite difference or margin.
    """
    rankings = dict(rankings)
    failed, reasons = [], {}
    margin_ok = _is_number(margin) and math.isfinite(float(margin)) and float(margin) >= 0.0
    if not margin_ok:
        failed.append("control_margin_invalid")
        reasons[failed[-1]] = (f"the declared control margin {margin!r} is not a finite "
                               "non-negative number, so no arm can be compared with the full arm")
    full = rankings.get(ARM_FULL)
    full_ok = full is not None and _is_number(full) and math.isfinite(float(full))
    if not full_ok:
        failed.append(f"{ARM_FULL}_ranking_undefined")
        reasons[failed[-1]] = (
            "the full arm's ranking is undefined or absent, so no control can be compared with it"
            if full is None else
            f"the full arm's ranking {full!r} is not a finite number, so no control can be "
            "compared with it")
    for arm in required:
        value = rankings.get(arm)
        if arm not in rankings:
            failed.append(arm)
            reasons[arm] = ("the required control arm has no ranking: it did not run or was not "
                            "scored; a missing arm cannot be taken as one the full arm beat")
        elif value is None:
            failed.append(arm)
            reasons[arm] = ("its ranking is undefined (constant scores, or fewer than two rows): "
                            "missing evidence, not a ranking of zero")
        elif not (_is_number(value) and math.isfinite(float(value))):
            failed.append(arm)
            reasons[arm] = (f"its ranking {value!r} is not a finite number: invalid evidence, "
                            "not a ranking")
        elif not full_ok:
            failed.append(arm)
            reasons[arm] = "cannot be compared: the full arm's ranking is undefined or not finite"
        elif not margin_ok:
            failed.append(arm)
            reasons[arm] = "cannot be compared: the declared control margin is invalid"
        elif float(full) - float(value) < float(margin):
            failed.append(arm)
            reasons[arm] = (f"the full arm ({float(full):.4f}) does not beat it "
                            f"({float(value):.4f}) by the declared margin {float(margin):g}")
    record = {} if e19 is None else dict(e19)
    passes = record.get("passes")
    problem = None
    if passes is True:
        nonfinite = [name for name in ("value", "margin") if record.get(name) is not None
                     and not (_is_number(record[name]) and math.isfinite(float(record[name])))]
        if nonfinite:
            problem = f"E19 reports passes=True but its {' and '.join(nonfinite)} is not finite"
        elif record.get("supported", True) is False:
            problem = "E19 reports passes=True but is marked unsupported"
        elif record.get("margin") is not None and float(record["margin"]) < 0.0:
            problem = "E19 reports passes=True under a negative margin, which no result can fail"
    elif passes is None:
        problem = "E19 produced no comparable stratum, so the distance gate was not evaluated"
    else:
        problem = ("the full arm does not beat the distance-only rule inside the joint strata by "
                   "the declared margin")
    if problem:
        failed.append("distance_stratified_concordance")
        reasons[failed[-1]] = problem
    return failed, reasons


def distance_only_baseline(windows, *, wild_type):
    """The -Hamming-to-wild-type score. The measured shortcut, at 0.4312 on SPR-152."""
    return [float(-hamming(window, wild_type)) for window in windows]


def shortcut_context():
    """The measured numbers that make E19 a gate rather than a panel."""
    return {"spr152_distance_only_spearman": 0.4312,
            "spr152_distance_only_interval": [0.3000, 0.5609],
            "spr152_parent_spearman": 0.5634,
            "spr152_nearest_labelled_neighbour": 0.1965,
            "source": "outputs/her2_biology_20260923/affinity_scoring.json, recomputed",
            "consequence": ("most of the signal on the 152-row cohort is distance. Any affinity "
                            "claim passes through E19 or it is not made. AlphaBind's validation "
                            "set is also trastuzumab-CDR variants, so distance carries signal "
                            "there too and the same gate applies.")}
