"""E1-E22 as executable panels: numeric where supported, support-gated with counts.

The earlier report assembled stage metadata and a list of constants. A panel is a
*computation* here, and every one of them either returns a number with its
population and interval or returns its blocking reason with a count. The
distinction the review insists on, and that this module enforces, is between

* **unavailable data** -- E12 has no measured HER2 counter-target and E20's only
  multi-target resource leaks 78.6% under its own split, so no claim is made; and
* **unfinished code** -- which is not a support gate, and :func:`require_no_placeholder`
  refuses a panel that reports a blocking reason naming an implementation.

Three panels were named as missing code paths and are real ones here: E13 trains
nested data fractions (incremental summary supervision over fixed display labels, not a
clean total-N curve), E15 scores unseen parents and unseen studies beside SEEN-TRAINING
controls of the same target/assay/readout/scale stratum (and reports no delta where no such
stratum exists), and E16 summarises affinity among expression-matched measured pairs and the
converse inside one declared assay pair (a data-support summary, not a model prediction).
Each is a function that takes the services object and runs.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

from . import her2_routes_controls as controls_lib
from . import her2_routes_eval as eval_lib
from .her2_routes_sources import SCALE_ENRICHMENT, SCALE_EXPRESSION, SCALE_ORDINAL, SCALE_PKD
from .her2_runtime import require

#: Panels whose blocking reason must name a DATA fact, never an implementation.
SUPPORT_GATED = ("E12", "E20")

#: What a summary MEASURES. Decided by its scale, never by its family: ``family="enrichment"``
#: covers a binding enrichment and an expression enrichment alike, and calling every one of them
#: "expression" joined a binding readout to an affinity as if it were expression.
ROLE_AFFINITY = "affinity"
ROLE_EXPRESSION = "expression"
ROLE_BINDING_ENRICHMENT = "binding_enrichment"
ROLE_DISPLAY = "display_bin"
ROLE_UNRECOGNISED = "unrecognised_scale"
_ROLE_OF_SCALE = {SCALE_PKD: ROLE_AFFINITY, SCALE_EXPRESSION: ROLE_EXPRESSION,
                  SCALE_ENRICHMENT: ROLE_BINDING_ENRICHMENT, SCALE_ORDINAL: ROLE_DISPLAY}


def readout_role(row):
    """The measured role of one summary, from its scale: affinity, expression, binding
    enrichment, display bin, or an unrecognised scale (which no panel joins to anything)."""
    return _ROLE_OF_SCALE.get(str(getattr(row, "scale", "")), ROLE_UNRECOGNISED)


def _digest_of(parts):
    return hashlib.sha256("\x1f".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def _short_id(row):
    """A short stable id of one summary's population key, for naming a selected control."""
    return _digest_of(row.population_key)[:16]


#: Words that, appearing in a blocking reason, mean the panel is unfinished rather
#: than unsupported. Checked, because "not implemented" in a blocking-reason field
#: is the substitution this campaign's acceptance criteria forbid.
PLACEHOLDER_WORDS = ("not implemented", "not yet implemented", "todo", "placeholder",
                     "unfinished", "coming soon", "not wired")


def require_no_placeholder(record):
    """A blocking reason may describe absent data. It may not describe absent code."""
    reason = str(record.get("blocking_reason") or "").lower()
    offending = [word for word in PLACEHOLDER_WORDS if word in reason]
    require(not offending,
            f"{record.get('panel')}: the blocking reason names {offending}, which describes "
            "unfinished code rather than unavailable data. A support gate is for a measurement "
            "that does not exist; it is not a place to record one that was not written.")
    return record


def panel(name, **kwargs):
    return require_no_placeholder(eval_lib.panel_record(name, **kwargs))


# ---------------------------------------------------------------------------
# E13: data efficiency
# ---------------------------------------------------------------------------

#: What the E13 curve varies, stated on the curve itself so it cannot be read without it.
E13_SCOPE = {
    "kind": "incremental_summary_supervision",
    "label": ("INCREMENTAL summary supervision, conditional on the fixed display labels and on the "
              "full-training-pool threshold prior. This is not a clean total-N learning curve."),
    "varies": ("the measured ConstructSummary population keys (study, assay, target, full "
               "construct) that feed the local, whole-window and head tasks"),
    "held_fixed": ("the sorted-bin display library (not subset), the unlabelled paired replay "
                   "(not subset), and the quality bins fitted ONCE on the full training pool "
                   "before the subset was chosen (not refit per fraction)"),
    "thresholds_refit_per_fraction": False,
    "clean_total_n_claim": None,
    "clean_total_n_reason": ("the display supervision and the threshold prior do not shrink with "
                             "the fraction, so the total supervised information is not N; a clean "
                             "N would need the display pool and the thresholds subset as well, "
                             "which this bounded panel does not do"),
}


def fraction_directory_name(fraction):
    """A directory name for one E13 fraction. Whole percents keep their historical name
    (``fraction010``); any other fraction gets its own exact spelling, so two fractions never
    share a directory (``int(fraction * 100)`` mapped 0.105 and 0.10 both to ``fraction010``)."""
    percent = float(fraction) * 100.0
    if abs(percent - round(percent)) <= 1e-9 and round(percent) >= 1:
        return f"fraction{int(round(percent)):03d}"
    return "fraction_" + format(float(fraction), ".12g").replace(".", "p").replace("-", "m")


def validated_fractions(fractions):
    """The E13 fractions as sorted floats: finite, in (0, 1], distinct, with distinct directories."""
    values = []
    for item in fractions:
        require(not isinstance(item, bool) and isinstance(item, (int, float, np.integer,
                                                                 np.floating)),
                f"an E13 fraction must be a number, got {item!r}")
        value = float(item)
        require(math.isfinite(value) and 0.0 < value <= 1.0,
                f"an E13 fraction must be a finite number in (0, 1], got {item!r}")
        values.append(value)
    require(values, "E13 needs at least one data fraction")
    ordered = sorted(values)
    require(len(set(ordered)) == len(ordered), f"E13 fractions repeat: {ordered}")
    names = [fraction_directory_name(value) for value in ordered]
    require(len(set(names)) == len(names),
            f"E13 fractions {ordered} collide on their output directory names {names}")
    return ordered


def population_accounting(rows):
    """What a set of acquired summaries is, counted separately: the label budget, not the parents.

    ``population_keys`` is the unit one acquired summary is -- (full construct, study, assay,
    target) -- and is what a label budget means here. ``distinct_constructs`` can be smaller (one
    construct measured in two assays is two population keys), ``parents`` is smaller still (a
    one-parent library of 2 versus 8 labels is 2 versus 8 keys over one parent), and
    ``replicate_observations_available`` counts what the summaries hold, which is not the number
    of times a replicate was PRESENTED during training (that comes from the exposure ledger).
    """
    rows = list(rows)
    keys = [tuple(row.population_key) for row in rows]
    require(len(set(keys)) == len(keys),
            "the measured pool names one population key twice; an acquired unit is one "
            "(construct, study, assay, target) summary")
    return {"population_keys": len(keys), "summary_outcomes": len(rows),
            "distinct_constructs": len({row.identity for row in rows}),
            "parents": len({row.parent for row in rows}),
            "replicate_observations_available": int(sum(len(row.measurements) for row in rows)),
            "selected_keys_sha256": _digest_of(["|".join(key) for key in sorted(keys)])}


def _ledger_presentations(record):
    """The fit's ACTUAL repeated presentations, from its exposure ledger, or ``None`` per figure
    that the ledger does not carry. Never the pre-fit replicate count."""
    exposure = dict((record or {}).get("exposure") or {})
    totals = dict(exposure.get("totals") or {})
    return {"measurement_presentations": totals.get("measurement_presentations"),
            "microbatch_rows": totals.get("microbatch_rows"),
            "unique_measured_constructs": exposure.get("unique_measured_constructs"),
            "source": "the fit's exposure ledger (what training actually presented)"}


def _fixed_supervision(services, pools):
    """What did NOT change with the fraction, counted where the services can say so."""
    cache = getattr(services, "_cache", None) or {}
    bins = cache.get("bins")
    counts = dict(pools.get("counts") or {})
    return {"display_rows": pools.get("display_rows"), "display_subset": False,
            "replay_rows": counts.get("paired_replay"), "replay_subset": False,
            "quality_bin_entries": None if bins is None else len(bins),
            "quality_thresholds_fit_on": "the full training pool, before the subset was chosen",
            "quality_thresholds_refit_per_fraction": False}


def e13_learning_curve(services, route, *, fractions=(0.10, 0.25, 0.50, 1.00), updates,
                       seed, heartbeat=None):
    """Train the selected recipe on nested data fractions. Real fits, real curve.

    Each fraction is a *nested* subset of one shuffled ordering of the measured training
    summaries, trained with the same recipe to its own best development result. The x-axis is the
    number of distinct **population keys** in the subset -- the unit a label budget means here --
    and the parents, full constructs, summary outcomes and available replicate observations are
    reported beside it as separate counts, because a parent-family count is not the label budget
    (a one-parent library with 2 versus 8 labels is 2 versus 8 keys over one parent). The repeated
    presentations are the exposure ledger's, never the pre-fit replicate count.

    The curve is INCREMENTAL summary supervision: the display labels, the replay pool and the
    quality thresholds (fitted once on the full training pool before the subset is chosen) do not
    shrink with the fraction. It is labelled as such (:data:`E13_SCOPE`) and no clean total-N claim
    is made. Thresholds are not refit here.
    """
    fractions = validated_fractions(fractions)
    panelrows = services.development_panel_outcomes()
    outcomes = [row["value"] for row in panelrows]
    pools = services.training_pools()
    her2 = list(pools["summaries"]["her2"])
    other = list(pools["summaries"]["other_functional"])
    measured = her2 + other
    require(measured, "E13 needs a measured training pool")
    population_accounting(measured)                   # one population key twice is refused here
    her2_keys = {tuple(row.population_key) for row in her2}
    rng = np.random.default_rng([int(seed), 13])
    order = rng.permutation(len(measured))
    counts = [max(1, int(round(fraction * len(measured)))) for fraction in fractions]
    require(len(set(counts)) == len(counts),
            f"E13 fractions {fractions} select the same number of summaries {counts} from the "
            f"{len(measured)} available, so two points would be one fit presented twice")
    points, records = {}, {}
    for fraction, count in zip(fractions, counts):
        subset = [measured[int(index)] for index in order[:count]]
        accounting = population_accounting(subset)
        in_her2 = sum(1 for row in subset if tuple(row.population_key) in her2_keys)
        name = fraction_directory_name(fraction)
        directory = services.context.path("e13", name)
        stream = services.subset_stream(route, subset, seed=int(seed), updates=int(updates))
        record = services.train(route, stream=stream, updates=int(updates), seed=int(seed),
                                directory=directory, arm="full", heartbeat=heartbeat,
                                evaluate=services.development_metric(route))
        best = directory / "best.pt"
        scores = services.score_panel(route, best if best.is_file() else None, panelrows)
        score = eval_lib.spearman(scores, outcomes)
        points[fraction] = {**accounting, "score": score,
                            "selected_by_family": {"her2": in_her2,
                                                   "other_functional": len(subset) - in_her2},
                            "presentations": _ledger_presentations(record),
                            "output_directory": name, "updates": record["updates"]}
        records[str(fraction)] = {"best": record["best"], "exposure": record["exposure"]}
    curve = eval_lib.learning_curve(points, scope=E13_SCOPE)
    largest = points[max(points)]["score"] if points else None
    return panel("E13", population=len(panelrows), value=largest,
                 extra={"curve": curve, "fits": records, "supervision_scope": dict(E13_SCOPE),
                        "fixed_supervision": _fixed_supervision(services, pools),
                        "label_budget_unit": "population_key",
                        "value_is": ("the development Spearman of the LARGEST listed fraction's "
                                     "own best checkpoint"),
                        "protocol": ("nested subsets of one shuffled ordering, each trained with "
                                     "the same recipe to its own best development result")})


# ---------------------------------------------------------------------------
# E15: functional transfer
# ---------------------------------------------------------------------------

#: How E15 scores a summary, said once in the record: every row is conditioned at the HIGH bin, and
#: at its own assay identity only when the condition vocabulary holds it, else at ``null``.
E15_SCORING_CONDITION = {
    "bin": "high (every scored row, held-out and control alike)",
    "assay": ("the row's own assay identity when it is in the condition vocabulary, otherwise "
              "null. A withheld study's assay identity is normally absent from the vocabulary, "
              "so it is scored under null; a control in the SAME stratum shares that assay "
              "identity and is conditioned identically"),
    "source": "FlightServices.summary_scores"}

E15_CONTROL_POLICY = (
    "SEEN-TRAINING controls, not an independent in-distribution validation: they are TRAINING "
    "rows of the stratum, the model was fitted on them, and the comparison tells you how much "
    "lower the unseen score is than the score on rows it saw. A control is chosen inside the "
    "held-out row's own stratum -- same target, same canonical assay, same measured readout, "
    "same scale, an exact scalar value, an edit map -- by ranking the stratum's training rows on "
    "a hash of (stratum, population key). The selection reads no outcome and no input order, so "
    "reordering rows cannot change it; it takes as many as the held-out stratum has, or all it "
    "has. With fewer than two compatible training rows there is NO comparison and no delta, and "
    "no unrelated row is substituted.")


def _transfer_stratum(row):
    """``(target, canonical assay, measured readout, scale)`` -- the unit a comparison may span.

    The assay key already embeds the canonical study, family and target. The measured readout is
    stated explicitly beside the scale instead of being inferred from the row kind, so an exact
    point from an interval source and one from a point source are the same endpoint semantics.
    """
    return (str(row.target), str(row.assay), readout_role(row), str(row.scale))


def _stratum_document(stratum):
    target, assay, role, scale = stratum
    return {"target": target, "assay": assay, "readout": role, "scale": scale}


def _control_rank(stratum, row):
    return _digest_of(["E15-control", *stratum, *row.population_key])


def _select_seen_training_controls(stratum, candidates, wanted):
    """The stable, outcome-independent controls for one stratum, and the record of the choice."""
    ranked = sorted(candidates, key=lambda row: _control_rank(stratum, row))
    chosen = ranked[:int(wanted)]
    ids = [_short_id(row) for row in chosen]
    return chosen, {"control_ids": ids[:20], "control_ids_listed": min(20, len(ids)),
                    "control_ids_sha256": _digest_of(sorted(ids)),
                    "selection": ("first N of the stratum's training rows ranked by a hash of "
                                  "(stratum, population key); no outcome and no input order is "
                                  "read")}


def e15_transfer(services, route, checkpoint, *, margin=0.0):
    """Unseen parents and unseen studies, each beside a SEEN-TRAINING control of its own stratum.

    The held-out parents and studies are the ones ``sources.withheld`` declared **before** the
    fit, which is the only way a transfer claim can be about generalisation rather than about the
    fit. A held-out population is split into strata -- target, canonical assay, measured readout
    and scale -- and each stratum is scored beside training rows of THE SAME stratum, so the delta
    compares like with like. Rows of another target, assay, readout or scale are never a control:
    a first-N pick over "the same kind" mixed them, and a reverse-ordered wrong-assay control
    turned a correct -0.5 into a +1.5.

    A stratum with no compatible training rows has no comparison. Its standalone held-out score is
    reported, its delta is ``None`` and says why, and the panel never manufactures a transfer
    number from unrelated rows. The controls are labelled SEEN-TRAINING (:data:`E15_CONTROL_POLICY`).
    """
    summaries = services.summaries()
    held, excluded = {}, {}
    training, training_all = {}, {}
    for row in summaries:
        marker = services.withheld_marker(row)
        if not marker:
            if row.is_training:
                stratum = _transfer_stratum(row)
                training_all[stratum] = training_all.get(stratum, 0) + 1
                if row.exact_value is not None and row.has_edit_map:
                    training.setdefault(stratum, []).append(row)
            continue
        # The target is the EXACT value -- every replicate an exact finite point. ``row.value``
        # is the mean of the uncensored replicates, which for a construct with a floor replicate
        # is a number the assay never produced. A row without one is excluded and counted.
        if not row.has_edit_map:
            reason = "no_edit_map"
        elif row.exact_value is None:
            reason = row.exact_status
        else:
            held.setdefault(marker, []).append(row)
            continue
        counts = excluded.setdefault(marker, {})
        counts[reason] = counts.get(reason, 0) + 1
    withheld = sum(len(rows) for rows in held.values())
    if not held and not excluded:
        return panel("E15", population=0, supported=False,
                     reason=("no parent or study was declared withheld in sources.withheld, so "
                             "there is no unseen background to measure transfer onto. This is a "
                             "configuration fact and it is reported as one."))
    if not held:
        return panel("E15", population=0, supported=False,
                     reason=("every withheld row was excluded from scoring (no declared edit map, "
                             "or not an exact scalar measurement), so there is no exact target to "
                             f"correlate the transfer score with: {excluded}"),
                     extra={"excluded_by_heldout": excluded})
    vocabulary = getattr(getattr(services, "condition_spec", None), "assays", None)
    model, heads, load = services.load_for_scoring(route, checkpoint)
    blocks, deltas = {}, []
    strata_without = 0
    for marker, rows in sorted(held.items()):
        by_stratum = {}
        for row in rows:
            by_stratum.setdefault(_transfer_stratum(row), []).append(row)
        records = []
        for stratum, members in sorted(by_stratum.items()):
            members = sorted(members, key=lambda row: tuple(row.population_key))
            heldout_rho = eval_lib.spearman(
                services.summary_scores(route, model, members),
                [float(row.exact_value) for row in members])
            candidates = training.get(stratum, [])
            chosen, selection = _select_seen_training_controls(stratum, candidates, len(members))
            comparator = {"compatible_training_rows": len(candidates),
                          "training_rows_in_stratum": int(training_all.get(stratum, 0)),
                          "control_rows": len(chosen), "heldout_rows": len(members),
                          "size_matched": len(chosen) == len(members)}
            control_rho = delta = None
            if len(chosen) < 2:
                comparator.update(
                    status="absent",
                    reason=(f"{len(candidates)} compatible seen-training row(s) share this target, "
                            "canonical assay, readout and scale (an exact scalar and an edit map "
                            "are required, and at least two); no control is substituted from "
                            "another stratum, so there is no transfer delta"))
            else:
                chosen = sorted(chosen, key=lambda row: tuple(row.population_key))
                control_rho = eval_lib.spearman(
                    services.summary_scores(route, model, chosen),
                    [float(row.exact_value) for row in chosen])
                if heldout_rho is None or control_rho is None:
                    comparator.update(status="undefined", reason=(
                        "a Spearman is undefined for constant scores or constant values on one "
                        "side, so the difference is not computed"))
                else:
                    delta = float(heldout_rho - control_rho)
                    comparator.update(status="available", reason=None)
                comparator.update(selection)
            comparator["control_spearman"] = control_rho
            if delta is None:
                strata_without += 1
            else:
                deltas.append(delta)
            assay = stratum[1]
            records.append({
                "stratum": _stratum_document(stratum), "heldout_rows": len(members),
                "heldout_kinds": sorted({row.kind for row in members}),
                "heldout_spearman": heldout_rho,
                "assay_conditioned_as": (None if vocabulary is None else
                                         ("own" if assay in vocabulary else "null")),
                "comparator": comparator, "delta": delta})
        single = records[0] if len(records) == 1 else None
        marker_deltas = [record["delta"] for record in records if record["delta"] is not None]
        blocks[marker] = {
            "rows": sum(record["heldout_rows"] for record in records),
            "excluded": dict(sorted(excluded.get(marker, {}).items())),
            "strata": records,
            "spearman": None if single is None else single["heldout_spearman"],
            "seen_training_control_rows": (None if single is None
                                           else single["comparator"]["control_rows"]),
            "seen_training_control_spearman": (None if single is None
                                               else single["comparator"]["control_spearman"]),
            "delta": float(np.mean(marker_deltas)) if marker_deltas else None,
            "spearman_note": (None if single is not None else
                              "this held-out population spans several strata; a Spearman pooled "
                              "across assays or scales is a number on no scale, so standalone "
                              "scores are reported per stratum only")}
    del model, heads
    scored_rows = sum(block["rows"] for block in blocks.values())
    common = {"by_heldout": blocks, "checkpoint": load,
              "margin": float(margin), "withheld_rows": int(withheld),
              "excluded_by_heldout": excluded,
              "strata_with_comparator": len(deltas), "strata_without_comparator": strata_without,
              "scoring_condition": dict(E15_SCORING_CONDITION),
              "control_policy": E15_CONTROL_POLICY,
              "target": ("exact scalar values only (ConstructSummary.exact_value); a "
                         "row that is not an exact point is excluded and counted in "
                         "excluded_by_heldout")}
    if not deltas:
        return panel("E15", population=scored_rows, supported=False,
                     reason=("no held-out stratum has a compatible seen-training comparison "
                             "(same target, canonical assay, readout and scale, with an exact "
                             "scalar and an edit map), so no transfer delta exists. The "
                             "standalone held-out scores are in by_heldout; nothing was "
                             "substituted from an unrelated stratum."),
                     extra=common)
    return panel("E15", population=scored_rows, value=float(np.mean(deltas)),
                 extra={**common,
                        "reading": ("the value is the mean over strata of (unseen standalone "
                                    "Spearman minus the Spearman of matched-size SEEN-TRAINING "
                                    "controls of the same stratum). A held-out population scored "
                                    "in isolation says nothing about transfer, and the controls "
                                    "are rows the model was fitted on, not independent "
                                    "in-distribution validation.")})


# ---------------------------------------------------------------------------
# E16: distinct functional readouts
# ---------------------------------------------------------------------------

#: The most pairs one E16 stratum materialises. Taken in sorted full-identity order, so it is
#: deterministic, and recorded (``pairs_capped``) when it bites.
E16_PAIR_CAP = 4096

E16_READING = (
    "a DATA-SUPPORT summary: it counts matched pairs among MEASURED constructs and the spread of "
    "the other readout inside them. No model prediction enters, so it says nothing about a "
    "model's ability to separate affinity from expression.")


def _declared_assay_pairs(assay_pairs):
    """The explicit selections, validated: ``[{study, target, affinity_assay, expression_assay}]``.

    A declaration names ONE affinity assay and ONE expression assay of one canonical study and
    target. Several declarations are several strata and stay separate; the same one twice is
    refused.
    """
    declared, seen = [], set()
    for item in list(assay_pairs or ()):
        require(isinstance(item, dict)
                and set(item) == {"study", "target", "affinity_assay", "expression_assay"}
                and all(isinstance(value, str) and value for value in item.values()),
                "an assay pair is {study, target, affinity_assay, expression_assay}, four "
                f"non-empty strings; got {item!r}")
        key = tuple(item[name] for name in ("study", "target", "affinity_assay",
                                            "expression_assay"))
        require(key not in seen, f"the assay pair {key} is declared twice")
        seen.add(key)
        declared.append(dict(item))
    return declared


def _assay_inventory(cell):
    return {role: [{"assay": assay, "scale": scale, "constructs": len({row.identity
                                                                       for row in rows}),
                    "summaries": len(rows)}
                   for (assay, scale), rows in sorted(cell[role].items())]
            for role in (ROLE_AFFINITY, ROLE_EXPRESSION)}


def e16_readouts(services, *, tolerance=None, affinity_tolerance=None, expression_tolerance=None,
                 assay_pairs=None):
    """Affinity among expression-matched variants, and expression among affinity-matched. DATA only.

    Constructs are joined by **exact full identity** between an affinity summary and an expression
    summary of one canonical study and target, inside ONE declared (affinity assay, expression
    assay) pair; nothing converts one readout into another, and the residual imbalance after
    approximate matching is reported.

    * A summary's role comes from its **scale** (:func:`readout_role`): pKD is affinity, the
      expression scale is expression, and a binding enrichment is neither -- ``family ==
      "enrichment"`` used to make every enrichment an "expression" and joined a binding readout to
      an affinity as if it measured expression.
    * Assays are never picked by position. A (study, target) with exactly one eligible affinity
      assay and one eligible expression assay has that pair inferred and recorded. Several of
      either is ambiguous: it needs an explicit ``assay_pairs`` selection and is otherwise
      reported unsupported with its inventory. Several explicit pairs stay separate strata.
    * Two summaries of one full construct in one role of one stratum are refused outright.
    * Affinity and expression have their own tolerances (:func:`~.her2_routes_eval.
      resolve_readout_tolerances`), in their own units; the legacy single ``tolerance`` is
      broadcast only explicitly and the record says so.
    """
    resolved = eval_lib.resolve_readout_tolerances(
        tolerance=tolerance, affinity_tolerance=affinity_tolerance,
        expression_tolerance=expression_tolerance,
        units={"affinity": SCALE_PKD, "expression": SCALE_EXPRESSION})
    declared = _declared_assay_pairs(assay_pairs)
    summaries = services.summaries()
    excluded = {"affinity": {}, "expression": {}}
    unsupported_roles = {}
    groups = {}
    for row in summaries:
        role = readout_role(row)
        if role not in (ROLE_AFFINITY, ROLE_EXPRESSION):
            unsupported_roles[role] = unsupported_roles.get(role, 0) + 1
            continue
        # Exact scalars only: the matching tolerance compares these numbers directly, and the
        # uncensored-only mean of a construct with a floor replicate is not a measurement.
        if row.exact_value is None:
            counts = excluded[role]
            counts[row.exact_status] = counts.get(row.exact_status, 0) + 1
            continue
        cell = groups.setdefault((str(row.study), str(row.target)),
                                 {ROLE_AFFINITY: {}, ROLE_EXPRESSION: {}})
        cell[role].setdefault((str(row.assay), str(row.scale)), []).append(row)
    explicit = {}
    for item in declared:
        explicit.setdefault((item["study"], item["target"]), []).append(item)
    unknown = sorted(set(explicit) - set(groups))
    require(not unknown,
            f"an explicit assay pair names {unknown[:3]}, which holds no exact affinity or "
            "expression summary; a selection that matches nothing is an error, not a fallback")
    specs, unsupported_groups, one_role_only = [], [], []
    for (study, target), cell in sorted(groups.items()):
        affinity_keys, expression_keys = sorted(cell[ROLE_AFFINITY]), sorted(cell[ROLE_EXPRESSION])
        inventory = _assay_inventory(cell)
        if (study, target) in explicit:
            for item in explicit[(study, target)]:
                left = [key for key in affinity_keys if key[0] == item["affinity_assay"]]
                right = [key for key in expression_keys if key[0] == item["expression_assay"]]
                require(len(left) == 1 and len(right) == 1,
                        f"the declared pair {item['affinity_assay']!r} / "
                        f"{item['expression_assay']!r} does not name exactly one affinity "
                        f"assay-and-scale and one expression assay-and-scale in {study!r} / "
                        f"{target!r}; inventory {inventory}")
                specs.append((study, target, left[0], right[0], "explicit"))
            continue
        if not affinity_keys or not expression_keys:
            one_role_only.append({"study": study, "target": target, "inventory": inventory})
        elif len(affinity_keys) == 1 and len(expression_keys) == 1:
            specs.append((study, target, affinity_keys[0], expression_keys[0], "inferred_unique"))
        else:
            unsupported_groups.append({
                "study": study, "target": target, "inventory": inventory,
                "reason": ("several eligible affinity or expression assays: the pair is "
                           "ambiguous and is never chosen by position. Declare it with "
                           "assay_pairs.")})
    strata, total_pairs = [], 0
    for study, target, affinity_key, expression_key, how in specs:
        cell = groups[(study, target)]
        tables = {}
        for role, key in ((ROLE_AFFINITY, affinity_key), (ROLE_EXPRESSION, expression_key)):
            table = {}
            for row in cell[role][key]:
                table.setdefault(row.identity, []).append(row)
            tables[role] = table
        repeated = {role: sorted(identity for identity, rows in table.items() if len(rows) > 1)
                    for role, table in tables.items()}
        require(not any(repeated.values()),
                f"E16 stratum {study!r} / {target!r} / {affinity_key[0]!r} + {expression_key[0]!r} "
                f"holds more than one summary of the same full construct within one role "
                f"(affinity {len(repeated[ROLE_AFFINITY])}, expression "
                f"{len(repeated[ROLE_EXPRESSION])} construct(s)); a pair cannot be built from an "
                "ambiguous measurement")
        both = sorted(set(tables[ROLE_AFFINITY]) & set(tables[ROLE_EXPRESSION]))
        candidate_pairs = len(both) * (len(both) - 1) // 2
        pairs = []
        for left in range(len(both)):
            for right in range(left + 1, len(both)):
                a_id, b_id = both[left], both[right]
                pairs.append({
                    "affinity_left": float(tables[ROLE_AFFINITY][a_id][0].exact_value),
                    "affinity_right": float(tables[ROLE_AFFINITY][b_id][0].exact_value),
                    "expression_left": float(tables[ROLE_EXPRESSION][a_id][0].exact_value),
                    "expression_right": float(tables[ROLE_EXPRESSION][b_id][0].exact_value)})
                if len(pairs) >= E16_PAIR_CAP:
                    break
            if len(pairs) >= E16_PAIR_CAP:
                break
        units = {"affinity": affinity_key[1], "expression": expression_key[1]}
        if resolved["source"] == "legacy_broadcast":
            block = eval_lib.readout_separation(pairs, tolerance=resolved["affinity"], units=units)
        else:
            block = eval_lib.readout_separation(
                pairs, affinity_tolerance=resolved["affinity"],
                expression_tolerance=resolved["expression"], units=units)
        total_pairs += len(pairs)
        strata.append({"study": study, "target": target, "selection": how,
                       "affinity_assay": affinity_key[0], "affinity_scale": affinity_key[1],
                       "expression_assay": expression_key[0], "expression_scale": expression_key[1],
                       "constructs_with_both": len(both), "candidate_pairs": candidate_pairs,
                       "pairs": len(pairs), "pairs_capped": candidate_pairs > len(pairs),
                       **block})
    common = {"strata": strata, "stratum_count": len(strata),
              "excluded_not_exact": excluded, "unsupported_roles": unsupported_roles,
              "unsupported_groups": unsupported_groups, "groups_with_one_role_only": one_role_only,
              "tolerances": {"affinity": resolved["affinity"], "expression": resolved["expression"],
                             "units": dict(resolved["units"]), "source": resolved["source"],
                             "note": resolved["note"]},
              "evidence_kind": "data_support_summary", "model_predictions": None,
              "reading": E16_READING,
              "rule": ("exact full identity inside one canonical study and target and one declared "
                       "or uniquely inferred (affinity assay, expression assay) pair, on exact "
                       "scalar values (a row that is not an exact point is excluded and counted). "
                       "A role comes from the scale, never from the family; a binding enrichment is "
                       "not expression. Nothing converts an expression ratio into an affinity.")}
    if total_pairs == 0:
        return panel("E16", population=0, supported=False,
                     reason=("no declared or uniquely inferable (study, target, affinity assay, "
                             "expression assay) pair holds two constructs that each carry BOTH an "
                             "exact affinity (pKD) value and an exact expression value, so there "
                             "is no matched pair to compare the two readouts on. Joining across "
                             "studies, targets or assays would compare different scales, a "
                             "binding enrichment is never an expression readout, and an "
                             "ambiguous assay choice needs an explicit selection: see "
                             "unsupported_groups and unsupported_roles."),
                     extra=common)
    if len(strata) == 1:
        single = strata[0]
        return panel("E16", population=total_pairs,
                     value=float(single["expression_matched_pairs"]),
                     extra={**common, **{key: value for key, value in single.items()
                                         if key not in common}})
    return panel("E16", population=total_pairs, value=None,
                 extra={**common,
                        "headline_value": ("withheld: several strata are reported separately and "
                                           "their pair counts are not summed, because each "
                                           "stratum's tolerances apply to its own scales")})


# ---------------------------------------------------------------------------
# E5 / E19: substitution reversals and distance shortcuts
# ---------------------------------------------------------------------------

def e5_reversals(services, route, checkpoint, *, tolerance=0.0):
    """Oriented substitution groups seen in both directions. Support-gated honestly."""
    pairs = services.reversal_pairs(route, checkpoint)
    block = controls_lib.matched_reversals(pairs, tolerance=float(tolerance))
    # The geometry the endpoint profiles were measured against, when the services record it.
    record = getattr(services, "adaptation_bank_record", None)
    block = {**block, "geometry_bank": (record() if callable(record) else None)}
    if block["support_gated"]:
        return panel("E5", population=block["groups_seen"], supported=False,
                     reason=("no oriented substitution group was observed in BOTH directions on "
                             f"this panel ({block['groups_seen']} groups seen). The ceiling "
                             "argument only holds under that exact grouping, so the panel is "
                             "reported with its counts rather than with a relaxed grouping."),
                     extra=block)
    return panel("E5", population=block["participating_comparisons"],
                 value=block["macro_accuracy"], extra=block)


def _require_bank(bank, *, label, labelled):
    """A bank record is ``{"windows", ["values"], "provenance"}``; refuse anything vaguer."""
    require(isinstance(bank, dict) and bank.get("windows") and isinstance(
        bank.get("provenance"), dict) and bank["provenance"],
        f"E19 needs {label} as a record with its windows and its provenance (count, digest, "
        "source semantics). A bare list says nothing about where its sequences came from.")
    windows = [str(window) for window in bank["windows"]]
    if labelled:
        values = list(bank.get("values") or ())
        require(len(values) == len(windows),
                f"E19's {label} carries {len(values)} labels for {len(windows)} windows")
        require(all(isinstance(value, (int, float)) and np.isfinite(value) for value in values),
                f"E19's {label} carries a non-finite label")
    else:
        require(bank.get("values") is None,
                f"E19's {label} is geometry only and must carry no labels. Display classes are "
                "never affinities.")
    return windows


def e19_shortcuts(services, route, checkpoint, *, margins, training_bank, adaptation_bank):
    """The distance gate. A reported shortcut artefact is a legitimate outcome.

    Two banks, and they are different things. ``training_bank`` is the EXACT labelled
    training population (windows and their exact pKD values) behind the nearest-labelled-
    neighbour baseline: it is the model-free predictor a learned scorer has to beat. The
    development panel's own leave-one-out outcomes are not training neighbours and are never
    used for it. ``adaptation_bank`` is the ELIGIBLE adaptation geometry -- every training
    sequence the model may adapt on, labelled or not, display cores included -- used only to
    measure each development window's distance to its nearest adaptation sequence. It carries no
    labels: a display class is not an affinity.

    The strata are joint (distance to wild type x distance to the nearest adaptation sequence),
    exact integers on both axes. A distance-only rule is constant inside a stratum, so it scores
    0.5 there and the model is credited only for what it knows beyond distance. Neither bank may
    contain a development window.
    """
    margin = eval_lib.finite_scalar(dict(margins)["distance_stratified"],
                                    "E19's distance_stratified margin")
    require(margin >= 0.0,
            f"E19's distance_stratified margin is {margin:g}; a negative margin passes a model "
            "that loses to the distance rule")
    rows = services.development_panel_outcomes()
    outcomes = [row["value"] for row in rows]
    windows = [row["window"] for row in rows]
    labelled = _require_bank(training_bank, label="an exact training-labelled bank",
                             labelled=True)
    geometry = _require_bank(adaptation_bank, label="an adaptation geometry bank",
                             labelled=False)
    for name, members in (("training-labelled", labelled), ("adaptation geometry", geometry)):
        leaked = sorted(set(windows) & set(members))
        require(not leaked,
                f"E19's {name} bank contains {len(leaked)} development window(s), e.g. "
                f"{leaked[:3]}. A bank that holds the windows it is scored on is not training "
                "data.")
    wild = services.config["her2"]["wild_type_core"]
    scores = services.score_panel(route, checkpoint, rows)
    distance = controls_lib.distance_only_baseline(windows, wild_type=wild)
    wild_distances = [controls_lib.hamming(window, wild) for window in windows]
    adaptation_distances = controls_lib.nearest_hamming(windows, geometry)
    strata, scheme = controls_lib.joint_distance_strata(wild_distances, adaptation_distances)
    full = controls_lib.stratified_concordance(scores, outcomes, strata)
    shortcut = controls_lib.stratified_concordance(distance, outcomes, strata)
    neighbours = [controls_lib.out_of_fold_nearest(window, bank=labelled,
                                                   values=training_bank["values"])
                  for window in windows]
    predictions = [entry["prediction"] for entry in neighbours]
    if all(prediction is not None for prediction in predictions):
        neighbour_block = {
            "stratified": controls_lib.stratified_concordance(predictions, outcomes, strata),
            "spearman": eval_lib.spearman(predictions, outcomes)}
    else:
        neighbour_block = {"stratified": None, "spearman": None,
                           "reason": "the labelled bank returned no neighbour for some window"}
    delta = (None if full["macro_average"] is None or shortcut["macro_average"] is None
             else full["macro_average"] - shortcut["macro_average"])
    require(delta is None or np.isfinite(delta),
            f"E19's stratified difference is {delta!r}; a non-finite difference is not a result "
            "and cannot be compared with the margin")
    reason = (None if delta is not None else
              "no stratum holds a comparable pair for both the model and the distance rule, so "
              "there is no difference to gate on")
    detail = {"stratified_concordance": full,
              "distance_only_stratified": shortcut,
              "strata_scheme": scheme,
              "training_neighbour_baseline": {
                  **neighbour_block, "predictions": neighbours[:50],
                  "bank": dict(training_bank["provenance"]),
                  "rule": ("nearest EXACT-labelled training neighbour's value; the "
                           "development panel is never its own bank")},
              "adaptation_geometry_bank": dict(adaptation_bank["provenance"]),
              "margin": margin, "reason": reason,
              "passes": None if delta is None else bool(delta >= margin),
              "context": controls_lib.shortcut_context()}
    if delta is None:
        # Missing data, not a pass and not a number: the panel says so in the fields the report's
        # unsupported list and the gate both read, rather than only in a None value.
        return panel("E19", population=len(rows), supported=False, reason=reason, extra=detail)
    return panel("E19", population=len(rows), value=delta, extra=detail)


# ---------------------------------------------------------------------------
# the assembled report
# ---------------------------------------------------------------------------

def assemble(*, panels, stages, unsupported, caveats, extra=None):
    """One report. Completeness is a separate fact from every panel's value."""
    for record in panels.values():
        if isinstance(record, dict) and "panel" in record:
            require_no_placeholder(record)
    complete = all(status == "completed" for status in dict(stages).values())
    numeric = sorted(name for name, record in panels.items()
                     if isinstance(record, dict) and record.get("value") is not None)
    document = {"panels": len(panels), "blocks": panels, "stages": dict(stages),
                "numeric_panels": numeric, "unsupported": sorted(unsupported),
                "completion": {"complete": complete, "stages": dict(stages),
                               "rule": ("complete means every required stage completed. A launch, "
                                        "a running worker and a partial stage list are three "
                                        "different facts.")},
                "support_gate_rule": ("a panel reports a number or the DATA fact that blocks it. "
                                      "Unfinished code is not a support gate and "
                                      "require_no_placeholder refuses one."),
                "standing_caveats": list(caveats)}
    if extra:
        document.update(dict(extra))
    return document


def no_gain_diagnosis(*, panels, controls, exposure, learning_curve):
    """What a null result means here, and what the finite next step is.

    A campaign that measures no gain has produced a finding, and the finding has to
    name which of the three explanations the evidence supports. It is not a
    successful short-circuit and it is not a reason to skip the remaining
    development.
    """
    rankings = dict(controls.get("rankings") or {})
    full = rankings.get("full")
    head_only = rankings.get("head_only")
    reconstruction = rankings.get("reconstruction_only")
    curve = dict(learning_curve or {})
    rows = list(curve.get("rows") or [])
    # Three states, not two: True, False, and UNKNOWN (``None``, a curve with fewer than two scored
    # fractions or none at all). Unknown is not a degradation, and ``bool(None)`` made it one.
    monotone = curve.get("monotone")
    diagnosis = []
    if full is not None and head_only is not None and abs(full - head_only) < 0.01:
        diagnosis.append({"explanation": "the gain is in the head, not the decoder",
                          "evidence": {"full": full, "head_only": head_only},
                          "next": ("the decoder objectives are the thing to change: the numeric "
                                   "term's coefficient, the contrast composition and the "
                                   "whole-window preference support")})
    if full is not None and reconstruction is not None and full - reconstruction < 0.01:
        diagnosis.append({"explanation": "the measured functional signal did nothing at all",
                          "evidence": {"full": full, "reconstruction_only": reconstruction},
                          "next": ("check the numeric objective's sensitivity probe and the "
                                   "per-task gradient fractions before adding updates")})
    if rows and monotone is False:
        diagnosis.append({
            "explanation": ("the learning-curve point estimates did not rise with more measured "
                            "data"),
            "evidence": {"fractions": [row.get("fraction") for row in rows],
                         "scores": [row.get("score") for row in rows],
                         "largest_decrease": curve.get("largest_decrease")},
            "interpretation": ("a description of single-fit point estimates with no interval, over "
                               "INCREMENTAL summary supervision on fixed display labels. It does "
                               "not by itself establish a capacity or conditioning limit, and it "
                               "does not rule out noise or a data limit"),
            "next": ("repeat the curve over seeds to get an interval and read the controls "
                     "before attributing it to capacity; do not extend the schedule on this "
                     "evidence alone")})
    if not diagnosis:
        diagnosis.append({"explanation": "no single explanation is isolated by the controls",
                          "evidence": {"rankings": rankings},
                          "next": ("the declared finite extension: 5,000-update increments while "
                                   "the development metric improves by more than the plateau "
                                   "delta, at most two learning-rate reductions")})
    return {"diagnosis": diagnosis,
            "learning_curve": {"monotone": monotone, "reason": curve.get("monotone_reason"),
                               "scored_points": curve.get("scored_points"),
                               "fractions": len(rows)},
            "exposure": dict(exposure or {}),
            "rule": ("a no-gain outcome is a finding with a diagnosis and a finite next step. It "
                     "is not a successful campaign and it is not a reason to omit the remaining "
                     "development.")}
