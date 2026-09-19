"""Prespecified selection, the staged allocation arithmetic, and the guarded freeze.

Three decisions are fixed here, before any fitting, because each of them is a
place where a result could otherwise be chosen after the fact:

**The primary statistic is the macro average of per-stratum validation AP over
the training-distance strata ``[1, 2, >=3]``.** The motivation is the original
report's **reserved-test** ranking table -- already inspected and published there,
and test evidence, not validation: it shows DPO lower in *every* stratum while
higher in aggregate, because that aggregate is dominated by the distance-1
stratum, which holds 70,856 of its 78,652 rows. A macro average over strata cannot
be rescued that way. What this module computes is that same statistic on the
**validation** rows; no reserved label enters it. Stratum ``0`` is reported
separately and never enters the mean.

**Candidacy requires an actually reached, gate- and diversity-eligible endpoint at
the exact nominal budget, for all three seeds.** Anything short is ineligible and
is reported as such. There is no backfill from the rolling last-passing state, no
cheaper-budget substitution and no "closest available" budget.

**Ties break on mean worst-stratum AP, then lexicographic arm id.** Both are
deterministic under any dict order. If nothing is eligible the result is "none
eligible", and that is the reported outcome rather than a prompt to relax
something.

A validation-selected positive delta against the matched continued-SFT control is
a *ranking* observation on validation rows. It is not a confirmatory affinity
claim, and the artifacts say so in those words.

**Two results, never merged into one sentence.** "The gate held" -- the
parent-relative likelihood stayed inside 1.0 nat/sequence and the trajectory ran
to its budget -- is a statement about the likelihood, and it is the thing this
revision was built to observe. "The prespecified ranking improved" requires an
eligible endpoint at the exact budget for every seed *and* an eligible matched
control to compare it against. A stage can produce the first and none of the
second; :func:`matched_control_deltas` therefore reports the raw descriptive
deltas with their eligibility flags and the eligible subset separately, and
:func:`select_coefficient` reports "none eligible" rather than reaching for the
nearest available number.
"""
from __future__ import annotations

from .her2_eval import DIVERSITY_GATES
from .her2_lineage import (GUARDED_SELECTION_SCHEMA, GUARDED_STAGE, assert_cannot_unlock,
                           verify_artifact)
from .her2_objectives import OBJECTIVES, arm_id, validate_coefficients
from .her2_runtime import require

#: The strata the macro average runs over. Fixed in advance.
SELECTION_STRATA = ("1", "2", ">=3")
#: Reported, but stratum 0 is not averaged in: it is a different population and,
#: in this library, often empty.
REPORTED_STRATA = ("0", "1", "2", ">=3")
#: The nominal budgets. No higher-budget control exists in this revision.
BUDGETS = (180.0, 360.0, 600.0)
#: Each trajectory runs once to the maximum budget and drops a checkpoint at each
#: crossing, so the allocation is the MAX per arm-seed, not the sum of the three.
MAX_BUDGET = 600.0

#: Field-name tokens that would mean a reserved outcome reached the selector.
#: Matched on ``_``-separated tokens rather than as substrings, so ``last_passing``
#: is not mistaken for a test field while ``test_average_precision`` is caught.
FORBIDDEN_SELECTION_TOKENS = frozenset({"test", "spr", "assay", "kd", "affinity", "outcome"})

SELECTION_RULE = (
    "at each exact nominal budget, one coefficient configuration per objective, chosen by the "
    "mean across all three seeds of the macro-average validation AP over the fixed strata "
    "[1, 2, >=3]; candidates must have a reached, gate- and diversity-eligible endpoint at that "
    "exact budget for every seed. Ties: mean worst-stratum AP, then lexicographic arm id.")

#: The identity blocks a later stage must find unchanged. A predecessor frozen
#: under different parents, different pinned sources, a different device or a
#: different batch size is a different experiment, and ``revision`` alone does not
#: say so: the revision block carries this repository's code and config hashes and
#: nothing about which parents were continued.
CAMPAIGN_IDENTITY_KEYS = ("schema_version", "revision", "inherited", "source_digests", "device",
                          "batch_sequences")


# ---------------------------------------------------------------------------
# the declared grid and its arithmetic
# ---------------------------------------------------------------------------

def declared_arms(config):
    """Every ``(stage, objective, coefficients)`` the protocol declares, validated."""
    arms = []
    for stage in sorted(config["stages"], key=lambda entry: entry["stage"]):
        for entry in stage["arms"]:
            coefficients = dict(entry.get("coefficients") or {})
            resolved = validate_coefficients(entry["objective"], coefficients)
            arms.append({"stage": int(stage["stage"]), "objective": entry["objective"],
                         "coefficients": resolved,
                         "arm_id": arm_id(entry["objective"], resolved),
                         "reused_from_stage": entry.get("reused_from_stage")})
    fitted = [arm["arm_id"] for arm in arms if not arm["reused_from_stage"]]
    require(len(set(fitted)) == len(fitted),
            f"The same arm is declared as fitted twice: {sorted(fitted)}. A control is fitted once "
            "and then declared with reused_from_stage, which is what keeps the allocation honest.")
    for arm in arms:
        if arm["reused_from_stage"]:
            require(arm["arm_id"] in fitted,
                    f"{arm['arm_id']} is declared reused but is never fitted anywhere")
    require(set(arm["objective"] for arm in arms) <= set(OBJECTIVES),
            "The grid names an objective this revision does not implement")
    return arms


def expected_trajectories(config, stage):
    """Exactly the ``{arm_id}_seed{seed}`` keys one stage must produce. No more, no fewer.

    This is what makes "the stage is complete" checkable. Without it a stage that
    fitted two of its eighteen declared arm-seeds freezes happily, and the marker
    the next stage verifies says nothing about the sixteen that never ran.
    """
    fitted = [arm for arm in declared_arms(config)
              if arm["stage"] == int(stage) and not arm["reused_from_stage"]]
    require(fitted, f"Stage {stage} declares no fitted arms")
    return sorted(f"{arm['arm_id']}_seed{int(seed)}"
                  for arm in fitted for seed in config["seeds"])


def campaign_identity_differences(left, right):
    """Which identity blocks differ between a marker and the current campaign."""
    left, right = dict(left or {}), dict(right or {})
    return sorted(key for key in CAMPAIGN_IDENTITY_KEYS if left.get(key) != right.get(key))


def stage_allocation(config):
    """Recompute the declared allocation from the grid and reconcile it.

    The number is ``sum over fitted arm-configs and seeds of the MAXIMUM budget``.
    It is emphatically not ``sum(180 + 360 + 600)``: one trajectory drops three
    checkpoints, so charging it three times would inflate the declared ceiling by
    a factor of 1.9. Both numbers are computed and the wrong one is recorded under
    a name that says it is wrong, so the distinction survives into the artifact.
    """
    seeds = list(config["seeds"])
    arms = declared_arms(config)
    budgets = [float(b) for b in config["budgets_gpu_seconds"]]
    maximum = max(budgets)
    stages, total, fitted = {}, 0.0, 0
    for stage in sorted({arm["stage"] for arm in arms}):
        rows = [arm for arm in arms if arm["stage"] == stage and not arm["reused_from_stage"]]
        reused = [arm for arm in arms if arm["stage"] == stage and arm["reused_from_stage"]]
        allocation = len(rows) * len(seeds) * maximum
        stages[str(stage)] = {
            "fitted_configurations": len(rows), "reused_configurations": len(reused),
            "reused_arm_ids": sorted(arm["arm_id"] for arm in reused),
            "seeds": len(seeds), "max_budget_gpu_seconds": maximum,
            "allocation_gpu_seconds": allocation,
            "arm_ids": sorted(arm["arm_id"] for arm in rows)}
        total += allocation
        fitted += len(rows)
    document = {
        "configurations_fitted": fitted,
        "configuration_rows": len(arms),
        "reused_rows": len(arms) - fitted,
        "seeds": len(seeds), "budgets_gpu_seconds": budgets,
        "stages": stages,
        "total_charged_training_gpu_seconds": total,
        "basis": ("sum over fitted arm-configurations and seeds of the maximum budget; one "
                  "continuous trajectory drops a checkpoint at each budget it crosses"),
        "incorrect_sum_of_budgets": len([a for a in arms if not a["reused_from_stage"]])
                                    * len(seeds) * sum(budgets),
        "incorrect_sum_note": ("recorded only to name the arithmetic this is NOT: charging each "
                              "trajectory 180 + 360 + 600 would count the same updates three "
                              "times"),
        "excluded_from_this_ceiling": [
            "one update of overshoot per budget crossing",
            "monitor GPU and wall seconds (measured per check, reported per trajectory)",
            "validation, generation, checkpoint I/O and whole-run elapsed time"],
        "declared_total_gpu_seconds": float(config["allocation"]["total_gpu_seconds"]),
        "declared_stage_allocations": {str(k): float(v) for k, v in
                                       config["allocation"]["stage_gpu_seconds"].items()},
    }
    document["reconciles"] = bool(
        abs(total - document["declared_total_gpu_seconds"]) < 1e-6
        and all(abs(stages[key]["allocation_gpu_seconds"]
                    - document["declared_stage_allocations"][key]) < 1e-6 for key in stages)
        and set(stages) == set(document["declared_stage_allocations"]))
    return document


def require_allocation_reconciles(config):
    """Fail closed when the config's declared totals disagree with the grid."""
    document = stage_allocation(config)
    computed = {key: value["allocation_gpu_seconds"] for key, value in document["stages"].items()}
    require(document["reconciles"],
            f"The declared allocation does not reconcile with the grid. Computed "
            f"{document['total_charged_training_gpu_seconds']} GPU seconds over {computed}; the "
            f"config declares {document['declared_total_gpu_seconds']} over "
            f"{document['declared_stage_allocations']}.")
    return document


# ---------------------------------------------------------------------------
# the primary statistic
# ---------------------------------------------------------------------------

def stratum_is_valid(entry):
    """A stratum contributes only if it has rows and both classes.

    A single-class stratum's average precision is its prevalence: a constant that
    differs per stratum, not a score. Averaging it in would move the macro mean
    for reasons that have nothing to do with the ranking.
    """
    if not entry or int(entry.get("n", 0)) == 0:
        return False
    return int(entry.get("positives", 0)) > 0 and int(entry.get("negatives", 0)) > 0


def stratum_mask(per_stratum, *, strata=SELECTION_STRATA):
    return tuple(name for name in strata if stratum_is_valid((per_stratum or {}).get(name)))


def macro_average_precision(per_stratum, *, strata=SELECTION_STRATA):
    """``(macro AP, worst-stratum AP, mask)`` over the valid declared strata."""
    mask = stratum_mask(per_stratum, strata=strata)
    require(mask, f"No declared stratum in {list(strata)} has rows in both classes")
    values = [float(per_stratum[name]["average_precision"]) for name in mask]
    return sum(values) / len(values), min(values), mask


def _reject_reserved_fields(record):
    offending = sorted(key for key in record
                       if set(key.lower().split("_")) & FORBIDDEN_SELECTION_TOKENS)
    require(not offending,
            f"Selection reads validation only; this endpoint record carries {offending}. Test and "
            "SPR outcomes select nothing in this protocol.")


def endpoint_is_eligible(record):
    """``(eligible, reasons)`` for one seed's endpoint at one nominal budget."""
    reasons = []
    if not record.get("reached"):
        reasons.append("budget_not_reached")
    if not record.get("gate_passed"):
        reasons.append("likelihood_gate")
    if not record.get("diversity_eligible"):
        reasons.append("diversity_gates")
    return (not reasons), reasons


def select_coefficient(endpoints, *, objective, budget, seeds, strata=SELECTION_STRATA):
    """The prespecified rule, for one objective at one exact nominal budget.

    ``endpoints`` is a flat list of per-seed endpoint records. Only records whose
    ``objective`` and ``nominal_budget`` match are considered; an arm missing a
    seed is ineligible, not partially scored.
    """
    seeds = [int(seed) for seed in seeds]
    considered, ineligible, candidates = {}, {}, []
    for record in endpoints:
        _reject_reserved_fields(record)
        if record.get("objective") != objective:
            continue
        if abs(float(record["nominal_budget"]) - float(budget)) > 1e-9:
            continue
        considered.setdefault(record["arm_id"], {})[int(record["seed"])] = record
    for name in sorted(considered):
        rows = considered[name]
        missing = [seed for seed in seeds if seed not in rows]
        if missing:
            ineligible[name] = {"reason": "missing_seeds", "seeds": missing}
            continue
        problems = {}
        for seed in seeds:
            ok, reasons = endpoint_is_eligible(rows[seed])
            if not ok:
                problems[str(seed)] = reasons
        if problems:
            ineligible[name] = {"reason": "ineligible_seeds", "seeds": problems}
            continue
        macro, worst, masks = [], [], set()
        for seed in seeds:
            value, low, mask = macro_average_precision(rows[seed].get("val_strata"),
                                                       strata=strata)
            macro.append(value)
            worst.append(low)
            masks.add(mask)
        require(len(masks) == 1,
                f"{name}: the valid stratum mask differs across seeds ({sorted(masks)}). The "
                "mask is a property of the fixed validation rows, so this is a bug, not a "
                "candidate to drop a stratum for.")
        candidates.append({
            "arm_id": name, "objective": objective, "nominal_budget": float(budget),
            "coefficients": rows[seeds[0]].get("coefficients"),
            "mean_macro_average_precision": sum(macro) / len(macro),
            "mean_worst_stratum_average_precision": sum(worst) / len(worst),
            "per_seed_macro_average_precision": {str(seed): value
                                                 for seed, value in zip(seeds, macro)},
            "strata": list(next(iter(masks)))})
    result = {"objective": objective, "nominal_budget": float(budget),
              "rule": SELECTION_RULE, "strata": list(strata),
              "seeds_required": seeds, "candidates": candidates,
              "ineligible": ineligible, "selection_inputs": ["validation"],
              "selected": None}
    if not candidates:
        result["reason"] = ("none eligible: no coefficient configuration has a reached, gate- and "
                            "diversity-eligible endpoint at this exact budget for every seed")
        return result
    masks = {tuple(candidate["strata"]) for candidate in candidates}
    require(len(masks) == 1,
            f"{objective} at {budget}: candidates disagree on the valid stratum mask "
            f"({sorted(masks)}). Selection aborts rather than dropping a stratum for one "
            "candidate; the mask is a property of the fixed validation rows.")
    best = min(candidates, key=lambda entry: (-entry["mean_macro_average_precision"],
                                              -entry["mean_worst_stratum_average_precision"],
                                              entry["arm_id"]))
    result["selected"] = best["arm_id"]
    result["selected_record"] = best
    return result


def select_all(endpoints, *, objectives, budgets=BUDGETS, seeds, strata=SELECTION_STRATA):
    """The full prespecified selection table: one winner per objective per budget."""
    return {str(float(budget)): {objective: select_coefficient(
        endpoints, objective=objective, budget=budget, seeds=seeds, strata=strata)
        for objective in sorted(objectives)} for budget in budgets}


def matched_control_deltas(endpoints, *, control_objective="continued_sft", budgets=BUDGETS,
                           controls=None):
    """Seed-level deltas against the matched continued-SFT endpoint. Descriptive only.

    Matched means the same seed and the **same** nominal budget. A trajectory that
    never reached that budget has no endpoint there and contributes no delta; the
    rolling last-passing state is never substituted for one.

    Two things are reported apart, because they answer different questions:

    ``rows``
        every raw matched difference, each carrying its own eligibility flags. A
        raw delta describes what the ranking did; it says nothing about whether
        the endpoint was admissible under the prespecified rule, and an arm that
        breached the gate or failed the diversity gates can still sit at the top
        of this list.
    ``eligible_rows``
        the subset where **both** sides were reached, gate-eligible and
        diversity-eligible. This is the only subset a claim of prespecified
        improvement may be read from.

    ``controls`` supplies the matched control records when they were fitted in an
    earlier stage; without it, the controls are taken from ``endpoints``.
    """
    pool = list(controls) if controls is not None else list(endpoints)
    control = {(int(r["seed"]), float(r["nominal_budget"])): r for r in pool
               if r.get("objective") == control_objective and r.get("reached")}
    rows, unmatched = [], []
    for record in endpoints:
        if record.get("objective") == control_objective or not record.get("reached"):
            continue
        key = (int(record["seed"]), float(record["nominal_budget"]))
        if float(record["nominal_budget"]) not in [float(b) for b in budgets]:
            continue
        if key not in control:
            unmatched.append({"arm_id": record["arm_id"], "seed": key[0],
                              "nominal_budget": key[1],
                              "reason": ("no continued_sft endpoint was reached at this seed and "
                                         "budget, so there is nothing to compare against and "
                                         "nothing is substituted")})
            continue
        partner = control[key]
        mine, _, mask = macro_average_precision(record.get("val_strata"))
        theirs, _, control_mask = macro_average_precision(partner.get("val_strata"))
        require(mask == control_mask, "Matched comparison crossed different stratum masks")
        mine_ok, mine_reasons = endpoint_is_eligible(record)
        control_ok, control_reasons = endpoint_is_eligible(partner)
        rows.append({"arm_id": record["arm_id"], "seed": key[0], "nominal_budget": key[1],
                     "macro_average_precision": mine,
                     "control_macro_average_precision": theirs,
                     "delta": mine - theirs, "control_arm_id": partner["arm_id"],
                     "control_reused_from_stage": partner.get("reused_from_stage"),
                     "eligible": bool(mine_ok and control_ok),
                     "arm_ineligible_reasons": mine_reasons,
                     "control_ineligible_reasons": control_reasons})
    ordered = sorted(rows, key=lambda row: (row["arm_id"], row["seed"], row["nominal_budget"]))
    document = {
        "rows": ordered,
        "eligible_rows": [row for row in ordered if row["eligible"]],
        "unmatched": sorted(unmatched, key=lambda row: (row["arm_id"], row["seed"])),
        "controls_available": bool(control),
        "matched_on": "same seed, same exact nominal budget",
        "interpretation": ("a descriptive validation-ranking difference. A positive delta is "
                           "not a confirmatory affinity claim and is not evidence of improved "
                           "binding; no assay endpoint enters this table."),
        "eligibility_note": ("`rows` is descriptive and includes ineligible endpoints with their "
                             "reasons. Only `eligible_rows` -- both sides reached, gate-eligible "
                             "and diversity-eligible at the exact prespecified budget -- can "
                             "support a claim about the prespecified comparison."),
    }
    if not control:
        document["unavailable_reason"] = (
            "no matched continued_sft endpoint exists at any seed and budget in this stage, so "
            "every comparison is unavailable. None is approximated from a different budget, a "
            "different seed or a rolling last-passing state.")
    return document


# ---------------------------------------------------------------------------
# the stage freeze
# ---------------------------------------------------------------------------

def stage_marker_name(stage):
    return f"stage{int(stage)}_complete.json"


def require_previous_stage(directory, stage, *, identity, root):
    """Stage N refuses to start without a verified frozen stage N-1.

    "Verified" means the marker parses, carries the guarded schema and the right
    stage number, matches this campaign's **whole** identity -- inherited parents,
    pinned sources, device and batch size included, not just the revision block --
    names its artifacts, and those artifacts still hash to what it recorded.

    The control block is verified the same way when it exists. When it does not,
    the marker must say so explicitly and say why: a stage whose every
    ``continued_sft`` trajectory stopped is a real outcome that the later stages
    inherit as "no matched control", and the difference between that and a marker
    that simply forgot to record its controls has to be visible here.
    """
    from pathlib import Path
    if int(stage) <= 1:
        return None
    path = Path(directory) / stage_marker_name(int(stage) - 1)
    require(path.is_file(),
            f"Stage {stage} requires a verified {path.name}; run and freeze stage {stage - 1} "
            "first. Stage order is a prerequisite, not a suggestion.")
    from .her2_runtime import load_json
    marker = load_json(path)
    require(marker.get("schema_version") == GUARDED_SELECTION_SCHEMA,
            f"{path} does not carry the guarded freeze schema {GUARDED_SELECTION_SCHEMA!r}")
    require(marker.get("stage") == int(stage) - 1, f"{path} is not the stage {stage - 1} marker")
    require(marker.get("stage_marker") == GUARDED_STAGE,
            f"{path} does not carry the guarded stage marker")
    require(marker["identity"].get("revision") == identity.get("revision"),
            f"{path} was frozen by different revision code or config; stage {stage} would be "
            "continuing somebody else's experiment")
    differing = campaign_identity_differences(marker["identity"], identity)
    require(not differing,
            f"{path} was frozen under a different campaign identity ({differing}). The revision "
            "hashes agree, so this is a change of parents, pinned sources, device or batch size: "
            f"stage {stage} would be continuing a different experiment under the same code.")
    for record in marker["artifacts"].values():
        verify_artifact(record, root)
    status = marker.get("control_status")
    require(isinstance(status, dict) and "available" in status,
            f"{path} records no control_status. Stages 2 and 3 reuse the stage-1 continued_sft "
            "control, so the marker must state whether one exists.")
    if status.get("available"):
        require(marker.get("control_artifacts"),
                f"{path} claims a control is available but names no continued_sft control "
                "artifacts; stages 2 and 3 reuse them and must verify the bytes")
        for record in marker["control_artifacts"].values():
            verify_artifact(record, root)
        require(marker.get("control_endpoints"),
                f"{path} names control artifacts but carries no control endpoint records; the "
                "later stages reuse the measured endpoints and must not re-measure them")
    else:
        require(status.get("reason") and status.get("trajectories"),
                f"{path} reports no available control without saying which trajectories failed to "
                "produce one and why; an absent control is evidence and must be carried forward")
    return marker


def reused_control_endpoints(marker, *, root, source_stage):
    """The verified stage-1 control endpoints, for reuse as stage 2/3 matched controls.

    Reuse means *reuse*: the bytes are re-verified and the already-measured
    validation metrics are carried forward under a flag saying which stage paid
    for them. Re-fitting or re-scoring them here would spend a second budget on a
    control that was fitted once, and the allocation arithmetic counts it once.
    """
    if marker is None:
        return [], {"available": False,
                    "reason": "this stage fits its own continued_sft control; nothing is reused"}
    status = dict(marker.get("control_status") or {})
    if not status.get("available"):
        return [], dict(status, available=False,
                        note=("the predecessor stage produced no matched control endpoint; the "
                              "comparisons that would have used it are reported unavailable"))
    for record in marker["control_artifacts"].values():
        verify_artifact(record, root)
    # The stage that FITTED the control keeps the attribution as the marker travels
    # forward: stage 3 inherits it through stage 2, and it was still paid for once.
    rows = [dict(row, reused_from_stage=int(row.get("reused_from_stage") or source_stage))
            for row in marker["control_endpoints"]]
    return rows, {"available": True, "source_stage": int(source_stage),
                  "fitted_in_stage": min((row["reused_from_stage"] for row in rows),
                                         default=int(source_stage)),
                  "endpoints": len(rows),
                  "artifacts_verified": sorted(marker["control_artifacts"]),
                  "note": ("the continued_sft control is fitted once, in stage 1. These endpoint "
                           "metrics are the ones measured there; no budget is charged twice and "
                           "nothing is re-measured.")}


def control_status_document(*, control_trajectories, control_endpoints, source_stage):
    """Whether a matched control endpoint exists for this stage, and if not, why.

    An all-stopped control is not a missing input. It is the outcome that the
    gate stopped every continued-SFT trajectory, and it belongs in the marker with
    its stop reasons so a later stage inherits the fact rather than an absence.
    """
    document = {
        "available": bool(control_endpoints),
        "objective": "continued_sft",
        "source_stage": int(source_stage),
        "endpoints": len(control_endpoints),
        "trajectories": {name: {"status": entry.get("status"),
                                "stop_reason": entry.get("stop_reason"),
                                "updates": entry.get("updates"),
                                "budgets_reached": sorted(entry.get("budgets") or {}),
                                "budgets_not_reached": sorted(entry.get("budgets_not_reached")
                                                              or {}),
                                "last_passing": entry.get("last_passing")}
                         for name, entry in sorted(control_trajectories.items())}}
    if not document["available"]:
        document["reason"] = (
            "no continued_sft trajectory reached a nominal budget under a passing gate, so this "
            "stage has no matched control endpoint. The stopped control trajectories and their "
            "stop reasons are recorded above; no endpoint is fabricated, no rolling last-passing "
            "state is promoted into one, and every comparison that needed a control is reported "
            "unavailable.")
    return document


def inherited_control_status(marker, *, control_endpoints, stage):
    """Carry the ORIGINAL control status forward, trajectories and stop reasons intact.

    Stage 2 and stage 3 do not fit a control, so there is no local trajectory to
    rebuild a status from. Rebuilding it from the immediate predecessor's *fitted*
    trajectories produces an empty table at stage 3 -- the stage-2 marker has no
    ``continued_sft`` runs either -- which turns "every control trajectory stopped,
    here are the three stop reasons" into "no controls were mentioned" after two
    hops. So the predecessor's own control status is carried, and only the fields
    this stage actually re-established (availability, endpoint count) are restated.
    """
    previous = dict((marker or {}).get("control_status") or {})
    require(previous,
            f"Stage {stage} reuses an earlier control and its predecessor marker carries no "
            "control_status to carry forward; an absent control is evidence and must travel with "
            "the marker rather than being rebuilt from trajectories this stage never fitted.")
    document = dict(previous)
    document["available"] = bool(control_endpoints) and bool(previous.get("available"))
    document["endpoints"] = len(control_endpoints or [])
    document["source_stage"] = int(previous.get("source_stage") or (int(stage) - 1))
    document["inherited_from_stage"] = int(stage) - 1
    document["carried_note"] = (
        "the continued_sft control was fitted once. This status -- including every control "
        "trajectory, its status, its stop reason and its reached/unreached budgets -- is the one "
        "the stage that fitted it recorded, carried forward unchanged rather than rebuilt from "
        "this stage's own (control-free) grid.")
    if not document["available"] and not document.get("reason"):
        document["reason"] = (
            "the stage that fitted the continued_sft control produced no matched control "
            "endpoint, so this stage inherits no control and every comparison that needed one is "
            "reported unavailable.")
    return document


# ---------------------------------------------------------------------------
# validated endpoints against what was actually reached
# ---------------------------------------------------------------------------

def endpoint_name(arm_id, seed, budget):
    """The single name an endpoint may carry for one (arm, seed, nominal budget)."""
    return f"{arm_id}_seed{int(seed)}_budget{int(float(budget))}"


def reached_endpoint_index(trajectories):
    """``{(trajectory, nominal budget): budget record}`` for every budget ACTUALLY reached.

    Built from the trajectory documents, which are the things that know what
    happened. The validated endpoint list is checked against this; it is never the
    other way round.
    """
    index = {}
    for name, document in sorted((trajectories or {}).items()):
        for budget, record in sorted((document.get("budgets") or {}).items()):
            index[(name, float(budget))] = record
    return index


def require_exact_endpoint_coverage(endpoints, reached, *, where):
    """Exactly one validated endpoint per reached trajectory-budget. No extras, no gaps.

    Four failures this makes impossible, each of which was reproducible before it
    existed: a validated document with its endpoints deleted freezing as though
    the stage had none; a foreign checkpoint substituted into an endpoint with its
    own new hash; an endpoint relabelled to a budget the trajectory never reached;
    and the same endpoint counted twice. The trajectory's own budget record is the
    authority for the checkpoint path and hash, and the endpoint must agree with it
    field for field.
    """
    seen = {}
    for record in endpoints:
        trajectory = record.get("trajectory")
        require(trajectory,
                f"{where}: a validated endpoint names no trajectory, so it cannot be tied to a "
                "run that produced it")
        budget = float(record["nominal_budget"])
        key = (trajectory, budget)
        require(key not in seen,
                f"{where}: two validated endpoints claim {trajectory} at {budget} s. One reached "
                "budget is one endpoint.")
        require(key in reached,
                f"{where}: an endpoint claims {trajectory} at {budget} s, which this stage's "
                "trajectory documents do not record as a reached budget. An unreached or "
                "undeclared budget has no checkpoint and cannot be validated into one.")
        expected_trajectory = f"{record['arm_id']}_seed{int(record['seed'])}"
        require(expected_trajectory == trajectory,
                f"{where}: endpoint {record.get('name')} carries arm {record['arm_id']} and seed "
                f"{record['seed']} while claiming trajectory {trajectory}")
        expected_name = endpoint_name(record["arm_id"], record["seed"], budget)
        require(record.get("name") == expected_name,
                f"{where}: endpoint {record.get('name')!r} is named for other coordinates than "
                f"the ones it carries ({expected_name!r})")
        require(bool(record.get("reached")),
                f"{where}: endpoint {record.get('name')} is recorded as not reached and is still "
                "in the validated endpoint list")
        budget_record = reached[key]
        require(record.get("checkpoint") == budget_record.get("checkpoint")
                and record.get("checkpoint_sha256") == budget_record.get("checkpoint_sha256"),
                f"{where}: endpoint {record.get('name')} names checkpoint "
                f"{record.get('checkpoint')} ({record.get('checkpoint_sha256')}) and the "
                f"trajectory recorded {budget_record.get('checkpoint')} "
                f"({budget_record.get('checkpoint_sha256')}). These are not the weights that run "
                "wrote at that budget.")
        seen[key] = record
    missing = sorted(f"{name} at {budget} s" for name, budget in reached if (name, budget)
                     not in seen)
    require(not missing,
            f"{where}: these reached budgets have no validated endpoint: {missing}. A reached "
            "endpoint that is missing from the validated document is missing evidence, not an "
            "absence of results.")
    return seen


#: Statuses a trajectory document can end in. All four mean nothing is still
#: running; only two of them advance the scientific stage.
TERMINAL_STATUSES = ("completed", "stopped", "incomplete", "failed")
#: The two outcomes this protocol prespecified: the trajectory reached every
#: declared budget, or the gate stopped it. An artificial ``max_updates`` cap
#: (``incomplete``) and an artifact/IO or step failure (``failed``) are neither.
#: Their evidence is kept and inspectable; what they may not do is be frozen into a
#: stage marker that the next stage reads as "the grid ran".
ADVANCING_STATUSES = ("completed", "stopped")
#: The stop reasons the gate itself produces. A ``stopped`` trajectory carrying
#: anything else did not stop for a declared condition, and a stage that accepted
#: it would be advancing on an undeclared stopping rule.
GATE_STOP_REASONS = ("parent_relative_likelihood_breach", "nonfinite_validation_score",
                     "gate_stop")


def stage_complete_document(*, stage, identity, trajectories, artifacts, control_artifacts,
                            selection, allocation, expected_trajectories, budgets,
                            control_status=None, control_endpoints=None, objectives=(),
                            git_commit=None):
    """The marker stage N+1 verifies. Every declared arm-seed appears, stopped ones included.

    Four completeness checks:

    * the trajectory set is **exactly** the declared arm x seed grid. A stage that
      fitted a subset of its grid does not freeze, and a stage carrying a
      trajectory the grid does not declare does not freeze either;
    * every trajectory either completed its declared budgets or stopped for a
      declared gate condition. An artificial ``max_updates`` cap and an
      artifact/IO failure are terminal and are kept as evidence, but neither
      advances the stage: a cap is not a prespecified likelihood stop;
    * every declared budget of every trajectory is accounted for exactly once --
      reached with a checkpoint, or recorded as not reached, never both and never
      neither, and never at a budget the stage did not declare;
    * every declared objective is enumerated whether or not any of its endpoints
      survived. An objective whose trajectories all stopped is reported with "none
      eligible", not omitted from the table.
    """
    expected = sorted(expected_trajectories)
    require(expected,
            "A stage freeze needs the declared arm x seed grid to check itself against")
    present = sorted(trajectories)
    missing = sorted(set(expected) - set(present))
    extra = sorted(set(present) - set(expected))
    require(not missing,
            f"These declared trajectories have no result: {missing}. The stage grid is "
            f"{len(expected)} arm-seed runs and {len(present)} are present; a stage is frozen "
            "when its grid has run, not when some of it has.")
    require(not extra,
            f"These trajectories are not in the declared stage grid: {extra}. A result that the "
            "protocol did not declare does not enter a freeze.")
    terminal = {name: entry.get("status") for name, entry in sorted(trajectories.items())}
    unfinished = sorted(name for name, status in terminal.items()
                        if status not in TERMINAL_STATUSES)
    require(not unfinished,
            f"These trajectories are not terminal: {unfinished}. A stage is frozen only when "
            "every trajectory has stopped or finished; a running one has no result yet.")
    blocked = {name: status for name, status in terminal.items()
               if status not in ADVANCING_STATUSES}
    require(not blocked,
            f"These trajectories neither completed their declared budgets nor stopped for a "
            f"declared gate condition: {blocked}. An artificial update cap is not a prespecified "
            "likelihood stop and an artifact or step failure is not a result; their evidence stays "
            "on disk and is inspectable, but the stage does not advance on it.")
    undeclared = {name: entry.get("stop_reason")
                  for name, entry in sorted(trajectories.items())
                  if entry.get("status") == "stopped"
                  and entry.get("stop_reason") not in GATE_STOP_REASONS}
    require(not undeclared,
            f"These trajectories are marked stopped under a reason the gate does not declare: "
            f"{undeclared}. The declared conditions are {list(GATE_STOP_REASONS)}; a stop for "
            "anything else is not a prespecified outcome.")
    declared_budgets = [float(budget) for budget in budgets]
    require(declared_budgets, "A stage freeze needs the declared budgets")
    declared_keys = {str(budget) for budget in declared_budgets}
    for name, entry in sorted(trajectories.items()):
        reached_keys = set(entry.get("budgets") or {})
        unreached_keys = set(entry.get("budgets_not_reached") or {})
        both = sorted(reached_keys & unreached_keys)
        require(not both,
                f"{name}: budgets {both} are recorded as reached AND as not reached. One "
                "trajectory cannot have two verdicts at one budget.")
        outstanding = sorted(declared_keys - (reached_keys | unreached_keys))
        require(not outstanding,
                f"{name}: budgets {outstanding} are neither reached nor recorded as not reached. "
                "Every declared budget is accounted for in a terminal trajectory; silence is not "
                "a verdict.")
        foreign = sorted((reached_keys | unreached_keys) - declared_keys)
        require(not foreign,
                f"{name}: budgets {foreign} are recorded but not declared by this stage "
                f"({sorted(declared_keys)}). An endpoint at an undeclared budget is not part of "
                "this protocol.")
        if entry.get("status") == "completed":
            require(not unreached_keys,
                    f"{name} is marked completed with budgets {sorted(unreached_keys)} not "
                    "reached. `completed` means every declared budget was reached under a passing "
                    "gate.")
    return {"schema_version": GUARDED_SELECTION_SCHEMA, "stage": int(stage),
            "stage_marker": GUARDED_STAGE,
            "identity": identity, "git_commit": git_commit,
            "declared_trajectories": expected,
            "declared_objectives": sorted(objectives),
            "declared_budgets": declared_budgets,
            "trajectories": {name: {
                "status": entry.get("status"), "stop_reason": entry.get("stop_reason"),
                "updates": entry.get("updates"), "checks": entry.get("checks"),
                "attempted_updates": entry.get("attempted_updates"),
                "budgets_reached": sorted(entry.get("budgets") or {}),
                "budgets_not_reached": sorted(entry.get("budgets_not_reached") or {}),
                "last_passing": entry.get("last_passing"),
                "cost": entry.get("cost")} for name, entry in sorted(trajectories.items())},
            "artifacts": artifacts, "control_artifacts": control_artifacts,
            "control_status": control_status,
            "control_endpoints": list(control_endpoints or []),
            "selection": selection, "selection_rule": SELECTION_RULE,
            "allocation": allocation,
            "diversity_gates": dict(DIVERSITY_GATES),
            "selection_read_test_or_assay": False,
            "prior_exposure": PRIOR_EXPOSURE}


#: Not "assays unseen". The 2026-09-18 test and SPR results were already inspected
#: and published in reference/her2-posttrain.md, and the Absci audit reads KD cells
#: by design. Neither selects a hyperparameter here, and that is the whole claim.
PRIOR_EXPOSURE = {
    "claim": "no reserved test label or assay outcome enters any selection in this revision",
    "not_claimed": "that these assays are unseen",
    "already_inspected": [
        "the 2026-09-18 reserved-test ranking results, reported in reference/her2-posttrain.md",
        "the independent SPR correlations on 152 finite-KD designs, reported in the same file",
        "the Absci KD cells, which the availability audit in this revision reads on purpose"],
    "consequence": ("those results are follow-up evidence, not virgin confirmatory data. A "
                    "validation-selected arm that later ranks well on them confirms less than a "
                    "prospective test would."),
}


def guarded_freeze_document(*, stage, identity, config, trajectories, endpoints, selection,
                            artifacts, control_artifacts, allocation, root, objectives=(),
                            control_status=None, control_endpoints=None, controls=None,
                            git_commit=None):
    """The guarded stage freeze: bytes re-verified, every declared trajectory enumerated."""
    for record in list(artifacts.values()) + list(control_artifacts.values()):
        verify_artifact(record, root)
    budgets = [float(b) for b in config["budgets_gpu_seconds"]]
    document = stage_complete_document(
        stage=stage, identity=identity, trajectories=trajectories, artifacts=artifacts,
        control_artifacts=control_artifacts, selection=selection, allocation=allocation,
        expected_trajectories=expected_trajectories(config, stage), budgets=budgets,
        control_status=control_status, control_endpoints=control_endpoints,
        objectives=objectives, git_commit=git_commit)
    document["endpoints"] = endpoints
    document["matched_control_deltas"] = matched_control_deltas(
        endpoints, budgets=budgets,
        controls=controls if controls is not None else list(endpoints))
    document["budgets_gpu_seconds"] = budgets
    document["strata"] = {"selection": list(SELECTION_STRATA), "reported": list(REPORTED_STRATA)}
    assert_cannot_unlock(document)
    return document
