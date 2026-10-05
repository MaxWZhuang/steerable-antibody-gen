"""Per-route GPU feasibility qualification for the native DPLM routes, in its own new run root.

What this is: a bounded, honest check that ONE DPLM route can be built, profiled, smoke-trained,
resumed and decoded on the real card, recorded in a run root that belongs to that check alone.
What it is not: a training launch, a reference-parity claim, a route selection, or any part of
the five-way competition. A ``qualified`` verdict says only that this route, at these settings,
passed these phases on this GPU on this day.

Isolation, in order of precedence
---------------------------------
* The run root is ``<artifact>/runs/<revision>/<run_root_leaf>`` and ``<revision>`` must start
  ``joint_qual_``, so it can never be ``t1_rev01``, ``dplm_rev01`` or ``joint_rev01``.
* Every refusal (a root that already holds anything, reserved-opening evidence, a held or
  unobservable lock) happens BEFORE any file is written and before any model or CUDA work; the
  refusal itself writes nothing. The checks are repeated inside the lock.
* ``effective_config.json`` is written only after the root and the lock have been checked, because
  writing it first would corrupt another run's root.
* Stage FUNCTIONS are called directly, never ``run_stage``, so no stage-completion record exists.
  Two PROVENANCE records are written on purpose, because they say what was qualified: the source
  snapshot (``phase_snapshot`` writes ``spec.FREEZE_MARKER``, i.e. ``source_snapshot.json``) and,
  once acquire has run, the external-source marker (``spec.EXTERNAL_MARKER``). ``seal`` re-verifies
  both. No PRODUCTION marker is created -- no revision, resolved-protocol, stage-completion,
  frozen-recipe or reserved-opening record -- and ``seal`` fails the run if one appears.
* Outside the run root the qualification writes only what the existing machinery already writes
  for any flight: the shared source-reader cache (``<artifact>/cache/sources``, created by
  ``services.build_sources``) and ``<artifact>/tmp`` (created by ``prepare_environment``). It
  reads the shared data, cache and dependency directories.

Device honesty
--------------
``qualified`` needs CUDA: the requested device is ``cuda``, CUDA is available, every task row at the
SELECTED microbatch shape was measured on it, ran every requested step, and had its NVML framebuffer
signal read and accepted (larger refused shapes are kept as evidence, not held against it). A CPU run can pass every
phase and its verdict is ``cpu_check_only``; it never emits a GPU-qualified record.

Refusal versus failure
----------------------
``refused_by_hardware`` is a capacity fact: a ``torch.cuda.OutOfMemoryError`` in a model phase, or a
profile with no accepted microbatch because a spill signal or an out-of-memory row said so.
Everything else -- missing or corrupt source data, a non-finite loss or gradient, a decode that
moved a scaffold position, a dispatch that ran the wrong objective, an unmeasurable signal,
insufficient task coverage -- is ``failed``, because it is a code or data defect and not a
statement about the card.
"""
from __future__ import annotations

import copy
import math
import re
from collections import Counter
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from . import her2_routes_campaign as campaign
from . import her2_routes_spec as spec
from . import her2_support_paths as paths
from .her2_replay_campaign import CampaignLock, lock_state, owner_identity
from .her2_runtime import require

VERDICT_QUALIFIED = "qualified"
VERDICT_REFUSED = "refused_by_hardware"
VERDICT_FAILED = "failed"
VERDICT_CPU_ONLY = "cpu_check_only"

QUALIFICATION_JSON = "qualification.json"
EFFECTIVE_CONFIG = "effective_config.json"

#: A qualification revision can never collide with a flight revision.
REVISION_PATTERN = re.compile(r"^joint_qual_[A-Za-z0-9][A-Za-z0-9_.-]*$")

DPLM_ROUTES = tuple(route for route in spec.COMPETITION_ROUTES if route.startswith("dplm_"))

PHASE_ORDER = ("environment", "snapshot", "acquire", "preflight", "split", "sources", "profile",
               "smoke", "decode")

LOCK_FILES = ("campaign.lock", "campaign.owner.json", "campaign.owners.jsonl")
#: Markers a qualification run must never create, and must refuse to find.
RESERVED_MARKERS = ("reserved_opening.json", "frozen_recipe.json")
FROZEN_MARKERS = (spec.FREEZE_MARKER, spec.EXTERNAL_MARKER, spec.REVISION_MARKER,
                  "resolved_protocol.json")
PREVIOUS_MARKERS = (QUALIFICATION_JSON, EFFECTIVE_CONFIG)

DECODE_DRAWS = 4
DECODE_BATCH = 2
DECODE_SEED = 20260929
#: The fresh-versus-recorded rescoring comparison uses the config's declared
#: ``tolerances.sum_log_probability_{atol,rtol}``; there is deliberately no qualification-only
#: constant, so this check can never be looser than the campaign's own definition of agreement.
TOLERANCE_KEYS = ("sum_log_probability_atol", "sum_log_probability_rtol")
#: The three acceptance signals ``FlightServices.profile`` records on every row.
SIGNAL_NAMES = ("driver_free_above_floor", "allocator_peak_within_budget",
                "step_time_not_degraded")


class QualificationRefused(ValueError):
    """The run root or lock cannot be used. Nothing was written."""


class PhaseFailure(Exception):
    """A phase found a code, data or evidence defect. Carries the evidence gathered so far."""

    def __init__(self, reason, evidence=None):
        super().__init__(reason)
        self.evidence = dict(evidence or {})


class PhaseRefusal(PhaseFailure):
    """A phase found a hardware capacity fact."""


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def json_safe(node):
    """Plain, finite JSON. Non-finite floats become a small marker so ``write_json`` never fails."""
    import numpy as np
    if isinstance(node, dict):
        return {str(key): json_safe(value) for key, value in node.items()}
    if isinstance(node, (list, tuple, set)):
        return [json_safe(value) for value in node]
    if isinstance(node, np.generic):
        return json_safe(node.item())
    if isinstance(node, np.ndarray):
        return json_safe(node.tolist())
    if isinstance(node, bool) or node is None or isinstance(node, (int, str)):
        return node
    if isinstance(node, float):
        if math.isfinite(node):
            return node
        return {"nonfinite": "nan" if math.isnan(node) else ("+inf" if node > 0 else "-inf")}
    if isinstance(node, Path):
        return str(node)
    return repr(node)


def classify_error(error):
    """``(status, reason, where)`` for an exception escaping a phase. OOM is the only capacity signal."""
    torch = sys.modules.get("torch")
    oom = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
    where = [f"{Path(frame.filename).name}:{frame.lineno} {frame.name}"
             for frame in traceback.extract_tb(error.__traceback__)[-4:]]
    text = f"{type(error).__name__}: {str(error)[:600]}"
    if oom is not None and isinstance(error, oom):
        return VERDICT_REFUSED, text, where
    return VERDICT_FAILED, text, where


class GpuProbe:
    """The two device facts the verdict depends on. Injectable so a CPU test cannot fake a GPU."""

    def available(self):
        import torch
        return bool(torch.cuda.is_available())

    def reading(self):
        from .her2_vram import vram_reading
        return vram_reading()


@dataclass
class Run:
    """What one qualification carries between phases."""

    context: object
    services: object
    route: str
    device: str
    gpu: object
    evidence: dict = field(default_factory=dict)

    @property
    def use_cuda(self):
        environment = self.evidence.get("environment") or {}
        return self.device.startswith("cuda") and bool(environment.get("cuda_available"))


def _pick(document, keys):
    return {key: document.get(key) for key in keys if key in document}


# ---------------------------------------------------------------------------
# effective config and root checks
# ---------------------------------------------------------------------------

def effective_config(document, *, route, source_sha256):
    """The joint config restricted to ONE route, with the source config's digest recorded.

    Restricting ``routes.enabled`` is what makes every stage function, ``dplm_variants`` and the
    training identity see only this route. ``config_path`` inside the document still names the
    joint config in the repository, which the source snapshot needs.
    """
    require(route in DPLM_ROUTES, f"{route!r} is not a native DPLM route; qualification is "
                                  f"defined for {DPLM_ROUTES}")
    enabled = list(document["routes"]["enabled"])
    require(route in enabled, f"{route} is not enabled in the source config ({enabled})")
    config = copy.deepcopy(document)
    config["routes"]["enabled"] = [route]
    config["qualification"] = {
        "route": route, "source_config": str(document["config_path"]),
        "source_config_sha256": str(source_sha256),
        "purpose": ("a per-route GPU feasibility check in its own run root. It is not a flight, "
                    "not a selection and not a training launch.")}
    return config


def root_blockers(run_root, *, inside_lock=False):
    """Every reason ``run_root`` cannot host a qualification. Read-only; ``[]`` means usable."""
    run_root = Path(run_root)
    reasons = []
    lock_path = run_root / campaign.LOCK_FILE
    if not inside_lock:
        observed = lock_state(lock_path)["state"]
        if observed in ("held", "inaccessible"):
            reasons.append(f"the campaign lock is {observed}: "
                           + ("a live writer owns this root" if observed == "held"
                              else "its state cannot be observed, so an unknown writer may"))
    if run_root.is_symlink() or (run_root.exists() and not run_root.is_dir()):
        reasons.append(f"{run_root} is not a plain directory")
        return reasons
    if not run_root.is_dir():
        return reasons
    names = sorted(entry.name for entry in run_root.iterdir())
    found = [name for name in names if name not in LOCK_FILES]
    if not found:
        return reasons
    labels = []
    if any(name in RESERVED_MARKERS for name in found):
        labels.append("reserved-opening or frozen-recipe evidence "
                      f"({[n for n in found if n in RESERVED_MARKERS]})")
    if any(name in FROZEN_MARKERS for name in found):
        labels.append(f"a frozen or completed run root ({[n for n in found if n in FROZEN_MARKERS]})")
    stages = run_root / campaign.STAGE_RECORDS
    if stages.is_dir() and any(stages.iterdir()):
        labels.append("completed stage records")
    if any(name in PREVIOUS_MARKERS for name in found):
        labels.append(f"a previous qualification ({[n for n in found if n in PREVIOUS_MARKERS]})")
    reasons.append("the run root already holds content, so it is not a new root: "
                   + (", ".join(labels) if labels else f"{found[:10]}")
                   + ". Choose a new --revision; a qualification never overwrites a root.")
    return reasons


# ---------------------------------------------------------------------------
# the phases
# ---------------------------------------------------------------------------

def phase_environment(run):
    available = bool(run.gpu.available())
    if run.device.startswith("cuda") and not available:
        raise PhaseFailure("CUDA was requested but is not available. A GPU qualification cannot "
                           "be produced on this host; --device cpu is a code check only.",
                           {"cuda_available": False, "device": run.device})
    use_cuda = run.device.startswith("cuda") and available
    return {"device": run.device, "cuda_available": available,
            "gpu_measurement": bool(use_cuda),
            "nvml_start": run.gpu.reading() if use_cuda else None}


def phase_snapshot(run):
    snapshot = spec.source_snapshot(run.context)
    return {**_pick(snapshot, ("snapshot_sha256", "file_count", "entry_points", "git_state")),
            "reused": bool(snapshot.get("reused"))}


def phase_acquire(run):
    result = campaign.stage_acquire(run.context, run.services)
    return _pick(result, ("external_files", "external_sha256", "closure_paths", "dplm"))


def native_evidence_findings(record, *, lora_rank=None):
    """Problems with one route's ``native_evidence`` block, from the fields it really emits.

    Read: ``parity_at_zero.bit_identical``, ``reference_parity_nonzero_tables.meaningful``,
    ``gradient.{trunk_reached, head_gradient_reaches_trunk, parameters_with_gradient,
    head_parameters_with_gradient, frozen_parameters}``, ``decode_probe.{canonical_only,
    immutable_scaffold_verified, trajectory_logprob}`` and ``parameters.{base, condition_tables,
    lora, total, trainable, frozen, lora_rank}``. The parameter block proves counts, not
    identities; the per-parameter check is made on the loaded model by the decode phase.
    """
    problems = []
    parity = record.get("parity_at_zero") or {}
    if parity.get("bit_identical") is not True:
        problems.append("the zero-table conditioned forward is not bit identical to the trunk")
    if (record.get("reference_parity_nonzero_tables") or {}).get("meaningful") is not True:
        problems.append("the perturbed-table reference parity was not meaningful")
    gradient = record.get("gradient") or {}
    for key in ("trunk_reached", "head_gradient_reaches_trunk"):
        if gradient.get(key) is not True:
            problems.append(f"gradient.{key} is not True")
    for key in ("parameters_with_gradient", "head_parameters_with_gradient"):
        if not int(gradient.get(key) or 0) > 0:
            problems.append(f"gradient.{key} is not positive")
    probe = record.get("decode_probe") or {}
    if probe.get("canonical_only") is not True or probe.get(
            "immutable_scaffold_verified") is not True:
        problems.append("the preflight decode probe did not report canonical-only immutable-scaffold")
    if probe.get("trajectory_logprob") is not None:
        problems.append("the decode probe reported a trajectory log probability; none exists")
    counts = record.get("parameters") or {}
    try:
        base, tables, lora = int(counts["base"]), int(counts["condition_tables"]), int(counts["lora"])
        total, trainable, frozen = (int(counts["total"]), int(counts["trainable"]),
                                    int(counts["frozen"]))
    except (KeyError, TypeError, ValueError):
        return problems + ["parameters block is missing a count"]
    if lora_rank is not None:
        if frozen != base:
            problems.append(f"LoRA route: {frozen} frozen parameters but the base holds {base}")
        if trainable != tables + lora or lora <= 0:
            problems.append(f"LoRA route: trainable {trainable} != tables {tables} + lora {lora}")
        if counts.get("lora_rank") != int(lora_rank):
            problems.append(f"LoRA rank {counts.get('lora_rank')} != configured {lora_rank}")
        if int(gradient.get("frozen_parameters") or 0) <= 0:
            problems.append("LoRA route: the gradient probe reports no frozen parameter")
        noop = record.get("lora_noop_parity") or {}
        if noop.get("max_abs_difference") != 0.0 or noop.get("meaningful") is not True:
            problems.append("the fresh LoRA adapter is not a meaningful exact no-op")
    else:
        if frozen != 0 or lora != 0 or trainable != total:
            problems.append(f"full-tune route: frozen {frozen}, lora {lora}, trainable "
                            f"{trainable} of {total}; every parameter must train")
    return problems


def phase_preflight(run):
    result = campaign.stage_preflight(run.context, run.services)
    document = paths.read_json(run.context.path("native_evidence.json"))
    record = document.get(run.route)
    if record is None:
        raise PhaseFailure(f"native_evidence.json has no block for {run.route}",
                           {"routes": sorted(document)})
    lora = run.services.route_block(run.route).get("lora")
    problems = native_evidence_findings(
        record, lora_rank=None if lora is None else int(lora["rank"]))
    evidence = {"capacity": result.get("capacity"),
                "parameters": record.get("parameters"),
                "gradient": _pick(record.get("gradient") or {},
                                  ("trunk_reached", "head_gradient_reaches_trunk",
                                   "parameters_with_gradient", "parameters_without",
                                   "head_parameters_with_gradient", "frozen_parameters")),
                "dtype_policy": record.get("dtype_policy"), "load": record.get("load"),
                "problems": problems}
    if problems:
        raise PhaseFailure("native evidence failed: " + "; ".join(problems), evidence)
    return evidence


def phase_split(run):
    result = campaign.stage_split(run.context, run.services)
    return _pick(result, ("counts", "reserved_status", "underpowered", "retention"))


def phase_sources(run):
    result = campaign.stage_sources(run.context, run.services)
    return _pick(result, ("sources", "observations", "unique_constructs"))


def joint_evidence_defects(row, requirement):
    """Why a local row does not show the executed work the declared recipe needs; ``[]`` if it does.

    ``requirement`` is :func:`~.her2_routes_data.local_joint_requirement` of the curriculum. The
    evidence is the row's ``interaction_evidence`` -- what the dispatched outcomes reported -- and
    NOT the generic ``interaction`` diagnostic key, which pairs carry too. A recipe that requests
    coupled quartets needs a profiled batch that held some; one that fits their joint needs at
    least one ``quartets_trained``. Old or forged rows that record none are refused, and so are
    counts that contradict each other or the recipe they claim to have run.

    A recipe that requests no coupled rows (pair-only, undeclared or unscheduled) asks for no
    evidence, so a row that records none -- or records ``None`` for a field it could not observe --
    is not refused. Evidence that IS recorded is still judged against the declared recipe: it must
    be well formed, agree with itself, carry the declared ``interaction_weight`` and
    ``max_joint_per_update``, and cannot report trained joints when the recipe runs none.
    """
    task = row.get("task")
    required = bool(requirement["coupled_rows_required"])
    evidence = row.get("interaction_evidence")
    if evidence is None or (required and not isinstance(evidence, dict)):
        if not required:
            return []
        return [f"{task}: the declared recipe requests coupled quartets but the row records no "
                "executed interaction evidence (interaction_evidence); a pair-only or older "
                "profile cannot certify it"]
    if not isinstance(evidence, dict):
        return [f"{task}: interaction_evidence is a {type(evidence).__name__}, not the record of "
                "executed work a profile row carries"]

    def recorded(name):
        # A required recipe needs every field; otherwise an absent or None field stays unrecorded.
        return required or evidence.get(name) is not None

    names = ("coupled_contrasts", "quartets_available", "quartets_trained")
    values = {name: evidence.get(name) for name in names}
    bad = [name for name, value in values.items()
           if recorded(name) and (isinstance(value, bool) or not isinstance(value, int)
                                  or value < 0)]
    if bad:
        return [f"{task}: interaction evidence {bad} is not a recorded non-negative integer "
                f"({values})"]
    weight, max_joint = evidence.get("interaction_weight"), evidence.get("max_joint_per_update")
    declared_joints = int(requirement["max_joint_per_update"])
    if ((recorded("interaction_weight")
         and (isinstance(weight, bool) or not isinstance(weight, (int, float))
              or float(weight) != float(requirement["interaction_weight"])))
            or (recorded("max_joint_per_update")
                and (isinstance(max_joint, bool) or not isinstance(max_joint, int)
                     or max_joint != declared_joints))):
        return [f"{task}: the row was profiled under interaction_weight {weight!r} and "
                f"max_joint_per_update {max_joint!r}, not the declared "
                f"{requirement['interaction_weight']:g} and {declared_joints}"]
    coupled, available, trained = (values[name] for name in names)

    def exceeds(smaller, larger):
        return smaller is not None and larger is not None and smaller > larger

    defects = []
    if required and coupled < 1:
        defects.append(f"{task}: the profiled local batch held no coupled contrast, but the "
                       "declared recipe requests quartets")
    if (exceeds(trained, available) or exceeds(available, coupled)
            or exceeds(trained, declared_joints)):
        defects.append(f"{task}: contradictory interaction evidence {values} "
                       f"(max_joint_per_update {declared_joints})")
    elif required and requirement["joint_required"] and trained < 1:
        defects.append(f"{task}: the recipe fits the quartet joint (interaction_weight "
                       f"{requirement['interaction_weight']:g}) but no joint executed "
                       f"(quartets_trained 0 of {available} available, "
                       f"{coupled} coupled contrast(s)); the profile timed "
                       "cheaper work than the run does")
    elif not requirement["joint_required"] and trained is not None and trained > 0:
        defects.append(f"{task}: {trained} joint(s) reported trained although "
                       "the declared recipe runs none")
    if requirement["mode"] == "pairs" and coupled is not None and coupled > 0:
        defects.append(f"{task}: the profiled local batch held {coupled} coupled contrast(s), but "
                       "the declared recipe is pair-only and draws none")
    return defects


def analyse_profile(profile, *, route, tasks, diffusion, gpu_required, curriculum=None):
    """``(status, reason, evidence)`` for one route's profile block. Pure; no device access.

    Order matters. A non-finite loss or gradient anywhere is a FAILURE even if another size ran
    out of memory, because it is a code defect and not a capacity fact. A profile with no accepted
    microbatch is a hardware refusal only if a row ran out of memory or a spill signal is False;
    signals that were merely unmeasurable fail the run instead.

    The SHAPE is judged too. A task row accepted at microbatch ``m`` must have measured exactly
    ``m`` rows -- a draw of four rows reported as an eight-row measurement times and sizes
    something else -- and the chosen microbatch must divide the recorded effective batch exactly.
    When the declared ``curriculum`` is supplied, the chosen microbatch must also carry its local
    recipe (an even size of at least four when it requests interactions), the profile must record
    its ``effective_batch`` (a raw positive integer; a float, bool or negative value is refused,
    never truncated), and -- when the recipe requests quartets -- the local row must show the work
    that EXECUTED (:func:`joint_evidence_defects`). Callers that pass no curriculum judge the
    measured shape only, which is how a reconstruction-only fixture stays distinct from a
    full-recipe qualification.
    """
    rows = list(profile.get("rows") or [])
    summary = [{"microbatch": row.get("microbatch"), "task": row.get("task"),
                "accepted": row.get("accepted"), "failure": row.get("failure"),
                "failure_detail": row.get("failure_detail"),
                "rows_measured": row.get("rows_measured"),
                "completed_steps": row.get("completed_steps"),
                "steps_requested": row.get("steps_requested"),
                "peak_allocated_mib": row.get("peak_allocated_mib"),
                "per_row_slowdown": row.get("per_row_slowdown"),
                "signals": row.get("signals"), "dispatched": row.get("dispatched"),
                "interaction_evidence": row.get("interaction_evidence"),
                "probe_device": row.get("probe_device")} for row in rows]
    evidence = {"chosen_microbatch": profile.get("chosen_microbatch"),
                "accepted_microbatches": profile.get("accepted_microbatches"),
                "tasks_profiled": profile.get("tasks_profiled"),
                "measured_on": profile.get("measured_on"), "rows": summary,
                "rejected_shapes": profile.get("rejected_shapes"),
                "training_mode": profile.get("training_mode"),
                "device_total_mib": profile.get("device_total_mib"),
                "gpu_measured": False}
    # Before ANY capacity conclusion, refusal included: an eval-mode profile disables the
    # per-layer checkpoints and dropout production runs, so its memory says nothing about training.
    mode = profile.get("training_mode")
    if (not isinstance(mode, dict)
            or any(mode.get(part) is not True for part in ("model", "native", "heads"))):
        return VERDICT_FAILED, (f"{route}: the profile does not record model, native and heads all "
                                f"in training mode ({mode!r}). It did not measure the production "
                                "step; this is a code/evidence failure, not a hardware refusal."), evidence
    nonfinite =[row for row in rows if str(row.get("failure") or "").startswith("nonfinite")]
    if nonfinite:
        return VERDICT_FAILED, (f"{route}: a non-finite loss or gradient in "
                                f"{[(r['microbatch'], r['task'], r['failure']) for r in nonfinite]}"
                                ". This is a code defect, not a capacity fact."), evidence
    for row in rows:
        if row.get("failure") is None and (row.get("dispatched") or {}).get(
                "scheduled_task") != row.get("task"):
            return VERDICT_FAILED, (f"{route}: the {row.get('task')} draw dispatched "
                                    f"{row.get('dispatched')}"), evidence
    chosen = profile.get("chosen_microbatch")
    if chosen is None:
        oom = [row for row in rows if str(row.get("failure") or "").startswith("OutOfMemory")]
        spilled = [row for row in rows if any(
            value is False for value in (row.get("signals") or {}).values())]
        if oom or spilled:
            return VERDICT_REFUSED, (f"{route}: no microbatch passed the declared spill signals "
                                     f"({len(oom)} out-of-memory row(s), {len(spilled)} row(s) with "
                                     "a failed signal). The route does not fit this card at the "
                                     "declared shapes."), evidence
        return VERDICT_FAILED, (f"{route}: no microbatch was accepted and no row ran out of memory "
                                "or failed a signal, so the memory signals were unmeasurable or the "
                                "tasks did not run; that is not a capacity refusal."), evidence
    # Judge the SELECTED shape only. A larger shape that was refused is capacity evidence, kept
    # below; it says nothing against a smaller shape the profile accepted.
    from .her2_vram import signals_accepted
    if isinstance(chosen, bool) or not isinstance(chosen, int) or chosen <= 0:
        return VERDICT_FAILED, (f"{route}: the chosen microbatch {chosen!r} is not a positive "
                                "integer"), evidence
    effective = profile.get("effective_batch")
    if effective is not None or curriculum is not None:
        from .her2_routes_data import validate_microbatch
        if effective is None:
            return VERDICT_FAILED, (f"{route}: the profile records no effective_batch, so the "
                                    f"chosen microbatch {chosen} cannot be judged against the "
                                    "declared recipe's divisibility"), evidence
        try:
            # The RAW value: a float, bool or negative effective_batch is invalid evidence and is
            # named as such, not quietly turned into "no divisibility check".
            validate_microbatch(
                chosen, effective_batch=effective, accumulation=profile.get("accumulation_steps"),
                curriculum=curriculum, label=f"{route}: the chosen microbatch")
        except ValueError as error:
            return VERDICT_FAILED, f"{route}: {error}", evidence
    evidence["shape_mismatches"] = [
        {"microbatch": row.get("microbatch"), "task": row.get("task"),
         "rows_measured": row.get("rows_measured")}
        for row in rows if row.get("accepted") is True and isinstance(row.get("microbatch"), int)
        and isinstance(row.get("rows_measured"), int) and row["rows_measured"] > 0
        and row["rows_measured"] != row["microbatch"]]
    at_shape = [row for row in rows if row.get("microbatch") == chosen]
    evidence["larger_shape_refusals"] = [
        entry for entry, row in zip(summary, rows)
        if isinstance(row.get("microbatch"), int) and row["microbatch"] > chosen
        and (row.get("accepted") is not True or row.get("failure") is not None)]
    accepted_sizes = profile.get("accepted_microbatches")
    if accepted_sizes is not None and chosen not in accepted_sizes:
        return VERDICT_FAILED, (f"{route}: chosen microbatch {chosen} is not among the accepted "
                                f"microbatches {accepted_sizes}"), evidence
    counted = Counter(str(row.get("task")) for row in at_shape)
    duplicated = sorted(task for task, count in counted.items() if count > 1)
    missing = sorted(set(tasks) - set(counted))
    unexpected = sorted(set(counted) - set(tasks))
    if (missing or duplicated or unexpected
            or sorted(profile.get("tasks_profiled") or []) != sorted(tasks)):
        return VERDICT_FAILED, (f"{route}: insufficient or ambiguous task coverage at microbatch "
                                f"{chosen}: missing {missing}, duplicated {duplicated}, "
                                f"unexpected {unexpected}"), evidence
    at_chosen = {row["task"]: row for row in at_shape}
    defects = []
    if curriculum is not None:
        from .her2_routes_data import TASK_LOCAL, local_joint_requirement
        if TASK_LOCAL in at_chosen:
            defects.extend(joint_evidence_defects(at_chosen[TASK_LOCAL],
                                                  local_joint_requirement(curriculum)))
    for task, row in at_chosen.items():
        measured = row.get("rows_measured")
        if isinstance(measured, bool) or not isinstance(measured, int) or measured <= 0:
            defects.append(f"{task}: no positive measured row count ({measured!r})")
        elif measured != chosen:
            defects.append(f"{task}: the draw held {measured} row(s) but the selected "
                           f"microbatch is {chosen}; the profile measured a different shape "
                           "than the one it accepted")
        if row.get("failure") is not None:
            defects.append(f"{task}: failure {row.get('failure')}")
        if row.get("accepted") is not True:
            defects.append(f"{task}: not accepted")
        steps, requested = row.get("completed_steps"), row.get("steps_requested")
        counts = (steps, requested)
        if any(isinstance(count, bool) or not isinstance(count, int) for count in counts):
            defects.append(f"{task}: completed_steps {steps!r} / steps_requested {requested!r} "
                           "are not both recorded as integers")
        elif requested < 2 or steps != requested:
            # one warmup plus at least one measured step, and every requested step finished
            defects.append(f"{task}: completed_steps {steps} of steps_requested {requested} "
                           "(need steps_requested >= 2 and every requested step completed)")
        signals = row.get("signals") or {}
        absent = [name for name in SIGNAL_NAMES if name not in signals]
        if absent:
            defects.append(f"{task}: signal(s) {absent} not recorded")
        elif not signals_accepted(signals, on_gpu=bool(gpu_required)):
            defects.append(f"{task}: signals {signals}")
    if defects:
        return VERDICT_FAILED, (f"{route}: the chosen microbatch {chosen} does not hold for every "
                                f"task: {'; '.join(defects)}"), evidence
    if diffusion:
        from .her2_routes_data import TASK_RECONSTRUCTION
        keys = set((at_chosen[TASK_RECONSTRUCTION].get("dispatched") or {}).get(
            "diagnostic_keys") or ())
        if not {"mean_timestep", "num_timesteps"} <= keys:
            return VERDICT_FAILED, (f"{route}: the reconstruction task did not run the diffusion "
                                    f"objective (diagnostic keys {sorted(keys)})"), evidence
    if gpu_required:
        # An NVML reading the card cannot have produced (once ~4,194,280 MiB free on a 4096 MiB
        # card) cannot qualify the selected shape. Larger refused rows are not judged here.
        total = profile.get("device_total_mib")
        implausible = []
        for row in at_shape:
            free = (row.get("vram") or {}).get("nvml_fb_free_mib")
            valid_total = (isinstance(total, (int, float)) and not isinstance(total, bool)
                           and math.isfinite(total) and total > 0)
            valid = (valid_total and isinstance(free, (int, float)) and not isinstance(free, bool)
                     and math.isfinite(free) and 0 <= free <= total)
            if not valid and free is not None:
                implausible.append((row.get("microbatch"), row.get("task"), free))
        if implausible or (at_shape and not (isinstance(total, (int, float))
                                             and not isinstance(total, bool)
                                             and math.isfinite(total) and total > 0)):
            return VERDICT_FAILED, (f"{route}: the selected shape carries an impossible NVML free-memory "
                                    f"reading or no recorded device total ({total!r} MiB): "
                                    f"{implausible[:4]}"), evidence
        weak =[(row["microbatch"], row["task"]) for row in at_shape
                if not str(row.get("probe_device") or "").startswith("cuda")
                or (row.get("vram") or {}).get("nvml_fb_free_mib") is None
                or (row.get("signals") or {}).get("driver_free_above_floor") is not True
                or (row.get("signals") or {}).get("allocator_peak_within_budget") is not True]
        if weak or not (profile.get("measured_on") or {}).get("cuda"):
            return VERDICT_FAILED, (f"{route}: the profile carries no accepted CUDA measurement "
                                    f"({weak[:4]}); it cannot support a GPU verdict."), evidence
        evidence["gpu_measured"] = True
    return None, None, evidence


def phase_profile(run):
    from .her2_routes_data import TASKS
    profiles, schedule = campaign.profile_routes(run.context, run.services, routes=[run.route])
    plain = campaign._plain(profiles)
    paths.write_json(run.context.path(spec.PROFILE_JSON), plain)
    # selected_shapes reads profile.json on every call (it no longer remembers a fallback taken
    # before the file existed), so the forecast and the smoke phase below use the shape this
    # profile measured and accepted without anything having to be invalidated.
    config = getattr(run.services, "config", None)
    status, reason, evidence = analyse_profile(
        plain[run.route], route=run.route, tasks=TASKS,
        diffusion=bool(run.services.uses_diffusion_reconstruction(run.route)),
        gpu_required=run.use_cuda,
        curriculum=dict(config.get("curriculum") or {}) if isinstance(config, dict) else None)
    if status is not None:
        cls = PhaseRefusal if status == VERDICT_REFUSED else PhaseFailure
        raise cls(reason, evidence)
    forecast = run.services.forecast(profiles, schedule=schedule,
                                     ledger_sample=run.services.ledger_sample())
    paths.write_json(run.context.path(spec.FORECAST_JSON), campaign._plain(forecast))
    return {**evidence, "forecast": _pick(forecast, ("total_hours", "with_reserve_hours"))}


def phase_smoke(run):
    result = run.services.smoke(run.route, directory=run.context.path("smoke", run.route))
    result = campaign._plain(result)
    paths.write_json(run.context.path("qualification_smoke.json"), result)
    observed = result.get("observed") or {}
    evidence = {"checks": result.get("checks"), "failures": result.get("failures"),
                "tasks_observed": observed.get("tasks"),
                "objectives_observed": observed.get("objectives"),
                "trajectory": _pick(result.get("trajectory") or {},
                                    ("identical", "compared_steps", "max_abs_difference")),
                "note": ("six seeded updates need not reach every task; per-task coverage is "
                         "established by the profile's direct per-task dispatch")}
    if not result.get("passed"):
        raise PhaseFailure(f"smoke failed: {result.get('failures')}", evidence)
    if len(observed.get("objectives") or ()) <= 1:
        raise PhaseFailure("smoke ran a single objective", evidence)
    return evidence


def inspect_trainability(model, *, lora):
    """Per-parameter proof from the real loaded model: base frozen (LoRA) and adapters training."""
    problems = {"base_trainable": [], "adapter_frozen": []}
    counts = {"base": 0, "adapter": 0}
    for name, parameter in model.named_parameters():
        adapter = name.rsplit(".", 1)[-1].startswith("lora_") or name.startswith("condition_")
        counts["adapter" if adapter else "base"] += int(parameter.numel())
        if lora and not adapter and parameter.requires_grad:
            problems["base_trainable"].append(name)
        if (adapter or not lora) and not parameter.requires_grad:
            problems["adapter_frozen"].append(name)
    return {"lora": bool(lora), "base_parameters": counts["base"],
            "adapter_parameters": counts["adapter"],
            "base_parameters_requiring_grad": problems["base_trainable"][:5],
            "trainable_parameters_frozen": problems["adapter_frozen"][:5],
            "passed": not problems["base_trainable"] and not problems["adapter_frozen"]
            and (counts["adapter"] > 0 or not lora)}


def declared_rescore_tolerance(config):
    """``(atol, rtol)`` from the config's ``tolerances`` block; anything else is refused."""
    block = (config or {}).get("tolerances")
    require(isinstance(block, dict), "the config declares no tolerances block, so the decode "
                                     "rescoring has no declared tolerance to be held to")
    values = []
    for key in TOLERANCE_KEYS:
        value = block.get(key)
        require(isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and value >= 0,
                f"tolerances.{key} must be a finite non-negative number, got {value!r}")
        values.append(float(value))
    return values[0], values[1]


def verify_native_decode(services, route, model, *, sites, condition, draws=DECODE_DRAWS,
                         batch_size=DECODE_BATCH, seed=DECODE_SEED):
    """Run the native decoder and check what it ACTUALLY returned against what it was given.

    ``model.iterative_decode`` is wrapped for the duration of the call: every invocation
    snapshots its input tokens and slots BEFORE the decoder runs (on the CPU, whatever the model's
    device) and records its returned native tokens, so a decoder that mutated a scaffold slot --
    even one that mutated the input and the output identically -- left a mask, emitted a
    non-canonical token, skipped the denoiser or lied in its metadata is caught here rather than
    trusted. The snapshot is compared with an independently built serialization of the same
    design sites, so the slots are checked against the site mapping and not merely counted. A
    forward hook on ``model.native`` counts the denoiser calls independently of what the decoder
    reports. The fresh rescore is held to the config's declared tolerances.
    """
    import numpy as np
    import torch
    from ..models import dplm_policy as dplm
    from . import her2_routes_data as data_lib
    from . import her2_routes_generate as generate
    from .her2_data import CANONICAL
    from .her2_routes_serial import ablang_serialization, her2_construct

    atol, rtol = declared_rescore_tolerance(services.config)
    sites = tuple(sites)
    reference = ablang_serialization(
        data_lib.with_sites(her2_construct(services.scaffold(),
                                           services.config["her2"]["wild_type_core"],
                                           record_id="expected"), sites),
        masked_sites=sites, order=sites)
    expected_tokens = torch.tensor(list(reference.tokens), dtype=torch.long)
    expected_slots = torch.tensor(list(reference.query_slots), dtype=torch.long)

    def snapshot(value):
        """An independent CPU copy; ``as_tensor`` alone would share memory with a numpy batch."""
        return torch.as_tensor(value).detach().to(device="cpu", dtype=torch.long).clone()

    captures = []
    original = model.iterative_decode

    def capturing(tokens, query_slots, **kwargs):
        before_tokens, before_slots = snapshot(tokens), snapshot(query_slots)
        result = original(tokens, query_slots, **kwargs)
        captures.append({
            "tokens": before_tokens, "slots": before_slots,
            "input_unchanged": bool(torch.equal(snapshot(tokens), before_tokens)
                                    and torch.equal(snapshot(query_slots), before_slots)),
            "native": snapshot(result.native_tokens), "canonical": snapshot(result.canonical),
            "forward_calls": int(result.forward_calls), "steps": int(result.steps)})
        return result

    calls = [0]
    handle = model.native.register_forward_hook(
        lambda *_: calls.__setitem__(0, calls[0] + 1))
    had = "iterative_decode" in vars(model)
    prior = vars(model).get("iterative_decode")
    object.__setattr__(model, "iterative_decode", capturing)
    try:
        result = generate.fill_batch(
            services, route, model, draws=int(draws), sites=tuple(sites), condition=condition,
            strategy=generate.ORDER_DIFFUSION, seed=int(seed), batch_size=int(batch_size))
    finally:
        handle.remove()
        if had:
            object.__setattr__(model, "iterative_decode", prior)
        else:
            vars(model).pop("iterative_decode", None)

    decode_steps = int(model.diffusion.decode_steps)
    batches = -(-int(draws) // int(batch_size))
    canonical_ids = torch.tensor(dplm.DPLM_CANONICAL_IDS, dtype=torch.long)
    to_canonical = torch.tensor(dplm.NATIVE_TO_CANONICAL, dtype=torch.long)
    flags = dict.fromkeys(("scaffold_identical", "design_slots_masked_in_input",
                           "design_tokens_canonical", "canonical_agrees_with_native_tokens",
                           "slots_match_sites", "input_matches_serialization",
                           "decoder_left_input_unmodified", "decoder_step_counts"), True)
    produced = []
    for capture in captures:
        tokens, slots, native = capture["tokens"], capture["slots"], capture["native"]
        native_in = dplm.ablang_to_dplm(tokens)
        flags["decoder_left_input_unmodified"] &= capture["input_unchanged"]
        flags["input_matches_serialization"] &= bool(
            tokens.dim() == 2 and tokens.shape[1] == expected_tokens.shape[0]
            and torch.equal(tokens, expected_tokens.expand_as(tokens)))
        flags["slots_match_sites"] &= bool(
            slots.dim() == 2 and slots.shape == (tokens.shape[0], expected_slots.shape[0])
            and torch.equal(slots, expected_slots.expand_as(slots)))
        if (native.shape != native_in.shape or slots.dim() != 2
                or slots.shape[0] != native_in.shape[0]):
            flags["scaffold_identical"] = False
            continue
        rows = torch.arange(native_in.shape[0])[:, None].expand_as(slots)
        scaffold = torch.ones_like(native_in, dtype=torch.bool)
        scaffold[rows, slots] = False
        flags["scaffold_identical"] &= bool(torch.equal(native[scaffold], native_in[scaffold]))
        flags["design_slots_masked_in_input"] &= bool((native_in[rows, slots]
                                                       == dplm.DPLM_MASK_ID).all())
        decoded = native[rows, slots]
        flags["design_tokens_canonical"] &= bool(torch.isin(decoded, canonical_ids).all())
        flags["canonical_agrees_with_native_tokens"] &= bool(
            torch.equal(to_canonical[decoded], capture["canonical"]))
        flags["decoder_step_counts"] &= (capture["steps"] == decode_steps
                                         and capture["forward_calls"] == decode_steps)
        try:
            produced.extend("".join(CANONICAL[int(index)] for index in row)
                            for row in capture["canonical"].tolist())
        except IndexError:
            flags["canonical_agrees_with_native_tokens"] = False
    diagnostics = dict(result.diagnostics)
    scores = (None if result.log_probabilities is None
              else np.asarray(result.log_probabilities, dtype=np.float64))
    fresh_difference, fresh_excess, fresh_agrees = None, None, False
    if scores is not None and scores.shape == (int(draws),) and np.isfinite(scores).all():
        with torch.no_grad():
            fresh = services.window_scores(route, model, list(result.windows),
                                           condition=condition, batch_size=32,
                                           sites=sites).detach().cpu().numpy()
        fresh = np.asarray(fresh, dtype=np.float64)
        if fresh.shape == scores.shape and np.isfinite(fresh).all():
            deviation = np.abs(fresh - scores)
            # the same allclose form as compare_sum_log_probabilities: atol + rtol * |expected|
            allowance = atol + rtol * np.abs(scores)
            fresh_difference = float(deviation.max())
            fresh_excess = float((deviation - allowance).max())
            fresh_agrees = bool((deviation <= allowance).all())
    checks = {
        "strategy_is_native_diffusion": result.strategy == generate.ORDER_DIFFUSION,
        "window_count": len(result.windows) == int(draws),
        "windows_canonical_and_full_width": all(
            len(window) == len(tuple(sites)) and set(window) <= set(CANONICAL)
            for window in result.windows),
        "decoder_called_once_per_batch": len(captures) == batches,
        "windows_come_from_the_decoder": tuple(produced) == tuple(result.windows),
        **flags,
        "denoiser_calls_match_the_hook": calls[0] == int(result.forward_calls),
        "decoding_calls_are_batches_times_steps": (
            diagnostics.get("denoiser_forward_calls_decoding") == batches * decode_steps
            and diagnostics.get("denoiser_forward_calls_decoding", 0) > 0),
        "rescore_calls_are_batches_times_sites": (
            diagnostics.get("final_rescore_forward_calls") == batches * len(tuple(sites))
            and diagnostics.get("final_rescore_forward_calls", 0) > 0),
        "rescored_scores_finite_and_nonpositive": bool(
            scores is not None and scores.shape == (int(draws),) and np.isfinite(scores).all()
            and (scores <= 1e-6).all()),
        "fresh_rescoring_agrees": fresh_agrees,
    }
    failures = sorted(name for name, ok in checks.items() if not ok)
    return {"passed": not failures, "failures": failures, "checks": checks,
            "windows": list(result.windows),
            "log_probabilities": None if scores is None else scores.tolist(),
            "decode_steps": decode_steps, "batches": batches,
            "hook_forward_calls": calls[0], "reported_forward_calls": int(result.forward_calls),
            "max_rescore_difference": fresh_difference,
            "max_rescore_excess_over_allowance": fresh_excess,
            "rescore_tolerance": {"atol": atol, "rtol": rtol,
                                  "source": "config tolerances.sum_log_probability_{atol,rtol}"}}


def _release_device(device):
    import gc
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def phase_decode(run):
    """Native decode from BOTH smoke checkpoints: the uninterrupted run's and the resumed run's."""
    from . import her2_routes_generate as generate
    from . import her2_routes_serial as serial
    from .her2_routes_conditions import BIN_HIGH, Condition
    services, route = run.services, run.route
    strategy = services.route_block(route).get("gate_generation_strategy")
    require(strategy == generate.ORDER_DIFFUSION,
            f"{route}: gate_generation_strategy is {strategy!r}, not {generate.ORDER_DIFFUSION!r}")
    smoke_root = run.context.path("smoke", route)
    checkpoints = {"first": smoke_root / "first" / "last.pt",
                   "resumed": smoke_root / "resumed" / "last.pt"}
    absent = [label for label, path in checkpoints.items() if not path.is_file()]
    if absent:
        raise PhaseFailure(f"the smoke checkpoint(s) {absent} are absent under {smoke_root}")
    sites = serial.her2_core_edit_map()
    condition = Condition(family="affinity", bin=BIN_HIGH,
                          assay=services.config["her2"]["assay"])
    lora = services.route_block(route).get("lora")
    results, trainability = {}, None
    for label, path in checkpoints.items():
        model, heads, record = services.load_for_scoring(route, path)
        try:
            if trainability is None:
                trainability = inspect_trainability(model, lora=lora is not None)
            model.eval()
            results[label] = verify_native_decode(services, route, model, sites=sites,
                                                  condition=condition)
            results[label]["checkpoint_step"] = record.get("step")
        finally:
            del model, heads
            _release_device(run.device)
    evidence = {"strategy": strategy, "trainability": trainability,
                "decode": {label: {key: value for key, value in block.items()
                                   if key not in ("windows", "log_probabilities")}
                           for label, block in results.items()},
                "first_versus_resumed": {
                    "windows_identical": results["first"]["windows"] == results["resumed"]["windows"],
                    "windows_agreeing": sum(a == b for a, b in zip(results["first"]["windows"],
                                                                    results["resumed"]["windows"])),
                    "draws": len(results["first"]["windows"]),
                    "gated": False,
                    "note": ("recorded, not gated. CUDA determinism for the DPLM routes is an "
                             "open gap; all this shows is that decoding runs from the resumed "
                             "checkpoint.")}}
    problems = [f"{label}: {block['failures']}" for label, block in results.items()
                if not block["passed"]]
    if not trainability["passed"]:
        problems.append(f"trainability: {trainability}")
    if problems:
        raise PhaseFailure("native decode check failed: " + "; ".join(problems), evidence)
    return evidence


DEFAULT_PHASES = {"environment": phase_environment, "snapshot": phase_snapshot,
                  "acquire": phase_acquire, "preflight": phase_preflight, "split": phase_split,
                  "sources": phase_sources, "profile": phase_profile, "smoke": phase_smoke,
                  "decode": phase_decode}


def seal(run):
    """The end-of-run checks. A failure here invalidates the evidence whatever else passed."""
    problems = []
    context = run.context
    if run.evidence.get("snapshot"):
        try:
            spec.verify_source_snapshot(context, label="qualification end")
        except ValueError as error:
            problems.append(f"the source snapshot no longer matches the tree: {error}")
    if context.path(spec.EXTERNAL_MARKER).is_file():
        try:
            spec.verify_external_sources(context, label="qualification end")
        except ValueError as error:
            problems.append(f"the external sources changed: {error}")
    forbidden = [name for name in (*RESERVED_MARKERS, spec.REVISION_MARKER,
                                   "resolved_protocol.json")
                 if context.path(name).exists()]
    stages = context.path(campaign.STAGE_RECORDS)
    if stages.exists():
        forbidden.append(campaign.STAGE_RECORDS)
    if forbidden:
        problems.append(f"production marker(s) exist in a qualification root: {forbidden}")
    effective = context.path(EFFECTIVE_CONFIG)
    expected = (context.config.get("qualification") or {}).get("source_config_sha256")
    logical = context.config.get("config_path")
    if expected and logical and context.repository(logical).is_file() and \
            paths.sha256_file(context.repository(logical)) != expected:
        problems.append("the source config changed while it was being qualified")
    return {"problems": problems, "effective_config_sha256": (
        paths.sha256_file(effective) if effective.is_file() else None)}


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def _digest(path):
    path = Path(path)
    return paths.sha256_file(path) if path.is_file() else None


def gather_digests(run):
    context = run.context
    digests = {"effective_config": _digest(context.path(EFFECTIVE_CONFIG)),
               "source_config": (context.config.get("qualification") or {}).get(
                   "source_config_sha256"),
               "gpu_profile": _digest(context.path(spec.PROFILE_JSON)),
               "runtime_forecast": _digest(context.path(spec.FORECAST_JSON))}
    for name, key, reader in (
            ("snapshot", context.path(spec.FREEZE_MARKER), "snapshot_sha256"),
            ("external", context.path(spec.EXTERNAL_MARKER), "external_sha256")):
        digests[name] = (paths.read_json(key).get(reader) if key.is_file() else None)
    split = context.path(spec.SPLIT_FREEZE)
    digests["split_freeze"] = (paths.digest_document(paths.read_json(split))
                               if split.is_file() else None)
    try:
        from . import her2_routes_dplm_pins as pins_lib
        digests["pins"] = pins_lib.pins_digest(run.services.dplm_pins())
        digests["native_identity"] = run.services.native_identity(run.route)
    except Exception as error:                            # noqa: BLE001 - evidence, not control
        digests["pins"] = digests["native_identity"] = f"unavailable: {type(error).__name__}"
    return digests


def qualify_route(context, services, route, *, device, phases=None, gpu=None):
    """Run every phase for ``route`` in ``context``'s root and write ``qualification.json``.

    The caller holds the campaign lock and has already checked the root. ``phases`` maps a phase
    name to ``callable(run) -> evidence`` and overrides the default; the first phase that does not
    complete stops the run, later phases are recorded as skipped, and the ``seal`` checks and the
    terminal record are written regardless. Returns the record.
    """
    target = context.path(QUALIFICATION_JSON)
    if target.exists():
        raise QualificationRefused(f"{target} already exists; a qualification is never overwritten")
    resolved = {**DEFAULT_PHASES, **dict(phases or {})}
    unknown = sorted(set(resolved) - set(PHASE_ORDER) - {"seal"})
    require(not unknown, f"unknown qualification phase(s) {unknown}")
    run = Run(context=context, services=services, route=route, device=str(device),
              gpu=gpu or GpuProbe())
    started = paths.utc_now()
    records, stopped = {}, None
    interrupt = None
    for name in PHASE_ORDER:
        if stopped is not None:
            records[name] = {"status": "skipped",
                             "reason": f"the {stopped} phase did not complete"}
            continue
        clock = time.perf_counter()
        try:
            evidence = resolved[name](run)
            records[name] = {"status": "completed", "evidence": evidence}
            run.evidence[name] = evidence
        except PhaseRefusal as error:
            records[name] = {"status": VERDICT_REFUSED, "reason": str(error),
                             "evidence": error.evidence}
            stopped = name
        except PhaseFailure as error:
            records[name] = {"status": VERDICT_FAILED, "reason": str(error),
                             "evidence": error.evidence}
            stopped = name
        except BaseException as error:                    # noqa: BLE001 - recorded, then re-raised
            status, reason, where = classify_error(error)
            records[name] = {"status": status, "reason": reason, "where": where}
            stopped = name
            if not isinstance(error, Exception):
                interrupt = error
        records[name]["seconds"] = time.perf_counter() - clock
        if interrupt is not None:
            break
    sealed = {"problems": ["the seal did not run"]}
    try:
        sealed = resolved.get("seal", seal)(run)
    except Exception as error:                            # noqa: BLE001 - evidence
        sealed = {"problems": [f"the seal itself failed: {type(error).__name__}: {error}"]}
    statuses = [record["status"] for record in records.values()]
    if VERDICT_FAILED in statuses or sealed["problems"]:
        verdict = VERDICT_FAILED
    elif VERDICT_REFUSED in statuses:
        verdict = VERDICT_REFUSED
    elif not str(device).startswith("cuda"):
        verdict = VERDICT_CPU_ONLY
    elif (run.evidence.get("environment", {}).get("cuda_available") is True
          and (run.evidence.get("profile") or {}).get("gpu_measured") is True):
        verdict = VERDICT_QUALIFIED
    else:
        verdict = VERDICT_FAILED
    reason = next((record.get("reason") for record in records.values()
                   if record["status"] in (VERDICT_FAILED, VERDICT_REFUSED)), None)
    if reason is None and sealed["problems"]:
        reason = "; ".join(sealed["problems"])
    if reason is None and verdict == VERDICT_FAILED:
        reason = "the run completed without accepted CUDA profile evidence"
    use_cuda = run.use_cuda
    document = {
        "schema_version": spec.ROUTES_SCHEMA, "record_kind": "dplm_qualification",
        "verdict": verdict, "gpu_qualified": verdict == VERDICT_QUALIFIED, "route": route,
        "revision": context.revision, "device": str(device), "reason": reason,
        "phases": records, "seal": sealed, "digests": gather_digests(run),
        "nvml": {"start": (run.evidence.get("environment") or {}).get("nvml_start"),
                 "end": run.gpu.reading() if use_cuda else None} if use_cuda else None,
        "reserved_opened": context.path("reserved_opening.json").exists(),
        "does_not_establish": [
            "training has been launched", "pretrained numerical parity against the reference",
            "route quality, selection or any competition result",
            "CUDA determinism of resume or decoding"],
        "cpu_note": (None if str(device).startswith("cuda") else
                     "a CPU run is a code check. It never certifies a GPU fit."),
        "started_at": started, "finished_at": paths.utc_now()}
    try:
        paths.write_json(target, json_safe(document))
    except (TypeError, ValueError) as error:
        document = {"schema_version": spec.ROUTES_SCHEMA, "record_kind": "dplm_qualification",
                    "verdict": VERDICT_FAILED, "gpu_qualified": False, "route": route,
                    "reason": f"the record could not be serialised: {error}"}
        paths.write_json(target, document)
    if interrupt is not None:
        raise interrupt
    return document


def run_qualification(repository_root, *, config_path, artifact_root, revision, route, device,
                      build_services, phases=None, gpu=None, environment=None):
    """Check the root, take the lock, recheck, write the effective config, then qualify.

    Raises :class:`QualificationRefused` -- having written nothing -- for a bad revision, a root
    that already holds anything, reserved-opening evidence or a lock that is held or cannot be
    observed. ``build_services(context, device=...)`` is called only after all of that.
    """
    require(route in DPLM_ROUTES, f"{route!r} is not a native DPLM route ({DPLM_ROUTES})")
    if REVISION_PATTERN.match(str(revision)) is None:
        raise QualificationRefused(
            f"revision {revision!r} must match {REVISION_PATTERN.pattern}: a qualification root "
            "must never be the root of a flight revision")
    config_path = Path(config_path)
    document = spec.load_config(config_path)
    # ``ensure=False``: binds and disjointness-checks the root without creating it.
    probe = spec.resolve_context(repository_root, config_path=config_path,
                                 artifact_root=artifact_root, revision=revision, ensure=False)
    root = probe.run_root
    reasons = root_blockers(root)
    if reasons:
        raise QualificationRefused("; ".join(reasons))
    lock = CampaignLock(root / campaign.LOCK_FILE, owner=owner_identity())
    try:
        lock.acquire()
    except (OSError, ValueError) as error:
        raise QualificationRefused(f"the campaign lock could not be taken: {error}") from error
    try:
        reasons = root_blockers(root, inside_lock=True)
        if reasons:
            raise QualificationRefused("after taking the lock: " + "; ".join(reasons))
        config = effective_config(document, route=route,
                                  source_sha256=paths.sha256_file(config_path))
        effective = root / EFFECTIVE_CONFIG
        paths.write_json(effective, config)
        context = spec.resolve_context(repository_root, config_path=effective,
                                       artifact_root=artifact_root, revision=revision)
        if environment is not None:
            paths.write_json(context.path("environment.json"), json_safe(environment))
        services = build_services(context, device=device)
        return qualify_route(context, services, route, device=device, phases=phases, gpu=gpu)
    finally:
        lock.release()
