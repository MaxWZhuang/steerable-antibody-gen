"""The queue, the stages, the measured forecast, and the durable supervisor.

Stage order is fixed and deterministic:

``recover -> preflight -> geometry -> mixtures -> profile -> calibrate -> freeze
-> production -> audit -> couple -> report -> verify``

Each stage writes a stage record, and ``run_all`` advances through them without
any follow-up manual call. It may stop -- loudly, with evidence retained -- at a
**readiness gate** (a missing input, a changed source snapshot, insufficient
storage) or a **scientific feasibility gate** (the proximity geometry fails at
the declared radius, or no calibrated configuration qualifies). What it may not
do is skip a required stage or report a partial flight as complete: a stopped
run records which stages ran, which are blocked, and why, and the completion
check in :mod:`her2_nf_report` refuses to call that complete.

Two separations are load-bearing:

* **profiling then forecasting.** Every cost category -- profiling compute,
  gates, generation, the 50k scoring pass, stage-1 fits, comparators and storage
  I/O -- is measured separately and written as a measurement. The forecast reads
  those measurements. There is no code path that writes an estimated cost with
  ``measured: true``, and the resolved protocol refuses to leave the unresolved
  list while the forecast is unmeasured.
* **launch and completion.** A verified worker with a PID, a lock and advancing
  journals establishes that the flight is *running*. Scientific completion comes
  only from the required stages finishing, and the two are reported as different
  facts.
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from . import her2_nf_banks as nf_banks
from . import her2_nf_contract as contract
from . import her2_nf_mixture as mixture_lib
from . import her2_nf_monitor as monitor_lib
from . import her2_nf_objectives as nf_objectives
from . import her2_nf_spec as spec
from . import her2_nf_trajectory as trajectory_lib
from . import her2_support_paths as paths
from .her2_replay_campaign import CampaignLock, Heartbeat as _BaseHeartbeat
from .her2_replay_campaign import lock_state, owner_identity
from .her2_runtime import require

CAMPAIGN_SCHEMA = contract.NF_SCHEMA

STAGES = ("recover", "preflight", "geometry", "mixtures", "profile", "calibrate", "freeze",
          "production", "audit", "couple", "report", "verify")
EXTRA_STAGES = ("smoke", "status", "run-all")

#: Stages whose completion the flight's own verification requires.
REQUIRED_STAGES = STAGES

QUEUE_JSON = "queue.json"
CAMPAIGN_STATUS_JSON = "campaign_status.json"
STAGE_RECORDS = "stages"
PROFILE_JSON = "profile.json"
FORECAST_JSON = "runtime_forecast.json"
LOCK_FILE = "campaign.lock"
HEARTBEAT_JSON = "heartbeat.json"


class Heartbeat(_BaseHeartbeat):
    """Keep liveness current during long calls; journals remain the progress evidence."""

    def __init__(self, path, *, owner, every=15.0):
        super().__init__(path, owner=owner, every=every)
        self._guard = threading.Lock()
        self._fields = {}
        self._failure = None

    def beat(self, *, _tick=False, **fields):
        with self._guard:
            if self._failure is not None:
                raise RuntimeError("The background heartbeat failed") from self._failure
            if not _tick:
                self._fields.update(fields)
                self._fields["last_work_callback_at"] = paths.utc_now()
            return super().beat(**dict(self._fields, force=bool(fields.get("force")),
                                       liveness_tick=bool(_tick)))

    @contextmanager
    def running(self):
        stopped = threading.Event()

        def tick():
            while not stopped.wait(max(0.01, self.every)):
                try:
                    self.beat(_tick=True, force=True)
                except Exception as error:
                    self._failure = error
                    stopped.set()

        worker = threading.Thread(target=tick, name="her2-flight-heartbeat", daemon=True)
        worker.start()
        try:
            yield self
        finally:
            stopped.set()
            worker.join(timeout=1.0)


def _plain(node):
    """numpy scalars and arrays coerced to JSON types, recursively.

    ``paths.write_json`` uses ``allow_nan=False`` json, which raises on a
    ``numpy.float64``. Coercing at the write site means a stage record cannot
    fail at the end of an hours-long stage for a type nobody looked at.
    """
    import numpy as np
    if isinstance(node, dict):
        return {str(key): _plain(value) for key, value in node.items()}
    if isinstance(node, (list, tuple)):
        return [_plain(value) for value in node]
    if isinstance(node, np.generic):
        return node.item()
    if isinstance(node, np.ndarray):
        return node.tolist()
    return node


class ReadinessGate(RuntimeError):
    """A prerequisite is missing. The flight pauses; nothing scientific is claimed."""


class FeasibilityGate(RuntimeError):
    """A declared scientific criterion failed. Dependent stages stop; the rest still run."""


# ---------------------------------------------------------------------------
# the queue
# ---------------------------------------------------------------------------

@dataclass
class QueueRow:
    trajectory: str
    block: str
    regime: str
    arm: str
    task: str
    preservation: str
    coefficients: dict
    seed: int
    parent_id: str
    endpoints: tuple
    checkpoints: tuple
    status: str = "queued"
    reuse: object = None
    optional: bool = False
    note: str = ""

    def document(self):
        return {"trajectory": self.trajectory, "block": self.block, "regime": self.regime,
                "arm": self.arm, "task": self.task, "preservation": self.preservation,
                "coefficients": dict(sorted(self.coefficients.items())), "seed": int(self.seed),
                "parent_id": self.parent_id, "endpoints": list(self.endpoints),
                "checkpoints": list(self.checkpoints), "status": self.status,
                "reuse": self.reuse, "optional": bool(self.optional), "note": self.note,
                "arm_id": nf_objectives.arm_identifier(self.task, self.preservation,
                                                       self.coefficients),
                "uses_rejected": nf_objectives.TASK_OBJECTIVES[self.task].uses_rejected}


def build_queue(config, *, frozen=None, reuse_verified=False, include_simpo=False,
                require_frozen=True):
    """Every declared training cell, in a fixed execution order, with its status.

    The order is fixed before fitting so an early stop still leaves the matched
    control for every arm that ran, and so no ordering decision can follow a
    result: ascending block, then regime, then seed, then the declared arm order.

    ``require_frozen=False`` is for the stages that only need the *shape* of the
    queue -- counting updates for a forecast, listing rows for a status table --
    before calibration has frozen any coefficient. Those rows are marked
    ``coefficients_pending`` and production still refuses to start one.
    """
    frozen = dict(frozen or {})
    rows = []
    block_a = config["block_a"]
    endpoints_a = tuple(int(value) for value in block_a["endpoint_updates"])
    checkpoints_a = tuple(int(value) for value in block_a["checkpoint_updates"])
    for seed in block_a["parent_seeds"]:
        for arm in block_a["arms"]:
            if arm.get("optional") and not include_simpo:
                rows.append(_optional_row(arm, seed, endpoints_a, checkpoints_a))
                continue
            coefficients, pending = _resolve_coefficients(arm, frozen,
                                                          require_frozen=require_frozen)
            reuse = arm.get("reuse_historical")
            status = _coefficient_status(pending, frozen)
            note = ("calibration has not frozen " + ", ".join(pending) + " yet; this row is "
                    "counted for the forecast and is refused by production."
                    if pending else "")
            if reuse:
                status = "reuse_verified" if reuse_verified else "reuse_pending_parity"
                note = ("a historical trajectory is reused only after the stream digests and the "
                        "old/new update-kernel parity checks pass on identical state, cache and "
                        "stream. Until then it is not a factorial cell.")
            rows.append(QueueRow(
                trajectory=f"A_{arm['id']}_seed{seed}", block="A", regime="original_split",
                arm=arm["id"], task=arm["task"], preservation=arm["preservation"],
                coefficients=coefficients, seed=int(seed),
                parent_id=f"parent::policy_sft_seed{seed}",
                endpoints=endpoints_a, checkpoints=checkpoints_a, status=status,
                reuse=reuse, note=note))
    block_b = config["block_b"]
    endpoints_b = tuple(int(value) for value in block_b["endpoint_updates"])
    checkpoints_b = tuple(int(value) for value in block_b["checkpoint_updates"])
    for regime in block_b["regimes"]:
        for seed in block_b["parent_seeds"]:
            for arm in block_b_arms(config, frozen):
                coefficients, pending = _resolve_coefficients(arm, frozen,
                                                              require_frozen=require_frozen)
                rows.append(QueueRow(
                    trajectory=f"B_{regime}_{arm['id']}_seed{seed}", block="B", regime=regime,
                    arm=arm["id"], task=arm["task"], preservation=arm["preservation"],
                    coefficients=coefficients, seed=int(seed),
                    parent_id=f"parent::nf_{regime}_seed{seed}",
                    endpoints=endpoints_b, checkpoints=checkpoints_b,
                    status=_coefficient_status(pending, frozen),
                    note=("fixed 64k-preference-exposure comparison at u1000. This is a "
                          "deliberately smaller-duration block and is not evidence that new "
                          "parents converge on the old split's timescale.")))
    return rows


def _optional_row(arm, seed, endpoints, checkpoints):
    return QueueRow(
        trajectory=f"A_{arm['id']}_seed{seed}", block="A", regime="original_split",
        arm=arm["id"], task=arm["task"], preservation=arm["preservation"],
        coefficients=dict(arm.get("coefficients") or {}), seed=int(seed),
        parent_id=f"parent::policy_sft_seed{seed}", endpoints=tuple(endpoints),
        checkpoints=tuple(checkpoints), status="deferred_optional", optional=True,
        note=("optional extension. Deferred until the measured core and audit reserves are "
              "secured; it is listed so its absence is visible, not omitted."))


def _resolve_coefficients(arm, frozen, *, require_frozen=True):
    """``(coefficients, pending)``; ``pending`` names the calibrated values still missing."""
    coefficients = dict(arm.get("coefficients") or {})
    pending = []
    for key, source in dict(arm.get("frozen_coefficients") or {}).items():
        value = frozen.get(source)
        if value is None:
            require(not require_frozen or str(source).split(".")[0] in frozen.get("__unqualified__", []),
                    f"arm {arm['id']} needs the calibrated value {source!r}, which the freeze "
                    "does not carry. A production cell is never started at an invented "
                    "coefficient.")
            pending.append(str(source))
            continue
        coefficients[key] = float(value)
    return coefficients, sorted(pending)


def _coefficient_status(pending, frozen):
    if not pending:
        return "queued"
    if all(source.split(".")[0] in frozen.get("__unqualified__", []) for source in pending):
        return "no_qualified_configuration"
    return "coefficients_pending"


def block_b_arms(config, frozen):
    """The explicitly frozen DPO alternative; no choice uses B or E outcomes."""
    arms = [dict(arm) for arm in config["block_b"]["arms"]]
    if frozen.get("__block_b_dpo__", {}).get("family") == "dpo_tail":
        for arm in arms:
            if arm["task"] == "dpo" and arm["preservation"] == "fkl":
                arm.update(id="DPO_TAIL", preservation="tail",
                           frozen_coefficients={"beta": "dpo_beta", "lambda": "dpo_tail.lambda"})
    return arms


COMPARATORS = (
    {"id": "parent", "kind": "reference_policy",
     "note": "the stage-1 parent of each seed and regime"},
    {"id": "continued_sft_lambda0p1", "kind": "historical_control",
     "endpoints": (1000, 3750), "note": "reused historical control, not retrained to fill a table"},
    {"id": "cnn_single", "kind": "discriminative", "seeds": 3,
     "note": "all-class supervised training; a different training signal from high-only stage 1"},
    {"id": "cnn_ensemble", "kind": "discriminative",
     "note": "a separate entry from the single-model mean, never averaged with it"},
    {"id": "additive_linear", "kind": "discriminative",
     "note": "single-position indicators only"},
    {"id": "interaction_classifier", "kind": "discriminative",
     "note": "single-position plus all pairwise indicators; P(high) is a ranking score"},
    {"id": "mixture_alpha", "kind": "derived_evaluation",
     "note": "a training-free preservation competitor derived from saved scores"},
    {"id": "historical_ipo0", "kind": "historical_reference",
     "note": "reused only after parity; early checkpoints are MISSING, never reconstructed"},
    {"id": "historical_ipo_fkl", "kind": "historical_reference",
     "note": "as above"},
)


def comparator_rows():
    return [dict(entry, status="declared") for entry in COMPARATORS]


def frozen_coefficients(context):
    """The calibrated values, flattened to the ``family.key`` names the arms cite."""
    target = context.path("calibration_outcome.json")
    require(target.is_file(),
            f"{target} is absent: the production queue is built from frozen coefficients and "
            "there are none. Run the calibration stage first.")
    return _flatten_frozen(paths.read_json(target)["frozen"])


def _frozen_if_available(context):
    target = context.path("calibration_outcome.json")
    if not target.is_file():
        return {}
    return _flatten_frozen(paths.read_json(target)["frozen"])


def queue_for_reporting(context):
    """The full declared queue for a status or a report, before or after the freeze.

    Reporting must be able to list every declared row even when calibration
    refused a family, so it never demands a frozen coefficient. Production does,
    and builds its queue separately.

    The recorded preflight reuse decision is carried through. Production SKIPS a
    verified-reuse row without writing a terminal record, so a reporting queue
    that rebuilt those rows under the conservative default left them
    ``reuse_pending_parity`` forever: the report could never call the flight
    ready and ``production_has_no_pending_rows`` failed after a completed run.
    """
    preflight = read_stage_record(context.run_root, "preflight") or {}
    return build_queue(context.config, frozen=_frozen_if_available(context),
                       require_frozen=False,
                       reuse_verified=bool(preflight.get("reuse_verified")))


def _flatten_frozen(frozen):
    flat = {"__unqualified__": [], "__block_b_dpo__": dict(frozen.get("block_b_dpo") or {})}
    for name, block in dict(frozen).items():
        if isinstance(block, dict) and block.get("coefficients"):
            for key, value in block["coefficients"].items():
                flat[f"{name}.{key}"] = value
        elif not isinstance(block, dict):
            flat[name] = block
            if name == "dpo_beta" and block is None:
                flat["__unqualified__"].append(name)
        elif "coefficients" in block and block.get("coefficients") is None:
            flat["__unqualified__"].append(name)
    return flat


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------

#: Statuses that mean the row has not reached a terminal outcome. A deferred
#: optional arm and a verified historical reuse are NOT here: the first is a
#: declared omission and the second is an already-finished path.
NON_TERMINAL_STATUSES = ("queued", "interrupted", "coefficients_pending",
                         "reuse_pending_parity")


def campaign_state(run_root, queue):
    """Every queued row's status from disk, including missing, stopped and failed."""
    run_root = Path(run_root)
    rows, counts = [], {}
    for row in queue:
        directory = run_root / "trajectories" / row.trajectory
        terminal = trajectory_lib.read_terminal_status(directory)
        durable = (trajectory_lib.durable_progress(directory) if directory.is_dir()
                   else {"updates": 0, "endpoints_reached_updates": []})
        if terminal is not None:
            status = terminal["status"]
        elif row.status in ("deferred_optional", "reuse_pending_parity", "reuse_verified",
                            "coefficients_pending", "no_qualified_configuration"):
            status = row.status
        elif durable["updates"] > 0:
            status = "interrupted"
        else:
            status = "queued"
        counts[status] = counts.get(status, 0) + 1
        rows.append({**row.document(), "observed_status": status,
                     "journalled_updates": int(durable["updates"]),
                     "endpoints_reached": list(durable["endpoints_reached_updates"]),
                     "terminal": terminal})
    return {"schema_version": CAMPAIGN_SCHEMA, "record_kind": "campaign_state",
            "rows": rows, "counts": counts, "queued_total": len(queue),
            "completeness_rule": ("every declared row appears with a status. A missing, stopped, "
                                  "deferred or failed row is a status, never an omission, and a "
                                  "table with fewer rows than the queue is a defect.")}


def stage_record_path(run_root, stage):
    return Path(run_root) / STAGE_RECORDS / f"{stage}.json"


def write_stage_record(run_root, stage, document):
    target = stage_record_path(run_root, stage)
    record = dict(document, schema_version=CAMPAIGN_SCHEMA, record_kind="stage_record",
                  stage=str(stage), recorded_at=paths.utc_now())
    paths.write_json(target, record)
    return record


def read_stage_record(run_root, stage):
    target = stage_record_path(run_root, stage)
    return paths.read_json(target) if target.is_file() else None


def stage_summary(run_root, stages=STAGES):
    out = {}
    for stage in stages:
        record = read_stage_record(run_root, stage)
        out[stage] = {"ran": record is not None,
                      "status": (record or {}).get("status", "not_run"),
                      "reason": (record or {}).get("reason"),
                      "recorded_at": (record or {}).get("recorded_at")}
    return out


# ---------------------------------------------------------------------------
# profiling and the measured forecast
# ---------------------------------------------------------------------------

#: The cost categories that are measured apart. A shared total would hide which
#: half of the budget the monitoring amendment actually recovered.
COST_CATEGORIES = ("profiling_compute", "optimizer_update", "full_gate", "sentinel",
                   "generation_draw", "score_50k_pass", "stage1_fit", "comparator_fit",
                   "storage_io")

#: Queue statuses whose work the forecast has to pay for. A verified historical
#: reuse is already trained; a deferred optional arm is declared not to run.
FORECASTED_STATUSES = ("queued", "reuse_pending_parity", "coefficients_pending")


@dataclass
class Profile:
    """Measured per-unit costs. Nothing here may be written without measuring it."""

    measurements: dict = field(default_factory=dict)

    def record(self, category, *, seconds, units, detail=None):
        require(category in COST_CATEGORIES, f"Unknown cost category {category!r}")
        require(float(seconds) >= 0 and int(units) > 0,
                "a measurement needs nonnegative seconds and a positive unit count")
        self.measurements[category] = {
            "seconds": float(seconds), "units": int(units),
            "seconds_per_unit": float(seconds) / int(units),
            "detail": dict(detail or {}), "measured": True,
            "measured_at": paths.utc_now()}
        return self.measurements[category]

    def missing(self):
        return [name for name in COST_CATEGORIES if name not in self.measurements]

    def document(self):
        return {"schema_version": CAMPAIGN_SCHEMA, "record_kind": "measured_profile",
                "measurements": dict(self.measurements), "missing": self.missing(),
                "complete": not self.missing(),
                "rule": ("every entry is a measurement taken on this box. There is no path that "
                         "writes an estimated cost into this record, and a forecast built on an "
                         "incomplete profile is refused rather than extrapolated.")}


def runtime_forecast(profile, *, queue, banks_plan, reserve_hours=4.0, target_hours=24.0,
                     config_monitor=None):
    """Total measured forecast by category. Refuses to run on an incomplete profile."""
    missing = profile.missing()
    require(not missing,
            f"the runtime forecast needs measured costs for {missing}. Profiling runs first and "
            "then fills the forecast; an invented cost is never written in its place.")
    per = {name: profile.measurements[name]["seconds_per_unit"] for name in COST_CATEGORIES}
    # Every row that will actually be trained, including the ones whose
    # coefficient the calibration has not frozen yet: leaving those out would
    # understate the queue by exactly the arms the flight exists to run.
    counted = [row for row in queue if row.status in FORECASTED_STATUSES]
    updates = sum(max(row.endpoints) for row in counted)
    monitor = dict(config_monitor or {})
    full_checks = 0
    checks_by_block = {"A": 0, "B": 0}
    sentinels = 0
    for row in counted:
        interval = int(monitor.get("block_b_full_interval" if row.block == "B"
                                   else "full_interval", monitor_lib.FULL_INTERVAL))
        sentinel_interval = int(monitor.get("sentinel_interval",
                                            monitor_lib.SENTINEL_INTERVAL))
        horizon = max(row.endpoints)
        scheduled = {u for u in range(1, horizon + 1)
                     if u == 1 or u % interval == 0 or u in row.endpoints
                     or u in row.checkpoints}
        full_checks += len(scheduled)
        checks_by_block[row.block] += len(scheduled)
        sentinels += len({u for u in range(1, horizon + 1)
                          if u % sentinel_interval == 0} - scheduled)
    draws = int(banks_plan["total_draws"])
    scored_rows = int(banks_plan["scored_rows"])
    stage1_fits = int(banks_plan["stage1_fits"])
    comparator_sets = int(banks_plan["comparator_sets"])
    comparator_row_ratio = float(banks_plan["comparator_row_ratio"])
    seconds = {
        "optimizer_update": per["optimizer_update"] * updates,
        "full_gate": per["full_gate"] * max(full_checks, 0),
        "sentinel": per["sentinel"] * max(sentinels, 0),
        "generation_draw": per["generation_draw"] * draws,
        # score_50k_pass is measured PER ROW, so the forecast multiplies rows.
        # Recording an extrapolated 50k total as a measurement with units=1 put
        # the extrapolation inside the profile instead of inside the forecast.
        "score_50k_pass": per["score_50k_pass"] * scored_rows,
        "stage1_fit": per["stage1_fit"] * stage1_fits,
        "comparator_fit": per["comparator_fit"] * comparator_sets * comparator_row_ratio,
        "storage_io": per["storage_io"] * (draws + scored_rows),
        "profiling_compute": profile.measurements["profiling_compute"]["seconds"]}
    family_cost = profile.measurements["optimizer_update"]["detail"].get("seconds_per_update_by_family")
    if family_cost:
        seconds["optimizer_update"] = sum(max(row.endpoints) * family_cost[f"{row.task}_{row.preservation}"]
                                          for row in counted)
    gate_detail = profile.measurements["full_gate"]["detail"]
    if "block_b_seconds" in gate_detail:
        seconds["full_gate"] = (checks_by_block["A"] * per["full_gate"]
                                + checks_by_block["B"] * gate_detail["block_b_seconds"])
    stage_detail = profile.measurements["stage1_fit"]["detail"]
    if "measured_steps" in stage_detail:
        seconds["stage1_fit"] = per["stage1_fit"] * banks_plan["stage1_steps"]
    comparator = profile.measurements["comparator_fit"]["detail"]
    if "training_rows" in comparator:
        measured_rows = max(1, int(comparator["training_rows"]))
        seconds["comparator_fit"] = (
            (comparator["additive_seconds"] + comparator["pairwise_seconds"])
            * banks_plan["regularization_candidates"] * banks_plan["statistical_training_rows"] / measured_rows
            + comparator["cnn_seconds"] * banks_plan["cnn_training_rows_epochs"] / measured_rows)
    generation = profile.measurements["generation_draw"]["detail"]
    if "conditional_seconds" in generation:
        conditional_per_row = generation["conditional_seconds"] / profile.measurements["generation_draw"]["units"]
        seconds["additional_conditionals"] = conditional_per_row * (
            banks_plan["conditional_audit_rows"] + banks_plan["teacher_rows"])
    extra = profile.measurements["profiling_compute"]["detail"]
    if "dependence_probe_seconds" in extra:
        seconds["dependence_cpu"] = extra["dependence_probe_seconds"] / extra["dependence_probe_rows"] * (
            banks_plan["finalist_draws"] + banks_plan["screen_draws"])
        seconds["checkpoint_io"] = extra["resume_checkpoint_seconds"] * (
            2 * full_checks + len(counted) * 3 + banks_plan["stage1_steps"] / 100)
    seconds["calibration_reserved_cap"] = banks_plan.get("calibration_cap_seconds", 10800.0)
    total = sum(seconds.values())
    return {"schema_version": CAMPAIGN_SCHEMA, "record_kind": "runtime_forecast",
            "measured": True, "seconds": seconds, "hours": {k: v / 3600.0
                                                            for k, v in seconds.items()},
            "queue_updates": int(updates), "full_checks": int(max(full_checks, 0)),
            "full_checks_by_block": checks_by_block,
            "sentinels": int(max(sentinels, 0)), "generation_draws": draws,
            "scored_rows": scored_rows, "comparator_sets": comparator_sets,
            "comparator_row_ratio": comparator_row_ratio,
            "extrapolations": {
                "score_50k_pass": "measured seconds per scored row, times the declared row count",
                "stage1_fit": "measured seconds per step, times the declared steps per fit",
                "comparator_fit": ("measured seconds for one probe-sized SET (additive, pairwise "
                                   "and one CNN epoch), times the number of sets and the row "
                                   "ratio to the real populations")},
            "total_seconds": total, "total_hours": total / 3600.0,
            "reserve_hours": float(reserve_hours),
            "with_reserve_hours": total / 3600.0 + float(reserve_hours),
            "target_hours": float(target_hours),
            "sentinel_escalation_extra_seconds_upper_scenario": sentinels * (
                per["full_gate"] + 10000 * per["score_50k_pass"]),
            "limitations": ["row-linear extrapolation of classifier/CPU work; convergence may vary",
                            "calibration is a reserved cap, not a fabricated measured duration",
                            "base estimate excludes unscheduled sentinel escalation; upper scenario shown",
                            "all required bank sizes retained; stopped paths may reduce actual work"],
            "fits_target": bool(total / 3600.0 + float(reserve_hours) <= float(target_hours)),
            "drop_order": ["context-averaged training penalty (already deferred)",
                           "optional SimPO",
                           "Block-B continuation beyond u1000"],
            "basis": ("measured per-unit costs on this box multiplied by the declared queue. A "
                      "spreadsheet of optimistic update counts is not a runtime guarantee, and "
                      "24 hours is an ideal, not permission to relabel an incomplete flight as "
                      "complete.")}


def banks_plan(config, *, geometry=None):
    """Count all required jobs; historical controls/banks are verified reuse."""
    a, b = config["block_a"], config["block_b"]
    na = len(a["parent_seeds"])
    nb = len(b["regimes"]) * len(b["parent_seeds"])
    arms = [arm for arm in a["arms"] if not arm.get("optional")]
    a_updates = set(a["endpoint_updates"]) | set(a["checkpoint_updates"])
    a_updates.discard(0)
    b_updates = (set(b["endpoint_updates"]) | set(b["checkpoint_updates"])) - {0}
    a_models = na * (1 + 2 + sum(3 if arm.get("reuse_historical") else len(a_updates) for arm in arms))
    b_models = nb * (1 + len(b["arms"]) * len(b_updates))
    screens = a_models + b_models
    finalists = int(config["banks"]["finalist_models"])
    repeats = int(config["banks"]["finalist_banks_per_model"])
    finalist_rows = int(config["banks"]["finalist_rows"])
    screen_rows = int(config["banks"]["screen_rows"])
    final_rows = int(config["banks"]["final_preservation_rows"])
    replay = int(config["banks"]["replay_rows"])
    development = int(config["banks"]["monitor_rows"])
    mixture_models = na + nb
    finalist_draws = finalists * repeats * finalist_rows
    screen_draws = screens * screen_rows
    parent_draws = na * final_rows + nb * (replay + development + final_rows)
    total_draws = finalist_draws + screen_draws + parent_draws + mixture_models * screen_rows
    audit_models = na * (len(arms) * 2 + 2 + 1) + nb * (len(b["arms"]) + 1)
    counts = dict((geometry or {}).get("counts") or {})
    train_rows = int(counts.get("train_rows", 367042))
    challenge_rows = int(counts.get("purge_rows", 190951))
    high_rows = int(counts.get("retained_high_rows", 61019))
    val_rows = 78652
    e_rows = 3000
    trained_a = na * sum(not arm.get("reuse_historical") for arm in arms)
    in_fit = development * (trained_a * len(a_updates | {1}) + nb * len(b["arms"]) * len(b_updates | {1}))
    validation = a_models * val_rows + b_models * e_rows
    # Every generated bank is rescored; every named policy is audited separately.
    scored_rows = total_draws + audit_models * final_rows + validation + in_fit
    stage1_steps = nb * int(b["stage1"]["epochs"]) * ((high_rows + int(b["stage1"]["batch_size"]) - 1) // int(b["stage1"]["batch_size"]))
    return {"parents_block_a": na, "parents_block_b": nb,
            "finalist_models": finalists, "finalist_banks_per_model": repeats,
            "finalist_draws": finalist_draws, "screen_models": screens,
            "screen_draws": screen_draws, "parent_bank_draws": parent_draws,
            "mixture_models": mixture_models, "total_draws": total_draws,
            "registry_models": screens, "audit_models": audit_models,
            "validation_rows": val_rows, "in_fit_preservation_rows": in_fit,
            "scored_rows": scored_rows, "stage1_fits": nb, "stage1_steps": stage1_steps,
            "conditional_audit_rows": 2 * audit_models * final_rows + in_fit,
            "teacher_rows": nb * (replay + development),
            "statistical_training_rows": train_rows + len(b["regimes"]) * challenge_rows,
            "cnn_training_rows_epochs": nb * challenge_rows * int(config["comparators"]["cnn"]["epochs"]),
            "regularization_candidates": len(config["coupling"]["classifier"]["regularization_grid"]),
            "comparator_sets": len(b["regimes"]) + 1,
            "comparator_row_ratio": (train_rows + len(b["regimes"]) * challenge_rows) / max(1, int(config["profile"]["comparator_rows"])),
            "calibration_cap_seconds": float(config["calibration"]["gpu_hour_cap"]) * 3600,
            "requirement": "all required models and full bank sizes; verified historical banks reused",
            "population_basis": "certified geometry counts; defaults are recovered input counts for planning only"}


def build_services(context, *, device="cuda"):
    """The concrete runtime, imported lazily so orchestration is testable without torch."""
    from .her2_nf_services import FlightServices
    return FlightServices(context, device=device)


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

def stage_recover(context, services, **_):
    block = spec.recover(context)
    unrecovered = block["recovered"]["unrecovered"]
    return {"status": "completed", "unrecovered": unrecovered,
            "discrepancies": len(block["discrepancies"]["entries"]),
            "capacity": block["recovered"]["capacity"],
            "note": ("facts that could not be recovered are listed, not patched. A stage that "
                     "depends on one stops at its own gate.")}


def stage_preflight(context, services, **_):
    """Contract proof, storage, source snapshot verification, and the reuse parity gate."""
    from . import her2_replay_spec as replay_spec
    capacity = replay_spec.capacity_record(
        context.run_root, minimum_bytes=int(context.config["storage"]["min_free_bytes"]),
        required=False)
    if not capacity["sufficient"]:
        raise ReadinessGate(
            f"the run volume has {capacity['free_gib']:.1f} GiB free and this flight declares "
            f"{capacity['required_free_gib']:.1f} GiB. The required finalist bank counts are NOT "
            "reduced to fit; provision the approved volume and rerun.")
    contract_block = services.probability_contract()
    streams = services.verify_historical_streams()
    parity = services.verify_update_kernel_parity()
    reuse_ok = bool(streams["all_match"] and parity["passed"])
    return {"status": "completed", "capacity": capacity, "contract": contract_block,
            "historical_streams": streams, "update_kernel_parity": parity,
            "reuse_verified": reuse_ok,
            "reuse_consequence": (None if reuse_ok else
                                  "the historical arms become external references rather than "
                                  "factorial cells; the affected paths are retrained and the "
                                  "forecast is updated.")}


def stage_geometry(context, services, **_):
    """The exact purge, its certificate and the published feasibility table."""
    result = services.build_geometry()
    # The geometry table is published BEFORE any panel revision, whether or not
    # it passed. The manifest and the certificate are written only when there is
    # something to certify: a placeholder under those names would be read later
    # as a certified split.
    paths.write_json(context.path("geometry_feasibility.json"), _plain(result["feasibility"]))
    if result["feasibility"]["passed"]:
        paths.write_json(context.path("split_manifest.json"), _plain(result["manifest"]))
        paths.write_json(context.path("neighbor_certificate.json"), _plain(result["certificate"]))
    if not result["feasibility"]["passed"]:
        raise FeasibilityGate(
            "the proximity geometry fails the declared criteria at radius "
            f"{result['feasibility']['radius']}. The table is published above before any panel "
            "revision. The predeclared radius-1 design is a DIFFERENT challenge and is never "
            "silently substituted.")
    return {"status": "completed", "radius": result["feasibility"]["radius"],
            "panel_size": result["panel_size"], "counts": result["feasibility"]["counts"],
            "zero_violations": result["certificate"]["zero_violations"]}


def stage_mixtures(context, services, **_):
    """The inexpensive rescore, run before the training queue, with every alpha persisted."""
    result = services.mixture_rescore()
    paths.write_json(context.path("mixture_alpha_curve.json"), _plain(result))
    return {"status": "completed", "curves": len(result["curves"]),
            "grid": list(mixture_lib.ALPHA_GRID),
            "note": ("this may change the practical deployment recommendation even if the "
                     "objective/preservation comparisons remain valuable.")}


def stage_profile(context, services, **_):
    """Measure every cost category on this box, then write the forecast from it."""
    profile = services.profile()
    paths.write_json(context.path(PROFILE_JSON), profile.document())
    # Shape only: the forecast counts updates, and calibration has not frozen a
    # coefficient yet. Production builds the queue with require_frozen=True.
    preflight = read_stage_record(context.run_root, "preflight") or {}
    queue = build_queue(context.config, require_frozen=False,
                        reuse_verified=bool(preflight.get("reuse_verified")))
    forecast = runtime_forecast(profile, queue=queue,
                                banks_plan=banks_plan(context.config,
                                    geometry=read_stage_record(context.run_root, "geometry")),
                                reserve_hours=float(context.config["runtime"]["reserve_hours"]),
                                target_hours=float(context.config["runtime"]["target_hours"]),
                                config_monitor=context.config["monitor"])
    paths.write_json(context.path(FORECAST_JSON), forecast)
    return {"status": "completed", "total_hours": forecast["total_hours"],
            "with_reserve_hours": forecast["with_reserve_hours"],
            "fits_target": forecast["fits_target"],
            "drop_order": forecast["drop_order"],
            "note": ("a forecast, measured on this box. It is reported whether or not it fits the "
                     "24-hour ideal; it is never used to relabel an incomplete flight.")}


def stage_calibrate(context, services, **_):
    outcome = services.calibrate()
    paths.write_json(context.path("calibration_ledger.json"), _plain(outcome["ledger"]))
    paths.write_json(context.path("calibration_outcome.json"), _plain(outcome))
    # Read from the FROZEN record, which is what production consults. A family
    # that produced no qualified 1,000-update continuation contributes no
    # coefficient, and the arms citing it stay ``coefficients_pending``.
    frozen = dict(outcome["frozen"])
    families = sorted(name for name, block in frozen.items()
                      if isinstance(block, dict) and "coefficients" in block)
    unqualified = sorted(name for name in families if not frozen[name].get("coefficients"))
    return {"status": "completed", "frozen": frozen,
            "qualified_families": [name for name in families if name not in unqualified],
            "unqualified_families": unqualified,
            "measured_gpu_hours": outcome["ledger"]["measured_gpu_hours"],
            "measured_updates": outcome["ledger"]["completed_updates"],
            "note": ("an unqualified family is a recorded outcome, not a reason to invent a "
                     "coefficient. The documented alternative is frozen before challenge "
                     "training and before any E scoring, and the arms that cite an unqualified "
                     "family have terminal no_qualified_configuration status. Independent "
                     "controls continue.")}


def stage_freeze(context, services, **_):
    snapshot = spec.source_snapshot(context)
    geometry = read_stage_record(context.run_root, "geometry")
    calibration = read_stage_record(context.run_root, "calibrate")
    forecast_path = context.path(FORECAST_JSON)
    forecast = paths.read_json(forecast_path) if forecast_path.is_file() else None
    protocol = spec.resolved_protocol(context, snapshot=snapshot, geometry=geometry,
                                      calibration=calibration, forecast=forecast)
    if protocol["unresolved"]:
        raise ReadinessGate(
            f"the resolved protocol still has unresolved fields {protocol['unresolved']}. "
            "Production runs against a filled protocol.")
    return {"status": "completed", "snapshot_sha256": snapshot["snapshot_sha256"],
            "files": snapshot["file_count"], "git_head": snapshot["git_state"]["head"],
            "dirty": snapshot["git_state"]["dirty"],
            "supersession": spec.historical_supersession_disclosure(context)}


def stage_production(context, services, *, max_trajectories=None, heartbeat=None, **_):
    """The long queue, and the cheap checks that must pass before it starts.

    The miniature end-to-end smoke path runs here if it has not already: resume,
    a deliberate gate stop, a wrong-identity refusal, a corrupt-state refusal and
    the forbidden-row guard, all against the production code. An orchestration
    defect found twenty hours in is not a cheap one, and this is the last point
    at which finding it is.
    """
    spec.verify_source_snapshot(context, label="production")
    spec.require_resolved(context, label="production")
    smoke = read_stage_record(context.run_root, "smoke")
    if smoke is None or smoke.get("status") != "completed":
        smoke = run_stage(context, "smoke", services)
        if smoke["status"] != "completed":
            raise ReadinessGate(
                "the miniature end-to-end smoke path did not pass, so the production queue does "
                f"not start: {smoke.get('reason')}")
    result = services.run_production(max_trajectories=max_trajectories, heartbeat=heartbeat)
    return {"status": "completed" if result["all_terminal"] else "partial",
            "started": result["started"], "completed": result["completed"],
            # The runner's key is ``stopped_by_gate``. Reading ``stopped`` raised
            # a KeyError at the end of the longest stage in the flight.
            "stopped_by_gate": result["stopped_by_gate"],
            "incomplete": result["incomplete"], "failed": result["failed"],
            "any_terminal": bool(result["any_terminal"]),
            "remaining": result["remaining"],
            "unrunnable": result.get("unrunnable") or [],
            "unrunnable_reason": result.get("unrunnable_reason"),
            "reason": (None if result["all_terminal"] else
                       "not every queued trajectory has a terminal status yet; the flight remains "
                       "scientifically in progress")}


def stage_audit(context, services, **_):
    """Freeze the finalist choice first, then draw the audit banks against it.

    The order is the guarantee. ``require_may_influence_selection`` refuses a
    final-preservation bank until a freeze record names the checkpoints, so the
    freeze cannot be written after looking at the bank.
    """
    spec.verify_source_snapshot(context, label="audit")
    freeze = services.freeze_finalists()
    result = services.run_audits()
    paths.write_json(context.path("preservation_audit.json"), _plain(result))
    return {"status": "completed", "models": len(result["models"]),
            "banks": len(result["banks"]),
            "finalists_named": len(freeze.get("named_checkpoints") or []),
            "tail_family": freeze.get("tail_family"),
            "note": ("fresh 50k parent audits on banks drawn after the reported checkpoints were "
                     "named. Bank counts are not reduced.")}


def stage_couple(context, services, **_):
    result = services.run_coupling()
    paths.write_json(context.path("coupling.json"), _plain(result))
    return {"status": "completed", "finalists": len(result["finalists"]),
            "banks": result["banks"], "screens": len(result["screens"]),
            "mixture_diagnostics": len(result.get("mixture_diagnostics") or [])}


def stage_report(context, services, **_):
    from . import her2_nf_report as report_lib
    evidence = services.report_evidence() if hasattr(services, "report_evidence") else None
    document = report_lib.build_report(context, services=services, evidence=evidence)
    return {"status": "completed", "rows": document["rows"],
            "artifacts": len(document["artifact_manifest"]),
            "challenge_scored": bool((evidence or {}).get("challenge")),
            "complete_flight": document["completion"]["complete"]}


def stage_verify(context, services, **_):
    from . import her2_nf_report as report_lib
    document = report_lib.verify(context)
    report_lib.finalize_verification(context, document)
    return {"status": "completed" if document["passed"] else "failed",
            "checks": document["checks"], "passed": document["passed"],
            "reason": None if document["passed"] else "one or more verification checks failed"}


def stage_smoke(context, services, **_):
    """A miniature end-to-end native run, including the exact negative cases.

    Real model, real optimizer, real journals, tiny sizes. It exercises the stage
    ordering, save/resume, a deliberately triggered gate stop, a wrong-parent
    refusal and a corrupt-state refusal *before* the launch, because an
    orchestration defect found after 20 hours of training is not a cheap one.
    """
    try:
        result = services.smoke()
    except ValueError as error:
        if "never erases an existing run" not in str(error):
            raise
        # An occupied namespace is a readiness problem with an operator action,
        # not a science result and not a reason to delete somebody's run.
        raise ReadinessGate(
            f"the smoke namespace is occupied and is not erased: {error}") from error
    paths.write_json(context.path("smoke.json"), _plain(result))
    if not result["passed"]:
        # Recorded as a failed stage rather than raised: the production stage
        # turns this into a readiness gate, and a failure that vanished into a
        # traceback would leave no record of which negative case did not hold.
        return {"status": "failed", "checks": result["checks"], "passed": False,
                "reason": f"failed checks: {result.get('failures')}"}
    return {"status": "completed", "checks": result["checks"], "passed": True}


STAGE_FUNCTIONS = {
    "recover": stage_recover, "preflight": stage_preflight, "geometry": stage_geometry,
    "mixtures": stage_mixtures, "profile": stage_profile, "calibrate": stage_calibrate,
    "freeze": stage_freeze, "production": stage_production, "audit": stage_audit,
    "couple": stage_couple, "report": stage_report, "verify": stage_verify,
    "smoke": stage_smoke}

#: Which stages a failed stage blocks. The rest still run, and their independence
#: is the reason a feasibility failure is not a whole-flight failure.
DEPENDENTS = {
    "geometry": ("production", "audit", "couple", "report", "verify"),
    "calibrate": ("production", "audit", "couple", "report", "verify"),
    "profile": ("calibrate", "freeze", "production", "audit", "couple", "report", "verify"),
    "preflight": tuple(STAGES[2:]),
    "recover": tuple(STAGES[1:]),
    "freeze": ("production", "audit", "couple", "report", "verify"),
    "production": ("audit", "couple", "report", "verify"),
    "audit": ("report", "verify"),
    "couple": ("report", "verify"),
    "report": ("verify",),
}


#: What a stage's reuse identity is made of. A completed record whose inputs
#: still hash the same is a finished stage; a completed record whose source
#: snapshot or configuration moved is not, and rerunning one blindly discards a
#: result that took hours.
#: Stages that consume what production produced rather than only its configuration.
PRODUCTION_DEPENDENT_STAGES = ("audit", "couple", "report", "verify")

#: The fields of the production record that describe what it PRODUCED. Wall clock
#: and the per-session start count are excluded: they move on a rerun that added
#: nothing, and reusing a stage is the right answer there.
PRODUCTION_OUTCOME_KEYS = ("status", "completed", "stopped_by_gate", "incomplete",
                           "failed", "any_terminal", "remaining", "unrunnable")


def production_outcome_digest(run_root):
    """What production has produced so far, as one digest, or ``None`` before it ran."""
    record = read_stage_record(run_root, "production")
    if record is None:
        return None
    return paths.sha256_text(paths.canonical_json(
        {key: record.get(key) for key in PRODUCTION_OUTCOME_KEYS}))


def stage_identity(context, stage):
    """Source, configuration, declared inputs and -- downstream -- production's outcome.

    A ``--max-trajectories`` session leaves the source snapshot and the
    configuration untouched while the queue advances. Without the production
    digest the audit, coupling and report stages recorded under the partial queue
    were reused on the next invocation, so the checkpoints produced in between
    were never audited and never appeared in the report.
    """
    marker = context.path(spec.FREEZE_MARKER)
    snapshot = paths.read_json(marker)["snapshot_sha256"] if marker.is_file() else None
    identity = {"stage": str(stage), "config_sha256": paths.sha256_file(context.config_path),
                "source_snapshot_sha256": snapshot}
    if str(stage) in PRODUCTION_DEPENDENT_STAGES:
        identity["production_outcome_sha256"] = production_outcome_digest(context.run_root)
    return identity


def stage_is_reusable(context, stage):
    """``(reusable, reason)`` for an already-recorded stage."""
    record = read_stage_record(context.run_root, stage)
    if record is None:
        return False, "no record"
    if record.get("status") != "completed":
        return False, f"the recorded status is {record.get('status')!r}"
    recorded = record.get("identity")
    if recorded is None:
        return False, "the record carries no stage identity to compare against"
    current = stage_identity(context, stage)
    differing = sorted(key for key in set(recorded) | set(current)
                       if recorded.get(key) != current.get(key))
    if differing:
        return False, f"the stage identity changed in {differing}"
    return True, "the stage completed under the same source and configuration"


def run_stage(context, stage, services, reuse=True, **kwargs):
    """Run one stage, record it, and classify any gate it hits.

    A completed stage whose source snapshot and configuration are unchanged is
    REUSED rather than rerun. Rerunning ``production`` because ``run_all`` was
    invoked a second time would discard the queue it just spent twenty hours on.
    """
    require(stage in STAGE_FUNCTIONS, f"Unknown stage {stage!r}")
    # Bind every stage, including calibration, to the same immutable bytes.
    # Freezing only after calibration changed earlier stage identities on resume.
    spec.source_snapshot(context)
    if reuse:
        reusable, reason = stage_is_reusable(context, stage)
        if reusable:
            record = dict(read_stage_record(context.run_root, stage))
            record["reused"] = {"reason": reason, "at": paths.utc_now()}
            return record
    started = time.perf_counter()
    try:
        document = _plain(STAGE_FUNCTIONS[stage](context, services, **kwargs))
        document["wall_seconds"] = time.perf_counter() - started
        document["identity"] = stage_identity(context, stage)
        return write_stage_record(context.run_root, stage, document)
    except ReadinessGate as error:
        return write_stage_record(context.run_root, stage, {
            "status": "blocked_readiness", "reason": str(error),
            "wall_seconds": time.perf_counter() - started,
            "consequence": ("a prerequisite is missing. The flight pauses here; no scientific "
                            "claim is made and nothing is marked complete.")})
    except FeasibilityGate as error:
        return write_stage_record(context.run_root, stage, {
            "status": "blocked_feasibility", "reason": str(error),
            "wall_seconds": time.perf_counter() - started,
            "consequence": ("a declared scientific criterion failed. Evidence is retained, "
                            "dependent stages stop, and independent stages still run.")})


def run_all(context, services, *, stages=STAGES, heartbeat=None, **kwargs):
    """Advance deterministically through the whole flight. One command, no follow-ups.

    A stage that produced PARTIAL production work does not block the audit: the
    endpoints that exist are real, and refusing to audit them because the queue
    is not finished loses the only scientific output an interrupted flight has.
    The stage list still records ``partial`` and the completion check still
    refuses to call that complete.
    """
    blocked, records = {}, {}
    for stage in stages:
        if heartbeat is not None:
            heartbeat.beat(force=True, phase="stage", stage=stage,
                           completed=sorted(name for name, record in records.items()
                                            if record["status"] == "completed"))
        if stage in blocked:
            records[stage] = write_stage_record(context.run_root, stage, {
                "status": "not_run", "reason": blocked[stage],
                "consequence": "a required stage that did not run; the flight is not complete"})
            continue
        record = run_stage(context, stage, services, heartbeat=heartbeat, **kwargs)
        records[stage] = record
        partial_but_usable = (stage == "production" and record["status"] == "partial"
                              and record.get("any_terminal"))
        if partial_but_usable:
            records[stage] = dict(record, downstream=(
                "some trajectories reached a terminal status, so the audit, coupling and report "
                "stages run on what exists. The queue is still incomplete and the completion "
                "check says so."))
            continue
        if record["status"].startswith("blocked") or record["status"] in ("failed", "partial"):
            for dependent in DEPENDENTS.get(stage, ()):  # only the dependents stop
                blocked.setdefault(dependent, f"blocked by {stage}: {record.get('reason')}")
    completed = [name for name, record in records.items() if record["status"] == "completed"]
    document = {"schema_version": CAMPAIGN_SCHEMA, "record_kind": "run_all",
                "stages": {name: record["status"] for name, record in records.items()},
                "completed_stages": completed,
                "blocked": blocked,
                "required_stages": list(REQUIRED_STAGES),
                "all_required_completed": all(records.get(name, {}).get("status") == "completed"
                                              for name in REQUIRED_STAGES if name in stages),
                "claim": ("a stage list with statuses. Completion of the FLIGHT is asserted only "
                          "by the verify stage against the required stage list; a launch, a "
                          "running worker and a partial stage list are all different facts."),
                "finished_at": paths.utc_now()}
    paths.write_json(context.path("run_all.json"), document)
    return document


# ---------------------------------------------------------------------------
# the durable supervisor
# ---------------------------------------------------------------------------

def supervise(context, services, *, stages=STAGES, heartbeat_seconds=15.0, owner=None,
              max_restarts=0, **kwargs):
    """Hold the single-writer lock, beat, run the flight, publish a truthful status.

    The lock is held for the whole writing life of the run, because bank
    generation, fitting and reporting all write into the same tree and two
    writers there would interleave journals and checkpoints. Liveness is the lock
    state; the heartbeat is a progress signal and is never read as liveness.
    """
    lock = CampaignLock(context.path(LOCK_FILE), owner=owner or owner_identity())
    beat = Heartbeat(context.path(HEARTBEAT_JSON), owner=lock.owner, every=float(heartbeat_seconds))
    started = paths.utc_now()
    with lock.held(), beat.running():
        beat.beat(force=True, phase="starting", stages=list(stages))
        attempts, document, failure = 0, None, None
        while True:
            attempts += 1
            try:
                # The heartbeat travels INTO the stages and into the trajectory
                # loop, so it advances with the work. Wrapping run_all without
                # propagating it meant the heartbeat stood still for the whole
                # twenty-hour production stage.
                document = run_all(context, services, stages=stages, heartbeat=beat, **kwargs)
                break
            except BaseException as error:                      # noqa: BLE001 - recorded, re-raised
                failure = f"{type(error).__name__}: {error}"
                paths.write_json(context.path("supervisor_failure.json"), {
                    "schema_version": CAMPAIGN_SCHEMA, "record_kind": "supervisor_failure",
                    "attempt": attempts, "error": failure, "at": paths.utc_now(),
                    "consequence": ("a crash, not a scientific outcome. The status below says "
                                    "`failed`; nothing is reported as complete.")})
                if attempts > int(max_restarts):
                    _publish_status(context, stages=stages, started=started, owner=lock.owner,
                                    status="failed", failure=failure, document=None)
                    raise
                beat.beat(force=True, phase="restarting", attempt=attempts, error=failure)
        status = ("completed" if document["all_required_completed"]
                  else "stopped" if document["blocked"] else "incomplete")
        beat.beat(force=True, phase="finished", status=status)
        return _publish_status(context, stages=stages, started=started, owner=lock.owner,
                               status=status, failure=None, document=document)


def _publish_status(context, *, stages, started, owner, status, failure, document):
    # The PUBLISHED queue and counts carry the recorded preflight reuse decision,
    # exactly as the report does. Rebuilding them under the conservative default
    # here left all six reused controls ``reuse_pending_parity`` in queue.json and
    # in trajectory_counts even after the flight finished, because production
    # skips a verified-reuse row without writing a local terminal record.
    queue = queue_for_reporting(context)
    state = campaign_state(context.run_root, queue)
    record = {"schema_version": CAMPAIGN_SCHEMA, "record_kind": "campaign_status",
              "campaign_id": context.campaign_id, "status": status, "failure": failure,
              "owner": dict(owner), "started_at": started, "finished_at": paths.utc_now(),
              "stages": stage_summary(context.run_root, stages),
              "trajectory_counts": state["counts"],
              "run_all": document,
              "launch_versus_completion": (
                  "a verified worker and advancing journals establish that this flight is "
                  "RUNNING. Scientific completion is a separate fact, asserted only when every "
                  "required stage has completed."),
              "lock": lock_state(context.path(LOCK_FILE))}
    paths.write_json(context.path(CAMPAIGN_STATUS_JSON), record)
    paths.write_json(context.path(QUEUE_JSON), state)
    return record


def launch_detached(context, *, python=None, script="scripts/her2_next_flight_supervisor.py",
                    arguments=()):
    """Start the supervisor as a detached process so it outlives this session.

    Background jobs started inside an agent turn are killed at the turn boundary,
    which is why this is a detached process with its own log rather than a child
    of the caller.
    """
    executable = str(python or sys.executable)
    target = context.repository_root / script
    require(target.is_file(), f"{script} is not present")
    require(lock_state(context.path(LOCK_FILE))["state"] != "held",
            "This flight already has a live writer; inspect health before launching another.")
    log = context.path("supervisor.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [executable, str(target), *[str(value) for value in arguments],
               "--config", str(context.config_path), "--output", str(context.run_root)]
    options = {}
    if sys.platform == "win32":
        # Windows only: ``creationflags`` is rejected outright on POSIX, and a
        # detached process group is what makes the supervisor outlive the shell.
        options["creationflags"] = (getattr(subprocess, "DETACHED_PROCESS", 0)
                                    | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    else:
        options["start_new_session"] = True
    with log.open("ab") as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL,
                                   cwd=str(context.repository_root), close_fds=True, **options)
    record = {"schema_version": CAMPAIGN_SCHEMA, "record_kind": "launch",
              "pid": int(process.pid), "command": command, "log": str(log),
              "launched_at": paths.utc_now(),
              "verification_required": ("a PID is not progress. Launch is verified by the lock "
                                        "being held, the heartbeat advancing and the trajectory "
                                        "journals gaining records.")}
    paths.write_json(context.path("launch.json"), record)
    return record


def launch_health(context, *, minimum_updates=1):
    """Is a worker actually running, and has it done anything? Two different questions."""
    launch = context.path("launch.json")
    record = paths.read_json(launch) if launch.is_file() else None
    lock = lock_state(context.path(LOCK_FILE))
    heartbeat_path = context.path(HEARTBEAT_JSON)
    heartbeat = paths.read_json(heartbeat_path) if heartbeat_path.is_file() else None
    queue = build_queue(context.config, frozen=_frozen_if_available(context),
                        require_frozen=False)
    state = campaign_state(context.run_root, queue)
    production = sum(int(row["journalled_updates"]) for row in state["rows"])
    # Production journals are only the LAST phase. A flight that has been
    # calibrating or fitting fresh stage-one parents for two hours has advanced;
    # reading only the production queue reported it as running without progress,
    # which is the signal an operator would use to decide it had hung.
    calibration = 0
    calibration_directory = context.path("calibration")
    if calibration_directory.is_dir():
        for directory in sorted(calibration_directory.iterdir()):
            if directory.is_dir():
                calibration += int(trajectory_lib.durable_progress(directory)["updates"])
    stage_one = 0
    parents_directory = context.path("parents")
    if parents_directory.is_dir():
        for directory in sorted(parents_directory.iterdir()):
            history = directory / "stage1_history.json"
            if history.is_file():
                stage_one += len(paths.read_json(history))
    stages_done = sum(1 for block in stage_summary(context.run_root).values()
                      if block["status"] == "completed")
    advanced = production + calibration + stage_one
    running = lock["state"] == "held"
    return {"schema_version": CAMPAIGN_SCHEMA, "record_kind": "launch_health",
            "launch_record": record, "lock": lock, "heartbeat": heartbeat,
            "journalled_updates_total": advanced,
            "work": {"production_updates": production, "calibration_updates": calibration,
                     "stage_one_epochs": stage_one, "stages_completed": stages_done},
            "worker_running": bool(running),
            "progress_observed": bool(advanced >= int(minimum_updates) or stages_done > 0),
            "health": ("running_with_progress"
                       if running and (advanced >= int(minimum_updates) or stages_done > 0)
                       else "running_without_progress_yet" if running
                       else "not_running"),
            "claim": ("scientific completion is never inferred from a successful launch. This "
                      "record answers 'is a writer alive' and 'has anything advanced', and "
                      "nothing more.")}
