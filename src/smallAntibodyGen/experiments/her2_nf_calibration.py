"""Bounded calibration: four ladders, one rule, a hard cap, and a truthful ledger.

The whole point of this stage is that it is *bounded*. A large Cartesian grid,
an unbounded rescue search, or a per-seed retune would each make the production
comparison uninterpretable, so the budget is declared up front and the ledger
records every attempt -- including the ones that failed and the ones that were
never started because the cap was reached. There is no survivor-only table.

The declared ladders, at parent seed 20260918 with its own spawned optimization
and RNG stream:

1. DPO ``beta in {.1, .5}`` x FKL ``lambda in {3, 10, 30}`` -- six pilots to 500.
2. One ``beta`` is chosen and held fixed across none / FKL / tail.
3. IPO-tail and selected-DPO-tail, each ``lambda in {.01, .1, 1}`` -- six pilots
   to 500.
4. The three selected extendable candidates go to 1,000 before the freeze.

Nominal total 7,500 pilot updates, capped at three **measured** GPU-hours, with
at most two additional 500-update bracket expansions across the whole stage and
at most one fallback per extended family. An expansion is a bounded rescue for a
ladder that is infeasible or effectively prevents learning -- it is not a default
reason to extend a tail ladder past 500 because its penalty was quiet.

Selection: the smallest ``lambda`` meeting the development-bank point rates
(inclusive tenfold <= 1%, inclusive hundredfold <= .1%) whose macro-AP exceeds
the parent's point estimate. Feasible betas are compared by macro-AP with a .001
engineering tie broken by Y@10k, then Y@1M, then the smaller beta. If nothing
qualifies, "no qualified calibrated configuration" is the recorded outcome for
that family and the documented alternative is frozen before any challenge
training and before any E scoring -- a failure does not disappear from the
report, and it is never relabelled as a tuned success.

Short-pilot preservation is provisional. Requiring every 10k-bank Wilson upper
bound to clear here would recreate the known power problem; the final selected
checkpoints are audited on the larger separate banks instead.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .her2_nf_contract import NF_SCHEMA
from .her2_runtime import require

#: The declared ladders.
DPO_BETAS = (0.1, 0.5)
FKL_LAMBDAS = (3.0, 10.0, 30.0)
TAIL_LAMBDAS = (0.01, 0.1, 1.0)
PILOT_UPDATES = 500
EXTENDED_UPDATES = 1000

#: Provisional development-bank feasibility criteria, inclusive tail events.
POINT_RATE_CRITERIA = {"tenfold": 0.01, "hundredfold": 0.001}

#: Bounds on the whole stage.
GPU_HOUR_CAP = 3.0
MAX_BRACKET_EXPANSIONS = 2
MAX_FALLBACKS_PER_FAMILY = 1
AP_TIE = 0.001

#: Optional SimPO's separate, smaller allowance.
SIMPO_OBJECTIVE_PILOTS = 2
SIMPO_FKL_BRACKET = 3


@dataclass
class CalibrationEntry:
    """One attempted setting: what it was, what it got, what it cost."""

    entry_id: str
    family: str
    task: str
    preservation: str
    coefficients: dict
    allocated_updates: int
    completed_updates: int = 0
    status: str = "queued"
    metrics: dict = field(default_factory=dict)
    cost: dict = field(default_factory=dict)
    reason: object = None
    #: For an ``EXT:`` continuation, the 500-update entry whose trajectory it
    #: resumed. ``None`` for a pilot that started from the parent.
    resumed_from: object = None

    def document(self):
        return {"entry_id": self.entry_id, "family": self.family, "task": self.task,
                "preservation": self.preservation,
                "coefficients": dict(sorted(self.coefficients.items())),
                "allocated_updates": int(self.allocated_updates),
                "completed_updates": int(self.completed_updates),
                "status": self.status, "metrics": dict(self.metrics), "cost": dict(self.cost),
                "resumed_from": self.resumed_from, "reason": self.reason}


def ladder_entries(*, stage, task, preservation, lambdas, betas=(None,),
                   updates=PILOT_UPDATES, fixed=None):
    """Build one ladder's entries in the declared order: ascending beta, ascending lambda."""
    entries = []
    for beta in betas:
        for value in lambdas:
            coefficients = dict(fixed or {})
            if beta is not None:
                coefficients["beta"] = float(beta)
            coefficients["lambda"] = float(value)
            suffix = "_".join(f"{k}{v:g}" for k, v in sorted(coefficients.items()))
            entries.append(CalibrationEntry(
                entry_id=f"{stage}:{task}_{preservation}_{suffix}",
                family=f"{task}_{preservation}", task=task, preservation=preservation,
                coefficients=coefficients, allocated_updates=int(updates)))
    return entries


def nominal_plan():
    """The declared ladders and the nominal update total, before any measurement."""
    stage_one = ladder_entries(stage="L1", task="dpo", preservation="fkl", lambdas=FKL_LAMBDAS,
                               betas=DPO_BETAS)
    stage_two = ladder_entries(stage="L2", task="ipo", preservation="tail", lambdas=TAIL_LAMBDAS,
                               fixed={"tau": 0.1})
    return {"stage_one": stage_one, "stage_two_ipo_tail": stage_two,
            "nominal_updates": (len(stage_one) * PILOT_UPDATES
                                + 2 * len(TAIL_LAMBDAS) * PILOT_UPDATES
                                + 3 * (EXTENDED_UPDATES - PILOT_UPDATES)),
            "note": ("stage two's DPO-tail ladder is built only after a beta is selected, so it "
                     "is not listed with a placeholder beta here.")}


# ---------------------------------------------------------------------------
# the selection rule
# ---------------------------------------------------------------------------

def is_feasible(metrics, *, criteria=None):
    """Development-bank point rates, inclusive thresholds, with the reason recorded."""
    table = dict(criteria or POINT_RATE_CRITERIA)
    reasons = []
    for name, limit in sorted(table.items()):
        observed = metrics.get(f"{name}_rate")
        if observed is None:
            reasons.append(f"{name}_rate not measured")
            continue
        if not math.isfinite(float(observed)) or not 0.0 <= float(observed) <= 1.0:
            reasons.append(f"{name}_rate is not a finite rate in [0, 1]")
            continue
        if float(observed) > float(limit):
            reasons.append(f"{name}_rate {observed:.5f} > {limit}")
    return {"feasible": not reasons, "reasons": reasons, "criteria": table,
            "tail_convention": ">= (inclusive)",
            "status": ("provisional short-pilot preservation. Requiring every 10k-bank Wilson "
                       "upper bound to clear here would recreate the known power problem; the "
                       "final checkpoints are audited on larger separate banks.")}


def qualifies(entry, *, parent_macro_ap, criteria=None):
    """Does one completed pilot meet BOTH declared conditions, with finite metrics?"""
    feasibility = is_feasible(entry.metrics, criteria=criteria)
    observed = entry.metrics.get("macro_average_precision")
    finite = observed is not None and math.isfinite(float(observed))
    improves = bool(finite and float(observed) > float(parent_macro_ap))
    return {"entry_id": entry.entry_id,
            "lambda": float(entry.coefficients.get("lambda", 0.0)),
            "completed_updates": int(entry.completed_updates),
            "status": entry.status,
            "feasible": feasibility["feasible"],
            "feasibility_reasons": feasibility["reasons"],
            "macro_average_precision_finite": bool(finite),
            "improves_on_parent": improves,
            "macro_average_precision": observed,
            "qualified": bool(entry.status == "completed"
                              and entry.completed_updates >= entry.allocated_updates
                              and feasibility["feasible"] and improves)}


def select_lambda(entries, *, parent_macro_ap, criteria=None):
    """The smallest qualified lambda, with EVERY other qualified one ranked behind it.

    The whole ladder is enumerated before anything is returned. Returning as soon
    as the first qualified entry was found left ``ranked_alternatives`` built from
    a partial list -- it named whatever happened to be after the winner in the
    already-walked prefix, so the one permitted fallback pointed at a candidate
    that had never been evaluated, or at nothing at all.
    """
    ranked = sorted(entries, key=lambda entry: float(entry.coefficients.get("lambda", 0.0)))
    considered = [qualifies(entry, parent_macro_ap=parent_macro_ap, criteria=criteria)
                  for entry in ranked]
    qualified = [block for block in considered if block["qualified"]]
    if not qualified:
        return {"selected": None, "coefficients": None,
                "outcome": "no qualified calibrated configuration for this family",
                "considered": considered, "ranked_alternatives": [],
                "consequence": ("recorded as a result. The documented alternative is frozen "
                                "BEFORE challenge training and before any E scoring; this is "
                                "never relabelled as a tuned success.")}
    winner = qualified[0]
    chosen = next(entry for entry in ranked if entry.entry_id == winner["entry_id"])
    return {"selected": chosen.entry_id, "coefficients": dict(chosen.coefficients),
            "macro_average_precision": winner["macro_average_precision"],
            "rule": ("smallest lambda meeting the inclusive point-rate criteria with macro-AP "
                     "above the parent point estimate, chosen after the WHOLE ladder is "
                     "evaluated"),
            "considered": considered,
            "ranked_alternatives": [block["entry_id"] for block in qualified[1:]]}


def select_beta(per_beta, *, tie=AP_TIE):
    """Compare feasible betas by macro-AP, then Y@10k, then Y@1M, then the smaller beta."""
    feasible = [block for block in per_beta if block.get("selected")]
    if not feasible:
        return {"selected_beta": None, "outcome": "no feasible DPO beta",
                "considered": list(per_beta)}
    best = max(float(block["macro_average_precision"]) for block in feasible)
    tied = [block for block in feasible
            if best - float(block["macro_average_precision"]) <= float(tie)]
    if len(tied) > 1:
        tie_rule = "Y@10k"
        best_yield = max(float(block.get("yield_10k") or -math.inf) for block in tied)
        narrowed = [block for block in tied
                    if float(block.get("yield_10k") or -math.inf) == best_yield]
        if len(narrowed) > 1:
            tie_rule = "Y@1M"
            best_million = max(float(block.get("yield_1m") or -math.inf) for block in narrowed)
            narrowed = [block for block in narrowed
                        if float(block.get("yield_1m") or -math.inf) == best_million]
        if len(narrowed) > 1:
            tie_rule = "smaller beta"
            narrowed = [min(narrowed, key=lambda block: float(block["beta"]))]
        chosen = narrowed[0]
    else:
        tie_rule = None
        chosen = tied[0]
    return {"selected_beta": float(chosen["beta"]), "selected_entry": chosen["selected"],
            "macro_average_precision": chosen["macro_average_precision"],
            "tie_broken_by": tie_rule, "tie_threshold": float(tie),
            "considered": list(per_beta),
            "rule": ("macro-AP, with a .001 absolute AP difference treated as an engineering tie "
                     "broken by Y@10k, then Y@1M, then the smaller beta. This is a reproducible "
                     "calibration heuristic, not a universal utility function.")}


def non_dominated(entries):
    """Pilot points not dominated on (AP up, Y@10k up, tenfold down, hundredfold down)."""
    points = [entry for entry in entries
              if entry.metrics.get("macro_average_precision") is not None]
    keep = []
    for entry in points:
        dominated = False
        for other in points:
            if other is entry:
                continue
            if (_ge(other, entry, "macro_average_precision") and _ge(other, entry, "yield_10k")
                    and _le(other, entry, "tenfold_rate") and _le(other, entry, "hundredfold_rate")
                    and (_gt(other, entry, "macro_average_precision")
                         or _gt(other, entry, "yield_10k")
                         or _lt(other, entry, "tenfold_rate")
                         or _lt(other, entry, "hundredfold_rate"))):
                dominated = True
                break
        if not dominated:
            keep.append(entry.entry_id)
    return {"non_dominated": keep, "considered": [entry.entry_id for entry in points],
            "axes": ["macro_average_precision up", "yield_10k up", "tenfold_rate down",
                     "hundredfold_rate down"]}


def _value(entry, key, default):
    value = entry.metrics.get(key)
    return default if value is None else float(value)


def _ge(a, b, key):
    return _value(a, key, -math.inf) >= _value(b, key, -math.inf)


def _gt(a, b, key):
    return _value(a, key, -math.inf) > _value(b, key, -math.inf)


def _le(a, b, key):
    return _value(a, key, math.inf) <= _value(b, key, math.inf)


def _lt(a, b, key):
    return _value(a, key, math.inf) < _value(b, key, math.inf)


# ---------------------------------------------------------------------------
# the bounded runner
# ---------------------------------------------------------------------------

class MachineryFailure(RuntimeError):
    """A pilot did not produce a scientific observation; something is broken.

    A shape mismatch, a missing bank, a wrong-parent cache or a programming error
    is not "this coefficient failed". Recording it as a failed pilot would let a
    whole ladder be declared infeasible because a path was wrong, so it aborts
    the stage with its diagnostics instead.
    """


class CalibrationLedger:
    """Every attempted setting, in order, with its cost. Never a survivor-only table.

    The ledger is the stage's durable memory: ``persist`` is called after every
    recorded entry and every decision, so a resumed calibration knows which
    settings already ran, how many updates they completed and what they cost.
    Restarting with a fresh ledger would spend the three-hour allocation twice.
    """

    def __init__(self, *, gpu_hour_cap=GPU_HOUR_CAP,
                 max_bracket_expansions=MAX_BRACKET_EXPANSIONS,
                 max_fallbacks_per_family=MAX_FALLBACKS_PER_FAMILY, persist=None):
        self.entries = []
        self.gpu_hour_cap = float(gpu_hour_cap)
        self.max_bracket_expansions = int(max_bracket_expansions)
        self.max_fallbacks_per_family = int(max_fallbacks_per_family)
        self.expansions = []
        self.fallbacks = {}
        self.fallback_requests = {}
        self.measured_gpu_seconds = 0.0
        self.uncertainty_debit_seconds = 0.0
        self.overhead = {}
        self.decisions = []
        self._persist = persist

    @classmethod
    def restore(cls, document, **kwargs):
        """Rebuild a ledger from its own persisted document."""
        ledger = cls(gpu_hour_cap=float(document.get("gpu_hour_cap", GPU_HOUR_CAP)),
                     max_bracket_expansions=int(document.get("bracket_expansion_allowance",
                                                             MAX_BRACKET_EXPANSIONS)),
                     **kwargs)
        for block in document.get("entries") or []:
            entry = CalibrationEntry(
                entry_id=block["entry_id"], family=block["family"], task=block["task"],
                preservation=block["preservation"], coefficients=dict(block["coefficients"]),
                allocated_updates=int(block["allocated_updates"]),
                completed_updates=int(block.get("completed_updates") or 0),
                status=str(block.get("status") or "queued"),
                metrics=dict(block.get("metrics") or {}), cost=dict(block.get("cost") or {}),
                resumed_from=block.get("resumed_from"), reason=block.get("reason"))
            ledger.entries.append(entry)
            ledger.measured_gpu_seconds += float(entry.cost.get("gpu_seconds") or 0.0)
            ledger.uncertainty_debit_seconds += float(entry.cost.get("uncertainty_debit_seconds") or 0.0)
        ledger.expansions = list(document.get("bracket_expansions") or [])
        ledger.fallbacks = dict(document.get("fallbacks") or {})
        ledger.fallback_requests = dict(document.get("fallback_requests") or {})
        ledger.decisions = list(document.get("decisions") or [])
        ledger.record_overhead(document.get("overhead") or {}, flush=False)
        return ledger

    @property
    def overhead_charged_seconds(self):
        return float(self.overhead.get("measured_seconds", 0.0)) + float(
            self.overhead.get("uncertainty_debit_seconds", 0.0))

    def record_overhead(self, document, *, flush=True):
        """Replace durable baseline totals; never charge the same setup twice on resume."""
        self.measured_gpu_seconds += float(document.get("measured_seconds", 0.0)) - float(
            self.overhead.get("measured_seconds", 0.0))
        self.uncertainty_debit_seconds += float(document.get("uncertainty_debit_seconds", 0.0)) - float(
            self.overhead.get("uncertainty_debit_seconds", 0.0))
        self.overhead = dict(document)
        if flush:
            self._flush()

    def completed(self, entry_id):
        """The already-measured entry with this id, if the ledger carries one."""
        for entry in self.entries:
            if entry.entry_id == str(entry_id):
                return entry
        return None

    @property
    def measured_gpu_hours(self):
        return self.measured_gpu_seconds / 3600.0

    def remaining_hours(self):
        return max(0.0, self.gpu_hour_cap - self.measured_gpu_hours
                   - self.uncertainty_debit_seconds / 3600.0)

    def remaining_seconds(self):
        return self.remaining_hours() * 3600.0

    def record(self, entry):
        if entry.status not in ("not_started",):
            cost = entry.cost.get("gpu_seconds")
            require(cost is not None,
                    f"{entry.entry_id} reports no measured gpu_seconds. Counting an unmeasured "
                    "pilot as zero would let the three-hour cap be overspent without any record "
                    "of it.")
            require(math.isfinite(float(cost)) and float(cost) >= 0,
                    "Measured calibration cost must be finite and nonnegative")
            self.measured_gpu_seconds += float(cost)
            self.uncertainty_debit_seconds += float(entry.cost.get("uncertainty_debit_seconds") or 0.0)
        self.entries.append(entry)
        self._flush()
        return entry

    def note(self, decision):
        self.decisions.append(dict(decision, at=len(self.entries)))
        self._flush()
        return decision

    def _flush(self):
        if self._persist is not None:
            self._persist(self.document())

    def request_expansion(self, *, family, coefficient, reason):
        for record in self.expansions:
            if record["family"] == family and record["coefficient"] == str(coefficient):
                return record
        if len(self.expansions) >= self.max_bracket_expansions:
            return {"granted": False, "family": family,
                    "reason": ("the declared allowance of "
                               f"{self.max_bracket_expansions} bracket expansions is spent. No "
                               "unbounded rescue search follows a spent allowance.")}
        record = {"granted": True, "family": family, "coefficient": str(coefficient),
                  "reason": str(reason), "updates": PILOT_UPDATES,
                  "index": len(self.expansions) + 1,
                  "policy": ("a bounded rescue for a ladder that is infeasible or effectively "
                             "prevents learning. Not a default reason to extend a quiet tail "
                             "ladder past 500 updates.")}
        self.expansions.append(record)
        self._flush()
        return record

    def request_fallback(self, *, family, reason, candidate=None):
        request_id = None if candidate is None else f"{family}::{candidate}"
        if request_id is not None and request_id in self.fallback_requests:
            return self.fallback_requests[request_id]
        used = int(self.fallbacks.get(family, 0))
        if used >= self.max_fallbacks_per_family:
            return {"granted": False, "family": family,
                    "reason": f"the one fallback allowed for {family} is already spent"}
        self.fallbacks[family] = used + 1
        record = {"granted": True, "family": family, "reason": str(reason),
                  "candidate": candidate,
                  "policy": "one fallback to the next already-ranked candidate, inside the same "
                            "overall time and expansion allowance"}
        if request_id is not None:
            self.fallback_requests[request_id] = record
        self._flush()
        return record

    def document(self):
        return {"schema_version": NF_SCHEMA, "record_kind": "calibration_ledger",
                "entries": [entry.document() for entry in self.entries],
                "allocated_updates": sum(max(0, int(entry.allocated_updates)
                                               - (PILOT_UPDATES if entry.resumed_from else 0))
                                         for entry in self.entries),
                "completed_updates": sum(max(0, int(entry.completed_updates)
                                               - (PILOT_UPDATES if entry.resumed_from else 0))
                                         for entry in self.entries),
                "measured_gpu_seconds": self.measured_gpu_seconds,
                "uncertainty_debit_seconds": self.uncertainty_debit_seconds,
                "overhead": self.overhead,
                "remaining_gpu_hours": self.remaining_hours(),
                "attempted": len(self.entries),
                "completed": sum(1 for entry in self.entries if entry.status == "completed"),
                "failed": sum(1 for entry in self.entries if entry.status == "failed"),
                "not_started": sum(1 for entry in self.entries if entry.status == "not_started"),
                "measured_gpu_hours": self.measured_gpu_hours,
                "gpu_hour_cap": self.gpu_hour_cap,
                "bracket_expansions": self.expansions,
                "bracket_expansion_allowance": self.max_bracket_expansions,
                "fallbacks": dict(self.fallbacks),
                "fallback_requests": dict(self.fallback_requests),
                "decisions": self.decisions,
                "non_dominated": non_dominated(self.entries),
                "completeness": ("every attempted setting appears here, including failures and "
                                 "settings that were never started because the cap was reached. "
                                 "A survivor-only table is not produced.")}


#: ``EXT:`` marks a CONTINUATION of the entry it names, not a new run from the
#: parent. The continued entry keeps the same coefficients, the same spawned
#: stream and the same trajectory directory, and resumes at
#: :data:`PILOT_UPDATES`.
EXTENSION_PREFIX = "EXT:"


def extension_of(entry_id):
    """The 500-update pilot an extension continues, or ``None``."""
    text = str(entry_id)
    return text[len(EXTENSION_PREFIX):] if text.startswith(EXTENSION_PREFIX) else None


def run_calibration(runner, *, parent_macro_ap, ledger=None, simpo=False, criteria=None):
    """Execute the declared ladders inside the cap. ``runner`` measures one pilot.

    ``runner(entry, resume_from=..., budget_seconds=...)`` must return
    ``{"metrics": {...}, "cost": {"gpu_seconds": float}, "completed_updates":
    int, "status": "completed"|"failed"|"stopped_by_gate", "reason": ...}`` for a
    SCIENTIFIC outcome, and raise :class:`MachineryFailure` when the pilot did
    not produce one. Separating it out is what lets the whole bounded policy be
    tested without a GPU while the production runner is the real trajectory loop.

    ``resume_from`` names the already-measured 500-update entry whose trajectory
    the extension continues. It is not a hint: restarting from the parent spends
    1,000 fresh updates per extension instead of 500, which turned the nominal
    7,500-update stage into 10,500.
    """
    ledger = ledger or CalibrationLedger()
    criteria = dict(criteria or POINT_RATE_CRITERIA)

    def execute(entry, *, resume_from=None):
        done = ledger.completed(entry.entry_id)
        if done is not None:
            return done                                    # already measured; the ledger persists
        entry.resumed_from = resume_from
        if ledger.remaining_seconds() <= 0:
            entry.status = "not_started"
            entry.reason = (f"the {ledger.gpu_hour_cap:g} measured GPU-hour cap was reached "
                            "before this setting started. It is recorded as not started, not "
                            "omitted.")
            return ledger.record(entry)
        result = runner(entry, resume_from=resume_from,
                        budget_seconds=ledger.remaining_seconds())
        entry.status = str(result.get("status", "completed"))
        entry.completed_updates = int(result.get("completed_updates", 0))
        entry.metrics = dict(result.get("metrics") or {})
        entry.cost = dict(result.get("cost") or {})
        entry.reason = result.get("reason")
        entry.resumed_from = resume_from
        return ledger.record(entry)

    stage_one = ladder_entries(stage="L1", task="dpo", preservation="fkl", lambdas=FKL_LAMBDAS,
                               betas=DPO_BETAS)
    for entry in stage_one:
        execute(entry)
    stage_one = [ledger.completed(entry.entry_id) or entry for entry in stage_one]
    per_beta = []
    for beta in DPO_BETAS:
        subset = [entry for entry in stage_one if float(entry.coefficients["beta"]) == float(beta)]
        choice = select_lambda(subset, parent_macro_ap=parent_macro_ap, criteria=criteria)
        block = {"beta": float(beta), **choice}
        if choice["selected"]:
            chosen = next(entry for entry in subset if entry.entry_id == choice["selected"])
            block.update(macro_average_precision=chosen.metrics.get("macro_average_precision"),
                         yield_10k=chosen.metrics.get("yield_10k"),
                         yield_1m=chosen.metrics.get("yield_1m"))
        else:
            _maybe_expand(ledger, execute, family=f"dpo_fkl_beta{beta:g}", entries=subset,
                          task="dpo", preservation="fkl", fixed={"beta": float(beta)},
                          lambdas=FKL_LAMBDAS, parent_macro_ap=parent_macro_ap,
                          criteria=criteria, block=block)
        per_beta.append(block)
    beta_choice = select_beta(per_beta)
    ledger.note({"decision": "dpo_beta", **beta_choice})

    outcomes = {"dpo_fkl": {"beta": beta_choice["selected_beta"],
                            "selection": next((block for block in per_beta
                                               if block.get("selected")
                                               == beta_choice.get("selected_entry")), None)}}
    if beta_choice["selected_beta"] is None:
        outcomes["dpo_fkl"]["outcome"] = "no qualified calibrated configuration for this family"
        ledger.note({"decision": "dpo_fkl_family",
                     "outcome": "no qualified calibrated configuration",
                     "consequence": ("the documented alternative (DPO + buffered tail) is frozen "
                                     "before challenge training and before any E scoring; if that "
                                     "family is also unqualified the block is stopped and "
                                     "reported, not silently replaced.")})

    tail_ladders = {"ipo_tail": ladder_entries(stage="L2", task="ipo", preservation="tail",
                                               lambdas=TAIL_LAMBDAS, fixed={"tau": 0.1})}
    tail_fixed = {"ipo_tail": {"tau": 0.1}}
    if beta_choice["selected_beta"] is not None:
        tail_ladders["dpo_tail"] = ladder_entries(
            stage="L2", task="dpo", preservation="tail", lambdas=TAIL_LAMBDAS,
            betas=(beta_choice["selected_beta"],))
        tail_fixed["dpo_tail"] = {"beta": float(beta_choice["selected_beta"])}
    else:
        ledger.note({"decision": "dpo_tail_ladder", "outcome": "not run",
                     "reason": "no beta was selected, and a tail ladder at an unselected beta "
                               "would not be the frozen objective"})
    for name, entries in sorted(tail_ladders.items()):
        for entry in entries:
            execute(entry)
        entries = [ledger.completed(entry.entry_id) or entry for entry in entries]
        outcomes[name] = select_lambda(entries, parent_macro_ap=parent_macro_ap, criteria=criteria)
        if not outcomes[name]["selected"]:
            _maybe_expand(ledger, execute, family=name, entries=entries,
                          task=("ipo" if name == "ipo_tail" else "dpo"), preservation="tail",
                          fixed=tail_fixed[name], lambdas=TAIL_LAMBDAS,
                          parent_macro_ap=parent_macro_ap, criteria=criteria,
                          block=outcomes[name])
        ledger.note({"decision": name, **outcomes[name]})

    extensions = {}
    for name in ("dpo_fkl", "ipo_tail", "dpo_tail"):
        selection = outcomes.get(name)
        if not selection:
            continue
        candidates = _extension_candidates(selection)
        if not candidates:
            extensions[name] = {
                "outcome": "no qualified 500-update candidate to extend",
                "consequence": ("no qualified calibrated configuration is recorded for this "
                                "family. That is a result, not a reason to invent a coefficient.")}
            continue
        extensions[name] = _extend_family(name, candidates, ledger=ledger, execute=execute,
                                          parent_macro_ap=parent_macro_ap, criteria=criteria)
    if simpo:
        ledger.note({"decision": "simpo",
                     "allowance": {"objective_pilots": SIMPO_OBJECTIVE_PILOTS,
                                   "fkl_bracket": SIMPO_FKL_BRACKET},
                     "status": "optional; admitted only after the core and audit reserves are "
                               "measured and secured"})
    return {"schema_version": NF_SCHEMA, "record_kind": "calibration_outcome",
            "parent_macro_average_precision": float(parent_macro_ap),
            "beta_choice": beta_choice, "family_outcomes": outcomes, "extensions": extensions,
            "ledger": ledger.document(),
            "frozen": _frozen_coefficients(outcomes, beta_choice, extensions),
            "freeze_rule": ("a family freezes only on an entry that COMPLETED 1,000 updates with "
                            "finite, feasible metrics and macro-AP above the parent. Anything "
                            "else is recorded as no qualified calibrated configuration."),
            "cap_note": ("the stage stops at the measured GPU-hour cap. Settings that were never "
                         "started are recorded as not started.")}


def _extension_candidates(selection):
    """Every qualified 500-update entry id for a family, best first."""
    if selection.get("selected"):
        return [selection["selected"], *(selection.get("ranked_alternatives") or [])]
    inner = selection.get("selection") or {}
    if inner.get("selected"):
        return [inner["selected"], *(inner.get("ranked_alternatives") or [])]
    return []


def _extend_family(name, candidates, *, ledger, execute, parent_macro_ap, criteria):
    """Continue the best qualified pilot to 1,000, with at most one ranked fallback.

    Each attempt CONTINUES its own 500-update trajectory rather than restarting,
    and a family freezes only on an attempt that completed 1,000 updates with
    finite, feasible metrics and macro-AP above the parent. An attempt that
    completed but failed the criteria spends the family's single permitted
    fallback, which then really runs the next already-ranked candidate and is
    itself held to the same bar.
    """
    attempts = []
    for position, entry_id in enumerate(candidates):
        if position > 0:
            fallback = ledger.request_fallback(
                family=name,
                candidate=entry_id,
                reason=(f"{candidates[position - 1]} completed 1,000 updates but did not meet "
                        "the provisional criteria"))
            attempts.append({"fallback_request": fallback})
            if not fallback["granted"]:
                break
        source = ledger.completed(entry_id)
        if source is None:
            attempts.append({"entry_id": entry_id, "outcome": "the ranked candidate is not in "
                                                              "the ledger; nothing is extended"})
            break
        extended = CalibrationEntry(
            entry_id=f"{EXTENSION_PREFIX}{entry_id}", family=source.family, task=source.task,
            preservation=source.preservation, coefficients=dict(source.coefficients),
            allocated_updates=EXTENDED_UPDATES)
        execute(extended, resume_from=entry_id)
        extended = ledger.completed(extended.entry_id) or extended
        verdict = qualifies(extended, parent_macro_ap=parent_macro_ap, criteria=criteria)
        reached = int(extended.completed_updates) >= EXTENDED_UPDATES
        attempts.append({"entry_id": extended.entry_id, "continued_from": entry_id,
                         "status": extended.status,
                         "completed_updates": int(extended.completed_updates),
                         "reached_1000": bool(reached),
                         "verdict": verdict,
                         "qualified": bool(verdict["qualified"] and reached)})
        if verdict["qualified"] and reached:
            return {"attempts": attempts, "frozen_entry": extended.entry_id,
                    "coefficients": dict(extended.coefficients),
                    "macro_average_precision": verdict["macro_average_precision"],
                    "outcome": "qualified at 1,000 updates",
                    "basis": ("continued from the 500-update pilot on the same stream identity, "
                              "completed 1,000 updates, finite and feasible metrics, macro-AP "
                              "above the parent")}
        if extended.status not in ("completed", "stopped_by_gate"):
            break                      # budget exhaustion cannot spend another continuation
    return {"attempts": attempts, "frozen_entry": None, "coefficients": None,
            "outcome": "no qualified calibrated configuration for this family",
            "consequence": ("recorded as a result. The documented alternative is frozen before "
                            "challenge training and before any E scoring; coefficients are never "
                            "invented to fill the cell.")}


def _maybe_expand(ledger, execute, *, family, entries, task, preservation, fixed, lambdas,
                  parent_macro_ap, criteria, block):
    """One bounded bracket expansion for a ladder that produced nothing usable.

    Permitted when the WHOLE ladder is infeasible, or when every rung was
    feasible but none improved on the parent -- the two ways a bracket can be in
    the wrong place. The new rung is one step outside the declared bracket in the
    indicated direction, capped globally at two expansions across the stage.
    """
    completed = [entry for entry in entries if entry.status == "completed"]
    if not completed:
        block["expansion"] = {"granted": False,
                              "reason": "no rung completed, so the bracket is not the problem"}
        return block
    verdicts = [qualifies(entry, parent_macro_ap=parent_macro_ap, criteria=criteria)
                for entry in completed]
    all_infeasible = all(not verdict["feasible"] for verdict in verdicts)
    none_improves = all(not verdict["improves_on_parent"] for verdict in verdicts)
    if all_infeasible:
        direction, value = "stronger preservation", max(float(v) for v in lambdas) * 10.0
        reason = "every rung of the declared ladder is infeasible on the development bank"
    elif none_improves:
        direction, value = "weaker preservation", min(float(v) for v in lambdas) / 10.0
        reason = ("every rung is feasible but none improves on the parent, so the preservation "
                  "term is effectively preventing learning")
    else:
        block["expansion"] = {"granted": False,
                              "reason": ("the ladder produced both feasible and improving rungs; "
                                         "its bracket is in the right place and the outcome "
                                         "stands as measured")}
        return block
    grant = ledger.request_expansion(family=family, coefficient=f"lambda -> {value:g}",
                                     reason=reason)
    block["expansion"] = dict(grant, direction=direction, lambda_value=value)
    if not grant.get("granted"):
        return block
    extra = ladder_entries(stage="X1", task=task, preservation=preservation, lambdas=(value,),
                           fixed=dict(fixed))
    for entry in extra:
        execute(entry)
    extra = [ledger.completed(entry.entry_id) or entry for entry in extra]
    widened = select_lambda(list(entries) + list(extra), parent_macro_ap=parent_macro_ap,
                            criteria=criteria)
    block["expansion"]["outcome"] = {key: widened[key] for key in
                                     ("selected", "ranked_alternatives")}
    if widened["selected"]:
        # ``select_beta`` reads a missing yield as -inf, so a block that carried
        # only its AP lost every AP tie whatever the expanded rung measured. The
        # yields come from the selected entry, exactly as on the unexpanded path.
        pool = {entry.entry_id: entry for entry in list(entries) + list(extra)}
        chosen = pool[widened["selected"]]
        block.update(selected=widened["selected"], coefficients=widened["coefficients"],
                     considered=widened["considered"],
                     ranked_alternatives=widened["ranked_alternatives"],
                     macro_average_precision=widened.get("macro_average_precision"),
                     yield_10k=chosen.metrics.get("yield_10k"),
                     yield_1m=chosen.metrics.get("yield_1m"))
        block.pop("outcome", None)
    else:
        block["considered"] = widened["considered"]
    return block


def _frozen_coefficients(outcomes, beta_choice, extensions):
    """Freeze only what a completed, qualified 1,000-update continuation produced."""
    frozen = {}
    for name in sorted(set(outcomes) | set(extensions) | {"dpo_fkl", "ipo_tail", "dpo_tail"}):
        extension = dict(extensions.get(name) or {})
        entry_id = extension.get("frozen_entry")
        coefficients = extension.get("coefficients")
        if entry_id is None or coefficients is None:
            frozen[name] = {
                "entry_id": None, "coefficients": None,
                "outcome": (extension.get("outcome")
                            or (outcomes.get(name) or {}).get("outcome")
                            or "no qualified calibrated configuration for this family"),
                "consequence": ("this family contributes NO coefficient. Arms that cite it stay "
                                "no_qualified_configuration and production never starts them.")}
            continue
        frozen[name] = {"entry_id": entry_id, "coefficients": dict(coefficients),
                        "outcome": "qualified",
                        "basis": "a completed 1,000-update continuation of a qualified pilot"}
    # Beta is chosen at the 500-update ladder and then held fixed across none /
    # FKL / tail, which is the declared order: it is not a product of the
    # 1,000-update lambda continuation and does not fall with it.
    frozen["dpo_beta"] = beta_choice.get("selected_beta")
    alternative = ("dpo_fkl" if frozen["dpo_fkl"].get("coefficients") else
                   "dpo_tail" if frozen["dpo_tail"].get("coefficients") else None)
    frozen["block_b_dpo"] = {"family": alternative,
                             "outcome": "qualified" if alternative else "unavailable",
                             "reason": "FKL preferred; qualified tail fallback fixed before B/E"}
    frozen["dpo_beta_basis"] = ("selected on the 500-update DPO+FKL ladder by macro-AP with the "
                                ".001 engineering tie rule, then held fixed across the three "
                                "preservation conditions")
    frozen["transfer_rule"] = ("Block B transfers these coefficients unchanged. Retuning per "
                               "regime would confound the proximity comparison; realized "
                               "preservation is reported rather than assumed to transfer.")
    frozen["per_seed_rule"] = "no per-seed retuning; one coefficient per family across all seeds"
    return frozen
