"""Durable work-unit accounting, including explicit uncertainty after a crash."""
from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path

from . import her2_support_paths as paths
from .her2_replay_campaign import CostLedger as BaseCostLedger
from .her2_runtime import require


class BudgetExhausted(RuntimeError):
    pass


class WorkBudget:
    """Admit bounded GPU work only while its conservative reservation fits.

    GPU-work wall time includes the host dispatch of that work. A killed unit
    has unknown duration: its reservation is a separately labelled uncertainty
    debit, never invented measured time. A single native call cannot be preempted;
    any reservation overrun is recorded and prohibits further admission.
    """

    def __init__(self, path, *, identity, limit_seconds, clock=time.perf_counter,
                 synchronize=None):
        self.path, self.clock, self.synchronize = Path(path), clock, synchronize
        self.data = {"identity": identity, "limit_seconds": float(limit_seconds),
                     "measured_seconds": 0.0, "uncertainty_debit_seconds": 0.0,
                     "categories": {}, "pending": None, "overruns": []}
        if self.path.is_file():
            self.data = paths.read_json(self.path)
            require(self.data["identity"] == identity, "Budget belongs to another pilot")
            self.data["limit_seconds"] = min(float(limit_seconds), self.data["limit_seconds"])
            pending = self.data.get("pending")
            if pending:
                self.data["uncertainty_debit_seconds"] += pending["reserved_seconds"]
                self.data.setdefault("interrupted_units", []).append(pending)
                self.data["pending"] = None
        self.flush()

    @property
    def charged_seconds(self):
        return self.data["measured_seconds"] + self.data["uncertainty_debit_seconds"]

    @property
    def remaining_seconds(self):
        return max(0.0, self.data["limit_seconds"] - self.charged_seconds)

    def flush(self):
        paths.write_json(self.path, self.data)

    @contextmanager
    def unit(self, category, *, reserve_seconds):
        require(self.data["pending"] is None, "GPU budget units cannot overlap")
        history = self.data["categories"].get(category, {})
        reserve = max(float(reserve_seconds), 1.5 * history.get("max_seconds", 0.0))
        if self.remaining_seconds < reserve or self.data["overruns"]:
            raise BudgetExhausted(f"Insufficient calibration allowance for {category}: "
                                  f"{self.remaining_seconds:.3f}s remains, {reserve:.3f}s reserved")
        self.data["pending"] = {"category": category, "reserved_seconds": reserve}
        self.flush()
        if self.synchronize:
            self.synchronize()
        started = self.clock()
        try:
            yield
        finally:
            if self.synchronize:
                self.synchronize()
            elapsed = max(0.0, self.clock() - started)
            self.data["measured_seconds"] += elapsed
            block = self.data["categories"].setdefault(category, {"seconds": 0.0,
                                                                 "units": 0, "max_seconds": 0.0})
            block["seconds"] += elapsed
            block["units"] += 1
            block["max_seconds"] = max(block["max_seconds"], elapsed)
            if elapsed > reserve:
                self.data["overruns"].append({"category": category, "measured_seconds": elapsed,
                                               "reserved_seconds": reserve})
            self.data["pending"] = None
            self.flush()


class CostLedger(BaseCostLedger):
    """Restore category totals; persist every completed segment rather than every pilot."""

    def __init__(self, *, path=None, synchronize=None, budget=None):
        super().__init__(synchronize=synchronize)
        self.path = None if path is None else Path(path)
        self.budget = budget
        self.prior_wall = 0.0
        if self.path is not None and self.path.is_file():
            saved = paths.read_json(self.path)
            self.seconds.update(saved["seconds"])
            self.counts.update(saved["segments"])
            self.prior_wall = float(saved["wall_seconds"])

    @property
    def wall_seconds(self):
        return self.prior_wall + super().wall_seconds

    @contextmanager
    def segment(self, category):
        from contextlib import nullcontext
        reserve = {"optimizer": 5.0, "gate": 150.0, "preservation": 1.0,
                   "evaluation": 180.0, "generation": 180.0, "teacher_cache": 180.0}
        admission = (self.budget.unit(category, reserve_seconds=reserve[category])
                     if self.budget is not None and category in reserve else nullcontext())
        try:
            with admission, super().segment(category):
                yield
        finally:
            if self.path is not None:
                paths.write_json(self.path, self.document())
