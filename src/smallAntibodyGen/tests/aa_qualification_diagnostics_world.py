"""Analytical software backends and plan helpers for the diagnostics/runner tests.

Every sequence, name and value is an invented test string. The backends are toys with closed-form answers:
none is, or stands in for, a native model, and nothing measured by them says anything about MINT.
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from smallAntibodyGen.experiments.interaction_qualification.arms import admit_by_joint_length, planned_feature_forwards
from smallAntibodyGen.experiments.interaction_qualification.config import QualificationConfig, parse_config
from smallAntibodyGen.experiments.interaction_qualification.diagnostics import DiagnosticPlan, plan_diagnostics
from smallAntibodyGen.models.antibody_antigen.denoising import N_CANONICAL, BackendOutput
from smallAntibodyGen.models.antibody_antigen.views import CANONICAL_RESIDUES, Coord, VisibleView
from smallAntibodyGen.tests import aa_qualification_world as world

# A positive 20 x 20 joint over two sites: ones everywhere except these four cells. Total 411.
DEFAULT_WEIGHTS: Mapping[Tuple[str, str], float] = {("A", "A"): 4.0, ("A", "C"): 1.0, ("C", "A"): 2.0, ("C", "C"): 8.0}
TOTAL = 411.0
ROW_SUM = {"A": 23.0, "C": 28.0}  # marginal weight of the FIRST site
COLUMN_SUM = {"A": 24.0, "C": 27.0}  # marginal weight of the SECOND site


class JointTwoSiteBackend:
    """Exact conditionals of one joint over two sites, computed from the VISIBLE state only.

    Both sites hidden: each site's marginal. One hidden: its conditional given the visible one. Any other
    position is flat. Nothing about the truth, an arm or a record reaches it.
    """

    def __init__(self, sites: Tuple[Coord, Coord], weights: Optional[Mapping[Tuple[str, str], float]] = None) -> None:
        self.sites = sites
        self.w = torch.ones(N_CANONICAL, N_CANONICAL, dtype=torch.float64)
        for (x, y), value in (DEFAULT_WEIGHTS if weights is None else weights).items():
            self.w[CANONICAL_RESIDUES.index(x), CANONICAL_RESIDUES.index(y)] = value
        self.forwards = 0

    def forward(self, view: VisibleView) -> BackendOutput:
        self.forwards += 1
        coords = view.coords()
        first, second = self.sites
        a, b = view.token(first), view.token(second)
        logits = torch.zeros(len(coords), N_CANONICAL)
        rows: Dict[Coord, torch.Tensor] = {}
        if a is None and b is None:
            rows[first], rows[second] = self.w.sum(dim=1), self.w.sum(dim=0)
        elif a is None:
            rows[first] = self.w[:, CANONICAL_RESIDUES.index(b)]
        elif b is None:
            rows[second] = self.w[CANONICAL_RESIDUES.index(a), :]
        for site, weight in rows.items():
            logits[view.position(site)] = torch.log(weight / weight.sum()).float()
        return BackendOutput(
            coords=coords,
            raw_logits=logits,
            canonical_mass=torch.full((len(coords),), 0.9),
            features=torch.zeros(len(coords), 1),
            mask_ratio=len(view.hidden_sites()) / len(coords),
        )


class AntigenPullBackend:
    """Toy whose hidden-site logits depend on whether an antigen chain is VISIBLE in the view.

    With an antigen present, every hidden site has C at ln 19 and the other 19 residues at 0, so p(C) = 1/2 and
    every other residue has p = 1/38; without one, all 20 logits are 0 and every residue has p = 1/20. The
    existing execution backends ignore the antigen, so their joint-minus-antibody-only differences are always
    zero; this one makes them nonzero. It is plumbing evidence only (the right arm gets the right number) and
    says nothing about the antigen specificity of any real model.
    """

    PULL = math.log(19.0)

    def forward(self, view: VisibleView) -> BackendOutput:
        coords = view.coords()
        logits = torch.zeros(len(coords), N_CANONICAL)
        if view.has_role("Ag"):
            for site in view.hidden_sites():
                logits[view.position(site), CANONICAL_RESIDUES.index("C")] = self.PULL
        return BackendOutput(
            coords=coords,
            raw_logits=logits,
            canonical_mass=torch.full((len(coords),), 0.9),
            features=torch.zeros(len(coords), 1),
            mask_ratio=len(view.hidden_sites()) / len(coords),
        )


def make_config(**kwargs: Any) -> QualificationConfig:
    return parse_config(json.dumps(world.config_dict(**kwargs)))


def region_of(heavy: str) -> List[Tuple[str, int]]:
    return [("H", index) for index in range(len(heavy))]


def drow(
    record_id: str,
    heavy: str,
    *,
    partition: str = "development",
    source: str = "s1",
    assay: str = "a1",
    value: float = 7.0,
    antigen: str = "KLM",
    light: Optional[str] = None,
    region: Any = "heavy",
    **kwargs: Any,
) -> Dict[str, Any]:
    """A record row whose region defaults to every heavy-chain site."""
    chosen = region_of(heavy) if region == "heavy" else region
    return world.row(
        record_id, heavy=heavy, light=light, antigen=antigen, partition=partition, source=source, assay=assay,
        value=value, region=chosen, **kwargs,
    )


def plan_for(rows: Sequence[Dict[str, Any]], config: Optional[QualificationConfig] = None, regime: str = "both") -> DiagnosticPlan:
    """The plan the runner would build for ``rows`` (sealed test withheld unless the config unseals it)."""
    config = make_config() if config is None else config
    panel = world.panel_of(list(rows), regime)
    active = [record for record in panel.records if config.evaluate_test or record.partition != "test"]
    length = admit_by_joint_length(active, config.max_tokens)
    return plan_diagnostics(
        length.eligible,
        config.diagnostics,
        planned_feature_forwards(length.eligible),
        allow_test=config.evaluate_test,
    )


def signature(plan: DiagnosticPlan) -> Dict[str, Any]:
    """What a plan decides about CONTENT: no record id, source, assay or measured value is in it."""
    return {
        "budget": plan.budget.to_dict(),
        "singletons": [
            (
                task.task_id,
                task.priority,
                task.pair_fingerprint,
                [(c.count, c.geometry, c.sites, c.order, c.reverse_order, c.view_digest, c.cost) for c in task.cases],
            )
            for task in plan.singleton_tasks
        ],
        "pairs": [
            (t.task_id, t.priority, t.a_fingerprint, t.b_fingerprint, t.changed, t.order, t.reverse_order, t.cost)
            for t in plan.pair_tasks
        ],
    }


LN20 = math.log(20.0)
