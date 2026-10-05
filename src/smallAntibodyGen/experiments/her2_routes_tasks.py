"""The four objectives, wired to the measurements. One module, four losses, no aliases.

This is the module whose absence made every earlier revision's ``head_only`` arm a
reconstruction loss with a different coefficient. The training loop called
``masked_nll`` for all four scheduled tasks and changed only the weight, so the
recorded ``objectives_wired`` block named four objectives that were one objective,
``ObservationHeads`` never ran a forward, and the measured values never entered the
graph at all. Nothing in a loss curve shows that.

What each task computes here, and what makes it *not* reconstruction:

* :func:`reconstruction_outcome` -- masked negative log likelihood over randomly
  masked positions of an unlabelled paired repertoire row. This one **is**
  reconstruction, and it is the only one that is.
* :func:`whole_window_outcome` -- the same likelihood over the declared common
  region under the row's **assigned quality bin**, plus a preference term between
  two conditions on the *same* row and the *same* targets: the assigned bin must
  explain the observed window better than a contrasting bin does. Two forwards,
  one difference, identical support -- which is what makes it a measurement of the
  conditioning pathway rather than of the window.
* :func:`local_outcome` -- the absolute conditional log odds fitted to
  ``kappa * delta`` on the measured margin, plus, for a complete quartet, the
  **sequentially revealed** 400-way joint fitted to the measured four-cell
  distribution. The sequential reveal is load-bearing: a product of two
  simultaneously masked marginals normalises over 400 outcomes, passes every
  normalisation check, and cannot represent an interaction.
* :func:`head_outcome` -- the observation heads on the full unmasked variant at
  ``bin=null``, with **each replicate's own** censoring in the interval
  likelihood and the reported uncertainty combined into the observation scale.

Two properties are asserted rather than trusted, because both failed before:

* every task returns an :class:`TaskOutcome` whose ``terms`` name the objectives
  that actually ran, and the loop records those names. A term that did not run
  does not appear;
* :func:`require_sensitive` measures that the loss moves when the *measurement*
  moves and when the *context* moves. A numeric objective that ignores its target
  still produces a falling curve.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np
import torch

from . import her2_routes_data as data_lib
from . import her2_routes_objectives as objectives_lib
from .her2_routes_conditions import BIN_HIGH, BIN_LOW, BIN_MID, BIN_NULL, Condition
from .her2_runtime import require
from ..models.her2_routes_heads import (
    KIND_ENRICHMENT,
    KIND_INTERVAL,
    KIND_ORDINAL,
    KIND_POINT,
    differentiable_zero,
)

#: The contrasting conditions a whole-window row may be scored against.
CONTRAST_BINS = (BIN_LOW, BIN_MID, BIN_HIGH, BIN_NULL)


@dataclass
class TaskOutcome:
    """One microbatch's contribution: a loss, its named terms, and its exposure.

    ``term`` is the coefficient key the loop multiplies by. It names the objective
    that **ran**, not the task that was scheduled, so a control arm that replaced a
    measured objective with reconstruction on the same rows is recorded as having
    trained reconstruction.
    """

    loss: object
    term: str = "reconstruction"
    terms: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    rows: int = 0
    forwards: int = 0
    targets: int = 0
    identities: tuple = ()
    measured: bool = False
    source: str = "unknown"
    family: str = data_lib.FAMILY_HER2
    scheduled_task: str = ""
    #: Per-source :class:`~.her2_routes_data.ExposureContribution` records. Their rows, forwards,
    #: targets and identities sum to this outcome's own totals (checked by :meth:`exposure`). Empty
    #: only for an outcome built without them, which then attributes everything to ``source``.
    contributions: tuple = ()

    def exposure(self):
        """The per-source contributions the ledger records, checked against this outcome's totals.

        ``rows``, ``forwards``, ``targets`` and ``identities`` are the outcome's own, written once
        where the objective ran; the contributions are the same numbers split by the source of each
        row. They must agree, or the global counters and the breakdowns would describe two
        different microbatches.
        """
        if not self.contributions:
            return (data_lib.ExposureContribution(
                source=self.source, rows=self.rows, forwards=self.forwards, targets=self.targets,
                identities=tuple(self.identities), measured=self.measured, family=self.family),)
        contributions = tuple(self.contributions)
        for name in ("rows", "forwards", "targets"):
            summed = sum(int(getattr(entry, name)) for entry in contributions)
            require(summed == int(getattr(self, name)),
                    f"The {self.term} outcome's per-source contributions sum to {summed} {name} "
                    f"but the outcome reports {int(getattr(self, name))}.")
        require(sorted(identity for entry in contributions for identity in entry.identities)
                == sorted(str(identity) for identity in self.identities),
                f"The {self.term} outcome's per-source identities are not the identities it "
                "presented.")
        return contributions

    def document(self):
        return {"term": self.term, "scheduled_task": self.scheduled_task,
                "terms": {name: float(value) for name, value in sorted(self.terms.items())},
                "rows": int(self.rows), "forwards": int(self.forwards),
                "targets": int(self.targets), "measured": bool(self.measured),
                "source": self.source, "family": self.family,
                "contributions": [entry.document() for entry in self.exposure()],
                "diagnostics": dict(self.diagnostics)}


@dataclass
class TaskDraw:
    """One microbatch of work, with every representation the arms need.

    ``views`` is populated for **every** task, including the observation-head task.
    That is what makes matched exposure achievable: an arm without a head objective
    trains reconstruction on the identical constructs, so the row presentations
    agree exactly and the arms differ only in which objective those rows fed. An
    arm that simply skipped the task would present fewer rows, and the comparison
    would confound the removed objective with a shorter schedule.
    """

    task: str
    family: str = data_lib.FAMILY_HER2
    views: tuple = ()
    contrasts: tuple = ()
    head_batch: object = None
    summaries: tuple = ()
    source: str = "unknown"

    def rows(self):
        if self.head_batch is not None:
            return int(self.head_batch.measurement.rows())
        if self.contrasts:
            return len(self.views)
        if self.views:
            return len(self.views)
        return 0

    def document(self):
        return {"task": self.task, "family": self.family, "views": len(self.views),
                "contrasts": len(self.contrasts), "summaries": len(self.summaries),
                "head_batch": self.head_batch is not None, "rows": self.rows(),
                "source": self.source}


#: Which objective each arm runs for each scheduled task. ``reconstruction`` means
#: "the same rows, the reconstruction likelihood", which is the matched control.
ARM_OBJECTIVES = {
    "full": dict.fromkeys(data_lib.TASKS, "own"),
    "reconstruction_only": dict.fromkeys(data_lib.TASKS, "reconstruction"),
    "no_functional_loss": dict.fromkeys(data_lib.TASKS, "reconstruction"),
    "head_only": {data_lib.TASK_RECONSTRUCTION: "own",
                  data_lib.TASK_LOCAL: "reconstruction",
                  data_lib.TASK_WHOLE: "reconstruction",
                  data_lib.TASK_HEADS: "own"},
    "outcome_permuted": dict.fromkeys(data_lib.TASKS, "own"),
    "background_permuted": dict.fromkeys(data_lib.TASKS, "own"),
    "local_only": {data_lib.TASK_RECONSTRUCTION: "own",
                   data_lib.TASK_LOCAL: "own",
                   data_lib.TASK_WHOLE: "reconstruction",
                   data_lib.TASK_HEADS: "own"},
}


def dispatch(services, route, model, heads, draw, *, arm="full", settings, rng,
             accumulator=None):
    """Run the objective this arm assigns to this scheduled task. One entry point.

    The loop calls only this. The arm table decides whether a measured objective
    runs or is replaced by reconstruction over the identical rows, and the returned
    ``term`` is what the coefficient is looked up under -- so a coefficient can
    scale an objective and cannot rename one.
    """
    require(arm in ARM_OBJECTIVES, f"Unknown control arm {arm!r}")
    require(draw.task in data_lib.TASKS, f"Unknown task {draw.task!r}")
    mode = ARM_OBJECTIVES[arm][draw.task]
    settings = dict(settings)
    if mode == "reconstruction" or draw.task == data_lib.TASK_RECONSTRUCTION:
        require(draw.views,
                f"{arm}/{draw.task}: the reconstruction objective needs masked views and the draw "
                "carries none. A control arm that silently skipped the microbatch would present "
                "fewer rows than the full arm and the matched-exposure comparison would confound "
                "the removed objective with a shorter schedule.")
        neutral = tuple((construct, serialization, services.condition_spec.answer_free(condition))
                        for construct, serialization, condition in draw.views)
        outcome = reconstruction_outcome(services, route, model, neutral,
                                         family=draw.family, rng=rng)
        outcome.scheduled_task = draw.task
        return outcome
    if draw.task == data_lib.TASK_WHOLE:
        outcome = whole_window_outcome(
            services, route, model, draw.views, rng=rng,
            preference_weight=float(settings.get("preference_weight", 0.0)),
            family=draw.family)
    elif draw.task == data_lib.TASK_LOCAL:
        outcome = local_outcome(
            services, route, model, draw.contrasts, kappa=float(settings["kappa"]),
            reconstruction_weight=float(settings.get("local_reconstruction_weight", 0.25)),
            interaction_weight=float(settings.get("interaction_weight", 0.0)),
            rng=rng, max_joint=int(settings.get("max_joint_per_update", 1)),
            family=draw.family)
    else:
        outcome = head_outcome(services, route, model, heads, draw.head_batch,
                               accumulator=accumulator, family=draw.family)
    outcome.scheduled_task = draw.task
    return outcome


def _window_log_probability(services, route, model, batch):
    """``(B,)`` summed realised-residue log probability over the masked queries."""
    per_target = services.window_log_probs(route, model, batch)
    return per_target.sum(dim=1)


def _masked_nll(services, route, model, batch):
    log_probs = services.canonical_log_probs(route, model, batch)
    targets = torch.as_tensor(batch.targets, dtype=torch.long, device=log_probs.device)
    mask = torch.as_tensor(batch.target_mask, device=log_probs.device)
    return objectives_lib.masked_nll(log_probs, targets, mask)


def _row_targets(batch):
    """Supervised targets per batch row, read off the collated batch's own target mask."""
    return np.asarray(batch.target_mask).astype(np.int64).sum(axis=1).tolist()


def _view_contributions(views, batch, *, family, measured, forwards=1):
    """Per-source contributions of ``views``, one entry per row, from the batch that was scored.

    Every row keeps its own construct's source: a microbatch drawn across several sources is
    attributed row by row. ``forwards`` is a per-row count (or one count for every row).
    """
    return data_lib.contributions_by_source(
        sources=[view[0].source for view in views],
        identities=[view[0].identity() for view in views],
        forwards=forwards, targets=_row_targets(batch), measured=measured, family=family)


# ---------------------------------------------------------------------------
# reconstruction
# ---------------------------------------------------------------------------

def diffusion_reconstruction_outcome(services, route, model, views, *, rng,
                                     family=data_lib.FAMILY_REPLAY):
    """Prior retention under sampled-timestep corruption (the native DPLM objective).

    Each row is presented UNMASKED to the model's ``diffusion_loss``, which draws a timestep
    ``t`` per row from ``rng`` (the checkpointed mask generator) and masks every non-special
    position with probability ``t / T``. The views' own fixed-fraction masks are not used:
    they are the pipeline's masked-editor curriculum, and applying a second corruption on
    top of them would make the objective neither one nor the other.
    """
    from .her2_routes_serial import ablang_token_ids
    require(rng is not None,
            "The diffusion reconstruction objective draws its timesteps from the checkpointed "
            "mask generator; none was supplied, and a fresh generator would break resume.")
    rows = [list(ablang_token_ids(view[0], masked_sites=())) for view in views]
    tokens, _attention = data_lib.pad_rows(rows, value=int(services.pad_id(route)))
    conditions = np.asarray([list(services.condition_spec.encode(view[2])) for view in views],
                            dtype=np.int64)
    loss, diagnostics = model.diffusion_loss(tokens, rng=rng, condition_indices=conditions)
    # The model reports ONE corrupted-position total for the batch, not one per row. With a single
    # source it belongs to that source; with several it is kept as an explicit batch-level
    # contribution, because splitting it across sources would be inventing the split.
    sources = [view[0].source for view in views]
    contributions = data_lib.contributions_by_source(
        sources=sources, identities=[view[0].identity() for view in views], forwards=1,
        targets=0, measured=False, family=family)
    total = int(diagnostics["active_targets"])
    if len(contributions) == 1:
        contributions = (dataclasses.replace(contributions[0], targets=total),)
        attribution = "single source: the corrupted-position total belongs to it"
    else:
        contributions = contributions + (data_lib.ExposureContribution(
            source=data_lib.UNATTRIBUTED_SOURCE, targets=total, measured=False, family=family),)
        attribution = ("mixed sources: the model reports one corrupted-position total per batch, "
                       "recorded as a batch-level contribution rather than split by guess")
    return TaskOutcome(
        loss=loss, term="reconstruction", terms={"reconstruction": float(loss.detach())},
        diagnostics={**dict(diagnostics), "target_attribution": attribution},
        rows=len(views), forwards=len(views), targets=total,
        identities=tuple(view[0].identity() for view in views),
        measured=False, source=views[0][0].source, family=family, contributions=contributions)


def reconstruction_outcome(services, route, model, views, *, family=data_lib.FAMILY_REPLAY,
                           rng=None):
    """Prior retention: masked NLL on an unlabelled paired row at ``bin=null``.

    A route that declares ``objective.reconstruction = "diffusion"`` trains this task with
    sampled-timestep corruption instead of the fixed mask fraction; every other route, and
    every other task of that route, is unchanged.
    """
    require(views, "A reconstruction microbatch with no views")
    diffusion = getattr(services, "uses_diffusion_reconstruction", None)
    if diffusion is not None and diffusion(route):
        return diffusion_reconstruction_outcome(services, route, model, views, rng=rng,
                                                family=family)
    batch = services.collate(route, views)
    loss, diagnostics = _masked_nll(services, route, model, batch)
    return TaskOutcome(
        loss=loss, term="reconstruction", terms={"reconstruction": float(loss.detach())},
        diagnostics=dict(diagnostics), rows=len(views), forwards=len(views),
        targets=int(diagnostics["active_targets"]),
        identities=tuple(view[0].identity() for view in views),
        measured=False, source=views[0][0].source, family=family,
        contributions=_view_contributions(views, batch, family=family, measured=False))


# ---------------------------------------------------------------------------
# whole window, with the conditional preference
# ---------------------------------------------------------------------------

def contrast_condition(condition, rng):
    """A different quality request for the same row. ``None`` at an unbinned row.

    The contrast is drawn from the remaining bins **including** ``null``, so the
    preference term is trained both against a wrong quality and against no quality
    at all. Drawing only ``null`` would train the model that conditioning helps and
    nothing about which bin was asked for.
    """
    if condition.bin == BIN_NULL:
        return None
    alternatives = [name for name in CONTRAST_BINS if name != condition.bin]
    chosen = alternatives[int(rng.integers(0, len(alternatives)))]
    return Condition(family=condition.family, bin=chosen, assay=condition.assay)


def whole_window_outcome(services, route, model, views, *, rng, preference_weight,
                         family=data_lib.FAMILY_HER2):
    """Conditioned window likelihood plus a same-row, same-target bin preference.

    The preference is ``softplus(-(log q(x | assigned) - log q(x | contrast)))``.
    Both terms score the identical realised residues over the identical region, so
    their difference is a statement about the condition and not about the window;
    that is what "fair support" means operationally, and it is why the contrast
    view is rebuilt from the same construct rather than drawn from the pool.

    Rows whose bin is ``null`` contribute the likelihood only, and the supported
    count is reported: a preference term computed over three rows is not evidence,
    and the number is in the record rather than implied by the coefficient.
    """
    require(views, "A whole-window microbatch with no views")
    batch = services.collate(route, views)
    # ONE forward for the likelihood and the assigned-condition window score. Two
    # calls would double the cost of the dominant task and the exposure ledger's
    # forward count would then describe an implementation detail.
    log_probs = services.canonical_log_probs(route, model, batch)
    targets = torch.as_tensor(batch.targets, dtype=torch.long, device=log_probs.device)
    mask = torch.as_tensor(batch.target_mask, device=log_probs.device)
    loss, diagnostics = objectives_lib.masked_nll(log_probs, targets, mask)
    picked = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    assigned_window = (picked * mask.to(picked.dtype)).sum(dim=1)
    terms = {"whole_window": float(loss.detach())}
    forwards = len(views)
    #: Student forwards per ROW: one for the assigned condition, plus one for each row that was
    #: also scored under a contrast condition. Those extra forwards belong to that row's source.
    row_forwards = [1] * len(views)
    preference = None
    supported = [index for index, view in enumerate(views) if view[2].bin != BIN_NULL]
    if supported and float(preference_weight) > 0:
        contrasted, kept = [], []
        for index in supported:
            construct, serialization, condition = views[index]
            other = contrast_condition(condition, rng)
            if other is None:
                continue
            contrasted.append((construct, serialization, other))
            kept.append(index)
        if contrasted:
            contrast_batch = services.collate(route, contrasted)
            other_probs = services.canonical_log_probs(route, model, contrast_batch)
            other_targets = torch.as_tensor(contrast_batch.targets, dtype=torch.long, device=other_probs.device)
            other_mask = torch.as_tensor(contrast_batch.target_mask, dtype=other_probs.dtype,
                                         device=other_probs.device)
            other_window = (other_probs.gather(2, other_targets.unsqueeze(-1)).squeeze(-1)
                            * other_mask).sum(dim=1)
            forwards += len(kept)
            for index in kept:
                row_forwards[index] += 1
            rows = torch.as_tensor(kept, dtype=torch.long, device=assigned_window.device)
            assigned = assigned_window.index_select(0, rows)
            preference = torch.nn.functional.softplus(-(assigned - other_window)).mean()
            loss = loss + float(preference_weight) * preference
            terms["conditional_preference"] = float(preference.detach())
            diagnostics["preference_rows"] = len(kept)
            diagnostics["preference_mean_margin"] = float(
                (assigned - other_window).detach().mean())
    diagnostics["preference_supported_rows"] = len(supported)
    diagnostics["preference_trained"] = preference is not None
    diagnostics["preference_support_note"] = (
        "a preference computed over a handful of rows is not evidence. The supported row count "
        "is recorded beside the term rather than implied by its coefficient.")
    return TaskOutcome(
        loss=loss, term="whole_window", terms=terms, diagnostics=dict(diagnostics),
        rows=len(views), forwards=forwards, targets=int(diagnostics["active_targets"]),
        identities=tuple(view[0].identity() for view in views),
        measured=True, source=views[0][0].source, family=family,
        contributions=_view_contributions(views, batch, family=family, measured=True,
                                          forwards=row_forwards))


# ---------------------------------------------------------------------------
# local measured effects
# ---------------------------------------------------------------------------

def local_outcome(services, route, model, contrasts, *, kappa, reconstruction_weight,
                  interaction_weight=0.0, rng=None, max_joint=1,
                  family=data_lib.FAMILY_HER2):
    """Fit the decoder's ABSOLUTE conditional log odds to the measured margin.

    One forward over every alternative of every contrast, all masked at the union
    of the changed coordinates, so the two members of a pair see a byte-identical
    visible background. The pair's rows are then differenced and the difference is
    fitted to ``kappa * margin`` -- the *absolute* odds, not the odds relative to a
    frozen reference, because a biased reference is carried through
    multiplicatively and can leave the better edit less probable than its
    alternative (the ablation is in ``her2_routes_objectives``).

    ``reconstruction_weight`` retains the ordinary likelihood of the alternatives
    beside the numeric term. Without it the fit can be satisfied by moving both
    conditionals apart in a way that makes neither residue likely.

    ``interaction_weight`` adds the quartet term: the sequentially revealed 400-way
    joint restricted to the four measured cells, fitted to the measured four-cell
    distribution. It is capped at ``max_joint`` quartets per microbatch because
    each one costs ``1 + residues`` scored rows.
    """
    require(contrasts, "A local microbatch with no contrasts")
    # Every endpoint of every contrast is checked BEFORE any forward: a hand-built contrast does not
    # pass through build_local_contrasts' exact-value filter, and a mixed-censored endpoint reaching a
    # quartet was reproduced training on a margin nobody measured.
    for contrast in contrasts:
        data_lib.require_exact_scalar_contrast(contrast)
    views = services.contrast_views(contrasts, route=route)
    payload = services.contrast_payload(contrasts)
    if rng is not None:
        views = list(views)
        for start, stop in payload["row_span"]:
            condition = services.condition_spec.apply_dropout(views[start][2], rng)
            for index in range(start, stop):
                views[index] = (views[index][0], views[index][1], condition)
    batch = services.collate(route, views)
    # ONE forward. The likelihood and the per-row window score come from the same
    # log probabilities, so the numeric term costs no extra forward -- which is
    # what makes the two-forwards-per-contrast figure in the exposure ledger a
    # statement about the two ALTERNATIVES and not about this implementation.
    log_probs = services.canonical_log_probs(route, model, batch)
    targets = torch.as_tensor(batch.targets, dtype=torch.long, device=log_probs.device)
    mask = torch.as_tensor(batch.target_mask, device=log_probs.device)
    nll, diagnostics = objectives_lib.masked_nll(log_probs, targets, mask)
    picked = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    window = (picked * mask.to(picked.dtype)).sum(dim=1)
    spans = payload["row_span"]
    margins = np.asarray(payload["margin"], dtype=np.float64)
    usable = [index for index, contrast in enumerate(contrasts)
              if np.isfinite(margins[index])]
    require(usable,
            "No contrast in this microbatch carries a finite measured margin, so there is no "
            "numeric target to fit. A contrast with no margin is excluded and counted by "
            "build_local_contrasts; one reaching here means the payload and the pool disagree.")
    left_rows, right_rows = [], []
    for index in usable:
        start = spans[index][0]
        offset_left, offset_right = contrasts[index].pair_rows()
        left_rows.append(start + offset_left)
        right_rows.append(start + offset_right)
    device = window.device
    left = window.index_select(0, torch.as_tensor(left_rows, dtype=torch.long, device=device))
    right = window.index_select(0, torch.as_tensor(right_rows, dtype=torch.long, device=device))
    delta = torch.as_tensor(margins[usable], dtype=window.dtype, device=device)
    numeric = torch.as_tensor([contrasts[index].left.kind != KIND_ENRICHMENT for index in usable],
                              dtype=torch.bool, device=device)
    pieces, effect_diagnostics = [], {}
    if bool(numeric.any()):
        numeric_loss, effect_diagnostics = objectives_lib.fit_absolute_log_odds(
            left[numeric], right[numeric], delta[numeric], kappa=float(kappa))
        pieces.append(numeric_loss * numeric.sum())
    ordered = ~numeric & (delta != 0)
    if bool(ordered.any()):
        ordering_loss = torch.nn.functional.softplus(
            -(left[ordered] - right[ordered]) * delta[ordered].sign()).mean()
        pieces.append(ordering_loss * ordered.sum())
        effect_diagnostics["enrichment_ordered_pairs"] = int(ordered.sum())
    count = int(numeric.sum() + ordered.sum())
    effect_loss = torch.stack(pieces).sum() / count if count else window.sum() * 0.0
    loss = effect_loss + float(reconstruction_weight) * nll
    terms = {"local_effect": float(effect_loss.detach()),
             "local_reconstruction": float(nll.detach())}
    forwards = len(views)
    #: ``(source, forwards)`` of every quartet joint: scored rows that are no microbatch row.
    joint_forwards = []
    interaction = {"quartets_available": 0, "quartets_trained": 0}
    coupled = [index for index in usable
               if contrasts[index].kind == data_lib.CONTRAST_COUPLED
               and contrasts[index].left.kind != KIND_ENRICHMENT
               and contrasts[index].interaction is not None]
    interaction["quartets_available"] = len(coupled)
    if coupled and float(interaction_weight) > 0:
        chosen = coupled[:max(0, int(max_joint))]
        joint_terms = []
        for index in chosen:
            block = quartet_joint_outcome(services, route, model, contrasts[index],
                                         kappa=float(kappa))
            if block is None:
                continue
            joint_terms.append(block["loss"])
            forwards += int(block["forwards"])
            # The joint is scored against the quartet's background, so its forwards are that
            # construct's source's.
            joint_forwards.append((contrasts[index].background.construct.source,
                                   int(block["forwards"])))
            interaction.setdefault("dependence", []).append(block["max_total_variation"])
        if joint_terms:
            joint = torch.stack(joint_terms).mean()
            loss = loss + float(interaction_weight) * joint
            terms["coupled_joint"] = float(joint.detach())
            interaction["quartets_trained"] = len(joint_terms)
    return TaskOutcome(
        loss=loss, term="local_effect", terms=terms,
        diagnostics={**diagnostics, **effect_diagnostics, "interaction": interaction,
                     "contrasts": len(contrasts), "with_margin": len(usable),
                     "by_kind": {kind: sum(1 for c in contrasts if c.kind == kind)
                                 for kind in data_lib.CONTRAST_KINDS}},
        rows=len(views), forwards=forwards,
        targets=int(diagnostics["active_targets"]),
        identities=tuple(view[0].identity() for view in views),
        measured=True, source=views[0][0].source, family=family,
        contributions=data_lib.with_extra_forwards(
            _view_contributions(views, batch, family=family, measured=True), joint_forwards))


def quartet_joint_outcome(services, route, model, contrast, *, kappa):
    """The sequentially revealed two-site joint, fitted to the measured quartet.

    The model's joint over the two coordinates is built by revealing the first
    residue **before** scoring the second, which is the only construction that can
    express an interaction. The measured target is a distribution over the four
    cells the quartet measures -- background, A, B and AB -- from
    ``softmax(kappa * effect)``, and the loss is the forward KL of that target
    against the model's joint restricted and renormalised over those same four
    cells.

    Returns ``None`` when the quartet's residues are not all canonical or the two
    coordinates are not distinct, rather than fabricating a cell.
    """
    from .her2_routes_serial import canonical_index
    require(len(contrast.sites) == 2, "A quartet names exactly two coordinates")
    # Before the joint is scored: every cell's endpoint must be an exact scalar of one assay.
    data_lib.require_exact_scalar_contrast(contrast, label="quartet")
    background = contrast.background
    if background is None or contrast.double is None:
        return None
    first, second = contrast.sites
    if first == second:
        return None
    cells = {}
    for label, record in (("background", background), ("left", contrast.left),
                          ("right", contrast.right), ("double", contrast.double)):
        try:
            a = canonical_index(record.construct.residue_at(first))
            b = canonical_index(record.construct.residue_at(second))
        except ValueError:
            return None
        cells[label] = (a, b)
    log_joint, forwards, dependence = services.joint_support(
        route, model, background, sites=(first, second),
        condition=Condition(family=background.family, bin=BIN_NULL, assay=background.assay),
        first_residues=tuple(sorted({a for a, _ in cells.values()})))
    order = ("background", "left", "right", "double")
    effects = np.asarray([0.0,
                          float(contrast.effect_left or 0.0),
                          float(contrast.effect_right or 0.0),
                          float(contrast.double_effect or 0.0)], dtype=np.float64)
    picked = torch.stack([log_joint[cells[name][0], cells[name][1]] for name in order])
    if background.family == "enrichment":
        # Enrichment units have no shared physical affinity scale. The four-cell
        # ordering still supervises background-dependent effects without assigning
        # a pKD multiplier to this study's enrichment statistic.
        comparisons = [(i, j) for i in range(4) for j in range(4)
                       if effects[i] > effects[j]]
        if not comparisons:
            return None
        loss = torch.stack([torch.nn.functional.softplus(-(picked[i] - picked[j]))
                            for i, j in comparisons]).mean()
        return {"loss": loss, "forwards": int(forwards),
                "max_total_variation": float(dependence), "target": None,
                "model": picked.detach().softmax(0).cpu().tolist(),
                "rule": "within-assay ordering of the measured quartet; no physical affinity scale"}
    scaled = float(kappa) * effects
    target = np.exp(scaled - scaled.max())
    target = target / target.sum()
    restricted = torch.log_softmax(picked, dim=0)
    target_tensor = torch.as_tensor(target, dtype=restricted.dtype, device=restricted.device)
    loss = torch.sum(target_tensor * (torch.log(target_tensor) - restricted))
    return {"loss": loss, "forwards": int(forwards),
            "max_total_variation": float(dependence),
            "target": [float(value) for value in target],
            "model": [float(value) for value in restricted.detach().exp()],
            "rule": ("forward KL of the measured four-cell distribution against the model's "
                     "sequentially revealed joint, restricted and renormalised over those cells. "
                     "A product of two simultaneously masked marginals also normalises over 400 "
                     "outcomes and cannot move this term.")}


# ---------------------------------------------------------------------------
# the observation heads
# ---------------------------------------------------------------------------

def head_outcome(services, route, model, heads, batch, *, accumulator=None,
                 family=data_lib.FAMILY_HER2, enrichment_pairs=None):
    """Run :class:`ObservationHeads` on the full unmasked variants. Four likelihoods.

    The representation comes from the trunk, so the auxiliary gradient reaches the
    shared encoder and not only the head: a head that trains on a frozen
    representation measures the head.

    The interval likelihood is evaluated **per replicate**, with each replicate's
    own censoring and reported uncertainty, plus learned assay noise, and the
    per-replicate terms are weighted to sum to one inside a construct. Evaluating
    the row aggregate is how a mixture of one floor replicate and two point
    replicates gets trained as an exact point at the mean of the two.
    """
    require(isinstance(batch, data_lib.HeadBatch), "head_outcome takes a HeadBatch")
    # The ENCODED condition is re-checked here, not a fabricated one. ``collate_heads``
    # checks the Condition objects; by this point they are integer indices, and the bin
    # column is the one the head must not be able to read the answer from.
    null_bin = services.condition_spec.encode(
        Condition(family="affinity", bin=BIN_NULL, assay=batch.assays[0]))[1]
    encoded = np.asarray(batch.condition_indices, dtype=np.int64)
    require(bool((encoded[:, 1] == int(null_bin)).all()),
            "A head batch reached the objective with a non-null quality bin in its ENCODED "
            "conditions. A head that saw the bin it is predicting measures the condition "
            "embedding rather than the representation, and the check on the Condition objects "
            "in collate_heads does not cover the encoded array the model actually reads.")
    representation = services.representation(route, model, batch)
    measurement = batch.measurement
    device = representation.device
    kinds = np.asarray(measurement.kind)
    terms, counts = {}, {}
    total = differentiable_zero(representation)
    assay_indices = torch.as_tensor(measurement.assay_index, dtype=torch.long, device=device)

    ordinal_mask = torch.as_tensor(
        (measurement.kind_mask(KIND_ORDINAL) *
         (np.asarray(measurement.ordinal_class) >= 0)).astype(np.float64),
        dtype=representation.dtype, device=device)
    classes = torch.as_tensor(np.clip(np.asarray(measurement.ordinal_class), 0, None),
                              dtype=torch.long, device=device)
    ordinal_loss, ordinal_rows = heads.ordinal_loss(representation, classes, ordinal_mask)
    if ordinal_rows:
        total = total + ordinal_loss
        terms["ordinal"] = float(ordinal_loss.detach())
    counts[KIND_ORDINAL] = int(ordinal_rows)

    replicate_rows = np.asarray(measurement.replicate_row, dtype=np.int64)
    interval_kinds = (measurement.kind_mask(KIND_INTERVAL)
                      + measurement.kind_mask(KIND_POINT))
    replicate_mask = interval_kinds[replicate_rows].astype(np.float64) * np.asarray(
        measurement.replicate_weight, dtype=np.float64)
    if replicate_rows.size and float(replicate_mask.sum()) > 0:
        rows = torch.as_tensor(replicate_rows, dtype=torch.long, device=device)
        expanded = representation.index_select(0, rows)
        if measurement.replicate_uncertainty is None:
            reported = torch.as_tensor(measurement.uncertainty, dtype=representation.dtype,
                                       device=device).index_select(0, rows)
        else:
            reported = torch.as_tensor(measurement.replicate_uncertainty,
                                       dtype=representation.dtype, device=device)
        interval_loss, interval_rows = heads.interval_loss(
            expanded,
            torch.as_tensor(measurement.replicate_lower, dtype=representation.dtype,
                            device=device),
            torch.as_tensor(measurement.replicate_upper, dtype=representation.dtype,
                            device=device),
            torch.as_tensor(measurement.replicate_finite_lower, device=device),
            torch.as_tensor(measurement.replicate_finite_upper, device=device),
            assay_indices.index_select(0, rows),
            torch.as_tensor(replicate_mask, dtype=representation.dtype, device=device),
            reported=reported)
        if interval_rows:
            total = total + interval_loss
            terms["interval"] = float(interval_loss.detach())
        counts[KIND_INTERVAL] = int(interval_rows)
        counts["replicate_rows"] = int(replicate_rows.size)
        counts["censored_replicates"] = int(np.asarray(measurement.replicate_censored).sum())
    else:
        counts[KIND_INTERVAL] = 0

    enrichment = measurement.kind_mask(KIND_ENRICHMENT)
    pairs = enrichment_pairs if enrichment_pairs is not None else \
        _enrichment_pairs(measurement, enrichment)
    if pairs["left"]:
        enrichment_loss, enrichment_rows = heads.enrichment_loss(
            representation,
            torch.as_tensor(pairs["left"], dtype=torch.long, device=device),
            torch.as_tensor(pairs["right"], dtype=torch.long, device=device),
            torch.as_tensor([1.0] * len(pairs["left"]), dtype=representation.dtype,
                            device=device))
        if enrichment_rows:
            total = total + enrichment_loss
            terms["enrichment"] = float(enrichment_loss.detach())
        counts[KIND_ENRICHMENT] = int(enrichment_rows)
    else:
        counts[KIND_ENRICHMENT] = 0
    if accumulator is not None:
        for kind, name in ((KIND_ORDINAL, "ordinal"), (KIND_INTERVAL, "interval"),
                           (KIND_ENRICHMENT, "enrichment")):
            if counts.get(kind):
                accumulator.add(kind, terms.get(name, 0.0), counts[kind])
    require(terms,
            "Every observation-head likelihood saw zero eligible rows in this microbatch, so the "
            "head contributed a differentiable zero and nothing else. That is a scheduling or "
            "eligibility defect, not a measurement: the stream drew rows the heads cannot read.")
    return TaskOutcome(
        loss=total, term="heads", terms={"heads": float(total.detach()), **terms},
        diagnostics={"eligible_rows": counts,
                     "by_kind": {str(name): int((kinds == name).sum())
                                 for name in sorted(set(kinds.tolist()))},
                     "answer_free": True,
                     "measurement": measurement.document()},
        rows=int(measurement.rows()), forwards=int(measurement.rows()),
        targets=int(measurement.rows()),
        identities=tuple(batch.identities), measured=True,
        source=batch.sources[0] if batch.sources else "unknown", family=family,
        contributions=data_lib.contributions_by_source(
            sources=(list(batch.sources) if len(batch.sources) == len(batch.identities)
                     else ["unknown"] * len(batch.identities)),
            identities=list(batch.identities), forwards=1, targets=1, measured=True,
            family=family))


def _enrichment_pairs(measurement, mask, *, maximum=64):
    """Within-assay ordered pairs among the enrichment rows, higher value first.

    A ranking loss needs pairs and the batch carries rows. Pairs are formed only
    inside one assay identity, because an enrichment ratio is a study-internal
    quantity and a pair across two studies trains a calibration nobody has.
    """
    rows = np.flatnonzero(np.asarray(mask) > 0)
    values = np.asarray(measurement.value, dtype=np.float64)
    assays = np.asarray(measurement.assay_index, dtype=np.int64)
    left, right = [], []
    for assay in sorted(set(int(assays[row]) for row in rows)):
        members = [int(row) for row in rows
                   if int(assays[row]) == assay and np.isfinite(values[row])]
        members.sort(key=lambda row: -float(values[row]))
        for position in range(len(members) - 1):
            if len(left) >= int(maximum):
                break
            if values[members[position]] == values[members[position + 1]]:
                continue
            left.append(members[position])
            right.append(members[position + 1])
    return {"left": left, "right": right, "maximum": int(maximum),
            "rule": "adjacent pairs of the within-assay ordering, higher value first"}


# ---------------------------------------------------------------------------
# the sensitivity probes
# ---------------------------------------------------------------------------

def require_sensitive(build, *, label, perturb, tolerance=1e-9):
    """The loss must MOVE when the thing it claims to fit moves. Measured.

    ``build()`` returns a scalar loss; ``perturb()`` returns a second one built from
    a changed measurement or a changed context. A numeric objective that ignores
    its target still produces a falling curve and every gradient check still
    passes, so the check is a difference and not a shape.
    """
    base = float(build())
    moved = float(perturb())
    difference = abs(moved - base)
    require(difference > float(tolerance),
            f"{label}: the loss did not move ({base} -> {moved}) when its input changed. An "
            "objective insensitive to the quantity it names is the failure this probe exists "
            "for; a falling curve does not exclude it.")
    return {"base": base, "perturbed": moved, "difference": difference,
            "tolerance": float(tolerance), "sensitive": True}


def gradient_fractions(outcomes, parameters):
    """Each task's share of the gradient norm, measured one task at a time.

    Reported beside its share of updates. A task with thirty per cent of the
    updates and three per cent of the gradient is not contributing thirty per cent,
    and that is an amendment signal rather than a number to report and move past.
    """
    norms = {}
    for name, loss in dict(outcomes).items():
        grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        total = 0.0
        for grad in grads:
            if grad is not None:
                total += float(grad.detach().pow(2).sum())
        norms[name] = float(total ** 0.5)
    return data_lib.task_gradient_fractions(norms)
