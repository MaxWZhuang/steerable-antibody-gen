"""T4: weighted conditional NLL, numeric margin matching, and local IPO/DPO with a reuse cap.

Four candidates, and the existing evidence constrains this stage more than the plan
assumed. On the display panel IPO without preservation was the best arm (macro-AP
.8776, Y@10k 573.3) and had the **worst** off-library rate (.5290); IPO + forward KL
had the best audit mean (.0753); the parent mixture at alpha .89 was the only
configuration with **zero** tenfold and hundredfold tail events. None of them
improved quantitative affinity ordering. Preference optimisation here trades
preservation against display ranking and leaves affinity untouched, so this stage is
designed to break that pattern rather than reproduce it -- which is why *conditional
supervised fitting on measured values* is the first arm and the preference arms are
compared against it, not against nothing.

**Contract version 2.** The first revision offered ``conditional_fit`` and
``numeric_margin_match`` as two arms. They were one objective: ``conditional_fit`` fitted
``log q(chosen) - log q(rejected)`` to ``kappa * delta`` and so did ``numeric_margin_match``,
so the comparison ran the same loss under two names. Version 2 makes them different
objectives and says what each one is:

* ``weighted_conditional_nll`` -- the full-vocabulary categorical negative log likelihood
  of the two measured alternatives against **pair-local soft targets**
  ``sigma(kappa * delta)``. It reads ``log q`` of each alternative out of the model's
  normalised distribution over the whole vocabulary, so it is sensitive to how much mass
  the alternatives hold in total. It is **not** an exact tilted-prior fit: the target is
  supported on the two measured alternatives, so it drives every unmeasured residue
  toward zero mass.
* ``numeric_margin_match`` -- **one-site** numeric log-odds matching,
  ``log q(chosen) - log q(rejected) -> kappa * delta``. It sees only the *ratio* of the two
  alternatives, so a model that gives the pair the right odds while holding a tenth of the
  mass has zero loss and zero gradient.
* ``local_ipo`` / ``local_dpo`` -- the inherited preference objectives, unchanged.

The legacy ``conditional_fit`` name is refused by :func:`resolve_methods` for selection;
historical ``posttrain/conditional_fit/last.pt`` artifacts and recipe strings stay
readable because nothing here parses them.

Two rules are enforced mechanically:

* **The reference is frozen after mid-training and never moves.** A live reference
  turns a preference objective into a joint optimisation of both sides and the
  recorded ``reference-relative`` label becomes false.
* **A measured pair may be reused a declared number of times.** Refresh draws
  difficult comparisons from *eligible measured training pairs*, choosing a background
  first, then a substitution within it, then a candidate (:class:`PairBank` states exactly
  which marginal that balances and which it does not), and the cap is a config
  field. Model-generated candidates do **not** acquire measured affinities by being
  generated, so :func:`require_measured_pairs` refuses a refresh pool containing one.
"""
from __future__ import annotations

import contextlib
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch

from . import her2_objectives as inherited
from .her2_routes_objectives import fit_absolute_log_odds, fit_reference_relative_log_odds
from .her2_runtime import require

#: The post-training contract this module implements. Config, records and checkpoint
#: identity all carry it, so weights and pools produced under version 1 semantics cannot be
#: resumed or loaded as version 2.
CONTRACT_VERSION = 2

#: Retained as a constant ONLY so historical artifacts and recipes that name it stay
#: readable. It is not a selectable method: see :data:`LEGACY_METHODS`.
METHOD_CONDITIONAL_FIT = "conditional_fit"
METHOD_WEIGHTED_CONDITIONAL_NLL = "weighted_conditional_nll"
METHOD_MARGIN_MATCH = "numeric_margin_match"
METHOD_IPO = "local_ipo"
METHOD_DPO = "local_dpo"
METHODS = (METHOD_WEIGHTED_CONDITIONAL_NLL, METHOD_MARGIN_MATCH, METHOD_IPO, METHOD_DPO)

#: Names a version-1 config may carry that version 2 refuses to run, with the migration.
LEGACY_METHODS = {
    METHOD_CONDITIONAL_FIT: (
        "'conditional_fit' was numerically identical to 'numeric_margin_match' (both fitted "
        "log q(chosen) - log q(rejected) to kappa * delta), so selecting it ran one objective "
        "under two names. Use 'weighted_conditional_nll' for the full-vocabulary weighted "
        "categorical likelihood, or 'numeric_margin_match' for explicit one-site log-odds "
        "matching. Historical posttrain/conditional_fit artifacts stay readable; they cannot "
        "be selected to run."),
}

#: What each selectable method optimises, stated once so a record cannot drift from the loss.
OBJECTIVES = {
    METHOD_WEIGHTED_CONDITIONAL_NLL: (
        "full-vocabulary weighted categorical NLL of the two measured alternatives against "
        "pair-local soft targets sigma(kappa * delta). Sensitive to the alternatives' total "
        "mass. Not an exact tilted-prior fit."),
    METHOD_MARGIN_MATCH: (
        "one-site numeric log-odds matching: log q(chosen) - log q(rejected) -> kappa * delta. "
        "Invariant to the alternatives' total mass."),
    METHOD_IPO: "inherited local IPO against the frozen reference",
    METHOD_DPO: "inherited local DPO against the frozen reference",
}

#: The inherited IPO temperature. A config naming another value is a different
#: experiment, not a variant.
IPO_TAU = 0.1


def resolve_methods(methods):
    """Validate ``posttrain.methods`` -- and refuse the legacy name -- before any model loads.

    Returns the methods as a tuple. A legacy name raises with its migration; an unknown
    name, a duplicate or an empty list raises with what is accepted. This runs first in
    :meth:`FlightServices.run_posttrain`, so a stale config fails in milliseconds instead
    of after a checkpoint load.
    """
    require(isinstance(methods, (list, tuple)) and len(methods) > 0,
            "posttrain.methods must be a non-empty list")
    names = [str(name) for name in methods]
    for name in names:
        if name in LEGACY_METHODS:
            raise ValueError(f"posttrain.methods: {LEGACY_METHODS[name]}")
    unknown = sorted({name for name in names if name not in METHODS})
    require(not unknown, f"posttrain.methods names unknown method(s) {unknown}; "
                         f"expected a subset of {list(METHODS)}")
    duplicated = sorted({name for name in names if names.count(name) > 1})
    require(not duplicated,
            f"posttrain.methods lists {duplicated} more than once. Each arm gets its own bank "
            "and its own checkpoint directory; a repeat would overwrite the first.")
    return tuple(names)


def require_contract(block):
    """The config's ``posttrain`` block must declare this contract version. Returns it."""
    version = dict(block or {}).get("contract_version")
    require(version == CONTRACT_VERSION and not isinstance(version, bool),
            f"posttrain.contract_version is {version!r} but this stage implements version "
            f"{CONTRACT_VERSION}. A configuration written for the earlier T4 -- a duplicated "
            "conditional-fit arm, pairs fixed to the HER2 assay and background strata keyed on "
            "a 24-residue sequence prefix -- would run under different semantics than it "
            "declares. Use configs/experiments/her2_generative_routes_t4v2.json or declare "
            f"contract_version: {CONTRACT_VERSION} in a new config and run root.")
    return int(version)


def arm_identity(method):
    """The ``arm`` a post-training checkpoint's training identity records: method + version."""
    require(method in METHODS, f"{method!r} is not a v{CONTRACT_VERSION} method")
    return f"{method}.v{CONTRACT_VERSION}"


@contextlib.contextmanager
def deterministic_scoring(model):
    """``model.eval()`` with gradients left ON; every module's training flag is restored.

    The preference and fitting objectives compare ``log q`` of the two members of a pair,
    and the two halves of a pair are separate forwards of one masked input. In training
    mode dropout makes them two different conditionals -- the objective then measures
    dropout noise -- and a ``no_grad`` evaluation would remove the gradient the update
    needs. This keeps the forward deterministic and differentiable. It restores the flag of
    every submodule, not only the top-level one, because a model may hold a module that was
    deliberately left in a different mode.
    """
    flags = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield model
    finally:
        for module, flag in flags:
            module.training = flag


def require_deterministic_scoring(model, *, label="policy", gradients=True):
    """Refuse a scoring forward with any module in training mode.

    ``gradients=True`` (the policy) also refuses a forward under ``no_grad``: a training
    objective scored without a graph has nothing to update with. The frozen reference passes
    ``gradients=False`` -- it is evaluated under ``no_grad`` on purpose.
    """
    training = [name or "<root>" for name, module in model.named_modules() if module.training]
    require(not training,
            f"{label}: {len(training)} module(s) are in training mode, e.g. {training[:3]}. "
            "Dropout would give the two halves of a pair different conditionals.")
    if gradients:
        require(torch.is_grad_enabled(),
                f"{label}: gradients are disabled. A training objective scored under no_grad has "
                "nothing to update with; use eval mode with gradients on.")
    return True


def schedule_digest(drawn_identities):
    """SHA-256 of an arm's draw schedule: the pair digests drawn at each step, in order.

    Two arms that drew the same pairs in the same order at every step share this digest,
    which is how the record shows the arms differ in their objective alone.
    """
    import hashlib
    import json
    text = json.dumps([list(map(str, step)) for step in drawn_identities],
                      separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def training_targets_digest(*, pair_pool_sha256, pair_targets_sha256, draw_schedule_sha256,
                            target_schedule_sha256):
    """The single value bound into a post-training checkpoint's ``schedule_sha256`` identity.

    The structural pool digest and the draw schedule say *which pairs* an arm trained on and in
    what order; the measured-target digests say *what values and winners* it was told. Binding
    all four means two arms (or two runs) that drew the same pairs from a pool whose measured
    values, winners or provenance changed do not share a training identity, which the
    structural digests alone cannot show. The contract version is part of the preimage, so a
    version bump also changes it.
    """
    import hashlib
    import json
    text = json.dumps({"contract_version": CONTRACT_VERSION,
                       "pair_pool_sha256": str(pair_pool_sha256),
                       "pair_targets_sha256": str(pair_targets_sha256),
                       "draw_schedule_sha256": str(draw_schedule_sha256),
                       "target_schedule_sha256": str(target_schedule_sha256)},
                      sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


#: What the completed campaign measured, carried here so T4's design cannot forget it.
PRIOR_EVIDENCE = {
    "display_macro_ap_parent": 0.8539,
    "display_macro_ap_best_ipo": 0.8776,
    "best_ipo_off_library_rate": 0.5290,
    "best_audit_mean_arm": "ipo_forward_kl",
    "best_audit_mean": 0.0753,
    "zero_tail_configuration": "parent mixture at alpha 0.89",
    "spr152_parent_spearman": 0.5634,
    "spr152_best_arm_difference": 0.0021,
    "spr152_best_arm_interval": [-0.0072, 0.0113],
    "spr152_worst_arm_difference": -0.0190,
    "spr152_worst_arm_interval": [-0.0244, -0.0136],
    "reading": ("the best quantitative cell is the best of TEN arm-by-endpoint cells; seven are "
                "negative and the only interval excluding zero is negative. Preference "
                "optimisation has not improved quantitative affinity ordering here, and its "
                "earliest checkpoint measurably degrades it."),
}


#: What :meth:`PairBank.draw` does, stated once so the draw record, the design metadata and the
#: docstring cannot drift apart. It names the marginal that is balanced and the ones that are not.
PAIR_SAMPLING_RULE = (
    "background-first. (1) Backgrounds that still hold an eligible pair are visited round-robin "
    "in an order shuffled once per draw from the seed, one pair per visit, so a draw spreads "
    "over as many distinct backgrounds as it can before returning to any one of them and a "
    "background's chance of being drawn does not grow with how many pairs it holds. (2) Within "
    "the visited background the substitutions that still have an eligible pair are rotated "
    "through in a per-draw shuffled order. (3) Within that (background, substitution) cell the "
    "candidate is the highest-difficulty eligible pair when a difficulty map is supplied, "
    "otherwise a uniformly random one. A pair that has reached the reuse cap leaves every "
    "level. BALANCED: the background marginal (until a background runs out of eligible pairs) "
    "and, inside one background, the substitution marginal. NOT BALANCED: the substitution "
    "marginal across the pool -- a substitution present in many backgrounds is drawn more "
    "often -- and the pair marginal, since a pair in a sparsely populated background is more "
    "likely to be drawn than one in a crowded background. Exposure is recorded per draw and "
    "cumulatively.")


def _exposure(totals, universe):
    """Min/max/mean draws over EVERY member of ``universe`` (never-drawn members count as zero)."""
    members = sorted(set(universe))
    drawn = [int(totals.get(member, 0)) for member in members]
    return {"in_pool": len(members), "touched": sum(1 for value in drawn if value > 0),
            "min_draws": min(drawn, default=0), "max_draws": max(drawn, default=0),
            "mean_draws": (sum(drawn) / len(drawn)) if drawn else 0.0}


@dataclass
class PairBank:
    """Eligible measured pairs, with a reuse counter and a declared cap.

    :meth:`draw` is deterministic **background-first** sampling, documented precisely in
    :data:`PAIR_SAMPLING_RULE`. In short, it balances the *background* marginal and the
    substitution marginal *within* a background; it does **not** make the substitution
    marginal uniform across the pool and it does not make pairs equiprobable. The earlier
    revision chose uniformly over joint ``(substitution, background)`` buckets, which are
    almost all singletons in a real pool, so it was close to a uniform draw over pairs: a
    background holding a hundred pairs was about a hundred times as likely to be drawn as one
    holding one.
    """

    pairs: tuple
    reuse_cap: int = 3
    counts: dict = field(default_factory=dict)

    def available(self):
        return tuple(pair for position, pair in enumerate(self.pairs)
                     if self.counts.get(position, 0) < int(self.reuse_cap))

    def _canonical(self, position):
        """Enumeration-independent order inside a cell: pair digest, then position."""
        return str(self.pairs[position].get("pair_digest") or ""), position

    def draw(self, count, *, seed, difficulty=None):
        """Draw up to ``count`` pairs, background first, never exceeding the reuse cap.

        The ordering rule is the plan's -- structure before loss -- made explicit: sample over
        backgrounds, then substitutions within the background, *before* preferring high-loss
        examples (selecting purely by loss concentrates the refresh on whatever the model is
        currently worst at, which is usually one background). See :data:`PAIR_SAMPLING_RULE`
        for which marginal is balanced and which is not. Every random choice comes from
        ``seed`` and cells are ordered by pair digest, so the same pool, seed and caps give
        the same draw however the pool was enumerated -- the property that lets every
        objective be trained on identical pairs.

        Raises when no pair is eligible. Returns ``(pairs, record)``; ``record`` carries the
        draw's own exposure (backgrounds available and drawn, the largest number of pairs
        taken from one background, substitution cells drawn) and the digests of what it drew.
        """
        rng = np.random.default_rng(int(seed))
        eligible = [position for position in range(len(self.pairs))
                    if self.counts.get(position, 0) < int(self.reuse_cap)]
        require(eligible,
                f"Every measured pair has reached the reuse cap of {self.reuse_cap}. Refresh draws "
                "from eligible MEASURED pairs only; generating more candidates does not create "
                "more measurements.")
        cells = {}
        for position in sorted(eligible, key=self._canonical):
            pair = self.pairs[position]
            cells.setdefault(str(pair.get("background")), {}).setdefault(
                str(pair.get("substitution")), []).append(position)
        order = sorted(cells)
        rng.shuffle(order)
        rotation = {}
        for background in sorted(cells):
            substitutions = sorted(cells[background])
            rng.shuffle(substitutions)
            rotation[background] = deque(substitutions)
        wanted = max(0, int(count))
        chosen, active = [], list(order)
        while len(chosen) < wanted and active:
            for background in list(active):
                if len(chosen) >= wanted:
                    break
                queue = rotation[background]
                substitution = queue.popleft()
                group = cells[background][substitution]
                if difficulty is not None:
                    pick = max(group, key=lambda position: float(difficulty.get(position, 0.0)))
                else:
                    pick = group[int(rng.integers(len(group)))]
                group.remove(pick)
                chosen.append(pick)
                if group:
                    queue.append(substitution)
                if not queue:
                    active.remove(background)
        for position in chosen:
            self.counts[position] = self.counts.get(position, 0) + 1
        drawn = tuple(self.pairs[position] for position in chosen)
        per_background, drawn_cells = {}, set()
        for pair in drawn:
            key = str(pair.get("background"))
            per_background[key] = per_background.get(key, 0) + 1
            drawn_cells.add((key, str(pair.get("substitution"))))
        return drawn, {
            "requested": int(count), "drawn": len(chosen), "reuse_cap": int(self.reuse_cap),
            "exhausted": len(chosen) < wanted,
            "backgrounds_available": len(cells),
            "substitution_cells_available": sum(len(cell) for cell in cells.values()),
            "backgrounds_drawn": len(per_background),
            "max_pairs_from_one_background": max(per_background.values(), default=0),
            "substitution_cells_drawn": len(drawn_cells),
            "pair_digests": [str(pair.get("pair_digest")) for pair in drawn],
            "target_digests": [str(pair.get("target_digest")) for pair in drawn],
            "rule": "background-first; see PAIR_SAMPLING_RULE"}

    def exposure(self):
        """Cumulative draws per background and per substitution label, over the whole pool.

        Never-drawn backgrounds and substitutions count as zero, so ``min_draws`` is the honest
        floor. The substitution figure is by LABEL across backgrounds, which is exactly the
        marginal the sampler does not balance.
        """
        by_background, by_substitution = {}, {}
        for position, used in self.counts.items():
            pair = self.pairs[position]
            background, substitution = str(pair.get("background")), str(pair.get("substitution"))
            by_background[background] = by_background.get(background, 0) + int(used)
            by_substitution[substitution] = by_substitution.get(substitution, 0) + int(used)
        return {"backgrounds": _exposure(
                    by_background, (str(pair.get("background")) for pair in self.pairs)),
                "substitutions": _exposure(
                    by_substitution, (str(pair.get("substitution")) for pair in self.pairs)),
                "balanced_marginal": "background (and substitution within a background)",
                "not_balanced": "the substitution marginal across the pool; the pair marginal"}

    def document(self):
        used = sum(self.counts.values())
        return {"pairs": len(self.pairs), "reuse_cap": int(self.reuse_cap),
                "draws_taken": used, "pairs_at_cap": sum(
                    1 for value in self.counts.values() if value >= int(self.reuse_cap)),
                "pairs_available": len(self.available()),
                "sampling": PAIR_SAMPLING_RULE, "exposure": self.exposure()}


def require_measured_pairs(pairs, *, label="refresh pool"):
    """Refuse a pool containing a generated candidate presented as a measurement."""
    offending = [pair.get("id") for pair in pairs
                 if not bool(pair.get("measured")) or pair.get("source") in (None, "", "generated")]
    require(not offending,
            f"{label}: {len(offending)} pair(s) are not measured, e.g. {offending[:5]}. "
            "Model-generated candidates do not acquire measured affinities by being generated; "
            "IRPO motivates winner-likelihood and refresh, not synthetic outcomes.")
    return True


def freeze_reference(model):
    """Snapshot the post-mid-training reference and detach it. Returns the digest."""
    import hashlib
    digest = hashlib.sha256()
    state = {}
    for name, tensor in sorted(model.state_dict().items()):
        detached = tensor.detach().cpu().clone()
        state[name] = detached
        digest.update(name.encode("utf-8"))
        digest.update(detached.cpu().contiguous().numpy().tobytes())
    return {"state": state, "sha256": digest.hexdigest(),
            "frozen_at": "after mid-training, before any preference update",
            "rule": "never refreshed during T4; a moving reference invalidates the arm's name"}


def require_frozen_reference(reference_log_probabilities, *, label="reference"):
    inherited.require_frozen_references(reference_log_probabilities, where=label)
    return True


def parameter_movement(before, after):
    """How far the arm actually moved the policy. The evidence that it ran.

    A preference stage that drew pairs and returned its design metadata produces a
    record indistinguishable from one that optimised, unless the parameters are
    compared. ``max_abs_change`` of zero means no update reached the weights,
    whatever the loss history says.
    """
    worst, moved, compared = 0.0, 0, 0
    for name, tensor in dict(after).items():
        reference = dict(before).get(name)
        if reference is None or not tensor.dtype.is_floating_point:
            continue
        compared += 1
        difference = float((tensor.detach().cpu() - reference.detach().cpu()).abs().max())
        if difference > 0:
            moved += 1
        worst = max(worst, difference)
    return {"tensors_compared": compared, "tensors_moved": moved,
            "max_abs_change": worst, "updated": bool(worst > 0.0),
            "asserts": ("that the optimisation reached the weights. A loss history alone does not "
                        "distinguish an arm that optimised from one that drew pairs and returned "
                        "its design.")}


def conditional_fit_loss(policy_log_probabilities, targets, mask, *, values, kappa,
                         mode="absolute", reference=None):
    """DEPRECATED (v1 arm 1). Retained for historical callers; no v2 stage selects it.

    ``absolute`` mode computes ``mean((log q(chosen) - log q(rejected) - kappa * delta)^2)``
    over the rows' summed masked log probabilities, which is exactly
    :func:`margin_match_loss`. It is therefore **not** a conditional likelihood and was never
    distinct from ``numeric_margin_match``; the pairing in the v1 comparison was one
    objective run twice. Use :func:`weighted_conditional_nll_loss` for a likelihood and
    :func:`margin_match_loss` for log-odds matching. ``resolve_methods`` refuses the
    ``conditional_fit`` name.
    """
    require(mode in ("absolute", "reference_relative"), f"Unknown fitting mode {mode!r}")
    picked = policy_log_probabilities.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    active = mask.to(picked.dtype)
    require(float(active.sum()) > 0, "A conditional fit with no active target")
    left = (picked * active).sum(dim=1)
    values = torch.as_tensor(values, dtype=left.dtype, device=left.device)
    require(values.shape == left.shape,
            f"{tuple(values.shape)} measured values against {tuple(left.shape)} rows")
    # The "alternative" and "reference" halves of each pair are adjacent rows, which
    # is how the collator builds a contrast batch: even rows are the alternative.
    require(left.shape[0] % 2 == 0,
            "A contrast batch must hold an even number of rows: alternative, then reference")
    alternative, baseline = left[0::2], left[1::2]
    delta = values[0::2] - values[1::2]
    if mode == "absolute":
        return fit_absolute_log_odds(alternative, baseline, delta, kappa=kappa)
    require(reference is not None, "Reference-relative fitting needs the frozen reference scores")
    reference = torch.as_tensor(reference, dtype=left.dtype, device=left.device)
    return fit_reference_relative_log_odds(alternative, baseline, reference[0::2],
                                          reference[1::2], delta, kappa=kappa)


def preference_loss(method, *, policy_chosen, policy_rejected, reference_chosen,
                    reference_rejected, coefficients, winner_likelihood=0.0):
    """Arms 3 and 4, through the inherited tested implementations.

    ``winner_likelihood`` adds ``-mean(log q(chosen)) / window`` -- IRPO's motivation
    -- as a *separate, named* term so its contribution is attributable rather than
    folded into the preference loss.
    """
    require(method in (METHOD_IPO, METHOD_DPO), f"{method!r} is not a preference method")
    require_frozen_reference(reference_chosen, label=f"{method} chosen reference")
    require_frozen_reference(reference_rejected, label=f"{method} rejected reference")
    values = {str(key): float(value) for key, value in dict(coefficients).items()}
    if method == METHOD_IPO:
        loss, diagnostics = inherited.batch_loss(
            "ipo", policy_chosen=policy_chosen, policy_rejected=policy_rejected,
            reference_chosen=reference_chosen, reference_rejected=reference_rejected,
            coefficients={"tau": values.get("tau", IPO_TAU)})
    else:
        loss, diagnostics = inherited.batch_loss(
            "dpo", policy_chosen=policy_chosen, policy_rejected=policy_rejected,
            reference_chosen=reference_chosen, reference_rejected=reference_rejected,
            coefficients={"beta": values["beta"]})
    record = {key: (float(value.detach().mean()) if isinstance(value, torch.Tensor) else value)
              for key, value in dict(diagnostics).items()}
    record["preference_loss"] = float(loss.detach())
    if float(winner_likelihood) > 0.0:
        length = int(values.get("window", 1))
        term = -policy_chosen.mean() / max(1, length)
        loss = loss + float(winner_likelihood) * term
        record["winner_likelihood_term"] = float(term.detach())
        record["winner_likelihood_coefficient"] = float(winner_likelihood)
    record["method"] = method
    return loss, record


def require_one_site_rows(mask, *, label="pair batch"):
    """Every row must score exactly ONE masked coordinate. Returns the row count.

    The pair objectives read ``log q(alternative | masked context)``. With two or more
    active coordinates the per-row sum is a chain (autoregressive route) or a product of
    simultaneous marginals (masked route), neither of which is the one-site conditional the
    objectives and their names describe.
    """
    active = torch.as_tensor(mask).to(torch.long).sum(dim=1)
    require(bool((active == 1).all()),
            f"{label}: every row must score exactly one masked coordinate, got per-row counts "
            f"{sorted(set(int(value) for value in active.tolist()))}. A multi-site window sum "
            "is not a one-site conditional.")
    return int(active.shape[0])


def alternative_mass(log_chosen, log_rejected):
    """``q(chosen) + q(rejected)`` per pair: the model's mass on the two measured alternatives.

    Computed from the full-vocabulary log probabilities, so it is the share of the whole
    distribution the pair holds. The log-odds objectives cannot see it.
    """
    return torch.exp(log_chosen) + torch.exp(log_rejected)


def margin_match_loss(policy_chosen, policy_rejected, measured_margin, *, kappa):
    """Arm 2: one-site numeric log-odds matching.

    ``policy_chosen`` / ``policy_rejected`` are ``log q`` of the two alternatives at the
    pair's single masked coordinate (:func:`require_one_site_rows`), so their difference is
    the conditional log odds of the edit and the loss is
    ``mean((log odds - kappa * measured_margin)^2)``. It is invariant to the alternatives'
    total mass: the odds can be exactly right while the pair holds almost none of the
    distribution, and the gradient is then zero. That is what separates it from
    :func:`weighted_conditional_nll_loss`.
    """
    margin = policy_chosen - policy_rejected
    target = torch.as_tensor(measured_margin, dtype=margin.dtype, device=margin.device) * float(kappa)
    require(margin.shape == target.shape,
            f"{tuple(margin.shape)} log odds against {tuple(target.shape)} measured margins")
    require(bool(torch.isfinite(margin).all()) and bool(torch.isfinite(target).all()),
            "numeric_margin_match: a non-finite log odds or measured margin reached the loss")
    residual = margin - target
    loss = (residual ** 2).mean()
    with torch.no_grad():
        record = {"pairs": int(margin.shape[0]), "kappa": float(kappa),
                  "mean_margin": float(margin.detach().mean()),
                  "mean_target": float(target.detach().mean()),
                  "mean_abs_residual": float(residual.detach().abs().mean()),
                  "mean_alternative_mass": float(alternative_mass(
                      policy_chosen.detach(), policy_rejected.detach()).mean()),
                  "method": METHOD_MARGIN_MATCH,
                  "note": ("ONE-SITE numeric log-odds matching: log q(chosen) - log q(rejected) "
                           "to kappa * delta at the pair's single masked coordinate. It is not a "
                           "sequence-level score and it is blind to the alternatives' total "
                           "mass; weighted_conditional_nll is the likelihood counterpart.")}
    return loss, record


def weighted_conditional_nll_loss(policy_log_probabilities, targets, mask, *, deltas, kappa,
                                  weights=None, assays=None, tolerance=1e-4):
    """Arm 1 (v2): full-vocabulary weighted categorical NLL against pair-local soft targets.

    ``policy_log_probabilities`` is ``(2P, Q, V)``: adjacent rows are the chosen and the
    rejected alternative of one pair, each scoring exactly one masked coordinate over the
    *whole* vocabulary ``V``. For pair ``i`` with measured difference ``delta_i`` the
    soft targets are ``t_chosen = sigmoid(kappa * delta_i)`` and ``t_rejected = 1 -
    t_chosen``, and the loss is ::

        -( t_chosen * log q(chosen) + t_rejected * log q(rejected) )

    averaged over pairs (or weighted by ``weights``). ``log q`` is read straight from the
    model's distribution over all ``V`` residues -- it is **not** renormalised over the two
    alternatives. Renormalised, it would be a pair-normalised logistic cross-entropy on the log
    odds ``o = log q(chosen) - log q(rejected)``, ``-(t log sigma(o) + (1 - t) log sigma(-o))``.
    That loss shares its OPTIMUM with :func:`margin_match_loss` (``o = kappa * delta`` gives
    ``sigma(o) = t``) but it is not the same objective: the gradients differ -- ``sigma(o) - t``
    with respect to ``o``, bounded, against the squared error's ``2 (o - kappa * delta)`` --
    so the two do not move a model the same way on the way to that shared optimum. The loss
    here differs from both: because the mass outside the pair is penalised, a model that holds
    the right odds on a pair but only a fraction of the distribution has a non-zero gradient
    here and a zero gradient under either log-odds loss.

    This is **not** an exact tilted-prior fit. The target distribution lives on the two
    measured alternatives alone, so its minimiser sets every unmeasured residue's
    probability to zero at that coordinate; an exact tilt ``q* ~ p * exp(u / tau)`` would keep
    the prior's mass on unmeasured residues. The loss is a *pair-local* soft-label
    likelihood and the record says so.

    Inputs are validated rather than coerced: shapes must align, ``kappa`` and ``deltas``
    must be finite, every row must score one coordinate, the two halves of a pair must name
    different residues, and every active conditional must sum to one over ``V`` to within
    ``tolerance`` (a renormalised or truncated vocabulary is refused).
    """
    require(isinstance(policy_log_probabilities, torch.Tensor)
            and policy_log_probabilities.dim() == 3,
            "weighted_conditional_nll: expected (rows, queries, vocabulary) log probabilities")
    rows, queries, vocabulary = policy_log_probabilities.shape
    require(rows >= 2 and rows % 2 == 0,
            f"A pair batch must hold an even number of rows (chosen, then rejected); got {rows}")
    pair_count = rows // 2
    kappa = float(kappa)
    require(math.isfinite(kappa) and kappa > 0.0, f"kappa must be finite and positive, got {kappa}")
    device = policy_log_probabilities.device
    targets = torch.as_tensor(targets, dtype=torch.long, device=device)
    mask = torch.as_tensor(mask, device=device).to(torch.bool)
    require(targets.shape == mask.shape == (rows, queries),
            f"targets {tuple(targets.shape)}, mask {tuple(mask.shape)} and log probabilities "
            f"{(rows, queries)} are not aligned")
    require_one_site_rows(mask, label="weighted_conditional_nll")
    require(bool((targets[mask] >= 0).all()) and bool((targets[mask] < vocabulary).all()),
            f"a target residue index lies outside the {vocabulary}-residue vocabulary")
    require(mask[0::2].equal(mask[1::2]),
            "the two halves of a pair must score the same query coordinate")
    deltas = torch.as_tensor(deltas, dtype=policy_log_probabilities.dtype, device=device)
    require(deltas.shape == (pair_count,),
            f"{tuple(deltas.shape)} measured differences against {pair_count} pairs")
    require(bool(torch.isfinite(deltas).all()), "a measured difference is not finite")
    chosen_target, rejected_target = targets[0::2][mask[0::2]], targets[1::2][mask[1::2]]
    require(bool((chosen_target != rejected_target).all()),
            "a pair names the same residue on both halves; there is no alternative to weigh")

    with torch.no_grad():
        active = policy_log_probabilities.detach()[mask].double()
        require(bool(torch.isfinite(torch.logsumexp(active, dim=-1)).all()),
                "an active conditional has no finite probability mass")
        worst = float((torch.logsumexp(active, dim=-1)).abs().max())
    require(worst <= float(tolerance),
            f"weighted_conditional_nll: an active conditional's log-sum-exp is {worst:.3g} from "
            f"zero, above {float(tolerance):.1e}. The loss reads log q over the FULL vocabulary; "
            "a renormalised or truncated distribution would change what it measures.")

    picked = policy_log_probabilities.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    log_q = torch.where(mask, picked, torch.zeros_like(picked)).sum(dim=1)
    require(bool(torch.isfinite(log_q).all()),
            "a measured alternative has zero modelled probability (log q is not finite)")
    log_chosen, log_rejected = log_q[0::2], log_q[1::2]
    scaled = kappa * deltas.detach()
    target_chosen, target_rejected = torch.sigmoid(scaled), torch.sigmoid(-scaled)
    per_pair = -(target_chosen * log_chosen + target_rejected * log_rejected)
    if weights is None:
        loss = per_pair.mean()
        weight_record = "uniform"
    else:
        weight = torch.as_tensor(weights, dtype=per_pair.dtype, device=device)
        require(weight.shape == (pair_count,) and bool(torch.isfinite(weight).all())
                and bool((weight >= 0).all()) and float(weight.sum()) > 0.0,
                "pair weights must be finite, non-negative, aligned and sum to a positive number")
        loss = (per_pair * weight).sum() / weight.sum()
        weight_record = "supplied"
    with torch.no_grad():
        mass = alternative_mass(log_chosen.detach(), log_rejected.detach())
        odds = (log_chosen - log_rejected).detach()
        residual = odds - scaled
        entropy = -(torch.special.xlogy(target_chosen, target_chosen)
                    + torch.special.xlogy(target_rejected, target_rejected))
        record = {"pairs": int(pair_count), "kappa": kappa, "method": METHOD_WEIGHTED_CONDITIONAL_NLL,
                  "weights": weight_record, "vocabulary": int(vocabulary),
                  "mean_target_chosen": float(target_chosen.mean()),
                  "mean_target_entropy": float(entropy.mean()),
                  "mean_alternative_mass": float(mass.mean()),
                  "min_alternative_mass": float(mass.min()),
                  "max_alternative_mass": float(mass.max()),
                  "mean_log_odds": float(odds.mean()),
                  "mean_target_log_odds": float(scaled.mean()),
                  "mean_abs_log_odds_residual": float(residual.abs().mean()),
                  "max_normalisation_error": worst,
                  "note": ("pair-local soft-label likelihood over the full vocabulary; NOT an "
                           "exact tilted-prior fit. Its minimiser puts all mass on the two "
                           "measured alternatives.")}
        if assays is not None:
            labels = [str(name) for name in assays]
            require(len(labels) == pair_count,
                    f"{len(labels)} assay labels for {pair_count} pairs")
            by_assay = {}
            for index, name in enumerate(labels):
                block = by_assay.setdefault(name, {"pairs": 0, "nll": 0.0, "mass": 0.0})
                block["pairs"] += 1
                block["nll"] += float(per_pair[index].detach())
                block["mass"] += float(mass[index])
            record["by_assay"] = {
                name: {"pairs": block["pairs"], "mean_nll": block["nll"] / block["pairs"],
                       "mean_alternative_mass": block["mass"] / block["pairs"]}
                for name, block in sorted(by_assay.items())}
    return loss, record


class AssayExposure:
    """Per-assay pair counts and mean alternative mass across one arm's updates.

    Pooled across assays, ``kappa * delta`` treats one pKD unit as the same log odds in
    every assay. This is the breakdown a reader needs to see whether one assay dominates the
    gradient, and it is identical in shape for every arm so the arms can be compared.
    """

    def __init__(self):
        self.pairs, self.mass_sum, self.mass_count = {}, {}, {}

    def add(self, assays, *, masses=None):
        labels = [str(name) for name in assays]
        values = None if masses is None else [float(value) for value in masses]
        require(values is None or len(values) == len(labels),
                f"{0 if values is None else len(values)} masses for {len(labels)} pairs")
        for index, name in enumerate(labels):
            self.pairs[name] = self.pairs.get(name, 0) + 1
            if values is not None:
                self.mass_sum[name] = self.mass_sum.get(name, 0.0) + values[index]
                self.mass_count[name] = self.mass_count.get(name, 0) + 1

    def document(self):
        return {name: {"pairs_drawn": int(count),
                       "mean_alternative_mass": (self.mass_sum[name] / self.mass_count[name]
                                                 if self.mass_count.get(name) else None)}
                for name, count in sorted(self.pairs.items())}


def design(*, methods=METHODS, reuse_cap=3, winner_likelihood=0.1):
    """The declared T4 comparison, with the prior evidence attached.

    ``winner_likelihood`` is the coefficient the caller actually runs with, and the stage must
    pass its configured value: the default here is only for a design read without a config, and
    metadata that quotes it beside an executed value that differs is wrong. The coefficient
    only enters the preference arms, so ``winner_likelihood_applied_to`` names those among
    ``methods`` and a likelihood or margin arm's record does not suggest it was used.
    """
    methods = resolve_methods(methods)
    return {"contract_version": CONTRACT_VERSION,
            "methods": list(methods), "reuse_cap": int(reuse_cap),
            "objectives": {name: OBJECTIVES[name] for name in methods},
            "winner_likelihood_coefficient": float(winner_likelihood),
            "winner_likelihood_applied_to": [name for name in methods
                                             if name in (METHOD_IPO, METHOD_DPO)],
            "reference": "frozen after mid-training, never refreshed",
            "scoring": ("policy and reference are scored deterministically (eval mode); the policy "
                        "forward keeps gradients, the reference runs frozen under no_grad"),
            "prior_evidence": dict(PRIOR_EVIDENCE),
            "design_intent": ("conditional supervised fitting on measured values is the FIRST arm "
                              "because the measured failure is quantitative. The preference arms "
                              "are compared against it, not against nothing."),
            "refresh_rule": ("difficult comparisons are drawn from eligible measured training "
                             "pairs, background first, then a substitution within the background, "
                             "before high-loss examples, capped at the declared reuse count"),
            "pair_sampling": PAIR_SAMPLING_RULE,
            "forbidden": ("a generated candidate entering the pair pool as a measurement")}
