"""Token-level parent replay: the distillation term, its reductions, and one update.

Phase D of :doc:`specs/her2_support_preservation_plan.md`. Everything here is
tensor-level and model-free on purpose, so the mathematics can be checked on a
tiny enumerable distribution in float64 on CPU without a GPU, a checkpoint or a
bank. The model seam lives in :mod:`her2_replay_campaign`.

Four properties are enforced here rather than left to convention:

* **The replay term is a sum over ten positions and a mean over rows.** No
  division by 10 and none by 20. ``K(y) = sum_t sum_a p0(a | scaffold, y_<t) *
  (log p0 - log ptheta)``; by the autoregressive chain rule its expectation under
  parent draws is ``KL(parent || policy)``. The per-row values of ``K(y)`` are a
  *different random variable* from the sequence drop with the same mean, so their
  quantiles are never reported as drop quantiles.
* **A zero teacher probability contributes exactly zero.** The difference is
  masked *before* the multiplication, so a ``-inf`` teacher log at an underflowed
  residue cannot become ``0 * -inf = NaN`` on the forward pass or a ``0 * NaN``
  on the backward one. No epsilon is added and no probability mass is invented.
* **Accumulation is row-weighted.** Each component reduces by a mean over its own
  rows, so a microbatch contributes ``n_i / N`` of that mean -- not ``1 / k``. The
  two agree only when every microbatch is full, which is exactly the case a
  partial-tail test does not exercise, so the weighting is the mechanism and the
  equal-split shortcut does not appear anywhere.
* **One combined gradient, one clip, one step.** :func:`combined_loss` is the only
  place ``lambda`` multiplies anything, and ``lambda = 0`` returns the task term
  itself -- not ``task + 0 * replay`` -- so a no-replay control performs no replay
  forward pass, allocates no replay graph and moves no replay RNG.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .her2_data import CANONICAL, CORE_LENGTH
from .her2_objectives import CORE_POSITIONS, batch_loss, require_frozen_references
from .her2_runtime import require

REPLAY_SCHEMA = "her2-parent-replay/1"

#: The two task objectives this screen fits. Both are inherited verbatim from
#: :mod:`her2_objectives`; this module re-implements neither.
REPLAY_TASKS = ("continued_sft", "ipo")
#: The IPO temperature the audit escalated on. Declared here so a config that
#: names another value is a different experiment rather than a silent variant.
IPO_TAU = 0.1

#: The teacher target block's declared shape: rows x ten positions x twenty
#: canonical residues. The full model vocabulary is deliberately absent -- the
#: distillation target is the same 20-way renormalized categorical the task loss,
#: the scorer and the sampler use.
TEACHER_WIDTH = len(CANONICAL)

#: How far a cached teacher probability row may sit from summing to one. Two
#: float32 ``exp(log_softmax)`` rows over twenty entries accumulate rounding of
#: order ``20 * 2**-24``; 1e-5 clears that by two orders of magnitude and still
#: rejects a truncated, re-normalized or mis-ordered cache.
PROBABILITY_SUM_ATOL = 1e-5
#: Agreement required between the cached probabilities and ``exp`` of the cached
#: logs. They are two stored views of one float32 computation, so this is a
#: storage check, not a model check.
PROBABILITY_EXP_ATOL = 1e-6


# ---------------------------------------------------------------------------
# teacher targets
# ---------------------------------------------------------------------------

def require_teacher_targets(probabilities, log_probabilities, *, rows, label, chunk=4096):
    """Shape, finiteness, normalization and probability/log agreement, in one place.

    Called at cache build, at cache load and inside the replay term. It is cheap
    relative to a forward pass and it is the only thing standing between a
    silently transposed ``(rows, 20, 10)`` block and a loss that still produces a
    number.

    The float64 reductions run in chunks: promoting a whole 100,000-row cache to
    float64 to check that it sums to one would allocate several hundred megabytes
    to answer a question about every row, and every row is still checked this way.
    """
    for name, tensor in (("probabilities", probabilities), ("log probabilities",
                                                            log_probabilities)):
        require(isinstance(tensor, torch.Tensor), f"{label}: teacher {name} must be a tensor")
        require(tuple(tensor.shape) == (int(rows), CORE_POSITIONS, TEACHER_WIDTH),
                f"{label}: teacher {name} has shape {tuple(tensor.shape)}, expected "
                f"({rows}, {CORE_POSITIONS}, {TEACHER_WIDTH})")
        require(bool(torch.isfinite(tensor).all()),
                f"{label}: teacher {name} carries a nonfinite value. A nonfinite target is not "
                "clipped here; the cache is rebuilt or the artifact fails.")
    require(not probabilities.requires_grad and not log_probabilities.requires_grad,
            f"{label}: teacher targets carry requires_grad. The teacher is frozen and detached; a "
            "live one turns distillation into a joint optimization of both sides.")
    require(bool((probabilities >= 0).all()),
            f"{label}: a teacher probability is negative")
    worst, agreement = 0.0, 0.0
    for start in range(0, int(rows), max(1, int(chunk))):
        block = probabilities[start:start + max(1, int(chunk))].double()
        logs = log_probabilities[start:start + max(1, int(chunk))].double()
        worst = max(worst, float((block.sum(dim=-1) - 1.0).abs().max()))
        agreement = max(agreement, float((block - logs.exp()).abs().max()))
    require(worst <= PROBABILITY_SUM_ATOL,
            f"{label}: the worst cached teacher row deviates from summing to one by {worst:.3g}, "
            f"above the declared {PROBABILITY_SUM_ATOL:.1e}. A cache that does not normalize is a "
            "different distribution, not a rounding difference.")
    require(agreement <= PROBABILITY_EXP_ATOL,
            f"{label}: cached probabilities and exp(cached logs) disagree by {agreement:.3g}, "
            f"above {PROBABILITY_EXP_ATOL:.1e}. The two are stored views of one computation.")
    return {"rows": int(rows), "positions": CORE_POSITIONS, "residues": TEACHER_WIDTH,
            "max_abs_probability_sum_error": worst,
            "max_abs_probability_exp_error": agreement,
            "probability_sum_atol": PROBABILITY_SUM_ATOL,
            "probability_exp_atol": PROBABILITY_EXP_ATOL,
            "zero_convention": "p == 0 contributes exactly 0; no epsilon is added"}


# ---------------------------------------------------------------------------
# the distillation term
# ---------------------------------------------------------------------------

def conditional_kl_terms(teacher_probabilities, teacher_log_probabilities,
                         student_log_probabilities):
    """``(B, 10)`` exact categorical KL at every visited prefix.

    Exact *per position*: at a visited prefix the sum runs over all twenty
    canonical residues, so nothing is sampled inside a position. What is sampled
    is the set of prefixes, which is why the sequence-level quantity below is an
    estimator of ``KL(parent || policy)`` and not that KL itself.
    """
    probabilities = teacher_probabilities
    require(isinstance(student_log_probabilities, torch.Tensor),
            "Student log probabilities must be a tensor")
    require(student_log_probabilities.shape == probabilities.shape,
            f"Student {tuple(student_log_probabilities.shape)} and teacher "
            f"{tuple(probabilities.shape)} target blocks are not aligned")
    teacher_log = teacher_log_probabilities.to(student_log_probabilities.dtype)
    probabilities = probabilities.to(student_log_probabilities.dtype)
    # Masked BEFORE the multiplication. A teacher log of -inf at an underflowed
    # residue would otherwise produce 0 * -inf = NaN on the forward pass, and
    # masking afterwards leaves 0 * NaN on the backward one.
    difference = torch.where(probabilities > 0, teacher_log - student_log_probabilities,
                             torch.zeros_like(teacher_log))
    return (probabilities * difference).sum(dim=-1)


def sequence_conditional_kl(teacher_probabilities, teacher_log_probabilities,
                            student_log_probabilities):
    """``(B,)`` ``K(y)``: the sum of the ten per-position categorical KLs."""
    return conditional_kl_terms(teacher_probabilities, teacher_log_probabilities,
                                student_log_probabilities).sum(dim=-1)


def replay_term(teacher_probabilities, teacher_log_probabilities, student_log_probabilities, *,
                label="replay"):
    """``(mean K(y) over rows, detached diagnostics)`` -- the loss the arms add.

    The reduction is a mean over replay rows of a quantity that is already a sum
    over ten positions. Nothing here divides by ten or by twenty, and the
    per-position vector travels in the diagnostics so a report can show where the
    mass moved without the loss having been reshaped to make that easy.
    """
    require_teacher_targets(teacher_probabilities, teacher_log_probabilities,
                            rows=int(teacher_probabilities.shape[0]), label=label)
    per_position = conditional_kl_terms(teacher_probabilities, teacher_log_probabilities,
                                        student_log_probabilities)
    per_row = per_position.sum(dim=-1)
    mean = per_row.mean()
    with torch.no_grad():
        detached = per_row.detach()
        diagnostics = {
            "rows": int(per_row.shape[0]),
            "conditional_kl_per_row": detached,
            "conditional_kl_per_position": per_position.detach().mean(dim=0),
            "mean_conditional_kl": float(detached.double().mean()),
            "max_conditional_kl": float(detached.double().max()),
            "min_conditional_kl": float(detached.double().min()),
            "units": "nats per ten-residue sequence, summed over positions and residues",
            "estimand_note": ("the mean of K(y) over parent draws estimates KL(parent || policy). "
                              "K(y) is not the sequence drop: they share a mean and have "
                              "different distributions, so their tails are never merged.")}
    return mean, diagnostics


def task_term(task, *, policy_chosen, policy_rejected=None, reference_chosen=None,
              reference_rejected=None, tau=IPO_TAU):
    """The inherited task loss, unchanged, dispatched by name.

    ``continued_sft`` is ``mean(-chosen_sum_log_probability / 10)`` and consumes no
    rejected sequence; ``ipo`` is ``mean((Delta - 1 / (2 tau))**2)`` against the
    frozen parent reference. Both come out of :mod:`her2_objectives`; this function
    exists so the campaign has one call site and cannot accidentally hand an IPO
    reference to the SFT arm.
    """
    require(task in REPLAY_TASKS, f"Unknown replay task {task!r}; expected one of {REPLAY_TASKS}")
    if task == "continued_sft":
        require(policy_rejected is None and reference_chosen is None
                and reference_rejected is None,
                "continued_sft consumes chosen sequences only; supplying a rejected or reference "
                "tensor would mean an inference pass was paid for that this arm cannot use")
        return batch_loss("continued_sft", policy_chosen=policy_chosen)
    require(policy_rejected is not None and reference_chosen is not None
            and reference_rejected is not None,
            "ipo needs the policy rejected scores and both frozen parent reference vectors")
    require_frozen_references(reference_chosen, reference_rejected, where="ipo replay batch")
    return batch_loss("ipo", policy_chosen=policy_chosen, policy_rejected=policy_rejected,
                      reference_chosen=reference_chosen, reference_rejected=reference_rejected,
                      coefficients={"tau": float(tau)})


def combined_loss(task_mean, replay_mean, *, replay_coefficient):
    """``L_task + lambda * L_replay``, with ``lambda = 0`` returning the task term itself.

    The distinction matters: a control that evaluated ``task + 0 * replay`` would
    still have run the replay forward pass, built its graph and consumed its
    stream. Returning ``task_mean`` unchanged makes the no-replay arm's update
    bit-identical to one produced by code that has no replay term at all, which is
    what "unchanged lambda-zero behaviour" has to mean to be checkable.
    """
    lam = float(replay_coefficient)
    require(math.isfinite(lam) and lam >= 0, f"lambda must be finite and >= 0, got {lam!r}")
    if lam == 0.0:
        require(replay_mean is None,
                "lambda is zero but a replay term was computed anyway. The no-replay control does "
                "no replay work: no forward pass, no cache read and no stream movement.")
        return task_mean
    require(replay_mean is not None, f"lambda is {lam} but no replay term was supplied")
    return task_mean + lam * replay_mean


# ---------------------------------------------------------------------------
# row-weighted accumulation: one gradient, one clip, one step
# ---------------------------------------------------------------------------

def _scalar(value):
    """A python float from a 0-d tensor or a plain number, without touching the graph."""
    if isinstance(value, torch.Tensor):
        return float(value.detach().double())
    return float(value)


class MicrobatchAccumulator:
    """Accumulate microbatch gradients into exactly one full-batch gradient.

    Each component declares its **total** row count for the update in advance, and
    every microbatch contributes ``n_i / N_i`` of its own mean. Summed over
    microbatches that is ``sum_i n_i * mean_i / N_i`` -- the full-batch mean -- for
    any partition, including one whose last chunk is short. Dividing by the number
    of microbatches instead would be correct only for an exact partition, which is
    precisely the case a partial-tail test does not cover.

    ``backward`` is called once per microbatch so the graph is freed as it goes;
    the gradients add up in ``.grad``. The clip and the optimizer step belong to
    the caller and happen once, after :meth:`finish`.
    """

    def __init__(self, totals, *, coefficients=None, label="update"):
        totals = {str(name): int(count) for name, count in dict(totals).items()}
        require(totals, f"{label}: an update declares at least one component")
        for name, count in sorted(totals.items()):
            require(count > 0, f"{label}: component {name!r} declares {count} rows")
        self.totals = totals
        self.coefficients = {str(name): float(value)
                             for name, value in dict(coefficients or {}).items()}
        self.label = str(label)
        self.seen = {name: 0 for name in totals}
        self.weighted = {name: 0.0 for name in totals}
        self.microbatches = 0

    def weight(self, name, count):
        require(name in self.totals, f"{self.label}: component {name!r} was never declared")
        return float(count) / float(self.totals[name])

    def add(self, counts, means, *, backward=None):
        """Scale one microbatch's component means, accumulate and optionally back-propagate."""
        counts = {str(name): int(value) for name, value in dict(counts).items()}
        means = {str(name): value for name, value in dict(means).items()}
        require(sorted(counts) == sorted(means),
                f"{self.label}: microbatch counts {sorted(counts)} and means {sorted(means)} "
                "describe different components")
        # Declared *before* anything is indexed. ``self.seen[name] += ...`` on an
        # undeclared component raises KeyError, which says nothing about what went
        # wrong; the named refusal is the contract this class exists to state.
        undeclared = sorted(name for name in counts if name not in self.totals)
        require(not undeclared,
                f"{self.label}: component {undeclared} was never declared for this update. An "
                f"update declares {sorted(self.totals)} in advance; a component that appears only "
                "at accumulation time has no declared row total to weight it by, so its "
                "contribution to the full-batch mean is undefined.")
        total = None
        for name in sorted(counts):
            require(counts[name] > 0, f"{self.label}: empty microbatch for {name!r}")
            self.seen[name] += counts[name]
            require(self.seen[name] <= self.totals[name],
                    f"{self.label}: component {name!r} has now seen {self.seen[name]} rows, more "
                    f"than the {self.totals[name]} declared for this update")
            scale = self.weight(name, counts[name]) * self.coefficients.get(name, 1.0)
            self.weighted[name] += _scalar(means[name]) * self.weight(name, counts[name])
            contribution = scale * means[name]
            total = contribution if total is None else total + contribution
        self.microbatches += 1
        if backward is not None and total is not None:
            backward(total)
        return total

    def finish(self):
        """Component means for the whole update, plus the exposure proof they rest on."""
        short = sorted(f"{name}: {self.seen[name]} of {self.totals[name]}"
                       for name, count in self.seen.items() if count != self.totals[name])
        require(not short,
                f"{self.label}: these components did not consume the rows they declared: {short}. "
                "An update that consumed a different number of rows than it declared is not the "
                "declared update, and its exposure count would be a fiction.")
        weighted_total = sum(self.weighted[name] * self.coefficients.get(name, 1.0)
                             for name in self.totals)
        return {"microbatches": self.microbatches,
                "rows": dict(self.seen),
                "declared_rows": dict(self.totals),
                "component_means": {name: self.weighted[name] for name in sorted(self.weighted)},
                "coefficients": dict(self.coefficients),
                "weighted_total": weighted_total,
                "reduction": ("each component is a mean over its own rows; a microbatch "
                              "contributes n_i / N_i of that mean, so a short final microbatch "
                              "is weighted by its true row count")}


def full_batch_reference(component_means, *, coefficients=None):
    """The direct full-batch total the accumulator must reproduce. For tests and logs."""
    coefficients = dict(coefficients or {})
    return sum(float(coefficients.get(name, 1.0)) * float(value)
               for name, value in dict(component_means).items())


# ---------------------------------------------------------------------------
# gradient bookkeeping
# ---------------------------------------------------------------------------

def clip_and_step(model, optimizer, scheduler, *, gradient_clip):
    """One clip, one optimizer step, one scheduler step, with the norm recorded.

    ``error_if_nonfinite`` is on: a nonfinite gradient stops the trajectory rather
    than stepping on it and then reporting a plausible loss two updates later.
    ``clipped`` is journalled because the loss scales of the two tasks and of six
    lambda values differ by orders of magnitude, so how often the clip binds is a
    property of the arm and not an incidental detail.

    ``learning_rate_used`` is read **before** ``optimizer.step()``. Reading it after
    ``scheduler.step()`` reports the rate the *next* update will use: under the
    inherited ``step + 1`` warmup that is an off-by-one of a factor of two at the
    first update (actual 1e-7, reported 2e-7), which would silently misattribute
    every logged rate by one position. The next rate is recorded separately and
    labelled as what it is. The schedule itself is untouched.
    """
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip),
                                          error_if_nonfinite=True)
    used = float(scheduler.get_last_lr()[0])
    optimizer.step()
    scheduler.step()
    value = float(norm)
    return {"gradient_norm": value, "gradient_clip": float(gradient_clip),
            "clipped": bool(value > float(gradient_clip)),
            "learning_rate_used": used,
            "learning_rate_next": float(scheduler.get_last_lr()[0]),
            "learning_rate_basis": ("the rate this step applied, captured before optimizer.step; "
                                    "learning_rate_next is what the following update will use")}


# ---------------------------------------------------------------------------
# an independent first-step AdamW oracle
# ---------------------------------------------------------------------------

#: ``torch.optim.AdamW``'s default epsilon. Named here because the oracle below
#: reproduces the first step in closed form and the epsilon is the only term in it
#: that does not come from the config.
ADAMW_EPS = 1e-8


def adamw_first_step(parameters, gradients, *, learning_rate, weight_decay, betas=(0.9, 0.999),
                     eps=ADAMW_EPS):
    """The closed-form first AdamW step, in float64, from one route's own gradients.

    With zero-initialized moments the bias corrections cancel exactly:
    ``m_hat = g``, ``sqrt(v_hat) = |g|``, so

        theta_1 = theta_0 * (1 - lr * wd) - lr * g / (|g| + eps).

    This is an *independent* oracle, not a second call into the optimizer: it is
    written from the update rule rather than measured from the thing it checks.
    Its value is that it is per-route -- each route's parameters are compared to the
    step its **own** gradients imply, so "the accumulated route stepped on the
    gradients it accumulated" is a checkable statement even where the two routes'
    near-zero gradients disagree in sign.
    """
    theta = np.asarray(parameters, dtype=np.float64).reshape(-1)
    grad = np.asarray(gradients, dtype=np.float64).reshape(-1)
    require(theta.shape == grad.shape,
            f"The oracle compared {theta.shape} parameters against {grad.shape} gradients")
    require(all(0.0 <= float(value) < 1.0 for value in betas),
            f"AdamW's betas live in [0, 1) and these are {tuple(float(v) for v in betas)}. At "
            "beta = 1 the bias correction 1 - beta**1 is zero; beta = 0 is a legal degenerate "
            "case and the closed form still holds there.")
    decayed = theta * (1.0 - float(learning_rate) * float(weight_decay))
    return decayed - float(learning_rate) * grad / (np.abs(grad) + float(eps))


def reconcile_post_step(*, actual, expected, gradient_actual, gradient_expected, atol, rtol,
                        learning_rate, weight_decay, parameters_before, label):
    """Compare two routes' post-step parameters and *explain* every disagreement.

    Not a widened tolerance. Coordinates that disagree beyond ``atol``/``rtol`` are
    kept, counted and attributed, and the comparison passes only when every one of
    them is a coordinate where the two routes' gradients bracket zero -- that is,
    where ``min(|g_a|, |g_b|)`` is no larger than the measured cross-route gradient
    discrepancy at that coordinate.

    Why that is the honest criterion rather than a bigger number. AdamW's first
    step is ``-lr * g / (|g| + eps)``, which saturates at ``-lr * sign(g)`` for
    ``|g| >> eps``. Two float32 reduction orders differ in ``g`` by some ``delta``;
    where ``|g| > delta`` the resulting parameter difference is of order
    ``lr * eps * delta / g**2`` and is invisible, and where ``|g| <= delta`` the
    sign can flip and the step differs by up to ``2 * lr``. So a disagreement is
    admissible exactly when the gradient was within its own noise of zero -- and a
    genuinely different update, which moves a coordinate with a decided gradient,
    is not admitted by this rule at all. Every disagreement is additionally
    required to stay inside the maximum movement one step can produce.

    **Prerequisite, and it is not optional.** This is the second half of a
    two-part criterion: the caller must first have established that the two routes'
    *gradients* agree within the separately declared gradient tolerance. Taken
    alone, the near-zero rule can admit a pair like ``+0.5`` against ``-0.5``,
    where the two gradients bracket zero because they disagree completely --
    which the gradient comparison rejects outright, and which is why
    :func:`synthetic_gradient_control` computes ``gradients`` before
    ``cross_route_parameters``. The returned ``prerequisite`` field names it so a
    later reader of ``preflight.json`` cannot mistake this report for the whole
    acceptance.
    """
    left = np.asarray(actual, dtype=np.float64).reshape(-1)
    right = np.asarray(expected, dtype=np.float64).reshape(-1)
    grad_left = np.asarray(gradient_actual, dtype=np.float64).reshape(-1)
    grad_right = np.asarray(gradient_expected, dtype=np.float64).reshape(-1)
    before = np.asarray(parameters_before, dtype=np.float64).reshape(-1)
    for name, values in (("parameters", right), ("gradients", grad_left),
                         ("expected gradients", grad_right), ("prior parameters", before)):
        require(values.shape == left.shape,
                f"{label}: {name} are {values.shape} against {left.shape} compared parameters")
    # Explicit, because every comparison below is a `>` against a tolerance and a
    # NaN loses all of them: an array of NaNs would report zero coordinates outside
    # the tolerance and be declared reconciled. The native caller checks finiteness
    # too; this function is public and does not rely on that.
    for name, values in (("parameters", left), ("expected parameters", right),
                         ("gradients", grad_left), ("expected gradients", grad_right),
                         ("prior parameters", before)):
        require(bool(np.isfinite(values).all()),
                f"{label}: a nonfinite value entered the {name}. A NaN silently satisfies every "
                "tolerance comparison here, so it is refused rather than reconciled.")
    difference = np.abs(left - right)
    allowance = float(atol) + float(rtol) * np.abs(right)
    outside = np.flatnonzero(difference > allowance)
    gradient_gap = np.abs(grad_left - grad_right)
    smaller = np.minimum(np.abs(grad_left), np.abs(grad_right))
    # The largest distance one first step can move a parameter: the full saturated
    # AdamW step in each direction, plus the decoupled decay.
    ceiling = 2.0 * float(learning_rate) * (1.0 + float(weight_decay) * np.abs(before))
    explained = smaller[outside] <= gradient_gap[outside]
    bounded = difference[outside] <= ceiling[outside]
    unexplained = outside[~(explained & bounded)]
    report = {
        "values": int(left.size), "atol": float(atol), "rtol": float(rtol),
        "max_abs_error": float(difference.max()) if left.size else 0.0,
        "coordinates_outside_tolerance": int(outside.size),
        "coordinates_explained_by_near_zero_gradients": int(explained.sum()),
        "coordinates_within_one_step_ceiling": int(bounded.sum()),
        "worst_outside_abs_gradient": (float(smaller[outside].max()) if outside.size else None),
        "worst_outside_gradient_gap": (float(gradient_gap[outside].max()) if outside.size
                                       else None),
        "max_single_step_movement": float(ceiling.max()) if left.size else 0.0,
        "unexplained_coordinates": int(unexplained.size),
        "unexplained_indices": unexplained[:10].tolist(),
        "within_tolerance": bool(outside.size == 0),
        "reconciled": bool(unexplained.size == 0),
        "criterion": ("every coordinate outside the declared tolerance must (a) have the two "
                      "routes' gradients bracket zero within their own measured discrepancy, "
                      "where AdamW's sign normalization amplifies float32 reduction noise, and "
                      "(b) stay inside the maximum movement a single first step can produce. "
                      "Tolerances are not widened and small batches are not retried."),
        "prerequisite": ("this is half of a two-part criterion and is meaningful only after the "
                         "two routes' gradients agree within the separately declared gradient "
                         "tolerance. Two gradients that disagree completely also bracket zero, "
                         "and it is the gradient comparison -- run first -- that rejects them.")}
    require(report["reconciled"],
            f"{label}: {unexplained.size} coordinates differ beyond the declared tolerance at "
            f"gradients that are not near zero (worst absolute error "
            f"{report['max_abs_error']:.6g}, first indices {report['unexplained_indices']}). That "
            "is a different update, not a reduction-order difference, and it is a finding rather "
            "than a tolerance to move.")
    return report


def teacher_is_frozen(module):
    """``(ok, report)``: the teacher has no gradients and no trainable parameters.

    Called before the first update and again after it. "No ``requires_grad``" and
    "no ``.grad``" are different claims -- a module frozen after a backward pass
    still carries the gradients from it -- and both are checked.
    """
    live = sorted(name for name, parameter in module.named_parameters()
                  if parameter.requires_grad)
    gradients = sorted(name for name, parameter in module.named_parameters()
                       if parameter.grad is not None)
    report = {"parameters": sum(1 for _ in module.parameters()),
              "requires_grad": live, "carrying_gradients": gradients,
              "frozen": not live and not gradients}
    return report["frozen"], report


def state_vector(module):
    """A float64 copy of every parameter, flattened. The before/after comparison basis."""
    with torch.no_grad():
        parts = [parameter.detach().double().reshape(-1).cpu().numpy()
                 for _, parameter in sorted(module.named_parameters())]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)


def compare_vectors(actual, expected, *, atol, rtol, label):
    """``allclose``-style comparison that reports the measured error before judging.

    Separate from :func:`her2_policy.compare_sum_log_probabilities` on purpose: a
    gradient and a post-AdamW parameter are not sum log probabilities, and reusing
    that tolerance for them would either be too loose for a parameter or force the
    score tolerance to move. The numbers are recorded in the preflight evidence so
    a later widening has to argue against a measurement.
    """
    left = np.asarray(actual, dtype=np.float64).reshape(-1)
    right = np.asarray(expected, dtype=np.float64).reshape(-1)
    require(left.shape == right.shape, f"{label}: compared {left.shape} against {right.shape}")
    require(left.size > 0, f"{label}: nothing to compare")
    require(bool(np.isfinite(left).all()) and bool(np.isfinite(right).all()),
            f"{label}: a nonfinite value entered the comparison")
    difference = np.abs(left - right)
    allowance = float(atol) + float(rtol) * np.abs(right)
    worst = int(np.argmax(difference - allowance))
    report = {"values": int(left.size), "max_abs_error": float(difference.max()),
              "max_relative_error": float(np.max(difference / np.maximum(np.abs(right), 1e-12))),
              "worst_index": worst, "worst_abs_error": float(difference[worst]),
              "worst_allowance": float(allowance[worst]),
              "atol": float(atol), "rtol": float(rtol),
              "within_tolerance": bool((difference <= allowance).all())}
    require(report["within_tolerance"],
            f"{label}: max absolute error {report['max_abs_error']:.6g} exceeds the declared "
            f"tolerance (atol {atol:.1e}, rtol {rtol:.1e}) at index {worst}. This is a numerical "
            "investigation and a recorded amendment, not a tolerance to widen in place.")
    return report


def cores_to_token_ids(policy, index):
    """``(B, 10)`` vocabulary ids for canonical core indices, through the policy's seam.

    Routed through :meth:`CorePolicy.token_ids` rather than indexed directly so a
    fractional, negative or wrongly shaped block is refused there, in the one place
    that refuses it for every other caller too.
    """
    return policy.token_ids(np.asarray(index))


def student_log_probabilities(policy, index):
    """``(B, 10, 20)`` student log probabilities, differentiable through the prefix.

    ``CorePolicy.core_logits`` builds its prefix cache inside the call and drops
    it, so the prefix computation is part of this graph and the student's prefix
    parameters receive gradient. Nothing is detached here: detaching the prefix
    would silently train only the core positions.
    """
    core_ids = cores_to_token_ids(policy, index)
    return torch.log_softmax(policy.core_logits(core_ids).float(), dim=-1)


def teacher_log_probabilities(policy, index):
    """The frozen parent's ``(B, 10, 20)`` log probabilities, detached and no-grad.

    ``log_softmax`` of finite logits is finite by construction, so the stored logs
    never carry ``-inf`` and the zero-probability convention never has to divide,
    mask or add an epsilon to a stored value.
    """
    with torch.no_grad():
        core_ids = cores_to_token_ids(policy, index)
        return torch.log_softmax(policy.core_logits(core_ids).float(), dim=-1).detach()


def require_core_block(index, *, label):
    """``(N, 10)`` integral canonical indices in range. Shared by banks and streams."""
    values = np.asarray(index)
    require(values.ndim == 2 and values.shape[1] == CORE_LENGTH,
            f"{label}: expected (N, {CORE_LENGTH}) cores, got shape {values.shape}")
    require(values.shape[0] > 0, f"{label}: empty core block")
    require(np.issubdtype(values.dtype, np.integer),
            f"{label}: core indices must be integral, got dtype {values.dtype}")
    require(bool(((values >= 0) & (values < TEACHER_WIDTH)).all()),
            f"{label}: a core index sits outside [0, {TEACHER_WIDTH})")
    return values
