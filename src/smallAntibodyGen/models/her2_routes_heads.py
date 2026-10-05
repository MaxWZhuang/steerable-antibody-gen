"""Observation heads with assay semantics: ordinal, interval-censored, enrichment.

The existing HER2 heads are three-class display-bin classifiers. That is one of
the four measurement kinds this campaign trains on, and treating the other three
the same way is what produced the result the plan is trying to break: display
ranking improved from .8539 to .8776 macro-AP while quantitative affinity ordering
did not move. So the four kinds stay distinguishable in the *likelihood*, not just
in a label:

* **ordinal** -- sorted FACS bins. A cumulative-logit model,
  ``P(y <= k) = sigmoid(theta_k - f(x))``, with ``theta_1 = theta_0 + softplus(d)``
  so the thresholds cannot cross. This uses all 367,042 display rows rather than
  the 120,504 positives a positives-only objective sees, and the matched
  positives-only control is a separate arm rather than a footnote.
* **interval-censored continuous** -- KD and pKD with a detection window.
  ``-log[Phi((u-mu)/s) - Phi((l-mu)/s)]`` with a per-assay learned ``log s``.
  An Adams Tite-Seq measurement at the bottom of the window is not "KD = 1e-9.5";
  it is ``pKD >= 9.5``, and a squared error against the boundary value trains the
  model to believe a number the assay did not produce.
* **point continuous** -- the same head with ``l == u``, which reduces to the
  Gaussian log density. Written as the same code path so a source cannot
  accidentally get a different noise model by being uncensored.
* **enrichment / pseudo-KD** -- Koenig's ``KD (nM)`` column is *derived from phage
  enrichment*, not a titration. It trains a **within-study pairwise ordering**
  loss and never the interval likelihood, because fitting a KD-scaled Gaussian to
  an enrichment-derived number states a precision the experiment does not have.

Two mechanical properties come from the ``add-model-head`` pattern's two
easy-to-miss steps, and both have tests:

* the masked-empty case returns a **differentiable** zero (``predictions.sum() *
  0.0``), not a fresh ``torch.zeros``, so an epoch with no eligible rows does not
  silently detach the head from the graph;
* the accumulators are updated in **train and eval**, and
  :meth:`HeadAccumulator.require_nonempty` refuses a reported metric whose
  eligible-row count is zero. An always-empty metric that reports 0.0 is the
  failure mode; it looks like a result.
"""
from __future__ import annotations

import math

import torch

from ..experiments.her2_runtime import require

#: The measurement kinds a head may be asked for.
KIND_ORDINAL = "ordinal"
KIND_INTERVAL = "interval"
KIND_POINT = "point"
KIND_ENRICHMENT = "enrichment"
KINDS = (KIND_ORDINAL, KIND_INTERVAL, KIND_POINT, KIND_ENRICHMENT)

#: Floor on the learned observation scale, in the units of the qualified value.
#: Not a tuned hyperparameter: it stops the interval likelihood from driving
#: ``s -> 0`` on a source whose replicate spread is genuinely zero because it was
#: reported once.
MIN_OBSERVATION_SCALE = 1e-3


def differentiable_zero(reference):
    """A zero that still carries a gradient path back to ``reference``.

    ``torch.zeros_like(x)`` is the wrong answer: it detaches. An epoch in which a
    head has no eligible rows must contribute exactly zero loss *and* leave the
    parameters connected, or the very first all-masked microbatch turns the head
    into a silent no-op for the rest of training.
    """
    return reference.sum() * 0.0


def _log_ndtr(value):
    return torch.special.log_ndtr(value)


def interval_log_probability(mu, scale, lower, upper, *, finite_lower, finite_upper):
    """``log P(lower <= Y <= upper)`` for ``Y ~ Normal(mu, scale)``, stably.

    Four cases, all in one function so a source cannot get a different noise model
    merely by being uncensored:

    * both bounds finite and distinct -- ``log(Phi(b) - Phi(a))``;
    * ``upper`` infinite -- right-censored, ``log_ndtr(-a)``;
    * ``lower`` infinite -- left-censored, ``log_ndtr(b)``;
    * bounds equal -- a point observation, the Gaussian log density.

    Two numerical decisions, both load-bearing:

    * **the two-sided case is computed in whichever tail has precision.** With
      ``a = 20, b = 21`` both ``Phi(a)`` and ``Phi(b)`` round to 1 and the naive
      difference loses every significant digit -- it would report about ``-16``
      where the answer is about ``-207``. So when ``a >= 0`` the identity
      ``Phi(b) - Phi(a) = Phi(-a) - Phi(-b)`` is used instead, where both terms are
      small and ``log_ndtr`` is accurate.
    * **infinite bounds are replaced by finite sentinels before any arithmetic.**
      ``torch.where`` evaluates both branches, so an ``inf`` in an unselected branch
      produces a NaN gradient even though the forward value is correct. The
      sentinels sit ten scales from the mean, which is beyond any branch that reads
      them.
    """
    scale = torch.clamp(scale, min=MIN_OBSERVATION_SCALE)
    # Finite sentinels first: an unselected branch must not see an infinity.
    safe_lower = torch.where(finite_lower, lower, mu - 10.0 * scale)
    safe_upper = torch.where(finite_upper, upper, mu + 10.0 * scale)
    a = (safe_lower - mu) / scale
    b = (safe_upper - mu) / scale
    # Work in the tail that carries precision. ``big`` is always the larger log.
    upper_tail = a >= 0
    big = torch.where(upper_tail, _log_ndtr(-a), _log_ndtr(b))
    small = torch.where(upper_tail, _log_ndtr(-b), _log_ndtr(a))
    ratio = torch.clamp(small - big, max=-1e-12)
    both = big + torch.log1p(-torch.exp(ratio))
    right = _log_ndtr(-a)
    left = _log_ndtr(b)
    point = (-0.5 * a ** 2 - torch.log(scale) - 0.5 * math.log(2.0 * math.pi))
    is_point = finite_lower & finite_upper & (safe_upper <= safe_lower)
    out = torch.where(finite_upper, torch.where(finite_lower, both, left), right)
    return torch.where(is_point, point, out)


class ObservationHeads(torch.nn.Module):
    """One shared trunk projection plus the per-kind likelihoods.

    The projection is shared so the *representation* is what the auxiliary losses
    teach; the per-assay scale and the ordinal thresholds are the only per-source
    parameters. A per-assay *predictor* would let each source fit its own function
    and would make transfer untestable.
    """

    def __init__(self, hidden_size, *, assays, ordinal_classes=3, width=256):
        super().__init__()
        assays = tuple(assays)
        require(assays, "The observation heads need at least one enrolled assay")
        require(int(ordinal_classes) >= 2, "An ordinal head needs at least two classes")
        self.assays = assays
        self.assay_index = {name: position for position, name in enumerate(assays)}
        self.ordinal_classes = int(ordinal_classes)
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(int(hidden_size), int(width)), torch.nn.GELU(),
            torch.nn.LayerNorm(int(width)))
        self.score = torch.nn.Linear(int(width), 1)
        # Ordinal thresholds: the first is free, each subsequent one is the previous
        # plus softplus(delta), so theta is strictly increasing by construction and
        # no penalty or projection step is needed to keep it that way.
        self.threshold_base = torch.nn.Parameter(torch.zeros(1))
        self.threshold_deltas = torch.nn.Parameter(torch.zeros(max(0, self.ordinal_classes - 2)))
        self.log_scale = torch.nn.Parameter(torch.zeros(len(assays)))

    def thresholds(self):
        """``(K-1,)`` strictly increasing cumulative-logit cut points."""
        steps = torch.nn.functional.softplus(self.threshold_deltas)
        return self.threshold_base + torch.cat(
            [torch.zeros(1, device=steps.device, dtype=steps.dtype), torch.cumsum(steps, 0)])[
            :self.ordinal_classes - 1]

    def latent(self, representation):
        """``(B,)`` the shared scalar every kind reads. Higher means stronger."""
        return self.score(self.trunk(representation)).squeeze(-1)

    def assay_scale(self, assay_indices):
        return torch.exp(self.log_scale.index_select(0, assay_indices))

    # -- likelihoods -------------------------------------------------------
    def ordinal_loss(self, representation, classes, mask):
        """Cumulative-logit negative log likelihood over ``ordinal_classes`` bins."""
        latent = self.latent(representation)
        if mask.sum() == 0:
            return differentiable_zero(latent), 0
        theta = self.thresholds()
        # P(y <= k) = sigmoid(theta_k - f); the per-class probability is the
        # difference of consecutive cumulatives, with the ends pinned at 0 and 1.
        cumulative = torch.sigmoid(theta.unsqueeze(0) - latent.unsqueeze(1))
        ones = torch.ones_like(cumulative[:, :1])
        zeros = torch.zeros_like(ones)
        padded = torch.cat([zeros, cumulative, ones], dim=1)
        probabilities = torch.clamp(padded[:, 1:] - padded[:, :-1], min=1e-12)
        picked = probabilities.gather(1, classes.unsqueeze(1).clamp(
            0, self.ordinal_classes - 1)).squeeze(1)
        per_row = -torch.log(picked) * mask
        return per_row.sum() / torch.clamp(mask.sum(), min=1.0), int(mask.sum())

    def interval_loss(self, representation, lower, upper, finite_lower, finite_upper,
                      assay_indices, mask, *, reported=None):
        """Interval-censored Gaussian negative log likelihood.

        ``reported`` is the measurement's own uncertainty -- a reported standard
        error, or the replicate spread where no SE was published -- and it is
        combined with the learned per-assay scale in quadrature:
        ``s^2 = exp(2 log s_assay) + se^2``. Ignoring it makes every replicate of a
        construct with a 0.4-pKD fit uncertainty as informative as one with 0.01,
        and the learned scale then absorbs the average of the two. A row whose
        reported value is absent or non-finite contributes the learned scale alone,
        which is the same code path rather than a second noise model.
        """
        latent = self.latent(representation)
        if mask.sum() == 0:
            return differentiable_zero(latent), 0
        scale = self.assay_scale(assay_indices)
        if reported is not None:
            extra = torch.as_tensor(reported, dtype=scale.dtype, device=scale.device)
            extra = torch.where(torch.isfinite(extra) & (extra > 0), extra,
                                torch.zeros_like(extra))
            scale = torch.sqrt(scale ** 2 + extra ** 2)
        log_probability = interval_log_probability(
            latent, scale, lower, upper, finite_lower=finite_lower, finite_upper=finite_upper)
        per_row = -log_probability * mask
        return per_row.sum() / torch.clamp(mask.sum(), min=1.0), int(mask.sum())

    def enrichment_loss(self, representation, left_rows, right_rows, mask):
        """Within-study pairwise ordering: ``softplus(-(f(left) - f(right)))``.

        ``left`` is the more-enriched member of each pair. Deliberately a ranking
        loss: an enrichment ratio and a titration KD do not share a scale, and the
        only claim this source supports is an ordering within its own study.
        """
        latent = self.latent(representation)
        if mask.sum() == 0:
            return differentiable_zero(latent), 0
        margin = latent.index_select(0, left_rows) - latent.index_select(0, right_rows)
        per_pair = torch.nn.functional.softplus(-margin) * mask
        return per_pair.sum() / torch.clamp(mask.sum(), min=1.0), int(mask.sum())

    def document(self):
        return {"assays": list(self.assays), "ordinal_classes": self.ordinal_classes,
                "kinds": list(KINDS),
                "threshold_parameterisation": "theta_0 free, theta_k = theta_{k-1} + softplus(d_k)",
                "scale_parameterisation": "one learned log sigma per enrolled assay",
                "shared": ("one projection and one scalar for every kind. A per-assay predictor "
                           "would let each source fit its own function and make transfer "
                           "untestable."),
                "min_observation_scale": MIN_OBSERVATION_SCALE}


class HeadAccumulator:
    """Per-kind loss and eligible-row accumulation, for train and eval alike.

    The eligible-row counter is the load-bearing part. ``add-model-head``'s two
    silent failures are (a) accumulating in train but not in eval, so the reported
    metric is always the initial value, and (b) a filter that matches nothing, so
    the reported metric is always zero. Both look like results.
    """

    def __init__(self, kinds=KINDS):
        self.kinds = tuple(kinds)
        self.reset()

    def reset(self):
        self.loss = {kind: 0.0 for kind in self.kinds}
        self.rows = {kind: 0 for kind in self.kinds}
        self.batches = {kind: 0 for kind in self.kinds}
        return self

    def add(self, kind, loss, rows):
        require(kind in self.kinds, f"Unknown head kind {kind!r}")
        self.loss[kind] += float(loss) * int(rows)
        self.rows[kind] += int(rows)
        self.batches[kind] += 1
        return self

    def mean(self, kind):
        return None if self.rows[kind] == 0 else self.loss[kind] / self.rows[kind]

    def require_nonempty(self, kinds, *, label="observation heads"):
        empty = sorted(kind for kind in kinds if self.rows.get(kind, 0) == 0)
        require(not empty,
                f"{label}: the head kind(s) {empty} saw zero eligible rows, so their reported "
                "metric is not a measurement. Either the eligibility filter matches nothing or "
                "the accumulator is not being updated on this code path; both produce a "
                "plausible-looking number.")
        return True

    def document(self):
        return {"mean_loss": {kind: self.mean(kind) for kind in self.kinds},
                "eligible_rows": dict(self.rows), "batches": dict(self.batches),
                "rule": "a kind with zero eligible rows reports null, never 0.0"}
