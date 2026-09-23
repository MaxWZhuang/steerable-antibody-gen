"""Task losses, the buffered tail penalty, and what each arm actually consumes.

The task losses are the inherited, tested ones: ``her2_objectives.ipo_per_pair``
and ``her2_objectives.dpo_per_pair`` are called here rather than re-derived, so
a reused historical IPO arm and this flight's IPO arm are the same function and
not two implementations that agree today.

What is new:

* **the DPO scale.** Raw DPO's derivative with respect to ``Delta`` at
  ``Delta = 0`` is ``-beta/2``; the flight's IPO, with ``tau = .1`` and target
  ``1/(2 tau) = 5``, has derivative ``-10``. Multiplying raw DPO by
  ``c = 4 * target / beta = 20 / beta`` matches those two scalars exactly, and
  leaves IPO untouched. That controls the *deterministic initial scale* and
  nothing later: not the curvature, not clipping, not AdamW's normalization. Raw
  and scaled losses are both logged, and the raw-DPO preservation weight
  ``lambda / c`` is recorded as the algebraic identity it is.
* **the buffered squared hinge.** ``h_t(d) = max(0, (d - (t - b)) / b) ** 2``
  with ``b = ln 2``, so the penalty is already 1 at ``d = t`` and the indicator
  ``1{d >= t}`` is bounded above by it. That bound is what makes the population
  mean an upper bound on the tail *rate*; an unbuffered hinge at the threshold
  penalizes severity without bounding the event.
* **an exposure registry.** ``run_trajectory``'s inherited accounting keys the
  rejected exposure on ``task == "ipo"``. DPO and SimPO consume a rejected row
  too, so counting them that way would under-report every new arm's exposure.
  Each objective declares ``uses_rejected``/``uses_reference`` and the loop reads
  the declaration.

The tail penalty is a proposed surrogate, not an established algorithm. At two
identical deterministic distributions ``d = 0``, the penalty and its gradient
are exactly zero, so no initial gradient-ratio calibration can set its lambda --
that is a property of the objective and the reason its ladder is run empirically.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from . import her2_objectives as objectives
from .her2_data import CORE_LENGTH
from .her2_nf_contract import LN10, LN100, NF_SCHEMA
from .her2_runtime import require

#: Inherited IPO settings. ``target = 1 / (2 * tau)``.
IPO_TAU = 0.1
IPO_TARGET = 1.0 / (2.0 * IPO_TAU)

#: Buffered-tail constants, from the specification.
TAIL_BUFFER = math.log(2.0)
TAIL_TOLERANCES = {"tenfold": 0.01, "hundredfold": 0.001}
TAIL_THRESHOLD_NATS = {"tenfold": LN10, "hundredfold": LN100}


def dpo_scale(beta, *, ipo_target=IPO_TARGET):
    """``c = 4 * target / beta``: the multiplier that matches IPO's initial slope.

    Derivation, so it can be checked rather than believed: ``d/dDelta
    softplus(-beta Delta) = -beta sigma(-beta Delta)``, which at ``Delta = 0`` is
    ``-beta/2``. ``d/dDelta (Delta - m)**2 = 2(Delta - m)``, which at
    ``Delta = 0`` is ``-2m``. Setting ``c * beta/2 = 2m`` gives ``c = 4m/beta``.
    With ``m = 5`` that is ``20/beta``.
    """
    beta = float(beta)
    require(math.isfinite(beta) and beta > 0, f"beta must be finite and positive, got {beta!r}")
    return 4.0 * float(ipo_target) / beta


# ---------------------------------------------------------------------------
# the objective registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ObjectiveSpec:
    """What one task loss is, and what one update of it actually consumes."""

    name: str
    formula: str
    uses_rejected: bool
    uses_reference: bool
    coefficient_keys: tuple
    note: str = ""

    def document(self):
        return {"objective": self.name, "formula": self.formula,
                "uses_rejected": bool(self.uses_rejected),
                "uses_reference": bool(self.uses_reference),
                "coefficients": list(self.coefficient_keys), "note": self.note}


TASK_OBJECTIVES = {
    "ipo": ObjectiveSpec(
        name="ipo", formula="mean_pairs (Delta - 1/(2*tau))**2", uses_rejected=True,
        uses_reference=True, coefficient_keys=("tau",),
        note="the inherited historical task loss, called unchanged through her2_objectives"),
    "dpo": ObjectiveSpec(
        name="dpo", formula="mean_pairs (4*target/beta) * softplus(-beta*Delta)",
        uses_rejected=True, uses_reference=True, coefficient_keys=("beta",),
        note="raw DPO times the initial-scale multiplier; raw and scaled are both logged"),
    "simpo": ObjectiveSpec(
        name="simpo", formula="mean_pairs softplus(gamma - beta_s * m_Q / 10)",
        uses_rejected=True, uses_reference=False, coefficient_keys=("beta_s", "gamma", "scale"),
        note=("no reference subtraction inside the task loss. An external FKL penalty still uses "
              "the parent, so the ARM is 'SimPO task loss + parent replay' and is not a wholly "
              "reference-free system. At length ten the normalization is a fixed scale, and gamma "
              "and reference removal change together: one arm cannot isolate either cause.")),
}

PRESERVATION_FAMILIES = {
    "none": {"family": "none", "coefficient": None,
             "note": "no preservation term; no replay forward pass, no bank read, no stream move"},
    "fkl": {"family": "fkl", "coefficient": "lambda",
            "formula": "mean_b sum_i sum_a p_bia * (log p_bia - log q_bia)",
            "note": "sums positions and residues, averages replay rows; P is frozen"},
    "tail": {"family": "tail", "coefficient": "lambda",
             "formula": "mean_x [ h_t10(d)/.01 + h_t100(d)/.001 ], h_t(d)=max(0,(d-(t-b))/b)**2",
             "note": "buffered squared hinge on the parent-draw sequence drop d = ell_P - ell_Q"},
}


def resolve_arm(task, preservation, coefficients):
    """The frozen identity of one training cell: names, coefficients, exposures."""
    require(task in TASK_OBJECTIVES, f"Unknown task objective {task!r}")
    require(preservation in PRESERVATION_FAMILIES, f"Unknown preservation family {preservation!r}")
    spec = TASK_OBJECTIVES[task]
    values = {str(k): float(v) for k, v in dict(coefficients or {}).items()}
    missing = [key for key in spec.coefficient_keys
               if key not in values and key not in ("scale",)]
    require(not missing, f"{task} requires coefficients {missing}")
    family = PRESERVATION_FAMILIES[preservation]
    if family["coefficient"] is not None:
        require("lambda" in values, f"{preservation} preservation requires a lambda coefficient")
        require(values["lambda"] >= 0, "lambda must be >= 0")
    else:
        require(float(values.get("lambda", 0.0)) == 0.0,
                "the 'none' preservation family must not carry a nonzero lambda; a control that "
                "computed a zero-weighted term would still have paid for its forward pass")
    record = {"schema_version": NF_SCHEMA, "record_kind": "arm_identity",
              "task": task, "preservation": preservation,
              "coefficients": dict(sorted(values.items())),
              "task_spec": spec.document(), "preservation_spec": dict(family),
              "arm_id": arm_identifier(task, preservation, values)}
    if task == "dpo":
        scale = dpo_scale(values["beta"])
        record["dpo_scale"] = scale
        record["raw_lambda_equivalent"] = (values.get("lambda", 0.0) / scale
                                           if preservation != "none" else None)
        record["scale_note"] = (
            "the scaled loss is c * raw with c = 4*target/beta. Dividing lambda by c gives an "
            "algebraically equivalent total objective up to one global multiplier; it does NOT "
            "imply identical optimizer trajectories, because AdamW normalizes by a running "
            "gradient scale and the clip threshold is absolute.")
    return record


def arm_identifier(task, preservation, coefficients):
    parts = [str(task), str(preservation)]
    for key in sorted(dict(coefficients)):
        parts.append(f"{key}{_format(float(coefficients[key]))}")
    return "_".join(parts)


def _format(value):
    text = f"{value:g}".replace("-", "m").replace(".", "p")
    return text


# ---------------------------------------------------------------------------
# task losses
# ---------------------------------------------------------------------------

def task_term(task, *, policy_chosen, policy_rejected, reference_chosen=None,
              reference_rejected=None, coefficients):
    """``(mean loss, detached diagnostics)`` for one microbatch of one task loss."""
    require(task in TASK_OBJECTIVES, f"Unknown task objective {task!r}")
    values = {str(k): float(v) for k, v in dict(coefficients).items()}
    if task == "ipo":
        _require_reference(reference_chosen, reference_rejected, task)
        mean, diagnostics = objectives.batch_loss(
            "ipo", policy_chosen=policy_chosen, policy_rejected=policy_rejected,
            reference_chosen=reference_chosen, reference_rejected=reference_rejected,
            coefficients={"tau": values.get("tau", IPO_TAU)})
        block = _detach_block(diagnostics)
        block.update(raw_task_loss=float(mean.detach()), scaled_task_loss=float(mean.detach()),
                     loss_multiplier=1.0)
        return mean, block
    if task == "dpo":
        _require_reference(reference_chosen, reference_rejected, task)
        beta = values["beta"]
        raw, diagnostics = objectives.batch_loss(
            "dpo", policy_chosen=policy_chosen, policy_rejected=policy_rejected,
            reference_chosen=reference_chosen, reference_rejected=reference_rejected,
            coefficients={"beta": beta})
        scale = dpo_scale(beta)
        mean = scale * raw
        block = _detach_block(diagnostics)
        block.update(raw_task_loss=float(raw.detach()), scaled_task_loss=float(mean.detach()),
                     loss_multiplier=float(scale), beta=float(beta))
        return mean, block
    return simpo_term(policy_chosen=policy_chosen, policy_rejected=policy_rejected,
                      beta_s=values["beta_s"], gamma=values["gamma"],
                      scale=values.get("scale", 1.0))


def _require_reference(reference_chosen, reference_rejected, task):
    require(reference_chosen is not None and reference_rejected is not None,
            f"{task} is reference-relative: both frozen parent vectors are required")
    objectives.require_frozen_references(reference_chosen, reference_rejected,
                                         where=f"{task} microbatch")


def simpo_term(*, policy_chosen, policy_rejected, beta_s, gamma, scale=1.0, length=CORE_LENGTH):
    """``mean softplus(gamma - beta_s * m_Q / length)`` with no reference subtraction.

    ``scale`` is the separately calibrated constant loss multiplier, frozen after
    one measurement on a fixed training-only batch. It is weaker matching than
    DPO's exact initial identity and must never be described as equivalent
    optimization.
    """
    require(float(beta_s) > 0 and float(gamma) >= 0, "SimPO needs beta_s > 0 and gamma >= 0")
    require(float(scale) > 0, "the SimPO loss multiplier must be positive")
    margin = policy_chosen - policy_rejected
    per_pair = torch.nn.functional.softplus(float(gamma) - float(beta_s) * margin / float(length))
    mean = float(scale) * per_pair.mean()
    with torch.no_grad():
        detached = margin.detach()
        block = {"margin": detached, "raw_margin_sum": detached,
                 "sign_correct": (detached > 0).double(),
                 "chosen_nll_per_residue": (-policy_chosen.detach() / float(length)),
                 "raw_task_loss": float(per_pair.detach().mean()),
                 "scaled_task_loss": float(mean.detach()),
                 "loss_multiplier": float(scale), "beta_s": float(beta_s),
                 "gamma": float(gamma),
                 "reference_note": ("no reference inside this task loss. An FKL penalty beside it "
                                    "still uses the parent.")}
    return mean, block


def _detach_block(diagnostics):
    out = {}
    for key, value in dict(diagnostics).items():
        out[key] = value.detach() if isinstance(value, torch.Tensor) else value
    return out


# ---------------------------------------------------------------------------
# the buffered tail penalty
# ---------------------------------------------------------------------------

def buffered_squared_hinge(drop, threshold, *, buffer=TAIL_BUFFER):
    """``max(0, (d - (t - b)) / b) ** 2``, elementwise, differentiable in ``d``.

    Two properties the flight relies on: it is >= 1 at and above ``t`` (so it
    upper-bounds the threshold indicator), and it is exactly zero below
    ``t - b`` (so a policy that has not lost more than half the parent's
    probability contributes nothing and receives no gradient).
    """
    require(float(buffer) > 0, "the buffer must be positive")
    return torch.clamp((drop - (float(threshold) - float(buffer))) / float(buffer),
                       min=0.0) ** 2


def tail_penalty_per_row(drop, *, buffer=TAIL_BUFFER, thresholds=None, tolerances=None):
    """``h_t10(d)/.01 + h_t100(d)/.001`` per parent draw, plus the two components."""
    table = dict(thresholds or TAIL_THRESHOLD_NATS)
    tol = dict(tolerances or TAIL_TOLERANCES)
    components = {}
    total = None
    for name in sorted(table):
        require(name in tol, f"no tolerance declared for tail threshold {name!r}")
        term = buffered_squared_hinge(drop, table[name], buffer=buffer) / float(tol[name])
        components[name] = term
        total = term if total is None else total + term
    return total, components


def tail_term(parent_log_probability, policy_log_probability, *, buffer=TAIL_BUFFER,
              thresholds=None, tolerances=None):
    """``(mean penalty, diagnostics)`` on one microbatch of parent draws.

    ``d = ell_P - ell_Q`` with the parent detached. The diagnostics are what a
    calibration ladder is read from: how often the penalty was active at all, how
    much each threshold contributed, and the largest drop seen -- because a
    ladder whose penalty never activated has not distinguished its lambda values,
    and reporting three indistinguishable numbers as a calibration would be the
    failure this logging exists to expose.
    """
    require(isinstance(parent_log_probability, torch.Tensor)
            and isinstance(policy_log_probability, torch.Tensor),
            "tail_term consumes tensors")
    require(not parent_log_probability.requires_grad,
            "the parent reference must be detached: a tail penalty that back-propagates into the "
            "frozen parent is not the declared objective")
    drop = parent_log_probability.detach() - policy_log_probability
    total, components = tail_penalty_per_row(drop, buffer=buffer, thresholds=thresholds,
                                             tolerances=tolerances)
    mean = total.mean()
    with torch.no_grad():
        detached = drop.detach().double()
        block = {
            "rows": int(detached.numel()),
            "mean_penalty": float(mean.detach()),
            "active_fraction": float((total.detach() > 0).double().mean()),
            "max_drop_nats": float(detached.max()),
            "mean_drop_nats": float(detached.mean()),
            "component_means": {name: float(term.detach().double().mean())
                                for name, term in sorted(components.items())},
            "active_fraction_by_threshold": {
                name: float((term.detach() > 0).double().mean())
                for name, term in sorted(components.items())},
            "inclusive_event_counts": {
                name: int((detached >= float(value)).sum())
                for name, value in sorted(dict(thresholds or TAIL_THRESHOLD_NATS).items())},
            "buffer_nats": float(buffer),
            "bound_note": ("1{d >= t} <= h_t(d), so the POPULATION mean of this penalty upper-"
                           "bounds the corresponding tail rate. A minibatch penalty is not a "
                           "preservation certificate."),
            "inert_note": ("at d = 0 the penalty and its gradient are exactly zero, so an initial "
                           "gradient-ratio calibration cannot set this lambda. Train-mode "
                           "fluctuations are a different quantity from the deterministic "
                           "evaluation-policy audit and are logged apart.")}
    return mean, block


# ---------------------------------------------------------------------------
# the initial-scale identity, as an arithmetic statement
# ---------------------------------------------------------------------------

def initial_scale_identity(beta, *, ipo_target=IPO_TARGET):
    """The two scalar derivatives at ``Delta = 0`` and the multiplier that equates them.

    Returned as numbers so a test compares them rather than reading a docstring.
    This is a statement about one point of two scalar functions. It says nothing
    about parameter gradients unless the task forward is identical and the policy
    and reference probabilities actually agree, and nothing at all about later
    dynamics.
    """
    beta = float(beta)
    scale = dpo_scale(beta, ipo_target=ipo_target)
    return {"beta": beta, "ipo_target": float(ipo_target),
            "ipo_derivative_at_zero": -2.0 * float(ipo_target),
            "raw_dpo_derivative_at_zero": -beta / 2.0,
            "loss_multiplier": scale,
            "scaled_dpo_derivative_at_zero": -scale * beta / 2.0,
            "holds_when": ("the task forward is identical and the policy and reference sequence "
                           "log probabilities agree exactly, so Delta is exactly zero. A cached "
                           "reference built at a different batch shape differs from a live one by "
                           "float32 reduction order, which is measured separately and is not a "
                           "failure of this identity."),
            "does_not_control": ["curvature", "gradient clipping", "AdamW normalization",
                                 "any update after the first"]}


def gradient_diagnostics(task_gradients, preservation_gradients):
    """Per-term gradient norms and their cosine, for the modest diagnostic cadence.

    The cosine is the number that says whether the preservation term is opposing
    the task term or merely shrinking it, and it cannot be recovered from the two
    norms or from the total.
    """
    task = _flatten(task_gradients)
    keep = _flatten(preservation_gradients)
    if task is None or keep is None:
        return {"task_gradient_norm": None if task is None else float(task.norm()),
                "preservation_gradient_norm": None if keep is None else float(keep.norm()),
                "cosine": None,
                "reason": "one of the two components produced no gradient on this update"}
    task_norm = float(task.norm())
    keep_norm = float(keep.norm())
    cosine = (float(torch.dot(task, keep)) / (task_norm * keep_norm)
              if task_norm > 0 and keep_norm > 0 else None)
    return {"task_gradient_norm": task_norm, "preservation_gradient_norm": keep_norm,
            "cosine": cosine,
            "ratio": (keep_norm / task_norm) if task_norm > 0 else None}


def _flatten(gradients):
    parts = [g.detach().reshape(-1).double() for g in (gradients or []) if g is not None]
    return torch.cat(parts) if parts else None
