"""Standalone scalar head, antibody pooling and pair loss for functional transfer (M1).

Nothing here is wired into the trainer, checkpoint or CLI yet.  The pieces are:

* :class:`ObservationHead` -- bias-free ``Linear(H, 1)`` in FP32, initialized
  ``Normal(0, H ** -0.5)`` on CPU under a purpose-keyed isolated RNG, then moved to the
  device (:func:`new_head`), so the weights do not depend on the device or on any outer
  RNG, and the outer RNG is left untouched.  Head and loss arithmetic run with autocast
  disabled on the feature device, so an ambient BF16/FP16 autocast cannot turn the score
  or the loss into low precision.
* :func:`antibody_features` -- mean of the fused final hidden state over the antibody
  query's H/L residue slots only (never ``Ag``, special tokens or padding).  Each endpoint
  gets a FRESH complete-view :class:`DynamicContextProvider`; a complete provider and a
  masked reconstruction provider are never interchangeable.
* :func:`pair_loss` -- ``ordering_loss`` (scalar mean of ``relu(margin - (better - worse))**2``)
  scaled by ``weight / pairs_per_update``.
* :func:`preflight_complete_views` -- validates the complete paired AND complete ab-only
  view of every construct through ``policy.preflight``, so the admitted set is the same in
  both arms (an over-long antigen is rejected even for an ab-only run).

The frozen encoder never receives a gradient.  ``gradient_mode`` only chooses whether the
policy features are detached before the head; it changes no data and no draw.

Clean (dropout-free) ordering features are the caller's responsibility: wrap the forward
AND the backward in ``runner.deterministic_ordering``.  This module cannot import that
context manager (``runner`` will import this module), so :func:`endpoint_features` instead
refuses to run while any active dropout would make the feature noisy.

``construct`` arguments are read through ``.H``, ``.L`` and ``.Ag`` only: identifiers,
groups, contexts and outcomes never reach a model input.
"""
from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn

from smallAntibodyGen.experiments.trastuzumab_poc.objectives import ordering_loss
from smallAntibodyGen.experiments.trastuzumab_poc.randomness import (
    isolated_torch_rng, keyed_seed,
)
from smallAntibodyGen.models.antibody_antigen.dynamic_context import DynamicContextProvider
from smallAntibodyGen.models.antibody_antigen.views import VisibleView, mask_complete

ARMS = ("paired", "abonly")
GRADIENT_MODES = ("shared", "detached")
HEAD_INIT_PURPOSE = "functional-head-init"


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")


def _autocast_disabled(device):
    """Context that turns autocast off on ``device``.

    An ambient autocast casts ``nn.Linear`` to its low-precision dtype even for FP32
    inputs, which would make the score and the loss BF16/FP16.
    """
    try:
        return torch.autocast(device_type=torch.device(device).type, enabled=False)
    except RuntimeError:                  # device type without autocast: nothing to disable
        return contextlib.nullcontext()


@dataclass(frozen=True)
class FunctionalConfig:
    gradient_mode: str = "shared"
    weight: float = 1.0
    pairs_per_update: int = 1
    margin: float = 0.1
    head_lr_scale: float = 1.0
    head_weight_decay: float = 0.0
    head_gradient_clip: float = 1.0
    minimum_compared_fraction: float = 1.0

    def __post_init__(self):
        if self.gradient_mode not in GRADIENT_MODES:
            raise ValueError(f"gradient_mode must be one of {GRADIENT_MODES}")
        if type(self.pairs_per_update) is not int or self.pairs_per_update < 1:
            raise ValueError("pairs_per_update must be a positive integer")
        for name in ("weight", "margin", "head_lr_scale", "head_weight_decay",
                     "head_gradient_clip", "minimum_compared_fraction"):
            _finite_number(getattr(self, name), name)
        if min(self.weight, self.margin, self.head_lr_scale, self.head_gradient_clip) <= 0:
            raise ValueError("weight, margin, head_lr_scale and head_gradient_clip must be positive")
        if self.head_weight_decay < 0:
            raise ValueError("head_weight_decay must be nonnegative")
        if not 0 <= self.minimum_compared_fraction <= 1:
            raise ValueError("minimum_compared_fraction must be between zero and one")

    @property
    def policy_receives_gradient(self) -> bool:
        return self.gradient_mode == "shared"


class ObservationHead(nn.Linear):
    """Bias-free ``Linear(H, 1)``; maps ``[..., H]`` features to ``[...]`` scores (FP32)."""

    def __init__(self, hidden_size: int):
        if type(hidden_size) is not int or hidden_size < 1:
            raise ValueError("hidden_size must be a positive integer")
        super().__init__(hidden_size, 1, bias=False, dtype=torch.float32)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0.0, std=self.in_features ** -0.5)

    def forward(self, features):
        if features.dtype != torch.float32 or features.shape[-1] != self.in_features:
            raise ValueError(f"head expects float32 features of width {self.in_features}")
        with _autocast_disabled(features.device):
            return super().forward(features).squeeze(-1)


def new_head(hidden_size: int, seed: int, device="cpu") -> ObservationHead:
    """Fresh head, drawn on CPU in FP32 under an isolated, purpose-keyed RNG."""
    with isolated_torch_rng(keyed_seed(seed, 0, 0, HEAD_INIT_PURPOSE)):
        head = ObservationHead(hidden_size)
    return head.to(device)


def head_optimizer(head: ObservationHead, config: FunctionalConfig, lr: float):
    if not isinstance(config, FunctionalConfig):
        raise TypeError("config must be FunctionalConfig")
    _finite_number(lr, "lr")
    if lr <= 0:
        raise ValueError("lr must be positive")
    return torch.optim.AdamW(head.parameters(), lr=lr * config.head_lr_scale,
                             weight_decay=config.head_weight_decay,
                             foreach=False, fused=False, capturable=False)


def clip_head_gradient(head: ObservationHead, config: FunctionalConfig) -> float:
    """Clip the head gradient norm; returns the norm before clipping."""
    return float(torch.nn.utils.clip_grad_norm_(
        list(head.parameters()), config.head_gradient_clip, error_if_nonfinite=True,
        foreach=False))


def require_dropout_disabled(policy) -> None:
    active = [name for name, module in policy.named_modules()
              if isinstance(module, nn.Dropout) and module.training and module.p > 0]
    if active:
        raise RuntimeError(
            "functional features need dropout disabled through forward and backward; wrap the "
            f"pair in runner.deterministic_ordering (active dropout: {active[:3]})")


def complete_view(construct, arm: str) -> VisibleView:
    """The complete (nothing masked) view of a construct for one context arm."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    sequences = {"H": construct.H}
    if construct.L is not None:
        sequences["L"] = construct.L
    if arm == "paired":
        sequences["Ag"] = construct.Ag
    view, _ = mask_complete(sequences)
    return view


def antibody_features(policy, provider: DynamicContextProvider, view: VisibleView) -> torch.Tensor:
    """Mean fused final hidden state over the antibody query's H/L residue slots (FP32, ``[H]``)."""
    if not isinstance(provider, DynamicContextProvider):
        raise TypeError("provider must be a DynamicContextProvider")
    if not isinstance(view, VisibleView):
        raise TypeError("view must be a VisibleView")
    if view.hidden_sites():
        raise ValueError("functional features need a complete view with no masked site")
    if provider.hidden_union:
        raise ValueError("functional features need a provider built from a complete view, not "
                         "a masked reconstruction provider")
    query = view.without_ag()
    expected = sum(len(query.chain(role)) for role in query.roles)
    _, hidden, mappings = policy.forward_views([query], [provider])
    mapping = mappings[0]
    slots = [mapping[coord] for coord in query.coords()]
    if len(slots) != expected or len(set(slots)) != len(slots):
        raise RuntimeError("antibody pooling must cover each H/L residue slot exactly once")
    feature = hidden[0, slots].mean(dim=0)
    if not bool(torch.isfinite(feature).all()):
        raise FloatingPointError("nonfinite antibody feature")
    return feature


def endpoint_features(policy, encoder, construct, arm: str, *, grad: bool) -> torch.Tensor:
    """Pooled feature of one construct with a fresh complete-view provider.

    ``grad`` False computes the feature without a graph and returns it detached; the
    values, the data and the draws are identical in both modes.
    """
    if type(grad) is not bool:
        raise TypeError("grad must be a bool")
    require_dropout_disabled(policy)
    view = complete_view(construct, arm)
    provider = DynamicContextProvider(encoder, view)
    with torch.set_grad_enabled(grad):
        feature = antibody_features(policy, provider, view)
    return feature if grad else feature.detach()


def preflight_complete_views(policy, encoder, constructs: Iterable) -> int:
    """Validate the complete paired and complete ab-only view of every construct.

    Both arms are always checked, so the admitted set is arm-independent and an over-long
    antigen is rejected even when only the ab-only arm would run.  Returns the number of
    views validated.
    """
    validated = 0
    for construct in constructs:
        for arm in ARMS:
            try:
                view = complete_view(construct, arm)
                encoder.validate_view(view, allow_unmasked=True)
                provider = DynamicContextProvider(encoder, view)
                policy.preflight([view.without_ag()], [provider])
            except ValueError as error:
                label = getattr(construct, "construct_id", "<unnamed construct>")
                raise ValueError(f"construct {label!r}, {arm} complete view: {error}") from error
            validated += 1
    return validated


def pair_loss(head: ObservationHead, better_feature: torch.Tensor, worse_feature: torch.Tensor,
              config: FunctionalConfig) -> Tuple[torch.Tensor, bool]:
    """``weight / pairs_per_update`` times the squared hinge on the head-score gap.

    Returns ``(loss, active)`` where ``active`` says the hinge was positive.
    """
    if not isinstance(head, ObservationHead):
        raise TypeError("head must be an ObservationHead")
    if not isinstance(config, FunctionalConfig):
        raise TypeError("config must be FunctionalConfig")
    for name, feature in (("better", better_feature), ("worse", worse_feature)):
        if (not isinstance(feature, torch.Tensor) or feature.dim() != 1
                or feature.shape[0] != head.in_features or feature.dtype != torch.float32
                or not bool(torch.isfinite(feature).all())):
            raise ValueError(f"{name} feature must be a finite float32 vector of width "
                             f"{head.in_features}")
    with _autocast_disabled(better_feature.device):
        scores = head(torch.stack([better_feature, worse_feature]))
        hinge = ordering_loss(scores, [(0, 1)], config.margin)
        loss = hinge * config.weight / config.pairs_per_update
        active = bool((config.margin - (scores[0] - scores[1])).detach() > 0)
    return loss, active
