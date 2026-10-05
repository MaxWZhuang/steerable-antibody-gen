"""Which operations a route can honestly perform, and the refusal when it cannot.

Torch-free and pure, so the services can call it before they build, load or write anything.

The two likelihood routes (AbLang2 and p-IgGen, and the optional DPLM routes) supply a normalised
per-token conditional: every generic operation in the flight -- the multi-task curriculum, the
masked-NLL training loop, the preference arms, the teacher distillation, the local-ordering and
retention panels -- is written over that conditional. The energy route supplies a smoothed energy
``E(y, context)`` and a walk-jump sampler. It has no token interface, so a generic operation asked
of it would fail deep inside a loader or a collator, after a model, an optimizer or an output
directory already existed, or worse would be answered with a number the route cannot produce.

:func:`require_token_likelihood` is the single early refusal. It raises
:class:`RouteCapabilityError`, a :class:`~.her2_routes_lineage.LineageError`, so a campaign stage
that reaches it records ``blocked_readiness`` with the reason instead of a crash. The energy route's
own training (``train_energy``), scoring (``score_panel`` / ``energy_panel_scores``) and sampling
(``energy_generate``) are not refused.
"""
from __future__ import annotations

from .her2_routes_lineage import LineageError

#: The energy route's name. ``her2_routes_services.ROUTE_ENERGY`` is the same string; a test
#: pins the two together, because this module cannot import the services (they import it).
ENERGY_ROUTE = "dwjs_energy"

#: Routes with no normalised per-token conditional.
TOKEN_LIKELIHOOD_FREE_ROUTES = frozenset({ENERGY_ROUTE})

#: What each such route does supply, stated so the refusal can name the supported alternative.
ENERGY_SUPPORTED = ("train_energy (denoiser, then the frozen-denoiser energy objective)",
                    "score_panel / energy_panel_scores (negative energy of the clean window)",
                    "energy_generate (walk, then jump)")


class RouteCapabilityError(LineageError):
    """The route cannot perform the requested operation; nothing was built, loaded or written."""


def supports_token_likelihood(route):
    """Does ``route`` supply a normalised per-token conditional?"""
    return str(route) not in TOKEN_LIKELIHOOD_FREE_ROUTES


def capabilities(route):
    """A JSON-able record of what ``route`` can do, for stage records and tests."""
    token = supports_token_likelihood(route)
    return {"route": str(route), "token_likelihood": token,
            "energy_scoring": not token, "walk_jump_sampling": not token,
            "supported_here": list(ENERGY_SUPPORTED) if not token else None}


def require_token_likelihood(route, operation):
    """Refuse ``operation`` on a route that has no token likelihood, before any work is done.

    ``operation`` names the caller (``"FlightServices.train"``) so the message says what was
    asked. Returns ``True`` for a route that does support it.
    """
    if supports_token_likelihood(route):
        return True
    raise RouteCapabilityError(
        f"{operation}: the route {route!r} cannot do this. It supplies a smoothed energy and a "
        "walk-jump sampler, not a normalised per-token conditional, and this operation is defined "
        "over one (token masking, a sequential conditional, a preference margin or a teacher "
        "distillation). It was refused before any bank, model, optimizer or output directory was "
        "created, and no likelihood metric is invented for the route. What this route supports: "
        + "; ".join(ENERGY_SUPPORTED) + ".")
