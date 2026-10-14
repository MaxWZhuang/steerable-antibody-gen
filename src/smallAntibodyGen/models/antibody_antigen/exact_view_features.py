"""Exact-view feature holder for the dynamic generation and scoring path.

Unlike :class:`~smallAntibodyGen.models.antibody_antigen.static_context.StaticContext`,
this object:

* accepts zero-hidden views (fully-revealed states encountered during iterative decoding),
* is device-resident — features stay on the extraction device with no CPU round-trip,
* carries no expensive byte hash (not a persistent cache),
* requires an **exact Ab-portion match** for every query (not union-subset semantics).

Only :mod:`views` is imported here so that :mod:`fusion` and :mod:`dynamic_context` can
both import this module without creating a cycle.
"""
from __future__ import annotations

import hashlib
import json
from typing import Dict, Tuple

import torch

from smallAntibodyGen.models.antibody_antigen.views import VisibleView

Coord = Tuple[str, int]


class ExactViewFeatures:
    """Per-residue encoder features bound to a single exact :class:`VisibleView`.

    Parameters
    ----------
    view:
        The exact view these features were extracted for.  May have zero hidden
        sites (fully-revealed antibody state).
    features:
        ``[len(view.coords()), D]`` floating-point tensor; may reside on any device.
        A detached clone is stored; the caller's tensor is not held.
    encoder_identity:
        The encoder identity dict as returned by the encoder.
    """

    def __init__(
        self,
        view: VisibleView,
        features: torch.Tensor,
        encoder_identity: Dict[str, str],
    ) -> None:
        if not isinstance(view, VisibleView):
            raise TypeError("view must be a VisibleView")
        if not isinstance(features, torch.Tensor):
            raise TypeError("features must be a torch.Tensor")
        if not features.is_floating_point():
            raise ValueError("features must be a floating-point tensor")
        n = len(view.coords())
        if features.dim() != 2 or features.shape[0] != n or features.shape[1] == 0:
            raise ValueError(
                f"features must have shape [{n}, D], got {tuple(features.shape)}"
            )
        if not bool(torch.isfinite(features).all()):
            raise ValueError("features must all be finite")
        if not isinstance(encoder_identity, dict):
            raise TypeError("encoder_identity must be a dict")

        self._view = view
        # Detached clone; stays on whatever device the caller supplied.
        with torch.inference_mode(False), torch.no_grad():
            self._features = features.detach().clone()
        self._encoder_identity: Dict[str, str] = dict(encoder_identity)

        payload = json.dumps(
            {
                "view": view.content_digest(),
                "encoder_identity": dict(sorted(encoder_identity.items())),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        self._content_key: str = hashlib.sha256(payload.encode("ascii")).hexdigest()

    # --- read-only accessors ---

    @property
    def encoder_identity(self) -> Dict[str, str]:
        return dict(self._encoder_identity)

    @property
    def content_key(self) -> str:
        return self._content_key

    @property
    def features(self) -> torch.Tensor:
        """Cloned copy of all feature rows (device-resident)."""
        return self._features.clone()

    # --- feature extraction ---

    def ab_features(self) -> torch.Tensor:
        """Rows for H (and L if present) in canonical ``(role, index)`` order.

        Antigen rows are excluded even when the stored view includes Ag context.
        """
        ab_rows = [
            i for i, (role, _) in enumerate(self._view.coords()) if role != "Ag"
        ]
        return self._features[ab_rows].clone()

    # --- query validation ---

    def check_query(self, query: VisibleView) -> None:
        """Raise unless *query*'s Ab portion exactly matches the stored view's Ab portion.

        Unlike :meth:`StaticContext.check_query` (which allows subsets of the hidden
        union), this requires role-for-role and chain-for-chain equality on the Ab
        chains.  Ag may be absent from the query even when the stored view contains it.
        """
        if not isinstance(query, VisibleView):
            raise TypeError("query must be a VisibleView")
        stored_ab_roles = tuple(r for r in self._view.roles if r != "Ag")
        query_ab_roles = tuple(r for r in query.roles if r != "Ag")
        if stored_ab_roles != query_ab_roles:
            raise ValueError(
                f"Ab roles mismatch: stored {stored_ab_roles}, query {query_ab_roles}"
            )
        for role in stored_ab_roles:
            if self._view.chain(role) != query.chain(role):
                raise ValueError(
                    f"Ab chain {role!r} content mismatch; "
                    "exact-view features are bound to a single view"
                )
        if query.has_role("Ag") and (not self._view.has_role("Ag") or
                                      query.chain("Ag") != self._view.chain("Ag")):
            raise ValueError("antigen differs from the exact feature view")

    def query_ab_features(self, query: VisibleView) -> torch.Tensor:
        """Validate exact Ab-portion match then return Ab-only feature rows."""
        self.check_query(query)
        return self.ab_features()
