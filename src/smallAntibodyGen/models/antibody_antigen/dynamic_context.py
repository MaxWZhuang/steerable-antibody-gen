"""Dynamic context provider: re-extracts frozen encoder features on every model call.

A :class:`DynamicContextProvider` wraps a :class:`FrozenESMPPI` encoder and an optional
immutable antigen sequence.  On each call to :meth:`~DynamicContextProvider.get_context`
it combines the caller's current antibody view with the fixed Ag (if any), runs the
encoder, and returns a fresh :class:`ExactViewFeatures`.

No per-mask cache is retained.  The encoder is always in eval/no_grad (guaranteed by
:class:`FrozenESMPPI`).  Immutable fields (Ag sequence, Ab roles, Ab chain lengths, and
the scaffold residues visible in the reference view) are fixed at construction and
validated on each call.  The provider never sees hidden gold residues, clean masked
tokens, or target labels.

Zero-hidden views (fully-revealed antibody states at the end of iterative decoding) are
accepted; the upstream :class:`ExactViewFeatures` does not require hidden sites.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from smallAntibodyGen.models.antibody_antigen.esm_ppi import FrozenESMPPI
from smallAntibodyGen.models.antibody_antigen.exact_view_features import ExactViewFeatures
from smallAntibodyGen.models.antibody_antigen.views import VisibleView, validate_residue

Coord = Tuple[str, int]


class DynamicContextProvider:
    """Produces per-step :class:`ExactViewFeatures` from the currently visible antibody view.

    Parameters
    ----------
    encoder:
        A frozen ESM-PPI encoder.  Must already carry its encoder_identity.
    view:
        A reference view that establishes Ab roles (H, optional L), chain lengths, and
        the immutable scaffold (visible positions).  If Ag is present its sequence is
        captured as the immutable antigen.  Masked positions in *view* define the editable
        union; their residues are **not** stored.
    """

    def __init__(self, encoder: FrozenESMPPI, view: VisibleView) -> None:
        if not isinstance(encoder, FrozenESMPPI):
            raise TypeError("encoder must be a FrozenESMPPI")
        if not isinstance(view, VisibleView):
            raise TypeError("view must be a VisibleView")
        if not view.has_role("H"):
            raise ValueError("reference view must include H")

        self._encoder = encoder
        self._encoder_identity: Dict[str, str] = dict(encoder.encoder_identity)
        self._hidden_union = view.hidden_sites()

        # Ab roles and chain lengths (H, optional L only).
        self._ab_roles: Tuple[str, ...] = tuple(r for r in view.roles if r != "Ag")
        self._ab_lengths: Tuple[int, ...] = tuple(
            len(view.chain(r)) for r in self._ab_roles
        )

        # Scaffold: visible (non-None) positions in the reference Ab view with their residues.
        self._scaffold: Dict[Coord, str] = {}
        for role in self._ab_roles:
            for i, token in enumerate(view.chain(role)):
                if token is not None:
                    self._scaffold[(role, i)] = token

        # Immutable Ag sequence (None = no antigen conditioning).
        if view.has_role("Ag"):
            ag_chain = view.chain("Ag")
            self._ag_residues: Optional[Tuple[str, ...]] = tuple(
                validate_residue(tok) for tok in ag_chain  # type: ignore[arg-type]
            )
        else:
            self._ag_residues = None

    @property
    def encoder_identity(self) -> Dict[str, str]:
        return dict(self._encoder_identity)

    @property
    def hidden_union(self):
        return self._hidden_union

    def check_query(self, ab_view: VisibleView) -> None:
        """Validate the current view without extracting features.

        *ab_view* must have the same Ab roles and chain lengths as the reference view,
        must not include Ag (the immutable Ag is appended internally when present), and
        must carry the same residues at all scaffold (originally-visible) positions.
        Zero-hidden views (fully revealed) are accepted.

        All validations run before any encoder forward.
        """
        if not isinstance(ab_view, VisibleView):
            raise TypeError("ab_view must be a VisibleView")
        if ab_view.has_role("Ag"):
            raise ValueError(
                "ab_view must not include Ag; the immutable antigen is managed by the provider"
            )
        if not ab_view.has_role("H"):
            raise ValueError("ab_view must include H")

        # Validate roles match the reference.
        actual_ab_roles = tuple(r for r in ab_view.roles if r != "Ag")
        if actual_ab_roles != self._ab_roles:
            raise ValueError(
                f"ab_view roles {actual_ab_roles} differ from the provider's "
                f"reference roles {self._ab_roles}"
            )

        # Validate chain lengths match.
        actual_lengths = tuple(len(ab_view.chain(r)) for r in self._ab_roles)
        if actual_lengths != self._ab_lengths:
            raise ValueError(
                f"ab_view chain lengths {actual_lengths} differ from the provider's "
                f"reference lengths {self._ab_lengths}"
            )

        # Validate immutable scaffold residues.
        for (role, i), residue in self._scaffold.items():
            actual = ab_view.token((role, i))
            if actual != residue:
                raise ValueError(
                    f"scaffold position {(role, i)!r} must be {residue!r}, "
                    f"got {actual!r}"
                )

        # Defensive check against encoder identity drift (should never happen at runtime).
        if self._encoder.encoder_identity != self._encoder_identity:
            raise ValueError(
                "encoder identity drift detected; provider cannot be reused"
            )

        # Build the combined view presented to the encoder.
        if self._ag_residues is not None:
            combined_view = VisibleView(
                ab_view.roles + ("Ag",),
                ab_view.chains + (self._ag_residues,),
            )
        else:
            combined_view = ab_view

        self._encoder.validate_view(combined_view, allow_unmasked=True)

    def get_context(self, ab_view: VisibleView) -> ExactViewFeatures:
        self.check_query(ab_view)
        combined = (VisibleView(ab_view.roles + ("Ag",),
                                ab_view.chains + (self._ag_residues,))
                    if self._ag_residues is not None else ab_view)
        return self._encoder.extract_exact(combined)
