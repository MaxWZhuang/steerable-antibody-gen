"""Bind prepared structural inputs to the existing constrained edit policy.

This is the seam, and it is deliberately thin: it builds a `ConstrainedEditSpace`
from the declared sites, encodes the packed coordinates once, and hands back both
alongside the permutation that connects the caller's site order to the policy's.
It supplies **no model and loads no weights** -- the model is an argument.

The site-order problem this exists to solve
-------------------------------------------
`ConstrainedEditSpace` sorts its sites by ascending position
(``esmif1_policy.py:368``), and every per-site vector the policy returns --
``site_log_probabilities``, ``ConstrainedSample.alleles``, the allele indices
``log_prob_from_logits`` expects -- follows that sorted order. A caller who
declared ``("site_b" at 5, "site_a" at 2)`` and then reads
``sample.alleles[0]`` as ``site_b`` is reading ``site_a``. Nothing raises: both
are valid allele indices. So the manifest's declared order is carried here
explicitly, and `BoundPolicy` converts between the two orders by name.

What binding checks
-------------------
`encode_geometry` accepts a geometry longer than the context -- upstream's
multichain route scores a short chain against complex-length coordinates -- and
`specs/esmif1_policy.md` records that *no residue correspondence between the two
is checked, because none is declared*. Here one **is** declared, so
`bind_policy` refuses unless the decoded chain occupies rows
``0 .. len(context) - 1`` of the packed array, which is the convention upstream's
own scoring relies on (``esm/inverse_folding/multichain_util.py:126-133``
multiplies a per-decoder-position loss by the *target* chain's coordinate mask).

It also re-checks the prepared inputs' own invariants before encoding
(`check_prepared_state`), rather than trusting that whatever produced the object
left it consistent: a `PreparedStructure` holds writeable numpy arrays, and the
encoder would accept a NaN in a chain row or a finite value in the pad without
complaint.

The declaration is never mutated by binding. `PreparedStructure` is frozen, and
the coordinate array is copied before it is handed to torch, so no downstream
tensor can alias it. The *model*, on the other hand, is modified by the policy's
own documented contract: `ConstrainedEditPolicy` freezes the encoder's parameters
with ``requires_grad_(False)``.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any, Sequence

from smallAntibodyGen.models.esmif1_policy import (
    ConstrainedEditPolicy,
    ConstrainedEditSpace,
    EditableSite,
    FixedGeometry,
)

from .declaration import (
    ManifestValidationError,
    StructureAdapterError,
    revalidate_manifest,
)
from .prepare import PreparedArtifactError, PreparedStructure, check_prepared_state


__all__ = ["BoundPolicy", "PolicyBindingError", "bind_policy", "build_edit_space"]


class PolicyBindingError(StructureAdapterError):
    """Prepared inputs and a supplied model could not be bound together."""


def build_edit_space(prepared: PreparedStructure) -> ConstrainedEditSpace:
    """The `ConstrainedEditSpace` for a prepared structure's declared sites.

    Sites are passed in the caller's declared order; the space sorts them itself.
    Allele order *within* a site is preserved exactly:
    ``allowed_residues[0]`` becomes allele index 0, because `EditableSite` takes
    allele indices from the caller's tuple (``esmif1_policy.py:257-261``).

    Raises:
        PolicyBindingError: If the manifest's parsed fields disagree with the
            canonical snapshot its digest is taken over -- the space would then be
            built from a declaration nobody hashed -- or if the policy refuses the
            space. The manifest validator already checks the same conditions with
            better messages, so reaching the latter is a bug in one of the two.
    """
    try:
        manifest = revalidate_manifest(prepared.manifest)
    except ManifestValidationError as error:
        raise PolicyBindingError(
            f"the declaration this space would be built from is not the one its digest "
            f"pins: {error}"
        ) from error
    try:
        return ConstrainedEditSpace(
            context=manifest.decoded_sequence,
            sites=tuple(
                EditableSite(site.sequence_index, site.allowed_residues)
                for site in manifest.sites
            ),
        )
    except ValueError as error:
        raise PolicyBindingError(
            f"the declared edit space was refused by the policy: {error}"
        ) from error


@dataclass(frozen=True, eq=False)
class BoundPolicy:
    """A policy, its encoded geometry, and the caller-to-policy site permutation.

    Attributes:
        prepared: The structural inputs this was built from.
        space: The edit space, with sites in ascending-position order.
        policy: The `ConstrainedEditPolicy` over `space` and the supplied model.
        geometry: The `FixedGeometry` encoded from the packed coordinates. Like
            every `FixedGeometry` it is **process-local**: bound to the model's
            device and dtype, and invalidated by moving or re-casting the model.
    """

    prepared: PreparedStructure
    space: ConstrainedEditSpace
    policy: ConstrainedEditPolicy
    geometry: FixedGeometry

    @property
    def declared_site_ids(self) -> tuple[str, ...]:
        """Site ids in the manifest's declared order."""
        return self.prepared.manifest.declared_site_ids

    @property
    def policy_site_ids(self) -> tuple[str, ...]:
        """Site ids in the order every per-site vector from `policy` uses."""
        return self.prepared.manifest.policy_site_ids

    @property
    def declared_to_policy(self) -> tuple[int, ...]:
        """Where each declared site sits in policy order."""
        return self.prepared.manifest.declared_to_policy

    @property
    def policy_to_declared(self) -> tuple[int, ...]:
        """Where each policy site sits in declared order."""
        return self.prepared.manifest.policy_to_declared

    def _checked(self, values: Sequence[Any], label: str) -> tuple[Any, ...]:
        items = tuple(values)
        if len(items) != len(self.policy_site_ids):
            raise PolicyBindingError(
                f"{label} has {len(items)} entries but there are "
                f"{len(self.policy_site_ids)} sites."
            )
        return items

    def to_declared_order(self, values: Sequence[Any]) -> tuple[Any, ...]:
        """Reorder one per-site vector from policy order into declared order."""
        items = self._checked(values, "per-site vector")
        return tuple(items[index] for index in self.declared_to_policy)

    def to_policy_order(self, values: Sequence[Any]) -> tuple[Any, ...]:
        """Reorder one per-site vector from declared order into policy order."""
        items = self._checked(values, "per-site vector")
        return tuple(items[index] for index in self.policy_to_declared)

    def alleles_by_site_id(self, alleles: Sequence[int]) -> dict[str, str]:
        """Name the residue each allele index selects, keyed by declared site id.

        Args:
            alleles: One allele index per site, **in policy order** -- what
                `ConstrainedSample.alleles` and `ConstrainedEditSpace.alleles_for`
                return.

        Returns:
            ``{site_id: residue}``. Reading this instead of indexing a positional
            vector is what makes the declared order impossible to get wrong.

        Raises:
            PolicyBindingError: On the wrong number of alleles, a value that is
                not an integer index (a ``bool`` and a ``float`` included), or an
                index outside ``{0, 1}``.
        """
        indices = self._checked(alleles, "allele vector")
        result: dict[str, str] = {}
        for site, site_id, value in zip(self.space.sites, self.policy_site_ids, indices):
            result[site_id] = site.alleles[_allele_index(value, site_id)]
        return result


def _allele_index(value: Any, site_id: str) -> int:
    """One allele index, required to be an actual integer in ``{0, 1}``.

    ``True == 1`` and ``0.0 == 0``, so a membership test accepts a bool and a
    float and then indexes a tuple with them -- the first silently names an
    allele, the second raises a bare ``TypeError`` from inside the tuple. Integral
    numpy and torch scalars are accepted, because those are what a sampled allele
    vector actually holds: ``operator.index`` is the same protocol a tuple
    subscript uses.
    """
    if isinstance(value, bool) or "bool" in str(getattr(value, "dtype", "")).lower():
        raise PolicyBindingError(
            f"allele index {value!r} for site {site_id!r} is a boolean, not an allele "
            "index; True would silently select allele 1."
        )
    try:
        index = operator.index(value)
    except TypeError as error:
        raise PolicyBindingError(
            f"allele index {value!r} for site {site_id!r} is not an integer "
            f"({type(value).__name__}); allele indices are exact, never rounded."
        ) from error
    if index not in (0, 1):
        raise PolicyBindingError(
            f"allele index {index!r} for site {site_id!r} is outside {{0, 1}}."
        )
    return index


def bind_policy(
    prepared: PreparedStructure, model: Any, alphabet: Any | None = None
) -> BoundPolicy:
    """Build a policy over a **supplied** model and encode the prepared geometry.

    Nothing here downloads or loads weights: ``model`` is whatever the caller
    already has, and the policy validates its alphabet against the native
    ESM-IF1 table.

    Args:
        prepared: Validated structural inputs.
        model: A `GVPTransformerModel`-shaped object.
        alphabet: Optional alphabet, validated against the native table.

    Returns:
        A :class:`BoundPolicy`.

    Raises:
        PolicyBindingError: If the decoded chain does not occupy rows
            ``0 .. len(context) - 1``, if the prepared inputs are not internally
            consistent *as they stand*, or if the policy or the encoder refuses
            them.
    """
    space = build_edit_space(prepared)
    expected_rows = tuple(range(len(space.context)))
    if prepared.decoded_rows != expected_rows:
        span = prepared.decoded_span
        raise PolicyBindingError(
            f"the decoded chain occupies rows {span.start_row}..{span.end_row - 1} of the "
            f"packed coordinates, but the policy's decision at position p is the decoded "
            f"chain's residue p, so it must occupy rows 0..{len(space.context) - 1}."
        )

    # Checked here, not assumed from wherever this object was built: `replace()`
    # and a writeable array are both enough to put NaN into a chain row or a
    # finite value into the pad, and the encoder would take either.
    try:
        check_prepared_state(prepared)
    except PreparedArtifactError as error:
        raise PolicyBindingError(
            f"the prepared inputs are not internally consistent: {error}"
        ) from error

    try:
        policy = ConstrainedEditPolicy(model, space, alphabet=alphabet)
    except (TypeError, ValueError) as error:
        raise PolicyBindingError(f"the supplied model was refused: {error}") from error

    try:
        geometry = policy.encode_geometry(
            # Copied so nothing downstream can alias -- and therefore mutate --
            # the prepared arrays: `torch.as_tensor` shares memory with a
            # matching-dtype CPU array.
            prepared.coordinates.copy(),
            prepared.confidence.copy(),
        )
    except ValueError as error:
        raise PolicyBindingError(
            f"the prepared coordinates were refused by the encoder: {error}"
        ) from error

    return BoundPolicy(prepared=prepared, space=space, policy=policy, geometry=geometry)
