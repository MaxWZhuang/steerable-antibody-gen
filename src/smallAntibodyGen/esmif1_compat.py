"""Make the archived `fair-esm` 2.0.0 inverse-folding stack importable here.

ESM-IF1 (`esm_if1_gvp4_t16_142M_UR50`) is the backbone selected for the
fixed-target CR9114/H1 experiment (`specs/decisions/0003-pretrained-conditioned-policy.md`
§"Scope clarification -- 2026-09-14"). Its upstream repository is archived at a
2022 dependency set and will not import on this repo's stack without three
unrelated patches. Each one is applied here, and each is skipped when it is not
needed, so this module is a no-op on a machine where upstream already works.

1. **`torch_scatter`.** `esm/inverse_folding/gvp_modules.py:34` imports
   `scatter_add` and `scatter` from it. PyPI ships torch-scatter 2.1.2 as an
   **sdist only** -- no wheel for any platform -- and building it needs an MSVC
   newer than the VS2017 toolchain on the training box (torch 2.11's headers use
   `__builtin_LINE`, which MSVC 14.16 lacks). The entire surface ESM uses is one
   call at `gvp_modules.py:452`, counting node in-degree; `scatter` is a dead
   import. `scatter_sum` below is torch_scatter's own composite implementation
   expressed with `Tensor.scatter_add_`.

2. **biotite 1.x renames.** `esm/inverse_folding/util.py` imports
   `biotite.structure.filter_backbone` and uses `pdbx.PDBxFile`, both renamed in
   biotite 1.0. Pinning biotite <1.0 would drag numpy from 2.4.3 down to 1.26.4
   (measured via `pip install --dry-run`), an environment-wide change affecting
   every package in the venv; it was avoided for that reason, not because it was
   shown to break anything. The imports are at module scope, so the package will
   not import even if the structure loader is never called.

3. **`torch.load` safe globals.** torch >= 2.6 defaults `weights_only=True`, and
   the ESM-IF1 checkpoint pickles an `argparse.Namespace` under `"args"`. This is
   torch's own sanctioned mechanism for a trusted legacy checkpoint, not a patch
   to upstream.

**Why a substitute rather than editing the installed package.** The promotion
gate requires pinning and hashing the exact upstream artifact. Nothing here
writes to `site-packages`, so the installed `esm` stays byte-identical to the
released wheel and that hash keeps meaning something.

**Known limitation.** `torch_scatter` cannot be built on this box, so the
substitute has never been run side by side against the genuine extension here.
Its correctness rests on `bincount` being a complete specification of the single
call site, which the tests pin, not on a same-machine A/B.

Nothing in the existing training chain imports this module; it is the boundary
the migration specification asks for, and it is inert until a caller invokes
`install()`.

Typical use, before the first `esm.inverse_folding` import in a process::

    from smallAntibodyGen.esmif1_compat import install
    install()
    import esm
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from dataclasses import dataclass

import torch


__all__ = [
    "CompatReport",
    "alias_biotite_renames",
    "allow_esm_checkpoint_globals",
    "install",
    "install_scatter_substitute",
    "scatter",
    "scatter_add",
    "scatter_sum",
]


# --------------------------------------------------------------------------
# The one upstream function ESM actually calls
# --------------------------------------------------------------------------

def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int) -> torch.Tensor:
    """Expand ``src`` to ``other``'s shape for a scatter along ``dim``.

    Mirrors ``torch_scatter.utils.broadcast`` so the substitute accepts the same
    argument shapes as the package it stands in for.

    Args:
        src: Index tensor, typically 1-D.
        other: The tensor whose shape ``src`` must match.
        dim: Dimension the scatter runs along; may be negative.

    Returns:
        A view of ``src`` expanded to ``other.size()``.
    """
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    return src.expand(other.size())


def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: torch.Tensor | None = None,
    dim_size: int | None = None,
) -> torch.Tensor:
    """Sum ``src`` into buckets given by ``index`` along ``dim``.

    Behavioural stand-in for ``torch_scatter.scatter_add``. Differentiable with
    respect to ``src``; the gradient of ``src[i]`` is the upstream gradient of
    the bucket it landed in.

    Args:
        src: Values to accumulate.
        index: Bucket index per element, broadcast against ``src``.
        dim: Dimension to scatter along.
        out: Optional pre-allocated destination, accumulated into in place.
        dim_size: Size of ``dim`` in the result. When omitted it is inferred as
            ``index.max() + 1``, which drops trailing empty buckets -- pass it
            explicitly whenever the bucket count is known.

    Returns:
        The accumulated tensor.
    """
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
    return out.scatter_add_(dim, index, src)


scatter_add = scatter_sum


def scatter(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: torch.Tensor | None = None,
    dim_size: int | None = None,
    reduce: str = "sum",
) -> torch.Tensor:
    """Dispatch to a reduction, supporting only the sum ESM would ever need.

    `esm.inverse_folding` imports this name but never calls it. Anything other
    than a sum raises rather than guessing at semantics that were never
    exercised -- a wrong reduction here would corrupt the encoder silently.

    Raises:
        NotImplementedError: For any ``reduce`` other than ``"sum"``/``"add"``.
    """
    if reduce in ("sum", "add"):
        return scatter_sum(src, index, dim, out, dim_size)
    raise NotImplementedError(
        f"scatter(reduce={reduce!r}) is not implemented: esm.inverse_folding "
        "never calls it, so its semantics here were never verified. Implement "
        "it deliberately, with a test, rather than inferring the behaviour."
    )


# --------------------------------------------------------------------------
# Installation
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CompatReport:
    """What `install` actually did, for logging into a readiness record.

    Attributes:
        scatter: Outcome of the `torch_scatter` step.
        biotite_aliases: Names aliased onto biotite; empty on biotite 0.x.
        checkpoint_globals: Outcome of the `torch.load` safe-globals step.
    """

    scatter: str
    biotite_aliases: tuple[str, ...]
    checkpoint_globals: str


def install_scatter_substitute() -> str:
    """Register this module's scatter functions under the name `torch_scatter`.

    Defers to a genuine compiled `torch_scatter` whenever one is importable, so a
    machine that has the real extension keeps using it. Idempotent.

    Returns:
        A short description of what happened, for `CompatReport.scatter`.
    """
    if "torch_scatter" in sys.modules:
        existing = sys.modules["torch_scatter"]
        if getattr(existing, "__name__", None) == __name__ + ".torch_scatter":
            return "already installed"
        return "skipped: upstream torch_scatter is present"
    if importlib.util.find_spec("torch_scatter") is not None:
        return "skipped: upstream torch_scatter is present"

    module = types.ModuleType(__name__ + ".torch_scatter")
    module.__doc__ = (
        "Substitute for the torch_scatter extension, provided by "
        f"{__name__}. Only the sum reduction is implemented."
    )
    module.broadcast = broadcast
    module.scatter_sum = scatter_sum
    module.scatter_add = scatter_add
    module.scatter = scatter
    sys.modules["torch_scatter"] = module
    return "installed"


def alias_biotite_renames() -> tuple[str, ...]:
    """Restore the biotite 0.x names `esm.inverse_folding.util` imports.

    Only fills in names that are absent, so a biotite 0.x install keeps its own
    `filter_backbone` rather than having the 1.x atom selection swapped in
    underneath it.

    Returns:
        The names aliased, in application order. Empty when nothing was needed.

    Raises:
        ModuleNotFoundError: If biotite is not installed at all.
    """
    import biotite.structure as bs
    from biotite.structure.io import pdbx

    aliased: list[str] = []
    # biotite 1.0 renamed this to disambiguate peptide from nucleotide backbones.
    # It is a coarse pre-filter: ESM re-selects N/CA/C explicitly downstream.
    if not hasattr(bs, "filter_backbone"):
        bs.filter_backbone = bs.filter_peptide_backbone
        aliased.append("filter_backbone")
    # biotite 1.0 replaced the PDBx reader class with CIFFile.
    if not hasattr(pdbx, "PDBxFile"):
        pdbx.PDBxFile = pdbx.CIFFile
        aliased.append("PDBxFile")
    return tuple(aliased)


def allow_esm_checkpoint_globals() -> str:
    """Permit the `argparse.Namespace` the ESM-IF1 checkpoint pickles.

    torch >= 2.6 loads with `weights_only=True`, which refuses arbitrary pickled
    classes. Exactly one class is allowed here, rather than disabling the check.

    Returns:
        A short description of what happened, for `CompatReport.checkpoint_globals`.
    """
    add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
    if add_safe_globals is None:
        return "unsupported: torch has no add_safe_globals"
    get_safe_globals = getattr(torch.serialization, "get_safe_globals", None)
    if get_safe_globals is not None and argparse.Namespace in get_safe_globals():
        return "already allowed"
    add_safe_globals([argparse.Namespace])
    return "allowed"


def install() -> CompatReport:
    """Apply every patch `esm.inverse_folding` needs on this stack.

    Idempotent, and a no-op for any step that is unnecessary. Call once before
    the first `esm.inverse_folding` import in a process.

    Returns:
        A `CompatReport` describing each step, suitable for recording alongside
        a readiness measurement.

    Raises:
        ModuleNotFoundError: If biotite is not installed; `esm.inverse_folding`
            could not import without it, so failing here names the missing
            dependency instead of surfacing it as an opaque ImportError later.
    """
    return CompatReport(
        scatter=install_scatter_substitute(),
        biotite_aliases=alias_biotite_renames(),
        checkpoint_globals=allow_esm_checkpoint_globals(),
    )
