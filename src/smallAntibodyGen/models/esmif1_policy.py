"""Constrained fixed-geometry editing policy over ESM-IF1's native decoder.

Implements the probability contract in
`reference/fixed-target-posttraining-recommendation.md` §"Exact probability
contract for the finite benchmark" for the CR9114/H1 finite benchmark, under the
[Decision 0003 clarification](../../../specs/decisions/0003-pretrained-conditioned-policy.md).
`specs/esmif1_policy.md` owns the full semantics, limitations, and pending gates;
this docstring states the contract the code implements.

For a declared fixed backbone geometry ``C`` and a context sequence in which
every residue is immutable except a declared set of editable sites, each with
exactly two allowed canonical residues::

    log q(y | C) = sum over editable sites t:
        log_softmax(logits(prefix_t, C)[allowed_t])[chosen_t]

The prefix at every decision is ESM-IF1's own autoregressive prefix -- ``<cath>``
followed by residues ``0 .. p-1`` of the *candidate*, so every forced residue
before ``p`` is present. Forced residues contribute probability one: they never
appear as a term in the sum. This is a **normalized constrained editing
process**, not the native sequence likelihood and not the native model
conditioned probabilistically on the fixed residues, and `log q` values are only
comparable within one `ConstrainedEditSpace` and one geometry.

Three properties are load-bearing and are pinned by the tests:

- **Decision index.** Upstream scores with ``prev = tokens[:, :-1]`` against
  ``target = tokens[:, 1:]`` (`esm/inverse_folding/util.py:113-118`) and returns
  logits as ``B x V x T`` (`transformer_decoder.py:125`). With the
  ``invariant_gvp`` alphabet ``append_eos`` is False, so ``tokens`` is
  ``<cath>, r_0 .. r_{L-1}`` and the decision for residue ``p`` is
  ``logits[:, :, p]``.
- **Caching the encoding is exact.** The encoder consumes coordinates,
  a padding mask, and confidence only; its token input is a constant
  ``<mask>``/``<pad>`` pattern derived from the padding mask
  (`gvp_transformer_encoder.py:83-87`). It never sees the sequence, so a
  detached encoding reused across candidates is identical to recomputing it --
  a fact, not an approximation. `encode_geometry` still runs the real encoder.
- **Skipping a forced step corrupts the cache.** Under ``incremental_state`` the
  decoder ingests only the last token (`transformer_decoder.py:162-165`) and the
  attention module appends exactly that one key/value
  (`esm/multihead_attention.py:286-323`). Upstream's `sample` skips the decoder
  call at an already-known position (`gvp_transformer.py:122-124`), so the token
  before it is never ingested and later steps attend over a key/value set with
  holes. `sample` here calls the decoder at **every** prefix position and only
  *samples* at editable ones, so its log-probabilities match teacher forcing.

Nothing here loads weights, declares a structure, or maps benchmark sites onto
real residue indices. Allele index order is the caller's tuple order; no
germline/somatic convention is imposed. The module imports only torch, so the
optional `esm-if1` extra is needed to *use* a real backbone, never to import
this file.
"""

from __future__ import annotations

import hashlib
import itertools
import operator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


__all__ = [
    "CANONICAL_RESIDUES",
    "ESMIF1_TOKENS",
    "ESMIF1Alphabet",
    "ConstrainedEditPolicy",
    "ConstrainedEditSpace",
    "ConstrainedSample",
    "EditableSite",
    "FixedGeometry",
    "PREFIX_TOKEN",
]


#: The prefix token ESM-IF1 decodes from. `CoordBatchConverter` overwrites the
#: alphabet's `cls_idx` with it (`esm/inverse_folding/util.py:233`), so it is the
#: only prefix any upstream inverse-folding path emits. `<af2>` exists in the
#: table but no upstream code selects it.
PREFIX_TOKEN = "<cath>"

#: The 20 canonical amino acids, in alphabetical order. Editable-site supports
#: must be drawn from this set; the context may additionally contain any other
#: standard token of the native alphabet.
CANONICAL_RESIDUES = "ACDEFGHIKLMNPQRSTVWY"

# Reproduces `esm.data.Alphabet.from_architecture("invariant_gvp")` exactly:
# the same prepend/standard/append groups (`esm/data.py:165-171`,
# `esm/constants.py:8`) assembled by the same `<null_n>` padding rule
# (`esm/data.py:108-112`). Replicated rather than imported so that importing this
# module never requires the optional `esm-if1` extra; `ESMIF1Alphabet.matching`
# rejects any upstream alphabet that disagrees with it.
_PREPEND_TOKENS = ("<null_0>", "<pad>", "<eos>", "<unk>")
_STANDARD_TOKENS = (
    "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K", "Q", "N",
    "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z", "O", ".", "-",
)
_APPEND_TOKENS = ("<mask>", "<cath>", "<af2>")


def _build_token_table() -> tuple[str, ...]:
    """Assemble the native token table with upstream's `<null_n>` padding rule."""
    toks = list(_PREPEND_TOKENS) + list(_STANDARD_TOKENS)
    for i in range((8 - (len(toks) % 8)) % 8):
        toks.append(f"<null_{i + 1}>")
    toks.extend(_APPEND_TOKENS)
    return tuple(toks)


#: The native ESM-IF1 token table: 35 entries, canonical residues at 4..23,
#: `<mask>` at 32, `<cath>` at 33, `<af2>` at 34.
ESMIF1_TOKENS = _build_token_table()


# --------------------------------------------------------------------------
# Alphabet
# --------------------------------------------------------------------------

class ESMIF1Alphabet:
    """The native ESM-IF1 token table, without importing `esm`.

    Attributes:
        all_toks: The 35 native tokens, in index order.
        tok_to_idx: Token -> index.
        padding_idx, eos_idx, unk_idx, mask_idx, prefix_idx: Native indices.
            `prefix_idx` is `<cath>`; there is deliberately no `cls_idx`,
            because upstream's `<cls>` lookup falls through to `<unk>` and is
            then overwritten by `CoordBatchConverter`.

    Raises:
        ValueError: If `all_toks` is not exactly the native table.
    """

    def __init__(self, all_toks: Sequence[str] = ESMIF1_TOKENS) -> None:
        toks = tuple(all_toks)
        if toks != ESMIF1_TOKENS:
            raise ValueError(
                "token table does not match the native ESM-IF1 alphabet "
                f"(expected {len(ESMIF1_TOKENS)} tokens, got {len(toks)}); "
                "scoring against a different vocabulary would silently change "
                "every logit index. Expected "
                f"{ESMIF1_TOKENS!r}, got {toks!r}."
            )
        self.all_toks = toks
        self.tok_to_idx = {tok: i for i, tok in enumerate(toks)}
        self.padding_idx = self.tok_to_idx["<pad>"]
        self.eos_idx = self.tok_to_idx["<eos>"]
        self.unk_idx = self.tok_to_idx["<unk>"]
        self.mask_idx = self.tok_to_idx["<mask>"]
        self.prefix_idx = self.tok_to_idx[PREFIX_TOKEN]

    def __len__(self) -> int:
        return len(self.all_toks)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ESMIF1Alphabet(<{len(self.all_toks)} native tokens>)"

    @classmethod
    def native(cls) -> "ESMIF1Alphabet":
        """The native table, built locally."""
        return cls(ESMIF1_TOKENS)

    @classmethod
    def matching(cls, alphabet: Any) -> "ESMIF1Alphabet":
        """Accept an upstream `esm.data.Alphabet` only if it is the native one.

        Args:
            alphabet: An `ESMIF1Alphabet` (returned unchanged) or any object
                exposing `all_toks` plus the derived native indices.

        Returns:
            An `ESMIF1Alphabet`.

        Raises:
            ValueError: If the table or any derived index disagrees. A near-miss
                alphabet is the failure mode that would corrupt every score
                without raising, so it is refused rather than coerced.
        """
        if isinstance(alphabet, cls):
            return alphabet
        toks = getattr(alphabet, "all_toks", None)
        if toks is None:
            raise ValueError(
                "alphabet has no `all_toks`; expected an `esm.data.Alphabet` "
                "built by `from_architecture('invariant_gvp')` or an "
                "`ESMIF1Alphabet`."
            )
        native = cls(toks)
        for token, attribute in (
            ("<pad>", "padding_idx"),
            ("<eos>", "eos_idx"),
            ("<unk>", "unk_idx"),
            ("<mask>", "mask_idx"),
        ):
            observed = getattr(alphabet, attribute, None)
            if observed is not None and int(observed) != native.tok_to_idx[token]:
                raise ValueError(
                    f"alphabet.{attribute} is {int(observed)}, but {token!r} is at "
                    f"index {native.tok_to_idx[token]} in the native table."
                )
        return native

    def index_of(self, token: str) -> int:
        """Native index of `token`.

        Raises:
            ValueError: For a token outside the table. Unlike upstream's
                `get_idx`, this never silently returns `<unk>`.
        """
        try:
            return self.tok_to_idx[token]
        except KeyError:
            raise ValueError(
                f"{token!r} is not a native ESM-IF1 token; it would be encoded "
                "as <unk>, which is not a residue."
            ) from None

    def token_at(self, index: int) -> str:
        """Token at a native index.

        Raises:
            ValueError: If `index` is outside the table.
        """
        position = operator.index(index)
        if not 0 <= position < len(self.all_toks):
            raise ValueError(
                f"token index {position} is outside the native table "
                f"(0..{len(self.all_toks) - 1})."
            )
        return self.all_toks[position]

    def encode(self, sequence: str) -> list[int]:
        """Encode residue characters, with no prefix and no `<eos>`.

        Raises:
            ValueError: For any character outside the native table.
        """
        return [self.index_of(char) for char in sequence]


# --------------------------------------------------------------------------
# The edit space
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class EditableSite:
    """One editable position with exactly two allowed canonical residues.

    Attributes:
        position: 0-based index into `ConstrainedEditSpace.context`. This is a
            *sequence* index, not a benchmark residue number and not a PDB
            number; no mapping between them is implied here.
        alleles: The two allowed residues. Order defines allele indices 0 and 1
            for this site and is the caller's choice -- no germline/somatic
            convention is imposed.
    """

    position: int
    alleles: tuple[str, str]

    def __post_init__(self) -> None:
        try:
            position = operator.index(self.position)
        except TypeError:
            raise ValueError(
                f"site position must be an integer, got {self.position!r}."
            ) from None
        if position < 0:
            raise ValueError(f"site position must be non-negative, got {position}.")
        object.__setattr__(self, "position", position)

        if isinstance(self.alleles, str):
            raise ValueError(
                "alleles must be a pair of residues, not the string "
                f"{self.alleles!r}."
            )
        alleles = tuple(self.alleles)
        if len(alleles) != 2:
            raise ValueError(
                f"site {position} needs exactly two allowed residues, got "
                f"{len(alleles)}: {alleles!r}. The declared benchmark support is "
                "binary; a different arity needs a deliberate contract change."
            )
        for allele in alleles:
            # `allele in CANONICAL_RESIDUES` is *substring* membership: "" and
            # "AC" both satisfy it. Either one would be written straight into the
            # sequence by `sequence_for`, changing its length at that site, so the
            # single-character requirement is explicit.
            if (
                not isinstance(allele, str)
                or len(allele) != 1
                or allele not in CANONICAL_RESIDUES
            ):
                raise ValueError(
                    f"site {position} allele {allele!r} is not one of the "
                    f"canonical residues {CANONICAL_RESIDUES!r}; each allele must "
                    "be exactly one canonical residue character."
                )
        if alleles[0] == alleles[1]:
            raise ValueError(
                f"site {position} has duplicate alleles {alleles!r}; its support "
                "would be a single residue with probability one."
            )
        object.__setattr__(self, "alleles", alleles)

    def index_of(self, residue: str) -> int:
        """Allele index of `residue` at this site.

        Raises:
            ValueError: If `residue` is outside this site's support.
        """
        if residue == self.alleles[0]:
            return 0
        if residue == self.alleles[1]:
            return 1
        raise ValueError(
            f"residue {residue!r} is outside the support {self.alleles!r} of "
            f"site {self.position}."
        )


@dataclass(frozen=True)
class ConstrainedEditSpace:
    """A fixed context plus the editable sites carved out of it.

    Every position not named by a site is immutable and is forced into the
    decoding prefix. The context itself must be a member of the space -- its
    residue at each site must be one of that site's two alleles -- otherwise the
    parent sequence would be unscorable under the policy.

    Attributes:
        context: The full sequence, including the context residue at each
            editable site.
        sites: Editable sites, stored sorted by ascending position.
    """

    context: str
    sites: tuple[EditableSite, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.context, str) or not self.context:
            raise ValueError("context must be a non-empty sequence string.")
        # Restricted to the native standard tokens so the context encodes without
        # falling through to <unk>. Editable supports are further restricted to
        # the canonical 20 by `EditableSite`.
        for index, char in enumerate(self.context):
            if char not in _STANDARD_TOKENS:
                raise ValueError(
                    f"context position {index} holds {char!r}, which is not a "
                    "standard ESM-IF1 residue token and would encode as <unk>."
                )

        sites = tuple(self.sites)
        if not sites:
            raise ValueError(
                "at least one editable site is required; a space with no sites "
                "defines a point distribution, not an editing policy."
            )
        for site in sites:
            if not isinstance(site, EditableSite):
                raise ValueError(f"expected EditableSite instances, got {site!r}.")
        sites = tuple(sorted(sites, key=lambda site: site.position))

        seen: set[int] = set()
        for site in sites:
            if site.position in seen:
                raise ValueError(f"duplicate editable site at position {site.position}.")
            seen.add(site.position)
            if site.position >= len(self.context):
                raise ValueError(
                    f"editable site at position {site.position} is outside the "
                    f"context of length {len(self.context)}."
                )
            if self.context[site.position] not in site.alleles:
                raise ValueError(
                    f"context residue {self.context[site.position]!r} at position "
                    f"{site.position} is outside that site's support "
                    f"{site.alleles!r}; the context must itself be a member of "
                    "the space."
                )
        object.__setattr__(self, "sites", sites)

    @property
    def num_sites(self) -> int:
        """Number of editable sites."""
        return len(self.sites)

    @property
    def positions(self) -> tuple[int, ...]:
        """Editable positions, ascending."""
        return tuple(site.position for site in self.sites)

    @property
    def size(self) -> int:
        """Number of genotypes in the space, ``2 ** num_sites``."""
        return 2 ** self.num_sites

    @property
    def context_alleles(self) -> tuple[int, ...]:
        """Allele indices of the context sequence itself."""
        return tuple(
            site.index_of(self.context[site.position]) for site in self.sites
        )

    def sequence_for(self, alleles: Sequence[int]) -> str:
        """Materialize the sequence for a genotype.

        Args:
            alleles: One allele index (0 or 1) per site, in site order.

        Returns:
            The full sequence.

        Raises:
            ValueError: On the wrong number of alleles or an index outside {0, 1}.
        """
        chosen = tuple(alleles)
        if len(chosen) != self.num_sites:
            raise ValueError(
                f"expected {self.num_sites} allele indices, got {len(chosen)}."
            )
        residues = list(self.context)
        for site, allele in zip(self.sites, chosen):
            try:
                index = operator.index(allele)
            except TypeError:
                raise ValueError(
                    f"allele index {allele!r} for site {site.position} is not an "
                    "integer."
                ) from None
            if index not in (0, 1):
                raise ValueError(
                    f"allele index {index} for site {site.position} is outside "
                    "{0, 1}."
                )
            residues[site.position] = site.alleles[index]
        return "".join(residues)

    def alleles_for(self, sequence: str) -> tuple[int, ...]:
        """Read a candidate's genotype, validating every position.

        Args:
            sequence: A candidate sequence.

        Returns:
            One allele index per site, in site order.

        Raises:
            ValueError: If the length differs from the context, an immutable
                residue changed, or an editable position holds a residue outside
                that site's support.
        """
        if not isinstance(sequence, str):
            raise ValueError(f"candidate must be a string, got {type(sequence)!r}.")
        if len(sequence) != len(self.context):
            raise ValueError(
                f"candidate length {len(sequence)} does not match the context "
                f"length {len(self.context)}; this policy edits in place and "
                "never changes length."
            )
        editable = {site.position: site for site in self.sites}
        for index, (candidate, fixed) in enumerate(zip(sequence, self.context)):
            if index in editable:
                continue
            if candidate != fixed:
                raise ValueError(
                    f"immutable position {index} changed from {fixed!r} to "
                    f"{candidate!r}; only declared editable sites may differ."
                )
        return tuple(
            site.index_of(sequence[site.position]) for site in self.sites
        )

    def validate_candidate(self, sequence: str) -> str:
        """Return `sequence` if it is a member of the space, else raise."""
        self.alleles_for(sequence)
        return sequence

    def enumerate_alleles(self, max_sequences: int = 65_536) -> tuple[tuple[int, ...], ...]:
        """Every genotype, last site varying fastest.

        Raises:
            ValueError: If the space is larger than `max_sequences`. Enumeration
                is a toy/verification device; guarding it keeps a 16-site space
                from silently materializing a large batch.
        """
        if self.size > max_sequences:
            raise ValueError(
                f"space has {self.size} genotypes, above max_sequences="
                f"{max_sequences}; enumeration is for verification, not for "
                "scoring a real library."
            )
        return tuple(itertools.product((0, 1), repeat=self.num_sites))

    def enumerate_sequences(self, max_sequences: int = 65_536) -> tuple[str, ...]:
        """Every sequence in the space, in `enumerate_alleles` order."""
        return tuple(
            self.sequence_for(alleles)
            for alleles in self.enumerate_alleles(max_sequences)
        )


# --------------------------------------------------------------------------
# Cached geometry
# --------------------------------------------------------------------------

def _tensor_digest(*tensors: Tensor) -> str:
    """A stable content digest over tensor shapes and values."""
    digest = hashlib.sha256()
    for tensor in tensors:
        array = tensor.detach().to(torch.float64).cpu().contiguous()
        digest.update(repr(tuple(array.shape)).encode("utf-8"))
        digest.update(array.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, eq=False)
class FixedGeometry:
    """A detached, reusable encoding of one declared backbone geometry.

    The encoder never sees the sequence, so reusing this across candidates is
    exact rather than approximate (see the module docstring). It is a
    **process-local cache, not a portable artifact**: its tensors live on the
    device and dtype of the model that produced them, and it carries no
    provenance for the weights, structure, or chain selection behind them.
    Re-encode after moving or re-casting the model, after any change to encoder
    parameters, and after any change to the coordinates or confidence.

    Attributes:
        encoder_out: Final encoder states, ``(T, 1, C)``, detached. ``T`` is
            ``num_residues + 2``: upstream pads one `inf` flank at each end.
        encoder_padding_mask: ``(1, T)`` bool, detached.
        num_residues: Residue count of the coordinates, excluding the flanks.
        digest: Content digest of the two tensors, for run records.
    """

    encoder_out: Tensor
    encoder_padding_mask: Tensor
    num_residues: int
    digest: str

    @property
    def device(self) -> torch.device:
        """Device the cached tensors live on."""
        return self.encoder_out.device

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the cached encoder states."""
        return self.encoder_out.dtype

    def decoder_encoder_out(self, batch_size: int) -> dict[str, list[Tensor]]:
        """The `encoder_out` mapping the decoder expects, broadcast to a batch.

        Expansion is a view, and both attention paths call `.contiguous()`
        before reshaping (`esm/multihead_attention.py:280-284`), so no copy is
        made per candidate.

        Raises:
            ValueError: If `batch_size` is not positive.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        return {
            "encoder_out": [self.encoder_out.expand(-1, batch_size, -1)],
            "encoder_padding_mask": [self.encoder_padding_mask.expand(batch_size, -1)],
            "encoder_embedding": [],
            "encoder_states": [],
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"FixedGeometry(num_residues={self.num_residues}, "
            f"shape={tuple(self.encoder_out.shape)}, device={self.device}, "
            f"dtype={self.dtype}, digest={self.digest[:12]}...)"
        )


@dataclass(frozen=True, eq=False)
class ConstrainedSample:
    """Sampled candidates and the exact log-probability of the actions taken.

    Attributes:
        sequences: One full sequence per sample.
        alleles: The matching genotypes, in site order.
        log_probability: ``(B,)`` detached sum of the chosen per-site
            constrained log-probabilities -- the log-probability of the *actions*
            under this editing process, which is `log q` for the sampled
            sequence. It is detached by construction; re-score with `log_prob`
            to obtain a gradient-carrying value.
    """

    sequences: tuple[str, ...]
    alleles: tuple[tuple[int, ...], ...]
    log_probability: Tensor


# --------------------------------------------------------------------------
# Scoring / sampling
# --------------------------------------------------------------------------

def _integer_alleles(alleles: Any) -> Tensor:
    """Coerce genotypes to a long tensor without ever truncating a value.

    ``torch.as_tensor(x, dtype=torch.long)`` rounds toward zero, so a genotype of
    ``0.7`` would silently become allele 0 and score a sequence nobody asked for.
    Every non-integral input is refused instead.

    Args:
        alleles: A tensor, or anything `torch.as_tensor` accepts, holding whole
            numbers. Bool and any integer or exactly-integral floating dtype are
            accepted; the values themselves are range-checked by the caller.

    Returns:
        The same values as a long tensor, on their original device.

    Raises:
        ValueError: If the input is not convertible to a tensor, is complex, or
            holds a NaN, an infinity, or a fractional value.
    """
    if isinstance(alleles, Tensor):
        values = alleles
    else:
        try:
            values = torch.as_tensor(alleles)
        except (TypeError, RuntimeError, ValueError) as error:
            raise ValueError(
                "allele indices must be an integer tensor or a nested sequence "
                f"of integers, got {alleles!r}."
            ) from error
    if values.is_complex():
        raise ValueError(
            "allele indices must be real whole numbers, got a complex tensor."
        )
    if values.is_floating_point():
        if not torch.isfinite(values).all():
            raise ValueError(
                "allele indices contain NaN or infinity; they must be whole "
                "numbers in {0, 1}."
            )
        if not torch.equal(values, values.round()):
            raise ValueError(
                "allele indices must be whole numbers; a fractional index would "
                "be truncated toward zero and silently score a different "
                "genotype."
            )
    return values.to(torch.long)


@contextmanager
def _dropout_off(module: nn.Module) -> Iterator[None]:
    """Run `module` in eval mode, restoring every submodule's flag afterwards.

    Deliberately does **not** disable gradients: the decoder must stay
    differentiable while scoring. `torch.no_grad` around the decoder would make
    the whole policy untrainable, which is the failure this separation prevents.
    """
    previous = [(child, child.training) for child in module.modules()]
    try:
        module.eval()
        yield
    finally:
        for child, was_training in previous:
            child.train(was_training)


class ConstrainedEditPolicy(nn.Module):
    """ESM-IF1's native decoder, restricted to a binary-allele edit space.

    Freezes the structure encoder (`requires_grad_(False)` on its parameters) and
    leaves the decoder trainable, matching the pilot's declared frozen-encoder /
    adapted-decoder split. Temperature is fixed at 1 and there is no nucleus or
    rejection knob, so the policy and a frozen reference copy cannot silently
    diverge in their sampling or scoring process.

    The site buffers are built on the backbone's device, so passing an
    already-moved model works; move the whole policy with `policy.to(device)`
    afterwards, never the backbone alone, and re-encode the geometry after any
    move.

    Args:
        model: A `GVPTransformerModel`-shaped object, i.e. one exposing
            `encoder(coords, padding_mask, confidence, ...)` returning upstream's
            `encoder_out` dict, and `decoder(prev_output_tokens, encoder_out=...,
            incremental_state=...)` returning ``(B x V x T, extra)``.
        space: The edit space. Site positions index `space.context`.
        alphabet: Optional alphabet to validate against the native table. When
            the model's decoder exposes a `dictionary`, that is validated too.

    Raises:
        TypeError: If `model` has no `encoder`/`decoder`, or `space` is not a
            `ConstrainedEditSpace`.
        ValueError: If a supplied alphabet or the model's own dictionary is not
            the native ESM-IF1 table.
    """

    def __init__(
        self,
        model: nn.Module,
        space: ConstrainedEditSpace,
        alphabet: Any | None = None,
    ) -> None:
        super().__init__()
        for attribute in ("encoder", "decoder"):
            if not hasattr(model, attribute):
                raise TypeError(
                    f"model has no `{attribute}`; expected a GVPTransformerModel-"
                    "shaped object."
                )
        if not isinstance(space, ConstrainedEditSpace):
            raise TypeError(f"space must be a ConstrainedEditSpace, got {space!r}.")

        self.alphabet = (
            ESMIF1Alphabet.native() if alphabet is None
            else ESMIF1Alphabet.matching(alphabet)
        )
        dictionary = getattr(model.decoder, "dictionary", None)
        if dictionary is not None:
            ESMIF1Alphabet.matching(dictionary)

        self.space = space
        self.model = model
        for parameter in self.model.encoder.parameters():
            parameter.requires_grad_(False)

        # Built on the backbone's own device, so a model that was already moved
        # (`ConstrainedEditPolicy(model.cuda(), space)`) works without a further
        # call. They are buffers, so `policy.to(device)` still carries them.
        device = self._backbone_device()
        self.register_buffer(
            "site_positions",
            torch.tensor(space.positions, dtype=torch.long, device=device),
            persistent=False,
        )
        self.register_buffer(
            "allele_token_ids",
            torch.tensor(
                [[self.alphabet.index_of(a) for a in site.alleles]
                 for site in space.sites],
                dtype=torch.long,
                device=device,
            ),
            persistent=False,
        )

    # -- helpers ---------------------------------------------------------

    def _backbone_device(self) -> torch.device:
        """Device the backbone lives on; CPU for a parameterless stub."""
        for parameter in self.model.parameters():
            return parameter.device
        for buffer in self.model.buffers():
            return buffer.device
        return torch.device("cpu")

    def _check_buffer_device(self, device: torch.device) -> None:
        """Refuse a policy whose buffers drifted away from its backbone.

        Moving `policy.model` alone leaves `site_positions`/`allele_token_ids`
        behind, and the resulting `index_select` failure names neither the cause
        nor the fix.
        """
        if self.site_positions.device != device:
            raise ValueError(
                f"the policy's site buffers are on {self.site_positions.device} "
                f"but the tensors being indexed are on {device}. Move the whole "
                "policy with `policy.to(device)`, which carries its buffers, "
                "rather than the backbone alone."
            )

    def _reference_parameter(self) -> Tensor:
        for parameter in self.model.parameters():
            return parameter
        raise ValueError(
            "model has no parameters, so its device and dtype cannot be "
            "determined."
        )

    def _validated_batch(self, sequences: str | Iterable[str]) -> tuple[
        tuple[str, ...], Tensor
    ]:
        """Validate a batch of candidates and read their genotypes."""
        if isinstance(sequences, str):
            batch: tuple[str, ...] = (sequences,)
        else:
            batch = tuple(sequences)
        if not batch:
            raise ValueError("empty candidate batch.")
        alleles = [self.space.alleles_for(sequence) for sequence in batch]
        return batch, torch.tensor(
            alleles, dtype=torch.long, device=self.site_positions.device
        )

    def _check_geometry(self, geometry: FixedGeometry) -> None:
        if not isinstance(geometry, FixedGeometry):
            raise TypeError(
                f"geometry must be a FixedGeometry from `encode_geometry`, got "
                f"{geometry!r}."
            )
        if geometry.num_residues < len(self.space.context):
            raise ValueError(
                f"geometry covers {geometry.num_residues} residues but the "
                f"context is {len(self.space.context)} long. The decoded chain "
                "may be shorter than the encoded complex, never longer."
            )
        reference = self._reference_parameter()
        self._check_buffer_device(reference.device)
        if geometry.device != reference.device:
            raise ValueError(
                f"geometry is cached on {geometry.device} but the model is on "
                f"{reference.device}; re-encode after moving the model."
            )
        if geometry.dtype != reference.dtype:
            raise ValueError(
                f"geometry is cached as {geometry.dtype} but the model is "
                f"{reference.dtype}; re-encode after re-casting the model."
            )

    def _check_logits(self, logits: Tensor) -> None:
        self._check_buffer_device(logits.device)
        if logits.dim() != 3:
            raise ValueError(
                f"expected decoder logits shaped (batch, vocab, length), got "
                f"{tuple(logits.shape)}."
            )
        if logits.shape[1] != len(self.alphabet):
            raise ValueError(
                f"decoder logits have {logits.shape[1]} vocabulary entries but "
                f"the native alphabet has {len(self.alphabet)}; the token "
                "indices used for each site would be meaningless."
            )
        last_position = int(self.site_positions.max())
        if logits.shape[2] <= last_position:
            raise ValueError(
                f"decoder logits cover {logits.shape[2]} decisions but the last "
                f"editable site is at position {last_position}."
            )

    # -- geometry --------------------------------------------------------

    def encode_geometry(
        self,
        coords: Any,
        confidence: Any | None = None,
    ) -> FixedGeometry:
        """Encode one declared backbone geometry, once, without gradients.

        Reproduces `CoordBatchConverter` for a single chain: an `inf` flank at
        each end of the coordinates, a `-1.` flank on confidence, then upstream's
        exact masking arithmetic.

        **Missing coordinates versus padding.** Upstream derives
        ``padding_mask = isnan(coords[:, :, 0, 0])``, i.e. from the *N atom's x
        coordinate alone*. For a single chain there is no batch padding, so every
        NaN here is caller-supplied, and the two cases diverge:

        - a residue whose N coordinate is NaN is treated as **padding** -- masked
          out of attention entirely and given confidence ``-1``;
        - a residue with a finite N but a NaN CA or C is **not** padding: it stays
          in attention with its coordinates zero-filled and its confidence forced
          to ``0``.

        Neither is "unknown but present with an unknown conformation". Declaring a
        missing-coordinate convention is a pending gate, not something this
        function decides. The `inf` flanks land in the second case by
        construction, which is upstream's intent.

        Args:
            coords: ``(L, 3, 3)`` N/CA/C backbone coordinates for one chain, as a
                tensor, array, or nested sequence. Cast to the model's parameter
                dtype (upstream does not cast, and a float64 array against
                float32 weights raises deep inside the encoder).
            confidence: `None` (all ones), a scalar in ``[0, 1]``, or a length-`L`
                sequence in ``[0, 1]``.

        Returns:
            A `FixedGeometry`.

        Raises:
            ValueError: On a wrong coordinate rank/shape, an empty chain, a
                confidence length that differs from `L`, a confidence value
                outside ``[0, 1]``, or an encoder output that is not
                ``(T, 1, C)``.
        """
        reference = self._reference_parameter()
        device, dtype = reference.device, reference.dtype

        coords_t = torch.as_tensor(coords, dtype=dtype, device=device)
        if coords_t.dim() != 3 or tuple(coords_t.shape[1:]) != (3, 3):
            raise ValueError(
                "coords must be a single chain shaped (L, 3, 3) for N, CA, C; "
                f"got {tuple(coords_t.shape)}. Batched or multichain packing is "
                "the caller's responsibility and is not implemented here."
            )
        num_residues = int(coords_t.shape[0])
        if num_residues < 1:
            raise ValueError("coords must contain at least one residue.")

        if confidence is None:
            confidence_t = torch.ones(num_residues, dtype=dtype, device=device)
        else:
            confidence_t = torch.as_tensor(confidence, dtype=dtype, device=device)
            if confidence_t.dim() == 0:
                confidence_t = confidence_t.expand(num_residues).clone()
            elif confidence_t.dim() != 1:
                raise ValueError(
                    "confidence must be a scalar or a 1-D sequence, got shape "
                    f"{tuple(confidence_t.shape)}."
                )
            elif confidence_t.shape[0] != num_residues:
                raise ValueError(
                    f"confidence has length {confidence_t.shape[0]} but the "
                    f"coordinates cover {num_residues} residues."
                )
            if torch.isnan(confidence_t).any():
                raise ValueError("confidence contains NaN.")
            if float(confidence_t.min()) < 0.0 or float(confidence_t.max()) > 1.0:
                raise ValueError(
                    "confidence values must lie in [0, 1]; a missing coordinate "
                    "is expressed as a NaN coordinate, not as a negative "
                    "confidence. Upstream derives its own -1 flank internally."
                )

        # Exactly `CoordBatchConverter.__call__` for a batch of one.
        padded_coords = F.pad(
            coords_t, (0, 0, 0, 0, 1, 1), value=float("inf")
        ).unsqueeze(0)
        padded_confidence = F.pad(confidence_t, (1, 1), value=-1.0).unsqueeze(0)
        padding_mask = torch.isnan(padded_coords[:, :, 0, 0])
        coord_mask = torch.isfinite(padded_coords.sum(-2).sum(-1))
        padded_confidence = (
            padded_confidence * coord_mask.to(dtype)
            + (-1.0) * padding_mask.to(dtype)
        )

        with torch.no_grad(), _dropout_off(self.model.encoder):
            encoder_out = self.model.encoder(
                padded_coords,
                padding_mask,
                padded_confidence,
                return_all_hiddens=False,
            )
        states = encoder_out["encoder_out"][0].detach()
        mask = encoder_out["encoder_padding_mask"][0].detach()
        if states.dim() != 3 or states.shape[1] != 1:
            raise ValueError(
                "expected encoder states shaped (length, 1, channels), got "
                f"{tuple(states.shape)}."
            )
        return FixedGeometry(
            encoder_out=states,
            encoder_padding_mask=mask,
            num_residues=num_residues,
            digest=_tensor_digest(states, mask),
        )

    # -- teacher-forced scoring -----------------------------------------

    def prefix_tokens(self, sequences: str | Iterable[str]) -> Tensor:
        """The decoder input `prev_output_tokens` for a batch of candidates.

        `<cath>` followed by residues ``0 .. L-2``: upstream's
        ``tokens[:, :-1]``, so that column ``p`` of the returned logits is the
        decision for residue ``p`` and every forced residue before ``p`` is in
        the prefix.

        Returns:
            ``(B, L)`` long tensor, where ``L = len(space.context)``.
        """
        batch, _ = self._validated_batch(sequences)
        rows = [
            [self.alphabet.prefix_idx, *self.alphabet.encode(sequence[:-1])]
            for sequence in batch
        ]
        return torch.tensor(
            rows, dtype=torch.long, device=self._reference_parameter().device
        )

    def native_logits(
        self, sequences: str | Iterable[str], geometry: FixedGeometry
    ) -> Tensor:
        """Raw native decoder logits, teacher-forced and differentiable.

        Dropout is off; gradients are **not** disabled. Equal to upstream's
        ``model.forward(coords, padding_mask, confidence, prev_output_tokens)``
        logits for the same inputs, because the cached encoding is exact.

        Returns:
            ``(B, V, L)`` over the full native vocabulary, unnormalized and
            untempered. Site restriction happens downstream.
        """
        self._check_geometry(geometry)
        prev_output_tokens = self.prefix_tokens(sequences)
        encoder_out = geometry.decoder_encoder_out(prev_output_tokens.shape[0])
        with _dropout_off(self.model):
            logits, _ = self.model.decoder(
                prev_output_tokens,
                encoder_out=encoder_out,
                incremental_state=None,
            )
        self._check_logits(logits)
        return logits

    def site_log_probabilities_from_logits(self, logits: Tensor) -> Tensor:
        """Per-site log-probabilities over each site's two-residue support.

        Args:
            logits: ``(B, V, T)`` native logits.

        Returns:
            ``(B, num_sites, 2)``, normalized over the last dimension at
            temperature 1. Allele order matches each site's `alleles` tuple.
        """
        self._check_logits(logits)
        site_logits = logits.index_select(2, self.site_positions).permute(0, 2, 1)
        allowed = site_logits.gather(
            2, self.allele_token_ids.unsqueeze(0).expand(site_logits.shape[0], -1, -1)
        )
        return torch.log_softmax(allowed, dim=-1)

    def site_log_probabilities(
        self, sequences: str | Iterable[str], geometry: FixedGeometry
    ) -> Tensor:
        """`site_log_probabilities_from_logits` on teacher-forced logits."""
        return self.site_log_probabilities_from_logits(
            self.native_logits(sequences, geometry)
        )

    def log_prob_from_logits(self, logits: Tensor, alleles: Tensor) -> Tensor:
        """`log q` from native logits and genotypes.

        Args:
            logits: ``(B, V, T)`` native logits.
            alleles: ``(B, num_sites)`` allele indices.

        Returns:
            ``(B,)`` summed constrained log-probabilities. Forced positions
            contribute nothing: they are never indexed.

        Raises:
            ValueError: On a genotype that is not integral (see
                `_integer_alleles`), a genotype shape that does not match the
                space, or an allele index outside {0, 1}.
        """
        chosen = _integer_alleles(alleles).to(device=logits.device)
        if chosen.dim() == 1:
            chosen = chosen.unsqueeze(0)
        if chosen.dim() != 2 or chosen.shape[1] != self.space.num_sites:
            raise ValueError(
                f"expected allele indices shaped (batch, {self.space.num_sites}), "
                f"got {tuple(chosen.shape)}."
            )
        if chosen.shape[0] != logits.shape[0]:
            raise ValueError(
                f"{chosen.shape[0]} genotypes for {logits.shape[0]} logit rows."
            )
        if chosen.numel() == 0:
            raise ValueError("empty genotype batch.")
        if int(chosen.min()) < 0 or int(chosen.max()) > 1:
            raise ValueError("allele indices must lie in {0, 1}.")
        site_log_probs = self.site_log_probabilities_from_logits(logits)
        return site_log_probs.gather(2, chosen.unsqueeze(-1)).squeeze(-1).sum(-1)

    def log_prob(
        self, sequences: str | Iterable[str], geometry: FixedGeometry
    ) -> Tensor:
        """`log q(y | C)` for a batch of candidates, differentiable.

        Returns:
            ``(B,)``. Comparable only within one space and one geometry: this is
            a renormalized editing process, not a sequence likelihood.
        """
        batch, alleles = self._validated_batch(sequences)
        return self.log_prob_from_logits(self.native_logits(batch, geometry), alleles)

    # -- sampling --------------------------------------------------------

    def sample(
        self,
        geometry: FixedGeometry,
        num_samples: int = 1,
        generator: torch.Generator | None = None,
    ) -> ConstrainedSample:
        """Sample genotypes under exactly the process `log_prob` scores.

        Walks prefix positions ``0 .. last_site_position`` in the native decoding
        order, calling the decoder at **every** position -- forced positions
        included -- with one shared `incremental_state`. Upstream's `sample`
        skips the call at known positions, which leaves the ingested key/value
        cache with holes; this does not, so the returned log-probability equals
        the teacher-forced `log_prob` of the same sequence.

        Runs under `no_grad`: the returned log-probability is the exact sum of
        the chosen action log-probabilities, detached. Re-score with `log_prob`
        for a gradient-carrying value (sample-then-score).

        Args:
            geometry: A `FixedGeometry` from `encode_geometry`.
            num_samples: Batch size.
            generator: Optional `torch.Generator` for reproducibility; must live
                on the model's device.

        Returns:
            A `ConstrainedSample`. Every immutable residue is forced, so every
            returned sequence is a member of the space.

        Raises:
            ValueError: If `num_samples` is not positive, or the geometry is
                unusable (see `encode_geometry`).
        """
        if num_samples < 1:
            raise ValueError(f"num_samples must be positive, got {num_samples}.")
        self._check_geometry(geometry)
        device = self._reference_parameter().device
        dtype = self._reference_parameter().dtype

        last_position = self.space.sites[-1].position
        site_of = {position: index for index, position in enumerate(self.space.positions)}

        # `<cath>` plus context residues 0..last_position: the decoder input at
        # step t is tokens[:, :t + 1], and the decision at step t is written to
        # tokens[:, t + 1].
        prefix = [self.alphabet.prefix_idx, *self.alphabet.encode(
            self.space.context[: last_position + 1]
        )]
        tokens = torch.tensor(prefix, dtype=torch.long, device=device).repeat(
            num_samples, 1
        )
        encoder_out = geometry.decoder_encoder_out(num_samples)
        chosen = torch.zeros(
            (num_samples, self.space.num_sites), dtype=torch.long, device=device
        )
        total = torch.zeros(num_samples, dtype=dtype, device=device)
        incremental_state: dict[str, dict[str, Tensor | None]] = {}

        with torch.no_grad(), _dropout_off(self.model):
            for step in range(last_position + 1):
                logits, _ = self.model.decoder(
                    tokens[:, : step + 1],
                    encoder_out=encoder_out,
                    incremental_state=incremental_state,
                )
                if step not in site_of:
                    # The token is forced, but the call above still had to run:
                    # skipping it would leave this position out of the cache.
                    continue
                site_index = site_of[step]
                allowed_ids = self.allele_token_ids[site_index]
                step_logits = logits[:, :, -1].index_select(1, allowed_ids)
                log_probs = torch.log_softmax(step_logits, dim=-1)
                picked = torch.multinomial(
                    log_probs.exp(), 1, generator=generator
                )
                chosen[:, site_index] = picked.squeeze(-1)
                total = total + log_probs.gather(1, picked).squeeze(-1)
                tokens[:, step + 1] = allowed_ids[picked.squeeze(-1)]

        genotypes = tuple(tuple(row) for row in chosen.tolist())
        return ConstrainedSample(
            sequences=tuple(self.space.sequence_for(g) for g in genotypes),
            alleles=genotypes,
            log_probability=total,
        )
