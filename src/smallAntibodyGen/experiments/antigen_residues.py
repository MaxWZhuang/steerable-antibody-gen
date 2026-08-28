"""
Give both J24 arms exactly the same antigen residues.

J24 asks whether a frozen ESM-2 antigen encoder beats the scratch one. That
question is only answerable if the two arms are shown the same antigen. They are
not, by default, because the two tokenizers spend different numbers of tokens on
special tokens inside the same token budget:

    scratch:  [CLS] [OTHER_CHAIN] ...residues... [EOS]      -> 3 specials
    ESM-2:    <cls>               ...residues... <eos>      -> 2 specials

Measured, not assumed (`test_j24_antigen_encoder.py` re-measures it): at a
1024-token budget the scratch arm retains 1021 residues and the ESM arm 1022. One
residue per antigen, always at the C-terminal end, always in the same arm's
favour. That is small, systematic, and exactly the kind of difference that turns
a one-axis comparison into a two-axis one -- and it would never show up in a loss
curve.

The fix is to stop letting each tokenizer truncate. Crop the amino-acid string
ONCE to a residue budget both arms can represent, then encode the identical
string in each arm. Cropping in residue space rather than token space is what
makes "the same residues" checkable: the two arms' inputs decode back to the same
substring, which is asserted directly rather than inferred from token counts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from smallAntibodyGen.antigen_tokenization import AntigenTokenizer

#: Residues used to probe a tokenizer's overhead. Deliberately ordinary: the
#: probe must not itself hit an unknown-token path, or the measured overhead
#: would describe the probe rather than the tokenizer.
_PROBE_RESIDUES = "ACDEFGHIKLMNPQRSTVWY"


class AntigenTokenizationError(ValueError):
    """A tokenizer does not satisfy the assumptions residue cropping relies on."""


def special_token_overhead(tokenizer: AntigenTokenizer, token_budget: int) -> int:
    """
    How many tokens this tokenizer spends on anything that is not a residue.

    Measured by encoding probes of known length rather than by reading a
    tokenizer's configuration, because the number that matters is what the
    adapter actually emits -- including any chain token the scratch path inserts,
    which no ESM-side configuration would ever mention.

    Raises:
        AntigenTokenizationError:
            If the overhead is not constant across probe lengths, or if the
            tokenizer does not emit exactly one token per residue. Residue
            cropping is only meaningful under those two properties, so they are
            checked rather than trusted.
    """
    overheads = set()
    for length in (1, 4, 16):
        probe = (_PROBE_RESIDUES * ((length // len(_PROBE_RESIDUES)) + 1))[:length]
        emitted = len(tokenizer.encode(probe, token_budget))
        overheads.add(emitted - length)
    if len(overheads) != 1:
        raise AntigenTokenizationError(
            f"{type(tokenizer).__name__} does not emit one token per residue plus a "
            f"constant overhead; measured overheads {sorted(overheads)}. Residue "
            "cropping cannot equalize two arms whose token count is not affine in "
            "the residue count."
        )
    overhead = overheads.pop()
    if overhead < 0:
        raise AntigenTokenizationError(
            f"{type(tokenizer).__name__} emitted fewer tokens than residues"
        )
    return overhead


def residue_capacity(tokenizer: AntigenTokenizer, token_budget: int) -> int:
    """How many residues fit in ``token_budget`` for this tokenizer."""
    return max(0, token_budget - special_token_overhead(tokenizer, token_budget))


@dataclass(frozen=True)
class ResidueBudget:
    """
    The residue budget shared by every arm, plus the per-arm capacities it came
    from. The per-arm numbers are kept because the interesting part of this
    object is usually the DIFFERENCE -- a report that only shows the minimum
    hides how much each arm gave up.
    """

    token_budget: int
    residues: int
    per_arm_capacity: Mapping[str, int]

    @property
    def sacrificed(self) -> Mapping[str, int]:
        """Residues each arm gives up to match the most constrained arm."""
        return {name: cap - self.residues for name, cap in self.per_arm_capacity.items()}


def common_residue_budget(
    tokenizers: Mapping[str, AntigenTokenizer],
    token_budget: int,
) -> ResidueBudget:
    """
    The largest residue count EVERY arm can represent within ``token_budget``.

    The minimum, not the mean and not each arm's own maximum: an arm that can
    hold more residues must give them up, because the comparison is between
    encoders and not between context windows.

    Args:
        tokenizers: Arm name -> its antigen tokenizer.
        token_budget: The shared ``antigen_max_length``.

    Returns:
        A :class:`ResidueBudget` carrying the shared budget and what each arm
        sacrificed to reach it.
    """
    if not tokenizers:
        raise ValueError("at least one tokenizer is required")
    capacity = {
        name: residue_capacity(tok, token_budget) for name, tok in tokenizers.items()
    }
    return ResidueBudget(
        token_budget=token_budget,
        residues=min(capacity.values()),
        per_arm_capacity=capacity,
    )


def crop_antigen(sequence: str, residues: int) -> str:
    """
    Crop an antigen to ``residues`` amino acids.

    N-terminal, matching what both tokenizers' own truncation already does, so
    cropping changes WHICH arm truncates rather than WHERE the sequence is cut.
    Both arms then receive a byte-identical string.
    """
    return (sequence or "").strip().upper()[:residues]


def encode_arms_identically(
    sequence: str,
    tokenizers: Mapping[str, AntigenTokenizer],
    budget: ResidueBudget,
) -> dict[str, list[int]]:
    """
    Encode one antigen for every arm from the SAME cropped residue string.

    Returns arm name -> token ids. Each arm's ids differ (different vocabularies
    and special tokens); the residues behind them do not, which is the invariant
    J24 needs and which
    :func:`~smallAntibodyGen.experiments.antigen_residues.retained_residue_count`
    lets a test confirm from the ids themselves.
    """
    cropped = crop_antigen(sequence, budget.residues)
    return {
        name: tok.encode(cropped, budget.token_budget)
        for name, tok in tokenizers.items()
    }


def retained_residue_count(
    token_ids: Sequence[int],
    tokenizer: AntigenTokenizer,
    token_budget: int,
) -> int:
    """
    How many residues an encoded antigen actually carries.

    Derived from the measured overhead rather than by decoding, so it works for
    any adapter satisfying the affine-token-count property, and so a test can
    compare two arms without depending on either vocabulary.
    """
    return len(token_ids) - special_token_overhead(tokenizer, token_budget)
