"""
Affine-gap pairwise alignment, the metric every relation in this package is built on.

WHY THIS MODULE EXISTS AT ALL
-----------------------------
Entity resolution over antigen sequences needs one number for "how much of A is
also in B", and the choice of alignment decides what that number can mean. Two
mistakes are cheap to make and expensive to find:

- **Exact containment.** `short in long` is O(n) and wrong. The 287-aa and
  289-aa Omicron spike NTDs differ by a two-residue *interior* insertion, so
  `287 in 289` is False even though every one of the 287-mer's residues is
  matched. A previous attempt in this repository used exact containment, tore
  that family in half, and leaked 335 validation rows whose exact heavy+light
  chains were on the train side.
- **Global alignment.** Needleman-Wunsch charges for the parts of the longer
  sequence that stick out, so a genuine truncation looks like a mismatch. The
  564-aa HER2 ECD is a byte-exact sub-region of the 607-aa one; global alignment
  scores that pair at 0.9292 identity, local alignment at 1.0000 identity with
  0.9292 coverage of the longer side. Only the second decomposition lets a caller
  say "same protein, different construct boundaries", which is precisely the
  distinction the three relations in this package turn on.

So the primitive here is **local (Smith-Waterman) alignment with affine gaps**,
scored with BLOSUM62 and the BLAST default penalties (open 11, extend 1), and it
reports identity and the two coverages *separately* rather than collapsing them.

WHAT AFFINE GAPS BUY
--------------------
A linear gap penalty charges `k * g` for a gap of length k, so it prefers many
short gaps to one long one. Affine charges `open + k * extend`, which prefers one
long gap -- the biologically right answer for an indel, a truncation, or a linker
in a fusion construct. `tests/test_entity_resolution_primitives.py` pins the
difference where it shows, so that swapping this module for a linear-gap
implementation fails a test rather than quietly re-partitioning the corpus.

DETERMINISM
-----------
Every result here is a pure function of the UNORDERED pair, and that is not free.

The affine score is unique. Identity and the coverages are not: they are read off
one member of the co-optimal path set, and ties in the traceback are broken in a
fixed order (diagonal, then horizontal, then vertical). Transposing the two
sequences swaps the roles of the horizontal and vertical gap states and turns
row-major tie-breaking into column-major, so an implementation that simply runs
the DP as given returns DIFFERENT identity and coverage for ``align_pair(a, b)``
and ``align_pair(b, a)``. Measured on real corpus pairs: 24 of 400.

So the orientation is canonicalised -- shorter first, ties by content -- and
flipped on the way out. `reference_align_pair` canonicalises identically, and
that one shared convention is the ONLY thing the oracle shares with the
implementation it checks. The alternative, leaving the function order-dependent
and documenting it, would make every threshold in this package depend on the sort
order of two digests: deterministic and arbitrary at the same time.

COST MODEL
----------
`align_pair` is O(len(a) * len(b)) time and memory: the traceback direction of
every cell is retained as one byte. MEASURED on this machine, a 420x430 pair
costs about 25 ms -- roughly 7 million DP cells per second, bound by per-row
numpy call overhead rather than by cells. Resolving the shipped corpus proposes
450,381 candidate pairs and therefore takes HOURS; `scripts/resolve_target_identity.py`
repeats that figure so nobody starts a run by accident.
Callers that align many pairs must therefore block first -- see
`entity_resolution.blocking`, whose candidate filter is a correctness contract in
its own right and not merely an optimisation. Pairs whose product exceeds
`max_cells` are refused with `AlignmentTooLarge` rather than silently truncated,
because a skipped comparison is an unmeasured leakage risk and has to be counted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

#: BLAST's default affine penalties for BLOSUM62. A gap of length k costs
#: ``GAP_OPEN + k * GAP_EXTEND``.
GAP_OPEN = 11
GAP_EXTEND = 1

#: Largest DP matrix `align_pair` will build, in cells. 50 million cells is about
#: 50 MB of int8 traceback plus three int32 score rows, and covers every pair the
#: length band admits in the shipped corpus. Exceeding it raises rather than
#: degrading, so the caller counts the refusal.
DEFAULT_MAX_CELLS = 50_000_000

_BLOSUM62_ORDER = "*ABCDEFGHIKLMNPQRSTVWXYZ"
_BLOSUM62_ROWS = (
    (  1, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4),  # *
    ( -4,  4, -2,  0, -2, -1, -2,  0, -2, -1, -1, -1, -1, -2, -1, -1, -1,  1,  0,  0, -3,  0, -2, -1),  # A
    ( -4, -2,  4, -3,  4,  1, -3, -1,  0, -3,  0, -4, -3,  3, -2,  0, -1,  0, -1, -3, -4, -1, -3,  1),  # B
    ( -4,  0, -3,  9, -3, -4, -2, -3, -3, -1, -3, -1, -1, -3, -3, -3, -3, -1, -1, -1, -2, -2, -2, -3),  # C
    ( -4, -2,  4, -3,  6,  2, -3, -1, -1, -3, -1, -4, -3,  1, -1,  0, -2,  0, -1, -3, -4, -1, -3,  1),  # D
    ( -4, -1,  1, -4,  2,  5, -3, -2,  0, -3,  1, -3, -2,  0, -1,  2,  0,  0, -1, -2, -3, -1, -2,  4),  # E
    ( -4, -2, -3, -2, -3, -3,  6, -3, -1,  0, -3,  0,  0, -3, -4, -3, -3, -2, -2, -1,  1, -1,  3, -3),  # F
    ( -4,  0, -1, -3, -1, -2, -3,  6, -2, -4, -2, -4, -3,  0, -2, -2, -2,  0, -2, -3, -2, -1, -3, -2),  # G
    ( -4, -2,  0, -3, -1,  0, -1, -2,  8, -3, -1, -3, -2,  1, -2,  0,  0, -1, -2, -3, -2, -1,  2,  0),  # H
    ( -4, -1, -3, -1, -3, -3,  0, -4, -3,  4, -3,  2,  1, -3, -3, -3, -3, -2, -1,  3, -3, -1, -1, -3),  # I
    ( -4, -1,  0, -3, -1,  1, -3, -2, -1, -3,  5, -2, -1,  0, -1,  1,  2,  0, -1, -2, -3, -1, -2,  1),  # K
    ( -4, -1, -4, -1, -4, -3,  0, -4, -3,  2, -2,  4,  2, -3, -3, -2, -2, -2, -1,  1, -2, -1, -1, -3),  # L
    ( -4, -1, -3, -1, -3, -2,  0, -3, -2,  1, -1,  2,  5, -2, -2,  0, -1, -1, -1,  1, -1, -1, -1, -1),  # M
    ( -4, -2,  3, -3,  1,  0, -3,  0,  1, -3,  0, -3, -2,  6, -2,  0,  0,  1,  0, -3, -4, -1, -2,  0),  # N
    ( -4, -1, -2, -3, -1, -1, -4, -2, -2, -3, -1, -3, -2, -2,  7, -1, -2, -1, -1, -2, -4, -2, -3, -1),  # P
    ( -4, -1,  0, -3,  0,  2, -3, -2,  0, -3,  1, -2,  0,  0, -1,  5,  1,  0, -1, -2, -2, -1, -1,  3),  # Q
    ( -4, -1, -1, -3, -2,  0, -3, -2,  0, -3,  2, -2, -1,  0, -2,  1,  5, -1, -1, -3, -3, -1, -2,  0),  # R
    ( -4,  1,  0, -1,  0,  0, -2,  0, -1, -2,  0, -2, -1,  1, -1,  0, -1,  4,  1, -2, -3,  0, -2,  0),  # S
    ( -4,  0, -1, -1, -1, -1, -2, -2, -2, -1, -1, -1, -1,  0, -1, -1, -1,  1,  5,  0, -2,  0, -2, -1),  # T
    ( -4,  0, -3, -1, -3, -2, -1, -3, -3,  3, -2,  1,  1, -3, -2, -2, -3, -2,  0,  4, -3, -1, -1, -2),  # V
    ( -4, -3, -4, -2, -4, -3,  1, -2, -2, -3, -3, -2, -1, -4, -4, -2, -3, -3, -2, -3, 11, -2,  2, -3),  # W
    ( -4,  0, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, -1, -1,  0,  0, -1, -2, -1, -1, -1),  # X
    ( -4, -2, -3, -2, -3, -2,  3, -3,  2, -1, -2, -1, -1, -2, -3, -1, -2, -2, -2, -1,  2, -1,  7, -2),  # Y
    ( -4, -1,  1, -3,  1,  4, -3, -2,  0, -3,  1, -3, -1,  0, -1,  3,  0,  0, -1, -2, -3, -1, -2,  4),  # Z
)

#: Residue codes `clean_aa_sequence` can emit that BLOSUM62 has no row for.
#: `J` (Leu/Ile), `O` (pyrrolysine) and `U` (selenocysteine) are folded onto `X`,
#: the ambiguity code, so an exotic residue never crashes and never scores a
#: spurious match against itself.
_UNKNOWN_RESIDUE_ALIASES = {"J": "X", "O": "X", "U": "X"}

_ALPHABET_SIZE = len(_BLOSUM62_ORDER)
_UNKNOWN_INDEX = _BLOSUM62_ORDER.index("X")

# Byte -> alphabet index. Anything unmapped becomes X, so a caller cannot inject a
# residue this module has no score for.
_CODE_OF = np.full(256, _UNKNOWN_INDEX, dtype=np.int32)
for _i, _letter in enumerate(_BLOSUM62_ORDER):
    _CODE_OF[ord(_letter)] = _i
for _alias, _target in _UNKNOWN_RESIDUE_ALIASES.items():
    _CODE_OF[ord(_alias)] = _BLOSUM62_ORDER.index(_target)

_SCORE = np.array(_BLOSUM62_ROWS, dtype=np.int32)
assert _SCORE.shape == (_ALPHABET_SIZE, _ALPHABET_SIZE)
assert np.array_equal(_SCORE, _SCORE.T), "BLOSUM62 must be symmetric"


def _swap(left: str, right: str) -> bool:
    """Whether this pair should be run in the other order.

    Identity and coverage are read off ONE member of the co-optimal path set and
    which member depends on argument order, so the pair is run in a canonical
    orientation and flipped afterwards. Shorter first, ties broken by content.

    Args:
        left: First residue string.
        right: Second residue string.

    Returns:
        True when the arguments should be exchanged before running the DP.
    """
    return (len(left), left) > (len(right), right)


class AlignmentTooLarge(RuntimeError):
    """Raised when a pair exceeds `max_cells`, so the caller counts the refusal."""


@dataclass(frozen=True)
class PairAlignment:
    """One aligned pair, decomposed so identity and extent stay separable.

    Identity and coverage are kept deliberately orthogonal. Identity asks "where
    these two line up, how well do they line up"; coverage asks "how much of each
    sequence is inside that aligned region". Multiplying them together -- which a
    definition of coverage as *matched* residues over length silently does -- is
    what makes a truncation indistinguishable from a divergence, and telling
    those two apart is the entire job of the three relations built on top of this.

    Attributes:
        identity:
            Matched residues over alignment columns, gap columns included.
        cov_left:
            Residues of ``left`` inside the aligned span, over ``len(left)``.
        cov_right:
            Residues of ``right`` inside the aligned span, over ``len(right)``.
        overlap:
            Matched residues, as an absolute count. Full coverage of a 9-residue
            sequence is not evidence of anything, so every threshold in this
            package is paired with a floor on this.
        columns:
            Alignment columns, gap columns included.
        matched:
            Matched residues. Same number as ``overlap``; both names exist
            because the DP speaks in matches and the relations speak in overlap.
        span_left:
            Residues of ``left`` inside the aligned span.
        span_right:
            Residues of ``right`` inside the aligned span.
        score:
            The BLOSUM62 affine score of the reported alignment.
    """

    identity: float
    cov_left: float
    cov_right: float
    overlap: int
    columns: int
    matched: int
    span_left: int
    span_right: int
    score: int

    @property
    def min_coverage(self) -> float:
        """The weaker of the two coverages -- what strict identity must clear."""
        return min(self.cov_left, self.cov_right)

    @property
    def max_coverage(self) -> float:
        """The stronger of the two coverages -- what containment fires on."""
        return max(self.cov_left, self.cov_right)

    def flipped(self) -> "PairAlignment":
        """The same alignment with the two sequences' roles exchanged."""
        return PairAlignment(
            identity=self.identity,
            cov_left=self.cov_right,
            cov_right=self.cov_left,
            overlap=self.overlap,
            columns=self.columns,
            matched=self.matched,
            span_left=self.span_right,
            span_right=self.span_left,
            score=self.score,
        )


#: An alignment between two sequences that share nothing.
EMPTY_ALIGNMENT = PairAlignment(
    identity=0.0, cov_left=0.0, cov_right=0.0, overlap=0, columns=0, matched=0,
    span_left=0, span_right=0, score=0,
)


def encode(sequence: str) -> np.ndarray:
    """Map a residue string onto BLOSUM62 alphabet indices.

    Args:
        sequence: Cleaned, upper-case amino-acid string.

    Returns:
        An ``int32`` array of alphabet indices, ``X`` for anything unscored.
    """
    if not sequence:
        return np.zeros(0, dtype=np.int32)
    return _CODE_OF[np.frombuffer(sequence.encode("ascii", "replace"), dtype=np.uint8)]


# Traceback is packed into one byte per cell so a 50-million-cell matrix costs
# 50 MB rather than 150 MB. Bits 0-1 say what produced H, bit 2 says whether the
# horizontal gap opened or extended, bit 3 the same for the vertical gap.
_H_STOP, _H_DIAG, _H_FROM_E, _H_FROM_F = 0, 1, 2, 3
_E_EXTEND_BIT = 1 << 2
_F_EXTEND_BIT = 1 << 3


def _fill(codes_a: np.ndarray, codes_b: np.ndarray, gap_open: int, gap_extend: int):
    """Run the Gotoh local-alignment recurrences, returning pointers and the best cell.

    The horizontal gap score ``E`` normally serialises the inner loop, because
    ``E[j]`` needs ``H[j-1]`` which needs ``E[j]``. It is unrolled here into a
    prefix maximum, which is exact rather than approximate::

        E[j] = max over j' < j of ( H[j'] - open - extend * (j - j') )
             = -extend * j  +  max over j' < j of ( H[j'] + extend * j' - open )

    and ``H[j']`` may be replaced by ``W[j'] = max(0, diag[j'], F[j'])`` inside
    that maximum. The only term ``W`` drops relative to ``H`` is one that already
    paid a second gap-open, and that term is dominated by the single-open term it
    descends from, so the maximum is unchanged. The whole row is then vectorised
    numpy, which is what makes a 600x1000 pair cost milliseconds.

    `reference_align_pair` implements the same recurrences the slow, obvious way
    and the test suite requires the two to agree exactly, so this unrolling is a
    checked claim rather than a comment.

    Args:
        codes_a: Encoded first sequence, length n.
        codes_b: Encoded second sequence, length m.
        gap_open: Gap-open penalty, a positive cost.
        gap_extend: Per-residue gap-extension penalty, a positive cost.

    Returns:
        ``(pointers, best_score, best_i, best_j)``. ``pointers`` is
        ``(n + 1, m + 1)`` uint8; ``best_i`` and ``best_j`` are 1-based DP indices
        of the highest-scoring cell, earliest in row-major order among ties.
    """
    n, m = len(codes_a), len(codes_b)
    pointers = np.zeros((n + 1, m + 1), dtype=np.uint8)

    # int32 rather than int64: the largest reachable score is bounded by the
    # highest BLOSUM62 diagonal (11) times the longest sequence, and halving the
    # width halves the memory traffic that dominates this loop.
    dtype = np.int32
    ramp = (gap_extend * np.arange(m + 1)).astype(dtype)
    neg_inf = dtype(-(1 << 28))
    open_extend = gap_open + gap_extend

    # The substitution scores for every residue against ALL of `codes_b`,
    # computed once. Indexing this by one residue code per row replaces a
    # two-step fancy index plus a dtype conversion inside the loop.
    profile = _SCORE[:, codes_b].astype(dtype)

    h_prev = np.zeros(m + 1, dtype=dtype)
    f_prev = np.full(m + 1, neg_inf, dtype=dtype)

    best_score, best_i, best_j = 0, 0, 0

    for i in range(1, n + 1):
        row_scores = profile[codes_a[i - 1]]

        # Vertical gaps depend only on the row above, so they vectorise as they
        # stand: open out of that row's H, or extend that row's F.
        f_open = h_prev - open_extend
        f_cur = np.maximum(f_open, f_prev - gap_extend)
        f_cur[0] = neg_inf

        diag = np.empty(m + 1, dtype=dtype)
        diag[0] = neg_inf
        np.add(h_prev[:-1], row_scores, out=diag[1:])

        # W is H without the horizontal-gap option.
        w = np.maximum(diag, f_cur)
        np.maximum(w, 0, out=w)
        w[0] = 0

        prefix = np.maximum.accumulate(w + ramp - gap_open)
        e_cur = np.empty(m + 1, dtype=dtype)
        e_cur[0] = neg_inf
        np.subtract(prefix[:-1], ramp[1:], out=e_cur[1:])

        h_cur = np.maximum(w, e_cur)

        # Traceback bits, built as one integer array and cast on store. Priority
        # is diagonal, then horizontal, then vertical, then stop, expressed by
        # the nesting so that ties resolve identically on every machine.
        row_ptr = np.where(
            h_cur > 0,
            np.where(h_cur == diag, _H_DIAG,
                     np.where(h_cur == e_cur, _H_FROM_E, _H_FROM_F)),
            _H_STOP,
        )
        # A gap counts as extended unless it could have opened at this cell.
        row_ptr[1:] += np.where(
            e_cur[1:] == h_cur[:-1] - open_extend, 0, _E_EXTEND_BIT
        )
        row_ptr += np.where(f_cur == f_open, 0, _F_EXTEND_BIT)
        pointers[i] = row_ptr

        row_argmax = int(h_cur.argmax())
        row_best = int(h_cur[row_argmax])
        if row_best > best_score:
            best_score, best_i, best_j = row_best, i, row_argmax

        h_prev, f_prev = h_cur, f_cur

    return pointers, best_score, best_i, best_j


def _walk(pointers: np.ndarray, left: str, right: str, start_i: int, start_j: int):
    """Walk the traceback from the best cell, counting matches and columns.

    Args:
        pointers: The packed pointer matrix from `_fill`.
        left: First sequence.
        right: Second sequence.
        start_i: 1-based row of the highest-scoring cell.
        start_j: 1-based column of the highest-scoring cell.

    Returns:
        ``(matched, columns, span_left, span_right)`` over the reported local
        alignment, where the spans are how many residues of each sequence the
        alignment consumed. A correct affine local alignment never begins or ends
        on a gap -- deleting such a column strictly raises the score -- so
        nothing is trimmed here and nothing needs to be.
    """
    matched = columns = span_left = span_right = 0
    i, j = start_i, start_j
    state = "H"
    while i > 0 and j > 0:
        cell = int(pointers[i, j])
        if state == "H":
            move = cell & 0b11
            if move == _H_STOP:
                break
            if move == _H_DIAG:
                if left[i - 1] == right[j - 1]:
                    matched += 1
                columns += 1
                span_left += 1
                span_right += 1
                i -= 1
                j -= 1
            elif move == _H_FROM_E:
                state = "E"
            else:
                state = "F"
        elif state == "E":
            columns += 1
            span_right += 1
            extended = bool(cell & _E_EXTEND_BIT)
            j -= 1
            state = "E" if extended else "H"
        else:
            columns += 1
            span_left += 1
            extended = bool(cell & _F_EXTEND_BIT)
            i -= 1
            state = "F" if extended else "H"
    return matched, columns, span_left, span_right


def align_pair(
    left: str,
    right: str,
    *,
    gap_open: int = GAP_OPEN,
    gap_extend: int = GAP_EXTEND,
    max_cells: Optional[int] = DEFAULT_MAX_CELLS,
) -> PairAlignment:
    """Locally align two residue strings and decompose the result.

    Smith-Waterman with affine gaps and BLOSUM62: local rather than global so a
    truncation reads as full identity over partial coverage instead of as partial
    identity, and affine rather than linear so one indel reads as one event
    instead of as many.

    Args:
        left: First residue string. May be empty.
        right: Second residue string. May be empty.
        gap_open: Gap-open cost, positive.
        gap_extend: Gap-extension cost per residue, positive.
        max_cells: Refuse pairs whose DP matrix exceeds this many cells. ``None``
            disables the guard. Refusing is deliberate -- a silently skipped
            comparison is an unmeasured leakage risk, so the caller has to see it.

    Returns:
        A `PairAlignment`. Two sequences with no positive-scoring local alignment
        return `EMPTY_ALIGNMENT`.

    Raises:
        AlignmentTooLarge: When the DP matrix would exceed ``max_cells``.
        ValueError: When a penalty is not positive.
    """
    if gap_open <= 0 or gap_extend <= 0:
        raise ValueError("affine penalties are costs and must be positive")
    if not left or not right:
        return EMPTY_ALIGNMENT
    cells = (len(left) + 1) * (len(right) + 1)
    if max_cells is not None and cells > max_cells:
        raise AlignmentTooLarge(
            f"{len(left)} x {len(right)} = {cells} cells exceeds max_cells={max_cells}"
        )

    if _swap(left, right):
        return align_pair(
            right, left,
            gap_open=gap_open, gap_extend=gap_extend, max_cells=max_cells,
        ).flipped()

    pointers, score, best_i, best_j = _fill(
        encode(left), encode(right), gap_open, gap_extend
    )
    if score <= 0:
        return EMPTY_ALIGNMENT
    matched, columns, span_left, span_right = _walk(
        pointers, left, right, best_i, best_j
    )
    if columns == 0:
        return EMPTY_ALIGNMENT
    return PairAlignment(
        identity=matched / columns,
        cov_left=span_left / len(left),
        cov_right=span_right / len(right),
        overlap=matched,
        columns=columns,
        matched=matched,
        span_left=span_left,
        span_right=span_right,
        score=score,
    )


def reference_align_pair(
    left: str,
    right: str,
    *,
    gap_open: int = GAP_OPEN,
    gap_extend: int = GAP_EXTEND,
) -> PairAlignment:
    """A deliberately naive rewrite of the same recurrences, for oracle tests.

    Three full matrices, one cell at a time, no prefix-maximum trick and no bit
    packing. It shares nothing with `align_pair` except the substitution table,
    so requiring the two to agree over randomised inputs is a real check on the
    vectorised version rather than a check of one implementation against itself.

    Args:
        left: First residue string.
        right: Second residue string.
        gap_open: Gap-open cost.
        gap_extend: Gap-extension cost.

    Returns:
        A `PairAlignment` computed the slow way.
    """
    if not left or not right:
        return EMPTY_ALIGNMENT
    if _swap(left, right):
        return reference_align_pair(
            right, left, gap_open=gap_open, gap_extend=gap_extend
        ).flipped()
    n, m = len(left), len(right)
    codes_a, codes_b = encode(left), encode(right)
    neg = float("-inf")
    h = [[0.0] * (m + 1) for _ in range(n + 1)]
    e = [[neg] * (m + 1) for _ in range(n + 1)]
    f = [[neg] * (m + 1) for _ in range(n + 1)]
    best, best_i, best_j = 0.0, 0, 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            e[i][j] = max(h[i][j - 1] - gap_open - gap_extend, e[i][j - 1] - gap_extend)
            f[i][j] = max(h[i - 1][j] - gap_open - gap_extend, f[i - 1][j] - gap_extend)
            diagonal = h[i - 1][j - 1] + float(_SCORE[codes_a[i - 1]][codes_b[j - 1]])
            h[i][j] = max(0.0, diagonal, e[i][j], f[i][j])
            if h[i][j] > best:
                best, best_i, best_j = h[i][j], i, j
    if best <= 0:
        return EMPTY_ALIGNMENT

    matched = columns = span_left = span_right = 0
    i, j, state = best_i, best_j, "H"
    while i > 0 and j > 0:
        if state == "H":
            if h[i][j] <= 0:
                break
            diagonal = h[i - 1][j - 1] + float(_SCORE[codes_a[i - 1]][codes_b[j - 1]])
            if h[i][j] == diagonal:
                if left[i - 1] == right[j - 1]:
                    matched += 1
                columns += 1
                span_left += 1
                span_right += 1
                i -= 1
                j -= 1
            elif h[i][j] == e[i][j]:
                state = "E"
            else:
                state = "F"
        elif state == "E":
            columns += 1
            span_right += 1
            opened = e[i][j] == h[i][j - 1] - gap_open - gap_extend
            j -= 1
            state = "H" if opened else "E"
        else:
            columns += 1
            span_left += 1
            opened = f[i][j] == h[i - 1][j] - gap_open - gap_extend
            i -= 1
            state = "H" if opened else "F"
    if columns == 0:
        return EMPTY_ALIGNMENT
    return PairAlignment(
        identity=matched / columns,
        cov_left=span_left / n,
        cov_right=span_right / m,
        overlap=matched,
        columns=columns,
        matched=matched,
        span_left=span_left,
        span_right=span_right,
        score=int(best),
    )
