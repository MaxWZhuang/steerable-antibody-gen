"""The single home of the encoder token-layout arithmetic and the truncation census.

Why this module exists
----------------------
"How many tokens does this record become, and where inside them does each CDR3
land" is asked in four places that were never tied together:

- the TOKENIZER, ``AminoAcidTokenizer.encode_sequence`` /
  ``encode_paired_sequences``, which is the only authority -- it is what the
  collator actually feeds the model;
- the READER fallback, ``OASSequenceDataset._load``, which reconstructs
  ``token_length`` arithmetically (``len(heavy) + len(light) + 5``, or
  ``len(sequence) + 3``) whenever the producer did not write the field --
  ``prepare_antibody_antigen.py`` never does;
- the PRODUCER, ``prepare_oas.py``, which writes the same ``+ 5`` for paired rows;
- the PREFLIGHT, :func:`summarize_length_truncation`, which needs CDR3 *token*
  offsets to say whether a truncation deleted a CDR3.

Only the first of those is ground truth. The other three are reconstructions of
it, and a reconstruction that drifts is invisible: it reports a length nobody
checks against the tokenizer. This module gives the reconstruction one name per
quantity (:func:`single_chain_token_length`, :func:`paired_token_length`,
:func:`heavy_cdr3_token_end`, :func:`light_cdr3_token_end`) so the drift has a
place to be tested against the tokenizer instead of being retyped per call site.
``scripts/context_length_census.py`` does exactly that: it uses these helpers for
the CDR3 offsets but takes the token LENGTH from the real tokenizer, and pins the
two against each other.

The layout the constants encode
-------------------------------
``encode_sequence``::

    [CLS] [chain] r0 r1 ... r{n-1} [EOS]          -> n + 3 tokens

``encode_paired_sequences``::

    [CLS] [IGH] h0 ... h{n-1} [SEP] [IGK] l0 ... l{m-1} [EOS]   -> n + m + 5 tokens

Both truncate with a hard ``tokens[:max_length]`` and then overwrite the final
token with ``[EOS]``, so a residue at token index ``>= max_length`` is simply
gone -- and so is the one at ``max_length - 1``, which the forced ``[EOS]``
overwrites.
"""
from __future__ import annotations

from typing import Any

# [CLS] and the chain token, ahead of the first residue of the first chain.
LEADING_SPECIAL_TOKENS = 2
# The trailing [EOS].
TRAILING_SPECIAL_TOKENS = 1
# [SEP] and the light chain token, between the heavy and light residues.
PAIRED_SEPARATOR_TOKENS = 2

SINGLE_CHAIN_SPECIAL_TOKENS = LEADING_SPECIAL_TOKENS + TRAILING_SPECIAL_TOKENS  # 3
PAIRED_SPECIAL_TOKENS = (
    LEADING_SPECIAL_TOKENS + PAIRED_SEPARATOR_TOKENS + TRAILING_SPECIAL_TOKENS
)  # 5


def single_chain_token_length(residue_count: int) -> int:
    """Token length of ``[CLS] [chain] residues... [EOS]``."""
    return residue_count + SINGLE_CHAIN_SPECIAL_TOKENS


def paired_token_length(heavy_residue_count: int, light_residue_count: int) -> int:
    """Token length of ``[CLS] [IGH] heavy... [SEP] [IGK] light... [EOS]``."""
    return heavy_residue_count + light_residue_count + PAIRED_SPECIAL_TOKENS


def heavy_cdr3_token_end(cdr3_end_aa: int) -> int:
    """
    Exclusive token index just past the heavy CDR3.

    The heavy chain's residues start at token index
    ``LEADING_SPECIAL_TOKENS`` in BOTH layouts, so this is the same expression
    for a single heavy chain and for the heavy half of a pair.
    """
    return LEADING_SPECIAL_TOKENS + cdr3_end_aa


def light_cdr3_token_end(heavy_residue_count: int, cdr3_end_aa: int) -> int:
    """
    Exclusive token index just past the light CDR3 of a PAIRED record.

    ``[CLS] [IGH]`` + every heavy residue + ``[SEP] [light chain]`` precede the
    light chain's first residue.
    """
    return (
        LEADING_SPECIAL_TOKENS
        + heavy_residue_count
        + PAIRED_SEPARATOR_TOKENS
        + cdr3_end_aa
    )


def summarize_length_truncation(
    dataset: Any,
    max_length: int,
) -> dict[str, int]:
    """
    How much of this corpus the encoder's ``max_length`` silently deletes.

    ``prepare_oas.py`` bounds the heavy and light chains independently
    (``--max-heavy`` / ``--max-light``, up to 180 + 160 + 5 = 345 tokens) and
    writes ``token_length`` unclamped. Nothing ties the corpus to any encoder
    budget, so the collator hard-truncates to ``max_length`` and forces a trailing
    ``[EOS]``. The tokenizer's UserWarning is deduplicated by Python's default
    filter, so at corpus scale this is effectively silent.

    It is not a rounding error. On the shipped paired corpus at ``max_length: 192``,
    99.97% of rows overflow and 99.77% lose their LIGHT CDR3 entirely -- and the
    paired stage's whole purpose is heavy/light compatibility, for which CDR-L3 is
    the most informative region. The shuffled-negative task then asks the model to
    tell native from non-cognate light chains with CDR-L3 deleted from both.

    Counts are computed arithmetically from stored coordinates (cheap, exact for
    the layouts the collator produces) rather than by encoding every row:

    - single-chain / antigen: ``[CLS] [chain] residues... [EOS]``, heavy CDR3 at
      token offset ``2 + aa_index``
    - paired: ``[CLS] [IGH] heavy... [SEP] [IGK] light... [EOS]``, light CDR3 at
      token offset ``2 + len(heavy) + 2 + aa_index``

    ``dataset`` is anything exposing ``.records`` (``OASSequenceDataset``,
    ``RecordSubsetDataset``, or any thin wrapper over a list of ``OASRecord``).

    Returns counts only; the caller decides whether to warn or stop. Raising here
    would be wrong -- whether to raise ``max_length`` (more compute, a retrain),
    filter the corpus, or accept the truncation is a research decision.

    For "what context length would we need", see ``scripts/context_length_census.py``:
    this function answers only "what does the CURRENT max_length cost", and it
    answers it from stored coordinates rather than from the tokenizer.
    """
    total = 0
    overflow = 0
    lost_heavy_cdr3 = 0
    lost_light_cdr3 = 0
    worst_overflow = 0
    for record in dataset.records:
        total += 1
        token_length = record.token_length or 0
        if token_length > max_length:
            overflow += 1
            worst_overflow = max(worst_overflow, token_length - max_length)

        heavy_seq = record.sequence_heavy or record.sequence or ""
        light_seq = record.sequence_light or ""
        is_paired = bool(heavy_seq and light_seq)

        heavy_end = record.cdr3_end_aa_heavy
        if heavy_end is None and not is_paired:
            heavy_end = record.cdr3_end_aa
        if isinstance(heavy_end, int) and heavy_cdr3_token_end(heavy_end) > max_length:
            lost_heavy_cdr3 += 1

        if is_paired:
            light_end = record.cdr3_end_aa_light
            if isinstance(light_end, int):
                # 2 leading specials + heavy residues + [SEP] + [light chain token]
                if light_cdr3_token_end(len(heavy_seq), light_end) > max_length:
                    lost_light_cdr3 += 1

    return {
        "total": total,
        "overflow": overflow,
        "worst_overflow": worst_overflow,
        "lost_heavy_cdr3": lost_heavy_cdr3,
        "lost_light_cdr3": lost_light_cdr3,
    }


def format_length_truncation_warning(
    counts: dict[str, int],
    max_length: int,
    split_name: str,
) -> str | None:
    """
    Render the truncation report, or ``None`` when nothing overflows.

    Always names the CDR3 losses explicitly. A bare "N rows truncated" reads as
    trimming a few framework residues at the C-terminus; losing the CDR3 is a
    different claim entirely, and it is the one that invalidates the objective.
    """
    total = counts["total"]
    if total == 0 or counts["overflow"] == 0:
        return None
    pct = 100.0 * counts["overflow"] / total
    lines = [
        f"[warn] {split_name}: {counts['overflow']}/{total} rows ({pct:.2f}%) exceed "
        f"max_length={max_length} and are TRUNCATED by the collator "
        f"(worst overflow: {counts['worst_overflow']} tokens)."
    ]
    if counts["lost_light_cdr3"]:
        pct_l = 100.0 * counts["lost_light_cdr3"] / total
        lines.append(
            f"[warn]   {counts['lost_light_cdr3']} ({pct_l:.2f}%) lose their LIGHT CDR3 "
            "entirely -- the heavy/light pairing objective is being trained without "
            "CDR-L3, the region that most determines pairing."
        )
    if counts["lost_heavy_cdr3"]:
        pct_h = 100.0 * counts["lost_heavy_cdr3"] / total
        lines.append(
            f"[warn]   {counts['lost_heavy_cdr3']} ({pct_h:.2f}%) lose their HEAVY CDR3 "
            "entirely -- these rows cannot train the HCDR3 objective at all."
        )
    lines.append(
        "[warn]   Fix by raising max_length (a retrain, and more compute per step), "
        "tightening prepare_oas.py's --max-heavy/--max-light, or filtering rows "
        "whose token_length exceeds max_length. Truncation is silent otherwise: "
        "Python dedupes the tokenizer's UserWarning."
    )
    return "\n".join(lines)
