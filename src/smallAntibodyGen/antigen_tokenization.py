from __future__ import annotations

"""
Antigen-stream tokenization adapters.

Direction 1 (hybrid PLM antigen encoder) lets the antigen stream be encoded either
by the repository's small amino-acid tokenizer (``"scratch"``) or by a pretrained
ESM-2 tokenizer (``"esm"``). Training (the collator), generation
(``FixedLengthHCDR3Infiller``), and scoring (``AntigenCompatibilityScorer``) all encode
the antigen independently, so they MUST share one tokenization definition or the model
would see a different antigen distribution at train vs inference time. This module is
that single definition: build one adapter and hand it to all three call sites.

The scratch adapter reproduces the previous behavior exactly
(``tokenizer.encode_sequence(antigen, locus=None, max_length=...)``), so switching the
call sites onto the adapter is a no-op for the from-scratch model.
"""

from abc import ABC, abstractmethod
from typing import List

from smallAntibodyGen.tokenizer import AminoAcidTokenizer


class AntigenTokenizer(ABC):
    """Interface for encoding one antigen amino-acid sequence into token ids."""

    @property
    @abstractmethod
    def pad_id(self) -> int:
        """Padding token id used when batching this stream."""

    @abstractmethod
    def encode(self, sequence: str, max_length: int) -> List[int]:
        """
        Encode one antigen sequence into token ids (with special tokens), truncated
        to at most ``max_length`` tokens. Token index 0 must be a CLS-like summary
        token, because the model reads ``antigen_hidden[:, 0, :]`` as the antigen
        summary in ``joint_representation``.
        """


class ScratchAntigenTokenizer(AntigenTokenizer):
    """
    Adapter over the repository amino-acid tokenizer.

    This is byte-for-byte equivalent to the previous inline call
    ``tokenizer.encode_sequence(antigen, locus=None, max_length=...)`` so the
    from-scratch dual-stream model is unaffected by routing through the adapter.
    """

    def __init__(self, tokenizer: AminoAcidTokenizer) -> None:
        self._tokenizer = tokenizer

    @property
    def pad_id(self) -> int:
        return self._tokenizer.pad_id

    def encode(self, sequence: str, max_length: int) -> List[int]:
        return self._tokenizer.encode_sequence(
            sequence or "",
            locus=None,
            max_length=max_length,
        )


class ESMAntigenTokenizer(AntigenTokenizer):
    """
    Adapter over a HuggingFace ESM-2 tokenizer.

    ``transformers`` is imported lazily so the base install and every from-scratch
    stage never require it. The ESM tokenizer tokenizes per residue and wraps the
    sequence as ``<cls> ... <eos>``; ``<cls>`` at index 0 supplies the antigen
    summary token the fusion layer expects.
    """

    def __init__(self, esm_model_name: str) -> None:
        try:
            from transformers import AutoTokenizer  # lazy: optional `esm` extra
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "antigen_encoder_type='esm' requires the optional 'esm' extra. "
                "Install it with `pip install -e \".[esm]\"` (transformers + peft)."
            ) from exc
        self.esm_model_name = esm_model_name
        self._hf_tokenizer = AutoTokenizer.from_pretrained(esm_model_name)

    @property
    def pad_id(self) -> int:
        return int(self._hf_tokenizer.pad_token_id)

    def encode(self, sequence: str, max_length: int) -> List[int]:
        encoded = self._hf_tokenizer(
            (sequence or "").upper().strip(),
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
        )
        return list(encoded["input_ids"])


def build_antigen_tokenizer(
    antigen_encoder_type: str,
    tokenizer: AminoAcidTokenizer,
    esm_model_name: str,
) -> AntigenTokenizer:
    """
    Construct the antigen tokenization adapter for a given encoder type.

    Args:
        antigen_encoder_type: ``"scratch"`` or ``"esm"``.
        tokenizer: The repository amino-acid tokenizer (used by the scratch path).
        esm_model_name: HuggingFace model id (used by the ESM path).

    Returns:
        An ``AntigenTokenizer`` matching ``antigen_encoder_type``.

    Raises:
        ValueError: If ``antigen_encoder_type`` is unknown.
    """
    if antigen_encoder_type == "scratch":
        return ScratchAntigenTokenizer(tokenizer)
    if antigen_encoder_type == "esm":
        return ESMAntigenTokenizer(esm_model_name)
    raise ValueError(
        f"unknown antigen_encoder_type {antigen_encoder_type!r}; expected 'scratch' or 'esm'"
    )
