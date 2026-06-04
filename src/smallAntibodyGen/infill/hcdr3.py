from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from smallAntibodyGen.tokenizer import AminoAcidTokenizer


CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"


@dataclass(frozen=True)
class HCDR3Span:
    """
    Heavy-chain CDR3 coordinates for one antibody record.

    The amino-acid coordinates are zero-based and end-exclusive relative to the
    cleaned heavy-chain variable-domain sequence. The token coordinates follow
    the same convention after tokenization. For the repository tokenizer, heavy
    residues begin at token index 2 because antibody streams start with
    ``[CLS]`` followed by a chain token. The token fields are optional because
    variable-length proposal can be planned in amino-acid space first and only
    materialized into tokens when an infiller builds masked model inputs.
    """

    aa_start: int
    aa_end: int
    original_hcdr3: str
    token_start: int | None = None
    token_end: int | None = None

    @property
    def length(self) -> int:
        """Return the known HCDR3 length in amino-acid residues."""
        return self.aa_end - self.aa_start

    @classmethod
    def from_record(cls, record: Any) -> "HCDR3Span":
        """
        Extract heavy-chain CDR3 coordinates from an OAS/ASD-style record.

        Heavy-specific fields are preferred. Generic CDR3 fields are accepted
        as a fallback for older heavy-chain records. The method raises a
        ``ValueError`` instead of returning a partial object because infilling
        needs a precise location where residues can be removed and replaced by
        mask tokens.
        """
        start = getattr(record, "cdr3_start_aa_heavy", None)
        end = getattr(record, "cdr3_end_aa_heavy", None)
        cdr3 = getattr(record, "cdr3_aa_heavy", None)
        if start is None or end is None:
            start = getattr(record, "cdr3_start_aa", None)
            end = getattr(record, "cdr3_end_aa", None)
            cdr3 = getattr(record, "cdr3_aa", None)
        if not isinstance(start, int) or not isinstance(end, int) or end <= start:
            raise ValueError("record does not contain a valid heavy-chain HCDR3 span")
        if not isinstance(cdr3, str) or len(cdr3) != (end - start):
            raise ValueError("record HCDR3 string is missing or inconsistent with span coordinates")
        return cls(aa_start=start, aa_end=end, original_hcdr3=cdr3)


@dataclass(frozen=True)
class HCDR3InfillCandidate:
    """
    One generated HCDR3 replacement and the antibody sequence it creates.

    ``generated_hcdr3`` is only the proposed CDR3 loop. ``heavy_sequence`` is
    the full heavy-chain variable sequence after replacing the original HCDR3
    with that proposal. ``log_probability`` is the sum of per-position log
    probabilities assigned by the MLM at the sampled mask positions, so it is
    useful for ranking candidates from the same context but should not be
    interpreted as a calibrated biological binding score. ``compatibility_score``
    is optional and comes from a separate antibody-antigen compatibility model.
    """

    generated_hcdr3: str
    heavy_sequence: str
    log_probability: float
    length: int
    compatibility_score: float | None = None


class LengthProposalStrategy(ABC):
    """
    Interface for unknown-length HCDR3 design.

    The fixed-length infiller needs to know how many ``[MASK]`` tokens to place
    between the heavy-chain framework prefix and suffix. A length proposal
    strategy owns that decision. The first implementation is an empirical
    positive-binder prior, but this interface is intentionally small so later
    work can swap in a learned length predictor without changing the residue
    infiller or generation CLI.
    """

    @abstractmethod
    def propose_lengths(
        self,
        record: Any,
        *,
        num_lengths: int,
        rng: random.Random,
    ) -> list[int]:
        """Return candidate HCDR3 lengths for one antibody-antigen context."""


class EmpiricalHCDR3LengthPrior(LengthProposalStrategy):
    """
    Sample HCDR3 lengths from an observed positive-binder histogram.

    This is deliberately simple and transparent. It does not claim that length
    is antigen-specific; it gives the variable-length infrastructure a usable
    baseline while preserving the fixed-length residue infiller. Records with
    ``binder_label != 1`` are ignored by default so the prior describes the
    same positive-binder population used for HCDR3 infill fine-tuning.
    """

    def __init__(self, length_counts: Counter[int]) -> None:
        if not length_counts:
            raise ValueError("EmpiricalHCDR3LengthPrior requires at least one observed length")
        self.length_counts = Counter({int(k): int(v) for k, v in length_counts.items() if int(k) > 0 and int(v) > 0})
        if not self.length_counts:
            raise ValueError("EmpiricalHCDR3LengthPrior received no positive lengths")
        self._lengths = sorted(self.length_counts)
        self._weights = [self.length_counts[length] for length in self._lengths]

    @classmethod
    def fit(
        cls,
        records: Sequence[Any],
        *,
        positive_only: bool = True,
    ) -> "EmpiricalHCDR3LengthPrior":
        """
        Fit the empirical length prior from dataset records.

        Args:
            records:
                In-memory dataset records. Each record should expose HCDR3 span
                fields compatible with ``HCDR3Span.from_record``.
            positive_only:
                When True, only rows with ``binder_label == 1`` contribute to
                the histogram.

        Returns:
            An empirical length prior that can be passed to the generation CLI
            or used directly by downstream scripts.
        """
        counts: Counter[int] = Counter()
        for record in records:
            if positive_only and getattr(record, "binder_label", None) != 1:
                continue
            try:
                span = HCDR3Span.from_record(record)
            except ValueError:
                continue
            counts[span.length] += 1
        return cls(counts)

    def propose_lengths(
        self,
        record: Any,
        *,
        num_lengths: int,
        rng: random.Random,
    ) -> list[int]:
        """
        Draw lengths with replacement from the fitted histogram.

        The ``record`` argument is accepted to satisfy the shared interface even
        though this baseline is context-independent.
        """
        if num_lengths <= 0:
            return []
        return rng.choices(self._lengths, weights=self._weights, k=num_lengths)


class FixedLengthHCDR3Infiller:
    """
    Generate HCDR3 residues by masking a fixed number of positions.

    This class turns a record into the exact model input used by fixed-length
    HCDR3 design: keep the heavy-chain framework, optional light chain, and
    antigen sequence visible; replace the HCDR3 interval with ``length`` mask
    tokens; then sample one amino-acid residue for each mask position from the
    model's antibody MLM logits. The class can also accept a proposed length
    that differs from the known record length, which is how the initial
    unknown-length infrastructure works: choose a length first, then reuse this
    same fixed-length masked-residue sampler.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: AminoAcidTokenizer,
        *,
        max_length: int,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = torch.device(device)
        self.canonical_token_ids = [
            self.tokenizer.token_to_id[aa]
            for aa in CANONICAL_AA
            if aa in self.tokenizer.token_to_id
        ]

    def _residue_ids(self, sequence: str) -> list[int]:
        """Encode residues without adding special tokens."""
        return [
            self.tokenizer.token_to_id.get(aa, self.tokenizer.unk_id)
            for aa in (sequence or "").upper().strip()
        ]

    def _heavy_sequence(self, record: Any) -> str:
        """Return the heavy-chain sequence used as the editable scaffold."""
        heavy_sequence = (getattr(record, "sequence_heavy", None) or getattr(record, "sequence", "") or "").strip()
        if not heavy_sequence:
            raise ValueError("record has no heavy-chain sequence")
        return heavy_sequence

    def _encode_antibody_with_masked_hcdr3(
        self,
        record: Any,
        span: HCDR3Span,
        *,
        proposed_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], str, str]:
        """
        Build one antibody token stream with a masked HCDR3 interval.

        Returns the token IDs, attention mask, token positions occupied by the
        proposed HCDR3 masks, and the heavy-chain prefix/suffix strings needed
        to reconstruct the generated full heavy sequence. A proposed length can
        differ from the known span length because the prefix and suffix are
        defined by the numbered HCDR3 boundaries rather than by the generated
        residue count.
        """
        if proposed_length <= 0:
            raise ValueError("proposed_length must be positive")
        heavy_sequence = self._heavy_sequence(record)
        if span.aa_end > len(heavy_sequence):
            raise ValueError("HCDR3 span extends beyond the heavy-chain sequence")

        prefix = heavy_sequence[:span.aa_start]
        suffix = heavy_sequence[span.aa_end:]
        light_sequence = (getattr(record, "sequence_light", None) or "").strip()
        heavy_locus = getattr(record, "heavy_locus", None) or getattr(record, "locus", None) or "IGH"
        light_locus = getattr(record, "light_locus", None) or "IGK"

        tokens = [
            self.tokenizer.cls_id,
            self.tokenizer.token_to_id[self.tokenizer.get_chain_token(heavy_locus)],
        ]
        tokens.extend(self._residue_ids(prefix))
        mask_positions = list(range(len(tokens), len(tokens) + proposed_length))
        tokens.extend([self.tokenizer.mask_id] * proposed_length)
        tokens.extend(self._residue_ids(suffix))
        if light_sequence:
            tokens.append(self.tokenizer.sep_id)
            tokens.append(self.tokenizer.token_to_id[self.tokenizer.get_chain_token(light_locus)])
            tokens.extend(self._residue_ids(light_sequence))
        tokens.append(self.tokenizer.eos_id)

        if len(tokens) > self.max_length:
            raise ValueError(
                "masked antibody input exceeds max_length; increase max_length or use a shorter proposed length"
            )
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=self.device)
        return input_ids, attention_mask, mask_positions, prefix, suffix

    def _encode_antigen(self, record: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the antigen stream for the dual-stream model."""
        antigen_sequence = (getattr(record, "sequence_antigen", None) or "").strip()
        if not antigen_sequence:
            raise ValueError("record has no antigen sequence")
        ids = self.tokenizer.encode_sequence(antigen_sequence, locus=None, max_length=self.max_length)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=self.device)
        return input_ids, attention_mask

    def _sample_token_id(
        self,
        logits: torch.Tensor,
        *,
        temperature: float,
        top_k: int | None,
    ) -> tuple[int, float]:
        """
        Sample one canonical amino-acid token from MLM logits.

        Returns the sampled token ID and its log probability under the filtered
        distribution used for sampling.
        """
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        candidate_logits = logits[self.canonical_token_ids] / temperature
        if top_k is not None and top_k > 0 and top_k < candidate_logits.numel():
            top_values, top_indices = torch.topk(candidate_logits, k=top_k)
            probs = torch.softmax(top_values, dim=-1)
            sampled_rank = torch.multinomial(probs, num_samples=1).item()
            token_id = self.canonical_token_ids[int(top_indices[sampled_rank].item())]
            log_prob = float(torch.log(probs[sampled_rank]).item())
            return token_id, log_prob

        probs = torch.softmax(candidate_logits, dim=-1)
        sampled_idx = torch.multinomial(probs, num_samples=1).item()
        token_id = self.canonical_token_ids[sampled_idx]
        log_prob = float(torch.log(probs[sampled_idx]).item())
        return token_id, log_prob

    @torch.no_grad()
    def infill(
        self,
        record: Any,
        *,
        length: int | None = None,
        num_samples: int = 1,
        temperature: float = 1.0,
        top_k: int | None = None,
        scorer: "AntigenCompatibilityScorer | None" = None,
    ) -> list[HCDR3InfillCandidate]:
        """
        Generate HCDR3 candidates for one antibody-antigen record.

        Args:
            record:
                Dataset record containing heavy sequence, antigen sequence, and
                known HCDR3 coordinates.
            length:
                Number of HCDR3 residues to generate. When omitted, the known
                record span length is used, which is fixed-length infilling.
            num_samples:
                Number of independent candidates to sample.
            temperature:
                Softmax temperature for residue sampling.
            top_k:
                Optional top-k filter over canonical amino acids at each masked
                position.
            scorer:
                Optional compatibility scorer used to attach binder
                probabilities to generated candidates.

        Returns:
            List of generated candidates.
        """
        if num_samples <= 0:
            return []
        span = HCDR3Span.from_record(record)
        proposed_length = span.length if length is None else int(length)
        antigen_input_ids, antigen_attention_mask = self._encode_antigen(record)
        self.model.eval()

        candidates: list[HCDR3InfillCandidate] = []
        for _ in range(num_samples):
            antibody_input_ids, antibody_attention_mask, mask_positions, prefix, suffix = (
                self._encode_antibody_with_masked_hcdr3(
                    record,
                    span,
                    proposed_length=proposed_length,
                )
            )
            mlm_logits, _ = self.model(
                antibody_input_ids=antibody_input_ids,
                antibody_attention_mask=antibody_attention_mask,
                antigen_input_ids=antigen_input_ids,
                antigen_attention_mask=antigen_attention_mask,
            )

            sampled_token_ids: list[int] = []
            log_probability = 0.0
            for pos in mask_positions:
                token_id, log_prob = self._sample_token_id(
                    mlm_logits[0, pos, :],
                    temperature=temperature,
                    top_k=top_k,
                )
                sampled_token_ids.append(token_id)
                log_probability += log_prob

            generated_hcdr3 = "".join(self.tokenizer.id_to_token[token_id] for token_id in sampled_token_ids)
            heavy_sequence = prefix + generated_hcdr3 + suffix
            compatibility_score = scorer.score(record, heavy_sequence=heavy_sequence) if scorer is not None else None
            candidates.append(
                HCDR3InfillCandidate(
                    generated_hcdr3=generated_hcdr3,
                    heavy_sequence=heavy_sequence,
                    log_probability=log_probability,
                    length=proposed_length,
                    compatibility_score=compatibility_score,
                )
            )
        return candidates


class AntigenCompatibilityScorer:
    """
    Score generated antibody-antigen pairs with a compatibility classifier.

    The scorer is intentionally separate from the infiller because the HCDR3 MLM
    objective and the binder-vs-non-binder objective answer different questions.
    The infiller proposes residue sequences; this scorer attaches the current
    model's estimated binder probability so downstream scripts can rank or
    filter candidates without mixing compatibility loss into positive-only
    HCDR3 reconstruction training.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: AminoAcidTokenizer,
        *,
        max_length: int,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = torch.device(device)

    def _encode_antibody(self, record: Any, heavy_sequence: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a generated heavy sequence plus the record's light chain."""
        light_sequence = (getattr(record, "sequence_light", None) or "").strip()
        if light_sequence:
            ids = self.tokenizer.encode_paired_sequences(
                heavy_sequence=heavy_sequence,
                light_sequence=light_sequence,
                heavy_locus=getattr(record, "heavy_locus", None) or "IGH",
                light_locus=getattr(record, "light_locus", None) or "IGK",
                max_length=self.max_length,
            )
        else:
            ids = self.tokenizer.encode_sequence(
                heavy_sequence,
                locus=getattr(record, "heavy_locus", None) or "IGH",
                max_length=self.max_length,
            )
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=self.device)
        return input_ids, attention_mask

    def _encode_antigen(self, record: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the antigen sequence paired with a generated antibody."""
        antigen_sequence = (getattr(record, "sequence_antigen", None) or "").strip()
        if not antigen_sequence:
            raise ValueError("record has no antigen sequence")
        ids = self.tokenizer.encode_sequence(antigen_sequence, locus=None, max_length=self.max_length)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=self.device)
        return input_ids, attention_mask

    @torch.no_grad()
    def score(self, record: Any, *, heavy_sequence: str) -> float:
        """
        Return the classifier probability for the binder/compatible class.
        """
        self.model.eval()
        antibody_input_ids, antibody_attention_mask = self._encode_antibody(record, heavy_sequence)
        antigen_input_ids, antigen_attention_mask = self._encode_antigen(record)
        _, compatibility_logits = self.model(
            antibody_input_ids=antibody_input_ids,
            antibody_attention_mask=antibody_attention_mask,
            antigen_input_ids=antigen_input_ids,
            antigen_attention_mask=antigen_attention_mask,
        )
        return float(torch.softmax(compatibility_logits, dim=-1)[0, 1].item())
