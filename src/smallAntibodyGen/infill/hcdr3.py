from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from smallAntibodyGen.tokenizer import AminoAcidTokenizer
from smallAntibodyGen.antigen_tokenization import build_antigen_tokenizer


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


def encode_masked_hcdr3_ids(
    tokenizer: AminoAcidTokenizer,
    record: Any,
    span: HCDR3Span,
    *,
    proposed_length: int,
) -> tuple[list[int], list[int], str, str]:
    """
    Build the masked-HCDR3 antibody token stream for one record.

    This is the SINGLE definition of the masked-HCDR3 antibody encoding. Two
    parties must agree on it byte-for-byte: the generation infiller
    (``FixedLengthHCDR3Infiller._encode_antibody_with_masked_hcdr3``) and the
    training collator's length-query encoding. Rather than re-implement the token
    layout in each place -- the shape of bug that only surfaces as a quiet
    train/inference mismatch -- both call this function, and a parity test pins
    that the collator's length-query bytes equal this function's output at
    ``proposed_length=1``.

    The layout keeps ``[CLS] [heavy-chain token]`` + prefix residues, then exactly
    ``proposed_length`` ``[MASK]`` tokens for the HCDR3, then the suffix residues,
    an optional ``[SEP] [light-chain token]`` + light residues, and ``[EOS]``. It
    is the standard train/inference encoding with the HCDR3 interval replaced by
    mask tokens, so it is byte-identical to the un-masked encoding everywhere
    except the span.

    Chain-token resolution matches the training collator's, which branches on
    pairing: on a PAIRED record ``locus`` is a record-type marker
    ("PAIRED_ANTIGEN"), not a chain identity, so consulting it would encode the
    heavy chain as ``[OTHER_CHAIN]``; on a single-chain record ``locus`` IS the
    chain identity and must be honored.

    Returns ``(token_ids, mask_positions, prefix, suffix)``. No ``max_length``
    check and no tensor wrapping -- the caller owns those, because the collator
    pads across a batch while the infiller checks overflow and builds device
    tensors. Raises ``ValueError`` for a non-positive ``proposed_length``, a
    missing heavy sequence, or a span that runs past that sequence.
    """
    if proposed_length <= 0:
        raise ValueError("proposed_length must be positive")
    heavy_sequence = (
        getattr(record, "sequence_heavy", None) or getattr(record, "sequence", "") or ""
    ).strip()
    if not heavy_sequence:
        raise ValueError("record has no heavy-chain sequence")
    if span.aa_end > len(heavy_sequence):
        raise ValueError("HCDR3 span extends beyond the heavy-chain sequence")

    prefix = heavy_sequence[: span.aa_start]
    suffix = heavy_sequence[span.aa_end :]
    light_sequence = (getattr(record, "sequence_light", None) or "").strip()
    heavy_locus = getattr(record, "heavy_locus", None)
    if not light_sequence:
        heavy_locus = heavy_locus or getattr(record, "locus", None)
    heavy_locus = heavy_locus or "IGH"
    light_locus = getattr(record, "light_locus", None) or "IGK"

    def _residue_ids(sequence: str) -> list[int]:
        return [
            tokenizer.token_to_id.get(aa, tokenizer.unk_id)
            for aa in (sequence or "").upper().strip()
        ]

    tokens = [
        tokenizer.cls_id,
        tokenizer.token_to_id[tokenizer.get_chain_token(heavy_locus)],
    ]
    tokens.extend(_residue_ids(prefix))
    mask_positions = list(range(len(tokens), len(tokens) + proposed_length))
    tokens.extend([tokenizer.mask_id] * proposed_length)
    tokens.extend(_residue_ids(suffix))
    if light_sequence:
        tokens.append(tokenizer.sep_id)
        tokens.append(tokenizer.token_to_id[tokenizer.get_chain_token(light_locus)])
        tokens.extend(_residue_ids(light_sequence))
    tokens.append(tokenizer.eos_id)
    return tokens, mask_positions, prefix, suffix


@dataclass(frozen=True)
class HCDR3InfillCandidate:
    """
    One generated HCDR3 replacement and the antibody sequence it creates.

    ``generated_hcdr3`` is only the proposed CDR3 loop. ``heavy_sequence`` is
    the full heavy-chain variable sequence after replacing the original HCDR3
    with that proposal. ``log_probability`` is the sum of per-position log
    probabilities assigned by the MLM (under its true, unfiltered, temperature-1
    distribution) at the sampled mask positions. Because that sum grows with
    length, it must not be compared across candidates of different lengths;
    ``mean_log_probability`` is the per-position mean and is the length-comparable
    ranking score. Neither should be read as a calibrated binding score.
    ``compatibility_score`` is optional and comes from a separate
    antibody-antigen compatibility model.
    """

    generated_hcdr3: str
    heavy_sequence: str
    log_probability: float
    mean_log_probability: float
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
    baseline while preserving the fixed-length residue infiller. Non-strong
    binders are ignored by default so the prior describes the same
    ``is_strong_binder`` population that ``is_hcdr3_infill_record`` selects for
    HCDR3 infill fine-tuning.
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
                When True, only rows with ``is_strong_binder`` contribute to
                the histogram. This matches the HCDR3 infill training gate in
                ``is_hcdr3_infill_record`` so proposed lengths come from the
                same observed-binder population the infiller learned, rather
                than only the ``affinity_type == "bool"`` subset that carries a
                ``binder_label``.

        Returns:
            An empirical length prior that can be passed to the generation CLI
            or used directly by downstream scripts.
        """
        counts: Counter[int] = Counter()
        for record in records:
            if positive_only and not getattr(record, "is_strong_binder", False):
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


class LearnedLengthProposal(LengthProposalStrategy):
    """
    Propose HCDR3 lengths from a trained categorical length head.

    Unlike ``EmpiricalHCDR3LengthPrior`` (a context-free histogram over the
    training corpus), this reads the model's *conditional* posterior
    ``p(L | scaffold, antigen)`` for the specific record. That is the difference
    between "HCDR3s are usually 12-17 residues" and "THIS framework against THIS
    antigen wants 15".

    It builds the COLLAPSED-span query -- the HCDR3 interval replaced by a single
    ``[MASK]``, so the head can never read L off a mask count -- through the
    infiller's own encoder path, so the query is byte-identical to what the head
    saw in training. The infiller must wrap a model built with
    ``use_length_head=True``; otherwise construction fails loudly.

    Feasibility renormalization: a proposed length L is feasible only when the
    fixed-length masked encoding for L fits the infiller's ``max_length``. The
    posterior is filtered to feasible lengths and renormalized to sum to 1 before
    sampling, mirroring the CLI's per-length overflow skip, so no probability mass
    is spent on lengths that could never generate.
    """

    def __init__(
        self,
        infiller: "FixedLengthHCDR3Infiller",
        *,
        length_head_max: int,
        mode: str = "sample",
    ) -> None:
        length_head = getattr(infiller.model, "length_head", None)
        if length_head is None:
            raise ValueError(
                "LearnedLengthProposal requires an infiller whose model was built "
                "with use_length_head=True (a length head must exist)."
            )
        if length_head_max <= 0:
            raise ValueError("length_head_max must be > 0")
        # Which value wins: the CHECKPOINT's head size is ground truth -- the
        # posterior has exactly `out_features` classes, so a larger
        # length_head_max cannot be honored and would die deep inside
        # propose_lengths with a bare IndexError. Fail loudly at construction
        # instead. A SMALLER length_head_max stays a deliberate, well-defined cap:
        # the support is truncated to 1..length_head_max and renormalized.
        head_size = int(length_head.out_features)
        if length_head_max > head_size:
            raise ValueError(
                f"length_head_max={length_head_max} exceeds the model's length-head "
                f"size {head_size} (the checkpoint's length_head_max). The checkpoint "
                "wins: pass a value <= the head size (the CLI threads the checkpoint's "
                "own config, which always matches)."
            )
        if mode not in {"sample", "top_k"}:
            raise ValueError("mode must be one of: sample, top_k")
        self.infiller = infiller
        self.length_head_max = int(length_head_max)
        self.mode = mode

    def _feasible_max_length(self, record: Any, span: HCDR3Span) -> int:
        """
        Largest HCDR3 length whose masked encoding still fits ``max_length``.

        The masked stream length is ``fixed + L`` where ``fixed`` is everything but
        the span masks; we recover ``fixed`` from the collapsed (L=1) encoding
        (``len(tokens) - 1``), so the largest feasible L is ``max_length - fixed``.
        Returns 0 when even a single-residue span overflows.
        """
        tokens, _, _, _ = encode_masked_hcdr3_ids(
            self.infiller.tokenizer, record, span, proposed_length=1
        )
        fixed = len(tokens) - 1
        return max(0, self.infiller.max_length - fixed)

    def _renormalized_posterior(
        self, record: Any, span: HCDR3Span
    ) -> tuple[list[int], list[float]]:
        """
        Return ``(feasible_lengths, probs)`` from the model's length posterior,
        filtered to feasible lengths and renormalized to sum to 1.

        ``feasible_lengths`` is empty when no length fits ``max_length`` or the
        feasible mass is exactly zero; the caller then proposes nothing.
        """
        # Feasibility FIRST, via the module-level encoder (which has no
        # max_length check): when even the collapsed L=1 stream overflows, return
        # empty instead of running the query forward. The infiller-side encode
        # raises ValueError on overflow, and the CLI calls propose_lengths OUTSIDE
        # its per-length try/except -- so without this guard one overflowing
        # record would abort a whole generation run, contradicting this class's
        # documented "returns [] when nothing is feasible" contract.
        feasible_cap = min(self.length_head_max, self._feasible_max_length(record, span))
        if feasible_cap < 1:
            return [], []

        antibody_input_ids, antibody_attention_mask, _, _, _ = (
            self.infiller._encode_antibody_with_masked_hcdr3(record, span, proposed_length=1)
        )
        antigen_input_ids, antigen_attention_mask = self.infiller._encode_antigen(record)
        self.infiller.model.eval()
        with torch.no_grad():
            length_logits = self.infiller.model.forward_length_query(
                antibody_input_ids=antibody_input_ids,
                antibody_attention_mask=antibody_attention_mask,
                antigen_input_ids=antigen_input_ids,
                antigen_attention_mask=antigen_attention_mask,
            )
        probs = torch.softmax(length_logits[0], dim=-1)

        feasible_lengths: list[int] = []
        feasible_probs: list[float] = []
        # Class index i (0-based) corresponds to HCDR3 length i + 1.
        for length in range(1, feasible_cap + 1):
            feasible_lengths.append(length)
            feasible_probs.append(float(probs[length - 1].item()))

        mass = sum(feasible_probs)
        if not feasible_lengths or mass <= 0.0:
            return [], []
        return feasible_lengths, [p / mass for p in feasible_probs]

    def propose_lengths(
        self,
        record: Any,
        *,
        num_lengths: int,
        rng: random.Random,
    ) -> list[int]:
        """
        Sample (or top-k) ``num_lengths`` feasible lengths from the renormalized
        posterior.

        ``mode == "sample"`` (default) draws with replacement using the shared
        ``rng``, so end-to-end generation stays reproducible from the CLI seed --
        exactly like ``EmpiricalHCDR3LengthPrior``. ``mode == "top_k"`` returns the
        ``num_lengths`` highest-probability feasible lengths (deterministic,
        tie-broken by ascending length). Returns ``[]`` when no length is feasible
        for this record's scaffold, which the CLI's assert-generated-any guard
        then surfaces rather than letting it pass as a silent no-op.
        """
        if num_lengths <= 0:
            return []
        feasible_lengths, probs = self._renormalized_posterior(
            record, span=HCDR3Span.from_record(record)
        )
        if not feasible_lengths:
            return []
        if self.mode == "top_k":
            ranked = sorted(
                range(len(feasible_lengths)),
                key=lambda i: (-probs[i], feasible_lengths[i]),
            )
            return [feasible_lengths[i] for i in ranked[:num_lengths]]
        return rng.choices(feasible_lengths, weights=probs, k=num_lengths)


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
        antigen_encoder_type: str = "scratch",
        esm_model_name: str = "facebook/esm2_t6_8M_UR50D",
        antigen_max_length: int | None = None,
        guidance_model: torch.nn.Module | None = None,
        guidance_antigen_encoder_type: str = "scratch",
        guidance_esm_model_name: str = "facebook/esm2_t6_8M_UR50D",
        guidance_antigen_max_length: int | None = None,
        guidance_max_length: int | None = None,
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
        # Antigen tokenization must match training exactly (see antigen_tokenization).
        self.antigen_tokenizer = build_antigen_tokenizer(
            antigen_encoder_type=antigen_encoder_type,
            tokenizer=tokenizer,
            esm_model_name=esm_model_name,
        )
        self._antigen_encode_max_length = (
            max_length
            if antigen_encoder_type == "scratch"
            else (antigen_max_length if antigen_max_length is not None else max_length)
        )

        # External-guidance seam. When ``guidance_model`` is None the infiller
        # behaves exactly as before: the binder term is taken from
        # ``self.model``'s own compatibility head using the generation model's
        # antigen tensors. When a separately trained classifier is attached, ONLY
        # the binder term switches to it -- position selection and the reported
        # unguided marginals keep coming from ``self.model`` (see
        # ``guided_infill``).
        #
        # This is what lets steering be judged by a head that is not the same
        # head the generator carries, i.e. it separates the steerer from the
        # judge.
        self.guidance_model = guidance_model
        if guidance_model is not None:
            self.guidance_model.eval()
        # The guidance model must tokenize antigens with ITS OWN checkpoint's
        # tokenizer and effective max length, computed with the identical formula
        # the generation/scoring sites use -- otherwise the two streams disagree
        # about what an antigen looks like.
        self.guidance_antigen_tokenizer = build_antigen_tokenizer(
            antigen_encoder_type=guidance_antigen_encoder_type,
            tokenizer=tokenizer,
            esm_model_name=guidance_esm_model_name,
        )
        _guidance_base_max_length = (
            guidance_max_length if guidance_max_length is not None else max_length
        )
        self._guidance_antigen_encode_max_length = (
            _guidance_base_max_length
            if guidance_antigen_encoder_type == "scratch"
            else (
                guidance_antigen_max_length
                if guidance_antigen_max_length is not None
                else _guidance_base_max_length
            )
        )
        # Antibody-state length ceiling for the guidance model. Unlike the scorer
        # (which truncates antigens with only a warning), the guidance model must
        # fail LOUDLY if the working antibody state exceeds its trained
        # max_length, so its positional embeddings are never silently overrun.
        self.guidance_max_length = _guidance_base_max_length

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
        # Delegates to the module-level encoder so the collator's length-query
        # encoding and this one cannot drift apart; only the max_length check and
        # the tensor wrapping are owned here.
        tokens, mask_positions, prefix, suffix = encode_masked_hcdr3_ids(
            self.tokenizer,
            record,
            span,
            proposed_length=proposed_length,
        )

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
        ids = self.antigen_tokenizer.encode(antigen_sequence, self._antigen_encode_max_length)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=self.device)
        return input_ids, attention_mask

    def _encode_guidance_antigen(self, record: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode the antigen stream for the attached external guidance model.

        Identical in shape to ``_encode_antigen`` but tokenizes with the guidance
        checkpoint's own antigen tokenizer and effective max length. Only called
        when a ``guidance_model`` is attached; consumes no RNG and runs no model
        forward, so computing it never perturbs the ``guidance_strength == 0``
        decode path.
        """
        antigen_sequence = (getattr(record, "sequence_antigen", None) or "").strip()
        if not antigen_sequence:
            raise ValueError("record has no antigen sequence")
        ids = self.guidance_antigen_tokenizer.encode(
            antigen_sequence, self._guidance_antigen_encode_max_length
        )
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=self.device)
        return input_ids, attention_mask

    def _draw_canonical_index(
        self,
        sampling_logits: torch.Tensor,
        *,
        temperature: float,
        top_k: int | None,
    ) -> int:
        """
        Draw one canonical-residue index from already-canonical logits.

        This is the shared sampling-shaping step used by both plain
        (``_sample_token_id``) and guided (``guided_infill``) decoding. It owns
        only the temperature/top-k transformation and the multinomial draw; it
        does not know whether the logits are the model's raw MLM logits or the
        guidance-reweighted logits. Keeping it separate means guidance and the
        baseline share one, identically-behaving sampler.

        Args:
            sampling_logits:
                1-D tensor of logits over the canonical residues, in
                ``self.canonical_token_ids`` order. These are the logits that
                *shape the draw*; callers report log-probabilities from the
                unshaped, temperature-1 distribution separately.
            temperature:
                Softmax temperature (> 0). Higher is more uniform.
            top_k:
                Optional top-k filter over canonical residues. ``None`` or ``0``
                disables filtering; must be ``>= 0``.

        Returns:
            The index (into the canonical order) of the sampled residue.
        """
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if top_k is not None and top_k < 0:
            raise ValueError("top_k must be >= 0 (0 disables top-k filtering)")
        scaled_logits = sampling_logits / temperature
        if top_k is not None and 0 < top_k < scaled_logits.numel():
            top_values, top_indices = torch.topk(scaled_logits, k=top_k)
            probs = torch.softmax(top_values, dim=-1)
            sampled_rank = int(torch.multinomial(probs, num_samples=1).item())
            return int(top_indices[sampled_rank].item())
        probs = torch.softmax(scaled_logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

    def _sample_token_id(
        self,
        logits: torch.Tensor,
        *,
        temperature: float,
        top_k: int | None,
    ) -> tuple[int, float]:
        """
        Sample one canonical amino-acid token from full-vocabulary MLM logits.

        Returns the sampled token ID and its log probability under the model's
        true, unfiltered, temperature-1 distribution over canonical residues —
        not the temperature/top_k-shaped distribution used to draw the sample —
        so candidate scores stay comparable across sampling settings. This is
        the unguided path used by ``infill``; the shaping/draw is delegated to
        ``_draw_canonical_index``.
        """
        canonical_logits = logits[self.canonical_token_ids]
        # Reported log-prob from the true distribution (unfiltered, temperature 1);
        # the shaped distribution inside _draw_canonical_index is used only to
        # draw the sample.
        true_log_probs = torch.log_softmax(canonical_logits, dim=-1)
        canonical_idx = self._draw_canonical_index(
            canonical_logits, temperature=temperature, top_k=top_k
        )
        token_id = self.canonical_token_ids[canonical_idx]
        log_prob = float(true_log_probs[canonical_idx].item())
        return token_id, log_prob

    @torch.no_grad()
    def _binder_logprobs_by_candidate(
        self,
        antibody_input_ids: torch.Tensor,
        antibody_attention_mask: torch.Tensor,
        antigen_input_ids: torch.Tensor,
        antigen_attention_mask: torch.Tensor,
        position: int,
        *,
        guidance_target: int = 1,
        antigen_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Binder log-probability for each canonical residue placed at one position.

        This is the exact-enumeration workhorse behind guidance. It tentatively
        substitutes every canonical amino acid at ``position`` in the current
        working antibody stream, scores all ~20 resulting sequences against the
        antigen in a **single batched forward pass**, and returns the
        compatibility head's log-probability of class ``guidance_target`` for
        each. Enumeration (rather than a gradient approximation) is affordable
        because the amino-acid vocabulary is tiny; batching keeps it to one
        forward instead of ~20 sequential calls.

        Args:
            antibody_input_ids:
                ``[1, seq_len]`` antibody stream in its current working state.
            antibody_attention_mask:
                ``[1, seq_len]`` antibody attention mask.
            antigen_input_ids:
                ``[1, antigen_len]`` antigen stream (broadcast across candidates).
            antigen_attention_mask:
                ``[1, antigen_len]`` antigen attention mask.
            position:
                Token index at which to enumerate candidate residues.
            guidance_target:
                Compatibility-head class index whose log-probability is returned.
            antigen_state:
                Optional pre-fuse ``(antigen_hidden, antigen_attention_mask)``
                cache as returned by ``match_model.encode_antigen`` on the
                ENUMERATION antigen already repeated to the candidate batch dim
                (``guided_infill`` supplies it once per call). When given, the
                model skips re-encoding the antigen every step and the antigen id
                tensors below are unused by the model (they are still passed
                unchanged so the signature and callers stay stable, and NOT
                repeated so the ~20x antigen copy is avoided). When ``None``
                (default, e.g. the tested primitives) the antigen is repeated and
                encoded inline exactly as before -- byte-identical to the
                pre-cache code path.

        Returns:
            1-D tensor of length ``len(self.canonical_token_ids)`` giving
            ``log p(y = guidance_target | x[position := a])`` for each canonical
            residue ``a``, in ``self.canonical_token_ids`` order.

        The binder term is scored by ``self.guidance_model`` when an external
        classifier is attached, otherwise by ``self.model``. The antigen tensors
        passed in MUST already be encoded for whichever model scores here (the
        caller pairs the guidance antigen encoding with the guidance model).
        """
        # Default: self.model. With a seam: the attached guidance classifier.
        # This is the single site that computes the binder term, so every caller
        # (guided_infill loop, _guided_scores_from_logits) routes consistently.
        match_model = self.guidance_model if self.guidance_model is not None else self.model
        # Fail LOUDLY rather than inherit the scorer's silent-truncation
        # asymmetry: if the working antibody state is longer than the guidance
        # checkpoint's trained max_length, its positional embeddings would be
        # overrun. (Only meaningful when a guidance model is attached; guidance is
        # consulted only for gamma > 0, so gamma == 0 never reaches this check.)
        if (
            self.guidance_model is not None
            and antibody_input_ids.size(1) > self.guidance_max_length
        ):
            raise ValueError(
                "working antibody state length "
                f"{antibody_input_ids.size(1)} exceeds the guidance checkpoint's "
                f"max_length {self.guidance_max_length}; guidance would silently "
                "truncate the state. Use a guidance checkpoint trained with a "
                "max_length >= the generation checkpoint's."
            )
        num_canonical = len(self.canonical_token_ids)
        canonical_id_tensor = torch.tensor(
            self.canonical_token_ids, dtype=torch.long, device=self.device
        )
        candidate_input_ids = antibody_input_ids.repeat(num_canonical, 1)
        candidate_input_ids[:, position] = canonical_id_tensor
        # With a supplied antigen_state cache the model ignores the antigen id
        # tensors, so skip the ~20x antigen repeat; without it, repeat and encode
        # inline so the None path is byte-identical to the pre-cache code.
        if antigen_state is None:
            antigen_input_ids_arg = antigen_input_ids.repeat(num_canonical, 1)
            antigen_attention_mask_arg = antigen_attention_mask.repeat(num_canonical, 1)
        else:
            antigen_input_ids_arg = antigen_input_ids
            antigen_attention_mask_arg = antigen_attention_mask
        _, compatibility_logits = match_model(
            antibody_input_ids=candidate_input_ids,
            antibody_attention_mask=antibody_attention_mask.repeat(num_canonical, 1),
            antigen_input_ids=antigen_input_ids_arg,
            antigen_attention_mask=antigen_attention_mask_arg,
            antigen_state=antigen_state,
        )
        return torch.log_softmax(compatibility_logits, dim=-1)[:, guidance_target]

    @torch.no_grad()
    def _guided_position_scores(
        self,
        antibody_input_ids: torch.Tensor,
        antibody_attention_mask: torch.Tensor,
        antigen_input_ids: torch.Tensor,
        antigen_attention_mask: torch.Tensor,
        position: int,
        *,
        guidance_strength: float,
        guidance_target: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute guided per-residue scores for one masked position.

        This is the order-independent core of ProteinGuide-style guided
        infilling. It answers a single, schedule-agnostic question: given the
        current partially-filled antibody/antigen state and one masked token
        ``position``, how should the model's own residue distribution at that
        position be reweighted so that residues which raise the antigen-binder
        probability become more likely? The caller (the unmasking loop in
        ``guided_infill``) owns the decision of *which* state and position to
        pass; this method owns only the reweighting math for that one position.

        The reweighting follows classifier guidance applied at the level of the
        MLM's categorical distribution over canonical amino acids. For each
        canonical residue ``a`` at ``position``::

            guided_logit[a] = log p_MLM(a | x)
                              + guidance_strength * log p(y = target | x[pos := a])

        where ``p_MLM(. | x)`` is the model's masked-token marginal at
        ``position`` (with ``position`` still masked) and ``p(y | x[pos := a])``
        is the compatibility head's probability for class ``guidance_target``
        after tentatively placing residue ``a``. The per-candidate binder term
        is evaluated by **exact enumeration** over the ~20 canonical residues in
        a single batched forward pass, which is tractable here precisely because
        the amino-acid vocabulary is tiny (unlike large-vocabulary language
        models, where a gradient approximation would be needed).

        Args:
            antibody_input_ids:
                ``[1, seq_len]`` antibody token stream in its current working
                state. ``position`` must currently hold ``[MASK]``; other HCDR3
                positions may be masked or already filled depending on the
                schedule.
            antibody_attention_mask:
                ``[1, seq_len]`` attention mask for the antibody stream.
            antigen_input_ids:
                ``[1, antigen_len]`` antigen token stream (constant across the
                whole generation).
            antigen_attention_mask:
                ``[1, antigen_len]`` attention mask for the antigen stream.
            position:
                Token index into the antibody stream to score.
            guidance_strength:
                The guidance factor ``gamma``. ``0.0`` disables guidance and the
                classifier is not consulted at all (the returned guided logits
                equal the unguided marginal). Larger positive values steer more
                strongly toward ``guidance_target``.
            guidance_target:
                Compatibility-head class index to steer toward (``1`` = the
                binder / compatible class).

        Returns:
            A ``(guided_logits, unguided_logprobs)`` pair of 1-D tensors, each
            indexed in the same order as ``self.canonical_token_ids``:

            - ``guided_logits``: the reweighted base logits to be
              temperature-scaled and sampled by the caller. At
              ``guidance_strength == 0`` these equal ``unguided_logprobs``.
            - ``unguided_logprobs``: the model's true, unfiltered, temperature-1
              log-probabilities over canonical residues. These are what the
              caller accumulates for candidate ``log_probability`` reporting, so
              guidance changes *sampling* without corrupting the reported
              likelihood (mirroring the reporting convention in
              ``_sample_token_id``).
        """
        # Run the model on the current working state (with `position` still
        # masked) to get its residue marginal, then defer the reweighting math to
        # the shared combination helper. `guided_infill` calls that same helper
        # with the forward it already runs for position selection, so the guided
        # scoring in this tested primitive and in the production loop can never
        # drift apart.
        mlm_logits, _ = self.model(
            antibody_input_ids=antibody_input_ids,
            antibody_attention_mask=antibody_attention_mask,
            antigen_input_ids=antigen_input_ids,
            antigen_attention_mask=antigen_attention_mask,
        )
        return self._guided_scores_from_logits(
            mlm_logits,
            antibody_input_ids,
            antibody_attention_mask,
            antigen_input_ids,
            antigen_attention_mask,
            position,
            guidance_strength=guidance_strength,
            guidance_target=guidance_target,
        )

    @torch.no_grad()
    def _guided_scores_from_logits(
        self,
        mlm_logits: torch.Tensor,
        antibody_input_ids: torch.Tensor,
        antibody_attention_mask: torch.Tensor,
        antigen_input_ids: torch.Tensor,
        antigen_attention_mask: torch.Tensor,
        position: int,
        *,
        guidance_strength: float,
        guidance_target: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Reweight one position's residue distribution given a precomputed forward.

        This is the pure post-forward core of ``_guided_position_scores``, split
        out so callers that have *already* run the model on the working state
        (the ``guided_infill`` unmasking loop, which runs one forward per step for
        position selection) can share the exact same guidance math without paying
        for a second forward. ``_guided_position_scores`` runs the forward and
        delegates here; the unmasking loop passes the ``mlm_logits`` it already
        holds. Both routes therefore produce identical guided distributions.

        Args:
            mlm_logits:
                ``[1, seq_len, vocab]`` MLM logits from a forward on the current
                working state, with ``position`` still masked.
            antibody_input_ids:
                ``[1, seq_len]`` antibody stream in its current working state.
            antibody_attention_mask:
                ``[1, seq_len]`` antibody attention mask.
            antigen_input_ids:
                ``[1, antigen_len]`` antigen stream (constant across generation).
            antigen_attention_mask:
                ``[1, antigen_len]`` antigen attention mask.
            position:
                Token index into the antibody stream to score.
            guidance_strength:
                The guidance factor ``gamma``. ``0.0`` skips the enumeration
                forward entirely so the classifier has provably zero effect.
            guidance_target:
                Compatibility-head class index to steer toward (``1`` = binder).

        Returns:
            The ``(guided_logits, unguided_logprobs)`` pair described in
            ``_guided_position_scores``, both in ``self.canonical_token_ids``
            order.
        """
        canonical_logits = mlm_logits[0, position][self.canonical_token_ids]
        unguided_logprobs = torch.log_softmax(canonical_logits, dim=-1)

        # When guidance is off, skip the enumeration forward entirely so the
        # classifier has provably zero effect and the call stays cheap.
        if guidance_strength == 0.0:
            return unguided_logprobs.clone(), unguided_logprobs

        # Exact enumeration of the binder term over the ~20 canonical residues
        # (batched into one forward), then combine. The helper returns values in
        # `self.canonical_token_ids` order, matching `unguided_logprobs`.
        binder_logprobs = self._binder_logprobs_by_candidate(
            antibody_input_ids,
            antibody_attention_mask,
            antigen_input_ids,
            antigen_attention_mask,
            position,
            guidance_target=guidance_target,
        )
        guided_logits = unguided_logprobs + guidance_strength * binder_logprobs
        return guided_logits, unguided_logprobs

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

        # The masked antibody input is fully determined by the record, span, and
        # proposed length, so it is identical across samples. Build it and run the
        # MLM forward once, then draw `num_samples` independent residue sets from
        # the shared per-position logits. Dropout is off under eval(), so the
        # single forward consumes no RNG: the draws match the previous
        # per-sample-forward behavior while doing one transformer pass instead of
        # `num_samples` of them.
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
        position_logits = [mlm_logits[0, pos, :] for pos in mask_positions]

        candidates: list[HCDR3InfillCandidate] = []
        for _ in range(num_samples):
            sampled_token_ids: list[int] = []
            log_probability = 0.0
            for logits in position_logits:
                token_id, log_prob = self._sample_token_id(
                    logits,
                    temperature=temperature,
                    top_k=top_k,
                )
                sampled_token_ids.append(token_id)
                log_probability += log_prob

            generated_hcdr3 = "".join(self.tokenizer.id_to_token[token_id] for token_id in sampled_token_ids)
            heavy_sequence = prefix + generated_hcdr3 + suffix
            mean_log_probability = log_probability / proposed_length if proposed_length > 0 else 0.0
            compatibility_score = scorer.score(record, heavy_sequence=heavy_sequence) if scorer is not None else None
            candidates.append(
                HCDR3InfillCandidate(
                    generated_hcdr3=generated_hcdr3,
                    heavy_sequence=heavy_sequence,
                    log_probability=log_probability,
                    mean_log_probability=mean_log_probability,
                    length=proposed_length,
                    compatibility_score=compatibility_score,
                )
            )
        return candidates

    def _select_next_position(
        self,
        mlm_logits: torch.Tensor,
        remaining_positions: list[int],
        *,
        order: str,
        rng: random.Random,
    ) -> int:
        """
        Choose the next masked position to fill under the given schedule.

        The unmasking *order* matters for iterative masked decoding: it sets the
        integration path and therefore how errors accumulate. This method owns
        that choice so ``guided_infill`` stays readable, and so new schedules can
        be added in one place.

        Args:
            mlm_logits:
                ``[1, seq_len, vocab_size]`` logits from a forward pass on the
                current working state. Only consulted by ``"confidence"``.
            remaining_positions:
                Still-masked token indices, in ascending (N->C) order.
            order:
                Unmasking schedule:

                - ``"confidence"`` (easy-first): fill the position whose residue
                  marginal has the lowest entropy (the model is most certain
                  about) first. MaskGIT-style default; generally strongest for
                  masked/diffusion decoding.
                - ``"random"``: fill remaining positions in a random order drawn
                  from ``rng``; mirrors the model's random-span MLM training.
                - ``"left_to_right"``: fill N->C terminus in sequence order.
            rng:
                Random source used only by ``order == "random"``.

        Returns:
            The chosen token index (an element of ``remaining_positions``).
        """
        if order == "left_to_right":
            return remaining_positions[0]
        if order == "random":
            return remaining_positions[rng.randrange(len(remaining_positions))]
        if order == "confidence":
            marginals = torch.stack(
                [mlm_logits[0, pos][self.canonical_token_ids] for pos in remaining_positions]
            )
            logprobs = torch.log_softmax(marginals, dim=-1)
            entropy = -(logprobs.exp() * logprobs).sum(dim=-1)
            return remaining_positions[int(torch.argmin(entropy).item())]
        raise ValueError(
            f"unknown order {order!r}; expected 'confidence', 'random', or 'left_to_right'"
        )

    @torch.no_grad()
    def guided_infill(
        self,
        record: Any,
        *,
        length: int | None = None,
        num_samples: int = 1,
        temperature: float = 1.0,
        top_k: int | None = None,
        guidance_strength: float = 0.0,
        guidance_target: int = 1,
        order: str = "confidence",
        scorer: "AntigenCompatibilityScorer | None" = None,
        rng: random.Random | None = None,
    ) -> list[HCDR3InfillCandidate]:
        """
        Generate HCDR3 candidates with ProteinGuide-style binder guidance.

        This is the guided counterpart to ``infill``. Where ``infill`` runs one
        forward pass and samples every masked position independently from its
        marginal, ``guided_infill`` unmasks **iteratively, one position per
        step**, so each residue is drawn conditioned on the residues already
        placed, and (when ``guidance_strength > 0``) reweighted toward antigen
        binding. The three pieces fit together as:

        1. ``_select_next_position`` picks which masked position to fill next
           under ``order`` (default easy-first / lowest-entropy).
        2. ``_guided_position_scores`` / ``_binder_logprobs_by_candidate``
           produce the guided distribution for that position by exact
           enumeration over the canonical residues.
        3. ``_draw_canonical_index`` performs the temperature/top-k draw — the
           same shaping used by the unguided sampler.

        The method is intentionally additive: ``infill`` and its tests are
        untouched, and ``guidance_strength == 0`` gives plain iterative MLM
        decoding (the compatibility head is never consulted).

        Args:
            record:
                Dataset record with heavy sequence, antigen sequence, and known
                HCDR3 coordinates (see ``HCDR3Span.from_record``).
            length:
                Number of HCDR3 residues to generate. Defaults to the record's
                known span length (fixed-length infilling). Unknown-length
                design proposes a length first (see ``LengthProposalStrategy``),
                then calls this per proposed length.
            num_samples:
                Number of independent candidates to generate. Unlike ``infill``,
                each sample runs its own unmasking loop (the chains diverge as
                residues are committed), so cost scales with ``num_samples``.
            temperature:
                Softmax temperature for the per-position residue draw.
            top_k:
                Optional top-k filter over canonical residues at each step.
            guidance_strength:
                The guidance factor ``gamma``. ``0.0`` disables guidance;
                larger positive values steer more strongly toward
                ``guidance_target``. See ``_guided_position_scores`` for the
                exact reweighting.
            guidance_target:
                Compatibility-head class index to steer toward (``1`` = binder).
            order:
                Unmasking schedule: ``"confidence"`` (default), ``"random"``, or
                ``"left_to_right"``. See ``_select_next_position``.
            scorer:
                Optional compatibility scorer used to attach a final binder
                probability to each candidate (scored on the completed sequence,
                exactly as in ``infill``).
            rng:
                Random source for ``order == "random"``. A fresh
                ``random.Random()`` is created when omitted. Residue sampling
                itself uses the global torch RNG (seed with
                ``torch.manual_seed``), matching ``infill``.

        Returns:
            List of generated candidates. ``log_probability`` /
            ``mean_log_probability`` are accumulated from the model's *unguided*
            marginals — guidance changes which residues are drawn, not how their
            likelihood is scored — so guided candidates stay comparable to each
            other and across lengths. Note they are **not** the same quantity as
            ``infill``'s score: ``infill`` sums independent per-position marginals
            from a single fully-masked forward, whereas ``guided_infill`` sums the
            unguided conditionals along the iterative unmasking path (each term
            conditioned on the residues already committed). The two summations
            differ even at ``guidance_strength == 0``, so do not pool guided and
            single-pass ``infill`` scores into one ranking.
        """
        if num_samples <= 0:
            return []
        if order not in ("confidence", "random", "left_to_right"):
            raise ValueError(
                f"unknown order {order!r}; expected 'confidence', 'random', or 'left_to_right'"
            )
        if rng is None:
            rng = random.Random()

        span = HCDR3Span.from_record(record)
        proposed_length = span.length if length is None else int(length)
        antigen_input_ids, antigen_attention_mask = self._encode_antigen(record)
        # The binder term is scored by the attached guidance model when present,
        # and that model owns its own antigen tokenization. With no seam attached
        # these are the same tensors, so the default path is unchanged.
        if self.guidance_model is not None:
            guidance_antigen_input_ids, guidance_antigen_attention_mask = (
                self._encode_guidance_antigen(record)
            )
        else:
            guidance_antigen_input_ids = antigen_input_ids
            guidance_antigen_attention_mask = antigen_attention_mask
        base_input_ids, antibody_attention_mask, mask_positions, prefix, suffix = (
            self._encode_antibody_with_masked_hcdr3(
                record,
                span,
                proposed_length=proposed_length,
            )
        )
        self.model.eval()

        # --- amortized antigen encoding (pre-fuse antigen_hidden cache) ---
        # The antigen stream is CONSTANT across every unmasking step and every
        # sample, and encode_antigen is the antigen-only prefix that ends exactly
        # where fuse (the first cross-stream contact) begins. In deterministic
        # eval mode the encoder consumes NO RNG and is a pure function of its
        # inputs, so encoding once here and reusing the pre-fuse
        # (antigen_hidden, antigen_attention_mask) pair yields hidden states
        # BYTE-IDENTICAL to what each step would recompute -- the same
        # determinism the single-forward `infill` already relies on. Two shapes
        # are needed and cached separately:
        #   * batch-1 for the per-step ordering forward (self.model, generation antigen),
        #   * batch-V for the ~20-way enumeration (match_model, guidance antigen).
        # match_model mirrors _binder_logprobs_by_candidate's resolution so the
        # seam routes identically. The batch-V state is built ONLY when guidance
        # is active, so the gamma == 0 path does NO extra encoder work.
        match_model = self.guidance_model if self.guidance_model is not None else self.model
        ordering_antigen_state = self.model.encode_antigen(
            antigen_input_ids, antigen_attention_mask
        )
        if guidance_strength != 0.0:
            num_candidates = len(self.canonical_token_ids)
            enum_antigen_state = match_model.encode_antigen(
                guidance_antigen_input_ids.repeat(num_candidates, 1),
                guidance_antigen_attention_mask.repeat(num_candidates, 1),
            )
        else:
            enum_antigen_state = None

        candidates: list[HCDR3InfillCandidate] = []
        for _ in range(num_samples):
            # Each candidate starts from the same fully-masked HCDR3 and fills it
            # in independently; committed residues change later positions' context.
            working_input_ids = base_input_ids.clone()
            remaining_positions = list(mask_positions)
            filled: dict[int, int] = {}
            log_probability = 0.0

            while remaining_positions:
                # Fresh forward on the current partial state: gives the ordering
                # signal and the chosen position's unguided marginal.
                mlm_logits, _ = self.model(
                    antibody_input_ids=working_input_ids,
                    antibody_attention_mask=antibody_attention_mask,
                    antigen_input_ids=antigen_input_ids,
                    antigen_attention_mask=antigen_attention_mask,
                    antigen_state=ordering_antigen_state,
                )
                position = self._select_next_position(
                    mlm_logits, remaining_positions, order=order, rng=rng
                )
                canonical_logits = mlm_logits[0, position][self.canonical_token_ids]
                unguided_logprobs = torch.log_softmax(canonical_logits, dim=-1)

                if guidance_strength == 0.0:
                    guided_logits = unguided_logprobs
                else:
                    binder_logprobs = self._binder_logprobs_by_candidate(
                        working_input_ids,
                        antibody_attention_mask,
                        guidance_antigen_input_ids,
                        guidance_antigen_attention_mask,
                        position,
                        guidance_target=guidance_target,
                        antigen_state=enum_antigen_state,
                    )
                    guided_logits = unguided_logprobs + guidance_strength * binder_logprobs

                canonical_idx = self._draw_canonical_index(
                    guided_logits, temperature=temperature, top_k=top_k
                )
                token_id = self.canonical_token_ids[canonical_idx]
                working_input_ids[0, position] = token_id
                filled[position] = token_id
                # Accumulate the UNGUIDED marginal log-prob for the chosen
                # residue so reported scores are not inflated by guidance.
                log_probability += float(unguided_logprobs[canonical_idx].item())
                remaining_positions.remove(position)

            generated_hcdr3 = "".join(
                self.tokenizer.id_to_token[filled[pos]] for pos in mask_positions
            )
            heavy_sequence = prefix + generated_hcdr3 + suffix
            mean_log_probability = (
                log_probability / proposed_length if proposed_length > 0 else 0.0
            )
            compatibility_score = (
                scorer.score(record, heavy_sequence=heavy_sequence)
                if scorer is not None
                else None
            )
            candidates.append(
                HCDR3InfillCandidate(
                    generated_hcdr3=generated_hcdr3,
                    heavy_sequence=heavy_sequence,
                    log_probability=log_probability,
                    mean_log_probability=mean_log_probability,
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
        antigen_encoder_type: str = "scratch",
        esm_model_name: str = "facebook/esm2_t6_8M_UR50D",
        antigen_max_length: int | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = torch.device(device)
        # Antigen tokenization must match training exactly (see antigen_tokenization).
        self.antigen_tokenizer = build_antigen_tokenizer(
            antigen_encoder_type=antigen_encoder_type,
            tokenizer=tokenizer,
            esm_model_name=esm_model_name,
        )
        self._antigen_encode_max_length = (
            max_length
            if antigen_encoder_type == "scratch"
            else (antigen_max_length if antigen_max_length is not None else max_length)
        )

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
            # Fall back through the record's own `locus` before "IGH", following
            # the training collator's single-chain branch (`heavy_locus or locus`,
            # MLMCollator._encode_record). Without the `locus` step a record
            # carrying its chain identity only in `locus` is scored as [IGH] while
            # training encoded its real chain token -- a silent first-position
            # train/inference mismatch. See _encode_antibody_with_masked_hcdr3 for
            # the one residual case (neither field set) where the trailing "IGH"
            # still diverges from training.
            ids = self.tokenizer.encode_sequence(
                heavy_sequence,
                locus=(
                    getattr(record, "heavy_locus", None)
                    or getattr(record, "locus", None)
                    or "IGH"
                ),
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
        ids = self.antigen_tokenizer.encode(antigen_sequence, self._antigen_encode_max_length)
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
