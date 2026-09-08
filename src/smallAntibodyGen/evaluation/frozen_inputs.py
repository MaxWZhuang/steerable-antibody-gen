"""The frozen benchmark artifact: build it once, score it forever.

WHY FROZEN INPUTS RATHER THAN A FROZEN SEED
-------------------------------------------
`MLMCollator` draws its corruption from an RNG. Reproducing a comparison by
re-seeding survives exactly as long as nobody touches the collator: change the
order of a draw, add a knob, reorder a loop, and the "identical masks" two
checkpoints were compared under silently stop being identical. The artifact
built here stores the corrupted token ids, the labels, the attention masks, the
region spans, the target positions and the donor assignments. Scoring loads
them. There is no second draw, so there is nothing to drift.

THE POSITIONAL TRAP THIS MODULE EXISTS TO AVOID
-----------------------------------------------
A paired record encodes as::

    [CLS] [IGH] H H H ... [SEP] [IGK] L L L ... [EOS]

The heavy chain therefore occupies positions 2 .. 2+len(H)-1 and the light chain
starts at 2+len(H)+2. That asymmetry is the whole difficulty of a two-directional
partner probe:

- **Predicting heavy while varying light** is positionally safe: the light chain
  is downstream, so swapping or deleting it moves no heavy position.
- **Predicting light while varying heavy** is not. A donor heavy of a different
  length SHIFTS the entire light chain. Under RoPE those residues then rotate at
  different positions, and a measured "partner effect" would be partly a
  positional effect. An EXACT token-length match is required for a partner that
  precedes the predicted chain, and an example with no such donor is dropped
  rather than quietly accepted.

The same asymmetry decides what "absent" can mean. Removing the light chain
preserves every heavy position (`ABSENT_CHAIN_REMOVED`). Removing the heavy chain
would shift the light chain, so the light direction instead substitutes a
same-length synthetic filler (`ABSENT_SYNTHETIC_FILLER`).

**The synthetic filler is a replacement control, not a neutral context.** A
sequence of one repeated residue is not in distribution merely because that
residue is in the vocabulary -- no real chain looks like that, and a model may
respond to its strangeness rather than to the absence of information. It is
named separately, recorded per case, never pooled with a true removal, and it is
SECONDARY: the primary comparison is native versus matched-alternative, both of
which are real chains.

THE INVARIANT
-------------
Within one case, every condition must present the predicted chain to the model
identically: the whole predicted span, its attention mask, and the labels -- not
merely the selected target positions. Checking only the targets would accept a
changed residue elsewhere in the same chain, which the model reads as context.
`verify_frozen` enforces the full span and `build_frozen_benchmark` runs it
before returning, so a violating artifact cannot be built.

Two digests are recorded. The TENSOR digest covers what the model saw. The
SEMANTIC digest covers what the analysis will condition on -- direction, donor
identity, grouping, absent mechanism, region spans -- because those decide what
a number MEANS even when no tensor changed.

Production path: encoding goes through `MLMCollator._encode_record`, corruption
through `_mask_tokens`, and heavy HCDR3 coordinates through
`_heavy_hcdr3_token_span`. These are private helpers, and calling them is
deliberate: a reimplementation here would be a second encoding call site, which
is this repository's most expensive and least visible bug class.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from smallAntibodyGen.data.MLMCollator import MLMCollator, OASRecord

FROZEN_SCHEMA = "frozen-eval-inputs/2"

PROBE_STAGE1_RETENTION = "stage1_retention"
PROBE_PARTNER = "partner_dependence"
PROBE_HCDR3_FULL_SPAN = "hcdr3_full_span"
PROBES = (PROBE_STAGE1_RETENTION, PROBE_PARTNER, PROBE_HCDR3_FULL_SPAN)

HEAVY_GIVEN_LIGHT = "heavy_given_light"
LIGHT_GIVEN_HEAVY = "light_given_heavy"
DIRECTIONS = (HEAVY_GIVEN_LIGHT, LIGHT_GIVEN_HEAVY)

NATIVE = "native"
MATCHED_ALTERNATIVE = "matched_alternative"
ABSENT = "absent"
CONDITIONS = (NATIVE, MATCHED_ALTERNATIVE, ABSENT)

ABSENT_CHAIN_REMOVED = "chain_removed"
ABSENT_SYNTHETIC_FILLER = "synthetic_filler"

#: Filler residue for `ABSENT_SYNTHETIC_FILLER`. "X" is the unknown-residue code.
#: It is a SYNTHETIC control: a whole chain of it is not in distribution.
#:
#: NOT usable: the text of `tokenizer.mask_token`. It is "[MASK]", the encoder
#: maps sequences character by character, so it expands to seven tokens per
#: residue -- four of which (M, A, S, K) are real amino acids.
DEFAULT_FILLER_RESIDUE = "X"

MLM_IGNORE_INDEX = -100


class FrozenInputsError(ValueError):
    """Raised when a frozen artifact violates its own contract."""


class PositionDriftError(FrozenInputsError):
    """The predicted chain differs between two conditions of one case."""


class EmptyProbeError(FrozenInputsError):
    """A required probe produced no cases.

    Its own reason for existing: a donor-selection bug once emptied the partner
    probe completely and verification passed, because every check it ran was
    vacuously true over zero cases.
    """


# --------------------------------------------------------------------------- #
# Stable identity, without touching the corpus producer
# --------------------------------------------------------------------------- #
ID_FROM_RECORD_ID = "record_id"
ID_FROM_PAIR_ID = "pair_id"
ID_FROM_CONTENT = "content_sha256"


def derive_artifact_id(record: OASRecord) -> tuple[str, str]:
    """A stable id for one record, and the source it came from.

    Processed OAS paired rows carry `pair_id` but NOT `record_id`. Comparing
    `record_id` between two such records compares `None` with `None`, which is
    True -- so a self-exclusion test written that way rejects every candidate and
    silently empties the probe. That is not hypothetical; it happened. Identity
    here is therefore always a real string, falling back to a content hash so it
    exists for every record and is stable across runs and machines.
    """
    if record.record_id:
        return str(record.record_id), ID_FROM_RECORD_ID
    if record.pair_id:
        return str(record.pair_id), ID_FROM_PAIR_ID
    payload = "|".join(
        [
            (record.sequence_heavy or "").strip(),
            (record.sequence_light or "").strip(),
            (record.sequence or "").strip(),
            str(record.heavy_locus or ""),
            str(record.light_locus or ""),
            str(record.locus or ""),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24], ID_FROM_CONTENT


# --------------------------------------------------------------------------- #
# Donor policy -- declared, seeded, and reported
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DonorPolicy:
    """How an alternative partner is chosen. Declared, never implicit.

    "First eligible" is not a policy: it makes the comparison depend on corpus
    order, and on real data it collapsed 126 of 128 heavy-direction groups onto
    just two donors. Selection is a seeded draw from the full eligible set.

    `max_length_ratio_delta` bounds how different the donor partner may be in
    length. Without it a three-residue light chain was accepted against a
    49-residue native one, which is a length ablation wearing a partner-identity
    label. It is ignored when exact matching applies, which is mandatory
    whenever the partner precedes the predicted chain.
    """

    name: str = "seeded_length_matched"
    max_length_ratio_delta: float = 0.2
    seed: int = 20260907

    def as_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "max_length_ratio_delta": self.max_length_ratio_delta,
            "seed": self.seed,
        }


@dataclass
class ExclusionLedger:
    """Why cases were dropped. Saved, because a silent drop is a silent bias."""

    counts: Counter = field(default_factory=Counter)

    def drop(self, reason: str) -> None:
        self.counts[reason] += 1

    def as_payload(self) -> dict[str, int]:
        return dict(sorted(self.counts.items()))


@dataclass
class FrozenCase:
    """One scorable unit: fixed inputs, fixed targets, fixed provenance."""

    case_id: str
    probe: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    target_positions: tuple[int, ...]
    artifact_id: str
    artifact_id_source: str
    direction: str | None = None
    condition: str | None = None
    predicted_span: tuple[int, int] | None = None
    donor_artifact_id: str | None = None
    absent_mechanism: str | None = None
    regions: dict[str, Any] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)

    def as_payload(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "probe": self.probe,
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "labels": self.labels,
            "target_positions": list(self.target_positions),
            "artifact_id": self.artifact_id,
            "artifact_id_source": self.artifact_id_source,
            "direction": self.direction,
            "condition": self.condition,
            "predicted_span": list(self.predicted_span) if self.predicted_span else None,
            "donor_artifact_id": self.donor_artifact_id,
            "absent_mechanism": self.absent_mechanism,
            "regions": dict(self.regions),
            "notes": dict(self.notes),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "FrozenCase":
        span = payload.get("predicted_span")
        return cls(
            case_id=payload["case_id"],
            probe=payload["probe"],
            input_ids=payload["input_ids"],
            attention_mask=payload["attention_mask"],
            labels=payload["labels"],
            target_positions=tuple(payload["target_positions"]),
            artifact_id=payload["artifact_id"],
            artifact_id_source=payload["artifact_id_source"],
            direction=payload.get("direction"),
            condition=payload.get("condition"),
            predicted_span=tuple(span) if span else None,
            donor_artifact_id=payload.get("donor_artifact_id"),
            absent_mechanism=payload.get("absent_mechanism"),
            regions=dict(payload.get("regions") or {}),
            notes=dict(payload.get("notes") or {}),
        )


# --------------------------------------------------------------------------- #
# Encoding + region helpers, all routed through the production collator
# --------------------------------------------------------------------------- #
def _encode(collator: MLMCollator, record: OASRecord) -> torch.Tensor:
    return torch.tensor(collator._encode_record(record), dtype=torch.long)


def _corrupt(
    collator: MLMCollator, record: OASRecord, input_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    masked, labels, target_mask = collator._mask_tokens(input_ids.unsqueeze(0), [record])
    return masked[0], labels[0], target_mask[0]


def heavy_span(record: OASRecord) -> tuple[int, int]:
    """[start, end) of heavy residues. Always begins at 2, partner-independent."""
    return (2, 2 + len((record.sequence_heavy or "").strip()))


def light_span(record: OASRecord) -> tuple[int, int]:
    """[start, end) of light residues: after [SEP] and the light chain token."""
    heavy = len((record.sequence_heavy or "").strip())
    light = len((record.sequence_light or "").strip())
    start = 2 + heavy + 2
    return (start, start + light)


def _predicted_span(record: OASRecord, direction: str) -> tuple[int, int]:
    return heavy_span(record) if direction == HEAVY_GIVEN_LIGHT else light_span(record)


def _light_hcdr3_token_span(
    collator: MLMCollator, input_ids_row: torch.Tensor, record: OASRecord
) -> tuple[int, int, bool]:
    """Light CDR3 in token coordinates.

    MIRRORS the production heavy rule (`_heavy_hcdr3_token_span`) with the light
    base offset, because the collator has no light-chain equivalent -- HCDR3
    masking is heavy-only by design. Validity is checked the same way: fully
    inside the row, and every position a residue rather than a special token.
    Flagged as `mirrored` in the artifact so a reader knows it is not the
    production helper.
    """
    start, end = record.cdr3_start_aa_light, record.cdr3_end_aa_light
    if start is None or end is None or end <= start:
        return -1, -1, False
    base = light_span(record)[0]
    token_start, token_end = base + int(start), base + int(end)
    if token_start < 0 or token_end > input_ids_row.size(0):
        return token_start, token_end, False
    span = input_ids_row[token_start:token_end].tolist()
    valid = bool(span) and all(int(t) not in collator.tokenizer.special_ids for t in span)
    return token_start, token_end, valid


def _build_regions(
    collator: MLMCollator,
    record: OASRecord,
    input_ids: torch.Tensor,
    target_positions: Sequence[int],
) -> dict[str, Any]:
    """Freeze the region spans scoring will need, plus their target overlap.

    Without this the artifact cannot answer the HCDR3-specific questions -- the
    forgetting measurement was HCDR3 token recovery, not overall MLM accuracy --
    and the scorer would need a second metadata lookup that could disagree with
    what was actually frozen.
    """
    targets = {int(p) for p in target_positions}
    regions: dict[str, Any] = {}

    paired = bool(
        (record.sequence_heavy or "").strip() and (record.sequence_light or "").strip()
    )
    if paired:
        regions["heavy"] = {"span": list(heavy_span(record)), "valid": True}
        regions["light"] = {"span": list(light_span(record)), "valid": True}
    else:
        regions["chain"] = {
            "span": [2, max(2, int(input_ids.numel()) - 1)],
            "valid": True,
        }

    h_start, h_end, h_valid = collator._heavy_hcdr3_token_span(input_ids, record)
    regions["hcdr3_heavy"] = {
        "span": [int(h_start), int(h_end)],
        "valid": bool(h_valid),
        "derivation": "production:_heavy_hcdr3_token_span",
    }
    if paired:
        l_start, l_end, l_valid = _light_hcdr3_token_span(collator, input_ids, record)
        regions["hcdr3_light"] = {
            "span": [int(l_start), int(l_end)],
            "valid": bool(l_valid),
            "derivation": "mirrored:light_base_offset",
        }

    for info in regions.values():
        span = info["span"]
        info["targets_in_region"] = (
            sum(1 for p in targets if span[0] <= p < span[1])
            if info["valid"] and span[1] > span[0]
            else 0
        )
    return regions


# --------------------------------------------------------------------------- #
# Donor selection
# --------------------------------------------------------------------------- #
def _partner_sequence(record: OASRecord, direction: str) -> str:
    seq = record.sequence_light if direction == HEAVY_GIVEN_LIGHT else record.sequence_heavy
    return (seq or "").strip()


def _partner_locus(record: OASRecord, direction: str) -> str:
    return (
        (record.light_locus or "IGK")
        if direction == HEAVY_GIVEN_LIGHT
        else (record.heavy_locus or "IGH")
    )


def eligible_donors(
    record: OASRecord,
    pool: Sequence[tuple[str, OASRecord]],
    direction: str,
    *,
    require_exact_length: bool,
    policy: DonorPolicy,
) -> list[tuple[str, OASRecord]]:
    """Every admissible alternative partner, not just the first one."""
    own_id, _ = derive_artifact_id(record)
    own_seq = _partner_sequence(record, direction)
    own_len = len(own_seq)
    own_locus = _partner_locus(record, direction)
    out: list[tuple[str, OASRecord]] = []
    for cand_id, candidate in pool:
        if cand_id == own_id:
            continue
        if _partner_locus(candidate, direction) != own_locus:
            continue
        cand_seq = _partner_sequence(candidate, direction)
        if not cand_seq or cand_seq == own_seq:
            # An identical partner makes the condition a no-op and deflates the
            # measured effect toward zero.
            continue
        if require_exact_length:
            if len(cand_seq) != own_len:
                continue
        elif own_len:
            if abs(len(cand_seq) - own_len) / own_len > policy.max_length_ratio_delta:
                continue
        out.append((cand_id, candidate))
    return out


def choose_donor(
    record: OASRecord,
    pool: Sequence[tuple[str, OASRecord]],
    direction: str,
    *,
    require_exact_length: bool,
    policy: DonorPolicy,
    rng: random.Random,
) -> tuple[str, OASRecord] | None:
    candidates = eligible_donors(
        record, pool, direction,
        require_exact_length=require_exact_length, policy=policy,
    )
    if not candidates:
        return None
    return candidates[rng.randrange(len(candidates))]


def _with_partner(record: OASRecord, direction: str, donor: OASRecord | None) -> OASRecord:
    variant = copy.copy(record)
    if direction == HEAVY_GIVEN_LIGHT:
        variant.sequence_light = None if donor is None else donor.sequence_light
        if donor is not None:
            variant.light_locus = donor.light_locus
    else:
        variant.sequence_heavy = None if donor is None else donor.sequence_heavy
        if donor is not None:
            variant.heavy_locus = donor.heavy_locus
    return variant


def _synthetic_filler_donor(
    record: OASRecord, direction: str, filler: str = DEFAULT_FILLER_RESIDUE
) -> OASRecord:
    """A same-length synthetic partner. A control, not a neutral context."""
    if len(filler) != 1:
        raise FrozenInputsError(
            f"filler residue must be a single character, got {filler!r}; the encoder "
            "maps sequences character by character, so a multi-character token would "
            "change the partner's length"
        )
    donor = copy.copy(record)
    length = len(_partner_sequence(record, direction))
    if direction == HEAVY_GIVEN_LIGHT:
        donor.sequence_light = filler * length
    else:
        donor.sequence_heavy = filler * length
    return donor


# --------------------------------------------------------------------------- #
# Case builders
# --------------------------------------------------------------------------- #
def _require_full_span_collator(collator: MLMCollator) -> None:
    """The full-span probe needs BOTH knobs, not just the selection one.

    `hcdr3_mask_mode="full_span"` selects the whole span as targets, but
    replacement still defaults to BERT 80/10/10 -- so 10% of the span stays
    VISIBLE and 10% becomes a random residue. A ten-residue HCDR3 was observed
    with eight [MASK], one random substitution and one correct residue left in
    plain sight. Scoring that as "full-span recovery" measures a partly revealed
    span. Both knobs are required, and an ordinary sampled-MLM collator is
    refused under this probe's name.
    """
    mode = getattr(collator, "hcdr3_mask_mode", None)
    if mode != "full_span":
        raise FrozenInputsError(
            f"the full-span probe requires hcdr3_mask_mode='full_span'; got {mode!r}"
        )
    strategy = getattr(collator, "mask_replacement_strategy", None)
    if strategy != "always_mask":
        raise FrozenInputsError(
            "the full-span probe requires mask_replacement_strategy='always_mask'; "
            f"got {strategy!r}. Under BERT 80/10/10 part of the span stays visible, "
            "so the probe would not measure recovery of a hidden span."
        )


def build_single_chain_cases(
    records: Sequence[OASRecord],
    collator: MLMCollator,
    probe: str,
    *,
    ledger: ExclusionLedger | None = None,
) -> list[FrozenCase]:
    """Freeze one corrupted view per record, straight from the production path."""
    ledger = ledger if ledger is not None else ExclusionLedger()
    if probe == PROBE_HCDR3_FULL_SPAN:
        _require_full_span_collator(collator)

    cases: list[FrozenCase] = []
    for index, record in enumerate(records):
        ids = _encode(collator, record)
        masked, labels, target_mask = _corrupt(collator, record, ids)
        positions = tuple(int(p) for p in torch.nonzero(target_mask, as_tuple=True)[0])
        if not positions:
            ledger.drop(f"{probe}:no_targets")
            continue
        if probe == PROBE_HCDR3_FULL_SPAN:
            mask_id = collator.tokenizer.mask_id
            if any(int(masked[p]) != mask_id for p in positions):
                ledger.drop(f"{probe}:target_not_fully_hidden")
                continue
            h_start, h_end, h_valid = collator._heavy_hcdr3_token_span(ids, record)
            if not h_valid or tuple(range(h_start, h_end)) != positions:
                # The declared targets must BE the production HCDR3 span, not a
                # subset that happens to sit inside it.
                ledger.drop(f"{probe}:targets_are_not_the_full_production_span")
                continue
        artifact_id, id_source = derive_artifact_id(record)
        cases.append(
            FrozenCase(
                case_id=f"{probe}:{index:06d}:{artifact_id}",
                probe=probe,
                input_ids=masked,
                attention_mask=torch.ones_like(masked),
                labels=labels,
                target_positions=positions,
                artifact_id=artifact_id,
                artifact_id_source=id_source,
                regions=_build_regions(collator, record, ids, positions),
                notes={"n_targets": len(positions)},
            )
        )
    return cases


def build_partner_cases(
    records: Sequence[OASRecord],
    collator: MLMCollator,
    direction: str,
    *,
    donor_pool: Sequence[OASRecord] | None = None,
    policy: DonorPolicy | None = None,
    filler_residue: str = DEFAULT_FILLER_RESIDUE,
    ledger: ExclusionLedger | None = None,
) -> list[FrozenCase]:
    """Freeze the three partner conditions for one direction."""
    if direction not in DIRECTIONS:
        raise FrozenInputsError(f"unknown direction {direction!r}")
    policy = policy or DonorPolicy()
    ledger = ledger if ledger is not None else ExclusionLedger()
    raw_pool = list(donor_pool if donor_pool is not None else records)
    pool = [(derive_artifact_id(r)[0], r) for r in raw_pool]
    require_exact = direction == LIGHT_GIVEN_HEAVY
    rng = random.Random(f"{policy.seed}:{direction}")

    cases: list[FrozenCase] = []
    for index, record in enumerate(records):
        if not (record.sequence_heavy or "").strip() or not (
            record.sequence_light or ""
        ).strip():
            ledger.drop(f"{direction}:not_paired")
            continue

        span = _predicted_span(record, direction)
        native_ids = _encode(collator, record)
        if span[1] > native_ids.numel():
            ledger.drop(f"{direction}:predicted_chain_truncated")
            continue

        masked, labels, target_mask = _corrupt(collator, record, native_ids)
        in_span = torch.zeros_like(target_mask)
        in_span[span[0]:span[1]] = True
        target_mask = target_mask & in_span
        positions = tuple(int(p) for p in torch.nonzero(target_mask, as_tuple=True)[0])
        if not positions:
            ledger.drop(f"{direction}:no_targets_in_predicted_chain")
            continue

        chosen = choose_donor(
            record, pool, direction,
            require_exact_length=require_exact, policy=policy, rng=rng,
        )
        if chosen is None:
            ledger.drop(f"{direction}:no_admissible_donor")
            continue
        donor_id, donor = chosen

        if direction == HEAVY_GIVEN_LIGHT:
            absent_donor, absent_mechanism = None, ABSENT_CHAIN_REMOVED
        else:
            absent_donor = _synthetic_filler_donor(record, direction, filler_residue)
            absent_mechanism = ABSENT_SYNTHETIC_FILLER

        artifact_id, id_source = derive_artifact_id(record)
        group = f"{PROBE_PARTNER}:{direction}:{index:06d}:{artifact_id}"
        regions = _build_regions(collator, record, native_ids, positions)

        variants = {
            NATIVE: (record, None),
            MATCHED_ALTERNATIVE: (_with_partner(record, direction, donor), donor_id),
            ABSENT: (_with_partner(record, direction, absent_donor), None),
        }

        built: list[FrozenCase] = []
        for condition, (variant, cond_donor_id) in variants.items():
            ids = _encode(collator, variant)
            if span[1] > ids.numel():
                built = []
                break
            v_masked = ids.clone()
            v_labels = torch.full_like(ids, MLM_IGNORE_INDEX)
            for pos in positions:
                v_masked[pos] = masked[pos]
                v_labels[pos] = labels[pos]
            built.append(
                FrozenCase(
                    case_id=f"{group}:{condition}",
                    probe=PROBE_PARTNER,
                    input_ids=v_masked,
                    attention_mask=torch.ones_like(v_masked),
                    labels=v_labels,
                    target_positions=positions,
                    artifact_id=artifact_id,
                    artifact_id_source=id_source,
                    direction=direction,
                    condition=condition,
                    predicted_span=span,
                    donor_artifact_id=cond_donor_id,
                    absent_mechanism=absent_mechanism if condition == ABSENT else None,
                    regions=regions,
                    notes={
                        "n_targets": len(positions),
                        "group": group,
                        **(
                            {"filler_residue": filler_residue}
                            if condition == ABSENT
                            and absent_mechanism == ABSENT_SYNTHETIC_FILLER
                            else {}
                        ),
                    },
                )
            )
        if len(built) == len(CONDITIONS):
            cases.extend(built)
        else:
            ledger.drop(f"{direction}:variant_truncated")
    return cases


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def verify_frozen(
    payload: Mapping[str, Any], *, required_probes: Sequence[str] = PROBES
) -> None:
    """Enforce the artifact's contract. Raises on any violation."""
    if payload.get("schema") != FROZEN_SCHEMA:
        raise FrozenInputsError(f"unknown schema {payload.get('schema')!r}")

    for probe in required_probes:
        if not payload["probes"].get(probe):
            raise EmptyProbeError(
                f"required probe {probe!r} has no cases. Every check over it would "
                "be vacuously true, so an empty probe is a failure, not a pass. "
                f"Exclusions: {payload.get('exclusions', {})}"
            )

    seen_ids: set[str] = set()
    groups: dict[str, list[FrozenCase]] = {}
    for probe, entries in payload["probes"].items():
        if probe not in PROBES:
            raise FrozenInputsError(f"unknown probe {probe!r}")
        for entry in entries:
            case = FrozenCase.from_payload(entry)
            if case.case_id in seen_ids:
                raise FrozenInputsError(f"duplicate case_id {case.case_id!r}")
            seen_ids.add(case.case_id)
            if case.labels.shape != case.input_ids.shape:
                raise FrozenInputsError(f"{case.case_id}: labels/input shape mismatch")
            if case.attention_mask.shape != case.input_ids.shape:
                raise FrozenInputsError(f"{case.case_id}: attention/input shape mismatch")
            if not case.target_positions:
                raise FrozenInputsError(f"{case.case_id}: no targets, nothing to score")
            for pos in case.target_positions:
                if case.labels[pos] == MLM_IGNORE_INDEX:
                    raise FrozenInputsError(
                        f"{case.case_id}: position {pos} is a declared target but "
                        "carries no label"
                    )
            off = case.labels.clone()
            off[list(case.target_positions)] = MLM_IGNORE_INDEX
            if bool((off != MLM_IGNORE_INDEX).any()):
                raise FrozenInputsError(
                    f"{case.case_id}: labels exist outside the declared targets"
                )
            if probe == PROBE_PARTNER:
                span = case.predicted_span
                if not span or span[1] <= span[0] or span[1] > case.input_ids.numel():
                    raise FrozenInputsError(
                        f"{case.case_id}: invalid predicted_span {span}"
                    )
                if not all(span[0] <= p < span[1] for p in case.target_positions):
                    raise FrozenInputsError(
                        f"{case.case_id}: targets fall outside the predicted span"
                    )
                groups.setdefault(str(case.notes.get("group")), []).append(case)

    for group_id, members in groups.items():
        counts = Counter(c.condition for c in members)
        if set(counts) != set(CONDITIONS) or set(counts.values()) != {1}:
            raise FrozenInputsError(
                f"{group_id}: expected exactly one of each of {sorted(CONDITIONS)}, "
                f"got {dict(counts)}"
            )
        if len({c.direction for c in members}) != 1:
            raise FrozenInputsError(f"{group_id}: inconsistent direction")
        if len({c.artifact_id for c in members}) != 1:
            raise FrozenInputsError(f"{group_id}: inconsistent artifact_id")
        reference = next(c for c in members if c.condition == NATIVE)
        lo, hi = reference.predicted_span
        for other in members:
            if other.condition == NATIVE:
                continue
            if other.target_positions != reference.target_positions:
                raise PositionDriftError(
                    f"{group_id}: target positions differ between {NATIVE} and "
                    f"{other.condition}; the predicted chain moved"
                )
            if other.predicted_span != reference.predicted_span:
                raise PositionDriftError(f"{group_id}: predicted_span differs")
            # The WHOLE predicted chain, not just the targets. A changed residue
            # elsewhere in the same chain is context the model reads.
            if not torch.equal(other.input_ids[lo:hi], reference.input_ids[lo:hi]):
                raise PositionDriftError(
                    f"{group_id}: the predicted chain's tokens differ between "
                    f"{NATIVE} and {other.condition} outside the target positions"
                )
            if not torch.equal(
                other.attention_mask[lo:hi], reference.attention_mask[lo:hi]
            ):
                raise PositionDriftError(
                    f"{group_id}: the predicted chain's attention mask differs"
                )
            idx = list(reference.target_positions)
            if not torch.equal(other.labels[idx], reference.labels[idx]):
                raise PositionDriftError(f"{group_id}: labels differ")


def build_frozen_benchmark(
    *,
    stage1_records: Sequence[OASRecord],
    paired_records: Sequence[OASRecord],
    retention_collator: MLMCollator,
    partner_collator: MLMCollator,
    full_span_collator: MLMCollator,
    donor_policy: DonorPolicy | None = None,
    provenance: Mapping[str, Any] | None = None,
    required_probes: Sequence[str] = PROBES,
) -> dict[str, Any]:
    """Build and VERIFY the artifact. A violating artifact is never returned."""
    policy = donor_policy or DonorPolicy()
    ledger = ExclusionLedger()

    partner_cases = [
        case
        for direction in DIRECTIONS
        for case in build_partner_cases(
            paired_records, partner_collator, direction, policy=policy, ledger=ledger
        )
    ]
    probes = {
        PROBE_STAGE1_RETENTION: [
            c.as_payload()
            for c in build_single_chain_cases(
                stage1_records, retention_collator, PROBE_STAGE1_RETENTION, ledger=ledger
            )
        ],
        PROBE_PARTNER: [c.as_payload() for c in partner_cases],
        PROBE_HCDR3_FULL_SPAN: [
            c.as_payload()
            for c in build_single_chain_cases(
                paired_records, full_span_collator, PROBE_HCDR3_FULL_SPAN, ledger=ledger
            )
        ],
    }
    donor_use = Counter(
        c.donor_artifact_id for c in partner_cases if c.donor_artifact_id is not None
    )
    groups_by_direction = Counter(
        c.direction for c in partner_cases if c.condition == NATIVE
    )
    payload = {
        "schema": FROZEN_SCHEMA,
        "provenance": dict(provenance or {}),
        "donor_policy": policy.as_payload(),
        "exclusions": ledger.as_payload(),
        "donor_usage": {
            "distinct_donors": len(donor_use),
            "groups_with_a_donor": sum(donor_use.values()),
            "groups_by_direction": dict(groups_by_direction),
            "most_reused": donor_use.most_common(5),
        },
        "probes": probes,
    }
    verify_frozen(payload, required_probes=required_probes)
    payload["tensor_digest"] = tensor_digest(payload)
    payload["semantic_digest"] = semantic_digest(payload)
    return payload


# --------------------------------------------------------------------------- #
# Digests
# --------------------------------------------------------------------------- #
def tensor_digest(payload: Mapping[str, Any]) -> str:
    """Hash of what the model SAW."""
    hasher = hashlib.sha256()
    for probe in PROBES:
        hasher.update(probe.encode("utf-8"))
        for entry in payload["probes"].get(probe, []):
            hasher.update(str(entry["case_id"]).encode("utf-8"))
            for key in ("input_ids", "attention_mask", "labels"):
                hasher.update(entry[key].detach().cpu().to(torch.int64).numpy().tobytes())
            hasher.update(str(entry["target_positions"]).encode("utf-8"))
    return hasher.hexdigest()


def semantic_digest(payload: Mapping[str, Any]) -> str:
    """Hash of what the ANALYSIS will condition on.

    Direction, donor identity, grouping, absent mechanism and region spans decide
    what a number MEANS. Relabelling a case's direction, or swapping its donor
    id, changes no tensor -- so the tensor digest cannot see it, and a report
    built on the relabelled artifact would be wrong while citing a matching hash.
    """
    fields = []
    for probe in PROBES:
        for entry in payload["probes"].get(probe, []):
            fields.append(
                {
                    "case_id": entry["case_id"],
                    "probe": entry["probe"],
                    "artifact_id": entry["artifact_id"],
                    "artifact_id_source": entry["artifact_id_source"],
                    "direction": entry.get("direction"),
                    "condition": entry.get("condition"),
                    "predicted_span": entry.get("predicted_span"),
                    "donor_artifact_id": entry.get("donor_artifact_id"),
                    "absent_mechanism": entry.get("absent_mechanism"),
                    "regions": entry.get("regions"),
                    "group": (entry.get("notes") or {}).get("group"),
                    "target_positions": entry["target_positions"],
                }
            )
    blob = json.dumps(
        {"donor_policy": payload.get("donor_policy"), "cases": fields},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_frozen(path: str | Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), path)
    return path


def load_frozen(
    path: str | Path, *, verify: bool = True, required_probes: Sequence[str] = PROBES
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if verify:
        verify_frozen(payload, required_probes=required_probes)
        for name, fn in (
            ("tensor_digest", tensor_digest),
            ("semantic_digest", semantic_digest),
        ):
            recorded = payload.get(name)
            actual = fn(payload)
            if recorded is not None and recorded != actual:
                raise FrozenInputsError(
                    f"frozen artifact {name} mismatch: recorded {recorded}, "
                    f"actual {actual}"
                )
    return payload


def iter_cases(payload: Mapping[str, Any], probe: str) -> Iterable[FrozenCase]:
    for entry in payload["probes"].get(probe, []):
        yield FrozenCase.from_payload(entry)
