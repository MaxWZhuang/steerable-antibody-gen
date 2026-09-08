"""The frozen benchmark artifact: build it once, score it forever.

WHY FROZEN INPUTS RATHER THAN A FROZEN SEED
-------------------------------------------
`MLMCollator` draws its corruption from an RNG. Reproducing a comparison by
re-seeding survives exactly as long as nobody touches the collator: change the
order of a draw, add a knob, reorder a loop, and the "identical masks" two
checkpoints were compared under silently stop being identical. The artifact
built here stores the corrupted token ids, the labels, the attention masks, the
target positions and the donor assignments. Scoring loads them. There is no
second draw, so there is nothing to drift.

THE POSITIONAL TRAP THIS MODULE EXISTS TO AVOID
-----------------------------------------------
A paired record encodes as::

    [CLS] [IGH] H H H ... [SEP] [IGK] L L L ... [EOS]

The heavy chain therefore occupies positions 2 .. 2+len(H)-1 and the light chain
starts at 2+len(H)+2. That asymmetry is the whole difficulty of a two-directional
partner probe:

- **Predicting heavy while varying light** is easy. The light chain sits AFTER
  the heavy chain, so swapping it cannot move a single heavy position. Even
  deleting it outright leaves the heavy span where it was.
- **Predicting light while varying heavy** is not. The heavy chain sits BEFORE
  the light chain, so a donor heavy of a different length SHIFTS the entire light
  chain. Under RoPE the light residues then rotate at different positions, and a
  measured "partner effect" would be partly a positional effect. This module
  requires an EXACT token-length match for a partner that precedes the predicted
  chain, and drops examples that have no such donor rather than quietly
  accepting a shifted one.

The same asymmetry decides what "absent" can mean. Removing the light chain
preserves every heavy position, so the heavy direction gets a true removal
(`ABSENT_CHAIN_REMOVED`). Removing the heavy chain would shift the light chain,
so the light direction instead replaces the heavy residues with a neutral
residue at the identical length (`ABSENT_NEUTRAL_FILLER`) -- information removed,
positions preserved. The two are NOT the same condition and the artifact records which one
each case used, so a report can never silently average them.

THE INVARIANT
-------------
Within one case, every condition must present the predicted chain to the model
identically: same absolute target positions, same corrupted tokens at those
positions, same labels. Only the partner may differ. `verify_frozen` enforces
this and `build_frozen_benchmark` runs it before returning, so an artifact that
violates it cannot be saved.

Production path: encoding goes through `MLMCollator._encode_record` and
corruption through `MLMCollator._mask_tokens` -- the same code training uses.
These are private helpers, and calling them is deliberate: a reimplementation
here would be a second encoding call site, which is this repository's most
expensive and least visible bug class.
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from smallAntibodyGen.data.MLMCollator import MLMCollator, OASRecord

FROZEN_SCHEMA = "frozen-eval-inputs/1"

#: Probe names. Kept as constants so a report cannot invent a fourth silently.
PROBE_STAGE1_RETENTION = "stage1_retention"
PROBE_PARTNER = "partner_dependence"
PROBE_HCDR3_FULL_SPAN = "hcdr3_full_span"
PROBES = (PROBE_STAGE1_RETENTION, PROBE_PARTNER, PROBE_HCDR3_FULL_SPAN)

#: Which chain is being predicted, and therefore which chain is varied.
HEAVY_GIVEN_LIGHT = "heavy_given_light"
LIGHT_GIVEN_HEAVY = "light_given_heavy"
DIRECTIONS = (HEAVY_GIVEN_LIGHT, LIGHT_GIVEN_HEAVY)

NATIVE = "native"
MATCHED_ALTERNATIVE = "matched_alternative"
ABSENT = "absent"
CONDITIONS = (NATIVE, MATCHED_ALTERNATIVE, ABSENT)

#: How `absent` was realized. See the module docstring -- these are different
#: conditions and must never be pooled.
ABSENT_CHAIN_REMOVED = "chain_removed"
ABSENT_NEUTRAL_FILLER = "neutral_residue_filler"

#: Default filler residue for `ABSENT_NEUTRAL_FILLER`. "X" is the standard
#: unknown-residue code and IS in the tokenizer vocabulary, so the model sees an
#: in-distribution token carrying essentially no identity information. The
#: tempting alternative -- a character outside the vocabulary, which maps to
#: [UNK] -- removes more information but is out of distribution, and a model
#: behaving oddly on a token it never trained on is its own confound. Whichever
#: is used is recorded in the artifact.
#:
#: NOT usable here: the literal text of `tokenizer.mask_token`. It is "[MASK]",
#: and the encoder maps sequences CHARACTER BY CHARACTER, so it would expand to
#: seven tokens per residue -- four of which (M, A, S, K) are real amino acids.
DEFAULT_FILLER_RESIDUE = "X"

MLM_IGNORE_INDEX = -100


class FrozenInputsError(ValueError):
    """Raised when a frozen artifact violates its own contract."""


class PositionDriftError(FrozenInputsError):
    """The predicted chain moved between two conditions of one case.

    This is the failure the module exists to prevent: it means a reported
    partner effect would be confounded with a positional shift.
    """


@dataclass
class FrozenCase:
    """One scorable unit: fixed inputs, fixed targets, fixed provenance."""

    case_id: str
    probe: str
    input_ids: torch.Tensor          # [L] corrupted, model-ready
    attention_mask: torch.Tensor     # [L]
    labels: torch.Tensor             # [L], MLM_IGNORE_INDEX off-target
    target_positions: tuple[int, ...]
    record_id: str | None = None
    direction: str | None = None
    condition: str | None = None
    predicted_span: tuple[int, int] | None = None   # [start, end) of predicted chain
    donor_record_id: str | None = None
    absent_mechanism: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    def as_payload(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "probe": self.probe,
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "labels": self.labels,
            "target_positions": list(self.target_positions),
            "record_id": self.record_id,
            "direction": self.direction,
            "condition": self.condition,
            "predicted_span": list(self.predicted_span) if self.predicted_span else None,
            "donor_record_id": self.donor_record_id,
            "absent_mechanism": self.absent_mechanism,
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
            record_id=payload.get("record_id"),
            direction=payload.get("direction"),
            condition=payload.get("condition"),
            predicted_span=tuple(span) if span else None,
            donor_record_id=payload.get("donor_record_id"),
            absent_mechanism=payload.get("absent_mechanism"),
            notes=dict(payload.get("notes") or {}),
        )


# --------------------------------------------------------------------------- #
# Encoding helpers -- all routed through the production collator
# --------------------------------------------------------------------------- #
def _encode(collator: MLMCollator, record: OASRecord) -> torch.Tensor:
    """Encode via the PRODUCTION path. Never reimplement this."""
    return torch.tensor(collator._encode_record(record), dtype=torch.long)


def _corrupt(
    collator: MLMCollator, record: OASRecord, input_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw production corruption once for one record."""
    batch = input_ids.unsqueeze(0)
    masked, labels, target_mask = collator._mask_tokens(batch, [record])
    return masked[0], labels[0], target_mask[0]


def heavy_span(collator: MLMCollator, record: OASRecord) -> tuple[int, int]:
    """[start, end) of the heavy residues in a PAIRED encoding.

    Layout is `[CLS] [IGH] H... [SEP] [IGK] L... [EOS]`, so the heavy residues
    always begin at index 2 regardless of the partner.
    """
    heavy = (record.sequence_heavy or "").strip()
    return (2, 2 + len(heavy))


def light_span(collator: MLMCollator, record: OASRecord) -> tuple[int, int]:
    """[start, end) of the light residues in a PAIRED encoding.

    `2 + len(H)` is the `[SEP]`, `+1` the light chain token, so the light
    residues begin at `2 + len(H) + 2`. This is exactly why a donor heavy chain
    of a different length shifts the whole light chain.
    """
    heavy = (record.sequence_heavy or "").strip()
    light = (record.sequence_light or "").strip()
    start = 2 + len(heavy) + 2
    return (start, start + len(light))


def _predicted_span(collator: MLMCollator, record: OASRecord, direction: str) -> tuple[int, int]:
    return (
        heavy_span(collator, record)
        if direction == HEAVY_GIVEN_LIGHT
        else light_span(collator, record)
    )


def _with_partner(record: OASRecord, direction: str, donor: OASRecord | None) -> OASRecord:
    """Return a copy of `record` whose PARTNER chain comes from `donor`."""
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


def _partner_length(record: OASRecord, direction: str) -> int:
    partner = (
        record.sequence_light if direction == HEAVY_GIVEN_LIGHT else record.sequence_heavy
    ) or ""
    return len(partner.strip())


def _partner_locus(record: OASRecord, direction: str) -> str:
    return (
        (record.light_locus or "IGK")
        if direction == HEAVY_GIVEN_LIGHT
        else (record.heavy_locus or "IGH")
    )


def find_matched_donor(
    record: OASRecord,
    pool: Sequence[OASRecord],
    direction: str,
    *,
    require_exact_length: bool,
) -> OASRecord | None:
    """Pick an alternative partner from `pool`.

    `require_exact_length` is NOT a tuning preference. When the partner precedes
    the predicted chain (predicting light), an inexact length shifts every
    predicted position, so the match must be exact or the case must be dropped.
    """
    own_id = record.record_id
    own_len = _partner_length(record, direction)
    own_locus = _partner_locus(record, direction)
    for candidate in pool:
        if candidate.record_id == own_id:
            continue
        if _partner_locus(candidate, direction) != own_locus:
            continue
        cand_len = _partner_length(candidate, direction)
        if cand_len == 0:
            continue
        if require_exact_length:
            if cand_len != own_len:
                continue
        elif cand_len == 0:
            continue
        # A donor identical to the native partner would make the condition a
        # no-op and quietly deflate the measured effect toward zero.
        native = (
            record.sequence_light if direction == HEAVY_GIVEN_LIGHT else record.sequence_heavy
        ) or ""
        cand = (
            candidate.sequence_light
            if direction == HEAVY_GIVEN_LIGHT
            else candidate.sequence_heavy
        ) or ""
        if cand.strip() == native.strip():
            continue
        return candidate
    return None


def _neutral_filler_donor(
    record: OASRecord, direction: str, filler: str = DEFAULT_FILLER_RESIDUE
) -> OASRecord:
    """A partner of identical length carrying no identity information.

    Length is preserved exactly, which is the entire point: this is the only way
    to remove the partner's INFORMATION from the light direction without also
    moving the predicted chain.
    """
    if len(filler) != 1:
        raise FrozenInputsError(
            f"filler residue must be a single character, got {filler!r}; the encoder "
            "maps sequences character by character, so a multi-character token would "
            "change the partner's length"
        )
    donor = copy.copy(record)
    length = _partner_length(record, direction)
    if direction == HEAVY_GIVEN_LIGHT:
        donor.sequence_light = filler * length
    else:
        donor.sequence_heavy = filler * length
    donor.record_id = f"{record.record_id}::neutral_filler"
    return donor


# --------------------------------------------------------------------------- #
# Case builders
# --------------------------------------------------------------------------- #
def build_single_chain_cases(
    records: Sequence[OASRecord],
    collator: MLMCollator,
    probe: str,
) -> list[FrozenCase]:
    """Freeze one corrupted view per record, straight from the production path.

    Used for stage-1 retention and for full-span HCDR3 recovery; the two differ
    only in the collator they are handed (`hcdr3_mask_mode="full_span"` for the
    latter), which is why they share a builder rather than duplicating one.
    """
    cases: list[FrozenCase] = []
    for index, record in enumerate(records):
        ids = _encode(collator, record)
        masked, labels, target_mask = _corrupt(collator, record, ids)
        positions = tuple(int(p) for p in torch.nonzero(target_mask, as_tuple=True)[0])
        if not positions:
            # No targets means nothing to score. Dropping it here keeps the
            # denominator honest instead of averaging in a zero-weight row.
            continue
        cases.append(
            FrozenCase(
                case_id=f"{probe}:{index:06d}",
                probe=probe,
                input_ids=masked,
                attention_mask=torch.ones_like(masked),
                labels=labels,
                target_positions=positions,
                record_id=record.record_id,
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
    filler_residue: str = DEFAULT_FILLER_RESIDUE,
) -> list[FrozenCase]:
    """Freeze the three partner conditions for one direction.

    The predicted chain's corruption is drawn ONCE, on the native variant, and
    then replayed byte-for-byte into the other two conditions. Only the partner
    differs.
    """
    if direction not in DIRECTIONS:
        raise FrozenInputsError(f"unknown direction {direction!r}")
    pool = list(donor_pool if donor_pool is not None else records)
    # The partner PRECEDES the predicted chain only when predicting the light
    # chain; that is the case which cannot tolerate a length change.
    require_exact = direction == LIGHT_GIVEN_HEAVY

    cases: list[FrozenCase] = []
    for index, record in enumerate(records):
        if not (record.sequence_heavy or "").strip():
            continue
        if not (record.sequence_light or "").strip():
            continue

        span = _predicted_span(collator, record, direction)
        native_ids = _encode(collator, record)
        if span[1] > native_ids.numel():
            # Truncated by max_length: the predicted chain is not fully present,
            # so this case cannot honour its own contract.
            continue

        masked, labels, target_mask = _corrupt(collator, record, native_ids)
        in_span = torch.zeros_like(target_mask)
        in_span[span[0]:span[1]] = True
        target_mask = target_mask & in_span
        positions = tuple(int(p) for p in torch.nonzero(target_mask, as_tuple=True)[0])
        if not positions:
            continue

        donor = find_matched_donor(record, pool, direction, require_exact_length=require_exact)
        if donor is None:
            # No admissible alternative. Dropping beats silently relaxing the
            # length rule and reporting a positional artefact as a partner
            # effect.
            continue

        if direction == HEAVY_GIVEN_LIGHT:
            absent_donor, absent_mechanism = None, ABSENT_CHAIN_REMOVED
        else:
            absent_donor = _neutral_filler_donor(record, direction, filler_residue)
            absent_mechanism = ABSENT_NEUTRAL_FILLER

        variants = {
            NATIVE: (record, None),
            MATCHED_ALTERNATIVE: (_with_partner(record, direction, donor), donor.record_id),
            ABSENT: (_with_partner(record, direction, absent_donor), None),
        }

        case_id = f"{PROBE_PARTNER}:{direction}:{index:06d}"
        built: list[FrozenCase] = []
        for condition, (variant, donor_id) in variants.items():
            ids = _encode(collator, variant)
            if span[1] > ids.numel():
                built = []
                break
            v_masked = ids.clone()
            v_labels = torch.full_like(ids, MLM_IGNORE_INDEX)
            # Replay the frozen corruption at the SAME absolute positions.
            for pos in positions:
                v_masked[pos] = masked[pos]
                v_labels[pos] = labels[pos]
            built.append(
                FrozenCase(
                    case_id=f"{case_id}:{condition}",
                    probe=PROBE_PARTNER,
                    input_ids=v_masked,
                    attention_mask=torch.ones_like(v_masked),
                    labels=v_labels,
                    target_positions=positions,
                    record_id=record.record_id,
                    direction=direction,
                    condition=condition,
                    predicted_span=span,
                    donor_record_id=donor_id,
                    absent_mechanism=absent_mechanism if condition == ABSENT else None,
                    notes={
                        "n_targets": len(positions),
                        "group": case_id,
                        **(
                            {"filler_residue": filler_residue}
                            if condition == ABSENT
                            and absent_mechanism == ABSENT_NEUTRAL_FILLER
                            else {}
                        ),
                    },
                )
            )
        if len(built) == len(CONDITIONS):
            cases.extend(built)
    return cases


# --------------------------------------------------------------------------- #
# Invariants
# --------------------------------------------------------------------------- #
def verify_frozen(payload: Mapping[str, Any]) -> None:
    """Enforce the artifact's contract. Raises on any violation.

    The load-bearing check is the partner one: within a group, the predicted
    chain must be presented identically under every condition.
    """
    if payload.get("schema") != FROZEN_SCHEMA:
        raise FrozenInputsError(f"unknown schema {payload.get('schema')!r}")

    groups: dict[str, list[FrozenCase]] = {}
    for probe, entries in payload["probes"].items():
        if probe not in PROBES:
            raise FrozenInputsError(f"unknown probe {probe!r}")
        for entry in entries:
            case = FrozenCase.from_payload(entry)
            if case.labels.shape != case.input_ids.shape:
                raise FrozenInputsError(f"{case.case_id}: labels/input shape mismatch")
            if not case.target_positions:
                raise FrozenInputsError(f"{case.case_id}: no targets, nothing to score")
            for pos in case.target_positions:
                if case.labels[pos] == MLM_IGNORE_INDEX:
                    raise FrozenInputsError(
                        f"{case.case_id}: position {pos} is a declared target but "
                        "carries no label"
                    )
            off_target = case.labels.clone()
            off_target[list(case.target_positions)] = MLM_IGNORE_INDEX
            if bool((off_target != MLM_IGNORE_INDEX).any()):
                raise FrozenInputsError(
                    f"{case.case_id}: labels exist outside the declared targets, so "
                    "the scored set is not the declared set"
                )
            if probe == PROBE_PARTNER:
                groups.setdefault(str(case.notes.get("group")), []).append(case)

    for group_id, members in groups.items():
        conditions = {c.condition for c in members}
        if conditions != set(CONDITIONS):
            raise FrozenInputsError(
                f"{group_id}: expected {sorted(CONDITIONS)}, found {sorted(conditions)}"
            )
        reference = next(c for c in members if c.condition == NATIVE)
        for other in members:
            if other.condition == NATIVE:
                continue
            if other.target_positions != reference.target_positions:
                raise PositionDriftError(
                    f"{group_id}: target positions differ between {NATIVE} and "
                    f"{other.condition}; the predicted chain moved"
                )
            idx = list(reference.target_positions)
            if not torch.equal(other.input_ids[idx], reference.input_ids[idx]):
                raise PositionDriftError(
                    f"{group_id}: corrupted tokens differ between {NATIVE} and "
                    f"{other.condition} at the predicted positions"
                )
            if not torch.equal(other.labels[idx], reference.labels[idx]):
                raise PositionDriftError(
                    f"{group_id}: labels differ between {NATIVE} and {other.condition}"
                )


def build_frozen_benchmark(
    *,
    stage1_records: Sequence[OASRecord],
    paired_records: Sequence[OASRecord],
    retention_collator: MLMCollator,
    partner_collator: MLMCollator,
    full_span_collator: MLMCollator,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and VERIFY the artifact. A violating artifact is never returned."""
    probes = {
        PROBE_STAGE1_RETENTION: [
            c.as_payload()
            for c in build_single_chain_cases(
                stage1_records, retention_collator, PROBE_STAGE1_RETENTION
            )
        ],
        PROBE_PARTNER: [
            c.as_payload()
            for direction in DIRECTIONS
            for c in build_partner_cases(paired_records, partner_collator, direction)
        ],
        PROBE_HCDR3_FULL_SPAN: [
            c.as_payload()
            for c in build_single_chain_cases(
                paired_records, full_span_collator, PROBE_HCDR3_FULL_SPAN
            )
        ],
    }
    payload = {
        "schema": FROZEN_SCHEMA,
        "provenance": dict(provenance or {}),
        "probes": probes,
    }
    verify_frozen(payload)
    payload["digest"] = frozen_payload_digest(payload)
    return payload


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def frozen_payload_digest(payload: Mapping[str, Any]) -> str:
    """Content hash over the scored bytes, so a report can name its inputs.

    Deliberately covers the tensors and the declared targets only. Provenance
    text is descriptive; the TENSORS are what a checkpoint actually saw.
    """
    hasher = hashlib.sha256()
    for probe in PROBES:
        hasher.update(probe.encode("utf-8"))
        for entry in payload["probes"].get(probe, []):
            hasher.update(str(entry["case_id"]).encode("utf-8"))
            hasher.update(str(entry.get("condition")).encode("utf-8"))
            for key in ("input_ids", "attention_mask", "labels"):
                tensor = entry[key]
                hasher.update(tensor.detach().cpu().to(torch.int64).numpy().tobytes())
            hasher.update(bytes(str(entry["target_positions"]), "utf-8"))
    return hasher.hexdigest()


def save_frozen(path: str | Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), path)
    return path


def load_frozen(path: str | Path, *, verify: bool = True) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if verify:
        verify_frozen(payload)
        recorded = payload.get("digest")
        actual = frozen_payload_digest(payload)
        if recorded is not None and recorded != actual:
            raise FrozenInputsError(
                f"frozen artifact digest mismatch: recorded {recorded}, actual {actual}"
            )
    return payload


def iter_cases(payload: Mapping[str, Any], probe: str) -> Iterable[FrozenCase]:
    for entry in payload["probes"].get(probe, []):
        yield FrozenCase.from_payload(entry)
