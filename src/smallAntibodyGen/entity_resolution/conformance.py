"""
The claim and release planes: what a split claims, and the evidence that it holds.

WHY A CLAIM IS NOT A THRESHOLD
------------------------------
A contract stated only as measures and thresholds leaves a loophole: dependence
along an axis nobody named becomes technically permitted. So the contract here
has two layers. The CLAIM says what "unseen" means and what generalisation is
being asserted; the OPERATIONALISATION says which detectors and thresholds
approximate it. The claim is authoritative and the thresholds are declared
approximations of it, which means that when a counterfactual exposes a dependence
the operationalisation missed, that is a benchmark defect and not a model
behaviour to be accepted because the measure failed to name it.

This matters concretely in this repository. The stage-4 split keys on the
ANTIGEN, and measured on the shipped corpus 78.6% of validation rows have their
HCDR3 in train and 77.3% have their exact heavy chain in train. Under a claim
stated only as thresholds, those numbers look like a passing leakage audit,
because the audit measures the axis the thresholds name. Under a claim stated as
a claim, they are what they are: the antigen axis is held out and the antibody
axis is record-random, and the standard leakage line has to say so on every
number quoted.

WHAT THIS MODULE PROVIDES
-------------------------
The four Level-1 evidence artifacts, as fixed schemas with a validator:

===================  ==========================================================
Claim manifest       Claim class, population, operationalisation, thresholds
Split manifest       Exact record-to-side assignment and the group evidence
Guard report         Achieved ratio, component sizes, eligibility, error rates
Leakage line         One human-readable line generated from controlled fields
===================  ==========================================================

plus a conformance attestation that a validator produces from them.

WHAT LEVEL 1 DOES AND DOES NOT SUPPORT
--------------------------------------
Level 1 supports exactly one sentence: *held out under the declared relation for
this frozen snapshot*. It does not support claims about opaque upstream training
(the ESM antigen encoder's pretraining corpus is unknown and unknowable from
here), about unmeasured structural or functional equivalence, about live tools,
or about adaptive reuse of a validation set across many runs. Those need Levels 2
and 3, and the reducer below removes each unsupported channel from the strong
claim rather than letting the whole thing collapse to a boolean.

SEALING
-------
Manifests are sealed with `ancestor_quarantine.manifest_hash`, the same
canonical-JSON SHA-256 the frozen-universe machinery already uses, so there is
one hashing convention in this repository rather than two. Sealing here is
tamper-EVIDENT and not tamper-proof: it proves a manifest has not changed since
it was written, which is what a single-operator research repository needs. A
signed or externally timestamped seal is a Level-2 requirement and is deliberately
out of scope.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ..ancestor_quarantine import ManifestDamaged, manifest_hash, verify_manifest

#: Controlled claim vocabulary. Free-form leakage statements are too easy to
#: weaken invisibly, so a claim must be one of these and the reducer below can
#: only move DOWN this list.
CLAIM_CLASSES: Tuple[str, ...] = (
    "closed-book unseen-entity",
    "closed-book unseen-broader-group",
    "open-book unseen-entity",
    "within-entity temporal",
    "record-random",
    "descriptive-only",
)

#: Controlled relation vocabulary. Every benchmark must say which of these it
#: assessed, so that "declared relation" cannot quietly mean exact deduplication.
RELATION_CLASSES: Tuple[str, ...] = (
    "exact",
    "lexical-near-duplicate",
    "semantic",
    "structural",
    "metadata/container",
    "pairwise-containment",
    "aggregate-mosaic",
)

#: What a piece of evidence is allowed to do. `monitor` is included because
#: pretending it does not happen is worse than recording it, but a monitor row
#: without an owner, a metric, a threshold, a deadline and an escalation is just
#: an undocumented acceptance of leakage, and `validate_level_1` rejects one.
POLICY_ACTIONS: Tuple[str, ...] = (
    "mask", "exclude", "must-link", "test-ineligible", "monitor",
)


class ConformanceError(ValueError):
    """Raised when an artifact is malformed or internally inconsistent."""


@dataclass(frozen=True)
class Claim:
    """What is being asserted, in the controlled vocabulary.

    Attributes:
        claim_class: One of `CLAIM_CLASSES`.
        target_population: The population the claim is about.
        unit_of_generalisation: What "unseen" is unseen at -- the construct, the
            biological target, the antibody, or a pair. Two claims with the same
            class and different units are different claims.
        allowed_evaluation_context: What the system may condition on.
        prohibited_exposure_relationships: Relationships that must not straddle.
        permitted_exposure_relationships: Relationships that may straddle, named
            so that finding one later is not treated as a defect.
        exclusions: What is removed from the population entirely.
        estimand: The quantity being estimated.
    """

    claim_class: str
    target_population: str
    unit_of_generalisation: str
    allowed_evaluation_context: Tuple[str, ...]
    prohibited_exposure_relationships: Tuple[str, ...]
    permitted_exposure_relationships: Tuple[str, ...]
    exclusions: Tuple[str, ...]
    estimand: str

    def __post_init__(self) -> None:
        if self.claim_class not in CLAIM_CLASSES:
            raise ConformanceError(
                f"claim_class {self.claim_class!r} is not in the controlled "
                f"vocabulary {CLAIM_CLASSES}"
            )


@dataclass(frozen=True)
class Operationalisation:
    """How the claim is approximated, and where the approximation is known to fail.

    Attributes:
        relation_classes: Which of `RELATION_CLASSES` were assessed.
        unassessed_relation_classes: Which were not. Required, not optional: a
            benchmark that lists only what it checked reads as if it checked
            everything.
        detectors: How each assessed relation was detected.
        thresholds: The sealed operating point.
        masks: Material removed identically everywhere before evidence was
            recomputed.
        promotion_rules: The evidence-to-action table.
        known_blind_spots: Named failures of the approximation.
        calibration_population: What the thresholds were chosen on.
        model_capability_envelope: The model class the thresholds were chosen
            against. A threshold sweep measures data, task and model capability
            together, so a threshold quoted without this is not transferable.
    """

    relation_classes: Tuple[str, ...]
    unassessed_relation_classes: Tuple[str, ...]
    detectors: Mapping[str, str]
    thresholds: Mapping[str, Any]
    masks: Tuple[str, ...]
    promotion_rules: Tuple[Mapping[str, Any], ...]
    known_blind_spots: Tuple[str, ...]
    calibration_population: str
    model_capability_envelope: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in tuple(self.relation_classes) + tuple(
            self.unassessed_relation_classes
        ):
            if name not in RELATION_CLASSES:
                raise ConformanceError(
                    f"relation class {name!r} is not in {RELATION_CLASSES}"
                )
        overlap = set(self.relation_classes) & set(self.unassessed_relation_classes)
        if overlap:
            raise ConformanceError(
                f"relation classes cannot be both assessed and unassessed: {overlap}"
            )
        missing = (
            set(RELATION_CLASSES)
            - set(self.relation_classes)
            - set(self.unassessed_relation_classes)
        )
        if missing:
            raise ConformanceError(
                f"every relation class must be declared assessed or unassessed; "
                f"unaccounted for: {sorted(missing)}"
            )
        for rule in self.promotion_rules:
            action = rule.get("action")
            if action not in POLICY_ACTIONS:
                raise ConformanceError(
                    f"policy action {action!r} is not in {POLICY_ACTIONS}"
                )
            if action == "monitor":
                required = {"owner", "metric", "threshold", "deadline", "escalation"}
                absent = required - set(rule)
                if absent:
                    raise ConformanceError(
                        f"a monitor rule without {sorted(absent)} is an "
                        f"undocumented acceptance of leakage, not a policy"
                    )


def build_claim_manifest(
    *,
    name: str,
    claim: Claim,
    operationalisation: Operationalisation,
    population: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> Dict[str, Any]:
    """Assemble and seal a claim manifest.

    Args:
        name: The benchmark version's name.
        claim: The authoritative claim.
        operationalisation: The declared approximation of it.
        population: Snapshot, counts, eligibility and exclusion figures.
        provenance: What this was derived from. Required, and bound by the seal:
            a hash that does not cover provenance lets the inputs change while
            the manifest still verifies.

    Returns:
        The manifest, carrying its own ``manifest_sha256``.

    Raises:
        ConformanceError: When provenance is empty.
    """
    if not provenance:
        raise ConformanceError(
            "a claim manifest without provenance seals nothing: its hash would "
            "not bind the corpus, the code or the operating point it came from"
        )
    payload: Dict[str, Any] = {
        "schema": "claim-manifest/1",
        "name": name,
        "claim": asdict(claim),
        "operationalisation": asdict(operationalisation),
        "population": dict(population),
        "provenance": dict(provenance),
    }
    payload["manifest_sha256"] = manifest_hash(payload)
    return payload


def build_split_manifest(
    *,
    claim_manifest_sha256: str,
    claim_name: str,
    assignments: Mapping[str, str],
    group_of: Mapping[str, str],
    group_evidence: Mapping[str, Mapping[str, Any]],
    target_val_fraction: float,
    row_weights: Optional[Mapping[str, int]] = None,
) -> Dict[str, Any]:
    """Assemble and seal the exact record-to-side assignment.

    The assignment itself is sealed, not merely enough parameters to reconstruct
    it. Reconstructing requires the code and the corpus to be byte-identical
    forever; recording the assignment does not.

    Args:
        claim_manifest_sha256: The claim this split serves.
        claim_name: That claim's name, for readability.
        assignments: Group id to ``"train"`` or ``"val"``.
        group_of: Record key to group id.
        group_evidence: Group id to the evidence that made it one group.
        target_val_fraction: The ratio that was aimed at.
        row_weights: Optional record key to source-row count. Without it the
            largest group's share is measured in RECORDS, which is the wrong
            unit for the failure it exists to catch: in this corpus one target
            carries 63% of the rows while being one record among 9,574, so a
            share measured in records cannot see concentration that a share
            measured in rows makes obvious. Both are reported when weights are
            given, and the row share is the one the validator gates on.

    Returns:
        The sealed split manifest, including the ACHIEVED ratio and the largest
        group's share -- both of which belong in every snapshot, because a split
        that hit its target ratio by putting most of the corpus in one group has
        not done what the ratio suggests.
    """
    sides = {"train": 0, "val": 0}
    for group, side in assignments.items():
        if side not in sides:
            raise ConformanceError(f"unknown side {side!r} for group {group!r}")
    for record, group in group_of.items():
        side = assignments.get(group)
        if side is None:
            raise ConformanceError(f"record {record!r} is in unassigned group {group!r}")
        sides[side] += 1
    total = max(1, sides["train"] + sides["val"])
    sizes: Dict[str, int] = {}
    row_sizes: Dict[str, int] = {}
    for record, group in group_of.items():
        sizes[group] = sizes.get(group, 0) + 1
        if row_weights is not None:
            row_sizes[group] = row_sizes.get(group, 0) + row_weights.get(record, 0)
    largest = max(sizes.values()) if sizes else 0
    total_rows = max(1, sum(row_sizes.values())) if row_sizes else 0
    largest_rows = max(row_sizes.values()) if row_sizes else 0

    payload: Dict[str, Any] = {
        "schema": "split-manifest/1",
        "claim_name": claim_name,
        "claim_manifest_sha256": claim_manifest_sha256,
        "groups": len(sizes),
        "records": total,
        "rows": total_rows if row_sizes else None,
        "target_val_fraction": target_val_fraction,
        "achieved_val_fraction": sides["val"] / total,
        "largest_group_share": largest / total,
        "largest_group_row_share": (
            largest_rows / total_rows if row_sizes else None
        ),
        "group_size_distribution": _distribution(sorted(sizes.values())),
        "assignments": dict(sorted(assignments.items())),
        # The record-to-group map, not just the group-to-side map. "The exact
        # record-to-side assignment is sealed, not merely enough parameters to
        # reconstruct it" is the requirement, and reconstructing which record
        # went where from group ids alone needs the resolver, the corpus and the
        # code to stay byte-identical forever.
        "group_of_record": dict(sorted(group_of.items())),
        "group_evidence": {k: dict(v) for k, v in sorted(group_evidence.items())},
        "provenance": {"derived_from": claim_manifest_sha256},
    }
    payload["manifest_sha256"] = manifest_hash(payload)
    return payload


def _distribution(sizes: Sequence[int]) -> Dict[str, Any]:
    """Summarise a size distribution without hiding its tail."""
    if not sizes:
        return {"count": 0}
    return {
        "count": len(sizes),
        "min": sizes[0],
        "p50": sizes[len(sizes) // 2],
        "p90": sizes[int(len(sizes) * 0.9)],
        "max": sizes[-1],
        "top10": list(reversed(sizes[-10:])),
    }


def build_guard_report(
    *,
    claim_manifest_sha256: str,
    resolution_stats: Mapping[str, Any],
    calibration: Mapping[str, Any],
    audit: Mapping[str, Any],
    blocking_recall: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the component-health and error-rate report.

    Args:
        claim_manifest_sha256: The claim this report is evidence for.
        resolution_stats: `TargetIdentityResolution.stats()`.
        calibration: The calibration `ErrorReport`, rendered.
        audit: The audit `ErrorReport`, rendered.
        blocking_recall: A measured `BlockingRecallReport`, when one was run.

    Returns:
        The guard report.
    """
    return {
        "schema": "guard-report/1",
        "claim_manifest_sha256": claim_manifest_sha256,
        "resolution": dict(resolution_stats),
        "calibration": dict(calibration),
        "audit": dict(audit),
        "blocking_recall": dict(blocking_recall) if blocking_recall else None,
    }


@dataclass(frozen=True)
class LeakageLine:
    """The standard line that must accompany every number.

    Generated from controlled machine-readable fields rather than written by
    hand, so that exact deduplication cannot be described merely as "a declared
    relation", a narrow eligible residue cannot be presented as the full
    population, and an unknown exposure channel cannot disappear into a footnote.

    Attributes:
        claim: The claim class actually supported, after reduction.
        relations: Relation classes assessed.
        unassessed: Relation classes not assessed, and other unknown channels.
        conditioning: What the system was allowed to condition on.
        population: Snapshot and eligibility, in figures.
        conformance: Claimed level and profile.
        result_identity: Test version, train snapshot and exposure state.
        status: Whether the numbers are valid for the declared channels.
    """

    claim: str
    relations: Tuple[str, ...]
    unassessed: Tuple[str, ...]
    conditioning: str
    population: str
    conformance: str
    result_identity: str
    status: str

    def render(self) -> str:
        """The human-readable block."""
        return "\n".join((
            f"Claim: {self.claim}",
            f"Relations: {' + '.join(self.relations) if self.relations else 'none'}",
            f"Unassessed: {', '.join(self.unassessed) if self.unassessed else 'none'}",
            f"Conditioning: {self.conditioning}",
            f"Population: {self.population}",
            f"Conformance: {self.conformance}",
            f"Result identity: {self.result_identity}",
            f"Status: {self.status}",
        ))

    def as_dict(self) -> Dict[str, Any]:
        """The machine-readable fields the line was generated from.

        Sequence fields are emitted as lists rather than tuples so that the
        artifact is byte-stable across a JSON round trip: a validator that read
        a written artifact back and compared it against a freshly built one
        would otherwise fail on the container type alone.
        """
        payload = asdict(self)
        payload["relations"] = list(self.relations)
        payload["unassessed"] = list(self.unassessed)
        payload["schema"] = "leakage-line/1"
        payload["rendered"] = self.render()
        return payload


#: How a missing conditioning channel weakens the claim. Sealed in advance, so
#: that discovering an unknown channel produces a predetermined weaker claim
#: rather than an argument about what the result still shows. A channel with no
#: entry here makes the result UNINTERPRETABLE pending adjudication -- which is
#: the honest answer, and deliberately more expensive than being listed.
CLAIM_REDUCTION: Mapping[str, str] = {
    "opaque_upstream_pretraining": "closed-book unseen-broader-group",
    "unassessed_structural_equivalence": "closed-book unseen-broader-group",
    "antibody_axis_not_held_out": "record-random",
    "adaptive_reuse_unbudgeted": "descriptive-only",
}


def reduce_claim(
    strong_claim: str, unsupported_channels: Sequence[str]
) -> Tuple[str, Tuple[str, ...]]:
    """Emit the strongest claim the evidence still supports.

    A binary publishable flag throws away useful partial evidence, and an ad-hoc
    rescue after seeing the numbers is worse than either. So the mapping from
    missing channels to weaker standard claims is sealed in `CLAIM_REDUCTION` in
    advance, and this applies it.

    Args:
        strong_claim: The claim that would hold with every channel supported.
        unsupported_channels: Channels that are unknown or violated.

    Returns:
        ``(supported_claim, unmapped_channels)``. When ``unmapped_channels`` is
        non-empty the result is uninterpretable, not merely weaker.

    Raises:
        ConformanceError: When ``strong_claim`` is outside the vocabulary.
    """
    if strong_claim not in CLAIM_CLASSES:
        raise ConformanceError(f"{strong_claim!r} is not a controlled claim class")
    supported = strong_claim
    unmapped = []
    for channel in unsupported_channels:
        weaker = CLAIM_REDUCTION.get(channel)
        if weaker is None:
            unmapped.append(channel)
            continue
        if CLAIM_CLASSES.index(weaker) > CLAIM_CLASSES.index(supported):
            supported = weaker
    return supported, tuple(sorted(unmapped))


@dataclass(frozen=True)
class ConformanceAttestation:
    """A validator's verdict, with the reasons it reached it.

    Attributes:
        level: The level the evidence supports -- 0 when it supports none.
        claimed_level: The level that was claimed.
        supported_claim: The reduced claim.
        uninterpretable_channels: Channels with no sealed reduction.
        failures: Why the claimed level was not reached.
        notes: Observations that do not block the level.
    """

    level: int
    claimed_level: int
    supported_claim: str
    uninterpretable_channels: Tuple[str, ...]
    failures: Tuple[str, ...]
    notes: Tuple[str, ...]

    @property
    def conformant(self) -> bool:
        """Whether the evidence supports the level that was claimed."""
        return self.level >= self.claimed_level and not self.uninterpretable_channels

    def as_dict(self) -> Dict[str, Any]:
        """Render for a JSON artifact."""
        payload = asdict(self)
        payload["schema"] = "conformance-attestation/1"
        payload["conformant"] = self.conformant
        return payload


#: Level 1 requires each of these, and the validator checks for each by name so
#: that a failure says which requirement was missed rather than that something
#: was wrong.
LEVEL_1_REQUIREMENTS: Tuple[str, ...] = (
    "explicit controlled claim",
    "declared relation classes",
    "group-disjoint split",
    "closed test snapshot",
    "label-blind grouping",
    "achieved ratio and largest-component reporting",
    "standard leakage line",
)


def validate_level_1(
    claim_manifest: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    guard_report: Mapping[str, Any],
    leakage_line: Mapping[str, Any],
    *,
    claimed_level: int = 1,
    unsupported_channels: Sequence[str] = (),
) -> ConformanceAttestation:
    """Check that the four artifacts exist, agree, and support Level 1.

    This verifies that the required evidence is present and internally
    consistent. It cannot verify that a semantic label is true, that a threshold
    is the right one, or that an upstream vendor retained nothing -- and saying
    so plainly is part of what the attestation is for.

    Args:
        claim_manifest: From `build_claim_manifest`.
        split_manifest: From `build_split_manifest`.
        guard_report: From `build_guard_report`.
        leakage_line: From `LeakageLine.as_dict`.
        claimed_level: The level being claimed.
        unsupported_channels: Conditioning channels known to be unsupported.

    Returns:
        A `ConformanceAttestation`.
    """
    failures = []
    notes = []

    for document, schema in (
        (claim_manifest, "claim-manifest/1"),
        (split_manifest, "split-manifest/1"),
        (guard_report, "guard-report/1"),
        (leakage_line, "leakage-line/1"),
    ):
        if document.get("schema") != schema:
            failures.append(f"expected schema {schema}, found {document.get('schema')!r}")

    for label, document in (("claim", claim_manifest), ("split", split_manifest)):
        try:
            verify_manifest(dict(document))
        except ManifestDamaged as error:
            failures.append(f"{label} manifest seal: {error}")

    if split_manifest.get("claim_manifest_sha256") != claim_manifest.get(
        "manifest_sha256"
    ):
        failures.append(
            "the split manifest is not bound to this claim manifest; a split "
            "that does not name its claim can be quoted under any claim"
        )
    if guard_report.get("claim_manifest_sha256") != claim_manifest.get(
        "manifest_sha256"
    ):
        failures.append("the guard report is not bound to this claim manifest")

    claim = claim_manifest.get("claim", {})
    if claim.get("claim_class") not in CLAIM_CLASSES:
        failures.append("claim_class is outside the controlled vocabulary")
    if not claim.get("unit_of_generalisation"):
        failures.append("no unit of generalisation declared")

    operationalisation = claim_manifest.get("operationalisation", {})
    if not operationalisation.get("relation_classes"):
        failures.append("no relation classes declared as assessed")
    if "unassessed_relation_classes" not in operationalisation:
        failures.append(
            "unassessed relation classes must be declared; listing only what was "
            "checked reads as if everything was"
        )

    for key in ("achieved_val_fraction", "largest_group_share",
                "group_size_distribution", "assignments"):
        if key not in split_manifest:
            failures.append(f"split manifest is missing {key}")

    resolution = guard_report.get("resolution", {})
    for key in ("component_min_pairwise_identity", "component_min_pairwise_coverage",
                "target_test_ineligible_constructs"):
        if key not in resolution:
            failures.append(f"guard report is missing {key}")
    # Presence is not enough. A validator that checks only that a key EXISTS
    # accepts a fully percolated partition reporting a minimum pairwise identity
    # of 0.0, which is the exact number the committed rule's worst component
    # would produce. Where the bounded criterion decided, its guarantee is a
    # property of the algorithm and must hold.
    thresholds = claim_manifest.get("operationalisation", {}).get("thresholds", {})
    family = thresholds.get("family", {}) if isinstance(thresholds, Mapping) else {}
    floor = family.get("identity") if isinstance(family, Mapping) else None
    criterion = resolution.get("criterion_min_pairwise_identity")
    if floor is not None and criterion is not None and criterion < floor:
        failures.append(
            f"a component the bounded criterion built has minimum pairwise "
            f"identity {criterion:.4f}, below the family threshold {floor:.4f}; "
            f"complete linkage cannot produce that, so the criterion is not the "
            f"one that ran"
        )

    audit = guard_report.get("audit", {})
    if not audit.get("pairs"):
        failures.append(
            "the audit report scored no pairs; zero errors over zero pairs is "
            "not evidence"
        )
    elif not audit.get("positive_pairs") or not audit.get("negative_pairs"):
        failures.append(
            "the audit report has no positive or no negative pairs, so one of "
            "its two error counts is free"
        )

    # Concentration, gated rather than noted, and gated in ROWS. The failure
    # this exists for is a previous redesign that percolated to a 77% component
    # on the relation the split keys on; a share measured in records could not
    # have seen it, because the offending component was a handful of records
    # carrying most of the corpus.
    #
    # Nothing in the construction PREVENTS percolation absolutely -- containers
    # are refused and composites are made test-ineligible, but a chain of
    # individually-legal constraints can still close into one large group, and
    # the design this implements says so: if a huge component remains after
    # defensible preprocessing, the intended held-out claim may genuinely be
    # unsupported. So it is detected here and it FAILS the level.
    bound = claim_manifest.get("population", {}).get("max_split_group_row_share")
    row_share = split_manifest.get("largest_group_row_share")
    if row_share is None:
        row_share = guard_report.get("resolution", {}).get(
            "largest_split_group_row_share"
        )
    if row_share is None:
        failures.append(
            "no largest-split-group ROW share reported; concentration measured "
            "in records cannot see the failure this gate exists for"
        )
    elif bound is None:
        failures.append(
            "the claim manifest predeclares no max_split_group_row_share, so "
            "there is no bound this result could have failed"
        )
    elif row_share > bound:
        failures.append(
            f"largest split group holds {row_share:.2%} of rows against a "
            f"predeclared bound of {bound:.2%}: the quarantine relation has "
            f"percolated and the held-out claim is not supported"
        )
    else:
        notes.append(
            f"largest split group holds {row_share:.2%} of rows, within the "
            f"predeclared {bound:.2%}"
        )

    largest = split_manifest.get("largest_group_share")
    if isinstance(largest, (int, float)) and largest > 0.5:
        notes.append(
            f"largest split group also holds {largest:.1%} of distinct records; "
            f"the achieved ratio is real but the effective number of "
            f"independent groups is not"
        )

    # The blocker's recall guarantee is derived, and blocking.py is explicit that
    # the derivation is not the evidence -- the measurement is. A Level-1
    # attestation that never looks at the measurement is quoting the derivation.
    recall = guard_report.get("blocking_recall")
    if not recall:
        failures.append(
            "no blocking-recall measurement in the guard report; a pair the "
            "blocker never proposed cannot be recovered downstream, so its "
            "recall is a correctness claim and has to be measured"
        )
    else:
        if not recall.get("qualifying_pairs"):
            failures.append(
                "the blocking-recall audit found no qualifying pairs, so a "
                "recall of 1.0 over them is free"
            )
        if recall.get("missed"):
            failures.append(
                f"the blocker lost {len(recall['missed'])} qualifying pairs; "
                f"that is silent leakage, not a tuning problem"
            )

    supported, unmapped = reduce_claim(
        claim.get("claim_class", "descriptive-only"), unsupported_channels
    )
    if leakage_line.get("claim") != supported:
        failures.append(
            f"the leakage line claims {leakage_line.get('claim')!r} but the "
            f"sealed reducer supports {supported!r}"
        )

    level = 0 if failures else 1
    return ConformanceAttestation(
        level=level,
        claimed_level=claimed_level,
        supported_claim=supported,
        uninterpretable_channels=unmapped,
        failures=tuple(failures),
        notes=tuple(notes),
    )


def write_artifact(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write one artifact as canonical, human-readable JSON.

    Args:
        path: Destination.
        payload: The artifact.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
