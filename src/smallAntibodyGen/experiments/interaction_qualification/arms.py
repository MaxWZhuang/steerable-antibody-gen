"""The three frozen-feature arms, built from visible views only.

Per record, three UNMASKED views are built from the record's sequences alone (never its ids,
groups, source, assay or label): the joint (H, L, Ag) view, its antibody-only (H, L) view and its
antigen-only (Ag) view. Each distinct view is run through the backend once, under ``no_grad``, and
only two pooled detached CPU float64 vectors survive (mean over antibody H/L residue rows, mean over
antigen rows; no CLS/EOS), never the full ``[N, D]`` output.

* ``joint``         = [joint antibody mean, joint antigen mean]            (width 2D)
* ``antibody_only`` = antibody-only antibody mean                          (width D)
* ``independent``   = [antibody-only antibody mean, antigen-only mean]     (width 2D)

The arms differ in width, so the shared fixed ridge alpha does not make the antibody-only arm
capacity-matched; this is reported, not hidden. Feature extraction is deliberately single-view: a
held-out row can never change a train feature through batch padding.

Features are returned keyed by ``record_id`` (an identity join, not a feature) and arms must cover
identical ids. The only exclusion is an arm-independent joint-serialized-length cap, applied before any
forward and counted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from smallAntibodyGen.experiments.interaction_qualification.config import expect_int
from smallAntibodyGen.experiments.interaction_qualification.records import Record
from smallAntibodyGen.models.antibody_antigen.views import VisibleView, mask_complete

if TYPE_CHECKING:  # torch is imported at run time only inside the functions that run a backend
    import torch

    from smallAntibodyGen.models.antibody_antigen.denoising import ResidueBackend

ARM_NAMES: Tuple[str, ...] = ("joint", "antibody_only", "independent")
ARM_WIDTH_FACTOR: Mapping[str, int] = MappingProxyType({"joint": 2, "antibody_only": 1, "independent": 2})
ARM_DESCRIPTIONS: Mapping[str, str] = MappingProxyType(
    {
        "joint": "[joint-MINT antibody mean, joint-MINT antigen mean], both pooled from one joint forward",
        "antibody_only": "antibody-only MINT mean over the H/L residues",
        "independent": "[antibody-only MINT mean, antigen-only MINT mean], encoded in separate forwards",
    }
)
ARM_NOTE = (
    "arms differ in width (D vs 2D); the same fixed ridge alpha is used for every arm, so the antibody-only "
    "arm is not capacity-matched"
)


class ArmMismatchError(ValueError):
    """Arms (or an id list) do not cover exactly the same record ids."""


# --- views and length admission -------------------------------------------------------------


@dataclass(frozen=True)
class FeatureViews:
    joint: VisibleView
    antibody: VisibleView
    antigen: VisibleView


def feature_views(record: Record) -> FeatureViews:
    """The three unmasked views of a record, from its sequences only."""
    sequences = {"H": record.heavy, "Ag": record.antigen}
    if record.light is not None:
        sequences["L"] = record.light
    joint, _ = mask_complete(sequences)
    return FeatureViews(joint=joint, antibody=joint.without_ag(), antigen=joint.ag_only())


def joint_serialized_length(record: Record) -> int:
    """Residues of every present chain plus two specials (CLS, EOS) per chain, in the joint view.

    The joint view is the longest of the three, so one cap on it admits the same records for every arm.
    """
    chains = 2 + (0 if record.light is None else 1)
    residues = len(record.heavy) + len(record.antigen) + (0 if record.light is None else len(record.light))
    return residues + 2 * chains


@dataclass(frozen=True)
class LengthAdmission:
    max_tokens: int
    eligible: Tuple[Record, ...]  # input order preserved
    excluded: Tuple[Tuple[str, int], ...]  # (record_id, joint serialized length), input order


def admit_by_joint_length(records: Sequence[Record], max_tokens: int) -> LengthAdmission:
    cap = expect_int(max_tokens, "max_tokens", minimum=1)
    eligible: List[Record] = []
    excluded: List[Tuple[str, int]] = []
    for record in records:
        length = joint_serialized_length(record)
        if length <= cap:
            eligible.append(record)
        else:
            excluded.append((record.record_id, length))
    return LengthAdmission(max_tokens=cap, eligible=tuple(eligible), excluded=tuple(excluded))


def planned_feature_forwards(records: Iterable[Record], memo: Optional["FullViewMemo"] = None) -> int:
    """Backend forwards feature extraction would run: distinct unmasked views not already memoized."""
    digests = set()
    for record in records:
        views = feature_views(record)
        digests.update(view.content_digest() for view in (views.joint, views.antibody, views.antigen))
    if memo is not None:
        digests = {digest for digest in digests if digest not in memo}
    return len(digests)


# --- full-view memo -------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class PooledVectors:
    antibody: Optional["torch.Tensor"]  # [D] float64 CPU, mean over H/L residue rows; None if no H/L present
    antigen: Optional["torch.Tensor"]  # [D] float64 CPU, mean over antigen rows; None if no Ag present


def _mean_rows(features: "torch.Tensor", rows: List[int]) -> Optional["torch.Tensor"]:
    import torch

    if not rows:
        return None
    return features[torch.tensor(rows, dtype=torch.long)].mean(dim=0).clone()


class FullViewMemo:
    """Pooled features of exact UNMASKED views, one backend forward per distinct visible content.

    It refuses any view with a masked site (masked diagnostics must never share it) and keeps only the
    two pooled detached CPU vectors, not the residue-level output.
    """

    def __init__(self, backend: "ResidueBackend") -> None:
        self._backend = backend
        self._entries: Dict[str, PooledVectors] = {}
        self.forward_calls = 0

    @property
    def entries(self) -> Mapping[str, PooledVectors]:
        return MappingProxyType(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, digest: object) -> bool:
        return digest in self._entries

    def pooled(self, view: VisibleView) -> PooledVectors:
        if not isinstance(view, VisibleView):
            raise TypeError(f"expected a VisibleView, got {type(view).__name__}")
        if view.hidden_sites():
            raise ValueError("the full-view memo refuses masked views; masked diagnostics must not use it")
        key = view.content_digest()
        cached = self._entries.get(key)
        if cached is None:
            cached = self._entries[key] = self._forward(view)
        return cached

    def _forward(self, view: VisibleView) -> PooledVectors:
        import torch

        from smallAntibodyGen.models.antibody_antigen.denoising import checked_forward

        with torch.no_grad():
            output = checked_forward(self._backend, view)
            features = output.features.detach().cpu().to(torch.float64)
        self.forward_calls += 1
        antibody_rows: List[int] = []
        antigen_rows: List[int] = []
        for row, (role, _) in enumerate(view.coords()):
            (antigen_rows if role == "Ag" else antibody_rows).append(row)
        return PooledVectors(antibody=_mean_rows(features, antibody_rows), antigen=_mean_rows(features, antigen_rows))


# --- per-arm features keyed by record id -----------------------------------------------------


@dataclass(frozen=True, eq=False)
class ArmFeatures:
    """One arm's feature vectors keyed by record id (read-only float64, equal width, finite)."""

    arm: str
    vectors: Mapping[str, np.ndarray]
    dimension: int = field(init=False)

    def __post_init__(self) -> None:
        if self.arm not in ARM_NAMES:
            raise ValueError(f"unknown arm {self.arm!r}; arms are {ARM_NAMES}")
        frozen: Dict[str, np.ndarray] = {}
        width: Optional[int] = None
        for record_id, vector in self.vectors.items():
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(f"feature keys must be non-empty record ids, got {record_id!r}")
            array = np.array(vector, dtype=np.float64)
            if array.ndim != 1 or array.size == 0:
                raise ValueError(f"features for {record_id!r} must be a non-empty 1-D vector")
            if not bool(np.all(np.isfinite(array))):
                raise ValueError(f"features for {record_id!r} are not finite")
            if width is not None and array.shape[0] != width:
                raise ValueError(f"features for {record_id!r} have width {array.shape[0]}, expected {width}")
            width = array.shape[0]
            array.flags.writeable = False
            frozen[record_id] = array
        object.__setattr__(self, "vectors", MappingProxyType(frozen))
        object.__setattr__(self, "dimension", 0 if width is None else width)


def common_ids(arms: Mapping[str, ArmFeatures]) -> Tuple[str, ...]:
    """The sorted record ids all arms cover; any difference between arms is an error, never an intersection."""
    if not arms:
        raise ValueError("no arms given")
    reference: Optional[frozenset] = None
    for name, features in arms.items():
        if features.arm != name:
            raise ArmMismatchError(f"arm key {name!r} holds features of arm {features.arm!r}")
        ids = frozenset(features.vectors)
        if reference is None:
            reference = ids
        elif ids != reference:
            only_here = sorted(ids - reference)[:3]
            only_there = sorted(reference - ids)[:3]
            raise ArmMismatchError(
                f"arm {name!r} does not cover the same record ids as the others "
                f"(extra {only_here}, missing {only_there})"
            )
    assert reference is not None
    return tuple(sorted(reference))


def assemble_matrix(features: ArmFeatures, ids: Sequence[str]) -> np.ndarray:
    """Rows in exactly the order of ``ids``; unknown or repeated ids are refused (no positional joins)."""
    if isinstance(ids, str):
        raise TypeError("pass a sequence of record ids, not a string")
    ordered = list(ids)
    if len(set(ordered)) != len(ordered):
        raise ArmMismatchError("record ids to assemble must be unique")
    missing = [record_id for record_id in ordered if record_id not in features.vectors]
    if missing:
        raise ArmMismatchError(f"arm {features.arm!r} has no features for record ids {missing[:3]}")
    matrix = np.empty((len(ordered), features.dimension), dtype=np.float64)
    for row, record_id in enumerate(ordered):
        matrix[row] = features.vectors[record_id]
    return matrix


def _numpy(vector: Optional["torch.Tensor"]) -> np.ndarray:
    assert vector is not None
    return vector.numpy().copy()


def extract_arm_features(records: Sequence[Record], memo: FullViewMemo) -> Dict[str, ArmFeatures]:
    """All three arms for ``records`` (already length-admitted), one view at a time, train partition first."""
    ids = [record.record_id for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError("record ids must be unique to key features by identity")
    collected: Dict[str, Dict[str, np.ndarray]] = {arm: {} for arm in ARM_NAMES}
    for record in sorted(records, key=lambda r: r.partition != "train"):  # stable: train first
        views = feature_views(record)
        joint = memo.pooled(views.joint)
        antibody = memo.pooled(views.antibody)
        antigen = memo.pooled(views.antigen)
        collected["joint"][record.record_id] = np.concatenate([_numpy(joint.antibody), _numpy(joint.antigen)])
        collected["antibody_only"][record.record_id] = _numpy(antibody.antibody)
        collected["independent"][record.record_id] = np.concatenate([_numpy(antibody.antibody), _numpy(antigen.antigen)])
    arms = {arm: ArmFeatures(arm, collected[arm]) for arm in ARM_NAMES}
    common_ids(arms)
    return arms


def arm_report(arms: Mapping[str, ArmFeatures]) -> Dict[str, object]:
    return {
        "arms": {
            name: {
                "description": ARM_DESCRIPTIONS[name],
                "width": arms[name].dimension,
                "n_records": len(arms[name].vectors),
            }
            for name in ARM_NAMES
            if name in arms
        },
        "note": ARM_NOTE,
    }
