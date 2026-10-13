"""Restore the frozen HER2 split from its record. Nothing here builds a split.

The production flight froze four things before any outcome was opened: which candidate windows are
development, which are reserved, which were excluded from training for lying within the purge radius
of a held-out core, and (by exclusion) which train. That record -- ``hard_split_freeze.json`` -- is
pinned twice by the base config (the digest of the record, and the digest of the sequence-only
candidate set). This module reads it back, recomputes everything that can be recomputed from the
sequence file alone, and refuses the pilot unless the two agree:

1. the candidate windows are re-derived from the sequence-only AlphaBind read (the outcome columns
   are never parsed), their digest must equal the pinned sequence digest and the record's;
2. the record's own digest must equal the pinned split digest, its settings must equal the base
   config's ``split`` block, and its development, reserved and excluded lists must be duplicate-free,
   pairwise disjoint and made of candidate windows;
3. the train set is ``candidates - development - reserved - union(excluded)``, and the counts that
   follow -- from the record and from the pilot config's ``expected_counts`` -- must reproduce;
4. an :class:`~..her2_routes_split.ExclusionClosure` is built on the held-out windows and their full
   scaffold identities at the frozen purge radius, and EVERY train window must clear it.

``build_hard_split`` and ``freeze_split`` are not imported. The split is a decision that was made
once; this restores it and checks it, so a changed library file, a changed seed or a drifted
algorithm cannot quietly produce a different panel under the same name.
"""
from __future__ import annotations

from dataclasses import dataclass

from .. import her2_routes_sources as sources
from .. import her2_routes_split as split_lib
from ..her2_routes_serial import her2_construct, her2_core_edit_map
from ..her2_runtime import require

#: The keys of the freeze record this module reads. Others (``acquisition``, ``order``...) are the
#: production flight's own bookkeeping and are covered by the record digest.
RECORD_KEYS = ("sequence_sha256", "settings", "counts", "development", "reserved", "excluded",
               "selection")


@dataclass(frozen=True)
class FrozenMembership:
    """What the split record decided, in the shape the services read.

    ``assignment`` maps every candidate window to ``train``, ``development``, ``reserved`` or
    ``excluded``. ``development`` keeps the record's order (the panel's order). This is a plain
    value: it has no ``selection`` batches because nothing here selected anything.
    """

    assignment: dict
    development: tuple
    reserved: tuple
    excluded: dict
    settings: dict
    counts: dict
    sequence_sha256: str
    split_sha256: str

    @property
    def train(self):
        return frozenset(core for core, kind in self.assignment.items()
                         if kind == split_lib.TRAIN)

    def document(self):
        return {"split_sha256": self.split_sha256, "sequence_sha256": self.sequence_sha256,
                "counts": dict(self.counts),
                "restored": ("read back from the frozen record and re-checked against the "
                             "sequence-only candidate set; no split was built")}


def candidate_windows(sequence_path, *, scaffold):
    """The compatible candidate cores of the AlphaBind HER2 file, from sequences alone.

    The outcome columns are never parsed (``load_alphaseq_sequences`` restricts the parser). The
    parent linker and light chain are derived from the file by majority, as in the original freeze,
    and a row is a candidate only if it is compatible with the fixed scaffold.
    """
    read = sources.load_alphaseq_sequences(sequence_path)
    sites = her2_core_edit_map()
    parent = sources.derive_alphaseq_parent(read["sequences"])
    windows, rejected = [], {}
    for sequence in read["sequences"]:
        ok, reason = sources.alphaseq_compatibility(sequence, parent=parent, scaffold=scaffold,
                                                    sites=sites)
        if not ok:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        heavy = sources.split_alphaseq_construct(sequence)[0]
        windows.append(heavy[sites[0][1]:sites[-1][1] + 1])
    require(windows, f"No compatible AlphaBind window survived: {rejected}")
    return {"windows": tuple(sorted(set(windows))), "rows": len(read["sequences"]),
            "rejected": dict(sorted(rejected.items())), "parent": parent,
            "file_sha256": read["sha256"], "outcome_columns_read": False}


def _cores(value, name, *, candidates, length):
    require(isinstance(value, list), f"{name} must be a list of windows, got "
                                     f"{type(value).__name__}")
    for core in value:
        require(isinstance(core, str) and len(core) == length,
                f"{name} holds {core!r}, not a {length}-residue window")
    require(len(set(value)) == len(value), f"{name} repeats a window")
    stray = sorted(set(value) - candidates)
    require(not stray, f"{name} holds {len(stray)} window(s) that are not candidates, e.g. "
                       f"{stray[:3]}")
    return tuple(value)


def restore_membership(record, *, record_sha256, candidates, expected_split_sha256,
                       expected_sequence_sha256, split_settings, expected_counts):
    """Validate ``record`` against the pins and the candidates; return the :class:`FrozenMembership`.

    ``candidates`` is the sorted tuple of compatible windows from :func:`candidate_windows`.
    Everything is checked before anything is returned, and nothing about the outcomes is touched.
    """
    require(isinstance(record, dict), "the split record must be a JSON object")
    missing = sorted(set(RECORD_KEYS) - set(record))
    require(not missing, f"the split record is missing {missing}")
    require(record_sha256 == expected_split_sha256,
            f"the split record digests to {record_sha256}; the base config pins "
            f"{expected_split_sha256}. The split is restored, never rebuilt, so a different record "
            "is a different panel")
    digest = split_lib.sequence_digest(candidates)
    require(digest == expected_sequence_sha256,
            f"the candidate windows digest to {digest}; the base config pins "
            f"{expected_sequence_sha256}. The sequence file or the compatibility rule changed")
    require(record["sequence_sha256"] == expected_sequence_sha256,
            f"the record's sequence digest is {record['sequence_sha256']}, not the pinned "
            f"{expected_sequence_sha256}")
    require(record["settings"] == dict(split_settings),
            "the record's split settings differ from the base config's split block, so the "
            "algorithm it was frozen under is not the one this pilot names")
    candidate_set = frozenset(candidates)
    length = len(her2_core_edit_map())
    development = _cores(record["development"], "development", candidates=candidate_set,
                         length=length)
    reserved = _cores(record["reserved"], "reserved", candidates=candidate_set, length=length)
    excluded_block = record["excluded"]
    require(isinstance(excluded_block, dict), "the record's excluded block must be an object")
    excluded = {str(reason): _cores(cores, f"excluded[{reason!r}]", candidates=candidate_set,
                                    length=length)
                for reason, cores in sorted(excluded_block.items())}
    held_out, purged = set(development) | set(reserved), set()
    require(not (set(development) & set(reserved)),
            "a window is both development and reserved")
    for reason, cores in excluded.items():
        require(not (set(cores) & held_out),
                f"excluded[{reason!r}] contains a development or reserved window")
        require(not (set(cores) & purged), f"excluded[{reason!r}] repeats a window of another reason")
        purged |= set(cores)
    train = candidate_set - held_out - purged
    counts = {"candidates": len(candidate_set), "development": len(development),
              "reserved": len(reserved), "excluded": len(purged),
              "retained_for_training": len(train)}
    require(counts == dict(expected_counts),
            f"the restored membership counts {counts}, the pilot config expects "
            f"{dict(expected_counts)}")
    recorded = dict(record["counts"])
    for key in ("candidates", "development", "reserved", "retained_for_training"):
        require(recorded.get(key) == counts[key],
                f"the record's counts[{key!r}] is {recorded.get(key)}, the sets give {counts[key]}")
    require(recorded.get("purged_within_radius") == counts["excluded"],
            f"the record's purged_within_radius is {recorded.get('purged_within_radius')}, the "
            f"excluded sets give {counts['excluded']}")
    assignment = {core: split_lib.TRAIN for core in sorted(train)}
    for reason, cores in excluded.items():
        assignment.update({core: split_lib.EXCLUDED for core in cores})
    assignment.update({core: split_lib.DEVELOPMENT for core in development})
    assignment.update({core: split_lib.RESERVED for core in reserved})
    require(len(assignment) == len(candidate_set),
            "the restored assignment does not cover the candidates exactly once")
    return FrozenMembership(assignment=assignment, development=development, reserved=reserved,
                            excluded=excluded, settings=dict(split_settings), counts=counts,
                            sequence_sha256=digest, split_sha256=record_sha256)


def build_closure(membership, *, scaffold):
    """The exclusion closure of the held-out windows at the frozen radius; train must clear it.

    Built exactly as the original freeze built it: the reserved and development windows, their full
    scaffold identities, the purge radius, and the note that the full AlphaBind file is not
    enrolled. Every train window is then tested; one that does not clear the closure means the
    record and the radius disagree, and the pilot does not start.
    """
    radius = int(membership.settings["purge_radius"])
    closure = split_lib.ExclusionClosure.build(
        reserved_windows=membership.reserved, development_windows=membership.development,
        reserved_identities=tuple(her2_construct(scaffold, core).identity()
                                  for core in (*membership.reserved, *membership.development)),
        radius=radius,
        extra={"alphabind_her2_full": sources.SOURCES["alphabind_her2_full"].reason_not_enrolled})
    violations = []
    for core in sorted(membership.train):
        reason = closure.violates(window=core, identity=her2_construct(scaffold, core).identity())
        if reason is not None:
            violations.append((core, reason))
    require(not violations,
            f"{len(violations)} train window(s) do not clear the closure at radius {radius}, e.g. "
            f"{violations[:2]}. The record and the purge radius disagree")
    return closure
