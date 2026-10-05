"""Synthetic world for the Build D selftest and bounded pilot (private companion of
``functional_shared_transfer``; it exists only to keep that module about the learner).

Everything here is DECLARED BEFORE ANY TRAINING and none of it is a biological claim:

* sequences are fixed-length H4/L4 over the alphabet A, C, D, E; the honest group of a sequence is its
  boundary pair ``(H0, L3)``, and ``t = (L3 - H0) mod 4`` partitions the sixteen groups into four strata
  of four groups. One stratum each is given to the source cohort (``t=0``), source development
  (``t=1``), target training (``t=2``) and target evaluation (``t=3``); every boundary symbol therefore
  occurs in every split, and no group is shared between cohorts;
* the source function is ``1[H1=H2] + 0.5[L0=L1] + 0.2 value(H3)`` and the related but different
  target function is ``1[H2=L2] + 0.5[H1=L1] + 0.2 value(L0)``, with ``value`` mapping A/C/D/E to
  0, 1/3, 2/3, 1. Both depend only on interior coordinates (H1, H2, H3, L0, L1, L2), never on a
  group boundary value. They are computational functions of the sequence, not affinities;
* the native token cache is produced by a real tiny RANDOMLY initialised CPU DPLM
  (``functional_native_export.tiny_model``) through the real exporter, so the student learns from
  cached native activations and never receives a one-hot residue or an interaction feature.

Source-only selection: which of a group's auxiliary sequences are the measured ``source_train``
sequences is drawn with a seeded generator from the sequences alone (no label is read).

Excluded auxiliary sequences are recorded, never hidden. The source cohort has ``t=0``, so its boundary
symbols are equal and an interior of that same symbol would be a CONSTANT sequence: every EQ over such a
sequence is true, so no complete auxiliary instance exists for it and ``functional_composition``
refuses it by name. Each source group has exactly one constant interior, so at most ``4095`` of its
``4096`` interiors are eligible and ``source_aux`` above 4095 is refused up front. A constant code that
the seeded draw selects is replaced by a code drawn WITHOUT replacement, by the same generator, from the
group's unused non-constant codes (a direct draw over the available codes, never a retry loop) and listed
in ``World.excluded`` with its replacement. A world whose draw selects no constant code (the usual case,
including the pilot at seed 0) is exactly what it would have been without the rule; the declared counts
are unchanged either way.

The exclusion record is PUBLISHED, not kept only in memory: :func:`write_inputs` writes
``synthetic_provenance.json`` (the actual spec, the source-only selection rule, every exclusion and its
replacement, and digests of the rosters and the cohort document) next to the other inputs and returns its
path. It is a software selection record, not a measured label and not a biological result.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import functional_baseline as fb
from . import functional_composition as fc
from . import functional_native_export as fne
from .her2_runtime import require

ALPHABET = "ACDE"
VALUES = {"A": 0.0, "C": 1.0 / 3.0, "D": 2.0 / 3.0, "E": 1.0}
INTERIOR = 6
#: interior codes per group, and the source cohort's usable share: one interior per group is constant
INTERIOR_CODES = len(ALPHABET) ** INTERIOR
SOURCE_AUX_CAPACITY = INTERIOR_CODES - 1
COHORT_STRATUM = {"source": 0, "source_dev": 1, "target_train": 2, "target_eval": 3}
PREFIX = {"source": "sa", "source_dev": "sd", "target_train": "tt", "target_eval": "te"}
PROVENANCE_SCHEMA = "functional-synthetic-provenance/1"
PROVENANCE_NAME = "synthetic_provenance.json"

SOURCE_FUNCTION_TEXT = "1[H1=H2] + 0.5[L0=L1] + 0.2*value(H3)"
TARGET_FUNCTION_TEXT = "1[H2=L2] + 0.5[H1=L1] + 0.2*value(L0)"
DECLARATION = (
    f"source = {SOURCE_FUNCTION_TEXT}; target = {TARGET_FUNCTION_TEXT}; value maps A/C/D/E to "
    "0, 1/3, 2/3, 1. Fixed before any training. Computational functions of interior coordinates "
    "(H1, H2, H3, L0, L1, L2), not biological labels and not affinities.")


def source_function(heavy, light):
    """The declared synthetic SOURCE function (units: arbitrary)."""
    return float(1.0 * (heavy[1] == heavy[2]) + 0.5 * (light[0] == light[1])
                 + 0.2 * VALUES[heavy[3]])


def target_function(heavy, light):
    """The declared synthetic TARGET function: related to, and different from, the source."""
    return float(1.0 * (heavy[2] == light[2]) + 0.5 * (heavy[1] == light[1])
                 + 0.2 * VALUES[light[0]])


@dataclass(frozen=True)
class SyntheticSpec:
    """Sequences PER GROUP (four groups per cohort). ``source_aux`` includes ``source_train``."""

    source_train: int
    source_aux: int
    source_dev: int
    target_train: int
    target_eval: int
    seed: int = 0

    def __post_init__(self):
        for name in ("source_train", "source_aux", "source_dev", "target_train", "target_eval"):
            require(isinstance(getattr(self, name), int) and getattr(self, name) >= 1,
                    f"{name} must be a positive integer per group")
        require(self.source_train <= self.source_aux, "source_train is a subset of source_aux")
        require(max(self.source_aux, self.source_dev, self.target_train, self.target_eval)
                <= INTERIOR_CODES,
                "more sequences per group than distinct interiors")
        require(self.source_aux <= SOURCE_AUX_CAPACITY,
                f"source_aux is {self.source_aux} per group but at most {SOURCE_AUX_CAPACITY} can be "
                f"used: the source cohort's boundary symbols are equal, so one of its {INTERIOR_CODES} "
                "interiors per group is a constant sequence that has no complete auxiliary instance "
                "and is never selected")

    def totals(self):
        return {"source_train": 4 * self.source_train, "source_aux": 4 * self.source_aux,
                "source_dev": 4 * self.source_dev, "target_train": 4 * self.target_train,
                "target_eval": 4 * self.target_eval}


#: 128 measured source labels, 512 auxiliary sequences (including them), 64 source-dev, 128 + 128 target
PILOT_SPEC = SyntheticSpec(source_train=32, source_aux=128, source_dev=16, target_train=32,
                           target_eval=32)
#: small enough for a CPU selftest, large enough to exercise every code path
SELFTEST_SPEC = SyntheticSpec(source_train=4, source_aux=8, source_dev=2, target_train=4,
                              target_eval=4)


@dataclass(frozen=True, eq=False)
class World:
    spec: SyntheticSpec
    rows: tuple          # dicts: candidate_id, heavy, light, group_id, split, cohort
    roles: dict          # role -> sorted candidate ids
    outcomes: dict       # "source" / "source_dev" / "target_train" / "target_eval" -> {id: value}
    excluded: tuple = ()  # constant auxiliary sequences the seeded draw selected and replaced

    def sequences(self, role):
        index = {row["candidate_id"]: row for row in self.rows}
        return [(cid, index[cid]["heavy"], index[cid]["light"]) for cid in self.roles[role]]

    def export_plan_document(self, representation_id="synthetic_tokens"):
        return {
            "schema": fne.PLAN_SCHEMA, "representation_id": representation_id,
            "conditioning_mode": fne.CONDITIONING_NONE,
            "activation": {"kind": fne.ACTIVATION_FINAL, "name": None},
            "feature_supervision": {
                "status": fb.HISTORY_LABEL_FREE,
                "description": "tiny random-initialised CPU DPLM; no labels of any kind",
                "sequence_only_exposure": {"status": "none"}},
            "extractor": {"kind": "synthetic", "checkpoint_sha256": None, "source_sha256": None,
                          "description": "tiny random-init NativeDPLM on CPU; not pretrained"},
            "rows": [{"candidate_id": row["candidate_id"], "heavy": row["heavy"],
                      "light": row["light"], "group_id": row["group_id"], "split": row["split"],
                      "condition": None} for row in self.rows]}

    def cohort_document(self):
        return {
            "schema": fc.COHORT_SCHEMA, "alphabet": ALPHABET,
            "source": {"target": "synthetic-source-function",
                       "assay": "closed-form interior function (source)",
                       "units": "arbitrary units (synthetic)", "scale": "linear",
                       "description": SOURCE_FUNCTION_TEXT},
            "target": {"target": "synthetic-target-function",
                       "assay": "closed-form interior function (target)",
                       "units": "arbitrary units (synthetic)", "scale": "linear",
                       "useful_threshold": 1.0, "description": TARGET_FUNCTION_TEXT},
            "cohorts": {role: list(self.roles[role]) for role in fc.ROLES},
            "grouping_rationale": ("a group is the boundary pair (H0, L3); t=(L3-H0) mod 4 gives "
                                   "four strata of four groups, one per cohort, so every boundary "
                                   "symbol occurs in every split and no group is shared"),
            "notes": DECLARATION}


def _is_constant(code, h0, l3):
    """True when the sequence of this interior code and boundary pair has a single residue throughout."""
    digits = [(code // len(ALPHABET) ** i) % len(ALPHABET) for i in range(INTERIOR)]
    return len({h0, l3, *digits}) == 1


def _sequence_of(code, h0, l3):
    """``(heavy, light)`` of an interior code and a boundary pair (H0 and L3 are the boundary symbols)."""
    digits = [ALPHABET[(code // len(ALPHABET) ** i) % len(ALPHABET)] for i in range(INTERIOR)]
    return ALPHABET[h0] + "".join(digits[:3]), "".join(digits[3:]) + ALPHABET[l3]


def _replace_constant(chosen, h0, l3, rng):
    """``(codes, replaced)``: constant codes in ``chosen`` replaced by codes drawn WITHOUT replacement
    from the unused non-constant codes, by the same generator. Draws nothing, and changes nothing, when
    no constant code was chosen.

    The replacement is a direct draw over the codes that are available, so it is bounded: when fewer
    codes are available than constants to replace (a request for every interior of a group) it is refused
    instead of searching for ever."""
    bad = [code for code in chosen if _is_constant(code, h0, l3)]
    if not bad:
        return chosen, []
    kept, taken = [code for code in chosen if code not in bad], set(chosen)
    available = [code for code in range(INTERIOR_CODES)
                 if code not in taken and not _is_constant(code, h0, l3)]
    require(len(available) >= len(bad),
            f"group (H0={ALPHABET[h0]}, L3={ALPHABET[l3]}) has {len(available)} unused non-constant "
            f"interior codes but {len(bad)} constant sequence(s) to replace: ask for at most "
            f"{SOURCE_AUX_CAPACITY} sequences per source group")
    picks = rng.choice(len(available), size=len(bad), replace=False)
    kept.extend(available[int(i)] for i in picks)
    return sorted(kept), bad


def build_world(spec):
    """Deterministically build the cohorts and every outcome from ``spec`` (and its seed) alone."""
    require(isinstance(spec, SyntheticSpec), "build_world needs a SyntheticSpec")
    rng = np.random.default_rng([int(spec.seed), 4242])
    counts = {"source": spec.source_aux, "source_dev": spec.source_dev,
              "target_train": spec.target_train, "target_eval": spec.target_eval}
    rows, roles, excluded = [], {role: [] for role in fc.ROLES}, []
    outcomes = {name: {} for name in counts}
    for cohort in ("source", "source_dev", "target_train", "target_eval"):
        t_value = COHORT_STRATUM[cohort]
        for h0 in range(len(ALPHABET)):
            l3 = (h0 + t_value) % len(ALPHABET)
            group = f"g-h{h0}-l{l3}"
            chosen = sorted(int(c) for c in rng.choice(len(ALPHABET) ** INTERIOR,
                                                       size=counts[cohort], replace=False))
            if cohort == "source":
                drawn = chosen
                chosen, replaced = _replace_constant(chosen, h0, l3, rng)
                added = sorted(set(chosen) - set(drawn))
                for code, new in zip(replaced, added):
                    heavy, light = _sequence_of(code, h0, l3)
                    excluded.append({
                        "cohort": cohort, "group": group, "interior_code": int(code),
                        "excluded_sequence": {"heavy": heavy, "light": light},
                        "replacement_interior_code": int(new),
                        "reason": ("constant sequence: every EQ over it is true, so no complete "
                                   "auxiliary instance exists; replaced by a code drawn without "
                                   "replacement from the group's unused non-constant codes by the "
                                   "same seeded generator")})
            train_pick = set()
            if cohort == "source":
                train_pick = {int(i) for i in rng.choice(counts[cohort], size=spec.source_train,
                                                         replace=False)}
            for position, code in enumerate(chosen):
                heavy, light = _sequence_of(code, h0, l3)
                cid = f"{PREFIX[cohort]}-{group}-{position:03d}"
                split = (fb.SPLIT_EVAL if cohort in ("source_dev", "target_eval")
                         else fb.SPLIT_TRAIN)
                rows.append({"candidate_id": cid, "heavy": heavy, "light": light,
                             "group_id": group, "split": split, "cohort": cohort})
                if cohort == "source":
                    roles["source_aux"].append(cid)
                    if position in train_pick:
                        roles["source_train"].append(cid)
                        outcomes["source"][cid] = source_function(heavy, light)
                elif cohort == "source_dev":
                    roles["source_dev"].append(cid)
                    outcomes["source_dev"][cid] = source_function(heavy, light)
                else:
                    roles[cohort].append(cid)
                    outcomes[cohort][cid] = target_function(heavy, light)
    world = World(spec=spec, rows=tuple(rows),
                  roles={role: tuple(sorted(ids)) for role, ids in roles.items()},
                  outcomes=outcomes, excluded=tuple(excluded))
    check_world(world)
    return world


def check_world(world):
    """The declared construction, re-derived: counts, unique physical sequences, one ``t`` stratum
    and all four boundary symbols per cohort, and no interior-independent boundary effect."""
    totals = world.spec.totals()
    for role, expected in totals.items():
        require(len(world.roles[role]) == expected,
                f"{role} has {len(world.roles[role])} sequences, expected {expected}")
    physical = [row["heavy"] + "|" + row["light"] for row in world.rows]
    require(len(set(physical)) == len(physical), "a physical sequence is repeated")
    by_cohort = {}
    for row in world.rows:
        by_cohort.setdefault(row["cohort"], []).append(row)
    for cohort, members in by_cohort.items():
        strata = {(ALPHABET.index(row["light"][3]) - ALPHABET.index(row["heavy"][0]))
                  % len(ALPHABET) for row in members}
        require(strata == {COHORT_STRATUM[cohort]},
                f"cohort {cohort} spans t-strata {sorted(strata)}")
        require(len({row["group_id"] for row in members}) == 4, f"cohort {cohort} needs 4 groups")
        require({row["heavy"][0] for row in members} == set(ALPHABET)
                and {row["light"][3] for row in members} == set(ALPHABET),
                f"cohort {cohort} lacks a boundary symbol")
    owners = {}
    for row in world.rows:
        owners.setdefault(row["group_id"], set()).add(row["cohort"])
    require(all(len(cohorts) == 1 for cohorts in owners.values()), "a group is shared by cohorts")
    return True


def _csv_text(values):
    return "candidate_id,utility\n" + "".join(f"{cid},{values[cid]!r}\n" for cid in sorted(values))


SELECTION_RULE = (
    "Source-only and label-free. One generator, numpy default_rng([seed, 4242]), draws for every cohort "
    "and group (in the order source, source_dev, target_train, target_eval; boundary symbol H0 = A, C, "
    "D, E) the interior codes without replacement from the 4096 interiors of the six interior "
    "positions; the source_train subset of the source auxiliary sequences is drawn from the same "
    "generator by position, never by outcome. The source cohort has equal boundary symbols (t=0), so "
    "the interior whose six symbols all equal that symbol makes a CONSTANT sequence (every EQ over it is "
    "true, so it has no complete auxiliary instance). A constant code the draw selects is excluded and "
    "replaced by one code drawn without replacement from the group's unused non-constant codes by the "
    "same generator; a world that draws none is exactly the unmodified draw.")


def provenance_document(world):
    """The synthetic world's selection record: the ACTUAL spec, the source-only rule, every exclusion with
    its replacement, and digests of the rosters and the cohort document written beside it. A software
    record of which sequences were drawn and replaced; not a measured label, not a biological result."""
    spec = world.spec
    return {
        "schema": PROVENANCE_SCHEMA,
        "kind": ("software selection record of a synthetic world (which sequences were drawn, excluded "
                 "and replaced); not a measured label and not a biological result"),
        "spec": {"source_train": spec.source_train, "source_aux": spec.source_aux,
                 "source_dev": spec.source_dev, "target_train": spec.target_train,
                 "target_eval": spec.target_eval, "seed": spec.seed},
        "totals": spec.totals(),
        "declaration": DECLARATION,
        "selection_rule": SELECTION_RULE,
        "constant_interiors": {"interior_codes_per_group": INTERIOR_CODES,
                               "constant_per_source_group": 1,
                               "largest_source_aux_per_group": SOURCE_AUX_CAPACITY},
        "excluded_count": len(world.excluded),
        "excluded": [dict(entry) for entry in world.excluded],
        "roster_sha256": {role: fb.digest(list(ids)) for role, ids in sorted(world.roles.items())},
        "cohort_document_sha256": fb.digest(world.cohort_document())}


def write_provenance(world, path):
    """Publish :func:`provenance_document` as strict JSON at ``path`` (a file this call creates in a new
    directory; an existing file is never replaced). Returns the path."""
    path = Path(path)
    require(not os.path.lexists(path), f"{path} already exists; the provenance record is written once")
    return fb.write_json(path, provenance_document(world))


def write_inputs(world, directory, *, model_seed=0, batch_size=64, withhold_eval=False):
    """Export a REAL native token cache from a tiny random CPU DPLM and write every input file the
    ``functional_shared_transfer`` CLI consumes into a NEW directory, plus ``synthetic_provenance.json``
    (the spec, the source-only selection rule and every exclusion/replacement; returned under
    ``"provenance"``). With ``withhold_eval`` the target-evaluation outcomes are written under ``held/``
    instead, so a caller can place them only after the freeze. Returns the paths."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    model = fne.tiny_model(int(model_seed))
    plan = fne.parse_export_plan(world.export_plan_document())
    tokens = fne.export_dplm_tokens(model, plan, directory / "tokens.json",
                                    directory / "tokens.npz", batch_size=batch_size)
    fb.write_json(directory / "cohorts.json", world.cohort_document())
    provenance = write_provenance(world, directory / PROVENANCE_NAME)
    paths = {"token_header": directory / "tokens.json", "token_npz": directory / "tokens.npz",
             "cohorts": directory / "cohorts.json",
             "source_train_outcomes": directory / "source_train_outcomes.csv",
             "source_dev_outcomes": directory / "source_dev_outcomes.csv",
             "target_train_outcomes": directory / "target_train_outcomes.csv",
             "target_eval_outcomes": directory / "target_eval_outcomes.csv"}
    (directory / "source_train_outcomes.csv").write_text(_csv_text(world.outcomes["source"]),
                                                         encoding="utf-8")
    (directory / "source_dev_outcomes.csv").write_text(_csv_text(world.outcomes["source_dev"]),
                                                       encoding="utf-8")
    (directory / "target_train_outcomes.csv").write_text(_csv_text(world.outcomes["target_train"]),
                                                         encoding="utf-8")
    eval_path = paths["target_eval_outcomes"]
    if withhold_eval:
        (directory / "held").mkdir()
        eval_path = directory / "held" / "target_eval_outcomes.csv"
    eval_path.write_text(_csv_text(world.outcomes["target_eval"]), encoding="utf-8")
    return {**{name: str(path) for name, path in paths.items()},
            "held_eval_outcomes": str(eval_path), "provenance": str(provenance), "export": tokens}
