"""
An independently implemented corpus with planted ground truth.

WHY A GENERATOR AND NOT MORE FIXTURES
-------------------------------------
Real fixtures prove that a specific rule exists and fires on a specific record.
They cannot measure precision or recall, because sixteen sequences drawn from a
corpus of 9,574 are enough to falsify a design and nowhere near enough to
validate one. Measuring an error rate needs a population where the right answer
is known for every pair, which in practice means planting it.

That population also has to be DISJOINT from the one the result is audited on.
The thresholds in `target_identity.OperatingPoint` are calibrated here and
audited on the five curated real families in `target_identity_labels`; nothing
crosses. A family that chooses a threshold and then confirms it reports its own
selection back as an achieved error rate.

INDEPENDENCE
------------
This module imports nothing from `alignment`, `blocking`, `clustering` or
`target_identity`. It builds sequences with its own string surgery and records
its own truth. If it shared the resolver's helpers, a bug in a helper would
plant a corpus that the same bug then scored correctly, and the measurement
would be of nothing.

WHAT IS PLANTED
---------------
Each pathology the design names, with a known answer:

- **Similarity chains.** A ladder of point mutants where adjacent rungs are close
  and the ends are far. Single-linkage merges the ladder; a bounded criterion
  does not.
- **Truncation series.** Nested sub-ranges of one protein: the same target, and
  deliberately NOT the same construct.
- **Containers.** A concatenated construct joining two unrelated targets through
  a poly-glycine linker, so containment fires and must not become identity.
- **Shared containers.** Two unrelated targets annotated with one structure id.
- **Conflicting identifiers.** One byte-identical sequence carrying two accessions.
- **High-degree bridges.** A generic name written across many unrelated targets.
- **Ties.** Two members equidistant from a third, so the merge order has to be
  pinned by something other than luck.
- **Unrelated singletons.** Padding that must not join anything, which is what
  makes a false-merge count meaningful.

DETERMINISM
-----------
Seeded, and the seed is part of the sealed operating evidence. `calibration_population`
is cached so that repeated `calibration_report()` calls cost nothing and cannot
drift.
"""
from __future__ import annotations

import functools
import random
from dataclasses import dataclass, field
from typing import Dict, List

#: The 20 canonical residues. `X`, `B`, `Z` and the stop character are left out
#: on purpose: ambiguity codes belong in the alignment module's tests, not in a
#: population whose job is to have an unambiguous right answer.
RESIDUES = "ACDEFGHIKLMNPQRSTVWY"

#: The linker the resolver recognises as a concatenated construct. Written here
#: as a literal rather than imported, so that a change to the resolver's pattern
#: shows up as a measurement change instead of silently agreeing with itself.
LINKER = "G" * 10

#: Seed for the calibration population. Part of the sealed operating evidence:
#: an error rate quoted without the population it was measured on is not a number
#: anybody can check.
CALIBRATION_SEED = 20260901


@dataclass(frozen=True)
class PlantedLabel:
    """The planted truth for one generated sequence.

    Attributes:
        family: Leave-one-out unit.
        target: Two sequences must land in one biological family exactly when
            these are equal.
        note: What was planted, in words.
    """

    family: str
    target: str
    note: str


@dataclass
class PlantedPopulation:
    """A generated corpus and the answers to it.

    Attributes:
        rows: Source rows in the shape the resolver ingests.
        truth: Residue string to `PlantedLabel`.
        pathologies: Count of each planted pathology, for the guard report.
    """

    rows: List[Dict[str, object]] = field(default_factory=list)
    truth: Dict[str, PlantedLabel] = field(default_factory=dict)
    pathologies: Dict[str, int] = field(default_factory=dict)

    def add(
        self,
        sequence: str,
        family: str,
        target: str,
        note: str,
        *,
        name: str = "",
        pdb: str = "",
        uniprot: str = "",
        copies: int = 1,
    ) -> None:
        """Record one generated sequence and its annotation rows.

        Args:
            sequence: The residue string.
            family: Planted family.
            target: Planted target.
            note: What this record is.
            name: Curator name written on the rows, if any.
            pdb: Structure id written on the rows, if any.
            uniprot: Accession written on the rows, if any.
            copies: How many source rows carry this annotation.
        """
        if sequence in self.truth and self.truth[sequence].target != target:
            raise ValueError(
                f"planted a sequence twice with different targets: "
                f"{self.truth[sequence].target} vs {target}"
            )
        self.truth.setdefault(sequence, PlantedLabel(family, target, note))
        for _ in range(copies):
            self.rows.append({
                "antigen_sequence": sequence,
                "metadata": {
                    "target_name": name,
                    "target_pdb": pdb,
                    "target_uniprot": uniprot,
                },
            })

    def count(self, pathology: str) -> None:
        """Note that one instance of a planted pathology was emitted."""
        self.pathologies[pathology] = self.pathologies.get(pathology, 0) + 1


def _random_protein(rng: random.Random, length: int) -> str:
    """A sequence with no relationship to anything else in the population."""
    return "".join(rng.choice(RESIDUES) for _ in range(length))


def _substitute(rng: random.Random, sequence: str, count: int) -> str:
    """Apply exactly ``count`` point substitutions at distinct positions."""
    residues = list(sequence)
    for position in rng.sample(range(len(residues)), count):
        current = residues[position]
        residues[position] = rng.choice([r for r in RESIDUES if r != current])
    return "".join(residues)


def _insert(rng: random.Random, sequence: str, count: int) -> str:
    """Insert ``count`` residues at one interior position."""
    position = rng.randrange(1, len(sequence) - 1)
    inserted = "".join(rng.choice(RESIDUES) for _ in range(count))
    return sequence[:position] + inserted + sequence[position:]


def build_population(seed: int = CALIBRATION_SEED) -> PlantedPopulation:
    """Generate the planted corpus.

    Args:
        seed: Seed for the generator. The default is the sealed calibration seed.

    Returns:
        A `PlantedPopulation`.
    """
    rng = random.Random(seed)
    population = PlantedPopulation()

    # --- Similarity chains ------------------------------------------------- #
    # A ladder of point mutants: rung k differs from rung k-1 by 1% of its
    # residues, so adjacent rungs sit above the construct threshold and the ends
    # sit far below it. Every rung is the same biological target, so a bounded
    # criterion that splits the ladder into several CONSTRUCTS is right, and a
    # single-linkage pass that welds the ends into one construct is not.
    for chain in range(3):
        length = rng.choice((240, 360, 480))
        base = _random_protein(rng, length)
        step = max(1, length // 100)
        rung = base
        for index in range(6):
            population.add(
                rung, f"chain_{chain}", f"chain_{chain}_target",
                f"rung {index} of a point-mutant ladder, step {step}",
                uniprot=f"c{chain}0000", copies=2,
            )
            rung = _substitute(rng, rung, step)
        population.count("similarity_chain")

    # --- Truncation series ------------------------------------------------- #
    # Nested sub-ranges of one protein. Same target, different constructs: the
    # coverage criterion, not the identity criterion, is what has to separate them.
    for series in range(2):
        base = _random_protein(rng, 600)
        for index, (start, end) in enumerate(((0, 600), (0, 540), (30, 500), (60, 400))):
            population.add(
                base[start:end], f"truncation_{series}", f"truncation_{series}_target",
                f"residues {start}-{end} of one protein",
                uniprot=f"t{series}0000", copies=3,
            )
        population.count("truncation_series")

    # --- Containers -------------------------------------------------------- #
    # A concatenated construct welding two unrelated targets together. Both parts
    # are exact substrings of it, so containment fires; neither may become
    # identity, and the container itself is a third target.
    for container in range(2):
        left = _random_protein(rng, 150)
        right = _random_protein(rng, 420)
        fusion = left + LINKER + right + LINKER + _random_protein(rng, 90)
        population.add(left, f"container_{container}", f"container_{container}_left",
                       "the small component of a fusion", uniprot=f"L{container}0000")
        population.add(right, f"container_{container}", f"container_{container}_right",
                       "the large component of a fusion", uniprot=f"R{container}0000")
        population.add(fusion, f"container_{container}", f"container_{container}_fusion",
                       "a concatenated construct joining two unrelated proteins",
                       name=f"container {container} fusion", copies=4)
        population.count("container")

    # --- Shared structural containers -------------------------------------- #
    # Two unrelated proteins annotated with one structure id, the way a PDB entry
    # holds several polymer chains. Sharing it must quarantine and must not merge.
    for entry in range(3):
        first = _random_protein(rng, 200)
        second = _random_protein(rng, 310)
        code = f"9x{entry}z"
        population.add(first, f"complex_{entry}", f"complex_{entry}_chain_a",
                       "chain A of a two-chain structure", pdb=code,
                       uniprot=f"a{entry}0000", copies=2)
        population.add(second, f"complex_{entry}", f"complex_{entry}_chain_b",
                       "chain B of the same structure, a different gene", pdb=code,
                       uniprot=f"b{entry}0000", copies=2)
        population.count("shared_container")

    # --- Conflicting identifiers ------------------------------------------- #
    # One byte-identical sequence carrying two accessions. There is no evidence
    # for either, so the resolver must report the disagreement rather than settle
    # it on sort order.
    for conflict in range(2):
        sequence = _random_protein(rng, 190)
        population.add(sequence, f"conflict_{conflict}", f"conflict_{conflict}_target",
                       "one sequence, first accession", uniprot=f"p{conflict}1000",
                       name=f"conflict {conflict} alpha", copies=3)
        population.add(sequence, f"conflict_{conflict}", f"conflict_{conflict}_target",
                       "one sequence, second accession", uniprot=f"p{conflict}1001",
                       name=f"conflict {conflict} beta", copies=1)
        population.count("accession_conflict")

    # --- High-degree bridges ----------------------------------------------- #
    # One generic curator name written across many unrelated proteins. Approving
    # it would merge them all; the name has to be refused and quarantine instead.
    for index in range(7):
        population.add(
            _random_protein(rng, 200 + index * 17),
            "generic_name", f"generic_name_target_{index}",
            "one of several unrelated proteins sharing a generic label",
            name="antigen", copies=2,
        )
    population.count("high_degree_name")

    # --- Ties -------------------------------------------------------------- #
    # Two mutants equidistant from one parent, so the merge order cannot be
    # decided by edge strength and something else has to pin it.
    parent = _random_protein(rng, 300)
    population.add(parent, "tie", "tie_target", "the parent of a tied pair",
                   uniprot="tie0000", copies=2)
    for side in ("left", "right"):
        population.add(
            _substitute(rng, parent, 3), "tie", "tie_target",
            f"the {side} arm of a tie, three substitutions from the parent",
            uniprot="tie0000", copies=2,
        )
    population.count("tie")

    # --- An interior indel, the case exact containment cannot see ----------- #
    indel_base = _random_protein(rng, 280)
    population.add(indel_base, "indel", "indel_target",
                   "the shorter member of an interior-insertion pair",
                   name="indel pair", copies=2)
    population.add(_insert(rng, indel_base, 2), "indel", "indel_target",
                   "the same construct plus a two-residue interior insertion",
                   name="indel pair", copies=2)
    population.count("interior_indel")

    # --- Unrelated singletons ---------------------------------------------- #
    # Padding. Without records that must join nothing, a false-merge count of
    # zero says only that the resolver merged nothing.
    for index in range(12):
        population.add(
            _random_protein(rng, rng.randint(120, 700)),
            "singletons", f"singleton_{index}",
            "an unrelated protein that must join nothing",
            uniprot=f"s{index:05d}", copies=1,
        )
    population.count("unrelated_singleton")

    return population


@functools.lru_cache(maxsize=4)
def calibration_population(seed: int = CALIBRATION_SEED) -> PlantedPopulation:
    """The cached calibration population.

    Args:
        seed: Generator seed.

    Returns:
        The `PlantedPopulation` used to calibrate the operating point.
    """
    return build_population(seed)
