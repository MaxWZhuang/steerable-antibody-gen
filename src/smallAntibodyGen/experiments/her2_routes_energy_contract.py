"""The energy route's training contract (version 2), its exposure trace and its runtime mode.

Torch-free and pure: hashing, validation and bookkeeping only, so the resume and load refusals run
before a model, an optimizer, a directory or a generator exists, and the tests can exercise them on
plain dictionaries.

What it exists to fix. The earlier ``train_energy`` coupled the *population* to the *conditioning*:
the conditioned arm trained on every eligible row and the density-only arm on the HER2 rows alone
(historically 365,917 against 2,146), so the comparison between them confounded "was the model given
a context" with "how much data did it see". A checkpoint also recorded nothing about which of those
it was: the arm name was an identity field, and nothing bound the population, the draw law, the
microbatch or the phase budgets that actually produced the weights, so a resume under a changed
``gpu_profile.json`` or a changed population continued a different experiment under the old name.

``energy_contract`` version 2 is the document that closes that. It is built from the RESOLVED
values (the profile-selected microbatch, the phase budgets, the cohort the random indices address,
the bin mapping the context was built from), hashed canonically, and the hash is the checkpoint
identity's ``schedule_sha256``. It is stored in ``run.json`` and in every checkpoint the run writes.

* ``common_sha256`` covers everything EXCEPT the arm and the conditioning, so two arms that differ
  only in whether the model was given a context share it, and a test can say so.
* :class:`ExposureTrace` is a cumulative, ordered hash chain over the draws themselves (family,
  group, tile, member indices, identities and sites per step) with per-phase counts. Equal row
  totals do not show that two arms saw the same rows; an equal chain does. The counts are not
  free-standing either: under a contract, ``rows`` is ``steps * microbatch`` and the site exposures
  lie inside the tile sizes the cohort's regions allow (:func:`exposure_bounds`), and a checkpoint's
  exposure ledger must add up to them.
* :class:`EnergyRuntime` is the validated, immutable mode a loaded model carries. The model's
  architecture (condition tables, context towers, ``document()['context']['conditioned']``) cannot
  say how an arm was trained, so scoring and sampling read the mode from here.

Nothing here states that a model is good. It states what a checkpoint was trained under, and refuses
to continue or to score it as anything else.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from . import her2_routes_data as data_lib
from . import her2_support_paths as paths
from .her2_routes_lineage import LineageError
from .her2_runtime import require

ENERGY_ROUTE = "dwjs_energy"
ENERGY_CONTRACT_VERSION = 2
ENERGY_CONTRACT_SCHEMA = "her2-routes-energy-contract/2"
TRACE_SCHEMA = "her2-routes-energy-trace/1"

#: The two arms an energy checkpoint can be, and the conditioning mode each one means. The arm is
#: an identity field; the mode is what the model was actually given.
ARM_FULL = "full"
ARM_NULL = "null"
MODE_CONDITIONED = "conditioned"
MODE_NULL = "null"
ARM_MODE = {ARM_FULL: MODE_CONDITIONED, ARM_NULL: MODE_NULL}

PHASE_DENOISER = "denoiser"
PHASE_ENERGY = "energy"
PHASES = (PHASE_DENOISER, PHASE_ENERGY)

#: Training populations. Neither names a conditioning mode: any arm may train on any of them.
POPULATION_ALL_ELIGIBLE = "all_eligible_training"
POPULATION_HER2_ONLY = "her2_only"
POPULATIONS = {
    POPULATION_ALL_ELIGIBLE: ("every eligible measured TRAINING row that declares an edit map, "
                              "from every enrolled source"),
    POPULATION_HER2_ONLY: ("the eligible measured TRAINING rows of the HER2 validation source "
                           "only"),
}

HER2_SOURCE = "alphabind_her2_validation"
FAMILY_HER2 = "her2"
FAMILY_OTHER = "other"
#: P(her2) when both families are present. A single present family is chosen without a draw.
HER2_PROBABILITY = 2.0 / 3.0

#: The fields of the contract that hold when only the conditioning differs.
COMMON_KEYS = ("route", "population", "sampling", "initialization", "phases", "forward_ledger")
_HASH_KEYS = ("energy_contract_sha256", "common_sha256")
_MISSING = object()


def _plain(document):
    """The document as plain JSON types, so equality and hashing never depend on tuple vs list."""
    return json.loads(json.dumps(document, sort_keys=True, allow_nan=False))


def _line(value):
    return (json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def arm_for(conditioned):
    """``full`` for a conditioned run, ``null`` otherwise. ``conditioned`` must be a real bool."""
    require(isinstance(conditioned, bool),
            f"conditioned must be True or False, got {conditioned!r}: the conditioning mode is "
            "never inferred from a population or a default")
    return ARM_FULL if conditioned else ARM_NULL


def require_population(name):
    require(name in POPULATIONS,
            f"Unknown energy training population {name!r}; the declared ones are "
            f"{sorted(POPULATIONS)}. A population never implies a conditioning mode.")
    return name


# ---------------------------------------------------------------------------
# the cohort the random indices address
# ---------------------------------------------------------------------------

def family_of(row):
    return FAMILY_HER2 if row.source == HER2_SOURCE else FAMILY_OTHER


def member_sort_key(row):
    return (str(row.identity), str(row.source), str(row.study), str(row.assay), str(row.target),
            str(row.parent), str(row.family), str(row.partition))


def member_record(row):
    """What identifies a cohort member: the full construct, its region and its provenance."""
    return [str(row.identity), str(row.source), str(row.study), str(row.assay), str(row.target),
            str(row.parent), str(row.family), str(row.partition),
            [[str(chain), int(index)] for chain, index in row.region]]


def canonical_groups(rows):
    """``{family: ((key, members), ...)}`` in a canonical order, not enumeration order.

    A group is one ``(assay, region)`` pool, and a step draws a family, then a group of it, then
    members of that group by index. Those indices address THIS ordering, so it is sorted by value:
    families by name, groups by ``(assay, region)``, members by their full identity and
    provenance. The same set of rows therefore always produces the same draws from the same seed,
    whatever order the sources were enumerated in.
    """
    table = {}
    for row in rows:
        key = (str(row.assay), tuple((str(chain), int(index)) for chain, index in row.region))
        table.setdefault(family_of(row), {}).setdefault(key, []).append(row)
    return {family: tuple((key, tuple(sorted(table[family][key], key=member_sort_key)))
                          for key in sorted(table[family]))
            for family in sorted(table)}


def cohort_document(groups):
    """The digest and counts of the ordered cohort. The digest covers every member's record.

    ``region_lengths`` is the sorted set of distinct region sizes over every group. It is what
    bounds a trace's site exposures (a step presents one tile of one group's region), so it is
    bound in the contract's digest with the cohort it describes.
    """
    digest, families, rows, region_lengths = hashlib.sha256(), {}, 0, set()
    for family in sorted(groups):
        members_in_family = 0
        for position, (key, members) in enumerate(groups[family]):
            digest.update(_line(["group", family, position, key[0],
                                 [list(site) for site in key[1]], len(members)]))
            for row in members:
                digest.update(_line(member_record(row)))
            members_in_family += len(members)
            region_lengths.add(len(key[1]))
        families[family] = {"groups": len(groups[family]), "members": members_in_family}
        rows += members_in_family
    return {"sha256": digest.hexdigest(), "rows": rows, "families": families,
            "region_lengths": sorted(region_lengths),
            "ordering": ("families by name; groups by (assay, region); members by full identity "
                         "and provenance. The sampler's random indices address exactly this "
                         "order, which the digest covers member by member."),
            "member_record": ("[identity (heavy|light), source, study, assay, target, parent, "
                              "family, partition, region]")}


def condition_assignment_document(groups, bins, *, default_bin, spec_document):
    """The bin each cohort member's condition is built from, as the conditioned arm uses it.

    ``bins`` is the frozen ``population_key -> bin`` mapping fitted on TRAIN exact values; a member
    with no entry trains at ``default_bin``. ``spec_document`` carries the vocabulary and the two
    dropout rates the context builder applies on top (the dropout draws come from ``mask_rng``).
    """
    digest, counts = hashlib.sha256(), {}
    for family in sorted(groups):
        for _key, members in groups[family]:
            for row in members:
                name = bins.get(row.population_key, default_bin)
                counts[name] = counts.get(name, 0) + 1
                digest.update(_line([str(row.identity), str(row.family), str(name),
                                     str(row.assay)]))
    spec = dict(spec_document)
    return {"sha256": digest.hexdigest(), "bins": dict(sorted(counts.items())),
            "default_bin": str(default_bin),
            "mapping_entries": len(bins),
            "vocabulary": {key: spec.get(key) for key in ("families", "bins", "assays")},
            "dropout": {"bin": spec.get("bin_dropout"), "assay": spec.get("assay_dropout")},
            "source": ("the frozen train-only quality cuts applied to each member's population "
                       "key, then bin/assay dropout drawn from mask_rng")}


def family_law(present):
    """The family sampling law for the families a cohort actually has, as recorded."""
    present = sorted(present)
    require(present, "The energy cohort has no family to draw from")
    if len(present) == 2:
        return {"present": present, "her2_probability": HER2_PROBABILITY,
                "rule": ("sampler_rng.random() < 2/3 chooses her2, otherwise other; one draw per "
                         "step"),
                "draw_consumed_per_step": True}
    return {"present": present, "her2_probability": None,
            "rule": (f"only {present[0]!r} exists, so it is chosen without a draw; a family with "
                     "no rows is never drawn"),
            "draw_consumed_per_step": False}


def draw_family(present, rng):
    """One step's family. Only a family that exists can be drawn."""
    present = set(present)
    require(present, "The energy cohort has no family to draw from")
    if len(present) == 2:
        return FAMILY_HER2 if rng.random() < HER2_PROBABILITY else FAMILY_OTHER
    return next(iter(present))


def forwards_per_row(phase, walk_steps):
    """Row-forwards one row costs: 1 in the denoiser phase, ``walk_steps + 2`` in the energy phase."""
    require(phase in PHASES, f"Unknown energy phase {phase!r}")
    return 1 if phase == PHASE_DENOISER else int(walk_steps) + 2


def forward_ledger_document(walk_steps):
    return {"counter": "student_forwards",
            "unit": ("ROW-FORWARDS: one per row per forward evaluation. Not batched calls, not "
                     "backward passes and not FLOPs."),
            "denoiser_phase_per_row": 1,
            "energy_phase_per_row": int(walk_steps) + 2,
            "energy_phase_breakdown": (f"{int(walk_steps)} Langevin score evaluations (the walk) "
                                       "+ 1 energy evaluation of the noisy real window + 1 of the "
                                       "detached fake"),
            "excludes": ("the context towers' forwards, the frozen denoiser (it is not evaluated "
                         "in the energy phase) and every backward pass")}


# ---------------------------------------------------------------------------
# the contract
# ---------------------------------------------------------------------------

def build_contract(*, arm, population, cohort, sampling, initialization, phases, forward_ledger,
                   condition_assignment=None):
    """The version-2 contract of one run, as plain JSON, with its hashes inside.

    ``condition_assignment`` is required for the conditioned arm (the mapping its context is built
    from) and forbidden for the null arm (which is given no context at all).
    """
    require(arm in ARM_MODE, f"Unknown energy arm {arm!r}; the arms are {sorted(ARM_MODE)}")
    require(population in POPULATIONS, f"Unknown energy training population {population!r}")
    mode = ARM_MODE[arm]
    require((mode == MODE_CONDITIONED) == (condition_assignment is not None),
            f"the {arm!r} arm "
            + ("needs the condition assignment its context is built from"
               if mode == MODE_CONDITIONED else
               "is given no context, so it has no condition assignment to bind"))
    common = {"route": ENERGY_ROUTE,
              "population": {"name": population, "rule": POPULATIONS[population],
                             "cohort": cohort},
              "sampling": sampling, "initialization": initialization, "phases": phases,
              "forward_ledger": forward_ledger}
    body = {"schema": ENERGY_CONTRACT_SCHEMA, "version": ENERGY_CONTRACT_VERSION, **common,
            "arm": arm,
            "conditioning": {
                "mode": mode,
                "context": ("built for every row by the context builder, then encoded by the "
                            "energy tower in the energy phase and the denoiser's own tower in the "
                            "denoiser phase" if mode == MODE_CONDITIONED else
                            "None on every call: the context builder and both context towers are "
                            "never invoked, in training, scoring or sampling"),
                "condition_assignment": condition_assignment,
                "requested_quality_bin": ("drawn from the assignment" if mode == MODE_CONDITIONED
                                          else "not applicable"),
                "dropout_generator": "mask_rng, separate from the sampler's generator"},
            "common_sha256": paths.digest_document(_plain(common))}
    body = _plain(body)
    body["energy_contract_sha256"] = paths.digest_document(body)
    return body


def contract_digest(contract):
    """The digest of ``contract`` without its own recorded hash."""
    return paths.digest_document({key: value for key, value in contract.items()
                                  if key != "energy_contract_sha256"})


def common_digest(contract):
    return paths.digest_document(_plain({key: contract.get(key) for key in COMMON_KEYS}))


def contract_differences(stored, requested, *, prefix="", limit=12):
    """Dotted paths at which two contracts differ, leaving the derived hashes out."""
    found = []
    for key in sorted(set(stored) | set(requested)):
        if not prefix and key in _HASH_KEYS:
            continue
        left, right = stored.get(key, _MISSING), requested.get(key, _MISSING)
        if left == right:
            continue
        name = f"{prefix}{key}"
        if isinstance(left, dict) and isinstance(right, dict):
            found.extend(contract_differences(left, right, prefix=name + ".", limit=limit) or [name])
        else:
            found.append(name)
        if len(found) >= limit:
            break
    return found[:limit]


class EnergyCheckpointIncompatible(LineageError):
    """A checkpoint written before energy_contract version 2.

    A :class:`~.her2_routes_lineage.LineageError`, so a campaign stage that meets one records
    ``blocked_readiness`` with the reason rather than crashing: the checkpoint is an incompatible
    prerequisite, preserved untouched, and is never resumed, scored or relabelled as a matched
    version-2 control.
    """


LEGACY_REFUSAL = (
    "carries no energy_contract: it was written before energy_contract version "
    f"{ENERGY_CONTRACT_VERSION}, under the earlier coupling of population to conditioning (the full "
    "arm saw every eligible row, the density-only arm only the HER2 rows) and without a record of "
    "its draws, microbatch or budgets. It cannot be resumed, scored or relabelled as a matched "
    f"version-{ENERGY_CONTRACT_VERSION} control. The artifact is preserved untouched; train the "
    "matched pair under a new revision.")


def validated_contract(stored, *, where):
    """The contract document itself, recomputed rather than trusted, or a clear refusal. Pure.

    The version and schema, the document's own hash, the common digest over its common fields,
    the arm/mode pairing and the route are all checked from the document alone. A checkpoint adds
    its identity on top (:func:`stored_contract`); a run record adds its own recorded fields
    (:func:`require_matched_controls`). ``where`` names the evidence in every message.
    """
    require(isinstance(stored, dict), f"{where}: the energy_contract is not a mapping")
    require(stored.get("version") == ENERGY_CONTRACT_VERSION
            and stored.get("schema") == ENERGY_CONTRACT_SCHEMA,
            f"{where}: the energy contract is version {stored.get('version')!r} "
            f"({stored.get('schema')!r}); this code reads version {ENERGY_CONTRACT_VERSION}")
    recomputed = contract_digest(stored)
    require(stored.get("energy_contract_sha256") == recomputed,
            f"{where}: the stored energy contract does not hash to its own recorded "
            f"energy_contract_sha256 (recorded {stored.get('energy_contract_sha256')}, recomputed "
            f"{recomputed}); it was edited after it was written")
    require(stored.get("common_sha256") == common_digest(stored),
            f"{where}: the stored energy contract's common digest does not match its own fields")
    arm = stored.get("arm")
    mode = (stored.get("conditioning") or {}).get("mode")
    require(arm in ARM_MODE and ARM_MODE[arm] == mode,
            f"{where}: the contract's arm {arm!r} and conditioning mode {mode!r} are inconsistent "
            f"(the {ARM_FULL!r} arm is {MODE_CONDITIONED!r} and the {ARM_NULL!r} arm is "
            f"{MODE_NULL!r})")
    require(stored.get("route") == ENERGY_ROUTE,
            f"{where}: the contract was not produced for the {ENERGY_ROUTE!r} route")
    return stored


def stored_contract(payload, *, path):
    """The validated contract a checkpoint carries, or a clear refusal. Pure; nothing is loaded.

    The stored hash is RECOMPUTED, never trusted: it must match the document, the identity's
    ``schedule_sha256`` and the common digest, and the arm must agree with the conditioning mode.
    """
    require(isinstance(payload, dict), f"{path} is not a checkpoint payload")
    identity = payload.get("identity")
    require(isinstance(identity, dict), f"{path} carries no identity block")
    stored = payload.get("energy_contract")
    if stored is None:
        raise EnergyCheckpointIncompatible(f"{path} {LEGACY_REFUSAL}")
    stored = validated_contract(stored, where=path)
    require(identity.get("schedule_sha256") == stored["energy_contract_sha256"],
            f"{path}: the energy contract's hash {stored['energy_contract_sha256']} is not the "
            f"identity's schedule_sha256 ({identity.get('schedule_sha256')}); the weights are "
            "bound to a different training schedule than the contract on file")
    require(identity.get("arm") == stored["arm"],
            f"{path}: the identity's arm {identity.get('arm')!r} is not the contract's arm "
            f"{stored['arm']!r}")
    require(identity.get("route") == ENERGY_ROUTE,
            f"{path} was not produced for the {ENERGY_ROUTE!r} route")
    return stored


# ---------------------------------------------------------------------------
# exposure arithmetic: what a trace and a ledger must add up to under a contract
# ---------------------------------------------------------------------------

def _is_count(value, *, minimum=0):
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def tile_lengths(region_length, max_query_sites):
    """The distinct tile sizes the sampler can draw from a region of ``region_length`` sites.

    The sampler cuts a region with ``data_lib._tiles`` and draws one tile per step, so this asks
    that function rather than restating its rule.
    """
    return sorted({len(tile) for tile in data_lib._tiles(range(int(region_length)),
                                                         max_query_sites)})


def exposure_bounds(contract, *, where):
    """What ``contract`` fixes about any trace or ledger that claims to have run it.

    The microbatch is the RESOLVED one the contract binds (``sampling.microbatch``); the tile
    sizes follow from ``sampling.tile_rule.max_query_sites`` and the cohort's recorded
    ``region_lengths``; the phase budgets and the row-forwards per row come from the contract too.
    A contract that records none of these cannot vouch for a trace, and is refused rather than
    waved through. Returns plain data; ``contract`` is only read.
    """
    def entry(document, key, label):
        require(isinstance(document, dict) and key in document,
                f"{where}: the energy contract records no {label}, so the exposure arithmetic of "
                "a trace cannot be validated against it")
        return document[key]

    sampling = entry(contract, "sampling", "sampling block")
    microbatch = entry(sampling, "microbatch", "sampling.microbatch")
    require(_is_count(microbatch, minimum=1),
            f"{where}: the contract's sampling.microbatch is {microbatch!r}, not a positive integer")
    tile_rule = entry(sampling, "tile_rule", "sampling.tile_rule")
    max_query_sites = entry(tile_rule, "max_query_sites", "sampling.tile_rule.max_query_sites")
    require(isinstance(max_query_sites, int) and not isinstance(max_query_sites, bool),
            f"{where}: the contract's max_query_sites is {max_query_sites!r}, not an integer")
    cohort = entry(entry(contract, "population", "population block"), "cohort",
                   "population.cohort")
    lengths = entry(cohort, "region_lengths", "population.cohort.region_lengths")
    require(isinstance(lengths, list) and lengths
            and all(_is_count(length, minimum=1) for length in lengths),
            f"{where}: the contract's cohort region_lengths {lengths!r} is not a non-empty list of "
            "positive integers")
    tiles = sorted({size for length in lengths for size in tile_lengths(length, max_query_sites)})
    phases = entry(contract, "phases", "phases block")
    budget = entry(phases, "budget", "phases.budget")
    require(isinstance(budget, dict) and set(budget) == set(PHASES)
            and all(_is_count(budget[name], minimum=1) for name in PHASES),
            f"{where}: the contract's phase budgets {budget!r} are not positive integers for "
            f"exactly {list(PHASES)}")
    ledger = entry(contract, "forward_ledger", "forward_ledger")
    forwards = {}
    for phase, key in ((PHASE_DENOISER, "denoiser_phase_per_row"),
                       (PHASE_ENERGY, "energy_phase_per_row")):
        value = entry(ledger, key, f"forward_ledger.{key}")
        require(_is_count(value, minimum=1),
                f"{where}: the contract's forward_ledger.{key} is {value!r}, not a positive "
                "integer")
        forwards[phase] = value
    walk = phases.get("walk")
    if isinstance(walk, dict) and "steps" in walk:
        require(_is_count(walk["steps"], minimum=1)
                and all(forwards[phase] == forwards_per_row(phase, walk["steps"])
                        for phase in PHASES),
                f"{where}: the contract's forward ledger {forwards} is not the row-forwards of "
                f"its own walk of {walk['steps']!r} steps")
    return {"microbatch": int(microbatch), "tiles": tiles,
            "budget": {name: int(budget[name]) for name in PHASES}, "forwards_per_row": forwards}


def require_phase_counts(phases, bounds, *, where):
    """Each phase's counts must be what its steps come to under the contract's bounds.

    ``phases`` is ``{phase: {"steps", "rows", "site_exposures"}}`` (an :class:`ExposureTrace`'s).
    A step presents ``microbatch`` rows at one tile, so ``rows`` is exactly ``steps * microbatch``
    and the site exposures are ``microbatch`` times the sum of the step's tile sizes: a multiple of
    the microbatch between the smallest and the largest tile size times the rows. A phase with no
    step has no row and no site. The energy phase starts only after the denoiser phase is done.
    """
    microbatch, tiles, budget = bounds["microbatch"], bounds["tiles"], bounds["budget"]
    for name in PHASES:
        steps, rows, sites = (phases[name][key] for key in ("steps", "rows", "site_exposures"))
        require(steps <= budget[name],
                f"{where}: its {name} phase holds {steps} steps against a budget of {budget[name]}")
        require(rows == steps * microbatch,
                f"{where}: its {name} phase holds {steps} steps and {rows} rows, but the "
                f"contract's microbatch of {microbatch} makes {steps * microbatch}; the trace is "
                "not the run the contract describes")
        low, high = rows * min(tiles), rows * max(tiles)
        require(low <= sites <= high and sites % microbatch == 0,
                f"{where}: its {name} phase holds {sites} site exposures for {steps} steps, but "
                f"the contract's tiles of {tiles} sites at a microbatch of {microbatch} allow "
                f"{low} to {high}, in multiples of {microbatch}")
    require(phases[PHASE_ENERGY]["steps"] == 0
            or phases[PHASE_DENOISER]["steps"] == budget[PHASE_DENOISER],
            f"{where}: its energy phase has steps although its denoiser phase holds "
            f"{phases[PHASE_DENOISER]['steps']} of {budget[PHASE_DENOISER]}; the energy phase "
            "starts only after the denoiser phase is complete")


def ledger_totals(state, *, where):
    """A checkpoint's exposure-ledger totals, after the ledger's own restore and additivity check.

    ``state`` is what ``checkpoints.save`` stored (``ExposureLedger.state()``). It is only READ:
    ``ExposureLedger.restore`` builds a new ledger and ``require_additive`` checks that every
    bucket kind sums to the global counters. A missing or malformed ledger is refused.
    """
    require(isinstance(state, dict),
            f"{where} carries no exposure ledger, so its exposure counters cannot be checked "
            "against its trace")
    try:
        ledger = data_lib.ExposureLedger.restore(state)
    except Exception as error:
        require(False, f"{where}: its exposure ledger does not restore "
                       f"({type(error).__name__}: {error})")
    try:
        ledger.require_additive()
    except ValueError as error:
        require(False, f"{where}: {error}")
    return dict(ledger.totals)


def require_ledger_totals(totals, phases, bounds, *, where):
    """The ledger's four step-derived counters must equal what the trace and contract imply.

    ``optimizer_updates`` is the steps; ``microbatch_rows`` the rows; ``supervised_targets`` the
    site exposures; ``student_forwards`` the denoiser rows plus the energy rows times
    ``walk_steps + 2``. The ledger is recorded by the same step that records the trace.
    """
    forwards = bounds["forwards_per_row"]
    expected = {
        "optimizer_updates": sum(phases[name]["steps"] for name in PHASES),
        "microbatch_rows": sum(phases[name]["rows"] for name in PHASES),
        "student_forwards": sum(phases[name]["rows"] * forwards[name] for name in PHASES),
        "supervised_targets": sum(phases[name]["site_exposures"] for name in PHASES)}
    for key, value in expected.items():
        require(totals.get(key) == value,
                f"{where}: its exposure ledger records {totals.get(key)!r} {key} but its trace and "
                f"the contract's microbatch, tiles and forwards per row make {value}; the ledger "
                "and the trace disagree")


def _phase_position(payload, budget, *, path):
    """``(phase, step)`` of a checkpoint, refused unless the phase is named and the step bounded."""
    phase = payload.get("energy_phase")
    require(phase in PHASES,
            f"{path} records energy_phase {phase!r}; a checkpoint names {list(PHASES)}. A missing "
            "or unknown phase would restart both phases on trained weights, so it is refused")
    step = payload.get("step")
    require(isinstance(step, int) and not isinstance(step, bool) and 0 <= step <= budget[phase],
            f"{path} records {phase} step {step!r} against a {phase} budget of {budget[phase]}; "
            "the phase-local step must be an integer inside it")
    require(payload.get("phase_budget") == budget,
            f"{path} records phase budgets {payload.get('phase_budget')!r} but the contract's are "
            f"{budget}; a changed budget is a different experiment")
    return phase, step


def _trace_for(payload, phase, step, contract, *, path):
    """The checkpoint's trace, refused unless it is the run ``contract`` describes at this position.

    Beyond the step counts: every phase's rows and site exposures must follow from the contract's
    microbatch and tiles, and the checkpoint's exposure ledger must restore, add up and agree with
    the trace. Every checkpoint ``train_energy`` writes carries all three, so a missing one is a
    refusal, not a skipped check.
    """
    budget = contract["phases"]["budget"]
    state = payload.get("energy_trace")
    require(isinstance(state, dict),
            f"{path} carries no energy_trace, so the draws already taken cannot be continued")
    trace = ExposureTrace.from_state(state, path=path)
    expected = {PHASE_DENOISER: step if phase == PHASE_DENOISER else budget[PHASE_DENOISER],
                PHASE_ENERGY: 0 if phase == PHASE_DENOISER else step}
    held = {name: trace.phases[name]["steps"] for name in PHASES}
    require(held == expected,
            f"{path}: its exposure trace holds {held} steps per phase but the checkpoint is at "
            f"{phase} step {step} (expected {expected}); the trace and the weights disagree")
    bounds = exposure_bounds(contract, where=path)
    require_phase_counts(trace.phases, bounds, where=path)
    totals = ledger_totals(payload.get("exposure"), where=path)
    require_ledger_totals(totals, trace.phases, bounds, where=path)
    return trace


def require_resumable(payload, *, contract, path):
    """Refuse a ``last.pt`` that is not a resumable state of exactly ``contract``'s run.

    Called on CPU metadata BEFORE anything is built, restored or created. Requires the version-2
    contract, an identical recomputed contract, a recognised phase, a phase-local step inside the
    exact budget, optimizer state, and a trace consistent with that position: per-phase steps,
    rows and site exposures that follow from the contract's microbatch and tiles, and an exposure
    ledger that restores, adds up and agrees with them. A checkpoint that completed a phase (step
    equals its budget) is legitimate: the resume starts the next phase. The payload is only read.
    """
    stored = stored_contract(payload, path=path)
    if stored != contract:
        changed = contract_differences(stored, contract) or ["(a derived hash)"]
        require(False,
                f"{path} was trained under a different energy contract: {changed} changed. "
                "Resuming would continue other draws, another population, shape or budget under "
                "the old checkpoint's name. Declare a new revision; the superseded root stays "
                "read-only.")
    budget = contract["phases"]["budget"]
    phase, step = _phase_position(payload, budget, path=path)
    require(payload.get("optimizer") is not None,
            f"{path} holds no optimizer state, so it is not a resumable checkpoint (a terminal "
            "best.pt is for inference only)")
    trace = _trace_for(payload, phase, step, stored, path=path)
    return {"phase": phase, "step": step, "completed_phase": step == budget[phase],
            "trace": trace}


def inference_binding(payload, *, path, arm=None):
    """``(stored contract, terminal)`` for a checkpoint loaded to score or sample.

    No requested contract exists here, so the stored one is validated against itself and the
    identity, and the checkpoint's trace and exposure ledger against the stored contract (the same
    arithmetic :func:`require_resumable` applies). An explicit ``arm`` that contradicts the
    checkpoint is refused. ``terminal`` is True for the energy phase's final step: the best
    checkpoint, which carries no optimizer.
    """
    stored = stored_contract(payload, path=path)
    if arm is not None:
        require(arm == stored["arm"],
                f"{path} is the {stored['arm']!r} arm but the {arm!r} arm was requested; scoring "
                "one arm's weights under the other's name is refused")
    budget = stored["phases"]["budget"]
    phase, step = _phase_position(payload, budget, path=path)
    _trace_for(payload, phase, step, stored, path=path)
    return stored, bool(phase == PHASE_ENERGY and step == budget[PHASE_ENERGY])


def _record_contract(record, *, label):
    """A run record's contract, recomputed, with the record's own recorded fields held to it."""
    where = f"the {label} run record"
    require(isinstance(record, dict), f"{where} is not a mapping")
    contract = validated_contract(record.get("energy_contract"), where=f"{where}'s energy contract")
    for key in ("energy_contract_sha256", "common_sha256", "arm"):
        require(key not in record or record[key] == contract[key],
                f"{where} records {key} {record.get(key)!r} but its energy contract has "
                f"{contract[key]!r}")
    require("conditioned" not in record
            or record["conditioned"] is (ARM_MODE[contract["arm"]] == MODE_CONDITIONED),
            f"{where} records conditioned={record.get('conditioned')!r} but its energy contract is "
            f"the {contract['arm']!r} arm")
    identity = record.get("identity")
    if identity is not None:
        require(isinstance(identity, dict), f"{where}'s identity is not a mapping")
        for key, value in (("schedule_sha256", contract["energy_contract_sha256"]),
                           ("arm", contract["arm"]), ("route", ENERGY_ROUTE)):
            require(identity.get(key) == value,
                    f"{where}'s identity has {key} {identity.get(key)!r}, not {value!r} as its "
                    "energy contract says")
    return contract


def _record_progress(record, contract, *, label):
    """``(phase_updates, ExposureTrace)`` of a run record, refused unless it is consistent.

    The recorded ``phase_updates`` must be the trace's own per-phase steps, within the contract's
    budgets; the trace's rows and site exposures must follow from the contract's microbatch and
    tiles; and the record's exposure-ledger totals, when it carries them, must agree with both.
    """
    where = f"the {label} run record"
    bounds = exposure_bounds(contract, where=where)
    state = record.get("exposure_trace")
    trace = ExposureTrace.from_state(state, path=where)
    updates = record.get("phase_updates")
    require(isinstance(updates, dict) and set(updates) == set(PHASES)
            and all(_is_count(updates[name]) for name in PHASES),
            f"{where}: phase_updates must count exactly the phases {list(PHASES)} with "
            f"non-negative integers, got {updates!r}")
    held = {name: trace.phases[name]["steps"] for name in PHASES}
    require({name: updates[name] for name in PHASES} == held,
            f"{where} records phase_updates {dict(updates)} but its exposure trace holds {held} "
            "steps per phase")
    if state.get("total") is not None:
        total = {key: sum(trace.phases[name][key] for name in PHASES)
                 for key in ("steps", "rows", "site_exposures")}
        require(state["total"] == total,
                f"{where}: its exposure trace records the total {state['total']} but its phases "
                f"add up to {total}")
    require_phase_counts(trace.phases, bounds, where=where)
    ledger = record.get("exposure")
    if ledger is not None:
        require(isinstance(ledger, dict) and isinstance(ledger.get("totals"), dict),
                f"{where}: its exposure ledger carries no totals")
        require_ledger_totals(ledger["totals"], trace.phases, bounds, where=where)
    return held, trace


def require_matched_controls(full, null):
    """Do a full and a null run share everything but the conditioning? Raises if not.

    ``full`` and ``null`` are run records carrying ``energy_contract``, ``exposure_trace`` and
    ``phase_updates``. Nothing recorded is trusted: BOTH contracts are recomputed (version, hash,
    common digest, arm/mode) and each record's own fields and identity are held to its contract;
    only then are the actual common fields compared. Each trace must be arithmetically the run
    its contract describes and agree with ``phase_updates``, the budgets and the ledger totals.

    When both ran equally far the traces MUST be equal: a different chain at equal progress means
    the arms saw different rows, and that is refused rather than reported. When they ran
    unequally far the common contract is matched but whether they saw the same rows is
    unverified (a cumulative chain has no comparable prefix), so ``same_draws`` and
    ``differs_only_in`` are ``None`` and no only-the-conditioning claim is made.
    """
    labelled = (("full", full), ("null", null))
    contracts = [_record_contract(record, label=label) for label, record in labelled]
    require(contracts[0]["arm"] == ARM_FULL and contracts[1]["arm"] == ARM_NULL,
            "the matched pair is one full and one null run")
    shared = [_plain({key: contract[key] for key in COMMON_KEYS}) for contract in contracts]
    if shared[0] != shared[1]:
        changed = contract_differences(shared[0], shared[1])
        require(False,
                f"the full and null energy runs do not share a common contract ({changed} differ), "
                "so they are not a matched control")
    (full_updates, full_trace), (null_updates, null_trace) = [
        _record_progress(record, contract, label=label)
        for (label, record), contract in zip(labelled, contracts)]
    matched = {"common_sha256": contracts[0]["common_sha256"], "common_contract_matched": True,
               "phase_updates": {"full": full_updates, "null": null_updates}}
    if full_updates != null_updates:
        return {**matched, "same_draws": None, "exposure_matched": None, "differs_only_in": None,
                "claim": (f"the common contract is matched, but the arms ran {full_updates} and "
                          f"{null_updates} steps, so whether they saw the same rows is unverified; "
                          "no claim that they differ only in the conditioning is made")}
    require(full_trace.state() == null_trace.state(),
            f"the full and null energy runs are at the same progress ({full_updates}) but hold "
            f"different draws (chains {full_trace.chain} and {null_trace.chain}), so they did not "
            "see the same rows and are not a matched control")
    return {**matched, "same_draws": True, "exposure_matched": True,
            "differs_only_in": "conditioning (context given or None)",
            "claim": ("the contracts share every common field and the traces are equal at equal "
                      "progress: the arms saw the same ordered draws and differ only in the "
                      "conditioning")}


# ---------------------------------------------------------------------------
# the exposure trace
# ---------------------------------------------------------------------------

class ExposureTrace:
    """A cumulative, ordered record of what each step presented, as a hash chain with counts.

    One entry per optimizer step: the phase, the phase-local step, the family, the group, the
    tile, the member indices, the full identities and the sites. The chain makes the ORDER part of
    the evidence; the per-phase counts say how far each phase got. It is saved in every checkpoint
    and continued on resume -- a resume that reset it would show a different history from the
    uninterrupted run while every weight check passed.
    """

    def __init__(self):
        self.chain = hashlib.sha256(TRACE_SCHEMA.encode("ascii")).hexdigest()
        self.phases = {name: {"steps": 0, "rows": 0, "site_exposures": 0} for name in PHASES}

    def record(self, *, phase, step, family, group, tile, sites, members, identities):
        require(phase in PHASES, f"Unknown energy phase {phase!r}")
        counts = self.phases[phase]
        require(int(step) == counts["steps"] + 1,
                f"the exposure trace expected {phase} step {counts['steps'] + 1}, not {step}: it "
                "is cumulative and is neither reset nor skipped on resume")
        site_list = [[str(chain), int(index)] for chain, index in sites]
        members = [int(index) for index in members]
        identities = [str(item) for item in identities]
        require(len(members) == len(identities),
                f"{len(members)} member indices for {len(identities)} identities")
        line = json.dumps([phase, int(step), str(family), int(group), int(tile), site_list,
                           members, identities], separators=(",", ":"), allow_nan=False)
        self.chain = hashlib.sha256((self.chain + "\n" + line).encode("utf-8")).hexdigest()
        counts["steps"] += 1
        counts["rows"] += len(identities)
        counts["site_exposures"] += len(identities) * len(site_list)
        return self.chain

    def state(self):
        """Plain data a checkpoint stores and :meth:`from_state` restores exactly."""
        return {"schema": TRACE_SCHEMA, "chain_sha256": self.chain,
                "phases": {name: dict(self.phases[name]) for name in PHASES}}

    def document(self):
        state = self.state()
        state["total"] = {key: sum(self.phases[name][key] for name in PHASES)
                          for key in ("steps", "rows", "site_exposures")}
        state["rule"] = ("a cumulative hash chain over every step's family, group, tile, member "
                         "indices, identities and sites, in order, with per-phase counts")
        return state

    @classmethod
    def from_state(cls, state, *, path="the checkpoint"):
        require(isinstance(state, dict) and state.get("schema") == TRACE_SCHEMA,
                f"{path}: the exposure trace is not a {TRACE_SCHEMA} record")
        chain = state.get("chain_sha256")
        require(isinstance(chain, str) and len(chain) == 64
                and all(character in "0123456789abcdef" for character in chain),
                f"{path}: the exposure trace's chain is not a sha256")
        phases = state.get("phases")
        require(isinstance(phases, dict) and set(phases) == set(PHASES),
                f"{path}: the exposure trace must count exactly the phases {list(PHASES)}")
        trace = cls()
        trace.chain = chain
        for name in PHASES:
            counts = phases[name]
            require(isinstance(counts, dict) and set(counts) == {"steps", "rows", "site_exposures"},
                    f"{path}: the {name} trace counts are malformed")
            for key, value in counts.items():
                require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                        f"{path}: the {name} trace count {key} is {value!r}, not a non-negative "
                        "integer")
            trace.phases[name] = {key: int(counts[key]) for key in ("steps", "rows",
                                                                    "site_exposures")}
        return trace


# ---------------------------------------------------------------------------
# the runtime mode a loaded model carries
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EnergyRuntime:
    """The validated, immutable mode of a loaded energy model.

    ``source`` is ``checkpoint`` (the mode was read from a validated contract) or
    ``checkpoint_free`` (a randomly initialised model with an explicitly stated mode). It is
    bound to the model once, by ``WalkJumpModel.bind_runtime``, so a ``model=`` reuse path keeps it.
    """

    arm: str
    mode: str
    source: str
    initialization: str
    contract_sha256: object = None
    contract_version: object = None
    checkpoint: object = None
    step: object = None
    terminal: object = None

    def __post_init__(self):
        require(self.arm in ARM_MODE and ARM_MODE[self.arm] == self.mode,
                f"the energy arm {self.arm!r} and mode {self.mode!r} are inconsistent")
        require(self.source in ("checkpoint", "checkpoint_free"),
                f"Unknown runtime source {self.source!r}")

    @property
    def conditioned(self):
        return self.mode == MODE_CONDITIONED

    def document(self):
        return {"arm": self.arm, "mode": self.mode, "conditioned": self.conditioned,
                "source": self.source, "initialization": self.initialization,
                "energy_contract_sha256": self.contract_sha256,
                "energy_contract_version": self.contract_version,
                "checkpoint": self.checkpoint, "step": self.step, "terminal": self.terminal,
                "requested_quality_bin": ("selectable" if self.conditioned
                                          else "not applicable: the null arm is given no context")}


def checkpoint_runtime(stored, *, path, step, terminal):
    """The runtime of a loaded, validated checkpoint (``stored`` is :func:`stored_contract`'s)."""
    return EnergyRuntime(arm=stored["arm"], mode=stored["conditioning"]["mode"],
                         source="checkpoint", initialization="trained_checkpoint",
                         contract_sha256=stored["energy_contract_sha256"],
                         contract_version=int(stored["version"]), checkpoint=str(path),
                         step=int(step), terminal=bool(terminal))


def checkpoint_free_runtime(arm):
    """A randomly initialised energy model, with its mode STATED rather than implied.

    There is no pretrained walk-jump energy for this construct family, so "no checkpoint" is not a
    frozen zero-shot backbone as it is for the likelihood routes: it is untrained random weights.
    """
    require(arm in ARM_MODE, f"Unknown energy arm {arm!r}; the arms are {sorted(ARM_MODE)}")
    return EnergyRuntime(arm=arm, mode=ARM_MODE[arm], source="checkpoint_free",
                         initialization="random_untrained")


def require_runtime(model, *, where, arm=None, mode=None):
    """The model's validated runtime, or a refusal. An explicit ``arm``/``mode`` must agree."""
    runtime = getattr(model, "runtime", None)
    require(isinstance(runtime, EnergyRuntime),
            f"{where}: this energy model carries no validated runtime contract, so how it was "
            "trained is unknown. Its architecture (condition tables, context towers, "
            "document()['context']['conditioned']) cannot say; obtain the model through "
            "FlightServices.load_for_scoring, which reads and validates the checkpoint's "
            "energy_contract")
    if arm is not None:
        require(arm == runtime.arm,
                f"{where}: the {arm!r} arm was requested but this model is the {runtime.arm!r} arm")
    if mode is not None:
        require(mode == runtime.mode,
                f"{where}: the {mode!r} mode was requested but this model is in "
                f"{runtime.mode!r} mode")
    return runtime
