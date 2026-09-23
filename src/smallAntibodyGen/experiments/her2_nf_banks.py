"""Bank roles, the streams that draw them, and the rule about what may select.

The inherited ``her2_replay_banks.draw_bank`` hard-requires
``role in {"replay", "monitor"}``. This flight needs five roles with different
selection permissions, so the role vocabulary is defined here and the drawing
goes through the same verified primitives -- ``CorePolicy.sample``,
``CorePolicy.score``, ``compare_sum_log_probabilities``, ``require_core_block``,
``build_teacher_cache`` -- rather than through a second implementation of them.
Nothing in the historical module is edited.

The load-bearing part is not the vocabulary, it is that
``may_influence_selection`` is an object the code consults:
:func:`require_may_influence_selection` refuses a selection call on a bank that
is flagged ``False`` until a freeze record naming the checkpoints exists. A
comment saying "do not select on the audit bank" is not a mechanism; this is.

Seeds are spawned from one recorded ``SeedSequence`` lineage and then mapped
into a band disjoint from the historical guarded draw seeds
``[20260921, 20360920]``. Two banks described as independent must not share a
sampling stream because one derived literal happened to land on a historical one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import her2_replay_banks as banks_lib
from . import her2_replay as replay_lib
from .her2_data import CORE_LENGTH
from .her2_nf_contract import NF_SCHEMA, index_hashes
from .her2_policy import compare_sum_log_probabilities
from .her2_replay_streams import HISTORICAL_SEED_BAND, require_disjoint_seeds
from .her2_runtime import require
from .her2_support_paths import array_digest, sha256_text

#: This flight's seed band. Every derived draw seed lands inside it, and it sits
#: entirely above the historical guarded band, so no derived literal can collide.
NF_SEED_BASE = 770_000_000
NF_SEED_SPAN = 100_000_000

#: The root entropy of every stream this flight spawns.
SEED_ENTROPY = 20260922


def seed_lineage(purpose, *, spawn_key):
    """A recorded spawned stream: its entropy, its key, and the literal it yields.

    ``spawn_key`` is the lineage, not a magic number: two purposes differ by
    their key and are reproducible from this record alone. Distance between the
    resulting integers is not an independence argument and is not offered as one.
    """
    key = tuple(int(value) for value in spawn_key)
    sequence = np.random.SeedSequence(SEED_ENTROPY, spawn_key=key)
    state = int(sequence.generate_state(1, dtype=np.uint32)[0])
    literal = NF_SEED_BASE + (state % NF_SEED_SPAN)
    low, high = HISTORICAL_SEED_BAND
    require(not (low <= literal <= high),
            f"derived seed {literal} for {purpose!r} fell inside the historical draw-seed band")
    return {"purpose": str(purpose), "entropy": SEED_ENTROPY, "spawn_key": list(key),
            "raw_state": state, "seed": int(literal),
            "band": [NF_SEED_BASE, NF_SEED_BASE + NF_SEED_SPAN - 1],
            "rule": ("numpy SeedSequence(entropy, spawn_key) -> one uint32 -> mapped into this "
                     "flight's band, which lies entirely above the historical guarded band "
                     f"{list(HISTORICAL_SEED_BAND)}"),
            "independence_note": ("a recorded lineage. Numerical spacing between seed values is "
                                  "not an independence test.")}


def seed_table(purposes):
    """``{purpose: lineage}`` for a mapping of purpose to spawn key, checked disjoint."""
    table = {name: seed_lineage(name, spawn_key=key) for name, key in sorted(dict(purposes).items())}
    require_disjoint_seeds({name: block["seed"] for name, block in table.items()},
                           label="her2 next flight draw seeds")
    return table


#: The domains a spawn key separates. A bank is identified by ALL of them: two
#: banks that differ only in their role, or only in which policy drew them, must
#: not share a sampling stream. The integers are the first element of the spawn
#: key and never change.
STREAM_DOMAINS = {
    "replay": 1,
    "development_preservation": 3,
    "final_preservation": 4,
    "q_generation_screen": 5,
    "q_generation_finalist": 6,
    "permutation_floor": 7,
    "calibration_stream": 8,
    "contract_probe": 9,
    "profile": 10,
    "bootstrap": 11,
    "mixture_component": 12,
    "smoke": 13,
}


def text_key(text, *, words=3):
    """A reproducible integer tuple from a name, wide enough not to collide.

    ``sha256`` of the name, taken ``words`` 32-bit words at a time, rather than a
    single value modulo 100,000. A five-digit modulus over forty-odd model names
    has an appreciable birthday probability, and two "independent" banks that
    collided there would share a sampling stream while every record said they did
    not. Python's ``hash()`` is never used: it is salted per process, so a seed
    derived from it would differ between two runs of the same configuration.
    """
    digest = sha256_text(str(text))
    return tuple(int(digest[index * 8:(index + 1) * 8], 16) for index in range(int(words)))


class StreamRegistry:
    """Every spawned stream this run derived, with a collision check that is real.

    Domain separation is the design: a stream is keyed by ``(domain, role,
    regime, parent, model, repeat)`` and every component enters the spawn key.
    The registry then checks that two different descriptors never produced the
    same literal seed -- not as a proof of independence, which numerical spacing
    cannot give, but because a collision means two banks described as
    independent share a sampling stream, and that is a defect the record would
    otherwise not show.
    """

    def __init__(self):
        self.entries = {}
        self._by_seed = {}

    def stream(self, *, domain, role=None, regime=None, parent=None, model=None, repeat=0):
        require(domain in STREAM_DOMAINS,
                f"unknown stream domain {domain!r}; every stream declares one of "
                f"{sorted(STREAM_DOMAINS)}")
        descriptor = {"domain": str(domain), "role": None if role is None else str(role),
                      "regime": None if regime is None else str(regime),
                      "parent": None if parent is None else str(parent),
                      "model": None if model is None else str(model),
                      "repeat": int(repeat)}
        name = "::".join(str(value) for value in
                         (domain, role, regime, parent, model, repeat))
        if name in self.entries:
            return self.entries[name]
        spawn_key = (STREAM_DOMAINS[domain], int(repeat),
                     *text_key("::".join(str(value) for value in (role, regime, parent, model))))
        lineage = seed_lineage(name, spawn_key=spawn_key)
        lineage["descriptor"] = descriptor
        clash = self._by_seed.get(lineage["seed"])
        require(clash is None,
                f"the spawned stream for {name!r} produced the same literal seed as {clash!r}. "
                "Two streams described as independent would share a sampling law, so this is a "
                "defect rather than a coincidence to accept.")
        self._by_seed[lineage["seed"]] = name
        self.entries[name] = lineage
        return lineage

    def document(self):
        return {"schema_version": NF_SCHEMA, "record_kind": "stream_registry",
                "streams": dict(sorted(self.entries.items())),
                "count": len(self.entries),
                "domains": dict(sorted(STREAM_DOMAINS.items())),
                "separation": ("role, regime, parent lineage, model identity and repeat index all "
                               "enter the spawn key, so two banks cannot share a stream by "
                               "differing only in one of them."),
                "independence_note": ("a recorded lineage with a checked absence of collisions. "
                                      "Numerical spacing between seed values is not an "
                                      "independence test and is not offered as one.")}


# ---------------------------------------------------------------------------
# roles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BankRole:
    """One bank's population, purpose, size, and whether it may inform a choice."""

    name: str
    rows: int
    purpose: str
    may_influence_selection: bool
    note: str = ""

    def document(self):
        return {"role": self.name, "rows": int(self.rows), "purpose": self.purpose,
                "may_influence_selection": bool(self.may_influence_selection), "note": self.note}


BANK_ROLES = {
    "replay": BankRole(
        name="replay", rows=100_000, purpose="training", may_influence_selection=True,
        note="parent-drawn replay contexts, 100k per parent; uniform sampling is the default"),
    "development_preservation": BankRole(
        name="development_preservation", rows=10_000, purpose="pilot and monitor diagnostics",
        may_influence_selection=True,
        note="independent 10k per parent; this is what the bounded calibration reads"),
    "final_preservation": BankRole(
        name="final_preservation", rows=50_000, purpose="audit of frozen named checkpoints",
        may_influence_selection=False,
        note=("drawn fresh per parent AFTER the reported checkpoints are named. Using it to "
              "compare checkpoints would make it developmental, and a selection-independent "
              "claim would then need yet another bank.")),
    "q_generation_screen": BankRole(
        name="q_generation_screen", rows=10_000,
        purpose="MI / total-correlation / diversity diagnostics", may_influence_selection=False,
        note="exploratory, per policy and checkpoint"),
    "q_generation_finalist": BankRole(
        name="q_generation_finalist", rows=50_000,
        purpose="dependence robustness for prespecified finalists", may_influence_selection=False,
        note=("two INDEPENDENT banks per finalist model and per parent, drawn after the finalist "
              "freeze. Fifteen single models x two banks x 50k = 1.5 million sequences.")),
}


def require_may_influence_selection(role, *, freeze_record=None, where):
    """Refuse a selection that reads a bank which is not allowed to inform one."""
    spec = BANK_ROLES[str(role)]
    if spec.may_influence_selection:
        return True
    require(freeze_record is not None,
            f"{where}: the {role!r} bank may not influence selection. It becomes readable only "
            "after a freeze record naming the reported checkpoints exists; before that, reading it "
            "for a choice turns the audit into a development bank.")
    named = list(freeze_record.get("named_checkpoints") or [])
    require(named,
            f"{where}: the supplied freeze record names no checkpoints, so it does not establish "
            "that the choice was made before this bank was scored")
    return True


# ---------------------------------------------------------------------------
# banks
# ---------------------------------------------------------------------------

@dataclass
class NfBank:
    """One drawn bank with its role, its parent's identity and its own digests."""

    role: str
    parent_id: str
    #: The digest of the policy that ACTUALLY DREW these rows. For a replay or a
    #: preservation bank that is the parent; for a Q-generation bank it is the
    #: checkpoint being screened, which is a different model from its lineage
    #: parent. Conflating the two makes "which distribution produced this bank"
    #: unanswerable from the record.
    parent_state_sha256: str
    index: np.ndarray
    sum_log_probability: np.ndarray
    draw_seed: int
    seed_lineage: dict
    temperature: float = 1.0
    conditionals: object = None
    timings: dict = field(default_factory=dict)
    #: The lineage parent's digest, when the drawing policy is not the parent.
    lineage_parent_state_sha256: object = None

    @property
    def rows(self):
        return int(self.index.shape[0])

    def document(self):
        spec = BANK_ROLES[self.role]
        block = {"schema_version": NF_SCHEMA, "record_kind": "nf_bank",
                 "role": self.role, "role_spec": spec.document(),
                 "rows": self.rows, "core_length": CORE_LENGTH,
                 "parent_id": self.parent_id,
                 "parent_state_sha256": self.parent_state_sha256,
                 "sampled_by_state_sha256": self.parent_state_sha256,
                 "lineage_parent_state_sha256": self.lineage_parent_state_sha256,
                 "identity_note": ("sampled_by_state_sha256 is the policy that DREW these rows. "
                                   "lineage_parent_state_sha256 is the model it descends from, "
                                   "and is null when they are the same model."),
                 "draw_seed": int(self.draw_seed), "seed_lineage": dict(self.seed_lineage),
                 "temperature": float(self.temperature),
                 "index_sha256": array_digest(self.index),
                 "sum_log_probability_sha256": array_digest(self.sum_log_probability),
                 "unique_cores": int(np.unique(index_hashes(self.index)).size),
                 "mean_sum_log_probability": float(np.mean(self.sum_log_probability)),
                 "duplicates_retained": True,
                 "timings": dict(self.timings)}
        if self.conditionals is not None:
            block["conditionals"] = {
                "shape": list(np.asarray(self.conditionals).shape),
                "dtype": str(np.asarray(self.conditionals).dtype),
                "sha256": array_digest(self.conditionals),
                "content": "log q(a | prefix, x_<i) at each REALIZED prefix, all 20 residues"}
        return block


def draw_bank(policy, *, role, parent_id, parent_state_sha256, seed_lineage_record,
              rows=None, temperature=1.0, batch_size=256, retain_conditionals=False,
              conditional_batch=64, atol=None, rtol=None, lineage_parent_state_sha256=None):
    """Draw one bank and prove the sampler and the scorer agree on it.

    Parity is re-established per bank rather than assumed from a global probe:
    the recorded ``sum_log_probability`` is the temperature-1 density and every
    later tail, yield and ratio is computed from it, so a silent divergence here
    would be invisible in every downstream number.
    """
    require(role in BANK_ROLES, f"Unknown bank role {role!r}")
    spec = BANK_ROLES[role]
    count = int(rows if rows is not None else spec.rows)
    require(count > 0, "A bank needs a positive row count")
    seed = int(seed_lineage_record["seed"])
    import time
    started = time.perf_counter()
    index, sampled = policy.sample(count, seed=seed, temperature=float(temperature),
                                   batch_size=int(batch_size))
    sample_seconds = time.perf_counter() - started
    replay_lib.require_core_block(index, label=f"{role} bank")
    started = time.perf_counter()
    scored = policy.score(index, batch_size=int(batch_size))["sum_log_probability"]
    score_seconds = time.perf_counter() - started
    kwargs = {}
    if atol is not None:
        kwargs["atol"] = float(atol)
    if rtol is not None:
        kwargs["rtol"] = float(rtol)
    parity = compare_sum_log_probabilities(scored, sampled, label=f"{role} bank parity", **kwargs)
    conditionals = None
    conditional_seconds = 0.0
    if retain_conditionals:
        from .her2_nf_mixture import position_log_conditionals
        started = time.perf_counter()
        # Native log_softmax, not log(clip(softmax)): the clip floor would become
        # the recorded value of every conditional below float32's smallest
        # normal, and those are exactly the terms a total correlation is made of.
        conditionals = position_log_conditionals(policy, index,
                                                 batch_size=int(conditional_batch))
        conditional_seconds = time.perf_counter() - started
    bank = NfBank(role=role, parent_id=str(parent_id),
                  parent_state_sha256=str(parent_state_sha256), index=index,
                  sum_log_probability=np.asarray(scored, dtype=np.float64), draw_seed=seed,
                  seed_lineage=dict(seed_lineage_record), temperature=float(temperature),
                  conditionals=conditionals,
                  lineage_parent_state_sha256=lineage_parent_state_sha256,
                  timings={"sample_seconds": sample_seconds, "score_seconds": score_seconds,
                           "conditional_seconds": conditional_seconds})
    document = bank.document()
    document["parity"] = parity
    return bank, document


def save_bank(path, bank, document):
    """Persist a drawn bank's arrays beside its identity document.

    An in-memory cache is not a bank: it dies with the process, and a resumed run
    that redraws "the same" bank has drawn a different one unless every seed,
    parent and role matched exactly. Writing the arrays makes the reuse checkable
    instead of assumed.
    """
    from . import her2_support_paths as paths

    target = Path(path)
    arrays = {"core_index": np.asarray(bank.index),
              "parent_sum_log_probability": np.asarray(bank.sum_log_probability,
                                                       dtype=np.float64)}
    if bank.conditionals is not None:
        # float64, as the protocol declares. The config budgets these at float64
        # ("No diagnostic downcast", 2.4 GB of finalist conditionals) and the
        # identity document digests the float64 in-memory array, so writing
        # float32 here made the recorded sha256 describe bytes that were never
        # stored -- a digest that could not check anything -- and quantized the
        # one place this flight does fine-grained log-space arithmetic, the
        # mixture posterior weights and the total-correlation terms.
        arrays["log_conditionals"] = np.asarray(bank.conditionals, dtype=np.float64)
    paths.write_arrays(target, arrays)
    paths.write_json(target.with_suffix(".json"), dict(document))
    return {"path": str(target), "rows": int(bank.rows),
            "index_sha256": array_digest(arrays["core_index"]),
            "arrays": sorted(arrays)}


def load_bank(path, *, role, parent_id, parent_state_sha256, draw_seed=None):
    """Reload a persisted bank, refusing a wrong parent, role, policy or seed.

    Every refusal here is the same failure in a different disguise: a bank drawn
    by one policy being read as another's. That does not raise anything later --
    it produces preservation numbers, tail rates and yields that look entirely
    ordinary and describe a model nobody trained.
    """
    from . import her2_support_paths as paths

    target = Path(path)
    sidecar = target.with_suffix(".json")
    if not (target.is_file() and sidecar.is_file()):
        return None
    document = paths.read_json(sidecar)
    require(document.get("role") == str(role),
            f"{target}: this bank's role is {document.get('role')!r} and {role!r} was requested. "
            "Bank roles carry different selection permissions and are not interchangeable.")
    require(document.get("parent_id") == str(parent_id),
            f"{target}: this bank belongs to {document.get('parent_id')!r}, not {parent_id!r}")
    require(document.get("parent_state_sha256") == str(parent_state_sha256),
            f"{target}: this bank was drawn by policy state {document.get('parent_state_sha256')} "
            f"and the policy reading it is {parent_state_sha256}. The draws are from a different "
            "distribution and every number derived from them would be attributed wrongly.")
    require(draw_seed is None or int(document.get("draw_seed", -1)) == int(draw_seed),
            f"{target}: this bank was drawn at seed {document.get('draw_seed')} and "
            f"{draw_seed} was requested")
    arrays = paths.read_arrays(target)
    index = np.asarray(arrays["core_index"])
    require(array_digest(index) == document.get("index_sha256"),
            f"{target}: the stored draws no longer match the digest recorded with them")
    # The DRAWS and the SCORES are two arrays and two separate claims. Checking
    # only the index accepted a bank whose probabilities had changed under an
    # intact set of cores -- and those probabilities are the parent reference the
    # preservation terms subtract from, the numbers the bounded calibration
    # decides on, and the denominators of every audited tail rate. A bank with no
    # recorded score digest cannot be verified at all, so it is refused for the
    # same reason rather than accepted as unverifiable.
    scores = np.asarray(arrays["parent_sum_log_probability"], dtype=np.float64)
    recorded_scores = document.get("sum_log_probability_sha256")
    require(recorded_scores is not None,
            f"{target}: this bank's document records no sum_log_probability_sha256, so its "
            "cached parent scores cannot be checked against what was drawn. It is redrawn "
            "rather than reused on the strength of its core indices alone.")
    require(array_digest(scores) == recorded_scores,
            f"{target}: the cached parent scores hash {array_digest(scores)} and the document "
            f"recorded {recorded_scores}. The cores are intact, so this is a changed probability "
            "array: every preservation term, calibration decision and tail rate derived from it "
            "would describe draws that were never scored this way.")
    # Conditionals are the third array and the third claim. They are persisted at
    # the declared float64, so the document's digest is checkable here on the same
    # terms as the scores: a stored block the document does not describe, or a
    # described block that is not stored, is a bank that cannot be reused.
    conditionals = None
    recorded_conditionals = (document.get("conditionals") or {}).get("sha256")
    if "log_conditionals" in arrays:
        conditionals = np.asarray(arrays["log_conditionals"], dtype=np.float64)
        require(recorded_conditionals is not None,
                f"{target}: this bank stores log_conditionals that its document does not "
                "describe, so they cannot be checked against what was drawn.")
        require(array_digest(conditionals) == recorded_conditionals,
                f"{target}: the cached log conditionals hash {array_digest(conditionals)} and "
                f"the document recorded {recorded_conditionals}. Every mixture posterior weight "
                "and total-correlation term read from them would describe a distribution that "
                "was never evaluated.")
    else:
        require(recorded_conditionals is None,
                f"{target}: the document describes log_conditionals that this bank does not "
                "store. The block is not silently treated as absent.")
    return {"index": index,
            "parent_scores": scores,
            "log_conditionals": conditionals,
            "document": document, "reused": True}


def teacher_cache(policy, index, *, label, batch_size=256):
    """The frozen 20-way teacher targets, through the inherited verified builder."""
    return banks_lib.build_teacher_cache(policy, index, batch_size=int(batch_size), label=label)


def probe_cache(policy, probabilities, log_probabilities, index, *, rows, label, seed,
                atol=1e-4, rtol=1e-4):
    """Re-evaluate the live teacher at a probe shape and compare to the cache."""
    return banks_lib.probe_teacher_cache(policy, probabilities, log_probabilities, index,
                                         rows=banks_lib.probe_rows(int(index.shape[0]),
                                                                   count=int(rows), seed=int(seed)),
                                         label=label, atol=atol, rtol=rtol)


def bank_overlap(first, second):
    """Sequence-identity overlap between two independently sampled banks.

    Reported and retained. Rejecting a coincident identity would change the
    sampling law; a strictly disjoint evaluation is a different estimand.
    """
    return banks_lib.bank_overlap(np.asarray(first), np.asarray(second))


def banks_manifest(entries, *, campaign_id, source_snapshot_sha256, freeze_record=None):
    """The immutable bank ledger: roles, digests, parents, and selection permissions."""
    documents = {str(name): dict(block) for name, block in sorted(dict(entries).items())}
    roles = sorted({block["role"] for block in documents.values()})
    return {"schema_version": NF_SCHEMA, "record_kind": "nf_banks_manifest",
            "campaign_id": str(campaign_id),
            "source_snapshot_sha256": str(source_snapshot_sha256),
            "banks": documents, "roles_present": roles,
            "role_table": {name: spec.document() for name, spec in sorted(BANK_ROLES.items())},
            "freeze_record": dict(freeze_record) if freeze_record else None,
            "selection_rule": ("a bank whose role carries may_influence_selection=False is "
                               "refused by require_may_influence_selection until a freeze record "
                               "names the reported checkpoints."),
            "reuse_rule": ("a final parent bank is reused across policies of the SAME parent for "
                           "paired preservation comparisons, and never across different parent "
                           "distributions.")}


def adopt_external_artifact(path, *, expected_sha256, label, rows=None):
    """Record an external artifact by digest without adopting its numbers as ours.

    The historical fresh-bank arrays are prior developmental evidence. Persisting
    and hashing them is useful; presenting their tail counts as this flight's
    measurements is not, and the returned record says which one this is. The
    digest is verified against the recorded one rather than taken from the file,
    so an artifact that changed is a failure here instead of a silent input.
    """
    from pathlib import Path

    from . import her2_support_paths as paths

    target = Path(path)
    require(target.is_file(), f"{label}: the external artifact is missing at {target}")
    digest = paths.sha256_file(target)
    require(digest == expected_sha256,
            f"{label}: the artifact hashes {digest} and the record expects {expected_sha256}")
    return {"schema_version": NF_SCHEMA, "record_kind": "external_artifact",
            "label": str(label), "path": target.name, "sha256": digest,
            "bytes": int(target.stat().st_size),
            "rows": None if rows is None else int(rows),
            "status": "external reference",
            "usage": ("cited as prior developmental evidence with its own provenance. This "
                      "flight's audits are drawn fresh from new spawned streams; no number from "
                      "this artifact is reported as a measurement of this flight."),
            "limitation": ("the generating RNG seed of the historical banks was not recorded, so "
                           "they are reproducible as BYTES and cannot be redrawn from a seed.")}
