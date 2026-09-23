"""The one probability event this flight measures, and the identity of a row.

Everything downstream -- tails, exact yield, mixtures, density ratios, the
proximity purge -- is defined on the ten editable residues under the fixed
prefix, 20-way renormalized, temperature 1, fixed length. The inherited
:class:`~smallAntibodyGen.experiments.her2_policy.CorePolicy` already implements
exactly that event; this module *proves* it on the live weights rather than
inferring it from a column name, and it fixes the row identity the split
manifests and the forbidden-row guards are written in terms of.

Three failures this exists to catch, all of which look like success:

* a scorer that renormalizes over the full vocabulary while the sampler does
  not, so ``exp(sum log p)`` is not the probability of the drawn core;
* a panel built from row *positions*, which silently renumber when a frame is
  filtered, so a "forbidden" evaluation row re-enters training under a new
  index;
* a tail rate computed with a strict ``>`` threshold in one place and an
  inclusive ``>=`` in another, which moves rare-event counts without moving any
  code that looks like it is about rare events.

The third is a real inherited difference: the historical post-hoc audit counted
``drop > ln 10``. This flight counts ``drop >= ln 10`` throughout, and the two
conventions are named apart here so a report can disclose the change instead of
mixing them.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from .her2_data import CANONICAL, CORE_LENGTH, decode_cores, encode_cores
from .her2_policy import (SUM_LOG_PROBABILITY_ATOL, SUM_LOG_PROBABILITY_RTOL,
                          compare_sum_log_probabilities)
from .her2_runtime import require

#: Schema every artifact of this flight carries.
NF_SCHEMA = "her2-next-flight/1"

#: The campaign identifier. Bound into every identity record and every bank.
CAMPAIGN_ID = "her2_next_flight_20260922"

#: The single modelled event, spelled out so a later reader can compare it to the
#: code rather than to a summary of the code. The conditioning prompt is exactly
#: ``her2_data.Scaffold.prefix``: the start sentinel ``"1"`` plus ``VH[:98]``,
#: which is 99 tokens and ends ``...YYCSR``. It carries NO light chain, no FR4,
#: and not the right-hand fixed Tyr; no EOS is generated or scored. Ten editable
#: residues follow it and nothing else enters this likelihood.
PROBABILITY_EVENT = ("the ten editable core residues under the fixed conditioning prompt "
                     "(start sentinel + VH[:98] = 99 prefix tokens, ending ...YYCSR): "
                     "sum over the ten positions of log q(x_i | prompt, x_<i), with each "
                     "position renormalized over exactly the twenty canonical residues, "
                     "temperature 1, fixed length 10, no filtering and no deduplication. "
                     "No VL residue, no FR4, no right-hand fixed Tyr and no EOS token is "
                     "part of this event.")

#: The prompt, as counts a reader can check against ``her2_data``.
PROMPT_SHAPE = {"start_tokens": 1, "heavy_prefix_residues": 98, "prompt_tokens": 99,
                "editable_residues": 10, "light_chain_residues": 0,
                "right_anchor_included": False, "eos_included": False,
                "ends_with": "...YYCSR"}

#: ``ell_Q/10``. Length is fixed, so this is an order-preserving rescale of the
#: sum and is the ranking score; it is NOT the density of the core.
RANKING_SCORE = "mean_log_probability = sum_log_probability / 10"

LN10 = math.log(10.0)
LN100 = math.log(100.0)

#: Inclusive tail events, for this flight, everywhere.
TAIL_THRESHOLDS = {"tenfold": LN10, "hundredfold": LN100}
TAIL_COMPARISON = ">="
TAIL_COMPARISON_NOTE = (
    "this flight counts a tail event as drop >= threshold (inclusive). The historical "
    "post-hoc audit counted drop > threshold (strict). The two are reported side by side "
    "and are never averaged or substituted for one another; the likelihood gate's own "
    "stop rule remains the inherited STRICT D > 1 nat/sequence and is a different "
    "statistic on a different population.")

#: The gate's rule, kept verbatim from the inherited guard so the amendment above
#: cannot be read as having moved it.
GATE_STOP_RULE = "strict D > 1.0 nat per sequence, or any nonfinite score"


# ---------------------------------------------------------------------------
# row identity
# ---------------------------------------------------------------------------

def core_hash(core):
    """``sha256`` of the ten-residue core's UTF-8 bytes. The row's immutable ID.

    The published splits carry no ID column and their ``seq`` is unique per
    split (``her2_data.load_split``), so the core string *is* the identity. A
    positional index is not: filtering a frame renumbers it, and a panel stored
    as positions silently changes meaning the first time anyone reorders a CSV.
    """
    text = core if isinstance(core, str) else str(core)
    require(len(text) == CORE_LENGTH, f"A core is {CORE_LENGTH} residues, got {len(text)!r}")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def core_hashes(cores):
    """``(N,)`` array of row IDs for an iterable of core strings."""
    values = [core_hash(core) for core in cores]
    require(values, "No cores to hash")
    return np.array(values, dtype="<U64")


def index_hashes(index):
    """Row IDs for an ``(N, 10)`` canonical index matrix."""
    return core_hashes(decode_cores(index))


def hash_order(cores):
    """Positions of ``cores`` in ascending lexicographic ``sha256(core)`` order.

    The deterministic panel order. It depends only on the core strings, so it is
    reproducible from the published splits alone and is independent of the row
    order of whatever file they were read from -- which a shuffle-based draw is
    not.
    """
    digests = core_hashes(cores)
    require(np.unique(digests).size == digests.size,
            "Duplicate cores inside one hash-ordered population; the panel order would not be "
            "well defined and a duplicate core is a duplicate modelled event")
    return np.argsort(digests, kind="stable")


# ---------------------------------------------------------------------------
# rows a stage is forbidden to touch
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ForbiddenRows:
    """A named set of row IDs a loader must refuse to resolve.

    Block B's challenge trainer may not see E, and no challenge-model score on E
    may influence a challenge selection. That is enforced here, at the point
    where cores become training rows, rather than by everybody remembering to
    filter: :meth:`check` is called by the population builder and by every
    scoring entry point that claims to be selection-relevant.
    """

    label: str
    hashes: frozenset
    reason: str

    @classmethod
    def from_cores(cls, cores, *, label, reason):
        return cls(label=str(label), hashes=frozenset(core_hashes(cores).tolist()),
                   reason=str(reason))

    @classmethod
    def empty(cls, *, label="none"):
        return cls(label=str(label), hashes=frozenset(),
                   reason="no rows are forbidden for this population")

    def __len__(self):
        return len(self.hashes)

    def violations(self, cores):
        """Row IDs of ``cores`` that are inside the forbidden set."""
        return sorted(set(core_hashes(cores).tolist()) & self.hashes)

    def check(self, cores, *, where):
        offending = self.violations(cores)
        require(not offending,
                f"{where}: {len(offending)} row(s) belong to the forbidden set {self.label!r} "
                f"(first: {offending[:3]}). {self.reason}")
        return True

    def document(self):
        return {"label": self.label, "rows": len(self.hashes), "reason": self.reason,
                "identity": "sha256 of the ten-residue core's UTF-8 bytes"}


# ---------------------------------------------------------------------------
# the contract proof
# ---------------------------------------------------------------------------

def canonical_support_record():
    """The support, its order, and what the event excludes. No model needed."""
    return {"alphabet": CANONICAL, "size": len(CANONICAL), "core_length": CORE_LENGTH,
            "order": "the twenty canonical residues in the inherited CANONICAL order",
            "temperature": 1.0, "prompt": dict(PROMPT_SHAPE),
            "excluded": ("<PAD>, B, Z and the three sentinel ids are outside the support and are "
                         "excluded identically from scoring and from sampling. No forced terminal "
                         "token, EOS or right-hand fixed Tyr enters this event, and no light-chain "
                         "residue is in the prompt: the conditioning prompt is the start sentinel "
                         "plus VH[:98] and the event is the ten editable positions only. The "
                         "left anchor SR is the last two residues of that prompt; the right anchor "
                         "Y follows the core and is scored by nothing here."),
            "renormalization": ("full-vocabulary probabilities followed by canonical "
                                "renormalization would describe a different distribution; "
                                "noncanonical_mass measures exactly what this discards")}


def verify_prompt_shape(scaffold):
    """Check the CONDITIONING PROMPT against :data:`PROMPT_SHAPE` on the real scaffold.

    Metadata that says "the fixed VH/VL prefix" describes a prompt this event
    does not use. The prompt is the start sentinel plus ``VH[:98]``; the light
    chain, FR4 and the right-hand anchor ``Y`` are all outside it, and the ten
    editable residues are the whole event. This measures that rather than
    restating it.
    """
    from .her2_data import ANCHOR_RIGHT, START_TOKEN
    prefix = str(scaffold.prefix)
    observed = {
        "start_tokens": int(prefix[:1] == START_TOKEN),
        "heavy_prefix_residues": len(prefix) - 1,
        "prompt_tokens": len(prefix),
        "editable_residues": CORE_LENGTH,
        "light_chain_residues": int(str(scaffold.light) in prefix) * len(str(scaffold.light)),
        "right_anchor_included": bool(prefix.endswith(ANCHOR_RIGHT)),
        "eos_included": False,
        "ends_with": "..." + prefix[-5:]}
    differing = sorted(key for key in PROMPT_SHAPE if observed[key] != PROMPT_SHAPE[key])
    require(not differing,
            f"the conditioning prompt disagrees with the declared event in {differing}: "
            f"observed {({k: observed[k] for k in differing})}, declared "
            f"{({k: PROMPT_SHAPE[k] for k in differing})}. Every tail, yield, mixture and "
            "density ratio in this flight is defined on that event.")
    return {"record_kind": "prompt_shape", "declared": dict(PROMPT_SHAPE),
            "observed": observed, "matches": True}


def verify_probability_contract(policy, index, *, draws=64, seed, batch_size=32,
                                atol=SUM_LOG_PROBABILITY_ATOL, rtol=SUM_LOG_PROBABILITY_RTOL):
    """Prove scorer/sampler/full-teacher-forcing agreement on the live weights.

    Three comparisons, because they fail differently:

    * **sampler vs scorer** -- the drawn core is re-scored and must return the
      density the sampler recorded. This is the one that makes ``exp(ell)`` the
      probability of the draw rather than a plausible number beside it.
    * **cached-prefix vs full teacher forcing** -- the shared-prefix cache is an
      optimization; if it disagreed with ordinary teacher forcing the whole
      event would be an artifact of the cache.
    * **train mode vs eval mode** -- zero hidden/attention dropout is already
      refused by ``CorePolicy.__init__``, so this difference must be exactly
      zero. Measuring it is how that refusal is shown to be effective rather
      than merely present.

      The mode comparison calls ``sequence_log_probs`` directly, once in each
      mode. ``CorePolicy.score`` puts the model into ``eval`` itself, so routing
      the comparison through it measures eval against eval and reports zero for
      a model with live dropout -- a deliberately mode-dependent policy with a
      real one-nat difference was accepted as zero that way. The original
      training/eval state is restored afterwards; nothing here disables dropout
      to manufacture parity.
    """
    import torch

    values = np.asarray(index)
    require(values.ndim == 2 and values.shape[1] == CORE_LENGTH, "Expected an (N, 10) core block")
    cores, sampled_lp = policy.sample(int(draws), seed=int(seed), batch_size=int(batch_size))
    rescored = policy.score(cores, batch_size=int(batch_size))["sum_log_probability"]
    sampler = compare_sum_log_probabilities(rescored, sampled_lp, label="sampler versus scorer",
                                            atol=atol, rtol=rtol)

    was_training = bool(policy.model.training)
    probe = values[: min(int(batch_size), values.shape[0])]
    try:
        policy.model.eval()
        with torch.no_grad():
            cached = policy.position_log_probs(values, cached=True).double().cpu().numpy()
            full = policy.position_log_probs(values, cached=False).double().cpu().numpy()
            eval_scores = policy.sequence_log_probs(probe).double().cpu().numpy()
        policy.model.train()
        with torch.no_grad():
            train_scores = policy.sequence_log_probs(probe).double().cpu().numpy()
    finally:
        policy.model.train() if was_training else policy.model.eval()
    cache_error = float(np.max(np.abs(cached.sum(axis=1) - full.sum(axis=1))))
    mode_error = float(np.max(np.abs(np.asarray(eval_scores) - np.asarray(train_scores))))
    noncanonical = policy.noncanonical_mass(values[: min(64, values.shape[0])],
                                            batch_size=int(batch_size))
    require(mode_error == 0.0,
            f"Scoring differs between train and eval mode by {mode_error:.3e}. CorePolicy refuses "
            "nonzero hidden/attention dropout, so this must be exactly zero; a nonzero value means "
            "some other stochastic path is live and the initial-gradient identity premise is gone.")
    require(cache_error <= float(atol) + float(rtol) * float(np.abs(full.sum(axis=1)).max()),
            f"The shared-prefix cache and full teacher forcing disagree by {cache_error:.3e}")
    return {"schema_version": NF_SCHEMA, "record_kind": "probability_contract",
            "event": PROBABILITY_EVENT, "ranking_score": RANKING_SCORE,
            "support": canonical_support_record(),
            "sampler_versus_scorer": sampler,
            "cached_versus_full_teacher_forcing_max_abs": cache_error,
            "train_versus_eval_max_abs": mode_error,
            "train_versus_eval_basis": ("sequence_log_probs called once in train mode and once in "
                                        "eval mode on the same rows, restoring the original "
                                        "state. CorePolicy.score forces eval and cannot measure "
                                        "this difference."),
            "train_versus_eval_rows": int(probe.shape[0]),
            "noncanonical_mass": {"rows": int(np.asarray(noncanonical).size),
                                  "mean": float(np.mean(noncanonical)),
                                  "max": float(np.max(noncanonical))},
            "tolerances": {"atol": float(atol), "rtol": float(rtol),
                           "basis": "the inherited her2_policy sum-log-probability tolerance; it "
                                    "is not reused for gradients or parameters"},
            "draws": int(draws), "sample_seed": int(seed), "rows": int(values.shape[0])}


# ---------------------------------------------------------------------------
# tails, under one stated convention
# ---------------------------------------------------------------------------

def tail_counts(drop, *, inclusive=True, thresholds=None):
    """Counts and rates of probability-loss events, under one explicit comparison.

    ``drop`` is ``ell_P - ell_Q`` in nats per sequence. Both conventions are
    computable here so a report can show the historical strict counts beside the
    inclusive ones this flight uses; neither is a default that travels silently.
    """
    values = np.asarray(drop, dtype=np.float64)
    require(values.ndim == 1 and values.size > 0, "Expected a one-dimensional drop vector")
    require(bool(np.isfinite(values).all()),
            "A nonfinite drop is not a tail event: the comparison has left the domain where it "
            "means anything and the caller must handle that as an observation, not count it")
    table = dict(thresholds or TAIL_THRESHOLDS)
    out = {"rows": int(values.size), "comparison": ">=" if inclusive else ">",
           "convention_note": TAIL_COMPARISON_NOTE, "events": {}}
    for name, threshold in sorted(table.items()):
        mask = values >= float(threshold) if inclusive else values > float(threshold)
        count = int(mask.sum())
        out["events"][name] = {"threshold_nats": float(threshold), "count": count,
                               "rate": count / float(values.size)}
    out["max_drop"] = float(values.max())
    out["mean_drop"] = float(values.mean())
    out["fraction_below_parent"] = float((values > 0).mean())
    return out


def both_tail_conventions(drop, *, thresholds=None):
    """Inclusive and strict counts in one record, so a disclosure can cite both."""
    return {"inclusive": tail_counts(drop, inclusive=True, thresholds=thresholds),
            "strict": tail_counts(drop, inclusive=False, thresholds=thresholds),
            "primary": "inclusive",
            "why_both": ("the historical post-hoc audit used the strict comparison. Publishing "
                         "only one of them would make this flight's counts look like a change in "
                         "the model when part of it is a change in the definition.")}


def audit_cores(frames, *, label):
    """Duplicate and conflicting-label audit across the populations that feed a panel.

    ``frames`` maps a name to a frame with ``seq`` and ``class``. Duplicate cores
    inside one split are already refused by the loader; what this adds is the
    cross-split view -- the same modelled core appearing in two populations with
    two class labels would make "the label of row h" ambiguous, and a panel keyed
    by core hash would silently inherit whichever one was read last.
    """
    seen, conflicts, shared = {}, [], []
    for name, frame in sorted(dict(frames).items()):
        digests = core_hashes(frame.seq)
        classes = list(frame["class"])
        require(np.unique(digests).size == digests.size, f"{label}: duplicate cores inside {name}")
        for digest, klass in zip(digests.tolist(), classes):
            if digest in seen:
                shared.append(digest)
                if seen[digest][1] != klass:
                    conflicts.append({"row_id": digest, "first": seen[digest], "second": [name, klass]})
            else:
                seen[digest] = [name, klass]
    return {"schema_version": NF_SCHEMA, "record_kind": "core_identity_audit", "label": str(label),
            "populations": {name: int(len(frame)) for name, frame in sorted(dict(frames).items())},
            "distinct_cores": len(seen),
            "cores_in_more_than_one_population": len(shared),
            "conflicting_class_labels": conflicts,
            "rule": ("cores are identified by sha256 of their UTF-8 bytes. A core present in two "
                     "populations is reported; a core present with two different class labels is "
                     "a conflict and stops panel construction.")}


def require_no_label_conflicts(audit):
    require(not audit["conflicting_class_labels"],
            f"{len(audit['conflicting_class_labels'])} core(s) carry different class labels in "
            "different populations. The panel would inherit whichever was read last, so the "
            "construction stops here rather than choosing one.")
    return True


def cores_of(index):
    """Decode an ``(N, 10)`` index back to core strings."""
    return decode_cores(index)


def index_of(cores):
    """Encode core strings to the ``(N, 10)`` canonical index."""
    return encode_cores(cores)
