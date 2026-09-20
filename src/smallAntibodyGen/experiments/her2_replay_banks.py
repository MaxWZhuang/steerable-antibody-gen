"""Bank generation and the frozen teacher cache, after the training freeze.

Two banks per parent seed, sampled independently and never reused across their
roles:

* the **replay bank** -- 100,000 IID temperature-1 parent draws that the
  distillation term is computed on;
* the **monitoring bank** -- a fresh 10,000 IID draws, sampled independently, used
  only to *observe* preservation during a fit.

Neither is the historical audit bank. The audit's 10,000 draws already
contributed to the historical diversity assessments and to the escalation
decision that justified this screen; reusing them here would make a monitoring
number a restatement of the input rather than an observation of the output. The
historical draws keep exactly one job in this campaign: they are the inherited
reference the *relative diversity floor* is measured against at endpoints, and
they are labelled as that and hash-pinned separately.

Fixed here rather than left to convention:

* **Duplicates and accidental overlap are retained and reported.** Rejecting a
  sequence because the other bank happens to contain it changes the sampling law.
  "Separate banks" means independent sampling and no deliberate record reuse, not
  a guarantee that one sequence never appears in both.
* **Both the teacher probabilities and their logs are cached**, float32,
  ``(rows, 10, 20)``, about 80 MB each and 160 MB per parent. Storing the
  log-softmax output directly is what keeps the teacher log finite by
  construction: reconstructing ``log p`` from an underflowed ``p == 0`` would make
  ``0 * -inf`` and put a NaN on the backward path, and no epsilon is added to
  avoid it.
* **A cache is validated before it is trusted, every time.** Normalization,
  probability/log agreement, residue order, prefix order, draw order, the parent's
  identity, and equality with a *fresh* teacher evaluation on fixed probe rows.
* **Generation happens after the freeze.** The freeze binds the sampling rule, the
  seeds, the counts and the dtypes; this module produces the bytes and binds their
  completed digests back to that marker.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import her2_replay as replay_lib
from . import her2_support_paths as paths
from .her2_data import CORE_LENGTH, decode_cores, encode_cores
from .her2_policy import (SUM_LOG_PROBABILITY_ATOL, SUM_LOG_PROBABILITY_RTOL,
                          compare_sum_log_probabilities)
from .her2_runtime import require

BANK_SCHEMA = "her2-parent-replay-bank/1"

#: The two bank roles. A cache, a score vector or an order digest is always bound
#: to one of them; there is no code path that reads a monitoring row into a replay
#: batch or trains on a monitoring record.
BANK_ROLES = ("replay", "monitor")


@dataclass(frozen=True)
class Bank:
    """One generated bank: its draws, its identity and the rule that produced it."""

    role: str
    parent_seed: int
    parent_id: str
    parent_state_sha256: str
    draw_seed: int
    temperature: float
    index: np.ndarray
    sampler_sum_log_probability: np.ndarray

    @property
    def rows(self):
        return int(self.index.shape[0])

    def document(self):
        values = np.asarray(self.index)
        cores = decode_cores(values)
        counts = {}
        for core in cores:
            counts[core] = counts.get(core, 0) + 1
        return {"schema_version": BANK_SCHEMA, "role": self.role,
                "parent_seed": int(self.parent_seed), "parent_id": self.parent_id,
                "parent_state_sha256": self.parent_state_sha256,
                "draw_seed": int(self.draw_seed), "temperature": float(self.temperature),
                "rows": self.rows, "core_length": CORE_LENGTH,
                "unique_cores": len(counts),
                "duplicate_draws": self.rows - len(counts),
                "max_single_core_count": max(counts.values()),
                "order_sha256": paths.array_digest(values),
                "sampler_sum_log_probability_sha256":
                    paths.array_digest(self.sampler_sum_log_probability),
                "duplicates_retained": True,
                "probability_convention":
                    "sum_log_probability_over_10_core_positions_20way_renormalized",
                "note": ("IID draws at temperature 1 from the frozen parent. Duplicates are the "
                         "sampling law and are kept; nothing is filtered by library membership, "
                         "measured label, novelty or predicted binding.")}


def draw_bank(policy, *, role, parent_seed, parent_id, parent_state_sha256, rows, draw_seed,
              temperature=1.0, batch_size=256, atol=SUM_LOG_PROBABILITY_ATOL,
              rtol=SUM_LOG_PROBABILITY_RTOL, progress=None):
    """Sample one bank and prove the draws re-score to the density they were drawn at.

    ``CorePolicy.sample`` records the temperature-1 density regardless of the
    sampling temperature, so re-scoring the draws must reproduce it. That parity is
    the cheapest available proof that the bank on disk is a bank *from this parent*
    rather than from whatever weights happened to be loaded.
    """
    require(role in BANK_ROLES, f"Unknown bank role {role!r}; expected one of {BANK_ROLES}")
    require(int(rows) > 0, "A bank needs a positive row count")
    started = time.perf_counter()
    index, sampled = policy.sample(int(rows), seed=int(draw_seed), temperature=float(temperature),
                                   batch_size=int(batch_size))
    replay_lib.require_core_block(index, label=f"{role} bank seed {parent_seed}")
    rescored = policy.score(index, batch_size=int(batch_size), progress=progress)
    parity = compare_sum_log_probabilities(
        rescored["sum_log_probability"], sampled,
        label=f"{role} bank seed {parent_seed}: sampler versus scorer", atol=atol, rtol=rtol)
    bank = Bank(role=role, parent_seed=int(parent_seed), parent_id=str(parent_id),
                parent_state_sha256=str(parent_state_sha256), draw_seed=int(draw_seed),
                temperature=float(temperature), index=np.asarray(index),
                sampler_sum_log_probability=np.asarray(sampled, dtype=np.float64))
    return bank, {"sampler_scorer_parity": parity,
                  "rescored_sum_log_probability": rescored["sum_log_probability"],
                  "wall_seconds": time.perf_counter() - started}


def bank_identity(bank, *, campaign_id, freeze_commit):
    """Everything that would change the bytes of this bank, in one comparable block."""
    return {"schema_version": BANK_SCHEMA, "campaign_id": campaign_id,
            "freeze_commit": freeze_commit, "role": bank.role,
            "parent_seed": int(bank.parent_seed), "parent_id": bank.parent_id,
            "parent_state_sha256": bank.parent_state_sha256,
            "draw_seed": int(bank.draw_seed), "temperature": float(bank.temperature),
            "rows": bank.rows, "order_sha256": paths.array_digest(np.asarray(bank.index))}


# ---------------------------------------------------------------------------
# the frozen teacher cache
# ---------------------------------------------------------------------------

def build_teacher_cache(policy, index, *, batch_size=256, label, progress=None):
    """``(probabilities, log_probabilities)`` float32 ``(N, 10, 20)`` from the parent.

    The teacher forward is float32 -- that is the computation being distilled --
    and the *stored* log is the log-softmax output itself, not ``log(p)``. That
    single choice is what makes the zero convention exact rather than defensive:
    ``logit - logsumexp`` is finite for finite logits, so no stored value is
    ``-inf`` and no product can become ``0 * -inf``.
    """
    import torch
    values = replay_lib.require_core_block(index, label=label)
    rows = int(values.shape[0])
    probabilities = np.empty((rows, CORE_LENGTH, replay_lib.TEACHER_WIDTH), dtype=np.float32)
    logs = np.empty_like(probabilities)
    was_training = bool(getattr(policy.model, "training", False))
    policy.model.eval()
    worst_sum = 0.0
    try:
        with torch.inference_mode():
            for start in range(0, rows, int(batch_size)):
                chunk = values[start:start + int(batch_size)]
                core_ids = policy.token_ids(chunk)
                logits = policy.core_logits(core_ids).float()
                require(bool(torch.isfinite(logits).all()),
                        f"{label}: the teacher produced a nonfinite logit at row offset {start}. "
                        "The cache fails here; no value is capped and no row is dropped.")
                log_probabilities = torch.log_softmax(logits, dim=-1)
                block = log_probabilities.exp()
                worst_sum = max(worst_sum,
                                float((block.double().sum(dim=-1) - 1.0).abs().max()))
                probabilities[start:start + len(chunk)] = block.cpu().numpy()
                logs[start:start + len(chunk)] = log_probabilities.cpu().numpy()
                if progress is not None:
                    progress.heartbeat(f"{label} {min(start + len(chunk), rows)}/{rows}")
    finally:
        if was_training:
            policy.model.train()
    return probabilities, logs, {"rows": rows, "batch_size": int(batch_size),
                                 "max_abs_probability_sum_error": worst_sum,
                                 "forward_dtype": "float32", "storage_dtype": "float32",
                                 "bytes_per_array": int(probabilities.nbytes),
                                 "arrays": ["probabilities", "log_probabilities"],
                                 "storage_note": ("both views are stored: the probabilities are "
                                                  "the distillation weights and the logs keep the "
                                                  "teacher term finite by construction. About "
                                                  "80 MB each, 160 MB per parent.")}


def cached_sequence_log_probability(log_probabilities, index):
    """``sum_t log p0(y_t | y_<t)`` read straight out of the cache, in float64.

    A free and unusually direct check: this must reproduce the sum log probability
    the sampler recorded for the same draw. If the cache were transposed, reordered
    against the draws, or built from other weights, the two would differ by nats.
    """
    values = replay_lib.require_core_block(index, label="cached sequence log probability")
    logs = np.asarray(log_probabilities, dtype=np.float64)
    require(logs.shape == (values.shape[0], CORE_LENGTH, replay_lib.TEACHER_WIDTH),
            f"The cache is {logs.shape}; expected "
            f"({values.shape[0]}, {CORE_LENGTH}, {replay_lib.TEACHER_WIDTH})")
    rows = np.arange(values.shape[0])[:, None]
    positions = np.arange(CORE_LENGTH)[None, :]
    return logs[rows, positions, values].sum(axis=1)


def validate_teacher_cache(probabilities, log_probabilities, index, *, sampler_sum_log_probability,
                           label, atol=SUM_LOG_PROBABILITY_ATOL, rtol=SUM_LOG_PROBABILITY_RTOL):
    """Normalization, log agreement, order and the sampler cross-check, in one report."""
    import torch
    rows = int(np.asarray(index).shape[0])
    block = replay_lib.require_teacher_targets(
        torch.from_numpy(np.asarray(probabilities)),
        torch.from_numpy(np.asarray(log_probabilities)), rows=rows, label=label)
    derived = cached_sequence_log_probability(log_probabilities, index)
    block["sampler_cross_check"] = compare_sum_log_probabilities(
        derived, np.asarray(sampler_sum_log_probability, dtype=np.float64),
        label=f"{label}: cached teacher logs versus the sampler's recorded density",
        atol=float(atol), rtol=float(rtol))
    block["residue_order"] = "canonical ACDEFGHIKLMNPQRSTVWY, the training and sampling order"
    block["prefix_order"] = ("position t is conditioned on the realized residues y_<t of the same "
                             "draw; the cache is aligned to the bank's draw order 0..N-1")
    return block


def probe_teacher_cache(policy, probabilities, log_probabilities, index, *, rows, label,
                        atol, rtol):
    """Re-evaluate the live teacher on fixed rows and compare to the cached block.

    The identity check proves the cache claims to come from this parent; this
    proves it does. Bounded to a few rows because it is a spot check on a
    deterministic function, not a second cache build.
    """
    import torch
    probe = np.asarray(rows, dtype=np.int64)
    require(probe.size > 0, f"{label}: a cache probe needs at least one row")
    fresh = replay_lib.teacher_log_probabilities(policy, np.asarray(index)[probe])
    cached = torch.from_numpy(np.asarray(log_probabilities)[probe]).to(fresh.device)
    report = replay_lib.compare_vectors(
        fresh.double().cpu().numpy().reshape(-1), cached.double().cpu().numpy().reshape(-1),
        atol=float(atol), rtol=float(rtol), label=f"{label}: live teacher versus cached logs")
    fresh_probabilities = fresh.exp().double().cpu().numpy().reshape(-1)
    report["probability_comparison"] = replay_lib.compare_vectors(
        fresh_probabilities, np.asarray(probabilities)[probe].astype(np.float64).reshape(-1),
        atol=float(atol), rtol=float(rtol),
        label=f"{label}: live teacher versus cached probabilities")
    report["probe_rows"] = probe.tolist()
    return report


def probe_rows(rows, *, count, seed):
    """A fixed, reproducible row sample for the cache probes. Seeded, never ``hash()``."""
    size = min(int(count), int(rows))
    return np.sort(np.random.default_rng([int(seed), int(rows)]).choice(
        int(rows), size=size, replace=False))


# ---------------------------------------------------------------------------
# overlap, reported and never removed
# ---------------------------------------------------------------------------

def bank_overlap(replay_index, monitor_index):
    """Accidental sequence-identity overlap between two independently sampled banks.

    Reported, never acted on. Removing a coincident identity would change the
    conditional law of the monitoring bank, and a strict identity-disjoint
    evaluation is a different estimand that would need its own declaration.
    """
    replay_cores = set(decode_cores(np.asarray(replay_index)))
    monitor_cores = decode_cores(np.asarray(monitor_index))
    shared = sorted(replay_cores.intersection(monitor_cores))
    draws = sum(1 for core in monitor_cores if core in replay_cores)
    return {"replay_unique_cores": len(replay_cores),
            "monitor_rows": len(monitor_cores),
            "monitor_unique_cores": len(set(monitor_cores)),
            "shared_unique_cores": len(shared),
            "monitor_draws_also_in_replay": int(draws),
            "monitor_draw_fraction_also_in_replay": draws / max(1, len(monitor_cores)),
            "action": "none",
            "note": ("'separate banks' means independent sampling and no deliberate record reuse. "
                     "Coincident identities are expected at these counts and are retained: "
                     "rejecting them would change the sampling law.")}


def historical_reference_document(path, *, expected_sha256, expected_rows, label):
    """The inherited parent draw bank behind the *relative diversity floor*.

    Pinned separately and labelled apart from the fresh monitoring bank on purpose.
    Endpoint diversity verdicts stay comparable with the completed campaign only if
    they are measured against the same reference it used, and the fresh monitoring
    bank -- which this campaign selects diagnostics on -- must never quietly become
    that reference.
    """
    import pandas as pd
    target = Path(path)
    require(target.is_file(), f"{label}: the inherited parent draw bank is missing at {target}")
    digest = paths.sha256_file(target)
    require(digest == expected_sha256,
            f"{label}: the inherited parent draws hash {digest}, the published audit input "
            f"manifest recorded {expected_sha256}")
    frame = pd.read_csv(target, dtype={"core": str}, keep_default_na=False)
    require(list(frame.columns) == ["draw_index", "core"],
            f"{label}: columns are {list(frame.columns)}, expected draw_index,core")
    require(len(frame) == int(expected_rows),
            f"{label}: {len(frame)} rows, expected {expected_rows}")
    require(bool((frame.draw_index.to_numpy() == np.arange(len(frame))).all()),
            f"{label}: draw_index is not the exact 0..N-1 order")
    index = encode_cores(frame.core)
    return index, {"role": "inherited_diversity_reference", "logical_path": str(label),
                   "sha256": digest, "rows": int(len(frame)),
                   "order_sha256": paths.array_digest(index),
                   "use": ("the relative diversity floor at endpoints, exactly as the completed "
                           "guarded campaign measured it"),
                   "not_used_for": ("preservation diagnostics, which are measured on the freshly "
                                    "generated monitoring bank alone")}


# ---------------------------------------------------------------------------
# the banks manifest: generated after the freeze, bound to it
# ---------------------------------------------------------------------------

def banks_manifest(entries, *, campaign_id, freeze_commit, freeze_sha256, overlap, timings):
    """The completion record that binds generated bank bytes to the training freeze."""
    return {"schema_version": BANK_SCHEMA, "record_kind": "banks_manifest",
            "campaign_id": campaign_id,
            "freeze": {"commit": freeze_commit, "marker_sha256": freeze_sha256},
            "banks": {str(key): dict(value) for key, value in sorted(dict(entries).items())},
            "bank_count": len(dict(entries)),
            "overlap": {str(key): dict(value) for key, value in sorted(dict(overlap).items())},
            "timings": dict(timings),
            "ordering_note": ("the freeze bound the sampling rule, the seeds, the counts and the "
                              "dtypes before these bytes existed. This manifest records what was "
                              "then produced and is bound to that marker; it cannot change the "
                              "rule it was produced under."),
            "generated_at": paths.utc_now()}


def require_banks_bound(manifest, *, freeze_commit, freeze_sha256):
    """A banks manifest is usable only under the marker it was generated against."""
    freeze = dict(manifest.get("freeze") or {})
    require(freeze.get("commit") == freeze_commit and freeze.get("marker_sha256") == freeze_sha256,
            f"The banks manifest was generated under freeze {freeze.get('commit')} "
            f"({freeze.get('marker_sha256')}) and this run holds {freeze_commit} "
            f"({freeze_sha256}). Banks are bound to the specification that declared their sampling "
            "rule; they are not carried across a re-freeze.")
    return True


#: Manifest key suffix for the identity document written beside an array. The
#: sidecar is what makes the array readable -- ``load_parent_reference`` and
#: ``load_reference_cache`` refuse to load without it, and ``read_shard`` refuses a
#: container whose ``.complete.json`` is absent -- so it is an artifact of the banks
#: stage in its own right and is bound like one.
SIDECAR_SUFFIX = "_sidecar"
#: Manifest key suffix for a shard's completion record beside its container.
SHARD_RECORD_SUFFIX = "_record"


def expected_bank_keys(seeds, *, roles=BANK_ROLES, include_ipo_reference=True):
    """Every artifact key the banks stage must have produced, from the declared grid.

    Derived from the declaration rather than from what is on disk. A scan of the
    surviving files can only enumerate what is still there, which is exactly the
    wrong instrument for finding a deleted one.

    The JSON sidecars are here beside their arrays. They were previously omitted,
    which left :func:`verify_bank_artifacts` saying it had checked "every role,
    seed, bank container and reference-cache sidecar" while no manifest entry named
    a sidecar at all -- and a deleted or edited identity document is exactly what
    that claim was supposed to cover.
    """
    keys = []
    for seed in sorted(int(value) for value in seeds):
        keys.append(f"seed{seed}::parent_validation_reference")
        keys.append(f"seed{seed}::parent_validation_reference{SIDECAR_SUFFIX}")
        if include_ipo_reference:
            keys.append(f"seed{seed}::ipo_reference_cache")
            keys.append(f"seed{seed}::ipo_reference_cache{SIDECAR_SUFFIX}")
        for role in roles:
            keys.append(f"seed{seed}::{role}_bank")
            keys.append(f"seed{seed}::{role}_bank{SHARD_RECORD_SUFFIX}")
    return sorted(keys)


def verify_bank_artifacts(manifest, *, resolve, expected_keys=(), label="banks"):
    """Re-hash every artifact the completed banks manifest recorded, before any fit.

    The manifest is the saved authority. ``read_shard`` already proves a shard is
    internally consistent -- its arrays reproduce the digests recorded *in its own
    sidecar* -- and an internally consistent rewrite is exactly what that check
    cannot see: regenerate the cache, regenerate the sidecar beside it, and the
    shard validates itself perfectly while being a different bank. Comparing the
    container bytes to the digest the **completed manifest** recorded is what makes
    that fail.

    Coverage is checked against ``expected_keys``, derived from the declared grid,
    so a deleted artifact is a missing key rather than one fewer row in a scan.
    Sidecars are entries like any other: this function can only check what the
    manifest names, so "the sidecars were checked" is true exactly when they are in
    it, which is why :func:`expected_bank_keys` requires them.
    """
    entries = dict(manifest.get("banks") or {})
    problems, checked = [], []
    for key in sorted(set(expected_keys) - set(entries)):
        problems.append({"artifact": key,
                         "problem": ("the declared grid requires this bank artifact and the "
                                     "completed banks manifest never recorded it")})
    for key, entry in sorted(entries.items()):
        target = resolve(entry["file"])
        if target is None or not Path(target).is_file():
            problems.append({"artifact": key,
                             "problem": f"{entry['file']} is recorded in the banks manifest and is "
                                        "absent from the run directory"})
            continue
        observed = paths.sha256_file(target)
        row = {"artifact": key, "file": entry["file"], "role": entry.get("role"),
               "recorded_sha256": entry.get("sha256"), "observed_sha256": observed,
               "matches": observed == entry.get("sha256")}
        if not row["matches"]:
            problems.append({"artifact": key,
                             "problem": ("bytes differ from the completed banks manifest. A "
                                         "coherently rewritten cache and sidecar still fail here, "
                                         "because the authority is the manifest and not the file "
                                         "checking itself.")})
        checked.append(row)
    require(not problems,
            f"{label}: the generated banks no longer match their completed manifest: "
            + "; ".join(f"{problem['artifact']}: {problem['problem']}" for problem in problems)
            + ". Fitting against a bank that is not the one the manifest bound to the freeze would "
              "attribute the fit to a sampling rule it did not run under.")
    return {"artifacts_checked": len(checked), "artifacts": checked,
            "expected_keys": sorted(set(expected_keys)),
            "basis": ("every role, seed, bank container and reference-cache sidecar re-hashed "
                      "against the completed banks manifest, once, before the first update")}
