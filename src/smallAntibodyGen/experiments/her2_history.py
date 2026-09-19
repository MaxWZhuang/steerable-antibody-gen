"""Recover what the original six trajectories actually recorded, and say what they did not.

The original campaign wrote one history entry per update for six runs (DPO and
continued SFT at three seeds). This module reads those summaries, hashes them,
and recomputes every finding from the bytes rather than restating a number from a
report. Nothing here writes anything under the original output root.

Two corrections to how the original fields read:

* **``learning_rate`` in the original history is the NEXT update's rate.** The old
  loop calls ``scheduler.step()`` inside the timed segment and then reads
  ``scheduler.get_last_lr()``, so the recorded value is the rate the *following*
  update will use, not the one this update ran at. It is relabelled
  ``learning_rate_next`` here; the guarded loop logs both.
* **The per-pair coefficient distribution is not recoverable.** The histories
  stored ``mean_margin`` per update and nothing else about the margin's shape.
  The DPO coefficient is ``beta * sigmoid(-beta * Delta)`` and sigma is nonlinear,
  so ``E[beta sigma(-beta Delta)] != beta sigma(-beta E[Delta])``. This module
  therefore emits a ``missing_diagnostics`` record instead of a function that
  reconstructs it; there is no API here that returns a fabricated distribution.

What *is* recoverable, and is recomputed here when the inputs are present, is the
**training batch-mean chosen drop**: the original pairing is deterministic, the
frozen reference cache is on disk with its digest, and each history entry records
its cycle and batch number. So the reference's mean over exactly the rows that
update trained on can be recomputed and differenced against the recorded
``mean_policy_chosen``. That is a batch mean over changing rows -- it is not the
fixed-validation gate measurement the original never took, and it is not a
per-pair distribution.

Two things that reconstruction depends on are **verified, not assumed**:

* the batch size comes from that run's own ``summary.json``
  (:func:`recorded_batch_size`), never from the current config, and every history
  entry's recorded pair count is checked against the reconstructed slice;
* the reference cache is checked against the identity the run recorded for it
  (:func:`verify_reference_provenance`) -- sidecar identity, value digest, row
  count and the parent checkpoint it was produced from -- so the numbers are that
  trajectory's frozen reference rather than an array that happens to be at that
  path.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .her2_objectives import CORE_POSITIONS
from .her2_preferences import array_digest
from .her2_runtime import load_json, require, sha256

HISTORY_REPORT_SCHEMA = "her2-original-history/1"

#: The windows the original campaign is summarized over. Declared here so two
#: reports cannot quietly average different ranges.
DEFAULT_WINDOWS = ((1, 10), (11, 25), (26, 50), (51, 100), (101, 500))
LAST_WINDOW = 100

#: Fields worth a window mean, by arm. Absent fields are reported absent.
#: ``accuracy`` is the original loop's batch sign accuracy -- the fraction of
#: pairs in that batch with a positive implicit-reward margin -- and ``loss`` is
#: the batch mean the optimizer stepped on. Every one of these is a per-batch
#: scalar; averaging them over a window gives an average of batch means and
#: nothing about the shape of any per-pair distribution.
DPO_FIELDS = ("loss", "mean_margin", "accuracy", "mean_policy_chosen", "mean_policy_rejected",
              "gradient_norm", "learning_rate")
SFT_FIELDS = ("loss", "nll_per_residue", "gradient_norm", "learning_rate")

#: The original field name on the left, what it actually is on the right. The
#: recorded ``learning_rate`` is the NEXT update's rate (the loop steps the
#: scheduler before reading it), so it is reported under a name that says so.
FIELD_LABELS = {"learning_rate": "learning_rate_next"}

LEARNING_RATE_NOTE = ("the original history's `learning_rate` is the rate for the NEXT update: "
                      "the loop steps the scheduler before reading it. Relabelled here; the "
                      "guarded loop records learning_rate_used and learning_rate_next separately.")

MISSING_DIAGNOSTICS = {
    "per_pair_coefficient_distribution": {
        "recoverable": False,
        "stored": "mean_margin only, one scalar per update",
        "why_not": ("the actual DPO coefficient is beta * sigmoid(-beta * Delta) per pair and "
                    "sigmoid is nonlinear, so E[beta sigma(-beta Delta)] != "
                    "beta sigma(-beta E[Delta]). Two batches with the same mean margin can have "
                    "materially different mean coefficients."),
        "not_done": "no distribution is reconstructed from the mean, and none is reported"},
    "fixed_validation_likelihood_before_180s": {
        "recoverable": False,
        "stored": "the first fixed-validation measurement is at the 180 s checkpoint",
        "why_not": ("the original loop evaluated validation only at budget crossings, so there is "
                    "no earlier fixed-validation chosen likelihood to recover. The batch-mean "
                    "reconstruction below is a different quantity on changing rows."),
        "not_done": "the onset of the chosen-likelihood decline is bracketed, not dated"},
}


def read_summary(path):
    """One original ``summary.json`` with its content hash. Opened read-only."""
    path = Path(path)
    require(path.is_file(), f"No original summary at {path}")
    return {"path": str(path), "sha256": sha256(path), "bytes": int(path.stat().st_size),
            "document": load_json(path)}


def window_statistics(history, field, *, start, end):
    """Mean/min/max of one field over the inclusive update window ``[start, end]``."""
    rows = [entry for entry in history
            if start <= int(entry["update"]) <= end and field in entry]
    if not rows:
        return {"updates": 0, "note": "window empty or field absent"}
    values = np.array([float(entry[field]) for entry in rows], dtype=np.float64)
    return {"updates": int(values.size), "start": int(start), "end": int(end),
            "mean": float(values.mean()), "min": float(values.min()), "max": float(values.max())}


def field_statistics(history, fields, *, start, end):
    """Every field's window statistics over ``[start, end]``, under its honest label."""
    return {FIELD_LABELS.get(field, field): window_statistics(history, field, start=start, end=end)
            for field in fields}


def window_summary(history, fields, *, windows=DEFAULT_WINDOWS, last=LAST_WINDOW):
    """Every declared window plus the final ``last`` updates, per field."""
    total = max(int(entry["update"]) for entry in history)
    document = {}
    for start, end in windows:
        document[f"{start}-{end}"] = field_statistics(history, fields, start=start, end=end)
    document[f"last_{last}"] = field_statistics(history, fields, start=max(1, total - last + 1),
                                                end=total)
    return document


def cycle_boundaries(history):
    """Where each pass over the chosen population began and ended.

    A cycle is one visit to every eligible chosen row. The boundary matters
    because the partner rotation advances there, so batch composition changes for
    a structural reason rather than by chance.
    """
    boundaries = {}
    for entry in history:
        cycle = entry.get("cycle")
        if cycle is None:
            continue
        record = boundaries.setdefault(str(int(cycle)), {
            "first_update": int(entry["update"]), "last_update": int(entry["update"]),
            "updates": 0, "first_cumulative_gpu_seconds": entry.get("cumulative_gpu_seconds"),
            "last_cumulative_gpu_seconds": entry.get("cumulative_gpu_seconds")})
        record["last_update"] = int(entry["update"])
        record["last_cumulative_gpu_seconds"] = entry.get("cumulative_gpu_seconds")
        record["updates"] += 1
    return boundaries


def cycle_summary(history, fields):
    """Cycle boundaries plus the regime inside each cycle, field by field.

    A cycle is the natural window here: the partner rotation advances at its
    boundary, so batch composition changes for a structural reason. What is
    reported per cycle is the same batch-mean aggregate the windows report --
    loss, margin, sign accuracy, both sides' mean log probability, gradient norm
    and the next-update learning rate -- plus the first and last recorded
    next-update rate, which is where the warmup shows up.

    These are aggregates of per-batch scalars. No per-pair distribution is
    reconstructed from them, because none is recoverable; see
    :data:`MISSING_DIAGNOSTICS`.
    """
    boundaries = cycle_boundaries(history)
    for name, record in boundaries.items():
        start, end = int(record["first_update"]), int(record["last_update"])
        rows = [entry for entry in history if start <= int(entry["update"]) <= end]
        record["metrics"] = field_statistics(history, fields, start=start, end=end)
        record["learning_rate_next"] = {
            "first": rows[0].get("learning_rate") if rows else None,
            "last": rows[-1].get("learning_rate") if rows else None,
            "note": LEARNING_RATE_NOTE}
        record["aggregate_note"] = ("means over the per-batch scalars this cycle recorded; the "
                                    "batches change from cycle to cycle, so this describes a "
                                    "regime and not a fixed population")
    return boundaries


def reconstruct_batch_chosen_drop(history, reference_values, *, chosen_count, pairing_seed,
                                  batch_pairs):
    """Recompute the TRAINING batch-mean chosen drop from the frozen reference.

    The original pairing is deterministic: cycle ``c`` visits
    ``default_rng([pairing_seed, c]).permutation(chosen_count)`` in chunks of
    ``batch_pairs``, and each history entry records its cycle and batch number. So
    the reference's mean over exactly those rows is recomputable, and

    ``drop = mean(reference_chosen) - mean(policy_chosen)``

    is the amount of chosen log probability that update had given up relative to
    the frozen parent, **on the rows it trained on**. Positive is worse.

    This is a batch mean over changing rows. It is not the fixed-validation
    measurement the gate makes, it is not a per-pair distribution, and a window
    mean of it is not an onset date.

    ``batch_pairs`` must be the batch size **that run** trained at. Every history
    entry that records its own pair count is checked against the reconstructed
    slice, so reconstructing with the wrong batch size -- the current config's
    default, say, instead of the recorded one -- raises here instead of quietly
    averaging the reference over rows that update never saw.
    """
    values = np.asarray(reference_values, dtype=np.float64)
    require(values.size >= chosen_count,
            f"The reference cache has {values.size} rows, fewer than the {chosen_count} chosen "
            "rows the pairing indexes")
    require(int(batch_pairs) > 0, "The reconstruction needs the original batch size in pairs")
    orders, rows = {}, []
    for entry in history:
        if "cycle" not in entry or "batch" not in entry or "mean_policy_chosen" not in entry:
            continue
        cycle = int(entry["cycle"])
        if cycle not in orders:
            orders[cycle] = np.random.default_rng([int(pairing_seed), cycle]).permutation(
                chosen_count)
        start = int(entry["batch"]) * int(batch_pairs)
        index = orders[cycle][start:start + int(batch_pairs)]
        if "pairs" in entry:
            require(int(entry["pairs"]) == int(index.size),
                    f"update {entry['update']}: the history recorded {entry['pairs']} pairs in "
                    f"this batch but reconstructing at {batch_pairs} pairs per batch selects "
                    f"{index.size} rows. The batch size being reconstructed with is not the one "
                    "that run trained at, so the reference means would be taken over rows that "
                    "update never saw.")
        if index.size == 0:
            continue
        reference_mean = float(values[index].mean())
        rows.append({"update": int(entry["update"]), "cycle": cycle, "batch": int(entry["batch"]),
                     "pairs": int(index.size),
                     "reference_mean_chosen": reference_mean,
                     "policy_mean_chosen": float(entry["mean_policy_chosen"]),
                     "batch_mean_chosen_drop": reference_mean - float(entry["mean_policy_chosen"])})
    return rows


def summarize_batch_chosen_drop(rows, *, windows=DEFAULT_WINDOWS, threshold=1.0):
    """Window means of the reconstructed drop, and the first update above ``threshold``."""
    if not rows:
        return {"available": False, "reason": "no reconstructable updates"}
    drops = {row["update"]: row["batch_mean_chosen_drop"] for row in rows}
    document = {"available": True, "updates": len(rows),
                "windows": {}, "threshold_nats_per_sequence": float(threshold)}
    for start, end in windows:
        values = [value for update, value in drops.items() if start <= update <= end]
        document["windows"][f"{start}-{end}"] = (
            {"updates": len(values), "mean_batch_chosen_drop": float(np.mean(values))}
            if values else {"updates": 0})
    above = sorted(update for update, value in drops.items() if value > threshold)
    document["first_update_above_threshold"] = above[0] if above else None
    document["note"] = ("batch means over changing training rows, reconstructed from the frozen "
                        "reference and the deterministic pairing. This is not the fixed-validation "
                        "monitor and does not date the first crossing of any gate.")
    return document


def recorded_batch_size(path, summary, *, batch_pairs=None, paired=True):
    """The batch size THAT run recorded, in sequences and in pairs. Not a config default.

    The original campaign resolved its batch size once, from a preflight decision,
    and wrote it into its own summary. A reconstruction that takes the number from
    the current config instead is reconstructing a different run whenever the two
    disagree -- and it disagrees silently, because both numbers are plausible.

    ``paired=False`` is the continued-SFT case: those runs recorded the same
    ``batch_sequences`` and trained on chosen positives only, so the sequence count
    is reported and pairs are stated to be inapplicable rather than left null, which
    read as "this run recorded no batch size".
    """
    sequences = summary.get("batch_sequences")
    require(sequences,
            f"{path} does not record the batch size it trained at. The batch composition cannot "
            "be reconstructed without it, and assuming one would be an invention.")
    sequences = int(sequences)
    require(sequences > 0, f"{path}: a batch needs a positive sequence count, got {sequences}")
    if not paired:
        return {"batch_sequences": sequences, "batch_pairs": None,
                "pairs_applicable": False,
                "pairs_note": ("continued SFT trains on the chosen positives only: this run had a "
                               "sequence batch and no pairs"),
                "source": "this run's own summary.json, not the current config"}
    require(sequences % 2 == 0,
            f"{path}: a pair batch needs an even, positive sequence count, got {sequences}")
    recorded_pairs = (summary.get("pairing") or {}).get("batch_pairs")
    pairs = int(recorded_pairs) if recorded_pairs is not None else sequences // 2
    require(pairs == sequences // 2,
            f"{path}: the summary records {sequences} sequences per update but {recorded_pairs} "
            "pairs per batch; those cannot both describe the same batch")
    if batch_pairs is not None:
        require(int(batch_pairs) == pairs,
                f"{path}: the caller asked to reconstruct at {batch_pairs} pairs per batch, but "
                f"this run recorded {pairs}. The current config's batch size is not evidence "
                "about what that run did; fix the caller rather than the record.")
    return {"batch_sequences": sequences, "batch_pairs": pairs,
            "recorded_batch_pairs": recorded_pairs, "pairs_applicable": True,
            "source": "this run's own summary.json, not the current config"}


def verify_reference_provenance(cache_path, summary):
    """Re-verify the ORIGINAL frozen reference against what that run recorded.

    Loading the array and hashing it proves the file is internally consistent with
    itself, which is not the question. The question is whether these are the
    numbers *that trajectory* trained against, so the sidecar identity is compared
    to the identity the run's own summary recorded, the values are re-digested
    against it, the row count is checked, and the cache's parent checkpoint is
    checked against the parent the run says it continued.
    """
    cache_path = Path(cache_path)
    recorded = dict(summary.get("reference_cache") or {})
    require(recorded.get("identity"),
            f"{cache_path}: this run's summary records no reference-cache identity, so there is "
            "nothing to verify the file against")
    sidecar = cache_path.with_suffix(".json")
    require(sidecar.is_file(),
            f"{sidecar} is missing. The cache array alone carries no identity, and an unverified "
            "array is not the frozen reference.")
    document = load_json(sidecar)
    stored = document.get("identity") or {}
    expected = recorded["identity"]
    differing = sorted(key for key in set(stored) | set(expected)
                       if stored.get(key) != expected.get(key))
    require(not differing,
            f"{sidecar}: the cache on disk does not carry the identity this run recorded. "
            f"Differing keys: {differing}")
    values = np.load(cache_path)
    require(array_digest(values) == stored.get("values_sha256"),
            f"{cache_path}: contents do not match the digest recorded with them")
    require(int(values.size) == int(stored.get("rows", values.size)),
            f"{cache_path}: {values.size} rows against the {stored.get('rows')} recorded")
    parent = (summary.get("identity") or {}).get("parent_sha256")
    if parent:
        require(stored.get("checkpoint_sha256") == parent,
                f"{cache_path}: the cache was produced by checkpoint "
                f"{stored.get('checkpoint_sha256')} and this run continued {parent}")
    return values, {"path": str(cache_path), "sha256": sha256(cache_path),
                    "sidecar": str(sidecar), "sidecar_sha256": sha256(sidecar),
                    "values_sha256": array_digest(values), "rows": int(values.size),
                    "identity": stored,
                    "creation_gpu_seconds": document.get("creation_gpu_seconds"),
                    "matches_summary_identity": True,
                    "parent_checkpoint_sha256": stored.get("checkpoint_sha256"),
                    "recorded_parity": recorded.get("fresh_parity"),
                    "note": ("verified against the identity this run's own summary recorded, not "
                            "merely re-hashed against itself")}


def summarize_run(path, *, reference_cache=None, pairing_seed_base=None, batch_pairs=None,
                  chosen_count=None, windows=DEFAULT_WINDOWS):
    """One original trajectory: identity, windows, cycles and the reconstruction.

    ``reference_cache`` is optional. When it is absent -- the caches live under
    the git-ignored output root -- the reconstruction is reported unavailable with
    the reason, rather than approximated from something else.
    """
    loaded = read_summary(path)
    summary = loaded["document"]
    history = summary.get("history") or []
    require(history, f"{path} carries no per-update history")
    method = summary.get("method")
    fields = DPO_FIELDS if method == "dpo" else SFT_FIELDS
    document = {
        "run": Path(path).parent.name, "method": method, "seed": summary.get("seed"),
        "summary": {"path": loaded["path"], "sha256": loaded["sha256"], "bytes": loaded["bytes"]},
        "updates": int(summary.get("updates", len(history))),
        "history_rows": len(history),
        "gpu_seconds": summary.get("gpu_seconds"),
        "precharged_gpu_seconds": summary.get("precharged_gpu_seconds"),
        "budgets_reached": sorted(summary.get("budgets") or {}),
        "windows": window_summary(history, fields, windows=windows),
        "cycles": cycle_summary(history, fields),
        "learning_rate": {"field_in_original_history": "learning_rate",
                          "actually": "learning_rate_next", "note": LEARNING_RATE_NOTE,
                          "first": history[0].get("learning_rate"),
                          "last": history[-1].get("learning_rate")},
        "fixed_validation_first_measurement": _first_validation(summary),
        "missing_diagnostics": MISSING_DIAGNOSTICS,
    }
    require(int(document["updates"]) == len(history),
            f"{path}: declares {document['updates']} updates but carries {len(history)} history "
            "rows; the recovered windows would not describe the trajectory")
    paired = method == "dpo"
    # Every run recorded the batch size it trained at, cache or no cache. Reporting
    # it only when a reference reconstruction was possible printed "recorded batch
    # None" for the three continued-SFT runs, which read as a missing record rather
    # than as 128 sequences and no pairs.
    batch = recorded_batch_size(path, summary, batch_pairs=batch_pairs if paired else None,
                                paired=paired)
    document["batch"] = batch
    if paired and reference_cache is not None and Path(reference_cache).is_file():
        values, provenance = verify_reference_provenance(reference_cache, summary)
        seed = int(summary["seed"])
        rows = reconstruct_batch_chosen_drop(
            history, values, chosen_count=int(chosen_count),
            pairing_seed=int(pairing_seed_base) + seed, batch_pairs=batch["batch_pairs"])
        document["reference_cache"] = provenance
        document["batch_chosen_drop"] = dict(
            summarize_batch_chosen_drop(rows, windows=windows),
            batch_pairs=batch["batch_pairs"],
            batch_source=batch["source"])
    elif not paired:
        document["batch_chosen_drop"] = {
            "available": False,
            "reason": ("continued SFT trains on the chosen positives only: it has no pairing and "
                       "no frozen preference reference, so there is no parent-relative batch drop "
                       "to reconstruct for this run")}
    else:
        document["batch_chosen_drop"] = {
            "available": False,
            "reason": ("the frozen reference cache for this run is not on disk (the output root "
                       "is git-ignored); nothing is estimated in its place")}
    return document


def _first_validation(summary):
    """The earliest fixed-validation pair metrics the original run recorded."""
    budgets = summary.get("budgets") or {}
    if not budgets:
        return {"available": False, "reason": "no budget was reached"}
    first = min(budgets, key=lambda key: float(key))
    metrics = (budgets[first].get("val_pair_metrics") or {})
    return {"target_gpu_seconds": float(first),
            "chosen_nll_per_residue": metrics.get("chosen_nll_per_residue"),
            "mean_chosen_sum_log_probability": metrics.get("mean_chosen_sum_log_probability"),
            "pairs": metrics.get("pairs"),
            "core_positions": CORE_POSITIONS,
            "note": ("the FIRST fixed-validation measurement in the original campaign. Nothing "
                     "earlier exists, so no onset time can be read off it.")}


def inspect_original_campaign(runs, *, root, reference=None):
    """The whole recovered campaign: six runs, hashed, with their windows.

    ``runs`` maps a run name to its ``summary.json`` path. ``reference`` optionally
    supplies ``{"cache_paths": {run: path}, "pairing_seed_base": int,
    "batch_pairs": int, "chosen_count": int}`` for the reconstruction.
    """
    reference = dict(reference or {})
    caches = reference.get("cache_paths") or {}
    document = {"schema_version": HISTORY_REPORT_SCHEMA, "root": str(root), "runs": {}}
    for name in sorted(runs):
        document["runs"][name] = summarize_run(
            runs[name], reference_cache=caches.get(name),
            pairing_seed_base=reference.get("pairing_seed_base"),
            batch_pairs=reference.get("batch_pairs"),
            chosen_count=reference.get("chosen_count"))
    # Every byte this inspection depended on, so a later stage can re-verify the
    # inputs rather than the conclusions. The reference caches and their sidecars
    # belong here: the reconstruction is a statement about those numbers, and a
    # cache that changed afterwards invalidates it exactly as a changed summary
    # would.
    document["inputs"] = {}
    for name in document["runs"]:
        entry = document["runs"][name]
        document["inputs"][f"{name}::summary"] = entry["summary"]
        cache = entry.get("reference_cache")
        if cache and cache.get("sha256"):
            document["inputs"][f"{name}::reference_cache"] = {
                "path": cache["path"], "sha256": cache["sha256"],
                "values_sha256": cache.get("values_sha256")}
            document["inputs"][f"{name}::reference_cache_sidecar"] = {
                "path": cache["sidecar"], "sha256": cache["sidecar_sha256"]}
    document["input_note"] = ("the summaries, the frozen reference caches and their identity "
                              "sidecars. A stage that depends on this inspection re-verifies all "
                              "of them, not just the summaries.")
    document["missing_diagnostics"] = MISSING_DIAGNOSTICS
    document["read_only"] = ("this inspection opens the original campaign's artifacts read-only "
                             "and writes nothing under its output root")
    return document
