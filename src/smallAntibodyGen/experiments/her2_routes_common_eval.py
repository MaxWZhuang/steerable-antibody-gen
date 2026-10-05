"""A read-only, CPU-only common evaluation of every route's checkpoints on one shared panel.

Why it exists
-------------
The bakeoff selected AbLang2 and p-IgGen checkpoints on ``rho(mean fixed-path log
p(window | high bin), pKD)`` (``FlightServices.score_panel``); the DPLM training phase selects on
``rho(high - null)`` (``hr17_selection_terms``), which is zero by construction at zero-initialised
condition tables. The two are not comparable, so no statement about which route is better can be
read off the recorded curves. This module scores the SAME development rows under the high bin, the
null bin and their difference for every route, saves the per-row scores, and reports Spearman rho
with intervals and a small set of pre-declared paired contrasts.

What it deliberately is not
---------------------------
* Not a held-out result. The development panel is the data every checkpoint was selected on; the
  sealed reserved panel is never opened here.
* Not a statement about training-seed variance. Each route has one seed; the intervals resample
  sequence neighbourhoods, which captures panel sampling noise only.
* Not a generation test. No sequence is produced.

Statistical contract (fixed before any score exists)
----------------------------------------------------
The primary selector is ``high`` (the legacy bakeoff definition, defined for every route and
inclusive of the pretrained prior); ``null`` and ``ratio`` are diagnostics that separate what the
prior and the learned conditioning each contribute. The bootstrap unit is a sequence
NEIGHBOURHOOD, resampled exactly as ``her2_routes_eval.paired_delta_spearman`` resamples it: the
development list carries 102 pairs of cores within Hamming distance two, so rows are not
independent. An undefined correlation (a constant score vector) is ``None`` -- never 0, never NaN.

How it stays out of a live training job's way (each point is tested; several were MEASURED)
-------------------------------------------------------------------------------------------
* Nothing is written into a flight. ``FlightServices`` would rewrite ``hard_split_freeze.json``
  and the shared source caches, so it runs on a scratch run root seeded from CAPTURED markers and
  on a scratch cache copy; the panel is extracted with ``freeze_hard_split(persist=False)``.
* No replaceable file is opened by its live name. On Windows an ordinary read handle makes the
  trainer's (or a restarting supervisor's) atomic replace fail with WinError 5, a delete-sharing
  handle does not help, and neither retries -- but a HARD LINK does not block it. Checkpoints,
  ``launch.json`` and the identity markers are therefore read through private hard-link pins.
* The GPU is untouched. ``CUDA_VISIBLE_DEVICES`` cannot hide it (on this torch build that makes a
  CPU backward exit 139), so ``forbid_cuda`` guards CUDA in-process and fails closed.
* One command at a time per output directory (an exclusive lock), saved panels are re-verified
  against their digest on every load, and the frozen source and external bytes are re-hashed
  before anything is scored.
"""
from __future__ import annotations

import contextlib
import csv
import gc
import hashlib
import io
import json
import math
import os
import platform
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import her2_routes_eval as eval_lib
from . import her2_support_paths as paths
from .her2_routes_conditions import BIN_HIGH, BIN_NULL
from .her2_runtime import require

SCHEMA = "her2-routes-common-eval/1"

#: ``high`` is the legacy bakeoff selector; ``null`` and ``ratio`` are diagnostics.
SELECTORS = ("high", "null", "ratio")
PRIMARY_SELECTOR = "high"

#: The statistical definitions, declared here once so they are fixed before any score exists.
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260929
CONFIDENCE = 0.95
UNIT = ("sequence neighbourhood: a connected component of the within-radius Hamming graph "
        "over the panel windows, so every pair within the radius travels together")

#: Every panel window is the ten-residue HER2 HCDR3 core.
PANEL_WIDTH = 10

#: Two windows within this Hamming distance belong to one sequence neighbourhood. It is the
#: evaluator's own declared analysis choice (equal to the bakeoff config's
#: ``confirmation.neighbourhood_radius``), NOT read from a flight: the DPLM training flight's
#: config has no ``confirmation`` block at all, and a per-flight radius would give two flights
#: different groupings and so different panels.
NEIGHBOURHOOD_RADIUS = 2


# ---------------------------------------------------------------------------
# the panel contract
# ---------------------------------------------------------------------------

def validate_panel(rows, *, development_windows):
    """Return the panel as sorted copies, or refuse it.

    ``development_panel_outcomes`` silently omits a development window that carries no qualified
    outcome and only requires the result to be non-empty, so two flights can agree on the same
    INCOMPLETE subset. Coverage of the frozen development list is therefore checked in both
    directions here: the panel must contain exactly the development windows, each once, each with
    a finite outcome. Rows are sorted by window so every consumer sees one canonical order.
    """
    from . import her2_data
    require(rows is not None and len(rows) > 0, "The panel is empty")
    canonical = set(her2_data.CANONICAL)
    seen, checked = set(), []
    for index, row in enumerate(rows):
        require(isinstance(row, dict) and "window" in row and "value" in row,
                f"panel row {index} is malformed: {row!r}")
        window, value = row["window"], row["value"]
        require(isinstance(window, str) and len(window) == PANEL_WIDTH,
                f"panel row {index} window {window!r} must be exactly ten residues")
        require(set(window) <= canonical,
                f"panel row {index} window {window!r} contains a non-canonical residue")
        require(window not in seen, f"panel row {index} repeats the window {window!r} "
                                    "(a duplicate window)")
        require(not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and math.isfinite(float(value)),
                f"panel row {index} value {value!r} is not a finite number")
        seen.add(window)
        checked.append({"window": window, "value": float(value)})
    expected = set(development_windows)
    missing, extra = sorted(expected - seen), sorted(seen - expected)
    require(not missing,
            f"{len(missing)} development window(s) missing from the panel (no qualified "
            f"outcome): {missing[:5]}")
    require(not extra,
            f"{len(extra)} panel window(s) not in the frozen development list: {extra[:5]}")
    return sorted(checked, key=lambda row: row["window"])


def panel_digest(rows, groups, *, context):
    """The digest every score file is bound to.

    Moves with every row, every neighbourhood label and every entry of ``context`` (assay,
    scaffold identity, edit-map size, split digest), so a score computed against a different
    panel can never be summarised with this one.
    """
    return paths.digest_document({
        "schema": SCHEMA, "record_kind": "panel",
        "rows": [{"window": row["window"], "value": float(row["value"])} for row in rows],
        "neighbourhoods": [str(label) for label in groups],
        "context": dict(context)})


# ---------------------------------------------------------------------------
# selectors and ranking statistics
# ---------------------------------------------------------------------------

def selector_vector(high, null, name):
    """One of the three declared per-row scores. The only place they are defined."""
    require(name in SELECTORS, f"Unknown selector {name!r}; expected one of {SELECTORS}")
    high, null = np.asarray(high, dtype=np.float64), np.asarray(null, dtype=np.float64)
    require(high.shape == null.shape,
            f"the high and null scores are misaligned: {high.shape} vs {null.shape}")
    return {"high": high, "null": null, "ratio": high - null}[name]


def spearman_or_none(scores, outcomes):
    """Spearman's rho (average ranks), ``None`` when undefined, a refusal when invalid.

    A constant score vector has no ranking: ``None``. A NaN or infinite value is invalid
    evidence and raises -- it is never ranked and never turned into a number.
    """
    return eval_lib.spearman(scores, outcomes)


def neighbourhood_components(windows, *, radius):
    """Label each window with its connected component under "within Hamming ``radius``".

    The repository's own ``FlightServices.neighbourhood_labels`` is a greedy cover: a window is
    assigned to its nearest medoid, so two windows within the radius of EACH OTHER can land on
    different medoids and be resampled as independent units. Measured on the frozen development
    list (100 cores): 59 greedy groups, 45 of the 102 near pairs split across groups. A
    connected component keeps every near pair, and every chain of them, together, which is what
    "the unit of resampling is a neighbourhood" has to mean. The cost is honest: one component
    can be large (27 of the 100 cores here), and the bootstrap is then dominated by whether it
    is drawn. The label is the smallest window of the component, so it does not depend on the
    order the windows were given in.
    """
    from .her2_routes_controls import hamming
    windows = [str(window) for window in windows]
    require(len(set(windows)) == len(windows), "the windows contain a duplicate")
    require(len({len(window) for window in windows}) <= 1,
            "the windows must all have the same length")
    parent = list(range(len(windows)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left in range(len(windows)):
        for right in range(left + 1, len(windows)):
            if hamming(windows[left], windows[right]) <= int(radius):
                parent[find(left)] = find(right)
    members = {}
    for index, window in enumerate(windows):
        members.setdefault(find(index), []).append(window)
    label = {root: min(group) for root, group in members.items()}
    return [label[find(index)] for index in range(len(windows))]


def grouping_record(labels, *, radius):
    """What the bootstrap unit looks like on this panel: how many groups, how big the largest is."""
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    histogram = {}
    for size in counts.values():
        histogram[str(size)] = histogram.get(str(size), 0) + 1
    return {"method": "connected_components", "radius": int(radius), "groups": len(counts),
            "largest": max(counts.values()), "rows": len(labels),
            "size_histogram": dict(sorted(histogram.items(), key=lambda item: int(item[0])))}


def _neighbourhoods(groups, outcomes_shape):
    labels = np.asarray([str(value) for value in groups])
    require(labels.shape == outcomes_shape,
            f"the neighbourhood labels are misaligned with the outcomes: {labels.shape} vs "
            f"{outcomes_shape}")
    unique = sorted(set(labels.tolist()))
    require(len(unique) > 1,
            f"A neighbourhood bootstrap needs more than one neighbourhood; the panel carries "
            f"{len(unique)}. One unit is not an interval.")
    return labels, unique


def resampled_correlations(scores, outcomes, groups, *, seed, draws):
    """Per-draw Spearman rho for each named score vector over resampled neighbourhoods.

    Returns ``{name: array(draws)}`` with NaN where a draw's correlation is undefined. The
    sampling is the one ``her2_routes_eval.paired_delta_spearman`` performs for the same seed
    (same sorted neighbourhood list, same ``rng.integers`` call), so a single-subject interval
    and a pairwise contrast are about the same draws; a test pins that equivalence.
    """
    outcomes = eval_lib.finite_vector(outcomes, "outcomes")
    vectors = {}
    for name, vector in scores.items():
        vectors[name] = eval_lib.finite_vector(vector, f"scores[{name}]")
        require(vectors[name].shape == outcomes.shape,
                f"scores[{name}] is misaligned with the outcomes: {vectors[name].shape} vs "
                f"{outcomes.shape}")
    labels, unique = _neighbourhoods(groups, outcomes.shape)
    members = {name: np.flatnonzero(labels == name) for name in unique}
    rng = np.random.default_rng(int(seed))
    result = {name: np.full(int(draws), np.nan) for name in vectors}
    for draw in range(int(draws)):
        picked = [unique[int(index)] for index in rng.integers(0, len(unique), len(unique))]
        rows = np.concatenate([members[name] for name in picked])
        for name, vector in vectors.items():
            rho = eval_lib.spearman(vector[rows], outcomes[rows])
            if rho is not None:
                result[name][draw] = rho
    return result


def neighbourhood_bootstrap(scores, outcomes, groups, *, seed=BOOTSTRAP_SEED,
                            draws=BOOTSTRAP_DRAWS, confidence=CONFIDENCE):
    """Observed rho, percentile interval and usable-draw count for each named score vector.

    The unit is a sequence neighbourhood, never a row. The interval describes panel sampling
    noise only: it contains no checkpoint-selection uncertainty and no training-seed variance.
    """
    outcome_vector = eval_lib.finite_vector(outcomes, "outcomes")
    sampled = resampled_correlations(scores, outcome_vector, groups, seed=seed, draws=draws)
    _labels, unique = _neighbourhoods(groups, outcome_vector.shape)
    alpha = (1.0 - float(confidence)) / 2.0
    subjects = {}
    for name, vector in scores.items():
        observed = eval_lib.spearman(vector, outcome_vector)
        usable = sampled[name][np.isfinite(sampled[name])]
        if observed is None or usable.size == 0:
            subjects[name] = {"rho": None, "interval": None, "usable_draws": 0}
            continue
        subjects[name] = {
            "rho": float(observed),
            "interval": [float(np.quantile(usable, alpha)),
                         float(np.quantile(usable, 1.0 - alpha))],
            "usable_draws": int(usable.size)}
    return {"neighbourhoods": len(unique), "rows": int(outcome_vector.size), "unit": UNIT,
            "draws": int(draws), "confidence": float(confidence), "seed": int(seed),
            "subjects": subjects}


def paired_contrast(scores_a, scores_b, outcomes, groups, *, seed=BOOTSTRAP_SEED,
                    draws=BOOTSTRAP_DRAWS, confidence=CONFIDENCE):
    """Difference of two subjects' Spearman correlations, over the same resampled neighbourhoods.

    Delegates to ``her2_routes_eval.paired_delta_spearman`` rather than re-implementing it. A
    contrast with an undefined half is undefined: it is reported as such, never computed from the
    other half.
    """
    rho_a = eval_lib.spearman(scores_a, outcomes)
    rho_b = eval_lib.spearman(scores_b, outcomes)
    if rho_a is None or rho_b is None:
        return {"delta": None, "interval": None, "excludes_zero": None,
                "spearman_a": rho_a, "spearman_b": rho_b,
                "reason": ("one score vector is constant on this panel, so its correlation -- "
                           "and any contrast with it -- is undefined")}
    try:
        record = eval_lib.paired_delta_spearman(scores_a, scores_b, outcomes, groups=groups,
                                                seed=seed, draws=draws, confidence=confidence)
    except ValueError as error:
        return {"delta": None, "interval": None, "excludes_zero": None,
                "spearman_a": rho_a, "spearman_b": rho_b, "reason": str(error)}
    record = dict(record)
    record["reason"] = None
    return record


# ---------------------------------------------------------------------------
# keeping the GPU out of it
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def forbid_cuda():
    """Make CUDA unreachable inside the block, and fail loudly if anything tries.

    A DPLM training job owns the 4 GB card (about 1 GB free); even a bare CUDA context here is
    VRAM it cannot spare. ``CUDA_VISIBLE_DEVICES`` is deliberately NOT used: on this machine's
    torch build any value that hides the device (``-1``, ``99``, a nonexistent UUID) makes a CPU
    backward end in a native access violation (exit 139), while an empty or absent variable does
    not. The guard is in-process instead:

    * ``torch.cuda.device_count`` is 0 and ``is_available`` is False, so a default
      ``torch.random.fork_rng()`` -- which otherwise initialises EVERY device -- has none to touch;
    * ``torch.cuda._lazy_init`` raises, which is the one path every real initialisation (a
      ``device="cuda"`` tensor, ``torch.cuda.init()``) goes through, so an attempt is an error
      rather than a silent context.

    A process that has already initialised CUDA is refused: the guard cannot undo that.
    """
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA is already initialised in this process; the common evaluation "
                           "must start in a process that has not touched the GPU")
    saved = (torch.cuda._lazy_init, torch.cuda.is_available, torch.cuda.device_count)

    def forbidden(*_args, **_kwargs):
        raise RuntimeError("CUDA initialization is forbidden in the common evaluation: a live "
                           "training job owns the GPU")

    torch.cuda._lazy_init = forbidden
    torch.cuda.is_available = lambda: False
    torch.cuda.device_count = lambda: 0
    try:
        yield
    finally:
        torch.cuda._lazy_init, torch.cuda.is_available, torch.cuda.device_count = saved
    require(not torch.cuda.is_initialized(), "CUDA was initialised while the guard was active")


# ---------------------------------------------------------------------------
# pinning a checkpoint: a hard link, never an open of the live name
# ---------------------------------------------------------------------------

_MILESTONE = re.compile(r"^step\d+\.pt$")


def checkpoint_policy(path, *, allow_live):
    """May the LIVE name of a checkpoint be opened with an ordinary handle? (the fallback only)

    The normal route never opens it: ``pinned_checkpoint`` takes a hard-link snapshot. This
    policy governs only the fallback when a link is impossible (another volume), and it is an
    ACKNOWLEDGEMENT, not a liveness test. MEASURED on this machine's NTFS volumes: an ordinary
    read handle makes the trainer's ``Path.replace(tmp, last.pt)`` fail with ``PermissionError:
    [WinError 5]``, a delete-sharing handle (FILE_SHARE_DELETE) does NOT help, and the trainer
    does not retry. Liveness cannot be the gate: a one-time probe goes stale while a long
    evaluation runs, and a supervisor that restarts re-saves ``last.pt`` within seconds of
    starting (a resumed bakeoff re-runs ``train()`` for finished routes). Earlier "stable file"
    exemptions (a ``stepN.pt`` milestone, a route with ``run.json``) fail for the same reason.
    Acknowledging the flag is safe when the flight is stopped and a risk when it is not.
    """
    if allow_live:
        return {"eligible": True, "kind": "live_acknowledged",
                "reason": "read on an explicit acknowledgement of the live-replace hazard"}
    return {"eligible": False, "kind": "refused",
            "reason": (f"{Path(path).name} may be replaced by a trainer, and an ordinary open "
                       "handle makes that replace fail on Windows (measured: WinError 5; a "
                       "delete-sharing handle does not help; the trainer does not retry). Put "
                       "the scratch directory on the checkpoint's volume so a hard-link "
                       "snapshot is possible, or pass allow_live (safe only if the flight is "
                       "stopped)")}


def _signature(path):
    status = Path(path).stat()
    return (status.st_size, status.st_mtime_ns)


def copy_with_hash(source, destination, *, chunk=8 << 20):
    """Stream ``source`` to ``destination`` and hash it in the same pass. Atomic, verified.

    The copy is the immutable input the scorer loads, so a later replacement of the original
    cannot change what was scored. Refused up front when the destination volume lacks the space,
    and refused afterwards -- with nothing left behind -- when the source's size or modification
    time moved while it was being read.
    """
    source, destination = Path(source), Path(destination)
    before = _signature(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(destination.parent).free
    require(free >= before[0] * 1.1 + (64 << 20),
            f"copying {source.name} needs about {before[0] / 1e9:.2f} GB and the destination has "
            f"{free / 1e9:.2f} GB free")
    temporary = destination.with_name(destination.name + ".tmp")
    digest, copied = hashlib.sha256(), 0
    try:
        with source.open("rb") as reader, temporary.open("wb") as writer:
            while True:
                block = reader.read(int(chunk))
                if not block:
                    break
                digest.update(block)
                writer.write(block)
                copied += len(block)
        after = _signature(source)
        require(before == after and copied == before[0],
                f"{source} changed while it was being copied (size/time {before} -> {after}, "
                f"{copied} bytes read); the copy is discarded")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"sha256": digest.hexdigest(), "bytes": copied}


def _hash_file(path, *, chunk=8 << 20):
    digest, size = hashlib.sha256(), 0
    with Path(path).open("rb") as reader:
        while True:
            block = reader.read(int(chunk))
            if not block:
                break
            digest.update(block)
            size += len(block)
    return {"sha256": digest.hexdigest(), "bytes": size}


PIN_PREFIX = "pin_"


def leftover_pins(scratch):
    """Pin directories an interrupted run left behind. They pin disk space; never auto-deleted.

    A pin directory belongs to ONE invocation, and this function cannot tell a crashed
    invocation's directory from a live one's, so it only reports. Remove them by hand when no
    evaluator is running.
    """
    root = Path(scratch)
    return sorted(path for path in root.glob(f"{PIN_PREFIX}*") if path.is_dir()) if root.is_dir() else []


@contextlib.contextmanager
def pinned_checkpoint(source, scratch, key, *, allow_live):
    """Yield a stable, hashed, private name for a checkpoint, then remove it.

    The trainer saves with ``Path.replace``. MEASURED on NTFS (this machine's C: and the F:
    artifact volume): an ordinary read handle on the live name makes that replace raise
    ``WinError 5`` and a delete-sharing handle does not help, but a HARD LINK does -- a second
    name for the same file can be opened and read for as long as needed while the trainer
    replaces the original name, and the link keeps the old bytes. So the live name is never
    opened: it is linked (no content handle; creating a link takes microseconds), the link is
    read and hashed, and the link is removed afterwards. The pinned bytes cannot change, because
    the trainer never writes a checkpoint in place.

    The pin lives in a directory private to THIS call, so two invocations that share a scratch
    root and a subject key can never delete or load each other's snapshot (which would pair one
    invocation's hash with another's weights, with no identity error: identity is constant
    across training steps). Fallback when a link is impossible (another volume): a streamed
    copy, allowed only with ``allow_live``.
    """
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    private = Path(tempfile.mkdtemp(prefix=f"{PIN_PREFIX}{key}_", dir=scratch))
    destination = private / f"{key}.pt"
    try:
        try:
            os.link(source, destination)
            method, record = "hardlink", None
        except OSError as error:
            verdict = checkpoint_policy(source, allow_live=allow_live)
            require(verdict["eligible"],
                    f"{source}: a hard-link snapshot is impossible here ({error}); "
                    f"{verdict['reason']}")
            method, record = "copy", copy_with_hash(source, destination)
        yield {"path": destination, "method": method, **(record or _hash_file(destination))}
    finally:
        shutil.rmtree(private, ignore_errors=True)


OUTPUT_LOCK = "evaluator.lock"
SCRATCH_LOCK = "scratch.lock"


@contextlib.contextmanager
def exclusive_invocation(directory, *, name=OUTPUT_LOCK):
    """Hold an exclusive lock on a directory the command writes, for the life of one command.

    Two commands writing the same output directory would race on subject files. A byte-range
    lock on ``<directory>/<name>`` -- the same mechanism and helpers the campaign lock uses --
    makes the second command fail at once instead of corrupting the first. Each command takes
    two: the output directory's, and the scratch root's (``SCRATCH_LOCK``), because the scratch
    flights (marker copies, the linked source archive) and the scratch source cache are rewritten
    by every command, and two commands with different output directories but one scratch root
    would otherwise truncate each other's markers mid-read. Private pin directories keep the
    checkpoint snapshots of two such commands apart in any case.
    """
    from . import her2_replay_campaign as replay
    lock = Path(directory) / name
    lock.parent.mkdir(parents=True, exist_ok=True)
    if not lock.is_file() or lock.stat().st_size == 0:
        lock.write_bytes(b"\0")
    handle = lock.open("r+b")
    try:
        try:
            replay._take_lock(handle)
        except OSError as error:
            raise ValueError(f"another evaluator invocation holds {lock}; one command at a time "
                             f"may use {Path(directory)} ({error})") from error
        try:
            yield lock
        finally:
            replay._drop_lock(handle)
    finally:
        handle.close()


# ---------------------------------------------------------------------------
# scoring one subject
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Subject:
    """One thing to score: a route's checkpoint, or its zero-shot backbone (no checkpoint)."""

    flight: str
    route: str
    label: str
    checkpoint: Path | None = None

    def __post_init__(self):
        for name in ("flight", "route", "label"):
            require(re.fullmatch(r"[A-Za-z0-9_.-]+", str(getattr(self, name))),
                    f"Subject.{name} {getattr(self, name)!r} is not filename-safe")
        if self.checkpoint is not None:
            object.__setattr__(self, "checkpoint", Path(self.checkpoint))

    @property
    def kind(self):
        return "zero_shot" if self.checkpoint is None else "checkpoint"

    @property
    def key(self):
        return f"{self.flight}__{self.route}__{self.label}"


#: The package whose sources the evaluator executes: ``smallAntibodyGen``, tests excluded.
PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def package_digest(root=None):
    """``(sha256, file count)`` over every ``.py`` file under the package, path and bytes alike.

    The services execute the WORKING TREE's models, serialisation and conditioning modules, and
    the frozen-flight verification does not cover them here (it runs with the runtime-closure
    check off, because a live flight's worktree is not this tree). Hashing a fixed handful of
    files would let an edit to any other module pass the resume gate, so the whole package is
    hashed; the tests and bytecode directories are not part of the scoring implementation.
    """
    root = Path(PACKAGE_ROOT if root is None else root)
    digest, count = hashlib.sha256(), 0
    for source in sorted(root.rglob("*.py")):
        relative = source.relative_to(root)
        if "tests" in relative.parts or "__pycache__" in relative.parts:
            continue
        digest.update(relative.as_posix().encode("utf-8") + b"\0")
        digest.update(paths.sha256_file(source).encode("ascii") + b"\0")
        count += 1
    return digest.hexdigest(), count


#: Native packages whose code runs inside the scoring (p-IgGen and DPLM execute ``transformers``
#: model code; AbLang2 needs ``einops``). Their installed versions are part of the identity.
DEPENDENCIES = ("transformers", "tokenizers", "safetensors", "einops", "peft")


def dependency_versions():
    """Installed version of each native scoring dependency, or ``None``; nothing is imported.

    ``importlib.metadata`` reads the distribution metadata, so this is the same before and after
    the lazy imports of the services. The flights' pinned dependency directory is on ``sys.path``
    by the time any scoring command runs (``prepare_environment``), so it is what is reported.
    """
    from importlib import metadata
    versions = {}
    for name in DEPENDENCIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def evaluator_identity():
    """The code and interpreter that produced a score, recorded beside it.

    The training identity a checkpoint is bound to covers the frozen flight's source; it says
    nothing about THIS evaluator, which runs from the working tree. So the evaluator's own bytes
    (and those of the services and statistics it leans on) are hashed separately, and so are the
    rest of the package they import (``package_digest``) and the native dependencies' versions.
    """
    from . import her2_routes_services as services_module
    modules = (sys.modules[__name__], services_module, eval_lib)
    package_sha256, package_files = package_digest()
    record = {"schema": SCHEMA,
              "sources": {Path(module.__file__).name: paths.sha256_file(module.__file__)
                          for module in modules},
              "package_sha256": package_sha256, "package_files": package_files,
              "dependencies": dependency_versions(),
              "python": platform.python_version(), "numpy": np.__version__}
    torch = sys.modules.get("torch")
    if torch is not None:
        record["torch"] = torch.__version__
    return record


# ---------------------------------------------------------------------------
# reading a flight without writing into it
# ---------------------------------------------------------------------------

#: The files ``FlightServices.training_identity`` and ``freeze_hard_split`` read from a run root.
#: A test pins this list against the spec's own constants so it cannot drift silently.
MARKERS = ("hard_split_freeze.json", "external_sources.json", "source_snapshot.json",
           "condition_vocabulary.json")


#: ``nf_spec.SNAPSHOT_DIR`` (a test pins the equality): the frozen source archive in a run root.
SNAPSHOT_DIR = "source_snapshot"


@dataclass(frozen=True)
class Flight:
    """A flight located from its own launch record, with its identity markers CAPTURED.

    ``markers`` holds the BYTES of the identity markers exactly as they were when the flight was
    captured, read through hard-link pins (see ``pinned_checkpoint``).
    A supervisor restart rewrites ``hard_split_freeze.json`` (``curriculum_stream`` -> ``closure``
    -> ``freeze_hard_split(persist=True)`` replaces it atomically) and can rewrite the others, and
    on Windows an ordinary read handle makes that replace fail, so these files are never opened by
    their live names: they are captured once, and everything afterwards uses the captured bytes.
    ``pin_parent`` is the directory (on the flight's volume) the private pin directories live in.
    """

    name: str
    run_root: Path
    artifact_root: Path
    revision: str
    config_path: Path
    worktree: Path
    pin_parent: Path | None = field(default=None, compare=False)
    markers: dict = field(default_factory=dict, compare=False, repr=False)


def _option(command, flag):
    require(flag in command, f"the launch record's command has no {flag}")
    at = command.index(flag)
    require(at + 1 < len(command), f"the launch record's {flag} has no value")
    return command[at + 1]


def _pinned_bytes(source, pin_parent, key):
    """The bytes of a replaceable file, read through a hard-link pin; the live name is never opened."""
    with pinned_checkpoint(source, pin_parent, key, allow_live=False) as pin:
        return Path(pin["path"]).read_bytes()


def flight_from_run_root(name, run_root, *, pin_parent):
    """Recover config, worktree, artifact root and revision from the flight's ``launch.json``,
    and capture its identity markers. Every replaceable file is read through a pin.

    The launch record is what the supervisor was actually started with, so the config used is the
    flight's frozen worktree copy (whose bytes the checkpoint identity hashes), never the
    current main-tree config.
    """
    require(re.fullmatch(r"[A-Za-z0-9_.-]+", str(name)), f"flight name {name!r} is not filename-safe")
    run_root = Path(run_root)
    launch = run_root / "launch.json"
    require(launch.is_file(), f"{launch} is absent; cannot recover how the flight was started")
    document = json.loads(_pinned_bytes(launch, pin_parent, f"launch_{name}").decode("utf-8"))
    require(document.get("record_kind") == "launch", f"{launch} is not a launch record")
    command = document.get("command")
    require(isinstance(command, list) and command, "the launch record carries no command")
    command = [str(part) for part in command]
    supervisors = [part for part in command if Path(part).name == "her2_routes_supervisor.py"]
    require(len(supervisors) == 1, "the launch command names no unique her2_routes_supervisor.py")
    worktree = Path(supervisors[0]).parents[1]
    config = Path(_option(command, "--config"))
    require(config.is_relative_to(worktree),
            f"the launch config {config} is not inside the flight's worktree {worktree}")
    revision = _option(command, "--revision")
    require(run_root.parent.name == revision,
            f"the run root {run_root} does not belong to revision {revision!r}")
    markers = {}
    for index, marker in enumerate(MARKERS):
        require((run_root / marker).is_file(),
                f"{marker} is absent from {run_root}; it is read for the checkpoint identity and "
                "is never synthesised")
        markers[marker] = _pinned_bytes(run_root / marker, pin_parent, f"marker_{name}_{index}")
    return Flight(name=str(name), run_root=run_root,
                  artifact_root=Path(_option(command, "--artifact-root")), revision=revision,
                  config_path=config, worktree=worktree, pin_parent=Path(pin_parent),
                  markers=markers)


def fingerprint_markers(flight):
    """Content hashes of the flight's identity markers, read through pins: the tripwire's picture."""
    require(flight.pin_parent is not None, "the flight carries no pin directory")
    return {marker: hashlib.sha256(
                _pinned_bytes(flight.run_root / marker, flight.pin_parent, f"fp_{index}")).hexdigest()
            for index, marker in enumerate(MARKERS) if (flight.run_root / marker).is_file()}


def assert_markers_unchanged(flight, before):
    """Refuse if any identity marker of the ORIGINAL run root moved while the evaluator ran.

    Deliberately narrow: a live flight legitimately writes heartbeats, progress records and
    checkpoints into its run root, so "nothing changed" is not assertable globally. The identity
    markers are the files a follow-up once corrupted by writing into a landed run root.
    """
    after = fingerprint_markers(flight)
    moved = sorted(name for name in set(before) | set(after) if before.get(name) != after.get(name))
    require(not moved, f"identity marker(s) {moved} of {flight.run_root} changed during the "
                       "evaluation; the evaluator must never write into a flight")


@dataclass(frozen=True)
class ScratchFlight:
    """The evaluator-owned run root and source cache the services are pointed at."""

    run_root: Path
    cache: Path
    markers: dict


def prepare_scratch_flight(flight, scratch_root):
    """Seed an evaluator-owned run root from the CAPTURED markers, link the frozen archive, copy the cache.

    ``FlightServices.split()`` would rewrite ``hard_split_freeze.json`` and the Li / paired-OAS
    readers rewrite their cache on a miss; with the services bound to this scratch pair, every
    write they make lands here. The markers are written from the bytes captured through pins, so
    no live marker is opened again, and the checkpoint identity (computed from them) is exactly the
    captured state. The frozen source archive (immutable) is hard-linked in so the routes verifier
    can run against this scratch run root instead of the live one.
    """
    scratch_root = Path(scratch_root)
    run_root = scratch_root / f"flight_{flight.name}"
    cache = scratch_root / "cache" / "sources"
    paths.require_disjoint_run_root(run_root, {
        "flight_run_root": flight.run_root, "artifact_root": flight.artifact_root,
        "worktree": flight.worktree})
    require(set(flight.markers) == set(MARKERS),
            f"the flight carries captured markers {sorted(flight.markers)}, not {list(MARKERS)}")
    run_root.mkdir(parents=True, exist_ok=True)
    copied = {}
    for marker, data in flight.markers.items():
        (run_root / marker).write_bytes(data)
        copied[marker] = paths.sha256_file(run_root / marker)
        require(copied[marker] == hashlib.sha256(data).hexdigest(),
                f"the scratch copy of {marker} is not byte-identical to the captured bytes")
    archive = flight.run_root / SNAPSHOT_DIR
    if archive.is_dir():
        for source in sorted(archive.rglob("*")):
            if not source.is_file():
                continue
            target = run_root / SNAPSHOT_DIR / source.relative_to(archive)
            target.parent.mkdir(parents=True, exist_ok=True)
            with contextlib.suppress(FileNotFoundError):
                target.unlink()
            try:
                os.link(source, target)
            except OSError:
                shutil.copyfile(source, target)     # frozen and immutable: a plain read is safe
    real_cache = flight.artifact_root / "cache" / "sources"
    if real_cache.is_dir():
        shutil.copytree(real_cache, cache, dirs_exist_ok=True)
    else:
        cache.mkdir(parents=True, exist_ok=True)
    return ScratchFlight(run_root=run_root, cache=cache, markers=copied)


def verify_flight(flight, scratch):
    """Re-hash the frozen source and every external source, against the SCRATCH copy of the markers.

    Copying a marker cannot verify what it points at, and ``training_identity`` reads the recorded
    digests without re-hashing the files behind them. So the verification runs here, before
    anything is scored. It runs on the scratch run root (captured markers + hard-linked frozen
    archive) rather than the live one, because the routes verifier reads the markers by name and
    a live marker may be replaced by a restarting supervisor; the worktree and external bytes it
    re-hashes are frozen and immutable.
    """
    from . import her2_routes_spec as spec
    context = spec.resolve_context(flight.worktree, config_path=flight.config_path,
                                   run_root=scratch.run_root, artifact_root=flight.artifact_root,
                                   revision=flight.revision, ensure=False)
    source = spec.verify_source_snapshot(context, label="common evaluation",
                                         check_runtime_closure=False)
    external = spec.verify_external_sources(context, label="common evaluation")
    return {"flight": flight.name, "source": source, "external": external}


def open_services(flight, scratch, *, threads=2):
    """``FlightServices`` on CPU, bound to the scratch run root and the scratch source cache."""
    import torch
    from . import her2_routes_services as services_module
    from . import her2_routes_spec as spec
    context = spec.resolve_context(flight.worktree, config_path=flight.config_path,
                                   run_root=scratch.run_root, artifact_root=flight.artifact_root,
                                   revision=flight.revision, ensure=True)
    services = services_module.FlightServices(context, device="cpu", source_cache=scratch.cache)
    # After construction: the services set torch's thread count from runtime.cpu_threads (4), and
    # the training job on the same machine needs the cores the evaluator does not.
    torch.set_num_threads(max(1, int(threads)))
    return services


# ---------------------------------------------------------------------------
# the panel
# ---------------------------------------------------------------------------

def extract_panel(services, *, flight_name):
    """The shared development panel, through the production code, without persisting anything.

    ``freeze_hard_split(persist=False)`` first: it still compares against the (scratch) record
    and caches the split, so ``split()`` never reaches its persisting default. The panel is
    validated for exact coverage of the frozen development list and bound to a digest of the
    rows, the neighbourhood labels, the assay, the scaffold and the split record.
    """
    services.freeze_hard_split(persist=False)
    split = services.split()
    checked = validate_panel(services.development_panel_outcomes(),
                             development_windows=split.development)
    radius = NEIGHBOURHOOD_RADIUS
    labels = neighbourhood_components([row["window"] for row in checked], radius=radius)
    scaffold = services.scaffold()
    context = {
        "assay": str(services.config["her2"]["assay"]),
        "split_sha256": paths.digest_document(paths.read_json(
            services.context.path("hard_split_freeze.json"))),
        "scaffold_sha256": hashlib.sha256(
            f"{scaffold.heavy}|{scaffold.light}".encode("utf-8")).hexdigest(),
        "sites": PANEL_WIDTH, "grouping": "connected_components",
        "neighbourhood_radius": radius}
    return {"schema": SCHEMA, "record_kind": "panel", "flight": str(flight_name),
            "digest": panel_digest(checked, labels, context=context), "rows": checked,
            "neighbourhoods": labels, "grouping": grouping_record(labels, radius=radius),
            "context": context, "reserved_opened": False, "evaluator": evaluator_identity()}


def load_panel(path):
    """Read a saved panel and PROVE it is what its digest says, or refuse it.

    A digest string stored beside the data is not evidence: edit the rows (a reversed panel
    flips every rho from +1 to -1) and leave the string, and a loader that only compares strings
    accepts it. So the digest is recomputed from the rows, the labels and the context; the rows
    must be in canonical form; the labels must be the connected components of the windows at the
    recorded radius; and the grouping record must describe those labels. A forger who recomputes
    the digest still cannot regroup without failing the component check.
    """
    path = Path(path)
    document = paths.read_json(path)
    require(document.get("schema") == SCHEMA and document.get("record_kind") == "panel",
            f"{path} is not a panel record of schema {SCHEMA}")
    rows, labels, context = (document.get("rows"), document.get("neighbourhoods"),
                             document.get("context"))
    require(isinstance(rows, list) and isinstance(labels, list) and isinstance(context, dict),
            f"{path} is missing its rows, neighbourhoods or context")
    require(panel_digest(rows, labels, context=context) == document.get("digest"),
            f"{path}: the stored digest does not match its content; the panel file was edited "
            "or damaged")
    require(document.get("grouping"), f"{path} carries no grouping provenance")
    checked = validate_panel(rows, development_windows=[row.get("window") for row in rows])
    require([(row["window"], float(row["value"])) for row in rows]
            == [(row["window"], row["value"]) for row in checked],
            f"{path}: the rows are not in canonical (sorted, float-valued) form")
    require(len(labels) == len(rows), f"{path}: {len(labels)} labels for {len(rows)} rows")
    radius = int(context["neighbourhood_radius"])
    require(labels == neighbourhood_components([row["window"] for row in rows], radius=radius),
            f"{path}: the neighbourhood labels are not the connected components of the panel "
            f"windows at Hamming radius {radius}")
    require(document["grouping"] == grouping_record(labels, radius=radius),
            f"{path}: the grouping record does not describe its labels")
    return document


def require_same_panel(panels):
    """Every flight must expose the IDENTICAL panel, not merely a similar one."""
    digests = {panel["flight"]: panel["digest"] for panel in panels}
    require(len(set(digests.values())) == 1,
            f"the flights expose a different panel: {digests}. Scores on different rows are "
            "not comparable.")


def _scores(values, name, expected):
    require(len(values) == expected,
            f"the {name} score has {len(values)} values for a panel of {expected} rows")
    vector = [float(value) for value in values]
    require(all(math.isfinite(value) for value in vector),
            f"the {name} score contains a non-finite value; refusing to record invalid evidence")
    return vector


def _score_loaded(services, subject, rows, pin, *, panel_digest, batch_size, started):
    """Load the pinned checkpoint (or the zero-shot backbone) and score both bins."""
    model, heads, loaded = services.load_for_scoring(
        subject.route, None if pin is None else pin["path"])
    identity = None
    if pin is not None:
        identity = dict(loaded["training_identity"])
        require(identity.get("route") == subject.route,
                f"{subject.checkpoint} was trained for route {identity.get('route')!r}, not "
                f"{subject.route!r}")
        require(identity.get("arm") == "full",
                f"{subject.checkpoint} is the {identity.get('arm')!r} arm; the common "
                "evaluation compares full arms only")
    high = _scores(services.score_panel(subject.route, None, rows, batch_size=batch_size,
                                        model=model, heads=heads, requested_bin=BIN_HIGH),
                   "high", len(rows))
    null = _scores(services.score_panel(subject.route, None, rows, batch_size=batch_size,
                                        model=model, heads=heads, requested_bin=BIN_NULL),
                   "null", len(rows))
    del model, heads
    gc.collect()
    import torch
    return {"schema": SCHEMA, "record_kind": "subject", "key": subject.key,
            "flight": subject.flight, "route": subject.route, "label": subject.label,
            "kind": subject.kind,
            "checkpoint": (None if pin is None else {
                "original_path": str(subject.checkpoint), "sha256": pin["sha256"],
                "bytes": pin["bytes"], "step": int(loaded["step"]),
                "training_identity": identity, "snapshot_method": pin["method"]}),
            "panel_digest": panel_digest, "rows": len(rows), "high": high, "null": null,
            "cuda_initialized": bool(torch.cuda.is_initialized()),
            "evaluator": evaluator_identity(),
            "seconds": round(time.monotonic() - started, 3)}


def _require_cpu(services):
    require(str(services.device) == "cpu",
            f"The common evaluation scores on cpu only; this services object uses "
            f"{services.device!r}")


def score_subject(services, subject, rows, *, scratch, panel_digest, allow_live, batch_size=4):
    """Score ``rows`` under the high and null bins for one subject. Returns its document.

    Reads a PINNED snapshot of the checkpoint (a hard link; the live name is never opened),
    loads it through ``FlightServices.load_for_scoring`` -- whose identity check is the only gate
    and is never bypassed -- and asks ``score_panel`` for both bins on the one loaded model, so
    the serialization and collation are the production ones for the route. The pin is removed
    afterwards. Nothing is written here; ``ensure_subject`` writes.
    """
    _require_cpu(services)
    started = time.monotonic()
    if subject.checkpoint is None:
        return _score_loaded(services, subject, rows, None, panel_digest=panel_digest,
                             batch_size=batch_size, started=started)
    with pinned_checkpoint(subject.checkpoint, scratch, subject.key,
                           allow_live=allow_live) as pin:
        return _score_loaded(services, subject, rows, pin, panel_digest=panel_digest,
                             batch_size=batch_size, started=started)


def evaluator_differences(saved, current):
    """The parts of two evaluator identities that differ (empty when they are the same code).

    ``torch`` is compared only when both records carry it: it is recorded when torch was loaded,
    and a command that only reads (``summarize``) never loads it.
    """
    saved, current = saved or {}, current or {}
    different = [key for key in ("schema", "sources", "package_sha256", "dependencies", "python",
                                 "numpy")
                 if saved.get(key) != current.get(key)]
    if "torch" in saved and "torch" in current and saved["torch"] != current["torch"]:
        different.append("torch")
    return different


def ensure_subject(services, subject, rows, *, out_dir, scratch, panel_digest, allow_live,
                   flight_markers, batch_size=4):
    """Score a subject once; reuse the saved document only while it is still the same evidence.

    A saved subject is reused only if ALL of these still hold: it was scored against the same
    panel digest; by the same evaluator code (otherwise a resume after an edit keeps old scores
    beside new ones and the summary contrasts two implementations); against the same flight
    identity markers (a zero-shot document carries no checkpoint hash, so this is the only tie
    between it and its flight); and, for a checkpoint, the pinned snapshot still hashes to what
    was scored. Path and training identity are not enough: the trainer replaces ``best.pt``
    atomically and its training identity does not change between steps, so a path-keyed resume
    would silently reuse yesterday's scores for today's weights.

    ``flight_markers`` is the sha256 of each captured identity marker (``ScratchFlight.markers``);
    it is recorded in every document written here.
    """
    flight_markers = dict(flight_markers)
    target = Path(out_dir) / f"subject_{subject.key}.json"
    existing = paths.read_json(target) if target.is_file() else None
    if existing is not None:
        require(existing.get("panel_digest") == panel_digest,
                f"{target} was scored against a different panel; it is not reused")
        different = evaluator_differences(existing.get("evaluator"), evaluator_identity())
        require(not different,
                f"{target} was scored by a different evaluator ({', '.join(different)}); it is "
                "not reused, because every subject of one comparison must come from one "
                "implementation. Score into a new output directory, or remove the subject "
                "files deliberately to rescore")
        require(existing.get("flight_markers") == flight_markers,
                f"{target} was scored against different (or unrecorded) flight identity markers "
                "than this flight's; it is not reused")
    if subject.checkpoint is None:
        if existing is not None:
            return existing
        document = score_subject(services, subject, rows, scratch=scratch,
                                 panel_digest=panel_digest, allow_live=allow_live,
                                 batch_size=batch_size)
    else:
        with pinned_checkpoint(subject.checkpoint, scratch, subject.key,
                               allow_live=allow_live) as pin:
            if existing is not None:
                require(existing["checkpoint"]["sha256"] == pin["sha256"],
                        f"{target}: the checkpoint content changed since it was scored; remove "
                        "the file deliberately to rescore")
                return existing
            _require_cpu(services)
            document = _score_loaded(services, subject, rows, pin, panel_digest=panel_digest,
                                     batch_size=batch_size, started=time.monotonic())
    document = {**document, "flight_markers": flight_markers}
    paths.write_json(target, document)
    return document


# ---------------------------------------------------------------------------
# which subjects exist
# ---------------------------------------------------------------------------

SUPPORTED_ROUTES = ("ablang2_masked_editor", "piggen_causal_infiller", "dplm_150m_masked")
NOT_SUPPORTED = {
    "dwjs_energy": ("the energy route scores -E(clean window, context) and has no bin, so there "
                    "is no high-versus-null score to compare"),
    "dplm_650m_lora_masked": ("a 650M base is about 2.6 GB in fp32 and this machine has roughly "
                              "4 GB of RAM free beside the training job; it is not qualified for "
                              "a CPU evaluation here")}


def plan_subjects(flight, *, routes, labels):
    """Which subjects to score for a flight, and which routes were skipped and why.

    With no explicit labels, every checkpoint of every supported route is planned (``best``,
    ``last``, each ``stepN`` milestone) plus a ``zero_shot`` comparator. Liveness does not
    matter here: checkpoints are PINNED by hard link, so a file the trainer is still replacing
    is as safe to score as a finished one. Only directory listings and ``stat`` are used; no
    file of the flight is opened. An explicit request for a label that does not exist, or for
    an unsupported route, is an error: asking for a thing and quietly not getting it is how a
    comparison loses a row.
    """
    bakeoff = Path(flight.run_root) / "bakeoff"
    require(bakeoff.is_dir(), f"{bakeoff} does not exist; the flight has no bakeoff checkpoints")
    for route in routes or ():
        require(route in SUPPORTED_ROUTES,
                f"route {route!r} is not supported by the common evaluation: "
                f"{NOT_SUPPORTED.get(route, 'unknown route')}")
    explicit = None if labels is None else list(labels)
    subjects, skipped = [], {}
    for directory in sorted(path for path in bakeoff.iterdir() if path.is_dir()):
        route = directory.name
        if routes is not None and route not in routes:
            continue
        if route not in SUPPORTED_ROUTES:
            skipped[route] = ("not supported: "
                              + NOT_SUPPORTED.get(route, "no bin-conditioned window score is "
                                                         "defined for it"))
            continue
        available = {}
        for name in ("best", "last"):
            if (directory / f"{name}.pt").is_file():
                available[name] = directory / f"{name}.pt"
        for path in sorted(directory.glob("step*.pt")):
            if _MILESTONE.match(path.name):
                available[path.stem] = path
        wanted = list(available) if explicit is None else [
            label for label in explicit if label != "zero_shot"]
        for label in wanted:
            require(label in available,
                    f"{route}: no checkpoint labelled {label!r}; available {sorted(available)}")
            subjects.append(Subject(flight=flight.name, route=route, label=label,
                                    checkpoint=available[label]))
        if explicit is None or "zero_shot" in explicit:
            subjects.append(Subject(flight=flight.name, route=route, label="zero_shot"))
    return {"subjects": sorted(subjects, key=lambda subject: subject.key), "skipped": skipped}


# ---------------------------------------------------------------------------
# the output directory
# ---------------------------------------------------------------------------

def _ignore(_message):
    return None


def refuse_destinations_near_flights(run_roots, destinations):
    """The first check, from run roots alone, before ANY file is created or read.

    Capturing a flight creates pin directories, so the destination for them must already be known
    to be safe; the launch records (which name the worktree) are not read yet. The artifact root
    follows the campaign layout ``<artifact>/runs/<revision>/<leaf>``. ``validate_destinations``
    repeats the check against the full set (worktree included) once the flights are captured.
    """
    for run_root in run_roots:
        run_root = Path(run_root)
        protected = {"flight_run_root": run_root}
        if len(run_root.parents) > 2:
            protected["artifact_root"] = run_root.parents[2]
        for destination in destinations:
            paths.require_disjoint_run_root(destination, protected)


def validate_destinations(flights, out_dir, scratch_root):
    """Refuse an output directory or scratch root inside any flight BEFORE anything is created.

    Environment preparation creates ``tmp`` under the scratch root and the output directory
    receives every artifact; neither may sit in a flight's run root, artifact root or worktree,
    and the marker tripwire would not notice a new directory there.
    """
    for label, destination in (("output directory", out_dir), ("scratch root", scratch_root)):
        for flight in flights:
            try:
                paths.require_disjoint_run_root(destination, {
                    "flight_run_root": flight.run_root, "artifact_root": flight.artifact_root,
                    "worktree": flight.worktree})
            except ValueError as error:
                raise ValueError(f"the {label} {destination} overlaps flight {flight.name!r}: "
                                 f"{error}") from error


def run_panel(flights, out_dir, scratch_root, *, threads=2, log=None):
    """Extract and save each flight's development panel. Verify first, write nothing in a flight.

    Order matters and is tested: the frozen source and external bytes are verified against the
    ORIGINAL run root before anything is built; the identity markers are fingerprinted before and
    re-checked after; the services run on the scratch run root and cache. Re-extracting a panel
    whose digest matches the saved one is allowed; a different one is refused, never overwritten.
    """
    log = log or _ignore
    out = Path(out_dir)
    validate_destinations(flights, out, scratch_root)
    ensure_output_directory(out, flights)
    with exclusive_invocation(out), exclusive_invocation(scratch_root, name=SCRATCH_LOCK):
        return _run_panel_locked(flights, out, scratch_root, threads=threads, log=log)


def _run_panel_locked(flights, out, scratch_root, *, threads, log):
    panels = []
    for flight in flights:
        before = fingerprint_markers(flight)
        scratch = prepare_scratch_flight(flight, scratch_root)
        log(f"[{flight.name}] verifying the frozen source and external bytes")
        verify_flight(flight, scratch)
        services = open_services(flight, scratch, threads=threads)
        log(f"[{flight.name}] extracting the development panel")
        panel = extract_panel(services, flight_name=flight.name)
        del services
        gc.collect()
        target = out / f"panel_{flight.name}.json"
        if target.is_file():
            existing = load_panel(target)
            require(existing.get("digest") == panel["digest"],
                    f"{target} already holds a different panel ({existing.get('digest')} vs "
                    f"{panel['digest']}); it is not overwritten")
        else:
            paths.write_json(target, panel)
        assert_markers_unchanged(flight, before)
        panels.append(panel)
    require_same_panel(panels)
    return panels


def run_score(flights, out_dir, scratch_root, *, routes, labels, allow_live, batch_size,
              threads, log=None):
    """Score every planned subject of every flight on the saved shared panel.

    Needs the panel files first (it never extracts one), and verifies each one against its own
    digest on load. Per flight: fingerprint the identity markers, seed the scratch run root from the
    captured markers, verify the frozen inputs against it, plan the subjects, score each through
    ``ensure_subject`` (which resumes only on matching evidence), and re-check the tripwire. No
    liveness, lock, heartbeat or progress file of the flight is read.
    """
    log = log or _ignore
    out = Path(out_dir)
    validate_destinations(flights, out, scratch_root)
    ensure_output_directory(out, flights)
    with exclusive_invocation(out), exclusive_invocation(scratch_root, name=SCRATCH_LOCK):
        return _run_score_locked(flights, out, scratch_root, routes=routes, labels=labels,
                                 allow_live=allow_live, batch_size=batch_size, threads=threads,
                                 log=log)


def _run_score_locked(flights, out, scratch_root, *, routes, labels, allow_live, batch_size,
                      threads, log):
    panels = []
    for flight in flights:
        target = out / f"panel_{flight.name}.json"
        require(target.is_file(),
                f"no panel file for flight {flight.name!r} in {out}; run the panel command first")
        panels.append(load_panel(target))
    require_same_panel(panels)
    panel = panels[0]
    copies = Path(scratch_root) / "copies"
    scored, skipped = [], {}
    for stale in leftover_pins(copies):
        log(f"leftover pin directory {stale} (an interrupted run); it pins disk space and is "
            "not removed automatically")
    for flight in flights:
        log(f"[{flight.name}] checkpoints are pinned by hard link, so a live trainer is not "
            "disturbed; no lock, heartbeat or progress file is read")
        plan = plan_subjects(flight, routes=routes, labels=labels)
        skipped[flight.name] = plan["skipped"]
        for name, reason in sorted(plan["skipped"].items()):
            log(f"[{flight.name}] skipped {name}: {reason}")
        before = fingerprint_markers(flight)
        scratch = prepare_scratch_flight(flight, scratch_root)
        log(f"[{flight.name}] verifying the frozen source and external bytes")
        verify_flight(flight, scratch)
        services = open_services(flight, scratch, threads=threads)
        # A self-consistent digest proves the FILE is intact, not that it describes THIS flight
        # (a panel copied from another evaluation, or a flight whose split or assay moved since).
        # Re-derive the panel from the flight now; the scores are recorded against what it yields.
        log(f"[{flight.name}] re-deriving the development panel to bind the saved one")
        fresh = extract_panel(services, flight_name=flight.name)
        require(fresh["digest"] == panel["digest"],
                f"flight {flight.name!r} no longer produces the saved panel ({fresh['digest']} "
                f"now vs {panel['digest']} saved): its split, scaffold, assay or outcomes differ "
                "from the ones the panel was extracted from. Nothing was scored")
        for subject in plan["subjects"]:
            log(f"[{flight.name}] scoring {subject.key}")
            ensure_subject(services, subject, panel["rows"], out_dir=out, scratch=copies,
                           panel_digest=panel["digest"], allow_live=allow_live,
                           flight_markers=scratch.markers, batch_size=batch_size)
            scored.append(subject.key)
        del services
        gc.collect()
        assert_markers_unchanged(flight, before)
    return {"scored": scored, "skipped": skipped}


def ensure_output_directory(out_dir, flights):
    """Create the output directory with its spec, or recognise it as ours and check the flights.

    A directory that already holds something else is refused. A recorded flight is never
    rewritten: naming it again with a different run root, revision or config is an error, because
    scores already in the directory were computed against the recorded one. A NEW flight may be
    added.
    """
    out = Path(out_dir)
    spec_path = out / "eval_spec.json"
    entries = {flight.name: {"artifact_root": str(flight.artifact_root),
                             "config_sha256": paths.sha256_file(flight.config_path),
                             "revision": flight.revision, "run_root": str(flight.run_root)}
               for flight in flights}
    if out.exists() and any(out.iterdir()) and not spec_path.is_file():
        raise ValueError(f"{out} is not empty and is not a common-evaluation directory; the "
                         "evaluator writes only into a directory of its own")
    if not spec_path.is_file():
        out.mkdir(parents=True, exist_ok=True)
        document = {"schema": SCHEMA, "record_kind": "eval_spec", "flights": entries}
        paths.write_json(spec_path, document)
        return document
    document = paths.read_json(spec_path)
    for name, entry in entries.items():
        if name in document["flights"]:
            require(document["flights"][name] == entry,
                    f"flight {name!r} is different from the one this directory was started "
                    f"with: {document['flights'][name]} vs {entry}")
    added = {name: entry for name, entry in entries.items() if name not in document["flights"]}
    if added:
        document["flights"].update(added)
        paths.write_json(spec_path, document)
    return document


# ---------------------------------------------------------------------------
# the summary
# ---------------------------------------------------------------------------

DOES_NOT_ESTABLISH = (
    "A held-out result. The development panel is the data every checkpoint was selected on "
    "(best of many evaluations), so each rho here is optimistic for a selected checkpoint; the "
    "sealed reserved panel was never opened.",
    "Training-seed variance. Each route has one seed, and the intervals resample sequence "
    "neighbourhoods, which captures panel sampling noise only.",
    "Checkpoint-selection uncertainty. The intervals do not account for having chosen among "
    "checkpoints.",
    "Independence of rows. The resampling unit is a sequence neighbourhood: a connected "
    "component of the within-radius Hamming graph. It keeps every near pair together but can be "
    "dominated by one large component (its size is in the panel record), and rows inside a "
    "component are correlated by construction.",
    "Generation quality. No sequence was generated, ranked or decoded here.",
    "Equal training. The routes differ in objective, parameter count and selector history; a "
    "common scoring endpoint does not equalise those.")

MULTIPLICITY = ("no correction for multiple comparisons was applied to the declared contrasts; "
                "read them as a short pre-declared list, not as a family-wise test")


def _read_documents(out, pattern, label):
    files = sorted(out.glob(pattern))
    require(files, f"no {label} file in {out}")
    return [paths.read_json(path) for path in files]


def _require_provenance(document):
    """A subject must NAME the implementation and the flight state it was scored under.

    Comparing identities is not enough: every field missing on every subject compares equal, so a
    whole directory scored before a fingerprint existed would be accepted as consistent. The
    evaluator record must carry every field ``evaluator_identity`` writes, and the flight markers
    must be a non-empty record.
    """
    evaluator, markers = document.get("evaluator"), document.get("flight_markers")
    complete = (isinstance(evaluator, dict)
                and all(evaluator.get(name) for name in (
                    "sources", "package_sha256", "dependencies", "python", "numpy"))
                and isinstance(markers, dict) and markers)
    require(complete,
            f"subject {document.get('key')!r} has incomplete provenance (the evaluator's "
            "sources, package hash, dependency versions and interpreter, or the flight identity "
            "markers are missing); it predates the current evaluator and cannot be summarised "
            "with the others. Rescore into a new output directory")


def _load_evidence(out):
    files = sorted(out.glob("panel_*.json"))
    require(files, f"no panel file in {out}")
    panels = [load_panel(path) for path in files]
    require_same_panel(panels)
    panel = panels[0]
    rows = len(panel["rows"])
    subjects = _read_documents(out, "subject_*.json", "subject")
    for document in subjects:
        key = document.get("key")
        require(document.get("schema") == SCHEMA,
                f"subject {key!r} has schema {document.get('schema')!r}, not {SCHEMA}")
        _require_provenance(document)
        require(document.get("panel_digest") == panel["digest"],
                f"subject {key!r} was scored against a different panel")
        require(document.get("cuda_initialized") is False,
                f"subject {key!r} recorded that CUDA was initialised; it may have touched the "
                "GPU and is not accepted")
        for name in ("high", "null"):
            require(len(document[name]) == rows,
                    f"subject {key!r}: the {name} score has {len(document[name])} values for a "
                    f"panel of {rows} rows")
    subjects = sorted(subjects, key=lambda document: document["key"])
    # Every subject of one comparison must come from one implementation and, within a flight,
    # from one frozen state of that flight; a resumed run after an edit would otherwise leave old
    # scores beside new ones and the contrasts would compare code, not models.
    baseline = subjects[0]
    for document in subjects[1:]:
        different = evaluator_differences(baseline.get("evaluator"), document.get("evaluator"))
        require(not different,
                f"subjects {baseline['key']!r} and {document['key']!r} were scored by different "
                f"evaluators ({', '.join(different)}); rescore into one new output directory")
    markers = {}
    for document in subjects:
        first = markers.setdefault(document["flight"], document)
        require(first.get("flight_markers") == document.get("flight_markers"),
                f"subjects {first['key']!r} and {document['key']!r} of flight "
                f"{document['flight']!r} were scored against different flight identity markers")
    return panel, subjects


def _declared_contrasts(subjects):
    """The pre-declared pairs: within a route best-vs-last and best-vs-zero-shot, then the
    best checkpoints of different routes against one another. Nothing else is tested."""
    groups = {}
    for document in subjects:
        groups.setdefault((document["flight"], document["route"]), {})[document["label"]] = (
            document["key"])
    pairs, bests = [], []
    for (flight, route), members in sorted(groups.items(), key=lambda item: (
            f"{item[0][0]}__{item[0][1]}")):
        if "best" not in members:
            continue
        bests.append(members["best"])
        for other, kind in (("last", "within_route_selection_decay"),
                            ("zero_shot", "within_route_learning_over_prior")):
            if other in members:
                pairs.append((members["best"], members[other], kind))
    for index, first in enumerate(bests):
        for second in bests[index + 1:]:
            pairs.append((first, second, "between_routes_best_checkpoints"))
    return pairs


def _fmt(block):
    if block is None or block["rho"] is None:
        return "undefined"
    low, high = block["interval"]
    return f"{block['rho']:+.3f} [{low:+.3f}, {high:+.3f}]"


def _report(summary):
    panel = summary["panel"]
    lines = [
        "# Common evaluation of route checkpoints", "",
        f"Panel `{panel['digest'][:12]}` -- {panel['rows']} development cores in "
        f"{panel['neighbourhoods']} sequence neighbourhoods (connected components at Hamming "
        f"radius {panel['grouping']['radius']}; the largest holds {panel['grouping']['largest']} "
        f"cores); assay `{panel['context']['assay']}`. "
        f"Primary selector `{summary['primary_selector']}` (mean fixed-path log probability of "
        "the window under the high-affinity bin, Spearman against measured pKD); `null` and "
        "`ratio` (high minus null) are diagnostics. Intervals: "
        f"{summary['bootstrap']['draws']} neighbourhood-bootstrap draws, "
        f"{int(summary['bootstrap']['confidence'] * 100)}%.", "",
        "This is not a held-out result: this is the development panel every checkpoint was "
        "selected on.", "",
        "## Subjects", "",
        "| subject | step | rho (high) | rho (null) | rho (ratio) |", "|---|---:|---|---|---|"]
    for entry in summary["subjects"]:
        step = "" if entry["checkpoint"] is None else str(entry["checkpoint"]["step"])
        lines.append(f"| `{entry['key']}` | {step} | {_fmt(entry['rho']['high'])} | "
                     f"{_fmt(entry['rho']['null'])} | {_fmt(entry['rho']['ratio'])} |")
    lines += ["", "## Declared contrasts (primary selector; a minus b)", "",
              "| a | b | delta rho | interval | excludes 0 |", "|---|---|---:|---|---|"]
    for contrast in summary["contrasts"]:
        if contrast["delta"] is None:
            lines.append(f"| `{contrast['a']}` | `{contrast['b']}` | undefined | "
                         f"{contrast['reason']} | |")
            continue
        low, high = contrast["interval"]
        lines.append(f"| `{contrast['a']}` | `{contrast['b']}` | {contrast['delta']:+.3f} | "
                     f"[{low:+.3f}, {high:+.3f}] | {'yes' if contrast['excludes_zero'] else 'no'} |")
    lines += ["", f"_{summary['multiplicity']}._", "", "## What this does not establish", ""]
    lines += [f"- {text}" for text in summary["does_not_establish"]]
    return "\n".join(lines) + "\n"


def _rows_csv(panel, subjects):
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["subject", "flight", "route", "label", "window", "outcome",
                     "neighbourhood", "high", "null", "ratio"])
    for document in subjects:
        for index, row in enumerate(panel["rows"]):
            high, null = float(document["high"][index]), float(document["null"][index])
            writer.writerow([document["key"], document["flight"], document["route"],
                             document["label"], row["window"], repr(float(row["value"])),
                             panel["neighbourhoods"][index], repr(high), repr(null),
                             repr(high - null)])
    return buffer.getvalue()


def summarize(out_dir, *, draws=BOOTSTRAP_DRAWS):
    """Turn the saved panel and subject files into ``summary.json``, a CSV and ``report.md``.

    Refuses inconsistent evidence (subjects scored against another panel, a recorded CUDA
    initialisation, a length mismatch, an unknown schema) rather than summarising around it.
    Deterministic: the bootstrap is seeded and every ordering is by key, so two runs over the same
    files produce byte-identical outputs. Holds the output directory's lock from the first read to
    the last write: two summaries share fixed temporary paths, and a scoring run finishing a
    subject between the reads would leave the three outputs describing different subject sets.
    That also means a summary cannot be taken while a scoring run is in progress.
    """
    out = Path(out_dir)
    require(out.is_dir(), f"{out} is not an output directory of the common evaluation")
    with exclusive_invocation(out):
        return _summarize_locked(out, draws=draws)


def _summarize_locked(out, *, draws):
    panel, subjects = _load_evidence(out)
    outcomes = np.array([row["value"] for row in panel["rows"]], dtype=np.float64)
    groups = panel["neighbourhoods"]
    vectors = {document["key"]: {name: selector_vector(document["high"], document["null"], name)
                                 for name in SELECTORS} for document in subjects}
    entries = []
    for document in subjects:
        bootstrap = neighbourhood_bootstrap(vectors[document["key"]], outcomes, groups,
                                            seed=BOOTSTRAP_SEED, draws=draws)
        checkpoint = document["checkpoint"]
        entries.append({
            "key": document["key"], "flight": document["flight"], "route": document["route"],
            "label": document["label"], "kind": document["kind"],
            "checkpoint": None if checkpoint is None else {
                "step": checkpoint["step"], "sha256": checkpoint["sha256"],
                "bytes": checkpoint["bytes"], "original_path": checkpoint["original_path"],
                "training_identity": checkpoint["training_identity"]},
            "rho": bootstrap["subjects"]})
    contrasts = []
    for a, b, kind in _declared_contrasts(subjects):
        record = paired_contrast(vectors[a][PRIMARY_SELECTOR], vectors[b][PRIMARY_SELECTOR],
                                 outcomes, groups, seed=BOOTSTRAP_SEED, draws=draws)
        contrasts.append({"a": a, "b": b, "kind": kind, "selector": PRIMARY_SELECTOR, **record})
    summary = {
        "schema": SCHEMA, "record_kind": "summary",
        "panel": {"digest": panel["digest"], "rows": len(panel["rows"]),
                  "neighbourhoods": len(set(groups)), "grouping": panel["grouping"],
                  "context": panel["context"],
                  "reserved_opened": bool(panel["reserved_opened"])},
        "primary_selector": PRIMARY_SELECTOR, "selectors": list(SELECTORS),
        "bootstrap": {"unit": UNIT, "draws": int(draws), "confidence": CONFIDENCE,
                      "seed": BOOTSTRAP_SEED},
        "subjects": entries, "contrasts": contrasts, "multiplicity": MULTIPLICITY,
        "does_not_establish": list(DOES_NOT_ESTABLISH),
        "scored_with": sorted({json.dumps(document["evaluator"], sort_keys=True)
                               for document in subjects}),
        "summarised_with": evaluator_identity()}
    paths.write_json(out / "summary.json", summary)
    paths.write_text(out / "per_row_scores.csv", _rows_csv(panel, subjects))
    paths.write_text(out / "report.md", _report(summary))
    return summary
