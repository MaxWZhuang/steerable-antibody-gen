"""Run plumbing shared by the three HER2 scripts: hashing, atomic writes, run identity.

Nothing here is HER2-specific science. It exists so that `audit_her2.py`,
`train_her2.py` and `evaluate_her2.py` agree on four things that are easy to get
subtly different: what a hash covers, how a partial write is prevented from
looking like a finished one, what makes two runs "the same run", and how a long
GPU loop reports progress without buffering it into oblivion.

The restart contract is the load-bearing part. A run directory carries
``run.json`` with an ``identity`` block (config hash, source hashes, arm, seed,
resolved batch size) and a ``status``. Re-running is allowed only when the
identity is byte-identical; a completed run is skipped, a half-written one has to
be discarded explicitly. Silently overwriting a finished fit with a differently
configured one is the failure this prevents.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(document):
    """Sorted-key, NaN-free JSON text. The hash of a config is the hash of this."""
    return json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"


def digest_document(document):
    return hashlib.sha256(canonical_json(document).encode("utf-8")).hexdigest()


def save_json(path, document):
    """Write through a temporary file and replace: a killed process leaves the old file.

    Every progress and result artifact goes through here. A half-written
    ``results.json`` that still parses is worse than no file at all, because an
    audit will read it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(canonical_json(document), encoding="utf-8")
    temporary.replace(path)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def relative_key(path, root):
    """Artifact keys carry the separator of the platform that wrote them; normalize."""
    return str(Path(path).relative_to(root)).replace("\\", "/")


def directory_digests(root, suffixes=(".json", ".csv", ".npy", ".pt", ".safetensors")):
    root = Path(root)
    return {relative_key(p, root): sha256(p) for p in sorted(root.rglob("*"))
            if p.is_file() and p.suffix in suffixes and p.name != "results.json"}


#: Every file whose contents can change what a fit computes. These hashes are part
#: of a run's identity, so editing the trainer and re-running the same config can no
#: longer "resume" a completed fit that the old code produced.
HER2_CODE_FILES = (
    "src/smallAntibodyGen/experiments/her2_data.py",
    "src/smallAntibodyGen/experiments/her2_policy.py",
    "src/smallAntibodyGen/experiments/her2_preferences.py",
    "src/smallAntibodyGen/experiments/her2_baselines.py",
    "src/smallAntibodyGen/experiments/her2_eval.py",
    "src/smallAntibodyGen/experiments/her2_runtime.py",
    "src/smallAntibodyGen/experiments/dpo.py",
    "scripts/train_her2.py",
    "scripts/posttrain_her2.py",
    "scripts/evaluate_her2.py",
)


def code_digests(repository_root, relative_paths=HER2_CODE_FILES):
    """SHA-256 of each scientific source file, by repository-relative path.

    Docs, configs and notes are deliberately absent: a documentation commit must
    not invalidate a finished fit, while a change to the loss must.
    """
    root = Path(repository_root)
    digests = {}
    for relative in sorted(relative_paths):
        target = root / relative
        require(target.is_file(),
                f"{relative} is part of the run identity but is missing; a silently absent source "
                "file would change the identity without anybody noticing")
        digests[relative] = sha256(target)
    return digests


class Progress:
    """Step/epoch progress with flushed prints and an atomically written record."""

    def __init__(self, path, total, *, every=100, label=""):
        # ``total`` is None for a budget-stopped loop: the update count is not known
        # in advance, and inventing one would put a fictional denominator in the log.
        require(total is None or (isinstance(total, int) and total > 0),
                "Progress needs a positive total, or None when the length is unknown")
        require(isinstance(every, int) and every > 0, "Progress interval must be positive")
        self.path, self.total, self.every, self.label = Path(path), total, every, label
        self.started = time.perf_counter()
        self.history = []

    def update(self, step, **fields):
        elapsed = time.perf_counter() - self.started
        entry = {"step": step, "elapsed_seconds": elapsed, **fields}
        self.history.append(entry)
        if step == 1 or step % self.every == 0 or step == self.total:
            save_json(self.path, {"label": self.label, "total": self.total, "history": self.history})
            detail = " ".join(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}"
                              for k, v in fields.items())
            position = f"{step}" if self.total is None else f"{step}/{self.total}"
            print(f"{self.label} step {position} {detail} {elapsed:.1f}s", flush=True)
        return entry

    def finish(self):
        save_json(self.path, {"label": self.label, "total": self.total, "history": self.history})
        return self.history


class RunLedger:
    """One run directory, its identity, and the refusal to overwrite a finished fit.

    The identity is expected to carry the config digest, the source-data digests
    **and** the scientific code digests. That last part is what stops an edited
    trainer from "resuming" a fit produced by the old code: same config, different
    loss, and the previous version of this class would have skipped the run as
    completed and then re-attributed it to the current commit.
    """

    def __init__(self, directory, identity, *, metadata=None):
        require(isinstance(identity, dict) and identity, "A run needs a non-empty identity")
        self.directory = Path(directory)
        self.identity = identity
        self.metadata = dict(metadata or {})
        self.path = self.directory / "run.json"

    def existing(self):
        return load_json(self.path) if self.path.is_file() else None

    def start(self, *, discard_incomplete=False):
        """Return "completed" to skip, or "start" after claiming the directory.

        A stale ``running`` record means a previous process died. That is not
        automatically discardable: the caller has to pass ``discard_incomplete``,
        because the alternative is a restart quietly writing over a fit whose
        checkpoints another artifact already references.
        """
        record = self.existing()
        if record is not None:
            require(record["identity"] == self.identity,
                    f"{self.path} exists with a different identity; choose a fresh output "
                    "directory rather than reusing one across configs. Scientific code hashes are "
                    "part of the identity, so an edited trainer is a different run.")
            if record["status"] == "completed":
                return "completed"
            require(discard_incomplete,
                    f"{self.path} is marked {record['status']!r}; pass --discard-incomplete to "
                    "restart it, or move it aside")
            # Whatever else is discarded, the commit that first claimed the
            # directory stays: it is the provenance of the checkpoints on disk.
            # ``setdefault`` is wrong here -- the caller's metadata already carries
            # the CURRENT commit, so the stored original would lose to it. The
            # restart's own commit is kept under a name that says what it is.
            stored = dict(record.get("metadata") or {})
            claimed = stored.get("first_claimed_git_commit")
            if claimed is not None:
                restarts = list(stored.get("restart_git_commits") or [])
                current = self.metadata.get("first_claimed_git_commit")
                if current is not None and current != claimed:
                    restarts.append(current)
                self.metadata["first_claimed_git_commit"] = claimed
                if restarts:
                    self.metadata["restart_git_commits"] = restarts
        self.directory.mkdir(parents=True, exist_ok=True)
        save_json(self.path, {"identity": self.identity, "status": "running",
                              "metadata": self.metadata})
        return "start"

    def complete(self, summary):
        """Mark completed, preserving the metadata (and commit) recorded at start."""
        record = self.existing() or {}
        metadata = dict(record.get("metadata") or self.metadata)
        save_json(self.path, {"identity": self.identity, "status": "completed",
                              "metadata": metadata, "summary": summary})


class GpuBudgetClock:
    """Accumulated **device-elapsed** seconds inside explicitly measured segments.

    The campaign's budgets (180/360/600 s, and 1200/1800 s for DPO) are defined as
    measured GPU elapsed time on the training updates, so this is what defines
    them. Three properties matter:

    * Only code inside :meth:`segment` is charged. Validation, generation and
      checkpoint I/O run outside it and are reported separately, not subtracted
      from anything.
    * Timing is by CUDA events with an explicit synchronize, so the number is
      device elapsed time, not the wall time of an asynchronous launch queue. It
      is **not** a FLOP count and not an energy measurement; a slower box would
      buy fewer updates for the same budget, and that is the intended semantics of
      "matched additional measured GPU budget".
    * A budget is checked *between* updates, so the recorded total may overshoot
      the target by at most one update. The recorded value is the actual one; the
      target is never written down as though it had been hit exactly.

    ``clock`` injects a deterministic monotonic counter for CPU tests. Production
    requires CUDA and refuses to pretend otherwise.
    """

    def __init__(self, *, device=None, clock=None):
        self.clock = clock
        self.device = device
        self.elapsed_seconds = 0.0
        self.segments = 0
        self.reused_charges = []
        if clock is None:
            import torch
            require(torch.cuda.is_available(),
                    "The declared budget is a measured GPU budget; no CUDA device is present. "
                    "Pass an explicit clock only in tests.")

    def segment(self):
        return _BudgetSegment(self)

    def charge_reused(self, seconds, *, reason):
        """Charge work that was really done earlier, e.g. a reused reference cache.

        The cost is real and belongs to this method's cold start, so it is charged
        rather than quietly forgiven by a rerun. The reason is recorded so the
        number is not mistaken for time this process spent.
        """
        require(seconds >= 0, "Cannot charge a negative duration")
        self.reused_charges.append({"seconds": float(seconds), "reason": reason})
        self.elapsed_seconds += float(seconds)
        return self.elapsed_seconds

    def _charge(self, seconds):
        self.elapsed_seconds += float(seconds)
        self.segments += 1
        return self.elapsed_seconds

    def document(self):
        return {"gpu_seconds": self.elapsed_seconds, "measured_segments": self.segments,
                "source": "injected_clock" if self.clock is not None else "cuda_events",
                "reused_charges": list(self.reused_charges),
                "definition": "device elapsed time of training updates and charged reference scoring"}


class _BudgetSegment:
    """One measured region. Charges the clock on exit, even if the body raised."""

    def __init__(self, budget):
        self.budget = budget
        self.start = None
        self.end = None
        self.seconds = None

    def __enter__(self):
        if self.budget.clock is None:
            import torch
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            self.start.record()
        else:
            self.start = self.budget.clock()
        return self

    def __exit__(self, exception_type, exception, traceback):
        if self.budget.clock is None:
            self.end.record()
            self.end.synchronize()
            self.seconds = self.start.elapsed_time(self.end) / 1000.0
        else:
            self.seconds = self.budget.clock() - self.start
        self.budget._charge(self.seconds)
        return False


def free_vram_mib():
    """Driver-level free VRAM. This is the reading the caching allocator cannot see.

    On this Windows box CUDA silently spills to system RAM instead of raising, so
    allocator statistics can look healthy while the run is thrashing. The driver
    number catches cases the allocator statistics miss; a comfortable reading is
    NOT proof that no spill occurred, which is why the step-time signal is
    recorded next to it.
    """
    import torch
    if not torch.cuda.is_available():
        return None
    free, _ = torch.cuda.mem_get_info()
    return free / 2 ** 20


def set_cpu_threads(count):
    """Pin the intra-op CPU thread count, so reported timings mean one thing.

    The measured probes this campaign quotes were taken at 4 threads, which is the
    repository's convention for the run scripts on this box. Leaving it to torch's
    default would make a later timing incomparable with the recorded probe while
    still looking like the same number.
    """
    import torch
    require(isinstance(count, int) and count > 0, "CPU thread count must be a positive integer")
    torch.set_num_threads(count)
    os.environ.setdefault("OMP_NUM_THREADS", str(count))
    return torch.get_num_threads()


def torch_runtime():
    import torch
    return {"torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda_available": bool(torch.cuda.is_available()),
            "cpu_threads": int(torch.get_num_threads()),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}
