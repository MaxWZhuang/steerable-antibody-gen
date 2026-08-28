"""Run fingerprinting and checkpoint lineage (ticket J03).

The goal is narrow and blunt: **an accidental resume against edited inputs, an
edited config, an edited tokenizer, edited approved contracts, or edited source
code must be impossible**, and every checkpoint must say what it descended from.

Six independent component fingerprints are computed, then combined:

===============  ==========================================================
component        what it pins
===============  ==========================================================
``architecture`` the CONSTRUCTED :class:`MLMConfig` plus the tokenizer identity
                 plus the model class. Deliberately NOT ``asdict(TrainConfig)``:
                 ``activation``, ``tie_weights``, ``initializer_range`` and
                 ``scale_residual_init`` are hardcoded inside ``MLMConfig`` and
                 unreachable from ``TrainConfig``, and ``vocab_size`` /
                 ``pad_token_id`` come from the tokenizer. ``MLMConfig`` is
                 built fresh in ``build_model`` and discarded, so nothing else
                 in the checkpoint records those four fields.
``objective``    every result-affecting ``TrainConfig`` field. The complement of
                 :data:`OPERATIONAL_ONLY_CONFIG_FIELDS`, which is the owner's
                 approved exclusion list and nothing else.
``tokenizer``    the actual vocabulary LIST (order included, because order IS
                 the id assignment) and the special-token ids -- never the
                 class name, which stays constant while the vocab moves.
``data``         the corpus file(s): normalized path, size, and content sha256.
``contracts``    every file under ``specs/`` by content: the approved
                 scientific contracts (D0/P0, decision records).
``source``       a content hash over every sorted regular file under ``src/``,
                 ``scripts/``, ``configs/``, ``specs/`` and ``pyproject.toml``,
                 UNTRACKED FILES INCLUDED, with an explicit allowlist of
                 generated artifacts (``__pycache__``, ``*.pyc``,
                 ``.pytest_cache``, ``*.egg-info``, ``.DS_Store``, ...) as the
                 only exclusions. The git commit and the dirty-path list are
                 recorded alongside it but do NOT enter the hash, so committing
                 unchanged content never invalidates a resume while a one-byte
                 edit always does. That is what makes a dirty edit identifiable
                 **by content**, not merely by a ``dirty: true`` flag.
===============  ==========================================================

Determinism is a hard requirement: sorted keys, relative paths only, no
timestamps, no dict-order dependence, no absolute paths. The same inputs give
the same hashes on any machine.

Nothing here imports torch or the training script; the module is a pure
function of files, a config mapping, an ``MLMConfig``, and a tokenizer.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "FINGERPRINT_SCHEMA_VERSION",
    "RUN_FINGERPRINT_KEY",
    "OPERATIONAL_ONLY_CONFIG_FIELDS",
    "PATHLIKE_CONFIG_FIELDS",
    "is_absolute_on_any_platform",
    "SOURCE_ROOTS",
    "SOURCE_FILES",
    "CONTRACT_ROOTS",
    "GENERATED_ARTIFACT_DIR_NAMES",
    "GENERATED_ARTIFACT_DIR_SUFFIXES",
    "GENERATED_ARTIFACT_SUFFIXES",
    "GENERATED_ARTIFACT_FILE_NAMES",
    "WARM_START_ARCHITECTURE_KEYS",
    "ANTIGEN_ONLY_ARCHITECTURE_KEYS",
    "ANTIBODY_ONLY_MODEL_CLASSES",
    "parent_has_antigen_stream",
    "FingerprintError",
    "ResumeFingerprintMismatch",
    "LegacyCheckpointResumeError",
    "WarmStartFingerprintMismatch",
    "DirtyWorktreeError",
    "canonical_json",
    "hash_payload",
    "hash_file",
    "normalize_path_value",
    "normalize_config_for_fingerprint",
    "objective_fingerprint",
    "tokenizer_manifest",
    "tokenizer_fingerprint",
    "architecture_manifest",
    "architecture_fingerprint",
    "data_manifest",
    "contracts_manifest",
    "source_revision",
    "is_generated_artifact",
    "iter_source_files",
    "compute_run_fingerprint",
    "read_fingerprint",
    "check_resume_fingerprint",
    "check_warm_start_fingerprint",
    "warm_start_lineage_warning",
    "require_clean_worktree",
    "describe_fingerprint_mismatch",
]


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
FINGERPRINT_SCHEMA_VERSION = 1

#: Top-level checkpoint key holding the run fingerprint payload.
RUN_FINGERPRINT_KEY = "run_fingerprint"

#: The ONLY ``TrainConfig`` fields treated as operational rather than
#: result-affecting. Owner-approved, 2026-08-27. Everything else -- including
#: ``seed``, ``use_amp``, ``device``, both worker counts, ``eval_batch_size``,
#: ``checkpoint_every_steps``, ``smoke_test_only``, every ``mask_*`` field,
#: every optimizer field, and ``conditional_denoising_eligibility`` -- is
#: result-affecting and rejects a resume when it changes.
#:
#: Why several obvious-looking "operational" fields are NOT here:
#:
#: * ``eval_batch_size`` feeds ``choose_probe_size``, and probe rows are REMOVED
#:   from the training set, so it changes the training population.
#: * ``train_num_workers`` / ``eval_num_workers`` change the collator RNG stream
#:   through ``seed_worker``.
#: * ``use_amp`` changes numerics and gates ``scheduler.step()``.
#: * ``smoke_test_only`` and ``checkpoint_every_steps`` change what the run does
#:   and when state is written.
OPERATIONAL_ONLY_CONFIG_FIELDS: frozenset[str] = frozenset({
    "output_dir",
    "tensorboard",
    "show_progress",
    "report_masked_fraction_bins",
    # The flag the resume check itself gates on. Fingerprinting it would make
    # the guard unable to fire on the very run that turns resuming on.
    "resume_from_last",
})

#: Config fields whose values are filesystem paths and must be normalized
#: (absolute -> repo-relative, or basename when outside the repo) before they
#: reach a hash.
PATHLIKE_CONFIG_FIELDS: frozenset[str] = frozenset({
    "data_path",
    "init_checkpoint",
    "output_dir",
    "config",
})

#: Directory roots walked for the source-revision content hash.
SOURCE_ROOTS: tuple[str, ...] = ("src", "scripts", "configs", "specs")

#: Individual files added to the source-revision content hash.
SOURCE_FILES: tuple[str, ...] = ("pyproject.toml",)

#: Roots holding the approved scientific contracts (D0/P0, decision records).
CONTRACT_ROOTS: tuple[str, ...] = ("specs",)

# --- The generated-artifact allowlist. Explicit and tested; nothing else is
# --- excluded from the source-content hash.
GENERATED_ARTIFACT_DIR_NAMES: frozenset[str] = frozenset({
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    ".git",
    ".tox",
    ".venv",
    "node_modules",
})
GENERATED_ARTIFACT_DIR_SUFFIXES: tuple[str, ...] = (".egg-info",)
GENERATED_ARTIFACT_SUFFIXES: frozenset[str] = frozenset({
    ".pyc", ".pyo", ".pyd", ".so", ".swp", ".swo", ".orig", ".rej", ".tmp",
})
GENERATED_ARTIFACT_FILE_NAMES: frozenset[str] = frozenset({
    ".DS_Store", "Thumbs.db",
})

#: Architecture keys a WARM START must match. Narrower than the architecture
#: fingerprint on purpose: ``initializer_range`` and ``scale_residual_init``
#: only shape a FROM-SCRATCH init, and the loaded weights overwrite it, so they
#: are part of a run's identity for a resume but irrelevant to a warm start.
#: The rest either change parameter shapes (caught here rather than by a
#: confusing strict-load error) or -- worse -- do NOT change shapes at all
#: (``activation``, ``norm_first``, ``compat_readout``), so a strict load would
#: accept the checkpoint and then compute something different.
WARM_START_ARCHITECTURE_KEYS: tuple[str, ...] = (
    "model_class",
    "vocab_size",
    "pad_token_id",
    "max_length",
    "d_model",
    "n_heads",
    "n_layers",
    "d_ff",
    "dropout",
    "activation",
    "tie_weights",
    "norm_first",
    "compat_readout",
    "use_strength_head",
    "use_length_head",
    "length_head_max",
    # Rung-1 architecture candidates (J10). `norm_type` and `ffn_type` change the
    # parameter set; `position_encoding` does not, which makes it the dangerous
    # one -- a rope checkpoint would load cleanly into a learned-position model
    # and silently compute something else.
    "position_encoding",
    "norm_type",
    "ffn_type",
    "swiglu_hidden_dim",
    "attention_bias",
    "ffn_bias",
    "encoder_n_heads",
    "cross_attention_n_heads",
    "antigen_encoder_type",
    "esm_model_name",
    "antigen_max_length",
    "antigen_encoder_finetune",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
)

#: The subset of :data:`WARM_START_ARCHITECTURE_KEYS` that describes ONLY the
#: antigen stream. These are exempt from warm-start equality when -- and only
#: when -- the parent checkpoint has no antigen stream at all.
#:
#: A stage-2 antibody-only checkpoint carries no antigen weights, so there is
#: nothing for these fields to be incompatible WITH: the antigen stream is
#: constructed by the translation step (`build_antigen_refine_init_state_dict`),
#: not loaded. Requiring equality there makes every legitimate stage-2 -> stage-3
#: transition unlaunchable -- stage 2 fingerprints `antigen_max_length: None`
#: while stage 3 fingerprints 1024, and an ESM arm additionally moves
#: `antigen_encoder_type` from "scratch" to "esm".
#:
#: When the parent IS dual-stream the equality still holds, and that is the point
#: of scoping the exemption rather than dropping the keys: a stage-3 -> stage-4
#: warm start carries TRAINED antigen weights, so a changed `antigen_max_length`
#: would silently reshape a learned positional table and a changed
#: `antigen_encoder_type` would discard a learned encoder.
ANTIGEN_ONLY_ARCHITECTURE_KEYS: frozenset[str] = frozenset({
    "antigen_encoder_type",
    "esm_model_name",
    "antigen_max_length",
    "antigen_encoder_finetune",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
})

#: Model classes that have no antigen stream. Kept as a set rather than a
#: negation so adding a third model class fails closed (unknown class => treated
#: as having an antigen stream => equality enforced).
ANTIBODY_ONLY_MODEL_CLASSES: frozenset[str] = frozenset({"AntibodyMLM"})


def parent_has_antigen_stream(parent_architecture: Mapping[str, Any]) -> bool:
    """
    Does the parent checkpoint carry antigen-stream weights?

    Answered from the recorded `model_class` rather than from the presence of
    antigen fields, because those fields are populated on every `MLMConfig`
    including an antibody-only one -- they are simply unused there.
    """
    model_class = str(parent_architecture.get("model_class", ""))
    return model_class not in ANTIBODY_ONLY_MODEL_CLASSES


_MAX_DIFF_LINES = 25
_MAX_VALUE_REPR = 160


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class FingerprintError(ValueError):
    """Base class for every lineage/provenance refusal."""


class ResumeFingerprintMismatch(FingerprintError):
    """`resume_from_last` was requested against a checkpoint from a different run."""


class LegacyCheckpointResumeError(FingerprintError):
    """`resume_from_last` was requested against a checkpoint with no fingerprint."""


class WarmStartFingerprintMismatch(FingerprintError):
    """`init_checkpoint` is architecture- or tokenizer-incompatible with this run."""


class DirtyWorktreeError(FingerprintError):
    """A promoted canonical run was requested from a dirty/unverifiable worktree."""


# --------------------------------------------------------------------------- #
# Canonical serialization
# --------------------------------------------------------------------------- #
def _json_default(value: Any) -> Any:
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, Path):
        return value.as_posix()
    return str(value)


def canonical_json(payload: Any) -> str:
    """
    Serialize `payload` deterministically: sorted keys, no whitespace, ASCII.

    Two mappings that differ only in insertion order serialize identically, so
    a fingerprint can never depend on dict ordering.
    """
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_json_default,
    )


def hash_payload(payload: Any) -> str:
    """SHA-256 over the canonical serialization of `payload`."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def hash_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 over a file's bytes, read in chunks so a 100 MB corpus is fine."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_absolute_on_any_platform(value: str | Path) -> bool:
    """
    Is ``value`` an absolute path under EITHER POSIX or Windows rules?

    ``os.path.isabs`` and ``Path.is_absolute`` answer only for the HOST os, which
    makes every caller here host-dependent in exactly the way a fingerprint must
    not be. On Windows ``Path("/scratch/corpus.gz").is_absolute()`` is ``False``
    (no drive letter), so a POSIX scratch path is not recognized as external and
    its machine-specific prefix is hashed into the objective fingerprint
    verbatim; on POSIX ``Path("C:/scratch/corpus.gz").is_absolute()`` is
    ``False`` and the same happens with a Windows path.

    Either way two machines fingerprint the same logical run differently, so a
    resume or warm-start is refused for a reason that has nothing to do with the
    objective. Deciding absoluteness under both rule sets makes the answer a
    property of the string rather than of the machine reading it.
    """
    text = str(value)
    windows = PureWindowsPath(text)
    # `PureWindowsPath.is_absolute()` requires BOTH a drive and a root, so it is
    # False for the root-relative "/scratch" and "\scratch". `.root` is the
    # discriminator that catches those, and it is also set for "C:/x", so it
    # alone would suffice; the explicit is_absolute() calls document intent.
    return (
        PurePosixPath(text).is_absolute()
        or windows.is_absolute()
        or bool(windows.root)
    )


def _cross_platform_basename(text: str) -> str:
    """
    Last path component, splitting on BOTH separators.

    ``Path("C:/data/corpus.gz").name`` is the whole string on POSIX, because a
    backslash or drive letter is an ordinary character there -- so a Windows
    path would keep its machine-specific prefix in the fingerprint on a Linux
    box, which is the same defect in the other direction.
    """
    return text.replace("\\", "/").rstrip("/").rpartition("/")[2] or text


def normalize_path_value(value: str | Path, repo_root: str | Path | None = None) -> str:
    """
    Turn a path into a machine-independent string.

    Inside the repo -> a POSIX-style repo-relative path. Outside the repo (a
    scratch dir, a tmp_path in a test, another user's home) -> the basename
    alone, because the absolute prefix is machine noise and the file's identity
    is carried by its content hash in the data manifest.
    """
    text = str(value)
    if not text:
        return text
    # Decide this from the ORIGINAL text: os.path.normpath rewrites separators
    # into the host's flavor, which is exactly the host-dependence being removed.
    absolute = is_absolute_on_any_platform(text)
    path = Path(os.path.normpath(text))
    if repo_root is not None:
        root = Path(repo_root)
        try:
            resolved_root = root.resolve()
            resolved = path if absolute else (resolved_root / path)
            resolved = Path(os.path.normpath(str(resolved)))
            return resolved.relative_to(resolved_root).as_posix()
        except (ValueError, OSError):
            pass
    if absolute:
        return _cross_platform_basename(text)
    return PurePosixPath(path.as_posix()).as_posix()


def normalize_config_for_fingerprint(
    config: Mapping[str, Any],
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """
    Drop the operational-only fields and normalize every path-shaped value.

    The result is the exact mapping the ``objective`` component hashes, and it
    is stored in the checkpoint so a mismatch can name FIELDS and VALUES rather
    than show two opaque hashes.
    """
    normalized: dict[str, Any] = {}
    for key in sorted(config):
        if key in OPERATIONAL_ONLY_CONFIG_FIELDS:
            continue
        value = config[key]
        if isinstance(value, (str, Path)) and value:
            if key in PATHLIKE_CONFIG_FIELDS or is_absolute_on_any_platform(value):
                value = normalize_path_value(value, repo_root)
            else:
                value = str(value)
        normalized[key] = value
    return normalized


def objective_fingerprint(
    config: Mapping[str, Any],
    repo_root: str | Path | None = None,
) -> str:
    """Hash of the result-affecting slice of the training config."""
    return hash_payload(normalize_config_for_fingerprint(config, repo_root))


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #
def tokenizer_manifest(tokenizer: Any) -> dict[str, Any]:
    """
    Describe a tokenizer by its VOCABULARY, not by its class name.

    The shipped vocabulary is a hardcoded 35-token list in
    ``smallAntibodyGen/tokenizer.py``. Editing that list silently remaps every
    token id in every corpus while the class name stays put, so the class name
    is recorded for readability only and the vocab list is what is hashed.
    Order matters: position in the list IS the token id.
    """
    vocab = [str(token) for token in getattr(tokenizer, "vocab", [])]
    special: dict[str, Any] = {}
    for name in ("pad", "cls", "eos", "sep", "mask", "unk"):
        token_attr = getattr(tokenizer, f"{name}_token", None)
        id_attr = getattr(tokenizer, f"{name}_id", None)
        if token_attr is not None:
            special[f"{name}_token"] = str(token_attr)
        if id_attr is not None:
            special[f"{name}_id"] = int(id_attr)
    return {
        "class": type(tokenizer).__name__,
        "vocab": vocab,
        "vocab_size": len(vocab),
        "chain_tokens": [str(t) for t in getattr(tokenizer, "chain_tokens", [])],
        "special": special,
    }


def tokenizer_fingerprint(tokenizer: Any) -> str:
    """Hash of :func:`tokenizer_manifest`."""
    return hash_payload(tokenizer_manifest(tokenizer))


# --------------------------------------------------------------------------- #
# Architecture
# --------------------------------------------------------------------------- #
def architecture_manifest(
    model_config: Any,
    tokenizer: Any,
    model_class: str,
) -> dict[str, Any]:
    """
    Describe the architecture from the CONSTRUCTED ``MLMConfig``.

    ``asdict(TrainConfig)`` is not a complete architecture description:
    ``activation``, ``tie_weights``, ``initializer_range`` and
    ``scale_residual_init`` live only on ``MLMConfig``, and ``vocab_size`` /
    ``pad_token_id`` come from the tokenizer. ``MLMConfig`` is built fresh in
    ``build_model`` and thrown away, so unless it is fingerprinted here nothing
    records those fields at all.

    ``model_class`` is included because ``training_stage`` picks between
    ``AntibodyMLM`` and ``AntibodyAntigenCrossAttention`` from the same config.
    """
    if is_dataclass(model_config) and not isinstance(model_config, type):
        fields = asdict(model_config)
    elif isinstance(model_config, Mapping):
        fields = dict(model_config)
    else:  # pragma: no cover - defensive
        raise TypeError(f"Unsupported model_config type: {type(model_config)!r}")
    return {
        "model_class": str(model_class),
        "model_config": {key: fields[key] for key in sorted(fields)},
        "tokenizer": tokenizer_fingerprint(tokenizer),
    }


def architecture_fingerprint(model_config: Any, tokenizer: Any, model_class: str) -> str:
    """Hash of :func:`architecture_manifest`."""
    return hash_payload(architecture_manifest(model_config, tokenizer, model_class))


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def data_manifest(
    data_paths: Iterable[str | Path],
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """
    Describe the corpus files a run consumes by normalized path, size, content.

    The content hash is what makes "the dataset was regenerated under the same
    filename" a resume-blocking event.
    """
    entries: dict[str, Any] = {}
    for raw in data_paths:
        path = Path(raw)
        key = normalize_path_value(path, repo_root)
        entry: dict[str, Any] = {"present": path.is_file()}
        if entry["present"]:
            entry["size_bytes"] = path.stat().st_size
            entry["sha256"] = hash_file(path)
        entries[key] = entry
    return {"files": {key: entries[key] for key in sorted(entries)}}


# --------------------------------------------------------------------------- #
# Source revision + contracts
# --------------------------------------------------------------------------- #
def is_generated_artifact(relative_path: str | Path) -> bool:
    """
    True for the explicit allowlist of generated artifacts, and nothing else.

    Everything not matched here participates in the source-content hash --
    including untracked files, which is deliberate: ``specs/`` is untracked in
    this repo today and still carries the approved contracts.
    """
    path = PurePosixPath(Path(relative_path).as_posix())
    parts = path.parts
    for part in parts[:-1]:
        if part in GENERATED_ARTIFACT_DIR_NAMES:
            return True
        if part.endswith(GENERATED_ARTIFACT_DIR_SUFFIXES):
            return True
    name = path.name
    if name in GENERATED_ARTIFACT_FILE_NAMES:
        return True
    if path.suffix in GENERATED_ARTIFACT_SUFFIXES:
        return True
    return False


def iter_source_files(
    repo_root: str | Path,
    roots: Sequence[str] = SOURCE_ROOTS,
    files: Sequence[str] = SOURCE_FILES,
) -> list[Path]:
    """Sorted repo-relative paths of every non-generated regular file to hash."""
    root = Path(repo_root)
    found: set[Path] = set()
    for name in roots:
        base = root / name
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.is_symlink() or not path.is_file():
                continue
            relative = path.relative_to(root)
            if is_generated_artifact(relative):
                continue
            found.add(relative)
    for name in files:
        path = root / name
        if path.is_file() and not path.is_symlink():
            relative = Path(name)
            if not is_generated_artifact(relative):
                found.add(relative)
    return sorted(found, key=lambda p: p.as_posix())


def _hash_tree(repo_root: str | Path, roots: Sequence[str], files: Sequence[str]) -> dict[str, str]:
    root = Path(repo_root)
    return {
        relative.as_posix(): hash_file(root / relative)
        for relative in iter_source_files(root, roots, files)
    }


def _git(repo_root: str | Path, args: Sequence[str]) -> str | None:
    """Run a git command, returning stdout or None. git is never required."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def _parse_porcelain(output: str) -> list[str]:
    paths: list[str] = []
    for line in output.splitlines():
        if len(line) < 4:
            continue
        entry = line[3:]
        if " -> " in entry:  # rename: record the destination
            entry = entry.split(" -> ", 1)[1]
        entry = entry.strip().strip('"')
        if entry:
            paths.append(entry)
    return sorted(set(paths))


def source_revision(repo_root: str | Path) -> dict[str, Any]:
    """
    Identify the source tree by CONTENT, with git metadata alongside it.

    ``content_hash`` is a hash over ``{relative_path: file_sha256}`` for every
    sorted, non-generated regular file under :data:`SOURCE_ROOTS` and
    :data:`SOURCE_FILES`. It is the only part that feeds the ``source``
    component hash: the commit id and dirty-path list are recorded for humans
    but excluded from the hash, so committing byte-identical content never
    invalidates a resume while a one-character edit always does.

    ``git`` is optional. In a non-git checkout the commit is recorded as absent
    and the content hash still fully identifies the tree.
    """
    root = Path(repo_root)
    files = _hash_tree(root, SOURCE_ROOTS, SOURCE_FILES)
    content_hash = hash_payload(files)

    commit_out = _git(root, ["rev-parse", "HEAD"])
    commit = commit_out.strip() if commit_out else None
    git_available = commit is not None

    dirty_paths: list[str] = []
    if git_available:
        status = _git(
            root,
            ["status", "--porcelain", "-uall", "--", *SOURCE_ROOTS, *SOURCE_FILES],
        )
        if status is None:
            git_available = False
        else:
            dirty_paths = [
                path for path in _parse_porcelain(status)
                if not is_generated_artifact(path)
            ]

    return {
        "content_hash": content_hash,
        "files": files,
        "commit": commit,
        "git_available": bool(git_available),
        "dirty": bool(dirty_paths),
        "dirty_paths": dirty_paths,
    }


def contracts_manifest(repo_root: str | Path) -> dict[str, Any]:
    """
    Hash every approved scientific contract (``specs/``) by content.

    Recorded as its own component so a run says exactly which D0/P0 and
    decision-record revisions it was produced under, independently of the fact
    that ``specs/`` also participates in the broader source hash.
    """
    files = _hash_tree(repo_root, CONTRACT_ROOTS, ())
    return {"files": files, "content_hash": hash_payload(files)}


# --------------------------------------------------------------------------- #
# Combined run fingerprint
# --------------------------------------------------------------------------- #
def compute_run_fingerprint(
    *,
    config: Mapping[str, Any],
    model_config: Any,
    tokenizer: Any,
    model_class: str,
    data_paths: Iterable[str | Path],
    repo_root: str | Path,
    parent_checkpoint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the full fingerprint payload stored in every checkpoint.

    Returns a plain JSON-serializable mapping with:

    ``components``
        the six component hashes;
    ``run_hash``
        the combined hash, over the components AND the parent-checkpoint hash,
        which is what ``resume_from_last`` must match exactly;
    ``parent_checkpoint`` / ``parent_checkpoint_hash``
        lineage back to the ``init_checkpoint`` this run warm-started from;
    ``worktree_dirty``
        the dirty indicator;
    ``manifests``
        the full per-component manifests, which is what lets a mismatch report
        FIELD NAMES and OLD/NEW VALUES instead of two opaque hashes.
    """
    architecture = architecture_manifest(model_config, tokenizer, model_class)
    objective = normalize_config_for_fingerprint(config, repo_root)
    tokenizer_info = tokenizer_manifest(tokenizer)
    data = data_manifest(data_paths, repo_root)
    contracts = contracts_manifest(repo_root)
    source = source_revision(repo_root)

    components = {
        "architecture": hash_payload(architecture),
        "objective": hash_payload(objective),
        "tokenizer": hash_payload(tokenizer_info),
        "data": hash_payload(data),
        "contracts": contracts["content_hash"],
        # Content only: see source_revision's docstring for why the commit id
        # and dirty flag are recorded but not hashed.
        "source": source["content_hash"],
    }

    parent = dict(parent_checkpoint) if parent_checkpoint else None
    parent_hash = None
    if parent is not None:
        parent_hash = parent.get("run_hash") or parent.get("file_sha256")

    run_hash = hash_payload({
        "schema_version": FINGERPRINT_SCHEMA_VERSION,
        "components": components,
        "parent_checkpoint_hash": parent_hash,
    })

    return {
        "schema_version": FINGERPRINT_SCHEMA_VERSION,
        "components": components,
        "run_hash": run_hash,
        "parent_checkpoint": parent,
        "parent_checkpoint_hash": parent_hash,
        "worktree_dirty": bool(source["dirty"]),
        "manifests": {
            "architecture": architecture,
            "objective": objective,
            "tokenizer": tokenizer_info,
            "data": data,
            "contracts": contracts,
            "source": source,
        },
    }


def read_fingerprint(checkpoint: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Extract a fingerprint payload from a loaded checkpoint, or None if legacy."""
    if not isinstance(checkpoint, Mapping):
        return None
    payload = checkpoint.get(RUN_FINGERPRINT_KEY)
    if isinstance(payload, Mapping) and payload.get("run_hash"):
        return dict(payload)
    return None


# --------------------------------------------------------------------------- #
# Mismatch reporting
# --------------------------------------------------------------------------- #
def _flatten(payload: Any, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    if isinstance(payload, Mapping):
        for key in sorted(payload, key=str):
            child = f"{prefix}.{key}" if prefix else str(key)
            flat.update(_flatten(payload[key], child))
    else:
        flat[prefix] = payload
    return flat


def _render(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        text = repr(list(value))
    else:
        text = repr(value) if isinstance(value, str) else str(value)
    if len(text) > _MAX_VALUE_REPR:
        text = text[: _MAX_VALUE_REPR - 3] + "..."
    return text


def _diff_lines(component: str, old: Any, new: Any) -> list[str]:
    old_flat = _flatten(old)
    new_flat = _flatten(new)
    lines: list[str] = []
    for key in sorted(set(old_flat) | set(new_flat)):
        old_value = old_flat.get(key, "<absent>")
        new_value = new_flat.get(key, "<absent>")
        if old_value == new_value:
            continue
        field = f"{component}.{key}" if key else component
        lines.append(f"  {field}: checkpoint={_render(old_value)}, run={_render(new_value)}")
    return lines


def describe_fingerprint_mismatch(
    stored: Mapping[str, Any],
    current: Mapping[str, Any],
    components: Sequence[str] | None = None,
) -> str:
    """
    Render a mismatch as field names with old/new values, never as two hashes.

    Falls back to naming the component when a manifest is missing (a checkpoint
    written by an older fingerprint schema).
    """
    stored_components = dict(stored.get("components") or {})
    current_components = dict(current.get("components") or {})
    stored_manifests = dict(stored.get("manifests") or {})
    current_manifests = dict(current.get("manifests") or {})

    names = list(components) if components is not None else sorted(
        set(stored_components) | set(current_components)
    )

    lines: list[str] = []
    for name in names:
        if stored_components.get(name) == current_components.get(name):
            continue
        if name in stored_manifests or name in current_manifests:
            detail = _diff_lines(
                name,
                stored_manifests.get(name, {}),
                current_manifests.get(name, {}),
            )
        else:
            detail = []
        if detail:
            lines.extend(detail)
        else:
            lines.append(
                f"  {name}: checkpoint={_render(stored_components.get(name))}, "
                f"run={_render(current_components.get(name))} (no manifest recorded)"
            )

    stored_parent = stored.get("parent_checkpoint_hash")
    current_parent = current.get("parent_checkpoint_hash")
    if components is None and stored_parent != current_parent:
        lines.append(
            f"  parent_checkpoint_hash: checkpoint={_render(stored_parent)}, "
            f"run={_render(current_parent)}"
        )

    if not lines:
        return "  (component hashes differ but no field-level difference was found)"
    if len(lines) > _MAX_DIFF_LINES:
        hidden = len(lines) - _MAX_DIFF_LINES
        lines = lines[:_MAX_DIFF_LINES] + [f"  ... and {hidden} more difference(s)"]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Policy checks
# --------------------------------------------------------------------------- #
def check_resume_fingerprint(
    stored: Mapping[str, Any] | None,
    current: Mapping[str, Any],
    checkpoint_path: str | Path,
) -> None:
    """
    Require an EXACT run-fingerprint match before a resume touches any state.

    Raises :class:`LegacyCheckpointResumeError` when the checkpoint predates
    fingerprinting (owner decision: legacy checkpoints are unsupported for
    resume), and :class:`ResumeFingerprintMismatch` when any component moved.
    Call this BEFORE model/optimizer/scaler/scheduler state is loaded.
    """
    path = Path(checkpoint_path).as_posix()
    if stored is None or not stored.get("run_hash"):
        raise LegacyCheckpointResumeError(
            f"Refusing to resume from {path}: the checkpoint carries no "
            f"'{RUN_FINGERPRINT_KEY}' payload, so there is no way to verify that this "
            "run's config, data, tokenizer, architecture, approved contracts and "
            "source revision match the ones that produced it.\n"
            "This checkpoint predates run fingerprinting (J03). Legacy checkpoints are "
            "supported for warm start (--init-checkpoint) but NOT for resume.\n"
            "Either start a fresh run with --no-resume-from-last, or warm-start from "
            f"this checkpoint with --init-checkpoint {path} and a new --output-dir."
        )

    if stored.get("run_hash") == current.get("run_hash"):
        return

    raise ResumeFingerprintMismatch(
        f"Refusing to resume from {path}: the run fingerprint does not match.\n"
        f"  checkpoint run_hash={stored.get('run_hash')}\n"
        f"  current    run_hash={current.get('run_hash')}\n"
        "Differences (checkpoint -> this run):\n"
        + describe_fingerprint_mismatch(stored, current)
        + "\nA resume must reproduce the interrupted run exactly. Revert the change, or "
        "start a new run with a new --output-dir and --no-resume-from-last."
    )


def check_warm_start_fingerprint(
    parent: Mapping[str, Any] | None,
    current: Mapping[str, Any],
    parent_path: str | Path,
) -> None:
    """
    Warm-start rules, deliberately SEPARATE from the resume rules.

    A warm start is *meant* to change the objective and the data -- that is what
    a stage transition is. What it may not change is the architecture the
    weights were trained under, or the tokenizer that assigned their token ids.
    Fields that only shape a from-scratch init are ignored (the loaded weights
    overwrite it); see :data:`WARM_START_ARCHITECTURE_KEYS`.

    A parent with no fingerprint is ALLOWED (owner decision) -- see
    :func:`warm_start_lineage_warning`.
    """
    if parent is None or not parent.get("run_hash"):
        return

    path = Path(parent_path).as_posix()
    parent_manifests = dict(parent.get("manifests") or {})
    current_manifests = dict(current.get("manifests") or {})

    mismatches: list[str] = []

    parent_tokenizer = parent_manifests.get("tokenizer")
    current_tokenizer = current_manifests.get("tokenizer")
    if parent.get("components", {}).get("tokenizer") != current.get("components", {}).get("tokenizer"):
        if parent_tokenizer is not None and current_tokenizer is not None:
            mismatches.extend(_diff_lines("tokenizer", parent_tokenizer, current_tokenizer))
        else:
            mismatches.append(
                f"  tokenizer: checkpoint={_render(parent.get('components', {}).get('tokenizer'))}, "
                f"run={_render(current.get('components', {}).get('tokenizer'))}"
            )

    parent_arch = (parent_manifests.get("architecture") or {}).get("model_config", {})
    current_arch = (current_manifests.get("architecture") or {}).get("model_config", {})
    parent_architecture = parent_manifests.get("architecture") or {}

    # Antigen-only fields are exempt ONLY when the parent has no antigen stream
    # to be incompatible with. See ANTIGEN_ONLY_ARCHITECTURE_KEYS.
    antigen_exempt = not parent_has_antigen_stream(parent_architecture)
    antigen_transitions: list[str] = []

    for key in WARM_START_ARCHITECTURE_KEYS:
        if key == "model_class":
            # A stage transition legitimately swaps AntibodyMLM for
            # AntibodyAntigenCrossAttention, and that path has its own
            # translation step (`build_antigen_refine_init_state_dict`). The key
            # stays in the tuple because it IS part of the architecture
            # identity; it is simply not a warm-start blocker.
            continue
        if key not in parent_arch or key not in current_arch:
            continue
        if parent_arch[key] == current_arch[key]:
            continue
        if antigen_exempt and key in ANTIGEN_ONLY_ARCHITECTURE_KEYS:
            # Allowed, but never silent: a reader must be able to see which
            # antigen fields the transition introduced.
            antigen_transitions.append(
                f"  architecture.{key}: {_render(parent_arch[key])} -> "
                f"{_render(current_arch[key])}"
            )
            continue
        mismatches.append(
            f"  architecture.{key}: checkpoint={_render(parent_arch[key])}, "
            f"run={_render(current_arch[key])}"
        )

    if antigen_transitions:
        print(
            f"[warm-start] {path}: antibody-only parent, so the antigen stream is "
            "CONSTRUCTED here rather than loaded. Antigen-only fields introduced by "
            "this transition:\n" + "\n".join(sorted(antigen_transitions))
        )

    if mismatches:
        hint = ""
        if not antigen_exempt and any(
            f"  architecture.{key}:" in line
            for line in mismatches
            for key in ANTIGEN_ONLY_ARCHITECTURE_KEYS
        ):
            hint = (
                "\n\nAn antigen-only field differs and the parent checkpoint ALREADY HAS a "
                "trained antigen stream, so this is not a stream-introducing transition: the "
                "parent's antigen weights would be silently reshaped or discarded. Root this "
                "run at an antibody-only (stage-2) checkpoint instead, which is also what the "
                "plan's J24 encoder comparison requires."
            )
        raise WarmStartFingerprintMismatch(
            f"init_checkpoint {path} is not warm-start compatible with this run.\n"
            "The architecture and tokenizer must match; the objective and the data are "
            "allowed to change (that is what a stage transition is).\n"
            "Mismatches (checkpoint -> this run):\n"
            + "\n".join(mismatches[:_MAX_DIFF_LINES])
            + hint
        )


def warm_start_lineage_warning(
    parent: Mapping[str, Any] | None,
    parent_path: str | Path,
) -> str | None:
    """
    The loud warning emitted when warm-starting from a fingerprintless parent.

    Owner decision: this is ALLOWED (all six checkpoints on disk predate J03),
    still goes through ``validate_init_checkpoint_compatibility``, and must say
    out loud that the lineage cannot be verified.
    """
    if parent is not None and parent.get("run_hash"):
        return None
    path = Path(parent_path).as_posix()
    return (
        f"[warn] init_checkpoint {path} carries no '{RUN_FINGERPRINT_KEY}': its lineage "
        "is UNVERIFIABLE. It predates run fingerprinting, so nothing records which "
        "config, corpus, tokenizer, contracts or source revision produced these "
        "weights. The architecture compatibility check still ran, but provenance for "
        "this chain starts here -- do not report a result from it as reproducible "
        "without re-deriving the parent."
    )


def require_clean_worktree(fingerprint: Mapping[str, Any]) -> None:
    """
    Gate for a PROMOTED canonical run: the worktree must be verifiably clean.

    A development run may be dirty, but only because the complete source-content
    hash and the dirty paths are recorded in the fingerprint. A promoted run
    gets no such latitude: an unverifiable worktree (no git) is refused too,
    because "not known to be dirty" is not "known to be clean".
    """
    source = dict((fingerprint.get("manifests") or {}).get("source") or {})
    if not source.get("git_available"):
        raise DirtyWorktreeError(
            "A promoted canonical run requires a clean worktree, but the worktree state "
            "cannot be verified: no usable git checkout was found. The full source-content "
            f"hash was still recorded ({source.get('content_hash')}), which is enough for a "
            "development run but not for promotion."
        )
    if source.get("dirty"):
        paths = list(source.get("dirty_paths") or [])
        shown = "\n".join(f"  {p}" for p in paths[:_MAX_DIFF_LINES])
        extra = "" if len(paths) <= _MAX_DIFF_LINES else f"\n  ... and {len(paths) - _MAX_DIFF_LINES} more"
        raise DirtyWorktreeError(
            "A promoted canonical run requires a clean worktree. These paths are "
            f"modified or untracked:\n{shown}{extra}\n"
            "Commit (or stash) them and rerun, or drop --require-clean-worktree to record "
            "this as a development run -- its source-content hash and dirty paths are "
            "already recorded either way."
        )
