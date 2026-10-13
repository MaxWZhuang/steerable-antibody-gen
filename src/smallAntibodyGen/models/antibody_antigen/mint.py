"""Frozen native MINT as a :class:`ResidueBackend`.

Three separable pieces:

* **Source** -- :func:`verify_source_closure` / :func:`load_native_package` check the 11
  pinned upstream files (CRLF normalised to LF, nothing else) *before* anything is
  imported, then execute the upstream package ``__init__`` through a standard
  ``importlib`` spec under a unique throw-away namespace. Nothing is rewritten, no
  ``sys.path`` entry or global alias is added, bytecode is not written, and an unrelated
  ``esm`` / ``mint`` module elsewhere in ``sys.modules`` is neither used nor disturbed.
  A cache file the current interpreter would read for a pinned source is refused, because
  ``importlib`` runs a matching-mtime ``.pyc`` without comparing it to the pinned source.
* **Weights** -- :func:`read_checkpoint_state` / :func:`load_native_state` load with
  ``weights_only=True`` on CPU, require the explicit ``state_dict`` envelope and a single
  leading ``model.`` prefix, and refuse any missing/unexpected/shape/dtype/non-finite/untied
  entry *before* a strict load. The ordinary reader has no exception of its own. The pinned
  release's run metadata pickles one ``argparse.Namespace``, which ``weights_only=True``
  refuses; :func:`_read_verified_release_state` admits exactly that one class, scoped to a
  load that begins with :func:`verify_checkpoint_file`; the architecture is never configured from the metadata.
* **Adapter** -- :class:`MintBackend` serialises a :class:`VisibleView` the way the native
  model expects (each present chain as ``<cls> residues <eos>``, chain ids covering the
  specials, an absent chain omitted entirely), and returns the 20 canonical raw logits, the
  canonical mass of the FULL 33-way softmax, the final-layer residue features and the mask
  ratio the native token-dropout rescale actually used.

The public loader is :func:`load_pinned_mint` and it builds only the pinned release.
Constructing :class:`MintBackend` directly around a tiny model is a software-test seam:
its :meth:`~MintBackend.describe` says so and no config or CLI reaches it.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import platform
import sys
import threading
import uuid
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Mapping, Sequence, Tuple

import torch

from smallAntibodyGen.models.antibody_antigen.denoising import BackendOutput, check_output
from smallAntibodyGen.models.antibody_antigen.views import CANONICAL_RESIDUES, VisibleView

CODE_COMMIT = "06694b7606e2d00b76ec58daf5c7aecdaf7cd283"
HF_REPO = "varunullanat2012/mint"
HF_REVISION = "75b1602c34e9df1a89089a1245f000a51ee3c746"
CHECKPOINT_BYTES = 3253773059
CHECKPOINT_SHA256 = "84a4016365997cd9f0bccb07d746fa8f076ffd8e45aa0cbcf4e50a037161a342"

NATIVE_ARCHITECTURE: Mapping[str, object] = {
    "num_layers": 33,
    "embed_dim": 1280,
    "attention_heads": 20,
    "alphabet": "ESM-1b",
    "token_dropout": True,
    "use_multimer": True,
}

# SHA-256 of each upstream file after CRLF -> LF.
SOURCE_SHA256: Mapping[str, str] = {
    "mint/__init__.py": "bb5e2e6e936a0626d584c7ef08d7b8c34a6a3574aec784551af66c93fbf2359e",
    "mint/version.py": "68e1dac0bd73b8731f056297ab05245dc5bdea85cb34263e0977470ea933e20e",
    "mint/data.py": "5ffa1d756fd449e3bc3b4321d4a3963c62f0d65af54ee25f3c2007d17bd4b588",
    "mint/constants.py": "8bf9231414c6e2e61efc53be627225597eeb9fc72310b756becb3c68139a59a1",
    "mint/modules.py": "a16578740742786255ddc8058b586ac09ab27b2944f45aa62566b123c781aa79",
    "mint/multihead_attention.py": "f0284b01c0d45a4d78b50503de733a893f0d4c9ede21475edf3959e610e90267",
    "mint/rotary_embedding.py": "b1dd8c57fd4829d40ac0f3fb498317f684704a670dccb4de62556b5867a56398",
    "mint/axial_attention.py": "ae2c58c650b3c0eee2e13c26210c2e19b97b27ad78c33517a5df18bfac0bd17f",
    "mint/model/__init__.py": "01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b",
    "mint/model/esm1.py": "74ed7e6b05c1e03fdfdcaaf5b2a08cde74ac8cfbd88cef9e24463c95285c30ed",
    "mint/model/esm2.py": "ba94cdd47f0d7242414424af0114f3bffe0677d47d21f213507f6c12a2de9cd7",
}

NATIVE_ALPHABET: Tuple[str, ...] = (
    "<cls>", "<pad>", "<eos>", "<unk>", "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K", "Q", "N",
    "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z", "O", ".", "-", "<null_1>", "<mask>",
)  # fmt: skip
CLS_IDX, PAD_IDX, EOS_IDX, MASK_IDX = 0, 1, 2, 32
# Native token id of each residue of CANONICAL_RESIDUES, in that order.
NATIVE_CANONICAL_IDS: Tuple[int, ...] = (5, 23, 13, 9, 18, 6, 21, 12, 15, 4, 20, 17, 14, 16, 10, 8, 11, 7, 22, 19)
_NATIVE_ID = dict(zip(CANONICAL_RESIDUES, NATIVE_CANONICAL_IDS))

SUPPORTED_DTYPES = (torch.float32, torch.float64)
_MODEL_PREFIX = "model."
_TORCH_LAYER_NORM = "torch.nn.modules.normalization.LayerNorm"
# The one global the release's metadata needs beyond torch's default weights-only allowlist.
_NAMESPACE_GLOBAL = "argparse.Namespace"
# Serialises this module's verified release loads. Torch's safe-globals set is process-global
# (a scope adds on enter and subtracts on exit; it is neither thread-local nor a snapshot restore),
# so the lock coordinates THIS loader only: it cannot stop other code mutating that set.
_RELEASE_LOAD_LOCK = threading.RLock()


class MintLoadError(RuntimeError):
    """The pinned MINT source, checkpoint or adapter preconditions were not met."""


# --------------------------------------------------------------------------- source


def _canonical_lf_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def verify_source_closure(source_root: Path, pins: Mapping[str, str] = SOURCE_SHA256) -> None:
    """Raise unless every pinned file exists under ``source_root`` with the pinned hash."""
    problems = []
    for relative, expected in pins.items():
        path = Path(source_root) / relative
        if not path.is_file():
            problems.append(f"{relative}: missing")
            continue
        actual = _canonical_lf_sha256(path)
        if actual != expected:
            problems.append(f"{relative}: sha256 {actual} != pinned {expected}")
    if problems:
        raise MintLoadError("MINT source does not match the pinned release:\n  " + "\n  ".join(problems))


def _forget(namespace: str) -> None:
    for key in [k for k in sys.modules if k == namespace or k.startswith(namespace + ".")]:
        del sys.modules[key]


def _check_native_origins(namespace: str, source_root: Path, pins: Mapping[str, str]) -> None:
    allowed = {(source_root / relative).resolve() for relative in pins}
    for key, module in list(sys.modules.items()):
        if key != namespace and not key.startswith(namespace + "."):
            continue
        origin = getattr(module, "__file__", None)
        if origin is None or Path(origin).resolve() not in allowed:
            raise MintLoadError(f"{key} resolved to {origin!r}, which is outside the pinned source closure")


def _current_bytecode_candidates(source_root: Path, pins: Mapping[str, str]) -> List[str]:
    """Cache files the running interpreter would read for a pinned source, named by importlib itself.

    ``lexists``, not ``exists``: a broken symlink at the candidate path is still refused.
    """
    found: List[str] = []
    for relative in pins:
        try:
            candidate = importlib.util.cache_from_source(str(source_root / relative))
        except NotImplementedError:  # no cache tag: importlib reads no bytecode either
            return []
        if os.path.lexists(candidate):
            found.append(candidate)
    return found


def load_native_package(source_root: Path, pins: Mapping[str, str] = SOURCE_SHA256) -> ModuleType:
    """Import the pinned upstream ``mint`` package as a uniquely named private namespace.

    ``pins`` defaults to the release pins; it is a parameter only so tests can exercise the
    refusal paths against a small fake source tree.

    ``importlib`` runs a cached ``.pyc`` whose recorded mtime and size match its source without
    comparing it to the source, so matching pins say nothing about it. Before anything is
    imported, every cache file the current interpreter would read for a pinned source (as named
    by ``importlib.util.cache_from_source``, so ``-O`` and ``sys.pycache_prefix`` are respected)
    must be absent. Nothing is deleted or modified here; remove stale cache separately. This is a
    refusal of stale cache, not a defence against a filesystem that changes during the import.
    """
    source_root = Path(source_root).resolve()
    verify_source_closure(source_root, pins)
    cached = _current_bytecode_candidates(source_root, pins)
    if cached:
        raise MintLoadError(
            "cached bytecode exists for the pinned source and importlib would run it without checking it "
            "against the pins:\n  "
            + "\n  ".join(cached)
            + "\nremove stale or unverified cached bytecode separately from the pinned source tree, then import again"
        )
    package_dir = source_root / "mint"
    namespace = f"_smallantibodygen_native_mint_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(
        namespace, package_dir / "__init__.py", submodule_search_locations=[str(package_dir)]
    )
    if spec is None or spec.loader is None:
        raise MintLoadError(f"cannot build an import spec for {package_dir}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.modules[namespace] = module
    try:
        spec.loader.exec_module(module)
        _check_native_origins(namespace, source_root, pins)
        verify_source_closure(source_root, pins)
    except BaseException:
        _forget(namespace)
        raise
    finally:
        sys.dont_write_bytecode = previous
    return module


# ------------------------------------------------------------------------- weights


def sha256_file(path: Path, chunk_bytes: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def verify_checkpoint_file(
    path: Path, expected_bytes: int = CHECKPOINT_BYTES, expected_sha256: str = CHECKPOINT_SHA256
) -> None:
    path = Path(path)
    if not path.is_file():
        raise MintLoadError(f"checkpoint {path} does not exist")
    size = path.stat().st_size
    if size != expected_bytes:
        raise MintLoadError(f"checkpoint {path} has {size} bytes, pinned release has {expected_bytes}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise MintLoadError(f"checkpoint {path} sha256 {actual} != pinned {expected_sha256}")


def read_checkpoint_state(path: Path, mmap: bool = False) -> Dict[str, torch.Tensor]:
    """``torch.load`` with ``weights_only=True`` on CPU; there is no unsafe fallback.

    The payload must be a mapping whose ``state_dict`` entry is the non-empty
    ``str -> Tensor`` state. A flat state dict, a non-mapping payload or a missing envelope
    is refused outright, before any key is looked at; the layout is never guessed.

    This reader makes no allowlist exception: metadata that pickles an ``argparse.Namespace``
    is refused here unless the caller has allowlisted it. Only the hash-verified release path,
    :func:`_read_verified_release_state`, admits that one class.
    """
    kwargs: Dict[str, object] = {"map_location": "cpu", "weights_only": True}
    if mmap:
        kwargs["mmap"] = True
    payload = torch.load(str(path), **kwargs)
    if not isinstance(payload, Mapping) or "state_dict" not in payload:
        raise MintLoadError("checkpoint is not a mapping with a 'state_dict' envelope")
    state = payload["state_dict"]
    if not isinstance(state, Mapping) or not state:
        raise MintLoadError("checkpoint 'state_dict' is not a non-empty mapping")
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise MintLoadError(f"state dict entry {key!r} is not a str -> Tensor pair")
    return dict(state)


def _named_safe_globals(entries: Sequence[object]) -> List[Tuple[str, object]]:
    """``(allowlist name, callable)`` for each torch safe-globals entry, named the way torch names it.

    An entry is a callable (named ``module.qualname``) or a ``(callable, full path)`` pair. An entry
    torch itself cannot name is skipped: it cannot claim a name, and torch rejects it on its own.
    """
    named: List[Tuple[str, object]] = []
    for entry in entries:
        if isinstance(entry, tuple):
            if len(entry) == 2 and isinstance(entry[1], str):
                named.append((entry[1], entry[0]))
            continue
        module, qualname = getattr(entry, "__module__", None), getattr(entry, "__qualname__", None)
        if isinstance(module, str) and isinstance(qualname, str):
            named.append((f"{module}.{qualname}", entry))
    return named


def _read_verified_release_state(path: Path, mmap: bool = False) -> Dict[str, torch.Tensor]:
    """Hash-verify the pinned release checkpoint, then read it with ``argparse.Namespace`` admitted.

    The release's ``archive/data.pkl`` pickles an ``argparse.Namespace`` of run metadata beside the
    state, and ``weights_only=True`` refuses that one global. :func:`verify_checkpoint_file` runs
    first and is the only hashing of the file; nothing touches torch's allowlist or deserialises
    until it passes. Under ``_RELEASE_LOAD_LOCK`` the real ``argparse.Namespace`` is then allowlisted
    for the duration of the ordinary :func:`read_checkpoint_state` (still ``weights_only=True``, still
    CPU, one attempt, no fallback). The Namespace is constructed and ignored: the architecture comes
    from :data:`NATIVE_ARCHITECTURE` alone.

    Torch's allowlist is a process-global set, so this is careful with it: it is never cleared and
    its unrelated entries are never touched; a caller who already allows the real class under that
    exact name keeps their entry (re-adding it would have the scope's exit remove it); any other
    callable claiming ``argparse.Namespace`` is refused. The lock serialises this module's verified
    loads and nothing else -- it cannot coordinate with other code editing torch's allowlist -- and
    the file is hashed and then reopened by ``torch.load``, so a writer swapping it in between
    is not guarded against. A torch without ``get_safe_globals`` / ``safe_globals`` is refused.
    """
    path = Path(path)
    verify_checkpoint_file(path)
    serialization = torch.serialization
    missing = [name for name in ("get_safe_globals", "safe_globals") if not callable(getattr(serialization, name, None))]
    if missing:
        raise MintLoadError(
            f"loading the pinned release needs torch.serialization.get_safe_globals and safe_globals; "
            f"torch {torch.__version__} lacks {', '.join(missing)}"
        )
    with _RELEASE_LOAD_LOCK:
        claimants = [obj for name, obj in _named_safe_globals(serialization.get_safe_globals()) if name == _NAMESPACE_GLOBAL]
        impostors = [obj for obj in claimants if obj is not argparse.Namespace]
        if impostors:
            raise MintLoadError(f"torch's safe globals already map {_NAMESPACE_GLOBAL!r} to {impostors!r}, not argparse.Namespace")
        if claimants:  # the real class is already allowed under this name: use it as is, never re-add or remove it
            return read_checkpoint_state(path, mmap=mmap)
        with serialization.safe_globals([argparse.Namespace]):
            return read_checkpoint_state(path, mmap=mmap)


def strip_model_prefix(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove exactly one leading ``model.``; every key must carry it."""
    bare = [key for key in state if not key.startswith(_MODEL_PREFIX)]
    if bare:
        kind = "mixed key format" if len(bare) < len(state) else "no key carries the model. prefix"
        raise MintLoadError(f"{kind}; first unprefixed key: {bare[0]!r}")
    stripped: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        name = key[len(_MODEL_PREFIX):]
        if not name:
            raise MintLoadError(f"key {key!r} is empty after removing the prefix")
        if name in stripped:
            raise MintLoadError(f"two checkpoint keys collapse to {name!r}")
        stripped[name] = value
    return stripped


def check_state_matches(model: torch.nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    """Collect every mismatch between ``state`` and ``model.state_dict()``, then raise once."""
    expected = model.state_dict()
    problems: List[str] = []
    problems += [f"missing: {key}" for key in sorted(set(expected) - set(state))]
    problems += [f"unexpected: {key}" for key in sorted(set(state) - set(expected))]
    for key in sorted(set(expected) & set(state)):
        want, got = expected[key], state[key]
        if got.shape != want.shape:
            problems.append(f"shape {key}: {tuple(got.shape)} != {tuple(want.shape)}")
        elif got.dtype != want.dtype:
            problems.append(f"dtype {key}: {got.dtype} != {want.dtype}")
        elif got.is_floating_point() and not bool(torch.isfinite(got).all()):
            problems.append(f"non-finite values in {key}")
    tied = ("embed_tokens.weight", "lm_head.weight")
    if all(key in state for key in tied) and state[tied[0]].shape == state[tied[1]].shape:
        if not torch.equal(state[tied[0]], state[tied[1]]):
            problems.append(f"{tied[0]} and {tied[1]} differ, but the model ties them")
    if problems:
        raise MintLoadError("checkpoint does not fit the model:\n  " + "\n  ".join(problems))


def load_native_state(model: torch.nn.Module, raw_state: Mapping[str, torch.Tensor]) -> None:
    """Validate everything, strict-load, then confirm the tie is still a single Parameter."""
    state = strip_model_prefix(raw_state)
    check_state_matches(model, state)
    model.load_state_dict(state, strict=True)
    if model.lm_head.weight is not model.embed_tokens.weight:
        raise MintLoadError("lm_head.weight is no longer the same Parameter as embed_tokens.weight")


# ------------------------------------------------------------------------- adapter


def _check_native_interface(native: torch.nn.Module) -> None:
    for name, expected in (("padding_idx", PAD_IDX), ("mask_idx", MASK_IDX), ("cls_idx", CLS_IDX), ("eos_idx", EOS_IDX)):
        if getattr(native, name, None) != expected:
            raise MintLoadError(f"native {name} is {getattr(native, name, None)!r}, expected {expected}")
    for name in ("token_dropout", "use_multimer"):
        if getattr(native, name, None) is not True:
            raise MintLoadError(f"native {name} must be True")
    for name in ("num_layers", "embed_dim", "attention_heads"):
        value = getattr(native, name, None)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise MintLoadError(f"native {name} must be a positive int, got {value!r}")
    alphabet = getattr(native, "alphabet", None)
    tokens = getattr(alphabet, "all_toks", None)
    index = getattr(alphabet, "tok_to_idx", None)
    if tokens is None or index is None:
        raise MintLoadError("native alphabet must expose all_toks and tok_to_idx")
    if tuple(tokens) != NATIVE_ALPHABET or dict(index) != {tok: i for i, tok in enumerate(NATIVE_ALPHABET)}:
        raise MintLoadError("native alphabet differs from the pinned 33-token ESM-1b alphabet")


def _norm_implementations(native: torch.nn.Module) -> List[str]:
    kinds = set()
    for module in native.modules():
        if isinstance(module, torch.nn.LayerNorm) or type(module).__name__.endswith("LayerNorm"):
            kinds.add(f"{type(module).__module__}.{type(module).__qualname__}")
    return sorted(kinds)


class MintBackend(torch.nn.Module):
    """Frozen native MINT behind the :class:`ResidueBackend` contract.

    Device and dtype are fixed here, before the first forward; ``train()`` can never
    take the native model out of eval; and parameters never require grad. At
    construction and around every native call (before and after), ``_assert_frozen``
    checks exactly this and nothing more:

    * no native module is in train mode and no parameter requires grad;
    * each parameter's in-place version counter equals its value at construction;
    * every parameter and buffer is on the fixed device, and every floating-point one has
      the fixed dtype;
    * where present, the ``_cos_cached`` / ``_sin_cached`` rotary caches match that device
      and dtype.

    Buffer *contents* are neither versioned nor fingerprinted, so an in-place edit of a
    buffer that keeps its device and dtype is not detected. Gradients are not blocked
    inside ``forward``, so a trainable wrapper added later keeps working.
    """

    def __init__(
        self,
        native: torch.nn.Module,
        *,
        device: object = "cpu",
        dtype: torch.dtype = torch.float32,
        pinned_release: bool = False,
    ) -> None:
        super().__init__()
        if dtype not in SUPPORTED_DTYPES:
            raise MintLoadError(f"dtype {dtype} is not supported; use one of {SUPPORTED_DTYPES}")
        _check_native_interface(native)
        norms = _norm_implementations(native)
        unknown = [kind for kind in norms if kind != _TORCH_LAYER_NORM]
        if unknown:
            raise MintLoadError(f"only torch LayerNorm is tested, found {unknown}")
        native.to(device=torch.device(device), dtype=dtype)
        native.eval()
        for parameter in native.parameters():
            parameter.requires_grad_(False)
        self.native = native
        self._device = next(native.parameters()).device
        self._dtype = dtype
        self._pinned_release = pinned_release
        self._versions = {name: p._version for name, p in native.named_parameters()}
        self.eval()
        self._assert_frozen()

    def train(self, mode: bool = True) -> "MintBackend":
        super().train(False)
        return self

    def _assert_frozen(self) -> None:
        problems = []
        if any(module.training for module in self.native.modules()):
            problems.append("a native module is in train mode")
        for name, parameter in self.native.named_parameters():
            if parameter.requires_grad:
                problems.append(f"{name} requires grad")
            if parameter._version != self._versions.get(name):
                problems.append(f"{name} was modified in place")
        for name, tensor in list(self.native.named_parameters()) + list(self.native.named_buffers()):
            if tensor.device != self._device or (tensor.is_floating_point() and tensor.dtype != self._dtype):
                problems.append(f"{name} is {tensor.dtype} on {tensor.device}, fixed at {self._dtype} on {self._device}")
        for module in self.native.modules():
            for attribute in ("_cos_cached", "_sin_cached"):
                cached = getattr(module, attribute, None)
                if isinstance(cached, torch.Tensor) and (cached.dtype != self._dtype or cached.device != self._device):
                    problems.append(f"stale rotary cache {type(module).__name__}.{attribute}: {cached.dtype} on {cached.device}")
        if problems:
            raise MintLoadError("frozen MINT backend was disturbed:\n  " + "\n  ".join(problems))

    def _pack(
        self, views: Sequence[VisibleView]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[int]], List[float]]:
        rows: List[List[int]] = []
        chain_rows: List[List[int]] = []
        positions: List[List[int]] = []
        ratios: List[float] = []
        for view in views:
            tokens: List[int] = []
            chain_ids: List[int] = []
            where: List[int] = []
            for k, chain in enumerate(view.chains):
                tokens.append(CLS_IDX)
                for token in chain:
                    where.append(len(tokens))
                    tokens.append(MASK_IDX if token is None else _NATIVE_ID[token])
                tokens.append(EOS_IDX)
                chain_ids.extend([k] * (len(chain) + 2))
            rows.append(tokens)
            chain_rows.append(chain_ids)
            positions.append(where)
            ratios.append(tokens.count(MASK_IDX) / len(tokens))
        width = max(len(row) for row in rows)
        pad_chain = max(len(view.chains) for view in views)  # distinct from every real chain id
        tokens_t = torch.tensor([row + [PAD_IDX] * (width - len(row)) for row in rows], dtype=torch.long)
        chains_t = torch.tensor([row + [pad_chain] * (width - len(row)) for row in chain_rows], dtype=torch.long)
        return tokens_t, chains_t, positions, ratios

    def forward_batch(self, views: Sequence[VisibleView]) -> List[BackendOutput]:
        if not views:
            raise ValueError("no views given")
        self._assert_frozen()
        tokens, chain_ids, positions, ratios = self._pack(views)
        layer = self.native.num_layers
        out = self.native(tokens.to(self._device), chain_ids.to(self._device), repr_layers=[layer])
        self._assert_frozen()
        logits = out["logits"]
        final = out["representations"][layer]
        batch, width = tokens.shape
        if tuple(logits.shape) != (batch, width, len(NATIVE_ALPHABET)) or tuple(final.shape[:2]) != (batch, width):
            raise MintLoadError(f"unexpected native output shapes {tuple(logits.shape)} / {tuple(final.shape)}")
        canonical = torch.tensor(NATIVE_CANONICAL_IDS, device=logits.device)
        results = []
        for b, view in enumerate(views):
            rows = torch.tensor(positions[b], device=logits.device)
            full = logits[b, rows]  # [N, 33]
            results.append(
                check_output(
                    view,
                    BackendOutput(
                        coords=view.coords(),
                        raw_logits=full[:, canonical],
                        canonical_mass=torch.softmax(full, dim=-1)[:, canonical].sum(dim=-1),
                        features=final[b, rows],
                        mask_ratio=ratios[b],
                    ),
                )
            )
        return results

    def forward(self, view: VisibleView) -> BackendOutput:
        return self.forward_batch([view])[0]

    def describe(self) -> Dict[str, object]:
        native = self.native
        return {
            "backend": "native MINT, frozen, eval",
            "release": "pinned release" if self._pinned_release else "SOFTWARE TEST MODEL (not the pinned release)",
            "code_commit": CODE_COMMIT if self._pinned_release else None,
            "hf_revision": HF_REVISION if self._pinned_release else None,
            # The exact weights-only exception the release load was allowed; a test model had none.
            "checkpoint_safe_globals": [_NAMESPACE_GLOBAL] if self._pinned_release else [],
            "parameter_count": sum(p.numel() for p in native.parameters()),
            "num_layers": native.num_layers,
            "embed_dim": native.embed_dim,
            "attention_heads": native.attention_heads,
            "token_dropout": native.token_dropout,
            "use_multimer": native.use_multimer,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "device": str(self._device),
            "dtype": str(self._dtype),
            "norm_implementation": _norm_implementations(native),
        }


def load_pinned_mint(
    source_root: Path,
    checkpoint_path: Path,
    *,
    device: object = "cpu",
    dtype: torch.dtype = torch.float32,
    mmap: bool = False,
) -> MintBackend:
    """Build the pinned release from explicit local paths. Nothing is downloaded.

    The checkpoint goes through :func:`_read_verified_release_state` (hash-verified once, then read
    with the single ``argparse.Namespace`` exception) before the native package is imported.
    """
    verify_source_closure(Path(source_root))
    raw_state = _read_verified_release_state(Path(checkpoint_path), mmap=mmap)
    package = load_native_package(Path(source_root))
    native = package.ESM2(**NATIVE_ARCHITECTURE)
    load_native_state(native, raw_state)
    return MintBackend(native, device=device, dtype=dtype, pinned_release=True)
