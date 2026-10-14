"""Pinned native tFold ESM-PPI extraction, without pair tensors or hidden truth.

Weights and upstream source are explicit local inputs. This module never downloads,
modifies upstream code or uses the high-level 80/10/10 MLM preprocessing helper.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch
from torch import nn

from smallAntibodyGen.models.antibody_antigen.static_context import StaticContext
from smallAntibodyGen.models.antibody_antigen.views import VisibleView

_IMPORTED_SOURCE = None


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_digest(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_sha256(path):
    """Source pins are portable across Git's LF/CRLF checkout conventions."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _normalization_identity(module, device, cache):
    """Describe the dispatch used by pinned tFold without importing optional kernels.

    Its LayerNorm class can select PyTorch or Apex internally. Inspect the forward
    function's bound globals, not sys.modules: Apex imports its CUDA extension lazily,
    and that incidental import must not change identity after the first forward.
    """
    cls = type(module)
    name = cls.__module__ + "." + cls.__qualname__
    forward = cls.forward
    backend, function, extension = "module_forward", forward, None
    flags = {}
    if name == "tfold.model.layer.layer_norm.LayerNorm":
        namespace = forward.__globals__
        flags = {"elementwise_affine": bool(module.elementwise_affine),
                 "persist_layer_norm": bool(module.persist_layer_norm),
                 "tracing": bool(torch.jit.is_tracing()), "scripting": bool(torch.jit.is_scripting())}
        if (namespace.get("fused_layer_norm") is None or torch.device(device).type != "cuda"
                or flags["tracing"] or flags["scripting"]):
            backend, function = "torch.layer_norm", namespace["F"].layer_norm
        elif flags["elementwise_affine"] and flags["persist_layer_norm"]:
            backend, function, extension = "apex.fast_layer_norm", namespace["fast_layer_norm"], "fast_layer_norm"
        else:
            key = "fused_layer_norm_affine" if flags["elementwise_affine"] else "fused_layer_norm"
            backend, function, extension = "apex." + key, namespace[key], "fused_layer_norm_cuda"
    elif isinstance(module, nn.LayerNorm) and forward is nn.LayerNorm.forward:
        backend, function = "torch.layer_norm", torch.nn.functional.layer_norm
        flags = {"elementwise_affine": bool(module.elementwise_affine)}
    if not callable(function):
        raise RuntimeError(f"selected normalization backend is unavailable: {backend}")
    key = (function, extension)
    if key not in cache:
        package = "apex" if extension is not None else function.__module__.split(".")[0]
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = None  # Source-tree Apex installs may have no distribution metadata.
        code = getattr(function, "__code__", None)
        source = Path(code.co_filename) if code is not None else None
        implementation = {"module": function.__module__, "function": function.__qualname__,
                          "package": package, "package_version": version,
                          "source_sha256": file_sha256(source) if source is not None and source.is_file() else None}
        # Some Apex releases expose FastLayerNormFN.apply, whose method body lives
        # in torch.autograd. Bind the actual Apex autograd forward as well.
        owner = getattr(function, "__self__", None)
        if isinstance(owner, type) and issubclass(owner, torch.autograd.Function):
            owner_code = getattr(owner.forward, "__code__", None)
            owner_source = Path(owner_code.co_filename) if owner_code is not None else None
            implementation["autograd_owner"] = {
                "class": owner.__module__ + "." + owner.__qualname__,
                "source_sha256": file_sha256(owner_source)
                    if owner_source is not None and owner_source.is_file() else None,
            }
        if extension is not None:
            # PathFinder bypasses sys.modules and never imports the extension. Both
            # standard Apex kernels are top-level modules, so no parent import occurs.
            spec = importlib.machinery.PathFinder.find_spec(extension)
            binary = Path(spec.origin) if spec is not None and spec.origin else None
            implementation["native_extension"] = {
                "module": extension,
                "file": binary.name if binary is not None else None,
                "sha256": file_sha256(binary) if binary is not None and binary.is_file() else None,
                "discovery": "hashed" if binary is not None and binary.is_file()
                    else "not discoverable without import",
            }
        cache[key] = implementation
    result = {"class": name, "backend": backend, "flags": flags, "implementation": cache[key]}
    if hasattr(module, "normalized_shape"):
        result["normalized_shape"] = list(module.normalized_shape)
    if hasattr(module, "eps"):
        result["eps"] = float(module.eps)
    return result


def import_native_esm2(source_root, source_pins):
    """Verify the complete Python closure before importing the ordinary upstream package."""
    global _IMPORTED_SOURCE
    root = Path(source_root).resolve(strict=True)
    expected = source_pins["files"]
    actual = {p.relative_to(root).as_posix(): p for p in (root / "tfold").rglob("*.py")}
    if not expected or set(actual) != set(expected):
        raise ValueError("tFold source inventory mismatch")
    for name, path in actual.items():
        if not path.resolve().is_relative_to(root) or source_sha256(path) != expected[name]:
            raise ValueError(f"tFold source hash mismatch: {name}")
        try:
            cached = importlib.util.cache_from_source(str(path))
        except NotImplementedError:
            cached = None
        if cached is not None and os.path.lexists(cached):
            raise RuntimeError(f"unverified cached bytecode exists for pinned source: {cached}")
    identity = (str(root), source_digest(expected))
    loaded = [name for name in sys.modules if name == "tfold" or name.startswith("tfold.")]
    if loaded and _IMPORTED_SOURCE != identity:
        raise RuntimeError("unverified or conflicting tFold package is already imported")
    old_path = list(sys.path)
    old_bytecode = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(root))
        module = importlib.import_module("tfold.model.arch.esm2")
        for name, value in list(sys.modules.items()):
            if name == "tfold" or name.startswith("tfold."):
                location = getattr(value, "__file__", None)
                if location is None or Path(location).resolve() not in {p.resolve() for p in actual.values()}:
                    raise RuntimeError(f"unexpected native module origin: {name}")
        _IMPORTED_SOURCE = identity
        return module.ESM2
    except BaseException:
        for name in list(sys.modules):
            if name not in loaded and (name == "tfold" or name.startswith("tfold.")):
                del sys.modules[name]
        raise
    finally:
        sys.path[:] = old_path
        sys.dont_write_bytecode = old_bytecode


def serialize_visible(view: VisibleView, alphabet, *, device="cpu", allow_unmasked=False):
    """Upstream BOS + concatenated chains + EOS, with genuine masks and visible-only IDs."""
    if "H" not in view.roles:
        raise ValueError("ESM-PPI context needs H")
    if not allow_unmasked and not view.hidden_sites():
        raise ValueError("ESM-PPI context needs a nonempty hidden editable union")
    # Equality groups use only visible chains. A hidden identity never determines an entity ID.
    strings = ["".join("-" if r is None else r for r in chain) for chain in view.chains]
    entities = sorted(set(strings))
    tokens, asym, enty = [alphabet.cls_idx], [0], [0]
    for chain_id, (chain, visible) in enumerate(zip(view.chains, strings), start=1):
        for residue in chain:
            tokens.append(alphabet.mask_idx if residue is None else alphabet.get_idx(residue))
            asym.append(chain_id)
            enty.append(entities.index(visible) + 1)
    tokens.append(alphabet.eos_idx)
    asym.append(0)
    enty.append(0)
    return tuple(torch.tensor([values], dtype=torch.long, device=device)
                 for values in (tokens, asym, enty))


class FrozenESMPPI(nn.Module):
    """Batch size one is intentional: native chain-relative bias mixes batch statistics."""
    def __init__(self, native_model, encoder_identity, *, max_tokens=1024,
                 compute_dtype="float32"):
        super().__init__()
        if type(max_tokens) is not int or max_tokens < 3:
            raise ValueError("max_tokens must be an integer >=3")
        from smallAntibodyGen.models.antibody_antigen.precision import validate_compute
        validate_compute(compute_dtype, next(native_model.parameters()).device)
        self.model = native_model
        self.encoder_identity = dict(encoder_identity)
        if self.encoder_identity.get("precision") != compute_dtype or any(
                p.is_floating_point() and p.dtype != torch.float32
                for p in (*self.model.parameters(), *self.model.buffers())):
            raise ValueError("frozen ESM-PPI requires float32 storage and matching compute identity")
        self.compute_dtype = compute_dtype
        self.max_tokens = max_tokens
        self.model.requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        super().train(False)
        return self

    def validate_view(self, view, *, allow_unmasked=False):
        """Cheap preflight, with no encoder inference or RNG consumption."""
        from smallAntibodyGen.models.antibody_antigen.precision import validate_compute
        if not isinstance(view, VisibleView) or not view.has_role("H"):
            raise ValueError("ESM-PPI context requires an antibody H chain")
        if not allow_unmasked and not view.hidden_sites():
            raise ValueError("ESM-PPI context needs a nonempty hidden editable union")
        if len(view.coords()) + 2 > self.max_tokens:
            raise ValueError("joint sequence exceeds context limit; no truncation is allowed")
        if self.encoder_identity.get("precision") != self.compute_dtype:
            raise ValueError("encoder precision identity drift")
        validate_compute(self.compute_dtype, next(self.model.parameters()).device)

    def runtime_identity(self):
        device = next(self.model.parameters()).device
        cache = {}
        normalizations = {
            json.dumps(_normalization_identity(module, device, cache), sort_keys=True)
            for module in self.model.modules() if "norm" in type(module).__name__.lower()
        }
        return {"encoder": dict(self.encoder_identity), "storage_dtype": "float32",
                "compute_dtype": self.compute_dtype, "native_batch_size": 1,
                "max_tokens": self.max_tokens,
                "normalization_implementations": [json.loads(item) for item in sorted(normalizations)]}

    def _run_encoder(self, view: VisibleView, *, allow_unmasked: bool = False):
        """Shared encoder forward (always eval/no_grad, autocast-isolated)."""
        from smallAntibodyGen.models.antibody_antigen.precision import compute_context
        self.validate_view(view, allow_unmasked=allow_unmasked)
        self.train(False)
        device = next(self.model.parameters()).device
        tokens, asym, enty = serialize_visible(
            view, self.model.alphabet, device=device, allow_unmasked=allow_unmasked
        )
        with torch.no_grad(), compute_context(self.compute_dtype, device):
            output = self.model(
                tokens, asym_ids=asym, enty_ids=enty,
                repr_layers=[self.model.num_layers],
                need_head_weights=False, return_contacts=False,
            )
        with torch.no_grad():
            features = output["representations"][self.model.num_layers][0, 1:-1].detach()
        return features, device

    @torch.no_grad()
    def extract(self, view: VisibleView) -> StaticContext:
        """Extract features for a view with at least one masked site; returns a CPU StaticContext."""
        features, _ = self._run_encoder(view, allow_unmasked=False)
        # Move to CPU; StaticContext.__init__ clones internally.
        return StaticContext(view, features.cpu(), self.encoder_identity)

    @torch.no_grad()
    def extract_exact(self, view: VisibleView) -> "ExactViewFeatures":
        """Extract features for any view (including zero-hidden); returns a device-resident ExactViewFeatures."""
        from smallAntibodyGen.models.antibody_antigen.exact_view_features import ExactViewFeatures
        features, _ = self._run_encoder(view, allow_unmasked=True)
        # Pass raw detached tensor; ExactViewFeatures.__init__ clones internally.
        return ExactViewFeatures(view, features, self.encoder_identity)


def load_released_encoder(source_root, source_pins, checkpoint_path, checkpoint_sha256,
                          *, device="cpu", max_tokens=1024, compute_dtype="float32"):
    """Strictly load the released 33-layer/1280-wide antibody ESM-PPI checkpoint."""
    from smallAntibodyGen.models.antibody_antigen.precision import validate_compute
    validate_compute(compute_dtype, device)
    if file_sha256(checkpoint_path) != checkpoint_sha256:
        raise ValueError("ESM-PPI checkpoint SHA256 mismatch")
    native_class = import_native_esm2(source_root, source_pins)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, dict) or set(payload) != {"model", "config"}:
        raise ValueError("unexpected ESM-PPI checkpoint envelope")
    expected_config = {"arch": "PPIModel", "attention_mode": "native", "embedding_dim": 1280,
                       "num_heads": 20, "num_layers": 33, "token_dropout": True,
                       "tokenizer": "ESM-1b", "use_crp_embeddings": True}
    config = payload["config"]
    if config == {"model": expected_config}:
        config = config["model"]
    if config != expected_config:
        raise ValueError("released ESM-PPI architecture/config mismatch")
    model = native_class(num_layers=33, embed_dim=1280, attention_heads=20,
                         token_dropout=True, use_crp_embeddings=True)
    state = payload["model"]
    expected = model.state_dict()
    if not isinstance(state, dict) or set(state) != {"model." + key for key in expected}:
        raise ValueError("ESM-PPI tensor inventory mismatch")
    translated = {}
    for key, reference in expected.items():
        value = state["model." + key]
        if (not isinstance(value, torch.Tensor) or value.shape != reference.shape or
                value.dtype != reference.dtype or not bool(torch.isfinite(value).all())):
            raise ValueError(f"invalid ESM-PPI tensor: {key}")
        translated[key] = value
    if not torch.equal(translated["embed_tokens.weight"], translated["lm_head.weight"]):
        raise ValueError("tied ESM-PPI checkpoint entries disagree")
    model.load_state_dict(translated, strict=True)
    if model.embed_tokens.weight is not model.lm_head.weight:
        raise ValueError("native ESM-PPI embedding/head parameter tie is absent")
    model.to(device)
    identity = {"source_sha256": source_digest(source_pins["files"]),
                "checkpoint_sha256": checkpoint_sha256, "extraction_layer": "layer_33",
                "precision": compute_dtype, "tokenizer": "ESM-1b", "eval_mode": "true"}
    return FrozenESMPPI(model, identity, max_tokens=max_tokens, compute_dtype=compute_dtype)
