"""Normalization identities follow actual dispatch, without GPU or released weights."""
import hashlib
import sys
import types

import pytest
import torch

from smallAntibodyGen.models.antibody_antigen import esm_ppi
from smallAntibodyGen.models.antibody_antigen.views import mask_complete


def _kernel(*args, **kwargs):
    raise AssertionError("identity inspection must not execute an optional kernel")


def native_norm(*, affine=True, persistent=False, apex=False):
    namespace = {"__name__": "tfold.model.layer.layer_norm", "torch": torch,
                 "F": torch.nn.functional, "fused_layer_norm": _kernel if apex else None,
                 "fused_layer_norm_affine": _kernel if apex else None,
                 "fast_layer_norm": _kernel if apex else None}
    exec("""
class LayerNorm(torch.nn.LayerNorm):
    def forward(self, value):
        if fused_layer_norm is None or not value.is_cuda:
            return F.layer_norm(value, self.normalized_shape, self.weight, self.bias, self.eps)
        if self.elementwise_affine:
            if self.persist_layer_norm:
                return fast_layer_norm(value, self.weight, self.bias, self.eps)
            return fused_layer_norm_affine(value, self.weight, self.bias, self.normalized_shape, self.eps)
        return fused_layer_norm(value, self.normalized_shape, self.eps)
""", namespace)
    norm = namespace["LayerNorm"](8, elementwise_affine=affine)
    norm.persist_layer_norm = persistent
    return norm


@pytest.mark.parametrize("apex,affine,persistent,device,expected", [
    (False, True, False, "cuda", "torch.layer_norm"),
    (True, True, False, "cpu", "torch.layer_norm"),
    (True, True, False, "cuda", "apex.fused_layer_norm_affine"),
    (True, False, False, "cuda", "apex.fused_layer_norm"),
    (True, True, True, "cuda", "apex.fast_layer_norm"),
])
def test_pinned_dispatch_is_explicit(apex, affine, persistent, device, expected):
    norm = native_norm(affine=affine, persistent=persistent, apex=apex)
    identity = esm_ppi._normalization_identity(norm, device, {})
    assert identity["class"] == "tfold.model.layer.layer_norm.LayerNorm"
    assert identity["backend"] == expected
    assert identity["flags"]["elementwise_affine"] is affine
    assert identity["flags"]["persist_layer_norm"] is persistent
    assert identity["normalized_shape"] == [8]
    assert identity["eps"] == norm.eps
    assert identity["implementation"]["module"]
    assert identity["implementation"]["function"]


def test_apex_availability_changes_identity_without_changing_class():
    norm = native_norm(apex=False)
    first = esm_ppi._normalization_identity(norm, "cuda", {})
    namespace = type(norm).forward.__globals__
    namespace.update(fused_layer_norm=_kernel, fused_layer_norm_affine=_kernel)
    second = esm_ppi._normalization_identity(norm, "cuda", {})
    assert first["class"] == second["class"]
    assert first != second


def test_tracing_selects_torch_even_when_apex_is_available(monkeypatch):
    monkeypatch.setattr(torch.jit, "is_tracing", lambda: True)
    identity = esm_ppi._normalization_identity(native_norm(apex=True), "cuda", {})
    assert identity["backend"] == "torch.layer_norm"
    assert identity["flags"]["tracing"] is True


def test_extension_build_hash_is_detected_without_import(tmp_path, monkeypatch):
    binary = tmp_path / "fused_layer_norm_cuda.pyd"
    binary.write_bytes(b"synthetic native build one")
    calls = []
    def locate(name):
        calls.append(name)
        return types.SimpleNamespace(origin=str(binary))
    monkeypatch.setattr(esm_ppi.importlib.machinery.PathFinder, "find_spec", locate)
    norm = native_norm(apex=True)
    first = esm_ppi._normalization_identity(norm, "cuda", {})
    extension = first["implementation"]["native_extension"]
    assert extension["module"] == "fused_layer_norm_cuda"
    assert extension["sha256"] == hashlib.sha256(binary.read_bytes()).hexdigest()
    binary.write_bytes(b"synthetic native build two")
    second = esm_ppi._normalization_identity(norm, "cuda", {})
    assert first != second
    assert calls == ["fused_layer_norm_cuda", "fused_layer_norm_cuda"]


def test_lazy_extension_import_does_not_change_identity(monkeypatch):
    norm = native_norm(apex=True)
    first = esm_ppi._normalization_identity(norm, "cuda", {})
    monkeypatch.setitem(sys.modules, "fused_layer_norm_cuda", types.ModuleType("fused_layer_norm_cuda"))
    type(norm).forward.__globals__["fused_layer_norm_cuda"] = sys.modules["fused_layer_norm_cuda"]
    second = esm_ppi._normalization_identity(norm, "cuda", {})
    assert first == second


def test_fast_autograd_apply_records_apex_owner():
    class FastNorm(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            raise AssertionError("must not execute while identifying the backend")
    norm = native_norm(apex=True, persistent=True)
    type(norm).forward.__globals__["fast_layer_norm"] = FastNorm.apply
    identity = esm_ppi._normalization_identity(norm, "cuda", {})
    implementation = identity["implementation"]
    assert implementation["package"] == "apex"
    assert implementation["autograd_owner"]["class"].endswith("FastNorm")
    assert len(implementation["autograd_owner"]["source_sha256"]) == 64


def test_backend_file_hashing_is_deduplicated_per_report(monkeypatch):
    actual = esm_ppi.file_sha256
    calls = []
    def record(path):
        calls.append(path)
        return actual(path)
    monkeypatch.setattr(esm_ppi, "file_sha256", record)
    cache = {}
    for _ in range(10):
        esm_ppi._normalization_identity(native_norm(), "cpu", cache)
    assert len(calls) == 1


class Alphabet:
    cls_idx, eos_idx, mask_idx = 0, 2, 32

    def get_idx(self, residue):
        return 4 + "ACDEFGHIKLMNPQRSTVWY".index(residue)


class Encoder(torch.nn.Module):
    alphabet = Alphabet()
    num_layers = 1

    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(33, 8)
        self.norm = native_norm()

    def forward(self, tokens, **kwargs):
        return {"representations": {1: self.norm(self.embedding(tokens))}}


def test_runtime_identity_is_stable_after_forward_and_records_limit():
    encoder = esm_ppi.FrozenESMPPI(Encoder(), {"precision": "float32"}, max_tokens=32)
    before = encoder.runtime_identity()
    view, _ = mask_complete({"H": "ACDE", "Ag": "FGH"}, [("H", 1)])
    encoder.extract_exact(view)
    assert encoder.runtime_identity() == before
    assert before["max_tokens"] == 32
    assert before["normalization_implementations"][0]["backend"] == "torch.layer_norm"
    encoder.max_tokens = 33
    assert encoder.runtime_identity() != before
    assert all(not p.requires_grad and p.grad is None for p in encoder.parameters())


def test_standard_torch_layernorm_records_package_and_source():
    identity = esm_ppi._normalization_identity(torch.nn.LayerNorm(8), "cpu", {})
    assert identity["implementation"]["package"] == "torch"
    assert identity["implementation"]["package_version"]
    assert len(identity["implementation"]["source_sha256"]) == 64
