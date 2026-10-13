"""Tests for the frozen native MINT adapter.

Three tiers, kept honest about what each can show:

* ``SoftwareStubNative`` tests check packing, position mapping, output selection,
  freezing and checkpoint validation. The stub is per-token and has no attention, so it
  can say nothing about native behaviour and is never used for parity.
* Verified-release-reader tests use tiny synthetic checkpoints only: ``ReleaseSeam`` stands the
  pins in for the tiny bytes and logs verification, allowlist scope and ``torch.load`` in
  order. They say nothing about the released weights, which are never touched here.
* Fake-source tests exercise the source-pin and stale-cached-bytecode refusal paths on a tiny
  throw-away tree.
* Actual-native tests (``needs_native``) run the real pinned upstream package, building a
  tiny 2-layer model from it. They skip unless ``SMALLANTIBODYGEN_MINT_SOURCE`` names the
  pinned checkout. The released weights run only with ``SMALLANTIBODYGEN_MINT_CHECKPOINT``
  and ``SMALLANTIBODYGEN_MINT_RUN_RELEASE=1`` as well.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import os
import pickle
import py_compile
import re
import sys
import threading
import types
from pathlib import Path
from typing import Dict, List

import pytest
import torch

from smallAntibodyGen.models.antibody_antigen import denoising as dn
from smallAntibodyGen.models.antibody_antigen import mint
from smallAntibodyGen.models.antibody_antigen.views import CANONICAL_RESIDUES, mask_complete

NATIVE_SOURCE = os.environ.get("SMALLANTIBODYGEN_MINT_SOURCE")
RELEASE_CHECKPOINT = os.environ.get("SMALLANTIBODYGEN_MINT_CHECKPOINT")
RUN_RELEASE = os.environ.get("SMALLANTIBODYGEN_MINT_RUN_RELEASE") == "1"
needs_native = pytest.mark.skipif(
    not NATIVE_SOURCE, reason="SMALLANTIBODYGEN_MINT_SOURCE is not set; actual-native checks skipped"
)
needs_release = pytest.mark.skipif(
    not (NATIVE_SOURCE and RELEASE_CHECKPOINT and RUN_RELEASE),
    reason="released-weights run needs SMALLANTIBODYGEN_MINT_SOURCE, _CHECKPOINT and _RUN_RELEASE=1",
)

NAMESPACE = "_smallantibodygen_native_mint_"
IDS = [mint.NATIVE_ALPHABET.index(aa) for aa in CANONICAL_RESIDUES]


def namespaces():
    return {key for key in sys.modules if key.startswith(NAMESPACE)}


# ------------------------------------------------------------- pins and constants


def test_pinned_alphabet_ids_and_release_constants_are_consistent():
    assert len(mint.NATIVE_ALPHABET) == 33 == len(set(mint.NATIVE_ALPHABET))
    specials = [mint.NATIVE_ALPHABET[i] for i in (mint.CLS_IDX, mint.PAD_IDX, mint.EOS_IDX, mint.MASK_IDX)]
    assert specials == ["<cls>", "<pad>", "<eos>", "<mask>"]
    assert list(mint.NATIVE_CANONICAL_IDS) == IDS
    assert IDS == [5, 23, 13, 9, 18, 6, 21, 12, 15, 4, 20, 17, 14, 16, 10, 8, 11, 7, 22, 19]
    assert len(mint.SOURCE_SHA256) == 11
    assert all(re.fullmatch("[0-9a-f]{64}", digest) for digest in mint.SOURCE_SHA256.values())
    assert all(re.fullmatch("[0-9a-f]{40}", sha) for sha in (mint.CODE_COMMIT, mint.HF_REVISION))
    assert re.fullmatch("[0-9a-f]{64}", mint.CHECKPOINT_SHA256) and mint.CHECKPOINT_BYTES == 3253773059
    assert dict(mint.NATIVE_ARCHITECTURE) == {
        "num_layers": 33,
        "embed_dim": 1280,
        "attention_heads": 20,
        "alphabet": "ESM-1b",
        "token_dropout": True,
        "use_multimer": True,
    }


# ----------------------------------------------------------------- software stub


class SoftwareStubNative(torch.nn.Module):
    """SOFTWARE-ONLY stand-in with the native call surface (tokens, chain_ids, repr_layers).

    Per-token and attention-free: good for packing, mapping, selection and freezing
    tests, and for nothing else. It is not native and proves no parity.
    """

    def __init__(self, seed: int = 0, tied: bool = True) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.num_layers, self.embed_dim, self.attention_heads = 2, 6, 2
        self.padding_idx, self.mask_idx, self.cls_idx, self.eos_idx = 1, 32, 0, 2
        self.token_dropout = True
        self.use_multimer = True
        self.alphabet = types.SimpleNamespace(
            all_toks=list(mint.NATIVE_ALPHABET), tok_to_idx={t: i for i, t in enumerate(mint.NATIVE_ALPHABET)}
        )
        self.embed_tokens = torch.nn.Embedding(33, 6, padding_idx=1)
        self.lm_head = torch.nn.Linear(6, 33, bias=False)
        if tied:
            self.lm_head.weight = self.embed_tokens.weight
        self.layer_norm = torch.nn.LayerNorm(6)
        self.register_buffer("inv_freq", torch.arange(3, dtype=torch.float32))
        self.calls: List[tuple] = []

    def forward(self, tokens, chain_ids, repr_layers=()):
        self.calls.append((tokens.clone(), chain_ids.clone(), list(repr_layers)))
        final = self.layer_norm(self.embed_tokens(tokens))
        return {"logits": self.lm_head(final), "representations": {layer: final for layer in repr_layers}}


def stub_backend(**kwargs):
    return mint.MintBackend(SoftwareStubNative(), **kwargs)


def test_stub_serialises_each_present_chain_with_specials_and_chain_ids():
    view, _ = mask_complete({"H": "ACD", "L": "EF", "Ag": "GHI"}, [("H", 1)])
    backend = stub_backend()
    out = backend.forward(view)
    tokens, chain_ids, layers = backend.native.calls[-1]
    assert tokens.tolist() == [[0, 5, 32, 13, 2, 0, 9, 18, 2, 0, 6, 21, 12, 2]]
    assert chain_ids.tolist() == [[0] * 5 + [1] * 4 + [2] * 5]  # the specials carry their chain's id
    assert layers == [2]
    assert out.coords == view.coords() and out.mask_ratio == 1 / 14


def test_stub_omits_an_absent_chain_entirely_and_handles_a_single_chain():
    backend = stub_backend()
    no_light, _ = mask_complete({"H": "ACD", "Ag": "GHI"}, [("H", 1)])
    backend.forward(no_light)
    tokens, chain_ids, _ = backend.native.calls[-1]
    assert tokens.tolist() == [[0, 5, 32, 13, 2, 0, 6, 21, 12, 2]]
    assert chain_ids.tolist() == [[0] * 5 + [1] * 5]
    single, _ = mask_complete({"Ag": "GHI"})
    out = backend.forward(single)
    tokens, chain_ids, _ = backend.native.calls[-1]
    assert tokens.tolist() == [[0, 6, 21, 12, 2]] and chain_ids.tolist() == [[0] * 5] and out.mask_ratio == 0.0


def test_stub_right_pads_a_mixed_batch_and_maps_positions_back():
    long_view, _ = mask_complete({"H": "ACD", "L": "EF", "Ag": "GHI"}, [("H", 1)])
    short_view, _ = mask_complete({"H": "AC"}, [("H", 0)])
    backend = stub_backend()
    outs = backend.forward_batch([long_view, short_view])
    tokens, chain_ids, _ = backend.native.calls[-1]
    assert tokens.tolist() == [[0, 5, 32, 13, 2, 0, 9, 18, 2, 0, 6, 21, 12, 2], [0, 32, 23, 2] + [1] * 10]
    assert chain_ids.tolist() == [[0] * 5 + [1] * 4 + [2] * 5, [0] * 4 + [3] * 10]  # pad id is no real chain's id
    with torch.no_grad():
        full = backend.native(tokens, chain_ids, [2])["logits"]
    assert torch.equal(outs[0].raw_logits, full[0, [1, 2, 3, 6, 7, 10, 11, 12]][:, IDS])
    assert torch.equal(outs[1].raw_logits, full[1, [1, 2]][:, IDS])
    assert outs[0].mask_ratio == 1 / 14 and outs[1].mask_ratio == 1 / 4  # padding is not counted


def test_stub_canonical_mass_comes_from_the_full_33_way_softmax():
    view, _ = mask_complete({"H": "ACD", "Ag": "GHI"}, [("H", 1)])
    backend = stub_backend()
    out = backend.forward(view)
    tokens, chain_ids, _ = backend.native.calls[-1]
    with torch.no_grad():
        full = backend.native(tokens, chain_ids, [2])["logits"][0, [1, 2, 3, 6, 7, 8]]
    assert torch.equal(out.raw_logits, full[:, IDS])
    assert torch.allclose(out.canonical_mass, torch.softmax(full, dim=-1)[:, IDS].sum(dim=-1))
    assert bool((out.canonical_mass < 1.0).all())
    # Normalising over 33 and slicing is NOT what the scorers do: the two differ by log(mass).
    assert not torch.allclose(torch.log_softmax(out.raw_logits, dim=-1), torch.log_softmax(full, dim=-1)[:, IDS])


def test_stub_backend_is_frozen_eval_and_stays_eval_when_a_parent_calls_train():
    view, _ = mask_complete({"H": "ACD"}, [("H", 1)])
    backend = stub_backend()
    assert not backend.native.training and all(not p.requires_grad for p in backend.parameters())
    torch.nn.Sequential(backend).train()
    assert not backend.training and all(not m.training for m in backend.native.modules())
    backend.forward(view)
    backend.native.train()
    with pytest.raises(mint.MintLoadError, match="train mode"):
        backend.forward(view)


def test_stub_forward_refuses_a_disturbed_backend():
    view, _ = mask_complete({"H": "ACD"}, [("H", 1)])
    modified = stub_backend()
    with torch.no_grad():
        modified.native.layer_norm.weight.add_(1.0)
    with pytest.raises(mint.MintLoadError, match="modified in place"):
        modified.forward(view)
    unfrozen = stub_backend()
    unfrozen.native.layer_norm.weight.requires_grad_(True)
    with pytest.raises(mint.MintLoadError, match="requires grad"):
        unfrozen.forward(view)
    recast = stub_backend(dtype=torch.float64)
    assert recast.forward(view).raw_logits.dtype == torch.float64
    recast.to(torch.float32)
    with pytest.raises(mint.MintLoadError, match="fixed at"):
        recast.forward(view)
    stale = stub_backend()
    stale.native.layer_norm._cos_cached = torch.zeros(2, dtype=torch.float64)
    with pytest.raises(mint.MintLoadError, match="stale rotary cache"):
        stale.forward(view)


def test_stub_refuses_unsupported_dtype_untested_norms_and_foreign_interfaces():
    with pytest.raises(mint.MintLoadError, match="dtype"):
        stub_backend(dtype=torch.float16)

    class FusedLayerNorm(torch.nn.Module):
        def forward(self, x):
            return x

    class OddLayerNorm(torch.nn.LayerNorm):
        pass

    for norm in (FusedLayerNorm(), OddLayerNorm(6)):
        stub = SoftwareStubNative()
        stub.layer_norm = norm
        with pytest.raises(mint.MintLoadError, match="LayerNorm"):
            mint.MintBackend(stub)

    def swapped(stub):
        stub.alphabet.all_toks[4], stub.alphabet.all_toks[5] = "A", "L"

    def reindexed(stub):
        stub.alphabet.tok_to_idx["L"] = 5

    for damage in (
        swapped,
        reindexed,
        lambda s: setattr(s, "mask_idx", 31),
        lambda s: setattr(s, "padding_idx", 0),
        lambda s: setattr(s, "token_dropout", False),
        lambda s: setattr(s, "use_multimer", False),
        lambda s: setattr(s, "num_layers", 0),
    ):
        stub = SoftwareStubNative()
        damage(stub)
        with pytest.raises(mint.MintLoadError):
            mint.MintBackend(stub)


def test_stub_describe_reports_actual_facts_and_labels_itself_a_software_test():
    backend = stub_backend()
    info = backend.describe()
    assert info["parameter_count"] == 210  # embedding 198 (tied head counted once) + layer norm 12
    assert info["torch"] == torch.__version__ and info["device"] == "cpu" and info["dtype"] == "torch.float32"
    assert info["norm_implementation"] == ["torch.nn.modules.normalization.LayerNorm"]
    assert "SOFTWARE TEST" in info["release"] and info["code_commit"] is None and info["hf_revision"] is None


def test_stub_weights_are_unchanged_by_forward_batch_and_completion():
    backend = stub_backend()
    before = {k: v.clone() for k, v in backend.native.state_dict().items()}
    view, _ = mask_complete({"H": "ACDEF", "L": "GHI", "Ag": "KLM"}, [("H", 1), ("L", 0)])
    backend.forward(view)
    backend.forward_batch([view, view.without_ag()])
    dn.complete(backend, view)
    after = backend.native.state_dict()
    assert before.keys() == after.keys() and all(torch.equal(before[k], after[k]) for k in before)


# --------------------------------------------------------------------- checkpoints


def prefixed(state):
    return {"model." + key: value for key, value in state.items()}


def test_strip_model_prefix_removes_exactly_one_prefix_and_rejects_other_layouts():
    x, y = torch.zeros(1), torch.ones(1)
    assert mint.strip_model_prefix({"model.model.x": x, "model.y": y}) == {"model.x": x, "y": y}
    with pytest.raises(mint.MintLoadError, match="mixed"):
        mint.strip_model_prefix({"model.x": x, "y": y})
    with pytest.raises(mint.MintLoadError, match="no key carries"):
        mint.strip_model_prefix({"x": x})
    with pytest.raises(mint.MintLoadError, match="empty"):
        mint.strip_model_prefix({"model.": x})


def test_state_validation_reports_every_class_of_mismatch_before_loading():
    stub = SoftwareStubNative()
    good = {k: v.clone() for k, v in stub.state_dict().items()}
    mint.check_state_matches(stub, good)
    nan = torch.full((6,), float("nan"))
    cases = {
        "missing: inv_freq": lambda s: s.pop("inv_freq"),  # persistent buffers are part of the layout
        "unexpected: extra.weight": lambda s: s.update({"extra.weight": torch.zeros(1)}),
        "shape layer_norm.weight": lambda s: s.update({"layer_norm.weight": torch.zeros(7)}),
        "dtype layer_norm.bias": lambda s: s.update({"layer_norm.bias": torch.zeros(6, dtype=torch.float64)}),
        "non-finite values in layer_norm.bias": lambda s: s.update({"layer_norm.bias": nan}),
        "differ": lambda s: s.update({"lm_head.weight": s["lm_head.weight"] + 1.0}),
    }
    for needle, damage in cases.items():
        state = {k: v.clone() for k, v in good.items()}
        damage(state)
        with pytest.raises(mint.MintLoadError, match=re.escape(needle)):
            mint.check_state_matches(stub, state)
    both = {k: v.clone() for k, v in good.items()}
    both.pop("inv_freq")
    both["extra.weight"] = torch.zeros(1)
    with pytest.raises(mint.MintLoadError) as caught:
        mint.check_state_matches(stub, both)
    assert "missing: inv_freq" in str(caught.value) and "unexpected: extra.weight" in str(caught.value)


def test_load_native_state_strips_the_prefix_loads_strictly_and_keeps_the_tie():
    source, target = SoftwareStubNative(seed=1), SoftwareStubNative(seed=2)
    assert not torch.equal(source.embed_tokens.weight, target.embed_tokens.weight)
    mint.load_native_state(target, prefixed(source.state_dict()))
    loaded = target.state_dict()
    assert all(torch.equal(loaded[k], v) for k, v in source.state_dict().items())
    assert target.lm_head.weight is target.embed_tokens.weight


def test_a_rejected_checkpoint_leaves_the_model_untouched_and_an_untied_model_is_refused():
    source, target = SoftwareStubNative(seed=1), SoftwareStubNative(seed=2)
    before = {k: v.clone() for k, v in target.state_dict().items()}
    broken = prefixed(source.state_dict())
    broken["model.layer_norm.weight"] = torch.zeros(7)
    with pytest.raises(mint.MintLoadError):
        mint.load_native_state(target, broken)
    assert all(torch.equal(before[k], v) for k, v in target.state_dict().items())
    with pytest.raises(mint.MintLoadError, match="same Parameter"):
        mint.load_native_state(SoftwareStubNative(seed=3, tied=False), prefixed(source.state_dict()))


class _NotAPlainTensor:
    pass


def test_read_checkpoint_state_is_weights_only_cpu_and_requires_the_state_dict_envelope(tmp_path):
    state = prefixed(SoftwareStubNative().state_dict())
    torch.save({"state_dict": state, "epoch": 3}, tmp_path / "wrapped.ckpt")
    loaded = mint.read_checkpoint_state(tmp_path / "wrapped.ckpt")
    assert set(loaded) == set(state) and all(torch.equal(loaded[k], v) for k, v in state.items())
    torch.save({"state_dict": state, "extra": _NotAPlainTensor()}, tmp_path / "unsafe.ckpt")
    with pytest.raises((pickle.UnpicklingError, RuntimeError)):
        mint.read_checkpoint_state(tmp_path / "unsafe.ckpt")  # no unsafe fallback
    # The flat mapping is a perfectly valid str -> Tensor state, so only the envelope check can refuse it.
    for payload in (state, {"weights": state}, {}, [1, 2], torch.zeros(1)):
        torch.save(payload, tmp_path / "bad.ckpt")
        with pytest.raises(mint.MintLoadError, match="'state_dict' envelope"):
            mint.read_checkpoint_state(tmp_path / "bad.ckpt")
    for envelope in ({}, [1, 2], 3):
        torch.save({"state_dict": envelope}, tmp_path / "bad.ckpt")
        with pytest.raises(mint.MintLoadError, match="non-empty mapping"):
            mint.read_checkpoint_state(tmp_path / "bad.ckpt")
    torch.save({"state_dict": {"model.x": 3}}, tmp_path / "bad.ckpt")  # entries are still validated
    with pytest.raises(mint.MintLoadError, match="str -> Tensor"):
        mint.read_checkpoint_state(tmp_path / "bad.ckpt")


def test_read_checkpoint_state_passes_weights_only_and_optional_mmap(monkeypatch, tmp_path):
    seen: Dict[str, object] = {}

    def fake_load(path, **kwargs):
        seen.clear()
        seen.update(kwargs)
        return {"state_dict": {"model.x": torch.zeros(1)}}

    monkeypatch.setattr(torch, "load", fake_load)
    mint.read_checkpoint_state(tmp_path / "x.ckpt")
    assert seen == {"map_location": "cpu", "weights_only": True}
    mint.read_checkpoint_state(tmp_path / "x.ckpt", mmap=True)
    assert seen == {"map_location": "cpu", "weights_only": True, "mmap": True}


def test_checkpoint_file_verification_checks_existence_size_and_sha256(tmp_path):
    path = tmp_path / "c.ckpt"
    path.write_bytes(b"abc")
    digest = hashlib.sha256(b"abc").hexdigest()
    mint.verify_checkpoint_file(path, 3, digest)
    with pytest.raises(mint.MintLoadError, match="bytes"):
        mint.verify_checkpoint_file(path, 4, digest)
    with pytest.raises(mint.MintLoadError, match="sha256"):
        mint.verify_checkpoint_file(path, 3, "0" * 64)
    with pytest.raises(mint.MintLoadError, match="does not exist"):
        mint.verify_checkpoint_file(tmp_path / "nope.ckpt", 3, digest)


# ------------------------------------------- verified release reader (tiny synthetic files)


class _Unrelated:
    pass


class _Elsewhere:
    pass


class _Impostor:
    pass


class _SubNamespace(argparse.Namespace):
    pass


def namespace_metadata():
    return argparse.Namespace(num_layers=99, embed_dim=7, note="never read")


def write_checkpoint(path, extra=None, state=None):
    state = prefixed(SoftwareStubNative(seed=1).state_dict()) if state is None else state
    torch.save({"state_dict": state, **(extra or {})}, path)
    return path


def with_namespace(tmp_path, name="meta.ckpt", **extra):
    return write_checkpoint(tmp_path / name, {"hyper_parameters": namespace_metadata(), **extra})


def other_thread_can_take_load_lock():
    outcome: List[bool] = []

    def attempt():
        got = mint._RELEASE_LOAD_LOCK.acquire(blocking=False)
        if got:
            mint._RELEASE_LOAD_LOCK.release()
        outcome.append(got)

    thread = threading.Thread(target=attempt)
    thread.start()
    thread.join()
    return outcome[0]


class ReleaseSeam:
    """Stands tiny synthetic bytes in for the release pins and logs what the verified reader does.

    ``events`` holds, in order, the pin check ("verify"), the allowlist scope ("scope-enter" /
    "scope-exit") and every ``torch.load`` ("load"). ``lock_free`` records, at each scope entry
    and load, whether another thread could have taken the module's load lock at that moment.
    ``registered`` is the REAL ``torch.serialization.safe_globals``, for tests that pre-register
    entries as a caller would; it is captured before the logging wrapper replaces the name.
    """

    def __init__(self, monkeypatch):
        self.events: List[str] = []
        self.load_kwargs: List[dict] = []
        self.scoped: List[list] = []
        self.lock_free: List[bool] = []
        self.pinned = None
        self.registered = real_scope = torch.serialization.safe_globals
        real_verify, real_load = mint.verify_checkpoint_file, torch.load
        seam = self

        def verify(path):
            seam.events.append("verify")
            return real_verify(path, *seam.pinned)

        def load(path, **kwargs):
            seam.events.append("load")
            seam.load_kwargs.append(kwargs)
            seam.lock_free.append(other_thread_can_take_load_lock())
            return real_load(path, **kwargs)

        class Scope:
            def __init__(self, entries):
                self.entries = list(entries)
                self.inner = real_scope(self.entries)

            def __enter__(self):
                seam.events.append("scope-enter")
                seam.scoped.append(self.entries)
                seam.lock_free.append(other_thread_can_take_load_lock())
                return self.inner.__enter__()

            def __exit__(self, *exc):
                seam.events.append("scope-exit")
                return self.inner.__exit__(*exc)

        monkeypatch.setattr(mint, "verify_checkpoint_file", verify)
        monkeypatch.setattr(torch, "load", load)
        monkeypatch.setattr(torch.serialization, "safe_globals", Scope)

    def pin(self, path):
        data = Path(path).read_bytes()
        self.pinned = (len(data), hashlib.sha256(data).hexdigest())
        return path

    @staticmethod
    def snapshot():
        return set(torch.serialization.get_safe_globals())


@pytest.fixture
def seam(monkeypatch):
    seam = ReleaseSeam(monkeypatch)
    assert argparse.Namespace not in seam.snapshot(), "argparse.Namespace is already allowlisted before the test"
    return seam


def install_fake_native(monkeypatch, seam, seed=2):
    """Replace the source check and package import so ``load_pinned_mint`` runs on the stub."""
    built: Dict[str, object] = {}

    def fake_package(source_root, *args):
        seam.events.append("package")

        def esm2(**kwargs):
            built.update(kwargs)
            return SoftwareStubNative(seed=seed)

        return types.SimpleNamespace(ESM2=esm2)

    monkeypatch.setattr(mint, "verify_source_closure", lambda *args, **kwargs: seam.events.append("source"))
    monkeypatch.setattr(mint, "load_native_package", fake_package)
    return built


def test_the_ordinary_reader_still_refuses_namespace_metadata_nobody_allowlisted(seam, tmp_path):
    path = with_namespace(tmp_path)
    before = seam.snapshot()
    with pytest.raises((pickle.UnpicklingError, RuntimeError)) as caught:
        mint.read_checkpoint_state(path)
    assert "Namespace" in str(caught.value)
    assert seam.events == ["load"] and seam.scoped == []  # no verification and no exception of its own
    assert seam.snapshot() == before


def test_the_verified_release_reader_admits_namespace_after_the_hash_inside_a_scope(seam, tmp_path):
    path = with_namespace(tmp_path)
    before = seam.snapshot()
    state = mint._read_verified_release_state(seam.pin(path))
    expected = prefixed(SoftwareStubNative(seed=1).state_dict())
    assert isinstance(state, dict) and set(state) == set(expected)
    assert all(torch.equal(state[key], value) for key, value in expected.items())
    assert seam.events == ["verify", "scope-enter", "load", "scope-exit"]
    assert seam.scoped == [[argparse.Namespace]]
    assert seam.load_kwargs == [{"map_location": "cpu", "weights_only": True}]  # one attempt, no fallback
    assert seam.snapshot() == before


def test_the_verified_release_reader_forwards_mmap_to_the_ordinary_reader(seam, monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(mint, "read_checkpoint_state", lambda path, mmap=False: seen.append((path, mmap)) or {})
    path = with_namespace(tmp_path)
    mint._read_verified_release_state(seam.pin(path), mmap=True)
    assert seen == [(path, True)]


@pytest.mark.parametrize("failure", ["bytes", "sha256", "does not exist"])
def test_a_failed_verification_precedes_the_allowlist_scope_and_torch_load(seam, tmp_path, failure):
    path = with_namespace(tmp_path)
    size = path.stat().st_size
    seam.pinned = (size + 1 if failure == "bytes" else size, "0" * 64)
    target = tmp_path / "nope.ckpt" if failure == "does not exist" else path
    before = seam.snapshot()
    with pytest.raises(mint.MintLoadError, match=failure):
        mint._read_verified_release_state(target)
    assert seam.events == ["verify"] and seam.scoped == [] and seam.load_kwargs == []
    assert seam.snapshot() == before


@pytest.mark.parametrize(
    "extra",
    [{"extra": _NotAPlainTensor()}, {"extra": types.SimpleNamespace(a=1)}, {"extra": _SubNamespace(a=1)}],
    ids=["plain-class", "types.SimpleNamespace", "Namespace-subclass"],
)
def test_any_other_metadata_class_is_still_refused_with_one_load_and_an_unchanged_allowlist(seam, tmp_path, extra):
    path = with_namespace(tmp_path, **extra)  # the admitted Namespace sits beside the refused class
    before = seam.snapshot()
    with pytest.raises((pickle.UnpicklingError, RuntimeError)):
        mint._read_verified_release_state(seam.pin(path))
    assert seam.events == ["verify", "scope-enter", "load", "scope-exit"]
    assert seam.load_kwargs == [{"map_location": "cpu", "weights_only": True}]
    assert seam.snapshot() == before


@pytest.mark.parametrize(
    "entry", [argparse.Namespace, (argparse.Namespace, "argparse.Namespace")], ids=["plain", "named"]
)
def test_a_callers_real_namespace_permission_is_used_as_is_and_survives(seam, tmp_path, entry):
    good, bad = with_namespace(tmp_path, "good.ckpt"), with_namespace(tmp_path, "bad.ckpt", extra=_NotAPlainTensor())
    baseline = seam.snapshot()
    with seam.registered([entry]):
        before = seam.snapshot()
        assert before != baseline
        mint._read_verified_release_state(seam.pin(good))
        assert seam.events == ["verify", "load"] and seam.scoped == []  # not re-added, so not removed on exit
        assert seam.snapshot() == before
        seam.events.clear()
        with pytest.raises((pickle.UnpicklingError, RuntimeError)):
            mint._read_verified_release_state(seam.pin(bad))
        assert seam.events == ["verify", "load"] and seam.snapshot() == before
    assert seam.snapshot() == baseline


def test_unrelated_allowlist_entries_survive_both_success_and_error(seam, tmp_path):
    good, bad = with_namespace(tmp_path, "good.ckpt"), with_namespace(tmp_path, "bad.ckpt", extra=_NotAPlainTensor())
    with seam.registered([_Unrelated, (_Elsewhere, "elsewhere.Name")]):
        before = seam.snapshot()
        assert {_Unrelated, (_Elsewhere, "elsewhere.Name")} <= before
        mint._read_verified_release_state(seam.pin(good))
        assert "scope-enter" in seam.events and seam.snapshot() == before
        with pytest.raises((pickle.UnpicklingError, RuntimeError)):
            mint._read_verified_release_state(seam.pin(bad))
        assert seam.snapshot() == before


IMPOSTORS = {
    "alias": lambda: [(_Impostor, "argparse.Namespace")],
    "lookalike": lambda: [type("Namespace", (), {"__module__": "argparse"})],
    "alias-beside-the-real-class": lambda: [(_Impostor, "argparse.Namespace"), argparse.Namespace],
}


@pytest.mark.parametrize("kind", sorted(IMPOSTORS))
def test_another_class_claiming_the_argparse_namespace_name_is_refused_before_any_load(seam, tmp_path, kind):
    path = with_namespace(tmp_path)
    with seam.registered(IMPOSTORS[kind]()):
        before = seam.snapshot()
        with pytest.raises(mint.MintLoadError, match="argparse.Namespace"):
            mint._read_verified_release_state(seam.pin(path))
        assert seam.events == ["verify"] and seam.scoped == [] and seam.load_kwargs == []
        assert seam.snapshot() == before


@pytest.mark.parametrize("missing", ["safe_globals", "get_safe_globals"])
def test_a_torch_without_the_scoped_allowlist_api_fails_clearly_and_never_deserialises(
    seam, monkeypatch, tmp_path, missing
):
    path = seam.pin(with_namespace(tmp_path))
    monkeypatch.delattr(torch.serialization, missing)
    with pytest.raises(mint.MintLoadError, match=f"lacks {missing}"):
        mint._read_verified_release_state(path)
    assert seam.events == ["verify"] and seam.scoped == [] and seam.load_kwargs == []


def test_verified_loads_hold_the_module_lock_around_the_scope_and_the_load(seam, tmp_path):
    good, bad = with_namespace(tmp_path, "good.ckpt"), with_namespace(tmp_path, "bad.ckpt", extra=_NotAPlainTensor())
    assert other_thread_can_take_load_lock()
    mint._read_verified_release_state(seam.pin(good))
    assert seam.lock_free == [False, False]  # at scope entry and at the load
    assert other_thread_can_take_load_lock()
    seam.lock_free.clear()
    with pytest.raises((pickle.UnpicklingError, RuntimeError)):
        mint._read_verified_release_state(seam.pin(bad))
    assert seam.lock_free == [False, False] and other_thread_can_take_load_lock()  # released after an error


def test_the_verified_reader_keeps_every_ordinary_envelope_check(seam, tmp_path):
    state = prefixed(SoftwareStubNative().state_dict())
    meta = {"hyper_parameters": namespace_metadata()}
    path = tmp_path / "envelope.ckpt"
    torch.save({"state_dict": state}, path)  # a plain envelope with no metadata at all still loads
    loaded = mint._read_verified_release_state(seam.pin(path))
    assert set(loaded) == set(state) and all(torch.equal(loaded[k], v) for k, v in state.items())
    for payload in ({**state, **meta}, {"weights": state, **meta}, dict(meta), [1, 2], torch.zeros(1)):
        torch.save(payload, path)
        with pytest.raises(mint.MintLoadError, match="'state_dict' envelope"):
            mint._read_verified_release_state(seam.pin(path))
    for envelope in ({}, [1, 2], 3):
        torch.save({"state_dict": envelope, **meta}, path)
        with pytest.raises(mint.MintLoadError, match="non-empty mapping"):
            mint._read_verified_release_state(seam.pin(path))
    torch.save({"state_dict": {"model.x": 3}, **meta}, path)
    with pytest.raises(mint.MintLoadError, match="str -> Tensor"):
        mint._read_verified_release_state(seam.pin(path))


def test_the_public_loader_verifies_once_reads_through_the_scope_and_builds_from_the_pins(seam, monkeypatch, tmp_path):
    built = install_fake_native(monkeypatch, seam)
    path = with_namespace(tmp_path)
    before = seam.snapshot()
    backend = mint.load_pinned_mint(tmp_path / "src", seam.pin(path))
    assert seam.events == ["source", "verify", "scope-enter", "load", "scope-exit", "package"]
    assert seam.events.count("verify") == 1  # the release is hashed once
    assert built == dict(mint.NATIVE_ARCHITECTURE)  # the metadata's num_layers=99 / embed_dim=7 are ignored
    loaded, source = backend.native.state_dict(), SoftwareStubNative(seed=1).state_dict()
    assert loaded.keys() == source.keys() and all(torch.equal(loaded[k], v) for k, v in source.items())
    assert backend.native.lm_head.weight is backend.native.embed_tokens.weight
    info = backend.describe()
    assert info["release"] == "pinned release" and info["checkpoint_safe_globals"] == ["argparse.Namespace"]
    assert seam.snapshot() == before


def test_the_public_loader_stops_on_a_bad_checkpoint_before_importing_the_native_package(seam, monkeypatch, tmp_path):
    install_fake_native(monkeypatch, seam)
    path = with_namespace(tmp_path)
    seam.pinned = (path.stat().st_size, "0" * 64)
    with pytest.raises(mint.MintLoadError, match="sha256"):
        mint.load_pinned_mint(tmp_path / "src", path)
    assert seam.events == ["source", "verify"]


STATE_DAMAGE = {
    "shape layer_norm.weight": lambda s: s.update({"model.layer_norm.weight": torch.zeros(7)}),
    "dtype layer_norm.bias": lambda s: s.update({"model.layer_norm.bias": torch.zeros(6, dtype=torch.float64)}),
    "differ": lambda s: s.update({"model.lm_head.weight": s["model.lm_head.weight"] + 1.0}),
}


@pytest.mark.parametrize("needle", sorted(STATE_DAMAGE))
def test_the_public_loader_still_refuses_shape_dtype_and_tie_damage_with_namespace_metadata(
    seam, monkeypatch, tmp_path, needle
):
    install_fake_native(monkeypatch, seam)
    state = prefixed(SoftwareStubNative(seed=1).state_dict())
    STATE_DAMAGE[needle](state)
    path = write_checkpoint(tmp_path / "damaged.ckpt", {"hyper_parameters": namespace_metadata()}, state)
    before = seam.snapshot()
    with pytest.raises(mint.MintLoadError, match=re.escape(needle)):
        mint.load_pinned_mint(tmp_path / "src", seam.pin(path))
    assert seam.events == ["source", "verify", "scope-enter", "load", "scope-exit", "package"]
    assert seam.snapshot() == before


def test_describe_lists_no_checkpoint_exception_for_the_software_test_seam():
    info = stub_backend().describe()
    assert info["checkpoint_safe_globals"] == []
    assert "SOFTWARE TEST" in info["release"] and info["code_commit"] is None and info["hf_revision"] is None


# ----------------------------------------------------------- source pinning (fake)


GOOD = {
    "mint/__init__.py": "from .model import VALUE\nSTAMP = VALUE\n",
    "mint/model/__init__.py": "VALUE = 7\n",
}


def write_tree(root: Path, files: Dict[str, str], crlf: bool = False) -> Dict[str, str]:
    pins = {}
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        data = text.encode()
        path.write_bytes(data.replace(b"\n", b"\r\n") if crlf else data)
        pins[relative] = hashlib.sha256(data).hexdigest()
    return pins


def test_a_valid_source_imports_under_a_private_namespace_without_side_effects(tmp_path):
    pins = write_tree(tmp_path, GOOD)
    path_before, flag_before = list(sys.path), sys.dont_write_bytecode
    first = mint.load_native_package(tmp_path, pins)
    second = mint.load_native_package(tmp_path, pins)
    try:
        assert first.STAMP == 7 and first.__name__.startswith(NAMESPACE) and first.__name__ != second.__name__
        assert sys.modules[first.__name__ + ".model"].VALUE == 7
        assert sys.path == path_before and sys.dont_write_bytecode == flag_before
        assert list(tmp_path.rglob("__pycache__")) == []
    finally:
        mint._forget(first.__name__)
        mint._forget(second.__name__)


def test_only_crlf_is_normalised_before_hashing(tmp_path):
    pins = write_tree(tmp_path, GOOD, crlf=True)
    module = mint.load_native_package(tmp_path, pins)
    mint._forget(module.__name__)
    (tmp_path / "mint/model/__init__.py").write_bytes(b"VALUE = 7\r")  # a lone CR is a real difference
    with pytest.raises(mint.MintLoadError, match="model/__init__.py"):
        mint.verify_source_closure(tmp_path, pins)


def test_a_tampered_or_missing_file_fails_before_any_source_is_executed(tmp_path):
    marker_source = "from pathlib import Path\nPath(__file__).with_name('ran.marker').write_text('x')\n"
    pins = write_tree(tmp_path, {**GOOD, "mint/__init__.py": marker_source})
    (tmp_path / "mint/model/__init__.py").write_bytes(b"VALUE = 8\n")
    before = namespaces()
    with pytest.raises(mint.MintLoadError, match="model/__init__.py"):
        mint.load_native_package(tmp_path, pins)
    assert not (tmp_path / "mint" / "ran.marker").exists() and namespaces() == before
    (tmp_path / "mint/model/__init__.py").unlink()
    with pytest.raises(mint.MintLoadError, match="missing"):
        mint.load_native_package(tmp_path, pins)
    assert not (tmp_path / "mint" / "ran.marker").exists()


def test_a_failing_import_leaves_no_namespace_behind(tmp_path):
    pins = write_tree(tmp_path, {**GOOD, "mint/model/__init__.py": "raise RuntimeError('boom')\n"})
    before = namespaces()
    with pytest.raises(RuntimeError, match="boom"):
        mint.load_native_package(tmp_path, pins)
    assert namespaces() == before


def test_an_import_outside_the_pinned_closure_is_refused(tmp_path):
    pins = write_tree(tmp_path, {**GOOD, "mint/__init__.py": "from . import extra\n"})
    (tmp_path / "mint" / "extra.py").write_text("X = 1\n")
    before = namespaces()
    with pytest.raises(mint.MintLoadError, match="outside the pinned source closure"):
        mint.load_native_package(tmp_path, pins)
    assert namespaces() == before


def test_source_that_changes_during_import_is_caught_by_the_second_hash_check(tmp_path):
    rewriting = (
        "from pathlib import Path\n"
        "from .model import VALUE\n"
        "Path(__file__).with_name('model').joinpath('__init__.py').write_text('VALUE = 8\\n')\n"
    )
    pins = write_tree(tmp_path, {**GOOD, "mint/__init__.py": rewriting})
    before = namespaces()
    with pytest.raises(mint.MintLoadError, match="model/__init__.py"):
        mint.load_native_package(tmp_path, pins)
    assert namespaces() == before


def test_unrelated_esm_and_mint_modules_are_neither_used_nor_disturbed(tmp_path, monkeypatch):
    fake_esm, fake_mint = types.ModuleType("esm"), types.ModuleType("mint")
    monkeypatch.setitem(sys.modules, "esm", fake_esm)
    monkeypatch.setitem(sys.modules, "mint", fake_mint)
    module = mint.load_native_package(tmp_path, write_tree(tmp_path, GOOD))
    try:
        assert module.STAMP == 7
        assert sys.modules["esm"] is fake_esm and sys.modules["mint"] is fake_mint
    finally:
        mint._forget(module.__name__)


def test_the_public_loader_refuses_a_wrong_source_before_importing_anything(tmp_path):
    before = namespaces()
    with pytest.raises(mint.MintLoadError, match="pinned release"):
        mint.load_pinned_mint(tmp_path, tmp_path / "mint.ckpt")
    assert namespaces() == before


# ------------------------------------------- cached bytecode beside the pinned source (fake)

STALE = "VALUE = 'wrong'\n"
CURRENT = "VALUE = 'right'\n"  # same size as STALE: a timestamp .pyc compiled from STALE still matches it


@pytest.fixture
def plain_cache_layout(monkeypatch):
    """Cache files go beside the source, whatever the environment says about a cache prefix."""
    monkeypatch.setattr(sys, "pycache_prefix", None)


class FlagsWithOptimize:
    """``sys.flags`` with only ``optimize`` changed: importlib names its cache files from that one field."""

    def __init__(self, real, optimize):
        self._real, self.optimize = real, optimize

    def __getattr__(self, name):
        return getattr(self._real, name)


@contextlib.contextmanager
def optimisation_level(monkeypatch, level):
    with monkeypatch.context() as patch:
        patch.setattr(sys, "flags", FlagsWithOptimize(sys.flags, level))
        yield


def pin_of(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def tree_state(root: Path) -> Dict[str, object]:
    """Every file (cache files included) with its bytes and mtime, and every directory: what 'unchanged' means."""
    return {
        str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns) if path.is_file() else "dir"
        for path in sorted(root.rglob("*"))
    }


def plant_stale_cache(path: Path, stale: str, current: str, **compile_args) -> str:
    """Leave ``current`` as ``path``'s source beside a cache compiled from ``stale``; importlib still trusts it.

    The two texts have the same size and the source mtime is restored after the rewrite, so the timestamp
    ``.pyc`` of the old text matches the new file. Returns the cache path.
    """
    assert len(stale) == len(current)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(stale.encode())
    cache = py_compile.compile(
        str(path), doraise=True, invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP, **compile_args
    )
    written = path.stat()
    path.write_bytes(current.encode())
    os.utime(path, ns=(written.st_atime_ns, written.st_mtime_ns))
    return cache


def value_under_plain_importlib(path: Path):
    """What ordinary importlib makes of ``path``: the control that shows a fixture really is a stale-cache hazard."""
    spec = importlib.util.spec_from_file_location("_stale_cache_control", path)
    module = importlib.util.module_from_spec(spec)
    previous, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module.VALUE


def test_a_same_size_same_mtime_rewrite_behind_a_current_cache_is_refused_not_executed(tmp_path, plain_cache_layout):
    root = tmp_path.resolve()
    source = root / "mint" / "__init__.py"
    pins = {"mint/__init__.py": pin_of(CURRENT)}
    cache = plant_stale_cache(source, STALE, CURRENT)
    assert cache == importlib.util.cache_from_source(str(source))
    assert value_under_plain_importlib(source) == "wrong"  # control: importlib runs the stale bytecode...
    mint.verify_source_closure(root, pins)  # ...although the source on disk matches its pin
    state, spaces = tree_state(root), namespaces()
    with pytest.raises(mint.MintLoadError) as caught:
        mint.load_native_package(root, pins)
    assert cache in str(caught.value) and "separately from the pinned source tree" in str(caught.value)
    assert tree_state(root) == state and namespaces() == spaces  # nothing deleted, rewritten or imported


def test_a_stale_cache_of_an_imported_child_is_refused_before_the_root_executes(tmp_path, plain_cache_layout):
    root = tmp_path.resolve()
    marker = "from pathlib import Path\nPath(__file__).with_name('ran.marker').write_text('x')\nfrom .model import VALUE\n"
    pins = write_tree(root, {**GOOD, "mint/__init__.py": marker})
    child_cache = plant_stale_cache(root / "mint" / "model" / "__init__.py", "VALUE = 6\n", GOOD["mint/model/__init__.py"])
    state, spaces = tree_state(root), namespaces()
    with pytest.raises(mint.MintLoadError) as caught:
        mint.load_native_package(root, pins)
    assert child_cache in str(caught.value)
    assert importlib.util.cache_from_source(str(root / "mint" / "__init__.py")) not in str(caught.value)
    assert not (root / "mint" / "ran.marker").exists() and namespaces() == spaces
    assert tree_state(root) == state


def test_every_current_cache_file_is_named_even_one_compiled_from_the_pinned_source(tmp_path, plain_cache_layout):
    root = tmp_path.resolve()
    pins = write_tree(root, GOOD)
    caches = [py_compile.compile(str(root / relative), doraise=True) for relative in GOOD]
    assert len(set(caches)) == 2
    state = tree_state(root)
    with pytest.raises(mint.MintLoadError) as caught:
        mint.load_native_package(root, pins)
    assert all(cache in str(caught.value) for cache in caches)
    assert tree_state(root) == state
    for cache in caches:  # the remedy is the caller's: remove the cache, then the same call works
        os.remove(cache)
    module = mint.load_native_package(root, pins)
    try:
        assert module.STAMP == 7
    finally:
        mint._forget(module.__name__)
    assert list(root.rglob("*.pyc")) == []


def test_a_broken_symlink_at_the_cache_path_is_refused_too(tmp_path, plain_cache_layout):
    root = tmp_path.resolve()
    pins = write_tree(root, GOOD)
    cache = Path(importlib.util.cache_from_source(str(root / "mint" / "__init__.py")))
    cache.parent.mkdir()
    try:
        cache.symlink_to(root / "no-such-target")
    except (OSError, NotImplementedError):
        pytest.skip("this account cannot create symlinks")
    assert cache.is_symlink() and not cache.exists()
    with pytest.raises(mint.MintLoadError) as caught:
        mint.load_native_package(root, pins)
    assert str(cache) in str(caught.value) and cache.is_symlink()


def test_a_pycache_prefix_moves_the_current_candidate_with_it(tmp_path, monkeypatch, plain_cache_layout):
    root, prefix = tmp_path.resolve() / "src", tmp_path.resolve() / "prefix"
    source = root / "mint" / "__init__.py"
    pins = {"mint/__init__.py": pin_of(CURRENT)}
    beside_source = plant_stale_cache(source, STALE, CURRENT)
    monkeypatch.setattr(sys, "pycache_prefix", str(prefix))
    state = tree_state(tmp_path.resolve())
    assert value_under_plain_importlib(source) == "right"  # control: with a prefix importlib ignores the cache beside the source
    module = mint.load_native_package(root, pins)  # so that cache is not a candidate and is not refused
    try:
        assert module.VALUE == "right"
    finally:
        mint._forget(module.__name__)
    assert tree_state(tmp_path.resolve()) == state
    under_prefix = plant_stale_cache(source, STALE, CURRENT)
    assert Path(under_prefix).is_relative_to(prefix) and under_prefix != beside_source
    assert value_under_plain_importlib(source) == "wrong"
    state = tree_state(tmp_path.resolve())
    with pytest.raises(mint.MintLoadError) as caught:
        mint.load_native_package(root, pins)
    assert under_prefix in str(caught.value) and beside_source not in str(caught.value)
    assert tree_state(tmp_path.resolve()) == state


@pytest.mark.parametrize("level", [1, 2], ids=["-O", "-OO"])
def test_the_optimisation_level_selects_which_cache_file_is_current(tmp_path, monkeypatch, plain_cache_layout, level):
    root = tmp_path.resolve()
    source = root / "mint" / "__init__.py"
    pins = {"mint/__init__.py": pin_of(CURRENT)}
    unoptimised = plant_stale_cache(
        source, STALE, CURRENT, cfile=importlib.util.cache_from_source(str(source), optimization=""), optimize=0
    )
    state = tree_state(root)
    with optimisation_level(monkeypatch, level):
        assert importlib.util.cache_from_source(str(source)) == importlib.util.cache_from_source(
            str(source), optimization=level
        )
        assert value_under_plain_importlib(source) == "right"  # control: at -O/-OO the unoptimised cache is not read
        module = mint.load_native_package(root, pins)  # so it is not a candidate and is not refused
        try:
            assert module.VALUE == "right"
        finally:
            mint._forget(module.__name__)
    assert tree_state(root) == state
    optimised = plant_stale_cache(
        source, STALE, CURRENT, cfile=importlib.util.cache_from_source(str(source), optimization=level), optimize=level
    )
    assert optimised != unoptimised
    state = tree_state(root)
    with optimisation_level(monkeypatch, level):
        assert value_under_plain_importlib(source) == "wrong"
        with pytest.raises(mint.MintLoadError) as caught:
            mint.load_native_package(root, pins)
    assert optimised in str(caught.value) and unoptimised not in str(caught.value)
    assert tree_state(root) == state


def test_cache_files_of_other_interpreters_and_empty_cache_directories_do_not_block_a_load(tmp_path, plain_cache_layout):
    root = tmp_path.resolve()
    pins = write_tree(root, GOOD)
    current = Path(importlib.util.cache_from_source(str(root / "mint" / "__init__.py")))
    foreign = current.with_name("__init__.otherimpl-0.pyc")
    foreign.parent.mkdir()
    foreign.write_bytes(b"not bytecode")
    (root / "mint" / "model" / "__pycache__").mkdir()
    state = tree_state(root)
    module = mint.load_native_package(root, pins)
    try:
        assert module.STAMP == 7
    finally:
        mint._forget(module.__name__)
    assert tree_state(root) == state


def test_without_a_cache_tag_there_is_no_candidate_and_the_load_goes_on(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    pins = write_tree(root, GOOD)

    def no_cache_tag(path, **kwargs):
        raise NotImplementedError("sys.implementation.cache_tag is None")

    monkeypatch.setattr(importlib.util, "cache_from_source", no_cache_tag)  # importlib's own loaders keep their binding
    module = mint.load_native_package(root, pins)
    try:
        assert module.STAMP == 7
    finally:
        mint._forget(module.__name__)


# ------------------------------------------------- actual pinned source, tiny model

LAYOUTS = {
    "three_chains": ({"H": "ACDEF", "L": "GHIK", "Ag": "LMNPQ"}, [("H", 1), ("H", 3), ("L", 0)]),
    "no_light": ({"H": "ACDEF", "Ag": "LMNPQ"}, [("H", 2)]),
    "two_chains": ({"H": "ACDEF", "L": "GHIKL"}, [("L", 4)]),
    "single_chain_h": ({"H": "ACDEFGH"}, [("H", 0), ("H", 6)]),
    "single_chain_ag": ({"Ag": "LMNPQ"}, []),
    "unmasked": ({"H": "ACD", "L": "GHI", "Ag": "KLM"}, []),
}


def layout_view(name):
    return mask_complete(*LAYOUTS[name])[0]


def hand_tokens(view):
    """The native serialisation restated independently of the adapter, from the alphabet itself."""
    tokens, chain_ids, where = [], [], []
    for k, role in enumerate(view.roles):
        tokens.append(mint.NATIVE_ALPHABET.index("<cls>"))
        for residue in view.chain(role):
            where.append(len(tokens))
            tokens.append(mint.NATIVE_ALPHABET.index("<mask>" if residue is None else residue))
        tokens.append(mint.NATIVE_ALPHABET.index("<eos>"))
        chain_ids += [k] * (len(view.chain(role)) + 2)
    return tokens, chain_ids, where


def run_native(native, tokens, chain_ids):
    with torch.no_grad():
        out = native(torch.tensor([tokens]), torch.tensor([chain_ids]), repr_layers=[native.num_layers])
    return out["logits"][0], out["representations"][native.num_layers][0]


@pytest.fixture(scope="module")
def native_package():
    package = mint.load_native_package(Path(NATIVE_SOURCE))
    yield package
    mint._forget(package.__name__)


@pytest.fixture
def tiny_native(native_package):
    torch.manual_seed(1234)
    return native_package.ESM2(
        num_layers=2, embed_dim=16, attention_heads=4, alphabet="ESM-1b", token_dropout=True, use_multimer=True
    )


@needs_native
def test_the_actual_pinned_source_imports_privately_and_its_hashes_hold(native_package, monkeypatch):
    mint.verify_source_closure(Path(NATIVE_SOURCE))
    assert native_package.__name__.startswith(NAMESPACE)
    assert native_package.ESM2.__module__.startswith(native_package.__name__ + ".")
    monkeypatch.setitem(sys.modules, "esm", types.ModuleType("esm"))  # an unrelated esm changes nothing
    again = mint.load_native_package(Path(NATIVE_SOURCE))
    try:
        assert again.__name__ != native_package.__name__
    finally:
        mint._forget(again.__name__)


@needs_native
@pytest.mark.parametrize("name", sorted(LAYOUTS))
def test_adapter_matches_a_direct_native_call(tiny_native, name):
    view = layout_view(name)
    backend = mint.MintBackend(tiny_native)
    tokens, chain_ids, where = hand_tokens(view)
    logits, final = run_native(tiny_native, tokens, chain_ids)
    assert logits.shape == (len(tokens), 33)
    out = backend.forward(view)
    full = logits[where]
    torch.testing.assert_close(out.raw_logits, full[:, IDS], rtol=0, atol=1e-6)
    torch.testing.assert_close(out.canonical_mass, torch.softmax(full, dim=-1)[:, IDS].sum(dim=-1), rtol=0, atol=1e-6)
    torch.testing.assert_close(out.features, final[where], rtol=0, atol=1e-6)
    assert out.coords == view.coords() and out.mask_ratio == tokens.count(32) / len(tokens)
    assert bool((out.canonical_mass < 1.0).all())


@needs_native
def test_native_token_dropout_rescales_even_in_eval_and_the_observed_ratio_is_reported(tiny_native):
    view, _ = mask_complete({"H": "ACDE", "Ag": "IKLMN"}, [("H", 1), ("H", 2)])
    backend = mint.MintBackend(tiny_native)
    tokens, chain_ids, _ = hand_tokens(view)
    assert len(tokens) == 13  # 4 + 5 residues plus <cls>/<eos> for each of the two chains
    rescaled, _ = run_native(tiny_native, tokens, chain_ids)
    tiny_native.token_dropout = False
    try:
        plain, _ = run_native(tiny_native, tokens, chain_ids)
    finally:
        tiny_native.token_dropout = True
    assert not torch.allclose(rescaled, plain, atol=1e-6)
    assert backend.forward(view).mask_ratio == 2 / 13


@needs_native
def test_a_mixed_length_padded_batch_matches_each_view_alone(tiny_native):
    views = [layout_view(n) for n in ("three_chains", "single_chain_h", "no_light", "two_chains", "single_chain_ag")]
    backend = mint.MintBackend(tiny_native)
    for view, together in zip(views, backend.forward_batch(views)):
        alone = backend.forward(view)
        assert together.coords == alone.coords and together.mask_ratio == alone.mask_ratio
        for field in ("raw_logits", "canonical_mass", "features"):
            torch.testing.assert_close(getattr(together, field), getattr(alone, field), rtol=1e-4, atol=1e-5)


@needs_native
def test_chain_id_relabelling_is_invariant_and_collapsing_the_chains_is_not(tiny_native):
    view = layout_view("three_chains")
    backend = mint.MintBackend(tiny_native)
    tokens, chain_ids, where = hand_tokens(view)
    base, _ = run_native(tiny_native, tokens, chain_ids)
    relabelled, _ = run_native(tiny_native, tokens, [{0: 2, 1: 0, 2: 1}[c] for c in chain_ids])
    collapsed, _ = run_native(tiny_native, tokens, [0] * len(chain_ids))
    torch.testing.assert_close(relabelled, base, rtol=1e-4, atol=1e-5)
    assert float((collapsed - base).abs().max()) > 1e-5
    torch.testing.assert_close(backend.forward(view).raw_logits, relabelled[where][:, IDS], rtol=1e-4, atol=1e-5)


@needs_native
def test_the_native_model_stays_frozen_eval_and_bit_identical(tiny_native):
    backend = mint.MintBackend(tiny_native)
    before = {k: v.clone() for k, v in tiny_native.state_dict().items()}
    views = [layout_view(n) for n in LAYOUTS]
    for view in views:
        backend.forward(view)
    backend.forward_batch(views)
    torch.nn.Sequential(backend).train()
    assert all(not m.training for m in tiny_native.modules())
    assert all(not p.requires_grad for p in tiny_native.parameters())
    after = tiny_native.state_dict()
    assert before.keys() == after.keys() and all(torch.equal(before[k], after[k]) for k in before)
    info = backend.describe()
    assert info["parameter_count"] == sum(p.numel() for p in tiny_native.parameters())
    assert (info["num_layers"], info["embed_dim"], info["attention_heads"]) == (2, 16, 4)
    assert info["norm_implementation"] == ["torch.nn.modules.normalization.LayerNorm"]
    assert "SOFTWARE TEST" in info["release"]


@needs_native
def test_scorers_and_completion_run_on_the_actual_native_model(tiny_native):
    sequences = {"H": "ACDEFG", "L": "HIK", "Ag": "LMNPQ"}
    masked = [("H", 1), ("L", 0), ("H", 3)]
    order = [("L", 0), ("H", 3), ("H", 1)]
    view, targets = mask_complete(sequences, masked)
    backend = mint.MintBackend(tiny_native)
    calls = []
    tiny_native.register_forward_hook(lambda module, args, output: calls.append(1))
    result = dn.sequential_log_prob(backend, view, targets, order)
    assert len(calls) == 3  # one full native forward per step
    state, manual = view, 0.0
    for coord in order:
        out = backend.forward(state)
        gold = targets.residue(coord)
        manual += float(torch.log_softmax(out.raw_logits[state.position(coord)], dim=-1)[CANONICAL_RESIDUES.index(gold)])
        state = state.reveal(coord, gold)
    assert result.log_prob.item() == pytest.approx(manual, abs=1e-5)
    assert not result.log_prob.requires_grad  # the frozen parent has nothing to differentiate
    done = dn.complete(backend, view)
    assert done.view.hidden_sites() == ()
    assert all(done.view.token(c) == view.token(c) for c in view.coords() if view.token(c) is not None)


# ---------------------------------------------- released weights (explicit opt-in)


@needs_release
def test_the_released_weights_load_and_run_one_forward():
    backend = mint.load_pinned_mint(Path(NATIVE_SOURCE), Path(RELEASE_CHECKPOINT))
    info = backend.describe()
    assert info["release"] == "pinned release" and info["code_commit"] == mint.CODE_COMMIT
    assert (info["num_layers"], info["embed_dim"], info["attention_heads"]) == (33, 1280, 20)
    view, _ = mask_complete({"H": "EVQLVESG", "Ag": "KVFGRCEL"}, [("H", 2)])
    out = backend.forward(view)
    assert out.raw_logits.shape == (16, 20) and bool(torch.isfinite(out.raw_logits).all())
