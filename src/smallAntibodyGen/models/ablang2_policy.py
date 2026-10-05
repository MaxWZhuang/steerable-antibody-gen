"""The pinned native AbLang2 trunk, loaded strictly, conditioned without a rewrite.

The pinned upstream package **is** the production trunk. It is not vendored and
not reimplemented, so trunk parity is definitional rather than something a test
has to keep re-establishing. Five decisions carry the scientific weight:

* **The native double attention scaling is retained, untouched.** The pinned
  ``encoderblock.py`` scales the queries by ``head_dim ** -0.5`` *and* divides the
  scores by ``sqrt(head_dim)``, so the effective logit scale is ``1 / head_dim``
  -- ``1/24`` at ``480/20`` heads, not ``1/sqrt(24)``. Whether that was intended
  upstream is irrelevant here: the released weights were trained under it, and
  "fixing" it would mis-calibrate every attention head in the checkpoint.
  :func:`attention_scaling_record` records the measured product so the fact is a
  number in the artifacts rather than a comment.
* **The constructor is called explicitly; ``hparams.json`` is mapped, not
  splatted.** The released file carries ``pad_tkn`` where the constructor takes
  ``padding_tkn``, and it carries ``start_tkn``/``end_tkn``/``sep_tkn``, which the
  constructor does not accept at all. ``**hparams`` therefore raises, and the
  obvious repair -- filtering to the accepted names -- would silently discard
  three sentinel declarations. :func:`native_hyperparameters` maps the one
  renamed key and *verifies* the three dropped ones against the tokenizer
  constants before dropping them.
* **The output projection is tied to the input embedding by object identity**,
  captured at ``AbHead.__init__`` as ``self.weights = weights``. Two consequences
  are asserted rather than discovered: ``state_dict()`` carries that one tensor
  under two keys (so a strict load touches it twice and both must agree), and
  replacing ``aa_embed_layer`` after construction would leave ``AbHead`` pointing
  at the old tensor. This route therefore **never grows the vocabulary** -- it
  does not need to, because mask id 23 is native.
* **Conditions are injected by a forward pre-hook on the first encoder block**,
  which is the same tensor an embedding-level addition would modify, reached
  without reimplementing ``AbRep.forward``. At zero initialisation the added
  vector is exactly ``0.0``, so the conditioned forward is **bit-identical** to
  the native one -- not "identical up to fp32 noise". :meth:`ConditionedAbLang2.
  native_parity` measures that. The hook is a **stateless callable whose injection
  lives on the hooked module**, not a closure over the wrapper: ``copy.deepcopy``
  treats a function as atomic, so a closure-based hook would leave every copy
  injecting the *original* wrapper's conditions. With nonzero tables that showed up
  as a 5.7955-logit disagreement between a model and its own frozen reference while
  the two state dicts were identical. :meth:`ConditionedAbLang2.frozen_reference`
  is the explicit construction path and :meth:`reference_parity` is the check.
* **Nothing is fetched.** The package modules are loaded from explicit file paths
  under a synthetic package hierarchy, so ``ablang2/__init__.py`` -- which is
  where any convenience download would live -- is never executed. Cache
  environment variables are set before torch or transformers is imported, and
  :func:`assert_no_home_cache` checks afterwards that no cache directory was
  created in the user profile.
"""
from __future__ import annotations

import importlib
import importlib.machinery
import itertools
import math
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import torch

from ..experiments.her2_routes_serial import (
    ABLANG_END_ID,
    ABLANG_MASK_ID,
    ABLANG_PAD_ID,
    ABLANG_SEP_ID,
    ABLANG_START_ID,
    ABLANG_VOCAB_SIZE,
    ablang_canonical_ids,
)
from ..experiments.her2_runtime import require, sha256

#: Parameter count of the released checkpoint, measured by strict load. Asserted
#: so an upstream re-release is a loud failure rather than a different model.
ABLANG2_PARAMETERS = 44811050

#: The constructor's accepted keyword names, transcribed from the pinned
#: ``AbLang.__init__`` signature.
CONSTRUCTOR_KEYS = ("vocab_size", "hidden_embed_size", "n_attn_heads", "n_encoder_blocks",
                    "padding_tkn", "mask_tkn", "layer_norm_eps", "a_fn", "dropout")
#: ``hparams.json`` key -> constructor keyword. The only rename in the release.
HPARAM_RENAMES = {"pad_tkn": "padding_tkn"}
#: ``hparams.json`` keys the constructor does not accept. Each is verified against
#: the tokenizer constant it declares and then dropped, with the check recorded.
HPARAM_SENTINEL_CHECKS = {"start_tkn": ABLANG_START_ID, "end_tkn": ABLANG_END_ID,
                          "sep_tkn": ABLANG_SEP_ID}
#: Keys that are descriptive only.
HPARAM_IGNORED = ("name",)

#: Files whose bytes are part of this campaign's frozen identity even though they
#: live outside the repository. ``her2_replay_spec.source_closure`` walks
#: repository-local modules and records externals by *version*; an upstream
#: package imported from an artifact volume has no distribution metadata, so its
#: version record would be ``unresolved`` and the bytes would be unfrozen.
NATIVE_SOURCE_FILES = (
    "ablang2/__init__.py",
    "ablang2/models/ablang2/ablang.py",
    "ablang2/models/ablang2/encoderblock.py",
    "ablang2/models/ablang2/tokenizers.py",
    "ablang2/models/ablang2/vocab.py",
)

# ---------------------------------------------------------------------------
# loading the pinned package without executing its __init__
# ---------------------------------------------------------------------------

def _synthetic_package(name, directory):
    module = types.ModuleType(name)
    spec = importlib.machinery.ModuleSpec(name, None, is_package=True)
    spec.submodule_search_locations = [str(directory)]
    module.__spec__ = spec
    module.__path__ = [str(directory)]
    module.__package__ = name
    return module


def load_native_modules(package_root):
    """Import the pinned AbLang2 model modules without running any ``__init__.py``.

    ``ablang.py`` uses a relative import (``from .encoderblock import ...``), so a
    bare ``spec_from_file_location`` is not enough -- it needs a package context.
    The three parent packages are therefore registered as synthetic modules whose
    ``__path__`` points at the pinned directories. The real ``__init__.py`` files
    are hashed (they are part of the frozen identity) and never executed, which is
    what makes "no default fetch" a property of the loader rather than a promise.
    """
    base = Path(package_root)
    inner = base / "ablang2"
    require(inner.is_dir(),
            f"{inner} is not a directory. ``package_root`` is the directory that CONTAINS the "
            "``ablang2`` package, i.e. ``<artifact root>/upstream/ablang2``.")
    hierarchy = (("ablang2", inner), ("ablang2.models", inner / "models"),
                 ("ablang2.models.ablang2", inner / "models" / "ablang2"))
    created = []
    for name, directory in hierarchy:
        require(directory.is_dir(), f"Pinned native package directory {directory} is missing")
        existing = sys.modules.get(name)
        if existing is not None:
            require(getattr(existing, "__path__", None) is not None
                    and str(directory) in list(existing.__path__),
                    f"{name} is already imported from {getattr(existing, '__path__', None)}, not "
                    f"from the pinned {directory}. Two AbLang2 packages in one process would make "
                    "the loaded weights ambiguous.")
            continue
        sys.modules[name] = _synthetic_package(name, directory)
        created.append(name)
    ablang = importlib.import_module("ablang2.models.ablang2.ablang")
    encoderblock = importlib.import_module("ablang2.models.ablang2.encoderblock")
    tokenizers = importlib.import_module("ablang2.models.ablang2.tokenizers")
    return {"AbLang": ablang.AbLang, "modules": {"ablang": ablang, "encoderblock": encoderblock,
                                                 "tokenizers": tokenizers},
            "synthetic_packages": created,
            "note": ("the package __init__.py files are hashed as part of the frozen identity and "
                     "are deliberately not executed")}


def native_source_digests(package_root, *, files=NATIVE_SOURCE_FILES):
    """``{logical path: sha256}`` for the pinned native sources.

    Folded into the campaign's freeze aggregate. A missing file is refused rather
    than skipped: the closure is the thing the numbers are attributed to.
    """
    base = Path(package_root)
    digests = {}
    for relative in files:
        target = base / relative
        require(target.is_file(),
                f"{relative} is part of the frozen external source closure but is missing under "
                f"{base}. Re-run the acquire stage; an absent file changes the identity without "
                "anybody noticing.")
        digests[relative] = sha256(target)
    return digests


def native_hyperparameters(hparams):
    """Map ``hparams.json`` onto the constructor, verifying what it drops.

    Returns ``(kwargs, record)``. The record names the rename, the verified
    sentinels and anything ignored, so the mapping is auditable instead of being
    a dictionary comprehension somebody has to reverse-engineer.
    """
    document = dict(hparams)
    kwargs, renamed, verified, ignored, unexpected = {}, {}, {}, [], []
    for key, value in document.items():
        if key in HPARAM_RENAMES:
            kwargs[HPARAM_RENAMES[key]] = value
            renamed[key] = HPARAM_RENAMES[key]
        elif key in HPARAM_SENTINEL_CHECKS:
            expected = HPARAM_SENTINEL_CHECKS[key]
            require(int(value) == int(expected),
                    f"hparams declares {key}={value} and the pinned tokenizer vocabulary says "
                    f"{expected}. These two must agree before the key is dropped; a disagreement "
                    "means the serialisation this campaign asserts is wrong.")
            verified[key] = int(value)
        elif key in HPARAM_IGNORED:
            ignored.append(key)
        elif key in CONSTRUCTOR_KEYS:
            kwargs[key] = value
        else:
            unexpected.append(key)
    require(not unexpected,
            f"hparams.json carries unrecognised key(s) {sorted(unexpected)}. A released "
            "hyperparameter this loader does not understand is not silently discarded.")
    missing = [key for key in ("vocab_size", "hidden_embed_size", "n_attn_heads",
                               "n_encoder_blocks", "padding_tkn", "mask_tkn")
               if key not in kwargs]
    require(not missing, f"hparams.json is missing required constructor value(s) {missing}")
    require(int(kwargs["vocab_size"]) == ABLANG_VOCAB_SIZE,
            f"The released vocabulary size is {kwargs['vocab_size']}, not {ABLANG_VOCAB_SIZE}")
    require(int(kwargs["padding_tkn"]) == ABLANG_PAD_ID
            and int(kwargs["mask_tkn"]) == ABLANG_MASK_ID,
            "The released pad/mask token ids disagree with the pinned vocabulary")
    kwargs.setdefault("dropout", 0.0)
    require(float(kwargs["dropout"]) == 0.0,
            "This campaign trains and scores at dropout 0. A nonzero attention dropout makes a "
            "batched score differ from an unbatched one, which is the property I8 asserts.")
    return kwargs, {"kwargs": {k: (float(v) if isinstance(v, float) else v)
                               for k, v in sorted(kwargs.items())},
                    "renamed": renamed, "verified_and_dropped": verified,
                    "ignored": sorted(ignored),
                    "rule": ("hparams.json is mapped key by key. ``**hparams`` raises on "
                             "``pad_tkn`` and filtering to the accepted names would discard three "
                             "sentinel declarations unchecked.")}


@dataclass(frozen=True)
class NativeAbLang2:
    """A strict-loaded native trunk plus the evidence that it is the pinned one."""

    model: object
    hyperparameters: dict
    load_record: dict

    @property
    def hidden_size(self):
        return int(self.hyperparameters["hidden_embed_size"])

    @property
    def blocks(self):
        return list(self.model.AbRep.encoder_blocks)

    def document(self):
        return {"hyperparameters": dict(self.hyperparameters), "load": dict(self.load_record)}


def load_pinned_ablang2(weights_root, *, package_root, device="cpu", expect_parameters=True):
    """Explicit construction plus ``strict=True`` load of ``model.pt``.

    ``weights_only=True`` on the torch load: the checkpoint is a tensor container
    and nothing in it should be executing code.
    """
    import json
    weights_root = Path(weights_root)
    hparams_path = weights_root / "hparams.json"
    weights_path = weights_root / "model.pt"
    for target in (hparams_path, weights_path):
        require(target.is_file(), f"Pinned AbLang2 artifact {target} is missing")
    hparams = json.loads(hparams_path.read_text(encoding="utf-8"))
    kwargs, mapping = native_hyperparameters(hparams)
    native = load_native_modules(package_root)
    model = native["AbLang"](**kwargs)

    head = model.AbHead
    embedding = model.AbRep.aa_embed_layer
    require(head.weights is embedding.weight,
            "AbLang2's output projection is no longer the input embedding tensor. The tie is by "
            "object identity captured at AbHead.__init__; if it has been broken upstream, the "
            "vocabulary and head assumptions in this module need re-deriving, not patching.")
    duplicated = sorted(name for name, tensor in model.state_dict().items()
                        if tensor is embedding.weight or (
                            name.endswith(("AbHead.weights", "aa_embed_layer.weight"))))

    state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    state = state.get("state_dict", state) if isinstance(state, dict) and "state_dict" in state \
        else state
    model.load_state_dict(state, strict=True)
    require(bool(torch.equal(model.AbHead.weights, model.AbRep.aa_embed_layer.weight)),
            "After the strict load the tied output projection and the input embedding hold "
            "different values. state_dict() carries the shared tensor under two keys, so a "
            "checkpoint whose two copies disagree silently keeps whichever loaded last.")
    total = int(sum(p.numel() for p in model.parameters()))
    if expect_parameters:
        require(total == ABLANG2_PARAMETERS,
                f"Loaded {total} AbLang2 parameters, expected {ABLANG2_PARAMETERS}. A different "
                "count is a different release; this campaign retrains rather than coercing.")
    model.eval()
    record = {"weights_sha256": sha256(weights_path), "hparams_sha256": sha256(hparams_path),
              "parameters": total, "strict": True, "weights_only": True,
              "tied_state_dict_keys": duplicated,
              "tie_note": ("AbHead.weights and AbRep.aa_embed_layer.weight are one tensor under "
                           "two state_dict keys. named_parameters() de-duplicates it, which is why "
                           f"the count is {total} and not {total + embedding.weight.numel()}."),
              "native_source_sha256": native_source_digests(package_root),
              "mapping": mapping}
    return NativeAbLang2(model=model.to(device), hyperparameters=kwargs, load_record=record)


def attention_scaling_record(native):
    """The measured effective attention scale of the pinned trunk."""
    attention = native.blocks[0].multihead_attention
    head_dim = int(attention.head_dim)
    query_scale = float(attention.scaling)
    score_divisor = math.sqrt(head_dim)
    return {"head_dim": head_dim, "query_scale": query_scale,
            "score_divisor": float(score_divisor),
            "effective_scale": query_scale / score_divisor,
            "single_scaling_would_be": 1.0 / score_divisor,
            "ratio_to_single": (query_scale / score_divisor) * score_divisor,
            "finding": ("the pinned encoder scales queries by head_dim**-0.5 and then divides the "
                        f"scores by sqrt(head_dim), so the effective logit scale is 1/{head_dim} "
                        f"rather than 1/sqrt({head_dim}). The released weights were trained under "
                        "this, so it is retained exactly; it is recorded as a measurement because "
                        "a later reader will otherwise assume the usual single scaling.")}


# ---------------------------------------------------------------------------
# the conditioned editor
# ---------------------------------------------------------------------------

#: Distinct slot names, so two wrappers that share one trunk object do not read one
#: another's injection. They both hook the same block, and each hook must see only
#: its own wrapper's condition; a single shared slot name would make one wrapper's
#: forward apply the injection once per registered hook.
_INJECTION_SLOTS = itertools.count()


class _ConditionInjection:
    """A deepcopy-safe forward pre-hook. Its state lives on the module it hooks.

    A closure over the wrapper is the obvious implementation and the wrong one:
    ``copy.deepcopy`` copies the hooked module but treats the hook *function* as
    atomic, so the copy's hook still reads the original wrapper's injection. This
    class holds no reference to the wrapper; the injection is an attribute of the
    block under a per-wrapper slot name, so a copied block carries a copied slot
    and a second wrapper on the same block carries its own.
    """

    def __init__(self, slot):
        self.slot = str(slot)

    def __call__(self, module, args, kwargs):
        injection = getattr(module, self.slot, None)
        if injection is None:
            return None
        return (args[0] + injection, *args[1:]), kwargs


class ConditionedAbLang2(torch.nn.Module):
    """The native trunk plus zero-initialised condition tables and an edit head.

    The trunk is the pinned module and is called through its own ``forward``.
    Conditioning enters through a forward pre-hook on ``encoder_blocks[0]``, which
    receives exactly the tensor an embedding-level addition would modify. Three
    properties follow, and all three are tested:

    * at zero initialisation the injected vector is exactly ``0.0``, so the
      conditioned logits are **bit-identical** to the native ones;
    * removing the hook restores the native module unchanged, so the same object
      can serve as its own frozen zero-shot comparator;
    * a deepcopy or a :meth:`frozen_reference` carries **its own** conditioning,
      because the hook is stateless and the injection is an attribute of the
      copied block.
    """

    def __init__(self, native, condition_spec, *, pointer_positions=512,
                 activation_checkpointing=False):
        super().__init__()
        self.native = native.model
        self.hyperparameters = dict(native.hyperparameters)
        self.load_record = dict(native.load_record)
        self.condition_spec = condition_spec
        self.activation_checkpointing = bool(activation_checkpointing)
        if self.activation_checkpointing:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
            # Wrap before installing the condition hook. The wrapper's input
            # already includes the condition during recomputation, so backward
            # cannot read a subsequently cleared mutable injection attribute.
            # PyTorch's wrapper preserves the native state_dict key namespace.
            self.native.AbRep.encoder_blocks = torch.nn.ModuleList([
                checkpoint_wrapper(block, preserve_rng_state=True)
                for block in self.native.AbRep.encoder_blocks])
        self._native_factory = {"hyperparameters": dict(native.hyperparameters),
                                "pointer_positions": int(pointer_positions),
                                "activation_checkpointing": self.activation_checkpointing}
        hidden = int(self.hyperparameters["hidden_embed_size"])
        sizes = condition_spec.sizes
        self.condition_family = torch.nn.Embedding(sizes["family"], hidden)
        self.condition_bin = torch.nn.Embedding(sizes["bin"], hidden)
        self.condition_assay = torch.nn.Embedding(sizes["assay"], hidden)
        # Route 1's query IS the mask position, so no pointer is needed to say
        # WHICH site is being predicted. The table is still allocated, zeroed and
        # left unused by the default path, so the two routes can be compared with
        # and without an explicit pointer without a second architecture.
        self.pointer = torch.nn.Embedding(int(pointer_positions), hidden)
        self.use_pointer = False
        for table in (self.condition_family, self.condition_bin, self.condition_assay,
                      self.pointer):
            torch.nn.init.zeros_(table.weight)
        self.register_buffer("_canonical_ids",
                             torch.tensor(ablang_canonical_ids(), dtype=torch.long),
                             persistent=False)
        self._injection = None
        self._handle = None
        self._slot = f"_her2_condition_injection_{next(_INJECTION_SLOTS)}"
        self.enable_conditioning()

    @classmethod
    def from_release(cls, weights_root, condition_spec, *, package_root, device="cpu",
                     pointer_positions=512, activation_checkpointing=False):
        """Strict-load the release and place the WHOLE wrapper on ``device``.

        The added tables are constructed on the default device, so a factory that
        moved only the trunk would leave them behind. The wrapper is moved as one
        object and the placement is asserted rather than assumed.
        """
        native = load_pinned_ablang2(weights_root, package_root=package_root, device=device)
        model = cls(native, condition_spec, pointer_positions=pointer_positions,
                    activation_checkpointing=activation_checkpointing).to(device)
        model.assert_device_consistency(label=f"from_release(device={device!r})")
        return model

    # -- conditioning seam -------------------------------------------------
    @property
    def _hooked_block(self):
        return self.native.AbRep.encoder_blocks[0]

    def _set_injection(self, value):
        """The injection lives on the hooked block, never in a closure."""
        setattr(self._hooked_block, self._slot, value)
        self._injection = value

    def enable_conditioning(self):
        if self._handle is not None:
            return self._handle
        self._slot = f"_her2_condition_injection_{next(_INJECTION_SLOTS)}"
        self._set_injection(None)
        self._handle = self._hooked_block.register_forward_pre_hook(
            _ConditionInjection(self._slot), with_kwargs=True)
        return self._handle

    def disable_conditioning(self):
        """Detach the hook. The remaining object is the native module, unmodified."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self._set_injection(None)
        return True

    def __deepcopy__(self, memo):
        """A copy that conditions itself.

        The handle is a ``RemovableHandle`` holding a weakref, and ``copy`` treats
        a weakref as atomic -- a naively copied handle points at the ORIGINAL
        block's hook table, so the copy's ``disable_conditioning`` would unhook the
        original. The handle is therefore dropped from the copy and a fresh hook is
        registered on the copied block. Every other attribute is copied wholesale,
        so a new field cannot be silently omitted.
        """
        import copy as copy_lib
        clone = object.__new__(type(self))
        memo[id(self)] = clone
        state = {key: value for key, value in self.__dict__.items()
                 if key not in ("_handle", "_compiled_call_impl")}
        object.__setattr__(clone, "__dict__", {key: copy_lib.deepcopy(value, memo)
                                               for key, value in state.items()})
        clone._handle = None
        clone._set_injection(None)
        clone.enable_conditioning()
        return clone

    def frozen_reference(self, *, strict=True):
        """An independent, eval-mode, non-trainable copy with its own conditioning.

        Built by deepcopy and then re-verified against this model's state dict, so
        the reference is the thing it claims to be rather than the thing whose
        weights happen to match. Preference optimisation compares against this; a
        reference that silently used the policy's own conditioning would make the
        KL term read zero for the wrong reason.
        """
        import copy as copy_lib
        reference = copy_lib.deepcopy(self)
        reference.load_state_dict(self.state_dict(), strict=bool(strict))
        reference.eval()
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        require(reference._hooked_block is not self._hooked_block,
                "The frozen reference shares the trunk block with the policy, so training the "
                "policy would move the reference")
        return reference

    @torch.no_grad()
    def reference_parity(self, other, tokens, *, condition_indices=None, tolerance=0.0):
        """Two models with identical state must produce identical conditioned logits.

        This is the check the closure bug defeated: parameter comparison passed and
        the conditioned forwards differed by 5.7955 logits, because the copy's hook
        was reading the original's injection. Run it with NONZERO condition tables
        or it proves nothing.
        """
        mine = self.forward(tokens, condition_indices=condition_indices)
        theirs = other.forward(torch.as_tensor(tokens, dtype=torch.long, device=mine.device),
                               condition_indices=condition_indices)
        difference = float((mine.double() - theirs.double()).abs().max())
        tables_zero = all(bool(torch.all(table.weight == 0)) for table in
                          (self.condition_family, self.condition_bin, self.condition_assay))
        require(difference <= float(tolerance),
                f"A reference constructed from this model disagrees by {difference} logits under "
                "the same conditions. Identical state dicts and different outputs means the "
                "conditioning seam is not travelling with the copy.")
        return {"max_abs_difference": difference, "tolerance": float(tolerance),
                "condition_tables_all_zero": bool(tables_zero),
                "meaningful": not tables_zero,
                "note": ("with zero-initialised tables this passes trivially. The evidence comes "
                         "from running it after the tables have nonzero rows.")}

    def condition_vector(self, condition_indices):
        """``(B, 1, hidden)`` additive conditioning, broadcast over positions."""
        indices = torch.as_tensor(condition_indices, dtype=torch.long,
                                  device=self.condition_family.weight.device)
        require(indices.dim() == 2 and indices.shape[1] == 3,
                f"Condition indices must be (B, 3), got {tuple(indices.shape)}")
        total = (self.condition_family(indices[:, 0]) + self.condition_bin(indices[:, 1])
                 + self.condition_assay(indices[:, 2]))
        return total.unsqueeze(1)

    def forward(self, tokens, *, condition_indices=None, pointer_indices=None):
        """Full-vocabulary logits for every position, with conditioning applied."""
        tokens = torch.as_tensor(tokens, dtype=torch.long,
                                 device=self.condition_family.weight.device)
        require(tokens.dim() == 2, f"Tokens must be (B, T), got {tuple(tokens.shape)}")
        injection = None
        if condition_indices is not None:
            injection = self.condition_vector(condition_indices)
        if pointer_indices is not None:
            require(self.use_pointer,
                    "Pointer indices were supplied but this editor was built without the pointer "
                    "path enabled; silently ignoring them would report an ablation as the default")
            pointer = self.pointer(torch.as_tensor(
                pointer_indices, dtype=torch.long, device=tokens.device))
            injection = pointer if injection is None else injection + pointer
        require(self._handle is not None or injection is None,
                "Conditioning was requested but the injection hook is detached, so the condition "
                "would be silently ignored and the run would report an unconditioned model as a "
                "conditioned one.")
        self._set_injection(injection)
        try:
            logits = self.native(tokens)
        finally:
            self._set_injection(None)
        return logits

    def representation(self, tokens, *, attention_mask=None, condition_indices=None):
        """``(B, hidden)`` mean-pooled final-block state. The head's input.

        Captured with a temporary forward hook on the last encoder block rather
        than by calling a submodule directly: the pinned release exposes ``AbRep``
        and its blocks, and which of them returns a bare tensor is a property of
        that release. A hook reads whatever the block emits and cannot be wrong
        about the module's signature.

        The pooling is over the real tokens only. Averaging the padding would make
        a head's prediction depend on the longest row in its batch, and the two
        routes pad to different widths.
        """
        captured = {}

        def hook(_module, _args, output):
            captured["state"] = output[0] if isinstance(output, tuple) else output

        handle = self.native.AbRep.encoder_blocks[-1].register_forward_hook(hook)
        try:
            self.forward(tokens, condition_indices=condition_indices)
        finally:
            handle.remove()
        require("state" in captured,
                "The last AbLang2 encoder block emitted nothing during the forward, so there is no "
                "representation to pool. The block layout changed and the head would otherwise "
                "train on whatever was in the buffer.")
        state = captured["state"]
        require(state.dim() == 3,
                f"Expected (B, T, hidden) from the final block, got {tuple(state.shape)}")
        if attention_mask is None:
            return state.mean(dim=1)
        weights = torch.as_tensor(attention_mask, dtype=state.dtype,
                                  device=state.device).unsqueeze(-1)
        return (state * weights).sum(dim=1) / torch.clamp(weights.sum(dim=1), min=1.0)

    def assert_device_consistency(self, *, label="ConditionedAbLang2"):
        """Every parameter and buffer on one device. Measured, not assumed."""
        devices = sorted({str(p.device) for p in self.parameters()}
                         | {str(b.device) for b in self.buffers()})
        require(len(devices) == 1,
                f"{label}: parameters and buffers span {devices}. A trunk on CUDA with the added "
                "tables on CPU fails at the first index_select, and only on the device path.")
        return {"device": devices[0], "parameters": int(sum(1 for _ in self.parameters()))}

    def canonical_log_probs(self, tokens, query_slots, *, condition_indices=None):
        """``(B, k, 20)`` log probabilities over the canonical residues at the queries.

        Renormalised over the twenty canonical residues exactly as
        ``her2_policy.CorePolicy`` does, so a Route-1 conditional and a Route-2
        conditional are the same functional and can be compared.
        """
        logits = self.forward(tokens, condition_indices=condition_indices)
        slots = torch.as_tensor(query_slots, dtype=torch.long, device=logits.device)
        require(slots.dim() == 2 and slots.shape[0] == logits.shape[0],
                f"Query slots {tuple(slots.shape)} do not correspond to a batch of "
                f"{logits.shape[0]}")
        gathered = logits.gather(1, slots.unsqueeze(-1).expand(-1, -1, logits.shape[-1]))
        canonical = gathered[..., self._canonical_ids.to(logits.device)]
        return torch.log_softmax(canonical.float(), dim=-1)

    def native_parity(self, tokens, *, condition_indices=None):
        """Measure the conditioned forward against the untouched native forward.

        At zero-initialised tables this is exact, and the record says so with the
        measured maximum absolute difference rather than a tolerance nobody looked
        at. A nonzero table makes it a diagnostic, not an assertion.
        """
        with torch.no_grad():
            conditioned = self.forward(tokens, condition_indices=condition_indices)
            handle, self._handle = self._handle, None
            try:
                if handle is not None:
                    handle.remove()
                reference = self.native(torch.as_tensor(
                    tokens, dtype=torch.long, device=conditioned.device))
            finally:
                self.enable_conditioning()
            difference = (conditioned.double() - reference.double()).abs()
        tables_zero = all(bool(torch.all(table.weight == 0)) for table in
                          (self.condition_family, self.condition_bin, self.condition_assay,
                           self.pointer))
        return {"max_abs_difference": float(difference.max()),
                "bit_identical": bool(float(difference.max()) == 0.0),
                "tables_zero_initialised": bool(tables_zero),
                "claim": ("with zero tables the injected vector is exactly 0.0 and the sum is "
                          "exact in fp32, so bit identity is asserted rather than approximated. "
                          "Once the tables have trained this number is a diagnostic.")}

    def parameter_report(self):
        """Native, added and total parameter counts, computed not quoted."""
        added = sum(int(p.numel()) for p in (self.condition_family.weight,
                                            self.condition_bin.weight,
                                            self.condition_assay.weight, self.pointer.weight))
        native_total = int(self.load_record["parameters"])
        total = int(sum(p.numel() for p in self.parameters()))
        require(total == native_total + added,
                f"The editor holds {total} parameters and the native trunk plus the added tables "
                f"is {native_total + added}; a silently shared or duplicated table would show up "
                "exactly here")
        return {"native": native_total, "added": added, "total": total,
                "vocabulary_grown": False,
                "vocabulary_note": ("route 1 adds no vocabulary entries. The output projection is "
                                    "the input embedding by object identity, so growing the "
                                    "vocabulary after construction would leave the head pointing "
                                    "at the old tensor.")}
