"""Native DPLM-150M / DPLM-650M: a pure-torch ESM-2 encoder, LoRA, conditions, diffusion.

Why this is not ``EsmForDPLM``
------------------------------
The upstream class (ByteDance DPLM, Apache-2.0, ``byprot/models/dplm/modules/
dplm_modeling_esm.py``) subclasses ``EsmSelfAttention`` and calls
``transpose_for_scores`` and per-layer ``rotary_embeddings``; the installed
``transformers`` 5.12.1 has neither (rotary is one model-level module whose cos/sin are
passed in). ``EsmForDPLM.__init__`` also calls ``AutoTokenizer.from_pretrained`` on
``facebook/esm2_*``, which is a network fetch, and imports ``byprot``. So this module
re-implements the encoder the released weights describe, with **the upstream/HF state-dict
key names**, and needs neither ``transformers`` nor ``byprot`` at import time.

Sources of the derived math, kept beside the code that uses them:

* encoder layer order, query pre-scaling by ``head_dim ** -0.5`` *before* rotary with SDPA
  ``scale=1.0``, pre-attention and pre-FFN LayerNorms, ``emb_layer_norm_after``, ESM's
  erf-form GELU, and the token-dropout rescale ``(1 - 0.12) / (1 - observed)``:
  ``transformers.models.esm.modeling_esm`` and DPLM's ``ModifiedEsm*`` classes (the latter
  force ``attention_mask = input_ids.ne(pad_id)``, which is what is done here);
* diffusion corruption ``q_sample`` (mask probability ``t / T`` over the non-special,
  maskable positions), the constant/linear weights, ``get_non_special_symbol_mask``,
  ``topk_masking`` and the ``reparam-<cond>-<topk>-<schedule>`` decoder with
  ``partial_masks``: ``byprot/models/dplm/dplm.py`` and ``byprot/models/utils.py``.

Where this differs from upstream, on purpose, and where it is unverified
-----------------------------------------------------------------------
* **Sampling support is exactly the twenty canonical residues.** Upstream removes only
  ``<mask>``, ``X``, pad, cls and eos from the logits, so ``B U Z O . -``, ``<unk>`` and
  ``<null_1>`` remain samplable. Here every other id is set to ``-inf`` before sampling.
* **Chain delimiters are specials.** The pipeline's paired layout ``<VH>|<VL>`` is
  translated at ONE boundary to ``<cls> VH <eos> <eos> <cls> VL <eos>``
  (heavy-only: ``<cls> VH <eos> <eos>``). There is no upstream precedent for a paired
  antibody layout; the choice is declared, and it has one property that matters: upstream's
  ``get_non_special_symbol_mask`` excludes pad, cls and eos from corruption, so a chain
  delimiter can neither be corrupted nor generated. Positions are 1:1 with the pipeline's.
* **Nothing here has been compared with the pretrained weights.** A tiny random model is
  cross-checked against the installed ``transformers`` ESM implementation in the tests;
  a numerical equivalence claim for the pretrained checkpoints needs the real files.
* **Diffusion scores are not normalized log probabilities.** The reparameterized decoder
  scores tokens by the argmax of Gumbel-perturbed log-softmax; nothing in a trajectory is a
  density. The decode result says so instead of reporting one.
* **The reconstruction loss is canonical-restricted.** The upstream task module that turns
  ``compute_loss``'s tuple into a scalar is not part of the pinned sources, so the scalar
  here is declared: the sum over corrupted positions of ``weight(t)`` times the negative
  log probability of the true residue (renormalised over the twenty canonical residues --
  the same functional the scoring path uses), divided by the number of corrupted positions.

Precision policy
----------------
Frozen base tensors may be *stored* in fp16 (650M on GPU) and are up-cast operation by
operation, so **all arithmetic is fp32** and the only fp16 objects are resting weights.
LoRA factors, condition tables and heads are fp32 and trainable. ``.to(device)`` keeps
dtypes; :meth:`ConditionedDPLM.load_checkpoint_state` asserts they are unchanged.
"""
from __future__ import annotations

import copy
import math
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F

from ..experiments import her2_routes_dplm_pins as pins_lib
from ..experiments.her2_data import CANONICAL
from ..experiments.her2_routes_serial import (
    ABLANG_END_ID,
    ABLANG_MASK_ID,
    ABLANG_PAD_ID,
    ABLANG_RESIDUE_IDS,
    ABLANG_SEP_ID,
    ABLANG_START_ID,
    ABLANG_VOCAB_SIZE,
    HEAVY,
    LIGHT,
    canonical_index,
)
from ..experiments.her2_runtime import require

# ---------------------------------------------------------------------------
# vocabulary and the public <-> native boundary
# ---------------------------------------------------------------------------

#: The 33-entry vocabulary of the pinned ``vocab.txt``, in id order. A test compares it
#: with the pinned file and the loader refuses a file that disagrees.
DPLM_VOCAB = ("<cls>", "<pad>", "<eos>", "<unk>", "L", "A", "G", "V", "S", "E", "R", "T",
              "I", "D", "P", "K", "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z",
              "O", ".", "-", "<null_1>", "<mask>")
DPLM_VOCAB_SIZE = 33
DPLM_CLS_ID = 0
DPLM_PAD_ID = 1
DPLM_EOS_ID = 2
DPLM_UNK_ID = 3
DPLM_X_ID = 24
DPLM_MASK_ID = 32
#: Native ids of the twenty canonical residues in ``her2_data.CANONICAL`` order
#: (``ACDEFGHIKLMNPQRSTVWY``). Written out, not derived, so a changed vocabulary fails a
#: test instead of silently re-indexing every weight row.
DPLM_CANONICAL_IDS = (5, 23, 13, 9, 18, 6, 21, 12, 15, 4, 20, 17, 14, 16, 10, 8, 11, 7, 22, 19)
DPLM_RESIDUE_IDS = dict(zip(CANONICAL, DPLM_CANONICAL_IDS))


def parse_vocab(text):
    """The token list of a ``vocab.txt``: one token per line, no blanks."""
    tokens = [line.strip() for line in str(text).splitlines() if line.strip()]
    return tuple(tokens)


def _public_to_native_table():
    table = [-1] * ABLANG_VOCAB_SIZE
    table[ABLANG_START_ID] = DPLM_CLS_ID
    table[ABLANG_END_ID] = DPLM_EOS_ID
    table[ABLANG_SEP_ID] = DPLM_EOS_ID
    table[ABLANG_PAD_ID] = DPLM_PAD_ID
    table[ABLANG_MASK_ID] = DPLM_MASK_ID
    for residue, public in ABLANG_RESIDUE_IDS.items():
        table[public] = DPLM_RESIDUE_IDS[residue]
    return tuple(table)


#: Public (AbLang) id -> native id. ``-1`` marks an id with no native meaning (the AbLang
#: unknown token 24), which the boundary refuses rather than guesses.
PUBLIC_TO_NATIVE = _public_to_native_table()


def _native_to_canonical_table():
    table = [-1] * DPLM_VOCAB_SIZE
    for index, native in enumerate(DPLM_CANONICAL_IDS):
        table[native] = index
    return tuple(table)


NATIVE_TO_CANONICAL = _native_to_canonical_table()


def ablang_to_dplm(tokens):
    """Translate public (AbLang-id) tokens to native DPLM ids. **The** boundary conversion.

    Called exactly once per public call, inside :class:`ConditionedDPLM`. Pad ``21`` becomes
    ``<pad>`` ``1`` and never ``<mask>``; ``<`` becomes ``<cls>``; ``>`` and ``|`` both become
    ``<eos>``; the AbLang unknown id, and anything outside the AbLang vocabulary, is refused.
    """
    tokens = torch.as_tensor(tokens, dtype=torch.long)
    if tokens.numel():
        require(int(tokens.min()) >= 0 and int(tokens.max()) < ABLANG_VOCAB_SIZE,
                f"Public token ids must lie in [0, {ABLANG_VOCAB_SIZE}); saw "
                f"[{int(tokens.min())}, {int(tokens.max())}]")
    table = torch.tensor(PUBLIC_TO_NATIVE, dtype=torch.long, device=tokens.device)
    native = table[tokens]
    require(bool((native >= 0).all()),
            "A public token has no native DPLM meaning (the AbLang unknown id 24). Nothing is "
            "silently mapped to <unk> or X.")
    return native


def dplm_token_ids(construct, *, masked_sites=()):
    """Native ids for one construct, built directly from the DPLM table.

    Independent of :func:`ablang_to_dplm` on purpose: a test asserts the two agree token
    for token, which is what makes the boundary conversion a checked claim.
    """
    masked = set(masked_sites)
    heavy = [DPLM_MASK_ID if (HEAVY, i) in masked else DPLM_RESIDUE_IDS[r]
             for i, r in enumerate(construct.heavy)]
    light = [DPLM_MASK_ID if (LIGHT, j) in masked else DPLM_RESIDUE_IDS[r]
             for j, r in enumerate(construct.light)]
    tokens = [DPLM_CLS_ID, *heavy, DPLM_EOS_ID, DPLM_EOS_ID]
    if not construct.is_heavy_only:
        tokens += [DPLM_CLS_ID, *light, DPLM_EOS_ID]
    return tuple(tokens)


def dplm_positions(construct):
    """``{site: position}`` in the native layout. Equals ``ablang_positions`` by construction.

    ``<cls> VH <eos> <eos> <cls> VL <eos>``: heavy residue ``i`` at ``1 + i`` and light
    residue ``j`` at ``len(H) + 4 + j``; heavy-only records have no light block.
    """
    if construct.is_heavy_only:
        return {site: 1 + site[1] for site in construct.sites}
    light_offset = len(construct.heavy) + 4
    return {site: (1 if site[0] == HEAVY else light_offset) + site[1]
            for site in construct.sites}


@dataclass(frozen=True)
class NativeSerialization:
    tokens: tuple
    query_slots: tuple
    targets: tuple
    order: tuple


def dplm_serialization(construct, *, masked_sites=None, order=None):
    """Native tokens plus query slots. Every queried slot must hold ``<mask>``."""
    order = tuple(order) if order is not None else tuple(construct.sites)
    masked = tuple(masked_sites) if masked_sites is not None else tuple(construct.sites)
    require(set(order) <= set(masked),
            "Every queried site must be masked in the input; an unmasked query reads its answer")
    tokens = dplm_token_ids(construct, masked_sites=masked)
    positions = dplm_positions(construct)
    for site in order:
        require(tokens[positions[site]] == DPLM_MASK_ID,
                f"Site {site} maps to position {positions[site]}, which does not hold <mask>")
    return NativeSerialization(
        tokens=tokens, query_slots=tuple(positions[site] for site in order),
        targets=tuple(canonical_index(construct.residue_at(site)) for site in order),
        order=order)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

#: Keys of the pinned ``config.json`` whose value is fixed by this implementation. A file
#: that disagrees describes a different architecture and is refused, not adapted to.
_FIXED_CONFIG = {"architectures": ["EsmForDPLM"], "dplm_type": "dplm_esm",
                 "emb_layer_norm_before": False, "hidden_act": "gelu",
                 "position_embedding_type": "rotary", "is_folding_model": False,
                 "esmfold_config": None, "vocab_list": None, "model_type": "esm",
                 "attention_probs_dropout_prob": 0.0, "classifier_dropout": None,
                 "token_dropout": True, "vocab_size": DPLM_VOCAB_SIZE,
                 "mask_token_id": DPLM_MASK_ID, "pad_token_id": DPLM_PAD_ID}
_VARIABLE_CONFIG = ("hidden_size", "num_hidden_layers", "num_attention_heads",
                    "intermediate_size", "max_position_embeddings", "layer_norm_eps",
                    "hidden_dropout_prob")
#: Descriptive keys that carry no architecture.
_IGNORED_CONFIG = ("_name_or_path", "initializer_range", "torch_dtype", "transformers_version",
                   "use_cache")
#: ESM training hard-codes this masking ratio in the token-dropout rescale.
TOKEN_DROPOUT_MASK_RATIO_TRAIN = 0.15 * 0.8


@dataclass(frozen=True)
class DPLMConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    vocab_size: int = DPLM_VOCAB_SIZE
    layer_norm_eps: float = 1e-5
    hidden_dropout_prob: float = 0.1
    max_position_embeddings: int = 1026
    mask_token_id: int = DPLM_MASK_ID
    pad_token_id: int = DPLM_PAD_ID
    token_dropout: bool = True
    rope_theta: float = 10000.0
    initializer_range: float = 0.02

    def __post_init__(self):
        require(self.hidden_size % self.num_attention_heads == 0,
                f"hidden_size {self.hidden_size} is not a multiple of {self.num_attention_heads} "
                "heads")
        require((self.hidden_size // self.num_attention_heads) % 2 == 0,
                "Rotary embeddings need an even head dimension")
        require(self.vocab_size == DPLM_VOCAB_SIZE and self.mask_token_id == DPLM_MASK_ID
                and self.pad_token_id == DPLM_PAD_ID,
                "This implementation is bound to the 33-entry DPLM vocabulary")
        require(0.0 <= float(self.hidden_dropout_prob) < 1.0, "hidden dropout must be in [0, 1)")

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_document(cls, document, *, rope_theta=10000.0, dropout=None):
        document = dict(document)
        unknown = sorted(set(document) - set(_FIXED_CONFIG) - set(_VARIABLE_CONFIG)
                         - set(_IGNORED_CONFIG))
        require(not unknown,
                f"config.json carries unrecognised key(s) {unknown}. A released setting this "
                "loader does not understand is not silently discarded.")
        for key, expected in _FIXED_CONFIG.items():
            require(key in document, f"config.json is missing {key!r}")
            require(document[key] == expected,
                    f"config.json declares {key}={document[key]!r}; this implementation is "
                    f"defined for {expected!r} and will not adapt to another value")
        missing = [key for key in _VARIABLE_CONFIG if key not in document]
        require(not missing, f"config.json is missing {missing}")
        return cls(hidden_size=int(document["hidden_size"]),
                   num_hidden_layers=int(document["num_hidden_layers"]),
                   num_attention_heads=int(document["num_attention_heads"]),
                   intermediate_size=int(document["intermediate_size"]),
                   layer_norm_eps=float(document["layer_norm_eps"]),
                   hidden_dropout_prob=float(document["hidden_dropout_prob"]
                                             if dropout is None else dropout),
                   max_position_embeddings=int(document["max_position_embeddings"]),
                   rope_theta=float(rope_theta),
                   initializer_range=float(document.get("initializer_range", 0.02)))

    def document(self):
        return {"hidden_size": self.hidden_size, "num_hidden_layers": self.num_hidden_layers,
                "num_attention_heads": self.num_attention_heads,
                "intermediate_size": self.intermediate_size, "vocab_size": self.vocab_size,
                "layer_norm_eps": self.layer_norm_eps,
                "hidden_dropout_prob": self.hidden_dropout_prob,
                "max_position_embeddings": self.max_position_embeddings,
                "rope_theta": self.rope_theta, "token_dropout": self.token_dropout}


def check_architecture(config, pinned):
    """The parsed config must agree with the pins' architecture block. Refuses, never adapts."""
    observed = {**config.document()}
    for key, value in pinned.items():
        require(key in observed and observed[key] == value,
                f"config.json gives {key}={observed.get(key)!r} and the pins say {value!r}")
    return True


# ---------------------------------------------------------------------------
# layers (state-dict key names equal the upstream / HF names)
# ---------------------------------------------------------------------------

def esm_gelu(x):
    """ESM's own GELU. ``F.gelu`` is subtly different and the released weights want this."""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def rotary_inv_freq(head_dim, theta):
    """The analytical inverse frequencies, fp32. Legacy checkpoint ``inv_freq`` is compared to this."""
    exponent = torch.arange(0, int(head_dim), 2, dtype=torch.int64).float() / float(head_dim)
    return 1.0 / (float(theta) ** exponent)


def rotary_cos_sin(length, head_dim, theta, device, inv_freq=None):
    """``(1, 1, L, D)`` cos and sin in fp32. Positions are ``arange(L)`` for every row, so
    right-padded rows agree with their unpadded selves.

    ``inv_freq`` is the model's own (possibly checkpoint-saved) frequency buffer; when absent
    the analytical value is used.
    """
    if inv_freq is None:
        inv_freq = rotary_inv_freq(head_dim, theta)
    inv_freq = inv_freq.to(device=device, dtype=torch.float32)
    positions = torch.arange(int(length), device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos()[None, None], emb.sin()[None, None]


class DPLMLinear(nn.Linear):
    """A linear layer over storage-dtype weights, computed in the activation dtype.

    Carries an optional LoRA branch **inside** the same module, so wrapping never renames
    a base key: the base tensors stay ``...query.weight`` and the adapters are the extra
    ``...query.lora_A`` / ``...query.lora_B``. B is zero-initialised, so a freshly attached
    adapter is an exact no-op.
    """

    def __init__(self, in_features, out_features, *, bias=True, dtype=None):
        super().__init__(in_features, out_features, bias=bias, dtype=dtype)
        self.register_parameter("lora_A", None)
        self.register_parameter("lora_B", None)
        self.lora_scaling = 0.0
        self.lora_dropout = None
        self.lora_enabled = True

    def attach_lora(self, *, rank, alpha, dropout):
        require(self.lora_A is None, "This linear layer already carries a LoRA adapter")
        device = self.weight.device
        a = torch.empty(int(rank), self.in_features, dtype=torch.float32, device=device)
        if device.type != "meta":
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        self.lora_A = nn.Parameter(a)
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, int(rank), dtype=torch.float32,
                                               device=device))
        self.lora_scaling = float(alpha) / float(rank)
        self.lora_dropout = nn.Dropout(float(dropout))

    def forward(self, x):
        weight = self.weight if self.weight.dtype == x.dtype else self.weight.to(x.dtype)
        bias = self.bias
        if bias is not None and bias.dtype != x.dtype:
            bias = bias.to(x.dtype)
        out = F.linear(x, weight, bias)
        if self.lora_A is not None and self.lora_enabled:
            branch = self.lora_dropout(x)
            branch = F.linear(F.linear(branch, self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype))
            out = out + branch * self.lora_scaling
        return out


class DPLMLayerNorm(nn.LayerNorm):
    """LayerNorm over storage-dtype parameters, computed in the activation dtype."""

    def forward(self, x):
        weight = self.weight if self.weight.dtype == x.dtype else self.weight.to(x.dtype)
        bias = self.bias if self.bias.dtype == x.dtype else self.bias.to(x.dtype)
        return F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)


class _SelfAttention(nn.Module):
    def __init__(self, config, dtype):
        super().__init__()
        hidden = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.query = DPLMLinear(hidden, hidden, dtype=dtype)
        self.key = DPLMLinear(hidden, hidden, dtype=dtype)
        self.value = DPLMLinear(hidden, hidden, dtype=dtype)

    def forward(self, x, cos, sin, key_mask):
        batch, length, _ = x.shape

        def split(tensor):
            return tensor.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        query, key, value = split(self.query(x)), split(self.key(x)), split(self.value(x))
        # ESM scales the query BEFORE rotary and runs SDPA with scale=1.0; the two orders
        # differ once rotary is involved, so this is the released order.
        query = query * self.head_dim ** -0.5
        query = query * cos + rotate_half(query) * sin
        key = key * cos + rotate_half(key) * sin
        context = F.scaled_dot_product_attention(query, key, value, attn_mask=key_mask,
                                                 dropout_p=0.0, scale=1.0)
        return context.transpose(1, 2).reshape(batch, length, self.num_heads * self.head_dim)


class _SelfOutput(nn.Module):
    def __init__(self, config, dtype):
        super().__init__()
        self.dense = DPLMLinear(config.hidden_size, config.hidden_size, dtype=dtype)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


class _Attention(nn.Module):
    def __init__(self, config, dtype):
        super().__init__()
        setattr(self, "self", _SelfAttention(config, dtype))
        self.output = _SelfOutput(config, dtype)
        self.LayerNorm = DPLMLayerNorm(config.hidden_size, eps=config.layer_norm_eps, dtype=dtype)


class _Intermediate(nn.Module):
    def __init__(self, config, dtype):
        super().__init__()
        self.dense = DPLMLinear(config.hidden_size, config.intermediate_size, dtype=dtype)


class _Output(nn.Module):
    def __init__(self, config, dtype):
        super().__init__()
        self.dense = DPLMLinear(config.intermediate_size, config.hidden_size, dtype=dtype)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


class _Layer(nn.Module):
    def __init__(self, config, dtype):
        super().__init__()
        self.attention = _Attention(config, dtype)
        self.intermediate = _Intermediate(config, dtype)
        self.output = _Output(config, dtype)
        self.LayerNorm = DPLMLayerNorm(config.hidden_size, eps=config.layer_norm_eps, dtype=dtype)

    def forward(self, x, cos, sin, key_mask):
        attention = self.attention
        context = getattr(attention, "self")(attention.LayerNorm(x), cos, sin, key_mask)
        x = x + attention.output.dropout(attention.output.dense(context))
        hidden = esm_gelu(self.intermediate.dense(self.LayerNorm(x)))
        return x + self.output.dropout(self.output.dense(hidden))


class _Embeddings(nn.Module):
    def __init__(self, config, dtype):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size,
                                            padding_idx=config.pad_token_id, dtype=dtype)


class _Encoder(nn.Module):
    def __init__(self, config, dtype):
        super().__init__()
        self.layer = nn.ModuleList([_Layer(config, dtype)
                                    for _ in range(config.num_hidden_layers)])
        self.emb_layer_norm_after = DPLMLayerNorm(config.hidden_size, eps=config.layer_norm_eps,
                                                  dtype=dtype)


class _Esm(nn.Module):
    def __init__(self, config, dtype):
        super().__init__()
        self.embeddings = _Embeddings(config, dtype)
        self.encoder = _Encoder(config, dtype)


class _LMHead(nn.Module):
    def __init__(self, config, dtype):
        super().__init__()
        self.dense = DPLMLinear(config.hidden_size, config.hidden_size, dtype=dtype)
        self.layer_norm = DPLMLayerNorm(config.hidden_size, eps=config.layer_norm_eps,
                                        dtype=dtype)
        self.decoder = DPLMLinear(config.hidden_size, config.vocab_size, bias=False, dtype=dtype)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size, dtype=dtype))

    def forward(self, features):
        x = self.layer_norm(esm_gelu(self.dense(features)))
        return self.decoder(x) + (self.bias if self.bias.dtype == x.dtype
                                  else self.bias.to(x.dtype))


#: The exact module paths inside one layer that carry LoRA: Q, K, V, attention output and
#: both FFN projections. Addressed by full path, never by an unanchored suffix -- a bare
#: ``output.dense`` would match the attention output as well and double-wrap it.
LORA_TARGETS = ("attention.self.query", "attention.self.key", "attention.self.value",
                "attention.output.dense", "intermediate.dense", "output.dense")


class DPLMEncoder(nn.Module):
    """The ESM-2 trunk plus the tied LM head, with the released state-dict key names.

    Compute is always fp32. ``storage_dtype`` is the dtype of the resting base tensors.
    """

    def __init__(self, config, *, storage_dtype=torch.float32, initialize=True):
        super().__init__()
        self.config = config
        self.storage_dtype = storage_dtype
        self.esm = _Esm(config, storage_dtype)
        self.lm_head = _LMHead(config, storage_dtype)
        # One shared, non-persistent fp32 rotary buffer (every released layer stores the same
        # values). Analytical by default; a pinned load replaces it with the saved frequencies.
        self.register_buffer("rotary_inv_freq",
                             rotary_inv_freq(config.head_dim, config.rope_theta), persistent=False)
        self.tie_weights()
        if initialize:
            self.initialize_weights()

    def set_rotary_inv_freq(self, values):
        """Carry saved frequencies into the forward. Shape-checked; kept fp32 on its device."""
        expected = (int(self.config.head_dim) // 2,)
        require(tuple(values.shape) == expected,
                f"rotary_inv_freq has shape {tuple(values.shape)}, expected {expected}")
        device = self.rotary_inv_freq.device
        if device.type == "meta":  # built on meta: materialize where the values already live
            device = values.device
        self._buffers["rotary_inv_freq"] = values.detach().to(device=device, dtype=torch.float32,
                                                              copy=True)

    def _apply(self, fn, *args, **kwargs):
        # ``.to(dtype)`` / ``.half()`` would round the frequencies to the storage dtype; they are
        # always fp32, so re-point the buffer at the exact saved values on the moved device.
        saved = self._buffers.get("rotary_inv_freq")
        result = super()._apply(fn, *args, **kwargs)
        if saved is not None:
            moved = self._buffers["rotary_inv_freq"]
            self._buffers["rotary_inv_freq"] = saved.to(device=moved.device, dtype=torch.float32)
        return result

    def tie_weights(self):
        """The output projection IS the input embedding, by object identity."""
        self.lm_head.decoder.weight = self.esm.embeddings.word_embeddings.weight

    @torch.no_grad()
    def initialize_weights(self, std=None):
        std = float(self.config.initializer_range if std is None else std)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        # The embedding is initialised last: the tied output projection is the same tensor,
        # and its Linear initialisation above would otherwise re-randomise the padding row.
        embedding = self.esm.embeddings.word_embeddings
        nn.init.normal_(embedding.weight, mean=0.0, std=std)
        embedding.weight[embedding.padding_idx].zero_()
        nn.init.zeros_(self.lm_head.bias)

    def freeze_base(self):
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def attach_lora(self, *, rank, alpha, dropout):
        """LoRA on exactly :data:`LORA_TARGETS` of every layer. Returns the wrapped names."""
        wrapped = []
        for index in range(self.config.num_hidden_layers):
            for target in LORA_TARGETS:
                name = f"esm.encoder.layer.{index}.{target}"
                module = self.get_submodule(name)
                require(isinstance(module, DPLMLinear), f"{name} is not a DPLMLinear")
                module.attach_lora(rank=rank, alpha=alpha, dropout=dropout)
                wrapped.append(name)
        return wrapped

    def lora_modules(self):
        return {name: module for name, module in self.named_modules()
                if isinstance(module, DPLMLinear) and module.lora_A is not None}

    def forward(self, input_ids, *, condition=None, activation_checkpointing=False):
        """Native ids ``(B, T)`` -> ``{"logits": (B, T, 33), "last_hidden_state": (B, T, H)}``.

        ``condition`` is an optional ``(B, 1, H)`` fp32 vector added to the embedding output,
        zeroed at padding. The attention mask is derived from the ids, as DPLM does.
        """
        ids = input_ids
        require(ids.dim() == 2, f"Native ids must be (B, T), got {tuple(ids.shape)}")
        length = int(ids.shape[1])
        limit = int(self.config.max_position_embeddings)
        require(length <= limit,
                f"A sequence of {length} tokens exceeds the model's {limit} positions. Nothing is "
                "truncated: a dropped light chain would leave the paired-context claim false.")
        attention = ids.ne(self.config.pad_token_id)
        require(bool(attention.any(dim=1).all()), "A row of only padding has no tokens to encode")
        embeddings = self.esm.embeddings.word_embeddings(ids).to(torch.float32)
        if self.config.token_dropout:
            is_mask = ids.eq(self.config.mask_token_id)
            embeddings = embeddings.masked_fill(is_mask.unsqueeze(-1), 0.0)
            observed = is_mask.sum(-1).float() / attention.sum(-1).float()
            embeddings = (embeddings * (1.0 - TOKEN_DROPOUT_MASK_RATIO_TRAIN)
                          / (1.0 - observed)[:, None, None])
        weight = attention.unsqueeze(-1).to(embeddings.dtype)
        hidden = embeddings * weight
        if condition is not None:
            hidden = hidden + condition.to(hidden.dtype) * weight
        cos, sin = rotary_cos_sin(length, self.config.head_dim, self.config.rope_theta,
                                  ids.device, inv_freq=self.rotary_inv_freq)
        key_mask = attention[:, None, None, :]
        for layer in self.esm.encoder.layer:
            if activation_checkpointing and self.training and torch.is_grad_enabled():
                hidden = torch.utils.checkpoint.checkpoint(
                    layer, hidden, cos, sin, key_mask, use_reentrant=False)
            else:
                hidden = layer(hidden, cos, sin, key_mask)
        hidden = self.esm.encoder.emb_layer_norm_after(hidden)
        return {"logits": self.lm_head(hidden), "last_hidden_state": hidden}


# ---------------------------------------------------------------------------
# the strict inventory
# ---------------------------------------------------------------------------

_LAYER_PARAMETERS = ("attention.LayerNorm", "attention.self.query", "attention.self.key",
                     "attention.self.value", "attention.output.dense", "intermediate.dense",
                     "output.dense", "LayerNorm")


def expected_state_shapes(config):
    """``{key: shape}`` of every learned tensor the config implies, derived from the config.

    Independent of the module tree; a test asserts it equals the constructed model's
    ``state_dict()``, which is the check that neither drifted.
    """
    h, i, v = config.hidden_size, config.intermediate_size, config.vocab_size
    shapes = {"esm.embeddings.word_embeddings.weight": (v, h)}
    dims = {"attention.LayerNorm": (h,), "attention.self.query": (h, h),
            "attention.self.key": (h, h), "attention.self.value": (h, h),
            "attention.output.dense": (h, h), "intermediate.dense": (i, h),
            "output.dense": (h, i), "LayerNorm": (h,)}
    for layer in range(config.num_hidden_layers):
        for name in _LAYER_PARAMETERS:
            shape = dims[name]
            prefix = f"esm.encoder.layer.{layer}.{name}"
            shapes[f"{prefix}.weight"] = shape
            shapes[f"{prefix}.bias"] = (shape[0],)
    shapes["esm.encoder.emb_layer_norm_after.weight"] = (h,)
    shapes["esm.encoder.emb_layer_norm_after.bias"] = (h,)
    shapes["lm_head.bias"] = (v,)
    shapes["lm_head.dense.weight"] = (h, h)
    shapes["lm_head.dense.bias"] = (h,)
    shapes["lm_head.layer_norm.weight"] = (h,)
    shapes["lm_head.layer_norm.bias"] = (h,)
    shapes["lm_head.decoder.weight"] = (v, h)
    return shapes


def legacy_inv_freq_keys(config):
    return [f"esm.encoder.layer.{layer}.attention.self.rotary_embeddings.inv_freq"
            for layer in range(config.num_hidden_layers)]


def contact_head_shapes(config):
    features = config.num_hidden_layers * config.num_attention_heads
    return {"esm.contact_head.regression.weight": (1, features),
            "esm.contact_head.regression.bias": (1,)}


POSITION_IDS_KEY = "esm.embeddings.position_ids"
#: The released HF ESM checkpoints keep a learned absolute-position table although the config
#: says ``position_embedding_type=rotary``: it is never read by a rotary forward. Accepted only
#: under this exact key and its exact native shape, then dropped.
POSITION_EMBEDDINGS_KEY = "esm.embeddings.position_embeddings.weight"

_TIED_OUTPUT_KEY = "native.lm_head.decoder.weight"
_TIED_INPUT_KEY = "native.esm.embeddings.word_embeddings.weight"


def validated_rotary_inv_freq(state, config, *, inv_freq_rtol=1e-5):
    """The legacy per-layer ``inv_freq`` buffers, validated. ``(fp32 tensor, record)``.

    ``(None, None)`` when the checkpoint carries none. Otherwise ALL layers must be present,
    each of the exact shape, each either analytical (within ``inv_freq_rtol``) or exactly the
    fp16-rounded analytical value (what the released ESM-2 files store, and what upstream's
    ``EsmModel.load_hook`` preserves rather than recomputes), and all layers identical.
    The saved values themselves are returned, never the recomputed ones.
    """
    legacy = legacy_inv_freq_keys(config)
    present = [key for key in legacy if key in state]
    if not present:
        return None, None
    require(len(present) == len(legacy),
            f"The checkpoint carries {len(present)} of {len(legacy)} per-layer inv_freq buffers; "
            "a partial set cannot be attributed to the shared rotary frequencies")
    analytical = rotary_inv_freq(config.head_dim, config.rope_theta)
    rounded = analytical.half().float()
    worst, encodings, first = 0.0, set(), None
    for key in present:
        stored = state[key].detach().float().cpu()
        require(tuple(stored.shape) == tuple(analytical.shape),
                f"{key} has shape {tuple(stored.shape)}, expected {tuple(analytical.shape)}")
        relative = float(((stored - analytical).abs() / analytical.abs()).max())
        worst = max(worst, relative)
        if bool(torch.equal(stored, rounded)):
            encodings.add("fp16-rounded-analytical")
        elif relative <= float(inv_freq_rtol):
            encodings.add("analytical")
        else:
            raise ValueError(f"{key} disagrees with the analytical rotary frequencies by relative "
                             f"error {relative:.3e} and is not their fp16 rounding either; the "
                             f"released rope_theta is not {config.rope_theta}")
        if first is None:
            first = stored
        require(bool(torch.equal(stored, first)),
                f"{key} disagrees with the other layers' inv_freq; the released buffers are one "
                "shared vector")
    require(len(encodings) == 1, f"The per-layer inv_freq buffers mix encodings {sorted(encodings)}")
    return first.clone(), {
        "keys": len(present), "encoding": next(iter(encodings)),
        "max_relative_error_vs_analytical": worst, "rtol_for_analytical": float(inv_freq_rtol),
        "action": "validated, then carried into the forward as the shared fp32 rotary_inv_freq "
                  "buffer; the saved values are used, not recomputed"}


def split_checkpoint_state(state, config, *, inv_freq_rtol=1e-5):
    """Validate a raw checkpoint against the full inventory. ``(clean state, record)``.

    * every expected learned key must be present with its exact shape;
    * nothing outside the expected keys and the **exact** allowlist below is accepted:
      the legacy per-layer ``inv_freq`` buffers (validated by
      :func:`validated_rotary_inv_freq` and preserved, not dropped), the unused contact head (exact keys and shapes, then dropped) and
      ``esm.embeddings.position_ids`` (must equal ``arange(max_positions)``, then dropped);
    * the tied output projection must equal the input embedding.

    No prefix is skipped and no key is coerced.
    """
    require(isinstance(state, dict) and state
            and all(isinstance(value, torch.Tensor) for value in state.values()),
            "The checkpoint is not a flat mapping of tensors")
    expected = expected_state_shapes(config)
    legacy = legacy_inv_freq_keys(config)
    contact = contact_head_shapes(config)
    allowed = set(legacy) | set(contact) | {POSITION_IDS_KEY, POSITION_EMBEDDINGS_KEY}
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected) - allowed)
    require(not missing, f"The checkpoint is missing {len(missing)} expected key(s): {missing[:6]}")
    require(not unexpected,
            f"The checkpoint carries {len(unexpected)} key(s) outside the expected inventory and "
            f"the exact allowlist: {unexpected[:6]}. Nothing is skipped by prefix.")
    for key, shape in expected.items():
        require(tuple(state[key].shape) == tuple(shape),
                f"{key} has shape {tuple(state[key].shape)}, the config implies {tuple(shape)}")
        require(state[key].dtype.is_floating_point, f"{key} is not floating point")
    require(bool(torch.equal(state["lm_head.decoder.weight"],
                             state["esm.embeddings.word_embeddings.weight"])),
            "The tied output projection differs from the input embedding. A checkpoint whose two "
            "copies disagree would silently keep whichever loaded last.")
    dropped = {}
    preserved = {}
    _frequencies, rotary_record = validated_rotary_inv_freq(state, config,
                                                            inv_freq_rtol=inv_freq_rtol)
    if rotary_record is not None:
        preserved["rotary_inv_freq"] = rotary_record
    present_contact = [key for key in contact if key in state]
    if present_contact:
        require(set(present_contact) == set(contact),
                f"The contact head is partial: {present_contact}")
        for key, shape in contact.items():
            require(tuple(state[key].shape) == tuple(shape),
                    f"{key} has shape {tuple(state[key].shape)}, expected {tuple(shape)}")
        dropped["contact_head"] = {"keys": sorted(contact), "shapes": {k: list(v) for k, v
                                                                       in contact.items()},
                                   "action": "exact keys and shapes validated, then dropped: "
                                             "DPLM sets contact_head=None"}
    if POSITION_IDS_KEY in state:
        expected_ids = torch.arange(config.max_position_embeddings).expand(1, -1)
        stored = state[POSITION_IDS_KEY].detach().cpu()
        require(tuple(stored.shape) == tuple(expected_ids.shape)
                and bool(torch.equal(stored.long(), expected_ids)),
                f"{POSITION_IDS_KEY} is not arange({config.max_position_embeddings})")
        dropped["position_ids"] = {"action": "equal to arange(max_position_embeddings), then "
                                             "dropped: rotary positions are computed"}
    if POSITION_EMBEDDINGS_KEY in state:
        shape = (config.max_position_embeddings, config.hidden_size)
        stored = state[POSITION_EMBEDDINGS_KEY]
        require(tuple(stored.shape) == shape and stored.dtype.is_floating_point,
                f"{POSITION_EMBEDDINGS_KEY} has shape {tuple(stored.shape)} / dtype "
                f"{stored.dtype}, expected floating point {shape}")
        dropped["position_embeddings"] = {
            "shape": list(shape),
            "action": "exact key, shape and dtype validated, then dropped: the released config "
                      "is rotary, so this learned absolute table is never read by the forward"}
    clean = {key: state[key] for key in expected}
    record = {"expected_keys": len(expected), "checkpoint_keys": len(state),
              "dropped": dropped, "preserved": preserved, "tied_output_equals_embedding": True,
              "file_dtypes": sorted({str(state[key].dtype) for key in expected}),
              "strict": True}
    return clean, record


# ---------------------------------------------------------------------------
# a loaded native model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NativeDPLM:
    """A trunk plus the evidence of what it is. ``identity`` binds the base bytes."""

    model: object
    config: DPLMConfig
    identity: dict
    load_record: dict

    @property
    def hidden_size(self):
        return int(self.config.hidden_size)

    @classmethod
    def random_init(cls, config, *, seed=0, label="tiny-random", storage_dtype=torch.float32,
                    std=None):
        """A randomly initialised trunk for LOW-LEVEL tests. Never a production path.

        ``std`` widens the initialisation so a tiny model's logits are not all near zero
        (which would make every comparison against it weak).
        """
        generator_state = torch.random.get_rng_state()
        torch.manual_seed(int(seed))
        try:
            model = DPLMEncoder(config, storage_dtype=torch.float32)
            if std is not None:
                model.initialize_weights(std)
            if storage_dtype != torch.float32:
                model = model.to(storage_dtype)
                model.tie_weights()
        finally:
            torch.random.set_rng_state(generator_state)
        return cls(model=model, config=config,
                   identity={"variant": str(label), "pretrained": False, "seed": int(seed),
                             "config": config.document(),
                             "storage_dtype": str(storage_dtype),
                             "compute_dtype": "torch.float32"},
                   load_record={"pretrained": False, "label": str(label),
                                "note": "random initialisation; not a pretrained model"})


_STATE_CACHE_NOTE = "torch.load(weights_only=True) of a pickle-only pytorch_model.bin"


def _read_state(path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
    except (RuntimeError, ValueError):
        return torch.load(str(path), map_location="cpu", weights_only=True)


def load_pinned_dplm(weights_dir, *, pins, variant, device="cpu", storage_dtype=torch.float32,
                     dropout=None):
    """Verify, then strictly load a pinned DPLM variant. Never downloads, never random-inits.

    Order: file digests against the pins -> architecture from the pinned ``config.json``
    -> vocabulary -> the full key inventory (:func:`split_checkpoint_state`) -> a strict
    ``load_state_dict`` into a meta-device model. Weights are cast to ``storage_dtype`` on
    the way in; the model computes in fp32 regardless.
    """
    pins_lib.validate_pins(pins, label="the DPLM pins")
    entry = pins_lib.variant_entry(pins, variant)
    directory = Path(weights_dir)
    files = pins_lib.verify_variant_files(directory, entry)
    config = DPLMConfig.from_document(
        _read_json(directory / "config.json"),
        rope_theta=entry["architecture"]["rope_theta"], dropout=dropout)
    check_architecture(config, entry["architecture"])
    vocabulary = parse_vocab((directory / "vocab.txt").read_text(encoding="utf-8"))
    require(vocabulary == DPLM_VOCAB,
            "vocab.txt does not list the 33 DPLM tokens in the pinned id order")
    state = _read_state(directory / "pytorch_model.bin")
    clean, inventory = split_checkpoint_state(state, config)
    saved_frequencies, _ = validated_rotary_inv_freq(state, config)
    converted = {key: value.detach().to(dtype=storage_dtype, copy=True)
                 for key, value in clean.items()}
    del state, clean
    with torch.device("meta"):
        model = DPLMEncoder(config, storage_dtype=storage_dtype, initialize=False)
    model.load_state_dict(converted, strict=True, assign=True)
    # the meta-built buffer is still meta: materialize it (saved values, else analytical) on CPU
    # BEFORE the device move so the forward uses the released frequencies
    model.set_rotary_inv_freq(saved_frequencies if saved_frequencies is not None
                              else rotary_inv_freq(config.head_dim, config.rope_theta))
    model.tie_weights()
    require(all(parameter.device.type != "meta" for parameter in model.parameters()),
            "A parameter is still on the meta device after the strict load")
    require(model.lm_head.decoder.weight is model.esm.embeddings.word_embeddings.weight,
            "The output projection is no longer the input embedding after loading")
    model.eval()
    total = int(sum(parameter.numel() for parameter in model.parameters()))
    identity = {"variant": str(variant), "pretrained": True, "repo": entry["repo"],
                "revision": entry["revision"],
                "weights_sha256": entry["files"]["pytorch_model.bin"].get("sha256"),
                "config_sha256": entry["files"]["config.json"].get("sha256"),
                "config_git_blob_sha1": entry["files"]["config.json"].get("git_blob_sha1"),
                "vocab_git_blob_sha1": entry["files"]["vocab.txt"].get("git_blob_sha1"),
                "vocab_sha256": entry["files"]["vocab.txt"].get("sha256"),
                "pins_sha256": pins_lib.pins_digest(pins),
                # the base bytes an adapter trained against include their resting precision
                "storage_dtype": str(storage_dtype), "compute_dtype": "torch.float32"}
    record = {"pretrained": True, "parameters": total, "strict": True, "weights_only": True,
              "rotary_inv_freq_source": (inventory["preserved"]["rotary_inv_freq"]["encoding"]
                                         if "rotary_inv_freq" in inventory["preserved"]
                                         else "analytical (checkpoint carried no inv_freq)"),
              "storage_dtype": str(storage_dtype), "compute_dtype": "torch.float32",
              "reader": _STATE_CACHE_NOTE, "files": files, "inventory": inventory,
              "config": config.document(),
              "numerical_equivalence_with_pretrained_reference": "NOT ESTABLISHED: no reference "
              "logits for the released weights were available to this build"}
    return NativeDPLM(model=model.to(device), config=config, identity=identity,
                      load_record=record)


def _read_json(path):
    import json
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# diffusion: corruption and iterative reparameterized decoding
# ---------------------------------------------------------------------------

_STRATEGY = re.compile(r"^reparam-(uncond|cond)-(deterministic|stochastic[0-9.]+)-(linear|cosine)$")
_WEIGHTINGS = ("constant", "linear")


@dataclass(frozen=True)
class DiffusionSettings:
    """The explicit diffusion settings of a route. Every one is written in the config."""

    num_timesteps: int = 500
    weighting: str = "constant"
    decode_steps: int = 20
    decoding_strategy: str = "reparam-uncond-deterministic-linear"

    def __post_init__(self):
        require(int(self.num_timesteps) >= 1, "num_timesteps must be positive")
        require(self.weighting in _WEIGHTINGS, f"weighting must be one of {_WEIGHTINGS}")
        require(int(self.decode_steps) >= 1, "decode_steps must be positive")
        require(_STRATEGY.match(self.decoding_strategy) is not None,
                f"decoding_strategy {self.decoding_strategy!r} is not "
                "reparam-<uncond|cond>-<deterministic|stochasticN>-<linear|cosine>")

    @classmethod
    def from_block(cls, block):
        block = dict(block or {})
        unknown = sorted(set(block) - {"num_timesteps", "weighting", "decode_steps",
                                       "decoding_strategy"})
        require(not unknown, f"Unknown diffusion setting(s) {unknown}")
        return cls(**block)

    def document(self):
        return {"num_timesteps": int(self.num_timesteps), "weighting": self.weighting,
                "decode_steps": int(self.decode_steps),
                "decoding_strategy": self.decoding_strategy,
                "sampling": "gumbel_argmax (noise scale 1.0), canonical residues only"}


def non_special_mask(tokens, *, immutable=None):
    """Upstream ``get_non_special_symbol_mask``: not pad, not cls, not eos, not ``immutable``."""
    mask = (tokens.ne(DPLM_PAD_ID) & tokens.ne(DPLM_CLS_ID) & tokens.ne(DPLM_EOS_ID))
    if immutable is not None:
        mask = mask & ~immutable
    return mask


def q_sample(x0, t, maskable, uniform, *, num_timesteps, mask_id=DPLM_MASK_ID):
    """Absorbing-state corruption: mask each maskable position with probability ``t / T``.

    ``uniform`` is a ``(B, T)`` tensor of ``U[0, 1)`` draws supplied by the caller, so a
    training step's corruption is reproducible from the checkpointed mask generator.
    """
    threshold = (t.to(torch.float32) / float(num_timesteps))[:, None]
    mask = (uniform < threshold) & maskable
    return {"x_t": x0.masked_fill(mask, int(mask_id)), "t": t, "mask_mask": mask}


def diffusion_weights(t, *, num_timesteps, weighting):
    """Per-row loss weight: constant 1, or ``(T - (t - 1)) / T`` (upstream's ``linear``)."""
    t = t.to(torch.float32)
    if weighting == "constant":
        return torch.ones_like(t)
    require(weighting == "linear", f"Unknown weighting {weighting!r}")
    return (float(num_timesteps) - (t - 1.0)) / float(num_timesteps)


def _gumbel(shape, device, generator):
    uniform = torch.rand(shape, generator=generator, device=device)
    return -torch.log(-torch.log(uniform + 1e-8) + 1e-8)


def topk_masking(scores, cutoff_len, *, stochastic=False, temperature=1.0, generator=None):
    """Upstream ``topk_masking``: True on the ``cutoff_len`` lowest scores of each row."""
    adjusted = scores
    if stochastic:
        adjusted = scores + float(temperature) * _gumbel(scores.shape, scores.device, generator)
    ordered = adjusted.sort(-1)[0]
    cutoff = ordered.gather(dim=-1, index=cutoff_len)
    return adjusted < cutoff


def _reparam_step(output_tokens, output_scores, cur_tokens, cur_scores, *, strategy,
                  xt_neq_x0, non_special, t, max_step, generator):
    """One ``_reparam_decoding`` update (upstream ``dplm.py``). Returns the new state."""
    _, condition, topk_mode, schedule = strategy.split("-")
    rate = 1.0 - t / max_step if schedule == "linear" else math.cos(t / max_step * math.pi * 0.5)
    cutoff_len = (non_special.sum(1, keepdim=True).to(output_scores.dtype) * rate).long()
    scores_for_topk = cur_scores.masked_fill(~non_special, 1000.0)
    if topk_mode.startswith("stochastic"):
        noise_scale = float(topk_mode.replace("stochastic", ""))
        lowest_k = topk_masking(scores_for_topk, cutoff_len, stochastic=True,
                                temperature=noise_scale * rate, generator=generator)
    else:
        lowest_k = topk_masking(scores_for_topk, cutoff_len, stochastic=False)
    if condition == "cond":
        not_v1_t = (cur_tokens == output_tokens) & (cur_scores < output_scores) & lowest_k
    else:
        not_v1_t = lowest_k
    not_v2_t = lowest_k
    masked_to_noise = (~xt_neq_x0 & not_v1_t) | (xt_neq_x0 & not_v2_t)
    output_tokens = output_tokens.masked_fill(masked_to_noise, DPLM_MASK_ID)
    output_scores = output_scores.masked_fill(masked_to_noise, -math.inf)
    masked_to_x0 = xt_neq_x0 & ~not_v2_t
    output_tokens = torch.where(masked_to_x0, cur_tokens, output_tokens)
    output_scores = torch.where(masked_to_x0, cur_scores, output_scores)
    new_xt_neq_x0 = (xt_neq_x0 | not_v1_t) & not_v2_t
    return new_xt_neq_x0, output_tokens, output_scores


@dataclass
class DecodeResult:
    """A decode: native tokens, the canonical residue at each design slot, and its honesty."""

    native_tokens: object
    canonical: object
    design: object
    forward_calls: int
    steps: int
    strategy: str
    history: list = field(default_factory=list)

    def document(self):
        return {"strategy": self.strategy, "steps": int(self.steps),
                "denoiser_forward_calls": int(self.forward_calls),
                "canonical_only": True, "immutable_scaffold_verified": True,
                "trajectory_logprob": None,
                "trajectory_logprob_note": (
                    "None on purpose. The reparameterized trajectory scores tokens by the argmax "
                    "of Gumbel-perturbed log-softmax and re-masks by those scores; nothing in it is "
                    "a normalized log probability, so none is reported.")}


@torch.no_grad()
def iterative_reparam_decode(logits_fn, tokens, immutable, *, steps, strategy, generator=None,
                             canonical_ids, keep_history=False):
    """Upstream ``generate`` with ``partial_masks``, restricted to the canonical residues.

    ``tokens`` are native ids whose scaffold positions hold real residues; ``immutable`` is
    ``True`` there (upstream's ``partial_masks``). Design positions start as ``<mask>`` and
    are the only positions that ever change; the result is asserted to leave every other
    position byte-identical and to contain no ``<mask>`` and no non-canonical token.
    """
    require(_STRATEGY.match(strategy) is not None, f"Unsupported decoding strategy {strategy!r}")
    require(int(steps) >= 1, "Decoding needs at least one step")
    design = non_special_mask(tokens, immutable=immutable)
    require(bool(design.any(dim=1).all()), "A row has no design position to decode")
    vocabulary = tokens.new_zeros(DPLM_VOCAB_SIZE, dtype=torch.bool)
    vocabulary[canonical_ids.to(tokens.device)] = True
    prev_tokens = tokens.masked_fill(design, DPLM_MASK_ID)
    prev_scores = torch.zeros(tokens.shape, dtype=torch.float32, device=tokens.device)
    prev_masks = design.clone()
    history, forwards = [], 0
    for step in range(int(steps)):
        logits = logits_fn(prev_tokens).to(torch.float32)
        forwards += 1
        logits = logits.masked_fill(~vocabulary, -math.inf)
        noisy = logits + _gumbel(logits.shape, logits.device, generator)
        cur_scores, cur_tokens = torch.log_softmax(noisy, dim=-1).max(dim=-1)
        new_tokens = torch.where(design, cur_tokens, prev_tokens)
        new_scores = torch.where(design, cur_scores, prev_scores)
        prev_masks, prev_tokens, prev_scores = _reparam_step(
            prev_tokens.clone(), prev_scores.clone(), new_tokens, new_scores, strategy=strategy,
            xt_neq_x0=prev_masks, non_special=design, t=step + 1, max_step=int(steps),
            generator=generator)
        if keep_history:
            history.append(prev_tokens.clone())
    require(bool(torch.equal(prev_tokens[~design], tokens[~design])),
            "Decoding changed a scaffold position. The non-design residues are immutable.")
    final = prev_tokens[design]
    require(bool(vocabulary[final].all()),
            "Decoding left a <mask> or a non-canonical token in a design position")
    return DecodeResult(native_tokens=prev_tokens, canonical=None, design=design,
                        forward_calls=forwards, steps=int(steps), strategy=strategy,
                        history=history)


# ---------------------------------------------------------------------------
# the conditioned wrapper the route services use
# ---------------------------------------------------------------------------

FORMAT_FULL = "dplm-full/1"
FORMAT_ADAPTER = "dplm-adapter-only/1"


@dataclass(frozen=True)
class LoraSettings:
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.1

    def __post_init__(self):
        require(int(self.rank) >= 1, "LoRA rank must be positive")
        require(float(self.alpha) > 0, "LoRA alpha must be positive")
        require(0.0 <= float(self.dropout) < 1.0, "LoRA dropout must be in [0, 1)")

    @classmethod
    def from_block(cls, block):
        block = dict(block)
        unknown = sorted(set(block) - {"rank", "alpha", "dropout"})
        require(not unknown, f"Unknown LoRA setting(s) {unknown}")
        return cls(**block)

    def document(self):
        return {"rank": int(self.rank), "alpha": float(self.alpha),
                "dropout": float(self.dropout), "targets": list(LORA_TARGETS),
                "target_rule": "exact module paths inside every layer; each wrapped exactly once"}


class ConditionedDPLM(nn.Module):
    """The trunk plus zero-initialised condition tables, optionally with LoRA.

    Public methods take and return the **pipeline's** token ids (AbLang ids, pad 21, mask
    23) and translate once, at the boundary. The ``*_native_ids`` methods speak DPLM ids and
    are never called by the services.
    """

    def __init__(self, native, condition_spec, *, lora=None, activation_checkpointing=False,
                 diffusion=None):
        super().__init__()
        self.native = native.model
        self.config = native.config
        self.base_identity = dict(native.identity)
        self.load_record = dict(native.load_record)
        self.condition_spec = condition_spec
        self.activation_checkpointing = bool(activation_checkpointing)
        self.diffusion = diffusion if isinstance(diffusion, DiffusionSettings) \
            else DiffusionSettings.from_block(diffusion)
        self._base_parameters = int(sum(p.numel() for p in self.native.parameters()))
        self.lora_settings = None
        if lora is not None:
            self.lora_settings = lora if isinstance(lora, LoraSettings) \
                else LoraSettings.from_block(lora)
            self.native.freeze_base()
            self.lora_names = self.native.attach_lora(
                rank=self.lora_settings.rank, alpha=self.lora_settings.alpha,
                dropout=self.lora_settings.dropout)
        else:
            self.lora_names = []
        hidden = int(self.config.hidden_size)
        sizes = condition_spec.sizes
        device = self.native.esm.embeddings.word_embeddings.weight.device
        self.condition_family = nn.Embedding(sizes["family"], hidden, dtype=torch.float32,
                                             device=device)
        self.condition_bin = nn.Embedding(sizes["bin"], hidden, dtype=torch.float32,
                                          device=device)
        self.condition_assay = nn.Embedding(sizes["assay"], hidden, dtype=torch.float32,
                                            device=device)
        for table in self._tables():
            nn.init.zeros_(table.weight)
        self.use_conditions = True
        self.register_buffer("_canonical_native",
                             torch.tensor(DPLM_CANONICAL_IDS, dtype=torch.long, device=device),
                             persistent=False)
        self.register_buffer("_native_to_canonical",
                             torch.tensor(NATIVE_TO_CANONICAL, dtype=torch.long, device=device),
                             persistent=False)

    # -- small helpers -----------------------------------------------------
    def _tables(self):
        return (self.condition_family, self.condition_bin, self.condition_assay)

    @property
    def hidden_size(self):
        return int(self.config.hidden_size)

    @property
    def is_lora(self):
        return self.lora_settings is not None

    def _device(self):
        return self.condition_family.weight.device

    def condition_vector(self, condition_indices):
        """``(B, 1, H)`` fp32 additive conditioning, or ``None`` when none is requested."""
        if condition_indices is None:
            return None
        require(self.use_conditions,
                "Conditions were requested but conditioning is disabled on this model; silently "
                "ignoring them would report an unconditioned model as a conditioned one.")
        indices = torch.as_tensor(condition_indices, dtype=torch.long, device=self._device())
        require(indices.dim() == 2 and indices.shape[1] == 3,
                f"Condition indices must be (B, 3), got {tuple(indices.shape)}")
        total = (self.condition_family(indices[:, 0]) + self.condition_bin(indices[:, 1])
                 + self.condition_assay(indices[:, 2]))
        return total.unsqueeze(1)

    # -- native-id methods (never called by the services) -----------------
    def forward_native_ids(self, native_tokens, *, condition_indices=None):
        tokens = torch.as_tensor(native_tokens, dtype=torch.long, device=self._device())
        return self.native(tokens, condition=self.condition_vector(condition_indices),
                           activation_checkpointing=self.activation_checkpointing)

    def logits_native_ids(self, native_tokens, *, condition_indices=None):
        return self.forward_native_ids(native_tokens,
                                       condition_indices=condition_indices)["logits"]

    # -- public methods: the ONE boundary translation ----------------------
    def _public(self, tokens):
        tokens = torch.as_tensor(tokens, dtype=torch.long, device=self._device())
        require(tokens.dim() == 2, f"Tokens must be (B, T), got {tuple(tokens.shape)}")
        return ablang_to_dplm(tokens)

    def canonical_log_probs(self, tokens, query_slots, *, condition_indices=None):
        """``(B, k, 20)`` log probabilities over the canonical residues at the queries."""
        native = self._public(tokens)   # a fresh tensor: table[tokens] never aliases the caller
        slots = torch.as_tensor(query_slots, dtype=torch.long, device=native.device)
        require(slots.dim() == 2 and slots.shape[0] == native.shape[0],
                f"Query slots {tuple(slots.shape)} do not correspond to a batch of "
                f"{native.shape[0]}")
        require(int(slots.min()) >= 0 and int(slots.max()) < native.shape[1],
                "A query slot lies outside the sequence")
        if not (self.activation_checkpointing and self.training and torch.is_grad_enabled()):
            return self._score_native(native, slots, condition_indices)
        # Recompute the WHOLE scoring call (condition lookup + trunk + head) in backward instead
        # of retaining its graph: a caller that scores several times before one backward
        # otherwise holds every call's non-layer activations. The per-layer checkpoints stay, so
        # the single recomputed forward is still bounded. Non-reentrant, so parameters reach the
        # graph through the closure and the RNG state (dropout) is replayed. The checkpoint holds
        # its inputs by reference, and as_tensor aliases numpy/same-device tensors, so the slots
        # and condition indices are copied here; the autoregressive/reveal code reuses buffers.
        slots = slots.clone()
        indices = None if condition_indices is None else torch.as_tensor(
            condition_indices, dtype=torch.long, device=self._device()).clone()
        if indices is None:
            return torch.utils.checkpoint.checkpoint(
                lambda ids, query: self._score_native(ids, query, None), native, slots,
                use_reentrant=False, preserve_rng_state=True)
        return torch.utils.checkpoint.checkpoint(
            self._score_native, native, slots, indices,
            use_reentrant=False, preserve_rng_state=True)

    def _score_native(self, native, slots, condition_indices):
        """The uncheckpointed-at-this-level scoring reference (layer checkpoints still apply)."""
        logits = self.logits_native_ids(native, condition_indices=condition_indices)
        gathered = logits.gather(1, slots.unsqueeze(-1).expand(-1, -1, logits.shape[-1]))
        canonical = gathered[..., self._canonical_native]
        return torch.log_softmax(canonical.float(), dim=-1)

    def representation(self, tokens, *, attention_mask=None, condition_indices=None):
        """``(B, hidden)`` masked mean of the final hidden state. The heads' input."""
        native = self._public(tokens)
        state = self.forward_native_ids(native, condition_indices=condition_indices)[
            "last_hidden_state"]
        real = native.ne(DPLM_PAD_ID)
        if attention_mask is not None:
            supplied = torch.as_tensor(attention_mask, device=state.device).bool()
            require(bool(torch.equal(supplied, real)),
                    "The supplied attention mask disagrees with the padding in the tokens. DPLM "
                    "derives its mask from the ids, and a caller's different mask would be ignored.")
        weights = real.to(state.dtype).unsqueeze(-1)
        return (state * weights).sum(dim=1) / torch.clamp(weights.sum(dim=1), min=1.0)

    def diffusion_loss(self, tokens, *, rng, condition_indices=None):
        """Timestep-sampled corruption + weighted canonical NLL. ``(loss, diagnostics)``.

        ``tokens`` are UNMASKED public rows. The timesteps and the uniform draws come from
        ``rng`` (a numpy ``Generator``), which is the checkpointed mask generator, so a
        resumed run corrupts identically.
        """
        native = self._public(tokens)
        require(not bool(native.eq(DPLM_MASK_ID).any()),
                "The diffusion loss corrupts an unmasked row; the input already holds <mask>")
        batch, length = native.shape
        settings = self.diffusion
        t = torch.as_tensor(rng.integers(1, int(settings.num_timesteps) + 1, size=batch),
                            dtype=torch.long, device=native.device)
        uniform = torch.as_tensor(rng.random((batch, length)), dtype=torch.float32,
                                  device=native.device)
        corrupted = q_sample(native, t, non_special_mask(native), uniform,
                             num_timesteps=settings.num_timesteps)
        logits = self.logits_native_ids(corrupted["x_t"], condition_indices=condition_indices)
        mask = corrupted["mask_mask"]
        log_probs = torch.log_softmax(logits[..., self._canonical_native].float(), dim=-1)
        targets = self._native_to_canonical[native]
        require(bool((targets[mask] >= 0).all()), "A corrupted position holds a non-residue")
        picked = log_probs.gather(-1, targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        weight = diffusion_weights(t, num_timesteps=settings.num_timesteps,
                                   weighting=settings.weighting)[:, None]
        active = int(mask.sum())
        if active == 0:
            loss = logits.sum() * 0.0
        else:
            loss = -(picked * weight * mask.to(picked.dtype)).sum() / float(active)
        return loss, {"active_targets": active, "corrupted_fraction": (
            active / max(1, int(non_special_mask(native).sum()))),
            "mean_timestep": float(t.float().mean()), "num_timesteps": int(settings.num_timesteps),
            "weighting": settings.weighting,
            "objective": ("sum over corrupted positions of weight(t) x the negative log "
                          "probability of the true residue (renormalised over the twenty "
                          "canonical residues), divided by the corrupted-position count")}

    @torch.no_grad()
    def iterative_decode(self, tokens, query_slots, *, condition_indices=None, generator=None,
                         steps=None, strategy=None, keep_history=False):
        """Reparameterized iterative decoding of the query slots; every other position fixed.

        ``tokens`` are public ids with ``<mask>`` (23) at the design slots. Returns a
        :class:`DecodeResult` whose ``canonical`` holds the canonical residue index at each
        slot, in slot order.
        """
        native = self._public(tokens)
        slots = torch.as_tensor(query_slots, dtype=torch.long, device=native.device)
        batch, length = native.shape
        require(slots.dim() == 2 and slots.shape[0] == batch, "Query slots do not match the batch")
        require(int(slots.min()) >= 0 and int(slots.max()) < length,
                "A query slot lies outside the sequence")
        rows = torch.arange(batch, device=native.device)[:, None].expand_as(slots)
        require(bool((native[rows, slots] == DPLM_MASK_ID).all()),
                "Every design slot must hold <mask> in the input; a slot holding a residue would "
                "let the decoder read the answer it is generating")
        immutable = torch.ones_like(native, dtype=torch.bool)
        immutable[rows, slots] = False
        condition = self.condition_vector(condition_indices)

        def logits_fn(current):
            return self.native(current, condition=condition, activation_checkpointing=False)[
                "logits"]
        # Decoding is a sampling procedure, not a training step: dropout is off for its whole
        # duration and EVERY module's own train/eval flag is restored afterwards. ``self.train(flag)``
        # would write the root's flag into all descendants and homogenise a deliberately mixed
        # model (a frozen child held in eval inside a training root, or the inverse).
        modes = [(module, module.training) for module in self.modules()]
        self.eval()
        try:
            result = iterative_reparam_decode(
                logits_fn, native, immutable, steps=int(steps or self.diffusion.decode_steps),
                strategy=strategy or self.diffusion.decoding_strategy, generator=generator,
                canonical_ids=self._canonical_native, keep_history=keep_history)
        finally:
            for module, flag in modes:
                module.training = flag
        canonical = self._native_to_canonical[result.native_tokens[rows, slots]]
        require(bool((canonical >= 0).all()), "A decoded slot is not a canonical residue")
        result.canonical = canonical
        return result

    # -- evidence ----------------------------------------------------------
    @torch.no_grad()
    def native_parity(self, tokens, *, condition_indices=None):
        """The conditioned forward against the same trunk with no condition at all."""
        native = self._public(tokens)
        if condition_indices is None:
            condition_indices = torch.zeros(native.shape[0], 3, dtype=torch.long)
        conditioned = self.logits_native_ids(native, condition_indices=condition_indices)
        plain = self.native(native, condition=None,
                            activation_checkpointing=False)["logits"]
        difference = (conditioned.double() - plain.double()).abs()
        tables_zero = all(bool(torch.all(table.weight == 0)) for table in self._tables())
        return {"max_abs_difference": float(difference.max()),
                "bit_identical": bool(float(difference.max()) == 0.0),
                "tables_zero_initialised": bool(tables_zero),
                "claim": ("with zero tables the added vector is exactly 0.0 and the fp32 sum is "
                          "exact, so bit identity is asserted, not approximated")}

    @contextmanager
    def lora_disabled(self):
        """Run with every adapter off, then put back each module's OWN prior flag.

        The flag is restored per module rather than set to ``True``: an inner exit inside an
        outer ``lora_disabled`` must leave the outer context disabled, and a module that was
        deliberately switched off before entry stays off after it.
        """
        previous = [(module, module.lora_enabled)
                    for module in self.native.lora_modules().values()]
        for module, _flag in previous:
            module.lora_enabled = False
        try:
            yield
        finally:
            for module, flag in previous:
                module.lora_enabled = flag

    @torch.no_grad()
    def lora_noop_parity(self, tokens, *, condition_indices=None):
        """A freshly attached adapter (B = 0) must not move the logits at all."""
        require(self.is_lora, "This model carries no LoRA adapter")
        native = self._public(tokens)
        with_adapter = self.logits_native_ids(native, condition_indices=condition_indices)
        with self.lora_disabled():
            without = self.logits_native_ids(native, condition_indices=condition_indices)
        difference = float((with_adapter.double() - without.double()).abs().max())
        b_all_zero = all(bool(torch.all(m.lora_B == 0)) for m in self.native.lora_modules().values())
        return {"max_abs_difference": difference, "lora_B_all_zero": bool(b_all_zero),
                "meaningful": bool(b_all_zero)}

    def frozen_reference(self):
        """An eval-mode, non-trainable copy that cannot move with the original.

        A full-tune model is deep-copied outright: its weights train, so the reference needs
        its own. A LoRA model's frozen base never changes, so the reference shares that
        storage (saving a second copy of the base) through its **own** ``Parameter`` wrappers
        over the detached tensors; moving or casting the reference re-points only its
        wrappers. Adapters, condition tables and buffers are always cloned.
        """
        memo = {}
        if self.is_lora:
            for name, parameter in self.native.named_parameters():
                if not parameter.requires_grad and "lora_" not in name \
                        and id(parameter) not in memo:
                    memo[id(parameter)] = nn.Parameter(parameter.detach(), requires_grad=False)
        reference = copy.deepcopy(self, memo)
        reference.eval()
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        return reference

    @torch.no_grad()
    def reference_parity(self, other, tokens, *, condition_indices=None, tolerance=0.0):
        mine = self.logits_native_ids(self._public(tokens), condition_indices=condition_indices)
        theirs = other.logits_native_ids(other._public(tokens),
                                         condition_indices=condition_indices)
        difference = float((mine.double() - theirs.double()).abs().max())
        require(difference <= float(tolerance),
                f"A reference built from this model disagrees by {difference} logits")
        tables_zero = all(bool(torch.all(table.weight == 0)) for table in self._tables())
        return {"max_abs_difference": difference, "tolerance": float(tolerance),
                "condition_tables_all_zero": bool(tables_zero), "meaningful": not tables_zero}

    def assert_device_consistency(self, *, label="ConditionedDPLM"):
        devices = sorted({str(p.device) for p in self.parameters()}
                         | {str(b.device) for b in self.buffers()})
        require(len(devices) == 1, f"{label}: parameters and buffers span {devices}")
        return {"device": devices[0], "parameters": int(sum(1 for _ in self.parameters()))}

    def dtype_policy(self):
        """Resting dtypes by role. Compared before and after every restore."""
        base, trainable = set(), set()
        for name, parameter in self.named_parameters():
            (trainable if parameter.requires_grad else base).add(str(parameter.dtype))
        return {"frozen_or_base_storage": sorted(base), "trainable": sorted(trainable),
                "compute": "torch.float32",
                "note": ("frozen base tensors may rest in fp16 and are up-cast per operation; "
                         "LoRA factors, conditions and heads are fp32")}

    def parameter_report(self):
        added = sum(int(table.weight.numel()) for table in self._tables())
        lora = int(sum(m.lora_A.numel() + m.lora_B.numel()
                       for m in self.native.lora_modules().values()))
        total = int(sum(p.numel() for p in self.parameters()))
        trainable = int(sum(p.numel() for p in self.parameters() if p.requires_grad))
        require(total == self._base_parameters + added + lora,
                f"The model holds {total} parameters; base {self._base_parameters} + tables "
                f"{added} + LoRA {lora} does not match")
        return {"base": self._base_parameters, "condition_tables": added, "lora": lora,
                "total": total, "trainable": trainable,
                "frozen": total - trainable, "lora_rank": (None if not self.is_lora
                                                          else int(self.lora_settings.rank))}

    # -- checkpoint protocol (explicit; state_dict itself is untouched) ----
    def base_state_keys(self):
        return sorted(f"native.{key}" for key in self.native.state_dict()
                      if "lora_" not in key.rsplit(".", 1)[-1])

    def adapter_state_keys(self):
        """Exactly the keys an adapter-only checkpoint holds: LoRA factors + condition tables."""
        base = set(self.base_state_keys())
        return sorted(key for key in self.state_dict() if key not in base)

    def checkpoint_state(self):
        """``(state, meta)``. Full for a full-tune route, adapter-only for a LoRA route."""
        state = self.state_dict()
        if not self.is_lora:
            return state, {"format": FORMAT_FULL, "base_identity": dict(self.base_identity),
                           "lora": None, "diffusion": self.diffusion.document()}
        keys = self.adapter_state_keys()
        return ({key: state[key] for key in keys},
                {"format": FORMAT_ADAPTER, "base_identity": dict(self.base_identity),
                 "lora": self.lora_settings.document(), "diffusion": self.diffusion.document(),
                 "expected_keys": len(keys)})

    def load_checkpoint_state(self, state, meta, *, strict=True):
        require(strict, "DPLM checkpoints are loaded strictly; a partial restore is refused")
        require(isinstance(meta, dict) and "format" in meta,
                "The checkpoint carries no model_format block, so its base identity cannot be "
                "checked against this model")
        wanted = FORMAT_ADAPTER if self.is_lora else FORMAT_FULL
        require(meta["format"] == wanted,
                f"The checkpoint is {meta['format']!r} and this model is {wanted!r}")
        require(meta["base_identity"] == self.base_identity,
                "The checkpoint was produced over a different base "
                f"({meta['base_identity']}) than this model's ({self.base_identity}). Resuming "
                "would attach adapters to weights they were not trained against.")
        if self.is_lora:
            require(meta.get("lora") == self.lora_settings.document(),
                    f"LoRA settings differ: checkpoint {meta.get('lora')}, model "
                    f"{self.lora_settings.document()}")
        require(meta.get("diffusion") == self.diffusion.document(),
                f"Diffusion settings differ: checkpoint {meta.get('diffusion')}, model "
                f"{self.diffusion.document()}")
        before = self.dtype_policy()
        current = self.state_dict()
        require(isinstance(state, dict) and all(isinstance(v, torch.Tensor) for v in state.values()),
                "The checkpoint state is not a flat mapping of tensors")
        expected = set(self.adapter_state_keys()) if self.is_lora else set(current)
        supplied = set(state)
        require(supplied == expected,
                f"The checkpoint's keys differ from the model's: missing "
                f"{sorted(expected - supplied)[:6]}, unexpected {sorted(supplied - expected)[:6]}")
        for key in sorted(expected):
            require(tuple(state[key].shape) == tuple(current[key].shape),
                    f"{key} has shape {tuple(state[key].shape)}, the model holds "
                    f"{tuple(current[key].shape)}")
        if not self.is_lora:
            # Both tied names are in a full state; loading conflicting copies would keep
            # whichever loaded last, so equality is checked before anything is mutated.
            require(bool(torch.equal(state[_TIED_OUTPUT_KEY].cpu(), state[_TIED_INPUT_KEY].cpu())),
                    "The checkpoint's tied output projection differs from its input embedding; "
                    "loading it would silently keep whichever copy loaded last")
            self.load_state_dict(state, strict=True)
        else:
            result = self.load_state_dict(state, strict=False)
            require(not result.unexpected_keys
                    and sorted(result.missing_keys) == self.base_state_keys(),
                    "After the adapter load the only missing keys must be exactly the frozen "
                    f"base keys; got unexpected {list(result.unexpected_keys)[:6]}")
        require(self.dtype_policy() == before,
                "Loading a checkpoint changed a parameter dtype; precision must survive a restore")
        return {"format": meta["format"], "keys": len(state), "strict": True}
