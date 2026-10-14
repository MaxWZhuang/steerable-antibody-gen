"""DPLM final-hidden fusion with frozen, visible-only interaction context.

All prediction paths share ``_forward_tensors``. The native DPLM implementation,
vocabulary, arithmetic policy and sampler remain unchanged.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from smallAntibodyGen.models.dplm_policy import (
    DPLM_CANONICAL_IDS, DPLM_MASK_ID, DPLM_PAD_ID, DPLM_VOCAB,
    NativeDPLM, dplm_positions, dplm_token_ids, iterative_reparam_decode,
)
from smallAntibodyGen.models.antibody_antigen.denoising import BackendOutput
from smallAntibodyGen.models.antibody_antigen.static_context import StaticContext
from smallAntibodyGen.models.antibody_antigen.dynamic_context import DynamicContextProvider
from smallAntibodyGen.models.antibody_antigen.exact_view_features import ExactViewFeatures
from smallAntibodyGen.models.antibody_antigen.precision import compute_context, validate_compute
from smallAntibodyGen.models.antibody_antigen.views import (
    CANONICAL_RESIDUES, RESIDUE_INDEX, VisibleView,
)


@dataclass(frozen=True)
class FusionConfig:
    context_dim: int = 1280
    bottleneck: int = 128
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    activation_checkpointing: bool = True
    update_mode: str = "lora"      # "lora" | "full"
    compute_dtype: str = "float32"  # "float32" | "bfloat16"

    def __post_init__(self):
        for value in (self.context_dim, self.bottleneck, self.lora_rank):
            if type(value) is not int or value < 1:
                raise ValueError("fusion dimensions and LoRA rank must be positive integers")
        if not 0 < self.lora_alpha < float("inf") or not 0 <= self.lora_dropout < 1:
            raise ValueError("invalid LoRA alpha or dropout")
        if type(self.activation_checkpointing) is not bool:
            raise ValueError("activation_checkpointing must be boolean")
        if self.update_mode not in ("lora", "full"):
            raise ValueError("update_mode must be 'lora' or 'full'")
        if self.compute_dtype not in ("float32", "bfloat16"):
            raise ValueError("compute_dtype must be 'float32' or 'bfloat16'")


def _tokens_to_ab_view(tokens_1d: torch.Tensor, original_view: VisibleView,
                       mapping: dict) -> VisibleView:
    """Reconstruct a VisibleView for H/L from a row of native DPLM token ids.

    ``mapping`` maps ``(role, index)`` -> native sequence slot (from ``dplm_positions``).
    Slots holding ``DPLM_MASK_ID`` become ``None``; canonical residue ids are decoded via
    ``DPLM_VOCAB``.  Only H and L are reconstructed; Ag is never in the native layout.
    One CPU transfer covers all residues.
    """
    # Single host transfer; avoids one CUDA→CPU sync per residue.
    token_list = tokens_1d.cpu().tolist()
    canonical_set = frozenset(int(x) for x in DPLM_CANONICAL_IDS)
    ab_roles = tuple(r for r in original_view.roles if r != "Ag")
    new_chains = []
    for role in ab_roles:
        chain = []
        for i in range(len(original_view.chain(role))):
            slot = mapping[(role, i)]
            tok_id = token_list[slot]
            if tok_id == DPLM_MASK_ID:
                chain.append(None)
            elif tok_id not in canonical_set:
                raise ValueError(
                    f"unexpected non-canonical token id {tok_id} at {(role, i)!r}"
                )
            else:
                chain.append(DPLM_VOCAB[tok_id])
        new_chains.append(tuple(chain))
    return VisibleView(ab_roles, tuple(new_chains))


def serialize_antibody(view: VisibleView):
    """Reuse the public native layout: CLS H EOS EOS [CLS L EOS]."""
    if "H" not in view.roles or "Ag" in view.roles:
        raise ValueError("DPLM queries require H and optional L, never Ag tokens")
    construct = SimpleNamespace(heavy=view.chain("H"),
                                light=view.chain("L") if "L" in view.roles else (),
                                sites=view.coords(), is_heavy_only="L" not in view.roles)
    return dplm_token_ids(construct, masked_sites=view.hidden_sites()), dplm_positions(construct)


class FusedDPLM(nn.Module):
    def __init__(self, native: NativeDPLM, encoder_identity: dict,
                 config: FusionConfig = FusionConfig()):
        super().__init__()
        self.config = config
        self.base_identity = copy.deepcopy(native.identity)
        self.encoder_identity = copy.deepcopy(encoder_identity)
        self.backbone = native.model
        if self.backbone.lora_modules():
            raise ValueError("supply a base DPLM without existing adapters")
        if config.update_mode == "full":
            # Preflight: refuse half-storage before mutating requires_grad flags.
            storage_dtype = self.base_identity.get("storage_dtype", "torch.float32")
            if storage_dtype not in ("torch.float32", "float32"):
                raise ValueError(
                    f"update_mode='full' requires float32 base storage; "
                    f"this model has storage_dtype={storage_dtype!r}. "
                    "Load the checkpoint in float32 or use update_mode='lora'."
                )
            # Validate actual parameter dtypes (not just identity metadata).
            non_fp32 = [
                n for n, p in self.backbone.named_parameters()
                if p.is_floating_point() and p.dtype != torch.float32
            ]
            if non_fp32:
                raise ValueError(
                    f"update_mode='full' requires all backbone floating-point parameters "
                    f"to be float32; non-fp32 params: {non_fp32[:3]!r}"
                    + (" ..." if len(non_fp32) > 3 else "")
                )
            # Ensure gradients are enabled even if caller froze the bare native model.
            self.backbone.requires_grad_(True)
            # All native DPLM parameters train; no freeze, no LoRA.
        else:
            self.backbone.freeze_base()
            self.backbone.attach_lora(rank=config.lora_rank, alpha=config.lora_alpha,
                                      dropout=config.lora_dropout)
        hidden = native.config.hidden_size
        self.norm = nn.LayerNorm(hidden, dtype=torch.float32)
        self.down = nn.Linear(hidden + config.context_dim, config.bottleneck, dtype=torch.float32)
        self.up = nn.Linear(config.bottleneck, hidden, dtype=torch.float32)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.register_buffer("canonical_ids", torch.tensor(DPLM_CANONICAL_IDS), persistent=False)
        # Native storage may be half, but its forward and trainable adapters use fp32.
        self.to(device=next(self.backbone.parameters()).device)
        # The strict native loader returns eval mode; newly attached modules start
        # in train mode. Synchronize all children so checkpointing/dropout agree.
        self.train(self.training)

    @property
    def device(self):
        return self.up.weight.device

    def identity(self):
        return {"schema": "dplm-visible-fusion/2", "base": copy.deepcopy(self.base_identity),
                "encoder": copy.deepcopy(self.encoder_identity), "fusion": asdict(self.config),
                "compute_dtype": self.config.compute_dtype,
                "update_mode": self.config.update_mode,
                "serialization": "native-dplm-paired/1"}

    def preflight(self, views, contexts):
        """Validate the entire query batch before any dynamic extraction."""
        if not views or len(views) != len(contexts):
            raise ValueError("one context is required per nonempty query batch")
        validate_compute(self.config.compute_dtype, self.device)
        serial = []
        for view, context in zip(views, contexts):
            if not isinstance(context, (StaticContext, ExactViewFeatures, DynamicContextProvider)):
                raise TypeError("unsupported context type")
            if context.encoder_identity != self.encoder_identity:
                raise ValueError("encoder identity differs from the fused policy")
            tokens, mapping = serialize_antibody(view)
            if len(tokens) > self.backbone.config.max_position_embeddings:
                raise ValueError("antibody exceeds native position limit; no truncation is allowed")
            context.check_query(view)
            features = None
            if not isinstance(context, DynamicContextProvider):
                features = context.query_ab_features(view)
                if features.shape[1] != self.config.context_dim:
                    raise ValueError("context feature width does not match fusion configuration")
            serial.append((tokens, mapping, features))
        return serial

    def prepare(self, views, contexts):
        serial = self.preflight(views, contexts)
        resolved = []
        for view, context, (tokens, mapping, features) in zip(views, contexts, serial):
            if isinstance(context, DynamicContextProvider):
                exact = context.get_context(view)
                if exact.encoder_identity != self.encoder_identity:
                    raise ValueError("extracted encoder identity differs from the fused policy")
                features = exact.query_ab_features(view)
                if features.shape[1] != self.config.context_dim:
                    raise ValueError("context feature width does not match fusion configuration")
            resolved.append((tokens, mapping, features))
        serial = resolved
        length = max(len(item[0]) for item in serial)
        ids = torch.full((len(views), length), DPLM_PAD_ID, dtype=torch.long, device=self.device)
        features = torch.zeros((len(views), length, self.config.context_dim),
                               device=self.device, dtype=torch.float32)
        residue_mask = torch.zeros((len(views), length), dtype=torch.bool, device=self.device)
        for row, (tokens, mapping, context_features) in enumerate(serial):
            ids[row, :len(tokens)] = torch.tensor(tokens, device=self.device)
            slots = [mapping[coord] for coord in views[row].coords()]
            features[row, slots] = context_features.to(self.device, dtype=torch.float32)
            residue_mask[row, slots] = True
        return ids, features, residue_mask, [item[1] for item in serial]

    def _forward_tensors(self, ids, context_features, residue_mask):
        with compute_context(self.config.compute_dtype, self.device):
            hidden = self.backbone(ids, activation_checkpointing=self.config.activation_checkpointing)[
                "last_hidden_state"]
            residual = self.up(torch.nn.functional.gelu(
                self.down(torch.cat((self.norm(hidden), context_features), dim=-1))))
            fused = hidden + residual * residue_mask.unsqueeze(-1)
            logits = self.backbone.lm_head(fused)
        # Probability normalization and loss always operate in FP32.
        return logits.float(), fused.float()

    def forward_views(self, views, contexts):
        ids, features, mask, mappings = self.prepare(views, contexts)
        logits, hidden = self._forward_tensors(ids, features, mask)
        return logits, hidden, mappings

    def queried_log_probs(self, views, contexts, coordinates, residues):
        """One masked queried residue per row, with whole-call activation checkpointing."""
        if len(coordinates) != len(views) or len(residues) != len(views):
            raise ValueError("query/target batch lengths differ")
        for view, coord, residue in zip(views, coordinates, residues):
            if view.token(coord) is not None:
                raise ValueError("the queried target must remain masked")
            if residue not in RESIDUE_INDEX:
                raise ValueError("target must be a canonical residue")
        ids, features, mask, mappings = self.prepare(views, contexts)
        slots = torch.tensor([mapping[c] for mapping, c in zip(mappings, coordinates)],
                             device=self.device)
        targets = torch.tensor([RESIDUE_INDEX[r] for r in residues], device=self.device)

        def score(tokens, context, residue_mask, query, gold):
            logits, _ = self._forward_tensors(tokens, context, residue_mask)
            rows = torch.arange(len(tokens), device=self.device)
            logp = torch.log_softmax(logits[rows, query][:, self.canonical_ids], dim=-1)
            return logp[rows, gold]

        # All tensors were freshly allocated here; none alias caller buffers. Keep
        # fusion and the native head inside the checkpoint, not only transformer layers.
        args = (ids, features, mask, slots, targets)
        if self.config.activation_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(score, *args, use_reentrant=False, preserve_rng_state=True)
        return score(*args)

    def bind(self, context: StaticContext):
        return BoundFusion(self, context)

    def reconstruction_nll(self, views, contexts, targets):
        """One forward per example; mean masked-site NLL per example, then batch mean."""
        if len(targets) != len(views):
            raise ValueError("one target set is required per view")
        for view, truth in zip(views, targets):
            if not view.hidden_sites() or set(truth.coords()) != set(view.hidden_sites()):
                raise ValueError("reconstruction targets must exactly cover hidden sites")
        ids, features, mask, mappings = self.prepare(views, contexts)
        row_ids, slots, gold = [], [], []
        for row, (mapping, truth) in enumerate(zip(mappings, targets)):
            for coord in truth.coords():
                row_ids.append(row)
                slots.append(mapping[coord])
                gold.append(RESIDUE_INDEX[truth.residue(coord)])
        rows = torch.tensor(row_ids, device=self.device)
        queries = torch.tensor(slots, device=self.device)
        labels = torch.tensor(gold, device=self.device)
        counts = torch.tensor([len(t.coords()) for t in targets], device=self.device)

        def loss(tokens, context, residue_mask, batch_rows, query, target, num_sites):
            logits, _ = self._forward_tensors(tokens, context, residue_mask)
            logp = torch.log_softmax(logits[batch_rows, query][:, self.canonical_ids], dim=-1)
            site_nll = -logp[torch.arange(len(target), device=self.device), target]
            sums = site_nll.new_zeros(len(tokens)).scatter_add(0, batch_rows, site_nll)
            return (sums / num_sites).mean()

        args = (ids, features, mask, rows, queries, labels, counts)
        if self.config.activation_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(loss, *args, use_reentrant=False, preserve_rng_state=True)
        return loss(*args)

    @torch.no_grad()
    def generate(self, view: VisibleView, context, *, steps=20,
                 strategy="reparam-uncond-deterministic-linear", generator=None):
        """Generate a fully revealed view.  ``context`` is a StaticContext or DynamicContextProvider.

        When a DynamicContextProvider is supplied, frozen encoder features are re-extracted
        from the currently visible token state at every decode step.  No clean sequence or
        prior-step context is reused.
        """
        if self.training:
            raise ValueError("generation requires evaluation mode")
        if not isinstance(context, (StaticContext, DynamicContextProvider)):
            raise TypeError("generation requires StaticContext or DynamicContextProvider; exact-view features cannot be reused across decoding states")
        is_dynamic = isinstance(context, DynamicContextProvider)
        if is_dynamic:
            provider = context
            # Build the native token layout from the view (no encoder call yet).
            raw_tokens, mapping, _ = self.preflight([view], [provider])[0]
            length = len(raw_tokens)
            if length > self.backbone.config.max_position_embeddings:
                raise ValueError("antibody exceeds native position limit; no truncation is allowed")
            ids = torch.full((1, length), DPLM_PAD_ID, dtype=torch.long, device=self.device)
            ids[0, :length] = torch.tensor(raw_tokens, device=self.device)
            immutable = ids.ne(DPLM_MASK_ID)

            def _dynamic_logits(tokens):
                # Refresh context from the current token state; supports zero-hidden views.
                ab_view = _tokens_to_ab_view(tokens[0], view, mapping)
                # Route through prepare() for unified identity check and feature assembly.
                fresh_ids, fresh_features, fresh_mask, _ = self.prepare(
                    [ab_view], [provider]
                )
                return self._forward_tensors(fresh_ids, fresh_features, fresh_mask)[0]

            result = iterative_reparam_decode(
                _dynamic_logits, ids, immutable, steps=steps, strategy=strategy,
                generator=generator, canonical_ids=self.canonical_ids)
            mappings = [mapping]
        else:
            ids, features, mask, mappings = self.prepare([view], [context])
            immutable = ids.ne(DPLM_MASK_ID)
            result = iterative_reparam_decode(
                lambda tokens: self._forward_tensors(tokens, features, mask)[0],
                ids, immutable, steps=steps, strategy=strategy, generator=generator,
                canonical_ids=self.canonical_ids)
        # One CPU transfer covers all residues; validate each is canonical.
        final_tok_list = result.native_tokens[0].cpu().tolist()
        values = {}
        for coord, slot in mappings[0].items():
            tok_id = final_tok_list[slot]
            if tok_id not in DPLM_CANONICAL_IDS:
                raise ValueError(
                    f"decoded token {tok_id} at {coord!r} is not in the canonical vocabulary"
                )
            values[coord] = DPLM_VOCAB[tok_id]
        return VisibleView(view.roles, tuple(tuple(values[(role, i)] for i in range(len(chain)))
                                            for role, chain in zip(view.roles, view.chains)))

    def trainable_state(self):
        return {name: value.detach().cpu().clone() for name, value in self.named_parameters()
                if value.requires_grad}

    def load_trainable_state(self, state):
        parameters = {name: value for name, value in self.named_parameters() if value.requires_grad}
        if not isinstance(state, dict) or set(state) != set(parameters):
            raise ValueError("trainable checkpoint inventory mismatch")
        for name, value in state.items():
            target = parameters[name]
            if (not isinstance(value, torch.Tensor) or value.shape != target.shape or
                    value.dtype != target.dtype or not bool(torch.isfinite(value).all())):
                raise ValueError(f"invalid trainable tensor: {name}")
        with torch.no_grad():
            for name, value in state.items():
                parameters[name].copy_(value)


class BoundFusion:
    """Ab-only ResidueBackend; antigen enters only through its bound static context."""
    def __init__(self, policy: FusedDPLM, context: StaticContext):
        self.policy, self.context = policy, context

    def forward(self, view: VisibleView):
        logits, hidden, mappings = self.policy.forward_views([view], [self.context])
        slots = [mappings[0][coord] for coord in view.coords()]
        selected = logits[0, slots]
        canonical = self.policy.canonical_ids
        mass = selected.softmax(-1)[:, canonical].sum(-1)
        return BackendOutput(view.coords(), selected[:, canonical], mass, hidden[0, slots],
                             float(len(view.hidden_sites()) / len(view.coords())))
