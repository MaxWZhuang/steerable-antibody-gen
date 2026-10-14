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

    def __post_init__(self):
        for value in (self.context_dim, self.bottleneck, self.lora_rank):
            if type(value) is not int or value < 1:
                raise ValueError("fusion dimensions and LoRA rank must be positive integers")
        if not 0 < self.lora_alpha < float("inf") or not 0 <= self.lora_dropout < 1:
            raise ValueError("invalid LoRA alpha or dropout")
        if type(self.activation_checkpointing) is not bool:
            raise ValueError("activation_checkpointing must be boolean")


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
        return {"schema": "dplm-static-fusion/1", "base": copy.deepcopy(self.base_identity),
                "encoder": copy.deepcopy(self.encoder_identity), "fusion": asdict(self.config),
                "compute_dtype": "float32", "serialization": "native-dplm-paired/1"}

    def prepare(self, views, contexts):
        if not views or len(views) != len(contexts):
            raise ValueError("one static context is required per nonempty query batch")
        serial = []
        for view, context in zip(views, contexts):
            context.check_query(view)
            if context.encoder_identity != self.encoder_identity:
                raise ValueError("static encoder identity differs from the fused policy")
            tokens, mapping = serialize_antibody(view)
            features = context.query_ab_features(view)
            if features.shape[1] != self.config.context_dim:
                raise ValueError("context feature width does not match fusion configuration")
            serial.append((tokens, mapping, features))
        length = max(len(item[0]) for item in serial)
        if length > self.backbone.config.max_position_embeddings:
            raise ValueError("antibody exceeds native position limit; no truncation is allowed")
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
        with torch.autocast(device_type=self.device.type, enabled=False):
            hidden = self.backbone(ids, activation_checkpointing=self.config.activation_checkpointing)[
                "last_hidden_state"]
            residual = self.up(torch.nn.functional.gelu(
                self.down(torch.cat((self.norm(hidden), context_features), dim=-1))))
            fused = hidden + residual * residue_mask.unsqueeze(-1)
            return self.backbone.lm_head(fused), fused

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
    def generate(self, view: VisibleView, context: StaticContext, *, steps=20,
                 strategy="reparam-uncond-deterministic-linear", generator=None):
        if self.training:
            raise ValueError("generation requires evaluation mode")
        ids, features, mask, mappings = self.prepare([view], [context])
        immutable = ids.ne(DPLM_MASK_ID)
        result = iterative_reparam_decode(
            lambda tokens: self._forward_tensors(tokens, features, mask)[0],
            ids, immutable, steps=steps, strategy=strategy, generator=generator,
            canonical_ids=self.canonical_ids)
        values = {coord: DPLM_VOCAB[int(result.native_tokens[0, slot])]
                  for coord, slot in mappings[0].items()}
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
