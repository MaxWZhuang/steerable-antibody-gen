"""One-stage training with matched draws and exact, purpose-keyed update boundaries."""
from __future__ import annotations

import copy
import math
import random
from contextlib import contextmanager
from dataclasses import asdict, dataclass

import torch

from smallAntibodyGen.experiments.trastuzumab_poc import functional_train as ft
from smallAntibodyGen.experiments.trastuzumab_poc.artifacts import digest, positive_count, seed_value
from smallAntibodyGen.experiments.trastuzumab_poc.exposure import plan_post_exposure
from smallAntibodyGen.experiments.trastuzumab_poc.functional_checkpoint import (
    FUNCTIONAL_PURPOSE, SEEN_KEYS, FunctionalRun, functional_identity, functional_input_identity,
    functional_progress, new_counters, new_seen, planned_functional_exposure,
)
from smallAntibodyGen.experiments.trastuzumab_poc.functional_protocol import FunctionalSchedule
from smallAntibodyGen.experiments.trastuzumab_poc.objectives import (
    UniformMaskSpec, draw_mask_uniform, ordering_loss, whole_core_scores,
)
from smallAntibodyGen.experiments.trastuzumab_poc.post_budget import post_budget_identity
from smallAntibodyGen.experiments.trastuzumab_poc.profiling import runtime_report
from smallAntibodyGen.experiments.trastuzumab_poc.protocol import RestrictedPostData
from smallAntibodyGen.experiments.trastuzumab_poc.randomness import (
    RNG_ALGORITHM, isolated_torch_rng, keyed_random, keyed_seed,
)
from smallAntibodyGen.experiments.trastuzumab_poc.sampling import (
    FamilyFirstSampler, SamplerConfig, build_abonly_view, build_paired_view,
)
from smallAntibodyGen.experiments.trastuzumab_poc.transfer_guard import (
    TRAINING_SCHEMA, validate_stage_lineage,
)
from smallAntibodyGen.models.antibody_antigen.dynamic_context import DynamicContextProvider
from smallAntibodyGen.models.antibody_antigen.views import mask_complete


@dataclass(frozen=True)
class TrainingConfig:
    stage: str = "mid"
    arm: str = "paired"
    seed: int = 0
    total_updates: int = 1000
    examples_per_update: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    schedule: str = "constant"
    reconstruction_weight: float = 1.0
    ordering_weight: float = 0.0
    ordering_margin: float = 0.1
    minimum_ordering_endpoint_coverage: float = 1.0
    span_weight: float = 0.0
    full_region_weight: float = 0.0

    def __post_init__(self):
        if self.stage not in ("mid", "post") or self.arm not in ("paired", "abonly"):
            raise ValueError("unsupported stage or arm")
        seed_value(self.seed)
        positive_count(self.total_updates, "total_updates")
        positive_count(self.examples_per_update, "examples_per_update")
        for name in ("learning_rate", "weight_decay", "gradient_clip", "reconstruction_weight",
                     "ordering_weight", "ordering_margin", "minimum_ordering_endpoint_coverage",
                     "span_weight", "full_region_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite numeric")
        if min(self.learning_rate, self.gradient_clip, self.reconstruction_weight, self.ordering_margin) <= 0:
            raise ValueError("learning rate, clipping, reconstruction weight and margin must be positive")
        if min(self.weight_decay, self.ordering_weight, self.span_weight, self.full_region_weight) < 0:
            raise ValueError("weights must be nonnegative")
        if self.span_weight + self.full_region_weight > 1:
            raise ValueError("mask mixture weights exceed one")
        if not 0 <= self.minimum_ordering_endpoint_coverage <= 1:
            raise ValueError("minimum_ordering_endpoint_coverage must be between zero and one")
        if self.schedule not in ("constant", "cosine"):
            raise ValueError("unsupported schedule")
        if self.stage == "mid" and self.ordering_weight:
            raise ValueError("midtraining does not consume measured ordering")
        if self.stage == "post" and self.arm != "paired":
            raise ValueError("the fixed post-training recipe is paired")


def learning_rate(config, update):
    if config.schedule == "constant":
        return float(config.learning_rate)
    # The declared horizon is immutable across stop/resume boundaries.
    return float(config.learning_rate) * 0.5 * (1 + math.cos(math.pi * update / config.total_updates))


def endpoint_view(construct, sites, arm):
    sequences = {"H": construct.full_heavy(), "L": construct.light}
    if arm == "paired":
        sequences["Ag"] = construct.Ag
    return mask_complete(sequences, sites)


def core_sites(construct):
    return tuple(("H", len(construct.heavy_prefix) + i) for i in range(10))


def check_exposure_plan(plan, config):
    fraction = plan["pair_labels"]["fraction_of_selected"]
    if config.ordering_weight and fraction < config.minimum_ordering_endpoint_coverage:
        raise ValueError(
            f"declared horizon compares only {fraction:.1%} of selected measurements; "
            f"required {config.minimum_ordering_endpoint_coverage:.1%}. Inspect the outcome-free plan "
            "and preregister a longer horizon or an explicit compute-matched coverage threshold.")


@contextmanager
def deterministic_ordering(policy):
    """Disable dropout through backward, retaining training-mode checkpointing.

    Whole-core ordering fits the same dropout-free scores used at evaluation.
    Switching the entire model to eval would disable activation recomputation.
    """
    modules = [(module, module.training) for module in policy.modules()
               if isinstance(module, torch.nn.Dropout)]
    try:
        for module, _ in modules:
            module.training = False
        yield
    finally:
        for module, training in modules:
            module.training = training


class StageTrainer:
    """No global RNG dependency and no implicit discovery of data or labels.

    Each training unit uses singleton encoder/native forwards and gradient
    accumulation. Increasing examples_per_update does not create encoder batching.

    ``functional`` optionally adds the midtraining-only functional arm: scheduled pairs of
    qualified constructs are ordered by their measurements and fitted through a training-only
    scalar head (``self.head``, separate AdamW, separate clipping) that is NOT a policy
    parameter.  Its draws, labels and Torch stream are separate from the denoising ones, so the
    denoising batches, masks and model RNG are identical with or without it.  In ``shared`` mode
    its policy gradient is accumulated with the denoising gradient before the single policy
    clip; in ``detached`` mode the policy receives nothing from it.  A scheduled pair is never
    redrawn: ties and unresolved pairs consume their opportunity without a backward.
    """
    def __init__(self, policy, encoder, config, *, population=None, post_data=None, functional=None,
                 data_identity, source_identity, initialization_identity, transfer_certificate=None):
        if not isinstance(config, TrainingConfig):
            raise TypeError("config must be TrainingConfig")
        if functional is not None:
            if not isinstance(functional, FunctionalRun):
                raise TypeError("functional must be a FunctionalRun")
            if config.stage != "mid":
                raise ValueError("functional supervision is a midtraining-only arm; post-training rejects it")
        if not isinstance(data_identity, dict) or {"budget", "functional"} & set(data_identity):
            raise ValueError("data_identity must be an object without the derived budget and "
                             "functional records")
        if encoder.encoder_identity != policy.encoder_identity:
            raise ValueError("encoder identity differs from policy")
        if any(p.requires_grad or p.grad is not None for p in encoder.parameters()):
            raise ValueError("encoder must be frozen and gradient-free")
        if any(p.dtype != torch.float32 for p in policy.parameters()):
            raise ValueError("training requires FP32 parameter storage")
        if config.stage == "post" and policy.config.update_mode != "lora":
            raise ValueError("post-training uses fresh LoRA in the fixed recipe")
        transfer_certificate = validate_stage_lineage(
            policy, config, initialization_identity,
            transfer_certificate=transfer_certificate, post_data=post_data,
            functional=None if functional is None else functional.qualified)
        encoder_runtime = encoder.runtime_identity()
        if initialization_identity["kind"] == "parent" and (
                initialization_identity["base"]["parent_provenance"]["training"]["encoder_runtime"]
                != encoder_runtime):
            raise ValueError("parent encoder runtime differs from the current encoder")
        self.policy, self.encoder, self.config = policy, encoder, config
        self.sampler = None
        self.post_data = None
        self.exposure_plan = None
        self.functional = self.functional_plan = self.functional_schedule = None
        self.head = self.head_optimizer = None
        self.functional_counters, self.functional_seen = new_counters(), new_seen()
        if config.stage == "mid":
            if population is None or post_data is not None:
                raise ValueError("midtraining requires only a qualified population")
            self.sampler = FamilyFirstSampler(population, SamplerConfig(
                config.examples_per_update, "family_first_v1", config.seed))
            actual_data = {"manifest": population.manifest_digest, "population": population.population_digest}
            if transfer_certificate is not None and (
                    transfer_certificate["population_digest"] != population.population_digest
                    or transfer_certificate["declaration_digest"] != population.manifest_digest):
                raise ValueError("transfer certificate does not match midtraining population")
            # Same full paired population must fit BOTH arms before any training.
            for item in population.train_records():
                sequences = {"H": item.H, "Ag": item.Ag}
                if item.L is not None:
                    sequences["L"] = item.L
                view, _ = mask_complete(sequences, (("H", item.h_eligible[0]),)
                                        if item.h_eligible else (("L", item.l_eligible[0]),))
                self._preflight(view)
                if (config.span_weight or config.full_region_weight) and not item.aux_region:
                    raise ValueError("auxiliary masks require a declared region on every training item")
            if functional is not None:
                self._admit_functional(functional, population)
        else:
            if population is not None or not isinstance(post_data, RestrictedPostData):
                raise ValueError("post-training requires only restricted selected data")
            if not post_data.selected_entries or post_data.selection_allowance.allowance:
                raise ValueError("fixed-last post-training needs selected endpoints and zero selection-label allowance")
            if config.ordering_weight and not post_data.pair_groups():
                raise ValueError("ordering needs at least two selected endpoints in the same assay")
            self.post_data = post_data
            self.exposure_plan = plan_post_exposure(post_data.selected_entries, config)
            check_exposure_plan(self.exposure_plan, config)
            actual_data = {"selected_data": post_data.data_digest()}
            for entry in post_data.selected_entries:
                self._preflight(endpoint_view(entry.construct, core_sites(entry.construct), "paired")[0])
        self.parameters = [(name, p) for name, p in policy.named_parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW([p for _, p in self.parameters], lr=config.learning_rate,
                                          weight_decay=config.weight_decay, foreach=False, fused=False, capturable=False)
        # The budget and functional records are derived from the validated data objects; a
        # caller-supplied identity can never claim them.
        recorded_input = {
            **data_identity,
            "budget": None if post_data is None else post_budget_identity(post_data.budget),
            "functional": None if functional is None else functional_input_identity(functional),
        }
        self.identity = copy.deepcopy({
            "schema": TRAINING_SCHEMA, "config": asdict(config),
            "model": policy.identity(), "encoder_runtime": encoder_runtime,
            "data": actual_data, "input": recorded_input, "source": source_identity,
            "initialization": initialization_identity, "rng": RNG_ALGORITHM,
            "transfer_certificate": transfer_certificate,
            "ordering_readout": "dropout-disabled-with-checkpointing/1",
            "runtime": runtime_report(policy, encoder),
            "optimizer": {"kind": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8, "foreach": False, "fused": False, "capturable": False},
            "selection": "fixed-last", "parameter_names": [n for n, _ in self.parameters],
            "functional": None if functional is None else functional_identity(
                functional, hidden_size=self.head.in_features, seed=config.seed),
        })
        digest(self.identity)  # Reject non-JSON/nonfinite metadata before first update.
        parent_progress = initialization_identity.get("base", {}).get("parent_provenance", {}).get("progress")
        declared_exposure = post_data.inherited_exposure_meta if post_data is not None else None
        self.inherited_exposure = None if parent_progress is None and declared_exposure is None else {
            "parent_progress": copy.deepcopy(parent_progress),
            "external_declaration": copy.deepcopy(declared_exposure),
        }
        self.next_update = 0
        self.counters = dict(examples=0, masked_residues=0, teacher_forced_states=0,
                             pair_draws=0, informative_pairs=0, ties=0, antigen_examples=0)
        self.seen = {key: set() for key in ("records", "antibody_families", "target_families",
                                          "sources", "endpoints", "reconstruction_endpoints",
                                          "ordering_endpoints", "informative_ordering_endpoints")}
        self.failed = False

    def _preflight(self, combined):
        provider = DynamicContextProvider(self.encoder, combined)
        self.encoder.validate_view(combined)
        self.policy.preflight([combined.without_ag()], [provider])

    def _admit_functional(self, functional, population):
        """Everything that can be checked before the first update, labels not consulted."""
        qualified = functional.qualified
        if (qualified.population_digest != population.population_digest
                or qualified.declaration_digest != population.manifest_digest):
            raise ValueError("functional qualification belongs to a different broad population")
        self.functional = functional
        self.functional_plan = planned_functional_exposure(
            functional, seed=self.config.seed, total_updates=self.config.total_updates)
        self.functional_schedule = FunctionalSchedule(functional.data.budget)
        # BOTH complete views of every admitted train construct, so the admitted set is
        # the same for either arm and for any budget drawn from this document.
        ft.preflight_complete_views(self.policy, self.encoder, qualified.train_constructs())
        self.head = ft.new_head(self.policy.backbone.config.hidden_size, self.config.seed, self.policy.device)
        self.head_optimizer = ft.head_optimizer(self.head, functional.config, float(self.config.learning_rate))

    def _functional_pairs(self, update):
        """The scheduled pairs of one update; every ordered pair backpropagates exactly once.

        Returns ``(counter delta, seen delta, summed weighted loss)``.  The pair is read from
        the outcome-free schedule BEFORE its values are consulted and is never replaced.
        """
        run, settings = self.functional, self.functional.config
        delta, seen, total = new_counters(), new_seen(), 0.0
        with isolated_torch_rng(keyed_seed(self.config.seed, update, 0, FUNCTIONAL_PURPOSE), self.policy.device):
            for k in range(settings.pairs_per_update):
                draw = self.functional_schedule.draw(self.config.seed, update, k)
                resolution = run.data.classify(draw)
                delta["pair_opportunities"] += 1
                seen["compared"].update(draw.observations)
                if not resolution.informative:
                    delta["tie_pairs" if resolution.outcome == "tie" else "unresolved_pairs"] += 1
                    continue
                delta["ordered_pairs"] += 1
                seen["ordered"].update(draw.observations)
                with deterministic_ordering(self.policy):
                    features = [ft.endpoint_features(self.policy, self.encoder, run.qualified.construct(name),
                                                     self.config.arm, grad=settings.policy_receives_gradient)
                                for name in (resolution.better_id, resolution.worse_id)]
                    loss, active = ft.pair_loss(self.head, features[0], features[1], settings)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("nonfinite functional loss")
                    loss.backward()
                delta["feature_forwards"] += 2
                delta["active_pairs"] += int(active)
                total += float(loss.detach())
        delta["ordered_updates"] = int(delta["ordered_pairs"] > 0)
        return delta, seen, total

    def _reconstruct(self, combined, target):
        context = DynamicContextProvider(self.encoder, combined)
        return self.policy.reconstruction_nll([combined.without_ag()], [context], [target])

    def _mask(self, union, region, rng):
        return draw_mask_uniform(union, rng, UniformMaskSpec(
            span_region=region, span_weight=self.config.span_weight,
            full_region_coords=region, full_region_weight=self.config.full_region_weight))

    def step(self):
        if self.failed:
            raise RuntimeError("failed trainer cannot continue or save; reload the last valid checkpoint")
        if self.next_update >= self.config.total_updates:
            raise ValueError("declared training horizon is complete")
        self.policy.train()
        self.encoder.eval()
        update = self.next_update
        delta = dict.fromkeys(self.counters, 0)
        seen = {key: set() for key in self.seen}
        metrics = {"reconstruction_nll": 0.0, "ordering_loss": 0.0}
        pairs = head_norm = None  # functional (delta, seen, loss) and the head's pre-clip gradient norm
        self.optimizer.zero_grad(set_to_none=True)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate(self.config, update)
        if self.functional is not None:
            # Tracks the GLOBAL update, also when no head step follows.
            self.head_optimizer.zero_grad(set_to_none=True)
            for group in self.head_optimizer.param_groups:
                group["lr"] = learning_rate(self.config, update) * self.functional.config.head_lr_scale
        try:
            with isolated_torch_rng(keyed_seed(self.config.seed, update, 0, "model"), self.policy.device):
                batch = self.sampler.draw_batch(update) if self.sampler is not None else None
                for example in range(self.config.examples_per_update):
                    if batch is not None:
                        item = batch.paired_items[example]
                        mask = self._mask(item.eligible_union(), item.aux_region,
                                          random.Random(batch.mask_seeds[example].seed))
                        builder = build_paired_view if self.config.arm == "paired" else build_abonly_view
                        combined, target = builder(item, mask.sites)
                        seen["records"].add(item.record_id)
                        seen["antibody_families"].add(item.antibody_family)
                        seen["target_families"].add(item.target_family)
                        seen["sources"].add(item.source_id)
                    else:
                        entry = keyed_random(self.config.seed, update, example, "endpoint").choice(
                            self.post_data.selected_entries)
                        union = core_sites(entry.construct)
                        mask = self._mask(union, union, keyed_random(self.config.seed, update, example, "mask"))
                        combined, target = endpoint_view(entry.construct, mask.sites, self.config.arm)
                        seen["endpoints"].add(entry.endpoint_id)
                        seen["reconstruction_endpoints"].add(entry.endpoint_id)
                    nll = self._reconstruct(combined, target)
                    loss = nll * self.config.reconstruction_weight / self.config.examples_per_update
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("nonfinite reconstruction loss")
                    loss.backward()
                    metrics["reconstruction_nll"] += float(nll.detach()) / self.config.examples_per_update
                    delta["examples"] += 1
                    delta["masked_residues"] += len(target)
                    delta["antigen_examples"] += int(self.config.arm == "paired")
                    if self.config.ordering_weight:
                        pair = self.post_data.draw_pair(keyed_random(self.config.seed, update, example, "pair"))
                        delta["pair_draws"] += 1
                        seen["endpoints"].update((pair.left_id, pair.right_id))
                        seen["ordering_endpoints"].update((pair.left_id, pair.right_id))
                        if not pair.informative:
                            delta["ties"] += 1
                            continue
                        constructs = dict(self.post_data.constructs())
                        best, worst = constructs[pair.better_id], constructs[pair.worse_id]
                        union = core_sites(best)
                        order = list(union)
                        keyed_random(self.config.seed, update, example, "order").shuffle(order)
                        view, best_target = endpoint_view(best, union, "paired")
                        _, worst_target = endpoint_view(worst, union, "paired")
                        with deterministic_ordering(self.policy):
                            scores = whole_core_scores(self.policy, DynamicContextProvider(self.encoder, view),
                                                       view.without_ag(), [best_target, worst_target], order,
                                                       state_microbatch=1)
                            penalty = ordering_loss(scores, [(0, 1)], self.config.ordering_margin)
                            weighted = penalty * self.config.ordering_weight / self.config.examples_per_update
                            if not bool(torch.isfinite(weighted)):
                                raise FloatingPointError("nonfinite ordering loss")
                            weighted.backward()
                        seen["informative_ordering_endpoints"].update((pair.left_id, pair.right_id))
                        metrics["ordering_loss"] += float(penalty.detach()) / self.config.examples_per_update
                        delta["informative_pairs"] += 1
                        delta["teacher_forced_states"] += 20
                if self.functional is not None:
                    # Shared gradients accumulate with the denoising gradients before the clip.
                    pairs = self._functional_pairs(update)
                norm = torch.nn.utils.clip_grad_norm_([p for _, p in self.parameters],
                                                     self.config.gradient_clip, error_if_nonfinite=True,
                                                     foreach=False)
                if any(p.grad is None for _, p in self.parameters):
                    raise RuntimeError("trainable parameter did not participate in this update")
                if pairs is not None and pairs[0]["ordered_updates"]:
                    # Its own clip, never the policy's; both clips precede both optimizer steps.
                    if self.head.weight.grad is None:
                        raise RuntimeError("head did not participate in this update")
                    head_norm = ft.clip_head_gradient(self.head, self.functional.config)
                self.optimizer.step()
                if head_norm is not None:
                    self.head_optimizer.step()
                # One device-to-host synchronization, rather than one per tensor.
                if not bool(torch.stack([torch.isfinite(p).all() for _, p in self.parameters]).all()):
                    raise FloatingPointError("nonfinite parameters after optimizer update")
                if head_norm is not None and not bool(torch.isfinite(self.head.weight).all()):
                    raise FloatingPointError("nonfinite head weights after optimizer update")
                if any(p.grad is not None or p.requires_grad for p in self.encoder.parameters()):
                    raise RuntimeError("frozen encoder gradient contract violated")
        except BaseException:
            self.failed = True
            self.optimizer.zero_grad(set_to_none=True)
            if self.head_optimizer is not None:
                self.head_optimizer.zero_grad(set_to_none=True)
            raise
        self.optimizer.zero_grad(set_to_none=True)
        result = {"update": update + 1, "learning_rate": self.optimizer.param_groups[0]["lr"],
                  "gradient_norm": float(norm), **metrics, **delta}
        if pairs is not None:
            self.head_optimizer.zero_grad(set_to_none=True)
            functional_delta, functional_seen, functional_loss = pairs
            self.functional_counters = {key: self.functional_counters[key] + functional_delta[key]
                                        for key in functional_delta}
            for key in SEEN_KEYS:
                self.functional_seen[key].update(functional_seen[key])
            result["functional"] = {"loss": functional_loss, "head_stepped": head_norm is not None,
                                    "head_gradient_norm": head_norm,
                                    "head_learning_rate": self.head_optimizer.param_groups[0]["lr"],
                                    **functional_delta}
        self.next_update += 1
        self.counters = {key: self.counters[key] + delta[key] for key in delta}
        for key in self.seen:
            self.seen[key].update(seen[key])
        return result

    def progress(self):
        return {"next_update": self.next_update, "counters": dict(self.counters),
                "seen": {key: sorted(values) for key, values in self.seen.items()},
                "unique": {key: len(values) for key, values in self.seen.items()},
                "fitting_endpoints_available": self.post_data.fitting_count if self.post_data else 0,
                "selection_labels": 0,
                "inherited_exposure": copy.deepcopy(self.inherited_exposure),
                "functional": None if self.functional is None else functional_progress(
                    self.functional_counters, self.functional_seen)}

    def run(self, *, stop_after_updates=None, on_update=None):
        stop = self.config.total_updates if stop_after_updates is None else stop_after_updates
        positive_count(stop, "stop_after_updates")
        if not self.next_update <= stop <= self.config.total_updates:
            raise ValueError("stop boundary must be between current update and original horizon")
        while self.next_update < stop:
            metrics = self.step()
            if on_update is not None:
                on_update(self, metrics)
        return self.progress()
