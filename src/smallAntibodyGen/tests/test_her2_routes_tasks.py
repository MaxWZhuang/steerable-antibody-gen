"""The four objectives, measured: gradients, sensitivity, and no answer leakage.

These are the checks that would have caught the defect the runtime shipped with for
two rounds -- ``masked_nll`` applied to all four scheduled tasks with only the
coefficient changing, so ``head_only`` was reconstruction under another name, the
observation heads never ran a forward, and the measured values never entered the
graph. A loss curve does not show any of that.

So each objective is exercised against a tiny real torch model through the real
serialisation, and what is asserted is:

* the loss MOVES when the measurement moves (the numeric term is not decoration);
* the loss MOVES when the context moves (the effect is contextual, not positional);
* the head objective's gradient reaches the TRUNK, not only the head;
* a quality bin never reaches a head batch;
* the two-site joint is built by revealing the first residue, so it can express a
  dependence that two simultaneously masked marginals cannot.

No pinned native weights and no artifact volume: the model here is small and local,
because what is under test is the wiring rather than the backbone.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from smallAntibodyGen.experiments import her2_routes_data as data_lib
from smallAntibodyGen.experiments import her2_routes_serial as serial
from smallAntibodyGen.experiments import her2_routes_tasks as tasks_lib
from smallAntibodyGen.experiments.her2_routes_conditions import (
    BIN_HIGH,
    BIN_NULL,
    Condition,
    ConditionSpec,
)
from smallAntibodyGen.models.her2_routes_heads import ObservationHeads

ASSAY = "synthetic:affinity:target"
HEAVY = "ACDEFGHIKLMNPQRSTVWY" * 2
LIGHT = "YWVTSRQPNMLKIHGFEDCA"


class TinyEditor(torch.nn.Module):
    """A minimal masked editor with the same interface the real routes expose.

    Deliberately small and deliberately real: it has an embedding, a mixing layer and
    a projection, so a gradient either reaches its parameters or does not, and the
    conditioning enters additively exactly as it does in the pinned wrapper.
    """

    def __init__(self, *, vocabulary=serial.ABLANG_VOCAB_SIZE, hidden=24, condition_sizes=None):
        super().__init__()
        self.embed = torch.nn.Embedding(int(vocabulary), int(hidden))
        self.mix = torch.nn.Linear(int(hidden), int(hidden))
        self.project = torch.nn.Linear(int(hidden), 20)
        sizes = condition_sizes or {"family": 3, "bin": 4, "assay": 2}
        self.condition_family = torch.nn.Embedding(int(sizes["family"]), int(hidden))
        self.condition_bin = torch.nn.Embedding(int(sizes["bin"]), int(hidden))
        self.condition_assay = torch.nn.Embedding(int(sizes["assay"]), int(hidden))
        for table in (self.condition_family, self.condition_bin, self.condition_assay):
            torch.nn.init.normal_(table.weight, std=0.2)

    def hidden_states(self, tokens, condition_indices=None):
        tokens = torch.as_tensor(tokens, dtype=torch.long)
        hidden = self.embed(tokens)
        if condition_indices is not None:
            indices = torch.as_tensor(condition_indices, dtype=torch.long)
            hidden = hidden + (self.condition_family(indices[:, 0])
                               + self.condition_bin(indices[:, 1])
                               + self.condition_assay(indices[:, 2])).unsqueeze(1)
        # One mixing step so a residue's logits depend on the whole row: without it a
        # local contrast's two alternatives could not differ through their context.
        pooled = hidden.mean(dim=1, keepdim=True)
        return torch.tanh(self.mix(hidden + pooled))

    def canonical_log_probs(self, tokens, query_slots, *, condition_indices=None):
        states = self.hidden_states(tokens, condition_indices)
        slots = torch.as_tensor(query_slots, dtype=torch.long)
        gathered = states.gather(1, slots.unsqueeze(-1).expand(-1, -1, states.shape[-1]))
        return torch.log_softmax(self.project(gathered), dim=-1)

    def representation(self, tokens, *, attention_mask=None, condition_indices=None):
        states = self.hidden_states(tokens, condition_indices)
        if attention_mask is None:
            return states.mean(dim=1)
        weights = torch.as_tensor(attention_mask, dtype=states.dtype).unsqueeze(-1)
        return (states * weights).sum(dim=1) / torch.clamp(weights.sum(dim=1), min=1.0)

    @property
    def hidden_size(self):
        return int(self.project.in_features)


class TinyServices:
    """The slice of ``FlightServices`` the objectives call. Real collation, real views."""

    device = "cpu"

    def __init__(self):
        self.condition_spec = ConditionSpec(assays=(ASSAY,))
        self.config = {"her2": {"wild_type_core": "WGGDGFYAMD"},
                       "curriculum": {"kappa": 2.0}}

    def collate(self, route, views):
        return data_lib.collate_masked(views, pad_id=serial.ABLANG_PAD_ID,
                                       condition_spec=self.condition_spec)

    def canonical_log_probs(self, route, model, batch):
        return model.canonical_log_probs(batch.tokens, batch.query_slots,
                                         condition_indices=batch.condition_indices)

    def window_log_probs(self, route, model, batch):
        log_probs = self.canonical_log_probs(route, model, batch)
        targets = torch.as_tensor(batch.targets, dtype=torch.long)
        mask = torch.as_tensor(batch.target_mask, dtype=log_probs.dtype)
        return log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1) * mask

    def representation(self, route, model, batch):
        return model.representation(batch.tokens, attention_mask=batch.attention_mask,
                                    condition_indices=batch.condition_indices)

    def contrast_views(self, contrasts, *, route):
        views = []
        for contrast in contrasts:
            for record in contrast.alternatives():
                construct = data_lib.with_sites(record.construct, contrast.sites)
                condition = Condition(family=record.family, bin=BIN_NULL, assay=record.assay)
                views.append((construct, serial.ablang_serialization(
                    construct, masked_sites=contrast.sites, order=contrast.sites), condition))
        return tuple(views)

    def contrast_payload(self, contrasts):
        spans, margins, interactions, cursor = [], [], [], 0
        for contrast in contrasts:
            width = len(contrast.alternatives())
            spans.append((cursor, cursor + width))
            cursor += width
            margins.append(float("nan") if contrast.margin is None else float(contrast.margin))
            interactions.append(float("nan") if contrast.interaction is None
                                else float(contrast.interaction))
        return {"row_span": tuple(spans), "rows": cursor,
                "margin": np.asarray(margins, dtype=np.float64),
                "interaction": np.asarray(interactions, dtype=np.float64)}

    def joint_support(self, route, model, summary, *, sites, condition, residues=20,
                      first_residues=None):
        from smallAntibodyGen.experiments.her2_routes_services import FlightServices
        return FlightServices.joint_support(self, route, model, summary, sites=sites,
                                           condition=condition, residues=residues,
                                           first_residues=first_residues)

    def full_joint_oracle(self, route, model, summary, *, sites, condition, residues=20):
        first, second = sites
        construct = data_lib.with_sites(summary.construct, (first, second))
        rows = [(construct, serial.ablang_serialization(
            construct, masked_sites=(first, second), order=(first,)), condition)]
        for residue in "ACDEFGHIKLMNPQRSTVWY"[:residues]:
            revealed = construct.with_residues((first,), residue)
            rows.append((revealed, serial.ablang_serialization(
                revealed, masked_sites=(second,), order=(second,)), condition))
        batch = self.collate(route, rows)
        log_probs = self.canonical_log_probs(route, model, batch)
        first_conditional = log_probs[0, 0]
        second_conditionals = log_probs[1:, 0]
        with torch.no_grad():
            probabilities = second_conditionals.detach().exp()
            worst = 0.0
            for left in range(residues):
                for right in range(left + 1, residues):
                    worst = max(worst, float(
                        0.5 * (probabilities[left] - probabilities[right]).abs().sum()))
        return first_conditional.unsqueeze(-1) + second_conditionals, len(rows), worst


def summary(heavy, *, value, region, kind="point", censoring="none", interval=None,
            light=LIGHT, mutated=()):
    construct = serial.Construct(heavy=heavy, light=light, sites=tuple(region),
                                 source="synthetic", record_id=heavy[:6])
    low, high = interval if interval is not None else (value, value)
    return data_lib.ConstructSummary(
        construct=construct, study="synthetic", assay=ASSAY, target="target",
        parent="synthetic::parent", family="affinity", kind=kind,
        measurements=(data_lib.ReplicateMeasurement(
            value=value, lower=low, upper=high, censoring=censoring, replicate=1,
            measurement_id=heavy[:6]),),
        region=tuple(region), mutated_sites=tuple(mutated), partition="train", scale="pKD",
        source="synthetic")


def mutate(heavy, position, residue):
    buffer = list(heavy)
    buffer[position] = residue
    return "".join(buffer)


@pytest.fixture
def pieces():
    services = TinyServices()
    torch.manual_seed(20260927)
    model = TinyEditor(condition_sizes=services.condition_spec.sizes)
    heads = ObservationHeads(model.hidden_size,
                             assays=services.condition_spec.assay_order)
    return services, model, heads


# --- the local numeric objective ------------------------------------------

def _contrast(margin, *, region=(("H", 3), ("H", 7))):
    left = summary(mutate(HEAVY, 3, "W"), value=8.0, region=region)
    right = summary(mutate(HEAVY, 3, "M"), value=8.0 - margin, region=region)
    return data_lib.LocalContrast(
        left=left, right=right, sites=(("H", 3),), left_value=left.value,
        right_value=right.value, assay=ASSAY, target="target", same_assay=True,
        coupled=False, kind=data_lib.CONTRAST_ALTERNATIVES, margin=margin, scale="pKD")


def test_the_local_objective_is_sensitive_to_the_measured_margin(pieces):
    """Change the MEASUREMENT, the loss must change. This is the whole point of it."""
    services, model, heads = pieces

    def build(margin):
        outcome = tasks_lib.local_outcome(
            services, "ablang2_masked_editor", model, [_contrast(margin)], kappa=2.0,
            reconstruction_weight=0.25)
        return float(outcome.loss.detach())
    record = tasks_lib.require_sensitive(lambda: build(0.5),
                                        perturb=lambda: build(-1.5),
                                        label="local numeric objective")
    assert record["sensitive"] and record["difference"] > 1e-6


def test_the_local_objective_is_sensitive_to_the_CONTEXT(pieces):
    """Same substitution, different background: the fitted odds must differ.

    The contextual claim is the campaign's whole reason for existing, so it is a
    measurement here: two contrasts with the identical differing coordinate and the
    identical measured margin, on two different backgrounds, must not produce the
    same loss. A position-only model would.
    """
    services, model, heads = pieces

    def build(background):
        left = summary(mutate(background, 3, "W"), value=8.0, region=(("H", 3),))
        right = summary(mutate(background, 3, "M"), value=7.5, region=(("H", 3),))
        contrast = data_lib.LocalContrast(
            left=left, right=right, sites=(("H", 3),), left_value=left.value,
            right_value=right.value, assay=ASSAY, target="target", same_assay=True,
            coupled=False, kind=data_lib.CONTRAST_ALTERNATIVES, margin=0.5, scale="pKD")
        outcome = tasks_lib.local_outcome(
            services, "ablang2_masked_editor", model, [contrast], kappa=2.0,
            reconstruction_weight=0.25)
        return float(outcome.loss.detach())
    other = mutate(mutate(HEAVY, 11, "W"), 15, "P")
    record = tasks_lib.require_sensitive(lambda: build(HEAVY),
                                        perturb=lambda: build(other),
                                        label="local objective context dependence")
    assert record["sensitive"]


def test_the_local_objective_reaches_the_trunk_and_names_its_terms(pieces):
    services, model, heads = pieces
    outcome = tasks_lib.local_outcome(
        services, "ablang2_masked_editor", model, [_contrast(0.5)], kappa=2.0,
        reconstruction_weight=0.25)
    assert outcome.term == "local_effect"
    assert set(outcome.terms) == {"local_effect", "local_reconstruction"}
    assert outcome.measured
    model.zero_grad(set_to_none=True)
    outcome.loss.backward()
    reached = [name for name, parameter in model.named_parameters()
               if parameter.grad is not None and float(parameter.grad.abs().sum()) > 0]
    assert "embed.weight" in reached and "project.weight" in reached


def test_a_contrast_with_no_measured_margin_is_refused_not_scored(pieces):
    services, model, heads = pieces
    contrast = _contrast(0.5)
    contrast.margin = None
    with pytest.raises(ValueError, match="no numeric target to fit"):
        tasks_lib.local_outcome(services, "ablang2_masked_editor", model, [contrast],
                                kappa=2.0, reconstruction_weight=0.25)


# --- the coupled quartet --------------------------------------------------

def test_the_two_site_joint_reveals_the_first_residue_before_the_second(pieces):
    """The dependence is measured. Two simultaneously masked marginals would give zero."""
    services, model, heads = pieces
    region = (("H", 3), ("H", 7))
    background = summary(HEAVY, value=8.0, region=region)
    left = summary(mutate(HEAVY, 3, "W"), value=7.5, region=region)
    right = summary(mutate(HEAVY, 7, "A"), value=8.2, region=region)
    double = summary(mutate(mutate(HEAVY, 3, "W"), 7, "A"), value=8.6, region=region)
    contrast = data_lib.LocalContrast(
        left=left, right=right, sites=region, left_value=left.value,
        right_value=right.value, assay=ASSAY, target="target", same_assay=True, coupled=True,
        kind=data_lib.CONTRAST_COUPLED, background=background, double=double,
        margin=left.value - right.value, effect_left=-0.5, effect_right=0.2,
        double_effect=0.6, interaction=0.9, scale="pKD")
    block = tasks_lib.quartet_joint_outcome(services, "ablang2_masked_editor", model,
                                            contrast, kappa=2.0)
    assert block is not None
    assert block["forwards"] == 3, "one doubly masked row plus the two measured first-residue branches"
    assert block["max_total_variation"] > 0.0, (
        "the second conditional must move when the first residue is revealed; zero means "
        "the two coordinates are modelled independently and no interaction is representable")
    assert sum(block["target"]) == pytest.approx(1.0)
    assert sum(block["model"]) == pytest.approx(1.0)
    assert float(block["loss"]) >= 0.0
    # The reduced branch computation must preserve the exact four-cell loss and
    # its gradient, not merely produce a normalized distribution.
    full, _, _ = services.full_joint_oracle(
        "ablang2_masked_editor", model, background, sites=region,
        condition=Condition(family=background.family, bin=BIN_NULL, assay=background.assay))
    cells = [(serial.canonical_index(row.construct.residue_at(region[0])),
              serial.canonical_index(row.construct.residue_at(region[1])))
             for row in (background, left, right, double)]
    log_q = torch.log_softmax(torch.stack([full[a, b] for a, b in cells]), dim=0)
    target = torch.as_tensor(block["target"], dtype=log_q.dtype)
    full_loss = (target * (target.log() - log_q)).sum()
    assert torch.allclose(block["loss"], full_loss, atol=1e-6)
    parameters = tuple(model.parameters())
    reduced_grads = torch.autograd.grad(block["loss"], parameters, allow_unused=True)
    full_grads = torch.autograd.grad(full_loss, parameters, allow_unused=True)
    for reduced, expanded in zip(reduced_grads, full_grads):
        assert (reduced is None) == (expanded is None)
        if reduced is not None:
            assert torch.allclose(reduced, expanded, atol=1e-6)


def test_the_quartet_term_is_sensitive_to_the_measured_interaction(pieces):
    services, model, heads = pieces
    region = (("H", 3), ("H", 7))

    def build(double_effect):
        background = summary(HEAVY, value=8.0, region=region)
        left = summary(mutate(HEAVY, 3, "W"), value=7.5, region=region)
        right = summary(mutate(HEAVY, 7, "A"), value=8.2, region=region)
        double = summary(mutate(mutate(HEAVY, 3, "W"), 7, "A"),
                         value=8.0 + double_effect, region=region)
        contrast = data_lib.LocalContrast(
            left=left, right=right, sites=region, left_value=left.value,
            right_value=right.value, assay=ASSAY, target="target", same_assay=True,
            coupled=True, kind=data_lib.CONTRAST_COUPLED, background=background,
            double=double, margin=left.value - right.value, effect_left=-0.5,
            effect_right=0.2, double_effect=double_effect,
            interaction=double_effect + 0.3, scale="pKD")
        return float(tasks_lib.quartet_joint_outcome(
            services, "ablang2_masked_editor", model, contrast, kappa=2.0)["loss"])
    record = tasks_lib.require_sensitive(lambda: build(0.6), perturb=lambda: build(-1.2),
                                        label="quartet joint term")
    assert record["sensitive"]


# --- the observation heads ------------------------------------------------

def head_batch(services, rows):
    views = data_lib.head_views(rows, condition_spec=services.condition_spec, route="masked",
                               max_positions=512)
    return data_lib.collate_heads(views, pad_id=serial.ABLANG_PAD_ID,
                                  condition_spec=services.condition_spec,
                                  assay_index={ASSAY: 0})


def test_the_head_objective_reaches_the_TRUNK_not_only_the_head(pieces):
    """A head trained on a detached representation has its own gradients and teaches
    the trunk nothing. Both facts are separated here."""
    services, model, heads = pieces
    batch = head_batch(services, [summary(HEAVY, value=8.0, region=(("H", 3),)),
                                  summary(mutate(HEAVY, 3, "W"), value=7.0,
                                          region=(("H", 3),))])
    outcome = tasks_lib.head_outcome(services, "ablang2_masked_editor", model, heads, batch)
    assert outcome.term == "heads"
    assert "interval" in outcome.terms
    model.zero_grad(set_to_none=True)
    heads.zero_grad(set_to_none=True)
    outcome.loss.backward()
    trunk = [name for name, parameter in model.named_parameters()
             if parameter.grad is not None and float(parameter.grad.abs().sum()) > 0]
    head_parameters = [name for name, parameter in heads.named_parameters()
                       if parameter.grad is not None and float(parameter.grad.abs().sum()) > 0]
    assert trunk, "the head loss did not reach the trunk"
    assert head_parameters, "the head loss did not reach the head"


def test_the_head_objective_is_sensitive_to_the_measurement(pieces):
    services, model, heads = pieces

    def build(value):
        batch = head_batch(services, [summary(HEAVY, value=value, region=(("H", 3),))])
        return float(tasks_lib.head_outcome(services, "ablang2_masked_editor", model,
                                            heads, batch).loss.detach())
    record = tasks_lib.require_sensitive(lambda: build(8.0), perturb=lambda: build(-3.0),
                                        label="observation-head likelihood")
    assert record["sensitive"]


def test_each_replicate_keeps_its_own_censoring_in_the_head_likelihood(pieces):
    """A floor replicate and two point replicates are not a point at the mean of two."""
    services, model, heads = pieces
    construct = serial.Construct(heavy=HEAVY, light=LIGHT, sites=(("H", 3),),
                                 source="synthetic", record_id="mixed")
    mixed = data_lib.ConstructSummary(
        construct=construct, study="synthetic", assay=ASSAY, target="target",
        parent="synthetic::parent", family="affinity", kind="interval",
        measurements=(
            data_lib.ReplicateMeasurement(value=None, lower=None, upper=6.0,
                                          censoring="left", replicate="a"),
            data_lib.ReplicateMeasurement(value=8.0, lower=8.0, upper=8.0,
                                          censoring="none", replicate="b"),
            data_lib.ReplicateMeasurement(value=8.4, lower=8.4, upper=8.4,
                                          censoring="none", replicate="c")),
        region=(("H", 3),), partition="train", scale="pKD", source="synthetic")
    assert mixed.censoring == "mixed"
    assert mixed.value == pytest.approx(8.2), "the point estimate averages the UNCENSORED only"
    assert mixed.interval == (None, 8.4), "the pooled bound is the union, so it is unbinnable"
    batch = head_batch(services, [mixed])
    payload = batch.measurement
    assert payload.replicates() == 3
    assert int(np.asarray(payload.replicate_censored).sum()) == 1
    assert np.asarray(payload.replicate_weight).sum() == pytest.approx(1.0), (
        "the three replicate terms are weighted to make the construct contribute once")
    outcome = tasks_lib.head_outcome(services, "ablang2_masked_editor", model, heads, batch)
    assert outcome.diagnostics["eligible_rows"]["censored_replicates"] == 1


def test_a_head_batch_carrying_a_quality_bin_is_refused(pieces):
    services, model, heads = pieces
    rows = [summary(HEAVY, value=8.0, region=(("H", 3),))]
    views = list(data_lib.head_views(rows, condition_spec=services.condition_spec,
                                    route="masked", max_positions=512))
    construct, tokens, condition, row = views[0]
    leaking = (construct, tokens, Condition(family=condition.family, bin=BIN_HIGH,
                                            assay=condition.assay), row)
    with pytest.raises(ValueError, match="answer-free stream"):
        data_lib.collate_heads([leaking], pad_id=serial.ABLANG_PAD_ID,
                              condition_spec=services.condition_spec,
                              assay_index={ASSAY: 0})


def test_a_head_batch_the_heads_cannot_read_is_refused_not_scored_as_zero(pieces):
    """An all-ineligible microbatch is a scheduling defect, not a measurement of zero."""
    services, model, heads = pieces
    quarantined = data_lib.ConstructSummary(
        construct=serial.Construct(heavy=HEAVY, light=LIGHT, sites=(("H", 3),),
                                   source="synthetic", record_id="q"),
        study="synthetic", assay=ASSAY, target="target", parent="synthetic::parent",
        family="enrichment", kind="enrichment",
        measurements=(data_lib.ReplicateMeasurement(
            value=None, lower=None, upper=None, censoring="none", replicate=1),),
        region=(("H", 3),), partition="train", scale="enrichment", source="synthetic")
    batch = head_batch(services, [quarantined])
    with pytest.raises(ValueError, match="zero eligible rows"):
        tasks_lib.head_outcome(services, "ablang2_masked_editor", model, heads, batch)


# --- the whole-window conditional preference ------------------------------

def whole_views(services, rows, *, bins):
    return data_lib.whole_window_views(rows, condition_spec=services.condition_spec,
                                       bins=bins, route="masked", max_positions=512)


def test_the_whole_window_preference_compares_two_conditions_on_one_row(pieces):
    services, model, heads = pieces
    rows = [summary(HEAVY, value=8.0, region=(("H", 3), ("H", 7)))]
    bins = {rows[0].population_key: BIN_HIGH}
    views = whole_views(services, rows, bins=bins)
    assert views[0][2].bin == BIN_HIGH
    rng = np.random.default_rng(3)
    outcome = tasks_lib.whole_window_outcome(services, "ablang2_masked_editor", model, views,
                                             rng=rng, preference_weight=1.0)
    assert outcome.term == "whole_window"
    assert "conditional_preference" in outcome.terms
    assert outcome.diagnostics["preference_rows"] == 1
    # One forward for the likelihood and one for the contrast condition, not two extra.
    assert outcome.forwards == len(views) + 1


def test_an_unbinned_whole_window_row_trains_the_likelihood_only(pieces):
    services, model, heads = pieces
    rows = [summary(HEAVY, value=8.0, region=(("H", 3), ("H", 7)))]
    views = whole_views(services, rows, bins={})
    outcome = tasks_lib.whole_window_outcome(services, "ablang2_masked_editor", model, views,
                                             rng=np.random.default_rng(1),
                                             preference_weight=1.0)
    assert "conditional_preference" not in outcome.terms
    assert outcome.diagnostics["preference_supported_rows"] == 0
    assert outcome.diagnostics["preference_trained"] is False


def test_the_contrast_condition_is_a_different_bin_including_null():
    condition = Condition(family="affinity", bin=BIN_HIGH, assay=ASSAY)
    rng = np.random.default_rng(11)
    seen = {tasks_lib.contrast_condition(condition, rng).bin for _ in range(64)}
    assert BIN_HIGH not in seen, "the contrast must not be the assigned bin"
    assert BIN_NULL in seen, "null must be reachable, or the term trains only 'conditioning helps'"
    assert tasks_lib.contrast_condition(
        Condition(family="affinity", bin=BIN_NULL, assay=ASSAY), rng) is None


# --- the arm dispatch -----------------------------------------------------

def test_a_control_arm_replaces_the_objective_and_keeps_the_ROWS(pieces):
    """Matched exposure is only achievable because the rows are identical."""
    services, model, heads = pieces
    rows = [summary(HEAVY, value=8.0, region=(("H", 3), ("H", 7)))]
    views = whole_views(services, rows, bins={rows[0].population_key: BIN_HIGH})
    draw = tasks_lib.TaskDraw(task=data_lib.TASK_WHOLE, family=data_lib.FAMILY_HER2,
                              views=views, summaries=tuple(rows), source="synthetic")
    settings = {"kappa": 2.0, "preference_weight": 1.0}
    rng = np.random.default_rng(5)
    full = tasks_lib.dispatch(services, "ablang2_masked_editor", model, heads, draw,
                              arm="full", settings=settings, rng=rng)
    stripped = tasks_lib.dispatch(services, "ablang2_masked_editor", model, heads, draw,
                                  arm="reconstruction_only", settings=settings, rng=rng)
    assert full.term == "whole_window" and stripped.term == "reconstruction"
    assert full.rows == stripped.rows, "a control arm must present the identical rows"
    assert stripped.scheduled_task == data_lib.TASK_WHOLE
    assert set(stripped.terms) == {"reconstruction"}


def test_head_only_keeps_the_head_objective_and_strips_the_decoder_ones(pieces):
    services, model, heads = pieces
    assert tasks_lib.ARM_OBJECTIVES["head_only"][data_lib.TASK_HEADS] == "own"
    assert tasks_lib.ARM_OBJECTIVES["head_only"][data_lib.TASK_LOCAL] == "reconstruction"
    assert tasks_lib.ARM_OBJECTIVES["local_only"][data_lib.TASK_WHOLE] == "reconstruction"
    assert tasks_lib.ARM_OBJECTIVES["local_only"][data_lib.TASK_LOCAL] == "own"


def test_an_insensitive_objective_is_reported_as_a_failure_not_a_pass():
    with pytest.raises(ValueError, match="did not move"):
        tasks_lib.require_sensitive(lambda: 1.0, perturb=lambda: 1.0, label="constant")


def test_gradient_fractions_report_each_task_share_of_the_gradient(pieces):
    services, model, heads = pieces
    rows = [summary(HEAVY, value=8.0, region=(("H", 3), ("H", 7)))]
    views = whole_views(services, rows, bins={})
    whole = tasks_lib.whole_window_outcome(services, "ablang2_masked_editor", model, views,
                                           rng=np.random.default_rng(2),
                                           preference_weight=0.0)
    local = tasks_lib.local_outcome(services, "ablang2_masked_editor", model,
                                    [_contrast(0.5)], kappa=2.0,
                                    reconstruction_weight=0.25)
    block = tasks_lib.gradient_fractions(
        {data_lib.TASK_WHOLE: whole.loss, data_lib.TASK_LOCAL: local.loss},
        [p for p in model.parameters() if p.requires_grad])
    assert set(block["fractions"]) == {data_lib.TASK_WHOLE, data_lib.TASK_LOCAL}
    assert sum(value for value in block["fractions"].values()) == pytest.approx(1.0)
    assert "not teaching" in block["reading"]
