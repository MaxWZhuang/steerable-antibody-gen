"""The replay term, its reductions and one combined update.

Every test here is about a failure that produces a *number* rather than an error:
a zero teacher probability that becomes a NaN gradient, an accumulation that is
right for full microbatches and wrong for a short tail, a lambda-zero control
that still moved the replay stream, or a sequence-drop tail quietly reported as a
conditional-KL tail. None of them raise on their own.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch

from smallAntibodyGen.experiments import her2_replay as replay


# ---------------------------------------------------------------------------
# an enumerable autoregressive distribution
# ---------------------------------------------------------------------------

class TinyAutoregressive:
    """Two positions over three residues, small enough to enumerate exactly.

    The whole support is nine sequences, so the sequence KL, the chain-rule
    conditional KL and the sampled estimator can all be computed exactly and
    compared with no Monte Carlo error in the way.
    """

    def __init__(self, seed):
        generator = np.random.default_rng(seed)
        self.first = self._normalize(generator.normal(size=3))
        self.second = np.stack([self._normalize(generator.normal(size=3)) for _ in range(3)])

    @staticmethod
    def _normalize(logits):
        exponent = np.exp(logits - logits.max())
        return exponent / exponent.sum()

    def sequence_probability(self, y):
        return float(self.first[y[0]] * self.second[y[0], y[1]])

    def conditional(self, y):
        """``(2, 3)`` conditional distributions at the prefixes this sequence visits."""
        return np.stack([self.first, self.second[y[0]]])


def enumerate_sequences():
    return list(itertools.product(range(3), repeat=2))


def test_conditional_kl_matches_a_direct_triple_sum():
    """The vectorized formula against a deliberately naive reference, in float64."""
    generator = torch.Generator().manual_seed(20260920)
    teacher = torch.log_softmax(torch.randn(5, 10, 20, generator=generator,
                                            dtype=torch.float64), dim=-1)
    student = torch.log_softmax(torch.randn(5, 10, 20, generator=generator,
                                            dtype=torch.float64), dim=-1)
    probabilities = teacher.exp()
    fast = replay.sequence_conditional_kl(probabilities, teacher, student)
    slow = torch.zeros(5, dtype=torch.float64)
    for row in range(5):
        for position in range(10):
            for residue in range(20):
                slow[row] += (probabilities[row, position, residue]
                              * (teacher[row, position, residue]
                                 - student[row, position, residue]))
    assert torch.allclose(fast, slow, atol=1e-12, rtol=0)


def test_a_zero_teacher_probability_contributes_exactly_zero_with_no_nan():
    """``0 * -inf`` is the NaN this masking exists to prevent, forward and backward."""
    probabilities = torch.zeros(1, 10, 20, dtype=torch.float64)
    probabilities[0, :, 0] = 1.0
    teacher = torch.full((1, 10, 20), float("-inf"), dtype=torch.float64)
    teacher[0, :, 0] = 0.0
    student = torch.log_softmax(torch.zeros(1, 10, 20, dtype=torch.float64), dim=-1)
    student.requires_grad_(True)
    value = replay.sequence_conditional_kl(probabilities, teacher, student)
    assert torch.isfinite(value).all(), "a masked -inf must not reach the forward value"
    value.sum().backward()
    assert torch.isfinite(student.grad).all(), "and it must not reach the backward path either"
    assert float(student.grad[0, 0, 1]) == 0.0, "a zero-probability residue gets zero gradient"


def test_student_equal_to_teacher_gives_zero_loss_and_zero_gradient():
    generator = torch.Generator().manual_seed(7)
    logits = torch.randn(4, 10, 20, generator=generator, dtype=torch.float64,
                         requires_grad=True)
    student = torch.log_softmax(logits, dim=-1)
    teacher = student.detach()
    mean, diagnostics = replay.replay_term(teacher.exp(), teacher, student)
    assert abs(float(mean)) < 1e-12
    mean.backward()
    assert float(logits.grad.abs().max()) < 1e-12
    assert diagnostics["rows"] == 4


def test_sequence_kl_equals_the_chain_rule_conditional_kl_in_expectation():
    """``E_parent[K(y)] = KL(parent || policy)`` exactly, enumerated rather than sampled."""
    parent, policy = TinyAutoregressive(1), TinyAutoregressive(2)
    sequence_kl, expected_conditional = 0.0, 0.0
    for y in enumerate_sequences():
        p = parent.sequence_probability(y)
        q = policy.sequence_probability(y)
        sequence_kl += p * (np.log(p) - np.log(q))
        teacher = torch.tensor(parent.conditional(y), dtype=torch.float64).unsqueeze(0)
        student = torch.tensor(policy.conditional(y), dtype=torch.float64).unsqueeze(0)
        block = replay.sequence_conditional_kl(teacher, teacher.log(), student.log())
        expected_conditional += p * float(block)
    assert expected_conditional == pytest.approx(sequence_kl, abs=1e-12)


def test_the_drop_and_the_conditional_kl_share_a_mean_and_not_a_distribution():
    """Their tails are different random variables and are never reported as one."""
    parent, policy = TinyAutoregressive(3), TinyAutoregressive(4)
    drops, conditionals, weights = [], [], []
    for y in enumerate_sequences():
        p = parent.sequence_probability(y)
        q = policy.sequence_probability(y)
        teacher = torch.tensor(parent.conditional(y), dtype=torch.float64).unsqueeze(0)
        student = torch.tensor(policy.conditional(y), dtype=torch.float64).unsqueeze(0)
        drops.append(np.log(p) - np.log(q))
        conditionals.append(float(replay.sequence_conditional_kl(teacher, teacher.log(),
                                                                 student.log())))
        weights.append(p)
    drops, conditionals, weights = np.array(drops), np.array(conditionals), np.array(weights)
    assert float(drops @ weights) == pytest.approx(float(conditionals @ weights), abs=1e-12)
    assert not np.allclose(np.sort(drops), np.sort(conditionals)), (
        "equal means do not make the two vectors interchangeable, and a tail taken from one is "
        "not a tail of the other")


# ---------------------------------------------------------------------------
# accumulation: one gradient, one clip, one step
# ---------------------------------------------------------------------------

class TinyModule(torch.nn.Module):
    """A two-parameter stand-in with a shared 'prefix' term every row depends on."""

    def __init__(self, seed=0):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.prefix = torch.nn.Parameter(torch.randn(1, generator=generator, dtype=torch.float64))
        self.table = torch.nn.Parameter(torch.randn(10, 20, generator=generator,
                                                     dtype=torch.float64))

    def logits(self, index):
        rows = torch.as_tensor(np.asarray(index), dtype=torch.long)
        shift = torch.nn.functional.one_hot(rows, num_classes=20).to(torch.float64)
        return self.table.unsqueeze(0) + self.prefix * shift


def gradients_of(module):
    return torch.cat([parameter.grad.detach().reshape(-1)
                      for _, parameter in sorted(module.named_parameters())])


@pytest.mark.parametrize("partition", [[16, 16, 16, 16], [24, 24, 16], [64]])
def test_row_weighted_accumulation_reproduces_the_full_batch_gradient(partition):
    """Including the partition whose last microbatch is short, which is the point."""
    rows = sum(partition)
    generator = np.random.default_rng(11)
    cores = generator.integers(0, 20, size=(rows, 10))
    teacher_logits = torch.tensor(generator.normal(size=(rows, 10, 20)), dtype=torch.float64)
    teacher_logs = torch.log_softmax(teacher_logits, dim=-1)
    probabilities = teacher_logs.exp()

    def components(module, block):
        student = torch.log_softmax(module.logits(cores[block]), dim=-1)
        task = -student.gather(2, torch.as_tensor(cores[block], dtype=torch.long)
                               .unsqueeze(-1)).squeeze(-1).sum(dim=1).mean() / 10
        replay_mean, _ = replay.replay_term(probabilities[block], teacher_logs[block], student)
        return task, replay_mean

    direct = TinyModule(seed=5)
    direct.zero_grad(set_to_none=True)
    task, replay_mean = components(direct, slice(0, rows))
    replay.combined_loss(task, replay_mean, replay_coefficient=3.0).backward()

    accumulated = TinyModule(seed=5)
    accumulated.zero_grad(set_to_none=True)
    accumulator = replay.MicrobatchAccumulator({"task": rows, "replay": rows},
                                               coefficients={"task": 1.0, "replay": 3.0})
    start = 0
    for size in partition:
        block = slice(start, start + size)
        task_block, replay_block = components(accumulated, block)
        accumulator.add({"task": size, "replay": size},
                        {"task": task_block, "replay": replay_block},
                        backward=lambda tensor: tensor.backward())
        start += size
    summary = accumulator.finish()
    assert torch.allclose(gradients_of(accumulated), gradients_of(direct), atol=1e-12, rtol=0)
    assert summary["rows"] == {"task": rows, "replay": rows}
    assert summary["microbatches"] == len(partition)


def test_the_equal_split_shortcut_is_wrong_exactly_where_the_tail_is_partial():
    """``sum(mean_i) / k`` and the row-weighted form differ only on a short final chunk."""
    values = [torch.tensor(1.0), torch.tensor(4.0)]
    counts = [24, 8]
    accumulator = replay.MicrobatchAccumulator({"task": 32})
    for count, value in zip(counts, values):
        accumulator.add({"task": count}, {"task": value})
    weighted = accumulator.finish()["component_means"]["task"]
    equal_split = float(sum(values) / len(values))
    assert weighted == pytest.approx((24 * 1.0 + 8 * 4.0) / 32)
    assert weighted != pytest.approx(equal_split), (
        "the two reductions agree only when every microbatch is full, which is the case a "
        "partial-tail test does not exercise")


def test_the_accumulator_refuses_an_update_that_consumed_the_wrong_rows():
    accumulator = replay.MicrobatchAccumulator({"task": 64})
    accumulator.add({"task": 32}, {"task": torch.tensor(1.0)})
    with pytest.raises(ValueError, match="did not consume the rows they declared"):
        accumulator.finish()
    with pytest.raises(ValueError, match="more than the 64 declared"):
        accumulator.add({"task": 64}, {"task": torch.tensor(1.0)})


def test_the_accumulator_refuses_an_undeclared_component():
    accumulator = replay.MicrobatchAccumulator({"task": 8})
    with pytest.raises(ValueError, match="was never declared"):
        accumulator.add({"replay": 8}, {"replay": torch.tensor(1.0)})


# ---------------------------------------------------------------------------
# lambda zero is a no-op, structurally
# ---------------------------------------------------------------------------

def test_lambda_zero_returns_the_task_term_itself():
    task = torch.tensor(2.5, requires_grad=True)
    assert replay.combined_loss(task, None, replay_coefficient=0.0) is task


def test_lambda_zero_refuses_a_replay_term_that_was_computed_anyway():
    """A control that evaluated the replay term did work its declared arm does not do."""
    with pytest.raises(ValueError, match="no replay work"):
        replay.combined_loss(torch.tensor(1.0), torch.tensor(1.0), replay_coefficient=0.0)


def test_a_positive_lambda_requires_a_replay_term():
    with pytest.raises(ValueError, match="no replay term was supplied"):
        replay.combined_loss(torch.tensor(1.0), None, replay_coefficient=0.1)


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_lambda_must_be_finite_and_nonnegative(value):
    with pytest.raises(ValueError, match="finite and >= 0"):
        replay.combined_loss(torch.tensor(1.0), torch.tensor(1.0), replay_coefficient=value)


# ---------------------------------------------------------------------------
# the task seam
# ---------------------------------------------------------------------------

def test_continued_sft_refuses_a_rejected_or_reference_tensor():
    chosen = torch.tensor([-12.0, -14.0], requires_grad=True)
    with pytest.raises(ValueError, match="chosen sequences only"):
        replay.task_term("continued_sft", policy_chosen=chosen, policy_rejected=chosen.detach())


def test_ipo_needs_both_frozen_reference_vectors():
    chosen = torch.tensor([-12.0, -14.0], requires_grad=True)
    with pytest.raises(ValueError, match="needs the policy rejected"):
        replay.task_term("ipo", policy_chosen=chosen, policy_rejected=chosen)


def test_ipo_refuses_a_live_reference():
    chosen = torch.tensor([-12.0, -14.0], requires_grad=True)
    live = torch.tensor([-12.0, -14.0], requires_grad=True)
    with pytest.raises(ValueError, match="requires_grad"):
        replay.task_term("ipo", policy_chosen=chosen, policy_rejected=chosen,
                         reference_chosen=live, reference_rejected=live)


def test_ipo_is_the_inherited_squared_target_margin():
    chosen = torch.tensor([-10.0, -11.0])
    rejected = torch.tensor([-12.0, -12.0])
    reference_chosen = torch.tensor([-10.5, -11.5])
    reference_rejected = torch.tensor([-12.5, -12.5])
    mean, _ = replay.task_term("ipo", policy_chosen=chosen, policy_rejected=rejected,
                               reference_chosen=reference_chosen,
                               reference_rejected=reference_rejected, tau=0.1)
    delta = (chosen - rejected) - (reference_chosen - reference_rejected)
    assert float(mean) == pytest.approx(float(((delta - 5.0) ** 2).mean()))


def test_an_unknown_task_is_refused():
    with pytest.raises(ValueError, match="Unknown replay task"):
        replay.task_term("dpo", policy_chosen=torch.tensor([-1.0]))


# ---------------------------------------------------------------------------
# teacher target validation
# ---------------------------------------------------------------------------

def valid_targets(rows=3):
    logits = torch.randn(rows, 10, 20, generator=torch.Generator().manual_seed(1))
    logs = torch.log_softmax(logits, dim=-1)
    return logs.exp(), logs


def test_teacher_targets_accept_a_well_formed_block():
    probabilities, logs = valid_targets()
    block = replay.require_teacher_targets(probabilities, logs, rows=3, label="probe")
    assert block["max_abs_probability_sum_error"] < replay.PROBABILITY_SUM_ATOL
    assert block["residues"] == 20 and block["positions"] == 10


def test_a_transposed_teacher_block_is_refused():
    probabilities, logs = valid_targets()
    with pytest.raises(ValueError, match="expected"):
        replay.require_teacher_targets(probabilities.transpose(1, 2), logs.transpose(1, 2),
                                       rows=3, label="probe")


def test_an_unnormalized_teacher_block_is_refused():
    probabilities, logs = valid_targets()
    broken = probabilities.clone()
    broken[0, 0, :] *= 0.5
    with pytest.raises(ValueError, match="summing to one"):
        replay.require_teacher_targets(broken, logs, rows=3, label="probe")


def test_probabilities_and_logs_must_be_two_views_of_one_computation():
    probabilities, logs = valid_targets()
    other = torch.log_softmax(torch.randn(3, 10, 20,
                                          generator=torch.Generator().manual_seed(2)), dim=-1)
    with pytest.raises(ValueError, match="disagree"):
        replay.require_teacher_targets(probabilities, other, rows=3, label="probe")


def test_a_live_teacher_is_refused():
    probabilities, logs = valid_targets()
    with pytest.raises(ValueError, match="requires_grad"):
        replay.require_teacher_targets(probabilities.requires_grad_(True), logs, rows=3,
                                       label="probe")


def test_a_nonfinite_teacher_probability_is_refused():
    probabilities, logs = valid_targets()
    broken = probabilities.clone()
    broken[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        replay.require_teacher_targets(broken, logs, rows=3, label="probe")


# ---------------------------------------------------------------------------
# comparisons and the step
# ---------------------------------------------------------------------------

def test_compare_vectors_reports_the_measured_error_before_it_judges():
    report = replay.compare_vectors([1.0, 2.0], [1.0, 2.0 + 1e-9], atol=1e-6, rtol=0,
                                    label="probe")
    assert report["within_tolerance"] and report["max_abs_error"] == pytest.approx(1e-9)


def test_compare_vectors_refuses_a_real_difference_and_names_the_index():
    with pytest.raises(ValueError, match="index 1"):
        replay.compare_vectors([1.0, 2.0], [1.0, 3.0], atol=1e-6, rtol=0, label="probe")


def test_compare_vectors_refuses_a_nonfinite_value():
    with pytest.raises(ValueError, match="nonfinite"):
        replay.compare_vectors([1.0, float("nan")], [1.0, 1.0], atol=1.0, rtol=0, label="probe")


def test_clip_and_step_records_whether_the_clip_bound():
    module = TinyModule(seed=3)
    module.zero_grad(set_to_none=True)
    (module.table.sum() * 1000).backward()
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    record = replay.clip_and_step(module, optimizer, scheduler, gradient_clip=1.0)
    assert record["clipped"] is True and record["gradient_norm"] > 1.0
    assert record["learning_rate_used"] == pytest.approx(1e-5)


def test_the_logged_learning_rate_is_the_one_the_step_applied_not_the_next_one():
    """The exact production schedule: step+1 warmup over 100 updates at lr 1e-5.

    The first optimizer step happens at scale 1/100, so it applies 1e-7. Reading
    ``get_last_lr`` after ``scheduler.step()`` reports 2e-7 -- the rate the *second*
    update will use -- and every journalled rate would be one position ahead of the
    step it claims to describe.
    """
    from smallAntibodyGen.experiments import her2_policy as policy_lib
    module = TinyModule(seed=11)
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: policy_lib.continuation_learning_rate_scale(
            step + 1, warmup_steps=100))
    applied = []
    for _ in range(3):
        module.zero_grad(set_to_none=True)
        module.table.sum().backward()
        applied.append(replay.clip_and_step(module, optimizer, scheduler, gradient_clip=1.0))
    assert applied[0]["learning_rate_used"] == pytest.approx(1e-7)
    assert applied[1]["learning_rate_used"] == pytest.approx(2e-7)
    assert applied[0]["learning_rate_next"] == pytest.approx(
        applied[1]["learning_rate_used"]), "the next rate is recorded, and it is the next one"
    assert applied[2]["learning_rate_used"] == pytest.approx(3e-7)


# ---------------------------------------------------------------------------
# the independent first-step AdamW oracle and the reconciliation
# ---------------------------------------------------------------------------

def test_the_adamw_oracle_reproduces_a_real_first_step():
    """Written from the update rule, checked against the optimizer it describes."""
    module = TinyModule(seed=12)
    module.zero_grad(set_to_none=True)
    weights = torch.linspace(-1.0, 1.0, 20, dtype=torch.float64)
    ((module.table * weights).sum() * module.prefix.sum()).backward()
    before = replay.state_vector(module)
    gradients = np.concatenate([p.grad.detach().double().reshape(-1).numpy()
                                for _, p in sorted(module.named_parameters())])
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-5, betas=(0.9, 0.999),
                                  weight_decay=0.01)
    optimizer.step()
    expected = replay.adamw_first_step(before, gradients, learning_rate=1e-5, weight_decay=0.01)
    report = replay.compare_vectors(replay.state_vector(module), expected,
                                    atol=1e-9, rtol=1e-6, label="oracle")
    assert report["within_tolerance"] and report["max_abs_error"] < 1e-9


def accept_two_routes(*, before, grad_a, grad_b, label):
    """The native control's acceptance, in the order the control applies it.

    The gradient comparison comes first and is the load-bearing one; only then are
    the post-step parameters reconciled. ``synthetic_gradient_control`` builds its
    per-microbatch block with ``gradients`` before ``cross_route_parameters``, so a
    route whose gradients disagree never reaches the reconciliation -- and testing
    the reconciliation alone therefore does not describe what production accepts.
    """
    replay.compare_vectors(grad_a, grad_b, atol=1e-4, rtol=1e-3,
                           label=f"{label}: accumulated versus direct gradients")
    return replay.reconcile_post_step(
        actual=replay.adamw_first_step(before, grad_a, learning_rate=1e-5, weight_decay=0.01),
        expected=replay.adamw_first_step(before, grad_b, learning_rate=1e-5, weight_decay=0.01),
        gradient_actual=grad_a, gradient_expected=grad_b,
        atol=3e-6, rtol=1e-4, learning_rate=1e-5, weight_decay=0.01,
        parameters_before=before, label=label)


def test_a_sign_flipped_near_zero_gradient_is_reconciled_and_a_real_difference_is_not():
    """The amended criterion, on the exact failure the reviewer measured.

    Two routes whose gradients differ only in the last bits of a ~1e-7 coordinate
    produce AdamW steps that differ by the full 2 * lr, because the first step
    saturates to ``-lr * sign(g)``. That is admitted -- and *explained* -- while a
    coordinate with a decided gradient that moved differently is not.
    """
    before = np.zeros(3)
    grad_a = np.asarray([1e-7, 0.5, -0.5])
    grad_b = np.asarray([-1e-7, 0.5, -0.5])         # one sign flip at a near-zero coordinate
    left = replay.adamw_first_step(before, grad_a, learning_rate=1e-5, weight_decay=0.01)
    right = replay.adamw_first_step(before, grad_b, learning_rate=1e-5, weight_decay=0.01)
    assert abs(left[0] - right[0]) > 1e-5, "the sign flip really does move the step by ~2 * lr"
    report = accept_two_routes(before=before, grad_a=grad_a, grad_b=grad_b, label="near zero")
    assert report["within_tolerance"] is False and report["reconciled"] is True
    assert report["coordinates_outside_tolerance"] == 1
    assert report["coordinates_explained_by_near_zero_gradients"] == 1
    assert "gradients agree" in report["prerequisite"]

    # A gradient that is not near zero and differs between the routes is rejected
    # by the FIRST half of the criterion. Reconciliation alone would admit it --
    # min(|0.5|, |-0.5|) <= |0.5 - -0.5| is true -- which is why the prerequisite is
    # part of the contract and is exercised here rather than assumed.
    grad_c = np.asarray([1e-7, -0.5, -0.5])
    with pytest.raises(ValueError, match="accumulated versus direct gradients"):
        accept_two_routes(before=before, grad_a=grad_a, grad_b=grad_c, label="decided gradient")


def test_a_parameter_that_drifted_under_identical_gradients_is_a_different_update():
    """The case the reconciliation is actually the last line of defence for.

    The gradients are identical and well determined, so the prerequisite passes and
    says nothing; one coordinate nevertheless moved differently. Nothing explains
    that, and it is reported as a different update rather than absorbed.
    """
    before = np.zeros(3)
    gradients = np.asarray([1e-7, 0.5, -0.5])
    stepped = replay.adamw_first_step(before, gradients, learning_rate=1e-5, weight_decay=0.01)
    drifted = stepped.copy()
    drifted[1] += 8e-6                      # inside one step's ceiling, outside the tolerance
    replay.compare_vectors(gradients, gradients, atol=1e-4, rtol=1e-3, label="identical gradients")
    with pytest.raises(ValueError, match="different update"):
        replay.reconcile_post_step(
            actual=drifted, expected=stepped,
            gradient_actual=gradients, gradient_expected=gradients,
            atol=3e-6, rtol=1e-4, learning_rate=1e-5, weight_decay=0.01,
            parameters_before=before, label="unexplained drift")


def test_a_nonfinite_coordinate_is_refused_rather_than_reported_as_reconciled():
    """Every test below is a ``>`` against a tolerance, and a NaN loses all of them."""
    before = np.zeros(2)
    gradients = np.asarray([0.5, -0.5])
    stepped = replay.adamw_first_step(before, gradients, learning_rate=1e-5, weight_decay=0.01)
    broken = stepped.copy()
    broken[0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite value entered"):
        replay.reconcile_post_step(
            actual=broken, expected=stepped, gradient_actual=gradients,
            gradient_expected=gradients, atol=3e-6, rtol=1e-4, learning_rate=1e-5,
            weight_decay=0.01, parameters_before=before, label="nonfinite")


def test_the_reconciliation_does_not_admit_a_movement_beyond_one_step():
    """An explained sign flip still cannot excuse a parameter that moved further."""
    before = np.zeros(2)
    grad_a, grad_b = np.asarray([1e-9, 1.0]), np.asarray([-1e-9, 1.0])
    left = replay.adamw_first_step(before, grad_a, learning_rate=1e-5, weight_decay=0.01)
    right = replay.adamw_first_step(before, grad_b, learning_rate=1e-5, weight_decay=0.01)
    left = left.copy()
    left[0] += 1.0                                   # far beyond any single step
    with pytest.raises(ValueError, match="different update"):
        replay.reconcile_post_step(
            actual=left, expected=right, gradient_actual=grad_a, gradient_expected=grad_b,
            atol=3e-6, rtol=1e-4, learning_rate=1e-5, weight_decay=0.01,
            parameters_before=before, label="beyond one step")


def test_a_nonfinite_gradient_stops_the_update_rather_than_stepping_on_it():
    module = TinyModule(seed=3)
    module.zero_grad(set_to_none=True)
    module.table.grad = torch.full_like(module.table, float("nan"))
    module.prefix.grad = torch.zeros_like(module.prefix)
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    with pytest.raises(RuntimeError):
        replay.clip_and_step(module, optimizer, scheduler, gradient_clip=1.0)


def test_teacher_is_frozen_separates_requires_grad_from_carrying_gradients():
    module = TinyModule(seed=4)
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    ok, report = replay.teacher_is_frozen(module)
    assert ok and report["requires_grad"] == [] and report["carrying_gradients"] == []
    module.table.grad = torch.zeros_like(module.table)
    ok, report = replay.teacher_is_frozen(module)
    assert not ok and report["carrying_gradients"] == ["table"]


def test_require_core_block_refuses_fractional_and_out_of_range_indices():
    with pytest.raises(ValueError, match="integral"):
        replay.require_core_block(np.full((2, 10), 1.5), label="probe")
    with pytest.raises(ValueError, match="outside"):
        replay.require_core_block(np.full((2, 10), 20), label="probe")
    with pytest.raises(ValueError, match="expected"):
        replay.require_core_block(np.zeros((2, 9), dtype=np.int8), label="probe")
